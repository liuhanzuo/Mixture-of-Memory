"""Dolmino Curriculum Dataset for Memory-Space Continued Pretraining.

Loads pre-tokenised Dolmino chunks (1024 tokens each) from a HuggingFace Arrow
dataset and groups consecutive shuffled samples into (N_ctx context chunks +
1 target chunk) for curriculum-based continued pretraining.

The curriculum works by gradually increasing N_ctx (the number of context chunks
the model must compress into its memory bank before predicting the target chunk).
N_ctx is controlled externally via ``set_n_context(n)`` — typically called by the
curriculum scheduler in the training script.

Data format
-----------
The Arrow dataset at ``dolmino_0.5B_1024/train`` has 463K rows, each containing
a single column ``input_ids`` which is a list of 1024 int32 token IDs (Llama-3
tokenizer).

CRITICAL — the Arrow rows are STREAM-ORDERED (NOT shuffled). MemLong's
``process_dolmino.py`` reads a flat token stream with ``np.fromfile`` then
``reshape(n_chunks, 1024)`` and concatenates shards in order — no shuffle
anywhere. So Arrow row ``i`` and row ``i+1`` are adjacent 1024-token blocks of
the SAME continuous token stream (``doc1<EOS>doc2<EOS>...`` packed end-to-end).

Two modes
---------
* ``contiguous=False`` (default, legacy): every epoch ``rng.shuffle`` the row
  indices and group arbitrary rows into (context, target). Because context and
  target are then UNRELATED random documents, "remembering the context" does not
  help predict the target — the correct solution under this data is uniform
  routing (memory collapse). Kept for backward compatibility.

* ``contiguous=True``: treat the Arrow data as one continuous token stream and
  re-slice it at a SMALLER ``chunk_size`` (e.g. 256). Consecutive 256-chunks
  then fall inside the same ~1k-token document => genuine intra-document
  cross-chunk dependency, which is what the memory bank needs in order to learn
  content addressing instead of collapsing to uniform.

* ``per_doc=True`` (recommended): each Arrow row is ONE COMPLETE document
  (produced by ``scripts/reprocess_dolmino_per_doc.py`` which re-slices the
  packed stream on EOS boundaries). Each document is cut into consecutive
  non-overlapping windows of ``(n_ctx+1)`` chunks (n_ctx context + 1 target,
  all from the same document). This gives exact document boundaries (cleaner
  than ``contiguous``, which only approximates them via EOS look-back).

DDP sharding
------------
* ``contiguous=False``: each consumer strides the shuffled index array.
* ``contiguous=True``: the continuous token stream is split into
  ``world_size * num_workers`` disjoint CONTIGUOUS segments; each consumer walks
  its own segment front-to-back taking groups of ``(n_ctx+1)*chunk_size``
  consecutive tokens. Per epoch only a small random START JITTER inside the
  segment is randomised (for novel groupings) — token order WITHIN a group is
  never shuffled.
"""
from __future__ import annotations

import math
import os
import random
from typing import Dict, Iterator, List, Optional

import numpy as np
import torch
import torch.utils.data


# Llama-3 special tokens (used for contiguous doc-isolation).
EOS_TOKEN_ID = 128001
BOS_TOKEN_ID = 128000


class DolminoCurriculumDataset(torch.utils.data.IterableDataset):
    """Iterable dataset that groups Dolmino chunks into (context, target) pairs.

    Each yielded sample is a dict with:
        ``context_chunks``: list of N_ctx tensors, each [chunk_size] (long)
        ``target_ids``:     tensor [chunk_size] (long)
        ``is_dolmino``:     True
        ``reset_flags``:    (only when ``doc_reset=True``) list of (n_ctx+1) bools,
                            one per chunk in stream order [ctx_0, ..., ctx_{n-1},
                            target]. ``reset_flags[k] == True`` means chunk k
                            begins a NEW document (the token immediately before
                            its first token in the continuous stream is an EOS),
                            so the training loop should reset the memory bank
                            BEFORE forwarding chunk k.

    Args:
        data_path: Path to the HuggingFace Arrow dataset directory
                   (e.g. ``MemLong/data/processed/dolmino_0.5B_1024/train``).
        chunk_size: Token count per chunk. In legacy mode this slices each Arrow
                    row; in contiguous mode this is the re-slice granularity
                    (e.g. 256) of the continuous stream.
        n_context: Initial number of context chunks per group (default 1).
        rank: DDP rank (for sharding).
        world_size: DDP world size (for sharding).
        seed: Base RNG seed.
        contiguous: If True, enable the continuous stream-reslice mode (see
                    module docstring). Default False (legacy shuffle behaviour).
        doc_reset: If True (only meaningful with ``contiguous=True``), attach a
                   ``reset_flags`` list to each sample marking per-chunk
                   document boundaries (see ``reset_flags`` above).
        per_doc: If True, treat each Arrow row as ONE COMPLETE (variable-length)
                 document (produced by ``scripts/reprocess_dolmino_per_doc.py``)
                 and slice it into intra-document chunk groups. For each document
                 we take consecutive non-overlapping windows of (n_ctx+1) chunks:
                 the first ``n_ctx`` chunks are ``context_chunks`` and the next
                 chunk is ``target_ids`` — all from the SAME document, so context
                 and target have genuine cross-chunk dependency. Documents shorter
                 than ``(n_ctx+1)*chunk_size`` are skipped. Documents are sharded
                 (and shuffled per epoch) across DDP ranks/workers; chunk order
                 WITHIN a document is never shuffled. This is the recommended mode
                 (cleaner than ``contiguous`` because doc boundaries are exact).
                 ``per_doc`` takes precedence over ``contiguous`` when both set.
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 1024,
        n_context: int = 1,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        contiguous: bool = False,
        doc_reset: bool = False,
        per_doc: bool = False,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.chunk_size = chunk_size
        self._n_context = n_context
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.contiguous = contiguous
        self.doc_reset = doc_reset
        self.per_doc = per_doc

        # Load dataset eagerly (memory-mapped Arrow is fast)
        import datasets
        self._ds = datasets.load_from_disk(data_path)
        self._num_samples = len(self._ds)

        # Detect Arrow row length from row 0 (uniform 1024 in this dataset).
        # Used by contiguous mode to map global token offsets -> (row, offset).
        self._arrow_row_len = len(self._ds[0]["input_ids"]) if self._num_samples else 0

        # We'll create index permutations per epoch for novel groupings
        self._epoch = 0

        # Single-row cache for the contiguous stream reader (reads advance
        # forward, so caching the most-recently-read row is enough).
        self._cache_row_idx: int = -1
        self._cache_row: Optional[List[int]] = None

    @property
    def n_context(self) -> int:
        return self._n_context

    def set_n_context(self, n: int) -> None:
        """Update the number of context chunks (called by curriculum scheduler)."""
        self._n_context = max(1, int(n))

    # ------------------------------------------------------------------ #
    # Contiguous-stream helpers
    # ------------------------------------------------------------------ #
    def _get_row(self, row_idx: int) -> List[int]:
        """Return the token list for Arrow ``row_idx``, caching the last one."""
        if row_idx == self._cache_row_idx and self._cache_row is not None:
            return self._cache_row
        row = self._ds[int(row_idx)]["input_ids"]
        self._cache_row_idx = int(row_idx)
        self._cache_row = row
        return row

    def _read_stream(self, start: int, length: int) -> List[int]:
        """Read ``length`` consecutive tokens of the global stream at ``start``.

        The global stream is the row-order concatenation of all Arrow rows
        (row i occupies tokens [i*L, (i+1)*L) for L = arrow_row_len). Reads are
        lazy: only the touched rows are fetched, sliced and concatenated.
        """
        rl = self._arrow_row_len
        out: List[int] = []
        pos = start
        end = start + length
        while pos < end:
            row_idx = pos // rl
            if row_idx >= self._num_samples:
                break  # ran off the end of the stream
            off = pos % rl
            row = self._get_row(row_idx)
            take = min(rl - off, end - pos, len(row) - off)
            if take <= 0:
                break
            out.extend(row[off: off + take])
            pos += take
        return out

    def _is_eos_before(self, pos: int) -> bool:
        """True if the token immediately before global position ``pos`` is EOS.

        Used for doc_reset: a chunk whose first token is at ``pos`` begins a new
        document iff the preceding token is an EOS (128001). Position 0 is never
        a "reset" (there is no preceding token).
        """
        if pos <= 0:
            return False
        prev = self._read_stream(pos - 1, 1)
        return bool(prev) and int(prev[0]) == EOS_TOKEN_ID

    # ------------------------------------------------------------------ #
    # Iteration
    # ------------------------------------------------------------------ #
    def __iter__(self) -> Iterator[Dict[str, object]]:
        if self.per_doc:
            return self._iter_per_doc()
        if self.contiguous:
            return self._iter_contiguous()
        return self._iter_shuffled()

    def _iter_shuffled(self) -> Iterator[Dict[str, object]]:
        """Legacy mode: shuffle Arrow rows and group arbitrary rows.

        Unchanged from the original implementation — preserved for backward
        compatibility (contiguous=False).
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Unique seed per (epoch, rank, worker)
        epoch = self._epoch

        while True:  # infinite iteration across epochs
            rng = random.Random(self.seed + epoch * 10007 + self.rank * 1000 + worker_id)

            # Create a shuffled index permutation for this epoch
            indices = list(range(self._num_samples))
            rng.shuffle(indices)

            # Shard across (world_size * num_workers) total consumers
            total_consumers = self.world_size * num_workers
            consumer_id = self.rank * num_workers + worker_id
            # Each consumer gets every total_consumers-th group start
            # We stride the index array by consumer_id
            my_indices = indices[consumer_id::total_consumers]

            # Use a pointer to advance through my_indices dynamically.
            # This allows n_context to change mid-epoch (curriculum updates)
            # without pre-computing group boundaries.
            ptr = 0
            while ptr + self._n_context < len(my_indices):
                # Read current n_context at each sample (may change between yields)
                n_ctx = self._n_context
                group_size = n_ctx + 1

                if ptr + group_size > len(my_indices):
                    break  # not enough samples left for a full group

                group_indices = my_indices[ptr: ptr + group_size]
                ptr += group_size

                # Read n_ctx context chunks + 1 target chunk
                context_chunks: List[torch.Tensor] = []
                for i in range(n_ctx):
                    row = self._ds[group_indices[i]]
                    tokens = row["input_ids"]
                    context_chunks.append(
                        torch.tensor(tokens[: self.chunk_size], dtype=torch.long)
                    )

                # Target is the last in the group
                target_row = self._ds[group_indices[-1]]
                target_ids = torch.tensor(
                    target_row["input_ids"][: self.chunk_size], dtype=torch.long
                )

                yield {
                    "context_chunks": context_chunks,
                    "target_ids": target_ids,
                    "is_dolmino": True,
                }

            # Advance epoch for next loop iteration (novel groupings)
            epoch += 1
            self._epoch = epoch

    def _iter_contiguous(self) -> Iterator[Dict[str, object]]:
        """Contiguous mode: re-slice the continuous token stream at chunk_size.

        The global token stream (row-order concat of all Arrow rows) is split
        into ``world_size * num_workers`` disjoint contiguous segments. Each
        consumer walks its own segment front-to-back, emitting groups of
        ``(n_ctx+1) * chunk_size`` CONSECUTIVE tokens (n_ctx context chunks +
        1 target chunk, all adjacent in the stream). Per epoch only a small
        random start jitter inside the segment is randomised; token order WITHIN
        a group is never shuffled.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        total_tokens = self._num_samples * self._arrow_row_len
        total_consumers = self.world_size * num_workers
        consumer_id = self.rank * num_workers + worker_id

        seg_len = total_tokens // total_consumers
        seg_start = consumer_id * seg_len
        seg_end = seg_start + seg_len  # disjoint half-open segment for this consumer

        epoch = self._epoch
        while True:  # infinite iteration across epochs
            rng = random.Random(self.seed + epoch * 10007 + self.rank * 1000 + worker_id)

            # Small per-epoch start jitter for novel groupings (bounded so we do
            # not waste much of the segment). Groups stay internally contiguous.
            max_jitter = min(self.chunk_size, max(0, seg_len // 8))
            jitter = rng.randint(0, max_jitter) if max_jitter > 0 else 0

            pos = seg_start + jitter
            while True:
                n_ctx = self._n_context  # may change mid-epoch (curriculum)
                group_size = n_ctx + 1
                group_len = group_size * self.chunk_size

                if pos + group_len > seg_end:
                    break  # not enough room left in this consumer's segment

                # Read one contiguous block of (n_ctx+1)*chunk_size tokens.
                block = self._read_stream(pos, group_len)
                if len(block) < group_len:
                    break  # ran off the end of the global stream

                context_chunks: List[torch.Tensor] = []
                reset_flags: List[bool] = []
                for k in range(group_size):
                    chunk_start = pos + k * self.chunk_size
                    toks = block[k * self.chunk_size: (k + 1) * self.chunk_size]
                    t = torch.tensor(toks, dtype=torch.long)
                    if k < n_ctx:
                        context_chunks.append(t)
                    else:
                        target_ids = t
                    if self.doc_reset:
                        # Chunk k starts a new document iff the token right before
                        # its first token is an EOS. (k==0 at stream pos 0 -> False.)
                        reset_flags.append(self._is_eos_before(chunk_start))

                pos += group_len

                sample: Dict[str, object] = {
                    "context_chunks": context_chunks,
                    "target_ids": target_ids,
                    "is_dolmino": True,
                }
                if self.doc_reset:
                    sample["reset_flags"] = reset_flags
                yield sample

            # Advance epoch for next loop iteration (re-jitter for novel groups)
            epoch += 1
            self._epoch = epoch

    def _iter_per_doc(self) -> Iterator[Dict[str, object]]:
        """Per-document mode: each Arrow row is one complete (variable-length) doc.

        Documents are sharded across (world_size * num_workers) consumers by
        striding a per-epoch shuffled document-index permutation. Each consumer
        slices its documents into consecutive non-overlapping windows of
        ``(n_ctx+1)*chunk_size`` tokens: the first ``n_ctx`` chunks are context,
        the next chunk is the target — all from the SAME document and adjacent,
        so context and target have genuine intra-document cross-chunk dependency.

        A document shorter than ``(n_ctx+1)*chunk_size`` (for the current n_ctx)
        is skipped. Longer documents emit multiple disjoint groups (stride =
        ``(n_ctx+1)*chunk_size``), fully using long documents. Document ORDER is
        reshuffled every epoch; token order WITHIN a document is never shuffled.
        """
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        total_consumers = self.world_size * num_workers
        consumer_id = self.rank * num_workers + worker_id

        epoch = self._epoch
        while True:  # infinite iteration across epochs
            rng = random.Random(self.seed + epoch * 10007)

            # Shuffle DOCUMENT order (same permutation across all consumers for a
            # given epoch); each consumer then strides to get its disjoint shard.
            doc_indices = list(range(self._num_samples))
            rng.shuffle(doc_indices)
            my_docs = doc_indices[consumer_id::total_consumers]

            yielded_this_epoch = 0
            for doc_idx in my_docs:
                tokens = self._ds[int(doc_idx)]["input_ids"]
                doc_len = len(tokens)

                # Slice this document into consecutive non-overlapping groups.
                pos = 0
                while True:
                    n_ctx = self._n_context  # may change mid-epoch (curriculum)
                    group_size = n_ctx + 1
                    group_len = group_size * self.chunk_size

                    if pos + group_len > doc_len:
                        break  # not enough tokens left in this doc for a group

                    context_chunks: List[torch.Tensor] = []
                    target_ids: Optional[torch.Tensor] = None
                    for k in range(group_size):
                        chunk_start = pos + k * self.chunk_size
                        toks = tokens[chunk_start: chunk_start + self.chunk_size]
                        t = torch.tensor(toks, dtype=torch.long)
                        if k < n_ctx:
                            context_chunks.append(t)
                        else:
                            target_ids = t

                    pos += group_len

                    yielded_this_epoch += 1
                    yield {
                        "context_chunks": context_chunks,
                        "target_ids": target_ids,
                        "is_dolmino": True,
                    }

            # Defensive guard (2026-06-15): if an ENTIRE epoch over every doc in
            # this consumer's shard produced zero groups, the loader is starved
            # and the `while True` above would spin forever yielding nothing,
            # which manifests as a silent DDP first-step hang (ranks that drew a
            # T2/babilong step block in all_reduce while dolmino ranks spin here).
            # Fail loudly instead.
            if yielded_this_epoch == 0:
                raise RuntimeError(
                    f"Dolmino per-doc loader produced ZERO groups in a full epoch: "
                    f"(n_ctx+1)*chunk_size = ({self._n_context}+1)*{self.chunk_size} "
                    f"= {(self._n_context + 1) * self.chunk_size} tokens exceeds the "
                    f"longest available document. per-doc dolmino data is capped at "
                    f"~4096 tokens; lower --chunk_size or keep dolmino's n_ctx small "
                    f"(grow long-context via the T2 stream / --t2_curriculum instead)."
                )

            # Advance epoch for next loop iteration (reshuffle document order)
            epoch += 1
            self._epoch = epoch
