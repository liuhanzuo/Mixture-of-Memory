"""T2 chunked associative-recall (NIAH) IterableDataset for Memory-Space training.

This is the **chunked** counterpart of ``niah_dataset.NIAHIterableDataset``.

Motivation (see status/STAGING_REPORT_readout_pivot.md §5)
---------------------------------------------------------
The legacy ``NIAHIterableDataset`` yields a FLAT ``input_ids``/``labels`` sequence
that goes through a plain forward pass — so local causal attention can read the
needle directly and the memory bank is never forced to carry it. T2 fixes this by
emitting samples in the SAME format as ``DolminoCurriculumDataset``:

    {context_chunks: [n_ctx tensors, each chunk_size], target_ids, answer_mask}

These are fed to ``dolmino_train_step`` (the HARDER objective): the context chunks
stream into the memory bank under ``no_grad`` and are detached, then LM loss is
computed only on the target (question) chunk — but here, further restricted by
``answer_mask`` to ONLY the answer digit tokens. The needle therefore lives in a
detached context chunk that is NOT in the target's attention window, so the answer
can only be recovered by precisely addressing the memory slot that encoded it.
This applies maximal "precise readout" pressure (the bottleneck identified in the
staging report).

chunk_size is a first-class experiment variable (§5.4): smaller chunks mean the
needle occupies a larger fraction of each memory write, so the (key->value)
encoding is cleaner and readout approaches a table lookup. To compare chunk_sizes
fairly we hold the **needle->query token distance** (``gap_tokens``) constant and
derive ``n_ctx = round(gap_tokens / chunk_size)`` — so chunk128 uses 4x as many
chunks as chunk512 to span the same token gap, isolating the granularity effect
from the distance effect.

Key construction guarantees
---------------------------
* The queried needle sentence is placed ENTIRELY inside a single context chunk
  (``insert_at + len(needle_ids) <= chunk_size``); it never straddles a boundary.
* The queried needle is placed at the START (offset 0) of a context chunk. By
  default that is the FIRST context chunk, so the needle->query token distance
  equals exactly ``n_ctx * chunk_size`` for every chunk_size (clean fixed gap).
  When ``random_needle_chunk=True`` (learn-to-select training) it is a RANDOM
  context chunk (offset 0); the chosen index is returned as
  ``needle_chunk_index`` so a selection loss can supervise against it.
* ``answer_mask`` is True only on the answer's digit-token positions in the target
  chunk; everything else (question prefix + padding) is masked out of the loss.
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, Iterator, List

import numpy as np
import torch
import torch.utils.data


class NIAHChunkedDataset(torch.utils.data.IterableDataset):
    """Infinite chunked associative-recall dataset (T2).

    Each yielded sample is a dict with the SAME keys as DolminoCurriculumDataset
    plus ``answer_mask``:

        ``context_chunks``: list of n_ctx LongTensors, each [chunk_size]
        ``target_ids``:     LongTensor [chunk_size]   (question + answer + pad)
        ``answer_mask``:    BoolTensor [chunk_size]    (True only on answer digits)
        ``is_t2``:          True
        ``code``:           str — the queried 5-digit code (for diagnostics)

    Args:
        background_data: [N_chunks, L] int array of pre-tokenised natural-text
            token IDs (e.g. ``data/pg19_chunks_llama3.npy``). Used to fill
            non-needle context chunks so the memory channel sees realistic text.
        chunk_size: Tokens per chunk. MUST match the training forward chunk_size
            and (downstream) the eval ``--chunk_size``.
        gap_tokens: Target needle->query token distance held constant across
            chunk_sizes for fair comparison. ``n_ctx = round(gap_tokens /
            chunk_size)`` (clamped to >= 1).
        tokenizer: HuggingFace tokenizer (encode / pad_token_id / eos_token_id).
        num_keys: Number of (name -> code) mappings written into the context.
            One of them is queried; the other ``num_keys - 1`` are distractors.
            Default 1 (single needle); >1 reserved for the T2b multi-key ablation.
        seed: Base RNG seed. Each DDP worker adds its worker_id so workers are
            de-correlated (mirrors NIAHIterableDataset).
        background_skip: Skip the first ``background_skip`` background chunks to
            avoid train/eval contamination.
    """

    def __init__(
        self,
        background_data: np.ndarray,
        chunk_size: int,
        gap_tokens: int,
        tokenizer: Any,
        num_keys: int = 1,
        seed: int = 42,
        background_skip: int = 0,
        random_needle_chunk: bool = False,
    ) -> None:
        """random_needle_chunk (2026-06-27, learn-to-select):
            False (default) -> the queried needle is ALWAYS at context chunk 0
                (legacy behaviour; the produced samples are BYTE-IDENTICAL to the
                prior dataset because no extra RNG calls are made).
            True  -> the queried needle is placed at a RANDOM context chunk index
                in [0, n_ctx-1] (offset 0 within that chunk). This is MANDATORY for
                the supervised-selection training (status/LEARN_TO_SELECT_DESIGN):
                with the needle always at chunk 0 a selector trivially learns
                "always pick chunk 0", which does not transfer. The chosen index is
                returned in the sample as ``needle_chunk_index`` so the train loop
                can supervise the per-chunk salience against it.
        """
        super().__init__()
        if background_data.ndim != 2:
            raise ValueError(
                f"background_data must be 2-D [N_chunks, L], got shape {background_data.shape}"
            )
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
        if gap_tokens < 1:
            raise ValueError(f"gap_tokens must be >= 1, got {gap_tokens}")
        if num_keys < 1:
            raise ValueError(f"num_keys must be >= 1, got {num_keys}")

        self.background_data = background_data
        self.chunk_size = chunk_size
        self.gap_tokens = gap_tokens
        self.tokenizer = tokenizer
        self.num_keys = num_keys
        self.seed = seed
        self.background_skip = background_skip
        self.random_needle_chunk = bool(random_needle_chunk)

        # n_ctx derived so the needle (at chunk 0 offset 0) -> query (target chunk
        # start) distance == n_ctx * chunk_size ~= gap_tokens, for any chunk_size.
        self.n_ctx = max(1, int(round(gap_tokens / chunk_size)))

        self._n_chunks = len(background_data)
        self._usable_start = (
            background_skip % self._n_chunks if self._n_chunks > 0 else 0
        )

        self._pad_id: int = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    # --------------------------------------------------------------------- #
    # Curriculum support: let the needle->query distance grow during training.
    # --------------------------------------------------------------------- #
    def set_n_ctx(self, n_ctx: int) -> None:
        """Override the number of context chunks (and hence the needle->query
        distance = n_ctx * chunk_size). Used by curriculum schedules so the T2
        recall distance grows alongside the dolmino context length, instead of
        staying fixed at construction time. Safe to call mid-training; the next
        sample built by ``_make_sample`` reads ``self.n_ctx`` fresh."""
        self.n_ctx = max(1, int(n_ctx))

    # --------------------------------------------------------------------- #
    # IterableDataset protocol
    # --------------------------------------------------------------------- #
    def __iter__(self) -> Iterator[Dict[str, Any]]:
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        rng = random.Random(self.seed + worker_id)

        all_indices = list(range(self._usable_start, self._n_chunks))
        if not all_indices:
            all_indices = list(range(self._n_chunks))
        worker_indices = all_indices[worker_id::num_workers]
        if not worker_indices:
            worker_indices = all_indices

        pos = 0  # cursor into worker_indices (cycled)

        while True:
            sample, pos = self._make_sample(rng, worker_indices, pos)
            yield sample

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #
    def _get_bg_chunk(self, worker_indices: List[int], pos: int) -> tuple[list[int], int]:
        """Return (token_list of length chunk_size, new_pos) cycling bg chunks."""
        idx = worker_indices[pos % len(worker_indices)]
        new_pos = (pos + 1) % len(worker_indices)
        tokens = self.background_data[idx, : self.chunk_size].tolist()
        if len(tokens) < self.chunk_size:
            tokens = tokens + [self._pad_id] * (self.chunk_size - len(tokens))
        return tokens, new_pos

    def _embed_needle(self, bg_tokens: list[int], needle_ids: list[int],
                      insert_at: int) -> list[int]:
        """Overwrite ``len(needle_ids)`` background tokens at ``insert_at`` with
        the needle, keeping the chunk exactly chunk_size tokens."""
        chunk = (
            bg_tokens[:insert_at]
            + needle_ids
            + bg_tokens[insert_at + len(needle_ids):]
        )
        chunk = (chunk + [self._pad_id] * self.chunk_size)[: self.chunk_size]
        return chunk

    def _make_sample(
        self,
        rng: random.Random,
        worker_indices: List[int],
        pos: int,
    ) -> tuple[Dict[str, Any], int]:
        """Build one chunked associative-recall sample.

        Layout (n_ctx context chunks + 1 target chunk):
            [chunk_0 (queried needle at offset 0) + bg] [chunk_1 bg] ...
            [chunk_{n_ctx-1} bg] [target: question + answer + pad]

        Distractor needles (num_keys-1 of them) are scattered into the remaining
        context chunks at random offsets that keep each needle inside one chunk.
        """
        n_ctx = self.n_ctx
        cs = self.chunk_size

        # 1. Sample num_keys distinct (name -> code) mappings.
        # codes are stored SPACE-SEPARATED ("8 0 4 0 2") so the Llama-3 BPE
        # tokenizer emits ONE token per digit (5 independent answer tokens),
        # rather than merging "80402" into ~2 BPE tokens. This gives a sharper
        # precise-readout signal: each of the 5 digits is a separate retrieval
        # target and the model cannot shortcut by predicting a later digit from
        # an earlier one within a merged token.
        names: list[str] = []
        codes: list[str] = []          # spaced form, e.g. "8 0 4 0 2"
        seen: set[str] = set()
        while len(names) < self.num_keys:
            nm = "".join(rng.choices(string.ascii_uppercase, k=6))
            if nm in seen:
                continue
            seen.add(nm)
            names.append(nm)
            codes.append(" ".join(rng.choices(string.digits, k=5)))

        query_k = 0  # the FIRST mapping is the queried one (placed at chunk0/offset0)

        # 2. Tokenise needles, question, answer.
        #    needle and query use the SAME spaced-code form so the written and
        #    read-out token shapes match exactly.
        needle_ids_list: list[list[int]] = []
        for nm, cd in zip(names, codes):
            sent = f"MEMORIZE: The secret code for agent {nm} is {cd}. END_MEMORIZE"
            needle_ids_list.append(
                self.tokenizer.encode(" " + sent, add_special_tokens=False)
            )
        question_ids: list[int] = self.tokenizer.encode(
            f"The secret code for agent {names[query_k]} is",
            add_special_tokens=False,
        )
        # Encode the answer as " 8 0 4 0 2" (leading space binds to the first
        # digit token, so each digit -> its own token; 5 answer tokens total).
        answer_ids: list[int] = self.tokenizer.encode(
            " " + codes[query_k], add_special_tokens=False
        )

        # 3. Build context chunks filled with background text.
        context_chunks: list[list[int]] = []
        for _ in range(n_ctx):
            bg, pos = self._get_bg_chunk(worker_indices, pos)
            context_chunks.append(bg)

        # 4a. Place the QUERIED needle. Legacy (random_needle_chunk=False): chunk 0,
        #     offset 0 -> fixed gap == n_ctx*chunk_size (NO rng call, so byte-
        #     identical to the prior dataset). Learn-to-select
        #     (random_needle_chunk=True): a RANDOM context chunk, offset 0, so the
        #     selector cannot shortcut to "always chunk 0". The chosen index is
        #     returned as ``needle_chunk_index`` for the supervised-selection loss.
        if self.random_needle_chunk and n_ctx > 1:
            needle_chunk_index = rng.randint(0, n_ctx - 1)
        else:
            needle_chunk_index = 0
        q_needle = needle_ids_list[query_k]
        if len(q_needle) > cs:
            q_needle = q_needle[:cs]  # safety (should never trigger for ~25-token needle)
        context_chunks[needle_chunk_index] = self._embed_needle(
            context_chunks[needle_chunk_index], q_needle, 0
        )

        # 4b. Scatter distractor needles into other context chunks (if num_keys>1).
        if self.num_keys > 1 and n_ctx > 1:
            other_chunks = [c for c in range(n_ctx) if c != needle_chunk_index]
            rng.shuffle(other_chunks)
            for ki in range(1, self.num_keys):
                d_needle = needle_ids_list[ki]
                if len(d_needle) > cs:
                    d_needle = d_needle[:cs]
                # pick a chunk (cycle through available ones)
                ci = other_chunks[(ki - 1) % len(other_chunks)]
                max_off = max(0, cs - len(d_needle))
                insert_at = rng.randint(0, max_off)
                context_chunks[ci] = self._embed_needle(
                    context_chunks[ci], d_needle, insert_at
                )

        # 5. Build target chunk: question + answer + padding to chunk_size.
        target_raw = question_ids + answer_ids
        if len(target_raw) < cs:
            target_tokens = target_raw + [self._pad_id] * (cs - len(target_raw))
        else:
            target_tokens = target_raw[:cs]

        # 6. answer_mask: True ONLY on the 5 digit-token positions.
        # The spaced answer " 6 7 0 0 8" tokenizes to [' ','6',' ','7',...] —
        # digit tokens interleaved with space tokens. We mask only the digit
        # tokens (skip the spaces) so the LM loss falls purely on the 5 retrieval
        # targets and is not diluted by trivial space-token prediction.
        answer_mask = [False] * cs
        ans_start = len(question_ids)
        for j, tid in enumerate(answer_ids):
            p = ans_start + j
            if p >= cs:
                break
            if self.tokenizer.decode([tid]).strip().isdigit():
                answer_mask[p] = True

        sample = {
            "context_chunks": [
                torch.tensor(c, dtype=torch.long) for c in context_chunks
            ],
            "target_ids": torch.tensor(target_tokens, dtype=torch.long),
            "answer_mask": torch.tensor(answer_mask, dtype=torch.bool),
            "is_t2": True,
            "code": codes[query_k],
            # Document-absolute context-chunk index holding the queried needle.
            # 0 in the legacy (chunk-0) layout; a random index when
            # random_needle_chunk=True. Used by the supervised-selection loss
            # (status/LEARN_TO_SELECT_DESIGN) as the CE target for the per-chunk
            # reader-attn salience.
            "needle_chunk_index": int(needle_chunk_index),
        }
        return sample, pos


# --------------------------------------------------------------------------- #
# Collate function (batch_size > 1) — mirrors dolmino_collate_fn style
# --------------------------------------------------------------------------- #


def niah_chunked_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Stack a list of NIAHChunkedDataset samples into batched tensors.

    All samples in a batch share the same ``n_ctx`` and ``chunk_size`` (T2 uses a
    fixed gap -> fixed n_ctx). Returns:

        ``context_chunks``: list of n_ctx tensors, each [B, chunk_size]
        ``target_ids``:     [B, chunk_size]
        ``answer_mask``:    [B, chunk_size]
        ``is_t2``:          True
    """
    if not batch:
        raise ValueError("niah_chunked_collate_fn received an empty batch")

    n_ctx = len(batch[0]["context_chunks"])
    for s in batch:
        if len(s["context_chunks"]) != n_ctx:
            raise ValueError(
                "niah_chunked_collate_fn: samples in a batch have different n_ctx "
                f"({len(s['context_chunks'])} vs {n_ctx}); T2 n_ctx must be constant."
            )

    context_chunks: List[torch.Tensor] = []
    for k in range(n_ctx):
        col = [s["context_chunks"][k] for s in batch]
        min_len = min(t.shape[0] for t in col)
        context_chunks.append(torch.stack([t[:min_len] for t in col], dim=0))

    tgt_col = [s["target_ids"] for s in batch]
    tgt_min = min(t.shape[0] for t in tgt_col)
    target_ids = torch.stack([t[:tgt_min] for t in tgt_col], dim=0)

    msk_col = [s["answer_mask"] for s in batch]
    answer_mask = torch.stack([t[:tgt_min] for t in msk_col], dim=0)

    # needle_chunk_index: [B] long tensor (CE target for the selection loss).
    # Absent in older samples -> default 0 (legacy chunk-0 layout).
    needle_chunk_index = torch.tensor(
        [int(s.get("needle_chunk_index", 0)) for s in batch], dtype=torch.long
    )

    return {
        "context_chunks": context_chunks,
        "target_ids": target_ids,
        "answer_mask": answer_mask,
        "is_t2": True,
        "needle_chunk_index": needle_chunk_index,
    }
