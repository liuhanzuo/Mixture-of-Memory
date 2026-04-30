"""NIAH (Needle-in-a-Haystack) IterableDataset for Memory-Space training.

Yields mixed batches of pg19 LM sequences and synthetic NIAH sequences so
the model simultaneously learns:
  1. Language modelling on natural text (pg19 batches).
  2. Content-addressed associative retrieval via NIAH supervision (model must
     output a 5-digit secret code hidden N chunks earlier in the sequence).

Reference: ops/research_notes/20260427_swa_memory_design.md §Stage1

Design decisions
----------------
* IterableDataset with infinite iteration — the training loop controls the
  number of steps via `max_steps`, not epoch counting.
* DDP sharding: workers shard the pg19 index space so each rank/worker sees a
  distinct sequence of pg19 chunks (no data duplication).
* NIAH sequences are generated entirely on-the-fly; no disk storage needed.
* Labels: -100 everywhere EXCEPT the 5 answer-digit tokens so that the
  standard HF cross-entropy (CausalLM.loss) only back-propagates on the code.
  pg19 batches use labels == input_ids (full LM supervision).
"""
from __future__ import annotations

import random
import string
from typing import Any, Dict, Iterator, Optional

import numpy as np
import torch
import torch.utils.data


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #


class NIAHIterableDataset(torch.utils.data.IterableDataset):
    """Infinite mixed-batch dataset: (1-p) pg19 + p NIAH.

    Args:
        pg19_data:          [N_chunks, chunk_size] int array of pre-tokenised
                            pg19 token IDs.
        chunk_size:         Number of tokens per chunk (must match the training
                            forward-pass sequence length).
        niah_mix_fraction:  Fraction of yielded batches that are NIAH samples
                            (e.g. 0.10 → 1 NIAH per 10 batches on average).
        niah_max_N:         Maximum number of pg19 background chunks between
                            needle insertion and query chunk (inclusive).  The
                            actual gap N_gap is sampled uniformly from [1, max_N].
        tokenizer:          HuggingFace tokenizer with `encode(str, ...)` method.
                            Used to tokenise needle/question strings.
        seed:               Base random seed.  Each DDP worker adds its worker_id
                            to the seed so workers are de-correlated.
        pg19_skip:          Skip the first `pg19_skip` chunks to avoid
                            train/eval contamination.  E.g., pass the number of
                            pg19 chunks reserved for the eval set.
    """

    def __init__(
        self,
        pg19_data: np.ndarray,
        chunk_size: int,
        niah_mix_fraction: float,
        niah_max_N: int,
        tokenizer: Any,
        seed: int = 42,
        pg19_skip: int = 0,
    ) -> None:
        super().__init__()
        if pg19_data.ndim != 2:
            raise ValueError(
                f"pg19_data must be 2-D [N_chunks, chunk_size], got shape {pg19_data.shape}"
            )
        if not 0.0 <= niah_mix_fraction <= 1.0:
            raise ValueError(f"niah_mix_fraction must be in [0, 1], got {niah_mix_fraction}")
        if niah_max_N < 1:
            raise ValueError(f"niah_max_N must be >= 1, got {niah_max_N}")

        self.pg19_data = pg19_data
        self.chunk_size = chunk_size
        self.niah_mix_fraction = niah_mix_fraction
        self.niah_max_N = niah_max_N
        self.tokenizer = tokenizer
        self.seed = seed
        self.pg19_skip = pg19_skip

        # Total usable chunks (skip the reserved eval prefix).
        self._n_chunks = len(pg19_data)
        self._usable_start = pg19_skip % self._n_chunks if self._n_chunks > 0 else 0

        # Pre-compute pad token ID once.
        self._pad_id: int = (
            tokenizer.pad_token_id
            if tokenizer.pad_token_id is not None
            else tokenizer.eos_token_id
        )

    # --------------------------------------------------------------------- #
    # IterableDataset protocol
    # --------------------------------------------------------------------- #

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        """Yield individual samples (not batches).  The DataLoader handles batching."""
        # DDP-aware worker seeding: each DDP worker (spawned by DataLoader)
        # gets a unique seed by adding its worker_id.  Two separate workers
        # on the same rank will still draw different data because they each
        # start at a different position in the pg19 index space.
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            worker_id: int = worker_info.id
            num_workers: int = worker_info.num_workers
        else:
            worker_id = 0
            num_workers = 1

        # Each worker maintains an independent RNG seeded distinctly.
        rng = random.Random(self.seed + worker_id)
        np_rng = np.random.default_rng(self.seed + worker_id + 10000)

        # Build a shuffled index list for this worker's shard of pg19 chunks.
        # Workers divide the available pg19 chunks by striding.
        all_indices = list(range(self._usable_start, self._n_chunks))
        if not all_indices:
            # Safety: if nothing is usable, wrap around to the full array.
            all_indices = list(range(self._n_chunks))

        # Each worker takes every `num_workers`-th chunk starting at `worker_id`.
        worker_indices = all_indices[worker_id::num_workers]
        if not worker_indices:
            worker_indices = all_indices  # fallback: no sharding

        pos = 0  # current position in `worker_indices` (cycled)

        while True:
            # Decide: emit a NIAH sample or a pg19 LM sample.
            if rng.random() < self.niah_mix_fraction:
                sample = self._make_niah_sample(rng, np_rng, worker_indices, pos)
            else:
                sample, pos = self._make_pg19_sample(worker_indices, pos)
            yield sample

    # --------------------------------------------------------------------- #
    # Internal helpers
    # --------------------------------------------------------------------- #

    def _next_chunk_idx(self, worker_indices: list[int], pos: int) -> tuple[int, int]:
        """Return (chunk_index, new_pos) cycling through worker_indices."""
        idx = worker_indices[pos % len(worker_indices)]
        return idx, (pos + 1) % len(worker_indices)

    def _get_chunk(self, worker_indices: list[int], pos: int) -> tuple[list[int], int]:
        """Return (token_list, new_pos) for the next pg19 chunk."""
        idx, new_pos = self._next_chunk_idx(worker_indices, pos)
        tokens = self.pg19_data[idx, : self.chunk_size].tolist()
        # Ensure exactly chunk_size tokens (pad if the stored chunk is shorter).
        if len(tokens) < self.chunk_size:
            tokens = tokens + [self._pad_id] * (self.chunk_size - len(tokens))
        return tokens, new_pos

    def _make_pg19_sample(
        self,
        worker_indices: list[int],
        pos: int,
    ) -> tuple[Dict[str, Any], int]:
        """Return a single pg19 LM batch item and the updated pg19 cursor."""
        tokens, new_pos = self._get_chunk(worker_indices, pos)
        input_ids = torch.tensor(tokens, dtype=torch.long)
        return {
            "input_ids": input_ids,
            "labels":    input_ids.clone(),
            "is_niah":   False,
        }, new_pos

    def _make_niah_sample(
        self,
        rng: random.Random,
        np_rng: np.random.Generator,
        worker_indices: list[int],
        pos: int,
    ) -> Dict[str, Any]:
        """Build one NIAH sequence and return the sample dict.

        Sequence layout (all chunks are exactly chunk_size tokens):

            [chunk_0 (pg19)] ... [chunk_{needle_chunk} (pg19 + needle)]
            ... [chunk_{N_gap-1} (pg19)] [chunk_{N_gap} (question, padded)]

        The model receives N_gap+1 chunks concatenated (length = (N_gap+1)*chunk_size).
        Labels are -100 everywhere except the 5 answer-digit positions at the
        very end of the last chunk (immediately after the question suffix tokens).

        Returns:
            Dict with keys:
                input_ids:          [total_len] long tensor
                labels:             [total_len] long tensor (-100 mask applied)
                is_niah:            True
                N_gap:              int — gap used
                code:               str — expected 5-digit code
                question_start_idx: int — token index of the question chunk start
        """
        # 1. Sample N_gap and needle parameters.
        N_gap: int = rng.randint(1, self.niah_max_N)
        name: str = "".join(rng.choices(string.ascii_uppercase, k=6))
        code: str = "".join(rng.choices(string.digits, k=5))

        needle_sentence = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        question_suffix = f"The secret code for agent {name} is "

        # 2. Tokenise needle and question.
        needle_ids: list[int] = self.tokenizer.encode(
            " " + needle_sentence, add_special_tokens=False
        )
        question_ids: list[int] = self.tokenizer.encode(
            question_suffix, add_special_tokens=False
        )
        # Tokenise the answer digits for label masking.
        answer_ids: list[int] = self.tokenizer.encode(
            code, add_special_tokens=False
        )

        # 3. Build background pg19 chunks (N_gap chunks).
        needle_chunk_pos: int = N_gap // 2  # midpoint chunk gets the needle
        all_chunks: list[list[int]] = []

        local_pos = pos  # we don't update the dataset-level pos here (NIAH is self-contained)
        for chunk_i in range(N_gap):
            bg_tokens, local_pos = self._get_chunk(worker_indices, local_pos)

            if chunk_i == needle_chunk_pos:
                # Insert needle at a random byte position within the chunk.
                insert_at: int = rng.randint(0, max(0, self.chunk_size - len(needle_ids)))
                # Replace `len(needle_ids)` tokens starting at `insert_at`
                # so the chunk stays exactly chunk_size tokens.
                chunk_with_needle = (
                    bg_tokens[:insert_at]
                    + needle_ids
                    + bg_tokens[insert_at + len(needle_ids):]
                )
                # Trim/pad to chunk_size.
                chunk_with_needle = (chunk_with_needle + [self._pad_id] * self.chunk_size)[: self.chunk_size]
                all_chunks.append(chunk_with_needle)
            else:
                all_chunks.append(bg_tokens)

        # 4. Build the question chunk: question_ids + answer_ids + padding.
        question_chunk_raw = question_ids + answer_ids
        # Pad to chunk_size with pad_id.
        if len(question_chunk_raw) < self.chunk_size:
            question_chunk_padded = (
                question_chunk_raw + [self._pad_id] * (self.chunk_size - len(question_chunk_raw))
            )
        else:
            # Truncate if somehow over budget (extremely long Q+A).
            question_chunk_padded = question_chunk_raw[: self.chunk_size]
        all_chunks.append(question_chunk_padded)

        # 5. Concatenate into flat sequence.
        flat_tokens: list[int] = []
        for chunk in all_chunks:
            flat_tokens.extend(chunk)
        total_len = len(flat_tokens)

        input_ids = torch.tensor(flat_tokens, dtype=torch.long)

        # 6. Build labels: -100 everywhere, then unmask answer tokens.
        labels = torch.full((total_len,), -100, dtype=torch.long)

        question_start_idx = N_gap * self.chunk_size  # byte position of question chunk
        # Within the question chunk, answer tokens start after question_ids.
        answer_start_in_chunk = len(question_ids)
        answer_global_start = question_start_idx + answer_start_in_chunk

        # Unmask only the answer digit token positions.
        n_ans = len(answer_ids)
        if answer_global_start + n_ans <= total_len:
            for j in range(n_ans):
                labels[answer_global_start + j] = input_ids[answer_global_start + j]

        return {
            "input_ids":          input_ids,
            "labels":             labels,
            "is_niah":            True,
            "N_gap":              N_gap,
            "code":               code,
            "question_start_idx": question_start_idx,
        }


# --------------------------------------------------------------------------- #
# Collate function (handles variable-length sequences if needed)
# --------------------------------------------------------------------------- #


def niah_collate_fn(batch: list[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate a list of samples from NIAHIterableDataset.

    Handles batches that may mix pg19 samples (shorter) with NIAH samples
    (longer) by padding the shorter inputs with pad_id and labels with -100.
    In practice the DataLoader batch_size=1 is safest for NIAH evaluation;
    this collate also supports batch_size>1 for training.
    """
    input_ids_list = [s["input_ids"] for s in batch]
    labels_list    = [s["labels"]    for s in batch]

    # Find max length in the batch.
    max_len = max(t.shape[0] for t in input_ids_list)

    padded_ids    = []
    padded_labels = []
    for ids, lbl in zip(input_ids_list, labels_list):
        pad_len = max_len - ids.shape[0]
        if pad_len > 0:
            ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
            lbl = torch.cat([lbl, torch.full((pad_len,), -100, dtype=torch.long)])
        padded_ids.append(ids)
        padded_labels.append(lbl)

    out: Dict[str, Any] = {
        "input_ids": torch.stack(padded_ids, dim=0),
        "labels":    torch.stack(padded_labels, dim=0),
        "is_niah":   any(s.get("is_niah", False) for s in batch),
    }
    # Pass through optional metadata for the first sample (batch_size=1 typical).
    if batch[0].get("is_niah", False):
        out["N_gap"]              = batch[0].get("N_gap", 0)
        out["code"]               = batch[0].get("code", "")
        out["question_start_idx"] = batch[0].get("question_start_idx", 0)
    return out
