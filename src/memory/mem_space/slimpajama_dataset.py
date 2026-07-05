"""SlimPajama prediction-style self-supervised dataset (2026-07-05).

WHY (user directive 2026-07-05): the babilong-qa5 synthetic SFT (recall sweep)
overfits the qa1/2/5 token-shortcut — it does NOT teach a *general* memory
readout ability (recall0.02 got qa5 16k=40 but LongBench f1<11, RULER 8k=8).
The literature (Beacon/MemoryLLM/M+/AutoCompressor/CEPE) all learn memory via
**generic long-document dense LM**, never small-answer-space synthetic NIAH.

This dataset feeds the EXISTING prediction self-supervision path
(`dolmino_train_step` with `--last_chunk_loss_only`, answer_mask=None): context
chunks stream into memory (no_grad → detached), then the target chunk's
next-token prediction can ONLY draw on prior context THROUGH the memory bank.
That is the "useful memory = can I still generate the following text from what I
remember" objective (prediction, NOT reconstruction — per
versions/v_prediction_not_reconstruction_2026-06-25.md).

Data: ``data/slimpajama_chunks_4096.npy`` — [N=1.57M, 4096] uint16, pre-tokenised
generic long documents (SlimPajama-6B), the same corpus family M+/AutoCompressor
train on. Each row is one 4096-token document window.

Drop-in compatible with DolminoCurriculumDataset: same yielded dict
(``context_chunks`` / ``target_ids`` / ``is_dolmino`` / ``sample_id``), same
``set_n_context`` curriculum hook, same DDP sharding, so it reuses the existing
DataLoader + dolmino_collate_fn + dolmino_train_step + curriculum scheduler with
ZERO changes to the training loop other than swapping which dataset is built.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.utils.data


class SlimPajamaPredictionDataset(torch.utils.data.IterableDataset):
    """Generic long-document dense-LM self-supervision from a pre-tokenised npy.

    Each yielded sample (identical schema to DolminoCurriculumDataset):
        ``context_chunks``: list of n_ctx LongTensors, each [chunk_size]
        ``target_ids``:     LongTensor [chunk_size]
        ``is_dolmino``:     True   (so the train loop routes it to
                            dolmino_train_step — the prediction path)
        ``sample_id``:      (row_idx, group_pos)  (stable, for distill cache)

    A document (npy row of length L=row_len, e.g. 4096) is sliced into
    consecutive non-overlapping windows of (n_ctx+1)*chunk_size tokens: the first
    n_ctx chunks are context, the last is the target — all adjacent within the
    SAME row, so context/target have genuine intra-document dependency. Rows too
    short for one group (given the current curriculum n_ctx) are skipped. Row
    order is reshuffled every epoch; token order within a row is never shuffled.
    """

    def __init__(
        self,
        data_path: str,
        chunk_size: int = 512,
        n_context: int = 3,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
        max_rows: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.data_path = data_path
        self.chunk_size = int(chunk_size)
        self._n_context = int(n_context)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        # mmap so 12.8GB npy is not copied into every DDP worker's RAM.
        self._data = np.load(data_path, mmap_mode="r")
        if self._data.ndim != 2:
            raise ValueError(
                f"SlimPajama npy must be 2-D [N_rows, row_len]; got {self._data.shape}"
            )
        self._num_rows = int(self._data.shape[0])
        self._row_len = int(self._data.shape[1])
        if max_rows is not None:
            self._num_rows = min(self._num_rows, int(max_rows))
        self._epoch = 0

    # curriculum hook — same name/semantics as DolminoCurriculumDataset
    def set_n_context(self, n: int) -> None:
        """Update context-chunk count (called by the curriculum scheduler).
        Safe mid-training; the next window built reads self._n_context fresh."""
        self._n_context = max(1, int(n))

    @property
    def n_context(self) -> int:
        return self._n_context

    def __iter__(self):
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
        while True:  # infinite across epochs
            rng = random.Random(self.seed + epoch * 10007)
            row_order = list(range(self._num_rows))
            rng.shuffle(row_order)
            my_rows = row_order[consumer_id::total_consumers]
            if not my_rows:
                my_rows = row_order  # degenerate: fewer rows than consumers

            yielded = 0
            for row_idx in my_rows:
                # materialise one row (mmap slice -> list of python ints)
                tokens = self._data[int(row_idx)].tolist()
                doc_len = len(tokens)

                pos = 0
                while True:
                    n_ctx = self._n_context  # may change mid-epoch (curriculum)
                    group_size = n_ctx + 1
                    group_len = group_size * self.chunk_size
                    if pos + group_len > doc_len:
                        break
                    group_pos = pos // group_len
                    context_chunks: List[torch.Tensor] = []
                    target_ids: Optional[torch.Tensor] = None
                    for k in range(group_size):
                        s = pos + k * self.chunk_size
                        toks = tokens[s: s + self.chunk_size]
                        t = torch.tensor(toks, dtype=torch.long)
                        if k < n_ctx:
                            context_chunks.append(t)
                        else:
                            target_ids = t
                    pos += group_len
                    yielded += 1
                    yield {
                        "context_chunks": context_chunks,
                        "target_ids": target_ids,
                        "is_dolmino": True,   # route to dolmino_train_step
                        "sample_id": (int(row_idx), int(group_pos)),
                    }

            if yielded == 0:
                raise RuntimeError(
                    f"SlimPajama loader produced ZERO groups in a full epoch: "
                    f"(n_ctx+1)*chunk_size = ({self._n_context}+1)*{self.chunk_size} "
                    f"= {(self._n_context + 1) * self.chunk_size} tokens exceeds the "
                    f"npy row length {self._row_len}. Lower --chunk_size or n_ctx."
                )
            epoch += 1
            self._epoch = epoch
