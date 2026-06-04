#!/usr/bin/env python
"""Re-slice the packed Dolmino Arrow dataset into per-DOCUMENT rows.

Background
----------
``MemLong/data/processed/dolmino_0.5B_1024/train`` is a HF Arrow dataset with
463,866 rows, each a list of 1024 int32 Llama-3 tokens. It was produced by
packing many documents end-to-end (``doc1<EOS>doc2<EOS>...``, EOS=128001,
BOS=128000) into one flat token stream and then ``reshape(-1, 1024)``. So Arrow
row boundaries do NOT coincide with document boundaries: ~2/3 of documents are
shorter than one 1024-chunk, which means the legacy "adjacent chunks =
context+target" dataloader was pairing UNRELATED documents -> the memory router
correctly collapses to uniform routing (no cross-chunk signal to exploit).

The EOS document boundaries are fully preserved in the stream, so we can recover
every document losslessly by concatenating all rows in ORIGINAL ROW ORDER and
splitting on EOS. No re-tokenisation / re-download is needed.

This script
-----------
1. Streams the Arrow rows in order, concatenating ``input_ids`` into one logical
   token stream (processed batch-by-batch to bound memory).
2. Splits on EOS=128001: every ``[start, eos_index]`` closed interval (including
   the trailing EOS) is one document. The final un-terminated tail is dropped.
3. Keeps documents with length >= ``--min_doc_len`` (default 512).
4. Writes a new HF DatasetDict with a 95/5 train/validation split (split is at
   the DOCUMENT level — token order WITHIN each document is preserved) to
   ``--out_path`` (default ``MemLong/data/processed/dolmino_per_doc``). Single
   column ``input_ids``; each row = one complete document (variable length).

Run with the project venv on the local node:
    .venv/bin/python scripts/reprocess_dolmino_per_doc.py
"""
from __future__ import annotations

import argparse
import os
from typing import Iterator, List

import numpy as np

EOS_TOKEN_ID = 128001
BOS_TOKEN_ID = 128000


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--in_path",
        type=str,
        default="MemLong/data/processed/dolmino_0.5B_1024/train",
        help="Path to the packed (1024-token) Dolmino Arrow dataset.",
    )
    p.add_argument(
        "--out_path",
        type=str,
        default="MemLong/data/processed/dolmino_per_doc",
        help="Output directory for the per-document DatasetDict.",
    )
    p.add_argument(
        "--min_doc_len",
        type=int,
        default=512,
        help="Drop documents shorter than this many tokens.",
    )
    p.add_argument(
        "--val_fraction",
        type=float,
        default=0.05,
        help="Fraction of documents held out for validation (document-level).",
    )
    p.add_argument(
        "--read_batch_size",
        type=int,
        default=2000,
        help="Number of Arrow rows to read per batch while streaming.",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def iter_documents(
    in_path: str, min_doc_len: int, read_batch_size: int
) -> Iterator[List[int]]:
    """Yield each complete document (>= min_doc_len) as a python int list.

    The Arrow rows are streamed in original order; tokens are accumulated into a
    ``carry`` buffer that holds the in-progress (not-yet-EOS-terminated) document
    across batch / row boundaries. Documents are closed at every EOS.
    """
    import datasets

    ds = datasets.load_from_disk(in_path)
    n_rows = len(ds)
    print(f"[reprocess] loaded {n_rows} rows from {in_path}", flush=True)

    carry = np.empty(0, dtype=np.int64)  # in-progress document tail

    n_rows_seen = 0
    for batch in ds.iter(batch_size=read_batch_size):
        # batch["input_ids"] is a list of lists (one per row). Flatten in order:
        # all rows are the same length (1024) and contiguous in the stream, so a
        # plain concatenation reconstructs the segment of the global stream.
        rows = batch["input_ids"]
        n_rows_seen += len(rows)
        # Concatenate this batch's tokens (np.fromiter-style via np.concatenate).
        flat = np.concatenate(
            [np.asarray(r, dtype=np.int64) for r in rows]
        )
        # Prepend any carried-over partial document.
        if carry.size:
            stream = np.concatenate([carry, flat])
        else:
            stream = flat
        carry = np.empty(0, dtype=np.int64)

        eos_positions = np.flatnonzero(stream == EOS_TOKEN_ID)
        prev = 0
        for ei in eos_positions:
            doc = stream[prev : ei + 1]  # inclusive of the trailing EOS
            prev = ei + 1
            if doc.shape[0] >= min_doc_len:
                yield doc.tolist()
        # Remaining tail (after the last EOS) is an unfinished document -> carry.
        if prev < stream.shape[0]:
            carry = stream[prev:].copy()

    print(
        f"[reprocess] streamed {n_rows_seen} rows; dropped final unterminated "
        f"tail of {carry.size} tokens",
        flush=True,
    )


def main() -> None:
    args = parse_args()

    import datasets

    # Materialise documents into one in-memory list of token lists. With
    # min_doc_len=512 we expect ~280K docs (~1.9GB of int data), which fits.
    docs: List[List[int]] = []
    total_docs_emitted = 0
    lengths: List[int] = []
    for doc in iter_documents(args.in_path, args.min_doc_len, args.read_batch_size):
        docs.append(doc)
        lengths.append(len(doc))
        total_docs_emitted += 1
        if total_docs_emitted % 50000 == 0:
            print(f"[reprocess] kept {total_docs_emitted} docs so far", flush=True)

    n_docs = len(docs)
    if n_docs == 0:
        raise RuntimeError("No documents survived filtering — check min_doc_len / data.")

    lengths_np = np.asarray(lengths)
    total_tokens = int(lengths_np.sum())
    print("[reprocess] ============ statistics ============", flush=True)
    print(f"[reprocess] kept documents (>= {args.min_doc_len} tok): {n_docs}", flush=True)
    print(f"[reprocess] total tokens in kept docs: {total_tokens}", flush=True)
    print(
        "[reprocess] doc length  min/median/mean/p90/p99/max = "
        f"{int(lengths_np.min())}/{int(np.median(lengths_np))}/"
        f"{lengths_np.mean():.1f}/{int(np.percentile(lengths_np, 90))}/"
        f"{int(np.percentile(lengths_np, 99))}/{int(lengths_np.max())}",
        flush=True,
    )
    for thr in (512, 768, 1024, 2048):
        frac = float((lengths_np >= thr).mean())
        print(f"[reprocess]   docs >= {thr:>5} tok: {frac * 100:.1f}%", flush=True)

    # Build the dataset from the in-memory documents.
    full = datasets.Dataset.from_dict({"input_ids": docs})

    # Document-level 95/5 split. shuffle=True only chooses WHICH docs go to val;
    # the token order within each document is untouched.
    split = full.train_test_split(
        test_size=args.val_fraction, shuffle=True, seed=args.seed
    )
    ddict = datasets.DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )

    os.makedirs(os.path.dirname(args.out_path) or ".", exist_ok=True)
    ddict.save_to_disk(args.out_path)
    print(
        f"[reprocess] wrote DatasetDict to {args.out_path}: "
        f"train={len(ddict['train'])}, validation={len(ddict['validation'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
