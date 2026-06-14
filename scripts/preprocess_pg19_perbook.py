#!/usr/bin/env python
"""Download PG19 from HuggingFace and build a per-BOOK Llama-3 tokenized dataset.

Motivation
----------
Our memory-augmented Llama-3-8B training data (dolmino) is all <=16k tokens, but
BABILong eval probes up to 32k. The model therefore never sees long sequences at
train time, a likely contributor to the long-range plateau. We need >=32k-token
REAL long documents, tokenized with the Meta-Llama-3 tokenizer.

PG19 (Project Gutenberg books) is an ideal source: a single book is naturally
60k-100k+ tokens. The canonical ``deepmind/pg19`` repo ships a legacy loading
script (``pg19.py``) that ``datasets`` 4.x refuses to run
("Dataset scripts are no longer supported"). We instead stream the parquet
mirror ``emozilla/pg19`` (same content, 23 train shards), which loads natively.

Pipeline
--------
1. ``load_dataset('emozilla/pg19', split='train', streaming=True)`` — each
   example is one book (fields: short_book_title, publication_date, url, text).
2. Tokenize each book's ``text`` with the Llama-3 tokenizer
   (``add_special_tokens=False``).
3. Keep books with >= ``--min_tokens`` tokens (default 8192); drop short books.
   Stop after ``--max_books`` kept books (default 3000) to bound runtime.
4. Save as a HF DatasetDict matching ``dolmino_per_doc``: a single ``input_ids``
   column (int32, variable length, one book per row), with a train/validation
   split (``--val_frac`` default 0.05) to
   ``MemLong/data/processed/pg19_perbook_min8k``.
5. Print a length distribution (min/median/p90/max + >=8k/16k/32k/64k fractions).

Run on the local node (has transformers + datasets):
    export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=$http_proxy all_proxy=$http_proxy
    export no_proxy=.woa.com,localhost,127.0.0.1
    export HF_HOME=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/.hf_cache
    /opt/conda/envs/torch-base/bin/python scripts/preprocess_pg19_perbook.py
"""
from __future__ import annotations

import argparse
import os
import time
from typing import List

import numpy as np


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--hf_dataset",
        type=str,
        default="emozilla/pg19",
        help="HF dataset repo (parquet mirror of deepmind/pg19).",
    )
    p.add_argument(
        "--tokenizer_path",
        type=str,
        default="models/Meta-Llama-3-8B",
        help="Path to the Meta-Llama-3 tokenizer.",
    )
    p.add_argument(
        "--output_dir",
        type=str,
        default="MemLong/data/processed/pg19_perbook_min8k",
        help="Output directory for the per-book DatasetDict.",
    )
    p.add_argument(
        "--min_tokens",
        type=int,
        default=8192,
        help="Drop books shorter than this many Llama-3 tokens.",
    )
    p.add_argument(
        "--max_books",
        type=int,
        default=3000,
        help="Stop after keeping this many books (bounds runtime/size).",
    )
    p.add_argument(
        "--val_frac",
        type=float,
        default=0.05,
        help="Fraction of kept books held out for validation (book-level).",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import datasets
    from transformers import AutoTokenizer

    print(f"[pg19] loading Llama-3 tokenizer from {args.tokenizer_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, use_fast=True)
    print(f"[pg19] tokenizer vocab_size={tok.vocab_size}", flush=True)

    print(
        f"[pg19] streaming '{args.hf_dataset}' split=train "
        f"(keep books >= {args.min_tokens} tok, max_books={args.max_books})",
        flush=True,
    )
    stream = datasets.load_dataset(args.hf_dataset, split="train", streaming=True)

    docs: List[List[int]] = []
    lengths: List[int] = []
    n_seen = 0
    n_kept = 0
    t0 = time.time()

    for ex in stream:
        n_seen += 1
        text = ex.get("text") or ""
        if not text:
            continue
        ids = tok.encode(text, add_special_tokens=False)
        if len(ids) >= args.min_tokens:
            docs.append([int(x) for x in ids])
            lengths.append(len(ids))
            n_kept += 1
            if n_kept % 100 == 0:
                elapsed = time.time() - t0
                print(
                    f"[pg19] kept {n_kept}/{args.max_books} books "
                    f"(seen {n_seen}, {elapsed:.0f}s)",
                    flush=True,
                )
            if n_kept >= args.max_books:
                break

    print(
        f"[pg19] done streaming: seen {n_seen} books, kept {n_kept} "
        f"(>= {args.min_tokens} tok) in {time.time() - t0:.0f}s",
        flush=True,
    )

    if n_kept == 0:
        raise RuntimeError(
            "No books survived the min_tokens filter — check data / min_tokens."
        )

    lengths_np = np.asarray(lengths)
    total_tokens = int(lengths_np.sum())
    print("[pg19] ============ length statistics (tokens) ============", flush=True)
    print(f"[pg19] kept books: {n_kept}", flush=True)
    print(f"[pg19] total tokens: {total_tokens:,}", flush=True)
    print(
        "[pg19] min/median/mean/p90/p99/max = "
        f"{int(lengths_np.min())}/{int(np.median(lengths_np))}/"
        f"{lengths_np.mean():.0f}/{int(np.percentile(lengths_np, 90))}/"
        f"{int(np.percentile(lengths_np, 99))}/{int(lengths_np.max())}",
        flush=True,
    )
    for thr in (8192, 16384, 32768, 65536):
        frac = float((lengths_np >= thr).mean())
        print(
            f"[pg19]   books >= {thr:>6} tok: {frac * 100:5.1f}% "
            f"({int((lengths_np >= thr).sum())} books)",
            flush=True,
        )

    full = datasets.Dataset.from_dict({"input_ids": docs})
    # Book-level train/val split. shuffle only chooses WHICH books go to val;
    # token order within each book is untouched.
    split = full.train_test_split(
        test_size=args.val_frac, shuffle=True, seed=args.seed
    )
    ddict = datasets.DatasetDict(
        {"train": split["train"], "validation": split["test"]}
    )

    os.makedirs(os.path.dirname(args.output_dir) or ".", exist_ok=True)
    ddict.save_to_disk(args.output_dir)
    print(
        f"[pg19] wrote DatasetDict to {args.output_dir}: "
        f"train={len(ddict['train'])}, validation={len(ddict['validation'])}",
        flush=True,
    )


if __name__ == "__main__":
    main()
