#!/usr/bin/env python3
"""Preprocess SlimPajama-6B into (N, CHUNK_SIZE) uint16 npy arrays.

Usage:
    python scripts/preprocess_slimpajama.py \
        --input_dir data/slimpajama-6b \
        --output data/slimpajama_chunks_4096.npy \
        --tokenizer meta-llama/Llama-2-7b-hf \
        --chunk_size 4096

Input: SlimPajama-6B JSONL files (each line has "text" field)
Output: (N, chunk_size) uint16 numpy array saved as .npy
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np

try:
    import pyarrow.parquet as pq
    HAS_PARQUET = True
except ImportError:
    HAS_PARQUET = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--chunk_size", type=int, default=4096)
    args = parser.parse_args()

    from transformers import AutoTokenizer

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, use_fast=True)
    eos_id = tokenizer.eos_token_id

    # Find data files (parquet or JSONL)
    parquet_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"), recursive=True))
    jsonl_files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.jsonl*"), recursive=True))
    files = parquet_files + jsonl_files
    if not files:
        print(f"No data files found in {args.input_dir}")
        sys.exit(1)

    print(f"Found {len(files)} files ({len(parquet_files)} parquet, {len(jsonl_files)} jsonl)")

    all_ids = []
    total_docs = 0
    total_tokens = 0
    start = time.time()

    for fi, fpath in enumerate(files):
        t0 = time.time()

        if fpath.endswith(".parquet"):
            if not HAS_PARQUET:
                print(f"  Skipping {fpath}: pyarrow not installed")
                continue
            pf = pq.read_table(fpath, columns=["text"])
            texts = pf["text"].to_pylist()
            for text in texts:
                if not text or not isinstance(text, str):
                    continue
                ids = tokenizer.encode(text, add_special_tokens=False)
                all_ids.extend(ids)
                total_docs += 1
                total_tokens += len(ids)
        else:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    doc = json.loads(line)
                    text = doc.get("text", "")
                    if not text:
                        continue
                    ids = tokenizer.encode(text, add_special_tokens=False)
                    all_ids.extend(ids)
                    total_docs += 1
                    total_tokens += len(ids)

        elapsed = time.time() - t0
        print(
            f"  [{fi+1}/{len(files)}] {os.path.basename(fpath)}: "
            f"{total_tokens:,} tokens, {elapsed:.1f}s",
            flush=True,
        )

    # Pad to multiple of chunk_size
    chunk_size = args.chunk_size
    pad = (chunk_size - len(all_ids) % chunk_size) % chunk_size
    if pad > 0:
        all_ids.extend([eos_id] * pad)

    arr = np.array(all_ids, dtype=np.uint16).reshape(-1, chunk_size)
    print(f"\nTotal: {total_docs:,} docs, {total_tokens:,} tokens → {arr.shape} array")
    print(f"Output: {args.output} ({arr.nbytes / 1e9:.1f} GB)")
    print(f"Time: {time.time() - start:.1f}s")

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, arr)
    print("Done!")


if __name__ == "__main__":
    main()
