"""Pre-tokenize pg19_train.jsonl into .npy for fast DataLoader."""
import json
import numpy as np
import sys
import time
from transformers import AutoTokenizer

DATA_PATH = "data/pg19_train.jsonl"
OUT_PATH = "data/pg19_train_4k.npy"
MODEL_PATH = "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Llama--Llama2-7b"
SEQ_LEN = 4096

def main():
    t0 = time.time()
    print(f"Loading tokenizer from {MODEL_PATH}...", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"Tokenizer loaded (vocab_size={tokenizer.vocab_size})", flush=True)

    all_ids = []
    count = 0
    REPORT = 500000

    print(f"Reading {DATA_PATH}...", flush=True)
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ids = tokenizer.encode(line, add_special_tokens=False)
            all_ids.extend(ids)
            count += 1
            if count % REPORT == 0:
                elapsed = time.time() - t0
                print(f"  {count:,} lines, {len(all_ids):,} tokens, {elapsed:.0f}s", flush=True)

    elapsed = time.time() - t0
    print(f"Done reading: {count:,} lines → {len(all_ids):,} tokens in {elapsed:.0f}s", flush=True)

    # Split into chunks
    chunk_len = SEQ_LEN + 1  # +1 for causal shift
    chunks = []
    for i in range(0, len(all_ids) - chunk_len + 1, SEQ_LEN):
        chunks.append(all_ids[i:i + chunk_len])

    arr = np.array(chunks, dtype=np.int32)
    np.save(OUT_PATH, arr)
    print(f"Saved {len(chunks):,} chunks ({SEQ_LEN}+1 tokens each) → {OUT_PATH}", flush=True)
    print(f"File size: {arr.nbytes / 1e9:.2f} GB", flush=True)

if __name__ == "__main__":
    main()
