#!/usr/bin/env python3
"""Pre-tokenize PG-19 into numpy chunks for fast training/eval.

Usage:
    python scripts/tokenize_pg19_fast.py \
        --input data/pg19_train.jsonl \
        --output data/pg19_chunks.npy \
        --seq_len 4096 \
        --max_lines 40000000
"""
import argparse, json, sys, os, time
from multiprocessing import Pool
import numpy as np

def tokenize_batch(args):
    """Tokenize a batch of lines."""
    lines, tokenizer_path, no_line_eos = args
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    eos = tokenizer.eos_token_id
    all_tokens = []
    for line in lines:
        text = line.strip()
        if text:
            tokens = tokenizer.encode(text, add_special_tokens=False)
            all_tokens.extend(tokens)
            if not no_line_eos:
                all_tokens.append(eos)
    return all_tokens

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seq_len", type=int, default=4096)
    parser.add_argument("--max_lines", type=int, default=30_000_000)
    parser.add_argument("--tokenizer", default="models/Llama--Llama2-7b")
    parser.add_argument("--workers", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=50000)
    parser.add_argument("--no_line_eos", action="store_true",
                        help="Do NOT insert tokenizer.eos_token_id between JSONL lines. "
                             "Llama-3's EOS (128001) is a hard document-boundary token, "
                             "and inserting ~260 per 4096-token chunk inflates PPL ~100x. "
                             "Set this for Llama-3; legacy pg19_chunks.npy was built without it.")
    args = parser.parse_args()

    t0 = time.time()
    print(f"Reading lines from {args.input}...")
    
    # Read lines (limit to max_lines)
    lines = []
    with open(args.input, 'r') as f:
        for i, line in enumerate(f):
            if i >= args.max_lines:
                break
            text = line.strip()
            if text:
                lines.append(text)
    print(f"Read {len(lines)} non-empty lines in {time.time()-t0:.1f}s")

    # Split into batches for parallel tokenization
    batches = []
    for i in range(0, len(lines), args.batch_size):
        batches.append((lines[i:i+args.batch_size], args.tokenizer, args.no_line_eos))
    
    print(f"Tokenizing {len(batches)} batches with {args.workers} workers...")
    t1 = time.time()
    
    with Pool(args.workers) as pool:
        results = pool.map(tokenize_batch, batches)
    
    # Flatten
    all_tokens = []
    for r in results:
        all_tokens.extend(r)
    
    print(f"Tokenized {len(all_tokens)} tokens in {time.time()-t1:.1f}s")
    
    # Split into chunks. Pick dtype based on tokenizer vocab: uint16 silently
    # truncates Llama-3 (vocab=128256), so we use uint32 whenever vocab > 65535.
    # We use `len(tokenizer)` (full vocab incl. added special tokens) rather
    # than `tokenizer.vocab_size` (BPE-only) so added specials (e.g. 128001 EOS)
    # never overflow.
    from transformers import AutoTokenizer
    _tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    vocab_size = max(len(_tok), _tok.vocab_size)
    dtype = np.uint16 if vocab_size <= np.iinfo(np.uint16).max else np.uint32
    print(f"Tokenizer effective vocab={vocab_size} -> storing as {np.dtype(dtype).name}")
    n_chunks = len(all_tokens) // args.seq_len
    all_tokens = all_tokens[:n_chunks * args.seq_len]
    arr = np.array(all_tokens, dtype=dtype).reshape(n_chunks, args.seq_len)
    
    print(f"Created {n_chunks} chunks of {args.seq_len} tokens")
    print(f"Array shape: {arr.shape}, dtype: {arr.dtype}, size: {arr.nbytes / 1e9:.2f} GB")
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else '.', exist_ok=True)
    np.save(args.output, arr)
    print(f"Saved to {args.output} in {time.time()-t0:.1f}s total")

if __name__ == "__main__":
    main()
