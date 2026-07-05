#!/usr/bin/env python3
"""Preprocess SlimPajama-6B into (N, CHUNK_SIZE) uint32 npy arrays.

Tokenizes with the **Llama-3** tokenizer (vocab 128256) and stores token ids as
**uint32** — Llama-3 ids reach 128255 which overflows uint16, so uint16 (the old
default) silently corrupted the data. Output format matches
``data/pg19_chunks_llama3.npy`` (uint32, (N, chunk_size), Llama-3 tok, EOS used as
inter-document separator, no BOS).

Memory-safe: token ids are streamed to a temporary raw ``.bin`` file as they are
produced (never accumulating the whole corpus in RAM), then reshaped into the
final ``.npy`` via a block copy through ``np.lib.format.open_memmap``.

Parallel: documents are tokenized with a ``multiprocessing.Pool`` (default: half
the cores) so we don't starve any concurrent GPU training of CPU.

Usage:
    python scripts/preprocess_slimpajama.py \
        --input_dir data/slimpajama-6b/data \
        --output data/slimpajama_chunks_4096_llama3.npy \
        --val_output data/slimpajama_val_4096_llama3.npy \
        --tokenizer models/Meta-Llama-3-8B \
        --chunk_size 4096 \
        --num_train_shards 12
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


# ----------------------------------------------------------------------------
# Worker: each process loads the tokenizer once, then tokenizes batches of docs.
# ----------------------------------------------------------------------------
_TOK = None
_EOS = None


def _init_worker(tokenizer_path):
    global _TOK, _EOS
    # Silence the per-process transformers chatter.
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    from transformers import AutoTokenizer
    _TOK = AutoTokenizer.from_pretrained(
        tokenizer_path, local_files_only=True, use_fast=True
    )
    _EOS = _TOK.eos_token_id


def _encode_batch(texts):
    """Tokenize a list of texts, append EOS after each doc, return a flat uint32 array.

    Returns (arr, n_docs, n_tokens_without_eos).
    """
    global _TOK, _EOS
    # Fast batch encode; add_special_tokens=False -> no BOS/EOS injected by tok.
    enc = _TOK(texts, add_special_tokens=False)["input_ids"]
    pieces = []
    n_docs = 0
    n_tokens = 0
    for ids in enc:
        if not ids:
            continue
        n_docs += 1
        n_tokens += len(ids)
        pieces.append(np.asarray(ids, dtype=np.uint32))
        pieces.append(np.asarray([_EOS], dtype=np.uint32))
    if not pieces:
        return np.empty(0, dtype=np.uint32), 0, 0
    return np.concatenate(pieces), n_docs, n_tokens


def _iter_doc_batches(files, batch_size):
    """Yield lists of text strings (batch_size docs each) from parquet/jsonl files."""
    buf = []
    for fpath in files:
        if fpath.endswith(".parquet"):
            if not HAS_PARQUET:
                print(f"  Skipping {fpath}: pyarrow not installed", flush=True)
                continue
            table = pq.read_table(fpath, columns=["text"])
            texts = table["text"].to_pylist()
            del table
        else:
            texts = []
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        doc = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    texts.append(doc.get("text", ""))
        for text in texts:
            if not text or not isinstance(text, str):
                continue
            buf.append(text)
            if len(buf) >= batch_size:
                yield buf
                buf = []
    if buf:
        yield buf


def tokenize_to_npy(files, output, tokenizer_path, chunk_size, eos_id,
                    num_proc, batch_size, tag):
    """Stream-tokenize `files` into `output` (N, chunk_size) uint32 npy."""
    import multiprocessing as mp

    tmp_bin = output + ".rawtokens.tmp"
    start = time.time()
    total_docs = 0
    total_tokens = 0  # content tokens (excludes EOS separators)
    total_written = 0  # includes EOS separators

    print(f"[{tag}] tokenizing {len(files)} file(s) with {num_proc} procs "
          f"(batch_size={batch_size}) -> {output}", flush=True)

    with open(tmp_bin, "wb") as fout:
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=num_proc, initializer=_init_worker,
                      initargs=(tokenizer_path,)) as pool:
            batches = _iter_doc_batches(files, batch_size)
            # imap keeps memory bounded; chunksize=1 since each item is already a batch.
            for arr, n_docs, n_tok in pool.imap(_encode_batch, batches, chunksize=1):
                if arr.size:
                    arr.tofile(fout)
                    total_written += arr.size
                total_docs += n_docs
                total_tokens += n_tok
                if total_docs and (total_docs % 100000 < batch_size):
                    el = time.time() - start
                    rate = total_tokens / el if el > 0 else 0
                    print(f"  [{tag}] {total_docs:,} docs, {total_tokens:,} tokens, "
                          f"{el:.0f}s, {rate/1e6:.2f}M tok/s", flush=True)

    # Pad the raw token stream up to a multiple of chunk_size with EOS.
    pad = (chunk_size - total_written % chunk_size) % chunk_size
    if pad > 0:
        with open(tmp_bin, "ab") as fout:
            np.full(pad, eos_id, dtype=np.uint32).tofile(fout)
        total_written += pad

    n_rows = total_written // chunk_size
    assert n_rows > 0, f"[{tag}] no full chunks produced (total_written={total_written})"

    # Block-copy the raw stream into the final (N, chunk_size) npy via memmap.
    print(f"[{tag}] reshaping {total_written:,} tokens -> ({n_rows}, {chunk_size}) uint32",
          flush=True)
    out_mm = np.lib.format.open_memmap(
        output, mode="w+", dtype=np.uint32, shape=(n_rows, chunk_size)
    )
    src = np.memmap(tmp_bin, dtype=np.uint32, mode="r", shape=(n_rows * chunk_size,))
    rows_per_block = max(1, (256 * 1024 * 1024) // (chunk_size * 4))  # ~256MB blocks
    for r0 in range(0, n_rows, rows_per_block):
        r1 = min(n_rows, r0 + rows_per_block)
        out_mm[r0:r1] = src[r0 * chunk_size:r1 * chunk_size].reshape(r1 - r0, chunk_size)
    out_mm.flush()
    del out_mm, src
    os.remove(tmp_bin)

    el = time.time() - start
    print(f"[{tag}] DONE: {total_docs:,} docs, {total_tokens:,} content tokens, "
          f"shape ({n_rows}, {chunk_size}), {n_rows * chunk_size * 4 / 1e9:.2f} GB, "
          f"{el:.0f}s", flush=True)
    return n_rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="data/slimpajama-6b/data")
    parser.add_argument("--output", type=str,
                        default="data/slimpajama_chunks_4096_llama3.npy")
    parser.add_argument("--val_output", type=str,
                        default="data/slimpajama_val_4096_llama3.npy")
    parser.add_argument("--tokenizer", type=str, default="models/Meta-Llama-3-8B")
    parser.add_argument("--chunk_size", type=int, default=4096)
    parser.add_argument("--num_train_shards", type=int, default=12,
                        help="Number of train-*.parquet shards to tokenize (0 = all).")
    parser.add_argument("--num_proc", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--batch_size", type=int, default=1000,
                        help="Docs per tokenization task.")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_val", action="store_true")
    args = parser.parse_args()

    # eos id from the (Llama-3) tokenizer, loaded locally.
    from transformers import AutoTokenizer
    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(
        args.tokenizer, local_files_only=True, use_fast=True
    )
    eos_id = tok.eos_token_id
    print(f"  vocab_size={tok.vocab_size} len(tok)={len(tok)} eos_id={eos_id}")
    assert eos_id is not None and eos_id < 2**32
    del tok  # workers load their own copies

    all_parquet = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.parquet"),
                                   recursive=True))
    train_files = [f for f in all_parquet if os.path.basename(f).startswith("train-")]
    val_files = [f for f in all_parquet
                 if os.path.basename(f).startswith("validation-")]
    if not all_parquet:
        # jsonl fallback
        jsonl = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.jsonl*"),
                                 recursive=True))
        train_files = [f for f in jsonl if "train" in os.path.basename(f).lower()]
        val_files = [f for f in jsonl if "val" in os.path.basename(f).lower()]

    if args.num_train_shards > 0:
        train_files = train_files[:args.num_train_shards]

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    if not args.skip_train:
        if not train_files:
            print("No train files found!", file=sys.stderr)
            sys.exit(1)
        print(f"\n=== TRAIN: {len(train_files)} shard(s) ===")
        for f in train_files:
            print(f"  {os.path.basename(f)}")
        tokenize_to_npy(train_files, args.output, args.tokenizer,
                        args.chunk_size, eos_id, args.num_proc, args.batch_size,
                        tag="train")

    if not args.skip_val:
        if not val_files:
            print("No validation files found; skipping val.", file=sys.stderr)
        else:
            print(f"\n=== VAL: {len(val_files)} file(s) ===")
            tokenize_to_npy(val_files, args.val_output, args.tokenizer,
                            args.chunk_size, eos_id, args.num_proc, args.batch_size,
                            tag="val")

    print("\nAll done!")


if __name__ == "__main__":
    main()
