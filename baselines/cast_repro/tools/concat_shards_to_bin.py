"""
concat_shards_to_bin.py
-----------------------
Concatenate shard_XXXX.npy tokenise shards into a single flat token train.bin
that train_cast_llama.py::BinDataset expects:

    np.memmap(path, dtype=np.dtype(dtype), mode="r")   # raw binary, no header

NOTE: despite the .npy extension the shards from download_and_tokenize_dolmino.py
are plain raw binary files (NOT numpy .npy format). They store tokens as uint32
(4 bytes per token) for the dolmino-mix-1124-llama2 dataset. The metadata.json
"dtype" field records this; train_cast_llama.py auto-reads it via
--data-dtype=auto.

Usage:
    python concat_shards_to_bin.py \
        --shards-dir data/dolmino-mix-1124-llama2/ \
        [--out      data/dolmino-mix-1124-llama2/train.bin] \
        [--val-out  data/dolmino-mix-1124-llama2/val.bin]   \
        [--dtype    uint32]    # default: auto from metadata.json  \
        [--limit-tokens N]     # for smoke testing
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--shards-dir", required=True, type=Path,
                   help="Directory containing shard_XXXX.npy + metadata.json")
    p.add_argument("--out", type=Path, default=None,
                   help="Output train.bin path (default: <shards-dir>/train.bin)")
    p.add_argument("--val-out", type=Path, default=None,
                   help="Output val.bin path (default: <shards-dir>/val.bin); "
                        "written from the LAST shard only")
    p.add_argument("--dtype", default=None,
                   help="Token dtype (default: auto from metadata.json, "
                        "typically uint32 for dolmino-mix-1124-llama2)")
    p.add_argument("--limit-tokens", type=int, default=None,
                   help="Stop after writing this many tokens (for quick smoke test)")
    return p.parse_args()


def main():
    args = parse_args()
    shards_dir = args.shards_dir.resolve()
    out_path   = args.out     or shards_dir / "train.bin"
    val_path   = args.val_out or shards_dir / "val.bin"

    # ---- load metadata ----
    meta_file = shards_dir / "metadata.json"
    if not meta_file.exists():
        print(f"ERROR: metadata.json not found in {shards_dir}", file=sys.stderr)
        sys.exit(1)

    with open(meta_file) as f:
        meta = json.load(f)

    total_tokens_meta: int = meta["total_tokens"]
    num_shards: int        = meta["num_shards"]
    shards_list: list      = meta["shards"]   # [{shard_idx, file, num_tokens}, ...]

    # Auto-resolve dtype from metadata unless explicitly overridden
    if args.dtype is None:
        dtype_str = meta.get("dtype", "uint32")
        print(f"[dtype auto] metadata.dtype = {dtype_str!r}")
    else:
        dtype_str = args.dtype
    dtype    = np.dtype(dtype_str)
    itemsize = dtype.itemsize

    # Sort by shard_idx to guarantee order
    shards_list = sorted(shards_list, key=lambda s: s["shard_idx"])
    assert len(shards_list) == num_shards, (
        f"metadata.num_shards={num_shards} but len(shards)={len(shards_list)}"
    )

    # ---- determine how many tokens to write ----
    if args.limit_tokens is not None:
        target_tokens = min(args.limit_tokens, total_tokens_meta)
        print(f"[limit-tokens] will write at most {target_tokens:,} tokens")
    else:
        target_tokens = total_tokens_meta

    print(f"shards dir  : {shards_dir}")
    print(f"out train   : {out_path}")
    print(f"out val     : {val_path}  (last shard only)")
    print(f"total tokens: {total_tokens_meta:,}  (from metadata)")
    print(f"target write: {target_tokens:,}")
    print(f"dtype       : {dtype}  itemsize={itemsize}")
    print(f"num shards  : {num_shards}")
    print()

    # ---- open train.bin memmap ----
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Creating {out_path}  ({target_tokens * itemsize / 1e9:.2f} GB) ...")
    train_mm = np.memmap(str(out_path), dtype=dtype, mode="w+", shape=(target_tokens,))

    offset  = 0
    t_start = time.time()

    for i, shard_info in enumerate(shards_list):
        if offset >= target_tokens:
            break

        shard_file             = shards_dir / shard_info["file"]
        shard_tokens_expected  = shard_info["num_tokens"]

        if not shard_file.exists():
            print(f"ERROR: shard file not found: {shard_file}", file=sys.stderr)
            sys.exit(1)

        # Shards are raw binary (NOT numpy .npy), read via memmap
        shard_mm     = np.memmap(str(shard_file), dtype=dtype, mode="r")
        actual_tokens = len(shard_mm)
        if actual_tokens != shard_tokens_expected:
            print(f"WARNING: shard {i} has {actual_tokens} tokens, "
                  f"metadata says {shard_tokens_expected}", file=sys.stderr)

        # How many tokens from this shard fit in the remaining budget?
        can_write = min(actual_tokens, target_tokens - offset)

        train_mm[offset : offset + can_write] = shard_mm[:can_write]
        del shard_mm  # release memmap handle
        offset += can_write

        if (i + 1) % 50 == 0 or i == num_shards - 1 or offset == target_tokens:
            elapsed = time.time() - t_start
            rate    = offset / elapsed / 1e6 if elapsed > 0 else 0
            pct     = 100.0 * offset / target_tokens
            print(f"  shard {i+1:4d}/{num_shards}  "
                  f"offset={offset:,}  {pct:.1f}%  {rate:.1f} M tok/s  "
                  f"elapsed={elapsed:.0f}s")
            train_mm.flush()

    assert offset == target_tokens, f"wrote {offset} != expected {target_tokens}"
    train_mm.flush()
    del train_mm  # close memmap

    elapsed_total = time.time() - t_start
    size_bytes    = out_path.stat().st_size
    print(f"\ntrain.bin written: {size_bytes:,} bytes ({size_bytes / 1e9:.3f} GB) "
          f"in {elapsed_total:.1f}s")

    # ---- sha256 spot-check: first + last 4 KiB ----
    with open(out_path, "rb") as fh:
        first4k = fh.read(4096)
        fh.seek(max(0, size_bytes - 4096))
        last4k  = fh.read(4096)
    print(f"sha256(first 4KB): {sha256_of_bytes(first4k)}")
    print(f"sha256(last  4KB): {sha256_of_bytes(last4k)}")
    print(f"token count check: {offset:,} tokens written  "
          f"(metadata total={total_tokens_meta:,})")

    # ---- write val.bin from the LAST shard ----
    if args.limit_tokens is None:
        last_shard_info = shards_list[-1]
        last_file       = shards_dir / last_shard_info["file"]
        val_n           = last_shard_info["num_tokens"]
        print(f"\nWriting val.bin from {last_file.name}  ({val_n:,} tokens) ...")
        val_path.parent.mkdir(parents=True, exist_ok=True)
        val_mm  = np.memmap(str(val_path), dtype=dtype, mode="w+", shape=(val_n,))
        src_mm  = np.memmap(str(last_file), dtype=dtype, mode="r")
        val_mm[:] = src_mm[:val_n]
        val_mm.flush()
        del val_mm, src_mm
        val_size = val_path.stat().st_size
        with open(val_path, "rb") as fh:
            val_first4k = fh.read(4096)
        print(f"val.bin written : {val_size:,} bytes  "
              f"sha256(first 4KB): {sha256_of_bytes(val_first4k)}")
    else:
        print("\n[limit-tokens active] skipping val.bin write")

    print("\nDONE.")


if __name__ == "__main__":
    main()
