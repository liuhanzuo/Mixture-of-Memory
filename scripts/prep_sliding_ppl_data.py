#!/usr/bin/env python3
"""Download + tokenize Proof-Pile / CodeParrot validation subsets for sliding-window PPL.

Produces a flat 1-D token stream .npy (Meta-Llama-3 tokenizer), matching the
``pg19_chunks_llama3_noeos.npy`` convention: documents are concatenated WITHOUT
inserting EOS/BOS between them (Llama-3's EOS 128001 is a hard boundary token;
sprinkling it inflates PPL ~100x — see scripts/tokenize_pg19_fast.py:40-44).

Datasets (validation subsets, not full):
  * proofpile  : EleutherAI/proof-pile-2 'open-web-math' (fallback hoskinson-center/proof-pile)
  * codeparrot : codeparrot/codeparrot-clean-valid

Requires HF proxy (set externally):
  export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=$http_proxy

Usage:
    python scripts/prep_sliding_ppl_data.py --dataset proofpile \
        --tokenizer models/Meta-Llama-3-8B --target_tokens 4000000 \
        --output data/proofpile_llama3_noeos.npy
"""
from __future__ import annotations

import argparse
import os
import time

import numpy as np


def load_texts(dataset: str, max_docs: int):
    """Return an iterator of text strings for the chosen dataset (streaming)."""
    from datasets import load_dataset

    if dataset == "proofpile":
        # Math/proof prose, low-redundancy LM target. proof-pile / proof-pile-2
        # are script-based (unsupported in datasets>=4). open-web-math is the
        # parquet-native standalone corpus = proof-pile-2's 'open-web-math' subset.
        candidates = [
            ("open-web-math/open-web-math", None, "train", "text"),
            ("EleutherAI/proof-pile-2", "open-web-math", "train", "text"),
            ("hoskinson-center/proof-pile", None, "train", "text"),
        ]
    elif dataset == "codeparrot":
        candidates = [
            ("codeparrot/codeparrot-clean-valid", None, "train", "content"),
            ("codeparrot/codeparrot-clean-valid", None, "validation", "content"),
        ]
    else:
        raise ValueError(dataset)

    last_err = None
    for name, config, split, field in candidates:
        try:
            print(f"[load] trying {name} config={config} split={split} field={field}")
            ds = load_dataset(name, config, split=split, streaming=True)
            texts = []
            for i, ex in enumerate(ds):
                if i >= max_docs:
                    break
                t = ex.get(field) or ex.get("text") or ex.get("content")
                if t and isinstance(t, str) and t.strip():
                    texts.append(t.strip())
            if texts:
                print(f"[load] OK {name}: {len(texts)} docs")
                return texts
        except Exception as e:  # noqa: BLE001
            print(f"[load] FAILED {name}: {e}")
            last_err = e
    raise RuntimeError(f"All candidates failed for {dataset}: {last_err}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["proofpile", "codeparrot"], required=True)
    ap.add_argument("--tokenizer", default="models/Meta-Llama-3-8B")
    ap.add_argument("--target_tokens", type=int, default=4_000_000,
                    help="Stop once this many tokens are collected.")
    ap.add_argument("--max_docs", type=int, default=20000)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    eff_vocab = max(len(tok), tok.vocab_size)
    dtype = np.uint16 if eff_vocab <= np.iinfo(np.uint16).max else np.uint32
    print(f"[tok] vocab={eff_vocab} -> {np.dtype(dtype).name}")

    t0 = time.time()
    texts = load_texts(args.dataset, args.max_docs)

    all_tokens: list[int] = []
    for i, text in enumerate(texts):
        ids = tok.encode(text, add_special_tokens=False)  # NO bos/eos
        all_tokens.extend(ids)
        if len(all_tokens) >= args.target_tokens:
            break
        if (i + 1) % 500 == 0:
            print(f"[tok] {i+1} docs, {len(all_tokens)} tokens ({time.time()-t0:.0f}s)")

    all_tokens = all_tokens[:args.target_tokens]
    arr = np.array(all_tokens, dtype=dtype)
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    np.save(args.output, arr)
    print(f"[save] {args.output}: {arr.shape} {arr.dtype} "
          f"({arr.nbytes/1e6:.1f} MB) in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
