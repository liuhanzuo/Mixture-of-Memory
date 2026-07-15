#!/usr/bin/env python
"""Training-free layer-depth retrieval probe for dense Qwen QCMem.

Natural SlimPajama windows are split into fixed chunks.  A RULER-style unique
key/value needle is inserted into one random chunk, and the query names the key.
For every requested depth j, this measures whether mean-pooled h_j retrieves the
needle chunk.  All depths share the same forward pass and examples.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.qcmem_qwen_jsweep import load_windows, parse_ints  # noqa: E402
from src.memory.qcmem import QCMemModel  # noqa: E402


@torch.inference_mode()
def encode_depths(qc, ids, depths):
    """Return {depth: L2-normalised mean-pooled vectors [batch, hidden]}."""
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    ids = ids.to(qc.device)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    if ids.dim() != 2:
        raise ValueError(f"expected [batch, tokens], got {tuple(ids.shape)}")
    ids = ids.long()
    hidden = qc.embed_tokens(ids)
    positions = torch.arange(ids.shape[1], device=qc.device).unsqueeze(0).expand(ids.shape[0], -1)
    mask, rope = qc._make_mask_and_rope(hidden, positions)
    want, out = set(depths), {}
    for li in range(max(depths)):
        hidden = qc.layers[li](
            hidden, attention_mask=mask, position_ids=positions,
            position_embeddings=rope, use_cache=False,
        )
        j = li + 1
        if j in want:
            vec = hidden.float().mean(dim=1)
            out[j] = torch.nn.functional.normalize(vec, dim=-1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Qwen3-32B")
    ap.add_argument(
        "--data_path",
        default="data/slimpajama-6b/data/validation-00000-of-00001-4fb685c22a3f91ef.parquet",
    )
    ap.add_argument("--depths", default="8,12,14,16,18,20,22,24,28,32")
    ap.add_argument("--topks", default="1,4,8")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--num_ctx", type=int, default=32)
    ap.add_argument("--samples", type=int, default=32)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(ROOT, args.model_path)
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(ROOT, args.data_path)
    device = torch.device(args.device)
    depths, topks = parse_ints(args.depths), parse_ints(args.topks)
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True,
        local_files_only=True, attn_implementation="sdpa",
    ).to(device).eval()
    model.config.use_cache = False
    depths = [j for j in depths if 1 <= j <= model.config.num_hidden_layers]
    qc = QCMemModel(model, resume_j=max(depths))
    windows = load_windows(
        data_path, tok, args.samples, args.num_ctx * args.chunk_size, args.seed
    )
    acc = {j: {"mrr": 0.0, **{f"hit@{k}": 0.0 for k in topks}} for j in depths}
    t0 = time.time()

    for si, window in enumerate(windows):
        ctx = window[0, :args.num_ctx * args.chunk_size].reshape(
            args.num_ctx, args.chunk_size
        ).clone()
        gold = rng.randrange(args.num_ctx)
        key = f"zircon-{rng.randrange(10**8, 10**9)}"
        value = str(rng.randrange(10**7, 10**8))
        needle = tok.encode(
            f" One of the special magic numbers for {key} is: {value}.",
            add_special_tokens=False,
        )
        needle = torch.tensor(needle[:args.chunk_size // 2], dtype=ctx.dtype)
        off = max(0, (args.chunk_size - len(needle)) // 2)
        ctx[gold, off:off + len(needle)] = needle
        query = tok.encode(
            f"What is the special magic number for {key}? Answer:",
            add_special_tokens=False, return_tensors="pt",
        )
        cv = encode_depths(qc, ctx.to(device), depths)
        qv = encode_depths(qc, query.to(device), depths)
        for j in depths:
            scores = cv[j] @ qv[j][0]
            order = scores.argsort(descending=True).tolist()
            rank = order.index(gold) + 1
            acc[j]["mrr"] += 1.0 / rank
            for k in topks:
                acc[j][f"hit@{k}"] += float(rank <= k)
        if (si + 1) % 8 == 0:
            print(f"[jprobe] {si+1}/{len(windows)} elapsed={time.time()-t0:.1f}s", flush=True)

    n = len(windows)
    rows = []
    for j in depths:
        row = {"j": j, **{k: round(v / n, 6) for k, v in acc[j].items()}}
        rows.append(row)
        print(row, flush=True)
    payload = {"model": model_path, "seed": args.seed, "samples": n,
               "chunk_size": args.chunk_size, "num_ctx": args.num_ctx,
               "topks": topks, "results": rows}
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"QWEN_RULER_JPROBE_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
