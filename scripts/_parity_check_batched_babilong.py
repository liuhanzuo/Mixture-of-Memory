"""Throwaway parity harness: bsz=1 vs batched generate_with_mem_space.

Loads a mem_space ckpt ONCE, runs the same BABILong cell through both the
per-sample (bsz=1) path and the batched path, and reports per-sample output
equality + wall-clock. Not committed for production use.

Usage:
  python scripts/_parity_check_batched_babilong.py \
    --model_path models/Meta-Llama-3-8B \
    --checkpoint outputs/mem_space_p11_chunk512_deltarule_normreadout/mem_space_adapter.pt \
    --adapter_config outputs/mem_space_p11_chunk512_deltarule_normreadout/adapter_config.json \
    --chunk_size 512 --task qa1 --length 1k --limit 16 --batch_size 8
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BAB = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BAB) and _BAB not in sys.path:
    sys.path.insert(0, _BAB)

import datasets  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402

import scripts.run_babilong_mem_space as R  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapter_config", required=True)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--task", default="qa1")
    p.add_argument("--length", default="1k")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=20)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--dataset_name", default="RMT-team/babilong")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = R.build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size

    model = R.load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl="sdpa",
    )

    t = args.task
    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[t]["instruction"],
        "examples": DEFAULT_PROMPTS[t]["examples"],
        "post_prompt": DEFAULT_PROMPTS[t]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
        "chat_template": False,
        "system_prompt": "",
    }
    data = datasets.load_dataset(args.dataset_name, args.length)
    task_data = data[t]
    n = min(args.limit, len(task_data))

    samples = []
    for i in range(n):
        s = task_data[i]
        text = get_formatted_input(
            s["input"], s["question"], prompt_cfg["examples"],
            prompt_cfg["instruction"], prompt_cfg["post_prompt"],
            template=prompt_cfg["template"],
        )
        ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")[0]
        samples.append((s["target"], s["question"], ids))

    import math
    ncs = [max(1, math.ceil(s[2].shape[0] / args.chunk_size)) for s in samples]
    print(f"[parity] {n} samples; chunk counts: min={min(ncs)} max={max(ncs)} "
          f"(multi-chunk={sum(c>1 for c in ncs)}, single={sum(c<=1 for c in ncs)})")

    # ---- bsz=1 reference ----
    out_bsz1 = []
    t0 = time.time()
    for (_tgt, _q, ids) in samples:
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            o = R.generate_with_mem_space(
                model=model, input_ids=ids.unsqueeze(0).to(device),
                tokenizer=tok, chunk_size=args.chunk_size,
                max_new_tokens=args.max_new_tokens, device=device,
            )
        out_bsz1.append(o)
    dt1 = time.time() - t0

    # ---- batched ----
    out_bN = [None] * n
    t0 = time.time()
    multi_idx = [i for i in range(n) if ncs[i] > 1]
    single_idx = [i for i in range(n) if ncs[i] <= 1]
    for i in single_idx:
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            out_bN[i] = R.generate_with_mem_space(
                model=model, input_ids=samples[i][2].unsqueeze(0).to(device),
                tokenizer=tok, chunk_size=args.chunk_size,
                max_new_tokens=args.max_new_tokens, device=device,
            )
    from collections import defaultdict
    by_nc = defaultdict(list)
    for i in multi_idx:
        by_nc[ncs[i]].append(i)
    for nc, idxs in by_nc.items():
        for s in range(0, len(idxs), args.batch_size):
            chunk = idxs[s:s + args.batch_size]
            tl = [samples[i][2] for i in chunk]
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                outs = R.generate_batch_with_mem_space(
                    model=model, token_list=tl, tokenizer=tok,
                    chunk_size=args.chunk_size,
                    max_new_tokens=args.max_new_tokens, device=device,
                )
            for i, o in zip(chunk, outs):
                out_bN[i] = o
    dtN = time.time() - t0

    n_match = 0
    print("\n[parity] per-sample comparison (idx | nc | match | bsz1 | batched):")
    for i in range(n):
        m = out_bsz1[i] == out_bN[i]
        n_match += int(m)
        flag = "OK " if m else "DIFF"
        print(f"  {i:3d} nc={ncs[i]} {flag} | {out_bsz1[i]!r} | {out_bN[i]!r}")
    print(f"\n[parity] exact-match {n_match}/{n}")

    # ---- BABILong score parity (the metric that actually matters) ----
    try:
        from babilong.metrics import compare_answers, TASK_LABELS
        labels = TASK_LABELS[t]
        score1 = 0
        scoreN = 0
        flip = 0
        for i in range(n):
            tgt, q = samples[i][0], samples[i][1]
            c1 = compare_answers(tgt, out_bsz1[i], q, labels)
            cN = compare_answers(tgt, out_bN[i], q, labels)
            score1 += int(c1)
            scoreN += int(cN)
            flip += int(c1 != cN)
        print(f"[parity] BABILong score  bsz1={score1}/{n} ({100*score1/n:.1f}%)  "
              f"batched={scoreN}/{n} ({100*scoreN/n:.1f}%)  judgment-flips={flip}")
    except Exception as e:
        print(f"[parity] score comparison skipped: {e}")

    print(f"[parity] wall-clock bsz1={dt1:.1f}s  batched(bs={args.batch_size})={dtN:.1f}s  "
          f"speedup={dt1/max(dtN,1e-6):.2f}x")


if __name__ == "__main__":
    main()
