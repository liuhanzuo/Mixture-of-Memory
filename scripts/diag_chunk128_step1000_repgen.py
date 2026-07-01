#!/usr/bin/env python
"""Standalone diagnostic (general-purpose-35): does chunk128 step1000 gibberish
survive repetition_penalty / temperature sampling, or is it a greedy artefact?

Read-only: imports helpers from scripts/run_babilong_mem_space.py, does NOT
modify training or the eval harness. Runs on a single free GPU.
"""
from __future__ import annotations
import argparse, sys, os
from pathlib import Path
import torch

ROOT = "/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
sys.path.insert(0, os.path.join(ROOT, "third_party/babilong-pkg"))
sys.path.insert(0, ROOT)

import datasets
from transformers import AutoTokenizer
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input

import importlib.util
spec = importlib.util.spec_from_file_location(
    "rbm", os.path.join(ROOT, "scripts/run_babilong_mem_space.py"))
rbm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rbm)


def gen(model, input_ids, tok, chunk_size, max_new_tokens, device, mode):
    """mode: 'greedy' | 'reppen' | 'temp'."""
    rbm._reset_banks(model); rbm._reset_l2(model)
    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            _ = model(input_ids=chunk.unsqueeze(0).to(device), use_cache=False)
    rbm._freeze_banks(model)
    rep_pen = 1.3
    temp = 0.7
    try:
        cur = chunks[-1].unsqueeze(0).to(device)
        gen_ids = []
        for step in range(max_new_tokens):
            out = model(input_ids=cur, use_cache=False)
            logits = out.logits[:, -1, :].float()
            if step == 0 and tok.eos_token_id is not None:
                logits[:, tok.eos_token_id] = float("-inf")
            if mode == "reppen" and gen_ids:
                for t in set(gen_ids):
                    if logits[0, t] > 0:
                        logits[0, t] /= rep_pen
                    else:
                        logits[0, t] *= rep_pen
            if mode == "temp":
                probs = torch.softmax(logits / temp, dim=-1)
                nxt = torch.multinomial(probs, 1)
            else:
                nxt = logits.argmax(dim=-1, keepdim=True)
            tid = int(nxt.item())
            if tok.eos_token_id is not None and tid == tok.eos_token_id and step > 0:
                break
            gen_ids.append(tid)
            cur = torch.cat([cur, nxt], dim=-1)
    finally:
        rbm._unfreeze_banks(model)
    return tok.decode(gen_ids, skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--model_path", default=os.path.join(ROOT, "models/Meta-Llama-3-8B"))
    ap.add_argument("--chunk_size", type=int, default=128)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--task", default="qa1")
    ap.add_argument("--length", default="0k")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    import json
    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = rbm.build_mem_space_config(adapter_cfg)
    model = rbm.load_mem_space_model(
        model_path=args.model_path, checkpoint_path=args.checkpoint,
        mem_config=mem_config, device=device, dtype=dtype, attn_impl="sdpa")

    pc = {"instruction": DEFAULT_PROMPTS[args.task]["instruction"],
          "examples": DEFAULT_PROMPTS[args.task]["examples"],
          "post_prompt": DEFAULT_PROMPTS[args.task]["post_prompt"],
          "template": DEFAULT_TEMPLATE}
    data = datasets.load_dataset("RMT-team/babilong", args.length)
    td = data[args.task]
    torch.manual_seed(0)
    n_corr = {"greedy": 0, "reppen": 0, "temp": 0}
    for idx in range(args.n):
        s = td[idx]
        txt = get_formatted_input(s["input"], s["question"], pc["examples"],
                                  pc["instruction"], pc["post_prompt"], template=pc["template"])
        ids = tok.encode(txt, add_special_tokens=True, return_tensors="pt").to(device)
        tgt = s["target"]
        print(f"\n[{idx}] target={tgt!r}")
        for mode in ("greedy", "reppen", "temp"):
            with torch.amp.autocast(device_type="cuda", dtype=dtype):
                o = gen(model, ids, tok, args.chunk_size, 20, device, mode)
            hit = tgt.lower() in o.lower()
            n_corr[mode] += int(hit)
            print(f"    {mode:7s} hit={hit} out={o[:100]!r}")
    print("\n=== SUMMARY (n=%d %s/%s) ===" % (args.n, args.task, args.length))
    for m, c in n_corr.items():
        print(f"  {m:7s}: {c}/{args.n} contain target")


if __name__ == "__main__":
    main()
