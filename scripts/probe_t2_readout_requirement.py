#!/usr/bin/env python3
"""T2 task-validity control: does answering REQUIRE cross-chunk readout?

team-lead's go/no-go before any retrain: if the answer can be predicted WITHOUT
the memory readout (in-window shortcut / leak), then answer-gradient never trains
the readout (loss->0 is spurious) and any retrain is wasted.

Test: build T2 chunked associative-recall samples (NIAHChunkedDataset, the exact
training data), stream the n_ctx context chunks into the memory store, then compute
the answer-digit NLL on the target chunk under two conditions:
  * memory ON  (readout active): the model CAN address chunk0's stored needle.
  * memory OFF (_memory_disabled = vanilla Llama, no readout): the model can ONLY
    use the target chunk's own tokens (question), the needle is invisible.

Interpretation:
  * OFF_nll >> ON_nll  (e.g. OFF~2.3 random-digit, ON~0) -> the answer REQUIRES
    readout. Task is valid; loss->0 in training = genuine readout learning.
  * OFF_nll ~= ON_nll ~= 0 -> SHORTCUT/LEAK: answer predictable without readout
    (needle leaked into target window, or copied). Task is invalid -> fix before
    retrain (otherwise readout never gets gradient).

Also reports ON-with-grouped vs ON-flat to check the readout path used.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from transformers import AutoTokenizer  # noqa: E402
from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model  # noqa: E402
from src.memory.mem_space.niah_chunked_dataset import NIAHChunkedDataset  # noqa: E402


def _set_memory_disabled(model, flag):
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []):
        w._memory_disabled = flag


def _reset_banks(model):
    root = getattr(model, "module", model)
    b = getattr(root, "_mem_space_shared_bank", None)
    targets = [b] if b is not None else [
        getattr(w, "memory_bank", None) for w in getattr(root, "_mem_space_layers", [])
    ]
    for bk in targets:
        if bk is None:
            continue
        if hasattr(bk, "reset"):
            bk.reset()
        object.__setattr__(bk, "_rawkv_readout_store", None)
        bk.frozen = False


def _freeze(model):
    root = getattr(model, "module", model)
    b = getattr(root, "_mem_space_shared_bank", None)
    targets = [b] if b is not None else [
        getattr(w, "memory_bank", None) for w in getattr(root, "_mem_space_layers", [])
    ]
    for bk in targets:
        if bk is not None:
            bk.frozen = True


def _answer_nll(model, ctx_chunks, target_ids, answer_mask, device, memory_off):
    """Stream context chunks (memory ON), then compute mean NLL on answer-digit
    positions of the target chunk. If memory_off, disable readout for BOTH the
    streaming and the target forward (vanilla Llama, needle invisible)."""
    _reset_banks(model)
    if memory_off:
        _set_memory_disabled(model, True)
    try:
        with torch.no_grad():
            # stream context (writes to store unless memory_off)
            for ch in ctx_chunks:
                model(input_ids=ch.unsqueeze(0).to(device), use_cache=False)
            _freeze(model)
            out = model(input_ids=target_ids.unsqueeze(0).to(device), use_cache=False)
            logits = out.logits if hasattr(out, "logits") else out[0]
            # next-token prediction: logits[t] predicts target[t+1]
            lp = F.log_softmax(logits[0].float(), dim=-1)
            nlls = []
            am = answer_mask.tolist()
            tgt = target_ids.tolist()
            for t in range(len(tgt) - 1):
                if am[t + 1]:  # position t predicts the answer-digit at t+1
                    nlls.append(-lp[t, tgt[t + 1]].item())
            return float(np.mean(nlls)) if nlls else float("nan")
    finally:
        if memory_off:
            _set_memory_disabled(model, False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--gap_tokens", type=int, default=8192)
    ap.add_argument("--num_keys", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--grouped", action="store_true", help="eval with grouped readout ON")
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    mc.l3_recon_max_positions = cli.chunk_size
    mc.rawkv_disable_col_bias = True
    mc.rawkv_readout_topk_chunks = 0  # keep_all (training condition)
    if cli.grouped:
        mc.rawkv_grouped_readout = True
        mc.rawkv_subblock_size = 64
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()

    bg = np.load(cli.background)
    ds = NIAHChunkedDataset(
        background_data=bg, chunk_size=cli.chunk_size, gap_tokens=cli.gap_tokens,
        tokenizer=tok, num_keys=cli.num_keys, seed=12345, background_skip=5000,
    )
    it = iter(ds)

    on_nlls, off_nlls = [], []
    on_correct, off_correct = 0, 0
    n = 0
    for _ in range(cli.n_samples):
        s = next(it)
        ctx = s["context_chunks"]; tgt = s["target_ids"]; am = s["answer_mask"]
        on = _answer_nll(model, ctx, tgt, am, device, memory_off=False)
        off = _answer_nll(model, ctx, tgt, am, device, memory_off=True)
        if not (np.isnan(on) or np.isnan(off)):
            on_nlls.append(on); off_nlls.append(off)
            on_correct += int(on < 0.05)   # ~exact digit
            off_correct += int(off < 0.05)
            n += 1

    print("\n==== T2 readout-REQUIREMENT control (does answer need readout?) ====")
    print(f"n={n} chunk_size={cli.chunk_size} gap={cli.gap_tokens} n_ctx={ds.n_ctx} "
          f"num_keys={cli.num_keys} grouped={cli.grouped} keep_all=True")
    print(f"answer-digit NLL  memory ON  (readout): {np.mean(on_nlls):.4f}")
    print(f"answer-digit NLL  memory OFF (no readout): {np.mean(off_nlls):.4f}")
    print(f"  (random-digit baseline NLL ~= ln(10) = 2.303)")
    print(f"near-exact frac   ON: {100.0*on_correct/max(n,1):.1f}%   OFF: {100.0*off_correct/max(n,1):.1f}%")
    print("\nVERDICT:")
    if np.mean(off_nlls) - np.mean(on_nlls) > 1.0:
        print("  ✓ OFF >> ON: answer REQUIRES readout -> T2 task is VALID (loss->0 = real learning)")
    elif np.mean(off_nlls) < 0.5:
        print("  ✗ OFF also low: SHORTCUT/LEAK -> answer predictable WITHOUT readout -> task INVALID, fix before retrain")
    else:
        print("  ~ partial: OFF elevated but not random -> some readout dependence, inspect")


if __name__ == "__main__":
    main()
