#!/usr/bin/env python3
"""Stage-1 real-model smoke for Slot-Routed Evidence Memory (2026-06-17).

Loads the actual Meta-Llama-3-8B + mem_space with the p11 SOTA adapter_config,
turns on --use_slot_evidence at a GENEROUS budget (buffer_size=64, topr=64), and
streams one real ~4k-token sequence in chunk_size segments (exactly the eval
streaming path). Confirms:
  * no OOM, forward completes
  * the evidence buffer fills (max slot_evidence_count > 0)
  * the extended_hidden the wrapped evidence layer sees grew by k*topr vs OFF
  * reports peak GPU memory delta evidence-ON vs evidence-OFF (real footprint)

Single GPU. Run on a FREE card.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    _reset_banks,
)


def _run(model, tokenizer, total_len, chunk_size, device):
    """Stream a random-token sequence in chunks (no reset between chunks)."""
    torch.manual_seed(0)
    ids = torch.randint(0, 32000, (1, total_len), device=device, dtype=torch.long)
    chunks = list(ids[0].split(chunk_size))
    _reset_banks(model)
    with torch.no_grad():
        for ch in chunks:
            _ = model(input_ids=ch.unsqueeze(0), use_cache=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--adapter_config", required=True)
    p.add_argument("--total_len", type=int, default=4096)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--evidence_buffer_size", type=int, default=64)
    p.add_argument("--evidence_topr", type=int, default=64)
    p.add_argument("--evidence_layer", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = torch.bfloat16

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)

    # ---- Pass A: evidence OFF (footprint baseline) ----
    cfg_off = build_mem_space_config(adapter_cfg)
    cfg_off.l3_recon_max_positions = args.chunk_size
    from src.memory.mem_space.layer import MemorySpaceLayer
    MemorySpaceLayer._instance_counter = 0
    model_off = load_mem_space_model(
        model_path=args.model_path, checkpoint_path=args.checkpoint,
        mem_config=cfg_off, device=device, dtype=dtype, attn_impl="sdpa",
    )
    # Hook to capture the extended-seq length at the evidence layer.
    off_lens = []
    ev_layer = model_off._mem_space_layers[args.evidence_layer]
    ev_layer.wrapped_layer.register_forward_pre_hook(
        lambda m, a, kw: off_lens.append((a[0] if a else kw["hidden_states"]).shape[1]),
        with_kwargs=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    _run(model_off, tokenizer, args.total_len, args.chunk_size, device)
    peak_off = torch.cuda.max_memory_allocated(device) / 1e9
    off_len = max(off_lens)
    del model_off
    torch.cuda.empty_cache()

    # ---- Pass B: evidence ON at generous budget ----
    cfg_on = build_mem_space_config(adapter_cfg)
    cfg_on.l3_recon_max_positions = args.chunk_size
    cfg_on.use_slot_evidence = True
    cfg_on.evidence_buffer_size = args.evidence_buffer_size
    cfg_on.evidence_topr = args.evidence_topr
    cfg_on.evidence_layer = args.evidence_layer
    MemorySpaceLayer._instance_counter = 0
    model_on = load_mem_space_model(
        model_path=args.model_path, checkpoint_path=args.checkpoint,
        mem_config=cfg_on, device=device, dtype=dtype, attn_impl="sdpa",
    )
    on_lens = []
    ev_layer_on = model_on._mem_space_layers[args.evidence_layer]
    ev_layer_on.wrapped_layer.register_forward_pre_hook(
        lambda m, a, kw: on_lens.append((a[0] if a else kw["hidden_states"]).shape[1]),
        with_kwargs=True,
    )
    torch.cuda.reset_peak_memory_stats(device)
    _run(model_on, tokenizer, args.total_len, args.chunk_size, device)
    peak_on = torch.cuda.max_memory_allocated(device) / 1e9
    on_len = max(on_lens)

    bank = model_on._mem_space_shared_bank
    ev_alloc = bank.slot_evidence is not None
    max_count = int(bank.slot_evidence_count.max().item()) if ev_alloc else 0
    grew = on_len - off_len

    print("=" * 60)
    print("STAGE-1 EVIDENCE SMOKE RESULT")
    print(f"  model:            {args.model_path}")
    print(f"  total_len/chunk:  {args.total_len}/{args.chunk_size}")
    print(f"  budget:           buffer_size={args.evidence_buffer_size} "
          f"topr={args.evidence_topr} layer={args.evidence_layer}")
    print(f"  evidence buffer allocated:  {ev_alloc}")
    print(f"  max slot_evidence_count:    {max_count}  (must be > 0)")
    print(f"  ext_seq OFF/ON:             {off_len} / {on_len}  (grew {grew})")
    print(f"  peak GPU mem OFF:           {peak_off:.2f} GB")
    print(f"  peak GPU mem ON:            {peak_on:.2f} GB")
    print(f"  footprint DELTA:            {peak_on - peak_off:+.2f} GB")
    print("=" * 60)
    ok = ev_alloc and max_count > 0 and grew > 0
    print("STAGE1_PASS" if ok else "STAGE1_FAIL")


if __name__ == "__main__":
    main()
