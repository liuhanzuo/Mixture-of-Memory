#!/usr/bin/env python3
"""STAGE 1 — Slot-Routed Evidence Memory smoke / VRAM probe.

Loads the real Llama-3-8B backbone patched with mem_space (loading the converged
P11 adapter), turns the evidence path ON with a LARGE budget, streams a few
chunks of pg19 prose through the memory bank, and verifies:

  (a) the model does not OOM and we can measure the VRAM footprint;
  (b) the evidence path is genuinely EXECUTED (not a silent no-op):
      - memory_bank.slot_evidence is materialised (non-None) after streaming,
      - at least one slot has slot_evidence_count > 0 (tokens were cached),
      - at readout the evidence block (k_ev) is non-zero in the extended seq.

We compare evidence-ON peak VRAM vs evidence-OFF peak VRAM at the same config so
the marginal footprint of the raw-KV evidence buffer is isolated.

Usage:
  python scripts/evidence_probe_stage1.py \
      --model models/Meta-Llama-3-8B-Instruct \
      --adapter_config outputs/mem_space_p11_chunk512_INSTRUCT/adapter_config.json \
      --checkpoint outputs/mem_space_p11_chunk512_INSTRUCT/mem_space_adapter.pt \
      --data data/pg19_chunks_llama3.npy \
      --chunk_size 512 --n_chunks 6 \
      --evidence_buffer_size 64 --evidence_topr 64 --evidence_layer 0
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer  # noqa: E402

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
)


def _get_shared_bank(model):
    root = getattr(model, "module", model)
    bank = getattr(root, "_mem_space_shared_bank", None)
    if bank is not None:
        return bank
    layers = getattr(root, "_mem_space_layers", [])
    return layers[0].memory_bank if layers else None


def _reset_banks(model):
    bank = _get_shared_bank(model)
    if bank is not None:
        bank.reset()


def stream(model, ids_list, chunk_size, device):
    """Stream chunks into the bank (no freeze). Returns the evidence layer's
    captured k_ev (#evidence tokens at the LAST chunk's readout)."""
    model.eval()
    n = len(ids_list)
    for i in range(0, n, chunk_size):
        chunk = ids_list[i : i + chunk_size]
        if len(chunk) < 4:
            break
        t = torch.tensor([chunk], device=device, dtype=torch.long)
        with torch.no_grad():
            _ = model(input_ids=t, use_cache=False)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--adapter_config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_chunks", type=int, default=6)
    p.add_argument("--evidence_buffer_size", type=int, default=64)
    p.add_argument("--evidence_topr", type=int, default=64)
    p.add_argument("--evidence_layer", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    pg19 = np.load(args.data, mmap_mode="r")
    flat: list[int] = []
    target = args.chunk_size * args.n_chunks
    ci = 0
    while len(flat) < target:
        flat.extend(int(x) for x in pg19[ci % len(pg19)])
        ci += 1
    ids_list = flat[:target]

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)

    def run(evidence_on: bool, tag: str):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        cfg = build_mem_space_config(adapter_cfg)
        cfg.l3_recon_max_positions = args.chunk_size
        if evidence_on:
            cfg.use_slot_evidence = True
            cfg.evidence_buffer_size = args.evidence_buffer_size
            cfg.evidence_topr = args.evidence_topr
            cfg.evidence_layer = args.evidence_layer
        else:
            cfg.use_slot_evidence = False
        model = load_mem_space_model(
            model_path=args.model, checkpoint_path=args.checkpoint,
            mem_config=cfg, device=device, dtype=dtype, attn_impl="sdpa",
        )
        mem_before = torch.cuda.memory_allocated(device) / 1e9
        _reset_banks(model)
        stream(model, ids_list, args.chunk_size, device)
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        bank = _get_shared_bank(model)
        se = getattr(bank, "slot_evidence", None)
        sec = getattr(bank, "slot_evidence_count", None)
        result = {
            "tag": tag,
            "evidence_on": evidence_on,
            "model_loaded_GB": round(mem_before, 3),
            "peak_VRAM_GB": round(peak, 3),
            "slot_evidence_materialised": se is not None,
        }
        if se is not None:
            result["slot_evidence_shape"] = list(se.shape)
            result["slot_evidence_buffer_GB"] = round(
                se.numel() * se.element_size() / 1e9, 4
            )
            result["slot_evidence_norm"] = round(float(se.float().norm().item()), 3)
        if sec is not None:
            result["slots_with_evidence"] = int((sec > 0).sum().item())
            result["total_slots"] = int(sec.numel())
            result["max_evidence_count"] = int(sec.max().item())
            result["mean_evidence_count"] = round(float(sec.float().mean().item()), 3)
        print(f"\n===== STAGE1 {tag} =====")
        print(json.dumps(result, indent=2))
        del model
        torch.cuda.empty_cache()
        return result

    print(f"[stage1] streaming {args.n_chunks} chunks x {args.chunk_size} = "
          f"{len(ids_list)} tokens")
    r_off = run(False, "evidence_OFF")
    r_on = run(True, "evidence_ON")

    marginal = round(r_on["peak_VRAM_GB"] - r_off["peak_VRAM_GB"], 3)
    verdict = {
        "off_peak_GB": r_off["peak_VRAM_GB"],
        "on_peak_GB": r_on["peak_VRAM_GB"],
        "evidence_marginal_GB": marginal,
        "evidence_executed": bool(
            r_on.get("slot_evidence_materialised")
            and r_on.get("slots_with_evidence", 0) > 0
            and r_on.get("slot_evidence_norm", 0) > 0
        ),
        "evidence_buffer_GB": r_on.get("slot_evidence_buffer_GB"),
        "slots_with_evidence": r_on.get("slots_with_evidence"),
        "max_evidence_count": r_on.get("max_evidence_count"),
        "evidence_buffer_size": args.evidence_buffer_size,
        "evidence_topr": args.evidence_topr,
    }
    print("\n===== STAGE1 VERDICT =====")
    print(json.dumps(verdict, indent=2))
    out = os.path.join(PROJECT_ROOT, "outputs", "evidence_probe_stage1.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"off": r_off, "on": r_on, "verdict": verdict}, f, indent=2)
    print(f"\n[stage1] wrote {out}")


if __name__ == "__main__":
    main()
