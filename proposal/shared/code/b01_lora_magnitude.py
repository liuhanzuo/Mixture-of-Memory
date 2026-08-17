#!/usr/bin/env python
"""Measure the Read-LoRA delta magnitude, to compare against CPT drift.

WHY
---
B01's four-arm gate arms 3/4 graft a Read-LoRA that was distilled against the STOCK
Qwen3-8B upper stack onto the FUNNEL CPT endpoint, whose layers 12..35 were retrained.
b01_drift_probe.py measured that retraining at rel_fro 0.050..0.093 (mean 0.064).

A LoRA is a delta: W_eff = W + (alpha/r) * B @ A. Whether the graft is legitimate turns
on how the size of that delta compares to how far the base moved. If the base moved by
MORE than the delta the LoRA applies, the adapter is being asked to correct a stack it
was never fit for, and arms 3/4 do not measure "bottleneck + Read-LoRA" -- they measure
a base mismatch.

This computes ||(alpha/r) B@A||_F / ||W_stock||_F per targeted tensor. CPU only.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict

import torch
from safetensors import safe_open


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter_dir", required=True)
    ap.add_argument("--base_dir", required=True)
    ap.add_argument("--json_out", required=True)
    ap.add_argument("--label", required=True)
    args = ap.parse_args()

    with open(os.path.join(args.adapter_dir, "adapter_config.json")) as f:
        cfg = json.load(f)
    r = int(cfg["r"])
    alpha = int(cfg["lora_alpha"])
    use_rslora = bool(cfg.get("use_rslora", False))
    scaling = (alpha / (r ** 0.5)) if use_rslora else (alpha / r)
    print(f"[lora] r={r} alpha={alpha} use_rslora={use_rslora} scaling={scaling}", flush=True)
    print(f"[lora] layers_to_transform={cfg.get('layers_to_transform')}", flush=True)

    idx_path = os.path.join(args.base_dir, "model.safetensors.index.json")
    with open(idx_path) as f:
        weight_map = json.load(f)["weight_map"]

    ad_path = os.path.join(args.adapter_dir, "adapter_model.safetensors")
    with safe_open(ad_path, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        tensors = {k: f.get_tensor(k) for k in keys}

    # pair up lora_A / lora_B by their common base-module prefix
    pairs = defaultdict(dict)
    for k in keys:
        m = re.match(r"base_model\.model\.(.+?)\.lora_([AB])\.weight$", k)
        if not m:
            continue
        pairs[m.group(1)][m.group(2)] = k

    open_shards = {}

    def stock(name):
        shard = weight_map.get(name)
        if shard is None:
            return None
        p = os.path.join(args.base_dir, shard)
        if p not in open_shards:
            open_shards.clear()
            open_shards[p] = safe_open(p, framework="pt", device="cpu")
        return open_shards[p].get_tensor(name)

    rows = {}
    # sort by shard so we open each stock shard once
    ordered = sorted(pairs.items(), key=lambda kv: weight_map.get(kv[0] + ".weight", ""))
    for mod, ab in ordered:
        if "A" not in ab or "B" not in ab:
            rows[mod] = {"error": "unpaired lora_A/lora_B"}
            continue
        A = tensors[ab["A"]].to(torch.float32)   # [r, in]
        B = tensors[ab["B"]].to(torch.float32)   # [out, r]
        W = stock(mod + ".weight")
        if W is None:
            rows[mod] = {"error": f"stock weight absent: {mod}.weight"}
            continue
        W = W.to(torch.float32)
        delta = (B @ A) * scaling
        if delta.shape != W.shape:
            rows[mod] = {"error": f"shape {tuple(delta.shape)} vs base {tuple(W.shape)}"}
            continue
        wn = torch.linalg.vector_norm(W).item()
        dn = torch.linalg.vector_norm(delta).item()
        rows[mod] = {
            "delta_fro": dn,
            "base_fro": wn,
            "rel_fro": (dn / wn) if wn > 0 else None,
            "shape": list(W.shape),
        }

    rels = [v["rel_fro"] for v in rows.values()
            if isinstance(v, dict) and v.get("rel_fro") is not None]
    out = {
        "label": args.label,
        "adapter_dir": args.adapter_dir,
        "base_dir": args.base_dir,
        "r": r, "alpha": alpha, "use_rslora": use_rslora, "scaling": scaling,
        "layers_to_transform": cfg.get("layers_to_transform"),
        "n_modules": len(rows),
        "n_scored": len(rels),
        "rel_fro_min": min(rels) if rels else None,
        "rel_fro_max": max(rels) if rels else None,
        "rel_fro_mean": (sum(rels) / len(rels)) if rels else None,
        "per_module": rows,
        "definition": ("rel_fro = ||(alpha/r) B@A||_F / ||W_stock||_F -- the size of the "
                       "correction the adapter applies, relative to the weight it corrects."),
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("n_modules", "n_scored", "rel_fro_min", "rel_fro_max", "rel_fro_mean")},
                     indent=2), flush=True)
    print(f"[lora] wrote {args.json_out}", flush=True)
    # fail loudly rather than reporting a clean-looking partial result
    if not rels or len(rels) != len(rows):
        print(f"[lora] FAIL: scored {len(rels)} of {len(rows)} modules", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
