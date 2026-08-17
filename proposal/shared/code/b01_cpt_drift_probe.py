#!/usr/bin/env python
"""B01 four-arm feasibility probe: measure CPT weight drift on the READ band.

WHY THIS EXISTS
---------------
The four-arm gate's arms 3/4 are "bottleneck + Read-LoRA [+ Write-LoRA]". The only
Read-LoRA on disk (outputs/qcmem_distill_qwen_j12_r32_4k) was distilled against the
STOCK Qwen3-8B upper stack (layers 12..35, resume_j=12). The funnel CPT endpoint
(outputs/qwenbott_funnel_L12_d512, unfreeze_from=12) RETRAINED layers 12..35.

If the retrained upper stack has drifted materially from stock, then grafting the
stock-distilled Read-LoRA onto it is a BASE MISMATCH: the LoRA delta B@A was fit for
different base weights W. This script MEASURES the drift instead of asserting it.

It also measures the FROZEN band (layers 0..11) as a negative control: unfreeze_from=12
means those should be EXACTLY stock (drift == 0). If the control is not 0, the reading
of the training script is wrong and nothing else here can be trusted.

CPU only. No GPU. No writes outside --json_out.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from safetensors import safe_open


def load_stock_index(base_dir):
    """Map param name -> shard file, from the safetensors index."""
    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if not os.path.exists(idx_path):
        raise SystemExit(f"FATAL: no index at {idx_path}")
    with open(idx_path) as f:
        idx = json.load(f)
    return idx["weight_map"]


def get_stock(base_dir, weight_map, name, _cache={}):
    """Fetch one stock tensor by param name (fp32 on CPU)."""
    if name not in weight_map:
        return None
    shard = weight_map[name]
    path = os.path.join(base_dir, shard)
    if path not in _cache:
        # keep at most 1 open handle; shards are read in name order so this is fine
        _cache.clear()
        _cache[path] = safe_open(path, framework="pt", device="cpu")
    return _cache[path].get_tensor(name)


def rel_drift(a, b):
    """||a-b||_F / ||b||_F in fp32, plus max abs diff and an exact-equality flag."""
    a = a.to(torch.float32)
    b = b.to(torch.float32)
    if a.shape != b.shape:
        return None
    d = a - b
    den = torch.linalg.vector_norm(b).item()
    num = torch.linalg.vector_norm(d).item()
    n_diff = int((a != b).sum().item())
    return {
        "rel_fro": (num / den) if den > 0 else None,
        "max_abs": float(d.abs().max().item()),
        "n_differing": n_diff,
        "numel": int(a.numel()),
        "bit_identical": n_diff == 0,
    }


# The funnel wraps layers[12] as BottleneckLayer(inner=<Qwen3DecoderLayer>), so the
# ckpt key for the wrapped layer's own params gains an ".inner" segment. Map a stock
# name onto its ckpt name.
def ckpt_key_for(stock_name, bottleneck_layer):
    m = re.match(r"model\.layers\.(\d+)\.(.+)$", stock_name)
    if not m:
        return stock_name
    li, rest = int(m.group(1)), m.group(2)
    if li == bottleneck_layer:
        return f"model.layers.{li}.inner.{rest}"
    return stock_name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", required=True, help="stock Qwen3-8B dir")
    ap.add_argument("--ckpt", required=True, help="CPT endpoint final.pt")
    ap.add_argument("--arch_meta", required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--json_out", required=True)
    # two representative tensors per layer: one attention, one MLP
    ap.add_argument("--probe_suffixes", default="self_attn.q_proj.weight,mlp.down_proj.weight")
    args = ap.parse_args()

    with open(args.arch_meta) as f:
        meta = json.load(f)
    b_layer = int(meta["bottleneck_layer"])
    b_dim = int(meta["bottleneck_dim"])
    L = int(meta["num_hidden_layers"])
    unfreeze_from = int(meta.get("unfreeze_from", -1))

    print(f"[probe] label={args.label} bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
          f"L={L} unfreeze_from={unfreeze_from}", flush=True)

    weight_map = load_stock_index(args.base_dir)
    print(f"[probe] stock index: {len(weight_map)} params", flush=True)

    print(f"[probe] loading ckpt {args.ckpt} (this is ~16 GB, CPU) ...", flush=True)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    state = ck.get("model_state", ck)
    step = ck.get("step")
    print(f"[probe] ckpt loaded: step={step} n_keys={len(state)}", flush=True)

    suffixes = [s.strip() for s in args.probe_suffixes.split(",") if s.strip()]

    per_layer = {}
    for li in range(L):
        row = {}
        for suf in suffixes:
            stock_name = f"model.layers.{li}.{suf}"
            ck_name = ckpt_key_for(stock_name, b_layer)
            if ck_name not in state:
                row[suf] = {"error": f"ckpt key absent: {ck_name}"}
                continue
            w_stock = get_stock(args.base_dir, weight_map, stock_name)
            if w_stock is None:
                row[suf] = {"error": f"stock key absent: {stock_name}"}
                continue
            row[suf] = rel_drift(state[ck_name], w_stock)
        per_layer[li] = row

    # funnel params (only exist on the funnel arm)
    funnel = {}
    for k in (f"model.layers.{b_layer}.down.weight", f"model.layers.{b_layer}.up.weight"):
        if k in state:
            t = state[k]
            funnel[k] = {"shape": list(t.shape), "dtype": str(t.dtype)}

    out = {
        "label": args.label,
        "ckpt": args.ckpt,
        "step": step,
        "arch_meta": {"bottleneck_layer": b_layer, "bottleneck_dim": b_dim,
                      "num_hidden_layers": L, "unfreeze_from": unfreeze_from},
        "base_dir": args.base_dir,
        "probe_suffixes": suffixes,
        "per_layer": {str(k): v for k, v in per_layer.items()},
        "funnel_params": funnel,
        "note": ("rel_fro = ||W_ckpt - W_stock||_F / ||W_stock||_F, fp32. "
                 "Layers < unfreeze_from were FROZEN and must be bit-identical "
                 "(n_differing == 0); that is the negative control."),
    }

    # ---- built-in adjudication so the reader does not have to eyeball 36 rows ----
    frozen_bad, frozen_ok = [], []
    for li in range(0, max(unfreeze_from, 0)):
        for suf, r in per_layer[li].items():
            if isinstance(r, dict) and "bit_identical" in r:
                (frozen_ok if r["bit_identical"] else frozen_bad).append(f"L{li}.{suf}")
    trained_drift = []
    for li in range(max(unfreeze_from, 0), L):
        for suf, r in per_layer[li].items():
            if isinstance(r, dict) and r.get("rel_fro") is not None:
                trained_drift.append(r["rel_fro"])
    out["adjudication"] = {
        "frozen_band_layers": f"0..{unfreeze_from - 1}",
        "frozen_band_bit_identical_count": len(frozen_ok),
        "frozen_band_VIOLATIONS": frozen_bad,
        "frozen_control_holds": len(frozen_bad) == 0,
        "trained_band_layers": f"{unfreeze_from}..{L - 1}",
        "trained_band_n_tensors": len(trained_drift),
        "trained_band_rel_fro_min": min(trained_drift) if trained_drift else None,
        "trained_band_rel_fro_max": max(trained_drift) if trained_drift else None,
        "trained_band_rel_fro_mean": (sum(trained_drift) / len(trained_drift))
                                     if trained_drift else None,
    }

    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out["adjudication"], indent=2), flush=True)
    print(f"[probe] wrote {args.json_out}", flush=True)

    if not out["adjudication"]["frozen_control_holds"]:
        print("[probe] FAIL: frozen-band control violated -- the reading of "
              "unfreeze_from is wrong, do NOT trust the drift numbers", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
