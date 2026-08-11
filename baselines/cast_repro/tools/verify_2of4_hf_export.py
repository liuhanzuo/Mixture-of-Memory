#!/usr/bin/env python3
"""Verify that the CAST HF export still holds exact 2:4 sparsity on a sample of layers.

This is the SPEC.md S8 criterion #1: "recomputing the 2:4 mask from the saved
weights in a checkpoint" must match what is loaded. Since final_sparse.pt was
already exported with an exact-2:4 assertion, this re-verifies from the HF
safetensors export, i.e. what the eval harness actually sees.

Reports zero_fraction and exact-2:4 tile fraction for a random sample of layers.
"""

import argparse
import json
import random
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--sample-layers", type=int, default=12,
                    help="how many random in-scope tensors to re-check")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[verify_2of4] loading {args.model} (bf16)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, local_files_only=True,
    )

    # Collect all in-scope linear weights.
    scope = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.ndim != 2:
            continue
        if any(name.endswith(p) for p in PROJECTIONS):
            scope.append((name, mod.weight))

    if not scope:
        print("[verify_2of4] FAIL: no in-scope layers found")
        return 1

    print(f"[verify_2of4] in-scope tensors: {len(scope)}", flush=True)

    # Global aggregate over the entire in-scope set.
    #
    # Two directions must be distinguished, because they are not symmetric on
    # hardware: 2:4 sparse tensor cores require *at most* 2 nonzeros per group of
    # 4. A tile with 1 (or 0) nonzeros is a legal subset -- it is over-pruned, and
    # it runs fine. Only a tile with >2 nonzeros violates the format.
    #
    # Gating on `tile_nz != 2` therefore FAILS arms that prune more aggressively
    # (SparseGPT and ProxSparse both trip it: their "bad tiles" are exactly their
    # excess zeros, i.e. 3-zero/1-nonzero tiles). The deployability gate is
    # `tiles_gt2 == 0`; `tiles_lt2` is reported for information only.
    total_elems = 0
    total_zeros = 0
    total_tiles = 0
    total_tiles_gt2 = 0
    total_tiles_lt2 = 0
    for name, w in scope:
        wf = w.detach().float()
        r, c = wf.shape
        if c % 4 != 0:
            print(f"[verify_2of4] WARN: {name} has c={c} not divisible by 4; skipping")
            continue
        total_elems += wf.numel()
        total_zeros += int((wf == 0).sum())
        tile_nz = (wf != 0).reshape(r, c // 4, 4).sum(-1)
        total_tiles += tile_nz.numel()
        total_tiles_gt2 += int((tile_nz > 2).sum())
        total_tiles_lt2 += int((tile_nz < 2).sum())

    zero_frac = total_zeros / total_elems if total_elems else 0.0
    # "exact" keeps its old meaning (tiles with precisely 2 nonzeros) but is now
    # descriptive, not the gate.
    total_bad_tiles = total_tiles_gt2 + total_tiles_lt2
    exact_frac = 1.0 - (total_bad_tiles / total_tiles) if total_tiles else 0.0
    legal_frac = 1.0 - (total_tiles_gt2 / total_tiles) if total_tiles else 0.0
    print(f"[verify_2of4] global: elems={total_elems:,} zeros={total_zeros:,} "
          f"zero_frac={zero_frac:.9f} tiles={total_tiles:,} "
          f"tiles_gt2={total_tiles_gt2} tiles_lt2={total_tiles_lt2} "
          f"exact_2of4_frac={exact_frac:.9f} legal_2of4_frac={legal_frac:.9f}", flush=True)

    # Detail per-sample-layer.
    random.seed(args.seed)
    sample = random.sample(scope, min(args.sample_layers, len(scope)))
    per_layer = []
    for name, w in sample:
        wf = w.detach().float()
        r, c = wf.shape
        tile_nz = (wf != 0).reshape(r, c // 4, 4).sum(-1)
        zeros = int((wf == 0).sum())
        gt2 = int((tile_nz > 2).sum())
        lt2 = int((tile_nz < 2).sum())
        entry = {
            "layer": name,
            "shape": [r, c],
            "zero_fraction": zeros / wf.numel(),
            "exact_2of4_fraction": 1.0 - (gt2 + lt2) / tile_nz.numel(),
            "legal_2of4_fraction": 1.0 - gt2 / tile_nz.numel(),
            "tiles_gt2": gt2,
            "tiles_lt2": lt2,
            "bad_tiles": gt2 + lt2,
        }
        per_layer.append(entry)
        print(f"[verify_2of4]   {name} shape={r}x{c} "
              f"zero_frac={entry['zero_fraction']:.6f} "
              f"exact_2of4={entry['exact_2of4_fraction']:.9f} "
              f"gt2={gt2} lt2={lt2}", flush=True)

    # The gate: no tile may exceed 2 nonzeros, and the layer scope must be complete.
    # zero_frac is checked as a floor (>= 0.5 - eps), not an equality, so that
    # over-pruned arms are not failed for being sparser than required.
    ok = (
        zero_frac >= 0.5 - 1e-4
        and total_tiles_gt2 == 0
        and len(scope) == 224
    )
    if ok and total_tiles_lt2 > 0:
        print(f"[verify_2of4] NOTE: {total_tiles_lt2:,} tiles are over-pruned (<2 nonzeros). "
              f"Legal on 2:4 tensor cores; reported for information.", flush=True)
    print(f"[verify_2of4] VERDICT: {'PASS' if ok else 'FAIL'}", flush=True)

    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
