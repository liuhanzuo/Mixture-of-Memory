#!/usr/bin/env python3
"""Classify 2:4 tile deviations by DIRECTION, and gate on the direction that matters.

WHY THIS EXISTS
---------------
``verify_2of4_hf_export.py`` gates on ``bad_tiles == 0`` where a tile is "bad" if
its nonzero count is anything other than exactly 2. That conflates two opposite
situations:

  * tile with **> 2** nonzeros -- a real 2:4 budget violation. The weight cannot be
    stored in a 2:4 compressed format and the arm is simply not 2:4.
  * tile with **< 2** nonzeros -- a strict *subset* of an allowed 2:4 tile. It has
    fewer live weights, not more; it is representable in the same 2:4 format (the
    extra slot just holds a zero) and it runs unchanged on 2:4 sparse tensor cores.
    This is what a solver produces when a group's weights are (near-)degenerate, and
    it is not a violation of anything.

Two SparseForge baseline checkpoints fail the strict gate purely in the harmless
direction (ProxSparse-official: 68 tiles with 1 nonzero; SparseGPT seed0: 462 with
1 nonzero; both have **zero** tiles above 2). Refusing to score them on the strict
gate would drop two published table rows over a technicality, so the gate used for
scoring is ``tiles_gt2 == 0`` while the strict result is still recorded.

Exit code 0 iff there are no tiles with more than 2 nonzeros and the in-scope set
is the expected 224 projections at the expected ~0.5 zero fraction.
"""

from __future__ import annotations

import argparse
import collections
import json

import torch
from transformers import AutoModelForCausalLM

PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--expect-tensors", type=int, default=224)
    ap.add_argument("--max-offenders", type=int, default=400,
                    help="cap the per-layer offender list written to JSON")
    args = ap.parse_args()

    print(f"[tile_dir] loading {args.model} (bf16, cpu)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
        local_files_only=True,
    )

    hist = collections.Counter()
    offenders = []
    n_scope = tot_elems = tot_zeros = tot_tiles = 0

    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.ndim != 2:
            continue
        if not any(name.endswith(p) for p in PROJECTIONS):
            continue
        n_scope += 1
        w = mod.weight.detach().float()
        rows, cols = w.shape
        if cols % 4 != 0:
            print(f"[tile_dir] WARN {name} cols={cols} not divisible by 4; skipped")
            continue
        nz = w != 0
        tot_elems += w.numel()
        tot_zeros += int((~nz).sum())
        tile_nz = nz.reshape(rows, cols // 4, 4).sum(-1)
        tot_tiles += tile_nz.numel()
        counts = torch.bincount(tile_nz.reshape(-1), minlength=5)
        for k in range(5):
            hist[k] += int(counts[k])
        over = int((tile_nz > 2).sum())
        under = int((tile_nz < 2).sum())
        if over or under:
            offenders.append({"layer": name, "shape": [rows, cols],
                              "tiles_gt2": over, "tiles_lt2": under})

    if tot_tiles == 0:
        print("[tile_dir] FAIL: no in-scope tiles found")
        return 1

    gt2 = hist[3] + hist[4]
    lt2 = hist[0] + hist[1]
    zero_frac = tot_zeros / tot_elems
    result = {
        "model": args.model,
        "in_scope_tensors": n_scope,
        "elems": tot_elems,
        "zeros": tot_zeros,
        "zero_frac": zero_frac,
        "tiles": tot_tiles,
        "tile_nonzero_histogram": {str(k): hist[k] for k in range(5)},
        "tiles_gt2_TOTAL_budget_violations": gt2,
        "tiles_lt2_TOTAL_sparser_than_2of4": lt2,
        "exact_2of4_tile_ratio": hist[2] / tot_tiles,
        "strict_gate_would_pass": gt2 == 0 and lt2 == 0,
        "deployable_2of4": gt2 == 0,
        "n_offending_layers": len(offenders),
        "offending_layers": offenders[: args.max_offenders],
        "note": ("A tile with <2 nonzeros is a strict subset of an allowed 2:4 tile "
                 "(fewer live weights, same compressed format, runs on 2:4 tensor "
                 "cores), so it does not break 2:4 deployability. A tile with >2 "
                 "nonzeros does. Scoring gates on tiles_gt2 == 0."),
    }

    ok = (
        gt2 == 0
        and n_scope == args.expect_tensors
        and abs(zero_frac - 0.5) < 1e-4
    )
    result["verdict"] = "PASS" if ok else "FAIL"

    with open(args.output, "w") as fh:
        json.dump(result, fh, indent=2)

    slim = {k: v for k, v in result.items() if k != "offending_layers"}
    print(json.dumps(slim, indent=2), flush=True)
    print(f"[tile_dir] VERDICT: {result['verdict']} "
          f"(gt2={gt2} lt2={lt2} scope={n_scope} zero_frac={zero_frac:.9f})", flush=True)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
