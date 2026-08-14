#!/usr/bin/env python3
"""STEP 1 validation of an ALPS 2:4 mask artifact for --initial_mask_path.

Asserts the five properties MAIN required before the mask may be used:
  (a) format == "sparseforge-mask-v1"
  (b) pattern == "2:4"
  (c) every group-of-4 along dim=1 has exactly 2 non-zeros
      (nnz-per-group histogram must be exactly {2: N})
  (d) number of covered 2-D linear modules == 224 (7B SparseLinear count)
  (e) global zero fraction == 0.5

CPU only; no GPU is touched.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import torch

EXPECTED_MODULES = 224
# The 224 in-scope tensors are the q/k/v/o + gate/up/down projections of 32 layers.
EXPECTED_ELEMS = 6_476_005_376


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mask_path", required=True)
    ap.add_argument("--out_json", default=None)
    ap.add_argument("--expect_modules", type=int, default=EXPECTED_MODULES)
    args = ap.parse_args()

    mask_path = Path(args.mask_path)
    print(f"[validate] loading {mask_path} ({mask_path.stat().st_size/2**30:.2f} GiB)", flush=True)
    try:
        art = torch.load(mask_path, map_location="cpu", weights_only=True)
    except TypeError:
        art = torch.load(mask_path, map_location="cpu")

    fmt = art.get("format")
    pattern = art.get("pattern")
    masks = art.get("masks")
    print(f"[validate] (a) format = {fmt!r}", flush=True)
    print(f"[validate] (b) pattern = {pattern!r}", flush=True)
    print(f"[validate] top-level keys = {sorted(art.keys())}", flush=True)

    if not isinstance(masks, dict):
        print("[validate] FAIL: 'masks' is not a dict", flush=True)
        return 2

    # Only 2-D entries are in scope (the same rule save_mask_artifact used, but it
    # also emits lm_head; count both so the discrepancy is explicit rather than hidden).
    two_d = {k: v for k, v in masks.items() if torch.is_tensor(v) and v.dim() == 2}
    print(f"[validate] entries total={len(masks)} two_dim={len(two_d)}", flush=True)

    proj_suffixes = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )
    in_scope = {k: v for k, v in two_d.items() if k.endswith(proj_suffixes)}
    out_of_scope = sorted(set(two_d) - set(in_scope))
    print(f"[validate] (d) in-scope projection modules = {len(in_scope)} "
          f"(expected {args.expect_modules})", flush=True)
    if out_of_scope:
        print(f"[validate]     out-of-scope 2-D entries also present: {out_of_scope}", flush=True)

    hist: collections.Counter = collections.Counter()
    total_elems = 0
    total_nnz = 0
    dtypes: collections.Counter = collections.Counter()
    ragged = []

    for i, (name, m) in enumerate(sorted(in_scope.items())):
        dtypes[str(m.dtype)] += 1
        rows, cols = m.shape
        if cols % 4 != 0:
            ragged.append((name, tuple(m.shape)))
            continue
        mi = m.to(torch.int16) if m.dtype != torch.int16 else m
        g = mi.reshape(-1, 4).sum(dim=1)
        # bincount over 0..4
        bc = torch.bincount(g.to(torch.int64), minlength=5)
        for v in range(5):
            c = int(bc[v].item())
            if c:
                hist[v] += c
        total_elems += m.numel()
        total_nnz += int(mi.sum().item())
        del mi, g, bc
        if (i + 1) % 32 == 0:
            print(f"[validate]   scanned {i+1}/{len(in_scope)} modules", flush=True)

    zero_frac = 1.0 - (total_nnz / max(total_elems, 1))
    print(f"[validate] (c) nnz-per-group-of-4 histogram = {dict(sorted(hist.items()))}", flush=True)
    print(f"[validate] (e) elems={total_elems:,} nnz={total_nnz:,} "
          f"zero_frac={zero_frac:.9f}", flush=True)
    print(f"[validate]     mask dtypes = {dict(dtypes)}", flush=True)
    if ragged:
        print(f"[validate]     ragged (cols%4!=0): {ragged}", flush=True)

    checks = {
        "a_format_is_sparseforge_mask_v1": fmt == "sparseforge-mask-v1",
        "b_pattern_is_2of4": pattern == "2:4",
        "c_histogram_is_exactly_two_per_group": (set(hist.keys()) == {2}),
        "d_module_count_is_expected": len(in_scope) == args.expect_modules,
        "e_zero_fraction_is_half": abs(zero_frac - 0.5) < 1e-9,
    }
    extra = {
        "elems_matches_verify_2of4_log": total_elems == EXPECTED_ELEMS,
    }
    for k, v in {**checks, **extra}.items():
        print(f"[validate] {'PASS' if v else 'FAIL'}  {k}", flush=True)

    verdict = all(checks.values())
    print(f"[validate] VERDICT: {'PASS' if verdict else 'FAIL'}", flush=True)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps({
            "mask_path": str(mask_path),
            "mask_bytes": mask_path.stat().st_size,
            "format": fmt,
            "pattern": pattern,
            "entries_total": len(masks),
            "entries_two_dim": len(two_d),
            "in_scope_modules": len(in_scope),
            "out_of_scope_two_dim": out_of_scope,
            "nnz_per_group_of_4_histogram": {str(k): v for k, v in sorted(hist.items())},
            "elems": total_elems,
            "nnz": total_nnz,
            "zero_fraction": zero_frac,
            "mask_dtypes": dict(dtypes),
            "checks": checks,
            "extra": extra,
            "verdict": "PASS" if verdict else "FAIL",
        }, indent=2))
        print(f"[validate] wrote {args.out_json}", flush=True)

    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
