#!/usr/bin/env python3
"""A04 margin guard evidence: the intact arm's OWN residual + CI under all five
MMLU longest-option tie conventions, at 1B, on the real item set (n=14,042),
plus a cross-disk stability check on the intact anchor.

WHY: Delta_x = 0.10 * residual(intact, x). The pre-registered rule assumes
residual(intact) is a comfortably positive number. This script measures:
  (i)   its value per convention                        -> guard condition D1/D2
  (ii)  its two-sided 95% CI                            -> guard condition D3
  (iii) whether two admissible measurements of the SAME
        intact model give the same residual              -> guard condition D5

Scorers/nulls are IMPORTED from A03's canonical
`analyze_1b_knowledge_floor.py`, never reimplemented.

CPU ONLY. No GPU, no model load, no torch. Read-only on all inputs.

Inputs (per-example MMLU dumps, 14,042 rows each):
  --zwfy6_intact  olmo2_mmlu_content_results/A03_1B_base/per_example_mmlu.jsonl
                  (zwfy6; md5 d1a7b1cefc0031afa84e7b9334a08bc5)
  --wzc1_intact   olmo2_mmlu_content_results/a01_1B_intact_base_full/
                  per_example_mmlu.jsonl (wzc1) -- optional, for the D5 check
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
# A03 was ARCHIVED 2026-08-11 (proposal/active -> proposal/archive). Its
# `analyze_1b_knowledge_floor.py` is still the canonical scorer/null source and
# is imported, never reimplemented. Resolve its directory through the shared
# helper so the location lives in ONE place and a missing A03 fails loudly
# instead of silently falling back to a re-derived metric.
_SHARED_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "shared", "code"))
if _SHARED_CODE not in sys.path:
    sys.path.insert(0, _SHARED_CODE)
from proposal_paths import a03_code_dir  # noqa: E402

_A03_CODE = a03_code_dir()
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    longest_option_vector,
    paired_bootstrap,
)

EXPECTED_N = 14042
DELTA_FRACTION = 0.10   # pre-registered, git d1ba737


def load_mmlu(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    assert len(rows) == EXPECTED_N, f"{path}: n={len(rows)} != {EXPECTED_N}"
    assert not any(r.get("nan") for r in rows), f"{path}: nan rows present"
    assert len({r["item_id"] for r in rows}) == EXPECTED_N, \
        f"{path}: duplicate item_id"
    return rows


def one_sided_lower95(d, seed):
    """5th percentile of the paired item bootstrap -- the same estimator the
    NI rule uses (`pilot_zero_rule_disagreement.py::ni_rule`)."""
    rng = np.random.default_rng(seed)
    vals, counts = np.unique(d, return_counts=True)
    draws = rng.multinomial(d.size, counts / d.size, size=N_BOOT)
    means = draws @ vals / d.size
    return float(np.percentile(means, 5.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zwfy6_intact", required=True,
                    help="A03_1B_base/per_example_mmlu.jsonl (the anchor "
                         "Pilot Zero used)")
    ap.add_argument("--wzc1_intact", default="",
                    help="a01_1B_intact_base_full/per_example_mmlu.jsonl "
                         "(second admissible measurement, for D5)")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    rows = load_mmlu(args.zwfy6_intact)
    gold = [r["gold_letter"] for r in rows]
    CN = np.array([1.0 if r["content_norm"]["correct"] else 0.0 for r in rows])

    out = {
        "what": ("intact 1B MMLU-content residual + CI under all five "
                 "longest-option tie conventions; the input to A04's Delta"),
        "date": "2026-08-10",
        "gpu_spent": 0,
        "source_anchor": args.zwfy6_intact,
        "n": len(rows),
        "n_boot": N_BOOT,
        "seed_base": SEED,
        "delta_fraction_prereg": DELTA_FRACTION,
        "reported_content_norm": float(CN.mean()),
        "by_convention": {},
    }
    print(f"intact 1B content_norm acc = {CN.mean():.10f}  n={len(rows)}")
    for i, conv in enumerate(TIE_CONVS):
        nv = longest_option_vector(rows, gold, conv)
        d = CN - nv
        mean, lo, hi, p = paired_bootstrap(d, seed=SEED + 200 + i)
        lo1s = one_sided_lower95(d, SEED + 200 + i)
        resid = float(CN.mean() - nv.mean())
        rec = {
            "null": float(nv.mean()),
            "residual_pp": 100 * resid,
            "boot_mean_pp": 100 * mean,
            "ci95_pp": [100 * lo, 100 * hi],
            "one_sided_lower95_pp": 100 * lo1s,
            "boot_p": p,
            "delta_pp_as_prereg": 100 * DELTA_FRACTION * resid,
            "residual_le_0": bool(resid <= 0),
            "ci_straddles_0": bool(lo < 0 < hi),
        }
        out["by_convention"][conv] = rec
        print(f"{conv:>7}: null={nv.mean():.8f} resid={100*resid:+8.4f}pp "
              f"CI95=[{100*lo:+8.4f},{100*hi:+8.4f}] p={p:.4f} "
              f"Delta={100*DELTA_FRACTION*resid:+7.4f}pp "
              f"straddles0={rec['ci_straddles_0']}")

    # ---- D5: is the intact anchor unique? -------------------------------
    if args.wzc1_intact:
        rows2 = load_mmlu(args.wzc1_intact)
        assert [r["item_id"] for r in rows2] == [r["item_id"] for r in rows], \
            "item_id misaligned between the two intact dumps"
        assert [r["gold_letter"] for r in rows2] == gold, \
            "gold misaligned between the two intact dumps"
        CN2 = np.array([1.0 if r["content_norm"]["correct"] else 0.0
                        for r in rows2])
        flips = int((CN != CN2).sum())
        # the null is a tokenizer+dataset property: assert it does NOT move
        for conv in TIE_CONVS:
            a = longest_option_vector(rows, gold, conv).mean()
            b = longest_option_vector(rows2, gold, conv).mean()
            assert abs(a - b) < 1e-12, f"null drifted between dumps: {conv}"
        drift_acc_pp = 100 * float(CN2.mean() - CN.mean())
        out["anchor_stability_D5"] = {
            "second_measurement": args.wzc1_intact,
            "note": ("same base model (models/OLMo-2-0425-1B, 16L, add_bos "
                     "false, content_desc full), same item set, same nulls; "
                     "different disk/node and a harness commit apart"),
            "item_flips": flips,
            "acc_drift_pp": drift_acc_pp,
            "residual_drift_pp": drift_acc_pp,   # null cancels
            "delta_drift_pp": DELTA_FRACTION * drift_acc_pp,
            "nulls_identical": True,
        }
        out["anchor_drift_delta_pp"] = DELTA_FRACTION * drift_acc_pp
        print(f"\nD5 anchor check: {flips} item flips / {len(rows)} -> "
              f"residual drift {drift_acc_pp:+.6f}pp -> "
              f"Delta drift {DELTA_FRACTION*drift_acc_pp:+.6f}pp")
    else:
        out["anchor_drift_delta_pp"] = None

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
