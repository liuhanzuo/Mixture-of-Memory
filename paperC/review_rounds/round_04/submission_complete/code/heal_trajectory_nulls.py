#!/usr/bin/env python3
"""paperC heal-confound — MMLU-Pro read-out of the HEAL TRAJECTORY.

WHAT THIS IS FOR
----------------
`HEAL_CONFOUND_PREREGISTRATION.md` §10 reserved 16 cards for "the offline
MMLU-Pro scoring of milestones (which is 8-GPU sharded and is the actual next
bottleneck)". This script is the read-out side of that reservation: it turns a
set of scored milestone directories into a step -> (letter accuracy vs the
pre-registered letter floor) trajectory, plus the P1/P2 contrasts of §8.

WHY NOT JUST RUN `mmlu_pro_power_nulls.py`
-----------------------------------------
That script is the #251 POWER WALL read-out and its `rollup()` is hard-wired to
the six OLMo-2 arm names + the 15 cross-family names (`DAMAGED_OLMO`,
`OLMO_ARMS`). Pointed at a results root containing only heal milestones it dies
in `rollup()` with `KeyError: 'letter_null'` (verified). Rather than edit a
shipped, already-cited analysis script -- which would put every #251 number at
risk of a silent change -- this file IMPORTS its estimators and its floor
definition verbatim and only replaces the arm-enumeration layer.

The estimators are therefore identical BY CONSTRUCTION, not by copy:
  * `best_constant_letter`  -- the floor (argmax of the gold-letter marginal)
  * `paired_boot`           -- paired bootstrap, n_boot=10000, seed=7
  * `two_sided_boot_p`      -- the R-7-fixed mid-p two-sided bootstrap p
  * `load_records`          -- 8/8 shard + duplicate-id + nan integrity gate
and `mmlu_pro_power_nulls` itself asserts its `two_sided_boot_p` /
`mcnemar_exact_p` are bit-identical to A01's, so this inherits that too.

INTEGRITY (each assertion is HARD -- it raises, it does not warn)
----------------------------------------------------------------
1. shard INDEX SET == {0..7} exactly (not a count of 8: a duplicated shard 3
   with shard 5 missing also counts to 8);
2. n_scored == 12032 exactly, 0 duplicate item_ids, 0 nan;
3. n_trunc == 0 -- re-derived from the per-shard json, not trusted from the log;
4. `chat_template` asserted with `is False` (so a missing/None field FAILS);
   `is not True` would have passed silently on None, which is the exact defect
   that produced a spurious-error cascade elsewhere in this project;
5. the letter floor is asserted BIT-IDENTICAL across every cell (it is a pure
   dataset property; drift means the item sets differ and the cells are not
   comparable);
6. every cell's `keep_front_layers`/`n_fresh_layers` is asserted against what
   the arm name claims, so a mislabelled directory cannot enter the table.

Usage:
  python heal_trajectory_nulls.py <results_root> <out_json> [--extra_root R ...]
CPU only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from mmlu_pro_power_nulls import (  # noqa: E402
    ESTIMATOR_SOURCE,
    TASK,
    best_constant_letter,
    load_records,
    paired_boot,
)

# The pre-registered floor. Asserted, never re-derived to taste
# (HEAL_CONFOUND_PREREGISTRATION.md §8: "the letter floor is always-A 0.116606
# and is asserted bit-identical across all 21 existing cells, so the healed
# arm's floor is FIXED IN ADVANCE and cannot be re-derived to taste").
PREREG_FLOOR_LETTER = "A"
PREREG_FLOOR_ACC = 0.116606
PREREG_N = 12032
# MMLU's own headline effect, the reference the #251 power analysis is against.
MMLU_REFERENCE_EFFECT_PP = 1.389


def _integrity(results_root, arm_dir, expect_keep=None, expect_fresh=None):
    """Hard integrity gate on one scored arm directory. Returns (recs, diag)."""
    d = os.path.join(results_root, arm_dir)
    # (1) shard INDEX SET, not a count.
    shard_files = glob.glob(os.path.join(d, f"per_example_{TASK}_shard*of8.jsonl"))
    idx = sorted(int(os.path.basename(f).split("_shard")[1].split("of")[0])
                 for f in shard_files)
    assert idx == list(range(8)), (
        f"{arm_dir}: shard index set {idx} != {list(range(8))} -- a count of 8 "
        f"is NOT sufficient (dup shard + missing shard also counts to 8)")

    # load_records is #251's own loader; it re-asserts 8/8 + dup ids + nan.
    recs, integ = load_records(results_root, arm_dir)

    # (2) cardinality / dup / nan, re-derived here rather than trusted.
    ids = [r["item_id"] for r in recs]
    assert len(ids) == PREREG_N, f"{arm_dir}: n={len(ids)} != {PREREG_N}"
    assert len(set(ids)) == len(ids), f"{arm_dir}: duplicate item_id"
    n_nan = sum(1 for r in recs if r.get("nan"))
    assert n_nan == 0, f"{arm_dir}: n_nan={n_nan} != 0"

    # (3)+(4) per-shard meta: n_trunc and the chat_template flag.
    n_trunc_total = 0
    metas = []
    for sf in sorted(glob.glob(os.path.join(d, "shard*of8.json"))):
        with open(sf) as fh:
            j = json.load(fh)
        m = j.get("meta", {})
        metas.append(m)
        # ⚠️ `is False`, NOT `is not True`: the latter passes on None, which is
        # exactly how a chat=True/None cell could slip into a base-LM table.
        ct = m.get("chat_template")
        assert ct is False, (
            f"{arm_dir}/{os.path.basename(sf)}: chat_template={ct!r}; every arm "
            f"here is a BASE LM (no SFT/RL) so it MUST be exactly False. "
            f"Checked with `is False` -- a None/missing field is a FAILURE, not "
            f"a pass (`is not True` would have let it through).")
        assert m.get("add_bos") is False, f"{arm_dir}: add_bos={m.get('add_bos')!r}"
        assert m.get("desc_style") == "none", f"{arm_dir}: desc_style"
        for k in ("n_trunc", "trunc"):
            if k in j:
                n_trunc_total += int(j[k])
    assert n_trunc_total == 0, f"{arm_dir}: n_trunc={n_trunc_total} != 0"

    # (6) architecture is what the name claims.
    m0 = metas[0] if metas else {}
    if expect_keep is not None:
        assert int(m0.get("keep_front_layers", -1)) == expect_keep, (
            f"{arm_dir}: keep_front_layers={m0.get('keep_front_layers')} "
            f"!= claimed {expect_keep}")
    if expect_fresh is not None:
        assert int(m0.get("n_fresh_layers", -1)) == expect_fresh, (
            f"{arm_dir}: n_fresh_layers={m0.get('n_fresh_layers')} "
            f"!= claimed {expect_fresh}")

    return recs, {
        "shard_index_set": idx,
        "n_scored": len(ids),
        "n_dup_ids": 0,
        "n_nan": n_nan,
        "n_trunc": n_trunc_total,
        "chat_template": False,
        "add_bos": False,
        "desc_style": "none",
        "meta": m0,
        "loader_integrity": integ,
    }


def _is_scored(results_root, arm_dir):
    """True iff `arm_dir` has a MERGED summary, i.e. scoring finished.

    Used only to decide whether a directory is READY to analyse, never to relax
    an integrity check. A directory that exists with 0 shards is a job still in
    flight (the driver mkdir's before scoring); a directory with 1-7 shards is a
    genuine integrity failure and MUST still reach the assertions in
    `_integrity`. So the skip condition is deliberately "no summary AND no
    shards at all", not "fewer than 8 shards".
    """
    d = os.path.join(results_root, arm_dir)
    if not os.path.isdir(d):
        return False
    if os.path.isfile(os.path.join(d, f"summary_{TASK}.json")):
        return True
    n_shards = len(glob.glob(os.path.join(d, f"per_example_{TASK}_shard*of8.jsonl")))
    if n_shards == 0:
        print(f"[skip] {arm_dir}: 0 shards on disk -> scoring still in flight, "
              f"not yet analysable (this is NOT an integrity pass)")
        return False
    return True  # 1-7 shards: fall through so _integrity RAISES


def analyse(results_root, arm_dir, label, step, family, regime,
            expect_keep=None, expect_fresh=None):
    recs, diag = _integrity(results_root, arm_dir, expect_keep, expect_fresh)

    bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
    # (5) floor is the pre-registered one, bit-identical.
    assert bc_letter == PREREG_FLOOR_LETTER, (
        f"{arm_dir}: floor letter {bc_letter} != pre-registered "
        f"{PREREG_FLOOR_LETTER}")
    assert abs(bc_diag["best_constant_acc"] - PREREG_FLOOR_ACC) < 5e-7, (
        f"{arm_dir}: floor {bc_diag['best_constant_acc']:.9f} != pre-registered "
        f"{PREREG_FLOOR_ACC}")

    lvec = np.array([1.0 if r["letter"]["correct"] else 0.0 for r in recs])
    acc = float(lvec.mean())
    m, lo, hi, p = paired_boot(lvec - bc_vec)

    # degeneracy diagnostics: WHICH letter the model collapses onto, and what a
    # constant predictor of that letter would score. This is the axis on which
    # the trajectory turns out to be interpretable at all.
    hist = {}
    for r in recs:
        L = "ABCDEFGHIJ"[r["letter"]["pred"]]
        hist[L] = hist.get(L, 0) + 1
    top_letter = max(hist.items(), key=lambda kv: kv[1])
    marg = bc_diag["gold_letter_marginal_frac"]
    hw = 100 * (hi - lo) / 2

    return {
        "label": label,
        "arm_dir": arm_dir,
        "results_root": results_root,
        "step": step,
        "family": family,
        "regime": regime,
        "letter_acc": acc,
        "floor_letter": bc_letter,
        "floor_acc": bc_diag["best_constant_acc"],
        "delta_pp": 100 * (acc - bc_diag["best_constant_acc"]),
        "ci95_lo_pp": 100 * lo,
        "ci95_hi_pp": 100 * hi,
        "ci95_half_width_pp": hw,
        "boot_p": p,
        "powered_vs_mmlu_effect": bool(hw < MMLU_REFERENCE_EFFECT_PP),
        "verdict": ("BELOW floor" if (p < 0.05 and acc < bc_diag["best_constant_acc"])
                    else "ABOVE floor" if (p < 0.05 and acc > bc_diag["best_constant_acc"])
                    else "AT floor"),
        "degeneracy": {
            "pred_hist": dict(sorted(hist.items())),
            "modal_pred_letter": top_letter[0],
            "modal_pred_share": top_letter[1] / len(recs),
            "always_modal_pred_acc": marg.get(top_letter[0]),
            "n_distinct_pred_letters": len(hist),
        },
        "integrity": diag,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("out_json")
    ap.add_argument("--extra_root", action="append", default=[],
                    help="root:arm_dir:label:step:family:regime:keep:fresh")
    args = ap.parse_args()

    cells = []
    # the healed Qwen3 trajectory
    for step in (5000, 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000):
        d = f"qwen3base_heal_k8f2_step{step}"
        if _is_scored(args.results_root, d):
            cells.append(analyse(args.results_root, d,
                                 f"qwen3_8b_base/k8+fresh2 heal@{step}", step,
                                 "qwen3_8b_base", "prune_then_heal",
                                 expect_keep=8, expect_fresh=2))
    # OLMo-2 keep8 earlier trajectory point, if scored into this root
    d = "7B_keep8_step45000"
    if _is_scored(args.results_root, d):
        cells.append(analyse(args.results_root, d,
                             "olmo2_7b/keep8+fresh2 heal@45000", 45000,
                             "olmo2_7b", "prune_then_heal",
                             expect_keep=8, expect_fresh=2))

    for spec in args.extra_root:
        root, arm, label, step, family, regime, keep, fresh = spec.split(":")
        cells.append(analyse(root, arm, label, int(step), family, regime,
                             expect_keep=int(keep) if keep else None,
                             expect_fresh=int(fresh) if fresh else None))

    out = {
        "what": "paperC heal-confound: MMLU-Pro letter-vs-floor read-out of the "
                "healed Qwen3-8B-Base front8+fresh2 heal TRAJECTORY",
        "estimator_source": ESTIMATOR_SOURCE,
        "prereg": {
            "floor": f"always-{PREREG_FLOOR_LETTER} {PREREG_FLOOR_ACC}",
            "n": PREREG_N,
            "read_out_step": 121000,
            "note": "121000 is the PRE-REGISTERED read-out. Every cell here is "
                    "a mid-run milestone at step <= 9000, i.e. 4-7% of it. These "
                    "are pipeline + trajectory-shape evidence, NOT the P1/P2 "
                    "verdict, and must never be quoted as the healed result.",
        },
        "n_cells": len(cells),
        "cells": cells,
    }
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"estimators: {ESTIMATOR_SOURCE}")
    print(f"floor: always-{PREREG_FLOOR_LETTER} {PREREG_FLOOR_ACC} (asserted "
          f"bit-identical in all {len(cells)} cells)")
    print()
    hdr = (f"{'label':44s} {'letter':>9s} {'d_pp':>8s} {'hw_pp':>7s} "
           f"{'p':>8s} {'verdict':>11s} {'modal':>6s} {'share':>6s} {'=always':>8s}")
    print(hdr)
    print("-" * len(hdr))
    for c in cells:
        g = c["degeneracy"]
        print(f"{c['label']:44s} {c['letter_acc']:9.6f} {c['delta_pp']:+8.3f} "
              f"{c['ci95_half_width_pp']:7.3f} {c['boot_p']:8.4f} "
              f"{c['verdict']:>11s} {g['modal_pred_letter']:>6s} "
              f"{g['modal_pred_share']:6.3f} {g['always_modal_pred_acc']:8.6f}")
    print()
    for c in cells:
        assert c["integrity"]["n_scored"] == PREREG_N
        assert c["integrity"]["n_nan"] == 0
        assert c["integrity"]["n_trunc"] == 0
        assert c["integrity"]["chat_template"] is False
        assert c["integrity"]["shard_index_set"] == list(range(8))
    print(f"INTEGRITY OK for all {len(cells)} cells: shard set {{0..7}}, "
          f"n=={PREREG_N}, 0 dup, 0 nan, 0 trunc, chat_template is False")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
