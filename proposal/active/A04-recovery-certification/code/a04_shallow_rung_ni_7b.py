#!/usr/bin/env python3
"""A04 — can NI(Delta) ever be observed to ACCEPT? The shallow-rung 7B test.

THE QUESTION
------------
A04's `NI(Delta)` has never been observed to accept. Every measured rung is
CONSTANT-REJECT: 1B `keep7+fresh2` rejects 4/4 axes at all four scored
checkpoints (Pilot Zero + the step-50k/100k/150k extensions), and 1B `keep12`
rejects 4/4 at 5,000 steps (Stage B, deficit/sd_run +27.0 to +90.4).

A rule that only ever rejects has not been shown to DISCRIMINATE. "PPL accepts
what NI rejects" is trivial on an arm that is simply bad. So the promotion
blocker is: is there ANY rung, anywhere in reach, where NI can be observed to
accept on >= 2 of the 3 decision axes?

The 7B ladder is the shallowest damage in the repository and is already scored
on all four axes, so it answers this for ZERO new GPU:
  * `keep14+fresh2`  -- 14 inherited + 2 fresh = 16 of 32 layers, 200k heal steps
  * `shortgpt16`     -- 16 of 32 layers kept non-contiguously, 200k heal steps
Both keep 50% of depth, versus 56.25% (keep7+fresh2 at 1B, 9/16) and 87.5%
(keep12 at 1B, 14/16). NOTE this is *less* depth-fraction than 1B keep12; what
makes these rungs the best available shot is 200k heal steps at 7B scale, i.e.
40x Stage B's token budget, on the only ladder in the repo already scored on
all four axes.

WHAT IS FROZEN AND REUSED, NEVER REIMPLEMENTED
----------------------------------------------
  * `ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `build_axis_data`,
    `mmlu_content_norm_vec`, `qa_metric_vec`, `EXPECTED_N`, `AXES`,
    `DEMOTED_AXES`, `PREREG`  <- imported from `pilot_zero_rule_disagreement`.
  * `paired_bootstrap`, `longest_option_vector`, `best_constant_qa`,
    `best_constant_letter`, `TIE_CONVS`, `N_BOOT`, `SEED` <- imported from A03's
    `analyze_1b_knowledge_floor` via `proposal_paths.a03_code_dir()`.
No metric, null or rule is re-derived here. Two subagents in this repository
have already produced spurious significance by reimplementing a metric.

THE INTACT ANCHOR (the part that decides whether any of this means anything)
---------------------------------------------------------------------------
`Delta_x = 0.10 * residual(intact, x)`, so the anchor IS the margin. A04's 1B
work pinned the anchor (guard G0) to `A03_1B_base`, whose summary meta is
`{"mode": "base", "num_hidden_layers": 16, "base_model": "../models/OLMo-2-0425-1B"}`
-- i.e. the VANILLA base model, not a continued-pretrained one.

By exact analogy the 7B anchor is the vanilla `models/OLMo-2-1124-7B`:
    `olmo2_closedbook_results/base_full`      (triviaqa, popqa)
    `olmo2_closedbook_results/base_full_nqopen`  (nq_open)
    `olmo2_mmlu_content_results/7B_base`      (mmlu_content)
all with meta `{"mode": "base", "num_hidden_layers": 32,
"base_model": "../models/OLMo-2-1124-7B", "add_bos": false}` and 8/8 shards.

`full32_step25000` is NOT the anchor and is not interchangeable with it: its
meta is `{"mode": "pruned", "keep_front_layers": 32, "n_fresh_layers": 0,
"ckpt": "outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt"}` -- an UNDAMAGED
32-layer model that has been continued-pretrained on the heal corpus. Guard G2
forbids changing the anchor, so `full32` is carried here as
  (a) an ARM tested against the pinned anchor -- a zero-damage control, and
  (b) a labelled DIAGNOSTIC that cannot alter any verdict.
This matters: full32 scores BELOW the vanilla base on all four axes, so the
heal corpus itself moves the model away from the anchor. Reporting that as a
"recovery deficit" would be a category error, which is exactly why it is a
separate diagnostic and not a substituted anchor.

GUARD D1-D6 PRECEDES NI (guard G1). `Delta` is NEVER substituted (guard G2).
`p*_crit = n * (Delta_x / (100*1.959964))**2` is the FROZEN FORMULA of guard G3;
its inputs `Delta_x` and `n` are per-scale, so the numeric `p*_crit` is
recomputed for the 7B `Delta` and BOTH values are reported. No threshold is
adjusted.

CROSS-SCALE: `sd_run` exists only at 1B (S=3 seeds, keep12). There is exactly
ONE seed per 7B rung, so no 7B `sd_run` is computable. Deficit/sd_run is
therefore reported ONLY as an explicitly-labelled 1B-imported extrapolation,
never as a 7B variance statement. A 7B measurement-jitter floor IS computable
from the repeated scorings of the same checkpoint and is reported separately --
it is harness jitter, NOT seed variance, and the two are not interchangeable.

CPU ONLY. No GPU, no model load, no torch. Read-only on all inputs.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
_SHARED_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "shared", "code"))
if _SHARED_CODE not in sys.path:
    sys.path.insert(0, _SHARED_CODE)

# Canonical rules/scorers/nulls -- IMPORTED, never reimplemented.
from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    load_shards,
    mmlu_content_norm_vec,
    ni_rule,
    qa_metric_vec,
    ratio_rule,
)
from proposal_paths import a03_code_dir  # noqa: E402

_A03_CODE = a03_code_dir()
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    paired_bootstrap,
)

# ---------------------------------------------------------------------------
# Guard G3 frozen numeric triggers (A04_MARGIN_GUARD_PREREG.md §4).
# ---------------------------------------------------------------------------
D2_RESIDUAL_FLOOR_PP = 1.0          # "0 <= residual(intact) <= 1.0pp"
D5_DRIFT_FRACTION = 0.10            # "|Delta drift| >= 0.10 * Delta"
Z95_TWO_SIDED = 1.959964            # the constant written into G3's formula
D4_CONSTANT_FRAC = 0.99             # ">99% of items a single constant"
D4_TIE_FRAC = 0.99                  # ">=99% tie rate in the null's winner set"

# `p*_crit` as PRE-COMPUTED in the prereg for the 1B cell family. Carried for
# transparency only; the 7B values are recomputed from the 7B Delta below.
PSTAR_CRIT_1B_PREREG = {"triviaqa": 7.6359, "popqa": 0.6476,
                        "mmlu_content": 0.3832, "nq_open": 0.0883}

# 1B sd_run, S=3 seeds, keep12 @5000 steps -- `evidence/pilot_one_stage_b_s3_verdict.json`.
# IMPORTED FOR LABELLED CROSS-SCALE EXTRAPOLATION ONLY. Not a 7B variance model.
SD_RUN_1B_PP = {"triviaqa": 0.30232964433611973,
                "popqa": 0.33280225733083795,
                "mmlu_content": 0.07834137679932787,
                "nq_open": 0.20913986061432647}

# The pinned 7B intact anchor (guard G0 analogue). Vanilla OLMo-2-1124-7B.
ANCHOR = {"mmlu": "7B_base", "cb": "base_full", "nq": "base_full_nqopen"}

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]


def _load_arm(mm_root, cb_root, spec):
    """Load every axis for one arm with the canonical hard assertions.

    Mirrors `pilot_zero_rule_disagreement.build_axis_data` for a single arm, so
    that per-arm directories can live under different roots (the shortgpt16
    closed-book shards exist only on wzc1 and were staged separately). The
    loader, the assertions and the metric functions are the imported ones.
    """
    out, prov = {}, {}
    if spec.get("mmlu"):
        d = spec["mmlu"] if os.path.isabs(spec["mmlu"]) \
            else os.path.join(mm_root, spec["mmlu"])
        rows = load_shards(d, "mmlu", EXPECTED_N["mmlu"])
        out["mmlu_content"] = mmlu_content_norm_vec(rows)
        out["_mmlu_rows"] = rows
        prov["mmlu_content"] = {"dir": d, "n": len(rows), "shards": 8}
    if spec.get("cb"):
        d = spec["cb"] if os.path.isabs(spec["cb"]) \
            else os.path.join(cb_root, spec["cb"])
        for task in ("triviaqa", "popqa"):
            rows = load_shards(d, task, EXPECTED_N[task])
            out[task] = qa_metric_vec(rows, "em")
            out[f"_{task}_rows"] = rows
            prov[task] = {"dir": d, "n": len(rows), "shards": 8, "metric": "em"}
    if spec.get("nq"):
        d = spec["nq"] if os.path.isabs(spec["nq"]) \
            else os.path.join(cb_root, spec["nq"])
        rows = load_shards(d, "nq_open", EXPECTED_N["nq_open"])
        out["nq_open"] = qa_metric_vec(rows, "em")
        out["_nq_open_rows"] = rows
        prov["nq_open"] = {"dir": d, "n": len(rows), "shards": 8,
                           "metric": "em"}
    return out, prov


def assert_aligned(data, prov):
    """nan-free + identical item_id sequence across arms, per axis.

    Verbatim in intent from `a04_step100k_plateau_vs_ni.assert_nan_free_and_aligned`:
    `load_shards` guarantees 8/8 shards, no duplicate item_id and the exact
    count, but NOT that the arms cover the SAME item_ids -- without which the
    paired difference silently compares different items.
    """
    report, ref = {}, {}
    for arm in data:
        report[arm] = {}
        for axis in AXES:
            key = {"mmlu_content": "_mmlu_rows"}.get(axis, f"_{axis}_rows")
            if key not in data[arm]:
                continue
            rows = data[arm][key]
            n_nan = sum(1 for r in rows if r.get("nan"))
            if n_nan:
                raise SystemExit(f"FATAL {arm}/{axis}: {n_nan} nan=true rows")
            ids = [r["item_id"] for r in rows]
            exp = EXPECTED_N["mmlu" if axis == "mmlu_content" else axis]
            if len(ids) != exp:
                raise SystemExit(f"FATAL {arm}/{axis}: n={len(ids)} != {exp}")
            if len(set(ids)) != len(ids):
                raise SystemExit(f"FATAL {arm}/{axis}: duplicate item_id")
            if axis not in ref:
                ref[axis] = ids
            elif ids != ref[axis]:
                raise SystemExit(
                    f"FATAL {arm}/{axis}: item_id sequence differs from the "
                    "reference arm -- the paired difference would compare "
                    "different items")
            report[arm][axis] = {"n": len(ids), "n_nan": 0, "shards": 8,
                                 "item_ids_aligned": True,
                                 "dir": prov[arm][axis]["dir"]}
    return report


def d4_interface_degenerate(data, arm, axis, nulls):
    """Guard D4's structural half: a single constant on >99% of items, or a
    >=99% tie rate in the null's winner set. Measured, not assumed."""
    if axis == "mmlu_content":
        rows = data[arm]["_mmlu_rows"]
        preds = [r["content_norm"].get("pred_letter") for r in rows]
        n = len(preds)
        top = max({p: preds.count(p) for p in set(preds)}.values()) / n
        tie = nulls["mmlu_content"]["frac_items_with_tied_longest"]
        return {"top_constant_frac": top, "tie_frac": tie,
                "degenerate": bool(top > D4_CONSTANT_FRAC
                                   or tie >= D4_TIE_FRAC)}
    rows = data[arm][f"_{axis}_rows"]
    preds = [(r.get("pred") or "").strip().lower() for r in rows]
    n = len(preds)
    top = max({p: preds.count(p) for p in set(preds)}.values()) / n
    return {"top_constant_frac": top, "tie_frac": None,
            "degenerate": bool(top > D4_CONSTANT_FRAC)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--shortgpt_cb", required=True,
                    help="staged dir holding shortgpt16 triviaqa/popqa shards")
    ap.add_argument("--shortgpt_nq", required=True,
                    help="staged dir holding shortgpt16 nq_open shards")
    ap.add_argument("--d5_cb", default="", help="2nd intact CB measurement dir")
    ap.add_argument("--d5_mm", default="", help="2nd intact MMLU measurement dir")
    ap.add_argument("--out_json", required=True)
    # 2026-08-12: independent-re-scoring overrides for the full32 zero-damage
    # control. DEFAULTS ARE THE ARCHIVED DIR NAMES, so omitting them reproduces
    # the 2026-08-12 16:20 run byte-for-byte -- verified by re-running with no
    # overrides and diffing the output JSON. Added because `RATIO(0.85)`'s
    # accept of full32 clears rho by only +0.0014951, i.e. 0.0924 pp (nq_open)
    # / 0.1116 pp (popqa) of accuracy from flipping, and full32 had exactly ONE
    # scoring per axis, so the fragility flagged in
    # `A04_SHALLOW_RUNG_NI_DISCRIMINATION_VERDICT.md` §6.2 was uncheckable.
    # These flags let a second, protocol-identical scoring be substituted
    # WITHOUT touching `ni_rule`, `ratio_rule`, the anchor (guard G2), `Delta`,
    # `rho`, or any threshold. Only the full32 ARM's input directories move.
    ap.add_argument("--full32_cb", default="full32_step25000",
                    help="full32 closed-book dir (triviaqa+popqa)")
    ap.add_argument("--full32_nq", default="full32_step25000_nqopen",
                    help="full32 nq_open dir")
    args = ap.parse_args()

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    arm_specs = {
        "intact_7B_base": dict(ANCHOR),
        "keep14fresh2_step200k": {"mmlu": "7B_keep14_step200000",
                                  "cb": "keep14_step200k",
                                  "nq": "keep14_step200k_nqopen"},
        "shortgpt16_step200k": {"mmlu": "7B_shortgpt16_step200000",
                                "cb": args.shortgpt_cb,
                                "nq": args.shortgpt_nq},
        # zero-damage control: undamaged 32L, continued-pretrained on the heal
        # corpus. An ARM, never the anchor (guard G2).
        "full32_dolmino_step25k": {"mmlu": "7B_full32_step25000",
                                   "cb": args.full32_cb,
                                   "nq": args.full32_nq},
    }

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])

    def null_acc(axis, conv):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][conv]
        return nulls[axis]["acc"]

    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    # Bootstrap seed offsets. Distinct from every offset used by Pilot Zero
    # (97*ai+13*xi, ai in {0,1}) and by the step100k pass (ai in 100..102), so
    # no archived cell can be perturbed and no two cells here collide.
    ARM_INDEX = {a: 200 + i for i, a in enumerate(arm_specs)}

    def seed_off(arm, axis):
        return 97 * ARM_INDEX[arm] + 13 * AXES.index(axis)

    # ---- 1. GUARD D1-D6, evaluated BEFORE any NI (guard G1) --------------
    guard = {}
    for conv in TIE_CONVS:
        guard[conv] = {}
        for axis in AXES:
            iv = data["intact_7B_base"][axis]
            nv = (nulls["mmlu_content"]["vectors"][conv]
                  if axis == "mmlu_content" else nulls[axis]["vector"])
            d = np.asarray(iv, float) - np.asarray(nv, float)
            resid = float(d.mean())
            resid_pp = 100.0 * resid
            _m, lo, hi, p = paired_bootstrap(d, seed=SEED + 700
                                             + 13 * AXES.index(axis))
            delta_pp = 100.0 * PREREG["delta_fraction"] * resid
            n = EXPECTED_N["mmlu" if axis == "mmlu_content" else axis]

            # D6: frozen FORMULA of G3; inputs Delta_x and n are per-scale.
            pstar = n * (delta_pp / (100.0 * Z95_TWO_SIDED)) ** 2
            pdisc = {}
            for arm in arm_specs:
                if arm == "intact_7B_base":
                    continue
                av = np.asarray(data[arm][axis], float)
                pdisc[arm] = float((av != np.asarray(iv, float)).mean())
            pdisc_max = max(pdisc.values())
            hw_by_arm = {a: 100.0 * Z95_TWO_SIDED * float(np.sqrt(v / n))
                         for a, v in pdisc.items()}

            d4 = {a: d4_interface_degenerate(data, a, axis, nulls)
                  for a in arm_specs}
            all_below_null = all(reported[a][axis] < null_acc(axis, conv)
                                 for a in arm_specs)

            cond = {
                "D1_residual_negative": bool(resid_pp < 0),
                "D2_residual_at_zero": bool(0 <= resid_pp
                                            <= D2_RESIDUAL_FLOOR_PP),
                "D3_ci_straddles_zero": bool(lo < 0 < hi),
                "D4_null_inadmissible": bool(
                    all_below_null or any(v["degenerate"]
                                          for v in d4.values())),
                "D6_delta_finer_than_instrument": bool(pdisc_max > pstar),
            }
            fatal = [k for k, v in cond.items() if v]
            guard[conv][axis] = {
                "residual_intact_pp": resid_pp,
                "null": float(np.asarray(nv, float).mean()),
                "reported_intact": float(np.asarray(iv, float).mean()),
                "ci95_pp": [100.0 * lo, 100.0 * hi],
                "boot_p": p,
                "delta_pp": delta_pp,
                "n": n,
                "pstar_crit_7B_recomputed": pstar,
                "pstar_crit_1B_prereg_for_reference": PSTAR_CRIT_1B_PREREG[axis],
                "p_disc_by_arm": pdisc,
                "p_disc_max": pdisc_max,
                "hw95_pp_by_arm": hw_by_arm,
                "delta_over_hw_worst": delta_pp / max(hw_by_arm.values()),
                "d4_interface_by_arm": d4,
                "all_arms_below_null": all_below_null,
                "conditions": cond,
                "fatal_conditions": fatal,
                "classification": ("CERTIFIABLE" if not fatal
                                   else "NOT_CERTIFIABLE"),
                "decision_axis": axis not in DEMOTED_AXES,
            }

    # ---- D5: anchor uniqueness (recorded, not fatal) ---------------------
    d5 = {"checked": False}
    if args.d5_cb and args.d5_mm:
        second, _ = _load_arm(mm_root, cb_root,
                              {"mmlu": args.d5_mm, "cb": args.d5_cb})
        d5 = {"checked": True, "second_cb": args.d5_cb,
              "second_mmlu": args.d5_mm, "per_axis": {}}
        for axis in ("triviaqa", "popqa", "mmlu_content"):
            a = np.asarray(data["intact_7B_base"][axis], float)
            b = np.asarray(second[axis], float)
            drift_pp = 100.0 * float(b.mean() - a.mean())
            dd = PREREG["delta_fraction"] * drift_pp
            base_delta = guard["split"][axis]["delta_pp"]
            d5["per_axis"][axis] = {
                "item_flips": int((a != b).sum()),
                "acc_drift_pp": drift_pp,
                "residual_drift_pp": drift_pp,   # the null cancels
                "delta_drift_pp": dd,
                "delta_pp_split": base_delta,
                "drift_fraction_of_delta": (abs(dd) / base_delta
                                            if base_delta else None),
                "D5_fires": bool(base_delta and abs(dd)
                                 >= D5_DRIFT_FRACTION * base_delta),
            }

    # ---- 2. NI, only on cells the guard did not retire (guard G1) --------
    per_conv = {}
    for conv in TIE_CONVS:
        cells, retired = [], []
        for arm in arm_specs:
            if arm == "intact_7B_base":
                continue
            for axis in AXES:
                g = guard[conv][axis]
                if g["classification"] == "NOT_CERTIFIABLE":
                    retired.append({"arm": arm, "axis": axis,
                                    "fatal_conditions": g["fatal_conditions"],
                                    "residual_intact_pp":
                                        g["residual_intact_pp"],
                                    "delta_pp": g["delta_pp"],
                                    "p_disc": g["p_disc_by_arm"][arm],
                                    "pstar_crit": g["pstar_crit_7B_recomputed"],
                                    "ni_run": False,
                                    "note": "NI NOT RUN; excluded from the "
                                            "BH/decision family. Never to be "
                                            "reported as 'NI rejected'."})
                    continue
                r = ni_rule(data[arm][axis], data["intact_7B_base"][axis],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=seed_off(arm, axis))
                arm_resid = reported[arm][axis] - null_acc(axis, conv)
                ir = g["residual_intact_pp"] / 100.0
                cells.append({
                    "arm": arm, "axis": axis,
                    "decision_axis": axis not in DEMOTED_AXES,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact_7B_base"][axis],
                    "null": null_acc(axis, conv),
                    "residual_arm_pp": 100.0 * arm_resid,
                    "residual_intact_pp": g["residual_intact_pp"],
                    "residual_fraction_recovered": (arm_resid / ir
                                                    if ir > 0 else None),
                    "deficit_pp": g["residual_intact_pp"] - 100.0 * arm_resid,
                    "margin_pp": r["diff_lower95_one_sided_pp"]
                    + r["delta_pp"],
                    "delta_over_deficit": (
                        r["delta_pp"] / (g["residual_intact_pp"]
                                         - 100.0 * arm_resid)
                        if (g["residual_intact_pp"] - 100.0 * arm_resid) else None),
                    **r,
                })
        n_dec = sum(1 for c in cells if c["decision_axis"])
        per_conv[conv] = {
            "intact_residual_pp": {x: guard[conv][x]["residual_intact_pp"]
                                   for x in AXES},
            "delta_pp": {x: guard[conv][x]["delta_pp"] for x in AXES},
            "cells": cells,
            "retired_cells": retired,
            "family_size_full": len(arm_specs[list(arm_specs)[0]]) and
            (len(arm_specs) - 1) * len(AXES),
            "family_size_after_guard": len(cells),
            "decision_family_size_full": (len(arm_specs) - 1)
            * len(DECISION_AXES),
            "decision_family_size_after_guard": n_dec,
            "ratio_rule": {a: ratio_rule(reported[a],
                                         reported["intact_7B_base"],
                                         PREREG["rho"],
                                         [x for x in AXES if x in data[a]])
                           for a in arm_specs if a != "intact_7B_base"},
        }

    # ---- 3. The verdict: does NI accept anywhere on >=2 of 3 decision axes?
    verdict = {}
    for conv in TIE_CONVS:
        per_arm = {}
        for arm in arm_specs:
            if arm == "intact_7B_base":
                continue
            dec = [c for c in per_conv[conv]["cells"]
                   if c["arm"] == arm and c["decision_axis"]]
            acc = [c["axis"] for c in dec if c["ni_accept"]]
            n_surv = len(dec)
            # guard G1 clarification (i): thresholds rescale to the surviving
            # decision-axis count, else retiring an axis makes the rule
            # unsatisfiable and therefore unfireable.
            need = int(np.ceil(0.50 * n_surv)) if n_surv else None
            per_arm[arm] = {
                "n_decision_axes_surviving_guard": n_surv,
                "n_decision_axes_accepting": len(acc),
                "axes_accepting": acc,
                "threshold_ge2of3_rescaled": need,
                "NI_OBSERVED_TO_ACCEPT": bool(n_surv and len(acc) >= need
                                              and len(acc) >= 1),
                "all_reject": bool(n_surv and not acc),
            }
        verdict[conv] = per_arm

    # ---- 4. Sensitivity, not point estimates -----------------------------
    # (a) 1B-imported sd_run: LABELLED cross-scale extrapolation only.
    # (b) 7B harness jitter from repeated scoring of the SAME checkpoint --
    #     measurement drift, NOT seed variance.
    sens = {"cross_scale_sd_run_1B_imported": {}, "note": (
        "sd_run is a 1B, S=3, keep12 quantity. There is ONE seed per 7B rung, "
        "so NO 7B sd_run exists. The deficit/sd_run column below is an "
        "explicitly-labelled cross-scale extrapolation and is NOT licensed as "
        "a 7B variance statement.")}
    for conv in ("split",):
        for c in per_conv[conv]["cells"]:
            if not c["decision_axis"]:
                continue
            sd = SD_RUN_1B_PP[c["axis"]]
            sens["cross_scale_sd_run_1B_imported"][
                f"{c['arm']}|{c['axis']}"] = {
                "deficit_pp": c["deficit_pp"],
                "sd_run_1B_pp": sd,
                "deficit_over_sd_run_1B": c["deficit_pp"] / sd,
                "delta_pp": c["delta_pp"],
                "sd_run_needed_to_flip_NI_pp": (
                    abs(c["diff_lower95_one_sided_pp"] + c["delta_pp"])
                    / 2.92 * np.sqrt(3)),
            }
    # NI's own sampling sensitivity: the one-sided bound already IS the
    # pessimistic end for the arm. Report how far the bound is from -Delta in
    # units of the bootstrap SE, so a reader sees the flip distance.
    for conv in TIE_CONVS:
        for c in per_conv[conv]["cells"]:
            se = ((c["diff_mean_pp"] - c["diff_lower95_one_sided_pp"])
                  / 1.6449) if c["diff_mean_pp"] != c[
                      "diff_lower95_one_sided_pp"] else None
            c["bootstrap_se_pp"] = se
            c["se_to_flip_NI"] = (abs(c["diff_lower95_one_sided_pp"]
                                      + c["delta_pp"]) / se) if se else None

    out = {
        "gate": "A04_shallow_rung_NI_discrimination_7B",
        "question": ("does NI(Delta) ever ACCEPT? Tested on the shallowest "
                     "damage in the repository (7B, 16 of 32 layers, 200k "
                     "heal steps), which is already scored on all four axes."),
        "date": "2026-08-12",
        "gpu_spent": 0,
        "gpu_note": ("CPU-only re-analysis of per-example shards already on "
                     "disk. Read-only ssh + scp -O to stage wzc1-only shards "
                     "onto zwfy6. No model was loaded."),
        "prereg": {
            "gate_design": "proposal/active/A04-recovery-certification/"
                           "A04_GATE_DESIGN.md §2",
            "margin_guard": "proposal/active/A04-recovery-certification/"
                            "A04_MARGIN_GUARD_PREREG.md §4",
            "delta_fraction": PREREG["delta_fraction"],
            "rho": PREREG["rho"],
            "commit_freezing_constants": PREREG["commit"],
            "ni_definition": ("accept iff one-sided lower 95% bound on "
                              "residual(arm)-residual(intact) > -Delta, "
                              "Delta = 0.10*residual(intact); imported "
                              "ni_rule, not reimplemented"),
            "decision_axes": DECISION_AXES,
            "demoted_axes": sorted(DEMOTED_AXES),
            "delta_never_substituted": True,
            "anchor_never_changed": True,
        },
        "intact_anchor": {
            "choice": "vanilla models/OLMo-2-1124-7B (mode=base, 32 layers)",
            "dirs": ANCHOR,
            "why": ("A04's 1B anchor (guard G0) is A03_1B_base, whose meta is "
                    "mode=base / base_model=../models/OLMo-2-0425-1B -- the "
                    "VANILLA base. The 7B analogue is therefore the vanilla "
                    "OLMo-2-1124-7B, not full32_step25000, which is "
                    "mode=pruned keep_front_layers=32 continued-pretrained on "
                    "the heal corpus for 25k steps."),
            "full32_is_not_the_anchor": (
                "full32_step25000 is carried as an ARM (zero-damage control) "
                "and as a labelled diagnostic. Guard G2 forbids changing the "
                "anchor, so it can never alter a verdict."),
        },
        "arms": {a: prov[a] for a in prov},
        "integrity": integrity,
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "guard_D5_anchor_uniqueness": d5,
        "per_convention": per_conv,
        "verdict_by_convention": verdict,
        "sensitivity": sens,
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---- console report --------------------------------------------------
    print("=" * 100)
    print("GUARD D1-D6 (pre-registered `split` convention), evaluated BEFORE NI")
    print("=" * 100)
    print(f"{'axis':<14}{'resid_intact':>13}{'Delta':>9}{'n':>8}"
          f"{'p*crit_7B':>11}{'p_disc_max':>11}{'D/hw':>7}  class")
    for axis in AXES:
        g = guard["split"][axis]
        print(f"{axis:<14}{g['residual_intact_pp']:>12.4f}pp"
              f"{g['delta_pp']:>8.4f}{g['n']:>8}"
              f"{g['pstar_crit_7B_recomputed']:>11.4f}"
              f"{g['p_disc_max']:>11.4f}{g['delta_over_hw_worst']:>7.2f}"
              f"  {g['classification']}"
              + (f"  <- {','.join(g['fatal_conditions'])}"
                 if g["fatal_conditions"] else ""))
    print()
    print("=" * 100)
    print("NI(Delta) on the shallow rungs -- `split` convention")
    print("=" * 100)
    print(f"{'arm':<26}{'axis':<14}{'reported':>9}{'recov%':>8}"
          f"{'deficit':>10}{'lo95':>11}{'Delta':>8}{'margin':>10}  NI")
    for c in per_conv["split"]["cells"]:
        rf = c["residual_fraction_recovered"]
        print(f"{c['arm']:<26}{c['axis']:<14}{100*c['reported']:>8.3f}%"
              f"{100*rf if rf is not None else float('nan'):>7.1f}%"
              f"{c['deficit_pp']:>10.4f}"
              f"{c['diff_lower95_one_sided_pp']:>11.4f}"
              f"{c['delta_pp']:>8.4f}{c['margin_pp']:>10.4f}"
              f"  {'ACCEPT' if c['ni_accept'] else 'REJECT'}"
              + ("" if c["decision_axis"] else "  (demoted)"))
    for r in per_conv["split"]["retired_cells"]:
        print(f"{r['arm']:<26}{r['axis']:<14}"
              f"{'NOT_CERTIFIABLE -- NI NOT RUN':>48}"
              f"  <- {','.join(r['fatal_conditions'])}")
    print()
    print(f"decision family: {per_conv['split']['decision_family_size_full']}"
          f" full -> {per_conv['split']['decision_family_size_after_guard']}"
          f" after guard")
    print()
    print("=" * 100)
    print("VERDICT (`split`): is NI ever OBSERVED TO ACCEPT?")
    print("=" * 100)
    for arm, v in verdict["split"].items():
        print(f"  {arm:<26} surviving={v['n_decision_axes_surviving_guard']} "
              f"accepting={v['n_decision_axes_accepting']} "
              f"{v['axes_accepting']} -> "
              f"{'ACCEPTS' if v['NI_OBSERVED_TO_ACCEPT'] else 'ALL REJECT'}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
