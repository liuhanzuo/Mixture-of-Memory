#!/usr/bin/env python3
"""A04 — PLATEAU-vs-NI at the repaired rule's OWN earliest accept checkpoint.

WHAT DEFECT THIS CLOSES
-----------------------
`A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md` §1.4 / §5 item 1:

  * The repaired PLATEAU rule R3 (`rate_5k = 100*(1-(ppl_c/ppl_prev)**(5000/d))
    < T`, T = 2.0 %/5k, pre-registered git `d1ba737`) FIRST ACCEPTS at step
    100 000, not step 200 000.
  * Pilot Zero scored capability axes ONLY at step 200 000.
  * So the PLATEAU-vs-NI cell at the rule's own earliest accept point was
    UNMEASURED, not resolved — and there is 4.6386 % further relative PPL
    improvement between 100k and 200k, so it is not a negligible relocation.
  * The document states: "any claim about *where* the earliest disagreement lies
    now requires step-100 000 capability scoring, which is GPU work and is not
    done here."

That GPU work is now done (`code/a04_step100k_axes_driver.sh`, `.73`, 8×H20).
This script consumes it.

WHAT IT DOES NOT DO
-------------------
It does NOT re-derive PLATEAU or NI. Both rules, the nulls, the scorers, the
bootstrap protocol and every pre-registered constant are IMPORTED from the
existing A04 / A03 code. Nothing here is a reimplementation:

    plateau rate_5k ....... reimplemented is FORBIDDEN -> asserted equal to
                            evidence/a04_plateau_rule_repair.json
    ni_rule ............... imported from pilot_zero_rule_disagreement
    nulls / scorers ....... imported from A03 analyze_1b_knowledge_floor
                            (via pilot_zero's build_nulls / build_axis_data)
    T, rho, Delta ......... imported PREREG dict (no CLI override exists)

REGRESSION GUARD (load-bearing)
-------------------------------
Before reporting anything new, the script recomputes the ARCHIVED step-200000
cells through the same code path and asserts they reproduce
`evidence/pilot_zero_rule_disagreement.json` to 1e-9 pp on all four axes. If the
old cells do not reproduce, the new cell is not comparable and the script exits
non-zero rather than publishing a number. This is the same discipline §3 of the
repair doc used ("the sweep's 0.10 column reproduces the pilot's cells
bit-for-bit").

INTEGRITY
---------
`build_axis_data` -> `load_shards` hard-asserts, per cell: exactly 8 shard files,
every shard index 0..7 present exactly once, no duplicate `item_id` after merge,
and the exact expected item count. `nan` rows are separately asserted absent
here (the harness drops them from its own accuracies, so a nonzero count would
break the identical-valid-item-set assumption `paired`/`ni_rule` depend on) and
cross-arm `item_id` alignment is asserted so the paired difference is really
item-paired. ⚠️ `nq_open` lives in a SEPARATE result dir suffixed `_nq`, never
alongside triviaqa/popqa; the arm spec below reflects that.

CPU ONLY. No GPU, no model load, no torch.

Usage (on a zwfy6 node, where the per-example dumps live):
  python a04_step100k_plateau_vs_ni.py \
    --raw_root  /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
    --ppl_json  <A04 evidence>/a04_1b_keep7f2_ppl_trajectory.json \
    --repair_json <A04 evidence>/a04_plateau_rule_repair.json \
    --pilot_zero_json <A04 evidence>/pilot_zero_rule_disagreement.json \
    --out_json  <A04 evidence>/a04_step100k_plateau_vs_ni.json
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

# Everything decision-relevant is IMPORTED. `pilot_zero_rule_disagreement` in
# turn imports A03's canonical scorers/nulls through proposal_paths, so there is
# exactly one definition of each quantity in play.
from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_axis_data,
    build_nulls,
    ni_rule,
    ratio_rule,
)
from analyze_1b_knowledge_floor import TIE_CONVS  # noqa: E402

# The pre-registered PLATEAU threshold, and the repaired rule's identity. T is
# NOT redefined here -- it is read out of the imported PREREG dict so a future
# edit to the prereg cannot silently diverge from this analysis.
T_PCT_PER_5K = PREREG["T_pct_per_5k"]

# The checkpoint the repair identifies as R3's earliest accept, and the one
# Pilot Zero actually measured. Both are FACTS ABOUT THE RULE, re-derived from
# the PPL trajectory below and cross-checked against the repair JSON.
STEP_EARLIEST_EXPECTED = 100000
STEP_PILOT_MEASURED = 200000

# Newly scored cells (this pass). Result-dir naming mirrors the baseline exactly.
NEW_STEPS = (100000, 150000, 50000)


def rate_5k(ppl_c, ppl_prev, d):
    """R3's per-5k GEOMETRIC improvement rate, in percent.

    Asserted below against `evidence/a04_plateau_rule_repair.json` rather than
    trusted: the repair document is the authority on this arithmetic and a
    second independent implementation of it is exactly the kind of drift this
    repo has been bitten by.
    """
    return 100.0 * (1.0 - (ppl_c / ppl_prev) ** (5000.0 / d))


def plateau_r3(ppl_traj, T):
    """R3 verdict at every checkpoint of the trajectory."""
    out = []
    for i, (step, ppl) in enumerate(ppl_traj):
        if i == 0:
            out.append({"step": step, "ppl": ppl, "interval_steps": None,
                        "rel_improve_pct": None, "rate_5k_pct": None,
                        "R3_accept": None, "R1_accept": None})
            continue
        pstep, pppl = ppl_traj[i - 1]
        d = step - pstep
        rel = 100.0 * (pppl - ppl) / pppl
        r = rate_5k(ppl, pppl, d)
        out.append({
            "step": step, "ppl": ppl, "interval_steps": d,
            "rel_improve_pct": rel, "rate_5k_pct": r,
            "R3_accept": bool(r < T),
            # R1 = the pilot's pre-registered (unscaled) reading, kept so the
            # relocation of the earliest accept is visible in one table.
            "R1_accept": bool(rel < T),
        })
    return out


def assert_nan_free_and_aligned(data, prov):
    """Per-cell nan assertion + cross-arm item_id alignment.

    `load_shards` already asserts 8/8 shards, no duplicate item_id, and the exact
    item count. Two things it does not do, both required before a PAIRED
    difference means anything:

      * a `nan:true` row merged as a real score (the harness excludes such items
        from its own accuracies, so its presence means the arms no longer share
        one valid item set);
      * the arms actually covering the SAME item_ids -- `load_shards` sorts by
        item_id, so a mismatched set would pair item k of one arm against a
        different item k of another and produce a silently wrong difference.
    """
    report = {}
    ref_ids = {}
    for arm in data:
        report[arm] = {}
        for axis in AXES:
            key = {"mmlu_content": "_mmlu_rows"}.get(axis, f"_{axis}_rows")
            if key not in data[arm]:
                continue
            rows = data[arm][key]
            n_nan = sum(1 for r in rows if r.get("nan"))
            if n_nan:
                raise SystemExit(
                    f"FATAL {arm}/{axis}: {n_nan} rows with nan=true -- paired "
                    "analysis requires an identical valid item set across arms")
            ids = [r["item_id"] for r in rows]
            if len(set(ids)) != len(ids):
                raise SystemExit(f"FATAL {arm}/{axis}: duplicate item_id")
            if len(ids) != EXPECTED_N[
                    "mmlu" if axis == "mmlu_content" else axis]:
                raise SystemExit(f"FATAL {arm}/{axis}: n={len(ids)} unexpected")
            if axis not in ref_ids:
                ref_ids[axis] = ids
            elif ids != ref_ids[axis]:
                raise SystemExit(
                    f"FATAL {arm}/{axis}: item_id sequence differs from the "
                    "reference arm -- the paired difference would compare "
                    "different items")
            report[arm][axis] = {"n": len(ids), "n_nan": 0, "shards": 8,
                                 "item_ids_aligned_with_intact": True,
                                 "dir": prov[arm].get(
                                     "mmlu" if axis == "mmlu_content"
                                     else axis, {}).get("dir")}
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--ppl_json", required=True)
    ap.add_argument("--repair_json", required=True)
    ap.add_argument("--pilot_zero_json", required=True)
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    # ---- 1. PLATEAU R3 on the real trajectory, cross-checked --------------
    ppl_traj = [(int(s), float(p)) for s, p in json.load(open(args.ppl_json))]
    r3 = plateau_r3(ppl_traj, T_PCT_PER_5K)
    repair = json.load(open(args.repair_json))

    # Cross-check every rate_5k against the repair JSON rather than trusting a
    # second implementation of the same formula.
    rep_rows = {int(r["step"]): r for r in repair["trajectory_per_checkpoint"]}
    xchecks = []
    for row in r3:
        if row["rate_5k_pct"] is None:
            continue
        s = row["step"]
        if s not in rep_rows:
            raise SystemExit(f"FATAL: step {s} absent from repair JSON")
        got = rep_rows[s]
        ref = None
        for k in ("rate_5k_pct", "rate_5k", "R3_rate_5k_pct"):
            if k in got:
                ref = float(got[k]); break
        if ref is None:
            raise SystemExit(
                f"FATAL: repair JSON row for step {s} has no rate_5k field "
                f"(keys={sorted(got)}) -- refusing to guess")
        d = abs(ref - row["rate_5k_pct"])
        if d > 1e-9:
            raise SystemExit(
                f"FATAL: rate_5k mismatch at step {s}: this script "
                f"{row['rate_5k_pct']!r} vs repair JSON {ref!r} (|d|={d:.3e})")
        xchecks.append({"step": s, "rate_5k_pct": ref, "abs_diff": d})

    accepts = [r["step"] for r in r3 if r["R3_accept"] is True]
    earliest_r3 = min(accepts) if accepts else None
    if earliest_r3 != STEP_EARLIEST_EXPECTED:
        raise SystemExit(
            f"FATAL: R3's earliest accept recomputes to {earliest_r3}, but the "
            f"repair document says {STEP_EARLIEST_EXPECTED}. The defect being "
            "closed is defined relative to that checkpoint; refusing to "
            "proceed on a different one.")
    r1_accepts = [r["step"] for r in r3 if r["R1_accept"] is True]

    # ---- 2. load every arm, with hard integrity assertions ----------------
    arm_dirs = {
        "intact": {"mmlu": "A03_1B_base", "cb": "A03_1B_base",
                   "nq": "A03_1B_base_nq"},
        # the archived, already-published cell -- loaded for the regression guard
        "keep7f2_step200000": {"mmlu": "A03_1B_keep7_step200k",
                               "cb": "A03_1B_keep7_step200k",
                               "nq": "A03_1B_keep7_step200k_nq"},
    }
    for s in NEW_STEPS:
        tag = f"A04_1B_keep7f2_step{s}"
        arm_dirs[f"keep7f2_step{s}"] = {"mmlu": tag, "cb": tag,
                                        "nq": f"{tag}_nq"}

    present, missing = {}, []
    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")
    for arm, spec in arm_dirs.items():
        ok = (os.path.isdir(os.path.join(mm_root, spec["mmlu"]))
              and os.path.isdir(os.path.join(cb_root, spec["cb"]))
              and os.path.isdir(os.path.join(cb_root, spec["nq"])))
        (present.setdefault(arm, spec) if ok else missing.append(arm))
    for req in ("intact", "keep7f2_step200000",
                f"keep7f2_step{STEP_EARLIEST_EXPECTED}"):
        if req not in present:
            raise SystemExit(
                f"FATAL: required arm {req} has no result dirs under "
                f"{args.raw_root}. The whole point of this pass is the "
                f"step-{STEP_EARLIEST_EXPECTED} cell; refusing to report "
                "a partial answer as if it were the answer.")

    data, prov = build_axis_data(args.raw_root, present)
    integrity = assert_nan_free_and_aligned(data, prov)
    nulls = build_nulls(data["intact"])

    def null_acc(axis, conv):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][conv]
        return nulls[axis]["acc"]

    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in present}

    # Bootstrap seed offsets MUST match Pilot Zero's, or the archived cells
    # cannot reproduce. Pilot Zero used seed_off = 97*ai + 13*xi with ai the
    # index of the arm in ITS arm_dirs dict (intact=0, keep7f2_step200000=1) and
    # xi the index in AXES. Replicated exactly for the two archived arms; new
    # arms get offsets that cannot collide with those.
    PZ_ARM_INDEX = {"intact": 0, "keep7f2_step200000": 1}
    NEW_ARM_INDEX = {f"keep7f2_step{s}": 100 + i
                     for i, s in enumerate(NEW_STEPS)}

    def seed_off_for(arm, axis):
        ai = PZ_ARM_INDEX.get(arm, NEW_ARM_INDEX.get(arm))
        if ai is None:
            raise SystemExit(f"FATAL: no seed offset defined for arm {arm}")
        return 97 * ai + 13 * AXES.index(axis)

    per_conv = {}
    for conv in TIE_CONVS:
        intact_resid = {x: reported["intact"][x] - null_acc(x, conv)
                        for x in AXES}
        cells = []
        for arm in present:
            if arm == "intact":
                continue
            for axis in AXES:
                if axis not in data[arm]:
                    continue
                r = ni_rule(data[arm][axis], data["intact"][axis],
                            PREREG["delta_fraction"], intact_resid[axis],
                            seed_off=seed_off_for(arm, axis))
                arm_resid = reported[arm][axis] - null_acc(axis, conv)
                cells.append({
                    "arm": arm, "axis": axis,
                    "demoted_descriptive_only": axis in DEMOTED_AXES,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact"][axis],
                    "null": null_acc(axis, conv),
                    "residual_arm_pp": 100.0 * arm_resid,
                    "residual_intact_pp": 100.0 * intact_resid[axis],
                    "residual_fraction_recovered": (
                        arm_resid / intact_resid[axis]
                        if intact_resid[axis] > 0 else None),
                    "delta_degenerate_negative_margin": bool(
                        intact_resid[axis] <= 0),
                    **r,
                })
        ratio = {a: ratio_rule(reported[a], reported["intact"], PREREG["rho"],
                               [x for x in AXES if x in data[a]])
                 for a in present if a != "intact"}
        per_conv[conv] = {"intact_residual_pp": {x: 100.0 * intact_resid[x]
                                                 for x in AXES},
                          "delta_pp": {x: 100.0 * PREREG["delta_fraction"]
                                       * intact_resid[x] for x in AXES},
                          "cells": cells, "ratio_rule": ratio}

    # ---- 3. REGRESSION GUARD: the archived step-200000 cells must reproduce -
    pz = json.load(open(args.pilot_zero_json))
    pz_cells = {(c["arm"], c["axis"]): c
                for c in pz["per_convention"]["split"]["cells"]}
    regression = []
    for c in per_conv["split"]["cells"]:
        if c["arm"] != "keep7f2_step200000":
            continue
        old = pz_cells.get((c["arm"], c["axis"]))
        if old is None:
            raise SystemExit(f"FATAL: archived cell {c['arm']}/{c['axis']} "
                             "absent from pilot_zero json")
        row = {"axis": c["axis"]}
        worst = 0.0
        for k in ("diff_mean_pp", "diff_lower95_one_sided_pp", "delta_pp"):
            d = abs(float(old[k]) - float(c[k]))
            row[k] = {"archived": float(old[k]), "recomputed": float(c[k]),
                      "abs_diff": d}
            worst = max(worst, d)
        row["ni_accept_archived"] = bool(old["ni_accept"])
        row["ni_accept_recomputed"] = bool(c["ni_accept"])
        row["max_abs_diff_pp"] = worst
        row["reproduces"] = bool(worst < 1e-9
                                 and old["ni_accept"] == c["ni_accept"])
        regression.append(row)
    if not regression or not all(r["reproduces"] for r in regression):
        raise SystemExit(
            "FATAL: the archived step-200000 cells do NOT reproduce through "
            "this code path:\n"
            + json.dumps(regression, indent=1)
            + "\nA new cell computed by a path that cannot reproduce the old "
              "one is not comparable to it. Refusing to publish a number.")

    # ---- 4. the actual question -------------------------------------------
    def verdict_at(step, conv="split"):
        arm = f"keep7f2_step{step}"
        cs = [c for c in per_conv[conv]["cells"]
              if c["arm"] == arm and not c["demoted_descriptive_only"]]
        if not cs:
            return None
        n_rej = sum(1 for c in cs if not c["ni_accept"])
        # PLATEAU is DEFINED only where an in-domain val PPL exists on disk.
        # `olmo2_ppl_results/` carries the keep7f2 trajectory at steps
        # {50000, 100000, 147000, 200000} and NOTHING at 150000, so R3 is
        # UNDEFINED at step 150000 -- exactly the reasoning Pilot Zero applied
        # to cpt20k/arm4_peaklr20k. And step 50000 is the FIRST trajectory
        # point, so it has no preceding interval and R3 is undefined there too.
        # Reporting either as though PLATEAU had a verdict would invent a rule
        # evaluation that no measurement supports, so both are marked
        # plateau_defined=False and CANNOT contribute a disagreement.
        row = next((r for r in r3 if r["step"] == step), None)
        plateau_defined = bool(row is not None and row["R3_accept"] is not None)
        return {
            "step": step, "arm": arm, "convention": conv,
            "plateau_defined": plateau_defined,
            "plateau_undefined_reason": (
                None if plateau_defined else
                ("step is the first point of the PPL trajectory, so it has no "
                 "preceding interval" if row is not None else
                 "no in-domain val PPL exists on disk for this step "
                 "(olmo2_ppl_results/ has the keep7f2 trajectory only at "
                 "steps 50000/100000/147000/200000)")),
            "plateau_R3_accepts": (
                None if not plateau_defined else bool(row["R3_accept"])),
            "plateau_R3_rate_5k_pct": (
                None if row is None else row["rate_5k_pct"]),
            "n_decision_axes": len(cs),
            "n_ni_reject": n_rej,
            "n_ni_accept": len(cs) - n_rej,
            "ni_reject_axes": [c["axis"] for c in cs if not c["ni_accept"]],
            # A disagreement requires BOTH rules to have a verdict. Where
            # PLATEAU is undefined there is nothing for NI to disagree WITH.
            "rules_disagree": bool(plateau_defined and row["R3_accept"]
                                   and n_rej > 0),
            "per_axis": {c["axis"]: {
                "reported": c["reported"],
                "residual_arm_pp": c["residual_arm_pp"],
                "residual_fraction_recovered": c["residual_fraction_recovered"],
                "diff_mean_pp": c["diff_mean_pp"],
                "diff_lower95_one_sided_pp": c["diff_lower95_one_sided_pp"],
                "delta_pp": c["delta_pp"],
                "ni_accept": c["ni_accept"],
                # how far from accepting, in units of Delta -- the quantity that
                # says whether the disagreement is marginal or overwhelming
                "reject_margin_multiple_of_delta": (
                    None if c["delta_pp"] <= 0 else
                    abs(c["diff_lower95_one_sided_pp"]) / c["delta_pp"]),
            } for c in cs},
            "demoted_nq_open": {
                c["axis"]: {"diff_lower95_one_sided_pp":
                            c["diff_lower95_one_sided_pp"],
                            "delta_pp": c["delta_pp"],
                            "ni_accept": c["ni_accept"]}
                for c in per_conv[conv]["cells"]
                if c["arm"] == arm and c["demoted_descriptive_only"]},
        }

    steps_measured = sorted(int(a.replace("keep7f2_step", ""))
                            for a in present if a.startswith("keep7f2_step"))
    verdicts = {str(s): verdict_at(s) for s in steps_measured}
    verdicts_all_conv = {
        conv: {str(s): verdict_at(s, conv) for s in steps_measured}
        for conv in TIE_CONVS}

    v_earliest = verdicts[str(STEP_EARLIEST_EXPECTED)]
    v_pilot = verdicts[str(STEP_PILOT_MEASURED)]

    # Where does the earliest disagreement now lie? Over the checkpoints where
    # BOTH rules have a verdict (i.e. an in-domain PPL exists AND R3 accepts
    # there AND capability is measured).
    both_defined = [s for s in steps_measured
                    if verdicts[str(s)]["plateau_defined"]
                    and verdicts[str(s)]["plateau_R3_accepts"]]
    plateau_undefined = {
        str(s): verdicts[str(s)]["plateau_undefined_reason"]
        for s in steps_measured if not verdicts[str(s)]["plateau_defined"]}
    disagree_steps = [s for s in both_defined if verdicts[str(s)]["rules_disagree"]]
    earliest_disagreement = min(disagree_steps) if disagree_steps else None

    claim_moves = bool(earliest_disagreement is not None
                       and earliest_disagreement != STEP_PILOT_MEASURED)

    out = {
        "what": ("A04 — PLATEAU(T) vs NI(Delta) at the REPAIRED rule R3's own "
                 "earliest accept checkpoint (step 100 000), which "
                 "A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md §1.4/§5 recorded "
                 "as UNMEASURED."),
        "closes_defect": {
            "document": "A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md",
            "sections": ["1.4", "5 item 1"],
            "verbatim": ("any claim about *where* the earliest disagreement "
                         "lies now requires step-100 000 capability scoring, "
                         "which is GPU work and is not done here"),
        },
        "preregistration": PREREG,
        "preregistration_note": (
            "T, rho and Delta are IMPORTED from pilot_zero_rule_disagreement's "
            "frozen PREREG dict (git d1ba737) and have no CLI override. The "
            "choice of rule reading R3 over R1/R2 is POST-HOC (2026-08-10, "
            "disclosed in the repair doc §1.3) and is NOT re-litigated here."),
        "plateau_rule_in_force": {
            "name": "R3 (repaired)",
            "formula": "rate_5k = 100*(1-(ppl_c/ppl_prev)**(5000/d)) < T",
            "T_pct_per_5k": T_PCT_PER_5K,
            "trajectory": r3,
            "cross_check_against_repair_json": {
                "file": os.path.abspath(args.repair_json),
                "rows": xchecks,
                "max_abs_diff": max((x["abs_diff"] for x in xchecks),
                                    default=0.0),
                "tolerance": 1e-9,
            },
            "R3_accept_steps": accepts,
            "R3_earliest_accept": earliest_r3,
            "R1_accept_steps": r1_accepts,
            "R1_earliest_accept": min(r1_accepts) if r1_accepts else None,
        },
        "integrity_asserts": {
            "per_cell": integrity,
            "asserted": [
                "exactly 8 shard files per (arm, axis)",
                "every shard index 0..7 present exactly once",
                "no duplicate item_id after merge",
                "exact expected item count "
                "(mmlu 14042 / triviaqa 17944 / popqa 14267 / nq_open 3610)",
                "zero rows with nan=true",
                "item_id sequence identical across all arms per axis "
                "(so the paired difference is genuinely item-paired)",
            ],
            "expected_item_counts": EXPECTED_N,
            "nq_open_dir_note": (
                "nq_open per-example files live in a SEPARATE result dir "
                "suffixed `_nq`, never alongside triviaqa/popqa."),
        },
        "regression_guard_archived_step200000_cells": {
            "why": ("A new cell computed by a code path that cannot reproduce "
                    "the published one is not comparable to it."),
            "source": os.path.abspath(args.pilot_zero_json),
            "rows": regression,
            "all_reproduce": True,
            "tolerance_pp": 1e-9,
        },
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "null_invariance_note": (
            "residual(arm)-residual(intact) = reported(arm)-reported(intact): "
            "the same input-blind null applies to both arms on the same item "
            "set, so it cancels EXACTLY in the difference. The tie convention "
            "therefore moves only Delta (=0.10*residual(intact)), never the "
            "measured difference."),
        "steps_scored_this_pass": [s for s in steps_measured
                                   if s in NEW_STEPS],
        "per_convention": per_conv,
        "verdicts_preregistered_split_convention": verdicts,
        "verdicts_all_five_conventions": verdicts_all_conv,
        "ANSWER": {
            "question": ("at step 100 000 — R3's own earliest accept — do "
                         "PLATEAU and NI agree or disagree?"),
            "step": STEP_EARLIEST_EXPECTED,
            "plateau_R3_accepts": v_earliest["plateau_R3_accepts"],
            "plateau_R3_rate_5k_pct": v_earliest["plateau_R3_rate_5k_pct"],
            "n_ni_reject_of_decision_axes":
                f"{v_earliest['n_ni_reject']}/{v_earliest['n_decision_axes']}",
            "ni_reject_axes": v_earliest["ni_reject_axes"],
            "rules_disagree": v_earliest["rules_disagree"],
            "same_at_step200000": {
                "plateau_R3_accepts": v_pilot["plateau_R3_accepts"],
                "n_ni_reject_of_decision_axes":
                    f"{v_pilot['n_ni_reject']}/{v_pilot['n_decision_axes']}",
                "rules_disagree": v_pilot["rules_disagree"],
            },
            "earliest_disagreement_step": earliest_disagreement,
            "earliest_disagreement_claim_moves": claim_moves,
            "steps_where_plateau_is_undefined": plateau_undefined,
            "steps_where_both_rules_have_a_verdict": both_defined,
            "claim_movement": (
                f"the earliest MEASURED PLATEAU-vs-NI disagreement moves from "
                f"step {STEP_PILOT_MEASURED} to step {earliest_disagreement}"
                if claim_moves else
                f"the earliest disagreement stays at step "
                f"{earliest_disagreement}"),
            "disagreement_unanimous_across_five_conventions": {
                conv: verdicts_all_conv[conv][
                    str(STEP_EARLIEST_EXPECTED)]["rules_disagree"]
                for conv in TIE_CONVS},
        },
        "WHAT_THIS_DOES_NOT_ESTABLISH": [
            "K1's >=24-cell denominator. Unchanged and still INDETERMINATE "
            "(retraction b93247f).",
            "K2. Unchanged; see the Stage-B seed work.",
            "That keep7+fresh2 is anything other than a CONSTANT-REJECT rung. "
            "It is one, and these cells reinforce that rather than rescue it: "
            "a rung where NI can never accept cannot demonstrate that a rule "
            "DISCRIMINATES, only that the two rules differ somewhere.",
            # R1's earliest accept is DERIVED, never asserted. This string used
            # to hardcode "step 200 000", which was true on the 4-point PPL grid
            # {50k,100k,147k,200k} but became FALSE the moment a 150 000 point
            # was added: R1 compares an interval-length-dependent quantity to a
            # per-5k threshold, so the short d=3000 interval 147k->150k yields
            # rel=0.1357 % < T and R1 accepts at 150 000. That is R1's
            # grid-dependence defect (the very thing R3 repairs) showing up as a
            # relocation of its own first accept, so it must be read off the
            # trajectory rather than frozen in prose.
            "Any claim about the pre-registered reading R1. R1's earliest "
            f"accept on this PPL grid is step {min(r1_accepts) if r1_accepts else None} "
            "and is unaffected by these capability cells (R1 is a function of "
            "the PPL trajectory alone). NOTE: unlike R3, R1's first accept is "
            "GRID-DEPENDENT -- adding a checkpoint with a short preceding "
            "interval can move it earlier without any change to the underlying "
            "run, which is exactly the defect the repair documents.",
        ],
        "provenance": {
            "raw_root": os.path.abspath(args.raw_root),
            "ppl_json": os.path.abspath(args.ppl_json),
            "repair_json": os.path.abspath(args.repair_json),
            "pilot_zero_json": os.path.abspath(args.pilot_zero_json),
            "arm_dirs": present,
            "arms_absent": missing,
            "gpu_driver": "code/a04_step100k_axes_driver.sh",
            "per_cell_dirs": prov,
        },
    }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {args.out_json}")

    # ---- console summary ---------------------------------------------------
    print(f"\nPLATEAU R3 (T={T_PCT_PER_5K} %/5k) on the real trajectory:")
    for r in r3:
        if r["rate_5k_pct"] is None:
            print(f"  step {r['step']:>7d}  ppl {r['ppl']:.6f}   (no preceding interval)")
        else:
            print(f"  step {r['step']:>7d}  ppl {r['ppl']:.6f}  d={r['interval_steps']:>6d}  "
                  f"rel={r['rel_improve_pct']:7.4f}%  rate_5k={r['rate_5k_pct']:7.5f}%/5k  "
                  f"R3={'ACCEPT' if r['R3_accept'] else 'reject'}  "
                  f"R1={'ACCEPT' if r['R1_accept'] else 'reject'}")
    print(f"\nregression guard on archived step200000 cells: "
          f"max |diff| = {max(r['max_abs_diff_pp'] for r in regression):.3e} pp "
          f"-> all reproduce")
    for s in steps_measured:
        v = verdicts[str(s)]
        if not v["plateau_defined"]:
            p = "R3=UNDEFINED (no in-domain PPL at this step)"
        else:
            p = (f"R3={'ACCEPT' if v['plateau_R3_accepts'] else 'reject'} "
                 f"(rate_5k={v['plateau_R3_rate_5k_pct']:.5f})")
        print(f"\nstep {s}  {p}  "
              f"NI rejects {v['n_ni_reject']}/{v['n_decision_axes']}  "
              f"disagree={v['rules_disagree']}")
        for ax, a in v["per_axis"].items():
            mult = a["reject_margin_multiple_of_delta"]
            fr = a["residual_fraction_recovered"]
            fr_s = "n/a" if fr is None else f"{100.0 * fr:.2f}%"
            print(f"    {ax:>13}  rep={a['reported']:.6f}  resid={a['residual_arm_pp']:7.4f}pp  "
                  f"frac={fr_s:>7}  "
                  f"lo95={a['diff_lower95_one_sided_pp']:9.4f}pp  D={a['delta_pp']:.4f}pp  "
                  f"NI={'accept' if a['ni_accept'] else 'REJECT'}"
                  + ("" if mult is None else f"  ({mult:.2f}x Delta)"))
    print(f"\nANSWER: earliest measured disagreement -> step "
          f"{earliest_disagreement}; claim moves = {claim_moves}")


if __name__ == "__main__":
    main()
