#!/usr/bin/env python3
"""A04 margin-sensitivity sweep — is the K1-shaped disagreement an artefact of
the arbitrary-looking `Delta = 0.10 * residual(intact)` fraction?

A reviewer's live objection: 0.10 is unmotivated. If the "NI rejects where
PLATEAU accepts" conclusion only holds because the margin was set narrow, the
conclusion is about the margin, not about the rules. This script answers that
by re-running the SAME NI(Delta) comparison over Delta fractions
0.10 -> 0.66 (0.02 steps, 29 points) on the SAME per-example dumps and
reporting where — if anywhere — any decision axis flips from REJECT to ACCEPT.

WHAT THIS IS NOT: it is NOT a re-registration of Delta. The pre-registered
Delta fraction stays 0.10 (git d1ba737); A04_MARGIN_GUARD_PREREG.md rule G2
prohibits substituting it. This is a *sensitivity report on the pre-registered
number*, which is the standard robustness answer to "your threshold is
arbitrary", and it can only ever weaken A04's claim (a wider margin makes NI
easier to accept, i.e. makes the disagreement harder to find). Reporting the
crossing point is therefore conservative, not self-serving.

Scope of the fraction range. The PRIMARY range 0.10 -> 0.66 was specified in the
task assignment *before* this script existed, so it is not a range I chose after
seeing the answer. 0.10 is the pre-registered value. The sweep is EXTENDED to
1.00 for one reason only: the assignment says "if it flips somewhere, report the
crossing point", and two of the three decision axes have their crossing above
0.66, so a sweep that stopped at 0.66 could not report them. Every fraction
above 0.66 is flagged `in_assigned_range: false`.

Honest note on a pre-run estimate that was WRONG. Before running the bootstrap I
estimated the MMLU-content crossing at 0.610 from the point estimate
(6.2455 / 10.2389). The actual crossing uses the one-sided lower 95% bound, not
the point estimate, and is 0.6642 under the pre-registered `split` convention --
i.e. 0.0042 ABOVE the assigned range's upper endpoint. Had the sweep stopped at
0.66 it would have reported "invariant" for `split` while sitting 0.4 pp of
fraction from a flip. That near-miss is reported here rather than buried; it is
also why the extension to 1.00 exists.

Scorers/nulls are IMPORTED from A03's canonical
`analyze_1b_knowledge_floor.py`, never reimplemented.

CPU ONLY. Read-only on all inputs. No GPU, no model load, no torch.

Usage:
  python a04_margin_sensitivity_sweep.py --raw_root <dir> --out_json <file>
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_A03_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "A03-parametric-vs-external-memory", "code"))
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    best_constant_qa,
    longest_option_vector,
)

# ---- pre-registered, frozen by git d1ba737 (NOT changed by this script) ----
PREREG_DELTA_FRACTION = 0.10

# ---- sweep grid, declared before the data is touched ----------------------
# ASSIGNED range (given before this script existed): 0.10 -> 0.66.
# EXTENDED to 1.00 so that crossings above 0.66 can be reported at all; every
# such point is flagged in the output. Nothing below 0.66 changes.
ASSIGNED_LO, ASSIGNED_HI = 0.10, 0.66
SWEEP_LO, SWEEP_HI, SWEEP_STEP = 0.10, 1.00, 0.02

EXPECTED_N = {"mmlu": 14042, "triviaqa": 17944, "popqa": 14267,
              "nq_open": 3610}
# design §5.2 demotes NQ-open: its item-level CI half-width already exceeds its
# own Delta at the pre-registered fraction, so it is descriptive only.
DEMOTED_AXES = {"nq_open"}
DECISION_AXES = ["triviaqa", "popqa", "mmlu_content"]
ALL_AXES = ["triviaqa", "popqa", "mmlu_content", "nq_open"]

# Fixed bootstrap seed offsets, copied verbatim from
# pilot_zero_rule_disagreement.py so the pre-registered-fraction column of this
# sweep is bit-comparable with the pilot's own cells.
SEED_OFF = {"triviaqa": 97, "popqa": 110, "mmlu_content": 123, "nq_open": 136}


def load_shards(d, stem, expected_n, n_shards=8):
    """Merge per_example_<stem>_shard{0..7}of8.jsonl with hard assertions."""
    pat = os.path.join(d, f"per_example_{stem}_shard*of{n_shards}.jsonl")
    files = sorted(glob.glob(pat),
                   key=lambda p: int(os.path.basename(p)
                                     .split("shard")[1].split("of")[0]))
    assert len(files) == n_shards, \
        f"{d}: found {len(files)} shards for '{stem}', expected {n_shards}"
    rows = []
    for p in files:
        with open(p) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    rows.sort(key=lambda r: r["item_id"])
    assert len(rows) == expected_n, \
        f"{d}/{stem}: n={len(rows)} != expected {expected_n}"
    assert len({r["item_id"] for r in rows}) == expected_n, \
        f"{d}/{stem}: duplicate item_id"
    assert not any(r.get("nan") for r in rows), f"{d}/{stem}: nan rows present"
    return rows


def mmlu_content_norm_vec(rows):
    """Per-item 0/1 correctness under the length-normalised content interface,
    recomputed from scores and cross-checked against the harness flag."""
    n = len(rows)
    out = np.zeros(n)
    stored = np.zeros(n)
    for i, r in enumerate(rows):
        sc = r["content_norm"]["scores"]
        letters = sorted(sc.keys())
        v = [sc[L] for L in letters]
        pred = max(range(len(v)), key=lambda k: v[k])
        out[i] = 1.0 if pred == r["gold"] else 0.0
        stored[i] = 1.0 if r["content_norm"]["correct"] else 0.0
    assert np.array_equal(out, stored), (
        "content_norm recomputation disagrees with harness-stored flag on "
        f"{int((out != stored).sum())} items")
    return out


def qa_em_vec(rows):
    return np.asarray([float(r["em"]) for r in rows], dtype=float)


def one_sided_lower95(d, seed_off):
    """5th percentile of the paired item bootstrap on the per-item difference
    vector — identical construction to pilot_zero_rule_disagreement.ni_rule."""
    d = np.asarray(d, float)
    n = d.size
    vals, counts = np.unique(d, return_counts=True)
    rng = np.random.default_rng(SEED + seed_off)
    draws = rng.multinomial(n, counts / n, size=N_BOOT)
    means = draws @ vals / n
    return float(np.percentile(means, 5.0)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True,
                    help="dir containing olmo2_{mmlu_content,closedbook}_results/")
    ap.add_argument("--out_json", required=True)
    args = ap.parse_args()

    mroot = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    croot = os.path.join(args.raw_root, "olmo2_closedbook_results")

    # ---- the ONE arm where PLATEAU(T) is defined AND accepts ---------------
    # (olmo2_ppl_results/ has in-domain PPL only for the keep7f2 trajectory, so
    # the cpt20k / arm4 checkpoints cannot form a PLATEAU-vs-NI cell at all.)
    ARM = "keep7f2_step200000"
    dirs = {
        "intact": {"mmlu": "A03_1B_base", "cb": "A03_1B_base",
                   "nq": "A03_1B_base_nq"},
        ARM: {"mmlu": "A03_1B_keep7_step200k", "cb": "A03_1B_keep7_step200k",
              "nq": "A03_1B_keep7_step200k_nq"},
    }

    vecs, prov = {}, {}
    for name, spec in dirs.items():
        vecs[name], prov[name] = {}, {}
        d = os.path.join(mroot, spec["mmlu"])
        rows = load_shards(d, "mmlu", EXPECTED_N["mmlu"])
        vecs[name]["mmlu_content"] = mmlu_content_norm_vec(rows)
        vecs[name]["_mmlu_rows"] = rows
        prov[name]["mmlu_content"] = {"dir": d, "n": len(rows), "shards": 8}
        d = os.path.join(croot, spec["cb"])
        for task in ("triviaqa", "popqa"):
            rows = load_shards(d, task, EXPECTED_N[task])
            vecs[name][task] = qa_em_vec(rows)
            vecs[name][f"_{task}_rows"] = rows
            prov[name][task] = {"dir": d, "n": len(rows), "shards": 8,
                                "metric": "em"}
        d = os.path.join(croot, spec["nq"])
        rows = load_shards(d, "nq_open", EXPECTED_N["nq_open"])
        vecs[name]["nq_open"] = qa_em_vec(rows)
        vecs[name]["_nq_open_rows"] = rows
        prov[name]["nq_open"] = {"dir": d, "n": len(rows), "shards": 8,
                                 "metric": "em"}

    # item-id alignment between the two arms, per axis
    for axis, stem in [("mmlu_content", "_mmlu_rows"),
                       ("triviaqa", "_triviaqa_rows"),
                       ("popqa", "_popqa_rows"),
                       ("nq_open", "_nq_open_rows")]:
        a = [r["item_id"] for r in vecs["intact"][stem]]
        b = [r["item_id"] for r in vecs[ARM][stem]]
        assert a == b, f"{axis}: item_id misalignment between arms"

    # ---- input-blind nulls on the SAME item set ---------------------------
    # MMLU: longest-option null under all five tie conventions (A01).
    # QA: best-constant answer string. Both convention-free for QA.
    nulls = {}
    mrows = vecs["intact"]["_mmlu_rows"]
    gold_letters = [r["gold_letter"] for r in mrows]
    for conv in TIE_CONVS:
        nv = longest_option_vector(mrows, gold_letters, conv)
        nulls[("mmlu_content", conv)] = float(np.mean(nv))
    qa_null_constants = {}
    for task in ("triviaqa", "popqa", "nq_open"):
        rows = vecs["intact"][f"_{task}_rows"]
        best_str, acc, _vec, _diag = best_constant_qa(rows, "em")
        qa_null_constants[task] = {"constant": best_str, "acc": float(acc)}
        # QA nulls are convention-FREE (the tie convention is an MMLU
        # longest-option artefact); assert by construction, not by hope.
        for conv in TIE_CONVS:
            nulls[(task, conv)] = float(acc)
    for task in ("triviaqa", "popqa", "nq_open"):
        vals = {nulls[(task, c)] for c in TIE_CONVS}
        assert len(vals) == 1, f"{task}: QA null is not convention-free"

    fracs = [round(SWEEP_LO + i * SWEEP_STEP, 4)
             for i in range(int(round((SWEEP_HI - SWEEP_LO) / SWEEP_STEP)) + 1)]

    # ---- the sweep --------------------------------------------------------
    out_axes = {}
    for conv in TIE_CONVS:
        for axis in ALL_AXES:
            rep_arm = float(np.mean(vecs[ARM][axis]))
            rep_int = float(np.mean(vecs["intact"][axis]))
            null = nulls[(axis, conv)]
            res_arm = rep_arm - null
            res_int = rep_int - null
            d = vecs[ARM][axis] - vecs["intact"][axis]
            lo95, n = one_sided_lower95(d, SEED_OFF[axis])
            row = {
                "axis": axis, "convention": conv, "n": n,
                "demoted_descriptive_only": axis in DEMOTED_AXES,
                "reported_arm": rep_arm, "reported_intact": rep_int,
                "null": null,
                "residual_arm_pp": 100.0 * res_arm,
                "residual_intact_pp": 100.0 * res_int,
                "diff_mean_pp": 100.0 * float(d.mean()),
                "diff_lower95_one_sided_pp": 100.0 * lo95,
                "intact_residual_negative_D1": bool(res_int < 0),
                "by_fraction": {},
            }
            # exact crossing fraction: NI accepts iff lo95 > -f*res_int.
            # For res_int > 0 that is f > -lo95/res_int (lo95 < 0 here).
            if res_int > 0:
                f_star = (-lo95) / res_int
                row["exact_crossing_fraction"] = float(f_star)
                row["crossing_in_assigned_range"] = bool(
                    ASSIGNED_LO <= f_star <= ASSIGNED_HI)
                row["crossing_in_swept_range"] = bool(
                    SWEEP_LO <= f_star <= SWEEP_HI)
                # the same crossing computed from the POINT ESTIMATE rather
                # than the one-sided bound: recorded because that is the
                # quantity a reader is likely to reconstruct by hand, and it
                # is systematically SMALLER (the CI bound is more demanding).
                row["crossing_fraction_point_estimate"] = float(
                    (-row["diff_mean_pp"]) / row["residual_intact_pp"])
            else:
                # D1: Delta is negative, NI is a strict-superiority test, and
                # WIDENING the fraction makes it HARDER, not easier. There is
                # no accept-crossing in f>0. Guard marks the cell
                # NOT_CERTIFIABLE; recorded, never tested.
                row["exact_crossing_fraction"] = None
                row["crossing_in_assigned_range"] = False
                row["crossing_in_swept_range"] = False
                row["crossing_fraction_point_estimate"] = None
                row["not_certifiable_reason"] = (
                    "D1: residual(intact) < 0, so Delta < 0 and NI(Delta) is a "
                    "strict-superiority test. Per guard G1/G2 the cell is "
                    "NOT_CERTIFIABLE and NI is not run; the sweep records it "
                    "but reports no accept/reject.")
            for f in fracs:
                delta = f * res_int
                row["by_fraction"][f"{f:.2f}"] = {
                    "delta_pp": 100.0 * delta,
                    "ni_accept": (None if res_int <= 0
                                  else bool(lo95 > -delta)),
                }
            out_axes[f"{conv}|{axis}"] = row

    # ---- K1-shaped verdict per (convention, fraction) ---------------------
    # K1 clause (a): at the PLATEAU-accept checkpoint, NI accepts on >= 3 of 4
    # axes. With NQ-open demoted the decision family is 3 axes, so the guard's
    # rescale applies: ceil(0.75 * n_surviving) = ceil(2.25) = 3 of 3.
    verdicts = {}
    for conv in TIE_CONVS:
        dec = [out_axes[f"{conv}|{a}"] for a in DECISION_AXES]
        certifiable = [r for r in dec if not r["intact_residual_negative_D1"]]
        for f in fracs:
            k = f"{f:.2f}"
            acc = [r for r in certifiable if r["by_fraction"][k]["ni_accept"]]
            rej = [r for r in certifiable if not r["by_fraction"][k]["ni_accept"]]
            n_surv = len(certifiable)
            need = int(np.ceil(0.75 * n_surv)) if n_surv else 0
            verdicts[f"{conv}|{k}"] = {
                "convention": conv, "delta_fraction": f,
                "in_assigned_range": bool(f <= ASSIGNED_HI),
                "n_decision_axes_total": len(dec),
                "n_surviving_after_guard": n_surv,
                "n_not_certifiable_D1": len(dec) - n_surv,
                "n_ni_accept": len(acc), "n_ni_reject": len(rej),
                "ni_reject_axes": [r["axis"] for r in rej],
                "k1_clause_a_threshold_rescaled": need,
                "K1_clause_a_would_pass": bool(n_surv > 0
                                               and len(acc) >= need),
                "disagreement_of_required_shape_exists": bool(len(rej) > 0),
            }

    prereg_key = f"{PREREG_DELTA_FRACTION:.2f}"
    assigned = [f for f in fracs if f <= ASSIGNED_HI]

    def _inv(conv, fs):
        return all(verdicts[f"{conv}|{f:.2f}"][
            "disagreement_of_required_shape_exists"] for f in fs)

    # two DIFFERENT questions, reported separately because they have different
    # answers and conflating them would overstate the robustness:
    #   (1) does a disagreement of the required shape still EXIST (>=1 axis
    #       rejects)? -- the K1-firing question;
    #   (2) does EVERY decision axis still reject? -- the stronger 3/3 claim.
    invariance_existence = {c: _inv(c, fracs) for c in TIE_CONVS}
    invariance_existence_assigned = {c: _inv(c, assigned) for c in TIE_CONVS}

    def _all_reject(conv, fs):
        return all(verdicts[f"{conv}|{f:.2f}"]["n_ni_accept"] == 0 for f in fs)

    invariance_all_axes = {c: _all_reject(c, fracs) for c in TIE_CONVS}
    invariance_all_axes_assigned = {c: _all_reject(c, assigned)
                                    for c in TIE_CONVS}

    # per-(convention) first fraction on the grid where ANY axis flips to accept
    first_flip = {}
    for conv in TIE_CONVS:
        hit = None
        for f in fracs:
            if verdicts[f"{conv}|{f:.2f}"]["n_ni_accept"] > 0:
                hit = f
                break
        first_flip[conv] = hit

    crossings = {f"{conv}|{a}": out_axes[f"{conv}|{a}"][
        "exact_crossing_fraction"] for conv in TIE_CONVS for a in ALL_AXES}

    out = {
        "what": "A04 margin-sensitivity sweep: is the K1-shaped NI-rejects-"
                "where-PLATEAU-accepts finding invariant to the Delta fraction? "
                "Assigned range 0.10 -> 0.66; extended to 1.00 so that "
                "crossings above 0.66 can be reported.",
        "date": "2026-08-10",
        "gpu_spent": 0,
        "arm": ARM,
        "why_this_arm": "the only (arm, checkpoint) where PLATEAU(T) is DEFINED "
                        "and ACCEPTS: olmo2_ppl_results/ carries in-domain PPL "
                        "only for the keep7+fresh2 trajectory (steps 50k/100k/"
                        "147k/200k), so no other cell can form a PLATEAU-vs-NI "
                        "disagreement at all.",
        "preregistration": {
            "delta_fraction": PREREG_DELTA_FRACTION,
            "frozen_by_commit": "d1ba737",
            "unchanged_by_this_script": True,
            "guard_G2": "margin substitution is prohibited; this is a "
                        "sensitivity report on the pre-registered number, not "
                        "a re-registration.",
        },
        "sweep_grid": {
            "assigned_lo": ASSIGNED_LO, "assigned_hi": ASSIGNED_HI,
            "lo": SWEEP_LO, "hi": SWEEP_HI, "step": SWEEP_STEP,
            "n_points": len(fracs), "fractions": fracs,
            "assigned_range_predates_this_script": True,
            "extension_reason":
                "the assigned range's upper endpoint 0.66 sits BELOW two of the "
                "three decision-axis crossings and only 0.0042 below the third "
                "(MMLU-content split, 0.6642), so a sweep truncated at 0.66 "
                "could not report the crossing points the task asks for. "
                "Extended to 1.00; every point above 0.66 carries "
                "in_assigned_range=false. Nothing at or below 0.66 changes.",
            "direction_is_conservative":
                "widening Delta can only make NI EASIER to accept, so sweeping "
                "upward can only weaken A04's claim, never strengthen it. A "
                "reported invariance is therefore not an artefact of the "
                "sweep direction.",
            "pre_run_estimate_that_was_wrong":
                "before running, MMLU-content's crossing was estimated at "
                "0.610 from the POINT ESTIMATE (6.2455/10.2389). The rule uses "
                "the one-sided lower 95% bound, giving 0.6642 -- 0.0042 above "
                "the assigned upper endpoint. Recorded because the assigned "
                "range came within 0.4pp-of-fraction of hiding a flip.",
        },
        "nulls_by_axis_convention": {
            f"{a}|{c}": nulls[(a, c)] for a in ALL_AXES for c in TIE_CONVS},
        "qa_null_constants": qa_null_constants,
        "per_axis_convention": out_axes,
        "k1_shaped_verdict_by_convention_and_fraction": verdicts,
        "exact_crossing_fractions": crossings,
        "HEADLINE": {
            "preregistered_fraction": PREREG_DELTA_FRACTION,
            "preregistered_convention": "split",
            "Q1_disagreement_still_EXISTS": {
                "over_assigned_range_0.10_0.66": invariance_existence_assigned,
                "over_extended_range_0.10_1.00": invariance_existence,
            },
            "Q2_ALL_three_axes_still_reject": {
                "over_assigned_range_0.10_0.66": invariance_all_axes_assigned,
                "over_extended_range_0.10_1.00": invariance_all_axes,
            },
            "first_fraction_on_grid_where_any_axis_accepts": first_flip,
            "n_ni_reject_at_prereg_split": verdicts[f"split|{prereg_key}"][
                "n_ni_reject"],
            "n_ni_reject_at_assigned_hi_split": verdicts[
                f"split|{ASSIGNED_HI:.2f}"]["n_ni_reject"],
            "earliest_crossing_axis_split": min(
                ((out_axes[f"split|{a}"]["exact_crossing_fraction"], a)
                 for a in DECISION_AXES
                 if out_axes[f"split|{a}"]["exact_crossing_fraction"]
                 is not None), default=(None, None))[1],
            "earliest_crossing_fraction_split": min(
                (out_axes[f"split|{a}"]["exact_crossing_fraction"]
                 for a in DECISION_AXES
                 if out_axes[f"split|{a}"]["exact_crossing_fraction"]
                 is not None), default=None),
            "interpretation":
                "Q1 is the question that bears on K1: K1 fires only if the "
                "rules AGREE, so as long as >=1 decision axis rejects at a "
                "PLATEAU-accept checkpoint, K1 does not fire on the cells that "
                "exist. Q2 is the stronger 3/3 claim quoted in the pilot "
                "verdict, and it is NOT invariant everywhere -- see "
                "first_fraction_on_grid_where_any_axis_accepts. Reporting both "
                "separately is the point: the weaker claim is robust, the "
                "stronger one has a crossing.",
        },
        "provenance": prov,
        "method_compliance": [
            "longest_option_vector / best_constant_qa IMPORTED from A03's "
            "analyze_1b_knowledge_floor.py, never reimplemented.",
            "8/8 shard completeness + exact item count + duplicate-item_id + "
            "nan-row assertions on every dump.",
            "item_id alignment asserted between the two arms on all four axes.",
            "content_norm correctness recomputed from scores and asserted "
            "equal to the harness-stored flag.",
            "bootstrap seed offsets copied from "
            "pilot_zero_rule_disagreement.py so the 0.10 column is "
            "bit-comparable with the pilot's cells.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1)

    # ---- console summary --------------------------------------------------
    print("=== A04 margin sensitivity sweep (arm =", ARM, ") ===")
    print(f"pre-registered Delta fraction = {PREREG_DELTA_FRACTION} "
          f"(unchanged); swept {SWEEP_LO} -> {SWEEP_HI} step {SWEEP_STEP}")
    print()
    print("split convention, decision axes:")
    for a in DECISION_AXES:
        r = out_axes[f"split|{a}"]
        fs = r["exact_crossing_fraction"]
        print(f"  {a:<13} residual(intact)={r['residual_intact_pp']:>8.4f}pp  "
              f"lower95(diff)={r['diff_lower95_one_sided_pp']:>9.4f}pp  "
              f"crossing f*={'n/a' if fs is None else f'{fs:.4f}'}")
    print()
    print("  f     n_accept/n_surv   reject axes")
    for f in fracs:
        v = verdicts[f"split|{f:.2f}"]
        flag = "" if f <= ASSIGNED_HI else "  (beyond assigned range)"
        print(f"  {f:.2f}   {v['n_ni_accept']}/{v['n_surviving_after_guard']}"
              f"              {','.join(v['ni_reject_axes']) or '-'}{flag}")
    print()
    print("Q1 disagreement EXISTS, assigned range 0.10-0.66:",
          invariance_existence_assigned)
    print("Q1 disagreement EXISTS, extended range 0.10-1.00:",
          invariance_existence)
    print("Q2 ALL 3 axes reject, assigned range:",
          invariance_all_axes_assigned)
    print("Q2 ALL 3 axes reject, extended range:", invariance_all_axes)
    print("first grid fraction where ANY axis accepts:", first_flip)
    print("wrote", args.out_json)


if __name__ == "__main__":
    main()
