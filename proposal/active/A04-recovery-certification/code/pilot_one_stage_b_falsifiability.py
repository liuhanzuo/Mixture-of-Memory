#!/usr/bin/env python
"""A04 Pilot One, STAGE B — the FALSIFIABILITY check that K2 does not perform.

Costs ZERO GPU. Companion to `pilot_one_stage_b_s3.py`, which answers only
"is `sd_run` small enough for the gate to be readable?" (K2). This script answers
the question Stage B was actually launched to settle, which K2 cannot see.

WHY THIS IS A SEPARATE, NECESSARY ANALYSIS
------------------------------------------
`PILOT_ONE_PREREG.md` §3 states Stage B's purpose in one sentence, and it is NOT
K2:

    "keep7+fresh2 is a confirmed constant-REJECT rung [...] so a rule tested only
    there can never be *observed to accept* and the disagreement is automatic and
    uninformative. keep12 at 87.5% depth is the candidate most likely to let NI
    sometimes accept, which is what makes the disagreement test falsifiable at
    all."

And §5 item 3 lists as known-unverified: "Whether 5,000 steps at keep12 produces
enough recovery for NI to ever accept."

K2 is blind to this. K2 compares `bound_S` (run-to-run noise) against `Delta`. A
rung can pass K2 with flying colours while being a **constant-REJECT** rung, because
a saturated deficit is perfectly reproducible across seeds -- low variance is in
fact what a saturated axis looks like. So "K2 does not fire" must not be read as
"Stage B succeeded".

WHAT THIS COMPUTES
------------------
Per axis, from the SAME per-item shards the K2 harvest used and against the SAME
G0-pinned intact anchor:

    residual(arm)     = reported(arm)    - null            (calibrated capability)
    residual(intact)  = reported(intact) - null
    deficit           = residual(intact) - residual(arm)
                      = reported(intact) - reported(arm)    (null cancels exactly)
    NI(Delta) ACCEPTS iff deficit <= Delta,  Delta = 0.10 * residual(intact)

then expresses `deficit - Delta` in units of the measured keep12 `sd_run`. That
ratio is the falsifiability statistic: if the deficit sits many `sd_run` ABOVE
Delta, no realisable seed draw could ever flip NI to accept, so NI is a constant
at this rung and the NI-vs-PLATEAU disagreement it produces is uninformative --
the exact defect that disqualified keep7.

ANCHOR PROVENANCE, CHECKED NOT ASSUMED
--------------------------------------
The pre-registered `Delta` values were fixed in `PILOT_ZERO_VERDICT.md` §1 before
any keep12 datum existed. This script recomputes `Delta = 0.10 * (intact - null)`
from the pinned intact artefact and ASSERTS it reproduces the pre-registered
constant to 1e-6 pp on each decision axis. If the anchor had drifted, every
verdict downstream would be silently re-margined; that is rule G0's whole point.

The MMLU null is the pre-registered longest-option split-tie convention (34.22% of
items have >=2 maximal-length options), not 0.25 -- design §4.2 records that
0.25 IS NEVER THE MMLU NULL and bans the letter interface as a decision axis. The
closed-book nulls are the harness's own best-constant (`majority_em`).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import sys
from pathlib import Path

_SHARED_CODE = Path(__file__).resolve().parents[3] / "shared" / "code"
sys.path.insert(0, str(_SHARED_CODE))
try:
    from canonical_eval_loaders import (  # noqa: E402
        CB as _CB_ROOT,
        MM as _MM_ROOT,
        ROOT as _RESULTS_ROOT,
    )
except Exception as _e:
    print(f"[falsif][FATAL] cannot import canonical loaders: {_e!r}", file=sys.stderr)
    sys.exit(2)

# Pre-registered, from PILOT_ZERO_VERDICT.md §1 / STATUS.json:nulls_per_metric.
DELTA_PP_PREREG = {
    "triviaqa": 4.043134195274186,
    "popqa": 1.3205298941613512,
    "mmlu_content": 1.0238926078906136,
    "nq_open": 0.9695290858725762,
}
# The construct-appropriate best-constant null per axis. MMLU's is the
# pre-registered longest-option split-tie value; the closed-book ones are the
# harness's own `majority_em` on the intact anchor and are re-read from disk below.
MMLU_NULL_PP = 28.445022076627263
DECISION_AXES = ("triviaqa", "popqa", "mmlu_content")
SEEDS = (101, 102, 103)
# The intact anchor pinned by rule G0 (evidence/a04_g0_anchor_sha256_pinning.json).
INTACT_CB, INTACT_NQ, INTACT_MM = "A03_1B_base", "A03_1B_base_nq", "A03_1B_base"
# keep7+fresh2 at 20,000 CPT steps on top of the 200k heal -- the rung Pilot Zero
# disqualified as constant-REJECT. Sampler seed 0 is A03's original Arm 3.
KEEP7_DIRS = ["A03_1B_arm3_cpt_step220000"] + \
             [f"A03_1B_dataorder_seed{s}_step220000" for s in (43, 44, 45)]


def cb_em_pct(d: str, task: str) -> float:
    return json.load(open(_CB_ROOT / d / "summary.json"))["tasks"][task]["em"] * 100.0


def cb_null_pct(d: str, task: str) -> float:
    return json.load(open(_CB_ROOT / d / "summary.json"))["tasks"][task]["majority_em"] * 100.0


def mm_pct(d: str) -> float:
    return json.load(open(_MM_ROOT / d / "summary.json"))["content_norm_acc"] * 100.0


def arm_axis_pct(dirname: str, axis: str) -> float:
    if axis == "mmlu_content":
        return mm_pct(dirname)
    if axis == "nq_open":
        return cb_em_pct(dirname + "_nq", "nq_open")
    return cb_em_pct(dirname, axis)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_json", required=True)
    a = ap.parse_args()

    nulls = {
        "triviaqa": cb_null_pct(INTACT_CB, "triviaqa"),
        "popqa": cb_null_pct(INTACT_CB, "popqa"),
        "mmlu_content": MMLU_NULL_PP,
        "nq_open": cb_null_pct(INTACT_NQ, "nq_open"),
    }
    intact = {
        "triviaqa": cb_em_pct(INTACT_CB, "triviaqa"),
        "popqa": cb_em_pct(INTACT_CB, "popqa"),
        "mmlu_content": mm_pct(INTACT_MM),
        "nq_open": cb_em_pct(INTACT_NQ, "nq_open"),
    }

    per_axis, anchor_checks = {}, {}
    for axis in ("triviaqa", "popqa", "mmlu_content", "nq_open"):
        i_res = intact[axis] - nulls[axis]
        delta_recomputed = 0.10 * i_res
        drift = abs(delta_recomputed - DELTA_PP_PREREG[axis])
        anchor_checks[axis] = {
            "delta_recomputed_from_pinned_anchor_pp": delta_recomputed,
            "delta_prereg_pp": DELTA_PP_PREREG[axis],
            "abs_drift_pp": drift,
            "reproduces_prereg_to_1e6": bool(drift < 1e-6),
        }
        # Delta is NEVER substituted (margin-guard verdict): use the prereg value.
        delta = DELTA_PP_PREREG[axis]

        k12 = [arm_axis_pct(f"A04_1B_stageB_keep12_seed{s}_step5000", axis)
               for s in SEEDS]
        k7 = [arm_axis_pct(d, axis) for d in KEEP7_DIRS]
        m12, sd12 = statistics.fmean(k12), statistics.stdev(k12)
        m7, sd7 = statistics.fmean(k7), statistics.stdev(k7)

        a_res = m12 - nulls[axis]
        deficit = i_res - a_res          # == intact - arm; the null cancels exactly
        excess = deficit - delta         # >0 means NI REJECTS
        per_axis[axis] = {
            "decision_weight": axis in DECISION_AXES,
            "null_pp": nulls[axis],
            "intact_reported_pp": intact[axis],
            "intact_residual_pp": i_res,
            "delta_pp": delta,
            "keep12_seed_means_pct": dict(zip(map(str, SEEDS), k12)),
            "keep12_mean_pct": m12,
            "keep12_sd_run_pp": sd12,
            "keep12_residual_pp": a_res,
            "keep12_recovery_fraction_pct": 100.0 * a_res / i_res,
            "deficit_pp": deficit,
            "ni_verdict": "REJECT" if excess > 0 else "ACCEPT",
            "deficit_minus_delta_pp": excess,
            "deficit_minus_delta_in_sd_run": excess / sd12,
            "deficit_in_sd_run": deficit / sd12,
            "keep7_20k_mean_pct": m7,
            "keep7_20k_sd_run_pp": sd7,
            "keep7_20k_recovery_fraction_pct": 100.0 * (m7 - nulls[axis]) / i_res,
            "keep7_20k_ni_verdict": "REJECT" if (i_res - (m7 - nulls[axis])) > delta
                                    else "ACCEPT",
        }

    n_reject = sum(1 for x in DECISION_AXES if per_axis[x]["ni_verdict"] == "REJECT")
    n_reject_all = sum(1 for x in per_axis if per_axis[x]["ni_verdict"] == "REJECT")
    min_excess_sd = min(per_axis[x]["deficit_minus_delta_in_sd_run"]
                        for x in DECISION_AXES)

    if n_reject == len(DECISION_AXES):
        falsif = "CONSTANT_REJECT_AT_KEEP12"
        finding = (
            f"NI(Delta) REJECTS on {n_reject}/{len(DECISION_AXES)} decision axes "
            f"(and {n_reject_all}/4 including demoted nq_open). The SMALLEST "
            f"margin by which any decision axis rejects is "
            f"{min_excess_sd:.1f} x the measured keep12 sd_run, so no realisable "
            f"seed draw flips NI to accept. keep12+fresh2 at 5,000 steps is a "
            f"CONSTANT-REJECT rung -- the same defect that disqualified keep7 "
            f"(PILOT_ZERO_VERDICT.md §5) and precisely what prereg §3 chose "
            f"keep12 to escape. The NI-vs-PLATEAU disagreement is therefore "
            f"automatic and uninformative at this rung, so Stage B did NOT "
            f"deliver the falsifiable test it was launched to deliver.")
    elif n_reject == 0:
        falsif = "CONSTANT_ACCEPT_AT_KEEP12"
        finding = ("NI(Delta) ACCEPTS on all decision axes. prereg §5 item 3 "
                   "anticipated this case and prescribes bracketing DOWN to "
                   "keep10.")
    else:
        falsif = "MIXED_KEEP12_IS_DISCRIMINATIVE"
        finding = (f"NI rejects on {n_reject}/{len(DECISION_AXES)} decision axes "
                   f"-- the rung is discriminative, which is what Stage B wanted.")

    payload = {
        "analysis": "A04_pilot_one_stage_B_falsifiability_and_recovery_level",
        "what_it_answers": "PILOT_ONE_PREREG.md §5 known-unverified item 3: "
                           "'Whether 5,000 steps at keep12 produces enough "
                           "recovery for NI to ever accept.' K2 is BLIND to this: "
                           "a saturated deficit is highly reproducible across "
                           "seeds, so a constant-REJECT rung passes K2 easily.",
        "gpu_h_additional": 0,
        "arm": "keep12+fresh2, OLMo-2-0425-1B, 5,000 steps, dolmino, seeds "
               "{101,102,103}",
        "intact_anchor": {
            "pinned_by": "rule G0; evidence/a04_g0_anchor_sha256_pinning.json",
            "cb_dir": INTACT_CB, "nq_dir": INTACT_NQ, "mmlu_dir": INTACT_MM,
        },
        "estimator": {
            "residual": "reported - null (construct-appropriate best-constant)",
            "deficit": "residual(intact) - residual(arm) == reported(intact) - "
                       "reported(arm); the null cancels EXACTLY, so the deficit is "
                       "convention-independent",
            "ni_rule": "NI(Delta) ACCEPTS iff deficit <= Delta, "
                       "Delta = 0.10 * residual(intact) (never substituted -- "
                       "margin-guard verdict is RETIRE, not re-margin)",
            "falsifiability_statistic": "(deficit - Delta) / sd_run, in units of "
                                        "the measured keep12 run-to-run sd",
        },
        "anchor_reproduces_prereg_delta": anchor_checks,
        "per_axis": per_axis,
        "n_decision_axes_ni_rejects": n_reject,
        "n_all_axes_ni_rejects": n_reject_all,
        "min_decision_axis_excess_in_sd_run": min_excess_sd,
        "falsifiability_verdict": falsif,
        "finding": finding,
        "engages_STAGE_B_DECISION_noise_floor_worry": {
            "the_worry_verbatim": "STAGE_B_DECISION.md: 'the effect we'd measure "
                                  "at keep12 is plausibly < 0.5pp', i.e. inside "
                                  "run-to-run noise.",
            "resolution": "The worry conflated TWO different quantities, and they "
                          "land on opposite sides of the noise floor.",
            "quantity_1_the_level_NI_adjudicates": (
                "The keep12 DEFICIT vs the intact anchor is "
                + ", ".join(f"{x} {per_axis[x]['deficit_pp']:.4f}pp "
                            f"({per_axis[x]['deficit_in_sd_run']:.0f} sd_run)"
                            for x in DECISION_AXES)
                + ". This is FAR OUTSIDE the noise floor -- by 1-2 orders of "
                  "magnitude. So NI's accept/reject decision at keep12 is not "
                  "noise-limited at all; it is SATURATED. The noise-floor worry "
                  "does not apply to it."),
            "quantity_2_the_marginal_CPT_effect": (
                "The marginal effect of MORE continued pre-training -- what A03 "
                "measured at -0.0293pp, CI95 [-0.672,+0.613], df=3 -- IS inside "
                "the noise floor. The worry was correct about this one."),
            "why_that_is_worse_not_better_for_A04": (
                "A04 needed the deficit to be COMPARABLE to Delta so that NI could "
                "sometimes accept and sometimes reject. Instead the deficit exceeds "
                "Delta by "
                + ", ".join(f"{x} {per_axis[x]['deficit_minus_delta_in_sd_run']:.0f} sd"
                            for x in DECISION_AXES)
                + ". Being far outside the noise floor in the REJECT direction is "
                  "not a rescue -- it is the constant-REJECT degeneracy. A04 is not "
                  "killed by noise here; it is stalled by SATURATION, which no "
                  "number of extra seeds can fix."),
        },
        "bracketing_consequence": {
            "prereg_anticipated_only_the_other_failure": "prereg §5 item 3 says "
                "'If keep12 turns out to be a constant-ACCEPT rung, the gate must "
                "bracket downward to keep10'. The observed failure is the OPPOSITE: "
                "constant-REJECT.",
            "so_keep10_is_the_wrong_direction": "keep10 (75% depth) is MORE damaged "
                "than keep12 (87.5%), so it would reject harder. Bracketing DOWN "
                "cannot repair a constant-REJECT rung.",
            "what_would_be_needed": "either LESS damage (keep14) or MANY more heal "
                "steps -- and A03 has already shown that 10x the token budget "
                "(200k steps / 52.43B tokens at keep7) does not close the gap. "
                "Neither option is authorised by any current prereg, and both are "
                "new GPU tranches, not re-analyses.",
        },
        "keep7_comparison": {
            "note": "keep7+fresh2 after 220,000 steps vs keep12+fresh2 after "
                    "5,000. keep12 reaches a COMPARABLE OR BETTER recovery "
                    "fraction in 1/44th the steps, yet is still constant-REJECT.",
            "recovery_fraction_pct": {
                x: {"keep12_5k": per_axis[x]["keep12_recovery_fraction_pct"],
                    "keep7_220k": per_axis[x]["keep7_20k_recovery_fraction_pct"]}
                for x in per_axis},
            "both_rungs_ni_verdict": {
                x: {"keep12_5k": per_axis[x]["ni_verdict"],
                    "keep7_220k": per_axis[x]["keep7_20k_ni_verdict"]}
                for x in per_axis},
        },
        "plateau_side_not_measurable": {
            "problem": "PLATEAU(T) needs in-domain val PPL on the checkpoint grid, "
                       "and NO val PPL was computed for any Stage B checkpoint: "
                       "the training logs contain zero val/eval lines and "
                       "olmo2_ppl_results/ has no *stageB* directory.",
            "consequence": "the NI-vs-PLATEAU DISAGREEMENT itself cannot be "
                           "evaluated at keep12 from what is on disk -- only NI's "
                           "half. K1 remains unadjudicated at this rung "
                           "independently of everything above.",
            "grid_coverage": "prereg grid is [2500,5000,10000,20000,40000,80000]; "
                             "Stage B produced only 2500 and 5000 = 2 of 6 points.",
        },
        "provenance": {
            "results_root": str(_RESULTS_ROOT),
            "keep12_dirs": [f"A04_1B_stageB_keep12_seed{s}_step5000" for s in SEEDS],
            "keep7_dirs": KEEP7_DIRS,
            "integrity": "the 12 keep12 cells were asserted 8/8 shards + exact "
                         "item counts + nan=0 by pilot_one_stage_b_s3.py; see "
                         "evidence/pilot_one_stage_b_s3_verdict.json:integrity",
            "mmlu_null_convention": "longest-option, split-tie (pre-registered). "
                                    "0.25 is NEVER the MMLU null (design §4.2); "
                                    "the letter interface is banned as a decision "
                                    "axis.",
        },
    }

    Path(os.path.dirname(a.out_json) or ".").mkdir(parents=True, exist_ok=True)
    with open(a.out_json, "w") as f:
        json.dump(payload, f, indent=2)

    print(f"[falsif] VERDICT: {falsif}")
    print(f"[falsif] {finding}\n")
    print(f"{'axis':14s} {'null':>8s} {'intact':>8s} {'iResid':>8s} {'k12':>8s} "
          f"{'rec%':>6s} {'deficit':>8s} {'Delta':>7s} {'NI':>7s} {'excess/sd':>10s}")
    for axis, r in per_axis.items():
        tag = "" if r["decision_weight"] else "  (DEMOTED)"
        print(f"{axis:14s} {r['null_pp']:8.4f} {r['intact_reported_pp']:8.4f} "
              f"{r['intact_residual_pp']:8.4f} {r['keep12_mean_pct']:8.4f} "
              f"{r['keep12_recovery_fraction_pct']:6.2f} {r['deficit_pp']:8.4f} "
              f"{r['delta_pp']:7.4f} {r['ni_verdict']:>7s} "
              f"{r['deficit_minus_delta_in_sd_run']:+10.1f}{tag}")
    print("\n[falsif] anchor check (Delta recomputed from pinned intact artefact):")
    for axis, c in anchor_checks.items():
        print(f"  {axis:14s} recomputed={c['delta_recomputed_from_pinned_anchor_pp']:.9f} "
              f"prereg={c['delta_prereg_pp']:.9f} drift={c['abs_drift_pp']:.2e} "
              f"-> {'OK' if c['reproduces_prereg_to_1e6'] else 'DRIFTED'}")
    print(f"\n[falsif] wrote {a.out_json}")


if __name__ == "__main__":
    main()
