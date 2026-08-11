#!/usr/bin/env python3
"""A04 margin guard — enumerate Delta-degeneracy conditions and classify every
planned gate cell as CERTIFIABLE / NOT_CERTIFIABLE / NEEDS_RECHECK_AFTER_DATA.

WHY THIS EXISTS
---------------
The pre-registered non-inferiority margin in `A04_GATE_DESIGN.md` §2 is

    Delta_x = 0.10 * residual(intact, x),   residual = reported - null_x

and `NI(Delta)` accepts iff the one-sided lower 95% bound on
`residual(arm) - residual(intact)` is `> -Delta_x`.

That rule is silent on what happens when `residual(intact, x)` is not a
comfortably positive number. Pilot Zero found one such case for real (the
`credit` MMLU convention: intact 1B scores 0.3868 against its own null 0.4537,
so residual = -6.687pp and Delta = -0.669pp). With Delta negative, `-Delta` is
POSITIVE and the test silently becomes a STRICT SUPERIORITY test. It never
produces a false accept -- it is strictly harder -- but it produces a silent
FALSE REJECT plus a semantic mislabel: the report says "NI did not pass" when
the test that actually ran was a different hypothesis.

This script does not change the pre-registered rule. It computes, from data on
disk, which of six enumerated degeneracy conditions fire in which cell, so the
guard in `A04_MARGIN_GUARD_PREREG.md` can be stated with numbers.

ALL NUMBERS ARE READ FROM FILES. Nothing is transcribed from prose.

CPU ONLY. No GPU, no model load, no torch. Read-only on all inputs.

Usage:
  python a04_margin_guard_classify.py \
     [--pilot_zero evidence/pilot_zero_rule_disagreement.json] \
     [--a03_4axes ../A03-.../evidence/a03_1b_floor_nulls_4axes.json] \
     [--a01_conv  ../A01-.../evidence/gate3_content_null_conventions.json] \
     [--intact_ci evidence/a04_intact_residual_ci_1b_mmlu.json] \
     --out_json evidence/a04_margin_guard_classification.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys


def _a03_dir(proposal_dir):
    """Absolute path to A03's proposal dir, wherever it currently lives.

    A03 was ARCHIVED 2026-08-11 (`proposal/active` -> `proposal/archive`), so a
    hard-coded `../A03-...` no longer resolves. The location is kept in ONE place
    (`proposal/shared/code/proposal_paths.py`) and a genuinely missing A03 raises
    rather than yielding a path that silently does not exist -- this script's
    `--a03_4axes` default is a *measured* null table, and a wrong-but-plausible
    default would be read as data.
    """
    shared = os.path.abspath(os.path.join(
        proposal_dir, "..", "..", "shared", "code"))
    if shared not in sys.path:
        sys.path.insert(0, shared)
    from proposal_paths import a03_code_dir  # noqa: E402
    return os.path.dirname(a03_code_dir())

# ---------------------------------------------------------------------------
# The pre-registered constants this guard is an amendment to. Copied here ONLY
# so the script is self-documenting; every one is re-read from the Pilot Zero
# JSON's own `preregistration` block and ASSERTED equal below.
# ---------------------------------------------------------------------------
PREREG_EXPECTED = {
    "T_pct_per_5k": 2.0,
    "rho": 0.85,
    "delta_fraction": 0.10,
}

# Guard thresholds. These are the NEW pre-registered numbers this amendment
# adds. They are module-level constants with no CLI override, for the same
# reason T/rho/Delta are: so no invocation can tune them.
GUARD = {
    # D2: |residual(intact)| below this => Delta is too small to be a margin
    # at all (NI degenerates towards an exact-equality test, power -> 0).
    # 1.0pp chosen because it makes Delta = 0.10 * 1.0pp = 0.10pp, which is
    # BELOW every item-level bootstrap half-width measured anywhere in this
    # project (min observed 0.4556pp, see `observed_bootstrap_halfwidths`).
    "residual_floor_pp": 1.0,
    # D6: Delta below this multiple of the cell's own achievable item-level
    # half-width means the margin is finer than the measurement. 1.0 = "the
    # margin must be at least as large as the CI half-width", which is the
    # same criterion the design already used to DEMOTE NQ-open in §5.2.
    "delta_over_halfwidth_min": 1.0,
    # D6 is the only condition whose input (the paired discordance rate
    # p_disc) is not knowable before the recovered arm exists. It is therefore
    # pre-registered as a CRITICAL VALUE on p_disc, computed now from Delta
    # and n, and checked once per cell after the data lands:
    #     hw(p_disc) = z2 * sqrt(p_disc/n);  D6 fires iff Delta < hw
    #  => D6 fires iff p_disc > p_disc_crit = n * (Delta / (100*z2))^2
    # A cell is pre-classified NOT_CERTIFIABLE only if p_disc_crit is below
    # the range already observed for that axis in this project; it is
    # NEEDS_RECHECK_AFTER_DATA if p_disc_crit lies above the observed range
    # but within this safety factor of it.
    "d6_pdisc_safety_factor": 2.0,
}

AXES = ["triviaqa", "popqa", "mmlu_content", "nq_open"]
CONVENTIONS = ["split", "first", "last", "credit", "wrong"]
# The gate's planned structure (A04_GATE_DESIGN.md §3.2 + §2 frozen grid).
ARMS = ["A1_prefix_fresh_tail", "A2_contiguous_keep_only",
        "A3_random_trunk", "A4_from_scratch"]
CKPT_GRID = [2500, 5000, 10000, 20000, 40000, 80000]

# Axes the design already excludes from decision cells, for reasons that are
# NOT this guard's concern. Recorded so the cell accounting is honest.
DEMOTED_AXES = {"nq_open"}       # design §5.2, item CI > its own Delta
BANNED_INTERFACES = {"mmlu_letter", "raw_contains_on_generative_QA"}


def z95_one_sided():
    """1.6449 -- the one-sided 95% normal quantile the NI bound corresponds to."""
    return 1.6448536269514722


def z95_two_sided():
    return 1.959963984540054


def halfwidth_from_pdisc(p_disc, n):
    """Two-sided 95% CI half-width of a paired accuracy difference, in pp."""
    return 100.0 * z95_two_sided() * math.sqrt(p_disc / n)


# ---------------------------------------------------------------------------
# the six enumerated conditions
# ---------------------------------------------------------------------------
CONDITIONS = {
    "D1_residual_negative": (
        "residual(intact, x) < 0 -- the intact arm is BELOW its own "
        "construct-appropriate null. Delta < 0, so -Delta > 0 and NI(Delta) "
        "silently becomes a STRICT SUPERIORITY test. Worse, the comparison "
        "target is itself sub-null, so 'recovered is non-inferior to intact' "
        "has no scientific content on this axis."),
    "D2_residual_at_zero": (
        "0 <= residual(intact, x) <= residual_floor_pp -- Delta -> 0, so NI "
        "degenerates towards an exact-equality test whose power against any "
        "real difference is ~0. Not a wrong hypothesis, but an unachievable "
        "one; a REJECT carries no information."),
    "D3_residual_ci_straddles_zero": (
        "the CI on residual(intact, x) contains 0 -- the SIGN of Delta is "
        "itself a sampling outcome. The rule's semantics (non-inferiority vs "
        "superiority) would then depend on a coin flip, which no "
        "pre-registration can license."),
    "D4_baseline_illegitimate": (
        "residual(intact, x) is well-defined and positive, but the NULL "
        "itself is not admissible on this interface -- e.g. the arm is "
        "significantly below the best-constant floor, or the interface is "
        "degenerate (100% tie rate / constant emission). Delta is then a "
        "well-formed number computed on top of an invalid measurement."),
    "D5_intact_anchor_unstable": (
        "residual(intact, x) differs between two admissible measurements of "
        "the SAME intact model, so Delta is not uniquely determined by the "
        "pre-registration. Requires pinning the intact anchor artefact."),
    "D6_delta_below_measurement_resolution": (
        "Delta_x < delta_over_halfwidth_min x (achievable item-level 95% CI "
        "half-width for that axis at that n). The margin is finer than the "
        "instrument, so NI can essentially never accept regardless of the "
        "arm's true quality. This is the condition the design already used to "
        "demote NQ-open. Pre-registered as a critical discordance rate "
        "p_disc_crit = n*(Delta/(100*z2))^2; D6 fires iff the cell's measured "
        "p_disc exceeds it."),
}


def d6_status(delta_pp, n, pdisc_observed_range, safety):
    """Pre-data D6 classification via the critical discordance rate.

    Returns (status, p_disc_crit) where status is one of
    'fires' / 'recheck' / 'clear'. 'fires' means p_disc_crit is at or below
    the discordance range this axis has already exhibited in this project, so
    the margin is known to be finer than the instrument. 'recheck' means
    p_disc_crit is above that range but within `safety` of it, so the verdict
    depends on the arm that does not exist yet.
    """
    if delta_pp <= 0 or n is None:
        return "not_applicable", None
    crit = n * (delta_pp / (100.0 * z95_two_sided())) ** 2
    hi = pdisc_observed_range["max"]
    if crit <= hi:
        return "fires", crit
    if crit <= safety * hi:
        return "recheck", crit
    return "clear", crit


def classify(residual_pp, ci_lo_pp, ci_hi_pp, halfwidth_pp,
             baseline_illegitimate, anchor_drift_pp, delta_fraction,
             n_items, pdisc_range, ):
    """Return (verdict, fired_conditions, numbers) for one (axis, convention)."""
    fired = []
    recheck = []
    delta_pp = delta_fraction * residual_pp

    if residual_pp < 0:
        fired.append("D1_residual_negative")
    elif residual_pp <= GUARD["residual_floor_pp"]:
        fired.append("D2_residual_at_zero")

    if ci_lo_pp is not None and ci_hi_pp is not None:
        if ci_lo_pp < 0.0 < ci_hi_pp:
            fired.append("D3_residual_ci_straddles_zero")

    if baseline_illegitimate:
        fired.append("D4_baseline_illegitimate")

    # D5 only bites if the drift is large enough to move a decision, i.e. if
    # it is comparable to Delta itself. Recorded either way.
    if anchor_drift_pp is not None and delta_pp != 0:
        if abs(anchor_drift_pp) >= 0.10 * abs(delta_pp):
            fired.append("D5_intact_anchor_unstable")

    d6, pcrit = d6_status(delta_pp, n_items, pdisc_range,
                          GUARD["d6_pdisc_safety_factor"])
    if d6 == "fires":
        fired.append("D6_delta_below_measurement_resolution")
    elif d6 == "recheck":
        recheck.append("D6_delta_below_measurement_resolution")

    # verdict policy = option (c) of the guard: a cell whose margin is not a
    # well-formed non-inferiority margin is RETIRED, not re-margined.
    hard = {"D1_residual_negative", "D2_residual_at_zero",
            "D3_residual_ci_straddles_zero", "D4_baseline_illegitimate",
            "D6_delta_below_measurement_resolution"}
    if set(fired) & hard:
        verdict = "NOT_CERTIFIABLE"
    elif fired or recheck:
        verdict = "NEEDS_RECHECK_AFTER_DATA"
    else:
        verdict = "CERTIFIABLE"
    return verdict, fired, recheck, {
        "residual_pp": residual_pp, "delta_pp": delta_pp,
        "halfwidth_pp": halfwidth_pp, "d6_status": d6,
        "p_disc_crit": pcrit}


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(here, ".."))
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot_zero", default=os.path.join(
        root, "evidence", "pilot_zero_rule_disagreement.json"))
    ap.add_argument("--a03_4axes", default=os.path.abspath(os.path.join(
        _a03_dir(root), "evidence", "a03_1b_floor_nulls_4axes.json")))
    ap.add_argument("--a01_conv", default=os.path.abspath(os.path.join(
        root, "..", "A01-null-calibration-methodology", "evidence",
        "gate3_content_null_conventions.json")))
    ap.add_argument("--intact_ci", default=os.path.join(
        root, "evidence", "a04_intact_residual_ci_1b_mmlu.json"))
    ap.add_argument("--pdisc_extra", default=os.path.join(
        root, "evidence", "a04_pdisc_mmlu_1b.json"),
        help="measured p_disc for arms Pilot Zero did NOT score, notably the "
             "barely-healed step500 arm. The gate's grid starts at 2,500 "
             "steps, so a barely-healed arm -- not keep7@200k -- is the "
             "worst case for D6, and leaving it out would understate the "
             "discordance range.")
    ap.add_argument("--out_json", default=os.path.join(
        root, "evidence", "a04_margin_guard_classification.json"))
    args = ap.parse_args()

    pz = json.load(open(args.pilot_zero))
    a03 = json.load(open(args.a03_4axes))
    a01 = json.load(open(args.a01_conv))
    ici = json.load(open(args.intact_ci))

    # --- assert the pre-registration we are amending is the one on disk -----
    for k, v in PREREG_EXPECTED.items():
        got = pz["preregistration"][k]
        assert abs(got - v) < 1e-12, f"prereg drift: {k} {got} != {v}"
    delta_fraction = pz["preregistration"]["delta_fraction"]

    # --- intact residual per (axis, convention), from Pilot Zero ------------
    intact_resid_pp = {}
    for conv in CONVENTIONS:
        ir = pz["per_convention"][conv]["intact_residual"]
        intact_resid_pp[conv] = {a: 100.0 * ir[a] for a in AXES}

    # --- cross-check MMLU against A01 and against my own CI recompute -------
    a01_nulls = a01["arms"]["7B_base"]["longest_option_floor_by_conv"]
    for conv in CONVENTIONS:
        assert abs(pz["nulls"]["mmlu_content"]["by_convention"][conv]
                   - a01_nulls[conv]) < 1e-12, f"MMLU null mismatch {conv}"
        mine = ici["by_convention"][conv]["residual_pp"]
        theirs = intact_resid_pp[conv]["mmlu_content"]
        assert abs(mine - theirs) < 1e-6, (
            f"intact MMLU residual mismatch {conv}: mine {mine} vs "
            f"pilot-zero {theirs}")

    # --- CI on residual(intact): measured for MMLU (all conventions), and
    #     available from A03's own cells for the QA axes ---------------------
    ci = {}
    for conv in CONVENTIONS:
        ci[conv] = {}
        c = ici["by_convention"][conv]
        ci[conv]["mmlu_content"] = list(c["ci95_pp"])
    a03_intact = {}
    for cell in a03["cells"]:
        if cell["arm"] != "intact":
            continue
        key = {"triviaqa": ("triviaqa", "em"), "popqa": ("popqa", "em"),
               "nq_open": ("nq_open", "em")}
        for axis, (task, iface) in key.items():
            if cell["task"] == task and cell["interface"] == iface:
                a03_intact[axis] = cell
    for axis in ("triviaqa", "popqa", "nq_open"):
        assert axis in a03_intact, f"no A03 intact cell for {axis}"
        for conv in CONVENTIONS:
            ci[conv][axis] = list(a03_intact[axis]["boot_ci95_pp"])
        # the QA nulls are convention-free, so the residual must not move
        for conv in CONVENTIONS:
            assert abs(100.0 * a03_intact[axis]["residual"]
                       - intact_resid_pp[conv][axis]) < 1e-6, (
                f"{axis} residual moved with MMLU convention -- impossible")

    # --- achievable item-level half-widths, from OBSERVED discordance -------
    # Pilot Zero's cells give (diff_mean, one-sided lower 95%) per cell; the
    # gap between them is z95_1s * SE, so SE, the two-sided half-width, and
    # the implied paired discordance rate p_disc = SE^2 * n are all
    # recoverable without re-opening the shards.
    observed_hw = {a: [] for a in AXES}
    observed_pdisc = {a: [] for a in AXES}
    n_items = {}
    for conv in CONVENTIONS:
        for cell in pz["per_convention"][conv]["cells"]:
            a = cell["axis"]
            n_items[a] = cell["n"]
            gap = cell["diff_mean_pp"] - cell["diff_lower95_one_sided_pp"]
            se = gap / z95_one_sided()          # in pp
            observed_hw[a].append(z95_two_sided() * se)
            observed_pdisc[a].append((se / 100.0) ** 2 * cell["n"])
    hw_axis = {a: {"min_pp": min(v), "max_pp": max(v), "n_cells": len(v),
                   "worst_case_used_pp": max(v)}
               for a, v in observed_hw.items()}
    pdisc_axis = {a: {"min": min(v), "max": max(v), "n_cells": len(v)}
                  for a, v in observed_pdisc.items()}
    # sanity: the recovered MMLU p_disc must match the value measured directly
    # from the shards in a04_intact_residual_ci's sibling analysis (0.168708
    # for keep7@200k). Assert only the containment, since the recovered range
    # spans several arms.
    assert pdisc_axis["mmlu_content"]["min"] <= 0.168708 \
        <= pdisc_axis["mmlu_content"]["max"], (
        "recovered MMLU p_disc range does not contain the directly measured "
        f"0.168708: {pdisc_axis['mmlu_content']}")

    # Pilot Zero scored only well-healed arms (>= 200k steps). The gate's grid
    # starts at 2,500 steps, where the arm is barely healed and its paired
    # discordance against intact is much HIGHER -- which makes the item-level
    # CI wider and D6 more likely, not less. Folding the barely-healed arm in
    # is therefore mandatory for an honest D6; omitting it would understate
    # the range and produce optimistic CERTIFIABLE verdicts.
    pdisc_extra_used = {}
    if args.pdisc_extra and os.path.exists(args.pdisc_extra):
        extra = json.load(open(args.pdisc_extra))
        for lab, rec in extra.items():
            pd = rec["p_disc"]
            pdisc_extra_used[lab] = pd
            observed_pdisc["mmlu_content"].append(pd)
        pdisc_axis["mmlu_content"] = {
            "min": min(observed_pdisc["mmlu_content"]),
            "max": max(observed_pdisc["mmlu_content"]),
            "n_cells": len(observed_pdisc["mmlu_content"]),
            "includes_barely_healed_arm": True,
            "barely_healed_source": args.pdisc_extra,
        }

    # --- D4: which (axis, convention) has an inadmissible baseline? ---------
    # Verified from A03's own cells / A01's own convention sweep, not asserted.
    d4 = {conv: {a: False for a in AXES} for conv in CONVENTIONS}
    d4_reason = {}
    # `credit` at 1B: EVERY arm including intact is below the null -> the
    # "floor" is above the whole arm population, so it is not a floor.
    for conv in CONVENTIONS:
        n_below = sum(1 for cell in pz["per_convention"][conv]["cells"]
                      if cell["axis"] == "mmlu_content"
                      and cell["residual_arm"] < 0)
        n_tot = sum(1 for cell in pz["per_convention"][conv]["cells"]
                    if cell["axis"] == "mmlu_content")
        intact_below = intact_resid_pp[conv]["mmlu_content"] < 0
        if intact_below and n_below == n_tot and n_tot > 0:
            d4[conv]["mmlu_content"] = True
            d4_reason[f"{conv}/mmlu_content"] = (
                f"all {n_tot}/{n_tot} damaged cells AND the intact arm are "
                f"below this null -- it is above the entire arm population, "
                f"so it is not a floor (A01 gate-3 found the same at 7B: "
                f"5 of 6 arms below, only intact above by 1.69pp)")

    # --- D5: intact anchor drift -------------------------------------------
    anchor_drift_pp = ici.get("anchor_drift_delta_pp")

    # --- classify every (axis, convention) --------------------------------
    per_axis_conv = {}
    for conv in CONVENTIONS:
        per_axis_conv[conv] = {}
        for axis in AXES:
            r = intact_resid_pp[conv][axis]
            lo, hi = ci[conv][axis]
            hw = hw_axis[axis]["worst_case_used_pp"]
            v, fired, recheck, nums = classify(
                r, lo, hi, hw, d4[conv][axis], anchor_drift_pp,
                delta_fraction, n_items[axis], pdisc_axis[axis])
            rec = {"verdict": v, "conditions_fired": fired,
                   "conditions_needing_recheck": recheck,
                   "residual_intact_pp": r,
                   "residual_intact_ci95_pp": [lo, hi],
                   "delta_pp": nums["delta_pp"],
                   "n_items": n_items[axis],
                   "worst_case_item_halfwidth_pp": hw,
                   "delta_over_halfwidth": (nums["delta_pp"] / hw
                                            if hw else None),
                   "d6_status": nums["d6_status"],
                   "p_disc_crit": nums["p_disc_crit"],
                   "p_disc_observed_range": pdisc_axis[axis],
                   "demoted_by_design": axis in DEMOTED_AXES}
            if f"{conv}/{axis}" in d4_reason:
                rec["d4_reason"] = d4_reason[f"{conv}/{axis}"]
            per_axis_conv[conv][axis] = rec

    # --- expand to the gate's full planned cell family ---------------------
    # A cell = (arm, checkpoint, axis, convention). The guard's verdict is a
    # property of (axis, convention) ONLY, because residual(intact) does not
    # depend on the arm or the checkpoint -- the intact anchor is the same
    # model in every cell. That is why the guard can be fixed before the
    # recovered arms exist.
    counts = {}
    for conv in CONVENTIONS:
        per_v = {"CERTIFIABLE": 0, "NOT_CERTIFIABLE": 0,
                 "NEEDS_RECHECK_AFTER_DATA": 0}
        per_v_decision = dict(per_v)
        for axis in AXES:
            v = per_axis_conv[conv][axis]["verdict"]
            per_v[v] += len(ARMS) * len(CKPT_GRID)
            if axis not in DEMOTED_AXES:
                per_v_decision[v] += len(ARMS) * len(CKPT_GRID)
        counts[conv] = {"all_axes": per_v, "decision_axes_only": per_v_decision,
                        "n_cells_total": len(AXES) * len(ARMS) * len(CKPT_GRID),
                        "n_decision_cells_total": (
                            (len(AXES) - len(DEMOTED_AXES))
                            * len(ARMS) * len(CKPT_GRID))}

    prereg_conv = pz["nulls"]["mmlu_content"]["preregistered_convention"]

    # --- impact on the pre-registered kill clauses -------------------------
    # K1/K2/K3 are all written as "N of the 4 axes". If the guard retires
    # axes, those literal counts change meaning -- and in the worst case
    # become UNSATISFIABLE, which would silently DISABLE a kill clause. That
    # is a loss of falsifiability and must be computed, not assumed.
    K3_RESIDUAL_FLOOR_PP = 5.0          # K3's own verbatim threshold
    kill_impact = {}
    for conv in CONVENTIONS:
        decision_axes = [a for a in AXES if a not in DEMOTED_AXES]
        surviving = [a for a in decision_axes
                     if per_axis_conv[conv][a]["verdict"] != "NOT_CERTIFIABLE"]
        n_surv = len(surviving)
        # K1 clause (a): "NI also accepts on >= 3 of the 4 axes"
        k1_a_literal_satisfiable = n_surv >= 3
        # K2: "... on >= 2 of the 4 axes"
        k2_literal_satisfiable = n_surv >= 2
        # K3: ">= 3 of the 4 axes have the INTACT residual below 5pp"
        n_below5 = sum(1 for a in AXES
                       if per_axis_conv[conv][a]["residual_intact_pp"]
                       < K3_RESIDUAL_FLOOR_PP)
        axes_below5 = [a for a in AXES
                       if per_axis_conv[conv][a]["residual_intact_pp"]
                       < K3_RESIDUAL_FLOOR_PP]
        kill_impact[conv] = {
            "decision_axes": decision_axes,
            "surviving_decision_axes": surviving,
            "n_surviving_decision_axes": n_surv,
            "K1_clause_a_literal_3_of_4_satisfiable": k1_a_literal_satisfiable,
            "K1_DISABLED_if_not_satisfiable": not k1_a_literal_satisfiable,
            "K1_rescaled_threshold": math.ceil(0.75 * n_surv) if n_surv else 0,
            "K2_literal_2_of_4_satisfiable": k2_literal_satisfiable,
            "K2_rescaled_threshold": math.ceil(0.50 * n_surv) if n_surv else 0,
            "K3_n_axes_intact_residual_below_5pp": n_below5,
            "K3_axes_below_5pp": axes_below5,
            "K3_literal_fires": n_below5 >= 3,
            "K3_coupling_warning": (
                "K3 counts a NEGATIVE intact residual as 'below 5pp', so a "
                "convention that pushes axes below their null pushes K3 "
                "towards firing -- but for the wrong reason (an inadmissible "
                "null, not an unmeasurable scale). Axes retired by D1/D4 must "
                "be EXCLUDED from K3's count, not counted as at-floor."
                if any(per_axis_conv[conv][a]["residual_intact_pp"] < 0
                       for a in AXES) else ""),
            "n_cells_remaining_decision": (
                n_surv * len(ARMS) * len(CKPT_GRID)),
            "K1_ge24_cell_precondition_met": (
                n_surv * len(ARMS) * len(CKPT_GRID)) >= 24,
        }

    out = {
        "what": ("A04 margin guard: enumeration of Delta-degeneracy conditions "
                 "+ pre-data classification of every planned gate cell"),
        "date": "2026-08-10",
        "gpu_spent": 0,
        "amends": "A04_GATE_DESIGN.md §2 definition of Delta_x",
        "does_not_change": ("the pre-registered Delta rule itself. The guard "
                           "only declares WHICH CELLS the rule is admissible "
                           "in; it never substitutes a different margin."),
        "preregistration_verified_on_disk": pz["preregistration"],
        "guard_thresholds": GUARD,
        "conditions": CONDITIONS,
        "preregistered_convention": prereg_conv,
        "intact_residual_pp_by_convention": intact_resid_pp,
        "residual_intact_ci95_pp_by_convention": ci,
        "observed_bootstrap_halfwidths_pp": hw_axis,
        "observed_discordance_rates": pdisc_axis,
        "barely_healed_pdisc_folded_in": pdisc_extra_used,
        "n_items_per_axis": n_items,
        "intact_anchor_drift_pp": anchor_drift_pp,
        "per_axis_convention": per_axis_conv,
        "cell_counts_by_convention": counts,
        "kill_clause_impact": kill_impact,
        "gate_structure_assumed": {"arms": ARMS, "checkpoint_grid": CKPT_GRID,
                                   "axes": AXES,
                                   "demoted_axes": sorted(DEMOTED_AXES),
                                   "banned_interfaces": sorted(BANNED_INTERFACES)},
        "sources": {
            "pilot_zero": os.path.relpath(args.pilot_zero, root),
            "a03_4axes": args.a03_4axes,
            "a01_conventions": args.a01_conv,
            "intact_ci_recompute": os.path.relpath(args.intact_ci, root),
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1)

    # ---- human-readable summary ----
    print(f"pre-registered convention = {prereg_conv}\n")
    hdr = (f"{'conv':>7} {'axis':>13} {'resid_pp':>9} {'ci95':>19} "
           f"{'Delta_pp':>9} {'hw_pp':>7} {'D/hw':>6} {'p*_crit':>8} "
           f"{'p_obs_max':>9}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for conv in CONVENTIONS:
        for axis in AXES:
            r = per_axis_conv[conv][axis]
            lo, hi = r["residual_intact_ci95_pp"]
            dh = r["delta_over_halfwidth"]
            pc = r["p_disc_crit"]
            pcs = f"{pc:8.4f}" if pc is not None else "     n/a"
            print(f"{conv:>7} {axis:>13} {r['residual_intact_pp']:9.4f} "
                  f"[{lo:8.4f},{hi:8.4f}] {r['delta_pp']:9.4f} "
                  f"{r['worst_case_item_halfwidth_pp']:7.4f} "
                  f"{dh:6.2f} {pcs} "
                  f"{r['p_disc_observed_range']['max']:9.4f}  {r['verdict']}"
                  + (f"  fired={r['conditions_fired']}"
                     if r["conditions_fired"] else "")
                  + (f"  recheck={r['conditions_needing_recheck']}"
                     if r["conditions_needing_recheck"] else ""))
        print()
    print("cell counts (4 arms x 6 checkpoints):")
    for conv in CONVENTIONS:
        c = counts[conv]
        print(f"  {conv:>7}: all-axes {c['all_axes']}  "
              f"decision-axes {c['decision_axes_only']}")
    print("\nkill-clause impact:")
    for conv in CONVENTIONS:
        k = kill_impact[conv]
        print(f"  {conv:>7}: surviving decision axes "
              f"{k['n_surviving_decision_axes']}/3 {k['surviving_decision_axes']}")
        print(f"           K1 literal '3 of 4' satisfiable="
              f"{k['K1_clause_a_literal_3_of_4_satisfiable']} "
              f"(rescaled -> {k['K1_rescaled_threshold']} of "
              f"{k['n_surviving_decision_axes']}); "
              f"K1 >=24-cell precondition met="
              f"{k['K1_ge24_cell_precondition_met']} "
              f"({k['n_cells_remaining_decision']} cells)")
        print(f"           K2 literal '2 of 4' satisfiable="
              f"{k['K2_literal_2_of_4_satisfiable']} "
              f"(rescaled -> {k['K2_rescaled_threshold']}); "
              f"K3 axes<5pp={k['K3_n_axes_intact_residual_below_5pp']} "
              f"{k['K3_axes_below_5pp']} -> K3 literal fires="
              f"{k['K3_literal_fires']}")
        if k["K3_coupling_warning"]:
            print(f"           WARNING: {k['K3_coupling_warning']}")
    print(f"\nwrote {args.out_json}")


if __name__ == "__main__":
    main()
