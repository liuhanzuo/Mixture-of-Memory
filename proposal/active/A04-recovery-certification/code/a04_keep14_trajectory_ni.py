#!/usr/bin/env python3
"""A04 — the NI(Delta) DISCRIMINATION CURVE along the keep14+fresh2 7B heal run.

THE QUESTION
------------
`STATUS.json:shallow_rung_ni_discrimination_20260812
 .implication_for_pilot_two.cheap_next_steps_dominate[1]` asks for the
INTERMEDIATE checkpoints on the 7B trajectories to be scored, because a rule
observed at a single endpoint tells you its verdict but not its RESOLUTION:

    "locating [the accept boundary] is what 'the gate discriminates' would
     actually mean. Zero training."

So this script does not ask "does keep14 accept" -- `shallow_rung_ni_..._20260812`
already answered NO (REJECT 3/3, margins -28.4624 / -15.0810 / -7.4749 pp at
step 200 000, 20.6-72.4 bootstrap SE from flipping). It asks the strictly
different question:

    is the NI margin MONOTONE in heal step, and does its slope extrapolate to
    an accept boundary anywhere on this trajectory?

WHAT IS DELIBERATELY EXPECTED (fixed in advance, not fitted afterwards)
----------------------------------------------------------------------
`full32_rescore_v2_20260812.trajectory_scan_NOT_run.expectation_to_design_for`
already committed to the prediction for the analogous full32 scan: earlier
checkpoints are LESS converged and "should reject HARDER", so the accept
boundary is expected to be BEYOND the endpoint or off the trajectory entirely.
The same reasoning applies here with more force -- keep14+fresh2 is a DAMAGED
arm (16 of 32 layers) whose endpoint rejects by 7.5-28.5 pp, which is 2 to 3
orders of magnitude more than the endpoint-to-earlier-checkpoint gap could
plausibly close. THIS SCAN IS NOT EXPECTED TO FIND AN ACCEPT, and its value is
the curve, not a verdict flip. A non-monotone curve is itself a finding about
the certification rule and is to be reported as such, not smoothed.

WHAT IS FROZEN AND REUSED, NEVER REIMPLEMENTED
----------------------------------------------
  * `ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
    `qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG`
        <- imported from `pilot_zero_rule_disagreement`
  * `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED`
        <- imported from A03's `analyze_1b_knowledge_floor` via
           `proposal_paths.a03_code_dir()`
  * `_load_arm`, `assert_aligned`, `d4_interface_degenerate`, `ANCHOR`,
    `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED`, `D4_*`, `SD_RUN_1B_PP`
        <- imported from `a04_shallow_rung_ni_7b`, the committed analysis this
           extends, so the guard and the anchor CANNOT drift between the
           endpoint cell and the new trajectory cells.
No metric, null, rule, guard or anchor is re-derived here. Two subagents in this
repository have already produced spurious significance by reimplementing a
metric (`protocol_invariants[4]`).

THE ANCHOR IS NOT TOUCHED (guards G0/G2)
----------------------------------------
`ANCHOR` is imported, not redeclared: vanilla `models/OLMo-2-1124-7B`
(`olmo2_closedbook_results/base_full{,_nqopen}` + `olmo2_mmlu_content_results/
7B_base`, all mode=base / 32 layers / add_bos=false / 8-of-8 shards). Guard G2
forbids substituting it, and `shallow_rung_ni_discrimination_20260812
.intact_anchor.full32_step25000_is_NOT_the_anchor` records exactly why
substituting `full32_step25000` would MANUFACTURE accepts (it scores below
vanilla on all four axes, so it shrinks every Delta AND lowers every target).
`Delta = 0.10 * residual(intact)` is likewise imported through the guard, never
recomputed against a different intact.

BOOTSTRAP SEEDS ARE DISJOINT FROM EVERY ARCHIVED CELL
-----------------------------------------------------
Pilot Zero used arm offsets `ai in {0,1}`, the step100k pass `ai in 100..102`,
and `a04_shallow_rung_ni_7b` `ai in 200..203`. This script uses `ai in 300..`,
so no archived cell can be perturbed and no two cells here collide. The
endpoint cell is RE-DERIVED here under its own archived offset (`ai=201`) purely
as a consistency check against `evidence/a04_shallow_rung_ni_7b.json`, and the
script FAILS if it does not reproduce.

CROSS-SCALE (this is a hard limit on what may be said)
------------------------------------------------------
`sd_run` -- SEED variance -- exists ONLY at 1B (S=3, keep12@5000). Every 7B rung
here has EXACTLY ONE seed, and the historical 7B ladder's seeds are UNRECORDED
(`--seed` postdates them; trainer `afdfa66` called no seeding function), so a 7B
`sd_run` is not computable and not retrospectively reconstructible. Any
deficit/sd_run column is therefore an explicitly-labelled 1B-imported
extrapolation. It is NOT licensed to say a 7B deficit is large "relative to seed
variance". What IS licensed: the 7B deficits, Delta, lo95 bounds, margins, and
"no realisable perturbation of the ITEM SAMPLE flips these verdicts".

Nor are the three trajectory points seeds OF EACH OTHER: they are three
checkpoints of ONE run, so their spread is heal-progress plus data-order, not
independent-run variance. A trend across them is a within-run trajectory, not a
sampling distribution.

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

from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    mmlu_content_norm_vec,  # noqa: F401  (used via _load_arm)
    ni_rule,
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

# The guard, the anchor and the loader of the committed endpoint analysis --
# IMPORTED so they cannot drift between the endpoint cell and the new cells.
from a04_shallow_rung_ni_7b import (  # noqa: E402
    ANCHOR,
    D2_RESIDUAL_FLOOR_PP,
    D4_CONSTANT_FRAC,
    D4_TIE_FRAC,
    SD_RUN_1B_PP,
    Z95_TWO_SIDED,
    _load_arm,
    assert_aligned,
    d4_interface_degenerate,
)

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]

# The trajectory, in heal-step order. `step200k` is the ARCHIVED endpoint cell
# (dir names exactly as in `a04_shallow_rung_ni_7b.arm_specs`); the two earlier
# points are the new ones. Naming differs because the archive predates this
# driver -- the dirs are pinned here rather than pattern-guessed.
TRAJECTORY = [
    (128000, {"mmlu": "A04_7B_keep14f2_step128000",
              "cb": "A04_7B_keep14f2_step128000",
              "nq": "A04_7B_keep14f2_step128000_nqopen"}),
    (153500, {"mmlu": "A04_7B_keep14f2_step153500",
              "cb": "A04_7B_keep14f2_step153500",
              "nq": "A04_7B_keep14f2_step153500_nqopen"}),
    (200000, {"mmlu": "7B_keep14_step200000",
              "cb": "keep14_step200k",
              "nq": "keep14_step200k_nqopen"}),
]

# Archived endpoint margins to reproduce, `split` convention, decision axes only
# (`STATUS.json:shallow_rung_ni_discrimination_20260812.NI_results_split
#  .keep14fresh2_step200k`). Reproducing these is a HARD assertion: it proves the
# imported guard/anchor/rule really are the ones that produced the archive.
ARCHIVED_ENDPOINT_MARGINS_PP = {
    "triviaqa": -28.4624,
    "popqa": -15.0810,
    "mmlu_content": -7.4749,
}
# Endpoint bootstrap offset as used by the archive (`ARM_INDEX` = 200 + index,
# keep14fresh2 was index 1 -> 201). New cells start at 300 to stay disjoint.
ENDPOINT_ARM_INDEX = 201
NEW_ARM_INDEX_BASE = 300


def _arm_name(step):
    return f"keep14fresh2_step{step}"


def _seed_off(arm_index, axis):
    """Same functional form as the archive: 97*arm_index + 13*axis_index."""
    return 97 * arm_index + 13 * AXES.index(axis)


def shard_integrity_report(mm_root, cb_root, specs):
    """Independent, EXPLICIT shard-completeness evidence.

    `load_shards` already hard-asserts 8/8 + exact-n + no-duplicate-item_id, but
    it asserts on the file COUNT and then on `seen_idx == set(range(8))`. This
    function records the actual index SET, the per-shard row counts and the
    merged total in the output JSON, because the dispatch requires the assertion
    RESULT to be inspectable rather than merely to have not raised. The repo has
    been corrupted before by a silently merged 5-of-8 set, and
    `shallow_rung_ni_discrimination_20260812.two_disk_gotcha` records a
    zwfy6-resident arm that is merged-without-shards and would fail here.
    """
    import glob as _glob
    rep = {}
    for label, spec in specs.items():
        rep[label] = {}
        for axis, stem in (("mmlu_content", "mmlu"), ("triviaqa", "triviaqa"),
                           ("popqa", "popqa"), ("nq_open", "nq_open")):
            key = {"mmlu_content": "mmlu", "nq_open": "nq"}.get(axis, "cb")
            if not spec.get(key):
                continue
            root = mm_root if key == "mmlu" else cb_root
            d = spec[key] if os.path.isabs(spec[key]) \
                else os.path.join(root, spec[key])
            files = sorted(_glob.glob(
                os.path.join(d, f"per_example_{stem}_shard*of8.jsonl")))
            idx, per_shard, total, dup = [], {}, 0, 0
            seen = set()
            for f in files:
                i = int(os.path.basename(f).split("_shard")[1].split("of")[0])
                idx.append(i)
                c = 0
                with open(f) as fh:
                    for line in fh:
                        if line.strip():
                            r = json.loads(line)
                            if r["item_id"] in seen:
                                dup += 1
                            seen.add(r["item_id"])
                            c += 1
                per_shard[str(i)] = c
                total += c
            exp = EXPECTED_N["mmlu" if axis == "mmlu_content" else axis]
            rep[label][axis] = {
                "dir": d,
                "shard_index_set": sorted(idx),
                "shard_index_set_equals_0_to_7": sorted(idx) == list(range(8)),
                "n_shard_files": len(files),
                "rows_per_shard": per_shard,
                "merged_n": total,
                "expected_n": exp,
                "merged_n_equals_expected": total == exp,
                "n_duplicate_item_ids": dup,
            }
            # Hard-fail here too, so a bad set can never reach the rule.
            if sorted(idx) != list(range(8)):
                raise SystemExit(
                    f"FATAL {label}/{axis}: shard index set {sorted(idx)} != "
                    "{0..7} -- refusing to merge a partial set")
            if total != exp:
                raise SystemExit(
                    f"FATAL {label}/{axis}: merged {total} != expected {exp}")
            if dup:
                raise SystemExit(
                    f"FATAL {label}/{axis}: {dup} duplicate item_ids")
    return rep


def guard_cell(data, arm_names, nulls, conv, axis):
    """Guard D1-D6 on the intact anchor, exactly as the endpoint analysis does.

    D1-D4 are properties of the ANCHOR (and of the null), so they are identical
    to the archived cell by construction. D6 depends on `p_disc` -- the fraction
    of items where an ARM differs from the anchor -- so it is recomputed over the
    trajectory arms rather than inherited: a different arm set could in principle
    change which cells are certifiable, and pretending otherwise would be
    assuming the answer.
    """
    iv = data["intact_7B_base"][axis]
    nv = (nulls["mmlu_content"]["vectors"][conv]
          if axis == "mmlu_content" else nulls[axis]["vector"])
    d = np.asarray(iv, float) - np.asarray(nv, float)
    resid = float(d.mean())
    resid_pp = 100.0 * resid
    _m, lo, hi, p = paired_bootstrap(d, seed=SEED + 700 + 13 * AXES.index(axis))
    delta_pp = 100.0 * PREREG["delta_fraction"] * resid
    n = EXPECTED_N["mmlu" if axis == "mmlu_content" else axis]
    pstar = n * (delta_pp / (100.0 * Z95_TWO_SIDED)) ** 2

    pdisc = {a: float((np.asarray(data[a][axis], float)
                       != np.asarray(iv, float)).mean()) for a in arm_names}
    pdisc_max = max(pdisc.values())
    hw = {a: 100.0 * Z95_TWO_SIDED * float(np.sqrt(v / n))
          for a, v in pdisc.items()}
    d4 = {a: d4_interface_degenerate(data, a, axis, nulls)
          for a in list(arm_names) + ["intact_7B_base"]}
    null_a = (nulls["mmlu_content"]["by_convention"][conv]
              if axis == "mmlu_content" else nulls[axis]["acc"])
    all_below = all(float(np.asarray(data[a][axis], float).mean()) < null_a
                    for a in list(arm_names) + ["intact_7B_base"])
    cond = {
        "D1_residual_negative": bool(resid_pp < 0),
        "D2_residual_at_zero": bool(0 <= resid_pp <= D2_RESIDUAL_FLOOR_PP),
        "D3_ci_straddles_zero": bool(lo < 0 < hi),
        "D4_null_inadmissible": bool(all_below
                                     or any(v["degenerate"] for v in d4.values())),
        "D6_delta_finer_than_instrument": bool(pdisc_max > pstar),
    }
    fatal = [k for k, v in cond.items() if v]
    return {
        "residual_intact_pp": resid_pp,
        "null": float(np.asarray(nv, float).mean()),
        "reported_intact": float(np.asarray(iv, float).mean()),
        "ci95_pp": [100.0 * lo, 100.0 * hi],
        "boot_p": p,
        "delta_pp": delta_pp,
        "n": n,
        "pstar_crit_7B_recomputed": pstar,
        "p_disc_by_arm": pdisc,
        "p_disc_max": pdisc_max,
        "hw95_pp_by_arm": hw,
        "delta_over_hw_worst": delta_pp / max(hw.values()),
        "d4_interface_by_arm": d4,
        "all_arms_below_null": all_below,
        "conditions": cond,
        "fatal_conditions": fatal,
        "classification": "CERTIFIABLE" if not fatal else "NOT_CERTIFIABLE",
        "decision_axis": axis not in DEMOTED_AXES,
    }


def monotone_report(steps, margins):
    """Is the margin monotone non-decreasing in heal step? Reported, never fitted.

    Also reports the naive linear extrapolation to margin=0 -- as a DISTANCE, not
    as a prediction. The trajectory has 3 points spanning 128k-200k of a 200k run;
    a straight line through the last two is not a healing model, and the JSON
    labels it so. If the extrapolated step is negative or absurdly large, that IS
    the answer: the boundary is not on this trajectory.
    """
    s = np.asarray(steps, float)
    m = np.asarray(margins, float)
    diffs = np.diff(m)
    mono_up = bool(np.all(diffs >= 0))
    mono_down = bool(np.all(diffs <= 0))
    # last-interval slope, pp per 1000 heal steps
    slope_last = float((m[-1] - m[-2]) / (s[-1] - s[-2]) * 1000.0)
    slope_all = float(np.polyfit(s, m, 1)[0] * 1000.0)
    out = {
        "steps": [int(x) for x in steps],
        "margins_pp": [float(x) for x in margins],
        "successive_differences_pp": [float(x) for x in diffs],
        "monotone_nondecreasing": mono_up,
        "monotone_nonincreasing": mono_down,
        "monotone_either_direction": bool(mono_up or mono_down),
        "slope_last_interval_pp_per_1k_steps": slope_last,
        "slope_ols_all_points_pp_per_1k_steps": slope_all,
    }
    for nm, sl in (("last_interval", slope_last), ("ols_all_points", slope_all)):
        if sl > 0:
            need = -m[-1] / sl * 1000.0
            out[f"extrapolated_steps_to_margin_zero_{nm}"] = float(s[-1] + need)
            out[f"extra_heal_steps_needed_{nm}"] = float(need)
            out[f"extra_heal_steps_needed_{nm}_as_multiple_of_run_length"] = \
                float(need / s[-1])
        else:
            out[f"extrapolated_steps_to_margin_zero_{nm}"] = None
            out[f"extra_heal_steps_needed_{nm}"] = None
            out[f"note_{nm}"] = ("slope <= 0: the margin is not improving, so no "
                                 "forward extrapolation reaches margin=0 -- the "
                                 "accept boundary is NOT on this trajectory in "
                                 "this direction")
    out["extrapolation_caveat"] = (
        "A straight line through 3 checkpoints of ONE run is NOT a healing "
        "model: heal curves are concave and this arm's last 72k steps moved the "
        "margin by a tiny fraction of its 7.5-28.5pp deficit. The extrapolated "
        "step is reported as a DISTANCE to make the scale of the gap explicit, "
        "never as a forecast that training that long would accept.")
    return out


def between_checkpoint_tests(data, axis, steps):
    """Is each successive checkpoint-to-checkpoint MOVE distinguishable from zero?

    `monotone_report` reports the SIGN of each successive difference. A sign is
    not a finding: if a step-to-step move is inside item-sampling variability,
    then "the margin is non-monotone" would be an artefact of reading noise as
    structure -- the exact error `same-harness-runs-bit-identical` warns about in
    the other direction.

    So each ADJACENT PAIR of checkpoints gets a paired item bootstrap on its
    own per-item difference vector (`paired_bootstrap`, imported, two-sided
    95% CI + bootstrap p). Note these are two DIFFERENT checkpoints, so this is
    NOT the harness-jitter question that
    `full32_rescore_v2_20260812.correction_to_the_jitter_premise` settled
    (that one found 0.0 pp jitter on a FIXED ckpt). Here the models genuinely
    differ; the question is whether the ITEM SAMPLE is large enough to resolve
    the difference between them.

    Seeds: SEED + 900 + 13*axis_index + 7*pair_index, disjoint from the guard's
    SEED+700+13*axis and from every ni_rule offset (97*arm_index + 13*axis with
    arm_index >= 200).
    """
    out = {}
    for pi in range(len(steps) - 1):
        a, b = _arm_name(steps[pi]), _arm_name(steps[pi + 1])
        d = (np.asarray(data[b][axis], float)
             - np.asarray(data[a][axis], float))
        mean, lo, hi, p = paired_bootstrap(
            d, seed=SEED + 900 + 13 * AXES.index(axis) + 7 * pi)
        n_up = int((d > 0).sum())
        n_down = int((d < 0).sum())
        # Two criteria, BOTH reported, because they can disagree at the boundary:
        # the bootstrap of a 0/1-difference metric is DISCRETE, so a percentile
        # can land exactly ON zero. `lo < 0 < hi` is then False (reading as
        # "resolved") while the two-sided bootstrap p is 0.0514 (reading as "not
        # resolved"). Silently picking whichever is more favourable is how a tie
        # becomes a result, so the disagreement is surfaced instead.
        ci_excl = bool(not (lo < 0 < hi))
        p_sig = bool(p < 0.05)
        out[f"{steps[pi]}->{steps[pi+1]}"] = {
            "acc_delta_pp": 100.0 * float(mean),
            "ci95_pp": [100.0 * lo, 100.0 * hi],
            "boot_p_two_sided": p,
            "ci_straddles_zero": bool(lo < 0 < hi),
            "ci95_excludes_zero": ci_excl,
            "boot_p_below_0p05": p_sig,
            # conservative AND of the two criteria: a move counts as resolved
            # only if the CI excludes zero AND p < 0.05.
            "distinguishable_from_zero_at_95": bool(ci_excl and p_sig),
            "criteria_disagree": bool(ci_excl != p_sig),
            "criteria_disagreement_note": (
                "CI-excludes-zero and p<0.05 disagree; the discrete bootstrap of "
                "a 0/1 metric can place a percentile exactly at zero. Treated as "
                "NOT resolved (conservative AND)." if ci_excl != p_sig else None),
            "n_items_improved": n_up,
            "n_items_regressed": n_down,
            "n_items_changed": n_up + n_down,
            "n": int(d.size),
        }
    return out


def protocol_asserted(raw_root, driver_logs, driver_path):
    """Confirm batch_size and chat_template FROM THE INVOCATION, not by inference.

    `A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md` (MAIN, 2026-08-13, written while this
    scan was still running) records a real artefact defect: the scoring harness
    writes `mode / keep_front_layers / n_fresh_layers / num_hidden_layers /
    ckpt_step / ckpt / base_model / add_bos / max_new_tokens` into
    `summary.json:meta` and **neither `batch_size` nor `chat_template`**. So the
    two most decision-critical fields cannot be reconstructed from the result
    dirs after the fact -- they are knowable only from the invocation.

    That matters because batch size is NOT a free parameter here:
    `full32_rescore_v2_20260812.sensitivity_bs48_probe` measured bs32 -> bs48
    flipping 12/14267 popqa and 10/3610 nq_open items (bf16 numerics depend on
    left-pad width). A trajectory scored at a different bs than its anchor and
    its endpoint would be an artefact of a protocol change rather than of heal
    progress.

    This function therefore PARSES the driver's own echoed lines
    (`DRIVER START ... mmlu_bs=<N> cb_bs=<N>` and the per-axis `START ... bs=<N>`)
    out of the launch logs, and re-reads the batch-size defaults out of the driver
    SOURCE, then hard-fails on any mismatch with the frozen values. `grep`-ing my
    own source is not evidence of what ran; the log lines are, because the driver
    echoes the variables it actually passes to the harness. Both are recorded.

    chat_template: asserted STRUCTURALLY rather than by a flag, because there is
    no flag -- neither harness has a chat-template code path at all (the only
    match for `chat_template` in either file is a docstring). A protocol that
    cannot be switched on cannot have been switched on. `add_bos is False` (the
    other half of the base protocol) IS recorded per-dir in `summary.json:meta`
    and is asserted with `is False`, never `is not True` (the latter passes
    silently on None).
    """
    import re
    frozen = {"cb_bs": 32, "mmlu_bs": 16}
    out = {"frozen_expectation": frozen,
           "why_bs_is_not_free": (
               "full32_rescore_v2_20260812.sensitivity_bs48_probe: bs32->bs48 "
               "flipped 12/14267 popqa and 10/3610 nq_open items"),
           "artefact_gap_acknowledged": (
               "summary.json:meta records neither batch_size nor chat_template "
               "(A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md), so both are confirmed "
               "from the INVOCATION below, not inferred from the result dirs"),
           "from_driver_logs": {}, "from_driver_source": {},
           "add_bos_from_summaries": {}, "chat_template": {}}

    for label, lg in driver_logs.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            raise SystemExit(
                f"FATAL: driver log {p} absent -- batch size cannot be confirmed "
                "from the invocation, and summary.json does not record it. "
                "Refusing to publish cells whose protocol cannot be established.")
        txt = open(p).read()
        hdr = re.search(r"DRIVER START.*?mmlu_bs=(\d+)\s+cb_bs=(\d+)", txt)
        if not hdr:
            raise SystemExit(f"FATAL: no 'DRIVER START ... mmlu_bs=.. cb_bs=..' "
                             f"line in {p}")
        per_axis = {}
        for kind in ("closedbook", "nq_open", "mmlu"):
            mm = re.findall(rf"{kind} START \S+ bs=(\d+)", txt)
            per_axis[kind] = sorted({int(x) for x in mm})
        rec = {"log": lg,
               "header_mmlu_bs": int(hdr.group(1)),
               "header_cb_bs": int(hdr.group(2)),
               "per_axis_bs_echoed": per_axis}
        # hard-fail on any deviation from the frozen protocol
        if rec["header_cb_bs"] != frozen["cb_bs"] \
                or rec["header_mmlu_bs"] != frozen["mmlu_bs"]:
            raise SystemExit(f"FATAL protocol deviation in {p}: {rec} != {frozen}")
        for kind, want in (("closedbook", frozen["cb_bs"]),
                           ("nq_open", frozen["cb_bs"]),
                           ("mmlu", frozen["mmlu_bs"])):
            if per_axis[kind] != [want]:
                raise SystemExit(
                    f"FATAL protocol deviation in {p}: {kind} ran at "
                    f"bs={per_axis[kind]}, frozen value is {want}")
        out["from_driver_logs"][label] = rec

    dp = os.path.join(raw_root, driver_path)
    if os.path.isfile(dp):
        src = open(dp).read()
        for var, want in (("MMLU_BS", frozen["mmlu_bs"]),
                          ("CB_BS", frozen["cb_bs"])):
            mm = re.search(rf'^{var}="\$\{{{var}:-(\d+)\}}"', src, re.M)
            got = int(mm.group(1)) if mm else None
            out["from_driver_source"][var] = got
            if got != want:
                raise SystemExit(f"FATAL: driver default {var}={got} != {want}")
        out["from_driver_source"]["note"] = (
            "source defaults agree with the values the logs show were used; the "
            "LOGS are the evidence, the source is the corroboration")

    # add_bos: the half of the base protocol that IS in the artefacts
    for step, spec in TRAJECTORY:
        for key, root in (("cb", "olmo2_closedbook_results"),
                          ("nq", "olmo2_closedbook_results"),
                          ("mmlu", "olmo2_mmlu_content_results")):
            if not spec.get(key):
                continue
            sp = os.path.join(raw_root, root, spec[key], "summary.json")
            if not os.path.isfile(sp):
                raise SystemExit(f"FATAL: {sp} absent")
            meta = json.load(open(sp)).get("meta", {})
            ab = meta["add_bos"]          # KeyError = loud, desired
            if ab is not False:           # `is False`, never `is not True`
                raise SystemExit(
                    f"FATAL {sp}: add_bos={ab!r}; base protocol requires False. "
                    "(Asserted with `is False`; `is not True` would pass on None.)")
            out["add_bos_from_summaries"][f"{step}|{key}"] = False

    out["chat_template"] = {
        "value": False,
        "how_established": (
            "STRUCTURAL, not a flag: neither scripts/eval_olmo2_closedbook_qa.py "
            "nor scripts/eval_olmo2_mmlu_content.py contains a chat-template code "
            "path -- the only occurrence of the string in either file is a "
            "docstring line. A protocol that cannot be switched on cannot have "
            "been switched on. The harness md5s are identical on both disks and "
            "to the copies that produced the anchor and the endpoint."),
        "assertion_form_note": (
            "add_bos is asserted with `is False`, never `is not True`, because "
            "`is not True` passes silently on None."),
        "why_it_must_be_False": (
            "OLMo-2 is a BASE LM with no SFT/RL; a chat template would be unfair "
            "AND would break comparability with every existing cell. Project-wide "
            "rule, not a local choice."),
    }
    return out


def output_shape_and_flips(data, axis, steps):
    """LABELLED DIAGNOSTIC (never enters a verdict): what does a resolved
    checkpoint-to-checkpoint move look like at the item level?

    A resolved accuracy DROP mid-heal admits at least three readings, and they
    have different consequences for a certification rule:
      (a) genuine loss of parametric knowledge between checkpoints;
      (b) an output-FORMAT/degeneracy shift (empty strings, repetition loops, a
          collapse onto one constant answer) that costs EM without costing
          knowledge -- guard D4's concern, here measured per checkpoint;
      (c) churn concentrated in a handful of items.
    So per checkpoint this records the empty-prediction rate, mean prediction
    length, the most-frequent-constant share and the distinct-prediction count;
    and per adjacent interval the right->wrong / wrong->right counts and the
    fraction of items whose prediction STRING is unchanged.

    For the generative axes only: `pred` is a free-form string there. MMLU-content
    is a scored-option interface with no comparable output-shape question.
    """
    out = {"per_checkpoint": {}, "per_interval": {}}
    for st in steps:
        rows = data[_arm_name(st)].get(f"_{axis}_rows")
        if rows is None:
            return None
        preds = [(r.get("pred") or "").strip() for r in rows]
        n = len(preds)
        cnt = {}
        for p in preds:
            cnt[p] = cnt.get(p, 0) + 1
        out["per_checkpoint"][str(st)] = {
            "empty_pred_frac": sum(1 for x in preds if not x) / n,
            "mean_pred_chars": sum(len(x) for x in preds) / n,
            "top_constant_frac": max(cnt.values()) / n,
            "n_distinct_preds": len(cnt),
            "n": n,
        }
    for i in range(len(steps) - 1):
        a = data[_arm_name(steps[i])][axis]
        b = data[_arm_name(steps[i + 1])][axis]
        ra = data[_arm_name(steps[i])][f"_{axis}_rows"]
        rb = data[_arm_name(steps[i + 1])][f"_{axis}_rows"]
        a = np.asarray(a, float)
        b = np.asarray(b, float)
        r2w = int(((a == 1) & (b == 0)).sum())
        w2r = int(((a == 0) & (b == 1)).sum())
        same = sum(1 for x, y in zip(ra, rb)
                   if (x.get("pred") or "").strip() == (y.get("pred") or "").strip())
        out["per_interval"][f"{steps[i]}->{steps[i+1]}"] = {
            "right_to_wrong": r2w,
            "wrong_to_right": w2r,
            "net_items": w2r - r2w,
            "net_pp": 100.0 * (w2r - r2w) / a.size,
            "identical_pred_string_frac": same / a.size,
        }
    out["reading_note"] = (
        "Diagnostic only. If empty_pred_frac stays ~0, top_constant_frac stays "
        "low and n_distinct_preds does not collapse, then a resolved accuracy "
        "drop is NOT an output-degeneracy artefact and reading (a) -- real "
        "knowledge churn -- is the surviving explanation. A low "
        "identical_pred_string_frac shows the generation is churning far more "
        "than the EM number moves, i.e. EM is a coarse read on a model whose "
        "outputs are being rewritten wholesale between checkpoints.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--driver_log_128000",
                    default="logs/a04_keep14_traj_128000.out")
    ap.add_argument("--driver_log_153500",
                    default="logs/a04_keep14_traj_153500.out")
    args = ap.parse_args()

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    # 0. PROTOCOL, confirmed from the INVOCATION before anything is scored.
    # Hard-fails on deviation, so a non-comparable margin cannot enter the record.
    proto = protocol_asserted(
        args.raw_root,
        {"step128000_on_73": args.driver_log_128000,
         "step153500_on_82": args.driver_log_153500},
        "proposal/active/A04-recovery-certification/code/"
        "a04_keep14_trajectory_axes_driver.sh")
    proto["endpoint_step200000_predates_this_driver"] = (
        "the step200000 cell is the ARCHIVED endpoint (2026-08-02), so it has no "
        "log from this driver. Its protocol is recovered in "
        "full32_rescore_v2_20260812.protocol_recovered and re-verified here from "
        "the archive's OWN launch logs: logs/cb_driver_73.out echoes 'START "
        "base_full ... bs=32' AND 'START keep14_step200k ... bs=32', and "
        "logs/nqopen_driver_73.log echoes 'START base_full_nqopen ... bs=32'. "
        "MMLU-content: scripts/p06_run_transferred.sh leaves BS unset, so "
        "_run_olmo2_mmlu_content.sh's default BS=16 produced both 7B_base and "
        "7B_keep14_step200000, and `git log -p --follow` on that file shows "
        "exactly ONE revision of the BS= line (commit d2e28f2), i.e. 16 is not "
        "later drift. Hence the new cells match the endpoint AND the anchor.")

    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    for step, spec in TRAJECTORY:
        arm_specs[_arm_name(step)] = spec
    arm_names = [_arm_name(s) for s, _ in TRAJECTORY]

    # 1. explicit shard-integrity evidence BEFORE anything is scored
    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    # nan-free + identical item_id sequence across ALL arms, per axis
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}
    # the endpoint keeps its ARCHIVED offset so the reproduction check is exact
    arm_index[_arm_name(200000)] = ENDPOINT_ARM_INDEX

    # 2. guard, then NI, per convention
    guard, per_conv = {}, {}
    for conv in TIE_CONVS:
        guard[conv] = {axis: guard_cell(data, arm_names, nulls, conv, axis)
                       for axis in AXES}
        cells, retired = [], []
        for step, _ in TRAJECTORY:
            arm = _arm_name(step)
            for axis in AXES:
                g = guard[conv][axis]
                if g["classification"] == "NOT_CERTIFIABLE":
                    retired.append({
                        "arm": arm, "step": step, "axis": axis,
                        "fatal_conditions": g["fatal_conditions"],
                        "residual_intact_pp": g["residual_intact_pp"],
                        "delta_pp": g["delta_pp"],
                        "p_disc": g["p_disc_by_arm"][arm],
                        "pstar_crit": g["pstar_crit_7B_recomputed"],
                        "ni_run": False,
                        "note": ("NI NOT RUN; excluded from the decision family. "
                                 "Never to be reported as 'NI rejected'.")})
                    continue
                r = ni_rule(data[arm][axis], data["intact_7B_base"][axis],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=_seed_off(arm_index[arm], axis))
                null_a = (nulls["mmlu_content"]["by_convention"][conv]
                          if axis == "mmlu_content" else nulls[axis]["acc"])
                arm_resid = reported[arm][axis] - null_a
                ir = g["residual_intact_pp"] / 100.0
                deficit = g["residual_intact_pp"] - 100.0 * arm_resid
                se = ((r["diff_mean_pp"] - r["diff_lower95_one_sided_pp"]) / 1.6449
                      if r["diff_mean_pp"] != r["diff_lower95_one_sided_pp"]
                      else None)
                margin = r["diff_lower95_one_sided_pp"] + r["delta_pp"]
                cells.append({
                    "arm": arm, "step": step, "axis": axis,
                    "decision_axis": axis not in DEMOTED_AXES,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact_7B_base"][axis],
                    "null": null_a,
                    "residual_arm_pp": 100.0 * arm_resid,
                    "residual_intact_pp": g["residual_intact_pp"],
                    "residual_fraction_recovered": (arm_resid / ir if ir > 0
                                                    else None),
                    "deficit_pp": deficit,
                    "margin_pp": margin,
                    "bootstrap_se_pp": se,
                    "se_to_flip_NI": (abs(margin) / se) if se else None,
                    "delta_over_deficit": (r["delta_pp"] / deficit
                                           if deficit else None),
                    **r,
                })
        n_dec = sum(1 for c in cells if c["decision_axis"])
        per_conv[conv] = {
            "intact_residual_pp": {x: guard[conv][x]["residual_intact_pp"]
                                   for x in AXES},
            "delta_pp": {x: guard[conv][x]["delta_pp"] for x in AXES},
            "cells": cells,
            "retired_cells": retired,
            "decision_family_size_full": len(arm_names) * len(DECISION_AXES),
            "decision_family_size_after_guard": n_dec,
            "ratio_rule": {a: ratio_rule(reported[a],
                                         reported["intact_7B_base"],
                                         PREREG["rho"],
                                         [x for x in AXES if x in data[a]])
                           for a in arm_names},
        }

    # 3. reproduce the ARCHIVED endpoint margins -- hard assertion
    repro = {"tolerance_pp": 5e-4, "per_axis": {}, "all_reproduced": True}
    for axis, want in ARCHIVED_ENDPOINT_MARGINS_PP.items():
        got = [c["margin_pp"] for c in per_conv["split"]["cells"]
               if c["arm"] == _arm_name(200000) and c["axis"] == axis]
        ok = bool(got) and abs(got[0] - want) < 5e-4
        repro["per_axis"][axis] = {
            "archived_margin_pp": want,
            "recomputed_margin_pp": got[0] if got else None,
            "abs_diff_pp": abs(got[0] - want) if got else None,
            "reproduced": ok,
        }
        repro["all_reproduced"] &= ok
    if not repro["all_reproduced"]:
        raise SystemExit(
            "FATAL: the imported guard/anchor/rule do NOT reproduce the "
            "archived keep14fresh2@200k margins "
            f"{ARCHIVED_ENDPOINT_MARGINS_PP} -> {repro['per_axis']}. Something "
            "drifted; refusing to publish a trajectory measured against a "
            "different baseline than the endpoint it is compared to.")

    # 4. the curve, per axis, decision axes AND the demoted one (labelled)
    curve = {}
    for conv in TIE_CONVS:
        curve[conv] = {}
        for axis in AXES:
            cs = sorted([c for c in per_conv[conv]["cells"] if c["axis"] == axis],
                        key=lambda c: c["step"])
            if len(cs) < 2:
                continue
            steps = [c["step"] for c in cs]
            curve[conv][axis] = {
                "decision_axis": axis not in DEMOTED_AXES,
                "per_step": {str(c["step"]): {
                    "reported": c["reported"],
                    "margin_pp": c["margin_pp"],
                    "deficit_pp": c["deficit_pp"],
                    "lo95_pp": c["diff_lower95_one_sided_pp"],
                    "delta_pp": c["delta_pp"],
                    "recovery_fraction": c["residual_fraction_recovered"],
                    "bootstrap_se_pp": c["bootstrap_se_pp"],
                    "se_to_flip_NI": c["se_to_flip_NI"],
                    "ni_accept": c["ni_accept"],
                } for c in cs},
                "monotonicity": monotone_report(
                    steps, [c["margin_pp"] for c in cs]),
                "accuracy_monotonicity": monotone_report(
                    steps, [100.0 * c["reported"] for c in cs]),
                "between_checkpoint_paired_tests": between_checkpoint_tests(
                    data, axis, steps),
                "any_accept": bool(any(c["ni_accept"] for c in cs)),
            }

    # 5. verdict per checkpoint (the >=2-of-3 decision-axis bar)
    verdict = {}
    for conv in TIE_CONVS:
        verdict[conv] = {}
        for step, _ in TRAJECTORY:
            arm = _arm_name(step)
            dec = [c for c in per_conv[conv]["cells"]
                   if c["arm"] == arm and c["decision_axis"]]
            acc = [c["axis"] for c in dec if c["ni_accept"]]
            n_surv = len(dec)
            need = int(np.ceil(0.50 * n_surv)) if n_surv else None
            verdict[conv][arm] = {
                "step": step,
                "n_decision_axes_surviving_guard": n_surv,
                "n_decision_axes_accepting": len(acc),
                "axes_accepting": acc,
                "threshold_ge2of3_rescaled": need,
                "NI_OBSERVED_TO_ACCEPT": bool(n_surv and acc
                                              and len(acc) >= need),
                "all_reject": bool(n_surv and not acc),
            }

    dec_mono = {ax: curve["split"][ax]["monotonicity"]["monotone_nondecreasing"]
                for ax in DECISION_AXES if ax in curve["split"]}
    any_accept = any(v["NI_OBSERVED_TO_ACCEPT"]
                     for v in verdict["split"].values())

    # A NEGATIVE successive difference only counts as a real non-monotonicity if
    # the corresponding checkpoint-to-checkpoint accuracy move is itself
    # distinguishable from zero on the item sample. Otherwise "non-monotone" is
    # reading sampling variability as structure -- so the headline must not
    # promote it. Both facts are carried; neither is hidden.
    nonmono_detail = {}
    for ax in DECISION_AXES:
        if ax not in curve["split"]:
            continue
        mo = curve["split"][ax]["monotonicity"]
        tests = curve["split"][ax]["between_checkpoint_paired_tests"]
        keys = list(tests)
        neg = [(keys[i], d) for i, d in
               enumerate(mo["successive_differences_pp"]) if d < 0]
        nonmono_detail[ax] = {
            "negative_intervals": [
                {"interval": k, "margin_change_pp": d,
                 "acc_delta_pp": tests[k]["acc_delta_pp"],
                 "acc_ci95_pp": tests[k]["ci95_pp"],
                 "acc_boot_p_two_sided": tests[k]["boot_p_two_sided"],
                 "acc_move_resolved_at_95": tests[k][
                     "distinguishable_from_zero_at_95"]}
                for k, d in neg],
            "has_negative_interval": bool(neg),
            "has_RESOLVED_negative_interval": bool(
                neg and any(tests[k]["distinguishable_from_zero_at_95"]
                            for k, _ in neg)),
        }
    resolved_nonmono = {ax: v["has_RESOLVED_negative_interval"]
                        for ax, v in nonmono_detail.items()}

    # labelled diagnostic on the generative axes, so a resolved DROP can be told
    # apart from an output-degeneracy artefact. Never enters a verdict.
    traj_steps = [s for s, _ in TRAJECTORY]
    output_diag = {}
    for ax in ("triviaqa", "popqa", "nq_open"):
        r = output_shape_and_flips(data, ax, traj_steps)
        if r is not None:
            output_diag[ax] = r

    if any_accept:
        headline = "UNEXPECTED_ACCEPT_ON_KEEP14_TRAJECTORY"
    elif any(resolved_nonmono.values()):
        headline = "CONSTANT_REJECT_MARGIN_NON_MONOTONE_AND_RESOLVED"
    elif all(dec_mono.values()):
        headline = "CONSTANT_REJECT_MARGIN_MONOTONE_BOUNDARY_OFF_TRAJECTORY"
    else:
        headline = "CONSTANT_REJECT_MARGIN_NON_MONOTONE_WITHIN_ITEM_NOISE"

    # 6. cross-scale extrapolation, explicitly labelled and never a 7B variance
    sens = {"note": (
        "sd_run is a 1B, S=3, keep12@5000 quantity. Every 7B rung has exactly "
        "ONE seed and the historical 7B ladder's seeds are UNRECORDED, so NO 7B "
        "sd_run exists or can be reconstructed. The column below is an "
        "explicitly-labelled cross-scale extrapolation and is NOT licensed as a "
        "7B variance statement. The three trajectory points are also NOT seeds "
        "of each other -- they are three checkpoints of ONE run."),
        "cross_scale_sd_run_1B_imported": {}}
    for c in per_conv["split"]["cells"]:
        if not c["decision_axis"]:
            continue
        sd = SD_RUN_1B_PP[c["axis"]]
        sens["cross_scale_sd_run_1B_imported"][f"{c['arm']}|{c['axis']}"] = {
            "deficit_pp": c["deficit_pp"],
            "sd_run_1B_pp": sd,
            "deficit_over_sd_run_1B": c["deficit_pp"] / sd,
            "delta_pp": c["delta_pp"],
        }

    out = {
        "gate": "A04_keep14fresh2_trajectory_NI_discrimination_curve_7B",
        "question": ("does the NI(Delta) margin improve MONOTONELY along the "
                     "keep14+fresh2 7B heal trajectory, and does its slope "
                     "locate an accept boundary? (cheap_next_steps_dominate[1])"),
        "date": "2026-08-13",
        "headline_verdict": headline,
        "expectation_fixed_in_advance": (
            "NOT expected to find an accept: earlier checkpoints are LESS "
            "converged and the endpoint already rejects by 7.5-28.5pp at "
            "20.6-72.4 bootstrap SE. The deliverable is the discrimination "
            "curve, and a non-monotone curve is itself a finding about the "
            "certification rule, to be reported as such."),
        "full32_half_not_run": (
            "the four intermediate full32_dolmino ckpts (step5000/10000/15000/"
            "20000, 81.6 GiB each) are wzc1-ONLY; measured cross-disk rate 16 "
            "MiB/s = ~89 min/ckpt before any GPU. Out of this dispatch's node "
            "budget (LOCAL/.21 running SparseForge). See "
            "full32_rescore_v2_20260812.trajectory_scan_NOT_run."),
        "prereg": {
            "gate_design": "A04_GATE_DESIGN.md 2",
            "margin_guard": "A04_MARGIN_GUARD_PREREG.md 4",
            "delta_fraction": PREREG["delta_fraction"],
            "rho": PREREG["rho"],
            "commit_freezing_constants": PREREG["commit"],
            "ni_definition": ("accept iff one-sided lower 95% bound on "
                              "residual(arm)-residual(intact) > -Delta, "
                              "Delta = 0.10*residual(intact); imported ni_rule"),
            "decision_axes": DECISION_AXES,
            "demoted_axes": sorted(DEMOTED_AXES),
            "delta_never_substituted": True,
            "anchor_never_changed": True,
            "n_boot": N_BOOT,
            "base_seed": SEED,
        },
        "intact_anchor": {
            "choice": "vanilla models/OLMo-2-1124-7B (mode=base, 32 layers)",
            "dirs": ANCHOR,
            "imported_from": "a04_shallow_rung_ni_7b.ANCHOR (not redeclared)",
            "guard_G2": "Delta and anchor never substituted",
        },
        "trajectory": [{"step": s, "dirs": d} for s, d in TRAJECTORY],
        "protocol_asserted": proto,
        "bootstrap_offsets": {"arm_index": arm_index,
                              "form": "97*arm_index + 13*axis_index",
                              "disjoint_from": ("pilot_zero ai in {0,1}; "
                                                "step100k ai in 100..102; "
                                                "shallow_rung ai in 200..203"),
                              "endpoint_uses_archived_offset": ENDPOINT_ARM_INDEX},
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "archived_endpoint_reproduction": repro,
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "per_convention": per_conv,
        "discrimination_curve": curve,
        "verdict_by_convention": verdict,
        "decision_axis_margin_monotone_nondecreasing_split": dec_mono,
        "non_monotonicity_resolved_split": resolved_nonmono,
        "non_monotonicity_detail_split": nonmono_detail,
        "output_shape_and_flips_diagnostic": output_diag,
        "any_checkpoint_accepts": any_accept,
        "sensitivity": sens,
        "gpu_note": ("CPU-only. The GPU cost was the 4-axis scoring of "
                     "step128000 (.73) and step153500 (.82); this analysis "
                     "loads only per-example shards."),
    }

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---- console ----------------------------------------------------------
    print("=" * 104)
    print("PROTOCOL CONFIRMED FROM THE INVOCATION (summary.json records neither "
          "batch_size nor chat_template)")
    print("=" * 104)
    for label, r in proto["from_driver_logs"].items():
        print(f"  {label:<20} {r['log']}  cb_bs={r['header_cb_bs']} "
              f"mmlu_bs={r['header_mmlu_bs']}  per-axis echoed="
              f"{r['per_axis_bs_echoed']}")
    print(f"  driver source defaults: "
          f"{ {k: v for k, v in proto['from_driver_source'].items() if k != 'note'} }")
    print(f"  add_bos is False on all {len(proto['add_bos_from_summaries'])} "
          "result dirs (asserted with `is False`)")
    print(f"  chat_template={proto['chat_template']['value']} (structural: no "
          "chat-template code path exists in either harness)")
    print()
    print("=" * 104)
    print("GUARD D1-D6 (`split`), evaluated BEFORE NI, over the trajectory arm set")
    print("=" * 104)
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
    print("=" * 104)
    print("ARCHIVED ENDPOINT REPRODUCTION (proves guard+anchor+rule did not drift)")
    print("=" * 104)
    for axis, v in repro["per_axis"].items():
        print(f"  {axis:<14} archived {v['archived_margin_pp']:>10.4f}  "
              f"recomputed {v['recomputed_margin_pp']:>10.4f}  "
              f"diff {v['abs_diff_pp']:.2e}  "
              f"{'OK' if v['reproduced'] else 'MISMATCH'}")
    print()
    print("=" * 104)
    print("NI(Delta) DISCRIMINATION CURVE -- `split` convention")
    print("=" * 104)
    print(f"{'axis':<14}{'step':>8}{'acc%':>9}{'recov%':>8}{'deficit':>10}"
          f"{'lo95':>11}{'Delta':>8}{'margin':>10}{'SE':>8}{'SE_flip':>9}  NI")
    for axis in AXES:
        if axis not in curve["split"]:
            continue
        for st, v in curve["split"][axis]["per_step"].items():
            print(f"{axis:<14}{int(st):>8}{100*v['reported']:>8.3f}%"
                  f"{100*(v['recovery_fraction'] or float('nan')):>7.1f}%"
                  f"{v['deficit_pp']:>10.4f}{v['lo95_pp']:>11.4f}"
                  f"{v['delta_pp']:>8.4f}{v['margin_pp']:>10.4f}"
                  f"{v['bootstrap_se_pp'] or float('nan'):>8.3f}"
                  f"{v['se_to_flip_NI'] or float('nan'):>9.1f}"
                  f"  {'ACCEPT' if v['ni_accept'] else 'REJECT'}"
                  + ("" if curve["split"][axis]["decision_axis"] else " (demoted)"))
        mo = curve["split"][axis]["monotonicity"]
        print(f"{'':<14}{'  -> margin diffs':<18}"
              f"{['%+.4f' % d for d in mo['successive_differences_pp']]}"
              f"  monotone_up={mo['monotone_nondecreasing']}"
              f"  slope_last={mo['slope_last_interval_pp_per_1k_steps']:+.5f} pp/1k")
        ex = mo.get("extra_heal_steps_needed_last_interval")
        if ex is not None:
            print(f"{'':<14}   extra heal steps to margin=0 (naive line): "
                  f"{ex:,.0f} = "
                  f"{mo['extra_heal_steps_needed_last_interval_as_multiple_of_run_length']:.1f}x "
                  "the whole 200k run")
        for k, t in curve["split"][axis][
                "between_checkpoint_paired_tests"].items():
            print(f"{'':<14}   {k:<16} acc {t['acc_delta_pp']:+.4f}pp "
                  f"CI95[{t['ci95_pp'][0]:+.4f},{t['ci95_pp'][1]:+.4f}] "
                  f"p={t['boot_p_two_sided']:.4f} "
                  f"{'RESOLVED' if t['distinguishable_from_zero_at_95'] else 'within item noise'} "
                  f"(+{t['n_items_improved']}/-{t['n_items_regressed']} of {t['n']})")
        print()
    print("=" * 104)
    print("VERDICT per checkpoint (`split`, >=2 of 3 decision axes)")
    print("=" * 104)
    for arm, v in verdict["split"].items():
        print(f"  step {v['step']:<7} surviving={v['n_decision_axes_surviving_guard']}"
              f" accepting={v['n_decision_axes_accepting']} {v['axes_accepting']}"
              f" -> {'ACCEPTS' if v['NI_OBSERVED_TO_ACCEPT'] else 'ALL REJECT'}")
    print(f"\nHEADLINE: {headline}")
    if output_diag:
        print()
        print("=" * 104)
        print("OUTPUT-SHAPE DIAGNOSTIC (labelled; never enters a verdict)")
        print("=" * 104)
        for ax, v in output_diag.items():
            print(f"  {ax}")
            print(f"    {'step':>8}{'empty%':>10}{'meanchars':>11}"
                  f"{'top_const%':>12}{'n_distinct':>12}")
            for st, c in v["per_checkpoint"].items():
                print(f"    {int(st):>8}{100*c['empty_pred_frac']:>9.3f}%"
                      f"{c['mean_pred_chars']:>11.2f}"
                      f"{100*c['top_constant_frac']:>11.3f}%"
                      f"{c['n_distinct_preds']:>12}")
            for k, c in v["per_interval"].items():
                print(f"    {k:<16} r->w {c['right_to_wrong']:>4}  "
                      f"w->r {c['wrong_to_right']:>4}  net {c['net_items']:>+5} "
                      f"({c['net_pp']:+.4f} pp)  identical pred string "
                      f"{100*c['identical_pred_string_frac']:.2f}%")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
