#!/usr/bin/env python3
"""A04 -- NI(Delta) discrimination CURVE along the full32_dolmino 7B trajectory.

WHAT THIS ANSWERS
-----------------
`STATUS.json:shallow_rung_ni_discrimination_20260812
 .implication_for_pilot_two.cheap_next_steps_dominate[1]`, the half that the
keep14 pass could not do:

    "full32 sits at 97.7% MMLU recovery with one axis 1.86 SE from accepting --
     the accept BOUNDARY is on that trajectory, and locating it is what 'the gate
     discriminates' would actually mean."

`full32_dolmino@step25000` is the ONLY NI accept in all of A04 (mmlu_content
margin +1.0495 pp). This script scores the four earlier checkpoints on the same
four axes under the same frozen protocol and reads the margin as a curve.

PRE-REGISTERED READINGS -- `A04_FULL32_TRAJECTORY_PREREG.md`, committed 537d323
BEFORE any of the four checkpoints had a summary.json:
  (a) accept only at step25000            -> the boundary is located in (20000, 25000]
  (b) accept ALSO at an earlier checkpoint -> step25000's accept is NOT the
      product of convergence, and reading it as "recovery" would be reading a
      continued-pretraining fluctuation as an achievement. (b) is WORSE for A04
      and is reported just as readily.

ZERO STRUCTURAL DAMAGE. full32 is `keep_front_layers=32 / n_fresh_layers=0` --
all 32 pretrained layers present, nothing transplanted, nothing pruned. It is a
CONTINUED-PRETRAINING control. Every conclusion here is about CPT drift on the
heal corpus, NOT about recovery from structural injury
(`shallow_rung_ni_discrimination_20260812.the_load_bearing_new_finding.caveat`).
The archived dirs nonetheless label it `mode=pruned`, because `load_pruned_model`
labels every ckpt-loaded model that way; that label is reproduced verbatim so the
new cells remain differenceable against the endpoint.

EVERYTHING DECISION-BEARING IS IMPORTED, NOTHING REIMPLEMENTED
--------------------------------------------------------------
`ni_rule / ratio_rule / build_nulls / load_shards / mmlu_content_norm_vec /
qa_metric_vec / EXPECTED_N / AXES / DEMOTED_AXES / PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap / TIE_CONVS / N_BOOT / SEED`
from A03's `analyze_1b_knowledge_floor`; and `ANCHOR / _load_arm / assert_aligned
/ d4_interface_degenerate / SD_RUN_1B_PP` plus the guard constants from
`a04_shallow_rung_ni_7b`, so the guard and the anchor CANNOT drift between the
archived endpoint cell and the new cells. The trajectory-shaped helpers
(`shard_integrity_report / guard_cell / monotone_report /
between_checkpoint_tests / output_shape_and_flips / protocol_asserted`) are
imported from `a04_keep14_trajectory_ni`, which is the committed keep14 half of
this same dispatch -- the two halves are therefore the same code by construction,
not merely the same intent.

ANCHOR IS NOT SUBSTITUTABLE. Vanilla `../models/OLMo-2-1124-7B`. Guard G2
explicitly FORBIDS using `full32_step25000` as the anchor: it scores below
vanilla on all four axes, so substituting it would shrink every Delta AND lower
every target -- manufacturing accepts. Delta is never substituted either.

ANALYSIS NODE IS PINNED TO .73 (numpy 2.5.1).
`neighbour_variability_20260813.reproducibility_defect_found` measured that
`Generator.multinomial` differs in 19/10000 rows between numpy 2.5.1 (.73) and
2.4.6 (.82) at the same seed, moving margins by up to 0.0053 pp -- an order of
magnitude LARGER than the 5e-4 pp archived-cell reproduction threshold this
script hard-fails on. The archived shallow-rung cells were produced on .73, so
the bootstrap must run there. GPU scoring is deterministic and was split across
.73/.82; the statistics are not split.

BOOTSTRAP SEEDS. New arm indices 500..503 (form `97*arm_index + 13*axis_index`);
the endpoint keeps its ARCHIVED offset (`arm_index=203`, from
`a04_shallow_rung_ni_7b`'s `ARM_INDEX = {a: 200 + i}` with full32 fourth) so the
reproduction check is exact and no archived cell is perturbed. Disjoint from
pilot_zero {0,1}, step100k 100..102, shallow_rung 200..203, keep14 300..301,
neighbour 400..408. Guard offset SEED+2700 and interval offset SEED+2900 avoid
the used 700 / 900 / 1700 / 1900 / 2400.

usage (on .73 ONLY):
  /opt/conda/envs/torch-base/bin/python a04_full32_trajectory_ni.py \
    --raw_root /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
    --out_json <proposal>/evidence/a04_full32_trajectory_ni.json
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import sys

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

# Canonical rules/scorers/nulls -- IMPORTED, never reimplemented.
from pilot_zero_rule_disagreement import (  # noqa: E402
    AXES,
    DEMOTED_AXES,
    EXPECTED_N,
    PREREG,
    build_nulls,
    ni_rule,
    ratio_rule,
)
# anchor, per-arm loader, alignment assertion, guard internals and the labelled
# 1B sd_run -- from the archived endpoint's OWN analysis module.
from a04_shallow_rung_ni_7b import (  # noqa: E402
    ANCHOR,
    SD_RUN_1B_PP,
    _load_arm,
    assert_aligned,
)
# trajectory-shaped helpers -- from the committed keep14 half of this dispatch.
import a04_keep14_trajectory_ni as K14  # noqa: E402
from a04_keep14_trajectory_ni import (  # noqa: E402
    guard_cell,
    monotone_report,
    protocol_asserted,
    shard_integrity_report,
)
from analyze_1b_knowledge_floor import (  # noqa: E402
    SEED,
    TIE_CONVS,
    paired_bootstrap,
)

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]

# ---------------------------------------------------------------------------
# the trajectory. step25000 is the ARCHIVED endpoint -- already scored 2026-08-03
# and already analysed in `a04_shallow_rung_ni_7b`; it is NOT re-scored, it is
# the point the four new ones are read against.
# ---------------------------------------------------------------------------
TRAJECTORY = [
    (5000,  {"mmlu": "A04_7B_full32_step5000",
             "cb": "A04_7B_full32_step5000",
             "nq": "A04_7B_full32_step5000_nqopen"}),
    (10000, {"mmlu": "A04_7B_full32_step10000",
             "cb": "A04_7B_full32_step10000",
             "nq": "A04_7B_full32_step10000_nqopen"}),
    (15000, {"mmlu": "A04_7B_full32_step15000",
             "cb": "A04_7B_full32_step15000",
             "nq": "A04_7B_full32_step15000_nqopen"}),
    (20000, {"mmlu": "A04_7B_full32_step20000",
             "cb": "A04_7B_full32_step20000",
             "nq": "A04_7B_full32_step20000_nqopen"}),
    (25000, {"mmlu": "7B_full32_step25000",
             "cb": "full32_step25000",
             "nq": "full32_step25000_nqopen"}),
]

# Archived endpoint margins to reproduce, `split` convention, decision axes only
# (`STATUS.json:shallow_rung_ni_discrimination_20260812.NI_results_split
#  .full32_dolmino_step25k`, confirmed unchanged by `full32_rescore_v2_20260812`
# which re-scored it and got a BIT-IDENTICAL result). Reproducing these is a HARD
# assertion: it proves the imported guard/anchor/rule really are the ones that
# produced the archive, so the four new points are on the same scale.
ARCHIVED_ENDPOINT_MARGINS_PP = {
    "triviaqa": -0.603544,
    "popqa": -4.539146,
    "mmlu_content": +1.049530,
}
ENDPOINT_ARM_INDEX = 203      # a04_shallow_rung_ni_7b: 200 + index, full32 is 4th
NEW_ARM_INDEX_BASE = 500      # disjoint from 0,1 / 100..102 / 200..203 / 300..301 / 400..408
GUARD_SEED_OFF = 2700         # disjoint from 700, 1700
INTERVAL_SEED_OFF = 2900      # disjoint from 900, 1900, 2400

# The RATIO(rho) accept at the endpoint, to be reproduced and then tracked along
# the trajectory (`full32_rescore_v2_20260812.result`).
ARCHIVED_ENDPOINT_RATIO = 0.8514950516430542


def _arm_name(step):
    return f"full32_dolmino_step{step}"


def _seed_off(arm_index, axis):
    """Same functional form as every archived cell: 97*arm_index + 13*axis."""
    return 97 * arm_index + 13 * AXES.index(axis)


def between_checkpoint_tests(data, axis, steps):
    """Is each successive checkpoint-to-checkpoint MOVE distinguishable from zero?

    Same construction and the same conservative AND-of-two-criteria as
    `a04_keep14_trajectory_ni.between_checkpoint_tests` (a sign is not a finding;
    the discrete bootstrap of a 0/1 metric can put a percentile exactly on zero,
    so CI-excludes-zero and p<0.05 are BOTH reported and disagreement is
    surfaced rather than resolved in the favourable direction).

    Reimplemented here for ONE reason only: the imported version hard-codes
    `_arm_name` from the keep14 module, so it would look for `keep14fresh2_step*`
    arms. The statistics (`paired_bootstrap`, imported) and the decision logic are
    identical; only the arm-naming and the seed offset differ. Seeds:
    SEED + 2900 + 13*axis_index + 7*pair_index.
    """
    out = {}
    for pi in range(len(steps) - 1):
        a, b = _arm_name(steps[pi]), _arm_name(steps[pi + 1])
        d = (np.asarray(data[b][axis], float)
             - np.asarray(data[a][axis], float))
        mean, lo, hi, p = paired_bootstrap(
            d, seed=SEED + INTERVAL_SEED_OFF + 13 * AXES.index(axis) + 7 * pi)
        n_up = int((d > 0).sum())
        n_down = int((d < 0).sum())
        ci_excl = bool(not (lo < 0 < hi))
        p_sig = bool(p < 0.05)
        out[f"{steps[pi]}->{steps[pi+1]}"] = {
            "acc_delta_pp": 100.0 * float(mean),
            "ci95_pp": [100.0 * lo, 100.0 * hi],
            "boot_p_two_sided": p,
            "ci_straddles_zero": bool(lo < 0 < hi),
            "ci95_excludes_zero": ci_excl,
            "boot_p_below_0p05": p_sig,
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


def output_shape_and_flips(data, axis, steps):
    """LABELLED DIAGNOSTIC, never enters a verdict. Delegates to the keep14
    implementation by temporarily pointing its module-level `_arm_name` at this
    trajectory's naming -- the analysis code itself is not copied, so the two
    halves of the dispatch cannot diverge in what they measure."""
    saved = K14._arm_name
    K14._arm_name = _arm_name
    try:
        return K14.output_shape_and_flips(data, axis, steps)
    finally:
        K14._arm_name = saved


def protocol_every_driver_start(raw_root, progress_logs, expect_steps):
    """STRICTER than the imported gate: assert EVERY `DRIVER START` line, and
    assert step coverage.

    Why this exists. The imported `protocol_asserted` takes the FIRST
    `DRIVER START ... mmlu_bs=.. cb_bs=..` match per file. That is sufficient
    when one file holds one driver invocation, which is how the keep14 half ran.
    Here two invocations per node wrote to the same append-only progress log, so
    "the first line was compliant" would be a weaker statement than "every
    invocation was compliant". This checks every line and records which steps
    each log accounts for, so the four published cells are each tied to an
    invocation whose batch sizes were echoed by the driver itself.

    PROVENANCE INCIDENT, recorded rather than smoothed over. The per-invocation
    stdout file `logs/a04_full32_traj_15000.out` was TRUNCATED: after step15000
    had already finished, a duplicate launch of the same driver was issued to
    .82, its shell redirection `>` re-created the file, and the driver then
    exited via its own `gpu_free_or_die` guard ("REFUSE: 140116MiB of GPU memory
    held by another process"). So the surviving stdout file begins mid-run and no
    longer contains its `DRIVER START` header -- and the imported gate correctly
    HARD-FAILED on it rather than publishing cells whose protocol it could not
    establish. Two things saved the run: the driver `tee -a`s every note to a
    per-node progress log (append-only, so it cannot be clobbered), and the
    refused duplicate never touched a GPU. The progress logs are therefore the
    authoritative protocol record here, and they are what both gates read.
    """
    import re
    frozen = {"cb_bs": 32, "mmlu_bs": 16}
    rec = {"frozen_expectation": frozen,
           "why_stricter": "asserts EVERY DRIVER START line, not just the first",
           "truncation_incident": (
               "logs/a04_full32_traj_15000.out lost its DRIVER START header to a "
               "duplicate launch's `>` redirection AFTER step15000 had completed; "
               "that duplicate was then refused by the driver's own "
               "gpu_free_or_die guard and never ran. The imported gate hard-failed "
               "on the truncated file, which is the designed behaviour. The "
               "append-only per-node progress logs hold the full record and are "
               "used instead."),
           "per_log": {}}
    steps_seen = set()
    for label, lg in progress_logs.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: progress log {p} absent -- refusing to "
                             "publish cells whose protocol cannot be established.")
        txt = open(p).read()
        starts = re.findall(r"DRIVER START.*?mmlu_bs=(\d+)\s+cb_bs=(\d+)\s+"
                            r"steps='([^']*)'", txt)
        if not starts:
            raise SystemExit(f"FATAL: no DRIVER START line in {p}")
        per_start = []
        for mb, cb, st in starts:
            if int(cb) != frozen["cb_bs"] or int(mb) != frozen["mmlu_bs"]:
                raise SystemExit(
                    f"FATAL protocol deviation in {p}: DRIVER START mmlu_bs={mb} "
                    f"cb_bs={cb} != {frozen}")
            ss = [int(x) for x in st.split()]
            steps_seen.update(ss)
            per_start.append({"mmlu_bs": int(mb), "cb_bs": int(cb), "steps": ss})
        per_axis = {}
        for kind in ("closedbook", "nq_open", "mmlu"):
            mm = re.findall(rf"{kind} START \S+ bs=(\d+)", txt)
            per_axis[kind] = sorted({int(x) for x in mm})
        for kind, want in (("closedbook", frozen["cb_bs"]),
                           ("nq_open", frozen["cb_bs"]),
                           ("mmlu", frozen["mmlu_bs"])):
            if per_axis[kind] != [want]:
                raise SystemExit(
                    f"FATAL protocol deviation in {p}: {kind} ran at "
                    f"bs={per_axis[kind]}, frozen value is {want}")
        rec["per_log"][label] = {"log": lg, "driver_starts": per_start,
                                 "per_axis_bs_echoed": per_axis}
    missing = set(expect_steps) - steps_seen
    if missing:
        raise SystemExit(
            f"FATAL: no driver invocation accounts for step(s) {sorted(missing)}; "
            "every published cell must be tied to an echoed invocation.")
    rec["steps_accounted_for"] = sorted(steps_seen)
    rec["all_four_new_steps_accounted_for"] = True
    return rec


def em_vs_contains_verbosity(data, axis, steps):
    """LABELLED DIAGNOSTIC -- never enters a verdict. Is an EM drop a loss of
    knowledge, or a loss of BREVITY?

    Why this is here rather than left out. The frozen decision metric is EM, and
    it stays EM: `contains` is a different, laxer metric and swapping it would
    change the rule, not measure it. But on this trajectory the two metrics
    DISAGREE in a way that changes what the EM number means, and the disagreement
    lands exactly on the cell A04's headline rests on
    (`triviaqa@step25000`, the axis reported as "1.86 SE from accepting"). So for
    every adjacent pair this records, among the items that went EM right->wrong:
    how many still CONTAIN the gold string, and how the prediction LENGTH moved.

    An item that answered `Joshua` at step20000 and `Joshua 10:13-14` at
    step25000 is scored as a regression by EM while still emitting the correct
    entity. That is an instruction-following / stopping-behaviour change, which is
    what continued pretraining on a raw corpus (no SFT) would be expected to
    produce -- and it is NOT the same phenomenon as forgetting.

    The number is reported. The verdict is still computed on EM.
    """
    out = {"metric_note": (
        "decision metric is and remains EM. `contains` is reported ONLY to "
        "characterise WHAT an EM move consists of, never to re-score a cell."),
        "per_interval": {}}
    for pi in range(len(steps) - 1):
        a, b = _arm_name(steps[pi]), _arm_name(steps[pi + 1])
        ra = data[a].get(f"_{axis}_rows")
        rb = data[b].get(f"_{axis}_rows")
        if ra is None or rb is None:
            return None
        ma = {r["item_id"]: r for r in ra}
        mb = {r["item_id"]: r for r in rb}
        ids = [i for i in ma if i in mb]
        r2w = [i for i in ids if ma[i].get("em") == 1 and mb[i].get("em") == 0]
        w2r = [i for i in ids if ma[i].get("em") == 0 and mb[i].get("em") == 1]
        def _contains(m, k):
            v = m[k].get("contains")
            return 1 if v in (1, True) else 0
        still = sum(_contains(mb, i) for i in r2w)
        def _mlen(m, keys):
            if not keys:
                return None
            return float(np.mean([len(m[k].get("pred") or "") for k in keys]))
        out["per_interval"][f"{steps[pi]}->{steps[pi+1]}"] = {
            "n_em_right_to_wrong": len(r2w),
            "n_em_wrong_to_right": len(w2r),
            "of_the_right_to_wrong_still_contains_gold": still,
            "frac_of_regressions_that_still_contain_gold": (
                still / len(r2w) if r2w else None),
            "mean_pred_chars_on_regressed_items_before": _mlen(ma, r2w),
            "mean_pred_chars_on_regressed_items_after": _mlen(mb, r2w),
            "reading": (
                "a regression that still CONTAINS the gold answer while the "
                "prediction gets longer is a stopping/verbosity change, not "
                "demonstrated forgetting"),
        }
    return out


def neighbour_precondition(curve_axis, step, steps):
    """`A04_GATE_DESIGN.md` §2.0.2 -- PER-AXIS, never blanket.

    Any accept reported by this gate must be accompanied by the same axis's
    margin at the immediately adjacent saved checkpoints on BOTH sides, or an
    explicit statement that none exist. This trajectory is saved every 5000
    steps, so the neighbours of stepN are stepN-5000 and stepN+5000; step5000 has
    no lower neighbour and step25000 no upper one.
    """
    idx = steps.index(step)
    lo = steps[idx - 1] if idx > 0 else None
    hi = steps[idx + 1] if idx + 1 < len(steps) else None
    per = curve_axis["per_step"]
    rec = {"axis_spacing_steps": 5000, "checkpoint": step}
    for label, nb in (("lower", lo), ("upper", hi)):
        if nb is None:
            rec[label] = {
                "exists": False,
                "why": ("no saved checkpoint on this side of the trajectory "
                        f"({'first' if label == 'lower' else 'last'} point)")}
        else:
            c = per[str(nb)]
            rec[label] = {"exists": True, "step": nb,
                          "margin_pp": c["margin_pp"],
                          "ni_accept": c["ni_accept"]}
    both = [rec[k] for k in ("lower", "upper") if rec[k]["exists"]]
    rec["n_neighbours_present"] = len(both)
    rec["all_present_neighbours_also_accept"] = (
        bool(both) and all(x["ni_accept"] for x in both))
    rec["verdict"] = (
        "ACCEPT_SURVIVES_ITS_NEIGHBOURS" if rec["all_present_neighbours_also_accept"]
        else "ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--driver_log_5000", default="logs/a04_full32_traj_5000.out")
    ap.add_argument("--driver_log_10000", default="logs/a04_full32_traj_10000.out")
    ap.add_argument("--driver_log_15000", default="logs/a04_full32_traj_15000.out")
    ap.add_argument("--driver_log_20000", default="logs/a04_full32_traj_20000.out")
    ap.add_argument(
        "--progress_log_73",
        default="logs/a04_full32_traj_progress_28.87.115.232.log",
        help="append-only per-node driver log (.73); holds steps 5000+10000")
    ap.add_argument(
        "--progress_log_82",
        default="logs/a04_full32_traj_progress_28.86.53.217.log",
        help="append-only per-node driver log (.82); holds steps 15000+20000")
    ap.add_argument("--stage_log_73", default="logs/a04_full32_stage_73.out")
    ap.add_argument("--stage_log_82", default="logs/a04_full32_stage_82.out")
    args = ap.parse_args()

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    # ---- 0. PROTOCOL, from the INVOCATION, before anything is scored -------
    # FAIL-CLOSED. Parses cb_bs/mmlu_bs out of the drivers' own echoed lines,
    # because summary.json:meta records NEITHER batch_size NOR chat_template
    # (A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md). bs is not free:
    # full32_rescore_v2_20260812.sensitivity_bs48_probe measured bs32->bs48
    # flipping 12/14267 popqa and 10/3610 nq_open items.
    #
    # Read from the APPEND-ONLY per-node progress logs, not the per-invocation
    # stdout files: one of the latter was truncated by a duplicate launch's `>`
    # redirection (see protocol_every_driver_start.__doc__). The imported gate
    # hard-failed on that truncated file, correctly, and is re-run below against
    # the intact logs so that BOTH gates pass on the same evidence.
    proto = protocol_asserted(
        args.raw_root,
        {"node_73_steps_5000_and_10000": args.progress_log_73,
         "node_82_steps_15000_and_20000": args.progress_log_82},
        "proposal/active/A04-recovery-certification/code/"
        "a04_full32_trajectory_axes_driver.sh")
    # stricter re-check: EVERY DRIVER START line, plus step coverage
    proto["every_driver_start_asserted"] = protocol_every_driver_start(
        args.raw_root,
        {"node_73": args.progress_log_73, "node_82": args.progress_log_82},
        [5000, 10000, 15000, 20000])
    proto["endpoint_step25000_predates_this_driver"] = (
        "the step25000 cell is the ARCHIVED endpoint (scored 2026-08-03, "
        "re-scored BIT-IDENTICALLY 2026-08-12), so it has no log from this "
        "driver. Its protocol is recorded in "
        "full32_rescore_v2_20260812.protocol_recovered, recovered there from the "
        "archive's OWN scheduler logs logs/cb_full32_step25000{,_nqopen}_sched.out "
        "which echo bs=32; MMLU-content bs=16 is _run_olmo2_mmlu_content.sh's "
        "default, which p06_run_transferred.sh leaves unset and whose BS= line "
        "has exactly ONE revision in git history (d2e28f2). So the four new cells "
        "match BOTH the endpoint and the anchor.")
    proto["full32_is_mode_pruned_in_the_archive"] = (
        "verified on disk 2026-08-13: olmo2_closedbook_results/full32_step25000/"
        "summary.json:meta reads mode=pruned keep_front_layers=32 n_fresh_layers=0 "
        "num_hidden_layers=32 base_model=../models/OLMo-2-1124-7B. full32 is "
        "structurally UNDAMAGED (all 32 pretrained layers, nothing transplanted); "
        "load_pruned_model labels any ckpt-loaded model mode=pruned regardless of "
        "shape. The archive's label is reproduced verbatim so the new cells stay "
        "differenceable against the endpoint.")

    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    for step, spec in TRAJECTORY:
        arm_specs[_arm_name(step)] = spec
    arm_names = [_arm_name(s) for s, _ in TRAJECTORY]
    traj_steps = [s for s, _ in TRAJECTORY]

    # ---- 1. explicit shard-integrity evidence BEFORE anything is scored ----
    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}
    arm_index[_arm_name(25000)] = ENDPOINT_ARM_INDEX   # archived offset, exact repro

    # ---- 2. guard, then NI, per tie convention ----------------------------
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

    # ---- 3. reproduce the ARCHIVED endpoint -- HARD assertion --------------
    repro = {"tolerance_pp": 5e-4, "per_axis": {}, "all_reproduced": True,
             "why_this_is_a_hard_gate": (
                 "if the imported guard/anchor/rule do not return the archived "
                 "step25000 margins, then the four new points are being compared "
                 "against a different baseline than the endpoint they are read "
                 "against, and the curve is meaningless.")}
    for axis, want in ARCHIVED_ENDPOINT_MARGINS_PP.items():
        got = [c["margin_pp"] for c in per_conv["split"]["cells"]
               if c["arm"] == _arm_name(25000) and c["axis"] == axis]
        ok = bool(got) and abs(got[0] - want) < 5e-4
        repro["per_axis"][axis] = {
            "archived_margin_pp": want,
            "recomputed_margin_pp": got[0] if got else None,
            "abs_diff_pp": abs(got[0] - want) if got else None,
            "reproduced": ok,
        }
        repro["all_reproduced"] &= ok
    ratio_end = per_conv["split"]["ratio_rule"][_arm_name(25000)]["mean_ratio"]
    repro["ratio_mean_ratio"] = {
        "archived": ARCHIVED_ENDPOINT_RATIO,
        "recomputed": ratio_end,
        "abs_diff": abs(ratio_end - ARCHIVED_ENDPOINT_RATIO),
        "reproduced": bool(abs(ratio_end - ARCHIVED_ENDPOINT_RATIO) < 1e-9),
    }
    if not repro["all_reproduced"]:
        raise SystemExit(
            "FATAL: the imported guard/anchor/rule do NOT reproduce the archived "
            f"full32@25k margins {ARCHIVED_ENDPOINT_MARGINS_PP} -> "
            f"{repro['per_axis']}. Something drifted; refusing to publish a "
            "trajectory measured against a different baseline than its endpoint. "
            "NOTE: this assertion is numpy-version sensitive at the 5e-3 pp "
            "level (neighbour_variability_20260813.reproducibility_defect_found) "
            "-- run on .73 (numpy 2.5.1), the node that produced the archive.")

    # ---- 4. the curve, per axis --------------------------------------------
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
                "accepting_steps": [c["step"] for c in cs if c["ni_accept"]],
                "argmax_margin_step": max(cs, key=lambda c: c["margin_pp"])["step"],
                "best_margin_pp": max(c["margin_pp"] for c in cs),
            }

    # ---- 5. verdict per checkpoint (the >=2-of-3 decision-axis bar) --------
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

    # ---- 6. THE PRE-REGISTERED QUESTION: where is the boundary? ------------
    # Per axis, which checkpoints accept, and is the accept set a suffix of the
    # trajectory (i.e. "appears once and stays") or scattered (i.e. not a
    # convergence phenomenon)?
    boundary = {}
    for axis in DECISION_AXES:
        if axis not in curve["split"]:
            continue
        cs = curve["split"][axis]
        acc_steps = cs["accepting_steps"]
        per = cs["per_step"]
        is_suffix = bool(acc_steps
                         and acc_steps == traj_steps[len(traj_steps) - len(acc_steps):])
        first_acc = acc_steps[0] if acc_steps else None
        idx = traj_steps.index(first_acc) if first_acc is not None else None
        boundary[axis] = {
            "accepting_steps": acc_steps,
            "n_accepting": len(acc_steps),
            "accept_set_is_a_suffix_of_the_trajectory": is_suffix,
            "first_accepting_step": first_acc,
            "boundary_bracket": (
                [traj_steps[idx - 1], first_acc] if idx not in (None, 0) else
                ([None, first_acc] if idx == 0 else None)),
            "boundary_located": bool(is_suffix and idx not in (None, 0)),
            "best_margin_step": cs["argmax_margin_step"],
            "best_margin_pp": cs["best_margin_pp"],
            "endpoint_margin_pp": per[str(traj_steps[-1])]["margin_pp"],
            "endpoint_is_the_best_checkpoint": bool(
                cs["argmax_margin_step"] == traj_steps[-1]),
            "margin_monotone_nondecreasing": cs["monotonicity"][
                "monotone_nondecreasing"],
            "prereg_reading": None,   # filled below
        }
        if not acc_steps:
            boundary[axis]["prereg_reading"] = "no accept on this axis anywhere"
        elif is_suffix and idx not in (None, 0):
            boundary[axis]["prereg_reading"] = (
                "(a) accept appears once and persists to the endpoint -> the "
                f"boundary is LOCATED in ({traj_steps[idx-1]}, {first_acc}]")
        else:
            boundary[axis]["prereg_reading"] = (
                "(b) the accept set is NOT a suffix, or it already accepts at the "
                "FIRST checkpoint -> the endpoint accept is not the product of "
                "convergence; it is consistent with non-monotone CPT drift")

    # §2.0.2 neighbour precondition on every accepting cell, PER AXIS
    neighbour = {}
    for axis in AXES:
        if axis not in curve["split"]:
            continue
        for st in curve["split"][axis]["accepting_steps"]:
            neighbour[f"{axis}|step{st}"] = {
                "decision_axis": axis not in DEMOTED_AXES,
                **neighbour_precondition(curve["split"][axis], st, traj_steps)}
    if not neighbour:
        neighbour["none"] = ("no cell accepts on any axis, so §2.0.2 has nothing "
                             "to gate. The precondition is not vacuously "
                             "satisfied -- it is not triggered.")

    # ---- 7. RATIO vs NI disagreement, tracked along the trajectory ---------
    rho = PREREG["rho"]
    ratio_vs_ni = {"rho": rho, "per_step": {}}
    for step, _ in TRAJECTORY:
        arm = _arm_name(step)
        rr = per_conv["split"]["ratio_rule"][arm]
        dec = [c for c in per_conv["split"]["cells"]
               if c["arm"] == arm and c["decision_axis"]]
        acc = [c["axis"] for c in dec if c["ni_accept"]]
        need = int(np.ceil(0.50 * len(dec))) if dec else None
        ni_acc = bool(dec and acc and len(acc) >= need)
        ratio_vs_ni["per_step"][str(step)] = {
            "ratio_mean_ratio": rr["mean_ratio"],
            "ratio_margin_over_rho": rr["mean_ratio"] - rho,
            "ratio_accept": rr["ratio_accept"],
            "ni_n_axes_accepting": len(acc),
            "ni_axes_accepting": acc,
            "ni_accept_overall": ni_acc,
            "rules_disagree": bool(rr["ratio_accept"] != ni_acc),
            "per_axis_ratio": rr.get("per_axis_ratio"),
        }
    dis = [int(k) for k, v in ratio_vs_ni["per_step"].items()
           if v["rules_disagree"]]
    ratio_vs_ni["disagreeing_steps"] = sorted(dis)
    ratio_vs_ni["n_disagreeing"] = len(dis)
    ratio_vs_ni["disagreement_is_everywhere"] = (len(dis) == len(traj_steps))
    ratio_vs_ni["archived_endpoint_disagreement_reproduced"] = bool(
        25000 in dis)
    ratio_vs_ni["reading"] = (
        "RATIO(0.85) vs NI(Delta) disagreement tracked along the trajectory. At "
        "the endpoint full32_rescore_v2_20260812 established RATIO ACCEPTS "
        "(mean_ratio 0.8515, margin over rho +0.0015) while NI rejects on 2 of 3 "
        "decision axes. Whether that disagreement widens, narrows or vanishes "
        "earlier is the second pre-registered question.")

    # ---- 8. monotonicity / resolved non-monotonicity -----------------------
    dec_mono = {ax: curve["split"][ax]["monotonicity"]["monotone_nondecreasing"]
                for ax in DECISION_AXES if ax in curve["split"]}
    any_accept_anywhere = any(v["NI_OBSERVED_TO_ACCEPT"]
                              for v in verdict["split"].values())
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

    output_diag = {}
    for ax in ("triviaqa", "popqa", "nq_open"):
        r = output_shape_and_flips(data, ax, traj_steps)
        if r is not None:
            output_diag[ax] = r

    # EM-vs-contains: what an EM move actually consists of. Labelled, never scored.
    verbosity_diag = {}
    for ax in ("triviaqa", "popqa", "nq_open"):
        r = em_vs_contains_verbosity(data, ax, traj_steps)
        if r is not None:
            verbosity_diag[ax] = r

    # ---- headline ----------------------------------------------------------
    mm = boundary.get("mmlu_content", {})
    if not any_accept_anywhere:
        headline = "NO_CHECKPOINT_MEETS_THE_2OF3_BAR_ANYWHERE_ON_THE_TRAJECTORY"
    else:
        headline = "SOME_CHECKPOINT_MEETS_THE_2OF3_BAR"
    axis_level = ("PREREG_A_BOUNDARY_LOCATED" if mm.get("boundary_located")
                  else ("PREREG_B_ACCEPT_NOT_FROM_CONVERGENCE"
                        if mm.get("accepting_steps") else "NO_AXIS_ACCEPT"))
    headline = f"{headline}__{axis_level}"

    # ---- cross-scale, explicitly labelled ---------------------------------
    sens = {"note": (
        "sd_run is a 1B, S=3, keep12@5000 quantity. Every 7B rung has exactly "
        "ONE seed and the historical 7B ladder's seeds are UNRECORDED (--seed "
        "postdates them), so NO 7B sd_run exists or can be reconstructed. The "
        "column below is an explicitly-labelled cross-scale extrapolation and is "
        "NOT licensed as a 7B variance statement. The five trajectory points are "
        "also NOT seeds or replicates of each other -- they are five checkpoints "
        "of ONE run, so their spread is training progress plus data order."),
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

    # ---- staging provenance ----------------------------------------------
    staging = {"why": (
        "the four intermediate ckpts are wzc1-resident; zwfy6 had only "
        "step25000. They were moved 2026-08-13 by "
        "code/a04_full32_stage_parallel.sh."),
        "verification_per_ckpt": (
            "size == source size; FULL-FILE sha256 equal on BOTH disks (not a "
            "prefix hash -- the known cluster failure mode is a truncated write, "
            "which a prefix hash cannot see); zip entry count == 1435 == source. "
            "Plus a torch.load probe in the driver asserting meta "
            "step/keep_front/n_fresh/num_hidden_layers/len(model_state)==355 "
            "before 8 GPUs were spent."),
        "records": {}}
    for lg in (args.stage_log_73, args.stage_log_82):
        p = os.path.join(args.raw_root, lg)
        if not os.path.isfile(p):
            # the stage logs were written on the wzc1 SOURCE host, not on zwfy6
            staging["records"][lg] = "log not present under raw_root (written on the wzc1 source host)"
            continue
        for line in open(p):
            if line.startswith("STAGE_RESULT"):
                kv = dict(x.split("=", 1) for x in line.split()[1:])
                staging["records"][f"step{kv['step']}"] = kv
    staging["measured_transfer_rate"] = {
        "single_stream": "2 GiB in 118 s = 17.4 MiB/s (reproduces the 16.3 MiB/s "
                         "recorded in full32_trajectory_staging_remeasured_20260813)",
        "8_streams_to_ONE_node": "6 GiB in 47 s = 130.7 MiB/s aggregate",
        "8_streams_split_over_two_nodes": "8 GiB in 61 s = 134.3 MiB/s aggregate",
        "per_ckpt_realised": "81.57 GiB in 605-628 s = 137.4 / 138.1 MiB/s",
        "CORRECTION": (
            "16-17 MiB/s is a PER-STREAM ceiling, NOT a property of the link. "
            "full32_trajectory_staging_remeasured_20260813 priced the staging at "
            "5.7 h from the single-stream number and deferred the scan on that "
            "basis; the realised cost was ~31 min of wall clock for all four "
            "ckpts (two in parallel per node, 4 files total). The deferral was "
            "based on a correct measurement and an incorrect inference from it."),
    }

    # per-checkpoint `contains` alongside EM, for the same labelled purpose:
    # to show that the EM curve and the contains curve tell different stories on
    # triviaqa. EM remains the decision metric.
    contains_table = {}
    for ax in ("triviaqa", "popqa", "nq_open"):
        if data[_arm_name(traj_steps[0])].get(f"_{ax}_rows") is None:
            continue
        contains_table[ax] = {}
        for arm_label, arm in ([("intact_7B_base", "intact_7B_base")]
                               + [(f"step{s}", _arm_name(s)) for s in traj_steps]):
            rows = data[arm].get(f"_{ax}_rows")
            if rows is None:
                continue
            em = float(np.mean([1.0 if r.get("em") in (1, True) else 0.0
                                for r in rows]))
            co = float(np.mean([1.0 if r.get("contains") in (1, True) else 0.0
                                for r in rows]))
            contains_table[ax][arm_label] = {
                "em_pp": 100.0 * em, "contains_pp": 100.0 * co,
                "contains_minus_em_pp": 100.0 * (co - em)}

    out = {
        "gate": "A04_full32_dolmino_trajectory_NI_discrimination_curve_7B",
        "question": (
            "shallow_rung_ni_discrimination_20260812.implication_for_pilot_two"
            ".cheap_next_steps_dominate[1], full32 half: full32_dolmino@step25000 "
            "is the ONLY NI accept in A04 (mmlu_content +1.0495 pp, 97.7% "
            "recovery, triviaqa 1.86 SE short). Where on the trajectory does the "
            "accept appear?"),
        "date": "2026-08-13",
        "headline_verdict": headline,
        "prereg": {
            "document": "A04_FULL32_TRAJECTORY_PREREG.md",
            "committed_before_any_result": "commit 537d323, 2026-08-13; the "
            "step5000/step15000 GPU jobs started 11:36:51/11:40:36 and no "
            "summary.json existed for any of the four new points when it was "
            "written",
            "reading_a": "accept only at step25000 -> boundary located in (20000, 25000]",
            "reading_b": "accept ALSO earlier -> endpoint accept is not from "
                         "convergence; worse for A04, reported just as readily",
        },
        "ZERO_DAMAGE_CAVEAT": (
            "full32 = keep_front_layers 32 / n_fresh_layers 0: all 32 pretrained "
            "layers present, nothing transplanted, nothing pruned. It is a "
            "CONTINUED-PRETRAINING control, not a healed model. Every statement "
            "here is about CPT drift on the heal corpus and NOT about recovery "
            "from structural injury."),
        "protocol_asserted": proto,
        "staging_provenance": staging,
        "analysis_environment": {
            "hostname": socket.gethostname(),
            "node": "pinned to .73 -- see numpy_pinning_rationale",
            "numpy": np.__version__,
            "python": platform.python_version(),
            "numpy_pinning_rationale": (
                "neighbour_variability_20260813.reproducibility_defect_found: "
                "Generator.multinomial differs in 19/10000 rows between numpy "
                "2.5.1 (.73) and 2.4.6 (.82) at the same seed, moving margins by "
                "up to 0.0053 pp -- an ORDER OF MAGNITUDE larger than this "
                "script's 5e-4 pp archived-repro hard-fail threshold. So the same "
                "code passes on .73 and assert-fails on .82, and the failure "
                "looks like a logic bug. GPU scoring was split .73/.82 (it is "
                "deterministic); the bootstrap is NOT split."),
        },
        "intact_anchor": {
            "anchor": "vanilla ../models/OLMo-2-1124-7B: "
                      "olmo2_closedbook_results/base_full{,_nqopen} + "
                      "olmo2_mmlu_content_results/7B_base",
            "imported_not_redeclared": "a04_shallow_rung_ni_7b.ANCHOR",
            "delta_never_substituted": True,
            "anchor_never_changed": True,
            "guard_G2": (
                "full32_step25000 was NOT used as the anchor -- it is an ARM. It "
                "scores BELOW vanilla on all four axes, so substituting it would "
                "shrink every Delta AND lower every target = manufacturing "
                "accepts. G2 forbids it."),
        },
        "bootstrap_offsets": {
            "arm_index": arm_index,
            "form": "97*arm_index + 13*axis_index",
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index "
                                 "(imported guard_cell uses SEED+700; see note)",
            "interval_seed_offset": f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair",
            "endpoint_keeps_archived_offset": ENDPOINT_ARM_INDEX,
            "disjoint_from": (
                "pilot_zero {0,1}; step100k 100..102; shallow_rung 200..203; "
                "keep14 trajectory 300..301; neighbour 400..408. Interval offset "
                "2900 avoids the used 900/1900/2400."),
            "guard_offset_note": (
                "guard_cell is IMPORTED from a04_keep14_trajectory_ni and uses "
                "SEED+700+13*axis internally. That is deliberate: the guard is a "
                "property of the ANCHOR alone (D1-D4) and reproducing the "
                "archived guard cell bit-for-bit is what proves the anchor did "
                "not drift. It perturbs no arm cell, because arm NI cells use "
                "the 97*arm_index family. GUARD_SEED_OFF is recorded for the "
                "avoidance of doubt but the imported code path is the one used."),
        },
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "archived_endpoint_reproduction": repro,
        "nulls": {k: (v if not isinstance(v, dict) else
                      {kk: vv for kk, vv in v.items()
                       if kk not in ("vector", "vectors")})
                  for k, v in nulls.items()},
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "per_convention": per_conv,
        "discrimination_curve": curve,
        "verdict_by_convention": verdict,
        "accept_boundary": boundary,
        "neighbour_precondition_2_0_2": neighbour,
        "ratio_vs_ni_along_trajectory": ratio_vs_ni,
        "decision_axis_margin_monotone_nondecreasing_split": dec_mono,
        "non_monotonicity_resolved_split": resolved_nonmono,
        "non_monotonicity_detail_split": nonmono_detail,
        "output_shape_and_flips_diagnostic": output_diag,
        "em_vs_contains_verbosity_diagnostic": verbosity_diag,
        "em_and_contains_per_checkpoint": contains_table,
        "any_checkpoint_accepts": any_accept_anywhere,
        "sensitivity": sens,
        "NOT_licensed": [
            "ANY claim that the 7B deficits are large or small 'relative to seed "
            "variance'. sd_run is 1B-ONLY (S=3, keep12@5000); every 7B rung has "
            "exactly ONE seed and the historical seeds are unrecorded.",
            "treating the five checkpoints as replicates of each other -- they "
            "are five checkpoints of ONE run, so their spread is training "
            "progress plus data order, not independent-run variance.",
            "calling any difference here 'harness noise': "
            "full32_rescore_v2_20260812 established there is NO measured "
            "runtime-jitter floor on this harness (same-code re-runs are "
            "bit-identical). These are five DIFFERENT models so bit-identity "
            "does not apply -- but 'noise' is equally unavailable as an "
            "explanation.",
            "any K1/K2/K3 clause: they are defined over the pre-registered 1B arm "
            "set and a 7B trajectory cannot fire them.",
            "any statement about recovery from STRUCTURAL DAMAGE: full32 has "
            "none. This is a continued-pretraining control.",
        ],
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=1, default=float)
    print(f"[ok] wrote {args.out_json}")
    print(f"[headline] {headline}")
    for ax in DECISION_AXES:
        if ax not in curve["split"]:
            continue
        c = curve["split"][ax]
        print(f"  {ax:14s} margins " + " ".join(
            f"{s}:{c['per_step'][str(s)]['margin_pp']:+.4f}"
            f"{'A' if c['per_step'][str(s)]['ni_accept'] else 'R'}"
            for s in traj_steps))
    print("  RATIO " + " ".join(
        f"{s}:{ratio_vs_ni['per_step'][str(s)]['ratio_mean_ratio']:.4f}"
        f"{'A' if ratio_vs_ni['per_step'][str(s)]['ratio_accept'] else 'R'}"
        for s in traj_steps))


if __name__ == "__main__":
    main()
