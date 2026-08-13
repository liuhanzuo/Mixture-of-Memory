#!/usr/bin/env python3
"""A04 — does the LR hypothesis survive a WITHIN-ARM contrast? (keep8+fresh2)

PRE-REGISTRATION: `A04_WITHIN_ARM_LR_PREREG.md` (committed BEFORE this ran).

WHAT IS BEING TESTED
--------------------
`A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md` 5 generated, from n=3 ARMS:

    H_LR: "checkpoint-to-checkpoint margin scatter is governed by where you are
    on the LR schedule" -- keep10 (LR 1.24e-5, triviaqa 500-step range 1.2149
    pp) > keep8 (6.80e-6, 1.1202) > keep12 (3.25e-6, 0.1951); the three arms
    rank-order identically by LR and by range, and NOT by depth.

It recorded its own confound (in these runs "late" == "low LR") and proposed a
~3.5 GPU-h probe: an EARLY, HIGH-LR 500-step triple on keep12. MAIN verified
that probe cannot be run without new training -- keep12's earliest ckpt is
124000, keep8's only early ckpt is 45000 with no neighbour, keep10 starts at
83500. NO ARM HAS AN EARLY 500-STEP TRIPLE.

This script runs the substitute MAIN identified: `a04_neighbour_variability`
scored TWO keep8 clusters, and only cluster2 was ever reported.

    cluster1 = 124000/124500/125000   (EARLIER, HIGHER LR)
    cluster2 = 130000/130500/131000   (LATER,   LOWER  LR)  <- the headline

Same arm, same depth (keep_front=8, 10 layers, 113 tensors), same corpus, same
repair mode, same protocol, same harness. Only the schedule position differs.
So it is the within-arm control H_LR needs, and it costs ZERO GPU.

TWO THINGS THE PREREG FIXED BEFORE ANY NUMBER WAS READ, BOTH OF WHICH LIMIT
WHAT THIS CAN CONCLUDE
--------------------------------------------------------------------------
(1) THE LR CONTRAST IS TINY. Measured from the training logs (not copied from
    the keep12 verdict's table): cluster1 mean LR 7.6288e-6, cluster2 6.8579e-6
    => 1.1124x. H_LR was fitted on a 3.82x cross-arm LR spread against a 6.23x
    range spread. 1.11x is ~13% of that contrast, so this design has almost no
    power to detect a smooth monotone LR effect. A NULL HERE DOES NOT REFUTE
    H_LR, and that is stated in the prereg, not discovered afterwards.
(2) CLUSTER1 STRADDLES A RESUME SEAM. 124000/124500 were written by the .73
    process of 2026-08-08 which then died (TCPStore); 125000 was written by a
    DIFFERENT .82 process on 2026-08-12 resuming FROM step124500.pt. The trainer
    rebuilds the loader without intra-epoch fast-forward
    (train_olmo2_arch_probe2.py:1011-1019), so 124500->125000 saw a different
    data order. cluster1 is therefore NOT a clean 500-step neighbourhood and
    may never be promoted to one. Re-verified here from the logs directly, and
    the verdict string carries an INADMISSIBLE_SEAM modifier when it holds.

THE STATISTIC IS THE NI MARGIN RANGE, NOT THE ACCURACY RANGE
------------------------------------------------------------
`margin_pp = diff_lower95_one_sided_pp + delta_pp`. MAIN's dispatch note used
hand-computed ACCURACY ranges; those are a different statistic (keep10 triviaqa
is margin 1.2149 vs accuracy 1.2093) and are not what GATE_DESIGN 2.0.2/2.5 are
written on. Both are emitted; only the MARGIN range is decision-bearing.

NOISE GATE CONSTANT IS k-DEPENDENT AND IS RE-DERIVED, NOT ASSERTED
------------------------------------------------------------------
E[range of k iid N(0,s)]/s: k=3 -> 3/sqrt(pi) = 1.6925687506432689; k=2 ->
2/sqrt(pi) = 1.1283791670955126. BOTH clusters are k=3, so 1.6926 is correct for
both and no 5-/8-point constant enters. `selftest_gate_constants` re-derives k=2,
3, 5, 8 by Monte Carlo and checks the closed forms, because on 2026-08-13 an
8-point grid was gated with the k=3 constant, which moved a floor 40.6% and
flipped a boolean. EACH CLUSTER'S sigma IS ITS OWN -- one cluster's SEs may never
gate the other (enforced structurally: range_report is called once per cluster
with that cluster's own SE list).

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`build_nulls`, `ni_rule`, `ratio_rule`, `AXES`, `DEMOTED_AXES`, `EXPECTED_N`,
`PREREG` from `pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`,
`N_BOOT`, `SEED` from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`,
`assert_aligned` from `a04_shallow_rung_ni_7b`; and -- the point of this file --
`range_report`, `guard_cell`, `protocol_asserted`, `shard_integrity_report`,
`adjacent_interval_tests`, `output_shape_and_flips`, `EXPECTED_RANGE_OVER_SD`,
`LEG_A_CLUSTERS` from `a04_neighbour_variability`, i.e. THE SAME CODE OBJECTS
that produced the archived keep8 numbers. Delta and the anchor are never
substituted (G0/G2). The self-excluding `assert_seeds_disjoint` is taken from
`a04_keep12_trajectory_monotonicity` unweakened.

CPU ONLY. Read-only on every input. ZERO GPU.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
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

from a04_shallow_rung_ni_7b import (  # noqa: E402
    ANCHOR,
    _load_arm,
    assert_aligned,
)

# THE POINT OF THIS FILE: identical code objects to the keep8 archive.
from a04_neighbour_variability import (  # noqa: E402
    ARM_ARCH,
    EXPECTED_RANGE_OVER_SD,
    LEG_A_CLUSTERS,
    adjacent_interval_tests,
    guard_cell,
    output_shape_and_flips,
    protocol_asserted,
    range_report,
    shard_integrity_report,
)
# the FIXED, self-excluding seed check -- taken as-is, not weakened.
from a04_keep12_trajectory_monotonicity import (  # noqa: E402
    assert_seeds_disjoint,
)

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]
PRIMARY_AXIS = "triviaqa"

ARM_KEY = "keep8fresh2"
ARM_DIR = "olmo2_probe2_7B_keep8fresh2"

# The two clusters, taken FROM the archive's own definition object so a drift
# between this script and the archive is impossible by construction.
CLUSTER_HI_LR = "cluster1_124000_125000"   # EARLIER  => HIGHER LR
CLUSTER_LO_LR = "cluster2_130000_131000"   # LATER    => LOWER  LR

# ---------------------------------------------------------------------------
# The LR schedule, taken from the TRAINER, not from any document.
# scripts/train_olmo2_arch_probe2.py imports get_lr from
# scripts/train_semantic_bottleneck_1b.py (line 88-93):
#     def get_lr(step, warmup, max_steps, base_lr, min_lr):
#         if step < warmup: return base_lr*step/max(warmup,1)
#         prog = (step-warmup)/max(max_steps-warmup,1)
#         return min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*min(prog,1.0)))
# and applies it per param group each accum boundary (line 1051-1053), logging
# param_groups[0]["lr"] (the fresh group) as `lr=` (line 1064-1067).
# base_lr/min_lr confirmed from the run's own `[optim] group ...` banner:
#   fresh_decay/fresh_nodecay/inh_decay/inh_nodecay ALL base_lr=2.00e-05
#   min_lr=2.00e-06  => the differential-LR bug means ALL FOUR GROUPS ARE EQUAL,
#   so "the LR" is unambiguous for this arm.
# ---------------------------------------------------------------------------
LR_SCHEDULE = {
    "source": ("scripts/train_semantic_bottleneck_1b.py:get_lr, imported by "
               "scripts/train_olmo2_arch_probe2.py:88-93 and applied at "
               "1051-1053"),
    "form": "min_lr + 0.5*(base_lr-min_lr)*(1+cos(pi*min(prog,1)))",
    "base_lr": 2e-05,
    "min_lr": 2e-06,
    "warmup_steps": 150,
    "max_steps": 200000,
    "banner_evidence": (
        "logs/olmo2_7B_keep8fresh2_resume200k_82.log:12-15 -- [optim] group "
        "fresh_decay 815.8M base_lr=2.00e-05 min_lr=2.00e-06; fresh_nodecay "
        "0.0M same; inh_decay 2030.0M same; inh_nodecay 0.1M same. All four "
        "groups share base_lr/min_lr (the known no-op differential-LR bug), so "
        "'the LR' is unambiguous."),
    "resume_banner": (
        "logs/olmo2_7B_keep8fresh2_resume200k_82.log:18 -- [resume] continue @ "
        "step=124500 epoch=1 warmup=150 max_steps=200000 lr_fresh(now)=7.629e-06 "
        "lr_inh(now)=7.629e-06 -- confirms warmup/max_steps AND that the cosine "
        "resumes on-curve at the ckpt step (no schedule restart)."),
}

# LR as LOGGED by the trainer at each of the six steps. Parsed from the logs at
# runtime; these literals exist ONLY so a mismatch raises instead of passing.
LOGGED_LR_EXPECTED = {
    124000: 7.69e-06, 124500: 7.63e-06, 125000: 7.56e-06,
    130000: 6.92e-06, 130500: 6.86e-06, 131000: 6.80e-06,
}
TRAIN_LOGS = {
    ".73": "logs/olmo2_7B_keep8fresh2_resume200k_73.log",
    ".82": "logs/olmo2_7B_keep8fresh2_resume200k_82.log",
}

# H_LR as generated, for the reader. NOT used as an input to any statistic --
# every number below is recomputed. Cross-arm LRs are re-parsed from those arms'
# own logs by `crossarm_lr_readback`.
H_LR_AS_GENERATED = {
    "source": "A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md 5",
    "statement": ("checkpoint-to-checkpoint margin scatter is governed by where "
                  "you are on the LR schedule, not by damage depth"),
    "n_arms": 3,
    "table_as_published": {
        "keep10": {"triple": [89000, 89500, 90000], "lr_at_triple": 1.24e-05,
                   "triviaqa_range_pp": 1.2149, "keep_front": 10},
        "keep8": {"triple": [130000, 130500, 131000], "lr_at_triple": 6.80e-06,
                  "triviaqa_range_pp": 1.1202, "keep_front": 8},
        "keep12": {"triple": [165000, 165500, 166000], "lr_at_triple": 3.25e-06,
                   "triviaqa_range_pp": 0.1951, "keep_front": 12},
    },
    "confound_the_verdict_itself_recorded": (
        "in these runs 'late' is the same thing as 'low LR'; three arms cannot "
        "separate LR from step count, epoch position or depth"),
    "probe_it_proposed": (
        "score a 500-step triple EARLY in keep12's own schedule (high LR, same "
        "arm, same depth, same corpus); ~3.5 GPU-h"),
    "why_that_probe_is_unrunnable": (
        "verified 2026-08-13 by ls on zwfy6: keep12's ckpts are "
        "124000/130000/135000/140000/145000/150000/155000/160000/165000/165500/"
        "166000 -- earliest is 124000; keep8's are 45000/121000/124000/124500/"
        "125000/130000/130500/131000 -- 45000 has no neighbour; keep10's are "
        "83500/85000/86500/89000/89500/90000. NO ARM HAS AN EARLY 500-STEP "
        "TRIPLE. The probe requires new TRAINING, not new eval."),
}

# Verdict criteria, frozen by the prereg commit. The string is emitted
# MECHANICALLY from these; it is not chosen after seeing the table.
VERDICT_CRITERIA = {
    "prereg_document": "A04_WITHIN_ARM_LR_PREREG.md",
    "primary_axis": PRIMARY_AXIS,
    "convention": "split",
    "R_definition": "range(cluster1_HIGHER_LR) / range(cluster2_LOWER_LR)",
    "direction_convention": (
        "cluster1 is the EARLIER/HIGHER-LR cluster (mean 7.6288e-6); cluster2 is "
        "the LATER/LOWER-LR cluster (mean 6.8579e-6). H_LR says higher LR => "
        "LARGER range, so H_LR predicts R > 1."),
    "REFUTED_WITHIN_ARM": "both clusters clear their OWN noise gate AND R <= 0.83",
    "SUPPORTED_WITHIN_ARM": "both clusters clear their OWN noise gate AND R >= 1.20",
    "UNRESOLVED_SUBNOISE": (
        ">=1 of the two clusters FAILS its own noise gate => the ratio of two "
        "ranges is undefined; NEITHER support NOR refutation"),
    "UNRESOLVED_UNDERPOWERED": "both clear but 0.83 < R < 1.20",
    "INADMISSIBLE_SEAM": (
        "cluster1's resume seam confirmed => a MODIFIER appended to whichever "
        "label fires; the result may not be reported as a clean within-arm LR "
        "contrast"),
    "why_20_percent": (
        "+-20% is LOOSER than the 8% agreement keep8-vs-keep10 already showed on "
        "this statistic (1.2149/1.1202 = 1.0846), so ordinary replication noise "
        "cannot manufacture a direction"),
    "banned_rewordings": (
        "if UNRESOLVED_SUBNOISE fires it must be reported as 'no detectable "
        "difference -- neither supports nor refutes H_LR'. It may NOT be "
        "re-described as 'consistent with noise, so H_LR is fine' NOR as 'the "
        "direction is reversed, so H_LR is dead'."),
}

# MAIN's hand-computed ACCURACY ranges, recorded so the report can state
# explicitly whether they reproduce. They are NOT inputs to anything.
MAIN_HAND_ACC_RANGES_PP = {
    "triviaqa": {"c1": 0.2786, "c2": 1.1090},
    "popqa": {"c1": 0.1192, "c2": 0.2453},
    "mmlu_content": {"c1": 0.1852, "c2": 0.2065},
    "nq_open": {"c1": 0.1939, "c2": 0.3324},
}

# keep8 archive literals -- read at runtime, used ONLY as assertions.
ARCHIVE_BASENAME = "a04_neighbour_variability.json"
ARCHIVE_EXPECTED_MARGIN_RANGES_PP = {
    CLUSTER_LO_LR: {"triviaqa": 1.1202, "popqa": 0.2523,
                    "mmlu_content": 0.2208, "nq_open": 0.3324},
    CLUSTER_HI_LR: {"triviaqa": 0.2675, "popqa": 0.1192,
                    "mmlu_content": 0.2136, "nq_open": 0.2216},
}
ARCHIVE_EXPECTED_GATE = {
    CLUSTER_LO_LR: {"triviaqa": True, "popqa": False,
                    "mmlu_content": False, "nq_open": False},
    CLUSTER_HI_LR: {"triviaqa": False, "popqa": False,
                    "mmlu_content": False, "nq_open": False},
}

NEW_ARM_INDEX_BASE = 1000   # disjoint from 0,1/100-102/200-203/300-301/400-408/
                            # 500-503/600-610/700-702/800-801/900-902
GUARD_SEED_OFF = 6700       # disjoint from 700,1700,2700,3700,4700,5700,8700
INTERVAL_SEED_OFF = 6900    # disjoint from 900,1900,2400,2900,3900,4600,4900,
                            # 7000,7100
PUBLISH_NUMPY = "2.4.6"     # the node/version that published the keep8 archive


def _arm_name(step):
    return f"{ARM_KEY}_step{step}"


def _seed_off(arm_index, axis):
    """Same functional form as every archived cell: 97*arm_index + 13*axis."""
    return 97 * arm_index + 13 * AXES.index(axis)


def _tag_dirs(tag_prefix, step):
    return {"mmlu": f"{tag_prefix}_step{step}",
            "cb": f"{tag_prefix}_step{step}",
            "nq": f"{tag_prefix}_step{step}_nqopen"}


# ---------------------------------------------------------------------------
def cosine_lr(step):
    """The trainer's own schedule, re-implemented ONLY to cross-check the LOGGED
    values. The logs are the evidence; this is the corroboration. If the two
    disagree beyond the logs' 3-sig-fig printing, the run aborts.
    """
    s = LR_SCHEDULE
    if step < s["warmup_steps"]:
        return s["base_lr"] * step / max(s["warmup_steps"], 1)
    prog = ((step - s["warmup_steps"])
            / max(s["max_steps"] - s["warmup_steps"], 1))
    return s["min_lr"] + 0.5 * (s["base_lr"] - s["min_lr"]) * (
        1 + math.cos(math.pi * min(prog, 1.0)))


def lr_from_logs(raw_root, steps):
    """MEASURE the LR at each step from the trainer's own log lines. FAIL CLOSED.

    The dispatch forbids copying the keep12 verdict's LR table, and rightly:
    that table quotes keep8 at 6.80e-6, which is the LR at step131000 only --
    the OTHER two members of that cluster sit at 6.92e-6 and 6.86e-6, and the
    cluster MEAN is what a cluster-level contrast needs.

    Both training logs are scanned; a step found in BOTH with DIFFERENT lr would
    be fatal (it cannot happen on one cosine curve, and if it did it would mean
    the schedule was restarted).
    """
    found = {}
    per_log = {}
    for node, lg in TRAIN_LOGS.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            raise SystemExit(
                f"FATAL: training log {p} absent -- the LR cannot be MEASURED, "
                "and the dispatch forbids copying it from a document.")
        txt = open(p).read()
        hits = {}
        for st in steps:
            m = re.search(rf"\[step\s*{st}/{LR_SCHEDULE['max_steps']}\]"
                          r".*?lr=([0-9.eE+-]+)", txt)
            if m:
                hits[st] = float(m.group(1))
        per_log[node] = {"log": lg, "steps_found": sorted(hits),
                         "lr_by_step": {str(k): v for k, v in sorted(hits.items())}}
        for st, v in hits.items():
            if st in found and abs(found[st][0] - v) > 1e-12:
                raise SystemExit(
                    f"FATAL: step {st} logs lr={found[st][0]} in "
                    f"{found[st][1]} but lr={v} in {lg} -- two different LRs "
                    "for one step means the schedule was restarted.")
            found.setdefault(st, (v, lg))
    missing = [s for s in steps if s not in found]
    if missing:
        raise SystemExit(
            f"FATAL: no logged lr= line for steps {missing}; refusing to "
            "publish an LR contrast with imputed values.")
    out = {"per_log": per_log, "per_step": {}}
    for st in steps:
        logged, src = found[st]
        recomp = cosine_lr(st)
        # the log prints 3 significant figures, so agreement is checked at that
        # resolution and the tolerance is derived, not guessed.
        tol = 10.0 ** (math.floor(math.log10(abs(logged))) - 2) * 0.5001
        ok = abs(logged - recomp) <= tol
        if not ok:
            raise SystemExit(
                f"FATAL: step {st} logged lr={logged:.3e} but the trainer's own "
                f"cosine gives {recomp:.6e} (tol {tol:.2e}) -- the schedule in "
                "LR_SCHEDULE does not describe this run.")
        exp = LOGGED_LR_EXPECTED.get(st)
        if exp is not None and abs(logged - exp) > 1e-12:
            raise SystemExit(
                f"FATAL: step {st} logs lr={logged} but this script's recorded "
                f"literal is {exp} -- the log moved; resolve before publishing.")
        out["per_step"][str(st)] = {
            "lr_logged": logged,
            "lr_recomputed_from_trainer_cosine": recomp,
            "agrees_at_log_precision": bool(ok),
            "tolerance_used": tol,
            "log_source": src,
        }
    out["measured_not_copied"] = (
        "every value above is parsed from a `[step N/200000] ... lr=` line "
        "written by the trainer itself, then cross-checked against the "
        "trainer's own cosine. NOTHING is copied from "
        "A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md's table.")
    return out


def cluster_lr_summary(lr_measured, clusters):
    """Cluster-level LR and the contrast between the two clusters."""
    out = {}
    for cname, steps in clusters.items():
        vals = [lr_measured["per_step"][str(s)]["lr_logged"] for s in steps]
        rec = [lr_measured["per_step"][str(s)]["lr_recomputed_from_trainer_cosine"]
               for s in steps]
        out[cname] = {
            "steps": list(steps),
            "lr_logged_per_step": vals,
            "lr_recomputed_per_step": rec,
            "lr_mean_logged": float(np.mean(vals)),
            "lr_mean_recomputed": float(np.mean(rec)),
            "lr_at_last_step": vals[-1],
        }
    hi, lo = out[CLUSTER_HI_LR], out[CLUSTER_LO_LR]
    contrast = {
        "hi_lr_cluster": CLUSTER_HI_LR,
        "lo_lr_cluster": CLUSTER_LO_LR,
        "lr_ratio_mean": hi["lr_mean_logged"] / lo["lr_mean_logged"],
        "lr_ratio_recomputed": hi["lr_mean_recomputed"] / lo["lr_mean_recomputed"],
        "lr_ratio_last_step": hi["lr_at_last_step"] / lo["lr_at_last_step"],
        "lr_ratio_max_over_min_all_six": (
            max(hi["lr_logged_per_step"] + lo["lr_logged_per_step"])
            / min(hi["lr_logged_per_step"] + lo["lr_logged_per_step"])),
        "ordering_is_hi_gt_lo": bool(hi["lr_mean_logged"] > lo["lr_mean_logged"]),
    }
    out["contrast"] = contrast
    return out


def crossarm_lr_readback(raw_root):
    """Re-measure the OTHER arms' LRs from THEIR logs, so the power statement is
    computed rather than quoted. Non-fatal if a log is absent (these arms are
    context, not inputs), but the absence is recorded.
    """
    specs = {
        "keep10fresh2": ("logs/olmo2_7B_keep10fresh2_resume200k_73.log",
                         [89000, 89500, 90000]),
        "keep12fresh2": ("logs/olmo2_7B_keep12fresh2_resume200k_v2.log",
                         [165000, 165500, 166000]),
    }
    out = {}
    for arm, (lg, steps) in specs.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            out[arm] = {"log": lg, "present": False,
                        "note": "log absent on this disk; not used in any statistic"}
            continue
        txt = open(p).read()
        got = {}
        for st in steps:
            m = re.search(rf"\[step\s*{st}/200000\].*?lr=([0-9.eE+-]+)", txt)
            if m:
                got[st] = float(m.group(1))
        out[arm] = {
            "log": lg, "present": True, "steps": steps,
            "lr_by_step": {str(k): v for k, v in sorted(got.items())},
            "lr_mean": (float(np.mean(list(got.values()))) if got else None),
            "lr_at_last_step": got.get(steps[-1]),
        }
    return out


# ---------------------------------------------------------------------------
def seam_verified_from_logs(raw_root, clusters):
    """RE-VERIFY the seam from the training logs, not from the archive's flag.

    The archive says cluster1 has resume_seam=true. That is a prose field in a
    JSON; this reads the logs and reconstructs which process wrote each save.
    A cluster is CLEAN iff every one of its saves appears in ONE log that
    contains no `[resume]` banner between the first and last of those saves.
    """
    logs = {}
    for node, lg in TRAIN_LOGS.items():
        p = os.path.join(raw_root, lg)
        if not os.path.isfile(p):
            raise SystemExit(f"FATAL: training log {p} absent -- the seam "
                             "precondition cannot be verified.")
        lines = open(p).read().split("\n")
        saves, resumes = {}, []
        for i, ln in enumerate(lines):
            m = re.search(r"saved .*?/step(\d+)\.pt\s*$", ln)
            if m:
                saves[int(m.group(1))] = i + 1          # 1-based line no.
            if "[resume] loading ckpt" in ln:
                resumes.append({"line": i + 1, "text": ln.strip()[-160:]})
        logs[node] = {"log": lg, "n_lines": len(lines),
                      "save_steps": sorted(saves), "save_line_by_step": saves,
                      "resume_banners": resumes,
                      "n_resume_banners": len(resumes)}

    out = {"per_log": {k: {kk: vv for kk, vv in v.items()
                           if kk != "save_line_by_step"}
                       for k, v in logs.items()}, "per_cluster": {}}
    for cname, steps in clusters.items():
        writer = {}
        for st in steps:
            owners = [n for n, v in logs.items() if st in v["save_line_by_step"]]
            if len(owners) != 1:
                raise SystemExit(
                    f"FATAL: step {st} is saved by {owners} logs -- cannot "
                    "attribute it to one process.")
            writer[st] = owners[0]
        one_process = len(set(writer.values())) == 1
        seam_inside = None
        if one_process:
            node = writer[steps[0]]
            v = logs[node]
            lo = v["save_line_by_step"][steps[0]]
            hi = v["save_line_by_step"][steps[-1]]
            inner = [r for r in v["resume_banners"] if lo < r["line"] < hi]
            seam_inside = len(inner) > 0
        clean = bool(one_process and seam_inside is False)
        out["per_cluster"][cname] = {
            "steps": list(steps),
            "writer_node_by_step": {str(k): v for k, v in writer.items()},
            "all_saves_in_one_process_log": one_process,
            "resume_banner_between_first_and_last_save": seam_inside,
            "is_clean_neighbour_measurement": clean,
            "resume_seam": bool(not clean),
            "archive_agrees": None,       # filled by the caller
        }
    out["method"] = (
        "a cluster is CLEAN iff all its `saved .../stepN.pt` lines occur in ONE "
        "training log AND that log contains no `[resume] loading ckpt` banner "
        "between the first and last of them. Attribution is by which log "
        "contains the save line, not by timestamp.")
    out["why_a_seam_disqualifies"] = (
        "train_olmo2_arch_probe2.py:1011-1019 does `sampler.set_epoch(epoch); "
        "data_iter = iter(loader)` on resume with NO intra-epoch fast-forward, "
        "so a seam-crossing 500-step interval saw a DIFFERENT data order than "
        "an uninterrupted one. Optimizer state and RNG are restored; the loader "
        "position is not.")
    return out


# ---------------------------------------------------------------------------
def selftest_gate_constants():
    """RE-DERIVE the noise-gate constants instead of trusting the dict.

    The whole gate is `E[range of k iid N(0,1)] * mean_SE`. On 2026-08-13 an
    8-point grid was gated with the k=3 constant, which moved a floor by 40.6%
    and flipped a boolean. So: closed forms for k=2,3 (2/sqrt(pi), 3/sqrt(pi),
    exact for the normal), a Monte-Carlo check for k=2,3,5,8, and a check that
    `range_report` actually USES the k that matches its input length. A failure
    aborts before anything is published.
    """
    out = {"closed_forms": {}, "monte_carlo": {}, "range_report_uses_k": {}}
    cf = {2: 2.0 / math.sqrt(math.pi), 3: 3.0 / math.sqrt(math.pi)}
    for k, v in cf.items():
        got = EXPECTED_RANGE_OVER_SD.get(k)
        ok = got is not None and abs(got - v) < 1e-12
        out["closed_forms"][str(k)] = {"closed_form": v, "in_table": got,
                                       "ok": bool(ok)}
        if not ok:
            raise SystemExit(f"FATAL: EXPECTED_RANGE_OVER_SD[{k}]={got} != {v}")

    rng = np.random.default_rng(20260813)
    for k, ref in ((2, cf[2]), (3, cf[3]), (5, 2.325929), (8, None)):
        x = rng.standard_normal((600000, k))
        mc = float((x.max(1) - x.min(1)).mean())
        rec = {"monte_carlo_600k": mc, "reference": ref}
        if ref is not None:
            rec["abs_err"] = abs(mc - ref)
            rec["ok"] = bool(abs(mc - ref) < 0.01)
            if not rec["ok"]:
                raise SystemExit(f"FATAL: MC E[range of {k}] = {mc} != {ref}")
        out["monte_carlo"][str(k)] = rec
    out["k5_and_k8_note"] = (
        "k=5 (2.325929) is the constant the full32 5-point scan uses and k=8 "
        "(MC ~2.847) would be the constant an 8-point grid needs. Both are "
        "re-derived here ONLY to show that 1.6926 is specific to k=3. NEITHER "
        "enters this analysis: both clusters here are k=3.")

    # 1. floor == constant * mean SE, and the gate is FALSE when range < floor
    r = range_report([0.0, 0.5, 1.0], [1.0, 1.0, 1.0], "selftest_k3_below")
    ok1 = (abs(r["range_pp"] - 1.0) < 1e-12
           and abs(r["expected_range_if_pure_noise_pp"]
                   - EXPECTED_RANGE_OVER_SD[3]) < 1e-12
           and r["range_exceeds_item_noise"] is False)
    # 2. and TRUE when range > floor
    r2 = range_report([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "selftest_k3_above")
    ok2 = r2["range_exceeds_item_noise"] is True
    # 3. k is taken from the INPUT LENGTH: a 2-point call must use 1.1284
    r3 = range_report([0.0, 1.0], [1.0, 1.0], "selftest_k2")
    ok3 = abs(r3["expected_range_if_pure_noise_pp"]
              - EXPECTED_RANGE_OVER_SD[2]) < 1e-12
    # 4. THE MISTAKE THIS GUARDS: gating a k=3 range with the k=2 constant would
    #    FLIP the boolean for a range of 1.5 against mean SE 1.0.
    flip = {"range": 1.5, "mean_se": 1.0,
            "floor_k3": EXPECTED_RANGE_OVER_SD[3],
            "floor_k2": EXPECTED_RANGE_OVER_SD[2],
            "gate_with_correct_k3": bool(1.5 > EXPECTED_RANGE_OVER_SD[3]),
            "gate_with_wrong_k2": bool(1.5 > EXPECTED_RANGE_OVER_SD[2]),
            "the_boolean_flips": bool((1.5 > EXPECTED_RANGE_OVER_SD[3])
                                      != (1.5 > EXPECTED_RANGE_OVER_SD[2]))}
    out["range_report_uses_k"] = {
        "k3_below_floor_gate_false": {"range": r["range_pp"],
                                      "floor": r["expected_range_if_pure_noise_pp"],
                                      "gate": r["range_exceeds_item_noise"],
                                      "ok": bool(ok1)},
        "k3_above_floor_gate_true": {"range": r2["range_pp"],
                                     "floor": r2["expected_range_if_pure_noise_pp"],
                                     "gate": r2["range_exceeds_item_noise"],
                                     "ok": bool(ok2)},
        "k2_call_uses_k2_constant": {
            "floor": r3["expected_range_if_pure_noise_pp"], "ok": bool(ok3)},
        "wrong_k_would_flip_a_boolean": flip,
    }
    out["all_ok"] = bool(ok1 and ok2 and ok3)
    if not out["all_ok"]:
        raise SystemExit(f"FATAL: gate-constant self-test failed: {out}")
    return out


def selftest_sigma_is_per_cluster():
    """Prove structurally that one cluster's sigma cannot gate the other.

    Fed two margin triples with the SAME range but very different SEs, the two
    floors must differ and the two booleans may differ. If a single pooled sigma
    were used, both would be identical -- which is exactly the error the
    dispatch names.
    """
    a = range_report([0.0, 0.6, 1.2], [0.30, 0.30, 0.30], "sigmaA")   # small SE
    b = range_report([0.0, 0.6, 1.2], [1.20, 1.20, 1.20], "sigmaB")   # big SE
    ok = (abs(a["range_pp"] - b["range_pp"]) < 1e-12
          and a["expected_range_if_pure_noise_pp"]
          != b["expected_range_if_pure_noise_pp"]
          and a["range_exceeds_item_noise"] is True
          and b["range_exceeds_item_noise"] is False)
    out = {"same_range_pp": a["range_pp"],
           "floor_small_se": a["expected_range_if_pure_noise_pp"],
           "floor_big_se": b["expected_range_if_pure_noise_pp"],
           "gate_small_se": a["range_exceeds_item_noise"],
           "gate_big_se": b["range_exceeds_item_noise"],
           "ok": bool(ok),
           "note": ("identical ranges, different per-cluster SEs => different "
                    "floors and different booleans. Each cluster is gated by "
                    "its OWN sigma; range_report is called once per cluster "
                    "with that cluster's own SE list, so pooling is impossible "
                    "by construction.")}
    if not ok:
        raise SystemExit(f"FATAL: per-cluster sigma self-test failed: {out}")
    return out


def archive_readback(archive_path):
    """Read the keep8 archive's OWN margin ranges for BOTH clusters and assert.

    This is the anti-drift check: if the archive has moved, a "recomputation"
    that silently disagrees with it would be indistinguishable from a bug.
    """
    if not os.path.isfile(archive_path):
        raise SystemExit(
            f"FATAL: archive {archive_path} absent -- there is nothing to "
            "recompute against.")
    blob = json.load(open(archive_path))
    la = blob["leg_A_neighbour_variability"]
    out = {"archive": os.path.basename(archive_path),
           "leg_A_clean_cluster_declared": blob.get("leg_A_clean_cluster"),
           "numpy_that_published_it": (blob.get("bootstrap_cross_node_drift", {})
                                       .get("published_on_node")),
           "per_cluster": {}}
    for cname in (CLUSTER_HI_LR, CLUSTER_LO_LR):
        cl = la[cname]
        rec = {"steps": cl["steps"], "spacing_steps": cl["spacing_steps"],
               "resume_seam_flag_in_archive": cl["resume_seam"],
               "is_clean_flag_in_archive": cl["is_clean_neighbour_measurement"],
               "per_axis": {}}
        for ax, v in cl["per_axis"].items():
            mr, ar = v["margin_range"], v["accuracy_range_pp"]
            rec["per_axis"][ax] = {
                "margins_pp": mr["margins_pp"],
                "margin_range_pp": mr["range_pp"],
                "margin_floor_pp": mr["expected_range_if_pure_noise_pp"],
                "margin_gate": mr["range_exceeds_item_noise"],
                "mean_bootstrap_se_pp": mr["mean_bootstrap_se_pp"],
                "accuracy_range_pp": ar["range_pp"],
                "acc_by_step": {s: d["acc"] for s, d in v["per_step"].items()},
            }
            exp = ARCHIVE_EXPECTED_MARGIN_RANGES_PP[cname].get(ax)
            if exp is not None and abs(mr["range_pp"] - exp) > 5e-4:
                raise SystemExit(
                    f"FATAL: archive {cname}/{ax} margin range "
                    f"{mr['range_pp']:.4f} != recorded {exp}")
            eg = ARCHIVE_EXPECTED_GATE[cname].get(ax)
            if eg is not None and bool(mr["range_exceeds_item_noise"]) != eg:
                raise SystemExit(
                    f"FATAL: archive {cname}/{ax} gate "
                    f"{mr['range_exceeds_item_noise']} != recorded {eg}")
        out["per_cluster"][cname] = rec
    if la[CLUSTER_LO_LR]["resume_seam"] is not False:
        raise SystemExit(f"FATAL: {CLUSTER_LO_LR} is not the seam-free cluster")
    out["literals_checked_against_archive"] = True
    out["note"] = (
        "BOTH clusters' per-axis numbers are already in this archive -- cluster1 "
        "was computed by the same run and simply never promoted past 'reported, "
        "not headline'. So the within-arm contrast needs no new scoring, and the "
        "recomputation below must REPRODUCE these values.")
    return out


def seed_mechanism_control(data, nulls, guard, archive_arm_index, clusters):
    """PROVE that the drift vs the archive is the MANDATED SEED CHANGE and
    nothing else -- not numpy, not code, not the data.

    The recomputation below uses arm_index 1000-1005, because reusing the
    archive's 400-405 is exactly what `assert_seeds_disjoint` forbids. But
    `ni_rule`'s bootstrap seed is `SEED + 97*arm_index + 13*axis`, so a different
    arm_index gives a different bootstrap draw and therefore a different 5th
    percentile. That is a real, unavoidable consequence of the disjointness rule,
    and it must be DEMONSTRATED rather than asserted.

    CONTROL: recompute the same cells with the ARCHIVE's OWN arm_index. If those
    reproduce the archive to 5e-4 pp, then the code, the data, the null, Delta
    and numpy are all identical and the ONLY difference is the seed. This is
    read-only: it recomputes numbers that already exist and writes nothing that
    anything else consumes.
    """
    out = {"purpose": ("isolate the cause of the margin drift vs the archive by "
                       "re-running the SAME cells with the ARCHIVE's arm_index"),
           "archive_arm_index": archive_arm_index, "per_cell": {},
           "n_compared": 0, "n_reproduced_5e_4": 0, "max_abs_drift_pp": 0.0}
    for cname, steps in clusters.items():
        for axis in AXES:
            g = guard["split"][axis]
            if g["classification"] == "NOT_CERTIFIABLE":
                continue
            for st in steps:
                nm = _arm_name(st)
                ai = archive_arm_index.get(nm)
                if ai is None:
                    continue
                r = ni_rule(data[nm][axis], data["intact_7B_base"][axis],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=_seed_off(ai, axis))
                out["per_cell"][f"{cname}|{axis}|{st}"] = {
                    "arm_index_used": ai,
                    "margin_pp_with_archive_seed":
                        r["diff_lower95_one_sided_pp"] + r["delta_pp"],
                    "lo95_pp_with_archive_seed": r["diff_lower95_one_sided_pp"],
                    "boot_seed": r["boot_seed"],
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--extra_evidence_dirs", default="",
                    help="comma-separated ADDITIONAL evidence dirs to scan for "
                         "seed collisions. REQUIRED IN PRACTICE: zwfy6's "
                         "evidence/ is missing 12 archives that wzc1 has "
                         "(including a04_sigma_run_postfix.json, which holds "
                         "arm_index 900-902), so a single-disk scan is a WEAKER "
                         "check than it looks. Every dir given is scanned and "
                         "any clash raises.")
    ap.add_argument("--archive", required=True)
    ap.add_argument("--tag_prefix", default="A04_7B_keep8f2")
    ap.add_argument("--driver_log", default="logs/a04_nbr_keep8_legA.out",
                    help="comma-separated; EVERY entry is protocol-gated")
    ap.add_argument("--node_label", required=True)
    ap.add_argument("--allow_other_numpy", action="store_true")
    args = ap.parse_args()

    if np.__version__ != PUBLISH_NUMPY and not args.allow_other_numpy:
        raise SystemExit(
            f"FATAL: numpy {np.__version__} != {PUBLISH_NUMPY}. The keep8 "
            "archive this recomputation must reproduce was published on numpy "
            f"{PUBLISH_NUMPY} (.82), and Generator.multinomial differs in 19 of "
            "10000 rows between 2.4.6 and 2.5.1. Run on .82, or pass "
            "--allow_other_numpy and accept ~0.006 pp drift.")

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    clusters = {CLUSTER_HI_LR: LEG_A_CLUSTERS[CLUSTER_HI_LR]["steps"],
                CLUSTER_LO_LR: LEG_A_CLUSTERS[CLUSTER_LO_LR]["steps"]}
    all_steps = sorted(set(clusters[CLUSTER_HI_LR] + clusters[CLUSTER_LO_LR]))

    # ---- 0. self-tests BEFORE anything is read -------------------------
    st_gates = selftest_gate_constants()
    st_sigma = selftest_sigma_is_per_cluster()

    # ---- 1. archive read-back, then protocol, then shards --------------
    arch = archive_readback(args.archive)

    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    proto_specs = {}
    for st in all_steps:
        spec = _tag_dirs(args.tag_prefix, st)
        arm_specs[_arm_name(st)] = spec
        proto_specs[f"legA|{st}"] = spec
    arm_names = [a for a in arm_specs if a != "intact_7B_base"]

    driver_logs = {}
    for i, lg in enumerate(
            [x.strip() for x in args.driver_log.split(",") if x.strip()]):
        driver_logs[f"legA_keep8_clusters_{i}" if i else
                    "legA_keep8_clusters"] = lg
    proto = protocol_asserted(
        args.raw_root, driver_logs,
        "proposal/active/A04-recovery-certification/code/"
        "a04_neighbour_variability_driver.sh",
        proto_specs)

    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    # ---- 2. seam precondition, from the LOGS ---------------------------
    seam = seam_verified_from_logs(args.raw_root, clusters)
    for cname in clusters:
        seam["per_cluster"][cname]["archive_agrees"] = bool(
            seam["per_cluster"][cname]["resume_seam"]
            == arch["per_cluster"][cname]["resume_seam_flag_in_archive"])
        if not seam["per_cluster"][cname]["archive_agrees"]:
            raise SystemExit(
                f"FATAL: seam verdict for {cname} disagrees with the archive's "
                "own flag -- resolve before publishing.")

    # ---- 3. the LR, MEASURED ------------------------------------------
    lr_measured = lr_from_logs(args.raw_root, all_steps)
    lr_clusters = cluster_lr_summary(lr_measured, clusters)
    crossarm_lr = crossarm_lr_readback(args.raw_root)

    # ---- 4. seeds --------------------------------------------------
    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}
    offsets_decl = {
        "arm_index_base": NEW_ARM_INDEX_BASE,
        "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
        "interval_seed_offset": f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair",
        "ni_seed_offset": "97*arm_index + 13*axis_index",
    }
    # SCAN EVERY DISK'S evidence/, not just this node's. zwfy6's copy is missing
    # 12 archives that wzc1 has, so a one-disk scan would silently pass on a
    # collision with an archive that only exists on the other disk.
    seed_dirs = [args.evidence_dir] + [
        d.strip() for d in args.extra_evidence_dirs.split(",") if d.strip()]
    seeds = {"dirs_scanned": [], "per_dir": {}}
    for d in seed_dirs:
        if not os.path.isdir(d):
            raise SystemExit(
                f"FATAL: evidence dir {d} absent -- refusing to publish a "
                "seed-disjointness claim that skipped a disk.")
        seeds["per_dir"][d] = assert_seeds_disjoint(
            d, list(arm_index.values()), offsets_decl,
            self_output_basename=os.path.basename(args.out_json))
        seeds["dirs_scanned"].append(d)
    seeds["n_dirs_scanned"] = len(seed_dirs)
    seeds["why_multi_disk"] = (
        "zwfy6's proposal/.../evidence/ is missing 12 json archives that wzc1 "
        "has (a04_sigma_run_postfix.json holds arm_index 900-902, "
        "a04_step100k_plateau_vs_ni.json holds 100-102, etc.). Scanning only "
        "the compute node's disk is a WEAKER check than it appears, so BOTH "
        "disks' evidence dirs are scanned and any clash raises.")

    # ---- 5. load, nulls, guard, NI ------------------------------------
    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    guard, per_conv = {}, {}
    for conv in TIE_CONVS:
        # NOTE: the imported guard_cell hardcodes its own GUARD_SEED_OFF
        # (SEED+1700) from a04_neighbour_variability. That is DELIBERATE: it is
        # what makes the intact residual / Delta reproduce the archive exactly.
        # Delta is a property of the ANCHOR, not of this arm set, so reusing the
        # archive's guard seed cannot perturb anything of the archive's -- it
        # recomputes the SAME quantity. Our own GUARD_SEED_OFF is recorded for
        # the ledger and is unused for that reason; this is stated rather than
        # silently ignored.
        guard[conv] = {axis: guard_cell(data, arm_names, nulls, conv, axis)
                       for axis in AXES}
        cells, retired = [], []
        for arm in arm_names:
            for axis in AXES:
                g = guard[conv][axis]
                if g["classification"] == "NOT_CERTIFIABLE":
                    retired.append({"arm": arm, "axis": axis,
                                    "fatal_conditions": g["fatal_conditions"],
                                    "ni_run": False})
                    continue
                r = ni_rule(data[arm][axis], data["intact_7B_base"][axis],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=_seed_off(arm_index[arm], axis))
                null_a = (nulls["mmlu_content"]["by_convention"][conv]
                          if axis == "mmlu_content" else nulls[axis]["acc"])
                arm_resid = reported[arm][axis] - null_a
                ir = g["residual_intact_pp"] / 100.0
                se = ((r["diff_mean_pp"] - r["diff_lower95_one_sided_pp"]) / 1.6449
                      if r["diff_mean_pp"] != r["diff_lower95_one_sided_pp"]
                      else None)
                margin = r["diff_lower95_one_sided_pp"] + r["delta_pp"]
                cells.append({
                    "arm": arm, "axis": axis,
                    "decision_axis": axis not in DEMOTED_AXES,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact_7B_base"][axis],
                    "null": null_a,
                    "residual_arm_pp": 100.0 * arm_resid,
                    "residual_intact_pp": g["residual_intact_pp"],
                    "residual_fraction_recovered": (arm_resid / ir if ir > 0
                                                    else None),
                    "deficit_pp": g["residual_intact_pp"] - 100.0 * arm_resid,
                    "margin_pp": margin,
                    "bootstrap_se_pp": se,
                    "se_to_flip_NI": (abs(margin) / se) if se else None,
                    **r,
                })
        per_conv[conv] = {
            "intact_residual_pp": {x: guard[conv][x]["residual_intact_pp"]
                                   for x in AXES},
            "delta_pp": {x: guard[conv][x]["delta_pp"] for x in AXES},
            "cells": cells, "retired_cells": retired,
            "ratio_rule": {a: ratio_rule(reported[a],
                                         reported["intact_7B_base"],
                                         PREREG["rho"],
                                         [x for x in AXES if x in data[a]])
                           for a in arm_names},
        }

    def cell(conv, arm, axis):
        for c in per_conv[conv]["cells"]:
            if c["arm"] == arm and c["axis"] == axis:
                return c
        return None

    # ---- 6. per-cluster ranges, each gated by its OWN sigma -----------
    per_cluster = {}
    for cname, steps in clusters.items():
        names = [_arm_name(s) for s in steps]
        entry = {
            "steps": list(steps), "spacing_steps": 500,
            "role": ("EARLIER / HIGHER LR" if cname == CLUSTER_HI_LR
                     else "LATER / LOWER LR (the archived headline)"),
            "arm_arch": ARM_ARCH[ARM_KEY],
            "seam": seam["per_cluster"][cname],
            "lr": lr_clusters[cname],
            "per_axis": {},
        }
        for axis in AXES:
            cs = [cell("split", nm, axis) for nm in names]
            if any(x is None for x in cs):
                continue
            mr = range_report([x["margin_pp"] for x in cs],
                              [x["bootstrap_se_pp"] for x in cs],
                              f"{cname}|{axis}|margin")
            ar = range_report([100.0 * x["reported"] for x in cs],
                              [x["bootstrap_se_pp"] for x in cs],
                              f"{cname}|{axis}|acc")
            entry["per_axis"][axis] = {
                "decision_axis": axis not in DEMOTED_AXES,
                "per_step": {str(st): {
                    "acc": cs[i]["reported"],
                    "margin_pp": cs[i]["margin_pp"],
                    "lo95_pp": cs[i]["diff_lower95_one_sided_pp"],
                    "delta_pp": cs[i]["delta_pp"],
                    "bootstrap_se_pp": cs[i]["bootstrap_se_pp"],
                    "ni_accept": cs[i]["ni_accept"],
                    "lr_logged": lr_measured["per_step"][str(st)]["lr_logged"],
                } for i, st in enumerate(steps)},
                "margin_range": mr,
                "accuracy_range_pp": ar,
                "sigma_is_this_clusters_own": {
                    "mean_bootstrap_se_pp": mr["mean_bootstrap_se_pp"],
                    "k_checkpoints": mr["k_checkpoints"],
                    "gate_constant_used": EXPECTED_RANGE_OVER_SD[
                        mr["k_checkpoints"]],
                    "floor_pp": mr["expected_range_if_pure_noise_pp"],
                    "note": ("computed from THIS cluster's three bootstrap SEs "
                             "only; the other cluster's sigma never enters"),
                },
                "adjacent_interval_paired_tests": adjacent_interval_tests(
                    data, axis, ARM_KEY, steps, SEED + INTERVAL_SEED_OFF),
                "any_accept": bool(any(x["ni_accept"] for x in cs)),
            }
            if axis != "mmlu_content":
                entry["per_axis"][axis]["output_shape_and_flips"] = (
                    output_shape_and_flips(data, axis, ARM_KEY, steps))
        per_cluster[cname] = entry

    # ---- 7. reproduction check vs the archive ------------------------
    # THE ACCURACY range is bootstrap-FREE (it is 100*max(acc) - 100*min(acc)),
    # so it MUST reproduce the archive EXACTLY. The MARGIN range contains a
    # bootstrap 5th percentile, so it moves when the bootstrap seed moves -- and
    # the seed MUST move, because reusing the archive's arm_index is exactly what
    # assert_seeds_disjoint forbids. So the two are checked with DIFFERENT
    # tolerances and the distinction is the diagnostic, not an excuse.
    repro = {"acc_tolerance_pp": 5e-4, "margin_tolerance_pp": 5e-4,
             "per_cell": {}, "n_compared": 0,
             "n_margin_within_tolerance": 0, "n_acc_exact": 0,
             "max_abs_margin_drift_pp": 0.0, "max_abs_acc_drift_pp": 0.0}
    for cname in clusters:
        for axis, v in per_cluster[cname]["per_axis"].items():
            a = arch["per_cluster"][cname]["per_axis"].get(axis)
            if a is None:
                continue
            mine = v["margin_range"]["range_pp"]
            theirs = a["margin_range_pp"]
            drift = abs(mine - theirs)
            amine = v["accuracy_range_pp"]["range_pp"]
            atheirs = a["accuracy_range_pp"]
            adrift = abs(amine - atheirs)
            gate_ok = (bool(v["margin_range"]["range_exceeds_item_noise"])
                       == bool(a["margin_gate"]))
            repro["per_cell"][f"{cname}|{axis}"] = {
                "recomputed_margin_range_pp": mine,
                "archive_margin_range_pp": theirs,
                "abs_margin_drift_pp": drift,
                "margin_within_5e_4": bool(drift <= 5e-4),
                "recomputed_acc_range_pp": amine,
                "archive_acc_range_pp": atheirs,
                "abs_acc_drift_pp": adrift,
                "acc_reproduces_exactly": bool(adrift <= 5e-4),
                "gate_agrees": gate_ok,
                "recomputed_gate": v["margin_range"]["range_exceeds_item_noise"],
                "archive_gate": a["margin_gate"],
            }
            repro["n_compared"] += 1
            repro["n_margin_within_tolerance"] += int(drift <= 5e-4)
            repro["n_acc_exact"] += int(adrift <= 5e-4)
            repro["max_abs_margin_drift_pp"] = max(
                repro["max_abs_margin_drift_pp"], drift)
            repro["max_abs_acc_drift_pp"] = max(
                repro["max_abs_acc_drift_pp"], adrift)
            if not gate_ok:
                raise SystemExit(
                    f"FATAL: recomputed gate for {cname}/{axis} disagrees with "
                    "the archive; the recomputation is not the same measurement.")
    # HARD requirement: the bootstrap-free statistic must be exact. If THIS
    # fails, the data or the metric changed and nothing may be published.
    if repro["n_acc_exact"] != repro["n_compared"]:
        raise SystemExit(
            "FATAL: the bootstrap-FREE accuracy range does not reproduce the "
            f"archive ({repro['n_acc_exact']}/{repro['n_compared']} exact, max "
            f"drift {repro['max_abs_acc_drift_pp']:.6f} pp). That cannot be a "
            "seed effect -- the data, the item set or the metric changed.")
    repro["all_gates_agree"] = True
    repro["all_acc_exact"] = True
    repro["all_margin_within_tolerance"] = bool(
        repro["n_margin_within_tolerance"] == repro["n_compared"])
    repro["diagnosis"] = (
        "The ACCURACY range reproduces the archive EXACTLY on all 8 cells "
        "(bootstrap-free: it is max(acc)-min(acc) over the same item vectors), "
        "which proves the item set, the metric, the null, Delta and the loaded "
        "shards are IDENTICAL to the archive's. The MARGIN range differs by "
        "0.007-0.028 pp because `ni_rule`'s bootstrap seed is "
        "SEED + 97*arm_index + 13*axis and this run is REQUIRED to use a "
        "disjoint arm_index (1000-1005 vs the archive's 400-405). Every gate "
        "boolean is unchanged. See `seed_mechanism_control` for the executed "
        "proof: re-running with the ARCHIVE's own arm_index reproduces the "
        "archive to <=5e-4 pp.")
    repro["this_is_NOT_the_numpy_drift"] = (
        "the documented cross-node numpy multinomial drift is 0.005294 pp max "
        "and triviaqa-only. The drifts here reach 0.028 pp and touch mmlu_content "
        "and nq_open, so they are NOT that effect and must not be attributed to "
        "it. This run and the archive are BOTH on .82 / numpy 2.4.6, so the "
        "numpy split cannot be operating at all.")
    repro["consequence_for_precision"] = (
        "a margin range from this family is reproducible to ~0.03 pp ACROSS "
        "SEED CHOICES, not to 5e-4 pp. Any future assertion that hard-fails on "
        "a 5e-4 pp margin reproduction is therefore only valid at FIXED "
        "arm_index -- a second latent tooling hard-fail alongside the numpy one "
        "already recorded in neighbour_variability_20260813 4.1. The finding "
        "sizes here (0.28 vs 1.10 pp) are 10-40x the seed drift.")

    # ---- 7b. EXECUTED proof that the drift is the seed, nothing else ---
    archive_ai = json.load(open(args.archive))["bootstrap_offsets"]["arm_index"]
    seedctl = seed_mechanism_control(data, nulls, guard, archive_ai, clusters)
    # now compare the archive-seed recomputation against the archive itself
    arch_blob = json.load(open(args.archive))["leg_A_neighbour_variability"]
    for key, rec in seedctl["per_cell"].items():
        cname, axis, st = key.split("|")
        ref = (arch_blob[cname]["per_axis"][axis]["per_step"][st]["margin_pp"])
        drift = abs(rec["margin_pp_with_archive_seed"] - ref)
        rec["archive_margin_pp"] = ref
        rec["abs_drift_pp"] = drift
        rec["reproduces_archive_5e_4"] = bool(drift <= 5e-4)
        seedctl["n_compared"] += 1
        seedctl["n_reproduced_5e_4"] += int(drift <= 5e-4)
        seedctl["max_abs_drift_pp"] = max(seedctl["max_abs_drift_pp"], drift)
    seedctl["all_reproduced"] = bool(
        seedctl["n_reproduced_5e_4"] == seedctl["n_compared"])
    if not seedctl["all_reproduced"]:
        raise SystemExit(
            "FATAL: even with the ARCHIVE's own arm_index the margins do not "
            f"reproduce ({seedctl['n_reproduced_5e_4']}/"
            f"{seedctl['n_compared']} within 5e-4 pp, max drift "
            f"{seedctl['max_abs_drift_pp']:.6f} pp). Then the difference is NOT "
            "the seed and something substantive changed -- refusing to publish.")
    seedctl["conclusion"] = (
        "With the archive's arm_index, all "
        f"{seedctl['n_compared']} per-checkpoint margins reproduce the archive "
        f"to <= {seedctl['max_abs_drift_pp']:.2e} pp. The code, data, item set, "
        "null, Delta and numpy are therefore identical to the archive's, and the "
        "0.007-0.028 pp margin drift in `reproduction_vs_archive` is ENTIRELY "
        "attributable to the MANDATED disjoint bootstrap seed. This is executed, "
        "not asserted.")
    seedctl["why_the_seed_had_to_change"] = (
        "assert_seeds_disjoint forbids reusing arm_index 400-405: if this run "
        "took them, re-running the ARCHIVE later would consume the same RNG "
        "stream and could silently perturb published numbers. The disjointness "
        "rule and bit-exact margin reproduction are MUTUALLY EXCLUSIVE for this "
        "estimator; the rule wins, and the cost is ~0.03 pp of margin precision.")

    # ---- 8. MAIN's hand-computed accuracy ranges, checked -----------
    main_check = {"what_MAIN_computed": "ACCURACY range, not NI-margin range",
                  "per_axis": {}}
    for axis, exp in MAIN_HAND_ACC_RANGES_PP.items():
        c1 = per_cluster[CLUSTER_HI_LR]["per_axis"].get(axis)
        c2 = per_cluster[CLUSTER_LO_LR]["per_axis"].get(axis)
        if not (c1 and c2):
            continue
        g1 = c1["accuracy_range_pp"]["range_pp"]
        g2 = c2["accuracy_range_pp"]["range_pp"]
        m1 = c1["margin_range"]["range_pp"]
        m2 = c2["margin_range"]["range_pp"]
        main_check["per_axis"][axis] = {
            "MAIN_acc_c1": exp["c1"], "canonical_acc_c1": g1,
            "acc_c1_agrees_to_1e_3": bool(abs(g1 - exp["c1"]) < 1e-3),
            "MAIN_acc_c2": exp["c2"], "canonical_acc_c2": g2,
            "acc_c2_agrees_to_1e_3": bool(abs(g2 - exp["c2"]) < 1e-3),
            "MAIN_acc_ratio_c1_over_c2": exp["c1"] / exp["c2"],
            "canonical_acc_ratio_c1_over_c2": g1 / g2 if g2 else None,
            "canonical_MARGIN_ratio_c1_over_c2": m1 / m2 if m2 else None,
            "margin_and_acc_ratio_differ": bool(
                g2 and m2 and abs((g1 / g2) - (m1 / m2)) > 1e-6),
        }
    main_check["verdict_on_MAINs_arithmetic"] = (
        "checked per axis above; `acc_*_agrees_to_1e_3` says whether MAIN's "
        "hand-computed ACCURACY ranges reproduce. Note they are a DIFFERENT "
        "statistic from the decision-bearing margin range, so even exact "
        "agreement would not license using them for Q1/Q2.")

    # ---- 9. THE VERDICT, emitted mechanically -----------------------
    prim = {}
    for cname in (CLUSTER_HI_LR, CLUSTER_LO_LR):
        v = per_cluster[cname]["per_axis"][PRIMARY_AXIS]["margin_range"]
        prim[cname] = {"range_pp": v["range_pp"],
                       "floor_pp": v["expected_range_if_pure_noise_pp"],
                       "ratio_to_floor": v["range_over_expected_noise_range"],
                       "clears": bool(v["range_exceeds_item_noise"])}
    R = (prim[CLUSTER_HI_LR]["range_pp"] / prim[CLUSTER_LO_LR]["range_pp"]
         if prim[CLUSTER_LO_LR]["range_pp"] else None)
    both_clear = prim[CLUSTER_HI_LR]["clears"] and prim[CLUSTER_LO_LR]["clears"]
    if not both_clear:
        label = "UNRESOLVED_SUBNOISE"
    elif R is not None and R <= 0.83:
        label = "REFUTED_WITHIN_ARM"
    elif R is not None and R >= 1.20:
        label = "SUPPORTED_WITHIN_ARM"
    else:
        label = "UNRESOLVED_UNDERPOWERED"
    seam_bad = per_cluster[CLUSTER_HI_LR]["seam"]["resume_seam"]
    verdict_string = label + ("__INADMISSIBLE_SEAM" if seam_bad else "")

    # per-axis, for completeness -- the primary axis decides the label
    per_axis_ratios = {}
    for axis in AXES:
        c1 = per_cluster[CLUSTER_HI_LR]["per_axis"].get(axis)
        c2 = per_cluster[CLUSTER_LO_LR]["per_axis"].get(axis)
        if not (c1 and c2):
            continue
        a, b = c1["margin_range"], c2["margin_range"]
        per_axis_ratios[axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "range_hi_lr_pp": a["range_pp"], "range_lo_lr_pp": b["range_pp"],
            "floor_hi_lr_pp": a["expected_range_if_pure_noise_pp"],
            "floor_lo_lr_pp": b["expected_range_if_pure_noise_pp"],
            "clears_hi_lr": bool(a["range_exceeds_item_noise"]),
            "clears_lo_lr": bool(b["range_exceeds_item_noise"]),
            "R_hi_over_lo": (a["range_pp"] / b["range_pp"]
                             if b["range_pp"] else None),
            "both_clear": bool(a["range_exceeds_item_noise"]
                               and b["range_exceeds_item_noise"]),
            "comparison_is_defined": bool(a["range_exceeds_item_noise"]
                                          and b["range_exceeds_item_noise"]),
        }

    # ---- 9b. is the LABEL seed-dependent? ----------------------------
    # The margin range moved 0.017 pp on triviaqa when the seed moved. If the
    # label can flip under a seed change, the label is not a finding. Tested by
    # recomputing the primary-axis decision under the ARCHIVE's seeds too.
    label_sens = {"seeds_tried": ["this_run_1000_1005", "archive_400_405"],
                  "per_seed": {}}
    for tag, ai_map in (("this_run_1000_1005", arm_index),
                        ("archive_400_405", archive_ai)):
        rec = {}
        for cname, steps in clusters.items():
            g = guard["split"][PRIMARY_AXIS]
            ms, ses = [], []
            for st in steps:
                nm = _arm_name(st)
                ai = ai_map.get(nm)
                if ai is None:
                    ms = None
                    break
                r = ni_rule(data[nm][PRIMARY_AXIS],
                            data["intact_7B_base"][PRIMARY_AXIS],
                            PREREG["delta_fraction"],
                            g["residual_intact_pp"] / 100.0,
                            seed_off=_seed_off(ai, PRIMARY_AXIS))
                ms.append(r["diff_lower95_one_sided_pp"] + r["delta_pp"])
                ses.append((r["diff_mean_pp"] - r["diff_lower95_one_sided_pp"])
                           / 1.6449)
            if ms is None:
                continue
            rr = range_report(ms, ses, f"labelsens|{tag}|{cname}")
            rec[cname] = {"range_pp": rr["range_pp"],
                          "floor_pp": rr["expected_range_if_pure_noise_pp"],
                          "clears": bool(rr["range_exceeds_item_noise"])}
        if len(rec) == 2:
            hi, lo = rec[CLUSTER_HI_LR], rec[CLUSTER_LO_LR]
            bc = hi["clears"] and lo["clears"]
            rv = hi["range_pp"] / lo["range_pp"] if lo["range_pp"] else None
            lb = ("UNRESOLVED_SUBNOISE" if not bc else
                  "REFUTED_WITHIN_ARM" if rv is not None and rv <= 0.83 else
                  "SUPPORTED_WITHIN_ARM" if rv is not None and rv >= 1.20 else
                  "UNRESOLVED_UNDERPOWERED")
            label_sens["per_seed"][tag] = {"per_cluster": rec, "R": rv,
                                           "both_clear": bool(bc),
                                           "label": lb}
    labels_seen = sorted({v["label"] for v in label_sens["per_seed"].values()})
    label_sens["labels_seen"] = labels_seen
    label_sens["label_is_seed_invariant"] = bool(len(labels_seen) == 1)
    label_sens["reading"] = (
        "the verdict label is recomputed under BOTH bootstrap seed choices. If "
        "`label_is_seed_invariant` is false the label is an artefact of the seed "
        "and may not be reported as a finding."
        if not label_sens["label_is_seed_invariant"] else
        "the label is IDENTICAL under both seed choices, so the ~0.03 pp seed "
        "drift cannot have manufactured it.")
    if not label_sens["label_is_seed_invariant"]:
        raise SystemExit(
            f"FATAL: the verdict label is seed-dependent ({labels_seen}). "
            "Refusing to publish a label that a mandated seed change flips.")

    # ---- 9c. POST-HOC SUPPLEMENT (NOT the pre-registered statistic) ----
    # The pre-registered comparison is the RANGE, and it is undefined here
    # because cluster1 fails its noise gate. A range is a max-minus-min of noisy
    # cells with no inferential content of its own -- which is why it needs a
    # gate at all. The ADJACENT-INTERVAL paired bootstrap is a proper test (own
    # CI, own p) and is therefore DEFINED regardless of the gate. It is reported
    # here as a supplement, and it is FLAGGED POST-HOC: it was not named in the
    # prereg, it does not set the verdict label, and it may not be substituted
    # for the pre-registered statistic. Recorded because suppressing a defined
    # comparison merely because the pre-registered one came out undefined would
    # itself be a selection effect.
    posthoc = {"IS_POST_HOC": True,
               "not_in_prereg": True,
               "does_not_set_the_verdict": True,
               "statistic": ("per-axis largest ABSOLUTE resolved adjacent "
                             "500-step accuracy move, where 'resolved' = the "
                             "imported conservative AND (CI95 excludes zero AND "
                             "boot p < 0.05)"),
               "why_it_is_defined_when_the_range_is_not": (
                   "each interval carries its own paired-bootstrap CI and p, so "
                   "it is an inferential test rather than a max-minus-min; the "
                   "item-noise gate that disqualifies a range does not apply"),
               "per_axis": {}}
    for axis in AXES:
        rec = {}
        for cname in (CLUSTER_HI_LR, CLUSTER_LO_LR):
            v = per_cluster[cname]["per_axis"].get(axis)
            if v is None:
                continue
            ivs = v["adjacent_interval_paired_tests"]
            res = {k: t for k, t in ivs.items()
                   if t["distinguishable_from_zero_at_95"]}
            best = (max(res.items(), key=lambda kv: abs(kv[1]["acc_delta_pp"]))
                    if res else None)
            rec[cname] = {
                "n_intervals": len(ivs),
                "n_resolved": len(res),
                "resolved_intervals": sorted(res),
                "largest_resolved_abs_move_pp": (abs(best[1]["acc_delta_pp"])
                                                 if best else None),
                "largest_resolved_interval": (best[0] if best else None),
                "largest_resolved_signed_pp": (best[1]["acc_delta_pp"]
                                               if best else None),
                "largest_resolved_p": (best[1]["boot_p_two_sided"]
                                       if best else None),
            }
        if len(rec) == 2:
            a = rec[CLUSTER_HI_LR]["largest_resolved_abs_move_pp"]
            b = rec[CLUSTER_LO_LR]["largest_resolved_abs_move_pp"]
            rec["R_hi_over_lo"] = (a / b) if (a and b) else None
            rec["both_have_a_resolved_move"] = bool(a and b)
            # is the hi-LR cluster's only resolved move the SEAM-CROSSING one?
            li = rec[CLUSTER_HI_LR]["largest_resolved_interval"]
            seam_iv = "124500->125000"
            rec["hi_lr_largest_resolved_is_the_seam_interval"] = bool(
                li == seam_iv)
        posthoc["per_axis"][axis] = rec
    pa = posthoc["per_axis"].get(PRIMARY_AXIS, {})
    posthoc["primary_axis_reading"] = (
        "On triviaqa the LOWER-LR cluster2 carries the larger resolved move "
        f"({pa.get(CLUSTER_LO_LR, {}).get('largest_resolved_abs_move_pp')} pp) "
        "and the HIGHER-LR cluster1 the smaller "
        f"({pa.get(CLUSTER_HI_LR, {}).get('largest_resolved_abs_move_pp')} pp), "
        f"R = {pa.get('R_hi_over_lo')}. The DIRECTION is opposite to H_LR. But "
        "this is NOT admissible as a refutation, for two independent reasons: "
        "(1) it is post-hoc, and (2) cluster1's ONLY resolved interval IS the "
        "seam-crossing 124500->125000 interval "
        f"(hi_lr_largest_resolved_is_the_seam_interval = "
        f"{pa.get('hi_lr_largest_resolved_is_the_seam_interval')}), so the one "
        "number that would carry the refutation is precisely the one the seam "
        "disqualifies. The two defects are not redundant -- either alone is "
        "sufficient to block the inference.")
    posthoc["what_it_DOES_license"] = (
        "a bound: within one arm, at two positions 6000 steps apart on the same "
        "cosine schedule, the largest resolved adjacent-500-step triviaqa move "
        "differs by roughly 4x. So checkpoint-selection exposure is NOT a "
        "constant of an arm. That statement needs no LR interpretation and "
        "survives the seam caveat, because it is about the SPREAD of the "
        "statistic across positions, not about which position is larger.")

    lrc = lr_clusters["contrast"]
    xarm = crossarm_lr
    k10 = xarm.get("keep10fresh2", {}).get("lr_at_last_step")
    k12 = xarm.get("keep12fresh2", {}).get("lr_at_last_step")
    crossarm_ratio = (k10 / k12) if (k10 and k12) else None
    power = {
        "within_arm_lr_ratio_mean": lrc["lr_ratio_mean"],
        "within_arm_lr_ratio_last_step": lrc["lr_ratio_last_step"],
        "crossarm_lr_ratio_keep10_over_keep12_measured": crossarm_ratio,
        "crossarm_range_ratio_keep10_over_keep12_as_published": (
            H_LR_AS_GENERATED["table_as_published"]["keep10"]["triviaqa_range_pp"]
            / H_LR_AS_GENERATED["table_as_published"]["keep12"]["triviaqa_range_pp"]),
        # THE FRACTION IS SCALE-DEPENDENT, so all three defensible scalings are
        # given rather than one cherry-picked number. The PREREG said "~13%",
        # which matches NONE of them -- corrected here, see `prereg_correction`.
        "within_over_crossarm_excess_ratio_scale": (
            (lrc["lr_ratio_mean"] - 1.0) / (crossarm_ratio - 1.0)
            if crossarm_ratio and crossarm_ratio != 1.0 else None),
        "within_over_crossarm_log_scale": (
            math.log(lrc["lr_ratio_mean"]) / math.log(crossarm_ratio)
            if crossarm_ratio and crossarm_ratio > 1.0 else None),
        "within_over_crossarm_raw_ratio_scale": (
            lrc["lr_ratio_mean"] / crossarm_ratio if crossarm_ratio else None),
        "prereg_correction": (
            "A04_WITHIN_ARM_LR_PREREG.md 3 wrote '~13 % of the 3.82x contrast'. "
            "That figure is WRONG and matches no defensible scaling: the excess-"
            "ratio scale gives 3.99 %, the log scale 7.93 %, the raw-ratio scale "
            "29.23 %. The prereg's QUALITATIVE point -- that the within-arm "
            "contrast is a small fraction of the cross-arm one and therefore "
            "underpowered -- is unaffected and is if anything STRENGTHENED (the "
            "true fraction on the two natural scales is SMALLER than 13 %). "
            "Recorded rather than silently fixed, because the prereg is "
            "committed and its arithmetic errors are part of the record."),
        "reading": (
            "the within-arm LR contrast is a small fraction of the cross-arm "
            "contrast H_LR was fitted on (3.99 % on the excess-ratio scale, "
            "7.93 % on the log scale). Under any smooth monotone LR effect the "
            "PREDICTED within-arm range difference is small, so a null result "
            "here DOES NOT refute H_LR. Recorded in the prereg BEFORE the "
            "ranges were read."),
        "crossarm_lr_note": (
            "keep12's LR at its triple is MEASURED here as 3.26e-6 at step166000 "
            "(mean over the triple 3.2933e-6), whereas "
            "A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md 5's table says "
            "3.25e-6. The measured value is the trainer's own logged line; the "
            "0.3 % discrepancy does not change any ordering. keep10 is logged at "
            "1.24e-6*10 = 1.24e-5 at step90000, matching that table."),
    }

    out = {
        "gate": "A04_within_arm_LR_contrast_keep8fresh2_cluster1_vs_cluster2",
        "question": ("does the LR hypothesis from "
                     "A04_KEEP12_TRAJECTORY_MONOTONICITY_VERDICT.md 5 survive a "
                     "WITHIN-ARM contrast between two 500-step clusters of "
                     "keep8+fresh2 that differ only in schedule position?"),
        "date": "2026-08-13",
        "gpu_h_spent": 0,
        "gpu_note": ("ZERO GPU. Both clusters' per-example shards were already "
                     "on disk from neighbour_variability_20260813 (8.6556 "
                     "GPU-h, already counted there). No model was loaded; no "
                     "node's GPU was touched. CPU-only re-analysis."),
        "node_label": args.node_label,
        "numpy_version": np.__version__,
        "numpy_required": PUBLISH_NUMPY,
        "prereg": {
            "document": "A04_WITHIN_ARM_LR_PREREG.md",
            "committed_before_any_number": True,
            "criteria": VERDICT_CRITERIA,
        },
        "hypothesis_under_test": H_LR_AS_GENERATED,
        "headline_verdict": verdict_string,
        "verdict_label": label,
        "verdict_seam_modifier": bool(seam_bad),
        "primary_axis_decision_inputs": prim,
        "R_hi_lr_over_lo_lr": R,
        "both_clusters_clear_their_own_gate": bool(both_clear),
        "per_axis_range_comparison": per_axis_ratios,
        "verdict_label_seed_sensitivity": label_sens,
        "post_hoc_supplement_resolved_intervals": posthoc,
        "power_statement": power,
        "lr_schedule": LR_SCHEDULE,
        "lr_measured_from_logs": lr_measured,
        "lr_by_cluster": lr_clusters,
        "crossarm_lr_measured": crossarm_lr,
        "seam_verification": seam,
        "archive_readback": arch,
        "reproduction_vs_archive": repro,
        "seed_mechanism_control": seedctl,
        "MAIN_hand_arithmetic_check": main_check,
        "per_cluster": per_cluster,
        "protocol_asserted": proto,
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "nulls": {k: (v if not isinstance(v, dict) else
                      {kk: vv for kk, vv in v.items()
                       if kk not in ("vectors", "vector")})
                  for k, v in nulls.items()},
        "per_convention": {c: {k: v for k, v in per_conv[c].items()}
                           for c in per_conv},
        "guard_D1_D6": {c: {a: {k: v for k, v in guard[c][a].items()}
                            for a in guard[c]} for c in guard},
        "bootstrap_offsets": {
            "arm_index": arm_index,
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
            "guard_seed_offset_actually_used": (
                "SEED+1700+13*axis_index -- the IMPORTED guard_cell's own "
                "constant. Delta/residual(intact) is a property of the ANCHOR, "
                "not of this arm set, so reusing it recomputes the SAME "
                "quantity and cannot perturb the archive. Our own 6700 is "
                "reserved and unused; stated rather than silently ignored."),
            "interval_seed_offset": f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair",
            "ni_seed_offset": "97*arm_index + 13*axis_index",
            "disjointness": seeds,
        },
        "selftest_gate_constants": st_gates,
        "selftest_sigma_per_cluster": st_sigma,
        "NOT_licensed": [
            "treating the two clusters as REPLICATES -- they are successive "
            "states of one optimisation; the range is a checkpoint-SELECTION "
            "quantity, never seed variance. No 7B sd_run exists.",
            "promoting cluster1 to a clean 500-step neighbourhood -- it "
            "straddles a resume seam (verified from the logs here).",
            "claiming H_LR is CONFIRMED: n=3 arms plus a ~1.11x within-arm LR "
            "contrast cannot confirm a schedule law.",
            "claiming H_LR is REFUTED unless the verdict label literally reads "
            "REFUTED_WITHIN_ARM.",
            "quoting any margin to better than 0.01 pp across nodes (numpy "
            "multinomial split 2.4.6 vs 2.5.1).",
            "any K1/K2/K3 clause -- those are defined over the 1B arm set.",
            "using the ACCURACY range in place of the MARGIN range for any "
            "gate-design statement.",
        ],
    }
    with open(args.out_json, "w") as f:
        json.dump(out, f, indent=2, default=float)
    print(f"[within_arm_lr] verdict = {verdict_string}")
    print(f"[within_arm_lr] R = {R}")
    print(f"[within_arm_lr] wrote {args.out_json}")


if __name__ == "__main__":
    main()
