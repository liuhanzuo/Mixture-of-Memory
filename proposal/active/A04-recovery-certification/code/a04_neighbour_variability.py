#!/usr/bin/env python3
"""A04 — NEIGHBOUR VARIABILITY of the NI margin, and cross-arm replication of
the popqa mid-heal regression.

WHAT IS BEING TESTED (and it is a claim, not a model)
-----------------------------------------------------
`A04_KEEP14_TRAJECTORY_NI_VERDICT.md` 7.2 (commit 517c8d2) concluded, from ONE
arm at 25 500-step spacing:

    "a certification rule evaluated at a single arbitrary checkpoint can return
     a BETTER verdict than a later checkpoint of the same run, so any future
     accept obtained at a hand-picked checkpoint must be shown to survive its
     neighbours."

Two things about that are untested, and this script tests both.

LEG A — HOW BIG IS THE HAND-PICKING GAP AT REALISTIC SPACING?
   Hand-picking does not happen 25 500 steps apart; it happens between ADJACENT
   saves. The claim therefore imposes a requirement ("must survive its
   neighbours") with NO TOLERANCE attached, because checkpoint-to-checkpoint
   margin variability has never been measured. Two 500-step-spaced clusters of
   keep8+fresh2 give the per-axis RANGE (max-min) directly, which IS the size of
   the false advantage available to someone who picks the best of three
   neighbours.
   Both outcomes are informative and both are pre-committed here:
     * range >= 0.5 pp  -> the neighbour requirement is a quantitative necessity
       and this measurement supplies the tolerance a future accept must clear.
     * range ~ 0 (< 0.05 pp) -> the rule-level claim must be NARROWED to
       long-range (10k+ step) spacing; the 25 500-step popqa dip may NOT be
       extrapolated to "adjacent checkpoints are untrustworthy".
   AND THE THIRD OUTCOME, which is the one that actually needs guarding against:
     * range comparable to or smaller than the bootstrap SE of a single cell
       -> the range is NOT DISTINGUISHABLE FROM ITEM NOISE and must not be
       reported as a finding at all. So this script computes, for every axis,
       range / mean(bootstrap SE) and an explicit
       `range_exceeds_item_noise` boolean, and the headline verdict cannot claim
       a measured gap unless that boolean is true. A range is a max-minus-min of
       3 noisy numbers and is BIASED UPWARD by noise even when the true spread
       is zero: E[range of 3 iid N(0,s)] = 1.693 s. That expectation is computed
       and reported next to every observed range, so "we measured a 0.3 pp
       neighbour gap" cannot be said when 1.693*SE is also 0.3 pp.

LEG B — DOES THE popqa 128k->153.5k REGRESSION REPLICATE ON ANOTHER ARM?
   n=1 arm cannot separate "a general property of healing" from "something that
   happened in this run". shortgpt16 has ckpts at the same three step numbers.
   BOTH READINGS ARE WRITTEN DOWN BEFORE THE NUMBERS ARE SEEN (and are in this
   docstring, in the JSON's `prereg_readings`, and in the verdict doc):
     * shortgpt16's popqa ALSO regresses resolvedly over 128k->153.5k -> the
       phenomenon is not keep14-specific; the rule-level claim is strengthened.
     * it does NOT regress -> the claim must be DOWNGRADED to "there exists at
       least one arm on which this happens", which still justifies requiring a
       neighbour check (one counterexample is enough to make single-checkpoint
       accepts unsafe) but may NOT be stated as a general property of healing.
   Neither reading is selected after the fact.

WHY THE TWO ARMS ARE NOT INTERCHANGEABLE (verified before any GPU was spent)
---------------------------------------------------------------------------
  * keep8fresh2: keep_front=8, n_fresh=2, num_hidden_layers=10, 113 tensors.
  * keep14fresh2: keep_front=14, n_fresh=2, 16 layers, 179 tensors.
  * shortgpt16: keep_front=16, n_fresh=0, 16 layers, 179 tensors,
    keep_layer_indices [0..12, 16, 17, 31] -- a NON-CONTIGUOUS selection.
  So absolute margins from the three arms live on different curves and are never
  tabulated as rungs of one ladder. Leg A is a WITHIN-ARM range (which is
  exactly what the claim is about); Leg B is a replication of a PHENOMENON, not
  a matched pairwise comparison. Also: "step" is only an optimizer-step count.
  shortgpt16 is at epoch 2 by 153500 and epoch 3 by 200000; keep14 is at epoch 1
  at 128000 and epoch 2 at 153500. SAME STEP != SAME DATA SEEN.

THE CLUSTER-1 RESUME SEAM (found before scoring; do not paper over it)
---------------------------------------------------------------------
Cluster 1 = {124000, 124500, 125000} STRADDLES A PROCESS BOUNDARY.
`logs/olmo2_7B_keep8fresh2_resume200k_73.log` (2026-08-08) resumed from
step121000_full.pt and wrote 124000 and 124500 before dying at 20:26 with a
TCPStore error. `logs/keep8_resume_82_launch.out` (2026-08-12 00:34, a DIFFERENT
process on a DIFFERENT node) resumed FROM step124500.pt and wrote 125000. The
trainer restores optimizer state and RNG but rebuilds the loader
(`sampler.set_epoch(epoch); data_iter = iter(loader)`) WITHOUT fast-forwarding
within the epoch, so the 124500->125000 interval saw a different data order than
an uninterrupted 500 steps would have. Cluster 2 = {130000, 130500, 131000} is
entirely inside the single .82 process with no seam. Both are reported; cluster 2
is the CLEAN neighbour measurement and the headline uses it. Whether the seam
widens the range is reported as a secondary observation, not as the answer.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
`qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`,
`SEED` from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`,
`assert_aligned`, `d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`,
`Z95_TWO_SIDED`, `D4_*`, `SD_RUN_1B_PP` from `a04_shallow_rung_ni_7b`. No
metric, null, rule, guard or anchor is re-derived. Two subagents in this repo
have already manufactured significance by reimplementing a metric, and the
previous pass's own hand-computed null was off by ~0.5 pp.

ANCHOR AND DELTA ARE NOT SUBSTITUTED (guards G0/G2). Anchor = vanilla
`models/OLMo-2-1124-7B` via the imported `ANCHOR`; `full32_step25000` is
forbidden (it scores BELOW vanilla on all four axes, so substituting it would
shrink every Delta AND lower every target = manufactured accepts).
`Delta = 0.10 * residual(intact)`, imported through the guard.

BOOTSTRAP SEEDS. Archived offsets in use: pilot_zero arm_index {0,1}; step100k
100..102; shallow_rung 200..203; keep14 trajectory 300..301 (+ endpoint 201).
This script uses arm_index 400.. for NI cells, SEED+1700+... for the guard and
SEED+1900+... for the adjacent-interval tests -- disjoint from all of the above,
and the offsets actually used are written into the JSON.

CPU ONLY. Read-only on every input.
"""
from __future__ import annotations

import argparse
import glob as _glob
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

# ---------------------------------------------------------------------------
# Arm/axis facts established from ckpt metas + training logs BEFORE any GPU was
# spent. Carried into the JSON so the reader does not have to trust prose.
# ---------------------------------------------------------------------------
ARM_ARCH = {
    "keep8fresh2": {"keep_front": 8, "n_fresh": 2, "num_hidden_layers": 10,
                    "n_tensors": 113, "contiguous_front": True},
    "shortgpt16": {"keep_front": 16, "n_fresh": 0, "num_hidden_layers": 16,
                   "n_tensors": 179, "contiguous_front": False,
                   "keep_layer_indices": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
                                          12, 16, 17, 31]},
    "keep14fresh2": {"keep_front": 14, "n_fresh": 2, "num_hidden_layers": 16,
                     "n_tensors": 179, "contiguous_front": True,
                     "note": "the ARCHIVED arm this work extends; not re-scored"},
}

# LEG A: keep8+fresh2, two 500-step clusters. `resume_seam` records whether the
# cluster crosses a process boundary (see the docstring).
LEG_A_CLUSTERS = {
    "cluster1_124000_125000": {
        "arm_dir": "olmo2_probe2_7B_keep8fresh2",
        "arm_key": "keep8fresh2",
        "steps": [124000, 124500, 125000],
        "spacing_steps": 500,
        "resume_seam": True,
        "resume_seam_detail": (
            "124000 and 124500 were written by the .73 process of 2026-08-08 "
            "(logs/olmo2_7B_keep8fresh2_resume200k_73.log, resumed from "
            "step121000_full.pt, died 20:26 TCPStore error). 125000 was written "
            "by a DIFFERENT process on .82 on 2026-08-12 "
            "(logs/keep8_resume_82_launch.out, resumed FROM step124500.pt). The "
            "trainer restores optimizer+RNG but rebuilds the loader without "
            "fast-forwarding inside the epoch, so 124500->125000 saw a "
            "different data order than an uninterrupted 500 steps."),
        "is_clean_neighbour_measurement": False,
    },
    "cluster2_130000_131000": {
        "arm_dir": "olmo2_probe2_7B_keep8fresh2",
        "arm_key": "keep8fresh2",
        "steps": [130000, 130500, 131000],
        "spacing_steps": 500,
        "resume_seam": False,
        "resume_seam_detail": (
            "all three saves are inside the single .82 process started "
            "2026-08-12 00:34 (logs/keep8_resume_82_launch.out lines 319/347/375) "
            "-- no process boundary, no loader restart, continuous data order. "
            "THIS is the clean neighbour measurement."),
        "is_clean_neighbour_measurement": True,
    },
}

# LEG B: shortgpt16 at the SAME three step numbers as the archived keep14 curve.
LEG_B = {
    "arm_dir": "olmo2_probe2_7B_shortgpt16",
    "arm_key": "shortgpt16",
    "steps": [128000, 153500, 200000],
    "replicates_interval": "128000->153500",
    "keep14_reference_acc_delta_pp": -0.6729,
    "keep14_reference_ci95_pp": [-0.9252, -0.4206],
    "keep14_reference_p": 0.0001,
    "keep14_reference_flips": {"wrong_to_right": 122, "right_to_wrong": 218},
}

# The zwfy6 copy of shortgpt16/step128000.pt is CORRUPT and had to be replaced
# from wzc1. Recorded here because "the number came from a file I had to repair"
# is exactly the provenance a later reader needs and cannot recover from the
# result dirs (summary.json:meta.ckpt records the STAGED path, not its origin).
LEG_B_CKPT_PROVENANCE = {
    "step128000": {
        "defect_found_on_zwfy6": (
            "outputs/olmo2_probe2_7B_shortgpt16/step128000.pt is 7,755,268,096 B "
            "-- 15.9% of its 48,724,473,978 B siblings (step153500/step200000). "
            "BOTH `zipfile.ZipFile` and `torch.load` fail with "
            "'PytorchStreamReader failed reading zip archive: failed finding "
            "central directory ... high likelihood that your checkpoint file is "
            "corrupted'. It is a TRUNCATED write, not a smaller dtype."),
        "why_ls_would_not_catch_it": (
            "the file is present, non-zero, and owned/dated like its siblings; "
            "only its SIZE differs, and this arm's other ckpts do not all share "
            "one size either. Any inventory that greps for existence -- "
            "including the dispatch's own ledger, which listed n_ckpt=6 for this "
            "arm -- reports it as available."),
        "repair": (
            "the wzc1 copy of the SAME path is intact (48,724,473,978 B, 731 zip "
            "entries, `zipfile.testzip()` clean). Staged wzc1 -> zwfy6 via "
            "`scp -O` into outputs/a04_staged/sg16_step128000_from_wzc1.pt, "
            "2592 s at 18.8 MiB/s."),
        "verified_after_transfer": {
            "src_sha256": ("858eb32f389b5fd9b95fa551f296cab329feb176efb49eb70"
                           "247eecec386643c"),
            "dst_sha256": ("858eb32f389b5fd9b95fa551f296cab329feb176efb49eb70"
                           "247eecec386643c"),
            "sha256_match": True,
            "bytes": 48724473978,
            "ckpt_meta_step_asserted_by_driver": 128000,
            "note": ("full-file sha256 on BOTH disks, not a head/tail sample: a "
                     "truncated transfer is precisely the failure being guarded "
                     "against, and the driver additionally asserts the loaded "
                     "meta's step == 128000 before spending GPU."),
        },
        "the_corrupt_file_was_not_deleted": (
            "the zwfy6 original is left in place. Removing it would destroy the "
            "evidence and could mask the same defect in other arms; it is "
            "recorded here and the driver REFUSES to fall back to it (the staged "
            "path must be passed explicitly via SG16_CKPT_128000)."),
    },
}

# Pre-committed readings, fixed BEFORE the numbers were computed (they are also
# in the module docstring, which is under git and predates the result).
PREREG_READINGS = {
    "leg_a_range_ge_0p5pp": (
        "the neighbour-robustness requirement is a QUANTITATIVE necessity and "
        "this range is the tolerance a future accept must clear"),
    "leg_a_range_lt_0p05pp": (
        "the rule-level claim must be NARROWED to long-range (10k+ step) "
        "spacing; the 25 500-step popqa dip may NOT be extrapolated to "
        "'adjacent checkpoints are untrustworthy'"),
    "leg_a_range_within_item_noise": (
        "the range is NOT a measured gap at all -- max-minus-min of 3 noisy "
        "cells is biased upward by noise (E[range of 3 iid N(0,s)] = 1.693 s), "
        "so it must be reported as indistinguishable from item noise and NOT "
        "as a finding"),
    "leg_b_replicates": (
        "the popqa mid-heal regression is NOT keep14-specific; the rule-level "
        "claim is strengthened"),
    "leg_b_does_not_replicate": (
        "the claim must be DOWNGRADED to 'there exists at least one arm on "
        "which this happens'. That still justifies REQUIRING a neighbour check "
        "(one counterexample makes single-checkpoint accepts unsafe) but may "
        "NOT be stated as a general property of healing"),
}

# E[range] of k iid N(0, 1) draws -- the noise floor a max-minus-min inherits.
# k=3 -> 3/sqrt(pi) = 1.6926; k=2 -> 2/sqrt(pi) = 1.1284. Exact for the normal.
EXPECTED_RANGE_OVER_SD = {2: 1.1283791670955126, 3: 1.6925687506432689}

# ---------------------------------------------------------------------------
# CROSS-NODE BOOTSTRAP NON-DETERMINISM (measured 2026-08-13; a REAL defect in
# the repo's reproducibility story, recorded rather than smoothed over)
# ---------------------------------------------------------------------------
# The standing rule `same-harness-runs-bit-identical` is about the SCORING
# harness. It does NOT extend to this ANALYSIS layer, and here is the
# counter-example: running this script on `.73` and `.82` -- same code, same
# input shards, same seeds -- gives margins that differ in the 4th decimal.
#
# Diagnosed to `np.random.Generator.multinomial`, NOT to the data and NOT to the
# metric:
#   * the per-item difference vector `d` is BIT-IDENTICAL on both nodes
#     (sha256 4d0d81b9…, identical (vals, counts) = ([-1,0,1], [8653,9092,199]),
#     identical d.mean() to 20 significant figures);
#   * the underlying bit stream is identical (`default_rng(39091).integers` and
#     `.binomial` agree exactly on both nodes);
#   * but `default_rng(seed).multinomial(n, p, size=10000)` returns arrays that
#     differ in 19 of 10 000 ROWS (first at row 2598: [8655,9082,207] vs
#     [8655,9115,174]). numpy 2.5.1 (.73) vs 2.4.6 (.82) -- the multinomial
#     SAMPLER CHANGED between those versions, so the same bits are consumed
#     differently.
# Effect, measured across all 24 published NI cells: 3 cells move, max
# |margin| drift 0.005294 pp (triviaqa only -- the axis whose p-vector has the
# rare third category). Every verdict boolean, every `range_exceeds_item_noise`
# and the headline are UNCHANGED, and the drift is 211x smaller than Leg A's
# 1.12 pp finding. WITHIN a node the output is BYTE-IDENTICAL on re-run
# (sha256 1f88d6eb… twice).
#
# So: this does not threaten anything claimed here, but it MUST be stated, for
# two reasons. (1) A future agent re-running this on a different node will get
# 4th-decimal disagreement and must not read it as a data problem. (2) The
# archived keep14 endpoint reproduction tolerance (5e-4 pp) is TIGHTER than this
# drift (5.3e-3 pp), so that assertion is only guaranteed to pass on a node with
# the numpy that produced the archive -- a latent hard-fail that has nothing to
# do with scientific drift. Recorded here; fixing it means pinning numpy, which
# is not this dispatch's to change.
BOOTSTRAP_CROSS_NODE_DRIFT = {
    "is_a_real_defect": True,
    "what_is_bit_identical": (
        "the per-item difference vector d (sha256 4d0d81b938392a262fd67c5e on "
        "both nodes), its (vals, counts) = ([-1,0,1], [8653,9092,199]), and "
        "d.mean() to 20 significant figures; also the raw RNG bit stream "
        "(default_rng(seed).integers and .binomial agree exactly)"),
    "what_differs": (
        "np.random.Generator.multinomial(n, p, size=10000) differs in 19 of "
        "10000 rows between numpy 2.5.1 (.73) and numpy 2.4.6 (.82) for the same "
        "seed -- first divergence at row 2598: [8655,9082,207] vs [8655,9115,174]. "
        "The multinomial sampler changed between these versions."),
    "numpy_by_node": {".73": "2.5.1", ".82": "2.4.6"},
    "published_on_node": ".82 (numpy 2.4.6)",
    "measured_effect": {
        "n_cells_compared": 24,
        "n_cells_differing": 3,
        "max_abs_margin_drift_pp": 0.005294,
        "axes_affected": ["triviaqa"],
        "why_only_triviaqa": (
            "its p-vector has a rare third category (199/17944), which is where "
            "the two samplers' handling diverges"),
        "verdict_booleans_changed": 0,
        "headline_changed": False,
        "leg_a_finding_pp": 1.1202,
        "finding_over_drift_ratio": 211.6,
    },
    "within_node_is_byte_identical": (
        "re-running this script twice on .82 gives a BYTE-IDENTICAL json "
        "(sha256 1f88d6eb2e75a45a5c3660ba both times), so the drift is purely "
        "cross-version, never run-to-run"),
    "consequence_for_the_archive": (
        "a04_keep14_trajectory_ni.py hard-fails if the archived endpoint margins "
        "do not reproduce to 5e-4 pp, which is TIGHTER than this 5.3e-3 pp "
        "cross-version drift. That assertion is therefore only guaranteed on a "
        "node whose numpy matches the one that produced the archive. This is a "
        "latent tooling hard-fail, NOT scientific drift. Fixing it means pinning "
        "numpy across the cluster."),
    "NOT_an_excuse": (
        "this may NOT be cited to explain away any move larger than ~0.006 pp. "
        "Every resolved interval reported here is 15-200x larger than the drift, "
        "and the drift does not touch popqa or mmlu_content at all."),
}

NEW_ARM_INDEX_BASE = 400          # disjoint from 0,1 / 100..102 / 200..203 / 300..301
GUARD_SEED_OFF = 1700             # disjoint from 700 (keep14) and 900
INTERVAL_SEED_OFF = 1900


def _arm_name(arm_key, step):
    return f"{arm_key}_step{step}"


def _seed_off(arm_index, axis):
    """Same functional form as every archived cell: 97*arm_index + 13*axis."""
    return 97 * arm_index + 13 * AXES.index(axis)


def _tag_dirs(tag_prefix, step):
    return {"mmlu": f"{tag_prefix}_step{step}",
            "cb": f"{tag_prefix}_step{step}",
            "nq": f"{tag_prefix}_step{step}_nqopen"}


# ---------------------------------------------------------------------------
# protocol, confirmed from the INVOCATION (summary.json records neither
# batch_size nor chat_template -- A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md)
# ---------------------------------------------------------------------------
def protocol_asserted(raw_root, driver_logs, driver_path, arm_specs_by_step):
    """Confirm batch_size and chat_template FROM THE INVOCATION. FAIL CLOSED.

    Identical mechanism to `a04_keep14_trajectory_ni.protocol_asserted`, and it
    exists for the same reason: the harness writes
    `mode / keep_front_layers / n_fresh_layers / num_hidden_layers / ckpt_step /
    ckpt / base_model / add_bos / max_new_tokens` into `summary.json:meta` and
    NEITHER `batch_size` NOR `chat_template`, so the two most decision-critical
    fields are knowable only from the invocation. Batch size is not free:
    `full32_rescore_v2_20260812.sensitivity_bs48_probe` measured bs32->bs48
    flipping 12/14267 popqa and 10/3610 nq_open items.

    The driver's own echoed lines are the evidence (it prints the variables it
    passes to the harness); grepping this script's source would not be. The
    driver source defaults are read as corroboration only. Any deviation raises
    and NO output file is written.

    chat_template is asserted STRUCTURALLY: neither harness contains a
    chat-template code path (the only occurrence of the string in either file is
    a docstring), so the protocol cannot have been switched on. `add_bos` IS in
    the artefacts and is asserted with `is False` -- never `is not False`'s evil
    twin `is not True`, which passes silently on None.
    """
    import re
    frozen = {"cb_bs": 32, "mmlu_bs": 16}
    out = {
        "frozen_expectation": frozen,
        "why_bs_is_not_free": (
            "full32_rescore_v2_20260812.sensitivity_bs48_probe: bs32->bs48 "
            "flipped 12/14267 popqa and 10/3610 nq_open items (bf16 numerics "
            "depend on left-pad width)"),
        "artefact_gap_acknowledged": (
            "summary.json:meta records neither batch_size nor chat_template "
            "(A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md), so both are confirmed "
            "from the INVOCATION, not inferred from the result dirs"),
        "from_driver_logs": {}, "from_driver_source": {},
        "add_bos_from_summaries": {}, "max_new_tokens_from_summaries": {},
        "chat_template": {},
    }

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
            raise SystemExit(
                f"FATAL: no 'DRIVER START ... mmlu_bs=.. cb_bs=..' line in {p}")
        per_axis = {}
        for kind in ("closedbook", "nq_open", "mmlu"):
            mm = re.findall(rf"{kind} START \S+ bs=(\d+)", txt)
            per_axis[kind] = sorted({int(x) for x in mm})
        rec = {"log": lg,
               "header_mmlu_bs": int(hdr.group(1)),
               "header_cb_bs": int(hdr.group(2)),
               "per_axis_bs_echoed": per_axis}
        if rec["header_cb_bs"] != frozen["cb_bs"] \
                or rec["header_mmlu_bs"] != frozen["mmlu_bs"]:
            raise SystemExit(f"FATAL protocol deviation in {p}: {rec} != {frozen}")
        for kind, want in (("closedbook", frozen["cb_bs"]),
                           ("nq_open", frozen["cb_bs"]),
                           ("mmlu", frozen["mmlu_bs"])):
            # a SKIPped axis echoes no START line; an empty list is only OK if
            # every cell for that axis was already present, which the shard
            # integrity check verifies separately. A WRONG value is fatal.
            if per_axis[kind] and per_axis[kind] != [want]:
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
    else:
        raise SystemExit(f"FATAL: driver source {dp} absent")

    for label, spec in arm_specs_by_step.items():
        for key, root in (("cb", "olmo2_closedbook_results"),
                          ("nq", "olmo2_closedbook_results"),
                          ("mmlu", "olmo2_mmlu_content_results")):
            if not spec.get(key):
                continue
            sp = os.path.join(raw_root, root, spec[key], "summary.json")
            if not os.path.isfile(sp):
                raise SystemExit(f"FATAL: {sp} absent")
            meta = json.load(open(sp)).get("meta", {})
            ab = meta["add_bos"]            # KeyError = loud, desired
            if ab is not False:             # `is False`, never `is not True`
                raise SystemExit(
                    f"FATAL {sp}: add_bos={ab!r}; base protocol requires False. "
                    "(Asserted with `is False`; `is not True` would pass on None.)")
            out["add_bos_from_summaries"][f"{label}|{key}"] = False
            if key != "mmlu":
                mnt = meta["max_new_tokens"]
                if int(mnt) != 32:
                    raise SystemExit(
                        f"FATAL {sp}: max_new_tokens={mnt!r} != 32")
                out["max_new_tokens_from_summaries"][f"{label}|{key}"] = int(mnt)

    out["chat_template"] = {
        "value": False,
        "how_established": (
            "STRUCTURAL, not a flag: neither scripts/eval_olmo2_closedbook_qa.py "
            "nor scripts/eval_olmo2_mmlu_content.py contains a chat-template "
            "code path -- the only occurrence of the string in either file is a "
            "docstring line. A protocol that cannot be switched on cannot have "
            "been switched on."),
        "assertion_form_note": (
            "add_bos is asserted with `is False`, never `is not True`, because "
            "`is not True` passes silently on None."),
        "why_it_must_be_False": (
            "OLMo-2 is a BASE LM with no SFT/RL; a chat template would be "
            "unfair AND would break comparability with every existing cell."),
    }
    return out


def shard_integrity_report(mm_root, cb_root, specs):
    """EXPLICIT shard evidence: the index SET must be exactly {0..7}.

    `load_shards` already asserts, but the dispatch requires the assertion
    RESULT to be inspectable rather than merely to have not raised. Checking the
    index set (not the file COUNT) is the point: this repo has been corrupted by
    a silently merged 5-of-8 set, and a zwfy6-resident arm exists that is
    merged-WITHOUT-shards (`shortgpt16_step200k`) and would pass a count check
    on the merged file alone.
    """
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
    """Guard D1-D6 on the intact anchor, exactly as the archived analyses do.

    D1-D4 are properties of the ANCHOR (and the null) so they are identical to
    the archived cells by construction. D6 depends on `p_disc` -- the fraction of
    items where an ARM differs from the anchor -- so it is RECOMPUTED over this
    arm set rather than inherited: a different arm set could change which cells
    are certifiable, and inheriting would be assuming the answer.
    """
    iv = data["intact_7B_base"][axis]
    nv = (nulls["mmlu_content"]["vectors"][conv]
          if axis == "mmlu_content" else nulls[axis]["vector"])
    d = np.asarray(iv, float) - np.asarray(nv, float)
    resid = float(d.mean())
    resid_pp = 100.0 * resid
    _m, lo, hi, p = paired_bootstrap(
        d, seed=SEED + GUARD_SEED_OFF + 13 * AXES.index(axis))
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


def adjacent_interval_tests(data, axis, arm_key, steps, seed_base):
    """Paired item bootstrap on every ADJACENT checkpoint pair.

    A sign is not a finding. Each pair gets its own paired bootstrap on the
    per-item difference vector (imported `paired_bootstrap`: two-sided 95% CI +
    bootstrap p). These are two DIFFERENT models, so this is NOT the
    harness-jitter question that `full32_rescore_v2_20260812
    .correction_to_the_jitter_premise` settled (same-code re-runs on a FIXED
    ckpt are BIT-IDENTICAL). The question here is whether the ITEM SAMPLE
    resolves the difference between the two models.

    Conservative AND of the two criteria (CI excludes zero AND p<0.05): the
    bootstrap of a 0/1 metric is DISCRETE, so a percentile can land exactly on
    zero while p = 0.0514. Picking the favourable criterion turns a tie into a
    result, so disagreement is surfaced and read as NOT resolved.
    """
    out = {}
    for pi in range(len(steps) - 1):
        a = _arm_name(arm_key, steps[pi])
        b = _arm_name(arm_key, steps[pi + 1])
        d = (np.asarray(data[b][axis], float)
             - np.asarray(data[a][axis], float))
        mean, lo, hi, p = paired_bootstrap(
            d, seed=seed_base + 13 * AXES.index(axis) + 7 * pi)
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
                "CI-excludes-zero and p<0.05 disagree; the discrete bootstrap "
                "of a 0/1 metric can place a percentile exactly at zero. "
                "Treated as NOT resolved (conservative AND)."
                if ci_excl != p_sig else None),
            "n_items_improved": n_up,
            "n_items_regressed": n_down,
            "wrong_to_right": n_up,
            "right_to_wrong": n_down,
            "n_items_changed": n_up + n_down,
            "n": int(d.size),
            "step_gap": int(steps[pi + 1] - steps[pi]),
        }
    return out


def range_report(margins_pp, ses_pp, label):
    """THE LEG A PRODUCT: the range, AND whether it beats item noise.

    max-minus-min of k noisy cells is BIASED UPWARD even when the true spread is
    zero: for iid N(0,s), E[range of 3] = 3/sqrt(pi) * s = 1.6926 s. So the
    observed range is reported next to `expected_range_if_pure_noise_pp`
    (= 1.6926 * mean SE for k=3) and the ratio of the two. `range_exceeds_item_noise`
    is the gate the headline must pass before any "measured neighbour gap" may be
    claimed. This is the guard the dispatch explicitly asked for: if the
    bootstrap SE is already larger than the range, the range is NOT a finding.
    """
    m = np.asarray(margins_pp, float)
    se = np.asarray([x for x in ses_pp if x is not None], float)
    k = int(m.size)
    rng = float(m.max() - m.min())
    mean_se = float(se.mean()) if se.size else float("nan")
    exp_rng = EXPECTED_RANGE_OVER_SD.get(k, float("nan")) * mean_se
    return {
        "label": label,
        "k_checkpoints": k,
        "margins_pp": [float(x) for x in m],
        "range_pp": rng,
        "argmax_index": int(np.argmax(m)),
        "argmin_index": int(np.argmin(m)),
        "best_minus_last_pp": float(m.max() - m[-1]),
        "best_is_not_last": bool(int(np.argmax(m)) != k - 1),
        "successive_differences_pp": [float(x) for x in np.diff(m)],
        "monotone_nondecreasing": bool(np.all(np.diff(m) >= 0)),
        "mean_bootstrap_se_pp": mean_se,
        "expected_range_if_pure_noise_pp": exp_rng,
        "expected_range_formula": (
            "E[range of k iid N(0,s)] * mean_SE; k=3 -> 3/sqrt(pi)=1.69257, "
            "k=2 -> 2/sqrt(pi)=1.12838 (exact for the normal)"),
        "range_over_expected_noise_range": (rng / exp_rng
                                            if exp_rng and exp_rng == exp_rng
                                            else None),
        "range_over_mean_se": (rng / mean_se if mean_se == mean_se and mean_se
                               else None),
        "range_exceeds_item_noise": bool(exp_rng == exp_rng and rng > exp_rng),
        "range_ge_0p5pp": bool(rng >= 0.5),
        "range_lt_0p05pp": bool(rng < 0.05),
        "reading_note": (
            "`range_pp` is the false advantage available to someone who reports "
            "the BEST of these k adjacent checkpoints instead of the last one "
            "(`best_minus_last_pp` is that advantage as actually realised). It "
            "may only be called a measured gap if `range_exceeds_item_noise` is "
            "true; otherwise it is a noise artefact of taking a max-minus-min."),
    }


def output_shape_and_flips(data, axis, arm_key, steps):
    """LABELLED DIAGNOSTIC, never enters a verdict.

    A resolved accuracy DROP could be (a) real knowledge churn, (b) an output
    format/degeneracy shift that costs EM without costing knowledge (guard D4's
    concern, measured per checkpoint), or (c) churn on a handful of items. So
    per checkpoint: empty-prediction rate, mean prediction length,
    most-frequent-constant share, distinct-prediction count; per interval:
    right->wrong / wrong->right and the fraction of prediction STRINGS unchanged.
    Generative axes only -- MMLU-content is a scored-option interface.
    """
    out = {"per_checkpoint": {}, "per_interval": {}}
    for st in steps:
        rows = data[_arm_name(arm_key, st)].get(f"_{axis}_rows")
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
        a = np.asarray(data[_arm_name(arm_key, steps[i])][axis], float)
        b = np.asarray(data[_arm_name(arm_key, steps[i + 1])][axis], float)
        ra = data[_arm_name(arm_key, steps[i])][f"_{axis}_rows"]
        rb = data[_arm_name(arm_key, steps[i + 1])][f"_{axis}_rows"]
        same = sum(1 for x, y in zip(ra, rb)
                   if (x.get("pred") or "").strip() == (y.get("pred") or "").strip())
        r2w = int(((a == 1) & (b == 0)).sum())
        w2r = int(((a == 0) & (b == 1)).sum())
        out["per_interval"][f"{steps[i]}->{steps[i+1]}"] = {
            "right_to_wrong": r2w,
            "wrong_to_right": w2r,
            "net_items": w2r - r2w,
            "net_pp": 100.0 * (w2r - r2w) / a.size,
            "identical_pred_string_frac": same / a.size,
        }
    out["reading_note"] = (
        "Diagnostic only. If empty_pred_frac stays ~0, top_constant_frac stays "
        "low and n_distinct_preds does not collapse, a resolved accuracy move is "
        "NOT an output-degeneracy artefact.")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--legA_tag_prefix", default="A04_7B_keep8f2")
    ap.add_argument("--legB_tag_prefix", default="A04_7B_sg16")
    ap.add_argument("--legA_driver_log", default="logs/a04_nbr_keep8_legA.out")
    ap.add_argument("--legB_driver_log", default="logs/a04_nbr_sg16_legB.out")
    ap.add_argument("--skip_legB", action="store_true",
                    help="publish Leg A alone (used only if Leg B's ckpt cannot "
                         "be made readable; the JSON then records WHY)")
    ap.add_argument("--legB_skip_reason", default="")
    args = ap.parse_args()

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    # ---- assemble the arm set -------------------------------------------
    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    legA_arms, legB_arms = {}, []
    proto_specs = {}
    for cname, c in LEG_A_CLUSTERS.items():
        names = []
        for st in c["steps"]:
            nm = _arm_name(c["arm_key"], st)
            spec = _tag_dirs(args.legA_tag_prefix, st)
            arm_specs[nm] = spec
            proto_specs[f"legA|{st}"] = spec
            names.append(nm)
        legA_arms[cname] = names
    if not args.skip_legB:
        for st in LEG_B["steps"]:
            nm = _arm_name(LEG_B["arm_key"], st)
            spec = _tag_dirs(args.legB_tag_prefix, st)
            arm_specs[nm] = spec
            proto_specs[f"legB|{st}"] = spec
            legB_arms.append(nm)
    arm_names = [a for a in arm_specs if a != "intact_7B_base"]

    # Every driver log listed here must pass the frozen-protocol gate. Leg B ran
    # as TWO invocations (153500+200000 first, then 128000 once its ckpt had been
    # staged across disks), so `--legB_driver_log` accepts a comma-separated list
    # and EVERY entry is checked -- a per-invocation protocol drift between the
    # three Leg B cells would otherwise be invisible.
    driver_logs = {}
    for i, lg in enumerate(
            [x.strip() for x in args.legA_driver_log.split(",") if x.strip()]):
        driver_logs[f"legA_keep8_clusters_{i}" if i else
                    "legA_keep8_clusters"] = lg
    if not args.skip_legB:
        for i, lg in enumerate(
                [x.strip() for x in args.legB_driver_log.split(",") if x.strip()]):
            driver_logs[f"legB_shortgpt16_{i}" if i else "legB_shortgpt16"] = lg

    # 0. PROTOCOL first, before anything is scored. Fails closed.
    proto = protocol_asserted(
        args.raw_root, driver_logs,
        "proposal/active/A04-recovery-certification/code/"
        "a04_neighbour_variability_driver.sh",
        proto_specs)

    # 1. explicit shard integrity BEFORE scoring
    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}

    # 2. guard then NI, per tie convention
    guard, per_conv = {}, {}
    for conv in TIE_CONVS:
        guard[conv] = {axis: guard_cell(data, arm_names, nulls, conv, axis)
                       for axis in AXES}
        cells, retired = [], []
        for arm in arm_names:
            for axis in AXES:
                g = guard[conv][axis]
                if g["classification"] == "NOT_CERTIFIABLE":
                    retired.append({
                        "arm": arm, "axis": axis,
                        "fatal_conditions": g["fatal_conditions"],
                        "ni_run": False,
                        "note": ("NI NOT RUN; excluded from the decision "
                                 "family. Never to be reported as 'NI "
                                 "rejected'.")})
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
                    "arm": arm, "axis": axis,
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
                    **r,
                })
        per_conv[conv] = {
            "intact_residual_pp": {x: guard[conv][x]["residual_intact_pp"]
                                   for x in AXES},
            "delta_pp": {x: guard[conv][x]["delta_pp"] for x in AXES},
            "cells": cells,
            "retired_cells": retired,
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

    # ---- LEG A: the neighbour range, per cluster per axis ----------------
    legA = {}
    for cname, c in LEG_A_CLUSTERS.items():
        names = legA_arms[cname]
        entry = {k: v for k, v in c.items()}
        entry["arm_arch"] = ARM_ARCH[c["arm_key"]]
        entry["per_axis"] = {}
        for axis in AXES:
            cs = [cell("split", nm, axis) for nm in names]
            if any(x is None for x in cs):
                continue
            entry["per_axis"][axis] = {
                "decision_axis": axis not in DEMOTED_AXES,
                "per_step": {str(st): {
                    "acc": cs[i]["reported"],
                    "margin_pp": cs[i]["margin_pp"],
                    "deficit_pp": cs[i]["deficit_pp"],
                    "lo95_pp": cs[i]["diff_lower95_one_sided_pp"],
                    "delta_pp": cs[i]["delta_pp"],
                    "recovery_fraction": cs[i]["residual_fraction_recovered"],
                    "bootstrap_se_pp": cs[i]["bootstrap_se_pp"],
                    "se_to_flip_NI": cs[i]["se_to_flip_NI"],
                    "ni_accept": cs[i]["ni_accept"],
                } for i, st in enumerate(c["steps"])},
                "margin_range": range_report(
                    [x["margin_pp"] for x in cs],
                    [x["bootstrap_se_pp"] for x in cs],
                    f"{cname}|{axis}|margin"),
                "accuracy_range_pp": range_report(
                    [100.0 * x["reported"] for x in cs],
                    [x["bootstrap_se_pp"] for x in cs],
                    f"{cname}|{axis}|acc"),
                "adjacent_interval_paired_tests": adjacent_interval_tests(
                    data, axis, c["arm_key"], c["steps"],
                    SEED + INTERVAL_SEED_OFF),
                "any_accept": bool(any(x["ni_accept"] for x in cs)),
            }
        legA[cname] = entry

    # ---- LEG B: cross-arm replication -----------------------------------
    legB = None
    if not args.skip_legB:
        legB = {k: v for k, v in LEG_B.items()}
        legB["arm_arch"] = ARM_ARCH[LEG_B["arm_key"]]
        legB["per_axis"] = {}
        for axis in AXES:
            cs = [cell("split", _arm_name(LEG_B["arm_key"], st), axis)
                  for st in LEG_B["steps"]]
            if any(x is None for x in cs):
                continue
            tests = adjacent_interval_tests(
                data, axis, LEG_B["arm_key"], LEG_B["steps"],
                SEED + INTERVAL_SEED_OFF + 500)
            legB["per_axis"][axis] = {
                "decision_axis": axis not in DEMOTED_AXES,
                "per_step": {str(st): {
                    "acc": cs[i]["reported"],
                    "margin_pp": cs[i]["margin_pp"],
                    "deficit_pp": cs[i]["deficit_pp"],
                    "recovery_fraction": cs[i]["residual_fraction_recovered"],
                    "bootstrap_se_pp": cs[i]["bootstrap_se_pp"],
                    "ni_accept": cs[i]["ni_accept"],
                } for i, st in enumerate(LEG_B["steps"])},
                "margin_successive_differences_pp": [
                    float(cs[i + 1]["margin_pp"] - cs[i]["margin_pp"])
                    for i in range(len(cs) - 1)],
                "margin_monotone_nondecreasing": bool(all(
                    cs[i + 1]["margin_pp"] >= cs[i]["margin_pp"]
                    for i in range(len(cs) - 1))),
                "adjacent_interval_paired_tests": tests,
                "any_accept": bool(any(x["ni_accept"] for x in cs)),
            }
        # the replication verdict, on popqa's 128000->153500 interval only
        k = LEG_B["replicates_interval"]
        pq = legB["per_axis"].get("popqa", {}).get(
            "adjacent_interval_paired_tests", {}).get(k)
        if pq is not None:
            same_sign = bool(pq["acc_delta_pp"] < 0)
            resolved = bool(pq["distinguishable_from_zero_at_95"])
            legB["replication_verdict"] = {
                "interval": k,
                "keep14_acc_delta_pp": LEG_B["keep14_reference_acc_delta_pp"],
                "shortgpt16_acc_delta_pp": pq["acc_delta_pp"],
                "shortgpt16_ci95_pp": pq["ci95_pp"],
                "shortgpt16_boot_p": pq["boot_p_two_sided"],
                "shortgpt16_flips": {
                    "wrong_to_right": pq["wrong_to_right"],
                    "right_to_wrong": pq["right_to_wrong"]},
                "same_sign_as_keep14": same_sign,
                "resolved_at_95": resolved,
                "REPLICATES": bool(same_sign and resolved),
                "reading": (PREREG_READINGS["leg_b_replicates"]
                            if (same_sign and resolved)
                            else PREREG_READINGS["leg_b_does_not_replicate"]),
                "caveat": (
                    "shortgpt16 is keep_front=16 n_fresh=0 with NON-CONTIGUOUS "
                    "keep_layer_indices [0..12,16,17,31]; keep14fresh2 is 14+2 "
                    "contiguous. Same step number != same data seen (shortgpt16 "
                    "is epoch 2 at 153500 / epoch 3 at 200000). This is a "
                    "replication of a PHENOMENON, not a matched comparison."),
            }
    else:
        legB = {"NOT_RUN": True, "reason": args.legB_skip_reason}
    if isinstance(legB, dict) and not legB.get("NOT_RUN"):
        legB["ckpt_provenance"] = LEG_B_CKPT_PROVENANCE

    # ---- headline -------------------------------------------------------
    clean = [c for c, v in LEG_A_CLUSTERS.items()
             if v["is_clean_neighbour_measurement"]]
    clean_name = clean[0] if clean else list(LEG_A_CLUSTERS)[0]
    dec_ranges = {ax: legA[clean_name]["per_axis"][ax]["margin_range"]
                  for ax in DECISION_AXES
                  if ax in legA[clean_name]["per_axis"]}
    any_real = any(v["range_exceeds_item_noise"] for v in dec_ranges.values())
    any_ge_half = any(v["range_ge_0p5pp"] for v in dec_ranges.values())
    all_tiny = all(v["range_lt_0p05pp"] for v in dec_ranges.values())
    best_not_last = {ax: v["best_is_not_last"] for ax, v in dec_ranges.items()}

    if not any_real:
        headline = "NEIGHBOUR_RANGE_WITHIN_ITEM_NOISE_NOT_A_MEASURED_GAP"
        reading = PREREG_READINGS["leg_a_range_within_item_noise"]
    elif any_ge_half:
        headline = "NEIGHBOUR_RANGE_MATERIAL_GE_0p5PP_TOLERANCE_REQUIRED"
        reading = PREREG_READINGS["leg_a_range_ge_0p5pp"]
    elif all_tiny:
        headline = "NEIGHBOUR_RANGE_NEGLIGIBLE_CLAIM_MUST_NARROW_TO_LONG_RANGE"
        reading = PREREG_READINGS["leg_a_range_lt_0p05pp"]
    else:
        headline = "NEIGHBOUR_RANGE_SMALL_BUT_ABOVE_NOISE_0p05_TO_0p5PP"
        reading = ("the neighbour range is resolvable but sub-0.5pp: a "
                   "hand-picked checkpoint buys less than half a point, so the "
                   "neighbour requirement is real but cheap to satisfy")
    if isinstance(legB, dict) and legB.get("replication_verdict"):
        headline += ("_LEGB_REPLICATES" if legB["replication_verdict"]["REPLICATES"]
                     else "_LEGB_DOES_NOT_REPLICATE")

    out = {
        "gate": "A04_neighbour_variability_and_cross_arm_replication_7B",
        "question": (
            "(A) how large is the NI-margin range across ADJACENT (500-step) "
            "checkpoints of one heal run -- i.e. how much can a hand-picked "
            "checkpoint buy? and (B) does the keep14 popqa 128k->153.5k "
            "regression replicate on an independent arm?"),
        "date": "2026-08-13",
        "headline_verdict": headline,
        "headline_reading": reading,
        "tests_the_claim_from": (
            "A04_KEEP14_TRAJECTORY_NI_VERDICT.md 7.2 / "
            "STATUS.json:keep14_trajectory_ni_20260813 -- 'any future accept "
            "obtained at a hand-picked checkpoint must be shown to survive its "
            "neighbours'. That claim rests on n=1 arm at 25,500-step spacing and "
            "carried NO tolerance."),
        "prereg_readings_fixed_in_advance": PREREG_READINGS,
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
            "guard_G2": ("Delta and anchor never substituted; full32_step25000 "
                         "is FORBIDDEN as an anchor -- it scores below vanilla "
                         "on all four axes, so it would shrink every Delta AND "
                         "lower every target = manufactured accepts"),
        },
        "arm_architectures_are_different": {
            "note": ("keep8fresh2 (10 layers/113 tensors), shortgpt16 (16 "
                     "layers/179 tensors, NON-CONTIGUOUS keep_layer_indices) "
                     "and keep14fresh2 (16 layers/179 tensors, contiguous) are "
                     "THREE DIFFERENT ARCHITECTURES. Absolute margins are never "
                     "tabulated as rungs of one ladder. Leg A is a WITHIN-arm "
                     "range; Leg B replicates a PHENOMENON."),
            "by_arm": ARM_ARCH,
            "step_is_not_synonymous_across_arms": (
                "'step' is an optimizer-step count only. shortgpt16 is epoch 2 "
                "at 153500 and epoch 3 at 200000; keep14fresh2 is epoch 1 at "
                "128000 and epoch 2 at 153500. Same step != same data seen."),
        },
        "protocol_asserted": proto,
        "bootstrap_cross_node_drift": {
            **BOOTSTRAP_CROSS_NODE_DRIFT,
            "numpy_this_run": np.__version__,
        },
        "bootstrap_offsets": {
            "arm_index": arm_index,
            "form": "97*arm_index + 13*axis_index",
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
            "interval_seed_offset": (
                f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair (legA), "
                f"SEED+{INTERVAL_SEED_OFF + 500}+13*axis+7*pair (legB)"),
            "disjoint_from": (
                "pilot_zero arm_index {0,1}; step100k 100..102; shallow_rung "
                "200..203; keep14 trajectory 300..301 + endpoint 201. Guard "
                "offsets 700 (keep14) and 900 (keep14 intervals) also avoided."),
        },
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "per_convention": per_conv,
        "leg_A_neighbour_variability": legA,
        "leg_A_clean_cluster": clean_name,
        "leg_A_decision_axis_margin_ranges_clean_cluster": {
            ax: {"range_pp": v["range_pp"],
                 "expected_range_if_pure_noise_pp":
                     v["expected_range_if_pure_noise_pp"],
                 "range_exceeds_item_noise": v["range_exceeds_item_noise"],
                 "best_is_not_last": v["best_is_not_last"],
                 "best_minus_last_pp": v["best_minus_last_pp"]}
            for ax, v in dec_ranges.items()},
        "leg_A_best_checkpoint_is_not_the_last": best_not_last,
        "leg_B_cross_arm_replication": legB,
        "output_shape_and_flips_diagnostic": {},
        "NOT_licensed": [
            "ANY statement of the form 'the 7B deficit is large relative to "
            "seed variance'. sd_run is a 1B-only quantity (S=3, keep12@5000). "
            "EVERY 7B rung has exactly ONE seed and the historical 7B ladder's "
            "seeds are UNRECORDED (--seed postdates them; trainer afdfa66 called "
            "no seeding function), so NO 7B sd_run is computable or "
            "retrospectively reconstructible.",
            "Treating the checkpoints within a cluster as REPLICATES of each "
            "other. They are successive states of ONE optimisation; their "
            "spread is heal progress + data order, not independent-run "
            "variance. The range measured here is a CHECKPOINT-SELECTION "
            "quantity, NOT an estimate of seed variance.",
            "Calling any of this 'harness noise'. "
            "full32_rescore_v2_20260812.correction_to_the_jitter_premise "
            "established there is NO measured runtime-jitter floor on this "
            "harness (same-code re-runs are BIT-IDENTICAL). These are different "
            "models, so bit-identity does not apply -- but it also removes "
            "'noise' as an available explanation for a model-to-model move. "
            "Item-sampling variability is a different thing and IS quantified.",
            "Comparing keep8fresh2 / shortgpt16 / keep14fresh2 margins as rungs "
            "of one ladder. Three different architectures; two corpora and "
            "unequal steps (STATUS.json:warning) still apply.",
            "Any K1/K2/K3 clause: they are defined over the pre-registered 1B "
            "arm set and a 7B ladder cannot fire them.",
            "Reading cluster 1 (124000/124500/125000) as an uninterrupted "
            "500-step neighbourhood. It straddles a process/resume boundary "
            "with a loader restart.",
            "Quoting any margin here to better than 0.01 pp ACROSS NODES. "
            "numpy's multinomial sampler changed between 2.4.6 (.82, which "
            "produced this file) and 2.5.1 (.73), moving 3 of 24 triviaqa cells "
            "by up to 0.005294 pp (see bootstrap_cross_node_drift). Within a "
            "node the output is byte-identical. This drift is 211x smaller than "
            "the Leg A finding and changes no verdict, and it may NOT be used to "
            "explain away any larger move.",
        ],
        "gpu_note": ("CPU-only analysis. The GPU cost was the 4-axis scoring of "
                     "the cluster + Leg B checkpoints; this script loads only "
                     "per-example shards."),
    }

    for arm_key, steps in [(LEG_A_CLUSTERS[c]["arm_key"], LEG_A_CLUSTERS[c]["steps"])
                           for c in LEG_A_CLUSTERS] + \
                          ([(LEG_B["arm_key"], LEG_B["steps"])]
                           if not args.skip_legB else []):
        for ax in ("triviaqa", "popqa", "nq_open"):
            r = output_shape_and_flips(data, ax, arm_key, steps)
            if r is not None:
                out["output_shape_and_flips_diagnostic"][
                    f"{arm_key}|{steps[0]}-{steps[-1]}|{ax}"] = r

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---- console --------------------------------------------------------
    print("=" * 108)
    print("PROTOCOL CONFIRMED FROM THE INVOCATION (summary.json records neither "
          "batch_size nor chat_template)")
    print("=" * 108)
    for label, r in proto["from_driver_logs"].items():
        print(f"  {label:<22} {r['log']}  cb_bs={r['header_cb_bs']} "
              f"mmlu_bs={r['header_mmlu_bs']}  per-axis={r['per_axis_bs_echoed']}")
    print(f"  driver source defaults: "
          f"{ {k: v for k, v in proto['from_driver_source'].items() if k != 'note'} }")
    print(f"  add_bos is False on all {len(proto['add_bos_from_summaries'])} "
          "result dirs (asserted with `is False`)")
    print()
    print("=" * 108)
    print("GUARD D1-D6 (`split`)")
    print("=" * 108)
    print(f"{'axis':<14}{'resid_intact':>13}{'Delta':>9}{'n':>8}"
          f"{'p*crit':>10}{'p_disc_max':>12}{'D/hw':>7}  class")
    for axis in AXES:
        g = guard["split"][axis]
        print(f"{axis:<14}{g['residual_intact_pp']:>12.4f}pp"
              f"{g['delta_pp']:>8.4f}{g['n']:>8}"
              f"{g['pstar_crit_7B_recomputed']:>10.4f}"
              f"{g['p_disc_max']:>12.4f}{g['delta_over_hw_worst']:>7.2f}"
              f"  {g['classification']}")
    print()
    print("=" * 108)
    print("LEG A -- NEIGHBOUR VARIABILITY across 500-step-spaced checkpoints")
    print("=" * 108)
    for cname, e in legA.items():
        seam = "RESUME SEAM (not a clean neighbourhood)" if e["resume_seam"] \
            else "clean, single process"
        print(f"\n  {cname}  steps={e['steps']}  spacing={e['spacing_steps']}  "
              f"[{seam}]")
        print(f"  {'axis':<14}{'margins (pp)':<40}{'range':>9}{'E[rng|noise]':>14}"
              f"{'>noise?':>9}{'best!=last':>12}")
        for ax, v in e["per_axis"].items():
            mr = v["margin_range"]
            ms = " ".join(f"{x:+.4f}" for x in mr["margins_pp"])
            print(f"  {ax:<14}{ms:<40}{mr['range_pp']:>9.4f}"
                  f"{mr['expected_range_if_pure_noise_pp']:>14.4f}"
                  f"{str(mr['range_exceeds_item_noise']):>9}"
                  f"{str(mr['best_is_not_last']):>12}"
                  + ("" if v["decision_axis"] else "  (demoted)"))
            for k, t in v["adjacent_interval_paired_tests"].items():
                print(f"  {'':<14}   {k:<18} acc {t['acc_delta_pp']:+.4f}pp "
                      f"CI95[{t['ci95_pp'][0]:+.4f},{t['ci95_pp'][1]:+.4f}] "
                      f"p={t['boot_p_two_sided']:.4f} "
                      f"{'RESOLVED' if t['distinguishable_from_zero_at_95'] else 'within item noise'}"
                      f" (+{t['wrong_to_right']}/-{t['right_to_wrong']})")
    if isinstance(legB, dict) and not legB.get("NOT_RUN"):
        print()
        print("=" * 108)
        print("LEG B -- CROSS-ARM REPLICATION (shortgpt16 at keep14's step numbers)")
        print("=" * 108)
        for ax, v in legB["per_axis"].items():
            print(f"  {ax:<14} margins " + " ".join(
                f"{v['per_step'][str(s)]['margin_pp']:+.4f}"
                for s in LEG_B["steps"])
                + f"   monotone_up={v['margin_monotone_nondecreasing']}"
                + ("" if v["decision_axis"] else "  (demoted)"))
            for k, t in v["adjacent_interval_paired_tests"].items():
                print(f"  {'':<14}   {k:<18} acc {t['acc_delta_pp']:+.4f}pp "
                      f"CI95[{t['ci95_pp'][0]:+.4f},{t['ci95_pp'][1]:+.4f}] "
                      f"p={t['boot_p_two_sided']:.4f} "
                      f"{'RESOLVED' if t['distinguishable_from_zero_at_95'] else 'within item noise'}"
                      f" (+{t['wrong_to_right']}/-{t['right_to_wrong']})")
        rv = legB.get("replication_verdict")
        if rv:
            print(f"\n  REPLICATION on popqa {rv['interval']}: "
                  f"keep14 {rv['keep14_acc_delta_pp']:+.4f}pp vs shortgpt16 "
                  f"{rv['shortgpt16_acc_delta_pp']:+.4f}pp "
                  f"(p={rv['shortgpt16_boot_p']:.4f}) -> "
                  f"REPLICATES={rv['REPLICATES']}")
    else:
        print(f"\nLEG B NOT RUN: {legB.get('reason')}")
    print(f"\nHEADLINE: {headline}")
    print(f"READING : {reading}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
