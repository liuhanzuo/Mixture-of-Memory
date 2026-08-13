#!/usr/bin/env python3
"""A04 — is the NI margin MONOTONE along a damaged arm's dense heal trajectory?

WHAT IS BEING TESTED (a claim, not a model). Pre-registered in
`A04_KEEP12_TRAJECTORY_PREREG.md`, committed 4840c10 BEFORE any keep12
capability number existed.

CLAIM P: the NI margin of a damaged, healing arm wanders NON-MONOTONICALLY
along training, with amplitude comparable to Delta, so a single-point accept is
uninterpretable without its neighbourhood.

P's three existing legs are each defective, and the defects are not cosmetic:
  * keep14 trajectory      -- 3 points, UNEVEN 25 500-step spacing. A 3-point
    series has 2 differences; "non-monotone" there means ONE sign flip.
  * neighbour variability   -- 7 of 8 decision-axis ranges fall INSIDE the noise
    gate E[range of 3 iid N(0,s)] = 1.6926*s. Only 1 of 8 cleared it.
  * full32 trajectory       -- a ZERO-DAMAGE CPT arm. Not a recovery arm at all.
So what P has never had is a genuinely damaged recovery arm on a DENSE, EVENLY
SPACED grid with enough points to fit a TREND. keep12+fresh2 supplies exactly
that: 8 points at exact 5 000-step spacing (130000..165000), all inside ONE
training process (verified from the trainer log before any GPU was spent).

THE FOUR QUESTIONS, and how each is decided MECHANICALLY
-------------------------------------------------------
Q1 monotonicity. Per axis, on the 8-point grid:
     MONOTONE        all 7 successive margin differences share a sign
     MONOTONE_TREND  not strictly monotone but |Spearman rho| >= 0.7 and p<0.05
     WANDER          neither, AND the range clears the noise gate (Q2)
     UNRESOLVED      neither, AND the range is INSIDE the gate -> reported as
                     "no detectable diffusion", explicitly NOT as a finding
   Reported alongside: OLS slope, sign-reversal count, and max|d margin| both in
   pp and as a ratio to that axis's Delta. P's amplitude claim is
   operationalised as max|d margin|/Delta >= 0.25.

Q2 the noise gate. A range is max-minus-min of k noisy cells and is BIASED
   UPWARD by noise even at zero true spread. The keep8 pass used the k=3
   constant 1.6926. THAT CONSTANT IS WRONG FOR 8 POINTS: E[range of 8 iid
   N(0,1)] = 2.8475 (measured), so c_8/c_3 = 1.683: reusing 1.6926 would make the
   floor 40.6% TOO LOW and could manufacture a finding. This script therefore Monte-Carlos the standard-normal
   range for the actual k (n=2e6, fixed seed) and VALIDATES the estimator
   against the closed form c_3 = 1.6925687506432689 before using it. sigma is the
   SAME item-level bootstrap SE used everywhere in A04, SE = (mean - lo95)/1.6449
   from the imported `ni_rule`. No new sigma estimator.

Q3 does keep14's popqa regression replicate? keep14's leg is popqa -0.6729 pp
   over 128000->153500 (p=0.0001). keep12 HAS NO step128000 (rotated away), so
   the interval is NOT step-matched and this script does not pretend it is. Two
   pre-committed reads: (a) the length-matched window 130000->155000 (25 000
   steps, within 2% of keep14's width) as the headline; (b) whether ANY adjacent
   5 000-step popqa interval is a resolved regression. With 7 intervals x 3
   decision axes = 21 tests, ~1 false positive is expected at alpha=0.05, so BH
   is applied and the monotonicity verdict rests on the TREND + the RANGE GATE,
   never on counting resolved intervals.

Q4 an independent neighbour-range replication. {165000, 165500, 166000} is a
   500-step triple inside the same single process, scored on the same protocol,
   giving a second arm's neighbour range to compare against keep8's 1.1202 pp
   triviaqa finding -- using the IDENTICAL k=3 convention so the two are
   directly comparable.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
`qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `bh_reject`, `TIE_CONVS`,
`N_BOOT`, `SEED` from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`,
`assert_aligned`, `d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`,
`Z95_TWO_SIDED`, `D4_*`, `SD_RUN_1B_PP` from `a04_shallow_rung_ni_7b`. No
metric, null, rule, guard or anchor is re-derived. THE NULL IS NEVER
HAND-COMPUTED: MAIN's own subtraction of a recorded null was ~0.5 pp off twice,
which is precisely why `build_nulls` is imported.

ANCHOR AND DELTA ARE NOT SUBSTITUTED (guards G0/G2). Anchor = vanilla
`models/OLMo-2-1124-7B` via the imported `ANCHOR`; `full32_step25000` is
FORBIDDEN as an anchor (it scores BELOW vanilla on all four axes, so
substituting it would shrink every Delta AND lower every target = manufactured
accepts). `Delta = 0.10 * residual(intact)`, imported through the guard.

BOOTSTRAP SEEDS. Archived offsets in use: pilot_zero arm_index {0,1}; step100k
100..102; shallow_rung 200..203; keep14 trajectory 300..301 (+endpoint 201);
neighbour variability 400..408 (guard 1700 / intervals 1900, 2400); AND
full32 trajectory 500..503 (guard 2700 / interval 2900). That last one was
written the SAME DAY at 12:33 and was NOT in any prose disjointness list -- my
first choice of arm_index 500 / guard 2700 / interval 2900 collided with it
EXACTLY, and only the executed check caught it. This script therefore uses
arm_index 600.., guard offset 3700, interval offset 3900, so NO ARCHIVED NUMBER
CAN BE PERTURBED. The assertion is EXECUTED (`assert_seeds_disjoint` reads every
archive's own recorded offsets and raises on intersection), not claimed in prose
-- prose claims of disjointness in this repo have already been wrong once today.

ONE NODE ONLY for statistics. numpy's `Generator.multinomial` differs in 19 of
10 000 rows between 2.5.1 (.73) and 2.4.6 (.82) -- max margin drift 0.005294 pp
(`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` 4.1). Every statistic here is computed
on ONE node and the node + numpy version are recorded in the JSON.

CPU ONLY. Read-only on every input.
"""
from __future__ import annotations

import argparse
import glob as _glob
import json
import os
import platform
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
    bh_reject,
    paired_bootstrap,
)

from a04_shallow_rung_ni_7b import (  # noqa: E402
    ANCHOR,
    D2_RESIDUAL_FLOOR_PP,
    D4_CONSTANT_FRAC,
    D4_TIE_FRAC,
    Z95_TWO_SIDED,
    _load_arm,
    assert_aligned,
    d4_interface_degenerate,
)

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]

ARM_KEY = "keep12fresh2"
ARM_DIR = "olmo2_probe2_7B_keep12fresh2"

# The PRIMARY grid: 8 points, EXACT 5000-step spacing. Uniform spacing is what
# makes a Spearman/OLS trend interpretable, so step124000 (the resume anchor,
# 6000 below 130000) is deliberately EXCLUDED from the trend statistics and
# reported separately. Including it would make the grid non-uniform and let a
# single off-grid point drive the slope.
GRID_STEPS = [130000, 135000, 140000, 145000, 150000, 155000, 160000, 165000]

# Reported, but NOT in the trend fit.
OFFGRID_ANCHOR_STEP = 124000

# Q4: a 500-step triple for an independent neighbour-range replication, using
# the SAME k=3 convention as A04_NEIGHBOUR_VARIABILITY_VERDICT.md 2.3.
Q4_TRIPLE = [165000, 165500, 166000]

ARM_ARCH = {
    "keep12fresh2": {"keep_front": 12, "n_fresh": 2, "num_hidden_layers": 14,
                     "n_tensors": 157, "contiguous_front": True,
                     "arch_meta": "outputs/olmo2_probe2_7B_keep12fresh2/arch_meta.json",
                     "arm_label": "healing_front12+fresh2"},
    "keep14fresh2": {"keep_front": 14, "n_fresh": 2, "num_hidden_layers": 16,
                     "n_tensors": 179, "contiguous_front": True,
                     "note": "the arm that GENERATED claim P; not re-scored here"},
    "keep8fresh2": {"keep_front": 8, "n_fresh": 2, "num_hidden_layers": 10,
                    "n_tensors": 113, "contiguous_front": True,
                    "note": "the arm that supplied the 500-step neighbour range"},
    "shortgpt16": {"keep_front": 16, "n_fresh": 0, "num_hidden_layers": 16,
                   "n_tensors": 179, "contiguous_front": False,
                   "note": "the arm on which keep14's popqa dip FAILED to replicate"},
}

# Checkpoint identity, established BEFORE any GPU was spent (2026-08-13, zwfy6,
# /opt/conda/envs/torch-base/bin/python, torch 2.13.0). BYTE SIZE IS NOT
# IDENTITY on this arm: 130000..166000 ALL share 43 867 049 986 B and only
# step124000 differs (43 867 047 810 B). A sibling in this repo
# (shortgpt16/step128000.pt on zwfy6) is a TRUNCATED zip that `ls -l` cannot
# distinguish from healthy, so identity is proven by content, not by size.
CKPT_IDENTITY = {
    "method": (
        "torch.load(weights_only=False, mmap=True) on every ckpt, then per-"
        "tensor sha256 of lm_head.weight / model.embed_tokens.weight / "
        "model.layers.0.self_attn.q_proj.weight AND the float64 sum of EVERY "
        "parameter. All eleven load (so the zip central directory is intact), "
        "all report keep_front=12 n_fresh=2 num_hidden_layers=14, 157 tensors, "
        "fp32, epoch=1, has_optimizer=True."),
    "why_size_is_not_identity": (
        "steps 130000..166000 all share 43,867,049,986 B; only step124000 "
        "differs at 43,867,047,810 B. And a sibling file in this repo "
        "(outputs/olmo2_probe2_7B_shortgpt16/step128000.pt on zwfy6) is a "
        "TRUNCATED write that is present, non-zero and dated like its siblings "
        "-- `ls -l` cannot distinguish it from healthy. Existence-based "
        "inventories are therefore not evidence of readability."),
    "f64_param_sum_by_step": {
        "124000": 68640.84302537073, "130000": 67443.38205783666,
        "135000": 66041.97777233174, "140000": 65032.45165322399,
        "145000": 63042.5410489878, "150000": 62267.043948394465,
        "155000": 61244.47615847642, "160000": 60618.237828557525,
        "165000": 59721.34368036948, "165500": 59683.79571813383,
        "166000": 59540.192013041116},
    "all_distinct_weights": True,
    "all_epoch_1": (
        "every ckpt meta reports epoch=1, so NO epoch boundary (and hence no "
        "sampler reshuffle) occurs anywhere inside the grid"),
    "driver_reasserts_before_gpu": (
        "the driver re-loads each ckpt and asserts meta step / keep_front / "
        "n_fresh / num_hidden_layers / 157 tensors BEFORE launching 8 GPUs, and "
        "the merge-time assert re-checks summary.json:meta.ckpt_step"),
}

# The seam check, run FIRST because A04_NEIGHBOUR_VARIABILITY_VERDICT.md 1.2
# found a 500-step cluster that straddled a process boundary.
SINGLE_PROCESS_PROVENANCE = {
    "training_log": "logs/olmo2_7B_keep12fresh2_resume200k_v2.log",
    "resume_seam_anywhere_in_124000_to_166000": False,
    "evidence": (
        "the log contains EXACTLY ONE process start: `[seed] set_seed(42) on all "
        "ranks` at 2026-08-08 13:58:02, one `[resume] loading ckpt "
        "outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt (saved at step "
        "124000, has_optimizer=True)`, and one `[resume] sampler.set_epoch(1)`. "
        "Every checkpoint in this dispatch is saved by that single process: grid "
        "saves at log lines 348/626/904/1182/1460/1738/2016/2294 for "
        "130000/135000/140000/145000/150000/155000/160000/165000, plus "
        "2323/2351 for 165500/166000. The log's final line is step 166020."),
    "why_this_matters": (
        "the trainer restores optimizer state and RNG but REBUILDS the loader "
        "(`sampler.set_epoch(epoch); data_iter = iter(loader)`) WITHOUT "
        "fast-forwarding inside the epoch, so an interval that crosses a "
        "process boundary sees a DIFFERENT DATA ORDER than an uninterrupted one. "
        "keep8's cluster 124000/124500/125000 had exactly that defect. This grid "
        "does not: it is one uninterrupted data order end to end, which makes "
        "both the 5000-step trend AND the Q4 500-step triple clean BY "
        "CONSTRUCTION rather than by assumption."),
    "training_config_from_the_same_log": (
        "world_size=8 bs=4 gaccum=4 eff_bs=128 seq_len=2048 lr_fresh=2e-05 "
        "lr_inh=2e-05 max_steps=200000, dataset rows=15491607 from "
        "/dev/shm/dolmino_now15b.npy, fp32 master weights, torch AdamW"),
    "lr_is_decaying_across_the_grid": (
        "lr fell from 7.694e-06 at the resume to 3.26e-06 by step 165900 (cosine "
        "schedule to max_steps=200000). So later grid points are taken at a "
        "SMALLER learning rate -- relevant to interpreting any trend, and NOT "
        "controlled for."),
}

# keep14's reference leg, for Q3. Copied from the archived verdict; the numbers
# are NOT recomputed here (that arm is not re-scored) and are used only as the
# comparison target.
KEEP14_POPQA_REFERENCE = {
    "arm": "keep14fresh2",
    "interval": "128000->153500",
    "width_steps": 25500,
    "acc_delta_pp": -0.6729,
    "ci95_pp": [-0.9252, -0.4206],
    "boot_p": 0.0001,
    "flips": {"wrong_to_right": 122, "right_to_wrong": 218},
    "source": "A04_KEEP14_TRAJECTORY_NI_VERDICT.md 3 / "
              "STATUS.json:keep14_trajectory_ni_20260813",
    "why_not_step_matched_here": (
        "keep12 has NO step128000 -- checkpoint rotation removed it (the "
        "trainer's keep_last_n=3 with keep_steps=[83500,121000,124000,150000,"
        "175000,200000] and milestone_every=5000). The closest LENGTH-matched "
        "window on keep12's grid is 130000->155000 = 25 000 steps, within 2% of "
        "keep14's 25 500. Q3 is therefore a PHENOMENON-level replication, never "
        "a matched pairwise comparison."),
}

# keep8's reference neighbour range, for Q4.
KEEP8_NEIGHBOUR_REFERENCE = {
    "arm": "keep8fresh2",
    "cluster": "130000/130500/131000 (clean, single process)",
    "spacing_steps": 500,
    "triviaqa_range_pp": 1.1202,
    "triviaqa_expected_range_if_pure_noise_pp": 0.6577,
    "triviaqa_range_over_noise": 1.70,
    "other_axes_ranges_pp": {"popqa": 0.2523, "mmlu_content": 0.2208,
                             "nq_open": 0.3324},
    "n_of_8_decision_ranges_clearing_the_gate": 1,
    "source": "A04_NEIGHBOUR_VARIABILITY_VERDICT.md 2.1",
}

# Pre-committed readings, fixed in A04_KEEP12_TRAJECTORY_PREREG.md (commit
# 4840c10) BEFORE the numbers existed. The verdict string is generated
# MECHANICALLY from these; it is not chosen after seeing the table.
PREREG_OUTCOMES = {
    "ge2_axes_monotone_improving_none_wander": (
        "P IS NARROWED HARD. Non-monotonicity would then be a property of "
        "keep14/keep8/full32 and NOT of healing at this damage level. P must be "
        "restated as arm-specific and may NOT be sold as a general "
        "methodological law."),
    "all_axes_unresolved": (
        "P's AMPLITUDE CLAIM IS DEAD ON THIS ARM. The margin does not measurably "
        "move at 5000-step spacing, so 'a single point is uninterpretable' loses "
        "its quantitative basis here and the neighbour precondition must be "
        "re-scoped to the arms/axes where a range was actually measured."),
    "ge1_axis_wander_and_amplitude": (
        "P REPLICATES on a second, independently damaged arm. This is the "
        "outcome that would license promotion to paperD."),
    "popqa_resolved_regression": (
        "Q3 REPLICATES; P's strongest leg becomes cross-arm rather than "
        "single-arm."),
    "popqa_monotone_improving": (
        "Q3 FAILS TO REPLICATE. Combined with the keep8->shortgpt16 replication "
        "failure already on record, the popqa dip becomes keep14-SPECIFIC and P "
        "must be argued from RANGE alone, never from directional regression."),
}

# Thresholds, all fixed in the prereg.
SPEARMAN_RHO_THRESHOLD = 0.7
SPEARMAN_P_THRESHOLD = 0.05
AMPLITUDE_RATIO_THRESHOLD = 0.25     # max|d margin| / Delta
BH_Q = 0.05

# Closed form for k=3, used to VALIDATE the Monte-Carlo estimator (exact for the
# normal: E[range of 3] = 3/sqrt(pi)). k=8 has no such simple closed form, which
# is exactly why the estimator is Monte-Carlo'd and then checked at k=3.
EXPECTED_RANGE_CLOSED_FORM = {2: 1.1283791670955126, 3: 1.6925687506432689}
RANGE_MC_N = 2_000_000
RANGE_MC_SEED = 20260813

# ! COLLISION FOUND AND AVOIDED. My first choice was arm_index 500.., guard
# 2700, interval 2900 -- and the MECHANICAL disjointness check (see
# `assert_seeds_disjoint`) discovered that `a04_full32_trajectory_ni.json`,
# written earlier the SAME DAY (2026-08-13 12:33), had already taken ALL THREE:
# arm_index {500,501,502,503}, guard SEED+2700, interval SEED+2900. The prose
# disjointness lists in the previous passes did not mention it because it did not
# exist when they were written. Had the check not been executed, re-running that
# archive would have produced different numbers for reasons unrelated to science.
# THIS IS WHY THE CHECK IS CODE AND NOT A COMMENT.
NEW_ARM_INDEX_BASE = 600          # disjoint from 0,1 / 100..102 / 200..203 / 300..301 / 400..408 / 500..503
GUARD_SEED_OFF = 3700             # disjoint from 700, 1700, 2700
INTERVAL_SEED_OFF = 3900          # disjoint from 900, 1900, 2400, 2900


def _arm_name(step):
    return f"{ARM_KEY}_step{step}"


def _seed_off(arm_index, axis):
    """Same functional form as every archived cell: 97*arm_index + 13*axis."""
    return 97 * arm_index + 13 * AXES.index(axis)


def _tag_dirs(tag_prefix, step):
    return {"mmlu": f"{tag_prefix}_step{step}",
            "cb": f"{tag_prefix}_step{step}",
            "nq": f"{tag_prefix}_step{step}_nqopen"}


def expected_range_constants(ks, n=RANGE_MC_N, seed=RANGE_MC_SEED):
    """E[max-min] of k iid N(0,1), by Monte Carlo, VALIDATED at k=2,3.

    WHY THIS FUNCTION EXISTS AT ALL. `A04_NEIGHBOUR_VARIABILITY_VERDICT.md` 2.3
    gates every range on `1.6926 * mean_SE`. That constant is E[range of THREE]
    (= 3/sqrt(pi), exact for the normal). It is the RIGHT constant for a 3-point
    cluster and the WRONG one for an 8-point grid: E[range of 8] = 2.8475
    (measured), i.e. 1.683x larger, so reusing 1.6926 on 8 points would make the
    noise floor 40.6% TOO LOW and could turn a pure-noise range into a "finding"
    -- exactly the error that guard was written to prevent, committed in the
    guard's own name.

    k=8 has no convenient closed form, so the expectation is estimated by direct
    simulation and the estimator is then CHECKED against the closed forms at
    k=2 (2/sqrt(pi)) and k=3 (3/sqrt(pi)). If the check fails the run aborts --
    a mis-estimated noise floor is worse than no gate, because it looks rigorous.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for k in sorted(set(ks)):
        acc, done, chunk = 0.0, 0, 200_000
        while done < n:
            m = min(chunk, n - done)
            z = rng.standard_normal((m, k))
            acc += float((z.max(axis=1) - z.min(axis=1)).sum())
            done += m
        out[k] = acc / n
    validation = {}
    for k, exact in EXPECTED_RANGE_CLOSED_FORM.items():
        if k in out:
            rel = abs(out[k] - exact) / exact
            validation[k] = {"mc": out[k], "closed_form": exact,
                             "rel_err": rel, "ok": bool(rel < 0.005)}
            if rel >= 0.005:
                raise SystemExit(
                    f"FATAL: Monte-Carlo E[range of {k}] = {out[k]:.6f} differs "
                    f"from the closed form {exact:.6f} by {100*rel:.3f}% -- the "
                    "noise-floor estimator is not trustworthy, refusing to gate "
                    "ranges with it.")
    return {"constants": out, "n_draws": n, "seed": seed,
            "validation_against_closed_form": validation,
            "why": ("the k=3 constant 1.6926 used by "
                    "A04_NEIGHBOUR_VARIABILITY_VERDICT.md 2.3 is E[range of 3]; "
                    "an 8-point grid needs E[range of 8] = 2.8475, which is "
                    "1.683x larger -- so the k=3 constant would make the floor "
                    "40.6% TOO LOW.")}


def spearman_with_perm_p(x, y, n_perm=20000, seed=0):
    """Spearman rho + an EXACT-style permutation p, with no scipy dependency.

    n=8 gives 8! = 40320 orderings, so the permutation null is enumerable in
    principle; 20000 random permutations of a fixed seed is ample and avoids a
    factorial loop. A t-approximation on n=8 would be the wrong instrument --
    with 8 points the sampling distribution of rho is visibly discrete.

    NOTE ON WHAT THIS P-VALUE IS AND IS NOT: it tests the null that the ORDER of
    the 8 margins is unrelated to step. It does NOT account for the item-level
    uncertainty of each margin (that is what the bootstrap SEs and the noise gate
    do, separately). Neither statistic alone is sufficient; the prereg requires
    both, which is why both are reported.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = x.size

    def _rank(v):
        order = np.argsort(v, kind="mergesort")
        r = np.empty(n, float)
        r[order] = np.arange(1, n + 1, dtype=float)
        # average ranks for ties
        uniq, inv, cnt = np.unique(v, return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            for i, c in enumerate(cnt):
                if c > 1:
                    r[inv == i] = r[inv == i].mean()
        return r

    rx, ry = _rank(x), _rank(y)

    def _rho(a, b):
        a = a - a.mean()
        b = b - b.mean()
        den = np.sqrt((a * a).sum() * (b * b).sum())
        return float((a * b).sum() / den) if den > 0 else float("nan")

    rho = _rho(rx, ry)
    rng = np.random.default_rng(seed)
    perm = np.empty(n_perm)
    for i in range(n_perm):
        perm[i] = _rho(rx, rng.permutation(ry))
    p = float((np.abs(perm) >= abs(rho) - 1e-12).mean())
    p = min(max(p, 1.0 / n_perm), 1.0)
    return {"rho": rho, "p_perm_two_sided": p, "n_perm": n_perm,
            "n_points": int(n), "perm_seed": seed,
            "p_note": ("two-sided permutation p on the ORDER of the margins vs "
                       "step. It does NOT incorporate each margin's item-level "
                       "uncertainty -- the bootstrap SEs and the noise gate do "
                       "that separately, and the prereg requires both.")}


def ols_slope(x, y):
    """Least-squares slope in pp per 1000 steps, plus R^2. Descriptive only."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    xc, yc = x - x.mean(), y - y.mean()
    sxx = float((xc * xc).sum())
    slope = float((xc * yc).sum() / sxx) if sxx > 0 else float("nan")
    inter = float(y.mean() - slope * x.mean())
    pred = inter + slope * x
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float((yc * yc).sum())
    return {"slope_pp_per_step": slope,
            "slope_pp_per_1000_steps": slope * 1000.0,
            "intercept_pp": inter,
            "r_squared": (1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
            "note": "descriptive; a straight line is not a heal model"}


def protocol_asserted(raw_root, driver_logs, driver_path, specs_by_label):
    """Confirm batch_size and chat_template FROM THE INVOCATION. FAIL CLOSED.

    Same mechanism as `a04_keep14_trajectory_ni.protocol_asserted` and
    `a04_neighbour_variability.protocol_asserted`, and it exists for the same
    reason: the harness writes `mode / keep_front_layers / n_fresh_layers /
    num_hidden_layers / ckpt_step / ckpt / base_model / add_bos /
    max_new_tokens` into `summary.json:meta` and NEITHER `batch_size` NOR
    `chat_template` (A04_KEEP14_TRAJECTORY_PROTOCOL_GAP.md). Batch size is not
    free: full32_rescore_v2_20260812.sensitivity_bs48_probe measured bs32->bs48
    flipping 12/14267 popqa and 10/3610 nq_open items.

    THE VALUES ARE RE-VERIFIED FROM THIS DISPATCH'S OWN LOGS, not inherited from
    keep14's. Inheriting would assume the thing being checked.

    The driver's own echoed lines are the evidence (it prints the variables it
    actually passes to the harness); grepping this script's source would not be.
    Driver source defaults are corroboration only. Any deviation raises and NO
    output file is written.

    `chat_template` is asserted STRUCTURALLY: neither harness contains a
    chat-template code path, so the protocol cannot have been switched on.
    `add_bos` IS in the artefacts and is asserted with `is False` -- never
    `is not True`, which passes silently on None.
    """
    import re
    frozen = {"cb_bs": 32, "mmlu_bs": 16}
    out = {
        "frozen_expectation": frozen,
        "re_verified_for_this_arm_not_inherited": (
            "keep14's scan established cb=32/mmlu=16; this gate re-parses THIS "
            "dispatch's driver logs rather than assuming those values carried "
            "over. Inheriting a protocol is assuming the thing being checked."),
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
        "ckpt_step_from_summaries": {},
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
               "per_axis_bs_echoed": per_axis,
               "driver_end_rc": (re.findall(r"DRIVER END rc=(\d+)", txt) or [None])[-1]}
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

    for label, spec in specs_by_label.items():
        want_step = spec.get("_step")
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
            # the ckpt the harness actually loaded must be the step we think it is
            if want_step is not None:
                got = int(meta["ckpt_step"])
                if got != int(want_step):
                    raise SystemExit(
                        f"FATAL {sp}: ckpt_step={got} != requested {want_step} "
                        "-- a result dir is labelled with the wrong step")
                out["ckpt_step_from_summaries"][f"{label}|{key}"] = got
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

    `load_shards` already asserts, but the assertion RESULT must be inspectable
    rather than merely have not raised. Checking the index SET (not the file
    COUNT) is the point: this repo has been corrupted by a silently merged
    5-of-8 set, and a zwfy6-resident arm exists that is merged-WITHOUT-shards
    and would pass a count check on the merged file alone.
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


def adjacent_interval_tests(data, axis, steps, seed_base):
    """Paired item bootstrap on every ADJACENT checkpoint pair.

    A sign is not a finding. Each pair gets its own paired bootstrap on the
    per-item difference vector (imported `paired_bootstrap`: two-sided 95% CI +
    bootstrap p). These are two DIFFERENT models, so this is NOT the
    harness-jitter question that full32_rescore_v2 settled (same-code re-runs on
    a FIXED ckpt are BIT-IDENTICAL). The question is whether the ITEM SAMPLE
    resolves the difference between the two models.

    Conservative AND of the two criteria (CI excludes zero AND p<0.05): the
    bootstrap of a 0/1 metric is DISCRETE, so a percentile can land exactly on
    zero while p = 0.0514. Picking the favourable criterion turns a tie into a
    result, so disagreement is surfaced and read as NOT resolved.
    """
    out = {}
    for pi in range(len(steps) - 1):
        a, b = _arm_name(steps[pi]), _arm_name(steps[pi + 1])
        d = (np.asarray(data[b][axis], float)
             - np.asarray(data[a][axis], float))
        mean, lo, hi, p = paired_bootstrap(
            d, seed=seed_base + 13 * AXES.index(axis) + 7 * pi)
        n_up, n_down = int((d > 0).sum()), int((d < 0).sum())
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


def span_interval_test(data, axis, step_a, step_b, seed):
    """One paired bootstrap over an arbitrary (non-adjacent) span.

    Used for Q3's length-matched window 130000->155000 and for the grid endpoints.
    """
    d = (np.asarray(data[_arm_name(step_b)][axis], float)
         - np.asarray(data[_arm_name(step_a)][axis], float))
    mean, lo, hi, p = paired_bootstrap(d, seed=seed)
    ci_excl = bool(not (lo < 0 < hi))
    p_sig = bool(p < 0.05)
    return {
        "interval": f"{step_a}->{step_b}",
        "width_steps": int(step_b - step_a),
        "acc_delta_pp": 100.0 * float(mean),
        "ci95_pp": [100.0 * lo, 100.0 * hi],
        "boot_p_two_sided": p,
        "ci95_excludes_zero": ci_excl,
        "boot_p_below_0p05": p_sig,
        "distinguishable_from_zero_at_95": bool(ci_excl and p_sig),
        "criteria_disagree": bool(ci_excl != p_sig),
        "wrong_to_right": int((d > 0).sum()),
        "right_to_wrong": int((d < 0).sum()),
        "n": int(d.size),
        "boot_seed": seed,
    }


def range_report(margins_pp, ses_pp, label, k_constants):
    """The range, AND whether it beats item noise -- with the RIGHT constant for k.

    max-minus-min of k noisy cells is BIASED UPWARD even when the true spread is
    zero. The gate is `range > c_k * mean(SE)` where c_k = E[range of k iid
    N(0,1)]. USING c_3 = 1.6926 ON AN 8-POINT GRID WOULD UNDERSTATE THE FLOOR BY
    40.6% TOO LOW (c_8/c_3 = 1.683 measured), which is why `k_constants` is
    indexed by the ACTUAL k.

    `range_exceeds_item_noise` is the gate the verdict must pass before any
    "measured gap" may be claimed. This is the guard that retired 7 of 8 ranges
    in A04_NEIGHBOUR_VARIABILITY_VERDICT.md and it is not being relaxed.
    """
    m = np.asarray(margins_pp, float)
    se = np.asarray([x for x in ses_pp if x is not None], float)
    k = int(m.size)
    rng = float(m.max() - m.min())
    mean_se = float(se.mean()) if se.size else float("nan")
    c_k = k_constants.get(k)
    exp_rng = c_k * mean_se if (c_k is not None and mean_se == mean_se) else float("nan")
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
        "monotone_nonincreasing": bool(np.all(np.diff(m) <= 0)),
        "n_sign_reversals": int(sum(
            1 for i in range(len(np.diff(m)) - 1)
            if np.sign(np.diff(m)[i]) * np.sign(np.diff(m)[i + 1]) < 0)),
        "mean_bootstrap_se_pp": mean_se,
        "expected_range_constant_c_k": c_k,
        "expected_range_if_pure_noise_pp": exp_rng,
        "expected_range_formula": (
            f"E[range of k={k} iid N(0,1)] * mean_SE, with c_k estimated by "
            f"Monte Carlo (n={RANGE_MC_N}) and validated against the closed "
            "forms c_2=1.12838 / c_3=1.69257. NOTE: c_3 (used by the keep8 "
            "500-step pass) is NOT valid for k=8."),
        "range_over_expected_noise_range": (rng / exp_rng
                                            if exp_rng == exp_rng and exp_rng
                                            else None),
        "range_over_mean_se": (rng / mean_se if mean_se == mean_se and mean_se
                               else None),
        "range_exceeds_item_noise": bool(exp_rng == exp_rng and rng > exp_rng),
        "reading_note": (
            "`range_pp` may only be called a MEASURED gap if "
            "`range_exceeds_item_noise` is true; otherwise it is a noise "
            "artefact of taking a max-minus-min over k noisy cells."),
    }


def classify_monotonicity(steps, margins_pp, ses_pp, delta_pp, k_constants,
                          perm_seed):
    """Q1's verdict for one axis, generated MECHANICALLY from the prereg criteria.

    The four labels and their thresholds were fixed in
    A04_KEEP12_TRAJECTORY_PREREG.md 2 (commit 4840c10) BEFORE any number existed:

      MONOTONE       all successive differences share a sign
      MONOTONE_TREND |Spearman rho| >= 0.7 AND p_perm < 0.05
      WANDER         neither, AND the range clears the noise gate
      UNRESOLVED     neither, AND the range is INSIDE the gate

    The order matters and is pre-registered: a strictly monotone series is
    MONOTONE even if its range is inside the noise gate, because monotonicity is
    a statement about ORDER and the gate is a statement about MAGNITUDE. Those
    are different questions and are reported separately -- a MONOTONE axis whose
    range is sub-noise is flagged `order_monotone_but_magnitude_unresolved`,
    which is the honest description and neither over- nor under-claims.
    """
    m = np.asarray(margins_pp, float)
    x = np.asarray(steps, float)
    diffs = np.diff(m)
    rr = range_report(m, ses_pp, "grid|margin", k_constants)
    sp = spearman_with_perm_p(x, m, seed=perm_seed)
    ols = ols_slope(x, m)

    mono_up = bool(np.all(diffs >= 0))
    mono_dn = bool(np.all(diffs <= 0))
    strictly_monotone = bool(mono_up or mono_dn)
    trend = bool(abs(sp["rho"]) >= SPEARMAN_RHO_THRESHOLD
                 and sp["p_perm_two_sided"] < SPEARMAN_P_THRESHOLD)
    clears_gate = bool(rr["range_exceeds_item_noise"])

    max_abs_step = float(np.abs(diffs).max())
    amp_ratio = max_abs_step / delta_pp if delta_pp else None
    amp_meets = bool(amp_ratio is not None
                     and amp_ratio >= AMPLITUDE_RATIO_THRESHOLD)

    if strictly_monotone:
        label = "MONOTONE"
    elif trend:
        label = "MONOTONE_TREND"
    elif clears_gate:
        label = "WANDER"
    else:
        label = "UNRESOLVED"

    return {
        "verdict": label,
        "verdict_criteria_source": ("A04_KEEP12_TRAJECTORY_PREREG.md 2, commit "
                                    "4840c10, fixed before any number existed"),
        "steps": [int(s) for s in steps],
        "margins_pp": [float(v) for v in m],
        "successive_differences_pp": [float(v) for v in diffs],
        "strictly_monotone": strictly_monotone,
        "monotone_direction": ("increasing" if mono_up and not mono_dn else
                               "decreasing" if mono_dn and not mono_up else
                               "flat" if mono_up and mono_dn else None),
        "n_sign_reversals": rr["n_sign_reversals"],
        "spearman": sp,
        "spearman_trend_criterion_met": trend,
        "ols": ols,
        "margin_range": rr,
        "range_clears_noise_gate": clears_gate,
        "max_abs_single_step_change_pp": max_abs_step,
        "delta_pp": delta_pp,
        "max_abs_single_step_over_delta": amp_ratio,
        "amplitude_criterion_met": amp_meets,
        "amplitude_criterion": (
            f"max|d margin| / Delta >= {AMPLITUDE_RATIO_THRESHOLD} "
            "operationalises claim P's 'amplitude comparable to Delta'"),
        "order_monotone_but_magnitude_unresolved": bool(
            strictly_monotone and not clears_gate),
        "reading_note": (
            "monotonicity is a statement about ORDER; the noise gate is a "
            "statement about MAGNITUDE. A strictly monotone series whose range "
            "is inside the gate is real as an ordering and unquantified as a "
            "size, and is labelled MONOTONE with "
            "order_monotone_but_magnitude_unresolved=true."),
    }


# An INDEPENDENT re-score of one of our own checkpoints already exists on disk at
# a DIFFERENT batch size, which makes a real bs-sensitivity measurement free.
#
# `olmo2_closedbook_results/7B_keep12_step124000_v2` and
# `olmo2_mmlu_content_results/7B_keep12_step124000_v2` were written 2026-08-08 by
# `scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh`, which passes
# `--batch_size 8` for BOTH the closed-book and the MMLU-content harness (lines
# 140/165/189). This dispatch re-scores the SAME step124000.pt at cb=32 / mmlu=16.
# Same ckpt file, same harness md5s, same `add_bos=false`, 8/8 shards on both
# sides -- so the ONLY difference is batch size.
#
# Why this matters: A04's only existing batch-size sensitivity number is
# `full32_rescore_v2_20260812.sensitivity_bs48_probe` (bs32->bs48 flipped
# 12/14267 popqa and 10/3610 nq_open). bs8->bs32 is a 4x wider gap and has never
# been measured. If it moves items materially, then EVERY cross-protocol
# comparison in A04 that mixes bs8-era dirs with bs32-era dirs inherits that
# much slop -- which is a fact about the archive, not about this trajectory.
ARCHIVED_BS8_RESCORE = {
    "step": 124000,
    "cb_dir": "7B_keep12_step124000_v2",
    "mmlu_dir": "7B_keep12_step124000_v2",
    "archived_bs": {"closedbook": 8, "mmlu_content": 8},
    "this_dispatch_bs": {"closedbook": 32, "mmlu_content": 16},
    "archived_written": "2026-08-08 03:15-03:19 (.73)",
    "archived_driver": "scripts/_run_olmo2_p24_eval_ladder_prev2_73.sh",
    "what_is_held_identical": (
        "the same physical ckpt (outputs/olmo2_probe2_7B_keep12fresh2/"
        "step124000.pt), the same harness files (md5 2ed41993… / fe4a62db…), "
        "add_bos=false on both sides, 8/8 shards on both sides, greedy decoding. "
        "ONLY batch size differs."),
    "why_it_is_worth_reporting": (
        "A04's only bs-sensitivity datum is bs32->bs48 (12/14267 popqa). "
        "bs8->bs32 is a 4x wider gap and unmeasured. Any A04 comparison that "
        "mixes bs8-era result dirs with bs32-era ones inherits whatever this is."),
}


def archived_bs8_comparison(data, raw_root, mm_root, cb_root, step):
    """Per-ITEM comparison of our step124000 cells against the archived bs=8 ones.

    LABELLED DIAGNOSTIC. It never enters the Q1/Q2/Q3/Q4 verdicts -- every cell
    used by those comes from THIS dispatch's uniform protocol. This block exists
    to price a protocol difference in the ARCHIVE, and it is reported whether the
    answer is comfortable or not.
    """
    spec = {"cb": ARCHIVED_BS8_RESCORE["cb_dir"],
            "mmlu": ARCHIVED_BS8_RESCORE["mmlu_dir"]}
    for key, root in (("cb", cb_root), ("mmlu", mm_root)):
        if not os.path.isdir(os.path.join(root, spec[key])):
            return {"run": False,
                    "reason": f"archived {key} dir {spec[key]} absent on this disk"}
    try:
        integ = shard_integrity_report(mm_root, cb_root, {"archived_bs8": spec})
        arch, _ = _load_arm(mm_root, cb_root, spec)
    except SystemExit as e:
        return {"run": False, "reason": f"archived dirs failed integrity: {e}"}

    # the archived summaries must agree that add_bos was False, else the
    # comparison confounds two protocol axes at once and is not interpretable
    meta_check = {}
    for key, root in (("cb", "olmo2_closedbook_results"),
                      ("mmlu", "olmo2_mmlu_content_results")):
        sp = os.path.join(raw_root, root, spec[key], "summary.json")
        m = json.load(open(sp)).get("meta", {})
        if m.get("add_bos") is not False:
            return {"run": False,
                    "reason": (f"archived {key} add_bos={m.get('add_bos')!r} is not "
                               "False -- comparison would confound two protocol "
                               "axes")}
        if int(m.get("ckpt_step", -1)) != int(step):
            return {"run": False,
                    "reason": (f"archived {key} ckpt_step={m.get('ckpt_step')} != "
                               f"{step}")}
        meta_check[key] = {"add_bos": False, "ckpt_step": int(m["ckpt_step"]),
                           "ckpt": m.get("ckpt")}

    ours = data[_arm_name(step)]
    per_axis = {}
    for axis in AXES:
        if axis not in arch or axis not in ours:
            continue
        a = np.asarray(ours[axis], float)
        b = np.asarray(arch[axis], float)
        if a.size != b.size:
            per_axis[axis] = {"skipped": f"size {a.size} vs {b.size}"}
            continue
        n_dis = int((a != b).sum())
        per_axis[axis] = {
            "n": int(a.size),
            "acc_this_dispatch": float(a.mean()),
            "acc_archived_bs8": float(b.mean()),
            "acc_diff_pp": 100.0 * float(a.mean() - b.mean()),
            "n_item_disagreements": n_dis,
            "frac_item_disagreements": n_dis / a.size,
            "right_in_ours_wrong_in_archived": int(((a == 1) & (b == 0)).sum()),
            "wrong_in_ours_right_in_archived": int(((a == 0) & (b == 1)).sum()),
            "bit_identical": bool(n_dis == 0),
        }
    return {
        "run": True,
        **ARCHIVED_BS8_RESCORE,
        "archived_meta_verified": meta_check,
        "archived_shard_integrity": integ["archived_bs8"],
        "per_axis": per_axis,
        "NOT_used_in_any_verdict": (
            "labelled diagnostic. Every cell in Q1-Q4 comes from THIS dispatch's "
            "uniform cb=32 / mmlu=16 protocol; this block only prices a protocol "
            "difference that exists in the ARCHIVE."),
    }


def selftest_statistics():
    """Executed self-test of the two statistics this verdict's Q1 label depends on.

    A monotonicity verdict is only as good as the rho estimator behind it, and
    this one is hand-rolled (no scipy on these nodes). So it is TESTED, in-process,
    every run, and the results go into the JSON. If any case fails the run aborts
    before a number is published.

    The V-shape case is the important one: a strong symmetric excursion has
    rho = 0 and no trend, so MONOTONE_TREND cannot fire on a wander. If that case
    ever passed the trend criterion, the classifier would be silently mislabelling
    exactly the phenomenon claim P is about.
    """
    x = list(range(8))
    cases = {}

    up = spearman_with_perm_p(x, [1, 2, 3, 4, 5, 6, 7, 8])
    cases["monotone_increasing"] = {
        "rho": up["rho"], "p": up["p_perm_two_sided"],
        "expect_rho": 1.0, "ok": bool(abs(up["rho"] - 1.0) < 1e-12)}

    dn = spearman_with_perm_p(x, [8, 7, 6, 5, 4, 3, 2, 1])
    cases["monotone_decreasing"] = {
        "rho": dn["rho"], "p": dn["p_perm_two_sided"],
        "expect_rho": -1.0, "ok": bool(abs(dn["rho"] + 1.0) < 1e-12)}

    # tie handling, against a hand-computed value (two tied y values)
    ty = [3, 1, 4, 1, 5, 9, 2, 6]
    tr = spearman_with_perm_p(x, ty)
    ry = np.array([4, 1.5, 5, 1.5, 6, 8, 3, 7], float)
    rx = np.arange(1, 9, dtype=float)
    a, b = rx - rx.mean(), ry - ry.mean()
    exact = float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))
    cases["tied_ranks_vs_hand_computed"] = {
        "rho": tr["rho"], "hand_computed": exact,
        "ok": bool(abs(tr["rho"] - exact) < 1e-12),
        "why": "average ranks for ties; a naive ordinal rank would differ"}

    vs = spearman_with_perm_p(x, [5, 3, 1, 0, 0, 1, 3, 5])
    v_trend = bool(abs(vs["rho"]) >= SPEARMAN_RHO_THRESHOLD
                   and vs["p_perm_two_sided"] < SPEARMAN_P_THRESHOLD)
    cases["symmetric_V_wander_must_not_be_a_trend"] = {
        "rho": vs["rho"], "p": vs["p_perm_two_sided"],
        "trend_criterion_met": v_trend, "ok": bool(v_trend is False),
        "why": ("a strong symmetric excursion is exactly the phenomenon claim P "
                "describes; if MONOTONE_TREND fired here the classifier would "
                "mislabel wander as trend")}

    o = ols_slope([0, 1000, 2000], [0.0, 1.0, 2.0])
    cases["ols_on_a_known_line"] = {
        "slope_pp_per_1000_steps": o["slope_pp_per_1000_steps"],
        "r_squared": o["r_squared"],
        "ok": bool(abs(o["slope_pp_per_1000_steps"] - 1.0) < 1e-12
                   and abs(o["r_squared"] - 1.0) < 1e-12)}

    failed = [k for k, v in cases.items() if not v["ok"]]
    if failed:
        raise SystemExit(
            f"FATAL: statistic self-test FAILED on {failed} -- refusing to "
            "publish a monotonicity verdict computed with a broken estimator.")
    return {"all_passed": True, "cases": cases,
            "min_attainable_p": 1.0 / 20000,
            "note": ("run in-process every invocation; a failure aborts before "
                     "any output is written")}


def output_shape_and_flips(data, axis, steps):
    """LABELLED DIAGNOSTIC, never enters a verdict.

    A resolved accuracy move could be (a) real knowledge churn, (b) an output
    format/degeneracy shift that costs EM without costing knowledge, or (c) churn
    on a handful of items. So per checkpoint: empty-prediction rate, mean
    prediction length, most-frequent-constant share, distinct-prediction count;
    per interval: right->wrong / wrong->right and the fraction of prediction
    STRINGS unchanged. Generative axes only -- MMLU-content is a scored-option
    interface with no free-form prediction.
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
        a = np.asarray(data[_arm_name(steps[i])][axis], float)
        b = np.asarray(data[_arm_name(steps[i + 1])][axis], float)
        ra = data[_arm_name(steps[i])][f"_{axis}_rows"]
        rb = data[_arm_name(steps[i + 1])][f"_{axis}_rows"]
        same = sum(1 for x, y in zip(ra, rb)
                   if (x.get("pred") or "").strip() == (y.get("pred") or "").strip())
        out["per_interval"][f"{steps[i]}->{steps[i+1]}"] = {
            "right_to_wrong": int(((a == 1) & (b == 0)).sum()),
            "wrong_to_right": int(((a == 0) & (b == 1)).sum()),
            "identical_pred_string_frac": same / a.size,
        }
    out["reading_note"] = (
        "Diagnostic only. If empty_pred_frac stays ~0, top_constant_frac stays "
        "low and n_distinct_preds does not collapse, a resolved accuracy move is "
        "NOT an output-degeneracy artefact.")
    return out


def assert_seeds_disjoint(evidence_dir, used_arm_indices, used_offsets):
    """EXECUTE the seed-disjointness claim instead of asserting it in prose.

    Every archived A04 json records the bootstrap offsets it used. If this run
    reused one, an archived number could be silently perturbed on a re-run of
    that archive. Prose claims of disjointness have been wrong in this repo
    before, so this reads the archives and raises.

    IT ALREADY EARNED ITS KEEP: it caught that `a04_full32_trajectory_ni.json`
    (written the same day at 12:33, after the previous passes' prose lists were
    authored) had taken arm_index 500..503 with guard 2700 and interval 2900 --
    which was this script's first choice, exactly. Hence 600/3700/3900.

    Defensive about SHAPE, not just content: two evidence files in this directory
    have a JSON LIST at top level (`a04_1b_keep7f2_ppl_trajectory*.json`), so a
    bare `blob.get(...)` raises AttributeError. A crash here would look like a
    code bug and invite someone to delete the check.
    """
    found, skipped = {}, {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json"):
            continue
        p = os.path.join(evidence_dir, fn)
        try:
            blob = json.load(open(p))
        except Exception as e:
            skipped[fn] = f"unreadable: {type(e).__name__}"
            continue
        if not isinstance(blob, dict):
            skipped[fn] = f"top-level {type(blob).__name__}, carries no offsets"
            continue
        bo = blob.get("bootstrap_offsets")
        if bo is None:
            continue
        if not isinstance(bo, dict):
            skipped[fn] = f"bootstrap_offsets is {type(bo).__name__}"
            continue
        ai = bo.get("arm_index")
        idxs = sorted(set(ai.values())) if isinstance(ai, dict) else []
        found[fn] = {"arm_index": idxs,
                     "guard_seed_offset": bo.get("guard_seed_offset"),
                     "interval_seed_offset": bo.get("interval_seed_offset")}
        clash = sorted(set(idxs) & set(used_arm_indices))
        if clash:
            raise SystemExit(
                f"FATAL: arm_index {clash} already used by {fn} -- re-running "
                "that archive would produce different numbers. Choose a "
                "disjoint base.")
    return {"archives_scanned": len(found), "per_archive": found,
            "archives_skipped": skipped,
            "this_run_arm_indices": sorted(used_arm_indices),
            "this_run_offsets": used_offsets,
            "checked_mechanically": True,
            "collision_this_check_actually_caught": (
                "a04_full32_trajectory_ni.json (same day, 12:33) holds arm_index "
                "500..503 + guard SEED+2700 + interval SEED+2900, which was this "
                "script's FIRST choice verbatim. Moved to 600/3700/3900. No prose "
                "list mentioned it because it postdated those lists."),
            "note": ("the archives' own recorded offsets were read and "
                     "intersected with this run's; a clash raises. Guard/interval "
                     "offsets are string-formatted in the archives so they are "
                     "reported for inspection rather than parsed -- they were "
                     "compared by eye and moved clear.")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--tag_prefix", default="A04_7B_keep12f2")
    ap.add_argument("--driver_log", default="logs/a04_keep12_traj_73.out",
                    help="comma-separated; EVERY entry is protocol-gated")
    ap.add_argument("--node_label", required=True,
                    help="which node computed these statistics (numpy matters)")
    ap.add_argument("--node82_ctrl_tag", default="",
                    help="optional: tag prefix of a same-ckpt re-score on the "
                         "other node, for the cross-node SCORING determinism "
                         "control (never mixed into a statistic)")
    ap.add_argument("--node82_ctrl_step", type=int, default=0)
    args = ap.parse_args()

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    all_steps = sorted(set(GRID_STEPS + [OFFGRID_ANCHOR_STEP] + Q4_TRIPLE))

    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    proto_specs = {}
    for st in all_steps:
        nm = _arm_name(st)
        spec = _tag_dirs(args.tag_prefix, st)
        arm_specs[nm] = spec
        proto_specs[f"grid|{st}"] = {**spec, "_step": st}
    arm_names = [a for a in arm_specs if a != "intact_7B_base"]

    driver_logs = {}
    for i, lg in enumerate(
            [x.strip() for x in args.driver_log.split(",") if x.strip()]):
        driver_logs[f"keep12_grid_{i}" if i else "keep12_grid"] = lg

    # ---- 0a. self-test the estimators the verdict depends on -----------
    selftest = selftest_statistics()

    # ---- 0b. seed disjointness, EXECUTED -------------------------------
    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}
    seed_check = assert_seeds_disjoint(
        args.evidence_dir, list(arm_index.values()),
        {"arm_index_base": NEW_ARM_INDEX_BASE,
         "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis",
         "interval_seed_offset": f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair"})

    # ---- 1. PROTOCOL, before anything is scored. Fails closed. ---------
    proto = protocol_asserted(
        args.raw_root, driver_logs,
        "proposal/active/A04-recovery-certification/code/"
        "a04_keep12_trajectory_axes_driver.sh",
        proto_specs)

    # ---- 2. explicit shard integrity BEFORE scoring --------------------
    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    # ---- 3. the noise-floor constants, validated ----------------------
    ks = sorted({len(GRID_STEPS), len(Q4_TRIPLE), 2, 3})
    range_consts = expected_range_constants(ks)
    k_constants = range_consts["constants"]

    # ---- 4. guard then NI, per tie convention -------------------------
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

    # ---- 5. Q1/Q2 -- the 8-point grid, per axis ------------------------
    q1 = {}
    for axis in AXES:
        cs = [cell("split", _arm_name(st), axis) for st in GRID_STEPS]
        if any(x is None for x in cs):
            continue
        delta_pp = guard["split"][axis]["delta_pp"]
        cls = classify_monotonicity(
            GRID_STEPS, [c["margin_pp"] for c in cs],
            [c["bootstrap_se_pp"] for c in cs], delta_pp, k_constants,
            perm_seed=SEED + 4300 + 13 * AXES.index(axis))
        cls["decision_axis"] = axis not in DEMOTED_AXES
        cls["per_step"] = {str(st): {
            "acc": cs[i]["reported"],
            "margin_pp": cs[i]["margin_pp"],
            "deficit_pp": cs[i]["deficit_pp"],
            "lo95_pp": cs[i]["diff_lower95_one_sided_pp"],
            "delta_pp": cs[i]["delta_pp"],
            "recovery_fraction": cs[i]["residual_fraction_recovered"],
            "bootstrap_se_pp": cs[i]["bootstrap_se_pp"],
            "se_to_flip_NI": cs[i]["se_to_flip_NI"],
            "ni_accept": cs[i]["ni_accept"],
        } for i, st in enumerate(GRID_STEPS)}
        cls["accuracy_range"] = range_report(
            [100.0 * c["reported"] for c in cs],
            [c["bootstrap_se_pp"] for c in cs], f"grid|{axis}|acc", k_constants)
        cls["adjacent_interval_paired_tests"] = adjacent_interval_tests(
            data, axis, GRID_STEPS, SEED + INTERVAL_SEED_OFF)
        cls["grid_endpoint_span_test"] = span_interval_test(
            data, axis, GRID_STEPS[0], GRID_STEPS[-1],
            SEED + INTERVAL_SEED_OFF + 3100 + 13 * AXES.index(axis))
        cls["any_accept_on_grid"] = bool(any(c["ni_accept"] for c in cs))
        # the off-grid resume anchor, reported but NOT in the trend
        oc = cell("split", _arm_name(OFFGRID_ANCHOR_STEP), axis)
        cls["offgrid_anchor_step124000"] = {
            "acc": oc["reported"], "margin_pp": oc["margin_pp"],
            "bootstrap_se_pp": oc["bootstrap_se_pp"],
            "ni_accept": oc["ni_accept"],
            "EXCLUDED_FROM_TREND_BECAUSE": (
                "it is 6000 steps below 130000, so including it would make the "
                "grid non-uniform and let one off-grid point drive the slope"),
        }
        q1[axis] = cls

    # ---- 6. BH over the decision-axis adjacent intervals --------------
    keys, pv = [], []
    for axis in DECISION_AXES:
        if axis not in q1:
            continue
        for k, t in q1[axis]["adjacent_interval_paired_tests"].items():
            keys.append(f"{axis}|{k}")
            pv.append(t["boot_p_two_sided"])
    rej, adj, nrej = bh_reject(np.asarray(pv), q=BH_Q) if pv else ([], [], 0)
    bh = {
        "family": "decision-axis adjacent 5000-step intervals on the 8-point grid",
        "n_tests": len(pv),
        "q": BH_Q,
        "n_rejected": int(nrej),
        "per_test": {k: {"p_raw": float(pv[i]), "p_adj_bh": float(adj[i]),
                         "bh_reject": bool(rej[i])}
                     for i, k in enumerate(keys)},
        "why": (f"{len(pv)} tests at alpha=0.05 expect ~{0.05*len(pv):.1f} false "
                "positives under a global null, so a single resolved interval is "
                "not evidence of an excursion. The Q1 verdict rests on the "
                "Spearman trend + the range gate, NOT on counting resolved "
                "intervals -- BH is reported so the reader can see the "
                "difference."),
    }

    # ---- 7. Q3 -- popqa replication ------------------------------------
    q3_span = span_interval_test(data, "popqa", 130000, 155000,
                                 SEED + INTERVAL_SEED_OFF + 3200)
    pq_intervals = q1.get("popqa", {}).get("adjacent_interval_paired_tests", {})
    resolved_regressions = {k: v for k, v in pq_intervals.items()
                            if v["distinguishable_from_zero_at_95"]
                            and v["acc_delta_pp"] < 0}
    resolved_improvements = {k: v for k, v in pq_intervals.items()
                             if v["distinguishable_from_zero_at_95"]
                             and v["acc_delta_pp"] > 0}
    span_is_regression = bool(q3_span["acc_delta_pp"] < 0
                              and q3_span["distinguishable_from_zero_at_95"])
    q3 = {
        "question": ("does keep14's popqa 128000->153500 resolved REGRESSION "
                     "(-0.6729 pp, p=0.0001) replicate on keep12?"),
        "keep14_reference": KEEP14_POPQA_REFERENCE,
        "read_a_length_matched_window": q3_span,
        "read_a_verdict_regression_replicates": span_is_regression,
        "read_b_any_adjacent_resolved_regression": {
            "n_resolved_regressions": len(resolved_regressions),
            "which": sorted(resolved_regressions),
            "n_resolved_improvements": len(resolved_improvements),
            "which_improvements": sorted(resolved_improvements),
            "n_intervals_tested": len(pq_intervals),
            "multiplicity_warning": (
                "with 7 intervals a resolved move somewhere is easier to find "
                "than in keep14's 2; see the BH block"),
        },
        "REPLICATES": bool(span_is_regression or len(resolved_regressions) > 0),
        "reading": None,   # filled below
        "caveat": (
            "keep12fresh2 is 12+2 = 14 layers / 157 tensors; keep14fresh2 is "
            "14+2 = 16 layers / 179 tensors. DIFFERENT DAMAGE LEVELS, and the "
            "interval is length-matched (25000 vs 25500 steps), NOT step-matched "
            "(keep12 has no step128000). This is a replication of a PHENOMENON, "
            "never a matched pairwise comparison. Also: keep12's grid sits at a "
            "DECAYING learning rate (7.694e-06 -> 3.26e-06), which is not "
            "controlled for."),
    }
    q3["reading"] = (PREREG_OUTCOMES["popqa_resolved_regression"]
                     if q3["REPLICATES"]
                     else PREREG_OUTCOMES["popqa_monotone_improving"])

    # ---- 8. Q4 -- the 500-step neighbour triple ------------------------
    q4 = {"question": ("independent replication of the 500-step neighbour range, "
                       "which currently exists on keep8 only"),
          "steps": Q4_TRIPLE, "spacing_steps": 500,
          "k_constant_used": k_constants.get(len(Q4_TRIPLE)),
          "uses_same_convention_as_keep8": (
              "yes -- k=3, so the constant is the SAME 1.6926 the keep8 pass "
              "used, making the two ranges directly comparable"),
          "single_process_no_seam": True,
          "keep8_reference": KEEP8_NEIGHBOUR_REFERENCE,
          "per_axis": {}}
    for axis in AXES:
        cs = [cell("split", _arm_name(st), axis) for st in Q4_TRIPLE]
        if any(x is None for x in cs):
            continue
        q4["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "per_step": {str(st): {"acc": cs[i]["reported"],
                                   "margin_pp": cs[i]["margin_pp"],
                                   "bootstrap_se_pp": cs[i]["bootstrap_se_pp"],
                                   "ni_accept": cs[i]["ni_accept"]}
                         for i, st in enumerate(Q4_TRIPLE)},
            "margin_range": range_report([c["margin_pp"] for c in cs],
                                         [c["bootstrap_se_pp"] for c in cs],
                                         f"q4|{axis}|margin", k_constants),
            "adjacent_interval_paired_tests": adjacent_interval_tests(
                data, axis, Q4_TRIPLE, SEED + INTERVAL_SEED_OFF + 700),
        }
    q4["n_decision_axes_clearing_gate"] = sum(
        1 for a in DECISION_AXES if a in q4["per_axis"]
        and q4["per_axis"][a]["margin_range"]["range_exceeds_item_noise"])

    # ---- 9. cross-node SCORING determinism control --------------------
    node_ctrl = {"run": False}
    if args.node82_ctrl_tag and args.node82_ctrl_step:
        st = args.node82_ctrl_step
        ctrl_spec = _tag_dirs(args.node82_ctrl_tag, st)
        try:
            ctrl_data, _ = _load_arm(mm_root, cb_root, ctrl_spec)
        except SystemExit as e:
            node_ctrl = {"run": False, "error": str(e)}
        else:
            per_axis = {}
            for axis in AXES:
                if axis not in ctrl_data:
                    continue
                a = np.asarray(data[_arm_name(st)][axis], float)
                b = np.asarray(ctrl_data[axis], float)
                per_axis[axis] = {
                    "n": int(a.size),
                    "n_item_disagreements": int((a != b).sum()),
                    "acc_this_node": float(a.mean()),
                    "acc_other_node": float(b.mean()),
                    "acc_diff_pp": 100.0 * float(b.mean() - a.mean()),
                    "bit_identical_per_item": bool((a == b).all()),
                }
            node_ctrl = {
                "run": True,
                "step": st,
                "why": ("A04 has repeatedly asserted that SCORING is "
                        "deterministic and only the ANALYSIS bootstrap drifts "
                        "across numpy versions. That claim had never actually "
                        "been tested for the GPU harness across two nodes. This "
                        "re-scores ONE grid checkpoint on the other node with "
                        "the identical driver and compares PER-ITEM."),
                "nodes": "grid scored on the stats node; control on the other H20",
                "per_axis": per_axis,
                "all_axes_bit_identical": bool(
                    all(v["bit_identical_per_item"] for v in per_axis.values())),
                "NOT_mixed_into_any_statistic": (
                    "this control is reported only; no margin, range, trend or "
                    "interval test in this file uses the control dirs"),
            }

    # ---- 9b. free bs8-vs-bs32 sensitivity from an archived re-score ----
    bs8_cmp = archived_bs8_comparison(
        data, args.raw_root, mm_root, cb_root, OFFGRID_ANCHOR_STEP)

    # ---- 10. the HEADLINE, generated mechanically ----------------------
    dec = [a for a in DECISION_AXES if a in q1]
    verdicts = {a: q1[a]["verdict"] for a in dec}
    n_mono = sum(1 for a in dec if verdicts[a] in ("MONOTONE", "MONOTONE_TREND"))
    n_mono_improving = sum(
        1 for a in dec if verdicts[a] in ("MONOTONE", "MONOTONE_TREND")
        and (q1[a]["monotone_direction"] == "increasing"
             or (q1[a]["monotone_direction"] is None
                 and q1[a]["spearman"]["rho"] > 0)))
    n_wander = sum(1 for a in dec if verdicts[a] == "WANDER")
    n_unresolved = sum(1 for a in dec if verdicts[a] == "UNRESOLVED")
    n_amp = sum(1 for a in dec if q1[a]["amplitude_criterion_met"])
    n_clear_gate = sum(1 for a in dec if q1[a]["range_clears_noise_gate"])

    if n_wander >= 1 and n_amp >= 1:
        headline = "P_REPLICATES_ON_KEEP12_WANDER_WITH_DELTA_SCALE_AMPLITUDE"
        reading = PREREG_OUTCOMES["ge1_axis_wander_and_amplitude"]
    elif n_unresolved == len(dec):
        headline = "P_AMPLITUDE_CLAIM_DEAD_ON_KEEP12_ALL_AXES_WITHIN_ITEM_NOISE"
        reading = PREREG_OUTCOMES["all_axes_unresolved"]
    elif n_mono_improving >= 2 and n_wander == 0:
        headline = "P_NARROWED_KEEP12_MONOTONE_IMPROVING_NOT_A_GENERAL_LAW"
        reading = PREREG_OUTCOMES["ge2_axes_monotone_improving_none_wander"]
    elif n_wander >= 1:
        headline = "P_PARTIAL_WANDER_PRESENT_BUT_AMPLITUDE_BELOW_DELTA_QUARTER"
        reading = ("at least one decision axis wanders above item noise, but no "
                   "axis's largest single-step move reaches 25% of its Delta, so "
                   "the 'amplitude comparable to Delta' half of P is NOT "
                   "supported on this arm")
    else:
        headline = "P_MIXED_NO_PREREG_BRANCH_DOMINATES"
        reading = ("the per-axis verdicts do not fall cleanly into a "
                   "pre-registered branch; see per_axis_verdicts")
    headline += ("_Q3_REPLICATES" if q3["REPLICATES"]
                 else "_Q3_DOES_NOT_REPLICATE")

    out = {
        "gate": "A04_keep12_dense_trajectory_monotonicity_7B",
        "question": (
            "on an 8-point, EXACTLY 5000-step-spaced trajectory of a genuinely "
            "damaged recovery arm (keep12+fresh2), is the NI margin monotone in "
            "heal step, is any wander larger than item noise, and does keep14's "
            "popqa mid-heal regression replicate?"),
        "date": "2026-08-13",
        "headline_verdict": headline,
        "headline_reading": reading,
        "claim_under_test": {
            "name": "P",
            "statement": (
                "the NI margin of a damaged, healing arm wanders "
                "NON-MONOTONICALLY along training with amplitude comparable to "
                "Delta, so a single-point accept is uninterpretable without its "
                "neighbourhood"),
            "why_keep12_is_the_right_test": (
                "P's three existing legs are each defective: keep14 has 3 points "
                "at uneven 25500-step spacing (2 differences, so 'non-monotone' "
                "means ONE sign flip); the neighbour-variability pass had 7 of 8 "
                "decision-axis ranges INSIDE the noise gate; and full32 is a "
                "ZERO-DAMAGE CPT arm, not a recovery arm. keep12+fresh2 is a "
                "genuinely damaged arm on a dense uniform grid inside one "
                "process -- the first setting where monotonicity can be tested "
                "as a TREND rather than as a sign."),
            "keep12_is_a_different_damage_level_from_keep14": (
                "12+2 = 14 layers / 157 tensors vs 14+2 = 16 layers / 179 "
                "tensors. This is a CROSS-ARM test, not extra points on the arm "
                "that generated P."),
        },
        "prereg": {
            "document": "A04_KEEP12_TRAJECTORY_PREREG.md",
            "commit": "4840c10",
            "committed_before_first_number": True,
            "outcomes_fixed_in_advance": PREREG_OUTCOMES,
            "monotone_criteria": {
                "MONOTONE": "all successive margin differences share a sign",
                "MONOTONE_TREND": (f"|Spearman rho| >= {SPEARMAN_RHO_THRESHOLD} "
                                   f"and p_perm < {SPEARMAN_P_THRESHOLD}"),
                "WANDER": "neither, AND range clears the noise gate",
                "UNRESOLVED": "neither, AND range is inside the noise gate",
            },
            "amplitude_criterion": (
                f"max|d margin| / Delta >= {AMPLITUDE_RATIO_THRESHOLD}"),
            "gate_design": "A04_GATE_DESIGN.md 2 (+ 2.0.2 neighbour precondition)",
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
            "nulls_imported_not_hand_computed": (
                "build_nulls is imported. MAIN's own subtraction of a recorded "
                "null was ~0.5 pp off TWICE (A04_FULL32_READING_B_IS_FIRING.md), "
                "so no null, residual or Delta is derived by hand anywhere here."),
            "guard_G2": ("Delta and anchor never substituted; full32_step25000 "
                         "is FORBIDDEN as an anchor -- it scores below vanilla "
                         "on all four axes, so it would shrink every Delta AND "
                         "lower every target = manufactured accepts"),
        },
        "arm": {"key": ARM_KEY, "dir": ARM_DIR, "arch": ARM_ARCH[ARM_KEY],
                "all_arms_for_context": ARM_ARCH},
        "grid": {
            "trend_steps": GRID_STEPS,
            "spacing_steps": 5000,
            "spacing_is_exactly_uniform": True,
            "n_points": len(GRID_STEPS),
            "offgrid_anchor_reported_not_fitted": OFFGRID_ANCHOR_STEP,
            "q4_triple": Q4_TRIPLE,
        },
        "single_process_provenance": SINGLE_PROCESS_PROVENANCE,
        "ckpt_identity": CKPT_IDENTITY,
        "protocol_asserted": proto,
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "noise_floor_constants": range_consts,
        "seed_disjointness_checked": seed_check,
        "statistic_selftest": selftest,
        "compute_environment": {
            "stats_node": args.node_label,
            "numpy": np.__version__,
            "python": platform.python_version(),
            "why_one_node": (
                "numpy's Generator.multinomial differs in 19 of 10000 rows "
                "between 2.5.1 (.73) and 2.4.6 (.82) for the same seed, moving "
                "triviaqa margins by up to 0.005294 pp "
                "(A04_NEIGHBOUR_VARIABILITY_VERDICT.md 4.1). EVERY statistic in "
                "this file was computed on ONE node so no comparison here mixes "
                "samplers."),
        },
        "cross_node_scoring_determinism_control": node_ctrl,
        "archived_bs8_vs_this_bs32_sensitivity": bs8_cmp,
        "bootstrap_offsets": {
            "arm_index": arm_index,
            "form": "97*arm_index + 13*axis_index",
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
            "interval_seed_offset": (
                f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair (grid), "
                f"SEED+{INTERVAL_SEED_OFF+700}+... (Q4), "
                f"SEED+{INTERVAL_SEED_OFF+3100}+... (endpoint spans), "
                f"SEED+{INTERVAL_SEED_OFF+3200} (Q3 window)"),
            "spearman_perm_seed": "SEED+4300+13*axis_index",
            "disjoint_from": (
                "pilot_zero arm_index {0,1}; step100k 100..102; shallow_rung "
                "200..203; keep14 trajectory 300..301 + endpoint 201; neighbour "
                "variability 400..408 with guard 1700 / intervals 1900, 2400; "
                "full32 trajectory 500..503 with guard 2700 / interval 2900 "
                "(written the SAME DAY, 12:33, and found ONLY by the mechanical "
                "check -- my first choice of 500/2700/2900 collided with it "
                "exactly). CHECKED MECHANICALLY -- see seed_disjointness_checked."),
        },
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
        },
        "reported_acc": reported,
        "guard_D1_D6": guard,
        "per_convention": per_conv,
        "Q1_Q2_monotonicity_by_axis": q1,
        "per_axis_verdicts": verdicts,
        "decision_axis_tally": {
            "n_decision_axes": len(dec),
            "n_monotone_or_trend": n_mono,
            "n_monotone_improving": n_mono_improving,
            "n_wander": n_wander,
            "n_unresolved": n_unresolved,
            "n_meeting_amplitude_criterion": n_amp,
            "n_range_clearing_noise_gate": n_clear_gate,
        },
        "bh_over_adjacent_intervals": bh,
        "Q3_popqa_replication": q3,
        "Q4_neighbour_triple": q4,
        "any_checkpoint_accepts": {
            conv: {"n_decision_accepts": sum(
                1 for c in per_conv[conv]["cells"]
                if c["decision_axis"] and c["ni_accept"]),
                   "n_decision_cells": sum(
                1 for c in per_conv[conv]["cells"] if c["decision_axis"])}
            for conv in TIE_CONVS},
        "output_shape_and_flips_diagnostic": {},
        "NOT_licensed": [
            "ANY statement of the form 'the 7B deficit is large relative to "
            "seed variance'. sd_run is a 1B-only quantity (S=3, keep12@5000 at "
            "1B). Every 7B rung here has exactly ONE seed and the historical 7B "
            "ladder's seeds are UNRECORDED, so no 7B sd_run is computable or "
            "retrospectively reconstructible.",
            "Treating the 8 grid checkpoints as REPLICATES of each other. They "
            "are successive states of ONE optimisation at a DECAYING learning "
            "rate (7.694e-06 -> 3.26e-06); their spread is heal progress + data "
            "order + lr schedule, NOT independent-run variance. Everything "
            "measured here is a CHECKPOINT-SELECTION / trajectory quantity.",
            "Calling any of this 'harness noise'. There is no measured "
            "runtime-jitter floor on this harness (same-code re-runs on a FIXED "
            "ckpt are BIT-IDENTICAL). These are different models, so "
            "bit-identity does not apply -- but it also removes 'noise' as an "
            "available explanation. Item-sampling variability is a different "
            "thing and IS quantified.",
            "Using the k=3 noise constant 1.6926 on the 8-point grid. E[range of "
            "8] = 2.8475 is 1.683x larger; the k=3 constant would make the "
            "floor 40.6% too low. The constants used are in noise_floor_constants.",
            "Comparing keep12fresh2 / keep14fresh2 / keep8fresh2 / shortgpt16 "
            "margins as rungs of one ladder. Different architectures; the "
            "two-corpora / unequal-steps STATUS.json:warning still applies.",
            "Reading Q3 as a step-matched comparison. keep12 has no step128000; "
            "the window is LENGTH-matched (25000 vs 25500 steps) only.",
            "Any K1/K2/K3 clause: they are defined over the pre-registered 1B "
            "arm set and a 7B ladder cannot fire them.",
            "Quoting any margin here to better than 0.01 pp ACROSS NODES "
            "(numpy multinomial drift; see compute_environment).",
        ],
        "gpu_note": ("CPU-only analysis. The GPU cost was the 4-axis scoring of "
                     "11 checkpoints; this script loads only per-example shards."),
    }

    for ax in ("triviaqa", "popqa", "nq_open"):
        r = output_shape_and_flips(data, ax, GRID_STEPS)
        if r is not None:
            out["output_shape_and_flips_diagnostic"][f"grid|{ax}"] = r
        r4 = output_shape_and_flips(data, ax, Q4_TRIPLE)
        if r4 is not None:
            out["output_shape_and_flips_diagnostic"][f"q4|{ax}"] = r4

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---- console --------------------------------------------------------
    print("=" * 112)
    print(f"PROTOCOL CONFIRMED FROM THE INVOCATION  (stats node "
          f"{args.node_label}, numpy {np.__version__})")
    print("=" * 112)
    for label, r in proto["from_driver_logs"].items():
        print(f"  {label:<18} {r['log']}  cb_bs={r['header_cb_bs']} "
              f"mmlu_bs={r['header_mmlu_bs']}  per-axis={r['per_axis_bs_echoed']} "
              f"rc={r['driver_end_rc']}")
    print(f"  add_bos is False on all {len(proto['add_bos_from_summaries'])} "
          f"result dirs (`is False`); ckpt_step verified on "
          f"{len(proto['ckpt_step_from_summaries'])} dirs")
    print(f"  noise-floor constants c_k = "
          f"{ {k: round(v, 5) for k, v in k_constants.items()} }  "
          f"(validated vs closed form: "
          f"{ {k: v['ok'] for k, v in range_consts['validation_against_closed_form'].items()} })")
    print()
    print("=" * 112)
    print("GUARD D1-D6 (`split`)")
    print("=" * 112)
    for axis in AXES:
        g = guard["split"][axis]
        print(f"  {axis:<14} resid_intact={g['residual_intact_pp']:>8.4f}pp  "
              f"Delta={g['delta_pp']:>7.4f}  n={g['n']:>6}  "
              f"{g['classification']}")
    print()
    print("=" * 112)
    print("Q1/Q2 -- MONOTONICITY ON THE 8-POINT, 5000-STEP GRID")
    print("=" * 112)
    for axis in AXES:
        if axis not in q1:
            continue
        c = q1[axis]
        mr = c["margin_range"]
        tag = "" if c["decision_axis"] else "  (DEMOTED)"
        print(f"\n  {axis}{tag}   VERDICT = {c['verdict']}")
        print("    margins pp: " + " ".join(f"{v:+.4f}" for v in c["margins_pp"]))
        print("    diffs   pp: " + " ".join(
            f"{v:+.4f}" for v in c["successive_differences_pp"]))
        print(f"    strictly_monotone={c['strictly_monotone']} "
              f"reversals={c['n_sign_reversals']}  "
              f"spearman rho={c['spearman']['rho']:+.4f} "
              f"p={c['spearman']['p_perm_two_sided']:.4f}  "
              f"OLS={c['ols']['slope_pp_per_1000_steps']:+.5f}pp/1k "
              f"R2={c['ols']['r_squared']:.3f}")
        print(f"    range={mr['range_pp']:.4f}pp  "
              f"E[rng|noise]={mr['expected_range_if_pure_noise_pp']:.4f}pp "
              f"(c_{mr['k_checkpoints']}={mr['expected_range_constant_c_k']:.4f})  "
              f"clears_gate={mr['range_exceeds_item_noise']}")
        print(f"    max|d margin|={c['max_abs_single_step_change_pp']:.4f}pp = "
              f"{c['max_abs_single_step_over_delta']:.3f} x Delta "
              f"({c['delta_pp']:.4f}pp)  amplitude_met="
              f"{c['amplitude_criterion_met']}")
        for k, t in c["adjacent_interval_paired_tests"].items():
            flag = "RESOLVED" if t["distinguishable_from_zero_at_95"] else "within item noise"
            print(f"      {k:<18} acc {t['acc_delta_pp']:+.4f}pp "
                  f"CI95[{t['ci95_pp'][0]:+.4f},{t['ci95_pp'][1]:+.4f}] "
                  f"p={t['boot_p_two_sided']:.4f} {flag} "
                  f"(+{t['wrong_to_right']}/-{t['right_to_wrong']})")
    print()
    print("=" * 112)
    print(f"BH over {bh['n_tests']} decision-axis adjacent intervals: "
          f"{bh['n_rejected']} rejected at q={BH_Q}")
    print("=" * 112)
    print(f"\nQ3 popqa replication: length-matched 130000->155000 = "
          f"{q3['read_a_length_matched_window']['acc_delta_pp']:+.4f}pp "
          f"CI95[{q3['read_a_length_matched_window']['ci95_pp'][0]:+.4f},"
          f"{q3['read_a_length_matched_window']['ci95_pp'][1]:+.4f}] "
          f"p={q3['read_a_length_matched_window']['boot_p_two_sided']:.4f}  "
          f"(keep14 was {KEEP14_POPQA_REFERENCE['acc_delta_pp']:+.4f}pp)")
    print(f"  resolved adjacent regressions: "
          f"{q3['read_b_any_adjacent_resolved_regression']['n_resolved_regressions']}"
          f" of {q3['read_b_any_adjacent_resolved_regression']['n_intervals_tested']}"
          f"  -> REPLICATES={q3['REPLICATES']}")
    print(f"\nQ4 500-step triple {Q4_TRIPLE} (k=3, same convention as keep8):")
    for axis in AXES:
        if axis not in q4["per_axis"]:
            continue
        mr = q4["per_axis"][axis]["margin_range"]
        print(f"  {axis:<14} margins " + " ".join(
            f"{v:+.4f}" for v in mr["margins_pp"])
            + f"  range={mr['range_pp']:.4f} "
              f"E[rng|noise]={mr['expected_range_if_pure_noise_pp']:.4f} "
              f"clears={mr['range_exceeds_item_noise']}")
    if node_ctrl.get("run"):
        print(f"\nCROSS-NODE SCORING CONTROL (step {node_ctrl['step']}): "
              f"all_axes_bit_identical={node_ctrl['all_axes_bit_identical']}")
        for ax, v in node_ctrl["per_axis"].items():
            print(f"  {ax:<14} disagreements {v['n_item_disagreements']}/{v['n']}"
                  f"  acc_diff={v['acc_diff_pp']:+.6f}pp")
    if bs8_cmp.get("run"):
        print("\nARCHIVED bs=8 vs THIS bs=32/16 on the SAME step124000 ckpt "
              "(diagnostic, in no verdict):")
        for ax, v in bs8_cmp["per_axis"].items():
            if v.get("skipped"):
                print(f"  {ax:<14} skipped: {v['skipped']}")
                continue
            print(f"  {ax:<14} acc {100*v['acc_this_dispatch']:.4f}% (bs32/16) vs "
                  f"{100*v['acc_archived_bs8']:.4f}% (bs8)  diff="
                  f"{v['acc_diff_pp']:+.4f}pp  items differing "
                  f"{v['n_item_disagreements']}/{v['n']} "
                  f"({100*v['frac_item_disagreements']:.3f}%)")
    else:
        print(f"\nARCHIVED bs=8 comparison NOT run: {bs8_cmp.get('reason')}")
    print(f"\nHEADLINE: {headline}")
    print(f"READING : {reading}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
