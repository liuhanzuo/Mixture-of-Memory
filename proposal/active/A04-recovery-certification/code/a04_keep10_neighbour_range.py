#!/usr/bin/env python3
"""A04 — DOES THE keep8 NEIGHBOUR RANGE REPLICATE ON A SECOND ARM? (keep10+fresh2)

WHAT IS UNDER TEST
------------------
`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` measured the NI-margin range across three
checkpoints 500 steps apart on ONE arm (keep8+fresh2) and produced SIX ranges, of
which EXACTLY ONE crossed the item-noise gate: triviaqa in the clean cluster,
1.1202 pp = 1.70x `E[range | pure noise]`. `A04_GATE_DESIGN.md` 2.0.2 (the
neighbour precondition on any reported accept) and 2.5 (the per-axis tolerance)
both rest on that single cell, and 2.5 says so in as many words:

    "These are one-arm numbers and should be widened if a second arm is ever
     measured."

This is that second arm: `outputs/olmo2_probe2_7B_keep10fresh2/`
step89000/89500/90000, a 500-step triple that had never been scored on any axis,
on a THIRD damage level (keep_front=10 => 12 layers; keep8 => 10, keep12 => 14,
keep14 => 16).

PRE-REGISTERED READINGS, fixed in `A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md` and
committed BEFORE the first number existed (they are also in PREREG_READINGS
below, which is under git with the same timestamp):

  * keep10 triviaqa range > gate AND >= 0.5 pp -> the pp-scale adjacent-500-step
    move is ARM-INDEPENDENT; 2.5's tolerance is corroborated on a second,
    differently-damaged arm.
  * > gate but < 0.5 pp -> replicates in KIND, not in SIZE; 2.5 stays
    keep8-specific.
  * <= gate -> keep8's 1.1202 pp is an ISOLATED CELL. 2.0.2 keeps its logical
    force (one counterexample still makes single-checkpoint accepts unsafe) but
    LOSES its claim to generality, and 2.5's numbers must be labelled
    keep8-only-not-reproduced. THIS WEAKENS OUR POSITION AND IS WRITTEN THAT WAY.

A non-replication is NOT permitted to be spun as "consistent with noise, so no
problem". The asymmetry is deliberate and was fixed in advance.

Q3 — THE SAME DATA IN HEINEMAN ET AL.'S UNIT (arXiv:2508.13144, NeurIPS 2025
Spotlight, OpenReview sAFottNlra; DBLP has CoRR only, so DBLP alone would misread
it as a preprint). They define benchmark noise as the RELATIVE STANDARD DEVIATION
of the accuracy over the final n intermediate checkpoints,
    Rel.Std.(m) = sqrt( sum_i (m_i - mbar)^2 / (n-1) ) / mbar,
and publish per-task values for OLMo-2 1.5B/7B/13B/32B over the final THIRTY
checkpoints at 1000-step spacing (their Table 4). The 7B-4T values relevant here:
TriviaQA 0.003, MMLU 0.023.

  ! HARD CONSTRAINT, imposed by the dispatch and enforced structurally below.
    Ours is n=3 at 500-step spacing on a DAMAGED arm under this repo's base
    protocol (cb_bs=32 / add_bos=False / max_new_tokens=32); theirs is n=30 at
    1000-step spacing on an INTACT model under OLMES. Different n, spacing,
    harness, metric convention and model condition. This is a CROSS-PROTOCOL
    HYPOTHESIS, never an equal-footing comparison, and every emitted record
    carries the asymmetry in-band (`comparability`), so a downstream reader
    cannot lift the ratio out of its caveat.
  ! Their "MMLU" is standard LETTER-choice MMLU. Our decision axis is
    `mmlu_content` (content-continuation). At the anchor those two interfaces
    disagree on 40.1% of items (7B_base/summary.json:
    letter_vs_content_norm.agreement = 0.5994), so `letter_acc` is ALSO recorded
    as the interface-matched secondary.
  ! At n=3 the sample SD is itself wildly noisy: for a normal sample, sd has
    relative SD = sqrt(1 - 2/(n-1) * (Gamma(n/2)/Gamma((n-1)/2))^2) = 52.3% at
    n=3 (computed and asserted below, not asserted in prose). So a ~2x rel.std
    ratio is uninformative EVEN BEFORE the protocol mismatch; only an
    order-of-magnitude gap means anything, and the JSON says so per cell.

WHAT IS IMPORTED AND NEVER REIMPLEMENTED
----------------------------------------
`ni_rule`, `ratio_rule`, `build_nulls`, `mmlu_content_norm_vec`, `qa_metric_vec`,
`EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `TIE_CONVS`, `N_BOOT`, `SEED`
from A03's `analyze_1b_knowledge_floor`; `ANCHOR`, `_load_arm`, `assert_aligned`,
`d4_interface_degenerate`, `D2_RESIDUAL_FLOOR_PP`, `Z95_TWO_SIDED` from
`a04_shallow_rung_ni_7b`; and — the point of this file — `range_report`,
`adjacent_interval_tests`, `guard_cell`, `output_shape_and_flips`,
`protocol_asserted`, `shard_integrity_report`, `EXPECTED_RANGE_OVER_SD` are
imported FROM `a04_neighbour_variability` itself, so the gate applied to keep10 is
the SAME CODE OBJECT that gated keep8. A re-implementation would make the
comparison meaningless, and this is exactly the failure mode the dispatch warned
about ("do not derive the margin by subtracting a recorded null" — that shortcut
produced three wrong numbers today, worst case a 3.0x underestimate).

BOOTSTRAP SEEDS. arm_index base 700, guard SEED+4700, intervals SEED+4900 —
mechanically intersected against every archived `bootstrap_offsets` block in
`evidence/` (currently 203 / 300-301 / 400-408 / 500-503; guards 700/1700/2700/
3700; intervals 900/1900/2400/2900/3900). A clash raises.

NODE OF RECORD. `.82`, numpy 2.4.6 — the SAME node and numpy that published the
keep8 Leg A numbers this file compares against, so the comparison is WITHIN one
multinomial sampler. `Generator.multinomial` differs in 19/10000 rows between
2.4.6 and 2.5.1 (max margin drift 0.005294 pp, triviaqa only), which is 10.6x
looser than the 5e-4 pp hard-fail in `a04_keep14_trajectory_ni.py`. Asserted, not
assumed: the run refuses to publish from a different numpy unless explicitly
overridden, and the recorded version goes into the JSON.

CPU ONLY. Read-only on every input.
"""
from __future__ import annotations

import argparse
import json
import math
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

# THE POINT OF THIS FILE: the gate, the interval test and the protocol/shard
# gates are the SAME CODE OBJECTS that produced the keep8 numbers.
from a04_neighbour_variability import (  # noqa: E402
    EXPECTED_RANGE_OVER_SD,
    adjacent_interval_tests,
    guard_cell,
    output_shape_and_flips,
    protocol_asserted,
    range_report,
    shard_integrity_report,
)

DECISION_AXES = [x for x in AXES if x not in DEMOTED_AXES]

ARM_KEY = "keep10fresh2"
ARM_DIR = "olmo2_probe2_7B_keep10fresh2"
STEPS = [89000, 89500, 90000]
SPACING_STEPS = 500

# From outputs/olmo2_probe2_7B_keep10fresh2/arch_meta.json, read before any GPU
# was spent, and re-asserted per checkpoint by the driver against the LOADED meta.
ARM_ARCH = {
    "keep10fresh2": {
        "keep_front": 10, "n_fresh": 2, "num_hidden_layers": 12,
        "contiguous_front": True, "n_params": 3250786304,
        "arch_meta": "outputs/olmo2_probe2_7B_keep10fresh2/arch_meta.json",
        "arm_label_in_meta": "healing_front10+fresh2",
        "lr_fresh": 2e-05, "lr_inherited": 2e-05, "seed_in_meta": 42,
        "note": ("--seed moves only the fresh-tail init, NOT data order "
                 "(DistributedSampler has no seed=), so `seed: 42` is NOT a "
                 "training-seed and no sd_run follows from it."),
    },
    # the arm this replicates, for the reader's convenience only. Its absolute
    # margins are on a DIFFERENT curve and are never tabulated as a rung.
    "keep8fresh2": {"keep_front": 8, "n_fresh": 2, "num_hidden_layers": 10,
                    "note": "the ONE arm A04_NEIGHBOUR_VARIABILITY measured"},
}

# Verified from logs/olmo2_7B_keep10fresh2_resume200k_73.log BEFORE scoring: the
# keep8-cluster-1 seam trap, checked and cleared in advance rather than
# discovered afterwards.
SINGLE_PROCESS_PROVENANCE = {
    "resume_seam": False,
    "is_clean_neighbour_measurement": True,
    "training_log": "logs/olmo2_7B_keep10fresh2_resume200k_73.log",
    "evidence": (
        "the log contains EXACTLY ONE '[resume] loading ckpt ... step86500.pt' "
        "banner (2026-08-12 03:57:09, has_optimizer=True, 135 model tensors "
        "restored, 'continue @ step=86500 epoch=0 warmup=150 max_steps=200000'), "
        "then 'saved ... step89000.pt' 08:44:52 (line 154), 'step89500.pt' "
        "09:42:19 (line 182), 'step90000.pt' 10:39:43 (line 210). The process "
        "died at 11:15 on a TCPStore/NCCL heartbeat error -- AFTER all three "
        "saves. One process, one loader, continuous data order."),
    "why_this_check_exists": (
        "A04_NEIGHBOUR_VARIABILITY 1.2: keep8's cluster 1 (124000/124500/125000) "
        "STRADDLED a process boundary -- 124000/124500 from a .73 run that died, "
        "125000 from a different .82 process four days later. The trainer "
        "restores optimizer+RNG but rebuilds the loader without fast-forwarding "
        "inside the epoch, so a seam-crossing interval saw a different data "
        "order than an uninterrupted 500 steps. That was found only after "
        "scoring; here it is cleared BEFORE."),
    "checkpoint_bytes_all_equal": 39009621855,
    "why_bytes_are_not_identity": (
        "keep8's 130000/130500/131000 also share one size, and "
        "shortgpt16/step128000.pt was a 7.7 GB TRUNCATED write that `ls -l` "
        "could not distinguish from healthy. The driver therefore asserts the "
        "LOADED meta's step/keep_front/n_fresh/depth per checkpoint before "
        "spending 8 GPUs, and the torch.load exercises the zip central "
        "directory."),
}

PREREG_READINGS = {
    "prereg_document": "A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md",
    "replicates_and_material": (
        "keep10 triviaqa range clears the item-noise gate AND is >= 0.5 pp: the "
        "pp-scale adjacent-500-step margin move is ARM-INDEPENDENT, and "
        "A04_GATE_DESIGN 2.5's per-axis tolerance is corroborated on a second, "
        "differently-damaged arm"),
    "replicates_in_kind_only": (
        "clears the gate but is < 0.5 pp: the phenomenon replicates in KIND but "
        "not in SIZE; 2.5's tolerance stays a keep8-specific number and the "
        "general statement must be 'supra-noise but arm-dependent in magnitude'"),
    "does_not_replicate": (
        "keep10's triviaqa range FAILS the gate: keep8's 1.1202 pp is an "
        "ISOLATED CELL. 2.0.2 retains its logical force (one counterexample "
        "still makes single-checkpoint accepts unsafe) but LOSES its claim to "
        "generality, and 2.5's per-axis numbers must be labelled keep8-only / "
        "not reproduced on keep10. This WEAKENS our position and is to be "
        "written exactly that way -- a non-replication may NOT be re-described "
        "as 'consistent with noise, so no problem'"),
}

# keep8's published clean-cluster numbers, for replication comparison. Read from
# the archived JSON at runtime (not hardcoded) -- these literals exist only so a
# mismatch against the archive raises instead of passing silently.
KEEP8_CLEAN_CLUSTER = "cluster2_130000_131000"
KEEP8_EXPECTED_RANGES_PP = {"triviaqa": 1.1202, "popqa": 0.2523,
                            "mmlu_content": 0.2208, "nq_open": 0.3324}
KEEP8_EXPECTED_GATE = {"triviaqa": True, "popqa": False,
                       "mmlu_content": False, "nq_open": False}

# Heineman et al. 2508.13144 Table 4, OLMo-2 7B-4T, final 30 ckpts @1000 steps.
# Extracted in proposal/shared/literature/
# MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md P1 via
# `pdftotext -layout -f 23 -l 24` of v1. PopQA and NQ-open are NOT in their
# suite; those axes get None and MUST be left blank, never filled with a
# neighbouring task.
HEINEMAN = {
    "cite": "Heineman, Hofmann, Magnusson, Gu, Smith, Hajishirzi, Lo, Dodge, "
            "'Signal and Noise: A Framework for Reducing Uncertainty in "
            "Language Model Evaluation', arXiv:2508.13144",
    "venue": "NeurIPS 2025 Spotlight",
    "venue_authority": (
        "OpenReview note sAFottNlra: venue='NeurIPS 2025 spotlight', "
        "venueid=NeurIPS.cc/2025/Conference, invitations include "
        "Submission26329/-/Camera_Ready_Revision. DBLP has CoRR ONLY "
        "(journals/corr/abs-2508-13144), so DBLP or S2 alone would misread this "
        "as a preprint."),
    "noise_definition": (
        "Rel.Std.(m) = sqrt(sum_i (m_i - mbar)^2 / (n-1)) / mbar over the final n "
        "intermediate checkpoints; validated against init-seed / data-order / "
        "whole-curve noise at R^2 = 0.82 / 0.86 / 0.95"),
    "their_n_checkpoints": 30,
    "their_spacing_steps": 1000,
    "their_model_condition": "INTACT OLMo-2 (no pruning, no injury, no heal)",
    "their_protocol": "OLMES / OLMo-2 evaluation setup",
    "their_shot_setting": (
        "FEW-SHOT. Their App. A.1: 'Notably, all tasks use few-shot examples and "
        "we evaluate MCQA benchmarks in both the rank choice (RC) and multiple "
        "choice (MC) setting.' OURS IS STRICT ZERO-SHOT "
        "(scripts/eval_olmo2_closedbook_qa.py docstring line 10: 'Strict "
        "zero-shot, NO retrieval, free-form greedy generation'; line 23: 'no "
        "few-shot exemplars'). This is an ADDITIONAL asymmetry beyond n and "
        "spacing, found by reading their PDF rather than the repo's summary of "
        "it, and it plausibly RAISES our noise relative to theirs independent of "
        "damage -- zero-shot short-form EM on a base LM is a less stabilised "
        "measurement than a few-shot one."),
    "their_triviaqa_metric": (
        "exact match of a short-form generation (their 4.1: 'generative tasks "
        "like TriviaQA or Jeopardy which evaluate the exact match of a "
        "short-form generation'), i.e. the SAME METRIC FAMILY as our triviaqa EM "
        "-- which is why triviaqa is the least unfair of the two comparators"),
    "olmo2_7B_4T_noise": {"triviaqa": 0.003, "mmlu": 0.023,
                          "popqa": None, "nq_open": None},
    "their_mmlu_interface": "standard LETTER-choice MMLU (RC and MC settings)",
    "table4_format_note": (
        "Table 4's cells are 'SNR_{signal/noise}', e.g. TriviaQA 7B-4T reads "
        "'47.03_{0.135/0.003}' => signal 0.135, NOISE 0.003; MMLU reads "
        "'3.39_{0.078/0.023}' => NOISE 0.023. The bare integers are SNR, NOT "
        "noise -- misreading that column would inflate our comparator ~15000x."),
    "extraction_provenance": (
        "INDEPENDENTLY RE-EXTRACTED 2026-08-13 from arXiv:2508.13144v1 by this "
        "agent (`curl` via hy-proxy -> `pdftotext -layout`), NOT copied from the "
        "repo's prior summary: Table 4 p.22 line 'TriviaQA 28.15_{0.411/0.015} "
        "47.03_{0.135/0.003} 60.37_{0.141/0.002} 27.19_{0.064/0.002}' and 'MMLU "
        "14.52_{0.139/0.010} 3.39_{0.078/0.023} 7.51_{0.044/0.006} "
        "5.19_{0.061/0.012}'. Their n and spacing re-verified from the same PDF: "
        "'For noise, we use the final 30 intermediate checkpoints, one "
        "checkpoint for every 1000 training steps until the end of training.' "
        "Their ddof re-verified: 'Rel. Std.(m) = sqrt(1/(n-1) sum_i (m_i - "
        "mbar)^2)/mbar' -- ddof=1, matching `relstd` here. Venue re-verified "
        "live: OpenReview sAFottNlra venueid=NeurIPS.cc/2025/Conference, "
        "venue='NeurIPS 2025 spotlight', Camera_Ready_Revision present. This "
        "matches the repo's earlier extraction in "
        "MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md P1 exactly."),
    "their_prescription": (
        "AVERAGE the final k checkpoints instead of reporting the last one "
        "(+2.4% decision accuracy on a 30-task average); Appendix A.3.2 gives "
        "n=5 for a +-1sigma bound, n=20 for +-0.2sigma"),
    "what_they_do_NOT_do": (
        "intact models only; their decision problem is SUPERIORITY/ranking and "
        "scaling-law extrapolation, never EQUIVALENCE-to-target; no "
        "construct-appropriate best-constant null; and they never ask what a "
        "single-checkpoint ACCEPT is worth, because they have no accept."),
}

NEW_ARM_INDEX_BASE = 700    # disjoint from 0,1 / 100-102 / 200-203 / 300-301 / 400-408 / 500-503 / 600+
GUARD_SEED_OFF = 4700       # disjoint from 700, 1700, 2700, 3700
INTERVAL_SEED_OFF = 4900    # disjoint from 900, 1900, 2400, 2900, 3900

PUBLISH_NUMPY = "2.4.6"     # the node of record for the keep8 comparison


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
def assert_seeds_disjoint(evidence_dir, used_arm_indices, used_offsets):
    """EXECUTE the seed-disjointness claim instead of asserting it in prose.

    Verbatim in mechanism from `a04_keep12_trajectory_monotonicity`, and it is
    here for the reason recorded there: this exact check caught that
    `a04_full32_trajectory_ni.json`, written the same day, had already taken the
    arm_index/guard/interval triple that a later script picked as its FIRST
    choice. A prose disjointness list cannot catch a file that postdates it.

    Defensive about SHAPE: two evidence files here have a JSON LIST at top level
    (`a04_1b_keep7f2_ppl_trajectory*.json`), so a bare `.get` raises
    AttributeError -- which would look like a code bug and invite deletion.
    """
    found, skipped = {}, {}
    for fn in sorted(os.listdir(evidence_dir)):
        if not fn.endswith(".json"):
            continue
        try:
            blob = json.load(open(os.path.join(evidence_dir, fn)))
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
            "this_run_offsets": used_offsets, "checked_mechanically": True}


def sd_rel_uncertainty(n):
    """Relative SD of the SAMPLE SD for a normal sample of size n.

    sd(s)/E[s] where E[s] = c4*sigma, c4 = sqrt(2/(n-1)) * Gamma(n/2)/Gamma((n-1)/2).
    Var(s) = sigma^2 (1 - c4^2), so sd(s)/E[s] = sqrt(1 - c4^2)/c4.

    This is NOT decoration. It is the reason a rel.std comparison at n=3 cannot
    settle anything: at n=3 the estimator's own relative SD is ~52%, so a 2x
    ratio against a published n=30 value is inside the estimator's noise BEFORE
    any protocol difference is considered. Reported per cell so the number cannot
    be quoted without it.
    """
    c4 = math.sqrt(2.0 / (n - 1)) * math.gamma(n / 2.0) / math.gamma((n - 1) / 2.0)
    return {"n": n, "c4_bias_factor": c4,
            "rel_sd_of_sample_sd": math.sqrt(max(0.0, 1.0 - c4 * c4)) / c4,
            "meaning": ("the sample SD over n points has this relative SD "
                        "itself; at n=3 it is ~0.52, so a ~2x rel.std ratio is "
                        "uninformative even before protocol differences")}


def relstd(vals):
    """Heineman et al.'s unit: sd(ddof=1)/mean of the raw ACCURACIES.

    ddof=1 is THEIR definition, re-verified from their PDF:
    `Rel. Std.(m) = sqrt(1/(n-1) sum_i (m_i - mbar)^2) / mbar`.

    `sd_pp` is reported alongside because rel.std has the MEAN IN THE
    DENOMINATOR: a damaged arm at 15% TriviaQA and an intact 7B at ~65% can
    differ in rel.std by 4x with IDENTICAL absolute stability. Anyone comparing
    rel.std across accuracy levels must see the absolute number too.
    """
    a = np.asarray(vals, float)
    m = float(a.mean())
    s = float(a.std(ddof=1))
    return {"n": int(a.size), "values": [float(x) for x in a],
            "mean": m, "sd_ddof1": s, "sd_pp": 100.0 * s,
            "mean_pp": 100.0 * m,
            "rel_std": (s / m) if m else None,
            "ddof": 1,
            "denominator_warning": (
                "rel.std divides by the mean, so a LOWER-accuracy model gets a "
                "HIGHER rel.std at equal absolute stability. Compare sd_pp too.")}


def selftest_estimators():
    """Executed self-test of the two estimators the verdict depends on.

    (1) `range_report`'s noise floor must reproduce the closed form
        E[range of 3 iid N(0,1)] = 3/sqrt(pi) when fed a known SE, because the
        ENTIRE gate is that constant times the mean SE. Tested by construction:
        margins with a known range and SEs with a known mean.
    (2) `relstd` must reproduce a hand-checkable case.
    (3) `sd_rel_uncertainty` must reproduce the textbook c4 values.
    A failing estimator that merely looks rigorous is worse than no gate, so a
    failure aborts before any number is published.
    """
    out = {}
    # (1) a range of exactly 1.0 against a mean SE of exactly 1.0 must give
    # ratio == 3/sqrt(pi)^-1, i.e. expected floor == 1.6925687506432689
    r = range_report([0.0, 0.5, 1.0], [1.0, 1.0, 1.0], "selftest")
    ok1 = (abs(r["range_pp"] - 1.0) < 1e-12
           and abs(r["expected_range_if_pure_noise_pp"]
                   - EXPECTED_RANGE_OVER_SD[3]) < 1e-12
           and r["range_exceeds_item_noise"] is False)
    out["range_report_floor_is_3_over_sqrt_pi"] = {
        "range_pp": r["range_pp"],
        "expected_floor": r["expected_range_if_pure_noise_pp"],
        "closed_form_3_over_sqrt_pi": 3.0 / math.sqrt(math.pi),
        "gate_false_as_expected": r["range_exceeds_item_noise"], "ok": ok1}
    # a range of 2.0 against mean SE 1.0 MUST pass the gate (2 > 1.6926)
    r2 = range_report([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "selftest2")
    ok2 = r2["range_exceeds_item_noise"] is True
    out["range_report_gate_fires_when_it_should"] = {
        "range_pp": r2["range_pp"], "ok": ok2}
    # (2) relstd of [1,2,3]: mean 2, sd_ddof1 = 1 -> 0.5
    rs = relstd([1.0, 2.0, 3.0])
    ok3 = abs(rs["rel_std"] - 0.5) < 1e-12
    out["relstd_hand_case"] = {**rs, "expected": 0.5, "ok": ok3}
    # (3) c4(2)=0.7978845608, c4(3)=0.8862269255 (textbook)
    c2, c3 = sd_rel_uncertainty(2), sd_rel_uncertainty(3)
    ok4 = (abs(c2["c4_bias_factor"] - 0.7978845608028654) < 1e-12
           and abs(c3["c4_bias_factor"] - 0.8862269254527580) < 1e-12)
    out["c4_textbook_values"] = {"c4_n2": c2["c4_bias_factor"],
                                 "c4_n3": c3["c4_bias_factor"], "ok": ok4}
    out["all_ok"] = bool(ok1 and ok2 and ok3 and ok4)
    if not out["all_ok"]:
        raise SystemExit(f"FATAL: estimator self-test failed: {out}")
    return out


def keep8_archive_readback(archive_path):
    """Read keep8's published ranges from its own JSON and CHECK the literals.

    The comparison is only meaningful if keep10 is gated by the same code AND
    compared against the archive's actual numbers. Hardcoding keep8's ranges
    would let the archive drift away silently, so they are read and the
    hardcoded literals are used only as an assertion.
    """
    if not os.path.isfile(archive_path):
        raise SystemExit(
            f"FATAL: keep8 archive {archive_path} absent -- there is nothing to "
            "replicate against, and quoting the 1.1202 pp from prose would be "
            "exactly the 'read the number from a document' error the dispatch "
            "forbids.")
    blob = json.load(open(archive_path))
    cl = blob["leg_A_neighbour_variability"][KEEP8_CLEAN_CLUSTER]
    if cl.get("resume_seam") is not False:
        raise SystemExit(
            f"FATAL: keep8 {KEEP8_CLEAN_CLUSTER} is not the seam-free cluster")
    out = {"archive": os.path.basename(archive_path),
           "cluster": KEEP8_CLEAN_CLUSTER,
           "steps": cl["steps"], "spacing_steps": cl["spacing_steps"],
           "resume_seam": cl["resume_seam"],
           "numpy_that_published_it": (blob.get("bootstrap_cross_node_drift", {})
                                       .get("published_on_node")),
           "per_axis": {}}
    for ax, v in cl["per_axis"].items():
        mr = v["margin_range"]
        out["per_axis"][ax] = {
            "margins_pp": mr["margins_pp"], "range_pp": mr["range_pp"],
            "expected_range_if_pure_noise_pp":
                mr["expected_range_if_pure_noise_pp"],
            "range_exceeds_item_noise": mr["range_exceeds_item_noise"],
            "mean_bootstrap_se_pp": mr["mean_bootstrap_se_pp"],
            "best_minus_last_pp": mr["best_minus_last_pp"],
            "acc_by_step": {s: d["acc"] for s, d in v["per_step"].items()},
        }
        exp_r = KEEP8_EXPECTED_RANGES_PP.get(ax)
        if exp_r is not None and abs(mr["range_pp"] - exp_r) > 5e-4:
            raise SystemExit(
                f"FATAL: keep8 archive {ax} range {mr['range_pp']:.4f} != the "
                f"documented {exp_r} -- the archive moved; resolve before "
                "publishing a replication against it.")
        exp_g = KEEP8_EXPECTED_GATE.get(ax)
        if exp_g is not None and bool(mr["range_exceeds_item_noise"]) != exp_g:
            raise SystemExit(
                f"FATAL: keep8 archive {ax} gate "
                f"{mr['range_exceeds_item_noise']} != documented {exp_g}")
    out["literals_checked_against_archive"] = True
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--evidence_dir", required=True)
    ap.add_argument("--keep8_archive", required=True)
    ap.add_argument("--tag_prefix", default="A04_7B_keep10f2_NBR")
    ap.add_argument("--driver_log", default="logs/a04_keep10_nbr_82.out",
                    help="comma-separated; EVERY entry is protocol-gated")
    ap.add_argument("--node_label", required=True)
    ap.add_argument("--allow_other_numpy", action="store_true")
    args = ap.parse_args()

    if np.__version__ != PUBLISH_NUMPY and not args.allow_other_numpy:
        raise SystemExit(
            f"FATAL: numpy {np.__version__} != {PUBLISH_NUMPY}. The keep8 Leg A "
            "numbers this file replicates were published on .82/numpy 2.4.6, and "
            "Generator.multinomial differs in 19/10000 rows across these "
            "versions. Run on .82 or pass --allow_other_numpy and say so in the "
            "verdict.")

    mm_root = os.path.join(args.raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(args.raw_root, "olmo2_closedbook_results")

    # ---- 0. estimator self-test, before anything is published ----------
    selftest = selftest_estimators()

    # ---- 1. keep8's archive, read (not quoted from prose) --------------
    keep8 = keep8_archive_readback(args.keep8_archive)

    # ---- 2. arm set + seed disjointness, EXECUTED ----------------------
    arm_specs = {"intact_7B_base": dict(ANCHOR)}
    proto_specs = {}
    for st in STEPS:
        spec = _tag_dirs(args.tag_prefix, st)
        arm_specs[_arm_name(st)] = spec
        proto_specs[f"keep10|{st}"] = spec
    arm_names = [a for a in arm_specs if a != "intact_7B_base"]
    arm_index = {a: NEW_ARM_INDEX_BASE + i for i, a in enumerate(arm_names)}
    seed_check = assert_seeds_disjoint(
        args.evidence_dir, list(arm_index.values()),
        {"arm_index_base": NEW_ARM_INDEX_BASE,
         "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
         "interval_seed_offset": f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair"})

    driver_logs = {}
    for i, lg in enumerate(
            [x.strip() for x in args.driver_log.split(",") if x.strip()]):
        driver_logs[f"keep10_nbr_{i}" if i else "keep10_nbr"] = lg

    # ---- 3. PROTOCOL first, before anything is scored. Fails closed. ---
    proto = protocol_asserted(
        args.raw_root, driver_logs,
        "proposal/active/A04-recovery-certification/code/"
        "a04_keep10_neighbour_range_driver.sh",
        proto_specs)

    # ---- 4. explicit shard integrity BEFORE scoring --------------------
    integrity_explicit = shard_integrity_report(mm_root, cb_root, arm_specs)

    data, prov = {}, {}
    for arm, spec in arm_specs.items():
        data[arm], prov[arm] = _load_arm(mm_root, cb_root, spec)
    integrity_aligned = assert_aligned(data, prov)

    nulls = build_nulls(data["intact_7B_base"])
    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in data}

    # ---- 5. guard then NI, per tie convention --------------------------
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

    # ---- 6. Q1 -- the per-axis range and the gate ----------------------
    q1 = {}
    for axis in AXES:
        cs = [cell("split", _arm_name(st), axis) for st in STEPS]
        if any(x is None for x in cs):
            continue
        q1[axis] = {
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
            } for i, st in enumerate(STEPS)},
            "margin_range": range_report(
                [x["margin_pp"] for x in cs],
                [x["bootstrap_se_pp"] for x in cs], f"keep10|{axis}|margin"),
            "accuracy_range_pp": range_report(
                [100.0 * x["reported"] for x in cs],
                [x["bootstrap_se_pp"] for x in cs], f"keep10|{axis}|acc"),
            "adjacent_interval_paired_tests": adjacent_interval_tests(
                data, axis, ARM_KEY, STEPS, SEED + INTERVAL_SEED_OFF),
            "any_accept": bool(any(x["ni_accept"] for x in cs)),
        }

    # ---- 7. Q2 -- replication against keep8, per axis ------------------
    q2 = {"keep8_reference": keep8, "per_axis": {}}
    for axis in AXES:
        if axis not in q1 or axis not in keep8["per_axis"]:
            continue
        k10 = q1[axis]["margin_range"]
        k8 = keep8["per_axis"][axis]
        gate10 = bool(k10["range_exceeds_item_noise"])
        gate8 = bool(k8["range_exceeds_item_noise"])
        if gate10 and k10["range_ge_0p5pp"]:
            reading = PREREG_READINGS["replicates_and_material"]
            label = "REPLICATES_AND_MATERIAL"
        elif gate10:
            reading = PREREG_READINGS["replicates_in_kind_only"]
            label = "REPLICATES_IN_KIND_ONLY"
        else:
            reading = PREREG_READINGS["does_not_replicate"]
            label = "DOES_NOT_REPLICATE"
        q2["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "keep8_range_pp": k8["range_pp"],
            "keep8_gate": gate8,
            "keep8_floor_pp": k8["expected_range_if_pure_noise_pp"],
            "keep10_range_pp": k10["range_pp"],
            "keep10_gate": gate10,
            "keep10_floor_pp": k10["expected_range_if_pure_noise_pp"],
            "keep10_over_floor": k10["range_over_expected_noise_range"],
            "keep10_over_keep8_range": (k10["range_pp"] / k8["range_pp"]
                                        if k8["range_pp"] else None),
            "both_arms_clear_gate": bool(gate10 and gate8),
            "gate_agreement": bool(gate10 == gate8),
            "replication_label": label,
            "prereg_reading": reading,
            "note": ("RANGES are compared across arms because a range is a "
                     "WITHIN-arm quantity. The two arms' ABSOLUTE margins are "
                     "NOT comparable: keep10 is 12 layers, keep8 is 10, and the "
                     "keepN ladder additionally spans two corpora and unequal "
                     "step counts (STATUS.json:warning)."),
        }
    tq = q2["per_axis"].get("triviaqa", {})
    q2["headline_axis"] = "triviaqa"
    q2["headline_axis_why"] = (
        "triviaqa is the ONLY axis on which keep8 cleared the gate (1.1202 pp = "
        "1.70x floor); it is therefore the axis on which replication is defined. "
        "The other three keep8 ranges were sub-noise and there is nothing there "
        "to replicate.")
    q2["REPLICATES"] = bool(tq.get("keep10_gate"))
    q2["replication_label"] = tq.get("replication_label")
    q2["reading"] = tq.get("prereg_reading")

    # ---- 8. Q3 -- Heineman restatement (kept structurally separate) ----
    q3 = {
        "their_work": HEINEMAN,
        "comparability": {
            "IS_AN_EQUAL_FOOTING_COMPARISON": False,
            "may_be_tabulated_together": False,
            "asymmetries": [
                "n = 3 (ours) vs n = 30 (theirs)",
                "spacing 500 steps (ours) vs 1000 steps (theirs)",
                "DAMAGED, mid-heal, layer-pruned arm (ours) vs INTACT OLMo-2 "
                "(theirs)",
                "this repo's base protocol (cb_bs=32 / add_bos=False / "
                "max_new_tokens=32 / greedy) vs OLMES",
                "STRICT ZERO-SHOT (ours) vs FEW-SHOT (theirs, App. A.1) -- this "
                "plausibly raises our noise independent of damage, so it cuts "
                "AGAINST reading a gap as evidence about injury",
                "our mmlu_content is CONTENT-continuation scoring; their MMLU is "
                "LETTER-choice (RC/MC). At the anchor the two interfaces "
                "disagree on 40.1% of items (7B_base "
                "letter_vs_content_norm.agreement = 0.5994)",
                "our margins/accuracies are on a 12-layer pruned model whose "
                "triviaqa EM is ~0.15-0.16; theirs is a 32-layer intact 7B whose "
                "TriviaQA is ~0.6-0.7. rel.std has the MEAN IN THE "
                "DENOMINATOR, so a lower-accuracy model inflates rel.std for "
                "purely arithmetic reasons -- this alone can produce a "
                "multiplicative gap with no difference in absolute stability. "
                "The absolute SD is therefore reported next to every rel.std.",
            ],
            "n3_estimator_noise": sd_rel_uncertainty(3),
            "how_it_may_be_used": (
                "as a HYPOTHESIS ('are damaged arms noisier than intact models "
                "of the same family and scale?') worth a controlled test, never "
                "as a result. Any ratio quoted must carry 'n=3 vs n=30, "
                "different protocol' in the same sentence."),
        },
        "per_axis": {},
    }
    for axis in AXES:
        if axis not in q1:
            continue
        accs = [q1[axis]["per_step"][str(st)]["acc"] for st in STEPS]
        rs = relstd(accs)
        their_key = "mmlu" if axis == "mmlu_content" else axis
        theirs = HEINEMAN["olmo2_7B_4T_noise"].get(their_key)
        q3["per_axis"][axis] = {
            "decision_axis": axis not in DEMOTED_AXES,
            "keep10_rel_std": rs,
            "keep8_clean_cluster_rel_std": relstd(
                list(keep8["per_axis"][axis]["acc_by_step"].values()))
            if axis in keep8["per_axis"] else None,
            "their_olmo2_7B_4T_noise": theirs,
            "their_task_name": (their_key if theirs is not None else None),
            "ratio_ours_over_theirs": ((rs["rel_std"] / theirs)
                                       if (theirs and rs["rel_std"]) else None),
            "their_absolute_sd_pp_implied": (
                # rel.std * their mean accuracy. Their accuracy is NOT in Table 4,
                # so this is intentionally left None rather than guessed: the
                # absolute-SD comparison would need their per-checkpoint scores.
                None),
            "why_absolute_sd_comparison_is_not_made": (
                "Table 4 publishes rel.std (noise) and Rel.Dispersion (signal), "
                "NOT the mean accuracy of the 30 checkpoints, so their absolute "
                "SD cannot be reconstructed from the table. Our sd_pp is "
                "reported so a future reader who obtains their raw scores can "
                "complete the comparison; inventing their mean from Figure 12 "
                "would be a guess."),
            "comparator_absent_reason": (
                None if theirs is not None else
                f"{axis} is not in Heineman et al.'s 30-benchmark suite; the "
                "cell is LEFT BLANK and must not be filled with a neighbouring "
                "task"),
            "interface_mismatch": (
                "their MMLU is LETTER-choice, our decision axis is "
                "content-continuation; letter_acc is recorded as the "
                "interface-matched secondary"
                if axis == "mmlu_content" else None),
            "caveat_in_band": (
                "n=3 @500 steps, damaged mid-heal arm, STRICT ZERO-SHOT, this "
                "repo's base protocol vs n=30 @1000 steps, intact OLMo-2, "
                "FEW-SHOT, OLMES. NOT an equal-footing comparison."),
        }

    # interface-matched MMLU secondary, read straight from the summaries
    letter = {}
    for st in STEPS:
        sp = os.path.join(mm_root, _tag_dirs(args.tag_prefix, st)["mmlu"],
                          "summary.json")
        s = json.load(open(sp))
        letter[str(st)] = {"letter_acc": s["letter_acc"],
                           "content_norm_acc": s["content_norm_acc"],
                           "content_raw_acc": s["content_raw_acc"]}
    q3["mmlu_letter_interface_matched_secondary"] = {
        "per_step": letter,
        "letter_rel_std": relstd([v["letter_acc"] for v in letter.values()]),
        "content_norm_rel_std": relstd(
            [v["content_norm_acc"] for v in letter.values()]),
        "their_mmlu_noise": HEINEMAN["olmo2_7B_4T_noise"]["mmlu"],
        "why": ("their MMLU is letter-choice. This is the closest INTERFACE "
                "match to what they measured, and it is reported so a reader "
                "can see both rather than being handed only the axis that "
                "flatters the comparison. It is NOT a decision axis here "
                "(A03_1B_FLOOR_VERDICT dropped MMLU-letter at 1B; at 7B it is "
                "carried descriptively)."),
        "still_not_equal_footing": True,
    }

    # ---- 9. headline ---------------------------------------------------
    dec_gate = {ax: bool(q1[ax]["margin_range"]["range_exceeds_item_noise"])
                for ax in DECISION_AXES if ax in q1}
    n_gate = sum(dec_gate.values())
    if q2["REPLICATES"] and tq.get("keep10_range_pp", 0) >= 0.5:
        headline = "KEEP10_NEIGHBOUR_RANGE_REPLICATES_MATERIAL_TRIVIAQA"
    elif q2["REPLICATES"]:
        headline = "KEEP10_NEIGHBOUR_RANGE_REPLICATES_IN_KIND_SUB_0p5PP"
    elif n_gate:
        headline = ("KEEP10_TRIVIAQA_DOES_NOT_REPLICATE_BUT_"
                    + "_".join(sorted(a for a, g in dec_gate.items() if g)).upper()
                    + "_CLEARS_GATE")
    else:
        headline = "KEEP10_NO_AXIS_CLEARS_NOISE_GATE_KEEP8_TRIVIAQA_IS_ISOLATED"

    out = {
        "gate": "A04_keep10_neighbour_range_second_arm_7B",
        "question": (
            "(Q1) per-axis NI-margin range across a 500-step checkpoint triple "
            "of keep10+fresh2, gated on 1.6926*sigma; (Q2) does keep8's single "
            "supra-noise range (triviaqa 1.1202 pp) replicate on this second, "
            "differently-damaged arm? (Q3) the same data restated as Heineman et "
            "al.'s rel.std, against their published intact OLMo-2 7B values."),
        "date": "2026-08-13",
        "headline_verdict": headline,
        "prereg_document": "A04_KEEP10_NEIGHBOUR_RANGE_PREREG.md",
        "prereg_readings_fixed_in_advance": PREREG_READINGS,
        "prereg_commit_note": (
            "the pre-registration was committed as its own commit BEFORE the "
            "first checkpoint was scored, so the readings' git timestamp "
            "precedes every number in this file"),
        "tests_the_claim_from": (
            "A04_GATE_DESIGN.md 2.0.2 + 2.5 / "
            "STATUS.json:neighbour_variability_20260813 -- the neighbour "
            "precondition and its per-axis tolerance, both resting on ONE arm "
            "and, on the decision axes, ONE supra-noise cell. 2.5 itself says "
            "the numbers 'should be widened if a second arm is ever measured'."),
        "arm": {"key": ARM_KEY, "dir": ARM_DIR, "steps": STEPS,
                "spacing_steps": SPACING_STEPS, "arch": ARM_ARCH,
                **SINGLE_PROCESS_PROVENANCE},
        "estimator_selftest": selftest,
        "seed_disjointness_checked": seed_check,
        "protocol_asserted": proto,
        "shard_integrity_explicit": integrity_explicit,
        "integrity_aligned": integrity_aligned,
        "node_of_record": {
            "node": args.node_label, "numpy": np.__version__,
            "why_it_matters": (
                "Generator.multinomial differs in 19/10000 rows between numpy "
                "2.4.6 (.82) and 2.5.1 (.73/.104/.21); max observed margin drift "
                "0.005294 pp, triviaqa only. The keep8 Leg A numbers replicated "
                "here were published on .82/2.4.6, so this comparison is WITHIN "
                "one sampler. The drift is 10.6x looser than the 5e-4 pp "
                "hard-fail in a04_keep14_trajectory_ni.py and may NOT be cited "
                "to explain away any move larger than ~0.006 pp."),
            "keep8_archive_published_on": keep8["numpy_that_published_it"],
        },
        "bootstrap_offsets": {
            "arm_index": arm_index, "form": "97*arm_index + 13*axis_index",
            "guard_seed_offset": f"SEED+{GUARD_SEED_OFF}+13*axis_index",
            "interval_seed_offset":
                f"SEED+{INTERVAL_SEED_OFF}+13*axis+7*pair",
            "n_boot": N_BOOT, "base_seed": SEED,
            "disjointness": "checked mechanically, see seed_disjointness_checked",
        },
        "prereg": {
            "gate_design": "A04_GATE_DESIGN.md 2 + 2.0.2",
            "delta_fraction": PREREG["delta_fraction"], "rho": PREREG["rho"],
            "commit_freezing_constants": PREREG["commit"],
            "ni_definition": ("accept iff one-sided lower 95% bound on "
                              "residual(arm)-residual(intact) > -Delta, "
                              "Delta = 0.10*residual(intact); imported ni_rule"),
            "noise_gate": ("range_exceeds_item_noise := range_pp > 1.6926 * "
                           "mean(per-cell bootstrap SE); 1.6926 = 3/sqrt(pi) = "
                           "E[range of 3 iid N(0,1)], exact for the normal. "
                           "Imported range_report -- the SAME code object that "
                           "gated keep8."),
            "decision_axes": DECISION_AXES,
            "demoted_axes": sorted(DEMOTED_AXES),
            "delta_never_substituted": True, "anchor_never_changed": True,
        },
        "intact_anchor": {
            "choice": "vanilla models/OLMo-2-1124-7B (mode=base, 32 layers)",
            "dirs": ANCHOR,
            "imported_from": "a04_shallow_rung_ni_7b.ANCHOR (not redeclared)",
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
        "Q1_per_axis_range_and_gate": q1,
        "Q1_decision_axis_summary": {
            ax: {"range_pp": q1[ax]["margin_range"]["range_pp"],
                 "expected_range_if_pure_noise_pp":
                     q1[ax]["margin_range"]["expected_range_if_pure_noise_pp"],
                 "range_over_floor":
                     q1[ax]["margin_range"]["range_over_expected_noise_range"],
                 "range_exceeds_item_noise": dec_gate[ax],
                 "best_is_not_last": q1[ax]["margin_range"]["best_is_not_last"],
                 "best_minus_last_pp":
                     q1[ax]["margin_range"]["best_minus_last_pp"]}
            for ax in dec_gate},
        "Q1_n_decision_axes_clearing_gate": n_gate,
        "Q2_replication_vs_keep8": q2,
        "Q3_heineman_relstd_restatement": q3,
        "output_shape_and_flips_diagnostic": {},
        "NOT_licensed": [
            "Treating the three checkpoints as REPLICATES of one another. They "
            "are successive states of ONE optimisation; their spread is heal "
            "progress + data order. This is a CHECKPOINT-SELECTION quantity, "
            "NOT seed variance. No 7B sd_run exists or is reconstructible (one "
            "seed per rung; --seed moves only fresh-tail init, never data "
            "order; historical 7B seeds unrecorded).",
            "Comparing keep10 / keep8 / keep12 / keep14 ABSOLUTE margins as "
            "rungs of one ladder. Four different depths, two corpora and "
            "unequal step counts (STATUS.json:warning). Only the RANGES -- a "
            "within-arm statistic -- are compared across arms.",
            "Calling any of this 'harness noise'. Same-code re-runs on a fixed "
            "checkpoint are BIT-IDENTICAL "
            "(full32_rescore_v2_20260812.correction_to_the_jitter_premise). "
            "Item-sampling variability is a different thing and is what the "
            "gate quantifies.",
            "Tabulating our rel.std alongside Heineman et al.'s as if measured "
            "together: n=3 vs n=30, 500 vs 1000-step spacing, damaged vs "
            "intact, base protocol vs OLMES, content-continuation vs "
            "letter-choice MMLU. At n=3 the sample SD's own relative SD is "
            "52.3%, so a ~2x ratio is uninformative before any protocol "
            "difference. HYPOTHESIS ONLY.",
            "Reporting any range that FAILS range_exceeds_item_noise as a "
            "measured neighbour gap. A max-minus-min of 3 noisy cells is biased "
            "upward even at zero true spread.",
            "Any K1/K2/K3 clause -- defined over the pre-registered 1B arm set.",
            "Quoting any margin here to better than 0.01 pp ACROSS NODES "
            "(numpy multinomial drift, see node_of_record).",
        ],
        "gpu_note": ("CPU-only analysis. GPU cost was the 4-axis scoring of the "
                     "three checkpoints on .82; this script loads only "
                     "per-example shards."),
    }

    for ax in ("triviaqa", "popqa", "nq_open"):
        r = output_shape_and_flips(data, ax, ARM_KEY, STEPS)
        if r is not None:
            out["output_shape_and_flips_diagnostic"][
                f"{ARM_KEY}|{STEPS[0]}-{STEPS[-1]}|{ax}"] = r

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump(out, open(args.out_json, "w"), indent=1, default=float)

    # ---- console -------------------------------------------------------
    print("=" * 110)
    print(f"PROTOCOL CONFIRMED FROM THE INVOCATION   node={args.node_label} "
          f"numpy={np.__version__}")
    print("=" * 110)
    for label, r in proto["from_driver_logs"].items():
        print(f"  {label:<18} {r['log']}  cb_bs={r['header_cb_bs']} "
              f"mmlu_bs={r['header_mmlu_bs']}  per-axis={r['per_axis_bs_echoed']}")
    print(f"  add_bos is False on all {len(proto['add_bos_from_summaries'])} "
          "result dirs (asserted with `is False`)")
    print(f"  estimator self-test: all_ok={selftest['all_ok']}")
    print()
    print("=" * 110)
    print("Q1 -- keep10+fresh2 step89000/89500/90000 (ONE process, NO resume seam)")
    print("=" * 110)
    print(f"  {'axis':<14}{'margins (pp)':<40}{'range':>9}{'E[rng|noise]':>14}"
          f"{'ratio':>8}{'>noise?':>9}{'best!=last':>12}")
    for ax, v in q1.items():
        mr = v["margin_range"]
        ms = " ".join(f"{x:+.4f}" for x in mr["margins_pp"])
        rat = mr["range_over_expected_noise_range"]
        print(f"  {ax:<14}{ms:<40}{mr['range_pp']:>9.4f}"
              f"{mr['expected_range_if_pure_noise_pp']:>14.4f}"
              f"{(rat if rat else float('nan')):>8.2f}"
              f"{str(mr['range_exceeds_item_noise']):>9}"
              f"{str(mr['best_is_not_last']):>12}"
              + ("" if v["decision_axis"] else "  (demoted)"))
        for k, t in v["adjacent_interval_paired_tests"].items():
            print(f"  {'':<14}   {k:<18} acc {t['acc_delta_pp']:+.4f}pp "
                  f"CI95[{t['ci95_pp'][0]:+.4f},{t['ci95_pp'][1]:+.4f}] "
                  f"p={t['boot_p_two_sided']:.4f} "
                  f"{'RESOLVED' if t['distinguishable_from_zero_at_95'] else 'within item noise'}"
                  f" (+{t['wrong_to_right']}/-{t['right_to_wrong']})")
    print()
    print("=" * 110)
    print("Q2 -- REPLICATION vs keep8fresh2 clean cluster (ranges only; absolute "
          "margins are NOT comparable)")
    print("=" * 110)
    print(f"  {'axis':<14}{'keep8 range':>13}{'gate':>7}{'keep10 range':>14}"
          f"{'gate':>7}{'k10/floor':>11}{'k10/k8':>9}  label")
    for ax, v in q2["per_axis"].items():
        print(f"  {ax:<14}{v['keep8_range_pp']:>13.4f}"
              f"{str(v['keep8_gate']):>7}{v['keep10_range_pp']:>14.4f}"
              f"{str(v['keep10_gate']):>7}"
              f"{(v['keep10_over_floor'] or float('nan')):>11.2f}"
              f"{(v['keep10_over_keep8_range'] or float('nan')):>9.2f}"
              f"  {v['replication_label']}")
    print(f"\n  REPLICATES (triviaqa, the only axis keep8 cleared): "
          f"{q2['REPLICATES']}  -> {q2['replication_label']}")
    print()
    print("=" * 110)
    print("Q3 -- Heineman et al. rel.std unit  ***n=3 vs n=30, DAMAGED vs INTACT, "
          "different protocol -- NOT equal footing***")
    print("=" * 110)
    print(f"  {'axis':<14}{'keep10 rel.std':>16}{'keep8 rel.std':>16}"
          f"{'their OLMo-2 7B':>18}{'ratio':>9}")
    for ax, v in q3["per_axis"].items():
        k8 = v["keep8_clean_cluster_rel_std"]
        th = v["their_olmo2_7B_4T_noise"]
        print(f"  {ax:<14}{v['keep10_rel_std']['rel_std']:>16.4f}"
              f"{(k8['rel_std'] if k8 else float('nan')):>16.4f}"
              f"{(th if th is not None else float('nan')):>18}"
              f"{(v['ratio_ours_over_theirs'] or float('nan')):>9.2f}")
    ls = q3["mmlu_letter_interface_matched_secondary"]
    print(f"  {'mmlu_LETTER':<14}{ls['letter_rel_std']['rel_std']:>16.4f}"
          f"{'--':>16}{ls['their_mmlu_noise']:>18}"
          f"{(ls['letter_rel_std']['rel_std']/ls['their_mmlu_noise']):>9.2f}"
          "   <- interface-matched to theirs")
    print(f"  at n=3 the sample SD's own relative SD is "
          f"{sd_rel_uncertainty(3)['rel_sd_of_sample_sd']:.3f} -- a ~2x ratio is "
          "uninformative before any protocol difference")
    print(f"\nHEADLINE: {headline}")
    print(f"READING : {q2['reading']}")
    print(f"wrote {args.out_json}")


if __name__ == "__main__":
    main()
