#!/usr/bin/env python3
"""A04 Pilot Zero — do PLATEAU(T) and NI(Delta) disagree at all?

Pilot Zero answers exactly one question (A04_GATE_DESIGN.md §6.1):

    does a disagreement of the required shape exist at all between the two
    acceptance rules, on one arm, one seed?

It can fire **K1** (kill: no disagreement anywhere -> A04 stops for ~0 GPU) and
provisionally bears on **K3** (axes at floor). It **cannot** clear **K2**, which
needs multiple seeds and is deliberately out of scope.

PRE-REGISTERED CONSTANTS (frozen in git commit d1ba737, 2026-08-09 23:02:27
+0800, which predates this analysis; verified with `git show`). They are NOT
tunable here and this script has no CLI flag to change them:

    T      = 2.0 % relative in-domain val PPL improvement per 5,000 steps
    rho    = 0.85 (RATIO rule)
    Delta_x = 0.10 * residual(intact, x)

Everything is recomputed from per-example shards / summary JSONs. Nothing is
copied from the design doc's prose table — the doc's numbers are secondary and
this repo has been bitten twice by headlines that existed only in prose.

The canonical scorers/nulls are IMPORTED from A03's
`analyze_1b_knowledge_floor.py` (never reimplemented — two subagents have
already produced spurious significance by reimplementing a metric).

Shard completeness is HARD-ASSERTED (8/8 + exact item count) before any merge:
a silently merged 5-of-8 shard set has corrupted results in this repo before.

CPU ONLY. No GPU, no model load, no torch.

Usage:
  python pilot_zero_rule_disagreement.py --raw_root <dir> --ppl_json <file> \
      --out_json <file> [--out_csv <file>]
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import Counter

import numpy as np

# ---- canonical scorers/nulls: IMPORT, never reimplement -------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
# A03 was ARCHIVED 2026-08-11 (proposal/active -> proposal/archive). Its
# `analyze_1b_knowledge_floor.py` is still the canonical scorer/null source and
# is imported, never reimplemented. Resolve its directory through the shared
# helper so the location lives in ONE place and a missing A03 fails loudly
# instead of silently falling back to a re-derived metric.
_SHARED_CODE = os.path.abspath(os.path.join(
    _HERE, "..", "..", "..", "shared", "code"))
if _SHARED_CODE not in sys.path:
    sys.path.insert(0, _SHARED_CODE)
from proposal_paths import a03_code_dir  # noqa: E402

_A03_CODE = a03_code_dir()
if _A03_CODE not in sys.path:
    sys.path.insert(0, _A03_CODE)
from analyze_1b_knowledge_floor import (  # noqa: E402
    N_BOOT,
    SEED,
    TIE_CONVS,
    best_constant_letter,
    best_constant_qa,
    longest_option_vector,
    paired_bootstrap,
)

# ---------------------------------------------------------------------------
# PRE-REGISTERED, frozen by git d1ba737. Deliberately module-level constants
# with no CLI override: "do not tune T, rho or Delta to obtain a result".
# ---------------------------------------------------------------------------
PREREG = {
    "commit": "d1ba737",
    "commit_utc": "2026-08-09 23:02:27 +0800",
    "T_pct_per_5k": 2.0,
    "rho": 0.85,
    "delta_rule": "0.10 * residual(intact, x)",
    "delta_fraction": 0.10,
}

# Expected item counts (A04_GATE_DESIGN.md §4 `n` column; each independently
# re-asserted against the merged shard sets below).
EXPECTED_N = {
    "mmlu": 14042,
    "triviaqa": 17944,
    "popqa": 14267,
    "nq_open": 3610,
}

# NQ-open was DEMOTED in the design (§5.2): its item-level 95% CI half-width
# 1.459-2.063pp at n=3610 already exceeds its own Delta=0.970pp, so it cannot
# carry a 10%-of-residual NI decision. Kept descriptive only.
DEMOTED_AXES = {"nq_open"}

# The four capability axes of the gate, in the design's order.
AXES = ["triviaqa", "popqa", "mmlu_content", "nq_open"]
PRIMARY_AXIS = "triviaqa"

LETTERS = "ABCDEFGHIJKLMNOP"


# ---------------------------------------------------------------------------
# shard loading with HARD completeness assertions
# ---------------------------------------------------------------------------
def load_shards(d, stem, expected_n, n_shards=8):
    """Merge per_example_<stem>_shard{0..7}of8.jsonl with hard assertions.

    Asserts: exactly `n_shards` shard files exist, every shard index 0..7 is
    present exactly once, no duplicate item_id after merge, and the merged count
    equals `expected_n`. Any failure raises -- a partial merge must never be
    silently scored.
    """
    pat = os.path.join(d, f"per_example_{stem}_shard*of{n_shards}.jsonl")
    files = sorted(glob.glob(pat))
    assert len(files) == n_shards, (
        f"SHARD INCOMPLETE {d} {stem}: expected {n_shards} shards, found "
        f"{len(files)}: {[os.path.basename(f) for f in files]}")
    seen_idx = set()
    rows = []
    for f in files:
        base = os.path.basename(f)
        idx = int(base.split("_shard")[1].split("of")[0])
        assert idx not in seen_idx, f"duplicate shard index {idx} in {d}"
        seen_idx.add(idx)
        with open(f) as fh:
            for line in fh:
                if line.strip():
                    rows.append(json.loads(line))
    assert seen_idx == set(range(n_shards)), (
        f"SHARD INDEX GAP {d} {stem}: have {sorted(seen_idx)}")
    ids = [r["item_id"] for r in rows]
    assert len(set(ids)) == len(ids), f"duplicate item_id after merge in {d}/{stem}"
    assert len(rows) == expected_n, (
        f"ITEM COUNT MISMATCH {d} {stem}: merged {len(rows)} != expected "
        f"{expected_n}")
    rows.sort(key=lambda r: r["item_id"])
    return rows


def mmlu_content_norm_vec(rows):
    """Per-item 0/1 correctness of the content_norm interface.

    Recomputed from the stored per-option scores with index tie-breaking, i.e.
    exactly what A03's `model_correct(..., norm=True)` does; asserted against the
    harness's own stored `content_norm.correct` flag.
    """
    out = np.zeros(len(rows))
    stored = np.zeros(len(rows))
    for i, r in enumerate(rows):
        n = r["n_opt"]
        sc = r["content_norm"]["scores"]
        v = [sc[LETTERS[k]] for k in range(n)]
        pred = max(range(n), key=lambda k: v[k])
        out[i] = 1.0 if pred == r["gold"] else 0.0
        stored[i] = 1.0 if r["content_norm"]["correct"] else 0.0
    # the harness's own flag must agree with the recomputation
    assert np.array_equal(out, stored), (
        "content_norm recomputation disagrees with harness-stored "
        f"'correct' flag on {int((out != stored).sum())} items")
    return out


def qa_metric_vec(rows, metric):
    """Per-item metric vector, taken from the harness's own stored field."""
    return np.asarray([float(r[metric]) for r in rows], dtype=float)


# ---------------------------------------------------------------------------
# the two acceptance rules
# ---------------------------------------------------------------------------
def plateau_rule(ppl_traj, T_pct_per_5k):
    """PLATEAU(T): accept at checkpoint c iff the relative in-domain val PPL
    improvement over the PRECEDING grid interval is < T.

    ⚠️ SUPERSEDED 2026-08-10 — BOTH readings computed below are dimensionally
    broken on an irregular grid, and neither should be used to run the gate.
    They are retained ONLY so this script keeps reproducing the numbers in
    evidence/pilot_zero_rule_disagreement.json byte-for-byte.

      accept_unscaled: 15.70x stringency spread across A04's own frozen grid
                       {2500,5000,10000,20000,40000,80000} when re-expressed in
                       the %/5k units the threshold claims; correct at exactly
                       one of the six checkpoints (d=5000).
      accept_scaled:   accepts a run still improving at exactly T for every
                       d > 5000 (relative improvement compounds, the linear
                       allowance does not); NOT composition-consistent
                       (174/200,000 random checkpoint pairs accept a merged
                       interval while accepting neither half); VACUOUS at
                       d >= 250,000 steps.

    The rule now in force converts to a per-5k GEOMETRIC rate first:
        rate_5k = 100 * (1 - (ppl_c/ppl_prev) ** (5000/d))  <  T
    which equals the arithmetic below EXACTLY at d = 5000 (asserted < 1e-12).
    → code/a04_plateau_rule_repair.py,
      evidence/a04_plateau_rule_repair.json,
      A04_PLATEAU_REPAIR_AND_MARGIN_SENSITIVITY.md section 1.

    Consequence for this script's output: under the repaired rule PLATEAU first
    accepts at step 100,000, NOT step 200,000. This script's K1 bookkeeping
    (PLATEAU_DEFINED_ARMS = {keep7f2_step200000}) therefore evaluates a cell
    that the repaired rule still accepts, but no longer the EARLIEST one. Step
    100,000 has no capability scoring, so that cell is UNMEASURED.

    T is expressed per 5,000 steps, so the raw interval improvement is compared
    against T scaled by (interval_steps / 5000). Reported both ways: the
    literal-rate reading (scaled, honours the units as written) and the
    per-interval reading (unscaled), because the design's own §6.1 arithmetic
    used the unscaled interval numbers while the definition carries /5k units.
    The grid here is 47k-53k spaced, so the two readings differ by ~10x and the
    design doc explicitly flags that this grid "cannot exercise a
    5,000-step-resolution rule".
    """
    out = []
    for i, (step, ppl) in enumerate(ppl_traj):
        if i == 0:
            out.append({"step": step, "ppl": ppl, "interval_steps": None,
                        "rel_improve_pct": None,
                        "accept_unscaled": None, "accept_scaled": None,
                        "T_scaled_pct": None})
            continue
        prev_step, prev_ppl = ppl_traj[i - 1]
        d_steps = step - prev_step
        rel = 100.0 * (prev_ppl - ppl) / prev_ppl
        T_scaled = T_pct_per_5k * (d_steps / 5000.0)
        out.append({
            "step": step, "ppl": ppl, "interval_steps": d_steps,
            "rel_improve_pct": rel,
            # literal units: T is "% per 5k steps" -> scale to the interval
            "accept_scaled": bool(rel < T_scaled),
            "T_scaled_pct": T_scaled,
            # the design's own §6.1 reading: compare raw interval improvement to T
            "accept_unscaled": bool(rel < T_pct_per_5k),
        })
    return out


def ni_rule(arm_vec, intact_vec, delta_fraction, intact_residual,
            null_vec_arm=None, null_vec_intact=None, seed_off=0):
    """NI(Delta): accept iff the one-sided lower 95% bound on
    residual(arm) - residual(intact) is > -Delta.

    Note residual = reported - null, and the SAME input-blind null applies to
    both arms on the same item set, so the null CANCELS exactly in the
    difference:
        residual(arm) - residual(intact) = reported(arm) - reported(intact).
    The difference is therefore null-invariant. Delta, however, is
    0.10 * residual(intact) and DOES depend on the null -- which is why the
    convention-sensitivity check below is meaningful for MMLU.

    One-sided lower 95% bound = 5th percentile of the paired item bootstrap on
    the per-item difference vector.
    """
    d = np.asarray(arm_vec, float) - np.asarray(intact_vec, float)
    rng = np.random.default_rng(SEED + seed_off)
    n = d.size
    vals, counts = np.unique(d, return_counts=True)
    draws = rng.multinomial(n, counts / n, size=N_BOOT)
    means = draws @ vals / n
    lo95_one_sided = float(np.percentile(means, 5.0))
    delta = delta_fraction * intact_residual
    return {
        "diff_mean_pp": 100.0 * float(d.mean()),
        "diff_lower95_one_sided_pp": 100.0 * lo95_one_sided,
        "delta_pp": 100.0 * delta,
        "ni_accept": bool(lo95_one_sided > -delta),
        "n": int(n),
        "n_boot": N_BOOT,
        "boot_seed": SEED + seed_off,
    }


def ratio_rule(reported_by_axis_arm, reported_by_axis_intact, rho, axes):
    """RATIO(rho): accept iff mean_over_axes(reported_a / reported_intact) >= rho."""
    ratios = {}
    for a in axes:
        ri = reported_by_axis_intact[a]
        ratios[a] = (reported_by_axis_arm[a] / ri) if ri > 0 else None
    vals = [v for v in ratios.values() if v is not None]
    m = float(np.mean(vals)) if vals else None
    return {"per_axis_ratio": ratios, "mean_ratio": m, "rho": rho,
            "ratio_accept": (None if m is None else bool(m >= rho)),
            "axes_used": axes}


# ---------------------------------------------------------------------------
def build_axis_data(raw_root, arm_dirs):
    """Load every axis for every arm, with completeness assertions, and build
    the input-blind nulls once on the shared item set."""
    mmlu_root = os.path.join(raw_root, "olmo2_mmlu_content_results")
    cb_root = os.path.join(raw_root, "olmo2_closedbook_results")

    data = {}   # arm -> axis -> per-item vector
    meta = {}   # provenance
    for arm, spec in arm_dirs.items():
        data[arm] = {}
        meta[arm] = {}
        # MMLU content
        if spec.get("mmlu"):
            d = os.path.join(mmlu_root, spec["mmlu"])
            rows = load_shards(d, "mmlu", EXPECTED_N["mmlu"])
            data[arm]["mmlu_content"] = mmlu_content_norm_vec(rows)
            data[arm]["_mmlu_rows"] = rows
            meta[arm]["mmlu"] = {"dir": d, "n": len(rows), "shards": 8}
        # closed-book generative
        if spec.get("cb"):
            d = os.path.join(cb_root, spec["cb"])
            for task in ("triviaqa", "popqa"):
                rows = load_shards(d, task, EXPECTED_N[task])
                data[arm][task] = qa_metric_vec(rows, "em")
                data[arm][f"_{task}_rows"] = rows
                meta[arm][task] = {"dir": d, "n": len(rows), "shards": 8,
                                   "metric": "em"}
        if spec.get("nq"):
            d = os.path.join(cb_root, spec["nq"])
            rows = load_shards(d, "nq_open", EXPECTED_N["nq_open"])
            data[arm]["nq_open"] = qa_metric_vec(rows, "em")
            data[arm]["_nq_open_rows"] = rows
            meta[arm]["nq_open"] = {"dir": d, "n": len(rows), "shards": 8,
                                    "metric": "em"}
    return data, meta


def build_nulls(intact_data):
    """Construct-appropriate, input-blind nulls, computed on the SAME item set
    as the reported numbers. 'Above chance' is never used.

    MMLU-content: longest-option null, ALL FIVE tie conventions (A01 established
    the convention itself is a degree of freedom worth 25.76 pp).
    Generative QA: best-constant answer string, argmax over candidates.
    Also records the MMLU-letter always-D null (0.2689, not 0.25) and BoolQ
    always-B (0.6217, not 0.50) for the record, though MMLU-letter is BANNED as
    an axis by the design.
    """
    nulls = {}

    rows = intact_data["_mmlu_rows"]
    gold_letters = [r["gold_letter"] for r in rows]
    letter, letter_acc, letter_vec, dist = best_constant_letter(gold_letters)
    conv_vecs, conv_accs = {}, {}
    for conv in TIE_CONVS:
        v = longest_option_vector(rows, gold_letters, conv)
        conv_vecs[conv] = v
        conv_accs[conv] = float(v.mean())
    # winner-set diagnostics
    mult = Counter()
    gold_in_win = 0
    for r, g in zip(rows, gold_letters):
        c = r["content_norm"]["cont_tokens"]
        keys = [k for k in LETTERS if k in c]
        top = max(c[k] for k in keys)
        win = [k for k in keys if c[k] == top]
        mult[len(win)] += 1
        gold_in_win += int(g in win)
    nulls["mmlu_content"] = {
        "kind": "longest-option (continuation-token argmax set)",
        "preregistered_convention": "split",
        "by_convention": conv_accs,
        "vectors": conv_vecs,
        "winner_set_size_hist": {str(k): mult[k] for k in sorted(mult)},
        "frac_items_with_tied_longest": 1.0 - mult[1] / len(rows),
        "frac_items_gold_in_winner_set": gold_in_win / len(rows),
        "convention_spread_pp": 100.0 * (max(conv_accs.values())
                                         - min(conv_accs.values())),
    }
    nulls["_mmlu_letter_for_the_record"] = {
        "kind": f"best-constant always-{letter} (NOT 0.25)",
        "acc": letter_acc,
        "gold_letter_distribution": dist,
        "chance_line_never_used": 0.25,
        "banned_as_axis_by_design": True,
    }

    for task in ("triviaqa", "popqa", "nq_open"):
        rows = intact_data[f"_{task}_rows"]
        s, acc, vec, diag = best_constant_qa(rows, "em")
        nulls[task] = {
            "kind": f"best-constant answer string {s!r} (argmax over candidates)",
            "acc": acc, "vector": vec, "diagnostics": diag,
        }
    return nulls


def recommend_j(pilot_zero_frac_recovered, ladder_json):
    """Pick the damage depth `j` the real gate should use.

    Evidence, all from files on disk:
      * 1B keep7+fresh2 (9/16 = 56.25% depth kept) after 52.43 B heal tokens
        recovers only the fractions measured by Pilot Zero -> a constant-REJECT
        rung, useless for testing a rule that is supposed to sometimes accept.
      * A 7B MMLU-content depth ladder with the SAME split-tie null exists in
        A01's gate3 evidence and is monotone in depth. It is the only measured
        depth->recovery curve in the repo, so it is the only evidence available
        for extrapolating which depth plausibly approaches non-inferiority.

    CAVEAT carried into the output: the ladder is 7B, not 1B, and (per
    A04_GATE_DESIGN.md §3) it spans two corpora and unequal step counts, so it
    is suggestive of ordering only, never of an absolute recovery level. It is
    used here purely to rank candidate j, which is what §6.1 asks for.
    """
    arms = ladder_json["arms"]
    base = arms["7B_base"]
    bn = base["longest_option_floor_by_conv"]["split"]
    bres = base["by_dtype"]["bf16"]["content_norm_acc"] - bn
    ladder = {}
    for k, v in arms.items():
        acc = v["by_dtype"]["bf16"]["content_norm_acc"]
        nul = v["longest_option_floor_by_conv"]["split"]
        ladder[k] = {
            "content_norm_acc_bf16": acc,
            "null_split": nul,
            "residual_pp": 100.0 * (acc - nul),
            "frac_of_intact_residual": (acc - nul) / bres,
        }
    # depth kept, from the arm names (OLMo-2-7B has 32 layers; keepN+fresh2 =>
    # N+2 layers of 32). shortgpt16 keeps 16 of 32 non-contiguously.
    depth = {"7B_keep8_step121000": (8 + 2) / 32,
             "7B_keep10_step83500": (10 + 2) / 32,
             "7B_keep12_step124000": (12 + 2) / 32,
             "7B_keep14_step200000": (14 + 2) / 32,
             "7B_shortgpt16_step200000": 16 / 32}
    for k, v in depth.items():
        ladder[k]["frac_depth_kept"] = v
    return {
        "question": "which j should the real gate use?",
        "pilot_zero_arm": {
            "arm": "1B keep7+fresh2 @ step200000",
            "layers": "9 of 16", "frac_depth_kept": 9 / 16,
            "heal_tokens_B": 52.43,
            "frac_of_intact_residual_recovered": pilot_zero_frac_recovered,
            "verdict": "CONSTANT-REJECT rung: NI rejects on every decision axis "
                       "by 5-30x the margin. A rule tested only here can never "
                       "be observed to accept, so it proves nothing about the "
                       "rule's discrimination.",
        },
        "depth_recovery_ladder_7B_mmlu_content_split_tie": ladder,
        "ladder_source":
            "proposal/active/A01-null-calibration-methodology/evidence/"
            "gate3_content_null_conventions.json (7B, bf16, split-tie null)",
        "ladder_caveats": [
            "7B, not the 1B scale the gate runs at.",
            "The keepN ladder spans TWO corpora and UNEQUAL steps "
            "(A04_GATE_DESIGN.md §3: 7,570,911 vs 15,491,607 rows; keep14 200k "
            "/ keep12 124k / keep10 83.5k / keep8 121k). Suggestive of ORDERING "
            "only, never of an absolute recovery level.",
            "One seed per rung; no run-to-run variance available (and see "
            "SEED_SEMANTICS_DEFECT.md -- the repo has no true seed variance at "
            "all yet).",
        ],
        "recommendation": {
            "j": 12,
            "arm": "keep12+fresh2 at 1B = 14 of 16 layers = 87.5% depth kept",
            "why": [
                "The gate needs a rung where NI can plausibly ACCEPT, otherwise "
                "PLATEAU-vs-NI disagreement is unfalsifiable: a constant-REJECT "
                "rung makes the disagreement automatic and uninformative.",
                "The only measured depth->recovery curve (7B MMLU-content, "
                "split-tie null) is monotone in depth kept: 31.1% @ 31.2% "
                "depth, 32.2% @ 37.5%, 42.2% @ 43.8%, 53.1% @ 50.0%. "
                "Recovery rises with depth kept across the whole measured "
                "range with no sign of saturating, so the shallowest cut "
                "available is the one most likely to reach non-inferiority.",
                "keep12+fresh2 at 1B keeps 87.5% of depth, far shallower than "
                "any rung on that 7B ladder (max 50%), so it is the candidate "
                "most likely to let NI accept at an affordable token budget.",
                "It also makes the 20,000-step (5.24 B token) affordable budget "
                "plausible: at keep7 (56.2% depth) 52.43 B tokens were not "
                "enough; a much shallower cut needs far less repair.",
            ],
            "second_choice": {
                "j": 10, "arm": "keep10+fresh2 at 1B = 12 of 16 = 75% depth",
                "why": "if keep12 turns out to be a constant-ACCEPT rung "
                       "(damage too mild to ever trip NI), keep10 is the next "
                       "rung down and brackets the interesting region."},
            "explicitly_NOT_recommended": {
                "j": 7,
                "why": "keep7+fresh2 is the arm Pilot Zero just measured and it "
                       "is a constant-REJECT rung. The design already says so; "
                       "Pilot Zero confirms it quantitatively."},
            "UNVERIFIED": [
                "No 1B arm at keep12 or keep10 exists on either disk, so the "
                "recovery level at those depths at 1B is UNMEASURED. The "
                "recommendation is an extrapolation from a 7B ladder whose own "
                "confounds are listed above.",
                "Whether keep12 at 1B is instead a constant-ACCEPT rung (too "
                "mild to ever trip NI) is UNKNOWN. The gate should be prepared "
                "to bracket, which is why a second choice is given.",
            ],
        },
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw_root", required=True,
                    help="dir holding olmo2_{mmlu_content,closedbook}_results/")
    ap.add_argument("--ppl_json", required=True,
                    help="JSON: [[step, ppl], ...] recomputed from summary.json")
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--ladder_json", default=None,
                    help="A01 gate3_content_null_conventions.json, used ONLY to "
                         "rank candidate j (7B depth->recovery ladder)")
    args = ap.parse_args()

    arm_dirs = {
        "intact": {"mmlu": "A03_1B_base", "cb": "A03_1B_base",
                   "nq": "A03_1B_base_nq"},
        "keep7f2_step200000": {"mmlu": "A03_1B_keep7_step200k",
                               "cb": "A03_1B_keep7_step200k",
                               "nq": "A03_1B_keep7_step200k_nq"},
        # free extra checkpoints (design §6.1). cpt20k = arm3.
        "cpt20k_step205000": {"mmlu": "A03_1B_arm3_cpt_step205000",
                              "cb": "A03_1B_arm3_cpt_step205000",
                              "nq": "A03_1B_arm3_cpt_step205000_nq"},
        "cpt20k_step210000": {"mmlu": "A03_1B_arm3_cpt_step210000",
                              "cb": "A03_1B_arm3_cpt_step210000",
                              "nq": "A03_1B_arm3_cpt_step210000_nq"},
        "cpt20k_step215000": {"mmlu": "A03_1B_arm3_cpt_step215000",
                              "cb": "A03_1B_arm3_cpt_step215000",
                              "nq": "A03_1B_arm3_cpt_step215000_nq"},
        "cpt20k_step220000": {"mmlu": "A03_1B_arm3_cpt_step220000",
                              "cb": "A03_1B_arm3_cpt_step220000",
                              "nq": "A03_1B_arm3_cpt_step220000_nq"},
        # arm4_peaklr20k: grid coverage ONLY. A03's STATUS says NOT_YET_JUDGED
        # and its intermediate cells swing +-1.3pp on triviaqa EM in BOTH
        # directions (Adam-moment mismatch, ARM4_DESIGN.md). NOT settled
        # capability numbers.
        "arm4_peaklr20k_step205000": {"mmlu": "A03_1B_arm4_peaklr_step205000",
                                      "cb": "A03_1B_arm4_peaklr_step205000",
                                      "nq": "A03_1B_arm4_peaklr_step205000_nq"},
        "arm4_peaklr20k_step210000": {"mmlu": "A03_1B_arm4_peaklr_step210000",
                                      "cb": "A03_1B_arm4_peaklr_step210000",
                                      "nq": "A03_1B_arm4_peaklr_step210000_nq"},
        "arm4_peaklr20k_step215000": {"mmlu": "A03_1B_arm4_peaklr_step215000",
                                      "cb": "A03_1B_arm4_peaklr_step215000",
                                      "nq": "A03_1B_arm4_peaklr_step215000_nq"},
    }
    UNSETTLED_ARMS = {a for a in arm_dirs if a.startswith("arm4_peaklr20k")}

    data, prov = build_axis_data(args.raw_root, arm_dirs)
    nulls = build_nulls(data["intact"])

    # ---- reported + residual per (arm, axis), per MMLU convention ----------
    def null_acc(axis, conv="split"):
        if axis == "mmlu_content":
            return nulls["mmlu_content"]["by_convention"][conv]
        return nulls[axis]["acc"]

    reported = {a: {x: float(data[a][x].mean()) for x in AXES if x in data[a]}
                for a in arm_dirs}

    # ---- PLATEAU on the real PPL trajectory -------------------------------
    ppl_traj = json.load(open(args.ppl_json))
    ppl_traj = [(int(s), float(p)) for s, p in ppl_traj]
    plateau = plateau_rule(ppl_traj, PREREG["T_pct_per_5k"])

    # ---- the disagreement table, for every MMLU tie convention ------------
    per_convention = {}
    for conv in TIE_CONVS:
        intact_resid = {x: reported["intact"][x] - null_acc(x, conv)
                        for x in AXES}
        cells = []
        for ai, arm in enumerate(arm_dirs):
            if arm == "intact":
                continue
            for xi, axis in enumerate(AXES):
                if axis not in data[arm]:
                    continue
                r = ni_rule(data[arm][axis], data["intact"][axis],
                            PREREG["delta_fraction"], intact_resid[axis],
                            seed_off=97 * ai + 13 * xi)
                arm_resid = reported[arm][axis] - null_acc(axis, conv)
                cells.append({
                    "arm": arm, "axis": axis,
                    "demoted_descriptive_only": axis in DEMOTED_AXES,
                    "arm_capability_unsettled": arm in UNSETTLED_ARMS,
                    "reported": reported[arm][axis],
                    "reported_intact": reported["intact"][axis],
                    "null": null_acc(axis, conv),
                    "residual_arm": arm_resid,
                    "residual_intact": intact_resid[axis],
                    "residual_fraction_recovered": (
                        arm_resid / intact_resid[axis]
                        if intact_resid[axis] > 0 else None),
                    # Delta = 0.10*residual(intact) is only a meaningful
                    # non-inferiority margin if residual(intact) > 0. Under the
                    # 'credit' (oracle tie-break) convention the INTACT model
                    # itself falls BELOW its own MMLU null, so residual(intact)
                    # < 0 and Delta < 0: NI then demands the lower bound exceed
                    # a POSITIVE number, i.e. it demands strict superiority and
                    # is no longer a non-inferiority test at all. Flagged, not
                    # silently scored.
                    "delta_degenerate_negative_margin": bool(
                        intact_resid[axis] <= 0),
                    **r,
                })
        ratio = {}
        for arm in arm_dirs:
            if arm == "intact":
                continue
            axes_present = [x for x in AXES if x in data[arm]]
            ratio[arm] = ratio_rule(reported[arm], reported["intact"],
                                    PREREG["rho"], axes_present)
        per_convention[conv] = {
            "intact_residual": intact_resid,
            "delta_pp": {x: 100.0 * PREREG["delta_fraction"] * intact_resid[x]
                         for x in AXES},
            "cells": cells,
            "ratio_rule": ratio,
        }

    # ---- K1 / K3 evaluation on the PRE-REGISTERED convention --------------
    # Which (arm, checkpoint) rows have a MEASURED in-domain PPL, i.e. where
    # PLATEAU(T) is even DEFINED? Only the keep7f2 trajectory has PPL on disk:
    # olmo2_ppl_results/ contains 1B_base_full and 1B_keep7_step{50000,100000,
    # 147000,200000} and NOTHING for cpt20k or arm4_peaklr20k (verified by ls on
    # zwfy6). For the extra checkpoints PLATEAU is UNDEFINED, so they cannot
    # contribute a PLATEAU-vs-NI disagreement; they contribute grid coverage and
    # NI/RATIO evidence only. Counting them as disagreements would be inventing
    # a rule evaluation that no measurement supports.
    PLATEAU_DEFINED_ARMS = {"keep7f2_step200000": 200000}

    def k_verdicts(conv, plateau_reading):
        pc = per_convention[conv]
        acc_key = "accept_" + plateau_reading
        plateau_accept_steps = {p["step"] for p in plateau
                                if p.get(acc_key) is True}
        # -- decision cells: exclude the DEMOTED axis (design §5.2) -----------
        dec_all = [c for c in pc["cells"] if not c["demoted_descriptive_only"]]
        # -- cells where BOTH rules are defined and PLATEAU accepts ----------
        confirm = [
            c for c in dec_all
            if c["arm"] in PLATEAU_DEFINED_ARMS
            and PLATEAU_DEFINED_ARMS[c["arm"]] in plateau_accept_steps
        ]
        ni_reject = [c for c in confirm if not c["ni_accept"]]
        n_axes_ni_accept = len(confirm) - len(ni_reject)
        # K1 (verbatim): at every checkpoint where PLATEAU accepts, NI also
        # accepts on >= 3 of the 4 axes, AND the count of disagreement cells is
        # <= 1 out of >= 24 evaluated cells.
        k1_clause_a = (len(confirm) > 0) and (n_axes_ni_accept >= 3)
        n_disagree = len(ni_reject)
        n_eval_cells = len(dec_all)
        k1_clause_b = n_disagree <= 1
        k1_fires = bool(k1_clause_a and k1_clause_b)
        # K3: >= 3 of 4 axes have the INTACT arm's calibrated residual < 5pp
        below5 = [x for x in AXES if 100.0 * pc["intact_residual"][x] < 5.0]
        k3_fires = len(below5) >= 3
        degen = sorted({c["axis"] for c in dec_all
                        if c.get("delta_degenerate_negative_margin")})
        # Does the K1 verdict survive dropping any axis whose Delta is
        # degenerate (negative) under this convention? If yes, the verdict is
        # not an artefact of the degenerate margin.
        confirm_nd = [c for c in confirm if c["axis"] not in set(degen)]
        dec_nd = [c for c in dec_all if c["axis"] not in set(degen)]
        k1_fires_nd = bool(
            len(confirm_nd) > 0
            and len([c for c in confirm_nd if c["ni_accept"]]) >= 3
            and len([c for c in dec_nd if not c["ni_accept"]]) <= 1)
        return {
            "convention": conv,
            "plateau_reading": plateau_reading,
            "plateau_accept_steps": sorted(plateau_accept_steps),
            "plateau_defined_only_for": PLATEAU_DEFINED_ARMS,
            "plateau_undefined_note":
                "no in-domain PPL exists on disk for cpt20k or arm4_peaklr20k "
                "(olmo2_ppl_results/ has only 1B_base_full and "
                "1B_keep7_step{50000,100000,147000,200000}), so PLATEAU is "
                "UNDEFINED there and those cells cannot form a "
                "PLATEAU-vs-NI disagreement.",
            "n_cells_both_rules_defined_and_plateau_accepts": len(confirm),
            "n_ni_accept_at_plateau_accept": n_axes_ni_accept,
            "n_ni_reject_at_plateau_accept": len(ni_reject),
            "ni_reject_axes_at_plateau_accept": [c["axis"] for c in ni_reject],
            "n_decision_cells_evaluated": n_eval_cells,
            "n_disagreement_cells": n_disagree,
            "K1_clause_a_ni_accepts_on_ge3_of_4": bool(k1_clause_a),
            "K1_clause_b_disagreements_le_1": bool(k1_clause_b),
            "K1_fires": k1_fires,
            "K1_reason": (
                "PLATEAU accepts and NI also accepts on >=3 of 4 axes with "
                "<=1 disagreement -> no certification problem" if k1_fires else
                f"at the PLATEAU-accept checkpoint (step 200,000) NI REJECTS on "
                f"{len(ni_reject)}/{len(confirm)} decision axes "
                f"({', '.join(c['axis'] for c in ni_reject)}), so NI accepts on "
                f"{n_axes_ni_accept}/4 < 3 -> clause (a) fails; and "
                f"{n_disagree} > 1 disagreement cells -> clause (b) fails. "
                f"A disagreement of the required shape EXISTS."),
            "n_cells_caveat": (
                f"K1's verbatim text asks for '>= 24 evaluated cells'. Pilot "
                f"Zero evaluates {n_eval_cells} decision cells but only "
                f"{len(confirm)} of them have PLATEAU defined (1 arm x 1 "
                f"checkpoint x 3 decision axes). The >=24-cell precondition is "
                f"therefore NOT met by Pilot Zero; the K1 finding here is that "
                f"clause (a) fails decisively on the cells that do exist, "
                f"which is sufficient to NOT fire K1 but is not the full "
                f"4-arm x 6-checkpoint family the gate would evaluate."),
            "intact_residual_pp": {x: 100.0 * pc["intact_residual"][x]
                                   for x in AXES},
            "axes_with_intact_residual_below_5pp": below5,
            "axes_with_degenerate_negative_delta": degen,
            "degenerate_delta_note": (
                "" if not degen else
                f"under convention '{conv}' the INTACT arm's calibrated "
                f"residual is <= 0 on {degen}, so Delta = 0.10*residual(intact) "
                f"is NEGATIVE and NI(Delta) stops being a non-inferiority test "
                f"(it would demand strict superiority). The K1 verdict on those "
                f"axes is therefore not interpretable as non-inferiority under "
                f"this convention; it is reported for completeness only. Note "
                f"the K1 verdict does NOT depend on those axes here — see "
                f"K1_survives_excluding_degenerate_axes."),
            "K1_survives_excluding_degenerate_axes": k1_fires_nd,
            "n_ni_reject_excluding_degenerate_axes": len(
                [c for c in confirm_nd if not c["ni_accept"]]),
            "n_cells_excluding_degenerate_axes": len(confirm_nd),
            "K3_fires": bool(k3_fires),
            "K3_note": (
                "K3 asks whether the SCALE can support the measurement. "
                "Pilot Zero bears on it provisionally only; A03 verified 4/5 "
                "interfaces above floor at 1B. Not cleared for the gate's "
                "chosen j, which is a different arm."),
            "K2_status": "OUT OF SCOPE — needs >=3 seeds; Pilot Zero cannot "
                         "clear or fire K2. And under the pre-fix trainer the "
                         "only multi-'seed' evidence in the repo is "
                         "fresh-block-INIT variance, not run-to-run variance "
                         "(see SEED_SEMANTICS_DEFECT.md).",
        }

    verdicts = {}
    for conv in TIE_CONVS:
        for reading in ("unscaled", "scaled"):
            verdicts[f"{conv}|{reading}"] = k_verdicts(conv, reading)

    prereg_v = verdicts[f"split|unscaled"]

    # ---- j recommendation --------------------------------------------------
    j_block = None
    if args.ladder_json:
        frac = {c["axis"]: c["residual_fraction_recovered"]
                for c in per_convention["split"]["cells"]
                if c["arm"] == "keep7f2_step200000"}
        j_block = recommend_j(frac, json.load(open(args.ladder_json)))

    out = {
        "what": "A04 Pilot Zero — does PLATEAU(T) vs NI(Delta) disagree at all?",
        "scope": "one arm, one seed. Can fire K1; bears provisionally on K3; "
                 "CANNOT clear K2.",
        "preregistration": PREREG,
        "preregistration_note":
            "T, rho, Delta are frozen in git d1ba737 and are module-level "
            "constants with no CLI override. They were NOT tuned to obtain "
            "this result.",
        "caveats_carried_from_design_doc": [
            "The band (1.388%, 3.297%] in A04_GATE_DESIGN.md §6.1 was computed "
            "AFTER seeing the numbers and is ILLUSTRATIVE, NOT CONFIRMATORY.",
            "The PPL grid has only 4 points at 47k-53k spacing and therefore "
            "CANNOT exercise a 5,000-step-resolution rule. Both the scaled "
            "(literal-units) and unscaled (design §6.1) readings of PLATEAU "
            "are reported for this reason.",
            "arm4_peaklr20k intermediate cells are NOT settled capability "
            "numbers (Adam-moment mismatch, +-1.3pp swings in both directions; "
            "A03 STATUS = NOT_YET_JUDGED). Used for grid coverage only.",
            "NQ-open is DEMOTED (item-level CI half-width 1.459-2.063pp "
            "exceeds its own Delta=0.970pp at n=3610): descriptive only, "
            "excluded from decision cells.",
        ],
        "provenance": prov,
        "expected_item_counts_asserted": EXPECTED_N,
        "nulls": {
            "mmlu_content": {k: v for k, v in nulls["mmlu_content"].items()
                             if k != "vectors"},
            "mmlu_letter_for_the_record": nulls["_mmlu_letter_for_the_record"],
            **{t: {k: v for k, v in nulls[t].items() if k != "vector"}
               for t in ("triviaqa", "popqa", "nq_open")},
            "boolq_for_the_record": {
                "kind": "always-B (NOT 0.50)", "acc": 0.6217,
                "source": "status/scout_21/lane2_a01_gate2.md via "
                          "A04_GATE_DESIGN.md §4.2 — not recomputed here",
                "UNVERIFIED_here": True},
        },
        "null_invariance_of_the_difference":
            "residual(arm)-residual(intact) = reported(arm)-reported(intact) "
            "because the same input-blind null applies to both arms on the same "
            "item set, so the null cancels EXACTLY in the difference. The "
            "convention therefore moves only Delta (=0.10*residual(intact)), "
            "never the measured difference.",
        "ppl_trajectory": [{"step": s, "ppl": p} for s, p in ppl_traj],
        "plateau_rule": plateau,
        "per_convention": {
            conv: {
                "intact_residual": per_convention[conv]["intact_residual"],
                "delta_pp": per_convention[conv]["delta_pp"],
                "ratio_rule": per_convention[conv]["ratio_rule"],
                "cells": per_convention[conv]["cells"],
            } for conv in TIE_CONVS
        },
        "verdicts_by_convention_and_reading": verdicts,
        "j_recommendation": j_block,
        "HEADLINE": {
            "preregistered_convention": "split",
            "preregistered_plateau_reading": "unscaled (design §6.1 arithmetic)",
            "K1_fires": prereg_v["K1_fires"],
            "K1_reason": prereg_v["K1_reason"],
            "K3_fires": prereg_v["K3_fires"],
            "K2_status": prereg_v["K2_status"],
            "K1_fires_under_all_five_conventions": {
                c: verdicts[f"{c}|unscaled"]["K1_fires"] for c in TIE_CONVS},
            "K1_fires_under_scaled_reading_all_conventions": {
                c: verdicts[f"{c}|scaled"]["K1_fires"] for c in TIE_CONVS},
            "RATIO_rho085_accepts_at_step200000": per_convention["split"][
                "ratio_rule"]["keep7f2_step200000"]["ratio_accept"],
            "RATIO_mean_ratio_at_step200000": per_convention["split"][
                "ratio_rule"]["keep7f2_step200000"]["mean_ratio"],
            "RATIO_note":
                "A04's claim §1 names TWO incumbent rules: (a) a PPL plateau "
                "and (b) an aggregate retained-accuracy ratio. Only (a) "
                "disagrees with NI here. RATIO(rho=0.85) REJECTS at step "
                "200,000 (mean ratio 0.4017 << 0.85), i.e. it AGREES with NI. "
                "So the disagreement Pilot Zero finds is specific to the "
                "PLATEAU rule and A04 must NOT claim it for the ratio rule. "
                "This narrows the claim and is a first-order finding, not a "
                "footnote.",
            "recommended_j": (None if j_block is None
                              else j_block["recommendation"]["j"]),
        },
    }

    with open(args.out_json, "w") as fh:
        json.dump(out, fh, indent=1, default=float)
    print(f"wrote {args.out_json}")

    if args.out_csv:
        import csv
        with open(args.out_csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["convention", "arm", "axis", "demoted", "unsettled",
                        "reported", "reported_intact", "null",
                        "residual_arm_pp", "residual_intact_pp",
                        "frac_recovered", "diff_mean_pp",
                        "diff_lower95_one_sided_pp", "delta_pp", "ni_accept"])
            for conv in TIE_CONVS:
                for c in per_convention[conv]["cells"]:
                    w.writerow([
                        conv, c["arm"], c["axis"],
                        c["demoted_descriptive_only"],
                        c["arm_capability_unsettled"],
                        f"{c['reported']:.6f}", f"{c['reported_intact']:.6f}",
                        f"{c['null']:.6f}", f"{100*c['residual_arm']:.4f}",
                        f"{100*c['residual_intact']:.4f}",
                        ("" if c["residual_fraction_recovered"] is None
                         else f"{c['residual_fraction_recovered']:.4f}"),
                        f"{c['diff_mean_pp']:.4f}",
                        f"{c['diff_lower95_one_sided_pp']:.4f}",
                        f"{c['delta_pp']:.4f}", c["ni_accept"]])
        print(f"wrote {args.out_csv}")

    # ---- human-readable summary -------------------------------------------
    print("\n=== PPL trajectory / PLATEAU(T=2.0%/5k) ===")
    for p in plateau:
        if p["interval_steps"] is None:
            print(f"  step {p['step']:>7}  ppl {p['ppl']:.4f}   (first point)")
        else:
            print(f"  step {p['step']:>7}  ppl {p['ppl']:.4f}   "
                  f"d_steps={p['interval_steps']:>6}  rel_improve="
                  f"{p['rel_improve_pct']:.3f}%  "
                  f"accept_unscaled={p['accept_unscaled']}  "
                  f"accept_scaled={p['accept_scaled']} "
                  f"(T_scaled={p['T_scaled_pct']:.2f}%)")

    print("\n=== intact calibrated residuals (prereg convention 'split') ===")
    for x in AXES:
        rr = 100.0 * per_convention["split"]["intact_residual"][x]
        dd = per_convention["split"]["delta_pp"][x]
        tag = "  [DEMOTED: descriptive only]" if x in DEMOTED_AXES else ""
        print(f"  {x:<14} residual={rr:7.3f}pp  Delta={dd:6.3f}pp{tag}")

    print("\n=== NI(Delta) at step 200,000, arm keep7+fresh2 (prereg 'split') ===")
    for c in per_convention["split"]["cells"]:
        if c["arm"] != "keep7f2_step200000":
            continue
        tag = " [DEMOTED]" if c["demoted_descriptive_only"] else ""
        print(f"  {c['axis']:<14} recovered="
              f"{100*(c['residual_fraction_recovered'] or 0):6.2f}%  "
              f"diff={c['diff_mean_pp']:8.3f}pp  "
              f"lo95={c['diff_lower95_one_sided_pp']:8.3f}pp  "
              f"-Delta={-c['delta_pp']:7.3f}pp  "
              f"NI_accept={c['ni_accept']}{tag}")

    print("\n=== RATIO(rho=0.85) ===")
    for arm, r in per_convention["split"]["ratio_rule"].items():
        if r["mean_ratio"] is None:
            continue
        print(f"  {arm:<28} mean_ratio={r['mean_ratio']:.4f}  "
              f"accept={r['ratio_accept']}")

    print("\n=== HEADLINE ===")
    print(f"  K1 fires (prereg split/unscaled): {prereg_v['K1_fires']}")
    print(f"    {prereg_v['K1_reason']}")
    print(f"  K1 across all 5 conventions (unscaled): "
          f"{out['HEADLINE']['K1_fires_under_all_five_conventions']}")
    print(f"  K1 across all 5 conventions (scaled):   "
          f"{out['HEADLINE']['K1_fires_under_scaled_reading_all_conventions']}")
    print(f"  K3 fires: {prereg_v['K3_fires']}  "
          f"(intact residuals pp: "
          f"{ {k: round(v,3) for k,v in prereg_v['intact_residual_pp'].items()} })")
    print(f"  K2: {prereg_v['K2_status']}")


if __name__ == "__main__":
    main()
