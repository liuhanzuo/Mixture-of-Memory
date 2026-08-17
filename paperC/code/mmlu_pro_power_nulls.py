#!/usr/bin/env python3
"""paperC task #251 — THE POWER WALL. Construct-appropriate nulls for the
letter-vs-content MC interface on MMLU-Pro (n=12032, 10-way).

WHAT THIS CLOSES
----------------
#248 and #250 both returned PARTIAL, and both hit the SAME wall: n.

  task            n     CI95 half-width   n needed for hw < 1.389 pp
  MMLU        14042      1.15 pp          (reference)
  arc_easy     2376      1.31 pp          ~2097   (0.9x -- already enough)
  piqa         1838      2.78 pp          ~7336   (4.0x)
  csqa         1221      3.40 pp          ~7312   (6.0x)
  arc_ch       1172      3.88 pp          ~9154   (7.8x)
  openbookqa    500      6.40 pp         ~10615  (21.2x -- test set is 500)

52 of #250's 60 damaged cells were underpowered to have detected MMLU's own
-1.389 pp effect. That is NOT an effect-size problem (arc_challenge's median
damaged effect, -3.840 pp, is LARGER than MMLU's -3.603 pp) and it cannot be
fixed by adding more SMALL benchmarks. MMLU-Pro has n=12032 -- MMLU's order of
magnitude -- so it is the first non-MMLU benchmark with the resolution to
answer the question at all.

THE ACCEPTANCE CRITERION of this script is therefore ONE number: the achieved
CI95 half-width on the letter-vs-floor test. If it is < 1.389 pp, the null
results here are interpretable and the MMLU headline is genuinely testable
off MMLU. If not, the direction can only be rescued by pooling.

10-WAY SPECIFICS (why MMLU-Pro is also the right SUBSTANTIVE test)
-----------------------------------------------------------------
#248 recorded honestly that on arc/obqa/piqa/csqa the best-constant letter
floor is only +0.43 to +2.60 pp above chance, i.e. paperC's "chance badly
misstates the null" rhetoric is WEAK there. On a 10-way benchmark the marginal
has 10 cells to be skewed across, so this is the strongest available test of
that rhetoric. `chance` has TWO defensible readings here and both are reported:
  * naive 1/10 = 0.100000 (what a reader assumes on seeing "10-way")
  * mean(1/n_opt) = 0.110877 (n_opt is NOT constant: {10:9981, 9:801, 8:320,
    7:158, 6:93, 5:52, 4:606, 3:21})
The RELATIVE misstatement (floor/chance) is the comparable quantity across
benchmarks with different option counts, and it is reported alongside the
absolute pp gap so a 10-way floor is never compared to a 4-way floor in pp
alone.

Nulls are IDENTICAL in definition to #248/#250 (no new estimator, no new
assumption), so every cell is directly comparable:
  letter  -> BEST-CONSTANT LETTER = argmax of the gold-letter marginal. NOT
             1/n_opt, NOT 0.10, NOT 0.25.
  content -> LONGEST-OPTION null, CONTINUATION-TOKEN unit, all FIVE tie
             conventions (split/first/last/credit/wrong).

Statistics: paired bootstrap n_boot=10000 seed=7, two-sided p from the R-7-fixed
mid-p `two_sided_boot_p`, exact McNemar vs the deterministic constant predictor.
Estimators are IMPORTED from the A01 code dir and asserted identical to local
verbatim copies -- the same arrangement gate2_crossfamily_nulls.py uses.

CPU only. No GPU, no model load.

Usage:
  python mmlu_pro_power_nulls.py <results_root> <out_json> [out_csv]
      [--olmo_json <#248 json>] [--xf_json <#250 json>]
      [--mmlu_root olmo2_mmlu_content_results]
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
from collections import Counter

import numpy as np

LETTERS = "ABCDEFGHIJ"
CONVS = ["split", "first", "last", "credit", "wrong"]
N_BOOT = 10000
BOOT_SEED = 7  # same seed as #248/#250 so the three tables are comparable

TASK = "mmlu_pro"
EXPECTED_N = 12032
# MMLU's own headline effect on OLMo-2 keep8: letter 0.2550 vs always-D 0.2689.
# The ACCEPTANCE CRITERION of #251 is achieved half-width < |this|.
MMLU_REFERENCE_EFFECT_PP = -1.389

# The six OLMo-2 arms of #248 / gate-3, ordered base -> most damaged. Directory
# names deliberately mirror olmo2_mmlu_content_results/ and
# olmo2_mc_letter_content_results/ so pairing is a name lookup.
OLMO_ARMS = [
    ("base", "7B_base"),
    ("shortgpt16", "7B_shortgpt16_step200000"),
    ("keep14", "7B_keep14_step200000"),
    ("keep12", "7B_keep12_step124000"),
    ("keep10", "7B_keep10_step83500"),
    ("keep8", "7B_keep8_step121000"),
]
# EVERY structurally pruned OLMo-2 arm is designated damaged. (Fixed 2026-08-16 after an
# audit of the denominator.) This set was previously {keep8, keep10, keep12} -- inherited
# from #248, where only three rungs were the relevant power contrast -- and that narrower
# list silently defined BOTH headline denominators (MMLU-Pro 14/15 and off-MMLU 0/15),
# omitting shortgpt16 (+3.674 pp above floor, p=0.0001) and keep14 (+0.324, p=0.3234)
# without naming them, which sections/09a_relocated.tex forbids. The omission was not
# defensible as a rule either, because XF_DAMAGED below counts the SAME retained depth
# of 14 layers as damaged for all three non-OLMo families in the same ratio.
# paperC/code/gate_designated_denominator.py now asserts this set equals the arm list
# declared in sections/04_experiments.tex. Do not narrow it without changing both.
#
# ⚠️ DE-ATTRIBUTED 2026-08-17, and keep it that way. The two lines above used to name a
# review round and an issue ID. This file is not cited by the prose today, so it is not
# copied into the blind submission snapshot -- but it IS imported by
# code/mmlu_pro_trunc_fix_compare.py, which IS cited, so one citation edit away it ships.
# An artifact that names the round tells a blind reader the review history; that class of
# disclosure voided a previous panel's scores. State the finding, not who raised it.
DAMAGED_OLMO = {"keep8", "keep10", "keep12", "keep14", "shortgpt16"}

XF_FAMILIES = ["llama2_7b", "llama3_8b", "qwen3_8b_base"]
XF_RUNGS = ["base", "k14", "k12", "k10", "k8"]
XF_DAMAGED = {"k8", "k10", "k12", "k14"}  # #250 counts k14 as damaged (no heal)
N_LAYERS = {"llama2_7b": 32, "llama3_8b": 32, "qwen3_8b_base": 36,
            "olmo2_7b": 32}

# The #248 five-benchmark power table, for the head-to-head. Values are quoted
# from paperC/evidence/SECOND_MC_BENCHMARK_VERDICT.md section 2b (keep8 cells).
REF_POWER_TABLE = {
    "mmlu":           {"n": 14042, "keep8_delta_pp": -1.389, "half_width_pp": 1.154},
    "arc_easy":       {"n": 2376,  "keep8_delta_pp": -0.800, "half_width_pp": 1.305},
    "winogrande":     {"n": 1267,  "keep8_delta_pp": +0.552, "half_width_pp": 1.184},
    "piqa":           {"n": 1838,  "keep8_delta_pp": +2.503, "half_width_pp": 2.775},
    "commonsense_qa": {"n": 1221,  "keep8_delta_pp": -1.065, "half_width_pp": 3.399},
    "arc_challenge":  {"n": 1172,  "keep8_delta_pp": -0.939, "half_width_pp": 3.882},
    "openbookqa":     {"n": 500,   "keep8_delta_pp": -1.800, "half_width_pp": 6.400},
}


# ---------------------------------------------------------------------------
# estimators: imported from the A01 code dir for provenance, local verbatim
# copies asserted identical. Same arrangement as gate2_crossfamily_nulls.py.
# ---------------------------------------------------------------------------
def _two_sided_boot_p_local(bs, n_boot=None):
    """R-7-fixed mid-p two-sided bootstrap p. Splits the atom at zero evenly
    between the tails so the tails sum to 1 and p <= 1 is STRUCTURAL (the old
    `2*min((bs<=0).mean(), (bs>=0).mean())` double-counted the zero atom and
    produced the illegal p=1.042 that R-7 fixed)."""
    bs = np.asarray(bs, dtype=np.float64)
    if n_boot is None:
        n_boot = bs.size
    tie = float((bs == 0).mean())
    p_lo = float((bs < 0).mean()) + 0.5 * tie
    p_hi = float((bs > 0).mean()) + 0.5 * tie
    return float(min(1.0, max(2.0 * min(p_lo, p_hi), 1.0 / n_boot)))


def _mcnemar_exact_p_local(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lh = n * math.log(0.5)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + lh
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, 2.0 * math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_A01 = os.path.join(_REPO, "proposal/active/A01-null-calibration-methodology/code")
ESTIMATOR_SOURCE = "local verbatim copies (A01 code dir not importable)"
two_sided_boot_p = _two_sided_boot_p_local
mcnemar_exact_p = _mcnemar_exact_p_local
if os.path.isdir(_A01):
    try:
        sys.path.insert(0, _A01)
        from a01_gate3_fp32_vs_bf16 import (  # noqa: E402
            mcnemar_exact_p as _mc_imported,
            two_sided_boot_p as _bp_imported,
        )
        _rng = np.random.default_rng(0)
        _bs = np.where(_rng.random(5000) < 0.4, 0.0, _rng.normal(0, 1e-4, 5000))
        assert abs(_bp_imported(_bs, 5000) - _two_sided_boot_p_local(_bs, 5000)) < 1e-12
        for _b, _c in ((0, 0), (1, 0), (28, 28), (300, 250)):
            assert abs(_mc_imported(_b, _c) - _mcnemar_exact_p_local(_b, _c)) < 1e-12
        two_sided_boot_p = _bp_imported
        mcnemar_exact_p = _mc_imported
        ESTIMATOR_SOURCE = ("imported from a01_gate3_fp32_vs_bf16.py; local "
                            "verbatim copies asserted identical")
    except Exception as e:  # pragma: no cover
        print(f"[warn] A01 estimator import failed ({e}); using local copies")


def paired_boot(d, n_boot=N_BOOT, seed=BOOT_SEED):
    d = np.asarray(d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    bs = np.empty(n_boot)
    for i in range(n_boot):
        bs[i] = d[rng.integers(0, d.size, d.size)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi), two_sided_boot_p(bs, n_boot)


def verdict(mean, p):
    if p >= 0.05:
        return "AT the floor (n.s.)"
    return "BELOW the floor" if mean < 0 else "above the floor"


# ---------------------------------------------------------------------------
def load_records(root, arm, expected_n=EXPECTED_N, num_shards=8):
    """Load one arm's per-item records WITH hard integrity asserts. Never
    analyse a partial set: a silent 5/8 merge destroys the protocol."""
    d = os.path.join(root, arm)
    shards = sorted(glob.glob(
        os.path.join(d, f"per_example_{TASK}_shard*of{num_shards}.jsonl")))
    idx = {int(os.path.basename(s).split("_shard")[1].split("of")[0])
           for s in shards}
    missing = sorted(set(range(num_shards)) - idx)
    assert not missing, f"{arm}: MISSING shards {missing} of {num_shards}"
    recs = []
    for s in shards:
        with open(s) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{arm}: duplicate item_id"
    assert len(recs) == expected_n, f"{arm}: n_scored={len(recs)} != {expected_n}"
    n_nan = sum(1 for r in recs if r["nan"])
    assert n_nan == 0, f"{arm}: n_nan={n_nan} != 0"
    recs.sort(key=lambda r: r["item_id"])
    return recs, {"n_shards": len(shards), "n_scored": len(recs),
                  "expected_n": expected_n, "n_nan": n_nan}


# ---------------------------------------------------------------------------
def best_constant_letter(recs):
    """Best-constant (input-blind) letter predictor = argmax of the gold-letter
    marginal. A pure DATASET property -> must be bit-identical across every arm
    and every family, which is asserted by the caller."""
    gold = Counter(r["gold_letter"] for r in recs)
    best, cnt = max(gold.items(), key=lambda kv: (kv[1], -LETTERS.index(kv[0])))
    vec = np.array([1.0 if r["gold_letter"] == best else 0.0 for r in recs])
    chance_mean = float(np.mean([1.0 / r["n_opt"] for r in recs]))
    # naive chance a reader assumes on being told "10-way"
    chance_naive = 1.0 / max(r["n_opt"] for r in recs)
    acc = float(vec.mean())
    # WORST constant, so the SPREAD of the marginal is on the record too: a flat
    # marginal means the best-constant floor is close to chance and paperC's
    # rhetoric is weak on this benchmark. This is the honesty check #248 made.
    worst, wcnt = min(gold.items(), key=lambda kv: (kv[1], LETTERS.index(kv[0])))
    return best, vec, {
        "gold_letter_marginal": dict(sorted(gold.items())),
        "gold_letter_marginal_frac": {k: v / len(recs)
                                      for k, v in sorted(gold.items())},
        "n_opt_hist": {str(k): v for k, v in
                       sorted(Counter(r["n_opt"] for r in recs).items())},
        "best_constant_letter": best,
        "best_constant_acc": acc,
        "worst_constant_letter": worst,
        "worst_constant_acc": wcnt / len(recs),
        "marginal_spread_pp": 100 * (cnt - wcnt) / len(recs),
        "chance_naive_1_over_max_nopt": chance_naive,
        "chance_mean_1_over_nopt": chance_mean,
        "floor_minus_chance_naive_pp": 100 * (acc - chance_naive),
        "floor_minus_chance_mean_pp": 100 * (acc - chance_mean),
        # the RELATIVE misstatement is what is comparable across benchmarks with
        # different option counts; pp alone is not.
        "floor_over_chance_naive": acc / chance_naive,
        "floor_over_chance_mean": acc / chance_mean,
        "sanity_best_constant_eq_count_over_n":
            abs(acc - cnt / len(recs)) < 1e-12,
    }


def longest_option_nulls(recs):
    """Longest-option content null, CONTINUATION-TOKEN unit, five tie
    conventions. ⚠️ paperC has falsified this null against itself THREE times:
    the tie convention, the character-vs-token length unit, and (within the
    token unit) the TOKENIZER. Only the token unit is recoverable from these
    records. At ~9.5 candidates/item the tie structure is under far more
    pressure than at 4-way, which is one of the two reasons MMLU-Pro was
    chosen."""
    out = {c: np.zeros(len(recs)) for c in CONVS}
    mult = Counter()
    gold_in_win = 0
    for i, r in enumerate(recs):
        ct = r["content_norm"]["cont_tokens"]
        keys = [LETTERS[k] for k in range(r["n_opt"])]
        top = max(ct[k] for k in keys)
        W = [k for k in keys if ct[k] == top]
        g = r["gold_letter"]
        mult[len(W)] += 1
        if g in W:
            gold_in_win += 1
        out["split"][i] = (1.0 / len(W)) if g in W else 0.0
        out["first"][i] = 1.0 if W[0] == g else 0.0
        out["last"][i] = 1.0 if W[-1] == g else 0.0
        out["credit"][i] = 1.0 if g in W else 0.0
        out["wrong"][i] = 1.0 if (len(W) == 1 and W[0] == g) else 0.0
    return out, {
        "length_unit": "continuation_tokens (MMLU's unit; the CHARACTER unit is "
                       "not recoverable from these records -- see README 5a)",
        "winner_set_size_hist": {str(k): mult[k] for k in sorted(mult)},
        "mean_winner_set_size": float(np.mean([
            sum(1 for k in range(r["n_opt"])
                if r["content_norm"]["cont_tokens"][LETTERS[k]] ==
                max(r["content_norm"]["cont_tokens"][LETTERS[j]]
                    for j in range(r["n_opt"])))
            for r in recs])),
        "frac_items_with_tied_longest": 1.0 - mult[1] / len(recs),
        "frac_items_gold_in_winner_set": gold_in_win / len(recs),
    }


def correct_vec(recs, interface):
    return np.array([1.0 if r[interface]["correct"] else 0.0 for r in recs])


def tie_rate(recs, interface):
    n = 0
    for r in recs:
        v = [r[interface]["scores"][LETTERS[k]] for k in range(r["n_opt"])]
        m = max(v)
        if sum(1 for x in v if x == m) > 1:
            n += 1
    return n / len(recs)


def analyse_cell(recs, integ, bc_vec, bc_letter, lo_vecs, rows_sink, key):
    """One arm: every interface vs every admissible null."""
    out = {"shard_integrity": integ, "interfaces": {}}
    for interface, nulls in (
            ("letter", {"best_constant_" + bc_letter: bc_vec}),
            ("content_raw", {"longest_" + c: lo_vecs[c] for c in CONVS}),
            ("content_norm", {"longest_" + c: lo_vecs[c] for c in CONVS})):
        cv = correct_vec(recs, interface)
        hist = Counter(r[interface]["pred_letter"] for r in recs)
        ent = {"acc": float(cv.mean()),
               "modal_pred_share": max(hist.values()) / len(recs),
               "modal_pred_letter": max(hist.items(), key=lambda kv: kv[1])[0],
               "pred_hist": dict(sorted(hist.items())),
               "exact_tie_rate": tie_rate(recs, interface),
               "vs_null": {}}
        for nname, nvec in nulls.items():
            m, lo, hi, p = paired_boot(cv - nvec)
            mc = None
            if set(np.unique(nvec)) <= {0.0, 1.0}:
                b = int(np.sum((cv == 1) & (nvec == 0)))
                c_ = int(np.sum((cv == 0) & (nvec == 1)))
                mc = {"arm_right_null_wrong": b, "arm_wrong_null_right": c_,
                      "mcnemar_exact_p": mcnemar_exact_p(b, c_)}
            half = (100 * hi - 100 * lo) / 2
            ent["vs_null"][nname] = {
                "null": float(nvec.mean()),
                "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi],
                "ci95_half_width_pp": half,
                "boot_p": p, "verdict": verdict(m, p), "mcnemar": mc,
                # THE acceptance criterion of #251
                "could_have_detected_mmlu_effect": bool(
                    half <= abs(MMLU_REFERENCE_EFFECT_PP)),
                "residual_fraction": ((float(cv.mean()) - float(nvec.mean()))
                                      / float(cv.mean())) if cv.mean() > 0 else None,
            }
            r = dict(key)
            r.update({
                "n": integ["n_scored"], "interface": interface,
                "null_name": nname,
                "acc": round(float(cv.mean()), 6),
                "null": round(float(nvec.mean()), 6),
                "delta_pp": round(100 * m, 4),
                "ci95_lo_pp": round(100 * lo, 4),
                "ci95_hi_pp": round(100 * hi, 4),
                "ci95_half_width_pp": round(half, 4),
                "boot_p": p,
                "mcnemar_p": (mc["mcnemar_exact_p"] if mc else ""),
                "verdict": verdict(m, p),
                "could_have_detected_mmlu_effect":
                    ent["vs_null"][nname]["could_have_detected_mmlu_effect"],
                "modal_pred_share": round(ent["modal_pred_share"], 6),
                "exact_tie_rate": round(ent["exact_tie_rate"], 6),
            })
            rows_sink.append(r)
        out["interfaces"][interface] = ent

    # within-arm letter-vs-content pairing (the MMLU headline pair). #248 found
    # confirmed_general[2]'s "+-3 pp" is FALSE off MMLU (arc_easy keep8: +38.76
    # pp) -- so this number is a re-test of a claim already re-scoped once.
    Lc = correct_vec(recs, "letter")
    CN = correct_vec(recs, "content_norm")
    b = int(np.sum((Lc == 1) & (CN == 0)))
    c_ = int(np.sum((Lc == 0) & (CN == 1)))
    m, lo, hi, p = paired_boot(CN - Lc)
    out["content_norm_minus_letter"] = {
        "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi], "boot_p": p,
        "mcnemar_exact_p": mcnemar_exact_p(b, c_),
        "letter_only_correct": b, "content_only_correct": c_,
        "agreement": float(np.mean(Lc == CN)),
    }
    return out


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_root")
    ap.add_argument("out_json")
    ap.add_argument("out_csv", nargs="?", default=None)
    ap.add_argument("--olmo_json", default=None,
                    help="#248 gate2_letter_content_nulls.json")
    ap.add_argument("--xf_json", default=None,
                    help="#250 gate2_crossfamily_nulls.json")
    ap.add_argument("--olmo_records_root", default=None,
                    help="#248 olmo2_mc_letter_content_results/ -- enables the "
                         "pooled 5-benchmark vs 5+MMLU-Pro comparison")
    args = ap.parse_args()

    res = {
        "what": "paperC task #251: MMLU's letter-vs-content contrast on MMLU-Pro "
                "(n=12032, up to 10-way), vs construct-appropriate nulls",
        "why": "#248 and #250 both returned PARTIAL for the SAME reason -- the "
               "five non-MMLU benchmarks are 6-28x smaller than MMLU, so 52/60 "
               "of #250's damaged cells could not have detected MMLU's own "
               "-1.389 pp effect. This is a POWER limit, not a null result "
               "(arc_challenge's median damaged effect -3.840 pp is LARGER than "
               "MMLU's -3.603 pp). No additional SMALL benchmark can fix it: "
               "openbookqa would need ~10615 items and its test set is 500.",
        "acceptance_criterion": {
            "question": "does the achieved CI95 half-width on the "
                        "letter-vs-best-constant-floor test fall below MMLU's "
                        "own effect size of 1.389 pp?",
            "reference_effect_pp": MMLU_REFERENCE_EFFECT_PP,
            "answered_in": "power_verdict",
        },
        "protocol": {
            "chat_template": False,
            "add_bos": 0,
            "weights": "fp32 master, bf16-autocast forward",
            "batch_size": 48,
            "max_len": 1536,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "estimators": ESTIMATOR_SOURCE,
            "note": "batch_size 48 and add_bos 0 are DELIBERATELY the #248/#250 "
                    "values: bf16-autocast batch composition perturbs the "
                    "low-order bits of the summed log-probs, so a different "
                    "batch size is a (small) protocol difference. Measured on "
                    ".73: peak 83.1 GiB of 97.8 (84%), and bs=96 gives NO "
                    "speedup (0.409 vs 0.398 s/item) -- the eval is "
                    "compute-bound, so raising bs buys nothing and costs "
                    "protocol identity.",
        },
        "nulls": {
            "letter": "best-constant letter = argmax of the gold-letter "
                      "marginal (NOT 1/n_opt, NOT 0.10, NOT 0.25)",
            "content": "longest-option, CONTINUATION-TOKEN unit, five tie "
                       "conventions (split canonical)",
        },
        "dataset": {
            "hf_id": "TIGER-Lab/MMLU-Pro",
            "split": "test",
            "n": EXPECTED_N,
            "n_opt_is_not_constant": True,
            "note": "read directly from parquet (no HF builder) so 8 shards "
                    "cannot race and item order is a pure function of the file. "
                    "0/12032 items contain an 'N/A' option and 0/12032 have "
                    "answer_index disagreeing with the answer letter, so the "
                    "full test split is used with no filtering.",
        },
        "reference_power_table_248": REF_POWER_TABLE,
        "olmo2": {},
        "crossfamily": {},
    }
    rows = []

    # ---- letter null invariance. A pure DATASET property, so it MUST be
    #      bit-identical across every arm and every family (all score the
    #      identical item set). Content null is tokenizer-dependent (#250's third
    #      self-falsification) so it is recorded PER TOKENIZER, never shared.
    letter_ref = None
    content_ref = {}

    def do_arm(arm_dir, key, sink_path):
        nonlocal letter_ref
        if not os.path.isdir(os.path.join(args.results_root, arm_dir)):
            return None
        if not glob.glob(os.path.join(args.results_root, arm_dir,
                                      f"per_example_{TASK}_shard*of8.jsonl")):
            return None
        recs, integ = load_records(args.results_root, arm_dir)
        bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
        lo_vecs, lo_diag = longest_option_nulls(recs)
        ref = (bc_letter, round(float(bc_vec.mean()), 12))
        if letter_ref is None:
            letter_ref = ref
            res["letter_null"] = bc_diag
        else:
            assert ref == letter_ref, (
                f"{arm_dir}: LETTER null drifted {ref} != {letter_ref} -> the "
                f"item sets differ, cells are NOT comparable")
        tk = key.get("tokenizer", "olmo2")
        cref = tuple(round(float(lo_vecs[c].mean()), 12) for c in CONVS)
        if tk not in content_ref:
            content_ref[tk] = cref
            res.setdefault("content_nulls_by_tokenizer", {})[tk] = {
                "floor_by_convention": {c: float(lo_vecs[c].mean())
                                        for c in CONVS},
                "diagnostics": lo_diag,
            }
        else:
            assert cref == content_ref[tk], (
                f"{arm_dir}: content null drifted WITHIN one tokenizer -> item "
                f"sets differ")
        cell = analyse_cell(recs, integ, bc_vec, bc_letter, lo_vecs, rows, key)
        sink_path[key["rung"]] = cell
        lv = cell["interfaces"]["letter"]["vs_null"]["best_constant_" + bc_letter]
        print(f"[done] {arm_dir}: letter={cell['interfaces']['letter']['acc']:.6f} "
              f"d={lv['delta_pp']:+.3f}pp hw={lv['ci95_half_width_pp']:.3f} "
              f"p={lv['boot_p']:.4f} {lv['verdict']}")
        return cell

    # ---------------- OLMo-2 six arms (paired with #248 and MMLU) ----------
    olmo_cells = {}
    for rung, arm_dir in OLMO_ARMS:
        do_arm(arm_dir, {"family": "olmo2_7b", "rung": rung, "task": TASK,
                         "arm_dir": arm_dir, "tokenizer": "olmo2",
                         "regime": "prune_then_heal"}, olmo_cells)
    res["olmo2"] = {"n_layers": 32, "regime": "prune-then-heal (121k-200k steps)",
                    "rungs": olmo_cells}

    # ---------------- optional cross-family (#250's 15 arms) --------------
    xf = {}
    for fam in XF_FAMILIES:
        cells = {}
        for rung in XF_RUNGS:
            arm_dir = f"{fam}_{rung}" if rung == "base" else f"{fam}_{rung}"
            do_arm(arm_dir, {"family": fam, "rung": rung, "task": TASK,
                             "arm_dir": arm_dir, "tokenizer": fam,
                             "regime": "eval_time_truncation_no_heal"}, cells)
        if cells:
            xf[fam] = {"n_layers": N_LAYERS[fam],
                       "regime": "eval-time front-N truncation, NO heal",
                       "rungs": cells}
    res["crossfamily"] = xf

    # ---------------- THE POWER VERDICT (the point of this whole task) -----
    def letter_v(cell):
        vn = next(iter(cell["interfaces"]["letter"]["vs_null"]))
        return cell["interfaces"]["letter"]["vs_null"][vn]

    all_cells = [("olmo2_7b", r, c) for r, c in olmo_cells.items()]
    for fam, f in xf.items():
        all_cells += [(fam, r, c) for r, c in f["rungs"].items()]
    hws = [letter_v(c)["ci95_half_width_pp"] for _f, _r, c in all_cells]
    n_pass = sum(1 for h in hws if h <= abs(MMLU_REFERENCE_EFFECT_PP))
    # n needed to reach hw < 1.389 pp, from the achieved hw (hw ~ 1/sqrt(n))
    med_hw = float(np.median(hws)) if hws else None
    n_needed = (EXPECTED_N * (med_hw / abs(MMLU_REFERENCE_EFFECT_PP)) ** 2
                if med_hw else None)
    res["power_verdict"] = {
        "criterion": "achieved CI95 half-width on letter-vs-floor < 1.389 pp "
                     "(MMLU's own effect size)",
        "n": EXPECTED_N,
        "n_cells": len(all_cells),
        "half_width_pp_min": (min(hws) if hws else None),
        "half_width_pp_median": med_hw,
        "half_width_pp_max": (max(hws) if hws else None),
        "n_cells_powered": n_pass,
        "n_cells_underpowered": len(all_cells) - n_pass,
        "WALL_PUSHED_DOWN": bool(hws and max(hws) <= abs(MMLU_REFERENCE_EFFECT_PP)),
        "implied_n_for_hw_below_1p389": (round(n_needed) if n_needed else None),
        "comparison_with_248": {
            t: {"n": v["n"], "half_width_pp": v["half_width_pp"],
                "powered": v["half_width_pp"] <= abs(MMLU_REFERENCE_EFFECT_PP)}
            for t, v in REF_POWER_TABLE.items()},
    }

    # ---------------- roll-up: did MMLU's conclusion replicate? -----------
    def rollup(cells, damaged_set, chance_key):
        ch = res["letter_null"][chance_key]
        floor = res["letter_null"]["best_constant_acc"]
        dmg = [r for r in cells if r in damaged_set]
        return {
            "damaged_rungs": dmg,
            "n_damaged": len(dmg),
            "n_damaged_at_or_below_floor": sum(
                1 for r in dmg if not letter_v(cells[r])["verdict"].startswith("above")),
            "n_damaged_strictly_below_significant": sum(
                1 for r in dmg if letter_v(cells[r])["verdict"].startswith("BELOW")),
            "n_damaged_above_chance_but_not_above_floor": sum(
                1 for r in dmg
                if cells[r]["interfaces"]["letter"]["acc"] > ch
                and not letter_v(cells[r])["verdict"].startswith("above")),
            "n_damaged_negative_point_estimate": sum(
                1 for r in dmg if letter_v(cells[r])["delta_pp"] < 0),
            "n_damaged_underpowered": sum(
                1 for r in dmg
                if not letter_v(cells[r])["could_have_detected_mmlu_effect"]),
            "chance_used": ch, "floor_used": floor,
        }

    res["rollup"] = {
        "chance_definition_used": "chance_mean_1_over_nopt (n_opt is not "
                                  "constant on MMLU-Pro; the naive 0.10 reading "
                                  "is also reported in letter_null and gives a "
                                  "LARGER wrong-null flip count, so the mean "
                                  "reading is the CONSERVATIVE choice)",
        "olmo2_7b": rollup(olmo_cells, DAMAGED_OLMO, "chance_mean_1_over_nopt"),
        "olmo2_7b_naive_chance": rollup(olmo_cells, DAMAGED_OLMO,
                                        "chance_naive_1_over_max_nopt"),
    }
    for fam, f in xf.items():
        res["rollup"][fam] = rollup(f["rungs"], XF_DAMAGED,
                                    "chance_mean_1_over_nopt")
        res["rollup"][fam + "_naive_chance"] = rollup(
            f["rungs"], XF_DAMAGED, "chance_naive_1_over_max_nopt")

    # ---------------- head-to-head with #248 (MMLU vs 5 small vs MMLU-Pro) --
    if args.olmo_json and os.path.exists(args.olmo_json):
        o = json.load(open(args.olmo_json))
        omap = {v: k for k, v in OLMO_ARMS}
        # ⚠️ #248's JSON predates the `ci95_half_width_pp` field: it stores only
        # `ci95_pp`. Derive the half-width from the interval rather than
        # requiring the key, so the head-to-head works against the archived file
        # without rewriting it (the archive is provenance; do not mutate it).
        def _hw(v):
            if "ci95_half_width_pp" in v:
                return v["ci95_half_width_pp"]
            lo, hi = v["ci95_pp"]
            return (hi - lo) / 2

        h2h = {}
        for task, t in o.get("tasks", {}).items():
            for oarm, a in t.get("arms", {}).items():
                rung = omap.get(oarm)
                if rung is None:
                    continue
                bcn = next(iter(a["interfaces"]["letter"]["vs_null"]))
                v = a["interfaces"]["letter"]["vs_null"][bcn]
                hw = _hw(v)
                h2h.setdefault(rung, {})[task] = {
                    "acc": a["interfaces"]["letter"]["acc"],
                    "delta_pp": v["delta_pp"],
                    "ci95_half_width_pp": hw,
                    "boot_p": v["boot_p"], "verdict": v["verdict"],
                    "underpowered": bool(hw > abs(MMLU_REFERENCE_EFFECT_PP)),
                }
        for rung, c in olmo_cells.items():
            v = letter_v(c)
            h2h.setdefault(rung, {})[TASK] = {
                "acc": c["interfaces"]["letter"]["acc"],
                "delta_pp": v["delta_pp"],
                "ci95_half_width_pp": v["ci95_half_width_pp"],
                "boot_p": v["boot_p"], "verdict": v["verdict"],
                "underpowered": not v["could_have_detected_mmlu_effect"],
            }
        res["head_to_head_248"] = h2h

    if args.xf_json and os.path.exists(args.xf_json):
        x = json.load(open(args.xf_json))
        res["reference_250_pooled"] = x.get("pooled_across_tasks_letter_floor",
                                            {}).get("by_family")
        res["reference_250_content_tokenizer_dependence"] = x.get(
            "content_null_tokenizer_dependence", {}).get("by_task", {}).get(
                "arc_challenge")

    # ---------------- POOLED: MMLU-Pro + the five #248 benchmarks -----------
    # MMLU-Pro's items are DISJOINT from all five #248 benchmarks (different
    # sources entirely), and the letter null is a per-item input-blind 0/1
    # vector on each, so concatenating the per-item difference vectors is the
    # SAME estimator on a larger item set -- no new modelling assumption. This
    # is #250's construction with MMLU-Pro added, taking n from 7107 to 19139.
    #
    # ⚠️ MUST NOT be quoted as a per-benchmark verdict: the pooled floor mixes
    # six floors spanning 0.1166 (mmlu_pro) to 0.5049 (piqa), so what is pooled
    # is the DEVIATION FROM EACH BENCHMARK'S OWN FLOOR, not raw accuracy.
    if args.olmo_records_root and os.path.isdir(args.olmo_records_root):
        FIVE = ["arc_challenge", "arc_easy", "openbookqa", "commonsense_qa",
                "piqa"]  # winogrande EXCLUDED: negative control
        FIVE_N = {"arc_challenge": 1172, "arc_easy": 2376, "openbookqa": 500,
                  "commonsense_qa": 1221, "piqa": 1838}
        pooled = {}
        for rung, arm_dir in OLMO_ARMS:
            dv, cvs, nvs, per_task_n = [], [], [], {}
            ok = True
            for task in FIVE:
                shards = sorted(glob.glob(os.path.join(
                    args.olmo_records_root, arm_dir,
                    f"per_example_{task}_shard*of8.jsonl")))
                if len(shards) != 8:
                    ok = False
                    break
                recs = []
                for s in shards:
                    for line in open(s):
                        line = line.strip()
                        if line:
                            recs.append(json.loads(line))
                assert len(recs) == FIVE_N[task], (arm_dir, task, len(recs))
                assert sum(1 for r in recs if r["nan"]) == 0, (arm_dir, task)
                recs.sort(key=lambda r: r["item_id"])
                _bl, bvec, _d = best_constant_letter(recs)
                cv = correct_vec(recs, "letter")
                cvs.append(cv); nvs.append(bvec); dv.append(cv - bvec)
                per_task_n[task] = len(recs)
            if not ok:
                continue
            # add MMLU-Pro
            recs, _ = load_records(args.results_root, arm_dir)
            _bl, bvec, _d = best_constant_letter(recs)
            cv = correct_vec(recs, "letter")
            cvs.append(cv); nvs.append(bvec); dv.append(cv - bvec)
            per_task_n[TASK] = len(recs)

            for label, keep in (("five_only", FIVE),
                                ("five_plus_mmlu_pro", FIVE + [TASK])):
                idx = [i for i, t in enumerate(FIVE + [TASK]) if t in keep]
                d_all = np.concatenate([dv[i] for i in idx])
                cv_all = np.concatenate([cvs[i] for i in idx])
                nv_all = np.concatenate([nvs[i] for i in idx])
                m, lo, hi, p = paired_boot(d_all)
                b = int(np.sum((cv_all == 1) & (nv_all == 0)))
                c_ = int(np.sum((cv_all == 0) & (nv_all == 1)))
                half = (100 * hi - 100 * lo) / 2
                pooled.setdefault(rung, {})[label] = {
                    "tasks_pooled": keep,
                    "n_pooled": int(d_all.size),
                    "per_task_n": {t: per_task_n[t] for t in keep},
                    "pooled_acc": float(cv_all.mean()),
                    "pooled_floor": float(nv_all.mean()),
                    "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi],
                    "ci95_half_width_pp": half, "boot_p": p,
                    "mcnemar_exact_p": mcnemar_exact_p(b, c_),
                    "verdict": verdict(m, p),
                    "could_have_detected_mmlu_effect": bool(
                        half <= abs(MMLU_REFERENCE_EFFECT_PP)),
                }
            pf = pooled[rung]["five_plus_mmlu_pro"]
            p5 = pooled[rung]["five_only"]
            print(f"[pooled] {rung}: 5-only n={p5['n_pooled']} "
                  f"d={p5['delta_pp']:+.3f} hw={p5['ci95_half_width_pp']:.3f} "
                  f"p={p5['boot_p']:.4f} | +mmlu_pro n={pf['n_pooled']} "
                  f"d={pf['delta_pp']:+.3f} hw={pf['ci95_half_width_pp']:.3f} "
                  f"p={pf['boot_p']:.4f}")
        res["pooled_olmo2"] = {
            "why": "MMLU-Pro's items are DISJOINT from all five #248 benchmarks "
                   "and the letter null is a per-item input-blind 0/1 vector on "
                   "each, so the concatenated paired difference is the SAME "
                   "estimator on a larger item set. #250's construction with "
                   "MMLU-Pro added.",
            "must_not_be_quoted_as": "a per-benchmark verdict. The pooled floor "
                                     "mixes six floors spanning 0.1166 "
                                     "(mmlu_pro) to 0.5049 (piqa), so what is "
                                     "pooled is the DEVIATION FROM EACH "
                                     "BENCHMARK'S OWN FLOOR, not raw accuracy.",
            "excluded": ["winogrande (negative control)"],
            "by_rung": pooled,
        }

    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2, sort_keys=False)
    print(f"wrote {args.out_json}")
    if args.out_csv and rows:
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv} ({len(rows)} rows)")

    pv = res["power_verdict"]
    print()
    print("=" * 72)
    print(f"POWER VERDICT: n={pv['n']}  cells={pv['n_cells']}")
    print(f"  CI95 half-width pp: min={pv['half_width_pp_min']:.4f} "
          f"median={pv['half_width_pp_median']:.4f} max={pv['half_width_pp_max']:.4f}")
    print(f"  criterion (< {abs(MMLU_REFERENCE_EFFECT_PP)} pp): "
          f"{pv['n_cells_powered']}/{pv['n_cells']} powered")
    print(f"  WALL PUSHED DOWN: {pv['WALL_PUSHED_DOWN']}")
    ln = res["letter_null"]
    print(f"  floor = always-{ln['best_constant_letter']} "
          f"{ln['best_constant_acc']:.6f}  vs naive chance "
          f"{ln['chance_naive_1_over_max_nopt']:.6f} "
          f"(+{ln['floor_minus_chance_naive_pp']:.3f} pp, "
          f"{ln['floor_over_chance_naive']:.4f}x)  vs mean 1/n_opt "
          f"{ln['chance_mean_1_over_nopt']:.6f} "
          f"(+{ln['floor_minus_chance_mean_pp']:.3f} pp)")
    print("=" * 72)


if __name__ == "__main__":
    main()
