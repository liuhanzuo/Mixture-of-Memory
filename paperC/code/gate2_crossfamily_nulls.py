#!/usr/bin/env python3
"""paperC gate-2 CROSS-FAMILY: construct-appropriate nulls for the
letter-vs-content MC interface on non-MMLU benchmarks, in NON-OLMo families.

What this closes
----------------
`paperC/README.md` open defect 2 was closed by task #248, but only inside ONE
model family. MMLU's headline is four-family; the second-benchmark leg was
OLMo-2-only. This script is the analysis half of the cross-family extension
(task #250): it consumes the per-item records written by
`scripts/eval_olmo2_mc_letter_content.py` for Llama-2-7B / Llama-3-8B /
Qwen3-8B-Base (intact + front-N truncated), and computes the same nulls and the
same statistics `a01_gate2_letter_content_nulls.py` computed for OLMo-2, so the
two tables are directly comparable cell for cell.

It ALSO recomputes the MMLU cross-family letter floor verdicts from the archived
gate-1 DAMAGED per-item records (`olmo2_mmlu_content_results/gate1_*`), because
`STATUS.json:gate1_third_model_family_DAMAGED` only recorded point deltas vs
`0.2689` with no CI / p-value, and the p-values it does have elsewhere predate
the R-7 mid-p fix. Same estimators, one script, so "did MMLU's conclusion
replicate on a second benchmark, in this family" is a like-for-like question.

Nulls (identical definitions to the OLMo-2 run)
----------------------------------------------
  letter  -> BEST-CONSTANT LETTER, i.e. argmax of the gold-letter marginal. NOT
             1/n_opt. Per task; the floors differ per benchmark and MUST NOT be
             replaced by 0.25 (see STATUS.json:must_not_resurrect).
  content -> LONGEST-OPTION null under all FIVE tie conventions
             (split / first / last / credit / wrong). The convention is a real
             degree of freedom that can reverse verdicts.

⚠️ LENGTH UNIT. The longest-option null is under-specified in TWO ways: the tie
convention AND the length unit (characters vs continuation tokens). The per-item
records store `cont_tokens` only, so this script reports the **token** unit —
MMLU's unit, and the unit #248 used. The character unit is NOT recoverable from
these records and is therefore not reported here rather than being guessed.

Statistics: paired bootstrap n_boot=10000 seed=7, two-sided p from the R-7-fixed
mid-p `two_sided_boot_p`, exact McNemar against the deterministic constant
predictor. Estimators are imported from the A01 code directory and, if that
import is unavailable, fall back to verbatim local copies whose agreement with
the imported versions is asserted when both are present.

POWER. Four of five non-MMLU tasks were underpowered in the OLMo-2 run to detect
MMLU's own −1.389 pp letter effect. Every letter cell here therefore carries its
CI95 half-width and an explicit `could_have_detected_mmlu_effect` flag. A null
result without that flag is not interpretable.

CPU only. No GPU, no model load.

Usage:
  python gate2_crossfamily_nulls.py <xf_results_root> <out_json> [out_csv]
      [--mmlu_root olmo2_mmlu_content_results] [--olmo_json <#248 json>]
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
BOOT_SEED = 7  # same seed as the OLMo-2 run, so the two tables are comparable

TASKS = ["arc_challenge", "arc_easy", "openbookqa", "commonsense_qa",
         "piqa", "winogrande"]
EXPECTED_N = {"arc_challenge": 1172, "arc_easy": 2376, "openbookqa": 500,
              "commonsense_qa": 1221, "piqa": 1838, "winogrande": 1267}
CONTROLS = {"winogrande"}  # never counted as interface evidence

FAMILIES = ["llama2_7b", "llama3_8b", "qwen3_8b_base"]
# increasing damage, matching the OLMo-2 six-arm ordering convention
RUNGS = ["base", "k14", "k12", "k10", "k8"]
N_LAYERS = {"llama2_7b": 32, "llama3_8b": 32, "qwen3_8b_base": 36}

# The MMLU cross-family arms already on disk (gate-1 INTACT + DAMAGED legs).
MMLU_ARM_MAP = {
    ("llama2_7b", "base"): "gate1_llama2_7b",
    ("llama3_8b", "base"): "gate1_llama3_8b",
    ("qwen3_8b_base", "base"): "gate1_qwen3_8b_base",
    ("llama2_7b", "k8"): "gate1_dmg_llama2_7b_k8",
    ("llama3_8b", "k8"): "gate1_dmg_llama3_8b_k8",
    ("qwen3_8b_base", "k8"): "gate1_dmg_qwen3_8b_k8",
    ("llama2_7b", "k12"): "gate1_dmg_llama2_7b_k12",
    ("llama3_8b", "k12"): "gate1_dmg_llama3_8b_k12",
    ("qwen3_8b_base", "k12"): "gate1_dmg_qwen3_8b_k12",
    # k10/k14 exist for llama2 only, under the depth-sweep naming
    ("llama2_7b", "k10"): "gate1_dmg_llama2_7b_depth_k10",
    ("llama2_7b", "k14"): "gate1_dmg_llama2_7b_depth_k14",
    ("llama3_8b", "k14"): "gate1_dmg_llama3_8b_depth_fine_21_k14",
}
MMLU_EXPECTED_N = 14042
# MMLU's own headline effect on OLMo-2 keep8: letter 0.2550 vs always-D 0.2689.
MMLU_REFERENCE_EFFECT_PP = -1.389


# ---------------------------------------------------------------------------
# estimators. Imported from the A01 code dir for provenance; local verbatim
# copies are kept so this script does not break if that tree is reorganised, and
# the two are asserted equal whenever both are available.
# ---------------------------------------------------------------------------
def _two_sided_boot_p_local(bs, n_boot=None):
    """R-7-fixed mid-p two-sided bootstrap p (verbatim from
    a01_gate3_fp32_vs_bf16.py::two_sided_boot_p). Splits the atom at zero evenly
    between the tails so the two tails sum to 1 and p <= 1 is structural."""
    bs = np.asarray(bs, dtype=np.float64)
    if n_boot is None:
        n_boot = bs.size
    tie = float((bs == 0).mean())
    p_lo = float((bs < 0).mean()) + 0.5 * tie
    p_hi = float((bs > 0).mean()) + 0.5 * tie
    p = 2.0 * min(p_lo, p_hi)
    return float(min(1.0, max(p, 1.0 / n_boot)))


def _mcnemar_exact_p_local(b, c):
    """Exact two-sided McNemar (verbatim from a01_gate3_fp32_vs_bf16.py)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    lh = n * math.log(0.5)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + lh
             for i in range(k + 1)]
    m = max(terms)
    return min(1.0, 2.0 * math.exp(m + math.log(sum(math.exp(t - m) for t in terms))))


_A01 = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))),
    "proposal/active/A01-null-calibration-methodology/code")
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
        # agreement check on a deliberately adversarial case: the base-arm-like
        # situation with a large atom at exactly zero (the R-7 bug's trigger).
        _rng = np.random.default_rng(0)
        _bs = np.where(_rng.random(5000) < 0.4, 0.0, _rng.normal(0, 1e-4, 5000))
        assert abs(_bp_imported(_bs, 5000) - _two_sided_boot_p_local(_bs, 5000)) < 1e-12, \
            "two_sided_boot_p: imported and local copies DISAGREE"
        for _b, _c in ((0, 0), (1, 0), (28, 28), (300, 250)):
            assert abs(_mc_imported(_b, _c) - _mcnemar_exact_p_local(_b, _c)) < 1e-12, \
                f"mcnemar_exact_p: imported and local disagree at ({_b},{_c})"
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
# load, WITH shard-integrity asserts (never analyse a partial set)
# ---------------------------------------------------------------------------
def load_records(root, arm, stem, expected_n, num_shards=8):
    """`stem` is the per_example file stem: the task name for the non-MMLU
    harness, 'mmlu' for the MMLU harness."""
    d = os.path.join(root, arm)
    shards = sorted(glob.glob(
        os.path.join(d, f"per_example_{stem}_shard*of{num_shards}.jsonl")))
    idx = {int(os.path.basename(s).split("_shard")[1].split("of")[0])
           for s in shards}
    missing = sorted(set(range(num_shards)) - idx)
    assert not missing, f"{arm}/{stem}: MISSING shards {missing} of {num_shards}"
    recs = []
    for s in shards:
        with open(s) as f:
            for line in f:
                line = line.strip()
                if line:
                    recs.append(json.loads(line))
    ids = [r["item_id"] for r in recs]
    assert len(set(ids)) == len(ids), f"{arm}/{stem}: duplicate item_id"
    assert len(recs) == expected_n, \
        f"{arm}/{stem}: n_scored={len(recs)} != {expected_n}"
    n_nan = sum(1 for r in recs if r["nan"])
    assert n_nan == 0, f"{arm}/{stem}: n_nan={n_nan} != 0"
    recs.sort(key=lambda r: r["item_id"])
    return recs, {"n_shards": len(shards), "n_scored": len(recs),
                  "expected_n": expected_n, "n_nan": n_nan}


# ---------------------------------------------------------------------------
# nulls (definitions verbatim from a01_gate2_letter_content_nulls.py)
# ---------------------------------------------------------------------------
def best_constant_letter(recs):
    gold = Counter(r["gold_letter"] for r in recs)
    best, cnt = max(gold.items(), key=lambda kv: (kv[1], -LETTERS.index(kv[0])))
    vec = np.array([1.0 if r["gold_letter"] == best else 0.0 for r in recs])
    return best, vec, {
        "gold_letter_marginal": dict(sorted(gold.items())),
        "best_constant_letter": best,
        "best_constant_acc": float(vec.mean()),
        "chance_1_over_nopt": float(np.mean([1.0 / r["n_opt"] for r in recs])),
        "floor_minus_chance_pp": 100 * (
            float(vec.mean()) - float(np.mean([1.0 / r["n_opt"] for r in recs]))),
        "sanity_best_constant_eq_count_over_n": abs(
            float(vec.mean()) - cnt / len(recs)) < 1e-12,
    }


def longest_option_nulls(recs):
    """Longest-option null, CONTINUATION-TOKEN length unit, five tie conventions."""
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


# ---------------------------------------------------------------------------
def analyse_cell(recs, integ, bc_vec, bc_letter, lo_vecs, rows_sink, key):
    """One (family, rung, task) cell: every interface vs every admissible null."""
    a_out = {"shard_integrity": integ, "interfaces": {}}
    for interface, nulls in (
            ("letter", {"best_constant_" + bc_letter: bc_vec}),
            ("content_raw", {"longest_" + c: lo_vecs[c] for c in CONVS}),
            ("content_norm", {"longest_" + c: lo_vecs[c] for c in CONVS})):
        cv = correct_vec(recs, interface)
        ent = {"acc": float(cv.mean()),
               "modal_pred_share": max(Counter(
                   r[interface]["pred_letter"] for r in recs).values()) / len(recs),
               "pred_hist": dict(sorted(Counter(
                   r[interface]["pred_letter"] for r in recs).items())),
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
                # a null result is only interpretable with this flag
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
        a_out["interfaces"][interface] = ent

    # within-arm letter-vs-content pairing (the MMLU headline pair)
    Lc = correct_vec(recs, "letter")
    CN = correct_vec(recs, "content_norm")
    b = int(np.sum((Lc == 1) & (CN == 0)))
    c_ = int(np.sum((Lc == 0) & (CN == 1)))
    m, lo, hi, p = paired_boot(CN - Lc)
    a_out["content_norm_minus_letter"] = {
        "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi], "boot_p": p,
        "mcnemar_exact_p": mcnemar_exact_p(b, c_),
        "letter_only_correct": b, "content_only_correct": c_,
        "agreement": float(np.mean(Lc == CN)),
    }
    return a_out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xf_root")
    ap.add_argument("out_json")
    ap.add_argument("out_csv", nargs="?", default=None)
    ap.add_argument("--mmlu_root", default="olmo2_mmlu_content_results")
    ap.add_argument("--olmo_json", default=None,
                    help="#248 OLMo-2 gate2_letter_content_nulls.json, for the "
                         "OLMo-vs-non-OLMo head-to-head")
    args = ap.parse_args()

    res = {
        "what": "paperC gate-2 CROSS-FAMILY: MMLU's letter-vs-content contrast on "
                "non-MMLU MC benchmarks, in three NON-OLMo families, vs "
                "construct-appropriate nulls",
        "closes": "the one-family scope of paperC/README.md open defect 2 "
                  "(#248 closed the contrast, in OLMo-2 only)",
        "damage": "front-N truncation applied AT EVAL TIME by "
                  "eval_olmo2_probe2_ppl.py::load_truncated_any_family: "
                  "model.model.layers <- layers[:N], config synced. NO fresh "
                  "block, NO heal steps, NO training. Identical construction to "
                  "the archived MMLU gate-1 DAMAGED leg, which is why the k8/k12 "
                  "rungs are directly comparable with it.",
        "protocol": {
            "chat_template": False, "add_bos": 0,
            "weights": "fp32 master, bf16-autocast forward", "batch_size": 48,
            "n_boot": N_BOOT, "boot_seed": BOOT_SEED,
            "estimators": ESTIMATOR_SOURCE,
        },
        "nulls": {
            "letter": "best-constant letter = argmax of the gold-letter marginal "
                      "(NOT 1/n_opt, NOT 0.25)",
            "content": "longest-option, CONTINUATION-TOKEN unit, five tie "
                       "conventions (split canonical)",
        },
        "depth_caveat": "OLMo-2-7B / Llama-2-7B / Llama-3-8B have 32 blocks, "
                        "Qwen3-8B-Base has 36, so kN is the same ABSOLUTE depth "
                        "but a smaller FRACTION of Qwen3's stack (k8 = 22.2% vs "
                        "25.0%). Absolute N was kept to stay comparable with the "
                        "archived MMLU cross-family numbers.",
        "negative_controls": sorted(CONTROLS),
        "mmlu_reference_effect_pp": MMLU_REFERENCE_EFFECT_PP,
        "families": {},
        "mmlu_crossfamily": {},
    }
    rows = []

    # ---------------- non-MMLU cross-family ----------------
    # NULL INVARIANCE, and where it BREAKS (finding of this run).
    #   * the LETTER null (argmax of the gold-letter marginal) is a pure DATASET
    #     property -> it must be bit-identical across every arm AND every family,
    #     since all of them score the identical item set. Asserted hard.
    #   * the CONTENT longest-option null in the CONTINUATION-TOKEN unit is NOT a
    #     pure dataset property: "longest" is measured in tokens, and the three
    #     families have three different tokenizers, so the winner set (and hence
    #     the floor) legitimately DIFFERS BY FAMILY. Measured on arc_challenge:
    #     Llama-2 0.268871 / Llama-3 0.283902 / Qwen3 0.282338 -- a 1.50 pp spread,
    #     with the tied-longest fraction moving 41.6% -> 50.8%. This is a THIRD
    #     under-specification of the longest-option null, on top of the two
    #     paperC/README.md already documents (tie convention, character-vs-token
    #     unit): within the token unit the null is TOKENIZER-dependent. It is
    #     therefore recorded PER FAMILY, never shared, and the invariance assert
    #     applies only to the arms of one family.
    letter_null_ref = {}
    content_null_ref = {}
    for fam in FAMILIES:
        f_out = {"n_layers": N_LAYERS[fam], "rungs": {}}
        for rung in RUNGS:
            arm = f"{fam}_{rung}" if rung == "base" else f"{fam}_{rung}"
            if not os.path.isdir(os.path.join(args.xf_root, arm)):
                continue
            r_out = {"arm_dir": arm, "tasks": {}}
            for task in TASKS:
                if not glob.glob(os.path.join(args.xf_root, arm,
                                              f"per_example_{task}_shard*of8.jsonl")):
                    continue
                recs, integ = load_records(args.xf_root, arm, task,
                                           EXPECTED_N[task])
                bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
                lo_vecs, lo_diag = longest_option_nulls(recs)

                # letter null: dataset property -> identical everywhere
                lref = (bc_letter, round(float(bc_vec.mean()), 12))
                if task not in letter_null_ref:
                    letter_null_ref[task] = lref
                    res.setdefault("task_letter_nulls", {})[task] = {
                        "is_negative_control": task in CONTROLS,
                        "n": integ["n_scored"],
                        "invariance": "pure dataset property; asserted identical "
                                      "across all 3 families x 5 rungs",
                        **bc_diag,
                    }
                else:
                    assert lref == letter_null_ref[task], (
                        f"{fam}/{rung}/{task}: LETTER null drifted "
                        f"{lref} != {letter_null_ref[task]} -> the item sets "
                        f"differ, cells are not comparable")

                # content null: tokenizer-dependent -> per family, invariant only
                # across the rungs of one family (same tokenizer)
                cref = tuple(round(float(lo_vecs[c].mean()), 12) for c in CONVS)
                if fam not in content_null_ref:
                    content_null_ref[fam] = {}
                if task not in content_null_ref[fam]:
                    content_null_ref[fam][task] = cref
                    res.setdefault("content_nulls_by_family", {}).setdefault(
                        fam, {})[task] = {
                            "floor_by_convention": {
                                c: float(lo_vecs[c].mean()) for c in CONVS},
                            "diagnostics": lo_diag}
                else:
                    assert cref == content_null_ref[fam][task], (
                        f"{fam}/{rung}/{task}: content null drifted WITHIN a "
                        f"family (same tokenizer) -> item sets differ")

                key = {"family": fam, "rung": rung, "task": task,
                       "is_control": task in CONTROLS, "benchmark_group": "non_mmlu"}
                r_out["tasks"][task] = analyse_cell(
                    recs, integ, bc_vec, bc_letter, lo_vecs, rows, key)
            f_out["rungs"][rung] = r_out
            print(f"[done] {fam}/{rung}: {len(r_out['tasks'])} tasks")
        res["families"][fam] = f_out

    # quantify the tokenizer-dependence of the content null, per task
    tok_dep = {}
    for task in TASKS:
        by_fam = {f: res.get("content_nulls_by_family", {}).get(f, {}).get(task)
                  for f in FAMILIES}
        by_fam = {f: v for f, v in by_fam.items() if v}
        if len(by_fam) < 2:
            continue
        tok_dep[task] = {}
        for conv in CONVS:
            vals = {f: v["floor_by_convention"][conv] for f, v in by_fam.items()}
            tok_dep[task][conv] = {
                "by_family": vals,
                "spread_pp": 100 * (max(vals.values()) - min(vals.values()))}
        tok_dep[task]["tied_longest_fraction_by_family"] = {
            f: v["diagnostics"]["frac_items_with_tied_longest"]
            for f, v in by_fam.items()}
    res["content_null_tokenizer_dependence"] = {
        "finding": "the longest-option content null in the CONTINUATION-TOKEN "
                   "unit is TOKENIZER-dependent, i.e. it is NOT a pure dataset "
                   "property. This is a THIRD under-specification on top of the "
                   "tie convention and the character-vs-token length unit that "
                   "paperC/README.md already documents.",
        "by_task": tok_dep,
    }

    # ---------------- MMLU cross-family, same estimators ----------------
    mmlu_ref = None
    for (fam, rung), arm in sorted(MMLU_ARM_MAP.items()):
        if not glob.glob(os.path.join(args.mmlu_root, arm,
                                      "per_example_mmlu_shard*of8.jsonl")):
            continue
        recs, integ = load_records(args.mmlu_root, arm, "mmlu", MMLU_EXPECTED_N)
        bc_letter, bc_vec, bc_diag = best_constant_letter(recs)
        lo_vecs, _ = longest_option_nulls(recs)
        ref = (bc_letter, round(float(bc_vec.mean()), 12))
        if mmlu_ref is None:
            mmlu_ref = ref
            res["mmlu_letter_null"] = bc_diag
        else:
            assert ref == mmlu_ref, f"{arm}: MMLU letter null drifted"
        key = {"family": fam, "rung": rung, "task": "mmlu",
               "is_control": False, "benchmark_group": "mmlu"}
        res["mmlu_crossfamily"].setdefault(fam, {})[rung] = analyse_cell(
            recs, integ, bc_vec, bc_letter, lo_vecs, rows, key)
        print(f"[done] MMLU {fam}/{rung} ({arm})")

    # ---------------- roll-up: replicate or not ----------------
    def letter_v(cell):
        vn = next(iter(cell["interfaces"]["letter"]["vs_null"]))
        return cell["interfaces"]["letter"]["vs_null"][vn]

    roll = {}
    for fam in FAMILIES:
        per_task = {}
        for task in TASKS:
            per_rung = {}
            for rung in RUNGS:
                c = res["families"].get(fam, {}).get("rungs", {}).get(
                    rung, {}).get("tasks", {}).get(task)
                if c is None:
                    continue
                v = letter_v(c)
                per_rung[rung] = {
                    "acc": c["interfaces"]["letter"]["acc"],
                    "delta_pp": v["delta_pp"], "boot_p": v["boot_p"],
                    "mcnemar_p": v["mcnemar"]["mcnemar_exact_p"] if v["mcnemar"] else None,
                    "verdict": v["verdict"],
                    "ci95_half_width_pp": v["ci95_half_width_pp"],
                    "could_have_detected_mmlu_effect":
                        v["could_have_detected_mmlu_effect"],
                    "modal_pred_share": c["interfaces"]["letter"]["modal_pred_share"],
                    "exact_tie_rate": c["interfaces"]["letter"]["exact_tie_rate"],
                }
            if per_rung:
                dmg = [r for r in ("k8", "k10", "k12") if r in per_rung]
                per_task[task] = {
                    "is_negative_control": task in CONTROLS,
                    "per_rung": per_rung,
                    "n_damaged_arms_at_or_below_floor": sum(
                        1 for r in dmg if not per_rung[r]["verdict"].startswith("above")),
                    "n_damaged_arms_strictly_below_sig": sum(
                        1 for r in dmg if per_rung[r]["verdict"].startswith("BELOW")),
                    "n_damaged_arms_above_chance_but_not_above_floor": sum(
                        1 for r in dmg
                        if per_rung[r]["acc"] >
                        res["task_letter_nulls"][task]["chance_1_over_nopt"]
                        and not per_rung[r]["verdict"].startswith("above")),
                    "n_damaged_arms_underpowered": sum(
                        1 for r in dmg
                        if not per_rung[r]["could_have_detected_mmlu_effect"]),
                }
        roll[fam] = per_task
    res["rollup_letter_floor_by_family"] = roll

    # ---------------- POOLED-ACROSS-TASKS test (recovers MMLU's power) --------
    # WHY THIS IS LEGITIMATE, and what it is NOT.
    # The per-task nulls above are underpowered: MMLU's own headline effect is
    # -1.389 pp and four of five non-MMLU tasks have CI95 half-widths of
    # 2.4-6.4 pp, so "n.s." there is uninformative about MMLU's effect size.
    # BUT the five evidence tasks are DISJOINT item sets, and the letter null is
    # a per-item input-blind 0/1 vector on each of them. Concatenating the five
    # per-item difference vectors (model_correct - null_correct) therefore gives
    # ONE paired sample of 7107 items -- comparable in size with MMLU's 14042 --
    # and the paired bootstrap over it is exactly the same estimator, just on a
    # larger item set. No new modelling assumption is introduced.
    #
    # What it IS: a test of "is this arm's letter interface, aggregated over a
    # 5-benchmark MC suite, at or below the suite's own best-constant floor".
    # What it is NOT: a per-benchmark claim. The pooled floor is a weighted mix
    # of five different floors (0.2088 to 0.5049) and the pooled accuracy is a
    # weighted mix of five accuracies; a pooled verdict must never be quoted as
    # if it held on any single benchmark. Winogrande is EXCLUDED (control).
    EVIDENCE = [t for t in TASKS if t not in CONTROLS]
    pooled = {}
    for fam in FAMILIES:
        pooled[fam] = {}
        for rung in RUNGS:
            fr = res["families"].get(fam, {}).get("rungs", {}).get(rung)
            if not fr:
                continue
            dv, cvs, nvs, per_task_n = [], [], [], {}
            for task in EVIDENCE:
                if task not in fr["tasks"]:
                    continue
                arm = f"{fam}_{rung}"
                recs, _ = load_records(args.xf_root, arm, task, EXPECTED_N[task])
                _bl, bvec, _d = best_constant_letter(recs)
                cv = correct_vec(recs, "letter")
                cvs.append(cv)
                nvs.append(bvec)
                dv.append(cv - bvec)
                per_task_n[task] = len(recs)
            if not dv:
                continue
            d_all = np.concatenate(dv)
            cv_all = np.concatenate(cvs)
            nv_all = np.concatenate(nvs)
            m, lo, hi, p = paired_boot(d_all)
            b = int(np.sum((cv_all == 1) & (nv_all == 0)))
            c_ = int(np.sum((cv_all == 0) & (nv_all == 1)))
            half = (100 * hi - 100 * lo) / 2
            pooled[fam][rung] = {
                "n_pooled": int(d_all.size), "per_task_n": per_task_n,
                "pooled_acc": float(cv_all.mean()),
                "pooled_floor": float(nv_all.mean()),
                "delta_pp": 100 * m, "ci95_pp": [100 * lo, 100 * hi],
                "ci95_half_width_pp": half, "boot_p": p,
                "mcnemar_exact_p": mcnemar_exact_p(b, c_),
                "arm_right_null_wrong": b, "arm_wrong_null_right": c_,
                "verdict": verdict(m, p),
                "could_have_detected_mmlu_effect": bool(
                    half <= abs(MMLU_REFERENCE_EFFECT_PP)),
            }
            print(f"[pooled] {fam}/{rung}: n={d_all.size} "
                  f"d={100*m:+.3f}pp p={p:.4f} {verdict(m, p)}")
    res["pooled_across_tasks_letter_floor"] = {
        "why": "the five evidence tasks are DISJOINT item sets and the letter "
               "null is a per-item input-blind 0/1 vector on each, so the "
               "concatenated paired difference is the same estimator on a larger "
               "item set (n=7107, comparable with MMLU's 14042). This recovers "
               "the power the per-task tests lack.",
        "must_not_be_quoted_as": "a per-benchmark verdict. The pooled floor is a "
                                 "weighted mix of five different floors (0.2088 "
                                 "to 0.5049).",
        "tasks_pooled": EVIDENCE,
        "excluded": sorted(CONTROLS),
        "by_family": pooled,
    }

    # MMLU roll-up in the same shape
    res["rollup_mmlu_letter_floor"] = {
        fam: {rung: {"acc": c["interfaces"]["letter"]["acc"],
                     "delta_pp": letter_v(c)["delta_pp"],
                     "boot_p": letter_v(c)["boot_p"],
                     "mcnemar_p": (letter_v(c)["mcnemar"]["mcnemar_exact_p"]
                                   if letter_v(c)["mcnemar"] else None),
                     "verdict": letter_v(c)["verdict"],
                     "ci95_half_width_pp": letter_v(c)["ci95_half_width_pp"],
                     "modal_pred_share": c["interfaces"]["letter"]["modal_pred_share"],
                     "exact_tie_rate": c["interfaces"]["letter"]["exact_tie_rate"]}
              for rung, c in rungs.items()}
        for fam, rungs in res["mmlu_crossfamily"].items()}

    # ---------------- head-to-head with OLMo-2 (#248) ----------------
    if args.olmo_json and os.path.exists(args.olmo_json):
        olmo = json.load(open(args.olmo_json))
        o_map = {"7B_keep8_step121000": "k8", "7B_keep10_step83500": "k10",
                 "7B_keep12_step124000": "k12", "7B_keep14_step200000": "k14",
                 "7B_base": "base"}
        h2h = {}
        for task, t in olmo.get("tasks", {}).items():
            bcname = None
            per_rung = {}
            for oarm, rung in o_map.items():
                a = t.get("arms", {}).get(oarm)
                if a is None:
                    continue
                if bcname is None:
                    bcname = next(iter(a["interfaces"]["letter"]["vs_null"]))
                v = a["interfaces"]["letter"]["vs_null"][bcname]
                per_rung[rung] = {"olmo2_acc": a["interfaces"]["letter"]["acc"],
                                  "olmo2_delta_pp": v["delta_pp"],
                                  "olmo2_boot_p": v["boot_p"],
                                  "olmo2_verdict": v["verdict"]}
                for fam in FAMILIES:
                    c = res["families"].get(fam, {}).get("rungs", {}).get(
                        rung, {}).get("tasks", {}).get(task)
                    if c is not None:
                        lv = letter_v(c)
                        per_rung[rung][fam] = {
                            "acc": c["interfaces"]["letter"]["acc"],
                            "delta_pp": lv["delta_pp"], "boot_p": lv["boot_p"],
                            "verdict": lv["verdict"],
                            "underpowered_for_mmlu_effect":
                                not lv["could_have_detected_mmlu_effect"]}
            if per_rung:
                h2h[task] = per_rung
        res["olmo2_vs_nonolmo_head_to_head"] = h2h

    with open(args.out_json, "w") as f:
        json.dump(res, f, indent=2)
    print(f"wrote {args.out_json}")
    if args.out_csv and rows:
        keys = list(rows[0].keys())
        with open(args.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
