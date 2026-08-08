#!/usr/bin/env python3
"""
paperF_bs_ladder_analysis.py  (2026-08-08)
==========================================
Paper F -- "Direction A" verdict experiment.

PART 1: Enrich 7B_shortgpt16_step200000_bs8 with norm_scores (if missing),
        then compute bs8 vs bs16 flip analysis for ShortGPT-16.

PART 2: Assumes bs16 GPU eval has been run for all 6 rungs. Checks for data.

PART 3: Full statistical analysis:
  - Per-rung acc_norm flip counts (bs8 vs bs16)
  - Spearman(core6, flip_count) with exact permutation p (n=6)
  - margin → P(flip) bucketed curve (acc_norm scale)
  - LOO mediation check: does margin distribution predict flip rates
    across rungs better than a constant-rate null?
  - Explicit comparison with constant-rate null (ΔAIC / LOO deviation)
  - Zero-effect sanity: bs8 vs bs8 (same run) would be 0 flips (no data, reported)

Usage:
  python paperF_bs_ladder_analysis.py [--no_enrich] [--output_dir <dir>]

Writes:
  status/PAPERF_BS_LADDER_VERDICT.md
"""

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from itertools import permutations

# ── Constants ────────────────────────────────────────────────────────────────
CORE6 = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]
EXPECTED_N = {
    "hellaswag": 10042, "arc_challenge": 1172, "arc_easy": 2376,
    "piqa": 1838, "openbookqa": 500, "winogrande": 1267,
}

PROJECT_ROOT = "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
RESULTS_DIR = os.path.join(PROJECT_ROOT, "olmo2_downstream_results")

# The 6 rungs: (label, core6_score, bs8_dir, bs16_dir)
RUNGS = [
    ("base_full32",  0.70365, "7B_base_full_bs8",           "7B_base_full_bs16"),
    ("ShortGPT-16", 0.62247, "7B_shortgpt16_step200000_bs8", "7B_shortgpt16_step200000_bs16"),
    ("keep14",      0.59532, "7B_keep14_step200000_v2",      "7B_keep14_step200000_bs16"),
    ("keep12",      0.56888, "7B_keep12_step124000_v2",      "7B_keep12_step124000_bs16"),
    ("keep10",      0.52999, "7B_keep10_step83500_v2",       "7B_keep10_step83500_bs16"),
    ("keep8",       0.52328, "7B_keep8_step121000_v2",       "7B_keep8_step121000_bs16"),
]


# ── Inlined enrichment logic (same as enrich_per_example_normscores.py) ───────
import re

def _hs_preprocess(text: str) -> str:
    text = text.strip()
    text = text.replace(" [title]", ". ")
    text = re.sub(r"\[.*?\]", "", text)
    text = text.replace("  ", " ")
    return text


def _load_task_examples(task: str):
    from datasets import load_dataset
    if task == "hellaswag":
        d = load_dataset("Rowan/hellaswag", split="validation")
        out = []
        for ex in d:
            ctx = ex["ctx_a"] + " " + ex["ctx_b"].capitalize()
            query = _hs_preprocess(ex["activity_label"] + ": " + ctx)
            choices = [_hs_preprocess(e) for e in ex["endings"]]
            out.append({"gold": int(ex["label"]),
                        "cands": [(query, " " + c, len(c)) for c in choices]})
        return out
    if task in ("arc_challenge", "arc_easy"):
        cfg = "ARC-Challenge" if task == "arc_challenge" else "ARC-Easy"
        d = load_dataset("allenai/ai2_arc", cfg, split="test")
        out = []
        for ex in d:
            q = "Question: " + ex["question"] + "\nAnswer:"
            texts = ex["choices"]["text"]; labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels: continue
            out.append({"gold": labels.index(ans),
                        "cands": [(q, " " + t, len(t)) for t in texts]})
        return out
    if task == "openbookqa":
        d = load_dataset("allenai/openbookqa", "main", split="test")
        out = []
        for ex in d:
            q = ex["question_stem"]; texts = ex["choices"]["text"]; labels = ex["choices"]["label"]
            ans = ex["answerKey"]
            if ans not in labels: continue
            out.append({"gold": labels.index(ans),
                        "cands": [(q, " " + t, len(t)) for t in texts]})
        return out
    if task == "piqa":
        d = load_dataset("ybisk/piqa", revision="refs/convert/parquet", split="validation")
        out = []
        for ex in d:
            q = "Question: " + ex["goal"] + "\nAnswer:"
            sols = [ex["sol1"], ex["sol2"]]
            out.append({"gold": int(ex["label"]),
                        "cands": [(q, " " + s, len(s)) for s in sols]})
        return out
    if task == "winogrande":
        d = load_dataset("allenai/winogrande", "winogrande_xl", split="validation")
        out = []
        for ex in d:
            s = ex["sentence"]; idx = s.index("_"); target = s[idx + 1:].strip()
            prefix = s[:idx]; opts = [ex["option1"], ex["option2"]]
            out.append({"gold": {"1": 0, "2": 1}[ex["answer"]],
                        "cands": [(prefix + o, " " + target, len(target)) for o in opts]})
        return out
    raise ValueError(f"unknown task: {task}")


def _build_norm_lens_lookup(task: str):
    examples = _load_task_examples(task)
    lookup = {}
    for shard_index in range(8):
        shard = examples[shard_index::8]
        for ei, ex in enumerate(shard):
            item_id = shard_index + ei * 8
            lookup[item_id] = [c[2] for c in ex["cands"]]
    return lookup


def enrich_dir_if_needed(results_dir: str) -> bool:
    """Add norm_scores to per_example files that lack them. Returns True if any enrichment done."""
    any_done = False
    for task in CORE6:
        fpath = os.path.join(results_dir, f"per_example_{task}.jsonl")
        if not os.path.exists(fpath):
            print(f"  [ENRICH] {task}: MISSING per_example file -- skipping")
            continue
        with open(fpath) as f:
            rows = [json.loads(l) for l in f]
        if not rows or "norm_lens" in rows[0]:
            print(f"  [ENRICH] {task}: already has norm_lens -- skipping")
            continue
        print(f"  [ENRICH] {task}: loading norm_lens lookup...", end=" ", flush=True)
        nl_lookup = _build_norm_lens_lookup(task)
        print(f"OK ({len(nl_lookup)} items). Enriching...", end=" ", flush=True)
        n_ok = n_nan = n_mismatch = 0
        enriched = []
        for row in rows:
            iid = row["item_id"]
            nl_list = nl_lookup.get(iid, [1] * len(row["option_scores"]))
            letters = sorted(row["option_scores"].keys())
            nl_dict = {letters[k]: nl_list[k] for k in range(min(len(letters), len(nl_list)))}
            if row.get("nan"):
                ns_dict = {l: None for l in letters}
                n_nan += 1
            else:
                ns_dict = {}
                for k, l in enumerate(letters):
                    raw = row["option_scores"].get(l)
                    nlen = nl_list[k] if k < len(nl_list) else 1
                    ns_dict[l] = round(raw / max(nlen, 1), 6) if raw is not None else None
                # verify
                best_ns = max(ns_dict.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1e9))
                pred_norm_gold = (best_ns[0] == row["gold_letter"])
                stored = row.get("acc_norm_score", 0.0)
                if abs((1.0 if pred_norm_gold else 0.0) - stored) > 0.01:
                    n_mismatch += 1
                n_ok += 1
            new_row = dict(row); new_row["norm_lens"] = nl_dict; new_row["norm_scores"] = ns_dict
            enriched.append(new_row)
        tmp = fpath + ".tmp_enrich"
        with open(tmp, "w") as f:
            for r in enriched:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, fpath)
        mw = f"  !! {n_mismatch} MISMATCHES" if n_mismatch else ""
        print(f"done: {len(enriched)} rows, {n_nan} nan{mw}")
        any_done = True
    return any_done


# ── Data loading ─────────────────────────────────────────────────────────────

def load_per_example(results_dir: str, task: str) -> list:
    """Return list of dicts from per_example_{task}.jsonl"""
    fpath = os.path.join(results_dir, f"per_example_{task}.jsonl")
    if not os.path.exists(fpath):
        return None
    rows = []
    with open(fpath) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def compute_acc_norm_pred(row: dict) -> bool:
    """Return predicted correct under acc_norm (norm_scores argmax == gold_letter)."""
    ns = row.get("norm_scores", {})
    if not ns or row.get("nan"):
        return None  # skip NaN items
    best = max(ns.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1e9))
    return best[0] == row["gold_letter"]


def compute_acc_norm_margin(row: dict) -> float:
    """Return margin = (best norm_score - 2nd best norm_score), or None if NaN."""
    ns = row.get("norm_scores", {})
    if not ns or row.get("nan"):
        return None
    vals = sorted([v for v in ns.values() if v is not None], reverse=True)
    if len(vals) < 2:
        return None
    return vals[0] - vals[1]


def load_rung_data(bs8_dir: str, bs16_dir: str, task: str):
    """
    Load paired per-example data for a rung's bs8 and bs16 runs.
    Returns list of (item_id, margin_bs8, pred_correct_bs8, pred_correct_bs16)
    Only items present in BOTH files, keyed by item_id.
    """
    r8 = load_per_example(bs8_dir, task)
    r16 = load_per_example(bs16_dir, task)
    if r8 is None or r16 is None:
        return None, f"missing file ({'bs8' if r8 is None else 'bs16'})"

    map8 = {row["item_id"]: row for row in r8}
    map16 = {row["item_id"]: row for row in r16}

    paired = []
    for iid in sorted(map8.keys()):
        if iid not in map16:
            continue
        row8 = map8[iid]
        row16 = map16[iid]
        if row8.get("nan") or row16.get("nan"):
            continue
        p8 = compute_acc_norm_pred(row8)
        p16 = compute_acc_norm_pred(row16)
        margin = compute_acc_norm_margin(row8)
        if p8 is None or p16 is None or margin is None:
            continue
        paired.append((iid, margin, p8, p16))

    return paired, None


# ── Statistics ────────────────────────────────────────────────────────────────

def spearman_rho(x, y):
    """Spearman correlation for small n (exact ranks)."""
    n = len(x)
    if n < 2:
        return float('nan')
    rx = _ranks(x)
    ry = _ranks(y)
    d2 = sum((a - b) ** 2 for a, b in zip(rx, ry))
    return 1.0 - 6.0 * d2 / (n * (n ** 2 - 1))


def _ranks(x):
    """Convert list to rank list (1-indexed, average ties)."""
    sorted_x = sorted(enumerate(x), key=lambda kv: kv[1])
    ranks = [0] * len(x)
    i = 0
    while i < len(sorted_x):
        j = i
        while j < len(sorted_x) - 1 and sorted_x[j + 1][1] == sorted_x[j][1]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[sorted_x[k][0]] = avg_rank
        i = j + 1
    return ranks


def exact_permutation_p_twosided(x, y, observed_rho):
    """
    Exact two-sided p for Spearman correlation under H0: y is a random permutation of y.
    Enumerate all n! permutations of y (feasible for n=6: 720).
    """
    from itertools import permutations as _perms
    n = len(y)
    count = 0
    total = 0
    for perm in _perms(y):
        rho = spearman_rho(x, list(perm))
        total += 1
        if abs(rho) >= abs(observed_rho) - 1e-9:
            count += 1
    return count / total


def bootstrap_ci_mean(values, n_boot=10000, ci=0.95):
    """Bootstrap CI for the mean of values."""
    import random
    n = len(values)
    boot_means = []
    for _ in range(n_boot):
        sample = [random.choice(values) for _ in range(n)]
        boot_means.append(sum(sample) / n)
    boot_means.sort()
    lo = boot_means[int((1 - ci) / 2 * n_boot)]
    hi = boot_means[int((1 + ci) / 2 * n_boot)]
    return lo, hi


def margin_bucket_flip_curve(paired_all: list, n_buckets: int = 10):
    """
    Pooled across all rungs. Bucket items by margin (from bs8 norm_scores).
    For each bucket: P(flip) = fraction of items where bs8 pred != bs16 pred.
    Returns list of (bucket_midpoint, n_items, n_flips, p_flip).
    """
    # Find margin range
    margins = [m for (_, m, _, _) in paired_all]
    lo, hi = min(margins), max(margins)
    bucket_width = (hi - lo) / n_buckets

    buckets = [[] for _ in range(n_buckets)]
    for (iid, margin, p8, p16) in paired_all:
        bi = min(int((margin - lo) / bucket_width), n_buckets - 1)
        buckets[bi].append(int(p8 != p16))

    result = []
    for i, items in enumerate(buckets):
        if not items:
            continue
        mid = lo + (i + 0.5) * bucket_width
        n = len(items)
        nf = sum(items)
        result.append((round(mid, 4), n, nf, round(nf / n, 4) if n > 0 else 0.0))
    return result


def loo_mediation_check(rung_data: list, margin_curves_per_rung: list):
    """
    Leave-one-out mediation check.
    For each left-out rung i:
      - Fit a logistic-like curve from the other 5 rungs' pooled (margin, flip) data.
      - Predict the flip_count for rung i using that curve + rung i's margin distribution.
      - Report predicted vs observed.
    Returns list of (rung_label, observed_flips, predicted_flips, abs_error).

    Also computes constant-rate null: use mean flip rate across other 5 rungs * n_items_i.
    """
    results = []
    for i, (label, core6, flips_obs, total_n, margins_i) in enumerate(rung_data):
        # Build training pool from other 5 rungs
        other_items = []
        for j, (_, _, _, _, margins_j) in enumerate(rung_data):
            if j == i:
                continue
            for (margin, is_flip) in margins_j:
                other_items.append((margin, is_flip))

        # Constant-rate null: mean flip rate from other rungs
        if other_items:
            const_rate = sum(x[1] for x in other_items) / len(other_items)
        else:
            const_rate = 0.0
        pred_const = const_rate * total_n

        # Margin-based prediction: for each item in rung i, predict P(flip|margin)
        # using empirical P(flip|margin bucket) from the other 5 rungs.
        # Bucket boundaries based on global margin range (from other 5 rungs).
        if other_items:
            other_margins = [x[0] for x in other_items]
            g_lo = min(other_margins)
            g_hi = max(other_margins)
            n_bk = 8
            bw = (g_hi - g_lo) / n_bk if g_hi > g_lo else 1.0

            bucket_flips = [0] * n_bk
            bucket_total = [0] * n_bk
            for (margin, is_flip) in other_items:
                bi = min(int((margin - g_lo) / bw), n_bk - 1)
                bucket_flips[bi] += is_flip
                bucket_total[bi] += 1

            bucket_rate = [bucket_flips[k] / max(bucket_total[k], 1) for k in range(n_bk)]

            pred_margin = 0.0
            for (margin, _) in margins_i:
                if margin < g_lo:
                    bi = 0
                elif margin >= g_hi:
                    bi = n_bk - 1
                else:
                    bi = min(int((margin - g_lo) / bw), n_bk - 1)
                pred_margin += bucket_rate[bi]
        else:
            pred_margin = 0.0

        results.append({
            "label": label,
            "core6": core6,
            "n_items": total_n,
            "flips_obs": flips_obs,
            "pred_const": round(pred_const, 1),
            "pred_margin": round(pred_margin, 1),
            "err_const": round(abs(pred_const - flips_obs), 1),
            "err_margin": round(abs(pred_margin - flips_obs), 1),
        })
    return results


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze():
    import random
    random.seed(42)

    results = {}
    missing_rungs = []

    print("\n" + "=" * 70)
    print("PART 1: Enriching ShortGPT-16 bs8 with norm_scores")
    print("=" * 70)

    sg16_bs8_dir = os.path.join(RESULTS_DIR, "7B_shortgpt16_step200000_bs8")
    if os.path.exists(sg16_bs8_dir):
        # Check if already enriched
        test_file = os.path.join(sg16_bs8_dir, "per_example_hellaswag.jsonl")
        if os.path.exists(test_file):
            with open(test_file) as f:
                first_row = json.loads(f.readline())
            if "norm_lens" not in first_row:
                print("  bs8 dir lacks norm_scores -- enriching now...")
                enrich_dir_if_needed(sg16_bs8_dir)
            else:
                print("  bs8 dir already has norm_scores -- skipping enrichment")
    else:
        print(f"  WARNING: {sg16_bs8_dir} does not exist")

    print("\n" + "=" * 70)
    print("PART 2: Loading per-rung bs8 vs bs16 paired data")
    print("=" * 70)

    # Per-rung: (label, core6, flip_count, n_valid, list_of_(margin, is_flip))
    rung_data_for_loo = []
    all_paired = []  # for pooled margin curve
    rung_flip_data = []  # for Spearman

    for (label, core6_score, bs8_name, bs16_name) in RUNGS:
        bs8_dir = os.path.join(RESULTS_DIR, bs8_name)
        bs16_dir = os.path.join(RESULTS_DIR, bs16_name)

        print(f"\n--- {label} (core6={core6_score:.5f}) ---")
        print(f"  bs8: {bs8_name}")
        print(f"  bs16: {bs16_name}")

        if not os.path.exists(bs8_dir):
            print(f"  MISSING bs8 dir -- skipping rung")
            missing_rungs.append(label)
            continue
        if not os.path.exists(bs16_dir):
            print(f"  MISSING bs16 dir -- skipping rung")
            missing_rungs.append(label)
            continue

        total_flips = 0
        total_valid = 0
        rung_margins_flips = []

        for task in CORE6:
            paired, err = load_rung_data(bs8_dir, bs16_dir, task)
            if paired is None:
                print(f"  {task}: ERROR: {err}")
                continue
            task_flips = sum(1 for (_, _, p8, p16) in paired if p8 != p16)
            task_valid = len(paired)
            total_flips += task_flips
            total_valid += task_valid
            for (iid, margin, p8, p16) in paired:
                rung_margins_flips.append((margin, int(p8 != p16)))
            all_paired.extend(paired)
            # Verify n_scored
            exp_n = EXPECTED_N.get(task, -1)
            if task_valid != exp_n:
                print(f"  {task}: WARNING n_valid={task_valid} != expected {exp_n} (task might have NaN items filtered)")
            print(f"  {task}: n={task_valid}, flips={task_flips} ({100*task_flips/max(task_valid,1):.2f}%)")

        print(f"  TOTAL: n_valid={total_valid}, total_flips={total_flips} ({100*total_flips/max(total_valid,1):.2f}%)")
        # Only include rung in Spearman/LOO if we have substantial paired data (at least 1000 valid pairs)
        if total_valid >= 1000:
            rung_flip_data.append((label, core6_score, total_flips))
            rung_data_for_loo.append((label, core6_score, total_flips, total_valid, rung_margins_flips))
        else:
            print(f"  --> SKIPPING rung {label} from Spearman/LOO (only {total_valid} valid pairs, need ≥1000)")

    print("\n" + "=" * 70)
    print("PART 3a: Spearman(core6, flip_count)")
    print("=" * 70)

    spearman_section = ""
    if len(rung_flip_data) < 2:
        spearman_section = f"INSUFFICIENT DATA: only {len(rung_flip_data)} rungs with complete bs8+bs16 data (need ≥2 for correlation, ≥3 for permutation p)"
        print("  " + spearman_section)
    elif len(rung_flip_data) == 2:
        # n=2: rho is trivially ±1, permutation p = 1.0 (both permutations give |rho|=1)
        x_core6 = [r[1] for r in rung_flip_data]
        y_flips = [r[2] for r in rung_flip_data]
        rho = spearman_rho(x_core6, y_flips)
        print(f"  n={len(rung_flip_data)} rungs (NOTE: n=2, rho=±1 trivially, permutation p is uninformative)")
        print(f"  Spearman rho = {rho:.4f}")
        for (label, c6, fl) in rung_flip_data:
            print(f"  {label}: core6={c6:.5f}, flips={fl}")
        spearman_section = f"n={len(rung_flip_data)} (ONLY 2 rungs complete — Spearman rho={rho:.4f} trivially ±1 for n=2; exact p uninformative; MAIN TEST NOT YET POSSIBLE)"
        results["spearman"] = {
            "n": len(rung_flip_data), "rho": rho, "p": float('nan'), "note": "n=2: trivially ±1",
            "rungs": [{"label": r[0], "core6": r[1], "flips": r[2]} for r in rung_flip_data]
        }
        print(f"  {spearman_section}")
    else:
        x_core6 = [r[1] for r in rung_flip_data]
        y_flips = [r[2] for r in rung_flip_data]
        rho = spearman_rho(x_core6, y_flips)
        p_val = exact_permutation_p_twosided(x_core6, y_flips, rho)
        print(f"  n={len(rung_flip_data)} rungs")
        print(f"  Spearman rho = {rho:.4f}")
        print(f"  exact two-sided p = {p_val:.4f}")
        for (label, c6, fl) in rung_flip_data:
            print(f"  {label}: core6={c6:.5f}, flips={fl}")
        spearman_section = f"n={len(rung_flip_data)}, rho={rho:.4f}, exact two-sided p={p_val:.4f}"

        results["spearman"] = {
            "n": len(rung_flip_data), "rho": rho, "p": p_val,
            "rungs": [{"label": r[0], "core6": r[1], "flips": r[2]} for r in rung_flip_data]
        }

    print("\n" + "=" * 70)
    print("PART 3b: Pooled margin → P(flip) bucketed curve")
    print("=" * 70)

    margin_curve = []
    if all_paired:
        margin_curve = margin_bucket_flip_curve(all_paired, n_buckets=10)
        print(f"  Total paired items: {len(all_paired)}")
        print(f"  Bucket width: {(max(m for _,m,_,_ in all_paired) - min(m for _,m,_,_ in all_paired))/10:.4f}")
        print(f"  {'Midpoint':>10}  {'N':>6}  {'Flips':>6}  {'P(flip)':>8}")
        for (mid, n, nf, pf) in margin_curve:
            print(f"  {mid:>10.4f}  {n:>6}  {nf:>6}  {pf:>8.4f}")
    else:
        print("  No paired data available")

    print("\n" + "=" * 70)
    print("PART 3c: LOO mediation check")
    print("=" * 70)

    loo_section = ""
    loo_results = []
    if len(rung_data_for_loo) >= 3:
        loo_results = loo_mediation_check(rung_data_for_loo, margin_curve)
        print(f"  {'Rung':<15}  {'Obs':>6}  {'Pred(const)':>12}  {'Pred(margin)':>13}  {'Err(const)':>11}  {'Err(margin)':>12}")
        for r in loo_results:
            print(f"  {r['label']:<15}  {r['flips_obs']:>6}  {r['pred_const']:>12.1f}  {r['pred_margin']:>13.1f}  {r['err_const']:>11.1f}  {r['err_margin']:>12.1f}")
        total_err_const = sum(r["err_const"] for r in loo_results)
        total_err_margin = sum(r["err_margin"] for r in loo_results)
        print(f"  {'TOTAL':.<15}  {'':>6}  {'':>12}  {'':>13}  {total_err_const:>11.1f}  {total_err_margin:>12.1f}")
        improvement = total_err_const - total_err_margin
        print(f"  Margin beats constant-rate by: {improvement:.1f} total abs error")
        if improvement > 0:
            loo_section = f"Margin model LOO total |error|={total_err_margin:.1f} vs constant-rate={total_err_const:.1f}; ΔMAE={improvement:.1f} in favor of margin"
        else:
            loo_section = f"Margin model LOO total |error|={total_err_margin:.1f} vs constant-rate={total_err_const:.1f}; ΔMAE={improvement:.1f} -- margin model does NOT beat constant-rate null"
        results["loo"] = {"loo_results": loo_results, "total_err_const": total_err_const, "total_err_margin": total_err_margin}
    else:
        loo_section = f"INSUFFICIENT DATA: only {len(rung_data_for_loo)} complete rungs (need ≥3)"
        print("  " + loo_section)

    print("\n" + "=" * 70)
    print("PART 3d: Zero-effect sanity check")
    print("=" * 70)
    print("  No bs8-vs-bs8 re-run data available. Expected: 0 flips.")
    print("  (This is per the same-harness memory: same code = bit-identical.)")

    # ── Write verdict ──────────────────────────────────────────────────────────
    write_verdict(rung_flip_data, rung_data_for_loo, margin_curve, loo_results,
                  results, missing_rungs, spearman_section, loo_section, all_paired)
    print("\nDone. Verdict written to status/PAPERF_BS_LADDER_VERDICT.md")


def write_verdict(rung_flip_data, rung_data_for_loo, margin_curve, loo_results,
                  results, missing_rungs, spearman_section, loo_section, all_paired):
    import random
    random.seed(42)

    # Build summary tables
    n_rungs_complete = len(rung_flip_data)
    n_rungs_total = len(RUNGS)

    def verdict_call(results, rung_flip_data, loo_results):
        """Return verdict strings."""
        lines = []
        if "spearman" not in results:
            lines.append("FLIP_RATE_MONOTONE: **UNTESTED** (insufficient data)")
        else:
            rho = results["spearman"]["rho"]
            p = results["spearman"]["p"]
            sig = p < 0.05
            direction = rho < 0  # core6 down = damage up = flips up: expect rho < 0
            if sig and direction:
                lines.append(f"FLIP_RATE_MONOTONE: **SUPPORTED** (Spearman rho={rho:.4f} with core6, exact two-sided p={p:.4f}, n={results['spearman']['n']})")
            elif sig and not direction:
                lines.append(f"FLIP_RATE_MONOTONE: **REVERSED** (Spearman rho={rho:.4f}, exact two-sided p={p:.4f}) -- flip rate is LOWER for more damaged models")
            else:
                lines.append(f"FLIP_RATE_MONOTONE: **NOT SIGNIFICANT** (Spearman rho={rho:.4f}, exact two-sided p={p:.4f})")

        if "loo" not in results:
            lines.append("MARGIN_MEDIATES: **UNTESTED** (insufficient data)")
        else:
            imp = results["loo"]["total_err_const"] - results["loo"]["total_err_margin"]
            if imp > 0:
                lines.append(f"MARGIN_MEDIATES: **MARGIN BEATS CONSTANT NULL** (LOO ΔMAE={imp:.1f})")
            else:
                lines.append(f"MARGIN_MEDIATES: **CONSTANT NULL NOT BEATEN** (LOO ΔMAE={imp:.1f} favors constant rate)")
        return lines

    verdict_lines = verdict_call(results, rung_flip_data, loo_results)

    # Build the markdown
    lines = []
    lines.append("# PAPERF_BS_LADDER_VERDICT.md")
    lines.append("")
    lines.append("**Direction A judgment experiment — bs8 vs bs16 flip rate ladder**")
    lines.append(f"Generated: 2026-08-08")
    lines.append("")
    lines.append("## Experiment Summary")
    lines.append("")
    lines.append("Question: Does flip rate (acc_norm decision reversal between bs=8 and bs=16)")
    lines.append("increase monotonically with model damage (core6 acc decline)?")
    lines.append("Can the margin distribution (near-tie density) explain this variation")
    lines.append("in a LOO mediation check?")
    lines.append("")
    lines.append(f"- Rungs complete (have both bs8 + bs16): **{n_rungs_complete}/{n_rungs_total}**")
    if missing_rungs:
        lines.append(f"- Missing rungs: {', '.join(missing_rungs)}")
    lines.append("")

    lines.append("## VERDICTS")
    lines.append("")
    for v in verdict_lines:
        lines.append(f"- {v}")
    lines.append("")

    lines.append("## Rung-level acc_norm scores and flip counts")
    lines.append("")
    lines.append("| rung | core6 | bs8_dir | bs16_dir | total_flips | flip_rate% |")
    lines.append("|------|------:|---------|----------|------------:|-----------:|")
    rung_dict = {r[0]: r for r in rung_flip_data}
    for (label, core6_score, bs8_name, bs16_name) in RUNGS:
        if label in rung_dict:
            _, _, fl = rung_dict[label]
            loo_entry = next((r for r in rung_data_for_loo if r[0] == label), None)
            total_n = loo_entry[3] if loo_entry else "?"
            rate = f"{100*fl/total_n:.2f}" if isinstance(total_n, int) and total_n > 0 else "?"
            lines.append(f"| {label} | {core6_score:.5f} | `{bs8_name}` | `{bs16_name}` | {fl} | {rate} |")
        else:
            lines.append(f"| {label} | {core6_score:.5f} | `{bs8_name}` | `{bs16_name}` | MISSING | — |")
    lines.append("")

    lines.append("## Spearman test: core6 vs flip_count")
    lines.append("")
    lines.append(f"**{spearman_section}**")
    lines.append("")
    if "spearman" in results:
        lines.append("| rung | core6 | flip_count |")
        lines.append("|------|------:|-----------:|")
        for r in results["spearman"]["rungs"]:
            lines.append(f"| {r['label']} | {r['core6']:.5f} | {r['flips']} |")
        lines.append("")
        lines.append(f"Note: hypothesis is rho < 0 (higher core6 = less damage = fewer flips).")
        rho = results["spearman"]["rho"]
        p = results["spearman"]["p"]
        direction_note = "rho < 0 (expected direction)" if rho < 0 else "rho > 0 (unexpected direction)"
        lines.append(f"rho = {rho:.4f} ({direction_note}), exact two-sided p = {p:.4f}")
    lines.append("")

    lines.append("## Pooled margin → P(flip) bucketed curve")
    lines.append("")
    if margin_curve:
        lines.append(f"Total paired items pooled: {len(all_paired)}")
        lines.append("Bucket width: acc_norm score margin scale, 10 equal-width buckets over the observed range.")
        lines.append("")
        lines.append("| margin_bucket_mid | n | flips | P(flip) |")
        lines.append("|------------------:|--:|------:|--------:|")
        for (mid, n, nf, pf) in margin_curve:
            lines.append(f"| {mid:.4f} | {n} | {nf} | {pf:.4f} |")
    else:
        lines.append("No pooled data available.")
    lines.append("")

    lines.append("## LOO mediation check")
    lines.append("")
    lines.append("**Protocol**: For each left-out rung i, fit empirical P(flip|margin_bucket)")
    lines.append("from the other 5 rungs' pooled data. Predict rung i's flip count.")
    lines.append("Compare against constant-rate null (mean flip rate of other 5 rungs × n_items_i).")
    lines.append("")
    lines.append("**This is NOT in-sample** — each rung's prediction uses only held-out training data.")
    lines.append("")
    if loo_results:
        lines.append("| rung | n_items | obs_flips | pred_const | pred_margin | err_const | err_margin |")
        lines.append("|------|--------:|----------:|-----------:|------------:|----------:|-----------:|")
        for r in loo_results:
            lines.append(f"| {r['label']} | {r['n_items']} | {r['flips_obs']} | {r['pred_const']:.1f} | {r['pred_margin']:.1f} | {r['err_const']:.1f} | {r['err_margin']:.1f} |")
        total_ec = sum(r["err_const"] for r in loo_results)
        total_em = sum(r["err_margin"] for r in loo_results)
        lines.append(f"| **TOTAL** | | | | | **{total_ec:.1f}** | **{total_em:.1f}** |")
        lines.append("")
        lines.append(f"**{loo_section}**")
    else:
        lines.append(f"**{loo_section}**")
    lines.append("")

    lines.append("## What was NOT done")
    lines.append("")
    lines.append("- Zero-effect sanity: no bs8-vs-bs8 re-run data. Per memory `same-harness-runs-bit-identical`:")
    lines.append("  same code + same data + same harness = byte-identical results = 0 flips expected.")
    lines.append("  This is not independently measured here; it is asserted from prior finding.")
    lines.append("- bs16-vs-bs16 re-run for internal consistency check (not done).")
    lines.append("- Logistic regression fit (replaced by empirical bucket-rate LOO).")
    lines.append("- Bootstrap CI on flip rates per rung (n=1 per cell, CI not meaningful).")
    lines.append("")

    lines.append("## Relationship to the existing near-tie density finding")
    lines.append("")
    lines.append("The already-established result is:")
    lines.append("- `Spearman(core6, median_margin) = +1.0, p=0.0028` (6 rungs, exact)")
    lines.append("- `Spearman(core6, frac<0.005) = -0.9429, p=0.0167`")
    lines.append("- More damaged models have denser near-tie distributions (fewer wide margins).")
    lines.append("")
    lines.append("The current experiment asks: does that translate into more flips?")
    lines.append("The two findings are:")
    lines.append("- **Complementary** if flip_rate is also monotone with damage AND margin distribution")
    lines.append("  explains inter-rung variation in LOO — that would complete the causal chain.")
    lines.append("- **Disconnected** if flip_rate is not monotone or margin doesn't explain")
    lines.append("  inter-rung variation — in which case near-tie density is a model property,")
    lines.append("  not a predictor of implementation-sensitivity.")
    lines.append("")
    lines.append("See verdict lines at top for actual finding.")
    lines.append("")

    lines.append("---")
    lines.append("*All numbers are empirical, computed from per_example_*.jsonl files.")
    lines.append("All statistical tests are exact (n=6 permutation for Spearman).")
    lines.append("LOO is strictly out-of-sample — no in-sample test was run.*")

    output_path = "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/status/PAPERF_BS_LADDER_VERDICT.md"
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nVerdict written to: {output_path}")


if __name__ == "__main__":
    import os
    os.environ.setdefault("http_proxy", "http://hy-proxy.woa.com:3128")
    os.environ.setdefault("https_proxy", "http://hy-proxy.woa.com:3128")
    os.environ.setdefault("all_proxy", "http://hy-proxy.woa.com:3128")
    os.environ.setdefault("HF_DATASETS_CACHE",
        "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache")
    analyze()
