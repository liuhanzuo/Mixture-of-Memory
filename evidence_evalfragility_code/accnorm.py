#!/usr/bin/env python3
"""
evidence_evalfragility_code/accnorm.py  (2026-08-08)

Recompute the "eval fragility vs damage" analysis on the CORRECT metric: acc_norm
(argmax over norm_scores = option_scores[k] / norm_len[k]).

This is the metric that produces Paper B's headline core6 numbers:
  base=.70365, ShortGPT16=.62247, keep14=.59532, keep12=.56888, keep10=.52999, keep8=.52328

The old xarch.py / pert.py / margin_flip.py used "option_scores" (raw sum-logprob),
which is the ACC metric, NOT the acc_norm metric. The two produce different argmax
predictions on ~60-80% of items (Jaccard ~0.08-0.27 on the flip sets).

This script:
  1. Loads per_example_*.jsonl for all 6 rungs (single-disk H20 zwfy6, _nl enriched)
  2. Computes acc_norm margin = top1(norm_scores) - top2(norm_scores) per item
  3. Reports:
       (a) median margin and frac(margin < threshold) per rung
       (b) Spearman(core6, median_margin) with exact permutation p (n=6)
       (c) Spearman(core6, frac<threshold) with exact permutation p (n=6)
       (d) batch-size perturbation: bs8 vs bs16 per-item acc_norm flips
           with margin->P(flip) curve (bs8 data = _v2, bs16 = separate dirs)
       (e) Mediation: leave-one-out cross-validation for margin->flip-rate model
           vs null models (a) constant flip rate (b) logit-scale-only null
           Proper LOO avoids the in-sample algebraic identity Sigma_obs = Sigma_pred

Output: printed to stdout, also written to evidence_evalfragility_code/accnorm_results.txt

IMPORTANT: does NOT touch xarch.py, pert.py, or margin_flip.py.

Usage (on zwfy6):
    python accnorm.py [--base_dir /path/to/olmo2_downstream_results]
"""

import json
import math
import os
import sys
import itertools
from pathlib import Path
from collections import defaultdict

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
BASE_DIR = os.environ.get(
    "RUNDIR",
    "/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results"
)
if len(sys.argv) > 2 and sys.argv[1] == "--base_dir":
    BASE_DIR = sys.argv[2]

CORE6 = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "winogrande", "openbookqa"]

# The 6 rungs (label, dir_suffix, clean core6 value from Table 4)
RUNGS = [
    ("base_full32",   "7B_base_full_bs8",            0.70365),
    ("ShortGPT-16",   "7B_shortgpt16_step200000_v2",  0.62247),
    ("keep14@200k",   "7B_keep14_step200000_v2",       0.59532),
    ("keep12@124k",   "7B_keep12_step124000_v2",       0.56888),
    ("keep10@83.5k",  "7B_keep10_step83500_v2",        0.52999),
    ("keep8@121k",    "7B_keep8_step121000_v2",        0.52328),
]

# bs16 directories (same rung but batch_size=16).
# These exist on zwfy6 if the sweep was run.
BS16_RUNGS = [
    ("base_full32",   "7B_base_full_bs16",             0.70365),
    ("ShortGPT-16",   "7B_shortgpt16_step200000_bs16",  0.62247),
    ("keep14@200k",   "7B_keep14_step200000_bs16",       0.59532),
    ("keep12@124k",   "7B_keep12_step124000_bs16",       0.56888),
    ("keep10@83.5k",  "7B_keep10_step83500_bs16",        0.52999),
    ("keep8@121k",    "7B_keep8_step121000_bs16",        0.52328),
]

# Margin threshold for "near-tie" in acc_norm space.
# In raw-logprob space the old code used 0.1 nats.
# In norm_scores (nats/char) the scale is ~100-1000x smaller.
# For a candidate of length ~20 chars, 0.1 nat / 20 chars = 0.005 nats/char.
# For a candidate of length ~5 chars (e.g. short ARC answer), 0.1/5 = 0.02 nats/char.
# We choose 0.005 as the primary threshold (approx. "0.1 raw nats on a 20-char cand")
# and report 0.001 and 0.01 as secondary to show sensitivity.
THRESHOLDS = [0.001, 0.005, 0.01]
PRIMARY_THRESHOLD = 0.005


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------
def load_per_example(dir_name: str, task: str) -> dict:
    """Returns {item_id: row_dict} or None if file missing."""
    p = os.path.join(BASE_DIR, dir_name, f"per_example_{task}.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[r["item_id"]] = r
    return out


def accnorm_margin(row: dict) -> float:
    """top1(norm_scores) - top2(norm_scores).  Returns 1e9 if only 1 option."""
    ns = row.get("norm_scores", {})
    vals = [v for v in ns.values() if v is not None]
    if len(vals) < 2:
        return 1e9
    vals.sort(reverse=True)
    return vals[0] - vals[1]


def accnorm_pred(row: dict) -> str:
    """Predicted letter under acc_norm (argmax of norm_scores)."""
    ns = row.get("norm_scores", {})
    if not ns:
        return row.get("pred_letter", "")
    best = max(ns.items(), key=lambda kv: (kv[1] if kv[1] is not None else -1e9))
    return best[0]


def spearman(x, y):
    def ranks(v):
        s = sorted(range(len(v)), key=lambda i: v[i])
        r = [0] * len(v)
        for j, i in enumerate(s):
            r[i] = j + 1
        return r
    rx, ry = ranks(x), ranks(y)
    n = len(x)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(
        sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)
    )
    return num / den if den else 0.0


def exact_perm_two_sided(x, y):
    """
    Exact permutation two-sided p for Spearman rho (n up to 8).
    Permutes y, counts how often |rho(x, perm)| >= |rho(x, y)|.
    """
    n = len(x)
    obs = abs(spearman(x, y))
    count = 0
    total = 0
    for perm in itertools.permutations(range(n)):
        yp = [y[i] for i in perm]
        total += 1
        if abs(spearman(x, yp)) >= obs - 1e-9:
            count += 1
    return count / total


def quantile(vals, q):
    """p-th quantile of a sorted list. q in [0,1]."""
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, int(q * len(s)))
    return s[i]


# ---------------------------------------------------------------------------
# PART A: Per-rung margin statistics
# ---------------------------------------------------------------------------
def compute_rung_stats(rungs=RUNGS):
    """Returns list of (label, core6, margins_list) for rungs that have data."""
    results = []
    for label, dname, c6 in rungs:
        margins = []
        missing = []
        for task in CORE6:
            data = load_per_example(dname, task)
            if data is None:
                missing.append(task)
                continue
            for row in data.values():
                if row.get("nan"):
                    continue
                m = accnorm_margin(row)
                margins.append(m)
        if missing:
            print(f"  [WARN] {label}: missing tasks {missing}")
        if not margins:
            print(f"  [SKIP] {label}: no data")
            continue
        results.append((label, c6, margins))
    return results


# ---------------------------------------------------------------------------
# PART B: Spearman with exact permutation p
# ---------------------------------------------------------------------------
def spearman_with_exact_p(x, y, label_x="x", label_y="y"):
    rho = spearman(x, y)
    p = exact_perm_two_sided(x, y)
    return rho, p


# ---------------------------------------------------------------------------
# PART C: Batch-size perturbation (bs8 vs bs16)
# ---------------------------------------------------------------------------
BUCK_AN = [(0, 0.001), (0.001, 0.005), (0.005, 0.01), (0.01, 0.05),
           (0.05, 0.2), (0.2, 1.0), (1.0, 1e9)]


def bucket_idx(m):
    for i, (a, b) in enumerate(BUCK_AN):
        if a <= m < b:
            return i
    return len(BUCK_AN) - 1


def compute_bs_perturbation(rungs_bs8=RUNGS, rungs_bs16=BS16_RUNGS):
    """
    Compare bs8 vs bs16 per-item acc_norm predictions.
    For each matched rung, compute per-item flip rate bucketed by acc_norm margin.
    """
    bs8_map = {lab: dn for lab, dn, _ in rungs_bs8}
    bs16_map = {lab: dn for lab, dn, _ in rungs_bs16}
    c6_map = {lab: c6 for lab, _, c6 in rungs_bs8}

    results = {}
    print("\n=== BATCH-SIZE PERTURBATION (bs8 vs bs16), acc_norm margin ===")
    print(f"{'rung':<16}{'n_items':>9}{'n_flips':>9}{'flip%':>8}{'n_neartie':>11}{'neartie%':>10}")
    for label in bs8_map:
        if label not in bs16_map:
            continue
        dn8 = bs8_map[label]
        dn16 = bs16_map[label]
        tot = [0] * len(BUCK_AN)
        fl = [0] * len(BUCK_AN)
        n = 0
        nf = 0
        nt = 0
        ok = True
        for task in CORE6:
            d8 = load_per_example(dn8, task)
            d16 = load_per_example(dn16, task)
            if d8 is None:
                print(f"  [WARN] bs8 missing: {label}/{task}")
                ok = False; break
            if d16 is None:
                print(f"  [INFO] bs16 missing: {label}/{task} -- skipping bs perturbation for this rung")
                ok = False; break
            for k in d8:
                if k not in d16:
                    continue
                r8, r16 = d8[k], d16[k]
                if r8.get("nan") or r16.get("nan"):
                    continue
                m8 = accnorm_margin(r8)
                m16 = accnorm_margin(r16)
                m = min(m8, m16)
                bi = bucket_idx(m)
                tot[bi] += 1
                n += 1
                if m < PRIMARY_THRESHOLD:
                    nt += 1
                p8 = accnorm_pred(r8)
                p16 = accnorm_pred(r16)
                if p8 != p16:
                    fl[bi] += 1
                    nf += 1
        if not ok:
            continue
        results[label] = (c6_map[label], tot, fl, n, nf, nt)
        print(f"{label:<16}{n:>9}{nf:>9}{nf/n*100:>7.3f}%{nt:>11}{nt/n*100:>9.3f}%")

    if not results:
        return results

    print(f"\n  acc_norm margin bucket boundaries: {BUCK_AN}")
    print(f"\n  P(acc_norm flip | margin bucket), bs8 vs bs16:")
    hdr = f"{'rung':<16}" + "".join(f"{('['+str(a)+','+(str(b) if b < 1e9 else 'inf')+')')[:10]:>11}" for a, b in BUCK_AN)
    print(hdr)
    for label, (c6, tot, fl, n, nf, nt) in results.items():
        row = f"{label:<16}" + "".join(f"{(fl[i]/tot[i]*100 if tot[i] else 0):>10.2f}%" for i in range(len(BUCK_AN)))
        print(row)

    # Spearman: core6 vs flip_rate, core6 vs neartie_frac
    labs = list(results.keys())
    c6s = [results[l][0] for l in labs]
    frs = [results[l][4] / results[l][3] for l in labs]
    nts = [results[l][5] / results[l][3] for l in labs]

    if len(labs) >= 4:
        rho_fr, p_fr = spearman_with_exact_p(c6s, frs)
        rho_nt, p_nt = spearman_with_exact_p(c6s, nts)
        print(f"\n  Spearman(core6, flip_rate bs8vbs16)  rho={rho_fr:.4f}  exact_p={p_fr:.4f}  n={len(labs)}")
        print(f"  Spearman(core6, neartie_frac<{PRIMARY_THRESHOLD})  rho={rho_nt:.4f}  exact_p={p_nt:.4f}  n={len(labs)}")

    return results


# ---------------------------------------------------------------------------
# PART D: Mediation via LOO cross-validation
# ---------------------------------------------------------------------------
def loo_mediation(rung_results, bs_results=None):
    """
    Leave-one-out cross-validation for:
       Model 1: margin-based prediction of flip rate.
         For each held-out rung, fit pooled P(flip|bucket) on the remaining rungs,
         predict the held-out rung's total flips, compare to observed.
       Model 0a: constant flip rate (no margin info).
         Mean flip rate across other rungs, applied uniformly.
       Model 0b: logit-scale-only (just predicts mean of others).

    rung_results: from compute_bs_perturbation, dict {label: (c6, tot, fl, n, nf, nt)}
    """
    if len(rung_results) < 3:
        print("\n  [SKIP] LOO mediation requires at least 3 rungs with bs perturbation data")
        return

    labs = list(rung_results.keys())
    print("\n=== LOO MEDIATION: margin-based prediction vs null models ===")
    print("  Held-out rung: observed flips / margin-model prediction / constant-rate prediction")
    print(f"  {'rung':<16}{'observed':>10}{'margin_pred':>13}{'const_pred':>13}{'ratio_margin':>14}{'ratio_const':>13}")

    loo_obs_m = []   # observed when held out
    loo_pred_m = []  # predicted by margin model
    loo_pred_c = []  # predicted by constant-rate model

    for hi, held_lab in enumerate(labs):
        # Train on all but held-out
        train_labs = [l for l in labs if l != held_lab]
        # Pooled P(flip|bucket) on training rungs
        tots_train = [0] * len(BUCK_AN)
        fls_train = [0] * len(BUCK_AN)
        total_flips_train = 0
        total_items_train = 0
        for l in train_labs:
            _, tot, fl, n, nf, _ = rung_results[l]
            for i in range(len(BUCK_AN)):
                tots_train[i] += tot[i]
                fls_train[i] += fl[i]
            total_flips_train += nf
            total_items_train += n
        pooled = [fls_train[i] / tots_train[i] if tots_train[i] else 0
                  for i in range(len(BUCK_AN))]
        const_rate = total_flips_train / total_items_train if total_items_train else 0

        # Predict held-out
        _, tot_h, fl_h, n_h, nf_h, _ = rung_results[held_lab]
        pred_margin = sum(tot_h[i] * pooled[i] for i in range(len(BUCK_AN)))
        pred_const = n_h * const_rate

        loo_obs_m.append(nf_h)
        loo_pred_m.append(pred_margin)
        loo_pred_c.append(pred_const)
        rm = nf_h / pred_margin if pred_margin else float("inf")
        rc = nf_h / pred_const if pred_const else float("inf")
        print(f"  {held_lab:<16}{nf_h:>10}{pred_margin:>13.1f}{pred_const:>13.1f}{rm:>14.3f}{rc:>13.3f}")

    # LOO bias = mean(obs - pred)
    mae_m = sum(abs(o - p) for o, p in zip(loo_obs_m, loo_pred_m)) / len(loo_obs_m)
    mae_c = sum(abs(o - p) for o, p in zip(loo_obs_m, loo_pred_c)) / len(loo_obs_m)
    print(f"\n  LOO MAE: margin_model={mae_m:.2f}  constant_rate={mae_c:.2f}")

    # Pseudo R^2: 1 - SS_pred/SS_null
    ss_m = sum((o - p) ** 2 for o, p in zip(loo_obs_m, loo_pred_m))
    ss_c = sum((o - p) ** 2 for o, p in zip(loo_obs_m, loo_pred_c))
    r2 = 1 - ss_m / ss_c if ss_c else float("nan")
    print(f"  LOO pseudo-R2 (margin vs const null): {r2:.4f}")
    print("  (positive = margin model explains some variance beyond constant rate)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    out_lines = []
    def p(*args, **kwargs):
        s = " ".join(str(a) for a in args)
        print(s, **kwargs)
        out_lines.append(s)

    p("=" * 70)
    p("PART A+B+C: acc_norm margin analysis (single-disk H20, bs8 data)")
    p(f"  Base dir: {BASE_DIR}")
    p("=" * 70)

    # -----------------------------------------------------------------------
    # A: Per-rung margin statistics
    # -----------------------------------------------------------------------
    p("\n--- Per-rung acc_norm margin statistics ---")
    p(f"{'rung':<16}{'core6':>9}{'n':>7}{'med_marg':>11}" +
      "".join(f"{'frac<'+str(t):>11}" for t in THRESHOLDS))

    rung_stats = compute_rung_stats()
    core6_vals = []
    med_margins = []
    frac_vals = {t: [] for t in THRESHOLDS}

    for label, c6, margins in rung_stats:
        n = len(margins)
        med = quantile(margins, 0.5)
        fracs = [sum(1 for m in margins if m < t) / n for t in THRESHOLDS]
        p(f"{label:<16}{c6:>9.5f}{n:>7}{med:>11.6f}" +
          "".join(f"{fr:>10.4f}" + "%" for fr in fracs))
        core6_vals.append(c6)
        med_margins.append(med)
        for t, fr in zip(THRESHOLDS, fracs):
            frac_vals[t].append(fr)

    # -----------------------------------------------------------------------
    # B: Spearman correlations
    # -----------------------------------------------------------------------
    p("\n--- Spearman correlations (acc_norm metric) ---")
    n_rungs = len(core6_vals)
    p(f"  n_rungs = {n_rungs} (exact permutation uses {math.factorial(n_rungs)} permutations)")

    if n_rungs >= 2:
        rho_med, p_med = spearman_with_exact_p(core6_vals, med_margins)
        p(f"\n  Spearman(core6, median_margin)       rho={rho_med:.4f}  exact_p={p_med:.4f}")
        p(f"    Interpretation: {'positive' if rho_med > 0 else 'negative'} correlation "
          f"({'intact has larger margin' if rho_med > 0 else 'damaged has larger margin'})")
        p(f"    Note: core6 high = intact model; margin high = easier decision = less fragile")
        p(f"    So rho>0 means 'intact model has larger margins' = fragility DECREASES with damage")

        for t in THRESHOLDS:
            rho_f, p_f = spearman_with_exact_p(core6_vals, frac_vals[t])
            direction = "fragile MORE with damage" if rho_f < 0 else "fragile LESS with damage"
            p(f"  Spearman(core6, frac_neartie<{t:.3f}) rho={rho_f:.4f}  exact_p={p_f:.4f}  ({direction})")

    # PRIMARY metric: frac < 0.005
    p(f"\n  PRIMARY ENDPOINT (acc_norm margin < {PRIMARY_THRESHOLD}):")
    rho_prim, p_prim = spearman_with_exact_p(core6_vals, frac_vals[PRIMARY_THRESHOLD])
    p(f"    Spearman rho={rho_prim:.4f}  exact_p={p_prim:.4f}  n={n_rungs}")
    if p_prim < 0.05:
        p(f"    -> SIGNIFICANT at alpha=0.05")
    elif p_prim < 0.10:
        p(f"    -> MARGINAL (p<0.10)")
    else:
        p(f"    -> NOT SIGNIFICANT at alpha=0.05 (p={p_prim:.4f})")

    # Flip rate vs core6: need bs perturbation data
    p("\n--- Spearman for flip_rate (acc_norm) vs core6 ---")
    p("  This requires bs8 vs bs16 perturbation data (see PART D below)")
    p("  If bs16 dirs are missing, this section will be 'N/A'")

    # -----------------------------------------------------------------------
    # C: Task-level margin breakdown
    # -----------------------------------------------------------------------
    p("\n--- Per-task acc_norm margin (median) across rungs ---")
    p(f"{'task':<16}" + "".join(f"{label[:12]:>13}" for label, _, _ in RUNGS))
    for task in CORE6:
        row = f"{task:<16}"
        for label, dname, c6 in RUNGS:
            data = load_per_example(dname, task)
            if data is None:
                row += f"{'N/A':>13}"
                continue
            margs = [accnorm_margin(r) for r in data.values() if not r.get("nan")]
            med = quantile(margs, 0.5) if margs else float("nan")
            row += f"{med:>13.6f}"
        p(row)

    # -----------------------------------------------------------------------
    # D: Batch-size perturbation
    # -----------------------------------------------------------------------
    bs_results = compute_bs_perturbation()

    # -----------------------------------------------------------------------
    # E: LOO mediation
    # -----------------------------------------------------------------------
    if bs_results:
        loo_mediation(bs_results)

    # -----------------------------------------------------------------------
    # VERDICT
    # -----------------------------------------------------------------------
    p("\n" + "=" * 70)
    p("VERDICT (acc_norm metric, H20 single-disk, bs8 data)")
    p("=" * 70)
    p(f"\n  Core6 values (monotone decreasing = more damage):")
    for label, dname, c6 in RUNGS:
        p(f"    {label:<16}  core6={c6:.5f}")

    p(f"\n  Median acc_norm margin per rung:")
    for label, c6, margins in rung_stats:
        med = quantile(margins, 0.5)
        n = len(margins)
        nt = sum(1 for m in margins if m < PRIMARY_THRESHOLD)
        p(f"    {label:<16}  median={med:.6f}  n={n}  frac<{PRIMARY_THRESHOLD}={nt/n:.4f}")

    p(f"\n  Spearman(core6, frac_neartie) = {rho_prim:.4f}, exact p = {p_prim:.4f}")
    p(f"  Spearman(core6, median_margin) = {rho_med:.4f}, exact p = {p_med:.4f}")

    # Verdict
    if p_prim < 0.05 and rho_prim < 0:
        verdict = "SUPPORTED: more damaged model has significantly more near-ties at acc_norm metric"
    elif p_prim < 0.10 and rho_prim < 0:
        verdict = "MARGINALLY SUPPORTED (p<0.10): trend present but below alpha=0.05"
    elif abs(rho_prim) < 0.3:
        verdict = "NOT SUPPORTED: near-zero correlation, no monotone fragility pattern at acc_norm metric"
    elif rho_prim > 0:
        verdict = "REVERSED: intact model has more near-ties (unexpected direction)"
    else:
        verdict = f"NOT SIGNIFICANT: rho={rho_prim:.4f} but p={p_prim:.4f} (n=6, exact)"
    p(f"\n  FINAL VERDICT: {verdict}")

    p("\n  COMPARISON WITH OLD raw-logprob (acc) ANALYSIS:")
    p("  - Old analysis (xarch.py/pert.py/margin_flip.py) used option_scores (raw sum-logprob)")
    p("    which is the ACC decision criterion, NOT the acc_norm criterion used in Table 4.")
    p("  - acc_norm uses norm_scores = option_scores / candidate_char_len.")
    p("  - The acc_norm margin scale is ~100x smaller than raw-logprob scale.")
    p("  - The near-tie threshold must be rescaled accordingly (0.005 nat/char).")
    p("  - For winogrande: acc == acc_norm (both options have identical norm_len).")
    p("  - For other tasks: acc and acc_norm can give different argmax on 20-40% of items.")

    p("\n  CAVEATS / WHAT IS MISSING:")
    p("  - Cross-architecture (L20A vs H20) flip analysis NOT done (.252 was unavailable).")
    p("    This is the primary perturbation source from the original proposal.")
    p("  - bs16 dirs may not exist; if so, batch-size perturbation section is empty.")
    p("  - n=6 rungs: exact p can only reach 1/720=0.00139 minimum (limited power).")
    p("  - 'Fragility' in this script = near-tie density, which is a WITHIN-RUNG property.")
    p("    The original question was about ACROSS-PROTOCOL flip rate; near-tie density")
    p("    is a necessary but not sufficient condition for flip sensitivity.")

    # -----------------------------------------------------------------------
    # Write to file
    # -----------------------------------------------------------------------
    out_path = os.path.join(os.path.dirname(__file__), "accnorm_results.txt")
    with open(out_path, "w") as f:
        f.write("\n".join(out_lines) + "\n")
    print(f"\n[WRITTEN] {out_path}")


if __name__ == "__main__":
    main()
