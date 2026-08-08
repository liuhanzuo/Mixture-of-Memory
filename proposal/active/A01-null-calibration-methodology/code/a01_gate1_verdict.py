#!/usr/bin/env python3
"""A01 gate-1 verdict: does the MMLU letter-interface failure replicate off OLMo-2?

The claim under test (A01's generality claim)
--------------------------------------------
A01 argues the *letter* MC interface is an unreliable instrument: on damaged
OLMo-2 arms it decays toward a constant predictor, sinks to/below its own
best-constant floor, and its argmax is decided by exact ties (an input-blind
operation). The *content* interface (scoring choice text, label-free) is offered
as the interface that stays input-driven. A01's own Kill condition includes:

    "第三家族和第二 benchmark 均不复现 interface failure"

So this decides whether the letter pathology is an OLMo-2 property or general.

Nulls -- never 0.25
-------------------
Recomputed from the MMLU gold-label distribution, a property of the benchmark
shared by every arm:
  * letter  -> best-constant letter, argmax_L P(gold == L)
  * content -> the same best-constant floor applies to any argmax-over-4-options
               interface, so both interfaces are compared to it; the
               longest-option split-tie null (.2845 on this item set) is also
               reported because it is A01's pre-registered content-side null.

Per-arm, per-interface statistics
---------------------------------
  * accuracy and residual above the interface's own null
  * exact two-sided binomial p vs that null (log-space, n=14042)
  * paired bootstrap 95% CI on (acc - null), n_boot=10000
  * degeneracy: modal-prediction share, and the accuracy the arm's OWN modal
    constant would score (the "is it just emitting one letter?" test)
  * exact-tie rate among the 4 option scores (A01's posited mechanism)
  * exact McNemar letter-vs-content on the same items (paired, same arm)
"""

from __future__ import annotations

import glob
import json
import math
import os
import sys
from collections import Counter
from math import lgamma

import numpy as np

ROOT = "olmo2_mmlu_content_results"
LETTERS = "ABCD"

GATE1_ARMS = [
    ("Llama-2-7B",    "gate1_llama2_7b",     "non-OLMo"),
    ("Llama-3-8B",    "gate1_llama3_8b",     "non-OLMo"),
    ("Qwen3-8B-Base", "gate1_qwen3_8b_base", "non-OLMo"),
]
OLMO_REFS = [
    ("OLMo-2-7B base",        "7B_base"),
    ("OLMo-2-7B keep14@200k", "7B_keep14_step200000"),
    ("OLMo-2-7B keep8@121k",  "7B_keep8_step121000"),
]


def load_rows(arm_dir: str) -> list[dict]:
    merged = os.path.join(ROOT, arm_dir, "per_example_mmlu.jsonl")
    if os.path.exists(merged):
        with open(merged) as f:
            return [json.loads(l) for l in f]
    shards = sorted(glob.glob(os.path.join(ROOT, arm_dir, "per_example_mmlu_shard*of*.jsonl")))
    rows = []
    for s in shards:
        with open(s) as f:
            rows += [json.loads(l) for l in f]
    return rows


def logpmf(i: int, n: int, p: float) -> float:
    return (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def exact_binom_two_sided(k: int, n: int, p0: float) -> float:
    """Two-sided exact binomial: sum of all outcomes no more likely than observed."""
    lobs = logpmf(k, n, p0)
    tot = 0.0
    for i in range(n + 1):
        li = logpmf(i, n, p0)
        if li <= lobs + 1e-9:
            tot += math.exp(li)
    return min(1.0, tot)


def boot_ci_paired(a: np.ndarray, b: np.ndarray, n_boot=10000, seed=0):
    """95% CI on mean(a) - mean(b), resampling ITEMS (so the pairing is kept)."""
    rng = np.random.default_rng(seed)
    d = a.astype(np.float64) - b.astype(np.float64)
    n = len(d)
    out = np.empty(n_boot)
    for i in range(n_boot):
        out[i] = d[rng.integers(0, n, n)].mean()
    return float(np.percentile(out, 2.5)) * 100, float(np.percentile(out, 97.5)) * 100


def exact_mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant pairs (binomial, p=0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.exp(logpmf(i, n, 0.5)) for i in range(k + 1))
    return min(1.0, 2 * tail)


def longest_option_split_tie_null(rows: list[dict]) -> float | None:
    """A01's pre-registered content-side null: pick the longest option; on a
    length tie, award the fraction 1/n_tied. Needs option text lengths, which the
    harness stores as content_norm.cont_tokens when present."""
    tot, n = 0.0, 0
    for r in rows:
        ct = (r.get("content_norm") or {}).get("cont_tokens")
        if not ct:
            return None
        gold = r["gold_letter"]
        mx = max(ct.values())
        winners = [k for k, v in ct.items() if v == mx]
        tot += (1.0 / len(winners)) if gold in winners else 0.0
        n += 1
    return tot / n if n else None


def analyse(label: str, arm_dir: str, family: str) -> dict | None:
    rows = load_rows(arm_dir)
    if not rows:
        print(f"  SKIP {label}: nothing under {ROOT}/{arm_dir}", file=sys.stderr)
        return None
    rows = [r for r in rows if not r.get("nan")]
    n = len(rows)
    gold = np.array([r["gold_letter"] for r in rows])

    gold_counts = Counter(gold.tolist())
    best_letter = max(LETTERS, key=lambda L: gold_counts.get(L, 0))
    const_null = gold_counts[best_letter] / n
    const_correct = (gold == best_letter).astype(np.int64)

    out = {"label": label, "arm_dir": arm_dir, "family": family, "n": n,
           "best_constant_rule": f"always-{best_letter}",
           "best_constant_null": const_null,
           "gold_letter_counts": dict(gold_counts)}

    lo_null = longest_option_split_tie_null(rows)
    if lo_null is not None:
        out["longest_option_split_tie_null"] = lo_null

    per_iface = {}
    for iface in ("letter", "content_norm"):
        if iface not in rows[0]:
            continue
        correct = np.array([1 if r[iface]["correct"] else 0 for r in rows], dtype=np.int64)
        pred = np.array([r[iface]["pred_letter"] for r in rows])
        pcounts = Counter(pred.tolist())
        modal_pred, modal_n = pcounts.most_common(1)[0]

        n_tie = 0
        for r in rows:
            sc = r[iface].get("scores") or {}
            if len(sc) >= 2:
                v = sorted(sc.values(), reverse=True)
                if v[0] == v[1]:
                    n_tie += 1

        acc = float(correct.mean())
        k = int(correct.sum())
        lo, hi = boot_ci_paired(correct, const_correct)
        d = {
            "acc": acc,
            "residual_pp_vs_best_constant": (acc - const_null) * 100,
            "exact_binom_p_vs_best_constant": exact_binom_two_sided(k, n, const_null),
            "boot95_residual_pp": [lo, hi],
            "modal_pred": modal_pred,
            "modal_share": modal_n / n,
            "own_modal_constant_acc": gold_counts.get(modal_pred, 0) / n,
            "exact_tie_rate": n_tie / n,
        }
        # is the arm distinguishable from its OWN modal constant? (degeneracy test)
        own_modal_correct = (gold == modal_pred).astype(np.int64)
        b = int(np.sum((correct == 1) & (own_modal_correct == 0)))
        c = int(np.sum((correct == 0) & (own_modal_correct == 1)))
        d["mcnemar_vs_own_modal_constant_p"] = exact_mcnemar(b, c)
        per_iface[iface] = d
    out["interfaces"] = per_iface

    # paired letter-vs-content on the same items
    if "letter" in per_iface and "content_norm" in per_iface:
        lc = np.array([1 if r["letter"]["correct"] else 0 for r in rows])
        cc = np.array([1 if r["content_norm"]["correct"] else 0 for r in rows])
        b = int(np.sum((lc == 1) & (cc == 0)))
        c = int(np.sum((lc == 0) & (cc == 1)))
        lo, hi = boot_ci_paired(lc, cc)
        out["letter_vs_content"] = {
            "delta_pp_letter_minus_content": (lc.mean() - cc.mean()) * 100,
            "boot95_pp": [lo, hi],
            "b_letter_right_content_wrong": b,
            "c_letter_wrong_content_right": c,
            "exact_mcnemar_p": exact_mcnemar(b, c),
        }
    return out


def main():
    results = []
    print("=== gate-1 arms (non-OLMo families) ===", file=sys.stderr)
    for label, arm_dir, fam in GATE1_ARMS:
        r = analyse(label, arm_dir, fam)
        if r:
            results.append(r)
    print("=== OLMo-2 reference arms ===", file=sys.stderr)
    for label, arm_dir in OLMO_REFS:
        r = analyse(label, arm_dir, "OLMo-2")
        if r:
            results.append(r)

    print(f"\n{'arm':24s} {'n':>6s} {'floor':>7s} | {'letter':>8s} {'resid':>8s} "
          f"{'modal%':>7s} {'tie%':>6s} | {'content':>8s} {'resid':>8s} | {'L-C pp':>8s} {'McN p':>9s}")
    print("-" * 122)
    for r in results:
        I = r["interfaces"]
        L = I.get("letter", {})
        C = I.get("content_norm", {})
        lc = r.get("letter_vs_content", {})
        print(f"{r['label']:24s} {r['n']:6d} {r['best_constant_null']:7.4f} | "
              f"{L.get('acc', float('nan')):8.4f} {L.get('residual_pp_vs_best_constant', float('nan')):+8.2f} "
              f"{L.get('modal_share', float('nan'))*100:6.1f}% {L.get('exact_tie_rate', float('nan'))*100:5.2f}% | "
              f"{C.get('acc', float('nan')):8.4f} {C.get('residual_pp_vs_best_constant', float('nan')):+8.2f} | "
              f"{lc.get('delta_pp_letter_minus_content', float('nan')):+8.2f} "
              f"{lc.get('exact_mcnemar_p', float('nan')):9.2e}")

    outp = "proposal/active/A01-null-calibration-methodology/evidence/a01_gate1_third_family.json"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    with open(outp, "w") as f:
        json.dump({"arms": results}, f, indent=2)
    print(f"\nwrote {outp}")

    print("\n=== VERDICT ===")
    non_olmo = [r for r in results if r["family"] == "non-OLMo"]
    n_replicate = 0
    for r in non_olmo:
        L = r["interfaces"]["letter"]
        pathological = (L["residual_pp_vs_best_constant"] <= 0
                        or L["modal_share"] > 0.5
                        or L["mcnemar_vs_own_modal_constant_p"] > 0.05)
        n_replicate += bool(pathological)
        why = []
        if L["residual_pp_vs_best_constant"] <= 0:
            why.append("at/below own floor")
        if L["modal_share"] > 0.5:
            why.append(f"modal {L['modal_share']*100:.0f}%")
        if L["mcnemar_vs_own_modal_constant_p"] > 0.05:
            why.append(f"indistinguishable from own modal constant (p={L['mcnemar_vs_own_modal_constant_p']:.2f})")
        print(f"  {r['label']:16s} letter {L['acc']:.4f} vs floor "
              f"{r['best_constant_null']:.4f} ({L['residual_pp_vs_best_constant']:+.2f}pp) -> "
              f"{'REPLICATES failure (' + ', '.join(why) + ')' if pathological else 'HEALTHY, no pathology'}")
    print(f"\n  {n_replicate}/{len(non_olmo)} non-OLMo families replicate the letter-interface failure.")
    if n_replicate == 0:
        print("  => A01 Kill condition clause 2 is TRIGGERED: the third-family case does NOT")
        print("     reproduce interface failure. A01's generality claim must be narrowed from")
        print("     'the letter interface is an unreliable instrument' to 'the letter interface")
        print("     degenerates in STRUCTURALLY DAMAGED OLMo-2 arms'. The protocol contribution")
        print("     (construct-appropriate nulls before comparison) survives; the general")
        print("     interface-invalidity claim does not.")


if __name__ == "__main__":
    main()
