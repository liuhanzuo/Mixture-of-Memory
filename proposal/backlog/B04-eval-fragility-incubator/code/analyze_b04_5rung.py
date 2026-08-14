#!/usr/bin/env python3
"""B04 eval-fragility: Spearman(core6, near-tie density) at n=5 rung, bs16, acc_norm 口径.

Rungs: base_full, keep14@200k, keep12@124k, keep10@83.5k, keep8@121k
Tasks (core6): hellaswag, arc_challenge, arc_easy, piqa, openbookqa, winogrande

Margin definition:
    margin_i = norm_scores[gold] - max_{d != gold} norm_scores[d]
    (positive => correct by acc_norm; near-zero => hard-to-decide item)

Fragility metric per rung:
    - median_margin (higher = less fragile)
    - frac(margin < 0.005) PRIMARY (higher = more near-ties = more fragile)
    - frac(margin < 0.001)
    - frac(margin < 0.010)

Then Spearman rho vs core6 (acc_norm mean over 6 tasks).
Exact permutation p at n=5 (5! = 120 perms).
"""

from __future__ import annotations

import json
import os
import statistics
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path("olmo2_downstream_results")
RUNGS = [
    ("base_full",     "7B_base_full_bs16"),
    ("keep14@200k",   "7B_keep14_step200000_bs16"),
    ("keep12@124k",   "7B_keep12_step124000_bs16"),
    ("keep10@83.5k",  "7B_keep10_step83500_bs16"),
    ("keep8@121k",    "7B_keep8_step121000_bs16"),
]
TASKS = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "openbookqa", "winogrande"]
THRESHOLDS = [0.001, 0.005, 0.010]


def compute_margin_from_norm_scores(scores: dict, gold: str) -> float:
    gold_s = scores[gold]
    others = [scores[k] for k in scores if k != gold]
    return gold_s - max(others)


def compute_margins_for_rung(rung_dir: Path) -> list[float]:
    all_margins = []
    for task in TASKS:
        p = rung_dir / f"per_example_{task}.jsonl"
        if not p.exists():
            print(f"  WARN: missing {p}", file=sys.stderr)
            continue
        n = 0
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                # winogrande uses raw option_scores (no norm) — fall back
                if "norm_scores" in d and d["norm_scores"]:
                    scores = d["norm_scores"]
                else:
                    scores = d["option_scores"]
                gold = d["gold_letter"]
                if gold not in scores:
                    continue
                m = compute_margin_from_norm_scores(scores, gold)
                all_margins.append(m)
                n += 1
        print(f"    {task}: n={n}", file=sys.stderr)
    return all_margins


def spearman_rho(x: list[float], y: list[float]) -> float:
    n = len(x)
    rx = _rank(x)
    ry = _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    if dx * dy == 0:
        return 0.0
    return num / (dx * dy)


def _rank(v: list[float]) -> list[float]:
    idx = sorted(range(len(v)), key=lambda i: v[i])
    rk = [0.0] * len(v)
    for r, i in enumerate(idx):
        rk[i] = r + 1
    return rk


def exact_p_two_sided(x: list[float], y: list[float]) -> float:
    obs = abs(spearman_rho(x, y))
    n = len(x)
    hits = 0
    total = 0
    for perm in permutations(y):
        total += 1
        if abs(spearman_rho(x, list(perm))) >= obs - 1e-12:
            hits += 1
    return hits / total


def main():
    # ---- Core6 from summary.json ----
    core6 = {}
    for label, rung_dir in RUNGS:
        p = ROOT / rung_dir / "summary.json"
        d = json.load(open(p))
        tasks = d.get("tasks", d)
        s = 0.0
        for t in TASKS:
            an = tasks[t].get("acc_norm")
            if an is None:  # winogrande has no distinct acc_norm; use acc
                an = tasks[t]["acc"]
            s += an
        core6[label] = s / len(TASKS)

    print("=== Core6 (acc_norm) per rung ===")
    for label, _ in RUNGS:
        print(f"  {label:16s}  {core6[label]:.4f}")

    # ---- Per-rung margin distributions ----
    print("\n=== Per-rung margin stats ===")
    frag = {}  # dict: label -> dict[metric] -> value
    for label, rung_dir in RUNGS:
        rd = ROOT / rung_dir
        print(f"\n  {label}: reading per-item jsonls", file=sys.stderr)
        margins = compute_margins_for_rung(rd)
        margins_abs = [abs(m) for m in margins]
        med = statistics.median(margins_abs)
        stats = {"median_margin": med, "n": len(margins)}
        for th in THRESHOLDS:
            stats[f"frac_lt_{th}"] = sum(1 for m in margins_abs if m < th) / len(margins_abs)
        frag[label] = stats
        print(f"  {label:16s}  n={len(margins):6d}  median={med:.4f}  "
              f"frac<.001={stats['frac_lt_0.001']*100:.3f}%  "
              f"frac<.005={stats['frac_lt_0.005']*100:.3f}%  "
              f"frac<.01={stats['frac_lt_0.01']*100:.3f}%")

    # ---- Spearman ----
    labels = [l for l, _ in RUNGS]
    x = [core6[l] for l in labels]
    print("\n=== Spearman(core6, fragility metric) at n=5 (exact p, 120 perms) ===")
    for metric in ["median_margin", "frac_lt_0.001", "frac_lt_0.005", "frac_lt_0.01"]:
        y = [frag[l][metric] for l in labels]
        rho = spearman_rho(x, y)
        p = exact_p_two_sided(x, y)
        expect = "+" if metric == "median_margin" else "-"
        sign_ok = ("+" if rho > 0 else "-") == expect
        print(f"  Spearman(core6, {metric:16s})  rho={rho:+.4f}  exact_p={p:.4f}  "
              f"expected_sign={expect}  match={sign_ok}")

    # ---- write JSON ----
    out = {
        "n_rungs": len(RUNGS),
        "rungs": labels,
        "core6": core6,
        "fragility_stats": frag,
    }
    outp = Path("olmo2_downstream_results/B04_5rung_bs16_analysis.json")
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
