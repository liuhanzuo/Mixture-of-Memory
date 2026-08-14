#!/usr/bin/env python3
"""B04 Qwen cross-family verdict (n=6).

The OLMo-2-7B side established Spearman(core6, median_margin) = +1.0000 and
Spearman(core6, frac<0.005) = -1.0000 at the n=6 exact-permutation lower bound
p = 0.0028. That verdict scoped itself explicitly:

    "NOT established beyond OLMo-2-7B. Cross-family replication
     (Qwen prune-heal ladder) is the next kill test."

This is that test. Same protocol, same margin definition, same nulls, run over
the six Qwen3-8B rungs whose bs16 downstream eval just completed on .21:

    qwen_base_full36_bs16          (36L, undamaged)
    qwen_f12k2_step200000_bs16     (14L, keep12+fresh2 @ 200k, most healed)
    qwen_f12k2_step20000_bs16      (14L, @ 20k)
    qwen_f12k2_step2000_bs16       (14L, @ 2k, minimally healed)
    qwen_f12k4_step2000_bs16       (16L, keep12+fresh4 @ 2k, wider fresh block)
    qwen_scratch14L_step2000_bs16  (14L, from_scratch @ 2k, no inheritance)

Margin definition (identical to OLMo side, verified by inspecting the OLMo
analyzer): margin_i = norm_scores[gold] - max_{d != gold} norm_scores[d], with
raw option_scores as fallback. The enrich step already wrote norm_lens/
norm_scores for all Qwen rungs, verified by [ENRICH CHECK] norm_scores present.

Outcomes:
  * rho at n=6 hits the exact-permutation floor (p=0.0028)
      -> "damage compresses per-item decision margins" is cross-family;
         B04 promotable to paper<X> (only CPU novelty check remains)
  * rho stays strong but not at floor
      -> B04 stays as OLMo-primary with Qwen as a strengthening appendix
  * rho flips sign or goes flat
      -> the effect is OLMo-2-specific; B04's headline must be scoped to
         a single-model observation.
"""
from __future__ import annotations

import json
import os
import statistics
import sys
from itertools import permutations
from pathlib import Path

ROOT = Path("qwen3_probe2_downstream_results")
RUNGS = [
    ("Qwen3-8B base (36L)",          "qwen_base_full36_bs16"),
    ("f12k2 @ step200000 (14L)",     "qwen_f12k2_step200000_bs16"),
    ("f12k2 @ step20000 (14L)",      "qwen_f12k2_step20000_bs16"),
    ("f12k2 @ step2000 (14L)",       "qwen_f12k2_step2000_bs16"),
    ("f12k4 @ step2000 (16L)",       "qwen_f12k4_step2000_bs16"),
    ("scratch14L @ step2000 (14L)",  "qwen_scratch14L_step2000_bs16"),
]
TASKS = ["hellaswag", "arc_challenge", "arc_easy", "piqa", "openbookqa", "winogrande"]
THRESHOLDS = [0.001, 0.005, 0.010]


def margin_from_row(r):
    scores = r.get("norm_scores") or r.get("option_scores")
    gold = r["gold_letter"]
    if not scores or gold not in scores:
        return None
    gs = scores[gold]
    others = [v for k, v in scores.items() if k != gold]
    return gs - max(others)


def collect_margins(rung_dir):
    margs = []
    for task in TASKS:
        p = ROOT / rung_dir / f"per_example_{task}.jsonl"
        if not p.exists():
            print(f"    WARN missing {p}", file=sys.stderr)
            continue
        n = 0
        with open(p) as f:
            for line in f:
                d = json.loads(line)
                m = margin_from_row(d)
                if m is not None:
                    margs.append(m)
                    n += 1
        print(f"    {task:16s} n={n}", file=sys.stderr)
    return margs


def rank(v):
    ordered = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    for k, i in enumerate(ordered):
        r[i] = k + 1
    return r


def spearman(x, y):
    n = len(x)
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = sum((rx[i] - mx) ** 2 for i in range(n)) ** 0.5
    dy = sum((ry[i] - my) ** 2 for i in range(n)) ** 0.5
    return num / (dx * dy) if dx * dy else 0.0


def exact_p_two_sided(x, y):
    obs = abs(spearman(x, y))
    tot = hits = 0
    for perm in permutations(y):
        tot += 1
        if abs(spearman(x, list(perm))) >= obs - 1e-12:
            hits += 1
    return hits / tot


def main():
    core6 = {}
    for label, rd in RUNGS:
        p = ROOT / rd / "summary.json"
        d = json.load(open(p))
        tasks = d.get("tasks", d)
        s = 0.0
        for t in TASKS:
            an = tasks[t].get("acc_norm")
            if an is None:
                an = tasks[t].get("acc")
            s += an
        core6[label] = s / len(TASKS)

    print("=== Core6 (acc_norm) per Qwen rung ===")
    for label, _ in RUNGS:
        print(f"  {label:32s}  {core6[label]:.4f}")

    print("\n=== Per-rung margin stats (Qwen3-8B, 6 core6 tasks pooled) ===")
    frag = {}
    for label, rd in RUNGS:
        print(f"\n  {label}: reading per-item jsonls", file=sys.stderr)
        margs = collect_margins(rd)
        abs_m = [abs(x) for x in margs]
        n = len(abs_m)
        med = statistics.median(abs_m)
        stats = {"n": n, "median_margin": med}
        for th in THRESHOLDS:
            stats[f"frac_lt_{th}"] = sum(1 for x in abs_m if x < th) / n
        frag[label] = stats
        print(f"  {label:32s}  n={n:6d}  median={med:.4f}  "
              f"<.001={stats['frac_lt_0.001']*100:6.3f}%  "
              f"<.005={stats['frac_lt_0.005']*100:6.3f}%  "
              f"<.01={stats['frac_lt_0.01']*100:6.3f}%")

    labels = [l for l, _ in RUNGS]
    x = [core6[l] for l in labels]

    print("\n=== Spearman(core6, fragility) Qwen n=6 (exact p, 720 perms) ===")
    olmo_ref = {
        "median_margin": ("+1.0000", "0.0028"),
        "frac_lt_0.005": ("-1.0000", "0.0028"),
        "frac_lt_0.001": ("-0.9429", "0.0083"),
        "frac_lt_0.01":  ("-0.9429", "0.0167"),
    }
    print(f"  {'metric':16s}  {'rho':>8s}  {'exact p':>8s}   {'OLMo-side':>18s}")
    for m in ["median_margin", "frac_lt_0.001", "frac_lt_0.005", "frac_lt_0.01"]:
        y = [frag[l][m] for l in labels]
        r = spearman(x, y)
        p = exact_p_two_sided(x, y)
        oref = olmo_ref[m]
        print(f"  {m:16s}  {r:+8.4f}  {p:8.4f}   rho={oref[0]}, p={oref[1]}")

    out = {"n_rungs": len(RUNGS), "rungs": labels, "core6": core6,
           "fragility_stats": frag}
    outp = Path("proposal/backlog/B04-eval-fragility-incubator/evidence/B04_Qwen_6rung_bs16_analysis.json")
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"\nwrote {outp}")


if __name__ == "__main__":
    main()
