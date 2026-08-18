#!/usr/bin/env python3
"""Assemble span-length stratified arm scores into comparison tables.

Reports, per stratum, both grading axes, and for each arm:
  raw pass@1, gold ceiling, ceiling-conditioned pass@1.
Plus paired McNemar tests between arms on the ceiling-conditioned subsets,
and a bootstrap CI on the (short-stratum minus long-stratum) gap per arm --
which is the actual quantity the DreamOn "advantage concentrated at extreme
lengths" hypothesis is about.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ORDER = ["0-4", "5-8", "9-16", "17-32", "33-64", "65-128", "129+"]


def load(split, arms, root):
    out = {}
    for a in arms:
        p = Path(root) / f"score_{split}_{a}.json"
        if p.exists():
            out[a] = json.loads(p.read_text())
    return out


def mcnemar(a_pass, b_pass):
    """Exact binomial McNemar on paired boolean dicts keyed by task_id."""
    from math import comb

    keys = sorted(set(a_pass) & set(b_pass))
    b = sum(1 for k in keys if a_pass[k] and not b_pass[k])
    c = sum(1 for k in keys if b_pass[k] and not a_pass[k])
    n = b + c
    if n == 0:
        return b, c, 1.0
    lo = min(b, c)
    p = sum(comb(n, i) for i in range(lo + 1)) / (2 ** n) * 2
    return b, c, min(1.0, p)


def boot_gap(rows, which, short_strata, long_strata, n_boot=10000, seed=7):
    """Bootstrap CI on cond_pass(short) - cond_pass(long) for one arm."""
    rng = np.random.default_rng(seed)
    gk = f"gold_{which}_pass"
    pk = f"{which}_pass"
    s = [r[pk] for r in rows if r["stratum"] in short_strata and r[gk]]
    l = [r[pk] for r in rows if r["stratum"] in long_strata and r[gk]]
    if not s or not l:
        return None
    s = np.array(s, dtype=float); l = np.array(l, dtype=float)
    obs = s.mean() - l.mean()
    d = np.empty(n_boot)
    for i in range(n_boot):
        d[i] = rng.choice(s, s.size, replace=True).mean() - rng.choice(l, l.size, replace=True).mean()
    return {"gap": float(obs), "ci95": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "n_short": int(s.size), "n_long": int(l.size)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="runs/spanlen")
    ap.add_argument("--splits", nargs="+", default=["RandomSpan", "MultiLine"])
    ap.add_argument("--arms", nargs="+", default=["qwen_fim", "dream_fim", "dreamon_fim"])
    ap.add_argument("--output", default="runs/spanlen/spanlen_summary.json")
    args = ap.parse_args()

    summary = {}
    for split in args.splits:
        S = load(split, args.arms, args.root)
        if not S:
            continue
        present = [a for a in args.arms if a in S]
        print("=" * 100)
        print(f"SPLIT {split}   arms present: {present}")
        for which in ("base", "plus"):
            print(f"\n--- grading axis = {which} " + "-" * 60)
            hdr = f"{'stratum':<9}{'n':>6}{'ceil':>8}"
            for a in present:
                hdr += f"{a[:11]+'_raw':>17}{a[:11]+'_cond':>18}"
            print(hdr)
            for k in ORDER:
                any_ = [a for a in present if k in S[a]["by_stratum"]]
                if not any_:
                    continue
                st0 = S[any_[0]]["by_stratum"][k]
                line = f"{k:<9}{st0['n']:>6}{(st0[f'gold_ceiling_{which}'] or 0):>8.3f}"
                for a in present:
                    st = S[a]["by_stratum"].get(k)
                    if st is None:
                        line += f"{'-':>17}{'-':>18}"
                        continue
                    raw = st[f"pass_at_1_{which}"]
                    cond = st[f"cond_pass_{which}"]
                    line += f"{raw:>17.4f}" + (f"{cond:>18.4f}" if cond is not None else f"{'-':>18}")
                print(line)
            line = f"{'OVERALL':<9}{S[present[0]]['n']:>6}{S[present[0]]['overall'][f'gold_ceiling_{which}']:>8.3f}"
            for a in present:
                line += f"{S[a]['overall'][f'pass_at_1_{which}']:>17.4f}{S[a]['overall'][f'cond_pass_{which}']:>18.4f}"
            print(line)

        print("\n--- termination accounting (separate from grading) ---")
        for a in present:
            o = S[a]["overall"]
            print(f"  {a:<14} truncated={o['truncated']:<5} aborts={o['aborts']:<4} "
                  f"gen_errors={o['generation_errors']:<4} not_parseable={o['not_parseable']:<5} "
                  f"latency_s_mean={o['latency_s_mean']:.3f}")

        # long-vs-short gap per arm, ceiling-conditioned
        short = {"0-4", "5-8", "9-16"}
        strata_present = {k for a in present for k in S[a]["by_stratum"]}
        long_ = {k for k in ("33-64", "65-128", "129+") if k in strata_present}
        print(f"\n--- ceiling-conditioned gap short{sorted(short)} - long{sorted(long_)} (bootstrap 95% CI) ---")
        gaps = {}
        for which in ("base", "plus"):
            gaps[which] = {}
            for a in present:
                g = boot_gap(S[a]["per_row"], which, short, long_)
                gaps[which][a] = g
                if g:
                    print(f"  {which:<5} {a:<14} gap={g['gap']:+.4f}  CI95=[{g['ci95'][0]:+.4f},{g['ci95'][1]:+.4f}]  "
                          f"n_short={g['n_short']} n_long={g['n_long']}")

        # paired McNemar between arms, per stratum, ceiling-conditioned plus axis
        print("\n--- paired McNemar (ceiling-conditioned, plus axis) ---")
        mc = {}
        for i, a in enumerate(present):
            for bb in present[i+1:]:
                ra = {r["task_id"]: r["plus_pass"] for r in S[a]["per_row"] if r["gold_plus_pass"]}
                rb = {r["task_id"]: r["plus_pass"] for r in S[bb]["per_row"] if r["gold_plus_pass"]}
                nb, nc, p = mcnemar(ra, rb)
                mc[f"{a}_vs_{bb}"] = {"a_only": nb, "b_only": nc, "p": p}
                print(f"  {a} vs {bb}: {a}-only={nb} {bb}-only={nc} p={p:.3g}")
                for k in ORDER:
                    ka = {r["task_id"]: r["plus_pass"] for r in S[a]["per_row"]
                          if r["gold_plus_pass"] and r["stratum"] == k}
                    kb = {r["task_id"]: r["plus_pass"] for r in S[bb]["per_row"]
                          if r["gold_plus_pass"] and r["stratum"] == k}
                    if not ka:
                        continue
                    nb2, nc2, p2 = mcnemar(ka, kb)
                    mc[f"{a}_vs_{bb}@{k}"] = {"a_only": nb2, "b_only": nc2, "p": p2, "n": len(ka)}
                    print(f"      {k:<8} n={len(ka):<5} {a}-only={nb2:<4} {bb}-only={nc2:<4} p={p2:.3g}")

        summary[split] = {
            "arms": present,
            "by_stratum": {a: S[a]["by_stratum"] for a in present},
            "overall": {a: S[a]["overall"] for a in present},
            "gaps": gaps, "mcnemar": mc,
        }

    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nwrote {args.output}")


if __name__ == "__main__":
    main()
