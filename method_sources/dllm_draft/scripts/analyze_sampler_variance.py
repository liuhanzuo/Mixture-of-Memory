#!/usr/bin/env python
"""Variance decomposition of the 25-cell sampler_audit HE+ grid.

The factorial is UNBALANCED, so classical ANOVA is inappropriate.
We instead report:
  - Full-grid spread (all 25 vs the 21 plausible-only).
  - Per-factor MARGINAL SPREAD holding the other factors fixed at
    a reference cell (T=0.1, top_p=0.95, alg=entropy, alg_temp=0.0).
  - Same-cell replication floor (T=0.1 ref x4 seeds, T=0.0 ref dup,
    T=0.7 x4 seeds).
  - Scaffolding for a Spearman-rho HE+ vs MBPP+ ranking correlation
    once MBPP+ audit (task #178) lands its 25-cell summary.
"""
from __future__ import annotations
import json
from collections import defaultdict
from pathlib import Path

SUMMARY = Path("/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft/"
               "runs/sampler_audit_mirror/summary.json")
MBPP_SUMMARY = Path("/apdcephfs_wzc1/share_304376610/pighzliu_code/"
                    "dllm_draft/runs/sampler_audit_mbpp_mirror/summary.json")

REF = dict(temperature=0.1, top_p=0.95, alg="entropy", alg_temp=0.0)


def key_for(cell):
    s = cell["sampler"]
    return (s["temperature"], s["top_p"], s["alg"], s["alg_temp"])


def marginal(cells, axis, reference):
    """Vary one axis, hold the others = reference. Return (list of (val, base, plus))."""
    out = []
    for c in cells:
        s = c["sampler"]
        ok = all(s[k] == reference[k] for k in reference if k != axis)
        if ok:
            out.append((s[axis], c["base"], c["plus"], c["run"]))
    # dedupe on axis value (keep first)
    seen = {}
    for v, b, p, r in out:
        seen.setdefault(v, (b, p, r))
    return sorted(seen.items())


def spread(values):
    if not values:
        return None, None, None
    vs = [v for v in values if v is not None]
    return max(vs) - min(vs), min(vs), max(vs)


def main():
    cells = json.loads(SUMMARY.read_text())
    print(f"# 25-cell HE+ sampler grid — variance decomposition\n")
    print(f"Total cells: {len(cells)}\n")

    # -- Full-grid spread --
    bases = [c["base"] for c in cells]
    pluses = [c["plus"] for c in cells]
    print(f"## Full-grid spread (all 25 cells)")
    print(f"  HE  base: min={min(bases):.4f} max={max(bases):.4f}"
          f" spread={100*(max(bases)-min(bases)):.1f} pt")
    print(f"  HE+ plus: min={min(pluses):.4f} max={max(pluses):.4f}"
          f" spread={100*(max(pluses)-min(pluses)):.1f} pt\n")

    # Plausible-only = drop known-degenerate combos:
    #   alg=origin  (0.18-0.26 base — that's rank-baseline; kept it grade to prove floor)
    #   alg_temp=0.5 (single cell 0.31 base — extreme)
    plausible = [c for c in cells
                 if c["sampler"]["alg"] != "origin"
                 and c["sampler"]["alg_temp"] == 0.0]
    print(f"## Plausible-only spread (n={len(plausible)}, drop alg=origin & alg_temp=0.5)")
    pb = [c["base"] for c in plausible]
    pp = [c["plus"] for c in plausible]
    print(f"  HE  base: min={min(pb):.4f} max={max(pb):.4f}"
          f" spread={100*(max(pb)-min(pb)):.1f} pt")
    print(f"  HE+ plus: min={min(pp):.4f} max={max(pp):.4f}"
          f" spread={100*(max(pp)-min(pp)):.1f} pt\n")

    # -- Same-cell replication floor --
    print("## Same-cell replication floor")
    ref_seeds = [c for c in cells if c["run"] in
                 ("he_ref_T0.1_p0.95_entropy_at0", "he_ref_seed1",
                  "he_ref_seed2", "he_ref_seed3")]
    rb = [c["base"] for c in ref_seeds]
    rp = [c["plus"] for c in ref_seeds]
    print(f"  T=0.1 ref x4 (seed=None x4): base={rb}, plus={rp}")
    print(f"    base spread = {100*(max(rb)-min(rb)):.2f} pt,"
          f" plus spread = {100*(max(rp)-min(rp)):.2f} pt")

    dup00 = [c for c in cells if c["run"] in
             ("he_T0.0_p0.95_entropy_at0", "he_T0.0_p0.95_entropy_dup")]
    db = [c["base"] for c in dup00]
    dp = [c["plus"] for c in dup00]
    print(f"  T=0.0 dup x2: base={db}, plus={dp}")
    print(f"    base spread = {100*(max(db)-min(db)):.2f} pt,"
          f" plus spread = {100*(max(dp)-min(dp)):.2f} pt")

    seed7 = [c for c in cells if c["run"] in
             ("he_T0.7_p0.95_entropy_at0", "he_T0.7_seed1",
              "he_T0.7_seed2", "he_T0.7_seed3")]
    sb = [c["base"] for c in seed7]
    sp = [c["plus"] for c in seed7]
    print(f"  T=0.7 x4 seeds: base={sb}, plus={sp}")
    print(f"    base spread = {100*(max(sb)-min(sb)):.2f} pt,"
          f" plus spread = {100*(max(sp)-min(sp)):.2f} pt\n")

    # -- Per-factor marginal spread holding others = REF cell --
    print("## Per-factor marginal spread (holding others at reference)")
    print(f"REF = T={REF['temperature']}, top_p={REF['top_p']},"
          f" alg={REF['alg']}, alg_temp={REF['alg_temp']}\n")

    for axis, label in [("temperature", "T"),
                        ("top_p", "top_p"),
                        ("alg", "alg"),
                        ("alg_temp", "alg_temp")]:
        vals = marginal(cells, axis, REF)
        if not vals:
            print(f"  [{label}] no marginal cells\n")
            continue
        print(f"  [{label}] varied while others fixed at REF:")
        bs = []
        ps = []
        for v, (b, p, r) in vals:
            bs.append(b)
            ps.append(p)
            print(f"    {label}={v!r:14s} base={b:.4f} plus={p:.4f}  ({r})")
        print(f"    -> HE  base marginal spread = {100*(max(bs)-min(bs)):.1f} pt")
        print(f"    -> HE+ plus marginal spread = {100*(max(ps)-min(ps)):.1f} pt\n")

    # -- Search-budget: "best" cell on HE+ from these 25 --
    print("## Protocol-search-budget accounting (Piece 3 scaffolding)")
    # Best on HE+:
    best_plus = max(cells, key=lambda c: c["plus"])
    print(f"  HE+ best across 25 cells: {best_plus['run']} plus={best_plus['plus']:.4f}"
          f" (base={best_plus['base']:.4f})")
    # HE base best:
    best_base = max(cells, key=lambda c: c["base"])
    print(f"  HE  base best: {best_base['run']} base={best_base['base']:.4f}"
          f" (plus={best_base['plus']:.4f})")
    # Reference cell rank on plus:
    plus_sorted = sorted(cells, key=lambda c: -c["plus"])
    ref_run = "he_ref_T0.1_p0.95_entropy_at0"
    rank = [i for i, c in enumerate(plus_sorted) if c["run"] == ref_run][0] + 1
    print(f"  Reference cell rank on HE+ (among 25): {rank}/25\n")

    # -- Spearman-rho scaffolding (only compute when MBPP+ summary exists) --
    print("## Spearman-rho HE+ vs MBPP+ (25 cells)")
    if MBPP_SUMMARY.exists():
        mbpp = json.loads(MBPP_SUMMARY.read_text())
        # Key both sets by sampler tuple
        he_by_key = {}
        for c in cells:
            k = key_for(c)
            # if duplicates, keep the first (the audit already dedup'd seeds)
            he_by_key.setdefault(k, c["plus"])
        mbpp_by_key = {}
        for c in mbpp:
            k = key_for(c)
            mbpp_by_key.setdefault(k, c["plus"])
        common = sorted(set(he_by_key) & set(mbpp_by_key))
        if not common:
            print("  MBPP+ summary present but no keys overlap with HE+ grid.")
        else:
            he_vec = [he_by_key[k] for k in common]
            mbpp_vec = [mbpp_by_key[k] for k in common]
            # Spearman via rank correlation without scipy
            def ranks(xs):
                order = sorted(range(len(xs)), key=lambda i: xs[i])
                r = [0.0] * len(xs)
                # average ranks for ties
                i = 0
                while i < len(order):
                    j = i
                    while j+1 < len(order) and xs[order[j+1]] == xs[order[i]]:
                        j += 1
                    avg = (i + j) / 2 + 1
                    for k in range(i, j+1):
                        r[order[k]] = avg
                    i = j + 1
                return r
            def pearson(a, b):
                n = len(a)
                ma = sum(a)/n; mb = sum(b)/n
                num = sum((ai-ma)*(bi-mb) for ai,bi in zip(a,b))
                da = sum((ai-ma)**2 for ai in a) ** 0.5
                db = sum((bi-mb)**2 for bi in b) ** 0.5
                return num/(da*db) if da*db > 0 else None
            rho = pearson(ranks(he_vec), ranks(mbpp_vec))
            print(f"  common cells: {len(common)}")
            print(f"  Spearman rho(HE+ plus, MBPP+ plus) = {rho:.4f}"
                  "   <-- DO NOT QUOTE BARE")
            # --- Stratified rho: the bare all-cells rho is carried by the very
            # regimes the HE+ headline excludes (alg=origin, alg_temp=0.5), and
            # by three byte-identical alg cells that are ONE point, not three.
            def is_plausible(k):
                return k[2] != "origin" and k[3] == 0.0
            plaus = [k for k in common if is_plausible(k)]
            # dedupe cells whose (HE+, MBPP+) pair is byte-identical
            seen, distinct = set(), []
            for k in plaus:
                v = (he_by_key[k], mbpp_by_key[k])
                if v not in seen:
                    seen.add(v)
                    distinct.append(k)

            def rho_of(keys):
                if len(keys) < 3:
                    return None
                return pearson(ranks([he_by_key[k] for k in keys]),
                               ranks([mbpp_by_key[k] for k in keys]))

            def perm_p(keys):
                """Exact permutation p (one-sided and two-sided) on rho."""
                from itertools import permutations
                if not (3 <= len(keys) <= 8):
                    return None, None
                rh = ranks([he_by_key[k] for k in keys])
                rm = ranks([mbpp_by_key[k] for k in keys])
                r0 = pearson(rh, rm)
                tot = one = two = 0
                for p in permutations(rm):
                    r = pearson(rh, list(p))
                    tot += 1
                    if r is not None and r >= r0 - 1e-12:
                        one += 1
                    if r is not None and abs(r) >= abs(r0) - 1e-12:
                        two += 1
                return one / tot, two / tot

            r_pl = rho_of(plaus)
            r_di = rho_of(distinct)
            print(f"  plausible only (drop origin/alg_temp=0.5): n={len(plaus)}"
                  f" rho={r_pl:.4f}" if r_pl is not None else
                  f"  plausible only: n={len(plaus)} (too few)")
            if r_di is not None:
                p1, p2 = perm_p(distinct)
                extra = ""
                if p1 is not None:
                    extra = f"  exact perm p={p1:.4f} one-sided / {p2:.4f} two-sided"
                print(f"  ** distinct plausible points: n={len(distinct)}"
                      f" rho={r_di:.4f}{extra}   <-- HEADLINE for rank transfer")
                collapsed = [k for k in plaus if k not in distinct]
                if collapsed:
                    print(f"     collapsed as byte-identical duplicates: {collapsed}")
            # top-1 agreement:
            best_he = common[max(range(len(common)), key=lambda i: he_vec[i])]
            best_mb = common[max(range(len(common)), key=lambda i: mbpp_vec[i])]
            print(f"  HE+ best cell: {best_he}")
            print(f"  MBPP+ best cell: {best_mb}")
            print(f"  same top-1 cell? {best_he == best_mb}")
    else:
        print("  MBPP+ summary NOT yet on disk (task #178 in flight).")
        print(f"  Expected path: {MBPP_SUMMARY}")
        print("  Once landed, rerun this script to fill in rho.")


if __name__ == "__main__":
    main()
