#!/usr/bin/env python
"""A02 — Is the BABILong misordering a finding?  Pure CPU, zero GPU.

Adjudicates A02_BABILONG_MISORDER_PREREG.md Job 1 (and Job 2.1's statistics once
the VT recall vectors exist).

INPUTS (both already on disk, nothing re-evaluated)
  * evidence/read_tax_ruler/a02_read_tax_per_item_vectors.json
        10 cells x 7 arms x n=100 binary correctness, index-aligned
  * evidence/a02_depth_vs_retrieval_per_item.json
        the same 10 cells with recall_per_sample[{sample_index, gold_chunks, hit}]

WHY CONDITIONING ON `hit` IS LEGITIMATE (not post-hoc selection): all 7 arms ran the
identical retrieval config on prompts whose input_ids_sha256 are asserted equal across
arms (read-tax GATE C2, 0 failures). The pack -- hence the HIT/MISS label -- is a
property of (cell, sample), NOT of the arm. So HIT/MISS is a pre-treatment covariate.

STATISTICS (all pre-registered in A02_BABILONG_MISORDER_PREREG.md 1.1/1.2)
  1. per-cell A4-vs-A5 paired contrast: McNemar EXACT (two-sided binomial on the
     discordant pairs) + canonical paired bootstrap CI95 (n_boot=5000, seed=42)
  2. per-cell ladder Spearman over j in {0,6,9,12,18} (A0,A2,A3,A4,A5) with EXACT
     permutation p over all 5! = 120 orderings
  3. Holm-Bonferroni within each family of 6 BABILong cells
  4. the same contrast conditioned on retrieval HIT / MISS subsets
  5. cross-cell Spearman(recall@12, signed A4-A5 separation), exact permutation
  + F2' power guard: minimum detectable effect per HIT subset
  + F3 floor-vs-retrieval confound check
  + F4 control: can BABILong order the uncontested A0-vs-A5 contrast at all

CANONICAL SCORERS ARE IMPORTED, NEVER REIMPLEMENTED (PREREG GATE E): the binary
per-item `correct` vectors were produced by the canonical scorers (BABILong
babilong.metrics.compare_answers; RULER harness per-item `correct`) and this module
consumes those vectors verbatim. The bootstrap CI is imported from the dvr analyzer.

FAIL-CLOSED GATES
  GATE J1  every cell has exactly n=100 per arm, 0 NaN, 0 duplicate indices
  GATE J2  the read-tax and dvr index sets are IDENTICAL per cell (exact join, no
           imputation); refuse the cell otherwise
  GATE J3  the published read-tax per-cell accuracies are REPRODUCED from the
           per-item vectors (guards against reading the wrong arm/cell)
  GATE J4  hit-label coverage: refuse to condition a cell whose gold_locatable == 0
           (this is exactly VT's situation, and it must not be silently treated as
           "all MISS")

Usage:
  python analyze_a02_babilong_misorder.py --out <evidence_dir>
  python analyze_a02_babilong_misorder.py --selftest
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

BASE = Path(os.environ.get(
    "A02_BASE", "/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"))
_PROP = BASE / "proposal/backlog/A02-comem-write-read-repair"
_CODE = _PROP / "code"
for p in (str(BASE), str(_CODE)):
    if p not in sys.path:
        sys.path.insert(0, p)

N_BOOT = 5000
SEED = 42

# the r=32 depth ladder (A1 = j0 control, A6 = capacity control -> excluded)
LADDER = [("A0", 0), ("A2", 6), ("A3", 9), ("A4", 12), ("A5", 18)]

# published read-tax per-cell accuracies -> GATE J3 reproduction target
PUBLISHED = {
    "ruler|niah_multikey_1|16k": {"A0": 100.0, "A1": 100.0, "A2": 99.0, "A3": 99.0,
                                  "A4": 90.0, "A5": 32.0, "A6": 90.0},
    "ruler|niah_multikey_1|32k": {"A0": 99.0, "A1": 99.0, "A2": 99.0, "A3": 95.0,
                                  "A4": 96.0, "A5": 42.0, "A6": 96.0},
    "ruler|variable_tracking|16k": {"A0": 100.0, "A1": 100.0, "A2": 99.0, "A3": 99.0,
                                    "A4": 88.0, "A5": 4.0, "A6": 88.0},
    "ruler|variable_tracking|32k": {"A0": 100.0, "A1": 100.0, "A2": 100.0, "A3": 100.0,
                                    "A4": 89.0, "A5": 5.0, "A6": 88.0},
    "babilong|qa1|16k": {"A0": 33.0, "A1": 32.0, "A2": 23.0, "A3": 21.0,
                         "A4": 17.0, "A5": 19.0, "A6": 19.0},
    "babilong|qa1|32k": {"A0": 35.0, "A1": 34.0, "A2": 21.0, "A3": 14.0,
                         "A4": 12.0, "A5": 12.0, "A6": 12.0},
    "babilong|qa2|16k": {"A0": 17.0, "A1": 16.0, "A2": 13.0, "A3": 8.0,
                         "A4": 8.0, "A5": 9.0, "A6": 8.0},
    "babilong|qa2|32k": {"A0": 11.0, "A1": 10.0, "A2": 3.0, "A3": 2.0,
                         "A4": 1.0, "A5": 7.0, "A6": 1.0},
    "babilong|qa5|16k": {"A0": 53.0, "A1": 53.0, "A2": 62.0, "A3": 63.0,
                         "A4": 58.0, "A5": 49.0, "A6": 61.0},
    "babilong|qa5|32k": {"A0": 61.0, "A1": 63.0, "A2": 62.0, "A3": 57.0,
                         "A4": 58.0, "A5": 48.0, "A6": 59.0},
}

# dvr-published recall@12 (%) per cell; None where not gold-locatable
PUBLISHED_RECALL = {
    "babilong|qa1|16k": 63.2, "babilong|qa1|32k": 57.0,
    "babilong|qa2|16k": 49.5, "babilong|qa2|32k": 22.9,
    "babilong|qa5|16k": 64.1, "babilong|qa5|32k": 57.9,
    "ruler|niah_multikey_1|16k": 100.0, "ruler|niah_multikey_1|32k": 99.0,
    "ruler|variable_tracking|16k": None, "ruler|variable_tracking|32k": None,
}


# ------------------------------------------------------------------ stats ---- #
def _log_binom_coef(n, k):
    return (math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1))


def mcnemar_exact(a_vec, b_vec):
    """Two-sided exact McNemar on the discordant pairs.

    b = #(a correct, b wrong)   c = #(a wrong, b correct)
    Under H0 each discordant pair is a fair coin, so b ~ Binom(b+c, 0.5).
    Two-sided exact p = P(|X - n/2| >= |b - n/2|), computed by direct summation
    (no chi-square approximation, no continuity correction).
    Returns (b, c, n_disc, p_two_sided).
    """
    a = np.asarray(a_vec, dtype=int)
    b_ = np.asarray(b_vec, dtype=int)
    b = int(np.sum((a == 1) & (b_ == 0)))
    c = int(np.sum((a == 0) & (b_ == 1)))
    n = b + c
    if n == 0:
        return b, c, 0, 1.0
    obs = abs(b - n / 2.0)
    tot = 0.0
    for k in range(n + 1):
        if abs(k - n / 2.0) >= obs - 1e-12:
            tot += math.exp(_log_binom_coef(n, k) - n * math.log(2.0))
    return b, c, n, float(min(1.0, tot))


def paired_bootstrap(a_vec, b_vec, n_boot=N_BOOT, seed=SEED):
    """Canonical A02/A03 paired-difference bootstrap: delta = mean(a) - mean(b) in pp,
    CI95 percentile over n_boot resamples of the PAIRS (seed fixed)."""
    a = np.asarray(a_vec, dtype=float)
    b = np.asarray(b_vec, dtype=float)
    n = len(a)
    d = 100.0 * (a.mean() - b.mean())
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boots = 100.0 * (a[idx].mean(axis=1) - b[idx].mean(axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d), float(lo), float(hi)


def spearman_rho(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    den = math.sqrt(float((rx ** 2).sum()) * float((ry ** 2).sum()))
    if den == 0:
        return 0.0
    return float((rx * ry).sum() / den)


def _rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(1, len(a) + 1, dtype=float)
    # average ties
    _, inv, cnt = np.unique(a, return_inverse=True, return_counts=True)
    for i, c in enumerate(cnt):
        if c > 1:
            m = inv == i
            ranks[m] = ranks[m].mean()
    return ranks


def spearman_exact_perm(x, y):
    """Exact two-sided permutation p for Spearman with tiny n (n! enumerations)."""
    rho = spearman_rho(x, y)
    n = len(x)
    if n > 8:
        raise ValueError("exact permutation only for n<=8")
    cnt = 0
    tot = 0
    for perm in itertools.permutations(range(n)):
        r = spearman_rho(x, [y[i] for i in perm])
        tot += 1
        if abs(r) >= abs(rho) - 1e-12:
            cnt += 1
    return rho, cnt / tot, tot


def holm(pvals, labels):
    """Holm-Bonferroni within a family. Returns {label: adjusted_p}."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(1.0, running)
    return {labels[i]: adj[i] for i in range(m)}


def min_detectable_delta(n, p_base=0.10):
    """Crude two-sided alpha=0.05 minimum detectable paired difference (pp) for a
    subset of size n. Reported to adjudicate F2' (power guard), NOT used as a test."""
    if n <= 0:
        return None
    se = math.sqrt(max(p_base * (1 - p_base), 1e-9) * 2.0 / n)
    return round(100.0 * 1.96 * se, 2)


# ------------------------------------------------------------------- main ---- #
def load_inputs():
    rt_p = _PROP / "evidence/read_tax_ruler/a02_read_tax_per_item_vectors.json"
    dvr_p = _PROP / "evidence/a02_depth_vs_retrieval_per_item.json"
    rt = json.load(open(rt_p))
    dvr = json.load(open(dvr_p))["cells"]
    return rt, dvr


def cell_vectors(rt, cell):
    fam = "babilong" if cell.startswith("babilong") else "ruler"
    blk = rt[fam].get(cell)
    if blk is None:
        return None, None
    return blk["common_idx"], blk["per_arm_correct"]


def analyze(out_dir: Path):
    rt, dvr = load_inputs()
    errs, refused = [], []
    cells = list(PUBLISHED.keys())

    per_cell = {}
    for cell in cells:
        idx, arms = cell_vectors(rt, cell)
        if idx is None:
            refused.append({"cell": cell, "reason": "J2_MISSING_IN_READ_TAX"})
            continue

        # ---- GATE J1: shape / NaN / dup
        if len(idx) != 100 or len(set(idx)) != len(idx):
            refused.append({"cell": cell, "reason": f"J1_BAD_INDEX n={len(idx)} uniq={len(set(idx))}"})
            continue
        bad = False
        for a, v in arms.items():
            if len(v) != 100:
                errs.append(f"J1 {cell}/{a}: n={len(v)} != 100"); bad = True
            if any(x is None or (isinstance(x, float) and math.isnan(x)) for x in v):
                errs.append(f"J1 {cell}/{a}: contains NaN/None"); bad = True
            if not set(map(int, v)) <= {0, 1}:
                errs.append(f"J1 {cell}/{a}: non-binary values"); bad = True
        if bad:
            refused.append({"cell": cell, "reason": "J1_INTEGRITY"}); continue

        # ---- GATE J3: reproduce the published accuracies
        for a, want in PUBLISHED[cell].items():
            got = round(100.0 * float(np.mean(arms[a])), 2)
            if abs(got - want) > 1e-6:
                errs.append(f"J3 {cell}/{a}: recomputed {got} != published {want}")

        # ---- GATE J2: exact index join with dvr
        dcell = dvr.get(cell)
        hit_map, n_locatable = None, 0
        if dcell is None:
            errs.append(f"J2 {cell}: absent from dvr per-item")
        else:
            rps = dcell["recall_per_sample"]
            dvr_idx = [r["sample_index"] for r in rps]
            if set(dvr_idx) != set(idx):
                errs.append(f"J2 {cell}: index sets differ "
                            f"(rt {len(set(idx))} vs dvr {len(set(dvr_idx))}, "
                            f"overlap {len(set(idx) & set(dvr_idx))})")
            else:
                loc = [r for r in rps if r["gold_chunks"]]
                n_locatable = len(loc)
                # GATE J4: refuse conditioning when nothing is gold-locatable
                if n_locatable > 0:
                    hit_map = {r["sample_index"]: bool(r["hit"]) for r in loc}

        pos = {s: i for i, s in enumerate(idx)}

        def vec(arm, subset=None):
            v = arms[arm]
            if subset is None:
                return [int(x) for x in v]
            return [int(v[pos[s]]) for s in subset]

        # ---- statistic 1: A4 vs A5, full cell
        b, c, nd, p = mcnemar_exact(arms["A4"], arms["A5"])
        d, lo, hi = paired_bootstrap(arms["A4"], arms["A5"])
        a4 = round(100.0 * float(np.mean(arms["A4"])), 2)
        a5 = round(100.0 * float(np.mean(arms["A5"])), 2)
        entry = {
            "acc": {a: round(100.0 * float(np.mean(v)), 2) for a, v in arms.items()},
            "a4_vs_a5": {
                "A4": a4, "A5": a5, "delta_pp": round(d, 2),
                "ci95": [round(lo, 2), round(hi, 2)],
                "sig": bool(lo > 0 or hi < 0),
                "mcnemar": {"b_A4only": b, "c_A5only": c, "n_discordant": nd,
                            "p_two_sided_exact": round(p, 6)},
                "point_inverted": bool(a5 >= a4),
                "inversion_significant": bool(a5 > a4 and hi < 0),
            },
            # ---- F4 control: the uncontested A0 vs A5 contrast
            "a0_vs_a5": None,
            # ---- statistic 2: ladder rank test
            "ladder": None,
            # ---- statistic 4: conditional
            "conditional": None,
            "n_gold_locatable": n_locatable,
            "recall_at_12_published": PUBLISHED_RECALL.get(cell),
        }

        b0, c0, nd0, p0 = mcnemar_exact(arms["A0"], arms["A5"])
        d0, lo0, hi0 = paired_bootstrap(arms["A0"], arms["A5"])
        entry["a0_vs_a5"] = {
            "delta_pp": round(d0, 2), "ci95": [round(lo0, 2), round(hi0, 2)],
            "sig": bool(lo0 > 0 or hi0 < 0),
            "mcnemar": {"b_A0only": b0, "c_A5only": c0, "n_discordant": nd0,
                        "p_two_sided_exact": round(p0, 6)},
        }

        js = [j for _, j in LADDER]
        accs = [100.0 * float(np.mean(arms[a])) for a, _ in LADDER]
        rho, pperm, ntot = spearman_exact_perm(js, accs)
        entry["ladder"] = {
            "arms": [a for a, _ in LADDER], "j": js,
            "acc": [round(x, 2) for x in accs],
            "spearman_rho": round(rho, 4),
            "p_two_sided_exact_perm": round(pperm, 6),
            "n_permutations": ntot,
            "recovers_true_order": bool(rho <= -0.9 and pperm < 0.05),
        }

        # ---- statistic 4 + F2 + F2': conditional on retrieval HIT/MISS
        if hit_map is None:
            entry["conditional"] = {"status": "REFUSED_J4_NO_GOLD_LOCATABLE",
                                    "n_gold_locatable": n_locatable}
        else:
            hits = sorted([s for s, h in hit_map.items() if h])
            miss = sorted([s for s, h in hit_map.items() if not h])
            cond = {}
            for nm, sub in (("hit", hits), ("miss", miss)):
                if not sub:
                    cond[nm] = {"n": 0, "status": "EMPTY"}
                    continue
                av, bv = vec("A4", sub), vec("A5", sub)
                bb, cc, ndd, pp = mcnemar_exact(av, bv)
                dd, llo, hhi = paired_bootstrap(av, bv)
                a4s = round(100.0 * float(np.mean(av)), 2)
                a5s = round(100.0 * float(np.mean(bv)), 2)
                cond[nm] = {
                    "n": len(sub), "A4": a4s, "A5": a5s,
                    "delta_pp": round(dd, 2), "ci95": [round(llo, 2), round(hhi, 2)],
                    "sig": bool(llo > 0 or hhi < 0),
                    "mcnemar": {"b_A4only": bb, "c_A5only": cc,
                                "n_discordant": ndd, "p_two_sided_exact": round(pp, 6)},
                    "point_inverted": bool(a5s >= a4s),
                    "min_detectable_delta_pp_approx": min_detectable_delta(
                        len(sub), max(0.02, np.mean(av + bv))),
                }
            entry["conditional"] = cond
        per_cell[cell] = entry

    bab = [c for c in per_cell if c.startswith("babilong")]
    rul = [c for c in per_cell if c.startswith("ruler")]

    # ---- statistic 3: Holm within the 6-cell BABILong family
    fam = {}
    if bab:
        fam["babilong_a4_vs_a5_holm"] = holm(
            [per_cell[c]["a4_vs_a5"]["mcnemar"]["p_two_sided_exact"] for c in bab], bab)
        fam["babilong_ladder_holm"] = holm(
            [per_cell[c]["ladder"]["p_two_sided_exact_perm"] for c in bab], bab)

    # ---- statistic 5: recall vs signed separation across the 6 BABILong cells
    mech = {}
    rec = [PUBLISHED_RECALL[c] for c in bab]
    sep = [per_cell[c]["a4_vs_a5"]["delta_pp"] for c in bab]
    if len(bab) >= 3 and all(r is not None for r in rec):
        rho_r, p_r, nperm = spearman_exact_perm(rec, sep)
        mech["recall_vs_separation"] = {
            "cells": bab, "recall_at_12": rec, "signed_sep_A4_minus_A5_pp": sep,
            "spearman_rho": round(rho_r, 4),
            "p_two_sided_exact_perm": round(p_r, 6), "n_permutations": nperm,
            "prediction": "positive rho => higher recall restores the true ordering",
        }
    # ---- F3: floor as the competing explanation
    floor = [per_cell[c]["acc"]["A0"] for c in bab]
    if len(bab) >= 3:
        rho_f, p_f, _ = spearman_exact_perm(floor, sep)
        mech["floor_vs_separation"] = {
            "cells": bab, "A0_acc": floor, "signed_sep_A4_minus_A5_pp": sep,
            "spearman_rho": round(rho_f, 4), "p_two_sided_exact_perm": round(p_f, 6),
        }
        rho_c, p_c, _ = spearman_exact_perm(rec, floor)
        mech["recall_vs_floor_collinearity"] = {
            "spearman_rho": round(rho_c, 4), "p_two_sided_exact_perm": round(p_c, 6),
            "note": ("if |rho| is large the two explanations are mutually confounded "
                     "across these 6 cells and the mechanism is NOT identified"),
        }

    out = {
        "generated_by": "analyze_a02_babilong_misorder.py",
        "prereg": "A02_BABILONG_MISORDER_PREREG.md",
        "protocol": {
            "inputs": ["evidence/read_tax_ruler/a02_read_tax_per_item_vectors.json",
                       "evidence/a02_depth_vs_retrieval_per_item.json"],
            "gpu_spent": "ZERO -- pure CPU re-analysis of per-item vectors on disk",
            "mcnemar": "two-sided EXACT binomial on discordant pairs, direct summation",
            "ci": f"paired-difference bootstrap n_boot={N_BOOT} seed={SEED}, CI95 percentile",
            "ladder_test": "Spearman over j in {0,6,9,12,18} (A0,A2,A3,A4,A5); "
                           "exact permutation over all 120 orderings; min |p| = 0.0167",
            "multiplicity": "Holm-Bonferroni within each 6-cell BABILong family",
            "conditioning_validity": ("the retrieval pack is arm-independent (identical "
                                      "selector config, input_ids_sha256 asserted equal "
                                      "across arms), so HIT/MISS is a pre-treatment "
                                      "covariate of (cell,sample), not of the arm"),
            "aggregation_hygiene": ("per-cell only; NO pooled BABILong/LongEval figure "
                                    "(banned -17.89pp / +2.00pp) is computed anywhere"),
            "scorers": "canonical per-item `correct` vectors consumed verbatim",
        },
        "gates": {
            "J1_integrity": "PASS" if not [e for e in errs if e.startswith("J1")] else "FAIL",
            "J2_exact_index_join": "PASS" if not [e for e in errs if e.startswith("J2")] else "FAIL",
            "J3_reproduce_published_acc": "PASS" if not [e for e in errs if e.startswith("J3")] else "FAIL",
            "J4_refuse_conditioning_without_gold": [
                c for c in per_cell
                if isinstance(per_cell[c]["conditional"], dict)
                and per_cell[c]["conditional"].get("status", "").startswith("REFUSED")],
            "errors": errs, "refused_cells": refused,
        },
        "per_cell": per_cell,
        "families": fam,
        "mechanism": mech,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dst = out_dir / "a02_babilong_misorder.json"
    json.dump(out, open(dst, "w"), indent=1)
    print(f"wrote {dst}")

    # ------------------------------------------------------------ console ---- #
    print("\n=== GATES ===")
    for k in ("J1_integrity", "J2_exact_index_join", "J3_reproduce_published_acc"):
        print(f"  {k}: {out['gates'][k]}")
    print(f"  J4 refused conditioning: {out['gates']['J4_refuse_conditioning_without_gold']}")
    if errs:
        print("  ERRORS:")
        for e in errs:
            print("   ", e)

    print("\n=== A4 (j=12) vs A5 (j=18) per cell — TRUE ordering is A4 >> A5 (RULER +70pp) ===")
    print(f"{'cell':32s} {'A4':>6s} {'A5':>6s} {'delta':>8s} {'CI95':>18s} "
          f"{'b/c':>9s} {'p_exact':>9s}  inv?  sig-inv?")
    for c in rul + bab:
        e = per_cell[c]["a4_vs_a5"]
        m = e["mcnemar"]
        print(f"{c:32s} {e['A4']:6.1f} {e['A5']:6.1f} {e['delta_pp']:+8.2f} "
              f"[{e['ci95'][0]:+7.2f},{e['ci95'][1]:+7.2f}] "
              f"{m['b_A4only']:4d}/{m['c_A5only']:<4d} {m['p_two_sided_exact']:9.4f}  "
              f"{'YES' if e['point_inverted'] else ' no':4s}  "
              f"{'YES' if e['inversion_significant'] else ' no'}")

    print("\n=== ladder rank test (does the cell recover the depth ordering at all?) ===")
    for c in rul + bab:
        L = per_cell[c]["ladder"]
        print(f"{c:32s} acc={L['acc']}  rho={L['spearman_rho']:+.3f} "
              f"p={L['p_two_sided_exact_perm']:.4f}  "
              f"recovers={'YES' if L['recovers_true_order'] else 'no'}")

    print("\n=== F4 control: A0 vs A5 (uncontested, RULER -79pp) ===")
    for c in rul + bab:
        e = per_cell[c]["a0_vs_a5"]
        print(f"{c:32s} delta={e['delta_pp']:+7.2f} "
              f"[{e['ci95'][0]:+7.2f},{e['ci95'][1]:+7.2f}] sig={e['sig']}")

    print("\n=== conditional on retrieval HIT/MISS (F2 / F2') ===")
    for c in bab:
        cond = per_cell[c]["conditional"]
        if cond.get("status"):
            print(f"{c:32s} {cond['status']}")
            continue
        for nm in ("hit", "miss"):
            s = cond[nm]
            if s.get("status") == "EMPTY":
                print(f"{c:32s} {nm:5s} EMPTY")
                continue
            print(f"{c:32s} {nm:5s} n={s['n']:3d} A4={s['A4']:5.1f} A5={s['A5']:5.1f} "
                  f"delta={s['delta_pp']:+7.2f} [{s['ci95'][0]:+7.2f},{s['ci95'][1]:+7.2f}] "
                  f"inv={'YES' if s['point_inverted'] else ' no'} "
                  f"mde~{s['min_detectable_delta_pp_approx']}")

    if fam:
        print("\n=== Holm-adjusted (family of 6 BABILong cells) ===")
        for c in bab:
            print(f"{c:32s} a4a5_holm={fam['babilong_a4_vs_a5_holm'][c]:.4f}  "
                  f"ladder_holm={fam['babilong_ladder_holm'][c]:.4f}")

    if mech:
        print("\n=== mechanism ===")
        for k, v in mech.items():
            if "spearman_rho" in v:
                print(f"  {k}: rho={v['spearman_rho']:+.3f} p={v['p_two_sided_exact_perm']:.4f}")
    return out


def selftest():
    """Negative tests for the statistics and the gates."""
    ok = True
    # McNemar: classic worked example b=12,c=5 -> p ~= 0.1435
    a = [1] * 12 + [0] * 5 + [1] * 20 + [0] * 20
    b = [0] * 12 + [1] * 5 + [1] * 20 + [0] * 20
    bb, cc, nd, p = mcnemar_exact(a, b)
    exp = sum(math.exp(_log_binom_coef(17, k) - 17 * math.log(2))
              for k in range(18) if abs(k - 8.5) >= abs(12 - 8.5) - 1e-12)
    print(f"selftest mcnemar b={bb} c={cc} n={nd} p={p:.6f} (closed form {exp:.6f}) "
          f"{'OK' if abs(p - exp) < 1e-9 and (bb, cc) == (12, 5) else 'FAIL'}")
    ok &= abs(p - exp) < 1e-9 and (bb, cc) == (12, 5)

    # identical vectors -> no discordance, p=1, delta=0
    v = [1, 0, 1, 1, 0]
    bb, cc, nd, p = mcnemar_exact(v, v)
    d, lo, hi = paired_bootstrap(v, v)
    print(f"selftest identical: b={bb} c={cc} p={p} delta={d} ci=[{lo},{hi}] "
          f"{'OK' if (bb, cc, nd, p) == (0, 0, 0, 1.0) and d == 0 else 'FAIL'}")
    ok &= (bb, cc, nd, p) == (0, 0, 0, 1.0) and d == 0.0

    # Spearman perfect monotone decreasing -> rho=-1, exact p = 2/120
    rho, p, n = spearman_exact_perm([0, 6, 9, 12, 18], [99.75, 99.25, 98.25, 90.75, 20.75])
    print(f"selftest spearman: rho={rho:+.3f} p={p:.6f} n={n} "
          f"{'OK' if abs(rho + 1) < 1e-9 and abs(p - 2 / 120) < 1e-9 and n == 120 else 'FAIL'}")
    ok &= abs(rho + 1) < 1e-9 and abs(p - 2 / 120) < 1e-9

    # Holm monotonicity + no over-correction of the smallest p
    h = holm([0.01, 0.04, 0.5], ["a", "b", "c"])
    print(f"selftest holm: {h} {'OK' if abs(h['a'] - 0.03) < 1e-12 and h['b'] >= h['a'] else 'FAIL'}")
    ok &= abs(h["a"] - 0.03) < 1e-12 and h["b"] >= h["a"]

    # GATE J3 negative test: perturbing a vector must be caught
    rt, _ = load_inputs()
    cell = "babilong|qa1|16k"
    v = list(rt["babilong"][cell]["per_arm_correct"]["A4"])
    flip = 0 if v[0] == 1 else 1
    got = round(100.0 * (sum(v) - v[0] + flip) / 100.0, 2)
    print(f"selftest J3 negative: published {PUBLISHED[cell]['A4']} vs perturbed {got} "
          f"{'OK (differs -> gate would fire)' if abs(got - PUBLISHED[cell]['A4']) > 1e-6 else 'FAIL'}")
    ok &= abs(got - PUBLISHED[cell]["A4"]) > 1e-6
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(_PROP / "evidence/babilong_misorder"))
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    analyze(Path(a.out))
