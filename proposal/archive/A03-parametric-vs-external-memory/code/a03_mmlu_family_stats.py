"""MMLU-family multiplicity + sign test on the newly recovered A03 cells.

Two-sided bootstrap p uses the SAME resampling protocol as the CI in
recompute_cpt_trajectory_paired.py (n_boot=5000, seed=42) so the p and the CI
cannot disagree: p = 2*min(frac(boot<=0), frac(boot>=0)), floored at 1/5000.
BH q=0.05 is applied over the 24-cell MMLU family (3 arms x 4 steps x 2
interfaces). Choice disclosed: this family is the MMLU axis alone, declared
AFTER seeing that 1/24 cells was nominally SIG -- so treat the BH result as a
robustness check on that one cell, not as a pre-registered test.
"""
import json, glob
from pathlib import Path
import numpy as np

MM = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_mmlu_content_results")
N_BOOT, SEED = 5000, 42
BASE = "A03_1B_keep7_step200k"
ARMS = {"arm3_cosine_tail": "A03_1B_arm3_cpt_step%d",
        "arm4_peaklr": "A03_1B_arm4_peaklr_step%d",
        "arm6_lowerband": "A03_1B_arm6_lowerband_step%d"}
STEPS = [205000, 210000, 215000, 220000]

def load(d):
    got = {}
    for f in sorted(MM.glob(f"{d}/per_example_mmlu_shard*of8.jsonl")):
        for ln in f.open():
            if not ln.strip():
                continue
            r = json.loads(ln)
            got[r["item_id"]] = (1.0 if r["letter"]["correct"] else 0.0,
                                 1.0 if r["content_norm"]["correct"] else 0.0)
    return got

b = load(BASE)
rows = []
for arm, pat in ARMS.items():
    for st in STEPS:
        a = load(pat % st)
        idx = sorted(set(b) & set(a))
        assert len(idx) == 14042, (arm, st, len(idx))
        for j, iface in enumerate(("letter", "content_norm")):
            d = np.array([a[i][j] - b[i][j] for i in idx])
            rng = np.random.default_rng(SEED)
            boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1) * 100.0
            lo, hi = np.percentile(boots, [2.5, 97.5])
            p = 2 * min((boots <= 0).mean(), (boots >= 0).mean())
            rows.append({"arm": arm, "step": st, "iface": iface,
                         "delta_pp": float(d.mean() * 100), "ci": [float(lo), float(hi)],
                         "p": max(float(p), 1.0 / N_BOOT),
                         "sig_ci": bool(lo > 0 or hi < 0)})

rows.sort(key=lambda r: r["p"])
m = len(rows)
for k, r in enumerate(rows, 1):
    r["bh_adj"] = min(1.0, r["p"] * m / k)
for k in range(m - 2, -1, -1):
    rows[k]["bh_adj"] = min(rows[k]["bh_adj"], rows[k + 1]["bh_adj"])

print(f"MMLU family m={m} (3 arms x 4 steps x 2 interfaces), BH q=0.05")
print(f"{'arm':16s} {'step':7s} {'iface':13s} {'delta_pp':>9s} {'boot_p':>8s} {'BH_adj':>8s} {'CI_SIG':>7s} {'BH_SIG':>7s}")
for r in rows:
    print(f"{r['arm']:16s} {r['step']:<7d} {r['iface']:13s} {r['delta_pp']:+9.4f} "
          f"{r['p']:8.4f} {r['bh_adj']:8.4f} {str(r['sig_ci']):>7s} {str(r['bh_adj']<0.05):>7s}")

cn = [r["delta_pp"] for r in rows if r["iface"] == "content_norm"]
lt = [r["delta_pp"] for r in rows if r["iface"] == "letter"]
from math import comb
def signtest(v):
    n = sum(1 for x in v if x != 0.0); neg = sum(1 for x in v if x < 0)
    k = min(neg, n - neg)
    p = 2 * sum(comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return n, neg, min(p, 1.0)
for nm, v in (("content_norm", cn), ("letter", lt)):
    n, neg, p = signtest(v)
    print(f"\nsign test {nm}: {neg}/{n} negative (nonzero cells), two-sided p={p:.5f}, "
          f"mean={np.mean(v):+.4f}pp, range=[{min(v):+.4f},{max(v):+.4f}]")
Path("/tmp/a03_mmlu_family_stats.json").write_text(json.dumps(
    {"family_size": m, "bh_q": 0.05, "n_boot": N_BOOT, "seed": SEED,
     "n_mmlu": 14042, "baseline": BASE,
     "family_declared": "post-hoc (after observing 1/24 nominally SIG); robustness check, not pre-registered",
     "cells": rows,
     "sign_test": {k: dict(zip(("n", "n_negative", "p_two_sided"), signtest(v)))
                   for k, v in (("content_norm", cn), ("letter", lt))}}, indent=2))
print("\nwrote /tmp/a03_mmlu_family_stats.json")
