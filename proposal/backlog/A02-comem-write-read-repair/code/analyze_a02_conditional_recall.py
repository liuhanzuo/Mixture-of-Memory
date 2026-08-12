#!/usr/bin/env python
"""A02 depth-vs-retrieval: accuracy CONDITIONED on retrieval hit/miss.

Internal-validity check for the main gate. If the retrieval attribution is real,
then on the cells where recall@12 is low, `j0_top12` (matched-pack text-RAG,
j=0, no LoRA) should be:
  * ~as good as c1_pack_all on the recall-HIT subset  (the pack contained the
    gold chunk, so restricting to it costs little), and
  * far worse than c1_pack_all on the recall-MISS subset (the gold chunk is not
    in the pack, so the answer is unreachable no matter how well it reads).
That pattern is what makes "retrieval caused it" a causal claim rather than a
correlation with context length.

Symmetrically, if the DEPTH attribution is real, `j12_frozen` should be bad even
on the recall-HIT subset -- the gold chunk was in the pack and the mid-depth
read still could not use it.

Consumes ONLY the artefacts the main analyzer already wrote (no GPU, no re-eval):
  evidence/a02_depth_vs_retrieval_per_item.json
Reuses the same bootstrap (n_boot=5000, seed=42) and Wilson CI.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

N_BOOT, SEED = 5000, 42
EV = Path(__file__).resolve().parent.parent / "evidence"


def wilson(k, n, z=1.96):
    if n == 0:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [round(100 * p, 2), round(100 * max(0, c - h), 2),
            round(100 * min(1, c + h), 2)]


def boot(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    if len(d) == 0:
        return None
    rng = np.random.default_rng(SEED)
    n = len(d)
    bs = np.empty(N_BOOT)
    for i in range(N_BOOT):
        bs[i] = d[rng.integers(0, n, n)].mean()
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return {"diff_pt": round(100 * float(d.mean()), 2),
            "ci": [round(100 * float(lo), 2), round(100 * float(hi), 2)],
            "n": n, "sig": "SIG" if (lo > 0 or hi < 0) else "ns"}


def main():
    src = json.load(open(EV / "a02_depth_vs_retrieval_per_item.json"))
    out = {"protocol": {
        "ci": f"paired bootstrap n_boot={N_BOOT} seed={SEED}; Wilson for proportions",
        "definition": "recall HIT iff every gold-support chunk in the top-12 pack; "
                      "decided independently of answer correctness",
        "reads": "evidence/a02_depth_vs_retrieval_per_item.json (no re-eval)",
    }, "cells": {}}

    rows = []
    for cell, d in src["cells"].items():
        rec = d.get("recall_per_sample")
        if not rec:
            out["cells"][cell] = {"error": "no recall per-sample (gold not locatable)"}
            continue
        idx = d["paired_index"]
        pos = {s: i for i, s in enumerate(idx)}
        arms = d["per_arm_correct"]
        hit_i, miss_i = [], []
        for r in rec:
            if not r["gold_chunks"]:
                continue                      # gold not locatable -> excluded
            i = pos.get(r["sample_index"])
            if i is None:
                continue
            (hit_i if r["hit"] else miss_i).append(i)

        blk = {"n_hit": len(hit_i), "n_miss": len(miss_i)}
        for name, sub in (("hit", hit_i), ("miss", miss_i)):
            if not sub:
                blk[name] = None
                continue
            acc = {a: [arms[a][i] for i in sub] for a in arms}
            blk[name] = {
                "acc_pct": {a: wilson(sum(v), len(v)) for a, v in acc.items()},
                "retrieval_step_c1_to_j0": boot(acc["c1_pack_all"], acc["j0_top12"]),
                "depth_step_j0_to_j12frozen": boot(acc["j0_top12"], acc["j12_frozen"]),
                "lora_step_j12frozen_to_comem": boot(acc["j12_frozen"], acc["c2_comem"]),
            }
        out["cells"][cell] = blk
        if blk.get("hit") and blk.get("miss"):
            rows.append((cell, blk))

    (EV / "a02_depth_vs_retrieval_conditional.json").write_text(json.dumps(out, indent=1))

    print("\n" + "=" * 100)
    print("Accuracy CONDITIONED on retrieval hit/miss  (acc%, [lo,hi] Wilson)")
    print("=" * 100)
    h = (f"{'cell':30s} {'subset':6s} {'n':>4s} {'c1':>6s} {'j0t12':>6s} "
         f"{'j12fz':>6s} {'comem':>6s} {'RETR step':>18s}")
    print(h); print("-" * len(h))
    for cell, blk in rows:
        for sub in ("hit", "miss"):
            b = blk[sub]
            if not b:
                continue
            a = b["acc_pct"]
            r = b["retrieval_step_c1_to_j0"]
            print(f"{cell:30s} {sub:6s} {blk['n_'+sub]:4d} "
                  f"{a['c1_pack_all'][0]:6.1f} {a['j0_top12'][0]:6.1f} "
                  f"{a['j12_frozen'][0]:6.1f} {a['c2_comem'][0]:6.1f} "
                  f"{r['diff_pt']:+8.1f} {r['sig']:>4s}")
    print("\nReading: if RETRIEVAL is the cause, the RETR step is ~0 on HIT rows and")
    print("strongly negative on MISS rows. If DEPTH is the cause, j12fz is low even")
    print("on HIT rows (gold was in the pack and the mid-depth read still failed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
