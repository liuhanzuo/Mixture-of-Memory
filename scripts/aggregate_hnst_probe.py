#!/usr/bin/env python
"""Aggregate hnst_retrieval_probe jsonl shards -> needle-recall table.

Reports, per (length, bucket in {early,mid,late}), the fraction of samples whose
needle chunk was SELECTED by each arm (tree / flat / b25), plus n. This is the
leak-immune decisive HNST metric: does reader-native tree beam-descent reach the
needle, especially EARLY needles that b25 FIFO evicts?
"""
import json, sys, glob
from collections import defaultdict

files = []
for a in sys.argv[1:]:
    files += glob.glob(a)
if not files:
    print("usage: aggregate_hnst_probe.py <jsonl glob...>"); sys.exit(1)

# (length,bucket) -> arm -> [hits, n]
agg = defaultdict(lambda: defaultdict(lambda: [0, 0]))
overall = defaultdict(lambda: defaultdict(lambda: [0, 0]))
for f in files:
    for line in open(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        key = (r["length"], r["bucket"])
        for arm in ("tree", "flat", "b25"):
            agg[key][arm][0] += r[arm]; agg[key][arm][1] += 1
            overall[r["length"]][arm][0] += r[arm]; overall[r["length"]][arm][1] += 1

def pct(h, n): return f"{100.0*h/n:5.1f}({h}/{n})" if n else "  -- "

lengths = sorted({k[0] for k in agg})
buckets = ["early", "mid", "late"]
print("=== HNST needle-recall (leak-immune, base Llama-3-8B q.k, no training) ===")
print("recall = P(true needle chunk in the arm's selected set)")
print(f"{'length':<6} {'bucket':<6} {'tree':>14} {'flat':>14} {'b25(FIFO)':>14}")
print("-"*60)
for L in lengths:
    for b in buckets:
        key = (L, b)
        if key not in agg: continue
        t = agg[key]["tree"]; f_ = agg[key]["flat"]; b25 = agg[key]["b25"]
        print(f"{L:<6} {b:<6} {pct(*t):>14} {pct(*f_):>14} {pct(*b25):>14}")
    o = overall[L]
    print(f"{L:<6} {'ALL':<6} {pct(*o['tree']):>14} {pct(*o['flat']):>14} {pct(*o['b25']):>14}")
    print("-"*60)
