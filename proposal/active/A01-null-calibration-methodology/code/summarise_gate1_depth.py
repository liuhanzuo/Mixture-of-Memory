#!/usr/bin/env python3
"""Summarise an A01 gate-1 damaged depth curve.

Usage: python summarise_gate1_depth.py '<glob for arm dirs>'

Emits, per keep-depth: n, letter acc, residual vs the MMLU best-constant floor
(always-D = 0.268908, a property of the benchmark's gold distribution and
therefore identical for every model), the modal-prediction share (the
"has it become a constant emitter?" diagnostic), the exact-tie rate among the
four letter-option logits, and content_norm accuracy.
"""
import glob
import json
import os
import sys
from collections import Counter

FLOOR = 0.268908  # always-D on cais/mmlu test, n=14042


def main():
    pattern = sys.argv[1]
    rows = []
    for d in sorted(glob.glob(pattern)):
        name = os.path.basename(d)
        per = []
        for s in sorted(glob.glob(os.path.join(d, "per_example_mmlu_shard*of8.jsonl"))):
            with open(s) as f:
                per += [json.loads(l) for l in f]
        per = [r for r in per if not r.get("nan")]
        if not per:
            continue
        n = len(per)
        lp = [r["letter"]["pred_letter"] for r in per]
        la = sum(1 for r in per if r["letter"]["correct"]) / n
        ca = sum(1 for r in per if r["content_norm"]["correct"]) / n
        _, mn = Counter(lp).most_common(1)[0]
        tie = 0
        for r in per:
            v = sorted(r["letter"]["scores"].values(), reverse=True)
            if len(v) >= 2 and v[0] == v[1]:
                tie += 1
        try:
            k = int(name.split("_k")[-1])
        except ValueError:
            continue
        rows.append((k, n, la, (la - FLOOR) * 100, mn / n * 100, tie / n * 100, ca))

    rows.sort()
    hdr = ("keep", "n", "letter", "vs_floor", "modal%", "tie%", "content")
    print("{:>5s} {:>6s} {:>8s} {:>9s} {:>7s} {:>7s} {:>8s}".format(*hdr))
    for k, n, la, res, mo, ti, ca in rows:
        print(f"{k:5d} {n:6d} {la:8.4f} {res:+9.2f} {mo:6.1f}% {ti:6.2f}% {ca:8.4f}")


if __name__ == "__main__":
    main()
