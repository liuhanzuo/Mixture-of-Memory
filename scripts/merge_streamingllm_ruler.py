#!/usr/bin/env python
"""Merge sharded StreamingLLM RULER CSVs -> per-(task,length) string_match_all
recall cell scores.

The eval driver writes a per-sample ``recall`` column (RULER string_match_all:
fraction of reference strings present as a case-insensitive substring of the
output) into ``<task>_<length>_shard{g}of8.csv``. Shard sample sets partition
the full sample set, so the cell score == mean of the recall column over all
shard rows, ×100 — identical口径 to eval_ruler_mem_space / eval_ruler_qcmem.

Usage:
    python scripts/merge_streamingllm_ruler.py <results_dir> \
        --lengths 8k 16k 64k 128k --tasks niah_single_2
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys


def merge_cell(results_dir: str, task: str, length: str):
    pat = os.path.join(results_dir, f"{task}_{length}_shard*of*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        # fall back to a single un-sharded CSV
        single = os.path.join(results_dir, f"{task}_{length}.csv")
        files = [single] if os.path.exists(single) else []
    recall_sum = 0.0
    n = 0
    peak = 0.0
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                r = row.get("recall")
                if r is None or r == "":
                    continue
                recall_sum += float(r)
                n += 1
    # peak mem: read from any shard's json
    for g in range(64):
        jf = os.path.join(results_dir, f"{task}_{length}_shard{g}of8.json")
        if os.path.exists(jf):
            import json
            try:
                pk = json.load(open(jf))["summary"].get("peak_mem_gb", 0.0)
                peak = max(peak, float(pk or 0.0))
            except Exception:
                pass
    score = (recall_sum / n * 100.0) if n else 0.0
    return score, n, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--tasks", nargs="+", default=["niah_single_2"])
    ap.add_argument("--lengths", nargs="+",
                    default=["8k", "16k", "64k", "128k"])
    args = ap.parse_args()

    print(f"results_dir={args.results_dir}")
    hdr = "  ".join(f"{ln:>8}" for ln in args.lengths)
    print(f"{'task':>16}  {hdr}")
    for task in args.tasks:
        cells = []
        for ln in args.lengths:
            score, n, peak = merge_cell(args.results_dir, task, ln)
            cells.append(f"{score:5.1f}(n{n},{peak:.1f}G)")
        print(f"{task:>16}  " + "  ".join(f"{c:>8}" for c in cells))


if __name__ == "__main__":
    main()
