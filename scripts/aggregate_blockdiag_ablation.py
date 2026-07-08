#!/usr/bin/env python
"""Aggregate the QCMem block-diagonal ablation shards into a comparison table.

Each ``eval_ruler_qcmem.py`` shard writes ``{task}_{length}_shard{i}of{N}.csv``
with a per-row ``recall`` column (RULER string_match_all). We concatenate the N
shards of every (arm, task, length) cell and report mean recall * 100, then print
the (i) standard vs (ii) block-diagonal comparison table.
"""
from __future__ import annotations

import argparse
import glob
import os
import re

import pandas as pd

_ARMS = {"standard": "qcmem_standard_j12", "blockdiag": "qcmem_blockdiag_j12"}
_CELL_RE = re.compile(r"^(?P<task>.+?)_(?P<length>\d+k)(?:_shard\d+of\d+)?\.csv$")


def _score_cell(csv_paths):
    frames = []
    for p in csv_paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        if "recall" in df.columns and len(df):
            frames.append(df[["recall"]])
    if not frames:
        return None, 0
    allrows = pd.concat(frames, ignore_index=True)
    allrows = allrows[pd.to_numeric(allrows["recall"], errors="coerce").notna()]
    n = len(allrows)
    if n == 0:
        return None, 0
    return round(float(allrows["recall"].astype(float).mean()) * 100.0, 2), n


def _collect(arm_dir):
    """{ (task,length): (score, n) } merging all shard CSVs in arm_dir."""
    cells = {}
    groups = {}
    for p in glob.glob(os.path.join(arm_dir, "*.csv")):
        m = _CELL_RE.match(os.path.basename(p))
        if not m:
            continue
        key = (m.group("task"), m.group("length"))
        groups.setdefault(key, []).append(p)
    for key, paths in groups.items():
        score, n = _score_cell(paths)
        cells[key] = (score, n)
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_folder", type=str,
                    default="ruler_results/qcmem_blockdiag_ablation")
    args = ap.parse_args()

    per_arm = {}
    for arm, subdir in _ARMS.items():
        d = os.path.join(args.results_folder, subdir)
        per_arm[arm] = _collect(d) if os.path.isdir(d) else {}

    keys = sorted(set(per_arm["standard"]) | set(per_arm["blockdiag"]))
    print("\n" + "=" * 78)
    print("QCMem ablation: (i) standard full-attn read  vs  (ii) block-diagonal read")
    print("  (same j=12, topk12, bm25, Qwen3-8B + distill LoRA; only ATTENTION differs)")
    print("=" * 78)
    hdr = f"{'task':>18} {'len':>5} | {'(i) std':>10} {'(ii) blkdiag':>13} {'delta(i-ii)':>12} {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    for task, length in keys:
        s_std, n_std = per_arm["standard"].get((task, length), (None, 0))
        s_bd, n_bd = per_arm["blockdiag"].get((task, length), (None, 0))
        d = (round(s_std - s_bd, 2)
             if (s_std is not None and s_bd is not None) else None)
        n = n_std or n_bd
        print(f"{task:>18} {length:>5} | "
              f"{('%.1f' % s_std) if s_std is not None else '-':>10} "
              f"{('%.1f' % s_bd) if s_bd is not None else '-':>13} "
              f"{('%+.1f' % d) if d is not None else '-':>12} {n:>5}")
    print("=" * 78)


if __name__ == "__main__":
    main()
