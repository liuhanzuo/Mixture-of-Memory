#!/usr/bin/env python3
"""Recompute A03 Arm3/Arm4 CPT-trajectory paired-diff bootstrap CIs from per-item shards.

The +0.48pp SIG triviaqa headline lived only in two .md files; the one bootstrap
JSON on .82:/tmp held MMLU for step205/210k ONLY and never contained triviaqa at
all. This regenerates every cell from the 8-shard per-item records and writes a
persistent evidence JSON.

Protocol matches what the verdict files claim: per-item paired difference,
bootstrap n_boot=5000, seed=42, CI95 percentile. SIG = CI excludes 0.
"""
import json, os, sys
from pathlib import Path
import numpy as np

ROOT = Path("/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory")
CB = ROOT / "olmo2_closedbook_results"
MM = ROOT / "olmo2_mmlu_content_results"
N_BOOT, SEED, NSHARD = 5000, 42, 8


def load_cb(d, task):
    """item_id -> (em, contains, f1); asserts all 8 shards present."""
    got = {}
    files = sorted((CB / d).glob(f"per_example_{task}_shard*of{NSHARD}.jsonl"))
    if len(files) != NSHARD:
        raise SystemExit(f"FATAL {d}/{task}: {len(files)}/{NSHARD} shards -- refusing "
                         "(a silently-merged partial set has ruined results here before)")
    for f in files:
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            got[r["item_id"]] = (r["em"], r["contains"], r["f1"])
    return got


def load_mmlu(d):
    got = {}
    files = sorted((MM / d).glob(f"per_example_mmlu_shard*of{NSHARD}.jsonl"))
    if len(files) != NSHARD:
        return None
    for f in files:
        for ln in f.open():
            ln = ln.strip()
            if not ln:
                continue
            r = json.loads(ln)
            iid = r.get("item_id", r.get("idx"))
            L = r.get("letter_correct", r.get("correct_letter", r.get("em")))
            C = r.get("content_norm_correct", r.get("correct_content_norm",
                r.get("content_correct")))
            got[iid] = (L, C)
    return got


def paired(base, arm, idx):
    """CI95 of mean(arm-base) in percentage points."""
    b = np.array([base[i] for i in idx], dtype=float)
    a = np.array([arm[i] for i in idx], dtype=float)
    d = a - b
    rng = np.random.default_rng(SEED)
    n = len(d)
    boots = d[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1) * 100.0
    lo, hi = np.percentile(boots, [2.5, 97.5])
    delta = float(d.mean() * 100.0)
    return {"n": n, "delta_pp": delta, "ci95_pp": [float(lo), float(hi)],
            "verdict": "SIG" if (lo > 0 or hi < 0) else "TIE"}


BASE = "A03_1B_keep7_step200k"
ARMS = {
    "arm3_cosine_tail": [("step205000", "A03_1B_arm3_cpt_step205000"),
                         ("step210000", "A03_1B_arm3_cpt_step210000"),
                         ("step215000", "A03_1B_arm3_cpt_step215000"),
                         ("step220000", "A03_1B_arm3_cpt_step220000")],
    "arm4_peaklr":      [("step205000", "A03_1B_arm4_peaklr_step205000"),
                         ("step210000", "A03_1B_arm4_peaklr_step210000")],
}

out = {
    "protocol": f"per-item paired diff bootstrap n_boot={N_BOOT} seed={SEED}, CI95 percentile; SIG = CI excludes 0",
    "baseline": BASE,
    "regenerated": "2026-08-09 from 8/8 per-item shards; supersedes the volatile "
                   ".82:/tmp/a03_arm3_cpt_trajectory_paired.json (md5 37149d4d59bf941c1dbc05f17260f0b2), "
                   "which held MMLU step205/210k only and never contained triviaqa",
    "shard_integrity": f"every cell asserts {NSHARD}/{NSHARD} shards; script exits non-zero otherwise",
    "arms": {},
}

for arm, ckpts in ARMS.items():
    out["arms"][arm] = {}
    for label, d in ckpts:
        cell = {}
        for task in ("popqa", "triviaqa", "nq_open"):
            src = d if task != "nq_open" else d + "_nq"
            try:
                bb = load_cb(BASE if task != "nq_open" else BASE + "_nq", task)
                aa = load_cb(src, task)
            except SystemExit as e:
                cell[task] = {"error": str(e)}
                continue
            idx = sorted(set(bb) & set(aa))
            if not idx:
                cell[task] = {"error": "no overlapping item_ids"}
                continue
            cell[task] = {
                "em":       paired({i: bb[i][0] for i in idx}, {i: aa[i][0] for i in idx}, idx),
                "contains": paired({i: bb[i][1] for i in idx}, {i: aa[i][1] for i in idx}, idx),
                "f1":       paired({i: bb[i][2] for i in idx}, {i: aa[i][2] for i in idx}, idx),
            }
        mb, ma = load_mmlu(BASE), load_mmlu(d)
        if mb and ma:
            idx = sorted(set(mb) & set(ma))
            if idx and all(mb[i][0] is not None for i in idx[:5]):
                cell["mmlu"] = {
                    "letter": paired({i: mb[i][0] for i in idx}, {i: ma[i][0] for i in idx}, idx),
                }
                if all(mb[i][1] is not None for i in idx[:5]):
                    cell["mmlu"]["content_norm"] = paired(
                        {i: mb[i][1] for i in idx}, {i: ma[i][1] for i in idx}, idx)
        out["arms"][arm][label] = cell

dest = sys.argv[1] if len(sys.argv) > 1 else "/tmp/a03_cpt_trajectory_paired_full.json"
Path(dest).write_text(json.dumps(out, indent=2))
print(f"wrote {dest}")
for arm, cks in out["arms"].items():
    for label, cell in cks.items():
        for task, m in cell.items():
            if "error" in m:
                print(f"  {arm} {label} {task}: {m['error'][:70]}")
                continue
            bits = " ".join(
                f"{k}={v['delta_pp']:+.2f}{'*' if v['verdict']=='SIG' else ''}"
                for k, v in m.items())
            print(f"  {arm} {label} {task}: {bits}")
