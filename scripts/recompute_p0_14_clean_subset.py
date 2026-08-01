#!/usr/bin/env python
"""Paper A P0.14 — clean-subset re-scoring of InfiniteBench predictions (CPU only).

Given EXISTING per-example prediction files from ``scripts/eval_qcmem_infbench.py``
(the ``{task}_shard*of*.jsonl`` records with fields ``index``/``id``, ``pred`` or
LL ``correct``, and ``answers``) and the CLEAN-subset id list produced by
``scripts/audit_p0_14_contamination.py`` (``clean_subset_ids.json``), recompute
the official InfiniteBench metric on ONLY the clean examples — NO model run.

This is the P0.14 remediation step: if the InfiniteBench quality table is to be
kept, its numbers must be recomputed on documents that are NOT in PG-19 training.

IMPORTANT (why this script may be a NO-OP on the wzc1 node)
-----------------------------------------------------------
The InfiniteBench predictions were produced on a remote GPU node (diskB zwfy6
``.73``) and are NOT present on this wzc1 repo, and that node is off-limits for
this audit. So on this node there is nothing to re-score and the correct P0.14
action is the recommendation in the audit README (withdraw / relabel the
quality table). This script exists so that, IF the raw prediction dirs are ever
copied over, the clean-subset numbers can be produced in one CPU command with the
exact same official scorers, no fabrication.

USAGE
-----
  python scripts/recompute_p0_14_clean_subset.py \
      --clean_ids bench_results/p0_14_contamination/clean_subset_ids.json \
      --qcmem_qa_dir     infbench_results/qcmem_8b_j12_lora \
      --qcmem_choice_dir infbench_results/qcmem_8b_j12_lora_llmc \
      --dense_qa_dir     infbench_results/kvdirect_8b \
      --dense_choice_dir infbench_results/kvdirect_8b_llmc \
      --out bench_results/p0_14_contamination/clean_subset_recomputed.json

Any --*_dir may be omitted; the script only re-scores arms whose prediction dir
is present, and prints the full vs clean-subset number for each.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse the EXACT official InfiniteBench scorers used to produce the paper numbers.
import scripts.eval_qcmem_infbench as ib  # noqa: E402


def load_preds(pred_dir: str, task: str):
    """Merge + dedup-by-index the {task}_shard*.jsonl (same logic as run_scoring)."""
    if not pred_dir or not os.path.isdir(pred_dir):
        return None
    files = sorted(glob.glob(os.path.join(pred_dir, f"{task}_*.jsonl")))
    if not files:
        return None
    preds, seen = {}, set()
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                idx = item.get("index", item.get("id"))
                if idx is None or idx in seen:
                    continue
                seen.add(idx)
                preds[int(idx)] = item
    return preds


def score_subset(preds: dict, task: str, keep_ids):
    """Return (score, n) over keep_ids only. Handles both generate (pred/F1,EM)
    and LL-MC (correct) record modes exactly as run_scoring does."""
    keep = set(int(i) for i in keep_ids)
    rows = [preds[i] for i in preds if i in keep]
    if not rows:
        return None, 0
    ll = [r for r in rows if r.get("mode") == "ll"]
    if ll:
        acc = 100.0 * sum(int(r.get("correct", 0)) for r in ll) / len(ll)
        return acc, len(ll)
    scores = [ib.score_one(task, r.get("pred", ""), r.get("answers", []))
              for r in rows]
    return 100.0 * sum(scores) / len(scores), len(scores)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clean_ids",
                    default="bench_results/p0_14_contamination/clean_subset_ids.json")
    ap.add_argument("--qcmem_qa_dir", default="")
    ap.add_argument("--qcmem_choice_dir", default="")
    ap.add_argument("--dense_qa_dir", default="")
    ap.add_argument("--dense_choice_dir", default="")
    ap.add_argument("--out",
                    default="bench_results/p0_14_contamination/clean_subset_recomputed.json")
    args = ap.parse_args()

    clean = json.load(open(args.clean_ids))["clean_subset_ids"]
    qa_clean = clean.get("longbook_qa_eng", [])
    choice_clean = clean.get("longbook_choice_eng", [])

    arms = [
        ("CoMem+LoRA", "longbook_qa_eng", args.qcmem_qa_dir, qa_clean),
        ("CoMem+LoRA", "longbook_choice_eng", args.qcmem_choice_dir, choice_clean),
        ("Dense(KV-Direct)", "longbook_qa_eng", args.dense_qa_dir, qa_clean),
        ("Dense(KV-Direct)", "longbook_choice_eng", args.dense_choice_dir, choice_clean),
    ]

    out = {"clean_ids_source": os.path.abspath(args.clean_ids),
           "clean_subset_size": {"longbook_qa_eng": len(qa_clean),
                                 "longbook_choice_eng": len(choice_clean)},
           "results": [], "found_any_predictions": False}

    for arm, task, d, keep in arms:
        preds = load_preds(d, task)
        if preds is None:
            out["results"].append({"arm": arm, "task": task, "pred_dir": d or None,
                                    "status": "NO_PREDICTIONS_ON_THIS_NODE"})
            print(f"[recompute] {arm} / {task}: no predictions at {d!r} -> skip")
            continue
        out["found_any_predictions"] = True
        full_score, full_n = score_subset(preds, task,
                                           list(preds.keys()))
        clean_score, clean_n = score_subset(preds, task, keep)
        row = {"arm": arm, "task": task, "pred_dir": d,
               "full_score": None if full_score is None else round(full_score, 2),
               "full_n": full_n,
               "clean_score": None if clean_score is None else round(clean_score, 2),
               "clean_n": clean_n}
        out["results"].append(row)
        print(f"[recompute] {arm} / {task}: full={row['full_score']} "
              f"(n={full_n}) -> CLEAN={row['clean_score']} (n={clean_n})")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[recompute] wrote {args.out}")
    if not out["found_any_predictions"]:
        print("[recompute] NOTE: no prediction dirs present on this node. Per "
              "P0.14, recommend WITHDRAW/RELABEL the InfiniteBench quality table "
              "(raw predictions are on the off-limits GPU node).")


if __name__ == "__main__":
    main()
