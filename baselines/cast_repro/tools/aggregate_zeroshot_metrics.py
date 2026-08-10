#!/usr/bin/env python3
"""Aggregate lm-eval 0.4.8 results into SPEC.md S7 zeroshot_metrics.json.

Reads results_*.json from a run's lm_eval_out/ directory and writes
zeroshot_metrics.json with per-task acc/acc_norm/stderr/n_samples, task
versions, harness version, and the SPEC.md S7 7-task average.

The average is computed with the standard sparsity-literature convention
(Wanda/SparseGPT/AST/CAST papers):
  - acc_norm for hellaswag, arc_easy, arc_challenge, openbookqa
  - acc for piqa, winogrande, race
For full transparency we also report the acc-only and acc_norm-only averages.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

TASKS = ("hellaswag", "race", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa")
# Standard "AST-7 avg" convention: acc for tasks whose canonical metric is raw acc,
# acc_norm for the length-normalized-preferred tasks.
PRIMARY_METRIC = {
    "hellaswag":     "acc_norm",
    "arc_easy":      "acc_norm",
    "arc_challenge": "acc_norm",
    "openbookqa":    "acc_norm",
    "piqa":          "acc",
    "winogrande":    "acc",
    "race":          "acc",
}


def latest_results_json(lm_eval_out: Path) -> Path:
    candidates = sorted(lm_eval_out.rglob("results_*.json"))
    if not candidates:
        raise SystemExit(f"no results_*.json under {lm_eval_out}")
    return candidates[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-out", required=True,
                    help="directory containing lm-eval results (contains results_*.json)")
    ap.add_argument("--output", required=True,
                    help="where to write zeroshot_metrics.json")
    ap.add_argument("--model", required=True,
                    help="model identifier to record in the JSON")
    args = ap.parse_args()

    results_path = latest_results_json(Path(args.lm_eval_out))
    print(f"[aggregate] reading {results_path}", flush=True)
    with open(results_path) as f:
        blob = json.load(f)

    per_task = {}
    missing = []
    for t in TASKS:
        r = blob["results"].get(t)
        if r is None:
            missing.append(t)
            continue
        entry = {
            "primary_metric": PRIMARY_METRIC[t],
            "n_samples": blob.get("n-samples", {}).get(t, {}).get("effective")
                         or blob.get("n-samples", {}).get(t, {}).get("original"),
            "task_version": blob["versions"].get(t),
            "acc":       r.get("acc,none"),
            "acc_stderr":       r.get("acc_stderr,none"),
            "acc_norm":  r.get("acc_norm,none"),
            "acc_norm_stderr":  r.get("acc_norm_stderr,none"),
        }
        per_task[t] = entry
    if missing:
        raise SystemExit(f"missing task results: {missing}")

    def mean(vals):
        return sum(vals) / len(vals) if vals else None

    primary_vals = [per_task[t][PRIMARY_METRIC[t]] for t in TASKS]
    acc_vals = [per_task[t]["acc"] for t in TASKS if per_task[t]["acc"] is not None]
    acc_norm_vals = [per_task[t]["acc_norm"] for t in TASKS if per_task[t]["acc_norm"] is not None]

    zeroshot_avg = mean(primary_vals)
    zeroshot_avg_acc = mean(acc_vals)
    zeroshot_avg_accnorm = mean(acc_norm_vals) if len(acc_norm_vals) == len(TASKS) else None

    out = {
        "model": args.model,
        "harness_version": blob.get("git_hash") or "unknown",
        "lm_eval_pip_version": "0.4.8",
        "n_fewshot": {t: blob.get("n-shot", {}).get(t) for t in TASKS},
        "batch_size": blob.get("config", {}).get("batch_size"),
        "add_bos_token": False,
        "chat_template": None,
        "dtype": "bfloat16",
        "tasks": list(TASKS),
        "primary_metric_per_task": PRIMARY_METRIC,
        "per_task": per_task,
        "zeroshot_avg_primary": zeroshot_avg,
        "zeroshot_avg_acc": zeroshot_avg_acc,
        "zeroshot_avg_acc_norm": zeroshot_avg_accnorm,
        "source_results_json": str(results_path.name),
        "source_model_args": (blob.get("config", {})
                              .get("model_args") if isinstance(blob.get("config"), dict)
                              else None),
        "note": (
            "primary_metric follows the standard sparsity-literature convention: "
            "acc_norm for hellaswag/arc_easy/arc_challenge/openbookqa, "
            "acc for piqa/winogrande/race. zeroshot_avg_primary is the "
            "mean of the per-task primary metrics; the other two averages "
            "are provided for transparency."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[aggregate] wrote {args.output}")
    print(f"[aggregate] zeroshot_avg_primary = {zeroshot_avg*100:.2f}")
    print(f"[aggregate] zeroshot_avg_acc = {zeroshot_avg_acc*100:.2f}")
    if zeroshot_avg_accnorm is not None:
        print(f"[aggregate] zeroshot_avg_acc_norm = {zeroshot_avg_accnorm*100:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
