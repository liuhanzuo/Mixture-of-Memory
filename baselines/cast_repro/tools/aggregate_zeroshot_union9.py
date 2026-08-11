#!/usr/bin/env python3
"""Aggregate lm-eval 0.4.8 results into a Union-9 zeroshot metrics JSON.

Why a NEW tool instead of extending aggregate_zeroshot_metrics.py:
the existing tool is the recorded provenance of the three already-published
zeroshot_metrics.json files (dense_ref / cast_7500 / wanda). Editing it would
retroactively change what produced those cited files. This tool is additive.

Union-9 = CAST-7 (HellaSwag, RACE, PIQA, WinoGrande, ARC-e, ARC-c, OBQA)
        U AST-7  (BoolQ, RTE, HellaSwag, WinoGrande, ARC-e, ARC-c, OBQA)
        = 9 tasks; intersection is 5.

PROTOCOL (fixed by the project lead, do not "improve"):
  * primary metric map, IDENTICAL across every arm:
      acc_norm : hellaswag, arc_easy, arc_challenge, openbookqa
      acc      : piqa, winogrande, race, boolq, rte
    BoolQ and RTE are binary classification / entailment; option length is not a
    confound there, so acc_norm is meaningless and acc is the reported metric.
  * a parallel plain-acc slice is also emitted for every task, because both
    source papers' headline aggregates (CAST-7 55.91, AST-7 58.62/57.94) are
    plain acc. Subset means are therefore reported under BOTH conventions and
    must be labelled when cited.

Also emits, from the same run (free, just different slices of one set of
numbers): union9 / cast7 / ast7 means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

UNION9 = (
    "boolq", "rte", "hellaswag", "race", "piqa",
    "winogrande", "arc_easy", "arc_challenge", "openbookqa",
)
CAST7 = ("hellaswag", "race", "piqa", "winogrande", "arc_easy", "arc_challenge", "openbookqa")
AST7 = ("boolq", "rte", "hellaswag", "winogrande", "arc_easy", "arc_challenge", "openbookqa")

PRIMARY_METRIC = {
    "hellaswag":     "acc_norm",
    "arc_easy":      "acc_norm",
    "arc_challenge": "acc_norm",
    "openbookqa":    "acc_norm",
    "piqa":          "acc",
    "winogrande":    "acc",
    "race":          "acc",
    "boolq":         "acc",
    "rte":           "acc",
}
assert set(PRIMARY_METRIC) == set(UNION9), "metric map must cover exactly the union-9"


def latest_results_json(lm_eval_out: Path) -> Path:
    candidates = sorted(lm_eval_out.rglob("results_*.json"))
    if not candidates:
        raise SystemExit(f"no results_*.json under {lm_eval_out}")
    return candidates[-1]


def mean(vals):
    return sum(vals) / len(vals) if vals else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lm-eval-out", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tasks", default=",".join(UNION9),
                    help="comma list of tasks that MUST be present (default: union-9)")
    args = ap.parse_args()

    required = tuple(t.strip() for t in args.tasks.split(",") if t.strip())
    for t in required:
        if t not in PRIMARY_METRIC:
            raise SystemExit(f"task {t!r} has no entry in the fixed PRIMARY_METRIC map")

    results_path = latest_results_json(Path(args.lm_eval_out))
    print(f"[agg9] reading {results_path}", flush=True)
    blob = json.load(open(results_path))

    per_task = {}
    missing = []
    for t in required:
        r = blob["results"].get(t)
        if r is None:
            missing.append(t)
            continue
        per_task[t] = {
            "primary_metric": PRIMARY_METRIC[t],
            "n_samples": (blob.get("n-samples", {}).get(t, {}).get("effective")
                          or blob.get("n-samples", {}).get(t, {}).get("original")),
            "task_version": blob["versions"].get(t),
            "acc": r.get("acc,none"),
            "acc_stderr": r.get("acc_stderr,none"),
            "acc_norm": r.get("acc_norm,none"),
            "acc_norm_stderr": r.get("acc_norm_stderr,none"),
        }
    if missing:
        # Hard failure: a silently-dropped task corrupts every average it feeds.
        raise SystemExit(f"missing task results: {missing} -- this arm's row is INVALID")

    for t, e in per_task.items():
        if e[PRIMARY_METRIC[t]] is None:
            raise SystemExit(
                f"task {t} is missing its primary metric {PRIMARY_METRIC[t]!r}; row INVALID")

    def slice_means(tasks):
        """Mean over a named slice. A slice that is not fully present yields NO
        mean -- a partial average silently mislabelled as e.g. 'ast7' is exactly
        the class of error this table exists to eliminate."""
        present = [t for t in tasks if t in per_task]
        if len(present) != len(tasks):
            return {
                "tasks": list(tasks),
                "n_present": len(present),
                "n_required": len(tasks),
                "mean_primary": None,
                "mean_plain_acc": None,
                "incomplete": True,
                "missing": [t for t in tasks if t not in per_task],
            }
        prim = [per_task[t][PRIMARY_METRIC[t]] for t in tasks]
        accs = [per_task[t]["acc"] for t in tasks]
        return {
            "tasks": list(tasks),
            "n_tasks": len(tasks),
            "mean_primary": mean(prim),
            "mean_plain_acc": mean(accs) if all(a is not None for a in accs) else None,
        }

    out = {
        "model": args.model,
        "harness_version": blob.get("git_hash") or "unknown",
        "lm_eval_pip_version": "0.4.8",
        "n_fewshot": {t: blob.get("n-shot", {}).get(t) for t in required},
        "batch_size": blob.get("config", {}).get("batch_size"),
        "batch_sizes_detected": blob.get("config", {}).get("batch_sizes"),
        "add_bos_token": False,
        "chat_template": None,
        "dtype": "bfloat16",
        "tasks": list(required),
        "primary_metric_per_task": {t: PRIMARY_METRIC[t] for t in required},
        "per_task": per_task,
        "union9": slice_means(UNION9),
        "cast7": slice_means(CAST7),
        "ast7": slice_means(AST7),
        "source_results_json": str(results_path.name),
        "source_model_args": (blob.get("config", {}).get("model_args")
                              if isinstance(blob.get("config"), dict) else None),
        "note": (
            "Union-9 = CAST-7 union AST-7. primary_metric_per_task is fixed and "
            "identical across all arms (acc_norm for hellaswag/arc_easy/arc_challenge/"
            "openbookqa; acc for piqa/winogrande/race/boolq/rte). Both source papers' "
            "headline aggregates are PLAIN ACC, so cast7/ast7 comparisons against "
            "CAST 55.91 / AST 58.62-57.94 must use mean_plain_acc, not mean_primary."
        ),
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[agg9] wrote {args.output}")
    for k in ("union9", "cast7", "ast7"):
        s = out[k]
        if s.get("incomplete"):
            print(f"[agg9] {k}: INCOMPLETE ({s['n_present']}/{s['n_required']} tasks, "
                  f"missing {s['missing']}) -- no mean emitted")
        else:
            print(f"[agg9] {k}: n={s['n_tasks']} primary={s['mean_primary']*100:.4f} "
                  f"plain_acc={s['mean_plain_acc']*100:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
