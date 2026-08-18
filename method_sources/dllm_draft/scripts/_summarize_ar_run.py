#!/usr/bin/env python3
"""Summarize an AR EvalPlus run: pass@1 plus the two cost quantities.

Reads the merged ``metrics.jsonl`` produced by ``generate_evalplus_ar.py`` and
the ``eval_results.json`` produced by the official ``evalplus.evaluate``.
Emits mean/median/total for ``tokens_fed`` and ``attended_context_sum`` so the
AR row can be placed on the same cost axes as the diffusion rows.

See the module docstring of ``scripts/generate_evalplus_ar.py`` for the precise
definition of both cost quantities and their caveats.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "total": sum(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--eval-results", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    metrics = read_jsonl(Path(args.metrics))
    evaluation = json.loads(Path(args.eval_results).read_text(encoding="utf-8"))

    ok = [row for row in metrics if row.get("process")]
    errors = [row for row in metrics if row.get("error")]
    inconsistent = [
        row["task_id"]
        for row in ok
        if not row["process"]["cost"].get("consistent", False)
    ]

    report = {
        "dataset": args.dataset,
        "tasks": len(metrics),
        "errors": len(errors),
        "error_task_ids": [row["task_id"] for row in errors][:20],
        "parseable": sum(
            bool(row["process"]["final_parseable"]) for row in ok
        ),
        "pass_at_1": {
            "base": evaluation["pass_at_k"]["base"]["pass@1"],
            "plus": evaluation["pass_at_k"]["plus"]["pass@1"],
        },
        "cost": {
            "tokens_fed": stats(
                [row["process"]["cost"]["tokens_fed"] for row in ok]
            ),
            "attended_context_sum": stats(
                [row["process"]["cost"]["attended_context_sum"] for row in ok]
            ),
            "forward_passes": stats(
                [row["process"]["cost"]["forward_passes"] for row in ok]
            ),
            "input_tokens": stats(
                [row["process"]["input_tokens"] for row in ok]
            ),
            "generated_tokens": stats(
                [row["process"]["generated_tokens"] for row in ok]
            ),
            "measured_matches_analytic": len(inconsistent) == 0,
            "inconsistent_task_ids": inconsistent[:20],
        },
        "runtime": {
            "wall_clock_seconds": stats(
                [row["wall_clock_seconds"] for row in metrics]
            ),
            "peak_memory_gib": stats(
                [row["peak_memory_gib"] for row in metrics]
            ),
        },
        "hit_token_budget": sum(
            1 for row in ok if row["process"]["stop_string_hit"] is None
        ),
    }
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
