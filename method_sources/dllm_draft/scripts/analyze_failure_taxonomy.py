#!/usr/bin/env python3
"""Classify generation, syntax, timeout, and semantic EvalPlus failures."""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def classify(
    *,
    solution: str,
    metric: dict[str, object],
    evaluation: dict[str, object],
) -> str:
    if metric.get("error"):
        return "generation_error"
    if not solution.strip():
        return "empty_output"
    try:
        ast.parse(solution)
    except SyntaxError:
        return "syntax_error"
    base = evaluation["base_status"]
    plus = evaluation["plus_status"]
    if base == "pass" and plus == "pass":
        return "plus_pass"
    if base == "pass" and plus == "timeout":
        return "plus_timeout"
    if base == "pass":
        return "plus_only_failure"
    if base == "timeout":
        return "base_timeout"
    return "base_semantic_failure"


def summarize(label: str, directory: Path) -> dict[str, object]:
    solutions = {
        row["task_id"]: str(row.get("solution", ""))
        for row in read_jsonl(directory / "solutions.jsonl")
    }
    metrics = {
        row["task_id"]: row
        for row in read_jsonl(directory / "metrics.jsonl")
    }
    evaluation = json.loads(
        (directory / "eval_results.json").read_text(encoding="utf-8")
    )["eval"]
    counts: Counter[str] = Counter()
    for task_id, rows in evaluation.items():
        counts[
            classify(
                solution=solutions[task_id],
                metric=metrics[task_id],
                evaluation=rows[0],
            )
        ] += 1
    total = len(evaluation)
    return {
        "label": label,
        "total": total,
        "counts": dict(counts),
        "rates": {
            key: value / total
            for key, value in sorted(counts.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=/path/to/eval/result/directory",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=PATH")
        rows.append(summarize(label, Path(path)))
    report = {"runs": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
