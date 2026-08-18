#!/usr/bin/env python3
"""Compare EvalPlus result directories with process-cost sidecars."""

from __future__ import annotations

import argparse
import ast
import json
import statistics
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(label: str, directory: Path) -> dict[str, object]:
    evaluation = json.loads(
        (directory / "eval_results.json").read_text(encoding="utf-8")
    )
    metrics = read_jsonl(directory / "metrics.jsonl")
    solutions = read_jsonl(directory / "solutions.jsonl")
    processes = [
        row["process"]
        for row in metrics
        if row.get("process") is not None
    ]
    parseable = 0
    nonempty = 0
    for row in solutions:
        solution = str(row.get("solution", ""))
        if not solution.strip():
            continue
        nonempty += 1
        try:
            ast.parse(solution)
            parseable += 1
        except SyntaxError:
            pass
    nfes = [
        float(process["nfe"])
        for process in processes
        if process.get("nfe") is not None
    ]
    return {
        "label": label,
        "rows": len(solutions),
        "base_pass1": evaluation["pass_at_k"]["base"]["pass@1"],
        "plus_pass1": evaluation["pass_at_k"]["plus"]["pass@1"],
        "generation_errors": sum(bool(row.get("error")) for row in metrics),
        "nonempty_outputs": nonempty,
        "parseable_outputs": parseable,
        "parse_rate": parseable / len(solutions),
        "mean_nfe": statistics.mean(nfes) if nfes else None,
        "median_nfe": statistics.median(nfes) if nfes else None,
        "mean_elapsed_seconds": statistics.mean(
            float(row["elapsed_seconds"]) for row in metrics
        ),
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
