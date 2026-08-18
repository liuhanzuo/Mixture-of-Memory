#!/usr/bin/env python3
"""Require full HumanEval semantic preservation before Scaffold evaluation."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--minimum-plus", type=float, default=0.45)
    args = parser.parse_args()

    evaluation = json.loads(
        (args.run / "eval_results.json").read_text(encoding="utf-8")
    )
    solutions = read_jsonl(args.run / "solutions.jsonl")
    metrics = read_jsonl(args.run / "metrics.jsonl")
    plus = float(evaluation["pass_at_k"]["plus"]["pass@1"])
    base = float(evaluation["pass_at_k"]["base"]["pass@1"])
    parseable = 0
    for row in solutions:
        try:
            ast.parse(str(row.get("solution", "")))
            parseable += 1
        except SyntaxError:
            pass
    errors = sum(bool(row.get("error")) for row in metrics)
    report = {
        "tasks": len(solutions),
        "base_pass1": base,
        "plus_pass1": plus,
        "parse_rate": parseable / len(solutions),
        "generation_errors": errors,
        "minimum_plus": args.minimum_plus,
        "passed": plus >= args.minimum_plus and errors == 0,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["passed"]:
        raise SystemExit(
            f"full vanilla HE+ {plus:.6f} failed preservation gate "
            f"{args.minimum_plus:.6f}"
        )


if __name__ == "__main__":
    main()
