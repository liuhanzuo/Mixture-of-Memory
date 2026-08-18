#!/usr/bin/env python3
"""Evaluate the preregistered HumanEval semantic-preservation gate."""

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


def summarize(directory: Path) -> dict[str, object]:
    evaluation = json.loads(
        (directory / "eval_results.json").read_text(encoding="utf-8")
    )
    metrics = read_jsonl(directory / "metrics.jsonl")
    solutions = read_jsonl(directory / "solutions.jsonl")
    parseable = 0
    for row in solutions:
        try:
            ast.parse(str(row.get("solution", "")))
            parseable += 1
        except SyntaxError:
            pass
    return {
        "tasks": len(solutions),
        "base_pass1": float(
            evaluation["pass_at_k"]["base"]["pass@1"]
        ),
        "plus_pass1": float(
            evaluation["pass_at_k"]["plus"]["pass@1"]
        ),
        "generation_failures": sum(
            bool(row.get("error")) for row in metrics
        ),
        "generation_failure_rate": (
            sum(bool(row.get("error")) for row in metrics) / len(metrics)
        ),
        "parseable": parseable,
        "parse_rate": parseable / len(solutions),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vanilla", type=Path, required=True)
    parser.add_argument("--scaffold", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    vanilla = summarize(args.vanilla)
    scaffold = summarize(args.scaffold)
    vanilla_plus = float(vanilla["plus_pass1"])
    scaffold_plus = float(scaffold["plus_pass1"])
    scaffold_failure = float(scaffold["generation_failure_rate"])
    report = {
        "protocol": {
            "benchmark": "HumanEval+",
            "vanilla_nfe": 512,
            "scaffold_runtime": "Medium",
            "scaffold_max_model_calls": 512,
        },
        "vanilla": vanilla,
        "scaffold": scaffold,
        "gates": {
            "vanilla_preservation_45": vanilla_plus >= 0.45,
            "vanilla_stop_below_40": vanilla_plus < 0.40,
            "scaffold_quality_30": scaffold_plus >= 0.30,
            "scaffold_failure_at_most_5": scaffold_failure <= 0.05,
        },
    }
    report["decision"] = (
        "continue_to_mbpp"
        if (
            report["gates"]["vanilla_preservation_45"]
            and report["gates"]["scaffold_quality_30"]
            and report["gates"]["scaffold_failure_at_most_5"]
        )
        else (
            "stop_and_reduce_structure"
            if report["gates"]["vanilla_stop_below_40"]
            else "hold_for_analysis"
        )
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
