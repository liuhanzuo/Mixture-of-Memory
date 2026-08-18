#!/usr/bin/env python3
"""Compare EvalPlus pass rates by canonical compound-statement depth."""

from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path


COMPOUND = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def maximum_compound_depth(source: str) -> int:
    tree = ast.parse(source)
    maximum = 0

    def visit_body(body: list[ast.stmt], depth: int) -> None:
        nonlocal maximum
        maximum = max(maximum, depth)
        for node in body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                visit_body(node.body, depth)
            elif isinstance(node, COMPOUND):
                child_depth = depth + 1
                maximum = max(maximum, child_depth)
                visit_body(node.body, child_depth)
                visit_body(getattr(node, "orelse", []), child_depth)
                visit_body(getattr(node, "finalbody", []), child_depth)
                for handler in getattr(node, "handlers", []):
                    visit_body(handler.body, child_depth)
            elif hasattr(ast, "Match") and isinstance(node, ast.Match):
                child_depth = depth + 1
                maximum = max(maximum, child_depth)
                for case in node.cases:
                    visit_body(case.body, child_depth)

    visit_body(tree.body, 0)
    return maximum


def depth_group(depth: int) -> str:
    if depth <= 1:
        return "depth_0_1"
    if depth == 2:
        return "depth_2"
    return "depth_3_plus"


def base_pass(evaluation: dict[str, object]) -> bool:
    return evaluation["base_status"] == "pass"


def plus_pass(evaluation: dict[str, object]) -> bool:
    # EvalPlus pass@1 for the Plus suite requires passing both the original
    # tests and the additional tests. ``plus_status`` alone can be "pass" for
    # a candidate that already failed the base suite.
    return (
        evaluation["base_status"] == "pass"
        and evaluation["plus_status"] == "pass"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-file", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=/path/to/eval_results.json",
    )
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    problems = {
        row["task_id"]: row
        for row in read_jsonl(Path(args.dataset_file))
    }
    depths = {
        task_id: maximum_compound_depth(
            str(problem.get("prompt", ""))
            + str(problem["canonical_solution"])
        )
        for task_id, problem in problems.items()
    }
    report: dict[str, object] = {
        "dataset_file": args.dataset_file,
        "depth_counts": {},
        "runs": {},
    }
    counts: dict[str, int] = {}
    for depth in depths.values():
        group = depth_group(depth)
        counts[group] = counts.get(group, 0) + 1
    report["depth_counts"] = counts

    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=PATH")
        evaluation = json.loads(Path(path).read_text(encoding="utf-8"))["eval"]
        groups: dict[str, dict[str, int]] = {}
        for task_id, rows in evaluation.items():
            group = depth_group(depths[task_id])
            current = groups.setdefault(
                group,
                {"count": 0, "base_pass": 0, "plus_pass": 0},
            )
            current["count"] += 1
            current["base_pass"] += base_pass(rows[0])
            current["plus_pass"] += plus_pass(rows[0])
        report["runs"][label] = {
            group: {
                **values,
                "base_pass1": values["base_pass"] / values["count"],
                "plus_pass1": values["plus_pass"] / values["count"],
            }
            for group, values in sorted(groups.items())
        }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
