#!/usr/bin/env python3
"""Build a deterministic non-benchmark held-out set for correction tuning."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import pyarrow.parquet as pq


COMPOUND = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)
HEADER_RE = re.compile(r"(?m)^def\s+(.+):\s*$")


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


def function_header(source: str) -> str:
    match = HEADER_RE.search(source)
    if match is None:
        raise ValueError("held-out row has no top-level function signature")
    return match.group(1).strip()


def prepare_row(row: dict[str, Any]) -> dict[str, Any]:
    code = str(row["code"])
    header = function_header(code)
    depth = maximum_compound_depth(code)
    prompt = (
        str(row["prompt"]).rstrip()
        + "\n\nUse exactly this required function signature:\n"
        + f"def {header}:\n"
    )
    return {
        "task_id": f"Heldout/{int(row['seq_id'])}",
        "seq_id": int(row["seq_id"]),
        "prompt": prompt,
        "function_header": header,
        "entry_point": str(row["entry_point"]),
        "testcase": [str(test) for test in row["testcase"]],
        "canonical_solution": code,
        "compound_depth": depth,
        "depth_group": depth_group(depth),
        "prompt_tokens": int(row["prompt_tokens"]),
        "response_tokens": int(row["response_tokens"]),
    }


def select_rows(
    rows: list[dict[str, Any]],
    *,
    quotas: dict[str, int],
    seed: int,
    validator: Callable[[dict[str, Any]], bool] | None = None,
) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = {
        group: [] for group in quotas
    }
    for row in rows:
        prepared = prepare_row(row)
        group = str(prepared["depth_group"])
        if group in groups:
            groups[group].append(prepared)
    selected: list[dict[str, Any]] = []
    for offset, (group, quota) in enumerate(quotas.items()):
        candidates = sorted(groups[group], key=lambda row: row["seq_id"])
        random.Random(seed + offset).shuffle(candidates)
        accepted = []
        for candidate in candidates:
            if validator is None or validator(candidate):
                accepted.append(candidate)
                if len(accepted) == quota:
                    break
        if len(accepted) < quota:
            raise ValueError(
                f"not enough validated {group} rows: "
                f"{len(accepted)} < {quota}"
            )
        selected.extend(accepted[:quota])
    return sorted(selected, key=lambda row: row["task_id"])


def load_evaluator():
    path = Path(__file__).with_name("evaluate_correction_calibration.py")
    spec = importlib.util.spec_from_file_location(
        "evaluate_correction_calibration",
        path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.evaluate_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--depth-0-1", type=int, default=12)
    parser.add_argument("--depth-2", type=int, default=12)
    parser.add_argument("--depth-3-plus", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument(
        "--exclude-task-file",
        type=Path,
        help="JSONL task file whose seq_id values must be excluded.",
    )
    args = parser.parse_args()

    rows = pq.read_table(args.input).to_pylist()
    excluded_seq_ids: set[int] = set()
    if args.exclude_task_file is not None:
        excluded_seq_ids = {
            int(json.loads(line)["seq_id"])
            for line in args.exclude_task_file.read_text(
                encoding="utf-8"
            ).splitlines()
            if line.strip()
        }
        rows = [
            row
            for row in rows
            if int(row["seq_id"]) not in excluded_seq_ids
        ]
    quotas = {
        "depth_0_1": args.depth_0_1,
        "depth_2": args.depth_2,
        "depth_3_plus": args.depth_3_plus,
    }
    evaluate_source = load_evaluator()
    rejected: list[dict[str, Any]] = []

    def oracle_valid(row: dict[str, Any]) -> bool:
        result = evaluate_source(
            str(row["canonical_solution"]),
            [str(test) for test in row["testcase"]],
            timeout_seconds=5.0,
            memory_mib=1024,
        )
        if result["status"] == "pass":
            return True
        rejected.append(
            {
                "task_id": row["task_id"],
                "status": result["status"],
                "detail": result.get("detail", ""),
            }
        )
        return False

    selected = select_rows(
        rows,
        quotas=quotas,
        seed=args.seed,
        validator=oracle_valid,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, sort_keys=True) + "\n"
        for row in selected
    )
    args.output.write_text(payload, encoding="utf-8")
    manifest = {
        "input": str(args.input.resolve()),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "output": str(args.output.resolve()),
        "output_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        "seed": args.seed,
        "quotas": quotas,
        "rows": len(selected),
        "depth_counts": dict(
            Counter(str(row["depth_group"]) for row in selected)
        ),
        "oracle_rejected": rejected,
        "excluded_seq_ids": sorted(excluded_seq_ids),
        "exclude_task_file": (
            str(args.exclude_task_file.resolve())
            if args.exclude_task_file is not None
            else None
        ),
        "task_ids": [row["task_id"] for row in selected],
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # Keep full oracle diagnostics in the artifact, but do not print embedded
    # tracebacks into a registered run log: the heartbeat correctly treats a
    # fresh "Traceback" signature as a possible fatal job error.
    console_manifest = {
        **manifest,
        "oracle_rejected": [
            {
                "task_id": row["task_id"],
                "status": row["status"],
            }
            for row in rejected
        ],
        "oracle_rejected_count": len(rejected),
    }
    print(json.dumps(console_manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
