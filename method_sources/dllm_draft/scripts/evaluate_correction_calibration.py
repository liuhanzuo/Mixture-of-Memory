#!/usr/bin/env python3
"""Execute held-out correction-calibration solutions under hard limits."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def limited_preexec(timeout_seconds: float, memory_mib: int):
    def configure() -> None:
        cpu = max(1, math.ceil(timeout_seconds))
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu + 1))
        memory = memory_mib * 2**20
        resource.setrlimit(resource.RLIMIT_AS, (memory, memory))
        resource.setrlimit(resource.RLIMIT_FSIZE, (16 * 2**20, 16 * 2**20))
        resource.setrlimit(resource.RLIMIT_NOFILE, (64, 64))
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
        if hasattr(resource, "RLIMIT_NPROC"):
            resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))

    return configure


def evaluate_source(
    solution: str,
    testcases: list[str],
    *,
    timeout_seconds: float,
    memory_mib: int,
) -> dict[str, Any]:
    if not solution.strip():
        return {"status": "empty_output", "elapsed_seconds": 0.0}
    try:
        ast.parse(solution)
    except SyntaxError as exc:
        return {
            "status": "syntax_error",
            "elapsed_seconds": 0.0,
            "detail": f"{exc.msg} at line {exc.lineno}",
        }
    source = (
        "import random as __scaffold_random\n"
        "__scaffold_random.seed(0)\n\n"
        + solution.rstrip()
        + "\n\n"
        + "\n".join(testcases)
        + "\n"
    )
    with tempfile.TemporaryDirectory(prefix="scaffold-calibration-") as directory:
        root = Path(directory)
        program = root / "candidate.py"
        stdout_path = root / "stdout.txt"
        stderr_path = root / "stderr.txt"
        program.write_text(source, encoding="utf-8")
        started = time.perf_counter()
        try:
            with stdout_path.open("wb") as stdout, stderr_path.open("wb") as stderr:
                completed = subprocess.run(
                    [sys.executable, "-I", str(program)],
                    cwd=root,
                    stdin=subprocess.DEVNULL,
                    stdout=stdout,
                    stderr=stderr,
                    env={
                        "PATH": os.environ.get("PATH", ""),
                        "PYTHONHASHSEED": "0",
                    },
                    timeout=timeout_seconds,
                    preexec_fn=limited_preexec(timeout_seconds, memory_mib),
                    check=False,
                )
        except subprocess.TimeoutExpired:
            return {
                "status": "timeout",
                "elapsed_seconds": time.perf_counter() - started,
            }
        elapsed = time.perf_counter() - started
        stderr_text = stderr_path.read_text(
            encoding="utf-8",
            errors="replace",
        )[-4000:]
        if completed.returncode == 0:
            status = "pass"
        elif "AssertionError" in stderr_text:
            status = "assertion_failure"
        else:
            status = "runtime_error"
        return {
            "status": status,
            "elapsed_seconds": elapsed,
            "returncode": completed.returncode,
            "detail": stderr_text,
        }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter(str(row["status"]) for row in rows)
    parseable = sum(
        row["status"] not in {"empty_output", "syntax_error", "generation_error"}
        for row in rows
    )
    passed = counts["pass"]
    nfes = [
        float(row["process"]["nfe"])
        for row in rows
        if row.get("process") is not None
    ]
    cumulative = [
        float(row["process"]["cumulative_model_tokens"])
        for row in rows
        if row.get("process") is not None
    ]
    backtracks = [
        float(row["process"].get("structural_backtracks", 0))
        for row in rows
        if row.get("process") is not None
    ]
    leaf_remasks = [
        float(row["process"].get("leaf_remasks", 0))
        for row in rows
        if row.get("process") is not None
    ]
    deferrals = [
        float(row["process"].get("structural_deferrals", 0))
        for row in rows
        if row.get("process") is not None
    ]
    correction_rounds = [
        float(row["process"].get("correction_rounds", 0))
        for row in rows
        if row.get("process") is not None
    ]
    by_depth: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_depth[str(row["depth_group"])][str(row["status"])] += 1
    return {
        "rows": len(rows),
        "counts": dict(sorted(counts.items())),
        "pass_rate": passed / len(rows) if rows else None,
        "parse_rate": parseable / len(rows) if rows else None,
        "mean_nfe": statistics.mean(nfes) if nfes else None,
        "median_nfe": statistics.median(nfes) if nfes else None,
        "mean_cumulative_model_tokens": (
            statistics.mean(cumulative) if cumulative else None
        ),
        "mean_structural_backtracks": (
            statistics.mean(backtracks) if backtracks else None
        ),
        "mean_leaf_remasks": (
            statistics.mean(leaf_remasks) if leaf_remasks else None
        ),
        "mean_structural_deferrals": (
            statistics.mean(deferrals) if deferrals else None
        ),
        "mean_correction_rounds": (
            statistics.mean(correction_rounds)
            if correction_rounds
            else None
        ),
        "by_depth": {
            group: {
                "rows": sum(group_counts.values()),
                "counts": dict(sorted(group_counts.items())),
                "pass_rate": (
                    group_counts["pass"] / sum(group_counts.values())
                ),
            }
            for group, group_counts in sorted(by_depth.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--memory-mib", type=int, default=1024)
    args = parser.parse_args()

    tasks = {row["task_id"]: row for row in read_jsonl(args.tasks)}
    solutions = {row["task_id"]: row for row in read_jsonl(args.solutions)}
    metrics = {row["task_id"]: row for row in read_jsonl(args.metrics)}
    if set(tasks) != set(solutions) or set(tasks) != set(metrics):
        raise SystemExit(
            "task coverage mismatch between tasks, solutions, and metrics"
        )
    results = []
    for task_id in sorted(tasks):
        task = tasks[task_id]
        solution = str(solutions[task_id].get("solution", ""))
        metric = metrics[task_id]
        if metric.get("error"):
            evaluation = {
                "status": "generation_error",
                "elapsed_seconds": 0.0,
                "detail": str(metric["error"]),
            }
        else:
            evaluation = evaluate_source(
                solution,
                [str(test) for test in task["testcase"]],
                timeout_seconds=args.timeout_seconds,
                memory_mib=args.memory_mib,
            )
        results.append(
            {
                "task_id": task_id,
                "depth_group": task["depth_group"],
                "compound_depth": task["compound_depth"],
                "status": evaluation["status"],
                "test_elapsed_seconds": evaluation["elapsed_seconds"],
                "detail": evaluation.get("detail", ""),
                "generation_error": metric.get("error"),
                "generation_elapsed_seconds": metric.get("elapsed_seconds"),
                "process": metric.get("process"),
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in results),
        encoding="utf-8",
    )
    report = summarize(results)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
