#!/usr/bin/env python3
"""Merge deterministic rank shards and verify task coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--expected", type=int, required=True)
    args = parser.parse_args()

    directory = Path(args.input_dir)
    solutions = [
        row
        for path in sorted(directory.glob("solutions.rank*.jsonl"))
        for row in read(path)
    ]
    metrics = [
        row
        for path in sorted(directory.glob("metrics.rank*.jsonl"))
        for row in read(path)
    ]
    solutions.sort(key=lambda row: row["task_id"])
    metrics.sort(key=lambda row: row["task_id"])
    if len(solutions) != args.expected or len(metrics) != args.expected:
        raise SystemExit(
            f"coverage mismatch solutions={len(solutions)} "
            f"metrics={len(metrics)} expected={args.expected}"
        )
    if len({row["task_id"] for row in solutions}) != args.expected:
        raise SystemExit("duplicate or missing solution task IDs")
    with Path(args.solutions).open("w", encoding="utf-8") as handle:
        for row in solutions:
            handle.write(json.dumps(row) + "\n")
    with Path(args.metrics).open("w", encoding="utf-8") as handle:
        for row in metrics:
            handle.write(json.dumps(row) + "\n")
    failures = [row for row in metrics if row["error"] is not None]
    print(
        json.dumps(
            {
                "solutions": len(solutions),
                "metrics": len(metrics),
                "failures": len(failures),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

