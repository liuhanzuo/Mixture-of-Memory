#!/usr/bin/env python3
"""Write full-solution oracle JSONL files to validate EvalPlus plumbing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    data = Path(args.data_dir)
    output = Path(args.output_dir)

    human = list(read_jsonl(data / "HumanEvalPlus-v0.1.10.jsonl"))
    mbpp = list(read_jsonl(data / "MbppPlus-v0.2.0.jsonl"))
    write(
        output / "humaneval_oracle.jsonl",
        (
            {
                "task_id": row["task_id"],
                "solution": row["prompt"] + row["canonical_solution"],
            }
            for row in human
        ),
    )
    write(
        output / "mbpp_oracle.jsonl",
        (
            {
                "task_id": row["task_id"],
                "solution": row["canonical_solution"],
            }
            for row in mbpp
        ),
    )
    print(
        json.dumps(
            {
                "humaneval": len(human),
                "mbpp": len(mbpp),
                "output_dir": str(output.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

