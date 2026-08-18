#!/usr/bin/env python3
"""Rebuild Dream/DreamOn solution JSONL from preserved raw model outputs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from generate_evalplus_dream import extract_python


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", required=True)
    parser.add_argument("--solutions", required=True)
    args = parser.parse_args()

    rows = [
        json.loads(line)
        for line in Path(args.metrics).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    output = Path(args.solutions)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    {
                        "task_id": row["task_id"],
                        "solution": extract_python(row.get("raw_output", "")),
                    }
                )
                + "\n"
            )
    print({"solutions": len(rows), "output": str(output)})


if __name__ == "__main__":
    main()
