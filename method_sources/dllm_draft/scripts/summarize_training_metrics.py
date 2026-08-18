#!/usr/bin/env python3
"""Summarize comparable distributed training telemetry JSONL files."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


METRICS = (
    "train/step_seconds",
    "train/examples_per_second",
    "train/nonpadding_tokens_per_second",
    "train/padded_tokens_per_second",
    "train/supervised_tokens_per_second",
    "train/padding_fraction",
    "train/maximum_sequence_length",
    "train/peak_allocated_gib",
    "train/peak_reserved_gib",
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summarize(
    label: str,
    path: Path,
    *,
    warmup_records: int,
) -> dict[str, object]:
    records = read_jsonl(path)
    steady = records[warmup_records:] or records
    result: dict[str, object] = {
        "label": label,
        "path": str(path),
        "records": len(records),
        "steady_records": len(steady),
    }
    for metric in METRICS:
        values = [
            float(record[metric])
            for record in steady
            if metric in record
        ]
        if values:
            result[metric] = {
                "mean": statistics.mean(values),
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=/path/to/training_metrics.jsonl",
    )
    parser.add_argument("--warmup-records", type=int, default=2)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = []
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=PATH")
        rows.append(
            summarize(
                label,
                Path(path),
                warmup_records=args.warmup_records,
            )
        )
    report = {"warmup_records": args.warmup_records, "runs": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
