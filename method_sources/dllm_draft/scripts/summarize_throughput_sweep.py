#!/usr/bin/env python3
"""Summarize 8-GPU micro-batch probes and choose a safe fastest setting."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    if not path.exists():
        return records
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def median(records: list[dict[str, object]], key: str) -> float:
    return float(statistics.median(float(record[key]) for record in records))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gpu-memory-mib", type=float, default=97871.0)
    parser.add_argument("--minimum-headroom-gib", type=float, default=5.0)
    parser.add_argument("--warmup-records", type=int, default=1)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    status_path = run_dir / "status.tsv"
    candidates: list[dict[str, object]] = []
    with status_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            micro_batch = int(row["micro_batch_size_per_gpu"])
            metrics_path = run_dir / row["metrics_file"]
            all_records = read_jsonl(metrics_path)
            steady = all_records[args.warmup_records :]
            if not steady:
                steady = all_records
            candidate: dict[str, object] = {
                "micro_batch_size_per_gpu": micro_batch,
                "status": row["status"],
                "exit_code": int(row["exit_code"]),
                "log_file": row["log_file"],
                "metrics_file": row["metrics_file"],
                "profile_records": len(all_records),
                "steady_records": len(steady),
            }
            if row["status"] == "success" and steady:
                peak_reserved = max(
                    float(record["train/peak_reserved_gib"])
                    for record in all_records
                )
                candidate.update(
                    {
                        "median_step_seconds": median(
                            steady, "train/step_seconds"
                        ),
                        "median_examples_per_second": median(
                            steady, "train/examples_per_second"
                        ),
                        "median_nonpadding_tokens_per_second": median(
                            steady, "train/nonpadding_tokens_per_second"
                        ),
                        "median_supervised_tokens_per_second": median(
                            steady, "train/supervised_tokens_per_second"
                        ),
                        "peak_allocated_gib": max(
                            float(record["train/peak_allocated_gib"])
                            for record in all_records
                        ),
                        "peak_reserved_gib": peak_reserved,
                        "memory_headroom_gib": (
                            args.gpu_memory_mib / 1024 - peak_reserved
                        ),
                    }
                )
                if all(
                    "train/padding_fraction" in record
                    for record in steady
                ):
                    candidate["median_padding_fraction"] = median(
                        steady,
                        "train/padding_fraction",
                    )
                if all(
                    "train/maximum_sequence_length" in record
                    for record in steady
                ):
                    candidate["maximum_sequence_length"] = max(
                        float(record["train/maximum_sequence_length"])
                        for record in steady
                    )
            candidates.append(candidate)

    successful = [
        candidate
        for candidate in candidates
        if candidate["status"] == "success"
        and candidate["steady_records"]
    ]
    if not successful:
        raise SystemExit("no successful micro-batch candidate")
    safe = [
        candidate
        for candidate in successful
        if float(candidate["memory_headroom_gib"])
        >= args.minimum_headroom_gib
    ]
    pool = safe or successful
    recommended = max(
        pool,
        key=lambda candidate: (
            float(candidate["median_nonpadding_tokens_per_second"]),
            int(candidate["micro_batch_size_per_gpu"]),
        ),
    )
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "gpu_memory_mib": args.gpu_memory_mib,
        "minimum_headroom_gib": args.minimum_headroom_gib,
        "warmup_records_excluded": args.warmup_records,
        "selection_used_headroom_constraint": bool(safe),
        "recommended_micro_batch_size_per_gpu": recommended[
            "micro_batch_size_per_gpu"
        ],
        "recommended": recommended,
        "candidates": sorted(
            candidates,
            key=lambda candidate: int(
                candidate["micro_batch_size_per_gpu"]
            ),
            reverse=True,
        ),
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
