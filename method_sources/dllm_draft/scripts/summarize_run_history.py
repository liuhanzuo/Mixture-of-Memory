#!/usr/bin/env python3
"""Summarize registered attempt wall time and allocated GPU-hours."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


TERMINAL = {
    "COMPLETED",
    "RECOVERED_COMPLETED",
    "RECORDED_ABLATION",
    "NEEDS_DEBUG",
    "RETRYING",
}
RESOURCE_RE = re.compile(r"(?:^|\s)resource=(\w+)")


def read_history(path: Path) -> list[dict[str, str]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = line.split("\t", 3)
        if len(fields) < 3:
            continue
        rows.append(
            {
                "timestamp": fields[0],
                "status": fields[1],
                "run_id": fields[2],
                "message": fields[3] if len(fields) > 3 else "",
            }
        )
    return rows


def summarize_run(
    events: list[dict[str, str]],
    run_id: str,
    *,
    gpu_count: int,
) -> dict[str, Any]:
    relevant = [event for event in events if event["run_id"] == run_id]
    attempts = []
    active: dict[str, Any] | None = None
    for event in relevant:
        timestamp = datetime.fromisoformat(event["timestamp"])
        if event["status"] == "LAUNCHED":
            if active is not None:
                raise ValueError(f"{run_id} has overlapping launch attempts")
            match = RESOURCE_RE.search(event["message"])
            resource = match.group(1) if match else "unknown"
            active = {
                "launched_at": event["timestamp"],
                "launched_datetime": timestamp,
                "resource": resource,
            }
            continue
        if active is not None and event["status"] in TERMINAL:
            elapsed = (
                timestamp - active.pop("launched_datetime")
            ).total_seconds()
            attempts.append(
                {
                    **active,
                    "ended_at": event["timestamp"],
                    "terminal_status": event["status"],
                    "elapsed_seconds": elapsed,
                }
            )
            active = None
    if active is not None:
        raise ValueError(f"{run_id} still has an open attempt")
    if not attempts:
        raise ValueError(f"{run_id} has no completed attempts")
    total_seconds = sum(float(row["elapsed_seconds"]) for row in attempts)
    first = datetime.fromisoformat(attempts[0]["launched_at"])
    last = datetime.fromisoformat(attempts[-1]["ended_at"])
    uses_gpu = any(row["resource"] == "gpu8" for row in attempts)
    return {
        "run_id": run_id,
        "attempts": attempts,
        "attempt_count": len(attempts),
        "total_active_seconds": total_seconds,
        "total_active_hours": total_seconds / 3600,
        "wall_span_seconds": (last - first).total_seconds(),
        "wall_span_hours": (last - first).total_seconds() / 3600,
        "gpu_count": gpu_count if uses_gpu else 0,
        "allocated_gpu_hours": (
            total_seconds * gpu_count / 3600 if uses_gpu else 0.0
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--history", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--gpu-count", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    events = read_history(args.history)
    runs = []
    for item in args.run:
        label, separator, run_id = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=RUN_ID")
        runs.append(
            {
                "label": label,
                **summarize_run(
                    events,
                    run_id,
                    gpu_count=args.gpu_count,
                ),
            }
        )
    report = {
        "gpu_count": args.gpu_count,
        "runs": runs,
        "total_allocated_gpu_hours": sum(
            float(row["allocated_gpu_hours"]) for row in runs
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
