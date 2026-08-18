#!/usr/bin/env python3
"""Summarize per-step training-loss trajectories from registered logs."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


TRAIN_RE = re.compile(r"step:(\d+)\s+-\s+train/loss:([0-9.eE+-]+)")
VAL_RE = re.compile(r"step:(\d+)\s+-\s+val/loss:([0-9.eE+-]+)")


def linear_slope(points: list[tuple[int, float]]) -> float | None:
    if len(points) < 2:
        return None
    x_mean = statistics.mean(point[0] for point in points)
    y_mean = statistics.mean(point[1] for point in points)
    denominator = sum((x - x_mean) ** 2 for x, _ in points)
    if denominator == 0:
        return None
    return sum(
        (x - x_mean) * (y - y_mean)
        for x, y in points
    ) / denominator


def window(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
    }


def summarize_log(text: str, *, window_size: int = 100) -> dict[str, Any]:
    # A resumed log may repeat steps; the latest observation is authoritative.
    train = {
        int(step): float(loss)
        for step, loss in TRAIN_RE.findall(text)
    }
    validation = {
        int(step): float(loss)
        for step, loss in VAL_RE.findall(text)
    }
    if not train:
        raise ValueError("log contains no train/loss records")
    points = sorted(train.items())
    if not all(math.isfinite(loss) for _, loss in points):
        raise ValueError("training loss contains non-finite values")
    first_values = [loss for _, loss in points[:window_size]]
    last_values = [loss for _, loss in points[-window_size:]]
    slope_points = points[-min(500, len(points)) :]
    slope = linear_slope(slope_points)
    last_val_step = max(validation) if validation else None
    return {
        "records": len(points),
        "first_step": points[0][0],
        "last_step": points[-1][0],
        "first_window": window(first_values),
        "last_window": window(last_values),
        "last_minus_first_mean": (
            statistics.mean(last_values) - statistics.mean(first_values)
        ),
        "last_over_first_mean": (
            statistics.mean(last_values) / statistics.mean(first_values)
        ),
        "last_500_slope_per_step": slope,
        "last_500_slope_per_100_steps": (
            slope * 100 if slope is not None else None
        ),
        "minimum_loss": min(loss for _, loss in points),
        "maximum_loss": max(loss for _, loss in points),
        "validation_records": len(validation),
        "last_validation_step": last_val_step,
        "last_validation_loss": (
            validation[last_val_step] if last_val_step is not None else None
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    runs = []
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=LOG_PATH")
        runs.append(
            {
                "label": label,
                "path": str(Path(path).resolve()),
                **summarize_log(
                    Path(path).read_text(
                        encoding="utf-8",
                        errors="replace",
                    ),
                    window_size=args.window_size,
                ),
            }
        )
    report = {"window_size": args.window_size, "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
