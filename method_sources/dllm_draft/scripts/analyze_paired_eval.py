#!/usr/bin/env python3
"""Paired pass@1 deltas, bootstrap intervals, and exact McNemar tests."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def read_outcomes(path: Path, metric: str) -> dict[str, bool]:
    evaluation = json.loads(path.read_text(encoding="utf-8"))["eval"]
    if metric == "base":
        return {
            task_id: rows[0]["base_status"] == "pass"
            for task_id, rows in evaluation.items()
        }
    return {
        task_id: (
            rows[0]["base_status"] == "pass"
            and rows[0]["plus_status"] == "pass"
        )
        for task_id, rows in evaluation.items()
    }


def wilson_interval(passes: int, total: int, z: float = 1.95996398454):
    if total <= 0:
        raise ValueError("total must be positive")
    proportion = passes / total
    denominator = 1 + z * z / total
    center = (proportion + z * z / (2 * total)) / denominator
    radius = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total
            + z * z / (4 * total * total)
        )
        / denominator
    )
    return [max(0.0, center - radius), min(1.0, center + radius)]


def exact_mcnemar_p(a_only: int, b_only: int) -> float:
    discordant = a_only + b_only
    if discordant == 0:
        return 1.0
    smaller = min(a_only, b_only)
    tail = sum(
        math.comb(discordant, index)
        for index in range(smaller + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def bootstrap_delta_interval(
    differences: list[int],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    if not differences:
        raise ValueError("differences cannot be empty")
    if replicates <= 0:
        raise ValueError("replicates must be positive")
    rng = random.Random(seed)
    count = len(differences)
    samples = []
    for _ in range(replicates):
        samples.append(
            sum(differences[rng.randrange(count)] for _ in range(count))
            / count
        )
    samples.sort()
    lower = samples[round(0.025 * (replicates - 1))]
    upper = samples[round(0.975 * (replicates - 1))]
    return [lower, upper]


def analyze(
    runs: dict[str, dict[str, bool]],
    pairs: list[tuple[str, str]],
    *,
    bootstrap_replicates: int,
    seed: int,
) -> dict[str, Any]:
    if not runs:
        raise ValueError("at least one run is required")
    task_sets = {label: set(values) for label, values in runs.items()}
    reference = next(iter(task_sets.values()))
    for label, tasks in task_sets.items():
        if tasks != reference:
            raise ValueError(f"task coverage differs for {label}")
    tasks = sorted(reference)
    run_rows = {}
    for label, outcomes in sorted(runs.items()):
        passes = sum(outcomes[task] for task in tasks)
        run_rows[label] = {
            "tasks": len(tasks),
            "passes": passes,
            "pass_rate": passes / len(tasks),
            "wilson_95": wilson_interval(passes, len(tasks)),
        }

    pair_rows = []
    for pair_index, (a_label, b_label) in enumerate(pairs):
        if a_label not in runs or b_label not in runs:
            raise ValueError(f"unknown pair {a_label},{b_label}")
        a = runs[a_label]
        b = runs[b_label]
        both_pass = sum(a[task] and b[task] for task in tasks)
        a_only = sum(a[task] and not b[task] for task in tasks)
        b_only = sum(b[task] and not a[task] for task in tasks)
        neither = len(tasks) - both_pass - a_only - b_only
        differences = [
            int(a[task]) - int(b[task])
            for task in tasks
        ]
        pair_rows.append(
            {
                "a": a_label,
                "b": b_label,
                "tasks": len(tasks),
                "a_pass_rate": run_rows[a_label]["pass_rate"],
                "b_pass_rate": run_rows[b_label]["pass_rate"],
                "delta": sum(differences) / len(differences),
                "bootstrap_95": bootstrap_delta_interval(
                    differences,
                    replicates=bootstrap_replicates,
                    seed=seed + pair_index,
                ),
                "both_pass": both_pass,
                "a_only": a_only,
                "b_only": b_only,
                "neither_pass": neither,
                "discordant": a_only + b_only,
                "mcnemar_exact_p": exact_mcnemar_p(a_only, b_only),
            }
        )
    adjusted = holm_adjust(
        [float(row["mcnemar_exact_p"]) for row in pair_rows]
    )
    for row, value in zip(pair_rows, adjusted, strict=True):
        row["mcnemar_holm_p"] = value
    return {
        "bootstrap_replicates": bootstrap_replicates,
        "seed": seed,
        "runs": run_rows,
        "pairs": pair_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--pair", action="append", required=True)
    parser.add_argument("--metric", choices=("base", "plus"), default="plus")
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260724)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runs = {}
    for item in args.run:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=PATH")
        runs[label] = read_outcomes(Path(path), args.metric)
    pairs = []
    for item in args.pair:
        a, separator, b = item.partition(",")
        if not separator:
            raise ValueError("--pair must be A,B")
        pairs.append((a, b))
    report = {
        "metric": args.metric,
        **analyze(
            runs,
            pairs,
            bootstrap_replicates=args.bootstrap_replicates,
            seed=args.seed,
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
