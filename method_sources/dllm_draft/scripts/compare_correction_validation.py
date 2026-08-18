#!/usr/bin/env python3
"""Compare selected correction against C0 on a disjoint held-out set."""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


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


def bootstrap_interval(
    differences: list[int],
    *,
    replicates: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    count = len(differences)
    values = sorted(
        sum(differences[rng.randrange(count)] for _ in range(count)) / count
        for _ in range(replicates)
    )
    return [
        values[round(0.025 * (replicates - 1))],
        values[round(0.975 * (replicates - 1))],
    ]


def compare(
    baseline_rows: list[dict[str, Any]],
    selected_rows: list[dict[str, Any]],
    baseline_summary: dict[str, Any],
    selected_summary: dict[str, Any],
    *,
    policy: str,
    label: str,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    baseline = {row["task_id"]: row["status"] == "pass" for row in baseline_rows}
    selected = {row["task_id"]: row["status"] == "pass" for row in selected_rows}
    if set(baseline) != set(selected):
        raise ValueError("baseline and selected task coverage differs")
    tasks = sorted(baseline)
    selected_only = sum(selected[task] and not baseline[task] for task in tasks)
    baseline_only = sum(baseline[task] and not selected[task] for task in tasks)
    both = sum(selected[task] and baseline[task] for task in tasks)
    neither = len(tasks) - selected_only - baseline_only - both
    differences = [
        int(selected[task]) - int(baseline[task])
        for task in tasks
    ]
    baseline_nfe = float(baseline_summary["mean_nfe"])
    baseline_tokens = float(baseline_summary["mean_cumulative_model_tokens"])
    report = {
        "policy": policy,
        "label": label,
        "tasks": len(tasks),
        "baseline_pass_rate": float(baseline_summary["pass_rate"]),
        "selected_pass_rate": float(selected_summary["pass_rate"]),
        "pass_delta": sum(differences) / len(differences),
        "bootstrap_95": bootstrap_interval(
            differences,
            replicates=replicates,
            seed=seed,
        ),
        "mcnemar_exact_p": exact_mcnemar_p(selected_only, baseline_only),
        "both_pass": both,
        "selected_only": selected_only,
        "baseline_only": baseline_only,
        "neither_pass": neither,
        "baseline_parse_rate": float(baseline_summary["parse_rate"]),
        "selected_parse_rate": float(selected_summary["parse_rate"]),
        "parse_delta": (
            float(selected_summary["parse_rate"])
            - float(baseline_summary["parse_rate"])
        ),
        "nfe_ratio": float(selected_summary["mean_nfe"]) / baseline_nfe,
        "token_ratio": (
            float(selected_summary["mean_cumulative_model_tokens"])
            / baseline_tokens
        ),
        "selected_mean_leaf_remasks": float(
            selected_summary.get("mean_leaf_remasks") or 0
        ),
        "selected_mean_structural_deferrals": float(
            selected_summary.get("mean_structural_deferrals") or 0
        ),
        "selected_mean_structural_backtracks": float(
            selected_summary.get("mean_structural_backtracks") or 0
        ),
    }
    report["passes_validation_gate"] = (
        report["pass_delta"] > 0
        and report["parse_delta"] >= 0
        and report["nfe_ratio"] <= 1.25
        and report["token_ratio"] <= 1.35
    )
    return report


def pct(value: float, signed: bool = False) -> str:
    return f"{100 * value:{'+' if signed else ''}.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    interval = report["bootstrap_95"]
    return "\n".join(
        [
            "# Selected Correction Disjoint Validation",
            "",
            f"Policy: **`{report['policy']}`** (`{report['label']}`).",
            "",
            "| Metric | C0 | Selected | Delta/ratio |",
            "|---|---:|---:|---:|",
            f"| Execution pass | {pct(report['baseline_pass_rate'])} | "
            f"{pct(report['selected_pass_rate'])} | "
            f"{pct(report['pass_delta'], signed=True)} |",
            f"| Parse rate | {pct(report['baseline_parse_rate'])} | "
            f"{pct(report['selected_parse_rate'])} | "
            f"{pct(report['parse_delta'], signed=True)} |",
            f"| Mean NFE ratio | — | — | {report['nfe_ratio']:.3f}× |",
            f"| Mean token ratio | — | — | {report['token_ratio']:.3f}× |",
            "",
            f"Paired bootstrap 95% delta interval: "
            f"`[{pct(interval[0], True)}, {pct(interval[1], True)}]`.",
            "",
            f"Discordant tasks: selected-only {report['selected_only']}, "
            f"C0-only {report['baseline_only']}; exact McNemar "
            f"`p={report['mcnemar_exact_p']:.4g}`.",
            "",
            f"Validation gate passed: **{report['passes_validation_gate']}**.",
            "",
            "This set is disjoint from correction calibration and from the "
            "reported code benchmarks.",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-evaluation", type=Path, required=True)
    parser.add_argument("--selected-evaluation", type=Path, required=True)
    parser.add_argument("--baseline-summary", type=Path, required=True)
    parser.add_argument("--selected-summary", type=Path, required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = compare(
        read_jsonl(args.baseline_evaluation),
        read_jsonl(args.selected_evaluation),
        json.loads(args.baseline_summary.read_text(encoding="utf-8")),
        json.loads(args.selected_summary.read_text(encoding="utf-8")),
        policy=args.policy,
        label=args.label,
        replicates=args.bootstrap_replicates,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(
        render_markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
