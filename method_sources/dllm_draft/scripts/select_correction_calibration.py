#!/usr/bin/env python3
"""Select a held-out C2 threshold under explicit quality/cost gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_runs(items: list[str]) -> dict[str, dict[str, Any]]:
    runs = {}
    for item in items:
        label, separator, path = item.partition("=")
        if not separator:
            raise ValueError("--run must be LABEL=PATH")
        runs[label] = json.loads(Path(path).read_text(encoding="utf-8"))
    return runs


def select_arm(
    runs: dict[str, dict[str, Any]],
    *,
    max_nfe_ratio: float,
    max_token_ratio: float,
    policy_name: str = "correction",
) -> dict[str, Any]:
    if "c0" not in runs:
        raise ValueError("calibration runs must include c0")
    baseline = runs["c0"]
    baseline_nfe = float(baseline["mean_nfe"])
    baseline_tokens = float(baseline["mean_cumulative_model_tokens"])
    comparisons = []
    for label, row in sorted(runs.items()):
        nfe_ratio = float(row["mean_nfe"]) / baseline_nfe
        token_ratio = (
            float(row["mean_cumulative_model_tokens"]) / baseline_tokens
        )
        comparisons.append(
            {
                "label": label,
                "pass_rate": float(row["pass_rate"]),
                "parse_rate": float(row["parse_rate"]),
                "mean_nfe": float(row["mean_nfe"]),
                "mean_cumulative_model_tokens": float(
                    row["mean_cumulative_model_tokens"]
                ),
                "mean_structural_backtracks": float(
                    row.get("mean_structural_backtracks") or 0.0
                ),
                "mean_leaf_remasks": float(
                    row.get("mean_leaf_remasks") or 0.0
                ),
                "mean_structural_deferrals": float(
                    row.get("mean_structural_deferrals") or 0.0
                ),
                "pass_delta": (
                    float(row["pass_rate"]) - float(baseline["pass_rate"])
                ),
                "parse_delta": (
                    float(row["parse_rate"]) - float(baseline["parse_rate"])
                ),
                "nfe_ratio": nfe_ratio,
                "token_ratio": token_ratio,
                "eligible_cost": (
                    nfe_ratio <= max_nfe_ratio
                    and token_ratio <= max_token_ratio
                ),
            }
        )
    eligible_c2 = [
        row
        for row in comparisons
        if (
            row["label"] != "c0"
            and row["eligible_cost"]
            and row["pass_delta"] > 0
            and row["parse_delta"] >= 0
        )
    ]
    if eligible_c2:
        selected = max(
            eligible_c2,
            key=lambda row: (
                row["pass_rate"],
                row["parse_rate"],
                -row["token_ratio"],
                row["label"],
            ),
        )
        reason = (
            f"Selected the highest-pass {policy_name} arm that improves held-out pass "
            "and parse rates while staying inside both cost gates."
        )
    else:
        selected = next(
            row for row in comparisons if row["label"] == "c0"
        )
        reason = (
            f"No {policy_name} arm jointly improved held-out pass and parse rates within "
            "the preregistered NFE and cumulative-token cost gates."
        )
    return {
        "policy": policy_name,
        "selected_label": selected["label"],
        "reason": reason,
        "max_nfe_ratio": max_nfe_ratio,
        "max_token_ratio": max_token_ratio,
        "baseline": baseline,
        "comparisons": comparisons,
    }


def percent(value: float) -> str:
    return f"{100 * value:.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# {report['policy']} Held-Out Calibration Results",
        "",
        "The 32-task set is drawn only from the fixed educational_instruct "
        "evaluation split. It is not HumanEval, MBPP, or a Plus benchmark.",
        "",
        "| Arm | Pass | Parse | Mean NFE | Mean cumulative tokens | "
        "C1 remasks | C3 deferrals | C2 backtracks | NFE ratio | "
        "Token ratio | Cost eligible |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    for row in report["comparisons"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["label"]),
                    percent(float(row["pass_rate"])),
                    percent(float(row["parse_rate"])),
                    f"{float(row['mean_nfe']):.2f}",
                    f"{float(row['mean_cumulative_model_tokens']):.1f}",
                    f"{float(row['mean_leaf_remasks']):.3f}",
                    f"{float(row['mean_structural_deferrals']):.3f}",
                    f"{float(row['mean_structural_backtracks']):.3f}",
                    f"{float(row['nfe_ratio']):.3f}",
                    f"{float(row['token_ratio']):.3f}",
                    "yes" if row["eligible_cost"] else "no",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Selected arm: **`{report['selected_label']}`**.",
            "",
            report["reason"],
            "",
            "Selection requires a strict execution-pass improvement, no "
            "parse-rate regression, mean NFE ≤1.25× C0, and cumulative model "
            "tokens ≤1.35× C0. This calibration routes a later correction "
            "experiment; it is not a benchmark or significance claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="append", required=True)
    parser.add_argument("--max-nfe-ratio", type=float, default=1.25)
    parser.add_argument("--max-token-ratio", type=float, default=1.35)
    parser.add_argument("--policy", default="correction")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path)
    args = parser.parse_args()
    report = select_arm(
        read_runs(args.run),
        max_nfe_ratio=args.max_nfe_ratio,
        max_token_ratio=args.max_token_ratio,
        policy_name=args.policy,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_markdown(report),
            encoding="utf-8",
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
