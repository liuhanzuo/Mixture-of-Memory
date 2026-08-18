#!/usr/bin/env python3
"""Combine independently calibrated C1/C2/C3 routing decisions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def selected_row(report: dict[str, Any]) -> dict[str, Any]:
    label = report["selected_label"]
    return next(row for row in report["comparisons"] if row["label"] == label)


def summarize(reports: list[dict[str, Any]]) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one correction report is required")
    baseline = next(
        row
        for row in reports[0]["comparisons"]
        if row["label"] == "c0"
    )
    baseline_keys = (
        "pass_rate",
        "parse_rate",
        "mean_nfe",
        "mean_cumulative_model_tokens",
    )
    for report in reports[1:]:
        candidate = next(
            row
            for row in report["comparisons"]
            if row["label"] == "c0"
        )
        if any(candidate[key] != baseline[key] for key in baseline_keys):
            raise ValueError(
                f"{report['policy']} does not share the identical C0 baseline"
            )
    rows = []
    for report in reports:
        row = selected_row(report)
        rows.append(
            {
                "policy": report["policy"],
                "selected_label": report["selected_label"],
                "selection_reason": report["reason"],
                **{
                    key: row[key]
                    for key in (
                        "pass_rate",
                        "parse_rate",
                        "pass_delta",
                        "parse_delta",
                        "mean_nfe",
                        "mean_cumulative_model_tokens",
                        "mean_leaf_remasks",
                        "mean_structural_deferrals",
                        "mean_structural_backtracks",
                        "nfe_ratio",
                        "token_ratio",
                        "eligible_cost",
                    )
                },
            }
        )
    improvements = [
        row
        for row in rows
        if row["selected_label"] != "c0"
    ]
    if improvements:
        best = max(
            improvements,
            key=lambda row: (
                row["pass_rate"],
                row["parse_rate"],
                -row["token_ratio"],
                row["policy"],
            ),
        )
        best_policy = best["policy"]
        best_label = best["selected_label"]
        reason = (
            "Selected the strongest independently eligible correction arm "
            "by pass rate, then parse rate and cumulative-token cost."
        )
    else:
        best_policy = "C0"
        best_label = "c0"
        reason = "No correction policy cleared its held-out quality/cost gate."
    return {
        "baseline": baseline,
        "policies": rows,
        "best_policy": best_policy,
        "best_label": best_label,
        "reason": reason,
    }


def pct(value: float, *, signed: bool = False) -> str:
    return f"{100 * value:{'+' if signed else ''}.2f}%"


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Unified Correction Policy Calibration",
        "",
        "| Policy | Selected arm | Pass | Parse | Pass delta | NFE ratio | "
        "Token ratio | C1 remasks | C3 deferrals | C2 backtracks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["policies"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["policy"]),
                    str(row["selected_label"]),
                    pct(float(row["pass_rate"])),
                    pct(float(row["parse_rate"])),
                    pct(float(row["pass_delta"]), signed=True),
                    f"{float(row['nfe_ratio']):.3f}",
                    f"{float(row['token_ratio']):.3f}",
                    f"{float(row['mean_leaf_remasks']):.3f}",
                    f"{float(row['mean_structural_deferrals']):.3f}",
                    f"{float(row['mean_structural_backtracks']):.3f}",
                )
            )
            + " |"
        )
    lines.extend(
        [
            "",
            f"Best routed policy: **`{report['best_policy']}`** "
            f"(`{report['best_label']}`).",
            "",
            report["reason"],
            "",
            "This selection is held-out routing for a later separately "
            "reported correction evaluation, not a benchmark claim.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        [
            json.loads(path.read_text(encoding="utf-8"))
            for path in args.report
        ]
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
