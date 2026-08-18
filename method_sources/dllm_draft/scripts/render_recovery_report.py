#!/usr/bin/env python3
"""Render the Base/Instruct, epoch-count, and high-noise recovery study."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def indexed(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["label"]: row for row in report["runs"]}


def summarize(
    comparison: dict[str, Any],
    paired: dict[str, Any],
) -> dict[str, Any]:
    runs = indexed(comparison)
    required = {
        "base_raw",
        "base_plain1",
        "base_plain5",
        "instruct_raw",
        "instruct_plain1",
        "instruct_highnoise1",
    }
    missing = required - set(runs)
    if missing:
        raise ValueError(f"missing recovery runs: {sorted(missing)}")

    def delta(a: str, b: str) -> float:
        return float(runs[a]["plus_pass1"]) - float(runs[b]["plus_pass1"])

    effects = {
        "base_plain1_minus_base_raw": delta("base_plain1", "base_raw"),
        "base_plain5_minus_base_plain1": delta("base_plain5", "base_plain1"),
        "instruct_plain1_minus_instruct_raw": delta(
            "instruct_plain1", "instruct_raw"
        ),
        "highnoise_minus_instruct_plain1": delta(
            "instruct_highnoise1", "instruct_plain1"
        ),
    }
    best = max(
        required,
        key=lambda label: (
            float(runs[label]["plus_pass1"]),
            float(runs[label]["parse_rate"]),
        ),
    )
    return {
        "runs": [runs[label] for label in sorted(required)],
        "effects": effects,
        "best_label": best,
        "paired": paired,
    }


def pct(value: float, signed: bool = False) -> str:
    return f"{100 * value:{'+' if signed else ''}.2f}%"


def render(report: dict[str, Any]) -> str:
    lines = [
        "# SFT Recovery Diagnostic",
        "",
        "| Run | HumanEval | HumanEval+ | Parseable | Mean seconds |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            f"| {row['label']} | {pct(row['base_pass1'])} | "
            f"{pct(row['plus_pass1'])} | {pct(row['parse_rate'])} | "
            f"{float(row['mean_elapsed_seconds']):.2f} |"
        )
    lines.extend(
        [
            "",
            "## Controlled effects",
            "",
            "- Base + one low-LR epoch minus raw Base: "
            + pct(
                report["effects"]["base_plain1_minus_base_raw"],
                signed=True,
            ),
            "- Base five epochs minus Base one epoch: "
            + pct(
                report["effects"]["base_plain5_minus_base_plain1"],
                signed=True,
            ),
            "- Instruct + one low-LR epoch minus raw Instruct: "
            + pct(
                report["effects"]["instruct_plain1_minus_instruct_raw"],
                signed=True,
            ),
            "- Instruct high-noise minus Instruct uniform one epoch: "
            + pct(
                report["effects"]["highnoise_minus_instruct_plain1"],
                signed=True,
            ),
            "",
            f"Best observed run: **`{report['best_label']}`**.",
            "",
            "These effects isolate initialization, over-training, and "
            "prompt-only/high-noise supervision; they do not reuse MBPP for "
            "selection.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        json.loads(args.comparison.read_text(encoding="utf-8")),
        json.loads(args.paired.read_text(encoding="utf-8")),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.write_text(render(report), encoding="utf-8")
    print(render(report))


if __name__ == "__main__":
    main()
