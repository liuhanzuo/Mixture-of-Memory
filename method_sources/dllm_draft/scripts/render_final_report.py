#!/usr/bin/env python3
"""Render the matched control matrix and a deterministic G1 recommendation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


QUALITY_LABELS = ("dream512", "scaffold", "schedule512", "plain512")
MID_NFE_LABELS = ("dream128", "schedule128", "plain128")
LOW_NFE_LABELS = ("dream64", "scaffold", "schedule64", "plain64")
TRAINING_LABELS = ("scaffold", "schedule", "plain")
FAILURE_KEYS = (
    "generation_error",
    "empty_output",
    "syntax_error",
    "base_timeout",
    "base_semantic_failure",
    "plus_timeout",
    "plus_only_failure",
    "plus_pass",
)
DEPTH_LABELS = (
    ("depth_0_1", "0–1"),
    ("depth_2", "2"),
    ("depth_3_plus", "3+"),
)
CURVE_LABELS = {
    "dream": ("dream512", "dream128", "dream64"),
    "schedule": ("schedule512", "schedule128", "schedule64"),
    "plain": ("plain512", "plain128", "plain64"),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def indexed_runs(report: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(row["label"]): row for row in report["runs"]}


def require_labels(
    runs: dict[str, dict[str, Any]],
    labels: tuple[str, ...],
    *,
    source: str,
) -> None:
    missing = [label for label in labels if label not in runs]
    if missing:
        raise ValueError(f"{source} is missing runs: {', '.join(missing)}")


def percent(value: float | int | None) -> str:
    return "—" if value is None else f"{100 * float(value):.2f}%"


def number(value: float | int | None, digits: int = 2) -> str:
    return "—" if value is None else f"{float(value):.{digits}f}"


def signed_percent(value: float) -> str:
    return f"{100 * value:+.2f}%"


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def efficiency_curves(
    humaneval: dict[str, Any],
) -> dict[str, dict[str, float]]:
    runs = indexed_runs(humaneval)
    required = tuple(
        label
        for labels in CURVE_LABELS.values()
        for label in labels
    )
    require_labels(runs, required, source="HumanEval efficiency curve")
    curves = {}
    for name, (label512, label128, label64) in CURVE_LABELS.items():
        pass512 = float(runs[label512]["plus_pass1"])
        pass128 = float(runs[label128]["plus_pass1"])
        pass64 = float(runs[label64]["plus_pass1"])
        # Trapezoidal AUC over log2(NFE): x={6,7,9}, normalized by width 3.
        log_auc = (
            0.5 * (pass64 + pass128)
            + 2 * 0.5 * (pass128 + pass512)
        ) / 3
        curves[name] = {
            "pass512": pass512,
            "pass128": pass128,
            "pass64": pass64,
            "drop_512_to_128": pass128 - pass512,
            "drop_128_to_64": pass64 - pass128,
            "drop_512_to_64": pass64 - pass512,
            "retention_64_over_512": (
                pass64 / pass512 if pass512 > 0 else 0.0
            ),
            "log2_nfe_auc": log_auc,
            "parse64": float(runs[label64]["parse_rate"]),
        }
    return curves


def select_g1_pivot(
    humaneval: dict[str, Any],
    mbpp: dict[str, Any],
) -> dict[str, Any]:
    """Apply the preregistered one-point matched-control decision margin."""

    he = indexed_runs(humaneval)
    mb = indexed_runs(mbpp)
    require_labels(
        he,
        ("scaffold", "schedule512", "schedule64", "plain512", "plain64"),
        source="HumanEval comparison",
    )
    require_labels(
        mb,
        ("schedule512", "plain512"),
        source="MBPP comparison",
    )

    schedule_quality_deltas = {
        "humaneval_plus": (
            float(he["schedule512"]["plus_pass1"])
            - float(he["plain512"]["plus_pass1"])
        ),
        "mbpp_plus": (
            float(mb["schedule512"]["plus_pass1"])
            - float(mb["plain512"]["plus_pass1"])
        ),
    }
    schedule_quality_deltas["mean"] = mean(
        list(schedule_quality_deltas.values())
    )
    low_nfe_deltas = {
        "humaneval_plus": (
            float(he["schedule64"]["plus_pass1"])
            - float(he["plain64"]["plus_pass1"])
        ),
        "parse_rate": (
            float(he["schedule64"]["parse_rate"])
            - float(he["plain64"]["parse_rate"])
        ),
    }
    scaffold_vs_schedule64 = {
        "humaneval_plus": (
            float(he["scaffold"]["plus_pass1"])
            - float(he["schedule64"]["plus_pass1"])
        ),
        "parse_rate": (
            float(he["scaffold"]["parse_rate"])
            - float(he["schedule64"]["parse_rate"])
        ),
    }
    curves = efficiency_curves(humaneval)

    margin = 0.01
    parse_tolerance = 0.02
    schedule_mean = schedule_quality_deltas["mean"]
    low_plus = low_nfe_deltas["humaneval_plus"]
    low_parse = low_nfe_deltas["parse_rate"]
    if (
        schedule_mean >= margin
        or (low_plus >= margin and low_parse >= -parse_tolerance)
    ):
        schedule_decision = "retain_depth_schedule"
        schedule_reason = (
            "The depth schedule clears the one-point quality margin, or "
            "clears it at 64 NFE without a material parseability regression."
        )
    elif schedule_mean <= -margin and low_plus <= 0:
        schedule_decision = "drop_depth_schedule"
        schedule_reason = (
            "The matched plain control is at least one point better on mean "
            "quality and is not worse on HumanEval+ at 64 NFE."
        )
    else:
        schedule_decision = "schedule_effect_inconclusive"
        schedule_reason = (
            "The schedule/plain deltas do not clear the preregistered "
            "one-point decision margin consistently."
        )

    meta_quality = scaffold_vs_schedule64["humaneval_plus"]
    meta_parse = scaffold_vs_schedule64["parse_rate"]
    if meta_quality >= -margin and meta_parse >= 0:
        meta_decision = "retain_meta_tokens"
        meta_reason = (
            "At matched HumanEval NFE, meta-token decoding is no more than one "
            "quality point below schedule-only (and may be better) while being "
            "at least as parseable."
        )
    else:
        meta_decision = "drop_meta_tokens_as_primary"
        meta_reason = (
            "At matched HumanEval NFE, the meta-token checkpoint misses the "
            "quality/parseability joint gate; retain it only as a process "
            "and constrained-decoding arm."
        )

    if schedule_decision == "retain_depth_schedule":
        next_action = (
            "Use schedule-only as the primary training arm; run explicit "
            "syntax repair and low-NFE tuning, while reporting C2/meta-token "
            "backtracking separately."
        )
    elif schedule_decision == "drop_depth_schedule":
        next_action = (
            "Use plain matched SFT as the primary checkpoint; evaluate "
            "syntax repair/constrained decoding and keep structural runtime "
            "experiments as separately costed arms."
        )
    else:
        next_action = (
            "Run a small replicated schedule/plain seed or overlap sweep "
            "before selecting the primary training distribution."
        )

    return {
        "rule_version": 1,
        "decision_margin": margin,
        "parse_tolerance": parse_tolerance,
        "schedule_quality_delta": schedule_quality_deltas,
        "schedule_low_nfe_delta": low_nfe_deltas,
        "scaffold_vs_schedule64": scaffold_vs_schedule64,
        "efficiency_curves": curves,
        "schedule_decision": schedule_decision,
        "schedule_reason": schedule_reason,
        "meta_token_decision": meta_decision,
        "meta_token_reason": meta_reason,
        "next_action": next_action,
    }


def comparison_table(
    report: dict[str, Any],
    labels: tuple[str, ...],
) -> list[str]:
    runs = indexed_runs(report)
    require_labels(runs, labels, source="comparison")
    lines = [
        "| Run | Base pass@1 | Plus pass@1 | Parseable | Gen. errors | "
        "Mean NFE | Median NFE | Mean seconds/sample |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in labels:
        row = runs[label]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    percent(row["base_pass1"]),
                    percent(row["plus_pass1"]),
                    percent(row["parse_rate"]),
                    str(row["generation_errors"]),
                    number(row.get("mean_nfe")),
                    number(row.get("median_nfe")),
                    number(row.get("mean_elapsed_seconds")),
                )
            )
            + " |"
        )
    return lines


def failure_table(
    report: dict[str, Any],
    labels: tuple[str, ...],
) -> list[str]:
    runs = indexed_runs(report)
    require_labels(runs, labels, source="failure taxonomy")
    headers = [key.replace("_", " ") for key in FAILURE_KEYS]
    lines = [
        "| Run | " + " | ".join(headers) + " |",
        "|---|" + "|".join("---:" for _ in headers) + "|",
    ]
    for label in labels:
        counts = runs[label]["counts"]
        lines.append(
            f"| {label} | "
            + " | ".join(str(counts.get(key, 0)) for key in FAILURE_KEYS)
            + " |"
        )
    return lines


def depth_table(report: dict[str, Any]) -> list[str]:
    runs = report["runs"]
    labels = tuple(runs)
    lines = [
        "| Depth | Tasks | "
        + " | ".join(f"{label} Plus pass@1" for label in labels)
        + " |",
        "|---:|---:|" + "|".join("---:" for _ in labels) + "|",
    ]
    for group, display in DEPTH_LABELS:
        count = int(report["depth_counts"].get(group, 0))
        values = [
            percent(runs[label].get(group, {}).get("plus_pass1"))
            for label in labels
        ]
        lines.append(f"| {display} | {count} | " + " | ".join(values) + " |")
    return lines


def training_table(report: dict[str, Any]) -> list[str]:
    runs = indexed_runs(report)
    require_labels(runs, TRAINING_LABELS, source="training efficiency")
    lines = [
        "| Run | Step seconds | Examples/s | Non-padding tokens/s | "
        "Padding fraction | Max sequence | Peak reserved GiB |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]

    def metric(row: dict[str, Any], name: str) -> float | None:
        value = row.get(name)
        return None if value is None else float(value["mean"])

    for label in TRAINING_LABELS:
        row = runs[label]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    number(metric(row, "train/step_seconds")),
                    number(metric(row, "train/examples_per_second")),
                    number(metric(row, "train/nonpadding_tokens_per_second")),
                    percent(metric(row, "train/padding_fraction")),
                    number(metric(row, "train/maximum_sequence_length"), 1),
                    number(metric(row, "train/peak_reserved_gib")),
                )
            )
            + " |"
        )
    return lines


def efficiency_curve_table(curves: dict[str, dict[str, float]]) -> list[str]:
    lines = [
        "| Model | HE+ @512 | HE+ @128 | HE+ @64 | 64/512 retention | "
        "512→64 delta | log2-NFE AUC | Parse @64 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label in ("dream", "schedule", "plain"):
        row = curves[label]
        lines.append(
            "| "
            + " | ".join(
                (
                    label,
                    percent(row["pass512"]),
                    percent(row["pass128"]),
                    percent(row["pass64"]),
                    percent(row["retention_64_over_512"]),
                    signed_percent(row["drop_512_to_64"]),
                    percent(row["log2_nfe_auc"]),
                    percent(row["parse64"]),
                )
            )
            + " |"
        )
    return lines


def run_cost_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Run | Attempts | Active hours | Wall-span hours | "
        "Allocated GPU-hours |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["label"]),
                    str(row["attempt_count"]),
                    f"{float(row['total_active_hours']):.3f}",
                    f"{float(row['wall_span_hours']):.3f}",
                    f"{float(row['allocated_gpu_hours']):.2f}",
                )
            )
            + " |"
        )
    lines.append(
        "| **total** | — | — | — | "
        f"**{float(report['total_allocated_gpu_hours']):.2f}** |"
    )
    return lines


def training_loss_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Run | Records | First-100 mean | Last-100 mean | Delta | "
        "Last-500 slope /100 steps | Last val loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["runs"]:
        slope = row["last_500_slope_per_100_steps"]
        validation = row["last_validation_loss"]
        lines.append(
            "| "
            + " | ".join(
                (
                    str(row["label"]),
                    str(row["records"]),
                    f"{float(row['first_window']['mean']):.4f}",
                    f"{float(row['last_window']['mean']):.4f}",
                    f"{float(row['last_minus_first_mean']):+.4f}",
                    "—" if slope is None else f"{float(slope):+.5f}",
                    "—" if validation is None else f"{float(validation):.4f}",
                )
            )
            + " |"
        )
    return lines


def paired_table(report: dict[str, Any]) -> list[str]:
    lines = [
        "| Pair (A − B) | A pass | B pass | Delta | Bootstrap 95% | "
        "A-only / B-only | McNemar raw / Holm p |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report["pairs"]:
        interval = row["bootstrap_95"]
        lines.append(
            "| "
            + " | ".join(
                (
                    f"{row['a']} − {row['b']}",
                    percent(row["a_pass_rate"]),
                    percent(row["b_pass_rate"]),
                    signed_percent(row["delta"]),
                    (
                        f"[{signed_percent(interval[0])}, "
                        f"{signed_percent(interval[1])}]"
                    ),
                    f"{row['a_only']} / {row['b_only']}",
                    (
                        f"{float(row['mcnemar_exact_p']):.4g} / "
                        f"{float(row['mcnemar_holm_p']):.4g}"
                    ),
                )
            )
            + " |"
        )
    return lines


def render_report(
    *,
    humaneval: dict[str, Any],
    mbpp: dict[str, Any],
    humaneval_depth: dict[str, Any],
    mbpp_depth: dict[str, Any],
    training: dict[str, Any],
    training_run_costs: dict[str, Any],
    training_loss: dict[str, Any],
    humaneval_failures: dict[str, Any],
    mbpp_failures: dict[str, Any],
    humaneval_paired: dict[str, Any],
    mbpp_paired: dict[str, Any],
    decision: dict[str, Any],
) -> str:
    lines = [
        "# Scaffold-Coder Final Matched Comparison",
        "",
        "This report is generated from the released JSON artifacts. Percentages "
        "are single-sample pass@1 or parse rates; NFE is the number of model "
        "forward evaluations.",
        "",
        "## HumanEval: quality and matched-cost runs",
        "",
        *comparison_table(humaneval, QUALITY_LABELS),
        "",
        "### HumanEval low-NFE comparison",
        "",
        *comparison_table(humaneval, LOW_NFE_LABELS),
        "",
        "### HumanEval 128-NFE comparison",
        "",
        *comparison_table(humaneval, MID_NFE_LABELS),
        "",
        "### HumanEval+ efficiency curve",
        "",
        *efficiency_curve_table(decision["efficiency_curves"]),
        "",
        "The AUC uses trapezoidal integration over log2 NFE at 64, 128, and "
        "512 calls, normalized to the same [6,9] interval.",
        "",
        "## MBPP quality runs",
        "",
        *comparison_table(mbpp, QUALITY_LABELS),
        "",
        "## Failure taxonomy",
        "",
        "### HumanEval",
        "",
        *failure_table(humaneval_failures, LOW_NFE_LABELS),
        "",
        "### MBPP",
        "",
        *failure_table(mbpp_failures, QUALITY_LABELS),
        "",
        "## HumanEval+ by canonical compound depth",
        "",
        *depth_table(humaneval_depth),
        "",
        "## MBPP+ by canonical compound depth",
        "",
        *depth_table(mbpp_depth),
        "",
        "## Training efficiency",
        "",
        *training_table(training),
        "",
        "### Registered training wall time and allocation",
        "",
        *run_cost_table(training_run_costs),
        "",
        "Allocated GPU-hours are eight times registered active wall time; "
        "they are allocation accounting, not measured electrical energy.",
        "",
        "### Loss trajectory",
        "",
        *training_loss_table(training_loss),
        "",
        "Loss windows summarize stochastic masked-diffusion batches and are "
        "diagnostic rather than directly comparable objectives across modes.",
        "",
        "## Paired task-level analysis",
        "",
        "### HumanEval+",
        "",
        *paired_table(humaneval_paired),
        "",
        "### MBPP+",
        "",
        *paired_table(mbpp_paired),
        "",
        "Intervals are deterministic paired bootstrap intervals. McNemar "
        "p-values are exact two-sided tests over discordant tasks. Holm values "
        "adjust within the listed HumanEval or MBPP comparison family.",
        "",
        "## Automatic G1 attribution",
        "",
        f"- Schedule decision: `{decision['schedule_decision']}`.",
        f"- Meta-token decision: `{decision['meta_token_decision']}`.",
        "- Mean schedule-only minus plain 512-NFE Plus delta: "
        f"{percent(decision['schedule_quality_delta']['mean'])}.",
        "- HumanEval+ schedule-only minus plain 64-NFE delta: "
        f"{percent(decision['schedule_low_nfe_delta']['humaneval_plus'])}.",
        "- HumanEval parse-rate schedule-only minus plain 64-NFE delta: "
        f"{percent(decision['schedule_low_nfe_delta']['parse_rate'])}.",
        "- Schedule minus plain log2-NFE AUC delta: "
        f"{signed_percent(decision['efficiency_curves']['schedule']['log2_nfe_auc'] - decision['efficiency_curves']['plain']['log2_nfe_auc'])}.",
        f"- Schedule rationale: {decision['schedule_reason']}",
        f"- Meta-token rationale: {decision['meta_token_reason']}",
        f"- Next action: {decision['next_action']}",
        "",
        "The automatic recommendation is an experiment-routing rule, not a "
        "statistical-significance claim.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--humaneval-comparison", type=Path, required=True)
    parser.add_argument("--mbpp-comparison", type=Path, required=True)
    parser.add_argument("--humaneval-depth", type=Path, required=True)
    parser.add_argument("--mbpp-depth", type=Path, required=True)
    parser.add_argument("--training-efficiency", type=Path, required=True)
    parser.add_argument("--training-run-costs", type=Path, required=True)
    parser.add_argument("--training-loss", type=Path, required=True)
    parser.add_argument("--humaneval-failures", type=Path, required=True)
    parser.add_argument("--mbpp-failures", type=Path, required=True)
    parser.add_argument("--humaneval-paired", type=Path, required=True)
    parser.add_argument("--mbpp-paired", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--decision-output", type=Path, required=True)
    args = parser.parse_args()

    humaneval = read_json(args.humaneval_comparison)
    mbpp = read_json(args.mbpp_comparison)
    decision = select_g1_pivot(humaneval, mbpp)
    report = render_report(
        humaneval=humaneval,
        mbpp=mbpp,
        humaneval_depth=read_json(args.humaneval_depth),
        mbpp_depth=read_json(args.mbpp_depth),
        training=read_json(args.training_efficiency),
        training_run_costs=read_json(args.training_run_costs),
        training_loss=read_json(args.training_loss),
        humaneval_failures=read_json(args.humaneval_failures),
        mbpp_failures=read_json(args.mbpp_failures),
        humaneval_paired=read_json(args.humaneval_paired),
        mbpp_paired=read_json(args.mbpp_paired),
        decision=decision,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report, encoding="utf-8")
    args.decision_output.parent.mkdir(parents=True, exist_ok=True)
    args.decision_output.write_text(
        json.dumps(decision, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(report, end="")


if __name__ == "__main__":
    main()
