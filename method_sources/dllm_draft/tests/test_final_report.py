from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_final_report",
    ROOT / "scripts" / "render_final_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def comparison(
    *,
    schedule_quality: float,
    plain_quality: float,
    schedule_low: float,
    plain_low: float,
    schedule_parse: float = 0.8,
    plain_parse: float = 0.8,
) -> dict[str, object]:
    def row(label: str, plus: float, parse: float) -> dict[str, object]:
        return {
            "label": label,
            "rows": 10,
            "base_pass1": plus,
            "plus_pass1": plus,
            "generation_errors": 0,
            "nonempty_outputs": 10,
            "parseable_outputs": round(10 * parse),
            "parse_rate": parse,
            "mean_nfe": 64.0,
            "median_nfe": 64.0,
            "mean_elapsed_seconds": 1.0,
        }

    return {
        "runs": [
            row("dream512", 0.5, 0.9),
            row("dream128", 0.4, 0.8),
            row("dream64", 0.3, 0.6),
            row("scaffold", 0.2, 0.85),
            row("schedule512", schedule_quality, schedule_parse),
            row("schedule128", schedule_low + 0.05, schedule_parse),
            row("schedule64", schedule_low, schedule_parse),
            row("plain512", plain_quality, plain_parse),
            row("plain128", plain_low + 0.05, plain_parse),
            row("plain64", plain_low, plain_parse),
        ]
    }


class FinalReportTests(unittest.TestCase):
    def test_retains_schedule_after_clear_matched_gain(self) -> None:
        he = comparison(
            schedule_quality=0.42,
            plain_quality=0.40,
            schedule_low=0.32,
            plain_low=0.30,
        )
        mbpp = comparison(
            schedule_quality=0.62,
            plain_quality=0.60,
            schedule_low=0.0,
            plain_low=0.0,
        )
        decision = MODULE.select_g1_pivot(he, mbpp)
        self.assertEqual(
            decision["schedule_decision"],
            "retain_depth_schedule",
        )
        self.assertEqual(
            decision["meta_token_decision"],
            "drop_meta_tokens_as_primary",
        )
        self.assertIn("schedule", decision["efficiency_curves"])

    def test_drops_schedule_after_clear_matched_regression(self) -> None:
        he = comparison(
            schedule_quality=0.37,
            plain_quality=0.40,
            schedule_low=0.28,
            plain_low=0.30,
        )
        mbpp = comparison(
            schedule_quality=0.56,
            plain_quality=0.60,
            schedule_low=0.0,
            plain_low=0.0,
        )
        decision = MODULE.select_g1_pivot(he, mbpp)
        self.assertEqual(
            decision["schedule_decision"],
            "drop_depth_schedule",
        )

    def test_renders_all_final_sections(self) -> None:
        he = comparison(
            schedule_quality=0.42,
            plain_quality=0.40,
            schedule_low=0.32,
            plain_low=0.30,
        )
        mbpp = comparison(
            schedule_quality=0.62,
            plain_quality=0.60,
            schedule_low=0.0,
            plain_low=0.0,
        )
        decision = MODULE.select_g1_pivot(he, mbpp)
        depth = {
            "depth_counts": {
                "depth_0_1": 6,
                "depth_2": 3,
                "depth_3_plus": 1,
            },
            "runs": {
                label: {
                    group: {
                        "count": count,
                        "plus_pass1": 0.5,
                    }
                    for group, count in (
                        ("depth_0_1", 6),
                        ("depth_2", 3),
                        ("depth_3_plus", 1),
                    )
                }
                for label in ("dream512", "scaffold", "schedule512", "plain512")
            },
        }
        failures = {
            "runs": [
                {
                    "label": label,
                    "counts": {"plus_pass": 5, "syntax_error": 5},
                }
                for label in (
                    "dream512",
                    "dream64",
                    "scaffold",
                    "schedule512",
                    "schedule64",
                    "plain512",
                    "plain64",
                )
            ]
        }
        metric_names = (
            "train/step_seconds",
            "train/examples_per_second",
            "train/nonpadding_tokens_per_second",
            "train/padding_fraction",
            "train/maximum_sequence_length",
            "train/peak_reserved_gib",
        )
        training = {
            "runs": [
                {
                    "label": label,
                    **{
                        name: {"mean": 1.0}
                        for name in metric_names
                    },
                }
                for label in ("scaffold", "schedule", "plain")
            ]
        }
        run_costs = {
            "total_allocated_gpu_hours": 24.0,
            "runs": [
                {
                    "label": label,
                    "attempt_count": 1,
                    "total_active_hours": 1.0,
                    "wall_span_hours": 1.0,
                    "allocated_gpu_hours": 8.0,
                }
                for label in ("scaffold", "schedule", "plain")
            ],
        }
        training_loss = {
            "runs": [
                {
                    "label": label,
                    "records": 1000,
                    "first_window": {"mean": 2.0},
                    "last_window": {"mean": 1.0},
                    "last_minus_first_mean": -1.0,
                    "last_500_slope_per_100_steps": -0.01,
                    "last_validation_loss": 1.1,
                }
                for label in ("scaffold", "schedule", "plain")
            ]
        }
        paired = {
            "pairs": [
                {
                    "a": "schedule512",
                    "b": "plain512",
                    "a_pass_rate": 0.42,
                    "b_pass_rate": 0.40,
                    "delta": 0.02,
                    "bootstrap_95": [-0.01, 0.05],
                    "a_only": 3,
                    "b_only": 1,
                    "mcnemar_exact_p": 0.625,
                    "mcnemar_holm_p": 0.625,
                }
            ]
        }
        report = MODULE.render_report(
            humaneval=he,
            mbpp=mbpp,
            humaneval_depth=depth,
            mbpp_depth=depth,
            training=training,
            training_run_costs=run_costs,
            training_loss=training_loss,
            humaneval_failures=failures,
            mbpp_failures=failures,
            humaneval_paired=paired,
            mbpp_paired=paired,
            decision=decision,
        )
        self.assertIn("# Scaffold-Coder Final Matched Comparison", report)
        self.assertIn("## Training efficiency", report)
        self.assertIn("### HumanEval 128-NFE comparison", report)
        self.assertIn("### HumanEval+ efficiency curve", report)
        self.assertIn("Allocated GPU-hours", report)
        self.assertIn("### Loss trajectory", report)
        self.assertIn("## Paired task-level analysis", report)
        self.assertIn("`retain_depth_schedule`", report)


if __name__ == "__main__":
    unittest.main()
