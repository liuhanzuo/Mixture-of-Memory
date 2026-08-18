from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_correction_policies",
    ROOT / "scripts" / "summarize_correction_policies.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def policy(name: str, label: str, pass_rate: float, token_ratio: float):
    base = {
        "label": "c0",
        "pass_rate": 0.25,
        "parse_rate": 0.75,
        "pass_delta": 0.0,
        "parse_delta": 0.0,
        "mean_nfe": 50.0,
        "mean_cumulative_model_tokens": 1000.0,
        "mean_leaf_remasks": 0.0,
        "mean_structural_deferrals": 0.0,
        "mean_structural_backtracks": 0.0,
        "nfe_ratio": 1.0,
        "token_ratio": 1.0,
        "eligible_cost": True,
    }
    selected = {
        **base,
        "label": label,
        "pass_rate": pass_rate,
        "parse_rate": 0.80,
        "pass_delta": pass_rate - 0.25,
        "parse_delta": 0.05,
        "token_ratio": token_ratio,
    }
    return {
        "policy": name,
        "selected_label": label,
        "reason": "test",
        "comparisons": [base, selected] if label != "c0" else [base],
    }


class CorrectionPolicySummaryTests(unittest.TestCase):
    def test_selects_best_independently_eligible_policy(self) -> None:
        report = MODULE.summarize(
            [
                policy("C1", "c1_t010", 0.31, 1.2),
                policy("C2", "c2_t010", 0.34, 1.3),
                policy("C3", "c0", 0.25, 1.0),
            ]
        )
        self.assertEqual(report["best_policy"], "C2")
        self.assertEqual(report["best_label"], "c2_t010")
        markdown = MODULE.render_markdown(report)
        self.assertIn("**`C2`**", markdown)

    def test_rejects_mismatched_c0_baseline(self) -> None:
        c1 = policy("C1", "c0", 0.25, 1.0)
        c2 = policy("C2", "c0", 0.25, 1.0)
        c2["comparisons"][0]["pass_rate"] = 0.30
        with self.assertRaises(ValueError):
            MODULE.summarize([c1, c2])


if __name__ == "__main__":
    unittest.main()
