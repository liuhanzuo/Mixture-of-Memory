from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "compare_correction_validation",
    ROOT / "scripts" / "compare_correction_validation.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class CorrectionValidationTests(unittest.TestCase):
    def test_paired_validation_gate(self) -> None:
        baseline = [
            {"task_id": str(i), "status": status}
            for i, status in enumerate(("pass", "pass", "fail", "fail"))
        ]
        selected = [
            {"task_id": str(i), "status": status}
            for i, status in enumerate(("pass", "pass", "pass", "fail"))
        ]
        base_summary = {
            "pass_rate": 0.5,
            "parse_rate": 0.75,
            "mean_nfe": 50,
            "mean_cumulative_model_tokens": 1000,
        }
        selected_summary = {
            "pass_rate": 0.75,
            "parse_rate": 1.0,
            "mean_nfe": 55,
            "mean_cumulative_model_tokens": 1100,
            "mean_leaf_remasks": 1,
        }
        report = MODULE.compare(
            baseline,
            selected,
            base_summary,
            selected_summary,
            policy="C1",
            label="c1_t010",
            replicates=1000,
            seed=9,
        )
        self.assertEqual(report["selected_only"], 1)
        self.assertEqual(report["baseline_only"], 0)
        self.assertTrue(report["passes_validation_gate"])
        self.assertIn("C1", MODULE.render_markdown(report))


if __name__ == "__main__":
    unittest.main()
