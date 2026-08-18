from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_recovery_report",
    ROOT / "scripts" / "render_recovery_report.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RecoveryReportTests(unittest.TestCase):
    def test_controlled_effects(self) -> None:
        rates = {
            "base_raw": 0.10,
            "base_plain1": 0.20,
            "base_plain5": 0.15,
            "instruct_raw": 0.50,
            "instruct_plain1": 0.45,
            "instruct_highnoise1": 0.48,
        }
        comparison = {
            "runs": [
                {
                    "label": label,
                    "base_pass1": rate,
                    "plus_pass1": rate,
                    "parse_rate": rate,
                    "mean_elapsed_seconds": 1.0,
                }
                for label, rate in rates.items()
            ]
        }
        report = MODULE.summarize(comparison, {"pairs": []})
        self.assertAlmostEqual(
            report["effects"]["base_plain1_minus_base_raw"],
            0.10,
        )
        self.assertAlmostEqual(
            report["effects"]["base_plain5_minus_base_plain1"],
            -0.05,
        )
        self.assertEqual(report["best_label"], "instruct_raw")
        self.assertIn("Controlled effects", MODULE.render(report))


if __name__ == "__main__":
    unittest.main()
