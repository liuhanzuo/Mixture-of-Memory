from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_paired_eval",
    ROOT / "scripts" / "analyze_paired_eval.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PairedEvalTests(unittest.TestCase):
    def test_paired_counts_and_exact_test(self) -> None:
        report = MODULE.analyze(
            {
                "a": {"0": True, "1": True, "2": False, "3": False},
                "b": {"0": True, "1": False, "2": True, "3": False},
            },
            [("a", "b")],
            bootstrap_replicates=1000,
            seed=7,
        )
        pair = report["pairs"][0]
        self.assertEqual(pair["both_pass"], 1)
        self.assertEqual(pair["a_only"], 1)
        self.assertEqual(pair["b_only"], 1)
        self.assertEqual(pair["neither_pass"], 1)
        self.assertEqual(pair["delta"], 0.0)
        self.assertEqual(pair["mcnemar_exact_p"], 1.0)
        self.assertEqual(pair["mcnemar_holm_p"], 1.0)

    def test_exact_mcnemar_and_wilson_bounds(self) -> None:
        self.assertEqual(MODULE.exact_mcnemar_p(4, 0), 0.125)
        lower, upper = MODULE.wilson_interval(5, 10)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)
        self.assertEqual(
            MODULE.holm_adjust([0.01, 0.04, 0.03]),
            [0.03, 0.06, 0.06],
        )

    def test_plus_outcome_requires_base_and_extra_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "eval.json"
            path.write_text(
                json.dumps(
                    {
                        "eval": {
                            "a": [
                                {
                                    "base_status": "fail",
                                    "plus_status": "pass",
                                }
                            ],
                            "b": [
                                {
                                    "base_status": "pass",
                                    "plus_status": "pass",
                                }
                            ],
                        }
                    }
                ),
                encoding="utf-8",
            )
            outcomes = MODULE.read_outcomes(path, "plus")
            self.assertFalse(outcomes["a"])
            self.assertTrue(outcomes["b"])


if __name__ == "__main__":
    unittest.main()
