from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_failure_taxonomy",
    ROOT / "scripts" / "analyze_failure_taxonomy.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class FailureTaxonomyTests(unittest.TestCase):
    def test_classification_priority(self) -> None:
        passed = {"base_status": "pass", "plus_status": "pass"}
        self.assertEqual(
            MODULE.classify(
                solution="",
                metric={"error": "budget"},
                evaluation=passed,
            ),
            "generation_error",
        )
        self.assertEqual(
            MODULE.classify(
                solution="def f(:\n",
                metric={"error": None},
                evaluation=passed,
            ),
            "syntax_error",
        )
        self.assertEqual(
            MODULE.classify(
                solution="def f():\n    pass\n",
                metric={"error": None},
                evaluation={
                    "base_status": "pass",
                    "plus_status": "fail",
                },
            ),
            "plus_only_failure",
        )
        self.assertEqual(
            MODULE.classify(
                solution="def f():\n    pass\n",
                metric={"error": None},
                evaluation=passed,
            ),
            "plus_pass",
        )


if __name__ == "__main__":
    unittest.main()
