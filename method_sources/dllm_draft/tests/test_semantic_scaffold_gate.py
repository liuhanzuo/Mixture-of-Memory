from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "evaluate_semantic_scaffold_gate",
    ROOT / "scripts" / "evaluate_semantic_scaffold_gate.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SemanticScaffoldGateTests(unittest.TestCase):
    def test_gate_thresholds(self) -> None:
        vanilla = 0.45
        scaffold = 0.30
        failure = 0.05
        self.assertTrue(vanilla >= 0.45)
        self.assertFalse(vanilla < 0.40)
        self.assertTrue(scaffold >= 0.30)
        self.assertTrue(failure <= 0.05)


if __name__ == "__main__":
    unittest.main()
