from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "analyze_training_loss",
    ROOT / "scripts" / "analyze_training_loss.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainingLossTests(unittest.TestCase):
    def test_summarizes_latest_unique_steps_and_validation(self) -> None:
        text = "\n".join(
            [
                "step:0 - train/loss:2.0",
                "step:1 - train/loss:1.5",
                "step:1 - train/loss:1.4",
                "step:2 - train/loss:1.0",
                "step:2 - val/loss:1.2",
            ]
        )
        report = MODULE.summarize_log(text, window_size=1)
        self.assertEqual(report["records"], 3)
        self.assertEqual(report["last_step"], 2)
        self.assertEqual(report["first_window"]["mean"], 2.0)
        self.assertEqual(report["last_window"]["mean"], 1.0)
        self.assertLess(report["last_500_slope_per_step"], 0)
        self.assertEqual(report["last_validation_loss"], 1.2)


if __name__ == "__main__":
    unittest.main()
