from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_training_metrics",
    ROOT / "scripts" / "summarize_training_metrics.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class TrainingMetricSummaryTests(unittest.TestCase):
    def test_excludes_warmup_and_summarizes_available_fields(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "metrics.jsonl"
            records = [
                {
                    "train/step_seconds": 10.0,
                    "train/nonpadding_tokens_per_second": 100.0,
                },
                {
                    "train/step_seconds": 2.0,
                    "train/nonpadding_tokens_per_second": 500.0,
                    "train/padding_fraction": 0.1,
                },
                {
                    "train/step_seconds": 4.0,
                    "train/nonpadding_tokens_per_second": 300.0,
                    "train/padding_fraction": 0.2,
                },
            ]
            path.write_text(
                "".join(json.dumps(record) + "\n" for record in records),
                encoding="utf-8",
            )
            result = MODULE.summarize(
                "test",
                path,
                warmup_records=1,
            )
            self.assertEqual(
                result["train/step_seconds"]["median"],
                3.0,
            )
            self.assertEqual(
                result["train/nonpadding_tokens_per_second"]["mean"],
                400.0,
            )
            self.assertAlmostEqual(
                result["train/padding_fraction"]["mean"],
                0.15,
            )


if __name__ == "__main__":
    unittest.main()
