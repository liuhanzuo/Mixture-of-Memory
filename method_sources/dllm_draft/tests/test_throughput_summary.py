from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class ThroughputSummaryTests(unittest.TestCase):
    def test_fast_candidate_without_headroom_is_not_selected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_dir = Path(temporary)
            (run_dir / "status.tsv").write_text(
                "micro_batch_size_per_gpu\tstatus\texit_code\t"
                "log_file\tmetrics_file\n"
                "16\tsuccess\t0\tmicro_16.log\tmicro_16.metrics.jsonl\n"
                "8\tsuccess\t0\tmicro_8.log\tmicro_8.metrics.jsonl\n"
                "4\tfailed\t1\tmicro_4.log\tmicro_4.metrics.jsonl\n",
                encoding="utf-8",
            )
            self.write_metrics(run_dir / "micro_16.metrics.jsonl", 2000, 93)
            self.write_metrics(run_dir / "micro_8.metrics.jsonl", 1500, 85)
            output = run_dir / "summary.json"
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "summarize_throughput_sweep.py"),
                    "--run-dir",
                    str(run_dir),
                    "--output",
                    str(output),
                    "--gpu-memory-mib",
                    str(96 * 1024),
                    "--minimum-headroom-gib",
                    "5",
                    "--warmup-records",
                    "1",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            report = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                report["recommended_micro_batch_size_per_gpu"], 8
            )
            self.assertTrue(report["selection_used_headroom_constraint"])
            candidates = {
                item["micro_batch_size_per_gpu"]: item
                for item in report["candidates"]
            }
            self.assertEqual(candidates[16]["steady_records"], 2)
            self.assertLess(candidates[16]["memory_headroom_gib"], 5)
            self.assertGreater(candidates[8]["memory_headroom_gib"], 5)
            self.assertEqual(
                candidates[8]["median_padding_fraction"],
                0.1,
            )

    @staticmethod
    def write_metrics(path: Path, tokens_per_second: float, reserved: float):
        records = [
            {
                "train/step_seconds": 99.0,
                "train/examples_per_second": 1.0,
                "train/nonpadding_tokens_per_second": 1.0,
                "train/supervised_tokens_per_second": 1.0,
                "train/peak_allocated_gib": reserved - 2,
                "train/peak_reserved_gib": reserved,
                "train/padding_fraction": 0.1,
                "train/maximum_sequence_length": 256,
            },
            {
                "train/step_seconds": 2.0,
                "train/examples_per_second": 64.0,
                "train/nonpadding_tokens_per_second": tokens_per_second,
                "train/supervised_tokens_per_second": tokens_per_second / 2,
                "train/peak_allocated_gib": reserved - 2,
                "train/peak_reserved_gib": reserved,
                "train/padding_fraction": 0.1,
                "train/maximum_sequence_length": 256,
            },
            {
                "train/step_seconds": 2.2,
                "train/examples_per_second": 60.0,
                "train/nonpadding_tokens_per_second": tokens_per_second * 0.9,
                "train/supervised_tokens_per_second": tokens_per_second / 2,
                "train/peak_allocated_gib": reserved - 1,
                "train/peak_reserved_gib": reserved,
                "train/padding_fraction": 0.1,
                "train/maximum_sequence_length": 256,
            },
        ]
        path.write_text(
            "".join(json.dumps(record) + "\n" for record in records),
            encoding="utf-8",
        )


if __name__ == "__main__":
    unittest.main()
