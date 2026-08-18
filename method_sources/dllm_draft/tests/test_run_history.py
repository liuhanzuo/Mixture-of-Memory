from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "summarize_run_history",
    ROOT / "scripts" / "summarize_run_history.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RunHistoryTests(unittest.TestCase):
    def test_sums_retried_gpu_attempts(self) -> None:
        events = [
            {
                "timestamp": "2026-01-01T00:00:00+00:00",
                "status": "LAUNCHED",
                "run_id": "RUN",
                "message": "pid=1 resource=gpu8",
            },
            {
                "timestamp": "2026-01-01T00:10:00+00:00",
                "status": "RETRYING",
                "run_id": "RUN",
                "message": "retry=1",
            },
            {
                "timestamp": "2026-01-01T00:12:00+00:00",
                "status": "LAUNCHED",
                "run_id": "RUN",
                "message": "pid=2 resource=gpu8",
            },
            {
                "timestamp": "2026-01-01T00:32:00+00:00",
                "status": "COMPLETED",
                "run_id": "RUN",
                "message": "done",
            },
        ]
        report = MODULE.summarize_run(events, "RUN", gpu_count=8)
        self.assertEqual(report["attempt_count"], 2)
        self.assertEqual(report["total_active_seconds"], 1800)
        self.assertEqual(report["wall_span_seconds"], 1920)
        self.assertEqual(report["allocated_gpu_hours"], 4.0)


if __name__ == "__main__":
    unittest.main()
