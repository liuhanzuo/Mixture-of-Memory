from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "build_runtime_capacity_screen",
    ROOT / "scripts" / "build_runtime_capacity_screen.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class RuntimeCapacityScreenTests(unittest.TestCase):
    def test_deterministic_depth_stratified_selection(self) -> None:
        rows = []
        for index in range(12):
            depth = index % 3
            body = "    return 1\n"
            if depth >= 1:
                body = "    if True:\n        return 1\n"
            if depth >= 2:
                body = (
                    "    if True:\n"
                    "        if True:\n"
                    "            return 1\n"
                )
            rows.append(
                {
                    "task_id": f"T/{index}",
                    "prompt": f"def f{index}():\n",
                    "canonical_solution": body,
                }
            )
        first = MODULE.build(rows, size=9, seed=4)
        second = MODULE.build(rows, size=9, seed=4)
        self.assertEqual(first, second)
        groups = {
            row["capacity_depth_group"] for row in first
        }
        self.assertEqual(
            groups,
            {"depth_0_1", "depth_2"},
        )


if __name__ == "__main__":
    unittest.main()
