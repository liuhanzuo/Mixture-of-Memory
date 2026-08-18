from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_correction_calibration_set",
    ROOT / "scripts" / "build_correction_calibration_set.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

EVAL_SPEC = importlib.util.spec_from_file_location(
    "evaluate_correction_calibration",
    ROOT / "scripts" / "evaluate_correction_calibration.py",
)
EVAL_MODULE = importlib.util.module_from_spec(EVAL_SPEC)
assert EVAL_SPEC.loader is not None
EVAL_SPEC.loader.exec_module(EVAL_MODULE)

SELECT_SPEC = importlib.util.spec_from_file_location(
    "select_correction_calibration",
    ROOT / "scripts" / "select_correction_calibration.py",
)
SELECT_MODULE = importlib.util.module_from_spec(SELECT_SPEC)
assert SELECT_SPEC.loader is not None
SELECT_SPEC.loader.exec_module(SELECT_MODULE)


def row(seq_id: int, code: str) -> dict[str, object]:
    return {
        "seq_id": seq_id,
        "prompt": f"Task {seq_id}",
        "code": code,
        "entry_point": f"f{seq_id}",
        "testcase": [f"assert f{seq_id}() is not None"],
        "prompt_tokens": 10,
        "response_tokens": 10,
    }


class CorrectionCalibrationSetTests(unittest.TestCase):
    def test_depth_and_header_extraction(self) -> None:
        source = (
            "def f(x):\n"
            "    for item in x:\n"
            "        if item:\n"
            "            return item\n"
            "    return None\n"
        )
        self.assertEqual(MODULE.function_header(source), "f(x)")
        self.assertEqual(MODULE.maximum_compound_depth(source), 2)
        prepared = MODULE.prepare_row(
            {
                **row(1, source),
                "entry_point": "f",
            }
        )
        self.assertEqual(prepared["depth_group"], "depth_2")
        self.assertIn("def f(x):", prepared["prompt"])

    def test_stratified_selection_is_deterministic(self) -> None:
        rows = [
            row(index, f"def f{index}():\n    return {index}\n")
            for index in range(4)
        ]
        rows.extend(
            row(
                10 + index,
                (
                    f"def f{10 + index}():\n"
                    "    if True:\n"
                    "        if True:\n"
                    "            return 1\n"
                ),
            )
            for index in range(4)
        )
        rows.extend(
            row(
                20 + index,
                (
                    f"def f{20 + index}():\n"
                    "    if True:\n"
                    "        if True:\n"
                    "            if True:\n"
                    "                return 1\n"
                ),
            )
            for index in range(4)
        )
        quotas = {
            "depth_0_1": 2,
            "depth_2": 2,
            "depth_3_plus": 2,
        }
        first = MODULE.select_rows(rows, quotas=quotas, seed=7)
        second = MODULE.select_rows(rows, quotas=quotas, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(
            {group: sum(row["depth_group"] == group for row in first)
             for group in quotas},
            quotas,
        )
        excluded = {row["seq_id"] for row in first}
        remaining = [
            candidate
            for candidate in rows
            if candidate["seq_id"] not in excluded
        ]
        third = MODULE.select_rows(remaining, quotas=quotas, seed=7)
        self.assertFalse(
            {row["seq_id"] for row in first}
            & {row["seq_id"] for row in third}
        )

    def test_sandboxed_execution_classifies_pass_and_failure(self) -> None:
        passed = EVAL_MODULE.evaluate_source(
            "def f(x):\n    return x + 1\n",
            ["assert f(1) == 2"],
            timeout_seconds=2.0,
            memory_mib=256,
        )
        self.assertEqual(passed["status"], "pass")
        failed = EVAL_MODULE.evaluate_source(
            "def f(x):\n    return x\n",
            ["assert f(1) == 2"],
            timeout_seconds=2.0,
            memory_mib=256,
        )
        self.assertEqual(failed["status"], "assertion_failure")
        syntax = EVAL_MODULE.evaluate_source(
            "def f(:\n    pass\n",
            [],
            timeout_seconds=2.0,
            memory_mib=256,
        )
        self.assertEqual(syntax["status"], "syntax_error")

    def test_selection_requires_quality_and_cost_improvement(self) -> None:
        baseline = {
            "pass_rate": 0.25,
            "parse_rate": 0.75,
            "mean_nfe": 50.0,
            "mean_cumulative_model_tokens": 1000.0,
        }
        report = SELECT_MODULE.select_arm(
            {
                "c0": baseline,
                "c2_good": {
                    "pass_rate": 0.30,
                    "parse_rate": 0.80,
                    "mean_nfe": 55.0,
                    "mean_cumulative_model_tokens": 1100.0,
                },
                "c2_expensive": {
                    "pass_rate": 0.40,
                    "parse_rate": 0.90,
                    "mean_nfe": 80.0,
                    "mean_cumulative_model_tokens": 1700.0,
                },
            },
            max_nfe_ratio=1.25,
            max_token_ratio=1.35,
            policy_name="C2",
        )
        self.assertEqual(report["selected_label"], "c2_good")
        self.assertEqual(report["policy"], "C2")
        markdown = SELECT_MODULE.render_markdown(report)
        self.assertIn("**`c2_good`**", markdown)
        self.assertIn("C2 backtracks", markdown)
        fallback = SELECT_MODULE.select_arm(
            {
                "c0": baseline,
                "c2_worse": {
                    "pass_rate": 0.20,
                    "parse_rate": 0.80,
                    "mean_nfe": 55.0,
                    "mean_cumulative_model_tokens": 1100.0,
                },
            },
            max_nfe_ratio=1.25,
            max_token_ratio=1.35,
            policy_name="C2",
        )
        self.assertEqual(fallback["selected_label"], "c0")


if __name__ == "__main__":
    unittest.main()
