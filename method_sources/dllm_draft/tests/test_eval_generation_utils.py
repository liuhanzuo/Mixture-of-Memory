from __future__ import annotations

import ast
import importlib.util
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "generate_evalplus_dream",
    ROOT / "scripts" / "generate_evalplus_dream.py",
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
SCAFFOLD_SPEC = importlib.util.spec_from_file_location(
    "generate_evalplus_scaffold",
    ROOT / "scripts" / "generate_evalplus_scaffold.py",
)
SCAFFOLD_MODULE = importlib.util.module_from_spec(SCAFFOLD_SPEC)
assert SCAFFOLD_SPEC.loader is not None
SCAFFOLD_SPEC.loader.exec_module(SCAFFOLD_MODULE)
DREAMON_SPEC = importlib.util.spec_from_file_location(
    "generate_evalplus_dreamon",
    ROOT / "scripts" / "generate_evalplus_dreamon.py",
)
DREAMON_MODULE = importlib.util.module_from_spec(DREAMON_SPEC)
assert DREAMON_SPEC.loader is not None
DREAMON_SPEC.loader.exec_module(DREAMON_MODULE)
DEPTH_SPEC = importlib.util.spec_from_file_location(
    "analyze_eval_by_depth",
    ROOT / "scripts" / "analyze_eval_by_depth.py",
)
DEPTH_MODULE = importlib.util.module_from_spec(DEPTH_SPEC)
assert DEPTH_SPEC.loader is not None
DEPTH_SPEC.loader.exec_module(DEPTH_MODULE)


class EvalGenerationUtilityTests(unittest.TestCase):
    def test_extracts_python_fence(self) -> None:
        text = (
            "Sure.\n```python\n"
            "def add(a, b):\n"
            "    return a + b\n"
            "```\n"
        )
        self.assertEqual(
            MODULE.extract_python(text),
            "def add(a, b):\n    return a + b\n",
        )

    def test_strips_leading_explanation_without_fence(self) -> None:
        text = "Here is the answer:\n\ndef identity(x):\n    return x\n"
        self.assertEqual(
            MODULE.extract_python(text),
            "def identity(x):\n    return x\n",
        )

    def test_extracts_unclosed_python_fence(self) -> None:
        text = (
            "Explanation mentioning from a list.\n"
            "```python\n"
            "from typing import List\n\n"
            "def rolling_max(values: List[int]):\n"
            "    return values\n"
        )
        self.assertEqual(
            MODULE.extract_python(text),
            "from typing import List\n\n"
            "def rolling_max(values: List[int]):\n"
            "    return values\n",
        )

    def test_base_continuation_prepends_bos_and_prompt(self) -> None:
        task = {"prompt": "def identity(x):\n    \"\"\"Return x.\"\"\"\n"}
        self.assertEqual(
            MODULE.raw_base_prompt(task, "<bos>"),
            "<bos>def identity(x):\n    \"\"\"Return x.\"\"\"\n",
        )
        self.assertEqual(
            MODULE.combine_base_continuation(task, "    return x\n"),
            "def identity(x):\n"
            "    \"\"\"Return x.\"\"\"\n"
            "    return x\n",
        )

    def test_extracts_seed_function_header(self) -> None:
        prompt = (
            "from typing import List\n\n"
            "def has_close_elements(numbers: List[float], "
            "threshold: float) -> bool:\n"
            "    \"\"\"Check values.\"\"\"\n"
        )
        self.assertEqual(
            SCAFFOLD_MODULE.function_header_from_prompt(prompt),
            "has_close_elements(numbers: List[float], "
            "threshold: float) -> bool",
        )
        self.assertEqual(
            SCAFFOLD_MODULE.function_header_for_task(
                {
                    "prompt": "def wrong():\n    pass\n",
                    "function_header": "required(x)",
                }
            ),
            "required(x)",
        )

    def test_combines_dreamon_body_with_humaneval_prompt(self) -> None:
        prompt = (
            "def identity(x):\n"
            "    \"\"\"Return x.\"\"\"\n"
        )
        self.assertEqual(
            DREAMON_MODULE.combine_humaneval_prompt(
                prompt,
                "return x\n",
            ),
            "def identity(x):\n"
            "    \"\"\"Return x.\"\"\"\n"
            "    return x\n",
        )
        self.assertEqual(
            DREAMON_MODULE.combine_humaneval_prompt(prompt, ""),
            "def identity(x):\n"
            "    \"\"\"Return x.\"\"\"\n"
            "    pass\n",
        )

    def test_combines_already_indented_dreamon_body(self) -> None:
        """Regression test for the double-indent stitch bug (A05 closeout, 2026-08-12).

        DreamOn emits a body that is ALREADY indented 4 spaces. The old stitch ran
        extract_python() first, whose trailing .strip() de-indents line 1 only; the
        subsequent textwrap.indent() then pushed line 1 to col 4 and line 2 to col 8,
        producing IndentationError. Only the UNINDENTED case was covered by the test
        above, which is why the bug reached 113 of the 117 unparseable HE+ items at
        canvas=128 before anyone noticed.
        """
        prompt = 'def f(xs):\n    """Doc."""\n'
        stitched = DREAMON_MODULE.combine_humaneval_prompt(
            prompt,
            "    total = 0\n    for x in xs:\n        total += x\n    return total\n",
        )
        self.assertEqual(
            stitched,
            'def f(xs):\n'
            '    """Doc."""\n'
            "    total = 0\n"
            "    for x in xs:\n"
            "        total += x\n"
            "    return total\n",
        )
        # the load-bearing invariant: the stitched program must actually parse
        ast.parse(stitched)

    def test_stitch_preserves_relative_indentation_depth(self) -> None:
        """A dedent applied AFTER extract_python would be a no-op -- guard that too."""
        prompt = "def g(n):\n"
        stitched = DREAMON_MODULE.combine_humaneval_prompt(
            prompt,
            "    if n:\n        return 1\n    return 0\n",
        )
        ast.parse(stitched)
        body = stitched[len("def g(n):\n"):]
        # 'if' at 4, 'return 1' at 8, trailing 'return 0' back at 4
        depths = [len(l) - len(l.lstrip()) for l in body.splitlines() if l.strip()]
        self.assertEqual(depths, [4, 8, 4])

    def test_compound_depth_ignores_function_wrapper(self) -> None:
        source = (
            "def f(xs):\n"
            "    for x in xs:\n"
            "        if x:\n"
            "            while False:\n"
            "                pass\n"
        )
        self.assertEqual(
            DEPTH_MODULE.maximum_compound_depth(source),
            3,
        )
        self.assertEqual(DEPTH_MODULE.depth_group(3), "depth_3_plus")

    def test_plus_pass_requires_base_and_extra_tests(self) -> None:
        self.assertTrue(
            DEPTH_MODULE.plus_pass(
                {"base_status": "pass", "plus_status": "pass"}
            )
        )
        self.assertFalse(
            DEPTH_MODULE.plus_pass(
                {"base_status": "fail", "plus_status": "pass"}
            )
        )

    def test_scaffold_failure_reason_classification(self) -> None:
        from scaffold_coder.errors import BudgetExceededError

        self.assertEqual(
            SCAFFOLD_MODULE.termination_reason(
                BudgetExceededError(
                    "generation exceeded 512 model calls"
                )
            ),
            "model_call_budget",
        )
        self.assertEqual(
            SCAFFOLD_MODULE.termination_reason(
                BudgetExceededError("total line budget exceeded")
            ),
            "total_line_capacity_exhausted",
        )

    def test_vanilla_scaffold_logit_suppression(self) -> None:
        import torch

        logits = torch.zeros((1, 2, 5), dtype=torch.float32)
        result = MODULE.suppress_token_logits(logits, (1, 3))
        self.assertEqual(float(result[0, 0, 0]), 0.0)
        self.assertEqual(
            float(result[0, 0, 1]),
            torch.finfo(torch.float32).min,
        )
        self.assertEqual(
            float(result[0, 1, 3]),
            torch.finfo(torch.float32).min,
        )


if __name__ == "__main__":
    unittest.main()
