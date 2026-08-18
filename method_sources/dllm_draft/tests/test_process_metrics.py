from __future__ import annotations

import unittest

from scaffold_coder.model_sampler import ScaffoldGenerationResult
from scaffold_coder.process_metrics import compute_process_metrics


class ProcessMetricTests(unittest.TestCase):
    def test_metric_summary(self) -> None:
        result = ScaffoldGenerationResult(
            text="def f():\n    return 1\n",
            model_calls=3,
            history=(),
            final_canvas_tokens=7,
            expansions=2,
            model_canvas_lengths=(20, 24, 22),
            cumulative_model_tokens=66,
            placeholder_history=(
                "pass\n",
                "def f():\n    pass\n",
                "def f():\n    return 1\n",
            ),
            leaf_remasks=2,
            correction_rounds=1,
            structural_deferrals=3,
            structural_backtracks=1,
            edit_cycle_breaks=4,
            line_capacity_hits=2,
            token_capacity_hits=3,
            depth_capacity_hits=4,
            total_line_capacity_hits=5,
            module_expand_suppressed=6,
            expand_budget_hits=7,
            maximum_tree_depth=8,
            maximum_total_lines=9,
            maximum_body_lines=10,
            maximum_tokens_per_hole=11,
            termination_reason="resolved",
        )
        metrics = compute_process_metrics(result)
        self.assertTrue(metrics.final_parseable)
        self.assertEqual(metrics.placeholder_parse_rate, 1.0)
        self.assertEqual(metrics.minimum_model_canvas_tokens, 20)
        self.assertEqual(metrics.maximum_model_canvas_tokens, 24)
        self.assertEqual(metrics.cumulative_model_tokens, 66)
        self.assertEqual(metrics.leaf_remasks, 2)
        self.assertEqual(metrics.correction_rounds, 1)
        self.assertEqual(metrics.structural_deferrals, 3)
        self.assertEqual(metrics.structural_backtracks, 1)
        self.assertEqual(metrics.edit_cycle_breaks, 4)
        self.assertEqual(metrics.line_capacity_hits, 2)
        self.assertEqual(metrics.total_line_capacity_hits, 5)
        self.assertEqual(metrics.maximum_total_lines, 9)
        self.assertEqual(metrics.termination_reason, "resolved")


if __name__ == "__main__":
    unittest.main()
