from __future__ import annotations

from pathlib import Path
import textwrap
import unittest

from transformers import AutoTokenizer

from scaffold_coder.canvas import (
    LineEditConfig,
    TokenEditConfig,
    TokenRegistry,
    build_body_plan,
    build_leaf_infill,
    build_root_plan,
    build_template_skeleton,
    iter_main_bodies,
    prepend_chat_prompt,
)
from scaffold_coder.parser import parse_source
from scaffold_coder.roles import DELETE, EXPAND, FOR, FUNC, MaskRole, STMT


MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "Dream-Coder-v0-Instruct-7B"


@unittest.skipUnless(MODEL_PATH.exists(), "Dream tokenizer is not available")
class CanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.registry = TokenRegistry.build(tokenizer)
        cls.module = parse_source(
            textwrap.dedent(
                """\
                def count_pairs(nums, target):
                    count = 0
                    for i in range(len(nums)):
                        if nums[i] == target:
                            count += 1
                    return count
                """
            )
        )

    def test_root_plan_predicts_one_function(self) -> None:
        state = build_root_plan(self.module, self.registry)
        state.validate(self.registry)
        positions = [i for i, value in enumerate(state.loss_mask) if value]
        self.assertEqual(len(positions), 1)
        index = positions[0]
        self.assertEqual(state.roles[index], MaskRole.LINE_MODULE)
        self.assertEqual(
            state.labels[index], self.registry.special_id(FUNC)
        )

    def test_template_skeleton_contains_rule_only_holes_and_stmt(self) -> None:
        state = build_template_skeleton(self.module, self.registry)
        decoded = self.registry.tokenizer.decode(state.input_ids)
        self.assertFalse(any(state.loss_mask))
        self.assertIn("<|sc_hdr|>", decoded)
        self.assertIn("<|sc_stmt|>", decoded)
        self.assertIn("<|sc_clauses|>", decoded)

    def test_leaf_infill_masks_only_content_roles(self) -> None:
        state = build_leaf_infill(
            self.module,
            self.registry,
            mask_probability=1.0,
            seed=1,
        )
        state.validate(self.registry)
        self.assertTrue(any(state.loss_mask))
        for role, supervised in zip(state.roles, state.loss_mask, strict=True):
            if supervised:
                self.assertIn(
                    role, {MaskRole.TOKEN_HDR, MaskRole.TOKEN_STMT}
                )
        decoded = self.registry.tokenizer.decode(state.input_ids)
        self.assertIn("def ", decoded)
        self.assertIn(":\n", decoded)

    def test_function_body_plan_targets_stmt_for_stmt_for(self) -> None:
        bodies = list(iter_main_bodies(self.module))
        function_body = bodies[1]
        state = build_body_plan(
            self.module,
            self.registry,
            target_body_id=function_body.body_id,
        )
        targets = [
            self.registry.id_to_notation[label]
            for label, supervised in zip(
                state.labels, state.loss_mask, strict=True
            )
            if supervised
        ]
        self.assertEqual(targets, [STMT, FOR, STMT])
        decoded = self.registry.tokenizer.decode(state.input_ids)
        self.assertIn("def count_pairs(nums, target):\n", decoded)

    def test_prompt_prepend_preserves_response_targets(self) -> None:
        response = build_root_plan(self.module, self.registry)
        combined = prepend_chat_prompt(
            response, self.registry, "Write a pair-counting function."
        )
        combined.validate(self.registry)
        self.assertEqual(sum(combined.loss_mask), sum(response.loss_mask))
        tensors = combined.to_tensors()
        self.assertEqual(tensors["input_ids"].shape, tensors["labels"].shape)

    def test_line_level_merge_and_delete_targets(self) -> None:
        three_lines = parse_source("x = 1\ny = 2\nz = 3\n")
        merged = build_root_plan(
            three_lines,
            self.registry,
            edit_config=LineEditConfig(merge_probability=1.0),
            seed=0,
        )
        merged_targets = [
            self.registry.id_to_notation[label]
            for label, supervised in zip(
                merged.labels, merged.loss_mask, strict=True
            )
            if supervised
        ]
        self.assertEqual(merged_targets, [EXPAND, STMT])

        one_line = parse_source("x = 1\n")
        with_delete = build_root_plan(
            one_line,
            self.registry,
            edit_config=LineEditConfig(max_delete=1),
            seed=0,
        )
        delete_targets = [
            self.registry.id_to_notation[label]
            for label, supervised in zip(
                with_delete.labels, with_delete.loss_mask, strict=True
            )
            if supervised
        ]
        self.assertEqual(delete_targets, [STMT, DELETE])

    def test_token_level_merge_and_delete_targets(self) -> None:
        module = parse_source("value = alpha + beta\n")
        found_delete = False
        for seed in range(20):
            state = build_leaf_infill(
                module,
                self.registry,
                mask_probability=1.0,
                seed=seed,
                edit_config=TokenEditConfig(
                    merge_probability=1.0,
                    max_delete=1,
                ),
            )
            targets = [
                self.registry.id_to_notation.get(label)
                for label, supervised in zip(
                    state.labels, state.loss_mask, strict=True
                )
                if supervised
            ]
            self.assertIn(EXPAND, targets)
            found_delete = found_delete or DELETE in targets
        self.assertTrue(found_delete)

    def test_partial_line_clock_keeps_some_labels_visible(self) -> None:
        module = parse_source("a = 1\nb = 2\nc = 3\nd = 4\n")
        state = build_root_plan(
            module,
            self.registry,
            line_mask_probability=0.5,
            seed=9,
        )
        line_positions = [
            index
            for index, role in enumerate(state.roles)
            if role is MaskRole.LINE_MODULE
        ]
        self.assertTrue(any(state.loss_mask[index] for index in line_positions))
        self.assertTrue(
            any(not state.loss_mask[index] for index in line_positions)
        )

    def test_depth_specific_leaf_clocks(self) -> None:
        state = build_leaf_infill(
            self.module,
            self.registry,
            mask_probability=0.0,
            depth_probabilities={0: 0.0, 1: 1.0, 2: 1.0, 3: 1.0},
            seed=1,
        )
        header_positions = [
            index
            for index, role in enumerate(state.roles)
            if role is MaskRole.TOKEN_HDR
        ]
        # Top-level function header remains clean; nested loop/if headers mask.
        self.assertTrue(any(not state.loss_mask[i] for i in header_positions))
        self.assertTrue(any(state.loss_mask[i] for i in header_positions))

    def test_fully_masked_regions_collapse_to_typed_markers(self) -> None:
        module = parse_source(
            "def f(x):\n"
            "    y = x + 1\n"
            "    return y\n"
        )
        state = build_leaf_infill(
            module,
            self.registry,
            mask_probability=1.0,
            seed=4,
            edit_config=TokenEditConfig(
                collapse_fully_masked=True,
            ),
        )
        decoded = self.registry.tokenizer.decode(state.input_ids)
        self.assertIn("<|sc_hdr|>", decoded)
        self.assertIn("<|sc_stmt|>", decoded)
        self.assertTrue(any(state.loss_mask))

    def test_coupled_region_collapse_is_not_length_dependent(self) -> None:
        short = parse_source("x = 1\n")
        long = parse_source(
            "result = alpha + beta + gamma + delta + epsilon + zeta\n"
        )

        def rate(module):
            collapsed = 0
            for seed in range(500):
                state = build_leaf_infill(
                    module,
                    self.registry,
                    mask_probability=0.4,
                    seed=seed,
                    edit_config=TokenEditConfig(
                        coupled_collapse_exponent=1.0
                    ),
                )
                collapsed += any(
                    self.registry.id_to_notation.get(label) == STMT
                    and role
                    in {MaskRole.LINE_MODULE, MaskRole.LINE_BODY}
                    for label, role in zip(
                        state.labels, state.roles, strict=True
                    )
                )
            return collapsed / 500

        self.assertLess(abs(rate(short) - rate(long)), 0.1)


if __name__ == "__main__":
    unittest.main()
