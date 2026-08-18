from __future__ import annotations

from collections import deque
from pathlib import Path
import ast
import unittest

import torch
from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.decoder_runtime import DecoderConfig, DecoderRuntime
from scaffold_coder.model_sampler import (
    SamplerConfig,
    ScaffoldModelSampler,
)
from scaffold_coder.roles import (
    DELETE,
    EXPAND,
    FOR,
    FUNC,
    IF,
    MaskRole,
    STMT,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Dream-Coder-v0-Instruct-7B"


@unittest.skipUnless(MODEL_PATH.exists(), "Dream tokenizer is unavailable")
class ModelSamplerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.registry = TokenRegistry.build(tokenizer)

    def resize_script(self, text: str, initial: int):
        ids = self.registry.tokenizer.encode(text, add_special_tokens=False)
        script = []
        if len(ids) > initial:
            script.extend(
                [self.registry.special_id("[expand]")] * (len(ids) - initial)
            )
        script.extend(ids)
        if len(ids) < initial:
            script.extend(
                [self.registry.special_id(DELETE)] * (initial - len(ids))
            )
        return script

    def test_scripted_provider_drives_full_generation_loop(self) -> None:
        script = deque()
        script.append(self.registry.special_id(FUNC))
        script.extend(self.resize_script("f(x)", 4))
        script.append(self.registry.special_id(STMT))
        script.append(self.registry.special_id(DELETE))
        script.extend(self.resize_script("return x", 4))

        def provider(runtime, canvas, refs):
            target = script.popleft()
            return {
                refs[0].mask_id: (target, 1.0),
            }

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=64,
                transfer_tokens=1,
                keep_history=True,
            ),
        )
        result = sampler.generate(
            "Write f.",
            runtime=DecoderRuntime(self.registry),
            prediction_provider=provider,
        )
        self.assertFalse(script)
        self.assertEqual(result.text, "def f(x):\n    return x\n")
        self.assertGreater(result.model_calls, 0)
        self.assertTrue(result.history)
        self.assertEqual(
            len(result.model_canvas_lengths), result.model_calls
        )
        self.assertEqual(
            result.cumulative_model_tokens,
            sum(result.model_canvas_lengths),
        )
        self.assertEqual(
            len(result.placeholder_history), result.model_calls
        )
        for partial in result.placeholder_history:
            ast.parse(partial)
        ast.parse(result.text)

    def test_c1_completion_remasks_low_confidence_leaf(self) -> None:
        wrong_id = self.registry.tokenizer.encode(
            "x", add_special_tokens=False
        )
        corrected_id = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(wrong_id), 1)
        self.assertEqual(len(corrected_id), 1)
        token_attempts = 0

        def provider(runtime, canvas, refs):
            nonlocal token_attempts
            ref = refs[0]
            if ref.role is MaskRole.LINE_MODULE:
                return {
                    ref.mask_id: (self.registry.special_id(STMT), 1.0)
                }
            self.assertIs(ref.role, MaskRole.TOKEN_STMT)
            token_attempts += 1
            if token_attempts == 1:
                return {ref.mask_id: (wrong_id[0], 0.1)}
            return {ref.mask_id: (corrected_id[0], 0.9)}

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=8,
                leaf_remask_fraction=1.0,
                leaf_remask_confidence_threshold=0.5,
                leaf_remask_min_age_calls=1,
                max_leaf_remasks=1,
                max_leaf_remasks_per_token=1,
            ),
        )
        result = sampler.generate(
            "Write one statement.",
            runtime=DecoderRuntime(
                self.registry,
                DecoderConfig(initial_statement_masks=1),
            ),
            prediction_provider=provider,
        )
        self.assertEqual(result.text, "pass\n")
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.leaf_remasks, 1)
        self.assertEqual(result.correction_rounds, 1)
        self.assertEqual(result.structural_deferrals, 0)

    def test_c3_defers_low_confidence_construct_before_expansion(self) -> None:
        pass_id = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(pass_id), 1)
        calls = 0

        def provider(runtime, canvas, refs):
            nonlocal calls
            ref = refs[0]
            calls += 1
            if calls == 1:
                self.assertIs(ref.role, MaskRole.LINE_MODULE)
                return {
                    ref.mask_id: (self.registry.special_id(FUNC), 0.1)
                }
            if calls == 2:
                self.assertIs(ref.role, MaskRole.LINE_MODULE)
                return {
                    ref.mask_id: (self.registry.special_id(STMT), 0.9)
                }
            self.assertIs(ref.role, MaskRole.TOKEN_STMT)
            return {ref.mask_id: (pass_id[0], 1.0)}

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=8,
                structural_confidence_threshold=0.8,
                structural_max_defer_calls=2,
                keep_history=True,
            ),
        )
        result = sampler.generate(
            "Write one statement.",
            runtime=DecoderRuntime(
                self.registry,
                DecoderConfig(initial_statement_masks=1),
            ),
            prediction_provider=provider,
        )
        self.assertEqual(result.text, "pass\n")
        self.assertEqual(result.model_calls, 3)
        self.assertEqual(result.structural_deferrals, 1)
        self.assertNotIn("<|sc_func|>", result.history[0])
        self.assertEqual(result.leaf_remasks, 0)

    def test_c2_collapses_low_confidence_completed_construct(self) -> None:
        header_ids = deque(
            self.registry.tokenizer.encode(
                "f()", add_special_tokens=False
            )
        )
        pass_ids = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(pass_ids), 1)
        root_attempts = 0

        def provider(runtime, canvas, refs):
            nonlocal root_attempts
            ref = refs[0]
            if ref.role is MaskRole.LINE_MODULE:
                root_attempts += 1
                target = (
                    self.registry.special_id(FUNC)
                    if root_attempts == 1
                    else self.registry.special_id(STMT)
                )
                return {ref.mask_id: (target, 0.9)}
            if ref.role is MaskRole.TOKEN_HDR:
                return {ref.mask_id: (header_ids.popleft(), 0.1)}
            if ref.role is MaskRole.LINE_BODY:
                return {
                    ref.mask_id: (self.registry.special_id(STMT), 0.9)
                }
            self.assertIs(ref.role, MaskRole.TOKEN_STMT)
            confidence = 0.1 if root_attempts == 1 else 0.9
            return {ref.mask_id: (pass_ids[0], confidence)}

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=32,
                structural_backtrack_confidence_threshold=0.5,
                structural_backtrack_min_age_calls=1,
                max_structural_backtracks=1,
                max_structural_backtracks_per_anchor=1,
            ),
        )
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(
                initial_function_header_masks=len(header_ids),
                initial_body_slots=1,
                initial_statement_masks=1,
            ),
        )
        result = sampler.generate(
            "Write one line.",
            runtime=runtime,
            prediction_provider=provider,
        )
        self.assertFalse(header_ids)
        self.assertEqual(root_attempts, 2)
        self.assertEqual(result.text, "pass\n")
        self.assertEqual(result.structural_backtracks, 1)
        self.assertEqual(result.correction_rounds, 1)

    def test_c2_preserves_completion_on_final_call_budget(self) -> None:
        header_ids = deque(
            self.registry.tokenizer.encode(
                "f()", add_special_tokens=False
            )
        )
        header_count = len(header_ids)
        pass_ids = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(pass_ids), 1)

        def provider(runtime, canvas, refs):
            ref = refs[0]
            if ref.role is MaskRole.LINE_MODULE:
                return {
                    ref.mask_id: (self.registry.special_id(FUNC), 0.9)
                }
            if ref.role is MaskRole.TOKEN_HDR:
                return {ref.mask_id: (header_ids.popleft(), 0.1)}
            if ref.role is MaskRole.LINE_BODY:
                return {
                    ref.mask_id: (self.registry.special_id(STMT), 0.9)
                }
            return {ref.mask_id: (pass_ids[0], 0.1)}

        exact_calls = header_count + 3
        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=exact_calls,
                structural_backtrack_confidence_threshold=0.5,
                max_structural_backtracks=1,
            ),
        )
        result = sampler.generate(
            "Write f.",
            runtime=DecoderRuntime(
                self.registry,
                DecoderConfig(
                    initial_function_header_masks=header_count,
                    initial_body_slots=1,
                    initial_statement_masks=1,
                ),
            ),
            prediction_provider=provider,
        )
        self.assertEqual(result.model_calls, exact_calls)
        self.assertEqual(result.text, "def f():\n    pass\n")
        self.assertEqual(result.structural_backtracks, 0)

    def test_completion_on_final_allowed_call_is_success(self) -> None:
        def provider(runtime, canvas, refs):
            return {
                refs[0].mask_id: (self.registry.special_id(DELETE), 1.0)
            }

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(max_model_calls=1),
        )
        result = sampler.generate(
            "Write an empty module.",
            runtime=DecoderRuntime(self.registry),
            prediction_provider=provider,
        )
        self.assertEqual(result.text, "")
        self.assertEqual(result.model_calls, 1)

    def test_failure_metrics_capture_partial_model_cost(self) -> None:
        calls = 0

        def provider(runtime, canvas, refs):
            nonlocal calls
            calls += 1
            if calls == 1:
                return {
                    refs[0].mask_id: (
                        self.registry.special_id(FUNC),
                        1.0,
                    )
                }
            if refs[0].role is MaskRole.TOKEN_HDR:
                predictions = {
                    ref.mask_id: (
                        self.registry.special_id(DELETE),
                        1.0,
                    )
                    for ref in refs[1:]
                }
                predictions[refs[0].mask_id] = (
                    self.registry.tokenizer.encode(
                        "f", add_special_tokens=False
                    )[0],
                    1.0,
                )
                return predictions
            return {
                refs[0].mask_id: (
                    self.registry.special_id(FOR),
                    1.0,
                )
            }

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(max_model_calls=8),
        )
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(
                initial_function_header_masks=1,
                initial_body_slots=1,
                max_tree_depth=0 + 1,
            ),
        )
        with self.assertRaisesRegex(
            Exception, "tree depth budget exceeded"
        ):
            sampler.generate(
                "Write nested code.",
                runtime=runtime,
                prediction_provider=provider,
            )

        failure = sampler.last_failure_metrics
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.model_calls, calls)
        self.assertLess(failure.model_calls, sampler.config.max_model_calls)
        self.assertEqual(
            len(failure.model_canvas_lengths),
            failure.model_calls,
        )
        self.assertEqual(
            failure.cumulative_model_tokens,
            sum(failure.model_canvas_lengths),
        )
        self.assertEqual(failure.depth_capacity_hits, 1)
        self.assertGreater(failure.maximum_tree_depth, 0)

    def test_failure_metrics_capture_model_call_budget(self) -> None:
        def provider(runtime, canvas, refs):
            return {
                refs[0].mask_id: (
                    self.registry.special_id(EXPAND),
                    1.0,
                )
            }

        sampler = ScaffoldModelSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(
                max_model_calls=2,
                break_edit_cycles=False,
            ),
        )
        with self.assertRaisesRegex(
            Exception, "generation exceeded 2 model calls"
        ):
            sampler.generate(
                "Keep expanding.",
                runtime=DecoderRuntime(
                    self.registry,
                    DecoderConfig(
                        initial_root_slots=1,
                        max_lines_per_body=8,
                    ),
                ),
                prediction_provider=provider,
            )
        failure = sampler.last_failure_metrics
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure.model_calls, 2)
        self.assertEqual(len(failure.model_canvas_lengths), 2)

    def test_repeated_edit_state_suppresses_expand_delete(self) -> None:
        pass_id = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(pass_id), 1)

        class CyclingSampler(ScaffoldModelSampler):
            def _model_predictions(
                inner_self,
                prompt_ids,
                canvas,
                refs,
                *,
                suppress_edits=False,
            ):
                if refs[0].role is MaskRole.LINE_MODULE:
                    return {
                        refs[0].mask_id: (
                            self.registry.special_id(STMT),
                            1.0,
                        )
                    }
                if len(refs) == 1:
                    target = (
                        pass_id[0]
                        if suppress_edits
                        else self.registry.special_id(EXPAND)
                    )
                    return {refs[0].mask_id: (target, 1.0)}
                return {
                    refs[0].mask_id: (
                        self.registry.special_id(DELETE),
                        1.0,
                    ),
                    refs[1].mask_id: (pass_id[0], 0.1),
                }

        sampler = CyclingSampler(
            model=None,
            registry=self.registry,
            config=SamplerConfig(max_model_calls=8),
        )
        result = sampler.generate(
            "Write one statement.",
            runtime=DecoderRuntime(
                self.registry,
                DecoderConfig(initial_statement_masks=1),
            ),
        )
        self.assertEqual(result.text, "pass\n")
        self.assertEqual(result.model_calls, 4)
        self.assertEqual(result.expansions, 1)
        self.assertEqual(result.edit_cycle_breaks, 1)

    def test_suppressed_token_support_excludes_edits(self) -> None:
        sampler = ScaffoldModelSampler(None, self.registry)
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_statement_masks=1),
        )
        runtime.commit(
            runtime.unresolved_masks()[0].mask_id,
            self.registry.special_id(STMT),
        )
        ref = runtime.unresolved_masks()[0]
        allowed = set(
            sampler._allowed_ids(
                ref,
                torch.device("cpu"),
                suppress_edits=True,
            ).tolist()
        )
        self.assertNotIn(self.registry.special_id(EXPAND), allowed)
        self.assertNotIn(self.registry.special_id(DELETE), allowed)
        self.assertIn(
            self.registry.tokenizer.encode(
                "pass", add_special_tokens=False
            )[0],
            allowed,
        )

    def test_body_construct_penalty_does_not_touch_module_plan(self) -> None:
        sampler = ScaffoldModelSampler(
            None,
            self.registry,
            SamplerConfig(body_construct_logit_penalty=2.5),
        )
        allowed = torch.tensor(
            [
                self.registry.special_id(STMT),
                self.registry.special_id(FOR),
                self.registry.special_id(IF),
            ]
        )
        logits = torch.tensor([1.0, 4.0, 3.0])
        body_ref = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_body_slots=1),
        )
        body_ref.commit(
            body_ref.unresolved_masks()[0].mask_id,
            self.registry.special_id(FUNC),
        )
        header = body_ref.unresolved_masks()[0]
        body_ref.commit(
            header.mask_id,
            self.registry.tokenizer.encode(
                "f", add_special_tokens=False
            )[0],
        )
        while body_ref.unresolved_masks()[0].role is MaskRole.TOKEN_HDR:
            body_ref.commit(
                body_ref.unresolved_masks()[0].mask_id,
                self.registry.special_id(DELETE),
            )
        nested_ref = body_ref.unresolved_masks()[0]
        self.assertIs(nested_ref.role, MaskRole.LINE_BODY)
        penalized = sampler._apply_body_construct_penalty(
            nested_ref,
            allowed,
            logits,
        )
        self.assertTrue(
            torch.equal(penalized, torch.tensor([1.0, 1.5, 0.5]))
        )

        module_ref = DecoderRuntime(self.registry).unresolved_masks()[0]
        unchanged = sampler._apply_body_construct_penalty(
            module_ref,
            allowed,
            logits,
        )
        self.assertTrue(torch.equal(unchanged, logits))

    def test_body_stmt_bonus_only_changes_statement_in_body(self) -> None:
        sampler = ScaffoldModelSampler(
            None,
            self.registry,
            SamplerConfig(body_stmt_logit_bonus=2.5),
        )
        allowed = torch.tensor(
            [
                self.registry.special_id(STMT),
                self.registry.special_id(FOR),
                self.registry.special_id(IF),
            ]
        )
        logits = torch.tensor([1.0, 4.0, 3.0])
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_body_slots=1),
        )
        runtime.commit(
            runtime.unresolved_masks()[0].mask_id,
            self.registry.special_id(FUNC),
        )
        runtime.commit(
            runtime.unresolved_masks()[0].mask_id,
            self.registry.tokenizer.encode(
                "f", add_special_tokens=False
            )[0],
        )
        while runtime.unresolved_masks()[0].role is MaskRole.TOKEN_HDR:
            runtime.commit(
                runtime.unresolved_masks()[0].mask_id,
                self.registry.special_id(DELETE),
            )
        body_ref = runtime.unresolved_masks()[0]
        boosted = sampler._apply_body_stmt_bonus(
            body_ref,
            allowed,
            logits,
        )
        self.assertTrue(
            torch.equal(boosted, torch.tensor([3.5, 4.0, 3.0]))
        )

        module_ref = DecoderRuntime(self.registry).unresolved_masks()[0]
        unchanged = sampler._apply_body_stmt_bonus(
            module_ref,
            allowed,
            logits,
        )
        self.assertTrue(torch.equal(unchanged, logits))

    def test_token_expand_bonus_does_not_change_line_edits(self) -> None:
        sampler = ScaffoldModelSampler(
            None,
            self.registry,
            SamplerConfig(token_expand_logit_bonus=2.5),
        )
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_statement_masks=1),
        )
        line_ref = runtime.unresolved_masks()[0]
        runtime.commit(
            line_ref.mask_id,
            self.registry.special_id(STMT),
        )
        token_ref = runtime.unresolved_masks()[0]
        allowed = torch.tensor(
            [
                self.registry.special_id(EXPAND),
                self.registry.tokenizer.encode(
                    "pass", add_special_tokens=False
                )[0],
            ]
        )
        logits = torch.tensor([1.0, 3.0])
        boosted = sampler._apply_token_expand_bonus(
            token_ref,
            allowed,
            logits,
        )
        self.assertTrue(torch.equal(boosted, torch.tensor([3.5, 3.0])))

        unchanged = sampler._apply_token_expand_bonus(
            line_ref,
            allowed,
            logits,
        )
        self.assertTrue(torch.equal(unchanged, logits))


if __name__ == "__main__":
    unittest.main()
