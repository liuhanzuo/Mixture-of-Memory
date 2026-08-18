from __future__ import annotations

from pathlib import Path
import ast
import unittest

from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.decoder_runtime import DecoderConfig, DecoderRuntime
from scaffold_coder.errors import RuntimeInvariantError
from scaffold_coder.errors import BudgetExceededError
from scaffold_coder.roles import (
    DELETE,
    ELIF,
    ELSE,
    EXPAND,
    FUNC,
    IF,
    MaskRole,
    STMT,
)


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "Dream-Coder-v0-Instruct-7B"


@unittest.skipUnless(MODEL_PATH.exists(), "Dream tokenizer is unavailable")
class DecoderRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            trust_remote_code=True,
            local_files_only=True,
        )
        cls.registry = TokenRegistry.build(tokenizer)

    def refs(self, runtime, role):
        return [
            ref for ref in runtime.unresolved_masks() if ref.role is role
        ]

    def fill_only_token_hole(
        self,
        runtime,
        role,
        text,
        *,
        confidence=None,
        model_call=None,
    ):
        target_ids = self.registry.tokenizer.encode(
            text, add_special_tokens=False
        )
        refs = self.refs(runtime, role)
        owners = {ref.owner_id for ref in refs}
        self.assertEqual(len(owners), 1, owners)
        while len(refs) < len(target_ids):
            runtime.commit(refs[0].mask_id, self.registry.special_id(EXPAND))
            refs = self.refs(runtime, role)
        while len(refs) > len(target_ids):
            runtime.commit(refs[-1].mask_id, self.registry.special_id(DELETE))
            refs = self.refs(runtime, role)
        runtime.commit_many(
            {
                ref.mask_id: token_id
                for ref, token_id in zip(refs, target_ids, strict=True)
            },
            confidences=(
                {
                    ref.mask_id: confidence
                    for ref in refs
                }
                if confidence is not None
                else None
            ),
            model_call=model_call,
        )

    def test_generate_simple_function(self) -> None:
        runtime = DecoderRuntime(self.registry)
        root = self.refs(runtime, MaskRole.LINE_MODULE)
        self.assertEqual(len(root), 1)
        runtime.commit(root[0].mask_id, self.registry.special_id(FUNC))
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "f(x)")

        body = self.refs(runtime, MaskRole.LINE_BODY)
        self.assertEqual(len(body), 2)
        runtime.commit(body[-1].mask_id, self.registry.special_id(DELETE))
        body = self.refs(runtime, MaskRole.LINE_BODY)
        runtime.commit(body[0].mask_id, self.registry.special_id(STMT))
        self.fill_only_token_hole(
            runtime, MaskRole.TOKEN_STMT, "return x"
        )

        self.assertTrue(runtime.complete)
        text = runtime.final_text()
        self.assertEqual(text, "def f(x):\n    return x\n")
        ast.parse(text)

    def test_seeded_function_header_opens_body_without_header_masks(self) -> None:
        runtime = DecoderRuntime.from_function_header(
            self.registry,
            "f(x)",
        )
        self.assertFalse(self.refs(runtime, MaskRole.LINE_MODULE))
        self.assertFalse(self.refs(runtime, MaskRole.TOKEN_HDR))
        body = self.refs(runtime, MaskRole.LINE_BODY)
        self.assertEqual(len(body), 2)
        runtime.commit(
            body[-1].mask_id,
            self.registry.special_id(DELETE),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_BODY)[0].mask_id,
            self.registry.special_id(STMT),
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_STMT,
            "return x",
        )
        self.assertEqual(runtime.final_text(), "def f(x):\n    return x\n")

    def test_local_body_barrier_holds_pending_labels(self) -> None:
        runtime = DecoderRuntime(
            self.registry, DecoderConfig(initial_root_slots=2)
        )
        root = self.refs(runtime, MaskRole.LINE_MODULE)
        runtime.commit(root[0].mask_id, self.registry.special_id(FUNC))
        self.assertFalse(self.refs(runtime, MaskRole.TOKEN_HDR))
        self.assertIn(
            "<|sc_func|>",
            self.registry.tokenizer.decode(runtime.render().input_ids),
        )
        remaining = self.refs(runtime, MaskRole.LINE_MODULE)
        runtime.commit(
            remaining[0].mask_id, self.registry.special_id(DELETE)
        )
        self.assertTrue(self.refs(runtime, MaskRole.TOKEN_HDR))

    def test_line_expand_and_required_empty_body_pass(self) -> None:
        runtime = DecoderRuntime(self.registry)
        root = self.refs(runtime, MaskRole.LINE_MODULE)
        runtime.commit(root[0].mask_id, self.registry.special_id(EXPAND))
        self.assertEqual(len(self.refs(runtime, MaskRole.LINE_MODULE)), 2)

        # Reset and make a function whose required body deletes every slot.
        runtime = DecoderRuntime(self.registry)
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(FUNC),
        )
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "f()")
        for ref in list(self.refs(runtime, MaskRole.LINE_BODY)):
            runtime.commit(ref.mask_id, self.registry.special_id(DELETE))
        self.assertTrue(runtime.complete)
        self.assertEqual(runtime.final_text(), "def f():\n    pass\n")
        self.assertFalse(runtime.committed_leaf_tokens())

    def test_if_clause_delete_and_constraints(self) -> None:
        runtime = DecoderRuntime(self.registry)
        root = self.refs(runtime, MaskRole.LINE_MODULE)[0]
        with self.assertRaises(RuntimeInvariantError):
            runtime.commit(
                root.mask_id,
                self.registry.tokenizer.encode(
                    "ordinary", add_special_tokens=False
                )[0],
            )

        runtime.commit(root.mask_id, self.registry.special_id(IF))
        with self.assertRaises(RuntimeInvariantError):
            runtime.commit(
                self.refs(runtime, MaskRole.TOKEN_HDR)[0].mask_id,
                self.registry.special_id(FUNC),
            )
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "x")
        clause = self.refs(runtime, MaskRole.LINE_CLAUSE)
        self.assertEqual(len(clause), 1)
        runtime.commit(clause[0].mask_id, self.registry.special_id(DELETE))
        body = self.refs(runtime, MaskRole.LINE_BODY)
        runtime.commit(body[-1].mask_id, self.registry.special_id(DELETE))
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_BODY)[0].mask_id,
            self.registry.special_id(STMT),
        )
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_STMT, "pass")
        self.assertTrue(runtime.complete)
        self.assertEqual(runtime.final_text(), "if x:\n    pass\n")

    def test_elif_else_clause_chain(self) -> None:
        runtime = DecoderRuntime(self.registry)
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(IF),
        )
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "x")

        # Resolve the main body to rule-emitted pass.
        for ref in list(self.refs(runtime, MaskRole.LINE_BODY)):
            runtime.commit(ref.mask_id, self.registry.special_id(DELETE))
        clause = self.refs(runtime, MaskRole.LINE_CLAUSE)[0]
        runtime.commit(clause.mask_id, self.registry.special_id(ELIF))
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "y")
        for ref in list(self.refs(runtime, MaskRole.LINE_BODY)):
            runtime.commit(ref.mask_id, self.registry.special_id(DELETE))
        next_clause = self.refs(runtime, MaskRole.LINE_CLAUSE)[0]
        runtime.commit(next_clause.mask_id, self.registry.special_id(ELSE))
        for ref in list(self.refs(runtime, MaskRole.LINE_BODY)):
            runtime.commit(ref.mask_id, self.registry.special_id(DELETE))

        self.assertTrue(runtime.complete)
        text = runtime.final_text()
        self.assertEqual(
            text,
            "if x:\n"
            "    pass\n"
            "elif y:\n"
            "    pass\n"
            "else:\n"
            "    pass\n",
        )
        ast.parse(text)

    def test_expansion_budget_is_enforced(self) -> None:
        runtime = DecoderRuntime(
            self.registry, DecoderConfig(max_expansions=0)
        )
        self.assertNotIn(
            EXPAND,
            self.refs(runtime, MaskRole.LINE_MODULE)[0].allowed_notations,
        )
        with self.assertRaises(BudgetExceededError):
            runtime.commit(
                self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
                self.registry.special_id(EXPAND),
            )

    def test_module_expand_can_be_disabled(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(allow_module_expand=False),
        )
        root = self.refs(runtime, MaskRole.LINE_MODULE)[0]
        self.assertNotIn(EXPAND, root.allowed_notations)
        with self.assertRaises(RuntimeInvariantError):
            runtime.commit(
                root.mask_id,
                self.registry.special_id(EXPAND),
            )
        self.assertEqual(runtime.module_expand_suppressed, 1)

    def test_global_line_budget_is_enforced_and_instrumented(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(
                initial_root_slots=1,
                max_total_lines=1,
            ),
        )
        root = self.refs(runtime, MaskRole.LINE_MODULE)[0]
        self.assertNotIn(EXPAND, root.allowed_notations)
        with self.assertRaises(BudgetExceededError):
            runtime.commit(
                root.mask_id,
                self.registry.special_id(EXPAND),
            )
        self.assertEqual(runtime.total_line_capacity_hits, 1)

    def test_capacity_metrics_measure_runtime_shape(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(
                initial_root_slots=1,
                initial_body_slots=1,
            ),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(FUNC),
        )
        self.fill_only_token_hole(runtime, MaskRole.TOKEN_HDR, "f()")
        metrics = runtime.capacity_metrics()
        self.assertEqual(metrics["maximum_total_lines"], 2)
        self.assertEqual(metrics["maximum_tree_depth"], 1)
        self.assertEqual(metrics["maximum_body_lines"], 1)
        self.assertGreaterEqual(metrics["maximum_tokens_per_hole"], 1)

    def test_required_final_token_does_not_offer_delete(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_statement_masks=1),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(STMT),
        )
        ref = self.refs(runtime, MaskRole.TOKEN_STMT)[0]
        self.assertIn(EXPAND, ref.allowed_notations)
        self.assertNotIn(DELETE, ref.allowed_notations)

    def test_shallow_statement_masks_override_default_length(self) -> None:
        runtime = DecoderRuntime.from_function_header(
            self.registry,
            "f()",
            DecoderConfig(
                initial_body_slots=1,
                initial_statement_masks=4,
                initial_statement_masks_shallow=12,
                statement_shallow_depth=1,
            ),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_BODY)[0].mask_id,
            self.registry.special_id(STMT),
        )
        self.assertEqual(
            len(self.refs(runtime, MaskRole.TOKEN_STMT)),
            12,
        )

    def test_committed_leaf_can_be_remasked_with_provenance(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_statement_masks=1),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(STMT),
            confidence=0.9,
            model_call=0,
        )
        token_id = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        self.assertEqual(len(token_id), 1)
        token_ref = self.refs(runtime, MaskRole.TOKEN_STMT)[0]
        runtime.commit(
            token_ref.mask_id,
            token_id[0],
            confidence=0.2,
            model_call=1,
        )

        leaves = runtime.committed_leaf_tokens()
        self.assertEqual(len(leaves), 1)
        self.assertEqual(leaves[0].confidence, 0.2)
        self.assertEqual(leaves[0].committed_at_call, 1)
        new_mask = runtime.remask_leaf(leaves[0].cell_id)
        self.assertNotEqual(new_mask, token_ref.mask_id)
        self.assertFalse(runtime.complete)
        remasked = self.refs(runtime, MaskRole.TOKEN_STMT)
        self.assertEqual([ref.mask_id for ref in remasked], [new_mask])

        runtime.commit(
            new_mask,
            token_id[0],
            confidence=0.8,
            model_call=2,
        )
        corrected = runtime.committed_leaf_tokens()
        self.assertEqual(len(corrected), 1)
        self.assertEqual(corrected[0].remask_count, 1)
        self.assertEqual(corrected[0].confidence, 0.8)
        self.assertTrue(runtime.complete)

    def test_completed_construct_can_collapse_back_to_line_mask(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_body_slots=1),
        )
        root = self.refs(runtime, MaskRole.LINE_MODULE)[0]
        runtime.commit(
            root.mask_id,
            self.registry.special_id(FUNC),
            confidence=0.9,
            model_call=0,
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_HDR,
            "f()",
            confidence=0.2,
            model_call=1,
        )
        body = self.refs(runtime, MaskRole.LINE_BODY)[0]
        runtime.commit(
            body.mask_id,
            self.registry.special_id(STMT),
            confidence=0.8,
            model_call=2,
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_STMT,
            "pass",
            confidence=0.1,
            model_call=3,
        )
        self.assertTrue(runtime.complete)
        candidates = runtime.completed_structural_subtrees()
        self.assertEqual(len(candidates), 1)
        candidate = candidates[0]
        self.assertEqual(candidate.anchor_type, "line")
        self.assertEqual(candidate.kind, FUNC)
        self.assertLess(candidate.mean_content_confidence, 0.2)

        mask_id = runtime.backtrack_structural_subtree(
            candidate.anchor_id
        )
        self.assertFalse(runtime.complete)
        root_refs = self.refs(runtime, MaskRole.LINE_MODULE)
        self.assertEqual([ref.mask_id for ref in root_refs], [mask_id])
        self.assertFalse(self.refs(runtime, MaskRole.TOKEN_HDR))
        self.assertFalse(self.refs(runtime, MaskRole.TOKEN_STMT))

        runtime.commit(
            mask_id,
            self.registry.special_id(STMT),
            confidence=0.9,
            model_call=4,
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_STMT,
            "pass",
            confidence=0.9,
            model_call=5,
        )
        self.assertEqual(runtime.final_text(), "pass\n")

    def test_completed_clause_can_collapse_back_to_clause_mask(self) -> None:
        runtime = DecoderRuntime(
            self.registry,
            DecoderConfig(initial_body_slots=1),
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_MODULE)[0].mask_id,
            self.registry.special_id(IF),
            confidence=0.9,
            model_call=0,
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_HDR,
            "True",
            confidence=0.8,
            model_call=1,
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_BODY)[0].mask_id,
            self.registry.special_id(DELETE),
            confidence=0.9,
            model_call=2,
        )
        clause_mask = self.refs(runtime, MaskRole.LINE_CLAUSE)[0]
        runtime.commit(
            clause_mask.mask_id,
            self.registry.special_id(ELSE),
            confidence=0.9,
            model_call=3,
        )
        runtime.commit(
            self.refs(runtime, MaskRole.LINE_BODY)[0].mask_id,
            self.registry.special_id(STMT),
            confidence=0.9,
            model_call=4,
        )
        self.fill_only_token_hole(
            runtime,
            MaskRole.TOKEN_STMT,
            "pass",
            confidence=0.1,
            model_call=5,
        )
        clause = next(
            ref
            for ref in runtime.completed_structural_subtrees()
            if ref.anchor_type == "clause"
        )
        new_mask = runtime.backtrack_structural_subtree(clause.anchor_id)
        refs = self.refs(runtime, MaskRole.LINE_CLAUSE)
        self.assertEqual([ref.mask_id for ref in refs], [new_mask])
        runtime.commit(
            new_mask,
            self.registry.special_id(DELETE),
            confidence=0.9,
            model_call=6,
        )
        self.assertEqual(runtime.final_text(), "if True:\n    pass\n")


if __name__ == "__main__":
    unittest.main()
