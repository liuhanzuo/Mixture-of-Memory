from __future__ import annotations

import ast
import textwrap
import unittest

from scaffold_coder.errors import BudgetExceededError
from scaffold_coder.oracle import EventKind, OracleConfig, OracleRuntime
from scaffold_coder.parser import parse_source
from scaffold_coder.roles import (
    BODY,
    CLAUSES,
    DOC,
    HDR,
    DELETE,
    ELIF,
    ELSE,
    EXPAND,
    MaskRole,
    RULE_ONLY_HOLES,
    STMT,
)


class OracleRuntimeTests(unittest.TestCase):
    def test_oracle_reaches_clean_program_and_single_token_targets(self) -> None:
        source = textwrap.dedent(
            """\
            def f(xs):
                total = 0
                for x in xs:
                    if x > 0:
                        total += x
                    elif x == 0:
                        continue
                    else:
                        total -= x
                return total
            """
        )
        module = parse_source(source)
        result = OracleRuntime().generate(module)
        result.validate()
        self.assertEqual(result.rendered_code, source)
        ast.parse(result.rendered_code)
        self.assertTrue(result.predictions)
        self.assertFalse(
            any(target.token in RULE_ONLY_HOLES for target in result.predictions)
        )
        self.assertEqual(
            len(result.predictions),
            sum(event.kind is EventKind.PREDICT for event in result.events),
        )

    def test_body_plan_finishes_before_children_open(self) -> None:
        module = parse_source(
            "def f(x):\n"
            "    y = x + 1\n"
            "    if y:\n"
            "        return y\n"
            "    return 0\n"
        )
        result = OracleRuntime().generate(module)
        events = list(result.events)
        for index, event in enumerate(events):
            if event.action == "BODY_PLAN_COMPLETE":
                owner = event.node_id
                later_open = [
                    item
                    for item in events[index + 1 :]
                    if item.action in {"OPEN_STMT", "EXPAND_IF", "EXPAND_FUNC"}
                ]
                # At least one child is opened after a completed local plan.
                if later_open:
                    self.assertGreater(later_open[0].index, event.index)
                    break
        else:
            self.fail("no body planning event found")

    def test_line_and_token_edit_roles(self) -> None:
        module = parse_source(
            "def f(a, b, c, d, e, f):\n"
            "    x = a + b + c + d + e + f\n"
            "    return x\n"
        )
        config = OracleConfig(
            initial_module_slots=4,
            initial_body_slots=1,
            initial_token_slots=1,
        )
        result = OracleRuntime(config).generate(module)
        line_edits = [
            target
            for target in result.predictions
            if target.token in {EXPAND, DELETE}
            and target.role in {MaskRole.LINE_MODULE, MaskRole.LINE_BODY}
        ]
        token_edits = [
            target
            for target in result.predictions
            if target.token in {EXPAND, DELETE}
            and target.role in {MaskRole.TOKEN_HDR, MaskRole.TOKEN_STMT}
        ]
        self.assertTrue(line_edits)
        self.assertTrue(token_edits)
        for target in result.predictions:
            target.validate()

    def test_clause_order_is_if_elif_else(self) -> None:
        source = textwrap.dedent(
            """\
            if x == 1:
                y = 1
            elif x == 2:
                y = 2
            elif x == 3:
                y = 3
            else:
                y = 4
            """
        )
        result = OracleRuntime().generate(parse_source(source))
        clause_targets = [
            target.token
            for target in result.predictions
            if target.role is MaskRole.LINE_CLAUSE
        ]
        self.assertEqual(clause_targets, [ELIF, ELIF, ELSE])

    def test_absent_clauses_predict_delete(self) -> None:
        source = "if ready:\n    run()\n"
        result = OracleRuntime().generate(parse_source(source))
        clause_targets = [
            target.token
            for target in result.predictions
            if target.role is MaskRole.LINE_CLAUSE
        ]
        self.assertEqual(clause_targets, [DELETE])

    def test_no_rule_only_hole_is_ever_predicted(self) -> None:
        source = "while x:\n    x -= 1\nelse:\n    done()\n"
        result = OracleRuntime().generate(parse_source(source))
        forbidden = {HDR, DOC, BODY, CLAUSES}
        self.assertFalse(
            forbidden.intersection(target.token for target in result.predictions)
        )
        self.assertIn(STMT, {target.token for target in result.predictions})

    def test_empty_module_deletes_initial_root_slots_and_terminates(self) -> None:
        result = OracleRuntime(OracleConfig(initial_module_slots=2)).generate(
            parse_source("")
        )
        self.assertEqual(result.rendered_code, "\n")
        ast.parse(result.rendered_code)
        root_deletes = [
            target
            for target in result.predictions
            if target.role is MaskRole.LINE_MODULE and target.token == DELETE
        ]
        self.assertEqual(len(root_deletes), 2)

    def test_budget_is_a_hard_failure_not_an_infinite_loop(self) -> None:
        runtime = OracleRuntime(OracleConfig(max_predictions=1))
        with self.assertRaises(BudgetExceededError):
            runtime.generate(parse_source("def f(x):\n    return x\n"))

    def test_event_indices_and_mask_ids_are_unique(self) -> None:
        result = OracleRuntime().generate(
            parse_source("def f(x):\n    if x:\n        return x\n    return 0\n")
        )
        self.assertEqual(
            [event.index for event in result.events],
            list(range(len(result.events))),
        )
        mask_ids = [target.mask_id for target in result.predictions]
        self.assertEqual(len(mask_ids), len(set(mask_ids)))


if __name__ == "__main__":
    unittest.main()
