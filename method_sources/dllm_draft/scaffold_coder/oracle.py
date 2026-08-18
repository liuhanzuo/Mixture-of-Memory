"""Symbolic oracle reverse trace for runtime/property testing.

This module deliberately uses a tokenizer-independent symbolic tokenization.
Real BPE integration is a later layer; the trace validates hierarchy, edit
semantics, target roles, clause order, and termination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from .errors import BudgetExceededError, RuntimeInvariantError
from .ir import (
    Body,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    LineNode,
    Module,
    SimpleStatement,
    WhileStatement,
    validate_module,
)
from .renderer import render_module
from .roles import (
    DELETE,
    ELIF,
    ELSE,
    EXPAND,
    MaskRole,
    PredictionTarget,
    TargetKind,
    line_label_for_kind,
)


_SYMBOLIC_TOKEN_RE = re.compile(
    r"\s+"
    r"|[A-Za-z_]\w*"
    r"|\d+(?:\.\d+)?"
    r"|==|!=|<=|>=|:=|//|\*\*|<<|>>|->|\+=|-=|\*=|/=|%=|&=|\|=|\^="
    r"|.",
    re.DOTALL,
)


class EventKind(str, Enum):
    PREDICT = "PREDICT"
    RULE = "RULE"


@dataclass(frozen=True, slots=True)
class RuntimeEvent:
    index: int
    kind: EventKind
    node_id: str
    action: str
    role: MaskRole | None = None
    target: str | None = None
    mask_id: str | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class OracleConfig:
    initial_module_slots: int = 2
    initial_body_slots: int = 2
    initial_token_slots: int = 4
    max_predictions: int = 100_000
    max_events: int = 200_000
    max_lines_per_body: int = 512
    max_tokens_per_region: int = 4096


@dataclass(frozen=True, slots=True)
class OracleResult:
    rendered_code: str
    predictions: tuple[PredictionTarget, ...]
    events: tuple[RuntimeEvent, ...]

    def validate(self) -> None:
        for prediction in self.predictions:
            prediction.validate()
        if len([event for event in self.events if event.kind is EventKind.PREDICT]) != len(
            self.predictions
        ):
            raise RuntimeInvariantError(
                "prediction-event count does not match target count"
            )


class OracleRuntime:
    def __init__(self, config: OracleConfig | None = None) -> None:
        self.config = config or OracleConfig()
        self._predictions: list[PredictionTarget] = []
        self._events: list[RuntimeEvent] = []
        self._mask_counter = 0

    def generate(self, module: Module) -> OracleResult:
        validate_module(module)
        self._predictions = []
        self._events = []
        self._mask_counter = 0
        self._rule(module.node_id, "OPEN_MODULE_BODY", "[BODY] -> line masks")
        self._plan_and_fill_body(
            module.body,
            role=MaskRole.LINE_MODULE,
            initial_slots=self.config.initial_module_slots,
        )
        code = render_module(module)
        result = OracleResult(
            rendered_code=code,
            predictions=tuple(self._predictions),
            events=tuple(self._events),
        )
        result.validate()
        return result

    def _plan_and_fill_body(
        self, body: Body, *, role: MaskRole, initial_slots: int
    ) -> None:
        target_count = len(body.lines)
        if target_count > self.config.max_lines_per_body:
            raise BudgetExceededError(
                f"body {body.body_id} has {target_count} lines"
            )
        if body.depth > 0 and target_count == 0:
            raise RuntimeInvariantError(
                f"required body {body.body_id} cannot be empty"
            )

        slot_count = max(1 if body.depth > 0 else 0, initial_slots)
        slot_count = self._resize_slots(
            node_id=body.body_id,
            role=role,
            current=slot_count,
            target=target_count,
            minimum=1 if body.depth > 0 else 0,
            unit="line",
        )
        if slot_count != target_count:
            raise RuntimeInvariantError("oracle failed to match body line count")

        # Local-body barrier: commit every line type before opening any child.
        for line in body.lines:
            label = line_label_for_kind(line.kind)
            self._predict(line.node_id, role, TargetKind.SPECIAL, label)
        self._rule(
            body.body_id,
            "BODY_PLAN_COMPLETE",
            f"{target_count} line labels committed before child opening",
        )

        for line in body.lines:
            self._fill_line(line)

    def _fill_line(self, line: LineNode) -> None:
        if isinstance(line, SimpleStatement):
            self._rule(line.node_id, "OPEN_STMT", "[STMT] -> token masks")
            self._fill_text(line.node_id, MaskRole.TOKEN_STMT, line.text)
            return

        if isinstance(line, FunctionDefinition):
            self._rule(line.node_id, "EXPAND_FUNC", "[FUNC] -> template holes")
            self._rule(line.node_id, "OPEN_HDR", "[HDR:function] -> token masks")
            self._fill_text(line.node_id, MaskRole.TOKEN_HDR, line.header)
            self._rule(line.node_id, "OPEN_BODY", "[BODY] -> line masks")
            self._plan_and_fill_body(
                line.body,
                role=MaskRole.LINE_BODY,
                initial_slots=self.config.initial_body_slots,
            )
            return

        if isinstance(line, IfStatement):
            self._rule(line.node_id, "EXPAND_IF", "[IF] -> template holes")
            self._rule(line.node_id, "OPEN_HDR", "[HDR:condition] -> token masks")
            self._fill_text(line.node_id, MaskRole.TOKEN_HDR, line.condition)
            self._rule(line.node_id, "OPEN_BODY", "[BODY] -> line masks")
            self._plan_and_fill_body(
                line.body,
                role=MaskRole.LINE_BODY,
                initial_slots=self.config.initial_body_slots,
            )
            for clause in line.elif_clauses:
                self._predict(
                    clause.node_id,
                    MaskRole.LINE_CLAUSE,
                    TargetKind.SPECIAL,
                    ELIF,
                )
                self._rule(
                    clause.node_id,
                    "EXPAND_ELIF",
                    "[ELIF] -> header + body + clauses",
                )
                self._fill_text(
                    clause.node_id, MaskRole.TOKEN_HDR, clause.condition
                )
                self._plan_and_fill_body(
                    clause.body,
                    role=MaskRole.LINE_BODY,
                    initial_slots=self.config.initial_body_slots,
                )
            if line.else_body is not None:
                self._predict(
                    line.node_id,
                    MaskRole.LINE_CLAUSE,
                    TargetKind.SPECIAL,
                    ELSE,
                )
                self._rule(line.node_id, "EXPAND_ELSE", "[ELSE] -> body")
                self._plan_and_fill_body(
                    line.else_body,
                    role=MaskRole.LINE_BODY,
                    initial_slots=self.config.initial_body_slots,
                )
            else:
                self._predict(
                    line.node_id,
                    MaskRole.LINE_CLAUSE,
                    TargetKind.SPECIAL,
                    DELETE,
                )
                self._rule(line.node_id, "DELETE_CLAUSES", "no trailing clause")
            return

        if isinstance(line, ForStatement):
            self._rule(line.node_id, "EXPAND_FOR", "[FOR] -> template holes")
            self._rule(line.node_id, "OPEN_HDR", "[HDR:for] -> token masks")
            self._fill_text(
                line.node_id,
                MaskRole.TOKEN_HDR,
                f"{line.target} in {line.iterator}",
            )
            self._rule(line.node_id, "OPEN_BODY", "[BODY] -> line masks")
            self._plan_and_fill_body(
                line.body,
                role=MaskRole.LINE_BODY,
                initial_slots=self.config.initial_body_slots,
            )
            self._fill_optional_else(line.node_id, line.else_body)
            return

        if isinstance(line, WhileStatement):
            self._rule(line.node_id, "EXPAND_WHILE", "[WHILE] -> template holes")
            self._rule(line.node_id, "OPEN_HDR", "[HDR:condition] -> token masks")
            self._fill_text(line.node_id, MaskRole.TOKEN_HDR, line.condition)
            self._rule(line.node_id, "OPEN_BODY", "[BODY] -> line masks")
            self._plan_and_fill_body(
                line.body,
                role=MaskRole.LINE_BODY,
                initial_slots=self.config.initial_body_slots,
            )
            self._fill_optional_else(line.node_id, line.else_body)
            return

        raise RuntimeInvariantError(
            f"oracle cannot fill line type {type(line).__name__}"
        )

    def _fill_optional_else(self, node_id: str, body: Body | None) -> None:
        if body is None:
            self._predict(
                node_id,
                MaskRole.LINE_CLAUSE,
                TargetKind.SPECIAL,
                DELETE,
            )
            self._rule(node_id, "DELETE_CLAUSES", "no trailing else")
            return
        self._predict(
            node_id,
            MaskRole.LINE_CLAUSE,
            TargetKind.SPECIAL,
            ELSE,
        )
        self._rule(node_id, "EXPAND_ELSE", "[ELSE] -> body")
        self._plan_and_fill_body(
            body,
            role=MaskRole.LINE_BODY,
            initial_slots=self.config.initial_body_slots,
        )

    def _fill_text(self, node_id: str, role: MaskRole, text: str) -> None:
        tokens = symbolic_tokenize(text)
        if len(tokens) > self.config.max_tokens_per_region:
            raise BudgetExceededError(
                f"region {node_id} has {len(tokens)} symbolic tokens"
            )
        slot_count = self._resize_slots(
            node_id=node_id,
            role=role,
            current=self.config.initial_token_slots,
            target=len(tokens),
            minimum=1,
            unit="token",
        )
        if slot_count != len(tokens):
            raise RuntimeInvariantError("oracle failed to match token count")
        for token in tokens:
            self._predict(node_id, role, TargetKind.LEXICAL, token)
        if "".join(tokens) != text:
            raise RuntimeInvariantError(
                f"symbolic tokenization is not lossless for {text!r}"
            )

    def _resize_slots(
        self,
        *,
        node_id: str,
        role: MaskRole,
        current: int,
        target: int,
        minimum: int,
        unit: str,
    ) -> int:
        if target < minimum:
            raise RuntimeInvariantError(
                f"target {unit} count {target} below minimum {minimum}"
            )
        while current < target:
            self._predict(node_id, role, TargetKind.SPECIAL, EXPAND)
            current += 1
            self._rule(
                node_id,
                f"EXPAND_{unit.upper()}",
                f"one {unit} mask -> two {unit} masks; count={current}",
            )
        while current > target:
            if current - 1 < minimum:
                raise RuntimeInvariantError(
                    f"cannot delete final required {unit} slot"
                )
            self._predict(node_id, role, TargetKind.SPECIAL, DELETE)
            current -= 1
            self._rule(
                node_id,
                f"DELETE_{unit.upper()}",
                f"remove one {unit} slot; count={current}",
            )
        return current

    def _predict(
        self,
        node_id: str,
        role: MaskRole,
        kind: TargetKind,
        token: str,
    ) -> None:
        if len(self._predictions) >= self.config.max_predictions:
            raise BudgetExceededError("oracle prediction budget exceeded")
        self._mask_counter += 1
        mask_id = f"mask-{self._mask_counter:08d}"
        target = PredictionTarget(
            mask_id=mask_id,
            node_id=node_id,
            role=role,
            kind=kind,
            token=token,
        )
        target.validate()
        self._predictions.append(target)
        self._event(
            kind=EventKind.PREDICT,
            node_id=node_id,
            action="COMMIT",
            role=role,
            target=token,
            mask_id=mask_id,
        )

    def _rule(self, node_id: str, action: str, detail: str) -> None:
        self._event(
            kind=EventKind.RULE,
            node_id=node_id,
            action=action,
            detail=detail,
        )

    def _event(
        self,
        *,
        kind: EventKind,
        node_id: str,
        action: str,
        role: MaskRole | None = None,
        target: str | None = None,
        mask_id: str | None = None,
        detail: str = "",
    ) -> None:
        if len(self._events) >= self.config.max_events:
            raise BudgetExceededError("oracle event budget exceeded")
        self._events.append(
            RuntimeEvent(
                index=len(self._events),
                kind=kind,
                node_id=node_id,
                action=action,
                role=role,
                target=target,
                mask_id=mask_id,
                detail=detail,
            )
        )


def symbolic_tokenize(text: str) -> tuple[str, ...]:
    if not text or "\n" in text or "\r" in text:
        raise RuntimeInvariantError(
            f"symbolic token region must be non-empty and single-line: {text!r}"
        )
    tokens = tuple(match.group(0) for match in _SYMBOLIC_TOKEN_RE.finditer(text))
    if "".join(tokens) != text:
        raise RuntimeInvariantError(f"symbolic tokenizer lost text: {text!r}")
    return tokens
