"""Canonical renderer and final clean source map."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

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
from .roles import MaskRole


class SpanKind(str, Enum):
    RULE = "RULE"
    CONTENT = "CONTENT"


@dataclass(frozen=True, slots=True)
class SourceSpan:
    start: int
    end: int
    node_id: str
    role: MaskRole
    kind: SpanKind


@dataclass(frozen=True, slots=True)
class RenderedProgram:
    text: str
    spans: tuple[SourceSpan, ...]

    def validate(self) -> None:
        cursor = 0
        for span in self.spans:
            if span.start != cursor:
                raise ValueError(
                    f"source map gap/overlap at {cursor}: next span={span}"
                )
            if span.end <= span.start:
                raise ValueError(f"empty source span: {span}")
            cursor = span.end
        if cursor != len(self.text):
            raise ValueError(
                f"source map ends at {cursor}, text has length {len(self.text)}"
            )


class _Builder:
    def __init__(self) -> None:
        self.parts: list[str] = []
        self.spans: list[SourceSpan] = []
        self.position = 0

    def add(
        self,
        text: str,
        *,
        node_id: str,
        role: MaskRole,
        kind: SpanKind,
    ) -> None:
        if not text:
            return
        start = self.position
        self.parts.append(text)
        self.position += len(text)
        self.spans.append(
            SourceSpan(
                start=start,
                end=self.position,
                node_id=node_id,
                role=role,
                kind=kind,
            )
        )

    def finish(self) -> RenderedProgram:
        result = RenderedProgram("".join(self.parts), tuple(self.spans))
        result.validate()
        return result


def render_module(module: Module) -> str:
    return render_with_source_map(module).text


def render_with_source_map(module: Module) -> RenderedProgram:
    validate_module(module)
    builder = _Builder()
    previous: LineNode | None = None
    for line in module.body.lines:
        if previous is not None and (
            isinstance(previous, FunctionDefinition)
            or isinstance(line, FunctionDefinition)
        ):
            builder.add(
                "\n\n",
                node_id=module.node_id,
                role=MaskRole.RULE,
                kind=SpanKind.RULE,
            )
        _render_line(builder, line)
        previous = line
    if not module.body.lines:
        builder.add(
            "\n",
            node_id=module.node_id,
            role=MaskRole.RULE,
            kind=SpanKind.RULE,
        )
    return builder.finish()


def _render_body(builder: _Builder, body: Body) -> None:
    for line in body.lines:
        _render_line(builder, line)


def _prefix(builder: _Builder, node_id: str, depth: int, text: str) -> None:
    builder.add(
        "    " * depth + text,
        node_id=node_id,
        role=MaskRole.RULE,
        kind=SpanKind.RULE,
    )


def _newline(builder: _Builder, node_id: str) -> None:
    builder.add(
        "\n",
        node_id=node_id,
        role=MaskRole.RULE,
        kind=SpanKind.RULE,
    )


def _content(
    builder: _Builder, node_id: str, role: MaskRole, text: str
) -> None:
    builder.add(
        text,
        node_id=node_id,
        role=role,
        kind=SpanKind.CONTENT,
    )


def _render_line(builder: _Builder, line: LineNode) -> None:
    if isinstance(line, SimpleStatement):
        _prefix(builder, line.node_id, line.depth, "")
        _content(builder, line.node_id, MaskRole.TOKEN_STMT, line.text)
        _newline(builder, line.node_id)
        return

    if isinstance(line, FunctionDefinition):
        _prefix(builder, line.node_id, line.depth, "def ")
        _content(builder, line.node_id, MaskRole.TOKEN_HDR, line.header)
        builder.add(
            ":",
            node_id=line.node_id,
            role=MaskRole.RULE,
            kind=SpanKind.RULE,
        )
        _newline(builder, line.node_id)
        _render_body(builder, line.body)
        return

    if isinstance(line, IfStatement):
        _prefix(builder, line.node_id, line.depth, "if ")
        _content(builder, line.node_id, MaskRole.TOKEN_HDR, line.condition)
        builder.add(
            ":",
            node_id=line.node_id,
            role=MaskRole.RULE,
            kind=SpanKind.RULE,
        )
        _newline(builder, line.node_id)
        _render_body(builder, line.body)
        for clause in line.elif_clauses:
            _prefix(builder, clause.node_id, clause.depth, "elif ")
            _content(
                builder, clause.node_id, MaskRole.TOKEN_HDR, clause.condition
            )
            builder.add(
                ":",
                node_id=clause.node_id,
                role=MaskRole.RULE,
                kind=SpanKind.RULE,
            )
            _newline(builder, clause.node_id)
            _render_body(builder, clause.body)
        if line.else_body:
            _prefix(builder, line.node_id, line.depth, "else:")
            _newline(builder, line.node_id)
            _render_body(builder, line.else_body)
        return

    if isinstance(line, ForStatement):
        _prefix(builder, line.node_id, line.depth, "for ")
        _content(
            builder,
            line.node_id,
            MaskRole.TOKEN_HDR,
            f"{line.target} in {line.iterator}",
        )
        builder.add(
            ":",
            node_id=line.node_id,
            role=MaskRole.RULE,
            kind=SpanKind.RULE,
        )
        _newline(builder, line.node_id)
        _render_body(builder, line.body)
        if line.else_body:
            _prefix(builder, line.node_id, line.depth, "else:")
            _newline(builder, line.node_id)
            _render_body(builder, line.else_body)
        return

    if isinstance(line, WhileStatement):
        _prefix(builder, line.node_id, line.depth, "while ")
        _content(builder, line.node_id, MaskRole.TOKEN_HDR, line.condition)
        builder.add(
            ":",
            node_id=line.node_id,
            role=MaskRole.RULE,
            kind=SpanKind.RULE,
        )
        _newline(builder, line.node_id)
        _render_body(builder, line.body)
        if line.else_body:
            _prefix(builder, line.node_id, line.depth, "else:")
            _newline(builder, line.node_id)
            _render_body(builder, line.else_body)
        return

    raise TypeError(f"unsupported IR line: {type(line).__name__}")
