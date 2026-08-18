"""Typed, formatting-independent IR for the v0 Python grammar."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, TypeAlias

from .errors import IRValidationError


@dataclass(frozen=True, slots=True)
class Body:
    body_id: str
    owner_id: str
    depth: int
    lines: tuple["LineNode", ...]


@dataclass(frozen=True, slots=True)
class SimpleStatement:
    node_id: str
    depth: int
    text: str
    kind: str = "simple"


@dataclass(frozen=True, slots=True)
class FunctionDefinition:
    node_id: str
    depth: int
    header: str
    body: Body
    kind: str = "function"


@dataclass(frozen=True, slots=True)
class ElifClause:
    node_id: str
    depth: int
    condition: str
    body: Body


@dataclass(frozen=True, slots=True)
class IfStatement:
    node_id: str
    depth: int
    condition: str
    body: Body
    elif_clauses: tuple[ElifClause, ...]
    else_body: Body | None
    kind: str = "if"


@dataclass(frozen=True, slots=True)
class ForStatement:
    node_id: str
    depth: int
    target: str
    iterator: str
    body: Body
    else_body: Body | None
    kind: str = "for"


@dataclass(frozen=True, slots=True)
class WhileStatement:
    node_id: str
    depth: int
    condition: str
    body: Body
    else_body: Body | None
    kind: str = "while"


LineNode: TypeAlias = (
    SimpleStatement
    | FunctionDefinition
    | IfStatement
    | ForStatement
    | WhileStatement
)


@dataclass(frozen=True, slots=True)
class Module:
    node_id: str
    body: Body


def iter_lines(body: Body) -> Iterator[LineNode]:
    for line in body.lines:
        yield line
        if isinstance(line, FunctionDefinition):
            yield from iter_lines(line.body)
        elif isinstance(line, IfStatement):
            yield from iter_lines(line.body)
            for clause in line.elif_clauses:
                yield from iter_lines(clause.body)
            if line.else_body:
                yield from iter_lines(line.else_body)
        elif isinstance(line, (ForStatement, WhileStatement)):
            yield from iter_lines(line.body)
            if line.else_body:
                yield from iter_lines(line.else_body)


def validate_module(module: Module) -> None:
    seen: set[str] = {module.node_id}
    _validate_body(module.body, expected_depth=0, seen=seen)


def _validate_body(body: Body, expected_depth: int, seen: set[str]) -> None:
    if body.body_id in seen:
        raise IRValidationError(f"duplicate body id {body.body_id}")
    seen.add(body.body_id)
    if body.depth != expected_depth:
        raise IRValidationError(
            f"body {body.body_id} depth={body.depth}, expected={expected_depth}"
        )
    if not body.lines and expected_depth > 0:
        raise IRValidationError(
            f"required suite {body.body_id} owned by {body.owner_id} must be non-empty"
        )
    for line in body.lines:
        if line.node_id in seen:
            raise IRValidationError(f"duplicate node id {line.node_id}")
        seen.add(line.node_id)
        if line.depth != expected_depth:
            raise IRValidationError(
                f"line {line.node_id} depth={line.depth}, expected={expected_depth}"
            )
        if isinstance(line, SimpleStatement):
            if not line.text or "\n" in line.text or "\r" in line.text:
                raise IRValidationError(
                    f"simple statement {line.node_id} must be one non-empty line"
                )
        elif isinstance(line, FunctionDefinition):
            _validate_header(line.header, line.node_id)
            _validate_body(line.body, expected_depth + 1, seen)
        elif isinstance(line, IfStatement):
            _validate_header(line.condition, line.node_id)
            _validate_body(line.body, expected_depth + 1, seen)
            for clause in line.elif_clauses:
                if clause.node_id in seen:
                    raise IRValidationError(f"duplicate node id {clause.node_id}")
                seen.add(clause.node_id)
                if clause.depth != expected_depth:
                    raise IRValidationError(
                        f"elif {clause.node_id} has wrong depth {clause.depth}"
                    )
                _validate_header(clause.condition, clause.node_id)
                _validate_body(clause.body, expected_depth + 1, seen)
            if line.else_body:
                _validate_body(line.else_body, expected_depth + 1, seen)
        elif isinstance(line, ForStatement):
            _validate_header(line.target, line.node_id)
            _validate_header(line.iterator, line.node_id)
            _validate_body(line.body, expected_depth + 1, seen)
            if line.else_body:
                _validate_body(line.else_body, expected_depth + 1, seen)
        elif isinstance(line, WhileStatement):
            _validate_header(line.condition, line.node_id)
            _validate_body(line.body, expected_depth + 1, seen)
            if line.else_body:
                _validate_body(line.else_body, expected_depth + 1, seen)
        else:  # pragma: no cover - defensive against future unregistered nodes
            raise IRValidationError(f"unsupported IR line {type(line).__name__}")


def _validate_header(text: str, node_id: str) -> None:
    if not text or "\n" in text or "\r" in text:
        raise IRValidationError(
            f"header content for {node_id} must be one non-empty line"
        )
