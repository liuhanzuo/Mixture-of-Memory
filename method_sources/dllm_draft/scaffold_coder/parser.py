"""Parse supported Python into the typed v0 IR."""

from __future__ import annotations

import ast
import copy
from dataclasses import dataclass

from .errors import UnsupportedSyntaxError
from .ir import (
    Body,
    ElifClause,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    LineNode,
    Module,
    SimpleStatement,
    WhileStatement,
    validate_module,
)


@dataclass
class _NodeIdFactory:
    next_value: int = 0

    def new(self, prefix: str) -> str:
        self.next_value += 1
        return f"{prefix}-{self.next_value:06d}"


def parse_source(source: str) -> Module:
    parsed = ast.parse(source)
    factory = _NodeIdFactory()
    module_id = factory.new("module")
    body_id = factory.new("body")
    lines = _parse_statement_list(
        _strip_leading_docstring(parsed.body),
        depth=0,
        factory=factory,
        allow_empty=True,
    )
    module = Module(
        node_id=module_id,
        body=Body(
            body_id=body_id,
            owner_id=module_id,
            depth=0,
            lines=lines,
        ),
    )
    validate_module(module)
    return module


def normalize_source(source: str) -> str:
    from .renderer import render_module

    normalized = render_module(parse_source(source))
    ast.parse(normalized)
    return normalized


def _parse_statement_list(
    statements: list[ast.stmt],
    *,
    depth: int,
    factory: _NodeIdFactory,
    allow_empty: bool = False,
) -> tuple[LineNode, ...]:
    lines = tuple(_parse_statement(stmt, depth=depth, factory=factory) for stmt in statements)
    if lines or allow_empty:
        return lines
    return (
        SimpleStatement(
            node_id=factory.new("stmt"),
            depth=depth,
            text="pass",
        ),
    )


def _parse_statement(
    statement: ast.stmt, *, depth: int, factory: _NodeIdFactory
) -> LineNode:
    if isinstance(statement, ast.AsyncFunctionDef):
        raise UnsupportedSyntaxError("async function definitions are disabled in v0")

    if isinstance(statement, ast.FunctionDef):
        if statement.decorator_list:
            raise UnsupportedSyntaxError("decorators are disabled in v0")
        node_id = factory.new("func")
        body_id = factory.new("body")
        body_nodes = _strip_leading_docstring(statement.body)
        body = Body(
            body_id=body_id,
            owner_id=node_id,
            depth=depth + 1,
            lines=_parse_statement_list(
                body_nodes,
                depth=depth + 1,
                factory=factory,
            ),
        )
        return FunctionDefinition(
            node_id=node_id,
            depth=depth,
            header=_function_header(statement),
            body=body,
        )

    if isinstance(statement, ast.If):
        return _parse_if(statement, depth=depth, factory=factory)

    if isinstance(statement, ast.AsyncFor):
        raise UnsupportedSyntaxError("async for is disabled in v0")

    if isinstance(statement, ast.For):
        node_id = factory.new("for")
        body_id = factory.new("body")
        body = Body(
            body_id=body_id,
            owner_id=node_id,
            depth=depth + 1,
            lines=_parse_statement_list(
                statement.body,
                depth=depth + 1,
                factory=factory,
            ),
        )
        else_body = None
        if statement.orelse:
            else_body = Body(
                body_id=factory.new("body"),
                owner_id=node_id,
                depth=depth + 1,
                lines=_parse_statement_list(
                    statement.orelse,
                    depth=depth + 1,
                    factory=factory,
                ),
            )
        return ForStatement(
            node_id=node_id,
            depth=depth,
            target=_one_line(ast.unparse(statement.target), "for target"),
            iterator=_one_line(ast.unparse(statement.iter), "for iterator"),
            body=body,
            else_body=else_body,
        )

    if isinstance(statement, ast.While):
        node_id = factory.new("while")
        body_id = factory.new("body")
        body = Body(
            body_id=body_id,
            owner_id=node_id,
            depth=depth + 1,
            lines=_parse_statement_list(
                statement.body,
                depth=depth + 1,
                factory=factory,
            ),
        )
        else_body = None
        if statement.orelse:
            else_body = Body(
                body_id=factory.new("body"),
                owner_id=node_id,
                depth=depth + 1,
                lines=_parse_statement_list(
                    statement.orelse,
                    depth=depth + 1,
                    factory=factory,
                ),
            )
        return WhileStatement(
            node_id=node_id,
            depth=depth,
            condition=_one_line(ast.unparse(statement.test), "while condition"),
            body=body,
            else_body=else_body,
        )

    unsupported = (
        ast.ClassDef,
        ast.Try,
        ast.With,
        ast.AsyncWith,
    )
    if hasattr(ast, "TryStar"):
        unsupported = (*unsupported, ast.TryStar)
    if hasattr(ast, "Match"):
        unsupported = (*unsupported, ast.Match)
    if isinstance(statement, unsupported):
        raise UnsupportedSyntaxError(
            f"{type(statement).__name__} is disabled in v0"
        )

    text = _one_line(ast.unparse(statement), type(statement).__name__)
    return SimpleStatement(
        node_id=factory.new("stmt"),
        depth=depth,
        text=text,
    )


def _parse_if(
    statement: ast.If, *, depth: int, factory: _NodeIdFactory
) -> IfStatement:
    node_id = factory.new("if")
    body_id = factory.new("body")
    body = Body(
        body_id=body_id,
        owner_id=node_id,
        depth=depth + 1,
        lines=_parse_statement_list(
            statement.body,
            depth=depth + 1,
            factory=factory,
        ),
    )
    clauses: list[ElifClause] = []
    tail = statement.orelse
    while len(tail) == 1 and isinstance(tail[0], ast.If):
        nested = tail[0]
        clause_id = factory.new("elif")
        clauses.append(
            ElifClause(
                node_id=clause_id,
                depth=depth,
                condition=_one_line(ast.unparse(nested.test), "elif condition"),
                body=Body(
                    body_id=factory.new("body"),
                    owner_id=clause_id,
                    depth=depth + 1,
                    lines=_parse_statement_list(
                        nested.body,
                        depth=depth + 1,
                        factory=factory,
                    ),
                ),
            )
        )
        tail = nested.orelse

    else_body = None
    if tail:
        else_body = Body(
            body_id=factory.new("body"),
            owner_id=node_id,
            depth=depth + 1,
            lines=_parse_statement_list(
                tail,
                depth=depth + 1,
                factory=factory,
            ),
        )

    return IfStatement(
        node_id=node_id,
        depth=depth,
        condition=_one_line(ast.unparse(statement.test), "if condition"),
        body=body,
        elif_clauses=tuple(clauses),
        else_body=else_body,
    )


def _function_header(statement: ast.FunctionDef) -> str:
    stub = copy.deepcopy(statement)
    stub.decorator_list = []
    stub.body = [ast.Pass()]
    ast.fix_missing_locations(stub)
    rendered = ast.unparse(stub)
    first_line = rendered.splitlines()[0]
    if not first_line.startswith("def ") or not first_line.endswith(":"):
        raise UnsupportedSyntaxError(
            f"could not canonicalize function header: {first_line!r}"
        )
    return _one_line(first_line[len("def ") : -1], "function header")


def _strip_leading_docstring(statements: list[ast.stmt]) -> list[ast.stmt]:
    if statements and isinstance(statements[0], ast.Expr):
        value = statements[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return statements[1:]
    return statements


def _one_line(text: str, role: str) -> str:
    stripped = text.strip()
    if not stripped or "\n" in stripped or "\r" in stripped:
        raise UnsupportedSyntaxError(
            f"{role} is not a canonical single physical line: {text!r}"
        )
    return stripped
