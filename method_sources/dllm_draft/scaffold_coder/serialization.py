"""JSON-serializable representation of the typed IR."""

from __future__ import annotations

from typing import Any

from .errors import IRValidationError
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


def module_to_dict(module: Module) -> dict[str, Any]:
    validate_module(module)
    return {
        "kind": "module",
        "node_id": module.node_id,
        "body": body_to_dict(module.body),
    }


def body_to_dict(body: Body) -> dict[str, Any]:
    return {
        "body_id": body.body_id,
        "owner_id": body.owner_id,
        "depth": body.depth,
        "lines": [line_to_dict(line) for line in body.lines],
    }


def line_to_dict(line: LineNode) -> dict[str, Any]:
    base = {"kind": line.kind, "node_id": line.node_id, "depth": line.depth}
    if isinstance(line, SimpleStatement):
        return {**base, "text": line.text}
    if isinstance(line, FunctionDefinition):
        return {**base, "header": line.header, "body": body_to_dict(line.body)}
    if isinstance(line, IfStatement):
        return {
            **base,
            "condition": line.condition,
            "body": body_to_dict(line.body),
            "elif_clauses": [
                {
                    "node_id": clause.node_id,
                    "depth": clause.depth,
                    "condition": clause.condition,
                    "body": body_to_dict(clause.body),
                }
                for clause in line.elif_clauses
            ],
            "else_body": body_to_dict(line.else_body)
            if line.else_body
            else None,
        }
    if isinstance(line, ForStatement):
        return {
            **base,
            "target": line.target,
            "iterator": line.iterator,
            "body": body_to_dict(line.body),
            "else_body": body_to_dict(line.else_body)
            if line.else_body
            else None,
        }
    if isinstance(line, WhileStatement):
        return {
            **base,
            "condition": line.condition,
            "body": body_to_dict(line.body),
            "else_body": body_to_dict(line.else_body)
            if line.else_body
            else None,
        }
    raise IRValidationError(f"cannot serialize line {type(line).__name__}")


def module_from_dict(value: dict[str, Any]) -> Module:
    if value.get("kind") != "module":
        raise IRValidationError("serialized root must have kind='module'")
    module = Module(
        node_id=str(value["node_id"]),
        body=body_from_dict(value["body"]),
    )
    validate_module(module)
    return module


def body_from_dict(value: dict[str, Any]) -> Body:
    return Body(
        body_id=str(value["body_id"]),
        owner_id=str(value["owner_id"]),
        depth=int(value["depth"]),
        lines=tuple(line_from_dict(line) for line in value["lines"]),
    )


def line_from_dict(value: dict[str, Any]) -> LineNode:
    kind = value["kind"]
    common = {"node_id": str(value["node_id"]), "depth": int(value["depth"])}
    if kind == "simple":
        return SimpleStatement(**common, text=str(value["text"]))
    if kind == "function":
        return FunctionDefinition(
            **common,
            header=str(value["header"]),
            body=body_from_dict(value["body"]),
        )
    if kind == "if":
        clauses = tuple(
            ElifClause(
                node_id=str(clause["node_id"]),
                depth=int(clause["depth"]),
                condition=str(clause["condition"]),
                body=body_from_dict(clause["body"]),
            )
            for clause in value["elif_clauses"]
        )
        return IfStatement(
            **common,
            condition=str(value["condition"]),
            body=body_from_dict(value["body"]),
            elif_clauses=clauses,
            else_body=body_from_dict(value["else_body"])
            if value.get("else_body")
            else None,
        )
    if kind == "for":
        return ForStatement(
            **common,
            target=str(value["target"]),
            iterator=str(value["iterator"]),
            body=body_from_dict(value["body"]),
            else_body=body_from_dict(value["else_body"])
            if value.get("else_body")
            else None,
        )
    if kind == "while":
        return WhileStatement(
            **common,
            condition=str(value["condition"]),
            body=body_from_dict(value["body"]),
            else_body=body_from_dict(value["else_body"])
            if value.get("else_body")
            else None,
        )
    raise IRValidationError(f"unknown serialized line kind {kind!r}")
