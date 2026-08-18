"""Mask roles, structural labels, and target validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import RuntimeInvariantError


class MaskRole(str, Enum):
    LINE_MODULE = "LINE_MODULE"
    LINE_BODY = "LINE_BODY"
    LINE_CLAUSE = "LINE_CLAUSE"
    TOKEN_STMT = "TOKEN_STMT"
    TOKEN_HDR = "TOKEN_HDR"
    TOKEN_DOC = "TOKEN_DOC"
    RULE = "RULE"


class TargetKind(str, Enum):
    SPECIAL = "SPECIAL"
    LEXICAL = "LEXICAL"


STMT = "[STMT]"
FUNC = "[FUNC]"
FOR = "[FOR]"
WHILE = "[WHILE]"
IF = "[IF]"
ELIF = "[ELIF]"
ELSE = "[ELSE]"
EXPAND = "[expand]"
DELETE = "[delete]"

HDR = "[HDR]"
DOC = "[DOC]"
BODY = "[BODY]"
CLAUSES = "[CLAUSES]"

RULE_ONLY_HOLES = frozenset({HDR, DOC, BODY, CLAUSES})
EDIT_LABELS = frozenset({EXPAND, DELETE})
V0_CONSTRUCT_LABELS = frozenset({FUNC, FOR, WHILE, IF})
V0_BODY_LINE_LABELS = frozenset({STMT, *V0_CONSTRUCT_LABELS})
V0_CLAUSE_LABELS = frozenset({ELIF, ELSE})


@dataclass(frozen=True, slots=True)
class PredictionTarget:
    """One immediate-rung target for one symbolic mask."""

    mask_id: str
    node_id: str
    role: MaskRole
    kind: TargetKind
    token: str

    def validate(self) -> None:
        validate_prediction_target(self)


def validate_prediction_target(target: PredictionTarget) -> None:
    if target.token in RULE_ONLY_HOLES:
        raise RuntimeInvariantError(
            f"rule-only hole {target.token} cannot be a prediction target"
        )

    if target.role in {MaskRole.LINE_MODULE, MaskRole.LINE_BODY}:
        legal = V0_BODY_LINE_LABELS | EDIT_LABELS
        if target.kind is not TargetKind.SPECIAL or target.token not in legal:
            raise RuntimeInvariantError(
                f"illegal body/module line target: {target}"
            )
        return

    if target.role is MaskRole.LINE_CLAUSE:
        legal = V0_CLAUSE_LABELS | {DELETE}
        if target.kind is not TargetKind.SPECIAL or target.token not in legal:
            raise RuntimeInvariantError(f"illegal clause target: {target}")
        return

    if target.role in {
        MaskRole.TOKEN_STMT,
        MaskRole.TOKEN_HDR,
        MaskRole.TOKEN_DOC,
    }:
        if target.kind is TargetKind.SPECIAL:
            if target.token not in EDIT_LABELS:
                raise RuntimeInvariantError(f"illegal token edit target: {target}")
        else:
            if not target.token or "\n" in target.token or "\r" in target.token:
                raise RuntimeInvariantError(
                    f"lexical target must be non-empty and single-line: {target}"
                )
        return

    raise RuntimeInvariantError(f"unsupported prediction role: {target.role}")


def line_label_for_kind(kind: str) -> str:
    mapping = {
        "simple": STMT,
        "function": FUNC,
        "for": FOR,
        "while": WHILE,
        "if": IF,
    }
    try:
        return mapping[kind]
    except KeyError as exc:
        raise RuntimeInvariantError(f"no v0 label for line kind {kind!r}") from exc

