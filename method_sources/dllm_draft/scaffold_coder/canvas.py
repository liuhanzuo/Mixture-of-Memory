"""Segmented token canvases for deterministic hierarchy rungs."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable

import torch
from transformers import PreTrainedTokenizerBase

from .errors import RuntimeInvariantError
from .ir import (
    Body,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    LineNode,
    Module,
    SimpleStatement,
    WhileStatement,
)
from .roles import (
    BODY,
    CLAUSES,
    DELETE,
    EXPAND,
    HDR,
    MaskRole,
    RULE_ONLY_HOLES,
    TargetKind,
    PredictionTarget,
    line_label_for_kind,
)
from .special_tokens import TOKEN_TEXT
from .tokenizer_utils import TokenExtension, extend_tokenizer


ROLE_ID = {
    MaskRole.RULE: 0,
    MaskRole.LINE_MODULE: 1,
    MaskRole.LINE_BODY: 2,
    MaskRole.LINE_CLAUSE: 3,
    MaskRole.TOKEN_STMT: 4,
    MaskRole.TOKEN_HDR: 5,
    MaskRole.TOKEN_DOC: 6,
}


@dataclass(frozen=True, slots=True)
class TokenEditConfig:
    merge_probability: float = 0.0
    max_delete: int = 0
    collapse_fully_masked: bool = False
    coupled_collapse_exponent: float | None = None

    def validate(self) -> None:
        if not 0 <= self.merge_probability <= 1:
            raise ValueError("merge_probability must be in [0,1]")
        if self.max_delete < 0:
            raise ValueError("max_delete must be non-negative")
        if (
            self.coupled_collapse_exponent is not None
            and self.coupled_collapse_exponent <= 0
        ):
            raise ValueError("coupled collapse exponent must be positive")


@dataclass(frozen=True, slots=True)
class LineEditConfig:
    merge_probability: float = 0.0
    max_delete: int = 0

    def validate(self) -> None:
        if not 0 <= self.merge_probability <= 1:
            raise ValueError("merge_probability must be in [0,1]")
        if self.max_delete < 0:
            raise ValueError("max_delete must be non-negative")


@dataclass(frozen=True, slots=True)
class _LinePlanTarget:
    notation: str
    node_id: str
    depth: int
    masked: bool


@dataclass(frozen=True, slots=True)
class TokenRegistry:
    tokenizer: PreTrainedTokenizerBase
    extensions: tuple[TokenExtension, ...]
    notation_to_id: dict[str, int]
    id_to_notation: dict[int, str]

    @classmethod
    def build(cls, tokenizer: PreTrainedTokenizerBase) -> "TokenRegistry":
        extensions = extend_tokenizer(tokenizer)
        notation_to_id = {
            extension.notation: extension.token_id for extension in extensions
        }
        return cls(
            tokenizer=tokenizer,
            extensions=extensions,
            notation_to_id=notation_to_id,
            id_to_notation={
                token_id: notation for notation, token_id in notation_to_id.items()
            },
        )

    def special_id(self, notation: str) -> int:
        try:
            return self.notation_to_id[notation]
        except KeyError as exc:
            raise RuntimeInvariantError(
                f"unregistered scaffold token {notation}"
            ) from exc


@dataclass(frozen=True, slots=True)
class CanvasState:
    state_name: str
    input_ids: tuple[int, ...]
    labels: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    roles: tuple[MaskRole, ...]
    node_ids: tuple[str, ...]
    eligible: tuple[bool, ...]

    def validate(self, registry: TokenRegistry) -> None:
        lengths = {
            len(self.input_ids),
            len(self.labels),
            len(self.loss_mask),
            len(self.roles),
            len(self.node_ids),
            len(self.eligible),
        }
        if len(lengths) != 1:
            raise RuntimeInvariantError(f"canvas fields differ in length: {lengths}")
        mask_id = registry.tokenizer.mask_token_id
        if mask_id is None:
            raise RuntimeInvariantError("tokenizer has no mask token")
        for index, supervised in enumerate(self.loss_mask):
            if not supervised:
                if self.eligible[index]:
                    raise RuntimeInvariantError(
                        f"non-mask position {index} cannot be eligible"
                    )
                continue
            if self.input_ids[index] != mask_id:
                raise RuntimeInvariantError(
                    f"supervised position {index} is not a mask token"
                )
            role = self.roles[index]
            target_id = self.labels[index]
            notation = registry.id_to_notation.get(target_id)
            if notation is not None:
                target = PredictionTarget(
                    mask_id=f"canvas-{index}",
                    node_id=self.node_ids[index],
                    role=role,
                    kind=TargetKind.SPECIAL,
                    token=notation,
                )
            else:
                target = PredictionTarget(
                    mask_id=f"canvas-{index}",
                    node_id=self.node_ids[index],
                    role=role,
                    kind=TargetKind.LEXICAL,
                    token=registry.tokenizer.decode([target_id]),
                )
            target.validate()
            if not self.eligible[index]:
                raise RuntimeInvariantError(
                    f"supervised mask position {index} must be eligible"
                )

    def to_tensors(self) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.tensor(self.input_ids, dtype=torch.long),
            "labels": torch.tensor(self.labels, dtype=torch.long),
            "loss_mask": torch.tensor(self.loss_mask, dtype=torch.bool),
            "role_ids": torch.tensor(
                [ROLE_ID[role] for role in self.roles], dtype=torch.long
            ),
            "eligible": torch.tensor(self.eligible, dtype=torch.bool),
            "attention_mask": torch.ones(len(self.input_ids), dtype=torch.long),
            "position_ids": torch.arange(len(self.input_ids), dtype=torch.long),
        }


class _Builder:
    def __init__(self, registry: TokenRegistry, state_name: str) -> None:
        self.registry = registry
        self.state_name = state_name
        self.input_ids: list[int] = []
        self.labels: list[int] = []
        self.loss_mask: list[bool] = []
        self.roles: list[MaskRole] = []
        self.node_ids: list[str] = []
        self.eligible: list[bool] = []

    def _append(
        self,
        *,
        input_id: int,
        label: int,
        loss: bool,
        role: MaskRole,
        node_id: str,
        eligible: bool,
    ) -> None:
        self.input_ids.append(input_id)
        self.labels.append(label)
        self.loss_mask.append(loss)
        self.roles.append(role)
        self.node_ids.append(node_id)
        self.eligible.append(eligible)

    def add_rule(self, text: str, node_id: str) -> None:
        for token_id in self.registry.tokenizer.encode(
            text, add_special_tokens=False
        ):
            self._append(
                input_id=token_id,
                label=token_id,
                loss=False,
                role=MaskRole.RULE,
                node_id=node_id,
                eligible=False,
            )

    def add_visible_special(
        self, notation: str, node_id: str, role: MaskRole
    ) -> None:
        token_id = self.registry.special_id(notation)
        self._append(
            input_id=token_id,
            label=token_id,
            loss=False,
            role=role,
            node_id=node_id,
            eligible=False,
        )

    def add_mask_target(
        self, notation: str, node_id: str, role: MaskRole
    ) -> None:
        self._append(
            input_id=self.registry.tokenizer.mask_token_id,
            label=self.registry.special_id(notation),
            loss=True,
            role=role,
            node_id=node_id,
            eligible=True,
        )

    def add_content(
        self,
        text: str,
        node_id: str,
        role: MaskRole,
        *,
        mask_probability: float,
        rng: random.Random,
        edit_config: TokenEditConfig | None = None,
        collapse_notation: str | None = None,
        collapse_role: MaskRole | None = None,
    ) -> None:
        edit_config = edit_config or TokenEditConfig()
        edit_config.validate()
        token_ids = self.registry.tokenizer.encode(
            text, add_special_tokens=False
        )
        if not token_ids:
            raise RuntimeInvariantError(
                f"content segment for {node_id} tokenized to empty"
            )
        if self.registry.tokenizer.decode(token_ids) != text:
            raise RuntimeInvariantError(
                f"segment tokenizer is not decode-exact for {text!r}"
            )
        if (
            edit_config.coupled_collapse_exponent is not None
            and rng.random()
            < mask_probability ** edit_config.coupled_collapse_exponent
        ):
            if collapse_notation is None or collapse_role is None:
                raise RuntimeInvariantError(
                    "coupled collapse requires notation and role"
                )
            self.add_visible_special(
                collapse_notation, node_id, collapse_role
            )
            return
        masked_flags = [
            rng.random() < mask_probability for _ in token_ids
        ]
        if (
            edit_config.collapse_fully_masked
            and masked_flags
            and all(masked_flags)
        ):
            if collapse_notation is None or collapse_role is None:
                raise RuntimeInvariantError(
                    "fully masked collapse requires notation and role"
                )
            self.add_visible_special(
                collapse_notation, node_id, collapse_role
            )
            return
        index = 0
        while index < len(token_ids):
            token_id = token_ids[index]
            masked = masked_flags[index]
            if (
                masked
                and index + 1 < len(token_ids)
                and masked_flags[index + 1]
                and rng.random() < edit_config.merge_probability
            ):
                self._append(
                    input_id=self.registry.tokenizer.mask_token_id,
                    label=self.registry.special_id(EXPAND),
                    loss=True,
                    role=role,
                    node_id=node_id,
                    eligible=True,
                )
                index += 2
                continue
            self._append(
                input_id=(
                    self.registry.tokenizer.mask_token_id if masked else token_id
                ),
                label=token_id,
                loss=masked,
                role=role,
                node_id=node_id,
                eligible=masked,
            )
            index += 1

        delete_count = (
            rng.randint(0, edit_config.max_delete)
            if edit_config.max_delete
            else 0
        )
        for _ in range(delete_count):
            self.add_mask_target(DELETE, node_id, role)

    def finish(self, ensure_supervision: bool = False) -> CanvasState:
        if ensure_supervision and not any(self.loss_mask):
            for index, role in enumerate(self.roles):
                if role in {
                    MaskRole.TOKEN_STMT,
                    MaskRole.TOKEN_HDR,
                    MaskRole.TOKEN_DOC,
                }:
                    self.input_ids[index] = self.registry.tokenizer.mask_token_id
                    self.loss_mask[index] = True
                    self.eligible[index] = True
                    break
            else:
                for index, role in enumerate(self.roles):
                    if role in {
                        MaskRole.LINE_MODULE,
                        MaskRole.LINE_BODY,
                        MaskRole.LINE_CLAUSE,
                    } and self.labels[index] in self.registry.id_to_notation:
                        self.input_ids[index] = (
                            self.registry.tokenizer.mask_token_id
                        )
                        self.loss_mask[index] = True
                        self.eligible[index] = True
                        break
        state = CanvasState(
            state_name=self.state_name,
            input_ids=tuple(self.input_ids),
            labels=tuple(self.labels),
            loss_mask=tuple(self.loss_mask),
            roles=tuple(self.roles),
            node_ids=tuple(self.node_ids),
            eligible=tuple(self.eligible),
        )
        state.validate(self.registry)
        return state


def build_root_plan(
    module: Module,
    registry: TokenRegistry,
    *,
    edit_config: LineEditConfig | None = None,
    seed: int = 0,
    line_mask_probability: float = 1.0,
) -> CanvasState:
    edit_config = edit_config or LineEditConfig()
    edit_config.validate()
    builder = _Builder(registry, "root_plan")
    rng = random.Random(seed)
    targets = _augment_line_targets(
        module.body,
        edit_config=edit_config,
        rng=rng,
        mask_probability=line_mask_probability,
    )
    for target in targets:
        if target.masked:
            builder.add_mask_target(
                target.notation,
                target.node_id,
                MaskRole.LINE_MODULE,
            )
        else:
            builder.add_visible_special(
                target.notation,
                target.node_id,
                MaskRole.LINE_MODULE,
            )
        builder.add_rule("\n", module.body.body_id)
    return builder.finish()


def build_template_skeleton(
    module: Module, registry: TokenRegistry
) -> CanvasState:
    builder = _Builder(registry, "template_skeleton")
    _render_skeleton_body(builder, module.body, module_level=True)
    return builder.finish()


def build_leaf_infill(
    module: Module,
    registry: TokenRegistry,
    *,
    mask_probability: float,
    seed: int,
    edit_config: TokenEditConfig | None = None,
    depth_probabilities: dict[int, float] | None = None,
) -> CanvasState:
    if not 0 <= mask_probability <= 1:
        raise ValueError("mask_probability must be in [0,1]")
    builder = _Builder(registry, "leaf_infill")
    rng = random.Random(seed)
    edit_config = edit_config or TokenEditConfig()
    edit_config.validate()
    _render_leaf_body(
        builder,
        module.body,
        rng=rng,
        mask_probability=mask_probability,
        module_level=True,
        edit_config=edit_config,
        depth_probabilities=depth_probabilities,
    )
    ensure_supervision = mask_probability > 0 or bool(
        depth_probabilities
        and any(value > 0 for value in depth_probabilities.values())
    )
    return builder.finish(ensure_supervision=ensure_supervision)


def build_body_plan(
    module: Module,
    registry: TokenRegistry,
    *,
    target_body_id: str,
    edit_config: LineEditConfig | None = None,
    seed: int = 0,
    line_mask_probability: float = 1.0,
) -> CanvasState:
    if not _body_contains(module.body, target_body_id):
        raise RuntimeInvariantError(f"body {target_body_id} is not in module")
    if not _is_main_body(module.body, target_body_id):
        raise RuntimeInvariantError(
            "v0 body-plan builder currently supports module/main construct "
            "bodies, not elif/else bodies"
        )
    builder = _Builder(registry, f"body_plan:{target_body_id}")
    edit_config = edit_config or LineEditConfig()
    edit_config.validate()
    _render_body_context(
        builder,
        module.body,
        target_body_id=target_body_id,
        module_level=True,
        edit_config=edit_config,
        rng=random.Random(seed),
        line_mask_probability=line_mask_probability,
    )
    return builder.finish()


def prepend_chat_prompt(
    state: CanvasState,
    registry: TokenRegistry,
    prompt: str,
    *,
    append_eos: bool = True,
) -> CanvasState:
    text = registry.tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_ids = registry.tokenizer.encode(text, add_special_tokens=False)
    suffix = [registry.tokenizer.eos_token_id] if append_eos else []
    count = len(prompt_ids)
    combined = CanvasState(
        state_name=f"chat+{state.state_name}",
        input_ids=tuple(prompt_ids) + state.input_ids + tuple(suffix),
        labels=tuple(prompt_ids) + state.labels + tuple(suffix),
        loss_mask=(False,) * count + state.loss_mask + (False,) * len(suffix),
        roles=(MaskRole.RULE,) * count
        + state.roles
        + (MaskRole.RULE,) * len(suffix),
        node_ids=("prompt",) * count
        + state.node_ids
        + ("response-eos",) * len(suffix),
        eligible=(False,) * count
        + state.eligible
        + (False,) * len(suffix),
    )
    combined.validate(registry)
    return combined


def iter_main_bodies(module: Module) -> Iterable[Body]:
    yield from _iter_main_bodies(module.body)


def _iter_main_bodies(body: Body) -> Iterable[Body]:
    yield body
    for line in body.lines:
        if isinstance(line, FunctionDefinition):
            yield from _iter_main_bodies(line.body)
        elif isinstance(line, IfStatement):
            yield from _iter_main_bodies(line.body)
        elif isinstance(line, (ForStatement, WhileStatement)):
            yield from _iter_main_bodies(line.body)


def _line_role(depth: int) -> MaskRole:
    return MaskRole.LINE_MODULE if depth == 0 else MaskRole.LINE_BODY


def _augment_line_targets(
    body: Body,
    *,
    edit_config: LineEditConfig,
    rng: random.Random,
    mask_probability: float,
) -> list[_LinePlanTarget]:
    if not 0 <= mask_probability <= 1:
        raise ValueError("line mask probability must be in [0,1]")
    masked_flags = [
        rng.random() < mask_probability for _ in body.lines
    ]
    if body.lines and mask_probability > 0 and not any(masked_flags):
        masked_flags[rng.randrange(len(masked_flags))] = True

    targets: list[_LinePlanTarget] = []
    index = 0
    while index < len(body.lines):
        line = body.lines[index]
        if (
            index + 1 < len(body.lines)
            and masked_flags[index]
            and masked_flags[index + 1]
            and rng.random() < edit_config.merge_probability
        ):
            targets.append(
                _LinePlanTarget(
                    notation=EXPAND,
                    node_id=body.body_id,
                    depth=body.depth,
                    masked=True,
                )
            )
            index += 2
            continue
        targets.append(
            _LinePlanTarget(
                notation=line_label_for_kind(line.kind),
                node_id=line.node_id,
                depth=line.depth,
                masked=masked_flags[index],
            )
        )
        index += 1
    delete_count = (
        rng.randint(0, edit_config.max_delete)
        if edit_config.max_delete
        else 0
    )
    targets.extend(
        _LinePlanTarget(
            notation=DELETE,
            node_id=body.body_id,
            depth=body.depth,
            masked=True,
        )
        for _ in range(delete_count)
    )
    return targets


def _indent(builder: _Builder, depth: int, node_id: str) -> None:
    builder.add_rule("    " * depth, node_id)


def _render_skeleton_body(
    builder: _Builder, body: Body, *, module_level: bool = False
) -> None:
    previous: LineNode | None = None
    for line in body.lines:
        if module_level and previous is not None and (
            isinstance(previous, FunctionDefinition)
            or isinstance(line, FunctionDefinition)
        ):
            builder.add_rule("\n\n", body.body_id)
        _render_skeleton_line(builder, line)
        previous = line


def _render_skeleton_line(builder: _Builder, line: LineNode) -> None:
    _indent(builder, line.depth, line.node_id)
    if isinstance(line, SimpleStatement):
        builder.add_visible_special(
            "[STMT]", line.node_id, _line_role(line.depth)
        )
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, FunctionDefinition):
        builder.add_rule("def ", line.node_id)
        builder.add_visible_special(HDR, line.node_id, MaskRole.RULE)
        builder.add_rule(":\n", line.node_id)
        _render_skeleton_body(builder, line.body)
        return
    if isinstance(line, IfStatement):
        builder.add_rule("if ", line.node_id)
        builder.add_visible_special(HDR, line.node_id, MaskRole.RULE)
        builder.add_rule(":\n", line.node_id)
        _render_skeleton_body(builder, line.body)
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, ForStatement):
        builder.add_rule("for ", line.node_id)
        builder.add_visible_special(HDR, line.node_id, MaskRole.RULE)
        builder.add_rule(":\n", line.node_id)
        _render_skeleton_body(builder, line.body)
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, WhileStatement):
        builder.add_rule("while ", line.node_id)
        builder.add_visible_special(HDR, line.node_id, MaskRole.RULE)
        builder.add_rule(":\n", line.node_id)
        _render_skeleton_body(builder, line.body)
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    raise RuntimeInvariantError(f"unsupported skeleton line {type(line).__name__}")


def _render_leaf_body(
    builder: _Builder,
    body: Body,
    *,
    rng: random.Random,
    mask_probability: float,
    module_level: bool = False,
    edit_config: TokenEditConfig,
    depth_probabilities: dict[int, float] | None,
) -> None:
    previous: LineNode | None = None
    for line in body.lines:
        if module_level and previous is not None and (
            isinstance(previous, FunctionDefinition)
            or isinstance(line, FunctionDefinition)
        ):
            builder.add_rule("\n\n", body.body_id)
        _render_leaf_line(
            builder,
            line,
            rng=rng,
            mask_probability=mask_probability,
            edit_config=edit_config,
            depth_probabilities=depth_probabilities,
        )
        previous = line


def _render_leaf_line(
    builder: _Builder,
    line: LineNode,
    *,
    rng: random.Random,
    mask_probability: float,
    edit_config: TokenEditConfig,
    depth_probabilities: dict[int, float] | None,
) -> None:
    local_probability = (
        depth_probabilities.get(line.depth, mask_probability)
        if depth_probabilities is not None
        else mask_probability
    )
    if not 0 <= local_probability <= 1:
        raise ValueError(
            f"mask probability for depth {line.depth} must be in [0,1]"
        )
    _indent(builder, line.depth, line.node_id)
    if isinstance(line, SimpleStatement):
        builder.add_content(
            line.text,
            line.node_id,
            MaskRole.TOKEN_STMT,
            mask_probability=local_probability,
            rng=rng,
            edit_config=edit_config,
            collapse_notation="[STMT]",
            collapse_role=_line_role(line.depth),
        )
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, FunctionDefinition):
        builder.add_rule("def ", line.node_id)
        builder.add_content(
            line.header,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=local_probability,
            rng=rng,
            edit_config=edit_config,
            collapse_notation=HDR,
            collapse_role=MaskRole.RULE,
        )
        builder.add_rule(":\n", line.node_id)
        _render_leaf_body(
            builder,
            line.body,
            rng=rng,
            mask_probability=mask_probability,
            edit_config=edit_config,
            depth_probabilities=depth_probabilities,
        )
        return
    if isinstance(line, IfStatement):
        builder.add_rule("if ", line.node_id)
        builder.add_content(
            line.condition,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=local_probability,
            rng=rng,
            edit_config=edit_config,
            collapse_notation=HDR,
            collapse_role=MaskRole.RULE,
        )
        builder.add_rule(":\n", line.node_id)
        _render_leaf_body(
            builder,
            line.body,
            rng=rng,
            mask_probability=mask_probability,
            edit_config=edit_config,
            depth_probabilities=depth_probabilities,
        )
        for clause in line.elif_clauses:
            _indent(builder, clause.depth, clause.node_id)
            builder.add_rule("elif ", clause.node_id)
            builder.add_content(
                clause.condition,
                clause.node_id,
                MaskRole.TOKEN_HDR,
                mask_probability=(
                    depth_probabilities.get(clause.depth, mask_probability)
                    if depth_probabilities is not None
                    else mask_probability
                ),
                rng=rng,
                edit_config=edit_config,
                collapse_notation=HDR,
                collapse_role=MaskRole.RULE,
            )
            builder.add_rule(":\n", clause.node_id)
            _render_leaf_body(
                builder,
                clause.body,
                rng=rng,
                mask_probability=mask_probability,
                edit_config=edit_config,
                depth_probabilities=depth_probabilities,
            )
        if line.else_body:
            _indent(builder, line.depth, line.node_id)
            builder.add_rule("else:\n", line.node_id)
            _render_leaf_body(
                builder,
                line.else_body,
                rng=rng,
                mask_probability=mask_probability,
                edit_config=edit_config,
                depth_probabilities=depth_probabilities,
            )
        return
    if isinstance(line, ForStatement):
        builder.add_rule("for ", line.node_id)
        builder.add_content(
            f"{line.target} in {line.iterator}",
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=local_probability,
            rng=rng,
            edit_config=edit_config,
            collapse_notation=HDR,
            collapse_role=MaskRole.RULE,
        )
        builder.add_rule(":\n", line.node_id)
        _render_leaf_body(
            builder,
            line.body,
            rng=rng,
            mask_probability=mask_probability,
            edit_config=edit_config,
            depth_probabilities=depth_probabilities,
        )
        if line.else_body:
            _indent(builder, line.depth, line.node_id)
            builder.add_rule("else:\n", line.node_id)
            _render_leaf_body(
                builder,
                line.else_body,
                rng=rng,
                mask_probability=mask_probability,
                edit_config=edit_config,
                depth_probabilities=depth_probabilities,
            )
        return
    if isinstance(line, WhileStatement):
        builder.add_rule("while ", line.node_id)
        builder.add_content(
            line.condition,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=local_probability,
            rng=rng,
            edit_config=edit_config,
            collapse_notation=HDR,
            collapse_role=MaskRole.RULE,
        )
        builder.add_rule(":\n", line.node_id)
        _render_leaf_body(
            builder,
            line.body,
            rng=rng,
            mask_probability=mask_probability,
            edit_config=edit_config,
            depth_probabilities=depth_probabilities,
        )
        if line.else_body:
            _indent(builder, line.depth, line.node_id)
            builder.add_rule("else:\n", line.node_id)
            _render_leaf_body(
                builder,
                line.else_body,
                rng=rng,
                mask_probability=mask_probability,
                edit_config=edit_config,
                depth_probabilities=depth_probabilities,
            )
        return
    raise RuntimeInvariantError(f"unsupported leaf line {type(line).__name__}")


def _body_contains(body: Body, target_body_id: str) -> bool:
    if body.body_id == target_body_id:
        return True
    return any(_line_contains(line, target_body_id) for line in body.lines)


def _line_contains(line: LineNode, target_body_id: str) -> bool:
    if isinstance(line, FunctionDefinition):
        return _body_contains(line.body, target_body_id)
    if isinstance(line, IfStatement):
        return (
            _body_contains(line.body, target_body_id)
            or any(
                _body_contains(clause.body, target_body_id)
                for clause in line.elif_clauses
            )
            or (
                line.else_body is not None
                and _body_contains(line.else_body, target_body_id)
            )
        )
    if isinstance(line, (ForStatement, WhileStatement)):
        return _body_contains(line.body, target_body_id) or (
            line.else_body is not None
            and _body_contains(line.else_body, target_body_id)
        )
    return False


def _is_main_body(body: Body, target_body_id: str) -> bool:
    if body.body_id == target_body_id:
        return True
    for line in body.lines:
        if isinstance(line, FunctionDefinition):
            if _is_main_body(line.body, target_body_id):
                return True
        elif isinstance(line, IfStatement):
            if _is_main_body(line.body, target_body_id):
                return True
            # Clause bodies are deliberately excluded in v0.
        elif isinstance(line, (ForStatement, WhileStatement)):
            if _is_main_body(line.body, target_body_id):
                return True
    return False


def _render_body_context(
    builder: _Builder,
    body: Body,
    *,
    target_body_id: str,
    module_level: bool = False,
    edit_config: LineEditConfig,
    rng: random.Random,
    line_mask_probability: float,
) -> None:
    if body.body_id == target_body_id:
        role = MaskRole.LINE_MODULE if module_level else MaskRole.LINE_BODY
        targets = _augment_line_targets(
            body,
            edit_config=edit_config,
            rng=rng,
            mask_probability=line_mask_probability,
        )
        for target in targets:
            _indent(builder, target.depth, target.node_id)
            if target.masked:
                builder.add_mask_target(
                    target.notation, target.node_id, role
                )
            else:
                builder.add_visible_special(
                    target.notation, target.node_id, role
                )
            builder.add_rule("\n", body.body_id)
        return

    for line in body.lines:
        _indent(builder, line.depth, line.node_id)
        if not _line_contains(line, target_body_id):
            builder.add_visible_special(
                line_label_for_kind(line.kind),
                line.node_id,
                _line_role(line.depth),
            )
            builder.add_rule("\n", line.node_id)
            continue
        _render_path_line_after_indent(
            builder,
            line,
            target_body_id=target_body_id,
            edit_config=edit_config,
            rng=rng,
            line_mask_probability=line_mask_probability,
        )


def _render_path_line_after_indent(
    builder: _Builder,
    line: LineNode,
    *,
    target_body_id: str,
    edit_config: LineEditConfig,
    rng: random.Random,
    line_mask_probability: float,
) -> None:
    clean_rng = random.Random(0)
    if isinstance(line, FunctionDefinition):
        builder.add_rule("def ", line.node_id)
        builder.add_content(
            line.header,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=0.0,
            rng=clean_rng,
        )
        builder.add_rule(":\n", line.node_id)
        _render_body_context(
            builder,
            line.body,
            target_body_id=target_body_id,
            edit_config=edit_config,
            rng=rng,
            line_mask_probability=line_mask_probability,
        )
        return
    if isinstance(line, IfStatement) and _body_contains(
        line.body, target_body_id
    ):
        builder.add_rule("if ", line.node_id)
        builder.add_content(
            line.condition,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=0.0,
            rng=clean_rng,
        )
        builder.add_rule(":\n", line.node_id)
        _render_body_context(
            builder,
            line.body,
            target_body_id=target_body_id,
            edit_config=edit_config,
            rng=rng,
            line_mask_probability=line_mask_probability,
        )
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, ForStatement) and _body_contains(
        line.body, target_body_id
    ):
        builder.add_rule("for ", line.node_id)
        builder.add_content(
            f"{line.target} in {line.iterator}",
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=0.0,
            rng=clean_rng,
        )
        builder.add_rule(":\n", line.node_id)
        _render_body_context(
            builder,
            line.body,
            target_body_id=target_body_id,
            edit_config=edit_config,
            rng=rng,
            line_mask_probability=line_mask_probability,
        )
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    if isinstance(line, WhileStatement) and _body_contains(
        line.body, target_body_id
    ):
        builder.add_rule("while ", line.node_id)
        builder.add_content(
            line.condition,
            line.node_id,
            MaskRole.TOKEN_HDR,
            mask_probability=0.0,
            rng=clean_rng,
        )
        builder.add_rule(":\n", line.node_id)
        _render_body_context(
            builder,
            line.body,
            target_body_id=target_body_id,
            edit_config=edit_config,
            rng=rng,
            line_mask_probability=line_mask_probability,
        )
        _indent(builder, line.depth, line.node_id)
        builder.add_visible_special(CLAUSES, line.node_id, MaskRole.RULE)
        builder.add_rule("\n", line.node_id)
        return
    raise RuntimeInvariantError(
        f"target body is not in a supported main-body path for {line.node_id}"
    )
