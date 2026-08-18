"""Mutable tree-aware reverse runtime for Scaffold-Coder decoding."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

from .canvas import TokenRegistry
from .errors import BudgetExceededError, RuntimeInvariantError
from .roles import (
    BODY,
    CLAUSES,
    DELETE,
    ELIF,
    ELSE,
    EXPAND,
    FOR,
    FUNC,
    HDR,
    IF,
    MaskRole,
    STMT,
    WHILE,
)


LINE_LABELS = frozenset({STMT, FUNC, FOR, WHILE, IF})
class HoleState(str, Enum):
    CLOSED = "closed"
    OPEN = "open"


@dataclass(frozen=True, slots=True)
class DecoderConfig:
    initial_root_slots: int = 1
    initial_body_slots: int = 2
    initial_statement_masks: int = 4
    initial_statement_masks_shallow: int | None = None
    statement_shallow_depth: int = 1
    initial_function_header_masks: int = 4
    initial_loop_header_masks: int = 4
    initial_condition_masks: int = 3
    max_canvas_tokens: int = 2048
    max_tree_depth: int = 16
    max_lines_per_body: int = 128
    max_total_lines: int = 1024
    max_tokens_per_hole: int = 512
    max_expansions: int = 1024
    allow_module_expand: bool = True

    def __post_init__(self) -> None:
        integer_fields = (
            "initial_root_slots",
            "initial_body_slots",
            "initial_statement_masks",
            "initial_function_header_masks",
            "initial_loop_header_masks",
            "initial_condition_masks",
            "max_canvas_tokens",
            "max_tree_depth",
            "max_lines_per_body",
            "max_total_lines",
            "max_tokens_per_hole",
        )
        for name in integer_fields:
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if (
            self.initial_statement_masks_shallow is not None
            and self.initial_statement_masks_shallow <= 0
        ):
            raise ValueError(
                "initial_statement_masks_shallow must be positive"
            )
        if self.statement_shallow_depth < 0:
            raise ValueError("statement_shallow_depth must be non-negative")
        if self.max_expansions < 0:
            raise ValueError("max_expansions must be non-negative")
        if self.initial_root_slots > self.max_total_lines:
            raise ValueError(
                "initial_root_slots cannot exceed max_total_lines"
            )
        if self.initial_body_slots > self.max_lines_per_body:
            raise ValueError(
                "initial_body_slots cannot exceed max_lines_per_body"
            )


@dataclass(slots=True)
class TokenCell:
    cell_id: str
    mask_id: str | None
    token_id: int | None = None
    confidence: float | None = None
    committed_at_call: int | None = None
    remask_count: int = 0

    @property
    def unresolved(self) -> bool:
        return self.mask_id is not None


@dataclass(slots=True)
class TokenHole:
    hole_id: str
    owner_id: str
    notation: str
    role: MaskRole
    depth: int
    initial_masks: int
    required: bool = True
    state: HoleState = HoleState.CLOSED
    cells: list[TokenCell] = field(default_factory=list)

    @property
    def resolved(self) -> bool:
        return self.state is HoleState.OPEN and not any(
            cell.unresolved for cell in self.cells
        )


@dataclass(slots=True)
class LineSlot:
    slot_id: str
    depth: int
    mask_id: str | None
    pending_label: str | None = None
    node: "RuntimeLine | None" = None
    confidence: float | None = None
    committed_at_call: int | None = None
    backtrack_count: int = 0

    @property
    def unresolved(self) -> bool:
        return self.mask_id is not None


@dataclass(slots=True)
class BodyHole:
    body_id: str
    owner_id: str
    depth: int
    required: bool = True
    state: HoleState = HoleState.CLOSED
    slots: list[LineSlot] = field(default_factory=list)

    @property
    def plan_resolved(self) -> bool:
        return self.state is HoleState.OPEN and not any(
            slot.unresolved or slot.pending_label is not None
            for slot in self.slots
        )


@dataclass(slots=True)
class ClauseHole:
    hole_id: str
    owner_id: str
    depth: int
    family: str
    state: HoleState = HoleState.CLOSED
    mask_id: str | None = None
    clause: "ClauseNode | None" = None
    deleted: bool = False
    confidence: float | None = None
    committed_at_call: int | None = None
    backtrack_count: int = 0

    @property
    def resolved(self) -> bool:
        if self.deleted:
            return True
        if self.clause is not None:
            return self.clause.complete
        return False


@dataclass(slots=True)
class StatementNode:
    node_id: str
    depth: int
    content: TokenHole

    @property
    def complete(self) -> bool:
        return self.content.resolved


@dataclass(slots=True)
class ConstructNode:
    node_id: str
    kind: str
    depth: int
    header: TokenHole
    body: BodyHole
    clauses: ClauseHole | None

    @property
    def complete(self) -> bool:
        return (
            self.header.resolved
            and _body_complete(self.body)
            and (self.clauses is None or self.clauses.resolved)
        )


@dataclass(slots=True)
class ClauseNode:
    node_id: str
    kind: str
    depth: int
    header: TokenHole | None
    body: BodyHole
    next_clauses: ClauseHole | None

    @property
    def complete(self) -> bool:
        return (
            (self.header is None or self.header.resolved)
            and _body_complete(self.body)
            and (self.next_clauses is None or self.next_clauses.resolved)
        )


RuntimeLine = StatementNode | ConstructNode


@dataclass(frozen=True, slots=True)
class MaskRef:
    mask_id: str
    role: MaskRole
    owner_id: str
    allowed_notations: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class LeafTokenRef:
    """One committed model-produced token that C1 may re-mask."""

    cell_id: str
    hole_id: str
    owner_id: str
    role: MaskRole
    depth: int
    token_id: int
    confidence: float | None
    committed_at_call: int | None
    remask_count: int


@dataclass(frozen=True, slots=True)
class StructuralSubtreeRef:
    """One completed model-created construct eligible for C2 collapse."""

    anchor_id: str
    anchor_type: str
    node_id: str
    kind: str
    depth: int
    confidence: float | None
    committed_at_call: int
    backtrack_count: int
    mean_content_confidence: float
    latest_content_commit_call: int
    content_tokens: int


@dataclass(frozen=True, slots=True)
class DecoderCanvas:
    input_ids: tuple[int, ...]
    roles: tuple[MaskRole, ...]
    owner_ids: tuple[str, ...]
    mask_refs: tuple[MaskRef | None, ...]

    @property
    def mask_positions(self) -> tuple[int, ...]:
        return tuple(
            index for index, ref in enumerate(self.mask_refs) if ref is not None
        )


class _CanvasBuilder:
    def __init__(self, registry: TokenRegistry) -> None:
        self.registry = registry
        self.input_ids: list[int] = []
        self.roles: list[MaskRole] = []
        self.owner_ids: list[str] = []
        self.mask_refs: list[MaskRef | None] = []

    def rule(self, text: str, owner_id: str) -> None:
        for token_id in self.registry.tokenizer.encode(
            text, add_special_tokens=False
        ):
            self.token(token_id, MaskRole.RULE, owner_id)

    def token(self, token_id: int, role: MaskRole, owner_id: str) -> None:
        self.input_ids.append(token_id)
        self.roles.append(role)
        self.owner_ids.append(owner_id)
        self.mask_refs.append(None)

    def special(self, notation: str, role: MaskRole, owner_id: str) -> None:
        self.token(self.registry.special_id(notation), role, owner_id)

    def mask(self, ref: MaskRef) -> None:
        self.input_ids.append(self.registry.tokenizer.mask_token_id)
        self.roles.append(ref.role)
        self.owner_ids.append(ref.owner_id)
        self.mask_refs.append(ref)

    def finish(self) -> DecoderCanvas:
        return DecoderCanvas(
            input_ids=tuple(self.input_ids),
            roles=tuple(self.roles),
            owner_ids=tuple(self.owner_ids),
            mask_refs=tuple(self.mask_refs),
        )


class DecoderRuntime:
    def __init__(
        self,
        registry: TokenRegistry,
        config: DecoderConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or DecoderConfig()
        self._counter = 0
        self.expansions = 0
        self.line_capacity_hits = 0
        self.token_capacity_hits = 0
        self.depth_capacity_hits = 0
        self.total_line_capacity_hits = 0
        self.module_expand_suppressed = 0
        self.expand_budget_hits = 0
        self.root = BodyHole(
            body_id=self._new_id("body"),
            owner_id="module",
            depth=0,
            required=False,
        )
        self._open_body(self.root, self.config.initial_root_slots)
        self.rule_fixed_point()

    @classmethod
    def from_function_header(
        cls,
        registry: TokenRegistry,
        header_text: str,
        config: DecoderConfig | None = None,
    ) -> "DecoderRuntime":
        """Initialize at a committed function header and open body plan."""

        if not header_text.strip():
            raise ValueError("function header text cannot be empty")
        if "\n" in header_text or "\r" in header_text:
            raise ValueError("v0 function header must be single-line")
        runtime = cls(registry, config)
        node = runtime._materialize_line(FUNC, 0)
        if not isinstance(node, ConstructNode):  # pragma: no cover
            raise RuntimeInvariantError("function label did not make a construct")
        token_ids = registry.tokenizer.encode(
            header_text.strip(),
            add_special_tokens=False,
        )
        node.header.state = HoleState.OPEN
        node.header.cells = [
            TokenCell(
                cell_id=runtime._new_id("cell"),
                mask_id=None,
                token_id=token_id,
            )
            for token_id in token_ids
        ]
        runtime.root.slots = [
            LineSlot(
                slot_id=runtime._new_id("slot"),
                depth=0,
                mask_id=None,
                node=node,
            )
        ]
        runtime.rule_fixed_point()
        return runtime

    def _new_id(self, prefix: str) -> str:
        self._counter += 1
        return f"{prefix}-{self._counter:08d}"

    def render(self) -> DecoderCanvas:
        builder = _CanvasBuilder(self.registry)
        self._render_body(builder, self.root, module_level=True)
        canvas = builder.finish()
        if len(canvas.input_ids) > self.config.max_canvas_tokens:
            raise BudgetExceededError(
                f"canvas length {len(canvas.input_ids)} exceeds "
                f"{self.config.max_canvas_tokens}"
            )
        return canvas

    def unresolved_masks(self) -> tuple[MaskRef, ...]:
        return tuple(
            ref for ref in self.render().mask_refs if ref is not None
        )

    def commit(
        self,
        mask_id: str,
        target_id: int,
        *,
        confidence: float | None = None,
        model_call: int | None = None,
    ) -> None:
        location = self._find_mask(mask_id)
        if location is None:
            raise RuntimeInvariantError(f"unknown mask id {mask_id}")
        kind, owner, item = location
        notation = self.registry.id_to_notation.get(target_id)
        if kind == "line":
            self._commit_line(
                owner,
                item,
                notation,
                confidence=confidence,
                model_call=model_call,
            )
        elif kind == "token":
            self._commit_token(
                owner,
                item,
                target_id,
                notation,
                confidence=confidence,
                model_call=model_call,
            )
        elif kind == "clause":
            self._commit_clause(
                owner,
                notation,
                confidence=confidence,
                model_call=model_call,
            )
        else:  # pragma: no cover
            raise RuntimeInvariantError(f"unknown mask location {kind}")
        self.rule_fixed_point()

    def commit_many(
        self,
        predictions: dict[str, int],
        *,
        confidences: dict[str, float] | None = None,
        model_call: int | None = None,
    ) -> None:
        for mask_id, token_id in predictions.items():
            self.commit(
                mask_id,
                token_id,
                confidence=(
                    confidences.get(mask_id)
                    if confidences is not None
                    else None
                ),
                model_call=model_call,
            )

    def committed_leaf_tokens(self) -> tuple[LeafTokenRef, ...]:
        """Return committed lexical cells, excluding all rule-emitted text."""

        refs: list[LeafTokenRef] = []
        _collect_leaf_tokens_from_body(self.root, refs)
        return tuple(refs)

    def remask_leaf(self, cell_id: str) -> str:
        """Apply one legal C1 forward-noising move to a lexical cell."""

        found = _find_token_cell_in_body(self.root, cell_id)
        if found is None:
            raise RuntimeInvariantError(f"unknown leaf cell id {cell_id}")
        hole, cell = found
        if cell.unresolved or cell.token_id is None:
            raise RuntimeInvariantError(
                f"leaf cell {cell_id} is not committed"
            )
        if cell.committed_at_call is None:
            raise RuntimeInvariantError(
                f"leaf cell {cell_id} was emitted by a rule, not the model"
            )
        if hole.state is not HoleState.OPEN:
            raise RuntimeInvariantError(
                f"leaf cell {cell_id} belongs to a closed hole"
            )
        cell.mask_id = self._new_id("mask")
        cell.token_id = None
        cell.confidence = None
        cell.committed_at_call = None
        cell.remask_count += 1
        self.rule_fixed_point()
        return cell.mask_id

    def remask_leaves(self, cell_ids: Iterable[str]) -> tuple[str, ...]:
        return tuple(self.remask_leaf(cell_id) for cell_id in cell_ids)

    def completed_structural_subtrees(
        self,
    ) -> tuple[StructuralSubtreeRef, ...]:
        """Return completed constructs with model-confidence provenance."""

        refs: list[StructuralSubtreeRef] = []
        _collect_structural_subtrees_from_body(self.root, refs)
        return tuple(refs)

    def backtrack_structural_subtree(self, anchor_id: str) -> str:
        """Apply C2: collapse an expanded construct/clause to its mask."""

        found_slot = _find_line_slot_in_body(self.root, anchor_id)
        if found_slot is not None:
            _, slot = found_slot
            if not isinstance(slot.node, ConstructNode):
                raise RuntimeInvariantError(
                    f"slot {anchor_id} does not contain a construct"
                )
            if not slot.node.complete:
                raise RuntimeInvariantError(
                    f"construct in slot {anchor_id} is not complete"
                )
            if slot.committed_at_call is None:
                raise RuntimeInvariantError(
                    f"construct in slot {anchor_id} was not committed by the model"
                )
            slot.node = None
            slot.pending_label = None
            slot.mask_id = self._new_id("mask")
            slot.confidence = None
            slot.committed_at_call = None
            slot.backtrack_count += 1
            self.rule_fixed_point()
            return slot.mask_id

        hole = _find_clause_hole_in_body(self.root, anchor_id)
        if hole is None:
            raise RuntimeInvariantError(
                f"unknown structural anchor id {anchor_id}"
            )
        if hole.clause is None or not hole.clause.complete:
            raise RuntimeInvariantError(
                f"clause in hole {anchor_id} is not complete"
            )
        if hole.committed_at_call is None:
            raise RuntimeInvariantError(
                f"clause in hole {anchor_id} was not committed by the model"
            )
        hole.clause = None
        hole.deleted = False
        hole.state = HoleState.OPEN
        hole.mask_id = self._new_id("mask")
        hole.confidence = None
        hole.committed_at_call = None
        hole.backtrack_count += 1
        self.rule_fixed_point()
        return hole.mask_id

    def rule_fixed_point(self) -> None:
        for _ in range(1000):
            changed = self._advance_body(self.root)
            if not changed:
                return
        raise RuntimeInvariantError("rule fixed point did not converge")

    @property
    def complete(self) -> bool:
        self.rule_fixed_point()
        return _body_complete(self.root) and not self.unresolved_masks()

    def final_text(self) -> str:
        if not self.complete:
            raise RuntimeInvariantError("decoder runtime is not complete")
        canvas = self.render()
        special_ids = set(self.registry.notation_to_id.values())
        if any(token_id in special_ids for token_id in canvas.input_ids):
            raise RuntimeInvariantError("final canvas still contains special tokens")
        return self.registry.tokenizer.decode(canvas.input_ids)

    def placeholder_text(self) -> str:
        """Render unresolved state as parseable Python for process metrics."""

        return self._placeholder_body(self.root, module_level=True)

    def total_line_slots(self) -> int:
        return _count_line_slots_in_body(self.root)

    def maximum_runtime_depth(self) -> int:
        return _maximum_depth_in_body(self.root)

    def maximum_body_line_slots(self) -> int:
        return _maximum_body_size(self.root)

    def maximum_token_hole_size(self) -> int:
        return _maximum_token_hole_size_in_body(self.root)

    def capacity_metrics(self) -> dict[str, int]:
        return {
            "line_capacity_hits": self.line_capacity_hits,
            "token_capacity_hits": self.token_capacity_hits,
            "depth_capacity_hits": self.depth_capacity_hits,
            "total_line_capacity_hits": self.total_line_capacity_hits,
            "module_expand_suppressed": self.module_expand_suppressed,
            "expand_budget_hits": self.expand_budget_hits,
            "maximum_tree_depth": self.maximum_runtime_depth(),
            "maximum_total_lines": self.total_line_slots(),
            "maximum_body_lines": self.maximum_body_line_slots(),
            "maximum_tokens_per_hole": self.maximum_token_hole_size(),
        }

    def _open_body(self, body: BodyHole, count: int) -> None:
        if body.depth > self.config.max_tree_depth:
            self.depth_capacity_hits += 1
            raise BudgetExceededError("tree depth budget exceeded")
        if count > self.config.max_lines_per_body:
            self.line_capacity_hits += 1
            raise BudgetExceededError("initial body line budget exceeded")
        if self.total_line_slots() + count > self.config.max_total_lines:
            self.total_line_capacity_hits += 1
            raise BudgetExceededError("total line budget exceeded")
        body.state = HoleState.OPEN
        body.slots = [self._new_line_slot(body.depth) for _ in range(count)]

    def _new_line_slot(self, depth: int) -> LineSlot:
        return LineSlot(
            slot_id=self._new_id("slot"),
            depth=depth,
            mask_id=self._new_id("mask"),
        )

    def _new_token_cell(self) -> TokenCell:
        return TokenCell(
            cell_id=self._new_id("cell"),
            mask_id=self._new_id("mask"),
        )

    def _open_token_hole(self, hole: TokenHole) -> None:
        if hole.initial_masks > self.config.max_tokens_per_hole:
            self.token_capacity_hits += 1
            raise BudgetExceededError(
                "initial token-hole budget exceeded"
            )
        hole.state = HoleState.OPEN
        hole.cells = [
            self._new_token_cell() for _ in range(hole.initial_masks)
        ]

    def _advance_body(self, body: BodyHole) -> bool:
        if body.state is HoleState.CLOSED:
            return False
        changed = False
        if not any(slot.unresolved for slot in body.slots):
            for slot in body.slots:
                if slot.pending_label is not None:
                    slot.node = self._materialize_line(
                        slot.pending_label, slot.depth
                    )
                    slot.pending_label = None
                    changed = True
        if body.required and not body.slots:
            body.slots.append(self._pass_slot(body.depth))
            changed = True

        if body.plan_resolved:
            for slot in body.slots:
                if slot.node is not None:
                    changed |= self._advance_line(slot.node)
        return changed

    def _advance_line(self, node: RuntimeLine) -> bool:
        if isinstance(node, StatementNode):
            if node.content.state is HoleState.CLOSED:
                self._open_token_hole(node.content)
                return True
            return False

        changed = False
        if node.header.state is HoleState.CLOSED:
            self._open_token_hole(node.header)
            changed = True
        if node.header.resolved:
            if node.body.state is HoleState.CLOSED:
                self._open_body(node.body, self.config.initial_body_slots)
                changed = True
            if node.clauses is not None and node.clauses.state is HoleState.CLOSED:
                node.clauses.state = HoleState.OPEN
                node.clauses.mask_id = self._new_id("mask")
                changed = True
        changed |= self._advance_body(node.body)
        if node.clauses is not None:
            changed |= self._advance_clause_hole(node.clauses)
        return changed

    def _advance_clause_hole(self, hole: ClauseHole) -> bool:
        if hole.clause is None:
            return False
        clause = hole.clause
        changed = False
        if clause.header is not None and clause.header.state is HoleState.CLOSED:
            self._open_token_hole(clause.header)
            changed = True
        header_ready = clause.header is None or clause.header.resolved
        if header_ready and clause.body.state is HoleState.CLOSED:
            self._open_body(clause.body, self.config.initial_body_slots)
            changed = True
        changed |= self._advance_body(clause.body)
        if (
            clause.next_clauses is not None
            and header_ready
            and clause.next_clauses.state is HoleState.CLOSED
        ):
            clause.next_clauses.state = HoleState.OPEN
            clause.next_clauses.mask_id = self._new_id("mask")
            changed = True
        if clause.next_clauses is not None:
            changed |= self._advance_clause_hole(clause.next_clauses)
        return changed

    def _materialize_line(self, notation: str, depth: int) -> RuntimeLine:
        if notation == STMT:
            statement_masks = self.config.initial_statement_masks
            if (
                self.config.initial_statement_masks_shallow is not None
                and depth <= self.config.statement_shallow_depth
            ):
                statement_masks = (
                    self.config.initial_statement_masks_shallow
                )
            return StatementNode(
                node_id=self._new_id("stmt"),
                depth=depth,
                content=TokenHole(
                    hole_id=self._new_id("hole"),
                    owner_id="stmt",
                    notation=STMT,
                    role=MaskRole.TOKEN_STMT,
                    depth=depth,
                    initial_masks=statement_masks,
                ),
            )
        if notation not in {FUNC, FOR, WHILE, IF}:
            raise RuntimeInvariantError(f"cannot materialize line {notation}")
        node_id = self._new_id(notation.strip("[]").lower())
        if notation == FUNC:
            header_masks = self.config.initial_function_header_masks
            clauses = None
        elif notation == FOR:
            header_masks = self.config.initial_loop_header_masks
            clauses = ClauseHole(
                hole_id=self._new_id("clauses"),
                owner_id=node_id,
                depth=depth,
                family="loop",
            )
        else:
            header_masks = self.config.initial_condition_masks
            clauses = ClauseHole(
                hole_id=self._new_id("clauses"),
                owner_id=node_id,
                depth=depth,
                family="if" if notation == IF else "loop",
            )
        return ConstructNode(
            node_id=node_id,
            kind=notation,
            depth=depth,
            header=TokenHole(
                hole_id=self._new_id("hole"),
                owner_id=node_id,
                notation=HDR,
                role=MaskRole.TOKEN_HDR,
                depth=depth,
                initial_masks=header_masks,
            ),
            body=BodyHole(
                body_id=self._new_id("body"),
                owner_id=node_id,
                depth=depth + 1,
                required=True,
            ),
            clauses=clauses,
        )

    def _pass_slot(self, depth: int) -> LineSlot:
        token_ids = self.registry.tokenizer.encode(
            "pass", add_special_tokens=False
        )
        hole = TokenHole(
            hole_id=self._new_id("hole"),
            owner_id="pass",
            notation=STMT,
            role=MaskRole.TOKEN_STMT,
            depth=depth,
            initial_masks=0,
            state=HoleState.OPEN,
            cells=[
                TokenCell(
                    cell_id=self._new_id("cell"),
                    mask_id=None,
                    token_id=token_id,
                )
                for token_id in token_ids
            ],
        )
        return LineSlot(
            slot_id=self._new_id("slot"),
            depth=depth,
            mask_id=None,
            node=StatementNode(self._new_id("stmt"), depth, hole),
        )

    def _commit_line(
        self,
        body: BodyHole,
        slot: LineSlot,
        notation: str | None,
        *,
        confidence: float | None,
        model_call: int | None,
    ) -> None:
        if notation == EXPAND:
            if (
                body.depth == 0
                and not self.config.allow_module_expand
            ):
                self.module_expand_suppressed += 1
                raise RuntimeInvariantError(
                    "module-level expansion is disabled"
                )
            if len(body.slots) + 1 > self.config.max_lines_per_body:
                self.line_capacity_hits += 1
                raise BudgetExceededError("body line budget exceeded")
            if self.total_line_slots() + 1 > self.config.max_total_lines:
                self.total_line_capacity_hits += 1
                raise BudgetExceededError("total line budget exceeded")
            self._charge_expansion()
            index = body.slots.index(slot)
            body.slots[index : index + 1] = [
                self._new_line_slot(slot.depth),
                self._new_line_slot(slot.depth),
            ]
        elif notation == DELETE:
            body.slots.remove(slot)
        elif notation in LINE_LABELS:
            slot.mask_id = None
            slot.pending_label = notation
            slot.confidence = confidence
            slot.committed_at_call = model_call
        else:
            raise RuntimeInvariantError(
                f"illegal line-level prediction {notation!r}"
            )
        if len(body.slots) > self.config.max_lines_per_body:
            self.line_capacity_hits += 1
            raise BudgetExceededError("body line budget exceeded")

    def _commit_token(
        self,
        hole: TokenHole,
        cell: TokenCell,
        target_id: int,
        notation: str | None,
        *,
        confidence: float | None,
        model_call: int | None,
    ) -> None:
        index = hole.cells.index(cell)
        if notation == EXPAND:
            if len(hole.cells) + 1 > self.config.max_tokens_per_hole:
                self.token_capacity_hits += 1
                raise BudgetExceededError("token-hole budget exceeded")
            self._charge_expansion()
            hole.cells[index : index + 1] = [
                self._new_token_cell(),
                self._new_token_cell(),
            ]
        elif notation == DELETE:
            if len(hole.cells) == 1 and hole.required:
                raise RuntimeInvariantError(
                    "cannot delete the final position of a required token hole"
                )
            hole.cells.pop(index)
        elif notation is not None or target_id == self.registry.tokenizer.mask_token_id:
            raise RuntimeInvariantError(
                f"illegal token-level prediction {notation or target_id}"
            )
        else:
            cell.mask_id = None
            cell.token_id = target_id
            cell.confidence = confidence
            cell.committed_at_call = model_call
        if len(hole.cells) > self.config.max_tokens_per_hole:
            self.token_capacity_hits += 1
            raise BudgetExceededError("token-hole budget exceeded")

    def _commit_clause(
        self,
        hole: ClauseHole,
        notation: str | None,
        *,
        confidence: float | None,
        model_call: int | None,
    ) -> None:
        legal = {DELETE, ELSE}
        if hole.family == "if":
            legal.add(ELIF)
        if notation not in legal:
            raise RuntimeInvariantError(
                f"illegal clause prediction {notation!r} for {hole.family}"
            )
        hole.mask_id = None
        hole.confidence = confidence
        hole.committed_at_call = model_call
        if notation == DELETE:
            hole.deleted = True
            return
        node_id = self._new_id(notation.strip("[]").lower())
        if notation == ELSE:
            header = None
            next_clauses = None
        else:
            header = TokenHole(
                hole_id=self._new_id("hole"),
                owner_id=node_id,
                notation=HDR,
                role=MaskRole.TOKEN_HDR,
                depth=hole.depth,
                initial_masks=self.config.initial_condition_masks,
            )
            next_clauses = ClauseHole(
                hole_id=self._new_id("clauses"),
                owner_id=node_id,
                depth=hole.depth,
                family="if",
            )
        hole.clause = ClauseNode(
            node_id=node_id,
            kind=notation,
            depth=hole.depth,
            header=header,
            body=BodyHole(
                body_id=self._new_id("body"),
                owner_id=node_id,
                depth=hole.depth + 1,
                required=True,
            ),
            next_clauses=next_clauses,
        )

    def _charge_expansion(self) -> None:
        if self.expansions >= self.config.max_expansions:
            self.expand_budget_hits += 1
            raise BudgetExceededError("expansion budget exceeded")
        self.expansions += 1

    def _find_mask(self, mask_id: str):
        return _find_mask_in_body(self.root, mask_id)

    def _render_body(
        self,
        builder: _CanvasBuilder,
        body: BodyHole,
        *,
        module_level: bool = False,
    ) -> None:
        if body.state is HoleState.CLOSED:
            builder.rule("    " * body.depth, body.body_id)
            builder.special(BODY, MaskRole.RULE, body.body_id)
            builder.rule("\n", body.body_id)
            return
        previous: RuntimeLine | None = None
        for slot in body.slots:
            if (
                module_level
                and previous is not None
                and (
                    isinstance(previous, ConstructNode)
                    and previous.kind == FUNC
                    or isinstance(slot.node, ConstructNode)
                    and slot.node.kind == FUNC
                )
            ):
                builder.rule("\n\n", body.body_id)
            if slot.unresolved:
                allowed = set(LINE_LABELS | {DELETE})
                line_room = (
                    len(body.slots) < self.config.max_lines_per_body
                    and self.total_line_slots() < self.config.max_total_lines
                    and self.expansions < self.config.max_expansions
                )
                if module_level and not self.config.allow_module_expand:
                    line_room = False
                if line_room:
                    allowed.add(EXPAND)
                builder.rule("    " * slot.depth, slot.slot_id)
                builder.mask(
                    MaskRef(
                        mask_id=slot.mask_id,
                        role=(
                            MaskRole.LINE_MODULE
                            if slot.depth == 0
                            else MaskRole.LINE_BODY
                        ),
                        owner_id=slot.slot_id,
                        allowed_notations=tuple(sorted(allowed)),
                    )
                )
                builder.rule("\n", slot.slot_id)
            elif slot.pending_label is not None:
                builder.rule("    " * slot.depth, slot.slot_id)
                builder.special(
                    slot.pending_label,
                    MaskRole.LINE_MODULE
                    if slot.depth == 0
                    else MaskRole.LINE_BODY,
                    slot.slot_id,
                )
                builder.rule("\n", slot.slot_id)
            elif slot.node is not None:
                self._render_line(builder, slot.node)
                previous = slot.node

    def _render_line(
        self, builder: _CanvasBuilder, node: RuntimeLine
    ) -> None:
        builder.rule("    " * node.depth, node.node_id)
        if isinstance(node, StatementNode):
            self._render_token_hole(builder, node.content)
            builder.rule("\n", node.node_id)
            return
        keyword = {
            FUNC: "def ",
            FOR: "for ",
            WHILE: "while ",
            IF: "if ",
        }[node.kind]
        builder.rule(keyword, node.node_id)
        self._render_token_hole(builder, node.header)
        builder.rule(":\n", node.node_id)
        self._render_body(builder, node.body)
        if node.clauses is not None:
            self._render_clause_hole(builder, node.clauses)

    def _render_token_hole(
        self, builder: _CanvasBuilder, hole: TokenHole
    ) -> None:
        if hole.state is HoleState.CLOSED:
            builder.special(hole.notation, MaskRole.RULE, hole.hole_id)
            return
        for cell in hole.cells:
            if cell.unresolved:
                allowed: list[str] = []
                if (
                    len(hole.cells) < self.config.max_tokens_per_hole
                    and self.expansions < self.config.max_expansions
                ):
                    allowed.append(EXPAND)
                if not (len(hole.cells) == 1 and hole.required):
                    allowed.append(DELETE)
                builder.mask(
                    MaskRef(
                        mask_id=cell.mask_id,
                        role=hole.role,
                        owner_id=hole.hole_id,
                        allowed_notations=tuple(allowed),
                    )
                )
            else:
                builder.token(cell.token_id, hole.role, hole.hole_id)

    def _render_clause_hole(
        self, builder: _CanvasBuilder, hole: ClauseHole
    ) -> None:
        if hole.deleted:
            return
        if hole.state is HoleState.CLOSED:
            builder.rule("    " * hole.depth, hole.hole_id)
            builder.special(CLAUSES, MaskRole.RULE, hole.hole_id)
            builder.rule("\n", hole.hole_id)
            return
        if hole.mask_id is not None:
            allowed = {DELETE, ELSE}
            if hole.family == "if":
                allowed.add(ELIF)
            builder.rule("    " * hole.depth, hole.hole_id)
            builder.mask(
                MaskRef(
                    mask_id=hole.mask_id,
                    role=MaskRole.LINE_CLAUSE,
                    owner_id=hole.hole_id,
                    allowed_notations=tuple(sorted(allowed)),
                )
            )
            builder.rule("\n", hole.hole_id)
            return
        if hole.clause is not None:
            self._render_clause(builder, hole.clause)

    def _render_clause(
        self, builder: _CanvasBuilder, clause: ClauseNode
    ) -> None:
        builder.rule("    " * clause.depth, clause.node_id)
        if clause.kind == ELSE:
            builder.rule("else:\n", clause.node_id)
        else:
            builder.rule("elif ", clause.node_id)
            self._render_token_hole(builder, clause.header)
            builder.rule(":\n", clause.node_id)
        self._render_body(builder, clause.body)
        if clause.next_clauses is not None:
            self._render_clause_hole(builder, clause.next_clauses)

    def _placeholder_body(
        self, body: BodyHole, *, module_level: bool = False
    ) -> str:
        if body.state is HoleState.CLOSED or not body.slots:
            return "    " * body.depth + "pass\n"
        pieces: list[str] = []
        previous_kind: str | None = None
        for slot in body.slots:
            if slot.node is not None:
                piece, kind = self._placeholder_line(slot.node)
            elif slot.pending_label is not None:
                piece, kind = self._placeholder_pending(
                    slot.pending_label, slot.depth
                )
            else:
                piece, kind = ("    " * slot.depth + "pass\n", STMT)
            if (
                module_level
                and pieces
                and (previous_kind == FUNC or kind == FUNC)
            ):
                pieces.append("\n\n")
            pieces.append(piece)
            previous_kind = kind
        return "".join(pieces)

    def _placeholder_pending(
        self, notation: str, depth: int
    ) -> tuple[str, str]:
        indent = "    " * depth
        child = "    " * (depth + 1) + "pass\n"
        if notation == FUNC:
            return f"{indent}def generated():\n{child}", FUNC
        if notation == FOR:
            return f"{indent}for x in []:\n{child}", FOR
        if notation == WHILE:
            return f"{indent}while False:\n{child}", WHILE
        if notation == IF:
            return f"{indent}if True:\n{child}", IF
        return f"{indent}pass\n", STMT

    def _placeholder_line(
        self, node: RuntimeLine
    ) -> tuple[str, str]:
        indent = "    " * node.depth
        if isinstance(node, StatementNode):
            if node.content.resolved:
                text = self._resolved_hole_text(node.content)
                return f"{indent}{text or 'pass'}\n", STMT
            return f"{indent}pass\n", STMT

        header = (
            self._resolved_hole_text(node.header)
            if node.header.resolved
            else {
                FUNC: "generated()",
                FOR: "x in []",
                WHILE: "False",
                IF: "True",
            }[node.kind]
        )
        keyword = {FUNC: "def ", FOR: "for ", WHILE: "while ", IF: "if "}[node.kind]
        body = self._placeholder_body(node.body)
        clauses = (
            self._placeholder_clause_hole(node.clauses)
            if node.clauses is not None
            else ""
        )
        return f"{indent}{keyword}{header}:\n{body}{clauses}", node.kind

    def _placeholder_clause_hole(self, hole: ClauseHole) -> str:
        if hole.deleted or hole.clause is None:
            return ""
        clause = hole.clause
        indent = "    " * clause.depth
        if clause.kind == ELSE:
            head = f"{indent}else:\n"
        else:
            header = (
                self._resolved_hole_text(clause.header)
                if clause.header is not None and clause.header.resolved
                else "True"
            )
            head = f"{indent}elif {header}:\n"
        body = self._placeholder_body(clause.body)
        tail = (
            self._placeholder_clause_hole(clause.next_clauses)
            if clause.next_clauses is not None
            else ""
        )
        return head + body + tail

    def _resolved_hole_text(self, hole: TokenHole) -> str:
        return self.registry.tokenizer.decode(
            [cell.token_id for cell in hole.cells if cell.token_id is not None]
        )


def _body_complete(body: BodyHole) -> bool:
    return (
        body.state is HoleState.OPEN
        and body.plan_resolved
        and all(slot.node is not None and slot.node.complete for slot in body.slots)
    )


def _count_line_slots_in_body(body: BodyHole) -> int:
    if body.state is HoleState.CLOSED:
        return 0
    total = len(body.slots)
    for slot in body.slots:
        node = slot.node
        if not isinstance(node, ConstructNode):
            continue
        total += _count_line_slots_in_body(node.body)
        if node.clauses is not None:
            total += _count_line_slots_in_clause(node.clauses)
    return total


def _count_line_slots_in_clause(hole: ClauseHole) -> int:
    if hole.clause is None:
        return 0
    total = _count_line_slots_in_body(hole.clause.body)
    if hole.clause.next_clauses is not None:
        total += _count_line_slots_in_clause(hole.clause.next_clauses)
    return total


def _maximum_depth_in_body(body: BodyHole) -> int:
    maximum = body.depth if body.state is HoleState.OPEN else 0
    if body.state is HoleState.CLOSED:
        return maximum
    for slot in body.slots:
        node = slot.node
        if node is None:
            maximum = max(maximum, slot.depth)
            continue
        maximum = max(maximum, node.depth)
        if isinstance(node, ConstructNode):
            maximum = max(maximum, _maximum_depth_in_body(node.body))
            if node.clauses is not None:
                maximum = max(
                    maximum,
                    _maximum_depth_in_clause(node.clauses),
                )
    return maximum


def _maximum_depth_in_clause(hole: ClauseHole) -> int:
    if hole.clause is None:
        return hole.depth
    maximum = max(
        hole.clause.depth,
        _maximum_depth_in_body(hole.clause.body),
    )
    if hole.clause.next_clauses is not None:
        maximum = max(
            maximum,
            _maximum_depth_in_clause(hole.clause.next_clauses),
        )
    return maximum


def _maximum_body_size(body: BodyHole) -> int:
    if body.state is HoleState.CLOSED:
        return 0
    maximum = len(body.slots)
    for slot in body.slots:
        node = slot.node
        if not isinstance(node, ConstructNode):
            continue
        maximum = max(maximum, _maximum_body_size(node.body))
        if node.clauses is not None:
            maximum = max(
                maximum,
                _maximum_body_size_in_clause(node.clauses),
            )
    return maximum


def _maximum_body_size_in_clause(hole: ClauseHole) -> int:
    if hole.clause is None:
        return 0
    maximum = _maximum_body_size(hole.clause.body)
    if hole.clause.next_clauses is not None:
        maximum = max(
            maximum,
            _maximum_body_size_in_clause(hole.clause.next_clauses),
        )
    return maximum


def _maximum_token_hole_size_in_body(body: BodyHole) -> int:
    if body.state is HoleState.CLOSED:
        return 0
    maximum = 0
    for slot in body.slots:
        if slot.node is not None:
            maximum = max(
                maximum,
                _maximum_token_hole_size_in_line(slot.node),
            )
    return maximum


def _maximum_token_hole_size_in_line(node: RuntimeLine) -> int:
    if isinstance(node, StatementNode):
        return len(node.content.cells)
    maximum = len(node.header.cells)
    maximum = max(maximum, _maximum_token_hole_size_in_body(node.body))
    if node.clauses is not None:
        maximum = max(
            maximum,
            _maximum_token_hole_size_in_clause(node.clauses),
        )
    return maximum


def _maximum_token_hole_size_in_clause(hole: ClauseHole) -> int:
    if hole.clause is None:
        return 0
    maximum = (
        len(hole.clause.header.cells)
        if hole.clause.header is not None
        else 0
    )
    maximum = max(
        maximum,
        _maximum_token_hole_size_in_body(hole.clause.body),
    )
    if hole.clause.next_clauses is not None:
        maximum = max(
            maximum,
            _maximum_token_hole_size_in_clause(
                hole.clause.next_clauses
            ),
        )
    return maximum


def _find_mask_in_body(body: BodyHole, mask_id: str):
    if body.state is HoleState.CLOSED:
        return None
    for slot in body.slots:
        if slot.mask_id == mask_id:
            return ("line", body, slot)
        if slot.node is not None:
            found = _find_mask_in_line(slot.node, mask_id)
            if found is not None:
                return found
    return None


def _find_mask_in_line(node: RuntimeLine, mask_id: str):
    if isinstance(node, StatementNode):
        return _find_mask_in_token_hole(node.content, mask_id)
    found = _find_mask_in_token_hole(node.header, mask_id)
    if found is not None:
        return found
    found = _find_mask_in_body(node.body, mask_id)
    if found is not None:
        return found
    if node.clauses is not None:
        return _find_mask_in_clause_hole(node.clauses, mask_id)
    return None


def _find_mask_in_token_hole(hole: TokenHole, mask_id: str):
    if hole.state is HoleState.CLOSED:
        return None
    for cell in hole.cells:
        if cell.mask_id == mask_id:
            return ("token", hole, cell)
    return None


def _find_mask_in_clause_hole(hole: ClauseHole, mask_id: str):
    if hole.mask_id == mask_id:
        return ("clause", hole, hole)
    if hole.clause is None:
        return None
    clause = hole.clause
    if clause.header is not None:
        found = _find_mask_in_token_hole(clause.header, mask_id)
        if found is not None:
            return found
    found = _find_mask_in_body(clause.body, mask_id)
    if found is not None:
        return found
    if clause.next_clauses is not None:
        return _find_mask_in_clause_hole(clause.next_clauses, mask_id)
    return None


def _collect_leaf_tokens_from_body(
    body: BodyHole, output: list[LeafTokenRef]
) -> None:
    if body.state is HoleState.CLOSED:
        return
    for slot in body.slots:
        if slot.node is not None:
            _collect_leaf_tokens_from_line(slot.node, output)


def _collect_structural_subtrees_from_body(
    body: BodyHole,
    output: list[StructuralSubtreeRef],
) -> None:
    if body.state is HoleState.CLOSED:
        return
    for slot in body.slots:
        node = slot.node
        if node is None:
            continue
        if isinstance(node, ConstructNode):
            _collect_structural_subtrees_from_body(node.body, output)
            if node.clauses is not None:
                _collect_structural_subtrees_from_clause(
                    node.clauses,
                    output,
                )
            if node.complete and slot.committed_at_call is not None:
                leaves: list[LeafTokenRef] = []
                _collect_leaf_tokens_from_line(node, leaves)
                confident = [
                    leaf
                    for leaf in leaves
                    if (
                        leaf.confidence is not None
                        and leaf.committed_at_call is not None
                    )
                ]
                if confident:
                    output.append(
                        StructuralSubtreeRef(
                            anchor_id=slot.slot_id,
                            anchor_type="line",
                            node_id=node.node_id,
                            kind=node.kind,
                            depth=node.depth,
                            confidence=slot.confidence,
                            committed_at_call=slot.committed_at_call,
                            backtrack_count=slot.backtrack_count,
                            mean_content_confidence=(
                                sum(
                                    float(leaf.confidence)
                                    for leaf in confident
                                )
                                / len(confident)
                            ),
                            latest_content_commit_call=max(
                                int(leaf.committed_at_call)
                                for leaf in confident
                            ),
                            content_tokens=len(confident),
                        )
                    )


def _collect_structural_subtrees_from_clause(
    hole: ClauseHole,
    output: list[StructuralSubtreeRef],
) -> None:
    if hole.clause is None:
        return
    clause = hole.clause
    _collect_structural_subtrees_from_body(clause.body, output)
    if clause.next_clauses is not None:
        _collect_structural_subtrees_from_clause(
            clause.next_clauses,
            output,
        )
    if clause.complete and hole.committed_at_call is not None:
        leaves: list[LeafTokenRef] = []
        _collect_leaf_tokens_from_clause(hole, leaves)
        confident = [
            leaf
            for leaf in leaves
            if (
                leaf.confidence is not None
                and leaf.committed_at_call is not None
            )
        ]
        if confident:
            output.append(
                StructuralSubtreeRef(
                    anchor_id=hole.hole_id,
                    anchor_type="clause",
                    node_id=clause.node_id,
                    kind=clause.kind,
                    depth=clause.depth,
                    confidence=hole.confidence,
                    committed_at_call=hole.committed_at_call,
                    backtrack_count=hole.backtrack_count,
                    mean_content_confidence=(
                        sum(float(leaf.confidence) for leaf in confident)
                        / len(confident)
                    ),
                    latest_content_commit_call=max(
                        int(leaf.committed_at_call)
                        for leaf in confident
                    ),
                    content_tokens=len(confident),
                )
            )


def _collect_leaf_tokens_from_line(
    node: RuntimeLine, output: list[LeafTokenRef]
) -> None:
    if isinstance(node, StatementNode):
        _collect_leaf_tokens_from_hole(node.content, output)
        return
    _collect_leaf_tokens_from_hole(node.header, output)
    _collect_leaf_tokens_from_body(node.body, output)
    if node.clauses is not None:
        _collect_leaf_tokens_from_clause(node.clauses, output)


def _collect_leaf_tokens_from_clause(
    hole: ClauseHole, output: list[LeafTokenRef]
) -> None:
    if hole.clause is None:
        return
    clause = hole.clause
    if clause.header is not None:
        _collect_leaf_tokens_from_hole(clause.header, output)
    _collect_leaf_tokens_from_body(clause.body, output)
    if clause.next_clauses is not None:
        _collect_leaf_tokens_from_clause(clause.next_clauses, output)


def _collect_leaf_tokens_from_hole(
    hole: TokenHole, output: list[LeafTokenRef]
) -> None:
    if hole.state is HoleState.CLOSED:
        return
    for cell in hole.cells:
        # Rule-emitted required-suite ``pass`` cells have no model-call
        # provenance and must never become C1 correction candidates.
        if (
            cell.unresolved
            or cell.token_id is None
            or cell.committed_at_call is None
        ):
            continue
        output.append(
            LeafTokenRef(
                cell_id=cell.cell_id,
                hole_id=hole.hole_id,
                owner_id=hole.owner_id,
                role=hole.role,
                depth=hole.depth,
                token_id=cell.token_id,
                confidence=cell.confidence,
                committed_at_call=cell.committed_at_call,
                remask_count=cell.remask_count,
            )
        )


def _find_token_cell_in_body(
    body: BodyHole, cell_id: str
) -> tuple[TokenHole, TokenCell] | None:
    if body.state is HoleState.CLOSED:
        return None
    for slot in body.slots:
        if slot.node is None:
            continue
        found = _find_token_cell_in_line(slot.node, cell_id)
        if found is not None:
            return found
    return None


def _find_line_slot_in_body(
    body: BodyHole,
    slot_id: str,
) -> tuple[BodyHole, LineSlot] | None:
    if body.state is HoleState.CLOSED:
        return None
    for slot in body.slots:
        if slot.slot_id == slot_id:
            return (body, slot)
        node = slot.node
        if not isinstance(node, ConstructNode):
            continue
        found = _find_line_slot_in_body(node.body, slot_id)
        if found is not None:
            return found
        if node.clauses is not None:
            found = _find_line_slot_in_clause(node.clauses, slot_id)
            if found is not None:
                return found
    return None


def _find_line_slot_in_clause(
    hole: ClauseHole,
    slot_id: str,
) -> tuple[BodyHole, LineSlot] | None:
    if hole.clause is None:
        return None
    clause = hole.clause
    found = _find_line_slot_in_body(clause.body, slot_id)
    if found is not None:
        return found
    if clause.next_clauses is not None:
        return _find_line_slot_in_clause(clause.next_clauses, slot_id)
    return None


def _find_clause_hole_in_body(
    body: BodyHole,
    hole_id: str,
) -> ClauseHole | None:
    if body.state is HoleState.CLOSED:
        return None
    for slot in body.slots:
        node = slot.node
        if not isinstance(node, ConstructNode):
            continue
        if node.clauses is not None:
            found = _find_clause_hole(node.clauses, hole_id)
            if found is not None:
                return found
        found = _find_clause_hole_in_body(node.body, hole_id)
        if found is not None:
            return found
    return None


def _find_clause_hole(
    hole: ClauseHole,
    hole_id: str,
) -> ClauseHole | None:
    if hole.hole_id == hole_id:
        return hole
    if hole.clause is None:
        return None
    found = _find_clause_hole_in_body(hole.clause.body, hole_id)
    if found is not None:
        return found
    if hole.clause.next_clauses is not None:
        return _find_clause_hole(hole.clause.next_clauses, hole_id)
    return None


def _find_token_cell_in_line(
    node: RuntimeLine, cell_id: str
) -> tuple[TokenHole, TokenCell] | None:
    if isinstance(node, StatementNode):
        return _find_token_cell_in_hole(node.content, cell_id)
    found = _find_token_cell_in_hole(node.header, cell_id)
    if found is not None:
        return found
    found = _find_token_cell_in_body(node.body, cell_id)
    if found is not None:
        return found
    if node.clauses is not None:
        return _find_token_cell_in_clause(node.clauses, cell_id)
    return None


def _find_token_cell_in_clause(
    hole: ClauseHole, cell_id: str
) -> tuple[TokenHole, TokenCell] | None:
    if hole.clause is None:
        return None
    clause = hole.clause
    if clause.header is not None:
        found = _find_token_cell_in_hole(clause.header, cell_id)
        if found is not None:
            return found
    found = _find_token_cell_in_body(clause.body, cell_id)
    if found is not None:
        return found
    if clause.next_clauses is not None:
        return _find_token_cell_in_clause(clause.next_clauses, cell_id)
    return None


def _find_token_cell_in_hole(
    hole: TokenHole, cell_id: str
) -> tuple[TokenHole, TokenCell] | None:
    if hole.state is HoleState.CLOSED:
        return None
    for cell in hole.cells:
        if cell.cell_id == cell_id:
            return (hole, cell)
    return None
