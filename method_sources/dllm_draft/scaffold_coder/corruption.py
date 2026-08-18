"""Sampled deterministic-rung mixture with DreamOn-style edit augmentation."""

from __future__ import annotations

import collections
import random
from dataclasses import dataclass, field
from enum import Enum

import torch

from .canvas import (
    CanvasState,
    LineEditConfig,
    TokenEditConfig,
    TokenRegistry,
    build_body_plan,
    build_leaf_infill,
    build_root_plan,
    iter_main_bodies,
    prepend_chat_prompt,
)
from .ir import (
    Body,
    ForStatement,
    FunctionDefinition,
    IfStatement,
    Module,
    WhileStatement,
    iter_lines,
)
from .roles import DELETE, MaskRole


class Rung(str, Enum):
    ROOT_PLAN = "root_plan"
    BODY_PLAN = "body_plan"
    LEAF_INFILL = "leaf_infill"
    MIXED = "mixed"


@dataclass(frozen=True, slots=True)
class RungMixtureConfig:
    root_probability: float = 0.20
    body_probability: float = 0.30
    leaf_probability: float = 0.50
    leaf_u_min: float = 0.05
    leaf_u_max: float = 1.0
    token_merge_base_probability: float = 0.5
    line_merge_probability: float = 0.5
    static_merge_mix_probability: float = 0.5
    # DreamOn applies a large delete range to one FIM middle. Applying that
    # range independently to every Scaffold hole would make delete targets
    # dominate, so the multi-hole v0 default is deliberately conservative.
    max_token_delete: int = 1
    max_line_delete: int = 1
    weighting: str = "mdlm"
    maximum_weight: float = 20.0
    normalize_within_sample: bool = True

    def validate(self) -> None:
        probabilities = (
            self.root_probability,
            self.body_probability,
            self.leaf_probability,
        )
        if any(value < 0 for value in probabilities):
            raise ValueError("rung probabilities must be non-negative")
        if abs(sum(probabilities) - 1.0) > 1e-8:
            raise ValueError("rung probabilities must sum to one")
        if not 0 < self.leaf_u_min <= self.leaf_u_max <= 1:
            raise ValueError("leaf local-clock range must lie in (0,1]")
        if self.weighting not in {"unit", "mdlm", "dreamon_linear"}:
            raise ValueError(f"unknown weighting mode {self.weighting!r}")


@dataclass(frozen=True, slots=True)
class SampledTrainingState:
    state: CanvasState
    rung: Rung
    local_u: float
    global_t_proxy: float
    loss_weights: tuple[float, ...]
    metadata: dict[str, object]

    def to_tensors(self) -> dict[str, torch.Tensor]:
        tensors = self.state.to_tensors()
        tensors["loss_weights"] = torch.tensor(
            self.loss_weights, dtype=torch.float32
        )
        tensors["t"] = torch.tensor(self.global_t_proxy, dtype=torch.float32)
        tensors["local_u"] = torch.tensor(self.local_u, dtype=torch.float32)
        return tensors


def _make_loss_weights(
    state: CanvasState,
    registry: TokenRegistry,
    *,
    base: float,
    normalize_within_sample: bool,
) -> tuple[float, ...]:
    delete_id = registry.special_id(DELETE)
    delete_count = sum(
        supervised and target_id == delete_id
        for target_id, supervised in zip(
            state.labels, state.loss_mask, strict=True
        )
    )
    weights: list[float] = []
    for target_id, supervised in zip(
        state.labels, state.loss_mask, strict=True
    ):
        if not supervised:
            weights.append(0.0)
        elif target_id == delete_id and delete_count:
            weights.append(base / delete_count)
        else:
            weights.append(base)
    if normalize_within_sample:
        total = sum(weights)
        if total > 0:
            scale = base / total
            weights = [weight * scale for weight in weights]
    return tuple(weights)


def _target_metadata(
    state: CanvasState, registry: TokenRegistry
) -> tuple[dict[str, int], dict[str, int]]:
    target_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    for target_id, supervised, role in zip(
        state.labels, state.loss_mask, state.roles, strict=True
    ):
        if not supervised:
            continue
        notation = registry.id_to_notation.get(target_id, "LEXICAL")
        target_counts[notation] += 1
        role_counts[role.value] += 1
    return dict(target_counts), dict(role_counts)


def _node_depth_map(module: Module) -> dict[str, int]:
    mapping = {module.node_id: 0}

    def visit_body(body: Body) -> None:
        mapping[body.body_id] = body.depth
        for line in body.lines:
            mapping[line.node_id] = line.depth
            if isinstance(line, FunctionDefinition):
                visit_body(line.body)
            elif isinstance(line, IfStatement):
                visit_body(line.body)
                for clause in line.elif_clauses:
                    mapping[clause.node_id] = clause.depth
                    visit_body(clause.body)
                if line.else_body:
                    visit_body(line.else_body)
            elif isinstance(line, (ForStatement, WhileStatement)):
                visit_body(line.body)
                if line.else_body:
                    visit_body(line.else_body)

    visit_body(module.body)
    return mapping


def _per_position_band_weights(
    state: CanvasState,
    registry: TokenRegistry,
    *,
    module: Module,
    schedule: "HierarchicalBandSchedule",
    t: float,
    phase: str,
    maximum_weight: float,
    normalize_within_sample: bool,
) -> tuple[tuple[float, ...], list[float]]:
    depth_map = _node_depth_map(module)
    bases: list[float] = []
    raw_weights: list[float] = []
    delete_id = registry.special_id(DELETE)
    delete_count = sum(
        supervised and target_id == delete_id
        for target_id, supervised in zip(
            state.labels, state.loss_mask, strict=True
        )
    )
    for node_id, target_id, supervised in zip(
        state.node_ids, state.labels, state.loss_mask, strict=True
    ):
        if not supervised:
            raw_weights.append(0.0)
            continue
        depth = min(schedule.max_depth, depth_map.get(node_id, 0))
        if phase == "content":
            band = schedule.content_bands[depth]
        elif phase == "structural":
            band = schedule.structural_bands[depth]
        else:
            band = schedule.final_root_band
        u = max(band.clock(t), 1e-8)
        derivative = max(band.derivative(t), 1e-8)
        base = min(maximum_weight, derivative / u)
        bases.append(base)
        weight = base / delete_count if target_id == delete_id and delete_count else base
        raw_weights.append(weight)

    if normalize_within_sample and bases:
        desired_mass = sum(bases) / len(bases)
        current_mass = sum(raw_weights)
        if current_mass > 0:
            scale = desired_mass / current_mass
            raw_weights = [weight * scale for weight in raw_weights]
    return tuple(raw_weights), bases


class RungMixtureSampler:
    def __init__(
        self,
        registry: TokenRegistry,
        config: RungMixtureConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or RungMixtureConfig()
        self.config.validate()

    def sample(
        self,
        module: Module,
        prompt: str,
        *,
        seed: int,
    ) -> SampledTrainingState:
        rng = random.Random(seed)
        draw = rng.random()
        if draw < self.config.root_probability:
            rung = Rung.ROOT_PLAN
        elif draw < self.config.root_probability + self.config.body_probability:
            rung = Rung.BODY_PLAN
        else:
            rung = Rung.LEAF_INFILL

        if rung is Rung.ROOT_PLAN:
            local_u = 1.0
            response = build_root_plan(
                module,
                self.registry,
                edit_config=LineEditConfig(
                    merge_probability=self.config.line_merge_probability,
                    max_delete=self.config.max_line_delete,
                ),
                seed=rng.randrange(2**63),
            )
            global_t = 0.90
            body_id = module.body.body_id
            merge_probability = self.config.line_merge_probability
        elif rung is Rung.BODY_PLAN:
            local_u = 1.0
            candidates = list(iter_main_bodies(module))
            if len(candidates) > 1:
                candidates = candidates[1:]
            body = rng.choice(candidates)
            response = build_body_plan(
                module,
                self.registry,
                target_body_id=body.body_id,
                edit_config=LineEditConfig(
                    merge_probability=self.config.line_merge_probability,
                    max_delete=self.config.max_line_delete,
                ),
                seed=rng.randrange(2**63),
            )
            global_t = 0.65
            body_id = body.body_id
            merge_probability = self.config.line_merge_probability
        else:
            local_u = rng.uniform(
                self.config.leaf_u_min, self.config.leaf_u_max
            )
            if rng.random() < self.config.static_merge_mix_probability:
                merge_probability = self.config.token_merge_base_probability
                merge_mode = "static"
            else:
                merge_probability = (
                    self.config.token_merge_base_probability * (1 - local_u)
                )
                merge_mode = "dynamic_inverse"
            response = build_leaf_infill(
                module,
                self.registry,
                mask_probability=local_u,
                seed=rng.randrange(2**63),
                edit_config=TokenEditConfig(
                    merge_probability=merge_probability,
                    max_delete=self.config.max_token_delete,
                    collapse_fully_masked=True,
                ),
            )
            global_t = 0.45 * local_u
            body_id = None

        state = prepend_chat_prompt(response, self.registry, prompt)
        weights = self._loss_weights(state, local_u)
        target_counts, role_counts = _target_metadata(state, self.registry)
        metadata = {
            "seed": seed,
            "rung": rung.value,
            "body_id": body_id,
            "merge_probability": merge_probability,
            "target_counts": target_counts,
            "role_counts": role_counts,
        }
        if rung is Rung.LEAF_INFILL:
            metadata["merge_mode"] = merge_mode
        return SampledTrainingState(
            state=state,
            rung=rung,
            local_u=local_u,
            global_t_proxy=global_t,
            loss_weights=weights,
            metadata=metadata,
        )

    def _loss_weights(
        self, state: CanvasState, local_u: float
    ) -> tuple[float, ...]:
        if self.config.weighting == "unit":
            base = 1.0
        elif self.config.weighting == "dreamon_linear":
            base = 1.0 - local_u
        else:
            base = min(
                self.config.maximum_weight,
                1.0 / max(local_u, 1e-8),
            )

        return _make_loss_weights(
            state,
            self.registry,
            base=base,
            normalize_within_sample=self.config.normalize_within_sample,
        )


@dataclass(frozen=True, slots=True)
class Band:
    start: float
    end: float

    def __post_init__(self) -> None:
        if not 0 <= self.start < self.end <= 1:
            raise ValueError(f"invalid band [{self.start}, {self.end}]")

    def clock(self, t: float) -> float:
        return min(1.0, max(0.0, (t - self.start) / (self.end - self.start)))

    def derivative(self, t: float) -> float:
        return 1.0 / (self.end - self.start) if self.start < t < self.end else 0.0

    @property
    def midpoint(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True, slots=True)
class HierarchicalBandConfig:
    content_end: float = 0.45
    structural_end: float = 0.95
    content_width: float = 0.30
    content_start_spread: float = 0.15
    structural_overlap_fraction: float = 0.15
    depth_cap: int = 6
    token_merge_base_probability: float = 0.5
    static_merge_mix_probability: float = 0.5
    line_merge_probability: float = 0.5
    max_token_delete: int = 1
    max_line_delete: int = 1
    maximum_weight: float = 20.0
    normalize_within_sample: bool = True
    region_collapse_mode: str = "all_mask"
    region_collapse_exponent: float = 1.0

    def validate(self) -> None:
        if not 0 < self.content_end < self.structural_end < 1:
            raise ValueError("expected 0 < content_end < structural_end < 1")
        if self.content_width <= 0:
            raise ValueError("content_width must be positive")
        if not 0 <= self.structural_overlap_fraction < 1:
            raise ValueError("structural overlap must lie in [0,1)")
        if self.depth_cap < 0:
            raise ValueError("depth_cap must be non-negative")
        if self.region_collapse_mode not in {"all_mask", "coupled"}:
            raise ValueError("region collapse mode must be all_mask or coupled")
        if self.region_collapse_exponent <= 0:
            raise ValueError("region collapse exponent must be positive")


@dataclass(frozen=True, slots=True)
class HierarchicalBandSchedule:
    max_depth: int
    content_bands: dict[int, Band]
    structural_bands: dict[int, Band]
    final_root_band: Band
    config: HierarchicalBandConfig

    @classmethod
    def build(
        cls, max_depth: int, config: HierarchicalBandConfig
    ) -> "HierarchicalBandSchedule":
        config.validate()
        effective_max = min(max_depth, config.depth_cap)
        denominator = max(1, effective_max)
        content_bands: dict[int, Band] = {}
        for depth in range(max_depth + 1):
            effective_depth = min(depth, effective_max)
            start = (
                config.content_start_spread
                * (effective_max - effective_depth)
                / denominator
            )
            end = min(config.content_end, start + config.content_width)
            if end <= start:
                end = min(config.content_end, start + 1e-6)
            content_bands[depth] = Band(start, end)

        structural_depths = list(range(max_depth, -1, -1))
        width = (
            config.structural_end - config.content_end
        ) / len(structural_depths)
        overlap = width * config.structural_overlap_fraction
        structural_bands: dict[int, Band] = {}
        for index, depth in enumerate(structural_depths):
            base_start = config.content_end + index * width
            base_end = base_start + width
            start = max(config.content_end, base_start - overlap / 2)
            end = min(config.structural_end, base_end + overlap / 2)
            structural_bands[depth] = Band(start, end)

        return cls(
            max_depth=max_depth,
            content_bands=content_bands,
            structural_bands=structural_bands,
            final_root_band=Band(config.structural_end, 1.0),
            config=config,
        )

    def content_clocks(self, t: float) -> dict[int, float]:
        return {depth: band.clock(t) for depth, band in self.content_bands.items()}

    def structural_clocks(self, t: float) -> dict[int, float]:
        return {
            depth: band.clock(t) for depth, band in self.structural_bands.items()
        }


class GlobalBandSampler:
    """A reachable-state v0 implementation of one global t and depth bands.

    The content phase masks all expanded leaves with depth-specific clocks. The
    structural phase samples one active body transition at the depth selected by
    the global structural bands. This is mode-matched for strict/local-body
    decoding; cross-subtree desynchronization is intentionally deferred.
    """

    def __init__(
        self,
        registry: TokenRegistry,
        config: HierarchicalBandConfig | None = None,
    ) -> None:
        self.registry = registry
        self.config = config or HierarchicalBandConfig()
        self.config.validate()

    def sample(
        self,
        module: Module,
        prompt: str,
        *,
        seed: int,
        t: float | None = None,
    ) -> SampledTrainingState:
        rng = random.Random(seed)
        global_t = rng.random() if t is None else t
        if not 0 <= global_t <= 1:
            raise ValueError("global t must lie in [0,1]")

        content_depths = [line.depth for line in iter_lines(module.body)]
        body_candidates = list(iter_main_bodies(module))
        body_depths = [body.depth for body in body_candidates]
        max_depth = max(content_depths + body_depths + [0])
        schedule = HierarchicalBandSchedule.build(max_depth, self.config)

        if global_t < self.config.content_end:
            phase = "content"
            rung = Rung.LEAF_INFILL
            clocks = schedule.content_clocks(global_t)
            depth_probabilities = {
                depth: clocks.get(depth, 0.0)
                for depth in set(content_depths)
            }
            active = [
                (depth, schedule.content_bands[depth])
                for depth, value in depth_probabilities.items()
                if 0 < value < 1
            ]
            if active:
                active_depth, active_band = max(
                    active, key=lambda item: item[1].derivative(global_t)
                )
            else:
                active_depth = max(depth_probabilities, default=0)
                active_band = schedule.content_bands[active_depth]
            local_u = max(depth_probabilities.values(), default=1e-6)
            local_u = max(local_u, 1e-6)
            if rng.random() < self.config.static_merge_mix_probability:
                merge_mode = "static"
                merge_probability = self.config.token_merge_base_probability
            else:
                merge_mode = "dynamic_inverse"
                merge_probability = (
                    self.config.token_merge_base_probability * (1 - local_u)
                )
            response = build_leaf_infill(
                module,
                self.registry,
                mask_probability=0.0,
                depth_probabilities=depth_probabilities,
                seed=rng.randrange(2**63),
                edit_config=TokenEditConfig(
                    merge_probability=merge_probability,
                    max_delete=self.config.max_token_delete,
                    collapse_fully_masked=(
                        self.config.region_collapse_mode == "all_mask"
                    ),
                    coupled_collapse_exponent=(
                        self.config.region_collapse_exponent
                        if self.config.region_collapse_mode == "coupled"
                        else None
                    ),
                ),
            )
            derivative = active_band.derivative(global_t)
            body_id = None
            selected_depth = active_depth
            all_clocks = depth_probabilities
        elif global_t < self.config.structural_end:
            phase = "structural"
            clocks = schedule.structural_clocks(global_t)
            existing_depths = sorted(set(body_depths), reverse=True)
            active_depths = [
                depth
                for depth in existing_depths
                if schedule.structural_bands[depth].derivative(global_t) > 0
            ]
            if active_depths:
                selected_depth = min(
                    active_depths,
                    key=lambda depth: abs(clocks[depth] - 0.5),
                )
            else:
                selected_depth = min(
                    existing_depths,
                    key=lambda depth: abs(
                        global_t - schedule.structural_bands[depth].midpoint
                    ),
                )
            candidates = [
                body for body in body_candidates if body.depth == selected_depth
            ]
            body = rng.choice(candidates)
            local_u = max(clocks[selected_depth], 1e-6)
            response = build_body_plan(
                module,
                self.registry,
                target_body_id=body.body_id,
                line_mask_probability=local_u,
                edit_config=LineEditConfig(
                    merge_probability=self.config.line_merge_probability,
                    max_delete=self.config.max_line_delete,
                ),
                seed=rng.randrange(2**63),
            )
            rung = (
                Rung.ROOT_PLAN
                if body.body_id == module.body.body_id
                else Rung.BODY_PLAN
            )
            derivative = schedule.structural_bands[selected_depth].derivative(
                global_t
            )
            body_id = body.body_id
            merge_probability = self.config.line_merge_probability
            merge_mode = "line"
            all_clocks = clocks
        else:
            phase = "root"
            rung = Rung.ROOT_PLAN
            local_u = max(schedule.final_root_band.clock(global_t), 1e-6)
            response = build_root_plan(
                module,
                self.registry,
                line_mask_probability=local_u,
                edit_config=LineEditConfig(
                    merge_probability=self.config.line_merge_probability,
                    max_delete=self.config.max_line_delete,
                ),
                seed=rng.randrange(2**63),
            )
            derivative = schedule.final_root_band.derivative(global_t)
            body_id = module.body.body_id
            selected_depth = 0
            merge_probability = self.config.line_merge_probability
            merge_mode = "line"
            all_clocks = {0: local_u}

        state = prepend_chat_prompt(response, self.registry, prompt)
        weights, position_bases = _per_position_band_weights(
            state,
            self.registry,
            module=module,
            schedule=schedule,
            t=global_t,
            phase=phase,
            maximum_weight=self.config.maximum_weight,
            normalize_within_sample=self.config.normalize_within_sample,
        )
        base_weight = (
            sum(position_bases) / len(position_bases)
            if position_bases
            else 0.0
        )
        target_counts, role_counts = _target_metadata(state, self.registry)
        return SampledTrainingState(
            state=state,
            rung=rung,
            local_u=local_u,
            global_t_proxy=global_t,
            loss_weights=weights,
            metadata={
                "seed": seed,
                "rung": rung.value,
                "global_t": global_t,
                "selected_depth": selected_depth,
                "body_id": body_id,
                "merge_probability": merge_probability,
                "merge_mode": merge_mode,
                "clocks": all_clocks,
                "base_weight": base_weight,
                "base_weight_min": min(position_bases) if position_bases else 0.0,
                "base_weight_max": max(position_bases) if position_bases else 0.0,
                "target_counts": target_counts,
                "role_counts": role_counts,
            },
        )
