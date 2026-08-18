"""Confidence-based neural decoding over the mutable Scaffold runtime."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch

from .canvas import TokenRegistry
from .decoder_runtime import (
    DecoderCanvas,
    DecoderRuntime,
    LeafTokenRef,
    MaskRef,
    StructuralSubtreeRef,
)
from .errors import BudgetExceededError, RuntimeInvariantError
from .roles import (
    DELETE,
    EXPAND,
    MaskRole,
    V0_CLAUSE_LABELS,
    V0_CONSTRUCT_LABELS,
)
from .tokenizer_utils import edit_source_token_ids


PredictionProvider = Callable[
    [DecoderRuntime, DecoderCanvas, tuple[MaskRef, ...]],
    dict[str, tuple[int, float]],
]


@dataclass(frozen=True, slots=True)
class SamplerConfig:
    max_model_calls: int = 1024
    transfer_tokens: int = 1
    temperature: float = 0.0
    confidence: str = "normalized_entropy"
    keep_history: bool = True
    structural_confidence_threshold: float | None = None
    structural_max_defer_calls: int = 0
    leaf_remask_fraction: float = 0.0
    leaf_remask_interval: int = 0
    leaf_remask_at_completion: bool = True
    leaf_remask_confidence_threshold: float | None = None
    leaf_remask_min_age_calls: int = 1
    max_leaf_remasks: int = 0
    max_leaf_remasks_per_token: int = 1
    structural_backtrack_confidence_threshold: float | None = None
    structural_backtrack_min_age_calls: int = 1
    max_structural_backtracks: int = 0
    max_structural_backtracks_per_anchor: int = 1
    body_construct_logit_penalty: float = 0.0
    body_stmt_logit_bonus: float = 0.0
    token_expand_logit_bonus: float = 0.0
    break_edit_cycles: bool = True

    def __post_init__(self) -> None:
        if self.max_model_calls <= 0:
            raise ValueError("max_model_calls must be positive")
        if self.transfer_tokens <= 0:
            raise ValueError("transfer_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if self.confidence not in {"normalized_entropy", "margin"}:
            raise ValueError(f"unsupported confidence score {self.confidence!r}")
        if self.structural_max_defer_calls < 0:
            raise ValueError("structural_max_defer_calls must be non-negative")
        if not 0 <= self.leaf_remask_fraction <= 1:
            raise ValueError("leaf_remask_fraction must be in [0,1]")
        if self.leaf_remask_interval < 0:
            raise ValueError("leaf_remask_interval must be non-negative")
        if self.leaf_remask_min_age_calls < 0:
            raise ValueError("leaf_remask_min_age_calls must be non-negative")
        if self.max_leaf_remasks < 0:
            raise ValueError("max_leaf_remasks must be non-negative")
        if self.max_leaf_remasks_per_token < 0:
            raise ValueError(
                "max_leaf_remasks_per_token must be non-negative"
            )
        if self.structural_backtrack_min_age_calls < 0:
            raise ValueError(
                "structural_backtrack_min_age_calls must be non-negative"
            )
        if self.max_structural_backtracks < 0:
            raise ValueError(
                "max_structural_backtracks must be non-negative"
            )
        if self.max_structural_backtracks_per_anchor < 0:
            raise ValueError(
                "max_structural_backtracks_per_anchor must be non-negative"
            )
        if self.body_construct_logit_penalty < 0:
            raise ValueError(
                "body_construct_logit_penalty must be non-negative"
            )
        if self.body_stmt_logit_bonus < 0:
            raise ValueError("body_stmt_logit_bonus must be non-negative")
        if self.token_expand_logit_bonus < 0:
            raise ValueError("token_expand_logit_bonus must be non-negative")
        if (
            self.max_leaf_remasks > 0
            and self.leaf_remask_fraction <= 0
        ):
            raise ValueError(
                "positive max_leaf_remasks requires leaf_remask_fraction > 0"
            )
        if (
            self.max_structural_backtracks > 0
            and self.structural_backtrack_confidence_threshold is None
        ):
            raise ValueError(
                "positive max_structural_backtracks requires "
                "structural_backtrack_confidence_threshold"
            )
        if (
            self.max_structural_backtracks > 0
            and self.max_structural_backtracks_per_anchor <= 0
        ):
            raise ValueError(
                "positive max_structural_backtracks requires a positive "
                "max_structural_backtracks_per_anchor"
            )


@dataclass(frozen=True, slots=True)
class ScaffoldGenerationResult:
    text: str
    model_calls: int
    history: tuple[str, ...]
    final_canvas_tokens: int
    expansions: int
    model_canvas_lengths: tuple[int, ...]
    cumulative_model_tokens: int
    placeholder_history: tuple[str, ...]
    leaf_remasks: int = 0
    correction_rounds: int = 0
    structural_deferrals: int = 0
    structural_backtracks: int = 0
    edit_cycle_breaks: int = 0
    line_capacity_hits: int = 0
    token_capacity_hits: int = 0
    depth_capacity_hits: int = 0
    total_line_capacity_hits: int = 0
    module_expand_suppressed: int = 0
    expand_budget_hits: int = 0
    maximum_tree_depth: int = 0
    maximum_total_lines: int = 0
    maximum_body_lines: int = 0
    maximum_tokens_per_hole: int = 0
    termination_reason: str = "resolved"


@dataclass(frozen=True, slots=True)
class ScaffoldFailureMetrics:
    """Partial decode cost captured immediately before an exception escapes."""

    model_calls: int
    model_canvas_lengths: tuple[int, ...]
    cumulative_model_tokens: int
    expansions: int
    leaf_remasks: int
    correction_rounds: int
    structural_deferrals: int
    structural_backtracks: int
    edit_cycle_breaks: int
    line_capacity_hits: int
    token_capacity_hits: int
    depth_capacity_hits: int
    total_line_capacity_hits: int
    module_expand_suppressed: int
    expand_budget_hits: int
    maximum_tree_depth: int
    maximum_total_lines: int
    maximum_body_lines: int
    maximum_tokens_per_hole: int

    def to_dict(self) -> dict[str, object]:
        return {
            "nfe": self.model_calls,
            "model_canvas_lengths": list(self.model_canvas_lengths),
            "minimum_model_canvas_tokens": (
                min(self.model_canvas_lengths)
                if self.model_canvas_lengths
                else None
            ),
            "maximum_model_canvas_tokens": (
                max(self.model_canvas_lengths)
                if self.model_canvas_lengths
                else None
            ),
            "cumulative_model_tokens": self.cumulative_model_tokens,
            "expansions": self.expansions,
            "leaf_remasks": self.leaf_remasks,
            "correction_rounds": self.correction_rounds,
            "structural_deferrals": self.structural_deferrals,
            "structural_backtracks": self.structural_backtracks,
            "edit_cycle_breaks": self.edit_cycle_breaks,
            "line_capacity_hits": self.line_capacity_hits,
            "token_capacity_hits": self.token_capacity_hits,
            "depth_capacity_hits": self.depth_capacity_hits,
            "total_line_capacity_hits": self.total_line_capacity_hits,
            "module_expand_suppressed": self.module_expand_suppressed,
            "expand_budget_hits": self.expand_budget_hits,
            "maximum_tree_depth": self.maximum_tree_depth,
            "maximum_total_lines": self.maximum_total_lines,
            "maximum_body_lines": self.maximum_body_lines,
            "maximum_tokens_per_hole": self.maximum_tokens_per_hole,
        }


class ScaffoldModelSampler:
    def __init__(
        self,
        model,
        registry: TokenRegistry,
        config: SamplerConfig | None = None,
    ) -> None:
        self.model = model
        self.registry = registry
        self.config = config or SamplerConfig()
        self._token_allowed_cpu: torch.Tensor | None = None
        self.last_failure_metrics: ScaffoldFailureMetrics | None = None

    def generate(
        self,
        prompt: str,
        *,
        runtime: DecoderRuntime | None = None,
        prediction_provider: PredictionProvider | None = None,
    ) -> ScaffoldGenerationResult:
        runtime = runtime or DecoderRuntime(self.registry)
        self.last_failure_metrics = None
        prompt_ids = self._prompt_ids(prompt)
        history: list[str] = []
        placeholder_history: list[str] = []
        model_canvas_lengths: list[int] = []
        model_calls = 0
        leaf_remasks = 0
        correction_rounds = 0
        structural_deferrals = 0
        structural_backtracks = 0
        edit_cycle_breaks = 0
        deferred_by_mask: dict[str, int] = {}
        last_remask_call: int | None = None
        edit_source_fingerprints: set[
            tuple[tuple[int, ...], tuple[MaskRole, ...]]
        ] = set()

        try:
            while True:
                runtime.rule_fixed_point()
                backtracked = self._maybe_backtrack_structure(
                    runtime,
                    model_calls=model_calls,
                    backtracks_used=structural_backtracks,
                )
                if backtracked is not None:
                    structural_backtracks += 1
                    correction_rounds += 1
                    deferred_by_mask.clear()
                    edit_source_fingerprints.clear()
                    continue
                if runtime.complete:
                    remasked = (
                        self._maybe_remask_leaves(
                            runtime,
                            model_calls=model_calls,
                            remasks_used=leaf_remasks,
                            at_completion=True,
                        )
                        if model_calls < self.config.max_model_calls
                        else ()
                    )
                    if remasked:
                        leaf_remasks += len(remasked)
                        correction_rounds += 1
                        last_remask_call = model_calls
                        continue
                    text = runtime.final_text()
                    capacity = runtime.capacity_metrics()
                    return ScaffoldGenerationResult(
                        text=text,
                        model_calls=model_calls,
                        history=tuple(history),
                        final_canvas_tokens=len(runtime.render().input_ids),
                        expansions=runtime.expansions,
                        model_canvas_lengths=tuple(model_canvas_lengths),
                        cumulative_model_tokens=sum(model_canvas_lengths),
                        placeholder_history=tuple(placeholder_history),
                        leaf_remasks=leaf_remasks,
                        correction_rounds=correction_rounds,
                        structural_deferrals=structural_deferrals,
                        structural_backtracks=structural_backtracks,
                        edit_cycle_breaks=edit_cycle_breaks,
                        **capacity,
                        termination_reason="resolved",
                    )

                if (
                    self.config.leaf_remask_interval > 0
                    and model_calls > 0
                    and model_calls % self.config.leaf_remask_interval == 0
                    and last_remask_call != model_calls
                ):
                    remasked = self._maybe_remask_leaves(
                        runtime,
                        model_calls=model_calls,
                        remasks_used=leaf_remasks,
                        at_completion=False,
                    )
                    if remasked:
                        leaf_remasks += len(remasked)
                        correction_rounds += 1
                        last_remask_call = model_calls

                if model_calls >= self.config.max_model_calls:
                    raise BudgetExceededError(
                        "generation exceeded "
                        f"{self.config.max_model_calls} model calls"
                    )
                canvas = runtime.render()
                fingerprint = (canvas.input_ids, canvas.roles)
                suppress_edits = (
                    self.config.break_edit_cycles
                    and fingerprint in edit_source_fingerprints
                )
                if suppress_edits:
                    edit_cycle_breaks += 1
                model_canvas_lengths.append(
                    len(prompt_ids) + len(canvas.input_ids) + 1
                )
                refs = tuple(
                    ref for ref in canvas.mask_refs if ref is not None
                )
                if not refs:
                    raise RuntimeInvariantError(
                        "runtime is incomplete but exposes no eligible masks"
                    )
                if prediction_provider is None:
                    candidates = self._model_predictions(
                        prompt_ids,
                        canvas,
                        refs,
                        suppress_edits=suppress_edits,
                    )
                else:
                    candidates = prediction_provider(runtime, canvas, refs)
                if not candidates:
                    raise RuntimeInvariantError(
                        "prediction provider returned no candidates"
                    )

                ranked = sorted(
                    candidates.items(),
                    key=lambda item: item[1][1],
                    reverse=True,
                )
                refs_by_mask = {ref.mask_id: ref for ref in refs}
                selected: list[tuple[str, tuple[int, float]]] = []
                seen_deferred: set[str] = set()
                for mask_id, token_and_confidence in ranked:
                    ref = refs_by_mask.get(mask_id)
                    if ref is None:
                        raise RuntimeInvariantError(
                            f"candidate references unknown mask {mask_id}"
                        )
                    target_id, confidence = token_and_confidence
                    if self._defer_structural_commit(
                        ref,
                        target_id,
                        confidence,
                        deferred_by_mask,
                    ):
                        structural_deferrals += 1
                        seen_deferred.add(mask_id)
                        continue
                    deferred_by_mask.pop(mask_id, None)
                    selected.append((mask_id, token_and_confidence))
                    if len(selected) >= self.config.transfer_tokens:
                        break

                live_masks = set(refs_by_mask)
                for mask_id in tuple(deferred_by_mask):
                    if (
                        mask_id not in live_masks
                        or mask_id not in seen_deferred
                    ):
                        deferred_by_mask.pop(mask_id, None)

                for mask_id, (target_id, confidence) in selected:
                    if self._is_elastic_edit(target_id):
                        edit_source_fingerprints.add(fingerprint)
                    runtime.commit(
                        mask_id,
                        target_id,
                        confidence=confidence,
                        model_call=model_calls,
                    )
                # A forward happened even when all candidates were deferred.
                # Increment after successful commits to preserve the call
                # index stored on committed tokens.
                model_calls += 1
                if self.config.keep_history:
                    history.append(
                        self.registry.tokenizer.decode(
                            runtime.render().input_ids
                        )
                    )
                    placeholder_history.append(runtime.placeholder_text())
        except Exception as exc:
            # Capacity errors may be raised while applying predictions from a
            # completed forward. Other errors (including rule-only failures at
            # loop entry) must not be charged an extra NFE.
            if (
                isinstance(exc, BudgetExceededError)
                and len(model_canvas_lengths) > model_calls
            ):
                model_calls += 1
            capacity = runtime.capacity_metrics()
            self.last_failure_metrics = ScaffoldFailureMetrics(
                model_calls=model_calls,
                model_canvas_lengths=tuple(model_canvas_lengths),
                cumulative_model_tokens=sum(model_canvas_lengths),
                expansions=runtime.expansions,
                leaf_remasks=leaf_remasks,
                correction_rounds=correction_rounds,
                structural_deferrals=structural_deferrals,
                structural_backtracks=structural_backtracks,
                edit_cycle_breaks=edit_cycle_breaks,
                **capacity,
            )
            raise

    def _defer_structural_commit(
        self,
        ref: MaskRef,
        target_id: int,
        confidence: float,
        deferred_by_mask: dict[str, int],
    ) -> bool:
        threshold = self.config.structural_confidence_threshold
        if threshold is None:
            return False
        notation = self.registry.id_to_notation.get(target_id)
        if notation not in V0_CONSTRUCT_LABELS | V0_CLAUSE_LABELS:
            return False
        if confidence >= threshold:
            return False
        deferred = deferred_by_mask.get(ref.mask_id, 0)
        if deferred >= self.config.structural_max_defer_calls:
            return False
        deferred_by_mask[ref.mask_id] = deferred + 1
        return True

    def _is_elastic_edit(self, target_id: int) -> bool:
        notation = self.registry.id_to_notation.get(target_id)
        return notation in {EXPAND, DELETE}

    def _maybe_remask_leaves(
        self,
        runtime: DecoderRuntime,
        *,
        model_calls: int,
        remasks_used: int,
        at_completion: bool,
    ) -> tuple[str, ...]:
        config = self.config
        if config.max_leaf_remasks <= remasks_used:
            return ()
        if config.leaf_remask_fraction <= 0:
            return ()
        if at_completion and not config.leaf_remask_at_completion:
            return ()

        candidates = [
            ref
            for ref in runtime.committed_leaf_tokens()
            if self._leaf_is_remask_candidate(ref, model_calls)
        ]
        if not candidates:
            return ()
        candidates.sort(
            key=lambda ref: (
                float("inf") if ref.confidence is None else ref.confidence,
                ref.cell_id,
            )
        )
        quota = max(
            1,
            math.ceil(len(candidates) * config.leaf_remask_fraction),
        )
        quota = min(quota, config.max_leaf_remasks - remasks_used)
        selected = candidates[:quota]
        runtime.remask_leaves(ref.cell_id for ref in selected)
        return tuple(ref.cell_id for ref in selected)

    def _leaf_is_remask_candidate(
        self, ref: LeafTokenRef, model_calls: int
    ) -> bool:
        config = self.config
        if ref.confidence is None or ref.committed_at_call is None:
            return False
        if ref.remask_count >= config.max_leaf_remasks_per_token:
            return False
        age = model_calls - ref.committed_at_call
        if age < config.leaf_remask_min_age_calls:
            return False
        threshold = config.leaf_remask_confidence_threshold
        if threshold is not None and ref.confidence >= threshold:
            return False
        return True

    def _maybe_backtrack_structure(
        self,
        runtime: DecoderRuntime,
        *,
        model_calls: int,
        backtracks_used: int,
    ) -> StructuralSubtreeRef | None:
        config = self.config
        threshold = config.structural_backtrack_confidence_threshold
        if threshold is None:
            return None
        if model_calls >= config.max_model_calls:
            return None
        if backtracks_used >= config.max_structural_backtracks:
            return None
        candidates = [
            ref
            for ref in runtime.completed_structural_subtrees()
            if (
                ref.mean_content_confidence < threshold
                and ref.backtrack_count
                < config.max_structural_backtracks_per_anchor
                and model_calls - ref.latest_content_commit_call
                >= config.structural_backtrack_min_age_calls
            )
        ]
        if not candidates:
            return None
        # Repair the deepest eligible subtree first. This minimizes deletion;
        # an ancestor remains eligible later if the corrected child does not
        # raise its aggregate confidence sufficiently.
        candidates.sort(
            key=lambda ref: (
                -ref.depth,
                ref.mean_content_confidence,
                ref.anchor_id,
            )
        )
        selected = candidates[0]
        runtime.backtrack_structural_subtree(selected.anchor_id)
        return selected

    def _prompt_ids(self, prompt: str) -> list[int]:
        text = self.registry.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            add_generation_prompt=True,
            tokenize=False,
        )
        return self.registry.tokenizer.encode(
            text, add_special_tokens=False
        )

    @torch.inference_mode()
    def _model_predictions(
        self,
        prompt_ids: list[int],
        canvas: DecoderCanvas,
        refs: tuple[MaskRef, ...],
        *,
        suppress_edits: bool = False,
    ) -> dict[str, tuple[int, float]]:
        if self.model is None:
            raise RuntimeInvariantError("no model or prediction provider")
        device = next(self.model.parameters()).device
        response_ids = list(canvas.input_ids)
        full_ids = (
            prompt_ids
            + response_ids
            + [self.registry.tokenizer.eos_token_id]
        )
        if len(full_ids) > self.model.config.max_position_embeddings:
            raise BudgetExceededError(
                f"prompt+canvas length {len(full_ids)} exceeds model context"
            )
        input_ids = torch.tensor([full_ids], device=device, dtype=torch.long)
        attention = torch.ones_like(input_ids, dtype=torch.bool)
        pairwise = torch.logical_and(
            attention.unsqueeze(1).unsqueeze(-2),
            attention.unsqueeze(1).unsqueeze(-1),
        )
        position_ids = torch.arange(
            input_ids.shape[1], device=device, dtype=torch.long
        ).unsqueeze(0)
        output = self.model(
            input_ids=input_ids,
            attention_mask=pairwise,
            position_ids=position_ids,
            use_cache=False,
        )
        shifted = torch.cat(
            [output.logits[:, :1], output.logits[:, :-1]], dim=1
        )
        response_offset = len(prompt_ids)
        ref_by_position = {
            index: ref
            for index, ref in enumerate(canvas.mask_refs)
            if ref is not None
        }
        predictions: dict[str, tuple[int, float]] = {}
        for response_position, ref in ref_by_position.items():
            logits = shifted[0, response_offset + response_position]
            allowed = self._allowed_ids(
                ref,
                logits.device,
                suppress_edits=suppress_edits,
            )
            allowed_logits = logits.index_select(0, allowed)
            allowed_logits = self._apply_body_construct_penalty(
                ref,
                allowed,
                allowed_logits,
            )
            allowed_logits = self._apply_body_stmt_bonus(
                ref,
                allowed,
                allowed_logits,
            )
            allowed_logits = self._apply_token_expand_bonus(
                ref,
                allowed,
                allowed_logits,
            )
            if self.config.temperature > 0:
                allowed_logits = allowed_logits / self.config.temperature
            local_index = int(torch.argmax(allowed_logits).item())
            target_id = int(allowed[local_index].item())
            confidence = self._confidence(allowed_logits)
            predictions[ref.mask_id] = (target_id, confidence)
        return predictions

    def _apply_body_construct_penalty(
        self,
        ref: MaskRef,
        allowed: torch.Tensor,
        allowed_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Discourage recursive compound statements inside opened bodies."""

        penalty = self.config.body_construct_logit_penalty
        if penalty <= 0 or ref.role is not MaskRole.LINE_BODY:
            return allowed_logits
        construct_ids = torch.tensor(
            [
                self.registry.special_id(notation)
                for notation in V0_CONSTRUCT_LABELS
            ],
            device=allowed.device,
            dtype=allowed.dtype,
        )
        penalized = torch.isin(allowed, construct_ids)
        if not penalized.any():
            return allowed_logits
        result = allowed_logits.clone()
        result[penalized] -= penalty
        return result

    def _apply_body_stmt_bonus(
        self,
        ref: MaskRef,
        allowed: torch.Tensor,
        allowed_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Raise the simple-statement prior for body-line decisions."""

        bonus = self.config.body_stmt_logit_bonus
        if bonus <= 0 or ref.role is not MaskRole.LINE_BODY:
            return allowed_logits
        stmt_id = self.registry.special_id("[STMT]")
        selected = allowed == stmt_id
        if not selected.any():
            return allowed_logits
        result = allowed_logits.clone()
        result[selected] += bonus
        return result

    def _apply_token_expand_bonus(
        self,
        ref: MaskRef,
        allowed: torch.Tensor,
        allowed_logits: torch.Tensor,
    ) -> torch.Tensor:
        """Raise the token-level expand prior without affecting line edits."""

        bonus = self.config.token_expand_logit_bonus
        if bonus <= 0 or ref.role not in {
            MaskRole.TOKEN_STMT,
            MaskRole.TOKEN_HDR,
            MaskRole.TOKEN_DOC,
        }:
            return allowed_logits
        expand_id = self.registry.special_id(EXPAND)
        selected = allowed == expand_id
        if not selected.any():
            return allowed_logits
        result = allowed_logits.clone()
        result[selected] += bonus
        return result

    def _allowed_ids(
        self,
        ref: MaskRef,
        device: torch.device,
        *,
        suppress_edits: bool = False,
    ) -> torch.Tensor:
        if ref.role in {
            MaskRole.LINE_MODULE,
            MaskRole.LINE_BODY,
            MaskRole.LINE_CLAUSE,
        }:
            notations = ref.allowed_notations
            if (
                suppress_edits
                and ref.role
                in {MaskRole.LINE_MODULE, MaskRole.LINE_BODY}
            ):
                notations = tuple(
                    notation
                    for notation in notations
                    if notation not in {EXPAND, DELETE}
                )
            return torch.tensor(
                [
                    self.registry.special_id(notation)
                    for notation in notations
                ],
                device=device,
                dtype=torch.long,
            )
        if self._token_allowed_cpu is None:
            self._token_allowed_cpu = self._build_token_allowed_ids()
        allowed = self._token_allowed_cpu
        permitted_edits = {
            self.registry.special_id(notation)
            for notation in ref.allowed_notations
        }
        if suppress_edits:
            permitted_edits.clear()
        edit_ids = {
            self.registry.special_id(EXPAND),
            self.registry.special_id(DELETE),
        }
        if permitted_edits != edit_ids:
            allowed = torch.tensor(
                [
                    token_id
                    for token_id in allowed.tolist()
                    if token_id not in edit_ids
                    or token_id in permitted_edits
                ],
                dtype=torch.long,
            )
        return allowed.to(device=device)

    def _build_token_allowed_ids(self) -> torch.Tensor:
        tokenizer = self.registry.tokenizer
        valid_length = len(tokenizer)
        banned = set(tokenizer.all_special_ids)
        banned.update(self.registry.notation_to_id.values())
        banned.update(edit_source_token_ids(tokenizer))
        banned.add(tokenizer.mask_token_id)
        allowed: list[int] = []
        for token_id in range(valid_length):
            if token_id in banned:
                continue
            text = tokenizer.decode([token_id])
            if "\n" in text or "\r" in text:
                continue
            allowed.append(token_id)
        allowed.extend(
            [
                self.registry.special_id(EXPAND),
                self.registry.special_id(DELETE),
            ]
        )
        return torch.tensor(sorted(set(allowed)), dtype=torch.long)

    def _confidence(self, logits: torch.Tensor) -> float:
        if self.config.confidence == "margin":
            top = torch.topk(logits, k=min(2, logits.numel())).values
            return float(top[0].item() - top[-1].item())
        probabilities = torch.softmax(logits.float(), dim=-1)
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-12))
        )
        normalizer = math.log(max(2, logits.numel()))
        return float(1.0 - entropy.item() / normalizer)
