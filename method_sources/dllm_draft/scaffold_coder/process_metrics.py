"""Process-level metrics for structured diffusion decoding."""

from __future__ import annotations

import ast
from dataclasses import dataclass, asdict

from .model_sampler import ScaffoldGenerationResult


@dataclass(frozen=True, slots=True)
class GenerationProcessMetrics:
    final_parseable: bool
    placeholder_steps: int
    placeholder_parseable_steps: int
    placeholder_parse_rate: float | None
    nfe: int
    final_canvas_tokens: int
    minimum_model_canvas_tokens: int | None
    maximum_model_canvas_tokens: int | None
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
    termination_reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def compute_process_metrics(
    result: ScaffoldGenerationResult,
) -> GenerationProcessMetrics:
    final_parseable = _parseable(result.text)
    placeholder_parseable = sum(
        _parseable(text) for text in result.placeholder_history
    )
    count = len(result.placeholder_history)
    lengths = result.model_canvas_lengths
    return GenerationProcessMetrics(
        final_parseable=final_parseable,
        placeholder_steps=count,
        placeholder_parseable_steps=placeholder_parseable,
        placeholder_parse_rate=(
            placeholder_parseable / count if count else None
        ),
        nfe=result.model_calls,
        final_canvas_tokens=result.final_canvas_tokens,
        minimum_model_canvas_tokens=min(lengths) if lengths else None,
        maximum_model_canvas_tokens=max(lengths) if lengths else None,
        cumulative_model_tokens=result.cumulative_model_tokens,
        expansions=result.expansions,
        leaf_remasks=result.leaf_remasks,
        correction_rounds=result.correction_rounds,
        structural_deferrals=result.structural_deferrals,
        structural_backtracks=result.structural_backtracks,
        edit_cycle_breaks=result.edit_cycle_breaks,
        line_capacity_hits=result.line_capacity_hits,
        token_capacity_hits=result.token_capacity_hits,
        depth_capacity_hits=result.depth_capacity_hits,
        total_line_capacity_hits=result.total_line_capacity_hits,
        module_expand_suppressed=result.module_expand_suppressed,
        expand_budget_hits=result.expand_budget_hits,
        maximum_tree_depth=result.maximum_tree_depth,
        maximum_total_lines=result.maximum_total_lines,
        maximum_body_lines=result.maximum_body_lines,
        maximum_tokens_per_hole=result.maximum_tokens_per_hole,
        termination_reason=result.termination_reason,
    )


def _parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
