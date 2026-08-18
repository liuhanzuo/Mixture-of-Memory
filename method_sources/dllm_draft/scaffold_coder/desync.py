"""Per-top-level-subtree desynchronization for soft decoding states."""

from __future__ import annotations

import random
from dataclasses import dataclass

from .canvas import CanvasState, TokenRegistry, prepend_chat_prompt
from .corruption import (
    GlobalBandSampler,
    Rung,
    SampledTrainingState,
)
from .ir import Body, FunctionDefinition, Module
from .roles import MaskRole


@dataclass(frozen=True, slots=True)
class DesyncConfig:
    sigma_d: float = 0.10

    def validate(self) -> None:
        if not 0 <= self.sigma_d <= 1:
            raise ValueError("sigma_d must lie in [0,1]")


class DesynchronizedGlobalSampler:
    """Apply one global t plus independent top-level subtree offsets."""

    def __init__(
        self,
        registry: TokenRegistry,
        *,
        base_sampler: GlobalBandSampler | None = None,
        config: DesyncConfig | None = None,
    ) -> None:
        self.registry = registry
        self.base_sampler = base_sampler or GlobalBandSampler(registry)
        self.config = config or DesyncConfig()
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
        lines = list(module.body.lines)
        if len(lines) <= 1 or self.config.sigma_d == 0:
            sampled = self.base_sampler.sample(
                module, prompt, seed=seed, t=global_t
            )
            metadata = dict(sampled.metadata)
            metadata["desync_offsets"] = [0.0]
            return SampledTrainingState(
                state=sampled.state,
                rung=sampled.rung,
                local_u=sampled.local_u,
                global_t_proxy=sampled.global_t_proxy,
                loss_weights=sampled.loss_weights,
                metadata=metadata,
            )

        prompt_length = len(
            self.registry.tokenizer.encode(
                self.registry.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    add_generation_prompt=True,
                    tokenize=False,
                ),
                add_special_tokens=False,
            )
        )
        response_states: list[CanvasState] = []
        response_weights: list[tuple[float, ...]] = []
        subtree_metadata: list[dict[str, object]] = []
        offsets: list[float] = []

        for index, line in enumerate(lines):
            offset = rng.uniform(-self.config.sigma_d, self.config.sigma_d)
            subtree_t = min(1.0, max(0.0, global_t + offset))
            offsets.append(offset)
            submodule = Module(
                node_id=f"{module.node_id}-sub-{index}",
                body=Body(
                    body_id=f"{module.body.body_id}-sub-{index}",
                    owner_id=f"{module.node_id}-sub-{index}",
                    depth=0,
                    lines=(line,),
                ),
            )
            sampled = self.base_sampler.sample(
                submodule,
                prompt,
                seed=rng.randrange(2**63),
                t=subtree_t,
            )
            response, weights = _strip_prompt_and_eos(
                sampled, prompt_length=prompt_length
            )
            response_states.append(response)
            response_weights.append(weights)
            subtree_metadata.append(
                {
                    "line_node_id": line.node_id,
                    "offset": offset,
                    "t": subtree_t,
                    "rung": sampled.rung.value,
                    "local_u": sampled.local_u,
                }
            )

        combined, combined_weights = _join_top_level_responses(
            response_states,
            response_weights,
            lines,
            self.registry,
            module.body.body_id,
        )
        state = prepend_chat_prompt(combined, self.registry, prompt)
        prefix = len(state.input_ids) - len(combined.input_ids) - 1
        scale = 1.0 / len(response_states)
        loss_weights = (
            (0.0,) * prefix
            + tuple(weight * scale for weight in combined_weights)
            + (0.0,)
        )
        target_counts: dict[str, int] = {}
        role_counts: dict[str, int] = {}
        for target_id, supervised, role in zip(
            state.labels, state.loss_mask, state.roles, strict=True
        ):
            if not supervised:
                continue
            notation = self.registry.id_to_notation.get(target_id, "LEXICAL")
            target_counts[notation] = target_counts.get(notation, 0) + 1
            role_counts[role.value] = role_counts.get(role.value, 0) + 1
        return SampledTrainingState(
            state=state,
            rung=Rung.MIXED,
            local_u=sum(
                float(item["local_u"]) for item in subtree_metadata
            )
            / len(subtree_metadata),
            global_t_proxy=global_t,
            loss_weights=loss_weights,
            metadata={
                "seed": seed,
                "rung": Rung.MIXED.value,
                "global_t": global_t,
                "desync_offsets": offsets,
                "subtrees": subtree_metadata,
                "target_counts": target_counts,
                "role_counts": role_counts,
            },
        )


def _strip_prompt_and_eos(
    sampled: SampledTrainingState, *, prompt_length: int
) -> tuple[CanvasState, tuple[float, ...]]:
    end = len(sampled.state.input_ids) - 1
    state = sampled.state
    response = CanvasState(
        state_name=state.state_name.replace("chat+", "", 1),
        input_ids=state.input_ids[prompt_length:end],
        labels=state.labels[prompt_length:end],
        loss_mask=state.loss_mask[prompt_length:end],
        roles=state.roles[prompt_length:end],
        node_ids=state.node_ids[prompt_length:end],
        eligible=state.eligible[prompt_length:end],
    )
    return response, sampled.loss_weights[prompt_length:end]


def _join_top_level_responses(
    states: list[CanvasState],
    weights: list[tuple[float, ...]],
    lines,
    registry: TokenRegistry,
    module_body_id: str,
) -> tuple[CanvasState, tuple[float, ...]]:
    fields = {
        "input_ids": [],
        "labels": [],
        "loss_mask": [],
        "roles": [],
        "node_ids": [],
        "eligible": [],
    }
    joined_weights: list[float] = []
    previous = None
    for index, (state, state_weights, line) in enumerate(
        zip(states, weights, lines, strict=True)
    ):
        if index > 0 and (
            isinstance(previous, FunctionDefinition)
            or isinstance(line, FunctionDefinition)
        ):
            separator = registry.tokenizer.encode(
                "\n\n", add_special_tokens=False
            )
            fields["input_ids"].extend(separator)
            fields["labels"].extend(separator)
            fields["loss_mask"].extend([False] * len(separator))
            fields["roles"].extend([MaskRole.RULE] * len(separator))
            fields["node_ids"].extend([module_body_id] * len(separator))
            fields["eligible"].extend([False] * len(separator))
            joined_weights.extend([0.0] * len(separator))
        for name in fields:
            fields[name].extend(getattr(state, name))
        joined_weights.extend(state_weights)
        previous = line
    combined = CanvasState(
        state_name="desynchronized_response",
        input_ids=tuple(fields["input_ids"]),
        labels=tuple(fields["labels"]),
        loss_mask=tuple(fields["loss_mask"]),
        roles=tuple(fields["roles"]),
        node_ids=tuple(fields["node_ids"]),
        eligible=tuple(fields["eligible"]),
    )
    combined.validate(registry)
    return combined, tuple(joined_weights)

