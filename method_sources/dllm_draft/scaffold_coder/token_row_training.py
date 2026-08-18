"""Utilities for adapting only selected vocabulary rows."""

from __future__ import annotations

import torch


def summarize_trainable_token_parameters(
    model: torch.nn.Module,
    *,
    token_ids: tuple[int, ...],
) -> dict[str, object]:
    """Validate that only compact trainable-token deltas are trainable."""

    if not token_ids:
        raise ValueError("token_row_only requires at least one token ID")
    matched: list[str] = []
    shapes: dict[str, list[int]] = {}
    total = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "trainable_tokens_delta" not in name:
            raise ValueError(
                f"unexpected trainable parameter in token-row mode: {name}"
            )
        if parameter.ndim != 2 or parameter.shape[0] != len(token_ids):
            raise ValueError(
                f"invalid trainable-token shape for {name}: "
                f"{tuple(parameter.shape)}"
            )
        matched.append(name)
        shapes[name] = list(parameter.shape)
        total += parameter.numel()
    if len(matched) != 2:
        raise ValueError(
            "token_row_only expected exactly input/output trainable-token "
            f"deltas, found {matched}"
        )
    return {
        "matched_parameters": matched,
        "parameter_shapes": shapes,
        "selected_token_ids": list(token_ids),
        "trainable_parameters": total,
    }


def configure_token_row_only_training(
    model: torch.nn.Module,
    *,
    token_ids: tuple[int, ...],
) -> tuple[torch.nn.Module, dict[str, object]]:
    """Wrap input/output vocabularies with PEFT's compact row parameters."""

    if not token_ids:
        raise ValueError("token_row_only requires at least one token ID")
    # PEFT 0.19 checks this namespace when Torch distributed is available.
    import torch.distributed.tensor  # noqa: F401
    from peft import TrainableTokensConfig, get_peft_model

    adapted = get_peft_model(
        model,
        TrainableTokensConfig(
            token_indices=list(token_ids),
            target_modules=["model.embed_tokens", "lm_head"],
        ),
    )
    report = summarize_trainable_token_parameters(
        adapted,
        token_ids=token_ids,
    )
    return adapted, report


def selected_token_target_mask(
    labels: torch.Tensor,
    *,
    token_ids: tuple[int, ...],
) -> torch.Tensor:
    """Return positions whose labels are one of the adapted vocabulary rows."""

    if not token_ids:
        raise ValueError("selected target mask requires token IDs")
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for token_id in token_ids:
        selected |= labels == token_id
    return selected


def lexical_teacher_target_mask(
    labels: torch.Tensor,
    role_ids: torch.Tensor,
    *,
    lexical_role_ids: tuple[int, ...],
    excluded_token_ids: tuple[int, ...] = (),
) -> torch.Tensor:
    """Select ordinary lexical targets for teacher distribution matching.

    Scaffold edit/meta labels may share lexical roles such as ``TOKEN_STMT``.
    They must remain supervised only by ground-truth CE; asking a pretrained
    teacher to match their logits directly suppresses the new edit behavior.
    """

    if labels.shape != role_ids.shape:
        raise ValueError("labels and role_ids must have the same shape")
    selected = torch.zeros_like(labels, dtype=torch.bool)
    for role_id in lexical_role_ids:
        selected |= role_ids == role_id
    for token_id in excluded_token_ids:
        selected &= labels != token_id
    return selected
