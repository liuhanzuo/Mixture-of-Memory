"""Weighted masked-diffusion loss shared by trainer and tests."""

from __future__ import annotations

import torch
from torch import nn


def shifted_weighted_masked_ce(
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Dream shift-op CE normalized by the sum of explicit token weights."""

    if logits.ndim != 3:
        raise ValueError(f"expected logits [B,S,V], got {logits.shape}")
    if labels.shape != logits.shape[:2]:
        raise ValueError("labels shape must match logits [B,S]")
    for name, tensor in {
        "loss_mask": loss_mask,
        "loss_weights": loss_weights,
    }.items():
        if tensor.shape != labels.shape:
            raise ValueError(f"{name} shape must match labels")

    shifted_logits = torch.cat(
        [logits[:, 0:1], logits[:, :-1]], dim=1
    ).contiguous()
    token_loss = nn.functional.cross_entropy(
        shifted_logits.view(-1, shifted_logits.shape[-1]),
        labels.reshape(-1),
        reduction="none",
    ).view_as(labels)
    effective_weights = (
        loss_weights.to(token_loss.dtype) * loss_mask.to(token_loss.dtype)
    )
    denominator = effective_weights.sum()
    if denominator.item() <= 0:
        raise ValueError("weighted loss has no supervised mass")
    loss = (token_loss * effective_weights).sum() / denominator
    metrics = {
        "weighted_token_loss_sum": (token_loss * effective_weights).sum().detach(),
        "weight_sum": denominator.detach(),
        "supervised_tokens": loss_mask.sum().detach(),
        "mean_raw_supervised_loss": token_loss[loss_mask].mean().detach(),
    }
    return loss, metrics


def shifted_masked_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    mask: torch.Tensor,
    weights: torch.Tensor,
    *,
    temperature: float = 1.0,
    topk: int | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Weighted ``KL(teacher || student)`` under Dream's shift operator."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("student and teacher logits must have the same shape")
    if student_logits.ndim != 3:
        raise ValueError(
            f"expected logits [B,S,V], got {student_logits.shape}"
        )
    for name, tensor in {"mask": mask, "weights": weights}.items():
        if tensor.shape != student_logits.shape[:2]:
            raise ValueError(f"{name} shape must match logits [B,S]")
    if topk is not None and topk <= 0:
        raise ValueError("topk must be positive")

    student_shifted = torch.cat(
        [student_logits[:, 0:1], student_logits[:, :-1]], dim=1
    )
    teacher_shifted = torch.cat(
        [teacher_logits[:, 0:1], teacher_logits[:, :-1]], dim=1
    )
    selected_student = student_shifted[mask].float() / temperature
    selected_teacher = teacher_shifted[mask].float() / temperature
    if topk is None or topk >= selected_teacher.shape[-1]:
        teacher_probability = nn.functional.softmax(
            selected_teacher, dim=-1
        )
        teacher_log_probability = nn.functional.log_softmax(
            selected_teacher, dim=-1
        )
        student_log_probability = nn.functional.log_softmax(
            selected_student, dim=-1
        )
        selected_kl = (
            teacher_probability
            * (teacher_log_probability - student_log_probability)
        ).sum(dim=-1)
    else:
        teacher_values, teacher_indices = selected_teacher.topk(
            topk, dim=-1
        )
        student_values = selected_student.gather(
            dim=-1, index=teacher_indices
        )
        teacher_probability = nn.functional.softmax(
            teacher_values, dim=-1
        )
        teacher_log_probability = nn.functional.log_softmax(
            teacher_values, dim=-1
        )
        student_log_probability = nn.functional.log_softmax(
            student_values, dim=-1
        )
        selected_kl = (
            teacher_probability
            * (teacher_log_probability - student_log_probability)
        ).sum(dim=-1)
    selected_kl = selected_kl * (temperature**2)

    effective_weights = weights[mask].to(selected_kl.dtype)
    denominator = effective_weights.sum()
    if denominator.item() <= 0:
        raise ValueError("weighted KL has no supervised mass")
    loss = (selected_kl * effective_weights).sum() / denominator
    metrics = {
        "weighted_token_kl_sum": (
            selected_kl * effective_weights
        ).sum().detach(),
        "weight_sum": denominator.detach(),
        "anchored_tokens": mask.sum().detach(),
        "mean_raw_kl": selected_kl.mean().detach(),
    }
    return loss, metrics
