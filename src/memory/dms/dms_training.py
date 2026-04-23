"""DMS Training: loss function and compression scheduler.

The DMS retrofitting uses:
1. Logit distillation loss (teacher-student KL divergence)
2. Auxiliary compression loss (one-sided L1 to match target compression ratio)
3. Linear ramp schedule for compression ratio during training
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class CompressionScheduler:
    """Linear ramp schedule for target compression ratio.

    CR(t) = t / steps_per_cr + 1, linearly annealed over training.

    Args:
        target_cr: Maximum compression ratio (e.g., 4 or 8).
        total_steps: Total training steps.
        warmup_steps: Steps before compression begins (CR = 1).
    """

    def __init__(
        self,
        target_cr: float = 8.0,
        total_steps: int = 1000,
        warmup_steps: int = 100,
    ):
        self.target_cr = target_cr
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        # α* = 1 - 1/CR, ramped linearly
        self.max_alpha_star = 1.0 - 1.0 / target_cr

    def get_alpha_star(self, step: int) -> float:
        """Get target compression rate at given step.

        Args:
            step: Current training step.

        Returns:
            Target average α* ∈ [0, 1 - 1/CR].
        """
        if step < self.warmup_steps:
            return 0.0

        effective_step = step - self.warmup_steps
        effective_total = self.total_steps - self.warmup_steps
        progress = min(effective_step / effective_total, 1.0)
        return progress * self.max_alpha_star

    def get_current_cr(self, step: int) -> float:
        """Get current effective compression ratio."""
        alpha_star = self.get_alpha_star(step)
        if alpha_star == 0:
            return 1.0
        return 1.0 / (1.0 - alpha_star)


class DMSLoss(nn.Module):
    """Combined DMS training loss.

    L = L_distill + λ * L_aux

    Where:
    - L_distill: KL divergence between teacher and student logits
    - L_aux: One-sided L1 compression matching loss

    Args:
        lambda_aux: Weight for auxiliary compression loss.
        teacher_model: Teacher model for logit distillation (optional, uses labels if None).
    """

    def __init__(
        self,
        lambda_aux: float = 1.0,
        teacher_model: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.lambda_aux = lambda_aux
        self.teacher_model = teacher_model

    def distillation_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        temperature: float = 2.0,
    ) -> torch.Tensor:
        """KL divergence distillation loss.

        Args:
            student_logits: Shape ``(B, T, V)``.
            teacher_logits: Shape ``(B, T, V)``.
            labels: Optional labels to mask loss on padding tokens.
            temperature: Softmax temperature for distillation.

        Returns:
            Scalar loss.
        """
        B, T, V = student_logits.shape

        # Softmax with temperature
        student_probs = F.log_softmax(student_logits / temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)

        # KL divergence (per-token, per-position)
        kl = F.kl_div(student_probs, teacher_probs, reduction="none").sum(dim=-1)

        # Mask padding positions
        if labels is not None:
            mask = (labels != -100).float()
            kl = kl * mask
            loss = kl.sum() / mask.sum().clamp(min=1)
        else:
            loss = kl.mean()

        # Scale by temperature^2 (standard practice)
        return loss * (temperature ** 2)

    def language_model_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """Standard cross-entropy language modeling loss (fallback when no teacher).

        Args:
            logits: Shape ``(B, T, V)``.
            labels: Shape ``(B, T)``.

        Returns:
            Scalar loss.
        """
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )

    def auxiliary_loss(
        self,
        alpha_star: float,
        model: nn.Module,
    ) -> torch.Tensor:
        """One-sided L1 compression loss.

        Forces average α across all layers to match target α*.

        L_aux = max(α* - mean(α), 0)

        Only penalizes when actual compression is below target (not enough eviction).

        Args:
            alpha_star: Target compression rate.
            model: Model with DMS wrappers (with `_last_alpha` attributes).

        Returns:
            Scalar loss.
        """
        from .dms_attention import DMSAttentionWrapper

        total_alpha = torch.tensor(0.0)
        num_layers = 0
        device = None

        for name, module in model.named_modules():
            if isinstance(module, DMSAttentionWrapper):
                if hasattr(module, "_last_alpha") and module._last_alpha is not None:
                    a = module._last_alpha  # (B, T), already detached
                    if device is None:
                        device = a.device
                    total_alpha = total_alpha.to(device) + a.sum()
                    num_layers += 1

        if num_layers == 0:
            return torch.tensor(0.0)

        mean_alpha = total_alpha / (num_layers * total_alpha.numel())
        # One-sided: penalize only if mean α < target α*
        loss = alpha_star - mean_alpha
        return torch.clamp(loss, min=0.0)

    def forward(
        self,
        student_logits: torch.Tensor,
        labels: torch.Tensor,
        teacher_logits: Optional[torch.Tensor] = None,
        alpha_star: float = 0.0,
        model: Optional[nn.Module] = None,
    ) -> dict[str, torch.Tensor]:
        """Compute combined DMS loss.

        Args:
            student_logits: Student model logits, ``(B, T, V)``.
            labels: Target labels, ``(B, T)``.
            teacher_logits: Teacher model logits (for distillation), ``(B, T, V)``.
            alpha_star: Target compression rate from scheduler.
            model: Student model (for auxiliary loss computation).

        Returns:
            Dict with 'total', 'distill', 'aux' loss values.
        """
        # Main loss
        if teacher_logits is not None and self.teacher_model is not None:
            main_loss = self.distillation_loss(student_logits, teacher_logits, labels)
            loss_name = "distill"
        else:
            main_loss = self.language_model_loss(student_logits, labels)
            loss_name = "lm"

        # Auxiliary compression loss
        aux_loss = torch.tensor(0.0, device=main_loss.device)
        if alpha_star > 0 and model is not None:
            aux_loss = self.auxiliary_loss(alpha_star, model)

        total = main_loss + self.lambda_aux * aux_loss

        return {
            "total": total,
            loss_name: main_loss,
            "aux": aux_loss,
        }
