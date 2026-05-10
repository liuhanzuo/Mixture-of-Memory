"""DMS Decision Head: per-token eviction score predictor.

Each layer gets a lightweight linear head that predicts whether a token's KV pair
should be evicted from the cache. During training, decisions are relaxed via
Gumbel-Sigmoid; during inference, they are thresholded to binary.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def gumbel_sigmoid(
    logits: torch.Tensor,
    tau: float = 0.1,
    hard: bool = True,
) -> torch.Tensor:
    """Sample from Gumbel-Sigmoid distribution.

    Args:
        logits: Pre-sigmoid logits, shape ``(*, 1)`` or ``(*,)``.
        tau: Temperature (lower → more discrete).
        hard: If True, use straight-through estimator.

    Returns:
        Relaxed binary decisions in [0, 1], shape ``(*,)``.
    """
    # Gumbel noise
    gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-20) + 1e-20)
    y_soft = torch.sigmoid((logits + gumbel_noise) / tau)

    if hard:
        # Straight-through estimator
        y_hard = (y_soft > 0.5).float()
        return y_hard - y_soft.detach() + y_soft
    return y_soft


class DMSDecisionHead(nn.Module):
    """Per-layer decision head for DMS eviction.

    Predicts a binary eviction decision α_t for each token at each layer.
    Initialized to always-keep (α ≈ 0) to prevent early disruption.

    Args:
        hidden_dim: Model hidden dimension (input to the head).
        bias_init: Initial bias value (negative → always keep at start).
        tau: Gumbel-Sigmoid temperature.
    """

    def __init__(
        self,
        hidden_dim: int,
        bias_init: float = -5.0,
        tau: float = 0.1,
    ):
        super().__init__()
        # Single linear projection: h_t @ w + b → eviction logit
        self.weight = nn.Parameter(torch.zeros(hidden_dim))
        self.bias = nn.Parameter(torch.full((1,), bias_init))
        self.tau = tau

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute eviction decisions.

        Args:
            hidden_states: Shape ``(B, T, D)``.

        Returns:
            Relaxed eviction decisions, shape ``(B, T)``.
        """
        # (B, T, D) @ (D,) → (B, T)
        dtype = hidden_states.dtype
        logits = torch.einsum("btd,d->bt", hidden_states, self.weight.to(dtype)) + self.bias.to(dtype)
        return gumbel_sigmoid(logits, tau=self.tau, hard=False)

    def forward_binary(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute hard binary eviction decisions for inference.

        Args:
            hidden_states: Shape ``(B, T, D)``.

        Returns:
            Binary decisions {0, 1}, shape ``(B, T)``. 1 = evict.
        """
        dtype = hidden_states.dtype
        logits = torch.einsum("btd,d->bt", hidden_states, self.weight.to(dtype)) + self.bias.to(dtype)
        return (torch.sigmoid(logits) > 0.5).float()
