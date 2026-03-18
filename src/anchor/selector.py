"""
AnchorSelector: 从 BlockEvaluator 的分数中选择 top-k anchor 位置。

设计要点:
- 输入: block scores [B, L], hidden states [B, L, D]
- 输出: 选中的 indices 和对应的 anchor hidden states
- 支持可配置的 top_k
- batch-safe
- eval 模式下确定性
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn as nn


class AnchorSelection(NamedTuple):
    """Anchor 选择结果。"""
    indices: torch.Tensor       # [B, K] 选中的位置索引
    hidden_states: torch.Tensor # [B, K, D] 选中位置的 hidden states
    scores: torch.Tensor        # [B, K] 选中位置的分数


class AnchorSelector(nn.Module):
    """
    从 evaluator 输出的分数中选择 top-k 个 anchor 位置。

    支持两种模式:
    - deterministic (eval): 直接取 top-k
    - stochastic (train): 基于分数概率采样 top-k（可选，用于探索）
    """

    def __init__(
        self,
        top_k: int = 4,
        stochastic_train: bool = False,
        temperature: float = 1.0,
    ) -> None:
        """
        Args:
            top_k: 每个 block 选择的 anchor 数量
            stochastic_train: 训练时是否用概率采样而非确定性 top-k
            temperature: 采样时的温度参数
        """
        super().__init__()
        self.top_k = top_k
        self.stochastic_train = stochastic_train
        self.temperature = temperature

    def forward(
        self,
        scores: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> AnchorSelection:
        """
        选择 top-k anchor 位置。

        Args:
            scores: [B, L] evaluator 输出的分数
            hidden_states: [B, L, D] block hidden states

        Returns:
            AnchorSelection: 包含 indices [B, K], hidden_states [B, K, D], scores [B, K]
        """
        B, L = scores.shape
        k = min(self.top_k, L)  # 防止 top_k > block 长度

        if self.training and self.stochastic_train:
            # 基于概率采样（Gumbel top-k trick 的简化版本）
            probs = torch.softmax(scores / self.temperature, dim=-1)  # [B, L]
            # 使用 multinomial 采样（无放回）
            selected_indices = torch.multinomial(probs, num_samples=k, replacement=False)  # [B, K]
            # 按位置排序，保持顺序一致性
            selected_indices, _ = selected_indices.sort(dim=-1)
        else:
            # 确定性 top-k
            _, selected_indices = scores.topk(k, dim=-1, largest=True, sorted=True)  # [B, K]
            # 按位置排序
            selected_indices, _ = selected_indices.sort(dim=-1)

        # 收集选中位置的 hidden states
        # selected_indices: [B, K] -> [B, K, 1] -> [B, K, D]
        gather_idx = selected_indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        selected_hidden = torch.gather(hidden_states, dim=1, index=gather_idx)  # [B, K, D]

        # 收集选中位置的 scores
        selected_scores = torch.gather(scores, dim=1, index=selected_indices)  # [B, K]

        return AnchorSelection(
            indices=selected_indices,
            hidden_states=selected_hidden,
            scores=selected_scores,
        )

    def extra_repr(self) -> str:
        return (
            f"top_k={self.top_k}, "
            f"stochastic_train={self.stochastic_train}, "
            f"temperature={self.temperature}"
        )
