"""
L1 写入器: 负责将 key-value 对写入关联矩阵记忆。

核心更新公式:
    M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T

其中 λ 为衰减率，ρ_i 为写入强度。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assoc_memory import L1Config


class L1Writer(nn.Module):
    """
    L1 记忆写入器。

    负责执行衰减 + 外积写入:
        M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T
    """

    def __init__(self, config: L1Config) -> None:
        super().__init__()
        self.decay = config.decay
        self.default_write_strength = config.write_strength
        self.n_heads = config.n_heads
        self.d_key = config.d_key
        self.d_value = config.d_value

    def write(
        self,
        memory: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        write_strengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        执行一次写入操作。

        Args:
            memory: 当前记忆状态 (n_heads, d_key, d_value)
            keys:   写入键 (batch, seq_len, n_heads, d_key)
            values: 写入值 (batch, seq_len, n_heads, d_value)
            write_strengths: 逐 token 写入强度 (batch, seq_len)，
                             None 则使用默认强度

        Returns:
            updated_memory: 更新后的记忆状态 (n_heads, d_key, d_value)
        """
        B, S, H, D_k = keys.shape
        D_v = values.shape[-1]

        # 衰减旧记忆
        new_memory = self.decay * memory  # (n_heads, d_key, d_value)

        # 计算写入强度
        if write_strengths is None:
            rho = self.default_write_strength
        else:
            # write_strengths: (B, S) -> (B, S, 1, 1, 1) 用于广播
            rho = write_strengths.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)

        # 计算外积并累加: Σ_i ρ_i k_i v_i^T
        # keys:   (B, S, H, D_k) -> (B, S, H, D_k, 1)
        # values: (B, S, H, D_v) -> (B, S, H, 1, D_v)
        k_expanded = keys.unsqueeze(-1)       # (B, S, H, D_k, 1)
        v_expanded = values.unsqueeze(-2)     # (B, S, H, 1, D_v)

        # 外积: (B, S, H, D_k, D_v)
        outer_product = k_expanded * v_expanded

        # 应用写入强度
        if isinstance(rho, float):
            weighted = rho * outer_product
        else:
            weighted = rho * outer_product

        # 对 batch 和 seq_len 维度求和，得到 (H, D_k, D_v)
        delta = weighted.sum(dim=0).sum(dim=0)

        # 更新记忆
        new_memory = new_memory + delta

        return new_memory

    def write_single_step(
        self,
        memory: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        write_strength: float | None = None,
    ) -> torch.Tensor:
        """
        单步写入（用于逐 token 在线更新）。

        Args:
            memory: (n_heads, d_key, d_value)
            key:    (n_heads, d_key)
            value:  (n_heads, d_value)
            write_strength: 标量写入强度

        Returns:
            updated_memory: (n_heads, d_key, d_value)
        """
        rho = write_strength if write_strength is not None else self.default_write_strength

        # 衰减
        new_memory = self.decay * memory

        # 外积: (n_heads, d_key, 1) * (n_heads, 1, d_value) -> (n_heads, d_key, d_value)
        outer = key.unsqueeze(-1) * value.unsqueeze(-2)
        new_memory = new_memory + rho * outer

        return new_memory
