"""
L1 读取器: 从关联矩阵记忆中检索信息。

核心读取公式:
    r_t = q_t^T M_t

其中 q_t 为查询向量，M_t 为当前记忆状态。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assoc_memory import L1Config


class L1Reader(nn.Module):
    """
    L1 记忆读取器。

    执行查询-记忆点积检索:
        r_t = q_t^T M_t

    对于多头变体，每个头独立读取后拼接。
    """

    def __init__(self, config: L1Config) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.d_key = config.d_key
        self.d_value = config.d_value

    def read(
        self,
        memory: torch.Tensor,
        queries: torch.Tensor,
    ) -> torch.Tensor:
        """
        从记忆中读取。

        Args:
            memory:  当前记忆状态 (n_heads, d_key, d_value)
            queries: 查询向量 (batch, seq_len, n_heads, d_key)

        Returns:
            readout: 读取结果 (batch, seq_len, n_heads * d_value)
        """
        B, S, H, D_k = queries.shape

        # queries: (B, S, H, D_k) -> (B, S, H, 1, D_k)
        # memory:  (H, D_k, D_v)  -> (1, 1, H, D_k, D_v)
        # 点积: (B, S, H, 1, D_k) @ (1, 1, H, D_k, D_v) -> (B, S, H, 1, D_v)
        q_expanded = queries.unsqueeze(-2)  # (B, S, H, 1, D_k)
        m_expanded = memory.unsqueeze(0).unsqueeze(0)  # (1, 1, H, D_k, D_v)

        readout = torch.matmul(q_expanded, m_expanded)  # (B, S, H, 1, D_v)
        readout = readout.squeeze(-2)  # (B, S, H, D_v)

        # 拼接所有头的输出
        readout = readout.reshape(B, S, H * self.d_value)  # (B, S, H * D_v)

        return readout

    def read_single_step(
        self,
        memory: torch.Tensor,
        query: torch.Tensor,
    ) -> torch.Tensor:
        """
        单步读取（用于逐 token 在线推理）。

        Args:
            memory: (n_heads, d_key, d_value)
            query:  (n_heads, d_key)

        Returns:
            readout: (n_heads * d_value,)
        """
        # query: (H, D_k) -> (H, 1, D_k)
        # memory: (H, D_k, D_v)
        # 点积: (H, 1, D_k) @ (H, D_k, D_v) -> (H, 1, D_v)
        q_expanded = query.unsqueeze(-2)  # (H, 1, D_k)
        readout = torch.matmul(q_expanded, memory)  # (H, 1, D_v)
        readout = readout.squeeze(-2)  # (H, D_v)
        readout = readout.reshape(-1)  # (H * D_v,)

        return readout
