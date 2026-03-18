"""
Memory Readout —— 记忆读取 + 学习路由。

在每个 token 步，从所有记忆中读取并通过学习的路由权重 γ 加权合并：
    r_t^{(i)} = q_t^T M_t^{(i)}
    r_t = Σ_i γ_t^{(i)} r_t^{(i)}

路由权重 γ 由当前 hidden state 决定，保持轻量级。
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MemoryReadout(nn.Module):
    """从 MOM 中读取并融合多记忆输出。

    流程:
        1. 将 backbone hidden state 投影为 query: q = W_Q h
        2. 从每个记忆读取: r^{(i)} = q^T M^{(i)}
        3. 计算读取路由权重 γ（learned router）
        4. 加权合并: r = Σ_i γ^{(i)} r^{(i)}
    """

    def __init__(
        self,
        hidden_dim: int,
        key_dim: int = 64,
        value_dim: int = 64,
        num_memories: int = 3,
        router_hidden_dim: int = 64,
    ) -> None:
        """
        Args:
            hidden_dim: backbone 隐藏状态维度
            key_dim: 记忆的 key 维度（query 投影目标维度）
            value_dim: 记忆的 value 维度（readout 输出维度）
            num_memories: 记忆槽数量
            router_hidden_dim: 路由 MLP 的隐藏维度
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_memories = num_memories

        # Query 投影: h_t -> q_t
        self.query_proj = nn.Linear(hidden_dim, key_dim)

        # 读取路由器: h_t -> γ_t ∈ Δ^{num_memories}
        self.read_router = nn.Sequential(
            nn.Linear(hidden_dim, router_hidden_dim),
            nn.GELU(),
            nn.Linear(router_hidden_dim, num_memories),
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_states: List[torch.Tensor],
        return_details: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, Dict]:
        """从记忆中读取。

        Args:
            hidden_states: [B, D_h] 当前 backbone hidden state
            memory_states: 长度 num_memories 的列表，每个 [B, D_k, D_v]
            return_details: 是否返回详细中间结果

        Returns:
            readout: [B, D_v] 融合后的记忆读取
            details (optional): 包含 query, per-memory readouts, router weights
        """
        # 1. 投影 query
        query = self.query_proj(hidden_states)  # [B, D_k]

        # 2. 从每个记忆读取
        per_memory_readouts = []
        for M_i in memory_states:
            # query: [B, D_k] -> [B, 1, D_k]
            # M_i: [B, D_k, D_v]
            # r_i: [B, 1, D_v] -> [B, D_v]
            r_i = torch.bmm(query.unsqueeze(1), M_i).squeeze(1)
            per_memory_readouts.append(r_i)

        # Stack: [B, num_memories, D_v]
        readouts_stack = torch.stack(per_memory_readouts, dim=1)

        # 3. 计算路由权重 γ
        gamma_logits = self.read_router(hidden_states)  # [B, num_memories]
        gamma = F.softmax(gamma_logits, dim=-1)          # [B, num_memories]

        # 4. 加权合并: [B, num_memories, 1] × [B, num_memories, D_v] -> sum -> [B, D_v]
        readout = (gamma.unsqueeze(-1) * readouts_stack).sum(dim=1)  # [B, D_v]

        if return_details:
            details = {
                "query": query,
                "per_memory_readouts": per_memory_readouts,
                "gamma": gamma,
                "gamma_logits": gamma_logits,
            }
            return readout, details

        return readout

    def forward_sequence(
        self,
        hidden_seq: torch.Tensor,
        memory_states: List[torch.Tensor],
    ) -> torch.Tensor:
        """对序列中每个 token 进行记忆读取（批量化）。

        Args:
            hidden_seq: [B, T, D_h] backbone hidden states 序列
            memory_states: 长度 num_memories 的列表，每个 [B, D_k, D_v]

        Returns:
            readout_seq: [B, T, D_v]
        """
        B, T, D_h = hidden_seq.shape

        # 批量投影 query: [B, T, D_k]
        query_seq = self.query_proj(hidden_seq)

        # 从每个记忆批量读取
        # query_seq: [B, T, D_k], M_i: [B, D_k, D_v]
        # -> [B, T, D_v]
        per_memory_readouts = []
        for M_i in memory_states:
            r_i = torch.bmm(query_seq.view(B * T, 1, self.key_dim).reshape(B, T, self.key_dim), M_i)
            # 简化: torch.matmul handles batched: [B, T, D_k] × [B, D_k, D_v] -> [B, T, D_v]
            per_memory_readouts.append(r_i)

        # Stack: [B, T, num_memories, D_v]
        readouts_stack = torch.stack(per_memory_readouts, dim=2)

        # 路由权重: [B, T, num_memories]
        gamma_logits = self.read_router(hidden_seq)
        gamma = F.softmax(gamma_logits, dim=-1)

        # 加权合并: [B, T, num_memories, 1] × [B, T, num_memories, D_v] -> [B, T, D_v]
        readout_seq = (gamma.unsqueeze(-1) * readouts_stack).sum(dim=2)

        return readout_seq

    def get_router_stats(
        self,
        hidden_states: torch.Tensor,
    ) -> Dict[str, float]:
        """获取路由统计（用于日志）。

        Args:
            hidden_states: [B, D_h]

        Returns:
            stats: 路由权重的统计量
        """
        with torch.no_grad():
            gamma_logits = self.read_router(hidden_states)
            gamma = F.softmax(gamma_logits, dim=-1)
            stats = {}
            for i in range(self.num_memories):
                stats[f"read_gamma_mem{i}"] = gamma[:, i].mean().item()
            entropy = -(gamma * (gamma + 1e-8).log()).sum(dim=-1).mean()
            stats["read_gamma_entropy"] = entropy.item()
        return stats

    def extra_repr(self) -> str:
        return (
            f"hidden_dim={self.hidden_dim}, key_dim={self.key_dim}, "
            f"value_dim={self.value_dim}, num_memories={self.num_memories}"
        )
