"""
Memory Writer —— 写入 / 路由 / 保留决策头。

给定 gathered vector z_k，决定：
- alpha_k: 写入强度 ∈ [0, 1]
- rho_k: 路由分布（softmax over memories）
- lambda_k: 每个记忆的保留因子 ∈ [0, 1]

同时包含将 gathered vector 映射为 (key, value) 对的投影。
"""

from __future__ import annotations

from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class WriteDecision(NamedTuple):
    """写入决策的结构化输出。"""
    key: torch.Tensor        # [B, D_k] 写入 key
    value: torch.Tensor      # [B, D_v] 写入 value
    alpha: torch.Tensor      # [B, 1] 写入强度
    rho: torch.Tensor        # [B, num_memories] 路由分布
    lam: torch.Tensor        # [B, num_memories] 保留因子


class MemoryWriter(nn.Module):
    """Memory Write / Route / Retain 决策头。

    接收 gathered vector z_k（来自 RetrospectiveGather），
    输出完整的写入决策：key, value, alpha, rho, lambda。
    """

    def __init__(
        self,
        input_dim: int,
        key_dim: int = 64,
        value_dim: int = 64,
        num_memories: int = 3,
        hidden_dim: int = 128,
        default_retention_bias: float = 0.9,
    ) -> None:
        """
        Args:
            input_dim: gathered vector 的维度（通常等于 backbone hidden_dim 或 gather_dim）
            key_dim: 记忆矩阵的 key 维度
            value_dim: 记忆矩阵的 value 维度
            num_memories: 记忆槽数量
            hidden_dim: MLP 隐藏层维度
            default_retention_bias: 保留因子的初始偏置（偏向高保留）
        """
        super().__init__()
        self.input_dim = input_dim
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_memories = num_memories

        # ---- Key / Value 投影 ----
        self.key_proj = nn.Linear(input_dim, key_dim)
        self.value_proj = nn.Linear(input_dim, value_dim)

        # ---- 写入强度头 alpha ∈ [0, 1] ----
        self.alpha_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

        # ---- 路由头 rho ∈ Δ^{num_memories} (softmax) ----
        self.route_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_memories),
        )

        # ---- 保留头 lambda ∈ [0, 1]^{num_memories} ----
        self.retain_head = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_memories),
        )

        # 初始化：让保留因子偏向高保留
        self._init_retention_bias(default_retention_bias)

    def _init_retention_bias(self, bias: float) -> None:
        """初始化保留头的偏置，使 sigmoid(bias_val) ≈ bias。"""
        # sigmoid(x) = bias => x = log(bias / (1 - bias))
        import math
        bias_val = math.log(bias / (1.0 - bias + 1e-8))
        last_layer = self.retain_head[-1]
        if hasattr(last_layer, "bias") and last_layer.bias is not None:
            nn.init.constant_(last_layer.bias, bias_val)

    def forward(
        self,
        z: torch.Tensor,
        temperature: float = 1.0,
    ) -> WriteDecision:
        """计算写入决策。

        Args:
            z: [B, input_dim] gathered vector（来自 anchor 的聚合表示）
            temperature: 路由 softmax 的温度参数

        Returns:
            WriteDecision: (key, value, alpha, rho, lam)
        """
        # Key / Value 投影
        key = self.key_proj(z)      # [B, D_k]
        value = self.value_proj(z)  # [B, D_v]

        # 写入强度
        alpha = torch.sigmoid(self.alpha_head(z))  # [B, 1]

        # 路由分布
        route_logits = self.route_head(z)  # [B, num_memories]
        rho = F.softmax(route_logits / temperature, dim=-1)  # [B, num_memories]

        # 保留因子
        lam = torch.sigmoid(self.retain_head(z))  # [B, num_memories]

        return WriteDecision(key=key, value=value, alpha=alpha, rho=rho, lam=lam)

    def get_decision_stats(self, decision: WriteDecision) -> Dict[str, float]:
        """获取决策统计信息（用于日志）。

        Args:
            decision: WriteDecision

        Returns:
            stats: 统计量字典
        """
        with torch.no_grad():
            stats = {
                "alpha_mean": decision.alpha.mean().item(),
                "alpha_std": decision.alpha.std().item(),
            }
            # 每个记忆的平均路由权重
            for i in range(self.num_memories):
                stats[f"rho_mem{i}_mean"] = decision.rho[:, i].mean().item()
            # 每个记忆的平均保留因子
            for i in range(self.num_memories):
                stats[f"lam_mem{i}_mean"] = decision.lam[:, i].mean().item()
            # 路由熵（衡量路由均匀性）
            entropy = -(decision.rho * (decision.rho + 1e-8).log()).sum(dim=-1).mean()
            stats["route_entropy"] = entropy.item()
        return stats

    def extra_repr(self) -> str:
        return (
            f"input_dim={self.input_dim}, key_dim={self.key_dim}, "
            f"value_dim={self.value_dim}, num_memories={self.num_memories}"
        )
