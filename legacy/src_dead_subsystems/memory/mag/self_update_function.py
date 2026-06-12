"""
SelfUpdateFunction: MemoryLLM (ICML 2024) 的自更新函数。

核心思想:
    记忆池 M 通过可学习的线性变换进行更新:
        M' = W_retain · M + W_inject · h
    其中:
        - M: 当前记忆池 (per-layer hidden states)
        - h: 新输入的隐藏状态
        - W_retain: 保留旧记忆的线性变换 (决定保留多少旧信息)
        - W_inject: 注入新信息的线性变换 (决定融入多少新信息)

    这比直接使用原始 KV 有两个优势:
        1. 记忆更新是可学习的, 而非固定的 concat
        2. backbone 完全冻结, 只有 SU 的参数在训练

参考: MemoryLLM (https://arxiv.org/abs/2402.04624)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class SelfUpdateConfig:
    """SelfUpdateFunction 配置。

    Attributes:
        hidden_dim: Transformer 隐藏维度.
        num_layers: 需要更新的层数 (对应注入层).
        retain_init_scale: W_retain 初始化缩放 (建议接近 1.0, 保留大部分旧记忆).
        inject_init_scale: W_inject 初始化缩放 (建议较小如 0.01, 初期少注入).
        use_gate: 是否使用 sigmoid gate 控制注入量.
        use_residual: 是否在更新后加上残差连接 (M' = SU(M, h) + M).
        normalize_memory: 是否对更新后的记忆做 LayerNorm.
        dropout: Dropout 率 (0=不使用).
    """
    hidden_dim: int = 4096
    num_layers: int = 4
    retain_init_scale: float = 1.0
    inject_init_scale: float = 0.01
    use_gate: bool = True
    use_residual: bool = True
    normalize_memory: bool = True
    dropout: float = 0.0


class SelfUpdateFunction(nn.Module):
    """MemoryLLM 的 Self-Update Function (SU)。

    对每层 l, 记忆池更新公式:
        M'_l = σ(g_l) · (W_retain_l · M_l + W_inject_l · h_l) + (1 - σ(g_l)) · M_l

    简化版 (无 gate):
        M'_l = W_retain_l · M_l + W_inject_l · h_l

    其中:
        - W_retain_l: (D, D) 保留旧记忆
        - W_inject_l: (D, D) 注入新信息
        - g_l: gate 标量, 控制新旧信息的混合比例
        - σ: sigmoid 函数

    可训练参数 (per layer):
        - W_retain: D × D (可低秩分解为 D × r + r × D)
        - W_inject: D × D (可低秩分解为 D × r + r × D)
        - gate: 1 (标量, 可选)

    为了控制参数量, 使用低秩分解:
        W_retain ≈ A_retain @ B_retain  (D×r, r×D)
        W_inject ≈ A_inject @ B_inject  (D×r, r×D)
    """

    def __init__(self, config: SelfUpdateConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = SelfUpdateConfig()
        elif isinstance(config, dict):
            config = SelfUpdateConfig(**{
                k: v for k, v in config.items()
                if k in SelfUpdateConfig.__dataclass_fields__
            })
        self.config = config
        D = config.hidden_dim

        # Per-layer 保留和注入的线性变换
        # 初始化: W_retain ≈ I (保留旧记忆), W_inject ≈ 0 (初期少注入)
        self.W_retain = nn.ModuleList([
            self._init_identity_linear(D, config.retain_init_scale)
            for _ in range(config.num_layers)
        ])
        self.W_inject = nn.ModuleList([
            self._init_small_linear(D, config.inject_init_scale)
            for _ in range(config.num_layers)
        ])

        # 可选: Gate 控制
        self.gates = None
        if config.use_gate:
            self.gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0))  # 初始 gate=0.5 (sigmoid(0))
                for _ in range(config.num_layers)
            ])

        # 可选: LayerNorm
        self.layer_norms = None
        if config.normalize_memory:
            self.layer_norms = nn.ModuleList([
                nn.LayerNorm(D) for _ in range(config.num_layers)
            ])

        # 可选: Dropout
        self.dropout = nn.Dropout(config.dropout) if config.dropout > 0 else None

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[SelfUpdateFunction] 初始化: hidden_dim={D}, "
            f"num_layers={config.num_layers}, "
            f"use_gate={config.use_gate}, use_residual={config.use_residual}, "
            f"normalize={config.normalize_memory}, "
            f"total_params={num_params:,}"
        )

    @staticmethod
    def _init_identity_linear(dim: int, scale: float) -> nn.Linear:
        """初始化接近单位矩阵的线性层 (保留旧记忆)。"""
        linear = nn.Linear(dim, dim, bias=False)
        # 用单位矩阵 × scale 初始化
        with torch.no_grad():
            nn.init.eye_(linear.weight)
            linear.weight.mul_(scale)
        return linear

    @staticmethod
    def _init_small_linear(dim: int, scale: float) -> nn.Linear:
        """初始化接近零的线性层 (初期少注入)。"""
        linear = nn.Linear(dim, dim, bias=False)
        with torch.no_grad():
            nn.init.normal_(linear.weight, std=scale)
        return linear

    def forward(
        self,
        memory_pool: list[torch.Tensor],
        hidden_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        """对记忆池进行自更新。

        Args:
            memory_pool: [M_0, M_1, ..., M_{L-1}] 每层的记忆池。
                         每个 M_l 形状为 (B, D) 或 (B, num_kv_heads, r, head_dim)。
            hidden_states: [h_0, h_1, ..., h_{L-1}] 每层的新输入隐藏状态。
                          每个 h_l 形状应与对应 M_l 兼容。

        Returns:
            updated_memory: [M'_0, M'_1, ..., M'_{L-1}] 更新后的记忆池。
        """
        assert len(memory_pool) == len(hidden_states) == self.config.num_layers, \
            f"层数不匹配: memory={len(memory_pool)}, hidden={len(hidden_states)}, config={self.config.num_layers}"

        updated_memory = []
        for layer_idx in range(self.config.num_layers):
            m = memory_pool[layer_idx]    # 旧记忆
            h = hidden_states[layer_idx]  # 新输入

            # 保留 + 注入
            retained = self.W_retain[layer_idx](m)
            injected = self.W_inject[layer_idx](h)
            m_new = retained + injected

            # Gate: σ(g) · m_new + (1 - σ(g)) · m_old
            if self.gates is not None:
                gate = torch.sigmoid(self.gates[layer_idx])
                m_new = gate * m_new + (1 - gate) * m

            # 残差: M' = m_new + M
            if self.config.use_residual:
                m_new = m_new + m

            # LayerNorm
            if self.layer_norms is not None:
                m_new = self.layer_norms[layer_idx](m_new)

            # Dropout
            if self.dropout is not None:
                m_new = self.dropout(m_new)

            updated_memory.append(m_new)

        return updated_memory

    def update_kv_pool(
        self,
        kv_cache: dict[int, tuple[torch.Tensor, torch.Tensor]],
        per_layer_hs: dict[int, torch.Tensor],
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """对 KV cache 形式的记忆池进行自更新。

        这是最常用的接口: 将记忆 KV pairs 作为记忆池, 用 SU 更新。

        Args:
            kv_cache: {layer_idx: (keys, values)} 原始 KV cache。
                      keys: (B, num_kv_heads, N, head_dim)
                      values: (B, num_kv_heads, N, head_dim)
            per_layer_hs: {layer_idx: hidden_states} 每层的新隐藏状态。
                         hidden_states: (B, T, D)

        Returns:
            updated_kv_cache: {layer_idx: (keys', values')} 更新后的 KV cache。
        """
        layer_indices = sorted(kv_cache.keys())
        assert len(layer_indices) == self.config.num_layers, \
            f"KV cache 层数 ({len(layer_indices)}) != config ({self.config.num_layers})"

        # 分别对 keys 和 values 进行更新
        key_memory_pool = []
        value_memory_pool = []
        hs_for_update = []

        for l_idx in layer_indices:
            keys, values = kv_cache[l_idx]
            # 对每个 head 独立更新: 将 (B, H, N, d) reshape 为 (B, H*N*d) 或逐 token 更新
            # 简化: 对 KV 做 mean pooling 得到 (B, D) 级别的表示, 更新后 broadcast 回去
            B, H, N, d = keys.shape
            D = H * d

            # Mean pool over token dimension: (B, H, N, d) → (B, H*d) = (B, D)
            key_repr = keys.mean(dim=2).reshape(B, D)
            value_repr = values.mean(dim=2).reshape(B, D)

            key_memory_pool.append(key_repr)
            value_memory_pool.append(value_repr)

            # 新隐藏状态: mean pool over sequence dim
            hs = per_layer_hs[l_idx]  # (B, T, D)
            hs_mean = hs.mean(dim=1)  # (B, D)
            hs_for_update.append(hs_mean)

        # 分别更新 key 和 value 表示
        updated_keys_repr = self.forward(key_memory_pool, hs_for_update)
        updated_values_repr = self.forward(value_memory_pool, hs_for_update)

        # Broadcast 回原始形状
        updated_kv_cache = {}
        for i, l_idx in enumerate(layer_indices):
            orig_keys, orig_values = kv_cache[l_idx]
            B, H, N, d = orig_keys.shape

            # (B, D) → (B, H, 1, d) → broadcast to (B, H, N, d)
            new_key_repr = updated_keys_repr[i].reshape(B, H, d).unsqueeze(2)
            new_value_repr = updated_values_repr[i].reshape(B, H, d).unsqueeze(2)

            # 加上残差: 用 delta 的方式, 避免完全替换
            key_delta = new_key_repr - key_memory_pool[i].reshape(B, H, d).unsqueeze(2)
            value_delta = new_value_repr - value_memory_pool[i].reshape(B, H, d).unsqueeze(2)

            updated_keys = orig_keys + key_delta.expand(-1, -1, N, -1)
            updated_values = orig_values + value_delta.expand(-1, -1, N, -1)

            updated_kv_cache[l_idx] = (updated_keys, updated_values)

        return updated_kv_cache

    def get_stats(self) -> dict[str, Any]:
        """返回 SU 函数的统计信息。"""
        stats = {}
        for i in range(self.config.num_layers):
            stats[f"retain_norm_L{i}"] = self.W_retain[i].weight.norm().item()
            stats[f"inject_norm_L{i}"] = self.W_inject[i].weight.norm().item()
            if self.gates is not None:
                stats[f"gate_L{i}"] = torch.sigmoid(self.gates[i]).item()
        return stats

    def __repr__(self) -> str:
        gate_info = ""
        if self.gates is not None:
            gates = [f"{torch.sigmoid(g).item():.3f}" for g in self.gates]
            gate_info = f", gates=[{','.join(gates)}]"
        return (
            f"SelfUpdateFunction(layers={self.config.num_layers}, "
            f"D={self.config.hidden_dim}{gate_info})"
        )


class LowRankSelfUpdateFunction(nn.Module):
    """低秩版本的 Self-Update Function, 大幅减少参数量。

    当 hidden_dim=4096 时, 标准 SU 每层有 2×4096²=33.5M 参数。
    低秩版本 (rank=64) 每层只有 2×2×4096×64=1.0M 参数, 减少 32x。

    数学:
        W_retain ≈ A_r @ B_r  (D×r, r×D)
        W_inject ≈ A_i @ B_i  (D×r, r×D)
        M' = A_r @ B_r @ M + A_i @ B_i @ h

    初始化:
        A_r ~ N(0, 1/√D), B_r = I[:r, :] (保留主成分)
        A_i ~ N(0, σ), B_i = 0 (初期不注入)
    """

    def __init__(
        self,
        hidden_dim: int = 4096,
        num_layers: int = 4,
        rank: int = 64,
        inject_scale: float = 0.01,
        use_gate: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rank = rank

        # 低秩分解参数
        self.A_retain = nn.ParameterList()
        self.B_retain = nn.ParameterList()
        self.A_inject = nn.ParameterList()
        self.B_inject = nn.ParameterList()

        for _ in range(num_layers):
            # Retain: A ~ small random, B ~ truncated identity
            a_r = nn.Parameter(torch.randn(hidden_dim, rank) / math.sqrt(hidden_dim))
            b_r = nn.Parameter(torch.zeros(rank, hidden_dim))
            # 用截断单位矩阵初始化 B_retain
            with torch.no_grad():
                b_r[:min(rank, hidden_dim), :] = torch.eye(hidden_dim)[:min(rank, hidden_dim), :]
            self.A_retain.append(a_r)
            self.B_retain.append(b_r)

            # Inject: A ~ small random, B = 0
            a_i = nn.Parameter(torch.randn(hidden_dim, rank) * inject_scale)
            b_i = nn.Parameter(torch.zeros(rank, hidden_dim))
            self.A_inject.append(a_i)
            self.B_inject.append(b_i)

        # Gate
        self.gates = None
        if use_gate:
            self.gates = nn.ParameterList([
                nn.Parameter(torch.tensor(0.0))
                for _ in range(num_layers)
            ])

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[LowRankSU] 初始化: D={hidden_dim}, layers={num_layers}, "
            f"rank={rank}, total_params={num_params:,}"
        )

    def forward(
        self,
        memory_pool: list[torch.Tensor],
        hidden_states: list[torch.Tensor],
    ) -> list[torch.Tensor]:
        updated = []
        for i in range(self.num_layers):
            m = memory_pool[i]
            h = hidden_states[i]

            # Low-rank: W @ x = A @ (B @ x)
            retained = m @ self.B_retain[i].t() @ self.A_retain[i].t()  # (B, D)
            injected = h @ self.B_inject[i].t() @ self.A_inject[i].t()  # (B, D)
            m_new = retained + injected

            if self.gates is not None:
                gate = torch.sigmoid(self.gates[i])
                m_new = gate * m_new + (1 - gate) * m

            m_new = m_new + m  # residual
            updated.append(m_new)

        return updated
