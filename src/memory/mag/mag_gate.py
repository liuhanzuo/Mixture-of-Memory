"""
MAGGate: Memory-Augmented Generation 门控模块。

实现 Titans 风格的记忆向量注入, 在 Transformer 中间若干层通过
CrossAttention + Sigmoid Gate + Residual 将 L2/L3 记忆融合进隐藏状态。

核心公式 (单层):
    m_agg = CrossAttn(Q=h, K=M, V=M)        # h ∈ (B, T, D), M ∈ (B, K, D)
    g = σ(W_g [h; m_agg])                     # 门控信号
    h' = h + g ⊙ (W_o m_agg)                  # 残差融合

与 L1 Gate 的关系:
    L1 Gate:  h' = h + g₁ ⊙ W₁ r₁   (r₁ = L1 token-level readout)
    MAG Gate: h' = h + g₂ ⊙ W₂ m    (m = L2/L3 turn-level memory vectors)
    最终:     h_out = MAG(L1(h))      # 串联, L1 先做, MAG 再做

注入位置:
    可配置在指定的中间层注入 (如 [4, 8, 12, 16] 层),
    每层共享或独立的 gate 参数。
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
class MAGGateConfig:
    """MAGGate 配置。

    Attributes:
        hidden_dim: Transformer 隐藏维度 (= backbone hidden_dim).
        num_heads: CrossAttention 头数.
        memory_dim: 记忆向量维度 (= MemoryEncoder output_dim, 通常 = hidden_dim).
        injection_layers: 注入的 Transformer 层索引列表 (从 0 开始).
                          例如 [4, 8, 12, 16] 表示在第 5, 9, 13, 17 层后注入.
                          如果为空, 则只在最后一层注入.
        share_parameters: 是否在不同注入层之间共享 gate 参数.
        gate_init_bias: gate 偏置初始值 (负值使初始 gate 接近 0, 避免早期破坏).
        dropout: 注意力和输出的 dropout.
        use_layer_norm: 是否在 CrossAttn 后使用 LayerNorm.
    """
    hidden_dim: int = 2048
    num_heads: int = 8
    memory_dim: int = 2048
    injection_layers: list[int] = field(default_factory=lambda: [])
    share_parameters: bool = True
    gate_init_bias: float = -2.0  # 初始 gate ≈ sigmoid(-2) ≈ 0.12, 保守注入
    dropout: float = 0.0
    use_layer_norm: bool = True


class _MAGCrossAttnBlock(nn.Module):
    """单个 MAG 注入块: CrossAttn + Gate + Residual。

    实现:
        m_agg = CrossAttn(Q=W_q h, K=W_k M, V=W_v M)
        g = σ(W_g [h_pooled; m_agg_pooled] + bias)   # 全局 gate
        h' = h + g ⊙ (W_o m_agg)
    """

    def __init__(self, config: MAGGateConfig):
        super().__init__()
        self.config = config
        D = config.hidden_dim
        M_dim = config.memory_dim
        n_heads = config.num_heads
        head_dim = D // n_heads

        assert D % n_heads == 0, f"hidden_dim({D}) 必须能被 num_heads({n_heads}) 整除"

        self.n_heads = n_heads
        self.head_dim = head_dim

        # CrossAttention 投影
        self.q_proj = nn.Linear(D, D, bias=False)       # Q: 来自 hidden states
        self.k_proj = nn.Linear(M_dim, D, bias=False)   # K: 来自 memory vectors
        self.v_proj = nn.Linear(M_dim, D, bias=False)   # V: 来自 memory vectors
        self.o_proj = nn.Linear(D, D, bias=False)        # 输出投影

        # 门控: sigmoid gate
        # 输入: [h_mean, m_agg_mean] → (2D,) → (D,)
        self.gate_proj = nn.Linear(D * 2, D, bias=True)
        # 初始化 bias 为负值, 使初始 gate 接近 0
        nn.init.constant_(self.gate_proj.bias, config.gate_init_bias)

        # 可选: LayerNorm
        self.layer_norm: nn.LayerNorm | None = None
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(D)

        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()
        self.output_dropout = nn.Dropout(config.dropout) if config.dropout > 0 else nn.Identity()

        # 缩放因子
        self._scale = math.sqrt(head_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_vectors: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        selection_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """MAG 注入: CrossAttn + Gate + Residual。

        Args:
            hidden_states: (B, T, D) Transformer 隐藏状态.
            memory_vectors: (B, K, M_dim) 编码后的记忆向量.
            memory_mask: (B, K) 记忆 mask (1=有效, 0=padding).
            selection_weights: (B, K) Context Selector 的选择权重, ∈ [0, 1].
                              如果提供, 会与 attention weights 相乘实现 soft selection.

        Returns:
            output: (B, T, D) 融合记忆后的隐藏状态.
        """
        B, T, D = hidden_states.shape
        K = memory_vectors.shape[1]

        if K == 0:
            return hidden_states

        # ---- Cross Attention ---- #
        # Q: (B, T, D) → (B, n_heads, T, head_dim)
        Q = self.q_proj(hidden_states).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        # K: (B, K, D) → (B, n_heads, K, head_dim)
        K_mem = self.k_proj(memory_vectors).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)
        # V: (B, K, D) → (B, n_heads, K, head_dim)
        V_mem = self.v_proj(memory_vectors).view(B, K, self.n_heads, self.head_dim).transpose(1, 2)

        # Attention scores: (B, n_heads, T, K)
        attn_weights = torch.matmul(Q, K_mem.transpose(-2, -1)) / self._scale

        # Memory mask: 将 padding 位置设为 -inf
        if memory_mask is not None:
            # (B, K) → (B, 1, 1, K)
            mask = memory_mask.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights.masked_fill(~mask.bool(), float("-inf"))

        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, n_heads, T, K)
        attn_weights = self.attn_dropout(attn_weights)

        # 如果有 selection_weights, 将其与 attention weights 相乘
        # 实现 soft gating: 被 selector 判断为无用的记忆, attention 权重被压低
        if selection_weights is not None:
            # (B, K) → (B, 1, 1, K)
            sel_w = selection_weights.unsqueeze(1).unsqueeze(2)
            attn_weights = attn_weights * sel_w

        # Attention output: (B, n_heads, T, head_dim)
        attn_output = torch.matmul(attn_weights, V_mem)

        # Reshape & project: (B, T, D)
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, D)
        m_agg = self.o_proj(attn_output)  # (B, T, D)

        # 可选 LayerNorm
        if self.layer_norm is not None:
            m_agg = self.layer_norm(m_agg)

        # ---- Sigmoid Gate ---- #
        # 全局 gate: 基于 hidden_states 和 m_agg 的交互
        # 逐 token 门控: (B, T, D)
        gate_input = torch.cat([hidden_states, m_agg], dim=-1)  # (B, T, 2D)
        gate = torch.sigmoid(self.gate_proj(gate_input))  # (B, T, D)

        # ---- Residual ---- #
        m_agg = self.output_dropout(m_agg)
        output = hidden_states + gate * m_agg  # (B, T, D)

        return output


class MAGGate(nn.Module):
    """MAG 门控管理器: 管理多个注入层的 _MAGCrossAttnBlock。

    根据配置在指定的 Transformer 层注入记忆向量。
    可选择共享或独立参数。

    Usage (在 backbone forward 中)::

        mag = MAGGate(config)
        for layer_idx, layer in enumerate(transformer_layers):
            h = layer(h)
            h = mag.inject(layer_idx, h, memory_vectors, memory_mask, selection_weights)

    或者作为后处理 (在 backbone forward 之后)::

        # 获取所有层的 hidden states
        outputs = backbone(input_ids, output_hidden_states=True)
        all_hidden = outputs.all_hidden_states  # [h_0, h_1, ..., h_L]

        # 在指定层注入 (修改 hidden states)
        for layer_idx in mag.injection_layers:
            all_hidden[layer_idx + 1] = mag.inject(
                layer_idx, all_hidden[layer_idx + 1], memory_vectors, ...
            )
    """

    def __init__(self, config: MAGGateConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = MAGGateConfig()
        elif isinstance(config, dict):
            config = MAGGateConfig(**{
                k: v for k, v in config.items()
                if k in MAGGateConfig.__dataclass_fields__
            })
        self.config = config

        # 确定注入层
        self._injection_layers = set(config.injection_layers)

        # 创建注入块
        if config.share_parameters:
            # 所有注入层共享同一组参数
            self._shared_block = _MAGCrossAttnBlock(config)
            self._layer_blocks: nn.ModuleDict = nn.ModuleDict()
        else:
            # 每层独立参数
            self._shared_block = None
            self._layer_blocks = nn.ModuleDict({
                str(layer_idx): _MAGCrossAttnBlock(config)
                for layer_idx in config.injection_layers
            })

        # 统计
        self._inject_count = 0
        self._last_gate_values: torch.Tensor | None = None

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[MAGGate] 初始化: injection_layers={config.injection_layers}, "
            f"share_params={config.share_parameters}, "
            f"num_heads={config.num_heads}, "
            f"total_params={num_params:,}"
        )

    @property
    def injection_layers(self) -> set[int]:
        """返回需要注入的层索引集合。"""
        return self._injection_layers

    def set_injection_layers(self, layers: list[int]) -> None:
        """动态修改注入层 (用于实验)。"""
        self._injection_layers = set(layers)
        if not self.config.share_parameters:
            # 为新的层创建块 (如果还没有)
            for idx in layers:
                if str(idx) not in self._layer_blocks:
                    self._layer_blocks[str(idx)] = _MAGCrossAttnBlock(self.config)

    def should_inject(self, layer_idx: int) -> bool:
        """判断某一层是否需要注入 MAG 记忆。"""
        return layer_idx in self._injection_layers

    def inject(
        self,
        layer_idx: int,
        hidden_states: torch.Tensor,
        memory_vectors: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        selection_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """在指定层注入 MAG 记忆。

        如果该层不在 injection_layers 中, 直接返回原始 hidden_states。

        Args:
            layer_idx: 当前 Transformer 层索引.
            hidden_states: (B, T, D) 当前层的隐藏状态.
            memory_vectors: (B, K, D) 编码后的记忆向量.
            memory_mask: (B, K) 记忆有效性 mask.
            selection_weights: (B, K) Context Selector 选择权重.

        Returns:
            output: (B, T, D) 可能注入记忆后的隐藏状态.
        """
        if not self.should_inject(layer_idx):
            return hidden_states

        if memory_vectors.shape[1] == 0:
            return hidden_states

        # 选择对应的注入块
        if self.config.share_parameters and self._shared_block is not None:
            block = self._shared_block
        else:
            block_key = str(layer_idx)
            if block_key not in self._layer_blocks:
                logger.warning(
                    f"[MAGGate] 层 {layer_idx} 没有对应的注入块, 跳过注入"
                )
                return hidden_states
            block = self._layer_blocks[block_key]

        output = block(
            hidden_states=hidden_states,
            memory_vectors=memory_vectors,
            memory_mask=memory_mask,
            selection_weights=selection_weights,
        )

        self._inject_count += 1
        return output

    def inject_into_all_hidden_states(
        self,
        all_hidden_states: list[torch.Tensor],
        memory_vectors: torch.Tensor,
        memory_mask: torch.Tensor | None = None,
        selection_weights: torch.Tensor | None = None,
    ) -> list[torch.Tensor]:
        """对一组 hidden states (来自 backbone output) 批量注入。

        all_hidden_states[0] = embedding output
        all_hidden_states[i+1] = layer i 的 output

        所以注入第 i 层 = 修改 all_hidden_states[i+1]。

        Args:
            all_hidden_states: [h_embed, h_layer0, h_layer1, ...] 共 L+1 个.
            memory_vectors: (B, K, D).
            memory_mask: (B, K).
            selection_weights: (B, K).

        Returns:
            modified_hidden_states: 注入记忆后的 hidden states 列表.
        """
        if memory_vectors.shape[1] == 0:
            return all_hidden_states

        result = list(all_hidden_states)  # 浅拷贝
        for layer_idx in sorted(self._injection_layers):
            # all_hidden_states 索引: layer_idx 对应 all_hidden_states[layer_idx + 1]
            hs_idx = layer_idx + 1
            if hs_idx < len(result):
                result[hs_idx] = self.inject(
                    layer_idx=layer_idx,
                    hidden_states=result[hs_idx],
                    memory_vectors=memory_vectors,
                    memory_mask=memory_mask,
                    selection_weights=selection_weights,
                )

        return result

    def get_stats(self) -> dict[str, Any]:
        """返回 MAG 统计信息。"""
        return {
            "mag_inject_count": self._inject_count,
            "mag_injection_layers": sorted(self._injection_layers),
            "mag_share_parameters": self.config.share_parameters,
        }
