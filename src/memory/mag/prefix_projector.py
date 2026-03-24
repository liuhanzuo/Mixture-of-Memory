"""
PrefixProjector: MAC (Memory-Augmented Context) 的核心组件。

将 MemoryEncoder 输出的记忆向量映射为 soft prefix tokens,
拼接到 backbone 的 input embedding 前面。

核心思路:
    与 MAG (在 hidden state 中间层做残差注入) 不同, MAC 在输入端注入:
    
    MAG:  [prompt] → backbone (被 MAG 反复修改 hidden state) → 可能崩溃的输出
    MAC:  [soft_prefix_tokens] + [prompt] → 原封不动的 backbone → 正常输出
    
    backbone 的注意力机制自然会从 prefix tokens 中提取有用信息,
    不需要任何"手术式注入", 零侵入地保持 backbone 语言能力。

公式:
    memory_vecs: (B, K, D) 来自 MemoryEncoder
    soft_tokens: (B, K*N, D) = PrefixProjector(memory_vecs)
    full_embeds: (B, K*N+T, D) = cat([soft_tokens, prompt_embeds])
    output = backbone(inputs_embeds=full_embeds)  # backbone 完全不修改
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


@dataclass
class PrefixProjectorConfig:
    """PrefixProjector 配置。

    Attributes:
        hidden_dim: backbone 隐藏维度 (= memory vector 维度).
        tokens_per_memory: 每条记忆映射为多少个 soft token.
                          推荐 2~8, 越大表达力越强但占用序列长度越多.
        num_mlp_layers: MLP 层数 (推荐 2~3).
        mlp_expansion: MLP 中间层相对 hidden_dim 的扩展倍数.
        dropout: MLP 内部 dropout.
        use_layer_norm: 是否在输出前做 LayerNorm.
        use_gating: 是否为每条记忆生成一个重要性 gate (标量).
                    如果启用, 允许模型学习对不同记忆施加不同的注入强度.
        init_scale: 输出层初始化缩放因子.
                    设置较小值 (如 0.01) 使初始 prefix 接近零向量,
                    训练初期不扰乱 backbone.
    """
    hidden_dim: int = 4096
    tokens_per_memory: int = 4
    num_mlp_layers: int = 2
    mlp_expansion: float = 1.5
    dropout: float = 0.1
    use_layer_norm: bool = True
    use_gating: bool = True
    init_scale: float = 0.01


class PrefixProjector(nn.Module):
    """将记忆向量映射为 soft prefix tokens。

    每条记忆向量 (D,) 通过 MLP 映射为 N 个 soft tokens (N, D),
    最终拼接到 prompt embedding 前面作为 prefix。

    与 MAG 的关键区别:
    - MAG: 在 backbone 中间层做残差注入, 侵入性强
    - MAC: 在输入端拼接 soft prefix, backbone 完全不修改

    Usage::

        projector = PrefixProjector(config)
        
        # 训练/推理
        memory_vecs = memory_encoder.encode_texts_deep(memory_texts)  # (K, D)
        memory_vecs = memory_vecs.unsqueeze(0)  # (1, K, D)
        prefix_tokens = projector(memory_vecs, selection_weights)  # (1, K*N, D)
        
        prompt_embeds = backbone.model.embed_tokens(input_ids)  # (1, T, D)
        full_embeds = torch.cat([prefix_tokens, prompt_embeds], dim=1)  # (1, K*N+T, D)
        
        # backbone 正常前向, 完全不修改!
        output = backbone(inputs_embeds=full_embeds)
    """

    def __init__(self, config: PrefixProjectorConfig | dict[str, Any] | None = None):
        super().__init__()
        if config is None:
            config = PrefixProjectorConfig()
        elif isinstance(config, dict):
            config = PrefixProjectorConfig(**{
                k: v for k, v in config.items()
                if k in PrefixProjectorConfig.__dataclass_fields__
            })
        self.config = config

        D = config.hidden_dim
        N = config.tokens_per_memory
        mid_dim = int(D * config.mlp_expansion)

        # ---- 主 MLP: (D,) → (N*D,) ---- #
        layers: list[nn.Module] = []

        # 第一层
        layers.append(nn.Linear(D, mid_dim))
        layers.append(nn.GELU())
        if config.dropout > 0:
            layers.append(nn.Dropout(config.dropout))

        # 中间层
        for _ in range(config.num_mlp_layers - 2):
            layers.append(nn.Linear(mid_dim, mid_dim))
            layers.append(nn.GELU())
            if config.dropout > 0:
                layers.append(nn.Dropout(config.dropout))

        # 输出层: mid_dim → N * D
        output_proj = nn.Linear(mid_dim, N * D)
        # ★ 小初始化: 训练初期 prefix 接近零, 不扰乱 backbone
        nn.init.normal_(output_proj.weight, std=config.init_scale)
        nn.init.zeros_(output_proj.bias)
        layers.append(output_proj)

        self.mlp = nn.Sequential(*layers)

        # ---- 可选: LayerNorm ---- #
        self.layer_norm: nn.LayerNorm | None = None
        if config.use_layer_norm:
            self.layer_norm = nn.LayerNorm(D)

        # ---- 可选: 记忆重要性 Gate ---- #
        # 为每条记忆生成一个标量 gate ∈ [0, 1]
        # 训练时学习: 哪些记忆应该被注入, 哪些应该被忽略
        self.gate_proj: nn.Linear | None = None
        if config.use_gating:
            self.gate_proj = nn.Linear(D, 1)
            # 初始 bias 为 0 → 初始 gate ≈ 0.5
            nn.init.zeros_(self.gate_proj.bias)

        # 记录
        self._tokens_per_memory = N
        self._hidden_dim = D

        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"[PrefixProjector] 初始化: hidden_dim={D}, "
            f"tokens_per_memory={N}, num_mlp_layers={config.num_mlp_layers}, "
            f"gating={config.use_gating}, total_params={num_params:,}"
        )

    @property
    def tokens_per_memory(self) -> int:
        return self._tokens_per_memory

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        memory_vectors: torch.Tensor,
        selection_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """将记忆向量映射为 soft prefix tokens。

        Args:
            memory_vectors: (B, K, D) 编码后的记忆向量.
            selection_weights: (B, K) ContextSelector 的选择权重, ∈ [0, 1].
                              如果提供, 会与 gate 相乘来控制每条记忆的注入强度.

        Returns:
            prefix_tokens: (B, K*N, D) soft prefix tokens,
                          可直接拼接到 prompt embedding 前面.
        """
        B, K, D = memory_vectors.shape
        N = self._tokens_per_memory

        # 统一 dtype
        param_dtype = next(self.mlp.parameters()).dtype
        memory_vectors = memory_vectors.to(dtype=param_dtype)

        # MLP: (B, K, D) → (B, K, N*D)
        projected = self.mlp(memory_vectors)

        # Reshape: (B, K, N*D) → (B, K, N, D)
        projected = projected.view(B, K, N, D)

        # 可选: LayerNorm (在每个 token 维度上)
        if self.layer_norm is not None:
            projected = self.layer_norm(projected)

        # ---- 计算每条记忆的 gate ---- #
        if self.gate_proj is not None:
            # gate: (B, K, 1)
            gate = torch.sigmoid(self.gate_proj(memory_vectors))  # (B, K, 1)
            # 扩展为 (B, K, N, 1) 并应用
            gate = gate.unsqueeze(2).expand(B, K, N, 1)
            projected = projected * gate

        # ---- 应用 selection_weights ---- #
        if selection_weights is not None:
            # (B, K) → (B, K, 1, 1) 扩展到 (B, K, N, D)
            sel_w = selection_weights.unsqueeze(-1).unsqueeze(-1).to(dtype=param_dtype)
            sel_w = sel_w.expand(B, K, N, 1)
            projected = projected * sel_w

        # Reshape: (B, K, N, D) → (B, K*N, D)
        prefix_tokens = projected.reshape(B, K * N, D)

        return prefix_tokens

    def compute_prefix_with_mask(
        self,
        memory_vectors: torch.Tensor,
        selection_weights: torch.Tensor | None = None,
        prefix_attention_value: float = 1.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """生成 prefix tokens 及对应的 attention mask。

        Args:
            memory_vectors: (B, K, D)
            selection_weights: (B, K)
            prefix_attention_value: prefix 位置的 attention mask 值.

        Returns:
            prefix_tokens: (B, K*N, D)
            prefix_mask: (B, K*N) attention mask for prefix positions.
        """
        prefix_tokens = self.forward(memory_vectors, selection_weights)
        B = prefix_tokens.shape[0]
        prefix_len = prefix_tokens.shape[1]

        # prefix mask: 全部为有效 (1)
        prefix_mask = torch.full(
            (B, prefix_len),
            prefix_attention_value,
            device=prefix_tokens.device,
            dtype=torch.long,
        )

        return prefix_tokens, prefix_mask
