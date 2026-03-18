"""
Block-level Evaluator: 在每个 block 结束时，对 block 内每个 token 位置打分，
判断哪些位置是有价值的 anchor 候选。

设计要点:
- 输入: block hidden states H_n [B, L, D], 可选的 memory summary r_n [B, D]
- 输出: scores [B, L]
- 轻量级: 默认用 1-2 层 MLP，可配置为小型 Transformer encoder
"""

from __future__ import annotations

import math
from typing import Optional, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPScorer(nn.Module):
    """简单的 MLP 打分器，对每个 token 独立打分。"""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 2,
        dropout: float = 0.1,
        use_memory_summary: bool = False,
    ) -> None:
        super().__init__()
        self.use_memory_summary = use_memory_summary
        input_dim = hidden_dim * 2 if use_memory_summary else hidden_dim

        layers = []
        for i in range(num_layers):
            in_d = input_dim if i == 0 else hidden_dim
            out_d = hidden_dim if i < num_layers - 1 else 1
            layers.append(nn.Linear(in_d, out_d))
            if i < num_layers - 1:
                layers.append(nn.GELU())
                layers.append(nn.Dropout(dropout))
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D]
            memory_summary: [B, D] 可选的 memory 上下文
        Returns:
            scores: [B, L]
        """
        if self.use_memory_summary and memory_summary is not None:
            # 将 memory summary 广播到每个 token 位置
            B, L, D = hidden_states.shape
            mem_expanded = memory_summary.unsqueeze(1).expand(B, L, -1)  # [B, L, D]
            x = torch.cat([hidden_states, mem_expanded], dim=-1)  # [B, L, 2D]
        else:
            x = hidden_states

        scores = self.net(x).squeeze(-1)  # [B, L]
        return scores


class TransformerScorer(nn.Module):
    """小型 Transformer encoder 打分器，允许 token 间交互后再打分。"""

    def __init__(
        self,
        hidden_dim: int,
        num_layers: int = 1,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_memory_summary: bool = False,
    ) -> None:
        super().__init__()
        self.use_memory_summary = use_memory_summary
        self.input_proj = None

        if use_memory_summary:
            # 将 concat 后的维度投影回 hidden_dim
            self.input_proj = nn.Linear(hidden_dim * 2, hidden_dim)

        if ff_dim is None:
            ff_dim = hidden_dim * 2  # 保持轻量

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.score_head = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        hidden_states: torch.Tensor,
        memory_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [B, L, D]
            memory_summary: [B, D]
        Returns:
            scores: [B, L]
        """
        x = hidden_states
        if self.use_memory_summary and memory_summary is not None:
            B, L, D = hidden_states.shape
            mem_expanded = memory_summary.unsqueeze(1).expand(B, L, -1)
            x = torch.cat([x, mem_expanded], dim=-1)
            x = self.input_proj(x)

        x = self.encoder(x)  # [B, L, D]
        scores = self.score_head(x).squeeze(-1)  # [B, L]
        return scores


class BlockEvaluator(nn.Module):
    """
    Block-level Evaluator: 对 block 内每个 token 位置打分。

    支持两种 scorer 类型:
    - "mlp": 独立对每个 token 打分（默认，最轻量）
    - "transformer": 允许 token 间交互后打分

    输出 scores 可直接交给 AnchorSelector 进行 top-k 选择。
    """

    def __init__(
        self,
        hidden_dim: int,
        scorer_type: Literal["mlp", "transformer"] = "mlp",
        num_layers: int = 2,
        num_heads: int = 4,
        ff_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_memory_summary: bool = False,
    ) -> None:
        """
        Args:
            hidden_dim: backbone 隐藏维度
            scorer_type: 评分器类型 "mlp" 或 "transformer"
            num_layers: 评分器层数
            num_heads: Transformer scorer 的注意力头数
            ff_dim: Transformer scorer 的 FFN 维度
            dropout: dropout 比例
            use_memory_summary: 是否使用 memory summary 辅助评分
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.scorer_type = scorer_type

        if scorer_type == "mlp":
            self.scorer = MLPScorer(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout,
                use_memory_summary=use_memory_summary,
            )
        elif scorer_type == "transformer":
            self.scorer = TransformerScorer(
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                ff_dim=ff_dim,
                dropout=dropout,
                use_memory_summary=use_memory_summary,
            )
        else:
            raise ValueError(f"未知的 scorer_type: {scorer_type}, 支持 'mlp' 或 'transformer'")

    def forward(
        self,
        block_hidden: torch.Tensor,
        memory_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        对 block 内每个 token 位置评分。

        Args:
            block_hidden: block 的 hidden states [B, L, D]
            memory_summary: 当前 memory 的聚合表示 [B, D]，可选

        Returns:
            scores: [B, L] 每个位置的 anchor 候选分数
        """
        return self.scorer(block_hidden, memory_summary)

    @torch.no_grad()
    def get_anchor_scores(
        self,
        block_hidden: torch.Tensor,
        memory_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理模式下获取 anchor 分数（不计算梯度）。"""
        self.eval()
        return self.forward(block_hidden, memory_summary)
