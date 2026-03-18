"""
回顾性注意力聚合（Retrospective Attention Gather）。

对每个选定的 anchor position，使用单头注意力在整个 block 上
聚合最相关的潜在信息，生成 gathered vector z_k。

公式：
    q_k = W_q h_{a_k}
    beta_{k,j} = softmax_j(q_k^T W_k h_j)
    z_k = sum_j beta_{k,j} W_v h_j
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RetrospectiveGather(nn.Module):
    """Anchor-guided retrospective attention gather。
    
    对每个 anchor，在当前 block 的所有 token 上做单头注意力，
    聚合出一个 gathered vector。v0 中不做跨 block 注意力。
    
    Args:
        hidden_dim: backbone hidden state 维度。
        gather_dim: gather 投影的内部维度（q/k/v 的维度），
                    默认为 hidden_dim // 4 以保持轻量。
        dropout: attention dropout 概率。
        scale_attn: 是否对注意力分数做 sqrt(d) scaling。
    """

    def __init__(
        self,
        hidden_dim: int,
        gather_dim: Optional[int] = None,
        dropout: float = 0.0,
        scale_attn: bool = True,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.gather_dim = gather_dim or (hidden_dim // 4)
        self.scale_attn = scale_attn

        # 低维 q/k/v 投影
        self.W_q = nn.Linear(hidden_dim, self.gather_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, self.gather_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, self.gather_dim, bias=False)

        # 输出投影：将 gather_dim 映射回 hidden_dim
        self.out_proj = nn.Linear(self.gather_dim, hidden_dim, bias=False)

        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        """Xavier 初始化投影权重。"""
        for module in [self.W_q, self.W_k, self.W_v, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)

    def forward(
        self,
        block_hidden: torch.Tensor,
        anchor_hidden: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """对每个 anchor 在 block 上做回顾性注意力聚合。
        
        Args:
            block_hidden: 整个 block 的 hidden states，
                          shape [B, L, D]。
            anchor_hidden: 选定的 anchor hidden states，
                           shape [B, K, D]，K 为 anchor 数量。
            block_mask: 可选的 block token mask，
                        shape [B, L]，True 表示有效 token。
        
        Returns:
            gathered: 聚合后的向量，shape [B, K, D]。
            attn_weights: 注意力权重，shape [B, K, L]，
                          用于可视化和调试。
        """
        B, L, D = block_hidden.shape
        K = anchor_hidden.shape[1]

        # 投影 anchor -> query, block -> key/value
        # q: [B, K, gather_dim]
        q = self.W_q(anchor_hidden)
        # k: [B, L, gather_dim]
        k = self.W_k(block_hidden)
        # v: [B, L, gather_dim]
        v = self.W_v(block_hidden)

        # 计算注意力分数: [B, K, L]
        # q: [B, K, d], k^T: [B, d, L] -> scores: [B, K, L]
        scores = torch.bmm(q, k.transpose(1, 2))

        if self.scale_attn:
            scores = scores / math.sqrt(self.gather_dim)

        # 应用 mask（如果提供）
        if block_mask is not None:
            # block_mask: [B, L] -> [B, 1, L]
            mask = block_mask.unsqueeze(1).expand(-1, K, -1)
            scores = scores.masked_fill(~mask, float("-inf"))

        # softmax 归一化
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # 加权聚合: [B, K, L] @ [B, L, gather_dim] -> [B, K, gather_dim]
        gathered = torch.bmm(attn_weights, v)

        # 投影回 hidden_dim: [B, K, D]
        gathered = self.out_proj(gathered)

        return gathered, attn_weights

    def gather_single_anchor(
        self,
        block_hidden: torch.Tensor,
        anchor_vec: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """便捷方法：对单个 anchor 向量做 gather。
        
        Args:
            block_hidden: [B, L, D]。
            anchor_vec: [B, D]。
            block_mask: 可选 [B, L]。
        
        Returns:
            gathered: [B, D]。
            attn_weights: [B, L]。
        """
        # 扩展维度 -> [B, 1, D]
        anchor_hidden = anchor_vec.unsqueeze(1)
        gathered, attn_weights = self.forward(
            block_hidden, anchor_hidden, block_mask
        )
        # 去掉 K 维度
        return gathered.squeeze(1), attn_weights.squeeze(1)
