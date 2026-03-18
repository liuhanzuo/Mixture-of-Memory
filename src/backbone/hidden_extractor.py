"""Hidden State 提取器。

提供便捷的工具函数，从 backbone 输出中按 block 提取、切分 hidden states，
供 evaluator / gather / memory 等下游模块使用。
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


class HiddenStateExtractor:
    """从 backbone hidden states 中按 block 提取特征。

    这是一个无参数的工具类，不继承 nn.Module。

    Args:
        block_size: 每个 block 的 token 数量，默认 64
    """

    def __init__(self, block_size: int = 64) -> None:
        self.block_size = block_size

    def split_into_blocks(
        self,
        hidden_states: torch.Tensor,
        pad_value: float = 0.0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """将连续 hidden states 切分为固定长度的 blocks。

        Args:
            hidden_states: [B, T, D] 完整序列的 hidden states
            pad_value: 对不足一个 block 的尾部进行 padding 的值

        Returns:
            blocks: [B, N, L, D]  N 是 block 数量，L 是 block_size
            block_mask: [B, N, L]  标记哪些位置是有效 token（非 padding）
        """
        B, T, D = hidden_states.shape
        L = self.block_size

        # 计算需要 padding 多少
        remainder = T % L
        if remainder != 0:
            pad_len = L - remainder
            padding = hidden_states.new_full((B, pad_len, D), pad_value)
            hidden_padded = torch.cat([hidden_states, padding], dim=1)
            # mask: 原始位置为 1，padding 为 0
            mask = torch.cat([
                hidden_states.new_ones(B, T),
                hidden_states.new_zeros(B, pad_len),
            ], dim=1)
        else:
            hidden_padded = hidden_states
            mask = hidden_states.new_ones(B, T)

        N = hidden_padded.size(1) // L
        blocks = hidden_padded.view(B, N, L, D)
        block_mask = mask.view(B, N, L)

        return blocks, block_mask

    def get_block(
        self,
        hidden_states: torch.Tensor,
        block_idx: int,
    ) -> torch.Tensor:
        """提取指定 block 的 hidden states。

        Args:
            hidden_states: [B, T, D]
            block_idx: block 索引（从 0 开始）

        Returns:
            [B, L', D]  其中 L' <= block_size
        """
        L = self.block_size
        start = block_idx * L
        end = min(start + L, hidden_states.size(1))
        return hidden_states[:, start:end, :]

    def num_blocks(self, seq_len: int) -> int:
        """计算序列长度对应的 block 数量（向上取整）。"""
        return (seq_len + self.block_size - 1) // self.block_size

    def get_block_boundaries(
        self,
        seq_len: int,
    ) -> list[tuple[int, int]]:
        """返回每个 block 的 (start, end) 边界列表。

        Args:
            seq_len: 序列总长度

        Returns:
            list of (start, end) tuples
        """
        boundaries = []
        for start in range(0, seq_len, self.block_size):
            end = min(start + self.block_size, seq_len)
            boundaries.append((start, end))
        return boundaries

    def extract_positions(
        self,
        hidden_states: torch.Tensor,
        indices: torch.Tensor,
    ) -> torch.Tensor:
        """从 hidden states 中按索引提取指定位置的特征。

        Args:
            hidden_states: [B, T, D]
            indices: [B, K] 要提取的位置索引

        Returns:
            [B, K, D] 提取的特征
        """
        B, T, D = hidden_states.shape
        K = indices.size(1)

        # 使用 gather 进行批量索引
        expanded_indices = indices.unsqueeze(-1).expand(B, K, D)  # [B, K, D]
        return torch.gather(hidden_states, dim=1, index=expanded_indices)
