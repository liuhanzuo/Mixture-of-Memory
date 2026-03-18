"""
Block Buffer：管理 block-wise 序列处理的缓冲区。

将输入序列按固定长度 L 切分成 block，提供迭代器接口
方便逐 block 处理。同时支持缓存每个 block 的 hidden states
以供 retrospective gather 使用。
"""

from typing import Dict, Iterator, List, Optional, Tuple

import torch


class BlockBuffer:
    """Block-wise 序列处理缓冲区。
    
    将长序列切分为固定长度的 block，维护处理状态，
    并缓存每个 block 的 hidden states 用于后续的
    anchor evaluation 和 retrospective gather。
    
    Args:
        block_size: 每个 block 的 token 数量，默认 64。
        max_cached_blocks: 最多缓存多少个 block 的 hidden states，
                           0 表示不限制。用于控制显存。
    """

    def __init__(
        self,
        block_size: int = 64,
        max_cached_blocks: int = 0,
    ):
        self.block_size = block_size
        self.max_cached_blocks = max_cached_blocks

        # 缓存：block_index -> hidden_states [B, L, D]
        self._cache: Dict[int, torch.Tensor] = {}
        self._block_count: int = 0

    def reset(self) -> None:
        """清空缓冲区，准备处理新序列。"""
        self._cache.clear()
        self._block_count = 0

    @property
    def num_blocks(self) -> int:
        """已处理的 block 数量。"""
        return self._block_count

    @property
    def cached_block_indices(self) -> List[int]:
        """当前缓存中的 block 索引。"""
        return sorted(self._cache.keys())

    def split_into_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> List[Dict[str, torch.Tensor]]:
        """将输入序列切分为 block 列表。
        
        Args:
            input_ids: token ids，shape [B, S]。
            attention_mask: 可选，shape [B, S]。
        
        Returns:
            blocks: 字典列表，每个字典包含：
                - "input_ids": [B, L] 或 [B, L'] (最后一个 block 可能不足 L)
                - "attention_mask": [B, L] 或 [B, L'] (如果提供了 mask)
                - "block_index": int
                - "start_pos": int，该 block 在原序列中的起始位置
        """
        B, S = input_ids.shape
        blocks = []

        for i in range(0, S, self.block_size):
            end = min(i + self.block_size, S)
            block_dict: Dict[str, torch.Tensor] = {
                "input_ids": input_ids[:, i:end],
                "block_index": len(blocks),
                "start_pos": i,
            }
            if attention_mask is not None:
                block_dict["attention_mask"] = attention_mask[:, i:end]
            blocks.append(block_dict)

        return blocks

    def iterate_blocks(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Iterator[Dict[str, torch.Tensor]]:
        """迭代器接口，逐 block 产出。
        
        Args:
            input_ids: [B, S]。
            attention_mask: 可选 [B, S]。
        
        Yields:
            block_dict: 同 split_into_blocks 的返回格式。
        """
        blocks = self.split_into_blocks(input_ids, attention_mask)
        for block in blocks:
            yield block

    def cache_block_hidden(
        self,
        block_index: int,
        hidden_states: torch.Tensor,
    ) -> None:
        """缓存一个 block 的 hidden states。
        
        Args:
            block_index: block 的序号。
            hidden_states: 该 block 的 hidden states，shape [B, L, D]。
        """
        # 如果有缓存上限，淘汰最旧的
        if (
            self.max_cached_blocks > 0
            and len(self._cache) >= self.max_cached_blocks
            and block_index not in self._cache
        ):
            oldest_key = min(self._cache.keys())
            del self._cache[oldest_key]

        # detach 并存储，避免梯度图过大
        self._cache[block_index] = hidden_states.detach()
        self._block_count = max(self._block_count, block_index + 1)

    def get_block_hidden(
        self,
        block_index: int,
    ) -> Optional[torch.Tensor]:
        """获取缓存的 block hidden states。
        
        Args:
            block_index: block 序号。
        
        Returns:
            hidden_states: [B, L, D] 或 None（未缓存时）。
        """
        return self._cache.get(block_index, None)

    def get_recent_blocks(
        self,
        n: int = 1,
    ) -> List[Tuple[int, torch.Tensor]]:
        """获取最近 n 个 block 的 hidden states。
        
        Args:
            n: 要获取的 block 数量。
        
        Returns:
            list of (block_index, hidden_states) 元组，
            按 block_index 升序排列。
        """
        indices = sorted(self._cache.keys())
        recent = indices[-n:] if n < len(indices) else indices
        return [(idx, self._cache[idx]) for idx in recent]

    def get_all_cached_hidden(self) -> Optional[torch.Tensor]:
        """将所有缓存的 block hidden states 拼接为一个张量。
        
        Returns:
            concatenated: [B, total_tokens, D] 或 None（缓存为空时）。
        """
        if not self._cache:
            return None

        indices = sorted(self._cache.keys())
        tensors = [self._cache[idx] for idx in indices]
        return torch.cat(tensors, dim=1)

    def compute_num_blocks(self, seq_len: int) -> int:
        """计算给定序列长度会产生多少个 block。
        
        Args:
            seq_len: 序列长度。
        
        Returns:
            block 数量。
        """
        return (seq_len + self.block_size - 1) // self.block_size

    def __repr__(self) -> str:
        return (
            f"BlockBuffer(block_size={self.block_size}, "
            f"max_cached={self.max_cached_blocks}, "
            f"cached_blocks={len(self._cache)}, "
            f"total_processed={self._block_count})"
        )
