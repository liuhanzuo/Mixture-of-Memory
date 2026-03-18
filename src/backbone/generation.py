"""记忆增强的自回归生成器。

在推理阶段，结合冻结 backbone 和 MOM 记忆系统进行自回归生成。
生成过程中：
1. backbone 正常产出 hidden states
2. 记忆模块读取 MOM 并 fuse 到 hidden states
3. fused hidden 经过 lm_head 得到 logits
4. 每处理完一个 block 就更新记忆

注意：v0 中不修改 backbone 的 attention / KV cache / position encoding。
"""

from __future__ import annotations

import logging
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MemoryAugmentedGenerator:
    """记忆增强的自回归生成器。

    将 backbone、memory system、fusion head 组合起来，
    支持 greedy / top-k / top-p 等采样策略。

    Args:
        backbone: 冻结的 LM wrapper
        memory_system: 完整的记忆系统（包含 MOM + evaluator + gather + write heads）
        fusion_head: fusion 模块
        block_size: block 大小，用于决定何时触发记忆更新
    """

    def __init__(
        self,
        backbone: nn.Module,
        memory_system: Optional[nn.Module] = None,
        fusion_head: Optional[nn.Module] = None,
        block_size: int = 64,
    ) -> None:
        self.backbone = backbone
        self.memory_system = memory_system
        self.fusion_head = fusion_head
        self.block_size = block_size

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 128,
        temperature: float = 1.0,
        top_k: int = 0,
        top_p: float = 1.0,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """自回归生成。

        Args:
            input_ids: [B, T] 输入 prompt 的 token IDs
            max_new_tokens: 最大生成 token 数
            temperature: 采样温度
            top_k: top-k 采样（0 表示不使用）
            top_p: nucleus 采样（1.0 表示不使用）
            eos_token_id: 停止符 ID

        Returns:
            [B, T + max_new_tokens] 生成的完整序列
        """
        self.backbone.eval()
        if self.memory_system is not None:
            self.memory_system.eval()
        if self.fusion_head is not None:
            self.fusion_head.eval()

        B = input_ids.size(0)
        device = input_ids.device
        generated = input_ids.clone()

        # 初始化记忆（如果有）
        if self.memory_system is not None and hasattr(self.memory_system, "reset"):
            self.memory_system.reset(B)

        # 先处理 prompt 的完整 block，更新记忆
        prompt_len = input_ids.size(1)
        if self.memory_system is not None and prompt_len >= self.block_size:
            self._process_prompt_blocks(input_ids)

        # 逐 token 生成
        for step in range(max_new_tokens):
            # 获取当前序列的 backbone 输出
            # 为了效率，只取最后一个 token 的 hidden state
            # （实际中可以用 KV cache 加速，但 v0 先用简单实现）
            lm_output = self.backbone(generated)
            last_hidden = lm_output.hidden_states[:, -1:, :]  # [B, 1, D]

            # Fusion: 如果有记忆系统和 fusion head
            if self.fusion_head is not None and self.memory_system is not None:
                fused_hidden = self.fusion_head(
                    last_hidden,
                    self.memory_system,
                )
            else:
                fused_hidden = last_hidden

            # 通过 lm_head 得到 logits
            logits = self.backbone.get_logits_from_hidden(fused_hidden)  # [B, 1, V]
            logits = logits[:, -1, :] / temperature  # [B, V]

            # 采样
            next_token = self._sample(logits, top_k=top_k, top_p=top_p)  # [B, 1]
            generated = torch.cat([generated, next_token], dim=1)

            # 检查 EOS
            if eos_token_id is not None:
                if (next_token == eos_token_id).all():
                    break

            # 检查是否需要更新记忆（每完成一个 block）
            current_len = generated.size(1)
            if (
                self.memory_system is not None
                and current_len % self.block_size == 0
            ):
                block_start = current_len - self.block_size
                block_ids = generated[:, block_start:current_len]
                self._update_memory_for_block(block_ids)

        return generated

    def _process_prompt_blocks(self, input_ids: torch.Tensor) -> None:
        """处理 prompt 中的完整 block 并更新记忆。

        Args:
            input_ids: [B, T] prompt token IDs
        """
        T = input_ids.size(1)
        num_full_blocks = T // self.block_size

        if num_full_blocks == 0:
            return

        # 一次性获取 prompt 的 hidden states
        lm_output = self.backbone(input_ids)
        hidden_states = lm_output.hidden_states  # [B, T, D]

        for i in range(num_full_blocks):
            start = i * self.block_size
            end = start + self.block_size
            block_hidden = hidden_states[:, start:end, :]  # [B, L, D]

            if hasattr(self.memory_system, "process_block"):
                self.memory_system.process_block(block_hidden)

    def _update_memory_for_block(self, block_ids: torch.Tensor) -> None:
        """为新完成的 block 更新记忆。

        Args:
            block_ids: [B, L] block 的 token IDs
        """
        lm_output = self.backbone(block_ids)
        block_hidden = lm_output.hidden_states  # [B, L, D]

        if hasattr(self.memory_system, "process_block"):
            self.memory_system.process_block(block_hidden)

    @staticmethod
    def _sample(
        logits: torch.Tensor,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> torch.Tensor:
        """从 logits 中采样下一个 token。

        Args:
            logits: [B, V]
            top_k: top-k 过滤（0 = 不过滤）
            top_p: nucleus 采样阈值（1.0 = 不过滤）

        Returns:
            [B, 1] 采样的 token ID
        """
        # Top-k 过滤
        if top_k > 0:
            top_k = min(top_k, logits.size(-1))
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float("-inf")

        # Top-p (nucleus) 过滤
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(
                F.softmax(sorted_logits, dim=-1), dim=-1
            )
            # 移除累积概率超过 top_p 的 token
            sorted_indices_to_remove = cumulative_probs > top_p
            # 保留第一个超过阈值的 token
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            # 还原到原始顺序
            indices_to_remove = sorted_indices_to_remove.scatter(
                dim=-1, index=sorted_indices, src=sorted_indices_to_remove
            )
            logits[indices_to_remove] = float("-inf")

        # 从概率分布中采样
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # [B, 1]

        return next_token

    @staticmethod
    def greedy_decode(logits: torch.Tensor) -> torch.Tensor:
        """贪心解码。

        Args:
            logits: [B, V]

        Returns:
            [B, 1]
        """
        return logits.argmax(dim=-1, keepdim=True)
