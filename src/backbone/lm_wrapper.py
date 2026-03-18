"""冻结的因果语言模型包装器。

核心设计：
- 使用 HuggingFace AutoModelForCausalLM 加载任意因果 LM
- 默认冻结所有 backbone 参数
- 暴露 last hidden states 用于下游记忆模块
- 支持 teacher forcing（训练）和 autoregressive generation（推理）
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel

logger = logging.getLogger(__name__)


@dataclass
class LMOutput:
    """LM 前向传播输出容器。"""
    logits: torch.Tensor               # [B, T, V]
    hidden_states: torch.Tensor         # [B, T, D]
    loss: Optional[torch.Tensor] = None # 标量


class FrozenLMWrapper(nn.Module):
    """冻结的 HuggingFace 因果语言模型包装器。

    职责：
    1. 加载并冻结 backbone 模型
    2. 运行前向传播，返回 logits + hidden states
    3. 支持按 block 提取 hidden states
    4. 为后续 LoRA / partial unfreezing 预留扩展点

    Args:
        model_name: HuggingFace 模型名称或路径
        freeze: 是否冻结所有 backbone 参数（默认 True）
        dtype: 模型精度
        device: 目标设备
    """

    def __init__(
        self,
        model_name: str = "gpt2",
        freeze: bool = True,
        dtype: torch.dtype = torch.float32,
        device: Optional[str] = None,
    ) -> None:
        super().__init__()

        logger.info(f"加载 backbone 模型: {model_name}")
        self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            output_hidden_states=True,
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # 确保 tokenizer 有 pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.model.config.eos_token_id

        self.hidden_dim: int = self.model.config.hidden_size
        self.vocab_size: int = self.model.config.vocab_size
        self._frozen = freeze

        if freeze:
            self._freeze_backbone()

        if device is not None:
            self.model = self.model.to(device)

        logger.info(
            f"Backbone 加载完成: hidden_dim={self.hidden_dim}, "
            f"vocab_size={self.vocab_size}, frozen={freeze}"
        )

    def _freeze_backbone(self) -> None:
        """冻结所有 backbone 参数。"""
        frozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = False
            frozen_count += 1
        logger.info(f"已冻结 {frozen_count} 个 backbone 参数")

    def unfreeze_backbone(self) -> None:
        """解冻 backbone（为后续 LoRA / partial unfreezing 预留）。"""
        for param in self.model.parameters():
            param.requires_grad = True
        self._frozen = False
        logger.info("Backbone 已解冻")

    @property
    def lm_head(self) -> nn.Module:
        """获取 lm_head 层，用于 fusion 后重新计算 logits。"""
        # 大多数 HuggingFace 模型的 lm_head 命名为 lm_head
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        # 部分模型（如 GPT-2）使用 tied embeddings
        raise AttributeError(
            f"无法找到 lm_head，请检查模型结构: {type(self.model)}"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> LMOutput:
        """前向传播，返回 logits、hidden states 和可选 loss。

        Args:
            input_ids: [B, T] 输入 token IDs
            attention_mask: [B, T] attention mask
            labels: [B, T] 标签（用于计算 CE loss）

        Returns:
            LMOutput 包含 logits, hidden_states, loss
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
        )

        # 取最后一层 hidden states
        # outputs.hidden_states 是 tuple，最后一个元素是最后一层输出
        last_hidden = outputs.hidden_states[-1]  # [B, T, D]

        return LMOutput(
            logits=outputs.logits,       # [B, T, V]
            hidden_states=last_hidden,   # [B, T, D]
            loss=outputs.loss,           # 标量或 None
        )

    def get_logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """从 hidden states 通过 lm_head 计算 logits。

        这是 fusion 后重新计算 logits 的关键方法。

        Args:
            hidden_states: [B, T, D] 或 [B, 1, D]

        Returns:
            logits: [B, T, V] 或 [B, 1, V]
        """
        return self.lm_head(hidden_states)

    @torch.no_grad()
    def extract_block_hidden_states(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        block_size: int = 64,
    ) -> list[torch.Tensor]:
        """将输入序列按 block 切分，提取每个 block 的 hidden states。

        Args:
            input_ids: [B, T]
            attention_mask: [B, T]
            block_size: block 长度

        Returns:
            list of [B, L, D] tensors，每个 block 一个
        """
        output = self.forward(input_ids, attention_mask)
        hidden = output.hidden_states  # [B, T, D]

        seq_len = hidden.size(1)
        blocks = []
        for start in range(0, seq_len, block_size):
            end = min(start + block_size, seq_len)
            blocks.append(hidden[:, start:end, :])

        return blocks

    def get_config(self) -> dict:
        """返回 backbone 配置摘要。"""
        return {
            "model_name": self.model.config._name_or_path,
            "hidden_dim": self.hidden_dim,
            "vocab_size": self.vocab_size,
            "frozen": self._frozen,
            "num_layers": getattr(self.model.config, "num_hidden_layers", None),
            "num_heads": getattr(self.model.config, "num_attention_heads", None),
        }
