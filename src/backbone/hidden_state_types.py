"""
骨干模型输出的统一数据类型定义。

BackboneOutput 是所有骨干模型（SWA / Full Attention）前向传播的统一输出格式，
方便下游 Memory 模块以统一方式消费隐藏状态。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class BackboneOutput:
    """骨干模型前向传播的统一输出。

    Attributes:
        last_hidden_state: 最后一层隐藏状态, shape ``(B, T, D)``。
        logits: 语言模型 head 的 logits, shape ``(B, T, V)``。可选。
        all_hidden_states: 每一层的隐藏状态列表，用于 L1 记忆写入。可选。
        attention_mask: 输入对应的 attention mask, shape ``(B, T)``。可选。
        loss: 语言模型损失（如果提供了 labels）。可选。
        extra: 任意附加信息字典，方便扩展。
    """

    last_hidden_state: torch.Tensor
    logits: Optional[torch.Tensor] = None
    all_hidden_states: Optional[list[torch.Tensor]] = None
    attention_mask: Optional[torch.Tensor] = None
    loss: Optional[torch.Tensor] = None
    extra: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # 便捷属性
    # ------------------------------------------------------------------

    @property
    def batch_size(self) -> int:
        return self.last_hidden_state.shape[0]

    @property
    def seq_len(self) -> int:
        return self.last_hidden_state.shape[1]

    @property
    def hidden_dim(self) -> int:
        return self.last_hidden_state.shape[2]

    @property
    def device(self) -> torch.device:
        return self.last_hidden_state.device

    @property
    def dtype(self) -> torch.dtype:
        return self.last_hidden_state.dtype
