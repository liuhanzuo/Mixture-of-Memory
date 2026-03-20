"""
骨干模型的抽象接口定义。

定义三层抽象:
1. BackboneModel      — 最基础的前向接口
2. MemoryReadableBackbone — 支持记忆读出注入的扩展接口
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

import torch
from omegaconf import DictConfig

from src.backbone.hidden_state_types import BackboneOutput


class BackboneModel(ABC):
    """所有骨干模型的基类接口。

    子类需实现:
    - ``forward``    : 标准前向传播
    - ``from_config``: 从 OmegaConf 配置构造实例
    """

    # ------------------------------------------------------------------
    # 构造
    # ------------------------------------------------------------------

    @classmethod
    @abstractmethod
    def from_config(cls, cfg: DictConfig) -> "BackboneModel":
        """从配置文件构造骨干模型实例。"""
        ...

    # ------------------------------------------------------------------
    # 前向
    # ------------------------------------------------------------------

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """标准前向传播。

        Args:
            input_ids: 输入 token id, shape ``(B, T)``。
            attention_mask: 注意力掩码, shape ``(B, T)``。
            labels: 目标 token id（用于计算 loss）。
            **kwargs: 额外参数。

        Returns:
            BackboneOutput 实例。
        """
        ...

    # ------------------------------------------------------------------
    # 辅助
    # ------------------------------------------------------------------

    @abstractmethod
    def get_hidden_dim(self) -> int:
        """返回模型隐藏维度 d_model。"""
        ...

    @abstractmethod
    def get_num_layers(self) -> int:
        """返回 Transformer 层数。"""
        ...

    def get_device(self) -> torch.device:
        """返回模型所在设备（默认实现: CPU）。"""
        return torch.device("cpu")


class MemoryReadableBackbone(BackboneModel):
    """支持外部记忆读出注入的骨干模型扩展接口。

    在 ``forward_with_memory`` 中，模型可以接收来自 L1 associative memory
    的 readout 向量，并将其融合进隐藏状态流。

    这是 MoM 系统的关键接口 —— SWA backbone 通过此接口获得超出
    局部注意力窗口的历史信息补偿。
    """

    @abstractmethod
    def forward_with_memory(
        self,
        input_ids: torch.Tensor,
        memory_readout: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> BackboneOutput:
        """带记忆读出的前向传播。

        Args:
            input_ids: 输入 token id, shape ``(B, T)``。
            memory_readout: L1 记忆读出向量, shape ``(B, T, D)`` 或 ``(B, D)``。
            attention_mask: 注意力掩码。
            labels: 目标 token id。
            **kwargs: 额外参数。

        Returns:
            BackboneOutput 实例。
        """
        ...

    def supports_memory_injection(self) -> bool:
        """是否支持记忆注入。默认返回 True。"""
        return True
