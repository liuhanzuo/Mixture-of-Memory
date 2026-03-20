"""
L1 门控模块: 控制记忆读取结果与隐藏状态的融合。

核心门控公式:
    h' = h + g ⊙ W_o r

其中 g 为门控信号，W_o 为输出投影，r 为记忆读取结果。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .assoc_memory import L1Config


class L1Gate(nn.Module):
    """
    L1 输出门控。

    将记忆读取结果通过门控机制融合到隐藏状态:
        h' = h + g ⊙ W_o r

    支持多种门控类型:
    - "sigmoid": g = σ(W_g h)，可学习的 sigmoid 门控
    - "fixed":   g = fixed_value，固定常数门控
    - "learned": g 为可学习标量参数
    """

    def __init__(self, config: L1Config) -> None:
        super().__init__()
        self.gate_type = config.write_gate_type
        self.n_heads = config.n_heads
        self.d_value = config.d_value
        self.readout_dim = config.n_heads * config.d_value

        # 输出投影: readout_dim -> hidden_dim
        # hidden_dim 在 set_hidden_dim() 中延迟初始化
        self._hidden_dim: int | None = None
        self.output_proj: nn.Linear | None = None
        self.gate_proj: nn.Linear | None = None

        # 固定门控值 (用于 "fixed" 类型)
        self._fixed_gate_value: float = 0.5

        # 可学习标量门控 (用于 "learned" 类型)
        if self.gate_type == "learned":
            self.learned_gate = nn.Parameter(torch.tensor(0.0))

    def set_hidden_dim(self, hidden_dim: int) -> None:
        """设置隐藏维度并初始化投影层。"""
        if self._hidden_dim == hidden_dim:
            return
        self._hidden_dim = hidden_dim

        # 输出投影: readout_dim -> hidden_dim
        self.output_proj = nn.Linear(self.readout_dim, hidden_dim, bias=False)

        # Sigmoid 门控投影: hidden_dim -> hidden_dim
        if self.gate_type == "sigmoid":
            self.gate_proj = nn.Linear(hidden_dim, hidden_dim, bias=True)

    def _compute_gate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        计算门控信号。

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            gate: (batch, seq_len, hidden_dim) 或标量
        """
        if self.gate_type == "sigmoid":
            assert self.gate_proj is not None, "请先调用 set_hidden_dim()"
            return torch.sigmoid(self.gate_proj(hidden_states))
        elif self.gate_type == "fixed":
            return self._fixed_gate_value
        elif self.gate_type == "learned":
            return torch.sigmoid(self.learned_gate)
        else:
            raise ValueError(f"未知的门控类型: {self.gate_type}")

    def forward(
        self,
        hidden_states: torch.Tensor,
        readout: torch.Tensor,
    ) -> torch.Tensor:
        """
        门控融合: h' = h + g ⊙ W_o r

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            readout:       (batch, seq_len, readout_dim)

        Returns:
            output: (batch, seq_len, hidden_dim)
        """
        # 延迟初始化
        if self._hidden_dim is None:
            self.set_hidden_dim(hidden_states.shape[-1])

        assert self.output_proj is not None

        # 投影读取结果到隐藏维度
        projected = self.output_proj(readout)  # (B, S, hidden_dim)

        # 计算门控
        gate = self._compute_gate(hidden_states)

        # 门控融合
        output = hidden_states + gate * projected

        return output
