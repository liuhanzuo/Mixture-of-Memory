"""
L1 关联矩阵记忆核心实现。

实现衰减关联矩阵记忆:
    M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T

支持:
- 单头 / 多头变体
- 按步 (step) 或按块 (chunk) 更新
- 可配置衰减率 λ 和写入强度 ρ
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .writer import L1Writer
from .reader import L1Reader
from .gating import L1Gate


@dataclass
class L1Config:
    """L1 关联记忆配置。"""

    d_key: int = 64
    d_value: int = 64
    n_heads: int = 4
    decay: float = 0.99
    write_strength: float = 1.0
    write_interval: int = 1  # 每隔多少步执行一次写入
    write_gate_type: str = "sigmoid"  # "sigmoid" | "fixed" | "learned"
    use_output_gate: bool = True
    dtype: str = "float32"

    def torch_dtype(self) -> torch.dtype:
        mapping = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        return mapping.get(self.dtype, torch.float32)


class AssociativeMemoryL1(nn.Module):
    """
    L1: 衰减关联矩阵记忆。

    核心状态: M ∈ R^{n_heads × d_key × d_value}
    更新: M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T
    读取: r_t = q_t^T M_t

    该模块不是条目存储，而是连续在线记忆状态，
    用于补偿 SWA 窗口之外的近期历史信息。
    L1 不向 L2 推送数据。
    """

    def __init__(self, config: L1Config) -> None:
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.d_key = config.d_key
        self.d_value = config.d_value

        # 投影层: 从隐藏状态导出 K, V, Q
        # 这些层将在 backbone hidden_dim -> L1 维度之间映射
        # 具体 hidden_dim 在 set_hidden_dim() 中延迟初始化
        self._hidden_dim: int | None = None
        self.proj_k: nn.Linear | None = None
        self.proj_v: nn.Linear | None = None
        self.proj_q: nn.Linear | None = None

        # 写入器和读取器
        self.writer = L1Writer(config)
        self.reader = L1Reader(config)

        # 输出门控
        self.gate = L1Gate(config) if config.use_output_gate else None

        # 记忆状态矩阵 M: (n_heads, d_key, d_value)
        # 使用 register_buffer 使其跟随设备迁移但不参与梯度
        self.register_buffer(
            "_memory",
            torch.zeros(
                config.n_heads,
                config.d_key,
                config.d_value,
                dtype=config.torch_dtype(),
            ),
            persistent=False,
        )

        # 步数计数器
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # 延迟初始化投影层
    # ------------------------------------------------------------------
    def set_hidden_dim(self, hidden_dim: int) -> None:
        """设置 backbone 隐藏维度，初始化投影层。"""
        if self._hidden_dim == hidden_dim:
            return
        self._hidden_dim = hidden_dim
        total_k = self.n_heads * self.d_key
        total_v = self.n_heads * self.d_value
        self.proj_k = nn.Linear(hidden_dim, total_k, bias=False)
        self.proj_v = nn.Linear(hidden_dim, total_v, bias=False)
        self.proj_q = nn.Linear(hidden_dim, total_k, bias=False)

    # ------------------------------------------------------------------
    # 属性
    # ------------------------------------------------------------------
    @property
    def memory(self) -> torch.Tensor:
        """当前记忆状态 (n_heads, d_key, d_value)。"""
        return self._memory

    @property
    def step_count(self) -> int:
        return self._step_count

    # ------------------------------------------------------------------
    # 核心操作
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """重置记忆状态和步数计数器。"""
        self._memory.zero_()
        self._step_count = 0

    def _project_kvq(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        从隐藏状态投影出 K, V, Q。

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            keys:   (batch, seq_len, n_heads, d_key)
            values: (batch, seq_len, n_heads, d_value)
            queries: (batch, seq_len, n_heads, d_key)
        """
        assert self.proj_k is not None, (
            "请先调用 set_hidden_dim() 初始化投影层"
        )
        B, S, _ = hidden_states.shape

        k = self.proj_k(hidden_states).view(B, S, self.n_heads, self.d_key)
        v = self.proj_v(hidden_states).view(B, S, self.n_heads, self.d_value)
        q = self.proj_q(hidden_states).view(B, S, self.n_heads, self.d_key)

        return k, v, q

    def update_step(
        self,
        hidden_states: torch.Tensor,
        write_strengths: torch.Tensor | None = None,
    ) -> None:
        """
        按步更新记忆: M_t = λ M_{t-1} + Σ_i ρ_i k_i v_i^T

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            write_strengths: (batch, seq_len) 可选的逐 token 写入强度
        """
        self._step_count += 1

        # 检查是否到达写入间隔
        if self._step_count % self.config.write_interval != 0:
            return

        keys, values, _ = self._project_kvq(hidden_states)
        self._memory = self.writer.write(
            memory=self._memory,
            keys=keys,
            values=values,
            write_strengths=write_strengths,
        )

    def update_chunk(
        self,
        hidden_states: torch.Tensor,
        write_strengths: torch.Tensor | None = None,
    ) -> None:
        """
        按块更新记忆 (整个 chunk 一次性写入)。

        Args:
            hidden_states: (batch, chunk_len, hidden_dim)
            write_strengths: (batch, chunk_len)
        """
        self._step_count += hidden_states.shape[1]
        keys, values, _ = self._project_kvq(hidden_states)
        self._memory = self.writer.write(
            memory=self._memory,
            keys=keys,
            values=values,
            write_strengths=write_strengths,
        )

    def read(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        从记忆中读取: r_t = q_t^T M_t

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            readout: (batch, seq_len, n_heads * d_value)
        """
        _, _, queries = self._project_kvq(hidden_states)
        return self.reader.read(memory=self._memory, queries=queries)

    def read_and_gate(
        self, hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        读取记忆并通过门控融合到隐藏状态:
            h' = h + g ⊙ W_o r

        Args:
            hidden_states: (batch, seq_len, hidden_dim)

        Returns:
            gated_output: (batch, seq_len, hidden_dim)
        """
        readout = self.read(hidden_states)

        if self.gate is not None:
            return self.gate(hidden_states, readout)
        else:
            return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        update: bool = True,
        write_strengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        完整的 L1 前向传播: 写入 + 读取 + 门控融合。

        Args:
            hidden_states: (batch, seq_len, hidden_dim)
            update: 是否在此步更新记忆
            write_strengths: (batch, seq_len) 可选写入强度

        Returns:
            output: (batch, seq_len, hidden_dim) 融合记忆后的隐藏状态
        """
        # 先读取 (使用更新前的记忆，避免信息泄露)
        output = self.read_and_gate(hidden_states)

        # 再更新
        if update:
            self.update_step(hidden_states, write_strengths)

        return output

    def get_stats(self) -> dict[str, float]:
        """返回当前记忆的统计信息，用于日志和调试。"""
        with torch.no_grad():
            m = self._memory
            return {
                "l1_memory_norm": m.norm().item(),
                "l1_memory_mean": m.mean().item(),
                "l1_memory_std": m.std().item(),
                "l1_memory_max": m.abs().max().item(),
                "l1_step_count": float(self._step_count),
            }
