"""
Mixture of Memory (MOM) —— 潜在矩阵记忆系统。

核心思想：
- 每个记忆是一个可学习更新的潜在矩阵 M ∈ R^{D_k × D_v}
- 更新规则: M_t = λ_t * M_{t-1} + ρ_t * α_t * (k_t v_t^T)
- 读取规则: r_t = q_t^T M_t
- 多个记忆槽（fast / medium / slow）通过不同的 retention / update 动态区分
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentMatrixMemory(nn.Module):
    """单个潜在矩阵记忆。

    维护一个形状为 [B, D_k, D_v] 的矩阵状态。

    更新规则:
        M_t = λ_t * M_{t-1} + ρ_t * α_t * (k_t ⊗ v_t^T)

    读取规则:
        r_t = q_t^T M_t  (形状 [B, D_v])
    """

    def __init__(
        self,
        key_dim: int,
        value_dim: int,
        init_scale: float = 0.01,
    ) -> None:
        """
        Args:
            key_dim: 记忆矩阵的 key 维度 (D_k)
            value_dim: 记忆矩阵的 value 维度 (D_v)
            init_scale: 初始化缩放因子
        """
        super().__init__()
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.init_scale = init_scale

        # 记忆状态（非参数，不参与梯度但需要持久化）
        # 在 reset() 时初始化为具体的 batch 大小
        self.register_buffer("_state", None, persistent=False)

    @property
    def state(self) -> Optional[torch.Tensor]:
        """当前记忆状态 [B, D_k, D_v]"""
        return self._state

    def reset(self, batch_size: int, device: torch.device = None) -> None:
        """重置记忆状态为近零初始化。

        Args:
            batch_size: 批大小
            device: 目标设备
        """
        if device is None:
            device = next(self.parameters(), torch.tensor(0.0)).device
        self._state = torch.randn(
            batch_size, self.key_dim, self.value_dim,
            device=device, dtype=torch.float32,
        ) * self.init_scale

    def read(self, query: torch.Tensor) -> torch.Tensor:
        """从记忆中读取。

        Args:
            query: [B, D_k] 查询向量

        Returns:
            readout: [B, D_v] 读取结果
        """
        assert self._state is not None, "记忆未初始化，请先调用 reset()"
        # query: [B, D_k] -> [B, 1, D_k]
        # state: [B, D_k, D_v]
        # result: [B, 1, D_v] -> [B, D_v]
        readout = torch.bmm(query.unsqueeze(1), self._state).squeeze(1)
        return readout

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
        lam: torch.Tensor,
    ) -> None:
        """更新记忆状态。

        M_t = λ * M_{t-1} + ρ * α * (k ⊗ v^T)

        Args:
            key: [B, D_k] 写入 key
            value: [B, D_v] 写入 value
            alpha: [B, 1] 写入强度 ∈ [0, 1]
            rho: [B, 1] 路由权重 ∈ [0, 1]
            lam: [B, 1] 保留因子 ∈ [0, 1]
        """
        assert self._state is not None, "记忆未初始化，请先调用 reset()"
        # 外积: [B, D_k, 1] × [B, 1, D_v] -> [B, D_k, D_v]
        outer = torch.bmm(key.unsqueeze(2), value.unsqueeze(1))

        # 扩展标量到矩阵维度
        lam_expand = lam.unsqueeze(-1)       # [B, 1, 1]
        write_scale = (rho * alpha).unsqueeze(-1)  # [B, 1, 1]

        # 就地更新（保持计算图连通以支持梯度回传）
        self._state = lam_expand * self._state + write_scale * outer

    def get_state(self) -> Optional[torch.Tensor]:
        """获取当前记忆状态的快照。

        Returns:
            state: [B, D_k, D_v] 或 None
        """
        if self._state is None:
            return None
        return self._state.detach().clone()

    def set_state(self, state: torch.Tensor) -> None:
        """设置记忆状态（用于恢复快照）。

        Args:
            state: [B, D_k, D_v]
        """
        self._state = state

    def extra_repr(self) -> str:
        return f"key_dim={self.key_dim}, value_dim={self.value_dim}, init_scale={self.init_scale}"


class MixtureOfMemory(nn.Module):
    """Mixture of Memory (MOM) —— 多记忆混合体。

    管理 N 个 LatentMatrixMemory 槽（默认 3：fast / medium / slow），
    每个记忆通过不同的 retention / update 动态来区分。

    提供统一的 reset / read / update / get_state 接口。
    """

    # 默认记忆名称
    DEFAULT_NAMES = ["fast", "medium", "slow"]

    def __init__(
        self,
        num_memories: int = 3,
        key_dim: int = 64,
        value_dim: int = 64,
        init_scale: float = 0.01,
        memory_names: Optional[List[str]] = None,
    ) -> None:
        """
        Args:
            num_memories: 记忆槽数量
            key_dim: 每个记忆的 key 维度
            value_dim: 每个记忆的 value 维度
            init_scale: 初始化缩放因子
            memory_names: 记忆槽名称（调试用）
        """
        super().__init__()
        self.num_memories = num_memories
        self.key_dim = key_dim
        self.value_dim = value_dim

        if memory_names is None:
            if num_memories <= len(self.DEFAULT_NAMES):
                memory_names = self.DEFAULT_NAMES[:num_memories]
            else:
                memory_names = [f"mem_{i}" for i in range(num_memories)]
        assert len(memory_names) == num_memories
        self.memory_names = memory_names

        # 创建 N 个独立的记忆槽
        self.memories = nn.ModuleList([
            LatentMatrixMemory(
                key_dim=key_dim,
                value_dim=value_dim,
                init_scale=init_scale,
            )
            for _ in range(num_memories)
        ])

    def reset(self, batch_size: int, device: torch.device = None) -> None:
        """重置所有记忆状态。

        Args:
            batch_size: 批大小
            device: 目标设备
        """
        for mem in self.memories:
            mem.reset(batch_size, device)

    def read(self, query: torch.Tensor) -> List[torch.Tensor]:
        """从所有记忆中读取。

        Args:
            query: [B, D_k] 查询向量

        Returns:
            readouts: 长度为 num_memories 的列表，每个元素 [B, D_v]
        """
        return [mem.read(query) for mem in self.memories]

    def update(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        alpha: torch.Tensor,
        rho: torch.Tensor,
        lam: torch.Tensor,
    ) -> None:
        """更新所有记忆。

        Args:
            key: [B, D_k] 写入 key
            value: [B, D_v] 写入 value
            alpha: [B, 1] 写入强度
            rho: [B, num_memories] 路由分布
            lam: [B, num_memories] 每个记忆的保留因子
        """
        for i, mem in enumerate(self.memories):
            rho_i = rho[:, i:i+1]  # [B, 1]
            lam_i = lam[:, i:i+1]  # [B, 1]
            mem.update(key, value, alpha, rho_i, lam_i)

    def get_state(self) -> Dict[str, Optional[torch.Tensor]]:
        """获取所有记忆的状态快照。

        Returns:
            states: {name: state_tensor} 字典
        """
        return {
            name: mem.get_state()
            for name, mem in zip(self.memory_names, self.memories)
        }

    def set_state(self, states: Dict[str, torch.Tensor]) -> None:
        """从快照恢复所有记忆状态。

        Args:
            states: {name: state_tensor} 字典
        """
        for name, mem in zip(self.memory_names, self.memories):
            if name in states and states[name] is not None:
                mem.set_state(states[name])

    def get_memory_stats(self) -> Dict[str, Dict[str, float]]:
        """获取记忆统计信息（用于日志）。

        Returns:
            stats: 每个记忆的 Frobenius 范数等统计量
        """
        stats = {}
        for name, mem in zip(self.memory_names, self.memories):
            if mem.state is not None:
                state = mem.state.detach()
                stats[name] = {
                    "frobenius_norm": state.norm(dim=(-2, -1)).mean().item(),
                    "max_abs": state.abs().max().item(),
                    "mean_abs": state.abs().mean().item(),
                }
            else:
                stats[name] = {"frobenius_norm": 0.0, "max_abs": 0.0, "mean_abs": 0.0}
        return stats

    def extra_repr(self) -> str:
        return (
            f"num_memories={self.num_memories}, "
            f"key_dim={self.key_dim}, value_dim={self.value_dim}, "
            f"names={self.memory_names}"
        )
