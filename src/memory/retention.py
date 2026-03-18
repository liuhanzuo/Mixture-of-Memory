"""
Retention Scheduler —— 记忆保留率调度策略。

为不同记忆槽提供默认的保留率先验和调度策略：
- fast memory: 低保留（快速遗忘，捕捉近期信息）
- medium memory: 中等保留
- slow memory: 高保留（长期记忆）

这些先验可以被 MemoryWriter 的学习输出覆盖，
但在训练早期提供合理的初始化非常重要。
"""

from __future__ import annotations

from typing import Dict, List, Optional

import torch
import torch.nn as nn


class RetentionScheduler(nn.Module):
    """记忆保留率调度器。

    提供两种模式：
    1. fixed: 使用固定的保留率（不可学习）
    2. learned: 使用可学习的 base retention，训练过程中可以调整

    在训练早期，保留率先验帮助稳定记忆动态。
    """

    # 默认的保留率先验（fast / medium / slow）
    DEFAULT_RETENTIONS = {
        "fast": 0.5,
        "medium": 0.85,
        "slow": 0.95,
    }

    def __init__(
        self,
        num_memories: int = 3,
        mode: str = "learned",
        default_retentions: Optional[List[float]] = None,
        memory_names: Optional[List[str]] = None,
        min_retention: float = 0.0,
        max_retention: float = 1.0,
    ) -> None:
        """
        Args:
            num_memories: 记忆槽数量
            mode: 'fixed' 或 'learned'
            default_retentions: 每个记忆的默认保留率
            memory_names: 记忆名称（用于查找默认值）
            min_retention: 保留率下界
            max_retention: 保留率上界
        """
        super().__init__()
        self.num_memories = num_memories
        self.mode = mode
        self.min_retention = min_retention
        self.max_retention = max_retention

        # 确定初始保留率
        if default_retentions is not None:
            assert len(default_retentions) == num_memories
            init_vals = default_retentions
        elif memory_names is not None:
            init_vals = [
                self.DEFAULT_RETENTIONS.get(name, 0.8)
                for name in memory_names
            ]
        else:
            # 从低到高线性分布
            init_vals = [
                0.5 + 0.45 * i / max(num_memories - 1, 1)
                for i in range(num_memories)
            ]

        # 将保留率转换为 logit 空间: sigmoid(logit) = retention
        import math
        init_logits = []
        for r in init_vals:
            r_clamped = max(min(r, 0.999), 0.001)
            init_logits.append(math.log(r_clamped / (1.0 - r_clamped)))

        if mode == "learned":
            self.retention_logits = nn.Parameter(
                torch.tensor(init_logits, dtype=torch.float32)
            )
        elif mode == "fixed":
            self.register_buffer(
                "retention_logits",
                torch.tensor(init_logits, dtype=torch.float32),
            )
        else:
            raise ValueError(f"未知模式: {mode}，支持 'fixed' 或 'learned'")

    def get_base_retention(self) -> torch.Tensor:
        """获取基础保留率。

        Returns:
            retention: [num_memories] 每个记忆的保留率 ∈ [min_retention, max_retention]
        """
        raw = torch.sigmoid(self.retention_logits)
        # 缩放到 [min_retention, max_retention]
        retention = self.min_retention + (self.max_retention - self.min_retention) * raw
        return retention

    def forward(
        self,
        batch_size: int,
        writer_lam: Optional[torch.Tensor] = None,
        blend_ratio: float = 0.5,
    ) -> torch.Tensor:
        """计算最终的保留率。

        支持将基础保留率与 MemoryWriter 输出的保留率混合。

        Args:
            batch_size: 批大小
            writer_lam: [B, num_memories] 来自 MemoryWriter 的保留率（可选）
            blend_ratio: 混合比例，0.0 = 纯基础，1.0 = 纯 writer

        Returns:
            lam: [B, num_memories] 最终保留率
        """
        base = self.get_base_retention()  # [num_memories]
        base_expanded = base.unsqueeze(0).expand(batch_size, -1)  # [B, num_memories]

        if writer_lam is None or blend_ratio == 0.0:
            return base_expanded

        # 混合基础保留率和 writer 输出
        lam = (1.0 - blend_ratio) * base_expanded + blend_ratio * writer_lam
        return lam

    def get_stats(self) -> Dict[str, float]:
        """获取当前保留率统计（用于日志）。

        Returns:
            stats: 每个记忆的保留率
        """
        with torch.no_grad():
            retention = self.get_base_retention()
            stats = {}
            for i in range(self.num_memories):
                stats[f"base_retention_mem{i}"] = retention[i].item()
        return stats

    def extra_repr(self) -> str:
        with torch.no_grad():
            vals = self.get_base_retention().tolist()
        return (
            f"num_memories={self.num_memories}, mode={self.mode}, "
            f"retentions={[f'{v:.3f}' for v in vals]}"
        )
