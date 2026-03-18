"""
RULER 基准评估脚手架。

v0 中为占位实现，预留接口供后续扩展:
- 支持 RULER 的多种长上下文任务
- 统一的评估接口

后续需要实现:
- RULER 数据集下载与预处理
- 各子任务的评估逻辑
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class RULEREvaluator:
    """RULER 基准评估器（占位实现）。

    RULER 包含多种长上下文能力测试:
    - Needle in a Haystack
    - Multi-hop reasoning
    - Aggregation
    - etc.

    v0 仅提供接口框架。
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self.cfg = cfg
        logger.info("RULER 评估器初始化（v0 占位实现）")

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """运行 RULER 评估。

        Args:
            model: 模型（MemoryTrainer 或 backbone）。
            tokenizer: tokenizer。

        Returns:
            metrics: 评估指标。
        """
        logger.warning("RULER 评估尚未实现，返回占位结果")
        return {
            "status": "not_implemented",
            "message": "RULER 评估将在后续版本实现",
        }

    @staticmethod
    def get_task_list() -> list:
        """获取支持的 RULER 任务列表。"""
        return [
            "needle_in_haystack",
            "multi_hop",
            "aggregation",
            "variable_tracking",
        ]
