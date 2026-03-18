"""
LongMemEval 基准评估脚手架。

v0 中为占位实现，预留接口供后续扩展:
- 支持 LongMemEval 的多种长程记忆任务
- 统一的评估接口

后续需要实现:
- LongMemEval 数据集下载与预处理
- 各子任务的评估逻辑
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class LongMemEvalEvaluator:
    """LongMemEval 基准评估器（占位实现）。

    LongMemEval 测试模型的长程记忆能力:
    - 跨会话信息保留
    - 偏好追踪与更新
    - 时间感知记忆

    v0 仅提供接口框架。
    """

    def __init__(self, cfg: Optional[Any] = None) -> None:
        self.cfg = cfg
        logger.info("LongMemEval 评估器初始化（v0 占位实现）")

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """运行 LongMemEval 评估。

        Args:
            model: 模型。
            tokenizer: tokenizer。

        Returns:
            metrics: 评估指标。
        """
        logger.warning("LongMemEval 评估尚未实现，返回占位结果")
        return {
            "status": "not_implemented",
            "message": "LongMemEval 评估将在后续版本实现",
        }

    @staticmethod
    def get_task_list() -> list:
        """获取支持的 LongMemEval 任务列表。"""
        return [
            "preference_retention",
            "preference_update",
            "temporal_memory",
            "cross_session_recall",
        ]
