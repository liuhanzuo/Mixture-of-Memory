"""
评估指标模块。

包含:
  - exact match 精确匹配
  - update accuracy (偏好更新后查询正确率)
  - temporal accuracy (旧值/新值查询正确率)
  - 按任务类型分组统计
"""

from __future__ import annotations

import logging
import re
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence

logger = logging.getLogger(__name__)


class MemoryMetrics:
    """记忆系统评估指标计算器。

    支持:
      - exact_match: 答案完全匹配（不区分大小写）
      - contains_match: 答案包含在生成文本中
      - 按 task_type 分组统计
    """

    def __init__(self, normalize: bool = True) -> None:
        """
        Args:
            normalize: 是否对答案做规范化处理（小写、去空格）。
        """
        self.normalize = normalize
        self.reset()

    def reset(self) -> None:
        """清空所有累积结果。"""
        self._results: List[Dict[str, Any]] = []

    @staticmethod
    def _normalize(text: str) -> str:
        """规范化文本: 小写、去除首尾空格、压缩连续空格。"""
        text = text.lower().strip()
        text = re.sub(r"\s+", " ", text)
        return text

    def add_result(
        self,
        prediction: str,
        answer: str,
        task_type: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """添加一条评估结果。

        Args:
            prediction: 模型生成的文本。
            answer: 正确答案。
            task_type: 任务类型。
            metadata: 可选的额外信息。
        """
        if self.normalize:
            pred_norm = self._normalize(prediction)
            ans_norm = self._normalize(answer)
        else:
            pred_norm = prediction
            ans_norm = answer

        exact = pred_norm == ans_norm
        contains = ans_norm in pred_norm

        self._results.append({
            "prediction": prediction,
            "answer": answer,
            "task_type": task_type,
            "exact_match": exact,
            "contains_match": contains,
            "metadata": metadata or {},
        })

    def add_batch_results(
        self,
        predictions: Sequence[str],
        answers: Sequence[str],
        task_types: Sequence[str],
        metadata_list: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> None:
        """批量添加评估结果。

        Args:
            predictions: 预测文本列表。
            answers: 答案列表。
            task_types: 任务类型列表。
            metadata_list: 可选的 metadata 列表。
        """
        meta_list = metadata_list or [{}] * len(predictions)
        for pred, ans, tt, meta in zip(predictions, answers, task_types, meta_list):
            self.add_result(pred, ans, tt, meta)

    def compute(self) -> Dict[str, Any]:
        """计算所有指标。

        Returns:
            metrics: 包含总体指标和按任务分组指标的字典。
        """
        if not self._results:
            return {"total_samples": 0}

        metrics: Dict[str, Any] = {}

        # ---- 总体指标 ----
        total = len(self._results)
        exact_correct = sum(r["exact_match"] for r in self._results)
        contains_correct = sum(r["contains_match"] for r in self._results)

        metrics["total_samples"] = total
        metrics["exact_match"] = exact_correct / total
        metrics["contains_match"] = contains_correct / total

        # ---- 按任务类型分组 ----
        by_task: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in self._results:
            by_task[r["task_type"]].append(r)

        metrics["per_task"] = {}
        for task_type, results in by_task.items():
            n = len(results)
            em = sum(r["exact_match"] for r in results) / n
            cm = sum(r["contains_match"] for r in results) / n
            metrics["per_task"][task_type] = {
                "num_samples": n,
                "exact_match": em,
                "contains_match": cm,
            }

        # ---- 特殊指标: update accuracy ----
        update_tasks = by_task.get("updated_preference", [])
        if update_tasks:
            metrics["update_accuracy"] = (
                sum(r["exact_match"] for r in update_tasks)
                / len(update_tasks)
            )
        else:
            metrics["update_accuracy"] = 0.0

        # ---- 特殊指标: temporal accuracy ----
        temporal_tasks = by_task.get("past_value_query", [])
        if temporal_tasks:
            metrics["temporal_accuracy"] = (
                sum(r["exact_match"] for r in temporal_tasks)
                / len(temporal_tasks)
            )
        else:
            metrics["temporal_accuracy"] = 0.0

        return metrics

    def log_metrics(self, metrics: Optional[Dict[str, Any]] = None) -> None:
        """格式化打印指标。

        Args:
            metrics: 如果为 None 则自动计算。
        """
        if metrics is None:
            metrics = self.compute()

        logger.info("=" * 60)
        logger.info("评估结果汇总")
        logger.info("=" * 60)
        logger.info(f"  总样本数: {metrics['total_samples']}")
        logger.info(f"  Exact Match: {metrics.get('exact_match', 0):.4f}")
        logger.info(f"  Contains Match: {metrics.get('contains_match', 0):.4f}")
        logger.info(f"  Update Accuracy: {metrics.get('update_accuracy', 0):.4f}")
        logger.info(f"  Temporal Accuracy: {metrics.get('temporal_accuracy', 0):.4f}")

        per_task = metrics.get("per_task", {})
        if per_task:
            logger.info("-" * 40)
            logger.info("按任务类型:")
            for task_type, task_metrics in sorted(per_task.items()):
                logger.info(
                    f"  [{task_type}] "
                    f"n={task_metrics['num_samples']}, "
                    f"EM={task_metrics['exact_match']:.4f}, "
                    f"CM={task_metrics['contains_match']:.4f}"
                )
        logger.info("=" * 60)


def extract_answer_from_generation(text: str) -> str:
    """从模型生成文本中提取 Answer 部分。

    尝试匹配 'Answer: ...' 模式。

    Args:
        text: 模型生成的完整文本。

    Returns:
        answer: 提取出的答案字符串。
    """
    # 尝试匹配 "Answer: xxx" 模式
    patterns = [
        r"Answer:\s*(.+?)(?:\.|$)",
        r"answer:\s*(.+?)(?:\.|$)",
        r"Answer:\s*(\S+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()

    # 如果没有匹配到，返回最后一行
    lines = text.strip().split("\n")
    return lines[-1].strip() if lines else ""
