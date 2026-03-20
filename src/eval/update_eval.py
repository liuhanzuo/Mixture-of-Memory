"""
记忆更新评估器 (UpdateEvaluator)。

评估 MoM 系统在记忆更新场景下的表现：
- 覆写准确率 (overwrite accuracy): 新信息是否正确覆盖旧信息
- 过期记忆错误率 (stale-memory error rate): 是否错误使用了过期记忆
- 矛盾处理率 (contradiction handling rate): 矛盾信息是否被正确处理

主要用于 synthetic_update_task 的评测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.eval.metrics import (
    compute_accuracy,
    compute_keyword_coverage,
    aggregate_metrics,
    format_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class UpdateTestCase:
    """单条更新评测用例。

    Attributes:
        case_id: 用例 ID.
        update_type: 更新类型, "overwrite" | "stale" | "contradiction" | "temporary".
        old_value: 旧值 (应被覆盖/过期的).
        new_value: 新值 (应被保留的).
        query: 用于查询记忆的 prompt.
        expected_keywords: 期望回复中包含的关键词.
        forbidden_keywords: 期望回复中不包含的关键词 (过期/矛盾信息).
    """

    case_id: str
    update_type: str
    old_value: str
    new_value: str
    query: str
    expected_keywords: list[str] = field(default_factory=list)
    forbidden_keywords: list[str] = field(default_factory=list)


@dataclass
class UpdateEvalResult:
    """单条评测结果。"""

    case_id: str
    update_type: str
    # 是否正确使用了新值
    has_new_value: bool = False
    # 是否错误使用了旧值
    has_old_value: bool = False
    # 关键词覆盖率
    expected_coverage: float = 0.0
    # 禁止关键词命中率 (越低越好)
    forbidden_hit_rate: float = 0.0
    # 综合正确
    is_correct: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "update_type": self.update_type,
            "has_new_value": self.has_new_value,
            "has_old_value": self.has_old_value,
            "expected_coverage": self.expected_coverage,
            "forbidden_hit_rate": self.forbidden_hit_rate,
            "is_correct": self.is_correct,
        }


class UpdateEvaluator:
    """记忆更新评估器。

    评估 agent 在记忆更新场景下的表现。

    Usage::

        evaluator = UpdateEvaluator()
        cases = [UpdateTestCase(...), ...]
        responses = ["agent reply 1", "agent reply 2", ...]
        report = evaluator.evaluate(cases, responses)
        print(report["summary"])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.case_sensitive: bool = self.config.get("case_sensitive", False)

    def evaluate_single(
        self,
        case: UpdateTestCase,
        response: str,
    ) -> UpdateEvalResult:
        """评估单条用例。

        Args:
            case: 评测用例.
            response: agent 的回复文本.

        Returns:
            UpdateEvalResult.
        """
        result = UpdateEvalResult(
            case_id=case.case_id,
            update_type=case.update_type,
        )

        check_response = response if self.case_sensitive else response.lower()

        # 检查是否包含新值
        new_val = case.new_value if self.case_sensitive else case.new_value.lower()
        result.has_new_value = new_val in check_response

        # 检查是否错误包含旧值
        old_val = case.old_value if self.case_sensitive else case.old_value.lower()
        result.has_old_value = old_val in check_response

        # 期望关键词覆盖率
        result.expected_coverage = compute_keyword_coverage(
            response, case.expected_keywords, case_sensitive=self.case_sensitive,
        )

        # 禁止关键词命中率
        result.forbidden_hit_rate = compute_keyword_coverage(
            response, case.forbidden_keywords, case_sensitive=self.case_sensitive,
        )

        # 综合判定：包含新值 & 不包含旧值 & 无禁止词
        result.is_correct = (
            result.has_new_value
            and not result.has_old_value
            and result.forbidden_hit_rate == 0.0
        )

        return result

    def evaluate(
        self,
        cases: Sequence[UpdateTestCase],
        responses: Sequence[str],
    ) -> dict[str, Any]:
        """批量评估。

        Args:
            cases: 评测用例列表.
            responses: 对应的 agent 回复列表.

        Returns:
            包含详细结果和汇总指标的字典::

                {
                    "results": [UpdateEvalResult, ...],
                    "metrics": {
                        "overwrite_accuracy": float,
                        "stale_error_rate": float,
                        "contradiction_handling_rate": float,
                        "overall_accuracy": float,
                    },
                    "summary": str,
                }
        """
        assert len(cases) == len(responses), (
            f"用例数 ({len(cases)}) 与回复数 ({len(responses)}) 不匹配"
        )

        results: list[UpdateEvalResult] = []
        for case, resp in zip(cases, responses):
            result = self.evaluate_single(case, resp)
            results.append(result)

        # 按类型分组统计
        type_stats: dict[str, dict[str, int]] = {}
        for r in results:
            if r.update_type not in type_stats:
                type_stats[r.update_type] = {"correct": 0, "total": 0}
            type_stats[r.update_type]["total"] += 1
            if r.is_correct:
                type_stats[r.update_type]["correct"] += 1

        # 计算指标
        metrics: dict[str, float] = {}

        # 覆写准确率
        ow = type_stats.get("overwrite", {"correct": 0, "total": 0})
        metrics["overwrite_accuracy"] = compute_accuracy(ow["correct"], ow["total"])

        # 过期记忆错误率 (有旧值的比例)
        stale_results = [r for r in results if r.update_type == "stale"]
        if stale_results:
            stale_errors = sum(1 for r in stale_results if r.has_old_value)
            metrics["stale_error_rate"] = stale_errors / len(stale_results)
        else:
            metrics["stale_error_rate"] = 0.0

        # 矛盾处理率
        ct = type_stats.get("contradiction", {"correct": 0, "total": 0})
        metrics["contradiction_handling_rate"] = compute_accuracy(ct["correct"], ct["total"])

        # 整体准确率
        total_correct = sum(1 for r in results if r.is_correct)
        metrics["overall_accuracy"] = compute_accuracy(total_correct, len(results))

        # 平均期望关键词覆盖率
        metrics["avg_expected_coverage"] = (
            sum(r.expected_coverage for r in results) / len(results) if results else 0.0
        )

        # 平均禁止关键词命中率
        metrics["avg_forbidden_hit_rate"] = (
            sum(r.forbidden_hit_rate for r in results) / len(results) if results else 0.0
        )

        summary = (
            f"UpdateEval: {len(results)} cases | "
            f"Overall accuracy: {metrics['overall_accuracy']:.2%}\n"
            + format_metrics(metrics, prefix="  ")
        )

        logger.info(f"[UpdateEvaluator] {summary}")

        return {
            "results": results,
            "metrics": metrics,
            "summary": summary,
        }
