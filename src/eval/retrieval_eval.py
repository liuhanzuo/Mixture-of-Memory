"""
检索效果评估器 (RetrievalEvaluator)。

评估 L2 / L3 记忆检索的质量：
- Hit@K: 前 k 个检索结果中是否包含至少一个相关项
- Recall@K: 前 k 个检索结果覆盖了多少相关项
- Precision@K: 前 k 个检索结果中有多少是相关的
- MRR: 第一个相关项的倒数排名
- 相关记忆使用率: 检索到的相关记忆中, 实际被 agent 使用的比例

主要用于衡量记忆检索模块对 agent 回复质量的贡献。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.eval.metrics import (
    compute_hit_at_k,
    compute_recall_at_k,
    compute_precision_at_k,
    compute_mrr,
    compute_keyword_coverage,
    aggregate_metrics,
    format_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class RetrievalTestCase:
    """单条检索评测用例。

    Attributes:
        case_id: 用例 ID.
        query: 用于检索的查询文本.
        relevant_ids: 真正相关的记忆对象 ID 集合 (ground truth).
        retrieved_ids: 系统实际检索到的记忆对象 ID 列表 (按排名排序).
        agent_response: agent 的最终回复文本 (用于计算使用率).
        relevant_keywords: 与相关记忆对应的关键词 (用于计算使用率).
    """

    case_id: str
    query: str
    relevant_ids: set[str] = field(default_factory=set)
    retrieved_ids: list[str] = field(default_factory=list)
    agent_response: str = ""
    relevant_keywords: list[str] = field(default_factory=list)


@dataclass
class RetrievalEvalResult:
    """单条检索评测结果。"""

    case_id: str
    hit_at_1: float = 0.0
    hit_at_3: float = 0.0
    hit_at_5: float = 0.0
    recall_at_1: float = 0.0
    recall_at_3: float = 0.0
    recall_at_5: float = 0.0
    precision_at_1: float = 0.0
    precision_at_3: float = 0.0
    precision_at_5: float = 0.0
    mrr: float = 0.0
    # 相关记忆使用率: 检索到的相关记忆被 agent 实际引用的比例
    memory_usage_rate: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "hit@1": self.hit_at_1,
            "hit@3": self.hit_at_3,
            "hit@5": self.hit_at_5,
            "recall@1": self.recall_at_1,
            "recall@3": self.recall_at_3,
            "recall@5": self.recall_at_5,
            "precision@1": self.precision_at_1,
            "precision@3": self.precision_at_3,
            "precision@5": self.precision_at_5,
            "mrr": self.mrr,
            "memory_usage_rate": self.memory_usage_rate,
        }


class RetrievalEvaluator:
    """检索效果评估器。

    评估记忆检索模块 (L2Retriever / L3 检索) 的质量。

    Usage::

        evaluator = RetrievalEvaluator()
        cases = [RetrievalTestCase(...), ...]
        report = evaluator.evaluate(cases)
        print(report["summary"])
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.k_values: list[int] = self.config.get("k_values", [1, 3, 5])

    def evaluate_single(self, case: RetrievalTestCase) -> RetrievalEvalResult:
        """评估单条检索用例。

        Args:
            case: 检索评测用例.

        Returns:
            RetrievalEvalResult.
        """
        result = RetrievalEvalResult(case_id=case.case_id)

        retrieved = case.retrieved_ids
        relevant = case.relevant_ids

        # ---- Hit@K ---- #
        result.hit_at_1 = compute_hit_at_k(retrieved, relevant, k=1)
        result.hit_at_3 = compute_hit_at_k(retrieved, relevant, k=3)
        result.hit_at_5 = compute_hit_at_k(retrieved, relevant, k=5)

        # ---- Recall@K ---- #
        result.recall_at_1 = compute_recall_at_k(retrieved, relevant, k=1)
        result.recall_at_3 = compute_recall_at_k(retrieved, relevant, k=3)
        result.recall_at_5 = compute_recall_at_k(retrieved, relevant, k=5)

        # ---- Precision@K ---- #
        result.precision_at_1 = compute_precision_at_k(retrieved, relevant, k=1)
        result.precision_at_3 = compute_precision_at_k(retrieved, relevant, k=3)
        result.precision_at_5 = compute_precision_at_k(retrieved, relevant, k=5)

        # ---- MRR ---- #
        result.mrr = compute_mrr(retrieved, relevant)

        # ---- 相关记忆使用率 ---- #
        if case.agent_response and case.relevant_keywords:
            result.memory_usage_rate = compute_keyword_coverage(
                case.agent_response, case.relevant_keywords,
            )
        elif case.agent_response and relevant:
            # 如果没有关键词, 检查 retrieved 中相关项的 ID 是否出现在回复中
            # (简化版: 假设如果 hit 了, 就算使用了)
            hit_count = sum(1 for rid in retrieved if rid in relevant)
            result.memory_usage_rate = hit_count / len(relevant) if relevant else 0.0

        return result

    def evaluate(
        self,
        cases: Sequence[RetrievalTestCase],
    ) -> dict[str, Any]:
        """批量评估检索质量。

        Args:
            cases: 检索评测用例列表.

        Returns:
            {
                "results": [...],
                "metrics": {
                    "avg_hit@1", "avg_hit@3", "avg_hit@5",
                    "avg_recall@1", "avg_recall@3", "avg_recall@5",
                    "avg_precision@1", "avg_precision@3", "avg_precision@5",
                    "avg_mrr", "avg_memory_usage_rate",
                },
                "summary": str,
            }
        """
        results: list[RetrievalEvalResult] = []
        for case in cases:
            result = self.evaluate_single(case)
            results.append(result)

        n = len(results) if results else 1

        metrics: dict[str, float] = {
            "avg_hit@1": sum(r.hit_at_1 for r in results) / n,
            "avg_hit@3": sum(r.hit_at_3 for r in results) / n,
            "avg_hit@5": sum(r.hit_at_5 for r in results) / n,
            "avg_recall@1": sum(r.recall_at_1 for r in results) / n,
            "avg_recall@3": sum(r.recall_at_3 for r in results) / n,
            "avg_recall@5": sum(r.recall_at_5 for r in results) / n,
            "avg_precision@1": sum(r.precision_at_1 for r in results) / n,
            "avg_precision@3": sum(r.precision_at_3 for r in results) / n,
            "avg_precision@5": sum(r.precision_at_5 for r in results) / n,
            "avg_mrr": sum(r.mrr for r in results) / n,
            "avg_memory_usage_rate": sum(r.memory_usage_rate for r in results) / n,
        }

        summary = (
            f"RetrievalEval: {len(results)} cases\n"
            + format_metrics(metrics, prefix="  ")
        )

        logger.info(f"[RetrievalEvaluator] {summary}")

        return {
            "results": results,
            "metrics": metrics,
            "summary": summary,
        }

    def evaluate_from_traces(
        self,
        traces: Sequence[dict[str, Any]],
        ground_truth: dict[str, set[str]],
    ) -> dict[str, Any]:
        """从 SessionTrace 和 ground truth 构建用例并评估。

        这是一个便捷方法，用于将 SessionRunner 的输出直接接入评估。

        Args:
            traces: SessionTrace.to_dict() 列表, 每个 trace 包含 turns。
            ground_truth: 映射 turn_id → 相关记忆 ID 集合。

        Returns:
            与 evaluate() 返回格式相同。
        """
        cases: list[RetrievalTestCase] = []

        for trace in traces:
            for turn in trace.get("turns", []):
                turn_id = turn.get("turn_id", "")
                if turn_id not in ground_truth:
                    continue

                # 从 turn 中提取 retrieved IDs
                # (需要 SessionTrace 保存了 retrieved IDs; 否则此处为空)
                retrieved_ids = turn.get("retrieved_ids", [])

                case = RetrievalTestCase(
                    case_id=turn_id,
                    query=turn.get("query", ""),
                    relevant_ids=ground_truth[turn_id],
                    retrieved_ids=retrieved_ids,
                    agent_response=turn.get("response_text", ""),
                )
                cases.append(case)

        if not cases:
            logger.warning("[RetrievalEvaluator] No matching cases found in traces.")
            return {"results": [], "metrics": {}, "summary": "No cases to evaluate."}

        return self.evaluate(cases)
