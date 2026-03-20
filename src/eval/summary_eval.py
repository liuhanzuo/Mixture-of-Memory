"""
摘要质量评估器 (SummaryEvaluator)。

评估 L3 摘要/画像生成的质量：
- 摘要覆盖率 (summary coverage): 摘要是否覆盖了源材料的关键信息
- 摘要精确度 (summary precision): 摘要中的内容是否都来自源材料
- 画像一致性 (profile consistency): 画像条目之间是否一致、无矛盾

主要用于 profile_task 的评测。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Sequence

from src.eval.metrics import (
    compute_keyword_coverage,
    compute_token_overlap,
    compute_f1,
    aggregate_metrics,
    format_metrics,
)

logger = logging.getLogger(__name__)


@dataclass
class SummaryTestCase:
    """摘要评测用例。

    Attributes:
        case_id: 用例 ID.
        source_texts: 源材料文本列表 (如 L2 对象的 summary_text).
        reference_summary: 参考摘要 (人工撰写的 gold standard).
        expected_keywords: 摘要中应该包含的关键词.
        contradiction_pairs: 已知的矛盾对列表, 格式 [(key_a, key_b), ...].
    """

    case_id: str
    source_texts: list[str] = field(default_factory=list)
    reference_summary: str = ""
    expected_keywords: list[str] = field(default_factory=list)
    contradiction_pairs: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class SummaryEvalResult:
    """摘要评测结果。"""

    case_id: str
    # token-level overlap with reference
    token_precision: float = 0.0
    token_recall: float = 0.0
    token_f1: float = 0.0
    # 关键词覆盖率
    keyword_coverage: float = 0.0
    # 源材料覆盖率 (摘要 tokens ∩ source tokens)
    source_coverage: float = 0.0
    # 矛盾检测数
    contradictions_found: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "token_precision": self.token_precision,
            "token_recall": self.token_recall,
            "token_f1": self.token_f1,
            "keyword_coverage": self.keyword_coverage,
            "source_coverage": self.source_coverage,
            "contradictions_found": self.contradictions_found,
        }


class SummaryEvaluator:
    """摘要质量评估器。

    Usage::

        evaluator = SummaryEvaluator()
        cases = [SummaryTestCase(...), ...]
        summaries = ["generated summary 1", ...]
        report = evaluator.evaluate(cases, summaries)
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}

    def evaluate_single(
        self,
        case: SummaryTestCase,
        generated_summary: str,
    ) -> SummaryEvalResult:
        """评估单条摘要。"""
        result = SummaryEvalResult(case_id=case.case_id)

        # ---- Token-level overlap with reference ---- #
        if case.reference_summary:
            overlap = compute_token_overlap(generated_summary, case.reference_summary)
            result.token_precision = overlap["precision"]
            result.token_recall = overlap["recall"]
            result.token_f1 = overlap["f1"]

        # ---- 关键词覆盖率 ---- #
        if case.expected_keywords:
            result.keyword_coverage = compute_keyword_coverage(
                generated_summary, case.expected_keywords,
            )

        # ---- 源材料覆盖率 ---- #
        if case.source_texts:
            # 将所有源材料合并, 计算摘要对源材料 tokens 的 recall
            combined_source = " ".join(case.source_texts)
            source_overlap = compute_token_overlap(generated_summary, combined_source)
            result.source_coverage = source_overlap["recall"]

        # ---- 矛盾检测 (简单启发式) ---- #
        if case.contradiction_pairs:
            summary_lower = generated_summary.lower()
            for key_a, key_b in case.contradiction_pairs:
                # 如果两个矛盾项都出现在摘要中, 计为一个矛盾
                if key_a.lower() in summary_lower and key_b.lower() in summary_lower:
                    result.contradictions_found += 1

        return result

    def evaluate(
        self,
        cases: Sequence[SummaryTestCase],
        generated_summaries: Sequence[str],
    ) -> dict[str, Any]:
        """批量评估摘要质量。

        Returns:
            {
                "results": [...],
                "metrics": {
                    "avg_token_f1", "avg_keyword_coverage",
                    "avg_source_coverage", "total_contradictions",
                    "consistency_rate"
                },
                "summary": str
            }
        """
        assert len(cases) == len(generated_summaries), (
            f"用例数 ({len(cases)}) 与摘要数 ({len(generated_summaries)}) 不匹配"
        )

        results: list[SummaryEvalResult] = []
        for case, summary in zip(cases, generated_summaries):
            result = self.evaluate_single(case, summary)
            results.append(result)

        # 聚合指标
        n = len(results) if results else 1
        metrics: dict[str, float] = {
            "avg_token_precision": sum(r.token_precision for r in results) / n,
            "avg_token_recall": sum(r.token_recall for r in results) / n,
            "avg_token_f1": sum(r.token_f1 for r in results) / n,
            "avg_keyword_coverage": sum(r.keyword_coverage for r in results) / n,
            "avg_source_coverage": sum(r.source_coverage for r in results) / n,
            "total_contradictions": sum(r.contradictions_found for r in results),
            # 一致性率: 没有矛盾的比例
            "consistency_rate": sum(
                1 for r in results if r.contradictions_found == 0
            ) / n,
        }

        summary_text = (
            f"SummaryEval: {len(results)} cases\n"
            + format_metrics(metrics, prefix="  ")
        )

        logger.info(f"[SummaryEvaluator] {summary_text}")

        return {
            "results": results,
            "metrics": metrics,
            "summary": summary_text,
        }

    def evaluate_profile_entries(
        self,
        entries: Sequence[dict[str, str]],
        expected_keys: Sequence[str] | None = None,
    ) -> dict[str, float]:
        """评估 L3 画像条目集合的整体质量。

        Args:
            entries: 画像条目列表, 每项含 "key", "value", "category" 等.
            expected_keys: 期望存在的画像 key 列表.

        Returns:
            {"key_coverage": float, "avg_value_length": float, "category_diversity": float}
        """
        if not entries:
            return {"key_coverage": 0.0, "avg_value_length": 0.0, "category_diversity": 0.0}

        # key 覆盖率
        entry_keys = {e.get("key", "").lower() for e in entries}
        if expected_keys:
            covered = sum(1 for k in expected_keys if k.lower() in entry_keys)
            key_coverage = covered / len(expected_keys)
        else:
            key_coverage = 1.0 if entries else 0.0

        # 平均 value 长度 (过短的 value 可能质量差)
        values = [e.get("value", "") for e in entries]
        avg_value_length = sum(len(v) for v in values) / len(values) if values else 0.0

        # category 多样性
        categories = {e.get("category", "unknown") for e in entries}
        category_diversity = len(categories) / max(len(entries), 1)

        return {
            "key_coverage": key_coverage,
            "avg_value_length": avg_value_length,
            "category_diversity": category_diversity,
        }
