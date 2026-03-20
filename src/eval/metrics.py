"""
通用评估指标计算工具。

提供基础的 precision / recall / F1 / hit@k / recall@k 等计算函数，
供各个专项评估器 (UpdateEvaluator, SummaryEvaluator 等) 复用。
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  基础分类指标
# ------------------------------------------------------------------ #

def compute_precision(tp: int, fp: int) -> float:
    """计算精确率。"""
    if tp + fp == 0:
        return 0.0
    return tp / (tp + fp)


def compute_recall(tp: int, fn: int) -> float:
    """计算召回率。"""
    if tp + fn == 0:
        return 0.0
    return tp / (tp + fn)


def compute_f1(precision: float, recall: float) -> float:
    """计算 F1 分数。"""
    if precision + recall == 0.0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def compute_accuracy(correct: int, total: int) -> float:
    """计算准确率。"""
    if total == 0:
        return 0.0
    return correct / total


# ------------------------------------------------------------------ #
#  检索指标
# ------------------------------------------------------------------ #

def compute_hit_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """计算 Hit@K: 前 k 个检索结果中是否包含至少一个相关项。

    Args:
        retrieved_ids: 检索结果 ID 列表 (按排名排序).
        relevant_ids: 真正相关的 ID 集合.
        k: 截断位置.

    Returns:
        1.0 如果命中, 否则 0.0.
    """
    top_k = retrieved_ids[:k]
    for rid in top_k:
        if rid in relevant_ids:
            return 1.0
    return 0.0


def compute_recall_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """计算 Recall@K: 前 k 个检索结果覆盖了多少相关项。

    Args:
        retrieved_ids: 检索结果 ID 列表 (按排名排序).
        relevant_ids: 真正相关的 ID 集合.
        k: 截断位置.

    Returns:
        Recall@K 值 [0, 1].
    """
    if not relevant_ids:
        return 0.0
    top_k = set(retrieved_ids[:k])
    hits = top_k & relevant_ids
    return len(hits) / len(relevant_ids)


def compute_precision_at_k(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """计算 Precision@K: 前 k 个检索结果中有多少是相关的。

    Args:
        retrieved_ids: 检索结果 ID 列表.
        relevant_ids: 真正相关的 ID 集合.
        k: 截断位置.

    Returns:
        Precision@K 值 [0, 1].
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for rid in top_k if rid in relevant_ids)
    return hits / len(top_k)


def compute_mrr(
    retrieved_ids: Sequence[str],
    relevant_ids: set[str],
) -> float:
    """计算 Mean Reciprocal Rank (MRR)。

    返回第一个相关项的倒数排名。

    Args:
        retrieved_ids: 检索结果 ID 列表 (按排名排序).
        relevant_ids: 真正相关的 ID 集合.

    Returns:
        1/rank 或 0.0 (如果没有命中).
    """
    for i, rid in enumerate(retrieved_ids):
        if rid in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


# ------------------------------------------------------------------ #
#  文本重叠指标
# ------------------------------------------------------------------ #

def compute_token_overlap(
    prediction: str,
    reference: str,
    tokenize_fn: Any | None = None,
) -> dict[str, float]:
    """计算 token 级别的 precision / recall / F1。

    简单实现: 按空白分词。可传入自定义 tokenize_fn。

    Args:
        prediction: 预测文本.
        reference: 参考文本.
        tokenize_fn: 自定义分词函数, 签名: str -> list[str].

    Returns:
        {"precision": float, "recall": float, "f1": float}
    """
    if tokenize_fn is None:
        tokenize_fn = lambda s: s.lower().split()

    pred_tokens = set(tokenize_fn(prediction))
    ref_tokens = set(tokenize_fn(reference))

    if not pred_tokens and not ref_tokens:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0}
    if not pred_tokens or not ref_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    overlap = pred_tokens & ref_tokens
    precision = len(overlap) / len(pred_tokens)
    recall = len(overlap) / len(ref_tokens)
    f1 = compute_f1(precision, recall)

    return {"precision": precision, "recall": recall, "f1": f1}


def compute_keyword_coverage(
    text: str,
    keywords: Sequence[str],
    case_sensitive: bool = False,
) -> float:
    """计算文本对关键词列表的覆盖率。

    Args:
        text: 待检查文本.
        keywords: 关键词列表.
        case_sensitive: 是否区分大小写.

    Returns:
        覆盖率 [0, 1].
    """
    if not keywords:
        return 0.0

    check_text = text if case_sensitive else text.lower()
    hits = 0
    for kw in keywords:
        check_kw = kw if case_sensitive else kw.lower()
        if check_kw in check_text:
            hits += 1

    return hits / len(keywords)


# ------------------------------------------------------------------ #
#  聚合工具
# ------------------------------------------------------------------ #

def aggregate_metrics(
    metric_dicts: Sequence[dict[str, float]],
) -> dict[str, float]:
    """对多个指标字典取平均值。

    Args:
        metric_dicts: 指标字典列表, 每个字典的 key 应一致.

    Returns:
        各 key 的平均值字典.
    """
    if not metric_dicts:
        return {}

    all_keys = set()
    for d in metric_dicts:
        all_keys.update(d.keys())

    result: dict[str, float] = {}
    for key in sorted(all_keys):
        values = [d[key] for d in metric_dicts if key in d]
        result[key] = sum(values) / len(values) if values else 0.0

    return result


def format_metrics(
    metrics: dict[str, float],
    prefix: str = "",
    decimal: int = 4,
) -> str:
    """将指标字典格式化为人类可读的字符串。"""
    lines: list[str] = []
    for k, v in sorted(metrics.items()):
        key_str = f"{prefix}{k}" if prefix else k
        lines.append(f"  {key_str}: {v:.{decimal}f}")
    return "\n".join(lines)
