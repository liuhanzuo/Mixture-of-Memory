"""L2 检索器: 根据当前 query 检索相关的 L2 记忆对象。

支持:
- 文本匹配检索 (基于关键词/相似度)
- 向量检索 (基于 latent, TODO)
- 类型过滤
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


class L2Retriever:
    """L2 记忆对象检索器。

    从 L2ObjectStore 的对象列表中检索与当前 query 最相关的对象。

    检索策略:
    1. 文本相似度匹配 (默认, 使用 SequenceMatcher)
    2. 向量相似度匹配 (TODO, 使用 latent 向量)
    3. 类型过滤 + 时间排序
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.top_k: int = config.get("retrieval_top_k", 5)

    def retrieve(
        self,
        query: str,
        objects: list[L2MemoryObject],
        top_k: int | None = None,
        type_filter: str | None = None,
    ) -> list[tuple[L2MemoryObject, float]]:
        """检索与 query 最相关的对象。

        Args:
            query: 查询文本.
            objects: 候选对象列表.
            top_k: 返回的最大对象数 (None 则使用配置值).
            type_filter: 如果指定, 只返回该类型的对象.

        Returns:
            按相关度降序排列的 (对象, 分数) 列表.
        """
        k = top_k or self.top_k
        candidates = objects

        # 类型过滤
        if type_filter:
            candidates = [obj for obj in candidates if obj.object_type == type_filter]

        # 过滤掉归档的
        candidates = [obj for obj in candidates if not obj.is_archived]

        if not candidates:
            return []

        # 计算相关度
        scored = []
        for obj in candidates:
            score = self._compute_relevance(query, obj)
            scored.append((obj, score))

        # 按分数降序排列
        scored.sort(key=lambda x: x[1], reverse=True)

        result = scored[:k]
        logger.debug(
            f"[L2 Retriever] Retrieved {len(result)} objects for query "
            f"(top score: {result[0][1]:.3f})" if result else "[L2 Retriever] No objects retrieved."
        )
        return result

    def _compute_relevance(self, query: str, obj: L2MemoryObject) -> float:
        """计算 query 与对象的相关度分数。

        综合考虑:
        - 文本相似度 (0.7 权重)
        - 置信度 (0.2 权重)
        - 新鲜度 (0.1 权重, 基于 last_accessed_turn)
        """
        # 文本相似度
        text_sim = SequenceMatcher(
            None, query.lower(), obj.summary_text.lower()
        ).ratio()

        # 关键词命中检查 (简单的词重叠)
        query_words = set(query.lower().split())
        obj_words = set(obj.summary_text.lower().split())
        if query_words and obj_words:
            keyword_overlap = len(query_words & obj_words) / max(len(query_words), 1)
        else:
            keyword_overlap = 0.0

        # 综合分数
        text_score = max(text_sim, keyword_overlap)
        confidence_score = obj.confidence
        # 新鲜度: 简单归一化 (假设 turn < 1000)
        freshness = min(obj.last_accessed_turn / 1000.0, 1.0)

        score = 0.7 * text_score + 0.2 * confidence_score + 0.1 * freshness
        return score

    def retrieve_by_type(
        self,
        objects: list[L2MemoryObject],
        object_type: str,
        top_k: int | None = None,
    ) -> list[L2MemoryObject]:
        """按类型检索, 按最后访问时间排序。"""
        k = top_k or self.top_k
        typed = [obj for obj in objects if obj.object_type == object_type and not obj.is_archived]
        typed.sort(key=lambda o: o.last_accessed_turn, reverse=True)
        return typed[:k]

    def format_for_prompt(
        self,
        retrieved: list[tuple[L2MemoryObject, float]],
        max_chars: int = 2000,
    ) -> str:
        """将检索结果格式化为可注入 prompt 的文本。

        Args:
            retrieved: (对象, 分数) 列表.
            max_chars: 最大字符数.

        Returns:
            格式化的文本.
        """
        if not retrieved:
            return ""

        lines = ["[Retrieved Memory Objects]"]
        total_chars = len(lines[0])

        for obj, score in retrieved:
            line = f"- [{obj.object_type}] {obj.summary_text} (conf={obj.confidence:.2f})"
            if total_chars + len(line) > max_chars:
                break
            lines.append(line)
            total_chars += len(line)

        return "\n".join(lines)
