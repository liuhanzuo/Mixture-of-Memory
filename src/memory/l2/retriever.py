"""L2 检索器: 根据当前 query 检索相关的 L2 记忆对象。

支持:
- 文本匹配检索 (基于关键词/相似度)
- 中文关键词/子串匹配
- 向量检索 (基于 latent, TODO)
- 类型过滤
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


class L2Retriever:
    """L2 记忆对象检索器。

    从 L2ObjectStore 的对象列表中检索与当前 query 最相关的对象。

    检索策略:
    1. 中文关键词/子串匹配 (高优先级)
    2. 文本相似度匹配 (SequenceMatcher)
    3. 向量相似度匹配 (TODO, 使用 latent 向量)
    4. 类型过滤 + 时间排序
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
        - 关键词/子串匹配 (0.5 权重, 最重要)
        - 文本相似度 (0.2 权重)
        - 置信度 (0.2 权重)
        - 新鲜度 (0.1 权重)
        """
        query_lower = query.lower()
        obj_text_lower = obj.summary_text.lower()

        # 1. 中文/通用关键词匹配
        keyword_score = self._keyword_match_score(query_lower, obj_text_lower)

        # 2. 子串匹配 (query 中的关键短语是否出现在 obj 中, 反之亦然)
        substring_score = self._substring_match_score(query_lower, obj_text_lower)

        # 3. 文本相似度 (SequenceMatcher)
        text_sim = SequenceMatcher(None, query_lower, obj_text_lower).ratio()

        # 4. 也检查 metadata 中的 raw_text
        raw_text = obj.metadata.get("raw_text", "").lower()
        if raw_text:
            raw_keyword = self._keyword_match_score(query_lower, raw_text)
            raw_substr = self._substring_match_score(query_lower, raw_text)
            keyword_score = max(keyword_score, raw_keyword)
            substring_score = max(substring_score, raw_substr)

        # 取关键词匹配和子串匹配的最大值
        match_score = max(keyword_score, substring_score)

        # 综合分数
        confidence_score = obj.confidence
        freshness = min(obj.last_accessed_turn / 1000.0, 1.0)

        score = 0.5 * match_score + 0.2 * text_sim + 0.2 * confidence_score + 0.1 * freshness
        return score

    def _keyword_match_score(self, query: str, text: str) -> float:
        """关键词匹配分数。

        提取 query 中有意义的关键词，计算在 text 中的命中率。
        """
        # 中文分词: 按标点/空格切分 + 2-gram
        stop_words = {
            "什么", "怎么", "哪里", "哪个", "多少", "如何", "为什么", "吗", "呢",
            "是", "的", "了", "在", "我", "你", "他", "她", "它", "吗",
            "现在", "目前", "最近", "请问", "告诉", "一下",
            "what", "where", "how", "which", "who", "when", "is", "are",
            "the", "a", "an", "my", "your", "do", "does", "this", "that",
        }

        # 提取关键词
        words = re.split(r"[\s，。？?！!、：:；;（）()]+", query)
        keywords = [w for w in words if w and w not in stop_words and len(w) >= 2]

        if not keywords:
            return 0.0

        hits = sum(1 for kw in keywords if kw in text)
        return hits / len(keywords)

    def _substring_match_score(self, query: str, text: str) -> float:
        """子串匹配分数。

        检查 query 的关键部分是否作为子串出现在 text 中, 或者反过来。
        对中文效果较好。
        """
        score = 0.0

        # 提取 query 中的实体级子串 (2~8 字的连续中文或字母数字)
        entities = re.findall(r"[\u4e00-\u9fffA-Za-z0-9]{2,8}", query)

        if not entities:
            return 0.0

        for entity in entities:
            if entity in text:
                score += 1.0 / len(entities)

        return min(score, 1.0)

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
