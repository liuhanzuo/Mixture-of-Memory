"""L2 检索器: 根据当前 query 检索相关的 L2 记忆对象。

支持:
- 文本匹配检索 (基于关键词/相似度)
- 中文关键词/子串匹配 + 反向匹配
- 中文 n-gram 滑窗匹配
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
        - 关键词/子串匹配 (0.55 权重, 最重要)
        - 反向匹配: obj→query (0.15 权重)
        - 文本相似度 (0.1 权重)
        - 置信度 (0.1 权重)
        - 新鲜度 (0.1 权重)
        """
        query_lower = query.lower()
        obj_text_lower = obj.summary_text.lower()

        # 1. 正向: query→obj 关键词匹配
        keyword_score = self._keyword_match_score(query_lower, obj_text_lower)

        # 2. 正向: query→obj 子串匹配
        substring_score = self._substring_match_score(query_lower, obj_text_lower)

        # 3. 反向: obj→query 子串匹配 (obj 的关键实体是否出现在 query 中)
        reverse_score = self._substring_match_score(obj_text_lower, query_lower)

        # 4. 中文 n-gram 匹配 (捕获无空格分词的中文短语)
        ngram_score = self._ngram_match_score(query_lower, obj_text_lower)

        # 5. 文本相似度 (SequenceMatcher)
        text_sim = SequenceMatcher(None, query_lower, obj_text_lower).ratio()

        # 6. 也检查 metadata 中的 raw_text
        raw_text = obj.metadata.get("raw_text", "").lower()
        if raw_text:
            raw_keyword = self._keyword_match_score(query_lower, raw_text)
            raw_substr = self._substring_match_score(query_lower, raw_text)
            raw_ngram = self._ngram_match_score(query_lower, raw_text)
            keyword_score = max(keyword_score, raw_keyword)
            substring_score = max(substring_score, raw_substr)
            ngram_score = max(ngram_score, raw_ngram)
            # 反向: raw_text→query
            reverse_score = max(reverse_score, self._substring_match_score(raw_text, query_lower))

        # 取正向匹配的最大值
        forward_score = max(keyword_score, substring_score, ngram_score)

        # 综合分数 (提高匹配权重, 降低 SequenceMatcher 权重)
        confidence_score = obj.confidence
        freshness = min(obj.last_accessed_turn / 1000.0, 1.0)

        score = (
            0.55 * forward_score
            + 0.15 * reverse_score
            + 0.1 * text_sim
            + 0.1 * confidence_score
            + 0.1 * freshness
        )
        return score

    # 扩展的停用词集合 (类级别共享)
    _STOP_WORDS = {
        # 中文疑问词/功能词
        "什么", "怎么", "哪里", "哪个", "多少", "如何", "为什么", "吗", "呢",
        "是", "的", "了", "在", "我", "你", "他", "她", "它", "吗",
        "现在", "目前", "最近", "请问", "告诉", "一下", "这个", "那个",
        "还是", "就是", "可以", "能", "会", "要", "有", "没有", "不",
        "说", "看", "想", "做", "知道", "记得", "对了", "好的",
        "之前", "以前", "后来", "然后", "所以", "因为", "但是",
        "帮我", "记一下", "一下", "顺便",
        # 英文功能词
        "what", "where", "how", "which", "who", "when", "is", "are",
        "the", "a", "an", "my", "your", "do", "does", "this", "that",
        "was", "were", "been", "being", "have", "has", "had",
        "will", "would", "could", "should", "can", "may", "might",
        "in", "on", "at", "to", "for", "of", "with", "from", "by",
        "it", "its", "he", "she", "they", "we", "you", "me",
    }

    def _keyword_match_score(self, query: str, text: str) -> float:
        """关键词匹配分数。

        提取 query 中有意义的关键词，计算在 text 中的命中率。
        支持精确匹配和模糊包含匹配。
        """
        # 提取关键词: 按标点/空格切分
        words = re.split(r"[\s，。？?！!、：:；;（）()\"']+", query)
        keywords = [w for w in words if w and w not in self._STOP_WORDS and len(w) >= 2]

        if not keywords:
            return 0.0

        # 精确匹配 + 部分包含匹配
        total_score = 0.0
        for kw in keywords:
            if kw in text:
                total_score += 1.0  # 精确匹配
            elif len(kw) >= 3:
                # 部分匹配: 关键词的核心部分 (去掉首尾1字) 是否出现
                core = kw[1:-1] if len(kw) >= 4 else kw[:2]
                if core in text:
                    total_score += 0.5  # 部分匹配给半分

        return min(total_score / len(keywords), 1.0)

    def _substring_match_score(self, source: str, target: str) -> float:
        """子串匹配分数。

        检查 source 的关键实体子串是否出现在 target 中。
        可用于正向 (query→obj) 和反向 (obj→query) 匹配。

        Args:
            source: 提取实体的文本 (如 query).
            target: 检查是否包含实体的文本 (如 obj summary).
        """
        score = 0.0

        # 提取实体级子串:
        #   - 中文连续字符 (2~20 字)
        #   - 英文/数字混合词组 (2~30 字符, 覆盖 "Flash Attention 2" 等)
        entities: list[str] = []

        # 中文实体
        cn_entities = re.findall(r"[\u4e00-\u9fff]{2,20}", source)
        entities.extend(cn_entities)

        # 英文/数字实体 (含空格分隔的复合词, 如 "Google Brain")
        en_entities = re.findall(r"[A-Za-z][A-Za-z0-9]+(?: [A-Za-z0-9]+)*", source)
        entities.extend([e for e in en_entities if len(e) >= 2])

        # 纯数字实体 (如 IP, 日期, 分数)
        num_entities = re.findall(r"\d+(?:[.:/-]\d+)+|\d+\.\d+|\d{2,}", source)
        entities.extend(num_entities)

        if not entities:
            return 0.0

        # 按长度降序排列, 长实体匹配给更多分数
        entities.sort(key=len, reverse=True)

        # 去重
        seen: set[str] = set()
        unique_entities: list[str] = []
        for e in entities:
            e_lower = e.lower()
            if e_lower not in seen:
                seen.add(e_lower)
                unique_entities.append(e)
        entities = unique_entities

        if not entities:
            return 0.0

        # 加权匹配: 长实体匹配权重更高
        total_weight = 0.0
        matched_weight = 0.0
        for entity in entities:
            weight = min(len(entity) / 4.0, 3.0)  # 长实体权重更高, 上限 3.0
            total_weight += weight
            if entity.lower() in target:
                matched_weight += weight

        return min(matched_weight / total_weight, 1.0) if total_weight > 0 else 0.0

    def _ngram_match_score(self, query: str, text: str, n_range: tuple[int, int] = (2, 5)) -> float:
        """中文 n-gram 滑窗匹配分数。

        对中文文本做 character-level n-gram 滑窗,
        检查 query 的 n-gram 子串有多少出现在 text 中。
        这弥补了中文无空格分词的不足。

        Args:
            query: 查询文本.
            text: 待匹配文本.
            n_range: n-gram 范围 (min_n, max_n).
        """
        # 只提取中文字符序列做 n-gram
        cn_query = re.sub(r"[^\u4e00-\u9fff]", "", query)
        cn_text = re.sub(r"[^\u4e00-\u9fff]", "", text)

        if len(cn_query) < n_range[0] or len(cn_text) < n_range[0]:
            return 0.0

        # 生成 query 的 n-gram 集合
        query_ngrams: set[str] = set()
        for n in range(n_range[0], min(n_range[1] + 1, len(cn_query) + 1)):
            for i in range(len(cn_query) - n + 1):
                gram = cn_query[i:i + n]
                # 过滤纯停用词组合
                if gram not in self._STOP_WORDS:
                    query_ngrams.add(gram)

        if not query_ngrams:
            return 0.0

        # 计算命中率
        hits = sum(1 for gram in query_ngrams if gram in cn_text)
        return hits / len(query_ngrams)

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
