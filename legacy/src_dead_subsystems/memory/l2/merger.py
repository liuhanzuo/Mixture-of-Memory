"""L2 合并器: 判断新对象是否应与已有对象合并/替换/追加。

使用简单的启发式规则:
- 相同 object_type
- 高文本相似度
- 匹配的 metadata keys
"""

from __future__ import annotations

import logging
from difflib import SequenceMatcher
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


class L2Merger:
    """L2 记忆对象合并器。

    职责:
    - 判断新对象是否应与已有对象合并
    - 提供合并/替换/追加的决策
    - 生成合并后的文本

    策略:
    - same_type: 新旧对象必须是同一类型
    - high_similarity: 文本相似度超过阈值
    - metadata_match: metadata 中的 key 有交集
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.similarity_threshold: float = config.get("merge_similarity_threshold", 0.75)

    def compute_text_similarity(self, text_a: str, text_b: str) -> float:
        """计算两段文本的相似度 (SequenceMatcher)。

        Returns:
            相似度 [0, 1].
        """
        return SequenceMatcher(None, text_a.lower(), text_b.lower()).ratio()

    def find_merge_candidate(
        self,
        new_obj: L2MemoryObject,
        existing_objects: list[L2MemoryObject],
    ) -> tuple[L2MemoryObject | None, str]:
        """为新对象寻找合并候选。

        Args:
            new_obj: 新提取的对象.
            existing_objects: 已有的对象列表.

        Returns:
            (candidate, action):
            - candidate: 最佳合并候选 (或 None)
            - action: "merge" | "replace" | "append"
        """
        best_candidate = None
        best_similarity = 0.0

        for existing in existing_objects:
            # 必须同类型
            if existing.object_type != new_obj.object_type:
                continue
            if existing.is_archived:
                continue

            sim = self.compute_text_similarity(
                new_obj.summary_text, existing.summary_text
            )

            if sim > best_similarity:
                best_similarity = sim
                best_candidate = existing

        if best_candidate is None:
            return None, "append"

        if best_similarity >= self.similarity_threshold:
            # 高相似度: 合并
            return best_candidate, "merge"
        elif best_similarity >= self.similarity_threshold * 0.7:
            # 中等相似度: 如果新对象更新, 替换
            if new_obj.confidence >= best_candidate.confidence:
                return best_candidate, "replace"
            else:
                return None, "append"
        else:
            return None, "append"

    def merge_texts(self, text_a: str, text_b: str) -> str:
        """合并两段文本。

        简单策略: 保留较长的文本, 如果内容差异大则拼接。
        """
        if self.compute_text_similarity(text_a, text_b) > 0.9:
            # 非常相似, 保留较新的 (text_b)
            return text_b

        # 拼接, 去掉重复前缀
        if len(text_b) > len(text_a):
            return text_b
        return f"{text_a} | Updated: {text_b}"

    def decide_and_merge(
        self,
        new_objects: list[L2MemoryObject],
        existing_objects: list[L2MemoryObject],
    ) -> list[tuple[L2MemoryObject, str, L2MemoryObject | None]]:
        """批量决策: 对每个新对象决定 merge/replace/append。

        Returns:
            列表, 每项为 (new_obj, action, merge_target):
            - action: "merge" | "replace" | "append"
            - merge_target: 合并对象 (action 为 merge/replace 时)
        """
        results = []
        for new_obj in new_objects:
            candidate, action = self.find_merge_candidate(new_obj, existing_objects)
            results.append((new_obj, action, candidate))
        return results
