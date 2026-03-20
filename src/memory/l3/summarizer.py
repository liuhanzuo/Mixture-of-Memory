"""L3 总结器: 从 L2 记忆对象中生成长期语义/画像记忆。

L3 通过总结和抽象 L2 的对象, 产生更高层次的长期记忆, 如:
- "用户最近在研究大语言模型"
- "用户偏好技术性、结构化的解释"
- "用户的长期项目是 agent 记忆系统"
"""

from __future__ import annotations

import hashlib
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from datetime import datetime
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


# --- L3 数据类型 ---

from dataclasses import dataclass, field


@dataclass
class L3ProfileEntry:
    """L3 层的画像/语义记忆条目。

    Attributes:
        entry_id: 唯一标识符.
        key: 记忆的键 (如 "research_interest", "preferred_style").
        value: 记忆的值 (自然语言描述).
        confidence: 置信度 [0, 1].
        evidence_ids: 支撑证据的 L2 对象 ID 列表.
        created_at: 创建时间.
        last_updated_at: 最后更新时间.
        category: 分类 (如 "research_interest", "preference", "identity" 等).
    """

    entry_id: str
    key: str
    value: str
    confidence: float = 1.0
    evidence_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    category: str = "factual"

    def to_dict(self) -> dict[str, Any]:
        """转为可序列化的字典。"""
        return {
            "entry_id": self.entry_id,
            "key": self.key,
            "value": self.value,
            "confidence": self.confidence,
            "evidence_ids": self.evidence_ids,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "category": self.category,
        }


# --- 总结器后端 ---

class SummarizerBackend(ABC):
    """总结器后端的抽象接口。"""

    @abstractmethod
    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """从 L2 对象中生成 L3 profile 条目。"""
        ...


class RuleBasedSummarizer(SummarizerBackend):
    """基于规则的总结器。

    策略:
    - 按 object_type 分组 L2 对象
    - 对每组生成一个摘要性 profile entry
    - 偏好类对象 → preference 类 entry
    - 话题类对象 → research_interest 类 entry
    - 任务类对象 → long_term_project 类 entry
    """

    TYPE_TO_CATEGORY = {
        "topic": "research_interest",
        "preference": "preference",
        "task": "long_term_project",
        "state": "identity",
        "entity": "factual",
        "relation": "factual",
    }

    def _generate_id(self, key: str) -> str:
        content = f"{key}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """基于规则从 L2 对象生成 L3 条目。"""
        if not l2_objects:
            return []

        # 按类型分组
        grouped: dict[str, list[L2MemoryObject]] = defaultdict(list)
        for obj in l2_objects:
            if not obj.is_archived:
                grouped[obj.object_type].append(obj)

        entries: list[L3ProfileEntry] = []

        for obj_type, objs in grouped.items():
            category = self.TYPE_TO_CATEGORY.get(obj_type, "factual")

            # 合并同类对象的摘要文本
            summaries = [obj.summary_text for obj in objs]
            combined = "; ".join(summaries[:10])  # 最多取 10 条

            # 生成描述
            if category == "research_interest":
                value = f"The user has been discussing: {combined}"
                key = "recent_research_topics"
            elif category == "preference":
                value = f"The user prefers: {combined}"
                key = "user_preferences"
            elif category == "long_term_project":
                value = f"The user is working on: {combined}"
                key = "active_tasks"
            elif category == "identity":
                value = f"Current user state: {combined}"
                key = "current_state"
            else:
                value = f"Known facts: {combined}"
                key = f"facts_{obj_type}"

            avg_confidence = sum(o.confidence for o in objs) / len(objs)
            evidence_ids = [o.object_id for o in objs]

            entry = L3ProfileEntry(
                entry_id=self._generate_id(key),
                key=key,
                value=value,
                confidence=avg_confidence,
                evidence_ids=evidence_ids,
                category=category,
            )
            entries.append(entry)

        logger.info(f"[L3 Summarizer] Generated {len(entries)} profile entries from {len(l2_objects)} L2 objects.")
        return entries


class LLMSummarizer(SummarizerBackend):
    """基于 LLM 的总结器 (TODO stub)。

    TODO: 在未来实现中:
    1. 将 L2 对象序列化为 prompt
    2. 调用 LLM 生成高层次语义摘要
    3. 解析为 L3ProfileEntry
    """

    def __init__(self, provider: Any = None):
        self.provider = provider
        logger.warning("[L3] LLMSummarizer is a stub. Using rule-based fallback.")

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        fallback = RuleBasedSummarizer()
        return fallback.summarize(l2_objects, existing_entries)


class L3Summarizer:
    """L3 总结器的统一入口。"""

    def __init__(self, config: dict[str, Any]):
        self.config = config
        backend_type = config.get("summarizer_backend", "rule_based")

        if backend_type == "rule_based":
            self.backend = RuleBasedSummarizer()
        elif backend_type == "llm":
            self.backend = LLMSummarizer()
        else:
            raise ValueError(f"Unknown summarizer backend: {backend_type}")

        logger.info(f"[L3 Summarizer] Using backend: {backend_type}")

    def summarize(
        self,
        l2_objects: list[L2MemoryObject],
        existing_entries: list[L3ProfileEntry] | None = None,
    ) -> list[L3ProfileEntry]:
        """从 L2 对象生成 L3 profile 条目。"""
        return self.backend.summarize(l2_objects, existing_entries)
