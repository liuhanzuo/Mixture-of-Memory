"""L2 聚合器: 从最近的消息/轮次/chunks 中提取记忆对象。

L2 不从 L1 提升，而是直接从原始对话消息中聚合。
支持:
- rule_based: 基于规则的简单聚合 (默认)
- llm: 基于 LLM 的聚合 (TODO stub)
"""

from __future__ import annotations

import hashlib
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from src.memory.l2.types import ChatMessage, L2MemoryObject

logger = logging.getLogger(__name__)


class AggregatorBackend(ABC):
    """聚合器后端的抽象接口。"""

    @abstractmethod
    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """从消息列表中提取 L2 记忆对象。

        Args:
            messages: 需要聚合的消息列表.
            existing_objects: 已有的记忆对象 (用于避免重复).

        Returns:
            提取出的新记忆对象列表.
        """
        ...


class RuleBasedAggregator(AggregatorBackend):
    """基于规则的聚合器。

    使用简单的启发式规则从消息中提取记忆对象:
    - topic: 从消息内容中提取话题关键词
    - preference: 检测偏好表达模式
    - task: 检测任务/需求表达模式
    - entity: 提取命名实体 (简化版)
    - state: 提取当前状态信息
    """

    # 偏好相关的模式
    PREFERENCE_PATTERNS = [
        r"(?:I prefer|I like|I want|I'd rather|偏好|喜欢|习惯用|倾向于)\s+(.+?)(?:\.|。|$)",
        r"(?:please use|always use|don't use|never use|请用|不要用)\s+(.+?)(?:\.|。|$)",
    ]

    # 任务相关的模式
    TASK_PATTERNS = [
        r"(?:help me|please|could you|can you|帮我|请|能否)\s+(.+?)(?:\.|。|$|\?|？)",
        r"(?:I need to|I want to|我需要|我想要)\s+(.+?)(?:\.|。|$)",
    ]

    # 状态相关的模式
    STATE_PATTERNS = [
        r"(?:I am|I'm|currently|right now|我正在|我目前)\s+(.+?)(?:\.|。|$)",
        r"(?:working on|researching|studying|在做|在研究)\s+(.+?)(?:\.|。|$)",
    ]

    def _generate_id(self, text: str, obj_type: str) -> str:
        """生成唯一 ID。"""
        content = f"{obj_type}:{text}:{datetime.now().isoformat()}"
        return hashlib.md5(content.encode()).hexdigest()[:12]

    def _extract_by_patterns(
        self,
        text: str,
        patterns: list[str],
        obj_type: str,
        turn_id: str,
    ) -> list[L2MemoryObject]:
        """使用正则模式提取对象。"""
        objects = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                if len(match_text) < 3:
                    continue
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, obj_type),
                    object_type=obj_type,
                    summary_text=match_text,
                    confidence=0.7,
                    source_turn_ids=[turn_id],
                )
                objects.append(obj)
        return objects

    def _extract_topic(self, messages: list[ChatMessage]) -> list[L2MemoryObject]:
        """从消息中提取话题对象。

        简单策略: 将连续消息的内容合并, 取前 200 字符作为话题摘要。
        """
        if not messages:
            return []

        combined_text = " ".join(m.content for m in messages if m.role == "user")
        if len(combined_text) < 10:
            return []

        summary = combined_text[:200].strip()
        turn_ids = [m.turn_id for m in messages if m.turn_id]

        return [
            L2MemoryObject(
                object_id=self._generate_id(summary, "topic"),
                object_type="topic",
                summary_text=f"Discussion topic: {summary}",
                confidence=0.6,
                source_turn_ids=turn_ids,
            )
        ]

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """基于规则从消息中聚合 L2 对象。"""
        objects: list[L2MemoryObject] = []

        # 提取话题
        objects.extend(self._extract_topic(messages))

        # 从每条用户消息中提取偏好、任务、状态
        for msg in messages:
            if msg.role != "user":
                continue
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.PREFERENCE_PATTERNS, "preference", msg.turn_id
                )
            )
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.TASK_PATTERNS, "task", msg.turn_id
                )
            )
            objects.extend(
                self._extract_by_patterns(
                    msg.content, self.STATE_PATTERNS, "state", msg.turn_id
                )
            )

        logger.debug(f"[L2 Aggregator] Extracted {len(objects)} objects from {len(messages)} messages.")
        return objects


class LLMAggregator(AggregatorBackend):
    """基于 LLM 的聚合器 (TODO stub)。

    TODO: 在未来实现中:
    1. 将消息序列化为 prompt
    2. 调用 LLM 提取结构化记忆对象
    3. 解析 LLM 输出为 L2MemoryObject
    """

    def __init__(self, provider: Any = None):
        """
        Args:
            provider: LLM 提供者接口实例. TODO: 定义 provider interface.
        """
        self.provider = provider
        logger.warning("[L2] LLMAggregator is a stub. Using rule-based fallback.")

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """TODO: LLM-based aggregation."""
        # 暂时退回到规则聚合
        fallback = RuleBasedAggregator()
        return fallback.aggregate(messages, existing_objects)


class L2Aggregator:
    """L2 聚合器的统一入口。

    根据配置选择 rule_based 或 llm 后端。
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        backend_type = config.get("aggregator_backend", "rule_based")

        if backend_type == "rule_based":
            self.backend = RuleBasedAggregator()
        elif backend_type == "llm":
            self.backend = LLMAggregator()
        else:
            raise ValueError(f"Unknown aggregator backend: {backend_type}")

        logger.info(f"[L2 Aggregator] Using backend: {backend_type}")

    def aggregate(
        self,
        messages: list[ChatMessage],
        existing_objects: list[L2MemoryObject] | None = None,
    ) -> list[L2MemoryObject]:
        """聚合消息为 L2 记忆对象。"""
        return self.backend.aggregate(messages, existing_objects)
