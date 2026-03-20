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
        r"(?:I prefer|I like|I want|I'd rather|偏好|喜欢|习惯用|倾向于)\s*([^。.！？\n]{2,60})",
        r"(?:please use|always use|don't use|never use|请用|不要用|以后都这样|就是这种)\s*([^。.！？\n]{2,60})",
        r"(?:回答别太长|简洁点|详细一?点|用技术性的方式|结构化|分点说明)([^。.！？\n]{0,50})",
    ]

    # 任务相关的模式
    TASK_PATTERNS = [
        r"(?:help me|please|could you|can you|帮我|请|能否)\s*([^。.！？\n]{2,60})",
        r"(?:I need to|I want to|我需要|我想要|目标是)\s*([^。.！？\n]{2,60})",
    ]

    # 状态相关的模式
    STATE_PATTERNS = [
        r"(?:I am|I'm|currently|right now|我正在|我目前)\s*([^。.！？\n]{2,60})",
        r"(?:working on|researching|studying|在做|在研究|在设计|在开发|在用|在尝试)\s*([^。.！？\n]{2,60})",
    ]

    # 事实声明模式 (中文) — 用于提取 "X是Y" 类的事实
    # 注意: 使用限制性字符类 [^。.！？\n] 代替 (.+?) 避免灾难性回溯
    FACT_PATTERNS = [
        # "X是Y" / "X变成Y" / "X现在变成Y了"
        r"([\u4e00-\u9fffA-Za-z_]{2,20})(?:是|变成了?|改为|更新为|现在(?:变成了?|是))([^。.！？\n]{1,50})",
        # "告诉你一下，X是Y"
        r"(?:告诉你|顺便说|说一下|记一下)[，,]\s*([^。.！？\n]{3,80})",
        # "我的项目叫X" / "我在XX工作"
        r"我(?:的)?(?:项目|工作|城市|编辑器)(?:叫|是|在|用)\s*([^。.！？\n]{1,50})",
        # "更新一下，X现在是Y"
        r"(?:更新一下|纠正一下|修改一下)[，,]\s*([^。.！？\n]{3,80})",
    ]

    # 无效化/过期模式
    INVALIDATION_PATTERNS = [
        r"(?:已经不准|忘掉|不再有效|忘记|清除|过期)([^。.！？\n]{0,50})",
        r"(?:之前说的)([^。.！？\n]{1,50})(?:已经不准|忘掉|不再有效)",
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

        # 从每条用户消息中提取偏好、任务、状态、事实
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
            # 提取事实声明
            objects.extend(
                self._extract_facts(msg.content, msg.turn_id)
            )
            # 提取无效化声明
            objects.extend(
                self._extract_invalidations(msg.content, msg.turn_id)
            )

        logger.debug(f"[L2 Aggregator] Extracted {len(objects)} objects from {len(messages)} messages.")
        return objects

    def _extract_facts(self, text: str, turn_id: str) -> list[L2MemoryObject]:
        """从消息中提取事实声明 (如 'X是Y')。"""
        objects = []
        for pattern in self.FACT_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    match_text = " ".join(m.strip() for m in match if m.strip())
                else:
                    match_text = match.strip()
                if len(match_text) < 3:
                    continue
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, "entity"),
                    object_type="entity",
                    summary_text=match_text,
                    confidence=0.8,
                    source_turn_ids=[turn_id],
                    metadata={"raw_text": text},
                )
                objects.append(obj)
        return objects

    def _extract_invalidations(self, text: str, turn_id: str) -> list[L2MemoryObject]:
        """提取无效化/过期声明。"""
        objects = []
        for pattern in self.INVALIDATION_PATTERNS:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                match_text = match.strip() if isinstance(match, str) else match[0].strip()
                obj = L2MemoryObject(
                    object_id=self._generate_id(match_text, "invalidation"),
                    object_type="state",
                    summary_text=f"[INVALIDATED] {text.strip()}",
                    confidence=0.9,
                    source_turn_ids=[turn_id],
                    metadata={"action": "invalidate", "raw_text": text},
                )
                objects.append(obj)
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
