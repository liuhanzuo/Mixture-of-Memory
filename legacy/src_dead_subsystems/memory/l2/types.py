"""L2 记忆对象的数据类型定义。

L2 是事件/状态级记忆对象，直接从最近的 messages/turns/chunks 聚合而来。
支持的对象类型: topic, state, preference, task, entity, relation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import torch


@dataclass
class L2MemoryObject:
    """L2 层的记忆对象。

    每个对象代表从对话历史中提取的一个结构化记忆单元,
    如: 话题、状态、偏好、任务、实体或关系。

    Attributes:
        object_id: 唯一标识符.
        object_type: 对象类型, 如 "topic", "state", "preference", "task", "entity", "relation".
        summary_text: 对象的文本摘要.
        latent: 对象的向量表示 (用于向量检索). 可选.
        confidence: 置信度 [0, 1].
        source_turn_ids: 产生该对象的对话轮次 ID 列表.
        created_at: 创建时间 (ISO 8601).
        last_updated_at: 最后更新时间 (ISO 8601).
        last_accessed_turn: 最后被访问的轮次号 (用于衰减/归档).
        metadata: 额外元数据 (如 key-value 对).
        is_archived: 是否已归档 (过期).
    """

    object_id: str
    object_type: str
    summary_text: str
    latent: torch.Tensor | None = None
    confidence: float = 1.0
    source_turn_ids: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed_turn: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    is_archived: bool = False

    def touch(self, turn_id: int) -> None:
        """更新最后访问轮次。"""
        self.last_accessed_turn = turn_id
        self.last_updated_at = datetime.now().isoformat()

    def archive(self) -> None:
        """归档该对象。"""
        self.is_archived = True
        self.last_updated_at = datetime.now().isoformat()

    def to_dict(self) -> dict[str, Any]:
        """转换为可序列化的字典。"""
        return {
            "object_id": self.object_id,
            "object_type": self.object_type,
            "summary_text": self.summary_text,
            "confidence": self.confidence,
            "source_turn_ids": self.source_turn_ids,
            "created_at": self.created_at,
            "last_updated_at": self.last_updated_at,
            "last_accessed_turn": self.last_accessed_turn,
            "metadata": self.metadata,
            "is_archived": self.is_archived,
        }


@dataclass
class ChatMessage:
    """对话消息的基本类型。"""

    role: str          # "user" | "assistant" | "system"
    content: str
    turn_id: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)
