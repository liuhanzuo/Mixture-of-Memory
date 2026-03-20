"""L2 记忆模块: 事件/状态级记忆对象。

L2 直接从最近的 messages/turns/chunks 聚合记忆对象,
不从 L1 提升。支持 topic, state, preference, task, entity, relation 等类型。
"""

from src.memory.l2.types import L2MemoryObject, ChatMessage
from src.memory.l2.aggregator import L2Aggregator
from src.memory.l2.object_store import L2ObjectStore
from src.memory.l2.merger import L2Merger
from src.memory.l2.retriever import L2Retriever

__all__ = [
    "L2MemoryObject",
    "ChatMessage",
    "L2Aggregator",
    "L2ObjectStore",
    "L2Merger",
    "L2Retriever",
]
