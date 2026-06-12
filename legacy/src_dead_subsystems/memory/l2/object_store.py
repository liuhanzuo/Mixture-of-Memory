"""L2 对象存储: 管理 L2 记忆对象的增删改查和衰减归档。

支持的操作:
- add: 添加新对象
- update: 更新现有对象
- merge: 合并两个对象
- retrieve: 检索相关对象
- decay / archive: 衰减和归档过期对象
"""

from __future__ import annotations

import logging
from typing import Any

from src.memory.l2.types import L2MemoryObject

logger = logging.getLogger(__name__)


class L2ObjectStore:
    """L2 记忆对象存储。

    维护一个内存中的对象列表, 支持基本的 CRUD 操作和衰减归档机制。

    Attributes:
        objects: 活跃的记忆对象列表.
        archived: 已归档的对象列表.
        max_objects: 最大活跃对象数.
        max_age_turns: 最大未访问轮次数, 超过则归档.
    """

    def __init__(self, config: dict[str, Any]):
        self.config = config
        self.max_objects: int = config.get("max_objects", 200)
        self.max_age_turns: int = config.get("max_age_turns", 50)

        self.objects: list[L2MemoryObject] = []
        self.archived: list[L2MemoryObject] = []

        logger.info(
            f"[L2 Store] Initialized: max_objects={self.max_objects}, "
            f"max_age_turns={self.max_age_turns}"
        )

    def add(self, obj: L2MemoryObject) -> None:
        """添加新对象。如果超出容量则归档最旧的对象。"""
        self.objects.append(obj)
        logger.debug(f"[L2 Store] Added object: {obj.object_id} ({obj.object_type})")

        # 容量控制
        while len(self.objects) > self.max_objects:
            oldest = min(self.objects, key=lambda o: o.last_accessed_turn)
            self._archive(oldest)

    def update(self, object_id: str, **kwargs) -> bool:
        """更新指定对象的属性。

        Returns:
            是否找到并更新了对象.
        """
        for obj in self.objects:
            if obj.object_id == object_id:
                for key, value in kwargs.items():
                    if hasattr(obj, key):
                        setattr(obj, key, value)
                from datetime import datetime
                obj.last_updated_at = datetime.now().isoformat()
                logger.debug(f"[L2 Store] Updated object: {object_id}")
                return True
        return False

    def merge(self, obj_id_a: str, obj_id_b: str, merged_text: str) -> L2MemoryObject | None:
        """合并两个对象为一个新对象。

        Args:
            obj_id_a: 第一个对象 ID.
            obj_id_b: 第二个对象 ID.
            merged_text: 合并后的摘要文本.

        Returns:
            合并后的新对象, 或 None (如果找不到原对象).
        """
        obj_a = self.get(obj_id_a)
        obj_b = self.get(obj_id_b)

        if obj_a is None or obj_b is None:
            logger.warning(f"[L2 Store] Cannot merge: missing objects ({obj_id_a}, {obj_id_b})")
            return None

        # 创建合并对象
        from datetime import datetime
        import hashlib

        merged_id = hashlib.md5(f"{obj_id_a}:{obj_id_b}".encode()).hexdigest()[:12]
        merged = L2MemoryObject(
            object_id=merged_id,
            object_type=obj_a.object_type,
            summary_text=merged_text,
            confidence=max(obj_a.confidence, obj_b.confidence),
            source_turn_ids=list(set(obj_a.source_turn_ids + obj_b.source_turn_ids)),
            created_at=obj_a.created_at,
            last_updated_at=datetime.now().isoformat(),
            last_accessed_turn=max(obj_a.last_accessed_turn, obj_b.last_accessed_turn),
            metadata={**obj_a.metadata, **obj_b.metadata},
        )

        # 移除旧对象, 添加新对象
        self.remove(obj_id_a)
        self.remove(obj_id_b)
        self.add(merged)

        logger.debug(f"[L2 Store] Merged {obj_id_a} + {obj_id_b} -> {merged_id}")
        return merged

    def get(self, object_id: str) -> L2MemoryObject | None:
        """根据 ID 获取对象。"""
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def get_by_type(self, object_type: str) -> list[L2MemoryObject]:
        """根据类型获取所有对象。"""
        return [obj for obj in self.objects if obj.object_type == object_type]

    def get_all_active(self) -> list[L2MemoryObject]:
        """获取所有活跃 (未归档) 对象。"""
        return [obj for obj in self.objects if not obj.is_archived]

    def remove(self, object_id: str) -> bool:
        """移除指定对象。"""
        for i, obj in enumerate(self.objects):
            if obj.object_id == object_id:
                self.objects.pop(i)
                return True
        return False

    def _archive(self, obj: L2MemoryObject) -> None:
        """归档一个对象。"""
        obj.archive()
        self.objects.remove(obj)
        self.archived.append(obj)
        logger.debug(f"[L2 Store] Archived object: {obj.object_id}")

    def decay_check(self, current_turn: int) -> list[str]:
        """检查并归档过期对象。

        Args:
            current_turn: 当前轮次号.

        Returns:
            被归档的对象 ID 列表.
        """
        archived_ids = []
        to_archive = [
            obj for obj in self.objects
            if (current_turn - obj.last_accessed_turn) > self.max_age_turns
        ]
        for obj in to_archive:
            self._archive(obj)
            archived_ids.append(obj.object_id)

        if archived_ids:
            logger.info(f"[L2 Store] Decay check: archived {len(archived_ids)} objects.")
        return archived_ids

    @property
    def active_count(self) -> int:
        """活跃对象数量。"""
        return len(self.objects)

    @property
    def archived_count(self) -> int:
        """归档对象数量。"""
        return len(self.archived)

    def to_dict_list(self) -> list[dict]:
        """将所有活跃对象转为字典列表 (用于序列化)。"""
        return [obj.to_dict() for obj in self.objects]

    def clear(self) -> None:
        """清空所有对象 (活跃 + 归档)。"""
        self.objects.clear()
        self.archived.clear()
        logger.info("[L2 Store] Cleared all objects.")

    def __len__(self) -> int:
        """返回活跃对象数量。"""
        return len(self.objects)

    def save(self, path: str | Path) -> None:
        """保存所有活跃对象到 JSON 文件。"""
        import json
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict_list(), f, ensure_ascii=False, indent=2)
        logger.info(f"[L2 Store] Saved {len(self.objects)} objects to {path}")

    def load(self, path: str | Path) -> None:
        """从 JSON 文件加载对象。"""
        import json
        from pathlib import Path
        path = Path(path)
        if not path.exists():
            logger.warning(f"[L2 Store] File not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for d in data:
            obj = L2MemoryObject(
                object_id=d["object_id"],
                object_type=d["object_type"],
                summary_text=d["summary_text"],
                confidence=d.get("confidence", 1.0),
                source_turn_ids=d.get("source_turn_ids", []),
                created_at=d.get("created_at", ""),
                last_updated_at=d.get("last_updated_at", ""),
                last_accessed_turn=d.get("last_accessed_turn", 0),
                metadata=d.get("metadata", {}),
                is_archived=d.get("is_archived", False),
            )
            self.objects.append(obj)
        logger.info(f"[L2 Store] Loaded {len(data)} objects from {path}")
