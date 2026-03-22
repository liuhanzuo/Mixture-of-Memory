"""L3 画像存储: 管理长期语义/画像记忆条目。

提供增删改查、衰减/归档、按类别/键检索等能力。
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from src.memory.l3.summarizer import L3ProfileEntry

logger = logging.getLogger(__name__)


class L3ProfileStore:
    """L3 长期画像记忆的持久化存储。

    内部使用 dict[entry_id -> L3ProfileEntry] 维护所有条目。
    支持:
    - add / update / merge
    - 按 category / key 检索
    - 置信度衰减与归档
    - 序列化与反序列化
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._store: dict[str, L3ProfileEntry] = {}
        # 衰减参数
        self.decay_rate: float = self.config.get("decay_rate", 0.02)
        self.archive_threshold: float = self.config.get("archive_threshold", 0.1)
        self.max_entries: int = self.config.get("max_entries", 200)
        logger.info(
            f"[L3 ProfileStore] Initialized. max_entries={self.max_entries}, "
            f"decay_rate={self.decay_rate}, archive_threshold={self.archive_threshold}"
        )

    # ---- 基本操作 ----

    def add(self, entry: L3ProfileEntry) -> None:
        """添加一条画像条目。如果 entry_id 已存在则覆盖。"""
        if len(self._store) >= self.max_entries and entry.entry_id not in self._store:
            self._evict_lowest_confidence()
        self._store[entry.entry_id] = entry
        logger.debug(f"[L3 ProfileStore] Added entry: {entry.entry_id} key={entry.key}")

    def update(self, entry_id: str, **kwargs: Any) -> bool:
        """更新已有条目的字段。返回是否成功更新。"""
        if entry_id not in self._store:
            logger.warning(f"[L3 ProfileStore] Entry {entry_id} not found for update.")
            return False
        entry = self._store[entry_id]
        for field_name, value in kwargs.items():
            if hasattr(entry, field_name):
                setattr(entry, field_name, value)
        entry.last_updated_at = datetime.now().isoformat()
        logger.debug(f"[L3 ProfileStore] Updated entry: {entry_id} fields={list(kwargs.keys())}")
        return True

    def get(self, entry_id: str) -> L3ProfileEntry | None:
        """按 ID 获取条目。"""
        return self._store.get(entry_id)

    def get_by_key(self, key: str) -> list[L3ProfileEntry]:
        """按 key 检索所有匹配条目。"""
        return [e for e in self._store.values() if e.key == key]

    def get_by_category(self, category: str) -> list[L3ProfileEntry]:
        """按 category 检索所有匹配条目。"""
        return [e for e in self._store.values() if e.category == category]

    def remove(self, entry_id: str) -> bool:
        """删除一条画像条目。"""
        if entry_id in self._store:
            del self._store[entry_id]
            logger.debug(f"[L3 ProfileStore] Removed entry: {entry_id}")
            return True
        return False

    def search(self, query: str, top_k: int = 10) -> list[L3ProfileEntry]:
        """基于关键词匹配检索与 query 相关的条目。

        评分策略:
        1. query 中的关键词在 entry 的 key/value/category 中出现 → 加分
        2. entry 的 key/value 中的子串在 query 中出现 → 加分
        3. 基础置信度作为兜底排序因子

        Args:
            query: 检索查询文本.
            top_k: 返回最相关的 top-k 条目.

        Returns:
            按相关度排序的条目列表.
        """
        import re

        if not self._store or not query:
            return self.list_all()[:top_k]

        query_lower = query.lower()

        # 提取 query 中的关键词 (去除常见停用词)
        stop_words = {
            "什么", "怎么", "哪里", "哪个", "多少", "如何", "为什么", "吗",
            "是", "的", "了", "在", "我", "你", "他", "她", "它",
            "现在", "目前", "最近", "请问", "告诉", "记得",
            "what", "where", "how", "which", "who", "when", "is", "are",
            "the", "a", "an", "my", "your", "do", "does",
        }
        query_keywords = set()
        for w in re.split(r"[\s，。？?！!、:：]+", query_lower):
            if w and w not in stop_words and len(w) >= 2:
                query_keywords.add(w)

        # 提取 query 中的实体 (连续中文/英文)
        query_entities = set(re.findall(r"[\u4e00-\u9fffA-Za-z_]{2,}", query))

        scored: list[tuple[L3ProfileEntry, float]] = []
        for entry in self._store.values():
            score = 0.0
            entry_text = f"{entry.key} {entry.value} {entry.category}".lower()

            # 关键词匹配: query 关键词出现在 entry 中
            for kw in query_keywords:
                if kw in entry_text:
                    score += 2.0

            # 实体匹配: query 实体出现在 entry 中
            for entity in query_entities:
                if entity.lower() in entry_text:
                    score += 3.0

            # 反向匹配: entry 的 key 出现在 query 中
            if entry.key.lower() in query_lower:
                score += 2.0

            # entry value 中的短语出现在 query 中
            value_words = set(re.findall(r"[\u4e00-\u9fffA-Za-z_]{2,}", entry.value))
            for vw in value_words:
                if vw.lower() in query_lower:
                    score += 1.5

            # 置信度加权 (作为 tie-breaker)
            score += entry.confidence * 0.1

            scored.append((entry, score))

        # 按分数排序，取 top_k
        scored.sort(key=lambda x: x[1], reverse=True)
        return [entry for entry, _ in scored[:top_k]]

    def list_all(self) -> list[L3ProfileEntry]:
        """列出所有条目, 按置信度降序排列。"""
        return sorted(self._store.values(), key=lambda e: e.confidence, reverse=True)

    def size(self) -> int:
        """返回当前存储条目数。"""
        return len(self._store)

    def clear(self) -> None:
        """清空所有条目。"""
        self._store.clear()
        logger.info("[L3 ProfileStore] Cleared all entries.")

    # ---- 衰减与归档 ----

    def decay_all(self) -> list[str]:
        """对所有条目执行一次置信度衰减, 返回被归档(移除)的 entry_id 列表。

        衰减公式: confidence *= (1 - decay_rate)
        低于 archive_threshold 的条目将被移除。
        """
        archived_ids: list[str] = []
        for entry_id, entry in list(self._store.items()):
            entry.confidence *= (1.0 - self.decay_rate)
            if entry.confidence < self.archive_threshold:
                archived_ids.append(entry_id)
                del self._store[entry_id]

        if archived_ids:
            logger.info(
                f"[L3 ProfileStore] Decayed all entries. Archived {len(archived_ids)} "
                f"entries below threshold {self.archive_threshold}."
            )
        return archived_ids

    def _evict_lowest_confidence(self) -> None:
        """淘汰置信度最低的条目以腾出空间。"""
        if not self._store:
            return
        worst_id = min(self._store, key=lambda eid: self._store[eid].confidence)
        del self._store[worst_id]
        logger.debug(f"[L3 ProfileStore] Evicted lowest-confidence entry: {worst_id}")

    # ---- 批量操作 ----

    def add_batch(self, entries: list[L3ProfileEntry]) -> None:
        """批量添加条目。"""
        for entry in entries:
            self.add(entry)

    def merge_entry(self, new_entry: L3ProfileEntry) -> None:
        """智能合并: 如果存在相同 key + category 的条目, 则更新; 否则添加。

        合并策略:
        - 如果新条目置信度更高, 替换 value
        - 合并 evidence_ids
        - 取最高置信度
        """
        existing = [
            e for e in self._store.values()
            if e.key == new_entry.key and e.category == new_entry.category
        ]
        if not existing:
            self.add(new_entry)
            return

        target = existing[0]
        # 合并证据
        all_evidence = list(set(target.evidence_ids + new_entry.evidence_ids))
        # 取较高置信度
        new_confidence = max(target.confidence, new_entry.confidence)
        # 如果新条目置信度更高, 替换 value
        new_value = new_entry.value if new_entry.confidence >= target.confidence else target.value

        self.update(
            target.entry_id,
            value=new_value,
            confidence=new_confidence,
            evidence_ids=all_evidence,
        )
        logger.debug(
            f"[L3 ProfileStore] Merged entry key={new_entry.key} "
            f"into existing {target.entry_id}"
        )

    # ---- 序列化 ----

    def to_list_of_dicts(self) -> list[dict[str, Any]]:
        """将所有条目序列化为字典列表。"""
        return [e.to_dict() for e in self.list_all()]

    def load_from_dicts(self, data: list[dict[str, Any]]) -> None:
        """从字典列表恢复条目。"""
        for d in data:
            entry = L3ProfileEntry(
                entry_id=d["entry_id"],
                key=d["key"],
                value=d["value"],
                confidence=d.get("confidence", 1.0),
                evidence_ids=d.get("evidence_ids", []),
                created_at=d.get("created_at", datetime.now().isoformat()),
                last_updated_at=d.get("last_updated_at", datetime.now().isoformat()),
                category=d.get("category", "factual"),
            )
            self._store[entry.entry_id] = entry
        logger.info(f"[L3 ProfileStore] Loaded {len(data)} entries from dicts.")

    # ---- 文件级序列化 ----

    def save(self, path: str | Path) -> None:
        """保存所有条目到 JSON 文件。"""
        import json
        from pathlib import Path as _Path
        path = _Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list_of_dicts(), f, ensure_ascii=False, indent=2)
        logger.info(f"[L3 ProfileStore] Saved {len(self._store)} entries to {path}")

    def load(self, path: str | Path) -> None:
        """从 JSON 文件加载条目。"""
        import json
        from pathlib import Path as _Path
        path = _Path(path)
        if not path.exists():
            logger.warning(f"[L3 ProfileStore] File not found: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.load_from_dicts(data)

    def __len__(self) -> int:
        """返回当前存储条目数。"""
        return len(self._store)
