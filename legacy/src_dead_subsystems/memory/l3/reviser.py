"""L3 修订器: 处理画像条目间的冲突、修正和覆写。

当新生成的 L3 条目与已有条目发生矛盾时, L3Reviser 负责:
- 检测冲突 (同 key 不同 value)
- 决定保留/覆写/合并策略
- 维护条目的时效性和一致性
"""

from __future__ import annotations

import logging
from datetime import datetime
from difflib import SequenceMatcher
from typing import Any

from src.memory.l3.summarizer import L3ProfileEntry
from src.memory.l3.profile_store import L3ProfileStore

logger = logging.getLogger(__name__)


class ConflictRecord:
    """记录一次冲突检测的结果。"""

    def __init__(
        self,
        existing_entry: L3ProfileEntry,
        new_entry: L3ProfileEntry,
        conflict_type: str,
        resolution: str,
    ):
        self.existing_entry = existing_entry
        self.new_entry = new_entry
        self.conflict_type = conflict_type  # "contradiction", "update", "redundant"
        self.resolution = resolution  # "overwrite", "merge", "keep_existing", "keep_new"
        self.timestamp = datetime.now().isoformat()

    def __repr__(self) -> str:
        return (
            f"ConflictRecord(key={self.existing_entry.key}, "
            f"type={self.conflict_type}, resolution={self.resolution})"
        )


class L3Reviser:
    """L3 画像修订器。

    在每次 L3 更新时, 对新生成的 entries 与现有 store 中的 entries 进行冲突检测,
    并根据策略进行修正。

    冲突检测策略:
    - 同 key + 同 category → 检查 value 相似度
      - 高相似度 (> merge_threshold) → redundant → merge
      - 低相似度 (< contradiction_threshold) → contradiction → 按置信度决定
      - 中等 → update → 覆写

    修正策略:
    - overwrite: 新条目替换旧条目
    - merge: 合并证据, 保留较高置信度的 value
    - keep_existing: 忽略新条目
    - keep_new: 删除旧条目, 添加新条目
    """

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self.merge_threshold: float = self.config.get("merge_threshold", 0.8)
        self.contradiction_threshold: float = self.config.get("contradiction_threshold", 0.3)
        # 冲突日志
        self.conflict_log: list[ConflictRecord] = []
        logger.info(
            f"[L3 Reviser] Initialized. merge_thresh={self.merge_threshold}, "
            f"contradiction_thresh={self.contradiction_threshold}"
        )

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        """计算两段文本的相似度 (SequenceMatcher ratio)。"""
        return SequenceMatcher(None, a.lower(), b.lower()).ratio()

    def detect_conflicts(
        self,
        new_entries: list[L3ProfileEntry],
        store: L3ProfileStore,
    ) -> list[ConflictRecord]:
        """检测新条目与已有条目之间的冲突。"""
        conflicts: list[ConflictRecord] = []

        for new_entry in new_entries:
            # 查找同 key + 同 category 的已有条目
            existing = [
                e for e in store.get_by_key(new_entry.key)
                if e.category == new_entry.category
            ]
            if not existing:
                continue

            for existing_entry in existing:
                sim = self._text_similarity(existing_entry.value, new_entry.value)

                if sim >= self.merge_threshold:
                    conflict_type = "redundant"
                    resolution = "merge"
                elif sim <= self.contradiction_threshold:
                    conflict_type = "contradiction"
                    # 按置信度决定: 高置信度胜出
                    if new_entry.confidence >= existing_entry.confidence:
                        resolution = "overwrite"
                    else:
                        resolution = "keep_existing"
                else:
                    conflict_type = "update"
                    resolution = "overwrite"

                record = ConflictRecord(
                    existing_entry=existing_entry,
                    new_entry=new_entry,
                    conflict_type=conflict_type,
                    resolution=resolution,
                )
                conflicts.append(record)

        return conflicts

    def apply_revisions(
        self,
        new_entries: list[L3ProfileEntry],
        store: L3ProfileStore,
    ) -> list[L3ProfileEntry]:
        """执行完整的修订流程: 检测冲突 → 应用策略 → 返回最终需要添加的条目。

        Args:
            new_entries: 新生成的 L3 条目.
            store: 当前的 L3ProfileStore.

        Returns:
            最终需要添加/更新到 store 中的条目列表.
        """
        conflicts = self.detect_conflicts(new_entries, store)
        self.conflict_log.extend(conflicts)

        # 构建冲突映射: new_entry_id → [ConflictRecord]
        conflict_map: dict[str, list[ConflictRecord]] = {}
        for c in conflicts:
            cid = c.new_entry.entry_id
            conflict_map.setdefault(cid, []).append(c)

        final_entries: list[L3ProfileEntry] = []

        for new_entry in new_entries:
            records = conflict_map.get(new_entry.entry_id, [])

            if not records:
                # 无冲突, 直接添加
                final_entries.append(new_entry)
                continue

            # 按最严重的冲突类型决策
            has_contradiction = any(r.conflict_type == "contradiction" for r in records)
            has_redundant = any(r.conflict_type == "redundant" for r in records)

            if has_redundant and not has_contradiction:
                # 冗余 → 合并到已有条目
                for r in records:
                    if r.resolution == "merge":
                        store.merge_entry(new_entry)
                logger.debug(f"[L3 Reviser] Merged redundant entry: {new_entry.key}")
            elif has_contradiction:
                # 矛盾 → 看 resolution
                for r in records:
                    if r.resolution == "overwrite":
                        # 删除旧条目, 添加新条目
                        store.remove(r.existing_entry.entry_id)
                        final_entries.append(new_entry)
                        logger.info(
                            f"[L3 Reviser] Overwrite contradiction: "
                            f"{r.existing_entry.entry_id} → {new_entry.entry_id}"
                        )
                        break
                    elif r.resolution == "keep_existing":
                        logger.debug(
                            f"[L3 Reviser] Kept existing over new: {new_entry.key}"
                        )
                        break
            else:
                # update 类型 → 覆写
                for r in records:
                    if r.resolution == "overwrite":
                        store.remove(r.existing_entry.entry_id)
                final_entries.append(new_entry)
                logger.debug(f"[L3 Reviser] Updated entry: {new_entry.key}")

        logger.info(
            f"[L3 Reviser] Revision complete. "
            f"{len(conflicts)} conflicts detected, "
            f"{len(final_entries)} entries to add/update."
        )
        return final_entries

    def get_conflict_log(self) -> list[ConflictRecord]:
        """返回历史冲突记录。"""
        return list(self.conflict_log)

    def clear_conflict_log(self) -> None:
        """清空冲突日志。"""
        self.conflict_log.clear()
