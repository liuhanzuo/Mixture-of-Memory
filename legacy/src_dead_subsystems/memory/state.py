"""
MoMState: 记忆系统的全局运行时状态容器。

持有 L1 / L2 / L3 各层实例的引用，提供:
- 统一的状态快照 (snapshot)
- 全局重置 (reset)
- 运行时统计 (stats)
- 序列化 / 反序列化 (save / load)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from src.memory.l1.assoc_memory import AssociativeMemoryL1
from src.memory.l2.object_store import L2ObjectStore
from src.memory.l3.profile_store import L3ProfileStore

logger = logging.getLogger(__name__)


@dataclass
class MoMStats:
    """运行时统计数据，用于 cost_eval 等评估。"""

    l1_write_count: int = 0
    l1_read_count: int = 0
    l2_aggregate_count: int = 0
    l2_retrieve_count: int = 0
    l2_merge_count: int = 0
    l3_summarize_count: int = 0
    l3_revise_count: int = 0
    total_turns: int = 0
    total_chunks: int = 0
    total_sessions: int = 0

    def to_dict(self) -> dict[str, int]:
        return {
            "l1_write_count": self.l1_write_count,
            "l1_read_count": self.l1_read_count,
            "l2_aggregate_count": self.l2_aggregate_count,
            "l2_retrieve_count": self.l2_retrieve_count,
            "l2_merge_count": self.l2_merge_count,
            "l3_summarize_count": self.l3_summarize_count,
            "l3_revise_count": self.l3_revise_count,
            "total_turns": self.total_turns,
            "total_chunks": self.total_chunks,
            "total_sessions": self.total_sessions,
        }

    def reset(self) -> None:
        for attr in self.__dataclass_fields__:
            setattr(self, attr, 0)


@dataclass
class MoMState:
    """
    Mixture-of-Memory 系统的全局运行时状态。

    包含 L1/L2/L3 各层的实例引用和运行时统计。
    由 MemoryScheduler 在初始化时构建并管理。
    """

    l1: AssociativeMemoryL1 | None = None
    l2_store: L2ObjectStore | None = None
    l3_store: L3ProfileStore | None = None

    # 运行时统计
    stats: MoMStats = field(default_factory=MoMStats)

    # 当前会话 ID
    session_id: str = ""
    # 当前 turn 序号
    current_turn: int = 0
    # 当前 chunk 序号 (在 turn 内)
    current_chunk: int = 0

    # 累积的消息缓冲区 (用于 L2 chunk/turn 聚合)
    message_buffer: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    #  状态管理
    # ------------------------------------------------------------------ #

    def reset(self) -> None:
        """重置全部状态 (包括 L3)。"""
        if self.l1 is not None:
            self.l1.reset()
        if self.l2_store is not None:
            self.l2_store.clear()
        if self.l3_store is not None:
            self.l3_store.clear()
        self.stats.reset()
        self.session_id = ""
        self.current_turn = 0
        self.current_chunk = 0
        self.message_buffer.clear()
        logger.info("MoMState 已重置。")

    def soft_reset(self) -> None:
        """软重置: 保留 L3 长期记忆，清空 L1/L2 和缓冲区。"""
        if self.l1 is not None:
            self.l1.reset()
        if self.l2_store is not None:
            self.l2_store.clear()
        self.current_turn = 0
        self.current_chunk = 0
        self.message_buffer.clear()
        logger.info("MoMState 软重置完成 (L3 已保留)。")

    # ------------------------------------------------------------------ #
    #  快照
    # ------------------------------------------------------------------ #

    def snapshot(self) -> dict[str, Any]:
        """返回当前状态的可序列化快照。"""
        snap: dict[str, Any] = {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "current_chunk": self.current_chunk,
            "message_buffer_size": len(self.message_buffer),
            "stats": self.stats.to_dict(),
        }

        # L1 快照: 记忆矩阵的 Frobenius 范数作为摘要
        if self.l1 is not None:
            snap["l1_norm"] = float(torch.norm(self.l1.memory).item())
            snap["l1_step"] = self.l1.step_count
        else:
            snap["l1_norm"] = 0.0
            snap["l1_step"] = 0

        # L2 快照: 对象数量
        if self.l2_store is not None:
            snap["l2_object_count"] = len(self.l2_store)
        else:
            snap["l2_object_count"] = 0

        # L3 快照: 条目数量
        if self.l3_store is not None:
            snap["l3_entry_count"] = len(self.l3_store)
        else:
            snap["l3_entry_count"] = 0

        return snap

    # ------------------------------------------------------------------ #
    #  消息缓冲区操作
    # ------------------------------------------------------------------ #

    def push_message(self, message: dict[str, Any]) -> None:
        """向缓冲区追加一条消息。"""
        self.message_buffer.append(message)

    def flush_buffer(self) -> list[dict[str, Any]]:
        """取出并清空缓冲区中的所有消息。"""
        messages = list(self.message_buffer)
        self.message_buffer.clear()
        return messages

    # ------------------------------------------------------------------ #
    #  持久化 (简易版本)
    # ------------------------------------------------------------------ #

    def save(self, directory: str | Path) -> None:
        """将状态保存到目录。"""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # 保存元信息
        meta = {
            "session_id": self.session_id,
            "current_turn": self.current_turn,
            "current_chunk": self.current_chunk,
            "stats": self.stats.to_dict(),
        }
        with open(directory / "meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # 保存 L1 记忆矩阵
        if self.l1 is not None:
            torch.save(self.l1.memory, directory / "l1_matrix.pt")

        # 保存 L2 对象
        if self.l2_store is not None:
            self.l2_store.save(directory / "l2_objects.json")

        # 保存 L3 画像
        if self.l3_store is not None:
            self.l3_store.save(directory / "l3_profiles.json")

        logger.info(f"MoMState 已保存到 {directory}")

    def load(self, directory: str | Path) -> None:
        """从目录恢复状态。"""
        directory = Path(directory)

        # 恢复元信息
        meta_path = directory / "meta.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.session_id = meta.get("session_id", "")
            self.current_turn = meta.get("current_turn", 0)
            self.current_chunk = meta.get("current_chunk", 0)
            stats_dict = meta.get("stats", {})
            for k, v in stats_dict.items():
                if hasattr(self.stats, k):
                    setattr(self.stats, k, v)

        # 恢复 L1
        l1_path = directory / "l1_matrix.pt"
        if l1_path.exists() and self.l1 is not None:
            self.l1._memory = torch.load(l1_path, weights_only=True)

        # 恢复 L2
        l2_path = directory / "l2_objects.json"
        if l2_path.exists() and self.l2_store is not None:
            self.l2_store.load(l2_path)

        # 恢复 L3
        l3_path = directory / "l3_profiles.json"
        if l3_path.exists() and self.l3_store is not None:
            self.l3_store.load(l3_path)

        logger.info(f"MoMState 已从 {directory} 恢复")

    # ------------------------------------------------------------------ #
    #  便捷属性
    # ------------------------------------------------------------------ #

    @property
    def has_l1(self) -> bool:
        return self.l1 is not None

    @property
    def has_l2(self) -> bool:
        return self.l2_store is not None

    @property
    def has_l3(self) -> bool:
        return self.l3_store is not None

    def __repr__(self) -> str:
        return (
            f"MoMState(session={self.session_id!r}, turn={self.current_turn}, "
            f"chunk={self.current_chunk}, "
            f"l1={'✓' if self.has_l1 else '✗'}, "
            f"l2={'✓' if self.has_l2 else '✗'}, "
            f"l3={'✓' if self.has_l3 else '✗'})"
        )
