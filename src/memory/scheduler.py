"""
MemoryScheduler: 三级记忆的协调调度器。

负责在不同事件粒度上驱动 L1 / L2 / L3 的读写:
- on_token_step:  每个 token/step 级别 → L1 同步在线更新
- on_chunk_end:   chunk 结束 → L2 异步聚合
- on_turn_end:    对话 turn 结束 → L2 聚合 + 合并
- on_session_end: 会话结束 → L3 异步总结 + 修订

"异步"在此处指事件触发的延迟更新, 并非多线程。
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any

import torch

from src.memory.l1.assoc_memory import AssociativeMemoryL1
from src.memory.l1.writer import L1Writer
from src.memory.l1.reader import L1Reader
from src.memory.l1.gating import L1Gate
from src.memory.l2.aggregator import L2Aggregator
from src.memory.l2.object_store import L2ObjectStore
from src.memory.l2.merger import L2Merger
from src.memory.l2.retriever import L2Retriever
from src.memory.l2.types import ChatMessage
from src.memory.l3.summarizer import L3Summarizer
from src.memory.l3.profile_store import L3ProfileStore
from src.memory.l3.reviser import L3Reviser
from src.memory.state import MoMState, MoMStats

logger = logging.getLogger(__name__)


@dataclass
class SchedulerConfig:
    """调度器配置。"""

    # 是否启用各层
    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True

    # L1 配置
    l1_d_key: int = 64
    l1_d_value: int = 64
    l1_num_heads: int = 4
    l1_decay: float = 0.95
    l1_write_strength: float = 1.0

    # L2 配置
    l2_max_objects: int = 200
    l2_similarity_threshold: float = 0.75
    l2_chunk_size: int = 5  # 每 N 条消息触发一次 chunk 聚合

    # L3 配置
    l3_max_entries: int = 100
    l3_conflict_threshold: float = 0.8

    # Gate 配置
    l1_gate_hidden_dim: int = 64
    l1_gate_d_model: int = 64


class MemoryScheduler:
    """
    三级记忆调度器。

    协调 L1 / L2 / L3 在不同事件粒度上的读写操作:
    - L1: 同步在线更新 (每个 token step)
    - L2: 异步事件触发 (chunk/turn 结束)
    - L3: 异步事件触发 (session 结束)

    Usage::

        scheduler = MemoryScheduler(config)
        scheduler.init_state(session_id="sess_001")

        # 每个 token step
        readout = scheduler.on_token_step(hidden_states=h, step=t)

        # turn 结束
        scheduler.on_turn_end(messages=[...])

        # 会话结束
        scheduler.on_session_end()
    """

    def __init__(self, config: SchedulerConfig | dict[str, Any] | None = None):
        if config is None:
            config = SchedulerConfig()
        elif isinstance(config, dict):
            config = SchedulerConfig(**{
                k: v for k, v in config.items()
                if k in SchedulerConfig.__dataclass_fields__
            })
        self.config = config

        # ---- 初始化各层组件 ---- #
        # L1
        self._l1: AssociativeMemoryL1 | None = None
        self._l1_writer: L1Writer | None = None
        self._l1_reader: L1Reader | None = None
        self._l1_gate: L1Gate | None = None

        if config.enable_l1:
            self._l1 = AssociativeMemoryL1(
                d_key=config.l1_d_key,
                d_value=config.l1_d_value,
                num_heads=config.l1_num_heads,
                decay=config.l1_decay,
            )
            self._l1_writer = L1Writer(
                d_model=config.l1_gate_d_model,
                d_key=config.l1_d_key,
                d_value=config.l1_d_value,
            )
            self._l1_reader = L1Reader(
                d_model=config.l1_gate_d_model,
                d_key=config.l1_d_key,
            )
            self._l1_gate = L1Gate(
                d_model=config.l1_gate_d_model,
                d_readout=config.l1_d_value,
            )

        # L2
        self._l2_aggregator: L2Aggregator | None = None
        self._l2_store: L2ObjectStore | None = None
        self._l2_merger: L2Merger | None = None
        self._l2_retriever: L2Retriever | None = None

        if config.enable_l2:
            self._l2_aggregator = L2Aggregator()
            self._l2_store = L2ObjectStore(max_objects=config.l2_max_objects)
            self._l2_merger = L2Merger(
                similarity_threshold=config.l2_similarity_threshold,
            )
            self._l2_retriever = L2Retriever(store=self._l2_store)

        # L3
        self._l3_summarizer: L3Summarizer | None = None
        self._l3_store: L3ProfileStore | None = None
        self._l3_reviser: L3Reviser | None = None

        if config.enable_l3:
            self._l3_summarizer = L3Summarizer()
            self._l3_store = L3ProfileStore(max_entries=config.l3_max_entries)
            self._l3_reviser = L3Reviser(
                conflict_threshold=config.l3_conflict_threshold,
            )

        # 全局状态
        self._state = MoMState(
            l1=self._l1,
            l2_store=self._l2_store,
            l3_store=self._l3_store,
        )

        logger.info(
            f"MemoryScheduler 初始化完成: "
            f"L1={'✓' if config.enable_l1 else '✗'}, "
            f"L2={'✓' if config.enable_l2 else '✗'}, "
            f"L3={'✓' if config.enable_l3 else '✗'}"
        )

    # ------------------------------------------------------------------ #
    #  属性访问
    # ------------------------------------------------------------------ #

    @property
    def state(self) -> MoMState:
        return self._state

    @property
    def stats(self) -> MoMStats:
        return self._state.stats

    # ------------------------------------------------------------------ #
    #  会话管理
    # ------------------------------------------------------------------ #

    def init_state(self, session_id: str | None = None) -> None:
        """初始化或重置会话状态。"""
        self._state.soft_reset()
        self._state.session_id = session_id or str(uuid.uuid4())[:8]
        self._state.stats.total_sessions += 1
        logger.info(f"会话 {self._state.session_id} 已初始化。")

    # ------------------------------------------------------------------ #
    #  事件: on_token_step (L1 同步更新)
    # ------------------------------------------------------------------ #

    def on_token_step(
        self,
        hidden_states: torch.Tensor,
        step: int | None = None,
    ) -> torch.Tensor | None:
        """
        每个 token/step 级别的同步事件。

        1. L1 Writer 从 hidden_states 生成 K, V 并写入记忆矩阵
        2. L1 Reader 从 hidden_states 生成 Q 并读出
        3. L1 Gate 计算门控并产出补偿后的 hidden_states

        Args:
            hidden_states: 当前 token 的隐状态, shape (batch, seq_len, d_model) 或 (seq_len, d_model)
            step: 可选的步数标记

        Returns:
            门控后的 memory readout, 与 hidden_states 同 shape; 或 None (L1 未启用)
        """
        if self._l1 is None or self._l1_writer is None or self._l1_reader is None:
            return None

        # 写入
        keys, values, strengths = self._l1_writer(hidden_states)
        self._l1.write(keys, values, write_strengths=strengths)
        self._state.stats.l1_write_count += 1

        # 读出
        queries = self._l1_reader.compute_queries(hidden_states)
        readout = self._l1.read(queries)
        self._state.stats.l1_read_count += 1

        # 门控
        if self._l1_gate is not None:
            gated = self._l1_gate(hidden_states, readout)
            return gated

        return readout

    # ------------------------------------------------------------------ #
    #  事件: on_chunk_end (L2 异步聚合)
    # ------------------------------------------------------------------ #

    def on_chunk_end(
        self,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        chunk 结束时的异步事件。

        从缓冲区或给定消息中聚合 L2 记忆对象。

        Args:
            messages: 该 chunk 内的消息列表; 若为 None, 从 state 缓冲区取。
        """
        if self._l2_aggregator is None or self._l2_store is None:
            return

        # 获取消息
        if messages is None:
            messages = self._state.flush_buffer()
        if not messages:
            return

        # 转换为 ChatMessage
        chat_msgs = []
        for msg in messages:
            chat_msgs.append(ChatMessage(
                role=msg.get("role", "user"),
                content=msg.get("content", ""),
                turn_id=msg.get("turn_id", ""),
                timestamp=msg.get("timestamp", ""),
            ))

        # 聚合
        new_objects = self._l2_aggregator.aggregate(chat_msgs)
        self._state.stats.l2_aggregate_count += 1

        # 合并到 store
        if self._l2_merger is not None:
            for obj in new_objects:
                merged = self._l2_merger.merge_or_add(obj, self._l2_store)
                if merged:
                    self._state.stats.l2_merge_count += 1
        else:
            for obj in new_objects:
                self._l2_store.add(obj)

        self._state.stats.total_chunks += 1
        logger.debug(
            f"chunk 结束: 聚合 {len(new_objects)} 个 L2 对象, "
            f"store 总量 {len(self._l2_store)}"
        )

    # ------------------------------------------------------------------ #
    #  事件: on_turn_end (L2 聚合 + 合并)
    # ------------------------------------------------------------------ #

    def on_turn_end(
        self,
        messages: list[dict[str, Any]] | None = None,
    ) -> None:
        """
        对话 turn 结束时的异步事件。

        等价于 on_chunk_end 但增加 turn 级统计。

        Args:
            messages: 该 turn 内的消息列表。
        """
        self.on_chunk_end(messages=messages)
        self._state.current_turn += 1
        self._state.stats.total_turns += 1
        logger.debug(f"Turn {self._state.current_turn} 结束。")

    # ------------------------------------------------------------------ #
    #  事件: on_session_end (L3 异步总结)
    # ------------------------------------------------------------------ #

    def on_session_end(self) -> None:
        """
        会话结束时的异步事件。

        1. 将 L2 store 中的所有对象交给 L3 Summarizer 生成 profile entries
        2. 通过 L3 Reviser 处理冲突和覆盖
        3. 写入 L3 ProfileStore
        """
        if self._l3_summarizer is None or self._l3_store is None:
            logger.info("L3 未启用, 跳过 session-end 总结。")
            return

        if self._l2_store is None or len(self._l2_store) == 0:
            logger.info("L2 store 为空, 跳过 session-end 总结。")
            return

        # 获取 L2 对象
        l2_objects = self._l2_store.get_all()

        # L3 总结
        new_entries = self._l3_summarizer.summarize(l2_objects)
        self._state.stats.l3_summarize_count += 1

        # 修订冲突
        if self._l3_reviser is not None:
            for entry in new_entries:
                revised = self._l3_reviser.revise_or_add(entry, self._l3_store)
                if revised:
                    self._state.stats.l3_revise_count += 1
        else:
            for entry in new_entries:
                self._l3_store.add(entry)

        logger.info(
            f"Session {self._state.session_id} 结束: "
            f"新增 {len(new_entries)} 条 L3 entries, "
            f"L3 store 总量 {len(self._l3_store)}"
        )

    # ------------------------------------------------------------------ #
    #  检索接口 (供 Agent 调用)
    # ------------------------------------------------------------------ #

    def retrieve_l2(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[Any]:
        """检索 L2 中与 query 相关的记忆对象。"""
        if self._l2_retriever is None:
            return []
        self._state.stats.l2_retrieve_count += 1
        return self._l2_retriever.retrieve(query=query, top_k=top_k)

    def retrieve_l3(
        self,
        query: str | None = None,
        category: str | None = None,
        top_k: int = 10,
    ) -> list[Any]:
        """检索 L3 中与 query/category 相关的 profile entries。"""
        if self._l3_store is None:
            return []
        if category is not None:
            return self._l3_store.get_by_category(category)
        if query is not None:
            return self._l3_store.search(query, top_k=top_k)
        return self._l3_store.get_all()[:top_k]

    # ------------------------------------------------------------------ #
    #  缓冲区: 供外部推送消息
    # ------------------------------------------------------------------ #

    def push_message(self, message: dict[str, Any]) -> None:
        """推送一条消息到缓冲区, 可能触发自动 chunk 聚合。"""
        self._state.push_message(message)

        # 检查是否达到 chunk 大小阈值
        if len(self._state.message_buffer) >= self.config.l2_chunk_size:
            self.on_chunk_end()

    # ------------------------------------------------------------------ #
    #  序列化
    # ------------------------------------------------------------------ #

    def save_state(self, directory: str) -> None:
        """保存全部状态。"""
        self._state.save(directory)

    def load_state(self, directory: str) -> None:
        """恢复全部状态。"""
        self._state.load(directory)

    def __repr__(self) -> str:
        return (
            f"MemoryScheduler(state={self._state!r}, "
            f"config_l1={self.config.enable_l1}, "
            f"config_l2={self.config.enable_l2}, "
            f"config_l3={self.config.enable_l3})"
        )
