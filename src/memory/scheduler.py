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

from src.memory.l1.assoc_memory import AssociativeMemoryL1, L1Config
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

# MAG 导入已改为延迟导入 (在 init_mag 方法内部按需导入)
# 避免 scheduler → mag.context_selector 的循环导入链

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

    # MAG 配置
    enable_mag: bool = False
    mag_num_heads: int = 8
    mag_injection_layers: list[int] = field(default_factory=list)
    mag_share_parameters: bool = True
    mag_gate_init_bias: float = -2.0
    mag_selector_hidden_dim: int = 256
    mag_selector_top_k: int = 5
    mag_max_memory_tokens: int = 64


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
            l1_cfg = L1Config(
                d_key=config.l1_d_key,
                d_value=config.l1_d_value,
                n_heads=config.l1_num_heads,
                decay=config.l1_decay,
                write_strength=config.l1_write_strength,
            )
            self._l1 = AssociativeMemoryL1(l1_cfg)
            # L1Writer / L1Reader / L1Gate 已在 AssociativeMemoryL1 内部创建
            # scheduler 只需通过 self._l1 调用即可
            self._l1_writer = self._l1.writer
            self._l1_reader = self._l1.reader
            self._l1_gate = self._l1.gate

        # L2
        self._l2_aggregator: L2Aggregator | None = None
        self._l2_store: L2ObjectStore | None = None
        self._l2_merger: L2Merger | None = None
        self._l2_retriever: L2Retriever | None = None

        if config.enable_l2:
            l2_agg_cfg = {"aggregator_backend": "rule_based"}
            l2_store_cfg = {
                "max_objects": config.l2_max_objects,
                "max_age_turns": 50,
            }
            l2_merger_cfg = {
                "merge_similarity_threshold": config.l2_similarity_threshold,
            }
            l2_retriever_cfg = {
                "retrieval_top_k": 5,
            }
            self._l2_aggregator = L2Aggregator(l2_agg_cfg)
            self._l2_store = L2ObjectStore(l2_store_cfg)
            self._l2_merger = L2Merger(l2_merger_cfg)
            self._l2_retriever = L2Retriever(l2_retriever_cfg)

        # L3
        self._l3_summarizer: L3Summarizer | None = None
        self._l3_store: L3ProfileStore | None = None
        self._l3_reviser: L3Reviser | None = None

        if config.enable_l3:
            l3_summ_cfg = {"summarizer_backend": "rule_based"}
            l3_store_cfg = {
                "max_entries": config.l3_max_entries,
            }
            l3_reviser_cfg = {
                "contradiction_threshold": config.l3_conflict_threshold,
            }
            self._l3_summarizer = L3Summarizer(l3_summ_cfg)
            self._l3_store = L3ProfileStore(l3_store_cfg)
            self._l3_reviser = L3Reviser(l3_reviser_cfg)

        # 全局状态
        self._state = MoMState(
            l1=self._l1,
            l2_store=self._l2_store,
            l3_store=self._l3_store,
        )

        # MAG 组件 (延迟初始化, 需要 backbone 信息)
        self._mag_encoder: Any = None
        self._mag_selector: Any = None
        self._mag_gate: Any = None
        self._mag_initialized: bool = False

        logger.info(
            f"MemoryScheduler 初始化完成: "
            f"L1={'✓' if config.enable_l1 else '✗'}, "
            f"L2={'✓' if config.enable_l2 else '✗'}, "
            f"L3={'✓' if config.enable_l3 else '✗'}, "
            f"MAG={'✓' if config.enable_mag else '✗'}"
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
        if self._l1 is None:
            return None

        # 确保投影层已初始化
        d_model = hidden_states.shape[-1]
        self._l1.set_hidden_dim(d_model)

        # 使用 AssociativeMemoryL1 的 forward 方法：写入 + 读取 + 门控
        output = self._l1.forward(hidden_states, update=True)
        self._state.stats.l1_write_count += 1
        self._state.stats.l1_read_count += 1

        return output

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
            existing = self._l2_store.get_all_active()
            decisions = self._l2_merger.decide_and_merge(new_objects, existing)
            for new_obj, action, merge_target in decisions:
                if action == "merge" and merge_target is not None:
                    self._l2_store.merge(
                        merge_target.object_id,
                        new_obj.object_id if self._l2_store.get(new_obj.object_id) else "",
                        self._l2_merger.merge_texts(
                            merge_target.summary_text, new_obj.summary_text
                        ),
                    )
                    self._state.stats.l2_merge_count += 1
                elif action == "replace" and merge_target is not None:
                    self._l2_store.remove(merge_target.object_id)
                    self._l2_store.add(new_obj)
                    self._state.stats.l2_merge_count += 1
                else:  # append
                    self._l2_store.add(new_obj)
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
        l2_objects = self._l2_store.get_all_active()

        # L3 总结
        new_entries = self._l3_summarizer.summarize(l2_objects)
        self._state.stats.l3_summarize_count += 1

        # 修订冲突
        if self._l3_reviser is not None:
            final_entries = self._l3_reviser.apply_revisions(new_entries, self._l3_store)
            for entry in final_entries:
                self._l3_store.add(entry)
            if len(new_entries) != len(final_entries):
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
        if self._l2_retriever is None or self._l2_store is None:
            return []
        self._state.stats.l2_retrieve_count += 1
        objects = self._l2_store.get_all_active()
        return self._l2_retriever.retrieve(query=query, objects=objects, top_k=top_k)

    def retrieve_l3(
        self,
        query: str | None = None,
        category: str | None = None,
        top_k: int = 3,
        min_score: float = 1.0,
    ) -> list[Any]:
        """检索 L3 中与 query/category 相关的 profile entries。

        Args:
            query: 检索查询文本.
            category: 按类别检索 (优先于 query).
            top_k: 返回最相关的 top-k 条目.
            min_score: 最低相关度分数阈值, 低于此分数的条目不返回.
        """
        if self._l3_store is None:
            return []
        if category is not None:
            return self._l3_store.get_by_category(category)
        # 使用语义检索（关键词匹配 + 置信度加权），设置最低分数门槛过滤噪声
        if query:
            return self._l3_store.search(query=query, top_k=top_k, min_score=min_score)
        return self._l3_store.list_all()[:top_k]

    # ------------------------------------------------------------------ #
    #  MAG: 记忆编码与选择
    # ------------------------------------------------------------------ #

    def init_mag(
        self,
        backbone_model: Any,
        tokenizer: Any,
        hidden_dim: int,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> Any:
        """初始化 MAG 组件 (MemoryEncoder + ContextSelector + MAGGate)。

        需要在 backbone 构建之后调用。

        Args:
            backbone_model: HuggingFace 模型或 Debug 模型.
            tokenizer: HuggingFace tokenizer.
            hidden_dim: backbone 隐藏维度.
            device: 设备.
            dtype: 数据类型.

        Returns:
            MAGGate 实例 (供 backbone.set_mag_gate() 使用).
        """
        if not self.config.enable_mag:
            logger.info("MAG 未启用, 跳过初始化")
            return None

        # 延迟导入 MAG 模块 (避免循环导入)
        try:
            from src.memory.mag.memory_encoder import MemoryEncoder, MemoryEncoderConfig
            from src.memory.mag.context_selector import ContextSelector, ContextSelectorConfig
            from src.memory.mag.mag_gate import MAGGate, MAGGateConfig
        except ImportError:
            logger.warning("MAG 模块不可用, 跳过 MAG 初始化")
            return None

        # 1. MemoryEncoder
        enc_cfg = MemoryEncoderConfig(
            max_memory_tokens=self.config.mag_max_memory_tokens,
            pooling="mean",
        )
        self._mag_encoder = MemoryEncoder(enc_cfg)
        self._mag_encoder.set_backbone(
            backbone_model=backbone_model,
            tokenizer=tokenizer,
            hidden_dim=hidden_dim,
            device=device,
            dtype=dtype,
        )

        # 2. ContextSelector
        sel_cfg = ContextSelectorConfig(
            input_dim=hidden_dim,
            hidden_dim=self.config.mag_selector_hidden_dim,
            top_k=self.config.mag_selector_top_k,
        )
        self._mag_selector = ContextSelector(sel_cfg)
        self._mag_selector = self._mag_selector.to(device=device)

        # 3. MAGGate
        # 确定注入层: 如果配置为空, 默认均匀选取 4 层
        injection_layers = self.config.mag_injection_layers
        if not injection_layers:
            # 默认: 在 1/4, 1/2, 3/4, 最后一层 注入
            num_layers = 24  # 默认值, 后续可从 backbone 获取
            injection_layers = [
                num_layers // 4,
                num_layers // 2,
                num_layers * 3 // 4,
                num_layers - 1,
            ]

        gate_cfg = MAGGateConfig(
            hidden_dim=hidden_dim,
            num_heads=self.config.mag_num_heads,
            memory_dim=hidden_dim,  # encoder output_dim = hidden_dim (无投影)
            injection_layers=injection_layers,
            share_parameters=self.config.mag_share_parameters,
            gate_init_bias=self.config.mag_gate_init_bias,
        )
        self._mag_gate = MAGGate(gate_cfg)
        self._mag_gate = self._mag_gate.to(device=device)

        self._mag_initialized = True
        logger.info(
            f"[MemoryScheduler] MAG 初始化完成: "
            f"encoder_dim={hidden_dim}, "
            f"selector_top_k={self.config.mag_selector_top_k}, "
            f"injection_layers={injection_layers}"
        )
        return self._mag_gate

    def encode_memories_for_mag(
        self, query: str,
    ) -> tuple[torch.Tensor, torch.Tensor | None, list[str]]:
        """编码 L2/L3 记忆为向量 (供 MAG forward 使用)。

        Args:
            query: 当前用户查询 (用于 context selection).

        Returns:
            (memory_vectors, selection_weights, memory_texts):
                memory_vectors: (1, K, D) 编码后的记忆向量.
                selection_weights: (1, K) 选择权重, 或 None.
                memory_texts: 对应的文本列表.
        """
        if not self._mag_initialized or self._mag_encoder is None:
            return torch.zeros(1, 0, 1), None, []

        # 获取 L2/L3 记忆对象
        l2_objects = []
        l3_entries = []
        if self._l2_store is not None:
            l2_results = self.retrieve_l2(query=query, top_k=10)  # 多取一些, selector 会筛选
            l2_objects = [obj for obj, _score in l2_results]
        if self._l3_store is not None:
            l3_entries = self.retrieve_l3(query=query, top_k=5)

        if not l2_objects and not l3_entries:
            D = self._mag_encoder.output_dim
            return torch.zeros(1, 0, D), None, []

        # 编码
        memory_vectors, memory_texts = self._mag_encoder.encode(
            l2_objects=l2_objects,
            l3_entries=l3_entries,
        )

        if memory_vectors.shape[0] == 0:
            D = self._mag_encoder.output_dim
            return torch.zeros(1, 0, D), None, []

        # 添加 batch 维度: (K, D) → (1, K, D)
        memory_vectors = memory_vectors.unsqueeze(0)

        # Context Selection
        selection_weights = None
        if self._mag_selector is not None:
            # 编码 query
            query_emb = self._mag_encoder.encode_texts([query])  # (1, D)
            selection_weights = self._mag_selector.soft_select(
                query_emb=query_emb,
                memory_embs=memory_vectors,
            )  # (1, K)

        return memory_vectors, selection_weights, memory_texts

    @property
    def mag_gate(self) -> Any:
        """返回 MAGGate 实例 (供外部使用)。"""
        return self._mag_gate

    @property
    def mag_encoder(self) -> Any:
        """返回 MemoryEncoder 实例。"""
        return self._mag_encoder

    @property
    def mag_selector(self) -> Any:
        """返回 ContextSelector 实例。"""
        return self._mag_selector

    @property
    def mag_initialized(self) -> bool:
        """MAG 是否已初始化。"""
        return self._mag_initialized

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
