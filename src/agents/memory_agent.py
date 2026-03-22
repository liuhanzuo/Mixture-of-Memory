"""
MemoryAgent: 带层次化记忆的智能体。

封装 backbone + MemoryScheduler + TurnProcessor，
对外提供简洁的 chat / reset / export 接口。

这是 MoM 系统的主入口类，使用者只需:
    agent = MemoryAgent(config)
    reply = agent.chat("Hello!")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf

from src.backbone.interfaces import BackboneModel
from src.memory.scheduler import MemoryScheduler, SchedulerConfig
from src.memory.l3.formatter import L3Formatter
from src.agents.turn_processor import TurnProcessor, TurnResult

logger = logging.getLogger(__name__)


@dataclass
class AgentConfig:
    """MemoryAgent 的统一配置。"""

    # ---- 记忆层开关 ----
    enable_l1: bool = True
    enable_l2: bool = True
    enable_l3: bool = True

    # ---- 记忆参数 (透传给 SchedulerConfig) ----
    l1_d_key: int = 64
    l1_d_value: int = 64
    l1_num_heads: int = 4
    l1_decay: float = 0.95
    l1_write_strength: float = 1.0
    l1_gate_d_model: int = 64
    l1_gate_hidden_dim: int = 64
    l2_max_objects: int = 200
    l2_similarity_threshold: float = 0.75
    l2_chunk_size: int = 5
    l3_max_entries: int = 100
    l3_conflict_threshold: float = 0.8

    # ---- TurnProcessor 参数 ----
    max_context_chars: int = 3000
    l2_top_k: int = 5
    l3_top_k: int = 10

    # ---- Agent 参数 ----
    system_prompt: str = "You are a helpful assistant with long-term memory."
    max_history_turns: int = 20

    @classmethod
    def from_omegaconf(cls, cfg: DictConfig) -> "AgentConfig":
        """从 OmegaConf DictConfig 构建。"""
        flat = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(flat, dict):
            flat = {}
        return cls(**{
            k: v for k, v in flat.items()
            if k in cls.__dataclass_fields__
        })


class MemoryAgent:
    """带层次化记忆 (MoM) 的智能体。

    使用方式::

        agent = MemoryAgent.from_config(cfg)
        agent.start_session("session_001")

        r1 = agent.chat("What is FSDP?")
        r2 = agent.chat("How does it differ from DDP?")

        agent.end_session()
        agent.export_profile("outputs/profile.md")

    内部架构::

        MemoryAgent
        ├── MemoryScheduler  (管理 L1/L2/L3)
        ├── TurnProcessor    (单轮处理: 检索→生成→更新)
        ├── BackboneModel    (可选, 语言模型)
        └── conversation_history (对话历史缓存)
    """

    def __init__(
        self,
        config: AgentConfig | dict[str, Any] | None = None,
        backbone: BackboneModel | None = None,
        tokenizer: Any | None = None,
    ):
        # ---- 配置 ---- #
        if config is None:
            config = AgentConfig()
        elif isinstance(config, dict):
            config = AgentConfig(**{
                k: v for k, v in config.items()
                if k in AgentConfig.__dataclass_fields__
            })
        self.config = config

        # ---- 构建 MemoryScheduler ---- #
        sched_cfg = SchedulerConfig(
            enable_l1=config.enable_l1,
            enable_l2=config.enable_l2,
            enable_l3=config.enable_l3,
            l1_d_key=config.l1_d_key,
            l1_d_value=config.l1_d_value,
            l1_num_heads=config.l1_num_heads,
            l1_decay=config.l1_decay,
            l1_write_strength=config.l1_write_strength,
            l1_gate_d_model=config.l1_gate_d_model,
            l1_gate_hidden_dim=config.l1_gate_hidden_dim,
            l2_max_objects=config.l2_max_objects,
            l2_similarity_threshold=config.l2_similarity_threshold,
            l2_chunk_size=config.l2_chunk_size,
            l3_max_entries=config.l3_max_entries,
            l3_conflict_threshold=config.l3_conflict_threshold,
        )
        self.scheduler = MemoryScheduler(sched_cfg)

        # ---- 构建 TurnProcessor ---- #
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.turn_processor = TurnProcessor(
            scheduler=self.scheduler,
            backbone=backbone,
            tokenizer=tokenizer,
            max_context_chars=config.max_context_chars,
            l2_top_k=config.l2_top_k,
            l3_top_k=config.l3_top_k,
        )

        # ---- 对话状态 ---- #
        self._conversation_history: list[dict[str, str]] = []
        self._turn_counter: int = 0
        self._session_active: bool = False

        # ---- L3 Formatter ---- #
        self._formatter = L3Formatter()

        logger.info(
            f"[MemoryAgent] 初始化完成. "
            f"L1={config.enable_l1}, L2={config.enable_l2}, L3={config.enable_l3}"
        )

    # ------------------------------------------------------------------ #
    #  工厂方法
    # ------------------------------------------------------------------ #

    @classmethod
    def from_config(
        cls,
        cfg: DictConfig | dict[str, Any],
        backbone: BackboneModel | None = None,
        tokenizer: Any | None = None,
    ) -> "MemoryAgent":
        """从配置构建 MemoryAgent。"""
        if isinstance(cfg, DictConfig):
            agent_cfg = AgentConfig.from_omegaconf(cfg)
        elif isinstance(cfg, dict):
            agent_cfg = AgentConfig(**{
                k: v for k, v in cfg.items()
                if k in AgentConfig.__dataclass_fields__
            })
        else:
            agent_cfg = AgentConfig()
        return cls(config=agent_cfg, backbone=backbone, tokenizer=tokenizer)

    # ------------------------------------------------------------------ #
    #  会话管理
    # ------------------------------------------------------------------ #

    def start_session(self, session_id: str | None = None) -> None:
        """开始一个新会话。

        清空短期记忆 (L1, L2), 保留 L3 长期画像。
        """
        self.scheduler.init_state(session_id=session_id)
        self._conversation_history.clear()
        self._turn_counter = 0
        self._session_active = True
        logger.info(f"[MemoryAgent] Session started: {self.scheduler.state.session_id}")

    def end_session(self) -> None:
        """结束当前会话。

        触发 L3 session-end 总结。
        """
        if not self._session_active:
            logger.warning("[MemoryAgent] No active session to end.")
            return

        self.scheduler.on_session_end()
        self._session_active = False
        logger.info(
            f"[MemoryAgent] Session ended: {self.scheduler.state.session_id}. "
            f"Stats: {self.scheduler.stats.to_dict()}"
        )

    # ------------------------------------------------------------------ #
    #  核心对话接口
    # ------------------------------------------------------------------ #

    def chat(self, user_message: str) -> str:
        """处理一条用户消息并返回助手回复。

        这是最简洁的对外接口。内部会:
        1. 递增 turn 计数
        2. 调用 TurnProcessor 执行完整的单轮流程
        3. 维护对话历史

        Args:
            user_message: 用户输入文本.

        Returns:
            助手的回复文本.
        """
        if not self._session_active:
            self.start_session()

        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter:04d}"

        # 准备最近的对话历史 (不超过 max_history_turns)
        recent_history = self._conversation_history[-self.config.max_history_turns:]

        # 调用 TurnProcessor (传入完整历史用于全量搜索)
        result = self.turn_processor.process_turn(
            user_message=user_message,
            turn_id=turn_id,
            conversation_history=recent_history,
            full_conversation_history=list(self._conversation_history),
            system_prompt=self.config.system_prompt,
        )

        # 更新对话历史
        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": result.response_text})

        return result.response_text

    def chat_detailed(self, user_message: str) -> TurnResult:
        """处理一条用户消息并返回完整的 TurnResult (含检索/耗时详情)。"""
        if not self._session_active:
            self.start_session()

        self._turn_counter += 1
        turn_id = f"turn_{self._turn_counter:04d}"

        recent_history = self._conversation_history[-self.config.max_history_turns:]

        result = self.turn_processor.process_turn(
            user_message=user_message,
            turn_id=turn_id,
            conversation_history=recent_history,
            full_conversation_history=list(self._conversation_history),
            system_prompt=self.config.system_prompt,
        )

        self._conversation_history.append({"role": "user", "content": user_message})
        self._conversation_history.append({"role": "assistant", "content": result.response_text})

        return result

    # ------------------------------------------------------------------ #
    #  状态查询
    # ------------------------------------------------------------------ #

    @property
    def session_id(self) -> str:
        return self.scheduler.state.session_id

    @property
    def turn_count(self) -> int:
        return self._turn_counter

    @property
    def conversation_history(self) -> list[dict[str, str]]:
        return list(self._conversation_history)

    @property
    def is_active(self) -> bool:
        return self._session_active

    def get_stats(self) -> dict[str, Any]:
        """返回记忆系统的运行时统计。"""
        stats = self.scheduler.stats.to_dict()
        stats["agent_turn_count"] = self._turn_counter
        stats["agent_session_active"] = self._session_active
        stats["conversation_length"] = len(self._conversation_history)
        return stats

    def get_state_snapshot(self) -> dict[str, Any]:
        """返回记忆系统的状态快照。"""
        return self.scheduler.state.snapshot()

    # ------------------------------------------------------------------ #
    #  画像导出
    # ------------------------------------------------------------------ #

    def export_profile(
        self,
        output_path: str,
        format: str = "markdown",
    ) -> str:
        """导出 L3 长期画像。

        Args:
            output_path: 输出文件路径.
            format: "markdown" 或 "json".

        Returns:
            实际写入的文件路径.
        """
        if self.scheduler._l3_store is None:
            logger.warning("[MemoryAgent] L3 store not available.")
            return ""

        if format == "json":
            result = self._formatter.export_json(self.scheduler._l3_store, output_path)
        else:
            result = self._formatter.export_markdown(self.scheduler._l3_store, output_path)

        return str(result)

    def get_profile_text(self) -> str:
        """获取当前 L3 画像的文本摘要。"""
        if self.scheduler._l3_store is None:
            return "[L3 not enabled]"
        return self._formatter.format_summary(self.scheduler._l3_store)

    # ------------------------------------------------------------------ #
    #  持久化
    # ------------------------------------------------------------------ #

    def save(self, directory: str) -> None:
        """保存 agent 状态到目录。"""
        self.scheduler.save_state(directory)
        logger.info(f"[MemoryAgent] State saved to {directory}")

    def load(self, directory: str) -> None:
        """从目录恢复 agent 状态。"""
        self.scheduler.load_state(directory)
        logger.info(f"[MemoryAgent] State loaded from {directory}")

    # ------------------------------------------------------------------ #
    #  重置
    # ------------------------------------------------------------------ #

    def reset(self, keep_l3: bool = True) -> None:
        """重置 agent 状态。

        Args:
            keep_l3: 是否保留 L3 长期画像 (默认 True).
        """
        if keep_l3:
            self.scheduler.state.soft_reset()
        else:
            self.scheduler.state.reset()
        self._conversation_history.clear()
        self._turn_counter = 0
        self._session_active = False
        logger.info(f"[MemoryAgent] Reset complete. keep_l3={keep_l3}")

    def __repr__(self) -> str:
        return (
            f"MemoryAgent(session={self.session_id!r}, turns={self._turn_counter}, "
            f"active={self._session_active}, "
            f"L1={self.config.enable_l1}, L2={self.config.enable_l2}, L3={self.config.enable_l3})"
        )
