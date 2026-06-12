"""
SessionRunner: 会话运行器。

管理多轮对话的完整生命周期，包括:
- 从文件/列表加载对话数据
- 逐轮驱动 MemoryAgent
- 收集每轮的 TurnResult 用于评估
- 会话结束后触发 L3 总结
- 导出对话 trace 用于分析

适用于:
- 交互式 chat demo
- 批量评测 (synthetic tasks)
- ablation 实验
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

from src.agents.memory_agent import MemoryAgent, AgentConfig
from src.agents.turn_processor import TurnResult

logger = logging.getLogger(__name__)


@dataclass
class SessionTrace:
    """一次完整会话的运行 trace。"""

    session_id: str = ""
    turns: list[TurnResult] = field(default_factory=list)
    user_messages: list[str] = field(default_factory=list)
    agent_replies: list[str] = field(default_factory=list)
    total_time_ms: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)
    state_snapshot: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """序列化为字典 (不含 backbone_output 等不可序列化对象)。"""
        turn_dicts = []
        for t in self.turns:
            turn_dicts.append({
                "turn_id": t.turn_id,
                "response_text": t.response_text,
                "memory_context": t.memory_context,
                "l2_retrieved_count": len(t.retrieved_l2),
                "l3_retrieved_count": len(t.retrieved_l3),
                "retrieval_time_ms": t.retrieval_time_ms,
                "generation_time_ms": t.generation_time_ms,
                "update_time_ms": t.update_time_ms,
            })
        return {
            "session_id": self.session_id,
            "num_turns": len(self.turns),
            "total_time_ms": self.total_time_ms,
            "user_messages": self.user_messages,
            "agent_replies": self.agent_replies,
            "turns": turn_dicts,
            "stats": self.stats,
            "state_snapshot": self.state_snapshot,
        }

    def save(self, path: str | Path) -> None:
        """保存 trace 到 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"[SessionTrace] Saved to {path}")


class SessionRunner:
    """会话运行器: 驱动 MemoryAgent 完成一次或多次完整会话。

    支持三种运行模式:

    1. **交互式**: ``run_interactive()`` — 从 stdin 读取用户输入
    2. **脚本式**: ``run_conversation(messages)`` — 从消息列表运行
    3. **批量式**: ``run_batch(conversations)`` — 运行多个对话

    Usage::

        agent = MemoryAgent(config)
        runner = SessionRunner(agent)

        # 脚本式
        trace = runner.run_conversation(
            messages=["Hi", "What is FSDP?", "Thanks"],
            session_id="demo_001",
        )
        trace.save("outputs/traces/demo_001.json")

        # 批量式
        traces = runner.run_batch([
            {"session_id": "s1", "messages": ["Hello", "Bye"]},
            {"session_id": "s2", "messages": ["Tell me about MoE"]},
        ])
    """

    def __init__(self, agent: MemoryAgent):
        self.agent = agent

    # ------------------------------------------------------------------ #
    #  脚本式运行
    # ------------------------------------------------------------------ #

    def run_conversation(
        self,
        messages: list[str],
        session_id: str | None = None,
    ) -> SessionTrace:
        """运行一次完整的脚本式对话。

        Args:
            messages: 用户消息列表 (按时间顺序).
            session_id: 会话 ID (可选).

        Returns:
            SessionTrace 包含完整的对话记录和统计。
        """
        trace = SessionTrace()
        t_start = time.monotonic()

        # 开始会话
        self.agent.start_session(session_id=session_id)
        trace.session_id = self.agent.session_id

        logger.info(
            f"[SessionRunner] Starting conversation: "
            f"session={trace.session_id}, {len(messages)} messages"
        )

        # 逐轮处理
        for i, user_msg in enumerate(messages):
            logger.info(f"[SessionRunner] Turn {i + 1}/{len(messages)}: {user_msg[:80]}...")
            result = self.agent.chat_detailed(user_msg)

            trace.turns.append(result)
            trace.user_messages.append(user_msg)
            trace.agent_replies.append(result.response_text)

        # 结束会话
        self.agent.end_session()

        # 收集统计
        trace.total_time_ms = (time.monotonic() - t_start) * 1000
        trace.stats = self.agent.get_stats()
        trace.state_snapshot = self.agent.get_state_snapshot()

        logger.info(
            f"[SessionRunner] Conversation complete: "
            f"{len(messages)} turns, {trace.total_time_ms:.1f}ms total"
        )

        return trace

    # ------------------------------------------------------------------ #
    #  批量运行
    # ------------------------------------------------------------------ #

    def run_batch(
        self,
        conversations: list[dict[str, Any]],
        keep_l3_across_sessions: bool = True,
    ) -> list[SessionTrace]:
        """批量运行多个对话。

        Args:
            conversations: 对话列表, 每个元素为:
                {"session_id": str, "messages": list[str]}
            keep_l3_across_sessions: 是否跨会话保留 L3 长期记忆.

        Returns:
            每个对话对应的 SessionTrace 列表.
        """
        traces: list[SessionTrace] = []

        for idx, conv in enumerate(conversations):
            session_id = conv.get("session_id", f"batch_{idx:04d}")
            messages = conv.get("messages", [])

            if not messages:
                logger.warning(f"[SessionRunner] Skipping empty conversation: {session_id}")
                continue

            # 根据配置决定是否在会话间保留 L3
            if not keep_l3_across_sessions and idx > 0:
                self.agent.reset(keep_l3=False)

            trace = self.run_conversation(messages=messages, session_id=session_id)
            traces.append(trace)

            logger.info(
                f"[SessionRunner] Batch progress: {idx + 1}/{len(conversations)} "
                f"sessions complete"
            )

        return traces

    # ------------------------------------------------------------------ #
    #  交互式运行
    # ------------------------------------------------------------------ #

    def run_interactive(
        self,
        session_id: str | None = None,
        greeting: str | None = None,
    ) -> SessionTrace:
        """运行交互式对话 (从 stdin 读取输入)。

        输入 'quit', 'exit' 或 'q' 结束对话。
        输入 '/stats' 查看记忆统计。
        输入 '/profile' 查看当前 L3 画像。
        输入 '/snapshot' 查看状态快照。

        Args:
            session_id: 会话 ID.
            greeting: 开场问候语 (可选).

        Returns:
            SessionTrace.
        """
        trace = SessionTrace()
        t_start = time.monotonic()

        self.agent.start_session(session_id=session_id)
        trace.session_id = self.agent.session_id

        if greeting:
            print(f"\n🤖 {greeting}\n")

        print("=" * 60)
        print(f"MoM Agent — Session: {trace.session_id}")
        print("Type 'quit' to exit | '/stats' for memory stats")
        print("=" * 60)

        while True:
            try:
                user_input = input("\n👤 You: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\n[Session interrupted]")
                break

            if not user_input:
                continue

            # 特殊命令
            if user_input.lower() in ("quit", "exit", "q"):
                print("\n[Ending session...]")
                break

            if user_input == "/stats":
                stats = self.agent.get_stats()
                print("\n📊 Memory Stats:")
                for k, v in stats.items():
                    print(f"   {k}: {v}")
                continue

            if user_input == "/profile":
                profile = self.agent.get_profile_text()
                print(f"\n📋 Profile: {profile}")
                continue

            if user_input == "/snapshot":
                snap = self.agent.get_state_snapshot()
                print("\n📸 State Snapshot:")
                for k, v in snap.items():
                    print(f"   {k}: {v}")
                continue

            # 正常对话
            result = self.agent.chat_detailed(user_input)

            print(f"\n🤖 Assistant: {result.response_text}")
            print(
                f"   [L2: {len(result.retrieved_l2)} items | "
                f"L3: {len(result.retrieved_l3)} items | "
                f"{result.retrieval_time_ms:.0f}ms retrieval | "
                f"{result.generation_time_ms:.0f}ms generation]"
            )

            trace.turns.append(result)
            trace.user_messages.append(user_input)
            trace.agent_replies.append(result.response_text)

        # 结束会话
        self.agent.end_session()
        trace.total_time_ms = (time.monotonic() - t_start) * 1000
        trace.stats = self.agent.get_stats()
        trace.state_snapshot = self.agent.get_state_snapshot()

        print(f"\n✅ Session ended. {len(trace.turns)} turns, {trace.total_time_ms:.0f}ms total.")
        return trace

    # ------------------------------------------------------------------ #
    #  从文件加载并运行
    # ------------------------------------------------------------------ #

    def run_from_file(
        self,
        input_path: str | Path,
        output_dir: str | Path | None = None,
    ) -> list[SessionTrace]:
        """从 JSON 文件加载对话并批量运行。

        文件格式::

            [
                {"session_id": "s1", "messages": ["Hello", "How are you?"]},
                {"session_id": "s2", "messages": ["Tell me about MoE"]}
            ]

        Args:
            input_path: 输入 JSON 文件路径.
            output_dir: 输出 trace 目录 (可选).

        Returns:
            SessionTrace 列表.
        """
        input_path = Path(input_path)

        with open(input_path, "r", encoding="utf-8") as f:
            conversations = json.load(f)

        if not isinstance(conversations, list):
            conversations = [conversations]

        logger.info(f"[SessionRunner] Loaded {len(conversations)} conversations from {input_path}")

        traces = self.run_batch(conversations)

        # 保存 traces
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            for trace in traces:
                trace_path = output_dir / f"{trace.session_id}.json"
                trace.save(trace_path)

        return traces

    # ------------------------------------------------------------------ #
    #  迭代器模式 (用于自定义评测循环)
    # ------------------------------------------------------------------ #

    def iter_turns(
        self,
        messages: list[str],
        session_id: str | None = None,
    ) -> Iterator[TurnResult]:
        """以迭代器方式逐轮运行，适用于自定义评测循环。

        Usage::

            for result in runner.iter_turns(messages):
                # 自定义评估逻辑
                score = my_eval(result)

        Args:
            messages: 用户消息列表.
            session_id: 会话 ID.

        Yields:
            每轮的 TurnResult.
        """
        self.agent.start_session(session_id=session_id)

        try:
            for user_msg in messages:
                result = self.agent.chat_detailed(user_msg)
                yield result
        finally:
            self.agent.end_session()
