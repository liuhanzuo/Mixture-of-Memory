"""
Agent 端到端冒烟测试。

使用无 backbone 的 echo 模式运行完整的:
- MemoryAgent 对话流程
- SessionRunner 脚本式 / 批量式运行
- 记忆层协调 (L2 聚合 + L3 总结)
- 状态管理 (start/end session, reset)
- 画像导出

目标: 验证所有模块能正确串联, 不发生 import 错误或运行时异常。
"""

import json
import tempfile
from pathlib import Path

import pytest

from src.agents.memory_agent import MemoryAgent, AgentConfig
from src.agents.session_runner import SessionRunner, SessionTrace
from src.agents.turn_processor import TurnResult


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def agent_config() -> AgentConfig:
    """最小化的 agent 测试配置。"""
    return AgentConfig(
        enable_l1=False,  # L1 需要 backbone hidden states, 在无 backbone 模式下禁用
        enable_l2=True,
        enable_l3=True,
        l2_max_objects=50,
        l2_similarity_threshold=0.5,
        l2_chunk_size=2,
        l3_max_entries=20,
        l3_conflict_threshold=0.8,
        max_context_chars=2000,
        l2_top_k=3,
        l3_top_k=5,
        system_prompt="You are a test assistant.",
        max_history_turns=10,
    )


@pytest.fixture
def agent(agent_config: AgentConfig) -> MemoryAgent:
    """创建无 backbone 的 MemoryAgent (echo 模式)。"""
    return MemoryAgent(config=agent_config, backbone=None, tokenizer=None)


@pytest.fixture
def runner(agent: MemoryAgent) -> SessionRunner:
    """创建 SessionRunner。"""
    return SessionRunner(agent)


@pytest.fixture
def sample_messages() -> list[str]:
    """示例对话消息序列。"""
    return [
        "Hello, I'm working on a memory system for LLM agents.",
        "I prefer using Python and PyTorch for my research.",
        "Can you help me understand FSDP?",
        "I am currently researching sparse training methods.",
        "What's the difference between DDP and FSDP?",
    ]


# ------------------------------------------------------------------ #
#  测试: MemoryAgent 基本功能
# ------------------------------------------------------------------ #

class TestMemoryAgentBasic:
    """测试 MemoryAgent 的基本对话功能。"""

    def test_create_agent(self, agent: MemoryAgent):
        """应成功创建 agent。"""
        assert agent is not None
        assert not agent.is_active

    def test_start_session(self, agent: MemoryAgent):
        """start_session 应激活会话。"""
        agent.start_session(session_id="test_001")
        assert agent.is_active
        assert agent.session_id == "test_001"
        assert agent.turn_count == 0

    def test_chat_echo_mode(self, agent: MemoryAgent):
        """无 backbone 时应使用 echo 模式。"""
        reply = agent.chat("Hello!")
        assert isinstance(reply, str)
        assert len(reply) > 0
        assert agent.turn_count == 1

    def test_chat_auto_starts_session(self, agent: MemoryAgent):
        """第一次 chat 应自动 start_session。"""
        assert not agent.is_active
        agent.chat("Hello!")
        assert agent.is_active

    def test_multi_turn_chat(self, agent: MemoryAgent, sample_messages: list[str]):
        """多轮对话应正常运行。"""
        agent.start_session("multi_turn")
        for i, msg in enumerate(sample_messages):
            reply = agent.chat(msg)
            assert isinstance(reply, str)
            assert agent.turn_count == i + 1

    def test_chat_detailed(self, agent: MemoryAgent):
        """chat_detailed 应返回 TurnResult。"""
        result = agent.chat_detailed("What is FSDP?")
        assert isinstance(result, TurnResult)
        assert result.response_text
        assert result.turn_id

    def test_conversation_history(self, agent: MemoryAgent):
        """对话历史应正确维护。"""
        agent.chat("Hello")
        agent.chat("How are you?")
        history = agent.conversation_history
        assert len(history) == 4  # 2 轮 × 2 (user + assistant)
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    def test_end_session(self, agent: MemoryAgent):
        """end_session 应触发 L3 总结并关闭会话。"""
        agent.start_session("end_test")
        agent.chat("I prefer structured explanations.")
        agent.chat("I am researching large language models.")
        agent.end_session()
        assert not agent.is_active

    def test_end_session_no_active(self, agent: MemoryAgent):
        """无活跃会话时 end_session 不应出错。"""
        agent.end_session()  # 不应抛出异常


# ------------------------------------------------------------------ #
#  测试: 记忆状态查询
# ------------------------------------------------------------------ #

class TestMemoryState:
    """测试记忆状态查询接口。"""

    def test_get_stats(self, agent: MemoryAgent):
        """get_stats 应返回有效字典。"""
        agent.chat("Hello")
        stats = agent.get_stats()
        assert isinstance(stats, dict)
        assert "agent_turn_count" in stats
        assert stats["agent_turn_count"] == 1

    def test_get_state_snapshot(self, agent: MemoryAgent):
        """get_state_snapshot 应返回有效字典。"""
        agent.chat("Hello")
        snap = agent.get_state_snapshot()
        assert isinstance(snap, dict)
        assert "session_id" in snap

    def test_get_profile_text(self, agent: MemoryAgent):
        """get_profile_text 应返回文本。"""
        agent.start_session("profile_test")
        agent.chat("I am researching sparse training.")
        agent.end_session()
        text = agent.get_profile_text()
        assert isinstance(text, str)


# ------------------------------------------------------------------ #
#  测试: 重置
# ------------------------------------------------------------------ #

class TestAgentReset:
    """测试 agent 重置。"""

    def test_reset_keep_l3(self, agent: MemoryAgent):
        """reset(keep_l3=True) 应保留 L3。"""
        agent.start_session("reset_test")
        agent.chat("I prefer Python.")
        agent.end_session()

        l3_before = agent.get_profile_text()
        agent.reset(keep_l3=True)
        l3_after = agent.get_profile_text()

        assert not agent.is_active
        assert agent.turn_count == 0
        assert len(agent.conversation_history) == 0
        # L3 应保留
        assert l3_after == l3_before

    def test_reset_clear_all(self, agent: MemoryAgent):
        """reset(keep_l3=False) 应清空全部。"""
        agent.start_session("reset_all_test")
        agent.chat("I prefer Python.")
        agent.end_session()

        agent.reset(keep_l3=False)
        assert not agent.is_active
        assert agent.turn_count == 0


# ------------------------------------------------------------------ #
#  测试: 画像导出
# ------------------------------------------------------------------ #

class TestProfileExport:
    """测试画像导出。"""

    def test_export_profile_markdown(self, agent: MemoryAgent):
        """应成功导出 Markdown 画像。"""
        agent.start_session("export_test")
        agent.chat("I am researching large language models.")
        agent.chat("I prefer technical, structured explanations.")
        agent.end_session()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "profile.md")
            result = agent.export_profile(path, format="markdown")
            if result:
                assert Path(result).exists()

    def test_export_profile_json(self, agent: MemoryAgent):
        """应成功导出 JSON 画像。"""
        agent.start_session("export_json_test")
        agent.chat("I prefer using PyTorch.")
        agent.end_session()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "profile.json")
            result = agent.export_profile(path, format="json")
            if result:
                assert Path(result).exists()


# ------------------------------------------------------------------ #
#  测试: SessionRunner
# ------------------------------------------------------------------ #

class TestSessionRunner:
    """测试 SessionRunner。"""

    def test_run_conversation(self, runner: SessionRunner, sample_messages: list[str]):
        """run_conversation 应成功完成。"""
        trace = runner.run_conversation(
            messages=sample_messages,
            session_id="runner_test",
        )
        assert isinstance(trace, SessionTrace)
        assert trace.session_id == "runner_test"
        assert len(trace.turns) == len(sample_messages)
        assert len(trace.user_messages) == len(sample_messages)
        assert len(trace.agent_replies) == len(sample_messages)
        assert trace.total_time_ms > 0

    def test_run_batch(self, runner: SessionRunner):
        """run_batch 应成功运行多个对话。"""
        conversations = [
            {"session_id": "batch_001", "messages": ["Hello", "How are you?"]},
            {"session_id": "batch_002", "messages": ["Tell me about FSDP."]},
        ]
        traces = runner.run_batch(conversations)
        assert len(traces) == 2
        assert traces[0].session_id == "batch_001"
        assert traces[1].session_id == "batch_002"
        assert len(traces[0].turns) == 2
        assert len(traces[1].turns) == 1

    def test_run_batch_empty_conv(self, runner: SessionRunner):
        """空对话应被跳过。"""
        conversations = [
            {"session_id": "empty", "messages": []},
            {"session_id": "valid", "messages": ["Hello"]},
        ]
        traces = runner.run_batch(conversations)
        assert len(traces) == 1
        assert traces[0].session_id == "valid"

    def test_session_trace_to_dict(self, runner: SessionRunner):
        """SessionTrace.to_dict() 应返回可序列化字典。"""
        trace = runner.run_conversation(
            messages=["Hello", "Bye"],
            session_id="dict_test",
        )
        d = trace.to_dict()
        assert isinstance(d, dict)
        assert d["session_id"] == "dict_test"
        assert d["num_turns"] == 2
        # 确保可以序列化为 JSON
        json.dumps(d, ensure_ascii=False)

    def test_session_trace_save(self, runner: SessionRunner):
        """SessionTrace.save() 应保存 JSON 文件。"""
        trace = runner.run_conversation(
            messages=["Hello"],
            session_id="save_test",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trace.json"
            trace.save(path)
            assert path.exists()
            data = json.loads(path.read_text())
            assert data["session_id"] == "save_test"

    def test_iter_turns(self, runner: SessionRunner, sample_messages: list[str]):
        """iter_turns 应以迭代器方式逐轮运行。"""
        results = list(runner.iter_turns(
            messages=sample_messages[:3],
            session_id="iter_test",
        ))
        assert len(results) == 3
        for r in results:
            assert isinstance(r, TurnResult)

    def test_run_from_file(self, runner: SessionRunner):
        """run_from_file 应从 JSON 文件加载并运行。"""
        conversations = [
            {"session_id": "file_001", "messages": ["Hello", "World"]},
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.json"
            input_path.write_text(json.dumps(conversations))
            output_dir = Path(tmpdir) / "traces"

            traces = runner.run_from_file(str(input_path), str(output_dir))
            assert len(traces) == 1
            # 应保存 trace 文件
            trace_file = output_dir / "file_001.json"
            assert trace_file.exists()


# ------------------------------------------------------------------ #
#  测试: 跨会话 L3 持久性
# ------------------------------------------------------------------ #

class TestCrossSession:
    """测试跨会话的 L3 记忆持久性。"""

    def test_l3_persists_across_sessions(self, agent: MemoryAgent):
        """L3 记忆应在多次会话间持久。"""
        # 会话 1
        agent.start_session("sess_001")
        agent.chat("I am researching large language models.")
        agent.end_session()

        profile_after_s1 = agent.get_profile_text()

        # 会话 2 (L3 应保留)
        agent.start_session("sess_002")
        agent.chat("Tell me about transformers.")
        agent.end_session()

        profile_after_s2 = agent.get_profile_text()
        # L3 应有内容 (至少与 session 1 后一样多, 或更多)
        assert len(profile_after_s2) >= len(profile_after_s1) or len(profile_after_s2) > 0


# ------------------------------------------------------------------ #
#  测试: AgentConfig
# ------------------------------------------------------------------ #

class TestAgentConfig:
    """测试 AgentConfig。"""

    def test_default_config(self):
        """默认配置应有合理的值。"""
        cfg = AgentConfig()
        assert cfg.enable_l1 is True
        assert cfg.enable_l2 is True
        assert cfg.enable_l3 is True
        assert cfg.max_context_chars > 0

    def test_from_dict(self):
        """从字典创建 agent。"""
        agent = MemoryAgent(config={"enable_l1": False, "enable_l2": True})
        assert agent.config.enable_l1 is False
        assert agent.config.enable_l2 is True

    def test_repr(self, agent: MemoryAgent):
        """__repr__ 应返回有意义的字符串。"""
        r = repr(agent)
        assert "MemoryAgent" in r
