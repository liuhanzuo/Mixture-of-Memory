"""
MemoryScheduler 的单元测试。

覆盖:
- 初始化与配置
- init_state / session 管理
- on_turn_end (L2 异步聚合)
- on_session_end (L3 异步总结)
- retrieve_l2 / retrieve_l3
- push_message 与自动 chunk 聚合
- MoMState / MoMStats
"""

import pytest
import torch

from src.memory.scheduler import MemoryScheduler, SchedulerConfig
from src.memory.state import MoMState, MoMStats


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def full_config() -> SchedulerConfig:
    """启用全部三层的配置。"""
    return SchedulerConfig(
        enable_l1=True,
        enable_l2=True,
        enable_l3=True,
        l1_d_key=16,
        l1_d_value=16,
        l1_num_heads=2,
        l1_decay=0.9,
        l2_max_objects=50,
        l2_similarity_threshold=0.5,
        l2_chunk_size=3,
        l3_max_entries=20,
        l3_conflict_threshold=0.8,
        l1_gate_d_model=32,
        l1_gate_hidden_dim=32,
    )


@pytest.fixture
def l2_only_config() -> SchedulerConfig:
    """仅启用 L2 的配置。"""
    return SchedulerConfig(
        enable_l1=False,
        enable_l2=True,
        enable_l3=False,
        l2_max_objects=50,
        l2_chunk_size=2,
    )


@pytest.fixture
def l2_l3_config() -> SchedulerConfig:
    """启用 L2 + L3 的配置 (不启用 L1)。"""
    return SchedulerConfig(
        enable_l1=False,
        enable_l2=True,
        enable_l3=True,
        l2_max_objects=50,
        l2_chunk_size=3,
        l3_max_entries=20,
    )


# ------------------------------------------------------------------ #
#  测试: MoMStats
# ------------------------------------------------------------------ #

class TestMoMStats:
    """测试运行时统计。"""

    def test_stats_initial(self):
        """初始统计应为零。"""
        stats = MoMStats()
        d = stats.to_dict()
        for key, val in d.items():
            assert val == 0, f"{key} should be 0 initially"

    def test_stats_reset(self):
        """reset 应将所有计数器归零。"""
        stats = MoMStats(l1_write_count=10, l2_aggregate_count=5)
        stats.reset()
        assert stats.l1_write_count == 0
        assert stats.l2_aggregate_count == 0

    def test_stats_to_dict(self):
        """to_dict 应返回所有字段。"""
        stats = MoMStats(l1_write_count=3, l2_retrieve_count=7)
        d = stats.to_dict()
        assert d["l1_write_count"] == 3
        assert d["l2_retrieve_count"] == 7
        assert "total_turns" in d
        assert "total_sessions" in d


# ------------------------------------------------------------------ #
#  测试: SchedulerConfig
# ------------------------------------------------------------------ #

class TestSchedulerConfig:
    """测试 SchedulerConfig 数据类。"""

    def test_default_config(self):
        """默认配置应有合理值。"""
        cfg = SchedulerConfig()
        assert cfg.enable_l1 is True
        assert cfg.enable_l2 is True
        assert cfg.enable_l3 is True
        assert cfg.l1_decay > 0 and cfg.l1_decay <= 1.0
        assert cfg.l2_max_objects > 0
        assert cfg.l3_max_entries > 0

    def test_custom_config(self, full_config: SchedulerConfig):
        """自定义配置应正确传递。"""
        assert full_config.l1_d_key == 16
        assert full_config.l2_chunk_size == 3
        assert full_config.l3_max_entries == 20


# ------------------------------------------------------------------ #
#  测试: 初始化
# ------------------------------------------------------------------ #

class TestSchedulerInit:
    """测试 Scheduler 初始化。"""

    def test_init_full(self, full_config: SchedulerConfig):
        """全部启用时应创建所有组件。"""
        scheduler = MemoryScheduler(full_config)
        assert scheduler._l1 is not None
        assert scheduler._l2_aggregator is not None
        assert scheduler._l2_store is not None
        assert scheduler._l3_summarizer is not None
        assert scheduler._l3_store is not None

    def test_init_l2_only(self, l2_only_config: SchedulerConfig):
        """仅启用 L2 时, L1 和 L3 应为 None。"""
        scheduler = MemoryScheduler(l2_only_config)
        assert scheduler._l1 is None
        assert scheduler._l2_aggregator is not None
        assert scheduler._l3_summarizer is None

    def test_init_none_config(self):
        """None 配置应使用默认值。"""
        scheduler = MemoryScheduler(None)
        assert scheduler.config.enable_l1 is True
        assert scheduler.config.enable_l2 is True

    def test_init_dict_config(self):
        """从字典初始化应正确。"""
        scheduler = MemoryScheduler({"enable_l1": False, "enable_l2": True})
        assert scheduler._l1 is None
        assert scheduler._l2_aggregator is not None


# ------------------------------------------------------------------ #
#  测试: 会话管理
# ------------------------------------------------------------------ #

class TestSessionManagement:
    """测试 init_state / 会话生命周期。"""

    def test_init_state(self, l2_only_config: SchedulerConfig):
        """init_state 应设置 session_id。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="test_session")
        assert scheduler.state.session_id == "test_session"
        assert scheduler.stats.total_sessions >= 1

    def test_init_state_auto_id(self, l2_only_config: SchedulerConfig):
        """不传 session_id 应自动生成。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state()
        assert scheduler.state.session_id != ""
        assert len(scheduler.state.session_id) > 0

    def test_init_state_soft_reset(self, l2_only_config: SchedulerConfig):
        """init_state 应执行 soft_reset。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="s1")

        # 推送一些消息
        scheduler.push_message({"role": "user", "content": "Hello"})

        # 重新 init_state
        scheduler.init_state(session_id="s2")
        assert scheduler.state.session_id == "s2"
        assert len(scheduler.state.message_buffer) == 0


# ------------------------------------------------------------------ #
#  测试: on_turn_end (L2 聚合)
# ------------------------------------------------------------------ #

class TestOnTurnEnd:
    """测试 turn 结束事件。"""

    def test_on_turn_end_with_messages(self, l2_only_config: SchedulerConfig):
        """显式传入消息时应触发 L2 聚合。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="turn_test")

        messages = [
            {"role": "user", "content": "I prefer using PyTorch for deep learning.", "turn_id": "t1"},
            {"role": "assistant", "content": "PyTorch is a great choice.", "turn_id": "t1"},
        ]
        scheduler.on_turn_end(messages=messages)
        assert scheduler.stats.total_turns == 1
        assert scheduler.stats.l2_aggregate_count >= 1

    def test_turn_counter_increments(self, l2_only_config: SchedulerConfig):
        """on_turn_end 应递增 turn 计数。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="turn_count_test")

        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "Test message.", "turn_id": "t1"},
        ])
        assert scheduler.state.current_turn == 1

        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "Another message.", "turn_id": "t2"},
        ])
        assert scheduler.state.current_turn == 2

    def test_on_turn_end_empty(self, l2_only_config: SchedulerConfig):
        """空消息时不应出错。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="empty_turn")
        scheduler.on_turn_end(messages=[])
        # 不应抛出异常，但也不触发聚合
        assert scheduler.state.current_turn == 1

    def test_on_turn_end_no_l2(self):
        """L2 未启用时不应出错。"""
        scheduler = MemoryScheduler(SchedulerConfig(enable_l2=False))
        scheduler.init_state(session_id="no_l2")
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "Test."},
        ])
        # 不应抛出异常
        assert scheduler.state.current_turn == 1


# ------------------------------------------------------------------ #
#  测试: on_session_end (L3 总结)
# ------------------------------------------------------------------ #

class TestOnSessionEnd:
    """测试 session 结束事件。"""

    def test_on_session_end_triggers_l3(self, l2_l3_config: SchedulerConfig):
        """session 结束时应触发 L3 总结。"""
        scheduler = MemoryScheduler(l2_l3_config)
        scheduler.init_state(session_id="session_end_test")

        # 先通过 turn_end 填充 L2
        messages = [
            {"role": "user", "content": "I prefer structured technical explanations.", "turn_id": "t1"},
            {"role": "assistant", "content": "Noted.", "turn_id": "t1"},
        ]
        scheduler.on_turn_end(messages=messages)

        # 触发 session end
        scheduler.on_session_end()
        assert scheduler.stats.l3_summarize_count >= 1

    def test_on_session_end_empty_l2(self, l2_l3_config: SchedulerConfig):
        """L2 为空时 session end 不应出错。"""
        scheduler = MemoryScheduler(l2_l3_config)
        scheduler.init_state(session_id="empty_session")
        scheduler.on_session_end()  # 不应抛出异常

    def test_on_session_end_no_l3(self, l2_only_config: SchedulerConfig):
        """L3 未启用时 session end 不应出错。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="no_l3")
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "Hello."},
        ])
        scheduler.on_session_end()  # 不应抛出异常


# ------------------------------------------------------------------ #
#  测试: 检索接口
# ------------------------------------------------------------------ #

class TestRetrieval:
    """测试检索接口。"""

    def test_retrieve_l2(self, l2_only_config: SchedulerConfig):
        """retrieve_l2 应返回结果列表。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="retrieve_test")

        # 填充 L2
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "I prefer using Python.", "turn_id": "t1"},
            {"role": "assistant", "content": "Python is good.", "turn_id": "t1"},
        ])
        results = scheduler.retrieve_l2(query="Python", top_k=3)
        # 不保证一定有结果，但不应出错
        assert isinstance(results, list)
        assert scheduler.stats.l2_retrieve_count >= 1

    def test_retrieve_l3(self, l2_l3_config: SchedulerConfig):
        """retrieve_l3 应返回结果列表。"""
        scheduler = MemoryScheduler(l2_l3_config)
        scheduler.init_state(session_id="l3_retrieve")

        # 填充 L2 → L3
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "I am researching large language models.", "turn_id": "t1"},
        ])
        scheduler.on_session_end()

        results = scheduler.retrieve_l3(query="language models", top_k=5)
        assert isinstance(results, list)

    def test_retrieve_l2_disabled(self):
        """L2 未启用时应返回空列表。"""
        scheduler = MemoryScheduler(SchedulerConfig(enable_l2=False))
        results = scheduler.retrieve_l2(query="test")
        assert results == []

    def test_retrieve_l3_disabled(self):
        """L3 未启用时应返回空列表。"""
        scheduler = MemoryScheduler(SchedulerConfig(enable_l3=False))
        results = scheduler.retrieve_l3(query="test")
        assert results == []


# ------------------------------------------------------------------ #
#  测试: push_message 与自动 chunk 聚合
# ------------------------------------------------------------------ #

class TestPushMessage:
    """测试消息推送与自动聚合。"""

    def test_push_accumulates_buffer(self, l2_only_config: SchedulerConfig):
        """push_message 应累积到缓冲区。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="buffer_test")

        scheduler.push_message({"role": "user", "content": "msg1"})
        # chunk_size=2, 一条消息还不触发
        assert len(scheduler.state.message_buffer) == 1

    def test_auto_chunk_on_threshold(self, l2_only_config: SchedulerConfig):
        """达到 chunk_size 阈值时应自动触发聚合。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="auto_chunk_test")

        # chunk_size=2
        scheduler.push_message({"role": "user", "content": "Hello"})
        initial_count = scheduler.stats.l2_aggregate_count
        scheduler.push_message({"role": "user", "content": "World"})
        # 第二条消息达到阈值，应触发聚合
        assert scheduler.stats.l2_aggregate_count > initial_count

    def test_push_message_no_l2(self):
        """L2 未启用时 push_message 不应出错。"""
        scheduler = MemoryScheduler(SchedulerConfig(enable_l2=False))
        scheduler.init_state(session_id="no_l2_push")
        scheduler.push_message({"role": "user", "content": "test"})
        # 消息应仍在缓冲区
        assert len(scheduler.state.message_buffer) == 1


# ------------------------------------------------------------------ #
#  测试: MoMState 快照与重置
# ------------------------------------------------------------------ #

class TestMoMState:
    """测试 MoMState 功能。"""

    def test_snapshot(self, l2_only_config: SchedulerConfig):
        """snapshot 应返回可序列化的字典。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="snap_test")
        snap = scheduler.state.snapshot()
        assert "session_id" in snap
        assert "current_turn" in snap
        assert "stats" in snap
        assert snap["session_id"] == "snap_test"

    def test_soft_reset(self, l2_l3_config: SchedulerConfig):
        """soft_reset 应保留 L3, 清空 L1/L2。"""
        scheduler = MemoryScheduler(l2_l3_config)
        scheduler.init_state(session_id="soft_reset_test")

        # 填充一些数据
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "I prefer Python.", "turn_id": "t1"},
        ])
        scheduler.on_session_end()

        l3_size_before = scheduler._l3_store.size() if scheduler._l3_store else 0

        scheduler.state.soft_reset()
        assert scheduler.state.current_turn == 0
        assert len(scheduler.state.message_buffer) == 0

        # L3 应保留
        l3_size_after = scheduler._l3_store.size() if scheduler._l3_store else 0
        assert l3_size_after == l3_size_before

    def test_full_reset(self, l2_only_config: SchedulerConfig):
        """reset 应清空所有状态。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="full_reset_test")
        scheduler.on_turn_end(messages=[
            {"role": "user", "content": "Test data.", "turn_id": "t1"},
        ])
        scheduler.state.reset()
        assert scheduler.state.current_turn == 0
        assert scheduler.state.session_id == ""

    def test_push_and_flush_buffer(self, l2_only_config: SchedulerConfig):
        """push_message + flush_buffer 应正确工作。"""
        scheduler = MemoryScheduler(l2_only_config)
        scheduler.init_state(session_id="flush_test")

        # 只推一条 (不达 chunk_size)
        scheduler.state.push_message({"role": "user", "content": "test"})
        assert len(scheduler.state.message_buffer) == 1

        flushed = scheduler.state.flush_buffer()
        assert len(flushed) == 1
        assert len(scheduler.state.message_buffer) == 0


# ------------------------------------------------------------------ #
#  测试: repr
# ------------------------------------------------------------------ #

class TestSchedulerRepr:
    """测试 repr。"""

    def test_repr(self, l2_only_config: SchedulerConfig):
        """repr 应返回有意义的字符串。"""
        scheduler = MemoryScheduler(l2_only_config)
        r = repr(scheduler)
        assert "MemoryScheduler" in r
