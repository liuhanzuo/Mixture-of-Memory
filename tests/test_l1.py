"""
L1 关联矩阵记忆的单元测试。

覆盖:
- 初始化与配置
- reset
- update_step / update_chunk
- read
- read_and_gate
- forward (读写联合)
- 衰减特性
- get_stats
"""

import pytest
import torch

from src.memory.l1.assoc_memory import AssociativeMemoryL1, L1Config
from src.memory.l1.writer import L1Writer
from src.memory.l1.reader import L1Reader
from src.memory.l1.gating import L1Gate


# ------------------------------------------------------------------ #
#  Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture
def default_config() -> L1Config:
    """默认的 L1 测试配置。"""
    return L1Config(
        d_key=16,
        d_value=16,
        n_heads=2,
        decay=0.9,
        write_strength=1.0,
        write_interval=1,
        write_gate_type="sigmoid",
        use_output_gate=True,
        dtype="float32",
    )


@pytest.fixture
def l1(default_config: L1Config) -> AssociativeMemoryL1:
    """构建一个 L1 实例并设置 hidden_dim。"""
    mem = AssociativeMemoryL1(default_config)
    mem.set_hidden_dim(32)
    return mem


@pytest.fixture
def hidden_states() -> torch.Tensor:
    """模拟的 backbone 隐藏状态 (batch=2, seq_len=4, hidden_dim=32)。"""
    return torch.randn(2, 4, 32)


# ------------------------------------------------------------------ #
#  测试: 初始化
# ------------------------------------------------------------------ #

class TestL1Init:
    """测试 L1 初始化和配置。"""

    def test_init_default(self, default_config: L1Config):
        """默认初始化应创建零记忆矩阵。"""
        mem = AssociativeMemoryL1(default_config)
        assert mem.memory.shape == (2, 16, 16)
        assert mem.memory.sum().item() == 0.0
        assert mem.step_count == 0

    def test_set_hidden_dim(self, default_config: L1Config):
        """设置 hidden_dim 后应初始化投影层。"""
        mem = AssociativeMemoryL1(default_config)
        mem.set_hidden_dim(64)
        assert mem.proj_k is not None
        assert mem.proj_v is not None
        assert mem.proj_q is not None
        # 输入维度应为 64，输出维度应为 n_heads * d_key = 2 * 16 = 32
        assert mem.proj_k.in_features == 64
        assert mem.proj_k.out_features == 2 * 16

    def test_set_hidden_dim_idempotent(self, l1: AssociativeMemoryL1):
        """重复调用 set_hidden_dim 相同值不应重新创建层。"""
        proj_k_ref = l1.proj_k
        l1.set_hidden_dim(32)
        assert l1.proj_k is proj_k_ref  # 同一对象

    def test_config_torch_dtype(self):
        """torch_dtype 应正确映射。"""
        cfg = L1Config(dtype="bfloat16")
        assert cfg.torch_dtype() == torch.bfloat16


# ------------------------------------------------------------------ #
#  测试: reset
# ------------------------------------------------------------------ #

class TestL1Reset:
    """测试 L1 重置功能。"""

    def test_reset_zeros_memory(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """reset 后记忆应为全零。"""
        # 先执行一次更新使记忆非零
        l1.update_step(hidden_states)
        assert l1.memory.abs().sum().item() > 0.0

        l1.reset()
        assert l1.memory.sum().item() == 0.0
        assert l1.step_count == 0


# ------------------------------------------------------------------ #
#  测试: 写入操作
# ------------------------------------------------------------------ #

class TestL1Write:
    """测试 L1 写入操作。"""

    def test_update_step_changes_memory(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """update_step 应改变记忆状态。"""
        old_norm = l1.memory.norm().item()
        l1.update_step(hidden_states)
        new_norm = l1.memory.norm().item()
        assert new_norm > old_norm  # 记忆范数应增加

    def test_update_step_increments_count(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """update_step 应递增步数计数器。"""
        l1.update_step(hidden_states)
        assert l1.step_count == 1
        l1.update_step(hidden_states)
        assert l1.step_count == 2

    def test_update_chunk(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """update_chunk 应一次性写入整个 chunk。"""
        l1.update_chunk(hidden_states)
        # step_count 应等于 seq_len
        assert l1.step_count == hidden_states.shape[1]
        assert l1.memory.abs().sum().item() > 0.0

    def test_write_interval(self, default_config: L1Config, hidden_states: torch.Tensor):
        """write_interval > 1 时应跳过部分写入。"""
        default_config.write_interval = 3
        mem = AssociativeMemoryL1(default_config)
        mem.set_hidden_dim(32)

        # step 1, 2 不写入 (step_count=1,2 不是 3 的倍数)
        mem.update_step(hidden_states)
        norm_after_1 = mem.memory.norm().item()
        mem.update_step(hidden_states)
        norm_after_2 = mem.memory.norm().item()

        # step 3 写入
        mem.update_step(hidden_states)
        norm_after_3 = mem.memory.norm().item()

        # 前两步不写入，记忆范数不变
        assert norm_after_1 == 0.0
        assert norm_after_2 == 0.0
        # 第三步写入，记忆范数增加
        assert norm_after_3 > 0.0


# ------------------------------------------------------------------ #
#  测试: 读取操作
# ------------------------------------------------------------------ #

class TestL1Read:
    """测试 L1 读取操作。"""

    def test_read_empty_memory(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """空记忆下读取应返回全零。"""
        readout = l1.read(hidden_states)
        assert readout.shape == (2, 4, l1.n_heads * l1.d_value)
        assert readout.abs().sum().item() == 0.0

    def test_read_nonempty_memory(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """写入后读取应返回非零结果。"""
        l1.update_step(hidden_states)
        readout = l1.read(hidden_states)
        assert readout.abs().sum().item() > 0.0

    def test_read_output_shape(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """read 输出 shape 应为 (batch, seq_len, n_heads * d_value)。"""
        readout = l1.read(hidden_states)
        B, S, _ = hidden_states.shape
        assert readout.shape == (B, S, l1.n_heads * l1.d_value)


# ------------------------------------------------------------------ #
#  测试: 门控
# ------------------------------------------------------------------ #

class TestL1Gate:
    """测试 L1 门控融合。"""

    def test_read_and_gate_shape(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """read_and_gate 输出应与输入 hidden_states 同 shape。"""
        output = l1.read_and_gate(hidden_states)
        assert output.shape == hidden_states.shape

    def test_gate_disabled(self, default_config: L1Config, hidden_states: torch.Tensor):
        """use_output_gate=False 时, read_and_gate 应返回原始 hidden_states。"""
        default_config.use_output_gate = False
        mem = AssociativeMemoryL1(default_config)
        mem.set_hidden_dim(32)
        output = mem.read_and_gate(hidden_states)
        # 空记忆 + 无门控 → 应返回原 hidden_states
        assert torch.allclose(output, hidden_states)


# ------------------------------------------------------------------ #
#  测试: forward (完整前向)
# ------------------------------------------------------------------ #

class TestL1Forward:
    """测试完整的 L1 forward。"""

    def test_forward_shape(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """forward 输出应与输入同 shape。"""
        output = l1.forward(hidden_states, update=True)
        assert output.shape == hidden_states.shape

    def test_forward_with_update(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """forward(update=True) 应更新记忆。"""
        l1.forward(hidden_states, update=True)
        assert l1.step_count == 1
        assert l1.memory.abs().sum().item() > 0.0

    def test_forward_without_update(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """forward(update=False) 不应更新记忆。"""
        l1.forward(hidden_states, update=False)
        assert l1.step_count == 0
        assert l1.memory.sum().item() == 0.0


# ------------------------------------------------------------------ #
#  测试: 衰减特性
# ------------------------------------------------------------------ #

class TestL1Decay:
    """测试衰减行为。"""

    def test_decay_reduces_memory(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """多次衰减写入后, 旧信息应逐渐消退。"""
        # 写入一次
        l1.update_step(hidden_states)
        norm_after_write = l1.memory.norm().item()

        # 用零写入模拟只衰减 (零 KV → delta=0, 只有衰减)
        zero_hidden = torch.zeros_like(hidden_states)
        for _ in range(10):
            l1.update_step(zero_hidden)

        norm_after_decay = l1.memory.norm().item()
        # 衰减后范数应小于初始写入后的范数
        assert norm_after_decay < norm_after_write


# ------------------------------------------------------------------ #
#  测试: 统计信息
# ------------------------------------------------------------------ #

class TestL1Stats:
    """测试统计信息接口。"""

    def test_get_stats(self, l1: AssociativeMemoryL1, hidden_states: torch.Tensor):
        """get_stats 应返回合法的统计字典。"""
        l1.update_step(hidden_states)
        stats = l1.get_stats()
        assert "l1_memory_norm" in stats
        assert "l1_memory_mean" in stats
        assert "l1_memory_std" in stats
        assert "l1_memory_max" in stats
        assert "l1_step_count" in stats
        assert stats["l1_step_count"] == 1.0
        assert stats["l1_memory_norm"] > 0.0


# ------------------------------------------------------------------ #
#  测试: Writer / Reader 独立测试
# ------------------------------------------------------------------ #

class TestL1Components:
    """测试 L1Writer 和 L1Reader 独立使用。"""

    def test_writer_write(self, default_config: L1Config):
        """Writer.write 应正确执行衰减+外积写入。"""
        writer = L1Writer(default_config)
        memory = torch.zeros(2, 16, 16)
        keys = torch.randn(1, 3, 2, 16)    # (B=1, S=3, H=2, D_k=16)
        values = torch.randn(1, 3, 2, 16)  # (B=1, S=3, H=2, D_v=16)

        updated = writer.write(memory, keys, values)
        assert updated.shape == (2, 16, 16)
        assert updated.abs().sum().item() > 0.0

    def test_writer_single_step(self, default_config: L1Config):
        """Writer.write_single_step 应正确执行单步写入。"""
        writer = L1Writer(default_config)
        memory = torch.zeros(2, 16, 16)
        key = torch.randn(2, 16)    # (H=2, D_k=16)
        value = torch.randn(2, 16)  # (H=2, D_v=16)

        updated = writer.write_single_step(memory, key, value)
        assert updated.shape == (2, 16, 16)
        assert updated.abs().sum().item() > 0.0

    def test_reader_read(self, default_config: L1Config):
        """Reader.read 应正确读取。"""
        reader = L1Reader(default_config)
        memory = torch.randn(2, 16, 16)
        queries = torch.randn(1, 3, 2, 16)  # (B=1, S=3, H=2, D_k=16)

        readout = reader.read(memory, queries)
        # 输出 shape 应为 (B, S, H * D_v)
        assert readout.shape == (1, 3, 2 * 16)
