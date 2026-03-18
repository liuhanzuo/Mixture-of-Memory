"""测试 MOM 记忆更新的形状和数值稳定性。"""

import pytest
import torch
import torch.nn as nn

from src.memory.mom import MixtureOfMemory, LatentMatrixMemory
from src.memory.update import MemoryWriter


class TestLatentMatrixMemory:
    """LatentMatrixMemory 基础测试。"""

    def test_init_shape(self):
        """初始化后 memory state 应为零矩阵。"""
        mem = LatentMatrixMemory(key_dim=32, value_dim=32)
        mem.reset(batch_size=2, device=torch.device("cpu"))
        state = mem.state
        assert state.shape == (2, 32, 32), f"期望 (2, 32, 32)，得到 {state.shape}"
        assert (state == 0).all(), "初始 state 应为零"

    def test_update_shape(self):
        """更新后 state 形状不变。"""
        mem = LatentMatrixMemory(key_dim=32, value_dim=32)
        mem.reset(batch_size=4, device=torch.device("cpu"))

        key = torch.randn(4, 32)
        value = torch.randn(4, 32)
        alpha = torch.ones(4, 1) * 0.5
        lam = torch.ones(4, 1) * 0.9

        mem.update(key=key, value=value, alpha=alpha, lam=lam)
        state = mem.state
        assert state.shape == (4, 32, 32)

    def test_update_nonzero(self):
        """更新后 state 不再全零。"""
        mem = LatentMatrixMemory(key_dim=16, value_dim=16)
        mem.reset(batch_size=2, device=torch.device("cpu"))

        key = torch.randn(2, 16)
        value = torch.randn(2, 16)
        alpha = torch.ones(2, 1)
        lam = torch.ones(2, 1) * 0.5

        mem.update(key=key, value=value, alpha=alpha, lam=lam)
        assert not (mem.state == 0).all(), "更新后 state 不应全零"

    def test_retention_decay(self):
        """高保留率应保留更多旧信息。"""
        mem = LatentMatrixMemory(key_dim=8, value_dim=8)
        mem.reset(batch_size=1, device=torch.device("cpu"))

        # 第一次写入
        k1 = torch.ones(1, 8)
        v1 = torch.ones(1, 8)
        mem.update(key=k1, value=v1, alpha=torch.ones(1, 1), lam=torch.ones(1, 1))
        state_after_first = mem.state.clone()

        # 第二次写入，高保留率
        k2 = torch.zeros(1, 8)
        v2 = torch.zeros(1, 8)
        mem.update(key=k2, value=v2, alpha=torch.ones(1, 1), lam=torch.ones(1, 1) * 0.99)

        # state 应该仍保留大部分第一次写入的信息
        retention_ratio = mem.state.norm() / state_after_first.norm()
        assert retention_ratio > 0.9, f"高保留率下信息应被保留，但比率为 {retention_ratio:.4f}"


class TestMixtureOfMemory:
    """MixtureOfMemory 整体测试。"""

    def test_init(self):
        """MOM 初始化应创建正确数量的记忆。"""
        mom = MixtureOfMemory(num_memories=3, key_dim=32, value_dim=32)
        assert len(mom.memories) == 3

    def test_reset(self):
        """reset 应将所有记忆清零。"""
        mom = MixtureOfMemory(num_memories=3, key_dim=16, value_dim=16)
        mom.reset(batch_size=2, device=torch.device("cpu"))
        for mem in mom.memories:
            assert (mem.state == 0).all()

    def test_update_all(self):
        """通过路由更新所有记忆。"""
        mom = MixtureOfMemory(num_memories=3, key_dim=16, value_dim=16)
        mom.reset(batch_size=2, device=torch.device("cpu"))

        key = torch.randn(2, 16)
        value = torch.randn(2, 16)
        alpha = torch.ones(2, 1) * 0.8
        rho = torch.ones(2, 3) / 3.0  # 均匀路由
        lam = torch.ones(2, 3) * 0.9

        mom.update(key=key, value=value, alpha=alpha, rho=rho, lam=lam)

        # 所有记忆都应被更新
        for mem in mom.memories:
            assert not (mem.state == 0).all(), "均匀路由下所有记忆都应被更新"

    def test_memory_stats(self):
        """get_memory_stats 应返回合理的统计。"""
        mom = MixtureOfMemory(num_memories=3, key_dim=16, value_dim=16)
        mom.reset(batch_size=2, device=torch.device("cpu"))
        stats = mom.get_memory_stats()
        assert len(stats) == 3


class TestMemoryWriter:
    """MemoryWriter 测试。"""

    def test_output_shapes(self):
        """检查 writer 输出形状。"""
        writer = MemoryWriter(
            input_dim=64, key_dim=16, value_dim=16, num_memories=3
        )
        z = torch.randn(4, 64)
        decision = writer(z)

        assert decision.key.shape == (4, 16)
        assert decision.value.shape == (4, 16)
        assert decision.alpha.shape == (4, 1)
        assert decision.rho.shape == (4, 3)
        assert decision.lam.shape == (4, 3)

    def test_alpha_range(self):
        """alpha 应在 [0, 1]。"""
        writer = MemoryWriter(input_dim=32, key_dim=16, value_dim=16)
        z = torch.randn(8, 32)
        decision = writer(z)
        assert (decision.alpha >= 0).all() and (decision.alpha <= 1).all()

    def test_rho_is_distribution(self):
        """rho 应为概率分布（和为 1）。"""
        writer = MemoryWriter(input_dim=32, key_dim=16, value_dim=16, num_memories=3)
        z = torch.randn(4, 32)
        decision = writer(z)
        rho_sum = decision.rho.sum(dim=-1)
        assert torch.allclose(rho_sum, torch.ones_like(rho_sum), atol=1e-5)

    def test_lam_range(self):
        """lam 应在 [0, 1]。"""
        writer = MemoryWriter(input_dim=32, key_dim=16, value_dim=16)
        z = torch.randn(4, 32)
        decision = writer(z)
        assert (decision.lam >= 0).all() and (decision.lam <= 1).all()

    def test_decision_stats(self):
        """get_decision_stats 应返回合理的统计。"""
        writer = MemoryWriter(input_dim=32, key_dim=16, value_dim=16, num_memories=3)
        z = torch.randn(4, 32)
        decision = writer(z)
        stats = writer.get_decision_stats(decision)
        assert "alpha_mean" in stats
        assert "route_entropy" in stats


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
