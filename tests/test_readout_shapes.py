"""测试 MemoryReadout 模块的输出形状。"""

import pytest
import torch

from src.memory.readout import MemoryReadout


class TestMemoryReadout:
    """MemoryReadout 测试。"""

    @pytest.fixture
    def readout(self):
        return MemoryReadout(
            hidden_dim=64,
            key_dim=32,
            value_dim=32,
            num_memories=3,
        )

    def test_single_step_shape(self, readout):
        """单步读取的输出形状应为 [B, value_dim]。"""
        h_t = torch.randn(4, 64)
        # 3 个记忆 state，每个 [B, D_k, D_v]
        memory_states = [torch.randn(4, 32, 32) for _ in range(3)]

        r_t = readout(h_t, memory_states)
        assert r_t.shape == (4, 32), f"期望 (4, 32)，得到 {r_t.shape}"

    def test_sequence_shape(self, readout):
        """序列读取的输出形状应为 [B, T, value_dim]。"""
        hidden_seq = torch.randn(2, 16, 64)  # [B, T, D]
        memory_states = [torch.randn(2, 32, 32) for _ in range(3)]

        r_seq = readout.forward_sequence(hidden_seq, memory_states)
        assert r_seq.shape == (2, 16, 32), f"期望 (2, 16, 32)，得到 {r_seq.shape}"

    def test_router_weights_sum_to_one(self, readout):
        """路由权重应和为 1。"""
        h_t = torch.randn(4, 64)
        memory_states = [torch.randn(4, 32, 32) for _ in range(3)]

        # 手动调用 read_router 检查
        gamma = torch.softmax(readout.read_router(h_t), dim=-1)
        gamma_sum = gamma.sum(dim=-1)
        assert torch.allclose(gamma_sum, torch.ones_like(gamma_sum), atol=1e-5)

    def test_zero_memory_output(self, readout):
        """零记忆应产生零（或近零）readout。"""
        h_t = torch.randn(2, 64)
        memory_states = [torch.zeros(2, 32, 32) for _ in range(3)]

        r_t = readout(h_t, memory_states)
        assert r_t.abs().max() < 1e-5, "零记忆应产生近零 readout"

    def test_different_num_memories(self):
        """不同数量的记忆应正常工作。"""
        for n_mem in [1, 2, 3, 5]:
            ro = MemoryReadout(
                hidden_dim=64, key_dim=16, value_dim=16, num_memories=n_mem,
            )
            h_t = torch.randn(2, 64)
            states = [torch.randn(2, 16, 16) for _ in range(n_mem)]
            r_t = ro(h_t, states)
            assert r_t.shape == (2, 16)

    def test_gradient_flow(self, readout):
        """梯度应能正确回传到 readout 参数。"""
        h_t = torch.randn(2, 64)
        memory_states = [torch.randn(2, 32, 32) for _ in range(3)]

        r_t = readout(h_t, memory_states)
        loss = r_t.sum()
        loss.backward()

        # 检查 query_proj 有梯度
        for p in readout.query_proj.parameters():
            if p.requires_grad:
                assert p.grad is not None, "readout 参数应有梯度"
                break


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
