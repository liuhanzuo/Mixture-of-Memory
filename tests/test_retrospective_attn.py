"""测试 RetrospectiveGather 模块。"""

import pytest
import torch

from src.gather.retrospective_attn import RetrospectiveGather


class TestRetrospectiveGather:
    """RetrospectiveGather 测试。"""

    @pytest.fixture
    def gather(self):
        return RetrospectiveGather(hidden_dim=64, gather_dim=32)

    def test_output_shape(self, gather):
        """输出形状应为 [B, K, hidden_dim]（经 out_proj 投影回 hidden_dim）。"""
        block_hidden = torch.randn(2, 16, 64)   # [B, L, D]
        anchor_hidden = torch.randn(2, 4, 64)   # [B, K, D]

        gathered, attn_weights = gather(
            block_hidden=block_hidden,
            anchor_hidden=anchor_hidden,
        )

        assert gathered.shape == (2, 4, 64), (
            f"期望 (2, 4, 64)，得到 {gathered.shape}"
        )

    def test_attention_weights_shape(self, gather):
        """注意力权重形状应为 [B, K, L]。"""
        block_hidden = torch.randn(3, 20, 64)
        anchor_hidden = torch.randn(3, 5, 64)

        _, attn_weights = gather(
            block_hidden=block_hidden,
            anchor_hidden=anchor_hidden,
        )

        assert attn_weights.shape == (3, 5, 20), (
            f"期望 (3, 5, 20)，得到 {attn_weights.shape}"
        )

    def test_attention_weights_sum_to_one(self, gather):
        """注意力权重在 L 维度上应和为 1。"""
        block_hidden = torch.randn(2, 10, 64)
        anchor_hidden = torch.randn(2, 3, 64)

        _, attn_weights = gather(
            block_hidden=block_hidden,
            anchor_hidden=anchor_hidden,
        )

        weight_sums = attn_weights.sum(dim=-1)  # [B, K]
        assert torch.allclose(weight_sums, torch.ones_like(weight_sums), atol=1e-5), (
            f"注意力权重和应为 1，得到 {weight_sums}"
        )

    def test_single_anchor(self, gather):
        """只有一个 anchor 时也应正常工作。"""
        block_hidden = torch.randn(1, 32, 64)
        anchor_hidden = torch.randn(1, 1, 64)

        gathered, attn_weights = gather(
            block_hidden=block_hidden,
            anchor_hidden=anchor_hidden,
        )

        assert gathered.shape == (1, 1, 64)
        assert attn_weights.shape == (1, 1, 32)

    def test_gradient_flow(self, gather):
        """梯度应能正确回传。"""
        block_hidden = torch.randn(2, 8, 64, requires_grad=True)
        anchor_hidden = torch.randn(2, 2, 64, requires_grad=True)

        gathered, _ = gather(
            block_hidden=block_hidden,
            anchor_hidden=anchor_hidden,
        )

        loss = gathered.sum()
        loss.backward()

        assert block_hidden.grad is not None, "block_hidden 应有梯度"
        assert anchor_hidden.grad is not None, "anchor_hidden 应有梯度"

    def test_different_gather_dims(self):
        """不同 gather_dim 配置应正常工作（输出始终是 hidden_dim）。"""
        for gd in [16, 32, 64, 128]:
            gather = RetrospectiveGather(hidden_dim=64, gather_dim=gd)
            block_hidden = torch.randn(1, 10, 64)
            anchor_hidden = torch.randn(1, 2, 64)

            gathered, _ = gather(
                block_hidden=block_hidden,
                anchor_hidden=anchor_hidden,
            )
            # out_proj 将 gather_dim 映射回 hidden_dim
            assert gathered.shape == (1, 2, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
