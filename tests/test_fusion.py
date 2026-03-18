"""测试 FusionHead 模块。"""

import pytest
import torch

from src.fusion.fusion_head import FusionHead
from src.common.seed import set_seed


class TestFusionHead:
    """FusionHead 测试。"""

    @pytest.fixture
    def fusion(self):
        return FusionHead(
            hidden_dim=64,
            memory_dim=32,
            gate_init_bias=-2.0,
        )

    def test_output_shape(self, fusion):
        """输出形状应与 hidden_states 相同: [B, T, D]。"""
        hidden = torch.randn(2, 16, 64)
        readout = torch.randn(2, 16, 32)

        fused = fusion(hidden_states=hidden, memory_readout=readout)
        assert fused.shape == (2, 16, 64), f"期望 (2, 16, 64)，得到 {fused.shape}"

    def test_2d_input(self, fusion):
        """应支持 2D 输入 [B, D]。"""
        hidden = torch.randn(4, 64)
        readout = torch.randn(4, 32)

        fused = fusion(hidden_states=hidden, memory_readout=readout)
        assert fused.shape == (4, 64)

    def test_residual_connection(self, fusion):
        """零 readout 时，输出应接近原始 hidden（因为 gate 初始偏向 0）。"""
        hidden = torch.randn(2, 8, 64)
        readout = torch.zeros(2, 8, 32)

        fused = fusion(hidden_states=hidden, memory_readout=readout)

        # 零 readout 经过线性层后仍是零，所以 fused ≈ hidden
        diff = (fused - hidden).abs().max().item()
        assert diff < 1e-5, f"零 readout 时 fused 应等于 hidden，最大差异: {diff}"

    def test_gate_init_near_zero(self, fusion):
        """gate 初始值应接近 0（由 gate_init_bias=-2.0 控制）。"""
        hidden = torch.randn(1, 4, 64)
        readout = torch.randn(1, 4, 32)

        # gate = sigmoid(gate_logit)，gate_init_bias=-2.0 → sigmoid(-2) ≈ 0.12
        # 所以 memory 影响应较小
        fused = fusion(hidden_states=hidden, memory_readout=readout)

        # 融合后与原始 hidden 的差异应较小
        diff_norm = (fused - hidden).norm() / hidden.norm()
        assert diff_norm < 1.0, (
            f"gate 初始偏向 0 时，融合影响不应太大，但 diff_norm={diff_norm:.4f}"
        )

    def test_gradient_flow(self, fusion):
        """梯度应能回传到 fusion 参数和 readout。"""
        hidden = torch.randn(2, 8, 64)
        readout = torch.randn(2, 8, 32, requires_grad=True)

        fused = fusion(hidden_states=hidden, memory_readout=readout)
        loss = fused.sum()
        loss.backward()

        assert readout.grad is not None, "readout 应有梯度"
        for name, p in fusion.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"fusion.{name} 应有梯度"

    def test_deterministic_under_seed(self, fusion):
        """固定种子下输出应确定性。"""
        hidden = torch.randn(2, 8, 64)
        readout = torch.randn(2, 8, 32)

        fusion.eval()
        out1 = fusion(hidden_states=hidden, memory_readout=readout)
        out2 = fusion(hidden_states=hidden, memory_readout=readout)

        assert torch.equal(out1, out2), "eval 模式下输出应确定性"

    def test_different_memory_dims(self):
        """不同 memory_dim 配置应正常工作。"""
        for md in [16, 32, 64, 128]:
            fh = FusionHead(hidden_dim=64, memory_dim=md)
            hidden = torch.randn(1, 4, 64)
            readout = torch.randn(1, 4, md)
            fused = fh(hidden_states=hidden, memory_readout=readout)
            assert fused.shape == (1, 4, 64)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
