"""测试 BlockEvaluator 和 AnchorSelector。"""

import pytest
import torch

from src.anchor.evaluator import BlockEvaluator
from src.anchor.selector import AnchorSelector


class TestBlockEvaluator:
    """BlockEvaluator 测试。"""

    @pytest.fixture(params=["mlp", "transformer"])
    def evaluator(self, request):
        """创建不同类型的 evaluator。"""
        return BlockEvaluator(
            hidden_dim=64,
            scorer_type=request.param,
            num_layers=2,
        )

    def test_output_shape(self, evaluator):
        """输出形状应为 [B, L]。"""
        hidden = torch.randn(2, 32, 64)
        scores = evaluator(hidden)
        assert scores.shape == (2, 32), f"期望 (2, 32)，得到 {scores.shape}"

    def test_batch_independence(self, evaluator):
        """不同 batch 元素的分数应独立计算。"""
        h1 = torch.randn(1, 16, 64)
        h2 = torch.randn(1, 16, 64)
        h_cat = torch.cat([h1, h2], dim=0)

        scores_cat = evaluator(h_cat)
        scores_1 = evaluator(h1)
        scores_2 = evaluator(h2)

        assert torch.allclose(scores_cat[0], scores_1[0], atol=1e-5)
        assert torch.allclose(scores_cat[1], scores_2[0], atol=1e-5)

    def test_different_block_lengths(self, evaluator):
        """应支持不同的 block 长度。"""
        for L in [8, 16, 32, 64, 128]:
            hidden = torch.randn(1, L, 64)
            scores = evaluator(hidden)
            assert scores.shape == (1, L)


class TestAnchorSelector:
    """AnchorSelector 测试。"""

    def test_top_k_selection(self):
        """应选出 top-k 个分数最高的位置。"""
        selector = AnchorSelector(top_k=3)
        scores = torch.tensor([[1.0, 5.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0]])
        hidden = torch.randn(1, 8, 32)

        result = selector(scores, hidden)

        # top-3 位置应是 3 (8.0), 5 (7.0), 7 (6.0)
        expected_indices = {3, 5, 7}
        actual_indices = set(result.indices[0].tolist())
        assert actual_indices == expected_indices, (
            f"期望位置 {expected_indices}，得到 {actual_indices}"
        )

    def test_output_shapes(self):
        """输出形状应正确。"""
        selector = AnchorSelector(top_k=4)
        scores = torch.randn(3, 16)
        hidden = torch.randn(3, 16, 64)

        result = selector(scores, hidden)
        assert result.indices.shape == (3, 4)
        assert result.hidden_states.shape == (3, 4, 64)

    def test_top_k_exceeds_length(self):
        """当 top_k > L 时应优雅处理。"""
        selector = AnchorSelector(top_k=10)
        scores = torch.randn(2, 4)
        hidden = torch.randn(2, 4, 32)

        result = selector(scores, hidden)
        # 应返回所有位置（最多 4 个）
        assert result.indices.shape[1] <= 4

    def test_deterministic_eval(self):
        """eval 模式下选择应确定性。"""
        selector = AnchorSelector(top_k=3)
        selector.eval()

        scores = torch.randn(2, 16)
        hidden = torch.randn(2, 16, 32)

        result1 = selector(scores, hidden)
        result2 = selector(scores, hidden)

        assert torch.equal(result1.indices, result2.indices)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
