"""Unit tests for L2Compressor (mem_space token-compressed KV memory).

Reference:
    src/memory/mem_space/l2_compressor.py
    docs/L2_DEEPSEEK_MLA_RESEARCH.md §5
    docs/L2_IMPLEMENTATION_PLAN_20260516.md §3
"""
from __future__ import annotations

import math

import pytest
import torch

from src.memory.mem_space.l2_compressor import L2Compressor


# ------------------------------------------------------------------ #
#  Fixtures / helpers
# ------------------------------------------------------------------ #

D_MODEL = 256       # small for fast tests; same arithmetic as 4096
N_HEADS = 8
D_HEAD = 32         # 8 * 32 = 256
D_C = 64
D_H_R = 16
G = 16


def make_compressor(**overrides):
    kw = dict(
        d_model=D_MODEL,
        n_heads=N_HEADS,
        d_head=D_HEAD,
        compress_ratio=G,
        d_c=D_C,
        d_h_rope=D_H_R,
        chunk_size=256,
        n_kv_heads=N_HEADS,
        init_scale=0.001,
    )
    kw.update(overrides)
    return L2Compressor(**kw)


# ------------------------------------------------------------------ #
#  Tests
# ------------------------------------------------------------------ #

class TestL2CompressorBasic:

    def test_output_shape_no_pad(self):
        comp = make_compressor()
        T = 256              # exact multiple of g=16 → 16 latents
        h = torch.randn(2, T, D_MODEL)
        out = comp.compress(h)
        assert out.shape == (2, T // G, D_C + D_H_R)

    def test_output_shape_with_pad(self):
        comp = make_compressor()
        T = 257              # 257 → pad 15 → 17 windows
        h = torch.randn(1, T, D_MODEL)
        out = comp.compress(h)
        assert out.shape == (1, math.ceil(T / G), D_C + D_H_R)

    def test_zeros_input_yields_finite(self):
        """compress(zeros) → finite output (only depends on biases / RMSNorm).

        With zero input + zero bias linears + zero APE, the unnormalized latent
        is zero. RMSNorm of zero is undefined (0/0); but PyTorch's RMSNorm
        handles this with eps and produces zeros. Output must be finite.
        """
        comp = make_compressor()
        h = torch.zeros(1, 64, D_MODEL)
        out = comp.compress(h)
        assert torch.isfinite(out).all(), "compress(zeros) produced NaN/Inf"

    def test_ones_input_finite(self):
        """compress(ones) → finite, no NaN/Inf."""
        comp = make_compressor()
        h = torch.ones(2, 128, D_MODEL)
        out = comp.compress(h)
        assert torch.isfinite(out).all(), "compress(ones) produced NaN/Inf"
        # Also sanity-check norm is non-explosive (init_scale=0.001 keeps it small)
        assert out.float().norm(dim=-1).mean().item() < 100.0

    def test_no_nan_random_input(self):
        comp = make_compressor()
        torch.manual_seed(0)
        for _ in range(3):
            h = torch.randn(2, 256, D_MODEL) * 5.0
            out = comp.compress(h)
            assert torch.isfinite(out).all()

    def test_default_chunk_4096_yields_256_latents(self):
        """Doc-spec: T=4096, g=16 → 256 latents at d_c=512 + d_h_rope=64."""
        comp = L2Compressor(
            d_model=4096, n_heads=32, d_head=128,
            compress_ratio=16, d_c=512, d_h_rope=64,
            chunk_size=4096, n_kv_heads=32, init_scale=0.001,
        )
        h = torch.randn(1, 4096, 4096)
        out = comp.compress(h)
        assert out.shape == (1, 256, 512 + 64)


class TestL2CompressorState:

    def test_prev_latents_starts_empty(self):
        comp = make_compressor()
        assert comp.prev_latents.numel() == 0

    def test_reset_clears_prev_latents(self):
        comp = make_compressor()
        # Manually populate prev_latents (as the post-forward hook would)
        h = torch.randn(1, 64, D_MODEL)
        with torch.no_grad():
            comp.prev_latents = comp.compress(h)
        assert comp.prev_latents.numel() > 0
        comp.reset()
        assert comp.prev_latents.numel() == 0

    def test_prev_latents_not_in_state_dict(self):
        """prev_latents is non-persistent; must not appear in state_dict."""
        comp = make_compressor()
        sd = comp.state_dict()
        assert "prev_latents" not in sd, (
            f"prev_latents should be non-persistent, found in state_dict: {list(sd)}"
        )


class TestL2CompressorParams:

    def test_kv_b_init_near_zero(self):
        """kv_b weight std should match init_scale (default 0.001)."""
        comp = make_compressor(init_scale=0.001)
        std = comp.kv_b.weight.detach().float().std().item()
        # std of normal(0, 0.001) on 64 × 1024 entries should be close to 0.001.
        assert std < 0.01, f"kv_b weight std={std:.4g} not near-zero"

    def test_other_weights_normal_scale(self):
        comp = make_compressor()
        for w in (comp.w_kv.weight, comp.w_gate.weight, comp.w_kR.weight):
            std = w.detach().float().std().item()
            # std=0.02 init: should be in (0.005, 0.05) loosely
            assert 0.005 < std < 0.05, f"weight std={std:.4g} unexpected"

    def test_ape_init_zeros(self):
        comp = make_compressor()
        assert torch.all(comp.ape == 0.0)

    def test_d_model_mismatch_raises(self):
        with pytest.raises(ValueError):
            L2Compressor(
                d_model=100, n_heads=8, d_head=32,
                compress_ratio=16, d_c=64, d_h_rope=16,
            )


class TestL2CompressorBackprop:

    def test_gradient_flows_through_compress(self):
        """Gradient flowing into the compressed latents must reach w_kv/w_gate/kv_b."""
        comp = make_compressor()
        h = torch.randn(1, 64, D_MODEL, requires_grad=True)
        out = comp.compress(h)
        # Pass through kv_b (this is what layer.py will do via L2 read block)
        kv = comp.kv_b(out[..., : comp.d_c])
        loss = kv.float().sum()
        loss.backward()
        assert comp.w_kv.weight.grad is not None
        assert comp.w_gate.weight.grad is not None
        assert comp.kv_b.weight.grad is not None
        # Non-zero (not all gradients are zero)
        assert comp.kv_b.weight.grad.abs().sum().item() > 0.0
