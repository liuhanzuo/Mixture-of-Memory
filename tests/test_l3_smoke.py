#!/usr/bin/env python3
"""CPU smoke test for L3 summary-token module.

Verifies:
1. L3SummaryPool standalone forward produces correct shape
2. Integration with MemorySpaceLayer via patch — extended mask shape
3. Gradient flows through L3 pool weights after backward
4. Dual-gate weights present and loadable alongside L3
"""
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from src.memory.mem_space.config import MemorySpaceConfig
from src.memory.mem_space.l3_summary import L3SummaryPool
from src.memory.mem_space.layer import MemorySpaceLayer, _build_extended_attn_mask
from src.memory.mem_space.memory_bank import MemoryBank


def test_l3_pool_standalone():
    """Test L3SummaryPool in isolation."""
    print("=== Test 1: L3SummaryPool standalone ===")
    B, T, d = 2, 16, 64
    K = 4
    pool = L3SummaryPool(d_model=d, num_summary=K, num_heads=4, n_layers=2)
    x = torch.randn(B, T, d)
    out = pool(x)
    assert out.shape == (B, K, d), f"Expected ({B},{K},{d}), got {out.shape}"
    print(f"  Output shape: {out.shape} -- PASS")

    # Test with mask
    mask = torch.ones(B, T)
    mask[:, -3:] = 0  # last 3 tokens are padding
    out_masked = pool(x, chunk_mask=mask)
    assert out_masked.shape == (B, K, d)
    print(f"  Masked output shape: {out_masked.shape} -- PASS")

    # Gradient check
    loss = out.sum()
    loss.backward()
    assert pool.queries.grad is not None, "queries should have gradient"
    assert pool.queries.grad.abs().sum() > 0, "queries gradient should be non-zero"
    print(f"  Gradient on queries: norm={pool.queries.grad.norm().item():.6f} -- PASS")

    # Check block params got gradient
    for i, blk in enumerate(pool.blocks):
        for name, p in blk.named_parameters():
            if p.grad is not None:
                assert p.grad.abs().sum() > 0, f"Block {i} {name} grad should be non-zero"
    print("  Block parameter gradients: non-zero -- PASS")
    print()


def test_extended_attn_mask_with_l3():
    """Test _build_extended_attn_mask handles L3 prefix correctly."""
    print("=== Test 2: Extended attention mask with L3 ===")
    B, T, k, k_l3 = 2, 16, 2, 4

    mask = _build_extended_attn_mask(
        k=k, T=T, dtype=torch.float32, device=torch.device("cpu"),
        batch_size=B, swa_window=0, k_l3=k_l3,
    )
    L = k_l3 + k + T
    expected_shape = (B, 1, L, L)
    assert mask.shape == expected_shape, f"Expected {expected_shape}, got {mask.shape}"
    print(f"  Mask shape: {mask.shape} = [B=2, 1, {L}, {L}] -- PASS")

    # L3 rows (0..k_l3-1) should attend everywhere (all zeros)
    l3_rows = mask[0, 0, :k_l3, :]
    assert (l3_rows == 0).all(), "L3 rows should be all zeros (attend everywhere)"
    print("  L3 rows attend everywhere: PASS")

    # L1 rows (k_l3..k_l3+k-1) should attend everywhere
    l1_rows = mask[0, 0, k_l3:k_l3+k, :]
    assert (l1_rows == 0).all(), "L1 rows should be all zeros (attend everywhere)"
    print("  L1 rows attend everywhere: PASS")

    # H rows should have causal pattern in the H×H block
    h_rows = mask[0, 0, k_l3+k:, k_l3+k:]  # H×H block
    # Upper triangle should be -inf (masked)
    neg_inf = torch.finfo(torch.float32).min
    for i in range(T):
        for j in range(T):
            if j > i:
                assert h_rows[i, j] == neg_inf, f"H[{i},{j}] should be masked"
            else:
                assert h_rows[i, j] == 0, f"H[{i},{j}] should be allowed"

    # H rows → L3/L1 columns should be all zeros (allowed)
    h_to_prefix = mask[0, 0, k_l3+k:, :k_l3+k]
    assert (h_to_prefix == 0).all(), "H should attend to all L3+L1 tokens"
    print("  H→L3/L1 columns all allowed: PASS")
    print("  H→H causal pattern: PASS")
    print()


def test_mask_backward_compat_no_l3():
    """Verify mask is unchanged when k_l3=0."""
    print("=== Test 3: Mask backward compat (k_l3=0) ===")
    B, T, k = 2, 16, 4
    mask_old = _build_extended_attn_mask(
        k=k, T=T, dtype=torch.float32, device=torch.device("cpu"),
        batch_size=B, swa_window=0, k_l3=0,
    )
    assert mask_old.shape == (B, 1, k+T, k+T)
    # Slot rows attend everywhere
    assert (mask_old[0, 0, :k, :] == 0).all()
    print(f"  k_l3=0 mask shape: {mask_old.shape} -- backward compat PASS")
    print()


def test_layer_integration():
    """Test MemorySpaceLayer with L3 summaries prepended."""
    print("=== Test 4: MemorySpaceLayer with L3 integration ===")
    B, T, d = 2, 16, 64
    K_sum = 4
    k = 2
    n_slots = 8

    # Build a minimal "wrapped layer" that does cross-position mixing
    # (needed for gradient to flow from H positions back to L3 positions).
    class FakeAttnLayer(nn.Module):
        """Fake layer that does mean-pool mixing so all positions influence output."""
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.linear.weight)

        def forward(self, hidden_states, **kwargs):
            # Simple "attention": each position attends to mean of all positions
            # This ensures gradient flows from any position to all others.
            mean = hidden_states.mean(dim=1, keepdim=True)
            return self.linear(hidden_states + 0.1 * mean)

    wrapped = FakeAttnLayer(d)

    cfg = MemorySpaceConfig(
        num_slots=n_slots,
        top_k=k,
        slot_dim=d,
        selector_dim=32,
        slot_init="random",
        slot_init_noise=0.1,
        shared_memory_bank=False,
        use_l3_summary=True,
        l3_n_summary=K_sum,
        l3_n_layers=1,
        l3_n_heads=4,
    )

    # Create L3 pool and layer
    l3_pool = L3SummaryPool(d_model=d, num_summary=K_sum, num_heads=4, n_layers=1)
    layer = MemorySpaceLayer(wrapped, cfg, d_model=d, l3_pool=l3_pool)

    # Create fake position embeddings (cos, sin) for RoPE
    head_dim = d // 4  # assume 4 heads
    cos = torch.ones(1, T, head_dim)
    sin = torch.zeros(1, T, head_dim)
    position_embeddings = (cos, sin)

    # Create L3 summaries (simulating previous chunk's output)
    l3_summaries = torch.randn(B, K_sum, d, requires_grad=True)

    hidden_states = torch.randn(B, T, d, requires_grad=True)

    # Forward with L3
    out = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        l3_summaries=l3_summaries,
    )
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (B, T, d), f"Expected ({B},{T},{d}), got {out.shape}"
    print(f"  Output shape with L3: {out.shape} -- PASS")

    # Verify gradient flows through L3 (since FakeAttnLayer mixes positions)
    loss = out.sum()
    loss.backward()
    assert l3_summaries.grad is not None, "l3_summaries should have gradient"
    assert l3_summaries.grad.abs().sum() > 0, "l3_summaries grad should be non-zero"
    print(f"  L3 summaries gradient: norm={l3_summaries.grad.norm().item():.6f} -- PASS")
    print()


def test_layer_without_l3():
    """Test that MemorySpaceLayer still works without L3 (backward compat)."""
    print("=== Test 5: MemorySpaceLayer without L3 (backward compat) ===")
    B, T, d = 2, 16, 64
    k = 2
    n_slots = 8

    class FakeDecoderLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.linear.weight)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    wrapped = FakeDecoderLayer(d)

    cfg = MemorySpaceConfig(
        num_slots=n_slots,
        top_k=k,
        slot_dim=d,
        selector_dim=32,
        slot_init="random",
        slot_init_noise=0.1,
        shared_memory_bank=False,
        use_l3_summary=False,
    )

    layer = MemorySpaceLayer(wrapped, cfg, d_model=d)

    head_dim = d // 4
    cos = torch.ones(1, T, head_dim)
    sin = torch.zeros(1, T, head_dim)
    position_embeddings = (cos, sin)

    hidden_states = torch.randn(B, T, d)
    out = layer(hidden_states, position_embeddings=position_embeddings)
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (B, T, d), f"Expected ({B},{T},{d}), got {out.shape}"
    print(f"  Output shape without L3: {out.shape} -- PASS")
    print()


def test_dual_gate_with_l3():
    """Test dual-gate weights present alongside L3."""
    print("=== Test 6: Dual-gate + L3 coexistence ===")
    B, T, d = 2, 16, 64
    K_sum = 4
    k = 2
    n_slots = 8

    class FakeDecoderLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.linear.weight)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    wrapped = FakeDecoderLayer(d)

    cfg = MemorySpaceConfig(
        num_slots=n_slots,
        top_k=k,
        slot_dim=d,
        selector_dim=32,
        slot_init="random",
        slot_init_noise=0.1,
        shared_memory_bank=False,
        use_l3_summary=True,
        l3_n_summary=K_sum,
        l3_n_layers=1,
        l3_n_heads=4,
        use_dual_gate=True,
        forget_bias_init=2.0,
    )

    l3_pool = L3SummaryPool(d_model=d, num_summary=K_sum, num_heads=4, n_layers=1)
    layer = MemorySpaceLayer(wrapped, cfg, d_model=d, l3_pool=l3_pool)

    # Check dual-gate params exist
    assert layer.gate_proj_new is not None, "gate_proj_new should exist"
    assert layer.gate_proj_mem is not None, "gate_proj_mem should exist"
    assert layer.gate_bias is not None, "gate_bias should exist"
    print("  Dual-gate params present: PASS")

    # Check L3 pool reference
    assert layer.l3_pool is not None, "l3_pool should be set"
    print("  L3 pool reference: PASS")

    # Forward
    head_dim = d // 4
    cos = torch.ones(1, T, head_dim)
    sin = torch.zeros(1, T, head_dim)
    position_embeddings = (cos, sin)

    l3_summaries = torch.randn(B, K_sum, d)
    hidden_states = torch.randn(B, T, d)
    out = layer(
        hidden_states,
        position_embeddings=position_embeddings,
        l3_summaries=l3_summaries,
    )
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == (B, T, d)
    print(f"  Dual-gate + L3 forward shape: {out.shape} -- PASS")
    print()


def test_l3_pool_param_count():
    """Verify L3 param count is in expected range."""
    print("=== Test 7: L3 param count check ===")
    # At full scale: d=4096, K=64, 2 layers, 8 heads
    # For smoke test: d=64, K=4, 1 layer, 4 heads
    pool_small = L3SummaryPool(d_model=64, num_summary=4, num_heads=4, n_layers=1)
    n_params_small = sum(p.numel() for p in pool_small.parameters())
    print(f"  Small pool (d=64, K=4, 1 layer): {n_params_small:,} params")

    pool_medium = L3SummaryPool(d_model=64, num_summary=4, num_heads=4, n_layers=2)
    n_params_medium = sum(p.numel() for p in pool_medium.parameters())
    print(f"  Medium pool (d=64, K=4, 2 layers): {n_params_medium:,} params")

    # Full-scale estimate
    pool_full = L3SummaryPool(d_model=4096, num_summary=64, num_heads=8, n_layers=2)
    n_params_full = sum(p.numel() for p in pool_full.parameters())
    print(f"  Full-scale pool (d=4096, K=64, 2 layers): {n_params_full:,} params")
    # Should be O(50-150M)
    assert 10_000_000 < n_params_full < 500_000_000, (
        f"Full-scale params {n_params_full} outside expected range [10M, 500M]"
    )
    print(f"  Full-scale in [10M, 500M] range: PASS")
    print()


def test_auto_read_from_pool():
    """Test that layer auto-reads l3_summaries from l3_pool._current_summary."""
    print("=== Test 8: Auto-read L3 from pool state ===")
    B, T, d = 2, 16, 64
    K_sum = 4
    k = 2
    n_slots = 8

    class FakeDecoderLayer(nn.Module):
        def __init__(self, d_model):
            super().__init__()
            self.linear = nn.Linear(d_model, d_model, bias=False)
            nn.init.eye_(self.linear.weight)

        def forward(self, hidden_states, **kwargs):
            return self.linear(hidden_states)

    wrapped = FakeDecoderLayer(d)

    cfg = MemorySpaceConfig(
        num_slots=n_slots,
        top_k=k,
        slot_dim=d,
        selector_dim=32,
        slot_init="random",
        slot_init_noise=0.1,
        shared_memory_bank=False,
        use_l3_summary=True,
        l3_n_summary=K_sum,
        l3_n_layers=1,
        l3_n_heads=4,
    )

    l3_pool = L3SummaryPool(d_model=d, num_summary=K_sum, num_heads=4, n_layers=1)
    layer = MemorySpaceLayer(wrapped, cfg, d_model=d, l3_pool=l3_pool)

    head_dim = d // 4
    cos = torch.ones(1, T, head_dim)
    sin = torch.zeros(1, T, head_dim)
    position_embeddings = (cos, sin)

    # Set _current_summary on pool (simulating previous chunk's hook)
    l3_pool._current_summary = torch.randn(B, K_sum, d)

    hidden_states = torch.randn(B, T, d)
    # Forward WITHOUT passing l3_summaries explicitly
    out = layer(hidden_states, position_embeddings=position_embeddings)
    if isinstance(out, tuple):
        out = out[0]

    assert out.shape == (B, T, d), f"Expected ({B},{T},{d}), got {out.shape}"
    print(f"  Auto-read from pool works, output: {out.shape} -- PASS")

    # Test cold start (no summary stashed)
    l3_pool._current_summary = None
    out2 = layer(hidden_states, position_embeddings=position_embeddings)
    if isinstance(out2, tuple):
        out2 = out2[0]
    assert out2.shape == (B, T, d)
    print(f"  Cold start (None summary), output: {out2.shape} -- PASS")
    print()


if __name__ == "__main__":
    print("=" * 60)
    print("L3 Summary-Token Module Smoke Test")
    print("=" * 60)
    print()

    test_l3_pool_standalone()
    test_extended_attn_mask_with_l3()
    test_mask_backward_compat_no_l3()
    test_layer_integration()
    test_layer_without_l3()
    test_dual_gate_with_l3()
    test_l3_pool_param_count()
    test_auto_read_from_pool()

    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
