#!/usr/bin/env python3
"""Smoke test: 10 forward passes of SparseMemoryAttention without errors.

Tests on CPU to avoid GPU dependency. Validates the top-k retrieval
doesn't index out-of-bounds even with empty/zero memory.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from src.memory.sparse_memory.memory_bank import MemoryBank
from src.memory.sparse_memory.attention import SparseMemoryAttention


class FakeLlamaAttn(nn.Module):
    """Minimal mock of LlamaAttention for testing."""
    def __init__(self, hidden_size=512, num_heads=8, num_kv_heads=4):
        super().__init__()
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.hidden_size = hidden_size
        self.q_proj = nn.Linear(hidden_size, num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.config = type('C', (), {
            'num_attention_heads': num_heads,
            'num_key_value_heads': num_kv_heads,
            'hidden_size': hidden_size,
            'attention_dropout': 0.0,
        })()
        self.layer_idx = 0


def test_basic_forward():
    """10 forward passes with various batch sizes and seq lengths."""
    hidden_size = 512
    num_heads = 8
    num_kv_heads = 4
    num_slots = 16
    top_k = 8
    window_size = 32

    attn = SparseMemoryAttention(
        original_attn=FakeLlamaAttn(hidden_size, num_heads, num_kv_heads),
        memory_bank=MemoryBank(num_slots=num_slots, hidden_dim=hidden_size, dtype=torch.float32),
        window_size=window_size,
        top_k=top_k,
    )
    attn.eval()

    configs = [
        (1, 64),
        (2, 64),
        (1, 128),
        (4, 32),
        (2, 256),
        (1, 16),   # very short
        (3, 48),
        (1, 512),
        (2, 96),
        (1, 100),  # odd length
    ]

    for i, (B, T) in enumerate(configs):
        attn.memory_bank.reset(batch_size=B)
        x = torch.randn(B, T, hidden_size)
        with torch.no_grad():
            out, _ = attn(x)
        assert out.shape == (B, T, hidden_size), f"Pass {i}: shape mismatch {out.shape}"
        print(f"  Pass {i+1}: B={B}, T={T} → OK (out shape {out.shape})")

    print("All 10 forward passes passed!")


def test_topk_exceeds_slots():
    """top_k > num_slots should not crash."""
    hidden_size = 256
    attn = SparseMemoryAttention(
        original_attn=FakeLlamaAttn(hidden_size, 4, 2),
        memory_bank=MemoryBank(num_slots=4, hidden_dim=hidden_size, dtype=torch.float32),
        window_size=16,
        top_k=32,  # much larger than num_slots=4
    )
    attn.memory_bank.reset(batch_size=1)
    x = torch.randn(1, 32, hidden_size)
    with torch.no_grad():
        out, _ = attn(x)
    assert out.shape == (1, 32, hidden_size)
    print("  top_k > num_slots: OK (clamped to num_slots)")


def test_empty_memory():
    """Forward with zero-initialized memory (before any write)."""
    hidden_size = 256
    attn = SparseMemoryAttention(
        original_attn=FakeLlamaAttn(hidden_size, 4, 2),
        memory_bank=MemoryBank(num_slots=8, hidden_dim=hidden_size, dtype=torch.float32),
        window_size=16,
        top_k=4,
    )
    attn.memory_bank.reset(batch_size=2)
    x = torch.randn(2, 64, hidden_size)
    with torch.no_grad():
        out, _ = attn(x)
    assert out.shape == (2, 64, hidden_size)
    print("  Empty memory forward: OK")


if __name__ == "__main__":
    print("=== Sparse Memory Smoke Test ===")
    test_basic_forward()
    test_topk_exceeds_slots()
    test_empty_memory()
    print("\n=== ALL TESTS PASSED ===")
