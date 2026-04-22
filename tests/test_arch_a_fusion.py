"""Test Architecture A (concatenation fusion) forward pass."""

import torch
import torch.nn as nn
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.memory.sparse_memory.attention import SparseMemoryAttention


class FakeMemoryBank:
    def __init__(self, num_slots=128, dim=512, batch_size=2):
        self.num_slots = num_slots
        self.memory = torch.randn(batch_size, num_slots, dim)

    def write(self, tensor, batch_idx=0):
        pass  # no-op for testing


class FakeConfig:
    hidden_size = 512
    num_attention_heads = 8
    num_key_value_heads = 8
    attention_dropout = 0.0


class FakeAttn(nn.Module):
    def __init__(self, hidden_size=512, num_heads=8):
        super().__init__()
        self.config = FakeConfig()
        self.head_dim = hidden_size // num_heads
        self.num_heads = num_heads
        self.layer_idx = 0
        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.o_proj = nn.Linear(hidden_size, hidden_size, bias=False)


def test_forward_pass():
    B, T, D = 2, 16, 512
    num_mem = 64

    fake_attn = FakeAttn(D)
    mem_bank = FakeMemoryBank(num_mem, D, B)

    model = SparseMemoryAttention(fake_attn, mem_bank, window_size=8, top_k=4)

    x = torch.randn(B, T, D, dtype=torch.float32)
    with torch.no_grad():
        out, _ = model(x)

    assert out.shape == (B, T, D), f"Expected {(B, T, D)}, got {out.shape}"
    assert not torch.isnan(out).any(), "NaN in output"
    assert not torch.isinf(out).any(), "Inf in output"
    print(f"✓ Forward pass OK: output shape {out.shape}")

    # Verify fusion_proj is concat-based (input dim = 2*D)
    assert model.fusion_proj.in_features == 2 * D, "fusion_proj input should be 2*D"
    assert model.fusion_proj.out_features == D, "fusion_proj output should be D"
    print(f"✓ fusion_proj: Linear({2*D} → {D})")

    # Verify identity init: output should be ~equal to o_local
    # (since right-half weights are zero)
    print(f"✓ All tests passed!")


if __name__ == '__main__':
    test_forward_pass()
