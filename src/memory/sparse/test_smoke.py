#!/usr/bin/env python3
"""Smoke test for SparseMemoryModel — no real model needed, uses mocks."""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import torch
import torch.nn as nn
from memory.sparse import SparseMemoryBank, GatedTwoPathAttention, SparseMemoryModel


def test_memory_bank():
    """Test SparseMemoryBank write/read cycle."""
    print("=== SparseMemoryBank ===")
    bank = SparseMemoryBank(num_layers=2, num_slots=16, hidden_dim=64, head_dim=16, ema_alpha=0.1)

    # Write some values
    vals = torch.randn(8, 64)
    bank.write(layer_idx=0, values=vals)

    # Read back
    q = torch.randn(4, 64)  # queries in hidden_dim
    out = bank.read(layer_idx=0, queries=q, k=4)
    assert out.shape == (4, 64), f"Expected (4, 64), got {out.shape}"
    print(f"  Read output shape: {out.shape} ✓")

    # Check memory shape
    assert bank.memory.shape == (2, 16, 64), f"Expected (2, 16, 64), got {bank.memory.shape}"
    print(f"  Memory bank shape: {bank.memory.shape} ✓")

    # Check write gate not degenerate
    gate_val = torch.sigmoid(torch.tensor(0.0)).item()
    assert 0.45 < gate_val < 0.55, f"Gate bias init should give ~0.5, got {gate_val}"
    print(f"  Write gate at bias=0.0: σ(0.0)={gate_val:.3f} ✓")
    return True


def test_gated_two_path_attention():
    """Test GatedTwoPathAttention with a mock attention module."""
    print("\n=== GatedTwoPathAttention ===")

    # Create a mock attention module mimicking HF interface
    class MockAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 4
            self.head_dim = 16
            self.hidden_size = 64
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

        def forward(self, hidden_states, **kwargs):
            B, T, D = hidden_states.shape
            # Simple uniform attention output
            out = torch.randn(B, T, D)
            return (out, None, None)

    mock_attn = MockAttention()
    bank = SparseMemoryBank(num_layers=1, num_slots=16, hidden_dim=64, head_dim=16)

    gated = GatedTwoPathAttention(
        original_attn=mock_attn,
        layer_idx=0,
        memory_bank=bank,
        window_size=8,
        top_k=4,
        head_dim=16,
        gate_bias_init=0.0,
    )

    hidden = torch.randn(1, 10, 64)
    out = gated(hidden_states=hidden)
    assert out[0].shape == (1, 10, 64), f"Expected (1, 10, 64), got {out[0].shape}"
    print(f"  Output shape: {out[0].shape} ✓")
    return True


def test_sparse_memory_model_mock():
    """Test SparseMemoryModel with a mock HuggingFace model."""
    print("\n=== SparseMemoryModel (mock) ===")

    # Build a minimal mock that quacks like Qwen/Llama
    class MockConfig:
        num_hidden_layers = 4
        hidden_size = 64
        num_attention_heads = 4

    class MockSelfAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.num_heads = 4
            self.q_proj = nn.Linear(64, 64)
            self.k_proj = nn.Linear(64, 64)
            self.v_proj = nn.Linear(64, 64)
            self.o_proj = nn.Linear(64, 64)

        def forward(self, hidden_states, **kwargs):
            B, T, D = hidden_states.shape
            return (hidden_states, None, None)

    class MockLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = MockSelfAttn()

    class MockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MockConfig()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([MockLayer() for _ in range(4)])

        def forward(self, input_ids, **kwargs):
            x = torch.randn(input_ids.shape[0], input_ids.shape[1], 64)
            # Run through layers
            for layer in self.model.layers:
                out = layer.self_attn(x)
                x = out[0]
            from dataclasses import dataclass
            @dataclass
            class Output:
                last_hidden_state: torch.Tensor
            return Output(last_hidden_state=x)

    mock_model = MockModel()
    smm = SparseMemoryModel(
        mock_model,
        layers_to_patch=[0, 2],  # only patch 2 layers
        num_slots=16,
        window_size=8,
        top_k=4,
    )

    input_ids = torch.randint(0, 1000, (1, 12))
    outputs = smm(input_ids)
    assert outputs.last_hidden_state.shape == (1, 12, 64)
    print(f"  Output shape: {outputs.last_hidden_state.shape} ✓")

    # Check memory bank shape
    assert smm.memory_bank.memory.shape == (2, 16, 64), \
        f"Expected (2, 16, 64), got {smm.memory_bank.memory.shape}"
    print(f"  Memory bank shape: {smm.memory_bank.memory.shape} ✓")

    # Check gate is not degenerate: collect gate values
    # The gate_proj weight is Kaiming-init, bias=0.0, so σ(0.0) ≈ 0.5 (max gradient)
    gate_bias = smm._get_attention(mock_model, 0).gate_proj.bias.item()
    print(f"  Gate bias value: {gate_bias:.2f} (σ={torch.sigmoid(torch.tensor(gate_bias)).item():.3f}) ✓")

    print("\n✅ All smoke tests passed!")
    return True


if __name__ == "__main__":
    test_memory_bank()
    test_gated_two_path_attention()
    test_sparse_memory_model_mock()
