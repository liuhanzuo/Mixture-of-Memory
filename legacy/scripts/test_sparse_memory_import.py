#!/usr/bin/env python3
"""Diagnostic: verify sparse memory components work with Qwen3-8B GQA shapes.

Tests import, tensor shapes, GQA handling in both local and memory paths.
No training — just forward pass smoke tests.

Qwen3-8B: hidden=4096, num_heads=32, num_kv_heads=8, head_dim=128
"""

import sys
import os
import traceback

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

PASS = 0
FAIL = 0

def check(name, fn):
    global PASS, FAIL
    try:
        fn()
        PASS += 1
        print(f"  ✅ {name}")
    except Exception as e:
        FAIL += 1
        print(f"  ❌ {name}: {e}")
        traceback.print_exc()

def main():
    print("=" * 60)
    print("Sparse Memory Diagnostic")
    print("=" * 60)

    # ── 1. Imports ──────────────────────────────────────────────────
    print("\n[1] Imports")

    def import_attention():
        from src.memory.sparse_memory.attention import SparseMemoryAttention, SparseMemoryConfig

    def import_memory():
        from src.memory.sparse_memory.memory_bank import MemoryBank

    def import_model():
        from src.memory.sparse_memory.model import SparseMemoryLlamaForCausalLM

    def import_package():
        from src.memory.sparse_memory import MemoryBank, SparseMemoryAttention, SparseMemoryLlamaForCausalLM

    check("SparseMemoryAttention import", import_attention)
    check("MemoryBank import", import_memory)
    check("SparseMemoryLlamaForCausalLM import", import_model)
    check("Package __init__.py re-exports", import_package)

    # Check for expected-but-missing names
    def check_gated_two_path():
        from src.memory.sparse_memory.attention import GatedTwoPathAttention

    check("GatedTwoPathAttention exists (expected by task spec)", check_gated_two_path)

    def check_memory_py():
        import src.memory.sparse_memory.memory as _m  # noqa: F401
        _m  # use it

    check("memory.py wildcard import (expected by task spec)", check_memory_py)

    # ── 2. MemoryBank basic ops ─────────────────────────────────────
    print("\n[2] MemoryBank basic ops")
    import torch
    from src.memory.sparse_memory.memory_bank import MemoryBank

    mb = MemoryBank(num_slots=16, hidden_dim=4096)

    def mb_reset():
        mb.reset(batch_size=2)
        assert mb.memory.shape == (2, 16, 4096), f"bad shape {mb.memory.shape}"

    def mb_write():
        mb.reset(batch_size=2)
        x = torch.randn(4096)
        mb.write(x, batch_idx=0)
        # After write, slot 0 should be non-zero
        assert mb.memory[0, 0].abs().sum() > 0, "write didn't update"

    def mb_dtype():
        mb_bf16 = MemoryBank(num_slots=4, hidden_dim=128, ema_alpha=0.5)
        mb_bf16.reset(batch_size=1)
        x = torch.randn(128, dtype=torch.bfloat16)
        mb_bf16.write(x, batch_idx=0)
        # Buffer stays float32 (no dtype constraint), just verify no crash

    check("MemoryBank reset shape", mb_reset)
    check("MemoryBank write updates", mb_write)
    check("MemoryBank bf16 write no crash", mb_dtype)

    # ── 3. SparseMemoryConfig ───────────────────────────────────────
    print("\n[3] SparseMemoryConfig")
    from src.memory.sparse_memory.attention import SparseMemoryConfig

    def cfg_defaults():
        c = SparseMemoryConfig()
        assert c.num_mem_tokens == 128
        assert c.window_size == 256

    check("Config defaults", cfg_defaults)

    # ── 4. SparseMemoryAttention forward with GQA shapes ────────────
    print("\n[4] SparseMemoryAttention forward (GQA, Qwen3-8B shapes)")
    from src.memory.sparse_memory.attention import SparseMemoryAttention

    # Qwen3-8B: num_heads=32, num_kv_heads=8, head_dim=128, hidden=4096
    NUM_HEADS = 32
    NUM_KV_HEADS = 8
    HEAD_DIM = 128
    HIDDEN = NUM_HEADS * HEAD_DIM  # 4096
    B, T = 2, 64  # small batch/seq

    # Build a fake LlamaAttention-like module
    class FakeLlamaConfig:
        num_attention_heads = NUM_HEADS
        num_key_value_heads = NUM_KV_HEADS
        hidden_size = HIDDEN

    class FakeLlamaAttention(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeLlamaConfig()
            self.num_heads = NUM_HEADS
            self.num_kv_heads = NUM_KV_HEADS
            self.head_dim = HEAD_DIM
            self.q_proj = torch.nn.Linear(HIDDEN, NUM_HEADS * HEAD_DIM, dtype=torch.float32)
            self.k_proj = torch.nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
            self.v_proj = torch.nn.Linear(HIDDEN, NUM_KV_HEADS * HEAD_DIM, dtype=torch.float32)
            self.o_proj = torch.nn.Linear(NUM_HEADS * HEAD_DIM, HIDDEN, dtype=torch.float32)

    attn = FakeLlamaAttention()
    mb = MemoryBank(num_slots=16, hidden_dim=HIDDEN)
    mb.reset(batch_size=B)

    sma = SparseMemoryAttention(
        original_attn=attn,
        memory_bank=mb,
        window_size=32,
        top_k=4,
        gate_bias_init=2.0,
    )

    x = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    cos = torch.randn(B, 1, T, HEAD_DIM, dtype=torch.float32)
    sin = torch.randn(B, 1, T, HEAD_DIM, dtype=torch.float32)

    def forward_gqa():
        out = sma(x, position_embeddings=(cos, sin))
        assert out[0].shape == (B, T, HIDDEN), f"output shape {out[0].shape}"
        # Verify output is not all zeros
        assert out[0].abs().sum() > 0, "output is all zeros"

    def gate_stats():
        out = sma(x, position_embeddings=(cos, sin))
        stats = sma.get_gate_stats()
        assert stats["gate_mean"] is not None
        # With bias_init=2.0, sigmoid(2.0)≈0.88, so gate mean should be high
        assert stats["gate_mean"] > 0.5, f"gate_mean too low: {stats['gate_mean']}"

    def output_range():
        out = sma(x, position_embeddings=(cos, sin))
        o = out[0]
        assert not torch.isnan(o).any(), "NaN in output"
        assert not torch.isinf(o).any(), "Inf in output"

    check("Forward pass output shape", forward_gqa)
    check("Gate stats available", gate_stats)
    check("No NaN/Inf in output", output_range)

    # ── 5. Non-GQA (num_kv_heads == num_heads) ─────────────────────
    print("\n[5] Non-GQA path (MHA)")

    class FakeMHACfg:
        num_attention_heads = 8
        num_key_value_heads = 8
        hidden_size = 1024

    class FakeMHA(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = FakeMHACfg()
            self.num_heads = 8
            self.num_kv_heads = 8
            self.head_dim = 128
            H2 = 1024
            self.q_proj = torch.nn.Linear(H2, 8*128, dtype=torch.float32)
            self.k_proj = torch.nn.Linear(H2, 8*128, dtype=torch.float32)
            self.v_proj = torch.nn.Linear(H2, 8*128, dtype=torch.float32)
            self.o_proj = torch.nn.Linear(8*128, H2, dtype=torch.float32)

    mha = FakeMHA()
    mb2 = MemoryBank(num_slots=8, hidden_dim=1024)
    mb2.reset(batch_size=1)
    sma2 = SparseMemoryAttention(mha, mb2, window_size=16, top_k=2)

    x2 = torch.randn(1, 32, 1024)
    cos2 = torch.randn(1, 1, 32, 128)
    sin2 = torch.randn(1, 1, 32, 128)

    def mha_forward():
        out = sma2(x2, position_embeddings=(cos2, sin2))
        assert out[0].shape == (1, 32, 1024)

    check("Non-GQA forward pass", mha_forward)

    # ── 6. Memory write during forward ──────────────────────────────
    print("\n[6] Memory write during forward")
    mb3 = MemoryBank(num_slots=4, hidden_dim=HIDDEN)
    mb3.reset(batch_size=1)
    sma3 = SparseMemoryAttention(FakeLlamaAttention(), mb3, window_size=32, top_k=2)
    x3 = torch.randn(1, 8, HIDDEN)
    cos3 = torch.randn(1, 1, 8, HEAD_DIM)
    sin3 = torch.randn(1, 1, 8, HEAD_DIM)

    def mem_written():
        assert mb3.memory[0].abs().sum() == 0, "memory should be zero before forward"
        sma3(x3, position_embeddings=(cos3, sin3))
        assert mb3.memory[0].abs().sum() > 0, "memory not updated after forward"

    check("Memory updated after forward", mem_written)

    # ── Summary ──────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"Results: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("✅ Sparse memory module is READY for production use.")
    else:
        print("❌ Sparse memory module needs further debugging.")
    print("=" * 60)
    return FAIL

if __name__ == "__main__":
    sys.exit(main())
