#!/usr/bin/env python3
"""Test that contrastive capture works under slot_forward + middle_layer_memory.

Validates the fix for the silent no-op bug where forward_niah_sample skipped
InfoNCE contrastive loss entirely when is_slot_forward=True.
"""
import sys
import os

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM


def test_contrastive_capture():
    """Build a small model with slot_forward=True, middle_layer_memory=True,
    call forward_chunk with capture_read_attn=True, and verify:
    1. result contains 'read_attn_weights' and 'read_attn_logits'
    2. Each dict entry has shape [B, n_heads, T, num_slots]
    3. Captured tensors are detached (no grad_fn)
    4. Gradient still flows through main loss to cross_attn.q_proj.weight
    """
    print("=" * 60)
    print("TEST: contrastive capture under slot_forward + middle_layer_memory")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32

    # Use local Llama-3-8B config (no internet) and shrink to tiny for testing
    model_path = "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
    config = AutoConfig.from_pretrained(model_path)
    # Override to make it tiny (mirror test_h9_grad_flow.py)
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_hidden_layers = 8
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.vocab_size = 1000

    base_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)

    # Import the model class from the training script
    sys.path.insert(0, os.path.join(PROJECT_ROOT, "scripts"))
    from train_cross_attn_memory import CrossAttentionMemoryModel

    # Build memory model with slot_forward + middle_layer_memory + dual_gate
    num_slots = 8
    model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=num_slots,
        slot_forward=True,
        middle_layer_memory=True,
        memory_write_layer=2,
        memory_read_layers="3,4,5",
        use_dual_gate=True,
        forget_bias_init=1.0,
        input_bias_init=0.0,
        dual_gate_tanh_new=True,
        gradient_checkpointing=False,
        residual_scale=0.01,
        use_memory=True,
        use_cross_attn_memory=True,
    )
    model = model.to(device).to(dtype)
    model.train()

    B = 1
    T = 32

    # First chunk: initialize slots (write layer needs a first pass)
    input_ids_1 = torch.randint(0, config.vocab_size, (B, T), device=device)
    model.forward_chunk(input_ids_1, enable_write_grad=True)

    # Second chunk: capture read attention
    input_ids_2 = torch.randint(0, config.vocab_size, (B, T), device=device)
    result = model.forward_chunk(
        input_ids_2, enable_write_grad=True, capture_read_attn=True
    )

    # ---- Check 1: result contains captured attention ----
    assert "read_attn_weights" in result, (
        "FAIL: result missing 'read_attn_weights'. "
        f"Keys present: {list(result.keys())}"
    )
    assert "read_attn_logits" in result, (
        "FAIL: result missing 'read_attn_logits'. "
        f"Keys present: {list(result.keys())}"
    )
    print("[PASS] result contains 'read_attn_weights' and 'read_attn_logits'")

    # ---- Check 2: dict has one entry per read layer ----
    read_layers = {3, 4, 5}
    assert set(result["read_attn_weights"].keys()) == read_layers, (
        f"FAIL: expected layers {read_layers}, got {set(result['read_attn_weights'].keys())}"
    )
    assert set(result["read_attn_logits"].keys()) == read_layers, (
        f"FAIL: expected layers {read_layers}, got {set(result['read_attn_logits'].keys())}"
    )
    print(f"[PASS] captured attention has entries for layers {sorted(read_layers)}")

    # ---- Check 3: shapes are [B, n_heads, T, num_slots] ----
    n_heads = config.num_attention_heads
    expected_shape = (B, n_heads, T, num_slots)
    for layer_idx in read_layers:
        w_shape = result["read_attn_weights"][layer_idx].shape
        l_shape = result["read_attn_logits"][layer_idx].shape
        assert w_shape == expected_shape, (
            f"FAIL: read_attn_weights[{layer_idx}] shape {w_shape} != expected {expected_shape}"
        )
        assert l_shape == expected_shape, (
            f"FAIL: read_attn_logits[{layer_idx}] shape {l_shape} != expected {expected_shape}"
        )
    print(f"[PASS] all captured tensors have shape {expected_shape}")

    # ---- Check 4: captured tensors are detached ----
    for layer_idx in read_layers:
        assert result["read_attn_weights"][layer_idx].grad_fn is None, (
            f"FAIL: read_attn_weights[{layer_idx}] has grad_fn (not detached)"
        )
        assert result["read_attn_logits"][layer_idx].grad_fn is None, (
            f"FAIL: read_attn_logits[{layer_idx}] has grad_fn (not detached)"
        )
    print("[PASS] all captured tensors are detached (no grad_fn)")

    # ---- Check 5: gradient flows through main path ----
    # Mirror test_h9_grad_flow.py logic: at step 0 only out_proj gets a nonzero
    # gradient (its zero-init weights block upstream). After one optimizer step,
    # out_proj is nonzero so step 1 should produce nonzero q_proj grad too.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Step 0
    optimizer.zero_grad()
    model.reset_slots()
    input_ids_3 = torch.randint(0, config.vocab_size, (B, T), device=device)
    model.forward_chunk(input_ids_3, enable_write_grad=True)
    input_ids_4 = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = torch.randint(0, config.vocab_size, (B, T), device=device)
    result2 = model.forward_chunk(
        input_ids_4, labels=labels, enable_write_grad=True, capture_read_attn=True
    )
    loss = result2["loss"]
    loss.backward()

    out_proj_grad = model.cross_attn_modules[0].out_proj.weight.grad
    assert out_proj_grad is not None and out_proj_grad.norm().item() > 0, (
        "FAIL: cross_attn_modules[0].out_proj.weight.grad is None or zero at step 0 "
        "— forward path is broken"
    )
    print(f"[PASS] step 0: out_proj.weight.grad.norm() = {out_proj_grad.norm().item():.6f}")

    # Optimizer step makes out_proj nonzero
    optimizer.step()

    # Step 1: gradient should now flow all the way to q_proj
    optimizer.zero_grad()
    model.reset_slots()
    input_ids_5 = torch.randint(0, config.vocab_size, (B, T), device=device)
    model.forward_chunk(input_ids_5, enable_write_grad=True)
    input_ids_6 = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels2 = torch.randint(0, config.vocab_size, (B, T), device=device)
    result3 = model.forward_chunk(
        input_ids_6, labels=labels2, enable_write_grad=True, capture_read_attn=True
    )
    loss2 = result3["loss"]
    loss2.backward()

    q_proj_grad = model.cross_attn_modules[0].q_proj.weight.grad
    assert q_proj_grad is not None and q_proj_grad.norm().item() > 0, (
        f"FAIL: cross_attn_modules[0].q_proj.weight.grad is "
        f"{'None' if q_proj_grad is None else f'zero (norm={q_proj_grad.norm().item()})'} at step 1"
    )
    print(f"[PASS] step 1: q_proj.weight.grad.norm() = {q_proj_grad.norm().item():.3e}")
    print("[PASS] gradient flows through main loss to cross_attn.q_proj.weight")

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    test_contrastive_capture()
