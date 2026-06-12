#!/usr/bin/env python3
"""Minimal gradient flow test for H9 (real cross-attention in middle_layer_memory mode).

Verifies that:
1. cross_attn_modules are actually built (non-empty nn.ModuleList)
2. Step 0: out_proj gets nonzero gradient (proves forward path works)
   - q/k/v_proj and dual_gate correctly get zero grad at step 0 because out_proj=0
3. Step 1: After optimizer step makes out_proj nonzero, ALL params get gradient
   - cross_attn q/k/v_proj should now have nonzero grad
   - dual_gate should now have nonzero grad (gradient flows through slots → cross_attn)

Usage:
    python test_h9_grad_flow.py
"""

import sys
import os
import torch
import torch.nn as nn

# Ensure project root is on path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoModelForCausalLM, AutoConfig


def check_gradients(cm_model, step_label, check_upstream=False):
    """Check gradient status of all key params. Returns (zero_list, none_list)."""
    print(f"\n{'=' * 60}")
    print(f"GRADIENT REPORT — {step_label}")
    print(f"{'=' * 60}")

    zero_grad = []
    none_grad = []

    # Check cross_attn_modules
    print("\n--- cross_attn_modules ---")
    for i, ca in enumerate(cm_model.cross_attn_modules):
        for pname, param in ca.named_parameters():
            if not param.requires_grad:
                continue
            full = f"cross_attn_modules.{i}.{pname}"
            if param.grad is None:
                print(f"  {full}: NONE")
                none_grad.append(full)
            elif param.grad.norm().item() == 0:
                print(f"  {full}: ZERO")
                zero_grad.append(full)
            else:
                print(f"  {full}: OK (norm={param.grad.norm().item():.6f})")

    # Check dual_gate params
    print("\n--- dual_gate ---")
    for name in ["dual_gate_proj_new", "dual_gate_proj_mem", "dual_gate_bias"]:
        attr = getattr(cm_model, name, None)
        if attr is None:
            print(f"  {name}: NOT PRESENT")
            continue
        if isinstance(attr, nn.Linear):
            p = attr.weight
            full_name = f"{name}.weight"
        elif isinstance(attr, nn.Parameter):
            p = attr
            full_name = name
        else:
            continue
        if not p.requires_grad:
            print(f"  {full_name}: not requires_grad")
            continue
        if p.grad is None:
            print(f"  {full_name}: NONE")
            none_grad.append(full_name)
        elif p.grad.norm().item() == 0:
            print(f"  {full_name}: ZERO")
            zero_grad.append(full_name)
        else:
            print(f"  {full_name}: OK (norm={p.grad.norm().item():.6f})")

    return zero_grad, none_grad


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32  # float32 for grad checking

    print(f"Device: {device}, dtype: {dtype}")
    print("=" * 60)

    # --- Step 1: Load a small model ---
    model_path = "/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
    config = AutoConfig.from_pretrained(model_path)
    config.num_hidden_layers = 8
    config.hidden_size = 256
    config.intermediate_size = 512
    config.num_attention_heads = 8
    config.num_key_value_heads = 4
    config.vocab_size = 1000

    base_model = AutoModelForCausalLM.from_config(config).to(device).to(dtype)
    print(f"Base model: {config.num_hidden_layers} layers, d_model={config.hidden_size}")

    # --- Step 2: Build CrossAttentionMemoryModel ---
    from scripts.train_cross_attn_memory import CrossAttentionMemoryModel

    num_slots = 16
    write_layer = 4
    read_layers = "5,6,7"

    cm_model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=num_slots,
        top_k=num_slots,
        full_finetune=True,
        use_memory=True,
        use_cross_attn_memory=True,
        gradient_checkpointing=False,
        cross_attn_dropout=0.0,
        residual_scale=0.01,
        swa_window=0,
        write_lr=0.1,
        slot_forward=True,
        slot_isolated=False,
        memory_init="strided",
        recon_loss_weight=0.0,
        cross_chunk_propagation=False,
        middle_layer_memory=True,
        memory_write_layer=write_layer,
        memory_read_layers=read_layers,
        use_dual_gate=True,
        forget_bias_init=1.0,
        input_bias_init=0.0,
        dual_gate_tanh_new=True,
    ).to(device).to(dtype)

    # --- Step 3: Verify cross_attn_modules are built ---
    n_ca = len(cm_model.cross_attn_modules)
    expected_n_read = len(read_layers.split(","))
    print(f"\ncross_attn_modules count: {n_ca}")
    assert n_ca == expected_n_read, f"FAIL: expected {expected_n_read}, got {n_ca}"
    print(f"  -> OK: {n_ca} modules for read layers {read_layers}")
    assert cm_model._read_layer_to_ca_idx is not None
    print(f"  -> _read_layer_to_ca_idx: {cm_model._read_layer_to_ca_idx}")

    # --- Step 4: Create optimizer ---
    optimizer = torch.optim.AdamW(cm_model.parameters(), lr=1e-3)

    B, T = 2, 128
    input_ids = torch.randint(0, config.vocab_size, (B, T), device=device)
    labels = input_ids.clone()

    # ================================================================
    # STEP 0: First forward-backward (out_proj=0, so upstream grads=0)
    # ================================================================
    cm_model.train()
    cm_model.reset_slots()
    optimizer.zero_grad()

    print(f"\n--- Step 0: Forward (out_proj=0 at init) ---")
    result = cm_model.forward_chunk(input_ids, labels=labels, enable_write_grad=True)
    loss = result["loss"]
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()

    zero0, none0 = check_gradients(cm_model, "Step 0 (out_proj=0 → upstream zero expected)")

    # At step 0, out_proj.weight should have nonzero grad
    out_proj_grad0 = cm_model.cross_attn_modules[0].out_proj.weight.grad
    assert out_proj_grad0 is not None, "FAIL: out_proj.weight.grad is None at step 0"
    assert out_proj_grad0.norm().item() > 0, "FAIL: out_proj.weight.grad is zero at step 0"
    print(f"\n  out_proj.weight.grad.norm() = {out_proj_grad0.norm().item():.6f} (should be > 0)")
    print("  -> Step 0 OK: out_proj gets gradient, proving forward path is connected")

    # q/k/v and dual_gate should be zero (out_proj blocks gradient to upstream)
    # This is expected and correct behavior
    q_grad0 = cm_model.cross_attn_modules[0].q_proj.weight.grad
    print(f"  q_proj.weight.grad.norm() = {q_grad0.norm().item():.6f} (expected 0 — out_proj=0 blocks upstream)")
    dg_grad0 = cm_model.dual_gate_proj_new.weight.grad
    print(f"  dual_gate.grad.norm() = {dg_grad0.norm().item():.6f} (expected 0 — no read-back path yet)")

    # ================================================================
    # OPTIMIZER STEP: Make out_proj nonzero
    # ================================================================
    optimizer.step()
    out_proj_norm = cm_model.cross_attn_modules[0].out_proj.weight.norm().item()
    print(f"\n--- After optimizer step: out_proj.weight.norm() = {out_proj_norm:.6f} (should be > 0)")

    # ================================================================
    # STEP 1: Second forward-backward (out_proj != 0, ALL grads should flow)
    # ================================================================
    optimizer.zero_grad()
    cm_model.reset_slots()

    print(f"\n--- Step 1: Forward (out_proj != 0, full gradient flow expected) ---")
    result = cm_model.forward_chunk(input_ids, labels=labels, enable_write_grad=True)
    loss = result["loss"]
    print(f"  Loss: {loss.item():.4f}")
    loss.backward()

    zero1, none1 = check_gradients(cm_model, "Step 1 (out_proj != 0 → ALL should be nonzero)")

    # --- Key assertions for Step 1 ---
    print(f"\n{'=' * 60}")
    print("FINAL ASSERTIONS")
    print(f"{'=' * 60}")

    # 1. out_proj should have nonzero grad
    out_proj_grad1 = cm_model.cross_attn_modules[0].out_proj.weight.grad
    assert out_proj_grad1 is not None and out_proj_grad1.norm().item() > 0
    print(f"  cross_attn_modules[0].out_proj.weight.grad.norm() = {out_proj_grad1.norm().item():.6f} PASS")

    # 2. q_proj should have nonzero grad (proves gradient flows through out_proj → attn → Q)
    q_grad1 = cm_model.cross_attn_modules[0].q_proj.weight.grad
    assert q_grad1 is not None and q_grad1.norm().item() > 0, \
        f"FAIL: q_proj.weight.grad is {'None' if q_grad1 is None else f'zero (norm={q_grad1.norm().item()})'}. " \
        f"Gradient should flow: loss → hidden → residual_scale*memory_output → out_proj → attn → q_proj"
    print(f"  cross_attn_modules[0].q_proj.weight.grad.norm() = {q_grad1.norm().item():.6f} PASS")

    # 3. k_proj/v_proj should have nonzero grad (proves gradient flows through slots → K/V)
    k_grad1 = cm_model.cross_attn_modules[0].k_proj.weight.grad
    assert k_grad1 is not None and k_grad1.norm().item() > 0, \
        f"FAIL: k_proj grad is zero. This means slots (K/V) are detached — Bug #2 not fixed!"
    print(f"  cross_attn_modules[0].k_proj.weight.grad.norm() = {k_grad1.norm().item():.6f} PASS")

    v_grad1 = cm_model.cross_attn_modules[0].v_proj.weight.grad
    assert v_grad1 is not None and v_grad1.norm().item() > 0, \
        f"FAIL: v_proj grad is zero. This means slots (K/V) are detached — Bug #2 not fixed!"
    print(f"  cross_attn_modules[0].v_proj.weight.grad.norm() = {v_grad1.norm().item():.6f} PASS")

    # 4. dual_gate should have nonzero grad
    #    gradient path: loss → read_layer_cross_attn → slots → write_layer_output → dual_gate
    dg_new_grad1 = cm_model.dual_gate_proj_new.weight.grad
    dg_mem_grad1 = cm_model.dual_gate_proj_mem.weight.grad
    dg_bias_grad1 = cm_model.dual_gate_bias.grad

    assert dg_new_grad1 is not None and dg_new_grad1.norm().item() > 0, \
        f"FAIL: dual_gate_proj_new grad is zero. Gradient path from read layers back to write layer broken!"
    print(f"  dual_gate_proj_new.weight.grad.norm() = {dg_new_grad1.norm().item():.6f} PASS")

    assert dg_mem_grad1 is not None and dg_mem_grad1.norm().item() > 0, \
        f"FAIL: dual_gate_proj_mem grad is zero."
    print(f"  dual_gate_proj_mem.weight.grad.norm() = {dg_mem_grad1.norm().item():.6f} PASS")

    assert dg_bias_grad1 is not None and dg_bias_grad1.norm().item() > 0, \
        f"FAIL: dual_gate_bias grad is zero."
    print(f"  dual_gate_bias.grad.norm() = {dg_bias_grad1.norm().item():.6f} PASS")

    # 5. Full parameter scan after step 1
    print(f"\n--- Full scan (step 1): trainable params with grad=None or zero ---")
    bad_count = 0
    for name, param in cm_model.named_parameters():
        if not param.requires_grad:
            continue
        if param.grad is None:
            print(f"  NONE grad: {name}")
            bad_count += 1
        elif param.grad.norm().item() == 0.0:
            print(f"  ZERO grad: {name}")
            bad_count += 1
    if bad_count == 0:
        print("  (all trainable params have nonzero gradient)")

    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    print(f"\n{'=' * 60}")
    print("FINAL SUMMARY")
    print(f"{'=' * 60}")
    print(f"  cross_attn_modules built: {n_ca} (expected {expected_n_read}) - PASS")
    print(f"  Step 0: out_proj grad nonzero (proves forward path) - PASS")
    print(f"  Step 1: q/k/v_proj grad nonzero (gradient through cross-attn) - PASS")
    print(f"  Step 1: dual_gate grad nonzero (gradient from read→write) - PASS")
    if bad_count > 0:
        print(f"  WARNING: {bad_count} params still have zero grad (see above)")

    print(f"\n  *** ALL CRITICAL CHECKS PASSED ***")
    print(f"  *** H9 gradient flow is working correctly ***")


if __name__ == "__main__":
    main()
