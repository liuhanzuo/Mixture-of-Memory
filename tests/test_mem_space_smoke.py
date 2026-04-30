#!/usr/bin/env python3
"""Smoke test for src/memory/mem_space/ — the Memory-Space v0 prototype.

Design reference:
    ops/research_notes/20260426_memory_space_design_direction.md

What we verify
--------------
1. A tiny random Llama model (2 layers, hidden=128) can be patched with a
   MemorySpaceLayer on layer 0 and still run a forward pass.
2. Output shape matches the un-patched baseline.
3. No NaN / Inf in outputs.
4. Slots have the expected shape [B, N, slot_dim].
5. With enable_writeback=True the slots change after a forward (the in-place
   EMA writeback is doing something).
6. With enable_writeback=False the slots stay pinned to their init.
7. The load-balance aux loss is a finite scalar.
8. Gradients flow into the selector projections (backward on the mean of
   the output populates Q_sel.weight.grad).
9. `forward_no_memory` reproduces a vanilla LlamaDecoderLayer output within
   fp32 tolerance.

Runs on CPU only — tensors are tiny enough that no GPU is needed.
"""
from __future__ import annotations

import copy
import os
import sys

import torch

# Make `src` importable when running from repo root or anywhere.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import LlamaConfig, LlamaModel                  # noqa: E402

from src.memory.mem_space import (                                 # noqa: E402
    MemorySpaceConfig,
    MemorySpaceLayer,
    apply_mem_space_to_model,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _tiny_llama_config() -> LlamaConfig:
    """Instantiate a small Llama config — NO pretrained weight download."""
    return LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=128,
    )


def _build_tiny_model() -> LlamaModel:
    torch.manual_seed(0)
    cfg = _tiny_llama_config()
    model = LlamaModel(cfg)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #


def test_patched_forward_shape_and_slots() -> None:
    """(1)+(2)+(3)+(4) forward shape, finiteness, slots shape."""
    model = _build_tiny_model()

    ms_cfg = MemorySpaceConfig(
        num_slots=8,
        top_k=2,
        selector_dim=32,
        writeback_gate_max=0.3,
        writeback_gate_warmup_steps=0,   # β active from step 0
        load_balance_weight=0.01,
    )
    _, mem_layers = apply_mem_space_to_model(model, ms_cfg, layer_indices=[0])
    assert len(mem_layers) == 1
    assert isinstance(model.layers[0], MemorySpaceLayer)

    # Pump the warmup at least once so β > 0 (warmup_steps=0 means β = σ(0)
    # = 0.5 · gate_max  = 0.15 immediately — still want to exercise step).
    mem_layers[0].step_counter = 1

    B, T = 2, 16
    ids = torch.randint(0, 1024, (B, T))
    out = model(ids)
    last = out.last_hidden_state
    assert last.shape == (B, T, 128), f"unexpected output shape: {last.shape}"
    assert torch.isfinite(last).all(), "non-finite values in output"

    slots = mem_layers[0].memory_bank.slots
    assert slots is not None, "slots were not initialised"
    assert slots.shape == (B, 8, 128), f"unexpected slots shape: {slots.shape}"
    assert torch.isfinite(slots).all(), "non-finite values in slots"


def test_writeback_enabled_mutates_slots() -> None:
    """(5) enable_writeback=True — slots differ between consecutive forwards."""
    model = _build_tiny_model()
    ms_cfg = MemorySpaceConfig(
        num_slots=8,
        top_k=2,
        selector_dim=32,
        enable_writeback=True,
        writeback_gate_init=4.0,         # σ(4) ≈ 0.98 → big β to make test robust
        writeback_gate_warmup_steps=0,
        writeback_gate_max=0.3,
    )
    _, mem_layers = apply_mem_space_to_model(model, ms_cfg, layer_indices=[0])
    mem_layers[0].step_counter = 1

    B, T = 2, 16
    ids = torch.randint(0, 1024, (B, T))
    model(ids)
    slots_after_first = mem_layers[0].memory_bank.slots.clone()

    # Same input again — writeback should move the *selected* slots further.
    model(ids)
    slots_after_second = mem_layers[0].memory_bank.slots.clone()

    assert not torch.allclose(slots_after_first, slots_after_second, atol=1e-6), (
        "slots did not change between forwards despite enable_writeback=True"
    )


def test_writeback_disabled_freezes_slots() -> None:
    """(6) enable_writeback=False — slots equal init value across forwards."""
    model = _build_tiny_model()
    ms_cfg = MemorySpaceConfig(
        num_slots=8,
        top_k=2,
        selector_dim=32,
        enable_writeback=False,
        writeback_gate_warmup_steps=0,
    )
    _, mem_layers = apply_mem_space_to_model(model, ms_cfg, layer_indices=[0])

    B, T = 2, 16
    ids = torch.randint(0, 1024, (B, T))
    model(ids)
    slots_after_first = mem_layers[0].memory_bank.slots.clone()

    model(ids)
    slots_after_second = mem_layers[0].memory_bank.slots.clone()

    assert torch.equal(slots_after_first, slots_after_second), (
        "slots mutated despite enable_writeback=False"
    )


def test_load_balance_loss_is_finite_scalar() -> None:
    """(7) load-balance aux loss is a finite scalar after forward."""
    model = _build_tiny_model()
    ms_cfg = MemorySpaceConfig(
        num_slots=8, top_k=2, selector_dim=32,
        writeback_gate_warmup_steps=0, return_aux_losses=True,
    )
    _, mem_layers = apply_mem_space_to_model(model, ms_cfg, layer_indices=[0])
    ids = torch.randint(0, 1024, (2, 16))
    model(ids)
    aux = mem_layers[0].last_aux_losses
    assert "load_balance" in aux, "expected load_balance key in aux"
    lb = aux["load_balance"]
    assert lb.dim() == 0, f"expected scalar, got shape {lb.shape}"
    assert torch.isfinite(lb), f"load_balance not finite: {lb.item()}"


def test_selector_receives_gradient() -> None:
    """(8) backward on output mean → Q_sel.weight.grad is populated."""
    model = _build_tiny_model()
    model.train()
    ms_cfg = MemorySpaceConfig(
        num_slots=8, top_k=2, selector_dim=32,
        writeback_gate_warmup_steps=0, enable_writeback=True,
    )
    _, mem_layers = apply_mem_space_to_model(model, ms_cfg, layer_indices=[0])

    ids = torch.randint(0, 1024, (2, 16))
    out = model(ids).last_hidden_state
    loss = out.mean()
    loss.backward()

    q_grad = mem_layers[0].selector.Q_sel.weight.grad
    k_grad = mem_layers[0].selector.K_sel.weight.grad
    assert q_grad is not None, "Q_sel.weight.grad is None (no gradient reached selector)"
    assert k_grad is not None, "K_sel.weight.grad is None (no gradient reached selector)"
    assert torch.isfinite(q_grad).all(), "Q_sel.weight.grad contains non-finite values"
    # Some Q_sel rows must have non-zero gradient, otherwise the selector is
    # silently disconnected from the graph.
    assert q_grad.abs().sum() > 0, "Q_sel.weight.grad is entirely zero"


def test_forward_no_memory_matches_baseline() -> None:
    """(9) forward_no_memory reproduces the vanilla LlamaDecoderLayer output."""
    model_a = _build_tiny_model()
    model_b = copy.deepcopy(model_a)

    # Patch model_b on layer 0 then test bypass.
    ms_cfg = MemorySpaceConfig(
        num_slots=8, top_k=2, selector_dim=32,
        writeback_gate_warmup_steps=0, enable_writeback=False,
    )
    _, mem_layers = apply_mem_space_to_model(model_b, ms_cfg, layer_indices=[0])

    # Swap model_b's layer-0 forward for the bypass so the outer LlamaModel
    # gets the baseline behaviour.
    wrapper = mem_layers[0]
    original_forward = wrapper.forward
    wrapper.forward = wrapper.forward_no_memory  # type: ignore[assignment]

    try:
        ids = torch.randint(0, 1024, (2, 16))
        with torch.no_grad():
            out_a = model_a(ids).last_hidden_state
            out_b = model_b(ids).last_hidden_state
    finally:
        wrapper.forward = original_forward  # type: ignore[assignment]

    # They should be *identical* because the wrapped layer parameters were
    # deep-copied from the same baseline.
    assert torch.allclose(out_a, out_b, atol=1e-5, rtol=1e-5), (
        f"forward_no_memory diverges from baseline: "
        f"max|a-b|={ (out_a - out_b).abs().max().item():.3e}"
    )


# --------------------------------------------------------------------------- #
# Entrypoint
# --------------------------------------------------------------------------- #


def _run_all() -> None:
    print("=== mem_space smoke test ===")
    tests = [
        test_patched_forward_shape_and_slots,
        test_writeback_enabled_mutates_slots,
        test_writeback_disabled_freezes_slots,
        test_load_balance_loss_is_finite_scalar,
        test_selector_receives_gradient,
        test_forward_no_memory_matches_baseline,
    ]
    for t in tests:
        print(f"  running {t.__name__}...", flush=True)
        t()
        print(f"    OK")
    print("=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    _run_all()
