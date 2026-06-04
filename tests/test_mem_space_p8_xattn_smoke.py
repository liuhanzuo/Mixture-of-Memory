#!/usr/bin/env python3
"""P8 smoke test for the dedicated memory cross-attention READ path.

status/MEMORY_PROTOCOL_PLAN.md [P8]: give the memory slots a DEDICATED
cross-attention read with its OWN softmax + a per-head content-dependent gate
(active at init), gated behind --use_memory_xattn.

What we verify (CPU, tiny random Llama — no weight download):
1. A model patched with use_memory_xattn=True runs a forward and produces the
   right shape with no NaN/Inf.
2. The memory_xattn module is constructed (and absent when the flag is off).
3. The read path is ACTIVE at init: the patched forward differs from the same
   model run with memory disabled (i.e. the gate is NOT ≈0 like the P2
   zero-init path). This is the whole point of P8.
4. Gradients flow into the cross-attn read params (q_proj / out_proj / gate)
   AND still into the selector — confirming memory gets gradient from step 0.
5. With use_memory_xattn=False the module is None (byte-for-byte legacy guard).

Run: .venv/bin/python -m pytest tests/test_mem_space_p8_xattn_smoke.py -q
 or: .venv/bin/python tests/test_mem_space_p8_xattn_smoke.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from transformers import LlamaConfig, LlamaModel  # noqa: E402

from src.memory.mem_space import (  # noqa: E402
    MemorySpaceConfig,
    apply_mem_space_to_model,
)


def _tiny_llama() -> LlamaModel:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,   # GQA — exercises _repeat_kv
        max_position_embeddings=256,
    )
    m = LlamaModel(cfg)
    m.eval()
    return m


def _cfg(**over) -> MemorySpaceConfig:
    base = dict(
        num_slots=8,
        top_k=2,
        selector_dim=32,
        writeback_gate_max=0.3,
        writeback_gate_warmup_steps=0,
        load_balance_weight=0.01,
        slot_init="random",
        slot_init_noise=0.05,
    )
    base.update(over)
    return MemorySpaceConfig(**base)


def test_xattn_off_module_absent() -> None:
    model = _tiny_llama()
    apply_mem_space_to_model(model, _cfg(use_memory_xattn=False), layer_indices=[0])
    layer = model.layers[0]
    assert getattr(layer, "memory_xattn", None) is None
    print("[P8 smoke] OK: memory_xattn is None when flag off")


def test_xattn_on_forward_shape_and_active() -> None:
    ids = torch.randint(0, 1024, (2, 32))

    # Memory-on (xattn) model.
    model = _tiny_llama()
    apply_mem_space_to_model(
        model, _cfg(use_memory_xattn=True, memory_xattn_gate_init=0.4),
        layer_indices=[0],
    )
    layer = model.layers[0]
    assert layer.memory_xattn is not None, "memory_xattn module not constructed"
    # GQA head counts threaded from the model config.
    assert layer.memory_xattn.n_heads == 8
    assert layer.memory_xattn.n_kv_heads == 4

    out = model(input_ids=ids).last_hidden_state
    assert out.shape == (2, 32, 128), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf in xattn forward output"

    # Active-at-init check: disable memory and compare. If the gate were ≈0
    # (the P2 zero-init failure mode), these would be ~identical.
    layer._memory_disabled = True
    out_nomem = model(input_ids=ids).last_hidden_state
    layer._memory_disabled = False
    delta = (out - out_nomem).abs().mean().item()
    print(f"[P8 smoke] mean|xattn - no_mem| = {delta:.6e} (must be > 0 at init)")
    assert delta > 1e-5, (
        "P8 read path is inactive at init — gate/out_proj effectively zero, "
        "which defeats the purpose of P8 (gradient must flow through memory)."
    )
    print(f"[P8 smoke] OK: forward shape {tuple(out.shape)}, read active at init")


def test_xattn_gradients_flow() -> None:
    ids = torch.randint(0, 1024, (2, 32))
    model = _tiny_llama()
    # Patch BOTH layers + shared bank (the realistic 32-layer config). The
    # selector's gradient path under a decoupled/xattn read goes through the
    # cross-layer write->read chain (layer i writes a slot that layer i+1
    # reads), so it only flows when >1 layer is patched — matching the actual
    # training. With a SINGLE patched layer the H->L1 mask severs the only
    # in-layer STE path and the selector gets 0 grad (true for P2 too); that is
    # expected, not a P8 regression.
    apply_mem_space_to_model(
        model, _cfg(use_memory_xattn=True, shared_memory_bank=True),
        layer_indices=[0, 1],
    )
    model.train()
    layer = model.layers[0]

    out = model(input_ids=ids).last_hidden_state
    out.mean().backward()

    mx = layer.memory_xattn
    for name, p in [
        ("q_proj", mx.q_proj.weight),
        ("k_proj", mx.k_proj.weight),
        ("v_proj", mx.v_proj.weight),
        ("out_proj", mx.out_proj.weight),
        ("gate_proj", mx.gate_proj.weight),
    ]:
        assert p.grad is not None, f"memory_xattn.{name} got no gradient"
        assert torch.isfinite(p.grad).all(), f"memory_xattn.{name} grad non-finite"
        assert p.grad.abs().sum().item() > 0, f"memory_xattn.{name} grad all-zero"

    # Selector must STILL get gradient (routing path untouched by P8) when the
    # cross-layer chain exists.
    qsel = layer.selector.Q_sel.weight
    assert qsel.grad is not None and qsel.grad.abs().sum().item() > 0, (
        "selector Q_sel got no gradient under P8 (multi-layer)"
    )
    print("[P8 smoke] OK: gradients flow into xattn read params + selector")


if __name__ == "__main__":
    test_xattn_off_module_absent()
    test_xattn_on_forward_shape_and_active()
    test_xattn_gradients_flow()
    print("\nAll P8 xattn smoke checks passed.")
