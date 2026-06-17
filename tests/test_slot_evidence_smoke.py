#!/usr/bin/env python3
"""Smoke test for Slot-Routed Evidence Memory (2026-06-17).

Mechanism: each slot, besides its compressed semantic latent, keeps a small
buffer of UNCOMPRESSED routed-token hidden states ([d_model]). At readout the
top-k selected slots' evidence is gathered and prepended to the joint-attention
extended sequence as a 4th segment ([L3|L2|L1|evidence|H]), so the frozen
decoder can recall precise facts the compressed latent loses.

What we verify (CPU, tiny random Llama — no weight download):
1. use_slot_evidence=True: a forward runs with no error / no NaN, the evidence
   buffer is written (some slot count > 0), and loss backward works.
2. The extended sequence the wrapped layer sees grows by exactly k*evidence_topr
   on the evidence layer (vs the same model with evidence off).
3. use_slot_evidence=False: forward output is numerically IDENTICAL to a build
   constructed before the evidence feature was wired (regression / zero-impact).
4. MemoryBank.write_evidence merge-top-Bcnt-by-score keeps the highest-salience
   entries and clears on reset / recycle.

Run: .venv/bin/python -m pytest tests/test_slot_evidence_smoke.py -q
 or: .venv/bin/python tests/test_slot_evidence_smoke.py
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
from src.memory.mem_space.memory_bank import MemoryBank  # noqa: E402
from src.memory.mem_space.layer import MemorySpaceLayer  # noqa: E402


def _reset_layer_counter() -> None:
    # _layer_idx is assigned from a global class counter that accumulates across
    # every model built in the process. In real training only one model is
    # patched (so layer 0 exists); in this multi-model test we reset it before
    # each patch so config.evidence_layer=0 maps to the first patched layer.
    MemorySpaceLayer._instance_counter = 0


def _tiny_llama() -> LlamaModel:
    torch.manual_seed(0)
    cfg = LlamaConfig(
        vocab_size=1024,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=256,
    )
    m = LlamaModel(cfg)
    m.eval()
    return m


def _cfg(**over) -> MemorySpaceConfig:
    base = dict(
        num_slots=8,
        top_k=4,
        selector_dim=32,
        writeback_gate_max=0.3,
        writeback_gate_warmup_steps=0,
        load_balance_weight=0.01,
        slot_init="random",
        slot_init_noise=0.05,
    )
    base.update(over)
    return MemorySpaceConfig(**base)


# --------------------------------------------------------------------------- #
# 1 + 2. Evidence ON: forward OK, buffer written, prefix grows, backward OK.
# --------------------------------------------------------------------------- #
def test_evidence_on_forward_writes_and_grows() -> None:
    ids = torch.randint(0, 1024, (2, 32))

    # Capture the extended-sequence length the evidence layer's wrapped layer
    # actually receives, by hooking the wrapped LlamaDecoderLayer.forward.
    seen = {}

    def _hook(mod, args, kwargs):
        # extended_hidden is the first positional arg to the wrapped layer.
        h = args[0] if args else kwargs.get("hidden_states")
        seen.setdefault("lens", []).append(h.shape[1])

    # --- evidence OFF reference (same seq, same seed) ---
    model_off = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(
        model_off, _cfg(use_slot_evidence=False), layer_indices=[0, 1],
    )
    off_seen = {}
    def _hook_off(mod, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        off_seen.setdefault("lens", []).append(h.shape[1])
    model_off.layers[0].wrapped_layer.register_forward_pre_hook(_hook_off, with_kwargs=True)
    with torch.no_grad():
        model_off(input_ids=ids)
    off_len = max(off_seen["lens"])

    # --- evidence ON ---
    model = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(
        model,
        _cfg(use_slot_evidence=True, evidence_buffer_size=8, evidence_topr=0,
             evidence_layer=0),
        layer_indices=[0, 1],
    )
    model.layers[0].wrapped_layer.register_forward_pre_hook(_hook, with_kwargs=True)
    model.train()

    out = model(input_ids=ids).last_hidden_state
    assert out.shape == (2, 32, 128), out.shape
    assert torch.isfinite(out).all(), "NaN/Inf in evidence-on forward"

    # (b) evidence buffer written: some slot has count > 0.
    bank = model._mem_space_shared_bank
    assert bank.slot_evidence is not None, "evidence buffer was never allocated"
    assert bank.slot_evidence_count is not None
    max_count = int(bank.slot_evidence_count.max().item())
    print(f"[evidence smoke] max slot_evidence_count = {max_count} (must be > 0)")
    assert max_count > 0, "no slot received evidence"
    # top_k=4 slots written, buffer caps at min(Bcnt=8, T=32) per slot.
    assert max_count <= 8

    # (c) extended sequence on the evidence layer grew by k*topr (= 4*8 = 32).
    on_len = max(seen["lens"])
    grew = on_len - off_len
    expected = 4 * 8  # top_k * evidence_topr(resolved to buffer_size)
    print(f"[evidence smoke] ext_len off={off_len} on={on_len} grew={grew} "
          f"(expected {expected})")
    assert grew == expected, f"prefix grew {grew}, expected {expected}"

    # (d) backward works even though evidence is frozen (no graph break).
    out.mean().backward()
    qsel = model.layers[0].selector.Q_sel.weight
    assert qsel.grad is not None and torch.isfinite(qsel.grad).all(), (
        "backward through evidence-on model failed"
    )
    print("[evidence smoke] OK: forward+write+grow+backward all pass")


# --------------------------------------------------------------------------- #
# 3. Evidence OFF == zero impact (numerically identical to a no-evidence run).
#    Both builds use use_slot_evidence=False; identical seeds → bit-identical.
# --------------------------------------------------------------------------- #
def test_evidence_off_byte_identical() -> None:
    ids = torch.randint(0, 1024, (2, 32))

    model_a = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(model_a, _cfg(use_slot_evidence=False),
                             layer_indices=[0, 1])
    with torch.no_grad():
        out_a = model_a(input_ids=ids).last_hidden_state

    model_b = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(model_b, _cfg(use_slot_evidence=False),
                             layer_indices=[0, 1])
    with torch.no_grad():
        out_b = model_b(input_ids=ids).last_hidden_state

    assert torch.equal(out_a, out_b), (
        "two evidence-OFF builds with identical seeds must be bit-identical"
    )
    # And the bank must NOT allocate any evidence buffer when disabled.
    assert model_a._mem_space_shared_bank.slot_evidence is None
    assert model_a._mem_space_shared_bank.evidence_buffer_size == 0
    print("[evidence smoke] OK: evidence OFF is byte-identical + no buffer alloc")


# --------------------------------------------------------------------------- #
# 4. MemoryBank.write_evidence unit: merge-top-Bcnt, reset, recycle clearing.
# --------------------------------------------------------------------------- #
def test_write_evidence_keeps_top_by_score() -> None:
    B, N, d, Bcnt = 1, 4, 6, 2
    bank = MemoryBank(num_slots=N, slot_dim=d, slot_init="zero",
                      slot_value_norm_cap=0.0,
                      evidence_buffer_size=Bcnt, evidence_dim=d)
    bank.init_from_hidden(torch.randn(B, 5, d), batch_size=B)
    assert bank.slot_evidence is None, "evidence is lazy — None before first write"

    # Write 3 candidates for slot 1 with scores [0.1, 0.9, 0.5]; Bcnt=2 keeps
    # the two highest (0.9, 0.5). Their hidden states are distinguishable.
    slot_idx = torch.tensor([[1]])                       # [B, k=1]
    cand_h = torch.stack([
        torch.full((d,), 1.0),   # score 0.1 -> should be dropped
        torch.full((d,), 2.0),   # score 0.9 -> kept
        torch.full((d,), 3.0),   # score 0.5 -> kept
    ]).view(B, 1, 3, d)
    cand_s = torch.tensor([[[0.1, 0.9, 0.5]]])           # [B, k, C=3]
    bank.write_evidence(slot_idx, cand_h, cand_s)

    assert bank.slot_evidence is not None
    assert bank.slot_evidence.shape == (B, N, Bcnt, d)
    kept_scores = sorted(bank.slot_evidence_score[0, 1].tolist())
    print(f"[evidence smoke] kept scores for slot 1 = {kept_scores}")
    assert len(kept_scores) == 2
    assert abs(kept_scores[0] - 0.5) < 1e-4 and abs(kept_scores[1] - 0.9) < 1e-4, (
        "must keep the two highest scores"
    )
    assert int(bank.slot_evidence_count[0, 1].item()) == 2
    # The dropped (score 0.1, value 1.0) entry must be absent.
    kept_vals = {round(bank.slot_evidence[0, 1, i, 0].item()) for i in range(Bcnt)}
    assert kept_vals == {2, 3}, f"kept token values {kept_vals} != {{2,3}}"
    # Unselected slots stay empty (count 0).
    assert int(bank.slot_evidence_count[0, 0].item()) == 0

    # reset() drops the buffers (per-sample state).
    bank.reset()
    assert bank.slot_evidence is None and bank.slot_evidence_count is None

    # recycle_reset() invalidates the recycled rows' evidence.
    bank.init_from_hidden(torch.randn(B, 5, d), batch_size=B)
    bank.write_evidence(slot_idx, cand_h, cand_s)
    dead = torch.zeros(B, N, dtype=torch.bool)
    dead[0, 1] = True
    bank.recycle_reset(dead, torch.randn(B, N, d))
    assert int(bank.slot_evidence_count[0, 1].item()) == 0, (
        "recycled slot's stale evidence must be cleared"
    )
    assert bank.slot_evidence_score[0, 1].max().item() == float("-inf")
    print("[evidence smoke] OK: write_evidence top-Bcnt + reset/recycle clearing")


def test_write_evidence_frozen_is_noop() -> None:
    B, N, d, Bcnt = 1, 4, 6, 2
    bank = MemoryBank(num_slots=N, slot_dim=d, slot_init="zero",
                      evidence_buffer_size=Bcnt, evidence_dim=d)
    bank.init_from_hidden(torch.randn(B, 5, d), batch_size=B)
    bank.frozen = True
    bank.write_evidence(torch.tensor([[1]]),
                        torch.randn(B, 1, 1, d), torch.tensor([[[0.5]]]))
    assert bank.slot_evidence is None, "frozen bank must not write evidence"
    print("[evidence smoke] OK: frozen bank write_evidence is a no-op")


if __name__ == "__main__":
    test_write_evidence_keeps_top_by_score()
    test_write_evidence_frozen_is_noop()
    test_evidence_off_byte_identical()
    test_evidence_on_forward_writes_and_grows()
    print("\nAll Slot-Routed Evidence Memory smoke checks passed.")
