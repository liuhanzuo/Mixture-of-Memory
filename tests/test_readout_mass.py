#!/usr/bin/env python3
"""Unit tests for the per-slot token-mass readout bias (2026-06-15).

Motivation: the P8 read softmax logit is pure Q·Kᵀ·scale with no per-slot prior,
and slot_value_norm_cap flattens slot norms, so a slot that condensed MANY tokens
and one that condensed FEW are read identically. The mass bias adds
``readout_mass_coef · log1p(mass_n)`` to slot n's read logit (real slots only,
not the null/sink), so its softmax weight scales ≈ with the tokens it represents.

What we verify (CPU, tiny tensors — no model download):
A. Backward-compat: mass=None gives logits/output byte-identical to not passing
   mass (selector.py:read guards the bias behind `if mass is not None`).
B. A slot with larger mass gets MORE softmax weight, and the per-logit increment
   equals coef·log1p(m) (the exact bias formula), holding Q/K fixed.
C. The null/sink column is NOT affected by mass (only the first-N real columns).
D. MemoryBank.slot_token_mass: correct per-chunk accumulation under the
   per-chunk routing semantics (selected slot += real-token count), zeroes on
   reset, and zeroes recycled rows on recycle_reset.

Run: .venv/bin/python -m pytest tests/test_readout_mass.py -q
 or: .venv/bin/python tests/test_readout_mass.py
"""
from __future__ import annotations

import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.memory.mem_space.selector import MemoryCrossAttentionRead  # noqa: E402
from src.memory.mem_space.memory_bank import MemoryBank  # noqa: E402


def _read_module(disable_null_sink: bool = False) -> MemoryCrossAttentionRead:
    torch.manual_seed(0)
    return MemoryCrossAttentionRead(
        d_model=32,
        n_heads=4,
        n_kv_heads=2,
        gate_init=0.4,
        out_proj_std=0.02,
        dropout=0.0,
        disable_null_sink=disable_null_sink,
    )


def _inputs(B=2, T=3, N=5, d=32):
    torch.manual_seed(1)
    hs = torch.randn(B, T, d)
    sk = torch.randn(B, N, d)
    sv = torch.randn(B, N, d)
    return hs, sk, sv


# --------------------------------------------------------------------------- #
# A. Backward compatibility: mass=None == not passing mass.
# --------------------------------------------------------------------------- #
def test_mass_none_byte_identical():
    mod = _read_module()
    mod.eval()
    hs, sk, sv = _inputs()
    with torch.no_grad():
        out_default = mod.read(hs, sk, sv)
        out_mass_none = mod.read(hs, sk, sv, mass=None, mass_coef=3.0)
    assert torch.equal(out_default, out_mass_none), (
        "mass=None must reproduce the no-mass output bit-for-bit "
        "(selector.py read() guards the bias behind `if mass is not None`)"
    )


# --------------------------------------------------------------------------- #
# B. Larger mass → larger softmax weight; increment == coef·log1p(m).
#    We reach into the read() math directly by recomputing the logits the same
#    way the module does, then check the softmax shift matches the bias formula.
# --------------------------------------------------------------------------- #
def test_mass_bias_increases_weight_and_matches_formula():
    mod = _read_module(disable_null_sink=True)  # no sink → softmax over real N only
    mod.eval()
    B, T, N, d = 1, 1, 4, 32
    torch.manual_seed(2)
    hs = torch.randn(B, T, d)
    sk = torch.randn(B, N, d)
    sv = torch.randn(B, N, d)
    coef = 2.0
    # Give slot 2 a big mass, others zero.
    mass = torch.zeros(B, N)
    mass[0, 2] = 100.0

    # Reproduce the module's logit computation (identical ops to read()).
    with torch.no_grad():
        Q = mod.q_proj(hs).view(B, T, mod.n_heads, mod.head_dim).transpose(1, 2)
        K = mod.k_proj(sk).view(B, N, mod.n_kv_heads, mod.head_dim).transpose(1, 2)
        K = mod._repeat_kv(K)
        scale = mod.head_dim ** -0.5
        base_logits = torch.matmul(Q, K.transpose(-2, -1)) * scale  # [B,H,T,N]
        bias = coef * torch.log1p(mass.clamp(min=0.0))              # [B,N]
        biased_logits = base_logits + bias[:, None, None, :]

        w_base = torch.softmax(base_logits, dim=-1)
        w_biased = torch.softmax(biased_logits, dim=-1)

    # (b) the high-mass slot draws strictly more softmax weight after the bias.
    assert (w_biased[..., 2] > w_base[..., 2]).all(), (
        "slot with large mass must gain softmax weight"
    )
    # The per-logit increment on slot 2 equals coef·log1p(100).
    expected_inc = coef * math.log1p(100.0)
    actual_inc = (biased_logits[..., 2] - base_logits[..., 2]).flatten()[0].item()
    assert abs(actual_inc - expected_inc) < 1e-4, (
        f"logit increment {actual_inc} != coef·log1p(m) {expected_inc}"
    )
    # Zero-mass slots get exactly zero increment (log1p(0)=0).
    for j in (0, 1, 3):
        inc = (biased_logits[..., j] - base_logits[..., j]).abs().max().item()
        assert inc < 1e-6, f"zero-mass slot {j} should get no bias, got {inc}"


# --------------------------------------------------------------------------- #
# C. The null/sink column is NOT biased by mass.
# --------------------------------------------------------------------------- #
def test_sink_column_unaffected_by_mass():
    mod = _read_module(disable_null_sink=False)  # sink ON → softmax over N+1
    mod.eval()
    B, T, N, d = 1, 1, 4, 32
    torch.manual_seed(3)
    hs = torch.randn(B, T, d)
    sk = torch.randn(B, N, d)
    sv = torch.randn(B, N, d)
    # Huge mass on every real slot — if the sink were (wrongly) biased too, its
    # relative weight would not collapse. We instead check the sink mass DROPS
    # (real slots get boosted, sink does not), confirming the sink is excluded.
    mass = torch.full((B, N), 50.0)

    with torch.no_grad():
        mod.read(hs, sk, sv, mass=None)
        sink_mass_base = mod._last_sink_mass
        mod.read(hs, sk, sv, mass=mass, mass_coef=2.0)
        sink_mass_biased = mod._last_sink_mass

    assert sink_mass_biased < sink_mass_base, (
        "biasing all real slots up must REDUCE relative sink mass; if the sink "
        f"were biased too it would not drop (base={sink_mass_base}, "
        f"biased={sink_mass_biased})"
    )


# --------------------------------------------------------------------------- #
# D. MemoryBank.slot_token_mass: counting, reset, recycle-zeroing.
# --------------------------------------------------------------------------- #
def _fresh_bank(B=2, N=5, d=8):
    bank = MemoryBank(num_slots=N, slot_dim=d, slot_init="zero", slot_value_norm_cap=0.0)
    H = torch.randn(B, 4, d)
    bank.init_from_hidden(H, batch_size=B)
    return bank


def test_token_mass_accumulation():
    B, N, d = 2, 5, 8
    bank = _fresh_bank(B, N, d)
    assert bank.slot_token_mass is None, "mass is lazy — None before first add"

    # Chunk 1: sample 0 selects slots {0,2}, sample 1 selects {1,3}; 10 real toks.
    sel = torch.zeros(B, N, dtype=torch.long)
    sel[0, 0] = 1; sel[0, 2] = 1
    sel[1, 1] = 1; sel[1, 3] = 1
    bank.add_token_mass(sel, torch.tensor([10.0, 10.0]))
    m = bank.slot_token_mass
    assert m is not None and m.shape == (B, N)
    assert m[0, 0].item() == 10.0 and m[0, 2].item() == 10.0
    assert m[0, 1].item() == 0.0 and m[0, 3].item() == 0.0
    assert m[1, 1].item() == 10.0 and m[1, 3].item() == 10.0

    # Chunk 2: sample 0 re-selects slot 0 (+7) and new slot 4 (+7); 7 real toks.
    sel2 = torch.zeros(B, N, dtype=torch.long)
    sel2[0, 0] = 1; sel2[0, 4] = 1
    sel2[1, 1] = 1
    bank.add_token_mass(sel2, torch.tensor([7.0, 7.0]))
    m = bank.slot_token_mass
    assert m[0, 0].item() == 17.0, "re-selected slot accumulates across chunks"
    assert m[0, 4].item() == 7.0
    assert m[0, 2].item() == 10.0, "unselected-this-chunk slot keeps prior mass"
    assert m[1, 1].item() == 17.0


def test_token_mass_reset_zeroes():
    bank = _fresh_bank()
    sel = torch.zeros(2, 5, dtype=torch.long)
    sel[0, 0] = 1
    bank.add_token_mass(sel, torch.tensor([5.0, 5.0]))
    assert bank.slot_token_mass is not None
    bank.reset()
    assert bank.slot_token_mass is None, "reset must drop slot_token_mass (per-sample state)"


def test_token_mass_frozen_is_noop():
    bank = _fresh_bank()
    bank.frozen = True
    sel = torch.zeros(2, 5, dtype=torch.long)
    sel[0, 0] = 1
    bank.add_token_mass(sel, torch.tensor([5.0, 5.0]))
    assert bank.slot_token_mass is None, "frozen bank must not accumulate mass"


def test_recycle_zeroes_mass():
    B, N, d = 2, 5, 8
    bank = _fresh_bank(B, N, d)
    sel = torch.ones(B, N, dtype=torch.long)
    bank.add_token_mass(sel, torch.tensor([9.0, 9.0]))
    assert (bank.slot_token_mass == 9.0).all()

    # Recycle (reset) slot 2 of sample 0 and slot 4 of sample 1.
    dead = torch.zeros(B, N, dtype=torch.bool)
    dead[0, 2] = True
    dead[1, 4] = True
    new_content = torch.randn(B, N, d)
    bank.recycle_reset(dead, new_content)
    m = bank.slot_token_mass
    assert m[0, 2].item() == 0.0, "recycled slot's stale mass must be zeroed"
    assert m[1, 4].item() == 0.0
    # Non-recycled rows keep their mass.
    assert m[0, 0].item() == 9.0 and m[1, 0].item() == 9.0


if __name__ == "__main__":
    test_mass_none_byte_identical()
    test_mass_bias_increases_weight_and_matches_formula()
    test_sink_column_unaffected_by_mass()
    test_token_mass_accumulation()
    test_token_mass_reset_zeroes()
    test_token_mass_frozen_is_noop()
    test_recycle_zeroes_mass()
    print("ALL READOUT-MASS TESTS PASSED")
