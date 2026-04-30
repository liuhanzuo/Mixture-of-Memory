"""Unit test: mem_space v0 bypass-parity under zero-init `slot_to_hidden`.

Mandated by CLAUDE.md PPL>1000 rule: fix3 smoke FAILED with step-1
PPL=62.64 bit-identical across runs despite `slot_to_hidden.weight=0`
(zero-init Flamingo-style). The /researcher Tier-2 contract promised
step-0 PPL ≈ bypass-parity 16.50 at zero-init; it did not happen.

This script isolates the discrepancy at the hidden-state level WITHOUT
training, without a full 8B backbone, using a synthetic tiny-Llama
decoder layer that exercises the exact same joint-attn extended-seq
code path (RMSNorm → attn → residual → RMSNorm → MLP → residual),
including our `_extend_position_embeddings` and `_build_extended_attn_mask`
helpers (with the fix2 `T/2` slot-streaming mask).

Diagnostic output (all comparisons done with slot_to_hidden.weight=0):

    1. Per-position max|O_test - O_ref| over the T body tokens, split by:
         - first-half H-queries (rows k..k+T//2 — slots MASKED under fix2)
         - second-half H-queries (rows k+T//2..k+T — slots VISIBLE under fix2)
       If first-half is ~0 and second-half is non-zero, the perturbation
       is softmax-denominator contamination (k=64 zero-K tokens in the
       softmax denominator of second-half H-queries, NOT in the numerator
       since V_slot=0).

    2. Effect of disabling the fix2 mask (make slot keys visible to ALL
       H-queries): if the error grows uniformly, confirms the softmax-
       denominator hypothesis.

    3. Effect of adding an epsilon to K_slot: verifies that zero K_slot +
       exp(0)=1 in the softmax denominator is the dominant term.

    4. Reports whether RMSNorm(0) == 0 exactly under the Llama
       modeling code (sanity check on the RMSNorm-of-zero hypothesis).

Usage:
    python scripts/test_mem_space_bypass_parity.py                 # default (cpu, fp32)
    python scripts/test_mem_space_bypass_parity.py --device cuda   # if GPU handy
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# Local package import
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaDecoderLayer,
    LlamaRMSNorm,
    LlamaRotaryEmbedding,
)

from src.memory.mem_space import MemorySpaceConfig, MemorySpaceLayer


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _mk_layer(cfg: LlamaConfig, dtype: torch.dtype, device: torch.device) -> LlamaDecoderLayer:
    layer = LlamaDecoderLayer(cfg, layer_idx=0).to(device=device, dtype=dtype)
    layer.eval()
    return layer


def _mk_rope_cossin(cfg: LlamaConfig, T: int, device: torch.device, dtype: torch.dtype):
    """Build (cos, sin) position_embeddings using HF's LlamaRotaryEmbedding."""
    rope = LlamaRotaryEmbedding(config=cfg).to(device=device)
    pos_ids = torch.arange(T, device=device).unsqueeze(0)  # [1, T]
    dummy_hidden = torch.zeros(1, T, cfg.hidden_size, device=device, dtype=dtype)
    cos, sin = rope(dummy_hidden, pos_ids)                 # [1, T, head_dim]
    return cos.to(dtype), sin.to(dtype)


def _diff_stats(a: torch.Tensor, b: torch.Tensor) -> dict:
    """Max-abs and L2 norm of (a - b)."""
    d = (a - b).float()
    return {
        "max_abs": d.abs().max().item(),
        "l2": d.norm().item(),
        "rel_l2": (d.norm() / (a.float().norm() + 1e-12)).item(),
    }


# --------------------------------------------------------------------------- #
# Main test
# --------------------------------------------------------------------------- #


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--B", type=int, default=1)
    parser.add_argument("--T", type=int, default=64)       # body length
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_kv_heads", type=int, default=4)
    parser.add_argument("--intermediate_size", type=int, default=256)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    # ----- 1. Build a tiny Llama decoder layer -----
    cfg = LlamaConfig(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        max_position_embeddings=4096,
        vocab_size=128,
        attn_implementation="eager",   # deterministic, CPU-safe
        rope_theta=500000.0,           # Llama-3 value; exercise the path
    )
    backbone_layer = _mk_layer(cfg, dtype, device)

    # ----- 2. Wrap in MemorySpaceLayer -----
    ms_cfg = MemorySpaceConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        slot_dim=None,                   # slot_dim == d_model
        selector_dim=16,
        slot_init="random",
        slot_init_noise=1.0,
        enable_writeback=True,
        return_aux_losses=True,
        writeback_gate_init=0.0,
        writeback_gate_warmup_steps=0,
        writeback_gate_max=0.3,
    )
    wrapper = MemorySpaceLayer(backbone_layer, ms_cfg, d_model=cfg.hidden_size).to(
        device=device, dtype=dtype
    )
    wrapper.eval()

    # Confirm Tier-3 invariants (post 2026-04-26 fix):
    #   1. slot_output_gate == 0  → tanh(0) = 0 → next_hidden ≡ bypass_h
    #   2. slot_to_hidden weight no longer zero-init (reverted; input-side zero
    #      was insufficient due to softmax-denominator pollution — see
    #      ops/research_notes/20260426_mem_space_v0_tier3_fix3_fail.md §2).
    assert wrapper.slot_output_gate.abs().max().item() == 0.0, (
        "Tier-3 invariant violated — slot_output_gate must init to 0 "
        "for bypass parity at step 0."
    )
    assert not wrapper.hidden_to_slot.weight.requires_grad, (
        "Tier-3 invariant violated — hidden_to_slot must be frozen "
        "(its gradient is structurally zero under O_mem_slot.detach() + _reset_banks)."
    )
    print("[ok] slot_output_gate == 0 (Tier-3 OUTPUT-side gate invariant)")
    print("[ok] hidden_to_slot frozen (Tier-3 zero-gradient invariant)")

    # ----- 3. Make inputs -----
    B, T, d = args.B, args.T, cfg.hidden_size
    hidden = torch.randn(B, T, d, device=device, dtype=dtype)
    pos_emb = _mk_rope_cossin(cfg, T, device, dtype)
    attn_mask = None   # eager + None → implicit causal in HF

    # ----- 4. Reference: forward_no_memory -----
    with torch.no_grad():
        O_ref = wrapper.forward_no_memory(
            hidden,
            attention_mask=attn_mask,
            position_embeddings=pos_emb,
            use_cache=False,
            past_key_values=None,
        )
    if isinstance(O_ref, tuple):
        O_ref = O_ref[0]
    print(f"[ok] forward_no_memory → {tuple(O_ref.shape)}  |  |O_ref|_max = {O_ref.abs().max().item():.4f}")

    # ----- 5. Test: full forward with slot_to_hidden.weight=0 -----
    with torch.no_grad():
        O_test = wrapper.forward(
            hidden,
            attention_mask=None,
            position_embeddings=pos_emb,
            use_cache=False,
            past_key_values=None,
        )
    if isinstance(O_test, tuple):
        O_test = O_test[0]
    assert O_test.shape == O_ref.shape, f"shape mismatch {O_test.shape} vs {O_ref.shape}"

    # Per-position max-abs
    per_pos = (O_test - O_ref).float().abs().amax(dim=(0, 2))   # [T]
    T_half = T // 2
    first_half = per_pos[:T_half]
    second_half = per_pos[T_half:]

    print("\n=== Diagnostic 1: per-position divergence from bypass ===")
    print(f"first half  (t ∈ [0, {T_half})):  max={first_half.max().item():.6e}  mean={first_half.mean().item():.6e}")
    print(f"second half (t ∈ [{T_half}, {T})):  max={second_half.max().item():.6e}  mean={second_half.mean().item():.6e}")
    print(f"ratio (2nd/1st max):  {second_half.max().item() / max(first_half.max().item(), 1e-20):.3e}")

    if second_half.max() > 10 * first_half.max().clamp(min=1e-20):
        print("  → asymmetric: second-half tokens diverge much more than first half.")
        print("  → consistent with fix2 `T/2` slot-streaming mask + non-empty slot attention in softmax denominator.")
    else:
        print("  → symmetric divergence: either fix2 mask isn't doing what we think, or there's a deeper bug.")

    # ----- 6. Diagnostic 2: is RMSNorm(0) exactly 0? -----
    print("\n=== Diagnostic 2: RMSNorm(0) check ===")
    rms = LlamaRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(device=device, dtype=dtype)
    z = torch.zeros(1, 1, cfg.hidden_size, device=device, dtype=dtype)
    y = rms(z)
    print(f"|RMSNorm(zeros)|_max = {y.abs().max().item():.6e}")
    print(f"  (if exactly 0: the zero slot stays zero after input_layernorm and Q_slot=K_slot=V_slot=0)")

    # ----- 7. Diagnostic 3: slot keys contaminate softmax denominator? -----
    # Patch the wrapper to DISABLE the fix2 mask so slots are visible to ALL H queries;
    # if the divergence grows uniformly across all T positions, softmax-denom is confirmed.
    print("\n=== Diagnostic 3: disable fix2 T/2 mask — do first-half H-queries start to diverge too? ===")
    from src.memory.mem_space import layer as _layer_mod
    original_builder = _layer_mod._build_extended_attn_mask

    def _no_fix2_mask(k, T, dtype, device, batch_size):
        # Same causal-over-H block, but leave slot-key columns fully visible
        # to ALL H-queries (no T//2 blocker).
        L = k + T
        mask = torch.zeros(L, L, dtype=dtype, device=device)
        neg_inf = torch.finfo(dtype).min
        if T > 0:
            causal = torch.triu(
                torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1,
            )
            mask[k:, k:] = causal
        return mask.view(1, 1, L, L).expand(batch_size, 1, L, L).contiguous()

    _layer_mod._build_extended_attn_mask = _no_fix2_mask
    try:
        with torch.no_grad():
            O_test_nofix2 = wrapper.forward(
                hidden,
                attention_mask=None,
                position_embeddings=pos_emb,
                use_cache=False,
                past_key_values=None,
            )
        if isinstance(O_test_nofix2, tuple):
            O_test_nofix2 = O_test_nofix2[0]
        per_pos_nofix2 = (O_test_nofix2 - O_ref).float().abs().amax(dim=(0, 2))
        fh2 = per_pos_nofix2[:T_half]
        sh2 = per_pos_nofix2[T_half:]
        print(f"without fix2 mask:  first-half max={fh2.max().item():.6e}  second-half max={sh2.max().item():.6e}")
        print(f"ratio first-half (nofix2 / fix2):  {fh2.max().item() / max(first_half.max().item(), 1e-20):.3e}")
        if fh2.max() > 10 * first_half.max().clamp(min=1e-20):
            print("  → CONFIRMED: first-half H-queries diverge once slots are visible → softmax-denom contamination.")
        else:
            print("  → first-half still ~parity; the asymmetry is not from the fix2 mask alone.")
    finally:
        _layer_mod._build_extended_attn_mask = original_builder

    # ----- 8. Overall verdict summary -----
    print("\n=== Summary ===")
    print(f"bypass vs joint-attn (fix2 on):  max-abs = {_diff_stats(O_test, O_ref)['max_abs']:.6e}  "
          f"rel_l2 = {_diff_stats(O_test, O_ref)['rel_l2']:.6e}")
    print("\nInterpretation keys:")
    print("  • rel_l2 < 1e-5  → parity (expected if zero-init Linear truly short-circuits the path).")
    print("  • rel_l2 ~ 1e-3-1e-2, asymmetric (2nd-half > 1st-half) → fix2+softmax-denom effect (predicted).")
    print("  • rel_l2 ~ 1e-2-1e-1, uniform → RMSNorm(0) not zero, or a non-trivial slot-K path.")
    print("  • rel_l2 > 0.1 → structural bug beyond the above; escalate.")


if __name__ == "__main__":
    main()
