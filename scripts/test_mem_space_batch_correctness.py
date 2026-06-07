#!/usr/bin/env python3
"""Numerical correctness test for batch_size > 1 in the mem_space Dolmino CPT
training path (2026-06-07).

Builds a TINY randomly-initialised Llama + mem_space adapter (no 8B download),
then verifies:

  (A) Per-sample independence: with bs=2, perturbing sample 1's tokens leaves
      sample 0's target-chunk logits byte-identical (the memory bank slot state
      must not bleed across batch elements).

  (B) Loss equivalence: the bs=2 dolmino_train_step_tbptt mean LM loss equals
      the average of two separate bs=1 calls on the SAME two samples (within
      bf16 / fp32 tolerance). This proves the batched memory rollout produces
      the same per-sample numbers as the trusted single-sample path.

Run (fp32, CPU or single GPU):
    .venv/bin/python scripts/test_mem_space_batch_correctness.py
"""
from __future__ import annotations

import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model  # noqa: E402

# Import the training-step functions + batch helpers under test.
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location(
    "train_mem_space_dolmino_cpt",
    os.path.join(PROJECT_ROOT, "scripts", "train_mem_space_dolmino_cpt.py"),
)
_tm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tm)


def build_tiny_model(device, dtype, seed=0):
    torch.manual_seed(seed)
    cfg = LlamaConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=512,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(cfg).to(device=device, dtype=dtype)
    ms_cfg = MemorySpaceConfig(
        num_slots=16,
        top_k=4,
        selector_dim=32,
        slot_dim=None,            # = hidden_size
        writeback_gate_max=0.3,
        writeback_gate_warmup_steps=0,   # gate active immediately so writes happen
        slot_init="hidden_pool",
        slot_init_noise=0.0,      # deterministic init for the independence check
        enable_writeback=True,
        return_aux_losses=True,
    )
    apply_mem_space_to_model(model, ms_cfg, layer_indices=None)
    model.to(device=device, dtype=dtype)
    model.eval()  # eval() -> deterministic (no dropout); grads still flow
    return model


def make_samples(device, n_ctx=2, chunk_size=16, vocab=256, seed=123):
    g = torch.Generator().manual_seed(seed)
    def rand_chunk():
        return torch.randint(0, vocab, (chunk_size,), generator=g)
    sample0 = {
        "context_chunks": [rand_chunk() for _ in range(n_ctx)],
        "target_ids": rand_chunk(),
        "is_dolmino": True,
    }
    sample1 = {
        "context_chunks": [rand_chunk() for _ in range(n_ctx)],
        "target_ids": rand_chunk(),
        "is_dolmino": True,
    }
    return sample0, sample1


@torch.no_grad()
def test_independence(model, device, dtype):
    """bs=2: changing sample 1 must NOT change sample 0's target logits."""
    s0, s1 = make_samples(device)
    s1_alt = {
        "context_chunks": [c + 1 for c in s1["context_chunks"]],  # different tokens
        "target_ids": s1["target_ids"].clone(),
        "is_dolmino": True,
    }

    def run_batched(a, b):
        batch = _tm.dolmino_collate_fn([a, b])
        _tm._reset_banks(model)
        for ctx in batch["context_chunks"]:
            model(input_ids=_tm._ensure_batched(ctx, device), use_cache=False)
        _tm._detach_banks(model)
        tgt = _tm._ensure_batched(batch["target_ids"], device)
        out = model(input_ids=tgt, use_cache=False)
        return out.logits  # [B, T, V]

    logits_a = run_batched(s0, s1)
    logits_b = run_batched(s0, s1_alt)
    # Sample 0 (row 0) must be identical despite sample 1 changing.
    max_diff0 = (logits_a[0] - logits_b[0]).abs().max().item()
    # Sample 1 (row 1) MUST differ (sanity: the perturbation actually mattered).
    max_diff1 = (logits_a[1] - logits_b[1]).abs().max().item()
    return max_diff0, max_diff1


def test_loss_equivalence(model, device, dtype, route_aux_weight=0.0):
    """bs=2 mean LM loss == mean of two bs=1 LM losses on the same samples.

    We use the gradient-bearing dolmino_train_step_tbptt (the real training
    step). It calls backward() internally; we zero grads between runs and only
    compare the returned (detached) total_lm loss scalars.
    """
    s0, s1 = make_samples(device)

    def bs1_loss(sample):
        model.zero_grad(set_to_none=True)
        ctx = sample["context_chunks"]
        tgt = sample["target_ids"]
        total_lm, _, _ = _tm.dolmino_train_step_tbptt(
            model, ctx, tgt, device, grad_accum=1, bptt_window=2,
            route_aux_weight=route_aux_weight,
        )
        # total_lm is SUM over (n_ctx+1) chunks scaled by 1/n_chunks then *grad_accum
        # -> it is the per-chunk MEAN lm loss for this sample. Return it directly.
        return float(total_lm.item())

    lm0 = bs1_loss(s0)
    lm1 = bs1_loss(s1)
    mean_bs1 = 0.5 * (lm0 + lm1)

    # Batched: collate the two samples, run the SAME tbptt step with B=2.
    model.zero_grad(set_to_none=True)
    batch = _tm.dolmino_collate_fn([s0, s1])
    ctx_b = batch["context_chunks"]          # list of [B, chunk_size]
    tgt_b = batch["target_ids"]              # [B, chunk_size]
    total_lm_b, _, _ = _tm.dolmino_train_step_tbptt(
        model, ctx_b, tgt_b, device, grad_accum=1, bptt_window=2,
        route_aux_weight=route_aux_weight,
    )
    mean_bs2 = float(total_lm_b.item())

    return lm0, lm1, mean_bs1, mean_bs2


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    print("\n========== FP32 ==========")
    dtype = torch.float32
    model = build_tiny_model(device, dtype, seed=0)

    d0, d1 = test_independence(model, device, dtype)
    print(f"[independence] sample0 max logit diff when sample1 perturbed: {d0:.3e} "
          f"(expect ~0)")
    print(f"[independence] sample1 max logit diff when sample1 perturbed: {d1:.3e} "
          f"(expect >0 -> perturbation real)")
    indep_ok = (d0 < 1e-5) and (d1 > 1e-4)
    print(f"[independence] PASS={indep_ok}")

    lm0, lm1, mean_bs1, mean_bs2 = test_loss_equivalence(model, device, dtype)
    rel = abs(mean_bs1 - mean_bs2) / max(1e-8, abs(mean_bs1))
    print(f"[loss-equiv] bs1 sample0 lm={lm0:.6f}  sample1 lm={lm1:.6f}")
    print(f"[loss-equiv] mean(2x bs1)={mean_bs1:.6f}  bs2 mean={mean_bs2:.6f}")
    print(f"[loss-equiv] abs diff={abs(mean_bs1-mean_bs2):.3e}  rel={rel:.3e}")
    loss_ok = rel < 1e-4
    print(f"[loss-equiv] PASS={loss_ok} (fp32 tol rel<1e-4)")

    # route_aux path too (cross-chunk routing supervision uses gather on [B,N]).
    model2 = build_tiny_model(device, dtype, seed=0)
    lm0r, lm1r, mb1r, mb2r = test_loss_equivalence(model2, device, dtype,
                                                    route_aux_weight=1.0)
    relr = abs(mb1r - mb2r) / max(1e-8, abs(mb1r))
    print(f"\n[loss-equiv+route_aux] mean(2x bs1)={mb1r:.6f}  bs2={mb2r:.6f}  "
          f"rel={relr:.3e}  PASS={relr < 1e-4}")

    all_ok = indep_ok and loss_ok and (relr < 1e-4)
    print(f"\n==== OVERALL: {'PASS' if all_ok else 'FAIL'} ====")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
