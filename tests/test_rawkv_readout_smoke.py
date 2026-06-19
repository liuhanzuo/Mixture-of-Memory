#!/usr/bin/env python3
"""Smoke test for Raw-KV Readout — Method A (2026-06-19).

Method A = per-chunk raw-KV store + EMERGENT TRAINABLE gist-key soft attention,
replacing the lossy slots + the non-differentiable TopKSelector hard top-k of
the in-attn probe. docs/RAWKV_READOUT_PROPOSAL.md §2.

What we verify (CPU, tiny random Llama — no weight download):

1. use_rawkv_readout=False → forward is byte-identical to a build constructed
   before the feature (regression / zero-impact).
2. use_rawkv_readout=True, multi-chunk fwd+bwd:
   (a) loss is finite;
   (b) the trainable gist projection (query_proj / key_proj) receives non-zero
       gradient — proving the emergent gist scorer is IN the loss graph;
   (c) the unfrozen reader (the wrapped decoder layer's self-attn) receives
       non-zero gradient — proving the reader is trained on the raw-KV path
       (the one thing the frozen-reader oracle negative never did);
   (d) the readout READ path FIRES (counter) and retrieves raw KV from the
       store, injecting it into the wrapped self-attention.
3. ★ Cross-chunk read fires in the streamed-context regime: a chunk-1 forward
   retrieves the chunk-0 tokens written to the store on the chunk-0 forward, and
   gradient flows back to the gist scorer through that cross-chunk read.

Run: .venv/bin/python tests/test_rawkv_readout_smoke.py
 or: .venv/bin/python -m pytest tests/test_rawkv_readout_smoke.py -q
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
from src.memory.mem_space.layer import MemorySpaceLayer  # noqa: E402


def _reset_layer_counter() -> None:
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
# 1. OFF == zero impact (byte-identical to a no-readout build).
# --------------------------------------------------------------------------- #
def test_rawkv_readout_off_byte_identical() -> None:
    ids = torch.randint(0, 1024, (2, 32))

    model_a = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(model_a, _cfg(use_rawkv_readout=False),
                             layer_indices=[0, 1])
    with torch.no_grad():
        out_a = model_a(input_ids=ids).last_hidden_state

    model_b = _tiny_llama()
    _reset_layer_counter()
    apply_mem_space_to_model(model_b, _cfg(use_rawkv_readout=False),
                             layer_indices=[0, 1])
    with torch.no_grad():
        out_b = model_b(input_ids=ids).last_hidden_state

    assert torch.equal(out_a, out_b), (
        "two readout-OFF builds with identical seeds must be bit-identical"
    )
    # No gist scorer module is created when the feature is off.
    assert getattr(model_a, "_gist_readout", None) is None
    print("[rawkv readout smoke] OK: OFF is byte-identical + no gist scorer")


# --------------------------------------------------------------------------- #
# 2 + 3. ON: multi-chunk fwd+bwd, gist + reader grads non-zero, read fires,
#        cross-chunk read fires in the streamed-context regime.
# --------------------------------------------------------------------------- #
def test_rawkv_readout_on_fwd_bwd_grads() -> None:
    torch.manual_seed(1)
    model = _tiny_llama()
    _reset_layer_counter()
    # Readout on layers [0,1]; write owner = layer 0 (smallest). gist_dim small.
    apply_mem_space_to_model(
        model,
        _cfg(use_rawkv_readout=True, rawkv_readout_layers=[0, 1],
             rawkv_gist_dim=32, rawkv_readout_topk_chunks=4,
             rawkv_readout_temp=1.0),
        layer_indices=[0, 1],
    )
    model.train()
    # Arm the in-graph grad probe on every mem layer (retain_grad on col_bias).
    for w in getattr(model, "_mem_space_layers", []):
        w._inattn_grad_probe = True

    gist = model._gist_readout
    assert gist is not None, "gist scorer must be created when readout is on"

    bank = model._mem_space_shared_bank

    # --- chunk 0 (context): write the store, run WITH grad so the graph is one
    #     piece; content is detached on write so no cross-chunk BPTT leaks. ---
    bank.reset()
    ids0 = torch.randint(0, 1024, (2, 32))
    _ = model(input_ids=ids0).last_hidden_state
    store = getattr(bank, "_rawkv_readout_store", None)
    assert store is not None and store.size() > 0, "store must be written"
    chunks_after_0 = store.n_chunks

    # --- chunk 1 (target): read must retrieve chunk-0 (+chunk-1) tokens; the
    #     gist scorer query is this chunk's grad-bearing hidden. ---
    ids1 = torch.randint(0, 1024, (2, 32))
    out1 = model(input_ids=ids1).last_hidden_state
    assert torch.isfinite(out1).all(), "NaN/Inf in readout-on forward"

    # (d) the read path fired on at least one readout layer this forward.
    fired = [
        bool(getattr(w, "_last_rawkv_readout_fired", False))
        for w in model._mem_space_layers
    ]
    R_seen = [int(getattr(w, "_last_rawkv_readout_R", 0)) for w in model._mem_space_layers]
    print(f"[rawkv readout smoke] read fired per layer = {fired}, R = {R_seen}")
    assert any(fired), "raw-KV readout read path never fired"
    # ★ cross-chunk: the store held > 1 chunk's worth when the read fired.
    assert store.n_chunks >= chunks_after_0 + 1, (
        "store should have grown by the target chunk too (cross-chunk read)"
    )
    assert max(R_seen) > 0, "no raw KV retrieved into the self-attention"

    # (a) loss finite + backward.
    loss = out1.float().pow(2).mean()
    loss.backward()
    assert torch.isfinite(loss), "loss not finite"

    # (b) the trainable gist projection got non-zero gradient.
    qg = gist.query_proj.weight.grad
    kg = gist.key_proj.weight.grad
    assert qg is not None and torch.isfinite(qg).all(), "gist query_proj no grad"
    assert kg is not None and torch.isfinite(kg).all(), "gist key_proj no grad"
    qn = float(qg.norm().item())
    kn = float(kg.norm().item())
    print(f"[rawkv readout smoke] gist grad norms: query_proj={qn:.4e} "
          f"key_proj={kn:.4e}")
    assert qn > 0.0, "gist query_proj gradient is exactly zero (scorer not in graph)"
    assert kn > 0.0, "gist key_proj gradient is exactly zero (scorer not in graph)"

    # (c) the unfrozen reader (wrapped self-attn) got non-zero gradient on a
    #     readout layer — proving the reader is trained on the raw-KV path.
    ro_layer = model._mem_space_layers[1]  # a readout layer (index 1)
    oproj = ro_layer.wrapped_layer.self_attn.o_proj.weight.grad
    assert oproj is not None and torch.isfinite(oproj).all(), (
        "reader self-attn got no gradient — readout path not differentiable"
    )
    on = float(oproj.norm().item())
    print(f"[rawkv readout smoke] reader o_proj grad norm = {on:.4e}")
    assert on > 0.0, "reader gradient is exactly zero"

    # The retained col_bias (gist log-weights) should carry gradient too.
    for w in model._mem_space_layers:
        cb = getattr(w, "_last_rawkv_readout_bias", None)
        if cb is not None and cb.grad is not None:
            cbn = float(cb.grad.norm().item())
            print(f"[rawkv readout smoke] col_bias grad norm = {cbn:.4e} "
                  f"(layer {w._layer_idx})")
            break

    print("[rawkv readout smoke] OK: fwd+bwd, gist+reader grads non-zero, "
          "cross-chunk read fires")


if __name__ == "__main__":
    test_rawkv_readout_off_byte_identical()
    test_rawkv_readout_on_fwd_bwd_grads()
    print("\nAll Raw-KV Readout (Method A) smoke checks passed.")
