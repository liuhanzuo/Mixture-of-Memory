"""Tiny single-GPU smoke for the parallel raw-KV retrieval channel (2026-06-18).

Builds a small random Llama (NOT the 8B — we only verify mechanics: store write,
top-k retrieval, EV-prefix injection shape, loss.backward graph integrity, and
byte-identical behaviour when the flag is off). Run on one free GPU.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import LlamaConfig, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space.config import MemorySpaceConfig  # noqa: E402
from src.memory.mem_space.layer import MemorySpaceLayer  # noqa: E402
from src.memory.mem_space.patch import apply_mem_space_to_model  # noqa: E402


def build(use_rawkv: bool, rawkv_layer: int, seed: int = 0):
    torch.manual_seed(seed)
    # Reset the class-level layer-index counter so each freshly built model's
    # decoder layers are indexed 0..L-1 (matches the single-model production
    # path; without this, building several models offsets the indices).
    MemorySpaceLayer._instance_counter = 0
    cfg = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=2048,
    )
    model = LlamaForCausalLM(cfg)
    mem_cfg = MemorySpaceConfig(
        num_slots=32,
        top_k=8,
        slot_dim=None,
        swa_window=0,
        use_rawkv_retrieval=use_rawkv,
        rawkv_layer=rawkv_layer,
        rawkv_topk=16,
    )
    apply_mem_space_to_model(model, mem_cfg, layer_indices=None)
    return model.to(DEVICE).to(DTYPE), mem_cfg


def run_chunks(model, n_chunks=3, T=24, B=1, seed=1):
    """Stream a few chunks through the model (memory accumulates), return last
    logits + the shared bank."""
    torch.manual_seed(seed)
    bank = model._mem_space_shared_bank
    if bank is not None:
        bank.reset(B)
    last = None
    ids_all = []
    for _ in range(n_chunks):
        ids = torch.randint(0, 512, (B, T), device=DEVICE)
        ids_all.append(ids)
        out = model(input_ids=ids)
        last = out.logits
    return last, bank, ids_all


DEVICE = torch.device("cuda:0")
DTYPE = torch.float32  # fp32 for a clean byte-identical comparison


def main():
    print(f"[smoke] device={DEVICE} dtype={DTYPE}")

    # ---- (1) rawkv OFF baseline ----
    model_off, _ = build(use_rawkv=False, rawkv_layer=2, seed=0)
    model_off.eval()
    with torch.no_grad():
        logits_off, bank_off, ids = run_chunks(model_off, seed=1)
    print(f"[smoke] OFF: last logits shape={tuple(logits_off.shape)} "
          f"rawkv_size={bank_off.rawkv_size()}")
    assert bank_off.rawkv_size() == 0, "OFF must not populate the raw-KV store"

    # ---- (2) rawkv OFF determinism reference: rebuild identically, compare ----
    model_off2, _ = build(use_rawkv=False, rawkv_layer=2, seed=0)
    model_off2.eval()
    with torch.no_grad():
        logits_off2, _, _ = run_chunks(model_off2, seed=1)
    max_diff_off = (logits_off - logits_off2).abs().max().item()
    print(f"[smoke] OFF vs OFF reproducibility max|Δ|={max_diff_off:.3e}")
    assert max_diff_off == 0.0, "OFF path must be deterministic / byte-identical"

    # ---- (3) rawkv ON: store written, retrieval + injection, grad flows ----
    model_on, mem_cfg = build(use_rawkv=True, rawkv_layer=2, seed=0)
    model_on.train()
    bank = model_on._mem_space_shared_bank
    bank.reset(1)
    T = 24
    n_chunks = 3
    torch.manual_seed(1)
    loss = None
    for c in range(n_chunks):
        ids = torch.randint(0, 512, (1, T), device=DEVICE)
        out = model_on(input_ids=ids, labels=ids)
        loss = out.loss
        print(f"[smoke] ON chunk {c}: rawkv_size={bank.rawkv_size()} "
              f"loss={loss.item():.4f}")
    # After 3 chunks the store should hold ~n_chunks*T entries (writer appends
    # every real token at the rawkv_layer).
    expect = n_chunks * T
    print(f"[smoke] ON: final rawkv_size={bank.rawkv_size()} (expect {expect})")
    assert bank.rawkv_size() == expect, "store size mismatch"

    # Retrieval shape check (directly call the bank API the layer uses).
    S = bank.rawkv_key.shape[-1]
    qk = torch.randn(1, T, S, device=DEVICE)
    ret = bank.retrieve_rawkv(qk, topk=mem_cfg.rawkv_topk)
    assert ret is not None
    rk_h, rk_pos = ret
    R = min(mem_cfg.rawkv_topk, expect)
    print(f"[smoke] ON: retrieve_rawkv -> hidden={tuple(rk_h.shape)} "
          f"pos={tuple(rk_pos.shape)} (expect R={R})")
    assert rk_h.shape == (1, R, 128), "retrieved hidden shape wrong"
    assert rk_pos.shape == (1, R), "retrieved pos shape wrong"

    # Backward: the graph must not be broken by the (no_grad) raw-KV ops.
    loss.backward()
    n_grad = sum(
        1 for p in model_on.parameters() if p.requires_grad and p.grad is not None
    )
    print(f"[smoke] ON: loss.backward() OK, {n_grad} params received grad")
    assert n_grad > 0, "no gradients flowed — graph broken"

    # ---- (4) ON-but-empty-store first chunk == retrieval returns None safely
    bank.reset(1)
    assert bank.retrieve_rawkv(torch.randn(1, T, S, device=DEVICE), 16) is None
    print("[smoke] ON: empty-store retrieve returns None (safe)")

    print("[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
