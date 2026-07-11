#!/usr/bin/env python
"""QCMem depth-partition self-test on the Hunyuan **Hy3** (``hy_v3``) MoE backbone,
sharded across GPUs (2026-07-11).

Mirrors ``scripts/qcmem_moe_selftest.py`` but (a) uses the real ``HYV3*`` classes
(native ``hy_v3`` support, transformers >= 5.13.1, NO trust_remote_code) and (b)
exercises :class:`src.memory.qcmem.qcmem_hy3.QCMemHy3Model`, the device-aware
subclass that moves the residual hidden / mask / RoPE onto each layer's GPU so the
model may be ``device_map``-sharded.

Two modes
---------
* ``--tiny`` (default, CPU, fp32, no weights, no GPU) — build a small random
  ``HYV3ForCausalLM`` (few layers, few experts) and prove the load-bearing claims:
    (A1) j=0 write/read packing reproduces a stock ``model(input_ids=packed)``
         forward to < 1e-4 (fp32).
    (A2) single-sequence split-at-j resume equals the full forward at EVERY
         j in {0, 1, L//2, L} — i.e. the depth partition (incl. the dense->MoE
         layer-0 boundary) is exact and every MoE router/expert is re-executed
         correctly on resume.
    (B)  MoE-router position-blindness: a chunk's bottom-stack hidden (hence the
         input to every ``HYV3MoE`` block, hence the discrete top-k expert pick)
         is invariant to a uniform RoPE position shift — the property that makes
         chunk-local WRITE reproducible.
  This runs on ONE device (CPU here), so it validates the QCMem plumbing +
  Hy3-MoE routing; the multi-GPU device-hop is a strict superset (no-op single
  device) and is validated by the ``--model_path`` mode.

* ``--model_path <Hy3 dir> --device_map auto`` — load the real 597 GB Hy3 sharded
  over the visible GPUs and run (A1)+(A2) with a bf16 tolerance (< 1e-2). Use a
  TINY input (short chunks, all chunks selected) — this only checks numerical
  correctness of the split, not long-context behaviour.

Usage:
    .venv_hy3/bin/python scripts/qcmem_hy3_selftest.py --tiny --resume_j 2
    .venv_hy3/bin/python scripts/qcmem_hy3_selftest.py \
        --model_path /apdcephfs_wzc1/.../models/Hy3 --device_map auto \
        --resume_j 8 --dtype bfloat16 --tol 1e-2
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers.masking_utils import create_causal_mask  # noqa: E402

from src.memory.qcmem.qcmem_hy3 import QCMemHy3Model  # noqa: E402


# --------------------------------------------------------------------------- #
# model builders
# --------------------------------------------------------------------------- #
def build_tiny_hy3(device, dtype, seed: int = 0):
    """Small random ``HYV3ForCausalLM``: 4 layers (layer 0 dense, 1..3 sparse MoE
    via first_k_dense_replace=1), hidden=64, 8 experts top-2, 1 shared expert.
    Uses the genuine ``HYV3*`` module stack — no weights to download."""
    from transformers import HYV3Config, HYV3ForCausalLM
    cfg = HYV3Config(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        num_experts=8,
        num_experts_per_tok=2,
        num_shared_experts=1,
        max_position_embeddings=2048,
        rope_parameters={"rope_type": "default", "rope_theta": 500000.0},
        tie_word_embeddings=False,
        enable_moe_fp32_combine=False,
        router_scaling_factor=1.5,
    )
    torch.manual_seed(seed)
    model = HYV3ForCausalLM(cfg).to(device=device, dtype=dtype).eval()
    return model


def load_real_hy3(model_path, dtype, device_map, attn_impl):
    from src.memory.qcmem.qcmem_hy3 import load_hy3_qcmem  # noqa: F401
    from transformers import AutoModelForCausalLM
    print(f"[hy3-selftest] loading {model_path} device_map={device_map} "
          f"dtype={dtype} attn={attn_impl} (local_files_only)")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, device_map=device_map,
        attn_implementation=attn_impl, low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    dm = getattr(model, "hf_device_map", None)
    if dm is not None:
        devs = sorted({str(v) for v in dm.values()})
        print(f"[hy3-selftest] hf_device_map spans {len(devs)} device(s): {devs}")
    return model


# --------------------------------------------------------------------------- #
# Hy3 MoE-block discovery + routing capture
# --------------------------------------------------------------------------- #
def find_hy3_moe_blocks(inner):
    """Return ``[(layer_idx, HYV3MoE)]`` for every sparse layer (mlp is HYV3MoE)."""
    blocks = []
    for i, layer in enumerate(inner.layers):
        mlp = getattr(layer, "mlp", None)
        if mlp is not None and hasattr(mlp, "gate") and hasattr(mlp, "experts"):
            blocks.append((i, mlp))
    return blocks


@torch.no_grad()
def bottom_forward_capture(qc, ids, offset, moe_blocks):
    """Run embed + all layers over ``ids`` with RoPE positions ``offset:offset+T``
    (uniform shift) and a causal mask, capturing each MoE block's INPUT hidden via
    a forward pre-hook. Returns ``{layer_idx: input_hidden}`` (on CPU float)."""
    inner = qc.inner
    T = ids.shape[1]
    inputs_embeds = qc.embed_tokens(qc._as_ids(ids))
    positions = (torch.arange(T, device=inputs_embeds.device) + offset).unsqueeze(0)
    causal_mask = create_causal_mask(
        config=qc.config, inputs_embeds=inputs_embeds, attention_mask=None,
        past_key_values=None, position_ids=positions,
    )
    position_embeddings = qc.rotary_emb(inputs_embeds, position_ids=positions)

    captured = {}
    idx_by_module = {id(m): i for i, m in moe_blocks}
    handles = []

    def pre_hook(module, args, kwargs):
        h = args[0] if args else kwargs.get("hidden_states")
        captured[idx_by_module[id(module)]] = h.detach().float().cpu().clone()

    for _, m in moe_blocks:
        handles.append(m.register_forward_pre_hook(pre_hook, with_kwargs=True))
    try:
        qc._run_layers(inputs_embeds, slice(0, qc.num_layers),
                       causal_mask, positions, position_embeddings)
    finally:
        for h in handles:
            h.remove()
    return captured


@torch.no_grad()
def hy3_selected_experts(mlp, hidden):
    """Discrete top-k expert ids for a HYV3MoE block given input hidden [1,T,d].

    HYV3TopKRouter.forward(hidden_states, e_score_correction_bias) ->
    (router_logits, top_k_weights, top_k_index). We reproduce it to get the
    sorted per-token expert selection [T, top_k]."""
    d = hidden.shape[-1]
    hs = hidden.reshape(-1, d).to(next(mlp.gate.parameters()).device)
    bias = mlp.e_score_correction_bias.to(hs.device)
    _, _, top_k_index = mlp.gate(hs, bias)
    return top_k_index.sort(dim=-1).values.cpu()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QCMem depth-partition self-test on Hy3 MoE")
    ap.add_argument("--model_path", type=str, default="")
    ap.add_argument("--tiny", action="store_true", default=False)
    ap.add_argument("--resume_j", type=int, default=2)
    ap.add_argument("--device", type=str, default="cpu",
                    help="single-device fallback for --tiny (cpu/cuda:0)")
    ap.add_argument("--device_map", type=str, default="auto",
                    help="device_map for the real model (auto / cuda:0 / ...)")
    ap.add_argument("--attn_impl", type=str, default="eager",
                    help="eager gives the tightest fp match; sdpa ok for bf16 real")
    ap.add_argument("--dtype", type=str, default="float32",
                    choices=["float32", "bfloat16", "float16"])
    ap.add_argument("--tol", type=float, default=-1.0,
                    help="logit-diff gate; default 1e-4 (fp32) / 1e-2 (bf16)")
    ap.add_argument("--pos_offset", type=int, default=257)
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
             "float16": torch.float16}[args.dtype]
    tol = args.tol if args.tol > 0 else (1e-4 if dtype == torch.float32 else 1e-2)

    if args.tiny or not args.model_path:
        print("[hy3-selftest] building TINY random HYV3ForCausalLM "
              "(4 layers: L0 dense + L1-3 MoE, 8 experts top-2)")
        device = torch.device(args.device if (not args.device.startswith("cuda")
                                              or torch.cuda.is_available()) else "cpu")
        model = build_tiny_hy3(device, dtype)
        qc = QCMemHy3Model(model, resume_j=0)
        qc_ctor = lambda j: QCMemHy3Model(model, resume_j=j)  # noqa: E731
    else:
        model = load_real_hy3(args.model_path, dtype, args.device_map, args.attn_impl)
        qc = QCMemHy3Model(model, resume_j=0)
        qc_ctor = lambda j: QCMemHy3Model(model, resume_j=j)  # noqa: E731

    L = qc.num_layers
    V = int(qc.config.vocab_size)
    inner = qc.inner
    moe_blocks = find_hy3_moe_blocks(inner)
    print(f"[hy3-selftest] L={L} vocab={V} dtype={dtype} tol={tol:.1e} "
          f"sharded={qc.is_sharded} | MoE-sparse layers: {len(moe_blocks)}/{L} "
          f"(idx {[i for i, _ in moe_blocks][:6]}{'...' if len(moe_blocks) > 6 else ''})")

    torch.manual_seed(0)

    def rand_ids(n):
        return torch.randint(0, V, (1, n), device=qc.device)

    bos_id = getattr(qc.config, "bos_token_id", None)
    bos_id = 1 if bos_id is None else int(bos_id)
    # keep bos within vocab for the tiny model
    if bos_id >= V:
        bos_id = 1
    sink_ids = torch.tensor([[bos_id]], device=qc.device)
    c1, c2, c3 = rand_ids(37), rand_ids(29), rand_ids(41)
    q = rand_ids(23)
    packed_ids = torch.cat([sink_ids, c1, c2, c3, q], dim=1)

    ref = qc.full_forward_logits(packed_ids).float().cpu()

    # ================= (A1) j=0 write/read packing =================
    print("-" * 72)
    print("(A) exact depth-partition (Hy3 MoE routing re-executed on resume)")
    sink_hj = qc.write_chunk(sink_ids)
    ctx_hj = [qc.write_chunk(c) for c in (c1, c2, c3)]
    q_hj = qc.write_chunk(q)
    out_pack = qc.read(sink_hj, ctx_hj, q_hj).float().cpu()
    diff_pack = (out_pack - ref).abs().max().item()
    a1 = diff_pack < tol
    print(f"  (A1) j=0 write/read packing   max|logit diff| = {diff_pack:.3e}  "
          f"{'PASS' if a1 else 'FAIL'}")

    # ================= (A2) split-at-j resume sweep =================
    diffs_j = {}
    for j in sorted({0, 1, L // 2, L}):
        qcj = qc_ctor(j)
        outj = qcj.resume_forward_ids(packed_ids).float().cpu()
        diffs_j[j] = (outj - ref).abs().max().item()
    a2 = all(d < tol for d in diffs_j.values())
    for j, d in diffs_j.items():
        print(f"  (A2) resume_forward_ids j={j:>2}    max|logit diff| = {d:.3e}  "
              f"{'PASS' if d < tol else 'FAIL'}")

    # ================= (B) router position-blindness =================
    print("-" * 72)
    print(f"(B) Hy3 MoE-router position-blindness (RoPE shift by +{args.pos_offset})")
    b1 = b2 = True
    if moe_blocks:
        chunk = rand_ids(48)
        cap0 = bottom_forward_capture(qc, chunk, 0, moe_blocks)
        capO = bottom_forward_capture(qc, chunk, args.pos_offset, moe_blocks)
        max_hid_diff = max((cap0[li] - capO[li]).abs().max().item() for li in cap0)
        b1 = max_hid_diff < 1e-3
        print(f"  (B1) MoE-block INPUT hidden max|diff| over shift = {max_hid_diff:.3e}  "
              f"{'PASS (position-invariant)' if b1 else 'FAIL'}")
        mism = checked = 0
        for li, mlp in moe_blocks:
            try:
                s0 = hy3_selected_experts(mlp, cap0[li])
                sO = hy3_selected_experts(mlp, capO[li])
            except Exception as e:  # pragma: no cover
                print(f"  (B2) gate decode failed on layer {li}: {e}")
                continue
            checked += 1
            mism += int((s0 != sO).any().item())
        b2 = (checked > 0 and mism == 0)
        print(f"  (B2) discrete expert selection identical across shift: "
              f"{checked - mism}/{checked} layers match  {'PASS' if b2 else 'FAIL'}")
    else:
        print("  (B) no MoE blocks found (dense model?) — skipping")

    all_ok = a1 and a2 and b1 and b2
    print("=" * 72)
    print(f"HY3 SELF-TEST: {'ALL PASS — QCMem depth-partition is exact on Hy3 MoE' if all_ok else 'FAILURE'}")
    print("=" * 72)
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
