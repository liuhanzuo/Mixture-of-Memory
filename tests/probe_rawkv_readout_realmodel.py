#!/usr/bin/env python3
"""Real-model full-stack gradient probe for Raw-KV Readout — Method A (2026-06-19).

Validates the caveat the CPU smoke could NOT close: the smoke ran a multi-chunk
fwd WITH grad, but the REAL training path (`dolmino_train_step`) streams context
chunks under ``torch.no_grad()`` then calls ``_detach_banks(model)`` before the
GRAD-bearing target-chunk forward. This probe runs the *real* production
functions (build_model / _freeze_backbone / dolmino_train_step) on a real
Llama-3-8B (single GPU, bf16, partial unfreeze L16) and asserts:

  (a) gist query_proj  grad nonzero   (query side — current chunk query)
  (b) gist key_proj    grad nonzero   (★ KEY side — projection of the DETACHED
                                        historical gist source; the open risk)
  (c) reader o_proj / q_proj grad nonzero on the readout layer (reader trained
      on the raw-KV-concat path — the one thing the frozen-reader oracle never did)
  (d) loss finite + the cross-chunk read actually fired (counter > 0)

Also reports peak CUDA memory for readout injection at L16 vs L16+L24 so the
team can judge how many layers to inject at in the real run.

Run (single free GPU):  CUDA_VISIBLE_DEVICES=<g> .venv/bin/python tests/probe_rawkv_readout_realmodel.py
Or CPU (slow, grad structure identical): CUDA_VISIBLE_DEVICES= .venv/bin/python tests/probe_rawkv_readout_realmodel.py
"""
from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import the REAL production module so we exercise the real code paths.
import scripts.train_mem_space_dolmino_cpt as T  # noqa: E402
from src.memory.mem_space.layer import MemorySpaceLayer  # noqa: E402


def _build(readout_layers: str, chunk_size: int, dtype_str: str = "bfloat16"):
    """Build the real Llama-3-8B mem_space model via the production build_model,
    with Method A readout on `readout_layers` and partial unfreeze from the
    smallest readout layer (so the reader on the readout layer is trainable)."""
    # Class-level instance counter must be reset or a second build in the same
    # process gets shifted _layer_idx (readout wiring + unfreeze range break).
    MemorySpaceLayer._instance_counter = 0
    unfreeze_from = min(int(x) for x in readout_layers.split(","))
    argv = [
        "--model_path", "models/Meta-Llama-3-8B",
        "--dtype", dtype_str,
        "--attn_impl", "sdpa",
        "--chunk_size", str(chunk_size),
        "--num_slots", "16",
        "--top_k", "8",
        "--slot_dim", "256",
        "--use_rawkv_readout",
        "--rawkv_readout_layers", readout_layers,
        "--rawkv_gist_dim", "128",
        "--rawkv_readout_topk_chunks", "4",
        "--rawkv_readout_temp", "1.0",
        "--unfreeze_backbone",
        "--unfreeze_layers_from", str(unfreeze_from),
        "--grad_flow_diag",
        # keep the rest at defaults
        "--output_dir", "/tmp/rawkv_probe_out",
        "--dolmino_path", "MemLong/data/processed/dolmino_per_doc",
        "--total_steps", "1",
    ]
    _saved_argv = sys.argv
    try:
        sys.argv = ["train_mem_space_dolmino_cpt.py"] + argv
        args = T.parse_args()
    finally:
        sys.argv = _saved_argv
    args = T.merge_adapter_config_into_args(args)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    # CPU cannot do bf16 matmul well; fall back to float32 there.
    if device.type == "cpu" and dtype is torch.bfloat16:
        dtype = torch.float32
        args.dtype = "float32"
    torch.manual_seed(0)
    model = T.build_model(args, device, dtype)
    T._freeze_backbone(
        model,
        unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_layers_from=args.unfreeze_layers_from,
    )
    # Arm the in-graph probe (retain_grad on col_bias / K_raw) on every mem layer.
    if args.grad_flow_diag:
        for _w in getattr(model, "_mem_space_layers", []) or []:
            _w._inattn_grad_probe = True
    return model, args, device, dtype


def _run_real_step(model, args, device, chunk_size):
    """Drive the REAL dolmino_train_step: a few short context chunks streamed
    under no_grad + _detach_banks, then a grad-bearing target chunk."""
    vocab = model.config.vocab_size if hasattr(model, "config") else 128256
    g = torch.Generator().manual_seed(7)
    n_ctx = 3
    context_chunks = [
        torch.randint(0, vocab, (chunk_size,), generator=g) for _ in range(n_ctx)
    ]
    target_ids = torch.randint(0, vocab, (chunk_size,), generator=g)
    # This is the production training step (no_grad context, detach_banks, grad target).
    lm_loss, aux_loss, *_ = T.dolmino_train_step(
        model, context_chunks, target_ids, device, grad_accum=1,
    )
    return float(lm_loss.item()), float(aux_loss.item())


def main() -> None:
    chunk_size = 256  # short chunks → fast; grad structure is size-independent.
    # Which readout-layer config to build this process (default single L16).
    # Pass a 2nd arg like "16,24" to measure the two-layer memory footprint in a
    # FRESH process (avoids the in-process double-build memory confound).
    readout_layers = sys.argv[1] if len(sys.argv) > 1 else "16"
    print("=" * 72)
    print(f"Real-model Method A grad probe — readout layers [{readout_layers}], "
          f"unfreeze from L{min(int(x) for x in readout_layers.split(','))}")
    print("=" * 72)
    model, args, device, dtype = _build(readout_layers, chunk_size)
    print(f"device={device} dtype={dtype} chunk_size={chunk_size}")

    gist = getattr(model, "_gist_readout", None)
    assert gist is not None, "gist scorer was not created (use_rawkv_readout off?)"

    # Zero any stale grads.
    for p in model.parameters():
        p.grad = None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        mem_before = torch.cuda.memory_allocated(device) / 1e9

    lm_loss, aux_loss = _run_real_step(model, args, device, chunk_size)
    print(f"\nlm_loss={lm_loss:.4f}  aux_loss={aux_loss:.6f}  finite={torch.isfinite(torch.tensor(lm_loss)).item()}")

    if device.type == "cuda":
        peak = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"[mem] readout=[{readout_layers}] peak CUDA alloc = {peak:.2f} GB "
              f"(delta over weights {peak - mem_before:.2f} GB)")
    else:
        peak = None

    # --- read fired? ---
    fired = [bool(getattr(w, "_last_rawkv_readout_fired", False))
             for w in model._mem_space_layers]
    R_seen = [int(getattr(w, "_last_rawkv_readout_R", 0))
              for w in model._mem_space_layers]
    print(f"[read] fired per mem layer = {fired}  R = {R_seen}")
    assert any(fired), "★ FAIL: raw-KV readout read NEVER fired in the real step"

    # --- (a)(b) gist scorer grads ---
    qg = gist.query_proj.weight.grad
    kg = gist.key_proj.weight.grad
    qn = float(qg.norm().item()) if qg is not None else None
    kn = float(kg.norm().item()) if kg is not None else None
    print(f"\n(a) gist.query_proj grad norm = {qn}")
    print(f"(b) gist.key_proj   grad norm = {kn}  ★ KEY SIDE (detached-source projection)")

    # --- (c) reader grads on the readout layer (L16) ---
    ro_layer = None
    for w in model._mem_space_layers:
        if getattr(w, "_is_rawkv_readout_layer", False):
            ro_layer = w
            break
    assert ro_layer is not None, "no readout layer found"
    attn = ro_layer.wrapped_layer.self_attn
    og = attn.o_proj.weight.grad
    qpg = attn.q_proj.weight.grad
    on = float(og.norm().item()) if og is not None else None
    qpn = float(qpg.norm().item()) if qpg is not None else None
    print(f"\n(c) reader L{ro_layer._layer_idx} o_proj grad norm = {on}")
    print(f"    reader L{ro_layer._layer_idx} q_proj grad norm = {qpn}")

    # --- col_bias retained grad (gist path through the one softmax) ---
    cb_norm = None
    for w in model._mem_space_layers:
        cb = getattr(w, "_last_rawkv_readout_bias", None)
        if cb is not None and cb.grad is not None:
            cb_norm = float(cb.grad.norm().item())
            print(f"    col_bias retained grad norm = {cb_norm} (layer {w._layer_idx})")
            break

    # --- verdicts ---
    print("\n" + "-" * 72)
    ok_q = qn is not None and qn > 0.0
    ok_k = kn is not None and kn > 0.0
    ok_r = (on is not None and on > 0.0) or (qpn is not None and qpn > 0.0)
    print(f"(a) query side grad nonzero : {'PASS' if ok_q else 'FAIL'}")
    print(f"(b) KEY side  grad nonzero : {'PASS' if ok_k else 'FAIL'}  ★")
    print(f"(c) reader    grad nonzero : {'PASS' if ok_r else 'FAIL'}")
    print(f"(d) loss finite + read fired: {'PASS' if (abs(lm_loss) < 1e6 and any(fired)) else 'FAIL'}")
    print("-" * 72)

    print("\nPROBE SUMMARY:",
          f"readout=[{readout_layers}], query={qn}, key={kn}, reader_o={on}, reader_q={qpn},",
          f"loss={lm_loss:.3f}, peak_GB={peak}")


if __name__ == "__main__":
    main()
