#!/usr/bin/env python3
"""Full-stack real-model gradient verification for Method A raw-KV readout.

Runs the REAL training entry points (build_model + _freeze_backbone +
dolmino_train_step) on a REAL Llama-3-8B, single GPU, bf16, so the
context-chunks-no_grad + _detach_banks regime is exercised exactly as in
training. Confirms gradient reaches BOTH the gist query_proj AND key_proj, the
unfrozen reader, loss is finite, and the read path fires. Also reports the GPU
memory increment of injecting at 1 vs 2 readout layers.

NOT a training run: 1 GPU, a couple of forward/backward steps, then exits.

Usage:
  CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/verify_rawkv_readout_fullstack.py \
      --layers 16 --chunk_size 256 --n_ctx 3
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Import the REAL training-script entry points.
import scripts.train_mem_space_dolmino_cpt as T  # noqa: E402


def _make_args(readout_layers, chunk_size, model_path):
    """Build the args namespace via the real parser, then override the few
    fields we need so build_model wires Method A + an unfrozen reader."""
    argv = [
        "--model_path", model_path,
        "--dolmino_path", "/dev/null",      # unused (we feed chunks manually)
        "--output_dir", "/tmp/rawkv_verify_out",
        "--chunk_size", str(chunk_size),
        "--num_slots", "128",
        "--top_k", "16",
        "--use_memory_xattn",               # P11/P8 read path (matches mem_space)
        "--shared_memory_bank",
        "--use_rawkv_readout",
        "--rawkv_readout_layers", ",".join(str(x) for x in readout_layers),
        "--rawkv_gist_dim", "128",
        "--rawkv_readout_topk_chunks", "8",
        "--unfreeze_backbone",
        "--unfreeze_layers_from", str(min(readout_layers)),
        "--attn_impl", "eager",             # deterministic + bias-mask friendly
    ]
    parser = T.build_parser() if hasattr(T, "build_parser") else None
    if parser is None:
        # The script parses inside parse_args(); call it with argv injected.
        old = sys.argv
        sys.argv = ["verify"] + argv
        try:
            args = T.parse_args()
        finally:
            sys.argv = old
    else:
        args = parser.parse_args(argv)
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", type=str, default="16",
                    help="comma readout layers, e.g. '16' or '16,24'")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--n_ctx", type=int, default=3)
    ap.add_argument("--model_path", type=str,
                    default="models/Meta-Llama-3-8B")
    ap.add_argument("--also_two_layer", action="store_true",
                    help="after the main run, also measure mem for L16+L24")
    cli = ap.parse_args()

    assert torch.cuda.is_available(), "need a GPU"
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    readout_layers = sorted(int(x) for x in cli.layers.split(","))

    print(f"=== Method A full-stack grad verify ===")
    print(f"readout_layers={readout_layers} chunk_size={cli.chunk_size} "
          f"n_ctx={cli.n_ctx} unfreeze_from={min(readout_layers)}")

    args = _make_args(readout_layers, cli.chunk_size, cli.model_path)

    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats(device)
    model = T.build_model(args, device, dtype)
    T._freeze_backbone(
        model, unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_layers_from=args.unfreeze_layers_from,
    )
    # Arm the in-graph grad probe on every mem layer (retain_grad on bias/K_raw).
    for w in getattr(model, "_mem_space_layers", []) or []:
        w._inattn_grad_probe = True
    model.train()
    mem_after_build = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[mem] after build+freeze: peak_alloc={mem_after_build:.2f} GB")

    gist = getattr(model, "_gist_readout", None)
    assert gist is not None, "gist scorer not created"
    # zero grads
    for p in model.parameters():
        p.grad = None

    # Build a few short context chunks + a target chunk (random token ids).
    V = model.config.vocab_size
    torch.manual_seed(1)
    context_chunks = [
        torch.randint(0, V, (1, cli.chunk_size)) for _ in range(cli.n_ctx)
    ]
    target_ids = torch.randint(0, V, (1, cli.chunk_size))

    torch.cuda.reset_peak_memory_stats(device)
    # === REAL training step path ===
    lm_loss, aux_loss, *_ = T.dolmino_train_step(
        model, context_chunks, target_ids, device, grad_accum=1,
    )
    mem_step = torch.cuda.max_memory_allocated(device) / 1e9
    print(f"[mem] peak during dolmino_train_step (fwd+bwd, inject "
          f"{len(readout_layers)} layer(s)): {mem_step:.2f} GB")

    print(f"[loss] lm_loss={float(lm_loss):.4f} aux_loss={float(aux_loss):.4f} "
          f"finite={bool(torch.isfinite(lm_loss))}")

    # --- (c) read path fired on target chunk? ---
    fired = [bool(getattr(w, "_last_rawkv_readout_fired", False))
             for w in model._mem_space_layers if getattr(w, "_is_rawkv_readout_layer", False)]
    Rseen = [int(getattr(w, "_last_rawkv_readout_R", 0))
             for w in model._mem_space_layers if getattr(w, "_is_rawkv_readout_layer", False)]
    print(f"[read] fired_on_readout_layers={fired} R={Rseen}")

    # --- (a) gist query_proj AND key_proj grad ---
    qg = gist.query_proj.weight.grad
    kg = gist.key_proj.weight.grad
    qn = float(qg.norm().item()) if qg is not None else None
    kn = float(kg.norm().item()) if kg is not None else None
    print(f"[gist] query_proj grad_norm = {qn}")
    print(f"[gist] key_proj   grad_norm = {kn}   <-- KEY-SIDE (critical)")

    # --- (b) unfrozen reader grad at a readout layer ---
    ro = [w for w in model._mem_space_layers
          if getattr(w, "_is_rawkv_readout_layer", False)][0]
    attn = ro.wrapped_layer.self_attn
    og = attn.o_proj.weight.grad
    qpg = attn.q_proj.weight.grad
    on = float(og.norm().item()) if og is not None else None
    qpn = float(qpg.norm().item()) if qpg is not None else None
    print(f"[reader L{ro._layer_idx}] o_proj grad_norm={on} q_proj grad_norm={qpn}")

    # --- verdicts ---
    print("\n=== VERDICT ===")
    ok_fire = any(fired) and max(Rseen) > 0
    ok_q = qn is not None and qn > 0.0
    ok_k = kn is not None and kn > 0.0
    ok_reader = on is not None and on > 0.0
    ok_loss = bool(torch.isfinite(lm_loss))
    print(f"(c) read fires + raw KV retrieved : {ok_fire}")
    print(f"(a) gist QUERY side grad non-zero : {ok_q}")
    print(f"(a) gist KEY   side grad non-zero : {ok_k}")
    print(f"(b) unfrozen reader grad non-zero : {ok_reader}")
    print(f"    loss finite                   : {ok_loss}")
    all_ok = ok_fire and ok_q and ok_k and ok_reader and ok_loss
    print(f"ALL: {'PASS' if all_ok else 'FAIL'}")
    if not ok_k:
        print("!! KEY-SIDE GRADIENT IS ZERO/None -> gist scorer only half-trained "
              "(detach_banks severed the historical-chunk projection path). "
              "Need self-retrieval or detach-strategy change.")


if __name__ == "__main__":
    main()
