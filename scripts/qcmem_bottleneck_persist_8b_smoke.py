#!/usr/bin/env python3
"""8B real-checkpoint smoke test for the d_bottle persist path (B01).

The CPU gate (`scripts/qcmem_bottleneck_persist_selftest.py`) proves the
rearrangement on random weights at the real widths. This proves it on the ACTUAL
trained endpoint the B01 gate will use:

    outputs/qwenbott_funnel_L12_d512/final.pt   (bottleneck_layer 12, dim 512,
                                                 hidden 4096, 36 layers, bf16)

Same three questions, on real weights:
  1. persist ON vs OFF read-side logits -- MEASURED max abs diff, no claim of
     bit-identity without the number;
  2. bytes/token MEASURED off the tensor WRITE returns, both paths;
  3. the arch_meta-driven CLI guards (bottleneck_dim>0, resume_j==b_layer+1).

1 GPU, bf16, no training. Loads the model ONCE and builds two QCMemModel views
over it (the flag lives on the wrapper, not the weights), so the two arms are
guaranteed to be the same parameters -- a second `torch.load` could not prove that.

Usage (check nvidia-smi first; this takes ~1 card and ~17 GB):
  CUDA_VISIBLE_DEVICES=0 python scripts/qcmem_bottleneck_persist_8b_smoke.py \
      --ckpt outputs/qwenbott_funnel_L12_d512/final.pt --json_out <path>
Capture rc as `cmd > file; echo $?` (a pipe would report the pipe's rc).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO not in sys.path:
    sys.path.insert(0, REPO)

from transformers import AutoModelForCausalLM                       # noqa: E402

from scripts.train_qwen_bottleneck_continued import inject_bottleneck  # noqa: E402
from src.memory.qcmem.qcmem_model import QCMemModel                 # noqa: E402

FAILURES: list = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(f"{name}: {detail}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/qwenbott_funnel_L12_d512/final.pt")
    ap.add_argument("--model_path", default="")
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--query_len", type=int, default=64)
    ap.add_argument("--json_out", default="")
    a = ap.parse_args()

    ckpt = os.path.join(REPO, a.ckpt) if not os.path.isabs(a.ckpt) else a.ckpt
    meta_path = os.path.join(os.path.dirname(ckpt), "arch_meta.json")
    for p in (ckpt, meta_path):
        if not os.path.exists(p):
            print(f"FATAL: missing {p}")
            return 2
    with open(meta_path) as f:
        meta = json.load(f)
    b_layer, b_dim = int(meta["bottleneck_layer"]), int(meta["bottleneck_dim"])
    base = a.model_path or meta["model_path"]
    j = b_layer + 1
    print(f"ckpt      : {ckpt} ({os.path.getsize(ckpt):,} B)")
    print(f"arch_meta : bottleneck_layer={b_layer} bottleneck_dim={b_dim} "
          f"hidden={meta['hidden_size']} layers={meta['num_hidden_layers']}")
    print(f"base      : {base}")
    print(f"resume_j  : {j} (== bottleneck_layer + 1)")
    if b_dim <= 0:
        print("FATAL: this arch_meta has bottleneck_dim=0 (the stock-continued "
              "control). The persist path needs the FUNNEL checkpoint.")
        return 2

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        base, dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True)
    inject_bottleneck(model, b_layer, b_dim, dtype)
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = sd.get("model_state", sd)
    missing, unexpected = model.load_state_dict(state, strict=False)
    bad = [k for k in missing if "inv_freq" not in k]
    model = model.to(dev).eval()
    print(f"loaded in {time.time() - t0:.1f}s  step={sd.get('step')}  "
          f"missing={len(bad)} unexpected={len(unexpected)}")
    check("checkpoint loaded with no unexpected/missing tensors",
          not bad and not unexpected, f"missing={bad[:5]} unexpected={unexpected[:5]}")
    funnel = model.model.layers[b_layer]
    check("layers[bottleneck_layer] is the funnel wrapper",
          hasattr(funnel, "down") and hasattr(funnel, "up"),
          f"type={type(funnel).__name__} "
          f"down={tuple(funnel.down.weight.shape) if hasattr(funnel, 'down') else None} "
          f"up={tuple(funnel.up.weight.shape) if hasattr(funnel, 'up') else None} "
          f"up_bias={getattr(funnel.up, 'bias', None) is not None}")

    # ONE model, TWO views -- same parameters by construction.
    qc_off = QCMemModel(model, resume_j=j)
    qc_on = QCMemModel(model, resume_j=j, persist_bottleneck_latent=True)

    V = int(model.config.vocab_size)
    g = torch.Generator().manual_seed(20260817)
    ids_chunks = [torch.randint(0, V, (a.chunk_size,), generator=g).to(dev)
                  for _ in range(a.chunks)]
    ids_query = torch.randint(0, V, (a.query_len,), generator=g).to(dev)
    sink_ids = [int(getattr(model.config, "bos_token_id", 151643) or 151643)]

    res = {"ckpt": ckpt, "arch_meta": meta, "resume_j": j, "device": dev,
           "chunks": a.chunks, "chunk_size": a.chunk_size,
           "query_len": a.query_len}

    def pipeline(qc):
        sink = qc.write_chunk(sink_ids)
        ctx = [qc.write_chunk(c) for c in ids_chunks]
        q = qc.write_chunk(ids_query)
        out = {"write_q": q,
               "read": qc.read(sink, ctx, q),
               "read_tail": qc.read_core(sink, ctx, q, logits_tail=8),
               "ctx_cat": torch.cat(ctx, dim=1)}
        q_hj, bottom, qpos = qc.write_prefill(ids_query)
        lg, top, pack = qc.read_prefill(sink, ctx, q_hj)
        outs = [lg]
        for k in range(3):
            tok = int(lg[0, -1].argmax().item())
            lg = qc.decode_step(tok, bottom, top, qpos + k, pack + k)
            outs.append(lg)
        out["decode"] = torch.cat(outs, dim=1)
        out["greedy_toks"] = [int(x[0, -1].argmax().item()) for x in outs]
        return out

    with torch.no_grad():
        t0 = time.time()
        off = pipeline(qc_off)
        t_off = time.time() - t0
        t0 = time.time()
        on = pipeline(qc_on)
        t_on = time.time() - t0
    print(f"pipeline wall: OFF {t_off:.2f}s  ON {t_on:.2f}s")

    # ---- 1. equivalence, MEASURED --------------------------------------
    per = {}
    for k in ("read", "read_tail", "decode"):
        d = (off[k].double() - on[k].double()).abs()
        per[k] = {"max_abs": float(d.max().item()),
                  "mean_abs": float(d.mean().item()),
                  "differing_elems": int((off[k] != on[k]).sum().item()),
                  "n": int(off[k].numel()),
                  "absmax_logit": float(off[k].abs().max().item())}
    res["equivalence"] = per
    worst_k = max(per, key=lambda k: per[k]["max_abs"])
    worst = per[worst_k]["max_abs"]
    scale = max(per[worst_k]["absmax_logit"], 1.0)
    tol = 1e-2 * scale                      # bf16, scaled to the logit magnitude
    check("8B real weights: persist ON == persist OFF (read-side logits)",
          worst <= tol,
          f"max_abs={worst:.6e} (worst {worst_k}, |logit|max={scale:.3f}, "
          f"tol={tol:.3e}); differing "
          + " ".join(f"{k}:{per[k]['differing_elems']}/{per[k]['n']}" for k in per))
    if all(per[k]["differing_elems"] == 0 for k in per):
        print(f"        -> BIT-IDENTICAL over "
              f"{sum(per[k]['n'] for k in per):,} logits")
    check("8B real weights: identical greedy tokens",
          off["greedy_toks"] == on["greedy_toks"],
          f"OFF={off['greedy_toks']} ON={on['greedy_toks']}")
    res["greedy_toks"] = {"off": off["greedy_toks"], "on": on["greedy_toks"]}

    # ---- 2. bytes/token, MEASURED --------------------------------------
    def bpt(h):
        return h.numel() * h.element_size() / h.shape[1]
    b_off, b_on = bpt(off["write_q"]), bpt(on["write_q"])
    ratio = b_off / b_on
    want = int(meta["hidden_size"]) / b_dim
    res["bytes_per_token"] = {
        "legacy_measured": b_off, "persist_measured": b_on,
        "ratio_measured": ratio, "ratio_expected": want,
        "legacy_shape": list(off["write_q"].shape),
        "persist_shape": list(on["write_q"].shape),
        "legacy_dtype": str(off["write_q"].dtype),
        "persist_dtype": str(on["write_q"].dtype),
        "api": [qc_off.store_bytes_per_token(), qc_on.store_bytes_per_token()],
        "ctx_total_bytes": [off["ctx_cat"].numel() * off["ctx_cat"].element_size(),
                            on["ctx_cat"].numel() * on["ctx_cat"].element_size()],
    }
    check("8B real weights: measured bytes/token ratio == hidden/d_bottle",
          abs(ratio - want) < 1e-9,
          f"{b_off:.0f} -> {b_on:.0f} B/tok = {ratio:.4f}x (expected {want:.4f}x); "
          f"shapes {tuple(off['write_q'].shape)} -> {tuple(on['write_q'].shape)}; "
          f"{a.chunks}x{a.chunk_size}-tok context store "
          f"{res['bytes_per_token']['ctx_total_bytes'][0]:,} -> "
          f"{res['bytes_per_token']['ctx_total_bytes'][1]:,} B")
    check("8B real weights: store_bytes_per_token() matches the measured tensors",
          res["bytes_per_token"]["api"] == [int(b_off), int(b_on)],
          f"api={res['bytes_per_token']['api']} measured=[{b_off}, {b_on}]")

    print("\n" + "=" * 72)
    if a.json_out:
        with open(a.json_out, "w") as f:
            json.dump({"failures": FAILURES, "results": res}, f, indent=1)
        print(f"wrote {a.json_out}")
    if FAILURES:
        print(f"FAILED ({len(FAILURES)}):")
        for x in FAILURES:
            print(f"  - {x}")
        return 1
    print("ALL CHECKS PASSED (8B real checkpoint)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
