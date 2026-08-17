#!/usr/bin/env python
"""MEASURE bytes/token of what WRITE hands to the store, for BOTH launched arms.

The gate's mandatory quantity is "bytes/token of what is written to the store (not of
the restored hidden)". store_bytes_per_token() computes that from the architecture;
this script instead measures it off the ACTUAL tensor write_chunk returns:

    bytes_per_token = h.numel() * h.element_size() / T

and then cross-checks the arithmetic helper against the measurement. Reporting the
helper's output alone would be "computed from the architecture", which is what the
gate explicitly says not to do.

Both arms are measured with the SAME context so the two numbers are commensurable:
  arm1 = stock backbone + Read-LoRA, resume_j=12, NO funnel      -> expect full width
  arm2 = funnel CPT ckpt, resume_j=13, --persist_bottleneck_latent -> expect d_bottle

1 GPU, two model loads. No writes outside --json_out.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

R = "/apdcephfs_wzz/share_303419932/pighzliu_code/Mixture-of-Memory"
sys.path.insert(0, R)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from src.memory.qcmem.qcmem_model import QCMemModel  # noqa: E402


def measure(qc, tok, n_ctx_tokens, chunk_size, label):
    """Write one chunk and measure the persisted tensor."""
    text = ("The archivist recorded that the brass key was hidden beneath the third "
            "flagstone of the west cloister, and that nobody had moved it since. ") * 200
    ids = tok.encode(text, add_special_tokens=False)[:n_ctx_tokens]
    ids = torch.tensor([ids], dtype=torch.long, device=qc.device)
    T = int(ids.shape[1])
    with torch.no_grad():
        h = qc.write_chunk(ids)
    numel = int(h.numel())
    esz = int(h.element_size())
    total = numel * esz
    bpt = total / T
    helper = qc.store_bytes_per_token()
    print(f"[{label}] write tensor shape={tuple(h.shape)} dtype={h.dtype}", flush=True)
    print(f"[{label}] MEASURED bytes/token = {numel} * {esz} / {T} = {bpt}", flush=True)
    print(f"[{label}] helper store_bytes_per_token() = {helper}", flush=True)
    return {
        "label": label,
        "write_shape": list(h.shape),
        "write_dtype": str(h.dtype),
        "T_tokens": T,
        "numel": numel,
        "element_size": esz,
        "total_bytes": total,
        "measured_bytes_per_token": bpt,
        "helper_store_bytes_per_token": helper,
        "helper_agrees_with_measurement": abs(bpt - helper) < 1e-9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True)
    ap.add_argument("--funnel_ckpt", required=True)
    ap.add_argument("--funnel_meta", required=True)
    ap.add_argument("--read_lora", required=True)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--json_out", required=True)
    args = ap.parse_args()

    dev = "cuda:0"
    dtype = torch.bfloat16
    tok = AutoTokenizer.from_pretrained(args.base, trust_remote_code=True,
                                        local_files_only=True)
    results = []

    # ---------------- arm 1: stock + Read-LoRA, resume_j=12, no funnel ----------
    print("[arm1] loading stock backbone ...", flush=True)
    m1 = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True).to(dev).eval()
    from peft import PeftModel
    p1 = PeftModel.from_pretrained(m1, args.read_lora).eval()
    m1 = p1.base_model.model
    qc1 = QCMemModel(m1, resume_j=12, top_prepay_b=0)
    results.append(measure(qc1, tok, args.chunk_size, args.chunk_size,
                           "arm1_stock_readlora"))
    del qc1, m1, p1
    torch.cuda.empty_cache()

    # ---------------- arm 2: funnel CPT + persist, resume_j=13 ------------------
    print("[arm2] loading funnel CPT ...", flush=True)
    with open(args.funnel_meta) as f:
        meta = json.load(f)
    b_layer, b_dim = int(meta["bottleneck_layer"]), int(meta["bottleneck_dim"])
    m2 = AutoModelForCausalLM.from_pretrained(
        args.base, torch_dtype=dtype, attn_implementation="sdpa",
        trust_remote_code=True, local_files_only=True).to(dev).eval()
    from scripts.train_qwen_bottleneck_continued import inject_bottleneck
    inject_bottleneck(m2, b_layer, b_dim, dtype)
    ck = torch.load(args.funnel_ckpt, map_location="cpu")
    state = ck.get("model_state", ck)
    missing, unexpected = m2.load_state_dict(state, strict=False)
    bad = [k for k in missing if "inv_freq" not in k]
    if bad or unexpected:
        print(f"[arm2][WARN] missing={bad[:5]} unexpected={unexpected[:5]}", flush=True)
    m2 = m2.to(dev).eval()
    qc2 = QCMemModel(m2, resume_j=b_layer + 1, top_prepay_b=0,
                     persist_bottleneck_latent=True)
    results.append(measure(qc2, tok, args.chunk_size, args.chunk_size,
                           "arm2_bottleneck_persist"))

    a1 = results[0]["measured_bytes_per_token"]
    a2 = results[1]["measured_bytes_per_token"]
    out = {
        "definition": ("numel * element_size() / T of the tensor write_chunk RETURNS "
                       "-- i.e. of what is written to the store, NOT of the restored "
                       "hidden. Measured, not derived from the architecture."),
        "arms": results,
        "arm1_bytes_per_token": a1,
        "arm2_bytes_per_token": a2,
        "ratio_arm1_over_arm2": (a1 / a2) if a2 else None,
        "hidden_size": int(results[0]["numel"] / results[0]["T_tokens"]),
        "all_helpers_agree": all(r["helper_agrees_with_measurement"] for r in results),
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps({k: out[k] for k in
                      ("arm1_bytes_per_token", "arm2_bytes_per_token",
                       "ratio_arm1_over_arm2", "all_helpers_agree")}, indent=2), flush=True)
    print(f"wrote {args.json_out}", flush=True)
    return 0 if out["all_helpers_agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
