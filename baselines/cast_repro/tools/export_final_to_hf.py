#!/usr/bin/env python3
"""Export a finalized CAST checkpoint (``final_sparse.pt``) to a plain HF model.

    python baselines/cast_repro/tools/export_final_to_hf.py \
        --ckpt outputs/cast_repro_zero2/final_sparse.pt \
        --output outputs/cast_repro_zero2/hf_final

After ``finalize_all()`` every ``CastSparseLinear`` is numerically a bare
``nn.Linear``: the mask has been applied to the weight (so masked entries are
*exact* zeros) and ``cast_scale`` has been folded in and reset to all-ones.
Therefore the only difference between the training state dict and a vanilla
LLaMA state dict is the two extra buffers per in-block projection
(``.mask``, ``.cast_scale``), which are dropped here.

This exists so the finished run can be scored by the *same* harness used for
every other baseline (``baselines/eval_hf_sparse_model.py``), rather than a
CAST-specific evaluation path.  Two invariants are asserted before writing,
because a silently-wrong export would be indistinguishable from a bad result:

1. the surviving key set equals the reference model's key set, modulo the
   non-persistent ``rotary_emb.inv_freq`` buffers, and
2. all 224 in-block projections are still *exactly* 2:4 sparse.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM

#: The 7 in-block projections that CAST puts in scope (see SPEC.md S1).
PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
EXPECTED_CAST_TENSORS = 224
EXPECTED_CAST_ELEMENTS = 6_476_005_376


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="finalized checkpoint (final_sparse.pt)")
    ap.add_argument("--output", required=True)
    ap.add_argument("--model", default="models/Llama--Llama2-7b",
                    help="dense reference, for config/tokenizer and key-set check")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
                    help="storage dtype; the eval harness loads in bf16 anyway")
    args = ap.parse_args()

    root = Path(args.project_root)
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path
    ckpt_path = Path(args.ckpt)
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = root / out_path

    print(f"[export] loading {ckpt_path}", flush=True)
    blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not blob.get("final"):
        raise SystemExit(
            f"{ckpt_path} is not a finalized checkpoint (no 'final' flag). "
            "Masked entries are only exact zeros after finalize_all(); "
            "exporting a pre-finalize checkpoint would silently score a DENSE model."
        )
    state = blob["model"]

    # ---- drop the CAST-only buffers -------------------------------------
    dropped = [k for k in state if k.endswith(".mask") or k.endswith(".cast_scale")]
    clean = {k: v for k, v in state.items() if not (k.endswith(".mask") or k.endswith(".cast_scale"))}
    print(f"[export] dropped {len(dropped)} CAST buffers, kept {len(clean)} tensors", flush=True)

    # ---- invariant 1: key set matches the dense reference ----------------
    index = json.loads((model_path / "model.safetensors.index.json").read_text())["weight_map"]
    ref = set(index)
    missing = ref - set(clean)
    extra = set(clean) - ref
    # inv_freq is a non-persistent buffer recomputed from config on load.
    missing = {k for k in missing if not k.endswith("rotary_emb.inv_freq")}
    if missing or extra:
        raise SystemExit(f"key-set mismatch: missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}")
    print("[export] key set matches dense reference (modulo rotary inv_freq)", flush=True)

    # ---- invariant 2: still exactly 2:4 ---------------------------------
    n_tensors = 0
    n_elements = 0
    violations = 0
    zeros = 0
    for k, v in clean.items():
        if not (k.endswith(".weight") and any(f".{p}." in k for p in PROJECTIONS)):
            continue
        n_tensors += 1
        n_elements += v.numel()
        w = v.float()
        r, c = w.shape
        nz = (w != 0).reshape(r, c // 4, 4).sum(-1)
        violations += int((nz != 2).sum())
        zeros += int((w == 0).sum())
    print(f"[export] cast scope: {n_tensors} tensors, {n_elements:,} elements, "
          f"zero_fraction={zeros / max(1, n_elements):.6f}, exact-2:4 violations={violations}",
          flush=True)
    if n_tensors != EXPECTED_CAST_TENSORS or n_elements != EXPECTED_CAST_ELEMENTS:
        raise SystemExit(
            f"scope mismatch: {n_tensors}/{n_elements} != "
            f"{EXPECTED_CAST_TENSORS}/{EXPECTED_CAST_ELEMENTS}"
        )
    if violations:
        raise SystemExit(f"{violations} groups are not exactly 2:4 -- refusing to export")

    # ---- write ----------------------------------------------------------
    dtype = getattr(torch, args.dtype)
    print(f"[export] materializing reference skeleton ({args.dtype})", flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=dtype, low_cpu_mem_usage=True, local_files_only=True,
    )
    incompat = model.load_state_dict({k: v.to(dtype) for k, v in clean.items()}, strict=False)
    unexpected = list(incompat.unexpected_keys)
    still_missing = [k for k in incompat.missing_keys if not k.endswith("rotary_emb.inv_freq")]
    if unexpected or still_missing:
        raise SystemExit(f"load_state_dict mismatch: unexpected={unexpected[:8]} missing={still_missing[:8]}")

    # Re-verify AFTER the dtype cast: bf16 rounding must not resurrect a zero.
    post_viol = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or mod.weight.ndim != 2:
            continue
        if not any(name.endswith(p) for p in PROJECTIONS):
            continue
        w = mod.weight.detach().float()
        r, c = w.shape
        nz = (w != 0).reshape(r, c // 4, 4).sum(-1)
        post_viol += int((nz != 2).sum())
    print(f"[export] post-cast exact-2:4 violations: {post_viol}", flush=True)
    if post_viol:
        raise SystemExit(f"{post_viol} groups broke 2:4 after the {args.dtype} cast")

    out_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_path, safe_serialization=True)

    # Copy the tokenizer files VERBATIM instead of round-tripping through
    # AutoTokenizer.save_pretrained.  CAST never touches the vocabulary, so the
    # tokenizer must be bit-identical to the dense reference's -- and a
    # round-trip through a newer transformers silently drops the SentencePiece
    # `tokenizer.model`, which then makes `use_fast=False` (what the shared eval
    # harness uses) fail to load.
    copied = []
    for fname in ("tokenizer.model", "tokenizer.json", "tokenizer_config.json",
                  "special_tokens_map.json"):
        src = model_path / fname
        if src.exists():
            shutil.copyfile(src, out_path / fname)
            copied.append(fname)
    if "tokenizer.model" not in copied:
        raise SystemExit(
            f"{model_path} has no tokenizer.model; the eval harness loads with "
            "use_fast=False and would fail. Refusing to export a broken tokenizer."
        )
    print(f"[export] copied tokenizer files verbatim: {copied}", flush=True)
    (out_path / "cast_export_meta.json").write_text(json.dumps({
        "source_ckpt": str(ckpt_path),
        "source_step": blob.get("step"),
        "dense_reference": str(model_path),
        "dtype": args.dtype,
        "cast_tensors": n_tensors,
        "cast_elements": n_elements,
        "cast_zero_fraction": round(zeros / max(1, n_elements), 6),
        "exact_2of4_violations": violations,
        "exact_2of4_violations_post_cast": post_viol,
        "dropped_buffers": len(dropped),
    }, indent=2))
    print(f"[export] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
