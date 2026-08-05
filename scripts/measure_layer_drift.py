#!/usr/bin/env python3
"""
scripts/measure_layer_drift.py  — Paper C M1

Per-layer ΔW relative Frobenius norm + linear CKA drift for Paper-B OLMo-2
checkpoints vs the OLMo-2-1124-7B base.

Part A  (CPU-only, no GPU required):
  delta_w[i] = sqrt(sum_j ||W_arm[i,j] - W_base[i,j]||_F^2)
             / sqrt(sum_j ||W_base[i,j]||_F^2)
  summed over the 11 parameter tensors in each OLMo-2 decoder layer.
  Only compares arm layers i in [0, n_kept); fresh layers (i >= n_kept) are
  skipped / marked N/A.
  For freeze_front arms, prints a WARNING if any kept-layer drift > 1e-4
  (correctness check: frozen layers should be exactly 0).

Part B  (GPU forward pass; skipped gracefully if CUDA unavailable):
  Linear CKA between base and arm residual stream outputs at each kept layer.
  CKA = 1.0 means identical representations; lower = more drift.
  Loads models sequentially (arm then base) to limit peak GPU memory.

Usage (run on .venv python once a GPU frees):
  .venv/bin/python scripts/measure_layer_drift.py \
      --arm_ckpt outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt \
      --arm_name healed \
      --output results/layer_drift_healed.json

Checkpoint format expected:
  {model_state: state_dict, step: int, optimizer_state: ..., ...}
  (matches train_olmo2_arch_probe2.py _save())
  Falls back gracefully if the .pt is a raw state dict.

OLMo-2 layer layout (verified against train_olmo2_arch_probe2.py):
  11 tensors per decoder layer (POST-norm, no input_layernorm):
    self_attn.{q,k,v,o}_proj.weight
    self_attn.{q,k}_norm.weight
    mlp.{gate,up,down}_proj.weight
    post_attention_layernorm.weight
    post_feedforward_layernorm.weight
  3 non-layer keys: model.embed_tokens.weight, model.norm.weight, lm_head.weight
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import warnings

import numpy as np
import torch

# ---------------------------------------------------------------------------
# OLMo-2 layer tensor suffixes (verified 2026-07-16 in train_olmo2_arch_probe2.py)
# ---------------------------------------------------------------------------
_LAYER_SUFFIXES = [
    "self_attn.q_proj.weight",
    "self_attn.k_proj.weight",
    "self_attn.v_proj.weight",
    "self_attn.o_proj.weight",
    "self_attn.q_norm.weight",
    "self_attn.k_norm.weight",
    "mlp.gate_proj.weight",
    "mlp.up_proj.weight",
    "mlp.down_proj.weight",
    "post_attention_layernorm.weight",
    "post_feedforward_layernorm.weight",
]


def _lk(layer_idx: int, suffix: str) -> str:
    """Full state-dict key for a layer tensor."""
    return f"model.layers.{layer_idx}.{suffix}"


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_arm_state_dict(ckpt_path: str) -> dict:
    """Load .pt checkpoint and return the model state dict.

    Expected format: {'model_state': sd, 'step': N, 'optimizer_state': ..., ...}
    Falls back to treating the dict as a raw state dict if 'model_state' is absent.
    Explicitly frees optimizer state to reduce peak RAM.
    """
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    print(f"  torch.load({os.path.basename(ckpt_path)}, map_location=cpu) ...")
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if not isinstance(ck, dict):
        raise TypeError(
            f"Checkpoint is a {type(ck).__name__}, expected dict. Path: {ckpt_path}"
        )

    if "model_state" in ck:
        meta_keys = [k for k in ck.keys() if k not in ("model_state", "optimizer_state")]
        print(f"  ckpt meta keys: {meta_keys}")
        sd = ck["model_state"]
        # Free optimizer state immediately (can be 2× the model size in RAM)
        ck.pop("optimizer_state", None)
        del ck
        gc.collect()
        return sd

    # Fallback: check if it looks like a raw state dict
    sample = list(ck.keys())[:5]
    if any(k.startswith("model.") or k.startswith("lm_head") for k in sample):
        print(f"  [note] no 'model_state' key; treating as raw state dict. sample: {sample}")
        return ck

    # Unknown format — show keys and raise
    raise ValueError(
        f"Cannot parse checkpoint. Top-level keys: {list(ck.keys())[:15]}\n"
        f"Expected a 'model_state' key or a raw state dict."
    )


# ---------------------------------------------------------------------------
# Part A helpers: selective base tensor loading via safetensors
# ---------------------------------------------------------------------------

def _load_base_tensors_safetensors(base_path: str, layer_indices: list) -> dict:
    """Load only the needed layer tensors from the base model's safetensors shards.
    Avoids pulling the full 7B model into RAM for Part A (CPU-only ΔW).
    """
    index_path = os.path.join(base_path, "model.safetensors.index.json")
    single_path = os.path.join(base_path, "model.safetensors")

    # --- collect needed keys ---
    needed = set()
    for i in layer_indices:
        for s in _LAYER_SUFFIXES:
            needed.add(_lk(i, s))

    # --- single-shard fallback ---
    if not os.path.exists(index_path) and os.path.exists(single_path):
        print(f"  [base] single safetensors shard found")
        from safetensors.torch import load_file
        sd = load_file(single_path, device="cpu")
        return {k: v for k, v in sd.items() if k in needed}

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No model.safetensors.index.json or model.safetensors at {base_path}"
        )

    with open(index_path) as f:
        index = json.load(f)
    weight_map = index["weight_map"]  # key -> shard filename

    # Group needed keys by shard
    shard_to_keys: dict = {}
    missing_from_index = []
    for k in needed:
        if k in weight_map:
            shard_to_keys.setdefault(weight_map[k], []).append(k)
        else:
            missing_from_index.append(k)
    if missing_from_index:
        print(f"  [base] {len(missing_from_index)} keys not in safetensors index "
              f"(unexpected for OLMo-2): {missing_from_index[:4]}")

    from safetensors.torch import safe_open
    result = {}
    for shard_fname, keys in sorted(shard_to_keys.items()):
        shard_path = os.path.join(base_path, shard_fname)
        print(f"  [base] {shard_fname}: loading {len(keys)} tensors")
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for k in keys:
                result[k] = f.get_tensor(k)
    return result


def load_base_layer_tensors(base_path: str, layer_indices: list) -> dict:
    """Load base model tensors for the given layer indices.
    Prefers selective safetensors loading; falls back to full from_pretrained on error."""
    try:
        import safetensors  # noqa: F401 — check availability
        return _load_base_tensors_safetensors(base_path, layer_indices)
    except (ImportError, ModuleNotFoundError):
        print("  [base] safetensors not available; falling back to full from_pretrained "
              "(loads full 7B model to CPU — requires ~28 GB RAM)")
    except Exception as e:
        print(f"  [base] safetensors load failed ({e}); falling back to from_pretrained")

    from transformers import Olmo2ForCausalLM
    model = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, local_files_only=True
    )
    sd = model.state_dict()
    needed = {_lk(i, s) for i in layer_indices for s in _LAYER_SUFFIXES}
    result = {k: v for k, v in sd.items() if k in needed}
    del model, sd
    gc.collect()
    return result


# ---------------------------------------------------------------------------
# Part A: per-layer ΔW relative Frobenius norm
# ---------------------------------------------------------------------------

def compute_delta_w(
    arm_sd: dict,
    base_tensors: dict,
    n_kept: int,
    arm_name: str,
) -> list:
    """Compute per-layer relative drift for arm layers 0..n_kept-1.

    Returns a list of dicts: {layer, delta_w_rel, [missing]}.
    Also prints a freeze_front correctness check.
    """
    results = []
    for i in range(n_kept):
        num_sq = 0.0  # sum ||delta_ij||_F^2 over tensors j in layer i
        den_sq = 0.0  # sum ||W_base_ij||_F^2
        missing = []
        for s in _LAYER_SUFFIXES:
            k = _lk(i, s)
            wa = arm_sd.get(k)
            wb = base_tensors.get(k)
            if wa is None:
                missing.append(f"arm:{k}")
                continue
            if wb is None:
                missing.append(f"base:{k}")
                continue
            wa = wa.float()
            wb = wb.float()
            diff = wa - wb
            num_sq += float(diff.norm(p="fro") ** 2)
            den_sq += float(wb.norm(p="fro") ** 2)

        if den_sq > 0:
            rel_drift = (num_sq ** 0.5) / (den_sq ** 0.5)
        else:
            rel_drift = float("nan")

        entry: dict = {"layer": i, "delta_w_rel": rel_drift}
        if missing:
            entry["missing"] = missing
        results.append(entry)

    # freeze_front correctness check: kept layers should be numerically unchanged
    is_frozen_arm = "freeze" in arm_name.lower() or "frozen" in arm_name.lower()
    if is_frozen_arm:
        valid_drifts = [r["delta_w_rel"] for r in results
                        if not (isinstance(r["delta_w_rel"], float) and
                                r["delta_w_rel"] != r["delta_w_rel"])]  # exclude nan
        if valid_drifts:
            max_drift = max(valid_drifts)
            if max_drift > 1e-4:
                print(f"\n  [WARNING] freeze_front kept-layer max drift = {max_drift:.4e} > 1e-4"
                      f" — the checkpoint may not actually be frozen or is from a different base!")
            else:
                print(f"\n  [CHECK OK] freeze_front kept-layer max drift = {max_drift:.4e} < 1e-4"
                      f" (correctly frozen)")

    return results


# ---------------------------------------------------------------------------
# Part B: CKA helpers
# ---------------------------------------------------------------------------

def _linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    """Centred linear CKA(X, Y) for n×d matrices X and Y.

    CKA = ||Y_c^T X_c||_F^2 / (||X_c^T X_c||_F * ||Y_c^T Y_c||_F)
    where X_c = X - mean_col(X).  Computed in feature space (d×d inner products)
    which is efficient when d (hidden_size 4096) is manageable on GPU.
    """
    X = X.float()
    Y = Y.float()
    # Column-centre
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    # d×d cross-products
    XtX = X.t().mm(X)   # (d, d)
    YtY = Y.t().mm(Y)   # (d, d)
    YtX = Y.t().mm(X)   # (d, d)
    num = float((YtX ** 2).sum())
    den = float(XtX.norm(p="fro")) * float(YtY.norm(p="fro"))
    if den == 0.0:
        return float("nan")
    return num / den


def _capture_residuals(
    model,
    input_ids: torch.Tensor,
    n_layers: int,
) -> list:
    """Forward pass with hooks; capture residual stream output of each block.

    Returns list of n_layers tensors, each (B, T, H), on CPU.
    Uses model.model.layers[i].register_forward_hook; the hook captures
    out[0] (hidden_states) from the layer's return tuple.
    """
    hidden_cache: dict = {}
    hooks = []

    def _make_hook(idx: int):
        def _hook(module, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            hidden_cache[idx] = t.detach().cpu()
        return _hook

    decoder_layers = model.model.layers
    for i in range(min(n_layers, len(decoder_layers))):
        hooks.append(decoder_layers[i].register_forward_hook(_make_hook(i)))

    with torch.no_grad():
        model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    return [hidden_cache[i] for i in range(min(n_layers, len(decoder_layers)))]


def _build_arm_for_cka(
    base_path: str,
    arm_sd: dict,
    n_kept: int,
    n_fresh: int,
    device: torch.device,
):
    """Reconstruct Olmo2ForCausalLM for the arm (n_kept+n_fresh layers) and load weights."""
    from transformers import Olmo2Config, Olmo2ForCausalLM
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    cfg.num_hidden_layers = n_kept + n_fresh
    cfg.use_cache = False
    model = Olmo2ForCausalLM(cfg)
    missing, unexpected = model.load_state_dict(arm_sd, strict=True)
    if missing or unexpected:
        raise ValueError(
            f"arm state dict mismatch: missing={missing[:4]} unexpected={unexpected[:4]}"
        )
    model = model.to(torch.float32).to(device).eval()
    model.config.use_cache = False
    return model


def _build_base_for_cka(base_path: str, device: torch.device):
    """Load full base OLMo-2 model for CKA forward pass."""
    from transformers import Olmo2ForCausalLM
    model = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=torch.float32, local_files_only=True
    ).to(device).eval()
    model.config.use_cache = False
    return model


def compute_cka(
    base_path: str,
    arm_sd: dict,
    n_kept: int,
    n_fresh: int,
    cka_data_path: str,
    cka_batch: int,
    cka_seqlen: int,
    device: torch.device,
) -> tuple:
    """Compute per-layer linear CKA (arm vs base) for layers 0..n_kept-1.

    Returns (list of {layer, cka}, notes_str).
    Loads arm first (smaller), captures hiddens, deletes arm, then loads base.
    """
    # --- load data ---
    print(f"\n  [CKA] loading data: {os.path.basename(cka_data_path)}")
    arr = np.load(cka_data_path, mmap_mode="r")  # (n_rows, chunk_len), uint32
    n_rows = min(cka_batch, arr.shape[0])
    T = min(cka_seqlen, arr.shape[1])
    rows = arr[:n_rows, :T].astype(np.int64)
    input_ids = torch.tensor(rows, dtype=torch.long).to(device)
    print(f"  [CKA] input_ids shape: {input_ids.shape} (batch={n_rows}, seqlen={T})")

    # --- arm hidden states ---
    print(f"  [CKA] building arm model ({n_kept}+{n_fresh} layers) ...")
    arm_model = _build_arm_for_cka(base_path, arm_sd, n_kept, n_fresh, device)
    print(f"  [CKA] capturing arm residual states ...")
    arm_hiddens = _capture_residuals(arm_model, input_ids, n_kept)
    del arm_model
    torch.cuda.empty_cache()
    gc.collect()

    # --- base hidden states ---
    print(f"  [CKA] loading base model (32 layers) ...")
    base_model = _build_base_for_cka(base_path, device)
    print(f"  [CKA] capturing base residual states ...")
    base_hiddens = _capture_residuals(base_model, input_ids, n_kept)
    del base_model
    torch.cuda.empty_cache()
    gc.collect()

    # --- compute CKA per layer ---
    print(f"  [CKA] computing linear CKA for {n_kept} layers ...")
    results = []
    for i in range(n_kept):
        H_arm = arm_hiddens[i].reshape(-1, arm_hiddens[i].shape[-1])   # (B*T, H)
        H_base = base_hiddens[i].reshape(-1, base_hiddens[i].shape[-1])
        cka_val = _linear_cka(H_arm, H_base)
        results.append({"layer": i, "cka": cka_val})

    # fresh layers: no CKA (no base counterpart in meaning)
    for i in range(n_kept, n_kept + n_fresh):
        results.append({"layer": i, "cka": None, "note": "fresh_no_base_counterpart"})

    notes = (f"centred linear CKA, {n_rows} seqs × {T} tokens = {n_rows*T} positions; "
             f"data={os.path.basename(cka_data_path)}")
    return results, notes


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Paper C M1 — per-layer ΔW norm + CKA drift for OLMo-2 arm checkpoints.\n"
            "Part A runs on CPU (no GPU needed). "
            "Part B (CKA) requires CUDA and is skipped gracefully if unavailable."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--base_path", type=str,
        default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B",
        help="Path to OLMo-2-1124-7B base model (HF format with safetensors). "
             "Default: %(default)s",
    )
    p.add_argument(
        "--arm_ckpt", type=str, required=True,
        help="Path to arm .pt checkpoint (train_olmo2_arch_probe2.py format: "
             "{model_state: sd, step: N, ...})",
    )
    p.add_argument(
        "--arm_name", type=str, required=True,
        help="Human-readable arm label (e.g. 'healed', 'freeze_front', 'fromscratch'). "
             "Used in output JSON and freeze_front correctness check (name contains 'freeze').",
    )
    p.add_argument(
        "--n_kept", type=int, default=14,
        help="Number of kept front layers transplanted from base (default: 14). "
             "ΔW and CKA are computed for layers 0..n_kept-1.",
    )
    p.add_argument(
        "--n_fresh", type=int, default=2,
        help="Number of fresh grafted top layers (default: 2). "
             "These have no base counterpart and are skipped/marked N/A.",
    )
    p.add_argument(
        "--device", type=str, default="cuda:0",
        help="Device for CKA forward pass (default: cuda:0). "
             "Part A (ΔW) always runs on CPU regardless of this setting.",
    )
    p.add_argument(
        "--cka_data", type=str,
        default=(
            "/apdcephfs_wzc1/share_304376610/pighzliu_code/"
            "Mixture-of-Memory/data/dolmino_now_val.npy"
        ),
        help="Path to OLMo-2-tokenised .npy file for CKA (shape: n_rows × chunk_len, "
             "dtype uint32). Default: dolmino_now_val.npy",
    )
    p.add_argument(
        "--cka_batch", type=int, default=16,
        help="Number of sequences (rows) to use for CKA (default: 16).",
    )
    p.add_argument(
        "--cka_seqlen", type=int, default=512,
        help="Sequence length (tokens) per sequence for CKA (default: 512). "
             "Truncated from each row.",
    )
    p.add_argument(
        "--output", type=str, required=True,
        help="Output JSON path (e.g. results/layer_drift_healed.json).",
    )
    return p.parse_args()


def main():
    args = _parse_args()

    # -----------------------------------------------------------------------
    # CUDA availability check (determines whether Part B runs)
    # -----------------------------------------------------------------------
    use_cuda = torch.cuda.is_available() and "cuda" in args.device
    if not use_cuda:
        if "cuda" in args.device:
            print("[INFO] CUDA not available — Part B (CKA) will be SKIPPED. "
                  "Part A (ΔW relative norm) runs on CPU and is still complete.")
        else:
            print("[INFO] CPU-only mode requested — Part B (CKA) SKIPPED.")
    device = torch.device(args.device if use_cuda else "cpu")

    print(f"\n{'='*65}")
    print(f"Paper C M1 — measure_layer_drift.py")
    print(f"  arm_name : {args.arm_name}")
    print(f"  arm_ckpt : {args.arm_ckpt}")
    print(f"  base     : {args.base_path}")
    print(f"  n_kept   : {args.n_kept}   n_fresh : {args.n_fresh}")
    print(f"  device   : {args.device}   use_cuda: {use_cuda}")
    print(f"  output   : {args.output}")
    print(f"{'='*65}\n")

    # -----------------------------------------------------------------------
    # Load arm checkpoint (CPU)
    # -----------------------------------------------------------------------
    print(f"[A] Loading arm state dict ...")
    arm_sd = load_arm_state_dict(args.arm_ckpt)

    arm_layer_ids = sorted({
        int(k.split(".")[2])
        for k in arm_sd if k.startswith("model.layers.")
    })
    expected_ids = list(range(args.n_kept + args.n_fresh))
    if arm_layer_ids != expected_ids:
        print(f"  [WARNING] arm layer ids {arm_layer_ids} != expected {expected_ids}")
    else:
        print(f"  arm state dict: {len(arm_sd)} tensors, "
              f"layers {arm_layer_ids[0]}..{arm_layer_ids[-1]} (OK)")

    # -----------------------------------------------------------------------
    # Part A: ΔW relative norm (CPU-only)
    # -----------------------------------------------------------------------
    print(f"\n[A] Part A: ΔW relative norm (CPU)")
    print(f"  Loading base layer tensors for layers 0..{args.n_kept - 1} ...")
    base_tensors = load_base_layer_tensors(args.base_path, list(range(args.n_kept)))
    print(f"  Loaded {len(base_tensors)} base tensors.")

    print(f"  Computing per-layer Frobenius relative drift ...")
    delta_w_results = compute_delta_w(arm_sd, base_tensors, args.n_kept, args.arm_name)
    del base_tensors
    gc.collect()

    # Print Part A table
    print(f"\n{'─'*50}")
    print(f"  Part A: ΔW relative norm  [{args.arm_name}]")
    print(f"  {'layer':>5}  {'delta_w_rel':>14}  {'notes':}")
    print(f"  {'─────':>5}  {'──────────':>14}")
    for r in delta_w_results:
        note = f"  missing: {r.get('missing')}" if r.get("missing") else ""
        print(f"  {r['layer']:>5}  {r['delta_w_rel']:>14.6f}{note}")
    for i in range(args.n_kept, args.n_kept + args.n_fresh):
        print(f"  {i:>5}  {'N/A (fresh)':>14}")
    print(f"{'─'*50}")

    # -----------------------------------------------------------------------
    # Part B: CKA (GPU, optional)
    # -----------------------------------------------------------------------
    cka_results: list = []
    cka_notes = ""

    if not use_cuda:
        cka_notes = "skipped: CUDA unavailable (Part A ΔW still complete)"
        print(f"\n[B] Part B (CKA): {cka_notes}")
        for i in range(args.n_kept):
            cka_results.append({"layer": i, "cka": None, "note": "cuda_unavailable"})
        for i in range(args.n_kept, args.n_kept + args.n_fresh):
            cka_results.append({"layer": i, "cka": None, "note": "fresh_and_cuda_unavailable"})

    else:
        if not os.path.exists(args.cka_data):
            cka_notes = f"skipped: CKA data file not found: {args.cka_data}"
            print(f"\n[B] Part B (CKA): {cka_notes}")
            for i in range(args.n_kept + args.n_fresh):
                cka_results.append({"layer": i, "cka": None, "note": "data_file_missing"})
        else:
            print(f"\n[B] Part B: linear CKA  [{args.arm_name}]")
            try:
                cka_results, cka_notes = compute_cka(
                    args.base_path,
                    arm_sd,
                    args.n_kept,
                    args.n_fresh,
                    args.cka_data,
                    args.cka_batch,
                    args.cka_seqlen,
                    device,
                )
                # Print CKA table
                print(f"\n{'─'*50}")
                print(f"  Part B: linear CKA  [{args.arm_name}]")
                print(f"  {'layer':>5}  {'cka':>10}")
                print(f"  {'─────':>5}  {'────────':>10}")
                for r in cka_results:
                    if r["cka"] is None:
                        print(f"  {r['layer']:>5}  {'N/A':>10}  ({r.get('note','')})")
                    else:
                        print(f"  {r['layer']:>5}  {r['cka']:>10.4f}")
                print(f"{'─'*50}")
            except Exception as exc:
                import traceback
                cka_notes = f"error: {exc}"
                print(f"\n[B] CKA error: {exc}")
                traceback.print_exc()
                for i in range(args.n_kept + args.n_fresh):
                    cka_results.append({"layer": i, "cka": None, "note": f"error: {exc}"})

    # -----------------------------------------------------------------------
    # Write output JSON
    # -----------------------------------------------------------------------
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)

    out = {
        "arm_name": args.arm_name,
        "base_path": args.base_path,
        "arm_ckpt": args.arm_ckpt,
        "n_kept": args.n_kept,
        "n_fresh": args.n_fresh,
        "per_layer_delta_w": delta_w_results,
        "per_layer_cka": cka_results,
        "notes": {
            "delta_w_formula": (
                "delta_w_rel[i] = sqrt(sum_j ||W_arm[i,j] - W_base[i,j]||_F^2) "
                "/ sqrt(sum_j ||W_base[i,j]||_F^2), "
                "summed over 11 tensors per OLMo-2 decoder layer"
            ),
            "cka": cka_notes if cka_notes else (
                "centred linear CKA between residual stream (post-block) representations; "
                "1.0=identical 0.0=orthogonal"
            ),
            "cka_batch": args.cka_batch,
            "cka_seqlen": args.cka_seqlen,
            "cka_data": args.cka_data,
        },
    }
    with open(args.output, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[done] Results written to {args.output}")


if __name__ == "__main__":
    main()
