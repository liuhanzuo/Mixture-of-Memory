"""§5.4 static probe — Branch-3 A.2 bypass parity at step 0.

Discriminator between H1/H3 and H2/H5 for the step-1 PPL=1001 pollution
observed in run `Branch-3 A.2 attempt 2 (21:13)`. See
`ops/research_notes/20260426_branch3_A2_pollution_debug.md` §5.4.

Rationale
---------
The bypass-parity argument says: with `slot_output_gate = 0` we have
`alpha = tanh(0) = 0` exactly (bf16-safe), so

    next_hidden = bypass_h + alpha · slot_delta  =  bypass_h

and `bypass_h` is produced by calling `wrapped_layer(hidden_states, ...)` —
which should be bit-exact equal to vanilla Llama decoder-layer output.

If the step-0 forward shows `max_abs_err(next_hidden_L, vanilla_L_out) < 1e-4`
at every layer, bypass parity holds → step-1 pollution must come from a
post-step-1 amplification (H1 σ-kick or H3 32-deep autograd). Dispatch
Experiment A (σ=0.02 + warmup=500) and Experiment B (no_shared_bank).

If `max_abs_err(next_hidden_L, vanilla_L_out) > 1e-3` at L=0 (or grows with
L), bypass parity is broken at step-0 already → H2 (`**kwargs`/SDPA-kernel
dispatch difference in the dual `wrapped_layer(...)` call at `layer.py:399`)
or H5 (bf16 rounding on `0·slot_delta`). Dispatch a targeted unit test on
the dual call — DO NOT train.

Reports `max_abs_err` at L ∈ {0, 8, 16, 24, 31} (4 Llama-3-8B waypoints).

Usage (single GPU, ~5 min wall on B200):
    python scripts/probe_branch3_bypass_parity.py \
        --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
        --data_path  /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/pg19_chunks_llama3.npy \
        --seq_len 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from transformers import AutoModelForCausalLM

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model


def _load_chunk(data_path: str, seq_len: int, device: torch.device) -> torch.Tensor:
    """Load one pg19 chunk of `seq_len` tokens."""
    arr = np.load(data_path, mmap_mode="r")
    if arr.ndim != 2:
        raise RuntimeError(f"expected [N, L] int64/int32 array, got shape {arr.shape}")
    if arr.shape[1] < seq_len:
        raise RuntimeError(f"pg19 chunk len {arr.shape[1]} < requested seq_len {seq_len}")
    ids = torch.from_numpy(np.ascontiguousarray(arr[0, :seq_len])).long().unsqueeze(0)
    return ids.to(device)


def _capture_layer_outputs(model, input_ids: torch.Tensor) -> list:
    """Register forward hooks on every decoder layer; return list of output
    hidden tensors in decoder-stack order (detached, cpu-float32 for compare)."""
    outs = []
    handles = []

    def mk_hook(idx):
        def _h(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            outs.append((idx, h.detach().to(torch.float32).cpu()))
        return _h

    for i, layer in enumerate(model.model.layers):
        handles.append(layer.register_forward_hook(mk_hook(i)))
    try:
        with torch.no_grad():
            model(input_ids, use_cache=False)
    finally:
        for h in handles:
            h.remove()
    outs.sort(key=lambda x: x[0])
    return [t for _, t in outs]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--data_path",  required=True)
    p.add_argument("--seq_len",    type=int, default=1024)
    p.add_argument("--num_slots",  type=int, default=512)
    p.add_argument("--top_k",      type=int, default=64)
    p.add_argument("--slot_init_noise", type=float, default=1.0,  # A.2 default
                   help="σ for slot init; A.2 used 1.0.")
    p.add_argument("--probe_layers", type=int, nargs="+", default=[0, 8, 16, 24, 31])
    p.add_argument("--threshold_tight", type=float, default=1e-4)
    p.add_argument("--threshold_loose", type=float, default=1e-3)
    p.add_argument("--json_out", default=None,
                   help="Optional path to dump decision JSON.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[probe] device={device} dtype={dtype}")
    print(f"[probe] loading vanilla Llama from {args.model_path}")

    # ---- (1) vanilla pass ------------------------------------------------
    model_v = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model_v.eval()
    for prm in model_v.parameters():
        prm.requires_grad_(False)

    input_ids = _load_chunk(args.data_path, args.seq_len, device)
    print(f"[probe] input_ids shape={tuple(input_ids.shape)}  "
          f"first tokens={input_ids[0, :8].tolist()}")

    print("[probe] vanilla forward (capturing per-layer outputs)...")
    vanilla_outs = _capture_layer_outputs(model_v, input_ids)
    print(f"[probe] got {len(vanilla_outs)} layer outputs; "
          f"shape[0]={tuple(vanilla_outs[0].shape)}")
    del model_v
    torch.cuda.empty_cache()

    # ---- (2) patched pass (A.2 config, untrained step-0) -----------------
    print(f"[probe] reloading model + applying MemorySpaceLayer wrappers "
          f"(num_slots={args.num_slots}, top_k={args.top_k}, σ={args.slot_init_noise})")
    model_p = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model_p.eval()

    cfg = MemorySpaceConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        slot_dim=None,
        selector_dim=64,
        slot_init="random",
        slot_init_noise=args.slot_init_noise,
        enable_writeback=True,
        writeback_gate_init=0.0,
        writeback_gate_warmup_steps=0,
        writeback_gate_max=0.3,
        shared_memory_bank=True,
    )
    model_p, mem_layers = apply_mem_space_to_model(model_p, cfg)
    # H7 fix v2 (2026-04-26 23:30): snapshot rotary inv_freq in fp32 BEFORE
    # the lossy `.to(dtype=bf16)` cast. The v1 approach (upcast after cast)
    # did NOT work: bf16 rounding of inv_freq is destructive, upcasting a
    # rounded tensor cannot recover mantissa bits. Direct evidence:
    #   inv_freq[1] = 0.81640625  (bf16-rounded)
    #   true fp32   = 0.81225...
    # At pos=1023, angle error ≈ 1023 × (0.81640625 - 0.81225) ≈ 4.25 rad
    # → cos drift up to ±2, matching the observed 1.578 absmax.
    # HF deliberately keeps inv_freq / original_inv_freq in fp32 (see
    # modeling_llama.LlamaRotaryEmbedding.__init__ and
    # modeling_rope_utils.dynamic_rope_update); blanket `.to(dtype=...)`
    # recurses into buffers and destroys that invariant.
    # Evidence: tests/test_wrapper_internal_parity.py H7 probe.
    _rope_snapshot = {}
    try:
        _rot = model_p.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass
    # Move newly-instantiated wrapper submodules (selector.Q_sel / Q_out,
    # slot_output_gate, shared MemoryBank, ...) onto the same device/dtype
    # as the underlying Llama body. apply_mem_space_to_model creates these
    # on CPU/float32 regardless of the host model's device.
    model_p = model_p.to(device=device, dtype=dtype)
    # Restore the rotary buffers to fp32 on the current device.
    try:
        _rot = model_p.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            # Re-register so PyTorch's buffer bookkeeping stays consistent.
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
    except AttributeError:
        pass
    for prm in model_p.parameters():
        prm.requires_grad_(False)

    # Invariant: every wrapper must have slot_output_gate == 0 at init,
    # which makes alpha = tanh(0) = 0 exactly in bf16.
    for i, ml in enumerate(mem_layers):
        g = ml.slot_output_gate.detach().abs().max().item()
        if g != 0.0:
            raise RuntimeError(
                f"Invariant violated: layer {i} slot_output_gate={g} != 0. "
                "Bypass-parity argument does not apply; fix init before probing."
            )
    print(f"[ok] all {len(mem_layers)} wrappers have slot_output_gate == 0")

    print("[probe] patched forward (capturing per-layer wrapper outputs)...")
    patched_outs = _capture_layer_outputs(model_p, input_ids)
    print(f"[probe] got {len(patched_outs)} layer outputs; "
          f"shape[0]={tuple(patched_outs[0].shape)}")

    # ---- (3) per-layer error comparison ----------------------------------
    n_layers = len(vanilla_outs)
    assert len(patched_outs) == n_layers, (len(vanilla_outs), len(patched_outs))

    per_layer = []
    for L in range(n_layers):
        d = (patched_outs[L] - vanilla_outs[L]).abs()
        per_layer.append({
            "L": L,
            "max_abs": float(d.max().item()),
            "mean_abs": float(d.mean().item()),
            "vanilla_max_abs": float(vanilla_outs[L].abs().max().item()),
        })

    print("\n=== §5.4 probe per-layer results (probe waypoints in bold) ===")
    print(f"{'L':>4} {'max_abs':>12} {'mean_abs':>12} {'|vanilla|':>12}")
    for row in per_layer:
        mark = " *" if row["L"] in args.probe_layers else "  "
        print(f"{row['L']:>4}{mark}{row['max_abs']:>11.3e}  "
              f"{row['mean_abs']:>11.3e}  {row['vanilla_max_abs']:>11.3e}")

    waypoint_errs = [r for r in per_layer if r["L"] in args.probe_layers]
    err_L0 = per_layer[0]["max_abs"]
    err_max_any = max(r["max_abs"] for r in per_layer)

    # ---- (4) decision ---------------------------------------------------
    if err_L0 > args.threshold_loose:
        decision = "H2_H5_hunt"
        reason = (
            f"err(L0)={err_L0:.3e} > {args.threshold_loose:.0e} → bypass parity "
            f"broken at step-0. Dispatch targeted unit test on dual "
            f"`wrapped_layer(...)` call at `layer.py:399-421`. NO training."
        )
    elif err_max_any < args.threshold_tight:
        decision = "H1_H3_experiment"
        reason = (
            f"max err across all layers = {err_max_any:.3e} < {args.threshold_tight:.0e} "
            f"→ bypass parity holds at step-0. Pollution must be post-step-1 "
            f"amplification. Dispatch Experiment A (σ=0.02 + warmup=500) on b200-1 "
            f"+ Experiment B (--no_shared_memory_bank, σ=1.0) on b200-2 in parallel."
        )
    else:
        decision = "ambiguous"
        reason = (
            f"err(L0)={err_L0:.3e}, err_max={err_max_any:.3e} lies in the "
            f"ambiguous band [{args.threshold_tight:.0e}, {args.threshold_loose:.0e}]. "
            f"Report layerwise trace to main before dispatching follow-ups."
        )

    print("\n=== §5.4 DECISION ===")
    print(f"decision: {decision}")
    print(f"reason:   {reason}")

    result = {
        "decision": decision,
        "reason": reason,
        "err_L0": err_L0,
        "err_max_any": err_max_any,
        "waypoints": waypoint_errs,
        "per_layer": per_layer,
        "config": {
            "seq_len": args.seq_len,
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "slot_init_noise": args.slot_init_noise,
        },
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[ok] dumped decision to {args.json_out}")


if __name__ == "__main__":
    main()
