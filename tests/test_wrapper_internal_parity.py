"""Localize where MemorySpaceLayer's forward diverges from vanilla.

Post-revert of the H2 "fix", the §5.4 probe STILL shows err(L0)=1.56e-02.
The kwargs-diagnostic (`tests/test_bypass_kwargs_diagnostic.py`) proved the
bypass `wrapped_layer(...)` call alone is bit-exact to vanilla when fed the
same kwargs HF passes internally. So the 1.56e-2 must come from OTHER steps
inside `MemorySpaceLayer.forward`. Candidates:

    (A) `wrapped_layer(...)` with the full wrapper's context produces a
        different bypass_h than a naked vanilla call — e.g., some CUDA
        state, RoPE cache mutation, etc. (unlikely given the diagnostic,
        but must be ruled out).

    (B) `next_hidden = bypass_h + alpha * slot_delta` produces a non-zero
        drift even with alpha==0 exact — due to bf16 rounding, dtype
        promotion, or non-finite slot_delta values.

    (C) The wrapper's preamble (selector, slot_to_hidden, cat, RoPE extend,
        ext_attn_mask build) sneakily mutates something shared with the
        inner layer (autograd state, RNG, buffers).

This test takes one wrapped layer, runs it on a random input, and breaks
down:

    bypass_h_internal  — bypass_h as computed inside the wrapper forward
                         (captured via monkey-patch of the wrapper's
                         self.wrapped_layer call)
    bypass_h_reference — bypass_h computed by calling wrapped_layer directly
                         OUTSIDE the wrapper, with the same kwargs HF
                         vanilla uses
    next_hidden        — wrapper's final output
    (bypass_h_internal + 0 * slot_delta) — what next_hidden SHOULD equal
                                            if alpha=0 is really a no-op

and reports pairwise max|·-·| between them. Breakdown identifies which
step introduces the drift.

Usage:
    python tests/test_wrapper_internal_parity.py \\
      --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \\
      --seq_len 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from transformers import AutoModelForCausalLM

from src.memory.mem_space import MemorySpaceConfig, apply_mem_space_to_model


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--num_slots", type=int, default=512)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--slot_init_noise", type=float, default=1.0)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[internal] device={device} dtype={dtype} layer={args.layer_idx}")
    print(f"[internal] loading Llama-3-8B")

    # --- load vanilla for ground-truth layer output -------------------------
    model_v = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model_v.eval()
    for prm in model_v.parameters():
        prm.requires_grad_(False)
    # H7 probe: report the rotary inv_freq buffer dtype (expected float32 on
    # HF modeling_llama; if we see bf16, that's the precision-loss culprit).
    try:
        _vif = model_v.model.rotary_emb.inv_freq
        print(f"[internal] VANILLA rotary_emb.inv_freq dtype={_vif.dtype} "
              f"device={_vif.device} shape={tuple(_vif.shape)} "
              f"absmax={_vif.abs().max().item():.3e}")
    except AttributeError:
        print("[internal] vanilla rotary_emb not accessible")

    T = args.seq_len
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, model_v.config.vocab_size, (1, T), device=device
    )

    # Capture vanilla's layer {L} input (hidden_states right before layer L)
    # and output (layer L's hidden output), via pre/post hooks.
    vanilla_capture = {}

    def pre_v(module, args_, kwargs_):
        vanilla_capture["hidden_in"] = (
            args_[0] if len(args_) else kwargs_["hidden_states"]
        ).detach().clone()
        snap = {}
        for k, v in kwargs_.items():
            if isinstance(v, torch.Tensor):
                snap[k] = v.detach().clone()
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                snap[k] = tuple(x.detach().clone() for x in v)
            else:
                snap[k] = v
        vanilla_capture["kwargs"] = snap

    def post_v(module, args_, kwargs_, output):
        h = output[0] if isinstance(output, tuple) else output
        vanilla_capture["hidden_out"] = h.detach().clone()

    v_layer = model_v.model.layers[args.layer_idx]
    h1 = v_layer.register_forward_pre_hook(pre_v, with_kwargs=True)
    h2 = v_layer.register_forward_hook(post_v, with_kwargs=True)
    with torch.no_grad():
        model_v(input_ids, use_cache=False)
    h1.remove()
    h2.remove()

    hidden_in_v  = vanilla_capture["hidden_in"]
    hidden_out_v = vanilla_capture["hidden_out"]
    vkwargs      = vanilla_capture["kwargs"]
    print(f"[internal] vanilla L{args.layer_idx} input  = "
          f"shape={tuple(hidden_in_v.shape)} dtype={hidden_in_v.dtype}")
    print(f"[internal] vanilla L{args.layer_idx} output = "
          f"shape={tuple(hidden_out_v.shape)} dtype={hidden_out_v.dtype}")

    del model_v
    torch.cuda.empty_cache()

    # --- load patched model + mem_space wrappers ----------------------------
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
    # H7 fix v2 (2026-04-26 23:30): snapshot fp32 inv_freq / original_inv_freq
    # BEFORE the destructive `.to(dtype=bf16)` cast; restore after. HF
    # deliberately keeps these rotary buffers in fp32; blanket bf16 cast
    # rounds them destructively (inv_freq[1] → 0.81640625 vs true 0.81225),
    # producing ~±2 cos drift at pos=1023.
    _rope_snapshot = {}
    try:
        _rot = model_p.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass
    model_p = model_p.to(device=device, dtype=dtype)
    try:
        _rot = model_p.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
    except AttributeError:
        pass
    for prm in model_p.parameters():
        prm.requires_grad_(False)

    # H7 probe: after the .to(dtype=bf16) cast, is rotary_emb.inv_freq
    # still float32? HF keeps this buffer in float32 even when weights are
    # bf16; blanket .to(dtype=...) recurses into buffers and can downcast it.
    try:
        _pif = model_p.model.rotary_emb.inv_freq
        print(f"[internal] PATCHED rotary_emb.inv_freq dtype={_pif.dtype} "
              f"device={_pif.device} shape={tuple(_pif.shape)} "
              f"absmax={_pif.abs().max().item():.3e}")
        _vif2 = model_v.model.rotary_emb.inv_freq if False else None  # model_v deleted
    except AttributeError:
        print("[internal] patched rotary_emb not accessible")

    # Verify gate invariant
    ml = mem_layers[args.layer_idx]
    g = ml.slot_output_gate.detach().abs().max().item()
    assert g == 0.0, f"slot_output_gate nonzero: {g}"

    # --- Attach an instrument to mem_layers[L] so we can grab bypass_h, ---
    # --- ext_h, slot_delta, alpha, next_hidden as they're computed in-situ
    # inside MemorySpaceLayer.forward.
    inner_trace = {}
    orig_wrapped_forward = ml.wrapped_layer.forward
    call_counter = {"n": 0}

    def wrapped_spy(*a, **kw):
        # This gets called twice: once for bypass, once for extended.
        out = orig_wrapped_forward(*a, **kw)
        h = out[0] if isinstance(out, tuple) else out
        if call_counter["n"] == 0:
            inner_trace["bypass_h_internal"] = h.detach().clone()
            inner_trace["bypass_kwargs"] = {
                k: (v.detach().clone() if isinstance(v, torch.Tensor) else
                    (tuple(x.detach().clone() for x in v)
                     if isinstance(v, tuple) and all(
                        isinstance(x, torch.Tensor) for x in v) else v))
                for k, v in kw.items()
            }
        else:
            inner_trace["ext_h_internal"] = h.detach().clone()
        call_counter["n"] += 1
        return out

    ml.wrapped_layer.forward = wrapped_spy

    # Capture final next_hidden as the wrapper output via forward hook
    outer_capture = {}

    def post_p(module, args_, kwargs_, output):
        h = output[0] if isinstance(output, tuple) else output
        outer_capture["next_hidden"] = h.detach().clone()

    def pre_p(module, args_, kwargs_):
        outer_capture["wrapper_hidden_in"] = (
            args_[0] if len(args_) else kwargs_["hidden_states"]
        ).detach().clone()
        snap = {}
        for k, v in kwargs_.items():
            if isinstance(v, torch.Tensor):
                snap[k] = v.detach().clone()
            elif isinstance(v, tuple) and all(isinstance(x, torch.Tensor) for x in v):
                snap[k] = tuple(x.detach().clone() for x in v)
            else:
                snap[k] = v
        outer_capture["wrapper_kwargs"] = snap

    h3 = ml.register_forward_pre_hook(pre_p, with_kwargs=True)
    h4 = ml.register_forward_hook(post_p, with_kwargs=True)

    with torch.no_grad():
        model_p(input_ids, use_cache=False)

    h3.remove()
    h4.remove()
    ml.wrapped_layer.forward = orig_wrapped_forward  # restore

    # --- Analyse --------------------------------------------------------
    print("\n=== WRAPPER INTERNAL TRACE ===")
    print(f"wrapped_layer was called {call_counter['n']} times (expect 2)")

    bypass_h_internal = inner_trace["bypass_h_internal"]
    ext_h_internal    = inner_trace["ext_h_internal"]
    next_hidden       = outer_capture["next_hidden"]
    wrapper_hidden_in = outer_capture["wrapper_hidden_in"]

    # 0. Input sanity
    err_in = (wrapper_hidden_in.float() - hidden_in_v.float()).abs().max().item()
    print(f"  [0] wrapper hidden_in vs vanilla hidden_in   max|diff| = {err_in:.3e}")

    # 1. bypass_h vs vanilla layer L out
    err_bypass = (bypass_h_internal.float() - hidden_out_v.float()).abs().max().item()
    print(f"  [1] bypass_h_internal vs vanilla hidden_out  max|diff| = {err_bypass:.3e}")

    # 2. next_hidden vs bypass_h  (this is what alpha=0 should produce)
    err_identity = (next_hidden.float() - bypass_h_internal.float()).abs().max().item()
    print(f"  [2] next_hidden vs bypass_h_internal         max|diff| = {err_identity:.3e}")

    # 3. next_hidden vs vanilla
    err_next_vs_v = (next_hidden.float() - hidden_out_v.float()).abs().max().item()
    print(f"  [3] next_hidden vs vanilla hidden_out        max|diff| = {err_next_vs_v:.3e}")

    # 4. kwargs the wrapper passed to bypass call vs what vanilla got
    print("\n=== KWARGS PARITY (wrapper's bypass call vs vanilla L_in kwargs) ===")
    bkw = inner_trace["bypass_kwargs"]
    def _summ(v):
        if v is None: return "None"
        if isinstance(v, torch.Tensor):
            return f"Tensor{tuple(v.shape)} {v.dtype}"
        if isinstance(v, tuple):
            return "(" + ",".join(_summ(x) for x in v) + ")"
        if isinstance(v, bool): return f"bool({v})"
        return f"{type(v).__name__}({repr(v)[:40]})"

    all_keys = set(bkw.keys()) | set(vkwargs.keys())
    for k in sorted(all_keys):
        b = bkw.get(k, "<MISSING>")
        v = vkwargs.get(k, "<MISSING>")
        same = "OK" if _summ(b) == _summ(v) else "DIFF"
        print(f"  [{same}] {k}:  wrapper={_summ(b)}  vanilla={_summ(v)}")

    # ---- VALUE-level comparison of position_embeddings --------------
    print("\n=== POSITION_EMBEDDINGS VALUE COMPARISON ===")
    if "position_embeddings" in bkw and "position_embeddings" in vkwargs:
        bc, bs = bkw["position_embeddings"]
        vc, vs = vkwargs["position_embeddings"]
        err_cos = (bc.float() - vc.float()).abs().max().item()
        err_sin = (bs.float() - vs.float()).abs().max().item()
        print(f"  cos: max|wrapper - vanilla| = {err_cos:.3e}")
        print(f"  sin: max|wrapper - vanilla| = {err_sin:.3e}")
    # Compare wrapper_kwargs (what outer model passed to wrapper) vs vkwargs
    print("\n=== OUTER-MODEL KWARGS COMPARISON (patched wrapper in vs vanilla in) ===")
    wkw = outer_capture.get("wrapper_kwargs", {})
    all_keys2 = set(wkw.keys()) | set(vkwargs.keys())
    for k in sorted(all_keys2):
        w = wkw.get(k, "<MISSING>")
        v = vkwargs.get(k, "<MISSING>")
        same = "OK" if _summ(w) == _summ(v) else "DIFF"
        print(f"  [{same}] {k}:  patched_outer={_summ(w)}  vanilla={_summ(v)}")
        # Value comparison for tensors
        if isinstance(w, torch.Tensor) and isinstance(v, torch.Tensor) and w.shape == v.shape:
            err = (w.float() - v.float()).abs().max().item()
            print(f"         value max|diff| = {err:.3e}")
        # Value comparison for tuples of tensors (e.g. position_embeddings)
        if (isinstance(w, tuple) and isinstance(v, tuple)
                and len(w) == len(v)
                and all(isinstance(x, torch.Tensor) for x in w)
                and all(isinstance(x, torch.Tensor) for x in v)):
            for i, (a, b) in enumerate(zip(w, v)):
                if a.shape == b.shape:
                    err = (a.float() - b.float()).abs().max().item()
                    print(f"         [{i}] value max|diff| = {err:.3e}  "
                          f"(ptr_match={a.data_ptr() == b.data_ptr()})")

    # Final sanity: is the wrapper's bypass_kwargs position_embeddings the SAME
    # tensor object as the wrapper's incoming position_embeddings? If yes,
    # preamble did not mutate (only re-referenced). If the VALUES differ but
    # the ptrs are the same, something in-between mutated the underlying
    # storage in-place.
    print("\n=== POS_EMB IDENTITY TRACE ===")
    if ("position_embeddings" in wkw and "position_embeddings" in bkw):
        (wc, ws), (bc, bs) = wkw["position_embeddings"], bkw["position_embeddings"]
        print(f"  outer-in cos vs bypass-cos  max|diff| = "
              f"{(wc.float() - bc.float()).abs().max().item():.3e}")
        print(f"  outer-in sin vs bypass-sin  max|diff| = "
              f"{(ws.float() - bs.float()).abs().max().item():.3e}")
        print(f"  outer-in cos dtype={wc.dtype} device={wc.device}  "
              f"bypass-in cos dtype={bc.dtype} device={bc.device}")
        print(f"  outer-in cos absmax={wc.abs().max().item():.3e}  "
              f"bypass-in cos absmax={bc.abs().max().item():.3e}")
        print(f"  vanilla-in cos absmax={vkwargs['position_embeddings'][0].abs().max().item():.3e}")

    # 5. If wrapper's kwargs match vanilla's, do an independent bit-exact
    #    replay: call wrapped_layer directly with vkwargs and compare.
    print("\n=== INDEPENDENT REPLAY ===")
    replay_kwargs = {k: v for k, v in vkwargs.items() if k != "hidden_states"}
    with torch.no_grad():
        replay_out = ml.wrapped_layer(hidden_in_v, **replay_kwargs)
        replay_h = replay_out[0] if isinstance(replay_out, tuple) else replay_out
    err_replay = (replay_h.float() - hidden_out_v.float()).abs().max().item()
    print(f"  wrapped_layer(hidden_in_v, **vkwargs)  vs  vanilla out  = {err_replay:.3e}")

    # 6. Verdict ----------------------------------------------------------
    print("\n=== VERDICT ===")
    if err_bypass < 1e-6 and err_identity < 1e-6:
        verdict = "PARITY_INTACT"
        reason = (
            "Both bypass_h matches vanilla and next_hidden == bypass_h. The "
            "1.56e-2 must come from hook-capture semantics of the probe, not "
            "the wrapper itself. Re-inspect `scripts/probe_branch3_bypass_parity.py`."
        )
    elif err_identity >= 1e-6 and err_bypass < 1e-6:
        verdict = "ADD_MUL_ROUNDING"
        reason = (
            f"bypass_h == vanilla (err={err_bypass:.2e}) but next_hidden != bypass_h "
            f"(err={err_identity:.2e}). The `next_hidden = bypass_h + alpha*slot_delta` "
            "combine is introducing drift even with alpha=0 exact. Likely dtype "
            "promotion on slot_output_gate Parameter."
        )
    elif err_bypass >= 1e-6:
        verdict = "BYPASS_H_CORRUPTED"
        reason = (
            f"bypass_h inside wrapper differs from vanilla (err={err_bypass:.2e}) "
            "despite the kwargs-diagnostic showing bit-exact parity with "
            "attention_mask=None. Some wrapper preamble step (selector, "
            "slot_to_hidden, RoPE extend) is mutating global state. Independent "
            f"replay error = {err_replay:.2e} — if replay is 0, wrapper preamble "
            "is the culprit."
        )
    else:
        verdict = "UNCLEAR"
        reason = "See numeric breakdown above."

    print(f"verdict: {verdict}")
    print(f"reason:  {reason}")

    result = {
        "verdict": verdict,
        "reason": reason,
        "err_input_parity": err_in,
        "err_bypass_vs_vanilla": err_bypass,
        "err_next_vs_bypass": err_identity,
        "err_next_vs_vanilla": err_next_vs_v,
        "err_independent_replay": err_replay,
        "config": {
            "layer_idx": args.layer_idx,
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
        print(f"[ok] dumped verdict to {args.json_out}")


if __name__ == "__main__":
    main()
