"""H2 follow-up — localize the *actual* kwarg that breaks bypass parity.

Context (post-compaction)
-------------------------
The initial H2 unit test `tests/test_bypass_call_dispatch.py` proved
`attention_mask=None` vs explicit 4-D causal mask produces
`max|A-B| = 1.562e-02` on one Llama-3-8B layer 0 in bf16. We landed the
fix at `src/memory/mem_space/layer.py:L439,L463` (pre-compute the 4-D
causal mask, pass it to the bypass `wrapped_layer(...)` call).

But the post-fix §5.4 re-probe returned **bit-identical** err(L0)=1.562e-02.
Inspecting HF 5.6.2 source:

  - `modeling_llama.LlamaModel.forward` → `create_causal_mask(...)`
  - `masking_utils.sdpa_mask` under `allow_is_causal_skip=True` for a
    plain causal prompt returns **None**.

⇒ vanilla `LlamaModel` passes `attention_mask=None` to each decoder layer
under SDPA. Our H2 fix therefore moved the bypass call *away* from
vanilla's dispatch path, not toward it. Yet the pre-fix call (also None)
did not match vanilla either. That means the real divergence is in some
**other** kwarg — not the attn mask.

This test captures the ground-truth kwargs HF vanilla actually passes to
layer 0 via a forward pre-hook, and then replays layer 0 under four
configurations to localize which kwarg is the culprit.

Config L (logged-vanilla):
    Record the exact args + kwargs HF's `LlamaModel` passes to layer 0
    during a normal `model_v(input_ids, use_cache=False)` forward, plus
    the layer's output tensor. This is the ground-truth reference.

Config A (pre-fix bypass form):
    `layer(hidden_states, attention_mask=None, position_ids=None,
           past_key_values=None, use_cache=False,
           position_embeddings=(cos, sin))`

Config B (post-fix bypass form):
    `layer(hidden_states, attention_mask=<4D causal>, position_ids=None,
           past_key_values=None, use_cache=False,
           position_embeddings=(cos, sin))`

Config X (kwarg-exact replay of L):
    Call layer 0 with *exactly* the kwargs logged in Config L. If this
    matches L bit-exact, the layer is a pure function of its kwargs and
    we know the capture is faithful; if it does not, either the hook
    semantics are wrong or there is global state (unlikely; RoPE is
    stateless at inference).

Expected outcomes
-----------------
- X vs L ≈ 0 (bit-exact replay): the test harness is sound.
- A vs L: first mismatch we find → that tells us which kwarg the wrapper
  is mis-setting. Print full kwarg diff.
- B vs L: should be ~1.56e-02 since we proved 4D-mask vs None diverge
  under SDPA in bf16.

Usage (single GPU, ~1-2 min):
    python tests/test_bypass_kwargs_diagnostic.py \
        --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
        --seq_len 1024
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))

from transformers import AutoModelForCausalLM


def _describe(v: Any) -> str:
    """Return a short string describing a kwarg value."""
    if v is None:
        return "None"
    if isinstance(v, torch.Tensor):
        return (
            f"Tensor(shape={tuple(v.shape)}, dtype={v.dtype}, "
            f"device={v.device}, contig={v.is_contiguous()})"
        )
    if isinstance(v, tuple):
        return "(" + ", ".join(_describe(x) for x in v) + ")"
    if isinstance(v, list):
        return "[" + ", ".join(_describe(x) for x in v) + "]"
    if isinstance(v, bool):
        return f"bool({v})"
    if isinstance(v, (int, float)):
        return f"{type(v).__name__}({v})"
    return f"{type(v).__name__}({repr(v)[:64]})"


def _kwarg_diff(a: Dict[str, Any], b: Dict[str, Any]) -> list:
    """Return a list of (key, a_desc, b_desc) where a[key] != b[key]
    (by type/shape/presence; we do NOT compare tensor values here)."""
    out = []
    keys = set(a.keys()) | set(b.keys())
    for k in sorted(keys):
        av, bv = a.get(k, "__MISSING__"), b.get(k, "__MISSING__")
        if av == "__MISSING__" or bv == "__MISSING__":
            out.append((k, _describe(av) if av != "__MISSING__" else "MISSING",
                        _describe(bv) if bv != "__MISSING__" else "MISSING"))
            continue
        # Compare cheaply — shape/dtype/None vs not-None matter most here.
        if isinstance(av, torch.Tensor) and isinstance(bv, torch.Tensor):
            if av.shape != bv.shape or av.dtype != bv.dtype:
                out.append((k, _describe(av), _describe(bv)))
        elif (av is None) != (bv is None):
            out.append((k, _describe(av), _describe(bv)))
        elif type(av) is not type(bv):
            out.append((k, _describe(av), _describe(bv)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--layer_idx", type=int, default=0)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[diag] device={device} dtype={dtype} layer={args.layer_idx}")
    print(f"[diag] loading Llama-3-8B from {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    layer = model.model.layers[args.layer_idx]
    T = args.seq_len

    # ---------------- Config L: capture vanilla's actual kwargs --------------
    captured: Dict[str, Any] = {}

    def pre_hook(module, args_, kwargs_):
        # Record **references** — these tensors are still live during the
        # forward; we'll snapshot after the full forward completes by
        # deep-copying before they go out of scope. Actually HF reuses the
        # same `attention_mask` object across layers so holding the ref is
        # fine; we detach/clone to be safe.
        captured["pos_args_len"] = len(args_)
        # positional: typically (hidden_states,) or nothing
        if len(args_) >= 1:
            captured["arg0_hidden"] = args_[0].detach().clone()
        snapshot = {}
        for k, v in kwargs_.items():
            if isinstance(v, torch.Tensor):
                snapshot[k] = v.detach().clone()
            elif isinstance(v, tuple) and all(
                isinstance(x, torch.Tensor) for x in v
            ):
                snapshot[k] = tuple(x.detach().clone() for x in v)
            else:
                snapshot[k] = v
        captured["kwargs"] = snapshot

    def fwd_hook(module, args_, kwargs_, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["output"] = h.detach().clone()

    h1 = layer.register_forward_pre_hook(pre_hook, with_kwargs=True)
    h2 = layer.register_forward_hook(fwd_hook, with_kwargs=True)

    # Seed & input
    torch.manual_seed(0)
    input_ids = torch.randint(
        0, model.config.vocab_size, (1, T), device=device
    )
    with torch.no_grad():
        _ = model(input_ids, use_cache=False)
    h1.remove()
    h2.remove()

    assert "output" in captured, "forward hook didn't fire"
    print(f"[diag] captured from vanilla forward on layer {args.layer_idx}:")
    print(f"       pos_args_len = {captured['pos_args_len']}")
    if "arg0_hidden" in captured:
        print(f"       arg0_hidden  = {_describe(captured['arg0_hidden'])}")
    for k, v in captured["kwargs"].items():
        print(f"       kwargs[{k!r}] = {_describe(v)}")
    print(f"       output       = {_describe(captured['output'])}")

    # Pull out the reference hidden_states + kwargs
    if "arg0_hidden" in captured:
        hidden_states = captured["arg0_hidden"]
    else:
        # HF may pass hidden via kwargs['hidden_states']
        hidden_states = captured["kwargs"]["hidden_states"]
    ref_kwargs = captured["kwargs"]
    out_L = captured["output"]

    # Grab vanilla's attention_mask and position_embeddings for downstream
    vanilla_attn_mask = ref_kwargs.get("attention_mask", None)
    vanilla_pos_emb = ref_kwargs.get("position_embeddings", None)
    assert vanilla_pos_emb is not None, "no position_embeddings in vanilla kwargs"

    # ---------------- Config X: replay with exact kwargs ---------------------
    # Strip hidden_states from the kwargs we pass, if present.
    replay_kwargs = {k: v for k, v in ref_kwargs.items() if k != "hidden_states"}
    with torch.no_grad():
        out_X = layer(hidden_states, **replay_kwargs)
        out_X = out_X[0] if isinstance(out_X, tuple) else out_X

    err_X = (out_X.float() - out_L.float()).abs()
    print(f"\n[X] exact-replay max|X-L| = {float(err_X.max()):.3e}  "
          f"mean = {float(err_X.mean()):.3e}")

    # ---------------- Config A: pre-fix bypass form --------------------------
    # Exactly what the wrapper called before the H2 fix: attention_mask=None,
    # position_ids=None, past_key_values=None, use_cache=False,
    # position_embeddings=(cos, sin). No other kwargs.
    with torch.no_grad():
        out_A = layer(
            hidden_states,
            attention_mask=None,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=vanilla_pos_emb,
        )
        out_A = out_A[0] if isinstance(out_A, tuple) else out_A
    err_A = (out_A.float() - out_L.float()).abs()
    print(f"[A] pre-fix  (None mask)  max|A-L| = {float(err_A.max()):.3e}  "
          f"mean = {float(err_A.mean()):.3e}")

    # ---------------- Config B: post-fix bypass form -------------------------
    neg_inf = torch.finfo(dtype).min
    causal = torch.triu(
        torch.full((T, T), neg_inf, dtype=dtype, device=device), diagonal=1
    ).view(1, 1, T, T).contiguous()
    with torch.no_grad():
        out_B = layer(
            hidden_states,
            attention_mask=causal,
            position_ids=None,
            past_key_values=None,
            use_cache=False,
            position_embeddings=vanilla_pos_emb,
        )
        out_B = out_B[0] if isinstance(out_B, tuple) else out_B
    err_B = (out_B.float() - out_L.float()).abs()
    print(f"[B] post-fix (4D mask)   max|B-L| = {float(err_B.max()):.3e}  "
          f"mean = {float(err_B.mean()):.3e}")

    # ---------------- Delta between A and B (the original H2 test) ----------
    err_AB = (out_A.float() - out_B.float()).abs()
    print(f"[A-B] mask-path dispatch max|A-B| = {float(err_AB.max()):.3e}")

    # ---------------- Kwarg diffs printed for A vs L -------------------------
    a_kwargs = {
        "attention_mask": None,
        "position_ids": None,
        "past_key_values": None,
        "use_cache": False,
        "position_embeddings": vanilla_pos_emb,
    }
    diff_A = _kwarg_diff(a_kwargs, ref_kwargs)
    print("\n=== Kwarg diff: Config A (pre-fix bypass) vs L (vanilla) ===")
    if not diff_A:
        print("  (no type/shape/presence differences)")
    for k, a_desc, b_desc in diff_A:
        print(f"  {k}: A={a_desc}  |  L={b_desc}")

    # ---------------- Verdict ------------------------------------------------
    print("\n=== DIAGNOSIS ===")
    if float(err_X.max()) > 1e-4:
        verdict = "CAPTURE_UNSOUND"
        reason = (
            f"Exact-replay max|X-L|={float(err_X.max()):.3e} > 1e-4. "
            "The forward hook did not capture all inputs; the diagnostic "
            "is invalid. Check HF layer signature / cache_position / "
            "global state."
        )
    elif float(err_A.max()) < 1e-4:
        verdict = "PRE_FIX_BYPASS_WAS_FINE"
        reason = (
            f"max|A-L|={float(err_A.max()):.3e} < 1e-4: the pre-fix "
            "bypass form (attention_mask=None) already matched vanilla "
            "bit-exactly. The §5.4 probe's err(L0)=1.56e-02 therefore "
            "does NOT come from the wrapper's bypass `wrapped_layer(...)` "
            "call. It comes from elsewhere in the wrapper forward — "
            "maybe `slot_output_gate==0` invariant violation, "
            "RoPE cache aliasing on ext_pos_emb, or global side-effect "
            "from the extended-seq forward. Re-read probe capture logic. "
            "IMPORTANT: revert the H2 fix (pass attention_mask=None) to "
            "realign the bypass call with vanilla's dispatch path."
        )
    elif float(err_B.max()) < 1e-4:
        verdict = "POST_FIX_IS_VANILLA"
        reason = (
            f"max|B-L|={float(err_B.max()):.3e} < 1e-4 but "
            f"max|A-L|={float(err_A.max()):.3e}: vanilla actually uses "
            "the explicit 4-D mask path, so the H2 fix is correct. The "
            "probe's residual err(L0) must come from elsewhere."
        )
    else:
        verdict = "BOTH_A_AND_B_DIVERGE"
        reason = (
            f"Neither A (max={float(err_A.max()):.3e}) nor B "
            f"(max={float(err_B.max()):.3e}) matches vanilla. Some "
            "OTHER kwarg (cache_position, position_ids, ...) is the "
            "real divergence. Inspect the kwarg diff above."
        )
    print(f"verdict: {verdict}")
    print(f"reason:  {reason}")

    result = {
        "verdict": verdict,
        "reason": reason,
        "err_X_max": float(err_X.max()),
        "err_A_max": float(err_A.max()),
        "err_B_max": float(err_B.max()),
        "err_AB_max": float(err_AB.max()),
        "err_X_mean": float(err_X.mean()),
        "err_A_mean": float(err_A.mean()),
        "err_B_mean": float(err_B.mean()),
        "kwarg_diff_A_vs_L": [
            {"key": k, "A": a_desc, "L": b_desc}
            for k, a_desc, b_desc in diff_A
        ],
        "vanilla_kwargs_summary": {
            k: _describe(v) for k, v in ref_kwargs.items()
        },
        "config": {
            "layer_idx": args.layer_idx,
            "seq_len": args.seq_len,
        },
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[ok] dumped verdict to {args.json_out}")


if __name__ == "__main__":
    main()
