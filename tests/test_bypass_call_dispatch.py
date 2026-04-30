"""H2 unit test — dual `wrapped_layer(...)` call dispatch mismatch.

Gated by §5.4 probe verdict (`err_L0 = 1.56e-02 ≫ 1e-3` → `H2_H5_hunt`).
See `ops/research_notes/20260426_branch3_A2_pollution_debug.md` §5.4 and
`status/TRAINER_ACTIVE.md` (2026-04-26 22:05 entry).

Hypothesis under test
---------------------
`MemorySpaceLayer.forward` at `src/memory/mem_space/layer.py:399-407` calls
the wrapped Llama decoder layer with `attention_mask=None`, expecting HF to
install its own causal path. The outer `LlamaModel` stack instead prepares
a 4-D additive causal mask once and passes it to every decoder layer. Under
bf16, the two paths dispatch to (possibly) different SDPA kernels and the
outputs differ at ≳0.2 % relative — exactly the err_L0 the static probe saw.

This test reproduces the mismatch in isolation on **one** decoder layer,
with zero memory-space state involved, to prove the call-site is the root
cause of the step-0 bypass-parity violation.

Pass criterion
--------------
If `max_abs_err < 1e-4` at 10-30 random seeds → H2 is NOT the issue;
escalate (likely a subtler kernel-dispatch bug or a bf16 accumulation order
sensitivity in the specific HF version).

If `max_abs_err > 1e-3` → H2 confirmed. The fix is to precompute the same
4-D additive causal mask that the outer stack uses and pass it to the
bypass `wrapped_layer(...)` call, so both branches dispatch SDPA through
the same code path.

Usage (single GPU, ~1-2 min wall):
    python tests/test_bypass_call_dispatch.py \
        --model_path /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
        --seq_len 1024 --n_seeds 5
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


def _build_additive_causal_mask(
    T: int, dtype: torch.dtype, device: torch.device, batch_size: int = 1
) -> torch.Tensor:
    """Build the 4-D additive causal mask HF's outer LlamaModel passes to
    each decoder layer under SDPA: shape [B, 1, T, T], 0 on lower triangle,
    ``torch.finfo(dtype).min`` on strict upper triangle."""
    mask = torch.zeros(T, T, dtype=dtype, device=device)
    neg_inf = torch.finfo(dtype).min
    mask = mask.masked_fill(
        torch.triu(torch.ones(T, T, dtype=torch.bool, device=device), diagonal=1),
        neg_inf,
    )
    return mask.unsqueeze(0).unsqueeze(0).expand(batch_size, 1, T, T).contiguous()


def _get_position_embeddings(model, input_ids: torch.Tensor):
    """Reach into the model's RoPE to produce (cos, sin) for `input_ids`'s
    length, matching what HF's LlamaModel.forward passes to every layer."""
    T = input_ids.shape[1]
    device = input_ids.device
    position_ids = torch.arange(T, device=device).unsqueeze(0)
    # HF LlamaModel keeps the RoPE module at model.model.rotary_emb.
    rotary = model.model.rotary_emb
    # Feed it a dummy hidden_states of the right dtype — its forward only
    # needs shape + position_ids for cos/sin.
    hidden0 = torch.zeros(
        1, T, model.config.hidden_size,
        dtype=next(model.parameters()).dtype, device=device,
    )
    cos, sin = rotary(hidden0, position_ids)
    return cos, sin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--seq_len", type=int, default=1024)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--layer_idx", type=int, default=0,
                   help="Decoder layer to probe (default 0).")
    p.add_argument("--threshold_tight", type=float, default=1e-4)
    p.add_argument("--threshold_loose", type=float, default=1e-3)
    p.add_argument("--json_out", default=None)
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    print(f"[h2-test] device={device} dtype={dtype} layer={args.layer_idx}")
    print(f"[h2-test] loading Llama-3-8B from {args.model_path}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, low_cpu_mem_usage=True,
    ).to(device)
    model.eval()
    for prm in model.parameters():
        prm.requires_grad_(False)

    layer = model.model.layers[args.layer_idx]
    d = model.config.hidden_size

    per_seed = []
    for seed in range(args.n_seeds):
        torch.manual_seed(seed)
        # Use real token-embedding activations, not pure randn, so the
        # statistics match a real forward. Cheap: just push a random id
        # sequence through the embedding.
        input_ids = torch.randint(
            0, model.config.vocab_size, (1, args.seq_len), device=device
        )
        with torch.no_grad():
            hidden_states = model.model.embed_tokens(input_ids)
            # Apply the outer stack's input_layernorm equivalent — no, the
            # decoder layer has its own input RMSNorm; feed the raw
            # embeddings as HF does.
            cos, sin = _get_position_embeddings(model, input_ids)

            # Call A: bypass-call form (attention_mask=None)
            out_A = layer(
                hidden_states,
                attention_mask=None,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            h_A = out_A[0] if isinstance(out_A, tuple) else out_A

            # Call B: outer-stack form (prepared 4-D additive causal mask)
            ext_attn_mask = _build_additive_causal_mask(
                T=args.seq_len, dtype=dtype, device=device, batch_size=1,
            )
            out_B = layer(
                hidden_states,
                attention_mask=ext_attn_mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                position_embeddings=(cos, sin),
            )
            h_B = out_B[0] if isinstance(out_B, tuple) else out_B

            diff = (h_A.float() - h_B.float()).abs()
            per_seed.append({
                "seed": seed,
                "max_abs": float(diff.max().item()),
                "mean_abs": float(diff.mean().item()),
                "vanilla_max_abs_A": float(h_A.float().abs().max().item()),
            })

    print(f"\n=== H2 dual-call dispatch diff (layer {args.layer_idx}, bf16) ===")
    print(f"{'seed':>4} {'max_abs(A-B)':>14} {'mean_abs':>12} {'|h_A|_max':>12}")
    for row in per_seed:
        print(f"{row['seed']:>4} {row['max_abs']:>14.3e} "
              f"{row['mean_abs']:>12.3e} {row['vanilla_max_abs_A']:>12.3e}")

    max_any = max(r["max_abs"] for r in per_seed)
    mean_of_means = sum(r["mean_abs"] for r in per_seed) / len(per_seed)

    if max_any > args.threshold_loose:
        verdict = "H2_CONFIRMED"
        reason = (
            f"max|A-B| = {max_any:.3e} > {args.threshold_loose:.0e} across "
            f"{args.n_seeds} seeds. The `attention_mask=None` (bypass) and "
            f"prepared 4-D causal mask (outer-stack) dispatches produce "
            f"numerically distinct outputs in bf16. Fix: pass the prepared "
            f"mask to the bypass `wrapped_layer(...)` call at "
            f"`src/memory/mem_space/layer.py:399-407`."
        )
    elif max_any < args.threshold_tight:
        verdict = "H2_FALSIFIED"
        reason = (
            f"max|A-B| = {max_any:.3e} < {args.threshold_tight:.0e}. "
            f"Mask-prep dispatch is NOT the root cause. Escalate: look for "
            f"RoPE cos/sin cache aliasing, or bf16 accumulation order "
            f"sensitivity introduced elsewhere in the wrapper."
        )
    else:
        verdict = "AMBIGUOUS"
        reason = (
            f"max|A-B| = {max_any:.3e} lies in the ambiguous band "
            f"[{args.threshold_tight:.0e}, {args.threshold_loose:.0e}]. "
            f"Increase `--n_seeds` and/or `--seq_len` and re-check."
        )

    print("\n=== VERDICT ===")
    print(f"verdict: {verdict}")
    print(f"reason:  {reason}")
    print(f"max_any: {max_any:.3e} · mean_of_means: {mean_of_means:.3e}")

    result = {
        "verdict": verdict,
        "reason": reason,
        "max_any": max_any,
        "mean_of_means": mean_of_means,
        "per_seed": per_seed,
        "config": {
            "layer_idx": args.layer_idx,
            "seq_len": args.seq_len,
            "n_seeds": args.n_seeds,
            "threshold_tight": args.threshold_tight,
            "threshold_loose": args.threshold_loose,
        },
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.json_out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"[ok] dumped verdict to {args.json_out}")


if __name__ == "__main__":
    main()
