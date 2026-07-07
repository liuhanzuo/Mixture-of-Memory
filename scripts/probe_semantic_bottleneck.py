#!/usr/bin/env python3
"""Layer-wise probe: does the semantic bottleneck sharpen the understand/generate split?

Compares two 1B from-scratch checkpoints (baseline vs bottleneck) trained by
``train_semantic_bottleneck_1b.py``. Self-contained + offline: uses the held-out
slimpajama val chunks only, no external labelled datasets.

Two per-layer curves (logit-lens style, on natural text):
  * GENERATION  = next-token top-1 accuracy reading each layer's hidden through
    the shared final RMSNorm + lm_head. Where this rises = where the AR
    next-token distribution "forms". Hypothesis: for the bottleneck arm it forms
    LATER / more concentrated in the back half (layers > j).
  * PREDICTIVITY(NLL) = per-layer logit-lens cross-entropy (lower = better),
    same signal in nats.

Reported: per-layer curve for both arms, the layer where generation reaches
95%/99% of the top-layer accuracy (j*_gen), and the jump across the bottleneck
layer j (acc[j+1]-acc[j]) — a big post-j jump in the bottleneck arm but not the
baseline = the back half is doing relatively more of the "turn meaning into
next-token" work, i.e. a cleaner division of labour.

Honesty (red line #2): print the raw curves. If the bottleneck arm's generation
curve does NOT form later / the split does not sharpen, say so.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantic_bottleneck_model import build_bottleneck_model  # noqa: E402


def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bl = ck.get("bottleneck_layer", 6)
    bd = ck.get("bottleneck_dim", 0)
    seq_len = ck.get("seq_len", 2048)
    model = build_bottleneck_model(bottleneck_layer=bl, bottleneck_dim=bd, seq_len=seq_len,
                                   dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} (bd={bd})")
    model.to(device).eval()
    return model, bl, bd


@torch.no_grad()
def layerwise_curves(model, tokens, device, batch_size=4):
    """Return (gen_acc[L+1], nll[L+1]) via logit-lens on each hidden state."""
    root = model
    norm = root.model.norm
    lm_head = root.lm_head
    n_layers = root.config.num_hidden_layers
    acc_sum = torch.zeros(n_layers + 1, device=device)
    nll_sum = torch.zeros(n_layers + 1, device=device)
    tok_count = 0
    for i in range(0, tokens.shape[0], batch_size):
        ids = tokens[i:i + batch_size].to(device)
        out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
        hs = out.hidden_states  # tuple len L+1 (embeddings + each layer)
        # next-token targets: predict ids[:,1:] from hidden[:, :-1]
        tgt = ids[:, 1:]
        n_tok = tgt.numel()
        for li, h in enumerate(hs):
            logits = lm_head(norm(h[:, :-1]))  # [B,T-1,V]
            logits = logits.float()
            pred = logits.argmax(-1)
            acc_sum[li] += (pred == tgt).sum()
            nll = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
            nll_sum[li] += nll
        tok_count += n_tok
    return (acc_sum / tok_count).cpu().numpy(), (nll_sum / tok_count).cpu().numpy()


def jstar(acc, frac):
    top = acc[-1]
    thr = frac * top
    for li in range(len(acc)):
        if acc[li] >= thr:
            return li
    return len(acc) - 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--bottleneck_ckpt", required=True)
    ap.add_argument("--val_path", default="data/slimpajama_val_4096_llama3.npy")
    ap.add_argument("--n_examples", type=int, default=64)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--out_json", default="outputs/sembott_probe_result.json")
    args = ap.parse_args()

    device = "cuda:0"
    arr = np.load(args.val_path, mmap_mode="r")
    tokens = torch.from_numpy(np.asarray(arr[: args.n_examples, : args.seq_len]).astype(np.int64))
    print(f"probe on {tokens.shape[0]} examples x {tokens.shape[1]} tokens")

    result = {}
    for name, ckpt in [("baseline", args.baseline_ckpt), ("bottleneck", args.bottleneck_ckpt)]:
        print(f"\n=== {name}: {ckpt} ===")
        model, bl, bd = load_ckpt(ckpt, device)
        acc, nll = layerwise_curves(model, tokens, device, args.batch_size)
        js95, js99 = jstar(acc, 0.95), jstar(acc, 0.99)
        j = bl  # bottleneck layer index (layer output index in hidden_states is bl+1)
        # hidden_states index: 0=embed, k+1 = output of layer k. bottleneck sits AFTER layer bl.
        jump = float(acc[j + 2] - acc[j + 1]) if (j + 2) < len(acc) else float("nan")
        result[name] = {
            "bottleneck_layer": bl, "bottleneck_dim": bd,
            "gen_acc": [round(float(x), 4) for x in acc],
            "nll": [round(float(x), 4) for x in nll],
            "jstar95": js95, "jstar99": js99,
            "top_acc": round(float(acc[-1]), 4),
            "post_bottleneck_jump": round(jump, 4),
        }
        print(f"  top-layer next-tok acc = {acc[-1]:.4f}")
        print(f"  gen curve (per hidden_state 0..L): " +
              " ".join(f"{x:.3f}" for x in acc))
        print(f"  j*95={js95} j*99={js99}  post-bottleneck(L{j}->L{j+1}) acc jump={jump:.4f}")
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # Verdict summary
    b, n = result["baseline"], result["bottleneck"]
    print("\n=== VERDICT (feasibility) ===")
    print(f"  j*95 baseline={b['jstar95']}  bottleneck={n['jstar95']}  "
          f"(higher for bottleneck => generation forms later => cleaner split)")
    print(f"  top-acc baseline={b['top_acc']}  bottleneck={n['top_acc']}  "
          f"(bottleneck should not badly hurt LM if info survives)")


if __name__ == "__main__":
    main()
