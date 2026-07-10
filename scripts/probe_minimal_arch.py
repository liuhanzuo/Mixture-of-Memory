#!/usr/bin/env python3
"""Layer-truncation probe for the "front-j + NTP head" minimal-architecture hypothesis.

Direction 4 (QCMEM_AUTONOMOUS_AGENDA.md §1): our layer-wise probing shows the
semantic state saturates by a mid layer j (j*~6-12). The MINIMAL-ARCH hypothesis
asks whether the middle layers are therefore REDUNDANT -- i.e. can a smaller
transformer of "front j layers (understand) + a few (k=1..2) tail layers that
only do next-token prediction (NTP)" match the full model?

This script runs the CHEAP, TRAINING-FREE probe (probe #1 of the design): take an
already-trained from-scratch Llama checkpoint and do "layer-skip" forward passes,
measuring next-token perplexity / accuracy on held-out text for three arms per
truncation point j:

  (a) FULL         : all L layers  ->  norm -> lm_head           (j-invariant baseline)
  (b) FRONT-j+TAILk: layers[0:j] then the last k layers          (the minimal-arch
                     layers[L-k:L] (SKIP the middle [j:L-k])       surrogate; k in --ntp_ks)
                     -> norm -> lm_head
  (c) FRONT-j direct: layers[0:j] -> norm -> lm_head             (logit-lens @ j;
                                                                   no NTP layer at all)

Interpretation
--------------
  * (b) ppl ~= (a) ppl  for small j  ->  the middle layers ARE largely redundant;
    "front-j + k NTP layers" preserves LM quality  ->  hypothesis SUPPORTED (worth
    the expensive from-scratch probe #2).
  * (b) ppl  >>  (a) ppl              ->  the skipped layers do real, non-removable
    work (progressive refinement, not redundancy)  ->  hypothesis FALSIFIED for the
    truncation route, consistent with Stages-of-Inference (2406.19384): mid-layer
    deletion is only "robust" relative to early/late, not free.
  * (c) tells you how much even a single dedicated NTP layer buys over reading the
    layer-j state raw (this is the logit-lens NLL, matches the sembott probe curve).

HONESTY / LIMITATION (project red-line #2) -- read before trusting the verdict
------------------------------------------------------------------------------
Truncating an ALREADY-TRAINED model is NOT the same as training a from-scratch
minimal architecture. The kept layers were trained to consume the *outputs of the
layers we skip*; feeding layer-j's state straight into the tail layers is off the
training manifold, so a large ppl jump is an UPPER bound on the damage -- a
from-scratch "front-j + k" net trained end-to-end could still learn to work. So:
  * a SMALL (b)-vs-(a) gap is strong positive evidence (redundancy survives even
    the off-manifold shortcut) and green-lights the expensive from-scratch probe;
  * a LARGE gap is only SUGGESTIVE against the hypothesis, not proof -- but taken
    with the Stages-of-Inference prior it is a reason NOT to burn GPU on probe #2.
This probe is a fast feasibility SIGNAL, not the final architectural claim.

A built-in sanity check verifies the manual layer-by-layer forward reproduces the
model's own logits exactly (identity), so any measured gap is real, not a
mask/rope/plumbing bug. Front-(L-k)+tail-k also equals FULL by construction.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from semantic_bottleneck_model import build_bottleneck_model  # noqa: E402


# ---------------------------------------------------------------------------
# checkpoint / model construction
# ---------------------------------------------------------------------------
def load_ckpt(path, device, dtype, n_layers_override=0):
    """Rebuild the exact from-scratch arch and load the raw state_dict."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bl = ck.get("bottleneck_layer", 6)
    bd = ck.get("bottleneck_dim", 0)
    size = ck.get("model_size", "1b")
    seq_len = ck.get("seq_len", 2048)
    model = build_bottleneck_model(bottleneck_layer=bl, bottleneck_dim=bd,
                                   seq_len=seq_len, dtype=dtype, size=size)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} "
              f"(bd={bd} size={size})", flush=True)
    model.to(device).eval()
    L = model.config.num_hidden_layers
    if n_layers_override and n_layers_override != L:
        print(f"  [warn] --n_layers={n_layers_override} != ckpt L={L}; using ckpt L", flush=True)
    return model, size, bd


def build_random_tiny(device, dtype):
    """Tiny random Llama for CPU smoke (validates truncation logic + identity)."""
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(vocab_size=512, hidden_size=128, intermediate_size=256,
                      num_hidden_layers=6, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=32, hidden_act="silu", max_position_embeddings=512,
                      rope_theta=500000.0, rms_norm_eps=1e-5, tie_word_embeddings=True,
                      attention_bias=False, attention_dropout=0.0)
    model = LlamaForCausalLM(cfg).to(dtype).to(device).eval()
    return model, "tiny", 0


# ---------------------------------------------------------------------------
# manual layer-skip forward
# ---------------------------------------------------------------------------
def _prep(model, ids):
    """Return (embed, position_embeddings, causal_mask, position_ids) matching
    LlamaModel.forward exactly, so a manual layer loop is bit-identical to model()."""
    from transformers.masking_utils import create_causal_mask
    base = model.model
    embeds = base.embed_tokens(ids)
    T = ids.shape[1]
    pos = torch.arange(T, device=ids.device).unsqueeze(0)
    pe = base.rotary_emb(embeds, pos)
    cm = create_causal_mask(config=model.config, inputs_embeds=embeds,
                            attention_mask=None, past_key_values=None, position_ids=pos)
    return embeds, pe, cm, pos


def _run_layers(model, h, layer_indices, pe, cm, pos):
    base = model.model
    for i in layer_indices:
        h = base.layers[i](h, attention_mask=cm, position_embeddings=pe,
                            position_ids=pos)
    return h


@torch.no_grad()
def _readout_metrics(model, h, ids):
    """norm -> lm_head -> next-token NLL(sum) + top1-correct(sum) + n_tok."""
    logits = model.lm_head(model.model.norm(h)).float()
    tgt = ids[:, 1:]
    logits = logits[:, :-1]
    nll = torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="sum")
    correct = (logits.argmax(-1) == tgt).sum()
    return float(nll), int(correct), int(tgt.numel())


@torch.no_grad()
def probe_batch(model, ids, js, ntp_ks, L):
    """One batch. Returns nested dict of running sums for every arm.

    Runs the full forward ONCE, snapshotting the hidden state after each front-j
    layers, then replays only the needed tail-k layers for each (j,k) arm.
    """
    embeds, pe, cm, pos = _prep(model, ids)
    js_set = set(js)
    snap = {}
    h = embeds
    for i in range(L):
        if i in js_set:
            snap[i] = h
        h = model.model.layers[i](h, attention_mask=cm, position_embeddings=pe,
                                  position_ids=pos)
    if L in js_set:
        snap[L] = h
    h_full = h  # output of all L layers (pre-final-norm)

    acc = {"full": _readout_metrics(model, h_full, ids), "front_direct": {}, "front_tail": {}}
    for j in js:
        hj = snap[j]
        # (c) front-j direct
        acc["front_direct"][j] = _readout_metrics(model, hj, ids)
        # (b) front-j + last-k layers (skip middle [j : L-k])
        for k in ntp_ks:
            tail_start = L - k
            if tail_start < j:
                # tail overlaps the front block -> would re-run kept layers; the
                # meaningful minimal-arch config needs j <= L-k. Mark as full-equiv.
                acc["front_tail"].setdefault(j, {})[k] = None
                continue
            ht = _run_layers(model, hj, range(tail_start, L), pe, cm, pos)
            acc["front_tail"].setdefault(j, {})[k] = _readout_metrics(model, ht, ids)
    return acc


@torch.no_grad()
def sanity_identity(model, ids, L):
    """Manual full forward must reproduce model(input_ids).logits (identity)."""
    embeds, pe, cm, pos = _prep(model, ids)
    h = _run_layers(model, embeds, range(L), pe, cm, pos)
    manual = model.lm_head(model.model.norm(h)).float()
    ref = model(input_ids=ids, use_cache=False).logits.float()
    return float((manual - ref).abs().max().item())


def _accumulate(dst, src):
    """Add (nll, correct, ntok) tuples into a running accumulator dict."""
    def add(a, b):
        return None if b is None else (a[0] + b[0], a[1] + b[1], a[2] + b[2])
    if dst is None:
        # deep copy of first batch structure
        return {
            "full": src["full"],
            "front_direct": dict(src["front_direct"]),
            "front_tail": {j: dict(kd) for j, kd in src["front_tail"].items()},
        }
    dst["full"] = add(dst["full"], src["full"])
    for j, v in src["front_direct"].items():
        dst["front_direct"][j] = add(dst["front_direct"][j], v)
    for j, kd in src["front_tail"].items():
        for k, v in kd.items():
            dst["front_tail"][j][k] = add(dst["front_tail"][j][k], v)
    return dst


def _finalize(t):
    if t is None:
        return None
    nll, correct, n = t
    mean_nll = nll / n
    return {"ppl": round(math.exp(min(mean_nll, 20)), 3),
            "nll": round(mean_nll, 4),
            "acc": round(correct / n, 4)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="", help="from-scratch state_dict .pt (train_semantic_bottleneck_1b)")
    ap.add_argument("--n_layers", type=int, default=0, help="override / assert L (0 = infer from ckpt)")
    ap.add_argument("--truncate_js", default="4 6 8 12", help="front-layer counts j (space/comma sep)")
    ap.add_argument("--ntp_ks", default="1 2", help="tail NTP layer counts k (space/comma sep)")
    ap.add_argument("--val_path", default="data/slimpajama_val_4096_llama3.npy")
    ap.add_argument("--n_examples", type=int, default=128)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "fp32"])
    ap.add_argument("--out_json", default="outputs/minimal_arch_probe.json")
    ap.add_argument("--smoke_random", action="store_true",
                    help="build a tiny random model instead of loading --ckpt (CPU logic smoke)")
    args = ap.parse_args()

    def _ints(s):
        return [int(x) for x in s.replace(",", " ").split() if x.strip()]
    js = sorted(set(_ints(args.truncate_js)))
    ntp_ks = sorted(set(_ints(args.ntp_ks)))
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    device = args.device

    if args.smoke_random:
        print("[smoke_random] tiny random Llama on", device, flush=True)
        model, size, bd = build_random_tiny(device, dtype)
    else:
        assert args.ckpt, "provide --ckpt (or --smoke_random)"
        print(f"loading ckpt {args.ckpt}", flush=True)
        model, size, bd = load_ckpt(args.ckpt, device, dtype, args.n_layers)
    L = model.config.num_hidden_layers
    js = [j for j in js if 0 <= j <= L]
    print(f"model size={size} bottleneck_dim={bd} L={L} | js={js} ntp_ks={ntp_ks} "
          f"seq_len={args.seq_len} n={args.n_examples}", flush=True)

    # data
    arr = None if args.smoke_random else (
        np.load(args.val_path, mmap_mode="r") if os.path.exists(args.val_path) else None)
    if arr is not None:
        n = min(args.n_examples, arr.shape[0])
        sl = min(args.seq_len, arr.shape[1])
        tokens = torch.from_numpy(np.asarray(arr[:n, :sl]).astype(np.int64))
    else:
        if not args.smoke_random:
            print(f"[warn] val_path {args.val_path} missing -> random ids (smoke only)", flush=True)
        V = model.config.vocab_size
        tokens = torch.randint(0, V, (args.n_examples, args.seq_len))
    print(f"data: {tokens.shape[0]} x {tokens.shape[1]}", flush=True)

    # sanity: manual forward == model()
    ids0 = tokens[: min(args.batch_size, tokens.shape[0])].to(device)
    id_diff = sanity_identity(model, ids0, L)
    print(f"[sanity] max|manual_full - model()| = {id_diff:.4g} "
          f"({'OK identity' if id_diff < 1e-1 else 'WARN plumbing mismatch'})", flush=True)

    # main loop
    accum = None
    for b0 in range(0, tokens.shape[0], args.batch_size):
        ids = tokens[b0:b0 + args.batch_size].to(device)
        accum = _accumulate(accum, probe_batch(model, ids, js, ntp_ks, L))
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    full = _finalize(accum["full"])
    front_direct = {j: _finalize(accum["front_direct"][j]) for j in js}
    front_tail = {j: {k: _finalize(accum["front_tail"][j][k]) for k in ntp_ks} for j in js}

    result = {
        "ckpt": args.ckpt or "smoke_random",
        "model_size": size, "bottleneck_dim": bd, "n_layers": L,
        "seq_len": int(tokens.shape[1]), "n_examples": int(tokens.shape[0]),
        "truncate_js": js, "ntp_ks": ntp_ks,
        "sanity_identity_max_abs_diff": round(id_diff, 6),
        "full_model": full,
        "front_j_direct": front_direct,       # (c) logit-lens @ j
        "front_j_plus_tail_k": front_tail,    # (b) minimal-arch surrogate
        "limitation": ("truncating a trained model != training a from-scratch minimal "
                       "arch; a large gap is an UPPER bound on damage (off-manifold), a "
                       "small gap is strong positive evidence. Fast feasibility signal only."),
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.out_json)), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)

    # ---- console report ----
    def _cell(m):
        return f"{m['ppl']:.2f} ({m['acc']:.3f})" if m else "= full/overlap"

    print("\n=== MINIMAL-ARCH LAYER-TRUNCATION PROBE ===")
    print(f"FULL (all {L} layers): ppl={full['ppl']} nll={full['nll']} acc={full['acc']}")
    col_titles = ["front-j direct"] + [f"front-j+tail{k}" for k in ntp_ks]
    hdr = f"{'j':>4} | " + " | ".join(t.rjust(20) for t in col_titles)
    print("\n" + hdr)
    print("-" * len(hdr))
    for j in js:
        cells = [_cell(front_direct[j])]
        cells += [_cell(front_tail[j][k]) for k in ntp_ks]
        print(f"{j:>4} | " + " | ".join(c.rjust(20) for c in cells))

    # verdict helper (smallest j whose front-j+tail-1 ppl within 10% of full)
    k1 = ntp_ks[0]
    print("\n=== VERDICT (front-j + tail-{}) ===".format(k1))
    base = full["ppl"]
    verdict_j = None
    for j in js:
        ft = front_tail[j].get(k1)
        if ft is None:
            continue
        ratio = ft["ppl"] / base if base > 0 else float("inf")
        tag = ""
        if ratio <= 1.10:
            tag = "  <= within 10% of full (REDUNDANT-supportive)"
            if verdict_j is None:
                verdict_j = j
        elif ratio >= 2.0:
            tag = "  >> 2x full (skipped layers do real work)"
        print(f"  j={j:>3}: front{j}+tail{k1} ppl={ft['ppl']:.2f}  ratio_to_full={ratio:.2f}x{tag}")
    if verdict_j is not None:
        print(f"  => smallest j with near-full quality (front{verdict_j}+tail{k1}, <=10%): j={verdict_j}\n"
              f"     hypothesis SUPPORTIVE at this scale -> consider from-scratch probe #2.")
    else:
        print("  => NO j reaches within 10% of full with a single tail layer.\n"
              "     hypothesis NOT supported by truncation (skipped mid layers matter).\n"
              "     NOTE: off-manifold upper bound; see 'limitation' in json.")
    print(f"\nsaved {args.out_json}")


if __name__ == "__main__":
    main()
