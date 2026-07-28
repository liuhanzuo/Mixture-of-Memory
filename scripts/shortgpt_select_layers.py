#!/usr/bin/env python3
"""ShortGPT layer-selection for Paper B (Men et al., 2024; arXiv:2403.03853).

Purpose
-------
Provide an EXTERNAL pruning-policy baseline for the Paper B prune-then-heal
protocol. The main arm ("Ours", keep14) keeps the FRONT `keep` decoder layers and
re-grows a fresh tail; ShortGPT instead keeps the layers with the highest
*Block Influence* (BI) and drops the lowest-BI ones. Both end up at 16 layers,
so the comparison isolates the LAYER-SELECTION POLICY (contiguous front-tail
truncation vs. influence-based selection) rather than depth.

Block Influence (faithful to ShortGPT)
--------------------------------------
For decoder layer i with input hidden state h_in^(i) and output
h_out^(i) = h_in^(i) + f_i(h_in^(i)), we compute (averaged over calibration
tokens t and examples x):

  * cosine  (ShortGPT canonical, the paper's headline BI):
        BI_cos(i) = 1 - E_{x,t}[ cos( h_in^(i), h_out^(i) ) ]
    A layer that barely rotates the residual stream has cos ~ 1  ->  BI ~ 0
    (redundant). Higher BI = the layer changes the representation more = more
    important.

  * relative_l2 (residual-magnitude restatement, reported for cross-check):
        BI_rel(i) = E_{x,t}[ || h_out^(i) - h_in^(i) ||_2 / || h_in^(i) ||_2 ]
                  = E_{x,t}[ || f_i(h_in^(i)) ||_2 / || h_in^(i) ||_2 ]

Both metrics are recorded in the output JSON. Layer SELECTION uses `--bi_metric`
(default `cosine`, the ShortGPT paper's definition). We keep the `keep_layers`
highest-BI layers (sorted ascending by original index to preserve stacking
order) and drop the rest.

Calibration data
----------------
Windows sampled (evenly spaced, deterministic) from `--calib_data`
(default data/dolmino_now15b.npy, the OLMo-2-tokenised Dolmino heal corpus). We
NEVER calibrate on MMLU or any downstream task -- BI is a data-agnostic
redundancy signal and using the heal corpus keeps the selection independent of
the evaluation suite.

Hidden-state indexing (HF `output_hidden_states=True`)
------------------------------------------------------
`hidden_states` is a tuple of length num_layers+1:
  hidden_states[0]      = embedding output          (= input to layer 0)
  hidden_states[i]      = input to layer i          (= output of layer i-1)
  hidden_states[i+1]    = output of layer i
so for layer i in 0..L-1: h_in = hidden_states[i], h_out = hidden_states[i+1].

Compute
-------
Forward-only. Runs on CPU or a single GPU. No gradients, no training. Output:
outputs/shortgpt_layer_selection.json (read by scripts/train_olmo2_shortgpt.py).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from transformers import Olmo2Config, Olmo2ForCausalLM


def _log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def _select_device(arg: str):
    if arg == "cpu" or (arg == "auto" and not torch.cuda.is_available()):
        return torch.device("cpu"), False
    return torch.device("cuda"), True


@torch.no_grad()
def compute_block_influence(model, windows, device, use_cuda, batch_size,
                            num_layers, dtype):
    """Accumulate per-layer BI over `windows` ([n, seq_len] int tokens).

    Returns (bi_cosine, bi_relative_l2, n_tokens_scored) where bi_* are length-L
    python lists. Cosine and relative-L2 are BOTH accumulated in one pass in
    fp32 (cast from the model's forward dtype) for numerical stability."""
    sum_cos = torch.zeros(num_layers, dtype=torch.float64)
    sum_rel = torch.zeros(num_layers, dtype=torch.float64)
    n_tokens = 0
    n = windows.shape[0]
    for b in range(0, n, batch_size):
        chunk = windows[b:b + batch_size]
        input_ids = torch.from_numpy(chunk.astype(np.int64)).to(device)
        if use_cuda:
            with torch.amp.autocast("cuda", dtype=dtype):
                out = model(input_ids=input_ids, output_hidden_states=True,
                            use_cache=False)
        else:
            out = model(input_ids=input_ids, output_hidden_states=True,
                        use_cache=False)
        hs = out.hidden_states  # tuple len L+1
        assert len(hs) == num_layers + 1, (
            f"expected {num_layers + 1} hidden states, got {len(hs)}"
        )
        bt = input_ids.shape[0] * input_ids.shape[1]
        for i in range(num_layers):
            h_in = hs[i].reshape(-1, hs[i].shape[-1]).float()      # [B*T, D]
            h_out = hs[i + 1].reshape(-1, hs[i + 1].shape[-1]).float()
            # cosine similarity per token, then sum
            cos = F.cosine_similarity(h_in, h_out, dim=-1, eps=1e-8)  # [B*T]
            sum_cos[i] += float(cos.double().sum().item())
            # relative residual magnitude per token, then sum
            diff = (h_out - h_in).norm(dim=-1)                       # [B*T]
            denom = h_in.norm(dim=-1).clamp_min(1e-8)                # [B*T]
            rel = diff / denom
            sum_rel[i] += float(rel.double().sum().item())
        n_tokens += bt
        del out, hs
        _log(f"  processed windows {b}..{min(b + batch_size, n)}/{n} "
             f"(n_tokens={n_tokens})")
    bi_cos = [1.0 - (sum_cos[i].item() / n_tokens) for i in range(num_layers)]
    bi_rel = [sum_rel[i].item() / n_tokens for i in range(num_layers)]
    return bi_cos, bi_rel, n_tokens


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", type=str,
                   default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B",
                   help="pretrained OLMo-2 to score (BI computed on its 32 layers)")
    p.add_argument("--calib_data", type=str, default="data/dolmino_now15b.npy",
                   help="tokenised calibration windows (NEVER MMLU/downstream)")
    p.add_argument("--num_calib_windows", type=int, default=128,
                   help="how many 2048-token windows to average BI over (64-256)")
    p.add_argument("--calib_seq_len", type=int, default=0,
                   help=">0 truncates each window to this many tokens (speed); "
                        "0 = full window length")
    p.add_argument("--keep_layers", type=int, default=16,
                   help="number of highest-BI layers to KEEP (drop the rest)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--bi_metric", type=str, default="cosine",
                   choices=["cosine", "relative_l2"],
                   help="metric used to RANK/select layers (both are recorded)")
    p.add_argument("--device", type=str, default="auto",
                   choices=["auto", "cpu", "cuda"])
    p.add_argument("--dtype", type=str, default="auto",
                   choices=["auto", "bf16", "fp32"],
                   help="forward dtype; auto = bf16 on cuda, fp32 on cpu")
    p.add_argument("--seed", type=int, default=0,
                   help="deterministic window sampling seed (unused for the "
                        "evenly-spaced sampler; kept for reproducibility record)")
    p.add_argument("--output", type=str,
                   default="outputs/shortgpt_layer_selection.json")
    args = p.parse_args()

    device, use_cuda = _select_device(args.device)
    if args.dtype == "bf16":
        load_dtype, fwd_dtype = torch.bfloat16, torch.bfloat16
    elif args.dtype == "fp32":
        load_dtype, fwd_dtype = torch.float32, torch.bfloat16
    else:  # auto
        load_dtype = torch.bfloat16 if use_cuda else torch.float32
        fwd_dtype = torch.bfloat16
    _log(f"device={device} use_cuda={use_cuda} load_dtype={load_dtype} "
         f"fwd_autocast={fwd_dtype if use_cuda else 'none(cpu)'}")

    # ---- config / num layers ----
    cfg = Olmo2Config.from_pretrained(args.model_path, local_files_only=True)
    num_layers = cfg.num_hidden_layers
    _log(f"model={args.model_path} num_hidden_layers={num_layers} "
         f"keep_layers={args.keep_layers} bi_metric={args.bi_metric}")
    if args.keep_layers >= num_layers:
        raise ValueError(f"keep_layers={args.keep_layers} >= num_layers={num_layers}")

    # ---- calibration windows (evenly spaced, deterministic) ----
    arr = np.load(args.calib_data, mmap_mode="r")
    assert arr.ndim == 2, f"expected [N, seq_len], got {arr.shape}"
    n_total, win_len = arr.shape
    k = min(args.num_calib_windows, n_total)
    idx = np.linspace(0, n_total - 1, k).astype(np.int64)
    windows = np.array(arr[idx])  # materialise into RAM
    if args.calib_seq_len and 0 < args.calib_seq_len < win_len:
        windows = windows[:, : args.calib_seq_len]
    _log(f"calib={args.calib_data} full_shape={arr.shape} -> using {windows.shape} "
         f"({k} evenly-spaced windows, seq_len={windows.shape[1]})")

    # ---- load model ----
    t0 = time.time()
    model = Olmo2ForCausalLM.from_pretrained(
        args.model_path, torch_dtype=load_dtype, local_files_only=True)
    model.config.use_cache = False
    model = model.to(device)
    model.eval()
    _log(f"loaded base model in {time.time() - t0:.1f}s")

    # ---- BI ----
    t0 = time.time()
    bi_cos, bi_rel, n_tok = compute_block_influence(
        model, windows, device, use_cuda, args.batch_size, num_layers, fwd_dtype)
    _log(f"BI computed over {n_tok} tokens in {time.time() - t0:.1f}s")

    # ---- select highest-BI layers by chosen metric ----
    metric = bi_cos if args.bi_metric == "cosine" else bi_rel
    ranking_desc = sorted(range(num_layers), key=lambda i: metric[i], reverse=True)
    kept = sorted(ranking_desc[: args.keep_layers])
    dropped = sorted(ranking_desc[args.keep_layers:])

    # ---- report ----
    _log("per-layer BI (layer: cosine | relative_l2)  [* = KEPT]:")
    for i in range(num_layers):
        star = "*" if i in kept else " "
        _log(f"  L{i:02d}{star}  cos={bi_cos[i]:.5f}  rel={bi_rel[i]:.5f}")
    kept_metric = [round(metric[i], 6) for i in kept]
    dropped_metric = [round(metric[i], 6) for i in dropped]
    _log(f"KEEP {len(kept)} layers (highest {args.bi_metric} BI): {kept}")
    _log(f"DROP {len(dropped)} layers (lowest  {args.bi_metric} BI): {dropped}")
    _log(f"lowest-BI (first dropped): "
         f"{[(i, round(metric[i], 5)) for i in ranking_desc[-5:]]}")

    payload = {
        "method": "ShortGPT",
        "reference": "Men et al., 2024, arXiv:2403.03853",
        "model_path": args.model_path,
        "num_layers": num_layers,
        "keep_layers": args.keep_layers,
        "bi_metric": args.bi_metric,
        "calib_data": args.calib_data,
        "num_calib_windows": int(windows.shape[0]),
        "calib_seq_len": int(windows.shape[1]),
        "n_tokens_scored": int(n_tok),
        "device": str(device),
        "load_dtype": str(load_dtype),
        # full per-layer BI for both metrics (index = original layer id)
        "bi_cosine": [round(x, 8) for x in bi_cos],
        "bi_relative_l2": [round(x, 8) for x in bi_rel],
        # ranking (by the selection metric) and the resulting split
        "layer_ranking_by_bi_desc": ranking_desc,
        "kept_layer_indices": kept,
        "dropped_layer_indices": dropped,
        "kept_bi": {str(i): round(metric[i], 8) for i in kept},
        "dropped_bi": {str(i): round(metric[i], 8) for i in dropped},
        "kept_metric_values": kept_metric,
        "dropped_metric_values": dropped_metric,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    _log(f"wrote selection -> {args.output}")
    # concise machine-parsable final line
    print("KEEP_LAYER_INDICES=" + ",".join(str(i) for i in kept), flush=True)


if __name__ == "__main__":
    main()
