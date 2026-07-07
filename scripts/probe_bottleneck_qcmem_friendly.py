#!/usr/bin/env python3
"""Is the semantic-bottleneck model more QCMem-friendly?  (compressibility + readout-under-compression)

Hypothesis (2026-07-07): QCMem caches a mid-layer hidden ``h_j`` and, on read,
recomputes ``layers[j:]`` from it. If we PRE-TRAIN a model with a HARD low-rank
bottleneck (funnel ``d->d_bottle->d``, no residual) on the OUTPUT of layer ``j``,
then the tensor QCMem would cache is *forced* into a rank-<=d_bottle subspace.
So the bottleneck arm's cache point should be (a) far more compressible, and
(b) far more robust when the cached ``h_j`` is compressed (rank truncation / int4)
before the ``layers[j:]`` readout. The baseline arm (no funnel) should not enjoy
either advantage. This directly answers the earlier probe that found early/mid
hidden states are NOT compressible in a vanilla model.

============================ CACHE POINT (important) ============================
The bottleneck sits on the *output* of decoder layer ``j`` (= input to layer j+1).
We define the QCMem cache tensor ``h_j`` = **post-funnel output of layer j**, and
the readout = recompute ``layers[j+1:] -> norm -> lm_head`` from it. For the
bottleneck arm this is the low-rank funnel output; for the baseline arm it is the
plain layer-``j`` output. Same cache depth + same #recomputed layers => fair.

  !! transformers>=5 ``output_hidden_states`` captures the INNER LlamaDecoderLayer
     output, which for the wrapped bottleneck layer is the PRE-funnel tensor (off
     by ~30 from the true post-funnel tensor that actually feeds layer j+1). So we
     MUST grab the cache tensor via a forward hook on ``model.model.layers[j]``,
     NOT via ``hidden_states``. Verified: hook-tensor -> recompute layers[j+1:]
     reproduces the full-forward logits exactly (max|Δlogit|=0.0). ================

Metrics (both arms, at cache layer j; optional layer-wise sweep):
  1. COMPRESSIBILITY of the cache tensor:
       - PCA cumulative-variance dims to reach 90/95/99%
       - effective rank (spectral entropy) + participation ratio
       - int4 per-channel absmax reconstruction rel-error
  2. READOUT under compression (the QCMem operation):
       compress h_j -> recompute layers[j+1:] -> next-token top-1 acc / NLL.
       Compressions: none(faithful), PCA rank-{512,256,128,64}, int4.
       Report DROP vs each model's OWN faithful readout (models differ in raw acc,
       so the fair metric is how much acc is LOST to compression).

Honesty (red line #2): these are weak 1B/2000-step from-scratch models; read the
RELATIVE trend (bottleneck vs baseline), not absolute numbers. If the bottleneck
arm is NOT more compressible / NOT more robust, the script says the hypothesis is
NOT supported.
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


# --------------------------------------------------------------------------- #
# model / cache-tensor plumbing
# --------------------------------------------------------------------------- #
def load_ckpt(path, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bl = ck.get("bottleneck_layer", 6)
    bd = ck.get("bottleneck_dim", 0)
    seq_len = ck.get("seq_len", 2048)
    model = build_bottleneck_model(bottleneck_layer=bl, bottleneck_dim=bd,
                                   seq_len=seq_len, dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} (bd={bd})")
    model.to(device).eval()
    return model, bl, bd


@torch.no_grad()
def collect_cache_hidden(model, tokens, j, device, batch_size=8):
    """Return the TRUE post-(bottleneck-)layer-``j`` output tensor (the QCMem cache
    point) for every example, plus the full-forward next-token acc/nll reference.

    Grabbed via a forward hook on model.model.layers[j] so the wrapped bottleneck
    arm yields its POST-funnel output (not the pre-funnel hidden_states tensor).
    Returns:
      H_cache : [N, T, d] bf16 on CPU (cache tensor per example)
      full_acc, full_nll : floats (full-forward logit reference at this cut)
    """
    mdl = model.model
    grabbed = {}

    def hook(mod, inp, out):
        grabbed["h"] = (out[0] if isinstance(out, tuple) else out).detach()

    handle = mdl.layers[j].register_forward_hook(hook)
    H_chunks = []
    acc_sum = 0.0
    nll_sum = 0.0
    tok_count = 0
    try:
        for i in range(0, tokens.shape[0], batch_size):
            ids = tokens[i:i + batch_size].to(device)
            out = model(input_ids=ids, use_cache=False)
            H_chunks.append(grabbed["h"].to("cpu"))
            logits = out.logits[:, :-1].float()
            tgt = ids[:, 1:]
            pred = logits.argmax(-1)
            acc_sum += (pred == tgt).sum().item()
            nll_sum += torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
                reduction="sum").item()
            tok_count += tgt.numel()
    finally:
        handle.remove()
    H_cache = torch.cat(H_chunks, dim=0)
    return H_cache, acc_sum / tok_count, nll_sum / tok_count


@torch.no_grad()
def readout_from_cache(model, H_cache, tokens, j, device, batch_size=8):
    """Recompute layers[j+1:] -> norm -> lm_head from a (possibly compressed)
    cache tensor H_cache [N,T,d]; return next-token top-1 acc + NLL."""
    mdl = model.model
    acc_sum = 0.0
    nll_sum = 0.0
    tok_count = 0
    N = H_cache.shape[0]
    for i in range(0, N, batch_size):
        h = H_cache[i:i + batch_size].to(device, dtype=torch.bfloat16)
        ids = tokens[i:i + batch_size].to(device)
        B, T, d = h.shape
        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        pe = mdl.rotary_emb(h, pos)
        for layer in mdl.layers[j + 1:]:
            h = layer(h, attention_mask=None, position_embeddings=pe,
                      position_ids=pos, past_key_values=None, use_cache=False)
            if isinstance(h, tuple):
                h = h[0]
        logits = model.lm_head(mdl.norm(h))[:, :-1].float()
        tgt = ids[:, 1:]
        acc_sum += (logits.argmax(-1) == tgt).sum().item()
        nll_sum += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), tgt.reshape(-1),
            reduction="sum").item()
        tok_count += tgt.numel()
    return acc_sum / tok_count, nll_sum / tok_count


# --------------------------------------------------------------------------- #
# compressibility metrics
# --------------------------------------------------------------------------- #
def pca_stats(H_flat_f32):
    """H_flat: [N, d] float32 (CPU/GPU). Return dict of PCA/effective-rank stats
    and (mean, eigvecs sorted desc) for later rank-r projection."""
    N, d = H_flat_f32.shape
    mean = H_flat_f32.mean(0, keepdim=True)
    Xc = H_flat_f32 - mean
    # covariance d x d (cheap: d=2048), symmetric eigendecomposition
    cov = (Xc.t() @ Xc) / max(N - 1, 1)
    evals, evecs = torch.linalg.eigh(cov)          # ascending
    evals = torch.clamp(evals.flip(0), min=0.0)    # descending
    evecs = evecs.flip(1)                          # match, columns = PCs desc
    total = evals.sum().clamp_min(1e-12)
    cum = torch.cumsum(evals, 0) / total
    dims = {}
    for thr in (0.90, 0.95, 0.99):
        idx = int(torch.searchsorted(cum, torch.tensor(thr, device=cum.device)).item()) + 1
        dims[f"dim_{int(thr*100)}"] = min(idx, d)
    p = evals / total
    p_nz = p[p > 0]
    spectral_entropy = float(-(p_nz * torch.log(p_nz)).sum().item())
    erank = float(np.exp(spectral_entropy))
    participation_ratio = float((evals.sum() ** 2 / (evals ** 2).sum().clamp_min(1e-12)).item())
    return {
        "n_vectors": int(N), "d": int(d),
        **dims,
        "effective_rank_spectral_entropy": round(erank, 2),
        "participation_ratio": round(participation_ratio, 2),
        "top1_var_frac": round(float((evals[0] / total).item()), 4),
    }, mean, evecs


def int4_quant_absmax_perchannel(H):
    """Symmetric per-channel (per-feature-dim) int4 (qmax=7) quantize->dequantize.
    H: [..., d] float32. Returns dequantized tensor same shape + rel Frob error."""
    absmax = H.abs().amax(dim=tuple(range(H.ndim - 1)), keepdim=True).clamp_min(1e-8)
    scale = absmax / 7.0
    q = torch.clamp(torch.round(H / scale), -8, 7)
    Hq = q * scale
    rel_err = float((H - Hq).norm() / H.norm().clamp_min(1e-12))
    return Hq, rel_err


def rank_r_project(H_cache_bf16, mean, evecs, r):
    """Project cache tensor onto top-r principal components (fit on same data).
    H_cache: [N,T,d] bf16 CPU. mean:[1,d], evecs:[d,d] desc (GPU/CPU float32).
    Returns reconstructed [N,T,d] bf16 CPU."""
    dev = evecs.device
    Vr = evecs[:, :r]                       # [d, r]
    N, T, d = H_cache_bf16.shape
    X = H_cache_bf16.reshape(-1, d).float().to(dev)
    mean_c = mean.to(dev)
    Xc = X - mean_c
    Xr = (Xc @ Vr) @ Vr.t() + mean_c        # reconstruct
    return Xr.reshape(N, T, d).to(torch.bfloat16).cpu()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def run_arm(name, ckpt, tokens, j, device, batch_size, ranks):
    print(f"\n================= {name}: {ckpt}  (cache layer j={j}) =================")
    model, bl, bd = load_ckpt(ckpt, device)

    H_cache, full_acc, full_nll = collect_cache_hidden(model, tokens, j, device, batch_size)
    print(f"  cache tensor shape={tuple(H_cache.shape)}  "
          f"full-forward next-tok acc={full_acc:.4f} nll={full_nll:.4f}")

    # ---- compressibility of the cache tensor ----
    d = H_cache.shape[-1]
    H_flat = H_cache.reshape(-1, d).float().to(device)
    comp, mean, evecs = pca_stats(H_flat)
    _, int4_recon_err = int4_quant_absmax_perchannel(H_flat)
    comp["int4_recon_relerr"] = round(int4_recon_err, 4)
    del H_flat
    torch.cuda.empty_cache()
    print(f"  COMPRESSIBILITY: dim90={comp['dim_90']} dim95={comp['dim_95']} "
          f"dim99={comp['dim_99']} / {d}  eff_rank={comp['effective_rank_spectral_entropy']} "
          f"part_ratio={comp['participation_ratio']}  int4_relerr={comp['int4_recon_relerr']}")

    # ---- readout under compression ----
    readout = {}
    # faithful (sanity: should ~= full_acc)
    f_acc, f_nll = readout_from_cache(model, H_cache, tokens, j, device, batch_size)
    readout["none"] = {"acc": round(f_acc, 4), "nll": round(f_nll, 4), "acc_drop": 0.0}
    print(f"  READOUT none(faithful): acc={f_acc:.4f} nll={f_nll:.4f} "
          f"(sanity vs full={full_acc:.4f}, Δ={f_acc-full_acc:+.4f})")

    ref_acc = f_acc
    for r in ranks:
        if r >= d:
            continue
        Hr = rank_r_project(H_cache, mean, evecs, r)
        a, nl = readout_from_cache(model, Hr, tokens, j, device, batch_size)
        readout[f"pca_rank{r}"] = {"acc": round(a, 4), "nll": round(nl, 4),
                                   "acc_drop": round(ref_acc - a, 4)}
        print(f"  READOUT pca_rank{r:4d}: acc={a:.4f} nll={nl:.4f}  acc_drop={ref_acc-a:+.4f}")
        del Hr
        torch.cuda.empty_cache()

    # int4 readout
    Hq, _ = int4_quant_absmax_perchannel(H_cache.float())
    Hq = Hq.to(torch.bfloat16)
    a, nl = readout_from_cache(model, Hq, tokens, j, device, batch_size)
    readout["int4"] = {"acc": round(a, 4), "nll": round(nl, 4),
                       "acc_drop": round(ref_acc - a, 4)}
    print(f"  READOUT int4       : acc={a:.4f} nll={nl:.4f}  acc_drop={ref_acc-a:+.4f}")
    del Hq

    del model, H_cache, evecs
    torch.cuda.empty_cache()
    return {"bottleneck_layer": bl, "bottleneck_dim": bd,
            "full_forward": {"acc": round(full_acc, 4), "nll": round(full_nll, 4)},
            "compressibility": comp, "readout": readout}


@torch.no_grad()
def layerwise_compressibility(name, ckpt, tokens, js, device, batch_size):
    """Compressibility (eff-rank / dim99 / int4 err) of cache tensor at several j,
    to check the bottleneck is special only at its own layer."""
    print(f"\n----- layer-wise compressibility {name} -----")
    model, bl, bd = load_ckpt(ckpt, device)
    rows = {}
    for j in js:
        H, _, _ = collect_cache_hidden(model, tokens, j, device, batch_size)
        d = H.shape[-1]
        Hf = H.reshape(-1, d).float().to(device)
        comp, _, _ = pca_stats(Hf)
        _, e = int4_quant_absmax_perchannel(Hf)
        rows[j] = {"dim99": comp["dim_99"], "eff_rank": comp["effective_rank_spectral_entropy"],
                   "int4_relerr": round(e, 4)}
        print(f"    j={j:2d}: dim99={comp['dim_99']:4d}/{d} eff_rank={comp['effective_rank_spectral_entropy']:7.2f} "
              f"int4_relerr={e:.4f}")
        del Hf, H
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_ckpt", default="outputs/sembott_1b_baseline/final.pt")
    ap.add_argument("--bottleneck_ckpt", default="outputs/sembott_1b_bottleneck/final.pt")
    ap.add_argument("--val_path", default="data/slimpajama_val_4096_llama3.npy")
    ap.add_argument("--cache_layer", type=int, default=6, help="j: cache OUTPUT of layer j")
    ap.add_argument("--n_examples", type=int, default=96)
    ap.add_argument("--seq_len", type=int, default=512)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="+", default=[512, 256, 128, 64])
    ap.add_argument("--layerwise", action="store_true", default=True)
    ap.add_argument("--layerwise_js", type=int, nargs="+", default=[2, 4, 6, 8, 10, 12])
    ap.add_argument("--out_json", default="outputs/sembott_qcmem_friendly.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = args.device
    arr = np.load(args.val_path, mmap_mode="r")
    tokens = torch.from_numpy(
        np.asarray(arr[: args.n_examples, : args.seq_len]).astype(np.int64))
    print(f"probe on {tokens.shape[0]} examples x {tokens.shape[1]} tokens "
          f"= {tokens.numel()} token vectors; cache layer j={args.cache_layer}")

    result = {"config": {
        "cache_layer": args.cache_layer, "n_examples": args.n_examples,
        "seq_len": args.seq_len, "ranks": args.ranks, "val_path": args.val_path}}

    result["baseline"] = run_arm("baseline", args.baseline_ckpt, tokens,
                                 args.cache_layer, device, args.batch_size, args.ranks)
    result["bottleneck"] = run_arm("bottleneck", args.bottleneck_ckpt, tokens,
                                   args.cache_layer, device, args.batch_size, args.ranks)

    if args.layerwise:
        result["layerwise_baseline"] = layerwise_compressibility(
            "baseline", args.baseline_ckpt, tokens, args.layerwise_js, device, args.batch_size)
        result["layerwise_bottleneck"] = layerwise_compressibility(
            "bottleneck", args.bottleneck_ckpt, tokens, args.layerwise_js, device, args.batch_size)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # ---------------------- VERDICT ----------------------
    b, n = result["baseline"], result["bottleneck"]
    bc, nc = b["compressibility"], n["compressibility"]
    print("\n==================== VERDICT (QCMem-friendliness) ====================")
    print(f"cache layer j={args.cache_layer}  (baseline bd={b['bottleneck_dim']}, "
          f"bottleneck bd={n['bottleneck_dim']})")
    print("\n[1] COMPRESSIBILITY of cache tensor  (lower dim / eff_rank / int4_err = more compressible)")
    print(f"    {'metric':<22}{'baseline':>12}{'bottleneck':>12}")
    for key, label in [("dim_90", "PCA dim@90%"), ("dim_95", "PCA dim@95%"),
                       ("dim_99", "PCA dim@99%"),
                       ("effective_rank_spectral_entropy", "eff_rank"),
                       ("participation_ratio", "part_ratio"),
                       ("int4_recon_relerr", "int4 recon relerr")]:
        print(f"    {label:<22}{bc[key]:>12}{nc[key]:>12}")

    print("\n[2] READOUT acc DROP under compression  (smaller drop = more QCMem-robust)")
    print(f"    {'compression':<16}{'baseline drop':>16}{'bottleneck drop':>18}")
    for kkey in ["none"] + [f"pca_rank{r}" for r in args.ranks] + ["int4"]:
        if kkey in b["readout"] and kkey in n["readout"]:
            bd_ = b["readout"][kkey]["acc_drop"]
            nd_ = n["readout"][kkey]["acc_drop"]
            print(f"    {kkey:<16}{bd_:>16}{nd_:>18}")

    # heuristic verdict
    more_compressible = (nc["dim_99"] < bc["dim_99"] and
                         nc["effective_rank_spectral_entropy"] < bc["effective_rank_spectral_entropy"])
    # robustness at the funnel rank (bottleneck_dim) and below
    robust_keys = [f"pca_rank{r}" for r in args.ranks
                   if f"pca_rank{r}" in b["readout"]]
    more_robust = all(n["readout"][k]["acc_drop"] <= b["readout"][k]["acc_drop"] + 1e-9
                      for k in robust_keys) if robust_keys else False
    print("\n[VERDICT]")
    print(f"  bottleneck MORE compressible at cache point?  {more_compressible}")
    print(f"  bottleneck MORE robust to compression (readout)?  {more_robust}")
    if more_compressible and more_robust:
        print("  => hypothesis SUPPORTED: semantic-bottleneck pretrain makes the QCMem")
        print("     cache point genuinely more compressible AND more robust to compression.")
    elif more_compressible or more_robust:
        print("  => PARTIAL support (see table); read the relative trend, not absolutes.")
    else:
        print("  => hypothesis NOT supported on these 1B/2000-step models (honest report).")


if __name__ == "__main__":
    main()
