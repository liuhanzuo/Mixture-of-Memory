#!/usr/bin/env python
"""QCMem Insight 2 probe — "the bottom-layer activations are compressible".

MECHANISM CLAIM UNDER TEST
--------------------------
QCMem caches the depth-j hidden ``h_j`` of every context chunk (see
``QCMemModel.write_chunk``). That cache is the memory-budget cost of the method.
Insight 2 claims that a SHALLOW ``h_j`` is redundant / low intrinsic dimension,
so it can be stored compressed (low-rank / fewer bits) without hurting the
downstream read-out. This script measures the intrinsic dimensionality of the
per-token ``h_j`` vectors across depths.

We collect the raw per-token depth-j hidden vectors (NOT mean-pooled — the cache
stores every token's ``h_j``), exactly as produced by ``write_chunk`` (chunk-local
embed + causal mask + RoPE 0:T), for a sweep of depths ``j``, and compute:

  (a) PCA cumulative variance — #principal components needed to retain
      90% / 95% / 99% of the (centered) variance. Fewer components == more
      linearly compressible.
  (b) Effective rank via two standard estimators over the covariance spectrum:
        * stable rank      = (Σσ_i) / σ_max              (nuclear/spectral ratio,
                             using singular values σ_i of the centered data;
                             equivalently trace/λ_max on the covariance)
        * entropy eff. rank = exp(−Σ p_i log p_i), p_i = λ_i / Σλ_j  (Roy &
                             Vetterli spectral entropy on the eigenvalues λ_i)
  (c) Low-rank reconstruction error — relative Frobenius error ‖X − X_r‖/‖X‖
      when X is projected onto its top-r PCA components, for a few r. This is the
      information actually lost by a rank-r store.

All computed in fp32 on centered data (mean subtracted) so raw scale / a large
mean vector doesn't dominate; d = hidden_size (4096 for Qwen3-8B).

KEY LOOK: if shallow j (6, 12) has markedly LOWER PCA-dims-for-95% and LOWER
effective rank than deep j (18, 24), the shallow cache is more compressible ->
supports Insight 2. Reported honestly, including the case where shallow layers
are NOT lower-dimensional than deep ones.

Forward-only, no generation — fast. Default budget: ~20 chunks × 512 tokens ≈
10k vectors per depth.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(PROJECT_ROOT, ".hf_cache", "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
# Cap BLAS threads: several probes may run in parallel (one per GPU), and an
# uncapped LAPACK/MKL grabs every core -> catastrophic oversubscription on the
# CPU eig/matmul path. 8 threads is plenty for a single 4096×4096 eigh.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input,
)

from src.memory.qcmem import QCMemModel  # noqa: E402

_harness_path = os.path.join(PROJECT_ROOT, "scripts", "run_babilong_mem_space.py")
_spec = _ilu.spec_from_file_location("_qcmem_harness_probe2", _harness_path)
harness = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(harness)


# --------------------------------------------------------------------------- #
# multi-depth chunk-local encoder — returns PER-TOKEN hidden at each depth
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_chunk_pertoken_multidepth(qc: QCMemModel, token_ids, depths):
    """Run one chunk-local forward (== ``write_chunk`` path) and return, for each
    depth j in ``depths``, the FULL per-token hidden ``[T, d]`` (fp32, on CPU to
    keep GPU memory flat across many chunks). depth j == hidden after j layers."""
    ids = qc._as_ids(token_ids)
    T = ids.shape[1]
    inputs_embeds = qc.embed_tokens(ids)
    positions = torch.arange(T, device=qc.device).unsqueeze(0)
    causal_mask, position_embeddings = qc._make_mask_and_rope(inputs_embeds, positions)

    want = set(int(j) for j in depths)
    max_j = max(want)
    hidden = inputs_embeds
    out = {}
    for li in range(max_j):
        hidden = qc.layers[li](
            hidden,
            attention_mask=causal_mask,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        depth = li + 1
        if depth in want:
            out[depth] = hidden.float().squeeze(0).cpu()  # [T, d]
    return out


# --------------------------------------------------------------------------- #
# compressibility metrics on a [N, d] fp32 matrix
# --------------------------------------------------------------------------- #
def _spectrum_metrics(eig, var_targets, recon_ranks):
    """Given descending non-negative eigenvalues ``eig`` (Tensor[k]) of a
    covariance/Gram matrix, compute the intrinsic-dimensionality read-outs.

    All quantities are scale-invariant ratios of the spectrum, so whether ``eig``
    holds covariance eigenvalues or Gram eigenvalues (differ by the 1/(N-1)
    factor) is irrelevant."""
    total_var = float(eig.sum().item())
    if total_var <= 0:
        return {"degenerate": True}
    sig = eig.sqrt()
    cum = torch.cumsum(eig, dim=0) / eig.sum()
    dims_for = {}
    for t in var_targets:
        idx = int(torch.searchsorted(cum, torch.tensor(float(t), dtype=cum.dtype)).item()) + 1
        dims_for[str(t)] = min(idx, len(eig))
    stable_rank = float((sig.sum() / (sig.max() + 1e-12)).item())      # (Σσ)/σ_max
    p = eig / eig.sum()
    ent = float(-(p * (p + 1e-300).log()).sum().item())
    entropy_eff_rank = float(torch.exp(torch.tensor(ent)).item())      # exp(H(λ))
    part_ratio = float(((eig.sum() ** 2) / (eig ** 2).sum()).item())   # (Σλ)²/Σλ²
    top1_share = float((eig[0] / eig.sum()).item())                    # λ_max / Σλ
    recon = {}
    for r in recon_ranks:
        r = int(min(r, len(eig)))
        if r <= 0:
            continue
        tail = float(eig[r:].sum().item())
        recon[str(r)] = (tail / total_var) ** 0.5   # rel Frobenius err of rank-r trunc
    return {
        "degenerate": False,
        "dims_for_var": dims_for,
        "stable_rank": stable_rank,
        "entropy_eff_rank": entropy_eff_rank,
        "participation_ratio": part_ratio,
        "top1_eig_share": top1_share,
        "recon_rel_frob_err": recon,
    }


def _cov_eigvals(Xc, eig_device):
    """Descending non-negative eigenvalues of the d×d Gram matrix ``Xcᵀ Xc``.

    For N ≫ d this d×d eigendecomposition is far cheaper than a full N×d SVD and
    numerically equivalent for the spectrum (PCA variance / effective rank /
    truncation error all depend only on the eigenvalues). float64 for stability."""
    dev = torch.device(eig_device)
    Xcd = Xc.to(dev).double()
    C = Xcd.t() @ Xcd                                        # [d, d]
    evals = torch.linalg.eigvalsh(C)                          # ascending
    return torch.flip(evals, dims=[0]).clamp_min(0).cpu()    # descending λ_i


def compressibility_metrics(X: torch.Tensor, var_targets, recon_ranks, eig_device="cpu"):
    """X: [N, d] fp32 (per-token depth-j hidden vectors). Returns a metrics dict
    with TWO spectral views + norm-outlier diagnostics.

    Why two views. Deep-layer transformer hidden states are dominated by a few
    "massive-activation" / attention-sink tokens whose norm is orders of
    magnitude larger than the rest (Sun et al. 2024). Those few outlier rows
    inflate the *raw* covariance so its top eigenvalue swallows ~all variance,
    making raw-PCA "#components for 95% var" collapse to ~1 at deep layers — an
    artefact of norm scale, NOT of genuine low-dimensional geometry. To probe the
    actual directional intrinsic dimensionality of the cache we therefore also
    report the spectrum of the L2-NORMALISED (unit-norm) tokens, which removes
    the per-token magnitude and asks "how many directions do the hiddens span?".

      * ``raw``        — spectrum of the mean-centered raw hiddens. This is what a
                         naive linear (PCA) store of the cache would see, INCLUDING
                         the outlier-norm effect. Reported for honesty.
      * ``unit``       — spectrum of the mean-centered, per-token L2-normalised
                         hiddens (directional intrinsic dimension; outlier-norm
                         removed).

    Plus outlier diagnostics: distribution of per-token L2 norms (a heavy tail ==
    massive activations present)."""
    N, d = X.shape
    # --- norm-outlier diagnostics -------------------------------------------
    norms = X.norm(dim=1)                                    # [N]
    med = float(norms.median().item())
    mx = float(norms.max().item())
    frac_gt_3x = float((norms > 3 * med).float().mean().item()) if med > 0 else 0.0
    frac_gt_10x = float((norms > 10 * med).float().mean().item()) if med > 0 else 0.0

    # --- raw-centered spectrum ----------------------------------------------
    mean = X.mean(dim=0, keepdim=True)
    Xc = X - mean
    eig_raw = _cov_eigvals(Xc, eig_device)
    if float(eig_raw.sum().item()) <= 0:
        return {"degenerate": True, "N": N, "d": d}
    raw = _spectrum_metrics(eig_raw, var_targets, recon_ranks)

    # --- unit-norm (directional) spectrum -----------------------------------
    Xn = X / (norms.unsqueeze(1) + 1e-8)
    Xn = Xn - Xn.mean(dim=0, keepdim=True)
    eig_unit = _cov_eigvals(Xn, eig_device)
    unit = _spectrum_metrics(eig_unit, var_targets, recon_ranks)

    return {
        "degenerate": False, "N": N, "d": d,
        "raw": raw,
        "unit": unit,
        "mean_norm": float(mean.norm().item()),
        "median_token_norm": med,
        "max_token_norm": mx,
        "rms_token_norm": float(norms.mean().item()),
        "frac_norm_gt_3x_median": frac_gt_3x,
        "frac_norm_gt_10x_median": frac_gt_10x,
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QCMem Insight-2 hidden compressibility probe")
    ap.add_argument("--model_path", type=str,
                    default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b")
    ap.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    ap.add_argument("--task", type=str, default="qa1")
    ap.add_argument("--length", type=str, default="8k")
    ap.add_argument("--depths", type=int, nargs="+", default=[6, 12, 18, 24])
    ap.add_argument("--n_chunks", type=int, default=20,
                    help="Number of 512-token chunks to pool per depth (~n*512 vectors).")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--var_targets", type=float, nargs="+", default=[0.90, 0.95, 0.99])
    ap.add_argument("--recon_ranks", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    ap.add_argument("--max_docs", type=int, default=40,
                    help="Max samples to walk when harvesting chunks.")
    ap.add_argument("--chunk_size_source", type=str, default="context",
                    choices=["context", "any"],
                    help="'context' skips each doc's last (query) chunk; 'any' keeps all.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--eig_device", type=str, default="",
                    help="Device for the d×d covariance eigendecomposition "
                         "(default: same as --device; use 'cpu' to force CPU).")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[probe2] model_path={args.model_path}")
    print(f"[probe2] depths={args.depths} n_chunks={args.n_chunks} chunk_size={args.chunk_size} "
          f"task={args.task} length={args.length} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    L = int(model.config.num_hidden_layers)
    depths = sorted(j for j in args.depths if 1 <= j <= L)
    if not depths:
        raise SystemExit(f"no valid depths in [1,{L}] from {args.depths}")
    qc = QCMemModel(model, resume_j=max(depths))
    print(f"[probe2] model layers L={L}; probing depths {depths}")

    if args.task not in DEFAULT_PROMPTS:
        raise SystemExit(f"task {args.task} not in DEFAULT_PROMPTS")
    cfg = {
        "instruction": DEFAULT_PROMPTS[args.task]["instruction"],
        "examples":    DEFAULT_PROMPTS[args.task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[args.task]["post_prompt"],
        "template":    DEFAULT_TEMPLATE,
    }
    data = harness.load_babilong_dataset(args.dataset_name, args.length)
    task_data = data[args.task]

    # Harvest per-token hidden vectors per depth until we have n_chunks chunks.
    pools = {j: [] for j in depths}
    collected = 0
    n_docs = min(len(task_data), args.max_docs)
    for idx in tqdm(range(n_docs), desc="harvest", leave=False):
        if collected >= args.n_chunks:
            break
        sample = task_data[idx]
        input_text = get_formatted_input(
            sample["input"], sample["question"],
            cfg["examples"], cfg["instruction"], cfg["post_prompt"],
            template=cfg["template"],
        )
        ids = tokenizer.encode(input_text, add_special_tokens=True,
                               return_tensors="pt").to(device)
        tokens = ids[0]
        chunks = list(tokens.split(args.chunk_size))
        if args.chunk_size_source == "context" and len(chunks) >= 2:
            chunks = chunks[:-1]  # drop the query chunk
        for c in chunks:
            if collected >= args.n_chunks:
                break
            if c.shape[0] < 8:      # skip tiny trailing chunk
                continue
            ev = encode_chunk_pertoken_multidepth(qc, c, depths)
            for j in depths:
                pools[j].append(ev[j])  # [T, d] on CPU
            collected += 1

    print(f"[probe2] harvested {collected} chunks "
          f"({sum(t.shape[0] for t in pools[depths[0]])} tokens) per depth")

    results = {}
    eig_device = args.eig_device or args.device
    for j in depths:
        X = torch.cat(pools[j], dim=0)          # [N, d]
        m = compressibility_metrics(X, args.var_targets, args.recon_ranks,
                                    eig_device=eig_device)
        results[j] = m
        del X

    _print_table(depths, args.var_targets, args.recon_ranks, results, L)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": vars(args), "L": L, "depths": depths,
                       "results": {str(j): v for j, v in results.items()}}, f, indent=2)
        print(f"[probe2] wrote {args.out_json}")


def _print_view(depths, var_targets, recon_ranks, results, view, title):
    print(f"\n  [{view.upper()} spectrum] {title}")
    var_cols = "".join(f"  #PC@{int(t*100)}% " for t in var_targets)
    print(f"  j  |{var_cols}| stable_rk  entropy_rk  part_ratio  top1_share")
    print("  " + "-" * 86)
    for j in depths:
        m = results[j]
        if m.get("degenerate") or m.get(view, {}).get("degenerate"):
            print(f" {j:>2}  | (degenerate)")
            continue
        v = m[view]
        cells = "".join(f"   {v['dims_for_var'][str(t)]:>5} " for t in var_targets)
        print(f" {j:>2}  |{cells}|  {v['stable_rank']:8.2f}  {v['entropy_eff_rank']:9.2f}  "
              f"{v['participation_ratio']:9.2f}  {v['top1_eig_share']:9.3f}")
    print("  " + "-" * 86)
    print(f"  [{view.upper()}] rank-r truncation -> relative Frobenius reconstruction error")
    rcols = "".join(f"  r={r:<4}" for r in recon_ranks)
    print(f"  j  |{rcols}")
    print("  " + "-" * 86)
    for j in depths:
        m = results[j]
        if m.get("degenerate") or m.get(view, {}).get("degenerate"):
            continue
        v = m[view]
        cells = "".join(
            f"  {v['recon_rel_frob_err'].get(str(int(min(r, m['d']))), float('nan')):.4f}"
            for r in recon_ranks)
        print(f" {j:>2}  |{cells}")
    print("  " + "-" * 86)


def _print_table(depths, var_targets, recon_ranks, results, L):
    d = results[depths[0]].get("d", "?")
    N = results[depths[0]].get("N", "?")
    print("\n" + "=" * 90)
    print(f"[probe2] intrinsic dimensionality of depth-j hidden  (d={d}, N={N} tokens/depth, L={L})")
    print("=" * 90)

    # per-token norm-outlier diagnostics (massive-activation / sink detector)
    print("  norm-outlier diagnostics (per-token L2 norm distribution):")
    print("  j  |  median   max      max/med   frac>3x   frac>10x")
    print("  " + "-" * 60)
    for j in depths:
        m = results[j]
        if m.get("degenerate"):
            continue
        ratio = (m["max_token_norm"] / m["median_token_norm"]) if m["median_token_norm"] > 0 else float("nan")
        print(f" {j:>2}  |  {m['median_token_norm']:7.2f} {m['max_token_norm']:9.1f}  "
              f"{ratio:8.1f}  {m['frac_norm_gt_3x_median']:8.4f}  {m['frac_norm_gt_10x_median']:8.4f}")
    print("  " + "-" * 60)

    _print_view(depths, var_targets, recon_ranks, results, "raw",
                "raw mean-centered hiddens (what a naive linear store sees; INCLUDES norm-outliers)")
    _print_view(depths, var_targets, recon_ranks, results, "unit",
                "L2-normalised (directional) hiddens (norm-outlier effect removed)")


if __name__ == "__main__":
    main()
