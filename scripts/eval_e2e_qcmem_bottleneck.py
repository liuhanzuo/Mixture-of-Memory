#!/usr/bin/env python3
"""End-to-end QCMem-backbone eval: does semantic-bottleneck pretrain help?

Question (2026-07-08)
---------------------
The compressibility probe (``probe_bottleneck_qcmem_friendly.py``) showed the
bottleneck arm's *cache tensor* is more compressible / more robust in ISOLATION
(readout with NO cross-attention, chunk-by-chunk). This script closes the loop
with the ACTUAL QCMem operation the project cares about: use each pretrained 1B
model as a **QCMem backbone**, cache long-context chunks at the mid layer, apply
a rank-r compression to the cache, then let QCMem's real read path (pack
``[sink ; ctx-chunks ; query]`` + fresh contiguous RoPE + causal attention +
recompute ``layers[j:]``) predict the document's continuation. We report the
next-token NLL / top-1 acc of the query tail.

This is prediction-based, NOT QA — a from-scratch 1B model cannot do instruction
QA, but it CAN do "given the cached context, predict the following text", which
is exactly the project's operational definition of a useful memory.

Central claim being tested
--------------------------
Under the SAME compression budget r, is the bottleneck model's end-to-end NLL
lower, and does it degrade MORE SLOWLY as r shrinks (64/128)? Because the two
arms have different raw NLL (the bottleneck carries a ~5-7% ppl tax), the PRIMARY
metric is the compression-induced increment

    ΔNLL(r) = NLL(rank r) - NLL(rank r_max)

(smaller = more cache-friendly). We also print absolute NLL/acc for reference.
If the hypothesis holds, ΔNLL(rank64) for the bottleneck arm is markedly smaller
than for the baseline — end-to-end evidence that bottleneck pretrain makes the
QCMem cache genuinely more compressible under real query-aware read, which is the
concrete answer to "was the pretrain worth it?" and "how does it fare vs a
KV-CAT / raw-KV-reuse baseline?".

=============================== CACHE POINT (important) ========================
The bottleneck funnel WRAPS decoder ``layers[bottleneck_layer]`` (default 6) and
acts on its OUTPUT (``h -> down -> gelu -> up``, no residual). QCMem's
``write_chunk`` with ``resume_j=J`` runs ``layers[0:J]`` and caches the output of
``layers[J-1]``; ``read`` resumes ``layers[J:]``. Therefore, to make the cached
``h_j`` equal the low-rank POST-funnel tensor (the entire point), we must split
at ``resume_j = bottleneck_layer + 1`` (write includes the funnel layer; read
resumes just above it). This mirrors the proven convention in
``probe_bottleneck_qcmem_friendly.py`` (cache = post-layer-j output, readout =
``layers[j+1:]``). We therefore DEFAULT ``--resume_j`` to
``bottleneck_layer + 1`` (=7). ``--resume_j`` can still be overridden, but the
spec's suggested 6 would cache the PRE-funnel full-rank tensor and test the wrong
quantity, so we warn if a value below ``bottleneck_layer + 1`` is used with a
bottleneck model. For the baseline arm (no funnel) any split is equally valid; we
use the same ``resume_j`` for both arms for a strict apples-to-apples comparison.
================================================================================

Honesty (red line #2): these are weak 1B / from-scratch models — read the
RELATIVE trend (bottleneck vs baseline ΔNLL), not the absolute numbers.
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
from src.memory.qcmem import QCMemModel  # noqa: E402


# --------------------------------------------------------------------------- #
# checkpoint loading (raw state_dict + arch meta, same as the qcmem-friendly probe)
# --------------------------------------------------------------------------- #
def load_ckpt(path, device):
    """Rebuild the exact bottleneck arch from the ckpt's arch fields and load the
    raw state_dict. Returns (model, bottleneck_layer, bottleneck_dim)."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    bl = int(ck.get("bottleneck_layer", 6))
    bd = int(ck.get("bottleneck_dim", 0))
    seq_len = int(ck.get("seq_len", 2048))
    model = build_bottleneck_model(bottleneck_layer=bl, bottleneck_dim=bd,
                                   seq_len=seq_len, dtype=torch.bfloat16)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing or unexpected:
        print(f"  [load] missing={len(missing)} unexpected={len(unexpected)} (bd={bd})")
    model.to(device).eval()
    return model, bl, bd


# --------------------------------------------------------------------------- #
# rank-r cache compression (PCA fit on the pooled cached h_j of the run)
# --------------------------------------------------------------------------- #
def fit_pca(H_flat_f32):
    """H_flat: [M, d] float32 (GPU). Returns (mean[1,d], evecs[d,d] desc, evals desc)."""
    M, d = H_flat_f32.shape
    mean = H_flat_f32.mean(0, keepdim=True)
    Xc = H_flat_f32 - mean
    cov = (Xc.t() @ Xc) / max(M - 1, 1)          # d x d (d=2048, cheap)
    evals, evecs = torch.linalg.eigh(cov)        # ascending
    evals = torch.clamp(evals.flip(0), min=0.0)  # descending
    evecs = evecs.flip(1)                        # columns = PCs, desc
    return mean, evecs, evals


def rank_r_reconstruct(h_bf16, mean, evecs, r):
    """Project one cached chunk h_j [1,T,d] onto the top-r PCs and reconstruct.
    r>=d is a no-op (returns the tensor unchanged). Returns [1,T,d] bf16 on h's
    device."""
    d = h_bf16.shape[-1]
    if r >= d:
        return h_bf16
    dev = evecs.device
    Vr = evecs[:, :r]                            # [d, r]
    X = h_bf16.reshape(-1, d).float().to(dev)
    Xc = X - mean
    Xr = (Xc @ Vr) @ Vr.t() + mean               # reconstruct
    return Xr.reshape(h_bf16.shape).to(torch.bfloat16).to(h_bf16.device)


# --------------------------------------------------------------------------- #
# document construction: concat consecutive val rows to reach ctx_len tokens
# --------------------------------------------------------------------------- #
def build_docs(arr, n_docs, ctx_len, row_len):
    """Yield n_docs LongTensor[ctx_len] documents by concatenating consecutive
    rows of the val array (each row is row_len tokens). Non-overlapping."""
    rows_per_doc = (ctx_len + row_len - 1) // row_len
    docs = []
    total_rows = arr.shape[0]
    r0 = 0
    while len(docs) < n_docs and r0 + rows_per_doc <= total_rows:
        seg = np.asarray(arr[r0:r0 + rows_per_doc]).reshape(-1)[:ctx_len]
        docs.append(torch.from_numpy(seg.astype(np.int64)))
        r0 += rows_per_doc
    return docs


# --------------------------------------------------------------------------- #
# per-arm evaluation
# --------------------------------------------------------------------------- #
@torch.no_grad()
def eval_arm(name, ckpt, docs, args, device):
    print(f"\n================= {name}: {ckpt} =================")
    model, bl, bd = load_ckpt(ckpt, device)

    resume_j = args.resume_j if args.resume_j is not None else (bl + 1)
    if bd > 0 and resume_j <= bl:
        print(f"  [WARN] resume_j={resume_j} <= bottleneck_layer={bl}: the cache "
              f"tensor is the PRE-funnel (full-rank) hidden, NOT the low-rank "
              f"post-funnel tensor. The bottleneck advantage will be hidden. "
              f"Recommended resume_j = bottleneck_layer+1 = {bl+1}.")
    qc = QCMemModel(model, resume_j=resume_j)
    print(f"  arch: bottleneck_layer={bl} bottleneck_dim={bd} num_layers={qc.num_layers} "
          f"| QCMem resume_j={resume_j} (write layers[0:{resume_j}] incl funnel; "
          f"read resumes layers[{resume_j}:])")

    chunk = args.chunk_size
    qlen = args.query_len
    ranks = sorted(set(args.ranks), reverse=True)
    r_max = max(ranks)

    # BOS sink id (Llama-3): matches the babilong eval convention (bos hidden at pos 0).
    bos_id = 128000

    # -- 1st pass: write every doc's chunks + query, collect cached h_j for PCA fit --
    per_doc = []          # list of dict(ctx_hj=[...], query_hj=tensor, tgt=LongTensor[qlen])
    pool = []             # flattened h_j vectors for PCA fit
    for doc in docs:
        doc = doc.to(device)
        ctx_ids = doc[:-qlen]                 # everything except the last qlen tokens
        query_ids = doc[-qlen:]               # the prediction target segment
        # write context chunks (chunk-local, cached at depth resume_j)
        ctx_chunks = list(ctx_ids.split(chunk))
        ctx_hj = [qc.write_chunk(c) for c in ctx_chunks if c.numel() > 0]
        # sink = BOS depth-j hidden (attention-sink anchor at packed pos 0)
        sink_hj = qc.write_chunk([bos_id])
        # query chunk cached the same way; its own h_j is packed last
        query_hj = qc.write_chunk(query_ids)
        per_doc.append({
            "sink_hj": sink_hj,
            "ctx_hj": ctx_hj,
            "query_hj": query_hj,
            "tgt": query_ids,
        })
        d = query_hj.shape[-1]
        # PCA basis is fit on the CONTEXT chunk h_j (the tensors we compress).
        # sink (1 token, degenerate) and query (live, uncompressed) are excluded.
        for h in ctx_hj:
            pool.append(h.reshape(-1, d).float())

    H_pool = torch.cat(pool, dim=0).to(device)    # [M, d]
    mean, evecs, evals = fit_pca(H_pool)
    total = evals.sum().clamp_min(1e-12)
    cum = torch.cumsum(evals, 0) / total
    dim99 = int(torch.searchsorted(cum, torch.tensor(0.99, device=cum.device)).item()) + 1
    print(f"  cache pool: {tuple(H_pool.shape)} vectors  PCA dim@99%={dim99}/{H_pool.shape[1]}  "
          f"top1_var={float(evals[0]/total):.4f}")
    del H_pool, pool
    torch.cuda.empty_cache()

    # -- 2nd pass: for each rank, compress cache, run QCMem read, score query tail --
    results = {}
    for r in ranks:
        nll_sum = 0.0
        acc_sum = 0
        tok_count = 0
        for rec in per_doc:
            # Compress ONLY the selected context chunks (the cached "memory") to
            # rank r. The sink (1-token anchor) and the query h_j are the LIVE
            # read input and stay uncompressed — matches the spec pack
            # [sink ; compressed ctx h_j ; query h_j] and the real QCMem operation
            # (only the retrieved memory is compressed, not the live query).
            sink_c = rec["sink_hj"]
            ctx_c = [rank_r_reconstruct(h, mean, evecs, r) for h in rec["ctx_hj"]]
            query_c = rec["query_hj"]
            # QCMem read: pack [sink ; ctx... ; query], resume layers[j:], logits at query tail.
            logits = qc.read_core(sink_c, ctx_c, query_c, logits_tail=qlen)  # [1, qlen, V]
            logits = logits[0].float()                      # [qlen, V]
            tgt = rec["tgt"]                                 # [qlen]
            # next-token: predict tgt[1:] from positions 0:qlen-1 (shift within query tail)
            pred_logits = logits[:-1]                        # [qlen-1, V]
            gold = tgt[1:]                                   # [qlen-1]
            nll_sum += torch.nn.functional.cross_entropy(
                pred_logits, gold, reduction="sum").item()
            acc_sum += int((pred_logits.argmax(-1) == gold).sum().item())
            tok_count += gold.numel()
        nll = nll_sum / max(tok_count, 1)
        acc = acc_sum / max(tok_count, 1)
        results[r] = {"nll": round(nll, 4), "acc": round(acc, 4),
                      "ppl": round(float(np.exp(min(nll, 20))), 2)}
        print(f"  rank {r:4d}: NLL={nll:.4f} ppl={results[r]['ppl']:.2f} acc={acc:.4f}")

    # ΔNLL relative to the largest (least-compressed) rank
    base_nll = results[r_max]["nll"]
    for r in ranks:
        results[r]["dNLL"] = round(results[r]["nll"] - base_nll, 4)

    del model, qc
    torch.cuda.empty_cache()
    return {
        "bottleneck_layer": bl, "bottleneck_dim": bd, "resume_j": resume_j,
        "chunk_size": chunk, "query_len": qlen, "n_docs": len(docs),
        "pca_dim99": dim99, "ranks": ranks, "r_max": r_max,
        "by_rank": {str(r): results[r] for r in ranks},
    }


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--baseline_ckpt", default="outputs/sembott_1b_baseline_8k/final.pt")
    ap.add_argument("--bottleneck_ckpt", default="outputs/sembott_1b_bottleneck_8k/final.pt")
    ap.add_argument("--val_path", default="data/slimpajama_val_4096_llama3.npy")
    ap.add_argument("--resume_j", type=int, default=None,
                    help="QCMem split depth. Default = bottleneck_layer+1 (cache the "
                         "post-funnel low-rank tensor). See CACHE POINT note in docstring.")
    ap.add_argument("--ranks", type=str, default="512,256,128,64",
                    help="comma-separated PCA ranks for cache compression; the max is the "
                         "least-compressed reference for ΔNLL.")
    ap.add_argument("--n_docs", type=int, default=50)
    ap.add_argument("--ctx_len", type=int, default=8192,
                    help="tokens per document (concatenated from consecutive val rows).")
    ap.add_argument("--chunk_size", type=int, default=512, help="context chunk size (tokens).")
    ap.add_argument("--query_len", type=int, default=512,
                    help="last query_len tokens = prediction target segment.")
    ap.add_argument("--out_json", default="outputs/e2e_qcmem_bottleneck.json")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    ranks = [int(x) for x in args.ranks.split(",") if x.strip()]
    args.ranks = ranks
    device = args.device

    arr = np.load(args.val_path, mmap_mode="r")
    row_len = arr.shape[1]
    docs = build_docs(arr, args.n_docs, args.ctx_len, row_len)
    if len(docs) < args.n_docs:
        print(f"[WARN] requested {args.n_docs} docs but val set only yields {len(docs)} "
              f"at ctx_len={args.ctx_len} (row_len={row_len}).")
    print(f"built {len(docs)} docs x {args.ctx_len} tok (chunk={args.chunk_size}, "
          f"query_len={args.query_len}); ranks={ranks}")

    result = {"config": {
        "baseline_ckpt": args.baseline_ckpt, "bottleneck_ckpt": args.bottleneck_ckpt,
        "val_path": args.val_path, "resume_j": args.resume_j, "ranks": ranks,
        "n_docs": len(docs), "ctx_len": args.ctx_len, "chunk_size": args.chunk_size,
        "query_len": args.query_len,
    }}
    result["baseline"] = eval_arm("baseline", args.baseline_ckpt, docs, args, device)
    result["bottleneck"] = eval_arm("bottleneck", args.bottleneck_ckpt, docs, args, device)

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nsaved {args.out_json}")

    # ------------------------------- VERDICT ------------------------------- #
    b, n = result["baseline"], result["bottleneck"]
    ranks_desc = sorted(set(ranks), reverse=True)
    print("\n==================== E2E QCMem-backbone VERDICT ====================")
    print(f"resume_j baseline={b['resume_j']} bottleneck={n['resume_j']}  "
          f"(bottleneck_dim: base={b['bottleneck_dim']}, bott={n['bottleneck_dim']})")
    print("\n[1] ABSOLUTE query-tail NLL  (lower = better prediction from compressed cache)")
    print(f"    {'rank':>6}{'baseline NLL':>16}{'bottleneck NLL':>18}{'winner':>10}")
    for r in ranks_desc:
        bn = b["by_rank"][str(r)]["nll"]
        nn = n["by_rank"][str(r)]["nll"]
        win = "bott" if nn < bn else "base"
        print(f"    {r:>6}{bn:>16.4f}{nn:>18.4f}{win:>10}")

    print("\n[2] ΔNLL(r) = NLL(rank r) - NLL(rank r_max)  (smaller rise = more cache-friendly)")
    print(f"    {'rank':>6}{'baseline ΔNLL':>16}{'bottleneck ΔNLL':>18}{'winner':>10}")
    for r in ranks_desc:
        bd_ = b["by_rank"][str(r)]["dNLL"]
        nd_ = n["by_rank"][str(r)]["dNLL"]
        win = "bott" if nd_ <= bd_ else "base"
        print(f"    {r:>6}{bd_:>16.4f}{nd_:>18.4f}{win:>10}")

    print("\n[3] query-tail top-1 acc")
    print(f"    {'rank':>6}{'baseline acc':>16}{'bottleneck acc':>18}")
    for r in ranks_desc:
        print(f"    {r:>6}{b['by_rank'][str(r)]['acc']:>16.4f}{n['by_rank'][str(r)]['acc']:>18.4f}")

    # heuristic verdict: bottleneck's ΔNLL at the smallest rank is <= baseline's
    r_min = min(ranks_desc)
    b_dnll = b["by_rank"][str(r_min)]["dNLL"]
    n_dnll = n["by_rank"][str(r_min)]["dNLL"]
    all_le = all(n["by_rank"][str(r)]["dNLL"] <= b["by_rank"][str(r)]["dNLL"] + 1e-9
                 for r in ranks_desc)
    print("\n[VERDICT]")
    print(f"  at smallest rank {r_min}: baseline ΔNLL={b_dnll:+.4f}  bottleneck ΔNLL={n_dnll:+.4f}")
    if n_dnll < b_dnll - 1e-4 and all_le:
        print("  => SUPPORTED: as an end-to-end QCMem backbone, the semantic-bottleneck")
        print("     model loses LESS to cache compression at every rank — the pretrain")
        print("     genuinely bought a more compressible / KV-CAT-competitive cache.")
    elif n_dnll < b_dnll - 1e-4:
        print("  => PARTIAL: bottleneck degrades less at the smallest rank but not")
        print("     monotonically at every rank; read the ΔNLL table.")
    else:
        print("  => NOT supported end-to-end on these 1B models: the bottleneck cache is")
        print("     not measurably more compression-robust under real QCMem read. Honest.")


if __name__ == "__main__":
    main()
