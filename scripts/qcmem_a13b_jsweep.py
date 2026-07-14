#!/usr/bin/env python
"""QCMem depth-partition **j-sweep** on the PUBLIC 32-layer **Hunyuan-A13B**
(``HunYuanMoEV1``) MoE backbone, sharded across the local GPUs (2026-07-14).

Sibling of ``scripts/qcmem_hy3_jsweep.py`` (the 80-layer internal Hy3 sweep). Same
metric definitions and control flow; the ONLY differences are the model family
(A13B via :func:`load_a13b_qcmem` — native ``HunYuanMoEV1``, ``head_dim=128``,
``experts_implementation="eager"``) and the data source (a pre-tokenised Hunyuan
``.npy`` of shape ``(N, seq_len)`` instead of raw PG19 text — the ids are already
Hunyuan-vocab so no tokeniser round-trip is needed).

Goal
----
Find the *split-j* = the "cacheable-semantic ceiling vs LM-tax" knee for QCMem on
A13B. The parent ``QCMemModel`` runs the bottom ``layers[0:j]`` chunk-local at WRITE
(cached as ``h_j``) and resumes ``layers[j:]`` over the packed
``[sink ; ctx chunks ; query]`` at READ. As ``j`` grows more depth is spent
chunk-local (no cross-chunk / query attention) and fewer top layers remain to
integrate the retrieved context, so readout fidelity degrades. ``j=0`` == selective
full re-forward (RAG upper bound; numerically == full forward per the self-test,
diff 0.0); ``j=L`` == closed-book. We locate the knee.

A13B is 32 layers, so the 8B-era ≈0.375·L / Hy3 0.4·L sweet spot predicts
split-j ≈ 12–13. Default ``--j_list`` brackets that:
``0,4,8,10,12,13,14,16,20,24``.

Metrics (per ``j``, averaged over ``num_docs`` real Hunyuan-tokenised windows)
------------------------------------------------------------------------------
Per document window we build ONE packed sequence ``[bos ; ctx_1..ctx_N ; query]``
(all context chunks selected — isolate the DEPTH-partition effect, no selector
noise) and compute:

  * ``ppl``      : next-token perplexity of the QCMem readout (LM quality).
  * ``ppl_full`` : next-token perplexity of the full-context reference.
  * ``ppl_gap``  : ppl / ppl_full  (the multiplicative "LM tax" at this j).
  * ``kl``       : mean KL(full || qcmem) over the query positions (nats).
  * ``top1``     : fraction of query positions where the readout argmax matches
                   the full-context argmax.

The knee in ``ppl_gap`` / ``kl`` as a function of ``j`` is the split-j.

Usage
-----
    PYTHONPATH=$PWD HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
    .venv_hy3/bin/python scripts/qcmem_a13b_jsweep.py \
        --model_path models/Hunyuan-A13B-Pretrain \
        --data_path data/slimpajama_val_2048_hunyuan.npy \
        --j_list 0,4,8,10,12,13,14,16,20,24 \
        --chunk_size 256 --num_ctx_chunks 6 --query_len 256 --num_docs 8 \
        --out logs/a13b_jsweep_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.qcmem.qcmem_hy3 import load_a13b_qcmem, QCMemHy3Model  # noqa: E402


# --------------------------------------------------------------------------- #
# data: sample windows of pre-tokenised Hunyuan ids from a (N, seq_len) npy.
# --------------------------------------------------------------------------- #
def sample_npy_windows(npy_path, num_docs, window_tokens, seed=0):
    """Return a list of ``[1, window_tokens]`` LongTensors of Hunyuan ids.

    The npy is ``(N, seq_len)`` uint32. Each stored row is a packed 2048-token
    window; we take the first ``window_tokens`` ids of ``num_docs`` DISTINCT rows
    (evenly strided so the passages are unrelated). If ``window_tokens`` exceeds a
    single row we concatenate consecutive rows to reach the length. Rows shorter
    than needed (after concat, at the end of the file) are skipped.
    """
    arr = np.load(npy_path, mmap_mode="r")
    N, S = arr.shape
    rows_per_window = int(math.ceil(window_tokens / S))
    # distinct, evenly-spaced starting rows
    max_start = max(N - rows_per_window, 1)
    stride = max(max_start // max(num_docs, 1), 1)
    windows = []
    for i in range(num_docs):
        r0 = (i * stride) % max_start
        block = np.asarray(arr[r0:r0 + rows_per_window]).reshape(-1)
        if block.shape[0] < window_tokens:
            continue
        ids = torch.from_numpy(block[:window_tokens].astype(np.int64)).unsqueeze(0)
        windows.append(ids)
    return windows


# --------------------------------------------------------------------------- #
# metric helpers (identical to the Hy3 sweep)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def span_nll(logits, target_ids):
    """Mean next-token NLL (nats) of ``target_ids`` under ``logits`` [1,S,V]."""
    lg = logits[:, :-1, :].float()
    tg = target_ids[:, 1:].to(lg.device)
    logp = torch.log_softmax(lg, dim=-1)
    nll = -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    return nll.mean().item(), int(nll.numel())


@torch.no_grad()
def span_kl_top1(qc_logits, full_logits):
    """Mean KL(full || qcmem) in nats + top-1 argmax agreement over the span."""
    q = qc_logits.float()
    fu = full_logits.float().to(q.device)
    logp_q = torch.log_softmax(q, dim=-1)
    logp_f = torch.log_softmax(fu, dim=-1)
    p_f = logp_f.exp()
    kl = (p_f * (logp_f - logp_q)).sum(-1)
    top1 = (q.argmax(-1) == fu.argmax(-1)).float()
    return kl.mean().item(), top1.mean().item()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QCMem A13B depth-partition j-sweep")
    ap.add_argument("--model_path", default="models/Hunyuan-A13B-Pretrain")
    ap.add_argument("--data_path", default="data/slimpajama_val_2048_hunyuan.npy")
    ap.add_argument("--j_list", type=str, default="0,4,8,10,12,13,14,16,20,24")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--num_ctx_chunks", type=int, default=6)
    ap.add_argument("--num_ctx_list", type=str, default="",
                    help="comma list of num_ctx_chunks to sweep (overrides "
                         "--num_ctx_chunks), e.g. '6,12,24'.")
    ap.add_argument("--query_len", type=int, default=256)
    ap.add_argument("--num_docs", type=int, default=8)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--out", type=str, default="logs/a13b_jsweep_results.json")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    j_list = [int(x) for x in args.j_list.split(",") if x.strip() != ""]

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    from transformers import AutoTokenizer
    print(f"[a13b-jsweep] loading tokenizer {model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True)

    t0 = time.time()
    print(f"[a13b-jsweep] loading A13B device_map=auto dtype={dtype} "
          f"attn={args.attn_impl} experts_implementation=eager", flush=True)
    # qc0 (resume_j=0) shares the loaded backbone; all other j re-wrap it cheaply.
    qc0 = load_a13b_qcmem(model_path, resume_j=0, dtype=dtype,
                          device_map="auto", attn_implementation=args.attn_impl)
    model = qc0.model
    dm = getattr(model, "hf_device_map", None)
    if dm is not None:
        devs = sorted({str(v) for v in dm.values()})
        print(f"[a13b-jsweep] hf_device_map spans {len(devs)} device(s): {devs}",
              flush=True)
    L = int(model.config.num_hidden_layers)
    V = int(model.config.vocab_size)
    print(f"[a13b-jsweep] loaded in {time.time()-t0:.0f}s | L={L} V={V}", flush=True)

    j_list = [j for j in j_list if 0 <= j <= L]
    print(f"[a13b-jsweep] j_list={j_list}", flush=True)

    ctx_list = ([int(x) for x in args.num_ctx_list.split(",") if x.strip()]
                if args.num_ctx_list.strip() else [args.num_ctx_chunks])
    print(f"[a13b-jsweep] num_ctx_list={ctx_list} (chunk_size={args.chunk_size})",
          flush=True)

    # BOS: Hunyuan pretrain has eos_token_id=None; bos_token_id may also be None.
    bos_id = getattr(tok, "bos_token_id", None)
    if bos_id is None:
        bos_id = getattr(model.config, "bos_token_id", None)
    if bos_id is None:
        bos_id = 1
    embed_dev = next(model.model.embed_tokens.parameters()).device
    bos = torch.tensor([[int(bos_id)]], device=embed_dev, dtype=torch.long)
    print(f"[a13b-jsweep] bos_id={int(bos_id)}", flush=True)

    qc_by_j = {j: QCMemHy3Model(model, resume_j=j) for j in j_list}

    all_blocks = []

    for num_ctx in ctx_list:
        C = args.chunk_size
        window_tokens = num_ctx * C + args.query_len
        windows = sample_npy_windows(data_path, args.num_docs, window_tokens)
        print(f"\n[a13b-jsweep] === ctx={num_ctx}x{C}={num_ctx*C} tok + "
              f"q{args.query_len} ({len(windows)} docs) ===", flush=True)

        docs = []
        for ids in windows:
            ids = ids.to(embed_dev)
            ctx = [ids[:, k * C:(k + 1) * C] for k in range(num_ctx)]
            query = ids[:, num_ctx * C:]
            packed_ids = torch.cat([bos] + ctx + [query], dim=1)
            full_logits = qc0.full_forward_logits(packed_ids)
            full_tail = full_logits[:, -args.query_len:, :].detach().float().cpu()
            docs.append(dict(ctx=ctx, query=query, full_tail=full_tail))
        print(f"[a13b-jsweep] computed full-context reference for {len(docs)} docs "
              f"(packed ~{num_ctx*C + args.query_len + 1} tok)", flush=True)

        results = []
        for j in j_list:
            qc = qc_by_j[j]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            tot_nll = tot_ntok = 0.0
            tot_nll_full = 0.0
            kl_sum = top1_sum = 0.0
            n_docs = 0
            tj = time.time()
            for doc in docs:
                ctx, query, full_tail = doc["ctx"], doc["query"], doc["full_tail"]
                sink_hj = qc.write_chunk(bos)
                ctx_hj = [qc.write_chunk(c) for c in ctx]
                query_hj = qc.write_chunk(query)
                qc_tail = qc.read_core(sink_hj, ctx_hj, query_hj,
                                       logits_tail=args.query_len).detach()
                nll, ntok = span_nll(qc_tail, query)
                nll_full, _ = span_nll(full_tail, query)
                kl, top1 = span_kl_top1(qc_tail, full_tail)
                tot_nll += nll * ntok
                tot_nll_full += nll_full * ntok
                tot_ntok += ntok
                kl_sum += kl
                top1_sum += top1
                n_docs += 1

            ppl = math.exp(tot_nll / tot_ntok)
            ppl_full = math.exp(tot_nll_full / tot_ntok)
            rec = dict(
                num_ctx=num_ctx, ctx_tokens=num_ctx * C,
                j=j, frac=round(j / L, 3),
                ppl=round(ppl, 4), ppl_full=round(ppl_full, 4),
                ppl_gap=round(ppl / ppl_full, 4),
                kl_nats=round(kl_sum / n_docs, 5),
                top1=round(top1_sum / n_docs, 4),
                n_docs=n_docs, secs=round(time.time() - tj, 1),
            )
            results.append(rec)
            print(f"[a13b-jsweep] ctx={num_ctx:>2} j={j:>3} (frac={rec['frac']:.3f})  "
                  f"ppl={rec['ppl']:.3f} ppl_full={rec['ppl_full']:.3f} "
                  f"gap={rec['ppl_gap']:.3f}x KL={rec['kl_nats']:.4f} "
                  f"top1={rec['top1']:.3f} ({rec['secs']:.0f}s)", flush=True)

        print("-" * 78, flush=True)
        print(f"ctx={num_ctx*C} tok | {'j':>4} {'frac':>6} {'ppl':>8} "
              f"{'gap':>7} {'KL':>8} {'top1':>7}", flush=True)
        for r in results:
            print(f"           | {r['j']:>4} {r['frac']:>6.3f} {r['ppl']:>8.3f} "
                  f"{r['ppl_gap']:>6.3f}x {r['kl_nats']:>8.4f} {r['top1']:>7.3f}",
                  flush=True)
        faithful = [r for r in results if r["ppl_gap"] <= 1.15 and r["top1"] >= 0.80]
        knee = max((r["j"] for r in faithful), default=None)
        print(f"[a13b-jsweep] ctx={num_ctx*C}: split-j hint (gap<=1.15x & "
              f"top1>=0.80): j={knee} "
              f"(frac={round(knee/L,3) if knee is not None else None})", flush=True)
        all_blocks.append(dict(num_ctx=num_ctx, ctx_tokens=num_ctx * C,
                               results=results, split_j_hint=knee))

    out = dict(
        model_path=model_path, model_family="hunyuan_v1_moe", L=L, V=V,
        dtype=args.dtype, chunk_size=args.chunk_size, query_len=args.query_len,
        num_docs=args.num_docs, j_list=j_list, ctx_list=ctx_list,
        data_path=data_path, blocks=all_blocks,
    )
    out_path = args.out if os.path.isabs(args.out) else os.path.join(PROJECT_ROOT, args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[a13b-jsweep] wrote {out_path}", flush=True)
    print("A13B_JSWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
