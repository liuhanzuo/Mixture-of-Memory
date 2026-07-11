#!/usr/bin/env python
"""QCMem depth-partition **j-sweep** on the Hunyuan Hy3 (``hy_v3``) 80-layer MoE
backbone, sharded across the local 8 GPUs (2026-07-12).

Goal
----
Find the *split-j* = the "cacheable-semantic ceiling vs LM-tax" knee for QCMem on
Hy3. The parent ``QCMemModel`` runs the bottom ``layers[0:j]`` chunk-local at WRITE
(cached as ``h_j``) and resumes ``layers[j:]`` over the packed
``[sink ; ctx chunks ; query]`` at READ. As ``j`` grows:

  * more depth is spent chunk-local (NO cross-chunk / query attention),
  * fewer TOP layers remain to integrate the retrieved context at read,

so readout fidelity degrades. ``j=0`` == selective full re-forward (RAG upper
bound; numerically == ``full_forward`` per the self-test A1/A2, diff 0.0), ``j=L``
== closed-book. We locate the knee where fidelity starts falling sharply.

Metrics (per ``j``, averaged over ``num_docs`` real PG19 windows)
----------------------------------------------------------------
For each document window we build ONE packed token sequence
``[bos ; ctx_1..ctx_N ; query]`` (all context chunks selected — we isolate the
DEPTH-partition effect, no bm25 selector noise) and compute:

  * ``full_forward`` logits over the whole packed sequence  → the RAG/full-context
    reference (this is also what ``j=0`` READ reproduces exactly).
  * QCMem READ logits on the QUERY tail (``logits_tail=query_len``).

We then report, over the query span:

  * ``ppl``        : next-token perplexity of the QCMem readout (LM quality).
  * ``ppl_full``   : next-token perplexity of the full-context reference.
  * ``ppl_gap``    : ppl / ppl_full  (the multiplicative "LM tax" at this j).
  * ``kl``         : mean KL(full || qcmem) over the query positions (readout
                     fidelity to the full-context ideal, in nats).
  * ``top1``       : fraction of query positions where the readout's argmax
                     matches the full-context argmax.

The knee in ``ppl_gap`` / ``kl`` as a function of ``j`` is the split-j.

Efficiency
----------
The 597 GB Hy3 is loaded ONCE with ``device_map="auto"`` (sharded over the 8
L20A). Every forward pipelines across all 8 GPUs (the model does not fit on one),
so all 8 cards are exercised. We then loop over ``--j_list`` re-wrapping the SAME
loaded backbone in a fresh ``QCMemHy3Model(resume_j=j)`` (cheap — no reload). WRITE
is repeated per-j (``h_j`` depends on j); the full-context reference is computed
once per doc and shared across all j.

Usage
-----
    .venv_hy3/bin/python scripts/qcmem_hy3_jsweep.py \
        --model_path /apdcephfs_wzc1/.../models/Hy3 \
        --j_list 0,8,16,24,32,40,48 \
        --chunk_size 512 --num_ctx_chunks 6 --query_len 256 --num_docs 6 \
        --out logs/hy3_jsweep_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.memory.qcmem.qcmem_hy3 import QCMemHy3Model  # noqa: E402


# --------------------------------------------------------------------------- #
# data: sample real PG19 windows from the raw text jsonl (one book / line, but
# the file is effectively raw text — we just read byte slices and tokenize).
# --------------------------------------------------------------------------- #
def sample_pg19_windows(tokenizer, path, num_docs, window_tokens, byte_step):
    """Return a list of ``[1, window_tokens]`` LongTensors of real PG19 ids.

    We seek to ``i * byte_step`` in the raw text file, read a generous block,
    decode (ignoring partial multibyte at the edges), tokenize with NO special
    tokens, and take the first ``window_tokens`` ids. Distinct offsets give
    distinct passages without loading the whole 11 GB file.
    """
    windows = []
    # read ~6 chars/token * window * safety; PG19 is dense prose.
    block_bytes = max(window_tokens * 12, 40_000)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for i in range(num_docs):
            off = (i * byte_step) % max(size - block_bytes, 1)
            f.seek(off)
            raw = f.read(block_bytes)
            # drop a possibly-partial leading line so we start on a clean boundary
            nl = raw.find(b"\n")
            if 0 <= nl < len(raw) - 1:
                raw = raw[nl + 1:]
            text = raw.decode("utf-8", errors="ignore")
            ids = tokenizer(text, return_tensors="pt",
                            add_special_tokens=False).input_ids
            if ids.shape[1] < window_tokens:
                continue
            windows.append(ids[:, :window_tokens].long())
    return windows


# --------------------------------------------------------------------------- #
# metric helpers
# --------------------------------------------------------------------------- #
@torch.no_grad()
def span_nll(logits, target_ids):
    """Mean next-token NLL (nats) of ``target_ids`` under ``logits``.

    ``logits`` : ``[1, S, V]`` predicting positions ``0..S-1``.
    ``target_ids`` : ``[1, S]`` the tokens sitting at those positions.
    We score ``logits[:, :-1]`` against ``target_ids[:, 1:]`` (next-token).
    Returns (mean_nll, n_tokens) as python floats.
    """
    lg = logits[:, :-1, :].float()
    tg = target_ids[:, 1:].to(lg.device)
    logp = torch.log_softmax(lg, dim=-1)
    nll = -logp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)  # [1, S-1]
    return nll.mean().item(), int(nll.numel())


@torch.no_grad()
def span_kl_top1(qc_logits, full_logits):
    """Mean KL(full || qcmem) in nats and top-1 argmax agreement over the span.

    Both ``[1, S, V]``, aligned position-for-position on the query tail.
    """
    q = qc_logits.float()
    fu = full_logits.float().to(q.device)
    logp_q = torch.log_softmax(q, dim=-1)
    logp_f = torch.log_softmax(fu, dim=-1)
    p_f = logp_f.exp()
    kl = (p_f * (logp_f - logp_q)).sum(-1)          # [1, S]
    top1 = (q.argmax(-1) == fu.argmax(-1)).float()  # [1, S]
    return kl.mean().item(), top1.mean().item()


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QCMem Hy3 depth-partition j-sweep")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--j_list", type=str, default="0,8,16,24,32,40,48")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--num_ctx_chunks", type=int, default=6)
    ap.add_argument("--num_ctx_list", type=str, default="",
                    help="comma list of num_ctx_chunks to sweep (context-length "
                         "generalisation of split-j); overrides --num_ctx_chunks "
                         "when set, e.g. '6,12,24' -> ~3k/6k/12k packed.")
    ap.add_argument("--query_len", type=int, default=256)
    ap.add_argument("--num_docs", type=int, default=6)
    ap.add_argument("--byte_step", type=int, default=3_000_000)
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--out", type=str, default="logs/hy3_jsweep_results.json")
    args = ap.parse_args()

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    j_list = [int(x) for x in args.j_list.split(",") if x.strip() != ""]

    from transformers import AutoModelForCausalLM, AutoTokenizer
    print(f"[jsweep] loading tokenizer {args.model_path}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)

    t0 = time.time()
    print(f"[jsweep] loading Hy3 device_map=auto dtype={dtype} attn={args.attn_impl}",
          flush=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation=args.attn_impl, low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    dm = getattr(model, "hf_device_map", None)
    if dm is not None:
        devs = sorted({str(v) for v in dm.values()})
        print(f"[jsweep] hf_device_map spans {len(devs)} device(s): {devs}", flush=True)
    L = int(model.config.num_hidden_layers)
    V = int(model.config.vocab_size)
    print(f"[jsweep] loaded in {time.time()-t0:.0f}s | L={L} V={V}", flush=True)

    j_list = [j for j in j_list if 0 <= j <= L]
    print(f"[jsweep] j_list={j_list}", flush=True)

    ctx_list = ([int(x) for x in args.num_ctx_list.split(",") if x.strip()]
                if args.num_ctx_list.strip() else [args.num_ctx_chunks])
    print(f"[jsweep] num_ctx_list={ctx_list} (chunk_size={args.chunk_size})", flush=True)

    bos_id = tok.bos_token_id if tok.bos_token_id is not None else 1
    embed_dev = next(model.model.embed_tokens.parameters()).device
    bos = torch.tensor([[bos_id]], device=embed_dev, dtype=torch.long)
    # reuse one resume_j=0 wrapper for the shared full-context reference
    qc0 = QCMemHy3Model(model, resume_j=0)
    # cache one QCMem wrapper per j (cheap — shares the loaded backbone)
    qc_by_j = {j: QCMemHy3Model(model, resume_j=j) for j in j_list}

    all_blocks = []  # one entry per context length

    for num_ctx in ctx_list:
        C = args.chunk_size
        window_tokens = num_ctx * C + args.query_len
        windows = sample_pg19_windows(
            tok, os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl"),
            args.num_docs, window_tokens, args.byte_step,
        )
        print(f"\n[jsweep] === ctx={num_ctx}x{C}={num_ctx*C} tok + q{args.query_len} "
              f"({len(windows)} docs) ===", flush=True)

        # full-context reference (query tail) per doc, shared across all j
        docs = []
        for di, ids in enumerate(windows):
            ids = ids.to(embed_dev)
            ctx = [ids[:, k * C:(k + 1) * C] for k in range(num_ctx)]
            query = ids[:, num_ctx * C:]
            packed_ids = torch.cat([bos] + ctx + [query], dim=1)
            full_logits = qc0.full_forward_logits(packed_ids)
            full_tail = full_logits[:, -args.query_len:, :].detach().float().cpu()
            docs.append(dict(ctx=ctx, query=query, full_tail=full_tail))
        print(f"[jsweep] computed full-context reference for {len(docs)} docs "
              f"(packed ~{num_ctx*C + args.query_len + 1} tok)", flush=True)

        results = []
        for j in j_list:
            qc = qc_by_j[j]
            # release cached activations from the previous j so the long-context
            # (8k/16k) reads don't accumulate fragmentation across the sweep — a
            # sharded 597 GB model has little headroom per shard.
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
            print(f"[jsweep] ctx={num_ctx:>2} j={j:>3} (frac={rec['frac']:.3f})  "
                  f"ppl={rec['ppl']:.3f} ppl_full={rec['ppl_full']:.3f} "
                  f"gap={rec['ppl_gap']:.3f}x KL={rec['kl_nats']:.4f} "
                  f"top1={rec['top1']:.3f} ({rec['secs']:.0f}s)", flush=True)

        # per-ctx report + knee
        print("-" * 78, flush=True)
        print(f"ctx={num_ctx*C} tok | {'j':>4} {'frac':>6} {'ppl':>8} "
              f"{'gap':>7} {'KL':>8} {'top1':>7}", flush=True)
        for r in results:
            print(f"           | {r['j']:>4} {r['frac']:>6.3f} {r['ppl']:>8.3f} "
                  f"{r['ppl_gap']:>6.3f}x {r['kl_nats']:>8.4f} {r['top1']:>7.3f}",
                  flush=True)
        faithful = [r for r in results if r["ppl_gap"] <= 1.15 and r["top1"] >= 0.80]
        knee = max((r["j"] for r in faithful), default=None)
        print(f"[jsweep] ctx={num_ctx*C}: split-j hint (gap<=1.15x & top1>=0.80): "
              f"j={knee} (frac={round(knee/L,3) if knee is not None else None})",
              flush=True)
        all_blocks.append(dict(num_ctx=num_ctx, ctx_tokens=num_ctx * C,
                               results=results, split_j_hint=knee))

    out = dict(
        model_path=args.model_path, L=L, V=V, dtype=args.dtype,
        chunk_size=args.chunk_size, query_len=args.query_len,
        num_docs=args.num_docs, j_list=j_list, ctx_list=ctx_list,
        blocks=all_blocks,
    )
    os.makedirs(os.path.dirname(os.path.join(PROJECT_ROOT, args.out)), exist_ok=True)
    with open(os.path.join(PROJECT_ROOT, args.out), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[jsweep] wrote {args.out}", flush=True)
    print("HY3_JSWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
