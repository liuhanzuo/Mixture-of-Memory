#!/usr/bin/env python3
"""Intrinsic QCMem split-depth sweep for dense Qwen/Llama backbones.

Loads one stock causal LM once, builds natural-text windows from a local
SlimPajama parquet, and compares chunk-local depth-j caching plus layers[j:]
readout against the full-context forward on the query tail.  All context chunks
are selected, so this isolates the layer split from retrieval quality.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.memory.qcmem import QCMemModel  # noqa: E402


def parse_ints(value: str):
    return [int(x) for x in value.split(",") if x.strip()]


def load_windows(parquet_path, tokenizer, num_docs, window_tokens, seed):
    frame = pd.read_parquet(parquet_path, columns=["text"])
    frame = frame[frame["text"].map(lambda x: isinstance(x, str) and len(x) > 0)]
    frame = frame.sample(frac=1.0, random_state=seed)
    windows = []
    carry = []
    for text in frame["text"]:
        carry.extend(tokenizer.encode(text, add_special_tokens=False))
        while len(carry) >= window_tokens and len(windows) < num_docs:
            windows.append(torch.tensor(carry[:window_tokens], dtype=torch.long).unsqueeze(0))
            carry = carry[window_tokens:]
        if len(windows) >= num_docs:
            break
    if len(windows) != num_docs:
        raise RuntimeError(
            f"only built {len(windows)}/{num_docs} windows of {window_tokens} tokens"
        )
    return windows


@torch.no_grad()
def next_token_nll(logits, target_ids):
    logits = logits[:, :-1].float()
    target = target_ids[:, 1:].to(logits.device)
    nll = -torch.log_softmax(logits, dim=-1).gather(
        -1, target.unsqueeze(-1)
    ).squeeze(-1)
    return nll.mean().item(), nll.numel()


@torch.no_grad()
def distribution_metrics(qc_logits, full_logits):
    qlogp = torch.log_softmax(qc_logits.float(), dim=-1)
    flogp = torch.log_softmax(full_logits.float().to(qlogp.device), dim=-1)
    fp = flogp.exp()
    kl = (fp * (flogp - qlogp)).sum(-1).mean().item()
    top1 = (qc_logits.argmax(-1) == full_logits.to(qc_logits.device).argmax(-1))
    return kl, top1.float().mean().item()


def main():
    ap = argparse.ArgumentParser(description="Dense-Qwen intrinsic QCMem j-sweep")
    ap.add_argument("--model_path", default="models/Qwen3-32B")
    ap.add_argument(
        "--data_path",
        default="data/slimpajama-6b/data/validation-00000-of-00001-4fb685c22a3f91ef.parquet",
    )
    ap.add_argument("--j_list", default="0,8,16,20,24,28,32,40")
    ap.add_argument("--chunk_size", type=int, default=256)
    ap.add_argument("--num_ctx_list", default="3,7")
    ap.add_argument("--query_len", type=int, default=256)
    ap.add_argument("--num_docs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")
    ap.add_argument("--attn_impl", default="sdpa")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--self_test", action="store_true")
    ap.add_argument("--out", default="logs/qwen3_32b_jsweep.json")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = args.model_path if os.path.isabs(args.model_path) else os.path.join(ROOT, args.model_path)
    data_path = args.data_path if os.path.isabs(args.data_path) else os.path.join(ROOT, args.data_path)
    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[args.dtype]
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    print(f"[qwen-jsweep] loading {model_path} dtype={dtype} device={device}", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=dtype,
        trust_remote_code=True,
        local_files_only=True,
        attn_implementation=args.attn_impl,
    ).to(device).eval()
    model.config.use_cache = False
    L = int(model.config.num_hidden_layers)
    print(
        f"[qwen-jsweep] loaded in {time.time()-t0:.1f}s L={L} "
        f"peak_gib={torch.cuda.max_memory_allocated(device)/2**30:.2f}",
        flush=True,
    )

    if args.self_test:
        ids = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]], device=device)
        ref = model(input_ids=ids, use_cache=False).logits
        diffs = {}
        for j in sorted({0, min(1, L), L // 2, L}):
            out = QCMemModel(model, resume_j=j).resume_forward_ids(ids)
            diffs[str(j)] = float((out.float() - ref.float()).abs().max().item())
        print(f"[qwen-jsweep] SELF_TEST diffs={diffs}", flush=True)
        if max(diffs.values()) > (2e-2 if dtype != torch.float32 else 1e-4):
            raise SystemExit("SELF_TEST_FAILED")
        print("QWEN_JSWEEP_SELF_TEST_OK", flush=True)
        return

    j_list = [j for j in parse_ints(args.j_list) if 0 <= j <= L]
    ctx_list = parse_ints(args.num_ctx_list)
    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = model.config.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    bos = torch.tensor([[int(bos_id)]], dtype=torch.long, device=device)
    qc0 = QCMemModel(model, resume_j=0)
    qc_by_j = {j: QCMemModel(model, resume_j=j) for j in j_list}
    blocks = []

    for num_ctx in ctx_list:
        needed = num_ctx * args.chunk_size + args.query_len
        windows = load_windows(data_path, tokenizer, args.num_docs, needed, args.seed)
        docs = []
        for ids in windows:
            ids = ids.to(device)
            ctx = [
                ids[:, k * args.chunk_size:(k + 1) * args.chunk_size]
                for k in range(num_ctx)
            ]
            query = ids[:, num_ctx * args.chunk_size:]
            packed = torch.cat([bos] + ctx + [query], dim=1)
            full_tail = qc0.full_forward_logits(packed)[:, -args.query_len:].detach().cpu()
            docs.append((ctx, query, full_tail))
        results = []
        print(f"[qwen-jsweep] ctx={num_ctx}x{args.chunk_size} docs={len(docs)}", flush=True)

        for j in j_list:
            qc = qc_by_j[j]
            nll_sum = full_nll_sum = token_sum = 0.0
            kl_sum = top1_sum = 0.0
            tj = time.time()
            for ctx, query, full_tail in docs:
                sink_h = qc.write_chunk(bos)
                ctx_h = [qc.write_chunk(chunk) for chunk in ctx]
                query_h = qc.write_chunk(query)
                logits = qc.read_core(
                    sink_h, ctx_h, query_h, logits_tail=args.query_len
                ).detach()
                nll, ntok = next_token_nll(logits, query)
                full_nll, _ = next_token_nll(full_tail, query)
                kl, top1 = distribution_metrics(logits, full_tail)
                nll_sum += nll * ntok
                full_nll_sum += full_nll * ntok
                token_sum += ntok
                kl_sum += kl
                top1_sum += top1
            mean_nll = nll_sum / token_sum
            mean_full = full_nll_sum / token_sum
            rec = {
                "j": j,
                "frac": round(j / L, 4),
                "ppl": round(math.exp(mean_nll), 5),
                "ppl_full": round(math.exp(mean_full), 5),
                "ppl_gap": round(math.exp(mean_nll - mean_full), 5),
                "kl_nats": round(kl_sum / len(docs), 6),
                "top1": round(top1_sum / len(docs), 5),
                "secs": round(time.time() - tj, 2),
            }
            results.append(rec)
            print(f"[qwen-jsweep] {rec}", flush=True)
            torch.cuda.empty_cache()
        faithful = [r for r in results if r["ppl_gap"] <= 1.15 and r["top1"] >= 0.80]
        knee = max((r["j"] for r in faithful), default=None)
        blocks.append({"num_ctx": num_ctx, "ctx_tokens": num_ctx * args.chunk_size,
                       "split_j_hint": knee, "results": results})
        print(f"[qwen-jsweep] ctx_tokens={num_ctx*args.chunk_size} knee={knee}", flush=True)

    payload = {
        "model_path": model_path,
        "L": L,
        "dtype": args.dtype,
        "chunk_size": args.chunk_size,
        "query_len": args.query_len,
        "num_docs": args.num_docs,
        "j_list": j_list,
        "blocks": blocks,
    }
    out = args.out if os.path.isabs(args.out) else os.path.join(ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"QWEN_JSWEEP_DONE {out}", flush=True)


if __name__ == "__main__":
    main()
