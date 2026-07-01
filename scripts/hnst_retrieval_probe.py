#!/usr/bin/env python
"""HNST leak-immune retrieval-precision probe (2026-06-25).

WHY this exists: the only mem_space FIFO checkpoint that actually *answers* qa5
via the FIFO-hidden attend path (b50) was trained with babilong_mix_fraction=0.15
(eval-split leak -> voided). The clean (mix=0) checkpoints were trained with
inject_gate_bias_init=-2.0 (FIFO injection ~off) and never learned to answer via
FIFO-hidden, so their generation is garbage. Generation accuracy is therefore
confounded and cannot be the decisive HNST metric.

This probe removes the confound: it measures the HNST premise DIRECTLY, on the
BASE Llama-3-8B (no adapter, no training, provably never saw babilong), as a
RETRIEVAL question: given the frozen reader's own q.k salience, does the tree
beam-descent SELECT the chunk that actually contains the needle? And crucially,
does it reach EARLY needles that a b25 FIFO structurally EVICTS?

Design (faithful to _fifo_select_keep_set_tree / _reader_attn_keep_set):
  * Stream each chunk (<=chunk_size tokens) through base Llama independently and
    grab layer-L input hidden (hidden_states[L]) -> the FIFO "stored hidden".
  * Leaf summary = max-pool chunk hidden over tokens (H1 anti-dilution).
  * query = last token of the QUESTION chunk's layer-L hidden, via layer-L
    self_attn.q_proj + RoPE (reader-native probe).
  * node key = k_proj(pre-attn-norm(summary)) (content match, no RoPE).
  * sal = mean_batch(amax_head(q.k / sqrt(hd))).
Arms (all pick a kept set of chunk indices; needle = doc-abs chunk of gold ans):
  * tree   : B-ary max-pool tree, beam-b descent over ALL chunks -> ~topk leaves.
  * flat   : flat top-k q.k over ALL chunks (the existing reader_attn selector).
  * b25    : the last `buffer` chunks only (FIFO amnesia) -> needle EVICTED if its
             chunk index < n_ctx_chunks - buffer.
Metric: needle-recall = P(needle chunk in kept set), stratified early/mid/late by
min(needle_chunk)/(n_ctx_chunks-1). Reported per (length, arm, bucket).
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))

from transformers import AutoTokenizer, LlamaForCausalLM
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

# reuse the harness's dataset loader + needle locator + prompt formatting
import scripts.run_babilong_mem_space as H


def _layer_L(model):
    return model.model.layers


@torch.no_grad()
def _chunk_hidden(model, chunk_ids, L, device):
    """Layer-L input hidden of a chunk forwarded alone: [1, T, d]."""
    out = model(input_ids=chunk_ids.unsqueeze(0).to(device),
                use_cache=False, output_hidden_states=True)
    return out.hidden_states[L]


@torch.no_grad()
def _query_probe(model, q_hidden, L, device):
    """Last-token RoPE'd q_proj probe from the question chunk's layer-L hidden.
    Returns qv [1, nh, hd]."""
    layer = _layer_L(model)[L]
    attn = layer.self_attn
    pre = layer.input_layernorm
    hd = attn.head_dim
    hs = pre(q_hidden)
    B, T, d = hs.shape
    q = attn.q_proj(hs).view(B, T, -1, hd).transpose(1, 2)   # [1,nh,T,hd]
    rot = model.model.rotary_emb
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = rot(q_hidden, pos)
    q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q_r[:, :, -1, :]                                   # [1,nh,hd]


@torch.no_grad()
def _score(model, summ, qv, L, device):
    """summ [N,1,d] node summaries -> sal [N]."""
    layer = _layer_L(model)[L]
    attn = layer.self_attn
    pre = layer.input_layernorm
    hd = attn.head_dim
    N = summ.shape[0]
    s = pre(summ.to(device, dtype=qv.dtype))
    kk = attn.k_proj(s).view(N, 1, -1, hd).permute(1, 2, 0, 3)  # [1,nkv,N,hd]
    nh = qv.shape[1]; nkv = kk.shape[1]
    if nh != nkv:
        kk = kk.repeat_interleave(nh // nkv, dim=1)
    aw = torch.einsum("bhd,bhnd->bhn", qv.float(), kk.float()) * (hd ** -0.5)
    return aw.amax(dim=1).mean(dim=0)                          # [N]


def _tree_select(model, leaves, qv, L, device, B_, beam, topk):
    """leaves: list of [1,d] summaries. Returns set of selected leaf indices."""
    C = len(leaves)
    if C <= topk:
        return set(range(C))
    cur = torch.stack(leaves, dim=0)                          # [C,1,d]
    levels = [cur]
    while cur.shape[0] > 1:
        n = cur.shape[0]; parts = []
        for g in range((n + B_ - 1) // B_):
            parts.append(cur[g*B_:min(n,(g+1)*B_)].amax(dim=0))
        cur = torch.stack(parts, dim=0); levels.append(cur)
    frontier = [0]; lvl = len(levels) - 1
    while lvl > 0:
        cl = lvl - 1; nch = levels[cl].shape[0]
        children = []
        for j in frontier:
            children += list(range(j*B_, min(nch, (j+1)*B_)))
        if not children: break
        idx = torch.tensor(children, device=device)
        sal = _score(model, levels[cl][idx], qv, L, device)
        kk = min(topk if cl == 0 else beam, len(children))
        order = torch.topk(sal, k=kk, dim=0).indices.tolist()
        frontier = [children[o] for o in order]; lvl = cl
    return set(int(i) for i in frontier)


def _flat_select(model, leaves, qv, L, device, topk):
    C = len(leaves)
    if C <= topk:
        return set(range(C))
    summ = torch.stack(leaves, dim=0)
    sal = _score(model, summ, qv, L, device)
    return set(int(i) for i in torch.topk(sal, k=min(topk, C), dim=0).indices.tolist())


def _bucket(nmin, n_ctx):
    frac = nmin / max(1, n_ctx - 1)
    return "early" if frac < 1/3 else ("mid" if frac < 2/3 else "late")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(ROOT / "models" / "Meta-Llama-3-8B"))
    ap.add_argument("--task", default="qa5")
    ap.add_argument("--lengths", nargs="+", default=["16k", "32k"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--select_layer", type=int, default=16)
    ap.add_argument("--branch", type=int, default=8)
    ap.add_argument("--beam", type=int, default=3)
    ap.add_argument("--topk", type=int, default=25)
    ap.add_argument("--buffer", type=int, default=25, help="b25 FIFO retained chunks")
    ap.add_argument("--out", required=True, help="jsonl output path")
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
    ).to(dev).eval()
    L = args.select_layer

    prompts = H.DEFAULT_PROMPTS[args.task]
    tmpl = H.DEFAULT_TEMPLATE
    outf = open(args.out, "w")

    for length in args.lengths:
        data = H.load_babilong_dataset("RMT-team/babilong", length)[args.task]
        n = len(data) if args.limit <= 0 else min(len(data), args.limit)
        idxs = list(range(n))[args.shard_index::args.num_shards]
        for i in idxs:
            s = data[i]
            text = H.get_formatted_input(s["input"], s["question"], prompts["examples"],
                                         prompts["instruction"], prompts["post_prompt"],
                                         template=tmpl)
            ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")[0]
            needle = H._locate_needle_chunks(ids.unsqueeze(0), s["target"], tok, args.chunk_size)
            chunks = list(ids.split(args.chunk_size))
            n_all = len(chunks)
            n_ctx = n_all - 1                       # last chunk = question
            if n_ctx < 2 or not needle:
                continue
            nmin = min(needle)
            if nmin >= n_ctx:                       # needle only in question chunk
                continue
            # per-chunk leaf summaries (context chunks only) at layer L
            leaves = []
            for c in range(n_ctx):
                h = _chunk_hidden(model, chunks[c], L, dev)     # [1,T,d]
                leaves.append(h[0].amax(dim=0, keepdim=False).unsqueeze(0))  # [1,d]
            qh = _chunk_hidden(model, chunks[-1], L, dev)
            qv = _query_probe(model, qh, L, dev)
            tree_sel = _tree_select(model, leaves, qv, L, dev, args.branch, args.beam, args.topk)
            flat_sel = _flat_select(model, leaves, qv, L, dev, args.topk)
            b25_sel = set(range(max(0, n_ctx - args.buffer), n_ctx))  # retained window
            b = _bucket(nmin, n_ctx)
            hit = lambda sset: int(any(nc in sset for nc in needle if nc < n_ctx))
            rec = {"length": length, "bucket": b, "n_ctx": n_ctx,
                   "needle_min": nmin,
                   "tree": hit(tree_sel), "flat": hit(flat_sel), "b25": hit(b25_sel),
                   "tree_k": len(tree_sel), "flat_k": len(flat_sel), "b25_k": len(b25_sel)}
            outf.write(json.dumps(rec) + "\n"); outf.flush()
    outf.close()
    print("PROBE_DONE", args.out)


if __name__ == "__main__":
    main()
