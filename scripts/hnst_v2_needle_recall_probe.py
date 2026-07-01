#!/usr/bin/env python
"""HNST v2 needle-recall probe (2026-06-25) — trained tree vs v1 max-pool vs flat.

Extends ``hnst_retrieval_probe.py`` to load a TRAINED mem_space checkpoint (full
model + trainable tree pool) and measure needle-recall for the arms that decide
whether the SELECTION WALL is broken:

  * v2tree : trainable-tree beam descent (uses model._tree_pool learned leaf/node
             aggregation + the TRAINED reader's layer-L q_proj/k_proj).
  * v1tree : SAME beam descent but with fixed MAX-POOL aggregation (the KILLED
             v1 baseline) using the trained reader — isolates the value of the
             learned aggregation.
  * flat   : flat top-k reader-attn over ALL chunk leaves (trained reader).
  * b25    : the last ``buffer`` chunks only (FIFO amnesia).

Needle-recall = P(a needle chunk is in the kept set), stratified early/mid/late.
Leak-immune: reads BABILong test docs ONLY to LOCATE the needle chunk (never as a
training signal), scores with the reader's own q.k salience. Official metric is a
RETRIEVAL question (does selection surface the needle chunk), not generation.

Usage:
  python scripts/hnst_v2_needle_recall_probe.py \
    --adapter_config outputs/RUN/adapter_config.json \
    --checkpoint outputs/RUN/full_model_stepNNNNNN.pt \
    --task qa5 --lengths 8k 16k 32k --limit 100 --branch 4 --beam 2 --topk 8 \
    --out logs/hnstv2_recall.jsonl
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "third_party" / "babilong-pkg"))

from transformers import AutoTokenizer
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

import scripts.run_babilong_mem_space as H


def _layers(model):
    root = getattr(model, "module", model)
    return root.model.layers


@torch.no_grad()
def _chunk_hidden(model, chunk_ids, L, device):
    out = model(input_ids=chunk_ids.unsqueeze(0).to(device),
                use_cache=False, output_hidden_states=True)
    return out.hidden_states[L]                       # [1, T, d]


@torch.no_grad()
def _query_probe(model, q_hidden, L, device):
    layer = _layers(model)[L]
    attn = layer.wrapped_layer.self_attn
    pre = layer.wrapped_layer.input_layernorm
    hd = attn.head_dim
    hs = pre(q_hidden)
    B, T, d = hs.shape
    q = attn.q_proj(hs).view(B, T, -1, hd).transpose(1, 2)
    root = getattr(model, "module", model)
    rot = root.model.rotary_emb
    pos = torch.arange(T, device=device).unsqueeze(0)
    cos, sin = rot(q_hidden, pos)
    q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
    return q_r[:, :, -1, :]                           # [1, nh, hd]


@torch.no_grad()
def _score(model, summ, qv, L, device):
    """summ [N,1,d] node summaries -> sal [N]."""
    layer = _layers(model)[L]
    attn = layer.wrapped_layer.self_attn
    pre = layer.wrapped_layer.input_layernorm
    hd = attn.head_dim
    N = summ.shape[0]
    s = pre(summ.to(device, dtype=qv.dtype))
    kk = attn.k_proj(s).view(N, 1, -1, hd).permute(1, 2, 0, 3)   # [1,nkv,N,hd]
    nh = qv.shape[1]; nkv = kk.shape[1]
    if nh != nkv:
        kk = kk.repeat_interleave(nh // nkv, dim=1)
    aw = torch.einsum("bhd,bhnd->bhn", qv.float(), kk.float()) * (hd ** -0.5)
    return aw.amax(dim=1).mean(dim=0)                            # [N]


@torch.no_grad()
def _build_levels(tree_pool, leaves, branch, mode):
    """leaves [C,1,d] -> list of levels. mode in {'v2','v1'}."""
    if mode == "v2" and tree_pool is not None:
        return tree_pool.build_levels(leaves, branch)
    # v1 max-pool
    levels = [leaves]; cur = leaves
    while cur.shape[0] > 1:
        n = cur.shape[0]; parts = []
        for g in range((n + branch - 1) // branch):
            parts.append(cur[g * branch:min(n, (g + 1) * branch)].amax(dim=0))
        cur = torch.stack(parts, dim=0); levels.append(cur)
    return levels


@torch.no_grad()
def _tree_select(model, tree_pool, leaves, qv, L, device, branch, beam, topk, mode):
    C = len(leaves)
    if C <= topk:
        return set(range(C))
    cur = torch.stack(leaves, dim=0)                            # [C,1,d]
    levels = _build_levels(tree_pool, cur, branch, mode)
    frontier = [0]; lvl = len(levels) - 1
    while lvl > 0:
        cl = lvl - 1; nch = levels[cl].shape[0]
        children = []
        for j in frontier:
            children += list(range(j * branch, min(nch, (j + 1) * branch)))
        if not children:
            break
        idx = torch.tensor(children, device=device)
        sal = _score(model, levels[cl][idx], qv, L, device)
        kk = min(topk if cl == 0 else beam, len(children))
        order = torch.topk(sal, k=kk, dim=0).indices.tolist()
        frontier = [children[o] for o in order]; lvl = cl
    return set(int(i) for i in frontier)


@torch.no_grad()
def _flat_select(model, leaves, qv, L, device, topk):
    C = len(leaves)
    if C <= topk:
        return set(range(C))
    summ = torch.stack(leaves, dim=0)
    sal = _score(model, summ, qv, L, device)
    return set(int(i) for i in torch.topk(sal, k=min(topk, C), dim=0).indices.tolist())


def _bucket(nmin, n_ctx):
    frac = nmin / max(1, n_ctx - 1)
    return "early" if frac < 1 / 3 else ("mid" if frac < 2 / 3 else "late")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=str(ROOT / "models" / "Meta-Llama-3-8B"))
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--task", default="qa5")
    ap.add_argument("--lengths", nargs="+", default=["8k", "16k", "32k"])
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--select_layer", type=int, default=16)
    ap.add_argument("--branch", type=int, default=4)
    ap.add_argument("--beam", type=int, default=2)
    ap.add_argument("--topk", type=int, default=8)
    ap.add_argument("--buffer", type=int, default=25)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(args.model_path)
    cfg = json.load(open(args.adapter_config))
    mc = H.build_mem_space_config(cfg)
    mc.gradient_checkpointing = False
    model = H.load_mem_space_model(args.model_path, args.checkpoint, mc, dev,
                                   torch.bfloat16, "sdpa")
    model.eval()
    root = getattr(model, "module", model)
    tree_pool = getattr(root, "_tree_pool", None)
    print("tree_pool present:", tree_pool is not None, flush=True)
    L = args.select_layer

    prompts = H.DEFAULT_PROMPTS[args.task]
    tmpl = H.DEFAULT_TEMPLATE
    outf = open(args.out, "w")

    for length in args.lengths:
        data = H.load_babilong_dataset("RMT-team/babilong", length)[args.task]
        n = len(data) if args.limit <= 0 else min(len(data), args.limit)
        for i in range(n):
            if (i % args.num_shards) != args.shard_index:
                continue
            s = data[i]
            text = H.get_formatted_input(s["input"], s["question"], prompts["examples"],
                                         prompts["instruction"], prompts["post_prompt"],
                                         template=tmpl)
            ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")[0]
            needle = H._locate_needle_chunks(ids.unsqueeze(0), s["target"], tok,
                                             args.chunk_size)
            chunks = list(ids.split(args.chunk_size))
            n_ctx = len(chunks) - 1
            if n_ctx < 2 or not needle:
                continue
            nmin = min(needle)
            if nmin >= n_ctx:
                continue
            leaves = []
            raw_leaves = []
            for c in range(n_ctx):
                h = _chunk_hidden(model, chunks[c], L, dev)      # [1,T,d] (ONE forward)
                raw_leaves.append(h[0].amax(dim=0, keepdim=False).unsqueeze(0))
                if tree_pool is not None:
                    leaves.append(tree_pool.pool_leaf(h))        # [1,d]
                else:
                    leaves.append(raw_leaves[-1])
            qh = _chunk_hidden(model, chunks[-1], L, dev)
            qv = _query_probe(model, qh, L, dev)
            v2_sel = _tree_select(model, tree_pool, leaves, qv, L, dev,
                                  args.branch, args.beam, args.topk, "v2")
            # v1 max-pool baseline: raw max-pool leaves (same trained reader q.k).
            v1_sel = _tree_select(model, None, raw_leaves, qv, L, dev,
                                  args.branch, args.beam, args.topk, "v1")
            flat_sel = _flat_select(model, leaves, qv, L, dev, args.topk)
            b25_sel = set(range(max(0, n_ctx - args.buffer), n_ctx))
            b = _bucket(nmin, n_ctx)
            hit = lambda sset: int(any(nc in sset for nc in needle if nc < n_ctx))
            rec = {"length": length, "bucket": b, "n_ctx": n_ctx, "needle_min": nmin,
                   "v2tree": hit(v2_sel), "v1tree": hit(v1_sel),
                   "flat": hit(flat_sel), "b25": hit(b25_sel),
                   "v2_k": len(v2_sel), "flat_k": len(flat_sel)}
            outf.write(json.dumps(rec) + "\n"); outf.flush()
    outf.close()
    print("PROBE_DONE", args.out)


if __name__ == "__main__":
    main()
