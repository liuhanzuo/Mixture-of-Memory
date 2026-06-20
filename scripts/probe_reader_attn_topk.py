#!/usr/bin/env python3
"""Reader-attn top-k HARD-ISOLATION go/no-go (the convergence experiment).

On the clean T2 task (needle MEMORIZE chunk known = chunk 0, 15 pg19 distractor
chunks), at the readout layer we:
  1. stream all n_ctx chunks into the rawkv store,
  2. score each chunk by the READER's native q.k salience (NOT gist),
  3. select top-k chunks, HARD-ISOLATE them (rebuild a store holding ONLY those
     chunks' raw-KV — physical exclusion, == Landmark take_along_dim gather, not
     a mask), then greedy-decode the 5-digit code.

Reports, for k in {1,2,3,5} plus oracle (only chunk0) and keep_all (all 16):
  * needle-in-topk hit rate (is chunk0 among the reader-attn top-k?)
  * end-to-end exact code recall (the W0-equivalent on this controlled task).

W0 upper bound = hit_rate x isolation_ceiling(~97.5% from Level1). This is the
decisive go/no-go: does reader-attn-topk + hard isolation break the dilution
wall (keep_all readout = 0%, Level1 needle-only = 97.5%).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from transformers import AutoTokenizer  # noqa: E402
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb  # noqa: E402
from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model  # noqa: E402


def _mem_layers(model):
    root = getattr(model, "module", model)
    return list(getattr(root, "_mem_space_layers", []))


def _readout_layer(model):
    for w in _mem_layers(model):
        if getattr(w, "_is_rawkv_readout_layer", False):
            return w
    # fallback: write owner / layer 16
    for w in _mem_layers(model):
        if getattr(w, "_layer_idx", None) == 16:
            return w
    return None


def _banks(model):
    root = getattr(model, "module", model)
    b = getattr(root, "_mem_space_shared_bank", None)
    if b is not None:
        return [b]
    return [getattr(w, "memory_bank", None) for w in _mem_layers(model)]


def _reset(model):
    for b in _banks(model):
        if b is None:
            continue
        if hasattr(b, "reset"):
            b.reset()
        object.__setattr__(b, "_rawkv_readout_store", None)
        b.frozen = False


def _freeze(model):
    for b in _banks(model):
        if b is not None:
            b.frozen = True


def _get_store(model):
    for b in _banks(model):
        s = getattr(b, "_rawkv_readout_store", None)
        if s is not None and s.size() > 0:
            return b, s
    return None, None


def _reader_attn_salience(ro, store, qh, device):
    """Per-chunk reader native q.k salience at the readout layer (no gist).
    Returns [C] numpy salience over chunk ids."""
    attn = ro.wrapped_layer.self_attn
    pre = getattr(ro.wrapped_layer, "input_layernorm", None)
    hd = attn.head_dim
    th = store.token_hidden                      # [1,M,d]
    tc = store.token_chunk[0].cpu().numpy()      # [M]
    C = int(tc.max()) + 1
    _h = pre(th) if pre is not None else th
    _q = pre(qh) if pre is not None else qh
    k = attn.k_proj(_h).view(1, th.shape[1], -1, hd).transpose(1, 2)   # [1,nkv,M,hd]
    q = attn.q_proj(_q).view(1, qh.shape[1], -1, hd).transpose(1, 2)   # [1,nh,Tq,hd]
    qv = q[0, :, -1, :]                                                # [nh,hd] last tok
    nh = qv.shape[0]; nkv = k.shape[1]
    Kr = k[0].repeat_interleave(nh // nkv, dim=0)                      # [nh,M,hd]
    aw = torch.einsum("hd,hmd->hm", qv.float(), Kr.float()) * (hd ** -0.5)
    aw = torch.softmax(aw, dim=-1).mean(0).cpu().numpy()              # [M]
    sal = np.zeros(C)
    for c in range(C):
        sal[c] = aw[tc == c].sum()
    return sal


def _restrict_store_to(store, keep_chunks, device):
    """Rebuild store tensors to hold ONLY tokens whose chunk in keep_chunks
    (HARD isolation: physical gather, others excluded)."""
    tc = store.token_chunk[0]
    mask = torch.zeros_like(tc, dtype=torch.bool)
    for c in keep_chunks:
        mask |= (tc == c)
    idx = mask.nonzero(as_tuple=False).squeeze(-1)
    store.token_hidden = store.token_hidden[:, idx, :].contiguous()
    store.token_pos = store.token_pos[:, idx].contiguous()
    # renumber kept chunks 0..k-1 so retrieve()'s kept=arange(C) matches
    # token_chunk, and trim gist_src to the kept chunks in the SAME order.
    keep_sorted = sorted(keep_chunks)
    old_tc = store.token_chunk[:, idx].contiguous()
    new_tc = torch.zeros_like(old_tc)
    for new_id, old_id in enumerate(keep_sorted):
        new_tc[old_tc == old_id] = new_id
    store.token_chunk = new_tc
    store.n_chunks = len(keep_sorted)
    if getattr(store, "gist_src", None) is not None:
        store.gist_src = store.gist_src[:, keep_sorted, :].contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_ctx", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    mc.l3_recon_max_positions = cli.chunk_size
    mc.rawkv_disable_col_bias = True
    mc.rawkv_readout_topk_chunks = 0        # keep_all at retrieve; we pre-restrict store
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()
    ro = _readout_layer(model)
    bg = np.load(cli.background)
    rng = random.Random(31)

    ks = [1, 2, 3, 5]
    modes = [f"reader_top{k}" for k in ks] + ["oracle_top1", "keep_all"]
    hit = {k: 0 for k in ks}            # needle (chunk0) in reader top-k
    exact = {m: 0 for m in modes}
    n = 0

    for i in range(cli.n_samples):
        name = "".join(rng.choices(string.ascii_uppercase, k=6))
        code = " ".join(rng.choices(string.digits, k=5))
        needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        q_text = f"The secret code for agent {name} is"
        ntok = tok.encode(" " + needle, add_special_tokens=False)
        bg0 = bg[(i + 900) % len(bg), :cli.chunk_size].tolist()
        chunk0 = (ntok + bg0[len(ntok):])[:cli.chunk_size]
        chunks = [torch.tensor([chunk0], device=device)]
        for c in range(1, cli.n_ctx):
            row = bg[(i + 900 + c) % len(bg), :cli.chunk_size].tolist()
            chunks.append(torch.tensor([row[:cli.chunk_size]], device=device))
        q_ids = torch.tensor([tok.encode(q_text, add_special_tokens=False)], device=device)
        gold = code.split()

        # capture query hidden at readout layer (for reader-attn salience) by a
        # question forward over the FULL streamed store.
        _reset(model)
        with torch.no_grad():
            for ch in chunks:
                model(input_ids=ch, use_cache=False)
        bank, store_full = _get_store(model)
        if store_full is None:
            continue
        # snapshot full store tensors so we can rebuild per mode.
        snap = (store_full.token_hidden.clone(), store_full.token_pos.clone(),
                store_full.token_chunk.clone(),
                None if store_full.gist_src is None else store_full.gist_src.clone())
        _freeze(model)
        cap = {}

        def _hook(mod, args, kwargs):
            hs = args[0] if args else kwargs.get("hidden_states")
            cap["h"] = hs.detach()
            return None
        h = ro.register_forward_pre_hook(_hook, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=q_ids, use_cache=False)
        h.remove()
        qh = cap["h"]
        sal = _reader_attn_salience(ro, store_full, qh, device)     # [C]
        order = list(np.argsort(-sal))                              # high->low
        for k in ks:
            hit[k] += int(0 in order[:k])

        def restore_full():
            store_full.token_hidden = snap[0].clone()
            store_full.token_pos = snap[1].clone()
            store_full.token_chunk = snap[2].clone()
            store_full.gist_src = None if snap[3] is None else snap[3].clone()

        def decode():
            cur = q_ids.clone(); gen = []
            with torch.no_grad():
                for _ in range(cli.max_new_tokens):
                    out = model(input_ids=cur, use_cache=False)
                    lg = out.logits if hasattr(out, "logits") else out[0]
                    nxt = int(lg[0, -1].argmax().item())
                    if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                        break
                    gen.append(nxt)
                    cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            od = [c for c in tok.decode(gen, skip_special_tokens=True) if c.isdigit()]
            return od[:5] == gold

        for k in ks:
            restore_full()
            _restrict_store_to(store_full, order[:k], device)
            if decode():
                exact[f"reader_top{k}"] += 1
        # oracle: only chunk0
        restore_full(); _restrict_store_to(store_full, [0], device)
        if decode():
            exact["oracle_top1"] += 1
        # keep_all
        restore_full()
        if decode():
            exact["keep_all"] += 1
        n += 1

    print("\n==== READER-ATTN TOP-K HARD-ISOLATION go/no-go ====")
    print(f"n={n} n_ctx={cli.n_ctx} (needle=chunk0, reader native q.k selection, gather isolation)")
    print("needle-in-topk hit rate (reader-attn):")
    for k in ks:
        print(f"  top{k}: {100.0*hit[k]/max(n,1):.1f}%")
    print("end-to-end exact code recall:")
    for m in modes:
        print(f"  {m:<14}: {100.0*exact[m]/max(n,1):.1f}%")
    print("\n(keep_all~0 reproduces dilution; oracle_top1~ceiling; reader_topK = the test)")


if __name__ == "__main__":
    main()
