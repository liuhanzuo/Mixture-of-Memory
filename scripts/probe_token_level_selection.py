#!/usr/bin/env python3
"""Token-level selection probe — the decisive 'fixable vs paradigm-limit' test.

Chunk-level isolation broke the wall partially (chunk64 oracle 95%, reader 67.5%)
but within-chunk dilution remains (a 512-chunk's 25-token needle drowns among 487
bg tokens in one flat softmax — we lack Landmark's 2nd-stage within-block token
attention). This probe tests TOKEN-level isolation:

  tiers (all gather-isolate, vary the selection source/granularity):
    keep_all          : all 16 chunks' tokens (dilution baseline ~0%)
    token_oracle      : gather ONLY the needle's ~S tokens (true isolation ceiling
                        — replaces the artifactual Level1 97.5%)
    token_reader_topN : reader-attn per-TOKEN salience, keep top-N tokens
                        (N in {32,64,128}) across all 8192, gather those

Reports token-in-topN hit (are the needle's digit tokens among the kept top-N?)
+ end-to-end exact code recall per tier.

  token_oracle ~high + token_reader climbs -> token-granularity isolation is the
    fix (raw-KV paradigm viable; selection at token level).
  token_oracle also low -> even clean needle tokens don't read out -> deeper issue.
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
from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model  # noqa: E402


def _mem_layers(model):
    root = getattr(model, "module", model)
    return list(getattr(root, "_mem_space_layers", []))


def _readout_layer(model):
    for w in _mem_layers(model):
        if getattr(w, "_is_rawkv_readout_layer", False):
            return w
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
            return s
    return None


def _per_token_salience(ro, store, qh):
    """Reader native q.k per-TOKEN salience over the store. Returns aw [M] np."""
    attn = ro.wrapped_layer.self_attn
    pre = getattr(ro.wrapped_layer, "input_layernorm", None)
    hd = attn.head_dim
    th = store.token_hidden
    _h = pre(th) if pre is not None else th
    _q = pre(qh) if pre is not None else qh
    k = attn.k_proj(_h).view(1, th.shape[1], -1, hd).transpose(1, 2)
    q = attn.q_proj(_q).view(1, qh.shape[1], -1, hd).transpose(1, 2)
    qv = q[0, :, -1, :]
    nh = qv.shape[0]; nkv = k.shape[1]
    Kr = k[0].repeat_interleave(nh // nkv, dim=0)
    aw = torch.einsum("hd,hmd->hm", qv.float(), Kr.float()) * (hd ** -0.5)
    aw = torch.softmax(aw, dim=-1).mean(0).cpu().numpy()
    return aw


def _restrict_to_tokens(store, keep_idx, device):
    """HARD-isolate: keep only the given flat token indices (gather)."""
    idx = torch.as_tensor(sorted(keep_idx), device=device, dtype=torch.long)
    d = store.token_hidden.shape[-1]
    store.token_hidden = store.token_hidden[:, idx, :].contiguous()
    store.token_pos = store.token_pos[:, idx].contiguous()
    store.token_chunk = torch.zeros_like(store.token_pos)  # single pseudo-chunk
    store.n_chunks = 1
    if getattr(store, "gist_src", None) is not None:
        store.gist_src = store.token_hidden.mean(dim=1, keepdim=True).contiguous()


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
    mc.rawkv_readout_topk_chunks = 0
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()
    ro = _readout_layer(model)
    bg = np.load(cli.background)
    rng = random.Random(53)

    Ns = [32, 64, 128]
    modes = ["keep_all", "token_oracle"] + [f"token_top{N}" for N in Ns]
    exact = {m: 0 for m in modes}
    hit = {N: 0 for N in Ns}     # needle digit-tokens captured in top-N
    n = 0

    for i in range(cli.n_samples):
        name = "".join(rng.choices(string.ascii_uppercase, k=6))
        code = " ".join(rng.choices(string.digits, k=5))
        needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        q_text = f"The secret code for agent {name} is"
        ntok = tok.encode(" " + needle, add_special_tokens=False)
        S = len(ntok)                                   # needle occupies store idx 0..S-1
        bg0 = bg[(i + 1100) % len(bg), :cli.chunk_size].tolist()
        chunk0 = (ntok + bg0[len(ntok):])[:cli.chunk_size]
        chunks = [torch.tensor([chunk0], device=device)]
        for c in range(1, cli.n_ctx):
            row = bg[(i + 1100 + c) % len(bg), :cli.chunk_size].tolist()
            chunks.append(torch.tensor([row[:cli.chunk_size]], device=device))
        q_ids = torch.tensor([tok.encode(q_text, add_special_tokens=False)], device=device)
        gold = code.split()

        _reset(model)
        with torch.no_grad():
            for ch in chunks:
                model(input_ids=ch, use_cache=False)
        store = _get_store(model)
        if store is None:
            continue
        snap = (store.token_hidden.clone(), store.token_pos.clone(),
                store.token_chunk.clone(),
                None if store.gist_src is None else store.gist_src.clone())
        M = snap[0].shape[1]
        needle_idx = list(range(min(S, M)))             # needle tokens in store
        _freeze(model)

        # reader per-token salience (one question forward over full store).
        cap = {}

        def _hook(mod, args, kwargs):
            cap["h"] = (args[0] if args else kwargs.get("hidden_states")).detach()
            return None
        h = ro.register_forward_pre_hook(_hook, with_kwargs=True)
        with torch.no_grad():
            model(input_ids=q_ids, use_cache=False)
        h.remove()
        aw = _per_token_salience(ro, store, cap["h"])   # [M]
        order = list(np.argsort(-aw))

        def restore():
            store.token_hidden = snap[0].clone()
            store.token_pos = snap[1].clone()
            store.token_chunk = snap[2].clone()
            store.gist_src = None if snap[3] is None else snap[3].clone()

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

        # keep_all
        restore()
        if decode():
            exact["keep_all"] += 1
        # token_oracle (only needle tokens)
        restore(); _restrict_to_tokens(store, needle_idx, device)
        if decode():
            exact["token_oracle"] += 1
        # token_reader_topN
        for N in Ns:
            sel = order[:N]
            hit[N] += int(len(set(sel) & set(needle_idx)) >= max(1, len(needle_idx) // 2))
            restore(); _restrict_to_tokens(store, sel, device)
            if decode():
                exact[f"token_top{N}"] += 1
        n += 1

    print("\n==== TOKEN-LEVEL selection probe (gather isolation) ====")
    print(f"n={n} n_ctx={cli.n_ctx} chunk_size={cli.chunk_size} M~{cli.n_ctx*cli.chunk_size} (needle~{S} tok in store idx 0..S-1)")
    print("needle-tokens-in-topN hit (>=half needle tokens captured):")
    for N in Ns:
        print(f"  top{N}: {100.0*hit[N]/max(n,1):.1f}%")
    print("end-to-end exact code recall:")
    for m in modes:
        print(f"  {m:<14}: {100.0*exact[m]/max(n,1):.1f}%")
    print("\n(token_oracle = TRUE isolation ceiling; token_topN = reader-attn token selection)")


if __name__ == "__main__":
    main()
