#!/usr/bin/env python3
"""Diagnose reader-attn topk selection in a FULL 16-chunk T2 haystack.

Answers: in the SAME controlled T2 setting where isolated-needle readout = 97.5%
(Level 1), when we let keep_set_mode='reader_attn' pick top-k chunks from the
full 16-chunk store and HARD-isolate them, does it (a) put the needle chunk in
the kept set (needle-in-topk hit rate), and (b) read out the code?

This disambiguates the W0 negative:
  * hit rate high + readout high -> reader-attn selection WORKS in T2; the W0
    failure is babilong-needle-distribution specific (not the mechanism).
  * hit rate LOW -> reader-attn cannot isolate the needle among 16 chunks
    (the probe's 55% top1 was over a different/cleaner condition); need stronger
    per-layer selection (Landmark landmark-key).
"""
from __future__ import annotations
import argparse, json, os, random, string, sys
import torch
_HERE = os.path.dirname(os.path.abspath(__file__)); _REPO = os.path.dirname(_HERE)
if _REPO not in sys.path: sys.path.insert(0, _REPO)
from transformers import AutoTokenizer  # noqa: E402
import numpy as np  # noqa: E402
from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model  # noqa: E402


def _mem_layers(model):
    root = getattr(model, "module", model)
    return list(getattr(root, "_mem_space_layers", []))

def _banks(model):
    root = getattr(model, "module", model)
    b = getattr(root, "_mem_space_shared_bank", None)
    return [b] if b is not None else [getattr(w, "memory_bank", None) for w in _mem_layers(model)]

def _reset(model):
    for b in _banks(model):
        if b is None: continue
        if hasattr(b, "reset"): b.reset()
        object.__setattr__(b, "_rawkv_readout_store", None); b.frozen = False

def _freeze(model):
    for b in _banks(model):
        if b is not None: b.frozen = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_ctx", type=int, default=16)
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()
    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    mc.l3_recon_max_positions = cli.chunk_size
    mc.rawkv_disable_col_bias = True
    mc.rawkv_keep_set_mode = "reader_attn"
    mc.rawkv_readout_topk_chunks = cli.topk
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()

    # instrument: capture the kept-set chosen by _reader_attn_keep_set on the
    # readout layer (wrap the method to record indices).
    ro = None
    for w in _mem_layers(model):
        if getattr(w, "_is_rawkv_readout_layer", False):
            ro = w; break
    captured = {}
    orig = ro._reader_attn_keep_set
    def wrapped(*a, **k):
        idx = orig(*a, **k)
        captured["idx"] = None if idx is None else idx.detach().cpu().tolist()
        return idx
    ro._reader_attn_keep_set = wrapped

    bg = np.load(cli.background); rng = random.Random(13)
    hit = 0; exact = 0; n = 0
    for i in range(cli.n_samples):
        name = "".join(rng.choices(string.ascii_uppercase, k=6))
        code = " ".join(rng.choices(string.digits, k=5))
        needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        ntok = tok.encode(" " + needle, add_special_tokens=False)
        chunk0 = (ntok + bg[(i+700) % len(bg), :cli.chunk_size].tolist()[len(ntok):])[:cli.chunk_size]
        chunks = [torch.tensor([chunk0], device=device)]
        for c in range(1, cli.n_ctx):
            chunks.append(torch.tensor([bg[(i+700+c) % len(bg), :cli.chunk_size].tolist()[:cli.chunk_size]], device=device))
        q_ids = torch.tensor([tok.encode(f"The secret code for agent {name} is", add_special_tokens=False)], device=device)
        gold = code.split()
        _reset(model)
        with torch.no_grad():
            for ch in chunks: model(input_ids=ch, use_cache=False)
        _freeze(model)
        captured.clear()
        cur = q_ids.clone(); gen = []
        with torch.no_grad():
            for _ in range(cli.max_new_tokens):
                out = model(input_ids=cur, use_cache=False)
                lg = out.logits if hasattr(out, "logits") else out[0]
                nxt = int(lg[0, -1].argmax().item())
                if tok.eos_token_id is not None and nxt == tok.eos_token_id: break
                gen.append(nxt); cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
        idx = captured.get("idx")
        if idx is not None and 0 in idx: hit += 1   # needle is chunk 0
        out_text = tok.decode(gen, skip_special_tokens=True)
        od = [c for c in out_text if c.isdigit()]
        if od[:5] == gold: exact += 1
        n += 1
        if i < 5:
            print(f"  needle@chunk0 kept_idx={idx} hit={0 in idx if idx else False} "
                  f"gold={code} out={out_text[:24]!r}")

    print(f"\n==== reader-attn topk={cli.topk} in FULL {cli.n_ctx}-chunk T2 haystack ====")
    print(f"n={n}  needle-in-topk hit rate = {100.0*hit/n:.1f}%  "
          f"(random {100.0*cli.topk/cli.n_ctx:.1f}%)")
    print(f"exact code readout = {100.0*exact/n:.1f}%  (isolated ceiling=97.5%)")
    print("INTERP: hit high+readout high=selection works in T2 (W0 fail is "
          "babilong-specific); hit LOW=reader-attn can't isolate among 16 "
          "(need stronger per-layer selection).")


if __name__ == "__main__":
    main()
