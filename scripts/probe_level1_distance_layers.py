#!/usr/bin/env python3
"""Level 1 distance x injection-layer probe (consumption wall localization).

Uses the TRAINED rawkv-readout path (not the inactive inattn-oracle path). For
each sample we stream ONLY the needle chunk so the readout store holds exactly
the needle's (L16) hidden, freeze, then control two crossed factors and greedy-
decode the 5-digit code:

  factor A = injected RoPE POSITION of the needle KV (overwrite store.token_pos):
      "near": positions 0..S-1 (adjacent to the question, which also starts ~0)
      "far" : positions far_pos..far_pos+S-1 (mimics a distant source chunk)
  factor B = READOUT LAYER SET (toggle per-layer _is_rawkv_readout_layer):
      {16} / {16,20,24} / {16,20,24,28} / {16..31}. Write-owner stays L16.

Selection is neutralised (disable_col_bias=True + keep_all) so this measures
pure CONSUMPTION: given the correct needle KV injected at a chosen distance over
a chosen #layers, can the reader read out the code?

  near>>far          -> distance/RoPE extrapolation (fix: position remap).
  near also low      -> consumption itself (injected-KV OOD / untrained readout).
  far rises w/ #layers -> multi-layer coverage is the lever (confirms S5).
"""
from __future__ import annotations

import argparse
import json
import os
import random
import string
import sys

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


def _set_readout_layers(model, layer_set):
    """Toggle which layers READ from the rawkv store (write-owner stays as-is)."""
    for w in _mem_layers(model):
        li = getattr(w, "_layer_idx", None)
        if hasattr(w, "_is_rawkv_readout_layer"):
            w._is_rawkv_readout_layer = (li in layer_set)


def _banks(model):
    root = getattr(model, "module", model)
    b = getattr(root, "_mem_space_shared_bank", None)
    if b is not None:
        return [b]
    return [getattr(w, "memory_bank", None) for w in _mem_layers(model)]


def _reset_banks(model):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--far_pos", type=int, default=400)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    mc.l3_recon_max_positions = cli.chunk_size
    mc.rawkv_disable_col_bias = True              # neutralise selection bias
    mc.rawkv_readout_topk_chunks = 0              # keep all (1 chunk anyway)
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()

    layer_sets = {
        "L16": [16],
        "L16_20_24": [16, 20, 24],
        "L16_20_24_28": [16, 20, 24, 28],
        "L16_31": list(range(16, 32)),
    }
    positions = ["near", "far"]
    rng = random.Random(11)
    results = {f"{ls}|{p}": {"exact": 0, "n": 0} for ls in layer_sets for p in positions}

    for i in range(cli.n_samples):
        name = "".join(rng.choices(string.ascii_uppercase, k=6))
        code = " ".join(rng.choices(string.digits, k=5))
        needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        q_text = f"The secret code for agent {name} is"
        needle_ids = torch.tensor([tok.encode(" " + needle, add_special_tokens=False)],
                                  device=device)
        q_ids = torch.tensor([tok.encode(q_text, add_special_tokens=False)], device=device)
        gold = code.split()

        for ls_name, ls in layer_sets.items():
            for pos in positions:
                # 1) reset + stream the needle chunk (populate store at L16).
                _reset_banks(model)
                _set_readout_layers(model, set(ls))
                with torch.no_grad():
                    model(input_ids=needle_ids, use_cache=False)
                # 2) grab store, overwrite token_pos for near/far.
                store = None
                for b in _banks(model):
                    s = getattr(b, "_rawkv_readout_store", None)
                    if s is not None and s.size() > 0:
                        store = s
                        break
                if store is None:
                    continue
                M = store.size()
                if pos == "near":
                    new_pos = torch.arange(M, device=device, dtype=torch.long)
                else:
                    new_pos = torch.arange(cli.far_pos, cli.far_pos + M,
                                           device=device, dtype=torch.long)
                store.token_pos = new_pos.unsqueeze(0).expand(store.token_pos.shape[0], -1).clone()
                # 3) freeze + decode the answer from the question.
                _freeze(model)
                cur = q_ids.clone()
                gen = []
                with torch.no_grad():
                    for _ in range(cli.max_new_tokens):
                        out = model(input_ids=cur, use_cache=False)
                        logits = out.logits if hasattr(out, "logits") else out[0]
                        nxt = int(logits[0, -1].argmax().item())
                        if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                            break
                        gen.append(nxt)
                        cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
                out_text = tok.decode(gen, skip_special_tokens=True)
                out_digits = [c for c in out_text if c.isdigit()]
                exact = out_digits[:5] == gold
                k = f"{ls_name}|{pos}"
                results[k]["exact"] += int(exact)
                results[k]["n"] += 1

    print("\n==== LEVEL 1: distance x readout-layer (rawkv path, col_bias=0) ====")
    print(f"n~{cli.n_samples}  far_pos={cli.far_pos}  (exact 5-digit recall)")
    print(f"{'layer_set':<16} {'near':>8} {'far':>8}")
    for ls_name in layer_sets:
        rn = results[f"{ls_name}|near"]; rf = results[f"{ls_name}|far"]
        pn = 100.0 * rn["exact"] / max(rn["n"], 1)
        pf = 100.0 * rf["exact"] / max(rf["n"], 1)
        print(f"{ls_name:<16} {pn:>7.1f}% {pf:>7.1f}%  (n={rn['n']}/{rf['n']})")
    print("\nINTERPRETATION: near>>far=distance/RoPE; near-also-low=consumption "
          "itself; far rises with #layers=multi-layer is the lever (S5).")


if __name__ == "__main__":
    main()
