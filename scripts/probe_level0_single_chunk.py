#!/usr/bin/env python3
"""Level 0 consumption diagnostic: SINGLE-CHUNK in-context recall.

The most basic read-out test. The queried needle AND the question live in ONE
512-token chunk, entirely inside the reader's native attention window — NO
cross-chunk, NO memory bank streaming, NO raw-KV injection, NO RoPE manipulation.
This is the thing the reader should most trivially be able to do.

Sample layout (one chunk, <= chunk_size tokens):
    [ "MEMORIZE: The secret code for agent ABCDEF is 8 0 4 0 2. END_MEMORIZE"
      + background filler ...
      + "The secret code for agent ABCDEF is" ]
We greedy-generate the answer and check whether the 5 digits are recovered
(exact ordered match + a looser "all 5 digits present" rate).

Run on the h1fix ckpt (the same trained reader used in the W0 eval). If single-
chunk recall is also poor (<70%) -> the wall is BASE read-out capability in our
eval regime, not cross-chunk distance/consumption. If it's good (>80%) ->
read-out is fine in-window; the wall is cross-chunk consumption -> Level 1.

Two model modes (run both for cross-check):
  --mode mem_space : load the h1fix mem_space ckpt, feed the single chunk as ONE
                     forward (bank stays empty: 1 chunk, nothing streamed).
  --mode vanilla   : plain HF Llama (no patch) — base read-out ceiling on the
                     SAME single-chunk format (controls for the mem_space wrapper).
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


def build_single_chunk(tok, bg_tokens, chunk_size, rng, fill_frac=0.7):
    """Return (input_ids list, gold_code str). Needle at offset 0, question at
    the end, background filler between, all within chunk_size."""
    name = "".join(rng.choices(string.ascii_uppercase, k=6))
    code = " ".join(rng.choices(string.digits, k=5))           # "8 0 4 0 2"
    needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
    question = f" The secret code for agent {name} is"
    needle_ids = tok.encode(" " + needle, add_special_tokens=False)
    q_ids = tok.encode(question, add_special_tokens=False)
    # background filler to push needle and question apart but still in-window.
    budget = chunk_size - len(needle_ids) - len(q_ids) - 2
    n_fill = int(max(0, min(budget, fill_frac * chunk_size)))
    fill = bg_tokens[:n_fill].tolist() if n_fill > 0 else []
    ids = needle_ids + fill + q_ids
    ids = ids[:chunk_size]
    return ids, code


def score_code(out_text, gold_code):
    gold_digits = gold_code.split()
    # exact ordered: the 5 digits appear in order at the start of output
    out_digits = [c for c in out_text if c.isdigit()]
    exact = out_digits[:5] == gold_digits
    present = all(d in out_digits for d in gold_digits)
    return exact, present


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["mem_space", "vanilla"], default="mem_space")
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_samples", type=int, default=50)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    if cli.mode == "vanilla":
        from transformers import LlamaForCausalLM
        model = LlamaForCausalLM.from_pretrained(
            cli.model_path, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        ).to(device).eval()
    else:
        from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model
        mc = build_mem_space_config(json.load(open(cli.adapter_config)))
        mc.l3_recon_max_positions = cli.chunk_size
        model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
        model.eval()

    bg = np.load(cli.background)
    rng = random.Random(2024)
    n_exact = 0
    n_present = 0
    with torch.no_grad():
        for i in range(cli.n_samples):
            bg_row = bg[(i + 500) % len(bg)]
            ids, gold = build_single_chunk(tok, bg_row, cli.chunk_size, rng)
            inp = torch.tensor([ids], dtype=torch.long, device=device)
            gen_ids = []
            cur = inp
            for _ in range(cli.max_new_tokens):
                out = model(input_ids=cur, use_cache=False)
                logits = out.logits if hasattr(out, "logits") else out[0]
                nxt = int(logits[0, -1].argmax().item())
                if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                    break
                gen_ids.append(nxt)
                cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
            out_text = tok.decode(gen_ids, skip_special_tokens=True)
            exact, present = score_code(out_text, gold)
            n_exact += int(exact)
            n_present += int(present)
            if i < 6:
                print(f"  gold={gold!r} out={out_text[:30]!r} exact={exact} present={present}")

    print(f"\n==== LEVEL 0 SINGLE-CHUNK IN-CONTEXT ({cli.mode}) ====")
    print(f"n={cli.n_samples} chunk_size={cli.chunk_size} (needle+question SAME chunk, no cross-chunk)")
    print(f"exact 5-digit ordered recall = {100.0*n_exact/cli.n_samples:.1f}%")
    print(f"all-5-digits-present rate     = {100.0*n_present/cli.n_samples:.1f}%")
    print("INTERPRETATION:")
    if n_exact / cli.n_samples >= 0.8:
        print("  >=80%: in-window read-out is FINE. Wall is CROSS-CHUNK consumption -> Level 1.")
    elif n_exact / cli.n_samples < 0.7:
        print("  <70%: even single-chunk in-context recall is weak -> the wall is "
              "BASE read-out capability in our eval regime, NOT cross-chunk/distance.")
    else:
        print("  70-80%: borderline; inspect outputs.")


if __name__ == "__main__":
    main()
