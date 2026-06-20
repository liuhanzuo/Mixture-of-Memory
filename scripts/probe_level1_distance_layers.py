#!/usr/bin/env python3
"""Level 1 consumption diagnostic: distance x injection-layer-set matrix.

Level 0 showed in-window read-out is fine (vanilla 100% / mem_space 96%) but
cross-chunk W0 collapses (<=14%). Level 1 localizes WHY, with the CORRECT needle
content fed directly (oracle, bypassing selection — selection is not the
bottleneck, probed at 55%):

For each sample we capture the queried needle span's per-layer INPUT hidden
states (the faithful raw-KV at each depth), then inject them via the oracle
EV path under a grid:

  distance in {near, far}:
    near = inject at RoPE positions ADJACENT to the query (pretend the needle is
           right before the question) -> tests read-out with no distance gap.
    far  = inject at the needle's REAL remote position (n_ctx*chunk_size before)
           -> the true long-range condition.
  layer_set in {L16, L16/20/24, L16-31-every4, L16-31-all}:
    how many decoder layers independently receive + consume the needle KV.

We greedy-decode the answer from the question chunk (memory bank empty; the ONLY
path to the answer is the injected oracle EV) and score 5-digit recall.

Verdict:
  * near >> far (at fixed layers)         -> distance/RoPE extrapolation wall.
  * far improves sharply with more layers -> layer-coverage wall (S5 confirmed
    on mem_space side): fix = multi-layer injection + (train) multi-layer readout.
  * far stays ~0 regardless of layers/distance -> injected-KV consumption itself
    is broken (OOD representation / never trained to read injected KV).
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


def _layers_of(model):
    root = getattr(model, "module", model)
    return {w._layer_idx: w for w in getattr(root, "_mem_space_layers", [])}


def _set_oracle(model, hidden_by_layer, layers, pos_by_layer):
    root = getattr(model, "module", model)
    bank = getattr(root, "_mem_space_shared_bank", None)
    targets = [bank] if bank is not None else [
        getattr(w, "memory_bank", None) for w in getattr(root, "_mem_space_layers", [])
    ]
    for b in targets:
        if b is None:
            continue
        b._oracle_layers = set(layers) if hidden_by_layer else None
        b._oracle_hidden_by_layer = hidden_by_layer
        b._oracle_pos_by_layer = pos_by_layer if hidden_by_layer else None


def _reset_banks(model):
    for w in getattr(model, "_mem_space_layers", []) or []:
        b = getattr(w, "memory_bank", None)
        if b is not None and hasattr(b, "reset"):
            b.reset()


def _capture_needle_hidden(model, needle_ids, layers, device):
    """Run the needle span alone, capture per-layer INPUT hidden [1,S,d]."""
    lm = _layers_of(model)
    cap = {}
    handles = []

    def mk(li):
        def hook(mod, args, kwargs):
            h = args[0] if args else kwargs.get("hidden_states")
            cap[li] = h.detach().clone()
        return hook
    for li in layers:
        if li in lm:
            handles.append(lm[li].register_forward_pre_hook(mk(li), with_kwargs=True))
    try:
        with torch.no_grad():
            model(input_ids=needle_ids.to(device), use_cache=False)
    finally:
        for h in handles:
            h.remove()
    return cap


LAYER_SETS = {
    "L16": [16],
    "L16_20_24": [16, 20, 24],
    "L16-31e4": [16, 20, 24, 28],
    "L16-31all": list(range(16, 32)),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_h1fix_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_h1fix_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--n_ctx", type=int, default=16, help="far distance = n_ctx*chunk_size")
    ap.add_argument("--n_samples", type=int, default=30)
    ap.add_argument("--max_new_tokens", type=int, default=12)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    mc.l3_recon_max_positions = cli.chunk_size
    # CRITICAL: oracle EV must use real (injected) positions so near/far differ.
    mc.evidence_real_positions = True
    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()
    all_layers = sorted({li for s in LAYER_SETS.values() for li in s})

    bg = np.load(cli.background)
    rng = random.Random(7)
    cs = cli.chunk_size
    far_pos_base = cli.n_ctx * cs  # real remote position of the needle (chunk0 start)

    # results[dist][lset] = list of exact-hit booleans
    results = {d: {k: [] for k in LAYER_SETS} for d in ("near", "far")}

    for i in range(cli.n_samples):
        name = "".join(rng.choices(string.ascii_uppercase, k=6))
        code = " ".join(rng.choices(string.digits, k=5))
        needle = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
        needle_ids = torch.tensor(
            [tok.encode(" " + needle, add_special_tokens=False)], dtype=torch.long)
        q_ids = tok.encode(f"The secret code for agent {name} is", add_special_tokens=False)
        S = needle_ids.shape[1]
        cap = _capture_needle_hidden(model, needle_ids, all_layers, device)
        gold = code.split()

        q_inp = torch.tensor([q_ids], dtype=torch.long, device=device)
        Lq = len(q_ids)
        for dist in ("near", "far"):
            if dist == "near":
                # EV positions right before the question (query starts at 0 in its
                # own forward; put needle just "before" at 0..S-1 -> minimal gap).
                pos = torch.arange(S, dtype=torch.long).unsqueeze(0)
            else:
                pos = torch.arange(far_pos_base, far_pos_base + S,
                                   dtype=torch.long).unsqueeze(0)
            for lname, lset in LAYER_SETS.items():
                hbl = {li: cap[li] for li in lset if li in cap}
                pbl = {li: pos for li in lset if li in cap}
                _reset_banks(model)
                _set_oracle(model, hbl, lset, pbl)
                try:
                    gen = []
                    cur = q_inp
                    with torch.no_grad():
                        for _ in range(cli.max_new_tokens):
                            out = model(input_ids=cur, use_cache=False)
                            logits = out.logits if hasattr(out, "logits") else out[0]
                            nxt = int(logits[0, -1].argmax().item())
                            if tok.eos_token_id is not None and nxt == tok.eos_token_id:
                                break
                            gen.append(nxt)
                            cur = torch.cat([cur, torch.tensor([[nxt]], device=device)], dim=1)
                    txt = tok.decode(gen, skip_special_tokens=True)
                finally:
                    _set_oracle(model, None, lset, None)
                out_digits = [c for c in txt if c.isdigit()]
                results[dist][lname].append(out_digits[:5] == gold)

    print("\n==== LEVEL 1: distance x injection-layer-set (oracle needle, bypass selection) ====")
    print(f"n={cli.n_samples} chunk_size={cs} far_gap={far_pos_base} tok (n_ctx={cli.n_ctx})")
    print(f"{'layer_set':<14} {'NEAR':>8} {'FAR':>8}")
    for lname in LAYER_SETS:
        nr = 100.0 * np.mean(results['near'][lname])
        fr = 100.0 * np.mean(results['far'][lname])
        print(f"{lname:<14} {nr:>7.1f}% {fr:>7.1f}%")
    # verdict helpers
    near_all = 100.0 * np.mean(results['near']['L16-31all'])
    far_all = 100.0 * np.mean(results['far']['L16-31all'])
    far_l16 = 100.0 * np.mean(results['far']['L16'])
    near_l16 = 100.0 * np.mean(results['near']['L16'])
    print("\nINTERPRETATION:")
    print(f"  far improves with layers: L16={far_l16:.0f}% -> L16-31all={far_all:.0f}%")
    print(f"  near vs far (all layers): near={near_all:.0f}% far={far_all:.0f}%")
    if far_all - far_l16 > 25:
        print("  -> LAYER-COVERAGE wall (S5 confirmed mem_space side): multi-layer "
              "injection sharply rescues far. Fix = multi-layer readout.")
    if near_all - far_all > 25 and far_all < 30:
        print("  -> DISTANCE/RoPE wall: near works, far fails even multi-layer. "
              "Fix = position remap / distance normalization.")
    if far_all < 15 and near_all < 30:
        print("  -> injected-KV CONSUMPTION itself broken (reader never learned to "
              "read injected KV regardless of distance/layers). Fix = train readout.")


if __name__ == "__main__":
    main()
