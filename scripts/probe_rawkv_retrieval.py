#!/usr/bin/env python3
"""H2 probe: does Method A's trained gist-scorer SELECT the needle-bearing chunk?

Loads the final rawkv_methodA ckpt, builds a T2-style associative-recall sample
(queried needle at chunk 0 offset 0, n_ctx context chunks of pg19 background,
then a question target chunk), streams the context chunks into the readout store
exactly as eval does (forward under bank-writable), then at question time calls
GistReadout.retrieve with the question hidden and inspects the per-chunk soft
weights — i.e. measures "needle precision": does chunk 0 (the only chunk holding
the answer) win the gist selection?

Verdict logic:
  * chunk-0 weight >> uniform (1/C) and chunk-0 = argmax  -> scorer DID learn to
    retrieve  -> H2 false (gradient signal worked; failure is elsewhere, e.g.
    consumption). Combined with the failed long-range eval that would be odd, so
    we expect the opposite:
  * chunk-0 weight ~= uniform / not argmax  -> scorer is RANDOM -> the trained
    cross-chunk scorer never learned selection (consistent with top1_sim~0.24).

Run on B200 (has ckpt + .venv). Single GPU.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

from scripts.run_babilong_mem_space import build_mem_space_config, load_mem_space_model  # noqa: E402
from src.memory.mem_space.niah_chunked_dataset import NIAHChunkedDataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402


def _find_readout_layer(model):
    for w in getattr(model, "_mem_space_layers", []) or []:
        if getattr(w, "_is_rawkv_readout_layer", False):
            return w
    return None


def _freeze_banks(model):
    for w in getattr(model, "_mem_space_layers", []) or []:
        b = getattr(w, "memory_bank", None)
        if b is not None:
            b.frozen = True


def _reset_banks(model):
    for w in getattr(model, "_mem_space_layers", []) or []:
        b = getattr(w, "memory_bank", None)
        if b is not None:
            if hasattr(b, "reset"):
                b.reset()
            # also clear the rawkv readout store
            object.__setattr__(b, "_rawkv_readout_store", None)
            b.frozen = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", default="outputs/rawkv_methodA_b200/full_model.pt")
    ap.add_argument("--adapter_config", default="outputs/rawkv_methodA_b200/adapter_config.json")
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--gap_tokens", type=int, default=3584)
    ap.add_argument("--num_keys", type=int, default=1,
                    help="T2 num_keys: 1 = single needle (orig); >1 adds "
                         "distractor needles in other chunks (match training).")
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    cfg_json = json.load(open(cli.adapter_config))
    mc = build_mem_space_config(cfg_json)
    print(f"use_rawkv_readout={mc.use_rawkv_readout} layers={mc.rawkv_readout_layers} "
          f"gist_dim={mc.rawkv_gist_dim} topk_chunks={mc.rawkv_readout_topk_chunks} "
          f"temp={mc.rawkv_readout_temp}")

    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()
    ro_layer = _find_readout_layer(model)
    assert ro_layer is not None, "no readout layer found"
    gist = model._gist_readout
    print(f"readout layer idx={ro_layer._layer_idx}  n_ctx will be "
          f"{max(1, round(cli.gap_tokens / cli.chunk_size))}")

    bg = np.load(cli.background)
    ds = NIAHChunkedDataset(
        background_data=bg, chunk_size=cli.chunk_size, gap_tokens=cli.gap_tokens,
        tokenizer=tok, num_keys=cli.num_keys, seed=1234, background_skip=10000,
    )
    it = iter(ds)
    C = ds.n_ctx  # needle at chunk 0; C context chunks total

    # Instrument GistReadout.retrieve to capture per-chunk weights of the FULL
    # set (we set topk to keep_all so we see every chunk's weight). We monkey-
    # patch by wrapping: call retrieve with topk_chunks=0 (keep all) and read
    # _last_* + recompute the weight vector here for clarity.
    rank_hits = []   # 1 if chunk0 is argmax of mean per-chunk weight
    top3_hits = []
    chunk0_w = []
    uniform = 1.0 / C
    per_chunk_w_accum = np.zeros(C)
    score_spread = []   # pre-softmax score range across chunks at query pos
    diag = {}

    n = 0
    with torch.no_grad():
        while n < cli.n_samples:
            sample = next(it)
            ctx = sample["context_chunks"]          # list[C] each [chunk_size]
            tgt = sample["target_ids"]              # [chunk_size]
            # eval lifecycle: reset, stream context (bank writable), freeze, then
            # read at the question.
            _reset_banks(model)
            for ci in ctx:
                model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)
            _freeze_banks(model)

            # Question hidden at the readout layer: we need the layer-input
            # hidden_states of the target chunk at ro_layer. Easiest: run the
            # target forward and grab gist scoring directly. We re-run retrieve
            # manually using the captured query hidden. To get the query hidden
            # at the readout layer we register a forward-pre hook capturing the
            # input to the wrapped readout layer.
            cap = {}

            def _hook(mod, args, kwargs):
                # the mem_space layer forward gets hidden_states as first arg
                hs = args[0] if args else kwargs.get("hidden_states")
                cap["h"] = hs.detach()
                return None

            h = ro_layer.register_forward_pre_hook(_hook, with_kwargs=True)
            try:
                model(input_ids=tgt.unsqueeze(0).to(device), use_cache=False)
            finally:
                h.remove()

            qh = cap.get("h")
            store = getattr(ro_layer.memory_bank, "_rawkv_readout_store", None)
            if qh is None or store is None or store.size() == 0:
                continue
            # Recompute the per-(query-token, chunk) weights with keep_all so we
            # can read every chunk's weight (not just the kept set).
            gkey = gist.key_proj(store.gist_src.to(device, dtype=qh.dtype))   # [B,C,g]
            gq = gist.query_proj(qh)                                          # [B,Tq,g]
            score = torch.einsum("bqg,bcg->bqc", gq, gkey) * gist._scale
            score_raw = score.clone()
            score = score / max(float(mc.rawkv_readout_temp), 1e-6)
            w = torch.softmax(score, dim=-1)            # [B,Tq,C]
            # Diagnostics to separate "projections collapsed" from "gist sources
            # indistinguishable": gist_src pairwise cosine + pre-softmax score
            # spread across chunks at the query position.
            if n == 0:
                gsrc = store.gist_src[0].float()        # [C,d]
                gsrc_n = torch.nn.functional.normalize(gsrc, dim=-1)
                cos = (gsrc_n @ gsrc_n.t())             # [C,C]
                offdiag = cos[~torch.eye(cos.shape[0], dtype=torch.bool, device=cos.device)]
                diag["gist_src_offdiag_cos_mean"] = float(offdiag.mean())
                diag["gist_src_offdiag_cos_min"] = float(offdiag.min())
                diag["gkey_norm_mean"] = float(gkey[0].norm(dim=-1).mean())
                diag["gq_norm_mean"] = float(gq[0].norm(dim=-1).mean())
                diag["qproj_w_norm"] = float(gist.query_proj.weight.norm())
                diag["kproj_w_norm"] = float(gist.key_proj.weight.norm())
            # Use the question's LAST real token as the retrieval query position
            # (that's where generation reads). Find last non-pad token in tgt.
            real = (tgt != tok.pad_token_id).nonzero().squeeze(-1)
            qpos = int(real[-1].item()) if real.numel() > 0 else tgt.shape[0] - 1
            qpos = min(qpos, w.shape[1] - 1)
            wq = w[0, qpos].float().cpu().numpy()        # [C]
            Cw = wq.shape[0]
            if Cw != C:
                # store may include the target chunk if it was written; trim to
                # the context chunks (first C).
                wq = wq[:C]
            per_chunk_w_accum[:len(wq)] += wq
            chunk0_w.append(float(wq[0]))
            sr = score_raw[0, qpos].float().cpu().numpy()[:C]
            score_spread.append(float(sr.max() - sr.min()))
            rank_hits.append(1 if int(wq.argmax()) == 0 else 0)
            top2 = set(np.argsort(wq)[-2:].tolist())
            top3_hits.append(1 if 0 in top2 else 0)
            n += 1

    chunk0_w = np.array(chunk0_w)
    print("\n==== H2 PROBE RESULT ====")
    print(f"samples={n}  C(context chunks)={C}  uniform weight=1/C={uniform:.4f}")
    print(f"needle chunk = chunk 0 (always)")
    print(f"chunk0 gist weight: mean={chunk0_w.mean():.4f} std={chunk0_w.std():.4f} "
          f"(uniform={uniform:.4f})  ratio_to_uniform={chunk0_w.mean()/uniform:.2f}x")
    print(f"needle precision @top1 (chunk0 == argmax): {np.mean(rank_hits)*100:.1f}%  "
          f"(random@top1 = {100.0/C:.1f}%)")
    print(f"needle precision @top2 (chunk0 in top2) : {np.mean(top3_hits)*100:.1f}%  "
          f"(random@top2 = {min(1.0,2.0/C)*100:.1f}%)  <- matches trained topk=2")
    print(f"mean per-chunk weight vector (idx0=needle):")
    print("  " + " ".join(f"{x/n:.3f}" for x in per_chunk_w_accum))
    ss = np.array(score_spread)
    print(f"pre-softmax score spread across chunks (max-min) @query pos: "
          f"mean={ss.mean():.4f} max={ss.max():.4f}")
    print(f"-- mechanism diag (first sample) --")
    for k, v in diag.items():
        print(f"   {k} = {v:.4f}")
    print("\nINTERPRETATION:")
    if np.mean(rank_hits) > 0.5 or chunk0_w.mean() > 2 * uniform:
        print("  -> scorer DID concentrate on the needle chunk: gradient signal "
              "WORKED (H2 false). Failure is downstream (consumption), or H1.")
    else:
        print("  -> scorer ~ RANDOM (chunk0 not preferentially selected): the "
              "trained cross-chunk gist-scorer never learned selection. Consistent "
              "with top1_sim~0.24. Points to H2 (gradient signal ineffective) "
              "unless H1 removed the pressure to begin with.")


if __name__ == "__main__":
    main()
