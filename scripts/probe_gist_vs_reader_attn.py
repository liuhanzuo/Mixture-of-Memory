#!/usr/bin/env python3
"""Mechanism probe (H2 root cause): does the gist scorer select the chunk the
READER's native attention is heavy on, rather than the chunk that holds the
NEEDLE?

For each sample (queried needle ALWAYS at chunk 0; num_keys distractors in other
chunks), at the readout layer we compute THREE per-chunk distributions at the
query's last real token:

  G[c] = gist soft-selection weight over chunks (what the trained scorer picks).
  A[c] = READER native-attention mass over chunks = softmax over the retrieved
         raw-KV columns of (q . k_raw / sqrt(hd)) WITHOUT the gist col_bias,
         aggregated per source chunk. This is "where the reader content-wants to
         look" independent of the gist scorer.
  needle = chunk 0.

Then we report:
  * gist argmax chunk vs reader argmax chunk agreement (does gist track reader?)
  * gist vs reader per-chunk weight correlation (Pearson over C chunks, mean).
  * needle precision of gist (top1) and of reader-attention (top1).

Interpretation:
  * high gist<->reader agreement + both miss the needle  -> scorer learned
    "where the reader wants to look" (LM-loss-driven diffuse attention), NOT
    "where the needle is" -> confirms selection must leave the LM-loss path
    (Landmark-style emergent, selection not in loss).
  * gist tracks needle (precision high) -> would contradict H2.
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
from src.memory.mem_space.inattn_kv import build_retrieved_kv  # noqa: E402
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
            object.__setattr__(b, "_rawkv_readout_store", None)
            b.frozen = False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--background", default="data/pg19_chunks_llama3.npy")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--gap_tokens", type=int, default=8192)
    ap.add_argument("--num_keys", type=int, default=3)
    ap.add_argument("--n_samples", type=int, default=40)
    ap.add_argument("--device", default="cuda:0")
    cli = ap.parse_args()

    device = torch.device(cli.device)
    mc = build_mem_space_config(json.load(open(cli.adapter_config)))
    tok = AutoTokenizer.from_pretrained(cli.model_path, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = load_mem_space_model(cli.model_path, cli.ckpt, mc, device, torch.bfloat16, "sdpa")
    model.eval()
    ro = _find_readout_layer(model)
    gist = model._gist_readout
    attn = ro.wrapped_layer.self_attn
    pre_norm = getattr(ro.wrapped_layer, "input_layernorm", None)
    hd = attn.head_dim

    bg = np.load(cli.background)
    ds = NIAHChunkedDataset(
        background_data=bg, chunk_size=cli.chunk_size, gap_tokens=cli.gap_tokens,
        tokenizer=tok, num_keys=cli.num_keys, seed=4321, background_skip=20000,
    )
    it = iter(ds)
    C = ds.n_ctx

    gist_top1_needle = []
    reader_top1_needle = []
    gist_reader_agree = []        # gist argmax == reader argmax
    corr_gr = []                  # pearson(G, A) over chunks
    gist_w0 = []
    reader_w0 = []

    n = 0
    with torch.no_grad():
        while n < cli.n_samples:
            sample = next(it)
            ctx = sample["context_chunks"]
            tgt = sample["target_ids"]
            _reset_banks(model)
            for ci in ctx:
                model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)
            _freeze_banks(model)

            cap = {}

            def _hook(mod, args, kwargs):
                hs = args[0] if args else kwargs.get("hidden_states")
                cap["h"] = hs.detach()
                pe = kwargs.get("position_embeddings")
                if pe is not None:
                    cap["pe"] = pe
                return None

            h = ro.register_forward_pre_hook(_hook, with_kwargs=True)
            try:
                model(input_ids=tgt.unsqueeze(0).to(device), use_cache=False)
            finally:
                h.remove()

            qh = cap.get("h")
            pe = cap.get("pe")
            store = getattr(ro.memory_bank, "_rawkv_readout_store", None)
            if qh is None or store is None or store.size() == 0 or pe is None:
                continue

            # query position = last real token of the question.
            real = (tgt != tok.pad_token_id).nonzero().squeeze(-1)
            qpos = int(real[-1].item()) if real.numel() > 0 else tgt.shape[0] - 1
            qpos = min(qpos, qh.shape[1] - 1)

            # ---- GIST distribution over chunks ----
            gkey = gist.key_proj(store.gist_src.to(device, dtype=qh.dtype))   # [1,C,g]
            gq = gist.query_proj(qh)                                          # [1,Tq,g]
            gscore = torch.einsum("bqg,bcg->bqc", gq, gkey) * gist._scale
            G = torch.softmax(gscore[0, qpos].float(), dim=-1).cpu().numpy()[:C]  # [C]

            # ---- READER native-attention mass over chunks (NO gist bias) ----
            # Build retrieved raw K from ALL stored tokens at their real positions
            # (same as the read path, keep_all), project the query through native
            # q_proj + RoPE, score q.k_raw, softmax over retrieved columns only,
            # then aggregate the attention mass per source chunk.
            ret_h = store.token_hidden                       # [1, M, d]
            ret_pos = store.token_pos                         # [1, M]
            ret_chunk = store.token_chunk[0].cpu().numpy()    # [M]
            K_raw, _V = build_retrieved_kv(
                attn, ret_h.to(qh.dtype), ret_pos, pe, pre_norm=pre_norm,
            )                                                 # [1, n_kv, M, hd]
            # query vector at qpos through native q_proj + RoPE.
            _hs = qh
            if pre_norm is not None:
                _hs = pre_norm(_hs)
            q = attn.q_proj(_hs).view(1, qh.shape[1], -1, hd).transpose(1, 2)  # [1,nh,Tq,hd]
            from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
            cos, sin = pe
            # rope q only (reuse k=q trick): apply to q with a dummy k
            q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)
            qv = q_r[0, :, qpos, :]                           # [nh, hd]
            # expand K_raw n_kv -> nh (GQA repeat)
            nh = qv.shape[0]
            n_kv = K_raw.shape[1]
            rep = nh // n_kv
            Kr = K_raw[0].repeat_interleave(rep, dim=0)       # [nh, M, hd]
            aw = torch.einsum("hd,hmd->hm", qv.float(), Kr.float()) * (hd ** -0.5)  # [nh,M]
            aw = torch.softmax(aw, dim=-1).mean(dim=0).cpu().numpy()  # [M] avg heads
            # aggregate per chunk
            A = np.zeros(C)
            for c in range(C):
                A[c] = aw[ret_chunk == c].sum()

            # ---- metrics ----
            g_arg = int(G.argmax())
            a_arg = int(A.argmax())
            gist_top1_needle.append(1 if g_arg == 0 else 0)
            reader_top1_needle.append(1 if a_arg == 0 else 0)
            gist_reader_agree.append(1 if g_arg == a_arg else 0)
            gist_w0.append(float(G[0]))
            reader_w0.append(float(A[0]))
            if G.std() > 1e-9 and A.std() > 1e-9:
                corr_gr.append(float(np.corrcoef(G, A)[0, 1]))
            n += 1

    uni = 1.0 / C
    print("\n==== MECHANISM PROBE: gist-selection vs reader-attention vs needle ====")
    print(f"samples={n} C={C} uniform=1/C={uni:.4f} (needle=chunk0, num_keys={cli.num_keys})")
    print(f"gist  needle precision @top1 = {np.mean(gist_top1_needle)*100:.1f}% "
          f"(random {100*uni:.1f}%)")
    print(f"reader-attn needle precision @top1 = {np.mean(reader_top1_needle)*100:.1f}% "
          f"(random {100*uni:.1f}%)")
    print(f"gist-argmax == reader-argmax agreement = {np.mean(gist_reader_agree)*100:.1f}% "
          f"(chance {100*uni:.1f}%)")
    print(f"mean Pearson corr(gist_weights, reader_attn) over chunks = "
          f"{np.mean(corr_gr):+.3f}  (n={len(corr_gr)})")
    print(f"chunk0 weight: gist={np.mean(gist_w0):.4f} reader={np.mean(reader_w0):.4f} "
          f"(uniform {uni:.4f})")
    print("\nINTERPRETATION:")
    agree = np.mean(gist_reader_agree)
    corr = np.mean(corr_gr) if corr_gr else 0.0
    g_prec = np.mean(gist_top1_needle)
    if g_prec > 2 * uni:
        print("  gist tracks the NEEDLE -> would contradict H2.")
    elif agree > 2 * uni or corr > 0.3:
        print("  gist SELECTION TRACKS READER ATTENTION (not the needle): the "
              "scorer learned 'where the reader content-wants to look' under LM "
              "loss, which is NOT needle location. -> selection must leave the LM-"
              "loss optimization path (Landmark-style emergent, not trained).")
    else:
        print("  gist neither tracks needle nor reader-attn cleanly: selection is "
              "~random/idiosyncratic. Still consistent with H2 (no gradient toward "
              "correct retrieval); reader-attn itself may also miss the needle.")


if __name__ == "__main__":
    main()
