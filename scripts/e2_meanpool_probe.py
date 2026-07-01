"""E2-meanpool — last-token vs mean-pool salience ablation probe.

Copied from e2_multiscorer_probe.py (2026-06-29). ONLY CHANGE: adds scorer S1mp
(mean-pool variant of S1 reader-attn salience). All other scorers (S1, S2) are
unchanged. S3/S4 are intentionally OMITTED (they cause hangs on 16k/large limit).

Key difference vs S1:
  S1   uses qv = q_r[:, :, -1, :]        (last query token only)
  S1mp uses qv = q_r.mean(dim=2)         (mean-pool over all query tokens)

This tests the architectural-defect hypothesis: does restricting the salience probe
to the last query token degrade recall because the last token ("is"/"?") carries
no entity information?

Usage example (qa5 16k, limit 30, GPUs 2/3):
  CUDA_VISIBLE_DEVICES=2 python scripts/e2_meanpool_probe.py \\
    --model_path models/Meta-Llama-3-8B \\
    --checkpoint outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt \\
    --adapter_config outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json \\
    --tasks qa5 --lengths 16k --limit 30 --scorers S1,S1mp,S2 \\
    --device cuda:0 --output_csv /tmp/e2_meanpool_qa5_16k_n30.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import json  # noqa: E402

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)

from src.memory.mem_space import MemorySpaceConfig, _reset_fifo_memory  # noqa: E402

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_babilong_dataset,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
    _locate_needle_chunks,
    _set_fifo_keep_all_buffer,
)

RECALL_KS = (1, 2, 4, 8, 16)
# S1mp = mean-pool variant of S1; S3/S4 intentionally omitted.
ALL_SCORERS = ("S1", "S1mp", "S1idf", "S2", "S1fuse")


# --------------------------------------------------------------------------- #
# Pool / streaming helpers (unchanged from e2_multiscorer_probe.py)
# --------------------------------------------------------------------------- #


def _stream_context(model, chunks, device):
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            _ = model(input_ids=chunk.unsqueeze(0).to(device), use_cache=False)


def _freeze(model):
    root = getattr(model, "module", model)
    sb = getattr(root, "_mem_space_shared_bank", None)
    if sb is not None:
        sb.frozen = True
        return
    for w in getattr(root, "_mem_space_layers", []) or []:
        w.memory_bank.frozen = True


def _unfreeze(model):
    root = getattr(model, "module", model)
    sb = getattr(root, "_mem_space_shared_bank", None)
    if sb is not None:
        sb.frozen = False
        return
    for w in getattr(root, "_mem_space_layers", []) or []:
        w.memory_bank.frozen = False


def _pool_from_buffer(model, n_chunks, select_layer):
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return None, 0
    L = int(select_layer)
    if L < 0 or L >= len(mem_layers):
        return None, 0
    buf = getattr(mem_layers[L], "_fifo_buf", None)
    if not buf:
        return None, 0
    C = len(buf)
    ingested = n_chunks - 1
    offset = ingested - C
    return int(offset), int(C)


# --------------------------------------------------------------------------- #
# S1: reader-attn salience — LAST TOKEN (unchanged from e2_multiscorer_probe.py)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_S1_reader_attn(model, last_chunk, device, select_layer):
    """Full-buffer reader-attn q.k salience [C] using LAST query token."""
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return None
    L = int(select_layer)
    if L < 0 or L >= len(mem_layers):
        return None
    sel = mem_layers[L]
    buf = getattr(sel, "_fifo_buf", None)
    if not buf:
        return None
    buf_snapshot = list(buf)
    C = len(buf_snapshot)

    cur = last_chunk.unsqueeze(0).to(device)
    out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None or L >= len(hs):
        return None
    q_hidden = hs[L]  # [1, last_len, d]

    rot = None
    inner = getattr(root, "model", None)
    if inner is not None:
        rot = getattr(inner, "rotary_emb", None)
    if rot is None:
        rot = sel._fifo_resolve_rotary_emb()
    if rot is None:
        return None
    Tq = q_hidden.shape[1]
    pos_ids = torch.arange(Tq, device=q_hidden.device).unsqueeze(0)
    cos, sin = rot(q_hidden, pos_ids)

    sal = sel._fifo_reader_attn_salience(
        hidden_states=q_hidden,
        chunk_hiddens=buf_snapshot,
        position_embeddings=(cos, sin),
    )
    if sal is None or sal.shape[0] != C:
        return None
    return sal.detach().float().cpu()


# --------------------------------------------------------------------------- #
# S1mp: reader-attn salience — MEAN POOL over all query tokens
# Key change vs S1/layer.py:
#   layer.py L1731: qv = q_r[:, :, -1, :]    (last token)
#   here:           qv = q_r.mean(dim=2)      (mean over all tokens in last chunk)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_S1mp_reader_attn(model, last_chunk, device, select_layer):
    """Full-buffer reader-attn q.k salience [C] using MEAN-POOL over all query tokens.

    Identical setup to score_S1_reader_attn but replaces the last-token qv with
    the mean over all token positions in the query chunk. This tests whether the
    last-token restriction degrades recall for qa5 (where the last token is
    always "is"/"?", not the entity being queried).

    Changed line (vs layer.py L1731 and score_S1_reader_attn):
        qv = q_r[:, :, -1, :]   ->   qv = q_r.mean(dim=2)   # [B, nh, hd]
    """
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb

    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return None
    L = int(select_layer)
    if L < 0 or L >= len(mem_layers):
        return None
    sel = mem_layers[L]
    buf = getattr(sel, "_fifo_buf", None)
    if not buf:
        return None
    buf_snapshot = list(buf)
    C = len(buf_snapshot)

    # Forward through the model to get hidden states at layer L
    cur = last_chunk.unsqueeze(0).to(device)
    out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None or L >= len(hs):
        return None
    q_hidden = hs[L]  # [1, last_len, d] = input to layer L (before self-attn)

    # Resolve rotary embedding
    rot = None
    inner = getattr(root, "model", None)
    if inner is not None:
        rot = getattr(inner, "rotary_emb", None)
    if rot is None:
        rot = sel._fifo_resolve_rotary_emb()
    if rot is None:
        return None

    # Build q projection + RoPE (mirrors layer.py L1725-1728)
    _attn = getattr(sel.wrapped_layer, "self_attn", None)
    if _attn is None:
        return None
    _pre_norm = getattr(sel.wrapped_layer, "input_layernorm", None)

    B, Tq, d = q_hidden.shape
    hd = _attn.head_dim

    _hs_q = _pre_norm(q_hidden) if _pre_norm is not None else q_hidden
    q = _attn.q_proj(_hs_q).view(B, Tq, -1, hd).transpose(1, 2)   # [B, nh, Tq, hd]

    pos_ids = torch.arange(Tq, device=q_hidden.device).unsqueeze(0)
    cos, sin = rot(q_hidden, pos_ids)
    q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)                  # [B, nh, Tq, hd]

    # *** THE KEY CHANGE: mean-pool over all Tq query positions ***
    # S1 (original):  qv = q_r[:, :, -1, :]    # [B, nh, hd] — last token only
    # S1mp (ours):    qv = q_r.mean(dim=2)      # [B, nh, hd] — mean over all tokens
    qv = q_r.mean(dim=2)                                            # [B, nh, hd]

    nh = qv.shape[1]
    sal_list = []
    for _kh in buf_snapshot:
        _kh_in = _kh.to(q_hidden.device, dtype=q_hidden.dtype)
        if _pre_norm is not None:
            _kh_in = _pre_norm(_kh_in)
        M = _kh_in.shape[1]
        kk = _attn.k_proj(_kh_in).view(B, M, -1, hd).transpose(1, 2)  # [B, nkv, M, hd]
        nkv = kk.shape[1]
        if nh != nkv:
            kk = kk.repeat_interleave(nh // nkv, dim=1)                # [B, nh, M, hd]
        aw = torch.einsum("bhd,bhmd->bhm", qv.float(), kk.float()) * (hd ** -0.5)
        aw = aw.amax(dim=1)                                             # [B, M]
        sal_list.append(aw.amax(dim=-1).mean().float())                 # scalar
    sal = torch.stack(sal_list, dim=0)                                  # [C]
    return sal.detach().float().cpu()


# --------------------------------------------------------------------------- #
# S1idf: reader-attn salience — IDF-WEIGHTED mean-pool over query tokens
#   (D agent方案2026-06-30): mean-pool把实体词信号被停用词稀释。按query token
#   的词频倒数 w_t=1/(count_in_query+1) 加权再pool, 近似BM25的IDF — 给稀有
#   (实体)词高权, 停用词近零权。纯推理零训练。qv改: q_r.mean → (q_r*w).sum/w.sum
# --------------------------------------------------------------------------- #


def score_S1idf_reader_attn(model, last_chunk, device, select_layer):
    """reader-attn salience [C] with IDF-weighted mean-pool query probe."""
    from transformers.models.llama.modeling_llama import apply_rotary_pos_emb
    from collections import Counter
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return None
    L = int(select_layer)
    if L < 0 or L >= len(mem_layers):
        return None
    sel = mem_layers[L]
    buf = getattr(sel, "_fifo_buf", None)
    if not buf:
        return None
    buf_snapshot = list(buf)
    C = len(buf_snapshot)
    cur = last_chunk.unsqueeze(0).to(device)
    out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None or L >= len(hs):
        return None
    q_hidden = hs[L]
    rot = None
    inner = getattr(root, "model", None)
    if inner is not None:
        rot = getattr(inner, "rotary_emb", None)
    if rot is None:
        rot = sel._fifo_resolve_rotary_emb()
    if rot is None:
        return None
    _attn = getattr(sel.wrapped_layer, "self_attn", None)
    if _attn is None:
        return None
    _pre_norm = getattr(sel.wrapped_layer, "input_layernorm", None)
    B, Tq, d = q_hidden.shape
    hd = _attn.head_dim
    _hs_q = _pre_norm(q_hidden) if _pre_norm is not None else q_hidden
    q = _attn.q_proj(_hs_q).view(B, Tq, -1, hd).transpose(1, 2)
    pos_ids = torch.arange(Tq, device=q_hidden.device).unsqueeze(0)
    cos, sin = rot(q_hidden, pos_ids)
    q_r, _ = apply_rotary_pos_emb(q, q, cos, sin)                  # [B, nh, Tq, hd]
    # IDF-like per-token weight from the query chunk's own token frequencies.
    toks = last_chunk.tolist()
    freq = Counter(toks)
    w = torch.tensor([1.0 / (freq[t] + 1.0) for t in toks],
                     device=q_r.device, dtype=torch.float32)        # [Tq]
    w = w[:Tq]
    if w.numel() < Tq:
        w = torch.cat([w, torch.ones(Tq - w.numel(), device=w.device)])
    w = w / w.sum().clamp(min=1e-6)
    qv = (q_r.float() * w[None, None, :, None]).sum(dim=2)          # [B, nh, hd]
    nh = qv.shape[1]
    sal_list = []
    for _kh in buf_snapshot:
        _kh_in = _kh.to(q_hidden.device, dtype=q_hidden.dtype)
        if _pre_norm is not None:
            _kh_in = _pre_norm(_kh_in)
        M = _kh_in.shape[1]
        kk = _attn.k_proj(_kh_in).view(B, M, -1, hd).transpose(1, 2)
        nkv = kk.shape[1]
        if nh != nkv:
            kk = kk.repeat_interleave(nh // nkv, dim=1)
        aw = torch.einsum("bhd,bhmd->bhm", qv, kk.float()) * (hd ** -0.5)
        aw = aw.amax(dim=1)
        sal_list.append(aw.amax(dim=-1).mean().float())
    return torch.stack(sal_list, dim=0).detach().float().cpu()


# --------------------------------------------------------------------------- #
# S2: lexical BM25 (unchanged)
# --------------------------------------------------------------------------- #

def score_S2_bm25(chunks, offset, C, question_ids, k1=1.5, b=0.75):
    """BM25 of the question token-IDs against each candidate chunk's token IDs."""
    if C <= 0:
        return None
    docs = [chunks[offset + i].tolist() for i in range(C)]
    N = C
    df = Counter()
    doc_tf = []
    doc_len = []
    for d in docs:
        c = Counter(d)
        doc_tf.append(c)
        doc_len.append(len(d))
        for t in c:
            df[t] += 1
    avgdl = (sum(doc_len) / N) if N > 0 else 0.0
    idf = {t: math.log((N - dft + 0.5) / (dft + 0.5) + 1.0) for t, dft in df.items()}
    qterms = set(int(t) for t in question_ids)

    sal = []
    for i in range(C):
        tf = doc_tf[i]
        dl = doc_len[i]
        s = 0.0
        for t in qterms:
            f = tf.get(t, 0)
            if f == 0:
                continue
            it = idf.get(t, 0.0)
            if avgdl > 0:
                denom = f + k1 * (1.0 - b + b * dl / avgdl)
            else:
                denom = f + k1
            s += it * (f * (k1 + 1.0)) / denom
        sal.append(s)
    return torch.tensor(sal, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Shared rank/recall
# --------------------------------------------------------------------------- #


def _rank_recall(sal, needle_local):
    if sal is None:
        return None, {k: "" for k in RECALL_KS}
    best_rank = min(int((sal > sal[j]).sum().item()) + 1 for j in needle_local)
    recalls = {k: (1 if best_rank <= k else 0) for k in RECALL_KS}
    return best_rank, recalls


# --------------------------------------------------------------------------- #
# Per-sample probe
# --------------------------------------------------------------------------- #


@torch.no_grad()
def probe_sample(model, input_ids, target, question, tokenizer, chunk_size,
                 device, select_layer, scorers, s3_max_new_tokens, s3_metric):
    """Run all enabled scorers on one encoded sample. Returns a list of CSV rows."""
    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)
    ingested = n_chunks - 1

    needle = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)

    _stream_context(model, chunks, device)
    _freeze(model)
    offset, C = _pool_from_buffer(model, n_chunks, select_layer)

    base = {
        "idx": None, "task": None, "length": None, "scorer": None,
        "n_chunks": n_chunks,
        "needle_chunks": "" if needle is None else ";".join(str(c) for c in sorted(needle)),
        "needle_located": needle is not None,
        "pool_offset": "" if offset is None else offset,
        "n_candidates": C,
        "needle_in_buffer": False,
        "needle_evicted": False,
        "needle_rank": "",
    }
    for k in RECALL_KS:
        base[f"recall@{k}"] = ""

    def _emit(scorer, rank=None, recalls=None, in_buf=None, evicted=None):
        row = dict(base)
        row["scorer"] = scorer
        if in_buf is not None:
            row["needle_in_buffer"] = in_buf
        if evicted is not None:
            row["needle_evicted"] = evicted
        if rank is not None:
            row["needle_rank"] = rank
        if recalls is not None:
            for k in RECALL_KS:
                row[f"recall@{k}"] = recalls[k]
        return row

    if C <= 0 or offset is None:
        _unfreeze(model)
        return [_emit(s) for s in scorers]

    needle_streamable = None
    needle_local = None
    status_evicted = False
    status_in_buf = False
    status_recall0 = False
    if needle is None:
        pass
    else:
        needle_streamable = sorted(c for c in needle if 0 <= c < ingested)
        if not needle_streamable:
            status_recall0 = True
        else:
            needle_local = [c - offset for c in needle_streamable
                            if 0 <= (c - offset) < C]
            if not needle_local:
                status_evicted = True
                status_recall0 = True
            else:
                status_in_buf = True

    rows = []

    need_ranking = status_in_buf
    sals = {}
    if need_ranking:
        last_chunk = chunks[-1]
        if "S1" in scorers:
            sals["S1"] = score_S1_reader_attn(model, last_chunk, device, select_layer)
        if "S1mp" in scorers:
            sals["S1mp"] = score_S1mp_reader_attn(model, last_chunk, device, select_layer)
        if "S1idf" in scorers:
            sals["S1idf"] = score_S1idf_reader_attn(model, last_chunk, device, select_layer)
        if "S2" in scorers:
            q_ids = tokenizer.encode((question or "").strip(), add_special_tokens=False)
            sals["S2"] = score_S2_bm25(chunks, offset, C, q_ids)
        # S1fuse: rank-fusion of mean-pool(S1mp) + bm25(S2). mean-pool强top1,
        # bm25强覆盖 → reciprocal-rank-fusion 各取所长(零训练)。需S1mp+S2都在。
        if "S1fuse" in scorers and sals.get("S1mp") is not None and sals.get("S2") is not None:
            import torch as _t
            def _rrf(sal):
                # sal[c]越大越好 → rank0=最高分; RRF score = 1/(60+rank)
                order = _t.argsort(sal, descending=True)
                rr = _t.zeros_like(sal, dtype=_t.float32)
                for r, c in enumerate(order.tolist()):
                    rr[c] = 1.0 / (60.0 + r)
                return rr
            sals["S1fuse"] = _rrf(sals["S1mp"]) + _rrf(sals["S2"])

    _unfreeze(model)

    for s in scorers:
        if status_in_buf:
            rank, recalls = _rank_recall(sals.get(s), needle_local)
            rows.append(_emit(s, rank=rank, recalls=recalls,
                              in_buf=True, evicted=False))
        elif status_recall0:
            recalls = {k: 0 for k in RECALL_KS}
            rows.append(_emit(s, recalls=recalls, in_buf=False,
                              evicted=status_evicted))
        else:
            rows.append(_emit(s))
    return rows


# --------------------------------------------------------------------------- #
# Encoding
# --------------------------------------------------------------------------- #


def _build_prompt_cfg(task):
    return {
        "instruction": DEFAULT_PROMPTS[task]["instruction"],
        "examples": DEFAULT_PROMPTS[task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
    }


def _encode_sample(sample, prompt_cfg, tokenizer):
    input_text = get_formatted_input(
        sample["input"],
        sample["question"],
        prompt_cfg["examples"],
        prompt_cfg["instruction"],
        prompt_cfg["post_prompt"],
        template=prompt_cfg["template"],
    )
    ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
    if isinstance(ids, list):
        ids = torch.tensor([ids], dtype=torch.long)
    return sample["target"], sample["question"], ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="E2 mean-pool salience ablation probe")
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--adapter_config", required=True)
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--tasks", nargs="+", default=["qa5"])
    ap.add_argument("--lengths", nargs="+", default=["16k"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--select_layer", type=int, default=16)
    ap.add_argument("--scorers", default="S1,S1mp,S2",
                    help="comma-separated; S1=last-token, S1mp=mean-pool, S2=bm25")
    ap.add_argument("--keep_all", action="store_true")
    # s3/s4 args kept for interface parity but disabled
    ap.add_argument("--s3_max_new_tokens", type=int, default=6)
    ap.add_argument("--s3_metric", default="maxprob", choices=["maxprob", "negentropy"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--output_csv", default=None)
    args = ap.parse_args()

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    bad = [s for s in scorers if s not in ALL_SCORERS]
    if bad:
        ap.error(f"unknown scorer(s) {bad}; choose from {ALL_SCORERS}")
    if not scorers:
        ap.error("--scorers is empty")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[E2-meanpool] Configuration:")
    print(f"  model_path     {args.model_path}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  adapter_config {args.adapter_config}")
    print(f"  tasks          {args.tasks}")
    print(f"  lengths        {args.lengths}")
    print(f"  scorers        {scorers}")
    print(f"  limit          {args.limit}")
    print(f"  device         {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config: MemorySpaceConfig = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size
    fifo_cap = int(getattr(mem_config, "fifo_buffer_chunks", 25))
    print(f"[E2-meanpool] use_fifo_memory={getattr(mem_config, 'use_fifo_memory', False)} "
          f"fifo_buffer_chunks(cap)={fifo_cap}")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )
    model.eval()

    if args.keep_all:
        _set_fifo_keep_all_buffer(model, True)
        print("[E2-meanpool] keep_all: FIFO eviction suppressed")

    out_path = args.output_csv
    if out_path is None:
        tag = f"L{args.select_layer}_cs{args.chunk_size}"
        if args.keep_all:
            tag += "_keepall"
        out_path = os.path.join(os.getcwd(), f"e2_meanpool_{tag}.csv")
    out_path = os.path.abspath(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["idx", "task", "length", "scorer", "n_chunks", "needle_chunks",
         "needle_located", "pool_offset", "n_candidates", "needle_in_buffer",
         "needle_evicted", "needle_rank"]
        + [f"recall@{k}" for k in RECALL_KS]
    )
    all_rows = []
    fcsv = open(out_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    for task in args.tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"[E2-meanpool] WARNING task {task} not in DEFAULT_PROMPTS; skipping")
            continue
        prompt_cfg = _build_prompt_cfg(task)
        for length in args.lengths:
            try:
                data = load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:
                print(f"[E2-meanpool] ERROR loading {args.dataset_name}/{length}/{task}: {e}")
                continue
            n = len(task_data)
            if args.limit > 0:
                n = min(n, args.limit)
            sample_indices = list(range(n))[args.shard_index::args.num_shards]
            print(f"\n[E2-meanpool] task={task} length={length}: {len(sample_indices)} samples")

            for idx in tqdm(sample_indices, desc=f"{task}/{length}", leave=False):
                target, question, input_ids = _encode_sample(task_data[idx], prompt_cfg, tokenizer)
                input_ids = input_ids.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    sample_rows = probe_sample(
                        model, input_ids, target, question, tokenizer,
                        args.chunk_size, device, args.select_layer,
                        scorers, args.s3_max_new_tokens, args.s3_metric,
                    )
                for row in sample_rows:
                    row["idx"] = idx
                    row["task"] = task
                    row["length"] = length
                    writer.writerow(row)
                    all_rows.append(row)
                fcsv.flush()

    fcsv.close()
    print(f"\n[E2-meanpool] Wrote {len(all_rows)} rows to {out_path}")

    _print_summary(all_rows, scorers)


def _print_summary(rows, scorers):
    print("\n" + "=" * 92)
    print("[E2-meanpool] SUMMARY  (mean recall@K / median rank over LOCATED samples)")
    print("  Scorers: S1=last-token  S1mp=mean-pool  S2=bm25-lexical")
    print("  recall denominator = located samples (evicted count as 0).")
    print("=" * 92)

    groups = defaultdict(list)
    for r in rows:
        groups[(r["scorer"], r["task"], r["length"])].append(r)

    header = (f"{'scorer':<8}{'task':<6}{'len':<6}{'N':>5}{'loc':>5}{'evct':>5}"
              + "".join(f"{'r@'+str(k):>8}" for k in RECALL_KS)
              + f"{'medRank':>9}{'nCand':>7}")
    print(header)
    print("-" * len(header))

    def _fmt(scorer, task, length, grp):
        N = len(grp)
        located = [r for r in grp if r["needle_located"]]
        nloc = len(located)
        nevct = sum(1 for r in located if r["needle_evicted"])
        line = f"{scorer:<8}{task:<6}{length:<6}{N:>5}{nloc:>5}{nevct:>5}"
        for k in RECALL_KS:
            vals = [r[f"recall@{k}"] for r in located if isinstance(r[f"recall@{k}"], int)]
            line += f"{(sum(vals)/len(vals)):>8.2f}" if vals else f"{'-':>8}"
        ranks = [r["needle_rank"] for r in located if isinstance(r["needle_rank"], int)]
        line += f"{median(ranks):>9.1f}" if ranks else f"{'-':>9}"
        ncands = [r["n_candidates"] for r in grp if isinstance(r["n_candidates"], int) and r["n_candidates"] > 0]
        line += f"{(sum(ncands)/len(ncands)):>7.1f}" if ncands else f"{'-':>7}"
        print(line)

    for (scorer, task, length) in sorted(groups.keys()):
        _fmt(scorer, task, length, groups[(scorer, task, length)])

    print("-" * len(header))
    by_sl = defaultdict(list)
    for r in rows:
        by_sl[(r["scorer"], r["length"])].append(r)
    for (scorer, length) in sorted(by_sl.keys()):
        _fmt(scorer, "ALL", length, by_sl[(scorer, length)])

    print("=" * 92)
    print("[E2-meanpool] INTERPRETATION:")
    print("  If S1mp recall@4 >> S1 recall@4 => last-token hypothesis CONFIRMED.")
    print("  If S1mp ~= S1 => last-token is NOT the bottleneck, look elsewhere.")
    print("  S2 (BM25) is the lexical ceiling for comparison.")


if __name__ == "__main__":
    main()
