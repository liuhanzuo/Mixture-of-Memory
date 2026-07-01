"""E2-multiscorer — needle-recall probe with several ZERO-TRAINING scorers.

Extends ``e2_needle_recall_probe.py``. Same question (does any zero-training
signal pick the true needle chunk out of the long-doc candidate pool?), but now
we score the SAME per-sample candidate-chunk pool with several different signals
and report needle_rank + recall@{1,2,4,8,16} per scorer, so the long-doc
"selection wall" can be attributed to the SCORER, not the pool.

This is a standalone, pure-eval probe. It imports loaders/locators from
``run_babilong_mem_space.py`` and the layer's own salience fn, and NEVER mutates
the checkpoint or the main eval/train path.

Scorers (all over the identical candidate pool for a given sample)
-----------------------------------------------------------------
  S1  reader-attn salience      reuse layer.py:_fifo_reader_attn_salience — the
                                 EXACT q.k signal the deployed selector ranks on
                                 (baseline; identical to e2_needle_recall_probe).
  S2  lexical BM25              pure-CPU BM25 of the question's token IDs against
                                 each candidate chunk's raw token IDs (corpus =
                                 the candidate pool). No model forward.
  S3  reforward-confidence      query-aware: re-forward [chunk_c | question] (the
                                 chunk's RAW tokens + the last/question chunk) as
                                 a plain causal window (FIFO buffer emptied so it
                                 is byte-identical to a vanilla Llama forward),
                                 greedily generate the answer, score = mean over
                                 generated steps of the model's own confidence
                                 (max softmax prob, or neg-entropy). One mini
                                 generation per candidate chunk (expensive).
  S4  reforward-oracle (UB)     same window as S3 but teacher-force the GOLD
                                 answer and score = mean gold-token logprob. A
                                 CHEATING upper bound: it answers "does the
                                 reforward logit signal even CONTAIN the needle?"

Candidate pool
--------------
The pool is the FIFO buffer at ``select_layer`` after the context chunks are
streamed (post-eviction; cap = fifo_buffer_chunks, eval default 25), mapped to
document-absolute chunk indices via the same offset arithmetic as the deployed
selector (``offset = (n_chunks-1) - len(buf)``; doc-abs idx i_local -> offset+i).
``--keep_all`` suppresses eviction (``_set_fifo_keep_all_buffer``) so the pool is
EVERY context chunk (0 .. n_chunks-2). All scorers see the SAME pool, so
needle_rank/recall are directly comparable across scorers for a sample.

Per (sample, scorer) we emit a CSV row with needle_rank + recall@K + the shared
pool/eviction bookkeeping (n_candidates, needle_in_buffer, needle_evicted).
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

# --------------------------------------------------------------------------- #
# Environment: offline HF + local arrow cache. MUST be set before importing the
# run module (which imports `datasets` at module top).
# --------------------------------------------------------------------------- #
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

# Reuse — do NOT re-implement — the loaders/locators from the main eval script.
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
ALL_SCORERS = ("S1", "S2", "S3", "S4")


# --------------------------------------------------------------------------- #
# Pool / streaming helpers
# --------------------------------------------------------------------------- #


def _stream_context(model, chunks, device):
    """Stream chunks[:-1] into the FIFO buffers (memory accumulation only).
    Byte-identical to the streaming loop in ``generate_with_mem_space``."""
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
    """Read the candidate pool from the select_layer FIFO buffer (post-stream).

    Returns (offset, C) where C == len(buf) at select_layer and
    offset = (n_chunks-1) - C is the document-absolute index of buffer-local 0.
    Returns (None, 0) when no buffer (e.g. single-chunk doc)."""
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
# S1: reader-attn salience (reuses the layer's exact salience function)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_S1_reader_attn(model, last_chunk, device, select_layer):
    """Full-buffer reader-attn q.k salience [C] (buffer-local == doc-abs order).

    Identical mechanism to e2_needle_recall_probe.score_all_chunks_reader_attn:
    the scoring uses the PRE-forward buffer snapshot (the extra last-chunk forward
    FIFO-writes the last chunk into every layer's buffer, so we score the snapshot
    taken before it). Returns a [C] CPU float tensor or None."""
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
    q_hidden = hs[L]  # [1, last_len, d] = input to layer L

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
# S2: lexical BM25 (pure CPU, no model)
# --------------------------------------------------------------------------- #


def score_S2_bm25(chunks, offset, C, question_ids, k1=1.5, b=0.75):
    """BM25 of the question token-IDs against each candidate chunk's token IDs.

    Corpus = the C candidate chunks (so IDF is over the candidate pool). Query =
    the (de-duplicated) question token IDs. Returns a [C] CPU float tensor. Pure
    CPU; no model forward. doc-abs order == buffer order (ascending)."""
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
# S3: reforward-confidence (query-aware mini re-forward, per chunk)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_S3_reforward_conf(model, chunks, offset, C, last_chunk, device,
                            tokenizer, max_new_tokens, metric):
    """Per candidate chunk c: re-forward window = [chunks[c] | last_chunk] as a
    PLAIN causal sequence (FIFO buffer emptied before every forward so P==0 ->
    the wrapped layer runs vanilla, no memory prefix), greedily generate up to
    ``max_new_tokens``, and score the model's own confidence over the generated
    steps. metric='maxprob' -> mean of per-step max softmax prob; 'negentropy'
    -> mean of per-step negative entropy. Higher == more confident. [C] tensor."""
    if C <= 0:
        return None
    eos = tokenizer.eos_token_id
    sal = []
    for i in range(C):
        window = torch.cat([chunks[offset + i], last_chunk], dim=0)
        cur = window.unsqueeze(0).to(device)
        scores = []
        for step in range(max_new_tokens):
            _reset_fifo_memory(model)  # empty buffer -> isolated plain forward
            out = model(input_ids=cur, use_cache=False)
            logits = out.logits[:, -1, :].float()  # [1, V]
            if step == 0 and eos is not None:
                logits[:, eos] = float("-inf")  # match deployed: no empty answer
            probs = torch.softmax(logits, dim=-1)
            if metric == "negentropy":
                ent = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1)
                scores.append(float((-ent).item()))
            else:  # maxprob
                scores.append(float(probs.max().item()))
            nxt = logits.argmax(dim=-1, keepdim=True)
            tok = int(nxt.item())
            if eos is not None and tok == eos and step > 0:
                break
            cur = torch.cat([cur, nxt], dim=-1)
        sal.append(sum(scores) / len(scores) if scores else 0.0)
    _reset_fifo_memory(model)  # leave buffers clean for the caller
    return torch.tensor(sal, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# S4: reforward-oracle (cheating upper bound: gold-answer logprob)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_S4_reforward_oracle(model, chunks, offset, C, last_chunk, gold_ids, device):
    """Per candidate chunk c: same window as S3 = [chunks[c] | last_chunk], but
    teacher-force the GOLD answer tokens right after the window and score = mean
    gold-token logprob (logits at position predicting each gold token). FIFO
    buffer emptied so the forward is plain causal. CHEATING upper bound. [C]
    tensor. Higher (less negative) == chunk better supports the gold answer."""
    if C <= 0 or not gold_ids:
        return None
    gold = torch.tensor(list(gold_ids), dtype=torch.long, device=device)
    G = gold.shape[0]
    sal = []
    for i in range(C):
        window = torch.cat([chunks[offset + i], last_chunk], dim=0).to(device)
        Wlen = int(window.shape[0])
        full = torch.cat([window, gold], dim=0).unsqueeze(0)
        _reset_fifo_memory(model)
        out = model(input_ids=full, use_cache=False)
        logits = out.logits[0].float()  # [S, V]
        logp = torch.log_softmax(logits, dim=-1)
        lps = []
        for j in range(G):
            pos = Wlen + j - 1  # position whose logits predict gold token j
            if 0 <= pos < logp.shape[0]:
                lps.append(float(logp[pos, int(gold[j].item())].item()))
        sal.append(sum(lps) / len(lps) if lps else float("-inf"))
    _reset_fifo_memory(model)
    return torch.tensor(sal, dtype=torch.float32)


# --------------------------------------------------------------------------- #
# Shared rank/recall over a salience vector for the located-needle case
# --------------------------------------------------------------------------- #


def _rank_recall(sal, needle_local):
    """Competition rank of the BEST in-buffer needle chunk + recall@K dict.
    rank(j) = (# chunks scoring STRICTLY higher than needle j) + 1."""
    if sal is None:
        return None, {k: "" for k in RECALL_KS}
    best_rank = min(int((sal > sal[j]).sum().item()) + 1 for j in needle_local)
    recalls = {k: (1 if best_rank <= k else 0) for k in RECALL_KS}
    return best_rank, recalls


# --------------------------------------------------------------------------- #
# Per-sample probe (one row per enabled scorer)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def probe_sample(model, input_ids, target, question, tokenizer, chunk_size,
                 device, select_layer, scorers, s3_max_new_tokens, s3_metric):
    """Run all enabled scorers on one encoded sample. Returns a list of CSV
    rows, one per scorer."""
    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)  # CRITICAL: buffer holds ONLY this doc's chunks

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)
    ingested = n_chunks - 1

    needle = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)

    # ---- stream context, then read the candidate pool from the buffer ----
    _stream_context(model, chunks, device)
    _freeze(model)
    offset, C = _pool_from_buffer(model, n_chunks, select_layer)

    # ---- shared per-sample bookkeeping (scorer-independent) ----
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

    # ---- no candidate pool (single-chunk doc / failure): blank rows ----
    if C <= 0 or offset is None:
        _unfreeze(model)
        return [_emit(s) for s in scorers]

    # ---- needle status (scorer-independent), mirrors e2 logic ----
    needle_streamable = None
    needle_local = None
    status_evicted = False
    status_in_buf = False
    status_recall0 = False  # needle exists but cannot be in pool -> recall 0
    if needle is None:
        # answer not located in tokens -> measurement gap; leave rank/recall blank
        pass
    else:
        needle_streamable = sorted(c for c in needle if 0 <= c < ingested)
        if not needle_streamable:
            # needle only in the (un-streamed) last/question chunk -> not a candidate
            status_recall0 = True
        else:
            needle_local = [c - offset for c in needle_streamable
                            if 0 <= (c - offset) < C]
            if not needle_local:
                # streamed but evicted from the capped buffer -> recall 0
                status_evicted = True
                status_recall0 = True
            else:
                status_in_buf = True

    rows = []

    # ---- compute the salience vectors only when we actually need a ranking ----
    need_ranking = status_in_buf
    sals = {}
    if need_ranking:
        last_chunk = chunks[-1]
        if "S1" in scorers:
            sals["S1"] = score_S1_reader_attn(model, last_chunk, device, select_layer)
        if "S2" in scorers:
            q_ids = tokenizer.encode((question or "").strip(), add_special_tokens=False)
            sals["S2"] = score_S2_bm25(chunks, offset, C, q_ids)
        if "S3" in scorers:
            sals["S3"] = score_S3_reforward_conf(
                model, chunks, offset, C, last_chunk, device,
                tokenizer, s3_max_new_tokens, s3_metric,
            )
        if "S4" in scorers:
            gold = tokenizer.encode(" " + (target or "").strip(), add_special_tokens=False)
            if not gold:
                gold = tokenizer.encode((target or "").strip(), add_special_tokens=False)
            sals["S4"] = score_S4_reforward_oracle(
                model, chunks, offset, C, last_chunk, gold, device,
            )

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
            # needle not located -> blank rank/recall
            rows.append(_emit(s))
    return rows


# --------------------------------------------------------------------------- #
# Encoding (mirrors run_babilong_mem_space._encode_sample, + question)
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
    ap = argparse.ArgumentParser(description="E2 multi-scorer needle-recall probe")
    ap.add_argument("--model_path", default="models/Meta-Llama-3-8B", help="Base Llama-3-8B dir")
    ap.add_argument("--checkpoint", required=True, help="mem_space adapter .pt (clean FIFO ckpt)")
    ap.add_argument("--adapter_config", required=True, help="adapter_config.json")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--tasks", nargs="+", default=["qa1"])
    ap.add_argument("--lengths", nargs="+", default=["16k", "32k"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=50, help="max samples per task/length (-1 = all)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--select_layer", type=int, default=16,
                    help="wrapped decoder layer whose buffer defines the pool / drives S1")
    ap.add_argument("--scorers", default="S1,S2,S3,S4",
                    help="comma-separated subset of S1,S2,S3,S4")
    ap.add_argument("--keep_all", action="store_true",
                    help="suppress FIFO eviction so the pool = ALL context chunks (no cap)")
    ap.add_argument("--s3_max_new_tokens", type=int, default=6,
                    help="S3 mini-generation length per candidate chunk")
    ap.add_argument("--s3_metric", default="maxprob", choices=["maxprob", "negentropy"],
                    help="S3 confidence metric")
    ap.add_argument("--batch_size", type=int, default=1,
                    help="MUST be 1 (reforward path); kept for interface parity")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--output_csv", default=None,
                    help="output CSV path (default: e2_multiscorer_<tag>.csv in cwd)")
    args = ap.parse_args()

    if args.batch_size != 1:
        ap.error("--batch_size must be 1 (the S3/S4 reforward path is per-sample)")
    if args.num_shards < 1:
        ap.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(f"--shard_index must be in [0, {args.num_shards})")

    scorers = [s.strip() for s in args.scorers.split(",") if s.strip()]
    bad = [s for s in scorers if s not in ALL_SCORERS]
    if bad:
        ap.error(f"unknown scorer(s) {bad}; choose from {ALL_SCORERS}")
    if not scorers:
        ap.error("--scorers is empty")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[E2-multi] Configuration:")
    print(f"  model_path     {args.model_path}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  adapter_config {args.adapter_config}")
    print(f"  tasks          {args.tasks}")
    print(f"  lengths        {args.lengths}")
    print(f"  chunk_size     {args.chunk_size}")
    print(f"  select_layer   {args.select_layer}")
    print(f"  scorers        {scorers}")
    print(f"  keep_all       {args.keep_all}")
    print(f"  s3_max_new_tok {args.s3_max_new_tokens}  s3_metric {args.s3_metric}")
    print(f"  limit          {args.limit}")
    if args.num_shards > 1:
        print(f"  shard          {args.shard_index}/{args.num_shards}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config: MemorySpaceConfig = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size  # mirror training (run script:1362)
    fifo_cap = int(getattr(mem_config, "fifo_buffer_chunks", 25))
    print(f"[E2-multi] use_fifo_memory={getattr(mem_config, 'use_fifo_memory', False)} "
          f"fifo_buffer_chunks(cap)={fifo_cap}")
    if not getattr(mem_config, "use_fifo_memory", False):
        print("[E2-multi] WARNING: adapter is NOT a FIFO model. The buffer-defined "
              "candidate pool only exists on FIFO ckpts; results will be empty.")

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
        print("[E2-multi] keep_all: FIFO eviction suppressed (pool = all context chunks)")

    out_path = args.output_csv
    if out_path is None:
        tag = f"L{args.select_layer}_cs{args.chunk_size}"
        if args.keep_all:
            tag += "_keepall"
        if args.num_shards > 1:
            tag += f"_shard{args.shard_index}of{args.num_shards}"
        out_path = os.path.join(os.getcwd(), f"e2_multiscorer_{tag}.csv")
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
            print(f"[E2-multi] WARNING task {task} not in DEFAULT_PROMPTS; skipping")
            continue
        prompt_cfg = _build_prompt_cfg(task)
        for length in args.lengths:
            try:
                data = load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:
                print(f"[E2-multi] ERROR loading {args.dataset_name}/{length}/{task}: {e}")
                continue
            n = len(task_data)
            if args.limit > 0:
                n = min(n, args.limit)
            sample_indices = list(range(n))[args.shard_index::args.num_shards]
            print(f"\n[E2-multi] task={task} length={length}: {len(sample_indices)} samples")

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
    print(f"\n[E2-multi] Wrote {len(all_rows)} rows to {out_path}")

    _print_summary(all_rows, scorers)


def _print_summary(rows, scorers):
    print("\n" + "=" * 92)
    print("[E2-multi] SUMMARY (mean recall@K / median rank over LOCATED samples)")
    print("           recall denominator = located samples (evicted count as 0).")
    print("=" * 92)

    groups = defaultdict(list)  # (scorer, task, length) -> rows
    for r in rows:
        groups[(r["scorer"], r["task"], r["length"])].append(r)

    header = (f"{'scorer':<7}{'task':<6}{'len':<6}{'N':>5}{'loc':>5}{'evct':>5}"
              + "".join(f"{'r@'+str(k):>8}" for k in RECALL_KS)
              + f"{'medRank':>9}{'nCand':>7}")
    print(header)
    print("-" * len(header))

    def _fmt(scorer, task, length, grp):
        N = len(grp)
        located = [r for r in grp if r["needle_located"]]
        nloc = len(located)
        nevct = sum(1 for r in located if r["needle_evicted"])
        line = f"{scorer:<7}{task:<6}{length:<6}{N:>5}{nloc:>5}{nevct:>5}"
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

    # per-(scorer,length) aggregate across tasks
    print("-" * len(header))
    by_sl = defaultdict(list)
    for r in rows:
        by_sl[(r["scorer"], r["length"])].append(r)
    for (scorer, length) in sorted(by_sl.keys()):
        _fmt(scorer, "ALL", length, by_sl[(scorer, length)])

    print("=" * 92)
    print("[E2-multi] Read: compare recall@K ACROSS scorers at fixed (task,length). "
          "S4 (oracle) is the upper bound of how much needle info the reforward "
          "logit carries; if S2/S3 approach S4 they beat the S1 wall.")


if __name__ == "__main__":
    main()
