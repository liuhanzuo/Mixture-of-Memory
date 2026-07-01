"""E2 — needle-recall diagnostic probe for the deployable reader-attn selector.

ZERO-TRAINING, pure-eval probe (NO checkpoint mutation, NO change to the main
eval/train path). Standalone script: imports the helpers it reuses from
``run_babilong_mem_space.py`` and never invokes its ``main()``.

Question E2 answers
-------------------
The deployed long-doc selector picks the top-K FIFO chunks by the reader's own
native q.k salience at one model-level selection layer (L16), then re-forwards
those chunks' raw tokens. On long docs this under-selects (K4 16k score ~38 vs
oracle-perfect-select ~70 → a 32-pt selection gap). E2 measures the gap at its
SOURCE: after the reader-attn scorer ranks EVERY buffered chunk, where does the
TRUE needle chunk land?

For every sample we report:
  * needle_rank   : 1-based competition rank of the (best) needle chunk among
                    all buffered candidate chunks (1 == top-scored).
  * recall@{1,2,4,8,16} : 1 if the needle chunk is within the top-K, else 0.
  * n_candidates  : size of the FIFO buffer the selector scored over (post-
                    eviction, cap = fifo_buffer_chunks, eval default 25).
  * needle_evicted: True if the needle's document-absolute chunk was located in
                    the input but is NOT in the (post-eviction) buffer — the
                    selector physically cannot pick it (it is the part of the
                    gap that NO re-ranking can fix; it needs a bigger buffer).

GO / NO-GO read
---------------
  recall@16 high (>70%, evicted excluded) → the selector keeps the needle in its
      candidate set and merely ranks it too low → fixable by widening K / re-rank.
  recall@16 ~= chance (~ K/N) → reader-attn salience does not separate the needle
      from distractors on long docs → an information wall; selection cannot be
      "saved" without a different scorer.

Mechanism (identical scorer to the deployed selector)
-----------------------------------------------------
We reuse the EXACT q.k salience the deployable selector ranks on:
  * stream all-but-last chunks → every FIFO layer's ``_fifo_buf`` holds the
    context chunks' hidden snapshots (with eviction at cap),
  * query = the last (question) chunk's hidden at the INPUT to ``select_layer``
    (one extra ``output_hidden_states=True`` forward; buffer snapshot/restored
    around it so the extra forward's FIFO-write does not pollute the candidate
    buffer — same trick as ``_select_chunks_reader_attn``),
  * score = ``MemorySpaceLayer._fifo_reader_attn_salience`` — the no-top-k twin
    of ``_fifo_select_keep_set_reader_attn`` that returns the FULL [C] salience
    vector (same q_proj/k_proj + RoPE-on-query + per-chunk amax pooling). We then
    rank/recall ourselves instead of taking a top-k.

Buffer -> document-absolute chunk mapping (copied from _select_chunks_reader_attn)
    document_chunk_index(buffer_local_i) = ingested - len(buf) + i,
    ingested = n_chunks - 1 (context chunks streamed; the last/question chunk is
    never streamed). Exact for both the no-eviction (offset 0) and evicted
    (offset > 0, oldest chunks dropped) cases.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
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
)

RECALL_KS = (1, 2, 4, 8, 16)


# --------------------------------------------------------------------------- #
# Reader-attn full-buffer scoring (reuses the layer's exact salience function)
# --------------------------------------------------------------------------- #


@torch.no_grad()
def score_all_chunks_reader_attn(model, last_chunk, n_chunks, device, select_layer):
    """Rank EVERY buffered chunk by the deployed reader-attn salience.

    Must be called AFTER the streaming-ingestion loop (so ``_fifo_buf`` on every
    layer holds chunks 0..n_chunks-2 with eviction) and AFTER ``_freeze_banks``.

    Returns a dict with:
        sal           : [C] float tensor (CPU), buffer-local salience.
        offset        : ingested - C (document-abs idx of buffer-local 0).
        n_candidates  : C (= len(buf) at the selection layer).
    or None on failure.
    """
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
        return None  # nothing streamed (short doc) — no candidates

    # Snapshot the selection layer's buffer: the extra last-chunk forward below
    # FIFO-writes the last chunk into every layer's _fifo_buf (the write at
    # layer.py:1249 is not frozen-gated). We score against the PRE-forward
    # buffer (context chunks only), exactly like the deployed selector.
    buf_snapshot = list(buf)
    C = len(buf_snapshot)

    # ---- query hidden at the input to layer L (= out.hidden_states[L]) ----
    cur = last_chunk.unsqueeze(0).to(device)  # [1, last_len]
    out = model(input_ids=cur, use_cache=False, output_hidden_states=True)
    hs = getattr(out, "hidden_states", None)
    if hs is None or L >= len(hs):
        return None
    q_hidden = hs[L]  # [1, last_len, d]

    # ---- RoPE cos/sin for the last-chunk positions 0..Tq-1 ----
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

    # ---- full per-chunk salience (no top-k, no recency floor) ----
    # _fifo_reader_attn_salience is the grad-bearing twin of the deployed
    # _fifo_select_keep_set_reader_attn scorer; it returns the FULL [C] vector.
    # Under no_grad the numbers are identical to the eval selector's.
    sal = sel._fifo_reader_attn_salience(
        hidden_states=q_hidden,
        chunk_hiddens=buf_snapshot,
        position_embeddings=(cos, sin),
    )
    if sal is None or sal.shape[0] != C:
        return None

    ingested = n_chunks - 1
    offset = ingested - C
    return {"sal": sal.detach().float().cpu(), "offset": int(offset), "n_candidates": int(C)}


def _stream_context(model, chunks, device):
    """Stream chunks[:-1] into the FIFO buffers (memory accumulation only).

    Byte-identical to the streaming loop in ``generate_with_mem_space``.
    """
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


# --------------------------------------------------------------------------- #
# Per-sample probe
# --------------------------------------------------------------------------- #


@torch.no_grad()
def probe_sample(model, input_ids, target, tokenizer, chunk_size, device, select_layer):
    """Run E2 on one encoded sample. Returns a result dict (one CSV row)."""
    _reset_banks(model)
    _reset_l2(model)
    _reset_fifo_memory(model)  # CRITICAL: buffer must hold ONLY this doc's chunks

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)

    needle = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)

    _stream_context(model, chunks, device)
    _freeze(model)
    try:
        scored = score_all_chunks_reader_attn(
            model, chunks[-1], n_chunks, device, select_layer
        )
    finally:
        _unfreeze(model)

    row = {
        "task": None,
        "length": None,
        "idx": None,
        "n_chunks": n_chunks,
        "needle_chunks": "" if needle is None else ";".join(str(c) for c in sorted(needle)),
        "needle_located": needle is not None,
        "n_candidates": 0 if scored is None else scored["n_candidates"],
        "needle_in_buffer": False,
        "needle_evicted": False,
        "needle_rank": "",
    }
    for k in RECALL_KS:
        row[f"recall@{k}"] = ""

    if scored is None:
        # No candidates scored (e.g. single-chunk doc). Nothing the selector can
        # rank; leave rank/recall blank.
        return row

    sal = scored["sal"]
    offset = scored["offset"]
    C = scored["n_candidates"]
    ingested = n_chunks - 1

    if needle is None:
        # Answer could not be located in the token stream — measurement gap, not
        # an eviction. Recall/rank left blank; counted separately in the summary.
        return row

    # Needle document-absolute indices that fall in the STREAMABLE range
    # [0, ingested). (A needle that only sits in the last/question chunk is not a
    # candidate the buffer ever holds.)
    needle_streamable = sorted(c for c in needle if 0 <= c < ingested)
    if not needle_streamable:
        # Needle only in the (un-streamed) last chunk → not a selector candidate.
        # Treat as not-in-buffer but NOT eviction (it was never streamable).
        row["needle_in_buffer"] = False
        row["needle_evicted"] = False
        for k in RECALL_KS:
            row[f"recall@{k}"] = 0
        return row

    # Map needle doc-abs idx -> buffer-local idx; in-buffer iff in [offset, offset+C).
    needle_local = [c - offset for c in needle_streamable if 0 <= (c - offset) < C]
    if not needle_local:
        # Needle was streamed but evicted from the (capped) buffer → selector
        # physically cannot pick it. This is the floor no re-ranking can fix.
        row["needle_evicted"] = True
        row["needle_in_buffer"] = False
        for k in RECALL_KS:
            row[f"recall@{k}"] = 0
        return row

    row["needle_in_buffer"] = True
    # Competition rank: # chunks scoring STRICTLY higher than the needle + 1.
    # Take the BEST (lowest) rank over all in-buffer needle chunks (multi-mention
    # / boundary-straddling needles).
    best_rank = min(int((sal > sal[j]).sum().item()) + 1 for j in needle_local)
    row["needle_rank"] = best_rank
    for k in RECALL_KS:
        row[f"recall@{k}"] = 1 if best_rank <= k else 0
    return row


# --------------------------------------------------------------------------- #
# Encoding (mirrors run_babilong_mem_space._encode_sample)
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
    return sample["target"], ids


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main():
    ap = argparse.ArgumentParser(description="E2 needle-recall probe for the reader-attn selector")
    ap.add_argument("--model_path", required=True, help="Base Llama-3-8B dir")
    ap.add_argument("--checkpoint", required=True, help="mem_space adapter .pt (clean FIFO ckpt)")
    ap.add_argument("--adapter_config", required=True, help="adapter_config.json")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa5"])
    ap.add_argument("--lengths", nargs="+", default=["8k", "16k", "32k"])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=100, help="max samples per task/length (-1 = all)")
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--select_layer", type=int, default=16,
                    help="wrapped decoder layer whose native q.k drives selection (deployed=16)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", default="sdpa", choices=["sdpa", "eager", "flash_attention_2"])
    ap.add_argument("--output_csv", default=None,
                    help="output CSV path (default: e2_recall_<tag>.csv in cwd)")
    args = ap.parse_args()

    if args.num_shards < 1:
        ap.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        ap.error(f"--shard_index must be in [0, {args.num_shards})")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    print("[E2] Configuration:")
    print(f"  model_path     {args.model_path}")
    print(f"  checkpoint     {args.checkpoint}")
    print(f"  adapter_config {args.adapter_config}")
    print(f"  tasks          {args.tasks}")
    print(f"  lengths        {args.lengths}")
    print(f"  chunk_size     {args.chunk_size}")
    print(f"  select_layer   {args.select_layer}")
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
    print(f"[E2] use_fifo_memory={getattr(mem_config, 'use_fifo_memory', False)} "
          f"fifo_buffer_chunks(cap)={fifo_cap}")
    if not getattr(mem_config, "use_fifo_memory", False):
        print("[E2] WARNING: adapter is NOT a FIFO model (use_fifo_memory=False). "
              "The reader-attn buffer selector only exists on FIFO ckpts; results "
              "will be empty.")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
        attn_impl=args.attn_impl,
    )

    out_path = args.output_csv
    if out_path is None:
        tag = f"L{args.select_layer}_cs{args.chunk_size}"
        if args.num_shards > 1:
            tag += f"_shard{args.shard_index}of{args.num_shards}"
        out_path = os.path.join(os.getcwd(), f"e2_recall_{tag}.csv")
    out_path = os.path.abspath(out_path)
    Path(os.path.dirname(out_path)).mkdir(parents=True, exist_ok=True)

    fieldnames = (
        ["idx", "task", "length", "n_chunks", "needle_chunks", "needle_located",
         "n_candidates", "needle_in_buffer", "needle_evicted", "needle_rank"]
        + [f"recall@{k}" for k in RECALL_KS]
    )
    all_rows = []
    fcsv = open(out_path, "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fieldnames)
    writer.writeheader()

    for task in args.tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"[E2] WARNING task {task} not in DEFAULT_PROMPTS; skipping")
            continue
        prompt_cfg = _build_prompt_cfg(task)
        for length in args.lengths:
            try:
                data = load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:
                print(f"[E2] ERROR loading {args.dataset_name}/{length}/{task}: {e}")
                continue
            n = len(task_data)
            if args.limit > 0:
                n = min(n, args.limit)
            sample_indices = list(range(n))[args.shard_index::args.num_shards]
            print(f"\n[E2] task={task} length={length}: {len(sample_indices)} samples")

            for idx in tqdm(sample_indices, desc=f"{task}/{length}", leave=False):
                target, input_ids = _encode_sample(task_data[idx], prompt_cfg, tokenizer)
                input_ids = input_ids.to(device)
                with torch.amp.autocast(device_type="cuda", dtype=dtype):
                    row = probe_sample(
                        model, input_ids, target, tokenizer,
                        args.chunk_size, device, args.select_layer,
                    )
                row["idx"] = idx
                row["task"] = task
                row["length"] = length
                writer.writerow(row)
                fcsv.flush()
                all_rows.append(row)

    fcsv.close()
    print(f"\n[E2] Wrote {len(all_rows)} rows to {out_path}")

    _print_summary(all_rows, args)


def _print_summary(rows, args):
    print("\n" + "=" * 78)
    print("[E2] SUMMARY (mean recall@K / median rank, computed over LOCATED samples)")
    print("     recall@K denominator = located samples (evicted count as recall=0,")
    print("     since the selector physically cannot pick an evicted needle).")
    print("=" * 78)
    # Group by (task, length).
    from collections import defaultdict
    groups = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["length"])].append(r)

    header = (f"{'task':<6}{'len':<6}{'N':>5}{'loc':>5}{'evct':>5}"
              + "".join(f"{'r@'+str(k):>8}" for k in RECALL_KS)
              + f"{'medRank':>9}{'nCand':>7}")
    print(header)
    print("-" * len(header))

    def _fmt_group(label_task, label_len, grp):
        N = len(grp)
        located = [r for r in grp if r["needle_located"]]
        nloc = len(located)
        evicted = [r for r in located if r["needle_evicted"]]
        nevct = len(evicted)
        # recall over located samples (blank recall -> treat as not-in-topK = 0,
        # but blank only occurs for non-located or no-candidate, excluded here).
        line = f"{label_task:<6}{label_len:<6}{N:>5}{nloc:>5}{nevct:>5}"
        for k in RECALL_KS:
            vals = [r[f"recall@{k}"] for r in located if isinstance(r[f"recall@{k}"], int)]
            line += f"{(sum(vals)/len(vals)):>8.2f}" if vals else f"{'-':>8}"
        ranks = [r["needle_rank"] for r in located if isinstance(r["needle_rank"], int)]
        line += f"{median(ranks):>9.1f}" if ranks else f"{'-':>9}"
        ncands = [r["n_candidates"] for r in grp if isinstance(r["n_candidates"], int) and r["n_candidates"] > 0]
        line += f"{(sum(ncands)/len(ncands)):>7.1f}" if ncands else f"{'-':>7}"
        print(line)

    for (task, length), grp in sorted(groups.items()):
        _fmt_group(task, length, grp)

    # Per-length aggregate across tasks.
    print("-" * len(header))
    by_len = defaultdict(list)
    for r in rows:
        by_len[r["length"]].append(r)
    for length in sorted(by_len):
        _fmt_group("ALL", length, by_len[length])

    # Evicted% note.
    print("=" * 78)
    located_all = [r for r in rows if r["needle_located"]]
    if located_all:
        evct = sum(1 for r in located_all if r["needle_evicted"])
        print(f"[E2] needle_evicted: {evct}/{len(located_all)} located samples "
              f"({100.0*evct/len(located_all):.1f}%) — needle dropped from the "
              f"capped FIFO buffer (no re-ranking can recover these).")
    nloc = sum(1 for r in rows if not r["needle_located"])
    if nloc:
        print(f"[E2] needle NOT located (answer string not found in tokens): "
              f"{nloc}/{len(rows)} samples — excluded from recall/rank stats.")
    print("[E2] GO/NO-GO: recall@16 (evicted-in-buffer pool) high (>0.70) => "
          "selection fixable (widen K / re-rank). recall@16 ~ chance (K/N) => "
          "reader-attn salience is an information wall on long docs.")


if __name__ == "__main__":
    main()
