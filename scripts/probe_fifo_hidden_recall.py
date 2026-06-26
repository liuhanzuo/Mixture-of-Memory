"""Logit-lens probe: does the FIFO-buffered needle hidden ENCODE the answer?

DIAGNOSTIC-ONLY. This script does NOT change any train/eval numeric path. It
reuses ``run_babilong_mem_space``'s model loader + bsz=1 streaming ingestion
verbatim and only READS the per-layer FIFO buffer (``MemorySpaceLayer._fifo_buf``)
that the forward pass already maintains. It sets NO probe flags and writes
nothing back into the model — pure read-out + a cheap logit-lens.

------------------------------------------------------------------------------
Scientific question
------------------------------------------------------------------------------
The FIFO buffer stores, for every chunk, the layer-INPUT hidden state that the
wrapped decoder later attends as a read-only prefix (layer.py:1505 /:1554,
``h_stored = hidden_states.detach()``). The oracle keep-set experiment showed
that even when the model is HANDED the correct needle chunk it still can't read
the answer out. Two hypotheses:

  (A) "didn't store it"  — the buffered hidden of the needle chunk does NOT
      encode the answer (the bf16 detached hidden lost the fact).
  (B) "stored but unreadable" — the buffered hidden DOES encode the answer
      (high rank under a logit-lens), so the failure is in the READ mechanism
      (RoPE position / attention dilution), not the stored representation.

A logit-lens (zero training, cheapest possible test) discriminates A vs B:
project a buffered hidden through the model's OWN final RMSNorm + lm_head ->
vocab logits, and check the rank/probability of the gold-answer token. If the
needle-chunk hidden ranks the answer token high (and far above a random
non-needle chunk + the question chunk), information IS stored -> hypothesis B.
If not -> hypothesis A.

------------------------------------------------------------------------------
Key implementation choices (flagged per the task brief)
------------------------------------------------------------------------------
* WHICH hidden we take:  ``layer._fifo_buf[-1]`` immediately after the forward
  of a given chunk. The FIFO write appends ``hidden_states.detach()`` at the END
  of ``_forward_fifo`` and only evicts index 0, so right after ``model(chunk_c)``
  every FIFO layer's ``_fifo_buf[-1]`` is EXACTLY that chunk's stored hidden
  (the thing the model actually uses as memory). We snapshot it in the streaming
  loop, BEFORE later chunks could evict it — so eviction (buffer cap = 25) never
  loses the needle hidden we measured. No hooks needed.

  NOTE on layer semantics: the buffer stores the layer-INPUT residual stream, so
  probing FIFO layer L reads the residual stream ENTERING decoder layer L (==
  the output of layer L-1). This is exactly the classic logit-lens object (an
  intermediate residual stream pushed through the final norm + unembed). Layer
  31's stored input == output of layer 30 (penultimate residual), the closest
  to the true final hidden; we deliberately probe a ladder of depths
  (default 8/16/24/31) because the lens sharpens with depth.

* WHICH lm_head / norm:  the model's OWN, untouched by the mem_space patch —
  ``model.model.norm`` (final LlamaRMSNorm) then ``model.lm_head``. This is the
  real unembedding, so a high answer-token rank means the residual is genuinely
  "pointing at" the answer in output space.

* WHICH token position:  the answer string can appear ANYWHERE inside the needle
  chunk, and a logit-lens at position t scores the token predicted AFTER t. So
  we scan ALL T positions of the chunk and report the BEST (min-rank / max-prob)
  position for the answer's first token, plus the mean over positions. The best
  position is "the residual right before the answer predicts the answer" — the
  cleanest 'is the fact recoverable from this stored block' signal.

* WHICH answer token:  the answer's FIRST sub-token, taken from BOTH ``encode(target)``
  and ``encode(' '+target)`` (BPE space-prefix merges differ in/out of context);
  we keep the union of candidate first-token ids and take the best rank over them.

------------------------------------------------------------------------------
Output
------------------------------------------------------------------------------
report/figs/fifo_hidden_recall_<task>_<length>.json  — per-layer aggregated
    mean best-rank / best-prob / mean-rank / top1-hit-rate for needle vs random
    non-needle vs question(last) chunk, plus per-sample raw rows.
report/figs/fifo_hidden_recall_<task>_<length>.txt   — human-readable summary.
(stdout mirrors the txt summary + the A-vs-B verdict heuristic.)

Usage:
    .venv/bin/python scripts/probe_fifo_hidden_recall.py \
        --model_path models/Meta-Llama-3-8B \
        --checkpoint outputs/mem_space_fifo_b25_chunk512_noleak/full_model.pt \
        --adapter_config outputs/mem_space_fifo_b25_chunk512_noleak/adapter_config.json \
        --task qa1 --length 8k --limit 30 --chunk_size 512 \
        --probe_layers 8 16 24 31
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

import datasets  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402
from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS,
    DEFAULT_TEMPLATE,
    get_formatted_input,
)

from scripts.run_babilong_mem_space import (  # noqa: E402
    build_mem_space_config,
    load_mem_space_model,
    _reset_banks,
    _reset_l2,
    _locate_needle_chunks,
)


# --------------------------------------------------------------------------- #
# FIFO buffer access helpers
# --------------------------------------------------------------------------- #


def _fifo_layers_by_idx(model):
    """Return {layer_idx: MemorySpaceLayer} for every FIFO-wrapped layer.

    ``_layer_idx`` is assigned in decoder order (0..L-1), so the keys are the
    real decoder layer indices.
    """
    root = getattr(model, "module", model)
    out = {}
    for w in getattr(root, "_mem_space_layers", []) or []:
        li = getattr(w, "_layer_idx", None)
        if li is not None:
            out[int(li)] = w
    return out


def _clear_fifo_buffers(model) -> None:
    """Per-sample reset of the FIFO buffer state on every layer.

    ``_reset_banks`` deliberately does NOT touch ``_fifo_buf`` (the slot reset
    leaves the FIFO buffer alone), so we clear it here so each document starts
    with an empty rolling buffer — matching the per-sample contract of the real
    eval ingestion path. Pure state reset, no numeric-path change.
    """
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []) or []:
        w._fifo_buf = []
        w._fifo_buf_abs_idx = []
        w._fifo_write_seq = 0


# --------------------------------------------------------------------------- #
# Logit-lens core
# --------------------------------------------------------------------------- #


@torch.no_grad()
def _logit_lens_metrics(model, hidden, ans_token_ids, device, dtype):
    """Push ``hidden`` [1, T, d] through final norm + lm_head and measure how
    well the answer token is predicted at the BEST position over the chunk.

    Returns a dict:
      best_rank   : min over positions of the answer-token rank (0 = top-1).
      best_prob   : max over positions of the answer-token softmax probability.
      mean_rank   : mean over positions of the answer-token rank.
      top1_hit    : 1 if ANY position's argmax is an answer-candidate token.
      best_pos    : the position achieving best_rank (for debugging).
    ``ans_token_ids`` is a list of candidate first-token ids (target vs ' '+target).
    rank of the answer = #vocab logits strictly greater than the answer logit
    (so smaller is better; 0 means the answer is the top-1 prediction).
    """
    root = getattr(model, "module", model)
    base = getattr(root, "model", None)        # LlamaModel (holds .norm)
    norm = getattr(base, "norm", None)
    lm_head = getattr(root, "lm_head", None)
    if norm is None or lm_head is None:
        raise RuntimeError("could not resolve model.model.norm / model.lm_head")

    h = hidden.to(device=device, dtype=dtype)
    with torch.amp.autocast(device_type="cuda", dtype=dtype):
        h_normed = norm(h)                     # [1, T, d]
        logits = lm_head(h_normed)             # [1, T, V]
    logits = logits[0].float()                 # [T, V]  (fp32 for stable ranks)
    T, V = logits.shape

    # For each candidate answer token id, rank over positions = how many vocab
    # entries beat the answer logit at that position.
    best_rank = None
    best_prob = 0.0
    best_pos = -1
    mean_rank_acc = None
    for tid in ans_token_ids:
        if tid < 0 or tid >= V:
            continue
        ans_logit = logits[:, tid]                              # [T]
        ranks = (logits > ans_logit.unsqueeze(1)).sum(dim=1)    # [T] (0 = top1)
        probs = torch.softmax(logits, dim=1)[:, tid]            # [T]
        cand_best_pos = int(ranks.argmin().item())
        cand_best_rank = int(ranks[cand_best_pos].item())
        cand_best_prob = float(probs.max().item())
        cand_mean_rank = float(ranks.float().mean().item())
        if best_rank is None or cand_best_rank < best_rank:
            best_rank = cand_best_rank
            best_pos = cand_best_pos
        best_prob = max(best_prob, cand_best_prob)
        mean_rank_acc = (
            cand_mean_rank if mean_rank_acc is None
            else min(mean_rank_acc, cand_mean_rank)
        )

    # top-1 hit: any position whose argmax is one of the candidate ids.
    argmax_ids = logits.argmax(dim=1)                           # [T]
    cand_set = set(int(t) for t in ans_token_ids)
    top1_hit = int(bool(any(int(a.item()) in cand_set for a in argmax_ids)))

    if best_rank is None:
        # No valid candidate token (shouldn't happen) — return sentinels.
        return {"best_rank": V, "best_prob": 0.0, "mean_rank": float(V),
                "top1_hit": 0, "best_pos": -1}
    return {
        "best_rank": best_rank,
        "best_prob": best_prob,
        "mean_rank": mean_rank_acc,
        "top1_hit": top1_hit,
        "best_pos": best_pos,
    }


def _answer_candidate_first_tokens(tokenizer, target: str):
    """First sub-token id(s) of the gold answer, both bare and space-prefixed."""
    tgt = (target or "").strip()
    ids = []
    for variant in (tgt, " " + tgt):
        enc = tokenizer.encode(variant, add_special_tokens=False)
        if enc:
            ids.append(int(enc[0]))
    # de-dup, preserve order
    seen, out = set(), []
    for i in ids:
        if i not in seen:
            seen.add(i)
            out.append(i)
    return out


# --------------------------------------------------------------------------- #
# Per-sample streaming + capture
# --------------------------------------------------------------------------- #


@torch.no_grad()
def probe_one_sample(model, input_ids, target, tokenizer, chunk_size,
                     probe_layers, device, dtype, rng):
    """Stream one sample, capture the buffered hidden of the needle / question /
    random chunk at each probe layer, and run the logit-lens on each.

    Returns (rows, info):
      rows : list of dicts {chunk_type, layer, best_rank, best_prob, mean_rank,
             top1_hit} — one per (chunk_type, probe layer).
      info : dict with n_chunks, needle_chunks, random_chunk, answer ids, etc.
             (or None if the needle could not be located -> caller skips sample).
    """
    needle_chunks = _locate_needle_chunks(input_ids, target, tokenizer, chunk_size)
    if not needle_chunks:
        return [], None

    tokens = input_ids[0]
    chunks = list(tokens.split(chunk_size))
    n_chunks = len(chunks)
    last_idx = n_chunks - 1

    # Pick ONE random non-needle, non-last chunk as a content control. Seeded
    # per-sample for reproducibility. If the doc is tiny (no eligible chunk),
    # the random control is skipped.
    eligible = [c for c in range(n_chunks)
                if c not in needle_chunks and c != last_idx]
    random_chunk = rng.choice(eligible) if eligible else None

    # Chunks whose buffered hidden we want to snapshot.
    capture = set(needle_chunks) | {last_idx}
    if random_chunk is not None:
        capture.add(random_chunk)

    ans_ids = _answer_candidate_first_tokens(tokenizer, target)
    layers = _fifo_layers_by_idx(model)
    probe_layers = [l for l in probe_layers if l in layers]

    _reset_banks(model)
    _reset_l2(model)
    _clear_fifo_buffers(model)

    # captured[chunk_type][layer] = metrics dict.  For "needle" (possibly
    # several chunks) we keep the BEST (min best_rank) across needle chunks.
    captured: dict = {"needle": {}, "question": {}, "random": {}}

    for c, chunk in enumerate(chunks):
        ct = chunk.unsqueeze(0).to(device)
        with torch.amp.autocast(device_type="cuda", dtype=dtype):
            _ = model(input_ids=ct, use_cache=False)
        if c not in capture:
            continue
        # Determine which bucket(s) this chunk falls into.
        buckets = []
        if c in needle_chunks:
            buckets.append("needle")
        if c == last_idx:
            buckets.append("question")
        if random_chunk is not None and c == random_chunk:
            buckets.append("random")
        for li in probe_layers:
            w = layers[li]
            if not w._fifo_buf:
                continue
            # ``_fifo_buf[-1]`` is THIS chunk's stored hidden (just appended).
            h = w._fifo_buf[-1]
            if h.shape[0] != 1:
                continue
            m = _logit_lens_metrics(model, h, ans_ids, device, dtype)
            for b in buckets:
                prev = captured[b].get(li)
                if prev is None or m["best_rank"] < prev["best_rank"]:
                    captured[b][li] = m

    rows = []
    for ct_name, per_layer in captured.items():
        for li, m in per_layer.items():
            rows.append({
                "chunk_type": ct_name,
                "layer": int(li),
                "best_rank": m["best_rank"],
                "best_prob": m["best_prob"],
                "mean_rank": m["mean_rank"],
                "top1_hit": m["top1_hit"],
            })
    info = {
        "n_chunks": n_chunks,
        "needle_chunks": sorted(int(c) for c in needle_chunks),
        "question_chunk": last_idx,
        "random_chunk": random_chunk,
        "answer_token_ids": ans_ids,
        "needle_is_question": bool(last_idx in needle_chunks),
    }
    return rows, info


# --------------------------------------------------------------------------- #
# Aggregation + reporting
# --------------------------------------------------------------------------- #


def _aggregate(all_rows, probe_layers):
    """Mean over samples of each metric, grouped by (chunk_type, layer)."""
    agg: dict = {}
    for ct in ("needle", "question", "random"):
        agg[ct] = {}
        for li in probe_layers:
            sel = [r for r in all_rows
                   if r["chunk_type"] == ct and r["layer"] == li]
            if not sel:
                continue
            agg[ct][li] = {
                "n": len(sel),
                "mean_best_rank": float(np.mean([r["best_rank"] for r in sel])),
                "median_best_rank": float(np.median([r["best_rank"] for r in sel])),
                "mean_best_prob": float(np.mean([r["best_prob"] for r in sel])),
                "mean_mean_rank": float(np.mean([r["mean_rank"] for r in sel])),
                "top1_hit_rate": float(np.mean([r["top1_hit"] for r in sel])),
            }
    return agg


def _format_summary(agg, probe_layers, task, length, n_samples):
    lines = []
    lines.append("=" * 76)
    lines.append(f"FIFO HIDDEN LOGIT-LENS RECALL  ({task} {length}, n={n_samples})")
    lines.append("=" * 76)
    lines.append("Per probe layer: mean BEST-position answer-token rank "
                 "(lower=better; 0=top-1),")
    lines.append("mean BEST-position prob, and top-1 hit-rate (any position "
                 "argmax == answer).")
    lines.append("")
    header = (f"{'layer':>5} | {'chunk':>9} | {'mean_best_rank':>15} | "
              f"{'median_best_rank':>16} | {'mean_best_prob':>14} | "
              f"{'top1_hit_rate':>13}")
    lines.append(header)
    lines.append("-" * len(header))
    for li in probe_layers:
        for ct in ("needle", "question", "random"):
            d = agg.get(ct, {}).get(li)
            if d is None:
                continue
            lines.append(
                f"{li:>5} | {ct:>9} | {d['mean_best_rank']:>15.1f} | "
                f"{d['median_best_rank']:>16.1f} | {d['mean_best_prob']:>14.4f} | "
                f"{d['top1_hit_rate']:>13.3f}"
            )
        lines.append("-" * len(header))

    # A-vs-B verdict heuristic: compare needle vs random at the DEEPEST probe
    # layer. If needle best-rank is dramatically lower (answer is recoverable
    # from the needle hidden but not a random chunk) -> stored (hypothesis B).
    deep = max(probe_layers)
    nd = agg.get("needle", {}).get(deep)
    rd = agg.get("random", {}).get(deep)
    lines.append("")
    if nd is not None and rd is not None:
        nr, rr = nd["mean_best_rank"], rd["mean_best_rank"]
        lines.append(f"VERDICT (deepest probe layer {deep}):")
        lines.append(f"  needle mean_best_rank = {nr:.1f}   "
                     f"random mean_best_rank = {rr:.1f}")
        if nr < 5 and nr < 0.2 * max(rr, 1.0):
            lines.append("  -> needle hidden ranks the answer NEAR TOP-1 and far "
                         "above a random chunk: INFORMATION IS STORED "
                         "(hypothesis B: stored-but-unreadable; the failure is "
                         "in the READ path).")
        elif nr < 0.5 * max(rr, 1.0):
            lines.append("  -> needle hidden ranks the answer notably better than "
                         "random: PARTIAL storage (answer is present but not "
                         "sharply; lean toward B with a weak representation).")
        else:
            lines.append("  -> needle hidden does NOT rank the answer above a "
                         "random chunk: INFORMATION NOT STORED (hypothesis A: "
                         "the bf16 detached hidden lost the fact).")
    else:
        lines.append("VERDICT: insufficient data (missing needle/random rows).")
    lines.append("=" * 76)
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(
        description="Logit-lens probe of FIFO-buffered needle hidden states "
                    "(diagnostic-only; reads MemorySpaceLayer._fifo_buf)."
    )
    ap.add_argument("--model_path", required=True,
                    help="Base Llama-3-8B model directory.")
    ap.add_argument("--checkpoint", required=True,
                    help="FIFO mem_space checkpoint (e.g. NOLEAK full_model.pt).")
    ap.add_argument("--adapter_config", required=True,
                    help="adapter_config.json describing the MemorySpaceConfig.")
    ap.add_argument("--dataset_name", default="RMT-team/babilong")
    ap.add_argument("--task", default="qa1",
                    help="BABILong task (qa1 = single supporting fact, cleanest).")
    ap.add_argument("--length", default="8k")
    ap.add_argument("--limit", type=int, default=30,
                    help="Number of samples to probe (default 30).")
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--probe_layers", type=int, nargs="+",
                    default=[8, 16, 24, 31],
                    help="Decoder layer indices to run the logit-lens on "
                         "(reads each layer's FIFO buffer = its input residual "
                         "stream). Default 8/16/24/31.")
    ap.add_argument("--seed", type=int, default=2026,
                    help="Seed for the random non-needle control chunk.")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--out_base", default=None,
                    help="Output path base (default "
                         "report/figs/fifo_hidden_recall_<task>_<length>).")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    rng = random.Random(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    with open(args.adapter_config) as f:
        adapter_cfg = json.load(f)
    mem_config = build_mem_space_config(adapter_cfg)
    mem_config.l3_recon_max_positions = args.chunk_size
    if not getattr(mem_config, "use_fifo_memory", False):
        print("[probe_fifo] WARNING: adapter_config has use_fifo_memory=False — "
              "this probe reads _fifo_buf and only makes sense for a FIFO ckpt.")
    print(f"[probe_fifo] use_fifo_memory={getattr(mem_config, 'use_fifo_memory', False)} "
          f"fifo_buffer_chunks={getattr(mem_config, 'fifo_buffer_chunks', None)} "
          f"chunk_size={args.chunk_size} probe_layers={args.probe_layers}")

    model = load_mem_space_model(
        model_path=args.model_path,
        checkpoint_path=args.checkpoint,
        mem_config=mem_config,
        device=device,
        dtype=dtype,
    )

    prompt_cfg = {
        "instruction": DEFAULT_PROMPTS[args.task]["instruction"],
        "examples": DEFAULT_PROMPTS[args.task]["examples"],
        "post_prompt": DEFAULT_PROMPTS[args.task]["post_prompt"],
        "template": DEFAULT_TEMPLATE,
    }
    data = datasets.load_dataset(args.dataset_name, args.length)
    task_data = data[args.task]
    n = len(task_data) if args.limit <= 0 else min(len(task_data), args.limit)
    print(f"[probe_fifo] probing {n} samples of {args.task}/{args.length}")

    all_rows = []
    sample_infos = []
    n_located = 0
    n_needle_is_question = 0
    for i in range(n):
        s = task_data[i]
        text = get_formatted_input(
            s["input"], s["question"], prompt_cfg["examples"],
            prompt_cfg["instruction"], prompt_cfg["post_prompt"],
            template=prompt_cfg["template"],
        )
        ids = tok.encode(text, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        ids = ids.to(device)
        rows, info = probe_one_sample(
            model, ids, s["target"], tok, args.chunk_size,
            list(args.probe_layers), device, dtype, rng,
        )
        if info is None:
            continue
        n_located += 1
        if info["needle_is_question"]:
            n_needle_is_question += 1
        for r in rows:
            r["sample_idx"] = i
        all_rows.extend(rows)
        sample_infos.append({"sample_idx": i, "target": s["target"], **info})
        if n_located % 5 == 0 or i == n - 1:
            print(f"  [{i+1}/{n}] located={n_located} tokens={ids.shape[1]} "
                  f"needle_chunks={info['needle_chunks']} "
                  f"n_chunks={info['n_chunks']}")

    if n_located == 0:
        print("[probe_fifo] ERROR: could not locate the needle in any sample. "
              "Aborting.")
        sys.exit(2)

    agg = _aggregate(all_rows, list(args.probe_layers))
    summary = _format_summary(agg, list(args.probe_layers), args.task,
                              args.length, n_located)
    print("\n" + summary)
    if n_needle_is_question:
        print(f"[probe_fifo] NOTE: {n_needle_is_question}/{n_located} samples had "
              f"the needle in the LAST (question) chunk — for those the 'needle' "
              f"and 'question' buckets overlap.")

    out_base = args.out_base or os.path.join(
        PROJECT_ROOT, "report", "figs",
        f"fifo_hidden_recall_{args.task}_{args.length}",
    )
    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    with open(out_base + ".json", "w") as f:
        json.dump({
            "task": args.task,
            "length": args.length,
            "n_samples_probed": n_located,
            "probe_layers": list(args.probe_layers),
            "chunk_size": args.chunk_size,
            "checkpoint": args.checkpoint,
            "aggregate": agg,
            "per_sample_rows": all_rows,
            "sample_infos": sample_infos,
        }, f, indent=2)
    with open(out_base + ".txt", "w") as f:
        f.write(summary + "\n")
    print(f"\n[probe_fifo] saved {out_base}.json / .txt")


if __name__ == "__main__":
    main()
