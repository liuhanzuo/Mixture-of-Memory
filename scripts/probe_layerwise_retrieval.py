#!/usr/bin/env python
"""QCMem Insight 1 probe — "the bottom layers already suffice to find the
relevant chunk".

MECHANISM CLAIM UNDER TEST
--------------------------
QCMem's ``reader_attn`` selector (``scripts/eval_qcmem_babilong.py``:
``_reader_attn_scores``) ranks context chunks by the cosine similarity between
the *mean-pooled depth-j hidden* of the query chunk and each context chunk. The
whole point of QCMem is that ``h_j`` is a SHALLOW representation (j ≪ L): if a
shallow ``h_j`` already lets us retrieve the chunk that holds the answer, then we
never need to pay for the deep layers just to locate information.

This script measures that directly: for a sweep of depths ``j``, encode every
512-token chunk *chunk-local* (exactly like ``QCMemModel.write_chunk``: RoPE
positions ``0:T`` + a chunk-local causal mask, no BOS prepended), mean-pool over
tokens, cosine-rank the context chunks against the query chunk, take top-k, and
score against the ORACLE needle-chunk set (``run_babilong_mem_space``
``_locate_needle_chunks``). The read-out we care about:

  * ``recall@k``  = mean over samples of |retrieved ∩ gold| / |gold|
                    (did the shallow representation surface the answer chunk(s)?)
  * ``hit@k``     = mean over samples of 1[retrieved ∩ gold ≠ ∅]
                    (was AT LEAST ONE gold chunk in the top-k?)

KEY LOOK: if recall@k saturates / peaks at shallow j (e.g. j=4,6,8) and does not
improve (or degrades) at deep j, that supports Insight 1. We report the full
sweep honestly — including the case where deep layers win.

Efficiency: one chunk-local forward per chunk runs ``layers[0:max_j]`` once and
hooks out the hidden at every requested depth (no per-j re-forward). Probing
default is ``--limit 40`` samples per (task, length); this is a mechanism probe,
not a benchmark, so n=100 is unnecessary.

Faithfulness note: the encoder here reuses ``QCMemModel``'s own helpers
(``_as_ids`` / ``_make_mask_and_rope`` / the raw layer call in ``_run_layers``)
so the depth-j hidden is byte-for-byte what ``write_chunk`` would cache; the
similarity is the same mean-pool cosine as the deployed ``_reader_attn_scores``.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Offline HF caches (babilong Arrow cache + local model). Set before importing
# datasets/transformers so the offline resolvers see them. Respect any values
# the caller already exported.
os.environ.setdefault("HF_HOME", os.path.join(PROJECT_ROOT, ".hf_cache"))
os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(PROJECT_ROOT, ".hf_cache", "datasets"))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
from tqdm.auto import tqdm  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from babilong.prompts import (  # noqa: E402
    DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input,
)

from src.memory.qcmem import QCMemModel  # noqa: E402

# Bind the canonical scripts/run_babilong_mem_space.py by explicit path (a stale
# root-level duplicate could otherwise shadow it) for the oracle needle locator
# and the offline dataset loader.
_harness_path = os.path.join(PROJECT_ROOT, "scripts", "run_babilong_mem_space.py")
_spec = _ilu.spec_from_file_location("_qcmem_harness_probe", _harness_path)
harness = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(harness)


# --------------------------------------------------------------------------- #
# multi-depth chunk-local encoder (one forward, hidden hooked at each depth)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def encode_chunk_multidepth(qc: QCMemModel, token_ids, depths):
    """Encode ONE chunk to every depth in ``depths`` in a single forward.

    Mirrors ``QCMemModel.write_chunk`` exactly (embed + chunk-local causal mask +
    RoPE positions 0:T), but instead of stopping at ``resume_j`` it walks
    ``layers[0:max(depths)]`` and records the running hidden after each requested
    layer. ``depth j`` == the hidden AFTER running ``j`` layers == what
    ``write_chunk(resume_j=j)`` returns.

    Returns ``dict[int, Tensor[d]]``: mean-pooled (over the token axis), fp32,
    L2-normalisation is left to the caller. Mean-pool matches the deployed
    ``_reader_attn_scores`` (collapses variable chunk lengths to a single vector).
    """
    ids = qc._as_ids(token_ids)
    T = ids.shape[1]
    inputs_embeds = qc.embed_tokens(ids)
    positions = torch.arange(T, device=qc.device).unsqueeze(0)
    causal_mask, position_embeddings = qc._make_mask_and_rope(inputs_embeds, positions)

    want = set(int(j) for j in depths)
    max_j = max(want)
    hidden = inputs_embeds
    out = {}
    for li in range(max_j):
        hidden = qc.layers[li](
            hidden,
            attention_mask=causal_mask,
            position_ids=positions,
            position_embeddings=position_embeddings,
            use_cache=False,
        )
        depth = li + 1
        if depth in want:
            out[depth] = hidden.float().mean(dim=1).squeeze(0)  # [d]
    return out


def _cosine_topk(query_vec, chunk_vecs, k):
    """Rank ``chunk_vecs`` (list[Tensor[d]]) by cosine to ``query_vec`` (Tensor[d]);
    return the sorted list of the top-k indices (highest first)."""
    q = query_vec / (query_vec.norm() + 1e-8)
    scores = []
    for c in chunk_vecs:
        cc = c / (c.norm() + 1e-8)
        scores.append(float(torch.dot(q, cc).item()))
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    return order[: max(0, int(k))]


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description="QCMem Insight-1 layerwise retrieval probe")
    ap.add_argument("--model_path", type=str,
                    default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b")
    ap.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    ap.add_argument("--tasks", type=str, nargs="+", default=["qa1", "qa5"])
    ap.add_argument("--lengths", type=str, nargs="+", default=["8k", "16k"])
    ap.add_argument("--depths", type=int, nargs="+",
                    default=[1, 2, 4, 6, 8, 12, 16, 20, 24],
                    help="Depth-j values to probe (j = #bottom layers run).")
    ap.add_argument("--topks", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--limit", type=int, default=40,
                    help="Samples per (task, length). Probe, not a benchmark.")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--dtype", type=str, default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--attn_impl", type=str, default="sdpa")
    ap.add_argument("--out_json", type=str, default="")
    args = ap.parse_args()

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[probe1] model_path={args.model_path}")
    print(f"[probe1] depths={args.depths} topks={args.topks} chunk_size={args.chunk_size} "
          f"limit={args.limit} dtype={dtype}")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device).eval()

    L = int(model.config.num_hidden_layers)
    depths = sorted(j for j in args.depths if 1 <= j <= L)
    if not depths:
        raise SystemExit(f"no valid depths in [1,{L}] from {args.depths}")
    # resume_j only needs to be large enough for the internal validation; we run
    # the layer loop directly up to max(depths). Build once with resume_j=max.
    qc = QCMemModel(model, resume_j=max(depths))
    print(f"[probe1] model layers L={L}; probing depths {depths}")

    # results[(task,length)][j] -> dict with running sums for each topk.
    results = {}
    prompt_cache = {}

    for task in args.tasks:
        if task not in DEFAULT_PROMPTS:
            print(f"[probe1][WARN] task {task} not in DEFAULT_PROMPTS; skip")
            continue
        cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"],
            "examples":    DEFAULT_PROMPTS[task]["examples"],
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"],
            "template":    DEFAULT_TEMPLATE,
        }
        prompt_cache[task] = cfg

        for length in args.lengths:
            try:
                data = harness.load_babilong_dataset(args.dataset_name, length)
                task_data = data[task]
            except Exception as e:  # noqa: BLE001
                print(f"[probe1][ERR] load {args.dataset_name}/{length}/{task}: {e}")
                continue

            n = len(task_data)
            if args.limit > 0:
                n = min(n, args.limit)

            # per-depth accumulators
            acc = {j: {k: {"recall": 0.0, "hit": 0.0} for k in args.topks}
                   for j in depths}
            n_valid = 0     # samples with a locatable needle inside the context
            n_seen = 0
            gold_sizes = []
            nctx_list = []

            for idx in tqdm(range(n), desc=f"{task}/{length}", leave=False):
                sample = task_data[idx]
                target = sample["target"]
                input_text = get_formatted_input(
                    sample["input"], sample["question"],
                    cfg["examples"], cfg["instruction"], cfg["post_prompt"],
                    template=cfg["template"],
                )
                ids = tokenizer.encode(input_text, add_special_tokens=True,
                                       return_tensors="pt").to(device)
                n_seen += 1

                # oracle needle chunks (doc-absolute) — same locator as the eval.
                needle_set = harness._locate_needle_chunks(
                    ids, target, tokenizer, args.chunk_size)

                tokens = ids[0]
                chunks = list(tokens.split(args.chunk_size))
                if len(chunks) < 2:
                    continue
                context_chunks = chunks[:-1]
                query_chunk = chunks[-1]
                n_ctx = len(context_chunks)

                # gold context chunk indices (0..n_ctx-1); query is index n_ctx.
                gold = set()
                if needle_set:
                    gold = {c for c in needle_set if 0 <= c < n_ctx}
                if not gold:
                    # cannot verify retrieval without a ground-truth chunk -> skip
                    continue
                n_valid += 1
                gold_sizes.append(len(gold))
                nctx_list.append(n_ctx)

                # encode query + every context chunk to all depths (one forward each)
                q_vecs = encode_chunk_multidepth(qc, query_chunk, depths)
                ctx_vecs = {j: [] for j in depths}
                for c in context_chunks:
                    ev = encode_chunk_multidepth(qc, c, depths)
                    for j in depths:
                        ctx_vecs[j].append(ev[j])

                for j in depths:
                    for k in args.topks:
                        topk_idx = set(_cosine_topk(q_vecs[j], ctx_vecs[j], k))
                        inter = topk_idx & gold
                        acc[j][k]["recall"] += len(inter) / len(gold)
                        acc[j][k]["hit"] += 1.0 if inter else 0.0

            cell = {
                "n_seen": n_seen, "n_valid": n_valid,
                "mean_gold_chunks": (sum(gold_sizes) / len(gold_sizes)) if gold_sizes else 0.0,
                "mean_n_ctx": (sum(nctx_list) / len(nctx_list)) if nctx_list else 0.0,
                "by_depth": {},
            }
            for j in depths:
                cell["by_depth"][j] = {}
                for k in args.topks:
                    denom = max(1, n_valid)
                    cell["by_depth"][j][k] = {
                        "recall": acc[j][k]["recall"] / denom,
                        "hit":    acc[j][k]["hit"] / denom,
                    }
            results[f"{task}/{length}"] = cell
            _print_cell(task, length, depths, args.topks, cell)

    _print_summary(results, depths, args.topks)

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump({"config": vars(args), "L": L, "depths": depths,
                       "results": results}, f, indent=2)
        print(f"[probe1] wrote {args.out_json}")


def _print_cell(task, length, depths, topks, cell):
    print("\n" + "=" * 78)
    print(f"[probe1] {task} / {length}   "
          f"n_valid={cell['n_valid']}/{cell['n_seen']}  "
          f"mean_gold_chunks={cell['mean_gold_chunks']:.2f}  "
          f"mean_n_ctx={cell['mean_n_ctx']:.1f}")
    header = "  j  |" + "".join(f"  recall@{k}  hit@{k} |" for k in topks)
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    for j in depths:
        row = f" {j:>2}  |"
        for k in topks:
            d = cell["by_depth"][j][k]
            row += f"   {d['recall']:.3f}   {d['hit']:.3f} |"
        print(row)
    print("-" * len(header))


def _print_summary(results, depths, topks):
    print("\n" + "#" * 78)
    print("# SUMMARY — recall@k by depth j (averaged over all task/length cells)")
    print("#" * 78)
    if not results:
        print("(no results)")
        return
    header = "  j  |" + "".join(f"  recall@{k}  hit@{k} |" for k in topks)
    print(header)
    for j in depths:
        row = f" {j:>2}  |"
        for k in topks:
            rs = [c["by_depth"][j][k]["recall"] for c in results.values()]
            hs = [c["by_depth"][j][k]["hit"] for c in results.values()]
            row += f"   {sum(rs)/len(rs):.3f}   {sum(hs)/len(hs):.3f} |"
        print(row)


if __name__ == "__main__":
    main()
