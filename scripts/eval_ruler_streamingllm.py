#!/usr/bin/env python
"""StreamingLLM (fixed KV-budget) baseline — RULER long-context eval driver.

A fixed-budget long-context baseline to put next to QCMem
(``scripts/eval_ruler_qcmem.py``) under an *identical* RULER口径: same sample
generation, same ``string_match_all`` recall scoring, same lengths — only the
model forward differs (StreamingLLM's sink+window KV budget instead of QCMem's
write/read chunk resume, or the base full-attention path).

WHY THE TRUNCATION APPROXIMATION (not a bit-faithful attention rewrite)
----------------------------------------------------------------------
StreamingLLM (Xiao et al. 2023) keeps only the KV of the first ``sink_size``
tokens (the "attention sink") plus the most-recent ``window_size`` tokens, and
DROPS everything in between; its rolling cache re-assigns RoPE positions so the
kept tokens occupy contiguous positions ``0 .. sink+window-1`` (no
extrapolation). transformers 5.5.4 no longer ships ``SinkCache`` (removed in the
5.x cache refactor — only ``SlidingWindowCache`` / ``StaticCache`` remain, and
neither preserves sink tokens), so rather than re-implement the attention /
rotary forward (fragile, and where a prior attempt stalled), we use the
functionally-equivalent **truncation approximation**:

    fed = concat( input_ids[:, :sink_size] , input_ids[:, -window_size:] )

and run the *unmodified* full-attention model on ``fed``. Because ``fed`` has
length ``sink_size + window_size`` (~6657) < the model window, the default
contiguous ``position_ids`` place the sink at ``0..sink-1`` and the recent
window at ``sink..sink+window-1`` — exactly StreamingLLM's position rolling, no
RoPE extrapolation. This reproduces StreamingLLM's two defining properties:

  * **constant peak memory / compute** regardless of nominal context length
    (the model only ever sees ``budget`` tokens), and
  * **the middle of the context is invisible** — a NIAH needle placed in the
    dropped middle is structurally unrecoverable.

This is NOT bit-faithful StreamingLLM (no token-by-token streaming re-rotation),
but it is functionally equivalent for a fixed-budget retrieval eval and, most
importantly, it does not touch attention. We label results
"StreamingLLM (truncation approx.)" throughout.

Budget is chosen comparable to QCMem's read length (~6657 tokens): the QCMem
read packs [sink; selected chunk h_j; query h_j]; here sink_size=4 +
window_size=6653 = 6657 keeps the same KV budget so the精度/显存 comparison is
apples-to-apples.

RULER framework (generation + scoring) is imported VERBATIM from
``scripts/eval_ruler_mem_space`` (``_build_sample`` / ``_make_vt_icl`` /
``_string_match_all_one`` / ``_LENGTH_TOKENS``), and the per-(task,length,i) RNG
+ shard filter are replicated bit-for-bit from ``eval_ruler_mem_space.main`` /
``eval_ruler_qcmem.main`` so the sample set matches the QCMem run exactly.

Usage (Qwen3-8B, niah_single, QCMem-comparable budget):
    python scripts/eval_ruler_streamingllm.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --sink_size 4 --window_size 6653 \
        --ruler_tasks niah_single --lengths 8k 16k 64k 128k \
        --limit 50 --output_name ruler_streamingllm_qwen \
        --results_folder ruler_results/streamingllm_qwen
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

# RULER task framework (generation + scoring) — reused verbatim, unmodified.
import scripts.eval_ruler_mem_space as ruler  # noqa: E402

# RULER task-name aliases -> canonical eval_ruler_mem_space task ids (mirrors
# scripts/eval_ruler_qcmem.py so the two drivers accept the same friendly names).
_TASK_ALIAS = {
    "niah_single": "niah_single_2",        # realistic PG19-prose haystack (default)
    "niah_single_noise": "niah_single_1",
    "niah_single_essay": "niah_single_2",
    "niah_multi": "niah_multikey_1",
    "niah_multikey": "niah_multikey_1",
    "vt": "variable_tracking",
}
_CANONICAL_TASKS = {
    "niah_single_1", "niah_single_2", "niah_multikey_1", "variable_tracking",
}


def _resolve_task(name: str) -> str:
    if name in _CANONICAL_TASKS:
        return name
    if name in _TASK_ALIAS:
        return _TASK_ALIAS[name]
    raise ValueError(
        f"unknown ruler task {name!r}; expected one of "
        f"{sorted(_CANONICAL_TASKS)} or aliases {sorted(_TASK_ALIAS)}"
    )


def _bare_question(prompt: str) -> str:
    """Trailing question line (RULER puts a \\n right before the question)."""
    return prompt[prompt.rfind("\n") + 1:].strip()


def _write_csv(df_rows, path: Path) -> None:
    """QUOTE_ALL CSV writer (matches the QCMem/babilong harness layout so the
    same scorers / inspection tooling apply). Columns: target,output,question,
    recall."""
    with open(path, "w", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["target", "output", "question", "recall"])
        for r in df_rows:
            w.writerow([r["target"], r["output"], r["question"], r["recall"]])


# --------------------------------------------------------------------------- #
# StreamingLLM forward: truncation approximation (sink + recent window).
# --------------------------------------------------------------------------- #
@torch.no_grad()
def streaming_generate(model, tokenizer, input_ids, sink_size, window_size,
                       max_new_tokens, device):
    """Keep the first ``sink_size`` tokens (attention sink) + the last
    ``window_size`` tokens (recent window), drop the middle, then greedily decode
    with the unmodified full-attention model. The kept sequence length is
    <= sink+window << model window, so default contiguous position_ids reproduce
    StreamingLLM's position rolling with no RoPE extrapolation.

    Returns (decoded_text, kept_len, orig_len).
    """
    ids = input_ids
    orig_len = int(ids.shape[1])
    budget = sink_size + window_size
    if orig_len > budget:
        sink = ids[:, :sink_size]
        recent = ids[:, -window_size:]
        ids = torch.cat([sink, recent], dim=1)
    kept_len = int(ids.shape[1])
    out = model.generate(
        ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        pad_token_id=tokenizer.pad_token_id,
    )
    gen = out[0, ids.shape[1]:]
    text = tokenizer.decode(gen, skip_special_tokens=True).strip()
    return text, kept_len, orig_len


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="StreamingLLM (fixed KV budget) — RULER (NIAH/VT) eval driver"
    )
    parser.add_argument("--model_path", type=str, required=True,
                        help="Plain backbone weights (Qwen3-8B / Llama-3-8B).")
    parser.add_argument("--sink_size", type=int, default=4,
                        help="Attention-sink KV kept (StreamingLLM default 4).")
    parser.add_argument("--window_size", type=int, default=6653,
                        help="Recent-window KV kept. sink+window is the fixed KV "
                             "budget; default 4+6653=6657 ~ QCMem read length.")
    parser.add_argument("--results_folder", type=str, default="./ruler_results")
    parser.add_argument("--output_name", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=48,
                        help="Greedy decode budget (RULER uses 48; VT is bumped "
                             "to >=60, matching eval_ruler_mem_space).")
    parser.add_argument("--limit", type=int, default=50,
                        help="Samples per (task,length) cell (RULER num_samples).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--attn_impl", type=str, default="sdpa")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base RNG seed (matches eval_ruler_mem_space / "
                             "eval_ruler_qcmem so the sample set is comparable).")
    parser.add_argument("--ruler_tasks", type=str, nargs="+",
                        default=["niah_single"],
                        help="RULER tasks / aliases: niah_single(_1/_2), "
                             "niah_multi(key_1), vt(variable_tracking).")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "64k", "128k"])
    args = parser.parse_args()

    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not (0 <= args.shard_index < args.num_shards):
        parser.error(f"--shard_index must be in [0, {args.num_shards})")

    tasks = [_resolve_task(t) for t in args.ruler_tasks]

    device = torch.device(args.device)
    cuda_mem = device.type == "cuda" and torch.cuda.is_available()
    dtype = {"bfloat16": torch.bfloat16,
             "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    budget = args.sink_size + args.window_size
    print(f"[StreamingLLM-RULER] model_path={args.model_path}")
    print(f"[StreamingLLM-RULER] TRUNCATION APPROX: sink_size={args.sink_size} "
          f"window_size={args.window_size} budget={budget} dtype={dtype} "
          f"attn_impl={args.attn_impl}")
    print(f"[StreamingLLM-RULER] tasks={tasks} lengths={args.lengths} "
          f"limit={args.limit}")

    # local_files_only=True: offline nodes otherwise treat a local dir path as an
    # HF repo_id and error ("Repo id must be in the form ...").
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        trust_remote_code=True,
        local_files_only=True,
    ).to(device).eval()

    outdir = Path(args.results_folder) / args.output_name
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    summary: dict = {}
    for task in tqdm(tasks, desc="tasks"):
        summary[task] = {}
        for length in tqdm(args.lengths, desc="lengths", leave=False):
            if length not in ruler._LENGTH_TOKENS:
                print(f"[WARN] unknown length {length}, skipping")
                continue
            target_tokens = ruler._LENGTH_TOKENS[length]
            # Deterministic per-(task,length) RNG so shards share the sample set
            # (identical construction to eval_ruler_mem_space / eval_ruler_qcmem).
            base_seed = args.seed + (hash((task, length)) % 100000)

            vt_icl = None
            if task == "variable_tracking":
                vt_icl = ruler._make_vt_icl(random.Random(base_seed + 777), 4)

            sample_indices = set(
                list(range(args.limit))[args.shard_index::args.num_shards]
            )
            if sharded:
                print(f"[StreamingLLM-RULER] {task}/{length} shard "
                      f"{args.shard_index}/{args.num_shards}: "
                      f"{len(sample_indices)} of {args.limit} samples")

            rows = []
            recall_sum = 0.0
            total = 0
            n_tok_seen = 0
            kept_seen = 0
            mnt = args.max_new_tokens if task != "variable_tracking" \
                else max(args.max_new_tokens, 60)

            if cuda_mem:
                torch.cuda.reset_peak_memory_stats(device)

            for i in tqdm(range(args.limit), desc=f"{task}/{length}", leave=False):
                # Build EVERY sample (fixed per-i seed) so shard sample sets align,
                # then process only this shard's indices.
                rng = random.Random(base_seed * 1000 + i)
                prompt, answers, gold_needle = ruler._build_sample(
                    task, target_tokens, tokenizer, rng, vt_icl)
                if i not in sample_indices:
                    continue

                ids = tokenizer.encode(prompt, add_special_tokens=True,
                                       return_tensors="pt")
                if isinstance(ids, list):
                    ids = torch.tensor([ids], dtype=torch.long)
                input_ids = ids.to(device)
                n_tok_seen = int(input_ids.shape[1])

                bare_q = _bare_question(prompt)

                try:
                    output, kept_len, _orig = streaming_generate(
                        model=model, tokenizer=tokenizer, input_ids=input_ids,
                        sink_size=args.sink_size, window_size=args.window_size,
                        max_new_tokens=mnt, device=device,
                    )
                    kept_seen = kept_len
                except RuntimeError as e:
                    if "out of memory" not in str(e).lower():
                        raise
                    output = "[OOM]"
                    print(f"[OOM] i={i} task={task} length={length}: {e}",
                          flush=True)
                    if cuda_mem:
                        torch.cuda.empty_cache()

                rec = ruler._string_match_all_one(output, answers)
                recall_sum += rec
                total += 1
                rows.append({"target": " | ".join(answers), "output": output,
                             "question": bare_q, "recall": rec})
                if len(rows) % 10 == 0:
                    _write_csv(rows, outdir / f"{task}_{length}{shard_tag}.csv")

            peak_gb = (torch.cuda.max_memory_allocated(device) / (1024 ** 3)
                       if cuda_mem else 0.0)
            score = (recall_sum / total * 100.0) if total else 0.0
            summary[task][length] = {
                "score": round(score, 2), "n": total,
                "approx_tokens": n_tok_seen, "kept_tokens": kept_seen,
                "peak_mem_gb": round(peak_gb, 3),
            }
            outfile = outdir / f"{task}_{length}{shard_tag}.csv"
            _write_csv(rows, outfile)
            cfg_file = outdir / f"{task}_{length}{shard_tag}.json"
            json.dump(
                {
                    "task": task, "length": length,
                    "summary": summary[task][length],
                    "streamingllm": {
                        "method": "truncation_approx",
                        "sink_size": args.sink_size,
                        "window_size": args.window_size,
                        "budget": budget,
                    },
                    "model": {"model_path": args.model_path},
                },
                open(cfg_file, "w"), indent=2,
            )
            print(f"[StreamingLLM-RULER] {task}/{length}: recall={score:.2f} "
                  f"({total} samples, ~{n_tok_seen} tok -> kept {kept_seen}, "
                  f"peak {peak_gb:.2f} GB) -> {outfile}")

    print("\n[StreamingLLM-RULER] SUMMARY")
    for task in summary:
        row = "  ".join(
            f"{ln}={summary[task][ln]['score']:.1f}"
            f"(pk{summary[task][ln]['peak_mem_gb']:.1f}G)"
            for ln in summary[task])
        print(f"  {task:>18}: {row}")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[StreamingLLM-RULER] Evaluation complete!")


if __name__ == "__main__":
    main()
