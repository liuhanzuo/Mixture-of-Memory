#!/usr/bin/env python
"""InfLLM baseline — LongEval (LongChat lines-retrieval) eval driver.

Head-to-head training-free peer of ``scripts/eval_qcmem_longeval.py``: SAME
LongEval prompt synthesis + judging + length buckets (reused verbatim from
``scripts/eval_qcmem_longeval.py`` -> ``build_lines_prompt`` / ``extract_prediction``
/ ``_LENGTH_TOKENS`` / ``score_longeval``), SAME stable per-sample RNG
(``zlib.crc32``-derived, so every shard sees an identical sample set and InfLLM
sees byte-identical raw prompts to QCMem), SAME strided
``[shard_index::num_shards]`` sharding and SAME on-disk layout
(``longeval_{length}{shard}.json`` with a ``records`` list keyed by
``sample_index``) so ``score_longeval`` merges shards unchanged. ONLY the model
forward differs — InfLLM training-free memory attention
(``scripts/infllm_qwen3.py``).

Pure single-hop exact retrieval: a record of N lines

    line <random-label>: REGISTER_CONTENT is <6-digit number>

after which the model must return the REGISTER_CONTENT of one queried line. A
sample is correct iff ``extract_prediction(output) == expected_value``.

Example (full eval on node .73):
    python scripts/eval_infllm_longeval.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --lengths 8k 16k 32k 64k 128k --limit 50 --use_chat_template \
        --output_dir longeval_results/infllm_8b \
        --num_shards 8 --shard_index 0
    # Score only (merge all shards):
    python scripts/eval_infllm_longeval.py --score_only \
        --lengths 8k 16k 32k 64k 128k \
        --output_dir longeval_results/infllm_8b
"""
from __future__ import annotations

import argparse
import json
import os
import random
import socket
import sys
import time
import zlib
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LongEval prompt synthesis + judging + extended length buckets (64k/128k) +
# shard-merge scorer — reused verbatim, unmodified, from the QCMem LongEval driver
# (which itself re-exports build_lines_prompt / extract_prediction from
# eval_longeval_mem_space). Same symbols -> InfLLM and QCMem generate identical
# samples and are scored by the identical merge/score path.
import scripts.eval_qcmem_longeval as leq  # noqa: E402
import scripts.infllm_qwen3 as infllm  # noqa: E402

build_lines_prompt = leq.build_lines_prompt
extract_prediction = leq.extract_prediction
_LENGTH_TOKENS = leq._LENGTH_TOKENS
score_longeval = leq.score_longeval


def _im_end_ids(tokenizer):
    ids = []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="InfLLM baseline — LongEval eval")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--lengths", type=str, nargs="+",
                        default=["8k", "16k", "32k", "64k", "128k"],
                        help=f"Length buckets (subset of {sorted(_LENGTH_TOKENS)}).")
    parser.add_argument("--limit", type=int, default=50,
                        help="Samples per length bucket.")
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--seed", type=int, default=1234,
                        help="Base RNG seed (per-sample seed derived stably with "
                             "zlib.crc32 so shards align — identical to the QCMem "
                             "LongEval driver so both see the same sample set).")
    parser.add_argument("--output_dir", type=str, default="longeval_results/infllm")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSON + recompute acc.")
    # InfLLM memory-config overrides (defaults = infllm.DEFAULT_MEM_CONFIG)
    parser.add_argument("--n_local", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="InfLLM prefill chunk size (execution granularity).")
    args = parser.parse_args()

    outdir = Path(args.output_dir)

    # --- score-only: merge shards + recompute per-length accuracy, then exit ---
    if args.score_only:
        score_longeval(outdir, args.lengths)
        return

    if args.num_shards < 1 or not (0 <= args.shard_index < args.num_shards):
        parser.error("bad shard config")
    if not args.model_path:
        parser.error("--model_path is required unless --score_only")

    device = torch.device(args.device)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    model_path = args.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(PROJECT_ROOT, model_path)

    mem_override = {}
    for k in ("n_local", "topk", "block_size", "n_init", "chunk_size"):
        v = getattr(args, k)
        if v is not None:
            mem_override[k] = v

    print(f"[InfLLM-LongEval] model_path={model_path}")
    print(f"[InfLLM-LongEval] lengths={args.lengths} limit={args.limit} "
          f"chat={args.use_chat_template} think={args.enable_thinking} "
          f"shard={args.shard_index}/{args.num_shards}")

    model, tokenizer, searcher, cfg = infllm.load_infllm_qwen3(
        model_path, device=str(device), dtype=dtype, mem_config=mem_override)
    print(f"[InfLLM-LongEval] mem_config={cfg}")

    L = int(model.config.num_hidden_layers)
    end_ids = _im_end_ids(tokenizer) if args.use_chat_template else []
    prefill_chunk = int(cfg["chunk_size"])

    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg_out = dict(vars(args))
        cfg_out.update({"resolved_model_path": model_path, "num_layers": L,
                        "baseline": "infllm", "infllm_mem_config": cfg,
                        "runtime": {"node": socket.gethostname(),
                                    "cuda_visible_devices":
                                        os.environ.get("CUDA_VISIBLE_DEVICES")}})
        json.dump(cfg_out, f, indent=2)

    summary: dict = {}
    for length in args.lengths:
        if length not in _LENGTH_TOKENS:
            print(f"[WARN] unknown length {length}, skipping")
            continue
        target_tokens = _LENGTH_TOKENS[length]
        # STABLE per-length seed (NOT Python's process-randomized hash()) — matches
        # eval_qcmem_longeval so every shard/driver derives the SAME per-sample
        # seeds -> identical sample set.
        length_seed = args.seed + (zlib.crc32(length.encode()) % 100000)

        sample_indices = list(range(args.limit))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[InfLLM-LongEval] {length} shard {args.shard_index}/"
                  f"{args.num_shards}: {len(sample_indices)} of {args.limit} samples")

        records = []
        correct = 0
        total = 0
        n_tok_seen = 0
        t0 = time.time()
        for pos, i in enumerate(tqdm(sample_indices, desc=f"{length}", leave=False)):
            # Per-sample RNG (stable across processes) — build ONLY this shard's
            # samples. The token budget is measured on the RAW prompt inside
            # build_lines_prompt (before any chat template), identical to QCMem.
            rng = random.Random(length_seed * 1000 + i)
            prompt, expected, target_label, n_lines = build_lines_prompt(
                target_tokens, tokenizer, rng)

            if args.use_chat_template:
                messages = [{"role": "user", "content": prompt}]
                try:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True,
                        enable_thinking=args.enable_thinking)
                except TypeError:
                    prompt = tokenizer.apply_chat_template(
                        messages, tokenize=False, add_generation_prompt=True)

            ids = tokenizer.encode(prompt, add_special_tokens=True,
                                   return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            n_tok_seen = int(input_ids.shape[1])

            try:
                output = infllm.infllm_generate(
                    searcher, input_ids, max_new_tokens=args.max_new_tokens,
                    chunk_size=prefill_chunk, extra_end_token_ids=end_ids)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                output = "[OOM]"
                print(f"[OOM] i={i} length={length} n_tok={n_tok_seen}: {e}",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pred = extract_prediction(output)
            ok = (pred == expected)
            correct += int(ok)
            total += 1
            records.append({
                "sample_index": i,
                "label": target_label, "expected": expected,
                "output": output, "pred": pred, "correct": ok,
                "n_lines": n_lines, "n_tokens": n_tok_seen,
                "read_len": None,
            })

            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                acc_cur = correct / total if total else 0.0
                with open(outdir / f"longeval_{length}{shard_tag}.json", "w") as f:
                    json.dump({"length": length,
                               "summary": {"accuracy": round(acc_cur, 4),
                                           "correct": correct, "total": total},
                               "records": records}, f, indent=2)

        acc = correct / total if total else 0.0
        summary[length] = {
            "accuracy": round(acc, 4), "correct": correct, "total": total,
            "approx_tokens": n_tok_seen,
        }
        with open(outdir / f"longeval_{length}{shard_tag}.json", "w") as f:
            json.dump({"length": length, "summary": summary[length],
                       "records": records}, f, indent=2)
        print(f"[InfLLM-LongEval] {length}: acc={acc:.3f} ({correct}/{total}) "
              f"~{n_tok_seen} tok  ({time.time()-t0:.1f}s) "
              f"-> longeval_{length}{shard_tag}.json")

    print("\n[InfLLM-LongEval] SUMMARY (this shard)")
    for length, s in summary.items():
        print(f"  {length:>4}: acc={s['accuracy']:.3f}  "
              f"({s['correct']}/{s['total']})  ~{s['approx_tokens']} tok")
    with open(outdir / f"_summary{shard_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Single-shard: auto-merge (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[InfLLM-LongEval] Running merged scoring (single-shard mode)...")
        score_longeval(outdir, args.lengths)

    print("\n[InfLLM-LongEval] Evaluation complete!")


if __name__ == "__main__":
    main()
