#!/usr/bin/env python
"""InfLLM baseline — LongBench (real long-document QA) eval driver.

Head-to-head training-free peer of ``scripts/eval_qcmem_longbench.py``: SAME
LongBench data + prompt templates + scoring (reused verbatim from
``scripts/eval_longbench_mem_space.py`` -> ``load_longbench_dataset`` /
``format_prompt`` / ``DATASET2MAXGEN`` / ``compute_f1_multi`` / ``run_scoring``),
SAME strided ``[shard_index::num_shards]`` sharding and SAME on-disk layout
(``{ds}_{shard}.jsonl`` with ``{index, pred, answers, dataset}`` lines) so the
existing ``run_scoring`` shard-merge + official qa_f1 scorer treats InfLLM and
QCMem identically. ONLY the model forward differs — InfLLM training-free memory
attention (``scripts/infllm_qwen3.py``) instead of the QCMem write/read resume.

The 6 QA subtasks reported in the MemoryLLM paper: narrativeqa, qasper, hotpotqa,
2wikimqa, musique, multifieldqa_en. Each sample's generation budget is that
dataset's ``DATASET2MAXGEN`` entry.

Example (full eval on node .73):
    python scripts/eval_infllm_longbench.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --tasks narrativeqa qasper hotpotqa 2wikimqa multifieldqa_en musique \
        --use_chat_template \
        --output_dir longbench_results/infllm_8b \
        --num_shards 8 --shard_index 0
    # Score only (merge all shards):
    python scripts/eval_infllm_longbench.py --score_only \
        --output_dir longbench_results/infllm_8b
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT,
           os.path.join(PROJECT_ROOT, "scripts"),
           os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# LongBench task framework (data loading + prompt templates + F1/EM scoring) —
# reused verbatim, unmodified. Importing it defines mem_space helper classes but
# does NOT patch any model (InfLLM never calls apply_mem_space_to_model).
import scripts.eval_longbench_mem_space as lb  # noqa: E402
import scripts.infllm_qwen3 as infllm  # noqa: E402


def _im_end_ids(tokenizer):
    """Chat end tokens so decode stops at <|im_end|> (Qwen3 chat)."""
    ids = []
    try:
        tid = tokenizer.convert_tokens_to_ids("<|im_end|>")
        if isinstance(tid, int) and tid >= 0:
            ids.append(tid)
    except Exception:
        pass
    return ids


def main():
    parser = argparse.ArgumentParser(description="InfLLM baseline — LongBench eval")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--tasks", type=str, nargs="+", default=None,
                        help="LongBench subtasks (default: the 6 QA tasks "
                             f"{lb.DEFAULT_DATASETS}).")
    parser.add_argument("--output_dir", type=str,
                        default="longbench_results/infllm")
    parser.add_argument("--hf_dataset", type=str, default="THUDM/LongBench",
                        help="HF dataset id, or a local dir of {ds}.jsonl. Default "
                             "falls back to PROJECT_ROOT/data/longbench_raw/data.")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples per task (-1 = all).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSONL + compute F1/EM.")
    # InfLLM memory-config overrides (defaults = infllm.DEFAULT_MEM_CONFIG)
    parser.add_argument("--n_local", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="InfLLM prefill chunk size (execution granularity).")
    args = parser.parse_args()

    datasets_list = args.tasks if args.tasks else lb.DEFAULT_DATASETS

    # --- score-only: reuse the LongBench shard-merge + F1/EM scorer verbatim ---
    if args.score_only:
        lb.run_scoring(args.output_dir, datasets_list)
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

    print(f"[InfLLM-LongBench] model_path={model_path}")
    print(f"[InfLLM-LongBench] tasks={datasets_list} max_samples={args.max_samples} "
          f"chat={args.use_chat_template} think={args.enable_thinking} "
          f"shard={args.shard_index}/{args.num_shards}")

    model, tokenizer, searcher, cfg = infllm.load_infllm_qwen3(
        model_path, device=str(device), dtype=dtype, mem_config=mem_override)
    print(f"[InfLLM-LongBench] mem_config={cfg}")

    L = int(model.config.num_hidden_layers)
    end_ids = _im_end_ids(tokenizer) if args.use_chat_template else []
    prefill_chunk = int(cfg["chunk_size"])

    # Load LongBench data (offline JSONL) via the shared loader. When --hf_dataset
    # names a local directory, use it as data_dir; else retain the HF fallback.
    local_data_dir = args.hf_dataset if os.path.isdir(args.hf_dataset) else None
    all_data = lb.load_longbench_dataset(
        args.hf_dataset, datasets_list, data_dir=local_data_dir)

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    # Per-shard file: run_scoring globs "{ds}_*.jsonl" and dedups by "index".
    shard_tag = f"shard{args.shard_index}of{args.num_shards}" if sharded else "0"

    task_tag = datasets_list[0] if len(datasets_list) == 1 else "multi"
    with open(output_path / f"eval_config_{task_tag}_{shard_tag}.json", "w") as f:
        cfg_out = dict(vars(args))
        cfg_out.update({"resolved_model_path": model_path, "num_layers": L,
                        "baseline": "infllm", "infllm_mem_config": cfg})
        json.dump(cfg_out, f, indent=2)

    for ds_name in datasets_list:
        samples = all_data.get(ds_name, [])
        if not samples:
            print(f"[InfLLM-LongBench] Skipping {ds_name} (no data)")
            continue
        if args.max_samples > 0:
            samples = samples[:args.max_samples]

        # Strided shard (matches the babilong/ruler/QCMem [shard_index::num_shards]
        # convention; global index recorded so run_scoring dedups correctly).
        sample_indices = list(range(len(samples)))[args.shard_index::args.num_shards]
        if sharded:
            print(f"[InfLLM-LongBench] {ds_name} shard {args.shard_index}/"
                  f"{args.num_shards}: {len(sample_indices)} of {len(samples)} samples")

        max_gen = lb.DATASET2MAXGEN.get(ds_name, 64)
        outfile = output_path / f"{ds_name}_{shard_tag}.jsonl"
        results_buffer = []
        oom_count = 0
        t0 = time.time()

        for pos, idx in enumerate(tqdm(sample_indices, desc=f"{ds_name}", leave=True)):
            sample = samples[idx]
            prompt = lb.format_prompt(sample, ds_name, tokenizer,
                                      use_chat_template=args.use_chat_template,
                                      enable_thinking=args.enable_thinking)
            ids = tokenizer.encode(prompt, add_special_tokens=True,
                                   return_tensors="pt")
            if isinstance(ids, list):
                ids = torch.tensor([ids], dtype=torch.long)
            input_ids = ids.to(device)
            n_tokens = int(input_ids.shape[1])

            try:
                pred = infllm.infllm_generate(
                    searcher, input_ids, max_new_tokens=max_gen,
                    chunk_size=prefill_chunk, extra_end_token_ids=end_ids)
            except RuntimeError as e:
                if "out of memory" not in str(e).lower():
                    raise
                pred = "[OOM]"
                oom_count += 1
                print(f"[OOM] idx={idx} task={ds_name} n_tok={n_tokens}: {e}",
                      flush=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            results_buffer.append({
                "index": idx,
                "pred": pred,
                "answers": sample["answers"],
                "dataset": ds_name,
                "n_tokens": n_tokens,
            })

            if (pos + 1) % 10 == 0 or pos == len(sample_indices) - 1:
                with open(outfile, "w") as f:
                    for r in results_buffer:
                        f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (pos + 1) % 20 == 0:
                cur_f1s = [lb.compute_f1_multi(r["pred"], r["answers"])
                           for r in results_buffer]
                avg_f1 = sum(cur_f1s) / len(cur_f1s) * 100
                speed = (pos + 1) / (time.time() - t0)
                print(f"  [{ds_name}] {pos+1}/{len(sample_indices)} | "
                      f"{speed:.2f} samples/s | running F1={avg_f1:.1f}% | "
                      f"last_pred='{pred[:60]}'")

        with open(outfile, "w") as f:
            for r in results_buffer:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        f1s = [lb.compute_f1_multi(r["pred"], r["answers"]) for r in results_buffer]
        avg_f1 = (sum(f1s) / len(f1s) * 100) if f1s else 0.0
        elapsed = time.time() - t0
        metrics = {
            "dataset": ds_name, "shard_index": int(args.shard_index),
            "num_shards": int(args.num_shards), "num_samples": len(results_buffer),
            "f1": avg_f1, "elapsed_seconds": elapsed, "output_file": str(outfile),
            "oom_count": oom_count,
            "empty_prediction_count": sum(not r["pred"].strip()
                                          for r in results_buffer),
            "baseline": "infllm", "infllm_mem_config": cfg,
            "chat_template": bool(args.use_chat_template),
        }
        with open(output_path / f"{ds_name}_{shard_tag}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[InfLLM-LongBench] {ds_name}: F1={avg_f1:.2f}% "
              f"({len(results_buffer)} samples, {elapsed:.1f}s) -> {outfile}")

    print(f"\n[InfLLM-LongBench] Shard {args.shard_index}/{args.num_shards} complete!")
    # Single-shard: auto-score (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[InfLLM-LongBench] Running scoring (single-shard mode)...")
        lb.run_scoring(args.output_dir, datasets_list)


if __name__ == "__main__":
    main()
