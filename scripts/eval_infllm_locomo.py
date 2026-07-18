#!/usr/bin/env python
"""InfLLM baseline — LoCoMo (long-conversation memory) eval driver.

Head-to-head training-free peer of ``scripts/eval_qcmem_locomo.py``: SAME LoCoMo
task frame + scoring (reused verbatim from ``scripts/eval_qcmem_locomo.py`` ->
``build_locomo_samples`` / ``render_locomo_history`` / ``score_sample`` /
``run_scoring`` / ``CATEGORY_NAMES`` — all self-contained, none depend on
``QCMemModel``), SAME stable sample order, SAME strided
``[shard_index::num_shards]`` sharding and SAME on-disk layout
(``preds{shard}.jsonl`` keyed by ``id``) so ``run_scoring`` (F1/EM/acc, per
category, plus the optional GPT-4o LLM judge) merges and grades InfLLM identically
to QCMem. ONLY the model forward differs — InfLLM training-free memory attention
(``scripts/infllm_qwen3.py``).

LoCoMo (ACL 2024, snap-research): 10 extended two-speaker conversations (up to
~19 dated sessions each) with ~1986 QA pairs across 5 reasoning categories
(multi-hop, single-hop, temporal, open-domain, adversarial). The whole dated
transcript + question is fed as one prompt; InfLLM's memory attention handles the
long context. Adversarial (category-5) QAs are scored as abstention-correct.

The GPT-4o judge is NOT run here (predictions are written for a separate judge
pass; ``--score_only --use_llm_judge`` grades them later, mirroring QCMem).

Example (full eval on node .73, n=1986):
    python scripts/eval_infllm_locomo.py \
        --model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen--Qwen3-8b \
        --data_path locomo/data/locomo10.json --use_chat_template \
        --output_dir locomo_results/infllm_8b \
        --num_shards 8 --shard_index 0
    # Score only (merge all shards, F1/EM/acc):
    python scripts/eval_infllm_locomo.py --score_only \
        --output_dir locomo_results/infllm_8b
    # Score only WITH GPT-4o judge (reads OPENAI_* from .env):
    python scripts/eval_infllm_locomo.py --score_only --use_llm_judge \
        --output_dir locomo_results/infllm_8b
"""
from __future__ import annotations

import argparse
import json
import os
import socket
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

from transformers import AutoTokenizer  # noqa: E402,F401  (kept for parity; model tokenizer comes from load_infllm_qwen3)

# LoCoMo task frame (data parse + prompt build + F1/EM/acc + LLM-judge scoring) —
# reused verbatim, unmodified. These symbols are self-contained (they do NOT touch
# QCMemModel). Importing the module pulls in the QCMem forward path at top level
# but nothing is instantiated on import.
import scripts.eval_qcmem_locomo as lc  # noqa: E402
import scripts.infllm_qwen3 as infllm  # noqa: E402

build_locomo_samples = lc.build_locomo_samples
score_sample = lc.score_sample
run_scoring = lc.run_scoring
CATEGORY_NAMES = lc.CATEGORY_NAMES


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
    parser = argparse.ArgumentParser(description="InfLLM baseline — LoCoMo eval")
    parser.add_argument("--model_path", type=str, default="")
    parser.add_argument("--data_path", type=str, default="locomo/data/locomo10.json",
                        help="Path to locomo10.json (1986 QA over 10 conversations). "
                             "Relative paths resolve against PROJECT_ROOT. Known "
                             "copy on diskB: data/dialogmem/locomo10.json.")
    parser.add_argument("--categories", type=str, default=None,
                        help="Comma-separated category numbers to keep (default all 5).")
    parser.add_argument("--output_dir", type=str, default="locomo_results/infllm")
    parser.add_argument("--max_samples", type=int, default=-1,
                        help="Max samples total (after category filter; -1 = all).")
    parser.add_argument("--max_new_tokens", type=int, default=48,
                        help="Greedy decode budget per answer (LoCoMo answers are "
                             "short phrases / dates / numbers).")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_index", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--use_chat_template", action="store_true", default=False)
    parser.add_argument("--enable_thinking", action="store_true", default=False)
    parser.add_argument("--score_only", action="store_true",
                        help="Only merge existing per-shard JSONL + recompute metrics.")
    # scoring passthrough (metrics + optional GPT-4o judge, run separately) --------
    parser.add_argument("--use_bertscore", action="store_true", default=False)
    parser.add_argument("--use_llm_judge", action="store_true", default=False,
                        help="Grade non-abstention preds CORRECT/WRONG with an LLM "
                             "judge (LoCoMo/mem0 protocol) at scoring time; reads "
                             "OPENAI_BASE_URL / OPENAI_API_KEY from .env.")
    parser.add_argument("--judge_model", type=str, default="gpt-4o")
    parser.add_argument("--judge_base_url", type=str, default=None)
    parser.add_argument("--judge_api_key", type=str, default=None)
    parser.add_argument("--judge_workers", type=int, default=8)
    # InfLLM memory-config overrides (defaults = infllm.DEFAULT_MEM_CONFIG)
    parser.add_argument("--n_local", type=int, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--block_size", type=int, default=None)
    parser.add_argument("--n_init", type=int, default=None)
    parser.add_argument("--chunk_size", type=int, default=None,
                        help="InfLLM prefill chunk size (execution granularity).")
    args = parser.parse_args()

    # --- score-only: merge shards + recompute metrics (+ optional judge), exit ---
    if args.score_only:
        run_scoring(args.output_dir, use_bertscore=args.use_bertscore,
                    use_llm_judge=args.use_llm_judge,
                    judge_model=args.judge_model,
                    judge_base_url=args.judge_base_url,
                    judge_api_key=args.judge_api_key,
                    judge_workers=args.judge_workers)
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
    data_path = args.data_path
    if not os.path.isabs(data_path):
        data_path = os.path.join(PROJECT_ROOT, data_path)

    categories = None
    if args.categories:
        categories = {int(c.strip()) for c in args.categories.split(",") if c.strip()}

    mem_override = {}
    for k in ("n_local", "topk", "block_size", "n_init", "chunk_size"):
        v = getattr(args, k)
        if v is not None:
            mem_override[k] = v

    print(f"[InfLLM-LoCoMo] model_path={model_path}")
    print(f"[InfLLM-LoCoMo] data_path={data_path}")
    print(f"[InfLLM-LoCoMo] categories={categories or 'all'} "
          f"max_samples={args.max_samples} chat={args.use_chat_template} "
          f"think={args.enable_thinking} shard={args.shard_index}/{args.num_shards}")

    model, tokenizer, searcher, cfg = infllm.load_infllm_qwen3(
        model_path, device=str(device), dtype=dtype, mem_config=mem_override)
    print(f"[InfLLM-LoCoMo] mem_config={cfg}")

    L = int(model.config.num_hidden_layers)
    end_ids = _im_end_ids(tokenizer) if args.use_chat_template else []
    prefill_chunk = int(cfg["chunk_size"])

    # --- load LoCoMo data + flatten to (conv x QA) samples in a stable order ---
    samples = build_locomo_samples(data_path)
    if categories is not None:
        samples = [s for s in samples if s["category"] in categories]
    print(f"[InfLLM-LoCoMo] total samples: {len(samples)}")
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # strided shard (matches the babilong/ruler/QCMem [shard_index::num_shards]
    # convention; global id recorded so run_scoring dedups correctly).
    shard = samples[args.shard_index::args.num_shards]
    print(f"[InfLLM-LoCoMo] shard {args.shard_index}/{args.num_shards}: "
          f"{len(shard)} samples")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sharded = args.num_shards > 1
    shard_tag = f"_shard{args.shard_index}of{args.num_shards}" if sharded else ""

    with open(outdir / f"eval_config{shard_tag}.json", "w") as f:
        cfg_out = dict(vars(args))
        cfg_out.update({"resolved_model_path": model_path,
                        "resolved_data_path": data_path, "num_layers": L,
                        "baseline": "infllm", "infllm_mem_config": cfg,
                        "runtime": {"node": socket.gethostname(),
                                    "cuda_visible_devices":
                                        os.environ.get("CUDA_VISIBLE_DEVICES")}})
        json.dump(cfg_out, f, indent=2)

    outfile = outdir / f"preds{shard_tag}.jsonl"
    results_buffer = []
    oom_count = 0
    t0 = time.time()

    for pos, sample in enumerate(tqdm(shard, desc="locomo", leave=True)):
        prompt = sample["prompt"]
        if args.use_chat_template:
            messages = [{"role": "user", "content": prompt}]
            try:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                    enable_thinking=args.enable_thinking)
            except TypeError:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True)
        ids = tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")
        if isinstance(ids, list):
            ids = torch.tensor([ids], dtype=torch.long)
        input_ids = ids.to(device)
        n_tokens = int(input_ids.shape[1])

        try:
            pred = infllm.infllm_generate(
                searcher, input_ids, max_new_tokens=args.max_new_tokens,
                chunk_size=prefill_chunk, extra_end_token_ids=end_ids)
        except RuntimeError as e:
            if "out of memory" not in str(e).lower():
                raise
            pred = "[OOM]"
            oom_count += 1
            print(f"[OOM] id={sample['id']} n_tok={n_tokens}: {e}", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        results_buffer.append({
            "id": sample["id"],
            "pred": pred,
            "answers": sample["answers"],
            "category": sample["category"],
            "is_abstention": sample["is_abstention"],
            "question": sample["question"],
            "n_tokens": n_tokens,
        })

        if (pos + 1) % 10 == 0 or pos == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in results_buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (pos + 1) % 20 == 0:
            cur = [score_sample(r) for r in results_buffer]
            avg_f1 = sum(c["f1"] for c in cur) / len(cur) * 100
            avg_acc = sum(c["acc"] for c in cur) / len(cur) * 100
            speed = (pos + 1) / (time.time() - t0)
            print(f"  [locomo] {pos+1}/{len(shard)} | {speed:.2f} samples/s | "
                  f"running F1={avg_f1:.1f}% acc={avg_acc:.1f}% | "
                  f"last_pred='{pred[:60]}'")

    with open(outfile, "w") as f:
        for r in results_buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"[InfLLM-LoCoMo] shard {args.shard_index}/{args.num_shards} done: "
          f"{len(results_buffer)} samples ({time.time()-t0:.1f}s, "
          f"oom={oom_count}) -> {outfile}")

    # Single-shard: auto-score (multi-shard: run --score_only after all finish).
    if args.num_shards == 1:
        print("\n[InfLLM-LoCoMo] Running scoring (single-shard mode)...")
        run_scoring(args.output_dir, use_bertscore=args.use_bertscore,
                    use_llm_judge=args.use_llm_judge,
                    judge_model=args.judge_model,
                    judge_base_url=args.judge_base_url,
                    judge_api_key=args.judge_api_key,
                    judge_workers=args.judge_workers)

    print("\n[InfLLM-LoCoMo] Evaluation complete!")


if __name__ == "__main__":
    main()
