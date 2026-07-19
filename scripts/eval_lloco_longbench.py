"""LLoCO baseline — LongBench eval driver (Paper A / CoMem head-to-head).

Reproduces LLoCO (Tan et al. 2024, arXiv:2404.07979) on LongBench QA tasks
using the officially released weights:
  - compressor : princeton-nlp/AutoCompressor-Llama-2-7b-6k
  - domain LoRA: xiuyul/Lloco-7b-{nqa,qasper,hqa}

Pipeline (per sample, combined preproc+inference — equivalent to the two-stage
preproc_embs.py + inference.py in the LLoCO repo, done inline so no .pth cache):
  1. Compress the document `context` into a soft prompt with the AutoCompressor
     (segment_lengths=1536, output_softprompt=True, no truncation — up to 122880
     context tokens, identical to LLoCO preproc_embs.preprocess_context_autocomp).
  2. Build the decoder query = LLoCO NATIVE per-task instruction prompt + question
     + "\nAnswer:" (the exact format the released LoRA was fine-tuned on — see
     external/lloco/finetune_scrolls.py sys_prompts and finetune_hotpot.py).
  3. model.generate(input_ids=query, softprompt=compressed_ctx, max_new_tokens=...)
     with the domain LoRA (PeftModel) applied — verbatim as external/lloco/inference.py.

Scoring: reuses OUR LongBench scorer verbatim (functions copied unmodified from
scripts/eval_longbench_mem_space.py: normalize_answer / compute_f1 / compute_f1_multi
/ compute_exact_match / compute_em_multi / run_scoring) so F1/EM are on the EXACT
same 口径 as our other LongBench baselines (CoMem / InfLLM / KV-Direct / HCache /
MemoryLLM). We do NOT import that module directly because it pulls
src.memory.mem_space which targets transformers 5.x, incompatible with the isolated
LLoCO env (transformers 4.37.2); copying keeps the metric bit-identical.

Data: LongBench narrativeqa/qasper/hotpotqa (local data/longbench_raw/data/*.jsonl,
200 samples each) — the SAME eval set as our other baselines, so this produces a
directly comparable head-to-head row. Note: LLoCO's own paper Table 4 numbers are
on the full tau/scrolls + hotpot_qa validation sets, so an exact Table-4 match is
not expected; Table 4 is used only as a ballpark 口径 sanity check.

Usage (single shard on one GPU):
    CUDA_VISIBLE_DEVICES=0 python scripts/eval_lloco_longbench.py \
        --dataset narrativeqa --num_shards 8 --shard_index 0 \
        --output_dir lloco_results/longbench

    # score only (aggregate shards):
    python scripts/eval_lloco_longbench.py --score_only \
        --datasets narrativeqa qasper hotpotqa --output_dir lloco_results/longbench
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import re
import string
import sys
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LLOCO_DIR = os.path.join(PROJECT_ROOT, "external", "lloco")
if LLOCO_DIR not in sys.path:
    sys.path.insert(0, LLOCO_DIR)

# --------------------------------------------------------------------------- #
# LLoCO native per-task decoder prompts (verbatim from the released repo).
#   external/lloco/finetune_scrolls.py : nqa_prompt, qasper_prompt
#   external/lloco/finetune_hotpot.py  : icl_prompt
# The document context is compressed into the soft prompt, so ONLY this
# instruction + question goes to the decoder — exactly the format each LoRA saw.
# --------------------------------------------------------------------------- #
NQA_PROMPT = (
    "You are given a story from above, which can be either a novel or a movie "
    "script, and a question. Answer the question as concisely as you can, using "
    "a single phrase if possible.\nQuestion:"
)
QASPER_PROMPT = (
    "You are just given an scientific article from above. I will now give you a "
    "question. Answer the question as concisely as you can, using a single phrase "
    "or sentence if possible. If the question cannot be answered based on the "
    "information in the article, write 'unanswerable'. If the question is a yes/no "
    "question, answer 'yes', 'no', or 'unanswerable'.\nQuestion: "
)
HQA_PROMPT = (
    "You were just given an article from above. I will now give you a question. "
    "Answer the question as concisely as you can, using a single phrase or "
    "sentence if possible.\nQuestion: "
)

# LongBench task name -> (native prompt, LoRA subdir, max_new_tokens)
# max_new_tokens follows OUR DATASET2MAXGEN (same as our other LongBench baselines).
TASK_CONFIG = {
    "narrativeqa": {"prompt": NQA_PROMPT, "lora": "Lloco-7b-nqa", "max_gen": 128},
    "qasper": {"prompt": QASPER_PROMPT, "lora": "Lloco-7b-qasper", "max_gen": 128},
    "hotpotqa": {"prompt": HQA_PROMPT, "lora": "Lloco-7b-hqa", "max_gen": 32},
}

DEFAULT_DATASETS = list(TASK_CONFIG.keys())


# =========================================================================== #
# SCORER — copied VERBATIM from scripts/eval_longbench_mem_space.py.
# DO NOT MODIFY: this guarantees identical F1/EM 口径 with our other baselines.
# =========================================================================== #
def normalize_answer(s: str) -> str:
    """Lower text, remove articles, punctuation, and extra whitespace."""
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Compute token-level F1 between prediction and a single ground truth."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()

    if len(prediction_tokens) == 0 and len(ground_truth_tokens) == 0:
        return 1.0
    if len(prediction_tokens) == 0 or len(ground_truth_tokens) == 0:
        return 0.0

    common = collections.Counter(prediction_tokens) & collections.Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(prediction_tokens)
    recall = num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_f1_multi(prediction: str, answers: list) -> float:
    """Max F1 across multiple reference answers."""
    if not answers:
        return 0.0
    return max(compute_f1(prediction, ans) for ans in answers)


def compute_exact_match(prediction: str, ground_truth: str) -> float:
    """Exact match after normalization."""
    return float(normalize_answer(prediction) == normalize_answer(ground_truth))


def compute_em_multi(prediction: str, answers: list) -> float:
    """Max EM across multiple reference answers."""
    if not answers:
        return 0.0
    return max(compute_exact_match(prediction, ans) for ans in answers)


def run_scoring(output_dir: str, datasets_list: list):
    """Aggregate prediction JSONL files and compute metrics.

    Copied verbatim from scripts/eval_longbench_mem_space.py (identical 口径)."""
    output_path = Path(output_dir)
    if not output_path.exists():
        print(f"[LLoCO] ERROR: output_dir does not exist: {output_dir}")
        return

    results = {}
    for ds_name in datasets_list:
        shard_files = sorted(output_path.glob(f"{ds_name}_*.jsonl"))
        if not shard_files:
            print(f"[LLoCO] No prediction files found for {ds_name}")
            continue

        predictions = []
        seen_indices = set()
        for sf in shard_files:
            with open(sf, "r") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    item = json.loads(line)
                    idx = item.get("index", len(predictions))
                    if idx not in seen_indices:
                        seen_indices.add(idx)
                        predictions.append(item)

        if not predictions:
            print(f"[LLoCO] No predictions for {ds_name}")
            continue

        f1_scores = []
        em_scores = []
        n_empty = 0
        for pred_item in predictions:
            pred = pred_item.get("pred", "")
            if not str(pred).strip():
                n_empty += 1
            answers = pred_item.get("answers", [])
            if isinstance(answers, str):
                try:
                    answers = json.loads(answers)
                except json.JSONDecodeError:
                    answers = [answers]
            if not isinstance(answers, list):
                answers = [answers]

            f1 = compute_f1_multi(pred, answers)
            em = compute_em_multi(pred, answers)
            f1_scores.append(f1)
            em_scores.append(em)

        avg_f1 = sum(f1_scores) / len(f1_scores) * 100
        avg_em = sum(em_scores) / len(em_scores) * 100
        results[ds_name] = {
            "f1": avg_f1,
            "em": avg_em,
            "n_samples": len(predictions),
            "n_empty_pred": n_empty,
        }
        print(f"[LLoCO] {ds_name:20s}: F1={avg_f1:.2f}  EM={avg_em:.2f}  "
              f"(n={len(predictions)}, empty_pred={n_empty})")

    if results:
        all_f1 = [v["f1"] for v in results.values()]
        all_em = [v["em"] for v in results.values()]
        avg_all_f1 = sum(all_f1) / len(all_f1)
        avg_all_em = sum(all_em) / len(all_em)
        print(f"\n[LLoCO] {'AVERAGE':20s}: F1={avg_all_f1:.2f}  EM={avg_all_em:.2f}  "
              f"(across {len(results)} datasets)")
        results["AVERAGE"] = {"f1": avg_all_f1, "em": avg_all_em, "n_datasets": len(results)}

    results_file = output_path / "scores.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[LLoCO] Results saved to: {results_file}")
    return results


# =========================================================================== #
# Data loading (local LongBench JSONL, matches eval_longbench_mem_space.py)
# =========================================================================== #
def load_longbench_task(ds_name: str, data_dir: str) -> list:
    local_path = os.path.join(data_dir, f"{ds_name}.jsonl")
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"LongBench file not found: {local_path}")
    samples = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            ans = item.get("answers", [])
            if isinstance(ans, str):
                try:
                    ans = json.loads(ans)
                except json.JSONDecodeError:
                    ans = [ans]
            if not isinstance(ans, list):
                ans = [ans]
            samples.append({
                "context": item.get("context", ""),
                "input": item.get("input", ""),
                "answers": ans,
                "dataset": ds_name,
            })
    return samples


# =========================================================================== #
# LLoCO model + compression
# =========================================================================== #
def load_lloco(autocomp_path: str, lora_path: str, device, dtype):
    from auto_compressor import LlocoAutoCompressorModel  # noqa: E402
    from peft import PeftModel  # noqa: E402
    from transformers import AutoTokenizer  # noqa: E402

    print(f"[LLoCO] Loading AutoCompressor: {autocomp_path}")
    model = LlocoAutoCompressorModel.from_pretrained(autocomp_path, torch_dtype=dtype)
    model = model.to(device)
    print(f"[LLoCO] Applying LoRA adapter: {lora_path}")
    model = PeftModel.from_pretrained(model, lora_path, torch_dtype=dtype)
    model = model.to(device).eval()

    tokenizer = AutoTokenizer.from_pretrained(autocomp_path)
    tokenizer.pad_token = "[PAD]"
    return model, tokenizer


@torch.inference_mode()
def compress_context(model, tokenizer, context: str, device, dtype):
    """Compress a document into a soft prompt.

    Replicates external/lloco/preproc_embs.preprocess_context_autocomp with
    truncation=False (full context, capped at 122880 tokens)."""
    if len(context) > 10000:
        toks = []
        for i in range(0, len(context), 10000):
            ctx = context[i:i + 10000]
            ids = tokenizer(ctx, add_special_tokens=False, return_tensors="pt",
                            truncation=False).input_ids  # [1, L]
            toks.append(ids)
        context_tokens = torch.cat(toks, dim=1)[:, :122880].to(device)
    else:
        context_tokens = tokenizer(context, add_special_tokens=False, max_length=120000,
                                   return_tensors="pt", truncation=False).input_ids.to(device)
    if context_tokens.shape[1] == 0:
        context_tokens = torch.tensor([[tokenizer.bos_token_id or 1]], device=device)
    softprompt = model(context_tokens.long(), segment_lengths=1536,
                       output_softprompt=True).softprompt  # [1, S, D]
    return softprompt.to(dtype), context_tokens.shape[1]


@torch.inference_mode()
def generate_answer(model, tokenizer, query_text: str, softprompt, max_gen: int, device):
    q_ids = tokenizer(query_text, add_special_tokens=False, return_tensors="pt").input_ids.to(device)
    prompt_len = q_ids.size(1)
    out = model.generate(input_ids=q_ids, softprompt=softprompt,
                         max_new_tokens=max_gen)[0]
    pred = tokenizer.decode(out[prompt_len:], skip_special_tokens=True)
    return pred.strip()


def main():
    ap = argparse.ArgumentParser(description="LLoCO LongBench eval driver")
    ap.add_argument("--dataset", type=str, default=None,
                    help="Single dataset to run (narrativeqa/qasper/hotpotqa)")
    ap.add_argument("--datasets", type=str, nargs="+", default=None,
                    help="Datasets for score_only mode")
    ap.add_argument("--data_dir", type=str,
                    default=os.path.join(PROJECT_ROOT, "data", "longbench_raw", "data"))
    ap.add_argument("--autocomp_path", type=str,
                    default=os.path.join(PROJECT_ROOT, "external", "lloco_weights",
                                         "AutoCompressor-Llama-2-7b-6k"))
    ap.add_argument("--lora_root", type=str,
                    default=os.path.join(PROJECT_ROOT, "external", "lloco_weights"))
    ap.add_argument("--output_dir", type=str,
                    default=os.path.join(PROJECT_ROOT, "lloco_results", "longbench"))
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--shard_index", type=int, default=0)
    ap.add_argument("--max_samples", type=int, default=-1)
    # float16 matches the released LLoCO inference.py (--fp16 True). NOTE: bf16
    # triggers a SIGFPE in the flash-attn 2.5.6 kv-cache decode kernel on H20
    # (sm_90); fp16 is both the faithful dtype and the stable one here.
    ap.add_argument("--dtype", type=str, default="float16",
                    choices=["bfloat16", "float16"])
    ap.add_argument("--score_only", action="store_true")
    args = ap.parse_args()

    if args.score_only:
        datasets_list = args.datasets or DEFAULT_DATASETS
        run_scoring(args.output_dir, datasets_list)
        return

    assert args.dataset in TASK_CONFIG, f"--dataset must be one of {list(TASK_CONFIG)}"
    ds_name = args.dataset
    cfg = TASK_CONFIG[ds_name]
    device = torch.device("cuda:0")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[args.dtype]

    lora_path = os.path.join(args.lora_root, cfg["lora"])
    max_gen = cfg["max_gen"]

    print(f"[LLoCO] ===== {ds_name} shard {args.shard_index}/{args.num_shards} =====")
    print(f"[LLoCO]   dtype={args.dtype} max_gen={max_gen} lora={lora_path}")

    model, tokenizer = load_lloco(args.autocomp_path, lora_path, device, dtype)

    samples = load_longbench_task(ds_name, args.data_dir)
    if args.max_samples > 0:
        samples = samples[:args.max_samples]

    # Shard by stride (index i handled by shard i%num_shards), keep global index.
    shard = [(i, s) for i, s in enumerate(samples)
             if i % args.num_shards == args.shard_index]
    if not shard:
        print(f"[LLoCO] shard {args.shard_index}: no samples")
        return

    out_path = Path(args.output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    outfile = out_path / f"{ds_name}_{args.shard_index}.jsonl"

    buffer = []
    t0 = time.time()
    for local_i, (gidx, sample) in enumerate(tqdm(
            shard, desc=f"{ds_name}[shard{args.shard_index}]", leave=True)):
        try:
            softprompt, n_ctx_tok = compress_context(
                model, tokenizer, sample["context"], device, dtype)
            query = cfg["prompt"] + sample["input"] + "\nAnswer:"
            pred = generate_answer(model, tokenizer, query, softprompt, max_gen, device)
        except Exception as e:  # noqa: BLE001
            print(f"[LLoCO] sample {gidx} FAILED: {type(e).__name__}: {e}")
            pred, n_ctx_tok = "", -1
        buffer.append({
            "index": gidx,
            "pred": pred,
            "answers": sample["answers"],
            "dataset": ds_name,
            "n_ctx_tokens": n_ctx_tok,
        })
        if (local_i + 1) % 5 == 0 or local_i == len(shard) - 1:
            with open(outfile, "w") as f:
                for r in buffer:
                    f.write(json.dumps(r, ensure_ascii=False) + "\n")
        if (local_i + 1) % 10 == 0:
            cur = [compute_f1_multi(r["pred"], r["answers"]) for r in buffer]
            print(f"  [{ds_name} s{args.shard_index}] {local_i+1}/{len(shard)} "
                  f"| {(local_i+1)/(time.time()-t0):.2f} smp/s "
                  f"| running F1={sum(cur)/len(cur)*100:.1f} "
                  f"| last='{pred[:50]}'")

    with open(outfile, "w") as f:
        for r in buffer:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    dt = time.time() - t0
    f1s = [compute_f1_multi(r["pred"], r["answers"]) for r in buffer]
    print(f"[LLoCO] {ds_name} shard {args.shard_index} done: "
          f"F1={sum(f1s)/len(f1s)*100:.2f} ({len(buffer)} samples in {dt:.1f}s)")


if __name__ == "__main__":
    main()
