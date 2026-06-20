"""BABILong evaluation for the official Landmark Attention ckpt (LLaMA-1-7B).

Evaluates the landmark-tuned model (external/landmark_ckpts/landmark_tuned/)
on BABILong qa1/qa2/qa5 tasks across 0k-32k context lengths.

Landmark inserts a <landmark> token every mem_freq (=50) tokens during
tokenization; its grouped-softmax attention then uses those boundary tokens
to retrieve top-k "blocks" when the sequence exceeds the training window.
We tokenize the full BABILong input and pass it directly to the pipeline
(no manual chunking needed — landmark handles long-context internally).

Usage (must use external/landmark_venv):
    cd /path/to/Mixture-of-Memory/external/landmark-attention/llama
    ../../landmark_venv/bin/python \
        ../../../scripts/run_babilong_landmark.py \
        --ckpt_path ../../landmark_ckpts/landmark_tuned \
        --output_name landmark_official \
        [--tasks qa1 qa2 qa5] [--lengths 0k 1k 2k 4k 8k 16k 32k] \
        [--top_k 5] [--device cuda:0] [--limit 100]

Note: must be run from the llama/ dir so that `import llama_mem` works.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

# ------------------------------------------------------------------ #
# Path setup — script is run from external/landmark-attention/llama/
# so llama_mem is importable directly. We still need to reach the
# repo root for babilong and datasets.
# ------------------------------------------------------------------ #

# llama/ dir (cwd when running) is already in sys.path implicitly;
# add repo root so we can import babilong third-party package.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))  # scripts/
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)                 # Mixture-of-Memory/
_BABILONG_PKG = os.path.join(_REPO_ROOT, "third_party", "babilong-pkg")
if _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import datasets  # noqa: E402
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input  # noqa: E402


# ------------------------------------------------------------------ #
# Model loading
# ------------------------------------------------------------------ #

def load_landmark_pipeline(ckpt_path: str, top_k: int, device: str, dtype=torch.bfloat16):
    """Load landmark LlamaForCausalLM and wrap in HF pipeline."""
    from llama_mem import LlamaForCausalLM  # noqa — must run from llama/ dir
    import transformers

    print(f"[landmark-BABILong] Loading model from: {ckpt_path}")
    model = LlamaForCausalLM.from_pretrained(
        ckpt_path,
        torch_dtype=dtype,
    )
    model.to(device)
    model.eval()

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        ckpt_path,
        model_max_length=model.config.train_context_length,
        padding_side="right",
        use_fast=False,
    )

    mem_id = tokenizer.convert_tokens_to_ids("<landmark>")
    if mem_id == tokenizer.unk_token_id:
        raise ValueError("<landmark> token not found in tokenizer vocab — wrong ckpt?")
    model.set_mem_id(mem_id)
    print(f"[landmark-BABILong] mem_id={mem_id}, train_context_length={model.config.train_context_length}, "
          f"mem_freq={model.config.mem_freq}, top_k={top_k}")

    from transformers import pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        offload_cache_to_cpu=False,
        use_flash=False,
        cache_top_k=top_k,
    )
    return pipe, tokenizer


# ------------------------------------------------------------------ #
# Generation
# ------------------------------------------------------------------ #

def insert_landmark_tokens(text: str, tokenizer, mem_freq: int) -> str:
    """Insert <landmark> tokens every mem_freq tokens into the text.

    Landmark's training pipeline inserts <landmark> tokens during data
    preparation so the model sees them as natural boundary markers.
    During eval we replicate this by tokenizing the text, inserting
    <landmark> token IDs every mem_freq positions, then decoding back.
    The pipeline then re-tokenizes the full string (with landmarks) before
    generating.
    """
    landmark_token = "<landmark>"
    ids = tokenizer.encode(text, add_special_tokens=False)
    new_ids = []
    for i, tok_id in enumerate(ids):
        new_ids.append(tok_id)
        if (i + 1) % mem_freq == 0:
            new_ids.append(tokenizer.convert_tokens_to_ids(landmark_token))
    return tokenizer.decode(new_ids, skip_special_tokens=False)


@torch.no_grad()
def generate_landmark(pipe, input_text: str, max_new_tokens: int) -> str:
    """Run landmark pipeline on input_text and return the new tokens only."""
    return generate_landmark_batch(pipe, [input_text], max_new_tokens, batch_size=1)[0]


@torch.no_grad()
def generate_landmark_batch(pipe, input_texts: list[str], max_new_tokens: int, batch_size: int) -> list[str]:
    """Run landmark pipeline on a batch of inputs and return new tokens only."""
    results = pipe(
        input_texts,
        num_return_sequences=1,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        batch_size=batch_size,
    )
    outputs = []
    for input_text, result in zip(input_texts, results):
        item = result[0] if isinstance(result, list) else result
        generated_text = item["generated_text"]
        outputs.append(generated_text[len(input_text):].strip())
    return outputs


# ------------------------------------------------------------------ #
# Main
# ------------------------------------------------------------------ #

def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for landmark attention")
    parser.add_argument("--ckpt_path", type=str,
                        default=None,
                        help="Path to landmark_tuned ckpt dir. "
                             "Defaults to ../../landmark_ckpts/landmark_tuned relative to script.")
    parser.add_argument("--results_folder", type=str, default=None,
                        help="Root folder for results. Defaults to <repo>/babilong_results.")
    parser.add_argument("--output_name", type=str, default="landmark_official",
                        help="Subfolder name for this eval run")
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa5"])
    parser.add_argument("--lengths", nargs="+",
                        default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"])
    parser.add_argument("--top_k", type=int, default=5,
                        help="Landmark cache_top_k: number of landmark blocks to retrieve")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Batch size passed to the text-generation pipeline.")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max samples per (task, length) cell. -1 = all.")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--insert_landmarks", action="store_true", default=True,
                        help="Insert <landmark> tokens every mem_freq positions before eval "
                             "(required for correct grouped-softmax retrieval)")
    parser.add_argument("--no_insert_landmarks", dest="insert_landmarks", action="store_false",
                        help="Skip landmark token insertion (ablation only)")
    parser.add_argument("--use_instruction", action="store_true", default=True)
    parser.add_argument("--use_examples", action="store_true", default=True)
    parser.add_argument("--use_post_prompt", action="store_true", default=True)
    args = parser.parse_args()

    # Resolve default paths relative to repo root
    if args.ckpt_path is None:
        args.ckpt_path = os.path.join(
            _REPO_ROOT, "external", "landmark_ckpts", "landmark_tuned"
        )
    if args.results_folder is None:
        args.results_folder = os.path.join(_REPO_ROOT, "babilong_results")

    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]

    print(f"[landmark-BABILong] ckpt:    {args.ckpt_path}")
    print(f"[landmark-BABILong] output:  {args.results_folder}/{args.output_name}")
    print(f"[landmark-BABILong] tasks:   {args.tasks}")
    print(f"[landmark-BABILong] lengths: {args.lengths}")
    print(f"[landmark-BABILong] top_k:   {args.top_k}, limit: {args.limit}")
    print(f"[landmark-BABILong] insert_landmarks: {args.insert_landmarks}")

    pipe, tokenizer = load_landmark_pipeline(
        ckpt_path=args.ckpt_path,
        top_k=args.top_k,
        device=args.device,
        dtype=dtype,
    )
    mem_freq = pipe.model.config.mem_freq  # 50

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if args.use_instruction else "",
            "examples":    DEFAULT_PROMPTS[task]["examples"]    if args.use_examples    else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if args.use_post_prompt else "",
            "template":    DEFAULT_TEMPLATE,
            "chat_template": False,
            "system_prompt": "",
        }

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[landmark-BABILong] task={task}, length={split_name}")

            try:
                data = datasets.load_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}.csv"
            cfg_file = outdir / f"{task}_{split_name}.json"

            json.dump(
                {
                    "ckpt_path": args.ckpt_path,
                    "task": task, "length": split_name,
                    "top_k": args.top_k,
                    "insert_landmarks": args.insert_landmarks,
                    "mem_freq": mem_freq,
                    "max_new_tokens": args.max_new_tokens,
                    "limit": args.limit,
                },
                open(cfg_file, "w"), indent=4,
            )

            num_samples = len(task_data)
            if args.limit > 0:
                num_samples = min(num_samples, args.limit)

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            batch_size = max(1, args.batch_size)
            for start in tqdm(range(0, num_samples, batch_size), desc=f"{task}/{split_name}", leave=False):
                end = min(start + batch_size, num_samples)
                batch_samples = [task_data[idx] for idx in range(start, end)]
                input_texts = []
                for sample in batch_samples:
                    input_text = get_formatted_input(
                        sample["input"],
                        sample["question"],
                        prompt_cfg["examples"],
                        prompt_cfg["instruction"],
                        prompt_cfg["post_prompt"],
                        template=prompt_cfg["template"],
                    )
                    if args.insert_landmarks:
                        input_text = insert_landmark_tokens(input_text, tokenizer, mem_freq)
                    input_texts.append(input_text)

                try:
                    outputs = generate_landmark_batch(
                        pipe, input_texts, args.max_new_tokens, batch_size=batch_size
                    )
                except Exception as exc:
                    print(f"[ERROR] batch start={start} generation failed: {exc}")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    outputs = []
                    for idx, input_text in enumerate(input_texts, start=start):
                        try:
                            outputs.append(generate_landmark(pipe, input_text, args.max_new_tokens))
                        except Exception as item_exc:
                            print(f"[ERROR] idx={idx} generation failed: {item_exc}")
                            outputs.append("")

                for sample, output in zip(batch_samples, outputs):
                    df.loc[len(df)] = [sample["target"], output, sample["question"]]
                if len(df) % 10 == 0 or end == num_samples:
                    df.to_csv(outfile, index=False, quoting=csv.QUOTE_ALL)

            print(f"[landmark-BABILong] Saved {len(df)} rows to {outfile}")

            # Quick accuracy report
            correct = sum(
                str(row["target"]).lower() in str(row["output"]).lower()
                for _, row in df.iterrows()
            )
            pct = 100 * correct / len(df) if len(df) > 0 else 0
            print(f"[landmark-BABILong] {task}/{split_name}: {correct}/{len(df)} = {pct:.1f}%")


if __name__ == "__main__":
    main()
