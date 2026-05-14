"""BABILong evaluation wrapper for MemoryLLM-8B-chat.

Reference: arXiv:2402.04624 (MemoryLLM, Wang et al. 2024)
ckpt: YuWangX/memoryllm-8b-chat (Apache 2.0)

MemoryLLM is a stateful Llama-3-8B + 12800 memory tokens per layer.
For BABILong: inject the long context via model.inject_memory(), then generate
with only the question. Memory is reset between samples.

Usage:
    python scripts/run_babilong_memoryllm.py [--tasks qa1 qa2 ...] [--lengths 0k 1k ...]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from tqdm.auto import tqdm

# MemoryLLM-source must be on PYTHONPATH so `modeling_memoryllm` can be imported
MEMORYLLM_SRC = "/apdcephfs_wzc1/share_303098609/pighzliu_code/MemoryLLM-source"
if MEMORYLLM_SRC not in sys.path:
    sys.path.insert(0, MEMORYLLM_SRC)

# Add babilong to path
BABILONG_ROOT = "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong"
sys.path.insert(0, BABILONG_ROOT)

import datasets
from transformers import AutoTokenizer

from modeling_memoryllm import MemoryLLM
from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input


def load_model(model_path: str, device: str = "cuda:0"):
    """Load MemoryLLM from local path."""
    print(f"[MemoryLLM-BABILong] Loading from: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # transformers 5.x removed rope_theta as a top-level config attr; MemoryLLM expects 4.x
    # Patch: read config.json directly and inject rope_theta back onto LlamaConfig
    from transformers import AutoConfig
    import json
    config = AutoConfig.from_pretrained(model_path)
    cfg_path = os.path.join(model_path, "config.json")
    with open(cfg_path) as f:
        raw = json.load(f)
    if "rope_theta" not in raw and isinstance(raw.get("rope_scaling"), dict):
        raw_rope_theta = raw["rope_scaling"].get("rope_theta", 500000.0)
    else:
        raw_rope_theta = raw.get("rope_theta", 500000.0)
    config.rope_theta = raw_rope_theta
    print(f"[MemoryLLM-BABILong] Patched config.rope_theta = {raw_rope_theta}")

    # First try flash_attention_2; fall back to sdpa
    try:
        model = MemoryLLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        print("[MemoryLLM-BABILong] Loaded with flash_attention_2")
    except (ValueError, ImportError) as e:
        print(f"[MemoryLLM-BABILong] FA2 failed: {e}; falling back to sdpa")
        model = MemoryLLM.from_pretrained(
            model_path,
            config=config,
            attn_implementation="sdpa",
            torch_dtype=torch.bfloat16,
        )

    model = model.to(device)
    model.eval()

    # Save initial memory state for reset between samples
    if hasattr(model, "memory"):
        # MemoryLLM has model.memory tensor — store a copy for reset
        initial_memory = model.memory.detach().clone()
    else:
        initial_memory = None
        print("[MemoryLLM-BABILong] WARNING: model has no .memory attribute; using save/restore alt path")

    return model, tokenizer, initial_memory


def reset_memory(model, initial_memory):
    """Reset model.memory to its initial state for a new sample."""
    if initial_memory is not None and hasattr(model, "memory"):
        model.memory.copy_(initial_memory)


def inject_long_context(model, tokenizer, context: str, device: str, max_chunk: int = 1024):
    """
    Inject a long context into MemoryLLM via inject_memory.

    The model's training-time chunk size is 1024 (default for MemoryLLM-8B).
    We chunk and inject sequentially so memory accumulates.
    """
    if not context or not context.strip():
        return

    ids = tokenizer(context, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    total_len = ids.shape[1]

    # Skip if context is too short — paper says <16 tokens disturbs memory
    if total_len < 16:
        return

    # Chunk into pieces of max_chunk (1024) tokens
    pos = 0
    while pos < total_len:
        chunk = ids[:, pos:pos + max_chunk]
        if chunk.shape[1] >= 16:
            with torch.no_grad():
                model.inject_memory(chunk, update_memory=True)
        pos += max_chunk


def generate_answer(model, tokenizer, question_prompt: str, device: str, max_new_tokens: int = 20):
    """Generate answer for the (already-memory-loaded) question."""
    # Use chat template (this is the "chat" variant)
    messages = [{"role": "user", "content": question_prompt}]
    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt", add_generation_prompt=True)
    # Drop BOS (model has its own trained bos)
    inputs = inputs[:, 1:].to(device)

    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>"),
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=max_new_tokens,
            eos_token_id=terminators,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        )

    # Strip prefix
    output_ids = outputs[0][inputs.shape[1]:]
    return tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="BABILong eval wrapper for MemoryLLM-8B-chat")
    parser.add_argument(
        "--model_path",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/baselines/memoryllm-8b-chat",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results",
    )
    parser.add_argument("--output_name", type=str, default="MemoryLLM-8B-chat")
    parser.add_argument(
        "--tasks", type=str, nargs="+", default=["qa1", "qa2", "qa3", "qa4", "qa5"]
    )
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"],
    )
    parser.add_argument("--dataset_name", type=str, default="RMT-team/babilong")
    parser.add_argument("--max_new_tokens", type=int, default=20)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--limit", type=int, default=None, help="If set, only run first N samples")
    args = parser.parse_args()

    model, tokenizer, initial_memory = load_model(args.model_path, args.device)

    use_chat_template = True
    use_instruction = True
    use_examples = True
    use_post_prompt = True

    suffix_parts = [
        "instruction_yes" if use_instruction else "instruction_no",
        "examples_yes" if use_examples else "examples_no",
        "post_prompt_yes" if use_post_prompt else "post_prompt_no",
        "chat_template_yes" if use_chat_template else "chat_template_no",
        "system_prompt_no",
    ]
    suffix = "_" + "_".join(suffix_parts) + ".csv"

    results_dir = Path(args.results_folder) / args.output_name
    results_dir.mkdir(parents=True, exist_ok=True)

    for task in tqdm(args.tasks, desc="tasks"):
        instruction = DEFAULT_PROMPTS[task].get("instruction", "") if use_instruction else ""
        examples = DEFAULT_PROMPTS[task].get("examples", "") if use_examples else ""
        post_prompt = DEFAULT_PROMPTS[task].get("post_prompt", "") if use_post_prompt else ""

        for length in tqdm(args.lengths, desc=f"{task} lengths", leave=False):
            data = datasets.load_dataset(args.dataset_name, length, split=task)
            outfile = results_dir / f"{task}_{length}{suffix.lstrip('_')}"
            outfile = results_dir / f"{task}_{length}_{'_'.join(suffix_parts)}.csv"

            rows = []
            samples = list(data)
            if args.limit:
                samples = samples[: args.limit]

            for sample in tqdm(samples, desc=f"{task}/{length}", leave=False):
                target = sample["target"]
                context = sample["input"]
                question = sample["question"]

                # Reset memory
                reset_memory(model, initial_memory)

                # Inject long context
                inject_long_context(model, tokenizer, context, args.device)

                # Build question-only prompt (memory holds the context)
                question_prompt = get_formatted_input(
                    "",  # context already in memory
                    question,
                    examples,
                    instruction,
                    post_prompt,
                    template=DEFAULT_TEMPLATE,
                )

                output = generate_answer(
                    model, tokenizer, question_prompt, args.device, args.max_new_tokens
                )
                rows.append({"target": target, "output": output, "question": question})

            df = pd.DataFrame(rows, columns=["target", "output", "question"])
            df.to_csv(outfile, index=False)
            print(f"[MemoryLLM-BABILong] Saved {len(rows)} -> {outfile}")

    print("[MemoryLLM-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
