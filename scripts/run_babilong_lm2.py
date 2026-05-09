"""BABILong evaluation wrapper for LM2 (Large Memory Model).

Evaluates LM2 on BABILong benchmark tasks (qa1-qa5) across multiple context
lengths (0k-32k). Handles chunked memory accumulation for long contexts.

Usage:
    python scripts/run_babilong_lm2.py [--tasks qa1 qa2 ...] [--lengths 0k 1k ...]
                                        [--ckpt_path ...] [--chunk_size 2048]
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

# Set RANK for LM2's print0 utility (expects DDP env vars)
os.environ.setdefault("RANK", "0")
os.environ.setdefault("LOCAL_RANK", "0")
os.environ.setdefault("WORLD_SIZE", "1")

# Add LM2 source to path
LM2_ROOT = "/apdcephfs_wzc1/share_303098609/pighzliu_code/LM2"
sys.path.insert(0, LM2_ROOT)

# Add babilong to path
BABILONG_ROOT = "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong"
sys.path.insert(0, BABILONG_ROOT)

import datasets
from transformers import AutoConfig, AutoTokenizer

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input
from src.model_memory_llama import CustomLlamaConfig, LlamaMem


def build_config(model_name: str, memory_slots: int = 2048, batch_size: int = 1) -> CustomLlamaConfig:
    """Build CustomLlamaConfig matching the training configuration."""
    base_config = AutoConfig.from_pretrained(model_name).to_dict()
    config = CustomLlamaConfig(
        use_memory=True,
        memory_slots=memory_slots,
        num_mem_heads=8,
        log_freq=100,
        batch_size=batch_size,
        **base_config,
    )
    return config


def _patch_for_transformers5(model):
    """Monkey-patch LM2 model for transformers >= 5.x compatibility.

    LM2's model_memory_llama.py was written for transformers 4.x:
    1. LlamaModel._update_causal_mask() existed in 4.x, removed in 5.x
    2. LlamaAttention.forward() signature changed:
       4.x: (hidden_states, attention_mask, position_ids=..., position_embeddings=...)
       5.x: (hidden_states, position_embeddings=..., attention_mask=..., ...)
    """
    # Patch 1: _update_causal_mask
    if not hasattr(model.model, "_update_causal_mask"):
        def _update_causal_mask(attention_mask, input_tensor, cache_position, past_key_values, output_attentions=False):
            return None

        model.model._update_causal_mask = _update_causal_mask
        print("  [PATCH] Added _update_causal_mask compatibility shim")

    # Patch 2: Fix self_attn calling convention in MemoryAttention and decoder layers
    # In transformers 5.x, LlamaAttention.forward signature is:
    #   forward(hidden_states, position_embeddings=None, attention_mask=None, ...)
    # But LM2 calls it as:
    #   self.self_attn(hidden_states, attention_mask, position_ids=..., position_embeddings=...)
    # We need to wrap each self_attn to translate the old calling convention.
    from src.model_memory_llama import MemoryAttention, CustomLlamaDecoderLayer

    class SelfAttnWrapper(torch.nn.Module):
        """Wraps a transformers 5.x LlamaAttention to accept 4.x calling convention.

        Translates:
          4.x call: self_attn(hidden_states, attention_mask, position_ids=..., position_embeddings=...)
                    returns (attn_output, attn_weights, past_key_values)
          5.x call: self_attn(hidden_states, position_embeddings=..., attention_mask=...)
                    returns (attn_output, attn_weights)
        """

        def __init__(self, real_attn):
            super().__init__()
            self._real_attn = real_attn

        def forward(self, hidden_states, attention_mask=None, position_ids=None, position_embeddings=None, **kwargs):
            # Call with 5.x keyword-only convention
            result = self._real_attn(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=attention_mask,
            )
            # 5.x returns (attn_output, attn_weights) — pad to 3-tuple for 4.x compat
            if len(result) == 2:
                return result[0], result[1], None
            return result

        def __getattr__(self, name):
            if name in ("_real_attn", "training", "_parameters", "_buffers", "_modules"):
                return super().__getattr__(name)
            return getattr(self._real_attn, name)

    patched_count = 0
    for layer in model.model.layers:
        # Wrap the self_attn at the decoder layer level
        if hasattr(layer, 'self_attn') and not isinstance(layer.self_attn, SelfAttnWrapper):
            layer.self_attn = SelfAttnWrapper(layer.self_attn)
            patched_count += 1
        # Also wrap inside mem_attn if it holds a reference
        if hasattr(layer, 'mem_attn') and hasattr(layer.mem_attn, 'self_attn'):
            if not isinstance(layer.mem_attn.self_attn, SelfAttnWrapper):
                layer.mem_attn.self_attn = SelfAttnWrapper(layer.mem_attn.self_attn)

    print(f"  [PATCH] Wrapped {patched_count} LlamaAttention layers for transformers 5.x calling convention")


def load_lm2_model(
    ckpt_path: str,
    model_name: str,
    rank: int = 0,
    memory_slots: int = 2048,
) -> tuple:
    """Load LM2 model from checkpoint."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Peek at ckpt to get training batch_size (memory_bank shape)
    snapshot = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    train_bs = snapshot["model_state_dict"]["memory_bank"].shape[0]
    del snapshot
    print(f"  Detected training batch_size={train_bs} from checkpoint")

    # Build config with training batch_size so state_dict loads cleanly
    config = build_config(model_name, memory_slots=memory_slots, batch_size=train_bs)
    model = LlamaMem.from_ckpt(
        pretrained_ckpt_path=ckpt_path,
        config=config,
        tokenizer=tokenizer,
        rank=rank,
        load_memory=True,
    )
    model.eval()

    # Patch for transformers 5.x compatibility
    _patch_for_transformers5(model)

    # Reshape memory from training batch_size to 1 for inference
    D = model.memory.shape[-1]
    device = model.memory.device
    dtype = model.memory.dtype
    model.memory = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)
    print(f"  Memory reshaped to batch_size=1: {model.memory.shape}")

    return model, tokenizer


def reset_memory(model: LlamaMem, device: torch.device):
    """Reset memory to fresh identity matrix (batch_size=1)."""
    D = model.memory.shape[-1]  # memory_slots
    dtype = model.memory.dtype
    model.memory = torch.eye(D, device=device, dtype=dtype).unsqueeze(0)


def generate_with_memory(
    model: LlamaMem,
    input_ids: torch.Tensor,
    tokenizer: AutoTokenizer,
    chunk_size: int,
    max_new_tokens: int = 20,
    device: torch.device = None,
) -> str:
    """Generate text using LM2 with chunked memory accumulation.

    LM2's memory module requires T == memory_slots (2048) for every forward pass.
    Strategy:
    1. Pad input to be a multiple of chunk_size
    2. Process all but the last chunk to accumulate memory
    3. The last chunk (padded) is used to get initial prediction
    4. For autoregressive generation: build a new chunk_size window containing
       recent context + generated tokens, pad if needed
    """
    if device is None:
        device = next(model.parameters()).device

    pad_token_id = tokenizer.pad_token_id or tokenizer.eos_token_id
    seq_len = input_ids.shape[1]

    # Pad input to multiple of chunk_size
    pad_len = (chunk_size - (seq_len % chunk_size)) % chunk_size
    if pad_len > 0:
        padding = torch.full((1, pad_len), pad_token_id, dtype=input_ids.dtype, device=device)
        padded_ids = torch.cat([input_ids, padding], dim=1)
    else:
        padded_ids = input_ids

    total_padded_len = padded_ids.shape[1]
    num_chunks = total_padded_len // chunk_size

    # Process all prefix chunks (all except last) for memory accumulation
    if num_chunks > 1:
        for i in range(num_chunks - 1):
            start = i * chunk_size
            end = start + chunk_size
            chunk = padded_ids[:, start:end]
            with torch.no_grad():
                _ = model(
                    input_ids=chunk,
                    targets=None,
                    attention_mask=None,
                    num_logits_to_keep=chunk_size,
                )

    # Process the last chunk to get logits at the prompt's last real token
    last_chunk_start = (num_chunks - 1) * chunk_size
    last_chunk = padded_ids[:, last_chunk_start:last_chunk_start + chunk_size]

    # The real prompt ends at position (seq_len - last_chunk_start - 1) within this chunk
    real_end_in_chunk = seq_len - last_chunk_start - 1  # 0-indexed position of last real token

    with torch.no_grad():
        logits, _, _ = model(
            input_ids=last_chunk,
            targets=None,
            attention_mask=None,
            num_logits_to_keep=chunk_size,
        )

    # Get prediction from the last real token position
    next_token_logits = logits[:, real_end_in_chunk, :]
    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
    generated_ids = [next_token_id.item()]

    # For remaining tokens: build a generation chunk with context
    # Use the tail of input + already generated tokens, padded to chunk_size
    for _ in range(max_new_tokens - 1):
        if generated_ids[-1] == tokenizer.eos_token_id:
            break

        # Build context: last (chunk_size - 1) tokens from input + all generated so far
        gen_tensor = torch.tensor([generated_ids], dtype=torch.long, device=device)
        # Combine original input with generated tokens
        full_seq = torch.cat([input_ids, gen_tensor], dim=1)
        # Take the last chunk_size tokens as context
        if full_seq.shape[1] >= chunk_size:
            gen_chunk = full_seq[:, -chunk_size:]
            target_pos = chunk_size - 1  # predict next token after last position
        else:
            # Pad to chunk_size
            pad_needed = chunk_size - full_seq.shape[1]
            gen_chunk = torch.cat([
                torch.full((1, pad_needed), pad_token_id, dtype=torch.long, device=device),
                full_seq
            ], dim=1)
            target_pos = chunk_size - 1

        with torch.no_grad():
            logits, _, _ = model(
                input_ids=gen_chunk,
                targets=None,
                attention_mask=None,
                num_logits_to_keep=chunk_size,
            )

        next_token_logits = logits[:, target_pos, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

    # Strip EOS if present
    if tokenizer.eos_token_id in generated_ids:
        eos_idx = generated_ids.index(tokenizer.eos_token_id)
        generated_ids = generated_ids[:eos_idx]

    # Decode generated tokens
    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output_text


def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for LM2")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/lm2_b200_4/ckpts_20260509_152240/ckpt_iter_12000.pth",
        help="Path to LM2 checkpoint",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama-3.2-1B",
        help="Path to base Llama model (for tokenizer and config)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results",
        help="Folder to store results",
    )
    parser.add_argument(
        "--results_name",
        type=str,
        default="LM2-iter12000",
        help="Subfolder name for this evaluation run",
    )
    parser.add_argument(
        "--tasks",
        type=str,
        nargs="+",
        default=["qa1", "qa2", "qa3", "qa4", "qa5"],
        help="BABILong tasks to evaluate",
    )
    parser.add_argument(
        "--lengths",
        type=str,
        nargs="+",
        default=["0k", "1k", "2k", "4k", "8k", "16k", "32k"],
        help="Context lengths to evaluate",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="RMT-team/babilong",
        help="HuggingFace dataset name",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=2048,
        help="Chunk size for memory accumulation",
    )
    parser.add_argument(
        "--memory_slots",
        type=int,
        default=2048,
        help="Number of memory slots",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=20,
        help="Maximum new tokens to generate",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum samples per task/length (-1 for all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run on",
    )
    args = parser.parse_args()

    print(f"[LM2-BABILong] Loading model from: {args.ckpt_path}")
    print(f"[LM2-BABILong] Base model: {args.model_name}")
    print(f"[LM2-BABILong] Tasks: {args.tasks}")
    print(f"[LM2-BABILong] Lengths: {args.lengths}")
    print(f"[LM2-BABILong] Chunk size: {args.chunk_size}")

    device = torch.device(args.device)
    rank = int(args.device.split(":")[-1]) if "cuda" in args.device else 0

    # Load model
    model, tokenizer = load_lm2_model(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        rank=rank,
        memory_slots=args.memory_slots,
    )
    print(f"[LM2-BABILong] Model loaded successfully. Memory shape: {model.memory.shape}")

    # Prompt configuration
    use_instruction = True
    use_examples = True
    use_post_prompt = True
    use_chat_template = False

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

        # Configure prompt
        prompt_cfg = {
            "instruction": DEFAULT_PROMPTS[task]["instruction"] if use_instruction else "",
            "examples": DEFAULT_PROMPTS[task]["examples"] if use_examples else "",
            "post_prompt": DEFAULT_PROMPTS[task]["post_prompt"] if use_post_prompt else "",
            "template": DEFAULT_TEMPLATE,
            "chat_template": use_chat_template,
            "system_prompt": "",
        }
        prompt_name = "_".join(
            [f"{k}_yes" if prompt_cfg[k] else f"{k}_no" for k in prompt_cfg if k != "template"]
        )

        for split_name in tqdm(args.lengths, desc="lengths", leave=False):
            print(f"\n[LM2-BABILong] Evaluating task={task}, length={split_name}")

            # Load dataset split
            try:
                data = datasets.load_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            # Prepare output directory
            outdir = Path(args.results_folder) / args.results_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}_{prompt_name}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}.json"

            # Save config
            json.dump(
                {"prompt": prompt_cfg, "generate_kwargs": {"max_new_tokens": args.max_new_tokens, "do_sample": False, "num_beams": 1}},
                open(cfg_file, "w"),
                indent=4,
            )

            df = pd.DataFrame({"target": [], "output": [], "question": []})

            num_samples = len(task_data)
            if args.max_samples > 0:
                num_samples = min(num_samples, args.max_samples)

            for idx in tqdm(range(num_samples), desc=f"{task}/{split_name}", leave=False):
                sample = task_data[idx]
                target = sample["target"]
                context = sample["input"]
                question = sample["question"]

                # Format input text
                input_text = get_formatted_input(
                    context,
                    question,
                    prompt_cfg["examples"],
                    prompt_cfg["instruction"],
                    prompt_cfg["post_prompt"],
                    template=prompt_cfg["template"],
                )

                # Tokenize
                input_ids = tokenizer.encode(input_text, add_special_tokens=True, return_tensors="pt")
                if isinstance(input_ids, list):
                    input_ids = torch.tensor([input_ids], dtype=torch.long)
                input_ids = input_ids.to(device)

                # Reset memory before each sample
                reset_memory(model, device)

                # Generate with chunked memory
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = generate_with_memory(
                        model=model,
                        input_ids=input_ids,
                        tokenizer=tokenizer,
                        chunk_size=args.chunk_size,
                        max_new_tokens=args.max_new_tokens,
                        device=device,
                    )

                df.loc[len(df)] = [target, output, question]

                # Write intermediate results
                if (idx + 1) % 10 == 0 or idx == num_samples - 1:
                    df.to_csv(outfile, index=False)

            # Final save
            df.to_csv(outfile, index=False)
            print(f"[LM2-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[LM2-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
