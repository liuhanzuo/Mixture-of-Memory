"""BABILong evaluation wrapper for H6 Cross-Attention Memory (dual-gate).

Evaluates H6 (Llama-3-8B + middle-layer cross-attention memory with LSTM-style
dual-gate writeback) on BABILong benchmark tasks (qa1-qa5) across multiple
context lengths (0k-32k).

H6 is stateful: input is chunked into 4096-token segments, memory slots
accumulate across chunks. Each new sample resets slots.

Usage:
    python scripts/run_babilong_h6.py [--tasks qa1 qa2 ...] [--lengths 0k 1k ...]
                                       [--ckpt_path ...] [--chunk_size 4096]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from tqdm.auto import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add babilong to path
BABILONG_ROOT = "/apdcephfs_wzc1/share_303098609/pighzliu_code/babilong"
sys.path.insert(0, BABILONG_ROOT)

import datasets
from transformers import AutoTokenizer, LlamaForCausalLM

from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input

# Import the model class and CrossAttentionMemoryV2 from our codebase
from src.memory.mem_space.selector import CrossAttentionMemoryV2


def build_h6_model(
    model_path: str,
    num_slots: int = 64,
    memory_write_layer: int = 16,
    memory_read_layers: str = "18,22,26,30",
    memory_init: str = "strided",
    use_dual_gate: bool = True,
    forget_bias_init: float = 1.0,
    input_bias_init: float = 0.0,
    device: torch.device = None,
) -> "CrossAttentionMemoryModel":
    """Build H6 CrossAttentionMemoryModel matching training configuration.

    We import the CrossAttentionMemoryModel class directly from the training
    script to ensure perfect architecture match.
    """
    # Import the model class from training script
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_cross_attn_memory",
        os.path.join(PROJECT_ROOT, "scripts", "train_cross_attn_memory.py"),
    )
    train_module = importlib.util.module_from_spec(spec)
    # Suppress argparse from running
    sys.modules["train_cross_attn_memory"] = train_module
    spec.loader.exec_module(train_module)
    CrossAttentionMemoryModel = train_module.CrossAttentionMemoryModel

    print(f"[H6-BABILong] Loading base model from: {model_path}")
    base_model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )

    model = CrossAttentionMemoryModel(
        base_model=base_model,
        num_slots=num_slots,
        top_k=8,
        full_finetune=True,
        use_memory=True,
        use_cross_attn_memory=True,
        gradient_checkpointing=False,  # Not needed for inference
        cross_attn_dropout=0.0,
        residual_scale=0.01,
        swa_window=0,
        write_lr=0.1,
        slot_forward=True,
        slot_isolated=False,
        memory_init=memory_init,
        recon_loss_weight=0.0,
        cross_chunk_propagation=False,
        middle_layer_memory=True,
        memory_write_layer=memory_write_layer,
        memory_read_layers=memory_read_layers,
        use_dual_gate=use_dual_gate,
        forget_bias_init=forget_bias_init,
        input_bias_init=input_bias_init,
        dual_gate_tanh_new=True,
    )

    return model


def load_h6_model(
    ckpt_path: str,
    model_path: str,
    num_slots: int = 64,
    memory_write_layer: int = 16,
    memory_read_layers: str = "18,22,26,30",
    memory_init: str = "strided",
    use_dual_gate: bool = True,
    forget_bias_init: float = 1.0,
    input_bias_init: float = 0.0,
    device: torch.device = None,
):
    """Load H6 model from checkpoint."""
    model = build_h6_model(
        model_path=model_path,
        num_slots=num_slots,
        memory_write_layer=memory_write_layer,
        memory_read_layers=memory_read_layers,
        memory_init=memory_init,
        use_dual_gate=use_dual_gate,
        forget_bias_init=forget_bias_init,
        input_bias_init=input_bias_init,
        device=device,
    )

    print(f"[H6-BABILong] Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # The checkpoint may store state_dict under different keys
    if "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # Assume the checkpoint IS the state_dict
        state_dict = ckpt

    # Strip DDP "module." prefix if present
    cleaned_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            cleaned_state_dict[k[7:]] = v
        else:
            cleaned_state_dict[k] = v

    # Load state dict (strict=False to handle minor mismatches)
    missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
    if missing:
        print(f"[H6-BABILong] WARNING: Missing keys ({len(missing)}): {missing[:5]}...")
    if unexpected:
        print(f"[H6-BABILong] WARNING: Unexpected keys ({len(unexpected)}): {unexpected[:5]}...")

    model = model.to(device).to(torch.bfloat16)
    model.eval()

    print(f"[H6-BABILong] Model loaded successfully. num_slots={num_slots}, "
          f"write_layer={memory_write_layer}, read_layers={memory_read_layers}")

    return model


def generate_with_h6_memory(
    model,
    input_ids: torch.Tensor,
    tokenizer,
    chunk_size: int = 4096,
    max_new_tokens: int = 20,
    device: torch.device = None,
) -> str:
    """Generate text using H6 with chunked memory accumulation.

    H6 is stateful with slot_forward + middle_layer_memory:
    1. Split input into chunk_size segments
    2. Process all prefix chunks to accumulate memory slots
    3. On the last chunk, get the logits at the last real token position
    4. Autoregressively generate from there
    """
    if device is None:
        device = next(model.parameters()).device

    seq_len = input_ids.shape[1]  # [1, total_len]

    # Split input_ids into chunks
    tokens = input_ids[0]  # [total_len]
    chunks = tokens.split(chunk_size)

    # Process all prefix chunks (accumulate memory)
    if len(chunks) > 1:
        for chunk in chunks[:-1]:
            chunk_tensor = chunk.unsqueeze(0).to(device)  # [1, chunk_size]
            with torch.no_grad():
                # forward_chunk accumulates slot state internally
                model.forward_chunk(chunk_tensor, enable_write_grad=False)

    # Process the last chunk to get logits
    last_chunk = chunks[-1]
    last_chunk_len = last_chunk.shape[0]
    last_chunk_tensor = last_chunk.unsqueeze(0).to(device)  # [1, last_chunk_len]

    with torch.no_grad():
        result = model.forward_chunk(last_chunk_tensor, enable_write_grad=False)

    logits = result["logits"]  # [1, last_chunk_len, vocab_size]

    # Get prediction from the last token position
    next_token_logits = logits[:, -1, :]  # [1, vocab_size]
    # Suppress EOS for first token to force meaningful output
    next_token_logits[:, tokenizer.eos_token_id] = float("-inf")
    next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)  # [1, 1]
    generated_ids = [next_token_id.item()]

    # Autoregressive generation for remaining tokens
    for _ in range(max_new_tokens - 1):
        if generated_ids[-1] == tokenizer.eos_token_id:
            break

        # Feed the newly generated token as a single-token chunk
        # This updates the slot state and produces next logits
        gen_input = torch.tensor([[generated_ids[-1]]], dtype=torch.long, device=device)
        with torch.no_grad():
            result = model.forward_chunk(gen_input, enable_write_grad=False)

        logits = result["logits"]  # [1, 1, vocab_size]
        next_token_logits = logits[:, -1, :]
        next_token_id = next_token_logits.argmax(dim=-1, keepdim=True)
        generated_ids.append(next_token_id.item())

    # Strip EOS if present
    if tokenizer.eos_token_id in generated_ids:
        eos_idx = generated_ids.index(tokenizer.eos_token_id)
        generated_ids = generated_ids[:eos_idx]

    output_text = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
    return output_text


def main():
    parser = argparse.ArgumentParser(description="BABILong evaluation for H6 cross-attention memory")
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/outputs/experiment_h6_dual_gate/step_1000.pt",
        help="Path to H6 checkpoint",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b",
        help="Path to base Llama-3-8B model (for tokenizer and config)",
    )
    parser.add_argument(
        "--results_folder",
        type=str,
        default="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/babilong_results",
        help="Folder to store results",
    )
    parser.add_argument(
        "--output_name",
        type=str,
        default="H6-step1000",
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
        default=4096,
        help="Chunk size for memory accumulation (matches H6 training seq_len)",
    )
    parser.add_argument(
        "--num_slots",
        type=int,
        default=64,
        help="Number of memory slots",
    )
    parser.add_argument(
        "--memory_write_layer",
        type=int,
        default=16,
        help="Layer index for memory write",
    )
    parser.add_argument(
        "--memory_read_layers",
        type=str,
        default="18,22,26,30",
        help="Comma-separated layer indices for memory read",
    )
    parser.add_argument(
        "--memory_init",
        type=str,
        default="strided",
        help="Memory initialization method",
    )
    parser.add_argument(
        "--use_dual_gate",
        action="store_true",
        default=True,
        help="Use dual-gate writeback (H6 default)",
    )
    parser.add_argument(
        "--no_dual_gate",
        action="store_true",
        default=False,
        help="Disable dual-gate (for H5-style eval)",
    )
    parser.add_argument(
        "--forget_bias_init",
        type=float,
        default=1.0,
        help="Forget gate bias init",
    )
    parser.add_argument(
        "--input_bias_init",
        type=float,
        default=0.0,
        help="Input gate bias init",
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

    # Handle dual-gate flag
    use_dual_gate = args.use_dual_gate and not args.no_dual_gate

    print(f"[H6-BABILong] Configuration:")
    print(f"  Checkpoint: {args.ckpt_path}")
    print(f"  Base model: {args.model_path}")
    print(f"  Tasks: {args.tasks}")
    print(f"  Lengths: {args.lengths}")
    print(f"  Chunk size: {args.chunk_size}")
    print(f"  Num slots: {args.num_slots}")
    print(f"  Write layer: {args.memory_write_layer}")
    print(f"  Read layers: {args.memory_read_layers}")
    print(f"  Dual-gate: {use_dual_gate}")
    print(f"  Device: {args.device}")

    device = torch.device(args.device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[H6-BABILong] Tokenizer loaded. Vocab size: {tokenizer.vocab_size}")

    # Load model
    model = load_h6_model(
        ckpt_path=args.ckpt_path,
        model_path=args.model_path,
        num_slots=args.num_slots,
        memory_write_layer=args.memory_write_layer,
        memory_read_layers=args.memory_read_layers,
        memory_init=args.memory_init,
        use_dual_gate=use_dual_gate,
        forget_bias_init=args.forget_bias_init,
        input_bias_init=args.input_bias_init,
        device=device,
    )

    # Prompt configuration (matching LM2 wrapper)
    use_instruction = True
    use_examples = True
    use_post_prompt = True
    use_chat_template = False

    for task in tqdm(args.tasks, desc="tasks"):
        if task not in DEFAULT_PROMPTS:
            print(f"[WARNING] Task {task} not in DEFAULT_PROMPTS, skipping.")
            continue

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
            print(f"\n[H6-BABILong] Evaluating task={task}, length={split_name}")

            # Load dataset split
            try:
                data = datasets.load_dataset(args.dataset_name, split_name)
                task_data = data[task]
            except Exception as e:
                print(f"[ERROR] Failed to load dataset {args.dataset_name}/{split_name}/{task}: {e}")
                continue

            # Prepare output directory
            outdir = Path(args.results_folder) / args.output_name
            outdir.mkdir(parents=True, exist_ok=True)
            outfile = outdir / f"{task}_{split_name}_{prompt_name}.csv"
            cfg_file = outdir / f"{task}_{split_name}_{prompt_name}.json"

            # Save config
            json.dump(
                {
                    "prompt": prompt_cfg,
                    "generate_kwargs": {
                        "max_new_tokens": args.max_new_tokens,
                        "do_sample": False,
                        "num_beams": 1,
                    },
                    "model": {
                        "ckpt_path": args.ckpt_path,
                        "num_slots": args.num_slots,
                        "chunk_size": args.chunk_size,
                        "memory_write_layer": args.memory_write_layer,
                        "memory_read_layers": args.memory_read_layers,
                        "use_dual_gate": use_dual_gate,
                    },
                },
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

                # Reset memory slots before each sample
                model.reset_slots()

                # Generate with chunked memory
                with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                    output = generate_with_h6_memory(
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
            print(f"[H6-BABILong] Saved {len(df)} results to {outfile}")

    print("\n[H6-BABILong] Evaluation complete!")


if __name__ == "__main__":
    main()
