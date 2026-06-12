"""Slot Memory Evaluation — Needle-in-a-Haystack.

Loads a trained SlotMemoryModel and evaluates retrieval accuracy
at different depths and context lengths.
"""

import os
import sys
import json
import math
import argparse
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.memory.slot.slot_memory_compressor import SlotMemoryCompressor, SlotMemoryWrapper

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def generate_nih_sample(
    tokenizer,
    context_length: int,
    depth: float,
    needle_text: str,
    answer_text: str,
    haystack_text: str,
) -> tuple:
    """Generate a Needle-in-a-Haystack sample.

    Returns (input_ids, question_ids, answer_text).
    """
    # Build context with needle at specified depth
    needle_len = len(tokenizer.encode(needle_text))
    total_content_tokens = context_length - needle_len - 50  # reserve for question
    insert_pos = int(depth * total_content_tokens)

    # Build haystack
    haystack_tokens = tokenizer.encode(haystack_text, add_special_tokens=False)
    # Repeat haystack to fill
    repeated = []
    while len(repeated) < total_content_tokens:
        repeated.extend(haystack_tokens)
    repeated = repeated[:total_content_tokens]

    # Insert needle
    needle_tokens = tokenizer.encode(needle_text, add_special_tokens=False)
    context_tokens = repeated[:insert_pos] + needle_tokens + repeated[insert_pos:]

    # Pad to context_length
    context_tokens = context_tokens[:context_length]

    # Question
    question = "\n\nBased on the text above, what is the special secret code mentioned in the document?"
    question_tokens = tokenizer.encode(question, add_special_tokens=False)[:100]

    return (
        torch.tensor(context_tokens, dtype=torch.long),
        torch.tensor(question_tokens, dtype=torch.long),
        answer_text,
    )


def evaluate_nih(
    model,
    tokenizer,
    num_trials: int = 5,
    context_lengths: List[int] = None,
    depths: List[float] = None,
    device: str = "cuda:0",
) -> Dict:
    """Run Needle-in-a-Haystack evaluation."""
    if context_lengths is None:
        context_lengths = [1024, 2048, 4096]
    if depths is None:
        depths = [0.0, 0.25, 0.5, 0.75, 1.0]

    needle_text = "the special secret code is Blue Monkey 42"
    answer_text = "Blue Monkey 42"
    haystack = "The history of computing spans several centuries. Early mechanical devices " * 50

    results = {"trials": [], "summary": {}}

    for ctx_len in context_lengths:
        for depth in depths:
            correct = 0
            for trial in range(num_trials):
                context_ids, question_ids, answer = generate_nih_sample(
                    tokenizer, ctx_len, depth, needle_text, answer_text, haystack,
                )

                context_ids = context_ids.unsqueeze(0).to(device)
                question_ids = question_ids.unsqueeze(0).to(device)

                with torch.no_grad():
                    generated = generate_with_slot_memory(model, tokenizer, context_ids, question_ids, max_new_tokens=30, device=device)
                    decoded = tokenizer.decode(generated[0], skip_special_tokens=True).strip().lower()
                    is_correct = answer.lower() in decoded
                    if is_correct:
                        correct += 1

            accuracy = correct / num_trials
            result = {
                "context_length": ctx_len,
                "depth": depth,
                "accuracy": accuracy,
                "correct": correct,
                "total": num_trials,
            }
            results["trials"].append(result)
            key = f"ctx{ctx_len}_d{depth}"
            results["summary"][key] = accuracy
            logger.info(f"  ctx={ctx_len} depth={depth}: {accuracy:.0%} ({correct}/{num_trials})")

    overall = sum(r["accuracy"] for r in results["trials"]) / len(results["trials"]) if results["trials"] else 0
    results["summary"]["overall_accuracy"] = overall
    logger.info(f"Overall NiH accuracy: {overall:.2%}")
    return results


def generate_with_slot_memory(
    slot_model, tokenizer, context_ids, question_ids, max_new_tokens=20, device="cuda:0",
) -> torch.Tensor:
    """Process context segments with slot memory, then generate on question."""
    B = context_ids.shape[0]
    seg_len = slot_model.segment_length
    num_segs = context_ids.shape[1] // seg_len
    old_slots = None

    with torch.no_grad():
        # Process context segments
        for seg_idx in range(num_segs):
            seg_ids = context_ids[:, seg_idx * seg_len : (seg_idx + 1) * seg_len]
            if old_slots is None:
                slots = slot_model.compressor.get_initial_slots(
                    seg_idx, B, device, next(slot_model.parameters()).dtype
                )
            else:
                slots = old_slots
            mem_tokens = slot_model.compressor.slots_to_memory_tokens(slots)

            _, _, seg_hidden = slot_model.forward_segment(seg_ids, None, mem_tokens, stage="ce_only")
            new_slots, _ = slot_model.compressor(seg_hidden, old_slots=old_slots, compute_recon=False)
            old_slots = new_slots

        # Generate on question segment
        if old_slots is None:
            old_slots = slot_model.compressor.get_initial_slots(
                0, B, device, next(slot_model.parameters()).dtype
            )
        mem_tokens = slot_model.compressor.slots_to_memory_tokens(old_slots)
        inputs_embeds = slot_model._embed_with_memory(question_ids, mem_tokens)

        K = mem_tokens.shape[1]
        T = question_ids.shape[1]
        attn_mask = slot_model._build_attention_mask(T, K, device)
        attn_mask = attn_mask.unsqueeze(0).expand(B, -1, -1)
        position_ids = slot_model._build_position_ids(T, K, device).unsqueeze(0).expand(B, -1)

        dtype = next(slot_model.parameters()).dtype
        bool_mask_4d = attn_mask.unsqueeze(1)
        attn_mask_4d = torch.zeros_like(bool_mask_4d, dtype=dtype)
        attn_mask_4d = attn_mask_4d.masked_fill(~bool_mask_4d, float('-inf'))

        inner = slot_model.model.get_base_model() if hasattr(slot_model.model, 'get_base_model') else slot_model.model
        outputs = inner.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask_4d,
            position_ids=position_ids,
            use_cache=True,
        )
        logits = inner.lm_head(outputs.last_hidden_state)
        past_kv = outputs.past_key_values

        # Track current position for position_ids during generation
        cur_pos = position_ids[:, -1:]  # [B, 1]
        # Build causal attention mask for the single new token attending to all prior tokens
        total_prior = K + T  # memory + question tokens processed so far

        next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = [next_token]
        for step in range(max_new_tokens - 1):
            token_embeds = inner.get_input_embeddings()(next_token)
            # Causal mask: new token can attend to all prior tokens
            step_pos = cur_pos + 1
            # Build 1 x 1 x 1 x (total_prior+step) causal mask — new token sees everything before it
            step_total = total_prior + step + 1  # +1 for the current token's own KV
            step_mask = torch.zeros(B, 1, 1, step_total, dtype=dtype, device=device)
            outputs = inner.model(
                inputs_embeds=token_embeds,
                attention_mask=step_mask,
                position_ids=step_pos,
                past_key_values=past_kv,
                use_cache=True,
            )
            logits = inner.lm_head(outputs.last_hidden_state)
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past_kv = outputs.past_key_values
            cur_pos = step_pos
            generated.append(next_token)

    return torch.cat(generated, dim=1)


def main():
    parser = argparse.ArgumentParser(description="Slot Memory Evaluation")
    parser.add_argument("--checkpoint_dir", required=True, help="Checkpoint directory (final/)")
    parser.add_argument("--base_model", default="../models/Qwen--Qwen3-8b")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--slot_dim", type=int, default=256)
    parser.add_argument("--segment_length", type=int, default=1024)
    parser.add_argument("--max_segments", type=int, default=4)
    parser.add_argument("--num_trials", "--num_tests", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="eval_results/slot_memory")
    parser.add_argument("--context_lengths", type=str, default=None,
        help="Comma-separated context lengths to test (e.g. '512,1024,2048'). Default: [1024,2048,4096]")
    parser.add_argument("--depths", type=float, nargs="+", default=None,
        help="Depths to test (e.g. 0.25 0.5 0.75 1.0). Default: [0.0, 0.25, 0.5, 0.75, 1.0]")

    args = parser.parse_args()

    # Parse context_lengths
    if args.context_lengths:
        context_lengths_arg = [int(x.strip()) for x in args.context_lengths.split(",")]
    else:
        context_lengths_arg = None

    # Parse depths
    depths_arg = args.depths  # None means use defaults inside evaluate_nih

    logger.info(f"Loading model from {args.checkpoint_dir}")
    device = torch.device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    # Check for merged checkpoint (model.safetensors) vs LoRA adapter
    merged_path = os.path.join(args.checkpoint_dir, "model.safetensors")
    adapter_path = os.path.join(args.checkpoint_dir, "adapter_model.safetensors")

    if os.path.exists(merged_path):
        # Merged checkpoint — load directly, no LoRA needed
        logger.info(f"Detected merged checkpoint: {merged_path}")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.checkpoint_dir, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map={"": device},
        )
    else:
        # Load base model + LoRA adapter
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map={"": device},
        )
        if os.path.exists(adapter_path):
            logger.info(f"Loading LoRA adapter from {args.checkpoint_dir}")
            base_model = PeftModel.from_pretrained(base_model, args.checkpoint_dir)
        else:
            logger.warning(f"No model.safetensors or adapter_model.safetensors found in {args.checkpoint_dir}")

    # Load slot compressor (must match training architecture)
    compressor = SlotMemoryCompressor(
        hidden_dim=4096, num_slots=args.num_slots, slot_dim=args.slot_dim,
        num_iterations=3, dropout=0.1,
        num_segments=args.max_segments + 1,
    )
    slot_weights_path = os.path.join(args.checkpoint_dir, "slot_weights.pt")
    if os.path.exists(slot_weights_path):
        state = torch.load(slot_weights_path, map_location=device, weights_only=True)
        compressor.load_state_dict(state)
        logger.info(f"Loaded slot weights from {slot_weights_path}")

    # Build model using SlotMemoryWrapper (same as training)
    slot_model = SlotMemoryWrapper(
        model=base_model,
        compressor=compressor,
        segment_length=args.segment_length,
    )
    slot_model = slot_model.to(device=device, dtype=torch.bfloat16)
    slot_model.eval()

    # Run eval
    results = evaluate_nih(
        slot_model, tokenizer,
        num_trials=args.num_trials,
        context_lengths=context_lengths_arg,
        depths=depths_arg,
        device=args.device,
    )

    # Save results
    output_dir = args.output_dir + f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "nih_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to {output_dir}/nih_results.json")


if __name__ == "__main__":
    main()
