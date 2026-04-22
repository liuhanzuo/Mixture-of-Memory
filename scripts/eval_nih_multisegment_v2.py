#!/usr/bin/env python3
"""
Multi-segment NIH evaluation for slot memory (v2 — fixed loading).

Uses SlotMemoryWrapper from src.memory.slot (the class actually used in training).
Loads model.safetensors as base model + slot_weights.pt for slot memory weights.
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("nih_multisegment_v2")

from transformers import AutoModelForCausalLM, AutoTokenizer
from src.memory.slot import SlotMemoryCompressor, SlotMemoryWrapper


def build_wrapper(model, slot_weights, segment_length=1024, device="cuda"):
    """Build SlotMemoryWrapper with loaded checkpoint weights."""
    num_slots = 16
    slot_dim = 256

    # Infer num_segments from initial_slots shape
    inferred_num_segments = 8
    if "initial_slots" in slot_weights:
        inferred_num_segments = slot_weights["initial_slots"].shape[0]
        logger.info(f"Inferred num_segments={inferred_num_segments} from initial_slots shape")

    compressor = SlotMemoryCompressor(
        hidden_dim=model.config.hidden_size,
        num_slots=num_slots,
        slot_dim=slot_dim,
        num_segments=inferred_num_segments,
    )

    # Cast all slot weights to bfloat16 to match model dtype
    state_map = {k: v.to(torch.bfloat16) for k, v in slot_weights.items()}

    # Keys in slot_weights.pt already match compressor state dict names exactly
    missing, unexpected = compressor.load_state_dict(state_map, strict=False)
    if missing:
        logger.warning(f"Missing compressor params: {missing}")
    if unexpected:
        logger.warning(f"Unexpected params: {unexpected}")

    loaded = len(state_map) - len(unexpected)
    logger.info(f"Compressor: loaded {loaded}/{len(state_map)} params, missing={len(missing)}, unexpected={len(unexpected)}")

    wrapper = SlotMemoryWrapper(
        model=model,
        compressor=compressor,
        segment_length=segment_length,
    )
    wrapper = wrapper.to(torch.bfloat16).to(device)
    return wrapper


def generate_with_memory(wrapper, tokenizer, document_tokens: list, question: str,
                         segment_length: int, max_new_tokens: int = 20) -> str:
    """Process document in segments, then generate answer with memory prefix."""
    device = next(wrapper.parameters()).device
    B = 1
    doc_len = len(document_tokens)
    num_segments = max(1, doc_len // segment_length)

    compressor = wrapper.compressor
    old_slots = None

    inner = wrapper.model.get_base_model() if hasattr(wrapper.model, "get_base_model") else wrapper.model
    backbone = inner.model

    for seg_idx in range(num_segments):
        start = seg_idx * segment_length
        end = min(start + segment_length, doc_len)
        seg_ids = document_tokens[start:end]
        if len(seg_ids) == 0:
            continue

        if old_slots is None:
            slots = compressor.get_initial_slots(seg_idx, B, device, torch.bfloat16)
        else:
            slots = old_slots

        mem_tokens = compressor.slots_to_memory_tokens(slots)
        seg_tensor = torch.tensor([seg_ids], dtype=torch.long, device=device)

        token_embeds = wrapper._embed_with_memory(seg_tensor, mem_tokens)
        K = mem_tokens.shape[1]
        T = len(seg_ids)

        attn_mask = wrapper._build_attention_mask(T, K, device)
        attn_mask_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_float = torch.zeros_like(attn_mask_4d, dtype=torch.bfloat16)
        attn_float = attn_float.masked_fill(~attn_mask_4d, torch.tensor(float('-inf'), dtype=torch.bfloat16))
        position_ids = wrapper._build_position_ids(T, K, device).unsqueeze(0)

        outputs = backbone(inputs_embeds=token_embeds, attention_mask=attn_float, position_ids=position_ids)
        seg_hidden = outputs.last_hidden_state[:, K:, :]

        new_slots, _ = compressor(seg_hidden, old_slots=old_slots, compute_recon=False)
        old_slots = new_slots.detach()

    # Generate from question with memory prefix using manual token-by-token loop
    # model.generate() doesn't support our custom attention pattern
    if old_slots is not None:
        mem_tokens = compressor.slots_to_memory_tokens(old_slots)
    else:
        mem_tokens = torch.zeros(B, compressor.num_slots, compressor.hidden_dim, device=device, dtype=torch.bfloat16)

    q_ids = tokenizer.encode(question, add_special_tokens=False)
    q_tensor = torch.tensor([q_ids], dtype=torch.long, device=device)

    # Build input: memory tokens (as embeddings) + question tokens
    token_embeds = wrapper._embed_with_memory(q_tensor, mem_tokens)  # [1, K+T, D]
    K = mem_tokens.shape[1]
    embed_layer = inner.get_input_embeddings()
    neg_inf = torch.tensor(float('-inf'), dtype=torch.bfloat16, device=device)

    all_embeds = token_embeds
    generated = []
    for step in range(max_new_tokens):
        content_len = all_embeds.shape[1] - K
        attn_mask = wrapper._build_attention_mask(content_len, K, device)
        attn_4d = attn_mask.unsqueeze(0).unsqueeze(0)
        attn_float = torch.zeros_like(attn_4d, dtype=torch.bfloat16)
        attn_float = attn_float.masked_fill(~attn_4d, neg_inf)
        position_ids = wrapper._build_position_ids(content_len, K, device).unsqueeze(0)

        outputs = backbone(inputs_embeds=all_embeds, attention_mask=attn_float, position_ids=position_ids)
        next_logits = inner.lm_head(outputs.last_hidden_state[:, -1:, :])
        next_token = torch.argmax(next_logits, dim=-1).squeeze(-1).item()
        generated.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

        next_embed = embed_layer(torch.tensor([[next_token]], device=device))  # [1, 1, D]
        all_embeds = torch.cat([all_embeds, next_embed], dim=1)

    return tokenizer.decode(generated, skip_special_tokens=True).strip()


@torch.no_grad()
def generate_baseline(model, tokenizer, document_text: str, question: str,
                      max_new_tokens: int = 20) -> str:
    """Single-segment baseline: just feed everything to the model."""
    prompt = document_text + "\n\nQuestion: " + question + "\nAnswer: The answer is"
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                           pad_token_id=tokenizer.pad_token_id)
    generated = output[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def check_answer(generated: str, expected_code: str) -> bool:
    return expected_code.upper() in generated.upper()


def make_nih_documents(context_length: int, num_facts: int = 10,
                       seed: int = 42) -> tuple:
    """Generate NIH-style documents with inserted facts.

    Returns (document_text, question, answer, facts_list).
    """
    rng = random.Random(seed)
    facts = []
    for i in range(num_facts):
        code = chr(ord('A') + i)
        value = rng.randint(100, 999)
        facts.append((code, value))

    # Build document: repeated filler chunks with facts distributed throughout
    filler = "The research continued to explore various aspects of the methodology. "
    filler_tokens_per_fact = context_length // (num_facts * 6)  # rough tokens per fact

    doc_parts = []
    for code, value in facts:
        filler_chunk = filler * filler_tokens_per_fact
        doc_parts.append(f"{filler_chunk}Passage {code} states that the value is {value}. ")
    # Pad to target length
    doc_text = " ".join(doc_parts)
    while len(doc_text.split()) < context_length // 1.3:
        doc_text += filler

    # Question: ask about a random subset of facts
    rng2 = random.Random(seed + 1)
    asked = rng2.sample(facts, min(5, len(facts)))
    q_parts = [f"What is the value for {code}?" for code, _ in asked]
    question = " ".join(q_parts) + " Answer with just the numbers in order."
    answer = " ".join(str(v) for _, v in asked)

    return doc_text, question, answer, facts


def run_evaluation(model_name: str, stage2_path: str, stage3_path: str,
                   output_dir: str, device: str = "cuda:0",
                   num_trials: int = 3, segment_length: int = 1024):
    """Run full multi-segment NIH evaluation."""
    logger.info("=" * 60)
    logger.info("Multi-Segment NIH Evaluation for Slot Memory")
    logger.info("=" * 60)

    os.makedirs(output_dir, exist_ok=True)
    results = {"config": {}, "trials": [], "summary": {}}

    # Load tokenizer
    logger.info(f"Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load base model
    logger.info(f"Loading base model from {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    base_model.eval()

    context_lengths = [2048, 4096, 6144]
    checkpoints = []

    if stage2_path and os.path.exists(os.path.join(stage2_path, "slot_weights.pt")):
        checkpoints.append(("stage2", stage2_path))
    if stage3_path and os.path.exists(os.path.join(stage3_path, "slot_weights.pt")):
        checkpoints.append(("stage3", stage3_path))

    if not checkpoints:
        logger.error("No slot_weights.pt found in any checkpoint path!")
        return results

    for ctx_len in context_lengths:
        logger.info(f"\n{'='*50}\nContext length: {ctx_len}\n{'='*50}")

        for trial in range(num_trials):
            seed = 42 + trial * 1000 + ctx_len
            doc_text, question, answer, facts = make_nih_documents(ctx_len, seed=seed)
            doc_tokens = tokenizer.encode(doc_text, add_special_tokens=False)

            trial_result = {
                "context_length": ctx_len,
                "trial": trial + 1,
                "seed": seed,
                "num_doc_tokens": len(doc_tokens),
                "num_segments": len(doc_tokens) // segment_length,
                "answer": answer,
            }

            # Evaluate each checkpoint
            for ckpt_name, ckpt_path in checkpoints:
                try:
                    slot_weights_path = os.path.join(ckpt_path, "slot_weights.pt")
                    logger.info(f"  Loading {ckpt_name} from {slot_weights_path}...")
                    slot_weights = torch.load(slot_weights_path, map_location=device, weights_only=False)

                    wrapper = build_wrapper(base_model, slot_weights, segment_length=segment_length, device=device)
                    wrapper.eval()

                    with torch.no_grad():
                        generated = generate_with_memory(
                            wrapper, tokenizer, doc_tokens, question,
                            segment_length=segment_length, max_new_tokens=20,
                        )

                    correct = check_answer(generated, answer)
                    trial_result[f"{ckpt_name}_generated"] = generated
                    trial_result[f"{ckpt_name}_correct"] = correct
                    logger.info(f"  {ckpt_name}: {'✓' if correct else '✗'} | Q: {question[:60]}... | A: {answer} | Gen: {generated[:80]}")
                except Exception as e:
                    import traceback; traceback.print_exc()
                    logger.error(f"  {ckpt_name}: ❌ Error: {e}")
                    trial_result[f"{ckpt_name}_generated"] = ""
                    trial_result[f"{ckpt_name}_correct"] = False

            # Evaluate baseline (no memory)
            try:
                with torch.no_grad():
                    generated = generate_baseline(base_model, tokenizer, doc_text, question)
                correct = check_answer(generated, answer)
                trial_result["baseline_generated"] = generated
                trial_result["baseline_correct"] = correct
                logger.info(f"  baseline: {'✓' if correct else '✗'} | Gen: {generated[:80]}")
            except Exception as e:
                import traceback; traceback.print_exc()
                logger.error(f"  baseline: ❌ Error: {e}")
                trial_result["baseline_generated"] = ""
                trial_result["baseline_correct"] = False

            results["trials"].append(trial_result)

    # Compute summary
    for ckpt_name, _ in checkpoints:
        for ctx_len in context_lengths:
            trials = [t for t in results["trials"] if t["context_length"] == ctx_len]
            total = len(trials)
            correct = sum(1 for t in trials if t.get(f"{ckpt_name}_correct", False))
            rate = correct / total if total > 0 else 0
            results["summary"][f"{ckpt_name}_{ctx_len}"] = {
                "accuracy": rate,
                "correct": correct,
                "total": total,
            }

        baseline_trials = [t for t in results["trials"]]
        baseline_correct = sum(1 for t in baseline_trials if t.get("baseline_correct", False))
        results["summary"]["baseline"] = {
            "accuracy": baseline_correct / max(1, len(baseline_trials)),
            "correct": baseline_correct,
            "total": len(baseline_trials),
        }

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for key, val in results["summary"].items():
        logger.info(f"  {key}: {val['accuracy']:.1%} ({val['correct']}/{val['total']})")

    # Save results
    out_path = os.path.join(output_dir, f"nih_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to {out_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-segment NIH evaluation for slot memory")
    parser.add_argument("--model_path", required=True, help="Path to base model")
    parser.add_argument("--stage2_path", default="", help="Path to Stage 2 checkpoint")
    parser.add_argument("--stage3_path", default="", help="Path to Stage 3 checkpoint")
    parser.add_argument("--output_dir", default="outputs/nih_multisegment", help="Output directory")
    parser.add_argument("--device", default="cuda:0", help="Device to use")
    parser.add_argument("--num_trials", type=int, default=3, help="Number of trials per config")
    parser.add_argument("--segment_length", type=int, default=1024, help="Segment length")

    args = parser.parse_args()
    run_evaluation(
        model_name=args.model_path,
        stage2_path=args.stage2_path,
        stage3_path=args.stage3_path,
        output_dir=args.output_dir,
        device=args.device,
        num_trials=args.num_trials,
        segment_length=args.segment_length,
    )
