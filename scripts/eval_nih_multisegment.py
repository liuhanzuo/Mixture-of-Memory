#!/usr/bin/env python3
"""
Multi-segment NIH-Extended Benchmark: Test slot memory cross-segment information preservation.

Tests whether slot memory can preserve information across segment boundaries by:
1. Processing documents in multiple segments (2-4 segments)
2. Injecting compressed slot memory at each boundary
3. Testing at context lengths: 2048, 4096, 6144 tokens
4. Comparing Stage 2 vs Stage 3 slot memory checkpoints
5. Calculating retention: (multi-segment accuracy) / (single-segment accuracy)

Success criteria:
- <10% retention: slot attention collapse
- 10-50% retention: partial functionality
- >50% retention: architecture validated

Usage:
    python scripts/eval_nih_multisegment.py \
        --model_path ../models/Qwen--Qwen3-8b/ \
        --stage2_path outputs/slot_memory_8gpu_stage2_.../final \
        --stage3_path outputs/slot_memory_8gpu_stage3_.../final \
        --output_dir outputs/nih_multisegment/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import string
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

import torch
import torch.nn as nn

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nih_multisegment")


# ======================================================================
# Import slot memory modules
# ======================================================================

from src.memory.slot_memory import SlotMemoryCompressor, SlotMemoryModel
from transformers import AutoModelForCausalLM, AutoTokenizer


# ======================================================================
# Haystack text pool (from eval_nih_extended.py)
# ======================================================================

HAYSTACK_PASSAGES = [
    "The history of computing stretches back thousands of years. From the abacus to modern supercomputers, "
    "humanity has continually sought better ways to process information. Early mechanical calculators gave way "
    "to electronic computers in the mid-twentieth century, transforming every aspect of science and industry. "
    "The development of integrated circuits in the 1960s enabled the miniaturization of computers, leading to "
    "personal computers in the 1970s and 1980s.",

    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. "
    "These systems improve their performance over time without being explicitly programmed. Common approaches include "
    "supervised learning, unsupervised learning, and reinforcement learning, each suited to different problem types. "
    "Deep learning, a subset of machine learning using neural networks with many layers, has achieved remarkable "
    "results in image recognition, natural language processing, and game playing.",

    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space and time. "
    "Special relativity showed that the speed of light is constant for all observers, while general relativity "
    "described gravity as the curvature of spacetime caused by mass and energy. These theories have been confirmed "
    "by numerous experiments and are essential to modern physics, including GPS satellite calculations.",

    "Natural language processing enables computers to understand and generate human language. Key tasks include "
    "translation, summarization, question answering, and sentiment analysis. Recent advances in transformer models "
    "have dramatically improved performance across all these benchmarks. Large language models trained on massive "
    "text corpora have demonstrated emergent abilities in reasoning and code generation.",

    "The water cycle describes the continuous movement of water on, above, and below the Earth's surface. "
    "Water evaporates from oceans and lakes, forms clouds through condensation, and returns to the surface as "
    "precipitation. This cycle is essential for sustaining life and shaping weather patterns. Approximately 97% "
    "of Earth's water is stored in the oceans, with only about 1% available as fresh liquid water.",

    "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform "
    "computations. Unlike classical bits, quantum bits can exist in multiple states simultaneously, potentially "
    "solving certain problems exponentially faster than classical computers. Key algorithms include Shor's "
    "algorithm for factoring and Grover's algorithm for search.",

    "The Roman Empire was one of the largest and most influential civilizations in history. At its height, "
    "it stretched from Britain to Mesopotamia, encompassing the entire Mediterranean basin. Roman law, engineering, "
    "architecture, and language continue to influence modern societies around the world. The fall of the Western "
    "Roman Empire in 476 AD marks the traditional beginning of the Middle Ages.",

    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Using chlorophyll, "
    "plants absorb carbon dioxide and water to produce glucose and oxygen. This process is fundamental to life on "
    "Earth, as it forms the basis of most food chains. Photosynthesis occurs primarily in the leaves of plants, "
    "within organelles called chloroplasts.",

    "Database management systems organize and store data efficiently. Relational databases use structured tables "
    "with defined relationships, while NoSQL databases offer flexible schemas for unstructured data. Both types "
    "are critical for modern applications that need reliable data storage and retrieval. SQL remains the most "
    "widely used query language for relational databases.",

    "The internet has transformed global communication and commerce. Starting as a military research project "
    "called ARPANET in the 1960s, it evolved through decades of innovation. Today, billions of people use it "
    "daily for communication, entertainment, education, and business, making it one of the most significant "
    "inventions in history. The World Wide Web was invented by Tim Berners-Lee in 1989.",

    "Climate change refers to long-term shifts in temperatures and weather patterns. Human activities, particularly "
    "the burning of fossil fuels, have been the main driver of climate change since the Industrial Revolution. "
    "The resulting increase in greenhouse gases traps heat in the atmosphere, leading to rising global temperatures, "
    "melting ice caps, and more extreme weather events.",

    "Neuroscience is the scientific study of the nervous system, including the brain, spinal cord, and peripheral "
    "nerves. Modern neuroscience combines biology, chemistry, physics, and psychology to understand how the brain "
    "processes information, controls behavior, and gives rise to consciousness. Brain imaging technologies like "
    "fMRI and EEG have revolutionized our ability to study the living brain.",

    "Organic chemistry is the study of the structure, properties, composition, and reactions of carbon-containing "
    "compounds. Carbon's ability to form four covalent bonds and chain with other carbon atoms makes it uniquely "
    "suited for creating the complex molecules essential for life. Organic chemistry is fundamental to pharmaceuticals, "
    "polymers, and biochemistry.",

    "The Silk Road was a network of trade routes connecting East Asia to the Mediterranean, facilitating the exchange "
    "of goods, ideas, and cultures for over 1500 years. Named after the lucrative silk trade, these routes also "
    "carried spices, precious metals, and technologies between civilizations. The Silk Road played a crucial role "
    "in the development of many of the world's great civilizations.",

    "Evolutionary biology studies the processes that produced the diversity of life on Earth. Charles Darwin's "
    "theory of natural selection, published in 1859, remains the cornerstone of modern evolutionary thought. "
    "Organisms with traits better suited to their environment tend to survive and reproduce more successfully, "
    "gradually leading to changes in populations over many generations.",
]


# ======================================================================
# Needle generation
# ======================================================================

def generate_random_code(rng: random.Random, length: int = 5) -> str:
    """Generate a random alphanumeric code."""
    chars = string.ascii_uppercase + string.digits
    return ''.join(rng.choices(chars, k=length))


def make_needle(code: str) -> str:
    """Create a needle statement."""
    return f"The secret code is {code}."


def make_needle_question(code: str) -> str:
    """Create the retrieval question for a needle."""
    return f"What is the secret code mentioned in the text above? Give only the code."


# ======================================================================
# Context building
# ======================================================================

def build_haystack_tokens(tokenizer, target_token_count: int, rng: random.Random) -> list[int]:
    """Build haystack tokens of approximately target_token_count by repeating passages."""
    parts = []
    passage_idx = 0
    tokens = []
    while len(tokens) < target_token_count:
        parts.append(HAYSTACK_PASSAGES[passage_idx % len(HAYSTACK_PASSAGES)])
        passage_idx += 1
        tokens = tokenizer.encode(" ".join(parts), add_special_tokens=False)
        if passage_idx > 500:
            break
    return tokens


def insert_needle_at_depth(
    haystack_tokens: list[int],
    needle_text: str,
    tokenizer,
    depth: float,  # 0.0 to 1.0
) -> list[int]:
    """Insert needle at a relative depth in the context."""
    needle_tokens = tokenizer.encode(" " + needle_text, add_special_tokens=False)
    position = int(len(haystack_tokens) * depth)
    position = max(0, min(position, len(haystack_tokens)))
    return haystack_tokens[:position] + needle_tokens + haystack_tokens[position:]


# ======================================================================
# Model loading
# ======================================================================

def load_base_model(model_path: str, device: str, dtype_str: str = "bfloat16"):
    """Load base Qwen model and tokenizer."""
    dtype = getattr(torch, dtype_str, torch.bfloat16)
    model_path = os.path.realpath(model_path)

    logger.info(f"Loading base model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )
    model.eval()

    logger.info(f"Base model loaded: {model.config.model_type}, hidden={model.config.hidden_size}, "
                f"layers={model.config.num_hidden_layers}")
    return model, tokenizer


def load_slot_memory_checkpoint(checkpoint_path: str, base_model: nn.Module, device: str,
                                 segment_length: int = 1024, max_segments: int = 4,
                                 num_slots: int = 16, slot_dim: int = 256):
    """Load slot memory checkpoint and wrap model.

    Args:
        checkpoint_path: Path to slot memory checkpoint directory
        base_model: Base Qwen model (PeftModel with LoRA)
        device: Device to load on
        segment_length: Segment length for processing
        max_segments: Maximum number of segments
        num_slots: Number of memory slots
        slot_dim: Slot dimension

    Returns:
        SlotMemoryModel wrapper
    """
    checkpoint_path = os.path.realpath(checkpoint_path)

    logger.info(f"Loading slot memory checkpoint from {checkpoint_path}")

    # Create compressor
    compressor = SlotMemoryCompressor(
        hidden_dim=base_model.config.hidden_size,
        num_slots=num_slots,
        slot_dim=slot_dim,
    )

    # Load checkpoint
    checkpoint_file = os.path.join(checkpoint_path, "model.safetensors")
    if not os.path.exists(checkpoint_file):
        # Try pytorch format
        checkpoint_file = os.path.join(checkpoint_path, "pytorch_model.bin")

    if os.path.exists(checkpoint_file):
        logger.info(f"Loading weights from {checkpoint_file}")
        if checkpoint_file.endswith('.safetensors'):
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_file, device=str(device))
        else:
            state_dict = torch.load(checkpoint_file, map_location=device, weights_only=False)

        # Filter for compressor weights
        compressor_state = {}
        prefix = "compressor."
        for k, v in state_dict.items():
            if k.startswith(prefix):
                compressor_state[k[len(prefix):]] = v

        if compressor_state:
            missing, unexpected = compressor.load_state_dict(compressor_state, strict=False)
            logger.info(f"Compressor loaded: missing={len(missing)}, unexpected={len(unexpected)}")
        else:
            logger.warning("No compressor weights found in checkpoint, using random init")

        # Also load into base_model if needed (for LoRA adapters)
        model_state = {k: v for k, v in state_dict.items() if not k.startswith(prefix)}
        if model_state:
            missing, unexpected = base_model.load_state_dict(model_state, strict=False)
            logger.info(f"Base model updated: missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        logger.warning(f"No checkpoint file found at {checkpoint_path}, using random initialization")

    # Wrap with slot memory model
    slot_model = SlotMemoryModel(
        model=base_model,
        compressor=compressor,
        segment_length=segment_length,
        max_segments=max_segments,
        bptt_depth=0,  # No BPTT for inference
    )
    slot_model.to(device)
    slot_model.eval()

    logger.info(f"Slot memory model ready: slots={num_slots}, slot_dim={slot_dim}, "
                f"segment_length={segment_length}")

    return slot_model


# ======================================================================
# Multi-segment generation
# ======================================================================

@torch.no_grad()
def generate_with_slots(
    model: SlotMemoryModel,
    input_ids: list[int],
    question_tokens: list[int],
    max_new_tokens: int = 50,
) -> str:
    """Generate answer using slot memory with multi-segment processing.

    Args:
        model: SlotMemoryModel
        input_ids: Document tokens
        question_tokens: Question tokens
        max_new_tokens: Max tokens to generate

    Returns:
        Generated text
    """
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Process document through segments
    with torch.no_grad():
        generated = model.generate(
            input_ids=input_tensor,
            question_segment=torch.tensor([question_tokens], dtype=torch.long, device=device),
            max_new_tokens=max_new_tokens,
        )

    tokenizer = model.model.get_input_embeddings().weight.device
    # Need to get tokenizer from somewhere - this is a limitation
    # We'll handle this at the call site
    return generated


# ======================================================================
# Single-segment baseline
# ======================================================================

@torch.no_grad()
def generate_single_segment(
    model: nn.Module,
    tokenizer,
    input_ids: list[int],
    max_new_tokens: int = 50,
) -> str:
    """Generate answer using single segment (baseline, no memory)."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    # Truncate if needed
    max_len = getattr(model.config, "max_position_embeddings", 32768)
    if input_tensor.shape[1] > max_len:
        input_tensor = input_tensor[:, -(max_len - max_new_tokens):]

    output = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )

    generated = output[0, input_tensor.shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


# ======================================================================
# Exact match check
# ======================================================================

def check_answer(generated: str, expected_code: str) -> bool:
    """Check if the expected code appears in generated text."""
    return expected_code.upper() in generated.upper()


# ======================================================================
# Evaluation configs
# ======================================================================

def build_multisegment_configs() -> list[dict]:
    """Build multi-segment test configurations.

    Context lengths: 2048, 4096, 6144
    Needle depths: 10%, 50%, 90%
    Segments: 2, 3, 4 (depending on context length)
    """
    configs = []

    context_lengths = [2048, 4096, 6144]
    depths = [0.10, 0.50, 0.90]

    for ctx_len in context_lengths:
        # Determine number of segments (2-4 segments)
        # 2048: 2 segments of 1024
        # 4096: 2 or 4 segments (test both)
        # 6144: 3 segments of 2048
        if ctx_len == 2048:
            segment_configs = [{"segment_length": 1024, "num_segments": 2}]
        elif ctx_len == 4096:
            segment_configs = [
                {"segment_length": 2048, "num_segments": 2},
                {"segment_length": 1024, "num_segments": 4},
            ]
        elif ctx_len == 6144:
            segment_configs = [{"segment_length": 2048, "num_segments": 3}]
        else:
            segment_configs = [{"segment_length": ctx_len, "num_segments": 1}]

        for seg_cfg in segment_configs:
            for depth in depths:
                configs.append({
                    "context_length": ctx_len,
                    "depth": depth,
                    "segment_length": seg_cfg["segment_length"],
                    "num_segments": seg_cfg["num_segments"],
                })

    return configs


# ======================================================================
# Main evaluation
# ======================================================================

def run_single_segment_baseline(
    model: nn.Module,
    tokenizer,
    configs: list[dict],
    num_trials: int,
    seed: int,
) -> Dict[str, Any]:
    """Run single-segment baseline evaluation (no memory)."""
    logger.info("\n" + "="*60)
    logger.info("Running single-segment baseline (no memory)")
    logger.info("="*60)

    results = []
    per_config_stats = {}
    total_correct = 0
    total_count = 0

    for ci, cfg in enumerate(configs):
        # For baseline, only test context_length and depth (ignore segments)
        ctx_len = cfg["context_length"]
        depth = cfg["depth"]
        cfg_key = f"single_seg_{ctx_len}_d{depth:.0%}"

        logger.info(f"\n[{ci+1}/{len(configs)}] {cfg_key}")

        correct = 0

        for trial in range(num_trials):
            rng = random.Random(seed + ci * 1000 + trial)

            # Generate needle
            code = generate_random_code(rng)
            needle_text = make_needle(code)

            # Build context
            haystack = build_haystack_tokens(tokenizer, ctx_len, rng)
            context_tokens = insert_needle_at_depth(haystack, needle_text, tokenizer, depth)

            # Build question
            prompt_text = make_needle_question(code)
            question_tokens = tokenizer.encode("\n\n" + prompt_text + "\nAnswer:", add_special_tokens=False)

            full_input = context_tokens + question_tokens

            try:
                generated = generate_single_segment(model, tokenizer, full_input, max_new_tokens=50)
                is_correct = check_answer(generated, code)

                if is_correct:
                    correct += 1
                    total_correct += 1

                status = "✅" if is_correct else "❌"
                logger.info(f"  Trial {trial+1}/{num_trials}: {status} Expected: {code} | Got: {generated!r}")

            except Exception as e:
                logger.error(f"  Trial {trial+1}/{num_trials}: ❌ Error: {e}")
                generated = f"ERROR: {e}"
                is_correct = False

            result = {
                "config": cfg_key,
                "context_length": ctx_len,
                "depth": depth,
                "trial": trial,
                "expected_code": code,
                "generated": generated,
                "correct": is_correct,
            }
            results.append(result)
            total_count += 1

            torch.cuda.empty_cache()

        accuracy = correct / num_trials
        per_config_stats[cfg_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": num_trials,
        }
        logger.info(f"  → {cfg_key}: {correct}/{num_trials} = {accuracy:.2%}")

    overall_acc = total_correct / max(total_count, 1)
    logger.info(f"\nSingle-segment baseline: {total_correct}/{total_count} = {overall_acc:.2%}")

    return {
        "results": results,
        "per_config": per_config_stats,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_count": total_count,
    }


def run_multisegment_eval(
    slot_model: SlotMemoryModel,
    tokenizer,
    configs: list[dict],
    num_trials: int,
    seed: int,
    checkpoint_name: str,
) -> Dict[str, Any]:
    """Run multi-segment evaluation with slot memory."""
    logger.info("\n" + "="*60)
    logger.info(f"Running multi-segment eval: {checkpoint_name}")
    logger.info("="*60)

    # Update model config for each config
    results = []
    per_config_stats = {}
    total_correct = 0
    total_count = 0

    for ci, cfg in enumerate(configs):
        ctx_len = cfg["context_length"]
        depth = cfg["depth"]
        seg_len = cfg["segment_length"]
        num_seg = cfg["num_segments"]
        cfg_key = f"{checkpoint_name}_ctx{ctx_len}_seg{seg_len}x{num_seg}_d{depth:.0%}"

        logger.info(f"\n[{ci+1}/{len(configs)}] {cfg_key}")

        # Update segment configuration
        slot_model.segment_length = seg_len
        slot_model.max_segments = num_seg

        correct = 0

        for trial in range(num_trials):
            rng = random.Random(seed + ci * 1000 + trial)

            # Generate needle
            code = generate_random_code(rng)
            needle_text = make_needle(code)

            # Build context
            haystack = build_haystack_tokens(tokenizer, ctx_len, rng)
            context_tokens = insert_needle_at_depth(haystack, needle_text, tokenizer, depth)

            # Build question
            prompt_text = make_needle_question(code)
            question_tokens = tokenizer.encode("\n\n" + prompt_text + "\nAnswer:", add_special_tokens=False)

            try:
                # Process through segments and generate
                device = next(slot_model.parameters()).device
                input_tensor = torch.tensor([context_tokens], dtype=torch.long, device=device)
                question_tensor = torch.tensor([question_tokens], dtype=torch.long, device=device)

                generated_tokens = slot_model.generate(
                    input_ids=input_tensor,
                    question_segment=question_tensor,
                    max_new_tokens=50,
                )

                generated = tokenizer.decode(generated_tokens[0], skip_special_tokens=True).strip()
                is_correct = check_answer(generated, code)

                if is_correct:
                    correct += 1
                    total_correct += 1

                status = "✅" if is_correct else "❌"
                logger.info(f"  Trial {trial+1}/{num_trials}: {status} Expected: {code} | Got: {generated!r}")

            except Exception as e:
                logger.error(f"  Trial {trial+1}/{num_trials}: ❌ Error: {e}")
                generated = f"ERROR: {e}"
                is_correct = False

            result = {
                "config": cfg_key,
                "context_length": ctx_len,
                "depth": depth,
                "segment_length": seg_len,
                "num_segments": num_seg,
                "trial": trial,
                "expected_code": code,
                "generated": generated,
                "correct": is_correct,
            }
            results.append(result)
            total_count += 1

            torch.cuda.empty_cache()

        accuracy = correct / num_trials
        per_config_stats[cfg_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": num_trials,
        }
        logger.info(f"  → {cfg_key}: {correct}/{num_trials} = {accuracy:.2%}")

    overall_acc = total_correct / max(total_count, 1)
    logger.info(f"\n{checkpoint_name} multi-segment: {total_correct}/{total_count} = {overall_acc:.2%}")

    return {
        "results": results,
        "per_config": per_config_stats,
        "overall_accuracy": overall_acc,
        "total_correct": total_correct,
        "total_count": total_count,
    }


def calculate_retention(single_seg_acc: float, multi_seg_acc: float) -> float:
    """Calculate retention percentage: (multi-seg acc) / (single-seg acc)."""
    if single_seg_acc == 0:
        return 0.0
    return (multi_seg_acc / single_seg_acc) * 100.0


def interpret_retention(retention: float) -> str:
    """Interpret retention percentage according to success criteria."""
    if retention < 10:
        return "⚠️ SLOT ATTENTION COLLAPSE"
    elif retention < 50:
        return "⚡ PARTIAL FUNCTIONALITY"
    else:
        return "✅ ARCHITECTURE VALIDATED"


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Multi-segment NIH-Extended Benchmark for Slot Memory")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base Qwen3-8B model")
    parser.add_argument("--stage2_path", type=str, default="",
                        help="Path to Stage 2 slot memory checkpoint (leave empty to skip)")
    parser.add_argument("--stage3_path", type=str, default="",
                        help="Path to Stage 3 slot memory checkpoint (leave empty to skip)")
    parser.add_argument("--output_dir", type=str, default="outputs/nih_multisegment/")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--num_trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_slots", type=int, default=16, help="Number of memory slots")
    parser.add_argument("--slot_dim", type=int, default=256, help="Slot dimension")
    parser.add_argument("--skip_baseline", action="store_true",
                        help="Skip single-segment baseline (use 98% from previous run)")
    args = parser.parse_args()

    # Validate that at least one checkpoint is provided
    if not args.stage2_path and not args.stage3_path:
        logger.error("Must provide at least one of --stage2_path or --stage3_path")
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load tokenizer first (needed for all evals)
    _, tokenizer = load_base_model(args.model_path, args.device, args.dtype)

    # Build configs
    configs = build_multisegment_configs()
    logger.info(f"Built {len(configs)} test configurations")

    # Run single-segment baseline
    baseline_results = None
    baseline_acc = 0.98  # Default: use 98% from previous NIH-Extended eval

    if not args.skip_baseline:
        # Load base model for baseline
        base_model, _ = load_base_model(args.model_path, args.device, args.dtype)
        baseline_results = run_single_segment_baseline(
            base_model, tokenizer, configs, args.num_trials, args.seed
        )
        baseline_acc = baseline_results["overall_accuracy"]
    else:
        logger.info(f"Skipping baseline, using {baseline_acc:.2%} from previous eval")

    all_results = {
        "timestamp": datetime.now().isoformat(),
        "model_path": args.model_path,
        "num_trials": args.num_trials,
        "seed": args.seed,
        "num_slots": args.num_slots,
        "slot_dim": args.slot_dim,
        "baseline": {
            "overall_accuracy": baseline_acc,
            "results": baseline_results["results"] if baseline_results else [],
            "per_config": baseline_results["per_config"] if baseline_results else {},
        } if baseline_results else {"overall_accuracy": baseline_acc, "skipped": True},
    }

    # Run Stage 2 eval
    stage2_results = None
    if args.stage2_path:
        # Load base model
        base_model, _ = load_base_model(args.model_path, args.device, args.dtype)
        # Load Stage 2 checkpoint
        slot_model = load_slot_memory_checkpoint(
            args.stage2_path, base_model, args.device,
            segment_length=1024, max_segments=4,
            num_slots=args.num_slots, slot_dim=args.slot_dim
        )
        stage2_results = run_multisegment_eval(
            slot_model, tokenizer, configs, args.num_trials, args.seed, "Stage2"
        )
        all_results["stage2"] = stage2_results

        # Calculate retention
        retention = calculate_retention(baseline_acc, stage2_results["overall_accuracy"])
        interpretation = interpret_retention(retention)
        all_results["stage2"]["retention_pct"] = retention
        all_results["stage2"]["interpretation"] = interpretation
        logger.info(f"\nStage 2 Retention: {retention:.1f}% - {interpretation}")

        # Clean up
        del slot_model
        torch.cuda.empty_cache()

    # Run Stage 3 eval
    stage3_results = None
    if args.stage3_path:
        # Load base model
        base_model, _ = load_base_model(args.model_path, args.device, args.dtype)
        # Load Stage 3 checkpoint
        slot_model = load_slot_memory_checkpoint(
            args.stage3_path, base_model, args.device,
            segment_length=1024, max_segments=4,
            num_slots=args.num_slots, slot_dim=args.slot_dim
        )
        stage3_results = run_multisegment_eval(
            slot_model, tokenizer, configs, args.num_trials, args.seed, "Stage3"
        )
        all_results["stage3"] = stage3_results

        # Calculate retention
        retention = calculate_retention(baseline_acc, stage3_results["overall_accuracy"])
        interpretation = interpret_retention(retention)
        all_results["stage3"]["retention_pct"] = retention
        all_results["stage3"]["interpretation"] = interpretation
        logger.info(f"\nStage 3 Retention: {retention:.1f}% - {interpretation}")

        # Clean up
        del slot_model
        torch.cuda.empty_cache()

    # Compare Stage 2 vs Stage 3
    if stage2_results and stage3_results:
        stage2_ret = calculate_retention(baseline_acc, stage2_results["overall_accuracy"])
        stage3_ret = calculate_retention(baseline_acc, stage3_results["overall_accuracy"])

        logger.info("\n" + "="*60)
        logger.info("STAGE COMPARISON")
        logger.info("="*60)
        logger.info(f"Stage 2 retention: {stage2_ret:.1f}% - {interpret_retention(stage2_ret)}")
        logger.info(f"Stage 3 retention: {stage3_ret:.1f}% - {interpret_retention(stage3_ret)}")

        if stage3_ret > stage2_ret:
            diff = stage3_ret - stage2_ret
            logger.info(f"Stage 3 outperforms Stage 2 by {diff:.1f}%")
        elif stage2_ret > stage3_ret:
            diff = stage2_ret - stage3_ret
            logger.info(f"Stage 2 outperforms Stage 3 by {diff:.1f}%")
        else:
            logger.info("Stage 2 and Stage 3 perform equally")

        all_results["comparison"] = {
            "stage2_retention_pct": stage2_ret,
            "stage3_retention_pct": stage3_ret,
            "stage2_interpretation": interpret_retention(stage2_ret),
            "stage3_interpretation": interpret_retention(stage3_ret),
        }

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"nih_multisegment_{timestamp}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info("="*60)
    logger.info(f"Baseline (single-seg): {baseline_acc:.2%}")
    if stage2_results:
        logger.info(f"Stage 2 multi-seg: {stage2_results['overall_accuracy']:.2%} "
                   f"(retention: {all_results['stage2']['retention_pct']:.1f}%)")
    if stage3_results:
        logger.info(f"Stage 3 multi-seg: {stage3_results['overall_accuracy']:.2%} "
                   f"(retention: {all_results['stage3']['retention_pct']:.1f}%)")

    # Critical finding check
    if stage2_results and stage3_results:
        stage2_ret = all_results["stage2"]["retention_pct"]
        stage3_ret = all_results["stage3"]["retention_pct"]
        if stage2_ret < 10 and stage3_ret < 10:
            logger.warning("\n⚠️ CRITICAL FINDING: Slot attention collapse detected (<10% retention)")
            logger.warning("This matches RMT V7-V10 failure pattern.")
        elif stage2_ret > 50 or stage3_ret > 50:
            logger.info("\n✅ Architecture validated (>50% retention)")
        else:
            logger.warning("\n⚡ Partial functionality (10-50% retention)")
            logger.warning("Consider investigating slot attention implementation or training quality")

    return all_results


if __name__ == "__main__":
    main()
