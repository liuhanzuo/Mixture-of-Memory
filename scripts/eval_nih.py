#!/usr/bin/env python3
"""
Needle-in-Haystack (NiH) evaluation for RMT v5 memory compression.

Tests whether the model can recall a specific fact (the "needle")
embedded at various distances from the end of a long context (the "haystack"),
after the context has been compressed through RMT memory segments.

Usage:
    python scripts/eval_nih.py --checkpoint_dir outputs/rmt_v5_8gpu_XXX/final
    python scripts/eval_nih.py --checkpoint_dir outputs/rmt_v5_8gpu_XXX/final --distances 2048,4096,8192 --num_samples 20
"""

import argparse
import glob
import json
import logging
import os
import random
import sys
from pathlib import Path

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from src.memory.rmt.rmt_module import RMTMemory, RMTModel, build_rmt_attention_mask, build_rmt_position_ids

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger("eval_nih")


# ======================================================================
# Needle-in-Haystack data generation
# ======================================================================

NEEDLE_TEMPLATES = [
    "The secret code for project {name} is {code}.",
    "The access PIN for building {name} is {code}.",
    "The magic number associated with {name} is {code}.",
    "The verification ID for {name} is {code}.",
    "The serial number of {name} is {code}.",
]

QUESTION_TEMPLATES = [
    "What is the secret code for project {name}?",
    "What is the access PIN for building {name}?",
    "What is the magic number associated with {name}?",
    "What is the verification ID for {name}?",
    "What is the serial number of {name}?",
]

HAYSTACK_PASSAGES = [
    "The history of computing stretches back thousands of years. From the abacus to modern supercomputers, "
    "humanity has continually sought better ways to process information. Early mechanical calculators gave way "
    "to electronic computers in the mid-twentieth century, transforming every aspect of science and industry.",
    "Machine learning is a subset of artificial intelligence that focuses on building systems that learn from data. "
    "These systems improve their performance over time without being explicitly programmed. Common approaches include "
    "supervised learning, unsupervised learning, and reinforcement learning, each suited to different problem types.",
    "The theory of relativity, developed by Albert Einstein, revolutionized our understanding of space and time. "
    "Special relativity showed that the speed of light is constant for all observers, while general relativity "
    "described gravity as the curvature of spacetime caused by mass and energy.",
    "Natural language processing enables computers to understand and generate human language. Key tasks include "
    "translation, summarization, question answering, and sentiment analysis. Recent advances in transformer models "
    "have dramatically improved performance across all these benchmarks.",
    "The water cycle describes the continuous movement of water on, above, and below the Earth's surface. "
    "Water evaporates from oceans and lakes, forms clouds through condensation, and returns to the surface as "
    "precipitation. This cycle is essential for sustaining life and shaping weather patterns.",
    "Quantum computing leverages quantum mechanical phenomena like superposition and entanglement to perform "
    "computations. Unlike classical bits, quantum bits (qubits) can exist in multiple states simultaneously, "
    "potentially solving certain problems exponentially faster than classical computers.",
    "The Roman Empire was one of the largest and most influential civilizations in history. At its height, "
    "it stretched from Britain to Mesopotamia. Roman law, engineering, architecture, and language continue to "
    "influence modern societies around the world.",
    "Photosynthesis is the process by which green plants convert sunlight into chemical energy. Using chlorophyll, "
    "plants absorb carbon dioxide and water to produce glucose and oxygen. This process is fundamental to life on "
    "Earth, as it forms the basis of most food chains.",
    "Database management systems organize and store data efficiently. Relational databases use structured tables "
    "with defined relationships, while NoSQL databases offer flexible schemas for unstructured data. Both types "
    "are critical for modern applications that need reliable data storage and retrieval.",
    "The internet has transformed global communication and commerce. Starting as a military research project, "
    "it evolved through decades of innovation. Today, billions of people use it daily for communication, "
    "entertainment, education, and business, making it one of the most significant inventions in history.",
]


def generate_needle_fact(rng: random.Random) -> tuple[str, str, str]:
    """Generate a (fact_statement, question, expected_answer) tuple."""
    idx = rng.randint(0, len(NEEDLE_TEMPLATES) - 1)
    name = f"{''.join(rng.choices('abcdefghijklmnopqrstuvwxyz', k=6))}"
    code = str(rng.randint(100000, 999999))
    fact = NEEDLE_TEMPLATES[idx].format(name=name, code=code)
    question = QUESTION_TEMPLATES[idx].format(name=name)
    return fact, question, code


def build_haystack_context(
    tokenizer,
    target_distance: int,
    needle_fact: str,
    question: str,
    rng: random.Random,
) -> tuple[list[int], str]:
    """
    Build a haystack context with a needle inserted at ~`target_distance` tokens
    from the end.

    Returns (full_token_ids, expected_answer).
    The context ends with the retrieval question.
    """
    # Build haystack by repeating passages until we have enough tokens
    haystack_parts = []
    passage_idx = 0
    while True:
        haystack_parts.append(HAYSTACK_PASSAGES[passage_idx % len(HAYSTACK_PASSAGES)])
        passage_idx += 1
        candidate = " ".join(haystack_parts)
        tokens = tokenizer.encode(candidate, add_special_tokens=False)
        if len(tokens) >= target_distance + 512:
            break
        if passage_idx > 200:
            break

    # Tokenize needle and question
    needle_tokens = tokenizer.encode(" " + needle_fact, add_special_tokens=False)
    # suffix after the needle
    suffix_len_needed = max(target_distance // 3, 64)
    total_context_tokens = target_distance + len(needle_tokens) + suffix_len_needed

    if len(tokens) < total_context_tokens:
        while len(tokens) < total_context_tokens + 256:
            haystack_parts.append(HAYSTACK_PASSAGES[passage_idx % len(HAYSTACK_PASSAGES)])
            passage_idx += 1
            tokens = tokenizer.encode(" ".join(haystack_parts), add_special_tokens=False)

    # Insert needle at target_distance from END
    needle_position = len(tokens) - target_distance
    needle_position = max(0, min(needle_position, len(tokens) - len(needle_tokens)))

    full_tokens = tokens[:needle_position] + needle_tokens + tokens[needle_position:]
    full_tokens = full_tokens[:len(tokens) + len(needle_tokens)]

    return full_tokens, question


# ======================================================================
# Model loading — same pattern as eval_rmt.py
# ======================================================================

def find_checkpoint(base_glob: str = "outputs/rmt_v5_8gpu_*/final/") -> str | None:
    """Find the latest RMT v5 checkpoint directory."""
    candidates = sorted(glob.glob(base_glob))
    if not candidates:
        return None
    return candidates[-1]


def load_rmt_config(checkpoint_dir: str) -> dict:
    """Load RMT config from checkpoint_dir (or its parent)."""
    # Try checkpoint_dir first, then parent
    for p in [checkpoint_dir, str(Path(checkpoint_dir).parent)]:
        config_path = os.path.join(p, "rmt_config.json")
        if os.path.exists(config_path):
            with open(config_path) as f:
                return json.load(f)
    # Defaults
    return {
        "base_model": "../models/Qwen--Qwen3-8b",
        "num_memory_tokens": 16,
        "segment_length": 1024,
        "max_segments": 6,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "bottleneck_dim": 64,
        "extractor_version": 5,
    }


def load_rmt_model(checkpoint_dir: str, config: dict, device: torch.device):
    """Load model + RMT memory — same approach as eval_rmt.py."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load tokenizer
    logger.info(f"Loading tokenizer from {checkpoint_dir}")
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Load model directly from checkpoint (LoRA adapter is already merged/saved there)
    logger.info(f"Loading model from {checkpoint_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_dir,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="eager",
    )
    model.eval()

    hidden_dim = model.config.hidden_size

    # Initialize RMT memory
    rmt_memory = RMTMemory(
        hidden_dim=hidden_dim,
        num_memory_tokens=config.get("num_memory_tokens", 16),
        num_heads=config.get("num_memory_heads", 8),
        max_segments=config.get("max_segments", 6) + 1,
        bottleneck_dim=config.get("bottleneck_dim", 64),
        extractor_version=config.get("extractor_version", 5),
    ).to(device=device, dtype=torch.bfloat16)

    # Load RMT weights
    rmt_path = os.path.join(checkpoint_dir, "rmt_memory.pt")
    logger.info(f"Loading RMT memory from {rmt_path}")
    rmt_memory.load_state_dict(torch.load(rmt_path, map_location=device))
    rmt_memory.eval()

    # Wrap in RMTModel (same as eval_rmt.py)
    rmt_model = RMTModel(model, rmt_memory, segment_length=config.get("segment_length", 1024))
    rmt_model = rmt_model.to(device=device, dtype=torch.bfloat16)
    rmt_model.eval()

    return rmt_model, tokenizer


# ======================================================================
# RMT inference — uses RMTModel wrapper
# ======================================================================

@torch.no_grad()
def rmt_inference(
    rmt_model: RMTModel,
    context_ids: list[int],
    question_ids: list[int],
    tokenizer,
    max_new_tokens: int = 32,
) -> tuple[str, dict]:
    """
    RMT inference:
    1. Process context through RMT segments to build compressed memory
    2. Forward question tokens WITH the final memory prefix
    3. Greedy-generate answer tokens

    Uses RMTModel._forward_single_segment (same as training forward pass).
    """
    device = next(rmt_model.parameters()).device
    segment_length = rmt_model.segment_length
    num_memory_tokens = rmt_model.num_memory_tokens
    B = 1

    # Pad context to multiple of segment_length for clean segment processing
    total_ctx = len(context_ids)
    num_segments = (total_ctx + segment_length - 1) // segment_length
    padded_len = num_segments * segment_length
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    if total_ctx < padded_len:
        context_ids = context_ids + [pad_id] * (padded_len - total_ctx)

    context_tensor = torch.tensor([context_ids], device=device, dtype=torch.long)

    # Process all context segments
    old_memory = None
    for seg_idx in range(num_segments):
        start = seg_idx * segment_length
        end = start + segment_length
        seg_ids = context_tensor[:, start:end]

        if old_memory is None:
            mem = rmt_model.rmt.get_initial_memory(seg_idx, B, device, torch.bfloat16)
        else:
            mem = old_memory

        _, seg_hidden = rmt_model._forward_single_segment(seg_ids, None, mem, seg_idx)

        # Handle v5 tuple returns: (new_memory, recon_loss)
        mem_result = rmt_model.rmt.extract_memory(seg_hidden, old_memory)
        old_memory = mem_result[0] if isinstance(mem_result, tuple) else mem_result

    # Process question with final memory
    question_tensor = torch.tensor([question_ids], device=device, dtype=torch.long)
    q_len = question_tensor.shape[1]

    inputs_embeds = rmt_model._embed_with_memory(question_tensor, old_memory)
    q_len = question_tensor.shape[1]
    total_len = rmt_model.num_memory_tokens + q_len

    # Build attention mask for memory + question tokens
    gen_attn_mask = build_rmt_attention_mask(q_len, rmt_model.num_memory_tokens, device)
    gen_attn_mask_4d = gen_attn_mask.unsqueeze(0).unsqueeze(0)  # [1, 1, total, total]
    gen_attn_mask_4d = torch.zeros_like(gen_attn_mask_4d, dtype=torch.float32).masked_fill(~gen_attn_mask_4d, float('-inf'))

    # Build position ids
    gen_position_ids = build_rmt_position_ids(q_len, rmt_model.num_memory_tokens, 0, device).unsqueeze(0)

    # Simple greedy generation (same approach as eval_rmt.py):
    # Re-feed full sequence each step. This avoids RoPE position and
    # KV-cache mask mismatches that were causing degenerate outputs.
    generated_ids = []
    eos_id = tokenizer.eos_token_id
    lm_head = rmt_model.model.lm_head

    for step in range(max_new_tokens):
        cur_len = inputs_embeds.shape[1]
        # Extend attention mask for newly generated tokens (causal)
        if step == 0:
            cur_mask = gen_attn_mask_4d.expand(1, 1, -1, -1)
        else:
            # Rebuild mask for current length: memory tokens attend to all, text tokens are causal
            cur_mask_bool = build_rmt_attention_mask(cur_len - rmt_model.num_memory_tokens, rmt_model.num_memory_tokens, device)
            cur_mask = torch.zeros(1, 1, cur_len, cur_len, device=device, dtype=torch.float32)
            cur_mask = cur_mask.masked_fill(~cur_mask_bool.unsqueeze(0).unsqueeze(0), float('-inf'))
        # Extend position ids
        if step == 0:
            cur_pos = gen_position_ids
        else:
            cur_pos = build_rmt_position_ids(cur_len - rmt_model.num_memory_tokens, rmt_model.num_memory_tokens, 0, device).unsqueeze(0)

        outputs = rmt_model.model.model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": cur_mask},
            position_ids=cur_pos,
            output_hidden_states=False,
        )
        logits = lm_head(outputs.last_hidden_state[:, -1:, :])
        next_token = logits.argmax(dim=-1)  # [1, 1]
        tid = next_token.item()
        generated_ids.append(tid)

        if tid == eos_id:
            break

        next_embed = rmt_model.model.get_input_embeddings()(next_token)
        inputs_embeds = torch.cat([inputs_embeds, next_embed], dim=1)

    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    debug_info = {
        "num_segments": num_segments,
        "context_length": total_ctx,
        "question_length": q_len,
        "generated_ids": generated_ids,
    }
    return generated_text, debug_info


@torch.no_grad()
def baseline_inference(
    model,
    context_ids: list[int],
    question_ids: list[int],
    tokenizer,
    max_new_tokens: int = 32,
) -> tuple[str, dict]:
    """Baseline: no RMT, feed everything through model.generate."""
    device = next(model.parameters()).device

    max_model_len = getattr(model.config, "max_position_embeddings", 32768)
    total_len = len(context_ids) + len(question_ids)
    if total_len > max_model_len:
        keep = max_model_len - len(question_ids)
        context_ids = context_ids[-keep:]

    full_ids = context_ids + question_ids
    input_tensor = torch.tensor([full_ids], device=device)

    output_ids = model.generate(
        input_tensor,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        top_p=1.0,
        use_cache=True,
    )

    generated = output_ids[0, input_tensor.shape[1]:]
    generated_text = tokenizer.decode(generated, skip_special_tokens=True)

    return generated_text, {"context_length": len(context_ids), "full_length": len(full_ids)}


# ======================================================================
# Evaluation
# ======================================================================

def exact_match_score(generated: str, expected: str) -> bool:
    """Check if the expected answer appears in the generated text."""
    return expected in generated


def run_evaluation(
    checkpoint_dir: str,
    distances: list[int],
    num_samples: int,
    output_dir: str,
    seed: int = 42,
    base_model_only: bool = False,
):
    """Run the full NiH evaluation."""
    rng = random.Random(seed)

    config = load_rmt_config(checkpoint_dir)
    segment_length = config.get("segment_length", 1024)
    num_memory_tokens = config.get("num_memory_tokens", 16)
    max_segments = config.get("max_segments", 6)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if not base_model_only:
        rmt_model, tokenizer = load_rmt_model(checkpoint_dir, config, device)
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        base_model_path = config.get("base_model", "../models/Qwen--Qwen3-8b")
        base_model_path = os.path.realpath(os.path.join(_project_root, base_model_path))
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path, trust_remote_code=True,
            torch_dtype=torch.bfloat16, device_map={"": str(device)},
            attn_implementation="eager",
        )
        model.eval()
        rmt_model = None

    all_results = {
        "config": {
            "checkpoint_dir": checkpoint_dir,
            "distances": distances,
            "num_samples": num_samples,
            "segment_length": segment_length,
            "num_memory_tokens": num_memory_tokens,
            "max_segments": max_segments,
            "base_model_only": base_model_only,
            "seed": seed,
        },
        "per_sample": [],
        "per_distance": {},
        "overall": {},
    }

    total_correct = 0
    total_samples = 0

    for distance in distances:
        logger.info(f"\n{'='*60}")
        logger.info(f"Distance: {distance} tokens | Samples: {num_samples}")
        logger.info(f"{'='*60}")

        dist_correct = 0
        dist_samples = []

        for sample_idx in range(num_samples):
            needle_fact, question, expected_answer = generate_needle_fact(rng)

            context_tokens, q_text = build_haystack_context(
                tokenizer, distance, needle_fact, question, rng
            )
            question_tokens = tokenizer.encode(
                "\n\nBased on the text above, " + question + " Give only the number as your answer.\nAnswer:",
                add_special_tokens=False,
            )

            logger.info(
                f"  Sample {sample_idx+1}/{num_samples}: "
                f"context={len(context_tokens)} tokens, "
                f"needle_at~{distance} from end, "
                f"expected={expected_answer}"
            )

            try:
                if base_model_only:
                    gen_text, debug = baseline_inference(
                        model, context_tokens, question_tokens, tokenizer,
                        max_new_tokens=32,
                    )
                else:
                    max_context_tokens = segment_length * max_segments
                    if len(context_tokens) > max_context_tokens:
                        context_tokens = context_tokens[-max_context_tokens:]

                    gen_text, debug = rmt_inference(
                        rmt_model, context_tokens, question_tokens, tokenizer,
                        max_new_tokens=32,
                    )

                match = exact_match_score(gen_text, expected_answer)
                if match:
                    dist_correct += 1
                    total_correct += 1

                total_samples += 1

                sample_result = {
                    "distance": distance,
                    "sample_idx": sample_idx,
                    "expected": expected_answer,
                    "generated": gen_text.strip(),
                    "match": match,
                    "context_length": len(context_tokens),
                    "question_length": len(question_tokens),
                }
                dist_samples.append(sample_result)
                all_results["per_sample"].append(sample_result)

                status = "✅" if match else "❌"
                logger.info(f"    {status} Generated: {gen_text.strip()!r}")

            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                import traceback
                traceback.print_exc()
                sample_result = {
                    "distance": distance,
                    "sample_idx": sample_idx,
                    "expected": expected_answer,
                    "generated": f"ERROR: {e}",
                    "match": False,
                    "error": str(e),
                }
                dist_samples.append(sample_result)
                all_results["per_sample"].append(sample_result)
                total_samples += 1

            torch.cuda.empty_cache()

        accuracy = dist_correct / max(num_samples, 1)
        all_results["per_distance"][str(distance)] = {
            "accuracy": accuracy,
            "correct": dist_correct,
            "total": num_samples,
            "samples": dist_samples,
        }
        logger.info(f"\n  Distance {distance}: {dist_correct}/{num_samples} = {accuracy:.2%}")

    overall_accuracy = total_correct / max(total_samples, 1)
    all_results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_samples,
    }

    logger.info(f"\n{'='*60}")
    logger.info("RESULTS SUMMARY")
    mode_str = "BASE MODEL (no RMT)" if base_model_only else "RMT v5"
    logger.info(f"Mode: {mode_str}")
    logger.info(f"{'='*60}")
    for distance in distances:
        d = all_results["per_distance"][str(distance)]
        logger.info(f"  Distance {distance:>5}: {d['correct']:>3}/{d['total']:>3} = {d['accuracy']:.2%}")
    logger.info(f"  {'Overall':>12}: {total_correct:>3}/{total_samples:>3} = {overall_accuracy:.2%}")
    logger.info(f"{'='*60}")

    os.makedirs(output_dir, exist_ok=True)
    suffix = "_baseline" if base_model_only else "_rmt"
    results_path = os.path.join(output_dir, f"nih_results{suffix}.json")
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {results_path}")

    return all_results


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack evaluation for RMT v5")
    parser.add_argument(
        "--checkpoint_dir", type=str, default=None,
        help="Path to RMT v5 checkpoint (final/ dir). Auto-detects latest if not specified.",
    )
    parser.add_argument(
        "--distances", type=str, default="2048,4096,8192",
        help="Comma-separated token distances to test (default: 2048,4096,8192)",
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of samples per distance (default: 20)",
    )
    parser.add_argument(
        "--output_dir", type=str, default=None,
        help="Output directory for results. Default: <checkpoint_dir>/eval/",
    )
    parser.add_argument(
        "--baseline_only", action="store_true",
        help="Only run baseline (no RMT)",
    )
    parser.add_argument(
        "--rmt_only", action="store_true",
        help="Only run RMT (no baseline)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )

    args = parser.parse_args()

    distances = [int(d.strip()) for d in args.distances.split(",")]

    if args.checkpoint_dir is None:
        ckpt = find_checkpoint()
        if ckpt is None:
            logger.error("No checkpoint found. Specify --checkpoint_dir explicitly.")
            sys.exit(1)
        args.checkpoint_dir = ckpt
        logger.info(f"Auto-detected checkpoint: {ckpt}")

    if args.output_dir is None:
        args.output_dir = os.path.join(args.checkpoint_dir, "eval")

    if not args.rmt_only:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING BASELINE (no RMT)")
        logger.info("=" * 60)
        run_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            distances=distances,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            base_model_only=True,
        )

    if not args.baseline_only:
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING RMT v5")
        logger.info("=" * 60)
        run_evaluation(
            checkpoint_dir=args.checkpoint_dir,
            distances=distances,
            num_samples=args.num_samples,
            output_dir=args.output_dir,
            seed=args.seed,
            base_model_only=False,
        )


if __name__ == "__main__":
    main()
