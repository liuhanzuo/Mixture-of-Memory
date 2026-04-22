#!/usr/bin/env python3
"""
NIH-Extended Benchmark: Extended Needle-in-a-Haystack for evaluating
long-context memory compression effectiveness.

Difficulty dimensions:
  1. Long context: 4K / 8K / 16K / 32K tokens
  2. Multi-needle: 2 / 3 / 5 needles per document
  3. Deep needle: inserted at 10% / 25% / 50% / 75% / 90% depth
  4. Latency test: distractor question about context prefix, then needle question

Test configs (3 trials each):
  - Single-needle: (4K,8K,16K,32K) × (10%,50%,90% depth) = 12 × 3 = 36
  - Multi-needle:  (2,3,5 needles) × (4K,16K) × (50% depth) = 6 × 3 = 18
  - Latency:       (8K,16K) × 3 distractor questions = 6 × 3 = 18
  Total: ~72 tests

Usage:
    python scripts/eval_nih_extended.py \
        --model_path ../models/Qwen--Qwen3-8b/ \
        --output_dir outputs/nih_extended/
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

import torch

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("nih_extended")


# ======================================================================
# Haystack text pool
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


def insert_multiple_needles(
    haystack_tokens: list[int],
    needles: list[str],  # [(code, needle_text), ...]
    tokenizer,
    depth: float,
) -> list[int]:
    """Insert multiple needles spread around the target depth."""
    n = len(needles)
    # Spread needles evenly around the depth position
    # e.g., for 3 needles at depth 0.5: place at 0.3, 0.5, 0.7
    spread = 0.15 * (n - 1)  # total spread range
    depths = []
    for i in range(n):
        if n == 1:
            d = depth
        else:
            d = depth - spread / 2 + spread * i / (n - 1)
            d = max(0.05, min(0.95, d))
        depths.append(d)

    # Sort by position descending so insertions don't shift each other
    items = sorted(zip(depths, needles), key=lambda x: x[0], reverse=True)

    result = list(haystack_tokens)
    for d, (code, needle_text) in items:
        result = insert_needle_at_depth(result, needle_text, tokenizer, d)

    return result


# ======================================================================
# Latency test helpers
# ======================================================================

LATENCY_DISTRACTOR_TEMPLATES = [
    "What is the main topic discussed in the first paragraph of the text above? Answer briefly in one sentence.",
    "Summarize the key argument made near the beginning of the text above in one short sentence.",
    "According to the text above, what historical period or scientific concept is mentioned first? Give a brief answer.",
]


def make_latency_prompt(distractor_idx: int, needle_code: str) -> str:
    """Create a two-question prompt for latency test."""
    distractor_q = LATENCY_DISTRACTOR_TEMPLATES[distractor_idx % len(LATENCY_DISTRACTOR_TEMPLATES)]
    needle_q = make_needle_question(needle_code)
    return f"{distractor_q}\n\n{needle_q}\nAnswer:"


# ======================================================================
# Model loading
# ======================================================================

def load_model(model_path: str, device: str, dtype_str: str = "bfloat16", max_seq_len: int = 0):
    """Load model and tokenizer.
    If max_seq_len > 0, extend RoPE scaling to support that sequence length.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = getattr(torch, dtype_str, torch.bfloat16)
    model_path = os.path.realpath(model_path)

    logger.info(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = dict(
        trust_remote_code=True,
        torch_dtype=dtype,
        device_map={"": device},
    )

    if max_seq_len > 0:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        native_max = getattr(cfg, 'max_position_embeddings', 40960)
        if max_seq_len > native_max:
            factor = max_seq_len / native_max
            logger.info(f"Extending context: {native_max} -> {max_seq_len} (factor={factor:.2f})")
            cfg.max_position_embeddings = max_seq_len
            # Qwen3 uses rope_parameters dict: must include rope_theta for linear scaling
            native_theta = getattr(cfg, 'rope_theta', None)
            if native_theta is None:
                # rope_theta is stored inside rope_parameters dict
                rp = getattr(cfg, 'rope_parameters', {}) or {}
                native_theta = rp.get('rope_theta', 1000000.0)
            cfg.rope_scaling = {
                "rope_type": "linear",
                "factor": factor,
                "rope_theta": native_theta,
            }
            kwargs['config'] = cfg

    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    model.eval()

    logger.info(f"Model loaded: {model.config.model_type}, hidden={model.config.hidden_size}, "
                f"layers={model.config.num_hidden_layers}")
    return model, tokenizer


# ======================================================================
# Inference
# ======================================================================

@torch.no_grad()
def generate_answer(model, tokenizer, input_ids: list[int], max_new_tokens: int = 50) -> str:
    """Greedy decode answer."""
    device = next(model.parameters()).device
    input_tensor = torch.tensor([input_ids], device=device, dtype=torch.long)

    # Truncate to model's max length if needed
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
# Exact match
# ======================================================================

def check_answer(generated: str, expected_code: str) -> bool:
    """Check if the expected code appears in generated text."""
    return expected_code.upper() in generated.upper()


# ======================================================================
# Test configurations
# ======================================================================

def build_configs(context_filter: str = "") -> list[dict]:
    """Build all test configurations.

    context_filter: if non-empty, only include single-needle configs whose
                    context_length matches one of the comma-separated values.
                    e.g. "32768" or "40960,49152,65536"
    """
    configs = []

    # Extended context lengths for degradation curve
    # Native max: 40960. Beyond that requires RoPE scaling.
    extended_lengths = [4096, 8192, 16384, 32768, 40960, 49152, 53248, 57344, 61440, 65536, 98304, 131072]

    if context_filter:
        allowed = set(int(x.strip()) for x in context_filter.split(",") if x.strip())
        extended_lengths = [L for L in extended_lengths if L in allowed]

    # Single-needle: extended_lengths × (10%,50%,90% depth)
    for ctx_len in extended_lengths:
        for depth in [0.10, 0.50, 0.90]:
            configs.append({
                "type": "single",
                "context_length": ctx_len,
                "depth": depth,
                "needle_count": 1,
            })

    # Multi-needle: (2,3,5 needles) × (4K,16K) × (50% depth)
    for needle_count in [2, 3, 5]:
        for ctx_len in [4096, 16384]:
            configs.append({
                "type": "multi",
                "context_length": ctx_len,
                "depth": 0.50,
                "needle_count": needle_count,
            })

    # Latency: (8K,16K) × 3 distractor questions
    for ctx_len in [8192, 16384]:
        for q_idx in range(3):
            configs.append({
                "type": "latency",
                "context_length": ctx_len,
                "depth": 0.75,
                "needle_count": 1,
                "distractor_idx": q_idx,
            })

    return configs


# ======================================================================
# Main evaluation
# ======================================================================

def run_eval(args):
    device = args.device
    model, tokenizer = load_model(args.model_path, device, args.dtype,
                                       max_seq_len=args.max_seq_len)
    configs = build_configs(context_filter=args.context_filter)
    num_trials = args.num_trials

    # Filter configs if --smoke or --config_filter
    if args.smoke:
        # Smoke test: just 2 configs, 1 trial each
        configs = configs[:2]
        num_trials = 1
        logger.info("★ SMOKE MODE: running 2 configs × 1 trial")

    if args.config_filter:
        filtered = []
        for c in configs:
            s = f"{c['type']}_{c['context_length']}_{c['depth']}_{c.get('needle_count',1)}"
            if args.config_filter in s:
                filtered.append(c)
        configs = filtered if filtered else configs
        logger.info(f"Filtered to {len(configs)} configs matching '{args.config_filter}'")

    all_results = []
    per_config_stats = {}
    total_correct = 0
    total_count = 0
    start_time = time.time()

    for ci, cfg in enumerate(configs):
        cfg_key = f"{cfg['type']}_{cfg['context_length']}_d{cfg['depth']:.0%}_n{cfg['needle_count']}"
        logger.info(f"\n{'='*60}")
        logger.info(f"[{ci+1}/{len(configs)}] {cfg_key}")
        logger.info(f"{'='*60}")

        correct = 0

        for trial in range(num_trials):
            rng = random.Random(args.seed + ci * 1000 + trial)

            # Generate needles
            needles = []
            for _ in range(cfg["needle_count"]):
                code = generate_random_code(rng)
                needles.append((code, make_needle(code)))

            # Build context
            ctx_len = cfg["context_length"]
            haystack = build_haystack_tokens(tokenizer, ctx_len, rng)

            if cfg["type"] == "multi":
                context_tokens = insert_multiple_needles(haystack, needles, tokenizer, cfg["depth"])
            else:
                context_tokens = insert_needle_at_depth(
                    haystack, needles[0][1], tokenizer, cfg["depth"]
                )

            # Build question
            if cfg["type"] == "latency":
                prompt_text = make_latency_prompt(cfg.get("distractor_idx", 0), needles[0][0])
                question_tokens = tokenizer.encode("\n\n" + prompt_text, add_special_tokens=False)
            else:
                if cfg["needle_count"] == 1:
                    prompt_text = make_needle_question(needles[0][0])
                else:
                    # Ask for all codes
                    codes_str = ", ".join(n[0] for n in needles)
                    prompt_text = (
                        f"The text above contains {cfg['needle_count']} different secret codes. "
                        f"List all of them separated by commas."
                    )
                question_tokens = tokenizer.encode(
                    "\n\n" + prompt_text + "\nAnswer:",
                    add_special_tokens=False,
                )

            full_input = context_tokens + question_tokens
            logger.info(f"  Trial {trial+1}/{num_trials}: "
                        f"ctx={len(context_tokens)} tokens, "
                        f"q={len(question_tokens)} tokens, "
                        f"total={len(full_input)} tokens")

            try:
                generated = generate_answer(model, tokenizer, full_input, args.max_new_tokens)

                if cfg["needle_count"] == 1:
                    is_correct = check_answer(generated, needles[0][0])
                else:
                    is_correct = all(check_answer(generated, code) for code, _ in needles)

                if is_correct:
                    correct += 1

                status = "✅" if is_correct else "❌"
                logger.info(f"    {status} Expected: {[c for c, _ in needles]} | Got: {generated!r}")

            except Exception as e:
                logger.error(f"    ❌ Error: {e}")
                generated = f"ERROR: {e}"
                is_correct = False

            result = {
                "config": cfg_key,
                "type": cfg["type"],
                "depth": cfg["depth"],
                "context_length": cfg["context_length"],
                "needle_count": cfg["needle_count"],
                "trial": trial,
                "expected_codes": [c for c, _ in needles],
                "generated": generated,
                "correct": is_correct,
            }
            all_results.append(result)
            total_count += 1
            if is_correct:
                total_correct += 1

            torch.cuda.empty_cache()

        accuracy = correct / num_trials
        per_config_stats[cfg_key] = {
            "accuracy": accuracy,
            "correct": correct,
            "total": num_trials,
        }
        logger.info(f"  → {cfg_key}: {correct}/{num_trials} = {accuracy:.2%}")

    elapsed = time.time() - start_time

    # Summary
    overall_acc = total_correct / max(total_count, 1)
    logger.info(f"\n{'='*60}")
    logger.info("SUMMARY")
    logger.info(f"{'='*60}")
    logger.info(f"Overall: {total_correct}/{total_count} = {overall_acc:.2%}")
    logger.info(f"Elapsed: {elapsed:.1f}s")
    logger.info(f"\nPer-config breakdown:")

    for key, stats in per_config_stats.items():
        logger.info(f"  {key}: {stats['correct']}/{stats['total']} = {stats['accuracy']:.2%}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"nih_extended_{timestamp}.json")
    output = {
        "summary": {
            "overall_accuracy": overall_acc,
            "total_correct": total_correct,
            "total_count": total_count,
            "elapsed_seconds": round(elapsed, 1),
            "model_path": args.model_path,
            "num_trials": num_trials,
            "seed": args.seed,
            "timestamp": timestamp,
        },
        "per_config": per_config_stats,
        "per_sample": all_results,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"\nResults saved to {output_path}")

    return output


# ======================================================================
# Args
# ======================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="NIH-Extended Benchmark")
    parser.add_argument("--model_path", type=str, required=True, help="Model path (e.g., Qwen3-8B)")
    parser.add_argument("--output_dir", type=str, default="outputs/nih_extended/", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"])
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--num_trials", type=int, default=3, help="Trials per config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test (2 configs × 1 trial)")
    parser.add_argument("--config_filter", type=str, default="", help="Substring filter for configs")
    parser.add_argument("--context_filter", type=str, default="",
                        help="Comma-separated context lengths to include (e.g. 32768,40960,49152,65536)")
    parser.add_argument("--max_seq_len", type=int, default=0,
                        help="If > 0, extend model context via RoPE scaling to this length")
    return parser.parse_args()


if __name__ == "__main__":
    run_eval(parse_args())
