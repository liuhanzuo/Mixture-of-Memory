#!/usr/bin/env python3
"""
RULER CWE (Common Words Extraction) baseline evaluation for Qwen3-8B base model.

Self-contained implementation following NVIDIA RULER benchmark methodology:
  https://arxiv.org/abs/2404.06654

CWE task: Given a long numbered list of words where some appear more frequently
than others, the model must identify the N most common words.

Usage:
    python scripts/eval_ruler_baseline.py \
        --model_path models/Qwen--Qwen3-8b \
        --context_length 65536 \
        --num_samples 50 \
        --output results/ruler_cwe_64k.json
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import torch
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── RULER CWE template (base model, no chat template) ──────────────────────────

CWE_TEMPLATE = """\
Some special magic words are hidden within the following text. Make sure to memorize them. I will quiz you about the words afterwards.

{context}
What are the {num_cw} most common words in the above list?"""

CWE_ANSWER_PREFIX = " The special magic words mentioned in the provided text are"

CWE_ONE_SHOT_TEMPLATE = """\
Below is a numbered list of words. In these words, some appear more often than others. Memorize the ones that appear most often.
{example_context}
Question: What are the {num_cw} most common words in the above list? Answer: The top {num_cw} words that appear most often in the list are: {example_answer}

{main_prompt}"""

TOKENS_TO_GENERATE = 150  # enough for 10 words + separators


# ── Word list generation ───────────────────────────────────────────────────────

def load_word_list(seed=42):
    """Load a diverse word list. Uses wonderwords if available, else random English."""
    try:
        from wonderwords import RandomWord
        rw = RandomWord()
        words = []
        for _ in range(5000):
            w = rw.word()
            if w and w.isalpha():
                words.append(w.lower())
        words = list(set(words))
    except ImportError:
        logger.warning("wonderwords not installed, using fallback word list")
        words = [
            "apple", "brave", "cloud", "dream", "eagle", "flame", "grace", "heart",
            "ivory", "jewel", "karma", "light", "magic", "noble", "ocean", "peace",
            "quest", "river", "stone", "tower", "unity", "vivid", "water", "xenon",
            "yield", "zebra", "amber", "blaze", "coral", "drift", "ember", "frost",
            "globe", "haven", "index", "joker", "kneel", "lunar", "medal", "nerve",
            "orbit", "piano", "quiet", "radar", "solar", "tiger", "ultra", "vigor",
            "whale", "youth", "zonal", "alpha", "brisk", "cedar", "delta", "epoch",
            "flare", "ghost", "hymnal", "irony", "jazzy", "knack", "lemon", "mirth",
            "north", "oxide", "plume", "quilt", "robin", "stone", "tunic", "umbra",
            "venom", "wheat", "xerox", "yeast", "zilch", "basic", "creed", "dense",
            "elder", "fiber", "grain", "hound", "inner", "joint", "knife", "latch",
            "miner", "nexus", "organ", "patch", "realm", "scent", "truce", "usual",
            "vault", "wrist", "acute", "blade", "charm", "dwarf", "equip", "flint",
            "giant", "haste", "input", "jumbo", "knelt", "lucid", "mango", "ninth",
            "outer", "prism", "quota", "ridge", "slate", "thumb", "uncut", "vocal",
            "weave", "pixel", "yacht", "zone", "brisk", "clamp", "dough", "exact",
        ]
    random.Random(seed).shuffle(words)
    return words


def generate_cwe_example(
    words_pool,
    num_words,
    freq_cw=30,
    freq_ucw=3,
    num_cw=10,
    seed=0,
):
    """Generate a CWE example: a numbered word list with common/uncommon words."""
    rng = random.Random(seed)
    # Use random.choices (with replacement) when pool is smaller than needed
    unique_needed = num_words
    if unique_needed <= len(words_pool):
        sampled = rng.sample(words_pool, unique_needed)
    else:
        sampled = rng.choices(words_pool, k=unique_needed)
    common = sampled[:num_cw]
    uncommon = sampled[num_cw:]
    word_list = common * freq_cw + uncommon * freq_ucw
    rng.shuffle(word_list)
    context = " ".join(f"{i+1}. {w}" for i, w in enumerate(word_list))
    return context, common


def build_prompt(context, num_cw, few_shot_context, few_shot_answer):
    """Build the full prompt with one-shot example (RULER default)."""
    main_prompt = CWE_TEMPLATE.format(context=context, num_cw=num_cw)
    if few_shot_context is not None:
        return CWE_ONE_SHOT_TEMPLATE.format(
            example_context=few_shot_context,
            example_answer=" ".join(f"1. {w}" for i, w in enumerate(few_shot_answer)),
            num_cw=num_cw,
            main_prompt=main_prompt,
        )
    return main_prompt


def normalize_word(w):
    return w.strip().lower().strip(".,;:!?\"'()[]{}")


def compute_score(predicted_text, ground_truth_words, num_cw):
    """Exact-match: count how many of the top-N ground truth words appear in prediction."""
    pred_words = set(normalize_word(w) for w in predicted_text.split())
    gt_words = set(normalize_word(w) for w in ground_truth_words)
    correct = len(pred_words & gt_words)
    return correct / num_cw


# ── Main evaluation ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RULER CWE baseline eval")
    parser.add_argument("--model_path", type=str,
                        default="models/Qwen--Qwen3-8b",
                        help="Path to Qwen3-8B model")
    parser.add_argument("--context_length", type=int, default=65536,
                        help="Target context length in tokens")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of evaluation samples")
    parser.add_argument("--num_cw", type=int, default=10,
                        help="Number of common words to retrieve")
    parser.add_argument("--freq_cw", type=int, default=30,
                        help="Frequency of common words (RULER default)")
    parser.add_argument("--freq_ucw", type=int, default=3,
                        help="Frequency of uncommon words (RULER default)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="results/ruler_cwe_64k.json",
                        help="Path to save results JSON")
    parser.add_argument("--max_new_tokens", type=int, default=TOKENS_TO_GENERATE)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    device = f"cuda:{args.gpu_id}"
    logger.info(f"Loading model from {args.model_path} on {device} ...")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": device},
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.eval()
    logger.info("Model loaded.")

    # Load word pool
    words_pool = load_word_list(args.seed)
    logger.info(f"Word pool size: {len(words_pool)}")

    # Determine max context tokens for the input (excluding generated tokens)
    max_input_tokens = args.context_length - args.max_new_tokens - 50  # safety margin

    # Binary search for optimal num_words that fits in max_input_tokens
    logger.info("Calibrating word list size to fit %d tokens ...", max_input_tokens)
    one_shot_ctx, one_shot_ans = generate_cwe_example(
        words_pool, 40, freq_cw=10, freq_ucw=3, num_cw=args.num_cw, seed=args.seed,
    )
    one_shot_prompt = CWE_ONE_SHOT_TEMPLATE.format(
        example_context=one_shot_ctx,
        example_answer=" ".join(f"1. {w}" for w in one_shot_ans),
        num_cw=args.num_cw,
        main_prompt="{placeholder}",
    )
    # Measure overhead of one-shot prefix
    prefix_tokens = len(tokenizer.encode(one_shot_prompt.split("{placeholder")[0]))

    # Estimate upper bound: each unique word contributes freq_cw entries,
    # so total entries = num_cw*freq_cw + (num_words-num_cw)*freq_ucw
    # Each entry ~4 tokens. Upper = max_input_tokens / avg_tokens_per_entry
    est_upper = max(500, min(max_input_tokens // 3, 50000))
    lower, upper = 100, est_upper
    optimal_num_words = 100
    while lower <= upper:
        mid = (lower + upper) // 2
        test_ctx, _ = generate_cwe_example(
            words_pool, mid, args.freq_cw, args.freq_ucw, args.num_cw, seed=0,
        )
        test_prompt = build_prompt(test_ctx, args.num_cw, one_shot_ctx, one_shot_ans)
        n_tokens = len(tokenizer.encode(test_prompt))
        if n_tokens <= max_input_tokens:
            optimal_num_words = mid
            lower = mid + 1
        else:
            upper = mid - 1
    logger.info(f"Optimal num_words={optimal_num_words}")

    # Run evaluation
    scores = []
    details = []
    t0 = time.time()

    for i in tqdm(range(args.num_samples), desc="Evaluating"):
        context, answer = generate_cwe_example(
            words_pool, optimal_num_words,
            freq_cw=args.freq_cw, freq_ucw=args.freq_ucw,
            num_cw=args.num_cw, seed=args.seed + i + 1,
        )
        prompt = build_prompt(context, args.num_cw, one_shot_ctx, one_shot_ans)
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        input_len = input_ids.shape[1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=1.0,
                top_k=1,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = outputs[0][input_len:]
        pred_text = tokenizer.decode(generated, skip_special_tokens=True)

        score = compute_score(pred_text, answer, args.num_cw)
        scores.append(score)
        details.append({
            "sample_idx": i,
            "score": score,
            "input_tokens": input_len,
            "prediction": pred_text[:200],
            "ground_truth": answer,
        })

    elapsed = time.time() - t0
    avg_score = sum(scores) / len(scores)

    results = {
        "model": args.model_path,
        "task": "cwe",
        "context_length": args.context_length,
        "num_samples": args.num_samples,
        "num_cw": args.num_cw,
        "freq_cw": args.freq_cw,
        "freq_ucw": args.freq_ucw,
        "avg_accuracy": avg_score,
        "scores": scores,
        "details": details,
        "elapsed_seconds": elapsed,
        "optimal_num_words": optimal_num_words,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"RULER CWE Evaluation Results")
    print(f"{'='*60}")
    print(f"Model:           {args.model_path}")
    print(f"Context length:  {args.context_length}")
    print(f"Task:            CWE (Common Words Extraction)")
    print(f"Num common:      {args.num_cw}, freq_cw={args.freq_cw}, freq_ucw={args.freq_ucw}")
    print(f"Samples:         {args.num_samples}")
    print(f"Avg Accuracy:    {avg_score:.4f} ({sum(s==1 for s in scores)}/{args.num_samples} perfect)")
    print(f"Time:            {elapsed:.1f}s")
    print(f"Results saved:   {out_path}")
    print(f"{'='*60}")

    return avg_score


if __name__ == "__main__":
    main()
