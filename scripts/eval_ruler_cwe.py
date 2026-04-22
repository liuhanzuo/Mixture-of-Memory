#!/usr/bin/env python3
"""
eval_ruler_cwe.py — RULER Common Words Extraction (CWE) evaluation at configurable context lengths.

Self-contained: generates CWE synthetic data, runs inference on a HuggingFace model,
and computes string-match accuracy — no external RULER repo needed at eval time.

Based on NVIDIA RULER (https://github.com/NVIDIA/RULER, arXiv:2404.06654).

Usage:
    python scripts/eval_ruler_cwe.py \
        --model_path models/Qwen--Qwen3-8b \
        --max_seq_length 65536 \
        --num_samples 50 \
        --output_file eval_results/ruler_cwe_64k.json

Dependencies:
    pip install wonderwords transformers torch accelerate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, List

import torch

# ---------------------------------------------------------------------------
# Project root for sibling imports
# ---------------------------------------------------------------------------
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ============================================================
# Word sources (mirrors RULER's approach)
# ============================================================

def _load_word_sources(seed: int = 42):
    """Load wonderwords noun/adj/verb lists; fall back to NLTK or a large synthetic list."""
    try:
        import wonderwords.random_word
        nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
        adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
        verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
        words = sorted(list(set(nouns + adjs + verbs)))
    except Exception:
        logger.warning("wonderwords unavailable, trying NLTK words...")
        words = None
        try:
            import nltk
            nltk.data.find('corpora/words')
        except Exception:
            try:
                import nltk
                nltk.download('words', quiet=True)
            except Exception:
                pass
        try:
            import nltk
            from nltk.corpus import words as nltk_words
            words = list(set(w.lower() for w in nltk_words.words() if w.isalpha() and 3 <= len(w) <= 12))
        except Exception:
            pass

        if not words or len(words) < 1000:
            logger.warning("NLTK words unavailable, generating large synthetic word list")
            import string, hashlib
            rng = random.Random(seed)
            # Generate ~20k unique pronounceable-like words
            seen = set()
            consonants = 'bcdfghjklmnpqrstvwxyz'
            vowels = 'aeiou'
            words = []
            while len(words) < 20000:
                length = rng.randint(4, 10)
                w = ''.join(rng.choice(consonants if i % 2 == 0 else vowels) for i in range(length))
                if w not in seen:
                    seen.add(w)
                    words.append(w)

    random.Random(seed).shuffle(words)
    return words


def _load_english_words_fallback(seed: int = 42):
    """Try to load RULER's english_words.json; fall back to word sources or synthetic generation."""
    candidates = [
        Path("/tmp/ruler_repo/scripts/data/synthetic/json/english_words.json"),
        _project_root / "data" / "english_words.json",
        _project_root / "repo" / "ruler" / "scripts" / "data" / "synthetic" / "json" / "english_words.json",
    ]
    for p in candidates:
        if p.exists() and p.stat().st_size > 10:
            try:
                with open(p) as f:
                    return list(json.load(f).values())
            except (json.JSONDecodeError, Exception):
                logger.warning(f"Failed to parse {p}, using synthetic fallback")

    # Reuse the same word source as primary
    return _load_word_sources(seed + 100)


# ============================================================
# CWE data generation
# ============================================================

CWE_TEMPLATE = (
    "Below is a numbered list of words. In these words, "
    "some appear more often than others. Memorize the ones that appear most often.\n"
    "{context}\n"
    "Question: What are the 10 most common words in the above list?"
)

CWE_ANSWER_PREFIX = "Answer: The top 10 words that appear most often in the list are:"

TOKENS_TO_GENERATE = 120


def generate_cwe_sample(
    words: List[str],
    randle_words: List[str],
    tokenizer,
    num_cw: int = 10,
    freq_cw: int = 30,
    freq_ucw: int = 3,
    max_seq_length: int = 65536,
    tokens_to_generate: int = TOKENS_TO_GENERATE,
    num_fewshot: int = 1,
    seed: int = 42,
):
    """Generate a single CWE example that fits within max_seq_length."""
    rng = random.Random(seed)

    def get_example(num_words, common_repeats, uncommon_repeats, common_nums):
        if num_words <= len(words):
            word_list_full = rng.sample(words, num_words)
        else:
            word_list_full = rng.sample(randle_words, num_words)

        common = word_list_full[:common_nums]
        uncommon = word_list_full[common_nums:]
        word_list = common * int(common_repeats) + uncommon * int(uncommon_repeats)
        rng.shuffle(word_list)
        context = ' '.join([f"{i + 1}. {w}" for i, w in enumerate(word_list)])
        return context, common

    # --- Few-shot examples ---
    few_shots = []
    if max_seq_length < 4096:
        for _ in range(num_fewshot):
            ctx_ex, ans_ex = get_example(20, 3, 1, num_cw)
            few_shots.append((ctx_ex, ans_ex))
    else:
        for _ in range(num_fewshot):
            ctx_ex, ans_ex = get_example(40, 10, 3, num_cw)
            few_shots.append((ctx_ex, ans_ex))

    # --- Main example: estimate optimal num_words to fill context ---
    # Build a small sample to estimate tokens-per-word
    sample_ctx, _ = get_example(100, freq_cw, freq_ucw, num_cw)
    sample_text = CWE_TEMPLATE.format(context=sample_ctx)
    sample_tok_count = len(tokenizer.encode(sample_text))
    tokens_per_word = sample_tok_count / 100

    # Build few-shot prefix once and measure its token cost
    fewshot_text = ""
    for fs_ctx, fs_ans in few_shots:
        fs_text = CWE_TEMPLATE.format(context=fs_ctx)
        fs_text += "\n" + CWE_ANSWER_PREFIX + " " + ', '.join(fs_ans)
        fewshot_text += fs_text + "\n\n"
    fewshot_tok_count = len(tokenizer.encode(fewshot_text))

    # Available tokens for the main example context
    available_for_context = max_seq_length - fewshot_tok_count - tokens_to_generate - 200  # 200 for template overhead
    # Each word appears in the list as "N. word " (~3-4 tokens per entry depending on number length)
    # but freq_cw copies of common + freq_ucw copies of uncommon
    total_words_placed = num_cw * freq_cw + (0)  # approximate
    tokens_per_entry = tokens_per_word  # rough estimate for "N. word"
    # More accurate: total tokens = template_overhead + sum_of_all_entries
    # Each entry: len(str(i+1)) + 2 (". ") + len(word_tokens)
    # Average number length: ~log10(total_entries)
    # So: available / (tokens_per_word + tokens_for_number) ≈ optimal_num_words
    tokens_for_number = 3  # average "N. " tokens
    optimal_num_words = max(10, int(available_for_context / ((freq_cw * tokens_per_entry + tokens_for_number) / freq_cw + tokens_for_number)))
    
    # Refine: binary search but with limited iterations (max 20)
    lower, upper = 10, max(optimal_num_words * 3, 100)
    for _ in range(20):
        if lower > upper:
            break
        mid = (lower + upper) // 2
        ctx, ans = get_example(mid, freq_cw, freq_ucw, num_cw)
        main_text = CWE_TEMPLATE.format(context=ctx)
        full_text = fewshot_text + main_text
        total = len(tokenizer.encode(full_text)) + tokens_to_generate
        if total <= max_seq_length:
            optimal_num_words = mid
            lower = mid + 1
        else:
            upper = mid - 1

    # Generate the actual sample
    context, answer = get_example(optimal_num_words, freq_cw, freq_ucw, num_cw)
    main_text = CWE_TEMPLATE.format(context=context)

    # Build few-shot prefix (with proper answer format)
    fewshot_text = ""
    for fs_ctx, fs_ans in few_shots:
        fs_text = CWE_TEMPLATE.format(context=fs_ctx)
        fs_text += "\n" + CWE_ANSWER_PREFIX + " " + ', '.join(fs_ans)
        fewshot_text += fs_text + "\n\n"

    full_input = fewshot_text + main_text

    return full_input, answer


def generate_cwe_dataset(
    tokenizer,
    num_samples: int = 50,
    max_seq_length: int = 65536,
    num_cw: int = 10,
    freq_cw: int = 30,
    freq_ucw: int = 3,
    seed: int = 42,
    num_fewshot: int = 1,
):
    """Generate a full CWE dataset."""
    words = _load_word_sources(seed)
    randle_words = _load_english_words_fallback(seed)

    samples = []
    for i in range(num_samples):
        input_text, answer = generate_cwe_sample(
            words=words,
            randle_words=randle_words,
            tokenizer=tokenizer,
            num_cw=num_cw,
            freq_cw=freq_cw,
            freq_ucw=freq_ucw,
            max_seq_length=max_seq_length,
            seed=seed + i,
            num_fewshot=num_fewshot,
        )
        samples.append({"index": i, "input": input_text, "answer": answer})
        if (i + 1) % 10 == 0:
            logger.info(f"Generated {i+1}/{num_samples} samples")

    return samples


# ============================================================
# Model inference
# ============================================================

@torch.no_grad()
def run_inference(
    model,
    tokenizer,
    prompt: str,
    answer_prefix: str = CWE_ANSWER_PREFIX,
    max_new_tokens: int = TOKENS_TO_GENERATE,
    temperature: float = 0.0,
    apply_chat_template: bool = False,
):
    """Run model inference on a single CWE prompt."""
    # Truncate if needed to fit within model's max length
    model_max = getattr(model.config, "max_position_embeddings", 32768)
    # Append the answer prefix to the prompt so the model starts generating the answer directly
    full_prompt = prompt + "\n" + answer_prefix

    # Apply chat template if requested (needed for base models like Qwen3)
    if apply_chat_template and hasattr(tokenizer, 'apply_chat_template'):
        # Put the full prompt (including question) as user message
        messages = [{"role": "user", "content": prompt}]
        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # Append answer prefix after the assistant generation prompt
        full_prompt = full_prompt + answer_prefix
    else:
        full_prompt = prompt + "\n" + answer_prefix
    input_ids = tokenizer.encode(full_prompt, add_special_tokens=True)
    original_len = len(input_ids)
    if len(input_ids) > model_max - max_new_tokens:
        input_ids = input_ids[:model_max - max_new_tokens]
        logger.warning(f"Truncated input from {original_len} to {len(input_ids)} tokens to fit model max length ({model_max})")

    input_tensor = torch.tensor([input_ids], device=model.device)
    attention_mask = torch.ones_like(input_tensor)

    outputs = model.generate(
        input_tensor,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0 else None,
        do_sample=temperature > 0,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
    )

    generated_ids = outputs[0][len(input_ids):]
    prediction = tokenizer.decode(generated_ids, skip_special_tokens=True)

    return prediction.strip()


# ============================================================
# Metrics
# ============================================================

def string_match_all(preds: List[str], refs: List[List[str]]) -> float:
    """
    RULER's CWE metric: for each sample, compute fraction of reference words
    found (case-insensitive substring match), then average across samples.
    """
    scores = []
    for pred, ref in zip(preds, refs):
        matched = sum(1.0 for r in ref if r.lower() in pred.lower())
        scores.append(matched / len(ref))
    return sum(scores) / len(scores) * 100 if scores else 0.0


def string_match_part(preds: List[str], refs: List[List[str]]) -> float:
    """At least one reference word found per sample."""
    scores = []
    for pred, ref in zip(preds, refs):
        matched = any(r.lower() in pred.lower() for r in ref)
        scores.append(1.0 if matched else 0.0)
    return sum(scores) / len(scores) * 100 if scores else 0.0


# ============================================================
# Main evaluation loop
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="RULER CWE evaluation")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--max_seq_length", type=int, default=65536,
                        help="Target context length (default: 64K)")
    parser.add_argument("--num_samples", type=int, default=50,
                        help="Number of CWE samples to evaluate")
    parser.add_argument("--num_cw", type=int, default=10,
                        help="Number of common words to extract")
    parser.add_argument("--freq_cw", type=int, default=30,
                        help="Frequency of common words")
    parser.add_argument("--freq_ucw", type=int, default=3,
                        help="Frequency of uncommon words")
    parser.add_argument("--tokens_to_generate", type=int, default=TOKENS_TO_GENERATE,
                        help="Max tokens to generate per sample")
    parser.add_argument("--num_fewshot", type=int, default=1,
                        help="Number of few-shot examples")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_file", type=str, default="",
                        help="Path to save results JSON")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype (bfloat16, float16, float32)")
    parser.add_argument("--attn_impl", type=str, default="sdpa",
                        help="Attention implementation (flash_attention_2, sdpa, eager)")
    parser.add_argument("--no_fewshot", action="store_true",
                        help="Disable few-shot examples")
    parser.add_argument("--dry_run", type=int, default=0,
                        help="If > 0, only run this many samples (for testing)")
    parser.add_argument("--max_new_tokens_override", type=int, default=0,
                        help="Override max_new_tokens for generation")
    parser.add_argument("--apply_chat_template", action="store_true",
                        help="Apply tokenizer's chat template to format prompts (needed for base models like Qwen3)")
    args = parser.parse_args()

    num_samples = args.dry_run if args.dry_run > 0 else args.num_samples
    max_new_tokens = args.max_new_tokens_override if args.max_new_tokens_override > 0 else args.tokens_to_generate
    num_fewshot = 0 if args.no_fewshot else args.num_fewshot

    logger.info(f"RULER CWE Evaluation")
    logger.info(f"  Model: {args.model_path}")
    logger.info(f"  Context length: {args.max_seq_length}")
    logger.info(f"  Samples: {num_samples}")
    logger.info(f"  CWE config: freq_cw={args.freq_cw}, freq_ucw={args.freq_ucw}, num_cw={args.num_cw}")
    logger.info(f"  Few-shot: {num_fewshot}")
    logger.info(f"  Chat template: {args.apply_chat_template}")
    logger.info(f"  Seed: {args.seed}")

    # --- Load tokenizer ---
    from transformers import AutoTokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # --- Generate dataset ---
    logger.info("Generating CWE dataset...")
    t0 = time.time()
    dataset = generate_cwe_dataset(
        tokenizer=tokenizer,
        num_samples=num_samples,
        max_seq_length=args.max_seq_length,
        num_cw=args.num_cw,
        freq_cw=args.freq_cw,
        freq_ucw=args.freq_ucw,
        seed=args.seed,
        num_fewshot=num_fewshot,
    )
    logger.info(f"Dataset generated in {time.time()-t0:.1f}s, {len(dataset)} samples")

    # Log sample stats
    for i in [0]:
        tok_count = len(tokenizer.encode(dataset[i]["input"]))
        logger.info(f"  Sample {i}: {tok_count} tokens, answer has {len(dataset[i]['answer'])} words")

    # --- Load model ---
    from transformers import AutoModelForCausalLM
    logger.info("Loading model...")
    torch_dtype = getattr(torch, args.dtype)
    attn_kwargs = {}
    if args.attn_impl == "flash_attention_2":
        attn_kwargs["attn_implementation"] = "flash_attention_2"
    elif args.attn_impl == "sdpa":
        attn_kwargs["attn_implementation"] = "sdpa"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=True,
        **attn_kwargs,
    )

    # Extend RoPE if target context length exceeds model default
    model_max_emb = getattr(model.config, "max_position_embeddings", 32768)
    if args.max_seq_length > model_max_emb:
        scaling_factor = args.max_seq_length / model_max_emb
        logger.info(f"Extending RoPE from {model_max_emb} to {args.max_seq_length} (factor={scaling_factor:.2f})")
        model.config.rope_scaling = {"type": "linear", "factor": scaling_factor}
        model.config.max_position_embeddings = args.max_seq_length
        # Re-initialize rotary embeddings with new max length
        for module in model.modules():
            if hasattr(module, 'max_seq_len'):
                module.max_seq_len = args.max_seq_length
            if hasattr(module, 'rope_theta') and hasattr(module, '_set_cos_sin_cache'):
                try:
                    module._set_cos_sin_cache(seqlen=args.max_seq_length, device=model.device, dtype=torch_dtype)
                except Exception:
                    pass
    model.eval()
    logger.info(f"Model loaded on {model.device}")

    # --- Run evaluation ---
    logger.info("Running evaluation...")
    predictions = []
    references = []
    detailed_results = []
    t_eval_start = time.time()

    for sample in dataset:
        idx = sample["index"]
        prompt = sample["input"]
        answer = sample["answer"]

        pred = run_inference(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            apply_chat_template=args.apply_chat_template,
        )
        predictions.append(pred)
        references.append(answer)

        # Per-sample accuracy
        matched = sum(1 for w in answer if w.lower() in pred.lower())
        sample_acc = matched / len(answer) * 100

        detailed_results.append({
            "index": idx,
            "prediction": pred,
            "answer": answer,
            "num_matched": matched,
            "num_total": len(answer),
            "accuracy": round(sample_acc, 2),
        })

        if (idx + 1) % 5 == 0 or (idx + 1) == len(dataset):
            elapsed = time.time() - t_eval_start
            logger.info(f"  [{idx+1}/{len(dataset)}] acc={sample_acc:.1f}% ({matched}/{len(answer)}) "
                        f"| elapsed={elapsed:.1f}s")

    # --- Compute metrics ---
    acc_all = string_match_all(predictions, references)
    acc_part = string_match_part(predictions, references)
    total_time = time.time() - t_eval_start

    logger.info("=" * 60)
    logger.info(f"RULER CWE Results @ {args.max_seq_length} tokens")
    logger.info(f"  string_match_all (primary): {acc_all:.2f}%")
    logger.info(f"  string_match_part:          {acc_part:.2f}%")
    logger.info(f"  Samples:                    {len(dataset)}")
    logger.info(f"  Total eval time:            {total_time:.1f}s")
    logger.info("=" * 60)

    # --- Save results ---
    result = {
        "task": "common_words_extraction",
        "model_path": args.model_path,
        "max_seq_length": args.max_seq_length,
        "num_samples": len(dataset),
        "num_cw": args.num_cw,
        "freq_cw": args.freq_cw,
        "freq_ucw": args.freq_ucw,
        "num_fewshot": num_fewshot,
        "seed": args.seed,
        "string_match_all": round(acc_all, 2),
        "string_match_part": round(acc_part, 2),
        "eval_time_seconds": round(total_time, 1),
        "detailed_results": detailed_results,
    }

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        logger.info(f"Results saved to {out_path}")
    else:
        # Print to stdout if no output file
        print(json.dumps(result, indent=2, ensure_ascii=False))

    return result


if __name__ == "__main__":
    main()
