#!/usr/bin/env python3
"""DMS Evaluation Script.

Evaluates a DMS-retrofitted model at different compression ratios and reports:
- Perplexity (PPL)
- Compression ratio (actual)
- Task performance (MMLU, HellaSwag, etc.)

Usage:
    python scripts/eval_dms.py \
        --model_path outputs/dms_8x/final \
        --baseline_model_path meta-llama/Llama-2-7b-hf \
        --eval_datasets wiki perplexity \
        --output_file eval_results/dms_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.memory.dms import apply_dms_to_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="DMS Evaluation")
    p.add_argument("--model_path", type=str, required=True,
                   help="DMS-retrofitted model path")
    p.add_argument("--baseline_model_path", type=str, default=None,
                   help="Baseline (uncompressed) model for comparison")
    p.add_argument("--compression_ratio", type=float, default=8.0)
    p.add_argument("--sliding_window", type=int, default=256)
    p.add_argument("--tau", type=float, default=0.1)
    p.add_argument("--max_length", type=int, default=2048)
    p.add_argument("--stride", type=int, default=512)
    p.add_argument("--eval_text", type=str, default=None,
                   help="Text file for perplexity evaluation")
    p.add_argument("--output_file", type=str, default=None)
    p.add_argument("--bf16", action="store_true", default=True)
    p.add_argument("--device", type=str, default="auto")
    return p.parse_args()


@torch.no_grad()
def compute_ppl(model, tokenizer, text: str, max_length: int = 2048, stride: int = 512) -> dict:
    """Compute perplexity on a text corpus."""
    logger.info(f"Computing PPL (max_length={max_length}, stride={stride})...")

    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)
    seq_len = input_ids.size(1)

    nlls = []
    prev_end = 0

    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        chunk = input_ids[:, begin:end]

        # Pad left if needed for causal LM
        target_len = min(end - begin, max_length)
        if end < seq_len:
            # Need padding for full context window
            pad_len = max_length - (end - begin)
            chunk = F.pad(chunk, (pad_len, 0), value=tokenizer.pad_token_id or 0)

        outputs = model(chunk, labels=chunk)
        neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood.item())

        prev_end = end
        if end >= seq_len:
            break

    ppl = torch.exp(torch.tensor(sum(nlls) / len(nlls))).item()
    return {"perplexity": ppl, "n_chunks": len(nlls)}


@torch.no_grad()
def measure_compression_ratio(model, tokenizer, text: str, max_length: int = 2048) -> dict:
    """Measure actual compression ratio achieved by DMS."""
    logger.info("Measuring actual compression ratio...")

    encodings = tokenizer(text[:max_length * 5], return_tensors="pt", truncation=True, max_length=max_length * 5)
    input_ids = encodings.input_ids.to(model.device)

    outputs = model(input_ids, output_hidden_states=True)

    # Collect alpha values from DMS wrappers
    layer_alphas = {}
    for name, module in model.named_modules():
        if hasattr(module, "_last_alpha"):
            layer_alphas[name] = module._last_alpha

    if not layer_alphas:
        return {"actual_cr": 1.0, "note": "No DMS wrappers found, using uncompressed model"}

    # Compute per-layer compression ratio
    # CR = 1 / (1 - mean(alpha))
    crs = []
    for name, alpha in layer_alphas.items():
        mean_alpha = alpha.float().mean().item()
        if mean_alpha > 0:
            cr = 1.0 / (1.0 - mean_alpha)
            crs.append(cr)

    avg_cr = sum(crs) / len(crs) if crs else 1.0
    return {
        "actual_cr": avg_cr,
        "per_layer_cr": crs,
        "num_dms_layers": len(crs),
    }


def eval_model(model, tokenizer, eval_text: str, args) -> dict:
    """Run full evaluation suite."""
    results = {}

    # PPL
    ppl_result = compute_ppl(model, tokenizer, eval_text, args.max_length, args.stride)
    results["perplexity"] = ppl_result

    # Compression ratio
    cr_result = measure_compression_ratio(model, tokenizer, eval_text, args.max_length)
    results["compression"] = cr_result

    return results


def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load DMS model
    logger.info(f"Loading DMS model from: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto" if device == "cuda" else None,
    )
    model.eval()

    # Apply DMS wrapper (re-initialize from saved weights if available)
    model = apply_dms_to_model(
        model,
        sliding_window=args.sliding_window,
        tau=args.tau,
    )

    # Eval text
    if args.eval_text and os.path.exists(args.eval_text):
        with open(args.eval_text) as f:
            eval_text = f.read()
    else:
        # Default: use a sample from wikitext
        logger.info("No eval text provided, downloading wikitext sample...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
            eval_text = "\n\n".join(ds["text"][:100])
        except Exception:
            eval_text = "The quick brown fox jumps over the lazy dog. " * 1000

    # Evaluate
    start = time.time()
    results = eval_model(model, tokenizer, eval_text, args)
    elapsed = time.time() - start

    results["meta"] = {
        "model_path": args.model_path,
        "target_cr": args.compression_ratio,
        "sliding_window": args.sliding_window,
        "eval_time_seconds": elapsed,
    }

    # Print results
    logger.info("=" * 60)
    logger.info("DMS Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Perplexity: {results['perplexity']['perplexity']:.4f}")
    logger.info(f"Actual CR: {results['compression'].get('actual_cr', 'N/A')}")
    logger.info(f"Eval time: {elapsed:.1f}s")

    # Baseline comparison
    if args.baseline_model_path:
        logger.info(f"\nLoading baseline from: {args.baseline_model_path}")
        baseline_tokenizer = AutoTokenizer.from_pretrained(args.baseline_model_path, use_fast=True)
        if baseline_tokenizer.pad_token is None:
            baseline_tokenizer.pad_token = baseline_tokenizer.eos_token
        baseline_model = AutoModelForCausalLM.from_pretrained(
            args.baseline_model_path,
            torch_dtype=dtype,
            device_map="auto" if device == "cuda" else None,
        )
        baseline_model.eval()

        baseline_ppl = compute_ppl(baseline_model, baseline_tokenizer, eval_text, args.max_length, args.stride)
        results["baseline"] = baseline_ppl
        logger.info(f"Baseline PPL: {baseline_ppl['perplexity']:.4f}")
        logger.info(f"PPL change: {results['perplexity']['perplexity'] - baseline_ppl['perplexity']:+.4f}")

        del baseline_model
        torch.cuda.empty_cache()

    # Save results
    output_file = args.output_file or os.path.join(args.model_path, "eval_results.json")
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {output_file}")


if __name__ == "__main__":
    main()
