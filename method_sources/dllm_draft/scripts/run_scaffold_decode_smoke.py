#!/usr/bin/env python3
"""Run constrained neural Scaffold decoding from the resumed smoke checkpoint."""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.decoder_runtime import DecoderConfig, DecoderRuntime
from scaffold_coder.model_sampler import SamplerConfig, ScaffoldModelSampler
from scaffold_coder.process_metrics import compute_process_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--structural-confidence-threshold", type=float)
    parser.add_argument("--structural-max-defer-calls", type=int, default=0)
    parser.add_argument("--leaf-remask-fraction", type=float, default=0.0)
    parser.add_argument("--leaf-remask-interval", type=int, default=0)
    parser.add_argument("--leaf-remask-confidence-threshold", type=float)
    parser.add_argument("--leaf-remask-min-age-calls", type=int, default=1)
    parser.add_argument("--max-leaf-remasks", type=int, default=0)
    parser.add_argument("--max-leaf-remasks-per-token", type=int, default=1)
    parser.add_argument(
        "--structural-backtrack-confidence-threshold",
        type=float,
    )
    parser.add_argument(
        "--structural-backtrack-min-age-calls",
        type=int,
        default=1,
    )
    parser.add_argument("--max-structural-backtracks", type=int, default=0)
    parser.add_argument(
        "--max-structural-backtracks-per-anchor",
        type=int,
        default=1,
    )
    args = parser.parse_args()

    checkpoint = str(Path(args.checkpoint).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    device = torch.device("cuda:0")
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    runtime = DecoderRuntime(
        registry,
        DecoderConfig(
            initial_root_slots=1,
            max_canvas_tokens=512,
            max_expansions=128,
        ),
    )
    sampler = ScaffoldModelSampler(
        model,
        registry,
        SamplerConfig(
            max_model_calls=256,
            transfer_tokens=1,
            temperature=0.0,
            confidence="normalized_entropy",
            keep_history=False,
            structural_confidence_threshold=(
                args.structural_confidence_threshold
            ),
            structural_max_defer_calls=args.structural_max_defer_calls,
            leaf_remask_fraction=args.leaf_remask_fraction,
            leaf_remask_interval=args.leaf_remask_interval,
            leaf_remask_confidence_threshold=(
                args.leaf_remask_confidence_threshold
            ),
            leaf_remask_min_age_calls=args.leaf_remask_min_age_calls,
            max_leaf_remasks=args.max_leaf_remasks,
            max_leaf_remasks_per_token=args.max_leaf_remasks_per_token,
            structural_backtrack_confidence_threshold=(
                args.structural_backtrack_confidence_threshold
            ),
            structural_backtrack_min_age_calls=(
                args.structural_backtrack_min_age_calls
            ),
            max_structural_backtracks=args.max_structural_backtracks,
            max_structural_backtracks_per_anchor=(
                args.max_structural_backtracks_per_anchor
            ),
        ),
    )
    started = time.perf_counter()
    result = sampler.generate(
        "Write only Python code for a function identity(x) that returns x.",
        runtime=runtime,
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started
    ast.parse(result.text)
    process_metrics = compute_process_metrics(result)
    report = {
        "checkpoint": checkpoint,
        "text": result.text,
        "model_calls": result.model_calls,
        "final_canvas_tokens": result.final_canvas_tokens,
        "expansions": result.expansions,
        "model_canvas_lengths": result.model_canvas_lengths,
        "cumulative_model_tokens": result.cumulative_model_tokens,
        "placeholder_parse_steps": len(result.placeholder_history),
        "process_metrics": process_metrics.to_dict(),
        "elapsed_seconds": elapsed,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
