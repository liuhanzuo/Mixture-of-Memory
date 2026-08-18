#!/usr/bin/env python3
"""Eight-shard full-solution EvalPlus generation with process sidecars."""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.decoder_runtime import DecoderConfig, DecoderRuntime
from scaffold_coder.errors import BudgetExceededError
from scaffold_coder.model_sampler import SamplerConfig, ScaffoldModelSampler
from scaffold_coder.process_metrics import compute_process_metrics


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prompt_for(dataset: str, task: dict) -> str:
    if dataset == "humaneval":
        return (
            "Write a complete Python solution for the following function. "
            "Return only Python code and preserve the required function name.\n\n"
            + task["prompt"]
        )
    return (
        "Write a complete Python solution for the following programming task. "
        "Return only Python code and preserve the required function name.\n\n"
        + task["prompt"]
    )


def function_header_from_prompt(prompt: str) -> str | None:
    match = re.search(r"(?m)^def\s+(.+):\s*$", prompt)
    return match.group(1).strip() if match else None


def function_header_for_task(task: dict) -> str | None:
    explicit = str(task.get("function_header") or "").strip()
    return explicit or function_header_from_prompt(str(task["prompt"]))


def termination_reason(exc: Exception) -> str:
    if isinstance(exc, BudgetExceededError):
        message = str(exc)
        if "model calls" in message:
            return "model_call_budget"
        if "tree depth" in message:
            return "depth_capacity_exhausted"
        if "total line" in message:
            return "total_line_capacity_exhausted"
        if "body line" in message:
            return "line_capacity_exhausted"
        if "token-hole" in message:
            return "token_capacity_exhausted"
        if "expansion" in message:
            return "expansion_budget_exhausted"
        if "canvas" in message or "context" in message:
            return "context_limit"
        return "budget_exhausted"
    return f"exception:{type(exc).__name__}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-model-calls", type=int, default=512)
    parser.add_argument("--transfer-tokens", type=int, default=1)
    parser.add_argument("--seed-function-signature", action="store_true")
    parser.add_argument("--initial-root-slots", type=int, default=1)
    parser.add_argument("--initial-body-slots", type=int, default=2)
    parser.add_argument("--initial-statement-masks", type=int, default=4)
    parser.add_argument("--initial-statement-masks-shallow", type=int)
    parser.add_argument("--statement-shallow-depth", type=int, default=1)
    parser.add_argument(
        "--initial-function-header-masks",
        type=int,
        default=4,
    )
    parser.add_argument("--initial-loop-header-masks", type=int, default=4)
    parser.add_argument("--initial-condition-masks", type=int, default=3)
    parser.add_argument("--max-canvas-tokens", type=int, default=1024)
    parser.add_argument("--max-tree-depth", type=int, default=16)
    parser.add_argument("--max-lines-per-body", type=int, default=128)
    parser.add_argument("--max-total-lines", type=int, default=1024)
    parser.add_argument("--max-tokens-per-hole", type=int, default=512)
    parser.add_argument("--max-expansions", type=int, default=512)
    parser.add_argument("--runtime-config-label", default="custom")
    parser.add_argument(
        "--no-module-expand",
        action="store_true",
    )
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
    parser.add_argument(
        "--body-construct-logit-penalty",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--body-stmt-logit-bonus",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--token-expand-logit-bonus",
        type=float,
        default=0.0,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--no-leaf-remask-at-completion",
        action="store_true",
    )
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    tasks = list(read_jsonl(Path(args.data_file)))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    assigned = [
        task for index, task in enumerate(tasks) if index % world_size == rank
    ]

    checkpoint = str(Path(args.checkpoint).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    sampler = ScaffoldModelSampler(
        model,
        registry,
        SamplerConfig(
            max_model_calls=args.max_model_calls,
            transfer_tokens=args.transfer_tokens,
            temperature=0.0,
            confidence="normalized_entropy",
            keep_history=True,
            structural_confidence_threshold=(
                args.structural_confidence_threshold
            ),
            structural_max_defer_calls=args.structural_max_defer_calls,
            leaf_remask_fraction=args.leaf_remask_fraction,
            leaf_remask_interval=args.leaf_remask_interval,
            leaf_remask_at_completion=(
                not args.no_leaf_remask_at_completion
            ),
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
            body_construct_logit_penalty=(
                args.body_construct_logit_penalty
            ),
            body_stmt_logit_bonus=args.body_stmt_logit_bonus,
            token_expand_logit_bonus=args.token_expand_logit_bonus,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = output_dir / f"solutions.rank{rank:02d}.jsonl"
    metrics_path = output_dir / f"metrics.rank{rank:02d}.jsonl"
    completed: set[str] = set()
    if args.resume and metrics_path.exists():
        completed = {
            row["task_id"]
            for row in read_jsonl(metrics_path)
        }
        assigned = [
            task for task in assigned if task["task_id"] not in completed
        ]
    mode = "a" if args.resume else "w"
    with solutions_path.open(mode, encoding="utf-8") as solutions, metrics_path.open(
        mode, encoding="utf-8"
    ) as metrics:
        for task in assigned:
            torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            runtime = None
            failure_process = None
            try:
                runtime_config = DecoderConfig(
                    initial_root_slots=args.initial_root_slots,
                    initial_body_slots=args.initial_body_slots,
                    initial_statement_masks=args.initial_statement_masks,
                    initial_statement_masks_shallow=(
                        args.initial_statement_masks_shallow
                    ),
                    statement_shallow_depth=args.statement_shallow_depth,
                    initial_function_header_masks=(
                        args.initial_function_header_masks
                    ),
                    initial_loop_header_masks=(
                        args.initial_loop_header_masks
                    ),
                    initial_condition_masks=args.initial_condition_masks,
                    max_canvas_tokens=args.max_canvas_tokens,
                    max_tree_depth=args.max_tree_depth,
                    max_lines_per_body=args.max_lines_per_body,
                    max_total_lines=args.max_total_lines,
                    max_tokens_per_hole=args.max_tokens_per_hole,
                    max_expansions=args.max_expansions,
                    allow_module_expand=not args.no_module_expand,
                )
                header = (
                    function_header_for_task(task)
                    if args.seed_function_signature
                    and args.dataset == "humaneval"
                    else None
                )
                runtime = (
                    DecoderRuntime.from_function_header(
                        registry,
                        header,
                        runtime_config,
                    )
                    if header is not None
                    else DecoderRuntime(registry, runtime_config)
                )
                result = sampler.generate(
                    prompt_for(args.dataset, task),
                    runtime=runtime,
                )
                elapsed = time.perf_counter() - started
                process = compute_process_metrics(result).to_dict()
                solution = result.text
                error = None
            except Exception as exc:
                elapsed = time.perf_counter() - started
                process = None
                partial = sampler.last_failure_metrics
                failure_process = {
                    **(
                        partial.to_dict()
                        if partial is not None
                        else (
                            runtime.capacity_metrics()
                            if runtime is not None
                            else {}
                        )
                    ),
                    "termination_reason": termination_reason(exc),
                }
                solution = ""
                error = f"{type(exc).__name__}: {exc}"

            solutions.write(
                json.dumps(
                    {"task_id": task["task_id"], "solution": solution}
                )
                + "\n"
            )
            metrics.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "rank": rank,
                        "runtime_config_label": args.runtime_config_label,
                        "runtime_config": {
                            "initial_root_slots": args.initial_root_slots,
                            "initial_body_slots": args.initial_body_slots,
                            "initial_statement_masks": (
                                args.initial_statement_masks
                            ),
                            "initial_statement_masks_shallow": (
                                args.initial_statement_masks_shallow
                            ),
                            "statement_shallow_depth": (
                                args.statement_shallow_depth
                            ),
                            "initial_function_header_masks": (
                                args.initial_function_header_masks
                            ),
                            "initial_loop_header_masks": (
                                args.initial_loop_header_masks
                            ),
                            "initial_condition_masks": (
                                args.initial_condition_masks
                            ),
                            "max_canvas_tokens": args.max_canvas_tokens,
                            "max_tree_depth": args.max_tree_depth,
                            "max_lines_per_body": (
                                args.max_lines_per_body
                            ),
                            "max_total_lines": args.max_total_lines,
                            "max_tokens_per_hole": (
                                args.max_tokens_per_hole
                            ),
                            "max_expansions": args.max_expansions,
                            "allow_module_expand": (
                                not args.no_module_expand
                            ),
                        },
                        "sampler_config": {
                            "max_model_calls": args.max_model_calls,
                            "transfer_tokens": args.transfer_tokens,
                            "body_construct_logit_penalty": (
                                args.body_construct_logit_penalty
                            ),
                            "body_stmt_logit_bonus": (
                                args.body_stmt_logit_bonus
                            ),
                            "token_expand_logit_bonus": (
                                args.token_expand_logit_bonus
                            ),
                        },
                        "elapsed_seconds": elapsed,
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device)
                        / 2**30,
                        "process": process,
                        "failure_process": failure_process,
                        "error": error,
                    }
                )
                + "\n"
            )
            solutions.flush()
            metrics.flush()
            print(
                {
                    "rank": rank,
                    "task_id": task["task_id"],
                    "elapsed_seconds": round(elapsed, 3),
                    "error": error,
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
