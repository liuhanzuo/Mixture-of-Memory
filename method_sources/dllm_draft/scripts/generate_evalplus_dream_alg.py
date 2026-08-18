#!/usr/bin/env python3
"""Eight-shard fixed-length Dream-Coder EvalPlus generation baseline."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from scaffold_coder.tokenizer_utils import (
    edit_source_token_ids,
    extend_tokenizer,
)


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


def raw_base_prompt(task: dict, bos_token: str | None = None) -> str:
    prompt = str(task["prompt"])
    return (bos_token or "") + prompt


def combine_base_continuation(task: dict, continuation: str) -> str:
    return extract_python(str(task["prompt"]) + continuation)


def extract_python(text: str) -> str:
    fences = re.findall(
        r"```(?:python)?\s*\n?(.*?)```",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fences:
        text = max(fences, key=len)
    else:
        unclosed = re.search(
            r"```(?:python)?\s*\n?(.*)$",
            text,
            flags=re.DOTALL | re.IGNORECASE,
        )
        if unclosed:
            text = unclosed.group(1)
    text = text.strip()
    starts = [
        match.start()
        for match in re.finditer(
            r"(?m)^(?:async\s+def|def|from|import|@)\s*",
            text,
        )
    ]
    if starts:
        text = text[min(starts) :]
    return text.rstrip() + ("\n" if text else "")


def parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def suppress_token_logits(
    logits: torch.Tensor,
    token_ids: tuple[int, ...],
) -> torch.Tensor:
    if token_ids:
        logits[..., list(token_ids)] = torch.finfo(logits.dtype).min
    return logits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument(
        "--alg",
        default="entropy",
        choices=("origin", "maskgit_plus", "topk_margin", "entropy"),
        help="Dream token-selection (unmasking-order) strategy.",
    )
    parser.add_argument(
        "--alg-temp",
        type=float,
        default=0.0,
        help="Temperature over the confidence ranking; 0 = deterministic top-k.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Base torch seed; rank is added so shards differ.",
    )
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Skip chat template; feed raw prompt (for base LMs).",
    )
    parser.add_argument(
        "--add-bos-token",
        action="store_true",
        help="Prepend the tokenizer BOS token, matching Dream-Coder Base eval.",
    )
    parser.add_argument(
        "--base-continuation",
        action="store_true",
        help=(
            "Use the benchmark prompt as raw code prefix and append the "
            "generated continuation before extraction."
        ),
    )
    parser.add_argument(
        "--suppress-scaffold-tokens",
        action="store_true",
        help="Mask all Scaffold meta/edit token logits during vanilla decode.",
    )
    parser.add_argument("--resume", action="store_true")
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
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if args.seed is not None:
        torch.manual_seed(args.seed + rank)
        torch.cuda.manual_seed_all(args.seed + rank)
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    suppressed_ids: tuple[int, ...] = ()
    if args.suppress_scaffold_tokens:
        suppressed_ids = tuple(sorted(set(
            extension.token_id for extension in extend_tokenizer(tokenizer)
        ) | set(edit_source_token_ids(tokenizer))))

    def suppress_scaffold_logits(step, x, logits):
        return suppress_token_logits(logits, suppressed_ids)

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
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            raw = ""
            solution = ""
            error = None
            process = None
            try:
                if args.base_continuation:
                    prompt = raw_base_prompt(
                        task,
                        tokenizer.bos_token if args.add_bos_token else None,
                    )
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        return_attention_mask=True,
                        add_special_tokens=False,
                    )
                elif args.no_chat:
                    prompt = prompt_for(args.dataset, task)
                    if args.add_bos_token:
                        prompt = tokenizer.bos_token + prompt
                    inputs = tokenizer(
                        prompt,
                        return_tensors="pt",
                        return_attention_mask=True,
                        add_special_tokens=False,
                    )
                else:
                    inputs = tokenizer.apply_chat_template(
                        [
                            {
                                "role": "user",
                                "content": prompt_for(args.dataset, task),
                            }
                        ],
                        return_tensors="pt",
                        return_dict=True,
                        add_generation_prompt=True,
                    )
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                with torch.inference_mode():
                    output = model.diffusion_generate(
                        input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=args.max_new_tokens,
                        output_history=False,
                        return_dict_in_generate=True,
                        steps=args.steps,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        alg=args.alg,
                        alg_temp=args.alg_temp,
                        generation_logits_hook_func=(
                            suppress_scaffold_logits
                        ),
                    )
                generated_ids = output.sequences[
                    0, input_ids.shape[1] :
                ].tolist()
                raw = tokenizer.decode(
                    generated_ids,
                    skip_special_tokens=True,
                )
                solution = (
                    combine_base_continuation(task, raw)
                    if args.base_continuation
                    else extract_python(raw)
                )
                process = {
                    "final_parseable": parseable(solution),
                    "nfe": args.steps,
                    "generated_tokens": len(generated_ids),
                    "input_tokens": int(input_ids.numel()),
                }
            except Exception as exc:
                error = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

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
                        "sampler": {
                            "alg": args.alg,
                            "alg_temp": args.alg_temp,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "steps": args.steps,
                            "max_new_tokens": args.max_new_tokens,
                            "seed": args.seed,
                        },
                        "elapsed_seconds": elapsed,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        ),
                        "process": process,
                        "suppressed_scaffold_token_ids": (
                            list(suppressed_ids)
                        ),
                        "raw_output": raw,
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
