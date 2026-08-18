#!/usr/bin/env python3
"""Eight-shard autoregressive (AR) EvalPlus generation baseline.

Protocol twin of ``scripts/generate_evalplus_dream.py``. Every knob that
affects *what is graded* (prompt construction, python extraction, task
sharding, output file contract) is imported from that module rather than
re-implemented, so the AR row and the diffusion rows are scored by an
identical downstream pipeline (``merge_evalplus_shards.py`` ->
``evalplus.evaluate``).

The only intentional differences from the diffusion script are:

* decoding uses HuggingFace ``model.generate()`` with a KV cache
  (``Qwen2ForCausalLM`` / any ``AutoModelForCausalLM``), instead of
  ``model.diffusion_generate()``;
* stop strings follow the official EvalPlus HF-provider convention
  (``evalplus.provider.utility.EOS`` plus the dataset-specific
  direct-completion extras), because an AR base LM does not stop on its own
  inside a fixed-length canvas the way a diffusion model does;
* the per-task metrics row carries an explicit compute-cost block (below).

=============================================================================
COST ACCOUNTING (read this before comparing AR against diffusion numbers)
=============================================================================
Let

  P = ``input_tokens``     = number of prompt tokens actually fed to the model
  G = ``generated_tokens`` = number of new tokens returned by ``generate()``
  L = diffusion canvas length (``max_new_tokens``)
  S = diffusion step count (``--steps``, i.e. NFE)

Both cost quantities below are *measured*, not derived: a
``forward_pre_hook`` on the top-level model records, for every single forward
invocation, how many token positions were newly fed and what the resulting
total sequence length was. Analytic predictions are also emitted so the two
can be cross-checked (see ``cost.analytic`` and ``cost.consistent``).

``forward_passes``
    Number of times the model's ``forward`` was invoked during generation.
    AR with a KV cache: 1 prefill + (G-1) single-token decode steps = max(1, G).
    Diffusion: S (one full-canvas denoising pass per step).

``tokens_fed``
    Sum over forward passes of the number of token positions *newly handed to
    the model* on that pass (i.e. the ``input_ids`` width of that call).
    AR with a KV cache: P + (G-1)  -- the prompt is embedded once and each
        subsequent token is fed exactly once, because the KV cache makes
        re-feeding the prefix unnecessary.
    Diffusion: S * (P + L)  -- every step re-feeds the entire canvas.
    Interpretation: this is the "how much text did the model have to ingest"
    axis. It is the quantity on which AR's KV cache wins by orders of
    magnitude, and it is *not* a proxy for FLOPs.

``attended_context_sum``
    Sum over forward passes of the *total sequence length attended on that
    pass*, i.e. (cached prefix length + newly fed length).
    AR with a KV cache: P + sum_{i=1}^{G-1} (P + i) = G*P + G*(G-1)/2.
        The prefill pass contributes P once; decode step i attends the
        prompt plus the i tokens emitted so far.
    Diffusion: S * (P + L)  -- identical to ``tokens_fed``, because a
        diffusion step feeds and attends the same full canvas.
    Interpretation: this is the "how much context did attention have to reach
    over" axis, and it is the closer analogue of attention cost. It grows
    quadratically in G for AR, so it is the axis on which a diffusion model
    with S << G can in principle be competitive.

    CAVEAT, stated explicitly because it is easy to over-read: this counts
    the sequence length **once per forward pass**, not the number of
    query-key pairs. The AR prefill pass over P tokens contributes P, not
    P^2/2, exactly as a diffusion step over a canvas of P+L contributes P+L
    rather than (P+L)^2. So the metric is an apples-to-apples per-pass
    context-length integral for both families; it is deliberately *not* a
    FLOP count. Downstream analysis that wants pairwise attention cost should
    recompute it from the raw components, which are all preserved:
    ``prefill_tokens``, ``decode_steps``, ``per_pass_new_tokens`` and
    ``per_pass_attended`` histograms are in the metrics row.

Neither quantity is claimed to be "the" cost. They are reported side by side
precisely because AR and diffusion trade off differently on the two, and the
choice of denominator belongs to the analysis, not to the generation script.
=============================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Protocol-critical helpers are imported, never re-implemented, so the AR row
# cannot silently drift from the diffusion rows it is meant to be compared to.
from generate_evalplus_dream import (
    combine_base_continuation,
    extract_python,
    parseable,
    prompt_for,
    raw_base_prompt,
    read_jsonl,
)
from evalplus.provider.utility import EOS, extra_eos_for_direct_completion


# Cost instrumentation now lives in forward_cost.py so that arms on nodes
# without the scaffold_coder package can import it too. Re-exported here so
# existing callers keep working; there is still exactly ONE implementation.
from forward_cost import ForwardCostTracker, analytic_cost  # noqa: F401


def truncate_at_eos(text: str, stop_strings: list[str]) -> tuple[str, str | None]:
    """EvalPlus HF-provider truncation: cut at the earliest stop string.

    Mirrors ``evalplus/provider/hf.py`` so the graded text matches what the
    official EvalPlus HF backend would have graded.
    """
    min_index = len(text)
    hit = None
    for stop in stop_strings:
        idx = text.find(stop)
        if idx != -1 and idx < min_index:
            min_index = idx
            hit = stop
    return text[:min_index].replace("\t", "    "), hit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--no-chat",
        action="store_true",
        help="Skip chat template; feed raw prompt (for base LMs).",
    )
    parser.add_argument(
        "--add-bos-token",
        action="store_true",
        help="Prepend the tokenizer BOS token (matches Dream-Coder Base eval).",
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
        "--no-stop-strings",
        action="store_true",
        help="Disable EvalPlus stop strings (generate the full budget).",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    # NOTE: when each shard is launched with CUDA_VISIBLE_DEVICES=<one gpu>,
    # torch only ever sees device 0, so LOCAL_RANK must be 0 while RANK stays
    # the logical shard id. Defaulting LOCAL_RANK to RANK (as the diffusion
    # script does for torchrun) would raise "invalid device ordinal" for
    # shards 1..7 under the per-shard launch style.
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
    model = (
        AutoModelForCausalLM.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    model.config.use_cache = True

    tracker = ForwardCostTracker()
    model.register_forward_pre_hook(tracker.hook, with_kwargs=True)

    direct_completion = bool(args.base_continuation)
    if args.no_stop_strings:
        stop_strings: list[str] = []
    elif direct_completion:
        stop_strings = list(EOS) + extra_eos_for_direct_completion(args.dataset)
    else:
        stop_strings = list(EOS) + ["\n```\n"]

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    do_sample = args.temperature > 0.0

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = output_dir / f"solutions.rank{rank:02d}.jsonl"
    metrics_path = output_dir / f"metrics.rank{rank:02d}.jsonl"
    if args.resume and metrics_path.exists():
        completed = {row["task_id"] for row in read_jsonl(metrics_path)}
        assigned = [task for task in assigned if task["task_id"] not in completed]
    mode = "a" if args.resume else "w"

    protocol = {
        "decoder": "autoregressive_kv_cache",
        "checkpoint": checkpoint,
        "dataset": args.dataset,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": do_sample,
        "seed": args.seed,
        "no_chat": bool(args.no_chat),
        "add_bos_token": bool(args.add_bos_token),
        "base_continuation": bool(args.base_continuation),
        "direct_completion": direct_completion,
        "stop_strings": stop_strings,
        "world_size": world_size,
    }
    if rank == 0:
        (output_dir / "protocol.json").write_text(
            json.dumps(protocol, indent=2, sort_keys=True) + "\n"
        )
    print({"rank": rank, "protocol": protocol}, flush=True)

    with solutions_path.open(mode, encoding="utf-8") as solutions, metrics_path.open(
        mode, encoding="utf-8"
    ) as metrics:
        for task in assigned:
            torch.cuda.reset_peak_memory_stats(device)
            tracker.reset()
            torch.manual_seed(args.seed)
            started = time.perf_counter()
            raw = ""
            truncated = ""
            solution = ""
            error = None
            process = None
            stop_hit = None
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
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)
                generate_kwargs = dict(
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=do_sample,
                    num_return_sequences=1,
                    pad_token_id=pad_token_id,
                    use_cache=True,
                )
                if do_sample:
                    generate_kwargs["temperature"] = args.temperature
                    generate_kwargs["top_p"] = args.top_p
                if stop_strings:
                    generate_kwargs["stop_strings"] = stop_strings
                    generate_kwargs["tokenizer"] = tokenizer

                tracker.enabled = True
                with torch.inference_mode():
                    output = model.generate(input_ids, **generate_kwargs)
                tracker.enabled = False

                generated_ids = output[0, input_ids.shape[1] :].tolist()
                raw = tokenizer.decode(generated_ids, skip_special_tokens=True)
                truncated, stop_hit = truncate_at_eos(raw, stop_strings)
                solution = (
                    combine_base_continuation(task, truncated)
                    if args.base_continuation
                    else extract_python(truncated)
                )
                measured = tracker.summary()
                prompt_tokens = int(input_ids.shape[1])
                gen_tokens = len(generated_ids)
                predicted = analytic_cost(prompt_tokens, gen_tokens)
                process = {
                    "final_parseable": parseable(solution),
                    "input_tokens": prompt_tokens,
                    "generated_tokens": gen_tokens,
                    "stop_string_hit": stop_hit,
                    "cost": {
                        # ---- the two headline cost quantities (measured) ----
                        "tokens_fed": measured["tokens_fed"],
                        "attended_context_sum": measured["attended_context_sum"],
                        # ---- raw components, so downstream can re-derive ----
                        "forward_passes": measured["forward_passes"],
                        "prefill_tokens": prompt_tokens,
                        "decode_steps": max(0, gen_tokens - 1),
                        "per_pass_new_tokens": measured["per_pass_new_tokens"],
                        "per_pass_attended": measured["per_pass_attended"],
                        "analytic": predicted,
                        "consistent": (
                            measured["forward_passes"] == predicted["forward_passes"]
                            and measured["tokens_fed"] == predicted["tokens_fed"]
                            and measured["attended_context_sum"]
                            == predicted["attended_context_sum"]
                        ),
                    },
                }
            except Exception as exc:  # noqa: BLE001
                tracker.enabled = False
                error = f"{type(exc).__name__}: {exc}"
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - started

            solutions.write(
                json.dumps({"task_id": task["task_id"], "solution": solution}) + "\n"
            )
            metrics.write(
                json.dumps(
                    {
                        "task_id": task["task_id"],
                        "rank": rank,
                        "elapsed_seconds": elapsed,
                        "wall_clock_seconds": elapsed,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        ),
                        "process": process,
                        "raw_output": raw,
                        "truncated_output": truncated,
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
                    "generated_tokens": (
                        process["generated_tokens"] if process else None
                    ),
                    "tokens_fed": (
                        process["cost"]["tokens_fed"] if process else None
                    ),
                    "attended_context_sum": (
                        process["cost"]["attended_context_sum"] if process else None
                    ),
                    "cost_consistent": (
                        process["cost"]["consistent"] if process else None
                    ),
                    "error": error,
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
