#!/usr/bin/env python3
"""A05 K1 -- DreamOn initial-mask (canvas) sweep on HumanEval+ / MBPP+.

Derived from dllm_draft/scripts/generate_evalplus_dreamon.py with four fixes that
the archived driver got wrong. Each fix is load-bearing for the K1 verdict.

FIX 1 -- true NFE.
  The archived driver recorded ``nfe = len(output.history)``. That is (a) None
  whenever ``output_history=False`` (which is why every r2 item logs
  ``nfe: null``), and (b) NOT the forward-pass count even when it is populated:
  ``_sample`` appends to ``history`` once after the transfer step, again after a
  delete batch, and again after an expand batch, so ``len(history)`` can reach
  ~3x the number of model calls. Here we wrap ``model.forward`` and count calls.

FIX 2 -- tokens_fed.
  DreamOn pads ``x`` to ``max_length`` on every forward and masks the tail via
  the attention mask, so the padded shape overstates real attended cost. We
  report BOTH: ``tokens_fed_effective`` (sum of attended window over forwards,
  the figure comparable to Scaffold's dynamically-padded cost) and
  ``tokens_fed_padded`` (NFE * max_length).

FIX 3 -- no inert kwargs.
  The archived driver passed ``mask_expansion=True`` / ``delete_eos_token=True``.
  Neither is a parameter: ``DreamGenerationConfig`` never defines them, and
  ``GenerationConfig.update`` only assigns keys that already exist, returning the
  rest as unused. They were silently dropped. Passing them would misrepresent
  the run, so they are omitted; behaviour is bit-for-bit the same.

FIX 4 -- per-item seeding.
  The archive did not seed, and temperature=0.2 makes ``sample_tokens`` draw from
  ``dists.Categorical``. We seed per task_id so that a given item sees the same
  RNG stream at every canvas setting; the canvas is then the only variable across
  settings. This does not alter the sampling distribution.

Everything else -- temperature 0.2, top_p 0.9, alg entropy, alg_temp 0.0,
number_transfer_tokens 1, max_new_tokens 512, the chat template, the prompt
wording, and the solution extraction -- is byte-identical to the archived r2 run,
so ``--canvas 8`` is a reproduction check against HE+ .122 / MBPP+ .085.

Note on what ``--canvas`` (initial_masks) actually controls: it is the STARTING
number of mask tokens, not a cap on output. ``max_length`` is
``max_new_tokens + prompt_len + 1`` independent of the canvas, and the sampler may
grow the canvas via ``<|expand|>`` up to that bound (and shrink it by emitting
the delete token). So this sweep varies the initial canvas while holding the
512-token generation ceiling fixed.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import re
import textwrap
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

HE_INSTRUCTION = (
    "Write a complete Python solution for the following function. "
    "Return only Python code and preserve the required function name.\n\n"
)
MBPP_INSTRUCTION = (
    "Write a complete Python solution for the following programming task. "
    "Return only Python code and preserve the required function name.\n\n"
)


def read_jsonl(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def prompt_for(dataset: str, task: dict) -> str:
    head = HE_INSTRUCTION if dataset == "humaneval" else MBPP_INSTRUCTION
    return head + task["prompt"]


def extract_python(text: str) -> str:
    """Verbatim from dllm_draft/scripts/generate_evalplus_dream.py."""
    fences = re.findall(r"```(?:python)?\s*\n?(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fences:
        text = max(fences, key=len)
    else:
        unclosed = re.search(r"```(?:python)?\s*\n?(.*)$", text, flags=re.DOTALL | re.IGNORECASE)
        if unclosed:
            text = unclosed.group(1)
    text = text.strip()
    starts = [
        match.start()
        for match in re.finditer(r"(?m)^(?:async\s+def|def|from|import|@)\s*", text)
    ]
    if starts:
        text = text[min(starts):]
    return text.rstrip() + ("\n" if text else "")


def combine_humaneval_prompt(prompt: str, generated: str) -> str:
    """Verbatim from dllm_draft/scripts/generate_evalplus_dreamon.py."""
    extracted = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", extracted):
        return extracted
    body = extracted.strip() or "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False
    except (ValueError, MemoryError, RecursionError):
        return False


def gold_text(dataset: str, task: dict) -> str:
    """Gold full program: HE+ needs the signature, MBPP+'s canonical is standalone."""
    if dataset == "humaneval":
        return task["prompt"] + task.get("canonical_solution", "")
    return task.get("canonical_solution", "")


def stable_seed(task_id: str) -> int:
    digest = hashlib.sha256(task_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big")


def install_counters(model):
    """Count forward passes and attended tokens. Accumulates on-device to avoid
    a host sync on every denoising step."""
    device = next(model.parameters()).device
    state = {
        "nfe": torch.zeros((), dtype=torch.long, device=device),
        "tok_eff": torch.zeros((), dtype=torch.long, device=device),
        "tok_pad": torch.zeros((), dtype=torch.long, device=device),
    }
    original = model.forward

    def counting_forward(x, attention_mask=None, tok_idx=None, *args, **kwargs):
        state["nfe"] += 1
        state["tok_pad"] += x.shape[1]
        if attention_mask is not None and attention_mask.dim() == 4:
            # attention_mask = and(am[...,None,:], am[...,:,None]) so its diagonal
            # is the 1-D per-position mask; its sum is the attended window length.
            state["tok_eff"] += attention_mask[0, 0].diagonal().sum().to(torch.long)
        else:
            state["tok_eff"] += x.shape[1]
        return original(x, attention_mask, tok_idx, *args, **kwargs)

    model.forward = counting_forward

    def reset():
        for value in state.values():
            value.zero_()

    def read():
        return {
            "nfe": int(state["nfe"].item()),
            "tokens_fed_effective": int(state["tok_eff"].item()),
            "tokens_fed_padded": int(state["tok_pad"].item()),
        }

    return reset, read


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--canvas",
        required=True,
        help="initial_masks: an integer, or 'oracle' for gold_tokens + --oracle-slack",
    )
    parser.add_argument("--oracle-slack", type=int, default=32)
    # Frozen sampler knobs -- identical to the archived r2 run. Exposed so the
    # values land in the manifest, NOT so they get swept.
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--transfer-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--alg", default="entropy")
    parser.add_argument("--alg-temp", type=float, default=0.0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    is_oracle = str(args.canvas).lower() == "oracle"
    fixed_canvas = None if is_oracle else int(args.canvas)

    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    tasks = list(read_jsonl(Path(args.data_file)))
    if args.limit is not None:
        tasks = tasks[: args.limit]
    assigned = [task for index, task in enumerate(tasks) if index % world_size == rank]

    checkpoint = str(Path(args.checkpoint).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, trust_remote_code=True, local_files_only=True
    )
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    model = (
        AutoModel.from_pretrained(
            checkpoint,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        .to(device)
        .eval()
    )
    reset_counters, read_counters = install_counters(model)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    solutions_path = output_dir / f"solutions.rank{rank:02d}.jsonl"
    metrics_path = output_dir / f"metrics.rank{rank:02d}.jsonl"
    if args.resume and metrics_path.exists():
        done = {row["task_id"] for row in read_jsonl(metrics_path)}
        assigned = [task for task in assigned if task["task_id"] not in done]
    mode = "a" if args.resume else "w"

    # max generated tokens is capped by max_length regardless of canvas size;
    # a canvas larger than that would make diffusion_generate raise.
    canvas_ceiling = args.max_new_tokens

    with solutions_path.open(mode, encoding="utf-8") as solutions, metrics_path.open(
        mode, encoding="utf-8"
    ) as metrics:
        for task in assigned:
            gold = gold_text(args.dataset, task)
            gold_tokens = len(tokenizer(gold, add_special_tokens=False)["input_ids"])
            if is_oracle:
                requested = gold_tokens + args.oracle_slack
            else:
                requested = fixed_canvas
            canvas = max(1, min(requested, canvas_ceiling))
            clamped = canvas != requested

            torch.cuda.reset_peak_memory_stats(device)
            reset_counters()
            seed = stable_seed(task["task_id"])
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            started = time.perf_counter()
            raw = ""
            solution = ""
            error = None
            process = None
            try:
                prompt_tensor = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt_for(args.dataset, task)}],
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                prompt_ids = prompt_tensor[0].tolist()
                initial = (
                    prompt_ids
                    + [tokenizer.mask_token_id] * canvas
                    + [tokenizer.eos_token_id]
                )
                input_ids = torch.tensor([initial], device=device, dtype=torch.long)
                with torch.inference_mode():
                    output = model.diffusion_generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        output_history=False,
                        return_dict_in_generate=True,
                        number_transfer_tokens=args.transfer_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        alg=args.alg,
                        alg_temp=args.alg_temp,
                    )
                response_ids = output.sequences[0, len(prompt_ids):].tolist()
                if tokenizer.eos_token_id in response_ids:
                    response_ids = response_ids[: response_ids.index(tokenizer.eos_token_id)]
                raw = tokenizer.decode(response_ids, skip_special_tokens=True)
                solution = (
                    combine_humaneval_prompt(task["prompt"], raw)
                    if args.dataset == "humaneval"
                    else extract_python(raw)
                )
                counters = read_counters()
                generated = len(response_ids)
                process = {
                    "final_parseable": parseable(solution),
                    "generated_tokens": generated,
                    "gold_tokens": gold_tokens,
                    "emitted_gold_ratio": (generated / gold_tokens) if gold_tokens else None,
                    "initial_masks": canvas,
                    "initial_masks_requested": requested,
                    "initial_masks_clamped": clamped,
                    "prompt_tokens": len(prompt_ids),
                    "transfer_tokens": args.transfer_tokens,
                    "seed": seed,
                    **counters,
                }
            except Exception as exc:  # noqa: BLE001 -- recorded per item, never fatal
                error = f"{type(exc).__name__}: {exc}"
                counters = read_counters()
                process = {
                    "final_parseable": False,
                    "generated_tokens": 0,
                    "gold_tokens": gold_tokens,
                    "emitted_gold_ratio": 0.0 if gold_tokens else None,
                    "initial_masks": canvas,
                    "initial_masks_requested": requested,
                    "initial_masks_clamped": clamped,
                    "transfer_tokens": args.transfer_tokens,
                    "seed": seed,
                    **counters,
                }
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
                        "canvas_setting": args.canvas,
                        "elapsed_seconds": elapsed,
                        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                        "process": process,
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
                    "canvas": canvas,
                    "nfe": (process or {}).get("nfe"),
                    "gen": (process or {}).get("generated_tokens"),
                    "elapsed_seconds": round(elapsed, 2),
                    "error": error,
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
