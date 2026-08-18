#!/usr/bin/env python3
"""Eight-shard DreamOn variable-length EvalPlus generation baseline.

⚠️ 2026-08-12 (A05 closeout) -- three defects in this driver are now fixed or
annotated in place. Read before reusing any number this script has produced.

1. ``nfe`` was ``len(output.history)``, which is NOT a forward-pass count (the
   model appends to ``histories`` at three separate sites). Field renamed to
   ``history_len_NOT_nfe``. Archived r1 "NFE 265.88/135.65" are void; true
   counted NFE at the same setting is 172.3/153.4.
2. ``mask_expansion=True`` / ``delete_eos_token=True`` were passed but are not
   parameters -- confirmed by execution. Removed.
3. ``combine_humaneval_prompt`` double-indented already-indented bodies, which
   UNDERSTATED every HE+ number this script produced (worse at larger canvases:
   pass@1 plus .1707 -> .4817 at canvas=128). Fixed.

Also note ``--initial-masks`` defaults to 4 here while the archived runs used 8;
the canvas is the dominant driver of DreamOn's score (.085 -> .3545 on MBPP+
going from 8 to 32), so it must always be reported alongside any pass@1.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import textwrap
import time
from pathlib import Path

import torch
from transformers import AutoModel, AutoTokenizer

from generate_evalplus_dream import extract_python, prompt_for, read_jsonl


def parseable(text: str) -> bool:
    try:
        ast.parse(text)
        return True
    except SyntaxError:
        return False


def combine_humaneval_prompt(prompt: str, generated: str) -> str:
    """Stitch a bare function body under the HumanEval prompt.

    FIXED 2026-08-12 (A05 closeout). The previous implementation was::

        extracted = extract_python(generated)
        if re.search(r"(?m)^(?:async\\s+def|def)\\s+", extracted):
            return extracted
        body = extracted.strip() or "pass"
        return prompt.rstrip() + "\\n" + textwrap.indent(body, "    ") + "\\n"

    ``extract_python`` ends in ``.strip()``, which removes leading whitespace from
    the FIRST line only. DreamOn emits an already-4-space-indented function body,
    so after extraction line 1 sat at column 0 while lines 2..n kept their
    original depth; ``textwrap.indent`` then shifted everything by 4, putting
    line 1 at 4 and line 2 at 8 -> ``IndentationError: unexpected indent``.

    Measured impact (A05 K1 cells, re-graded with evalplus, generation unchanged):
    HE+ parseability .287 -> .963 and pass@1 plus .1707 -> .4817 at canvas=128;
    .860 -> .982 and .2134 -> .2561 at canvas=32; .988 -> 1.000 and .1280 -> .1341
    at canvas=8 (the archived setting, so the archived HE+ .122 is understated too).
    The bug grows with canvas size, so every HE+ number this repo has ever produced
    through this function is understated, most severely at large canvases.

    NOTE the ordering trap: applying ``textwrap.dedent`` AFTER ``extract_python``
    is a NO-OP, because the first line has already been stripped and the common
    leading prefix is therefore 0. The dedent must happen BEFORE extraction.
    """
    extracted = extract_python(generated)
    if re.search(r"(?m)^(?:async\s+def|def)\s+", extracted):
        return extracted
    body = textwrap.dedent(generated.replace("\t", "    ")).strip("\n").rstrip()
    if not body.strip():
        body = "pass"
    return prompt.rstrip() + "\n" + textwrap.indent(body, "    ") + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", choices=("humaneval", "mbpp"), required=True)
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--initial-masks", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--transfer-tokens", type=int, default=1)
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
    model = AutoModel.from_pretrained(
        checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()

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
                prompt_tensor = tokenizer.apply_chat_template(
                    [
                        {
                            "role": "user",
                            "content": prompt_for(args.dataset, task),
                        }
                    ],
                    return_tensors="pt",
                    add_generation_prompt=True,
                )
                prompt_ids = prompt_tensor[0].tolist()
                initial = (
                    prompt_ids
                    + [tokenizer.mask_token_id] * args.initial_masks
                    + [tokenizer.eos_token_id]
                )
                input_ids = torch.tensor(
                    [initial],
                    device=device,
                    dtype=torch.long,
                )
                with torch.inference_mode():
                    output = model.diffusion_generate(
                        input_ids,
                        max_new_tokens=args.max_new_tokens,
                        output_history=False,
                        return_dict_in_generate=True,
                        number_transfer_tokens=args.transfer_tokens,
                        temperature=0.2,
                        top_p=0.9,
                        alg="entropy",
                        alg_temp=0.0,
                        # REMOVED 2026-08-12 (A05 closeout): mask_expansion=True and
                        # delete_eos_token=True were passed here but are NOT parameters.
                        # Verified by execution, not by reading:
                        #   DreamGenerationConfig().update(mask_expansion=True,
                        #       delete_eos_token=True)
                        #   -> returns {'mask_expansion': True, 'delete_eos_token': True}
                        #      as UNUSED, and hasattr(cfg, 'mask_expansion') is False
                        #      both before and after. (Control: update(temperature=0.2)
                        #      returns {} and assigns correctly.)
                        # They were silently swallowed by **kwargs, so every archived
                        # DreamOn number was produced WITHOUT them. Passing them
                        # misrepresents the run; removing them changes nothing
                        # behaviourally. Do not re-add.
                    )
                response_ids = output.sequences[0, len(prompt_ids) :].tolist()
                if tokenizer.eos_token_id in response_ids:
                    response_ids = response_ids[
                        : response_ids.index(tokenizer.eos_token_id)
                    ]
                raw = tokenizer.decode(
                    response_ids,
                    skip_special_tokens=True,
                )
                solution = (
                    combine_humaneval_prompt(task["prompt"], raw)
                    if args.dataset == "humaneval"
                    else extract_python(raw)
                )
                process = {
                    "final_parseable": parseable(solution),
                    # RENAMED 2026-08-12 (A05 closeout). This field used to be called
                    # "nfe", which was wrong: len(output.history) is NOT a forward-pass
                    # count. models/DreamOn-v0-7B/generation_utils.py appends to
                    # `histories` at THREE sites -- line 445 (transfer step), 476
                    # (delete batch), 495 (expand batch) -- so it can reach ~3x the
                    # number of model calls. It is also None whenever
                    # output_history=False (as here), which is why every r2 item
                    # logged nfe: null (measured: 0/164 and 0/378 non-null).
                    # The archived r1 aggregates 265.88 (HE+) / 135.65 (MBPP+) are
                    # therefore mean(len(history)), NOT NFE. True counted NFE at this
                    # same setting is 172.3 / 153.4 -- note MBPP+ moves the OTHER WAY,
                    # so this is not a uniform rescaling.
                    # To measure real NFE, wrap model.forward and count calls; see
                    # Mixture-of-Memory/proposal/archive/A05-structural-dllm-cost-frontier/
                    # code/a05_k1_dreamon_canvas.py::install_counters.
                    "history_len_NOT_nfe": (
                        len(output.history)
                        if output.history is not None
                        else None
                    ),
                    "generated_tokens": len(response_ids),
                    "initial_masks": args.initial_masks,
                    "transfer_tokens": args.transfer_tokens,
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
                        "elapsed_seconds": elapsed,
                        "peak_memory_gib": (
                            torch.cuda.max_memory_allocated(device) / 2**30
                        ),
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
                    "elapsed_seconds": round(elapsed, 3),
                    "error": error,
                },
                flush=True,
            )


if __name__ == "__main__":
    main()
