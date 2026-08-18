#!/usr/bin/env python3
"""Dream-tokenizer length statistics for normalized educational_instruct."""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import statistics
import sys
import time
import warnings
from pathlib import Path
from typing import Any, Iterable

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.errors import UnsupportedSyntaxError
from scaffold_coder.ir import iter_lines
from scaffold_coder.parser import parse_source
from scaffold_coder.renderer import render_module


TOKENIZER = None


def init_worker(model_path: str) -> None:
    global TOKENIZER
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    TOKENIZER = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )


def token_length(text: str) -> int:
    return len(TOKENIZER.encode(text, add_special_tokens=False))


def analyze_one(item: tuple[int, str, str]) -> dict[str, Any]:
    seq_id, instruction, code = item
    result: dict[str, Any] = {
        "seq_id": seq_id,
        "raw_code_tokens": token_length(code),
    }
    try:
        module = parse_source(code)
    except UnsupportedSyntaxError:
        result["v0_ok"] = False
        return result

    normalized = render_module(module)
    prompt_text = TOKENIZER.apply_chat_template(
        [{"role": "user", "content": instruction}],
        add_generation_prompt=True,
        tokenize=False,
    )
    prompt_tokens = token_length(prompt_text)
    code_tokens = token_length(normalized)
    result.update(
        {
            "v0_ok": True,
            "normalized_code_tokens": code_tokens,
            "prompt_tokens": prompt_tokens,
            "prompt_plus_code_tokens": prompt_tokens + code_tokens + 1,
            "ir_line_count": sum(1 for _ in iter_lines(module.body)),
            "top_level_lines": len(module.body.lines),
        }
    )
    return result


def iter_parquet(path: str) -> Iterable[tuple[int, str, str]]:
    parquet = pq.ParquetFile(path)
    columns = ["seq_id", "instruction", "code"]
    for batch in parquet.iter_batches(columns=columns, batch_size=1024):
        ids = batch.column("seq_id").to_pylist()
        instructions = batch.column("instruction").to_pylist()
        codes = batch.column("code").to_pylist()
        yield from zip(ids, instructions, codes, strict=True)


def percentile(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def distribution(values: list[int]) -> dict[str, Any]:
    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": percentile(values, 0.50),
        "p90": percentile(values, 0.90),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--chunksize", type=int, default=32)
    args = parser.parse_args()

    total_hint = pq.ParquetFile(args.parquet).metadata.num_rows
    values: dict[str, list[int]] = {
        "raw_code_tokens": [],
        "normalized_code_tokens": [],
        "prompt_tokens": [],
        "prompt_plus_code_tokens": [],
        "ir_line_count": [],
        "top_level_lines": [],
    }
    total = accepted = 0
    started = time.time()
    context_thresholds = [512, 1024, 2048, 4000, 8192, 32768]
    exceeds = {str(threshold): 0 for threshold in context_thresholds}

    context = mp.get_context("fork")
    with context.Pool(
        args.workers,
        initializer=init_worker,
        initargs=(args.model_path,),
    ) as pool:
        results = pool.imap_unordered(
            analyze_one,
            iter_parquet(args.parquet),
            chunksize=args.chunksize,
        )
        for result in results:
            total += 1
            values["raw_code_tokens"].append(result["raw_code_tokens"])
            if result.get("v0_ok"):
                accepted += 1
                for key in values:
                    if key != "raw_code_tokens":
                        values[key].append(result[key])
                combined = result["prompt_plus_code_tokens"]
                for threshold in context_thresholds:
                    exceeds[str(threshold)] += int(combined > threshold)
            if total % 10_000 == 0:
                elapsed = time.time() - started
                print(
                    f"processed={total}/{total_hint} "
                    f"rate={total/elapsed:.1f}/s accepted={accepted}",
                    file=sys.stderr,
                    flush=True,
                )

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "total": total,
        "v0_accepted": accepted,
        "workers": args.workers,
        "distributions": {
            key: distribution(value) for key, value in values.items()
        },
        "prompt_plus_code_exceeds": {
            threshold: {
                "count": count,
                "fraction_of_v0": count / accepted if accepted else None,
            }
            for threshold, count in exceeds.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
