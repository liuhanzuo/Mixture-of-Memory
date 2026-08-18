#!/usr/bin/env python3
"""Build canonical Scaffold-Coder train/eval parquet files with cached IR."""

from __future__ import annotations

import argparse
import json
import random
import time
import warnings
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.errors import UnsupportedSyntaxError
from scaffold_coder.parser import parse_source
from scaffold_coder.renderer import render_module
from scaffold_coder.serialization import module_to_dict


def main() -> None:
    warnings.filterwarnings("ignore", category=SyntaxWarning)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )

    accepted: list[dict] = []
    rejected: list[dict] = []
    total = 0
    started = time.time()
    parquet = pq.ParquetFile(args.input)
    columns = [
        "seq_id",
        "instruction",
        "code",
        "entry_point",
        "testcase",
    ]
    for batch in parquet.iter_batches(columns=columns, batch_size=1024):
        for row in batch.to_pylist():
            total += 1
            try:
                module = parse_source(row["code"])
                response = render_module(module)
                prompt_text = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["instruction"]}],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                prompt_tokens = len(
                    tokenizer.encode(prompt_text, add_special_tokens=False)
                )
                response_tokens = len(
                    tokenizer.encode(response, add_special_tokens=False)
                )
                total_tokens = prompt_tokens + response_tokens + 1
                if total_tokens > args.max_length:
                    rejected.append(
                        {
                            "seq_id": row["seq_id"],
                            "reason": "length",
                            "detail": str(total_tokens),
                        }
                    )
                    continue
                accepted.append(
                    {
                        "seq_id": row["seq_id"],
                        "prompt": row["instruction"],
                        "response": response,
                        "code": response,
                        "entry_point": row["entry_point"],
                        "testcase": row["testcase"],
                        "ir_json": json.dumps(
                            module_to_dict(module),
                            separators=(",", ":"),
                            sort_keys=True,
                        ),
                        "prompt_tokens": prompt_tokens,
                        "response_tokens": response_tokens,
                        "total_tokens": total_tokens,
                    }
                )
            except UnsupportedSyntaxError as exc:
                rejected.append(
                    {
                        "seq_id": row["seq_id"],
                        "reason": "unsupported",
                        "detail": str(exc),
                    }
                )
            except Exception as exc:
                rejected.append(
                    {
                        "seq_id": row["seq_id"],
                        "reason": f"internal:{type(exc).__name__}",
                        "detail": str(exc)[:500],
                    }
                )
            if total % 10_000 == 0:
                print(
                    f"processed={total} accepted={len(accepted)} "
                    f"rejected={len(rejected)}",
                    flush=True,
                )

    rng = random.Random(args.seed)
    rng.shuffle(accepted)
    eval_size = min(args.eval_size, len(accepted))
    eval_rows = accepted[:eval_size]
    train_rows = accepted[eval_size:]

    pd.DataFrame(train_rows).to_parquet(
        output_dir / "train_data.parquet", index=False
    )
    pd.DataFrame(eval_rows).to_parquet(
        output_dir / "eval_data.parquet", index=False
    )
    pd.DataFrame(rejected).to_parquet(
        output_dir / "rejected_data.parquet", index=False
    )

    manifest = {
        "input": str(Path(args.input).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "seed": args.seed,
        "max_length": args.max_length,
        "total": total,
        "accepted": len(accepted),
        "train": len(train_rows),
        "eval": len(eval_rows),
        "rejected": len(rejected),
        "elapsed_seconds": time.time() - started,
        "files": {
            "train": "train_data.parquet",
            "eval": "eval_data.parquet",
            "rejected": "rejected_data.parquet",
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
