#!/usr/bin/env python3
"""Validate deterministic canvas rungs over a normalized parquet split."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.canvas import (
    TokenRegistry,
    build_body_plan,
    build_leaf_infill,
    build_root_plan,
    build_template_skeleton,
    iter_main_bodies,
    prepend_chat_prompt,
)
from scaffold_coder.serialization import module_from_dict


def distribution(values: list[int]) -> dict:
    ordered = sorted(values)

    def p(fraction: float):
        return ordered[round(fraction * (len(ordered) - 1))] if ordered else None

    return {
        "count": len(values),
        "mean": statistics.fmean(values) if values else None,
        "p50": p(0.50),
        "p90": p(0.90),
        "p95": p(0.95),
        "p99": p(0.99),
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    parquet = pq.ParquetFile(args.parquet)
    columns = ["seq_id", "prompt", "response", "ir_json"]
    lengths = {
        "clean_leaf": [],
        "root_plan": [],
        "template_skeleton": [],
        "half_masked_leaf": [],
        "chat_root_plan": [],
        "body_plan": [],
    }
    mask_counts = {
        "root_plan": [],
        "half_masked_leaf": [],
        "body_plan": [],
    }
    rows = body_states = 0
    started = time.time()

    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            rows += 1
            module = module_from_dict(json.loads(row["ir_json"]))

            clean = build_leaf_infill(
                module, registry, mask_probability=0.0, seed=row["seq_id"]
            )
            if tokenizer.decode(clean.input_ids) != row["response"]:
                raise ValueError(
                    f"segmented clean canvas mismatch seq_id={row['seq_id']}"
                )

            root = build_root_plan(module, registry)
            skeleton = build_template_skeleton(module, registry)
            masked = build_leaf_infill(
                module, registry, mask_probability=0.5, seed=row["seq_id"]
            )
            chat_root = prepend_chat_prompt(root, registry, row["prompt"])

            lengths["clean_leaf"].append(len(clean.input_ids))
            lengths["root_plan"].append(len(root.input_ids))
            lengths["template_skeleton"].append(len(skeleton.input_ids))
            lengths["half_masked_leaf"].append(len(masked.input_ids))
            lengths["chat_root_plan"].append(len(chat_root.input_ids))
            mask_counts["root_plan"].append(sum(root.loss_mask))
            mask_counts["half_masked_leaf"].append(sum(masked.loss_mask))

            for body in iter_main_bodies(module):
                body_state = build_body_plan(
                    module, registry, target_body_id=body.body_id
                )
                body_states += 1
                lengths["body_plan"].append(len(body_state.input_ids))
                mask_counts["body_plan"].append(sum(body_state.loss_mask))

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "rows": rows,
        "body_states": body_states,
        "lengths": {key: distribution(value) for key, value in lengths.items()},
        "mask_counts": {
            key: distribution(value) for key, value in mask_counts.items()
        },
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

