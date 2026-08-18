#!/usr/bin/env python3
"""Validate every cached IR and split invariant in a normalized dataset."""

from __future__ import annotations

import argparse
import ast
import json
import time
from pathlib import Path

import pyarrow.parquet as pq

from scaffold_coder.renderer import render_module
from scaffold_coder.serialization import module_from_dict


def iter_rows(path: Path):
    parquet = pq.ParquetFile(path)
    columns = ["seq_id", "response", "ir_json", "total_tokens"]
    for batch in parquet.iter_batches(columns=columns, batch_size=2048):
        yield from batch.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    started = time.time()
    split_ids: dict[str, set[int]] = {}
    stats = {}
    for split in ("train", "eval"):
        path = data_dir / f"{split}_data.parquet"
        ids: set[int] = set()
        rows = 0
        max_tokens = 0
        for row in iter_rows(path):
            rows += 1
            seq_id = int(row["seq_id"])
            if seq_id in ids:
                raise ValueError(f"duplicate seq_id {seq_id} within {split}")
            ids.add(seq_id)
            module = module_from_dict(json.loads(row["ir_json"]))
            rendered = render_module(module)
            if rendered != row["response"]:
                raise ValueError(
                    f"IR/render mismatch split={split} seq_id={seq_id}"
                )
            ast.parse(rendered)
            total_tokens = int(row["total_tokens"])
            max_tokens = max(max_tokens, total_tokens)
            if total_tokens > 1024:
                raise ValueError(
                    f"length filter violation split={split} seq_id={seq_id} "
                    f"tokens={total_tokens}"
                )
        split_ids[split] = ids
        stats[split] = {"rows": rows, "max_total_tokens": max_tokens}

    overlap = split_ids["train"].intersection(split_ids["eval"])
    if overlap:
        raise ValueError(f"train/eval overlap: {len(overlap)} IDs")

    report = {
        "data_dir": str(data_dir.resolve()),
        "stats": stats,
        "train_eval_overlap": 0,
        "validated_rows": sum(item["rows"] for item in stats.values()),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

