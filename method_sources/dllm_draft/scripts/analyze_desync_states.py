#!/usr/bin/env python3
"""Measure cross-subtree state mixing under per-top-level clock offsets."""

from __future__ import annotations

import argparse
import collections
import json
import random
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.desync import (
    DesyncConfig,
    DesynchronizedGlobalSampler,
)
from scaffold_coder.serialization import module_from_dict


def distribution(values):
    ordered = sorted(values)

    def p(fraction):
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
    parser.add_argument("--sigma-d", type=float, default=0.1)
    parser.add_argument("--samples-per-row", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    sampler = DesynchronizedGlobalSampler(
        registry, config=DesyncConfig(sigma_d=args.sigma_d)
    )
    rows = multi_rows = samples = mixed_rung_samples = 0
    subtree_rungs: collections.Counter[str] = collections.Counter()
    rung_combinations: collections.Counter[str] = collections.Counter()
    lengths = []
    masks = []
    max_abs_offsets = []
    started = time.time()

    parquet = pq.ParquetFile(args.parquet)
    columns = ["seq_id", "prompt", "ir_json"]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            rows += 1
            module = module_from_dict(json.loads(row["ir_json"]))
            if len(module.body.lines) <= 1:
                continue
            multi_rows += 1
            for draw in range(args.samples_per_row):
                seed = int(row["seq_id"]) * 1_000_003 + draw
                t = random.Random(seed ^ 0xD35A).random()
                sampled = sampler.sample(
                    module, row["prompt"], seed=seed, t=t
                )
                sampled.state.validate(registry)
                samples += 1
                rungs = [item["rung"] for item in sampled.metadata["subtrees"]]
                subtree_rungs.update(rungs)
                combination = "+".join(sorted(rungs))
                rung_combinations[combination] += 1
                mixed_rung_samples += int(len(set(rungs)) > 1)
                lengths.append(len(sampled.state.input_ids))
                masks.append(sum(sampled.state.loss_mask))
                max_abs_offsets.append(
                    max(abs(value) for value in sampled.metadata["desync_offsets"])
                )

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "sigma_d": args.sigma_d,
        "rows": rows,
        "multi_top_level_rows": multi_rows,
        "multi_top_level_fraction": multi_rows / rows if rows else None,
        "samples": samples,
        "mixed_rung_samples": mixed_rung_samples,
        "mixed_rung_fraction": mixed_rung_samples / samples if samples else None,
        "subtree_rungs": dict(subtree_rungs),
        "rung_combinations": dict(rung_combinations.most_common()),
        "lengths": distribution(lengths),
        "masks": distribution(masks),
        "max_abs_offsets": distribution(max_abs_offsets),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

