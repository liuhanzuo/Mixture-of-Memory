#!/usr/bin/env python3
"""Sample stochastic rung/edit states and report target distributions."""

from __future__ import annotations

import argparse
import collections
import json
import math
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.corruption import RungMixtureConfig, RungMixtureSampler
from scaffold_coder.serialization import module_from_dict


def distribution(values: list[float]) -> dict:
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
    parser.add_argument("--samples-per-row", type=int, default=5)
    parser.add_argument("--root-probability", type=float, default=0.20)
    parser.add_argument("--body-probability", type=float, default=0.30)
    parser.add_argument("--leaf-probability", type=float, default=0.50)
    parser.add_argument("--token-merge-probability", type=float, default=0.50)
    parser.add_argument("--line-merge-probability", type=float, default=0.50)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    config = RungMixtureConfig(
        root_probability=args.root_probability,
        body_probability=args.body_probability,
        leaf_probability=args.leaf_probability,
        token_merge_base_probability=args.token_merge_probability,
        line_merge_probability=args.line_merge_probability,
    )
    sampler = RungMixtureSampler(registry, config)

    rung_counts: collections.Counter[str] = collections.Counter()
    target_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    merge_mode_counts: collections.Counter[str] = collections.Counter()
    lengths: list[float] = []
    masks: list[float] = []
    local_u: list[float] = []
    positive_weights: list[float] = []
    sample_weight_sums: list[float] = []
    samples = rows = 0
    started = time.time()

    parquet = pq.ParquetFile(args.parquet)
    columns = ["seq_id", "prompt", "ir_json"]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            rows += 1
            module = module_from_dict(json.loads(row["ir_json"]))
            for draw_index in range(args.samples_per_row):
                seed = int(row["seq_id"]) * 1_000_003 + draw_index
                sampled = sampler.sample(module, row["prompt"], seed=seed)
                sampled.state.validate(registry)
                if not any(sampled.state.loss_mask):
                    raise ValueError(
                        f"sample has no supervised mask seq_id={row['seq_id']}"
                    )
                if not all(
                    math.isfinite(weight) and weight >= 0
                    for weight in sampled.loss_weights
                ):
                    raise ValueError("non-finite or negative loss weight")
                samples += 1
                rung_counts[sampled.rung.value] += 1
                target_counts.update(sampled.metadata["target_counts"])
                role_counts.update(sampled.metadata["role_counts"])
                if "merge_mode" in sampled.metadata:
                    merge_mode_counts[sampled.metadata["merge_mode"]] += 1
                lengths.append(len(sampled.state.input_ids))
                masks.append(sum(sampled.state.loss_mask))
                local_u.append(sampled.local_u)
                positive_weights.extend(
                    weight for weight in sampled.loss_weights if weight > 0
                )
                sample_weight_sums.append(sum(sampled.loss_weights))

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "config": {
            key: getattr(config, key)
            for key in config.__dataclass_fields__
        },
        "rows": rows,
        "samples": samples,
        "rung_counts": dict(rung_counts),
        "target_counts": dict(target_counts.most_common()),
        "role_counts": dict(role_counts.most_common()),
        "merge_mode_counts": dict(merge_mode_counts),
        "lengths": distribution(lengths),
        "masks": distribution(masks),
        "local_u": distribution(local_u),
        "positive_loss_weights": distribution(positive_weights),
        "sample_weight_sums": distribution(sample_weight_sums),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
