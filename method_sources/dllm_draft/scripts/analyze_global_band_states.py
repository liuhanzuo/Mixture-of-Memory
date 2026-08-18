#!/usr/bin/env python3
"""Validate the one-global-t / per-depth-band state sampler."""

from __future__ import annotations

import argparse
import collections
import json
import math
import random
import statistics
import time
from pathlib import Path

import pyarrow.parquet as pq
from transformers import AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.corruption import GlobalBandSampler
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
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    sampler = GlobalBandSampler(registry)

    rung_counts: collections.Counter[str] = collections.Counter()
    depth_counts: collections.Counter[str] = collections.Counter()
    target_counts: collections.Counter[str] = collections.Counter()
    role_counts: collections.Counter[str] = collections.Counter()
    t_bins: collections.Counter[str] = collections.Counter()
    phase_bins: dict[str, collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    lengths: list[float] = []
    masks: list[float] = []
    local_u: list[float] = []
    base_weights: list[float] = []
    sample_weight_sums: list[float] = []
    rows = samples = 0
    started = time.time()

    parquet = pq.ParquetFile(args.parquet)
    columns = ["seq_id", "prompt", "ir_json"]
    for batch in parquet.iter_batches(columns=columns, batch_size=128):
        for row in batch.to_pylist():
            rows += 1
            module = module_from_dict(json.loads(row["ir_json"]))
            for draw_index in range(args.samples_per_row):
                seed = int(row["seq_id"]) * 1_000_003 + draw_index
                t = random.Random(seed ^ 0x5A17).random()
                sampled = sampler.sample(
                    module, row["prompt"], seed=seed, t=t
                )
                sampled.state.validate(registry)
                if not any(sampled.state.loss_mask):
                    raise ValueError(
                        f"no supervised mask seq_id={row['seq_id']} t={t}"
                    )
                if not all(
                    math.isfinite(weight) and weight >= 0
                    for weight in sampled.loss_weights
                ):
                    raise ValueError("invalid loss weight")

                samples += 1
                rung_counts[sampled.rung.value] += 1
                depth = str(sampled.metadata["selected_depth"])
                depth_counts[depth] += 1
                target_counts.update(sampled.metadata["target_counts"])
                role_counts.update(sampled.metadata["role_counts"])
                bin_index = min(9, int(t * 10))
                bin_name = f"{bin_index / 10:.1f}-{(bin_index + 1) / 10:.1f}"
                t_bins[bin_name] += 1
                phase_bins[bin_name][sampled.rung.value] += 1
                lengths.append(len(sampled.state.input_ids))
                masks.append(sum(sampled.state.loss_mask))
                local_u.append(sampled.local_u)
                base_weights.append(sampled.metadata["base_weight"])
                sample_weight_sums.append(sum(sampled.loss_weights))

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "rows": rows,
        "samples": samples,
        "rung_counts": dict(rung_counts),
        "selected_depth_counts": dict(
            sorted(depth_counts.items(), key=lambda item: int(item[0]))
        ),
        "target_counts": dict(target_counts.most_common()),
        "role_counts": dict(role_counts.most_common()),
        "t_bins": dict(sorted(t_bins.items())),
        "phase_by_t_bin": {
            key: dict(value) for key, value in sorted(phase_bins.items())
        },
        "lengths": distribution(lengths),
        "masks": distribution(masks),
        "local_u": distribution(local_u),
        "base_weights": distribution(base_weights),
        "sample_weight_sums": distribution(sample_weight_sums),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

