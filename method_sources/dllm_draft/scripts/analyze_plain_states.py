#!/usr/bin/env python3
"""Statistics for the matched plain uniform-masking SFT control."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from transformers import AutoTokenizer

from scaffold_coder.sft_dataset import PlainMaskedSFTDataset


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
    parser.add_argument("--samples-per-row", type=int, default=5)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = PlainMaskedSFTDataset(
        args.parquet,
        tokenizer,
        max_length=1024,
        training=True,
        seed=1,
    )
    lengths = []
    masks = []
    timesteps = []
    weight_sums = []
    started = time.time()
    for epoch in range(args.samples_per_row):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            item = dataset[index]
            lengths.append(int(item["length"]))
            masks.append(int(item["loss_mask"].sum()))
            timesteps.append(float(item["t"]))
            weight_sums.append(float(item["loss_weights"].sum()))

    report = {
        "parquet": str(Path(args.parquet).resolve()),
        "model_path": str(Path(args.model_path).resolve()),
        "rows": len(dataset),
        "samples": len(lengths),
        "lengths": distribution(lengths),
        "masks": distribution(masks),
        "t": distribution(timesteps),
        "weight_sums": distribution(weight_sums),
        "elapsed_seconds": time.time() - started,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

