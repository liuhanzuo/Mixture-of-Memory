#!/usr/bin/env python3
"""Statistics for the dynamic-padding DreamOn matched control."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from transformers import AutoTokenizer

from scaffold_coder.sft_dataset import DreamOnDynamicDataset
from scaffold_coder.tokenizer_utils import extend_dreamon_tokenizer


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
    expand_id = extend_dreamon_tokenizer(tokenizer)[0].token_id
    dataset = DreamOnDynamicDataset(
        args.parquet,
        tokenizer,
        expand_token_id=expand_id,
        training=True,
        max_length=1024,
        seed=1,
    )
    lengths = []
    effective_lengths = []
    masks = []
    expand_targets = []
    delete_targets = []
    weight_sums = []
    for epoch in range(args.samples_per_row):
        dataset.set_epoch(epoch)
        for index in range(len(dataset)):
            item = dataset[index]
            loss_mask = item["loss_mask"]
            lengths.append(int(item["length"]))
            effective_lengths.append(int(item["attention_mask"].sum()))
            masks.append(int(loss_mask.sum()))
            expand_targets.append(
                int(((item["labels"] == expand_id) & loss_mask).sum())
            )
            delete_targets.append(
                int(
                    (
                        (item["labels"] == tokenizer.eos_token_id)
                        & loss_mask
                    ).sum()
                )
            )
            weight_sums.append(float(item["loss_weights"].sum()))

    report = {
        "rows": len(dataset),
        "samples": len(lengths),
        "expand_id": expand_id,
        "lengths": distribution(lengths),
        "effective_attention_lengths": distribution(effective_lengths),
        "masks": distribution(masks),
        "expand_targets": distribution(expand_targets),
        "delete_targets": distribution(delete_targets),
        "weight_sums": distribution(weight_sums),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

