#!/usr/bin/env python3
"""Summarize schedule-only masking states over repeated eval rows."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scaffold_coder.canvas import ROLE_ID
from scaffold_coder.roles import MaskRole
from scaffold_coder.sft_dataset import ScheduleOnlySFTDataset


def quantile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(fraction * len(ordered)))
    return float(ordered[index])


def summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "p50": statistics.median(values),
        "p90": quantile(values, 0.90),
        "p99": quantile(values, 0.99),
        "max": max(values),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--rows", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=29)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = ScheduleOnlySFTDataset(
        args.parquet,
        tokenizer,
        training=True,
        max_length=1024,
        seed=args.seed,
    )
    count = min(args.rows, len(dataset))
    lengths: list[float] = []
    masks: list[float] = []
    masses: list[float] = []
    role_counts: Counter[str] = Counter()
    t_bins: dict[str, list[int]] = defaultdict(list)
    structural_masks = 0
    content_masks = 0
    id_to_role = {identifier: role for role, identifier in ROLE_ID.items()}
    for epoch in range(args.epochs):
        dataset.set_epoch(epoch)
        for index in range(count):
            item = dataset[index]
            selected = item["loss_mask"]
            role_ids = item["role_ids"][selected].tolist()
            for role_id in role_ids:
                role = id_to_role[role_id]
                role_counts[role.value] += 1
                if role is MaskRole.RULE:
                    structural_masks += 1
                else:
                    content_masks += 1
            t = float(item["t"])
            lower = int(t * 10) / 10
            label = f"{lower:.1f}-{min(1.0, lower + 0.1):.1f}"
            masked = int(selected.sum())
            t_bins[label].append(masked)
            lengths.append(float(item["length"]))
            masks.append(float(masked))
            masses.append(float(item["loss_weights"].sum()))

    report = {
        "examples": count * args.epochs,
        "rows": count,
        "epochs": args.epochs,
        "length": summary(lengths),
        "supervised_masks": summary(masks),
        "weight_mass": summary(masses),
        "role_counts": dict(role_counts),
        "structural_masks": structural_masks,
        "content_masks": content_masks,
        "structural_fraction": (
            structural_masks / (structural_masks + content_masks)
        ),
        "mask_count_by_t_bin": {
            key: summary([float(value) for value in values])
            for key, values in sorted(t_bins.items())
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
