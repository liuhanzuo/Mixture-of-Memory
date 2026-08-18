#!/usr/bin/env python3
"""Validate released DreamOn collator against the matched canonical dataset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoTokenizer

from src.trainer.sft_expand_dataset import SFTExpandDataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parquet", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rows", type=int, default=100)
    args = parser.parse_args()

    torch.manual_seed(1)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = SFTExpandDataset(
        parquet_files=args.parquet,
        tokenizer=tokenizer,
        prompt_key="prompt",
        response_key="response",
        max_length=1024,
        truncation="right",
        middle_strategy="line",
        middle_line_num=None,
        merge_prob=0.5,
        max_delete=64,
        merge_schedule="dynamic_inverse",
        use_uniform_merge_prob=0.5,
    )
    expand_id = 151667
    rows = min(args.rows, len(dataset))
    expand_targets = delete_targets = supervised = 0
    sequence_lengths = []
    t_values = []
    for index in range(rows):
        item = dataset[index]
        mask = item["loss_mask"].bool()
        labels = item["labels"]
        expand_targets += int(((labels == expand_id) & mask).sum())
        delete_targets += int(
            ((labels == tokenizer.eos_token_id) & mask).sum()
        )
        supervised += int(mask.sum())
        sequence_lengths.append(int(item["attention_mask"].sum()))
        t_values.append(float(item["t"]))

    report = {
        "dataset_rows": len(dataset),
        "validated_rows": rows,
        "expand_id": expand_id,
        "tokenizer_length": len(tokenizer),
        "model_vocab_reserved_expand": expand_id >= len(tokenizer),
        "expand_targets": expand_targets,
        "delete_targets": delete_targets,
        "supervised_targets": supervised,
        "mean_effective_length": sum(sequence_lengths) / rows,
        "mean_t": sum(t_values) / rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

