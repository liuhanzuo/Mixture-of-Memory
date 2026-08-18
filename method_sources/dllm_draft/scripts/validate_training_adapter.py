#!/usr/bin/env python3
"""CPU validation of the SFT dataset, config, and trainer imports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoTokenizer

from scaffold_coder.sft_dataset import (
    ScaffoldBatchCollator,
    ScaffoldSFTDataset,
)
from scaffold_coder.tokenizer_utils import validate_ids_within_model
from scaffold_coder.training.scaffold_sft_trainer import (
    ScaffoldFSDPSFTTrainer,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()

    config = OmegaConf.load(
        root
        / "scaffold_coder"
        / "training"
        / "config"
        / "scaffold_sft.yaml"
    )
    model_path = root / "models" / "Dream-Coder-v0-Base-7B"
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    dataset = ScaffoldSFTDataset(
        root / "data" / "scaffold_edu_v0" / "eval_data.parquet",
        tokenizer,
        max_length=config.data.max_length,
        training=False,
        seed=config.trainer.seed,
    )
    model_config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    validate_ids_within_model(
        dataset.registry.extensions, model_config.vocab_size
    )
    items = [dataset[index] for index in range(8)]
    collator = ScaffoldBatchCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=config.data.max_length,
        pad_to_max_length=False,
    )
    batch = collator(items)
    supervised = batch["loss_mask"].sum(dim=1)
    weight_sums = (
        batch["loss_weights"] * batch["loss_mask"].float()
    ).sum(dim=1)
    report = {
        "trainer_class": ScaffoldFSDPSFTTrainer.__name__,
        "config_train_batch_size": config.data.train_batch_size,
        "config_micro_batch_size_per_gpu": config.data.micro_batch_size_per_gpu,
        "config_max_length": config.data.max_length,
        "dataset_rows": len(dataset),
        "batch_shape": list(batch["input_ids"].shape),
        "lengths": batch["length"].tolist(),
        "supervised_tokens": supervised.tolist(),
        "weight_sums": weight_sums.tolist(),
        "max_target_id": int(batch["labels"][batch["loss_mask"]].max()),
        "model_vocab_size": model_config.vocab_size,
        "token_extensions": {
            item.notation: item.token_id
            for item in dataset.registry.extensions
        },
        "all_finite": bool(
            torch.isfinite(batch["loss_weights"]).all().item()
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

