#!/usr/bin/env python3
"""Validate special-token row initialization on the real Dream checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scaffold_coder.tokenizer_utils import (
    extend_tokenizer,
    initialize_model_token_rows,
    validate_ids_within_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    extensions = extend_tokenizer(tokenizer)
    validate_ids_within_model(extensions, config.vocab_size)
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )

    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    before = {
        item.notation: {
            "token_id": item.token_id,
            "input_norm": float(
                input_weight[item.token_id].float().norm().item()
            ),
            "output_norm": float(
                output_weight[item.token_id].float().norm().item()
            ),
        }
        for item in extensions
    }
    initialization = initialize_model_token_rows(model, tokenizer, extensions)
    after = {
        item.notation: {
            "token_id": item.token_id,
            "input_norm": float(
                input_weight[item.token_id].float().norm().item()
            ),
            "output_norm": float(
                output_weight[item.token_id].float().norm().item()
            ),
        }
        for item in extensions
    }

    failures = [
        notation
        for notation, row in after.items()
        if row["input_norm"] == 0.0 or row["output_norm"] == 0.0
    ]
    report = {
        "configured_vocab_size": config.vocab_size,
        "tokenizer_length": len(tokenizer),
        "before": before,
        "after": after,
        "initialization": initialization,
        "zero_norm_failures": failures,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"zero-norm initialized rows: {failures}")


if __name__ == "__main__":
    main()

