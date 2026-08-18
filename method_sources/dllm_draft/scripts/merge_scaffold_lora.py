#!/usr/bin/env python3
"""Merge a Scaffold LoRA adapter into a standalone Dream checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModel, AutoTokenizer

from scaffold_coder.tokenizer_utils import (
    extend_tokenizer,
    initialize_model_token_rows,
)


REMOTE_CODE_FILES = (
    "configuration_dream.py",
    "generation_utils.py",
    "modeling_dream.py",
    "tokenization_dream.py",
)


def has_scaffold_token_rows(checkpoint: Path) -> bool:
    return (checkpoint / "scaffold_tokens.json").is_file()


def scale_lora_adapters(model, factor: float) -> int:
    """Scale every active LoRA delta before merge; return adapter count."""

    count = 0
    for module in model.modules():
        scaling = getattr(module, "scaling", None)
        if not isinstance(scaling, dict):
            continue
        for adapter_name in tuple(scaling):
            scaling[adapter_name] *= factor
            count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if not 0 <= args.scale <= 1:
        raise ValueError("--scale must lie in [0,1]")

    base = Path(args.base).resolve()
    adapter = Path(args.adapter).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=True,
    )
    extensions = extend_tokenizer(tokenizer)
    model = AutoModel.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    initialization = initialize_model_token_rows(
        model,
        tokenizer,
        extensions,
    )
    if has_scaffold_token_rows(base):
        initialization = {
            extension.notation: {
                "token_id": extension.token_id,
                "preserved": True,
                "source_ids": [],
            }
            for extension in extensions
        }
    peft_model = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    )
    if args.scale != 1.0:
        scaled_adapters = scale_lora_adapters(peft_model, args.scale)
        if scaled_adapters == 0:
            raise RuntimeError("no LoRA scaling entries were found")
    else:
        scaled_adapters = None
    merged = peft_model.merge_and_unload()
    token_ids = {
        extension.notation: extension.token_id
        for extension in extensions
    }
    merged.config.scaffold_mode = "rung_mixture"
    merged.config.scaffold_token_ids = token_ids
    merged.config.scaffold_spec_version = "v0"
    merged.config.expand_token_id = token_ids["[expand]"]
    merged.config.delete_token_id = token_ids["[delete]"]
    merged.save_pretrained(
        output,
        safe_serialization=True,
        max_shard_size="4GB",
    )
    tokenizer.save_pretrained(output)

    for name in REMOTE_CODE_FILES:
        source = base / name
        if source.exists():
            shutil.copy2(source, output / name)
    for name in ("scaffold_tokens.json",):
        source = adapter / name
        if source.exists():
            shutil.copy2(source, output / name)

    manifest = {
        "base": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "dtype": "bfloat16",
        "lora_scale": args.scale,
        "scaled_adapter_entries": scaled_adapters,
        "token_initialization": initialization,
        "scaffold_token_ids": token_ids,
    }
    (output / "merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
