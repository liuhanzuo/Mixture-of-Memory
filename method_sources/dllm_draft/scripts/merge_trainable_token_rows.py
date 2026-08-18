#!/usr/bin/env python3
"""Merge a compact PEFT TrainableTokens adapter into a standalone checkpoint."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import torch.distributed.tensor  # noqa: F401
from peft import PeftModel
from safetensors import safe_open
from transformers import AutoModel, AutoTokenizer


REMOTE_CODE_FILES = (
    "configuration_dream.py",
    "generation_utils.py",
    "modeling_dream.py",
    "tokenization_dream.py",
)


def tensor_rows(
    checkpoint: Path,
    tensor_name: str,
    row_ids: list[int],
) -> torch.Tensor:
    index = json.loads(
        (checkpoint / "model.safetensors.index.json").read_text(
            encoding="utf-8"
        )
    )
    shard = checkpoint / index["weight_map"][tensor_name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        tensor = handle.get_slice(tensor_name)
        return torch.stack([tensor[row_id] for row_id in row_ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.resolve()
    adapter = args.adapter.resolve()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"refusing to overwrite output: {output}")
    adapter_config = json.loads(
        (adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if adapter_config.get("peft_type") != "TRAINABLE_TOKENS":
        raise SystemExit("adapter is not a PEFT TrainableTokens checkpoint")
    token_ids = sorted(set(adapter_config["token_indices"]))

    tokenizer = AutoTokenizer.from_pretrained(
        base,
        trust_remote_code=True,
        local_files_only=True,
    )
    model = AutoModel.from_pretrained(
        base,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
        device_map="cpu",
    )
    adapted = PeftModel.from_pretrained(
        model,
        adapter,
        is_trainable=False,
    )
    merged = adapted.merge_and_unload()
    merged.config.scaffold_mode = "rung_mixture"
    merged.config.scaffold_token_ids = {
        notation: value["token_id"]
        for notation, value in json.loads(
            (adapter / "scaffold_tokens.json").read_text(encoding="utf-8")
        ).items()
    }
    merged.config.scaffold_spec_version = "v0"
    merged.config.expand_token_id = merged.config.scaffold_token_ids["[expand]"]
    merged.config.delete_token_id = merged.config.scaffold_token_ids["[delete]"]

    output.mkdir(parents=True)
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
    shutil.copy2(adapter / "scaffold_tokens.json", output / "scaffold_tokens.json")

    ordinary_ids = [0, 1, 100, 1000, 50000, 150000, 151666, 151686]
    matrices: dict[str, object] = {}
    selected_changed = False
    for tensor_name in ("model.embed_tokens.weight", "lm_head.weight"):
        base_ordinary = tensor_rows(base, tensor_name, ordinary_ids)
        merged_ordinary = tensor_rows(output, tensor_name, ordinary_ids)
        ordinary_equal = torch.equal(base_ordinary, merged_ordinary)
        base_selected = tensor_rows(base, tensor_name, token_ids).float()
        merged_selected = tensor_rows(output, tensor_name, token_ids).float()
        selected_max = float(
            (merged_selected - base_selected).abs().max().item()
        )
        selected_changed |= selected_max > 0
        matrices[tensor_name] = {
            "ordinary_rows_bit_exact": ordinary_equal,
            "selected_max_abs_delta": selected_max,
        }
        if not ordinary_equal:
            raise SystemExit(f"ordinary rows changed in {tensor_name}")
    if not selected_changed:
        raise SystemExit("merged checkpoint contains no selected-row change")

    manifest = {
        "base": str(base),
        "adapter": str(adapter),
        "output": str(output),
        "token_ids": token_ids,
        "matrices": matrices,
    }
    (output / "token_row_merge_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
