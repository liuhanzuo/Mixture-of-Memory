#!/usr/bin/env python3
"""Verify a compact PEFT TrainableTokens adapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


ADAPTER_TO_BASE = {
    "base_model.model.model.embed_tokens.trainable_tokens_delta": (
        "model.embed_tokens.weight"
    ),
    "base_model.model.lm_head.trainable_tokens_delta": "lm_head.weight",
}


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
        return torch.stack(
            [tensor[row_id].float() for row_id in row_ids]
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-ids", type=int, nargs="+", required=True)
    args = parser.parse_args()

    adapter_file = args.adapter / "adapter_model.safetensors"
    selected = sorted(set(args.token_ids))
    config = json.loads(
        (args.adapter / "adapter_config.json").read_text(encoding="utf-8")
    )
    if config.get("peft_type") != "TRAINABLE_TOKENS":
        raise SystemExit(f"unexpected PEFT type: {config.get('peft_type')}")
    if sorted(config.get("token_indices", [])) != selected:
        raise SystemExit("adapter token indices do not match requested IDs")
    report: dict[str, object] = {
        "base": str(args.base.resolve()),
        "adapter": str(args.adapter.resolve()),
        "selected_token_ids": selected,
        "peft_type": config["peft_type"],
        "matrices": {},
    }
    selected_changed = False
    with safe_open(adapter_file, framework="pt", device="cpu") as handle:
        keys = set(handle.keys())
        if keys != set(ADAPTER_TO_BASE):
            raise SystemExit(f"unexpected adapter tensors: {sorted(keys)}")
        for adapter_name, base_name in ADAPTER_TO_BASE.items():
            trained_selected = handle.get_tensor(adapter_name).float()
            if tuple(trained_selected.shape) != (
                len(selected),
                trained_selected.shape[1],
            ):
                raise SystemExit(
                    f"invalid compact row shape: {trained_selected.shape}"
                )
            base_selected = tensor_rows(args.base, base_name, selected)
            selected_max = float(
                (trained_selected - base_selected).abs().max().item()
            )
            selected_changed |= selected_max > 0
            report["matrices"][adapter_name] = {
                "stored_rows": trained_selected.shape[0],
                "hidden_size": trained_selected.shape[1],
                "selected_max_abs_delta": selected_max,
                "ordinary_rows_stored": 0,
            }
    if not selected_changed:
        raise SystemExit("no selected token row changed")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
