#!/usr/bin/env python3
"""Audit Dream tokenizer boundaries and proposed Scaffold-Coder tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from scaffold_coder.special_tokens import ALL_TOKEN_TEXTS, TOKEN_TEXT


SAMPLES = {
    "function": (
        "def count_pairs(nums, target):\n",
        ["def ", "count_pairs(nums, target)", ":\n"],
    ),
    "indented_statement": (
        "    count += 1\n",
        ["    ", "count += 1", "\n"],
    ),
    "nested_if": (
        "            if nums[i] + nums[j] == target:\n",
        ["            ", "if ", "nums[i] + nums[j] == target", ":\n"],
    ),
    "newline_indent": (
        "\n        return count\n",
        ["\n        ", "return count", "\n"],
    ),
}


def encode(tokenizer, text: str) -> list[int]:
    return tokenizer.encode(text, add_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    report: dict[str, object] = {
        "model_path": str(Path(args.model_path).resolve()),
        "tokenizer_class": type(tokenizer).__name__,
        "base_vocab_size": tokenizer.vocab_size,
        "base_len": len(tokenizer),
        "model_max_length": tokenizer.model_max_length,
        "mask_token": tokenizer.mask_token,
        "mask_token_id": tokenizer.mask_token_id,
        "bos_token": tokenizer.bos_token,
        "bos_token_id": tokenizer.bos_token_id,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "samples_before_add": {},
    }

    for name, (text, segments) in SAMPLES.items():
        whole = encode(tokenizer, text)
        segmented = [token_id for segment in segments for token_id in encode(tokenizer, segment)]
        report["samples_before_add"][name] = {
            "text": text,
            "segments": segments,
            "whole_ids": whole,
            "segmented_ids": segmented,
            "whole_length": len(whole),
            "segmented_length": len(segmented),
            "same_ids": whole == segmented,
            "segmented_decodes_exactly": tokenizer.decode(segmented) == text,
            "whole_tokens": tokenizer.convert_ids_to_tokens(whole),
            "segmented_tokens": tokenizer.convert_ids_to_tokens(segmented),
        }

    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": list(ALL_TOKEN_TEXTS)}
    )
    report["num_added"] = added
    report["extended_len"] = len(tokenizer)
    report["special_tokens"] = {}
    for notation, physical in TOKEN_TEXT.items():
        ids = encode(tokenizer, physical)
        report["special_tokens"][notation] = {
            "physical": physical,
            "ids": ids,
            "atomic": len(ids) == 1,
            "id": ids[0] if len(ids) == 1 else None,
            "decoded": tokenizer.decode(ids),
        }

    failures = [
        notation
        for notation, item in report["special_tokens"].items()
        if not item["atomic"]
    ]
    report["atomic_failures"] = failures

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if failures:
        raise SystemExit(f"non-atomic added tokens: {failures}")


if __name__ == "__main__":
    main()

