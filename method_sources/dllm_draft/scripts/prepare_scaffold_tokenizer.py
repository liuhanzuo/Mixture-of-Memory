#!/usr/bin/env python3
"""Create and audit a Scaffold-Coder tokenizer without loading model weights."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from transformers import AutoConfig, AutoTokenizer

from scaffold_coder.tokenizer_utils import (
    extend_tokenizer,
    validate_ids_within_model,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config = AutoConfig.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    original_length = len(tokenizer)
    extensions = extend_tokenizer(tokenizer)
    validate_ids_within_model(extensions, config.vocab_size)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_dir)

    manifest = {
        "model_path": str(Path(args.model_path).resolve()),
        "configured_vocab_size": config.vocab_size,
        "original_tokenizer_length": original_length,
        "extended_tokenizer_length": len(tokenizer),
        "embedding_resize_required": len(tokenizer) > config.vocab_size,
        "extensions": [asdict(item) for item in extensions],
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()

