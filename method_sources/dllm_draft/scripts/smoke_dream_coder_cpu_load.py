#!/usr/bin/env python3
"""Load the full Dream-Coder checkpoint on CPU to validate files/API."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path

import psutil
import torch
import transformers
from transformers import AutoModel


def rss_gib() -> float:
    return psutil.Process(os.getpid()).memory_info().rss / 2**30


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.set_num_threads(min(32, os.cpu_count() or 1))
    before = rss_gib()
    start = time.perf_counter()
    model = AutoModel.from_pretrained(
        str(Path(args.model_path).resolve()),
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    load_seconds = time.perf_counter() - start
    after = rss_gib()
    input_weight = model.get_input_embeddings().weight
    output_weight = model.get_output_embeddings().weight
    probe_rows = {}
    for token_id in range(151643, 151686):
        if token_id >= input_weight.shape[0]:
            break
        probe_rows[str(token_id)] = {
            "input_norm": float(input_weight[token_id].float().norm().item()),
            "output_norm": float(output_weight[token_id].float().norm().item()),
        }

    report = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model_class": type(model).__name__,
        "model_type": model.config.model_type,
        "max_position_embeddings": model.config.max_position_embeddings,
        "configured_vocab_size": model.config.vocab_size,
        "input_embedding_shape": list(input_weight.shape),
        "output_embedding_shape": list(output_weight.shape),
        "num_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "load_seconds": load_seconds,
        "rss_before_gib": before,
        "rss_after_load_gib": after,
        "probe_rows": probe_rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    del model
    gc.collect()


if __name__ == "__main__":
    main()

