#!/usr/bin/env python3
"""Minimal DreamOn variable-length infilling GPU smoke."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--initial-masks", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(1)
    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    prefix = "def square(x):\n"
    suffix = "\n"
    ids = (
        [tokenizer.bos_token_id]
        + tokenizer.encode(prefix, add_special_tokens=False)
        + [tokenizer.mask_token_id] * args.initial_masks
        + tokenizer.encode(suffix, add_special_tokens=False)
        + [tokenizer.eos_token_id]
    )

    torch.cuda.set_device(0)
    torch.cuda.init()
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats()
    load_started = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    ).to(device).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_started

    input_ids = torch.tensor([ids], dtype=torch.long, device=device)
    generation_started = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            input_ids,
            max_new_tokens=args.max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            number_transfer_tokens=1,
            temperature=0.0,
            top_p=1.0,
            alg="entropy",
            alg_temp=0.0,
        )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_started
    sequence = output.sequences[0].tolist()
    decoded = tokenizer.decode(sequence)
    report = {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(device),
        "model_path": model_path,
        "max_position_embeddings": model.config.max_position_embeddings,
        "initial_masks": args.initial_masks,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "history_length": len(output.history) if output.history is not None else None,
        "input_decoded": tokenizer.decode(ids),
        "output_decoded": decoded,
        "output_token_count": len(sequence),
    }
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
