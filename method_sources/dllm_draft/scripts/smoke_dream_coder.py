#!/usr/bin/env python3
"""Minimal deterministic Dream-Coder GPU inference smoke."""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch
import transformers
from transformers import AutoModel, AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)

    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    messages = [
        {
            "role": "user",
            "content": (
                "Write only Python code for a function add(a, b) that returns "
                "the sum of a and b."
            ),
        }
    ]
    inputs = tokenizer.apply_chat_template(
        messages,
        return_tensors="pt",
        return_dict=True,
        add_generation_prompt=True,
    )

    torch.cuda.set_device(0)
    torch.cuda.init()
    device = torch.device("cuda", 0)
    torch.cuda.reset_peak_memory_stats()
    load_start = time.perf_counter()
    model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    model = model.to(device).eval()
    torch.cuda.synchronize(device)
    load_seconds = time.perf_counter() - load_start

    input_ids = inputs.input_ids.to(device)
    attention_mask = inputs.attention_mask.to(device)
    generation_start = time.perf_counter()
    with torch.inference_mode():
        output = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            output_history=True,
            return_dict_in_generate=True,
            steps=args.steps,
            temperature=0.0,
            top_p=1.0,
            alg="entropy",
            alg_temp=0.0,
        )
    torch.cuda.synchronize(device)
    generation_seconds = time.perf_counter() - generation_start

    generated_ids = output.sequences[0, input_ids.shape[1] :]
    generated = tokenizer.decode(generated_ids.tolist())
    generated_before_eos = generated.split(tokenizer.eos_token)[0]
    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "transformers": transformers.__version__,
        "device": torch.cuda.get_device_name(device),
        "model_path": model_path,
        "config_max_position_embeddings": model.config.max_position_embeddings,
        "input_tokens": int(input_ids.numel()),
        "steps": args.steps,
        "max_new_tokens": args.max_new_tokens,
        "load_seconds": load_seconds,
        "generation_seconds": generation_seconds,
        "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30,
        "generated": generated,
        "generated_before_eos": generated_before_eos,
        "history_length": len(output.history) if output.history is not None else None,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
