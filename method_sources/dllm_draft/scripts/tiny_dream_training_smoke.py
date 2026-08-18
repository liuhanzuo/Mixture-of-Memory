#!/usr/bin/env python3
"""CPU end-to-end forward/backward smoke with a tiny real Dream architecture."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from scaffold_coder.canvas import TokenRegistry
from scaffold_coder.corruption import GlobalBandSampler
from scaffold_coder.loss import shifted_weighted_masked_ce
from scaffold_coder.parser import parse_source
from scaffold_coder.sft_dataset import ScaffoldBatchCollator
from scaffold_coder.tokenizer_utils import initialize_model_token_rows


def sample_item(sampled, seq_id: int):
    tensors = sampled.to_tensors()
    tensors["seq_id"] = torch.tensor(seq_id, dtype=torch.long)
    tensors["length"] = torch.tensor(
        len(sampled.state.input_ids), dtype=torch.long
    )
    return tensors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    torch.manual_seed(1)
    model_path = str(Path(args.model_path).resolve())
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    registry = TokenRegistry.build(tokenizer)
    config = AutoConfig.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
    )
    config.hidden_size = 64
    config.intermediate_size = 128
    config.num_hidden_layers = 2
    config.num_attention_heads = 4
    config.num_key_value_heads = 2
    config.max_window_layers = 2
    config.max_position_embeddings = 1024
    config.use_cache = False
    config._attn_implementation = "sdpa"

    model = AutoModel.from_config(
        config,
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    initialize_model_token_rows(
        model, tokenizer, registry.extensions
    )
    model.train()

    module = parse_source(
        "def add_positive(xs):\n"
        "    total = 0\n"
        "    for x in xs:\n"
        "        if x > 0:\n"
        "            total += x\n"
        "    return total\n"
    )
    sampler = GlobalBandSampler(registry)
    prompt = "Write a Python function that sums positive values."
    root = sampler.sample(module, prompt, seed=10, t=0.98)
    leaf = sampler.sample(module, prompt, seed=11, t=0.20)
    items = [sample_item(root, 1), sample_item(leaf, 2)]
    batch = ScaffoldBatchCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=1024,
    )(items)

    attention = batch["attention_mask"].bool()
    pairwise = torch.logical_and(
        attention.unsqueeze(1).unsqueeze(-2),
        attention.unsqueeze(1).unsqueeze(-1),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    started = time.perf_counter()
    output = model(
        input_ids=batch["input_ids"],
        attention_mask=pairwise,
        position_ids=batch["position_ids"],
        use_cache=False,
    )
    loss, metrics = shifted_weighted_masked_ce(
        output.logits,
        batch["labels"],
        batch["loss_mask"],
        batch["loss_weights"],
    )
    loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    func_id = registry.special_id("[FUNC]")
    func_output_grad = float(
        model.get_output_embeddings().weight.grad[func_id].norm().item()
    )
    mask_input_grad = float(
        model.get_input_embeddings()
        .weight.grad[tokenizer.mask_token_id]
        .norm()
        .item()
    )
    optimizer.step()
    elapsed = time.perf_counter() - started

    report = {
        "loss": float(loss.detach().item()),
        "grad_norm": float(grad_norm),
        "func_output_grad_norm": func_output_grad,
        "mask_input_grad_norm": mask_input_grad,
        "elapsed_seconds": elapsed,
        "batch_shape": list(batch["input_ids"].shape),
        "supervised_tokens": batch["loss_mask"].sum(dim=1).tolist(),
        "weight_sums": (
            batch["loss_weights"] * batch["loss_mask"].float()
        ).sum(dim=1).tolist(),
        "metrics": {
            key: float(value.item()) for key, value in metrics.items()
        },
        "tiny_config": {
            "hidden_size": config.hidden_size,
            "intermediate_size": config.intermediate_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "vocab_size": config.vocab_size,
        },
    }
    if not torch.isfinite(loss):
        raise SystemExit("tiny Dream loss is non-finite")
    if func_output_grad <= 0 or mask_input_grad <= 0:
        raise SystemExit("expected structural/output and mask/input gradients")
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

