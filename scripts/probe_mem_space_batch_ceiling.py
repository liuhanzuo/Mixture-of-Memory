#!/usr/bin/env python3
"""Batch-size memory ceiling probe for Memory-Space v0 on a single GPU.

Goal (2026-04-26 Branch-3 pre-dispatch): given the upcoming code change,
figure out the largest `--batch_size` we can pass to
``scripts/train_mem_space_pg19.py`` on a B200 (183 GiB) before OOM, so the
first Branch-3 8-GPU run saturates the card to ≥80 % per the new CLAUDE.md
rule.

Strategy: replicate the train-step shape (forward + backward through
MemorySpaceLayer with hidden_to_slot writeback re-attached after the
coder patch) at increasing ``batch_size`` ∈ {1, 2, 4, 8}; record peak
allocated bytes per iteration. Single-GPU only (nproc=1), so the other
7 GPUs remain free for the main Branch-3 run once it's green-lit.

Output: JSON written to ``--output_dir/batch_ceiling_probe.json`` with
per-batch-size peak memory + status (ok / oom / other_error).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time

import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space import (  # noqa: E402
    MemorySpaceConfig,
    apply_mem_space_to_model,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mem_space v0 batch-size ceiling probe")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--num_slots", type=int, default=512)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--dtype", type=str, default="bfloat16")
    p.add_argument("--batch_sizes", type=str, default="1,2,4,8",
                   help="Comma-separated list to probe in order.")
    p.add_argument("--unfreeze_hidden_to_slot", action="store_true",
                   help="Include hidden_to_slot in the trainable set (Branch-3 sim).")
    return p.parse_args()


def _build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    model = LlamaForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, attn_implementation="sdpa",
    ).to(device)
    ms_cfg = MemorySpaceConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        selector_dim=128,
        writeback_gate_warmup_steps=0,
        writeback_gate_max=0.3,
        load_balance_weight=0.01,
        slot_init="random",
        slot_init_noise=1.0,
        enable_writeback=True,
        return_aux_losses=True,
        hidden_to_slot_frozen=not args.unfreeze_hidden_to_slot,
    )
    apply_mem_space_to_model(model, ms_cfg, layer_indices=None)
    # H7 fix v2 (2026-04-26 23:30): snapshot rotary inv_freq / original_inv_freq
    # in fp32 BEFORE the lossy `.to(dtype=bf16)` cast. Upcasting after the
    # cast cannot recover mantissa bits — the round-to-bf16 already happened.
    # See scripts/probe_branch3_bypass_parity.py for the full rationale.
    _rope_snapshot = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass
    model.to(device=device, dtype=dtype)
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
    except AttributeError:
        pass
    # Freeze backbone, unfreeze mem_space params.
    for p in model.parameters():
        p.requires_grad = False
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", [])
    for w in mem_layers:
        for p in w.selector.parameters():
            p.requires_grad = True
        w.gate_param.requires_grad = True
        if hasattr(w, "slot_output_gate"):
            w.slot_output_gate.requires_grad = True
        for p in w.slot_to_hidden.parameters():
            p.requires_grad = True
        if args.unfreeze_hidden_to_slot:
            for p in w.hidden_to_slot.parameters():
                p.requires_grad = True
    return model


def _reset_banks(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    for w in getattr(root, "_mem_space_layers", []):
        w.memory_bank.reset()


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda:0")
    batch_sizes = [int(x) for x in args.batch_sizes.split(",") if x]

    # Load a few chunks into CPU mem.
    raw = np.load(args.data, mmap_mode="r")
    # use chunks starting at 0 (skip_chunks=0 mirror the actual training)
    max_bs = max(batch_sizes)
    samples = torch.tensor(raw[:max_bs, :args.seq_len].astype(np.int64), dtype=torch.long)
    logger.info("Probe samples loaded: shape=%s", tuple(samples.shape))

    results = []
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        entry = {"batch_size": bs, "status": "pending"}
        try:
            model = _build_model(args, device)
            trainable = [p for p in model.parameters() if p.requires_grad]
            optim = torch.optim.AdamW(trainable, lr=1e-3, weight_decay=0.0, betas=(0.9, 0.95))
            model.train()
            batch = samples[:bs].to(device)
            labels = batch.clone()
            t0 = time.time()
            optim.zero_grad(set_to_none=True)
            _reset_banks(model)
            out = model(input_ids=batch, labels=labels, use_cache=False)
            loss = out.loss
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated(device) / 1024**3
            reserved = torch.cuda.max_memory_reserved(device) / 1024**3
            dt = time.time() - t0
            entry.update({
                "status": "ok",
                "peak_allocated_gib": round(peak, 2),
                "peak_reserved_gib": round(reserved, 2),
                "step_time_s": round(dt, 2),
                "loss": float(loss.detach().item()),
            })
            logger.info("bs=%d OK peak=%.2f GiB reserved=%.2f GiB dt=%.1fs loss=%.4f",
                        bs, peak, reserved, dt, float(loss.detach().item()))
        except torch.cuda.OutOfMemoryError as e:
            entry["status"] = "oom"
            entry["error"] = str(e)[:200]
            logger.warning("bs=%d OOM: %s", bs, str(e)[:200])
        except Exception as e:  # noqa: BLE001
            entry["status"] = "error"
            entry["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            logger.error("bs=%d error: %s", bs, entry["error"])
        finally:
            results.append(entry)
            try:
                del model  # noqa: F821
            except Exception:
                pass
            torch.cuda.empty_cache()

    out_path = os.path.join(args.output_dir, "batch_ceiling_probe.json")
    with open(out_path, "w") as f:
        json.dump({
            "model": args.model,
            "seq_len": args.seq_len,
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "dtype": args.dtype,
            "unfreeze_hidden_to_slot": args.unfreeze_hidden_to_slot,
            "results": results,
            "gpu_total_gib": round(torch.cuda.get_device_properties(device).total_memory / 1024**3, 2),
        }, f, indent=2)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
