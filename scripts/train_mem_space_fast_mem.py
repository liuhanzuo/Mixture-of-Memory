#!/usr/bin/env python3
"""FastMem + Dolmino Continued Pretraining (CPT).

Trains the mem_space memory adapter with the FastMem (Gated Delta Rule)
module enabled. FastMem captures a continuous running summary of ALL tokens
via a per-layer fast-weight memory, complementing the discrete slot routing.

Key differences from train_mem_space_dolmino_cpt.py:
    - use_fast_mem=True by default
    - FastMem parameters use a separate optimizer group with higher LR (3x)
    - FastMem state management: reset at sample boundary, carry (detached)
      across chunk boundaries within the same document
    - Logs fusion_gate value to wandb for monitoring ramp-up

Training step (same as Dolmino CPT):
    1. _reset_banks(model) + _reset_fast_mem(model) — fresh memory per document
    2. For each of N context chunks:
       - model(ctx, use_cache=False) with no_grad → memory accumulates
       - _detach_fast_mem(model) — break grad graph but KEEP state
    3. _detach_banks(model) + _detach_fast_mem(model)
    4. out = model(target_ids, labels=target_ids, use_cache=False)
    5. loss = (out.loss + aux_loss) / grad_accum_steps
    6. loss.backward()

Fork of scripts/train_mem_space_dolmino_cpt.py (2026-05-21).
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make sure third_party/babilong-pkg is on sys.path for BABILong mix.
_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space import (  # noqa: E402
    MemorySpaceConfig,
    apply_mem_space_to_model,
    _reset_fast_mem,
    _detach_fast_mem,
)
from src.memory.mem_space.dolmino_dataset import DolminoCurriculumDataset  # noqa: E402
from src.memory.mem_space.babilong_dataset import (  # noqa: E402
    BABILongTrainDataset,
    babilong_collate_fn,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Curriculum Scheduler
# --------------------------------------------------------------------------- #


class CurriculumScheduler:
    """Parses curriculum string and returns n_ctx for a given step.

    Format: "step1:n_ctx1,step2:n_ctx2,..."
    Example: "0:1,5000:2,10000:4,15000:8,25000:16"
    """

    def __init__(self, curriculum_str: str) -> None:
        self.schedule: List[Tuple[int, int]] = []
        for entry in curriculum_str.split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) != 2:
                raise ValueError(f"Invalid curriculum entry: {entry!r}")
            step, n_ctx = int(parts[0]), int(parts[1])
            self.schedule.append((step, n_ctx))
        self.schedule.sort(key=lambda x: x[0])
        if not self.schedule:
            self.schedule = [(0, 1)]

    def get_n_ctx(self, step: int) -> int:
        n_ctx = self.schedule[0][1]
        for s, n in self.schedule:
            if step >= s:
                n_ctx = n
            else:
                break
        return n_ctx

    def __repr__(self) -> str:
        return f"CurriculumScheduler({self.schedule})"


# --------------------------------------------------------------------------- #
# Cosine LR Schedule with Warmup
# --------------------------------------------------------------------------- #


def cosine_lr_schedule(step: int, total_steps: int, warmup_steps: int,
                       base_lr: float, min_lr_ratio: float = 0.1) -> float:
    """Compute learning rate with linear warmup + cosine decay."""
    if step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay)


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #


def init_distributed() -> Tuple[int, int, int]:
    if "RANK" not in os.environ:
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Mem-space + FastMem helpers
# --------------------------------------------------------------------------- #


def _mem_space_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Collect all mem_space trainable params EXCLUDING fast_mem."""
    params: List[torch.nn.Parameter] = []
    seen: set = set()
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return params
    for wrapper in mem_layers:
        for p in wrapper.selector.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
        if id(wrapper.gate_param) not in seen:
            params.append(wrapper.gate_param); seen.add(id(wrapper.gate_param))
        slot_gate = getattr(wrapper, "slot_output_gate", None)
        if slot_gate is not None and id(slot_gate) not in seen:
            params.append(slot_gate); seen.add(id(slot_gate))
        for p in wrapper.slot_to_hidden.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
        if not getattr(wrapper.config, "hidden_to_slot_frozen", True):
            for p in wrapper.hidden_to_slot.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        # Dual-gate projections
        if wrapper.gate_proj_new is not None:
            for p in wrapper.gate_proj_new.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        if wrapper.gate_proj_mem is not None:
            for p in wrapper.gate_proj_mem.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        gate_bias = getattr(wrapper, "gate_bias", None)
        if gate_bias is not None and id(gate_bias) not in seen:
            params.append(gate_bias); seen.add(id(gate_bias))
    # L3 summary pool params
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        for p in l3_pool.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    # L2 token compressor params
    l2_comp = getattr(root, "_l2_compressor", None)
    if l2_comp is not None:
        for p in l2_comp.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    return params


def _fast_mem_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Collect FastMem parameters (separate LR group)."""
    params: List[torch.nn.Parameter] = []
    seen: set = set()
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return params
    for wrapper in mem_layers:
        fast_mem = getattr(wrapper, "fast_mem", None)
        if fast_mem is not None:
            for p in fast_mem.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
    return params


def _freeze_backbone(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in _mem_space_params(model):
        p.requires_grad = True
    for p in _fast_mem_params(model):
        p.requires_grad = True


def _step_counters_inc(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.step_counter += 1


def _reset_banks(model: torch.nn.Module) -> None:
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
    else:
        mem_layers = getattr(root, "_mem_space_layers", None)
        if mem_layers:
            for w in mem_layers:
                w.memory_bank.reset()
    # Reset L3 summary state
    if hasattr(root, "_l3_summary_for_next_chunk"):
        root._l3_summary_for_next_chunk = None
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None


def _detach_banks(model: torch.nn.Module) -> None:
    """Detach memory bank tensors to break gradient graph from context passes."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.detach_()
        return
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.memory_bank.detach_()


def _collect_aux_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    total = torch.zeros((), device=device)
    if not mem_layers:
        return total
    for w in mem_layers:
        for key in ("load_balance", "entropy", "key_repulsion", "peak_routing"):
            v = w.last_aux_losses.get(key)
            if v is not None:
                total = total + v
    return total


def _get_fusion_gate_value(model: torch.nn.Module) -> float:
    """Get mean fusion gate sigmoid value from layer 0 for logging."""
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return 0.0
    w0 = mem_layers[0]
    fast_mem = getattr(w0, "fast_mem", None)
    if fast_mem is None:
        return 0.0
    with torch.no_grad():
        return torch.sigmoid(fast_mem.fusion_gate).mean().item()


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="FastMem + Memory-Space Dolmino CPT",
    )

    # Base
    p.add_argument("--model_path", type=str, default="models/Meta-Llama-3-8B",
                   help="Path to base Llama-3-8B directory")
    p.add_argument("--output_dir", type=str, required=True)

    # Warm-start
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Path to an existing mem_space adapter .pt to warm-start from.")
    p.add_argument("--init_adapter_config", type=str, default=None,
                   help="Path to adapter_config.json for inheriting hyperparams.")

    # Dolmino data
    p.add_argument("--dolmino_path", type=str,
                   default="MemLong/data/processed/dolmino_0.5B_1024/train",
                   help="Path to pre-tokenised Dolmino Arrow dataset.")
    p.add_argument("--chunk_size", type=int, default=1024,
                   help="Token count per chunk.")

    # Curriculum
    p.add_argument("--curriculum", type=str,
                   default="0:1,5000:2,10000:4,15000:8,25000:16",
                   help="Curriculum schedule: 'step:n_ctx,step:n_ctx,...'")

    # BABILong mix
    p.add_argument("--babilong_mix_fraction", type=float, default=0.15)
    p.add_argument("--babilong_dataset", type=str, default="RMT-team/babilong")
    p.add_argument("--babilong_tasks", type=str, default="qa1,qa2,qa5")
    p.add_argument("--babilong_lengths", type=str, default="0k,1k,2k,4k")
    p.add_argument("--use_chat_template", action="store_true", default=False)

    # Training
    p.add_argument("--total_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--fast_mem_lr_mult", type=float, default=3.0,
                   help="FastMem params LR multiplier relative to base LR.")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--start_step", type=int, default=0)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proj_grad_clip", type=float, default=0.1)
    p.add_argument("--batch_size", type=int, default=1)

    # Logging / saving / eval
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=2000)
    p.add_argument("--eval_samples", type=int, default=50)

    # mem_space hyperparams
    p.add_argument("--num_slots", type=int, default=512)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--selector_dim", type=int, default=128)
    p.add_argument("--writeback_gate_max", type=float, default=0.3)
    p.add_argument("--writeback_warmup_steps", type=int, default=0)
    p.add_argument("--load_balance_weight", type=float, default=0.01)
    p.add_argument("--entropy_aux_weight", type=float, default=0.001)
    p.add_argument("--selector_temperature", type=float, default=20.0)
    p.add_argument("--key_repulsion_weight", type=float, default=0.05)
    p.add_argument("--key_repulsion_threshold", type=float, default=0.3)
    p.add_argument("--peak_routing_weight", type=float, default=0.05)
    p.add_argument("--slot_value_norm_cap", type=float, default=5.0)
    p.add_argument("--slot_init", type=str, default="random",
                   choices=["zero", "random", "hidden_pool", "strided_token"])
    p.add_argument("--slot_init_noise", type=float, default=0.05)
    p.add_argument("--unfreeze_hidden_to_slot", action="store_true", default=True)
    p.add_argument("--shared_memory_bank", action="store_true", default=True)
    p.add_argument("--swa_window", type=int, default=0)

    # Dual-gate
    p.add_argument("--use_dual_gate", action="store_true", default=True)
    p.add_argument("--input_bias_init", type=float, default=0.0)
    p.add_argument("--forget_bias_init", type=float, default=2.0)
    p.add_argument("--dual_gate_tanh_new", action="store_true", default=True)

    # L3 Summary
    p.add_argument("--use_l3_summary", action="store_true", default=True)
    p.add_argument("--l3_n_summary", type=int, default=64)
    p.add_argument("--l3_n_layers", type=int, default=2)
    p.add_argument("--l3_n_heads", type=int, default=8)
    p.add_argument("--disable_l1_inject", action="store_true", default=False)

    # v6/v7 writeback
    p.add_argument("--use_replace_writeback", action="store_true", default=False)
    p.add_argument("--num_global_slots", type=int, default=0)
    p.add_argument("--global_slot_forget_bias", type=float, default=1.0)
    p.add_argument("--global_slot_input_gate_only", action="store_true", default=False)

    # FastMem (Gated Delta Rule)
    p.add_argument("--use_fast_mem", action="store_true", default=True,
                   help="Enable FastMem (Gated Delta Rule) module.")
    p.add_argument("--fast_mem_heads", type=int, default=4,
                   help="Number of fast-weight heads.")
    p.add_argument("--fast_mem_d_state", type=int, default=128,
                   help="Key/value dimension per fast-weight head.")
    p.add_argument("--fast_mem_chunk_size", type=int, default=16,
                   help="BPTT window for sequential fallback (ignored when fla available).")
    p.add_argument("--fast_mem_fusion_init", type=float, default=-2.0,
                   help="Initial fusion gate logit (sigmoid(-2)≈0.12).")

    # Activation-memory reduction
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Misc
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)

    # Wandb
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model build
# --------------------------------------------------------------------------- #


def build_model(args, device, dtype) -> torch.nn.Module:
    """Load Llama + patch with mem_space (including FastMem) + warm-start."""
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    ).to(device)

    ms_cfg = MemorySpaceConfig(
        num_slots=args.num_slots,
        top_k=args.top_k,
        selector_dim=args.selector_dim,
        writeback_gate_warmup_steps=args.writeback_warmup_steps,
        writeback_gate_max=args.writeback_gate_max,
        load_balance_weight=args.load_balance_weight,
        entropy_aux_weight=args.entropy_aux_weight,
        selector_temperature=args.selector_temperature,
        key_repulsion_weight=args.key_repulsion_weight,
        key_repulsion_threshold=args.key_repulsion_threshold,
        peak_routing_weight=args.peak_routing_weight,
        slot_init=args.slot_init,
        slot_init_noise=args.slot_init_noise,
        enable_writeback=True,
        return_aux_losses=True,
        hidden_to_slot_frozen=not args.unfreeze_hidden_to_slot,
        shared_memory_bank=args.shared_memory_bank,
        swa_window=args.swa_window,
        slot_value_norm_cap=args.slot_value_norm_cap,
        use_dual_gate=args.use_dual_gate,
        input_bias_init=args.input_bias_init,
        forget_bias_init=args.forget_bias_init,
        dual_gate_tanh_new=args.dual_gate_tanh_new,
        use_l3_summary=args.use_l3_summary,
        l3_n_summary=args.l3_n_summary,
        l3_n_layers=args.l3_n_layers,
        l3_n_heads=args.l3_n_heads,
        disable_l1_inject=args.disable_l1_inject,
        use_replace_writeback=args.use_replace_writeback,
        num_global_slots=args.num_global_slots,
        global_slot_forget_bias=args.global_slot_forget_bias,
        global_slot_input_gate_only=args.global_slot_input_gate_only,
        gradient_checkpointing=args.gradient_checkpointing,
        # FastMem config
        use_fast_mem=args.use_fast_mem,
        fast_mem_num_heads=args.fast_mem_heads,
        fast_mem_d_state=args.fast_mem_d_state,
        fast_mem_chunk_size=args.fast_mem_chunk_size,
        fast_mem_fusion_init=args.fast_mem_fusion_init,
    )

    # H7 rotary fp32 fix — snapshot before bf16 cast
    _rope_snapshot = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass

    apply_mem_space_to_model(model, ms_cfg, layer_indices=None)
    model.to(device=device, dtype=dtype)

    # Restore rotary buffers in fp32
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if _rope_snapshot:
            logger.info("H7 fix v2 applied: rotary buffers %s in fp32",
                        sorted(_rope_snapshot.keys()))
    except AttributeError:
        logger.warning("H7 fix v2: rotary_emb not accessible")

    # Warm-start from existing adapter
    if args.init_checkpoint and os.path.isfile(args.init_checkpoint):
        logger.info("Loading warm-start adapter from %s", args.init_checkpoint)
        ckpt = torch.load(args.init_checkpoint, map_location=device, weights_only=False)
        if isinstance(ckpt, dict):
            if "model_state_dict" in ckpt:
                state_dict = ckpt["model_state_dict"]
            elif "state_dict" in ckpt:
                state_dict = ckpt["state_dict"]
            else:
                state_dict = ckpt
        else:
            state_dict = ckpt
        cleaned = {}
        for k, v in state_dict.items():
            cleaned[k[7:] if k.startswith("module.") else k] = v
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        logger.info("init_checkpoint loaded: %d keys, missing=%d, unexpected=%d",
                    len(cleaned), len(missing), len(unexpected))
    elif args.init_checkpoint:
        logger.warning("init_checkpoint=%s not found — random init", args.init_checkpoint)

    return model


# --------------------------------------------------------------------------- #
# Training steps
# --------------------------------------------------------------------------- #


def dolmino_train_step(
    model: torch.nn.Module,
    context_chunks: List[torch.Tensor],
    target_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute one Dolmino CPT step with FastMem state management.

    FastMem state persists (detached) across context chunks within the same
    document, but resets at sample boundaries.
    """
    # Reset everything at sample boundary
    _reset_banks(model)
    _reset_fast_mem(model)

    # Stream context chunks through memory (no gradient)
    with torch.no_grad():
        for ctx in context_chunks:
            ctx_input = ctx.unsqueeze(0).to(device)
            model(input_ids=ctx_input, use_cache=False)
            # Detach fast_mem state between chunks (keep value, break grad)
            _detach_fast_mem(model)

    # Detach memory banks (break gradient graph before target)
    _detach_banks(model)
    _detach_fast_mem(model)

    # Forward target chunk with gradient
    target_input = target_ids.unsqueeze(0).to(device)
    out = model(input_ids=target_input, labels=target_input, use_cache=False)

    lm_loss = out.loss
    aux_loss = _collect_aux_loss(model, device)

    return lm_loss, aux_loss


def babilong_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stream BABILong sample with FastMem state management."""
    _reset_banks(model)
    _reset_fast_mem(model)

    total_len = input_ids.shape[1]
    n_chunks = max(1, math.ceil(total_len / chunk_size))

    if n_chunks == 1:
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        return out.loss, _collect_aux_loss(model, device)

    pieces_in = list(input_ids[0].split(chunk_size))
    pieces_lbl = list(labels[0].split(chunk_size))

    with torch.no_grad():
        for ci in pieces_in[:-1]:
            model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)
            _detach_fast_mem(model)

    # Detach before the gradient-bearing last chunk
    _detach_banks(model)
    _detach_fast_mem(model)

    last_in = pieces_in[-1].unsqueeze(0).to(device)
    last_lbl = pieces_lbl[-1].unsqueeze(0).to(device)
    out = model(input_ids=last_in, labels=last_lbl, use_cache=False)

    return out.loss, _collect_aux_loss(model, device)


# --------------------------------------------------------------------------- #
# Quick BABILong eval
# --------------------------------------------------------------------------- #


@torch.no_grad()
def quick_eval_babilong(
    model: torch.nn.Module,
    tokenizer,
    device: torch.device,
    chunk_size: int,
    n_samples: int = 50,
    dataset_name: str = "RMT-team/babilong",
    task: str = "qa1",
    length: str = "1k",
) -> float:
    """Quick BABILong qa1 accuracy check."""
    model.eval()
    try:
        import datasets
        data = datasets.load_dataset(dataset_name, length)
        split = data[task]
    except Exception as e:
        logger.warning("quick_eval_babilong failed to load data: %s", e)
        model.train()
        return -1.0

    from babilong.prompts import DEFAULT_PROMPTS, DEFAULT_TEMPLATE, get_formatted_input

    correct = 0
    total = 0
    indices = list(range(min(n_samples, len(split))))

    for idx in indices:
        sample = split[idx]
        prompt_cfg = DEFAULT_PROMPTS[task]
        input_text = get_formatted_input(
            sample["input"], sample["question"],
            prompt_cfg["examples"], prompt_cfg["instruction"],
            prompt_cfg["post_prompt"], template=DEFAULT_TEMPLATE,
        )
        input_ids = tokenizer.encode(input_text, add_special_tokens=True,
                                     return_tensors="pt").to(device)

        _reset_banks(model)
        _reset_fast_mem(model)
        total_len = input_ids.shape[1]
        n_chunks = max(1, math.ceil(total_len / chunk_size))

        if n_chunks > 1:
            pieces = list(input_ids[0].split(chunk_size))
            for ci in pieces[:-1]:
                model(input_ids=ci.unsqueeze(0), use_cache=False)
                _detach_fast_mem(model)
            last_chunk = pieces[-1].unsqueeze(0)
        else:
            last_chunk = input_ids

        gen_ids = []
        cur_input = last_chunk
        for _ in range(20):
            out = model(input_ids=cur_input, use_cache=False)
            next_id = out.logits[0, -1].argmax().item()
            gen_ids.append(next_id)
            if next_id == tokenizer.eos_token_id:
                break
            cur_input = torch.tensor([[next_id]], device=device)

        generated = tokenizer.decode(gen_ids, skip_special_tokens=True).strip().lower()
        target = str(sample["target"]).strip().lower()

        if target in generated or generated in target:
            correct += 1
        total += 1

    accuracy = (correct / total * 100) if total > 0 else 0.0
    model.train()
    return accuracy


# --------------------------------------------------------------------------- #
# Adapter checkpoint save
# --------------------------------------------------------------------------- #


def _save_adapter(model, args, step: int, final: bool = False) -> None:
    """Save mem_space + fast_mem adapter weights + config."""
    fragments = (
        "selector", "gate_param", "slot_output_gate",
        "slot_to_hidden", "hidden_to_slot", "memory_bank",
        "gate_proj_new", "gate_proj_mem", "gate_bias",
        "l3_pool", "l2_compressor",
        "fast_mem",  # FastMem parameters
    )

    root = model.module if isinstance(model, DDP) else model
    state = {
        k: v.detach().cpu()
        for k, v in root.state_dict().items()
        if any(frag in k for frag in fragments)
    }

    if final:
        ckpt_path = os.path.join(args.output_dir, "mem_space_adapter.pt")
    else:
        ckpt_path = os.path.join(args.output_dir, f"mem_space_adapter_step{step:06d}.pt")
    torch.save(state, ckpt_path)
    logger.info("Saved adapter ckpt: %s (%d keys)", ckpt_path, len(state))

    cfg_path = os.path.join(args.output_dir, "adapter_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "selector_dim": args.selector_dim,
            "writeback_gate_max": args.writeback_gate_max,
            "writeback_warmup_steps": args.writeback_warmup_steps,
            "load_balance_weight": args.load_balance_weight,
            "entropy_aux_weight": args.entropy_aux_weight,
            "selector_temperature": args.selector_temperature,
            "key_repulsion_weight": args.key_repulsion_weight,
            "key_repulsion_threshold": args.key_repulsion_threshold,
            "peak_routing_weight": args.peak_routing_weight,
            "slot_value_norm_cap": args.slot_value_norm_cap,
            "slot_init": args.slot_init,
            "slot_init_noise": args.slot_init_noise,
            "shared_memory_bank": args.shared_memory_bank,
            "unfreeze_hidden_to_slot": args.unfreeze_hidden_to_slot,
            "swa_window": args.swa_window,
            "use_dual_gate": args.use_dual_gate,
            "input_bias_init": args.input_bias_init,
            "forget_bias_init": args.forget_bias_init,
            "dual_gate_tanh_new": args.dual_gate_tanh_new,
            "use_l3_summary": args.use_l3_summary,
            "l3_n_summary": args.l3_n_summary,
            "l3_n_layers": args.l3_n_layers,
            "l3_n_heads": args.l3_n_heads,
            "disable_l1_inject": args.disable_l1_inject,
            "use_replace_writeback": args.use_replace_writeback,
            "num_global_slots": args.num_global_slots,
            "global_slot_forget_bias": args.global_slot_forget_bias,
            "global_slot_input_gate_only": args.global_slot_input_gate_only,
            "gradient_checkpointing": args.gradient_checkpointing,
            # FastMem config
            "use_fast_mem": args.use_fast_mem,
            "fast_mem_heads": args.fast_mem_heads,
            "fast_mem_d_state": args.fast_mem_d_state,
            "fast_mem_chunk_size": args.fast_mem_chunk_size,
            "fast_mem_fusion_init": args.fast_mem_fusion_init,
            "fast_mem_lr_mult": args.fast_mem_lr_mult,
            # Training metadata
            "curriculum": args.curriculum,
            "lr": args.lr,
            "total_steps": args.total_steps,
            "babilong_mix_fraction": args.babilong_mix_fraction,
            "babilong_tasks": args.babilong_tasks,
            "babilong_lengths": args.babilong_lengths,
            "dolmino_path": args.dolmino_path,
            "init_checkpoint": args.init_checkpoint,
            "step_at_save": step,
            "final": final,
        }, f, indent=2)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    # Curriculum scheduler
    curriculum = CurriculumScheduler(args.curriculum)

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        if _WANDB_AVAILABLE and args.wandb_project:
            wandb_name = args.wandb_run_name or os.path.basename(args.output_dir)
            wandb.init(
                project=args.wandb_project,
                name=wandb_name,
                config=vars(args),
                dir=args.output_dir,
            )
            logger.info("Wandb initialized: project=%s run=%s", args.wandb_project, wandb_name)
        logger.info(
            "FastMem CPT | model=%s | dolmino=%s | curriculum=%s | "
            "fast_mem_heads=%d d_state=%d chunk=%d fusion_init=%.1f lr_mult=%.1f | "
            "total_steps=%d | grad_accum=%d | lr=%.2e | world_size=%d",
            args.model_path, args.dolmino_path, args.curriculum,
            args.fast_mem_heads, args.fast_mem_d_state, args.fast_mem_chunk_size,
            args.fast_mem_fusion_init, args.fast_mem_lr_mult,
            args.total_steps, args.gradient_accumulation_steps, args.lr, world_size,
        )

    # --- tokenizer --- #
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model --- #
    model = build_model(args, device, dtype)
    _freeze_backbone(model)

    if is_main(rank):
        n_base = sum(p.numel() for p in _mem_space_params(model))
        n_fast = sum(p.numel() for p in _fast_mem_params(model))
        logger.info("Trainable: mem_space=%.2fM, fast_mem=%.2fM, total=%.2fM",
                    n_base / 1e6, n_fast / 1e6, (n_base + n_fast) / 1e6)

    # DDP wrap
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # --- Dolmino dataset --- #
    dolmino_ds = DolminoCurriculumDataset(
        data_path=args.dolmino_path,
        chunk_size=args.chunk_size,
        n_context=curriculum.get_n_ctx(0),
        rank=rank,
        world_size=world_size,
        seed=args.seed,
    )
    dolmino_loader = DataLoader(
        dolmino_ds, batch_size=None,
        num_workers=args.num_workers, pin_memory=False,
    )
    dolmino_iter = iter(dolmino_loader)

    # --- BABILong dataset (for mix) --- #
    babilong_iter = None
    if args.babilong_mix_fraction > 0.0:
        babilong_tasks = [t.strip() for t in args.babilong_tasks.split(",") if t.strip()]
        babilong_lengths = [l.strip() for l in args.babilong_lengths.split(",") if l.strip()]

        if world_size > 1:
            if rank == 0:
                logger.info("[rank 0] Pre-fetching BABILong cache...")
                try:
                    import datasets as _hfds
                    for _length in babilong_lengths:
                        try:
                            _data = _hfds.load_dataset(args.babilong_dataset, _length)
                            for _task in babilong_tasks:
                                try:
                                    _ = _data[_task]
                                except Exception:
                                    pass
                        except Exception:
                            pass
                except Exception:
                    pass
            dist.barrier()

        babilong_ds = BABILongTrainDataset(
            tokenizer=tokenizer,
            dataset_name=args.babilong_dataset,
            tasks=babilong_tasks,
            lengths=babilong_lengths,
            max_seq_len=args.chunk_size * 4,
            seed=args.seed + rank,
            use_chat_template=args.use_chat_template,
        )
        babilong_loader = DataLoader(
            babilong_ds, batch_size=1,
            num_workers=args.num_workers, collate_fn=babilong_collate_fn,
        )
        babilong_iter = iter(babilong_loader)

    mix_rng = random.Random(args.seed + rank)

    # --- optimizer with separate FastMem param group --- #
    raw_model = model.module if isinstance(model, DDP) else model
    base_params = _mem_space_params(raw_model)
    fast_params = _fast_mem_params(raw_model)

    if not base_params and not fast_params:
        raise RuntimeError("No trainable params found.")

    param_groups = []
    if base_params:
        param_groups.append({
            "params": base_params,
            "lr": args.lr,
            "name": "mem_space",
        })
    if fast_params:
        param_groups.append({
            "params": fast_params,
            "lr": args.lr * args.fast_mem_lr_mult,
            "name": "fast_mem",
        })

    if is_main(rank):
        logger.info("Optimizer: %d base params, %d fast_mem params (%.1fx LR)",
                    len(base_params), len(fast_params), args.fast_mem_lr_mult)

    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.0, betas=(0.9, 0.95))

    # --- training loop --- #
    model.train()
    global_step = args.start_step
    micro_step = 0
    n_dolmino = 0
    n_babilong = 0
    n_nonfinite = 0
    accum_lm_loss = 0.0
    accum_aux_loss = 0.0
    t0 = time.time()
    grad_accum = args.gradient_accumulation_steps

    while global_step < args.total_steps:
        # Update curriculum
        current_n_ctx = curriculum.get_n_ctx(global_step)
        dolmino_ds.set_n_context(current_n_ctx)

        # Update learning rate (cosine with warmup)
        lr_base = cosine_lr_schedule(global_step, args.total_steps, args.warmup_steps,
                                     args.lr)
        for pg in optimizer.param_groups:
            if pg.get("name") == "fast_mem":
                pg["lr"] = lr_base * args.fast_mem_lr_mult
            else:
                pg["lr"] = lr_base

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        step_lm_loss = 0.0
        step_aux_loss = 0.0
        step_valid_micros = 0

        for micro in range(grad_accum):
            use_babilong = (babilong_iter is not None and
                            mix_rng.random() < args.babilong_mix_fraction)

            if use_babilong:
                try:
                    batch = next(babilong_iter)
                except StopIteration:
                    babilong_iter = iter(babilong_loader)
                    batch = next(babilong_iter)

                input_ids = batch["input_ids"].to(device, non_blocking=True)
                labels = batch["labels"].to(device, non_blocking=True)

                lm_loss, aux_loss = babilong_train_step(
                    model, input_ids, labels, args.chunk_size, device
                )
                n_babilong += 1
            else:
                try:
                    sample = next(dolmino_iter)
                except StopIteration:
                    dolmino_iter = iter(dolmino_loader)
                    sample = next(dolmino_iter)

                context_chunks = sample["context_chunks"]
                target_ids = sample["target_ids"]

                lm_loss, aux_loss = dolmino_train_step(
                    model, context_chunks, target_ids, device
                )
                n_dolmino += 1

            if lm_loss is None or not torch.isfinite(lm_loss + aux_loss):
                n_nonfinite += 1
                continue

            loss = (lm_loss + aux_loss) / grad_accum
            loss.backward()

            step_lm_loss += lm_loss.item()
            step_aux_loss += aux_loss.item()
            step_valid_micros += 1

        # Optimizer step
        if step_valid_micros > 0:
            _grad_root = model.module if isinstance(model, DDP) else model
            for _n, _p in _grad_root.named_parameters():
                if _p.grad is not None and ("slot_to_hidden" in _n or "hidden_to_slot" in _n):
                    torch.nn.utils.clip_grad_norm_([_p], args.proj_grad_clip)
            # Clip all trainable params
            all_trainable = base_params + fast_params
            torch.nn.utils.clip_grad_norm_(all_trainable, args.grad_clip)

            optimizer.step()

        _step_counters_inc(model)
        global_step += 1

        # Logging
        if is_main(rank) and (global_step % args.log_interval == 0):
            avg_lm = step_lm_loss / max(1, step_valid_micros)
            avg_aux = step_aux_loss / max(1, step_valid_micros)
            elapsed = time.time() - t0
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
            fusion_val = _get_fusion_gate_value(raw_model)
            logger.info(
                "[step %d/%d] lm=%.4f aux=%.4f lr=%.2e n_ctx=%d "
                "fusion_gate=%.4f dolmino=%d babi=%d nf=%d speed=%.2f s/s",
                global_step, args.total_steps, avg_lm, avg_aux, lr_base,
                current_n_ctx, fusion_val, n_dolmino, n_babilong, n_nonfinite,
                steps_per_sec,
            )
            if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
                wandb.log({
                    "train/lm_loss": avg_lm,
                    "train/aux_loss": avg_aux,
                    "train/lr": lr_base,
                    "train/fast_mem_lr": lr_base * args.fast_mem_lr_mult,
                    "train/n_ctx": current_n_ctx,
                    "train/fusion_gate_sigmoid": fusion_val,
                    "train/speed_steps_s": steps_per_sec,
                    "train/n_nonfinite": n_nonfinite,
                    "train/dolmino_count": n_dolmino,
                    "train/babilong_count": n_babilong,
                }, step=global_step)

        # Save checkpoint
        if (args.save_interval > 0
                and global_step % args.save_interval == 0
                and global_step < args.total_steps
                and is_main(rank)):
            _save_adapter(model, args, global_step)

        # Quick eval
        if (args.eval_interval > 0
                and global_step % args.eval_interval == 0
                and is_main(rank)):
            acc = quick_eval_babilong(
                model.module if isinstance(model, DDP) else model,
                tokenizer, device, args.chunk_size,
                n_samples=args.eval_samples,
            )
            logger.info("[eval step %d] BABILong qa1@1k accuracy: %.1f%%", global_step, acc)
            if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
                wandb.log({"eval/babilong_qa1_acc": acc}, step=global_step)
            model.train()

    # Final save
    if is_main(rank):
        _save_adapter(model, args, global_step, final=True)
        if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
            wandb.finish()
        logger.info(
            "Training complete: steps=%d dolmino=%d babilong=%d non-finite=%d "
            "time=%.1f min",
            global_step, n_dolmino, n_babilong, n_nonfinite,
            (time.time() - t0) / 60,
        )

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
