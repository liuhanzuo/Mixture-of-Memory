#!/usr/bin/env python3
"""Dolmino Continued Pretraining (CPT) for Memory-Space adapter.

Trains the mem_space memory adapter on diverse text (Dolmino, 0.47B tokens)
with curriculum learning, following the MemoryLLM methodology. The key idea:
progressively increase the number of context chunks the model must compress
into memory before predicting the next chunk (NTP on target).

Training step (Dolmino):
    1. _reset_banks(model) — fresh memory per document group
    2. For each of N context chunks: model(ctx, use_cache=False) with no_grad
       → memory accumulates in banks
    3. _detach_banks(model) — break gradient graph from context forward passes
    4. out = model(target_ids, labels=target_ids, use_cache=False) — gradient here
    5. loss = (out.loss + aux_loss) / grad_accum_steps
    6. loss.backward()

Mixed with BABILong SFT (--babilong_mix_fraction, default 15%) to maintain
task-specific retrieval ability.

Curriculum schedule (default):
    Step 0:      n_ctx=1   (2K effective context)
    Step 10000:  n_ctx=2   (3K effective context)
    Step 15000:  n_ctx=4   (5K effective context)
    Step 25000:  n_ctx=8   (9K effective context)
    Step 40000:  n_ctx=16  (17K effective context)

Fork of ``scripts/train_mem_space_babilong.py``.
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
from contextlib import nullcontext
from datetime import timedelta
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
    Example: "0:1,10000:2,15000:4,25000:8,40000:16"

    At step S, the active n_ctx is the largest entry whose step <= S.
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
        """Return the n_ctx value active at the given step."""
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
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(minutes=30))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# Mem-space helpers (from train_mem_space_babilong.py)
# --------------------------------------------------------------------------- #


def _mem_space_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
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


def _freeze_backbone(model: torch.nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = False
    for p in _mem_space_params(model):
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
        l3_pool._prev_summary = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None
    # Reset FastMem states (document boundary)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if mem_layers:
        for w in mem_layers:
            if hasattr(w, "_fast_mem_state"):
                w._fast_mem_state = None


def _detach_banks(model: torch.nn.Module) -> None:
    """Detach memory bank tensors to break gradient graph from context passes."""
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.detach_()
    else:
        mem_layers = getattr(root, "_mem_space_layers", None)
        if not mem_layers:
            return
        for w in mem_layers:
            w.memory_bank.detach_()
    # Detach FastMem states (keep state but break grad graph)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if mem_layers:
        for w in mem_layers:
            if hasattr(w, "_fast_mem_state") and w._fast_mem_state is not None:
                w._fast_mem_state = w._fast_mem_state.detach()


def _collect_top1_sim(model: torch.nn.Module) -> float:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return 0.0
    vals = [getattr(w, "_last_top1_sim", 0.0) for w in mem_layers]
    return vals[0] if vals else 0.0


def _collect_aux_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    total = torch.zeros((), device=device)
    if not mem_layers:
        return total
    for w in mem_layers:
        for key in ("load_balance", "entropy", "key_repulsion", "weight_ortho", "l3_diversity", "q_multi_diversity"):
            v = w.last_aux_losses.get(key)
            if v is not None:
                total = total + v
    return total


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory-Space Dolmino CPT (curriculum continued pretraining)",
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
                   help="Token count per chunk (must match Dolmino preprocessing).")

    # Curriculum
    p.add_argument("--curriculum", type=str,
                   default="0:1,10000:2,15000:4,25000:8,40000:16",
                   help="Curriculum schedule: 'step:n_ctx,step:n_ctx,...'")

    # BABILong mix
    p.add_argument("--babilong_mix_fraction", type=float, default=0.15,
                   help="Probability of sampling a BABILong step vs Dolmino.")
    p.add_argument("--babilong_dataset", type=str, default="RMT-team/babilong")
    p.add_argument("--babilong_tasks", type=str, default="qa1,qa2,qa5",
                   help="Comma-separated BABILong tasks for the mix.")
    p.add_argument("--babilong_lengths", type=str, default="0k,1k,2k,4k",
                   help="Comma-separated BABILong lengths for the mix.")
    p.add_argument("--use_chat_template", action="store_true", default=False,
                   help="Use chat template for BABILong (for Instruct models).")

    # Training
    p.add_argument("--total_steps", type=int, default=60000)
    p.add_argument("--lr", type=float, default=5e-6)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--start_step", type=int, default=0,
                   help="Resume from this step count (for curriculum and LR schedule).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proj_grad_clip", type=float, default=0.1)
    p.add_argument("--bptt_window", type=int, default=2,
                   help="Windowed BPTT: accumulate this many context chunks into "
                        "one gradient graph before backward+detach. window=1 "
                        "recovers the old per-chunk detach behavior; window=2 "
                        "(default) lets gradients flow across one chunk boundary "
                        "so 'chunk_i writes info that helps chunk_{i+1}' gets a "
                        "credit-assignment signal. Higher window ≈ more VRAM "
                        "(roughly ×window).")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-rank batch size (1 for Dolmino, BABILong is always 1).")

    # Logging / saving / eval
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=5000)
    p.add_argument("--eval_interval", type=int, default=2000,
                   help="Evaluate BABILong qa1 accuracy every N steps.")
    p.add_argument("--eval_samples", type=int, default=10,
                   help="Number of BABILong samples for quick eval.")

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
    p.add_argument("--l3_diversity_weight", type=float, default=0.1)
    p.add_argument("--l3_diversity_threshold", type=float, default=0.5)
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

    # FastMem (Gated Delta Rule continuous memory)
    p.add_argument("--use_fast_mem", action="store_true", default=False)
    p.add_argument("--fast_mem_num_heads", type=int, default=4)
    p.add_argument("--fast_mem_d_state", type=int, default=128)

    # P1-v2: break gate freeze deadlock
    p.add_argument("--no_detach_slots_in_selector", action="store_true", default=False)
    p.add_argument("--no_slot_delta_clip", action="store_true", default=False)
    p.add_argument("--inject_gate_bias_init", type=float, default=-0.1523)
    p.add_argument("--routing_pool_mode", type=str, default="max_pool",
                   choices=["max_pool", "chunk_query", "multi_query"])
    p.add_argument("--multi_query_tau", type=float, default=1.0,
                   help="logsumexp temperature for multi_query routing aggregation")

    # v6/v7 writeback (disabled by default for CPT)
    p.add_argument("--use_replace_writeback", action="store_true", default=False)
    p.add_argument("--num_global_slots", type=int, default=0)
    p.add_argument("--global_slot_forget_bias", type=float, default=1.0)
    p.add_argument("--global_slot_input_gate_only", action="store_true", default=False)

    # Activation-memory reduction
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # Misc
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0,
                   help="DataLoader workers (0 = main process, recommended for "
                        "Arrow mmap datasets).")

    # Wandb
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory",
                   help="Wandb project name (set to '' to disable).")
    p.add_argument("--wandb_run_id", type=str, default=None,
                   help="Wandb run ID to resume (avoids creating duplicate runs)")
    p.add_argument("--wandb_run_name", type=str, default=None,
                   help="Wandb run name (auto-generated if None).")

    return p.parse_args()


def merge_adapter_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    """Inherit mem-space hyperparams from --init_adapter_config when not explicit."""
    if not args.init_adapter_config:
        return args
    if not os.path.isfile(args.init_adapter_config):
        logger.warning("init_adapter_config %s not found — using CLI defaults",
                       args.init_adapter_config)
        return args

    explicit = {a.lstrip("-").split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    with open(args.init_adapter_config, "r") as f:
        cfg = json.load(f)

    cfg_to_attr = {
        "num_slots": "num_slots", "top_k": "top_k",
        "selector_dim": "selector_dim", "writeback_gate_max": "writeback_gate_max",
        "writeback_warmup_steps": "writeback_warmup_steps",
        "load_balance_weight": "load_balance_weight",
        "entropy_aux_weight": "entropy_aux_weight",
        "selector_temperature": "selector_temperature",
        "key_repulsion_weight": "key_repulsion_weight",
        "key_repulsion_threshold": "key_repulsion_threshold",
        "l3_diversity_weight": "l3_diversity_weight",
        "l3_diversity_threshold": "l3_diversity_threshold",
        "peak_routing_weight": "peak_routing_weight",
        "slot_value_norm_cap": "slot_value_norm_cap",
        "slot_init": "slot_init", "slot_init_noise": "slot_init_noise",
        "unfreeze_hidden_to_slot": "unfreeze_hidden_to_slot",
        "shared_memory_bank": "shared_memory_bank", "swa_window": "swa_window",
    }
    inherited = []
    for k_json, attr in cfg_to_attr.items():
        if k_json not in cfg:
            continue
        if attr in explicit:
            continue
        setattr(args, attr, cfg[k_json])
        inherited.append(f"{k_json}={cfg[k_json]}")
    if inherited:
        logger.info("Inherited from adapter_config: %s", ", ".join(inherited))
    return args


# --------------------------------------------------------------------------- #
# Model build
# --------------------------------------------------------------------------- #


def build_model(args, device, dtype) -> torch.nn.Module:
    """Load Llama + patch with mem_space + optionally warm-start adapter."""
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
        l3_diversity_weight=args.l3_diversity_weight,
        l3_diversity_threshold=args.l3_diversity_threshold,
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
        use_fast_mem=args.use_fast_mem,
        fast_mem_num_heads=args.fast_mem_num_heads,
        fast_mem_d_state=args.fast_mem_d_state,
        no_detach_slots_in_selector=args.no_detach_slots_in_selector,
        no_slot_delta_clip=args.no_slot_delta_clip,
        inject_gate_bias_init=args.inject_gate_bias_init,
        routing_pool_mode=args.routing_pool_mode,
        multi_query_tau=args.multi_query_tau,
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
# Dolmino training step
# --------------------------------------------------------------------------- #


def dolmino_train_step(
    model: torch.nn.Module,
    context_chunks: List[torch.Tensor],
    target_ids: torch.Tensor,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Execute one Dolmino CPT step:
    1. Reset banks
    2. Stream context chunks through model (no grad) → memory accumulates
    3. Detach banks (break gradient graph)
    4. Forward target chunk with grad → compute NTP loss

    Returns: (lm_loss, aux_loss)
    """
    _reset_banks(model)

    # Stream context chunks through memory (no gradient)
    with torch.no_grad():
        for ctx in context_chunks:
            ctx_input = ctx.unsqueeze(0).to(device)  # [1, chunk_size]
            model(input_ids=ctx_input, use_cache=False)

    # Detach memory banks to prevent gradient flow through context passes
    _detach_banks(model)

    # Forward target chunk with gradient
    target_input = target_ids.unsqueeze(0).to(device)  # [1, chunk_size]
    out = model(input_ids=target_input, labels=target_input, use_cache=False)

    lm_loss = out.loss
    aux_loss = _collect_aux_loss(model, device)

    return lm_loss, aux_loss


def dolmino_train_step_tbptt(
    model: torch.nn.Module,
    context_chunks: List[torch.Tensor],
    target_ids: torch.Tensor,
    device: torch.device,
    grad_accum: int = 1,
    bptt_window: int = 2,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Windowed-BPTT Dolmino CPT step: every chunk gets gradient, and gradients
    flow ACROSS chunk boundaries within a window of ``bptt_window`` chunks.

    Unlike dolmino_train_step (which processes context with no_grad), this
    version gives every chunk its own forward and contributes to the loss.
    Memory state carries forward across chunks. The key difference vs the old
    per-chunk-detach implementation: instead of ``backward()`` + ``_detach_banks``
    after EVERY chunk (which severed the long-range credit-assignment signal
    "chunk_i writes info that helps chunk_{i+k}"), we accumulate ``bptt_window``
    chunks' losses into ONE connected autograd graph (memory bank reads stay
    attached to prior chunks' writes), call ``backward()`` once at the window
    boundary, and only then ``_detach_banks``.

    The target chunk is folded into the final window so its loss can back-
    propagate into the writes that produced the memory it reads.

    Peak VRAM scales ~linearly with ``bptt_window`` (window=1 recovers the old
    per-chunk peak; window=2 ≈ ×2). gradient_checkpointing keeps this tractable
    on H20 (97.8 GiB) at window=2.

    Returns: (total_lm * grad_accum, total_aux * grad_accum) — semantics
    unchanged from the prior implementation.
    """
    _reset_banks(model)

    n_chunks = len(context_chunks) + 1
    scale = n_chunks * grad_accum
    total_lm = torch.zeros((), device=device)
    total_aux = torch.zeros((), device=device)

    bptt_window = max(1, int(bptt_window))

    # Build the full chunk list: all context chunks followed by the target.
    # Each entry: (input_ids_for_forward, is_target).
    all_inputs = [(ctx, False) for ctx in context_chunks]
    all_inputs.append((target_ids, True))

    # Accumulator for the current window's loss (keeps the autograd graph
    # connecting consecutive chunks alive until we backward at the boundary).
    window_loss = None  # type: Optional[torch.Tensor]

    for i, (chunk_ids, _is_target) in enumerate(all_inputs):
        chunk_input = chunk_ids.unsqueeze(0).to(device)
        out = model(input_ids=chunk_input, labels=chunk_input, use_cache=False)
        chunk_lm = out.loss / scale
        chunk_aux = _collect_aux_loss(model, device) / scale

        # Add this chunk's loss to the live window graph (do NOT backward yet:
        # the next chunk's memory read must stay connected to this chunk's
        # write graph for true cross-chunk BPTT).
        step_loss = chunk_lm + chunk_aux
        window_loss = step_loss if window_loss is None else (window_loss + step_loss)

        total_lm = total_lm + chunk_lm.detach()
        total_aux = total_aux + chunk_aux.detach()

        # Window boundary: either we've packed ``bptt_window`` chunks since the
        # last backward, or this is the final (target) chunk.
        is_last = (i == len(all_inputs) - 1)
        at_window_edge = ((i + 1) % bptt_window == 0)
        if at_window_edge or is_last:
            window_loss.backward()
            window_loss = None
            # Detach the memory bank so the NEXT window starts a fresh graph
            # (bounds VRAM and BPTT depth to one window). No detach after the
            # final chunk is necessary, but it is harmless and keeps the bank
            # in a clean detached state for any post-step inspection.
            _detach_banks(model)

    return total_lm * grad_accum, total_aux * grad_accum


# --------------------------------------------------------------------------- #
# BABILong training step (chunked, from babilong script)
# --------------------------------------------------------------------------- #


def babilong_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stream BABILong sample through memory in chunks, gradient on last chunk."""
    _reset_banks(model)
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

    # Detach before the gradient-bearing last chunk
    _detach_banks(model)

    last_in = pieces_in[-1].unsqueeze(0).to(device)
    last_lbl = pieces_lbl[-1].unsqueeze(0).to(device)
    out = model(input_ids=last_in, labels=last_lbl, use_cache=False)

    return out.loss, _collect_aux_loss(model, device)


# --------------------------------------------------------------------------- #
# Quick BABILong eval (qa1 accuracy)
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
    """Run quick BABILong qa1 accuracy check (greedy generate, exact match).

    Returns accuracy (0-100).
    """
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

    with torch.no_grad():
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
            total_len = input_ids.shape[1]
            n_chunks = max(1, math.ceil(total_len / chunk_size))

            if n_chunks > 1:
                pieces = list(input_ids[0].split(chunk_size))
                for ci in pieces[:-1]:
                    model(input_ids=ci.unsqueeze(0), use_cache=False)
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
    """Save mem_space adapter weights + config."""
    fragments = (
        "selector", "gate_param", "slot_output_gate",
        "slot_to_hidden", "hidden_to_slot", "memory_bank",
        "gate_proj_new", "gate_proj_mem", "gate_bias",
        "l3_pool", "l2_compressor",
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
    args = merge_adapter_config_into_args(args)
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
        # Wandb init
        if _WANDB_AVAILABLE and args.wandb_project:
            wandb_name = args.wandb_run_name or os.path.basename(args.output_dir)
            wandb_kwargs = dict(
                project=args.wandb_project,
                name=wandb_name,
                config=vars(args),
                dir=args.output_dir,
            )
            if args.wandb_run_id:
                wandb_kwargs["id"] = args.wandb_run_id
                wandb_kwargs["resume"] = "allow"
            wandb.init(**wandb_kwargs)
            logger.info("Wandb initialized: project=%s run=%s", args.wandb_project, wandb_name)
        logger.info(
            "Dolmino CPT | model=%s | dolmino=%s | curriculum=%s | "
            "babilong_mix=%.2f | total_steps=%d | grad_accum=%d | lr=%.2e | "
            "world_size=%d",
            args.model_path, args.dolmino_path, args.curriculum,
            args.babilong_mix_fraction, args.total_steps,
            args.gradient_accumulation_steps, args.lr, world_size,
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
        n_trainable = sum(p.numel() for p in _mem_space_params(model))
        logger.info("mem_space trainable: %.2fM params", n_trainable / 1e6)

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
        dolmino_ds, batch_size=None,  # dataset yields single samples
        num_workers=args.num_workers, pin_memory=False,
    )
    dolmino_iter = iter(dolmino_loader)

    # --- BABILong dataset (for mix) --- #
    babilong_iter = None
    if args.babilong_mix_fraction > 0.0:
        babilong_tasks = [t.strip() for t in args.babilong_tasks.split(",") if t.strip()]
        babilong_lengths = [l.strip() for l in args.babilong_lengths.split(",") if l.strip()]

        # Rank-0 prefetch for HF datasets
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
            max_seq_len=args.chunk_size * 4,  # up to 4 chunks for BABILong
            seed=args.seed + rank,
            use_chat_template=args.use_chat_template,
        )
        babilong_loader = DataLoader(
            babilong_ds, batch_size=1,
            num_workers=args.num_workers, collate_fn=babilong_collate_fn,
        )
        babilong_iter = iter(babilong_loader)

    mix_rng = random.Random(args.seed + rank)

    # --- optimizer + LR scheduler --- #
    trainable = _mem_space_params(
        model.module if isinstance(model, DDP) else model
    )
    if not trainable:
        raise RuntimeError("No mem_space trainable params found.")
    if is_main(rank):
        logger.info("Optimizer: %d trainable parameter tensors", len(trainable))

    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0,
                                  betas=(0.9, 0.95))

    # --- training loop --- #
    model.train()
    global_step = args.start_step if hasattr(args, 'start_step') and args.start_step else 0
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
        lr = cosine_lr_schedule(global_step, args.total_steps, args.warmup_steps,
                                args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        step_lm_loss = 0.0
        step_aux_loss = 0.0
        step_valid_micros = 0

        for micro in range(grad_accum):
            # Always suppress DDP auto-sync: TBPTT does multiple backward()
            # calls per micro-step, and Dolmino vs BABILong have different
            # counts. We manually allreduce once after the accumulation loop.
            sync_ctx = model.no_sync() if (
                world_size > 1 and isinstance(model, DDP)
            ) else nullcontext()

            with sync_ctx:
                # Decide: Dolmino or BABILong
                use_babilong = (babilong_iter is not None and
                                mix_rng.random() < args.babilong_mix_fraction)

                if use_babilong:
                    # BABILong step
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
                    # Dolmino step
                    try:
                        sample = next(dolmino_iter)
                    except StopIteration:
                        dolmino_iter = iter(dolmino_loader)
                        sample = next(dolmino_iter)

                    context_chunks = sample["context_chunks"]
                    target_ids = sample["target_ids"]

                    lm_loss, aux_loss = dolmino_train_step_tbptt(
                        model, context_chunks, target_ids, device,
                        grad_accum=grad_accum,
                    )
                    n_dolmino += 1

                # Check for non-finite
                if lm_loss is None or not torch.isfinite(lm_loss + aux_loss):
                    n_nonfinite += 1
                    _zero = torch.zeros(1, device=device, requires_grad=False)
                    _zero = _zero + 0.0 * next(p for p in model.parameters() if p.requires_grad)
                    _zero.backward()
                    continue

                # TBPTT already called backward() inside; for BABILong we still
                # need the outer backward.
                if use_babilong:
                    loss = (lm_loss + aux_loss) / grad_accum
                    loss.backward()

                step_lm_loss += lm_loss.item()
                step_aux_loss += aux_loss.item()
                step_valid_micros += 1

        # Manual gradient allreduce (since we always use no_sync above)
        if world_size > 1 and isinstance(model, DDP) and step_valid_micros > 0:
            for p in trainable:
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # Optimizer step (only if we got at least one valid micro-step)
        if step_valid_micros > 0:
            # Per-projection grad clip
            _grad_root = model.module if isinstance(model, DDP) else model
            for _n, _p in _grad_root.named_parameters():
                if _p.grad is not None and ("slot_to_hidden" in _n or "hidden_to_slot" in _n):
                    torch.nn.utils.clip_grad_norm_([_p], args.proj_grad_clip)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

            optimizer.step()

        _step_counters_inc(model)
        global_step += 1

        # Logging
        if is_main(rank) and (global_step % args.log_interval == 0):
            avg_lm = step_lm_loss / max(1, step_valid_micros)
            avg_aux = step_aux_loss / max(1, step_valid_micros)
            elapsed = time.time() - t0
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[step %d/%d] lm=%.4f aux=%.4f lr=%.2e n_ctx=%d "
                "dolmino=%d babi=%d nf=%d speed=%.2f steps/s",
                global_step, args.total_steps, avg_lm, avg_aux, lr,
                current_n_ctx, n_dolmino, n_babilong, n_nonfinite,
                steps_per_sec,
            )
            if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
                wandb.log({
                    "train/lm_loss": avg_lm,
                    "train/aux_loss": avg_aux,
                    "train/lr": lr,
                    "train/n_ctx": current_n_ctx,
                    "train/speed_steps_s": steps_per_sec,
                    "train/n_nonfinite": n_nonfinite,
                    "train/dolmino_count": n_dolmino,
                    "train/babilong_count": n_babilong,
                    "memory/top1_sim": _collect_top1_sim(model),
                }, step=global_step)

        # Save checkpoint
        if (args.save_interval > 0
                and global_step % args.save_interval == 0
                and global_step < args.total_steps):
            if is_main(rank):
                _save_adapter(model, args, global_step)
            if world_size > 1:
                dist.barrier()

        # Quick eval — ALL ranks run eval to avoid DDP collective mismatch
        if (args.eval_interval > 0
                and global_step % args.eval_interval == 0):
            acc = quick_eval_babilong(
                model.module if isinstance(model, DDP) else model,
                tokenizer, device, args.chunk_size,
                n_samples=args.eval_samples,
            )
            if is_main(rank):
                logger.info("[eval step %d] BABILong qa1@1k accuracy: %.1f%%", global_step, acc)
                if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
                    wandb.log({"eval/babilong_qa1_acc": acc}, step=global_step)
            if world_size > 1:
                dist.barrier()
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
