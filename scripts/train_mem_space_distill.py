#!/usr/bin/env python3
"""Teacher distillation training for Memory-Space compression.

Core idea: the same Llama-3-8B backbone serves as both teacher (memory disabled,
sees full context via sliding window) and student (memory enabled, compresses
context into slots). KD loss on target-chunk logits provides dense per-token
supervision that is far richer than the sparse LM loss alone.

Teacher: model with _memory_disabled=True on all MemorySpaceLayer instances.
         Sees as much context as fits in a sliding window (last N chunks + target).
Student: model with _memory_disabled=False (normal memory forward).
         Context chunks compressed into memory bank, then predicts target chunk.

Loss: L = alpha_lm * L_lm + alpha_kd * L_kd
      L_kd = KL(student_logits/T || teacher_logits/T) * T^2

Design references:
    - ops/research_notes/20260531_compression_memory_training_methods.md Part E.3.4
    - scripts/train_mem_space_babilong.py (chunked forward pattern)
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
from typing import List, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

_BABILONG_PKG = os.path.join(PROJECT_ROOT, "third_party", "babilong-pkg")
if os.path.isdir(_BABILONG_PKG) and _BABILONG_PKG not in sys.path:
    sys.path.insert(0, _BABILONG_PKG)

from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM  # noqa: E402

from src.memory.mem_space import (  # noqa: E402
    MemorySpaceConfig,
    apply_mem_space_to_model,
)
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
# PG-19 dataset (minimal re-use from train_mem_space_babilong.py)
# --------------------------------------------------------------------------- #


class PG19ChunksDataset(Dataset):
    """Pre-tokenised PG-19 chunks loaded via numpy mmap."""

    def __init__(self, npy_path: str, seq_length: int, skip_chunks: int, max_chunks: int) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks: skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        if len(self.data) == 0:
            raise RuntimeError(f"PG19ChunksDataset is empty: skip={skip_chunks}, max={max_chunks}")
        logger.info("Loaded %d PG-19 chunks from %s", len(self.data), npy_path)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)[: self.seq_length]
        return {"input_ids": tokens, "labels": tokens.clone()}


def pg19_collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #


def init_distributed() -> tuple:
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
        # Dual-gate params
        if wrapper.gate_proj_new is not None:
            for p in wrapper.gate_proj_new.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        if wrapper.gate_proj_mem is not None:
            for p in wrapper.gate_proj_mem.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        if wrapper.gate_bias is not None and id(wrapper.gate_bias) not in seen:
            params.append(wrapper.gate_bias); seen.add(id(wrapper.gate_bias))
    # L3 summary pool params
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        for p in l3_pool.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    # L2 compressor params
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
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None


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


# --------------------------------------------------------------------------- #
# Memory disable/enable helpers (teacher/student mode switching)
# --------------------------------------------------------------------------- #


def _set_memory_disabled(model: torch.nn.Module, disabled: bool) -> None:
    """Toggle _memory_disabled on all MemorySpaceLayer instances.

    When disabled=True, each layer's forward() bypasses memory logic entirely
    and calls the wrapped LlamaDecoderLayer directly (teacher mode).
    When disabled=False, normal memory forward is restored (student mode).
    """
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w._memory_disabled = disabled


# --------------------------------------------------------------------------- #
# KD loss computation
# --------------------------------------------------------------------------- #


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    temperature: float = 2.0,
    reduction: str = "batchmean",
) -> torch.Tensor:
    """KL-divergence distillation loss.

    L_kd = KL(student_soft || teacher_soft) * T^2
    where soft = softmax(logits / T).

    Args:
        student_logits: [B, T, V] raw logits from student.
        teacher_logits: [B, T, V] raw logits from teacher (detached).
        temperature: softmax temperature for softening distributions.
        reduction: 'batchmean' (default) or 'none'.

    Returns:
        Scalar KD loss (already scaled by T^2).
    """
    student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
    kl = F.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return kl * (temperature ** 2)


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Memory-Space Teacher Distillation Training")

    # Base
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Warm-start from existing mem_space adapter .pt")

    # Data
    p.add_argument("--pg19_data", type=str, default="data/pg19_chunks_llama3.npy")
    p.add_argument("--pg19_max_chunks", type=int, default=5000)
    p.add_argument("--pg19_skip_chunks", type=int, default=200)

    # Training shape
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--chunk_size", type=int, default=1024)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)

    # KD hyperparams
    p.add_argument("--kd_temperature", type=float, default=2.0,
                   help="Temperature for softening logits in KD loss.")
    p.add_argument("--alpha_kd", type=float, default=0.5,
                   help="Weight for KD loss (L = alpha_kd*L_kd + alpha_lm*L_lm).")
    p.add_argument("--alpha_lm", type=float, default=0.5,
                   help="Weight for standard LM loss.")
    p.add_argument("--teacher_window_chunks", type=int, default=4,
                   help="Max number of context chunks teacher sees (sliding window). "
                        "If total context > this * chunk_size, teacher only sees "
                        "the last teacher_window_chunks chunks + target chunk.")

    # Optim
    p.add_argument("--total_steps", type=int, default=500)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)

    # mem_space config
    p.add_argument("--num_slots", type=int, default=512)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--selector_dim", type=int, default=128)
    p.add_argument("--writeback_gate_max", type=float, default=0.3)
    p.add_argument("--writeback_warmup_steps", type=int, default=0)
    p.add_argument("--load_balance_weight", type=float, default=0.01)
    p.add_argument("--entropy_aux_weight", type=float, default=0.001)
    p.add_argument("--selector_temperature", type=float, default=1.0)
    p.add_argument("--key_repulsion_weight", type=float, default=0.01)
    p.add_argument("--key_repulsion_threshold", type=float, default=0.3)
    p.add_argument("--peak_routing_weight", type=float, default=0.1)
    p.add_argument("--slot_value_norm_cap", type=float, default=5.0)
    p.add_argument("--slot_init", type=str, default="random",
                   choices=["zero", "random", "hidden_pool", "strided_token"])
    p.add_argument("--slot_init_noise", type=float, default=0.05)
    p.add_argument("--unfreeze_hidden_to_slot", action="store_true", default=True)
    p.add_argument("--shared_memory_bank", action="store_true", default=True)
    p.add_argument("--swa_window", type=int, default=0)

    # Dual-gate
    p.add_argument("--use_dual_gate", action="store_true", default=False)
    p.add_argument("--input_bias_init", type=float, default=0.0)
    p.add_argument("--forget_bias_init", type=float, default=2.0)
    p.add_argument("--dual_gate_tanh_new", action="store_true", default=True)

    # L3 / L2
    p.add_argument("--use_l3_summary", action="store_true", default=False)
    p.add_argument("--l3_n_summary", type=int, default=64)
    p.add_argument("--l3_n_layers", type=int, default=2)
    p.add_argument("--l3_n_heads", type=int, default=8)
    p.add_argument("--disable_l1_inject", action="store_true", default=False)
    p.add_argument("--use_l2", action="store_true", default=False)
    p.add_argument("--l2_compress_ratio", type=int, default=16)
    p.add_argument("--l2_d_c", type=int, default=512)
    p.add_argument("--l2_d_h_rope", type=int, default=64)
    p.add_argument("--l2_init_scale", type=float, default=0.001)

    # v6/v7/v8
    p.add_argument("--use_replace_writeback", action="store_true", default=False)
    p.add_argument("--num_global_slots", type=int, default=0)
    p.add_argument("--global_slot_forget_bias", type=float, default=1.0)
    p.add_argument("--global_slot_input_gate_only", action="store_true", default=False)

    # Misc
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--zero_alpha_on_cold_start", action="store_true", default=False)
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100)
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p.parse_args()


# --------------------------------------------------------------------------- #
# Model build
# --------------------------------------------------------------------------- #


def build_model(args, device, dtype) -> torch.nn.Module:
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
        use_l2=args.use_l2,
        l2_compress_ratio=args.l2_compress_ratio,
        l2_d_c=args.l2_d_c,
        l2_d_h_rope=args.l2_d_h_rope,
        l2_init_scale=args.l2_init_scale,
        gradient_checkpointing=args.gradient_checkpointing,
        zero_alpha_on_cold_start=args.zero_alpha_on_cold_start,
        use_replace_writeback=args.use_replace_writeback,
        num_global_slots=args.num_global_slots,
        global_slot_forget_bias=args.global_slot_forget_bias,
        global_slot_input_gate_only=args.global_slot_input_gate_only,
    )

    # Snapshot rotary inv_freq in fp32 before bf16 cast
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
    except AttributeError:
        pass

    # Warm-start from adapter checkpoint
    if args.init_checkpoint and os.path.isfile(args.init_checkpoint):
        logger.info("Loading adapter from %s", args.init_checkpoint)
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
        logger.info("Loaded: %d keys, missing=%d, unexpected=%d",
                    len(cleaned), len(missing), len(unexpected))
        # Ramp step_counter so beta is fully active
        from src.memory.mem_space.layer import MemorySpaceLayer as _MSL
        warmup = max(args.writeback_warmup_steps, 1)
        for w in getattr(model, "_mem_space_layers", []):
            if isinstance(w, _MSL):
                w.step_counter = warmup

    return model


# --------------------------------------------------------------------------- #
# Distillation training step
# --------------------------------------------------------------------------- #


def _distill_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,   # [1, total_len]
    labels: torch.Tensor,      # [1, total_len]
    chunk_size: int,
    device: torch.device,
    kd_temperature: float = 2.0,
    alpha_kd: float = 0.5,
    alpha_lm: float = 0.5,
    teacher_window_chunks: int = 4,
):
    """One distillation step: teacher pass (no memory) + student pass (memory).

    Returns:
        dict with keys: loss, lm_loss, kd_loss, aux_loss
    """
    total_len = input_ids.shape[1]
    n_chunks = max(1, math.ceil(total_len / chunk_size))

    # Split into chunks
    pieces_in = list(input_ids[0].split(chunk_size))
    pieces_lbl = list(labels[0].split(chunk_size))

    # --- Teacher pass: memory disabled, full/sliding-window context ---
    # Teacher sees as much context as possible (up to teacher_window_chunks
    # chunks before the target) concatenated with the target chunk.
    # This gives the teacher full-context soft labels.
    _set_memory_disabled(model, True)

    # Determine teacher input: last teacher_window_chunks context + target
    if n_chunks <= teacher_window_chunks + 1:
        # Teacher can see everything
        teacher_input = input_ids  # [1, total_len]
    else:
        # Sliding window: take last teacher_window_chunks context chunks + target
        start_chunk = n_chunks - 1 - teacher_window_chunks
        teacher_pieces = pieces_in[start_chunk:]
        teacher_input = torch.cat(teacher_pieces).unsqueeze(0).to(device)

    # Target chunk is always the last chunk
    target_len = pieces_in[-1].shape[0]

    with torch.no_grad():
        teacher_out = model(input_ids=teacher_input, use_cache=False)
        # Extract logits for the target chunk positions (last target_len tokens)
        teacher_logits = teacher_out.logits[:, -target_len:, :].detach()

    # --- Student pass: memory enabled, chunked forward ---
    _set_memory_disabled(model, False)
    _reset_banks(model)

    # Stream context chunks through memory (no grad)
    if n_chunks > 1:
        with torch.no_grad():
            for ci in pieces_in[:-1]:
                model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)

    # Forward on target chunk WITH gradient
    last_in = pieces_in[-1].unsqueeze(0).to(device)
    last_lbl = pieces_lbl[-1].unsqueeze(0).to(device)
    student_out = model(input_ids=last_in, labels=last_lbl, use_cache=False)

    # Student logits for target chunk
    student_logits = student_out.logits  # [1, target_len, V]

    # --- Compute losses ---
    # LM loss (from HF model forward with labels)
    lm_loss = student_out.loss if student_out.loss is not None else torch.tensor(0.0, device=device)

    # KD loss: align student to teacher on target positions
    # Shift by 1 for next-token prediction alignment:
    # student_logits[:, :-1] predicts token at position +1
    # teacher_logits[:, :-1] predicts token at position +1
    kd_l = kd_loss(
        student_logits[:, :-1, :].contiguous(),
        teacher_logits[:, :-1, :].contiguous(),
        temperature=kd_temperature,
    )

    # Aux losses from memory layers
    aux_loss = _collect_aux_loss(model, device)

    # Combined loss
    total_loss = alpha_lm * lm_loss + alpha_kd * kd_l + aux_loss

    return {
        "loss": total_loss,
        "lm_loss": lm_loss.detach(),
        "kd_loss": kd_l.detach(),
        "aux_loss": aux_loss.detach() if aux_loss.requires_grad else aux_loss,
    }


# --------------------------------------------------------------------------- #
# Save adapter
# --------------------------------------------------------------------------- #


def _save_adapter(model: torch.nn.Module, path: str, step: int) -> None:
    """Save only the mem_space adapter parameters."""
    root = getattr(model, "module", model)
    state = {}
    for name, param in root.named_parameters():
        if param.requires_grad:
            state[name] = param.detach().cpu()
    # Also save shared bank if present
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None and shared_bank.slots is not None:
        state["_shared_bank_slots"] = shared_bank.slots.detach().cpu()
    torch.save({"model_state_dict": state, "step": step}, path)


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    dtype = dtype_map[args.dtype]
    torch.manual_seed(args.seed + rank)

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "Distillation Training | model=%s | total_steps=%d | "
            "chunk_size=%d | kd_T=%.1f | alpha_kd=%.2f | alpha_lm=%.2f | "
            "teacher_window=%d chunks | world_size=%d",
            args.model_path, args.total_steps, args.chunk_size,
            args.kd_temperature, args.alpha_kd, args.alpha_lm,
            args.teacher_window_chunks, world_size,
        )

    # --- Model --- #
    model = build_model(args, device, dtype)
    _freeze_backbone(model)

    if is_main(rank):
        n_trainable = sum(p.numel() for p in _mem_space_params(model))
        logger.info("mem_space trainable: %.2fM params", n_trainable / 1e6)

    # --- DDP --- #
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)

    # --- Data --- #
    if not os.path.isfile(args.pg19_data):
        raise RuntimeError(f"PG-19 data not found: {args.pg19_data}")
    pg19_ds = PG19ChunksDataset(
        npy_path=args.pg19_data,
        seq_length=args.max_seq_len,
        skip_chunks=args.pg19_skip_chunks,
        max_chunks=args.pg19_max_chunks,
    )
    pg19_sampler = None
    if world_size > 1:
        pg19_sampler = DistributedSampler(
            pg19_ds, num_replicas=world_size, rank=rank,
            shuffle=True, drop_last=False, seed=args.seed,
        )
    pg19_loader = DataLoader(
        pg19_ds, batch_size=args.batch_size, sampler=pg19_sampler,
        shuffle=(pg19_sampler is None), num_workers=args.num_workers,
        collate_fn=pg19_collate_fn, pin_memory=True, drop_last=False,
    )

    def _cycle_pg19():
        _epoch = 0
        while True:
            if pg19_sampler is not None:
                pg19_sampler.set_epoch(_epoch)
            for _b in pg19_loader:
                yield _b
            _epoch += 1

    data_iter = _cycle_pg19()

    # --- Optimizer --- #
    trainable = _mem_space_params(
        model.module if hasattr(model, "module") else model
    )
    if not trainable:
        raise RuntimeError("No mem_space trainable params found.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.01)

    if is_main(rank):
        logger.info("Optimizer: AdamW, lr=%.2e, %d param groups (%d tensors)",
                    args.lr, 1, len(trainable))

    # --- Wandb --- #
    wandb_run = None
    if is_main(rank):
        try:
            import wandb
            run_name = args.wandb_run_name or f"distill_T{args.kd_temperature}_a{args.alpha_kd}"
            wandb_run = wandb.init(
                project=args.wandb_project,
                name=run_name,
                config=vars(args),
            )
            logger.info("W&B initialized: %s", run_name)
        except Exception as e:
            logger.warning("W&B init failed: %s — continuing without logging", e)

    # --- Training loop --- #
    model.train()
    t0 = time.time()
    running_loss = 0.0
    running_lm = 0.0
    running_kd = 0.0

    for step in range(1, args.total_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        result = _distill_train_step(
            model=model,
            input_ids=input_ids,
            labels=labels,
            chunk_size=args.chunk_size,
            device=device,
            kd_temperature=args.kd_temperature,
            alpha_kd=args.alpha_kd,
            alpha_lm=args.alpha_lm,
            teacher_window_chunks=args.teacher_window_chunks,
        )

        loss = result["loss"]
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

        optimizer.step()
        optimizer.zero_grad()
        _step_counters_inc(model)

        # Logging
        running_loss += loss.item()
        running_lm += result["lm_loss"].item()
        running_kd += result["kd_loss"].item()

        if step % args.log_interval == 0 and is_main(rank):
            avg_loss = running_loss / args.log_interval
            avg_lm = running_lm / args.log_interval
            avg_kd = running_kd / args.log_interval
            elapsed = time.time() - t0
            logger.info(
                "step %d/%d | loss=%.4f (lm=%.4f kd=%.4f) | "
                "%.1f s elapsed | %.2f steps/s",
                step, args.total_steps, avg_loss, avg_lm, avg_kd,
                elapsed, step / elapsed,
            )
            if wandb_run:
                wandb_run.log({
                    "loss": avg_loss,
                    "lm_loss": avg_lm,
                    "kd_loss": avg_kd,
                    "step": step,
                })
            running_loss = 0.0
            running_lm = 0.0
            running_kd = 0.0

        # Save checkpoint
        if args.save_interval > 0 and step % args.save_interval == 0 and is_main(rank):
            ckpt_path = os.path.join(args.output_dir, f"adapter_step{step}.pt")
            _save_adapter(model, ckpt_path, step)
            logger.info("Saved checkpoint: %s", ckpt_path)

    # Final save
    if is_main(rank):
        final_path = os.path.join(args.output_dir, "adapter_final.pt")
        _save_adapter(model, final_path, args.total_steps)
        logger.info("Training complete. Final adapter: %s", final_path)

    if wandb_run:
        wandb_run.finish()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
