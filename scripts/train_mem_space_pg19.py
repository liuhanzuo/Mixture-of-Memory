#!/usr/bin/env python3
"""Memory-Space v0 training/eval driver on pg19 chunks (Llama-3-8B).

This script does a short LM training rollout on pg19 with every decoder layer
patched by `MemorySpaceLayer` and reports final perplexity on the training
chunks plus a NaN count.  It intentionally mirrors the patterns used by:

    * ``scripts/train_llama_baseline.py`` (DDP init, PreTokenizedDataset)
    * ``scripts/eval_qfilters.py``       (PPL accumulation, distributed reduce)

Design reference:
    ops/research_notes/20260426_memory_space_design_direction.md

Smoke contract (single GPU, 10 chunks):
    torchrun --nproc_per_node=1 scripts/train_mem_space_pg19.py \
        --model /apdcephfs_wzc1/.../Llama--Llama3-8b \
        --data  /apdcephfs_wzc1/.../data/pg19_chunks_llama3.npy \
        --max_chunks 10 --seq_len 4096 \
        --num_slots 512 --top_k 64 \
        --output_dir outputs/mem_space_v0_smoke_llama3 \
        --max_train_steps 10

Full run (8 GPUs, 200 chunks):
    torchrun --nproc_per_node=8 scripts/train_mem_space_pg19.py \
        --model /apdcephfs_wzc1/.../Llama--Llama3-8b \
        --data  /apdcephfs_wzc1/.../data/pg19_chunks_llama3.npy \
        --max_chunks 200 --seq_len 4096 \
        --num_slots 512 --top_k 64 \
        --output_dir outputs/mem_space_v0_full_llama3

Important:
    * Mem-Space slots are per-sample state.  We `reset` the memory banks at
      the start of every chunk so different documents do not contaminate
      each other (same policy as Q-Filters cache allocation in
      eval_qfilters.py).
    * v0 cannot do incremental decoding; we therefore run each chunk as a
      single forward (no KV-cache, no sub-window carryover).
    * Training optimiser steps only touch the newly-added memory-space
      parameters (selector, gate, projections).  The backbone is frozen so
      the 10-step rollout is fast and only exercises the new code paths.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import string
import sys
import time
from typing import Iterable, List

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from transformers import AutoTokenizer, LlamaForCausalLM, LlamaConfig

from src.memory.mem_space import (
    MemorySpaceConfig,
    apply_mem_space_to_model,
)
from src.memory.mem_space.niah_dataset import NIAHIterableDataset, niah_collate_fn

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #


class PreTokenizedEvalDataset(Dataset):
    """Same layout as scripts/eval_qfilters.py's dataset."""

    def __init__(self, npy_path: str, seq_length: int, skip_chunks: int, max_chunks: int) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks : skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        logger.info(
            "Loaded %d chunks of %d tokens from %s",
            len(self.data), self.seq_length, npy_path,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)[: self.seq_length]
        return {"input_ids": tokens, "labels": tokens.clone()}


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "labels": torch.stack([b["labels"] for b in batch]),
    }


# --------------------------------------------------------------------------- #
# Distributed helpers
# --------------------------------------------------------------------------- #


def init_distributed() -> tuple[int, int, int]:
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
# Core
# --------------------------------------------------------------------------- #


def _mem_space_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    """Return the subset of params that belong to MemorySpaceLayer-only add-ons.

    We train only the selector / slot projections / gate — the backbone
    LlamaDecoderLayer weights remain frozen for the v0 smoke.  This keeps
    the 10-step rollout memory-light and makes the gradient signal
    interpretable (anything that changes is from mem_space).
    """
    params: List[torch.nn.Parameter] = []
    seen: set[int] = set()
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return params
    for wrapper in mem_layers:
        for p in wrapper.selector.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
        # gate is a single scalar nn.Parameter (writeback β gate)
        if id(wrapper.gate_param) not in seen:
            params.append(wrapper.gate_param); seen.add(id(wrapper.gate_param))
        # Tier-3 (2026-04-26): Flamingo-style OUTPUT-side gate on slot_delta.
        # alpha=tanh(slot_output_gate); init 0 → exact bypass parity at step 0.
        slot_gate = getattr(wrapper, "slot_output_gate", None)
        if slot_gate is not None and id(slot_gate) not in seen:
            params.append(slot_gate); seen.add(id(slot_gate))
        # slot_to_hidden: trainable. hidden_to_slot: frozen in __init__ because
        # it participates in no gradient-bearing op (O_mem_slot.detach() +
        # _reset_banks). Excluding from the trainable set reclaims ~540M params.
        for p in wrapper.slot_to_hidden.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
        # Fix I (2026-04-29): include hidden_to_slot when explicitly unfrozen via --unfreeze_hidden_to_slot
        # Root cause of routing degeneracy: hidden_to_slot was always excluded from optimizer,
        # making --unfreeze_hidden_to_slot a no-op. Write path must be trainable
        # for slot content to diversify → routing signal to develop.
        if not getattr(wrapper.config, 'hidden_to_slot_frozen', True):
            for p in wrapper.hidden_to_slot.parameters():
                if id(p) not in seen:
                    params.append(p)
                    seen.add(id(p))
    return params


def _freeze_backbone(model: torch.nn.Module) -> None:
    """Freeze every parameter, then unfreeze mem_space-only params."""
    for p in model.parameters():
        p.requires_grad = False
    for p in _mem_space_params(model):
        p.requires_grad = True


def _step_counters_inc(model: torch.nn.Module) -> None:
    """Walk the patched layers and bump their step counter (used to warm up β)."""
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.step_counter += 1


def _reset_banks(model: torch.nn.Module) -> None:
    """Wipe per-sample slot state between documents (chunks).

    Branch-3 (2026-04-26): under ``config.shared_memory_bank=True`` patch.py
    exposes ``_mem_space_shared_bank``; resetting it once is equivalent to
    resetting every wrapper's bank (they all point to the same object) but
    avoids 32× wasted work and, more importantly, guarantees only one
    inter-chunk graph break even if the wrappers were mid-iteration.
    """
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.reset()
        return
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w.memory_bank.reset()


def _detach_banks(model: torch.nn.Module) -> None:
    """Break autograd graph across chunk/step boundary; preserve slot content for carry-over."""
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
    """Sum the per-layer load-balance and entropy aux losses for the last forward.
    """
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    total = torch.zeros((), device=device)
    if not mem_layers:
        return total
    for w in mem_layers:
        lb = w.last_aux_losses.get("load_balance")
        if lb is not None:
            total = total + lb
        ent = w.last_aux_losses.get("entropy")
        if ent is not None:
            total = total + ent
        kr = w.last_aux_losses.get("key_repulsion")
        if kr is not None:
            total = total + kr
        pk = w.last_aux_losses.get("peak_routing")
        if pk is not None:
            total = total + pk
    return total



def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="mem_space v0 training / eval driver on pg19")
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--max_chunks", type=int, default=10)
    p.add_argument("--seq_len", type=int, default=4096)
    p.add_argument("--skip_chunks", type=int, default=40000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--output_dir", type=str, required=True)

    # Mem-Space hypers
    p.add_argument("--num_slots", type=int, default=512,
                   help="N — per-layer, per-sample slot count (kv_budget-equivalent)")
    p.add_argument("--top_k", type=int, default=64,
                   help="k — number of slots prepended to the extended sequence")
    p.add_argument("--selector_dim", type=int, default=128)
    p.add_argument("--writeback_gate_max", type=float, default=0.3)
    p.add_argument("--writeback_warmup_steps", type=int, default=0,
                   help="set 0 so β kicks in immediately during the 10-step smoke")
    p.add_argument("--load_balance_weight", type=float, default=0.01)
    p.add_argument("--entropy_aux_weight", type=float, default=0.001,
                   help="Fix D.2: weight for routing entropy aux loss")
    p.add_argument("--selector_temperature", type=float, default=1.0,
                   help="Fix O (2026-04-29): softmax temperature for slot-routing logits. "
                        "Was hardcoded 10.0 (amplified LM→slot_keys gradient 10×, caused routing collapse). "
                        "Default 1.0 restores balanced LM:SKRL gradient ratio.")
    p.add_argument("--slot_value_norm_cap", type=float, default=5.0,
                   help="FIX X.1 (2026-04-30): Clamp slot value norms after writeback. "
                        "Fix Z.1 default 5.0 (frozen random keys, no SKRL/VQ-EMA). 0 = disabled.")
    p.add_argument("--key_repulsion_weight", type=float, default=0.01,
                   help="Fix Z.2c: weight for key repulsion loss. "
                        "Prevents slot_keys from collapsing to the same direction.")
    p.add_argument("--key_repulsion_threshold", type=float, default=0.3,
                   help="Fix Z.2c: cosine similarity threshold for repulsion. "
                        "Only penalize pairs with cos > threshold.")
    p.add_argument("--peak_routing_weight", type=float, default=0.1,
                   help="Fix Z.2g: weight for peak routing loss. "
                        "Pushes per-chunk routing to be peaked (low conditional entropy).")

    # Training knobs
    p.add_argument("--max_train_steps", type=int, default=0,
                   help="If >0, run backward / optimizer.step for this many chunks; "
                        "0 = eval-only (forward + PPL).")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--bypass_memory", action="store_true",
                   help="Ablation: monkey-patch every MemorySpaceLayer.forward "
                        "to call forward_no_memory (no slot prepend, no writeback). "
                        "Used to parity-check that the wrapping itself is clean.")
    p.add_argument("--slot_init", type=str, default="hidden_pool",
                   choices=["zero", "random", "hidden_pool", "strided_token"],
                   help="Slot initialisation strategy (see memory_bank.init_from_hidden). "
                        "Use 'random' to avoid the oracle-leak failure mode where pooled "
                        "slots expose future-chunk information to early H-queries. "
                        "'strided_token': Fix K (2026-04-29) -- slot i = H[i*stride %% T], "
                        "slot 0 = last token; avoids oracle-leak and spreads init across chunk.")
    p.add_argument("--slot_init_noise", type=float, default=0.02,
                   help="std of the slot init noise. 1.0 recommended when slot_init=random "
                        "(matches Llama post-rmsnorm magnitude).")
    p.add_argument("--unfreeze_hidden_to_slot", action="store_true",
                   help="Stage-2a: allow the `hidden_to_slot` projection to train. "
                        "Default (flag absent) keeps it frozen — matches the Tier-3 "
                        "cure that produced held-out PPL=2.1278 on Llama-3-8B. Set "
                        "this flag to test whether a gradient-bearing write path "
                        "improves PPL further (Branch 1 of the Stage-2 decision tree).")
    # Branch-3 (2026-04-26): shared-bank + gradient-bearing writeback. Default ON
    # so `python scripts/train_mem_space_pg19.py` without flags runs the Option
    # A.2 configuration (intra-chunk BPTT through depth). Use --no_shared_memory_bank
    # as an ablation to fall back to per-layer banks.
    mbg = p.add_mutually_exclusive_group()
    mbg.add_argument("--shared_memory_bank", dest="shared_memory_bank",
                     action="store_true", default=True,
                     help="Share one MemoryBank across all patched decoder layers "
                          "so intra-chunk writeback-BPTT threads through depth "
                          "(Branch 3 Option A.2; ref §3 of "
                          "20260426_mem_space_v0_branch3_writeback_bptt.md). "
                          "DEFAULT.")
    mbg.add_argument("--no_shared_memory_bank", dest="shared_memory_bank",
                     action="store_false",
                     help="Ablation: per-layer banks (no cross-layer BPTT). "
                          "Layer-local writeback gradient still flows via the "
                          "selector path but cannot compound across depth.")

    # SWA + NIAH training args (Stage 2, 2026-04-27)
    p.add_argument("--swa_window", type=int, default=0,
                   help="SWA sliding window size (0=full causal). Passed to MemorySpaceConfig. "
                        "Recommended: chunk_size // 8. E.g. 512 for chunk_size=4096.")
    p.add_argument("--niah_mix_fraction", type=float, default=0.0,
                   help="Fraction of training steps that use NIAH sequences. "
                        "0.0 = pure LM (default). 0.10 = 10%% NIAH.")
    p.add_argument("--niah_max_N", type=int, default=16,
                   help="Max number of haystack chunks between needle and query in NIAH samples.")
    p.add_argument("--max_steps", type=int, default=30000,
                   help="Total optimizer steps. Replaces --max_train_steps for long runs.")
    p.add_argument("--init_from", type=str, default=None,
                   help="Path to an adapter checkpoint (.pt) to warm-start from (e.g. Stage 1 ckpt).")

    return p.parse_args()


@torch.no_grad()
def _evaluate_chunks(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    pad_token_id: int,
    world_size: int,
) -> tuple[float, float, int, int]:
    """Return (ppl, avg_loss, total_tokens, n_nonfinite_chunks)."""
    model.eval()
    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)
    n_bad = 0

    for i, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)
        # fresh slots per document
        _reset_banks(model)
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        loss = out.loss.detach()
        if not torch.isfinite(loss):
            logger.warning("[eval] chunk %d: non-finite loss %s", i, loss.item())
            n_bad += 1
            continue
        n_tok = (labels != pad_token_id).sum()
        total_loss += loss.double() * n_tok.double()
        total_tokens += n_tok.double()
        if (i + 1) % 20 == 0:
            cur_ppl = math.exp((total_loss / total_tokens.clamp_min(1)).item())
            logger.info("  [eval] chunk %d: loss=%.4f cumul_ppl=%.4f", i + 1, loss.item(), cur_ppl)

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)
        n_bad_t = torch.tensor([n_bad], device=device, dtype=torch.long)
        dist.all_reduce(n_bad_t, op=dist.ReduceOp.SUM)
        n_bad = int(n_bad_t.item())

    tot_tok = int(total_tokens.item())
    if tot_tok == 0:
        raise RuntimeError("_evaluate_chunks: 0 tokens scored (all chunks NaN?)")
    avg_loss = (total_loss / total_tokens).item()
    return math.exp(avg_loss), avg_loss, tot_tok, n_bad


def _make_streaming_niah_sample(
    pg19_data: np.ndarray,
    rng: random.Random,
    tokenizer,
    chunk_size: int,
    niah_max_N: int,
    pad_id: int,
) -> tuple:
    """Generate a streaming NIAH training sample.

    Returns (N_gap, code, chunks) where chunks is a list of
    (token_list, label_list) tuples.  First N_gap chunks are haystack
    (labels all -100); last chunk is the question (labels unmasked at answer
    token positions).

    Streaming design: caller streams the first N_gap chunks through the model
    in torch.no_grad() so MemoryBank accumulates without gradient overhead,
    then does one gradient-bearing forward on the final question chunk.  This
    is ~17x more memory-efficient than a single 69K-token flat forward.
    """
    N_gap: int = rng.randint(1, niah_max_N)
    name: str = "".join(rng.choices(string.ascii_uppercase, k=6))
    code: str = "".join(rng.choices(string.digits, k=5))

    needle_sentence = f"MEMORIZE: The secret code for agent {name} is {code}. END_MEMORIZE"
    question_suffix = f"The secret code for agent {name} is "

    needle_ids: list = tokenizer.encode(" " + needle_sentence, add_special_tokens=False)
    question_ids: list = tokenizer.encode(question_suffix, add_special_tokens=False)
    answer_ids: list = tokenizer.encode(code, add_special_tokens=False)

    needle_chunk_pos: int = N_gap // 2
    chunks: list = []

    n_chunks = len(pg19_data)
    for i in range(N_gap):
        chunk_idx = rng.randint(0, n_chunks - 1)
        bg_tokens: list = pg19_data[chunk_idx, :chunk_size].tolist()
        if len(bg_tokens) < chunk_size:
            bg_tokens = bg_tokens + [pad_id] * (chunk_size - len(bg_tokens))

        if i == needle_chunk_pos:
            insert_at: int = rng.randint(0, max(0, chunk_size - len(needle_ids)))
            chunk = (
                bg_tokens[:insert_at]
                + needle_ids
                + bg_tokens[insert_at + len(needle_ids):]
            )
            chunk = (chunk + [pad_id] * chunk_size)[:chunk_size]
        else:
            chunk = bg_tokens

        chunks.append((chunk, [-100] * chunk_size))

    # Question chunk: question_ids + answer_ids + padding
    q_raw = question_ids + answer_ids
    if len(q_raw) < chunk_size:
        q_chunk = q_raw + [pad_id] * (chunk_size - len(q_raw))
    else:
        q_chunk = q_raw[:chunk_size]

    q_labels = [-100] * chunk_size
    ans_start = len(question_ids)
    for j, tok in enumerate(answer_ids):
        pos = ans_start + j
        if pos < chunk_size:
            q_labels[pos] = q_chunk[pos]

    chunks.append((q_chunk, q_labels))
    return N_gap, code, chunks


def main() -> None:
    args = parse_args()
    rank, world_size, local_rank = init_distributed()
    device = torch.device(f"cuda:{local_rank}") if torch.cuda.is_available() else torch.device("cpu")
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[args.dtype]

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(
            "mem_space v0 | model=%s | N=%d k=%d | max_chunks=%d seq_len=%d | train_steps=%d | "
            "world_size=%d",
            args.model, args.num_slots, args.top_k, args.max_chunks, args.seq_len,
            args.max_train_steps, world_size,
        )

    # --- tokenizer (pad id) --- #
    # Load config explicitly first to avoid AutoConfig.from_pretrained calling
    # hf_hub_download with local paths (transformers 5.5.4 repo-id validation bug).
    llama_cfg = LlamaConfig.from_pretrained(args.model, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, config=llama_cfg, trust_remote_code=True, local_files_only=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    pad_id = tokenizer.pad_token_id

    # --- model --- #
    if is_main(rank):
        logger.info("Loading Llama model in %s with attn=%s ...", args.dtype, args.attn_impl)
    model = LlamaForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        attn_implementation=args.attn_impl,
        local_files_only=True,
    ).to(device)

    # --- patch every layer with MemorySpaceLayer --- #
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
    )
    # Patch underlying LlamaModel.layers (HF's LlamaForCausalLM wraps it under .model)
    apply_mem_space_to_model(model, ms_cfg, layer_indices=None)
    # H7 fix v2 (2026-04-26 23:30): snapshot rotary inv_freq in fp32 BEFORE
    # the lossy `.to(dtype=bf16)` cast. The v1 approach (upcast after cast)
    # did NOT work: bf16 rounding of inv_freq is destructive, upcasting a
    # rounded tensor cannot recover mantissa bits. Direct evidence:
    #   inv_freq[1] = 0.81640625  (bf16-rounded)
    #   true fp32   = 0.81225...
    # At pos=1023, angle error ≈ 1023 × (0.81640625 - 0.81225) ≈ 4.25 rad
    # → cos drift up to ±2, matching the observed 1.578 absmax.
    # HF deliberately keeps inv_freq / original_inv_freq in fp32 (see
    # modeling_llama.LlamaRotaryEmbedding.__init__ and
    # modeling_rope_utils.dynamic_rope_update); blanket `.to(dtype=...)`
    # recurses into buffers and destroys that invariant.
    # Evidence: tests/test_wrapper_internal_parity.py H7 probe,
    #           scripts/probe_branch3_bypass_parity.py §5.4 decision tree,
    #           ops/research_notes/20260426_branch3_A2_pollution_debug.md §5.4.
    _rope_snapshot = {}
    try:
        _rot = model.model.rotary_emb
        for _name in ("inv_freq", "original_inv_freq"):
            if hasattr(_rot, _name):
                _rope_snapshot[_name] = getattr(_rot, _name).detach().to(torch.float32).clone()
    except AttributeError:
        pass
    # Mem-space modules were freshly constructed on CPU in fp32 — align them
    # with the backbone (device + dtype) so everything runs on one device.
    model.to(device=device, dtype=dtype)
    # Restore the rotary buffers to fp32 on the current device.
    try:
        _rot = model.model.rotary_emb
        for _name, _buf in _rope_snapshot.items():
            _buf = _buf.to(device=device, dtype=torch.float32)
            # Re-register so PyTorch's buffer bookkeeping stays consistent.
            _rot._buffers[_name] = _buf
            setattr(_rot, _name, _buf)
        if is_main(rank) and _rope_snapshot:
            logger.info(
                "H7 fix v2 applied: restored rotary_emb buffers %s to float32",
                sorted(_rope_snapshot.keys()),
            )
    except AttributeError:
        if is_main(rank):
            logger.warning("H7 fix v2: rotary_emb not accessible on model.model — skipping")

    # --- warm-start from Stage-1 adapter checkpoint (--init_from) --- #
    if args.init_from is not None:
        if is_main(rank):
            logger.info("Loading warm-start adapter from %s ...", args.init_from)
        ckpt_state = torch.load(args.init_from, map_location=device)
        missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
        if is_main(rank):
            logger.info("init_from loaded: %d keys  missing=%d  unexpected=%d",
                        len(ckpt_state), len(missing), len(unexpected))

    # Optional parity/ablation bypass: monkey-patch every MemorySpaceLayer so
    # its forward becomes forward_no_memory. This tests whether the wrapping
    # itself is clean (expect parity PPL == vanilla Llama PPL).
    if args.bypass_memory:
        from src.memory.mem_space.layer import MemorySpaceLayer
        mem_layers = getattr(model, "_mem_space_layers", [])
        for w in mem_layers:
            w.forward = w.forward_no_memory.__get__(w, MemorySpaceLayer)
        if is_main(rank):
            logger.info("BYPASS MODE: patched %d MemorySpaceLayer.forward → forward_no_memory",
                        len(mem_layers))
    if is_main(rank):
        n_layers = len(model.model.layers)
        n_trainable = sum(p.numel() for p in _mem_space_params(model))
        logger.info("Patched %d decoder layers | mem_space trainable params: %.2fM",
                    n_layers, n_trainable / 1e6)

    # Freeze backbone; only mem_space adds trainable params.
    _freeze_backbone(model)

    # Fix Z.2g: register forward hook to pop key_repulsion and peak_routing aux
    # losses into output.loss so DDP sees slot_keys in the computation graph.
    # Without this, DDP's find_unused_parameters pre-marks slot_keys as unused,
    # then the aux loss backward tries to mark them again → double-mark crash.
    if world_size > 1:
        _mem_layers_hook = getattr(model, '_mem_space_layers', None)
        def _slot_key_aux_hook(module, inputs, output):
            if _mem_layers_hook is None or output.loss is None:
                return
            aux_total = torch.zeros((), device=output.loss.device, dtype=output.loss.dtype)
            for w in _mem_layers_hook:
                for key in ('key_repulsion', 'peak_routing'):
                    v = w.last_aux_losses.pop(key, None)
                    if v is not None and v.requires_grad:
                        aux_total = aux_total + v
            if aux_total.requires_grad:
                output.loss = output.loss + aux_total
        model.register_forward_hook(_slot_key_aux_hook)

    # DDP wrap (only when world_size > 1 and there are trainable params).
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=True)   # fup=True needed: some params unused in NIAH/SWA paths.
                                                   # slot_keys frozen (Fix Z.1), DDP ignores them automatically.

    # --- data --- #
    dataset = PreTokenizedEvalDataset(
        npy_path=args.data,
        seq_length=args.seq_len,
        skip_chunks=args.skip_chunks,
        max_chunks=args.max_chunks,
    )
    if world_size > 1:
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    else:
        sampler = None
    loader = DataLoader(
        dataset, batch_size=args.batch_size, sampler=sampler, shuffle=False,
        collate_fn=collate_fn, num_workers=2, pin_memory=True, drop_last=False,
    )

    # --- training (Stage 2: SWA + NIAH long-run, 2026-04-27) --- #
    # effective_max_steps: --max_steps (default 30000) takes precedence when >0;
    # falls back to --max_train_steps for backward compatibility with the
    # original short-rollout invocation pattern (--max_train_steps N --max_steps 0).
    effective_max_steps = args.max_steps if args.max_steps > 0 else args.max_train_steps

    if effective_max_steps > 0:
        trainable = [p for p in (_mem_space_params(model.module) if isinstance(model, DDP) else _mem_space_params(model))]
        if not trainable:
            logger.warning("No trainable mem_space params found — skipping training.")
        else:
            optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
            model.train()
            n_done = 0
            n_nonfinite = 0
            niah_correct = 0
            niah_total = 0
            t0 = time.time()

            # Checkpoint save interval: every max_steps//5 or 5000 steps (whichever smaller).
            save_interval = min(5000, effective_max_steps // 5) if effective_max_steps >= 5 else effective_max_steps

            # Data source: separate pg19 loader (full batch_size) + NIAH-only loader
            # (batch_size=1, NIAH seqs are variable-length multi-chunk).
            # mix_fraction controls per-step sampling in the training loop.
            # Previously niah_loader had batch_size=1 and handled ALL batches
            # (both pg19 and NIAH via niah_mix_fraction inside the dataset),
            # which meant every step only processed 1 seq regardless of args.batch_size.
            # FIX (2026-04-27): pg19 batches come from the existing `loader`
            # (batch_size=args.batch_size); NIAH batches come from niah_only_loader
            # (batch_size=1, correct since NIAH seqs are much longer).

            def _cycle_pg19():
                _epoch = 0
                while True:
                    if sampler is not None:
                        sampler.set_epoch(_epoch)
                    for _batch in loader:
                        _batch["is_niah"] = False
                        yield _batch
                    _epoch += 1

            _pg19_gen = _cycle_pg19()

            if args.niah_mix_fraction > 0:
                pg19_data_niah = np.load(args.data, mmap_mode='r')
                niah_only_ds = NIAHIterableDataset(
                    pg19_data_niah,
                    chunk_size=args.seq_len,
                    niah_mix_fraction=1.0,   # NIAH-only; pg19 comes from loader above
                    niah_max_N=args.niah_max_N,
                    tokenizer=tokenizer,
                    seed=42 + rank,          # de-correlate across DDP ranks
                )
                niah_only_loader = DataLoader(niah_only_ds, batch_size=1, num_workers=2,
                                              collate_fn=niah_collate_fn)
                _niah_iter = iter(niah_only_loader)
                _mix_rng = random.Random(42 + rank)

                def _next_batch():
                    if _mix_rng.random() < args.niah_mix_fraction:
                        return next(_niah_iter)
                    else:
                        return next(_pg19_gen)
            else:
                def _next_batch():
                    return next(_pg19_gen)

            if is_main(rank):
                logger.info(
                    "Starting training: effective_max_steps=%d swa_window=%d "
                    "niah_mix_fraction=%.2f niah_max_N=%d save_interval=%d",
                    effective_max_steps, args.swa_window,
                    args.niah_mix_fraction, args.niah_max_N, save_interval,
                )

            while n_done < effective_max_steps:
                batch = _next_batch()
                optimizer.zero_grad(set_to_none=True)

                if batch.get("is_niah", False):
                    # --- NIAH streaming training ---
                    _reset_banks(model)
                    input_ids = batch["input_ids"][0]  # shape [total_len]
                    labels    = batch["labels"][0]     # shape [total_len]
                    chunks       = input_ids.split(args.seq_len)
                    label_chunks = labels.split(args.seq_len)

                    # Stream all chunks except the last with no_grad (accumulate memory only).
                    with torch.no_grad():
                        for _c, _l_c in zip(chunks[:-1], label_chunks[:-1]):
                            _c_in = _c.unsqueeze(0).to(device)
                            model(input_ids=_c_in, use_cache=False)

                    # Last chunk: compute loss with gradient.
                    last_in  = chunks[-1].unsqueeze(0).to(device)
                    last_lbl = label_chunks[-1].unsqueeze(0).to(device)
                    out = model(input_ids=last_in, labels=last_lbl, use_cache=False)
                    lm_loss = out.loss

                    # NIAH accuracy: check if first 5 predicted answer tokens match expected code.
                    answer_positions = (last_lbl[0] != -100).nonzero(as_tuple=True)[0]
                    if len(answer_positions) > 0:
                        pred_tokens = out.logits[0].argmax(dim=-1)
                        ans_start = answer_positions[0].item()
                        # Causal LM: logits[i] predicts token i+1, so to predict token at
                        # ans_start we read logits[ans_start - 1]
                        pred_start = max(0, ans_start - 1)
                        pred_ans = pred_tokens[pred_start : pred_start + 5]
                        pred_str = tokenizer.decode(pred_ans.tolist(), skip_special_tokens=True)
                        expected_code = batch.get("code", "")
                        if isinstance(expected_code, (list, tuple)):
                            expected_code = expected_code[0]
                        if expected_code and expected_code in pred_str:
                            niah_correct += 1
                        niah_total += 1
                else:
                    # --- Standard pg19 LM step ---
                    _detach_banks(model)   # Fix K (2026-04-29): carry-over instead of reset
                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels    = batch["labels"].to(device, non_blocking=True)
                    out = model(input_ids=input_ids, labels=labels, use_cache=False)
                    lm_loss = out.loss

                aux_loss = _collect_aux_loss(model, device)
                loss = lm_loss + aux_loss
                if not torch.isfinite(loss):
                    logger.warning("[train] step %d: non-finite loss lm=%s aux=%s",
                                   n_done, lm_loss.item(), float(aux_loss.item()))
                    n_nonfinite += 1
                    _step_counters_inc(model)
                    n_done += 1
                    continue
                loss.backward()
                # GATE_GRAD_DIAG: diagnose zero gradient to slot_output_gate (remove after Fix F identified)
                if rank == 0 and n_done <= 20:
                    _root = model.module if isinstance(model, DDP) else model
                    if hasattr(_root, '_mem_space_layers') and _root._mem_space_layers:
                        _w0 = _root._mem_space_layers[0]
                        _g_gate = _w0.slot_output_gate.grad
                        _g_s2h = _w0.slot_to_hidden.weight.grad
                        _g_h2s = _w0.hidden_to_slot.weight.grad
                        _g_sk = _w0.selector.slot_keys.grad if hasattr(_w0.selector, 'slot_keys') else None
                        _g_gp = _w0.gate_param.grad if hasattr(_w0, 'gate_param') else None
                        _n_with_grad = sum(1 for p in trainable if p.grad is not None)
                        _n_total = len(trainable)
                        print(
                            f"[GATE_GRAD_DIAG step={n_done}] "
                            f"slot_output_gate.grad={_g_gate} "
                            f"gate_param.grad={_g_gp} "
                            f"slot_to_hidden.weight.grad_norm={_g_s2h.norm().item() if _g_s2h is not None else None} "
                            f"hidden_to_slot.weight.grad_norm={_g_h2s.norm().item() if _g_h2s is not None else None} "
                            f"slot_keys.grad_norm={_g_sk.norm().item() if _g_sk is not None else None} "
                            f"trainable_with_grad={_n_with_grad}/{_n_total}",
                            flush=True,
                        )
                # Fix L-2 (2026-04-29): Per-parameter grad clip for slot projection matrices.
                # slot_to_hidden/hidden_to_slot receive large gradients at lr=1e-3 due to
                # the now-live gradient path (Fix I+J-A). Clip to 0.1 (10× tighter than global)
                # to prevent weight growth runaway while still allowing learning.
                _PROJ_GRAD_CLIP = 0.1
                for _n, _p in model.named_parameters():
                    if _p.grad is not None and ('slot_to_hidden' in _n or 'hidden_to_slot' in _n):
                        torch.nn.utils.clip_grad_norm_([_p], _PROJ_GRAD_CLIP)
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                _step_counters_inc(model)

                n_done += 1

                if is_main(rank) and n_done % 100 == 0:
                    niah_acc = niah_correct / max(niah_total, 1)
                    logger.info(
                        "[train] step %d/%d lm_loss=%.4f aux=%.6f lm_ppl=%.4f "
                        "niah_acc=%.3f dt=%.1fs",
                        n_done, effective_max_steps,
                        lm_loss.item(), float(aux_loss.item()),
                        math.exp(min(lm_loss.item(), 20.0)),
                        niah_acc, time.time() - t0,
                    )
                elif is_main(rank) and n_done % 10 == 0:
                    logger.info(
                        "[train] step %d/%d lm_loss=%.4f aux=%.6f lm_ppl=%.4f dt=%.1fs",
                        n_done, effective_max_steps,
                        lm_loss.item(), float(aux_loss.item()),
                        math.exp(min(lm_loss.item(), 20.0)),
                        time.time() - t0,
                    )

                # Intermediate checkpoint saves (rank 0 only).
                if (is_main(rank) and save_interval > 0
                        and n_done % save_interval == 0
                        and n_done < effective_max_steps):
                    _root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
                    _ADAPTER_KEY_FRAGS = (
                        "selector", "gate_param", "slot_output_gate",
                        "slot_to_hidden", "hidden_to_slot", "memory_bank",
                    )
                    _ckpt_state = {
                        k: v.detach().cpu()
                        for k, v in _root.state_dict().items()
                        if any(frag in k for frag in _ADAPTER_KEY_FRAGS)
                    }
                    _ckpt_path = os.path.join(
                        args.output_dir, f"mem_space_adapter_step{n_done:06d}.pt"
                    )
                    torch.save(_ckpt_state, _ckpt_path)
                    logger.info("Saved intermediate checkpoint: %s (%d keys)",
                                _ckpt_path, len(_ckpt_state))

            if is_main(rank):
                logger.info(
                    "Training complete: %d steps, %d non-finite losses, "
                    "niah_acc=%.3f (%d/%d)",
                    n_done, n_nonfinite,
                    niah_correct / max(niah_total, 1), niah_correct, niah_total,
                )

    # --- save final adapter checkpoint (rank 0 only) --- #
    if is_main(rank) and effective_max_steps > 0:
        root = model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model
        # Save only the mem_space adapter weights (NOT the frozen Llama backbone).
        # Keys to include: anything from MemorySpaceLayer wrappers.
        _ADAPTER_KEY_FRAGMENTS = (
            "selector", "gate_param", "slot_output_gate",
            "slot_to_hidden", "hidden_to_slot", "memory_bank",
        )
        adapter_state = {
            k: v.detach().cpu()
            for k, v in root.state_dict().items()
            if any(frag in k for frag in _ADAPTER_KEY_FRAGMENTS)
        }
        ckpt_path = os.path.join(args.output_dir, "mem_space_adapter.pt")
        torch.save(adapter_state, ckpt_path)
        logger.info(
            "Saved adapter checkpoint: %s  (%d keys, %.1f MB)",
            ckpt_path, len(adapter_state),
            sum(v.numel() * v.element_size() for v in adapter_state.values()) / 1e6,
        )
        # Also write the hyperparameters alongside the checkpoint for reference.
        config_path = os.path.join(args.output_dir, "adapter_config.json")
        with open(config_path, "w") as _f:
            json.dump({
                "num_slots":                args.num_slots,
                "top_k":                    args.top_k,
                "selector_dim":             args.selector_dim,
                "writeback_gate_max":       args.writeback_gate_max,
                "writeback_warmup_steps":   args.writeback_warmup_steps,
                "load_balance_weight":      args.load_balance_weight,
                "slot_init":                args.slot_init,
                "slot_init_noise":          args.slot_init_noise,
                "shared_memory_bank":       args.shared_memory_bank,
                "unfreeze_hidden_to_slot":  args.unfreeze_hidden_to_slot,
                "max_train_steps":          args.max_train_steps,
                "max_steps":                args.max_steps,
                "effective_max_steps":      effective_max_steps,
                "swa_window":               args.swa_window,
                "niah_mix_fraction":        args.niah_mix_fraction,
                "niah_max_N":               args.niah_max_N,
                "lr":                       args.lr,
                "selector_temperature":     args.selector_temperature,
            }, _f, indent=2)
        logger.info("Saved adapter config: %s", config_path)

    # --- final eval pass --- #
    ppl, avg_loss, tot_tok, n_bad = _evaluate_chunks(
        model, loader, device, pad_id, world_size,
    )

    if is_main(rank):
        logger.info("=== FINAL ===")
        logger.info("chunks=%d seq_len=%d tokens=%d nan_chunks=%d",
                    args.max_chunks, args.seq_len, tot_tok, n_bad)
        logger.info("avg_loss=%.6f  ppl=%.4f", avg_loss, ppl)
        results = {
            "model": args.model,
            "data": args.data,
            "max_chunks": args.max_chunks,
            "seq_len": args.seq_len,
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "max_train_steps": args.max_train_steps,
            "max_steps": args.max_steps,
            "swa_window": args.swa_window,
            "niah_mix_fraction": args.niah_mix_fraction,
            "niah_max_N": args.niah_max_N,
            "total_tokens": tot_tok,
            "nan_chunks": n_bad,
            "avg_loss": avg_loss,
            "ppl": ppl,
            "world_size": world_size,
            "shared_memory_bank": args.shared_memory_bank,
            "unfreeze_hidden_to_slot": args.unfreeze_hidden_to_slot,
            "slot_init": args.slot_init,
            "slot_init_noise": args.slot_init_noise,
            "batch_size": args.batch_size,
            "skip_chunks": args.skip_chunks,
        }
        out_json = os.path.join(args.output_dir, "eval_results.json")
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Wrote %s", out_json)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
