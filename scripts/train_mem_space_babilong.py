#!/usr/bin/env python3
"""BABILong task-specific SFT for the Memory-Space adapter (Fix 2).

Continues from the champion mem_space adapter (``outputs/champion_ckpt/``) by
running short answer-only SFT on BABILong qa-tasks, with optional PG-19 LM
mixed in to mitigate catastrophic forgetting.

Design references
-----------------
* Base structure mirrors ``scripts/train_mem_space_pg19.py`` (DDP boot, mem-
  space patch, rotary fp32 fix, frozen-backbone optimisation, chunked stream
  training loop, adapter-only checkpoint save).
* Eval-parity prompt construction is implemented inside
  ``src/memory/mem_space/babilong_dataset.py`` — we instantiate that dataset
  here and route its samples through the same chunked-forward path that
  NIAH uses.  This guarantees the train-time prompt is byte-identical to the
  one ``scripts/run_babilong_mem_space.py`` builds at eval.
* PG-19 mix follows the ``niah_mix_fraction`` pattern from
  ``train_mem_space_pg19.py:687-720``: separate generators for the two data
  sources, sample one per step according to ``--pg19_mix_fraction``.

Adapter-config inheritance
--------------------------
When ``--init_adapter_config`` is provided, mem_space hyper-parameters
(num_slots, top_k, selector_dim, …) are read from the JSON file FIRST and
then any matching CLI flag the user passes wins (so we can ablate one knob
at a time).  This avoids a footgun where the CLI defaults silently override
the champion config.

Important constraints
---------------------
* The script is dependency-free of training launch state — it only sets up
  optimisation when invoked.  ``scripts/launch_mem_space_babilong.sh`` is the
  intended entry point and defaults to dry-run.
* No HuggingFace dataset download is triggered until ``__iter__`` runs inside
  a DataLoader worker, so ``python -m py_compile`` and the launch script's
  dry-run mode are safe to run on machines without network.
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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# FSDP imports (lazy-imported below for the FSDP path only; reside at top level
# so type checks `isinstance(model, FSDP)` work in helpers without re-importing).
try:
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
        StateDictType,
        FullStateDictConfig,
    )
    _FSDP_AVAILABLE = True
except ImportError:  # pragma: no cover — pre-2.0 PyTorch (we require 2.x)
    FSDP = None  # type: ignore[assignment]
    _FSDP_AVAILABLE = False

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Make sure third_party/babilong-pkg is on sys.path before we import the
# dataset class (which imports babilong.prompts).
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
# PG-19 dataset (re-used from train_mem_space_pg19.py — minimal duplication)
# --------------------------------------------------------------------------- #


class PG19ChunksDataset(Dataset):
    """Pre-tokenised PG-19 chunks loaded via numpy mmap.

    Same layout as ``PreTokenizedEvalDataset`` in train_mem_space_pg19.py:85.
    """

    def __init__(self, npy_path: str, seq_length: int, skip_chunks: int, max_chunks: int) -> None:
        data = np.load(npy_path, mmap_mode="r")
        self.data = data[skip_chunks: skip_chunks + max_chunks].astype(np.int32)
        self.seq_length = seq_length
        if len(self.data) == 0:
            total = len(np.load(npy_path, mmap_mode="r"))
            raise RuntimeError(
                f"PG19ChunksDataset is empty: skip={skip_chunks}, "
                f"max={max_chunks}, npy total chunks={total}. "
                f"Likely skip_chunks is past the end of the data. "
                f"Reduce --pg19_skip_chunks (e.g. to 200) or set "
                f"--pg19_mix_fraction 0.0 to skip PG-19 entirely."
            )
        logger.info(
            "Loaded %d PG-19 chunks of %d tokens from %s",
            len(self.data), self.seq_length, npy_path,
        )

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        tokens = torch.tensor(self.data[idx], dtype=torch.long)[: self.seq_length]
        return {"input_ids": tokens, "labels": tokens.clone(), "is_babilong": False}


def pg19_collate_fn(batch):
    return {
        "input_ids":   torch.stack([b["input_ids"] for b in batch]),
        "labels":      torch.stack([b["labels"]    for b in batch]),
        "is_babilong": False,
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
# Mem-space helpers (verbatim from train_mem_space_pg19.py)
# --------------------------------------------------------------------------- #


def _mem_space_params(model: torch.nn.Module) -> List[torch.nn.Parameter]:
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
    # L3 summary pool params (shared single module on root)
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        for p in l3_pool.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    # L2 token compressor params (shared single module on root, Phase 11)
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
    # Reset L3 summary state (cold start for new example)
    if hasattr(root, "_l3_summary_for_next_chunk"):
        root._l3_summary_for_next_chunk = None
    l3_pool = getattr(root, "_l3_pool", None)
    if l3_pool is not None:
        # Old name (kept for backward compat) and new names from BPTT fix
        l3_pool._current_summary = None
        l3_pool._prev_chunk_h = None
        if hasattr(l3_pool, "_chunk_summary_cache"):
            l3_pool._chunk_summary_cache = None


def _detach_banks(model: torch.nn.Module) -> None:
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


# --------------------------------------------------------------------------- #
# FSDP helpers (Phase 11 retry, 2026-05-16)
# --------------------------------------------------------------------------- #


def _is_distributed_wrapper(m: torch.nn.Module) -> bool:
    """True if ``m`` is wrapped in DDP or top-level FSDP (i.e. ``m.module`` is
    the user model)."""
    if isinstance(m, DDP):
        return True
    if _FSDP_AVAILABLE and FSDP is not None and isinstance(m, FSDP):
        return True
    return False


def _wrap_model_fsdp(
    model: torch.nn.Module,
    local_rank: int,
    use_checkpoint_wrapper: bool,
) -> torch.nn.Module:
    """Wrap each MemorySpaceLayer (and L3 pool + L2 compressor if present) in
    FSDP. Frozen Llama backbone stays replicated.

    Strategy (Option (b) from FSDP_MIGRATION_PLAN_20260516.md §2):
      * Wrap each ``MemorySpaceLayer`` as its own FSDP unit (FULL_SHARD).
      * Wrap the shared L3 pool and L2 compressor as separate FSDP units.
      * Leave the frozen backbone replicated (no sharding overhead, read-only).
      * use_orig_params=True so the optimizer keeps original Parameter objects
        and no rewrite of `_mem_space_params` / optimizer step is needed.

    Args:
        model: model after ``apply_mem_space_to_model`` + ``_freeze_backbone``.
        local_rank: GPU index for ``device_id``.
        use_checkpoint_wrapper: if True, wrap each MemorySpaceLayer in
            FSDP-native ``checkpoint_wrapper`` BEFORE the FSDP wrap so
            activations are recomputed on backward without conflicting with
            FSDP's reshard_after_forward.

    Returns:
        The same ``model`` object with in-place layer-list replacement plus a
        top-level FSDP wrap (so ``model.module`` is the original model).
    """
    if not _FSDP_AVAILABLE or FSDP is None:
        raise RuntimeError(
            "torch.distributed.fsdp is not available. Need PyTorch >= 2.0."
        )

    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        checkpoint_wrapper,
        CheckpointImpl,
    )

    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    common_fsdp_kwargs = dict(
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        device_id=local_rank,
        use_orig_params=True,
        sync_module_states=False,  # all ranks already hold identical init
    )

    root = getattr(model, "model", model)
    mem_layers = getattr(model, "_mem_space_layers", None) or []
    layers_list = root.layers  # nn.ModuleList of MemorySpaceLayer

    # 0) FSDP refuses 0-dim (scalar) Parameters (`numel() == 1` but `dim() == 0`).
    #    MemorySpaceLayer has `slot_output_gate` and `gate_param` declared as
    #    scalar nn.Parameter(torch.tensor(X)). Reshape them in-place to 1-D
    #    shape (1,) — all uses (`torch.sigmoid(p) * ...`, `torch.tanh(p) * x`,
    #    `.float().item()`) broadcast / work identically with a 1-element 1-D
    #    tensor.  Re-loaded scalar checkpoints are handled separately by the
    #    state_dict loader (load_state_dict with strict=False tolerates the
    #    shape mismatch when reshaping during load).
    for layer in mem_layers:
        for _pname in ("slot_output_gate", "gate_param"):
            _p = getattr(layer, _pname, None)
            if _p is not None and _p.dim() == 0:
                _new = torch.nn.Parameter(_p.detach().reshape(1).clone())
                # Preserve requires_grad (these are trainable in v2).
                _new.requires_grad_(_p.requires_grad)
                setattr(layer, _pname, _new)

    # 1) Wrap each MemorySpaceLayer in-place inside root.layers.
    #
    # Activation-checkpointing strategy with FSDP:
    #   - The trainable mem_space-only params are inside an FSDP unit; the
    #     frozen LlamaDecoderLayer is inside the wrapper at `self.wrapped_layer`
    #     and has NO FSDP wrap.
    #   - MemorySpaceLayer._maybe_ckpt_wrapped_layer applies a manual
    #     ``torch.utils.checkpoint(use_reentrant=False)`` around ONLY the
    #     frozen wrapped_layer call (not the whole MemorySpaceLayer). Since
    #     the frozen layer is not FSDP-managed, manual ckpt is safe — there
    #     is no reshard_after_forward to fight with.
    #   - We do NOT use FSDP-native ``checkpoint_wrapper`` on the outer
    #     MemorySpaceLayer.  Initial trials with checkpoint_wrapper around the
    #     FSDP unit caused state-machine errors during the BABILong chunked
    #     training step (multi-chunk no_grad + grad sequence, see
    #     ``_chunked_train_step``) of the form
    #         "ValueError: expected to be in states [TrainingState.IDLE]
    #          but current state is TrainingState.FORWARD_BACKWARD"
    #     because each chunk re-enters _pre_backward_hook with stale state.
    #   - Therefore: ``use_checkpoint_wrapper`` is interpreted as a request
    #     to leave ``_inside_fsdp_unit`` UNSET so the manual ckpt path inside
    #     MemorySpaceLayer is enabled.
    n_wrapped = 0
    for i, layer in enumerate(layers_list):
        # We only wrap MemorySpaceLayer instances; if for some reason a layer
        # was not patched (layer_indices=None patches all, but be defensive),
        # leave it alone.
        if layer not in mem_layers:
            continue
        if not use_checkpoint_wrapper:
            # Tell MemorySpaceLayer to skip the manual ckpt (it would be a
            # no-op anyway since gradient_checkpointing flag is False).
            layer._inside_fsdp_unit = True
        # else: leave _inside_fsdp_unit unset → manual torch.utils.checkpoint
        # is used inside MemorySpaceLayer.forward around the frozen
        # wrapped_layer call only.
        wrapped = FSDP(layer, **common_fsdp_kwargs)
        layers_list[i] = wrapped
        n_wrapped += 1

    # 2) Wrap shared L3 pool / L2 compressor (these hold trainable params too).
    l3_pool = getattr(model, "_l3_pool", None)
    if l3_pool is not None:
        # l3_pool was added via root.add_module("l3_pool", l3_pool) in patch.py
        wrapped_l3 = FSDP(l3_pool, **common_fsdp_kwargs)
        setattr(root, "l3_pool", wrapped_l3)
        model._l3_pool = wrapped_l3

    l2_comp = getattr(model, "_l2_compressor", None)
    if l2_comp is not None:
        wrapped_l2 = FSDP(l2_comp, **common_fsdp_kwargs)
        setattr(root, "l2_compressor", wrapped_l2)
        model._l2_compressor = wrapped_l2

    # 3) NO top-level FSDP wrap.
    #
    # Earlier versions wrapped the whole model in a top-level FSDP with
    # ShardingStrategy.NO_SHARD to provide a single root for state_dict gather.
    # However, with ``use_orig_params=True`` a top-level NO_SHARD wrap tracks
    # the FROZEN embedding/lm_head params in its flat-param table; on the second
    # forward (or after an optimizer step) FSDP's `_writeback_orig_params`
    # raises "Cannot writeback when the parameter shape changes" because the
    # inner FSDP units have already replaced sub-modules' original Parameter
    # objects. Skipping the top-level wrap avoids this entirely; the per-layer
    # FSDP units still expose `state_dict_type()` correctly when the context
    # manager is given the (un-wrapped) top-level model — FSDP walks
    # ``m.modules()`` and applies the policy to each FSDP submodule.
    #
    # Tag the model so ``_save_adapter`` knows to use the FSDP gather path
    # even though ``isinstance(model, FSDP)`` is now False.
    setattr(model, "_uses_partial_fsdp", True)

    logger.info(
        "FSDP wrap complete: %d MemorySpaceLayer units + (l3_pool=%s, l2=%s); "
        "top-level NO_SHARD over backbone.",
        n_wrapped,
        "yes" if l3_pool is not None else "no",
        "yes" if l2_comp is not None else "no",
    )
    return model


# --------------------------------------------------------------------------- #
# Args
# --------------------------------------------------------------------------- #


# Adapter-config JSON keys that are training-only (not MemorySpaceConfig fields).
_ADAPTER_CFG_TRAIN_ONLY_KEYS = {
    "max_train_steps", "max_steps", "effective_max_steps", "lr", "niah_max_N",
    "niah_mix_fraction",
}

# Translate adapter_config.json key → MemorySpaceConfig dataclass field name.
# (Same logic as scripts/run_babilong_mem_space.py:_ADAPTER_CONFIG_FIELD_MAP.)
_ADAPTER_CFG_FIELD_MAP = {
    "writeback_warmup_steps": "writeback_gate_warmup_steps",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Memory-Space SFT on BABILong (Fix 2)",
    )

    # Base
    p.add_argument("--model_path", type=str, required=True,
                   help="Path to base Llama-3-8B(-Instruct) directory")
    p.add_argument("--output_dir", type=str, required=True)

    # Champion warm-start
    p.add_argument("--init_checkpoint", type=str, default=None,
                   help="Path to a mem_space adapter .pt (e.g. champion_ckpt). "
                        "If None, mem_space starts from random init.")
    p.add_argument("--init_adapter_config", type=str, default=None,
                   help="Path to adapter_config.json. mem_space hyperparams "
                        "are inherited from this file by default; CLI flags "
                        "explicitly passed by the user override the config.")

    # BABILong data
    p.add_argument("--babilong_dataset", type=str, default="RMT-team/babilong")
    p.add_argument("--babilong_tasks", type=str, default="qa1",
                   help="Comma-separated qa-task list: e.g. qa1 or qa1,qa2,qa5")
    p.add_argument("--babilong_lengths", type=str, default="1k,2k",
                   help="Comma-separated length splits: e.g. 1k,2k or 1k,2k,4k,8k")
    p.add_argument("--babilong_length_weights", type=str, default="",
                   help="Optional comma-separated non-negative weights, one per "
                        "entry in --babilong_lengths, controlling the per-step "
                        "length sampling distribution. Empty (default) = uniform. "
                        "Example for v3 short-fix: --babilong_lengths=0k,1k,2k,4k "
                        "--babilong_length_weights=3,3,1,1 oversamples 0k+1k.")
    p.add_argument("--babilong_limit_per_cell", type=int, default=0,
                   help="0 = use all rows; >0 = sample only first N rows per cell")
    p.add_argument("--use_chat_template", action="store_true",
                   help="Wrap prompt with tokenizer chat template (use for "
                        "Instruct backbones).")

    # PG-19 mix (optional, for forgetting mitigation)
    p.add_argument("--pg19_data", type=str, default="data/pg19_chunks_llama3.npy",
                   help="Path to pre-tokenised pg19 chunks .npy")
    p.add_argument("--pg19_mix_fraction", type=float, default=0.2,
                   help="Probability of drawing a PG-19 LM batch each step "
                        "(0.0 = pure BABILong SFT).")
    p.add_argument("--pg19_max_chunks", type=int, default=2000,
                   help="How many PG-19 chunks to expose to the loader (mmap).")
    p.add_argument("--pg19_skip_chunks", type=int, default=200,
                   help="Skip first N PG-19 chunks (hold-out window so we don't "
                        "train on chunks the NIAH/BABILong eval might reuse). "
                        "Default 200 matches the actual size of "
                        "data/pg19_chunks_llama3.npy (~5916 chunks). Setting "
                        "this higher than the dataset size makes "
                        "PG19ChunksDataset empty and silently disables PG-19 "
                        "mix — the dataset now raises if that happens.")

    # Training shape
    p.add_argument("--max_seq_len", type=int, default=4096,
                   help="Hard cap on a sample's token length (BABILong dataset "
                        "left-truncates context to fit; PG-19 chunks are this "
                        "size already).")
    p.add_argument("--chunk_size", type=int, default=1024,
                   help="mem_space forward-pass chunk size: long BABILong "
                        "samples are split into chunks of this length and "
                        "streamed through the bank, mirroring the eval path.")
    p.add_argument("--batch_size", type=int, default=1,
                   help="Per-rank batch size for PG-19 loader. BABILong loader "
                        "uses batch_size=1 (variable-length samples).")
    p.add_argument("--num_workers", type=int, default=2)

    # Optim
    p.add_argument("--total_steps", type=int, default=500,
                   help="Total optimiser steps across BABILong + PG-19 mix.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proj_grad_clip", type=float, default=0.1,
                   help="Per-param clip for slot_to_hidden / hidden_to_slot "
                        "(mirrors Fix L-2 in train_mem_space_pg19.py).")

    # mem_space hyper-params (defaults match champion_ckpt/adapter_config.json)
    p.add_argument("--num_slots", type=int, default=512)
    p.add_argument("--top_k", type=int, default=64)
    p.add_argument("--selector_dim", type=int, default=128)
    p.add_argument("--writeback_gate_max", type=float, default=0.3)
    p.add_argument("--writeback_warmup_steps", type=int, default=0,
                   help="0 = β fully ramped at step 0 (we are CONTINUING from a "
                        "trained adapter, not warming up from scratch).")
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

    # Dual-gate (LM2-inspired) writeback ─ optional alternative to single-β EMA.
    # When enabled, replaces ``slots ← (1-β)·slots + β·new`` with
    # ``slots ← g_forget·slots + g_in·tanh(new)`` where g_in/g_forget are
    # learnt sigmoid gates conditioned on (new_repr, current_slot).
    # The new params (gate_proj_new, gate_proj_mem, gate_bias) are NOT in the
    # champion adapter and start from xavier-uniform init; forget_bias_init
    # should be ≥1.5 so initial g_forget is high (slots not wiped at step 0).
    p.add_argument("--use_dual_gate", action="store_true", default=False,
                   help="Replace EMA β with LM2-style dual gate (input + forget). "
                        "Cold-starts new gate params; pair with high "
                        "--forget_bias_init to keep slot content at step 0.")
    p.add_argument("--input_bias_init", type=float, default=0.0,
                   help="Bias on input gate logit (sigmoid → g_in at init).")
    p.add_argument("--forget_bias_init", type=float, default=2.0,
                   help="Bias on forget gate logit. Default 2.0 → "
                        "sigmoid(2.0)=0.88, slots retained at step 0.")
    p.add_argument("--dual_gate_tanh_new", action="store_true", default=True,
                   help="Apply tanh to O_mem_slot before gating "
                        "(LM2 default; bounds new content to [-1,1]).")

    # L3 Summary-Token module (Q-Former-style cross-attn pool over top-layer H).
    p.add_argument("--use_l3_summary", action="store_true", default=False,
                   help="Enable L3 summary-token module (64 dense summary tokens "
                        "per chunk via Q-Former-style cross-attn pool).")
    p.add_argument("--l3_n_summary", type=int, default=64,
                   help="Number of L3 summary tokens per chunk (K_sum).")
    p.add_argument("--l3_n_layers", type=int, default=2,
                   help="Number of cross-attn blocks in L3 pool (1=~50M, 2=~150M).")
    p.add_argument("--l3_n_heads", type=int, default=8,
                   help="Number of attention heads in L3 cross-attn blocks.")
    p.add_argument("--disable_l1_inject", action="store_true", default=False,
                   help="Skip L1 slot prepending + dual-gate writeback. "
                        "Used for pure-L3 ablation (only L3 summary tokens active).")

    # L2 Token-Compressed KV memory (NSA / DeepSeek-V4-CSA style learned-gated
    # attention pool over groups of g=16 tokens). Phase 11 (2026-05-16).
    p.add_argument("--use_l2", action="store_true", default=False,
                   help="Enable L2 token-compressed KV memory (256 latents per "
                        "4k chunk via learned-gated soft-pool over groups of g tokens).")
    p.add_argument("--l2_compress_ratio", type=int, default=16,
                   help="L2 group / window size g (chunk_size/g latents per chunk).")
    p.add_argument("--l2_d_c", type=int, default=512,
                   help="L2 latent / content dimension (matches V2 MLA).")
    p.add_argument("--l2_d_h_rope", type=int, default=64,
                   help="L2 decoupled-RoPE per-latent dimension.")
    p.add_argument("--l2_init_scale", type=float, default=0.001,
                   help="L2 kv_b weight init std (near-zero so L2 contribution starts ≈ 0).")

    # Activation-memory reduction (Phase 11, 2026-05-16)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False,
                   help="Wrap wrapped_layer.forward in torch.utils.checkpoint for "
                        "L2-induced memory pressure (~50% activation memory cut at "
                        "~2x compute). Required for L1+L2+L3 stack on H20 (97GB) at "
                        "chunk_size=1024 + 4k context.")

    # FSDP migration (Phase 11 retry, 2026-05-16): shard trainable mem_space
    # adapter (~1.4 GB params + 16.6 GB AdamW + 5.5 GB grads) across ranks via
    # FSDP option (b) — keep frozen Llama-3-8B backbone replicated. Saves ~22
    # GB/rank vs DDP, enabling P11 training on H20 (97 GB) which kept OOM'ing
    # under DDP at ~48 GB peak due to fragmentation + cuBLAS workspace failure.
    p.add_argument("--use_fsdp", action="store_true", default=False,
                   help="Use FullyShardedDataParallel (ZeRO-3) on trainable "
                        "mem_space layers instead of DDP. Frozen backbone stays "
                        "replicated. Required for L1+L2+L3 stack on H20 (97 GB).")

    # Misc
    p.add_argument("--attn_impl", type=str, default="sdpa",
                   choices=["sdpa", "eager", "flash_attention_2"])
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--save_interval", type=int, default=100,
                   help="Save intermediate adapter checkpoint every N steps "
                        "(0 = only save at end).")
    p.add_argument("--skip_mem_when_short", action="store_true", default=False,
                   help="v3 short-fix (2026-05-16): when the whole sample fits "
                        "in one chunk (n_chunks==1), suppress L1/L3 writeback "
                        "for that step. The forward still goes through the "
                        "mem_space layer (selector + joint-attn) but slots are "
                        "not written back, so the model learns to ignore the "
                        "memory bank at short range.")

    return p.parse_args()


def merge_adapter_config_into_args(args: argparse.Namespace) -> argparse.Namespace:
    """Populate mem-space hyperparams from --init_adapter_config when not explicit.

    We can't tell directly which CLI flags the user actually passed (argparse
    fills defaults silently), so we use the heuristic: if a user wants to
    override a champion hyper-param, they can pass the CLI flag and we'll
    detect it via ``sys.argv``.  Anything *not* on the command line gets
    overwritten by the JSON file.
    """
    if not args.init_adapter_config:
        return args
    if not os.path.isfile(args.init_adapter_config):
        logger.warning("init_adapter_config %s not found — using CLI defaults",
                       args.init_adapter_config)
        return args

    explicit = {a.lstrip("-").split("=")[0] for a in sys.argv[1:] if a.startswith("--")}
    with open(args.init_adapter_config, "r") as f:
        cfg = json.load(f)

    # Map adapter_config keys → argparse attr names where they differ.
    cfg_to_attr = {
        "num_slots":              "num_slots",
        "top_k":                  "top_k",
        "selector_dim":           "selector_dim",
        "writeback_gate_max":     "writeback_gate_max",
        "writeback_warmup_steps": "writeback_warmup_steps",
        "load_balance_weight":    "load_balance_weight",
        "entropy_aux_weight":     "entropy_aux_weight",
        "selector_temperature":   "selector_temperature",
        "key_repulsion_weight":   "key_repulsion_weight",
        "key_repulsion_threshold":"key_repulsion_threshold",
        "peak_routing_weight":    "peak_routing_weight",
        "slot_value_norm_cap":    "slot_value_norm_cap",
        "slot_init":              "slot_init",
        "slot_init_noise":        "slot_init_noise",
        "unfreeze_hidden_to_slot":"unfreeze_hidden_to_slot",
        "shared_memory_bank":     "shared_memory_bank",
        "swa_window":             "swa_window",
    }
    inherited = []
    for k_json, attr in cfg_to_attr.items():
        if k_json not in cfg:
            continue
        if attr in explicit:
            continue   # user override wins
        setattr(args, attr, cfg[k_json])
        inherited.append(f"{k_json}={cfg[k_json]}")
    if inherited:
        logger.info("Inherited from adapter_config: %s", ", ".join(inherited))
    return args


# --------------------------------------------------------------------------- #
# Model build
# --------------------------------------------------------------------------- #


def build_model(args, device, dtype) -> torch.nn.Module:
    llama_cfg = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
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
    )

    # Snapshot rotary inv_freq in fp32 BEFORE the .to(dtype=bf16) cast so the
    # H7 fix v2 (see train_mem_space_pg19.py:540-581) is preserved.
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

    # Warm-start from champion adapter
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
        logger.info("init_checkpoint loaded: %d keys missing=%d unexpected=%d",
                    len(cleaned), len(missing), len(unexpected))
        # Force step_counter so β/warmup_frac is fully ramped (we resume).
        from src.memory.mem_space.layer import MemorySpaceLayer as _MSL
        warmup = max(args.writeback_warmup_steps, 1)
        for w in getattr(model, "_mem_space_layers", []):
            if isinstance(w, _MSL):
                w.step_counter = warmup
    elif args.init_checkpoint:
        logger.warning("init_checkpoint=%s does not exist — proceeding from random",
                       args.init_checkpoint)

    return model


# --------------------------------------------------------------------------- #
# Training step helpers
# --------------------------------------------------------------------------- #


def _set_skip_writeback(model: torch.nn.Module, value: bool) -> None:
    """Toggle the per-call ``_skip_writeback_this_call`` flag on every
    MemorySpaceLayer wrapper. Used by ``_chunked_train_step`` to suppress
    L1/L3 writeback when the whole sample fits in one chunk (v3 short-fix)."""
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return
    for w in mem_layers:
        w._skip_writeback_this_call = bool(value)


def _chunked_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,   # [1, total_len]
    labels:    torch.Tensor,   # [1, total_len]
    chunk_size: int,
    device: torch.device,
    skip_mem_when_short: bool = False,
):
    """Stream input through mem_space in chunks of ``chunk_size``.

    Strategy
    --------
    1.  Reset banks at the start of each sample (per-doc state).
    2.  All but the LAST chunk forwarded with no_grad: the bank accumulates
        memory but no gradients flow.  This matches the streaming-NIAH path
        in ``train_mem_space_pg19.py:743-746``.
    3.  Last chunk gets a gradient-bearing forward; the answer-only label
        mask ensures only answer tokens in that final chunk drive the loss.
    4.  Aux losses from the last chunk are added to lm_loss in the caller.

    For samples shorter than chunk_size, just one gradient-bearing forward
    happens (no streaming overhead).
    """
    _reset_banks(model)
    total_len = input_ids.shape[1]
    n_chunks = max(1, math.ceil(total_len / chunk_size))

    if n_chunks == 1:
        # v3 short-fix: optionally suppress L1/L3 writeback so the model
        # learns to rely on standard self-attn at short range.  Selector +
        # joint-attn still run, but the bank is not updated this call —
        # which combined with _reset_banks() above means slots stay at the
        # init/random distribution and the model effectively bypasses memory.
        if skip_mem_when_short:
            _set_skip_writeback(model, True)
        try:
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
        finally:
            if skip_mem_when_short:
                _set_skip_writeback(model, False)
        return out

    # Chunk-split.  We make sure ALL answer-bearing tokens land in the last
    # chunk; otherwise loss may be 0 (no labels) and the gradient signal
    # vanishes.  The dataset puts answer tokens at the very end of the
    # sequence, so the last chunk always contains the answer span.
    pieces_in  = list(input_ids[0].split(chunk_size))
    pieces_lbl = list(labels[0].split(chunk_size))

    with torch.no_grad():
        for ci, _l in zip(pieces_in[:-1], pieces_lbl[:-1]):
            model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)

    last_in  = pieces_in[-1].unsqueeze(0).to(device)
    last_lbl = pieces_lbl[-1].unsqueeze(0).to(device)
    out = model(input_ids=last_in, labels=last_lbl, use_cache=False)
    return out


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

    if is_main(rank):
        os.makedirs(args.output_dir, exist_ok=True)
        logger.info("BABILong SFT (Fix 2) | model=%s | tasks=%s | lengths=%s | "
                    "pg19_mix=%.2f | total_steps=%d | chunk_size=%d | world_size=%d",
                    args.model_path, args.babilong_tasks, args.babilong_lengths,
                    args.pg19_mix_fraction, args.total_steps, args.chunk_size,
                    world_size)

    # --- tokenizer --- #
    llama_cfg = LlamaConfig.from_pretrained(args.model_path, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, config=llama_cfg, trust_remote_code=True, local_files_only=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- model + warm-start --- #
    model = build_model(args, device, dtype)
    _freeze_backbone(model)

    if is_main(rank):
        n_layers = len(model.model.layers)
        n_trainable = sum(p.numel() for p in _mem_space_params(model))
        logger.info("Patched %d decoder layers | mem_space trainable: %.2fM params",
                    n_layers, n_trainable / 1e6)

    # DDP-side aux-loss hook (mirrors train_mem_space_pg19.py:617-630)
    if world_size > 1:
        _mem_layers_hook = getattr(model, "_mem_space_layers", None)

        def _slot_key_aux_hook(module, inputs, output):
            if _mem_layers_hook is None or output.loss is None:
                return
            aux_total = torch.zeros((), device=output.loss.device, dtype=output.loss.dtype)
            for w in _mem_layers_hook:
                for key in ("key_repulsion", "peak_routing"):
                    v = w.last_aux_losses.pop(key, None)
                    if v is not None and v.requires_grad:
                        aux_total = aux_total + v
            if aux_total.requires_grad:
                output.loss = output.loss + aux_total
        model.register_forward_hook(_slot_key_aux_hook)

    if world_size > 1:
        if args.use_fsdp:
            if not _FSDP_AVAILABLE:
                raise RuntimeError(
                    "--use_fsdp set but torch.distributed.fsdp is unavailable."
                )
            # FSDP path: wrap trainable mem_space units. Use FSDP-native
            # checkpoint_wrapper when --gradient_checkpointing is set (manual
            # torch.utils.checkpoint is incompatible with FSDP's
            # reshard_after_forward). MemorySpaceLayer._maybe_ckpt_wrapped_layer
            # is told to skip the manual ckpt path via _inside_fsdp_unit=True.
            model = _wrap_model_fsdp(
                model,
                local_rank=local_rank,
                use_checkpoint_wrapper=args.gradient_checkpointing,
            )
            if is_main(rank):
                logger.info("Using FSDP (Option b): trainable mem_space sharded, "
                            "backbone replicated. use_ckpt_wrapper=%s",
                            args.gradient_checkpointing)
        else:
            model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                        find_unused_parameters=True)

    # --- BABILong dataset --- #
    babilong_tasks = [t.strip() for t in args.babilong_tasks.split(",") if t.strip()]
    babilong_lengths = [l.strip() for l in args.babilong_lengths.split(",") if l.strip()]
    babilong_length_weights: Optional[List[float]] = None
    if args.babilong_length_weights.strip():
        try:
            babilong_length_weights = [
                float(w) for w in args.babilong_length_weights.split(",") if w.strip()
            ]
        except ValueError as _e:
            raise ValueError(
                f"--babilong_length_weights={args.babilong_length_weights!r} "
                f"must be comma-separated floats"
            ) from _e
        if len(babilong_length_weights) != len(babilong_lengths):
            raise ValueError(
                f"--babilong_length_weights has {len(babilong_length_weights)} "
                f"entries but --babilong_lengths has {len(babilong_lengths)}."
            )
        if is_main(rank):
            logger.info(
                "BABILong length-weighted sampling: %s",
                ", ".join(f"{l}={w:g}" for l, w in zip(babilong_lengths, babilong_length_weights)),
            )

    # Rank-0 prefetch of BABILong dataset cache.
    # ----------------------------------------------------------------------
    # The HF datasets library is NOT distributed-safe when the dataset has
    # never been cached locally: every rank tries to download and call
    # dataset_infos.json simultaneously, deadlocking on 404 retries while
    # rank-0 holds the cache lock (observed 2026-05-15 — training stuck
    # 5 min on RMT-team/babilong with rank-0 in sleep, rank-1+ in NCCL
    # barrier).  HuggingFace's recommended pattern is: rank-0 fetches first,
    # all other ranks wait on a torch.distributed barrier, then they read
    # the populated cache.
    #
    # The babilong_dataset.py loader calls
    #     datasets.load_dataset(args.babilong_dataset, length)
    # then indexes data[task].
    #
    # 2026-05-15 update: per-length prefetch is NOT enough. ``BABILongTrainDataset``
    # lazy-loads in ``_load_split(task, length)``, and that triggers a
    # ``load_dataset`` call PER (task, length) pair the first time each is
    # iterated by each rank.  Even with HF_HUB_OFFLINE=1, cache lookups
    # competing across 8 ranks deadlock.  Fix: rank-0 prefetches every
    # (task, length) combination and indexes data[task] to force the per-task
    # split to materialise in cache too.
    if world_size > 1:
        if rank == 0:
            logger.info("[rank 0] Pre-fetching BABILong dataset cache for "
                        "tasks=%s lengths=%s ...", babilong_tasks, babilong_lengths)
            try:
                import datasets as _hfds  # noqa: WPS433
                for _length in babilong_lengths:
                    try:
                        _data = _hfds.load_dataset(args.babilong_dataset, _length)
                        for _task in babilong_tasks:
                            try:
                                # Touch every per-task split so it materialises
                                # in cache for the lazy loader path.
                                _ = _data[_task]
                            except Exception as _e_task:  # pragma: no cover
                                logger.warning("  prefetch task=%s length=%s: %s",
                                               _task, _length, _e_task)
                        logger.info("  cached length=%s (tasks=%s)", _length, babilong_tasks)
                    except Exception as _e:  # pragma: no cover
                        logger.warning("  prefetch failed length=%s: %s",
                                       _length, _e)
                logger.info("[rank 0] BABILong pre-fetch complete")
            except Exception as _e:  # pragma: no cover
                logger.warning("[rank 0] BABILong prefetch crashed: %s", _e)
        # All ranks (0 and others) wait here; non-zero ranks block until the
        # cache is populated, then read from it without re-downloading.
        dist.barrier()

    babilong_ds = BABILongTrainDataset(
        tokenizer=tokenizer,
        dataset_name=args.babilong_dataset,
        tasks=babilong_tasks,
        lengths=babilong_lengths,
        max_seq_len=args.max_seq_len,
        seed=args.seed + rank,
        use_chat_template=args.use_chat_template,
        limit_per_cell=args.babilong_limit_per_cell,
        length_weights=babilong_length_weights,
    )
    babilong_loader = DataLoader(
        babilong_ds, batch_size=1,
        num_workers=args.num_workers, collate_fn=babilong_collate_fn,
    )
    _babilong_iter = iter(babilong_loader)

    # --- PG-19 dataset (optional mix) --- #
    pg19_iter: Optional[object] = None
    pg19_sampler = None
    if args.pg19_mix_fraction > 0.0:
        if not os.path.isfile(args.pg19_data):
            logger.warning("pg19_mix_fraction=%.2f but %s missing — DISABLING PG-19 mix",
                           args.pg19_mix_fraction, args.pg19_data)
            args.pg19_mix_fraction = 0.0
        else:
            pg19_ds = PG19ChunksDataset(
                npy_path=args.pg19_data,
                seq_length=args.max_seq_len,
                skip_chunks=args.pg19_skip_chunks,
                max_chunks=args.pg19_max_chunks,
            )
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

            pg19_iter = _cycle_pg19()

    mix_rng = random.Random(args.seed + rank)

    # --- optimiser --- #
    trainable = _mem_space_params(
        model.module if _is_distributed_wrapper(model) else model
    )
    if not trainable:
        raise RuntimeError("No mem_space trainable params found.")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0,
                                  betas=(0.9, 0.95))

    # --- training loop --- #
    model.train()
    n_done = 0
    n_nonfinite = 0
    n_babilong = 0
    n_pg19 = 0
    t0 = time.time()

    while n_done < args.total_steps:
        use_pg19 = (pg19_iter is not None) and (mix_rng.random() < args.pg19_mix_fraction)
        if use_pg19:
            try:
                batch = next(pg19_iter)
            except StopIteration:
                continue
        else:
            try:
                batch = next(_babilong_iter)
            except StopIteration:
                _babilong_iter = iter(babilong_loader)
                batch = next(_babilong_iter)

        optimizer.zero_grad(set_to_none=True)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels    = batch["labels"].to(device, non_blocking=True)

        if use_pg19:
            # Standard NTP step (matches PG-19 path in train_mem_space_pg19).
            _detach_banks(model)
            out = model(input_ids=input_ids, labels=labels, use_cache=False)
            n_pg19 += 1
        else:
            # BABILong step: chunked stream, answer-only loss handled by labels.
            out = _chunked_train_step(
                model, input_ids, labels, args.chunk_size, device,
                skip_mem_when_short=args.skip_mem_when_short,
            )
            n_babilong += 1

        lm_loss = out.loss
        aux_loss = _collect_aux_loss(model, device)
        loss = lm_loss + aux_loss

        if (lm_loss is None) or not torch.isfinite(loss):
            logger.warning("[step %d] non-finite loss lm=%s aux=%s — skipping",
                           n_done,
                           "None" if lm_loss is None else f"{lm_loss.item():.4f}",
                           f"{float(aux_loss.item()):.6f}")
            n_nonfinite += 1
            _step_counters_inc(model)
            n_done += 1
            continue

        loss.backward()

        # Per-projection grad clip (Fix L-2).
        _grad_root = model.module if _is_distributed_wrapper(model) else model
        for _n, _p in _grad_root.named_parameters():
            if _p.grad is not None and ("slot_to_hidden" in _n or "hidden_to_slot" in _n):
                torch.nn.utils.clip_grad_norm_([_p], args.proj_grad_clip)
        torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

        optimizer.step()
        _step_counters_inc(model)
        n_done += 1

        if is_main(rank) and (n_done % args.log_interval == 0):
            tag = "PG19" if use_pg19 else "BABI"
            logger.info(
                "[%s step %d/%d] lm_loss=%.4f aux=%.6f bab=%d pg19=%d nf=%d dt=%.1fs",
                tag, n_done, args.total_steps,
                lm_loss.item(), float(aux_loss.item()),
                n_babilong, n_pg19, n_nonfinite, time.time() - t0,
            )

        # FSDP state_dict gather is a collective: ALL ranks must enter the
        # context manager (rank 0 then writes the gathered state). For DDP,
        # only rank 0 needs to call _save_adapter. _save_adapter() handles
        # the rank gating internally for FSDP.
        if (args.save_interval > 0
                and n_done % args.save_interval == 0
                and n_done < args.total_steps):
            if args.use_fsdp:
                _save_adapter(model, args, n_done)
            elif is_main(rank):
                _save_adapter(model, args, n_done)

    if args.use_fsdp:
        _save_adapter(model, args, n_done, final=True)
    elif is_main(rank):
        _save_adapter(model, args, n_done, final=True)
    if is_main(rank):
        logger.info("Training complete: steps=%d babilong=%d pg19=%d non-finite=%d",
                    n_done, n_babilong, n_pg19, n_nonfinite)

    if world_size > 1:
        dist.destroy_process_group()


def _save_adapter(model, args, step: int, final: bool = False) -> None:
    """Save mem_space adapter weights + config (matches train_mem_space_pg19 layout).

    Supports both DDP and FSDP. For FSDP, gathers FULL_STATE_DICT to rank 0
    (CPU-offloaded) via the FSDP.state_dict_type context manager. Only rank 0
    writes the checkpoint.
    """
    fragments = (
        "selector", "gate_param", "slot_output_gate",
        "slot_to_hidden", "hidden_to_slot", "memory_bank",
        # Dual-gate (LM2-style) writeback params — only populated when
        # --use_dual_gate is set, but always include in fragments so the
        # ckpt round-trips dual-gate weights when present.
        "gate_proj_new", "gate_proj_mem", "gate_bias",
        # L3 summary pool params (Q-Former-style cross-attn pool)
        "l3_pool",
        # L2 token compressor params (Phase 11)
        "l2_compressor",
    )

    is_fsdp = _FSDP_AVAILABLE and FSDP is not None and (
        isinstance(model, FSDP) or getattr(model, "_uses_partial_fsdp", False)
    )

    if is_fsdp:
        # FSDP path: gather full state_dict to rank 0 (CPU-offloaded). All ranks
        # must execute the context, but only rank 0 receives the full state.
        save_policy = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(
            model, StateDictType.FULL_STATE_DICT, save_policy
        ):
            full_state = model.state_dict()
        if dist.is_initialized() and dist.get_rank() != 0:
            return
        # Strip any "module." / "_fsdp_wrapped_module." / "_checkpoint_wrapped_module."
        # prefix added by FSDP / activation checkpoint wrapper, then filter by
        # fragments so we only keep mem_space-related weights.
        cleaned = {}
        for k, v in full_state.items():
            nk = k
            for prefix_marker in (
                "_fsdp_wrapped_module.",
                "_checkpoint_wrapped_module.",
                "module.",
            ):
                nk = nk.replace(prefix_marker, "")
            cleaned[nk] = v
        state = {
            k: v.detach().cpu() if isinstance(v, torch.Tensor) else v
            for k, v in cleaned.items()
            if any(frag in k for frag in fragments)
        }
    else:
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
            "num_slots":               args.num_slots,
            "top_k":                   args.top_k,
            "selector_dim":            args.selector_dim,
            "writeback_gate_max":      args.writeback_gate_max,
            "writeback_warmup_steps":  args.writeback_warmup_steps,
            "load_balance_weight":     args.load_balance_weight,
            "entropy_aux_weight":      args.entropy_aux_weight,
            "selector_temperature":    args.selector_temperature,
            "key_repulsion_weight":    args.key_repulsion_weight,
            "key_repulsion_threshold": args.key_repulsion_threshold,
            "peak_routing_weight":     args.peak_routing_weight,
            "slot_value_norm_cap":     args.slot_value_norm_cap,
            "slot_init":               args.slot_init,
            "slot_init_noise":         args.slot_init_noise,
            "shared_memory_bank":      args.shared_memory_bank,
            "unfreeze_hidden_to_slot": args.unfreeze_hidden_to_slot,
            "swa_window":              args.swa_window,
            # Dual-gate (LM2-style) writeback config — required at eval time
            # so MemorySpaceConfig is built with the right gate path.
            "use_dual_gate":           args.use_dual_gate,
            "input_bias_init":         args.input_bias_init,
            "forget_bias_init":        args.forget_bias_init,
            "dual_gate_tanh_new":      args.dual_gate_tanh_new,
            # L3 summary-token config
            "use_l3_summary":          args.use_l3_summary,
            "l3_n_summary":            args.l3_n_summary,
            "l3_n_layers":             args.l3_n_layers,
            "l3_n_heads":              args.l3_n_heads,
            # Pure-L3 ablation flag
            "disable_l1_inject":       args.disable_l1_inject,
            # L2 token-compressed KV memory config (Phase 11)
            "use_l2":                  args.use_l2,
            "l2_compress_ratio":       args.l2_compress_ratio,
            "l2_d_c":                  args.l2_d_c,
            "l2_d_h_rope":             args.l2_d_h_rope,
            "l2_init_scale":           args.l2_init_scale,
            "gradient_checkpointing":  args.gradient_checkpointing,
            "lr":                      args.lr,
            "total_steps":             args.total_steps,
            "babilong_tasks":          args.babilong_tasks,
            "babilong_lengths":        args.babilong_lengths,
            "pg19_mix_fraction":       args.pg19_mix_fraction,
            "use_chat_template":       args.use_chat_template,
            "init_checkpoint":         args.init_checkpoint,
            "step_at_save":            step,
            "final":                   final,
        }, f, indent=2)


if __name__ == "__main__":
    main()
