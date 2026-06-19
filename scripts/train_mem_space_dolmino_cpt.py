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

# FSDP imports (lazy-imported below for the FSDP path only; reside at top level
# so type checks `isinstance(model, FSDP)` work in helpers without re-importing).
# Ported from scripts/train_mem_space_babilong.py (Phase 11 retry, 2026-05-16).
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
from src.memory.mem_space.niah_chunked_dataset import (  # noqa: E402
    NIAHChunkedDataset,
    niah_chunked_collate_fn,
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
    # NOTE: timeout must be far larger than the slowest collective gap. The
    # step-500 checkpoint save writes a ~7.5GB adapter to shared CEPH FS on
    # rank0 while ranks 1-7 wait on dist.barrier(); with the old 30min timeout
    # this slow save was caught by the NCCL watchdog as a stuck collective and
    # the whole job SIGABRT'd. Use 2h so save/barrier never trips the watchdog.
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size,
                            timeout=timedelta(hours=2))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def is_main(rank: int) -> bool:
    return rank == 0


# --------------------------------------------------------------------------- #
# FSDP helpers (ported from train_mem_space_babilong.py, Phase 11 retry)
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
    unfreeze_backbone: bool = False,
) -> torch.nn.Module:
    """Wrap each MemorySpaceLayer (and L3 pool + L2 compressor + recon decoder
    if present) in FSDP. Frozen Llama backbone stays replicated.

    Strategy (Option (b) from FSDP_MIGRATION_PLAN_20260516.md §2):
      * Wrap each ``MemorySpaceLayer`` as its own FSDP unit (FULL_SHARD). NOTE
        the 32 frozen LlamaDecoderLayers live INSIDE the MemorySpaceLayers (as
        ``wrapped_layer``), so they are sharded together with the adapter — this
        is why full fine-tune (``unfreeze_backbone``) shards the ~7B decoder
        stack for free.
      * Wrap the shared L3 pool / L2 compressor / recon decoder as separate
        FSDP units.
      * Leave the frozen backbone replicated (no sharding overhead, read-only).
      * use_orig_params=True so the optimizer keeps original Parameter objects
        and no rewrite of `_mem_space_params` / optimizer step is needed.

    Full fine-tune (``unfreeze_backbone=True``, 2026-06-18): the embed_tokens,
    final norm, and lm_head live at the model ROOT (outside every MemorySpaceLayer)
    so without extra handling they stay replicated AND get NO gradient sync across
    ranks under FSDP (FSDP only reduce-scatters the params it manages). We wrap
    each of them as its own FSDP unit so their grads are reduced like the rest.

    Args:
        model: model after ``apply_mem_space_to_model`` + ``_freeze_backbone``.
        local_rank: GPU index for ``device_id``.
        use_checkpoint_wrapper: if True, leave ``_inside_fsdp_unit`` UNSET so the
            manual ``torch.utils.checkpoint`` path inside MemorySpaceLayer (which
            wraps ONLY the frozen wrapped_layer) stays active. We do NOT use
            FSDP-native checkpoint_wrapper — it caused state-machine errors with
            the chunked / TBPTT multi-backward training step (see babilong note).
        unfreeze_backbone: if True, also wrap embed_tokens / norm / lm_head so the
            full-FT root params get sharded + grad-synced.

    Returns:
        The same ``model`` object with in-place layer-list replacement (no
        top-level FSDP wrap), tagged ``_uses_partial_fsdp = True``.
    """
    if not _FSDP_AVAILABLE or FSDP is None:
        raise RuntimeError(
            "torch.distributed.fsdp is not available. Need PyTorch >= 2.0."
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
    #    scalar nn.Parameter(torch.tensor(X)); the selector has `write_gate`.
    #    Reshape them in-place to 1-D shape (1,) — all uses
    #    (`torch.sigmoid(p) * ...`, `torch.tanh(p) * x`, `.float().item()`)
    #    broadcast / work identically with a 1-element 1-D tensor.
    def _reshape_scalar(obj, pname):
        _p = getattr(obj, pname, None)
        if _p is not None and isinstance(_p, torch.nn.Parameter) and _p.dim() == 0:
            _new = torch.nn.Parameter(_p.detach().reshape(1).clone())
            _new.requires_grad_(_p.requires_grad)
            setattr(obj, pname, _new)

    for layer in mem_layers:
        for _pname in ("slot_output_gate", "gate_param"):
            _reshape_scalar(layer, _pname)
        # selector.write_gate is a 0-dim scalar (legacy single-gate path); even
        # when use_dual_gate=True it is still a registered Parameter and FSDP
        # walks it, so reshape it too.
        _sel = getattr(layer, "selector", None)
        if _sel is not None:
            _reshape_scalar(_sel, "write_gate")

    # 1) Wrap each MemorySpaceLayer in-place inside root.layers.
    #
    # Activation-checkpointing strategy with FSDP (same as babilong):
    #   - The trainable mem_space-only params are inside an FSDP unit; the
    #     frozen LlamaDecoderLayer at self.wrapped_layer has NO FSDP wrap.
    #   - MemorySpaceLayer._maybe_ckpt_wrapped_layer applies a manual
    #     torch.utils.checkpoint(use_reentrant=False) around ONLY the frozen
    #     wrapped_layer. Since the frozen layer is not FSDP-managed, manual ckpt
    #     is safe — no reshard_after_forward to fight with.
    #   - We do NOT use FSDP-native checkpoint_wrapper on the outer
    #     MemorySpaceLayer. Initial trials caused state-machine errors during
    #     the chunked / TBPTT multi-backward step
    #     ("expected to be in states [IDLE] but current state is
    #     FORWARD_BACKWARD"). Therefore use_checkpoint_wrapper=True is
    #     interpreted as "leave _inside_fsdp_unit UNSET" so the manual ckpt
    #     path stays enabled.
    n_wrapped = 0
    for i, layer in enumerate(layers_list):
        if layer not in mem_layers:
            continue
        if not use_checkpoint_wrapper:
            # Tell MemorySpaceLayer to skip the manual ckpt (no-op anyway since
            # gradient_checkpointing flag is False).
            layer._inside_fsdp_unit = True
        # else: leave _inside_fsdp_unit unset → manual torch.utils.checkpoint
        # is used inside MemorySpaceLayer.forward around the frozen
        # wrapped_layer call only.
        wrapped = FSDP(layer, **common_fsdp_kwargs)
        layers_list[i] = wrapped
        n_wrapped += 1

    # 2) Wrap shared L3 pool / L2 compressor / recon decoder (trainable params).
    l3_pool = getattr(model, "_l3_pool", None)
    if l3_pool is not None:
        wrapped_l3 = FSDP(l3_pool, **common_fsdp_kwargs)
        setattr(root, "l3_pool", wrapped_l3)
        model._l3_pool = wrapped_l3

    l2_comp = getattr(model, "_l2_compressor", None)
    if l2_comp is not None:
        wrapped_l2 = FSDP(l2_comp, **common_fsdp_kwargs)
        setattr(root, "l2_compressor", wrapped_l2)
        model._l2_compressor = wrapped_l2

    recon_decoder = getattr(model, "_recon_decoder", None)
    if recon_decoder is not None:
        wrapped_recon = FSDP(recon_decoder, **common_fsdp_kwargs)
        setattr(root, "recon_decoder", wrapped_recon)
        model._recon_decoder = wrapped_recon

    # ICAE token-recon head (2026-06-07): wrap like recon_decoder so its params
    # are sharded/trained/saved consistently. The training step accesses it via
    # model._l3_token_recon_head, so keep that pointer in sync with the wrap.
    l3_token_recon_head = getattr(model, "_l3_token_recon_head", None)
    if l3_token_recon_head is not None:
        wrapped_trh = FSDP(l3_token_recon_head, **common_fsdp_kwargs)
        setattr(root, "l3_token_recon_head", wrapped_trh)
        model._l3_token_recon_head = wrapped_trh

    # 3b) Full fine-tune (2026-06-18): wrap the root-level backbone params
    #     (embed_tokens, final norm, lm_head) so they are sharded + grad-synced.
    #     Skipped when frozen (they stay replicated, read-only, no sync needed).
    if unfreeze_backbone:
        for _root_attr in ("embed_tokens", "norm"):
            _mod = getattr(root, _root_attr, None)
            if _mod is not None and not isinstance(_mod, FSDP):
                setattr(root, _root_attr, FSDP(_mod, **common_fsdp_kwargs))
        # lm_head is on the OUTER model (LlamaForCausalLM), not root (LlamaModel).
        _lm_head = getattr(model, "lm_head", None)
        if _lm_head is not None and not isinstance(_lm_head, FSDP):
            model.lm_head = FSDP(_lm_head, **common_fsdp_kwargs)
        logger.info("FSDP full-FT: wrapped root embed_tokens/norm/lm_head.")

    # 3) NO top-level FSDP wrap (babilong lesson): with use_orig_params=True a
    #    top-level wrap tracks the FROZEN embedding/lm_head in its flat-param
    #    table and FSDP's _writeback_orig_params raises "Cannot writeback when
    #    the parameter shape changes" once inner FSDP units replace sub-module
    #    Parameters. Skipping the top-level wrap avoids this; per-layer FSDP
    #    units still expose state_dict_type() correctly via the un-wrapped root.
    #
    #    Tag the model so ``_save_adapter`` uses the FSDP gather path even
    #    though ``isinstance(model, FSDP)`` is now False.
    setattr(model, "_uses_partial_fsdp", True)

    logger.info(
        "FSDP wrap complete: %d MemorySpaceLayer units + (l3_pool=%s, l2=%s, "
        "recon=%s); frozen backbone replicated.",
        n_wrapped,
        "yes" if l3_pool is not None else "no",
        "yes" if l2_comp is not None else "no",
        "yes" if recon_decoder is not None else "no",
    )
    return model


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
        # FIX v13 (2026-06-02): inject_gate (layer.py:400, the per-token fusion
        # gate g=sigmoid(inject_gate(hidden))) was missing here, so _freeze_backbone
        # set its requires_grad=False and never re-enabled it → g was frozen at
        # its init (≈0.46) forever, blocking the model from learning when to
        # inject memory. Collect it like slot_to_hidden.
        inj = getattr(wrapper, "inject_gate", None)
        if inj is not None:
            for p in inj.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        # P2 (2026-06-03): decoupled cross-attn READ module params.
        dr = getattr(wrapper, "decoupled_read", None)
        if dr is not None:
            for p in dr.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        # P8 fix (2026-06-05): the dedicated cross-attn READ module (q/k/v/out_proj,
        # per-head gate, and the null/sink slot from 1f46b4d) was never collected,
        # so _freeze_backbone left the whole read path + zero-init null_value frozen
        # -> the sink could never learn and the random-init read injected pure noise.
        mx = getattr(wrapper, "memory_xattn", None)
        if mx is not None:
            for p in mx.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        # EXP-W2 (2026-06-11): dense soft-write content module (slots-as-query
        # write-attention). Only present when soft_write_weight>0; collect its
        # q/k/v/out_proj params so _freeze_backbone re-enables grad and they
        # enter the optimizer (else the content head stays frozen at init).
        swc = getattr(wrapper, "soft_write_content_mod", None)
        if swc is not None:
            for p in swc.parameters():
                if id(p) not in seen:
                    params.append(p); seen.add(id(p))
        # Writeback-gate params (2026-06-04). Previously the dual_gate
        # gate_proj_new/gate_proj_mem/gate_bias were NOT collected here, so with
        # --use_dual_gate the gate projections never entered the optimizer (frozen
        # at xavier init). Collect them now plus the lowrank / diag mode params.
        # Robust approach: collect any of these attrs that exist and are not None.
        #   dual_gate:    gate_proj_new, gate_proj_mem, gate_bias
        #   lowrank_gate: lr_V_new, lr_V_mem, lr_U, lr_gate_bias
        #   diag_gate:    diag_a_in, diag_c_in, diag_a_f, diag_c_f, diag_b_in, diag_b_f
        #   scalar_beta:  none (uses gate_param, already collected above)
        _gate_module_attrs = ("gate_proj_new", "gate_proj_mem",
                              "lr_V_new", "lr_V_mem", "lr_U")
        for _attr in _gate_module_attrs:
            _mod = getattr(wrapper, _attr, None)
            if _mod is not None:
                for p in _mod.parameters():
                    if id(p) not in seen:
                        params.append(p); seen.add(id(p))
        _gate_param_attrs = ("gate_bias", "lr_gate_bias",
                            "diag_a_in", "diag_c_in", "diag_a_f", "diag_c_f",
                            "diag_b_in", "diag_b_f")
        for _attr in _gate_param_attrs:
            _par = getattr(wrapper, _attr, None)
            if _par is not None and id(_par) not in seen:
                params.append(_par); seen.add(id(_par))
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
    # MemoryReconDecoder params (P1/v12)
    recon_decoder = getattr(root, "_recon_decoder", None)
    if recon_decoder is not None:
        for p in recon_decoder.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    # L3TokenReconHead params (ICAE token recon, 2026-06-07). Like recon_decoder,
    # this is a shared singleton registered on the root; collect its params so
    # _freeze_backbone re-enables their grad and they enter the optimizer (else
    # the head would stay frozen at init and the CE loss would never improve).
    l3_token_recon_head = getattr(root, "_l3_token_recon_head", None)
    if l3_token_recon_head is not None:
        for p in l3_token_recon_head.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    # Raw-KV readout gist scorer (Method A, 2026-06-19). Shared singleton on the
    # root (peer to l3_pool); collect its query/key_proj params so the trainable
    # gist-key soft attention enters the optimizer (else it stays frozen at init
    # and the emergent selection never learns).
    gist_readout = getattr(root, "_gist_readout", None)
    if gist_readout is not None:
        for p in gist_readout.parameters():
            if id(p) not in seen:
                params.append(p); seen.add(id(p))
    return params


def _freeze_backbone(
    model: torch.nn.Module,
    unfreeze_backbone: bool = False,
    unfreeze_layers_from: int = -1,
) -> None:
    """Set requires_grad for training.

    Default (unfreeze_backbone=False): freeze everything, then re-enable ONLY
    the mem_space adapter params — byte-identical to the original behaviour.

    Full fine-tune (unfreeze_backbone=True): make the ENTIRE model trainable
    (Llama backbone attn+FFN+embed+lm_head + the mem_space adapter). This is the
    Landmark-faithful path: training on the injection mechanism lets the model
    recalibrate to the injected-KV distribution the frozen reader could not
    consume. The mem_space params are explicitly re-enabled too (no-op since they
    are already on, but keeps the contract that adapter always trains).

    Partial unfreeze (unfreeze_backbone=True, unfreeze_layers_from=N>=0, v2):
    only decoder layers with index >= N are trainable; layers 0..N-1 and
    embed_tokens stay FROZEN, protecting the base LM's lower-level competence
    (v1 full-unfreeze on short data damaged niah OFF 22→11). The final norm +
    lm_head + the mem_space adapter stay trainable so the readout can adapt.
    The MemorySpaceLayer wraps each decoder layer; we identify a layer's index
    via its `_layer_idx` and freeze/unfreeze the wrapped Llama params accordingly
    while ALWAYS keeping that layer's mem_space adapter params trainable.
    """
    if unfreeze_backbone and unfreeze_layers_from is not None and unfreeze_layers_from >= 0:
        # Start fully frozen, then re-enable: (a) adapter params on every layer,
        # (b) the WRAPPED Llama decoder params only for layers idx >= N,
        # (c) final norm + lm_head.
        for p in model.parameters():
            p.requires_grad = False
        for p in _mem_space_params(model):
            p.requires_grad = True
        root = getattr(model, "module", model)
        mem_layers = getattr(root, "_mem_space_layers", None) or []
        n_unfrozen_layers = 0
        for w in mem_layers:
            _idx = getattr(w, "_layer_idx", None)
            _wrapped = getattr(w, "wrapped_layer", None)
            if _idx is not None and _wrapped is not None and _idx >= unfreeze_layers_from:
                for p in _wrapped.parameters():
                    p.requires_grad = True
                n_unfrozen_layers += 1
        # Final norm + lm_head (the readout head) stay trainable.
        _inner = getattr(model, "model", root)
        _norm = getattr(_inner, "norm", None)
        if _norm is not None:
            for p in _norm.parameters():
                p.requires_grad = True
        _lm_head = getattr(model, "lm_head", None)
        if _lm_head is not None:
            for p in _lm_head.parameters():
                p.requires_grad = True
        logger.info(
            "PARTIAL unfreeze: decoder layers idx>=%d trainable (%d layers) + "
            "adapter + final-norm + lm_head; layers 0..%d + embed_tokens FROZEN.",
            unfreeze_layers_from, n_unfrozen_layers, unfreeze_layers_from - 1,
        )
        return
    if unfreeze_backbone:
        for p in model.parameters():
            p.requires_grad = True
        return
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


def _set_banks_frozen(model: torch.nn.Module, frozen: bool) -> None:
    """Toggle the memory bank(s) ``frozen`` flag (mirrors run_babilong_mem_space
    _freeze_banks/_unfreeze_banks). When frozen, MemoryBank.write() is a no-op
    (early return), so a forward READS the slots without WRITING them.

    Used by the cross-chunk SWA TRAIN window (``--swa_train_chunks``): the last
    W context chunks are streamed into the bank by the normal TBPTT loop, then
    the target forward concatenates those W chunks + target into one window so
    the target tokens can directly attend their raw KV. We freeze the bank
    around that window forward so the re-presented W chunks do NOT get written a
    second time (they are already in the bank) — frozen blocks writes only, the
    read path is unaffected, so the SWA window still reads the accumulated bank.
    """
    root = getattr(model, "module", model)
    shared_bank = getattr(root, "_mem_space_shared_bank", None)
    if shared_bank is not None:
        shared_bank.frozen = frozen
        return
    for w in getattr(root, "_mem_space_layers", []) or []:
        b = getattr(w, "memory_bank", None)
        if b is not None:
            b.frozen = frozen


def _collect_top1_sim(model: torch.nn.Module) -> float:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return 0.0
    vals = [getattr(w, "_last_top1_sim", 0.0) for w in mem_layers]
    return vals[0] if vals else 0.0


def _collect_mem_diag(model: torch.nn.Module) -> Dict[str, float]:
    """Layer-0 memory routing diagnostics for wandb (mirrors QUERY_DIAG).

    Returns key_max_cos (slot-key separability; ->1 = collapsed), usage_cov
    (#slots used / N across the diag window), usage_ent (normalized usage
    entropy in [0,1]; 1 = uniform load), and slot_attn_entropy (slot_query
    mode; high = slots smear attention over tokens). Empty dict if no layers.
    """
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    if not mem_layers:
        return {}
    L0 = mem_layers[0]
    sel0 = getattr(L0, "selector", None)
    swc0 = getattr(L0, "soft_write_content_mod", None)
    return {
        "memory/key_max_cos": getattr(L0, "_last_key_max_cos", 0.0),
        "memory/usage_cov": getattr(L0, "_last_usage_cov", 0.0),
        "memory/usage_ent": getattr(L0, "_last_usage_ent", 0.0),
        "memory/usage_var": getattr(L0, "_last_usage_var", 0.0),
        "memory/slot_attn_entropy": getattr(sel0, "_last_slot_attn_entropy", 0.0),
        # EXP-D3 (2026-06-11): mean pairwise cosine of slot CONTENT — the direct
        # homogenisation signal for broad-write methods (W1/W2/R1). ->1 == bank
        # collapsed onto one direction (the failure mode W2's dense write risks).
        "memory/slot_content_cos": getattr(L0, "_last_slot_content_cos", 0.0),
        # EXP-W2 (2026-06-11): slot-query write-attention entropy of the dense
        # soft-write content module (high == slots smear over all tokens →
        # content converges → homogenisation). 0.0 when soft-write disabled.
        "memory/soft_write_attn_entropy": (
            getattr(swc0, "_last_attn_entropy", 0.0) if swc0 is not None else 0.0
        ),
        # EXP-R1 / EXP-D2 dead-slot recycling telemetry (no-op / 0.0 when disabled).
        "memory/dead_slot_frac": getattr(L0, "_last_dead_slot_frac", 0.0),
        "memory/max_slot_select_count": getattr(L0, "_last_max_slot_select_count", 0.0),
        "memory/recycle_resets": getattr(L0, "_last_recycle_resets", 0.0),
        "memory/n_recycled": getattr(L0, "_last_n_recycled", 0.0),
    }


def _collect_xattn_diag(model: torch.nn.Module) -> Dict[str, float]:
    """P8 memory cross-attention READ diagnostics for wandb (mirrors the
    selector.py:1443 cache block of MemoryCrossAttentionRead).

    Averages the per-layer cached scalars across all decoder layers that hold a
    ``memory_xattn`` module:
      - sink_mass: mean softmax mass on the learnable null/sink column. Rising
        sink_mass on cold/irrelevant context == the model learned to "attend to
        nothing" (the escape valve is working).
      - gate_mean: mean per-head content gate (read magnitude).
      - attn_entropy: mean entropy of the slot+sink softmax (high = smeared).
    Returns an empty dict when no layer exposes memory_xattn (e.g. runs without
    --use_memory_xattn), so non-P8 runs are unaffected and never error.
    """
    mem_layers = _mem_layers(model)
    if not mem_layers:
        return {}
    sink_vals, gate_vals, ent_vals = [], [], []
    for w in mem_layers:
        mx = getattr(w, "memory_xattn", None)
        if mx is None:
            continue
        sink_vals.append(float(getattr(mx, "_last_sink_mass", 0.0)))
        gate_vals.append(float(getattr(mx, "_last_gate_mean", 0.0)))
        ent_vals.append(float(getattr(mx, "_last_attn_entropy", 0.0)))
    if not sink_vals:
        return {}
    return {
        "memory/xattn_sink_mass": sum(sink_vals) / len(sink_vals),
        "memory/xattn_gate_mean": sum(gate_vals) / len(gate_vals),
        "memory/xattn_attn_entropy": sum(ent_vals) / len(ent_vals),
    }


def _install_score_hook(mem_layers):
    """Install a forward hook on the layer-0 selector to capture the
    grad-bearing softmax routing scores [B, N] from the next model forward.

    The MemorySpaceLayer detaches ``last_scores`` after the forward, so a hook
    on the selector module is the only way to obtain a gradient-bearing copy
    (matches scripts/toy_memory_bootstrap.py:398-412). Returns
    ``(captured_dict, handle)`` where ``captured_dict["scores"]`` is populated
    once the forward runs, and ``handle`` must be ``.remove()``-d afterwards.
    """
    captured: Dict[str, torch.Tensor] = {}
    handle = None
    if not mem_layers:
        return captured, handle
    sel0 = getattr(mem_layers[0], "selector", None)
    if sel0 is None:
        return captured, handle

    def _score_hook(_mod, _inp, _out):
        # selector returns (idx, scores, ste_weights); scores is differentiable.
        if isinstance(_out, (tuple, list)) and len(_out) >= 2:
            captured["scores"] = _out[1]

    handle = sel0.register_forward_hook(_score_hook)
    return captured, handle


def _compute_route_aux(scores2, idx1, device: torch.device):
    """Cross-entropy routing supervision: push the grad-bearing chunk scores
    ``scores2`` [B, N] to place probability mass on the slot indices ``idx1``
    [B, k] that a prior chunk wrote into.

    ``route_aux = -mean_b mean_{j in idx1} log scores2[b, j]`` with a 1e-9 clamp
    for numerical safety (matches toy E2). Returns ``None`` when inputs are
    unusable (missing scores/idx, shape mismatch, or non-finite)."""
    if scores2 is None or idx1 is None:
        return None
    if scores2.dim() != 2 or scores2.shape[0] != idx1.shape[0]:
        return None
    sel_p = scores2.gather(1, idx1.to(scores2.device))      # [B, k]
    ra = -(sel_p.clamp(min=1e-9).log().mean())
    if not torch.isfinite(ra):
        return None
    return ra


def _compute_l3_token_recon(model: torch.nn.Module, chunk_input: torch.Tensor):
    """ICAE-style token-level reconstruction CE loss (2026-06-07).

    Reconstructs the CURRENT chunk's DISCRETE input tokens from a fresh,
    grad-bearing L3 summary of that chunk, decoded through the frozen lm_head.

    Pipeline (gradient path annotated):
        cur_h = grad-bearing top-layer hidden of THIS chunk (stashed by the L3
                post-forward hook; None under no_grad context passes)
        summary = l3_pool(cur_h, prev_summary=None)      # [B, K, d]  GRAD→l3_pool
        dec_h   = head(summary, seq_len=T)               # [B, T, d]  GRAD→head,summary
        logits  = frozen_lm_head(dec_h)                  # [B, T, V]  (no grad to lm_head)
        loss    = CE(logits[:, :T'], chunk_input[:, :T'])

    The target ``chunk_input`` is NOT detached from the loss in the usual sense:
    it is the integer label tensor, and the CE pulls the *predicted distribution*
    toward it. Gradient flows back through ``summary`` into the L3 pool (the
    whole point — forces semantic compression), and through ``cur_h`` into the
    memory write/read path (also desirable; backbone params are frozen so they
    do not move).

    Returns ``None`` when the head is absent, no grad-bearing current-chunk
    hidden is available (e.g. context pass under no_grad), or the result is
    non-finite. The caller clears the stashed hidden after consumption.
    """
    root = getattr(model, "module", model)
    head = getattr(root, "_l3_token_recon_head", None)
    pool = getattr(root, "_l3_pool", None)
    if head is None or pool is None:
        return None
    cur_h = getattr(root, "_l3_token_recon_cur_h", None)
    if cur_h is None or not cur_h.requires_grad:
        return None

    # chunk_input: [B, T] token ids for THIS chunk. Align T with cur_h's seq len
    # (they should match — both are the chunk's content length).
    B, T_h, _ = cur_h.shape
    if chunk_input.dim() == 1:
        chunk_input = chunk_input.unsqueeze(0)
    T = min(T_h, chunk_input.shape[1])
    if T < 1:
        return None
    cur_h = cur_h[:, :T, :]
    tgt = chunk_input[:, :T].to(cur_h.device)

    # Fresh, INDEPENDENT summary of this chunk (prev_summary=None → use the
    # pool's learnable queries; we want "summarize THIS chunk", not the
    # recurrent accumulation used for routing). Grad flows into l3_pool.
    summary = pool(cur_h, prev_summary=None)                 # [B, K, d]
    dec_h = head(summary, seq_len=T)                         # [B, T, d]

    # Frozen lm_head → vocab logits. lm_head requires_grad is False, so no param
    # update; we only use it to project decoder hidden into vocab space.
    lm_head = root.get_output_embeddings()
    logits = lm_head(dec_h)                                  # [B, T, V]

    loss = torch.nn.functional.cross_entropy(
        logits.float().reshape(-1, logits.shape[-1]),
        tgt.reshape(-1),
    )
    if not torch.isfinite(loss):
        return None
    return loss


def _clear_l3_token_recon_cur_h(model: torch.nn.Module) -> None:
    """Clear the stashed grad-bearing current-chunk hidden after consumption so
    it is not accidentally reused on a later chunk / step."""
    root = getattr(model, "module", model)
    if hasattr(root, "_l3_token_recon_cur_h"):
        root._l3_token_recon_cur_h = None


def _mem_layers(model: torch.nn.Module):
    """Return the list of MemorySpaceLayer wrappers (or None)."""
    root = getattr(model, "module", model)
    return getattr(root, "_mem_space_layers", None)


def _collect_aux_loss(model: torch.nn.Module, device: torch.device) -> torch.Tensor:
    root = getattr(model, "module", model)
    mem_layers = getattr(root, "_mem_space_layers", None)
    total = torch.zeros((), device=device)
    if not mem_layers:
        return total
    for w in mem_layers:
        for key in ("load_balance", "entropy", "key_repulsion", "weight_ortho", "l3_diversity", "q_multi_diversity", "recon"):
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
    p.add_argument("--contiguous_chunks", action="store_true", default=False,
                   help="Treat the Dolmino Arrow rows as one continuous "
                        "STREAM-ORDERED token stream and re-slice it at "
                        "--chunk_size granularity (e.g. 256), so consecutive "
                        "context chunks + target fall inside the same document "
                        "= genuine intra-document cross-chunk dependency. "
                        "Default False = legacy random-row shuffle (context and "
                        "target are unrelated docs -> memory routing collapses "
                        "to uniform).")
    p.add_argument("--doc_reset", action="store_true", default=False,
                   help="Only meaningful with --contiguous_chunks. Attach a "
                        "per-chunk reset_flags list to each sample (True where a "
                        "chunk begins a new document, i.e. the preceding stream "
                        "token is EOS). The training loop resets the memory bank "
                        "BEFORE forwarding such a chunk so the BPTT graph never "
                        "spans a document boundary.")
    p.add_argument("--per_doc_data", action="store_true", default=False,
                   help="Use the per-document Dolmino dataset (each Arrow row is "
                        "one complete document, produced by "
                        "scripts/reprocess_dolmino_per_doc.py) and slice each "
                        "document into intra-document (n_ctx context + 1 target) "
                        "chunk groups. context and target are then consecutive "
                        "chunks of the SAME document => genuine cross-chunk "
                        "dependency. When set, point --dolmino_path at "
                        "MemLong/data/processed/dolmino_per_doc/train. Takes "
                        "precedence over --contiguous_chunks.")
    p.add_argument("--min_doc_len", type=int, default=512,
                   help="Minimum document length used when the per-document "
                        "dataset was built (recorded for provenance; the actual "
                        "filtering happens in reprocess_dolmino_per_doc.py).")

    # Curriculum
    p.add_argument("--curriculum", type=str,
                   default="0:1,10000:2,15000:4,25000:8,40000:16",
                   help="Curriculum schedule: 'step:n_ctx,step:n_ctx,...'")
    p.add_argument("--t2_curriculum", type=str, default="",
                   help="Independent curriculum for the T2 recall needle->query "
                        "distance: 'step:n_ctx,...'. Empty = T2 keeps the fixed "
                        "n_ctx derived from --t2_gap_tokens. Use this to grow ONLY "
                        "the T2 distance (pg19 long background supports it) while "
                        "dolmino stays at a small fixed n_ctx (dolmino per-doc data "
                        "is capped at 4096 tokens, so (n_ctx+1)*chunk_size>4096 "
                        "starves the dolmino loader and hangs DDP).")

    # Self-study distillation (v21, 2026-06-15). All default OFF/empty → when no
    # --distill_* flag is given the dolmino step is BYTE-IDENTICAL to before.
    p.add_argument("--distill_logits", action="store_true", default=False,
                   help="Enable scheme-A logits distillation (bidirectional KL on "
                        "the teacher top-64 support) for dolmino steps. Requires "
                        "--distill_cache_dir. Default off.")
    p.add_argument("--distill_hidden", action="store_true", default=False,
                   help="Enable scheme-B hidden matching (1-cos on selected layers) "
                        "for dolmino steps. Requires --distill_cache_dir. Off.")
    p.add_argument("--distill_lambda", type=float, default=0.6,
                   help="Scheme-A bidirectional KL weight: "
                        "lambda*KL(p||q)+(1-lambda)*KL(q||p). 0.6 = KV-Distill.")
    p.add_argument("--distill_layers", type=str, default="12,20,28",
                   help="Comma-separated DECODER-layer indices for scheme-B "
                        "hidden matching. Must match the cache's meta_layers.")
    p.add_argument("--distill_cache_dir", type=str, default="",
                   help="Dir of teacher .npz cache from build_distill_cache.py. "
                        "Empty = distillation disabled regardless of flags.")
    p.add_argument("--distill_weight", type=float, default=1.0,
                   help="Overall multiplier on the distill term: "
                        "total = lm + aux + distill_weight*(L_A + beta*L_B).")
    p.add_argument("--distill_hidden_beta", type=float, default=1.0,
                   help="beta: relative weight of scheme-B (hidden) vs scheme-A.")

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
    p.add_argument("--weight_decay", type=float, default=0.0,
                   help="AdamW weight decay. 0.0 for adapter-only training "
                        "(original); Landmark full FT uses 0.1.")
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--gradient_accumulation_steps", type=int, default=4)
    p.add_argument("--start_step", type=int, default=0,
                   help="Resume from this step count (for curriculum and LR schedule).")
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--proj_grad_clip", type=float, default=0.1)
    p.add_argument("--loss_spike_skip", action="store_true",
                   help="Skip optimizer.step() when the (DDP-averaged) lm_loss "
                        "for a step exceeds running_mean + sigma*running_std. "
                        "Default off (preserves existing behavior).")
    p.add_argument("--loss_spike_sigma", type=float, default=3.0,
                   help="Sigma threshold for --loss_spike_skip.")
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
    p.add_argument("--last_chunk_loss_only", action="store_true",
                   help="HARDER objective (2026-06-13): stream context chunks "
                        "no_grad into memory, compute LM loss ONLY on the target "
                        "chunk (via dolmino_train_step). Forces the target's "
                        "prediction to rely on the memory bank for prior context "
                        "instead of local causal attn. Default off = tbptt "
                        "(every chunk gets gradient, easy, memory barely needed).")
    p.add_argument("--swa_train_chunks", type=int, default=0,
                   help="Cross-chunk SWA TRAIN window W (D2b, 2026-06-09). "
                        "Default 0 = current behavior, bit-identical. W>0: the "
                        "TARGET chunk's grad-bearing forward is widened from a "
                        "single chunk to the concatenation of the last W context "
                        "chunks + the target, so target tokens directly attend "
                        "those W chunks' raw KV (sliding window) IN ADDITION to "
                        "the memory bank. Prefix (W*chunk_size) labels are masked "
                        "(-100) so LM loss is computed ONLY on the target tokens "
                        "(no double-counting of context loss; context chunks keep "
                        "their own per-chunk loss exactly as in the W0 TBPTT "
                        "path). The bank is frozen around the window forward so "
                        "the re-presented W chunks (already streamed in) are not "
                        "written twice. Train-side analogue of the eval-side "
                        "--swa_eval_chunks (D2a). Falls back to W0 when there are "
                        "fewer than W context chunks, or under doc_reset "
                        "(reset_flags) to avoid spanning a document boundary.")

    # T2 chunked associative-recall mix (2026-06-14, readout-pivot)
    p.add_argument("--t2_recall_mix_fraction", type=float, default=0.0,
                   help="Probability of sampling a T2 chunked associative-recall "
                        "step (NIAHChunkedDataset) vs a Dolmino step. Default 0 "
                        "= no T2 (bit-identical to prior behaviour). T2 samples "
                        "go through dolmino_train_step with an answer_mask so LM "
                        "loss falls ONLY on the answer digits — forcing precise "
                        "memory readout. Mutually composed with "
                        "--babilong_mix_fraction (BABILong checked first).")
    p.add_argument("--t2_num_keys", type=int, default=1,
                   help="Number of (name->code) mappings written into the T2 "
                        "context. 1 = single needle; >1 adds distractor keys "
                        "(T2b multi-key ablation).")
    p.add_argument("--t2_gap_tokens", type=int, default=3584,
                   help="T2 needle->query token distance, held CONSTANT across "
                        "chunk_sizes for fair comparison. n_ctx is derived as "
                        "round(t2_gap_tokens / chunk_size) (default 3584 = 7*512).")
    p.add_argument("--t2_background_data", type=str,
                   default="data/pg19_chunks_llama3.npy",
                   help="Pre-tokenised [N, L] natural-text npy used to fill T2 "
                        "non-needle context chunks (Llama-3 tokens).")
    p.add_argument("--t2_background_skip", type=int, default=0,
                   help="Skip the first N background chunks (train/eval split).")

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
    p.add_argument("--slot_dim", type=int, default=None,
                   help="L1 slot vector dimensionality. None = backbone hidden "
                        "size (d_model). Set e.g. 16384 for the large-slot "
                        "writeback-mode experiment.")
    p.add_argument("--writeback_mode", type=str, default="dual_gate",
                   choices=["dual_gate", "lowrank_gate", "diag_gate", "scalar_beta"],
                   help="Writeback gate parameterisation. dual_gate=full LM2 "
                        "gate (4*slot_dim^2/layer); lowrank_gate=low-rank "
                        "(4*slot_dim*r/layer); diag_gate=per-feature diagonal "
                        "(6*slot_dim/layer); scalar_beta=legacy single-β EMA.")
    p.add_argument("--lowrank_gate_rank", type=int, default=256,
                   help="Rank r for --writeback_mode lowrank_gate.")
    p.add_argument("--writeback_gate_max", type=float, default=0.3)
    p.add_argument("--writeback_warmup_steps", type=int, default=0)
    p.add_argument("--load_balance_weight", type=float, default=0.01)
    p.add_argument("--entropy_aux_weight", type=float, default=0.001)
    # P7 loss-free balancing (arXiv 2408.15664): online per-slot routing-logit bias
    # that balances slot usage WITHOUT injecting a uniform-pushing gradient. When
    # enabled, set --load_balance_weight 0.0 (the two must not both be nonzero).
    p.add_argument("--use_loss_free_balance", action="store_true")
    p.add_argument("--loss_free_update_rate", type=float, default=0.001)
    # P10 (arXiv ST-Gumbel-topk): inject Gumbel noise into the selection logits
    # before top-k (training only) for exploration / reduced key over-smearing.
    # Default off → selector byte-identical to pre-P10.
    p.add_argument("--use_st_gumbel_topk", action="store_true")
    p.add_argument("--st_gumbel_temperature", type=float, default=1.0)
    # P11 (2026-06-06): delta-rule writeback + normalized readout. Both default
    # off → byte-identical to pre-P11. delta-rule switches the gated writeback
    # to a residual (slot + g·(new−slot)) update; normalize_readout rescales the
    # readout memory vector to the local hidden-state magnitude before injection.
    p.add_argument("--use_delta_rule_writeback", action="store_true")
    p.add_argument("--normalize_readout", action="store_true")
    p.add_argument("--readout_norm_scale", type=float, default=1.0)
    # Dead-slot-binding fix: pure learnable per-slot routing key (independent of
    # slot content). Default off → content-based routing (byte-identical).
    p.add_argument("--independent_slot_key", action="store_true")
    # DeltaNet-style erase-then-write associative update on the gated writeback.
    # Default off → existing EMA / delta-rule write (byte-identical).
    p.add_argument("--delta_erase_write", action="store_true")
    # L2 token-compressed KV memory (Phase 11). Default off → byte-identical to
    # the L1(+L3) stack. When --use_l2 is set, a shared L2Compressor is
    # instantiated in patch.py (gated on config.use_l2) producing token-
    # compressed KV latents for the next chunk. Defaults mirror MemorySpaceConfig.
    p.add_argument("--use_l2", action="store_true")
    p.add_argument("--l2_compress_ratio", type=int, default=16)
    p.add_argument("--l2_d_c", type=int, default=512)
    p.add_argument("--l2_d_h_rope", type=int, default=64)
    p.add_argument("--l2_init_scale", type=float, default=0.001)
    # EXP-R1 (2026-06-11): two-stage dead-slot recycling. Default off
    # (interval=0) → byte-identical to P11. See config.py for the mechanism.
    p.add_argument("--dead_slot_reset_interval", type=int, default=0,
                   help="EXP-R1: every R chunks (per-sample), reset slots that "
                        "were never top-k-selected in the last R chunks to "
                        "diverse strided current-chunk content, then force-write "
                        "them for the next --dead_slot_grace_chunks chunks. "
                        "0 = disabled (byte-identical to P11).")
    p.add_argument("--dead_slot_reset_mode", type=str, default="strided_current",
                   choices=["strided_current", "zero"],
                   help="EXP-R1: content source for reset dead slots. "
                        "strided_current = diverse strided current-chunk tokens "
                        "(maximise distance from live slots); zero = zero them.")
    p.add_argument("--dead_slot_grace_chunks", type=int, default=1,
                   help="EXP-R1: # chunks after a reset to force the recycled "
                        "slots into the WRITE set (write-only; read unchanged).")
    p.add_argument("--dead_slot_criterion", type=str, default="window",
                   choices=["window", "cumulative"],
                   help="EXP-R1c: dead-slot judge at a reset boundary. "
                        "window (default, R1, byte-identical) = zero selections "
                        "in the last --dead_slot_reset_interval chunks; "
                        "cumulative (R1c) = zero selections over the WHOLE "
                        "sample so far (spares long-range memory slots, recycles "
                        "strictly fewer).")
    # EXP-W2 (2026-06-11): dense all-slot soft delta-write. Default off
    # (weight=0.0) → byte-identical to P11. Orthogonal to EXP-R1; see config.py.
    p.add_argument("--soft_write_weight", type=float, default=0.0,
                   help="EXP-W2: λ for the dense all-slot soft delta-write "
                        "(slot_n += λ·g_n·(content_n − slot_n) over ALL N slots, "
                        "IN ADDITION to the top-k hard write). 0.0 = disabled "
                        "(byte-identical to P11). Recommended 0.02-0.05.")
    p.add_argument("--soft_write_content", type=str, default="slot_query",
                   choices=["slot_query"],
                   help="EXP-W2: per-slot content source for the dense write. "
                        "slot_query = slots-as-query cross-attention over the "
                        "chunk tokens (per-slot distinct, anti-homogenisation).")
    # v20 (2026-06-12): read-based slot lifecycle. Two INDEPENDENT arms, both
    # default OFF → byte-identical to P11. See versions/
    # v20_read_based_slot_lifecycle.md.
    p.add_argument("--slot_read_decay_rate", type=float, default=0.0,
                   help="v20 Arm A: R for soft read-decay. Every "
                        "--slot_read_decay_interval chunks, multiply each slot's "
                        "value by clamp(1-R·(1-recent_read_frac), min_keep, 1). "
                        "0.0 = disabled (byte-identical to P11).")
    p.add_argument("--slot_read_decay_interval", type=int, default=8,
                   help="v20 Arm A: # chunks between read-decay applications.")
    p.add_argument("--slot_read_decay_min_keep", type=float, default=0.5,
                   help="v20 Arm A: floor on the per-interval decay factor so an "
                        "unread slot cannot be zeroed in one step (password "
                        "protection).")
    p.add_argument("--slot_evict_mode", type=str, default="off",
                   choices=["off", "readmass"],
                   help="v20 Arm B: slot eviction policy. off (default) = "
                        "existing dead_slot_criterion path unchanged; readmass = "
                        "evict the coldest (lowest cumulative read-mass) slots "
                        "that have stayed cold past the protection window.")
    p.add_argument("--slot_evict_max_frac", type=float, default=0.1,
                   help="v20 Arm B: max fraction of slots evicted per boundary "
                        "(ceil(frac·N)).")
    p.add_argument("--slot_evict_floor", type=float, default=0.0,
                   help="v20 Arm B: recent read-mass threshold at/below which a "
                        "slot counts as cold (eligible for the cold-streak).")
    p.add_argument("--slot_evict_protect_chunks", type=int, default=2,
                   help="v20 Arm B: # consecutive cold boundaries a slot must "
                        "stay cold before it becomes evictable (protection "
                        "window for rarely-read but critical slots).")
    p.add_argument("--selector_temperature", type=float, default=20.0)
    p.add_argument("--key_repulsion_weight", type=float, default=0.05)
    p.add_argument("--key_repulsion_threshold", type=float, default=0.3)
    p.add_argument("--l3_diversity_weight", type=float, default=0.1)
    p.add_argument("--l3_diversity_threshold", type=float, default=0.5)
    p.add_argument("--l_recon_weight", type=float, default=0.0,
                   help="P1/v12 summary-reconstruction aux loss weight. >0 "
                        "enables the MemoryReconDecoder (requires use_l3_summary). "
                        "0 = disabled (default).")
    p.add_argument("--l3_recon_token_weight", type=float, default=0.0,
                   help="ICAE-style token-level reconstruction aux loss weight "
                        "(2026-06-07). >0 enables an L3TokenReconHead that "
                        "reconstructs the CURRENT chunk's DISCRETE input tokens "
                        "(CE through the frozen lm_head) from a fresh grad-bearing "
                        "L3 summary of that chunk; gradient flows back into the "
                        "L3 pool, forcing genuine semantic compression. Requires "
                        "use_l3_summary. 0 = disabled (default; back-compat).")
    p.add_argument("--peak_routing_weight", type=float, default=0.05)
    p.add_argument("--route_aux_weight", type=float, default=0.0,
                   help="E2 routing-supervision aux loss weight (2026-06-04). "
                        ">0 enables a cross-entropy that supervises the current "
                        "chunk's grad-bearing routing scores to put mass on the "
                        "slots the PREVIOUS chunk wrote into (cross-chunk "
                        "write->read routing supervision). Bootstraps content "
                        "addressing that pure LM loss cannot (toy E2). 0 = "
                        "disabled, training path identical to before. Toy used "
                        "1.0.")
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
    p.add_argument("--no_l3_summary", dest="use_l3_summary", action="store_false",
                   help="Disable L3 summary pool (l3_pool=None). For L3-isolation ablation.")
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
                   choices=["max_pool", "chunk_query", "multi_query", "slot_query"])
    p.add_argument("--multi_query_tau", type=float, default=1.0,
                   help="logsumexp temperature for multi_query routing aggregation")

    # P2 (2026-06-03): decoupled cross-attention READ path.
    p.add_argument("--use_decoupled_read", action="store_true", default=False,
                   help="P2: route the memory READ via a dedicated "
                        "CrossAttentionMemoryV2 (slots single softmax, out_proj "
                        "zero-init) and mask H->L1 prepend attention, bypassing "
                        "the injection-dilution root cause. False = legacy "
                        "prepend path (backward-compatible).")

    # P8 (2026-06-05): dedicated memory cross-attention READ with independent
    # softmax + per-head content-dependent gate, ACTIVE at init.
    p.add_argument("--use_memory_xattn", action="store_true", default=False,
                   help="P8: route the memory READ via a dedicated "
                        "MemoryCrossAttentionRead (slots get their OWN softmax) "
                        "and mask H->L1 prepend, fixing the ~0.2%% attn-mass "
                        "dilution. Unlike --use_decoupled_read (P2, zero-init "
                        "out_proj + tiny shared gate ≈ dead at start), the read "
                        "output is per-head content-gated and ACTIVE at init "
                        "(out_proj small-random, gate≈memory_xattn_gate_init), "
                        "so gradient flows through memory from step 0. "
                        "False = legacy prepend path (backward-compatible).")
    p.add_argument("--memory_xattn_gate_init", type=float, default=0.4,
                   help="P8: effective per-head gate contribution at init "
                        "(sigmoid space, 0.3-0.5 band). Default 0.4.")
    p.add_argument("--memory_xattn_disable_null_sink", action="store_true", default=False,
                   help="D6: disable the learnable null/sink slot inside "
                        "MemoryCrossAttentionRead (single-variable ablation). "
                        "The read softmax then has NO 'attend to nothing' escape "
                        "column. Default False = sink ON (P8/P11 baseline).")
    p.add_argument("--use_readout_mass_bias", action="store_true", default=False,
                   help="2026-06-15: add a per-slot token-mass bias "
                        "(readout_mass_coef·log1p(mass)) to the P8 read softmax "
                        "logits, so a slot that condensed MANY real tokens gets "
                        "proportionally more read weight (mass = cumulative real "
                        "tokens absorbed, tracked in MemoryBank.slot_token_mass). "
                        "Default False = no bias (softmax byte-identical to P8/P11).")
    p.add_argument("--readout_mass_coef", type=float, default=1.0,
                   help="Scale on the log1p(mass) readout bias (1.0 = exact "
                        "log1p; larger = stronger mass preference). Ignored when "
                        "--use_readout_mass_bias is off.")

    # Slot-Routed Evidence Memory (2026-06-17)
    p.add_argument("--use_slot_evidence", action="store_true", default=False,
                   help="Enable Slot-Routed Evidence Memory: each slot keeps a "
                        "small buffer of UNCOMPRESSED routed-token hidden states; "
                        "at readout the top-k slots' evidence is prepended as a "
                        "4th joint-attn segment [L3|L2|L1|evidence|H] so the "
                        "frozen decoder can recall precise facts the compressed "
                        "latent loses. Default False = no-op (byte-identical).")
    p.add_argument("--evidence_buffer_size", type=int, default=8,
                   help="Bcnt — evidence entries cached per slot (priority-"
                        "replaced by routing salience when full).")
    p.add_argument("--evidence_topr", type=int, default=0,
                   help="Evidence entries retrieved per slot at readout. 0 = use "
                        "evidence_buffer_size (retrieve all). Clamped to <= buffer.")
    p.add_argument("--evidence_layer", type=int, default=0,
                   help="The single memory layer index that caches + reads the "
                        "evidence buffer (shared bank → one owner). Default 0.")
    p.add_argument("--evidence_isolate_softmax", action="store_true", default=False,
                   help="Landmark EV-isolation: restrict H's prefix softmax to "
                        "the EV block (sever H->L3/L2/L1) so the precise evidence "
                        "tokens are not diluted by the compressed prefix.")

    # v6/v7 writeback (disabled by default for CPT)
    p.add_argument("--use_replace_writeback", action="store_true", default=False)
    p.add_argument("--num_global_slots", type=int, default=0)
    p.add_argument("--global_slot_forget_bias", type=float, default=1.0)
    p.add_argument("--global_slot_input_gate_only", action="store_true", default=False)

    # Activation-memory reduction
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)

    # FSDP (2026-06-04): shard the trainable mem_space adapter optimizer state
    # across ranks. With slot_dim=16384 the mem_space params balloon to ~6.5B;
    # DDP replicates the full AdamW state (6.5B*2*4B ≈ 52 GB) on every rank,
    # which OOMs the 95 GiB H20 alongside the 16 GB frozen backbone + grads +
    # activations. FSDP FULL_SHARD splits the optimizer state across 8 ranks.
    # Frozen Llama backbone stays replicated. DDP remains the default fallback.
    p.add_argument("--use_fsdp", action="store_true", default=False,
                   help="Use FullyShardedDataParallel (ZeRO-3) on trainable "
                        "mem_space layers instead of DDP. Frozen backbone stays "
                        "replicated. Required for large slot_dim on H20 (95 GiB).")

    # ----- TRUE in-attention K/V concat readout (2026-06-18) ----- #
    # The architecturally-correct injection channel: the inattn_kv_layer writes
    # every chunk's uncompressed token hidden into a per-sequence raw-KV store,
    # then at readout retrieves the top-inattn_kv_topk and concatenates their
    # native-projected K/V directly onto layer-N self-attention's K/V so the
    # H-query tokens attend over [native_KV ; retrieved_KV] in ONE softmax. The
    # injection K/V are produced by the LIVE k_proj/v_proj and feed the
    # differentiable softmax→o_proj path, so when the backbone is unfrozen
    # (--unfreeze_backbone) gradient flows through the injection → the model
    # learns to consume it (Landmark-faithful, fixes the OOD readout failure).
    p.add_argument("--use_inattn_kv", action="store_true", default=False,
                   help="Enable the TRUE in-attention K/V concat readout channel "
                        "(retrieve raw KV → concat onto native K/V at "
                        "inattn_kv_layer's self-attn). Injection path is in-graph; "
                        "pair with --unfreeze_backbone so grad flows into it.")
    p.add_argument("--inattn_kv_layer", type=int, default=16,
                   help="Single layer whose self-attention is wrapped for the "
                        "in-attention K/V concat (also the layer that writes the "
                        "raw-KV store). Default 16.")
    p.add_argument("--inattn_kv_topk", type=int, default=64,
                   help="Number of retrieved raw tokens concatenated onto the "
                        "K/V axis at readout. Default 64.")
    p.add_argument("--inattn_kv_layers", type=str, default=None,
                   help="Multi-layer injection (v2): comma-separated decoder "
                        "layer indices to inject the retrieved K/V at, e.g. "
                        "'16,20,24'. Overrides the single --inattn_kv_layer for "
                        "the READ/inject; the WRITE owner is the smallest index. "
                        "All listed layers MUST be within the unfrozen range so "
                        "they can learn to attend the injection. Default None → "
                        "single-layer behaviour (byte-identical to v1).")

    # Raw-KV READOUT — Method A (per-chunk raw-KV + emergent trainable gist-key
    # soft attention). docs/RAWKV_READOUT_PROPOSAL.md §2. The structural
    # successor to --use_inattn_kv: same in-attention raw-KV concat readout, but
    # the retrieval is a DIFFERENTIABLE, TRAINABLE gist-key soft attention (the
    # Landmark mechanism) instead of the non-differentiable TopKSelector hard
    # top-k. MUST pair with --unfreeze_backbone --unfreeze_layers_from N so the
    # reader learns to consume the injected raw KV AND the gist scorer (whose
    # selection weight rides into the native softmax as a per-column log-bias)
    # gets gradient. Default off → byte-identical to pre.
    p.add_argument("--use_rawkv_readout", action="store_true", default=False,
                   help="Enable Method A raw-KV readout (trainable gist-key soft "
                        "attention + in-attention raw-KV concat). Pair with "
                        "--unfreeze_layers_from so the reader + gist scorer train.")
    p.add_argument("--rawkv_readout_layer", type=int, default=16,
                   help="Single layer whose self-attn consumes the retrieved raw "
                        "KV + writes the store. Default 16.")
    p.add_argument("--rawkv_readout_layers", type=str, default=None,
                   help="Multi-layer readout: comma-separated decoder layer "
                        "indices that consume the retrieved raw KV, e.g. "
                        "'16,20,24'. Overrides --rawkv_readout_layer; the store "
                        "WRITE owner is the smallest index. All listed layers "
                        "should be within the unfrozen range. Default None → "
                        "single-layer.")
    p.add_argument("--rawkv_gist_dim", type=int, default=128,
                   help="Dim of the trainable gist-key scoring space. Default 128.")
    p.add_argument("--rawkv_readout_topk_chunks", type=int, default=8,
                   help="Soft-top-k: keep this many highest-weight chunks at "
                        "read (0 / >= n_chunks → keep all = pure soft attention; "
                        "weights stay differentiable either way). Default 8.")
    p.add_argument("--rawkv_readout_temp", type=float, default=1.0,
                   help="Softmax temperature for the gist soft selection. "
                        "Default 1.0.")

    # ----- Full fine-tune: unfreeze the entire Llama backbone (2026-06-18) ----- #
    # Landmark-faithful SFT: the frozen reader cannot consume injected KV through
    # an attention path it was never trained on (oracle-perfect needle still ≈OFF
    # → OOD mismatch). The fix is to train ON the injection mechanism so the model
    # recalibrates to the injected distribution. --unfreeze_backbone makes the
    # ENTIRE Llama-3-8B backbone (attention + FFN + embed + lm_head) trainable
    # alongside the memory adapter. Default False → byte-identical to the existing
    # frozen-backbone pipeline (only mem_space params train).
    p.add_argument("--unfreeze_backbone", action="store_true", default=False,
                   help="Full fine-tune: unfreeze the ENTIRE Llama backbone "
                        "(attn+FFN+embed+lm_head) in addition to the mem_space "
                        "adapter. Default False keeps the frozen-backbone pipeline "
                        "byte-identical. Use lr~2e-5 + FSDP for 8B full FT.")
    # Partial unfreeze (v2, 2026-06-19): unfreeze only the TOP decoder layers
    # (>= this index) + the memory adapter; keep layers below + embed_tokens
    # frozen. v1's full unfreeze on short dolmino DAMAGED the base LM (niah
    # OFF 22→11). Protecting the lower layers + embed preserves the base LM's
    # competence while still letting the upper layers (where injection happens)
    # learn to consume injected KV. Requires --unfreeze_backbone. Default -1 →
    # disabled (full unfreeze, byte-identical to v1 when set).
    p.add_argument("--unfreeze_layers_from", type=int, default=-1,
                   help="Partial unfreeze: only decoder layers with index >= N "
                        "are trainable (plus the mem_space adapter + final norm "
                        "+ lm_head); layers 0..N-1 and embed_tokens stay frozen. "
                        "Requires --unfreeze_backbone. Default -1 = full unfreeze.")
    # One-time grad-flow diagnostic: on the FIRST optimizer step, print backbone
    # param grad norms + the injected K_raw grad status, then disable itself.
    # Used by the unfreeze/in-graph smoke; no-op (and zero overhead) by default.
    p.add_argument("--grad_flow_diag", action="store_true", default=False,
                   help="Print a one-time grad-flow diagnostic on the first step "
                        "(backbone param grad norms + injected-KV in-graph check) "
                        "then disable. For the unfreeze/in-graph smoke only.")

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

    args = p.parse_args()
    # Parse comma-separated --inattn_kv_layers "16,20,24" → sorted unique list.
    # Empty/None → None (single-layer behaviour). Validated against the unfrozen
    # range later in main() once num_layers is known.
    if isinstance(args.inattn_kv_layers, str) and args.inattn_kv_layers.strip():
        args.inattn_kv_layers = sorted({
            int(x) for x in args.inattn_kv_layers.split(",") if x.strip() != ""
        })
    else:
        args.inattn_kv_layers = None
    # Parse comma-separated --rawkv_readout_layers (Method A) → sorted list.
    if isinstance(args.rawkv_readout_layers, str) and args.rawkv_readout_layers.strip():
        args.rawkv_readout_layers = sorted({
            int(x) for x in args.rawkv_readout_layers.split(",") if x.strip() != ""
        })
    else:
        args.rawkv_readout_layers = None
    return args


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
        slot_dim=args.slot_dim,
        writeback_mode=args.writeback_mode,
        lowrank_gate_rank=args.lowrank_gate_rank,
        writeback_gate_warmup_steps=args.writeback_warmup_steps,
        writeback_gate_max=args.writeback_gate_max,
        load_balance_weight=args.load_balance_weight,
        entropy_aux_weight=args.entropy_aux_weight,
        use_loss_free_balance=args.use_loss_free_balance,
        loss_free_update_rate=args.loss_free_update_rate,
        use_st_gumbel_topk=args.use_st_gumbel_topk,
        st_gumbel_temperature=args.st_gumbel_temperature,
        use_delta_rule_writeback=args.use_delta_rule_writeback,
        normalize_readout=args.normalize_readout,
        readout_norm_scale=args.readout_norm_scale,
        independent_slot_key=args.independent_slot_key,
        delta_erase_write=args.delta_erase_write,
        use_l2=args.use_l2,
        l2_compress_ratio=args.l2_compress_ratio,
        l2_d_c=args.l2_d_c,
        l2_d_h_rope=args.l2_d_h_rope,
        l2_init_scale=args.l2_init_scale,
        selector_temperature=args.selector_temperature,
        key_repulsion_weight=args.key_repulsion_weight,
        key_repulsion_threshold=args.key_repulsion_threshold,
        l3_diversity_weight=args.l3_diversity_weight,
        l3_diversity_threshold=args.l3_diversity_threshold,
        l_recon_weight=args.l_recon_weight,
        l3_recon_token_weight=args.l3_recon_token_weight,
        l3_recon_max_positions=args.chunk_size,
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
        use_decoupled_read=args.use_decoupled_read,
        use_memory_xattn=args.use_memory_xattn,
        memory_xattn_gate_init=args.memory_xattn_gate_init,
        memory_xattn_disable_null_sink=args.memory_xattn_disable_null_sink,
        use_readout_mass_bias=args.use_readout_mass_bias,
        readout_mass_coef=args.readout_mass_coef,
        use_slot_evidence=args.use_slot_evidence,
        evidence_buffer_size=args.evidence_buffer_size,
        evidence_topr=args.evidence_topr,
        evidence_layer=args.evidence_layer,
        evidence_isolate_softmax=args.evidence_isolate_softmax,
        dead_slot_reset_interval=args.dead_slot_reset_interval,
        dead_slot_reset_mode=args.dead_slot_reset_mode,
        dead_slot_grace_chunks=args.dead_slot_grace_chunks,
        dead_slot_criterion=args.dead_slot_criterion,
        soft_write_weight=args.soft_write_weight,
        soft_write_content=args.soft_write_content,
        slot_read_decay_rate=args.slot_read_decay_rate,
        slot_read_decay_interval=args.slot_read_decay_interval,
        slot_read_decay_min_keep=args.slot_read_decay_min_keep,
        slot_evict_mode=args.slot_evict_mode,
        slot_evict_max_frac=args.slot_evict_max_frac,
        slot_evict_floor=args.slot_evict_floor,
        slot_evict_protect_chunks=args.slot_evict_protect_chunks,
        use_inattn_kv=args.use_inattn_kv,
        inattn_kv_layer=args.inattn_kv_layer,
        inattn_kv_topk=args.inattn_kv_topk,
        inattn_kv_layers=args.inattn_kv_layers,
        use_rawkv_readout=args.use_rawkv_readout,
        rawkv_readout_layer=args.rawkv_readout_layer,
        rawkv_readout_layers=args.rawkv_readout_layers,
        rawkv_gist_dim=args.rawkv_gist_dim,
        rawkv_readout_topk_chunks=args.rawkv_readout_topk_chunks,
        rawkv_readout_temp=args.rawkv_readout_temp,
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
# Batch collation + batch-dim helpers (batch_size > 1 support, 2026-06-07)
# --------------------------------------------------------------------------- #


def dolmino_collate_fn(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Stack ``batch_size`` Dolmino samples into batched tensors.

    Each sample (from ``DolminoCurriculumDataset``) has:
        ``context_chunks``: list of n_ctx tensors, each [chunk_size]
        ``target_ids``:     tensor [chunk_size]
        (optionally ``reset_flags``)

    All samples in a batch share the SAME ``n_ctx`` (curriculum n_ctx is fixed
    for the whole step) and the SAME ``chunk_size``. We stack the k-th context
    chunk across the batch -> [B, chunk_size], and the target -> [B, chunk_size].

    Returns a dict with:
        ``context_chunks``: list of n_ctx tensors, each [B, chunk_size]
        ``target_ids``:     tensor [B, chunk_size]
        ``is_dolmino``:     True

    Robustness: if samples disagree on chunk length (should not happen with the
    uniform-1024 Dolmino rows), we truncate every chunk to the per-position
    minimum length across the batch. Truncation (not padding) keeps numerical
    correctness — we never feed fake pad tokens as NTP targets.

    Per-sample ``reset_flags`` (only produced by ``--doc_reset`` contiguous
    mode) are INCOMPATIBLE with batching: the memory bank cannot reset a single
    batch element mid-rollout. We assert they are absent here; ``main`` only
    uses this collate when ``batch_size > 1`` and rejects ``--doc_reset`` in
    that case at startup.
    """
    if not batch:
        raise ValueError("dolmino_collate_fn received an empty batch")

    n_ctx = len(batch[0]["context_chunks"])
    for s in batch:
        if len(s["context_chunks"]) != n_ctx:
            raise ValueError(
                "dolmino_collate_fn: samples in a batch have different n_ctx "
                f"({len(s['context_chunks'])} vs {n_ctx}); curriculum n_ctx must "
                "be constant within a step."
            )
        if s.get("reset_flags") is not None:
            raise ValueError(
                "dolmino_collate_fn: reset_flags present (--doc_reset). "
                "doc_reset is incompatible with batch_size > 1; the memory bank "
                "cannot reset a single batch element mid-rollout."
            )

    # Stack the k-th context chunk across the batch -> [B, chunk_size].
    context_chunks: List[torch.Tensor] = []
    for k in range(n_ctx):
        col = [s["context_chunks"][k] for s in batch]
        min_len = min(t.shape[0] for t in col)
        context_chunks.append(torch.stack([t[:min_len] for t in col], dim=0))

    tgt_col = [s["target_ids"] for s in batch]
    tgt_min = min(t.shape[0] for t in tgt_col)
    target_ids = torch.stack([t[:tgt_min] for t in tgt_col], dim=0)

    return {
        "context_chunks": context_chunks,
        "target_ids": target_ids,
        "is_dolmino": True,
    }


def _ensure_batched(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Move ``t`` to ``device`` and guarantee a leading batch dim.

    Accepts either a 1-D [chunk_size] tensor (legacy batch_size=1 path, where
    the DataLoader yields single un-collated samples) or an already-batched
    2-D [B, chunk_size] tensor (batch_size>1 path via ``dolmino_collate_fn``).
    Returns [B, chunk_size]. This is the single replacement for the previously
    hard-coded ``.unsqueeze(0)`` so both paths share one code path and the
    batch_size=1 behavior stays byte-identical.
    """
    t = t.to(device)
    if t.dim() == 1:
        t = t.unsqueeze(0)
    return t


# --------------------------------------------------------------------------- #
# Self-study distillation helpers (v21, 2026-06-15)
# --------------------------------------------------------------------------- #
# numpy has no native bfloat16, so build_distill_cache.py stores bf16 tensors as
# their raw int16 bit-pattern (tensor.view(torch.int16)). We reconstruct here by
# reinterpreting the int16 bits back into bfloat16. This keeps full bf16 fidelity
# and a single source of truth shared with the cache builder + unit tests.


def _bf16_from_int16(arr) -> torch.Tensor:
    """Reinterpret an int16 numpy array (bf16 bit-pattern) back to bfloat16."""
    return torch.from_numpy(arr).view(torch.bfloat16)


def load_distill_npz(path: str, device: torch.device):
    """Load one teacher cache .npz and return a dict of tensors on ``device``.

    Returns:
        dict with
          logit_idx  long  [A, topk]
          logit_val  bf16  [A, topk]  (raw teacher logits)
          hidden     bf16  [A, n_sel, 4096]
          answer_mask bool [A]
          layers     long  [n_sel]    (decoder-layer indices)
    """
    import numpy as _np
    z = _np.load(path)
    return {
        "logit_idx": torch.from_numpy(z["logit_idx"]).to(device).long(),
        "logit_val": _bf16_from_int16(z["logit_val"]).to(device),
        "hidden": _bf16_from_int16(z["hidden"]).to(device),
        "answer_mask": torch.from_numpy(z["answer_mask"]).to(device).bool(),
        "layers": torch.from_numpy(z["meta_layers"]).long(),
    }


def distill_logits_kl(
    student_logits: torch.Tensor,
    teacher_idx: torch.Tensor,
    teacher_val: torch.Tensor,
    lam: float = 0.6,
) -> torch.Tensor:
    """Scheme-A bidirectional KL on the teacher top-k support.

    For each answer token: teacher logits (top-k) are softmaxed over the k
    support -> p; student logits gathered at the SAME k indices, softmaxed over
    that support -> q. Loss = lam*KL(p||q) + (1-lam)*KL(q||p), averaged over
    answer tokens. teacher (p) carries no grad (cache constant); grad flows only
    through q -> the readout.

    Args:
        student_logits: [A, V] float (the student's answer-segment logits)
        teacher_idx:    [A, k] long  (top-k vocab indices)
        teacher_val:    [A, k] (teacher raw logits at those indices)
    """
    teacher_val = teacher_val.to(student_logits.dtype)
    # Teacher distribution over its own top-k support (renormalised). Constant.
    p = torch.softmax(teacher_val, dim=-1)  # [A, k]
    # Student logits gathered at the same support, softmaxed over the support.
    q_logits = torch.gather(student_logits, -1, teacher_idx)  # [A, k]
    log_q = torch.log_softmax(q_logits, dim=-1)  # [A, k]
    log_p = torch.log_softmax(teacher_val, dim=-1)  # [A, k]
    # KL(p||q) = sum p*(log p - log q); KL(q||p) = sum q*(log q - log p).
    q = log_q.exp()
    kl_pq = (p * (log_p - log_q)).sum(dim=-1)  # [A]
    kl_qp = (q * (log_q - log_p)).sum(dim=-1)  # [A]
    loss = lam * kl_pq + (1.0 - lam) * kl_qp  # [A]
    return loss.mean()


def distill_hidden_cosine(
    student_hidden: torch.Tensor,
    teacher_hidden: torch.Tensor,
) -> torch.Tensor:
    """Scheme-B hidden matching: 1 - cos, averaged over answer tokens, summed
    over layers. teacher_hidden is a cache constant (stopgrad — no grad anyway).

    Args:
        student_hidden: [A, n_sel, D] (selected student layers, answer segment)
        teacher_hidden: [A, n_sel, D] (cached teacher hidden, same layers/tokens)
    """
    teacher_hidden = teacher_hidden.to(student_hidden.dtype).detach()
    cos = torch.nn.functional.cosine_similarity(
        student_hidden, teacher_hidden, dim=-1)  # [A, n_sel]
    per_layer = (1.0 - cos).mean(dim=0)  # [n_sel] (avg over answer tokens)
    return per_layer.sum()  # sum over layers


def assert_distill_cache_consistent(cache_dir, train_chunk_size, train_n_ctx,
                                    train_layers, train_fingerprint=None):
    """Guard against silently training on a mis-sliced teacher cache (v21).

    The teacher cache is keyed by stable (doc_idx, group_pos) windows of length
    (n_ctx+1)*chunk_size and stores per-layer hidden states for specific decoder
    layers. If the training config does not match how the cache was built, the
    (doc_idx, group_pos) keys point at DIFFERENT token windows and/or the hidden
    layers misalign -> the teacher signal gets 张冠李戴 (mismatched) and we
    silently train garbage. This reads <cache_dir>/meta.json (written by
    build_distill_cache.py) and raises RuntimeError on any mismatch.

    Args:
        cache_dir: --distill_cache_dir.
        train_chunk_size: args.chunk_size.
        train_n_ctx: curriculum.get_n_ctx(0) (starting n_ctx; dolmino is locked).
        train_layers: parsed list[int] of --distill_layers.

    Returns the loaded meta dict on success.
    """
    meta_path = os.path.join(cache_dir, "meta.json")
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"[distill] cache meta.json not found at {meta_path}. The distill "
            "cache must be built first with scripts/build_distill_cache.py "
            "(which writes meta.json). Run e.g. "
            "`python scripts/build_distill_cache.py --dolmino_path ... "
            "--chunk_size {cs} --n_ctx {n} --distill_layers {l} "
            "--out_dir {d}`.".format(
                cs=train_chunk_size, n=train_n_ctx,
                l=",".join(str(x) for x in train_layers), d=cache_dir))
    with open(meta_path) as f:
        meta = json.load(f)
    meta_n_ctx = int(meta.get("n_ctx"))
    meta_chunk = int(meta.get("chunk_size"))
    meta_layers = [int(x) for x in meta.get("distill_layers", [])]
    train_layers = [int(x) for x in train_layers]
    if (meta_chunk != int(train_chunk_size)
            or meta_n_ctx != int(train_n_ctx)
            or meta_layers != train_layers):
        raise RuntimeError(
            "[distill] cache/training config MISMATCH. The teacher cache at "
            f"{cache_dir} was built with n_ctx={meta_n_ctx} "
            f"chunk_size={meta_chunk} distill_layers={meta_layers}, but this "
            f"training run uses n_ctx={int(train_n_ctx)} "
            f"chunk_size={int(train_chunk_size)} distill_layers={train_layers}. "
            "The cache is keyed by (doc_idx, group_pos) over windows of length "
            "(n_ctx+1)*chunk_size and stores per-layer hidden states, so a "
            "group-window / layer mismatch makes the teacher signal 张冠李戴 "
            "(misaligned) and silently trains garbage. Rebuild the cache with "
            "matching params (build_distill_cache.py) or change the training "
            "config to match the cache."
        )
    # Dataset-identity guard (2026-06-16): sample_id=(doc_idx,group_pos) is a
    # POSITIONAL index into the Arrow the cache was built from. If training uses
    # a dataset with a different _fingerprint (different row order — e.g. an
    # independently reprocessed copy on another disk), every (doc_idx,group_pos)
    # points at a DIFFERENT document -> 100% silent cache miss -> distill_kl/hid
    # stay 0 and the run trains as if no distillation (wasted compute). Fail fast.
    meta_fp = meta.get("dataset_fingerprint")
    if train_fingerprint is not None and meta_fp is not None \
            and str(meta_fp) != str(train_fingerprint):
        raise RuntimeError(
            "[distill] cache/dataset FINGERPRINT MISMATCH. The teacher cache at "
            f"{cache_dir} was built from a dolmino dataset with "
            f"_fingerprint={meta_fp}, but this training run's dolmino dataset has "
            f"_fingerprint={train_fingerprint}. sample_id=(doc_idx,group_pos) is "
            "a positional index into the Arrow, so a different row order makes "
            "every key point at a different document -> 100% silent cache miss "
            "(distill loss stays 0, no distillation actually happens). Use the "
            "SAME physical Arrow that the cache was built from (rsync the dataset "
            "+ cache together; never independently reprocess per node), or "
            "rebuild the cache against this dataset."
        )
    return meta


# --------------------------------------------------------------------------- #
# Dolmino training step
# --------------------------------------------------------------------------- #


def dolmino_train_step(
    model: torch.nn.Module,
    context_chunks: List[torch.Tensor],
    target_ids: torch.Tensor,
    device: torch.device,
    grad_accum: int = 1,
    answer_mask: Optional[torch.Tensor] = None,
    teacher_cache: Optional[dict] = None,
    distill_cfg: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor]:
    """HARDER objective (2026-06-13, last_chunk_loss_only): force memory to work.
    1. Reset banks
    2. Stream context chunks through model (NO GRAD) → memory accumulates
    3. Detach banks (break gradient graph)
    4. Forward target chunk with grad → NTP loss ONLY on target chunk

    Unlike dolmino_train_step_tbptt (every chunk predicts its own next-token via
    local causal attn → memory barely needed), here the target chunk's prediction
    can ONLY draw on prior context through the memory bank (context ran no_grad,
    not in the target's attention window). This is the BABILong-style streaming
    objective but on GENERIC dolmino text (no QA → no eval contamination), to
    pressure the memory channel to actually carry long-range info.

    answer_mask (T2, 2026-06-14): optional [chunk_size] (or [B, chunk_size]) bool
    tensor. When provided (NIAHChunkedDataset associative-recall samples), LM loss
    is restricted to ONLY the positions where answer_mask is True (the answer digit
    tokens) — every other target position is set to -100. This sharpens the
    "precise readout" pressure: the answer can ONLY be recovered by addressing the
    exact memory slot that encoded the needle. When None (default), behaviour is
    BIT-IDENTICAL to the original full-target-chunk NTP loss.

    Self-study distillation (v21, 2026-06-15): when ``teacher_cache`` AND
    ``distill_cfg`` are provided (only for dolmino steps, never T2), the target
    forward additionally requests ``output_hidden_states=True`` and we compute:
      A) bidirectional KL between student answer logits and the cached teacher
         top-64 distribution (distill_logits_kl);
      B) 1-cos hidden matching on the selected layers (distill_hidden_cosine).
    total = lm_loss + aux_loss + distill_weight*(L_A + beta*L_B), and THIS is
    what gets backward'd. When teacher_cache/distill_cfg is None (default), NOT
    a single extra kwarg is passed to model(...) → byte-identical to before.

    Returns: (lm_loss, aux_loss, route_aux=0, l3recon=0, distill_kl, distill_hidden)
    — all scaled by grad_accum where applicable; distill terms are detached and
    returned only for logging. Backward is done HERE (mirrors tbptt).
    """
    _reset_banks(model)
    scale = float(grad_accum)

    # Stream context chunks through memory (no gradient)
    with torch.no_grad():
        for ctx in context_chunks:
            ctx_input = _ensure_batched(ctx, device)  # [B, chunk_size]
            model(input_ids=ctx_input, use_cache=False)

    # Detach memory banks to prevent gradient flow through context passes
    _detach_banks(model)

    # Forward target chunk with gradient
    target_input = _ensure_batched(target_ids, device)  # [B, chunk_size]
    if answer_mask is None:
        # Default path: full-target NTP loss (bit-identical to original).
        labels = target_input
    else:
        # T2 path: keep labels only on answer-digit positions, -100 elsewhere.
        # HF CausalLM shifts internally (labels[i] supervises the logits that
        # predict token i), so we place the true token id at each answer
        # position and mask everything else — matching niah_dataset.py's scheme.
        mask = _ensure_batched(answer_mask, device).to(torch.bool)  # [B, chunk_size]
        labels = torch.where(
            mask, target_input, torch.full_like(target_input, -100)
        )

    distill_on = teacher_cache is not None and distill_cfg is not None
    _zero = torch.zeros((), device=device)

    if not distill_on:
        # ORIGINAL path — no output_hidden_states kwarg → byte-identical.
        out = model(input_ids=target_input, labels=labels, use_cache=False)
        lm_loss = out.loss / scale
        aux_loss = _collect_aux_loss(model, device) / scale
        (lm_loss + aux_loss).backward()
        return lm_loss.detach(), aux_loss.detach(), _zero, _zero, _zero, _zero

    # ----- distillation path (dolmino only) ----- #
    want_hidden = bool(distill_cfg.get("distill_hidden", False))
    out = model(input_ids=target_input, labels=labels, use_cache=False,
                output_hidden_states=want_hidden)
    lm_loss = out.loss / scale
    aux_loss = _collect_aux_loss(model, device) / scale

    # Student answer-segment is the WHOLE target chunk (dolmino). Batch dim is 1
    # for the distill path (cache is per-sample, batch_size=1 dolmino).
    chunk_size = target_input.shape[1]
    distill_kl = _zero
    distill_hidden = _zero
    L_A = torch.zeros((), device=device)
    L_B = torch.zeros((), device=device)

    if distill_cfg.get("distill_logits", False):
        student_logits = out.logits[0]  # [chunk_size, V]
        t_idx = teacher_cache["logit_idx"]  # [A, k]
        t_val = teacher_cache["logit_val"]  # [A, k]
        A = min(chunk_size, t_idx.shape[0])
        L_A = distill_logits_kl(
            student_logits[:A], t_idx[:A], t_val[:A],
            lam=float(distill_cfg.get("distill_lambda", 0.6)),
        )
        distill_kl = L_A.detach()

    if want_hidden:
        # hidden_states is tuple len n_layers+1; decoder layer L -> index L+1.
        layers = teacher_cache["layers"].tolist()
        hs = out.hidden_states
        student_layers = torch.stack(
            [hs[L + 1][0] for L in layers], dim=1)  # [chunk_size, n_sel, D]
        t_hidden = teacher_cache["hidden"]  # [A, n_sel, D]
        A = min(chunk_size, t_hidden.shape[0])
        L_B = distill_hidden_cosine(student_layers[:A], t_hidden[:A])
        distill_hidden = L_B.detach()

    distill_weight = float(distill_cfg.get("distill_weight", 1.0))
    beta = float(distill_cfg.get("distill_hidden_beta", 1.0))
    # distill term scaled by grad_accum to match lm_loss/aux_loss scaling.
    distill_term = distill_weight * (L_A + beta * L_B) / scale
    total = lm_loss + aux_loss + distill_term
    total.backward()
    return (lm_loss.detach(), aux_loss.detach(), _zero, _zero,
            distill_kl, distill_hidden)


def dolmino_train_step_tbptt(
    model: torch.nn.Module,
    context_chunks: List[torch.Tensor],
    target_ids: torch.Tensor,
    device: torch.device,
    grad_accum: int = 1,
    bptt_window: int = 2,
    route_aux_weight: float = 0.0,
    reset_flags: Optional[List[bool]] = None,
    l3_recon_token_weight: float = 0.0,
    swa_train_chunks: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

    Returns: (total_lm * grad_accum, total_aux * grad_accum,
    total_route_aux * grad_accum) — the LM/aux semantics are unchanged from the
    prior implementation; total_route_aux is the (unweighted) routing-supervision
    CE summed over chunks for logging (always 0 when route_aux_weight == 0).

    route_aux (E2, 2026-06-04): when ``route_aux_weight > 0``, for each chunk we
    capture the grad-bearing routing scores [B, N] via a forward hook on the
    layer-0 selector, and supervise them (cross-entropy) to place mass on the
    slots the PREVIOUS chunk wrote into (``prev_idx1`` = that chunk's detached
    last_idx). This is the real-path analogue of toy E2's cross-chunk
    write->read routing supervision. The CE term is folded into the live window
    loss so its gradient flows to the selector through the same backward.
    """
    _reset_banks(model)

    mem_layers = _mem_layers(model) if route_aux_weight > 0.0 else None
    use_route_aux = bool(route_aux_weight > 0.0 and mem_layers)

    n_chunks = len(context_chunks) + 1
    scale = n_chunks * grad_accum
    total_lm = torch.zeros((), device=device)
    total_aux = torch.zeros((), device=device)
    total_route_aux = torch.zeros((), device=device)
    total_l3recon = torch.zeros((), device=device)

    use_l3_recon = bool(l3_recon_token_weight > 0.0)

    bptt_window = max(1, int(bptt_window))

    # Build the full chunk list: all context chunks followed by the target.
    # Each entry: (input_ids_for_forward, is_target).
    all_inputs = [(ctx, False) for ctx in context_chunks]
    all_inputs.append((target_ids, True))

    # Accumulator for the current window's loss (keeps the autograd graph
    # connecting consecutive chunks alive until we backward at the boundary).
    window_loss = None  # type: Optional[torch.Tensor]

    # route_aux: detached slot indices written by the PREVIOUS chunk; used as
    # the routing-supervision target for the current chunk's scores.
    prev_idx1 = None  # type: Optional[torch.Tensor]

    for i, (chunk_ids, _is_target) in enumerate(all_inputs):
        # doc_reset (contiguous mode): if this chunk begins a NEW document
        # (reset_flags[i] True; i==0 is the group start whose bank is already
        # fresh from _reset_banks above), flush the pending window so the BPTT
        # graph never spans the document boundary, then reset the memory bank
        # BEFORE forwarding this chunk. Cross-document credit assignment is
        # meaningless, so we sever it here. No-op when reset_flags is None
        # (doc_reset disabled) -> zero behaviour change.
        if reset_flags is not None and i > 0 and i < len(reset_flags) and reset_flags[i]:
            if window_loss is not None:
                window_loss.backward()
                window_loss = None
                _detach_banks(model)
            _reset_banks(model)
            prev_idx1 = None  # previous doc's writes are gone; don't supervise across

        chunk_input = _ensure_batched(chunk_ids, device)

        # D2b cross-chunk SWA TRAIN window (2026-06-09): for the TARGET chunk
        # only, optionally widen the grad-bearing forward from a single chunk to
        # the concatenation of the last W context chunks' raw tokens + the
        # target, so the target tokens can directly attend those W chunks' raw KV
        # (sliding window) IN ADDITION to the memory bank. Train-side analogue of
        # the eval-side --swa_eval_chunks (D2a). The prefix (re-presented context)
        # labels are masked to -100 so the LM loss is computed ONLY on the target
        # tokens — context chunks already contributed their own per-chunk loss
        # earlier in this loop, so masking avoids double-counting and keeps the
        # loss magnitude / scale identical to the W0 path. The bank is frozen
        # around this forward so the re-presented W chunks (already streamed in
        # by the loop above) are NOT written a second time; frozen blocks writes
        # only, so the read path still reads the fully-accumulated bank and the
        # cross-chunk BPTT credit (target read -> prior chunk writes) is intact.
        # Concatenation gives the window correct RELATIVE RoPE positions (the
        # whole point), instead of the target restarting at position 0.
        swa_active = (
            _is_target
            and swa_train_chunks > 0
            and reset_flags is None          # never span a doc_reset boundary
            and len(context_chunks) >= 1     # need >=1 context chunk to attend
        )
        fwd_labels = chunk_input
        swa_froze = False
        if swa_active:
            w = min(swa_train_chunks, len(context_chunks))
            win_pieces = [
                _ensure_batched(context_chunks[j], device)
                for j in range(len(context_chunks) - w, len(context_chunks))
            ]
            win_pieces.append(chunk_input)
            window = torch.cat(win_pieces, dim=1)   # [B, w*chunk_size + T]
            prefix_len = window.shape[1] - chunk_input.shape[1]
            fwd_labels = window.clone()
            fwd_labels[:, :prefix_len] = -100       # loss only on target tokens
            chunk_input = window
            _set_banks_frozen(model, True)
            swa_froze = True

        # route_aux: capture this chunk's grad-bearing routing scores via a
        # forward hook on the layer-0 selector (last_scores is detached on the
        # layer, so the hook is the only grad-bearing source).
        captured = {}
        hook_handle = None
        if use_route_aux:
            captured, hook_handle = _install_score_hook(mem_layers)

        out = model(input_ids=chunk_input, labels=fwd_labels, use_cache=False)
        if hook_handle is not None:
            hook_handle.remove()
        if swa_froze:
            _set_banks_frozen(model, False)
        chunk_lm = out.loss / scale
        chunk_aux = _collect_aux_loss(model, device) / scale

        # Add this chunk's loss to the live window graph (do NOT backward yet:
        # the next chunk's memory read must stay connected to this chunk's
        # write graph for true cross-chunk BPTT).
        step_loss = chunk_lm + chunk_aux

        # ICAE token recon (2026-06-07): reconstruct THIS chunk's discrete tokens
        # from a fresh grad-bearing L3 summary of this chunk. The L3 post-forward
        # hook stashed the grad-bearing top-layer hidden during the forward
        # above. Folded into the live window so its gradient flows to l3_pool +
        # head through the same backward. Computed per chunk (every chunk here
        # is grad-bearing). Scaled by 1/scale to match chunk_lm/chunk_aux.
        if use_l3_recon:
            l3r = _compute_l3_token_recon(model, chunk_input)
            _clear_l3_token_recon_cur_h(model)
            if l3r is not None:
                step_loss = step_loss + l3_recon_token_weight * (l3r / scale)
                total_l3recon = total_l3recon + (l3r / scale).detach()

        # route_aux: supervise the current chunk's routing scores against the
        # slots the previous chunk wrote into (cross-chunk write->read).
        if use_route_aux and prev_idx1 is not None:
            scores2 = captured.get("scores")
            ra = _compute_route_aux(scores2, prev_idx1, device)
            if ra is not None:
                step_loss = step_loss + route_aux_weight * (ra / scale)
                total_route_aux = total_route_aux + (ra / scale).detach()

        window_loss = step_loss if window_loss is None else (window_loss + step_loss)

        total_lm = total_lm + chunk_lm.detach()
        total_aux = total_aux + chunk_aux.detach()

        # route_aux: record this chunk's written slots (detached) as the target
        # for the NEXT chunk's routing supervision.
        if use_route_aux:
            li = getattr(mem_layers[0], "last_idx", None)
            prev_idx1 = li.detach() if li is not None else None

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

    return (total_lm * grad_accum, total_aux * grad_accum,
            total_route_aux * grad_accum, total_l3recon * grad_accum)


# --------------------------------------------------------------------------- #
# BABILong training step (chunked, from babilong script)
# --------------------------------------------------------------------------- #


def babilong_train_step(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    chunk_size: int,
    device: torch.device,
    route_aux_weight: float = 0.0,
    l3_recon_token_weight: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Stream BABILong sample through memory in chunks, gradient on last chunk.

    Returns (lm_loss, aux_loss, route_aux, l3recon). route_aux (E2) supervises
    the grad-bearing last chunk's routing scores against the slots the last
    context chunk wrote into (cross-chunk write->read). It is returned UNWEIGHTED
    for logging; the caller folds ``route_aux_weight * route_aux`` into the loss.
    Always 0 when route_aux_weight == 0 or the sample is a single chunk (no
    prior write to supervise against).

    l3recon (ICAE token recon, 2026-06-07): the UNWEIGHTED token-level
    reconstruction CE for the grad-bearing chunk; the caller folds
    ``l3_recon_token_weight * l3recon`` into the loss. Always 0 when the head is
    absent / weight 0.
    """
    _reset_banks(model)
    total_len = input_ids.shape[1]
    n_chunks = max(1, math.ceil(total_len / chunk_size))

    zero = torch.zeros((), device=device)
    use_l3_recon = bool(l3_recon_token_weight > 0.0)

    if n_chunks == 1:
        out = model(input_ids=input_ids, labels=labels, use_cache=False)
        l3recon = zero
        if use_l3_recon:
            _l3r = _compute_l3_token_recon(model, input_ids)
            _clear_l3_token_recon_cur_h(model)
            if _l3r is not None:
                l3recon = _l3r
        return out.loss, _collect_aux_loss(model, device), zero, l3recon

    mem_layers = _mem_layers(model) if route_aux_weight > 0.0 else None
    use_route_aux = bool(route_aux_weight > 0.0 and mem_layers)

    pieces_in = list(input_ids[0].split(chunk_size))
    pieces_lbl = list(labels[0].split(chunk_size))

    with torch.no_grad():
        for ci in pieces_in[:-1]:
            model(input_ids=ci.unsqueeze(0).to(device), use_cache=False)

    # route_aux: the slots the last context chunk wrote into (detached) are the
    # routing-supervision target for the grad-bearing last chunk.
    prev_idx1 = None
    if use_route_aux:
        li = getattr(mem_layers[0], "last_idx", None)
        prev_idx1 = li.detach() if li is not None else None

    # Detach before the gradient-bearing last chunk
    _detach_banks(model)

    last_in = pieces_in[-1].unsqueeze(0).to(device)
    last_lbl = pieces_lbl[-1].unsqueeze(0).to(device)

    captured = {}
    hook_handle = None
    if use_route_aux and prev_idx1 is not None:
        captured, hook_handle = _install_score_hook(mem_layers)

    out = model(input_ids=last_in, labels=last_lbl, use_cache=False)
    if hook_handle is not None:
        hook_handle.remove()

    route_aux = zero
    if use_route_aux and prev_idx1 is not None:
        ra = _compute_route_aux(captured.get("scores"), prev_idx1, device)
        if ra is not None:
            route_aux = ra

    # ICAE token recon on the grad-bearing last chunk.
    l3recon = zero
    if use_l3_recon:
        _l3r = _compute_l3_token_recon(model, last_in)
        _clear_l3_token_recon_cur_h(model)
        if _l3r is not None:
            l3recon = _l3r

    return out.loss, _collect_aux_loss(model, device), route_aux, l3recon


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
    """Save mem_space adapter weights + config.

    Supports both DDP and FSDP. For FSDP, gathers FULL_STATE_DICT to rank 0
    (CPU-offloaded) via the FSDP.state_dict_type context manager; all ranks must
    enter the context (it is a collective), but only rank 0 writes the ckpt.
    """
    fragments = (
        "selector", "gate_param", "slot_output_gate",
        "slot_to_hidden", "hidden_to_slot", "memory_bank",
        "gate_proj_new", "gate_proj_mem", "gate_bias",
        "lr_V_new", "lr_V_mem", "lr_U", "lr_gate_bias",
        "diag_a_in", "diag_c_in", "diag_a_f", "diag_c_f", "diag_b_in", "diag_b_f",
        "l3_pool", "l2_compressor", "memory_xattn",
        "l3_token_recon_head",
    )

    is_fsdp = _FSDP_AVAILABLE and FSDP is not None and (
        isinstance(model, FSDP) or getattr(model, "_uses_partial_fsdp", False)
    )

    # Full fine-tune (2026-06-18): the backbone is trainable, so the adapter-only
    # fragment filter would silently drop the fine-tuned Llama weights. When
    # --unfreeze_backbone, save the ENTIRE state dict (backbone + adapter) to a
    # full_model checkpoint instead.
    save_full_model = bool(getattr(args, "unfreeze_backbone", False))

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
            if save_full_model or any(frag in k for frag in fragments)
        }
    else:
        root = model.module if isinstance(model, DDP) else model
        state = {
            k: v.detach().cpu()
            for k, v in root.state_dict().items()
            if save_full_model or any(frag in k for frag in fragments)
        }

    if final:
        _stem = "full_model" if save_full_model else "mem_space_adapter"
        ckpt_path = os.path.join(args.output_dir, f"{_stem}.pt")
    else:
        _stem = "full_model" if save_full_model else "mem_space_adapter"
        ckpt_path = os.path.join(args.output_dir, f"{_stem}_step{step:06d}.pt")
    torch.save(state, ckpt_path)
    logger.info("Saved %s ckpt: %s (%d keys)",
                "full-model" if save_full_model else "adapter",
                ckpt_path, len(state))

    cfg_path = os.path.join(args.output_dir, "adapter_config.json")
    with open(cfg_path, "w") as f:
        json.dump({
            "num_slots": args.num_slots,
            "top_k": args.top_k,
            "selector_dim": args.selector_dim,
            "slot_dim": args.slot_dim,
            "writeback_mode": args.writeback_mode,
            "lowrank_gate_rank": args.lowrank_gate_rank,
            "writeback_gate_max": args.writeback_gate_max,
            "writeback_warmup_steps": args.writeback_warmup_steps,
            "load_balance_weight": args.load_balance_weight,
            "entropy_aux_weight": args.entropy_aux_weight,
            "use_loss_free_balance": args.use_loss_free_balance,
            "loss_free_update_rate": args.loss_free_update_rate,
            "use_st_gumbel_topk": args.use_st_gumbel_topk,
            "st_gumbel_temperature": args.st_gumbel_temperature,
            "use_delta_rule_writeback": args.use_delta_rule_writeback,
            "normalize_readout": args.normalize_readout,
            "readout_norm_scale": args.readout_norm_scale,
            "independent_slot_key": args.independent_slot_key,
            "delta_erase_write": args.delta_erase_write,
            "use_l2": args.use_l2,
            "l2_compress_ratio": args.l2_compress_ratio,
            "l2_d_c": args.l2_d_c,
            "l2_d_h_rope": args.l2_d_h_rope,
            "l2_init_scale": args.l2_init_scale,
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
            "use_slot_evidence": args.use_slot_evidence,
            "evidence_buffer_size": args.evidence_buffer_size,
            "evidence_topr": args.evidence_topr,
            "evidence_layer": args.evidence_layer,
            "evidence_isolate_softmax": args.evidence_isolate_softmax,
            "l3_recon_token_weight": args.l3_recon_token_weight,
            "disable_l1_inject": args.disable_l1_inject,
            "use_replace_writeback": args.use_replace_writeback,
            # P2/P8 read-path flags. These add/remove module params, so they MUST
            # round-trip through adapter_config.json or eval reconstructs a model
            # whose state_dict mismatches the checkpoint.
            "use_decoupled_read": args.use_decoupled_read,
            "use_memory_xattn": args.use_memory_xattn,
            "memory_xattn_gate_init": args.memory_xattn_gate_init,
            "memory_xattn_disable_null_sink": args.memory_xattn_disable_null_sink,
            "use_readout_mass_bias": args.use_readout_mass_bias,
            "readout_mass_coef": args.readout_mass_coef,
            # EXP-R1 (2026-06-11): dead-slot recycling. No new params, but it
            # changes STORED slot state, so eval-haystack ingestion must apply
            # the same recycling as training → round-trip these flags.
            "dead_slot_reset_interval": args.dead_slot_reset_interval,
            "dead_slot_reset_mode": args.dead_slot_reset_mode,
            "dead_slot_grace_chunks": args.dead_slot_grace_chunks,
            "dead_slot_criterion": args.dead_slot_criterion,
            # EXP-W2 (2026-06-11): dense all-slot soft delta-write. Changes
            # STORED slot state, so eval-haystack ingestion must apply the same
            # soft-write as training → round-trip these flags.
            "soft_write_weight": args.soft_write_weight,
            "soft_write_content": args.soft_write_content,
            # v20 (2026-06-12): read-based slot lifecycle. Changes STORED slot
            # state, so eval-haystack ingestion must apply the same lifecycle as
            # training → round-trip these flags.
            "slot_read_decay_rate": args.slot_read_decay_rate,
            "slot_read_decay_interval": args.slot_read_decay_interval,
            "slot_read_decay_min_keep": args.slot_read_decay_min_keep,
            "slot_evict_mode": args.slot_evict_mode,
            "slot_evict_max_frac": args.slot_evict_max_frac,
            "slot_evict_floor": args.slot_evict_floor,
            "slot_evict_protect_chunks": args.slot_evict_protect_chunks,
            "num_global_slots": args.num_global_slots,
            "global_slot_forget_bias": args.global_slot_forget_bias,
            "global_slot_input_gate_only": args.global_slot_input_gate_only,
            "gradient_checkpointing": args.gradient_checkpointing,
            # In-attention K/V injection (round-trip so eval rebuilds the same
            # injection layers; multi-layer v2 stores the explicit layer list).
            "use_inattn_kv": args.use_inattn_kv,
            "inattn_kv_layer": args.inattn_kv_layer,
            "inattn_kv_topk": args.inattn_kv_topk,
            "inattn_kv_layers": args.inattn_kv_layers,
            # Raw-KV readout (Method A) — round-trip so eval rebuilds the same
            # gist scorer + readout layers (adds the shared gist_readout module).
            "use_rawkv_readout": args.use_rawkv_readout,
            "rawkv_readout_layer": args.rawkv_readout_layer,
            "rawkv_readout_layers": args.rawkv_readout_layers,
            "rawkv_gist_dim": args.rawkv_gist_dim,
            "rawkv_readout_topk_chunks": args.rawkv_readout_topk_chunks,
            "rawkv_readout_temp": args.rawkv_readout_temp,
            # Partial-unfreeze metadata (v2): records which layers were trainable.
            "unfreeze_backbone": args.unfreeze_backbone,
            "unfreeze_layers_from": args.unfreeze_layers_from,
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
    # Independent T2 recall curriculum (optional). When set, only the T2 needle
    # distance grows with training; dolmino n_ctx stays on its own (small, fixed)
    # schedule so the 4096-token-capped per-doc loader never starves.
    t2_curriculum = (
        CurriculumScheduler(args.t2_curriculum) if args.t2_curriculum else None
    )

    # batch_size > 1 guards (2026-06-07). The memory bank holds per-sample slot
    # state [B, N, slot_dim] and is fully batch-native, BUT two features cannot
    # be combined with a batch dim:
    #   * --doc_reset attaches per-sample reset_flags; the bank cannot reset a
    #     single batch element mid-rollout (reset is whole-bank).
    #   * --num_workers > 1 with an IterableDataset + mid-epoch set_n_context
    #     curriculum updates would not propagate to worker processes; this is a
    #     pre-existing constraint, but with batched collate a stale-n_ctx worker
    #     would produce a batch the collate rejects. Keep workers at 0/1.
    if args.batch_size > 1:
        if args.doc_reset:
            raise ValueError(
                "--doc_reset is incompatible with --batch_size > 1 (the memory "
                "bank cannot reset a single batch element mid-rollout). Use "
                "--batch_size 1, or --per_doc_data / plain contiguous without "
                "doc_reset for batched training."
            )
        if args.num_workers > 1:
            raise ValueError(
                "--batch_size > 1 requires --num_workers <= 1 so mid-epoch "
                "curriculum set_n_context() updates reach the dataset iterator "
                "(IterableDataset workers do not see them)."
            )

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
    _freeze_backbone(
        model,
        unfreeze_backbone=args.unfreeze_backbone,
        unfreeze_layers_from=args.unfreeze_layers_from,
    )

    # Arm the one-time in-graph probe on the inattn layer(s) (no-op otherwise).
    if args.grad_flow_diag:
        for _w in getattr(model, "_mem_space_layers", []) or []:
            _w._inattn_grad_probe = True

    if is_main(rank):
        if args.unfreeze_backbone:
            n_total = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_mem = sum(p.numel() for p in _mem_space_params(model))
            logger.info(
                "FULL FINE-TUNE: %.2fM trainable params (backbone+adapter); "
                "of which mem_space adapter = %.2fM",
                n_total / 1e6, n_mem / 1e6,
            )
        else:
            n_trainable = sum(p.numel() for p in _mem_space_params(model))
            logger.info("mem_space trainable: %.2fM params", n_trainable / 1e6)

    # Distributed wrap: FSDP (shard optimizer state) or DDP (replicate).
    if world_size > 1:
        if args.use_fsdp:
            if not _FSDP_AVAILABLE:
                raise RuntimeError(
                    "--use_fsdp set but torch.distributed.fsdp is unavailable."
                )
            model = _wrap_model_fsdp(
                model,
                local_rank=local_rank,
                use_checkpoint_wrapper=args.gradient_checkpointing,
                unfreeze_backbone=args.unfreeze_backbone,
            )
            if is_main(rank):
                logger.info("Using FSDP (Option b): trainable mem_space sharded, "
                            "backbone replicated. use_ckpt_wrapper=%s",
                            args.gradient_checkpointing)
        else:
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
        contiguous=args.contiguous_chunks,
        doc_reset=args.doc_reset,
        per_doc=args.per_doc_data,
    )
    dolmino_loader = DataLoader(
        dolmino_ds,
        batch_size=(args.batch_size if args.batch_size > 1 else None),
        collate_fn=(dolmino_collate_fn if args.batch_size > 1 else None),
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

    # --- T2 chunked associative-recall dataset (for mix) --- #
    t2_iter = None
    t2_loader = None
    if args.t2_recall_mix_fraction > 0.0:
        import numpy as _np
        if not os.path.isfile(args.t2_background_data):
            raise FileNotFoundError(
                f"--t2_background_data not found: {args.t2_background_data}"
            )
        t2_bg = _np.load(args.t2_background_data, mmap_mode="r")  # [N, L]
        t2_ds = NIAHChunkedDataset(
            background_data=t2_bg,
            chunk_size=args.chunk_size,
            gap_tokens=args.t2_gap_tokens,
            tokenizer=tokenizer,
            num_keys=args.t2_num_keys,
            seed=args.seed + rank,
            background_skip=args.t2_background_skip,
        )
        if is_main(rank):
            logger.info(
                "T2 recall mix: frac=%.2f num_keys=%d gap_tokens=%d "
                "chunk_size=%d -> n_ctx=%d bg=%s",
                args.t2_recall_mix_fraction, args.t2_num_keys,
                args.t2_gap_tokens, args.chunk_size, t2_ds.n_ctx,
                args.t2_background_data,
            )
        t2_loader = DataLoader(
            t2_ds,
            batch_size=(args.batch_size if args.batch_size > 1 else None),
            collate_fn=(niah_chunked_collate_fn if args.batch_size > 1 else None),
            num_workers=args.num_workers, pin_memory=False,
        )
        t2_iter = iter(t2_loader)

    mix_rng = random.Random(args.seed + rank)

    # --- optimizer + LR scheduler --- #
    if args.use_fsdp:
        # CRITICAL (ported fix from babilong, 2026-05-17): under partial FSDP
        # with use_orig_params=True, the attribute-walked Parameter handles
        # produced by _mem_space_params() do NOT receive gradient writeback —
        # FSDP attaches grads to its internal FlatParameter, not to those
        # attribute-accessed handles. In babilong this silently froze 75% of
        # trainable params (grad=None → checkpoint stored init values → eval
        # crash). With use_orig_params=True, model.parameters() yields the
        # original Parameter handles FSDP writes grads to, so walk those.
        trainable = [p for p in model.parameters() if p.requires_grad]
    elif args.unfreeze_backbone:
        # Full fine-tune (DDP / single-GPU): every requires_grad param trains —
        # the whole backbone + the mem_space adapter. _mem_space_params() would
        # collect ONLY the adapter, silently freezing the backbone, so walk
        # model.parameters() instead.
        trainable = [p for p in model.parameters() if p.requires_grad]
    else:
        trainable = _mem_space_params(
            model.module if isinstance(model, DDP) else model
        )
    if not trainable:
        raise RuntimeError("No mem_space trainable params found.")

    # Sanity: optimizer param count must match the number of trainable params
    # reported by named_parameters(). If FSDP drops or duplicates a handle,
    # fail loudly so we don't silently retrain a broken adapter again.
    n_optim = len(trainable)
    n_named_trainable = sum(
        1 for _, p in model.named_parameters() if p.requires_grad
    )
    if is_main(rank):
        logger.info(
            "Optimizer param-collection sanity: optim_params=%d  "
            "named_parameters(requires_grad)=%d  use_fsdp=%s",
            n_optim, n_named_trainable, bool(args.use_fsdp),
        )
    if n_optim != n_named_trainable:
        raise RuntimeError(
            f"Optimizer param mismatch: collected {n_optim} but "
            f"named_parameters() reports {n_named_trainable} trainable params. "
            "Refusing to start training (would silently freeze some adapter "
            "params under FSDP)."
        )

    optimizer = torch.optim.AdamW(trainable, lr=args.lr,
                                  weight_decay=args.weight_decay,
                                  betas=(0.9, 0.95))

    # --- training loop --- #
    model.train()
    global_step = args.start_step if hasattr(args, 'start_step') and args.start_step else 0
    micro_step = 0
    n_dolmino = 0
    n_babilong = 0
    n_t2 = 0
    n_nonfinite = 0
    # --- loss-spike skip state (DDP-consistent: keyed off the all_reduced
    # average lm_loss, identical on every rank) ---
    spike_skip_count = 0
    _spike_window = []  # sliding window of recent (post-warmup) avg lm_loss
    _spike_window_max = 100
    _spike_min_samples = 20
    accum_lm_loss = 0.0
    accum_aux_loss = 0.0
    t0 = time.time()
    grad_accum = args.gradient_accumulation_steps

    # Self-study distillation config (v21). Active ONLY when a cache dir is set
    # AND at least one distill flag is on. When inactive, distill_cfg stays None
    # and the dolmino step takes the byte-identical original path.
    distill_cfg = None
    if args.distill_cache_dir and (args.distill_logits or args.distill_hidden):
        _distill_layers_parsed = [int(x) for x in args.distill_layers.split(",")
                                  if x.strip() != ""]
        # --- consistency guard (v21): refuse to start if the teacher cache was
        # built with a different n_ctx/chunk_size/distill_layers than this run
        # uses (else the (doc_idx, group_pos) keys张冠李戴). See
        # assert_distill_cache_consistent for details. ---
        _train_fp = getattr(getattr(dolmino_ds, "_ds", None), "_fingerprint", None)
        _meta = assert_distill_cache_consistent(
            args.distill_cache_dir, args.chunk_size, curriculum.get_n_ctx(0),
            _distill_layers_parsed, train_fingerprint=_train_fp)
        if is_main(rank):
            logger.info(
                "[distill] cache meta OK: n_ctx=%d chunk_size=%d layers=%s "
                "fingerprint=%s (matches training config + dataset)",
                int(_meta["n_ctx"]), int(_meta["chunk_size"]),
                [int(x) for x in _meta["distill_layers"]],
                _meta.get("dataset_fingerprint"))
        distill_cfg = {
            "distill_logits": args.distill_logits,
            "distill_hidden": args.distill_hidden,
            "distill_lambda": args.distill_lambda,
            "distill_weight": args.distill_weight,
            "distill_hidden_beta": args.distill_hidden_beta,
            "distill_layers": _distill_layers_parsed,
        }
        if is_main(rank):
            logger.info("[distill] ENABLED cache_dir=%s logits=%s hidden=%s "
                        "lambda=%.2f weight=%.2f beta=%.2f layers=%s",
                        args.distill_cache_dir, args.distill_logits,
                        args.distill_hidden, args.distill_lambda,
                        args.distill_weight, args.distill_hidden_beta,
                        distill_cfg["distill_layers"])

    # Small LRU cache of decoded teacher .npz (keyed by sample_id) to avoid
    # re-reading hot samples; correctness does not depend on it.
    from collections import OrderedDict as _OrderedDict
    _distill_lru: "_OrderedDict[tuple, dict]" = _OrderedDict()
    _DISTILL_LRU_MAX = 64

    # distill cache hit/miss counters (fail-fast guard against silent 100% miss,
    # e.g. wrong dataset fingerprint or incomplete cache — see 2026-06-16 incident
    # where a whole 500-step run trained with distill_kl=0 because the cache was
    # keyed to a different Arrow row order). [hits, misses].
    _distill_hitmiss = [0, 0]

    def _get_teacher_cache(sample_id):
        """Return decoded teacher cache dict for sample_id, or None if missing."""
        if distill_cfg is None or sample_id is None:
            return None
        key = tuple(int(x) for x in sample_id)
        if key in _distill_lru:
            _distill_lru.move_to_end(key)
            _distill_hitmiss[0] += 1
            return _distill_lru[key]
        path = os.path.join(args.distill_cache_dir, f"{key[0]}_{key[1]}.npz")
        if not os.path.exists(path):
            _distill_hitmiss[1] += 1
            return None  # not cached → fall back to plain dolmino step
        data = load_distill_npz(path, device)
        _distill_lru[key] = data
        if len(_distill_lru) > _DISTILL_LRU_MAX:
            _distill_lru.popitem(last=False)
        _distill_hitmiss[0] += 1
        return data

    while global_step < args.total_steps:
        # Update curriculum
        current_n_ctx = curriculum.get_n_ctx(global_step)
        dolmino_ds.set_n_context(current_n_ctx)
        # T2 recall distance follows its OWN curriculum (--t2_curriculum), kept
        # independent of dolmino. The pg19 T2 background is long enough to grow
        # n_ctx to 32+, whereas dolmino per-doc data is capped at 4096 tokens, so
        # we must NOT push dolmino past a small n_ctx (else (n_ctx+1)*chunk_size
        # >4096 -> zero eligible docs -> loader starves -> DDP first-step hang).
        # If --t2_curriculum is unset, T2 keeps its fixed gap-derived n_ctx.
        # Requires num_workers<=1 (default 0) so the update reaches the iterator.
        if t2_iter is not None and t2_curriculum is not None:
            t2_ds.set_n_ctx(t2_curriculum.get_n_ctx(global_step))

        # Update learning rate (cosine with warmup)
        lr = cosine_lr_schedule(global_step, args.total_steps, args.warmup_steps,
                                args.lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        step_lm_loss = 0.0
        step_aux_loss = 0.0
        step_route_aux = 0.0
        step_l3recon = 0.0
        step_distill_kl = 0.0
        step_distill_hidden = 0.0
        step_valid_micros = 0

        for micro in range(grad_accum):
            # Always suppress DDP auto-sync: TBPTT does multiple backward()
            # calls per micro-step, and Dolmino vs BABILong have different
            # counts. We manually allreduce once after the accumulation loop.
            sync_ctx = model.no_sync() if (
                world_size > 1 and isinstance(model, DDP)
            ) else nullcontext()

            with sync_ctx:
                # Decide step type. BABILong checked first, then T2 recall, then
                # Dolmino. Each draws an independent Bernoulli so the fractions
                # compose (P(T2) is over the remaining mass after BABILong).
                use_babilong = (babilong_iter is not None and
                                mix_rng.random() < args.babilong_mix_fraction)
                use_t2 = (not use_babilong and t2_iter is not None and
                          mix_rng.random() < args.t2_recall_mix_fraction)

                if use_babilong:
                    # BABILong step
                    try:
                        batch = next(babilong_iter)
                    except StopIteration:
                        babilong_iter = iter(babilong_loader)
                        batch = next(babilong_iter)

                    input_ids = batch["input_ids"].to(device, non_blocking=True)
                    labels = batch["labels"].to(device, non_blocking=True)

                    lm_loss, aux_loss, route_aux, l3recon = babilong_train_step(
                        model, input_ids, labels, args.chunk_size, device,
                        route_aux_weight=args.route_aux_weight,
                        l3_recon_token_weight=args.l3_recon_token_weight,
                    )
                    micro_distill_kl = 0.0
                    micro_distill_hidden = 0.0
                    n_babilong += 1
                elif use_t2:
                    # T2 chunked associative-recall step: always the HARDER
                    # objective (dolmino_train_step) with an answer_mask so LM
                    # loss falls ONLY on the answer digits. backward() is done
                    # inside dolmino_train_step (no outer backward needed).
                    # T2 is NEVER distilled (technical point 3): teacher cache and
                    # distill_cfg are both omitted → original path.
                    try:
                        sample = next(t2_iter)
                    except StopIteration:
                        t2_iter = iter(t2_loader)
                        sample = next(t2_iter)

                    context_chunks = sample["context_chunks"]
                    target_ids = sample["target_ids"]
                    answer_mask = sample["answer_mask"]

                    (lm_loss, aux_loss, route_aux, l3recon,
                     _dk, _dh) = dolmino_train_step(
                        model, context_chunks, target_ids, device,
                        grad_accum=grad_accum, answer_mask=answer_mask,
                    )
                    micro_distill_kl = 0.0
                    micro_distill_hidden = 0.0
                    n_t2 += 1
                else:
                    # Dolmino step
                    try:
                        sample = next(dolmino_iter)
                    except StopIteration:
                        dolmino_iter = iter(dolmino_loader)
                        sample = next(dolmino_iter)

                    context_chunks = sample["context_chunks"]
                    target_ids = sample["target_ids"]
                    reset_flags = sample.get("reset_flags", None)

                    # Distillation only applies to the last_chunk_loss_only
                    # (dolmino_train_step) path; tbptt path is unchanged.
                    teacher_cache = None
                    if args.last_chunk_loss_only and distill_cfg is not None:
                        teacher_cache = _get_teacher_cache(sample.get("sample_id"))

                    if args.last_chunk_loss_only:
                        (lm_loss, aux_loss, route_aux, l3recon,
                         micro_distill_kl, micro_distill_hidden) = dolmino_train_step(
                            model, context_chunks, target_ids, device,
                            grad_accum=grad_accum,
                            teacher_cache=teacher_cache,
                            distill_cfg=(distill_cfg if teacher_cache is not None
                                         else None),
                        )
                    else:
                        lm_loss, aux_loss, route_aux, l3recon = (
                            dolmino_train_step_tbptt(
                                model, context_chunks, target_ids, device,
                                grad_accum=grad_accum,
                                route_aux_weight=args.route_aux_weight,
                                reset_flags=reset_flags,
                                l3_recon_token_weight=args.l3_recon_token_weight,
                                swa_train_chunks=args.swa_train_chunks,
                            )
                        )
                        micro_distill_kl = 0.0
                        micro_distill_hidden = 0.0
                    n_dolmino += 1

                # Check for non-finite
                if lm_loss is None or not torch.isfinite(lm_loss + aux_loss):
                    n_nonfinite += 1
                    _zero = torch.zeros(1, device=device, requires_grad=False)
                    _zero = _zero + 0.0 * next(p for p in model.parameters() if p.requires_grad)
                    _zero.backward()
                    continue

                # TBPTT (dolmino) already called backward() inside, route_aux +
                # l3recon included; for BABILong we still need the outer backward
                # and must fold route_aux + l3recon into the loss here.
                if use_babilong:
                    loss = (lm_loss + aux_loss
                            + args.route_aux_weight * route_aux
                            + args.l3_recon_token_weight * l3recon) / grad_accum
                    loss.backward()

                step_lm_loss += lm_loss.item()
                step_aux_loss += aux_loss.item()
                step_route_aux += float(route_aux.item()) if isinstance(
                    route_aux, torch.Tensor) else float(route_aux)
                step_l3recon += float(l3recon.item()) if isinstance(
                    l3recon, torch.Tensor) else float(l3recon)
                step_distill_kl += float(micro_distill_kl.item()) if isinstance(
                    micro_distill_kl, torch.Tensor) else float(micro_distill_kl)
                step_distill_hidden += (
                    float(micro_distill_hidden.item())
                    if isinstance(micro_distill_hidden, torch.Tensor)
                    else float(micro_distill_hidden))
                step_valid_micros += 1

        # Manual gradient allreduce (since we always use no_sync above).
        # CRITICAL: every rank MUST issue the SAME sequence of all_reduce
        # collectives (same order, same shapes). Previously this was guarded by
        # `if p.grad is not None` AND `step_valid_micros > 0`, both of which are
        # per-rank conditions: when a slot/memory param got no gradient on one
        # rank (e.g. uniq_sel_slots=0) that rank skipped its all_reduce while
        # other ranks issued it -> collective size mismatch (262144 vs 16777216)
        # -> deterministic NCCL hang at step ~490-493 -> watchdog SIGABRT.
        # Fix: iterate the FULL trainable list unconditionally on every rank,
        # zero-filling any missing grad so the collective stays in lockstep.
        if world_size > 1 and isinstance(model, DDP):
            for p in trainable:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)

        # --- One-time grad-flow diagnostic (--grad_flow_diag) --- #
        # Runs once, on the first step that produced gradients, then disables
        # itself. Proves (a) backbone params receive non-zero gradient under
        # --unfreeze_backbone, and (b) the injected in-attention K_raw is in-graph
        # (its retained .grad is non-None). For the unfreeze/in-graph smoke only.
        if args.grad_flow_diag and step_valid_micros > 0 and is_main(rank):
            _root = model.module if isinstance(model, DDP) else model
            # (a) backbone grad norms: sample a few decoder-layer weight params
            #     that are NOT part of the mem_space adapter (i.e. the wrapped
            #     frozen Llama layer's q/k/v/mlp). Under --unfreeze_backbone these
            #     must have non-None, non-zero grads.
            _bb_lines = []
            _n_bb = 0
            for _n, _p in _root.named_parameters():
                if not _p.requires_grad:
                    continue
                if any(k in _n for k in ("self_attn.q_proj", "self_attn.k_proj",
                                         "mlp.gate_proj", "mlp.down_proj")):
                    _gn = (_p.grad.norm().item() if _p.grad is not None else None)
                    _bb_lines.append(f"    {_n}: grad_norm={_gn}")
                    _n_bb += 1
                    if _n_bb >= 6:
                        break
            logger.info("[grad_flow_diag] backbone trainable sample grad norms "
                        "(unfreeze_backbone=%s):\n%s",
                        args.unfreeze_backbone,
                        "\n".join(_bb_lines) if _bb_lines
                        else "    (no backbone params requires_grad)")
            # (b) injected in-attention K_raw in-graph check.
            _inattn_ok = "n/a (use_inattn_kv off or no injection this step)"
            for _w in getattr(_root, "_mem_space_layers", []) or []:
                _kr = getattr(_w, "_last_inattn_K_raw", None)
                if _kr is not None:
                    _g = _kr.grad
                    _inattn_ok = (
                        f"K_raw.requires_grad={_kr.requires_grad} "
                        f"grad={'None' if _g is None else f'norm={_g.norm().item():.3e}'} "
                        f"(R={_kr.shape[2]})"
                    )
                    break
            logger.info("[grad_flow_diag] injected in-attention KV in-graph: %s",
                        _inattn_ok)
            args.grad_flow_diag = False  # one-shot

        # --- Loss-spike skip decision (DDP-consistent) ---
        # Compute a single avg lm_loss scalar that is IDENTICAL on every rank by
        # all_reducing the per-rank average. The skip decision is derived purely
        # from this shared scalar + a window that only ever ingests this shared
        # scalar, so all ranks decide identically -> no collective desync.
        # The whole block is a no-op (and issues no extra collectives) unless
        # --loss_spike_skip is set, preserving prior behavior by default.
        do_skip = False
        if args.loss_spike_skip:
            _avg_lm_local = step_lm_loss / max(1, step_valid_micros)
            if world_size > 1 and dist.is_initialized():
                _t = torch.tensor([_avg_lm_local], device=device)
                dist.all_reduce(_t, op=dist.ReduceOp.AVG)
                avg_lm_shared = float(_t.item())
            else:
                avg_lm_shared = _avg_lm_local

            if (step_valid_micros > 0
                    and global_step >= args.warmup_steps
                    and len(_spike_window) >= _spike_min_samples
                    and math.isfinite(avg_lm_shared)):
                _mean = sum(_spike_window) / len(_spike_window)
                _var = sum((x - _mean) ** 2 for x in _spike_window) / len(_spike_window)
                _std = math.sqrt(max(_var, 0.0))
                if avg_lm_shared > _mean + args.loss_spike_sigma * _std:
                    do_skip = True

        # Optimizer step (only if we got at least one valid micro-step and the
        # step is not flagged as a loss spike).
        if step_valid_micros > 0 and not do_skip:
            # Per-projection grad clip
            _grad_root = model.module if isinstance(model, DDP) else model
            for _n, _p in _grad_root.named_parameters():
                if _p.grad is not None and ("slot_to_hidden" in _n or "hidden_to_slot" in _n):
                    torch.nn.utils.clip_grad_norm_([_p], args.proj_grad_clip)
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)

            optimizer.step()

        if do_skip:
            spike_skip_count += 1
            optimizer.zero_grad(set_to_none=True)
            if is_main(rank):
                logger.warning(
                    "[loss_spike_skip] step=%d avg_lm=%.4f exceeded "
                    "mean+%.1f*std (n_skipped=%d) -> skipping optimizer.step()",
                    global_step, avg_lm_shared, args.loss_spike_sigma,
                    spike_skip_count,
                )

        # Update the spike baseline window AFTER the decision so the current
        # step is judged against prior history only. Only ingest finite,
        # post-warmup, non-skipped steps to keep the baseline clean.
        if (args.loss_spike_skip and step_valid_micros > 0
                and global_step >= args.warmup_steps
                and math.isfinite(avg_lm_shared) and not do_skip):
            _spike_window.append(avg_lm_shared)
            if len(_spike_window) > _spike_window_max:
                _spike_window.pop(0)

        _step_counters_inc(model)
        global_step += 1

        # Fail-fast: distill enabled but cache 100% missing after warmup (2026-06-16).
        # Catches silent distillation no-op (wrong dataset fingerprint / incomplete
        # cache) before wasting a whole run. Checked once at step 50.
        if distill_cfg is not None and global_step == 50:
            _h, _m = _distill_hitmiss
            if _h == 0 and _m > 0:
                raise RuntimeError(
                    f"[distill] cache 100% MISS after 50 steps ({_m} dolmino "
                    "lookups, 0 hits) -> distillation is a silent no-op. Likely "
                    "wrong dataset fingerprint (cache built from a different Arrow "
                    "row order) or an incomplete/empty cache dir. Check "
                    f"{args.distill_cache_dir}/meta.json dataset_fingerprint vs the "
                    "training dataset, and that the cache finished building.")
            if is_main(rank):
                logger.info("[distill] cache hit-rate @step50: %d hits / %d miss",
                            _h, _m)

        # Logging
        if is_main(rank) and (global_step % args.log_interval == 0):
            avg_lm = step_lm_loss / max(1, step_valid_micros)
            avg_aux = step_aux_loss / max(1, step_valid_micros)
            avg_route_aux = step_route_aux / max(1, step_valid_micros)
            avg_l3recon = step_l3recon / max(1, step_valid_micros)
            avg_distill_kl = step_distill_kl / max(1, step_valid_micros)
            avg_distill_hidden = step_distill_hidden / max(1, step_valid_micros)
            elapsed = time.time() - t0
            steps_per_sec = global_step / elapsed if elapsed > 0 else 0.0
            logger.info(
                "[step %d/%d] lm=%.4f aux=%.4f route_aux=%.4f l3recon=%.4f "
                "distill_kl=%.4f distill_hid=%.4f lr=%.2e n_ctx=%d "
                "dolmino=%d babi=%d nf=%d skip=%d speed=%.2f steps/s",
                global_step, args.total_steps, avg_lm, avg_aux, avg_route_aux,
                avg_l3recon, avg_distill_kl, avg_distill_hidden, lr,
                current_n_ctx, n_dolmino, n_babilong, n_nonfinite, spike_skip_count,
                steps_per_sec,
            )
            _xattn_diag = _collect_xattn_diag(model)
            if _xattn_diag:
                logger.info(
                    "[XATTN_DIAG step=%d] sink_mass=%.4f gate_mean=%.4f attn_entropy=%.4f",
                    global_step,
                    _xattn_diag["memory/xattn_sink_mass"],
                    _xattn_diag["memory/xattn_gate_mean"],
                    _xattn_diag["memory/xattn_attn_entropy"],
                )
            # Slot-Routed Evidence activity probe: confirm the evidence buffer is
            # actually populated DURING training (not silently inactive), so the
            # decoder read pathway is being exercised. Reads the shared bank's
            # per-slot evidence count populated by write_evidence each chunk.
            if getattr(args, "use_slot_evidence", False):
                _root = getattr(model, "module", model)
                _bank = getattr(_root, "_mem_space_shared_bank", None)
                _ev_cnt = getattr(_bank, "slot_evidence_count", None) if _bank else None
                if _ev_cnt is not None:
                    logger.info(
                        "[EVIDENCE_DIAG step=%d] alloc=True max_count=%d "
                        "mean_count=%.2f nonempty_slots=%d",
                        global_step,
                        int(_ev_cnt.max().item()),
                        float(_ev_cnt.float().mean().item()),
                        int((_ev_cnt > 0).sum().item()),
                    )
                else:
                    logger.info(
                        "[EVIDENCE_DIAG step=%d] alloc=False (evidence buffer NOT "
                        "populated — write path may be inactive!)", global_step,
                    )
            if getattr(args, "dead_slot_reset_interval", 0) and args.dead_slot_reset_interval > 0:
                _mem_diag = _collect_mem_diag(model)
                logger.info(
                    "[DEADSLOT_DIAG step=%d] criterion=%s dead_slot_frac=%.4f usage_cov=%.4f "
                    "max_slot_select_count=%.1f recycle_resets=%.0f n_recycled=%.0f",
                    global_step,
                    getattr(args, "dead_slot_criterion", "window"),
                    _mem_diag.get("memory/dead_slot_frac", 0.0),
                    _mem_diag.get("memory/usage_cov", 0.0),
                    _mem_diag.get("memory/max_slot_select_count", 0.0),
                    _mem_diag.get("memory/recycle_resets", 0.0),
                    _mem_diag.get("memory/n_recycled", 0.0),
                )
            if _WANDB_AVAILABLE and args.wandb_project and wandb.run:
                _log_dict = {
                    "train/lm_loss": avg_lm,
                    "train/aux_loss": avg_aux,
                    "train/route_aux": avg_route_aux,
                    "train/l3recon": avg_l3recon,
                    "train/distill_kl": avg_distill_kl,
                    "train/distill_hidden": avg_distill_hidden,
                    "train/lr": lr,
                    "train/n_ctx": current_n_ctx,
                    "train/speed_steps_s": steps_per_sec,
                    "train/n_nonfinite": n_nonfinite,
                    "train/dolmino_count": n_dolmino,
                    "train/babilong_count": n_babilong,
                    "train/t2_count": n_t2,
                    "memory/top1_sim": _collect_top1_sim(model),
                }
                _log_dict.update(_collect_mem_diag(model))
                _log_dict.update(_xattn_diag)
                wandb.log(_log_dict, step=global_step)

        # Save checkpoint.
        # FSDP state_dict gather is a collective: ALL ranks must enter
        # _save_adapter (which gathers FULL_STATE_DICT then writes only on
        # rank 0). For DDP, only rank 0 needs to call _save_adapter.
        if (args.save_interval > 0
                and global_step % args.save_interval == 0
                and global_step < args.total_steps):
            if args.use_fsdp:
                if is_main(rank):
                    logger.info("[save] start adapter save at step %d", global_step)
                _t_save = time.time()
                _save_adapter(model, args, global_step)
                if is_main(rank):
                    logger.info("[save] done adapter save at step %d (%.1fs)",
                                global_step, time.time() - _t_save)
            elif is_main(rank):
                logger.info("[save] start adapter save at step %d", global_step)
                _t_save = time.time()
                _save_adapter(model, args, global_step)
                logger.info("[save] done adapter save at step %d (%.1fs)",
                            global_step, time.time() - _t_save)
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

    # Final save. Under FSDP all ranks must enter _save_adapter (collective
    # state_dict gather); only rank 0 writes. Under DDP only rank 0 calls it.
    if args.use_fsdp:
        _save_adapter(model, args, global_step, final=True)
    elif is_main(rank):
        _save_adapter(model, args, global_step, final=True)
    if is_main(rank):
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
