#!/usr/bin/env python3
"""Probe #2 (minimal-architecture hypothesis) ported to the **Hunyuan-A13B MoE**.

This is the large-MoE sibling of scripts/train_qwen3_arch_probe2.py. Same idea
(Direction 4 / QCMem §3.1 "understand-then-generate division of labour"): keep the
FRONT `keep_front_layers` decoder layers of a pretrained model (semantic
understanding is believed to saturate there), DROP the top layers, append
`n_fresh_layers` FRESH decoder layers to re-learn next-token prediction, and
continue-train.

Base model: Hunyuan-A13B-Pretrain (tencent/Hunyuan-A13B-Pretrain).
  * 80B total / ~13B active, fine-grained MoE.
  * config: model_type=hunyuan (== the native transformers `hunyuan_v1_moe`),
    num_hidden_layers=32, hidden_size=4096, num_experts=64, moe_topk=8,
    1 shared expert/layer, intermediate/moe_intermediate=3072, GQA 32/8 heads,
    head_dim 128, use_qk_norm=True, tie_word_embeddings=True, vocab_size=128167.

Differences vs the Qwen3-8B port (all handled below):
  1. **MoE, not dense**: each decoder layer's `mlp` is a HunYuanMoEV1Moe (gate +
     64 experts + 1 shared MLP). Fresh tail layers get a correctly-initialised
     MoE (post_init -> _init_weights fills experts.gate_up_proj/down_proj with
     N(0, initializer_range=0.02)). We NEVER hand-build a decoder layer.
  2. **tie_word_embeddings=True**: there is NO separate lm_head weight on disk.
     lm_head.weight IS embed_tokens.weight (tied). Transplanting the pretrained
     embed therefore also fixes the (tied) lm_head. `embed`/`lm_head` are treated
     as INHERITED (low LR), unlike Qwen3 where lm_head was fresh.
  3. **native-config gotchas** (verified 2026-07-12 on the local checkpoint):
       * `HunYuanMoEV1Config.from_pretrained` leaves `head_dim=None` (the config
         only carries `attention_head_dim=128`) -> the native attention does
         `self.head_dim ** -0.5` and crashes. We set cfg.head_dim explicitly.
       * per-layer list fields (moe_topk / num_shared_expert / moe_intermediate_size)
         are length-32 and are NOT recomputed from num_hidden_layers; we truncate
         them to keep+fresh (analogous to Qwen3's layer_types reset).
       * model_type stays "hunyuan"; we set it to "hunyuan_v1_moe" so the native
         class is the one instantiated (avoids the trust_remote_code path).
  4. **80B is too big for DDP** (no full replica fits a single GPU). We use FSDP
     FULL_SHARD (params+grads+optim sharded) with fp32 master weights +
     MixedPrecision(bf16 compute / fp32 reduce). The pruned keep24+fresh2 model
     is ~65B params. Only rank 0 materialises the transplanted weights on CPU;
     other ranks build on meta device and receive weights via
     sync_module_states=True (avoids world_size x 65B host-RAM blow-up).

Data / tokenizer -- IMPORTANT (embedding out-of-range trap):
  Hunyuan uses its own tiktoken-based tokenizer (HYTokenizer, vocab 128167). The
  Qwen3/Llama3 slimpajama .npy files hold INCOMPATIBLE token ids (Qwen3 ~151k,
  Llama3 128256) and will index out of Hunyuan's 128167-row embedding. You MUST
  train on a Hunyuan-tokenized corpus (see the module docstring in
  scripts/preprocess_slimpajama.py; run it with --tokenizer <hunyuan_path>
  --trust_remote_code and an explicit EOS id 127960). --data_path must point at a
  Hunyuan-tokenized (N, seq_len) uint32 npy. As a guard, we assert
  data.max() < vocab_size at startup.

fp32 MASTER WEIGHTS: params stay fp32 (do NOT model.to(bf16)); FSDP MixedPrecision
casts to bf16 for compute and reduces grads in fp32; AdamW states are fp32. Single
most important anti-catastrophic-forgetting knob for finetuning a pretrained stack.

Shares data loading / cosine schedule / null-context with
scripts/train_semantic_bottleneck_1b.py (imported, NOT modified). Does NOT touch
scripts/train_qwen3_arch_probe2.py.
"""
from __future__ import annotations

import argparse
import functools
import gc
import json
import logging
import math
import os
import re
import sys
import time
from datetime import timedelta

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from transformers.models.hunyuan_v1_moe.modeling_hunyuan_v1_moe import (
    HunYuanMoEV1Config,
    HunYuanMoEV1DecoderLayer,
    HunYuanMoEV1ForCausalLM,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Reuse (do NOT modify) the sibling Llama training script's data + schedule utils.
from train_semantic_bottleneck_1b import (  # noqa: E402
    NpyChunkDataset,
    collate_fn,
    get_lr,
    _nullctx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = "/apdcephfs_wzc1/share_304376610/pighzliu_code/models/Hunyuan-A13B-Pretrain"


# ---------------------------------------------------------------------------
# config construction
# ---------------------------------------------------------------------------
def build_pruned_config(model_path, keep_front_layers, n_fresh_layers):
    """Load the Hunyuan config, shrink it to keep+fresh layers, and patch the
    native-config gotchas (head_dim, per-layer list fields, model_type).

    Returns (cfg, total_layers)."""
    cfg = HunYuanMoEV1Config.from_pretrained(model_path, local_files_only=True)
    # (i) select the native class (not the trust_remote_code auto_map path).
    cfg.model_type = "hunyuan_v1_moe"
    # (ii) native attention/rotary read config.head_dim (None here); the checkpoint
    #      only carries attention_head_dim. Set it or attn crashes on head_dim**-0.5.
    if getattr(cfg, "head_dim", None) is None:
        cfg.head_dim = getattr(cfg, "attention_head_dim", None) or (
            cfg.hidden_size // cfg.num_attention_heads
        )
    total_layers = keep_front_layers + n_fresh_layers
    cfg.num_hidden_layers = total_layers
    # (iii) per-layer list fields are length-32 and NOT recomputed; truncate.
    for field in ("moe_topk", "num_shared_expert", "moe_intermediate_size"):
        v = getattr(cfg, field, None)
        if isinstance(v, (list, tuple)):
            setattr(cfg, field, list(v)[:total_layers])
            assert len(getattr(cfg, field)) == total_layers
    cfg.use_cache = False
    return cfg, total_layers


# ---------------------------------------------------------------------------
# model construction / transplant
# ---------------------------------------------------------------------------
def _layer_id(key):
    m = re.search(r"(?:^|\.)layers\.(\d+)\.", key)
    return int(m.group(1)) if m else None


def _copied_keys(state_dict, keep_front_layers):
    """Base-ckpt keys we transplant: all non-layer keys (embed / final-norm and,
    if present, the tied lm_head) + the front decoder layers layers.0..keep-1.
    Top layers dropped."""
    keys = []
    for k in state_dict:
        lid = _layer_id(k)
        if k.startswith("model.layers."):
            if lid is not None and lid < keep_front_layers:
                keys.append(k)
        else:
            keys.append(k)
    return keys


def _fresh_init_stats(model, keep_front_layers):
    """Fresh tail layers must retain proper Hunyuan init after transplant:
      * input_layernorm.weight all-ones (RMSNorm),
      * self_attn.q_proj.weight std ~ initializer_range (0.02),
      * mlp.experts.gate_up_proj std ~ initializer_range (0.02)  [MoE-specific].
    Returns a dict; asserts on failure."""
    sd = model.state_dict()
    p = f"model.layers.{keep_front_layers}."
    ln = sd[p + "input_layernorm.weight"]
    ln_all_ones = bool(torch.all(ln == 1.0).item())
    q_std = sd[p + "self_attn.q_proj.weight"].float().std().item()
    gup_std = sd[p + "mlp.experts.gate_up_proj"].float().std().item()
    assert ln_all_ones, (
        f"fresh layer {keep_front_layers} input_layernorm not all-ones "
        f"(min={ln.min().item()}, max={ln.max().item()})"
    )
    assert 0.005 < q_std < 0.05, (
        f"fresh layer {keep_front_layers} q_proj std={q_std:.4f} not ~0.02 -> wrong init"
    )
    assert 0.005 < gup_std < 0.05, (
        f"fresh layer {keep_front_layers} experts.gate_up_proj std={gup_std:.4f} "
        f"not ~0.02 -> wrong MoE init"
    )
    return {
        "fresh_input_layernorm_all_ones": ln_all_ones,
        "fresh_q_proj_std": q_std,
        "fresh_experts_gate_up_std": gup_std,
    }


def transplant_front(model, model_path, keep_front_layers, n_fresh_layers,
                     base_load_dtype, is_main):
    """Load the pretrained Hunyuan-A13B (native class, fused-expert layout via the
    from_pretrained auto-conversion) and transplant front keep_front layers +
    embed + final norm (+ tied lm_head) into `model`. Fresh tail layers keep their
    Hunyuan random init. Runs the sanity asserts. Raises on any mismatch.

    `base_load_dtype` (default bf16 == the checkpoint's native dtype) keeps host
    RAM lower; copying into an fp32 model is a lossless upcast, so the transplant
    max|diff| is still exactly 0.  Returns a sanity dict."""
    full_cfg, _ = build_pruned_config(model_path, 32, 0)  # full 32-layer native cfg
    if is_main:
        logger.info(f"[transplant] loading full base ({base_load_dtype}) from {model_path} ...")
    base = HunYuanMoEV1ForCausalLM.from_pretrained(
        model_path, config=full_cfg, torch_dtype=base_load_dtype,
        low_cpu_mem_usage=True, local_files_only=True,
    )
    base_sd = base.state_dict()

    keep_keys = _copied_keys(base_sd, keep_front_layers)
    filtered = {k: base_sd[k] for k in keep_keys}
    missing, unexpected = model.load_state_dict(filtered, strict=False)
    # re-tie (embed just got overwritten; tied lm_head must point at the new tensor)
    model.tie_weights()

    tie = bool(getattr(model.config, "tie_word_embeddings", False))

    # assert 1: no unexpected keys
    assert unexpected == [], f"[sanity1] unexpected keys: {unexpected[:8]}"

    # assert 2: the only missing keys are the fresh tail layers (+ tied lm_head,
    #           which shares embed and is therefore fine).
    missing_layer_ids, bad_missing = set(), []
    for mk in missing:
        lid = _layer_id(mk)
        if mk.startswith("model.layers.") and lid is not None:
            missing_layer_ids.add(lid)
        elif tie and mk.startswith("lm_head"):
            continue  # tied -> shares the (transplanted) embed weight
        else:
            bad_missing.append(mk)
    expected_fresh = set(range(keep_front_layers, keep_front_layers + n_fresh_layers))
    assert not bad_missing, f"[sanity2] non-fresh keys missing: {bad_missing[:8]}"
    assert missing_layer_ids == expected_fresh, (
        f"[sanity2] missing layer-ids {sorted(missing_layer_ids)} != "
        f"fresh set {sorted(expected_fresh)}"
    )

    # assert 3: copied-key count == (per-layer tensor count)*keep + non-layer keys
    per_layer = len([k for k in base_sd if _layer_id(k) == 0
                     and k.startswith("model.layers.")])
    n_nonlayer = len([k for k in base_sd if not k.startswith("model.layers.")])
    expected_copied = n_nonlayer + per_layer * keep_front_layers
    assert len(keep_keys) == expected_copied, (
        f"[sanity3] copied {len(keep_keys)} != expected {expected_copied} "
        f"(={n_nonlayer}+{per_layer}*{keep_front_layers})"
    )

    # assert 4: transplanted tensors match base elementwise (upcast-lossless -> 0)
    model_sd = model.state_dict()
    max_diff = 0.0
    for k in keep_keys:
        d = (model_sd[k].float() - base_sd[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff == 0.0, f"[sanity4] transplant max|model-base| = {max_diff:.3e} != 0"

    fresh_stats = _fresh_init_stats(model, keep_front_layers)

    del base, base_sd, filtered
    gc.collect()

    sanity = {
        "transplanted": True,
        "base_num_layers": 32,
        "per_layer_tensors": per_layer,
        "n_nonlayer_keys": n_nonlayer,
        "n_copied": len(keep_keys),
        "expected_copied": expected_copied,
        "missing_fresh_layer_ids": sorted(missing_layer_ids),
        "transplant_max_abs_diff": max_diff,
        "tie_word_embeddings": tie,
        **fresh_stats,
    }
    if is_main:
        logger.info(
            f"[transplant] copied {len(keep_keys)} tensors (front {keep_front_layers} "
            f"layers + {n_nonlayer} non-layer keys); fresh tail {sorted(missing_layer_ids)} "
            f"left at Hunyuan init | max|model-base|={max_diff:.3e} (exact) | "
            f"fresh_ln_ones={fresh_stats['fresh_input_layernorm_all_ones']} "
            f"fresh_q_std={fresh_stats['fresh_q_proj_std']:.4f} "
            f"fresh_expert_std={fresh_stats['fresh_experts_gate_up_std']:.4f} -> ALL CHECKS PASS"
        )
    return sanity


def build_hunyuan_minimal(model_path, keep_front_layers, n_fresh_layers, dtype,
                          transplant=True, meta=False, is_main=True):
    """Build a (keep_front + n_fresh)-layer Hunyuan-A13B MoE model.

    meta=True  -> instantiate on the meta device (no host RAM; for non-rank0 FSDP
                  ranks and for dry-run structure checks). Never transplants.
    transplant -> overwrite front keep layers + embed + norm (+ tied lm_head) with
                  the pretrained weights and run sanity asserts (rank0 / single proc).

    `dtype` should be torch.float32 for continue-training (fp32 master weights).
    Returns (model, cfg, sanity_dict)."""
    cfg, total_layers = build_pruned_config(model_path, keep_front_layers, n_fresh_layers)
    if meta:
        with torch.device("meta"):
            model = HunYuanMoEV1ForCausalLM(cfg)
        return model, cfg, {"transplanted": False, "meta": True}

    model = HunYuanMoEV1ForCausalLM(cfg).to(dtype)
    if transplant:
        # base loaded in its native bf16 (lossless upcast into the fp32 model).
        sanity = transplant_front(
            model, model_path, keep_front_layers, n_fresh_layers,
            torch.bfloat16, is_main,
        )
    else:
        sanity = {"transplanted": False}
    return model, cfg, sanity


# ---------------------------------------------------------------------------
# param classification / freezing (differential LR)
# ---------------------------------------------------------------------------
def _classify_param(name, keep_front_layers, from_scratch):
    """'fresh' (fresh tail layers -> high LR) vs 'inherited' (front layers + embed
    + norm + tied lm_head -> low LR). from_scratch -> everything 'fresh'.

    Robust to FSDP name prefixes (_fsdp_wrapped_module. etc.) via regex."""
    if from_scratch:
        return "fresh"
    lid = _layer_id(name)
    if lid is not None and "layers." in name:
        return "inherited" if lid < keep_front_layers else "fresh"
    # embed_tokens / final norm / (tied) lm_head -> all pretrained -> inherited
    return "inherited"


def apply_freeze_front(model, keep_front_layers, is_main):
    """Arm A: freeze inherited front layers + embed + norm; train fresh tail only."""
    n_frozen = n_train = 0
    for name, p in model.named_parameters():
        if _classify_param(name, keep_front_layers, from_scratch=False) == "inherited":
            p.requires_grad_(False)
            n_frozen += p.numel()
        else:
            n_train += p.numel()
    if is_main:
        logger.info(f"[freeze] inherited frozen={n_frozen/1e9:.3f}B "
                    f"trainable(fresh)={n_train/1e9:.3f}B")
    return n_frozen, n_train


def build_param_groups(model, args, is_main):
    """Differential-LR param groups (also splitting weight-decay by ndim).
    fresh (tail layers): base_lr=args.lr. inherited (front+embed+norm): lr_inherited."""
    specs = {
        "fresh_decay":  {"params": [], "weight_decay": args.weight_decay,
                         "base_lr": args.lr, "min_lr": args.min_lr},
        "fresh_nodecay": {"params": [], "weight_decay": 0.0,
                          "base_lr": args.lr, "min_lr": args.min_lr},
        "inh_decay":    {"params": [], "weight_decay": args.weight_decay,
                         "base_lr": args.lr_inherited, "min_lr": args.min_lr_inherited},
        "inh_nodecay":  {"params": [], "weight_decay": 0.0,
                         "base_lr": args.lr_inherited, "min_lr": args.min_lr_inherited},
    }
    for name, pp in model.named_parameters():
        if not pp.requires_grad:
            continue
        cls = _classify_param(name, args.keep_front_layers, args.from_scratch)
        prefix = "fresh" if cls == "fresh" else "inh"
        key = f"{prefix}_decay" if pp.ndim >= 2 else f"{prefix}_nodecay"
        specs[key]["params"].append(pp)

    param_groups = [g for g in specs.values() if len(g["params"]) > 0]
    if is_main:
        for gname, g in specs.items():
            n = sum(p.numel() for p in g["params"])
            if n > 0:
                logger.info(f"[optim] group {gname}: {n/1e9:.3f}B params "
                            f"base_lr={g['base_lr']:.2e} min_lr={g['min_lr']:.2e}")
    return param_groups


# ---------------------------------------------------------------------------
# FSDP tied-weight deadlock fix: untie embed / lm_head before wrapping
# ---------------------------------------------------------------------------
def untie_output_embeddings(model, is_main):
    """Make `lm_head.weight` an INDEPENDENT parameter (a copy of the input
    embedding) on every rank, and flip cfg.tie_word_embeddings=False.

    WHY (first-forward FSDP deadlock, diagnosed 2026-07-12 py-spy):
      With tie_word_embeddings=True, `lm_head.weight IS model.embed_tokens.weight`
      (single shared tensor, see _tied_weights_keys). Under FSDP FULL_SHARD:
        * rank0 materialises the transplant on CPU with the tie intact, so FSDP
          de-duplicates the shared tensor into ONE root flat-param -> root layout
          = [embed_tokens.weight, norm.weight].
        * the meta (non-rank0) ranks go through `param_init_fn` -> `to_empty(
          recurse=False)`, which allocates FRESH per-module storage and silently
          BREAKS the tie; those ranks then register `lm_head.weight` as a SEPARATE
          flat-param -> root layout = [embed_tokens.weight, norm.weight,
          lm_head.weight].
      The root all-gather therefore has a different numel across ranks. rank0
      finishes its (smaller) unshard and races ahead into the embed forward while
      the other ranks are still casting a larger flat-param in pre-unshard -> the
      collective never matches -> NCCL busy-wait hang on the very first forward
      (exactly the observed rank0@post-unshard / rank1,4@pre-unshard divergence).

    Explicit untie removes the asymmetry: embed & lm_head are two independent
    params on ALL ranks, so the root flat-param layout is identical everywhere and
    the all-gather sizes match. lm_head is then a normal param -- on rank0 it holds
    a copy of the (transplanted) embedding, on meta ranks it is materialised empty
    and filled by sync_module_states' broadcast from rank0.

    MUST run BEFORE wrap_fsdp, and AFTER transplant / resume / freeze so lm_head
    inherits the correct weights and requires_grad state."""
    cfg = model.config
    if not bool(getattr(cfg, "tie_word_embeddings", False)):
        return False
    embed = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()
    assert embed is not None and lm_head is not None, (
        "[untie] could not locate input/output embeddings to untie")
    w = embed.weight
    if lm_head.weight is w:
        if w.is_meta:
            new_w = torch.nn.Parameter(torch.empty_like(w), requires_grad=w.requires_grad)
        else:
            new_w = torch.nn.Parameter(w.detach().clone(), requires_grad=w.requires_grad)
        lm_head.weight = new_w
    cfg.tie_word_embeddings = False
    # Belt-and-suspenders: neutralise any later post_init()/tie_weights() that
    # would re-share the tensors and re-introduce the flat-param asymmetry.
    model.tie_weights = (lambda *a, **k: None)  # type: ignore[assignment]
    if is_main:
        loc = "meta" if w.is_meta else "materialised"
        logger.info(
            f"[untie] embed/lm_head UNTIED before FSDP ({loc} rank0 build); "
            f"lm_head.weight now independent shape={tuple(lm_head.weight.shape)} "
            f"requires_grad={lm_head.weight.requires_grad}; cfg.tie_word_embeddings=False "
            f"-> identical root flat-param layout on all ranks")
    return True


# ---------------------------------------------------------------------------
# FSDP wrapping
# ---------------------------------------------------------------------------
def wrap_fsdp(model, args, local_rank, is_main):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        MixedPrecision,
        ShardingStrategy,
        CPUOffload,
    )
    from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

    auto_wrap = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={HunYuanMoEV1DecoderLayer},
    )
    # fp32 flat params (the master); compute in bf16; grad reduce in fp32.
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.float32,
    )
    # non-rank0 ranks build on meta -> materialise empty on device, then
    # sync_module_states broadcasts rank0's real weights.
    def _param_init_fn(module):
        module.to_empty(device=torch.device("cuda", local_rank), recurse=False)

    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp,
        device_id=torch.device("cuda", local_rank),
        sync_module_states=True,          # broadcast rank0 weights to meta ranks
        param_init_fn=_param_init_fn,
        use_orig_params=True,             # name-based differential-LR param groups
        limit_all_gathers=True,
        cpu_offload=CPUOffload(offload_params=True) if args.fsdp_cpu_offload else None,
    )
    if is_main:
        logger.info(f"[fsdp] FULL_SHARD wrapped on HunYuanMoEV1DecoderLayer; "
                    f"mp(param=bf16, reduce=fp32) use_orig_params=True "
                    f"cpu_offload={bool(args.fsdp_cpu_offload)}")
    return model


# ---------------------------------------------------------------------------
# checkpoint save (FSDP full state dict, rank0 only)
# ---------------------------------------------------------------------------
def _save(model, optimizer, args, step, epoch, cfg, is_ddp, is_main, final=False):
    from torch.distributed.fsdp import (
        FullyShardedDataParallel as FSDP,
        StateDictType,
        FullStateDictConfig,
        FullOptimStateDictConfig,
    )
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")

    if is_ddp:
        save_cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        optim_cfg = FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, save_cfg, optim_cfg):
            model_state = model.state_dict()
            optim_state = FSDP.optim_state_dict(model, optimizer)
    else:
        model_state = model.state_dict()
        optim_state = optimizer.state_dict()

    if not is_main:
        return
    torch.save({
        "model_state": model_state,
        "optimizer_state": optim_state,
        "step": step,
        "epoch": epoch,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "train_args": vars(args),
        # arch descriptors for downstream eval rebuild
        "model_family": "hunyuan_v1_moe",
        "base_model_path": args.model_path,
        "keep_front_layers": args.keep_front_layers,
        "n_fresh_layers": args.n_fresh_layers,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "head_dim": cfg.head_dim,
        "tie_word_embeddings": bool(cfg.tie_word_embeddings),
        "freeze_front": bool(args.freeze_front),
        "from_scratch": bool(args.from_scratch),
        "seq_len": args.seq_len,
    }, path)
    logger.info(f"saved {path}")


# ---------------------------------------------------------------------------
# dry-run structural validation (no GPUs, no 160GB base load)
# ---------------------------------------------------------------------------
def _dry_run(args):
    total = args.keep_front_layers + args.n_fresh_layers
    logger.info(f"[dry_run] keep={args.keep_front_layers} fresh={args.n_fresh_layers} "
                f"total={total}")

    # (1) meta build of the REAL pruned config: structure + param count only.
    model, cfg, _ = build_hunyuan_minimal(
        args.model_path, args.keep_front_layers, args.n_fresh_layers,
        torch.float32, transplant=False, meta=True, is_main=True,
    )
    n_layers = len(model.model.layers)
    n_params = sum(p.numel() for p in model.parameters())
    l0 = model.model.layers[0]
    assert n_layers == total, f"{n_layers} != {total}"
    assert cfg.num_hidden_layers == total and cfg.head_dim is not None
    tied = model.lm_head.weight is model.model.embed_tokens.weight
    logger.info(f"[dry_run] REAL cfg -> layers={n_layers} params={n_params/1e9:.2f}B "
                f"head_dim={cfg.head_dim} mlp={type(l0.mlp).__name__} "
                f"experts.gate_up_proj={tuple(l0.mlp.experts.gate_up_proj.shape)} "
                f"experts.down_proj={tuple(l0.mlp.experts.down_proj.shape)} "
                f"tie(lm_head is embed)={tied}")
    assert tied == bool(cfg.tie_word_embeddings)
    del model
    gc.collect()

    # (2) shrunk end-to-end CPU build: exercises real init + transplant + all
    #     asserts + fresh-MoE-init + param classification WITHOUT the 160GB base.
    small, _ = build_pruned_config(args.model_path, args.keep_front_layers,
                                   args.n_fresh_layers)
    small.hidden_size = 128
    small.num_attention_heads = 8
    small.num_key_value_heads = 2
    small.head_dim = 16
    small.attention_head_dim = 16
    small.intermediate_size = 64
    small.num_experts = 4
    small.num_local_experts = 4
    small.moe_topk = [2] * (args.keep_front_layers + args.n_fresh_layers)
    small.num_shared_expert = [1] * (args.keep_front_layers + args.n_fresh_layers)
    small.moe_intermediate_size = [64] * (args.keep_front_layers + args.n_fresh_layers)
    small.vocab_size = 512
    small.org_vocab_size = 512
    small.pad_token_id = 0  # shrunk vocab: keep padding_idx < vocab_size
    small.pad_id = 0
    small.eos_token_id = 1
    small.bos_token_id = 1

    pruned = HunYuanMoEV1ForCausalLM(small).to(torch.float32)
    base = HunYuanMoEV1ForCausalLM(small).to(torch.float32)   # stand-in "pretrained"
    base_sd = base.state_dict()
    keep_keys = _copied_keys(base_sd, args.keep_front_layers)
    missing, unexpected = pruned.load_state_dict({k: base_sd[k] for k in keep_keys},
                                                 strict=False)
    pruned.tie_weights()
    # replicate the sanity asserts against the stand-in base
    assert unexpected == [], f"[dry_run] unexpected {unexpected[:4]}"
    tie = bool(small.tie_word_embeddings)
    miss_lids, bad = set(), []
    for mk in missing:
        lid = _layer_id(mk)
        if mk.startswith("model.layers.") and lid is not None:
            miss_lids.add(lid)
        elif tie and mk.startswith("lm_head"):
            continue
        else:
            bad.append(mk)
    assert not bad, f"[dry_run] bad missing {bad[:4]}"
    assert miss_lids == set(range(args.keep_front_layers,
                                  args.keep_front_layers + args.n_fresh_layers))
    fresh = _fresh_init_stats(pruned, args.keep_front_layers)
    # param classification sanity
    cls_counts = {"fresh": 0, "inherited": 0}
    for name, p in pruned.named_parameters():
        cls_counts[_classify_param(name, args.keep_front_layers, False)] += p.numel()
    logger.info(f"[dry_run] shrunk transplant OK: copied={len(keep_keys)} "
                f"unexpected=0 missing_fresh={sorted(miss_lids)} "
                f"fresh_ln_ones={fresh['fresh_input_layernorm_all_ones']} "
                f"fresh_q_std={fresh['fresh_q_proj_std']:.4f} "
                f"fresh_expert_std={fresh['fresh_experts_gate_up_std']:.4f}")
    logger.info(f"[dry_run] param classify: fresh={cls_counts['fresh']/1e3:.1f}K "
                f"inherited={cls_counts['inherited']/1e3:.1f}K")
    logger.info("[dry_run] ALL structural + transplant-logic checks PASS; exiting (no train).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, default="",
                   help="Hunyuan-tokenized (N, seq_len) uint32 npy. Required for training.")
    p.add_argument("--output_dir", type=str, default="")
    p.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    p.add_argument("--keep_front_layers", type=int, default=24)
    p.add_argument("--n_fresh_layers", type=int, default=2)
    p.add_argument("--freeze_front", action="store_true",
                   help="Arm A: freeze inherited (front+embed+norm), train fresh tail only")
    p.add_argument("--from_scratch", action="store_true",
                   help="Control: ignore base weights, random-init all layers, train all")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accumulation_steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4, help="LR for fresh tail layers")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lr_inherited", type=float, default=2e-5,
                   help="LR for inherited front layers + embed + norm")
    p.add_argument("--min_lr_inherited", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--max_rows", type=int, default=0, help=">0 to subset dataset (smoke)")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--fsdp_cpu_offload", action="store_true",
                   help="offload FSDP params+optim to CPU (needed on 95GB H20; slower). "
                        "Not needed on 183GB L20A.")
    p.add_argument("--dry_run_build", action="store_true",
                   help="meta + shrunk-CPU structural/transplant-logic validation, then exit. "
                        "No GPUs, no 160GB base load.")
    args = p.parse_args()

    if args.dry_run_build:
        _dry_run(args)
        return

    assert args.data_path and args.output_dir, "--data_path and --output_dir required for training"

    ddp = "RANK" in os.environ
    if ddp:
        # NCCL init-timeout fix (bug: rank1-7 crash "wait timeout after 600000ms").
        #   rank0 materialises the ~65B transplant on CPU (>10min: 160GB base disk
        #   load + fresh-model init) BEFORE it reaches the first collective (the
        #   FSDP sync_module_states broadcast). With the default 600s PG timeout,
        #   the NCCL communicator is created lazily on that first collective, so
        #   rank1-7 (which finish their fast meta build in seconds) block in the
        #   comm bootstrap waiting for rank0 -> abort after 10min -> whole job dies.
        #   Two-part fix:
        #     (a) timeout=timedelta(hours=2): the PG (and every collective on it,
        #         incl. the FSDP broadcast) now tolerates rank0's slow assembly.
        #     (b) device_id=...: forces EAGER communicator formation inside
        #         init_process_group itself, while all ranks are still aligned here
        #         (before the transplant) -> the comm bootstrap happens fast and the
        #         later broadcast just reuses it, so we never hit the lazy-init race.
        local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", 0)))
        assert torch.cuda.is_available(), "training requires CUDA (FSDP)"
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            "nccl",
            timeout=timedelta(hours=2),
            device_id=torch.device("cuda", local_rank),
        )
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        # Sanity collective while all ranks are aligned (confirms comm is up before
        # rank0 disappears into the multi-minute transplant).
        dist.barrier()
    else:
        rank, world_size, local_rank = 0, 1, 0
        assert torch.cuda.is_available(), "training requires CUDA (FSDP)"
        torch.cuda.set_device(local_rank)
    is_main = rank == 0
    device = torch.device("cuda", local_rank)
    model_dtype = torch.float32  # fp32 master weights

    total_layers = args.keep_front_layers + args.n_fresh_layers
    if args.from_scratch:
        arm = f"scratch{total_layers}L"
    elif args.freeze_front:
        arm = f"frozen_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"
    else:
        arm = f"healing_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== Hunyuan-A13B minimal-arch probe#2 [{arm}] ===")
        logger.info(f"world_size={world_size} bs={args.batch_size} "
                    f"gaccum={args.grad_accumulation_steps} eff_bs={eff_bs} "
                    f"seq_len={args.seq_len} lr_fresh={args.lr} lr_inh={args.lr_inherited} "
                    f"max_steps={args.max_steps} fp32 master + FSDP(bf16 compute)")

    # ---- build model ----
    # rank0 (or single proc) materialises the transplanted weights on CPU; other
    # FSDP ranks build on meta (no host RAM) and receive weights via sync_module_states.
    resume_ckpt = None
    if args.resume_from and is_main:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu", weights_only=False)
        logger.info(f"[resume] loaded ckpt {args.resume_from} "
                    f"(step {resume_ckpt.get('step')})")

    build_real = is_main
    do_transplant = (not args.from_scratch) and (resume_ckpt is None)
    model, cfg, sanity = build_hunyuan_minimal(
        args.model_path, args.keep_front_layers, args.n_fresh_layers,
        model_dtype, transplant=(do_transplant and build_real),
        meta=(not build_real), is_main=is_main,
    )
    if resume_ckpt is not None and is_main:
        model.load_state_dict(resume_ckpt["model_state"], strict=True)
        model.tie_weights()
        logger.info(f"[resume] restored {len(resume_ckpt['model_state'])} tensors")

    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if args.freeze_front and not args.from_scratch:
        apply_freeze_front(model, args.keep_front_layers, is_main)

    if is_main:
        n = sum(pp.numel() for pp in model.parameters())
        logger.info(f"model params = {n/1e9:.3f}B num_hidden_layers={cfg.num_hidden_layers}")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "arm": arm, "model_family": "hunyuan_v1_moe",
                "base_model_path": args.model_path,
                "keep_front_layers": args.keep_front_layers,
                "n_fresh_layers": args.n_fresh_layers,
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size, "vocab_size": cfg.vocab_size,
                "head_dim": cfg.head_dim,
                "tie_word_embeddings": bool(cfg.tie_word_embeddings),
                "freeze_front": bool(args.freeze_front),
                "from_scratch": bool(args.from_scratch),
                "seq_len": args.seq_len, "lr_fresh": args.lr,
                "lr_inherited": args.lr_inherited, "sanity": sanity,
            }, f, indent=2)

    # ---- FSDP wrap ----
    if ddp:
        # CRITICAL (see untie_output_embeddings docstring): with tied embed/lm_head,
        # FSDP's meta-rank param_init_fn (to_empty) breaks the tie -> root flat-param
        # composition diverges across ranks -> first-forward all-gather size mismatch
        # -> NCCL deadlock. Untie on ALL ranks BEFORE wrapping so the layout matches.
        untie_output_embeddings(model, is_main)
        model = wrap_fsdp(model, args, local_rank, is_main)
    else:
        model = model.to(device)

    # ---- data (Hunyuan-tokenized; guard against embedding out-of-range) ----
    ds = NpyChunkDataset(args.data_path, args.seq_len)
    if args.max_rows and args.max_rows > 0:
        ds.arr = ds.arr[: args.max_rows]
    if is_main:
        vmax = int(np.asarray(ds.arr[: min(len(ds), 1000)]).max())
        assert vmax < cfg.vocab_size, (
            f"data max token id {vmax} >= vocab_size {cfg.vocab_size} -> WRONG tokenizer "
            f"(embedding out-of-range). Re-tokenize slimpajama with the Hunyuan tokenizer.")
        logger.info(f"dataset rows={len(ds)} seq_len={ds.seq_len} max_id(sample)={vmax} "
                    f"< vocab {cfg.vocab_size} OK  from {args.data_path}")

    if ddp:
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True)

    # ---- optimizer (differential-LR groups over FSDP orig params) ----
    param_groups = build_param_groups(model, args, is_main)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)

    # ---- resume optimizer / step / epoch ----
    step, epoch = 0, 0
    if resume_ckpt is not None or (args.resume_from and ddp):
        # broadcast step/epoch from rank0
        meta_t = torch.zeros(2, dtype=torch.long, device=device)
        if is_main and resume_ckpt is not None:
            meta_t[0] = int(resume_ckpt.get("step", 0))
            meta_t[1] = int(resume_ckpt.get("epoch", 0))
        if ddp:
            dist.broadcast(meta_t, src=0)
        step, epoch = int(meta_t[0].item()), int(meta_t[1].item())
        if resume_ckpt is not None and "optimizer_state" in resume_ckpt and is_main:
            logger.info("[resume] optimizer_state present (rank0); FSDP optim resume "
                        "requires scatter -- best-effort warm-restart otherwise")
        if is_main:
            logger.info(f"[resume] continue @ step={step} epoch={epoch}")
        del resume_ckpt
        gc.collect()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    micro, accum_loss, accum_cnt = 0, 0.0, 0
    if sampler is not None and epoch > 0:
        sampler.set_epoch(epoch)
    data_iter = iter(loader)
    t0 = time.time()

    while step < args.max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            epoch += 1
            if sampler is not None:
                sampler.set_epoch(epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        is_boundary = (micro + 1) % args.grad_accumulation_steps == 0
        sync_ctx = model.no_sync() if (ddp and not is_boundary) else _nullctx()
        with sync_ctx:
            # FSDP MixedPrecision handles bf16 compute; no explicit autocast needed.
            out = model(input_ids=input_ids, labels=labels)
            loss = out.loss / args.grad_accumulation_steps
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_cnt += 1
        micro += 1

        if is_boundary:
            for g in optimizer.param_groups:
                g["lr"] = get_lr(step, args.warmup_steps, args.max_steps,
                                 g["base_lr"], g["min_lr"])
            if ddp:
                gnorm = model.clip_grad_norm_(args.grad_clip)
            else:
                gnorm = torch.nn.utils.clip_grad_norm_(
                    [pp for pp in model.parameters() if pp.requires_grad], args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg,20)):.2f} "
                            f"lr={optimizer.param_groups[0]['lr']:.2e} "
                            f"gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss, accum_cnt = 0.0, 0
                t0 = time.time()

            if step % args.save_every == 0 and step > 0:
                _save(model, optimizer, args, step, epoch, cfg, ddp, is_main)

    _save(model, optimizer, args, step, epoch, cfg, ddp, is_main, final=True)
    if is_main:
        logger.info(f"DONE [{arm}] at step {step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
