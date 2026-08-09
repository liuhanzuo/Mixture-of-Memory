#!/usr/bin/env python3
"""Probe #2 for the minimal-architecture hypothesis, ported to REAL OLMo-2.

OLMo-2 port of scripts/train_qwen3_arch_probe2.py (armB = keep-front + fresh).
Same "understand-then-generate division of labour" probe: semantic understanding
saturates in the front layers, next-token prediction forms in the top layers, so
a "front-j layers + a few fresh NTP layers" model, CONTINUE-TRAINED (healed),
could approach the full-depth model. This script scales that experiment onto a
pretrained OLMo-2 (1B / 7B): keep the FRONT `keep_front_layers` decoder layers +
embed + final norm + lm_head (tie_word_embeddings=False -> lm_head transplanted
independently), DROP the top layers, append `n_fresh_layers` FRESH Olmo2-init
decoder layers, and continue-train.

Construction (default keep_front=14, n_fresh=2 -> 16 layers total for 7B):
  (a) cfg = Olmo2Config.from_pretrained(...); cfg.num_hidden_layers = keep+fresh.
      (OLMo-2 has NO `layer_types` field -- unlike Qwen3 -- so nothing else to
      reset; we defensively reset it only if the attr is present.)
  (b) model = Olmo2ForCausalLM(cfg).to(dtype)   [post_init gives ALL layers,
      including the fresh tail, the correct Olmo2 init -- never hand-build an
      Olmo2DecoderLayer, that risks the wrong init].
  (c) transplant front keep_front layers (layers.0..keep-1) + embed_tokens +
      model.norm + lm_head from the pretrained base; fresh tail layers stay random.

Arms (via flags):
  * Arm A --freeze_front : freeze inherited front layers, train fresh + norm +
    lm_head/embed only.
  * Arm B (default)      : train ALL layers ("healing"). Differential LR:
    fresh+lm_head high, front layers + embed + norm low.
  * Control 2 --from_scratch : ignore base weights, random-init all layers, train
    everything at a single LR (embed / final norm / lm_head are random too).
  * Control 3 --random_trunk : SAME depth/shape as the keep+fresh arm, trunk
    (model.layers.*) fully random-init, but embed_tokens / model.norm / lm_head
    TRANSPLANTED from the pretrained base. Isolates "does inheriting trunk
    weights help?" from the confound that --from_scratch also throws away the
    pretrained vocab embedding + readout head (which a ~1.6M-token SFT can never
    relearn for a 100352-vocab model). Mutually exclusive with --from_scratch.

OLMo-2 layer layout (verified 2026-07-16 against the local checkpoints):
  Each Olmo2DecoderLayer has 11 tensors and NO input_layernorm (OLMo-2 is
  POST-norm): self_attn.{q,k,v,o}_proj + self_attn.{q,k}_norm (QK-norm) +
  mlp.{gate,up,down}_proj + post_attention_layernorm + post_feedforward_layernorm.
  3 non-layer keys: model.embed_tokens.weight, model.norm.weight, lm_head.weight
  (tie_word_embeddings=False). RMSNorm, SwiGLU, RoPE theta=5e5, untied embeddings,
  vocab 100352 (7B) / 100352 (1B). This is why the fresh-init assert checks
  post_attention_layernorm all-ones (NOT input_layernorm, which does not exist).

CRITICAL: fp32 MASTER WEIGHTS. We continue-train pretrained weights, so params
stay fp32 (do NOT model.to(bf16)); forward runs under bf16 autocast and AdamW
states are fp32. Single most important anti-catastrophic-forgetting knob.

Shares data loading / cosine schedule / null-context with
scripts/train_semantic_bottleneck_1b.py (imported, NOT modified). Does NOT touch
scripts/train_qwen3_arch_probe2.py or any Llama-only script. Checkpoints are raw
state_dict (+ arch meta) so a matching eval can rebuild an identical model.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Olmo2Config, Olmo2ForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_rotation import (  # noqa: E402
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)

# Reuse (do NOT modify) the sibling Llama training script.
from train_semantic_bottleneck_1b import (  # noqa: E402
    NpyChunkDataset,
    collate_fn,
    get_lr,
    _nullctx,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OLMo-2 layout facts (verified 2026-07-16 against the local 1B/7B checkpoints):
#   * 11 tensors per decoder layer (q/k/v/o_proj + q_norm/k_norm + gate/up/down_proj
#     + post_attention_layernorm + post_feedforward_layernorm). POST-norm: NO
#     input_layernorm.
#   * 3 non-layer keys (model.embed_tokens.weight, model.norm.weight, lm_head.weight).
#   * tie_word_embeddings=False -> lm_head.weight is a real, separate tensor.
N_TENSORS_PER_LAYER = 11
N_NONLAYER_KEYS = 3  # model.embed_tokens.weight, model.norm.weight, lm_head.weight


def set_seed(seed: int):
    """Seed python/numpy/torch (CPU + all CUDA devices) for reproducible +
    DDP-consistent fresh-tail init. Called on every rank BEFORE model build.
    Mirrors scripts/train_olmo2_shortgpt_fresh.py::set_seed."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
def _copied_keys(state_dict, keep_front_layers):
    """Base-ckpt keys we transplant: the 3 non-layer keys (embed / final-norm /
    lm_head) + the front decoder layers layers.0..keep-1. Top layers dropped."""
    keys = []
    for k in state_dict:
        if k.startswith("model.layers."):
            try:
                lid = int(k.split(".")[2])
            except (IndexError, ValueError):
                continue
            if lid < keep_front_layers:
                keys.append(k)
        else:
            # model.embed_tokens.weight, model.norm.weight, lm_head.weight
            keys.append(k)
    return keys


def _assert_fresh_init(model, keep_front_layers):
    """Fresh tail layers must retain proper Olmo2 init after transplant.

    OLMo-2 is POST-norm and has NO input_layernorm; the first RMSNorm inside a
    layer is post_attention_layernorm. Checks: post_attention_layernorm.weight
    all-ones (RMSNorm), q_norm.weight all-ones (QK-norm RMSNorm), and
    q_proj.weight std ~= 0.02 (initializer_range). Returns
    (ln_all_ones, qnorm_all_ones, q_std)."""
    sd = model.state_dict()
    fresh_ln = sd[f"model.layers.{keep_front_layers}.post_attention_layernorm.weight"]
    ln_all_ones = bool(torch.all(fresh_ln == 1.0).item())
    fresh_qnorm = sd[f"model.layers.{keep_front_layers}.self_attn.q_norm.weight"]
    qnorm_all_ones = bool(torch.all(fresh_qnorm == 1.0).item())
    fresh_q = sd[f"model.layers.{keep_front_layers}.self_attn.q_proj.weight"]
    q_std = fresh_q.float().std().item()
    assert ln_all_ones, (
        f"fresh layer {keep_front_layers} post_attention_layernorm not all-ones "
        f"(min={fresh_ln.min().item()}, max={fresh_ln.max().item()})"
    )
    assert qnorm_all_ones, (
        f"fresh layer {keep_front_layers} q_norm not all-ones "
        f"(min={fresh_qnorm.min().item()}, max={fresh_qnorm.max().item()})"
    )
    assert 0.01 < q_std < 0.04, (
        f"fresh layer {keep_front_layers} q_proj.weight std={q_std:.4f} "
        f"not ~0.02 -> wrong init"
    )
    return ln_all_ones, qnorm_all_ones, q_std


def transplant_front(model, base_path, keep_front_layers, n_fresh_layers, dtype, is_main):
    """Load the pretrained base OLMo-2, transplant front keep_front layers + embed
    + norm + lm_head into `model` (fresh tail layers left at their Olmo2 random
    init), and run the 4 required sanity asserts + the fresh-init assert.

    Returns a sanity dict. On any assert failure this raises (must crash the run)."""
    base = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True
    )
    base_num_layers = base.config.num_hidden_layers
    base_sd = base.state_dict()
    keep_keys = _copied_keys(base_sd, keep_front_layers)
    filtered = {k: base_sd[k] for k in keep_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # --- assert 1: no unexpected keys ---
    assert unexpected == [], f"[sanity1] unexpected keys when transplanting: {unexpected[:8]}"

    # --- assert 2: the ONLY missing keys are the fresh tail layers ---
    missing_layer_ids = set()
    bad_missing = []
    for mk in missing:
        if mk.startswith("model.layers."):
            missing_layer_ids.add(int(mk.split(".")[2]))
        else:
            bad_missing.append(mk)
    expected_fresh_ids = set(range(keep_front_layers, keep_front_layers + n_fresh_layers))
    assert not bad_missing, f"[sanity2] non-layer keys unexpectedly missing: {bad_missing}"
    assert missing_layer_ids == expected_fresh_ids, (
        f"[sanity2] missing layer-ids {sorted(missing_layer_ids)} != "
        f"fresh set {sorted(expected_fresh_ids)}"
    )

    # --- assert 3: number of copied keys == 3 + 11*keep_front ---
    expected_copied = N_NONLAYER_KEYS + N_TENSORS_PER_LAYER * keep_front_layers
    assert len(keep_keys) == expected_copied, (
        f"[sanity3] copied {len(keep_keys)} keys != expected {expected_copied} "
        f"(={N_NONLAYER_KEYS}+{N_TENSORS_PER_LAYER}*{keep_front_layers})"
    )

    # --- assert 4: transplanted tensors match base elementwise (max diff == 0) ---
    model_sd = model.state_dict()
    max_diff = 0.0
    for k in keep_keys:
        d = (model_sd[k].float() - base_sd[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff == 0.0, (
        f"[sanity4] transplant max|model_param - base| = {max_diff:.3e} != 0.0"
    )

    # --- fresh-init assert (tail layers untouched, correct Olmo2 init) ---
    # n_fresh_layers==0 (e.g. full-32L continued-pretraining control): every
    # layer is transplanted, there is no fresh tail, and layer index
    # `keep_front_layers` does not exist -> skip the fresh-init check.
    if n_fresh_layers > 0:
        ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, keep_front_layers)
    else:
        ln_all_ones, qnorm_all_ones, q_std = None, None, None

    del base, base_sd, filtered
    gc.collect()

    sanity = {
        "transplanted": True,
        "base_num_layers": base_num_layers,
        "n_copied": len(keep_keys),
        "expected_copied": expected_copied,
        "missing_fresh_layer_ids": sorted(missing_layer_ids),
        "transplant_max_abs_diff": max_diff,
        "fresh_post_attention_layernorm_all_ones": ln_all_ones,
        "fresh_q_norm_all_ones": qnorm_all_ones,
        "fresh_q_proj_std": q_std,
    }
    if is_main:
        logger.info(
            f"[transplant] copied {len(keep_keys)} tensors "
            f"(front {keep_front_layers} layers + embed/norm/lm_head) from a "
            f"{base_num_layers}-layer base; fresh tail layer-ids "
            f"{sorted(missing_layer_ids)} left at Olmo2 init"
        )
        logger.info(
            f"[sanity] unexpected=0 | copied={len(keep_keys)}=={expected_copied} | "
            f"max|model-base|={max_diff:.3e} (exact) | "
            f"fresh_post_attn_ln_all_ones={ln_all_ones} "
            f"fresh_q_norm_all_ones={qnorm_all_ones} fresh_q_std={q_std} "
            f"-> ALL 6 CHECKS PASS"
        )
    return sanity


def transplant_readout_only(model, base_path, total_layers, dtype, is_main):
    """Control 3 (--random_trunk): transplant ONLY the 3 non-layer tensors
    (model.embed_tokens.weight, model.norm.weight, lm_head.weight) from the
    pretrained base; leave EVERY decoder layer at its Olmo2 post_init random init.

    Rationale: --from_scratch randomises the trunk AND the vocab embedding AND
    the readout head, so "A4 (inherit front-j) beats A3 (from_scratch)" is
    confounded -- a ~1.6M-token SFT cannot learn a 100352-vocab embedding + output
    map from scratch. This arm shares A4's tokeniser interface exactly (embed /
    final norm / lm_head bit-identical to base) and differs from A4 ONLY in where
    the trunk weights come from.

    The trunk uses the SAME random init as the fresh tail blocks of A4: both come
    from Olmo2ForCausalLM(cfg) post_init in build_olmo2_minimal step (b) -- we
    never re-initialise anything here, we only decline to overwrite the layers.

    Returns a sanity dict (same shape/keys as transplant_front's where meaningful).
    Raises on any assert failure (must crash the run)."""
    base = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True
    )
    base_num_layers = base.config.num_hidden_layers
    base_sd = base.state_dict()
    # keep_front_layers=0 -> _copied_keys returns exactly the 3 non-layer keys.
    keep_keys = _copied_keys(base_sd, 0)
    filtered = {k: base_sd[k] for k in keep_keys}

    missing, unexpected = model.load_state_dict(filtered, strict=False)

    # --- assert 1: no unexpected keys ---
    assert unexpected == [], (
        f"[rt-sanity1] unexpected keys when transplanting readout: {unexpected[:8]}"
    )

    # --- assert 2: the ONLY missing keys are decoder layers (all of them) ---
    missing_layer_ids = set()
    bad_missing = []
    for mk in missing:
        if mk.startswith("model.layers."):
            missing_layer_ids.add(int(mk.split(".")[2]))
        else:
            bad_missing.append(mk)
    assert not bad_missing, f"[rt-sanity2] non-layer keys unexpectedly missing: {bad_missing}"
    assert missing_layer_ids == set(range(total_layers)), (
        f"[rt-sanity2] missing layer-ids {sorted(missing_layer_ids)} != "
        f"all layers {list(range(total_layers))}"
    )

    # --- assert 3: exactly the 3 non-layer tensors were copied ---
    assert len(keep_keys) == N_NONLAYER_KEYS, (
        f"[rt-sanity3] copied {len(keep_keys)} keys != expected {N_NONLAYER_KEYS} "
        f"(embed_tokens / model.norm / lm_head)"
    )

    # --- assert 4: copied tensors match base elementwise (max diff == 0) ---
    model_sd = model.state_dict()
    max_diff = 0.0
    for k in keep_keys:
        d = (model_sd[k].float() - base_sd[k].float()).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff == 0.0, (
        f"[rt-sanity4] readout transplant max|model_param - base| = {max_diff:.3e} != 0.0"
    )

    # --- fresh-init assert on layer 0 (the whole trunk is fresh here) ---
    ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, 0)

    del base, base_sd, filtered
    gc.collect()

    sanity = {
        "transplanted": True,
        "random_trunk": True,
        "base_num_layers": base_num_layers,
        "n_copied": len(keep_keys),
        "expected_copied": N_NONLAYER_KEYS,
        "missing_fresh_layer_ids": sorted(missing_layer_ids),
        "transplant_max_abs_diff": max_diff,
        "fresh_post_attention_layernorm_all_ones": ln_all_ones,
        "fresh_q_norm_all_ones": qnorm_all_ones,
        "fresh_q_proj_std": q_std,
    }
    if is_main:
        logger.info(
            f"[random_trunk] copied ONLY {len(keep_keys)} non-layer tensors "
            f"(embed_tokens / model.norm / lm_head) from a {base_num_layers}-layer "
            f"base; ALL {total_layers} decoder layers left at Olmo2 random init"
        )
        logger.info(
            f"[rt-sanity] unexpected=0 | copied={len(keep_keys)}=={N_NONLAYER_KEYS} | "
            f"max|model-base|={max_diff:.3e} (exact) | random layer-ids "
            f"{sorted(missing_layer_ids)} | trunk_post_attn_ln_all_ones={ln_all_ones} "
            f"trunk_q_norm_all_ones={qnorm_all_ones} trunk_q_std={q_std} "
            f"-> ALL CHECKS PASS"
        )
    return sanity


def build_olmo2_minimal(base_path, keep_front_layers, n_fresh_layers, dtype,
                        transplant=True, is_main=True, random_trunk=False):
    """Build a (keep_front + n_fresh)-layer OLMo-2 model.

    (a) shrink the pretrained Olmo2 config to `keep_front + n_fresh` layers.
        (OLMo-2 has no `layer_types`; we reset it only if present, for safety.)
    (b) instantiate via Olmo2ForCausalLM(cfg) so post_init gives every layer,
        including the fresh tail, the correct Olmo2 init.
    (c) if transplant: overwrite front keep_front layers + embed + norm + lm_head
        with the pretrained base weights and run the sanity asserts.
        if transplant and random_trunk (Control 3): overwrite ONLY embed + norm +
        lm_head; every decoder layer stays at its Olmo2 random init. Depth/shape
        are identical to the (b) shell either way, so the arms stay comparable.

    Returns (model, cfg, sanity_dict). `dtype` should be torch.float32 for
    continue-training (fp32 master weights)."""
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    total_layers = keep_front_layers + n_fresh_layers
    cfg.num_hidden_layers = total_layers
    # OLMo-2 config has no layer_types; reset defensively only if it exists so a
    # future transformers version that adds it cannot leave a stale length.
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * total_layers
        assert len(cfg.layer_types) == total_layers

    model = Olmo2ForCausalLM(cfg).to(dtype)

    if transplant and random_trunk:
        sanity = transplant_readout_only(model, base_path, total_layers, dtype, is_main)
    elif transplant:
        sanity = transplant_front(
            model, base_path, keep_front_layers, n_fresh_layers, dtype, is_main
        )
    else:
        sanity = {"transplanted": False}
    return model, cfg, sanity


def apply_freeze_front(model, keep_front_layers, is_main):
    """Arm A: freeze inherited front layers; keep everything else trainable
    (fresh tail layers + final norm + lm_head + embed). Returns (n_frozen, n_train)."""
    n_frozen = n_train = 0
    for name, p in model.named_parameters():
        freeze = False
        if name.startswith("model.layers."):
            lid = int(name.split(".")[2])
            if lid < keep_front_layers:
                freeze = True
        if freeze:
            p.requires_grad_(False)
            n_frozen += p.numel()
        else:
            n_train += p.numel()
    if is_main:
        logger.info(
            f"[freeze] front {keep_front_layers} layers frozen: "
            f"frozen={n_frozen/1e6:.1f}M trainable={n_train/1e6:.1f}M params"
        )
    return n_frozen, n_train


def _classify_param(name, keep_front_layers, from_scratch, random_trunk=False):
    """'fresh' (fresh tail layers + lm_head -> high LR) vs 'inherited' (front
    layers + embed + norm -> low LR). from_scratch -> everything 'fresh'.

    Strips a leading 'module.' so classification is correct whether called on a
    bare model or a DDP-wrapped one (build_param_groups runs AFTER DDP wrap, whose
    named_parameters() prefix all names with 'module.'; without this strip every
    trainable param mis-fell-through to 'inherited' and the fresh cap trained at
    lr_inherited instead of lr). from_scratch is unaffected (returns 'fresh' first).

    random_trunk (Control 3, default False -> every pre-existing call site keeps
    its exact behaviour): the whole trunk is random -> every 'model.layers.*'
    param is 'fresh', while embed_tokens / model.norm / lm_head are transplanted
    from the base -> 'inherited'. Note lm_head flips to 'inherited' here (it IS
    inherited in this arm), so a launch that wants a specific LR for it should set
    --lr / --lr_inherited deliberately."""
    if from_scratch:
        return "fresh"
    if name.startswith("module."):
        name = name[len("module."):]
    if random_trunk:
        # trunk fully random -> fresh; embed_tokens / model.norm / lm_head copied
        # from the base -> inherited.
        return "fresh" if name.startswith("model.layers.") else "inherited"
    if name.startswith("model.layers."):
        lid = int(name.split(".")[2])
        return "inherited" if lid < keep_front_layers else "fresh"
    if name.startswith("lm_head"):
        return "fresh"
    # model.embed_tokens.weight, model.norm.weight
    return "inherited"


def build_param_groups(model, args, is_main):
    """Differential-LR param groups (also splitting weight-decay by ndim).

    fresh (tail layers + lm_head): base_lr=args.lr, min_lr=args.min_lr.
    inherited (front layers + embed + norm): base_lr=args.lr_inherited,
        min_lr=args.min_lr_inherited.
    from_scratch: single 'fresh' bucket at args.lr.
    random_trunk: 'fresh' = the whole trunk, 'inherited' = embed/norm/lm_head."""
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
        cls = _classify_param(name, args.keep_front_layers, args.from_scratch,
                              random_trunk=getattr(args, "random_trunk", False))
        prefix = "fresh" if cls == "fresh" else "inh"
        key = f"{prefix}_decay" if pp.ndim >= 2 else f"{prefix}_nodecay"
        specs[key]["params"].append(pp)

    param_groups = [g for g in specs.values() if len(g["params"]) > 0]
    if is_main:
        for gname, g in specs.items():
            n = sum(p.numel() for p in g["params"])
            if n > 0:
                logger.info(f"[optim] group {gname}: {n/1e6:.1f}M params "
                            f"base_lr={g['base_lr']:.2e} min_lr={g['min_lr']:.2e}")
    return param_groups


def build_optimizer(param_groups, args, is_main):
    """AdamW over the differential-LR param groups. Default = fp32 torch AdamW
    (unchanged). --optimizer bnb_adamw8bit uses bitsandbytes 8-bit AdamW to halve
    optimizer-state memory so full-param 7B/4B arms fit a single H20 (Paper C
    A1/A3 fallback). betas/eps match the torch path."""
    if args.optimizer == "bnb_adamw8bit":
        import bitsandbytes as bnb  # noqa: F401  (import errors early if missing)
        opt = bnb.optim.AdamW8bit(
            param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
        )
        if is_main:
            logger.info("[optim] using bitsandbytes AdamW8bit (8-bit optimizer "
                        "state) -- optimizer differs from the fp32-AdamW default")
        return opt
    opt = torch.optim.AdamW(param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8)
    if is_main:
        logger.info("[optim] using torch AdamW (fp32 optimizer state, default)")
    return opt


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def _save(model, optimizer, args, step, epoch, cfg, final=False, rotate=True):
    """Checkpoint = model weights + arch meta + optimizer state / step / epoch /
    max_steps / training args / RNG state (for clean resume). New keys are
    additive; old model-only evals ignore them, resume degrades to warm-restart.

    rotate=False disables the rolling-retention pruning below; used by the
    --save_step0_and_exit path so writing step0.pt can NEVER delete an existing
    training checkpoint in the same output_dir."""
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    torch.save({
        "model_state": root.state_dict(),
        "step": step,
        # --- resume-enabling fields (additive) ---
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "train_args": vars(args),
        "rng_state": rng,
        # arch descriptors so downstream eval can rebuild an identical model.
        "model_family": "olmo2",
        "base_model_path": args.model_path,
        "keep_front_layers": args.keep_front_layers,
        "n_fresh_layers": args.n_fresh_layers,
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": False,
        "freeze_front": bool(args.freeze_front),
        "from_scratch": bool(args.from_scratch),
        "random_trunk": bool(getattr(args, "random_trunk", False)),
        "seq_len": args.seq_len,
        "seed": args.seed,
    }, path)
    logger.info(f"saved {path}")

    # --- rolling retention (shared policy, see scripts/ckpt_rotation.py) -----
    # After the new step*.pt is fully written, keep (a) the --keep_last_n newest
    # step ckpts, (b) step0, (c) every --keep_steps entry, (d) the newest
    # --keep_milestones multiples of --milestone_every; delete the rest. This
    # bounds volume (the keep14 run was once killed mid-save on a full disk)
    # while preserving milestones for heal-curve analysis / resume.
    # final.pt is never rotated; the just-written path is never removed; a failed
    # save rotates nothing; --keep_last_n 0 disables rotation entirely (which is
    # what dense-save runs such as #103's matched-PPL crossing-point capture MUST
    # pass). Runs only where _save runs (rank 0).
    if not final and rotate:
        rotate_checkpoints(
            args.output_dir,
            just_written=path,
            log=logger.info,
            **rotation_kwargs_from_args(args, default_milestone_every=5000),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_path", type=str, required=True)
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--model_path", type=str,
                   default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B",
                   help="pretrained OLMo-2 path (cfg + front-layer transplant source)")
    p.add_argument("--keep_front_layers", type=int, default=14)
    p.add_argument("--n_fresh_layers", type=int, default=2)
    p.add_argument("--freeze_front", action="store_true",
                   help="Arm A: freeze inherited front layers, train fresh+norm+head only")
    p.add_argument("--from_scratch", action="store_true",
                   help="Control 2: ignore base weights, random-init all layers, train all")
    p.add_argument("--random_trunk", action="store_true",
                   help="Control 3: random-init the trunk (model.layers.*) but "
                        "TRANSPLANT embed_tokens / model.norm / lm_head from the base. "
                        "Same depth/shape as the matching --keep_front_layers j "
                        "--n_fresh_layers K arm, so the ONLY variable vs that arm is "
                        "where the trunk weights come from -- unlike --from_scratch, "
                        "which additionally randomises the 100352-vocab embedding and "
                        "the readout head (an unlearnable confound at SFT token budgets). "
                        "Mutually exclusive with --from_scratch.")
    p.add_argument("--max_steps", type=int, default=2000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accumulation_steps", type=int, default=2)
    # differential LR: fresh (tail + lm_head) high; inherited (front + embed + norm) low.
    p.add_argument("--lr", type=float, default=1e-4, help="LR for fresh layers + lm_head")
    p.add_argument("--min_lr", type=float, default=1e-5)
    p.add_argument("--lr_inherited", type=float, default=2e-5,
                   help="LR for inherited front layers + embed + norm")
    p.add_argument("--min_lr_inherited", type=float, default=2e-6)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--optimizer", type=str, default="adamw",
                   choices=["adamw", "bnb_adamw8bit"],
                   help="'adamw' (default, unchanged fp32 torch AdamW) or "
                        "'bnb_adamw8bit' (bitsandbytes 8-bit AdamW; halves optimizer "
                        "state memory so full-param 7B/4B arms fit a single H20). "
                        "Paper C A1/A3 fallback; A4/LoRA stay on adamw. Any arm using "
                        "bnb8bit must note the optimizer difference in its report.")
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--milestone_every", type=int, default=5000,
                   help="rolling-retention milestone modulus: step*.pt whose step "
                        "is a multiple of this are kept permanently (plus latest-2). "
                        "Default 5000 preserves the keep14/12/10 ladder behavior; "
                        "set smaller (e.g. 2500) to durably retain a denser early "
                        "heal curve for matched-PPL crossing-point analysis.")
    # --keep_last_n / --keep_steps / --keep_milestones. milestone_every is
    # declared above, so pass None to avoid an argparse conflict.
    # Defaults are behaviour-preserving for every previously-run config:
    # keep_last_n=3 (was a hardcoded 2 -> now keeps one MORE, strictly safer) and
    # keep_milestones=0 (unlimited = exactly the old milestone semantics), so a
    # resumed run prunes nothing it would not have pruned before.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=None,
                      default_keep_milestones=0)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume_from", type=str, default="",
                   help="path to a step{N}.pt / final.pt to resume from. Restores model "
                        "weights + (if present) optimizer state + global_step + epoch + RNG. "
                        "Old model-only ckpts degrade gracefully to a warm-restart.")
    p.add_argument("--max_rows", type=int, default=0, help=">0 to subset dataset (smoke)")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (python/numpy/torch/CUDA), set on every rank "
                        "BEFORE model build for reproducible + DDP-consistent "
                        "fresh-tail random init.")
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (cuda if available else cpu), 'cpu', or 'cuda'")
    p.add_argument("--dry_run_build", action="store_true",
                   help="build the model shell (no base transplant) + validate arch/init "
                        "logic, then exit. For CPU smoke without loading the base weights.")
    p.add_argument("--save_step0_and_exit", action="store_true",
                   help="construct the keep_front+n_fresh student EXACTLY as training does "
                        "(same transplant + same fresh-block init), save it to "
                        "output_dir/step0.pt in the identical checkpoint format the trainer "
                        "uses (strict-loadable by the eval harness, same keys as any step{N}.pt), "
                        "then exit BEFORE loading data / building the training loop. The "
                        "initial (step-0) reference point for the heal curve. Single-process "
                        "(CUDA_VISIBLE_DEVICES=0) is enough; rotation is disabled so an existing "
                        "step{N}.pt in output_dir is never deleted.")
    args = p.parse_args()

    # --random_trunk and --from_scratch are different controls and cannot be combined.
    if args.random_trunk and args.from_scratch:
        p.error(
            "--random_trunk and --from_scratch are mutually exclusive.\n"
            "  --from_scratch : NOTHING is inherited -- trunk AND embed_tokens AND "
            "model.norm AND lm_head are all random-init (Control 2).\n"
            "  --random_trunk : ONLY the trunk (model.layers.*) is random-init; "
            "embed_tokens / model.norm / lm_head are transplanted from the pretrained "
            "base so the arm shares the keep-front arm's vocab+readout interface "
            "(Control 3).\n"
            "Pick exactly one."
        )
    if args.random_trunk and args.freeze_front:
        p.error(
            "--random_trunk with --freeze_front would freeze RANDOM front layers "
            "(nothing is inherited in the trunk), which trains nothing meaningful. "
            "Use --random_trunk alone (train all layers, the A4-matched control)."
        )

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0

    # Seed on EVERY rank before any model construction so the fresh-tail random
    # init is reproducible and identical across DDP ranks.
    set_seed(args.seed)
    if is_main:
        logger.info(f"[seed] set_seed({args.seed}) on all ranks")

    # device selection. NOTE: params are fp32 (master weights) on BOTH cpu and
    # cuda; only the forward pass uses bf16 autocast (cuda only).
    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        use_cuda = False
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_cuda = True
    model_dtype = torch.float32  # fp32 master weights (do NOT cast to bf16)

    total_layers = args.keep_front_layers + args.n_fresh_layers
    if args.from_scratch:
        arm = f"scratch{total_layers}L"
    elif args.random_trunk:
        arm = f"randtrunk{total_layers}L_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"
    elif args.freeze_front:
        arm = f"frozen_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"
    else:
        arm = f"healing_front{args.keep_front_layers}+fresh{args.n_fresh_layers}"

    # ---- dry-run build validation (no base transplant) ----
    if args.dry_run_build:
        logger.info(f"[dry_run_build] olmo2 arm={arm} keep={args.keep_front_layers} "
                    f"fresh={args.n_fresh_layers} total={total_layers}")
        model, cfg, _ = build_olmo2_minimal(
            args.model_path, args.keep_front_layers, args.n_fresh_layers,
            model_dtype, transplant=False, is_main=True,
        )
        assert cfg.num_hidden_layers == total_layers
        n_layers_in_sd = len({k.split(".")[2] for k in model.state_dict()
                              if k.startswith("model.layers.")})
        assert n_layers_in_sd == total_layers, f"{n_layers_in_sd} != {total_layers}"
        expected_copied = N_NONLAYER_KEYS + N_TENSORS_PER_LAYER * args.keep_front_layers
        if args.n_fresh_layers > 0:
            ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, args.keep_front_layers)
        else:
            ln_all_ones, qnorm_all_ones, q_std = None, None, None
        logger.info(f"[dry_run_build] cfg.num_hidden_layers={cfg.num_hidden_layers} "
                    f"layers_in_sd={n_layers_in_sd} expected_copied_keys={expected_copied} "
                    f"fresh_post_attn_ln_all_ones={ln_all_ones} "
                    f"fresh_q_norm_all_ones={qnorm_all_ones} fresh_q_std={q_std} -> OK")
        logger.info("[dry_run_build] arch/init logic validated; exiting (no training).")
        return

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== OLMo-2 minimal-arch probe#2 [{arm}] ===")
        logger.info(f"device={device} dtype={model_dtype} (fp32 master weights) "
                    f"world_size={world_size} bs={args.batch_size} "
                    f"gaccum={args.grad_accumulation_steps} eff_bs={eff_bs} "
                    f"seq_len={args.seq_len} lr_fresh={args.lr} lr_inh={args.lr_inherited} "
                    f"max_steps={args.max_steps}")

    # ---- build model ----
    # When resuming we skip the base front-layer transplant: the resume ckpt
    # already holds ALL trained weights, so build the shell without touching the
    # base model, then overwrite below. (from_scratch never transplants.)
    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu",
                                 weights_only=False)
        if is_main:
            logger.info(f"[resume] loading ckpt {args.resume_from} "
                        f"(saved at step {resume_ckpt.get('step')}, "
                        f"has_optimizer={'optimizer_state' in resume_ckpt})")

    do_transplant = (not args.from_scratch) and (resume_ckpt is None)
    model, cfg, sanity = build_olmo2_minimal(
        args.model_path, args.keep_front_layers, args.n_fresh_layers,
        model_dtype, transplant=do_transplant, is_main=is_main,
        random_trunk=args.random_trunk,
    )
    if args.from_scratch and is_main and resume_ckpt is None:
        logger.info(f"[from_scratch] random-init {cfg.num_hidden_layers}-layer model "
                    f"(base weights IGNORED); training all layers")
    if args.random_trunk and is_main and resume_ckpt is None:
        logger.info(f"[random_trunk] random-init trunk of the {cfg.num_hidden_layers}-layer "
                    f"model (all model.layers.*), embed_tokens/model.norm/lm_head "
                    f"INHERITED from base; training all layers")

    if resume_ckpt is not None:
        missing, unexpected = model.load_state_dict(
            resume_ckpt["model_state"], strict=True)
        assert not missing and not unexpected, (
            f"[resume] model_state mismatch missing={missing[:4]} "
            f"unexpected={unexpected[:4]}")
        if is_main:
            logger.info(f"[resume] restored {len(resume_ckpt['model_state'])} "
                        f"model tensors (strict, fp32 master weights)")

    if args.freeze_front and not args.from_scratch:
        apply_freeze_front(model, args.keep_front_layers, is_main)

    # ---- step-0 checkpoint (initial training state) + exit -------------------
    # P0.2 reference point: the student EXACTLY as training constructs it above
    # (same do_transplant path -> same front-layer transplant + same fresh-block
    # Olmo2 post_init random init + fp32 master weights), saved in the IDENTICAL
    # checkpoint format the trainer's _save writes, so step0.pt strict-loads into
    # the eval harness like any step{N}.pt (same keys / shapes / tensor count).
    # We build a real AdamW over the same differential-LR param groups so
    # optimizer_state matches a mid-training ckpt's structure (fresh Adam moments
    # = the genuine step-0 state). rotate=False so writing step0.pt can NEVER
    # delete an existing trained step{N}.pt in output_dir. Runs BEFORE data / DDP
    # / the training loop. Single-process (CUDA_VISIBLE_DEVICES=0) is enough; the
    # model stays on CPU here (no .to(device)), which is fine for saving weights.
    if args.save_step0_and_exit:
        os.makedirs(args.output_dir, exist_ok=True)
        param_groups = build_param_groups(model, args, is_main)
        optimizer = build_optimizer(param_groups, args, is_main)
        _save(model, optimizer, args, 0, 0, cfg, final=False, rotate=False)
        n_tensors = len(model.state_dict())
        n = sum(pp.numel() for pp in model.parameters())
        logger.info(
            f"[save_step0] wrote {args.output_dir}/step0.pt arm={arm} "
            f"do_transplant={do_transplant} n_tensors={n_tensors} "
            f"num_hidden_layers={cfg.num_hidden_layers} params={n/1e9:.4f}B; "
            f"exiting before training (no data / DDP / loop)."
        )
        if ddp:
            dist.destroy_process_group()
        return

    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False

    if is_main:
        n = sum(pp.numel() for pp in model.parameters())
        n_tr = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
        logger.info(f"model params = {n/1e9:.4f}B (trainable {n_tr/1e9:.4f}B) "
                    f"num_hidden_layers={cfg.num_hidden_layers}")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "arm": arm,
                "model_family": "olmo2",
                "base_model_path": args.model_path,
                "keep_front_layers": args.keep_front_layers,
                "n_fresh_layers": args.n_fresh_layers,
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "vocab_size": cfg.vocab_size,
                "tie_word_embeddings": False,
                "freeze_front": bool(args.freeze_front),
                "from_scratch": bool(args.from_scratch),
                "random_trunk": bool(args.random_trunk),
                "seq_len": args.seq_len,
                "lr_fresh": args.lr,
                "lr_inherited": args.lr_inherited,
                "seed": args.seed,
                "n_params": n,
                "n_trainable": n_tr,
                "sanity": sanity,
            }, f, indent=2)

    if ddp:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank,
                    find_unused_parameters=False)

    # ---- data ----
    ds = NpyChunkDataset(args.data_path, args.seq_len)
    if args.max_rows and args.max_rows > 0:
        ds.arr = ds.arr[: args.max_rows]
    if is_main:
        logger.info(f"dataset rows={len(ds)} seq_len={ds.seq_len} from {args.data_path}")

    if ddp:
        # seed=args.seed is LOAD-BEARING -- do not delete as "redundant with
        # set_seed()/torch.manual_seed() above". DistributedSampler.__iter__ builds
        # its OWN generator (`g = torch.Generator(); g.manual_seed(self.seed + self.epoch)`)
        # and `self.seed` defaults to 0, so the global torch RNG cannot reach it.
        # Without this argument every --seed value yields a BYTE-IDENTICAL data order,
        # and "seed variance" collapses to fresh-block-init variance only.
        sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True, drop_last=True,
                            multiprocessing_context="fork" if use_cuda else None)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn, num_workers=0 if not use_cuda else 4,
                            pin_memory=use_cuda, drop_last=True,
                            multiprocessing_context="fork" if use_cuda else None)

    # ---- optimizer (differential-LR param groups; only params requiring grad) ----
    param_groups = build_param_groups(model, args, is_main)
    optimizer = build_optimizer(param_groups, args, is_main)

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0

    # ---- resume: optimizer state + global_step + epoch + RNG ----
    if resume_ckpt is not None:
        step = int(resume_ckpt.get("step", 0))
        epoch = int(resume_ckpt.get("epoch", 0))
        if "optimizer_state" in resume_ckpt:
            ckpt_optim = resume_ckpt["optimizer_state"]
            n_ckpt_groups = len(ckpt_optim["param_groups"])
            n_new_groups = len(optimizer.param_groups)
            if n_ckpt_groups == n_new_groups:
                # Normal path: group counts match, use standard load
                try:
                    optimizer.load_state_dict(ckpt_optim)
                    if is_main:
                        logger.info(f"[resume] optimizer state restored "
                                    f"({len(optimizer.state)} param states) -> "
                                    f"Adam momentum preserved")
                except (ValueError, KeyError) as e:
                    if is_main:
                        logger.warning(f"[resume] optimizer.load_state_dict failed "
                                       f"({e}); WARM-RESTART (Adam moments re-init)")
            elif n_ckpt_groups == 2 and n_new_groups == 4:
                # ---- Compatibility shim for keep10/keep12/keep8 ckpts ----
                # These ckpts were saved with a buggy _classify_param (no module.
                # prefix stripping) -> all params fell into inh_* -> 2 groups in ckpt.
                # HEAD builds 4 groups (fresh_decay/fresh_nodecay/inh_decay/inh_nodecay).
                # We remap by param-name using old scheme: group0=ndim>=2, group1=ndim<2,
                # both in model.named_parameters() iteration order.
                if is_main:
                    logger.info("[resume] ckpt has 2 groups, optimizer has 4 groups; "
                                "applying keep10/12/8 compatibility remap...")
                try:
                    ms = resume_ckpt["model_state"]
                    ms_keys = list(ms.keys())
                    ndim2_keys = [k for k in ms_keys if ms[k].ndim >= 2]
                    ndim1_keys = [k for k in ms_keys if ms[k].ndim < 2]
                    n2 = len(ndim2_keys)
                    # old scheme: group0 = ndim>=2 in ms_keys order,
                    #             group1 = ndim<2 in ms_keys order
                    old_name_to_idx = {k: i for i, k in enumerate(ndim2_keys)}
                    old_name_to_idx.update({k: n2 + i for i, k in enumerate(ndim1_keys)})
                    old_state = ckpt_optim["state"]

                    # Walk HEAD optimizer's param groups and fill optimizer.state by name
                    # model may be DDP-wrapped; strip module. to get bare names
                    root_model = model.module if hasattr(model, "module") else model
                    name_to_param = {n: p for n, p in root_model.named_parameters()}

                    restored = 0
                    for g in optimizer.param_groups:
                        for p in g["params"]:
                            # Find param name by tensor identity (data_ptr + shape)
                            matched_name = None
                            for n, mp in name_to_param.items():
                                if (mp is p or
                                        (mp.data_ptr() == p.data_ptr() and
                                         tuple(mp.shape) == tuple(p.shape))):
                                    matched_name = n
                                    break
                            if matched_name is not None and matched_name in old_name_to_idx:
                                old_i = old_name_to_idx[matched_name]
                                if old_i in old_state:
                                    optimizer.state[p] = {
                                        k: (v.to(p.device)
                                            if isinstance(v, torch.Tensor) else v)
                                        for k, v in old_state[old_i].items()
                                    }
                                    restored += 1

                    if is_main:
                        n_total = len(old_state)
                        logger.info(
                            f"[resume] optimizer state REMAPPED 2-group -> 4-group "
                            f"({restored}/{n_total} param states, Adam moments preserved)")
                        if restored < n_total * 0.9:
                            logger.warning(
                                f"[resume] WARNING: only {restored}/{n_total} "
                                f"states restored; check name matching")
                except Exception as e:  # noqa: BLE001
                    if is_main:
                        logger.warning(
                            f"[resume] optimizer remap failed ({e}); "
                            f"WARM-RESTART (Adam moments re-init)")
            else:
                if is_main:
                    logger.warning(
                        f"[resume] group count mismatch: ckpt={n_ckpt_groups} "
                        f"optimizer={n_new_groups}; WARM-RESTART (Adam moments re-init)")
        else:
            if is_main:
                logger.warning("[resume] ckpt has NO optimizer_state (old "
                               "model-only format) -> WARM-RESTART: Adam moments "
                               "re-init, LR resumes on cosine curve at ckpt step")
        rng = resume_ckpt.get("rng_state")
        if rng is not None:
            try:
                torch.set_rng_state(rng["torch"])
                if use_cuda and "cuda" in rng and \
                        len(rng["cuda"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:  # noqa: BLE001 - RNG restore is non-critical
                if is_main:
                    logger.warning(f"[resume] RNG restore skipped ({e})")
        if is_main:
            old_ms = resume_ckpt.get("max_steps", "?")
            lr_now = get_lr(step, args.warmup_steps, args.max_steps,
                            args.lr, args.min_lr)
            lr_inh_now = get_lr(step, args.warmup_steps, args.max_steps,
                                args.lr_inherited, args.min_lr_inherited)
            if isinstance(old_ms, int) and args.max_steps != old_ms:
                logger.info(f"[resume] EXTEND max_steps {old_ms} -> {args.max_steps}; "
                            f"cosine re-scaled to new horizon")
            logger.info(f"[resume] continue @ step={step} epoch={epoch} "
                        f"warmup={args.warmup_steps} max_steps={args.max_steps} "
                        f"lr_fresh(now)={lr_now:.3e} lr_inh(now)={lr_inh_now:.3e}")
            if step >= args.max_steps:
                logger.warning(f"[resume] step {step} >= max_steps "
                               f"{args.max_steps}: nothing to train. Did you "
                               f"forget to raise --max_steps?")
        del resume_ckpt
        gc.collect()

    # ---- data loader position ----
    if sampler is not None and epoch > 0:
        sampler.set_epoch(epoch)
        if is_main:
            logger.info(f"[resume] sampler.set_epoch({epoch}) "
                        f"(deterministic reshuffle for this epoch)")
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

        input_ids = batch["input_ids"].to(device, non_blocking=use_cuda)
        labels = batch["labels"].to(device, non_blocking=use_cuda)

        is_accum_boundary = (micro + 1) % args.grad_accumulation_steps == 0
        sync_ctx = model.no_sync() if (ddp and not is_accum_boundary) else _nullctx()
        with sync_ctx:
            if use_cuda:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = model(input_ids=input_ids, labels=labels)
                    loss = out.loss / args.grad_accumulation_steps
            else:
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accumulation_steps
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_cnt += 1
        micro += 1

        if is_accum_boundary:
            for g in optimizer.param_groups:
                g["lr"] = get_lr(step, args.warmup_steps, args.max_steps,
                                 g["base_lr"], g["min_lr"])
            gnorm = torch.nn.utils.clip_grad_norm_(
                [pp for pp in model.parameters() if pp.requires_grad], args.grad_clip)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

            if is_main and step % args.log_every == 0:
                avg = accum_loss / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
                lr_fresh = optimizer.param_groups[0]["lr"]
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg,20)):.2f} lr={lr_fresh:.2e} "
                            f"gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            if is_main and step % args.save_every == 0 and step > 0:
                _save(model, optimizer, args, step, epoch, cfg)

    if is_main:
        _save(model, optimizer, args, step, epoch, cfg, final=True)
        logger.info(f"DONE [{arm}] at step {step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
