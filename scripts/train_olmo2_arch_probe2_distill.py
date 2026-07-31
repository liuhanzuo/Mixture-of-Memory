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
    everything at a single LR.

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
import sys
import time

import torch
import torch.distributed as dist
import bitsandbytes as bnb  # 8-bit AdamW to fit keep14 train-all + teacher in H20 95GB
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from transformers import Olmo2Config, Olmo2ForCausalLM

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
    ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, keep_front_layers)

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
            f"fresh_q_norm_all_ones={qnorm_all_ones} fresh_q_std={q_std:.4f} "
            f"-> ALL 6 CHECKS PASS"
        )
    return sanity


def build_olmo2_minimal(base_path, keep_front_layers, n_fresh_layers, dtype,
                        transplant=True, is_main=True):
    """Build a (keep_front + n_fresh)-layer OLMo-2 model.

    (a) shrink the pretrained Olmo2 config to `keep_front + n_fresh` layers.
        (OLMo-2 has no `layer_types`; we reset it only if present, for safety.)
    (b) instantiate via Olmo2ForCausalLM(cfg) so post_init gives every layer,
        including the fresh tail, the correct Olmo2 init.
    (c) if transplant: overwrite front keep_front layers + embed + norm + lm_head
        with the pretrained base weights and run the sanity asserts.

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

    if transplant:
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


def _classify_param(name, keep_front_layers, from_scratch):
    """'fresh' (fresh tail layers + lm_head -> high LR) vs 'inherited' (front
    layers + embed + norm -> low LR). from_scratch -> everything 'fresh'."""
    if from_scratch:
        return "fresh"
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
    from_scratch: single 'fresh' bucket at args.lr."""
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
                logger.info(f"[optim] group {gname}: {n/1e6:.1f}M params "
                            f"base_lr={g['base_lr']:.2e} min_lr={g['min_lr']:.2e}")
    return param_groups


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def _save(model, optimizer, args, step, epoch, cfg, final=False):
    """Checkpoint = model weights + arch meta + optimizer state / step / epoch /
    max_steps / training args / RNG state (for clean resume). New keys are
    additive; old model-only evals ignore them, resume degrades to warm-restart."""
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
        "seq_len": args.seq_len,
    }, path)
    logger.info(f"saved {path}")

    # --- rolling retention: keep latest-2 + every-5000-step milestones -------
    # After writing the new step*.pt, keep (a) the 2 most-recent step ckpts and
    # (b) every 5000-step milestone (step5000/10000/...); delete the rest. This
    # bounds volume usage (the keep14 run was once killed mid-save on a full disk)
    # while preserving milestones for heal-curve analysis / resume, matching the
    # ckpt-rotation policy. final.pt is never rotated. Runs only where _save runs
    # (rank 0), and never removes the just-written ``path``.
    if not final:
        import glob as _glob
        import re as _re
        keep_abs = os.path.abspath(path)
        cks = []  # (step, abspath)
        for old in _glob.glob(os.path.join(args.output_dir, "step*.pt")):
            m = _re.search(r"step(\d+)\.pt$", os.path.basename(old))
            if m:
                cks.append((int(m.group(1)), os.path.abspath(old)))
        latest2 = {ap for _s, ap in sorted(cks, reverse=True)[:2]}
        for s, ap in cks:
            keep = (ap == keep_abs) or (ap in latest2) or (s % 5000 == 0)
            if not keep:
                try:
                    os.remove(ap)
                    logger.info(f"rotated old ckpt {ap}")
                except OSError as e:
                    logger.warning(f"could not remove old ckpt {ap}: {e}")


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
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume_from", type=str, default="",
                   help="path to a step{N}.pt / final.pt to resume from. Restores model "
                        "weights + (if present) optimizer state + global_step + epoch + RNG. "
                        "Old model-only ckpts degrade gracefully to a warm-restart.")
    p.add_argument("--max_rows", type=int, default=0, help=">0 to subset dataset (smoke)")
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--device", type=str, default="auto",
                   help="'auto' (cuda if available else cpu), 'cpu', or 'cuda'")
    p.add_argument("--dry_run_build", action="store_true",
                   help="build the model shell (no base transplant) + validate arch/init "
                        "logic, then exit. For CPU smoke without loading the base weights.")
    # ---- distillation args (keep14-distill: OLMo base 32L teacher → keep14 16L student) ----
    p.add_argument("--distill_teacher_model", type=str, default="",
                   help="path to teacher model (OLMo-2-7B base 32L). Empty = no distill (pure NTP).")
    p.add_argument("--distill_lambda", type=float, default=0.6,
                   help="weight on top-k KL(teacher||student) distill loss")
    p.add_argument("--teacher_topk", type=int, default=64,
                   help="top-k logits for KL (Hinton-style, teacher top-k positions)")
    args = p.parse_args()

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0

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
        ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, args.keep_front_layers)
        logger.info(f"[dry_run_build] cfg.num_hidden_layers={cfg.num_hidden_layers} "
                    f"layers_in_sd={n_layers_in_sd} expected_copied_keys={expected_copied} "
                    f"fresh_post_attn_ln_all_ones={ln_all_ones} "
                    f"fresh_q_norm_all_ones={qnorm_all_ones} fresh_q_std={q_std:.4f} -> OK")
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
    )
    if args.from_scratch and is_main and resume_ckpt is None:
        logger.info(f"[from_scratch] random-init {cfg.num_hidden_layers}-layer model "
                    f"(base weights IGNORED); training all layers")

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

    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    model.config.use_cache = False

    # ---- load frozen teacher (OLMo-2-7B base 32L) for distillation ----
    teacher_model = None
    if args.distill_teacher_model:
        from transformers import AutoModelForCausalLM
        if is_main:
            logger.info(f"[distill] loading teacher {args.distill_teacher_model} (frozen, eval)")
        teacher_model = AutoModelForCausalLM.from_pretrained(
            args.distill_teacher_model, dtype=torch.bfloat16).to(device)
        teacher_model.eval()
        for pp in teacher_model.parameters():
            pp.requires_grad = False
        if args.gradient_checkpointing:
            teacher_model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False})
        teacher_model.config.use_cache = False
        if is_main:
            logger.info(f"[distill] teacher loaded: {sum(p.numel() for p in teacher_model.parameters())/1e9:.4f}B "
                        f"lambda={args.distill_lambda} topk={args.teacher_topk}")

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
                "seq_len": args.seq_len,
                "lr_fresh": args.lr,
                "lr_inherited": args.lr_inherited,
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
        sampler = DistributedSampler(ds, shuffle=True)
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
    optimizer = bnb.optim.AdamW8bit(
        param_groups, lr=args.lr, betas=(0.9, 0.95), eps=1e-8,
    )

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_ntp = 0.0
    accum_kl = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0

    # ---- resume: optimizer state + global_step + epoch + RNG ----
    if resume_ckpt is not None:
        step = int(resume_ckpt.get("step", 0))
        epoch = int(resume_ckpt.get("epoch", 0))
        if "optimizer_state" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                if is_main:
                    logger.info(f"[resume] optimizer state restored "
                                f"({len(optimizer.state)} param states) -> "
                                f"Adam momentum preserved")
            except (ValueError, KeyError) as e:
                if is_main:
                    logger.warning(f"[resume] optimizer.load_state_dict failed "
                                   f"({e}); WARM-RESTART (Adam moments re-init)")
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
                    if teacher_model is not None:
                        # distill: student logits (no labels) + NTP + top-k KL
                        out = model(input_ids=input_ids)
                        s_logits = out.logits  # [B, T, V]
                        # NTP loss (shift)
                        shift_s = s_logits[..., :-1, :].contiguous()
                        shift_l = labels[..., 1:].contiguous()
                        ntp_loss = torch.nn.functional.cross_entropy(
                            shift_s.view(-1, shift_s.size(-1)), shift_l.view(-1))
                        # teacher forward (frozen, no_grad)
                        with torch.no_grad():
                            t_out = teacher_model(input_ids=input_ids)
                            t_logits = t_out.logits  # [B, T, V]
                            shift_t = t_logits[..., :-1, :].contiguous()
                        # top-k KL(teacher || student) on teacher top-k positions
                        k = min(args.teacher_topk, shift_t.size(-1))
                        t_vals, t_idx = shift_t.topk(k, dim=-1)  # [B,T-1,k]
                        s_vals = shift_s.gather(-1, t_idx)       # [B,T-1,k]
                        t_logp = t_vals.log_softmax(dim=-1)       # [B,T-1,k]
                        s_logp = s_vals.log_softmax(dim=-1)       # [B,T-1,k]
                        t_p = t_logp.exp()
                        kl_loss = (t_p * (t_logp - s_logp)).sum(dim=-1).mean()
                        loss = (ntp_loss + args.distill_lambda * kl_loss) / args.grad_accumulation_steps
                        ntp_item = ntp_loss.item(); kl_item = kl_loss.item()
                    else:
                        out = model(input_ids=input_ids, labels=labels)
                        loss = out.loss / args.grad_accumulation_steps
                        ntp_item = out.loss.item(); kl_item = 0.0
            else:
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss / args.grad_accumulation_steps
                ntp_item = out.loss.item(); kl_item = 0.0
            loss.backward()
        accum_loss += loss.item() * args.grad_accumulation_steps
        accum_ntp += ntp_item
        accum_kl += kl_item
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
                avg_ntp = accum_ntp / max(accum_cnt, 1)
                avg_kl = accum_kl / max(accum_cnt, 1)
                dt = time.time() - t0
                mem = torch.cuda.max_memory_allocated() / 1e9 if use_cuda else 0.0
                lr_fresh = optimizer.param_groups[0]["lr"]
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg,20)):.2f} "
                            f"ntp={avg_ntp:.4f} ntp_ppl={math.exp(min(avg_ntp,20)):.2f} "
                            f"kl={avg_kl:.4f} lr={lr_fresh:.2e} "
                            f"gnorm={float(gnorm):.2f} {dt/args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_ntp = 0.0
                accum_kl = 0.0
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
