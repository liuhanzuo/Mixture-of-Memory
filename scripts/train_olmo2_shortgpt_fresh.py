#!/usr/bin/env python3
"""ShortGPT-style arbitrary-index transplant + FRESH tail heal (Paper B P0.5).

Structural-isolation control for the minimal-architecture thesis. Combines the
two existing OLMo-2 trainers:

  * scripts/train_olmo2_shortgpt.py -- keep an ARBITRARY set of base decoder
    layers (`keep_layer_indices`, sorted ascending, compacted to 0..K_keep-1).
  * scripts/train_olmo2_arch_probe2.py -- append `n_fresh_layers` FRESH
    Olmo2-init decoder layers as a re-grown NTP tail, and heal with a two-bucket
    DIFFERENTIAL LR (inherited layers + embed + norm + lm_head at the low
    inherited LR; the fresh tail at the high fresh LR).

Construction (P0.5 Arm B: keep [0..12, 31] = 14 layers + 2 fresh = 16 total):
  (a) cfg = Olmo2Config.from_pretrained(base); cfg.num_hidden_layers = K_keep +
      n_fresh. model = Olmo2ForCausalLM(cfg) -> post_init gives EVERY layer,
      including the fresh tail, the correct Olmo2 init (never hand-build a
      DecoderLayer).
  (b) transplant the `keep_layer_indices` base layers (ascending) into
      model.layers[0..K_keep-1] + embed_tokens + model.norm + lm_head from the
      base; model.layers[K_keep..K-1] stay at their fresh Olmo2 random init.
  (c) heal on Dolmino with the EXACT keep14 recipe (seq_len 2048, effective batch
      128, gradient checkpointing, fp32 master weights, cosine, warmup 150),
      differential LR: inherited 2e-5, fresh 1e-4.

Why the two-bucket LR here maps lm_head to INHERITED (not fresh, unlike the
keep-front arm): lm_head IS transplanted from the base (untied embeddings), so it
is inherited weight and heals at the low inherited LR; only the freshly-init tail
layers get the high fresh LR. (P0.5 spec, 2026-08-02.)

fp32 MASTER WEIGHTS: params stay fp32 (do NOT model.to(bf16)); forward under bf16
autocast, AdamW states fp32. Same anti-catastrophic-forgetting knob as both
sibling arms.

Checkpoints are raw state_dict (+ arch meta). The eval harness
(scripts/eval_olmo2_probe2_ppl.py::build_pruned_shell) rebuilds a shell of size
keep_front_layers + n_fresh_layers, so the ckpt records keep_front_layers =
len(keep_layer_indices) (the count of inherited layers = eval shell inherited
size) and n_fresh_layers, giving keep_front + fresh = num_hidden_layers = 16.
keep_layer_indices records the TRUE inherited identity ([0..12, 31]).
"""
from __future__ import annotations

import argparse
import gc
import glob
import json
import logging
import math
import os
import random
import re
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

# Reuse (do NOT modify) the shared data loading / cosine schedule / null-context.
from train_semantic_bottleneck_1b import (  # noqa: E402
    NpyChunkDataset,
    collate_fn,
    get_lr,
    _nullctx,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# OLMo-2 layout: 11 tensors per decoder layer (POST-norm, QK-norm, untied embeds),
# 3 non-layer keys (embed_tokens / norm / lm_head).
N_TENSORS_PER_LAYER = 11
N_NONLAYER_KEYS = 3


def set_seed(seed: int):
    """Seed python/numpy/torch (CPU + all CUDA devices) for reproducible +
    DDP-consistent fresh-layer init. Called on every rank BEFORE model build."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# model construction (arbitrary kept indices -> compacted 0..K_keep-1, then a
# fresh Olmo2-init tail of n_fresh layers at K_keep..K-1)
# ---------------------------------------------------------------------------
def parse_indices(spec: str):
    """'0,2,5,...' -> sorted unique list of ints."""
    idx = sorted({int(x) for x in spec.split(",") if x.strip() != ""})
    if not idx:
        raise ValueError(f"empty keep_layer_indices from {spec!r}")
    return idx


def _remap_state_dict(base_sd, keep_indices):
    """Build the transplant state_dict for the inherited layers: base layer
    keep_indices[j] -> new layer position j (j in 0..K_keep-1). Non-layer keys
    (embed/norm/lm_head) verbatim. Fresh tail layers (>= K_keep) are NOT included
    -> they remain at their Olmo2 random init (strict=False load below)."""
    pos_of = {src: j for j, src in enumerate(keep_indices)}
    new_sd = {}
    for k, v in base_sd.items():
        if k.startswith("model.layers."):
            src = int(k.split(".")[2])
            if src in pos_of:
                parts = k.split(".")
                parts[2] = str(pos_of[src])
                new_sd[".".join(parts)] = v
            # dropped layer -> skip
        else:
            new_sd[k] = v  # model.embed_tokens.weight / model.norm.weight / lm_head.weight
    return new_sd


def _assert_fresh_init(model, fresh_layer_ids):
    """Fresh tail layers must retain proper Olmo2 init after transplant. OLMo-2
    is POST-norm (no input_layernorm); the first RMSNorm in a layer is
    post_attention_layernorm. For EVERY fresh layer id, checks:
    post_attention_layernorm.weight all-ones, q_norm.weight all-ones (QK-norm),
    q_proj.weight std ~= 0.02 (initializer_range). Returns
    (ln_all_ones, qnorm_all_ones, q_std) for the FIRST fresh layer (summary)."""
    sd = model.state_dict()
    first_ln = first_qnorm = first_qstd = None
    for lid in fresh_layer_ids:
        fresh_ln = sd[f"model.layers.{lid}.post_attention_layernorm.weight"]
        ln_all_ones = bool(torch.all(fresh_ln == 1.0).item())
        fresh_qnorm = sd[f"model.layers.{lid}.self_attn.q_norm.weight"]
        qnorm_all_ones = bool(torch.all(fresh_qnorm == 1.0).item())
        fresh_q = sd[f"model.layers.{lid}.self_attn.q_proj.weight"]
        q_std = fresh_q.float().std().item()
        assert ln_all_ones, (
            f"fresh layer {lid} post_attention_layernorm not all-ones "
            f"(min={fresh_ln.min().item()}, max={fresh_ln.max().item()})")
        assert qnorm_all_ones, (
            f"fresh layer {lid} q_norm not all-ones "
            f"(min={fresh_qnorm.min().item()}, max={fresh_qnorm.max().item()})")
        assert 0.01 < q_std < 0.04, (
            f"fresh layer {lid} q_proj.weight std={q_std:.4f} not ~0.02 -> wrong init")
        if first_ln is None:
            first_ln, first_qnorm, first_qstd = ln_all_ones, qnorm_all_ones, q_std
    return first_ln, first_qnorm, first_qstd


def transplant_indices_fresh(model, base_path, keep_indices, n_fresh, dtype, is_main):
    """Load the pretrained base OLMo-2, transplant the `keep_indices` decoder
    layers (compacted to 0..K_keep-1) + embed + norm + lm_head into `model`, and
    leave the n_fresh tail layers (K_keep..K-1) at their fresh Olmo2 init. Runs
    all sanity asserts + the fresh-init assert. Raises on any failure. Returns
    a sanity dict."""
    base = Olmo2ForCausalLM.from_pretrained(
        base_path, torch_dtype=dtype, local_files_only=True)
    base_num_layers = base.config.num_hidden_layers
    for i in keep_indices:
        assert 0 <= i < base_num_layers, (
            f"keep index {i} out of range [0,{base_num_layers})")
    base_sd = base.state_dict()
    new_sd = _remap_state_dict(base_sd, keep_indices)

    missing, unexpected = model.load_state_dict(new_sd, strict=False)

    K_keep = len(keep_indices)
    fresh_layer_ids = list(range(K_keep, K_keep + n_fresh))

    # --- assert 1: no unexpected keys (everything transplanted exists in model) ---
    assert unexpected == [], (
        f"[sanity1] unexpected keys when transplanting: {unexpected[:8]}")

    # --- assert 2: the ONLY missing keys are the fresh tail layers ---
    missing_layer_ids = set()
    bad_missing = []
    for mk in missing:
        if mk.startswith("model.layers."):
            missing_layer_ids.add(int(mk.split(".")[2]))
        else:
            bad_missing.append(mk)
    assert not bad_missing, f"[sanity2] non-layer keys unexpectedly missing: {bad_missing}"
    assert missing_layer_ids == set(fresh_layer_ids), (
        f"[sanity2] missing layer-ids {sorted(missing_layer_ids)} != "
        f"fresh set {fresh_layer_ids}")

    # --- assert 3: number of transplanted keys == 3 + 11*K_keep ---
    expected = N_NONLAYER_KEYS + N_TENSORS_PER_LAYER * K_keep
    assert len(new_sd) == expected, (
        f"[sanity3] transplanted {len(new_sd)} keys != expected {expected} "
        f"(={N_NONLAYER_KEYS}+{N_TENSORS_PER_LAYER}*{K_keep})")

    # --- assert 4: EVERY inherited layer j matches base layer keep_indices[j],
    #     and embed/norm/lm_head match exactly -> max|model - base| == 0 ---
    model_sd = model.state_dict()
    max_diff = 0.0
    per_layer_max = []
    for j, src in enumerate(keep_indices):
        lmax = 0.0
        for t in ("self_attn.q_proj.weight", "self_attn.k_proj.weight",
                  "self_attn.v_proj.weight", "self_attn.o_proj.weight",
                  "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                  "mlp.gate_proj.weight", "mlp.up_proj.weight",
                  "mlp.down_proj.weight", "post_attention_layernorm.weight",
                  "post_feedforward_layernorm.weight"):
            nk = f"model.layers.{j}.{t}"
            bk = f"model.layers.{src}.{t}"
            d = (model_sd[nk].float() - base_sd[bk].float()).abs().max().item()
            lmax = max(lmax, d)
        per_layer_max.append(lmax)
        max_diff = max(max_diff, lmax)
    for nlk in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"):
        d = (model_sd[nlk].float() - base_sd[nlk].float()).abs().max().item()
        max_diff = max(max_diff, d)
    assert max_diff == 0.0, (
        f"[sanity4] transplant max|model_param - base| = {max_diff:.3e} != 0.0")

    # --- fresh-init assert (tail layers untouched, correct Olmo2 init) ---
    ln_all_ones, qnorm_all_ones, q_std = _assert_fresh_init(model, fresh_layer_ids)

    del base, base_sd, new_sd
    gc.collect()

    sanity = {
        "transplanted": True,
        "base_num_layers": base_num_layers,
        "keep_layer_indices": keep_indices,
        "n_kept": K_keep,
        "n_fresh_layers": n_fresh,
        "fresh_layer_ids": fresh_layer_ids,
        "n_transplanted_keys": expected,
        "transplant_max_abs_diff": max_diff,
        "per_new_layer_max_abs_diff": per_layer_max,
        "fresh_post_attention_layernorm_all_ones": ln_all_ones,
        "fresh_q_norm_all_ones": qnorm_all_ones,
        "fresh_q_proj_std": q_std,
    }
    if is_main:
        logger.info(
            f"[transplant] kept {K_keep} layers {keep_indices} (compacted "
            f"0..{K_keep - 1}) + embed/norm/lm_head from a {base_num_layers}-layer "
            f"base; fresh tail layer-ids {fresh_layer_ids} left at Olmo2 init")
        logger.info(
            f"[sanity] unexpected=0 | missing=fresh {fresh_layer_ids} | "
            f"transplanted_keys={expected} | max|model-base|={max_diff:.3e} (exact) "
            f"| fresh_post_attn_ln_all_ones={ln_all_ones} "
            f"fresh_q_norm_all_ones={qnorm_all_ones} fresh_q_std={q_std} "
            f"-> ALL CHECKS PASS")
    return sanity


def build_shortgpt_fresh_model(base_path, keep_indices, n_fresh, dtype,
                               transplant=True, is_main=True):
    """Build a (len(keep_indices) + n_fresh)-layer OLMo-2 and (optionally)
    transplant the selected base layers into the front, leaving the fresh tail
    at its Olmo2 init. Returns (model, cfg, sanity)."""
    cfg = Olmo2Config.from_pretrained(base_path, local_files_only=True)
    K_keep = len(keep_indices)
    K = K_keep + n_fresh
    cfg.num_hidden_layers = K
    if getattr(cfg, "layer_types", None) is not None:
        cfg.layer_types = ["full_attention"] * K
        assert len(cfg.layer_types) == K
    model = Olmo2ForCausalLM(cfg).to(dtype)
    if transplant:
        sanity = transplant_indices_fresh(model, base_path, keep_indices, n_fresh,
                                          dtype, is_main)
    else:
        sanity = {"transplanted": False, "keep_layer_indices": keep_indices,
                  "n_fresh_layers": n_fresh}
    return model, cfg, sanity


def _classify_param(name, n_kept):
    """Two buckets. 'fresh' = fresh tail layers (index >= n_kept) -> high LR.
    'inherited' = transplanted layers (index < n_kept) + embed + norm + lm_head
    -> low inherited LR. lm_head is INHERITED here (it is transplanted from the
    base, untied embeddings), unlike the keep-front arm."""
    if name.startswith("model.layers."):
        lid = int(name.split(".")[2])
        return "fresh" if lid >= n_kept else "inherited"
    # lm_head.weight, model.embed_tokens.weight, model.norm.weight -> inherited
    return "inherited"


def build_param_groups(model, args, n_kept, is_main):
    """Differential-LR param groups (also splitting weight-decay by ndim).

    fresh (tail layers): base_lr=args.lr_fresh, min_lr=args.min_lr_fresh.
    inherited (transplanted layers + embed + norm + lm_head): base_lr=
        args.lr_inherited, min_lr=args.min_lr_inherited."""
    specs = {
        "fresh_decay":   {"params": [], "weight_decay": args.weight_decay,
                          "base_lr": args.lr_fresh, "min_lr": args.min_lr_fresh},
        "fresh_nodecay": {"params": [], "weight_decay": 0.0,
                          "base_lr": args.lr_fresh, "min_lr": args.min_lr_fresh},
        "inh_decay":     {"params": [], "weight_decay": args.weight_decay,
                          "base_lr": args.lr_inherited, "min_lr": args.min_lr_inherited},
        "inh_nodecay":   {"params": [], "weight_decay": 0.0,
                          "base_lr": args.lr_inherited, "min_lr": args.min_lr_inherited},
    }
    for name, pp in model.named_parameters():
        if not pp.requires_grad:
            continue
        cls = _classify_param(name, n_kept)
        prefix = "fresh" if cls == "fresh" else "inh"
        key = f"{prefix}_decay" if pp.ndim >= 2 else f"{prefix}_nodecay"
        specs[key]["params"].append(pp)
    param_groups = [g for g in specs.values() if g["params"]]
    if is_main:
        for gname, g in specs.items():
            n = sum(p.numel() for p in g["params"])
            if n > 0:
                logger.info(f"[optim] group {gname}: {n / 1e6:.1f}M params "
                            f"base_lr={g['base_lr']:.2e} min_lr={g['min_lr']:.2e}")
    return param_groups


# ---------------------------------------------------------------------------
# save (records keep_front_layers = len(keep_indices) so the eval harness rebuilds
# a keep_front + n_fresh = 16-layer shell; keep_layer_indices = true identity)
# ---------------------------------------------------------------------------
def _save(model, optimizer, args, step, epoch, cfg, keep_indices, n_fresh,
          protect_steps, final=False):
    root = model.module if hasattr(model, "module") else model
    name = "final" if final else f"step{step}"
    path = os.path.join(args.output_dir, f"{name}.pt")
    rng = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        rng["cuda"] = torch.cuda.get_rng_state_all()
    K_keep = len(keep_indices)
    torch.save({
        "model_state": root.state_dict(),
        "step": step,
        "optimizer_state": optimizer.state_dict(),
        "epoch": epoch,
        "max_steps": args.max_steps,
        "warmup_steps": args.warmup_steps,
        "train_args": vars(args),
        "rng_state": rng,
        # arch descriptors: eval rebuilds a (keep_front_layers + n_fresh_layers)-
        # layer shell = 14 + 2 = 16, then strict-loads this state_dict.
        "model_family": "olmo2",
        "base_model_path": args.model_path,
        "keep_front_layers": K_keep,          # inherited layer count = eval shell inh size
        "n_fresh_layers": n_fresh,
        "keep_layer_indices": keep_indices,   # TRUE inherited identity ([0..12,31])
        "arm": "shortgpt_fresh",
        "num_hidden_layers": cfg.num_hidden_layers,
        "hidden_size": cfg.hidden_size,
        "vocab_size": cfg.vocab_size,
        "tie_word_embeddings": False,
        "freeze_front": False,
        "from_scratch": False,
        "seq_len": args.seq_len,
    }, path)
    logger.info(f"saved {path}")

    # rolling retention: keep latest-2 + every-5000 milestones + protected steps
    # (step0 + extra_save_steps). final.pt is never rotated; never remove `path`.
    if not final:
        keep_abs = os.path.abspath(path)
        cks = []
        for old in glob.glob(os.path.join(args.output_dir, "step*.pt")):
            m = re.search(r"step(\d+)\.pt$", os.path.basename(old))
            if m:
                cks.append((int(m.group(1)), os.path.abspath(old)))
        latest2 = {ap for _s, ap in sorted(cks, reverse=True)[:2]}
        for s, ap in cks:
            keep = (ap == keep_abs) or (ap in latest2) or (s % 5000 == 0) \
                or (s in protect_steps)
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
                   default="/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B")
    p.add_argument("--keep_layer_indices", type=str, default="",
                   help="comma-separated original layer ids to KEEP, e.g. '0,...,12,31'")
    p.add_argument("--selection_json", type=str, default="",
                   help="selection JSON (reads kept_layer_indices); "
                        "--keep_layer_indices overrides it if both given")
    p.add_argument("--n_fresh_layers", type=int, default=0,
                   help="number of FRESH Olmo2-init tail layers to re-grow (P0.5 Arm B=2)")
    # differential LR: inherited (transplant + embed + norm + lm_head) low; fresh tail high.
    p.add_argument("--lr_inherited", type=float, default=2e-5,
                   help="LR for inherited layers + embed + norm + lm_head")
    p.add_argument("--min_lr_inherited", type=float, default=2e-6)
    p.add_argument("--lr_fresh", type=float, default=1e-4,
                   help="LR for the fresh tail layers")
    p.add_argument("--min_lr_fresh", type=float, default=1e-5)
    p.add_argument("--max_steps", type=int, default=200000)
    p.add_argument("--seq_len", type=int, default=2048)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accumulation_steps", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=150)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_every", type=int, default=5000)
    p.add_argument("--extra_save_steps", type=str, default="50000,100000,150000",
                   help="comma steps to force-save (protected from rotation)")
    p.add_argument("--log_every", type=int, default=20)
    p.add_argument("--resume_from", type=str, default="")
    p.add_argument("--max_rows", type=int, default=0)
    p.add_argument("--gradient_checkpointing", type=int, default=1)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--seed", type=int, default=42,
                   help="RNG seed (python/numpy/torch/CUDA), set on every rank "
                        "before model build for reproducible + DDP-consistent init")
    p.add_argument("--dry_run_build", action="store_true",
                   help="build the (K_keep+n_fresh)-layer model shell WITH the base "
                        "transplant, validate arch + transplant + fresh-init asserts, "
                        "then exit (no data / no training / no big ckpt saved)")
    args = p.parse_args()

    # ---- resolve keep_layer_indices ----
    if args.keep_layer_indices:
        keep_indices = parse_indices(args.keep_layer_indices)
    elif args.selection_json:
        with open(args.selection_json) as f:
            sel = json.load(f)
        keep_indices = sorted(int(i) for i in sel["kept_layer_indices"])
    else:
        raise ValueError("provide --keep_layer_indices or --selection_json")
    K_keep = len(keep_indices)
    n_fresh = args.n_fresh_layers
    K = K_keep + n_fresh

    extra_steps = {int(x) for x in args.extra_save_steps.split(",") if x.strip()}
    protect_steps = {0} | extra_steps

    ddp = "RANK" in os.environ
    if ddp:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
    else:
        rank, world_size, local_rank = 0, 1, 0
    is_main = rank == 0

    set_seed(args.seed)  # reproducible + DDP-consistent fresh-tail init on every rank

    if args.device == "cpu" or (args.device == "auto" and not torch.cuda.is_available()):
        device = torch.device("cpu")
        use_cuda = False
    else:
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        use_cuda = True
    model_dtype = torch.float32  # fp32 master weights

    arm = f"shortgpt_keep{K_keep}+fresh{n_fresh}"

    # ---- dry-run build (WITH transplant + fresh-init asserts) ----
    if args.dry_run_build:
        logger.info(f"[dry_run_build] {arm} keep_indices={keep_indices} "
                    f"K_keep={K_keep} n_fresh={n_fresh} total={K}")
        model, cfg, sanity = build_shortgpt_fresh_model(
            args.model_path, keep_indices, n_fresh, model_dtype,
            transplant=True, is_main=True)
        assert cfg.num_hidden_layers == K
        n_layers_in_sd = len({k.split(".")[2] for k in model.state_dict()
                              if k.startswith("model.layers.")})
        assert n_layers_in_sd == K, f"{n_layers_in_sd} != {K}"
        logger.info(f"[dry_run_build] cfg.num_hidden_layers={cfg.num_hidden_layers} "
                    f"layers_in_sd={n_layers_in_sd} transplant_max_abs_diff="
                    f"{sanity.get('transplant_max_abs_diff')} "
                    f"fresh_ids={sanity.get('fresh_layer_ids')} "
                    f"fresh_post_attn_ln_all_ones="
                    f"{sanity.get('fresh_post_attention_layernorm_all_ones')} "
                    f"fresh_q_norm_all_ones={sanity.get('fresh_q_norm_all_ones')} "
                    f"fresh_q_std={sanity.get('fresh_q_proj_std')} -> OK (no training)")
        return

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)
        eff_bs = args.batch_size * args.grad_accumulation_steps * world_size
        logger.info(f"=== OLMo-2 ShortGPT+fresh prune-heal [{arm}] ===")
        logger.info(f"keep_layer_indices={keep_indices} n_fresh={n_fresh}")
        logger.info(f"device={device} dtype={model_dtype} (fp32 master weights) "
                    f"world_size={world_size} bs={args.batch_size} "
                    f"gaccum={args.grad_accumulation_steps} eff_bs={eff_bs} "
                    f"seq_len={args.seq_len} lr_inh={args.lr_inherited} "
                    f"lr_fresh={args.lr_fresh} max_steps={args.max_steps}")

    resume_ckpt = None
    if args.resume_from:
        resume_ckpt = torch.load(args.resume_from, map_location="cpu",
                                 weights_only=False)
        if is_main:
            logger.info(f"[resume] loading {args.resume_from} "
                        f"(step {resume_ckpt.get('step')}, "
                        f"has_optimizer={'optimizer_state' in resume_ckpt})")

    do_transplant = resume_ckpt is None
    model, cfg, sanity = build_shortgpt_fresh_model(
        args.model_path, keep_indices, n_fresh, model_dtype,
        transplant=do_transplant, is_main=is_main)

    if resume_ckpt is not None:
        missing, unexpected = model.load_state_dict(
            resume_ckpt["model_state"], strict=True)
        assert not missing and not unexpected, (
            f"[resume] model_state mismatch missing={missing[:4]} "
            f"unexpected={unexpected[:4]}")
        if is_main:
            logger.info(f"[resume] restored {len(resume_ckpt['model_state'])} "
                        f"tensors (strict, fp32 master weights)")

    model = model.to(device)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.config.use_cache = False

    if is_main:
        n = sum(pp.numel() for pp in model.parameters())
        n_tr = sum(pp.numel() for pp in model.parameters() if pp.requires_grad)
        logger.info(f"model params = {n / 1e9:.4f}B (trainable {n_tr / 1e9:.4f}B) "
                    f"num_hidden_layers={cfg.num_hidden_layers}")
        with open(os.path.join(args.output_dir, "arch_meta.json"), "w") as f:
            json.dump({
                "arm": arm,
                "model_family": "olmo2",
                "base_model_path": args.model_path,
                "keep_layer_indices": keep_indices,
                "keep_front_layers": K_keep,   # inherited count = eval shell inh size
                "n_fresh_layers": n_fresh,
                "num_hidden_layers": cfg.num_hidden_layers,
                "hidden_size": cfg.hidden_size,
                "vocab_size": cfg.vocab_size,
                "tie_word_embeddings": False,
                "freeze_front": False,
                "from_scratch": False,
                "seq_len": args.seq_len,
                "lr_inherited": args.lr_inherited,
                "lr_fresh": args.lr_fresh,
                "selection_json": args.selection_json,
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
        sampler = DistributedSampler(ds, shuffle=True)
        loader = DataLoader(ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_fn, num_workers=4, pin_memory=True,
                            drop_last=True,
                            multiprocessing_context="fork" if use_cuda else None)
    else:
        sampler = None
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                            collate_fn=collate_fn,
                            num_workers=0 if not use_cuda else 4,
                            pin_memory=use_cuda, drop_last=True,
                            multiprocessing_context="fork" if use_cuda else None)

    # ---- optimizer ----
    param_groups = build_param_groups(model, args, K_keep, is_main)
    optimizer = torch.optim.AdamW(
        param_groups, lr=args.lr_fresh, betas=(0.9, 0.95), eps=1e-8)

    model.train()
    step = 0
    micro = 0
    accum_loss = 0.0
    accum_cnt = 0
    optimizer.zero_grad(set_to_none=True)
    epoch = 0

    # ---- resume state ----
    if resume_ckpt is not None:
        step = int(resume_ckpt.get("step", 0))
        epoch = int(resume_ckpt.get("epoch", 0))
        if "optimizer_state" in resume_ckpt:
            try:
                optimizer.load_state_dict(resume_ckpt["optimizer_state"])
                if is_main:
                    logger.info(f"[resume] optimizer state restored "
                                f"({len(optimizer.state)} param states)")
            except (ValueError, KeyError) as e:
                if is_main:
                    logger.warning(f"[resume] optimizer.load_state_dict failed "
                                   f"({e}); WARM-RESTART")
        else:
            if is_main:
                logger.warning("[resume] no optimizer_state -> WARM-RESTART")
        rng = resume_ckpt.get("rng_state")
        if rng is not None:
            try:
                torch.set_rng_state(rng["torch"])
                if use_cuda and "cuda" in rng and \
                        len(rng["cuda"]) == torch.cuda.device_count():
                    torch.cuda.set_rng_state_all(rng["cuda"])
            except Exception as e:  # noqa: BLE001
                if is_main:
                    logger.warning(f"[resume] RNG restore skipped ({e})")
        if is_main:
            lr_inh_now = get_lr(step, args.warmup_steps, args.max_steps,
                                args.lr_inherited, args.min_lr_inherited)
            lr_fresh_now = get_lr(step, args.warmup_steps, args.max_steps,
                                  args.lr_fresh, args.min_lr_fresh)
            logger.info(f"[resume] continue @ step={step} epoch={epoch} "
                        f"max_steps={args.max_steps} lr_inh(now)={lr_inh_now:.3e} "
                        f"lr_fresh(now)={lr_fresh_now:.3e}")
        resume_ckpt = None
        gc.collect()

    # ---- step0 checkpoint: pruned-but-not-healed (fresh run only) ----
    if is_main and step == 0:
        _save(model, optimizer, args, 0, epoch, cfg, keep_indices, n_fresh,
              protect_steps)
        logger.info("[step0] saved pruned+fresh-tail-but-not-healed ckpt "
                    "(heal-free eval point)")

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
                lr_now = optimizer.param_groups[0]["lr"]
                logger.info(f"[step {step:5d}/{args.max_steps}] loss={avg:.4f} "
                            f"ppl={math.exp(min(avg, 20)):.2f} lr={lr_now:.2e} "
                            f"gnorm={float(gnorm):.2f} {dt / args.log_every:.2f}s/step "
                            f"maxmem={mem:.1f}GB")
                accum_loss = 0.0
                accum_cnt = 0
                t0 = time.time()

            save_now = (step % args.save_every == 0) or (step in extra_steps)
            if is_main and save_now and step > 0:
                _save(model, optimizer, args, step, epoch, cfg, keep_indices,
                      n_fresh, protect_steps)

    if is_main:
        _save(model, optimizer, args, step, epoch, cfg, keep_indices, n_fresh,
              protect_steps, final=True)
        logger.info(f"DONE [{arm}] at step {step}")
    if ddp:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
