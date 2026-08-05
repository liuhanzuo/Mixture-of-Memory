#!/usr/bin/env python
"""QCMem self-distillation on the Tencent Hunyuan **Hy3** (``hy_v3``) 80-layer
sparse-MoE backbone (597 GB bf16), sharded across the local 8 GPUs (2026-07-12).

This is the Hy3 analogue of ``scripts/train_qcmem_distill.py`` (Qwen3-8B). The
distillation *recipe* is unchanged from the proven Qwen r32 run
(``outputs/qcmem_distill_qwen_j12_r32_4k``): teacher = QCMem read at ``j=0`` (RAG
upper bound = full recompute of the selected chunks with the query present),
student = QCMem read at ``j=--resume_j`` (default **32**, the Hy3 split-j found by
the j-sweep, ``versions/v_qcmem_hy3_port.md``) with LoRA on ``layers[j:]`` only.
Loss = bidirectional top-k KL on the query-tail tokens over pure PG19 natural text
(``data/pg19_train.jsonl`` — NO babilong / NO needles / NO synthetic long-context,
red line). Goal: push the zero-shot 1.25-1.5x LM tax at j=32 toward 1.0, mirroring
the 8B result (1000-step PG19 self-distill lifted every qa cell).

Why a SEPARATE trainer (not the Qwen file with a flag)
------------------------------------------------------
The 597 GB Hy3 does NOT fit on one GPU, so it must be loaded ONCE with
``device_map="auto"`` and pipelined across all 8 L20A. That is fundamentally
different parallelism from the Qwen trainer, which replicates an 8B model on every
rank and runs 8-way DDP with explicit grad all-reduce. Here there is exactly ONE
model instance and ONE process — no DDP, no torchrun, no grad all-reduce. Every
forward already exercises all 8 cards (the model is sharded), so a single process
saturates the node. The device-crossing WRITE/READ loop lives in
``QCMemHy3Model`` (``src/memory/qcmem/qcmem_hy3.py``): it hops the residual-stream
hidden + mask + RoPE onto each layer's GPU before the call, and its ``read_core``
is grad-bearing (LoRA in ``layers[j:]`` trains through it).

Memory model
------------
* ONE model copy (597 GB) sharded over 8x183 GB = ~75 GB weights / GPU, leaving
  ~108 GB / GPU for activations. Teacher (no_grad, freed) and student writes
  (no_grad) leave no graph. Only the student READ over ``layers[j:]`` (48 layers)
  retains activations for backward, and only the ~tens-of-M LoRA params carry grad
  (the frozen backbone + the 192-expert MoE ``nn.Parameter`` weights do NOT).
* The MoE experts (``HYV3Experts.gate_up_proj`` / ``down_proj``) are raw 3D
  ``nn.Parameter`` tensors, NOT ``nn.Linear`` submodules, so PEFT's LoRA cannot and
  does NOT touch them — the ``gate_proj/up_proj/down_proj`` targets only match the
  per-layer dense ``shared_experts`` MLP + the dense layer-0 MLP; ``q/k/v/o_proj``
  match ``self_attn``. No param explosion, no collision.
* If the grad-bearing read OOMs, pass ``--gradient_checkpointing`` (honoured by
  ``QCMemHy3Model._run_layers``) and/or shrink ``--n_ctx``.

Correctness: the depth partition is exact on Hy3 — the real-model self-test
(``versions/v_qcmem_hy3_port.md`` v2) has A1/A2 PASS at max|diff|=0.0 including the
MoE routing on the resume half; ``j=0`` READ == full forward.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from ckpt_rotation import (  # noqa: E402
    STEP_DIR_PATTERN,
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

from src.memory.qcmem.qcmem_hy3 import QCMemHy3Model  # noqa: E402

# Reuse the PROVEN PG19 packer + top-k KL from the Qwen distill trainer verbatim
# (same data windows, same loss). Importing the module does not run training.
import train_qcmem_distill as _qwen  # noqa: E402

PG19Packer = _qwen.PG19Packer
distill_logits_kl = _qwen.distill_logits_kl


def main():
    p = argparse.ArgumentParser(
        description="QCMem LoRA self-distillation on Hy3 (device_map sharded, single process)")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--resume_j", type=int, default=32,
                   help="Student resume depth j (Hy3 split-j=32/80). Teacher is always j=0.")
    p.add_argument("--top_prepay_b", type=int, default=0)
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-sep nn.Linear submodule names to LoRA-fy in layers[j:]. "
                        "MoE expert Parameters are NOT modules -> never matched.")
    # data
    p.add_argument("--pg19_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl"))
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_ctx", type=int, default=3,
                   help="Number of context chunks per pack (window = (n_ctx+1)*chunk).")
    p.add_argument("--query_loss_tokens", type=int, default=0,
                   help="If >0, take the loss on only the last N query tokens (0=all).")
    # teacher / loss
    p.add_argument("--teacher_topk", type=int, default=64)
    p.add_argument("--distill_lambda", type=float, default=0.6,
                   help="lam*KL(p||q)+(1-lam)*KL(q||p) on teacher top-k support.")
    p.add_argument("--ce_weight", type=float, default=0.0)
    # optim
    p.add_argument("--total_steps", type=int, default=4000)
    p.add_argument("--lr", type=float, default=8e-5)
    p.add_argument("--warmup_steps", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    # io
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_interval", type=int, default=500)
    # adapter-dir rotation; --keep_last_n 0 restores keep-everything.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=0,
                      default_keep_milestones=0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default="")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]

    print(f"[hy3-distill] model={args.model_path} resume_j={args.resume_j} "
          f"top_prepay_b={args.top_prepay_b} lora_r={args.lora_rank}/a{args.lora_alpha} "
          f"chunk={args.chunk_size} n_ctx={args.n_ctx} dtype={dtype} "
          f"steps={args.total_steps} lr={args.lr}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ------------------------------------------------------------------ #
    # load Hy3 sharded across all visible GPUs (ONE instance, no DDP)
    # ------------------------------------------------------------------ #
    t_load = time.time()
    print(f"[hy3-distill] loading Hy3 device_map=auto (this takes ~2min)...", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, torch_dtype=dtype, device_map="auto",
        attn_implementation=args.attn_impl, low_cpu_mem_usage=True,
        local_files_only=True,
    )
    base.config.use_cache = False
    dm = getattr(base, "hf_device_map", None)
    if dm is not None:
        devs = sorted({str(v) for v in dm.values()})
        print(f"[hy3-distill] loaded in {time.time()-t_load:.0f}s | "
              f"hf_device_map spans {len(devs)} device(s): {devs}", flush=True)
    L = int(base.config.num_hidden_layers)
    if not (0 <= args.resume_j <= L):
        raise SystemExit(f"resume_j must be in [0,{L}]; got {args.resume_j}")
    if not (0 <= args.top_prepay_b <= L - args.resume_j):
        raise SystemExit(f"top_prepay_b must be in [0,{L-args.resume_j}]; got {args.top_prepay_b}")

    # Freeze the whole backbone (incl. the 192-expert MoE Parameters), attach LoRA
    # to layers[resume_j:] ONLY. get_peft_model places each LoRA A/B on the SAME
    # device as its (sharded) base Linear, so device_map + LoRA works out of the box.
    for prm in base.parameters():
        prm.requires_grad = False
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets,
        layers_to_transform=list(range(args.resume_j, L)),
        layers_pattern="layers",
    )
    peft_model = get_peft_model(base, lora_cfg)
    causal_lm = peft_model.base_model.model  # underlying HYV3ForCausalLM
    train_params = [prm for prm in peft_model.parameters() if prm.requires_grad]
    n_train = sum(prm.numel() for prm in train_params)
    print(f"[hy3-distill] LoRA on layers[{args.resume_j}:{L}] targets={targets} "
          f"-> trainable {n_train/1e6:.2f}M params ({len(train_params)} tensors)", flush=True)
    if n_train == 0:
        raise SystemExit("LoRA matched 0 modules — check --lora_targets / layers_pattern")

    peft_model.train()

    # QCMem orchestrators (thin, device-aware, no params of their own) reading the
    # SAME sharded backbone. teacher j=0 (adapters OFF at call), student j=resume_j.
    qc_teacher = QCMemHy3Model(causal_lm, resume_j=0, top_prepay_b=0)
    qc_student = QCMemHy3Model(causal_lm, resume_j=args.resume_j,
                               top_prepay_b=args.top_prepay_b)
    qc_student.grad_checkpoint = bool(args.gradient_checkpointing)
    embed_dev = qc_student.embed_device
    print(f"[hy3-distill] embed device={embed_dev} sharded={qc_student.is_sharded} "
          f"grad_ckpt={qc_student.grad_checkpoint}", flush=True)

    opt = torch.optim.AdamW(train_params, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    # wandb (offline-safe)
    wb = None
    if args.wandb_run_name:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                            config=vars(args))
        except Exception as e:  # pragma: no cover
            print(f"[hy3-distill] wandb init failed ({e}); continuing", flush=True)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    # single process => full stream (rank 0 of world 1)
    packer = PG19Packer(args.pg19_path, tokenizer, args.chunk_size, args.n_ctx,
                        rank=0, world_size=1, seed=args.seed)
    stream = packer.stream()

    os.makedirs(args.output_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.output_dir, "distill_args.json"), "w"),
              indent=2)

    t0 = time.time()
    running = 0.0
    seen = 0
    step = 0
    micro = 0
    opt.zero_grad(set_to_none=True)
    print(f"[hy3-distill] starting training loop ({args.total_steps} steps)", flush=True)

    while step < args.total_steps:
        sample = next(stream)
        ctx_chunks = [c.to(embed_dev) for c in sample["ctx"]]
        query_chunk = sample["query"].to(embed_dev)
        T_q = (int(query_chunk.shape[1]) if query_chunk.dim() == 2
               else int(query_chunk.shape[0]))
        n_loss = T_q if args.query_loss_tokens <= 0 else min(args.query_loss_tokens, T_q)

        for lr_grp in opt.param_groups:
            lr_grp["lr"] = lr_at(step)

        # ======== TEACHER: j=0 read (adapters disabled), no grad ========
        with torch.no_grad():
            with peft_model.disable_adapter():
                t_sink = qc_teacher.write_chunk([bos_id])
                t_ctx = [qc_teacher.write_chunk(c) for c in ctx_chunks]
                t_q = qc_teacher.write_chunk(query_chunk)
                t_logits = qc_teacher.read_core(t_sink, t_ctx, t_q, logits_tail=T_q)
                t_logits = t_logits[0].float()          # [T_q, V] on lm_head device
                t_loss_logits = t_logits[-n_loss:]      # [A, V]
                tk = torch.topk(
                    t_loss_logits,
                    k=min(args.teacher_topk, t_loss_logits.shape[-1]), dim=-1)
                teacher_idx = tk.indices                # [A, k]
                teacher_val = tk.values                 # [A, k]
                teacher_argmax = teacher_idx[:, 0]      # [A]

        # ======== STUDENT: j=resume_j read (adapters ON), grad on layers[j:] ========
        # writes use layers[0:j] (NO LoRA there) -> frozen bottom cache, no grad.
        with torch.no_grad():
            s_sink = qc_student.write_chunk([bos_id])
            s_ctx = [qc_student.write_chunk(c) for c in ctx_chunks]
            s_q = qc_student.write_chunk(query_chunk)
        # grad-bearing resume over layers[j:] (LoRA lives here)
        s_logits = qc_student.read_core(s_sink, s_ctx, s_q, logits_tail=T_q)
        s_loss_logits = s_logits[0][-n_loss:].float()   # [A, V] on lm_head device

        kl = distill_logits_kl(s_loss_logits, teacher_idx, teacher_val,
                               lam=args.distill_lambda)
        loss = kl
        if args.ce_weight > 0.0:
            ce = F.cross_entropy(s_loss_logits, teacher_argmax)
            loss = loss + args.ce_weight * ce

        (loss / args.grad_accum).backward()
        running += float(loss.detach().item())
        seen += 1
        micro += 1

        if micro % args.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if step % args.log_interval == 0:
                avg = running / max(1, seen)
                dt = time.time() - t0
                print(f"[hy3-distill] step {step}/{args.total_steps} "
                      f"loss {avg:.4f} lr {lr_at(step):.2e} "
                      f"{seen/dt:.2f} samp/s", flush=True)
                if wb is not None:
                    wb.log({"loss": avg, "lr": lr_at(step), "step": step})
                running = 0.0
                seen = 0
                t0 = time.time()

            if step % args.save_interval == 0 or step == args.total_steps:
                save_dir = os.path.join(args.output_dir, f"step{step}")
                os.makedirs(save_dir, exist_ok=True)
                peft_model.save_pretrained(save_dir)
                print(f"[hy3-distill] saved LoRA adapter -> {save_dir}", flush=True)
                # rotation (shared policy, scripts/ckpt_rotation.py). Adapters are
                # DIRECTORIES (step{N}/) -> rmtree. final/ is never touched; the
                # just-written dir is never removed; an empty save rotates nothing;
                # --keep_last_n 0 disables rotation. This trainer is single-process
                # (world_size=1), so there is no DDP race.
                rotate_checkpoints(
                    args.output_dir,
                    just_written=save_dir,
                    pattern=STEP_DIR_PATTERN,
                    is_dir=True,
                    log=lambda m: print(f"[hy3-distill] {m}", flush=True),
                    **rotation_kwargs_from_args(args),
                )

    final_dir = os.path.join(args.output_dir, "final")
    os.makedirs(final_dir, exist_ok=True)
    peft_model.save_pretrained(final_dir)
    print(f"[hy3-distill] DONE. final adapter -> {final_dir}", flush=True)
    if wb is not None:
        wb.finish()


if __name__ == "__main__":
    main()
