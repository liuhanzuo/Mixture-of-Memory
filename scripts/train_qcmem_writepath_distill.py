#!/usr/bin/env python
"""QCMem WRITE-PATH self-distillation — train the lower-12 Write to be
document-contextual (Paper A P1.10, 2026-08-03).

Motivation
----------
CoMem's deployable configuration caches, per 512-token chunk, the depth-``j``
hidden ``h_j`` produced by the FROZEN bottom ``layers[0:j]`` run CHUNK-LOCALLY
(each chunk encoded in isolation, RoPE 0:T). P0.16/P0.18 showed that this
chunk-local Write is the dominant source of the deployable quality gap: replacing
it with a DOCUMENT-CONTEXTUAL Write (lower-12 run continuously / full-causal over
the whole document so each token's ``h_j`` sees the preceding context) closes the
gap to ~100 WITH the flagship READ LoRA present (P0.18 E0, "closes-to-100"), and
the zero-training overlap-Write (P0.17) already recovers 80-87% of it.

This script trains the *trained upper bound* of that finding: a LoRA on the LOWER
``j`` layers (indices ``0..j-1``) so the CHEAP chunk-local Write learns to emit an
``h_j`` that behaves like the (expensive, non-deployable) document-contextual
Write — distilled against it token-for-token.

  * TEACHER = DOCUMENT-CONTEXTUAL Write (the P0.18 E0 "closes-to-100" construction).
    ``layers[0:j]`` are run ONCE, continuously and full-causally, over the whole
    packed window ``[sink ; ctx_0 ; … ; ctx_{n-1} ; query]`` in its natural order
    (contiguous positions 0:N), producing a per-token ``h_j`` that sees the full
    preceding context. The WRITE LoRA is DISABLED here (``peft.disable_adapter``)
    so the bottom band is the FROZEN base — exactly E0's stock lower-12 (the flagship
    READ LoRA lives on ``layers[j:]``, so the bottom band is adapter-independent).
    Then ``layers[j:]`` (with the merged flagship READ LoRA) resume over the pack →
    query-tail logits. NO grad.
    (In this PG19 training regime EVERY chunk is used, in order, so the document
    positions equal the pack positions — the store→read repositioning gap that E0
    isolates against Arm B is ZERO, and the teacher is precisely "document-contextual
    Write + trained Read". The student's ONLY deficiency vs. this teacher is the
    chunk-local ISOLATION of its Write — which is exactly what the WRITE LoRA learns
    to overcome.)

  * STUDENT = DEPLOYABLE chunk-local Write + SAME (merged, frozen) Read. Each of
    sink / ctx_k / query is encoded to depth ``j`` CHUNK-LOCALLY (isolated, RoPE
    0:T) with the WRITE LoRA ON and GRAD-BEARING; the per-chunk ``h_j`` are packed
    into fresh contiguous pack positions 0:H and ``layers[j:]`` resume → query-tail
    logits. Gradient flows query-tail-logits → (frozen) upper band → packed ``h_j``
    → WRITE LoRA on ``layers[0:j]``.

  * LOSS = bidirectional top-k KL (teacher top-k support; reused verbatim from
    ``train_mem_space_dolmino_cpt.distill_logits_kl``) on the QUERY-chunk tokens,
    ``lam`` default 0.6, optional CE-to-teacher-argmax. This mirrors the flagship
    READ-path distillation objective family (``train_qcmem_distill.py``); we keep
    top-k logit KD (the mechanism is "make the two Reads agree", which is a logit-
    level statement) rather than an ``h_j`` MSE, so the objective is directly
    comparable to the flagship row and is robust to the frozen Read re-weighting the
    contribution of individual ``h_j`` dimensions.

  * DATA = PG19 natural text (``data/pg19_train.jsonl``), streamed + tokenised on the
    fly, packed into ``[sink ; ctx chunks ; query]`` windows. PURE self-supervision
    — NO BABILong / NO needles / NO eval data (red line).

Isolation
---------
This is a NEW file (P1.10). It does NOT edit ``train_qcmem_distill.py`` (the
flagship READ-path trainer) nor ``qcmem_model.py``. It reuses ``QCMemModel``'s
public low-level accessors (``embed_tokens``, ``rotary_emb``, ``_run_layers``,
``norm``, ``lm_head``) to build BATCHED grad-bearing Write/Read forwards (the stock
``read_core`` is batch-1; batching along the batch axis is how we fill the H20), and
imports ``distill_logits_kl`` by file path from ``train_mem_space_dolmino_cpt.py``.

Correctness gate (``--self_test``): with the WRITE LoRA disabled, the custom
batched contiguous Write(0:j)+Read(j:L) over a single sequence reproduces the
merged model's full forward to fp tolerance (validates the whole batched pipeline
+ the SDPA implicit-causal read == an explicit causal mask). Zero-init LoRA-B means
the adapter is identity at step 0, so this also fixes the training graph at init.
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import math
import os
import random
import sys
import time
from typing import List, Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, PeftModel, get_peft_model, TaskType  # noqa: E402

from src.memory.qcmem import QCMemModel  # noqa: E402

# Reuse the proven top-k KL from the mem_space distill trainer (import by file
# path so a stale root-level shadow can't hijack it). Identical file on all nodes.
_cpt_path = os.path.join(PROJECT_ROOT, "scripts", "train_mem_space_dolmino_cpt.py")
_spec = _ilu.spec_from_file_location("_qcmem_cpt", _cpt_path)
_cpt = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_cpt)
distill_logits_kl = _cpt.distill_logits_kl


# --------------------------------------------------------------------------- #
# distributed helpers
# --------------------------------------------------------------------------- #
def _dist_setup():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def _is_main(rank):
    return rank == 0


# --------------------------------------------------------------------------- #
# batched grad-bearing Write / Read over QCMemModel's low-level accessors.
#
# The stock ``QCMemModel.read_core`` / ``write_chunk`` are batch-1 (write_chunk is
# also ``@torch.no_grad``). We reproduce the SAME math here batched along the batch
# axis and WITHOUT the no_grad guard, using SDPA implicit-causal masking (pass
# ``attention_mask=None`` with ``q_len>1`` -> the sdpa/flash kernel applies a causal
# mask, bit-identical to an explicit causal mask; validated by ``--self_test``). RoPE
# positions are the contiguous ``0:S`` a stock forward would use; ``rotary_emb`` with
# ``[1,S]`` positions broadcasts over the batch. Requires an sdpa/flash backbone
# (asserted in ``main``); an eager backbone would need an explicit mask.
# --------------------------------------------------------------------------- #
def _rope_pe(qc, ref: torch.Tensor, S: int):
    positions = torch.arange(S, device=qc.device).unsqueeze(0)  # [1, S]
    return positions, qc.rotary_emb(ref, position_ids=positions)


def _lower_write(qc, ids: torch.Tensor) -> torch.Tensor:
    """Chunk-local (or, for the teacher, continuous) lower band ``layers[0:j]``.

    ``ids`` is ``[B, T]``. Returns ``h_j`` ``[B, T, d]``. Runs under the ambient
    autograd context: grad-bearing iff called outside ``no_grad`` (student), so the
    WRITE LoRA on ``layers[0:j]`` accumulates gradient.
    """
    emb = qc.embed_tokens(ids)                       # [B, T, d]
    T = int(ids.shape[1])
    positions, pe = _rope_pe(qc, emb, T)
    return qc._run_layers(emb, slice(0, qc.resume_j), None, positions, pe)


def _upper_read(qc, packed_hj: torch.Tensor, tail: Optional[int] = None) -> torch.Tensor:
    """Resume ``layers[j:L]`` over the packed ``h_j`` with fresh contiguous pack
    positions 0:H, then ``norm + lm_head``. ``packed_hj`` is ``[B, H, d]``. If
    ``tail`` is set, only the last ``tail`` positions' logits are materialised
    (avoids the full ``[B, H, V]`` tensor; the pre-lm_head stack is unchanged so the
    tail is numerically identical to slicing the full output)."""
    H = int(packed_hj.shape[1])
    positions, pe = _rope_pe(qc, packed_hj, H)
    hidden = qc._run_layers(packed_hj, slice(qc.resume_j, qc.num_layers),
                            None, positions, pe)
    if tail is not None and tail > 0:
        hidden = hidden[:, -int(tail):, :]
    hidden = qc.norm(hidden)
    return qc.lm_head(hidden)                         # [B, H or tail, V]


# --------------------------------------------------------------------------- #
# adapter save (rank-0; DDP path — 8B fits a full replica per GPU, no FSDP)
# --------------------------------------------------------------------------- #
def _save_adapter(peft_model, save_dir, rank):
    if not _is_main(rank):
        return
    os.makedirs(save_dir, exist_ok=True)
    peft_model.save_pretrained(save_dir)


# --------------------------------------------------------------------------- #
# PG19 streaming packer — [sink ; ctx chunks ; query] windows (== flagship packer)
# --------------------------------------------------------------------------- #
class PG19Packer:
    def __init__(self, path, tokenizer, chunk_size, n_ctx, rank, world_size, seed):
        self.path = path
        self.tok = tokenizer
        self.chunk_size = int(chunk_size)
        self.n_ctx = int(n_ctx)
        self.window_len = (self.n_ctx + 1) * self.chunk_size
        self.rank = rank
        self.world_size = world_size
        self.seed = seed

    def _windows(self):
        buf: List[int] = []
        wcount = 0
        while True:  # loop the corpus indefinitely (training is step-bounded)
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    buf.extend(self.tok.encode(line, add_special_tokens=False))
                    while len(buf) >= self.window_len:
                        w = buf[: self.window_len]
                        buf = buf[self.window_len:]
                        if wcount % self.world_size == self.rank:
                            yield w
                        wcount += 1

    def stream(self):
        for w in self._windows():
            toks = torch.tensor(w, dtype=torch.long)
            chunks = list(toks.split(self.chunk_size))  # n_ctx+1 chunks of chunk_size
            yield {"ctx": chunks[: self.n_ctx], "query": chunks[self.n_ctx]}


# --------------------------------------------------------------------------- #
# self-test gate: custom batched contiguous Write(0:j)+Read(j:L) == full forward
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_self_test(peft_model, qc, tokenizer, device, resume_j):
    print("=" * 72)
    print(f"QCMem WRITE-PATH distill self-test (resume_j={resume_j})")
    print("=" * 72)
    V = int(qc.config.vocab_size)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    def rid(n):
        return torch.randint(0, V, (1, n), device=device)

    seq = torch.cat([torch.tensor([[bos_id]], device=device),
                     rid(37), rid(29), rid(41), rid(23)], dim=1)  # [1, S]

    with peft_model.disable_adapter():  # WRITE LoRA off -> frozen (merged-READ) base
        ref = peft_model(input_ids=seq, use_cache=False).logits.float()  # [1,S,V]
        h = _lower_write(qc, seq)                       # contiguous lower band
        out = _upper_read(qc, h).float()                # full-position read
    diff = (out - ref).abs().max().item()
    tol = 1e-3  # bf16-loaded weights promoted to fp32 here -> slightly looser than 1e-4
    ok = diff < tol
    print(f"  contiguous Write(0:{resume_j})+Read({resume_j}:L) vs full forward "
          f"max|diff| = {diff:.3e}  {'PASS' if ok else 'FAIL'}  (tol {tol:.0e})")
    print("-" * 72)
    print(f"SELF-TEST: {'PASS' if ok else 'FAILURE — DO NOT TRAIN'}")
    print("=" * 72)
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(
        description="QCMem WRITE-path LoRA self-distillation on PG19 (P1.10)")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--read_lora_path", type=str,
                   default="outputs/qcmem_distill_qwen_j12_r32_4k/final",
                   help="Flagship READ LoRA (layers[j:L]); merged into the base as a "
                        "frozen Read so both teacher & student share the SAME trained "
                        "Read and only the WRITE path differs (P0.18 E0 config).")
    p.add_argument("--resume_j", type=int, default=12,
                   help="Split depth j: Write=layers[0:j] (LoRA here), Read=layers[j:L].")
    p.add_argument("--lora_rank", type=int, default=32)
    p.add_argument("--lora_alpha", type=int, default=64)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-sep linear submodule names to LoRA-fy in layers[0:j].")
    # data
    p.add_argument("--pg19_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl"))
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_ctx", type=int, default=3,
                   help="Context chunks per pack (window = (n_ctx+1)*chunk).")
    p.add_argument("--batch_size", type=int, default=8,
                   help="Windows per optimizer micro-step (batched along batch axis "
                        "to fill the H20; all windows are fixed length so no padding).")
    p.add_argument("--query_loss_tokens", type=int, default=0,
                   help="If >0, distill only the last N query tokens (0 = all).")
    # teacher / loss
    p.add_argument("--teacher_topk", type=int, default=64)
    p.add_argument("--distill_lambda", type=float, default=0.6)
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
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self_test", action="store_true", default=False)
    p.add_argument("--max_steps_smoke", type=int, default=0,
                   help="If >0, stop after this many optimizer steps (multi-GPU sanity).")
    p.add_argument("--wandb_project", type=str, default="mixture-of-memory")
    p.add_argument("--wandb_run_name", type=str, default="")
    args = p.parse_args()

    rank, world_size, local_rank = _dist_setup()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    random.seed(args.seed + rank)

    dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
             "float32": torch.float32}[args.dtype]
    if args.self_test:
        dtype = torch.float32
    if args.attn_impl not in ("sdpa", "flash_attention_2", "flash_attention_3"):
        raise SystemExit(
            f"--attn_impl must be sdpa/flash (batched implicit-causal read relies on "
            f"it); got {args.attn_impl}. An eager backbone needs an explicit mask.")

    if _is_main(rank):
        print(f"[wp-distill] model={args.model_path} read_lora={args.read_lora_path} "
              f"resume_j={args.resume_j} lora_r={args.lora_rank}/a{args.lora_alpha} "
              f"chunk={args.chunk_size} n_ctx={args.n_ctx} batch={args.batch_size} "
              f"dtype={dtype} world_size={world_size}", flush=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, attn_implementation=args.attn_impl,
        trust_remote_code=True, local_files_only=True,
    ).to(device)
    base.config.use_cache = False
    L = int(base.config.num_hidden_layers)
    if not (1 <= args.resume_j <= L):
        raise SystemExit(f"resume_j must be in [1,{L}]; got {args.resume_j}")

    # ---- Load the flagship READ LoRA (layers[j:L]) and MERGE it into the base ----
    # After merge the base carries the trained Read as frozen weights; the bottom band
    # layers[0:j] is untouched (READ LoRA never lived there) -> still the E0 stock
    # lower-12. This makes both teacher & student share the SAME Read; only the WRITE
    # path (which we now attach a fresh LoRA to) differs.
    read_peft = PeftModel.from_pretrained(base, args.read_lora_path, is_trainable=False)
    base = read_peft.merge_and_unload()
    if _is_main(rank):
        print(f"[wp-distill] merged flagship READ LoRA from {args.read_lora_path} "
              f"into the frozen base", flush=True)

    # Freeze everything, then attach a NEW WRITE LoRA on layers[0:j] ONLY.
    for prm in base.parameters():
        prm.requires_grad = False
    targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
    write_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=targets,
        layers_to_transform=list(range(0, args.resume_j)),   # LOWER band (Write)
        layers_pattern="layers",
    )
    peft_model = get_peft_model(base, write_cfg)
    causal_lm = peft_model.base_model.model
    n_train = sum(prm.numel() for prm in peft_model.parameters() if prm.requires_grad)
    if _is_main(rank):
        print(f"[wp-distill] WRITE LoRA on layers[0:{args.resume_j}] targets={targets} "
              f"-> trainable {n_train/1e6:.2f}M params", flush=True)

    qc = QCMemModel(causal_lm, resume_j=args.resume_j, top_prepay_b=0)
    qc.grad_checkpoint = bool(args.gradient_checkpointing)

    if args.self_test:
        ok = run_self_test(peft_model, qc, tokenizer, device, args.resume_j) \
            if _is_main(rank) else True
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    peft_model.train()
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    train_params = [prm for prm in peft_model.parameters() if prm.requires_grad]

    # DDP note (identical to train_qcmem_distill.py): the forward runs through
    # ``causal_lm.layers[...]`` DIRECTLY, not ``DistributedDataParallel.forward``, so
    # DDP's grad hooks would never fire. We do EXPLICIT grad all-reduce (mean over
    # ranks) after backward + a rank-0 broadcast of the init, keeping the single
    # replicated WRITE adapter in sync. Each rank streams a disjoint PG19 shard.
    def _allreduce_grads_mean():
        if world_size <= 1:
            return
        for prm in train_params:
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)
            dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
            prm.grad /= world_size

    opt = torch.optim.AdamW(train_params, lr=args.lr,
                            weight_decay=args.weight_decay, betas=(0.9, 0.95))
    if world_size > 1:
        for prm in train_params:
            dist.broadcast(prm.data, src=0)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    wb = None
    if _is_main(rank) and args.wandb_run_name:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                            config=vars(args))
        except Exception as e:  # pragma: no cover
            print(f"[wp-distill] wandb init failed ({e}); continuing", flush=True)

    bos_id = tokenizer.bos_token_id
    if bos_id is None:
        bos_id = tokenizer.eos_token_id
    packer = PG19Packer(args.pg19_path, tokenizer, args.chunk_size, args.n_ctx,
                        rank, world_size, args.seed)
    stream = packer.stream()

    os.makedirs(args.output_dir, exist_ok=True)
    if _is_main(rank):
        json.dump(vars(args), open(os.path.join(args.output_dir, "distill_args.json"), "w"),
                  indent=2)

    def next_batch(B):
        samples = [next(stream) for _ in range(B)]
        # stack fixed-length chunks along the batch axis (no padding)
        ctx_batch = [torch.stack([s["ctx"][k] for s in samples], dim=0).to(device)
                     for k in range(args.n_ctx)]        # each [B, chunk]
        query_batch = torch.stack([s["query"] for s in samples], dim=0).to(device)  # [B, chunk]
        bos_col = torch.full((B, 1), int(bos_id), dtype=torch.long, device=device)
        return bos_col, ctx_batch, query_batch

    t0 = time.time()
    running = 0.0
    seen = 0
    step = 0
    opt.zero_grad(set_to_none=True)
    micro = 0
    B = args.batch_size

    while step < args.total_steps:
        bos_col, ctx_batch, query_batch = next_batch(B)
        T_q = int(query_batch.shape[1])
        n_loss = min(args.query_loss_tokens, T_q) if args.query_loss_tokens > 0 else T_q

        for g in opt.param_groups:
            g["lr"] = lr_at(step)

        # ======== TEACHER: DOCUMENT-CONTEXTUAL Write (continuous lower band), Read;
        #          WRITE LoRA disabled, no grad. h_j sees the full preceding context.
        with torch.no_grad():
            with peft_model.disable_adapter():
                doc_ids = torch.cat([bos_col] + ctx_batch + [query_batch], dim=1)  # [B,N]
                h_doc = _lower_write(qc, doc_ids)                    # [B, N, d]
                t_logits = _upper_read(qc, h_doc, tail=T_q)          # [B, T_q, V]
                t_ll = t_logits[:, -n_loss:, :].reshape(-1, t_logits.shape[-1]).float()
                tk = torch.topk(t_ll, k=min(args.teacher_topk, t_ll.shape[-1]), dim=-1)
                teacher_idx = tk.indices          # [B*n_loss, k]
                teacher_val = tk.values           # [B*n_loss, k]
                teacher_argmax = teacher_idx[:, 0]

        # ======== STUDENT: DEPLOYABLE chunk-local Write (WRITE LoRA ON, grad) + Read.
        s_sink = _lower_write(qc, bos_col)                           # [B, 1, d]
        s_ctx = [_lower_write(qc, c) for c in ctx_batch]             # each [B, chunk, d]
        s_q = _lower_write(qc, query_batch)                          # [B, chunk, d]
        packed = torch.cat([s_sink] + s_ctx + [s_q], dim=1)          # [B, H, d]
        s_logits = _upper_read(qc, packed, tail=T_q)                 # [B, T_q, V]
        s_ll = s_logits[:, -n_loss:, :].reshape(-1, s_logits.shape[-1]).float()

        kl = distill_logits_kl(s_ll, teacher_idx, teacher_val, lam=args.distill_lambda)
        loss = kl
        if args.ce_weight > 0.0:
            ce = F.cross_entropy(s_ll, teacher_argmax)
            loss = loss + args.ce_weight * ce

        (loss / args.grad_accum).backward()
        running += float(loss.detach().item())
        seen += 1
        micro += 1

        if micro % args.grad_accum == 0:
            _allreduce_grads_mean()
            torch.nn.utils.clip_grad_norm_(train_params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if _is_main(rank) and step % args.log_interval == 0:
                avg = running / max(1, seen)
                dt = time.time() - t0
                print(f"[wp-distill] step {step}/{args.total_steps} loss {avg:.4f} "
                      f"lr {lr_at(step):.2e} "
                      f"{seen*world_size*B/dt:.1f} win/s", flush=True)
                if wb is not None:
                    wb.log({"loss": avg, "lr": lr_at(step), "step": step})
                running = 0.0
                seen = 0
                t0 = time.time()

            if _is_main(rank) and (step % args.save_interval == 0
                                   or step == args.total_steps):
                save_dir = os.path.join(args.output_dir, f"step{step}")
                _save_adapter(peft_model, save_dir, rank)
                print(f"[wp-distill] saved WRITE LoRA adapter -> {save_dir}", flush=True)

            if args.max_steps_smoke > 0 and step >= args.max_steps_smoke:
                if _is_main(rank):
                    print(f"[wp-distill] smoke stop at step {step} "
                          f"(max_steps_smoke={args.max_steps_smoke})", flush=True)
                break

    if _is_main(rank) and args.max_steps_smoke == 0:
        final_dir = os.path.join(args.output_dir, "final")
        _save_adapter(peft_model, final_dir, rank)
        print(f"[wp-distill] DONE. final WRITE adapter -> {final_dir}", flush=True)
    if wb is not None:
        wb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
