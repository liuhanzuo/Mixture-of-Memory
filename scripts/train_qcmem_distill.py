#!/usr/bin/env python
"""QCMem self-distillation — recover the mid-depth-resume depth cliff with LoRA.

Direction A (2026-07-05). The zero-training QCMem j-sweep shows the mid-depth
resume mechanism (``src/memory/qcmem/qcmem_model.py``) already breaks the
read-out gap on Qwen3-8B for qa5 (j12=51 vs j0=57) but COLLAPSES on precise-
localisation tasks (qa1 j12=11 vs j0=81). The QCMem paper reports a LoRA
*self-distillation* that pushes the depth cliff back (their Qwen qa1 recovers
.14 -> .67 after distilling j12 against the j0 teacher). This script implements
that self-distillation on our Qwen3-8B backbone:

  * TEACHER  = the QCMem read at ``j = 0`` (RAG upper bound: the selected chunks
    are re-forwarded through the WHOLE model with the query present). Run with
    the LoRA adapters DISABLED (``peft.disable_adapter()``) under ``no_grad`` so
    it is exactly the frozen base model on the packed sequence.
  * STUDENT  = the QCMem read at ``j = --resume_j`` (default 12), i.e. the memory
    chunks are cached at depth ``j`` (query-blind, from the FROZEN bottom
    ``layers[0:j]``) and only ``layers[j:]`` are re-run at read. LoRA (r16,
    all-linear) is attached to ``layers[j:]`` ONLY, so the student learns to
    reconstruct the teacher's behaviour from the shallow cache.
  * LOSS = bidirectional top-k KL (teacher top-k support, reused from
    ``train_mem_space_dolmino_cpt.distill_logits_kl``) on the QUERY-segment
    tokens, optionally + a small CE-to-teacher-argmax term.
  * DATA = PG19 natural text (``data/pg19_train.jsonl``), streamed and tokenised
    on the fly with the Qwen tokenizer, packed into
    ``[sink ; ctx chunks ; query chunk]``. PURE self-supervision — NO BABILong,
    NO needles, NO eval data (red line). Whether pure-PG19 KL suffices for Qwen
    (the paper needed a needle-mix on Llama, §4.6/§4.9) is exactly the question.

The teacher and student share ONE model instance (adapters on/off), so there is
no second copy in memory. Only the ~29M LoRA params (+ the AdamW state) are
trainable; the 8B backbone is frozen. Single- or multi-GPU DDP. For 30B/32B
backbones that OOM under DDP (one full replica per GPU), pass ``--use_fsdp`` to
FULL_SHARD the frozen backbone across ranks while keeping the (small) LoRA a
full replica (grads synced by the same manual all-reduce as the DDP path).

Correctness: at ``--resume_j 0`` teacher==student by construction (both are the
full forward, adapters are zero-init so make no difference at step 0), and the
``--self_test`` gate re-checks the read/write packing == full forward exactly.
"""
from __future__ import annotations

import argparse
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

from ckpt_rotation import (  # noqa: E402
    STEP_DIR_PATTERN,
    add_rotation_args,
    rotate_checkpoints,
    rotation_kwargs_from_args,
)


from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402
from peft import LoraConfig, get_peft_model, TaskType  # noqa: E402

from src.memory.qcmem import QCMemModel  # noqa: E402

# Reuse the proven top-k KL from the mem_space distill trainer (import by file
# path so a stale root-level shadow can't hijack it).
import importlib.util as _ilu  # noqa: E402

_cpt_path = os.path.join(PROJECT_ROOT, "scripts", "train_mem_space_dolmino_cpt.py")
_spec = _ilu.spec_from_file_location("_qcmem_cpt", _cpt_path)
_cpt = _ilu.module_from_spec(_spec)
# The cpt module imports mem_space etc.; that is side-effect free (no patching).
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
# FSDP: shard the FROZEN backbone, keep LoRA replicated (30B/32B distill)
# --------------------------------------------------------------------------- #
# QCMem drives the forward by calling ``causal_lm.model.layers[i](...)`` DIRECTLY
# (never ``causal_lm.forward``), and also calls ``embed_tokens / norm / lm_head``
# directly. A ROOT FSDP wrap would leave those residual modules SHARDED when
# called outside ``root.forward`` (garbage output), so instead we wrap EACH
# decoder layer as its own FSDP unit (FULL_SHARD): calling ``layer(...)`` then
# transparently all-gathers that layer's shard, runs, and reshards. embed/norm/
# lm_head are frozen and left as full replicas (a few GB — acceptable; the 30B
# bulk lives in the decoder layers, which ARE sharded).
#
# The backbone is FROZEN (no grads); only LoRA trains. We keep LoRA OUT of the
# FSDP flat params via ``ignored_modules`` so LoRA stays a FULL replica on every
# rank — its grads are then synced by the SAME manual all-reduce the DDP path
# already uses, its init by the SAME rank-0 broadcast, and it saves with clean
# names (only the FSDP module-wrapper prefix needs stripping). ``use_orig_params
# =True`` lets a single FSDP unit hold mixed frozen(sharded)+ignored(LoRA) params.
def _fsdp_wrap_backbone(causal_lm, dtype, local_rank, rank):
    from torch.distributed.fsdp import (  # noqa: E402
        FullyShardedDataParallel as FSDP,
        ShardingStrategy,
        MixedPrecision,
        BackwardPrefetch,
    )

    if not dist.is_initialized():
        raise SystemExit(
            "--use_fsdp requires a distributed process group; launch with torchrun "
            "(e.g. torchrun --nproc_per_node 8 scripts/train_qcmem_distill.py ...)."
        )

    inner = getattr(causal_lm, "model", causal_lm)
    layers = inner.layers

    mp = None
    if dtype in (torch.bfloat16, torch.float16):
        # params/compute in low precision; reduce grads in fp32 for stability.
        mp = MixedPrecision(param_dtype=dtype, reduce_dtype=torch.float32,
                            buffer_dtype=dtype)

    _LORA_LEAVES = ("lora_A", "lora_B", "lora_embedding_A", "lora_embedding_B",
                    "lora_magnitude_vector")

    def _lora_ignored(module):
        ig = []
        for name, sub in module.named_modules():
            if name.split(".")[-1] in _LORA_LEAVES:
                ig.append(sub)
        return ig

    n = 0
    for i in range(len(layers)):
        ignored = _lora_ignored(layers[i])  # empty for layers < resume_j
        layers[i] = FSDP(
            layers[i],
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp,
            use_orig_params=True,
            device_id=torch.device(f"cuda:{local_rank}"),
            sync_module_states=False,  # backbone loaded identically from disk
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            limit_all_gathers=True,
            ignored_modules=(ignored if ignored else None),
        )
        n += 1
    if _is_main(rank):
        print(f"[qcmem-distill] FSDP FULL_SHARD wrapped {n} decoder layers "
              f"(frozen backbone sharded; LoRA ignored -> full replica, grads via "
              f"manual all-reduce)", flush=True)


def _save_adapter(peft_model, save_dir, use_fsdp, rank):
    """Save the LoRA adapter (rank-0 only).

    DDP path: plain ``peft_model.save_pretrained`` (names already clean).

    FSDP path: the LoRA params are FSDP-IGNORED full replicas, so we can read
    them straight off ``named_parameters`` WITHOUT any collective, strip the
    ``_fsdp_wrapped_module.`` module-wrapper prefix the per-layer FSDP inserts,
    then round-trip through ``get_peft_model_state_dict(state_dict=...)`` to get
    the exact peft save-format keys, and write ``adapter_config.json`` +
    ``adapter_model.safetensors`` — byte-compatible with a normal peft adapter.
    """
    if not _is_main(rank):
        return
    os.makedirs(save_dir, exist_ok=True)
    if not use_fsdp:
        peft_model.save_pretrained(save_dir)
        return

    from peft import get_peft_model_state_dict  # noqa: E402
    lora_raw = {
        name.replace("_fsdp_wrapped_module.", ""): prm.detach()
        for name, prm in peft_model.named_parameters()
        if "lora_" in name
    }
    peft_sd = get_peft_model_state_dict(peft_model, state_dict=lora_raw)
    peft_sd = {k: v.detach().cpu().contiguous() for k, v in peft_sd.items()}
    active = peft_model.active_adapter
    if not isinstance(active, str):
        active = "default"
    peft_model.peft_config[active].save_pretrained(save_dir)
    try:
        from safetensors.torch import save_file  # noqa: E402
        save_file(peft_sd, os.path.join(save_dir, "adapter_model.safetensors"),
                  metadata={"format": "pt"})
    except Exception:  # pragma: no cover - safetensors always present here
        torch.save(peft_sd, os.path.join(save_dir, "adapter_model.bin"))


# --------------------------------------------------------------------------- #
# PG19 streaming packer — builds [sink ; ctx chunks ; query chunk] windows
# --------------------------------------------------------------------------- #
class PG19Packer:
    """Stream ``pg19_train.jsonl`` (raw wrapped text), tokenise with the Qwen
    tokenizer on the fly, and yield packed sample dicts.

    Each yielded sample is a window of ``(n_ctx + 1) * chunk_size`` tokens split
    into ``n_ctx`` context chunks + 1 query chunk (chunk_size tokens each). The
    QCMem read packs ``[sink ; ctx_0 ; ... ; ctx_{n_ctx-1} ; query]``; the
    distill loss is taken on the query chunk's tokens.

    Sharded across DDP ranks by ``[rank::world_size]`` over produced windows so
    each rank sees a disjoint stream.
    """

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
            chunks = list(toks.split(self.chunk_size))
            # exactly n_ctx+1 chunks of chunk_size (window_len is a multiple)
            ctx_chunks = chunks[: self.n_ctx]
            query_chunk = chunks[self.n_ctx]
            yield {"ctx": ctx_chunks, "query": query_chunk}


# --------------------------------------------------------------------------- #
# self-test gate (mirrors eval_qcmem_babilong.run_self_test, on the LoRA model)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def run_self_test(peft_model, base_causal_lm, tokenizer, device, resume_j, top_b):
    """At j=0 (adapters OFF), QCMem read/write packing == stock full forward.

    Uses fp32 for the <1e-4 gate. Runs on ``base_causal_lm`` (the underlying
    Qwen3ForCausalLM that QCMemModel reads layers off) with adapters disabled so
    the check is on the frozen backbone.
    """
    print("=" * 72)
    print(f"QCMem distill self-test (resume_j={resume_j}, top_prepay_b={top_b})")
    print("=" * 72)
    V = int(base_causal_lm.config.vocab_size)
    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0

    def rid(n):
        return torch.randint(0, V, (1, n), device=device)

    sink = torch.tensor([[bos_id]], device=device)
    c1, c2, c3 = rid(37), rid(29), rid(41)
    q = rid(23)
    packed = torch.cat([sink, c1, c2, c3, q], dim=1)

    with peft_model.disable_adapter():
        qc0 = QCMemModel(base_causal_lm, resume_j=0, top_prepay_b=0)
        ref = qc0.full_forward_logits(packed)
        sh = qc0.write_chunk(sink)
        ch = [qc0.write_chunk(c) for c in (c1, c2, c3)]
        qh = qc0.write_chunk(q)
        out_pack = qc0.read(sh, ch, qh)
        diff_pack = (out_pack.float() - ref.float()).abs().max().item()

        # single-seq resume at the training (resume_j, top_b) must ALSO be exact
        # (one contiguous chunk => top-prepay reduces to connective resume).
        qcj = QCMemModel(base_causal_lm, resume_j=resume_j, top_prepay_b=top_b)
        out_j = qcj.resume_forward_ids(packed)
        diff_j = (out_j.float() - ref.float()).abs().max().item()

    tol = 1e-4
    print(f"  (A) read/write packing (j=0) max|diff| = {diff_pack:.3e}  "
          f"{'PASS' if diff_pack < tol else 'FAIL'}")
    print(f"  (B) resume_forward_ids (j={resume_j},b={top_b}) single-seq max|diff| "
          f"= {diff_j:.3e}  {'PASS' if diff_j < tol else 'FAIL'}")
    ok = diff_pack < tol and diff_j < tol
    print("-" * 72)
    print(f"SELF-TEST: {'ALL PASS' if ok else 'FAILURE — DO NOT TRAIN'}")
    print("=" * 72)
    return ok


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser(description="QCMem LoRA self-distillation on PG19")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--resume_j", type=int, default=12,
                   help="Student resume depth j. Teacher is always j=0.")
    p.add_argument("--top_prepay_b", type=int, default=0,
                   help="Student top-prepay b (Direction B, 0=exact connective).")
    p.add_argument("--lora_rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.0)
    p.add_argument("--lora_targets", type=str,
                   default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
                   help="Comma-sep linear submodule names to LoRA-fy in layers[j:].")
    # data
    p.add_argument("--pg19_path", type=str,
                   default=os.path.join(PROJECT_ROOT, "data", "pg19_train.jsonl"))
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--n_ctx", type=int, default=3,
                   help="Number of context chunks per pack (window = (n_ctx+1)*chunk).")
    p.add_argument("--query_loss_tokens", type=int, default=0,
                   help="If >0, take the distill loss on only the last N query "
                        "tokens (0 = all query-chunk tokens).")
    # teacher
    p.add_argument("--teacher_topk", type=int, default=64,
                   help="Top-k teacher-logit support for the KL.")
    p.add_argument("--distill_lambda", type=float, default=0.6,
                   help="lam*KL(p||q)+(1-lam)*KL(q||p) on the teacher top-k support.")
    p.add_argument("--ce_weight", type=float, default=0.0,
                   help="Optional CE-to-teacher-argmax weight added to the KL.")
    # optim
    p.add_argument("--total_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--gradient_checkpointing", action="store_true", default=False)
    p.add_argument("--use_fsdp", action="store_true", default=False,
                   help="Shard the FROZEN backbone decoder layers with FSDP "
                        "(FULL_SHARD) so 30B/32B models fit (each rank holds 1/N "
                        "of the backbone). LoRA params are FSDP-IGNORED (kept as "
                        "full replicas, grads synced by the same manual all-reduce "
                        "as the DDP path). Default False -> unchanged DDP path for "
                        "small models. Requires launching under torchrun.")
    # io
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--save_interval", type=int, default=250)
    # adapter-dir rotation. Adapters are tiny (~120 MB) so the default keeps 3;
    # --keep_last_n 0 restores the old keep-every-step behaviour.
    add_rotation_args(p, default_keep_last_n=3, default_milestone_every=0,
                      default_keep_milestones=0)
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--dtype", type=str, default="bfloat16",
                   choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn_impl", type=str, default="sdpa")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--self_test", action="store_true", default=False)
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
        dtype = torch.float32  # tight tolerance

    if _is_main(rank):
        print(f"[qcmem-distill] model={args.model_path} resume_j={args.resume_j} "
              f"top_prepay_b={args.top_prepay_b} lora_r={args.lora_rank} "
              f"chunk={args.chunk_size} n_ctx={args.n_ctx} dtype={dtype} "
              f"world_size={world_size}", flush=True)

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
    if not (0 <= args.resume_j <= L):
        raise SystemExit(f"resume_j must be in [0,{L}]; got {args.resume_j}")
    if not (0 <= args.top_prepay_b <= L - args.resume_j):
        raise SystemExit(f"top_prepay_b must be in [0,{L-args.resume_j}]; got {args.top_prepay_b}")

    # Freeze all backbone params, then attach LoRA to layers[resume_j:] ONLY.
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
    # underlying Qwen3ForCausalLM that QCMemModel reads layers off
    causal_lm = peft_model.base_model.model
    n_train = sum(prm.numel() for prm in peft_model.parameters() if prm.requires_grad)
    if _is_main(rank):
        print(f"[qcmem-distill] LoRA on layers[{args.resume_j}:{L}] targets={targets} "
              f"-> trainable {n_train/1e6:.2f}M params", flush=True)

    if args.self_test:
        if _is_main(rank):
            ok = run_self_test(peft_model, causal_lm, tokenizer, device,
                               args.resume_j, args.top_prepay_b)
        else:
            ok = True
        if world_size > 1:
            dist.barrier()
            dist.destroy_process_group()
        sys.exit(0 if ok else 1)

    peft_model.train()
    if args.gradient_checkpointing:
        base.gradient_checkpointing_enable()

    # Shard the frozen backbone across ranks (30B/32B). Must happen BEFORE the
    # QCMemModel orchestrator captures ``inner.layers`` (it mutates the ModuleList
    # entries in place to FSDP units) and BEFORE collecting the trainable params.
    if args.use_fsdp:
        _fsdp_wrap_backbone(causal_lm, dtype, local_rank, rank)

    # QCMem orchestrator (thin, no params) reading layers off the peft-wrapped LM.
    qc = QCMemModel(causal_lm, resume_j=args.resume_j, top_prepay_b=args.top_prepay_b)
    qc.grad_checkpoint = bool(args.gradient_checkpointing)

    # NOTE on data-parallelism: the student forward runs through
    # ``qc.read_core`` which calls ``causal_lm.model.layers[...]`` DIRECTLY, not
    # through a ``DistributedDataParallel.forward()``. DDP only installs its
    # gradient-reduction hooks during its own ``forward``, so wrapping the model
    # in DDP here would NOT synchronise gradients (each rank would train its own
    # adapter). Instead we do EXPLICIT gradient all-reduce (mean over ranks)
    # after ``backward`` — the correct pattern for a custom non-``forward`` graph.
    # Each rank streams a disjoint PG19 shard ([rank::world_size]).
    #
    # Under ``--use_fsdp`` the FROZEN backbone is FULL_SHARD-ed per decoder layer
    # (memory), but LoRA is FSDP-ignored (a full replica on every rank), so the
    # trainable-param handling below — rank-0 broadcast at init + manual grad
    # all-reduce + standard clip — is IDENTICAL for both paths (FSDP touches only
    # the frozen params, which carry no grad).
    train_params = [prm for prm in peft_model.parameters() if prm.requires_grad]

    def _allreduce_grads_mean():
        if world_size <= 1:
            return
        for prm in train_params:
            if prm.grad is None:
                prm.grad = torch.zeros_like(prm)  # keep ranks in lock-step
            dist.all_reduce(prm.grad, op=dist.ReduceOp.SUM)
            prm.grad /= world_size

    opt = torch.optim.AdamW(
        train_params,
        lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.95),
    )

    # Broadcast the (randomly-initialised LoRA-A) params from rank 0 so every
    # rank starts identical; combined with grad all-reduce this keeps the single
    # replicated adapter in sync across ranks.
    if world_size > 1:
        for prm in train_params:
            dist.broadcast(prm.data, src=0)

    def lr_at(step):
        if step < args.warmup_steps:
            return args.lr * (step + 1) / max(1, args.warmup_steps)
        prog = (step - args.warmup_steps) / max(1, args.total_steps - args.warmup_steps)
        return 0.5 * args.lr * (1.0 + math.cos(math.pi * min(1.0, prog)))

    # wandb (main only, offline-safe)
    wb = None
    if _is_main(rank) and args.wandb_run_name:
        try:
            import wandb
            wb = wandb.init(project=args.wandb_project, name=args.wandb_run_name,
                            config=vars(args))
        except Exception as e:  # pragma: no cover
            print(f"[qcmem-distill] wandb init failed ({e}); continuing", flush=True)

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

    t0 = time.time()
    running = 0.0
    seen = 0
    step = 0
    opt.zero_grad(set_to_none=True)
    micro = 0

    while step < args.total_steps:
        sample = next(stream)
        ctx_chunks = [c.to(device) for c in sample["ctx"]]
        query_chunk = sample["query"].to(device)
        T_q = int(query_chunk.shape[1]) if query_chunk.dim() == 2 else int(query_chunk.shape[0])

        # ---- sink hidden at depth j (BOS), adapters OFF (frozen bottom, no grad) ----
        for lr_grp in opt.param_groups:
            lr_grp["lr"] = lr_at(step)

        # ======== TEACHER: j=0 read (adapters disabled), no grad ========
        with torch.no_grad():
            with peft_model.disable_adapter():
                qc_teacher = QCMemModel(causal_lm, resume_j=0, top_prepay_b=0)
                t_sink = qc_teacher.write_chunk([bos_id])
                t_ctx = [qc_teacher.write_chunk(c) for c in ctx_chunks]
                t_q = qc_teacher.write_chunk(query_chunk)
                # only need query-tail logits
                t_logits = qc_teacher.read_core(
                    t_sink, t_ctx, t_q, logits_tail=T_q,
                )  # [1, T_q, V]
                t_logits = t_logits[0].float()  # [T_q, V]
                # take loss on last query_loss_tokens (or all)
                if args.query_loss_tokens > 0:
                    n_loss = min(args.query_loss_tokens, T_q)
                else:
                    n_loss = T_q
                t_loss_logits = t_logits[-n_loss:]  # [A, V]
                tk = torch.topk(t_loss_logits, k=min(args.teacher_topk, t_loss_logits.shape[-1]),
                                dim=-1)
                teacher_idx = tk.indices          # [A, k]
                teacher_val = tk.values           # [A, k]
                teacher_argmax = teacher_idx[:, 0]  # [A]

        # ======== STUDENT: j=resume_j read (adapters ON), with grad ========
        # sink + context caches come from the FROZEN bottom layers[0:j] (no grad,
        # no adapters below j) -> compute under no_grad; only the resume path
        # layers[j:] (with LoRA) is grad-bearing.
        with torch.no_grad():
            s_sink = qc.write_chunk([bos_id])          # [1,1,d] at depth j
            s_ctx = [qc.write_chunk(c) for c in ctx_chunks]
            s_q = qc.write_chunk(query_chunk)          # [1,T_q,d] at depth j
        # grad-bearing resume over layers[j:] (LoRA lives here)
        s_logits = qc.read_core(s_sink, s_ctx, s_q, logits_tail=T_q)  # [1,T_q,V]
        s_loss_logits = s_logits[0][-n_loss:].float()  # [A, V]

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
            params = [prm for prm in peft_model.parameters() if prm.requires_grad]
            _allreduce_grads_mean()
            torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            if _is_main(rank) and step % args.log_interval == 0:
                avg = running / max(1, seen)
                dt = time.time() - t0
                msg = (f"[qcmem-distill] step {step}/{args.total_steps} "
                       f"loss {avg:.4f} lr {lr_at(step):.2e} "
                       f"{seen*world_size/dt:.1f} samp/s")
                print(msg, flush=True)
                if wb is not None:
                    wb.log({"loss": avg, "lr": lr_at(step), "step": step})
                running = 0.0
                seen = 0
                t0 = time.time()

            if _is_main(rank) and (step % args.save_interval == 0 or step == args.total_steps):
                save_dir = os.path.join(args.output_dir, f"step{step}")
                _save_adapter(peft_model, save_dir, args.use_fsdp, rank)
                print(f"[qcmem-distill] saved LoRA adapter -> {save_dir}", flush=True)
                # rotation (shared policy, scripts/ckpt_rotation.py). Adapters are
                # DIRECTORIES (step{N}/), so this rmtree's whole dirs. final/ is
                # never touched; the just-written dir is never removed; an empty
                # save rotates nothing; --keep_last_n 0 disables rotation.
                # rank-0 only (we are inside the _is_main guard).
                rotate_checkpoints(
                    args.output_dir,
                    just_written=save_dir,
                    pattern=STEP_DIR_PATTERN,
                    is_dir=True,
                    log=lambda m: print(f"[qcmem-distill] {m}", flush=True),
                    **rotation_kwargs_from_args(args),
                )

    if _is_main(rank):
        final_dir = os.path.join(args.output_dir, "final")
        _save_adapter(peft_model, final_dir, args.use_fsdp, rank)
        print(f"[qcmem-distill] DONE. final adapter -> {final_dir}", flush=True)
        if wb is not None:
            wb.finish()

    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
