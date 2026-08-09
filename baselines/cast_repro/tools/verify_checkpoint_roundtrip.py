#!/usr/bin/env python3
"""Bit-exact verification that a CAST checkpoint round-trips its FULL state.

WHY THIS EXISTS AND NOT JUST A LOSS-TRACE COMPARISON
----------------------------------------------------
The obvious faithfulness test -- run N+M steps straight through, separately run N
-> save -> resume -> M, and diff the loss curves -- was tried first and is
**inconclusive on this hardware**.  Measured on 8x L20A (torch 2.13.0, bf16
autocast + gradient checkpointing + SDPA, 32-micro-batch accumulation), two runs
with *identical config, identical seed, and no resume at all* already diverge:

    step 0   A=1.107989544049  B=1.107989544049   d=0
    step 1   A=6.252430841327  B=6.252966612577   d=5.4e-04
    step 2   A=3.457070469856  B=3.451371617615   d=5.7e-03   <-- before any ckpt

Step 0 is bit-identical (same weights, same batch), then the trajectories separate.
The cause is non-deterministic reduction order in the backward kernels (atomics in
SDPA/gradient-checkpoint recompute), which bf16 accumulation amplifies. So the
run-to-run noise floor is ~5.7e-3 in the loss, and a resumed arm differed from the
control by 3.6e-3 -- i.e. *below* the noise floor. That comparison therefore cannot
distinguish a perfect resume from a slightly wrong one, and quoting it as proof
would be exactly the kind of unfalsifiable "looks fine" claim this project has
already been burned by.

WHAT THIS SCRIPT DOES INSTEAD
-----------------------------
It tests the thing the checkpoint code is actually responsible for: that the state
written to disk is the state that comes back, **bit for bit**. GPU kernel
non-determinism cannot contaminate this, because no training step is involved --
only serialise, deserialise, compare.

Under ``--parallel zero2`` on 8 ranks it:

  1. builds the model + ZeRO(AdamS) exactly as the trainer does;
  2. runs a few real training steps so the Adam moments and masks are non-trivial;
  3. saves a checkpoint;
  4. captures a full fingerprint of live state (every weight, every mask, every
     Adam moment, per-parameter step, RNG, next data indices);
  5. *perturbs* everything in memory -- weights, masks, moments, RNG -- so a load
     that silently does nothing cannot pass;
  6. loads the checkpoint back;
  7. compares the fingerprint element-wise and reports max |delta| per category.

Step 5 is the part that makes this a real test rather than a tautology. Without it,
a ``load_state_dict`` that quietly skipped every tensor would still "pass".

Run:
    torchrun --nproc_per_node 8 tools/verify_checkpoint_roundtrip.py \
        --parallel zero2 --steps 3
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

HERE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(HERE))

from cast import (  # noqa: E402
    LLAMA2_7B_CAST_TENSORS,
    AdamS,
    build_param_groups,
    cast_loss,
    cast_modules,
    convert_llama_to_cast,
    refresh_all_masks,
)
from cast.checkpoint import (  # noqa: E402
    assert_optimizer_state_restored,
    load_training_state,
    save_training_state,
)


def log(msg: str) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[verify] {msg}", flush=True)


def main() -> int:  # noqa: C901
    ap = argparse.ArgumentParser()
    ap.add_argument("--parallel", default="zero2", choices=["ddp", "zero2"])
    ap.add_argument("--steps", type=int, default=3, help="real steps before saving")
    ap.add_argument("--model", default="models/Llama--Llama2-7b")
    ap.add_argument("--data", default="Mixture-of-Memory/data/dolmino-mix-1124-llama2")
    ap.add_argument("--project-root", default="/apdcephfs_wzc1/share_304376610/pighzliu_code")
    ap.add_argument("--out", default="outputs/cast_ckpt_roundtrip")
    ap.add_argument("--seq-len", type=int, default=4096)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--accum", type=int, default=2, help="kept small: state fidelity, not throughput")
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--l1-decay", type=float, default=4e-7)
    ap.add_argument("--max-steps", type=int, default=7500, help="declared horizon (feeds alpha_t)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--keep", action="store_true")
    args = ap.parse_args()

    # The resume guard checks every key in RESUME_CRITICAL_ARGS and refuses if the
    # checkpoint cannot vouch for one -- which is the correct behaviour and it fired
    # on the first version of this tool. This tool has its own (smaller) arg surface,
    # so fill in the trainer's remaining recipe keys explicitly. They go into the
    # saved args and are then compared against themselves, which is exactly what we
    # want here: this tool tests state fidelity, not launch-flag validation (that is
    # covered by tests/test_cast.py::test_resume_* on the pure-python path).
    saved_args = dict(vars(args))
    saved_args.update(
        lr_schedule="constant", min_lr=2e-6, warmup=375, global_batch=256,
        mask_period=10, scale_groups=2, eta=1.0 / 3.0, kl_temperature=1.0,
        betas=(0.9, 0.999), eps=1e-8, grad_clip=1.0, no_teacher=False,
        data_dtype="auto",
    )

    root = Path(args.project_root)
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world > 1:
        import datetime as dt

        dist.init_process_group("nccl", timeout=dt.timedelta(seconds=3600))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(args.seed + rank)
    is_zero = args.parallel == "zero2"

    outdir = root / args.out
    if rank == 0:
        if outdir.exists():
            shutil.rmtree(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()

    # ---- model, exactly as the trainer builds it ----
    from transformers import LlamaForCausalLM

    mp = root / args.model
    log(f"loading {mp}")
    model = LlamaForCausalLM.from_pretrained(str(mp), torch_dtype=torch.float32,
                                             attn_implementation="sdpa")
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    convert_llama_to_cast(model, n=2, m=4, scale_groups=2)
    model.to(device)
    refresh_all_masks(model)

    teacher = LlamaForCausalLM.from_pretrained(str(mp), torch_dtype=torch.bfloat16,
                                               attn_implementation="sdpa")
    teacher.config.use_cache = False
    teacher.eval().to(device)
    for p in teacher.parameters():
        p.requires_grad_(False)

    if world > 1:
        kw = dict(device_ids=[local_rank], gradient_as_bucket_view=True)
        import inspect as _i

        if "forward_sync_buffers" in _i.signature(DDP.__init__).parameters:
            kw["forward_sync_buffers"] = False
        else:
            kw["broadcast_buffers"] = False
        student = DDP(model, **kw)
    else:
        student = model
    inner = model

    adams_kw = dict(lr=args.lr, betas=(0.9, 0.999), eps=1e-8, total_steps=args.max_steps,
                    l1_decay=args.l1_decay, require_fp32=True)
    if is_zero:
        from torch.distributed.optim import ZeroRedundancyOptimizer

        opt = ZeroRedundancyOptimizer(
            build_param_groups(inner, lr=args.lr), optimizer_class=AdamS,
            expected_scope_elements=None, expected_scope_tensors=None, **adams_kw,
        )
    else:
        opt = AdamS(build_param_groups(inner, lr=args.lr), **adams_kw)

    # ---- data ----
    ddir = root / args.data
    dtype = json.loads((ddir / "metadata.json").read_text())["dtype"]
    data = np.memmap(ddir / "train.bin", dtype=np.dtype(dtype), mode="r")
    gen = np.random.default_rng(args.seed + rank)
    n_tok = len(data)

    def batch(bs: int):
        idx = gen.integers(0, n_tok - args.seq_len - 1, size=bs)
        x = np.stack([data[i:i + args.seq_len].astype(np.int64) for i in idx])
        y = np.stack([data[i + 1:i + 1 + args.seq_len].astype(np.int64) for i in idx])
        return (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device))

    # ---- a few real steps so the state is non-trivial ----
    for step in range(args.steps):
        if step % 10 == 0:
            refresh_all_masks(inner)
        opt.zero_grad(set_to_none=True)
        for micro in range(args.accum):
            x, y = batch(args.micro_batch)
            ctx = student.no_sync() if (world > 1 and micro < args.accum - 1) else torch.enable_grad()
            with ctx:
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = student(input_ids=x).logits
                    with torch.no_grad():
                        tl = teacher(input_ids=x).logits
                    loss, _ = cast_loss(out, tl, y, eta=1.0 / 3.0, temperature=1.0)
                (loss / args.accum).backward()
        torch.nn.utils.clip_grad_norm_(inner.parameters(), 1.0)
        opt.step()
        log(f"step {step} done (loss {float(loss):.4f})")
    last = args.steps - 1

    # ---- save ----
    ck = outdir / f"ckpt_step{last}"
    save_training_state(ck, step=last, model=inner, opt=opt, args=saved_args,
                        np_generator=gen, is_zero=is_zero, rank=rank, world=world)
    log(f"saved {ck}")

    # ---- fingerprint the LIVE state ----
    live_model = {k: v.detach().to("cpu", copy=True) for k, v in inner.state_dict().items()}
    o_inner = getattr(opt, "optim", opt) if is_zero else opt
    live_opt = {}
    for gi, group in enumerate(o_inner.param_groups):
        for pi, p in enumerate(group["params"]):
            st = o_inner.state.get(p)
            if st:
                live_opt[(gi, pi)] = {
                    "exp_avg": st["exp_avg"].detach().to("cpu", copy=True),
                    "exp_avg_sq": st["exp_avg_sq"].detach().to("cpu", copy=True),
                    "step": int(st["step"].item()) if torch.is_tensor(st["step"])
                            else int(st["step"]),
                }
    live_rng = gen.bit_generator.state
    # what the NEXT batch would read, i.e. the data-order continuation point
    probe = np.random.default_rng()
    probe.bit_generator.state = live_rng
    live_next_idx = probe.integers(0, n_tok - args.seq_len - 1, size=4).tolist()

    # ---- PERTURB everything, so a no-op load cannot pass ----
    with torch.no_grad():
        for _, mod in cast_modules(inner):
            mod.weight.add_(1.0)          # weights
            mod.mask.copy_(~mod.mask)     # masks (flip every bit)
            mod.cast_scale.add_(0.5)
        for st in o_inner.state.values():
            st["exp_avg"].add_(7.0)       # moments
            st["exp_avg_sq"].add_(7.0)
            if torch.is_tensor(st["step"]):
                st["step"].add_(999)
            else:
                st["step"] = st["step"] + 999
    gen.bit_generator.state = np.random.default_rng(999).bit_generator.state  # RNG
    log("perturbed live state (weights +1, masks inverted, moments +7, step +999, RNG reseeded)")

    # ---- load back ----
    meta = load_training_state(ck, model=inner, opt=opt, cur_args=saved_args,
                               np_generator=gen, is_zero=is_zero, rank=rank, world=world,
                               device=device, expected_mask_buffers=LLAMA2_7B_CAST_TENSORS)
    counts = assert_optimizer_state_restored(opt, expected_step=meta["step"] + 1,
                                             is_zero=is_zero, rank=rank, world=world,
                                             device=device)

    # ---- compare bit-exactly ----
    now_model = inner.state_dict()
    worst_w, worst_w_name, n_w = 0.0, "", 0
    mask_mismatch, n_mask = 0, 0
    for k, ref in live_model.items():
        cur = now_model[k].detach().cpu()
        if ref.dtype == torch.bool:
            n_mask += 1
            mask_mismatch += int((cur != ref).sum())
        else:
            n_w += 1
            d = float((cur.float() - ref.float()).abs().max())
            if d > worst_w:
                worst_w, worst_w_name = d, k

    worst_m, worst_m_name, n_opt = 0.0, "", 0
    step_mismatch = 0
    for (gi, pi), ref in live_opt.items():
        p = o_inner.param_groups[gi]["params"][pi]
        st = o_inner.state[p]
        n_opt += 1
        for key in ("exp_avg", "exp_avg_sq"):
            d = float((st[key].detach().cpu().float() - ref[key].float()).abs().max())
            if d > worst_m:
                worst_m, worst_m_name = d, f"{key}[g{gi}p{pi}] shape={tuple(p.shape)}"
        cur_step = int(st["step"].item()) if torch.is_tensor(st["step"]) else int(st["step"])
        if cur_step != ref["step"]:
            step_mismatch += 1

    probe2 = np.random.default_rng()
    probe2.bit_generator.state = gen.bit_generator.state
    now_next_idx = probe2.integers(0, n_tok - args.seq_len - 1, size=4).tolist()

    stats = torch.tensor(
        [worst_w, worst_m, float(mask_mismatch), float(step_mismatch),
         float(now_next_idx != live_next_idx), float(n_w), float(n_mask), float(n_opt)],
        dtype=torch.float64, device=device,
    )
    if world > 1:
        dist.all_reduce(stats, op=dist.ReduceOp.MAX)
    gw, gm, gmask, gstep, gidx, tw, tmask, topt = stats.tolist()

    log("")
    log("=== CHECKPOINT ROUND-TRIP (bit-exactness of the SAVED vs RESTORED state) ===")
    log(f"parallel={args.parallel} world={world} steps_before_save={args.steps} ckpt={ck.name}")
    log(f"float model tensors per rank : {int(tw)}   max |delta| = {gw:.3e}  (worst: {worst_w_name})")
    log(f"bool mask buffers per rank   : {int(tmask)}   mismatching elements = {int(gmask)}")
    log(f"optimizer state tensors      : {int(topt)} params/rank, max |delta| = {gm:.3e}"
        f"  (worst: {worst_m_name})")
    log(f"per-parameter step counters   : {int(gstep)} mismatches")
    log(f"next-batch data indices       : {'IDENTICAL' if gidx == 0 else 'DIFFERENT'}"
        f"  (rank0 {live_next_idx} -> {now_next_idx})")
    log(f"optimizer coverage            : {counts.get('global_with_state', counts['with_state'])}"
        f"/{counts.get('global_params', counts['params'])} owned params carry moments")
    ok = (gw == 0.0 and gm == 0.0 and gmask == 0 and gstep == 0 and gidx == 0)
    log("")
    log(f"VERDICT: {'PASS - state round-trips BIT-EXACTLY' if ok else 'FAIL - state changed'}")
    log("(All values were perturbed in memory before the load -- weights +1, masks")
    log(" inverted, moments +7, step +999, RNG reseeded -- so a no-op load would fail.)")

    if rank == 0 and not args.keep:
        shutil.rmtree(outdir, ignore_errors=True)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
