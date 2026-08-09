#!/usr/bin/env python3
"""Which parallelism wrappers preserve CAST's weight<->mask element alignment?

This is the empirical evidence behind the parallelism choice in
``cast/train_cast_llama.py``, and the regression guard for the bug that broke the
previous reproduction.  Run on >= 2 GPUs (8 preferred):

    torchrun --nproc_per_node 8 tools/fsdp_misalignment_demo.py

Claim originally under test (CAST_REPRODUCTION_AUDIT.md section 4.1): under FSDP
FULL_SHARD, a rank's slice of `weight` and its slice of the same layer's `mask`
are taken at different FlatParameter offsets, so they are not element-aligned and
their numel can even differ.  Any optimizer that pairs them element-wise is then
either wrong or -- as in the old code -- silently falls back to plain Adam.

The refined claim this tool now tests: the hazard is **parameter flattening /
re-viewing**, not sharding per se.  A wrapper is safe iff, at ``opt.step()``
time, the object in ``optimizer.param_groups`` is still the original
``nn.Parameter`` carrying the original 2-D shape and the original ``cast_mask``
attribute.  Sharding *optimizer state* over whole tensors (ZeRO) does that;
sharding *the tensors themselves* does not.

Each candidate is scored on four things:
  A. does every in-scope Parameter still carry ``cast_in_scope``/``cast_mask``?
     (silent attribute loss is the most dangerous outcome: the old AdamS then
     saw no CAST params at all and every check vacuously "passed")
  B. does ``weight.shape == mask.shape`` hold?
  C. does a real ``AdamS.step()`` succeed, and what coverage does it report?
  D. how many optimizer-state elements does this rank actually allocate?
     (the whole point of the exercise -- (C) passing with zero saving is useless)
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cast as cast_pkg  # noqa: E402
from cast import CastSparseLinear, refresh_all_masks  # noqa: E402
from cast.adams import AdamS, MaskCoverageError, build_param_groups  # noqa: E402

DIM = 512
N_BLOCKS = 16  # 32 in-scope tensors -> 4 per rank at world=8, so no rank is empty


class Block(nn.Module):
    def __init__(self, device):
        super().__init__()
        self.q_proj = CastSparseLinear(DIM, DIM, bias=False, scale_groups=2, device=device)
        self.down_proj = CastSparseLinear(DIM, DIM, bias=False, scale_groups=2, device=device)

    def forward(self, x):
        return self.down_proj(self.q_proj(x))


def build_model(device) -> nn.Module:
    m = nn.Sequential(*[Block(device) for _ in range(N_BLOCKS)])
    refresh_all_masks(m)
    return m


# ---------------------------------------------------------------------------
def inspect_params(named_params) -> dict:
    """(A) attribute survival + (B) shape alignment, over the optimizer's view."""
    n_tagged = 0
    n_mask_attr = 0
    n_aligned = 0
    examples = []
    for name, p in named_params:
        if not getattr(p, "cast_in_scope", False):
            continue
        n_tagged += 1
        msk = getattr(p, "cast_mask", None)
        if msk is None:
            examples.append((name, tuple(p.shape), None))
            continue
        n_mask_attr += 1
        if tuple(msk.shape) == tuple(p.shape) and msk.numel() == p.numel():
            n_aligned += 1
        else:
            examples.append((name, tuple(p.shape), tuple(msk.shape)))
    return {
        "tagged": n_tagged,
        "with_mask_attr": n_mask_attr,
        "aligned": n_aligned,
        "bad_examples": examples[:3],
    }


def try_adams_step(opt, model_for_backward, device, expect_tensors=None) -> dict:
    """(C) run one real AdamS step and report what the optimizer saw."""
    x = torch.randn(2, DIM, device=device)
    out = model_for_backward(x)
    out.float().pow(2).mean().backward()
    try:
        opt.step()
    except MaskCoverageError as e:
        return {"ok": False, "error": f"MaskCoverageError: {str(e).splitlines()[0][:160]}"}
    except Exception as e:  # noqa: BLE001 - any failure is a disqualification
        return {"ok": False, "error": f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"}
    inner = getattr(opt, "optim", opt)
    s = getattr(inner, "last_stats", {}) or {}
    res = {
        "ok": True,
        "cast_tensors": s.get("cast_tensors"),
        "cast_tensors_aligned": s.get("cast_tensors_aligned"),
        "cast_elements": s.get("cast_elements"),
        "decayed_elements": s.get("decayed_elements"),
    }
    if expect_tensors is not None:
        res["expected_tensors"] = expect_tensors
    return res


def optimizer_state_elements(opt) -> int:
    """(D) exp_avg + exp_avg_sq elements this rank actually allocated."""
    inner = getattr(opt, "optim", opt)
    tot = 0
    for st in inner.state.values():
        for k in ("exp_avg", "exp_avg_sq"):
            t = st.get(k)
            if torch.is_tensor(t):
                tot += t.numel()
    return tot


# ---------------------------------------------------------------------------
def cand_ddp(device, local_rank, log) -> dict:
    from torch.nn.parallel import DistributedDataParallel as DDP

    m = build_model(device)
    wrapped = DDP(m, device_ids=[local_rank], broadcast_buffers=False)
    info = inspect_params(wrapped.named_parameters())
    opt = AdamS(build_param_groups(m, lr=1e-4), lr=1e-4, total_steps=100, require_fp32=True)
    step = try_adams_step(opt, wrapped, device, expect_tensors=2 * N_BLOCKS)
    return {"params": info, "step": step, "opt_state_elements": optimizer_state_elements(opt)}


def cand_fsdp1(device, local_rank, log, strategy_name: str) -> dict:
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import ShardingStrategy

    m = build_model(device)
    wrapped = FSDP(
        m,
        sharding_strategy=getattr(ShardingStrategy, strategy_name),
        use_orig_params=True,
        device_id=local_rank,
    )
    info = inspect_params(wrapped.named_parameters())
    opt = AdamS(
        [
            {"params": [p for p in wrapped.parameters() if getattr(p, "cast_in_scope", False)],
             "cast_scope": True, "lr": 1e-4},
            {"params": [p for p in wrapped.parameters() if not getattr(p, "cast_in_scope", False)],
             "cast_scope": False, "lr": 1e-4},
        ],
        lr=1e-4,
        total_steps=100,
        require_fp32=True,
    )
    step = try_adams_step(opt, wrapped, device, expect_tensors=2 * N_BLOCKS)
    return {"params": info, "step": step, "opt_state_elements": optimizer_state_elements(opt)}


def cand_fsdp2(device, local_rank, log) -> dict:
    from torch.distributed.fsdp import fully_shard

    m = build_model(device)
    for blk in m:
        fully_shard(blk, reshard_after_forward=False)
    wrapped = fully_shard(m, reshard_after_forward=False)
    info = inspect_params(wrapped.named_parameters())
    # Extra FSDP2-specific probe: DTensor keeps the GLOBAL shape, so the shape
    # check can PASS while the local shard is a different set of elements from
    # the (unsharded) mask buffer.  Record that explicitly -- it is a *new*
    # silent-failure candidate, worse than FULL_SHARD's loud numel mismatch.
    dt = []
    for name, p in wrapped.named_parameters():
        if not getattr(p, "cast_in_scope", False):
            continue
        loc = p.to_local() if hasattr(p, "to_local") else None
        msk = getattr(p, "cast_mask", None)
        dt.append(
            {
                "name": name,
                "is_dtensor": hasattr(p, "to_local"),
                "global_shape": tuple(p.shape),
                "local_shape": tuple(loc.shape) if loc is not None else None,
                "mask_shape": tuple(msk.shape) if msk is not None else None,
            }
        )
    info["dtensor_probe"] = dt[:2]
    opt = AdamS(
        [
            {"params": [p for p in wrapped.parameters() if getattr(p, "cast_in_scope", False)],
             "cast_scope": True, "lr": 1e-4},
            {"params": [p for p in wrapped.parameters() if not getattr(p, "cast_in_scope", False)],
             "cast_scope": False, "lr": 1e-4},
        ],
        lr=1e-4,
        total_steps=100,
        require_fp32=True,
    )
    step = try_adams_step(opt, wrapped, device, expect_tensors=2 * N_BLOCKS)
    return {"params": info, "step": step, "opt_state_elements": optimizer_state_elements(opt)}


def cand_zero(device, local_rank, log) -> dict:
    """DDP + ZeroRedundancyOptimizer(AdamS): shards optimizer STATE over WHOLE tensors."""
    from torch.distributed.optim import ZeroRedundancyOptimizer
    from torch.nn.parallel import DistributedDataParallel as DDP

    m = build_model(device)
    wrapped = DDP(m, device_ids=[local_rank], broadcast_buffers=False)
    info = inspect_params(wrapped.named_parameters())
    opt = ZeroRedundancyOptimizer(
        build_param_groups(m, lr=1e-4),
        optimizer_class=AdamS,
        lr=1e-4,
        total_steps=100,
        require_fp32=True,
    )
    step = try_adams_step(opt, wrapped, device)
    # Whole-tensor partition => this rank owns a strict SUBSET of the 32 tensors,
    # and the union over ranks must be exactly 32.  Verify globally.
    owned = torch.tensor(
        [float(step.get("cast_tensors") or 0), float(step.get("cast_elements") or 0)],
        dtype=torch.float64,
        device=device,
    )
    dist.all_reduce(owned, op=dist.ReduceOp.SUM)
    step["global_cast_tensors"] = int(owned[0].item())
    step["global_cast_elements"] = int(owned[1].item())
    step["expected_global_tensors"] = 2 * N_BLOCKS
    return {"params": info, "step": step, "opt_state_elements": optimizer_state_elements(opt)}


# ---------------------------------------------------------------------------
def main() -> int:
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world < 2:
        print("needs >= 2 ranks: torchrun --nproc_per_node 8 tools/fsdp_misalignment_demo.py")
        return 2
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    torch.manual_seed(0)

    def log(m=""):
        if rank == 0:
            print(f"[demo] {m}", flush=True)

    n_scope = 2 * N_BLOCKS
    baseline_state = None
    log(f"world={world}  toy model: {N_BLOCKS} blocks x 2 CastSparseLinear({DIM}x{DIM}) "
        f"= {n_scope} in-scope tensors, {n_scope * DIM * DIM:,} in-scope elements")
    log()

    candidates = [
        ("DDP (current, no sharding)", cand_ddp, {}),
        ("FSDP1 FULL_SHARD  use_orig_params=True", cand_fsdp1, {"strategy_name": "FULL_SHARD"}),
        ("FSDP1 SHARD_GRAD_OP use_orig_params=True", cand_fsdp1, {"strategy_name": "SHARD_GRAD_OP"}),
        ("FSDP2 fully_shard reshard_after_forward=False", cand_fsdp2, {}),
        ("DDP + ZeroRedundancyOptimizer(AdamS)", cand_zero, {}),
    ]

    verdicts = []
    for name, fn, kw in candidates:
        try:
            res = fn(device, local_rank, log, **kw)
        except Exception as e:  # noqa: BLE001
            res = {"fatal": f"{type(e).__name__}: {e}"}
            if rank == 0:
                traceback.print_exc()
        torch.cuda.empty_cache()
        dist.barrier()

        log("=" * 78)
        log(name)
        if "fatal" in res:
            log(f"  FATAL during wrapping: {res['fatal']}")
            verdicts.append((name, "UNUSABLE (wrapping failed)", None))
            log()
            continue

        p = res["params"]
        log(f"  (A) in-scope Parameters visible to optimizer : {p['tagged']}/{n_scope} tagged "
            f"cast_in_scope, {p['with_mask_attr']} carry .cast_mask")
        log(f"  (B) weight.shape == mask.shape               : {p['aligned']}/{p['tagged'] or 0}")
        for nm, ws, ms in p["bad_examples"]:
            log(f"        MISALIGNED {nm}: param {ws} vs mask {ms}")
        if "dtensor_probe" in p:
            for d in p["dtensor_probe"]:
                log(f"        DTensor {d['name']}: is_dtensor={d['is_dtensor']} "
                    f"global={d['global_shape']} local={d['local_shape']} mask={d['mask_shape']}")
        s = res["step"]
        if s.get("ok"):
            log(f"  (C) AdamS.step()                             : OK  aligned="
                f"{s['cast_tensors_aligned']}/{s['cast_tensors']} "
                f"elements={s['cast_elements']:,} decayed={s['decayed_elements']:,}")
            if "global_cast_tensors" in s:
                log(f"        global over {world} ranks: tensors={s['global_cast_tensors']} "
                    f"(expected {s['expected_global_tensors']}) "
                    f"elements={s['global_cast_elements']:,}")
        else:
            log(f"  (C) AdamS.step()                             : REJECTED -> {s['error']}")
        st = res["opt_state_elements"]
        if baseline_state is None:
            baseline_state = st or 1
        log(f"  (D) optimizer state elements on this rank    : {st:,} "
            f"({st / baseline_state:.3f}x of DDP baseline)")

        # ---- verdict ----
        full_local_cov = (
            s.get("ok")
            and s.get("cast_tensors")
            and s["cast_tensors"] == s["cast_tensors_aligned"]
        )
        attrs_ok = p["tagged"] == n_scope or (
            "global_cast_tensors" in s and s["global_cast_tensors"] == s["expected_global_tensors"]
        )
        saves = st < 0.9 * baseline_state
        if not s.get("ok"):
            v = "UNSAFE (AdamS refuses)"
        elif p["tagged"] == 0:
            v = "UNSAFE (attributes lost -> checks vacuous, silent vanilla Adam)"
        elif p["aligned"] != p["tagged"]:
            v = "UNSAFE (mask misaligned)"
        elif not full_local_cov or not attrs_ok:
            v = "UNSAFE (incomplete coverage)"
        elif saves:
            v = "SAFE + SAVES MEMORY"
        else:
            v = "SAFE but no memory saving"
        log(f"  ==> {v}")
        log()
        verdicts.append((name, v, st))

    log("=" * 78)
    log("SUMMARY")
    for name, v, st in verdicts:
        log(f"  {v:<58} {name}")
    log()
    log("Interpretation: the hazard is FLATTENING/RE-VIEWING the parameter, not sharding.")
    log("ZeRO shards optimizer state over WHOLE tensors, so every rank's weight is still")
    log("the original full 2-D Parameter with its original mask -> alignment by construction,")
    log("exactly as under DDP, while exp_avg/exp_avg_sq shrink by ~1/world_size.")

    dist.barrier()
    dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
