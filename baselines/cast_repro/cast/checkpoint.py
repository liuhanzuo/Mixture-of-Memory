"""Faithful save/resume for CAST training, including under ``--parallel zero2``.

WHY THIS FILE EXISTS
--------------------
A sibling project in this repo attempted to resume three 200k-step arms from
checkpoints.  All three silently became WARM RESTARTS: the optimizer's
param-group layout did not match what was saved, torch quietly re-initialised the
Adam moments, and a differential-LR setting never took effect.  Nothing raised.
Weeks of GPU time produced arms that could not be compared to anything.

The lesson taken here: a resume is either *provably* a continuation, or it is a
crash.  There is no third state.  Every guard below turns a would-be silent
degradation into an exception.

WHY NOT ``ZeroRedundancyOptimizer.consolidate_state_dict``
---------------------------------------------------------
The obvious implementation -- gather all shards to rank 0 and write one file -- was
tried first and **hangs at LLaMA2-7B scale**.  Measured on 8x L20A, torch 2.13.0:
``consolidate_state_dict(to=0)`` sat in ``_broadcast_object`` for >10 minutes with
no progress, all 8 ranks pinned at 100% GPU util, and never produced a byte.

py-spy on all 8 ranks located it exactly: rank 3 (the current sender) was parked on
``zero_redundancy_optimizer.py:102``

    data_send_tensor = torch.ByteTensor(data).to(device)

where ``data`` is a ``bytearray`` holding the pickled shard -- ~7 GB here.
``torch.ByteTensor(bytearray)`` builds the tensor **element by element through the
Python C-API**, so it is O(7e9) Python-level operations while holding the GIL.
The other 7 ranks were correctly blocked in the matching receive.  This is a
scalability limit of ZeRO's object-broadcast helper, not a bug in our usage: it is
fine for the small optimizer states it was written for and unusable for a 7B
model's moments.  Do not "fix" this by waiting longer.

WHAT WE DO INSTEAD: PARALLEL PER-RANK SHARD FILES
-------------------------------------------------
A checkpoint is a *directory*:

    ckpt_step1000/
      meta.json          step, args, world_size, parallel, torch version
      model.pt           rank 0: fp32 master weights + cast_scale + 224 masks
      optim_rank<k>.pt   rank k: ITS OWN Adam shard, keyed by GLOBAL param index
      rng_rank<k>.pt     rank k: torch CPU + torch CUDA + numpy Generator state
      DONE               written last, after a barrier

No cross-rank transfer happens at all, so the pathological path is never entered.
Each rank writes ~7 GB concurrently; rank 0 additionally writes the ~34 GB model.
This is strictly *better* than consolidation: state is already where it belongs, so
it cannot be mis-routed.

The ``DONE`` marker replaces an atomic rename: a crash mid-write leaves a
directory without ``DONE``, and :func:`find_latest_checkpoint` ignores those, so a
torn checkpoint can never be mistaken for a loadable one.

Optimizer state is keyed by ZeRO's **global** parameter index (a flat index over
``param_groups``), not by the rank-local index.  That makes the mapping explicit
and checkable: on load, each rank asserts that the set of global indices in its own
file is exactly the set it now owns (:func:`load_training_state`).  ZeRO's
partition is deterministic -- ``_partition_parameters`` sorts by ``numel``
descending and greedily assigns each whole tensor to the least-loaded rank -- so
for a fixed model and world size the partition is reproduced exactly.  If it ever
were not, the assertion fires instead of silently loading another parameter's
moments.

WHAT IS PERSISTED, AND WHY EACH PIECE IS LOAD-BEARING
-----------------------------------------------------
``model``
    fp32 master weights, ``cast_scale``, and the 224 ``mask`` buffers.  ``mask`` is
    a *persistent* buffer on ``CastSparseLinear``, so ``state_dict()`` already
    carries it; the loader asserts the expected count came back rather than
    trusting that.

    The mask is saved rather than recomputed **on purpose**.  It is refreshed only
    every ``T1=10`` steps, so it is not a function of the current weights: the mask
    live at step 503 was computed from the weights as of step 500.  Recomputing it
    at resume time from post-step-503 weights flips the entries sitting near the
    intra-group threshold, changing *which* weights receive the L1 decay.  "Close
    enough" is exactly the unfalsifiable drift this module exists to prevent, so
    the live mask is restored verbatim.

``optim``
    Adam moments (``exp_avg``, ``exp_avg_sq``) and the per-parameter ``step``
    counter.  ``step`` is not bookkeeping -- AdamS reads it for both bias
    correction and ``alpha_t = (step-1)/T``, the decay ramp.  Losing it restarts
    the sparsification schedule from alpha=0.

``rng``
    Per-rank torch CPU state, torch CUDA state, and -- most importantly -- the
    ``numpy`` Generator driving ``BinDataset``.  Data order is the biggest
    determinism lever in this loop: without it a resumed run reads *different
    tokens* from step one and its loss trajectory does not continue the original.

``args``
    Every hyperparameter that shapes the trajectory, re-checked by
    :func:`check_resume_args`.

WHAT IS **NOT** BIT-REPRODUCIBLE, STATED HONESTLY
-------------------------------------------------
Restoring all of the above makes a resumed step *mathematically* identical to the
uninterrupted one.  Whether it is also **bit**-identical is an empirical question
about cuBLAS/cuDNN reduction order and bf16 autocast accumulation over 32
micro-batches, neither of which is pinned here.  ``tools/resume_faithfulness.py``
measures it rather than assuming it, and reports the actual max |delta| against an
uninterrupted control plus a deliberately warm-restarted arm as a positive
control.  See README for the measured numbers.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.distributed as dist

#: Bumped when the on-disk layout changes incompatibly.
#: 1 = legacy single-file weights-only (NOT resumable)
#: 2 = abandoned single-file consolidated design (never shipped; see module docstring)
#: 3 = directory of parallel per-rank shards (current)
CKPT_FORMAT = 3

DONE_MARKER = "DONE"


class ResumeMismatchError(RuntimeError):
    """Raised when a checkpoint cannot be resumed *faithfully*.

    Deliberately a hard error, in the same spirit as
    :class:`cast.adams.MaskCoverageError`: the failure mode being defended against
    (a resume that silently becomes a warm restart, or that silently changes the
    decay schedule) is invisible in the logs, so continuing past it must be
    impossible.
    """


#: Hyperparameters whose value changes the trajectory, so resuming with a
#: different one is not a continuation of the same experiment.
#:
#: ``max_steps`` is here because it is *not* merely a stopping condition: AdamS uses
#: ``alpha_t = t/max_steps`` as the decay ramp (Alg. 1 line 12) and the cosine
#: schedule uses it as the horizon.  Resuming a 7500-step run as a 10000-step run
#: silently rescales the entire sparsification schedule.  Use ``--stop-after`` to
#: stop early without touching the declared horizon.
RESUME_CRITICAL_ARGS = (
    "lr",
    "l1_decay",
    "max_steps",
    "lr_schedule",
    "min_lr",
    "warmup",
    "global_batch",
    "micro_batch",
    "seq_len",
    "mask_period",
    "scale_groups",
    "eta",
    "kl_temperature",
    "betas",
    "eps",
    "grad_clip",
    "seed",
    "no_teacher",
    "data",
    "data_dtype",
    "parallel",  # ddp and zero2 partition the optimizer state differently
)

#: Changing these does not alter the math, so they may differ across a resume.
RESUME_FREE_ARGS = (
    "out",
    "project_root",
    "model",
    "log_every",
    "diag_every",
    "save_every",
    "keep_last",
    "resume",
    "dist_timeout",
    "stop_after",  # where a run was interrupted is not part of the recipe
    "gradient_checkpointing",  # recompute vs store: same math, different memory
    "smoke",
)


def _norm(v: Any) -> Any:
    """Normalise for comparison: argparse yields tuples/lists inconsistently."""
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    return v


def check_resume_args(ckpt_args: Dict[str, Any], cur_args: Dict[str, Any]) -> Dict[str, Any]:
    """Raise unless ``cur_args`` continues the same experiment as ``ckpt_args``.

    Returns a dict of benign differences (for logging).  Raises
    :class:`ResumeMismatchError` listing *every* offending key at once -- reporting
    one per invocation would turn fixing a launch command into a guessing game.
    """
    bad = []
    for k in RESUME_CRITICAL_ARGS:
        if k not in ckpt_args:
            # Checkpoint predates this key: equality cannot be proven, so refuse.
            bad.append(f"{k}: checkpoint has no record of it (cannot verify)")
            continue
        old, new = _norm(ckpt_args[k]), _norm(cur_args.get(k))
        if old != new:
            bad.append(f"{k}: checkpoint={old!r} but this run has {new!r}")
    if bad:
        raise ResumeMismatchError(
            "refusing to resume: the checkpoint was produced by a DIFFERENT "
            "configuration, so continuing would silently splice two experiments "
            "together.\n  " + "\n  ".join(bad) + "\n"
            "Either fix the launch flags to match, or start a fresh run in a new "
            "--out directory. To stop a run early WITHOUT changing its declared "
            "horizon, use --stop-after (lowering --max-steps rescales alpha_t = t/T). "
            "(This guard exists because a sibling project lost three 200k-step arms "
            "to a resume that silently became a warm restart.)"
        )
    return {
        k: (ckpt_args.get(k), cur_args.get(k))
        for k in RESUME_FREE_ARGS
        if k in ckpt_args and _norm(ckpt_args[k]) != _norm(cur_args.get(k))
    }


# ---------------------------------------------------------------------------
# RNG
# ---------------------------------------------------------------------------
def rng_state(np_generator: Optional[np.random.Generator]) -> Dict[str, Any]:
    """Capture this rank's RNG state.

    The numpy Generator picks training batch offsets, so it is the piece that
    actually controls reproducibility of the data order.
    """
    st: Dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "numpy": np_generator.bit_generator.state if np_generator is not None else None,
    }
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def load_rng_state(st: Dict[str, Any], np_generator: Optional[np.random.Generator]) -> None:
    torch.set_rng_state(st["torch_cpu"].cpu().to(torch.uint8))
    if st.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["torch_cuda"].cpu().to(torch.uint8))
    if st.get("numpy") is not None:
        if np_generator is None:
            raise ResumeMismatchError(
                "checkpoint carries a numpy RNG state but no dataset generator was "
                "passed to restore it into -- the resumed run would read a different "
                "token stream than the original."
            )
        np_generator.bit_generator.state = st["numpy"]


# ---------------------------------------------------------------------------
# ZeRO plumbing
# ---------------------------------------------------------------------------
def _local_optimizer(opt: torch.optim.Optimizer, is_zero: bool) -> torch.optim.Optimizer:
    return getattr(opt, "optim", opt) if is_zero else opt


def _global_index_map(opt: torch.optim.Optimizer, is_zero: bool) -> Dict[int, int]:
    """``id(param) -> global index`` in the world-size-independent ordering.

    For ZeRO this is its own ``_param_to_index`` (a flat enumeration over the
    *unpartitioned* ``param_groups``).  For a plain optimizer the same enumeration
    is what ``state_dict()`` already uses, so the two agree and a ddp checkpoint
    stays interchangeable in structure.
    """
    if is_zero:
        return {id(p): i for p, i in opt._param_to_index.items()}
    idx = 0
    out = {}
    for g in opt.param_groups:
        for p in g["params"]:
            out[id(p)] = idx
            idx += 1
    return out


def _owned_state_by_global_index(
    opt: torch.optim.Optimizer, is_zero: bool
) -> Dict[int, Dict[str, Any]]:
    """This rank's optimizer state, re-keyed from parameter identity to global index."""
    inner = _local_optimizer(opt, is_zero)
    gmap = _global_index_map(opt, is_zero)
    out: Dict[int, Dict[str, Any]] = {}
    for group in inner.param_groups:
        for p in group["params"]:
            st = inner.state.get(p)
            if not st:
                continue
            gi = gmap.get(id(p))
            if gi is None:
                raise ResumeMismatchError(
                    "a parameter held by the local optimizer is absent from the global "
                    "index map; the ZeRO partition and the exposed param_groups disagree."
                )
            out[gi] = {
                k: (v.detach().to("cpu", copy=True) if torch.is_tensor(v) else v)
                for k, v in st.items()
            }
    return out


def _owned_global_indices(opt: torch.optim.Optimizer, is_zero: bool) -> list:
    inner = _local_optimizer(opt, is_zero)
    gmap = _global_index_map(opt, is_zero)
    return sorted(gmap[id(p)] for g in inner.param_groups for p in g["params"])


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------
def save_training_state(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    args: Dict[str, Any],
    np_generator: Optional[np.random.Generator],
    is_zero: bool,
    rank: int,
    world: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[Path]:
    """Write a fully resumable checkpoint DIRECTORY. Collective: call on ALL ranks.

    ``step`` is the index of the step that has just *completed*, so a resume starts
    at ``step + 1``.

    Every rank writes its own optimizer shard and RNG file in parallel; rank 0 adds
    the replicated model and the metadata.  The ``DONE`` marker is written only
    after a barrier, so an interrupted save is never loadable.
    """
    path = Path(path)
    if rank == 0:
        path.mkdir(parents=True, exist_ok=True)
    if world > 1:
        dist.barrier()  # ensure the directory exists before other ranks write into it

    # -- per-rank RNG (tiny) --------------------------------------------------
    torch.save(rng_state(np_generator), path / f"rng_rank{rank}.pt")

    # -- per-rank optimizer shard --------------------------------------------
    # Under ddp every rank holds the identical full state, so writing 8 copies
    # would waste ~430 GB; rank 0's copy is complete and correct.
    if is_zero or rank == 0:
        torch.save(
            {
                "state": _owned_state_by_global_index(opt, is_zero),
                "owned_global_indices": _owned_global_indices(opt, is_zero),
                "rank": rank,
                "world_size": world,
            },
            path / f"optim_rank{rank}.pt",
        )

    # -- replicated model + metadata (rank 0) --------------------------------
    if rank == 0:
        torch.save({"model": model.state_dict()}, path / "model.pt")
        meta = {
            "format": CKPT_FORMAT,
            "step": step,
            "args": args,
            "world_size": world,
            "parallel": "zero2" if is_zero else "ddp",
            "torch_version": torch.__version__,
            "optim_shards": world if is_zero else 1,
        }
        if extra:
            meta.update(extra)
        (path / "meta.json").write_text(json.dumps(meta, indent=2, default=str))

    if world > 1:
        dist.barrier()  # everyone's shard is on disk before the checkpoint is published
    if rank == 0:
        (path / DONE_MARKER).write_text(f"step {step}\n")
    if world > 1:
        dist.barrier()  # nobody proceeds until the marker exists
    return path if rank == 0 else None


def checkpoint_size_bytes(path: Path) -> int:
    return sum(p.stat().st_size for p in Path(path).glob("*") if p.is_file())


def prune_old_checkpoints(outdir: Path, keep_last: int, prefix: str = "ckpt_step") -> list:
    """Delete all but the ``keep_last`` newest complete checkpoints.

    These are ~87 GB each for LLaMA2-7B under this recipe (fp32 model ~27 GB +
    masks ~6.5 GB + Adam moments ~54 GB), so retaining all 30 of a 7500-step run at
    --save-every 250 would need ~2.6 TB.  Sorted by the embedded step number, not by
    mtime: mtime ordering is unreliable on a shared filesystem with clock skew.
    """
    import shutil

    if keep_last <= 0:
        return []
    found = []
    for p in Path(outdir).glob(prefix + "*"):
        if not p.is_dir():
            continue
        digits = p.name[len(prefix):]
        if digits.isdigit():
            found.append((int(digits), p))
    found.sort()
    removed = []
    for _, p in found[:-keep_last]:
        try:
            shutil.rmtree(p)
            removed.append(p)
        except OSError:
            pass
    return removed


def find_latest_checkpoint(outdir: Path, prefix: str = "ckpt_step") -> Optional[Path]:
    """Newest COMPLETE checkpoint, i.e. one carrying the ``DONE`` marker.

    Incomplete directories are skipped rather than reported, so a crash during a
    save degrades to "resume from the previous one" instead of loading a torn
    checkpoint.
    """
    best, best_step = None, -1
    for p in Path(outdir).glob(prefix + "*"):
        digits = p.name[len(prefix):]
        if not (p.is_dir() and digits.isdigit()):
            continue
        if not (p / DONE_MARKER).exists():
            continue
        if int(digits) > best_step:
            best, best_step = p, int(digits)
    return best


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------
def load_training_state(
    path: Path,
    *,
    model: torch.nn.Module,
    opt: torch.optim.Optimizer,
    cur_args: Dict[str, Any],
    np_generator: Optional[np.random.Generator],
    is_zero: bool,
    rank: int,
    world: int,
    device: Optional[torch.device] = None,
    expected_mask_buffers: Optional[int] = None,
) -> Dict[str, Any]:
    """Restore model + optimizer + RNG from a checkpoint directory.

    Each rank reads only ``optim_rank<rank>.pt`` and ``rng_rank<rank>.pt``; the
    replicated ``model.pt`` is read by everyone (mmap'd, so the 8 processes share
    one page cache copy instead of pulling ~34 GB each through the FS).
    """
    path = Path(path)
    if path.is_file():
        raise ResumeMismatchError(
            f"{path} is a FILE. Checkpoints written before the shard-file redesign "
            "(and the plain weights-only step*_prefinal.pt) have no resumable optimizer "
            "state: resuming from one would re-initialise the Adam moments and the "
            "alpha_t decay ramp, i.e. a WARM RESTART masquerading as a continuation. "
            "Point --resume at a ckpt_step<N>/ directory."
        )
    if not (path / DONE_MARKER).exists():
        raise ResumeMismatchError(
            f"{path} has no {DONE_MARKER} marker: the save was interrupted, so the shards "
            "may be torn or missing. Resume from the previous checkpoint instead."
        )

    meta = json.loads((path / "meta.json").read_text())
    if meta.get("format") != CKPT_FORMAT:
        raise ResumeMismatchError(
            f"{path} is format {meta.get('format')}, this code writes/reads {CKPT_FORMAT}"
        )
    if int(meta["world_size"]) != world:
        raise ResumeMismatchError(
            f"checkpoint was written with world_size={meta['world_size']} but this run has "
            f"{world}. Both the optimizer partition AND the per-rank data sharding "
            "(seed+rank) depend on world size, so a different one reads a different token "
            "stream with a different state layout -- that is a new experiment, not a "
            "continuation. Relaunch with --nproc_per_node "
            f"{meta['world_size']}."
        )
    benign = check_resume_args(meta["args"], cur_args)

    # -- model ----------------------------------------------------------------
    try:
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=False, mmap=True)
    except (RuntimeError, TypeError):
        blob = torch.load(path / "model.pt", map_location="cpu", weights_only=False)
    if expected_mask_buffers is not None:
        n_masks = sum(1 for k in blob["model"] if k.endswith(".mask"))
        if n_masks != expected_mask_buffers:
            raise ResumeMismatchError(
                f"checkpoint carries {n_masks} mask buffers, expected {expected_mask_buffers}. "
                "The mask IS training state (refreshed only every T1 steps); a missing mask "
                "means the resumed run would decay a different set of weights than the run "
                "being continued."
            )
    model.load_state_dict(blob["model"], strict=True)
    del blob

    # -- optimizer ------------------------------------------------------------
    shard_file = path / f"optim_rank{rank if is_zero else 0}.pt"
    if not shard_file.exists():
        raise ResumeMismatchError(f"missing optimizer shard {shard_file}")
    shard = torch.load(shard_file, map_location="cpu", weights_only=False)

    inner = _local_optimizer(opt, is_zero)
    gmap = _global_index_map(opt, is_zero)
    want = set(_owned_global_indices(opt, is_zero))
    have = set(int(i) for i in shard["owned_global_indices"])
    if want != have:
        raise ResumeMismatchError(
            f"rank{rank}: this rank now owns {len(want)} parameters but its checkpoint shard "
            f"was written for {len(have)} ({len(want - have)} missing, {len(have - want)} "
            "extra). The ZeRO partition differs from the one at save time, so loading would "
            "install one parameter's Adam moments onto another. Refusing."
        )

    # Rebuild the LOCAL state dict: local index -> state, translated from the
    # global indices stored on disk. Reusing the current param_groups (rather than
    # the saved ones) is deliberate: it preserves `cast_scope`, without which AdamS
    # would silently run vanilla Adam on everything.
    local_sd = inner.state_dict()
    saved = {int(k): v for k, v in shard["state"].items()}
    new_state: Dict[int, Any] = {}
    local_index = 0
    for group in inner.param_groups:
        for p in group["params"]:
            gi = gmap[id(p)]
            if gi in saved:
                new_state[local_index] = saved[gi]
            local_index += 1
    local_sd["state"] = new_state
    inner.load_state_dict(local_sd)
    if is_zero:
        # Keep ZeRO's exposed param_groups in step with the local optimizer's.
        opt._sync_param_groups(inner.param_groups, opt.param_groups)
    del shard, saved, new_state

    # -- RNG ------------------------------------------------------------------
    rng_file = path / f"rng_rank{rank}.pt"
    if not rng_file.exists():
        raise ResumeMismatchError(
            f"missing {rng_file}: without the per-rank RNG state the resumed run reads a "
            "different token stream and its loss trajectory does not continue the original."
        )
    load_rng_state(torch.load(rng_file, map_location="cpu", weights_only=False), np_generator)

    return {
        "step": int(meta["step"]),
        "benign_arg_diffs": benign,
        "saved_world_size": int(meta["world_size"]),
        "saved_parallel": meta.get("parallel"),
        "saved_torch": meta.get("torch_version"),
    }


def assert_optimizer_state_restored(
    opt: torch.optim.Optimizer,
    *,
    expected_step: int,
    is_zero: bool,
    rank: int,
    world: int,
    device: Optional[torch.device] = None,
) -> Dict[str, int]:
    """Prove the Adam moments actually came back. THE anti-warm-restart guard.

    A warm restart is invisible: training continues, the loss looks plausible, and
    only a careful trajectory comparison much later reveals the moments were zeros.
    So this checks, for every parameter this rank owns:

      * ``exp_avg`` and ``exp_avg_sq`` exist, match the parameter's shape, and sit
        on the parameter's device;
      * ``exp_avg_sq`` is not identically zero -- a freshly-initialised moment is
        exactly zero, whereas a trained one cannot be, since AdamS feeds it
        ``mu~^2`` which includes the ``alpha*lambda*sign(theta)`` decay term even
        where the gradient vanishes;
      * the per-parameter ``step`` counter equals ``expected_step`` everywhere;
        AdamS derives bias correction AND ``alpha_t = (step-1)/T`` from it, so a
        reset silently rewinds the sparsification ramp to its start.

    Counts are all-reduced, so a *single* rank returning empty is fatal for the
    whole job rather than quietly training one eighth of the model wrong.
    """
    inner = _local_optimizer(opt, is_zero)
    n_params = n_with_state = n_zero_v = 0
    steps_seen = set()
    for group in inner.param_groups:
        for p in group["params"]:
            n_params += 1
            st = inner.state.get(p)
            if not st:
                continue
            n_with_state += 1
            for key in ("exp_avg", "exp_avg_sq"):
                t = st.get(key)
                if not torch.is_tensor(t):
                    raise ResumeMismatchError(
                        f"rank{rank}: restored optimizer state is missing '{key}' for a "
                        f"parameter of shape {tuple(p.shape)} -- the moments were NOT "
                        "restored (this is the silent warm-restart failure)."
                    )
                if tuple(t.shape) != tuple(p.shape):
                    raise ResumeMismatchError(
                        f"rank{rank}: '{key}' has shape {tuple(t.shape)} but its parameter is "
                        f"{tuple(p.shape)} -- state was routed to the wrong parameter."
                    )
                if t.device != p.device:
                    raise ResumeMismatchError(
                        f"rank{rank}: '{key}' on {t.device} but parameter on {p.device}"
                    )
            if float(st["exp_avg_sq"].abs().max()) == 0.0:
                n_zero_v += 1
            s = st.get("step")
            steps_seen.add(int(s.item()) if torch.is_tensor(s) else int(s))

    if n_with_state == 0:
        raise ResumeMismatchError(
            f"rank{rank} restored ZERO optimizer state entries for {n_params} owned "
            "parameters. Resuming would run vanilla-initialised Adam: a WARM RESTART. "
            "Refusing to continue."
        )
    if n_with_state != n_params:
        raise ResumeMismatchError(
            f"rank{rank}: only {n_with_state}/{n_params} owned parameters have restored "
            "optimizer state; the rest would silently restart from zero moments."
        )
    if n_zero_v:
        raise ResumeMismatchError(
            f"rank{rank}: {n_zero_v}/{n_with_state} restored exp_avg_sq buffers are "
            "identically zero, which is what a freshly-initialised moment looks like. The "
            "optimizer state did not survive the round trip."
        )
    if steps_seen != {expected_step}:
        raise ResumeMismatchError(
            f"rank{rank}: restored per-parameter step counters are {sorted(steps_seen)}, "
            f"expected all == {expected_step}. AdamS derives both bias correction and "
            "alpha_t = (step-1)/T from this, so a wrong value silently rewinds the decay "
            "schedule."
        )

    counts = {"params": n_params, "with_state": n_with_state}
    if world > 1:
        t = torch.tensor([n_params, n_with_state], dtype=torch.float64, device=device or "cpu")
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        counts["global_params"] = int(t[0].item())
        counts["global_with_state"] = int(t[1].item())
        if counts["global_with_state"] != counts["global_params"]:
            raise ResumeMismatchError(
                f"globally {counts['global_with_state']}/{counts['global_params']} owned "
                "parameters have optimizer state after resume; some rank's shard was dropped."
            )
    return counts


def summarize_for_log(meta: Dict[str, Any], counts: Dict[str, int]) -> str:
    return (
        f"resumed at step {meta['step']} (next step {meta['step'] + 1}); optimizer state "
        f"verified: {counts.get('global_with_state', counts['with_state'])}"
        f"/{counts.get('global_params', counts['params'])} owned params carry non-zero moments "
        f"with step=={meta['step'] + 1}; benign arg diffs="
        f"{json.dumps(meta['benign_arg_diffs'], default=str)}"
    )
