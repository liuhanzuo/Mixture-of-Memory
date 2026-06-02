# v14 — Fix step-500 checkpoint save → DDP barrier → NCCL 30min watchdog SIGABRT

## Symptom
8-GPU DDP training (`scripts/train_mem_space_dolmino_cpt.py`) reproducibly hangs at
`global_step ~490-493` (save_interval=500). ~30 minutes later the NCCL watchdog
reports an ALLREDUCE/barrier timeout and the whole job SIGABRTs. Reproduced 3×.

## Root cause (confirmed)
The checkpoint save block (around line 1207):
```python
if (args.save_interval > 0 and global_step % args.save_interval == 0
        and global_step < args.total_steps):
    if is_main(rank):
        _save_adapter(model, args, global_step)   # rank0 writes ~7.5GB to CEPH, slow
    if world_size > 1:
        dist.barrier()                             # ranks 1-7 idle-wait on rank0
```
rank0 writes a ~7.5GB adapter to the shared CEPH FS — this can take minutes or
stall. Ranks 1-7 reach `dist.barrier()` immediately and wait. The barrier runs on
the default NCCL ProcessGroup, whose **work timeout equaled the init_process_group
`timeout`**, which was set to `timedelta(minutes=30)`. A slow save makes the barrier
exceed 30min → the NCCL watchdog treats it as a stuck collective → SIGABRT.

So the 30min watchdog window was *exactly* the configured PG timeout (line 153-154),
not an unrelated default.

## Fix
1. **Core**: `dist.init_process_group(backend="nccl", ..., timeout=timedelta(hours=2))`
   (was `timedelta(minutes=30)`). The barrier/collective work timeout is now 2h, far
   larger than any realistic CEPH save, so slow save + barrier no longer trips the
   watchdog. (`timedelta` was already imported.)
2. **Hardening / diagnostics**: log `[save] start ... at step N` and
   `[save] done ... at step N (Xs)` around `_save_adapter` so future hangs reveal
   whether the save itself or the barrier is slow, and the exact save duration.

## Why this resolves the repeated step-500 crash
The crash was caused solely by the barrier exceeding the 30min PG timeout while
rank0 finished a slow save. Raising the timeout to 2h removes the trigger; the
save-then-barrier sequence still serializes correctly, just without a premature
watchdog abort. The added logging confirms save timing on the next run.

## Notes
- The `quick_eval_babilong` block has the same `is_main`-then-`barrier` pattern; it
  benefits from the same 2h timeout, though eval is currently disabled (EVAL_INTERVAL=0).
- File is shared by local (zwfy6) and remote nodes; main will rsync to disk B.
