# SR-STE silently degrades to plain Adam on shard mismatch (CAST path raises, SR-STE path does not)

**Recorded 2026-08-15 by MAIN, 0 GPU. Verified by reading `adamw.py` on wzc1, not inferred.**

## The asymmetry

`/apdcephfs_wzc1/share_304376610/pighzliu_code/adamw.py` handles weight↔mask alignment twice,
with **opposite** failure behaviour:

| path | line | on `mask.numel() != param.numel()` |
|---|---|---|
| **SR-STE** (`self.decay`, i.e. AST's rule) | ~102-112 | `else: mask = None` → **silently skips the SR-STE term**, training continues as plain Adam |
| **CAST** (`cast_mode`) | ~165-180 | `raise RuntimeError("[CAST AdamS] weight/mask shard mismatch ... Do not run CAST with unaligned FSDP shards.")` |

Verbatim from the SR-STE path:

```python
mask = getattr(p, 'mask', None)
if mask is not None:
    decay = self.decay
    if mask.shape != param_data.shape:
        if mask.numel() == param_data.numel() and param_data.numel() > 0:
            mask = mask.view_as(param_data)
        else:
            mask = None          # <-- silent
    if mask is not None:
        grad = grad.add(param_data * (1 - mask), alpha=decay)
```

## Why this matters

`CAST_REPRODUCTION_AUDIT.md` §4.1 attributes the CAST reproduction failure to **exactly this
pattern**: element-wise weight↔mask pairing inside the optimizer under FSDP sharding silently
disabled the algorithm on most weights, produced a **plausible-looking loss curve**, and only
showed up as a garbage PPL (23.45) at the end. The CAST path was subsequently hardened to raise.
**The SR-STE path was not.**

## Why it is currently inert — and exactly when it stops being inert

`SPARSEFORGE_TOKENMATCHED_PREP.md` §TASK 3 concludes FSDP is safe *for SparseForge*, and that is
correct, but the **stated reason is that `srste_decay = 0.0` makes this code path unreachable**.
So the protection is a *configuration* accident, not a code property:

> **Turning SR-STE on (any AST-style arm, any `--srste_decay > 0`) removes the protection.**

There is no `srste_aligned_count` diagnostic, so a degraded run is not distinguishable from a
healthy one by inspecting the log.

## What to do before any SR-STE run

1. Make the SR-STE path **raise**, matching the CAST path (one-line symmetry fix), **or** emit an
   `srste_aligned_count` and assert `== 224` (the SparseLinear module count at 7B).
2. Run a **20-step GATE0** (~0.05 GPU-h) asserting 224/224 alignment before committing to a long
   run. The failure mode is silent, so a cheap probe is the only thing that separates
   "algorithm ran" from "algorithm was skipped".
3. Note `deepspeed` is **not installed** (`ModuleNotFoundError`), so the ZeRO-1 path that was the
   only config which fit CAST at 7B is not currently available as a fallback.

## Provenance

Surfaced by the 0-GPU recon for task #245 (agent `a00457655f646502b`) and re-verified by MAIN by
reading both code paths. No code changed by this note — it is a hazard record, not a fix.
Related: `CAST_REPRODUCTION_AUDIT.md` §4.1, `SPARSEFORGE_TOKENMATCHED_PREP.md` §TASK 3,
`memory/read-the-trainer-docstring-before-designing-a-control.md`.
