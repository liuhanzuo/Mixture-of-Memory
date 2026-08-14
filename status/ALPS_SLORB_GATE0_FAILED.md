> # ⛔ SUPERSEDED / RETRACTED 2026-08-15
>
> **This verdict was wrong. Both failure signals below are artifacts of the probe's own
> configuration, not properties of the ALPS+SLoRB arm.** See
> **`status/ALPS_SLORB_GATE0_VERDICT.md`** for the corrected adjudication (GATE0 **PASSED**,
> `aligned=224/224`, twice, both `rc=0`).
>
> - "Failure 1: only 96 of 224 masks visible" — `processed`/`skipped` in `nm_2_4_tile_stats` are
>   **rank-0-local** counters incremented *before* the FSDP `all_reduce`. The reduced
>   `total_tiles = 1619001344` on the same line equals `elems/4 = 6476005376/4` **exactly**, so
>   every tile was counted across ranks. Nothing was lost.
> - "Failure 2: loss frozen at 27.9" — the probe passed `--output_flip_every 1000000`, and the
>   tqdm postfix is only refreshed inside `if iter_num % args.output_flip_every == 0`
>   (`main_llama.py:2917`). It was **one iter-0 value redrawn 24 times.** Re-running with
>   `--output_flip_every 1` gives a moving, descending loss (27.9 → 24.0). Also, the logged
>   number is `task_loss + 4*KL`, not a cross-entropy, so `exp(27.9)` is not a perplexity and
>   the `PPL > 1000` rule does not apply.
> - The `SLoRB`-inits-to-zero concern was also disproved by measurement: with the **real ALPS
>   mask** installed, `max|W*(1-mask)| = 1.240e-01` and `max|SLoRB_Weight| = 2.666e-01` (it is
>   only zero in the all-ones-mask context, which is not this path).
>
> Kept verbatim below as provenance. **Do not cite it as the outcome.**

---

# GATE0 FAILED for ALPS+SLoRB — the gate did its job (0.06 GPU-h, 2026-08-15)

**Verdict: DO NOT proceed to a long run.** The 20-step probe returned two independent
failure signals. Recorded by MAIN from the live log, not from the agent's summary.

Run: `logs/alps_slorb_gate0_20260815_021407.log`, 20 steps on LOCAL GPUs 0-3
(`CUDA_VISIBLE_DEVICES=0,1,2,3` verified via `/proc/2334521/environ`), process has exited.
GPUs 4-7 were never touched (watcher PID 176751 still holds them for the `.212` slorb arm).

## What PASSED — STEP 1, the mask itself

The agent did **not** need to generate a mask: a real ALPS 2:4 mask already existed at
`outputs/paper_v2/alps/llama2_wandb_sf_alps_v1_alps_seed0/mask.pt` (6,607,138,721 bytes,
dated Jul 31). All five of my required checks pass, recorded in
`SparseForge_Data/results/alps_mask_seed0/mask_validation.json`:

| check | measured |
|---|---|
| `format == sparseforge-mask-v1` | ✅ |
| `pattern == 2:4` | ✅ |
| nnz-per-group-of-4 histogram | **`{2: 1619001344}`** — exactly two per group, nothing else |
| in-scope module count | **224** (plus `lm_head` correctly out of scope; 225 entries total) |
| zero fraction | **0.5** exactly (nnz 3,238,002,688 / elems 6,476,005,376) |

So the mask is sound. **The failure is downstream of it.**

## Failure 1 — only 96 of 224 masks are visible to this rank

```
[nm_2_4_tile_stats DEBUG] sparse_linear_count: 224, processed: 96, total_tiles: 1619001344.0
[nm_2_4_tile_stats DEBUG] skipped 128 masks, first 3:
  [('...layers.0._fsdp_wrapped_module.self_attn.o_proj', torch.Size([0]), 1),
   ('...layers.0._fsdp_wrapped_module.mlp.gate_proj',   torch.Size([0]), 1),
   ('...layers.0._fsdp_wrapped_module.mlp.up_proj',     torch.Size([0]), 1)]
```

`96 + 128 = 224` exactly, and every skipped entry has `torch.Size([0])` — this is **FSDP
shard visibility**, not a lost mask. My gate required **224/224 aligned**; this is **96/224**.

This is the *same class* of hazard as `status/SRSTE_SILENT_DEGRADATION_HAZARD.md`: a
per-rank shard makes a mask invisible, the code proceeds anyway, and nothing raises. Note the
run reached `[Training End] finalizing masks... will finalize 224 SparseLinear modules` and
printed `sparsity=0.5000` per module — **the finalization report says 224 while the training-time
tile stats say 96.** A run that only read the finalization line would look healthy.

## Failure 2 — the loss is frozen at a garbage value

`loss = 27.9` on **all 24 readings across all 20 steps**, byte-identical, `flip_ratio = 0`
throughout. `exp(27.9) ≈ 1.3e12`.

Per `CLAUDE.md`'s own PPL rule, `PPL > 1000` means "模型已经不会说话了 … 先不要调
hyperparameter". And a loss that does not move *at all* over 20 steps means gradients are not
reaching any trainable weight. With `--freeze_non_slorb True`, the **only** trainable path is
the SLoRB branch — and SLoRB initialises to zero. The recon for the AST arm already measured
that `init_type=sum` gives `SLoRB_Weight = (W*(1-mask)).sum(dim=2) = 0` when the checkpoint is
already exactly 2:4. If the same degeneracy holds here, the forward pass is being computed
through a masked-out weight with a zero-valued side branch, and there is nothing to learn.

**These two failures are consistent with one cause and must not be reported as "it ran".**

## What must happen before any long ALPS+SLoRB run

1. **Make the 96/224 discrepancy loud.** Either gather the mask across shards before the tile
   stats, or assert `processed == sparse_linear_count` and exit non-zero. Right now the
   training-time count and the finalization count disagree silently.
2. **Establish that SLoRB is non-degenerate at init on THIS mask.** Measure
   `max|W*(1-mask)|` on the ALPS-masked Llama-2 weights. If it is 0, `init_type=sum` produces a
   zero branch and `--freeze_non_slorb` leaves the model with no trainable signal — the arm is
   ill-posed as configured, and the fix is an init/config change, not more steps.
3. **Re-run GATE0 and require: `processed == 224`, loss strictly decreasing over 20 steps,
   `flip_ratio` not identically 0.** Only then price the long run.

## Cost accounting

20 steps at ~25.1 s/it = **8.4 min wall on 4 cards ≈ 0.56 GPU-h**. That is what the gate cost,
against the ~226 GPU-h a blind long run would have burned before the same two signals appeared
at the end. This is the CAST failure mode (plausible curve, garbage final PPL) caught at step 20
instead of step 7500.
