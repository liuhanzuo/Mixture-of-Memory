# ±SLoRB token-matched pair COMPLETE — both arms at iter 7501 (2026-08-15)

**Recorded by MAIN. Every number below read from the run's own `eval.json` / `args.json` /
training log on wzc1; nothing transcribed from a summary.**

| arm | iter | train_loss | val_loss | **wiki_ppl** | finalization | node |
|---|---|---|---|---|---|---|
| `noslorb` | 7501 | 1.6357 | 1.6691 | **6.6798** | `finalization_done: true` | LOCAL (sm_100) |
| `slorb`   | 7501 | 1.5207 | 1.5847 | **6.1919** | `finalization_done: true` | `.212` (sm_100) |
| **Δ (slorb − noslorb)** | | **−0.1150** | **−0.0844** | **−0.4879 (−7.30%)** | | same disk, same harness |

Both exited cleanly: `[Prefetcher] Stopped.` + `Finalizing masks: 7501it`, no Traceback/OOM,
and `.212` released all 8 GPUs (0 MiB, 0 apps) at 04:18.

## Paired at identical eval iters — 9/9

Not `noslorb@7500` vs `slorb@7300`. At every shared eval point slorb is lower, and the gap
widens monotonically then plateaus:

```
iter   noslorb    slorb     delta
6700    6.4337   6.1577   -0.2760
6800    6.5888   6.1612   -0.4276
6900    6.6329   6.1824   -0.4505
7000    6.6506   6.1871   -0.4635
7100    6.6658   6.1907   -0.4751
7200    6.6709   6.1930   -0.4779
7300    6.6786   6.1907   -0.4879
7400    6.6786   6.1930   -0.4856
7500    6.6798   6.1919   -0.4879
```

**slorb lower at 9/9.** No noslorb-only eval iters; slorb has two extra early points
(6500, 6600) from its earlier resume.

## The erratum is now NARROWED, not repealed

`status/SPARSEFORGE_SLORB_ONLY_DIFFERENCE_ERRATUM.md` flagged that
`_run_sparseforge_tokenmatched_resume.sh:310`'s claim "THE ONLY DIFFERENCE BETWEEN THE TWO ARMS"
was false because the arms resumed from different iters. Both halves of that are now measurable:

**What the args diff shows.** Diffing the two `args.json` end-to-end: **exactly 3 keys differ**,
and two are bookkeeping (`out_dir`, `resume_dir`). The only substantive difference is
**`SLoRB: False` vs `True`**. Everything else is byte-identical, including `srste_decay=0.0`
(so the silent SR-STE branch of `status/SRSTE_SILENT_DEGRADATION_HAZARD.md` was never armed in
either arm), `max_iters=7500`, `seed=1234`, `block_size=4096`, `global_batch_size=256`,
`sparsity_ratio=0.5`, `mask_penalty_mode=nm_2_4`, `mask_hardening_start=5294`.

**What remains true.** The resume origins still differ — noslorb from **6700**, slorb from
**6500** — so slorb ran 200 more iterations of this segment. The claim that survives is therefore:

> Over the 9 eval points both arms share, and with all trainer flags but `SLoRB` identical,
> +SLoRB is lower on held-out wikitext2 ppl at every point, by 0.276–0.488.

The claim that does **not** survive is "the only difference is ±SLoRB" stated without
qualification: 200 iterations of extra segment history is a second difference, even if a small
one, and it is not something this pair can separate from the SLoRB effect.

## Still pending before this becomes a paper row

The union-9 zero-shot chain for slorb has **not** run yet. Watcher PID 176751 (`ARM=slorb`,
`GPUS=4,5,6,7`) uses a **log-staleness** trigger, not process death:
`LOG_STALE_S` defaults to 1800 s (`_run_sparseforge_tokenmatched_union9_watcher.sh:216`) and is
not overridden in its environment, so it waits for 30 minutes of log silence. Training last wrote
at 04:18, so the chain is expected to fire ~04:48 and then produce
`outputs/cast_eval_spec/sparseforge_tokenmatched_slorb/tokenmatched_union9_summary.json`.
That directory was pre-checked and is **empty**, so nothing stale can be mistaken for the result.

noslorb's reference row, for same-caliper comparison once slorb's lands:
**union9 mean_primary 0.5955 / mean_plain_acc 0.5586**, cast7 0.5676, ast7 0.6065,
`verify_pre_rc=0`, `verify_post_rc=0`, `exact_2of4_tile_ratio=1.0`, 9/9 tasks,
lm_eval 0.4.8 / bf16 / 0-shot / seed 0.
