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

## Union-9 zero-shot: BOTH arms measured (2026-08-15 05:00)

The watcher fired at 04:48 as predicted and completed all 6 stages at 05:00
(`STAGE 4 DONE rc=0`, `STAGE 5 DONE`, summary written). Caliper verified identical, not assumed:

- `harness` string **byte-identical**: `lm_eval 0.4.8, --model hf, dtype=bfloat16, parallelize=True, add_bos_token=False, --batch_size auto, --num_fewshot 0, --seed 0`
- same 9-task list, same `primary_metric` per task, same `n_samples` per task (0 mismatches)
- `source_iter = 7501` for **both** arms; `scope_elements = 6476005376` and `scope_tensors = 224` for both

| task | metric | n | noslorb | slorb | Δ |
|---|---|---|---|---|---|
| arc_challenge | acc_norm | 1172 | 0.4130 | 0.4386 | +0.0256 |
| arc_easy | acc_norm | 2376 | 0.7121 | 0.7306 | +0.0185 |
| boolq | acc | 3270 | 0.7043 | 0.7107 | +0.0064 |
| hellaswag | acc_norm | 10042 | 0.6891 | 0.7009 | +0.0118 |
| openbookqa | acc_norm | 500 | 0.3780 | 0.4040 | +0.0260 |
| piqa | acc | 1838 | 0.7399 | 0.7546 | +0.0147 |
| race | acc | 1045 | 0.3742 | 0.4000 | +0.0258 |
| rte | acc | 277 | 0.6823 (189/277) | 0.7292 (202/277) | +0.0469 |
| winogrande | acc | 1267 | 0.6669 | 0.6701 | +0.0032 |
| **union9 mean_primary** | | | **0.5955** | **0.6154** | **+0.0199** |
| union9 mean_plain_acc | | | 0.5586 | 0.5761 | +0.0176 |
| cast7 mean_primary | | | 0.5676 | 0.5855 | +0.0179 |
| ast7 mean_primary | | | 0.6065 | 0.6263 | +0.0198 |

**slorb better on 9/9 tasks.** RTE integral check passes on both (189.0 and 202.0, both within
0.01 of an integer).

Offline harness PPL, both seqlens, both arms — and it independently reproduces the training-time
number to ~3e-4, which is a cross-check the training log alone could not give:

| seqlen | noslorb | slorb | Δ | tokens |
|---|---|---|---|---|
| 4096 | 6.679524 | 6.193779 | −0.485745 (−7.27%) | 335872 |
| 2048 | 7.147421 | 6.604782 | −0.542639 (−7.59%) | 335872 |

⚠️ The headline Δ in the table at the top of this file is a **seqlen-4096** comparison. Both arms
are 4096, so the Δ is valid — but `SPEC.md:213`'s "the whole PPL column is 2048" assumption is
false here too, and the 2048 pair is a *different* Δ. Always carry the seqlen label.

## ⚠️ This is NOT a matched-capacity comparison — the +1.99 pp is bought

My own earlier pairing plan said to check that slorb's `verify_pre_rc`/`verify_post_rc`/
`exact_2of4_tile_ratio` match noslorb's `0/0/1.0`. **That was wrong, and wrong by design** —
the watcher's own header (lines 77-88) distinguishes two FAIL modes, and for a folded arm FAIL is
the *expected* outcome. Measured:

| arm | `2of4_eligible` | pre_rc | post_rc | `linear_zero_ratio` | `exact_2of4_tile_ratio` | `exact_2of4_violations` |
|---|---|---|---|---|---|---|
| noslorb | `true` | 0 | 0 | 0.5 | 1.0 | 0 |
| slorb | `false` | 2 | 2 | 1.08e-09 | 0.0 | 1619001344 |

`1619001344 == 6476005376/4` exactly — **every tile** is violated. The fold is exact linear
algebra (`W_eff = W*mask + SLoRB_Weight @ x_proj`, `export_sparseforge_to_hf.py:62`), so the
quality numbers faithfully represent what SparseForge deploys; but the deployed artifact is
**dense**, and `max_abs_slorb_effective = 0.0293` confirms the branch is materially nonzero,
not a no-op.

Parameter cost, counted from the checkpoints themselves (`model_state_dict`, not from the
docstring):

| arm | `SLoRB_Weight` | `x_proj` | extra live params |
|---|---|---|---|
| slorb | 224 tensors / 404,750,336 | 224 tensors / 443,678,720 | **848,429,056 (848.4 M)** |
| noslorb | 0 | 0 | **0** |

848.4 M on top of 3,238,002,688 surviving weights = **+26.2%**. noslorb's zero confirms `drop`
was a true no-op there rather than silently removing something.

⇒ **The correct statement is:** +SLoRB gains +1.99 pp union-9 and −0.486 ppl@4096, at the cost of
+26.2% live parameters and the complete loss of 2:4 structure. It must **never** appear in a 2:4
column, and it is not an iso-parameter ablation. The watcher enforces the first half of this
mechanically (`** 2:4 COLUMN: BARRED **`); the parameter cost has to be carried in prose.

## Correction to this file's own erratum section (same day)

The section above states the resume origins differ (6700 vs 6500) and treats that as a surviving
second difference. That remains the honest description of the **segment history**, but the
endpoint is now measured **iter-matched**: `source_iter_num = 7501` in both
`sparseforge_export_meta.json` files. So the two arms are compared at the same iteration with the
same token budget; what differs is 200 iterations of extra *within-segment* history for slorb.
The 848 M parameter gap above is a far larger confound than those 200 iterations, and it is the
one that must lead any write-up.

Provenance: `outputs/cast_eval_spec/sparseforge_tokenmatched_{noslorb,slorb}/tokenmatched_union9_summary.json`,
`outputs/sparseforge_tokenmatched_{noslorb,slorb}_hf/hard_{drop,fold}/sparseforge_export_meta.json`,
`logs/sparseforge_tm_union9_slorb_progress.log`.
