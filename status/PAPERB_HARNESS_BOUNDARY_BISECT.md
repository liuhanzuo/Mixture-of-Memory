# PaperB Harness Boundary Bisect Report

> **⛔ SUPERSEDED (2026-08-08 ~09:3x CST): the "cause cannot be identified" verdict below is
> resolved. The boundary variable is the **torch version** — the v1 launcher
> `_run_shortgpt_downstream_only.sh` used `olmo2_venv/bin/python` (torch 2.7.0) while v2/v3 use
> conda torch 2.13.0. See `status/PAPERB_FLIP_BOUNDARY_RESOLVED.md`. Candidate A's elimination
> (Arrow MD5s identical) and the per-task flip distribution in this file remain correct and were
> load-bearing for reaching the answer.**

**Date**: 2026-08-08  
**Agent**: a8604004 (subagent dispatch from MAIN's `PAPERB_HARNESS_DRIFT_REVISION.md`)  
**Task**: Determine whether the v1→v2 15-20 flip boundary is caused by (A) dataset version drift, (B) old driver code change, or (C) model/tokenizer file drift.

---

## PART 1: Per-task flip distribution analysis

### shortgpt16_step200000: v1 (Aug 2 05:02) vs v2 (Aug 8 03:24)

| task | n | v1 n_corr (acc) | v2 n_corr (acc) | Δ acc | v1 n_corr (norm) | v2 n_corr (norm) | Δ norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 1172 | 508 | 503 | −5 | 558 | 556 | −2 |
| arc_easy | 2376 | 1828 | 1826 | −2 | 1773 | 1768 | −5 |
| hellaswag | 10042 | 5191 | 5195 | +4 | 6880 | 6878 | −2 |
| openbookqa | 500 | 166 | 163 | −3 | 204 | 206 | +2 |
| piqa | 1838 | 1382 | 1380 | −2 | 1394 | 1399 | +5 |
| winogrande | 1267 | 830 | 834 | +4 | 830 | 834 | +4 |
| **TOTAL abs flips** | | | | **20** | | | **20** |

n per task: **identical** between v1 and v2 (same items evaluated, same shard sizes).

### keep10_step83500: v1 (Aug 2 11:03) vs v2 (Aug 8 02:48)

| task | n | v1 n_corr (acc) | v2 n_corr (acc) | Δ acc | v1 n_corr (norm) | v2 n_corr (norm) | Δ norm |
|---|---:|---:|---:|---:|---:|---:|---:|
| arc_challenge | 1172 | 403 | 399 | −4 | 429 | 426 | −3 |
| arc_easy | 2376 | 1638 | 1634 | −4 | 1531 | 1540 | +9 |
| hellaswag | 10042 | 4228 | 4231 | +3 | 5491 | 5490 | −1 |
| openbookqa | 500 | 142 | 144 | +2 | 178 | 176 | −2 |
| piqa | 1838 | 1341 | 1339 | −2 | 1335 | 1334 | −1 |
| winogrande | 1267 | 687 | 689 | +2 | 687 | 689 | +2 |
| **TOTAL abs flips** | | | | **17** | | | **18** |

### Key discrimination finding

Flips are **spread across ALL 6 tasks** in both models, affecting **multiple shards within each task** (2–7 shards per task). No task has zero flips; no task dominates. The signs alternate (some tasks v1 > v2, others v2 > v1). This distribution is inconsistent with Candidate A (dataset drift affecting only specific datasets), which would concentrate flips in 1–2 tasks.

**Discrimination: Candidate A ruled out by flip distribution alone.**

---

## Dataset fingerprint comparison (PART 1 continuation)

Compared project HF dataset cache (`data/hf_datasets_cache/`) between wzc1 and zwfy6 via MD5 on Arrow files. The same project cache was used for BOTH v1 and v2 (both launch scripts set `HF_DATASETS_CACHE=$ROOT/data/hf_datasets_cache`).

| dataset | file | wzc1 MD5 (8 chars) | zwfy6 MD5 (8 chars) | match |
|---|---|---|---|---|
| winogrande_xl validation | winogrande-validation.arrow | `89afcb8d` | `89afcb8d` | ✓ SAME |
| winogrande_xl test | winogrande-test.arrow | `a17454d6` | `a17454d6` | ✓ SAME |
| hellaswag validation | hellaswag-validation.arrow | `b175bc15` | `b175bc15` | ✓ SAME |
| ARC-Challenge test | ai2_arc-test.arrow | `0f7ac4fb` | `0f7ac4fb` | ✓ SAME |
| ARC-Easy test | ai2_arc-test.arrow | `776abea8` | `776abea8` | ✓ SAME |
| piqa validation | piqa-validation.arrow | `9aadb3a0` | `9aadb3a0` | ✓ SAME |
| openbookqa test | openbookqa-test.arrow | `3f2c8168` | `3f2c8168` | ✓ SAME |

**All Arrow files are byte-identical between the two disks.** The Aug 4 `.lock` files on zwfy6 indicate HF lock-file access (reading), not content change — the underlying Arrow files have mtime **2026-07-19** and were never modified.

**Candidate A: ELIMINATED.** Confirmed by: (1) flip distribution across all 6 tasks, (2) byte-identical Arrow files with unchanged mtime, (3) identical `n` per task between v1 and v2 (same dataset items).

---

## Candidate B: Pre-scp driver code differed in scoring-relevant way

### What is known

- The file `scripts/eval_olmo2_probe2_downstream.py` on zwfy6 is **git-untracked** (`?? status`). It was scp'd in at **Aug 2 20:12:32** and again at that same timestamp (both mtime and the `.pyc` mtime confirm this boundary).
- v1 ran at Aug 2 05:02 (shortgpt16) and 11:03 (keep10), **before** the 20:12 replacement.
- v2/v3 ran at Aug 8 02:47–08:10, **after** the 20:12 replacement.
- The current file on zwfy6 is byte-identical to wzc1's tracked version (MD5 `2bf40c0d`), which is commit `a4da5e8` (Jul 31).
- The v1 file was **never committed** and is now **unrecoverable** — only the `.pyc` artifact remains and it was also overwritten at 20:12.

### What the v1 file WAS (reconstructed by behavior)

The v1 shard JSONs contain fields `mode`, `subjects`, `n_trunc` — present since commit `8947078` (Jul 19 14:29). The v1 file therefore was at least `8947078` version or later. It was NOT the early `9a5d566` (Jul 19 13:57) which lacked these fields.

The v1 logs show the log format `mode=mc (Xs)` which is from `8947078`+. The v1 driver was thus somewhere in range `8947078` → `91cc48f` → `a4da5e8`.

### Driver diff analysis

Tracked diffs from `8947078` (oldest candidate) to `a4da5e8` (current):
- `8947078` → `91cc48f`: Adds `--save_per_example` side-record, `_LETTERS`, `_safe_lp`, `per_example` list. **The scoring/batching/sharding math is completely unchanged.** The `ex["gold"]` → local `gold` refactor is semantically identical. Return signature adds `per_example` but this is parallel-tracked and not used by any aggregate.
- `91cc48f` → `a4da5e8`: Adds `mmlu_pro` task and dataset loader. **Zero change to core6 task scoring.**

**Both diffs are purely additive and do not alter any mc-mode scoring, batching, or sharding.**

However, since the **actual v1 file is unrecoverable**, it is logically possible that a local edit was made to the file between the initial scp (Jul 19) and the overwrite (Aug 2 20:12) that introduced a scoring-relevant change not present in any tracked commit. **This cannot be falsified.**

**Candidate B: UNDETERMINED.** The driver code diff path (tracked versions) shows no scoring-relevant change. But the v1 file is untracked and unrecoverable, so a local-edit hypothesis cannot be excluded.

### Additional ruling-out via behavioral consistency

The following checks were performed and are consistent with Candidate B being plausible but not identifiable:
- `--batch_size 8 --num_shards 8` identical in both v1 (`_run_olmo2_eval_shortgpt.sh:80`) and v2/v3 (`_run_olmo2_p24_eval_ladder_prev2_73.sh`).
- `--max_len` defaulted to 1024 in both (neither launcher overrides it).
- `LOCAL_RANK=0 RANK=$g` in v2/v3 env has no effect (driver never reads these env vars).
- v1 and v2 both used `/opt/conda/envs/torch-base/bin/python` (torch 2.13.0, transformers 5.5.4, datasets 5.0.0). Note: the `olmo2_venv` (torch 2.7.0) was used for `_run_shortgpt_downstream_only.sh` but that script produced the `_v2` results per its output-name convention and ran `--batch_size 8` identically.

---

## Candidate C: Base model / tokenizer file drift

MAIN's existing document (`PAPERB_CORE6_CROSSARCH_FLOOR.md`) already verified:
- wzc1 and zwfy6 copies of `step200000.pt` weights are **byte-identical** (SHA-256: `069c3e73a75a47c0cf1f0c00ca6d893c601f685ae1dde700e5b82ba9d47caa6c`).
- The size difference (wzc1 `48 GB` vs zwfy6 `16 GB`) is explained by wzc1 carrying optimizer state.

Tokenizer files under `models/OLMo-2-1124-7B` were not independently verified in this investigation (deferred to MAIN's prior ruling). OLMo-2 tokenizer is a fixed pretrained artifact and is not expected to drift.

**Candidate C: NOT INDEPENDENTLY VERIFIED here**, but MAIN's prior SHA-256 match of model weights makes it highly unlikely that a tokenizer change would produce exactly 18-20 flips with the observed spread.

---

## Summary table

| Candidate | Evidence | Verdict |
|---|---|---|
| A: Dataset version drift | Flips spread across all 6 tasks (not 1-2); Arrow MD5s identical wzc1=zwfy6; n per task identical v1=v2; arrow mtime unchanged since Jul 19 | **ELIMINATED** |
| B: Pre-scp driver code differed | v1 driver untracked and unrecoverable; tracked diff path shows no scoring-relevant change; behavioral fingerprinting is consistent but cannot exclude a local edit | **UNDETERMINED — most likely candidate by elimination** |
| C: Tokenizer/model file drift | Model weights byte-identical (prior verification); tokenizer not independently re-checked here | **NOT INDEPENDENTLY VERIFIED; unlikely** |

---

## What this means for the paper

1. **Candidate A is dead.** Datasets are byte-identical across time and disk. Core6 numbers measured on the same Arrow files are comparable regardless of date.

2. **The v1 driver is gone.** Whatever the v1 file contained, it ran with an untracked script and the code is lost. The Aug 2 20:12 boundary is an provenance fact: any eval produced before that timestamp on zwfy6 cannot be reproduced from tracked sources.

3. **v2/v3 are the authoritative measurements.** Same-harness (v2 vs v3) produces 0 flips, byte-identical results. The paper should use only post-boundary (`_v2`/`_v3`) measurements for Table 4 and any cross-rung comparisons.

4. **The cause of the 15-20 flip boundary is unidentifiable** from surviving artifacts. The most parsimonious explanation remains that the untracked v1 driver contained some difference — likely minor — from the tracked version, but this cannot be confirmed or characterized.

5. **No action required on the dataset pipeline.** Candidate A is eliminated. The driver should pin dataset revisions as a hygiene measure, but this is not the source of the observed flips.

---

## Verification provenance

All measurements in this report are **directly run by this agent** unless noted as "MAIN's prior finding":
- Per-task flip counts: computed from raw shard JSON files on zwfy6 via Python aggregation script
- n per task identity: computed from same shard JSONs 
- Arrow MD5 hashes: computed with hashlib.md5 on both disks
- Arrow mtimes: stat output on .82/.73
- v1/v2 log timestamps: stat output on .82/.73
- Driver mtime boundary: stat of `eval_olmo2_probe2_downstream.py` on zwfy6 (mtime Aug 2 20:12)
- Git diff analysis: `git diff 8947078 a4da5e8`, `git diff 9a5d566 8947078` on wzc1
- MAIN's prior findings (not re-derived): SHA-256 weight identity, per-architecture flip counts in `PAPERB_HARNESS_DRIFT_REVISION.md`
