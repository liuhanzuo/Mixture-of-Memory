# PAPERB_RESUME_ON_B200_21.md — Paper B resume migration: H20 → .21 (B200/L20A)

> Created: 2026-08-08. User instruction: "B200可以拿去直接跑resume 然后H20跑得比较慢 可以跑你的新方向. 其他H20你想用可以随时kill paperB的resume"

## Summary

- **Strategy**: Option (c) — `.21` runs **keep10** (largest gap: 116.5k steps); `.73` runs **keep8** (H20 idle, faithful resume with optimizer); `.82`/`.104` continue keep10/keep12 until user kills them for new direction
- **Rationale**: `.21` (8×L20A 183GB) is fastest node, given to the arm with most remaining work (keep10: 83.5k→200k = 116.5k steps). `.73` is idle H20, keep8's `step121000_full.pt` is already on zwfy6 with optimizer state. keep12 on `.104` continues undisturbed (user can kill anytime).

## PART 1: Data Transfer Status

### Transfer 1: dolmino_now15b.npy (15.5M rows, 118 GB)
- **Source**: `.73` `/dev/shm/dolmino_now15b.npy` (zwfy6, shape=(15491607, 2048))
- **Dest**: `.21` `/dev/shm/dolmino_now15b_zwfy6.npy` (NEW name with `_zwfy6` suffix, does NOT overwrite wzc1 version)
- **Method**: Local-machine pipe: `.73 cat | .21 cat` (sshpass on both ends)
- **PID**: local machine PID 2568652 (bash transfer_dolmino_pipe.sh)
- **Started**: 2026-08-08 14:52 CST
- **Progress** (14:57): 4.9 GB / 118 GB ≈ 4%, speed ~1 GB/min
- **ETA**: ~113 minutes from 14:57 ≈ arrives ~16:50 CST
- **Status**: IN PROGRESS

### Transfer 2: step124000.pt (41 GB)
- **Source**: `.73` `/apdcephfs_zwfy6/share_304376610/.../outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt`
- **Dest**: `.21` `/apdcephfs_wzc1/share_304376610/.../outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt`
- **Method**: Local-machine pipe: `.73 cat | .21 cat`
- **PID**: local machine PID 2568653
- **Started**: 2026-08-08 14:52 CST
- **Progress** (14:58): 6.0 GB / 41 GB ≈ 15%, speed ~1 GB/min
- **ETA**: ~35 minutes from 14:58 ≈ arrives ~15:33 CST
- **Status**: IN PROGRESS
- **MD5 verification**: will be computed by transfer_ckpt124k_pipe.sh on completion

### Two-disk context (why we need transfers at all)
- wzc1 `dolmino_now15b.npy` = 7,570,911 rows (62 GB) — WRONG for keep8/10/12
- zwfy6 `dolmino_now15b.npy` = 15,491,607 rows (118 GB) — CORRECT (keep8/10/12 trained on this)
- Ratio: 2.046× — mixing them changes epochs, sampler shuffle, steps/epoch (59147 vs 121028)
- See `status/PAPERB_TWO_CORPORA_DEFECT.md` for full analysis

## PART 2: Training Launch Plan

### .21 — keep10+fresh2 (WAITING for dolmino transfer)
| Parameter | Value |
|---|---|
| Script | `scripts/launch_keep10_resume_b200_21.sh` |
| Ckpt | `outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt` (37 GB, wzc1, has optimizer) |
| Data | `/dev/shm/dolmino_now15b_zwfy6.npy` (15.5M rows, **being transferred**) |
| batch_size/grad_accum | 8 / 2 (eff_batch = 128, same as H20 4/4) |
| Rationale for bs8/ga2 | L20A has 183 GB vs H20 97 GB; keep10 H20 used 82.7 GB maxmem at bs4/ga4. With bs8 per-rank activation ~2× → ~165 GB/card on L20A (fits); ga=2 means fewer gradient accumulation ops → slightly faster per-step. Need to confirm in first step log. |
| Expected s/step | ~3-4s (vs H20 6.8s) — L20A is faster on same model size |
| Status | Launch script PID 4516 on .21; **waiting for dolmino size=126907244672** |

### .73 — keep8+fresh2 (RUNNING)
| Parameter | Value |
|---|---|
| Script | `scripts/launch_keep8_resume_h20_73.sh` |
| Ckpt | `outputs/olmo2_probe2_7B_keep8fresh2/step121000_full.pt` (28 GB, zwfy6, HAS optimizer) |
| Data | `/dev/shm/dolmino_now15b.npy` (already in /dev/shm on .73, 15.5M rows) |
| batch_size/grad_accum | 4 / 4 (eff_batch = 128, same as original) |
| PID | 3319196 (launcher) |
| Started | 2026-08-08 14:40:36 CST |

### Confirmation lines from keep8 log (VERIFIED ✓)
```
[resume] optimizer state REMAPPED 2-group -> 4-group (113/113 param states, Adam moments preserved)
[optim] group fresh_decay: 815.8M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group fresh_nodecay: 0.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_decay: 2030.0M params base_lr=2.00e-05 min_lr=2.00e-06
[optim] group inh_nodecay: 0.1M params base_lr=2.00e-05 min_lr=2.00e-06
dataset rows=15491607
[step 121020/200000] loss=2.5703 ppl=13.07 lr=8.09e-06 gnorm=0.48 5.85s/step maxmem=73.5GB
```
**All 3 requirements met**: REMAPPED ✓, all base_lr=2.00e-05 ✓, dataset rows=15491607 ✓

## PART 3: H20 Triage

| Node | Arm | Status | Action |
|---|---|---|---|
| `.82` | keep10 | Running at step 83980 (6.80s/step, maxmem=82.7GB) | **Kill AFTER** `.21` keep10 is confirmed running at correct step with REMAPPED/LR/rows checks |
| `.104` | keep12 | Running at step 124220 (7.87s/step, maxmem=91.9GB) | Leave running. User can kill when new direction needs the card. |
| `.73` | keep8 | **Running ✓** (migrated to .73) | Keep running, user can kill if new direction needs .73 |

### When to kill .82 keep10
1. `.21` dolmino transfer completes (ETA ~16:50)
2. `.21` keep10 training starts and shows first step log
3. Confirm REMAPPED ✓ / all base_lr=2.00e-05 ✓ / dataset rows=15491607 ✓
4. Then: `kill -9 <PID 1418803>` on `.82` (NOT pkill)
5. Append kill record to gpu_runs.jsonl

## .21 vs H20 Speed Comparison
H20 reference:
- keep10 on H20 (.82): 6.80s/step, maxmem=82.7GB at bs4/ga4
- keep12 on H20 (.104): 7.87s/step, maxmem=91.9GB at bs4/ga4
- keep8 on H20 (.73): 5.85s/step, maxmem=73.5GB at bs4/ga4

Expected on .21 (L20A 183GB, bs8/ga2, same eff_batch=128):
- Predicted: ~3-4s/step (larger batch → better GPU utilization, L20A has higher bandwidth)
- Actual: **TBD — will fill in after keep10 starts on .21**

## PART 4: Files created this task

| File | Purpose |
|---|---|
| `scripts/launch_keep10_resume_b200_21.sh` | keep10 launch script for .21 (waits for dolmino) |
| `scripts/launch_keep12_resume_b200_21.sh` | keep12 launch script for .21 (future use) |
| `scripts/launch_keep8_resume_h20_73.sh` | keep8 launch script for .73 |
| `scripts/transfer_dolmino_pipe.sh` | Transfer dolmino (118GB) via local pipe .73→.21 |
| `scripts/transfer_ckpt124k_pipe.sh` | Transfer step124000.pt (41GB) via local pipe .73→.21 |
| `status/PAPERB_RESUME_ON_B200_21.md` | This file |

## Issues encountered
1. `sshpass` not in PATH on .73 → solved by copying local `sshpass` binary to `.73:/tmp/sshpass`
2. Orphan GPU processes on .73 from earlier crashed training → killed all training procs, GPU freed
3. `--wandb_project` not supported by zwfy6 train script → removed from all launch scripts  
4. rsync binary missing `libiconv.so.2` on .73 → switched to local-machine pipe method
5. dolmino scp killed by SSH session termination (pkill cleanup) → restarted as background pipe
6. Previous migration was cancelled for SparseForge priority → resumed per user 2026-08-08 instruction

## TODO (still in progress)
- [ ] Wait for dolmino transfer to complete (~16:50 CST)
- [ ] Verify keep10 on .21 starts with REMAPPED/LR/rows checks
- [ ] Kill .82 keep10 after .21 confirmed running
- [ ] Update GPU_STATUS.md after .82 kill
- [ ] Record .21 vs H20 s/step comparison
- [ ] Verify step124000.pt md5 (auto-checked by transfer script)
