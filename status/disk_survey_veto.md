# Disk Survey — VETO LIST (lane: WHAT WOULD BREAK)

Started 2026-08-16T20:38:50+08:00
Surveyor lane. READ-ONLY. No deletions performed.

## In progress...

---
## STEP 1 — Live training command lines (verified by pgrep at survey time)

Method: `pgrep -af 'scripts/train_'` locally; via ssh (no `-p`) on .212/.73/.82/.104.
All 5 runs confirmed ALIVE. Every path argument extracted verbatim below.

### LOCAL (wzc1) — keep10fresh2, 8 ranks + torchrun
`scripts/train_olmo2_arch_probe2.py`
- `--resume_from /apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep10fresh2/step90000.pt`
- `--output_dir  /apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep10fresh2`
- `--model_path  /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`
- `--data_path   /dev/shm/dolmino_now15b_wzc1.npy`  (tmpfs, 126,907,244,672 B = 118.19 GiB, mtime Aug 15 10:15)
- `--keep_steps 83500,90000,121000,124000,150000,175000,200000` `--keep_last_n 3 --keep_milestones 8 --save_every 500`

### .212 (28.89.18.212, wzc1) — keep14fresh2_distill
`scripts/train_olmo2_arch_probe2_distill.py`
- `--resume_from /apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep14fresh2_distill/step5000.pt`
- `--output_dir  /apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep14fresh2_distill`
- `--model_path  /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`
- `--distill_teacher_model /apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`  <-- SAME dir loaded TWICE (student init + live teacher). Deleting it kills this run instantly.
- `--data_path /dev/shm/dolmino_now15b_wzc1.npy` (126,907,244,672 B, mtime Aug 15 21:12)
- `--save_every 500` (no --keep_steps / --keep_last_n given -> trainer defaults apply)

### .73 (zwfy6) — keep12fresh2
`scripts/train_olmo2_arch_probe2.py`
- `--resume_from /apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep12fresh2/step166000.pt`
- `--output_dir  /apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep12fresh2`
- `--model_path  /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`
- `--data_path /dev/shm/dolmino_now15b.npy` (NOTE: different filename from wzc1 nodes; 126,907,244,672 B, mtime Jul 16)
- `--keep_steps 83500,121000,124000,150000,175000,200000`
- wrapper alive: `bash scripts/launch_keep12_resume_h20_73_0814.sh` (PID 3913526)
- WATCHER alive: `bash scripts/chain_keep12_eval_200k.sh` (PID 1243702), log `logs/chain_keep12_eval_200k.log`

### .82 (zwfy6) — keep8fresh2
`scripts/train_olmo2_arch_probe2.py`
- `--resume_from /apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep8fresh2/step131000.pt`
- `--output_dir  /apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep8fresh2`
- `--model_path  /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B`
- `--data_path /dev/shm/dolmino_now15b.npy`
- `--keep_steps 83500,121000,124000,150000,175000,200000`
- wrapper alive: `bash scripts/launch_keep8_resume_h20_82_0814.sh` (PID 1329275)

### .104 (zwfy6) — paperC_qwen3base_heal_k8f2  ** RELATIVE PATHS, resolve against zwfy6 root **
`scripts/train_qwen3_arch_probe2.py`  (cwd = /apdcephfs_zwfy6/.../Mixture-of-Memory per forkserver sys_path)
- `--data_path data/slimpajama_chunks_2048_qwen3base_full.npy`  -> zwfy6 `data/slimpajama_chunks_2048_qwen3base_full.npy`
- `--output_dir outputs/paperC_qwen3base_heal_k8f2`              -> zwfy6 `outputs/paperC_qwen3base_heal_k8f2`
- `--model_path /apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen3-8B-Base`
- NO `--resume_from` (fresh from base) ; `--keep_steps 121000` `--keep_last_n 3 --keep_milestones 8 --save_every 500`

### Measured sizes of the hard-locked inputs

| path | disk | measured bytes | GiB |
|---|---|---|---|
| `/dev/shm/dolmino_now15b_wzc1.npy` (LOCAL + .212) | tmpfs, NOT ceph | 126,907,244,672 | 118.19 |
| `/dev/shm/dolmino_now15b.npy` (.73/.82) | tmpfs, NOT ceph | 126,907,244,672 | 118.19 |
| wzc1 `models/OLMo-2-1124-7B` (dir) | wzc1 | see below | — |
| zwfy6 `models/OLMo-2-1124-7B` (dir, `du -sb`) | zwfy6 | 29,204,228,800 | 27.20 |
| zwfy6 `data/dolmino_now_val.npy` | zwfy6 | 33,554,560 | 0.03125 |
| zwfy6 `data/hf_datasets_cache` (`du -sb`) | zwfy6 | 405,101,986 | 0.377 |

NOTE: `/dev/shm` is tmpfs (RAM), NOT counted in the 110T/667T ceph figures. Deleting it frees
zero ceph bytes and instantly kills 4 of the 5 trainers (they mmap it). It is NOT a disk target.

### Live output dirs, actual on-disk contents (measured `ls -la`)

LOCAL `outputs/olmo2_probe2_7B_keep10fresh2/` — 15 ckpt + arch_meta.json, each 39,009,622,410 B
(36.33 GiB): step83500, step90000 (=the live `--resume_from`, PRESENT), 121000, 124000, 140000,
145000, 150000, 155000, 160000, 165000, 170000, 175000, 178000, 178500, 179000.
Dir total from `ls` block count: 571,430,018 KiB ≈ 545.0 GiB.

LOCAL `outputs/olmo2_probe2_7B_keep14fresh2_distill/` — 10 ckpt + arch_meta.json,
each 24,489,781,329 B (22.81 GiB): step5000 (=live `--resume_from`, PRESENT), 10000, 15000,
20000, 25000, 30000, 35000, 37500, 38000, 38500. Dir ≈ 239,157,570 KiB ≈ 228.1 GiB.

.73 `outputs/olmo2_probe2_7B_keep12fresh2/` — 12 ckpt + arch_meta.json, each 43,867,04x,xxx B
(~40.9 GiB): step124000, 150000, 155000, 160000, 165000, 170000, 175000, 180000, 185000,
188000, 188500, 189000. Dir ≈ 514,066,987 KiB ≈ 490.2 GiB.
** ⚠️ `step166000.pt` — the path in the LIVE `--resume_from` — IS ALREADY GONE (rotated out). **
The trainer read it at resume and no longer needs it, so the run is safe, but this proves the
rotation policy is actively deleting ckpts under us: a path being in a live argv does NOT imply
it is still on disk, and conversely, an mtime-fresh ckpt can vanish within hours.

.73 `df -h` on zwfy6: 689T size / 667T used / 22T avail / 97%.

### zwfy6 live output dirs (measured `ls -la` on .82)

.82 `outputs/olmo2_probe2_7B_keep8fresh2/` — 14 files. Regular ckpts 34,152,196,306 B (31.81 GiB):
step124000, 125000, 130000, 135000, 140000, 145000, 150000, 155000, 160000, 161000, 161500,
162000. PLUS two odd ones: `step121000.pt` = 11,384,060,758 B (10.60 GiB, weights-only) and
`step121000_full.pt` = 34,152,195,666 B. Dir ≈ 444,690,058 KiB ≈ 424.1 GiB.
** ⚠️ `step131000.pt` — the LIVE `--resume_from` — IS ALREADY GONE (rotated out). **

.104's target `outputs/paperC_qwen3base_heal_k8f2/` (visible from .82, same disk) — 12 files,
each 38,089,694,111 B (35.47 GiB): step25000, 30000, 35000, 40000, 45000, 50000, 55000, 60000,
61500, 62000, 62500. Dir ≈ 409,166,639 KiB ≈ 390.2 GiB.

zwfy6 `models/Qwen3-8B-Base` = 16,393,040,530 B = 15.27 GiB (`du -sb`).
zwfy6 `data/slimpajama_chunks_2048_qwen3base_full.npy` = 22,164,381,824 B = 20.64 GiB.
wzc1 `models/OLMo-2-1124-7B` = 29,204,228,800 B = 27.20 GiB (`du -sb`, rc=0).

### ★★ RULE DERIVED FROM MEASUREMENT: an argv path is not a disk fact

All THREE resumed runs point `--resume_from` at a ckpt that has ALREADY been deleted by the
trainer's own rotation (`scripts/train_olmo2_arch_probe2.py:555-569`):
  LOCAL keep10   --resume_from step90000.pt   -> PRESENT (still in --keep_steps list)
  .73   keep12   --resume_from step166000.pt  -> **ABSENT** (166000 not in its --keep_steps)
  .82   keep8    --resume_from step131000.pt  -> **ABSENT** (131000 not in its --keep_steps)
  .212  distill  --resume_from step5000.pt    -> PRESENT
Consequence for the cleanup lanes: the resume ckpt is read ONCE at process start. It is NOT a
live file handle. But the converse is the dangerous half -- **the newest 2-3 ckpts in each live
output dir are the ONLY thing standing between a crash and losing days of compute**, and they
rotate every ~500 steps. Any lane that measures a live output dir and then deletes "the old
ones" is racing a writer. VETO the whole directory, not a subset.

## STEP 2 — What `chain_keep12_eval_200k.sh` + `eval_paperb_ladder_200k.sh` will need in ~24h

Watcher is ALIVE on .73: PID 1243702 `bash scripts/chain_keep12_eval_200k.sh`, log
`logs/chain_keep12_eval_200k.log`. Both scripts are BYTE-IDENTICAL on the two disks (md5
`0076ff4b88f75723daf69af770e096d2` chain / `649b602738666b4719a514e8d2b09b84` driver).

The chain waits on `$PROJECT_ROOT/outputs/olmo2_probe2_7B_keep12fresh2/step200000.pt`
(PROJECT_ROOT defaults to the **zwfy6** root) and also reads, for progress display,
`logs/olmo2_7B_keep12fresh2_resume200k_*.log`. It then `cd`s to PROJECT_ROOT and runs the driver.

The driver hard-asserts (all fatal, all before GPU work) — every one of these is a veto entry:

| # | needed path (relative to zwfy6 root) | measured | why fatal |
|---|---|---|---|
| P0 | `/opt/conda/envs/torch-base/bin/python` (torch must be 2.13.x) | n/a (not ceph) | dies if torch != 2.13.x |
| P1 | `scripts/eval_olmo2_probe2_ppl.py` | 20,097 B | `die "missing harness"` |
| P1 | `scripts/eval_olmo2_probe2_downstream.py` | 32,554 B | same |
| P1 | `scripts/_ladder200k_assert.py` | 10,583 B | same |
| P3 | `../models/OLMo-2-1124-7B` = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-1124-7B` | 29,204,228,800 B = 27.20 GiB | `[ -d ]` fatal |
| P3 | `data/dolmino_now_val.npy` | 33,554,560 B = 32.0 MiB | `[ -f ]` fatal AND md5 must equal `f2ea48a2074a2f38fc3b4477fceecf11` — VERIFIED MATCHING on .73 now. Any byte change makes the whole PPL column non-comparable. |
| P5 | `outputs/olmo2_probe2_7B_keep12fresh2/step200000.pt` + `arch_meta.json` | not yet written (~40.9 GiB expected); arch_meta.json = 567 B present | the object of the whole chain |
| P7 | `olmo2_ppl_results/7B_keep12_step200000/` and `olmo2_downstream_results/7B_keep12_step200000{,_know}/` must be ABSENT-or-empty | — | driver REFUSES to run if a `summary.json` is already there. So these three dirs must not be *created*, and existing same-named dirs must not be left half-populated. |
| P8 | ≥5 GiB free on zwfy6 | 22T free now | `die "less than 5 GiB free"` |
| runtime | `data/hf_datasets_cache/` (set as `HF_DATASETS_CACHE`) | 405,101,986 B = 0.377 GiB | deleting it forces a re-download of 11 HF datasets through the proxy mid-eval |
| writes | `logs/`, `olmo2_ppl_results/`, `olmo2_downstream_results/`, `paperB/evidence/` | — | `mkdir -p`'d, but `paperB/evidence/ladder200k_keep12_{ckptmeta,run}.json` is the provenance record |

Note `NUM_SHARDS=8` -> the driver will use all 8 H20 on .73 for ~1 h. It is a GPU consumer, not
just a file consumer: another lane must not schedule .73 for that window.

## STEP 3 — Paper provenance (paperA/ paperB/ paperC/)

`.tex` files name almost NO data paths (only `models/Qwen3-8b-local`, `data/training`). Provenance
lives in the **evidence JSON / MD**, so a tex-only grep would have produced a nearly empty veto
list. Full sweep over paperA+paperB+paperC (json/md/tex/log) yielded 740 distinct path strings;
after dropping `.texlive/` and intra-paper paths, 172 reference `outputs/`, `data/` or `models/`.

### ★★ THE SINGLE MOST IMPORTANT FINDING OF THIS LANE

`/dev/shm/dolmino_now15b_wzc1.npy` (LOCAL + .212) and `/dev/shm/dolmino_now15b.npy` (.73/.82) are
**tmpfs** — 0 ceph bytes, so NOT a cleanup target. But their **on-disk regeneration source is
NOT what the name suggests**:

- wzc1 `data/dolmino_now15b.npy` = 62,020,903,040 B (57.76 GiB) is a **7,570,911-row PARTIAL
  PREFIX**. `scripts/build_dolmino_corpus_wzc1.py:21` says verbatim: *"Do NOT substitute wzc1's
  data/dolmino_now15b.npy: it is a 7,570,911-row PREFIX"*.
- The real source is **`data/dolmino_olmo2_shards/dolmino_chunks_2048_olmo2_shard{0000..0083}.npy`
  — 84 shards, measured `du -sb` = 126,940,820,276 B = 118.22 GiB (rc=0)**. The builder concats
  them to the 15,491,607-row corpus with md5 `7df19b217e5b0670d58bf6e01e6559d0`, and that md5
  equality is the ONLY thing making the .212 distill arm comparable to the zwfy6 NTP arms.
- zwfy6 has the assembled 118.19 GiB `data/dolmino_now15b.npy` directly.

**Therefore: deleting `data/dolmino_olmo2_shards/` on wzc1 is a one-way door.** /dev/shm does not
survive reboot. If a wzc1 node reboots and the 84 shards are gone, the keep10 and keep14-distill
arms cannot be resumed *on the comparable corpus* — you would have to re-tokenize dolmino from
raw, and any substitute array breaks the md5 assert and voids arm comparability. Total 118.22 GiB
protecting two multi-week runs.

### Single-disk-only paper provenance (measured; NO second copy exists)

| path | disk | measured | why veto |
|---|---|---|---|
| `outputs/olmo2_probe2_7B_keep14fresh2_seed1234/` (18 ckpt @48,724,474,298 B + final.pt) | wzc1 ONLY | 761,319,902 KiB ≈ 726.1 GiB | PaperB seed-variance arm (task #181). `step200000.pt` and `final.pt` share an inode (link count 2). |
| `outputs/olmo2_p05_armA_contig16/` (step0 16,241,491,563 + step50000 + step80000 @48,724,473,119) | wzc1 ONLY | 111,025,822 KiB ≈ 105.9 GiB | B-P0.4 gate arm; CLAUDE.md already records zwfy6 has only `arch_meta.json` + empty `step0.pt`. |
| `outputs/olmo2_p05_armB_final14_fresh2/` (same shape) | wzc1 ONLY | ≈ 105.9 GiB | ditto, the paired arm |
| `outputs/olmo2_keep14_densesave_reheal/step67500.pt` = 48,724,473,375 B + arch_meta.json 844 B | wzc1 ONLY (`ZWFY6_ABSENT` verified) | 47,582,496 KiB ≈ 45.4 GiB | B-P0.7 matched-PPL crossing MMLU (task #160); paperB evidence names step67500/70000/72500/27500 |
| `data/dolmino_olmo2_shards/` (84 shards) | wzc1 ONLY as a regeneration path | 118.22 GiB | see above |
| `/apdcephfs_wzc1/.../data/hf_datasets/TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet` | wzc1 ONLY (`ZWFY6_ABSENT`) | 4,144,185 B | source of paperC/paperG MMLU-Pro power-wall verdict (#251/#252). Note it is at `pighzliu_code/data/`, NOT `Mixture-of-Memory/data/` — repo-relative `data/hf_datasets` does NOT exist on either disk. |
| `outputs/qcmem_distill_qwen_j{6,9,18}_r32_4k/final`, `..._j12_r32_4k_seed{1,2}/final`, `outputs/qcmem_writepath_distill_qwen_j12_r32/`, `outputs/paperB_finding3_lr_control_randinit2e5/`, `outputs/olmo2_p13_scratch16_lr2e5_uniform/`, `outputs/paperC_qwen3base_heal_k8f2_pinned/`, `outputs/olmo2_probe2_7B_keep14_step0_pruned/step0.pt`, `outputs/olmo2_probe2_7B_keep12fresh2/step124000.pt`, `data/infinitebench/longbook_choice_eng.jsonl` | **zwfy6 ONLY** (all `WZC1_ABSENT` + `ZWFY6_PRESENT`) | not individually sized | paperA multi-depth curve (#122), paperB Finding-3 LR control (#149), paperC pinned arm. A wzc1-side lane that assumes "the papers' stuff is on wzc1" will read these as orphans on zwfy6. |

### Referenced by a paper but ABSENT on BOTH disks (searched both — do NOT "restore", just know)
- `outputs/comem_distill_8b_j12` — named only in `paperA/benchmark.md`, gone from both disks.
- `outputs/olmo2_probe2_7B_keep8fresh2/step45000.pt` — named in `paperC/evidence/heal_trajectory_mmlu_pro.json` and `heal_readout_v2_permutation_null.json`, **already rotated off both disks.** PaperC has a published trajectory number whose ckpt no longer exists. Not a cleanup issue, but it is a provenance hole another lane must not widen.
- repo-relative `data/closedbook`, `data/hf_datasets` — absent both disks (the real MMLU-Pro parquet is under `pighzliu_code/data/`).

### Duplicated across disks -> ONE copy is redundant, but NEITHER is safe to drop unilaterally
`outputs/olmo2_probe2_7B_{keep14fresh2,shortgpt16,full32_dolmino,keep14fresh2_freezefront,keep14fresh2_fromscratch}` step200000/step25000 are PRESENT on BOTH. These are the Table-4 headline rows (21/20/19/15/14 mentions). The cross-arch floor discipline in `eval_paperb_ladder_200k.sh` P4 means the *disk a number was measured on* is part of its protocol, so "dedupe to one disk" is a scientific decision, not a storage one. Flag for a human; do not auto-dedupe.

### Base models / eval assets that are veto by live-consumer, not by paper
wzc1 `models/OLMo-2-1124-7B` 29,204,228,800 B (27.20 GiB) — student init for LOCAL keep10 AND
loaded a 2nd time as the live distill teacher on .212. zwfy6 `models/OLMo-2-1124-7B` same size —
`--model_path` for .73+.82 AND the eval driver's P3 `[ -d ]` assert. zwfy6
`models/Qwen3-8B-Base` 16,393,040,530 B (15.27 GiB) — .104's `--model_path`.
Both disks' `data/dolmino_now_val.npy` md5 `f2ea48a2074a2f38fc3b4477fceecf11` — VERIFIED
identical on wzc1 and .73 by me. Driver P3 dies on any other md5.
Paper-cited model dirs still on wzc1: `Llama--Llama2-7b` 40,434,334,229 B, `Llama--Llama3-8b`
16,069,769,606 B, `Qwen--Qwen3-8b` 16,397,462,922 B (= the `models/Qwen3-8b-local` symlink
target, 104 paper mentions), `Qwen3-8B-Base` 16,393,044,618 B, `Qwen3-32B` 65,540,278,672 B,
`bge-large-en-v1.5` 1,341,564,829 B.

## READ-ONLY ATTESTATION
No rm / mv / truncate / chmod / redirect-into-repo was executed by this lane except appends to
this notes file. All remote commands were pgrep / ls / stat / du / md5sum / cat / df.
