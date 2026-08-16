# Disk survey — LANE: OUR CHECKPOINTS (outputs/ on both disks)

Surveyor run 2026-08-16. READ-ONLY. No rm/mv/truncate executed.

Disks:
- wzc1 = /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory  (120T, 110T used, 10T avail, 92%)
- zwfy6 = /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory (689T, 667T used, 22T avail, 97%)

FIVE LIVE RUNS (untouchable):
- LOCAL/wzc1  outputs/olmo2_probe2_7B_keep10fresh2          ~178600/200000
- .212/wzc1   outputs/olmo2_probe2_7B_keep14fresh2_distill  ~38560/200000
- .73/zwfy6   outputs/olmo2_probe2_7B_keep12fresh2          ~189140/200000
- .82/zwfy6   outputs/olmo2_probe2_7B_keep8fresh2           ~162260/200000
- .104/zwfy6  outputs/paperC_qwen3base_heal_k8f2            ~62760/200000

## Progress log
- [t0] `du -h --max-depth=1 --block-size=1 outputs/` on wzc1 -> rc=0, 105 entries. Fast (<1 min).
- [t1] wzc1 `outputs/` TOTAL = **5,348,681,873,408 B = 4,981.4 GiB = 4.865 TiB** (measured, du apparent-ish/`--block-size=1`)
- [t2] zwfy6 `outputs/` TOTAL = **8,512,646,721,024 B = 7,927.0 GiB = 7.742 TiB** (measured via .73, rc=0, 1036 entries)

## wzc1 outputs/ — top dirs (bytes, measured)
| bytes | GiB | dir |
|---|---|---|
| 730867110400 | 680.7 | olmo2_probe2_7B_keep14fresh2_seed1234 |
| 585144337920 | 545.0 | olmo2_probe2_7B_keep10fresh2  **LIVE (LOCAL)** |
| 437919411200 | 407.8 | olmo2_probe2_7B_full32_dolmino |
| 406037278720 | 378.2 | olmo2_probe2_7B_shortgpt16 |
| 285436736000 | 265.8 | hyv3_probe2_keep36_fresh2 |
| 244897350656 | 228.1 | olmo2_probe2_7B_keep14fresh2_distill  **LIVE (.212)** |
| 219034802176 | 204.0 | paperC_depthsweep_keep24_scratch_refusal25 |
| 203870742016 | 189.9 | paperC_depthsweep_keep28_scratch_refusal25 |
| 175167764992 | 163.1 | olmo2_p24_sft_full32 |
| 165452903936 | 154.1 | paperC_pc1_squad_A3_keep28fresh2 |
| 165452903424 | 154.1 | paperC_depthsweep_keep28_rtrunk_refusal25 |
| 146023200768 | 136.0 | paperC_depthsweep_keep24_rtrunk_refusal25 |
| 146023199232 | 136.0 | paperC_pc1_squad_A3_keep24fresh2 |
| 126593496576 | 117.9 | paperC_depthsweep_keep20_rtrunk_refusal25 |
| 126593495552 | 117.9 | paperC_depthsweep_keep20_scratch_refusal25 |
| 126593495552 | 117.9 | paperC_pc1_squad_A3_keep20fresh2 |
| 113690440704 | 105.9 | olmo2_p05_armA_contig16 |
| 113690440704 | 105.9 | olmo2_p05_armB_final14_fresh2 |
| 97448947200 | 90.8 | olmo2_p24_sft_shortgpt16 |
| 97448942592 | 90.8 | paperC_depthsweep_keep14_rtrunk_refusal25 |
| 97448941568 | 90.8 | paperC_depthsweep_keep14_scratch_refusal25 |
| 48724475392 | 45.4 | olmo2_keep14_densesave_reheal |
| 48724469760 | 45.4 | olmo2_probe2_7B_keep14fresh2 |
| 48724469248 | 45.4 | olmo2_probe2_7B_keep14fresh2_fromscratch |
| 47351177728 | 44.1 | qwen3_minarch_armB_f12k2_200k |
| 43867049472 | 40.9 | olmo2_probe2_7B_keep12fresh2 (stale wzc1 copy; LIVE one is on zwfy6) |
| 34152197632 | 31.8 | olmo2_probe2_7B_keep8fresh2 (stale wzc1 copy; LIVE one is on zwfy6) |
| 26056481280 | 24.3 | olmo2_probe2_7B_keep14fresh2_freezefront |

## zwfy6 outputs/ — top dirs (bytes, measured)
| bytes | GiB | dir |
|---|---|---|
| 526404594176 | 490.3 | olmo2_probe2_7B_keep12fresh2  **LIVE (.73)** |
| 455362618880 | 424.1 | olmo2_probe2_7B_keep8fresh2  **LIVE (.82)** |
| 418986637824 | 390.2 | paperC_qwen3base_heal_k8f2  **LIVE (.104)** |
| 304717551104 | 283.8 | paperC_qwen3base_heal_k8f2_pinned |
| 243622366720 | 226.9 | olmo2_p13_scratch16_lr2e5_uniform |
| 234057736192 | 218.1 | olmo2_probe2_7B_keep10fresh2 (zwfy6 copy; LIVE one is wzc1/LOCAL) |
| 141933737472 | 132.2 | mem_space_fifo_b25_c512_supervised_select |
| 141915100160 | 132.2 | mem_space_fifo_b25_chunk512 |
| 141915034624 | 132.2 | mem_space_fifo_b25_chunk512_noleak |
| 141912243200 | 132.2 | mem_space_fifo_b25_chunk512_t2align |
| 141910860288 | 132.2 | mem_space_fifo_b100_chunk512 |
| 113690434560 | 105.9 | olmo2_probe2_7B_keep14fresh2 |
| 105204217344 | 98.0 | olmo2_probe2_7B_shortgpt16 |
| 97448947200 | 90.8 | olmo2_p24_sft_keep14fresh2 |
| 87734095872 | 81.7 | olmo2_p24_sft_keep12fresh2 |
| 87583882752 | 81.6 | olmo2_probe2_7B_full32_dolmino |
| 85179897344 | 79.3 | sparse_memory_ablation_8gpu_v5 |
| 82256066560 | 76.6 | progressive_chunk_diskB_stable |
| 78019243520 | 72.7 | olmo2_p24_sft_keep10fresh2 |
| 74780946432 | 69.6 | paperC_depthsweep_keep28_graft_refusal25 |
| 68304391680 | 63.6 | olmo2_p24_sft_keep8fresh2 |
| 67592882176 | 63.0 | sparse_memory_concat_fusion_v1_fixed |
| 65570139136 | 61.1 | dms_8x |
| 61827813376 | 57.6 | paperC_depthsweep_keep20_graft_refusal25 |
| 61756720640 | 57.5 | progressive_chunk_diskB_v3_improved |

---

## THE HEADLINE FINDING — `outputs/olmo2_probe2_7B_keep14fresh2_seed1234` tail

Run is **COMPLETE** (`step200000.pt` written 2026-08-12, `final.pt` hardlinked to it, nlink=2 —
the 2026-08-12 audit already deduped that pair, so du counts it once).

Measured `stat -c '%i %h %s %n' *.pt`: **16 filenames, 15 unique inodes**, dir total
730,867,110,400 B = 680.7 GiB.

| file | bytes | GiB | nlink | referenced anywhere? |
|---|---|---|---|---|
| step25000.pt | 48724473567 | 45.38 | 1 | YES — B04 `EVAL_FILL_READY_20260816.md` rung |
| step50000.pt | 48724473567 | 45.38 | 1 | YES — B04 rung |
| step100000.pt | 48724474298 | 45.38 | 1 | YES — B04 rung |
| step128000.pt | 48724474298 | 45.38 | 1 | YES — B04 rung + paperB apex pairing |
| step153500.pt | 48724474298 | 45.38 | 1 | YES — B04 rung |
| step165000.pt | 48724474298 | 45.38 | 1 | **NO** (the one `step165000` hit is `keep12fresh2_step165000`, a different run) |
| step170000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step175000.pt | 48724474298 | 45.38 | 1 | YES — B04 rung |
| step180000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step185000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step190000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step195000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step199000.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step199500.pt | 48724474298 | 45.38 | 1 | **NO** (0 files) |
| step200000.pt == final.pt | 48724468275 | 45.38 | 2 | YES — endpoint, paperB `SEEDVAR_KEEP14_PROTOCOL.md:21` |

**8 unreferenced tail checkpoints × 48,724,474,298 B = 389,795,794,384 B = 363.1 GiB.**
(step165000, 170000, 180000, 185000, 190000, 195000, 199000, 199500.)

⚠️ **DO NOT touch step{25000,50000,100000,128000,153500,175000}** — `proposal/backlog/B04-eval-fragility-incubator/EVAL_FILL_READY_20260816.md`
(written **today**) says "the 6 missing read-out arms are ready to evaluate in one command" and
names exactly those six paths with byte-size assertions (`size 48724474298 == expected`). Deleting
any of them breaks a launch-ready gate. That is 6 × 45.38 = 272.3 GiB that is NOT free.

The 8 tail ckpts are the dense every-5000 tail the trainer wrote near the end. They are a
`ckpt_rotation.py` clause-5 artefact (unbounded milestone clause — the very failure mode that
module's docstring says produced "multi-TB output dirs ... forced a manual 9.4 TiB cleanup").
No paper table, no `STATUS.json`, no script names them.

### Live-writer check for the seed1234 tail (safety check #1)
- `pgrep -af "train_olmo2|train_qwen3|main_llama|train_qcmem"` on LOCAL returns **only**
  `train_olmo2_arch_probe2.py ... --output_dir .../outputs/olmo2_probe2_7B_keep10fresh2` (11 ranks).
  Its `--keep_steps 83500,90000,121000,124000,150000,175000,200000` and `--output_dir` are
  keep10fresh2 — it cannot write to seed1234.
- Newest `.pt` mtime in seed1234 is **2026-08-12**; nothing has been written for 4 days.
- `fuser` is **not installed** on LOCAL (`rc=127`), so the open-fd check could not be run — the
  writer check above is process/argv-based, which is why I quote the `--output_dir` verbatim.

---

## SECONDARY WIN — `outputs/olmo2_probe2_7B_shortgpt16` (wzc1, 378.2 GiB total)

Complete run (`step200000.pt` == `final.pt`, nlink=2, already deduped 08-12).
`step0.pt` is 16,241,491,627 B (weights-only, protected by `ckpt_rotation.py` clause 4 — it is a
paper-table row AND the recovery-fraction denominator, and is *irreproducible* once the init seed
is lost). Referenced steps: 200000 (84 files), 128000 (13), 153500 (2), 25000 (1), 5000 (1).

**Unreferenced: step50000.pt, step100000.pt, step125000.pt (0 exact-path hits each)
= 48,724,473,247 + 2 × 48,724,473,978 = 146,173,421,203 B = 136.1 GiB.**

Lower confidence than the seed1234 tail: shortgpt16 is the published ShortGPT baseline for Paper B
and 128000/153500 are already-published trajectory points, so a future reviewer asking for a
denser trajectory would want 50k/100k/125k back. Classify as **yes_but_regenerable_at_cost**.

---

## NOT deletable — explicitly checked and cleared as KEEP

| path | GiB | why not |
|---|---|---|
| `outputs/olmo2_probe2_7B_keep10fresh2` (wzc1) | 545.0 | **LIVE** — LOCAL trainer's `--output_dir`, verified in argv |
| `outputs/olmo2_probe2_7B_keep14fresh2_distill` (wzc1) | 228.1 | **LIVE** — .212 |
| `outputs/olmo2_probe2_7B_keep12fresh2` (zwfy6) | 490.3 | **LIVE** — .73 |
| `outputs/olmo2_probe2_7B_keep8fresh2` (zwfy6) | 424.1 | **LIVE** — .82 |
| `outputs/paperC_qwen3base_heal_k8f2` (zwfy6) | 390.2 | **LIVE** — .104 |
| seed1234 step{25000,50000,100000,128000,153500,175000} | 272.3 | B04 `EVAL_FILL_READY_20260816.md` launch-ready rungs, named with byte assertions |
| `outputs/olmo2_probe2_7B_full32_dolmino` (wzc1) | 407.8 | **all 5 steps cited**: 25000 (75 files), 20000 (16), 5000 (9), 15000 (5), 10000 (2) — `a04_full32_trajectory_ni.json` is a *published trajectory* over 10000/15000/20000/25000 |
| `outputs/hyv3_probe2_keep36_fresh2/step200.pt` (wzc1) | 265.8 | Sole survivor a prior cleanup deliberately kept (`DISK_CLEANUP_AUDIT_20260812.md` §6.5, §8.3). Overriding needs a decision, not an inference. |
| `outputs/paperC_qwen3base_heal_k8f2_pinned` (zwfy6) | 283.8 | 8 ckpts; **step{5000,5500,6000,6500,7000} cited** by `paperC/HEAL_TRAJECTORY_READOUT_1.md` + `evidence/heal_trajectory_mmlu_pro.json`. step{7500,8000,8500} not cited = ~106 GiB *possible*, but this is a paperC round_04 submission artifact under active review → NOT recommended. |
| `outputs/paperC_depthsweep_keep24_scratch_refusal25` (wzc1) | 204.0 | 3 ckpts × 68 GiB; `final.pt`/`step1000.pt` are NOT hardlinked here (the 08-12 audit's zip oracle **rejected** this pair as genuinely different) |
| `outputs/olmo2_p13_scratch16_lr2e5_uniform` (zwfy6) | 226.9 | Finding-3 LR control, cited by `paperB/FINDING3_LR_CONTROL_DESIGN.md` |
| `outputs/olmo2_probe2_7B_keep10fresh2` (zwfy6 copy, 218.1) | 218.1 | step{89000,89500} each cited by **89 files**; this is the pre-migration resume lineage of the LIVE wzc1 run |

## Cross-disk duplicate note (NOT a safe win)
`keep12fresh2` / `keep8fresh2` / `keep10fresh2` / `keep14fresh2` / `shortgpt16` / `full32_dolmino`
exist on BOTH disks with **different step sets** (verified `stat` on both sides). e.g. wzc1
`keep14fresh2` has step200000 at 48.7 GB while zwfy6 `keep14fresh2/step200000.pt` is 16.2 GB
(weights-only). These are **not redundant copies** — the two disks hold different slices of the
same lineage and some are the resume ancestors of the live runs. Do not treat "same dirname on both
disks" as duplication.

---

## VERDICT — single largest safe win

**`outputs/olmo2_probe2_7B_keep14fresh2_seed1234/step{165000,170000,180000,185000,190000,195000,199000,199500}.pt`
on wzc1 = 389,795,794,384 B = 363.0 GiB (0.354 TiB).**

All eight: nlink=1 (no hardlink alias), zero references in `paper{A,B,C}/`, `proposal/*/`,
`scripts/`, `status/`; run complete since 08-12; no live writer; the endpoint (step200000==final.pt)
and all six B04 rungs survive untouched.

Adding the shortgpt16 mid-trajectory trio gives **535,969,215,587 B = 499.2 GiB** at mixed
confidence.

Neither is executed here. This is a survey.
