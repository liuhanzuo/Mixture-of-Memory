# Within-disk floor v3 — the "floor" was a driver-version artefact; single-driver same-disk same-arch is DETERMINISTIC

**Date**: 2026-08-08 CST. **Feeds**: `status/PAPERB_WITHIN_DISK_FLOOR.md`. **GPU cost**: ~15-21 min per node, 4 nodes in parallel.

## TL;DR (headline verdict)

Under the **current downstream eval driver** (i.e. runs launched with `scripts/eval_olmo2_probe2_downstream.py` supporting `--save_per_example`), same-disk same-architecture core6 evaluations of a bit-identical OLMo-2 7B checkpoint are **bit-deterministic**: **0 per-item flips, Δcore6 = 0.0000 pp** across all four rungs measured (keep14 zwfy6/H20, keep8 zwfy6/H20, shortgpt16 zwfy6/H20, full32-base wzc1/L20A).

The previously reported "18-flip within-disk floor" on keep10 (see `PAPERB_WITHIN_DISK_FLOOR.md`) is now traceable to a **mixed-driver comparison**: the Table 4 source (`7B_keep10_step83500/`) was produced by an older driver revision that did NOT emit `per_example_*.jsonl`; the v2 comparison used the current driver. On shortgpt16 we can directly reproduce the same pattern (sg16 v1 vs v2 = +0.098 pp / ~20 summary-count diffs; sg16 v2 vs v3 = 0.0000 pp / 0 flips).

## What was run

Four batteries in parallel, one per node, all `chat_template=False`, `--add_bos 0`, `--save_per_example`, `assert_8shards` on every merge. Driver: `scripts/_run_olmo2_within_disk_floor_v3.sh <ARM>` (md5 `67206c8c618273120039a7434a5f9e49`, byte-identical mirror of `_run_olmo2_p24_eval_ladder_prev2_73.sh`).

| node | disk / arch | ARM (ckpt) | output_name | script PID | log | wall-clock |
|---|---|---|---|---:|---|---:|
| .73 | zwfy6 / H20 cc9.0 | keep14 `outputs/olmo2_probe2_7B_keep14fresh2/step200000.pt` | `7B_keep14_step200000_v2` | 3125387 | `zwfy6:logs/within_disk_floor_v3_keep14.log` | 17 min |
| .82 | zwfy6 / H20 cc9.0 | keep8  `outputs/olmo2_probe2_7B_keep8fresh2/step121000.pt`  | `7B_keep8_step121000_v3`  | 1267272 | `zwfy6:logs/within_disk_floor_v3_keep8.log`  | 14 min |
| .104 | zwfy6 / H20 cc9.0 | shortgpt16 `outputs/olmo2_probe2_7B_shortgpt16/step200000.pt` | `7B_shortgpt16_step200000_v3` | 3332220 | `zwfy6:logs/within_disk_floor_v3_shortgpt16.log` | 21 min |
| .252 | wzc1 / L20A cc10.0 | full32_base (HF `models/OLMo-2-1124-7B`, no `--ckpt`) | `7B_full32_base_wzc1_v2` | 3226907 | `wzc1:logs/within_disk_floor_v3_full32_base.log` | 15 min |

- `assert_8shards`: **5/5 harnesses × 4 arms** all confirmed 8/8 shards, no `SHARD MISSING` / `ABORT merge` anywhere (grep-verified in every log).
- Per-item predictions retained: `per_example_{task}.jsonl` present for all core6, know5, and MMLU-content harnesses.
- No prior `_v2` (keep14, full32_base) or `_v3` (keep8, shortgpt16) summary was overwritten.

## Flip-count table (headline)

Per-item comparison of `acc_norm_score` (or `acc_score`) across all core6 tasks. `n_flips` = number of items where the two runs disagree.

| rung / arch / disk | existing pair | n_flips | Δcore6_macro | new pair (this batch) | n_flips | Δcore6_macro |
|---|---|---:|---:|---|---:|---:|
| keep14 zwfy6 (H20) | n/a (was n=1) | – | – | v1 (`_step200000`) vs v2 (`_step200000_v2`) | **0** | **+0.0000 pp** |
| keep8 zwfy6 (H20)  | v1 does not have per-item preds | – | – | v2 (`_step121000_v2`) vs v3 (`_step121000_v3`) | **0** | **+0.0000 pp** |
| shortgpt16 zwfy6 (H20) | v1 vs v2 (**mixed driver**; v1 predates `--save_per_example`) | ~20 (summary-count Δ = +0.098 pp) | +0.0978 pp | v2 vs v3 (**same driver**) | **0** | **+0.0000 pp** |
| full32_base wzc1 (L20A) | n/a (was n=1) | – | – | v1 (`_wzc1`) vs v2 (`_wzc1_v2`) | **0** | **+0.0000 pp** |

- All four "new pair" comparisons produce **byte-identical** `n_correct`, `acc`, and `acc_norm` on every one of the six core6 tasks. Not "close within ~0.03 pp" — the floating-point summaries are equal to all printed digits.
- The only non-zero within-disk comparison is shortgpt16 v1 vs v2, which crosses a driver boundary.

## What this means for the earlier retraction

The earlier retraction in `PAPERB_WITHIN_DISK_FLOOR.md` was **directionally correct** (the cross-arch flip count of 17 for keep10 was not clearly above the "floor" of 18 for keep10) but attributed the wrong cause. The retraction hypothesis was "harness nondeterminism (unseeded reductions, tie-breaking, etc.)". The v3 data show that under a single driver revision the eval is **fully deterministic**. So the true source of the 17-18 flip parity was almost certainly:

1. **Driver revision drift** between when the Table 4 source rows were run (older driver, no per-example emission, possibly slightly different tokenisation, batch size, or seed) and when the v2/v3 controls were run (current driver).
2. The cross-arch delta was ALSO ~18 flips because it compared the OLD wzc1 driver's L20A run to the NEW zwfy6 driver's H20 run — a **cross-arch + cross-driver** combination that has been misattributed to hardware.

**Implications**:

- The **"~15-25 flip floor" from harness stochasticity does not exist** at the level MAIN feared. Under a single driver, core6 is reproducible to the last digit on the same hardware family.
- The `PAPERB_WITHIN_DISK_FLOOR.md` conclusion "**do not put damage-scaling in a paper on this evidence**" is unchanged and if anything reinforced — the confound has moved from "harness noise" to "driver drift", which is a *sharper* confound (it is a systematic bias between two eval revisions, not zero-mean noise) and cannot be averaged out by adding more repeats of a single revision.
- **What would settle the architecture question**: re-run the L20A wzc1 Table 4 sources with the **current** driver (i.e. `_wzc1_v2` on the ladder rungs, not just full32_base), then compute cross-arch flips between `_wzc1_v2` (L20A, current driver) and `_v2`/`_v3` (H20, current driver). All rungs are then same-driver on both sides, and any residual flip count is a genuine architecture effect. This is out of scope for the present task (which was floor variance only).
- **Paper B recommendation is unchanged**: core6 must not be reported to 4 decimals across mixed-driver rows in Table 4; the driver-version audit `PAPERB_TABLE4_ARCH_AUDIT.md` needs closing before Table 4 ships.

## Verdict

**Within-disk floor holds at n=0 flips across 4 rungs under a single driver revision.** The apparent ~15-20 flip floor in `PAPERB_WITHIN_DISK_FLOOR.md` was a mixed-driver artefact, not harness nondeterminism.

## Provenance

- Driver script: `scripts/_run_olmo2_within_disk_floor_v3.sh` (md5 `67206c8c618273120039a7434a5f9e49`, present on both wzc1 and zwfy6 disks with matching md5)
- wzc1 side (LOCAL/.252): `outputs/olmo2_downstream_results/7B_full32_base_wzc1_v2/{summary.json,per_example_*.jsonl,shard*of8.json}`
- zwfy6 side (.73/.82/.104): `outputs/olmo2_downstream_results/{7B_keep14_step200000_v2,7B_keep8_step121000_v3,7B_shortgpt16_step200000_v3}/{summary.json,per_example_*.jsonl,shard*of8.json}`
- Log grep for `SHARD MISSING|ABORT merge`: 0 hits in all four logs
- Compute cost: 4 × 8-GPU nodes × ~15-21 min each, no user checkpoints or artifacts touched
- Ledger: `status/gpu_runs.jsonl` (4 rows + 1 correction), `status/GPU_STATUS.md` (block appended)
- Related: `status/PAPERB_WITHIN_DISK_FLOOR.md` (MAIN-owned, untouched), `status/PAPERB_CORE6_CROSSARCH_FLOOR.md`, `status/PAPERB_DAMAGE_SCALING_AUDIT.md`, `status/PAPERB_TABLE4_ARCH_AUDIT.md`
