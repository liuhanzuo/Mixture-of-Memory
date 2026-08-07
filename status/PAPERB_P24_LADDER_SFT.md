# Paper B P2.4 — n=6 damage-sensitivity ladder SFT dispatch

**Status: `[RUNNING — 3 arms SFT on zwfy6 H20 nodes; observed-Δ pending completion]`**
**Date**: 2026-08-08 04:35 +08:00 | **Nodes**: `.73` / `.82` / `.104` (all zwfy6 disk) | **Commit at launch**: `0ffac52`
**Author**: experiment agent responding to MAIN P2.4 ladder dispatch.

## 0. What this extends

The three completed P2.4 arms give a suspiciously clean Pearson r=0.998 (n=3) between the
pre-SFT PPL of a healed pruned base and the Δ PPL% induced by applying the same 842-step Tulu-3
general SFT recipe. This dispatch extends the ladder to **n=6** so the linear fit can be
falsified (or, if it holds, reinforced) with three testable predictions.

**Completed 3-arm fit** (from `status/PAPERB_P24_SFT_REPAIRABILITY.md`):

| arm | pre-SFT PPL | Δ PPL% |
|---|---:|---:|
| full32 (intact 32L) | 7.398 | +4.46% |
| shortgpt16 (16L pruned) | 9.780 | +8.51% |
| keep14+fresh2 (16L pruned) | 10.561 | +9.43% |

## 1. Predicted-Δ table (linear extrapolation on n=3)

The n=3 fit is `Δ% ≈ 1.51 × PPL - 6.72` (least squares). Extrapolating to the three lower rungs:

| arm | pre-SFT PPL | **predicted Δ PPL%** | observed Δ PPL% | notes |
|---|---:|---:|---:|---|
| keep8+fresh2 (10L) | 13.333 | **≈ +14.0%** | *pending* | widest extrapolation; largest lever for falsification |
| keep10+fresh2 (12L) | 12.816 | **≈ +13.2%** | *pending* | mid-rung |
| keep12+fresh2 (14L) | 11.443 | **≈ +11.0%** | *pending* | closest to fitted range |

**Falsification band**: with n=3 the 95% residual band is wide. Anything within 10–18% for keep8,
9–17% for keep10, 7–15% for keep12 is consistent with the linear story. Departures >5 pp from
prediction — either direction — are the informative outcome and MUST be reported loudly. Two
qualitative failure modes to watch for:

  1. **Saturation**: Δ plateaus at some ceiling (~15%) for deep prunes → the linear model breaks
     because the model can't absorb even more SFT damage on top of pre-existing damage.
  2. **Coincidence**: the three lower rungs land far from the line → n=3 fit was chance; the
     paper cannot use it.

## 2. Preflight (all passed before launch)

| check | result |
|---|---|
| trainer md5 identical on all 3 nodes | `02d8b9ead6cafdf5893d6e59df6ad196` ✅ |
| trainer bnb imports | `grep -c bitsandbytes = 0` ✅ (`.82` safe) |
| SFT data md5 input_ids | `b1e6fe4e11351e208da24b03d96a762a` ✅ all 3 nodes match spec |
| SFT data md5 labels | `bf7c57746f05b1ac73ccdaa07b1481b7` ✅ all 3 nodes match spec |
| ckpt keep8 step121000 | 11.4 GB present on `.73` ✅ |
| ckpt keep10 step83500 | 39.0 GB present on `.82` ✅ |
| ckpt keep12 step124000 | 43.9 GB present on `.104` ✅ (Table 4 headline; NOT `step111500`) |
| pre-launch GPU state | 0% util, 0 MiB on all 3 nodes ✅ |
| driver md5 identical on all 3 nodes | `62a640b1b3f344226e33d1b98b720e0b` ✅ |

## 3. Launch record (byte-identical to keep14 / shortgpt16 arms)

Diff-check against the completed `shortgpt16` arm from `logs/p24_sft_shortgpt16.log`
(cfg line: `arm=shortgpt16 mode=sft world=8 BS=1 GA=16 eff_batch=128 seq_len=2048 max_steps=842 -> token_budget=220,725,248`)
— the only differences in our launch cmdlines are (`--ckpt`, `--arm_name`, `--output_dir`), as intended.

| arm | node | PID (torchrun) | log | started (CST) | init step-20 loss | s/step init |
|---|---|---:|---|---|---:|---:|
| keep8 | `.73` (28.85.35.73) | 2997933 | `logs/p24_sft_keep8.log` | 04:27:01 | **2.0422** ✅ finite | 6.44 |
| keep10 | `.82` (28.82.250.82) | 1129375 | `logs/p24_sft_keep10.log` | 04:30:59 | **1.9671** ✅ finite | 7.51 |
| keep12 | `.104` (28.83.24.104) | 3187927 | `logs/p24_sft_keep12.log` | 04:31:29 | **1.8399** ✅ finite | 8.66 |

All three:
- `torch.distributed.run --standalone --nproc_per_node 8`
- `--seq_len 2048 --batch_size 1 --grad_accumulation_steps 16 --max_steps 842`
- `--lr 1e-5 --min_lr 1e-6 --warmup_steps 100 --weight_decay 0.1 --grad_clip 1.0`
- `--gradient_checkpointing 1 --save_every 500 --seed 42`
- `NCCL_IB_DISABLE=1`, `WANDB_MODE=offline`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

Loss trajectory (initial): all three finite and in a plausible range for pruned OLMo-2 (shortgpt16
was 1.67 at step 20 and dropped to 1.0-1.2; keep14 similar). No NaN. `0fd051a` NaN-fix is in effect.

## 4. Memory / feasibility on H20 (97.8 GiB/card)

| arm | layers | measured per-card mem @ steady | headroom |
|---|---:|---:|---:|
| keep8 | 10 | 68.3 GiB | 29.5 GiB — comfortable |
| keep10 | 12 | 77.5 GiB | 20.3 GiB — comfortable |
| keep12 | 14 | 86.9 GiB | 10.9 GiB — tight but below 90 GiB threshold |

All three stayed under the 90 GiB WARNING threshold at steady state through step 20. keep12
(14L) is the closest to the ceiling. This is consistent with the observed keep14 (16L) SFT which
used ~96 GiB/card and completed. **No memory action required.**

## 5. ETA & next action

Per-arm SFT: 842 steps × ~2.15 s/step steady (after ckpt warmup + first cosine ramp) ≈ **30-35
min**, plus ~1 min ckpt load + ~2 min final save. Total per arm ~35-40 min.

- keep8 (`.73`): finish ~**05:05 CST**
- keep10 (`.82`): finish ~**05:15 CST** (init s/step higher, but steady-state should match)
- keep12 (`.104`): finish ~**05:20 CST**

**No eval is queued in this dispatch.** Per MAIN's brief, the paired pre/post-SFT eval battery
(PPL + core6 + know5 + MMLU dual + closedbook) is a separate MAIN decision after SFT completion
+ headline PPL delta check. Each eval battery is ~90 min on H20.

## 6. Compute-confound caveat (still applies)

Per `status/PAPERB_P24_LADDER_WZC1_EVAL.md` and `PAPERB_TABLE4_BUDGET_DEFECT.md`, the three lower
rungs never reached 200k steps of healing:

- keep8 pre-SFT anchor = `step121000` (60.5% of nominal budget)
- keep10 pre-SFT anchor = `step83500` (**41.75%** of nominal budget)
- keep12 pre-SFT anchor = `step124000` (62% of nominal budget)

This is fine for a **damage-sensitivity experiment** (we hold each arm's pre-SFT ckpt fixed and
measure only the SFT-induced Δ), but the Δ PPL% numbers **must not be sold as depth-only** or as
"compute-matched depth ladder". The paper text needs to say: "we measure repairability at each
rung's Table 4 headline checkpoint", not "we measure depth vs SFT interaction at equal healing
compute".

## 7. Files

- Driver: `scripts/_run_olmo2_p24_sft_ladder_zwfy6.sh` (wzc1 canonical; scp'd byte-identical to
  all 3 zwfy6 nodes; md5 `62a640b1b3f344226e33d1b98b720e0b`).
- gpu_runs.jsonl: 3 new entries appended at 04:35 CST 2026-08-08 (commit `0ffac52`).
- GPU_STATUS.md: 3 new `▶️` blocks for `.73` / `.82` / `.104` at 04:35 CST.
- Report (this file): `status/PAPERB_P24_LADDER_SFT.md`.

## 8. Do NOT touch (per dispatch spec)

`paper*/*.tex`, `versions/*.md`, `paperB/TODOList.md`, sibling `PAPERB_P24_*.md` files
(`_ARM.md`, `_KEEP14_EVAL.md`, `_LADDER_PREV2_EVAL.md`, `_WZC1_EVAL.md`, `_LADDER_WZC1_EVAL.md`,
`_SFT_REPAIRABILITY.md`), `PAPERB_CORE6_CROSSARCH_FLOOR.md`, `PAPERB_TABLE4_BUDGET_DEFECT.md` —
verified untouched.
