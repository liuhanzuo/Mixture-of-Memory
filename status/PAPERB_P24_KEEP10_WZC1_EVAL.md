# Paper B P2.4 / Task #189 (extension) — keep10 pre-SFT wzc1/L20A eval on .252

**Owner**: subagent dispatched by MAIN 2026-08-08 04:53 CST (this task).
**Purpose**: Complete the **5×2 damage-scaling grid** for Paper B Table 4.
Prior wzc1-side pre-SFT evals: **n=4** — full-32L, keep8, keep14+fresh2,
ShortGPT-16 (from `PAPERB_P24_LADDER_WZC1_EVAL.md`, `PAPERB_P24_FULL32_ARM.md`,
`PAPERB_P24_SHORTGPT16_ARM.md`, `PAPERB_P24_WZC1_EVAL.md`). **keep10** was the
only rung with **no wzc1-side eval** because its Table-4-headline `step83500.pt`
lived only on zwfy6. This subagent (a) scp-transferred that ckpt to wzc1 and
(b) ran the full L20A pre-SFT eval battery, finally letting us compute
`core6_wzc1(keep10) − core6_zwfy6_v2(keep10)` — an **arch-only** cross-arch
delta (not arch+step confounded like the wzc1 keep12 row).

## 1. Phase A — Cross-disk scp

**Recipe**: `scp -O` from `.73` (zwfy6) to LOCAL wzc1 (shared FS with `.252`),
initiated from LOCAL (had sshpass). Alternative "push from .73 to .252" was
not possible because neither `.73` nor `.252` has `sshpass` installed. Since
LOCAL and `.252` share the wzc1 physical disk, LOCAL was a valid entry point
for the wzc1 destination.

  * **Command**:
    ```
    sshpass -f configs/password_h20_853573.txt scp -O \
      root@28.85.35.73:/apdcephfs_zwfy6/.../outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt \
      /apdcephfs_wzc1/.../outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt
    ```
  * **Source md5** (`.73`): `8bf07fa0d08ddfdf66bd80fbc6721b33` (computed pre-transfer)
  * **Started**: 2026-08-08 04:57:XX CST
  * **Finished**: 2026-08-08 05:31:12 CST
  * **Duration**: **~34 minutes**
  * **Size**: 39,009,621,151 bytes = 36.3 GiB (task description said 48.7 GB; that was the on-disk allocation, real content is 36.3 GiB)
  * **Throughput**: **~18.4 MB/s** (better than the CLAUDE.md 12 MB/s worst case)
  * **Dest md5 (LOCAL wzc1)**: `8bf07fa0d08ddfdf66bd80fbc6721b33` ✅ MATCH
  * **Dest md5 (`.252` via ssh, absolute path)**: `8bf07fa0d08ddfdf66bd80fbc6721b33` ✅ MATCH
  * **wzc1 disk after**: 7.2 T used / 28 T total, **21 T free** (plenty)

**No retries, no truncation, no hash mismatch.** Transfer was clean end-to-end.

## 2. Phase B — Eval battery on .252 (L20A, wzc1)

  * **Node**: `.252` (`28.89.19.252`), 8×L20A cc10.0 wzc1
  * **Verified idle** at 05:37 CST (0/8 procs, 0 MiB used)
  * **Python**: `/opt/conda/envs/torch-base/bin/python`
  * **Driver**: `scripts/_run_olmo2_p24_eval_keep10_wzc1_252.sh`
    (committed as `b7d2d39`; byte-identical mirror of
    `_run_olmo2_p24_eval_ladder_wzc1_252.sh` except: 1 arm instead of 2,
    ckpt path = `outputs/olmo2_probe2_7B_keep10fresh2/step83500.pt`,
    output name = `7B_keep10_step83500_wzc1`)
  * **Launched**: 2026-08-08 **05:38:02** CST
    (`setsid nohup bash scripts/_run_olmo2_p24_eval_keep10_wzc1_252.sh > logs/p24_eval_keep10_wzc1_252.log 2>&1 &`)
  * **Parent bash PID**: `3170804`
  * **Log**: `.252:logs/p24_eval_keep10_wzc1_252.log`

### Harnesses (5 total; each 8-shard + merge with `assert_8shards` invariant)

| harness | status | landed | key numbers |
|---|---|---|---|
| PPL (dolmino_now_val) | ✅ done | 05:39:37 | **12.8158** (n_tokens=8.38M, n_windows=4096) |
| downstream core6 | ✅ done | 05:41:59 | see §3 below; per-item preds retained ✅ |
| downstream know5 | ✅ done | 05:44:35 | mmlu 27.13 / lambada 49.66 / boolq 62.87accn / cs_qa 43.00accn / social_iqa 43.86accn ✅ per-item |
| MMLU dual (letter+content) | ✅ done | 05:47:00 | letter 27.17 / content_raw 32.25 / content_norm **34.52** (content > letter, McNemar p=1.5e-40, Δ +7.35pp) |
| closedbook (PopQA, TriviaQA) | ✅ done | 05:51 | PopQA em 4.75/contains 13.21/f1 8.64 · TriviaQA em 18.15/contains 31.18/f1 26.25 |

**Total wall-clock**: 05:38:02 → 05:51:XX = **~13 minutes** (much faster than the 60-min ETA — 1-arm × 8-shard on 8×L20A crushed it).

### Shard-invariant compliance

Driver runs `assert_8shards <RROOT> <NAME> "shard{g}of8.json"` before every
merge. PPL and core6 merges both completed → both passed the 8/8 shard
assertion. If any subsequent harness has a missing shard, the merge will
abort loudly rather than silently contaminate. `--save_per_example` on the
downstream harness produced `per_example_*.jsonl` for all six core6 tasks
(hellaswag/arc_challenge/arc_easy/piqa/winogrande/openbookqa) — required
for downstream per-item McNemar / paired bootstrap vs the zwfy6 sibling.

## 3. Cross-arch delta (keep10, arch-only)

Baseline for delta: zwfy6 sibling `_v2` produced by `_run_olmo2_p24_eval_ladder_prev2_73.sh`
on `.73` (agent a50df6cd). **Same ckpt exactly** (both md5 =
`8bf07fa0d08ddfdf66bd80fbc6721b33`), so this is a **pure arch-only** delta
(unlike the keep12 wzc1 row which was step111500 vs zwfy6 step124000).

### core6 (acc_norm — canonical for Table 4)

| task | L20A accn | H20 accn | Δ pp | correct L20A − H20 (n) |
|---|---:|---:|---:|---:|
| hellaswag       | 0.5467 | 0.5467 | +0.000 |  +0  (n=10042) |
| arc_challenge   | 0.3669 | 0.3635 | +0.341 |  +4  (n=1172)  |
| arc_easy        | 0.6456 | 0.6481 | −0.253 |  −6  (n=2376)  |
| piqa            | 0.7269 | 0.7258 | +0.109 |  +2  (n=1838)  |
| winogrande      | 0.5509 | 0.5438 | +0.710 |  +9  (n=1267)  |
| openbookqa      | 0.3560 | 0.3520 | +0.400 |  +2  (n=500)   |
| **core6 avg**   | **0.5322** | **0.5300** | **+0.218** | — |

**Sum of |net flips|** (accn correct-count deltas) = **23** (Δc =
+0+4−6+2+9+2 = +11 signed; 4+6+2+9+2 = 23 absolute-sum ignoring zero rows).

### core6 (acc — raw match)

| task | L20A acc | H20 acc | Δ pp | net flips |
|---|---:|---:|---:|---:|
| hellaswag       | 0.4230 | 0.4213 | +0.169 | +17  |
| arc_challenge   | 0.3413 | 0.3404 | +0.085 |  +1  |
| arc_easy        | 0.6877 | 0.6877 | +0.000 |  +0  |
| piqa            | 0.7274 | 0.7285 | −0.109 |  −2  |
| winogrande      | 0.5509 | 0.5438 | +0.710 |  +9  |
| openbookqa      | 0.2840 | 0.2880 | −0.400 |  −2  |
| **core6 avg**   | **0.5024** | **0.5016** | **+0.076** | Σ\|Δc\| = 31 |

### PPL (held-out Dolmino)

  * **L20A**: 12.8158
  * **H20 (Table 4 headline)**: 12.816 (per task description) — matches to 4 sig figs.
    Full anchor from `_v2` sibling: sum_nll and n_tokens identical up to
    rounding → cross-arch PPL delta is **≤ 0.001** (well within FP tolerance).

## 4. Damage-scaling grid: n=4 → n=5 update

Combining with MAIN's prior n=4 audit
(`PAPERB_CORE6_CROSSARCH_FLOOR.md`, `PAPERB_DAMAGE_SCALING_AUDIT.md`):

| rung | pruning depth | pre-SFT PPL | core6 Δ (accn, pp) | Σ\|net flips\| (acc) |
|---|---:|---:|---:|---:|
| full-32L intact     |  0 layers dropped |  7.398 | +0.034 | 10 |
| ShortGPT-16         | 16 layers dropped |  9.780 | +0.045 | 13 |
| keep14+fresh2       | 18 layers dropped | 10.561 | +0.156 | 28 |
| **keep10+fresh2 (new)** | **22 layers dropped** | **12.816** | **+0.218** | **31** |
| keep8+fresh2        | 24 layers dropped | 13.333 | (pending re-derive on .252 fresh output — see MAIN's audit) | (same) |

**Story update**: keep10 fits the monotone pattern **weakly but not broken**:
  * PPL 12.816 sits between keep14 (10.561) and keep8 (13.333) — monotone.
  * core6 Δ (accn) = +0.218 pp is the **largest of the 4 measured rungs**,
    consistent with "more damage → larger arch-sensitivity".
  * Σ|net flips (acc)| = 31 is also the largest — again monotone.
  * BUT: the effect is small (all 4 measured rungs are within a ~0.2 pp band
    and 10–31 flips), so the "flip count scales with damage" claim remains
    **weak-monotone at best**. The full row confirms MAIN's audit conclusion
    that the effect is real but modest.

**Load-bearing tripwire (per task text): the eval did NOT break the n=4
pattern.** The delta is neither negative (would have overturned "L20A
slightly higher on pruned rungs") nor |flips| > 50 (would have escalated the
arch-sensitivity story). It is a **quiet confirmation** row.

## 5. Remaining outputs (ETA ~40 min from launch = ~06:20 CST)

  * `.252:olmo2_downstream_results/7B_keep10_step83500_wzc1_know/summary.json` (know5) ✅ landed 05:44:35
  * `.252:olmo2_mmlu_content_results/7B_keep10_step83500_wzc1/summary.json` (MMLU dual) ✅ landed 05:47:00
  * `.252:olmo2_closedbook_results/7B_keep10_step83500_wzc1/summary.json` (PopQA+TriviaQA) ✅ landed 05:51:XX

**ALL 5 HARNESSES DONE** — actual total wall-clock 13 min, not 60. Per-item
predictions retained on all downstream + MMLU-content harnesses. The core6
delta above is the load-bearing headline; the remaining harnesses populate
the full 5×2 audit table.

### know5 numbers (7B_keep10_step83500_wzc1)

  * mmlu: acc 27.13 / accn 27.13 (n=14042)
  * lambada_openai: acc 49.66 (n=5153)
  * boolq: acc 60.86 / accn 62.87 (n=3270)
  * commonsense_qa: acc 45.37 / accn 43.00 (n=1221)
  * social_iqa: acc 41.40 / accn 43.86 (n=1954)

### MMLU dual (letter + content) (7B_keep10_step83500_wzc1)

  * letter_acc: 27.17 (+2.17pp above chance 25.00)
  * content_raw_acc: 32.25 (+7.25pp)
  * content_norm_acc: **34.52** (+9.52pp above chance)
  * both_correct: 1326 · letter_only: 2489 · content_only: 3521 · neither: 6706
  * agreement: 57.20% · **McNemar exact p = 1.5e-40** (content NORM strictly > letter)
  * Bootstrap Δ (content_norm − letter): **+7.35 pp** (95% CI [+6.29, +8.45], n_boot=10000)

### Closedbook (7B_keep10_step83500_wzc1)

  * PopQA (n=14267): em 4.75 / contains 13.21 / f1 8.64 (majority_em baseline 2.29 → +2.46pp lift on em)
  * TriviaQA (n=17944): em 18.15 / contains 31.18 / f1 26.25 (majority_em baseline 0.26 → substantial lift)

## 6. Ledger + provenance

  * Driver commit: `b7d2d39` (this repo, main branch, LOCAL wzc1)
  * `status/gpu_runs.jsonl`: 2 entries appended
    - scp phase (04:57–05:31, md5 match)
    - eval phase (05:38 started, running)
  * ckpt provenance: zwfy6 origin (Aug 2 10:46), wzc1 landing (Aug 8 05:31),
    both md5 `8bf07fa0d08ddfdf66bd80fbc6721b33` (verified LOCAL and `.252`)
  * per-example files (core6, 6 tasks × ~10042 max rows): retained
