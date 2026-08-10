# lane2_a01_gate2 — A01 gate-2 ("a non-MMLU MC benchmark") readiness scout

STATUS: COMPLETE. Verdict = **NOT_A_GPU_JOB**. Confidence **high**.

## 0. Headline

Gate-2 needs **zero GPU work**. Nine non-MMLU multiple-choice benchmarks are already
scored **per item** on **wzc1** (the disk `.21` shares with LOCAL), across **6 item-aligned
arms**, with all shard-integrity and cardinality asserts passing. I ran the gate-2
statistic end-to-end during this scout on LOCAL CPU in ~6 minutes. **Do not spend an
8-GPU launch on this.** `.21` and `.82` should be filled with something else.

A stronger-than-expected result fell out: **BoolQ's construct-appropriate null is 0.6217
(always-B), not 0.5**, and under that null 3 of 6 arms sit at-or-below the floor on the
raw-LL interface. That is a second self-falsification case with the same shape as the MMLU
one, on a benchmark nobody would suspect (a "2-way" task whose real floor is 62%).

## 1. The gate spec (quoted)

`proposal/active/A01-null-calibration-methodology/PROPOSAL.md` L71-90:

> ## 下一步 gate
> ### 必做
> 1. 第三个模型家族的 MC interface case。
> 2. **非 MMLU 的一个 MC benchmark。**
> 3. OLMo full-fp32 forward：检验 bf16 exact tie 是否为因果机制。
> 4. C4 aggregation 预注册，不再选择性报告 10×。
>
> ### 成功条件
> - 至少三个 construct 的 null calibration 改变科学结论，而非仅缩小数字；
> - **第三模型/第二 benchmark 保持"instrument validity before comparison"结论**；
>
> ### Kill 条件
> - 除 representation 外，其他 construct 的结论在严格 null 下都不改变；
> - **第三家族和第二 benchmark 均不复现 interface failure**；

Also `STATUS.json` `next_gate[1] = "second multiple-choice benchmark"`.

**Kill semantics — important nuance.** The kill condition is a **conjunction**: it fires
only if *both* the third family *and* the second benchmark fail ("均不复现"). So gate-2
**alone cannot kill A01** — it is one of two required legs of a single AND-gate. Failure of
gate-2 alone would trim the generality claim ("MC interface fragility is an MMLU artifact")
from "MC content scoring in general" down to "MMLU specifically". Combined failure with
gate-1 would kill. Conversely, gate-2 *passing* does not by itself save A01 either.

**What I actually determined:** gate-2 does not fail. It reproduces the interface/floor
failure on non-MMLU MC, so the AND-gate cannot fire from this side.

## 2. Which non-MMLU MC benchmarks are scored per-item, and on which disk

### 2a. Disk visibility (the load-bearing point for this lane)

* `/apdcephfs_zwfy6` is **NOT mounted on LOCAL** (`ls: cannot access '/apdcephfs_zwfy6/'`).
  Only `dop-fuse 120T ... /apdcephfs_wzc1/share_304376610` is mounted. zwfy6 facts below
  were gathered via ssh to `.82`.
* **`.21` can read the wzc1 per-item files directly** — verified by running `wc -l` on `.21`:
  `.../7B_full32_base_wzc1_v2/per_example_openbookqa.jsonl` = 500,
  `.../7B_full32_base_wzc1_v2_know/per_example_boolq.jsonl` = 3270. No transfer needed.

### 2b. wzc1 — `olmo2_downstream_results/` (11 arms x core6, exact cardinalities)

Root: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results/`

All eleven of these have **54 per_example files** (6 tasks x (1 merged + 8 shards)) and
totals exactly 17,195:

| arm dir | hellaswag | arc_ch | arc_easy | piqa | obqa | winog | total |
|---|---|---|---|---|---|---|---|
| `7B_full32_base_wzc1` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_full32_base_wzc1_v2` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_keep14_step200000_perex` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_keep14_step200000_wzc1_v2` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_keep12_step111500_wzc1` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_keep10_step83500_wzc1` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_keep8_step121000_wzc1` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_shortgpt16_step200000_wzc1` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_fromscratch_step200000_perex` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_p24_sft_full32_final` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |
| `7B_p24_sft_shortgpt16_final` | 10042 | 1172 | 2376 | 1838 | 500 | 1267 | 17195 |

30 other dirs under the same root have **0** per_example files (summary-only, older runs):
`7B_keep14_step200000`, `7B_keep8_step{100000,110000,121000}`, `7B_shortgpt_step{0,200000}`,
`7B_scratch16L_step200000`, `7B_freezefront_step200000`, `7B_full32_step25000`,
`7B_keep14_reheal_step67500`, and their `_know` twins. **Do not build gate-2 on those.**

### 2c. wzc1 — the `_know` dirs are the bigger find (3 MORE non-MMLU MC benchmarks)

The `*_know` sibling dirs (45 files each) carry **BoolQ (2-way), CommonsenseQA (5-way),
SocialIQA (3-way)** plus lambada + MMLU. All 10 `_know` arms complete:

| task | n | n_options | present in |
|---|---|---|---|
| `boolq` | 3270 | 2 | all 10 `_know` arms |
| `commonsense_qa` | 1221 | **5** | all 10 `_know` arms |
| `social_iqa` | 1954 | **3** | all 10 `_know` arms |
| `lambada_openai` | 5153 | (not MC) | all 10 |
| `mmlu` | 14042 | 4 | all 10 |

**CommonsenseQA is a 5-option MC benchmark and the harness scores it fine** — so the
lane-brief worry "a benchmark with >4 options the harness cannot score" does **not** apply.
`option_scores` keys go `A..E`. This answers gate-2 with option-count diversity
(2 / 3 / 4 / 5-way) rather than only 4-way clones of MMLU.

### 2d. zwfy6 — 6 `*_bs16` arms, and they ALREADY have norm_lens

Root: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/olmo2_downstream_results/`
(observed via `.82`). The brief's expectation confirmed exactly:

| arm dir (zwfy6) | total | `norm_lens` present |
|---|---|---|
| `7B_base_full_bs16` | 17195 | **yes** |
| `7B_keep14_step200000_bs16` | 17195 | **yes** |
| `7B_keep12_step124000_bs16` | 17195 | **yes** |
| `7B_keep10_step83500_bs16` | 17195 | **yes** |
| `7B_keep8_step121000_bs16` | 17195 | **yes** |
| `7B_shortgpt16_step200000_bs16` | 17195 | **yes** |

Sizes: 11M each except shortgpt16 8.5M; **62M total** for all six. That is a ~2-5 minute
`scp -O`, not a blocker — but it is also **not needed**, see §3.

zwfy6 additionally has ~22 arms with 54 per_example files (`*_v2`, `*_v3`, `*_bs4/8/32`,
`7B_p24_sft_keep{8,10,12,14}fresh2_final`), i.e. the bs-ladder and SFT-ladder. Those are
B04/Table-4 assets, not needed for gate-2.

## 3. Do the per-item files carry what null calibration needs? (ACTUAL keys)

**The expected schema in the brief is WRONG for wzc1.** Observed on
`olmo2_downstream_results/7B_full32_base_wzc1_v2/per_example_hellaswag.jsonl`:

```
KEYS: ['acc_norm_score', 'correct', 'gold_letter', 'item_id', 'nan', 'option_scores', 'pred_letter']
{"item_id": 0, "gold_letter": "D", "pred_letter": "C", "correct": false,
 "option_scores": {"A": -43.516998, "B": -33.61351, "C": -25.603573, "D": -26.748075},
 "acc_norm_score": 1.0, "nan": false}
```

* `item_id`, `gold_letter`, `pred_letter`, `correct`, `option_scores`, `acc_norm_score`,
  `nan` — **all present**, all 6 core6 tasks + boolq/csqa/siqa.
* **`norm_lens` and `norm_scores` are ABSENT on all 11 wzc1 arms** (checked head of each
  `per_example_hellaswag.jsonl`; grep count 0). They are **present on zwfy6 `*_bs16`**.
* **`option_scores` are raw summed log-likelihoods, NOT length-normalized.** Verified:
  `pred_letter` = argmax(`option_scores`) and `correct` tracks that, while
  `acc_norm_score` is an independent 0/1 that disagrees with `correct` on 12-33% of items
  (raw-vs-norm agreement: hellaswag 0.6764, obqa 0.7240, arc_ch 0.8029, arc_easy 0.8725,
  piqa 0.8634, winogrande **1.0000**).
* So **each row already encodes TWO scoring interfaces**: `correct` (raw sum-LL) and
  `acc_norm_score` (char-length-normalized). This is the direct analogue of MMLU's
  letter-vs-content pair, which is exactly what gate-2 needs.

**winogrande, as the brief suspected, is degenerate**: 2 options, and its two options are
scored on a *shared continuation* (`(prefix + option, " " + target, len(target))` in
`load_task_examples`), so both candidates have **identical `norm_lens`** (e.g. `{A:28,B:28}`).
Consequence: length normalization is a no-op → `acc == acc_norm` exactly (0.7459 = 0.7459),
raw-vs-norm agreement 1.0000, and its longest-option null is exactly 0.5000 with a
**100% tie rate**. **Winogrande cannot serve as a gate-2 interface case.** Report it as the
structural-negative control, do not count it as evidence.

**Recovering `norm_lens` on wzc1 costs nothing.** `scripts/enrich_per_example_normscores.py`
rebuilds them from HF datasets. I verified all six rebuild successfully **and match zwfy6's
stored values bit-for-bit** (hellaswag item 0 → `[37,27,26,36]`, arc_easy item 0 →
`[59,54,50,43]`, piqa item 0 → `[164,166]` — identical to the zwfy6 `_bs16` rows).

> ⚠️ **Requires the proxy.** With `HF_DATASETS_OFFLINE=1` only `arc_easy` resolves; the
> local cache `/root/.cache/huggingface/datasets/` holds only ai2_arc(ARC-Easy)/mmlu/glue/
> etc. — **no Rowan/hellaswag, no ybisk/piqa, no allenai/openbookqa, no allenai/winogrande
> dataset cache** (hub blobs exist under `.../hub/datasets--*` but `datasets` still wants a
> network resolve). With `http_proxy=http://hy-proxy.woa.com:3128` all six succeeded.
> Net: gate-2 needs **outbound proxy**, not GPUs.

## 4. Gate-2 answered — results I computed during this scout

Provenance asserts run first, all PASS:
* **Item alignment**: for all 6 core6 tasks and all 3 know-MC tasks, `item_id` and
  `gold_letter` sequences are **identical across the 6 arms** (the assert that
  `load_mmlu_arms` makes for MMLU).
* **Shard + cardinality integrity**: 12 dirs x tasks — every `summary.json` has
  `n_shards=8`, `n_scored` equals expected cardinality, `n_nan=0`, and all **8/8** shard
  files exist on disk. Zero failures. (This is the `assert n_scored==expected` discipline
  the brief demands, applied per task rather than only checking n_nan.)

### 4a. The nulls are NOT the naive chance line (this is the whole point)

`always-<best letter>` best-constant null, from gold distribution, base arm:

| task | n | naive chance | **best-constant** | longest-option (split) | shortest-option (split) |
|---|---|---|---|---|---|
| hellaswag | 10042 | .2500 | .2573 (always-C) | .2408 | .2541 |
| arc_challenge | 1172 | .2501 | .2654 (always-B) | **.2741** | .1979 |
| arc_easy | 2376 | .2501 | .2664 (always-C) | .2553 | .2526 |
| piqa | 1838 | .5000 | .5049 (always-B) | .4752 | **.5248** |
| **openbookqa** | 500 | .2500 | .2760 (always-A) | **.3635** | .1542 |
| winogrande | 1267 | .5000 | .5043 (always-B) | .5000 (100% tie) | .5000 |
| **boolq** | 3270 | .5000 | **.6217 (always-B)** | n/a | n/a |
| commonsense_qa | 1221 | .2000 | .2088 (always-B) | n/a | n/a |
| social_iqa | 1954 | .3333 | .3362 (always-C) | n/a | n/a |

**Two headline non-MMLU nulls that a chance line badly misstates:**
* **BoolQ: .6217 vs .5000** — the gold distribution is 2033 "B" / 1237 "A". A 12.2pp
  understatement. Anyone reporting BoolQ against 50% is crediting the model for 12pp of
  label skew.
* **OpenBookQA: longest-option .3635 vs .2500** — an 11.4pp understatement, on the
  *content* interface, structurally identical to MMLU's `.2845` case.

### 4b. Arms at or below their own construct-appropriate floor

Floor for the norm interface = max(longest-option-split, best-constant);
for the raw interface = max(shortest-option-split, best-constant).

* **openbookqa / keep10, acc_norm = 0.3560 vs floor 0.3635 → residual −0.0075 (BELOW).**
* **openbookqa / keep8, acc (raw) = 0.2700 vs floor 0.2760 → residual −0.0060 (BELOW).**
* **boolq / raw interface: keep12 0.6101, keep10 0.6086, keep8 0.5948 — all BELOW the
  0.6217 always-B floor.** keep14 0.6382 is +0.0165 but McNemar p=0.20, i.e. **n.s.**

So on BoolQ's raw interface, **4 of 6 arms are not distinguishable from a constant
predictor**, while their nominal accuracies (0.59-0.64) look like "passing" numbers against
a 0.50 chance line. This is precisely the A01 thesis, reproduced off MMLU.

McNemar exact (paired, arm vs the deterministic constant predictor on identical items),
BoolQ raw interface:

| arm | acc | resid vs .6217 | McNemar b / c | p | verdict |
|---|---|---|---|---|---|
| base | .8156 | +.1939 | 832 / 198 | 6.3e-93 | ABOVE |
| keep14 | .6382 | +.0165 | 892 / 838 | 2.0e-01 | **n.s.** |
| keep12 | .6101 | −.0116 | 804 / 842 | 3.6e-01 | **BELOW** |
| keep10 | .6086 | −.0131 | 557 / 600 | 2.2e-01 | **BELOW** |
| keep8 | .5948 | −.0269 | 468 / 556 | 6.5e-03 | **BELOW, sig.** |
| shortgpt16 | .7297 | +.1080 | 780 / 427 | 1.9e-24 | ABOVE |

BoolQ acc_norm interface is friendlier (keep12 +.0324 p=2.2e-06 ABOVE; keep10 +.0070 and
keep8 +.0052 both n.s.) — **so BoolQ also exhibits an interface-dependent verdict flip**:
keep12 is "significantly above floor" under acc_norm but "below floor" under raw LL.

CommonsenseQA and SocialIQA are clean (all 6 arms significantly ABOVE their constant-letter
nulls on both interfaces; e.g. csqa keep8 +.2432 p=9.5e-35, siqa keep8 +.0614 p=8.0e-05).
They are **calibration-survives** cases — good for the "not everything collapses" side of
the protocol, but they do not change conclusions.

### 4c. Interface ranking flips on non-MMLU MC (acc vs acc_norm, 15 pairs each)

| task | sign flips /15 | note |
|---|---|---|
| hellaswag | 0 | order stable |
| arc_challenge | 0 | order stable |
| arc_easy | 0 | order stable |
| piqa | 1 | keep14 vs shortgpt16 swap |
| **openbookqa** | **3** | acc_norm: base>sg16>k14>k12>k8>k10; acc: base>k14>sg16>k10>k12>k8 |
| winogrande | 0 | degenerate by construction (acc≡acc_norm) |

**OpenBookQA is the cleanest gate-2 analogue of the MMLU case**: 3/15 sign flips AND an
arm (keep10) sitting below the content-interface floor — i.e. the flips involve arms that
have no valid signal, exactly the self-falsification structure A01 already used for MMLU.

### 4d. Residual fractions, base arm, acc_norm interface — null choice matters a lot

| task | reported | floor | residual | resid frac | resid frac vs naive chance |
|---|---|---|---|---|---|
| hellaswag | .8048 | .2573 | .5475 | 68.0% | 68.9% |
| arc_challenge | .5725 | .2741 | .2984 | 52.1% | 56.3% |
| arc_easy | .8283 | .2664 | .5619 | 67.8% | 69.8% |
| piqa | .8107 | .5049 | .3058 | 37.7% | 38.3% |
| **openbookqa** | **.4620** | **.3635** | **.0985** | **21.3%** | **45.9%** |
| winogrande | .7459 | .5043 | .2415 | 32.4% | 33.0% |

OpenBookQA: using the chance line inflates the "real signal" by **2.15x** (45.9% → 21.3%).
That is a direct non-MMLU replication of the "wrong null inflates the claim" pattern, and
it lands inside the paper's stated 8%-77% residual-fraction band.

### 4e. Tie-convention robustness (openbookqa longest-option, keep10 = 0.3560)

| conv | null | keep10 residual |
|---|---|---|
| split (pre-registered) | .3635 | **−0.0075** |
| first | .3620 | **−0.0060** |
| last | .3620 | **−0.0060** |
| credit (optimistic) | .4160 | **−0.0600** |
| wrong (pessimistic) | .3200 | +0.0360 |

Below-floor under **4 of 5** conventions; only the maximally pessimistic `wrong` convention
rescues keep10. MAIN should report the sweep, as A01 already does for MMLU — do not report
`split` alone.

## 5. Is any new GPU work needed?

**No.** Concretely:
* No missing arms: 6 item-aligned arms (base, keep14, keep12, keep10, keep8, shortgpt16)
  exist on wzc1 for all 9 MC benchmarks, all shard-complete, all `n_nan=0`.
* No >4-option scoring gap: CommonsenseQA (5-way) is already scored with A..E.
* The two interfaces needed for the flip analysis are **already both stored per row**
  (`correct` = raw sum-LL, `acc_norm_score` = length-normalized).
* The one genuinely missing field (`norm_lens`) is deterministic dataset metadata,
  recomputable on CPU in minutes, and I verified it matches zwfy6's stored values.

**Caveat worth flagging to MAIN, not a blocker:** the `keep12` arm on wzc1 is
`step111500` whereas zwfy6's bs16 ladder uses `step124000`. If MAIN wants gate-2's arm
set to be step-identical to a table already in the paper, check which step that table used.
For gate-2's *methodological* claim (does the null/interface change the verdict) the exact
step is immaterial — but for a cross-table consistency claim it is not.

## 6. Environment gaps found

* **`.21` has no `scipy`** in `/opt/conda/envs/torch-base/bin/python` (numpy 2.5.1 and
  datasets 2.21.0 are there). My McNemar/binomtest code uses `scipy.stats`. **LOCAL conda
  has scipy 1.18.0 / numpy 2.3.5**; LOCAL `.venv` does **not** have scipy. So run the
  recompute on **LOCAL conda**, or implement the exact binomial by hand, or `pip install
  scipy` on `.21`.
* **HF datasets need the proxy** for hellaswag / piqa / openbookqa / winogrande /
  ARC-Challenge (see §3). ARC-Easy alone is fully cached offline.
* `.21` PID 25999 (`prepare_dolmino_llama2.py --stage download`) is CPU+network. A gate-2
  recompute wants the same proxy — **another reason to run it on LOCAL, not `.21`**, so as
  not to compete with that download's bandwidth.
* `.21` GPUs verified idle: all 8 at `0 MiB, 0 %`. 256 cores.

## 7. Recommendation to MAIN

1. **Do not launch gate-2 on `.21` or `.82`.** It is a CPU job; I effectively already ran
   it. Give both nodes to a real GPU lane.
2. Promote this to a persisted artifact: extend
   `proposal/active/A01-null-calibration-methodology/code/build_null_calibration_table.py`
   with a `leg_mc_nonmmlu()` that reads `olmo2_downstream_results/{arm,arm_know}/`, reuses
   the existing `longest_option_vector` tie-convention sweep, and writes
   `evidence/null_calibration_gate2_nonmmlu.json`.
3. **Lead with BoolQ (.6217 always-B) and OpenBookQA (.3635 longest-option).** Those two
   change conclusions. HellaSwag / ARC-easy have ~68% residual fractions and are only
   "decorated" by calibration — reporting them as gate-2 evidence would weaken the case.
4. **Exclude winogrande from the interface-case count** and report it as the degenerate
   structural control (shared continuation → norm is a no-op, acc ≡ acc_norm).
5. Pre-register the tie convention before writing (§4e) — `credit` vs `wrong` moves the
   openbookqa verdict, and A01 already has a retraction on exactly this kind of
   selective-convention reporting (`.2822` vs `.2845`).
6. Note A01's current `evidence/` + `claims/` + `code/` contain **zero** mention of
   hellaswag/arc/piqa/openbookqa/winogrande/boolq/commonsense/social_iqa (verified by
   grep) — so this is genuinely new gate-2 evidence, not a re-derivation.
