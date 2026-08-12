# paperC gate-2 — CROSS-FAMILY REPLICATION (task #250)

Ran 2026-08-11 22:19–22:32 on `.73` (8×H20), **12.5 min wall**, 15 arms × 6 tasks =
**90 arm×task cells**, every cell 8/8 shards, `n_scored == expected`, `n_nan = 0`.
Analysis (CPU, 0 GPU) adds 12 recomputed MMLU cross-family cells.

**Verdict: `REPLICATES_IN_DIRECTION_ACROSS_FAMILIES_BUT_THE_LADDER_DOES_NOT`.**

Short form, for the impatient:

* **The floor claim replicates.** **0 of 60** damaged non-OLMo arm×task cells clear
  their own best-constant letter floor, on any of the three families, on any of the
  five evidence benchmarks. **25 of those 60 read "above chance"** under the naive
  null — the wrong-null flip paperC exists to point at, now reproduced in three
  families off MMLU. 51 of 60 point estimates are negative.
* **The *significance* does not replicate per-benchmark, for a reason that is now
  measured rather than assumed: power.** Only 7 of 60 reach p<0.05, and **52 of 60
  cells are underpowered to have detected MMLU's own −1.389 pp effect**. On
  arc_challenge the median damaged effect is **−3.840 pp — larger than MMLU's
  −3.603 pp — and still n.s., because the CI95 half-width is 3.92 pp vs MMLU's
  1.18 pp.** Pooling the five disjoint benchmarks into one n=7107 paired sample
  recovers part of it: **12/12 damaged arms negative, 4/12 significant.**
* **What does NOT replicate, and is a real narrowing:** #248's "healthy → damaged
  ordering, and k14 is the last arm above its floor" is **false in these three
  families**. Their damaged ladder is a **cliff, not a gradient** — k14 is *already*
  at the floor and the whole k14→k8 spread is 0.5–4.5 pp, non-monotone in 13 of 15
  family×task ladders. That is a **regime difference, not a family difference**
  (these arms are truncate-only, no heal; OLMo-2's are pruned *and* healed), and it
  is a confound that must be stated whenever the two are put side by side.
* **Two new self-falsifications** fell out, both against paperC's own text — see §6.

---

## 0. Damage is an eval-time construction. Verified, not assumed.

`eval_olmo2_probe2_ppl.py::load_truncated_any_family` (lines 175–237):
`AutoModelForCausalLM.from_pretrained(...)` the intact base, then
`model.model.layers = torch.nn.ModuleList(layers[:N])`, sync
`config.num_hidden_layers` and `config.layer_types`, `.to(device)`, `.eval()`.
**No fresh block, no heal steps, no optimizer, no gradient, no checkpoint.** The
returned meta records `"n_fresh_layers": 0, "heal_steps": 0`.

So these 15 arms cost 12.5 minutes of forward passes, and the k8/k12 rungs are the
*same construction* as the archived MMLU gate-1 DAMAGED leg — which is why the
head-to-head in §3 is legitimate. **No training was started for this task.**

⚠️ This is also the source of the regime confound in §5: OLMo-2's arms in #248 are
prune-**then-heal** (121k–200k steps of Dolmino), these are truncate-only. Any
sentence that puts them in one table must say so.

## 1. Model weights — which disk, and one trap

| family | path | disk | note |
|---|---|---|---|
| Llama-2-7B | `../models/Llama--Llama2-7b` | **both** wzc1 + zwfy6 | 32 blocks |
| Llama-3-8B | `../models/Llama--Llama3-8b` | **both** wzc1 + zwfy6 | 32 blocks |
| Qwen3-8B-Base | `../models/Qwen3-8B-Base` | **wzc1 ONLY** → copied | 36 blocks |

**Trap (cost 8 min, would have cost the whole result silently).** zwfy6 has
`models/Qwen--Qwen3-8b` and a `Mixture-of-Memory/models/Qwen3-8b-local` symlink to
it. That is **Qwen3-8B-Instruct, not Base**: `eos_token_id 151645` (`<|im_end|>`),
a `chat_template` in `tokenizer_config.json`, `max_position_embeddings 40960`, and
a README whose `base_model:` field points *at* `Qwen3-8B-Base`. The real Base has
`eos 151643`, no chat template, `32768` ctx. Using it would have silently compared
an **instruction-tuned** model against three base models under a `chat=False`
protocol. `Qwen3-8B-Base` was `scp -O`'d wzc1→zwfy6 (16 GB, ~4 min, **12/12 files
md5-verified**) and lives at
`/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/Qwen3-8B-Base`.

⚠️ `paperB/TODOList`'s `models/Qwen3-8b-local` is therefore **wrong twice**: it is
not just a bad path, it resolves to the wrong *model class*.

**Depth caveat.** OLMo-2-7B / Llama-2 / Llama-3 have 32 blocks; Qwen3-8B-Base has
36. `kN` is the same **absolute** depth but a smaller **fraction** of Qwen3's stack
(k8 = 22.2% vs 25.0%). Absolute N was kept to stay comparable with the archived
MMLU numbers, and Qwen3's rung is the harsher one, not the gentler one.

## 2. Nulls, and where the null itself is under-specified

### 2a. The letter null is a dataset property — asserted, not hoped

The best-constant letter floor is identical across all 15 arms and all three
families (hard-asserted in the analysis script; a drift means the item sets differ
and the cells are not comparable). Every floor reproduces #248's value exactly:

| task | n | n_opt | gold marginal | **best-constant floor** | chance | floor − chance |
|---|---|---|---|---|---|---|
| arc_challenge | 1172 | 4 (+3/5-opt) | A 266 / B 311 / C 310 / D 285 | **always-B `0.265358`** | `0.250156` | +1.520 pp |
| arc_easy | 2376 | 4 (+3/5-opt) | A 596 / B 585 / C 633 / D 561 / E 1 | **always-C `0.266414`** | `0.250161` | +1.625 pp |
| openbookqa | 500 | 4 | A 138 / B 126 / C 132 / D 104 | **always-A `0.276000`** | `0.250000` | +2.600 pp |
| commonsense_qa | 1221 | 5 | A 239 / B 255 / C 241 / D 251 / E 235 | **always-B `0.208845`** | `0.200000` | +0.885 pp |
| piqa | 1838 | 2 | A 910 / B 928 | **always-B `0.504897`** | `0.500000` | +0.490 pp |
| winogrande *(control)* | 1267 | 2 | A 628 / B 639 | **always-B `0.504341`** | `0.500000` | +0.434 pp |
| MMLU (reference) | 14042 | 4 | — | **always-D `0.268908`** | `0.250000` | +1.891 pp |

⚠️ **These floors are only +0.43 to +2.60 pp above chance** — flatter than MMLU's
+1.89 pp. paperC's "chance badly misstates the null" rhetoric stays **weak on the
letter side of these five tasks**; the dramatic case is still BoolQ (`0.6217` vs
`0.50`). **`0.25` is never the null for any of these.**

### 2b. NEW: the longest-option content null is **TOKENIZER-dependent**

`paperC/README.md` documents *two* under-specifications of the longest-option null
(tie convention; character-vs-token length unit). There is a **third**: *within* the
token unit the null is **not a dataset property at all** — "longest" is counted in
continuation tokens, and three families have three tokenizers, so the winner set
and hence the floor differ by family. Measured:

| task | conv | Llama-2 | Llama-3 | Qwen3 | spread |
|---|---|---|---|---|---|
| arc_challenge | `split` | `0.268871` | `0.283902` | `0.282338` | **1.50 pp** |
| arc_challenge | `credit` | `0.446246` | `0.543515` | `0.526451` | **9.73 pp** |
| arc_easy | `credit` | `0.405724` | `0.507155` | `0.503367` | **10.14 pp** |
| openbookqa | `credit` | `0.536000` | `0.642000` | `0.640000` | **10.60 pp** |
| openbookqa | `wrong` | `0.278000` | `0.228000` | `0.230000` | 5.00 pp |
| piqa | `wrong` | `0.323177` | `0.252992` | `0.255169` | 7.02 pp |

Driver: the tied-longest fraction moves a lot (arc_challenge 41.6% → 50.8%;
openbookqa 33.8% → 48.2%; winogrande 48.4% → **86.5%**), because Llama-3's and
Qwen3's larger BPE vocabularies collapse more short option texts to equal token
counts. **Consequence for writing:** a content-interface floor must be quoted
**with the tokenizer**, not just with the convention and the unit. The analysis code
therefore records content nulls **per family** and only asserts invariance *within*
a family. The character unit is not recoverable from these per-item records and is
deliberately not guessed.

## 3. Letter vs its own floor — the full cross-family table

`+` above floor (p<0.05), `=` AT the floor (n.s.), `−` BELOW floor (p<0.05).
`~` = **this cell could not have detected MMLU's own −1.389 pp effect** (CI95
half-width > 1.389 pp). A `=` carrying `~` is *uninformative*, not negative
evidence.

### Llama-2-7B (32 blocks)

| rung | arc_ch | arc_easy | obqa | csqa | piqa | winog (ctrl) |
|---|---|---|---|---|---|---|
| base | `0.4343` `+`~ | `0.5783` `+`~ | `0.4000` `+`~ | `0.3251` `+`~ | `0.5664` `+`~ | `0.4886` `=`~ |
| k14 | `0.2261` **`−`**~ | `0.2517` `=`~ | `0.2720` `=` | `0.1966` `=`~ | `0.4935` `=`~ | `0.4957` `=`~ |
| k12 | `0.2261` **`−`**~ | `0.2508` `=`~ | `0.2760` `=` | `0.1957` `=`~ | `0.4946` `=`~ | `0.4957` `=`~ |
| k10 | `0.2218` **`−`**~ | `0.2433` `=`~ | `0.2600` `=`~ | `0.1974` `=`~ | `0.4804` `=`~ | `0.4862` `=`~ |
| k8 | `0.2304` `=`~ | `0.2630` `=`~ | `0.2500` `=`~ | `0.2039` `=`~ | `0.4984` `=`~ | `0.5012` `=` |

### Llama-3-8B (32 blocks)

| rung | arc_ch | arc_easy | obqa | csqa | piqa | winog (ctrl) |
|---|---|---|---|---|---|---|
| base | `0.7782` `+`~ | `0.9120` `+`~ | `0.7460` `+`~ | `0.6953` `+`~ | `0.7791` `+`~ | `0.5320` `=`~ |
| k14 | `0.2628` `=`~ | `0.2673` `=`~ | `0.2640` `=`~ | `0.2138` `=` | `0.4940` `=` | `0.5043` `=` |
| k12 | `0.2585` `=`~ | `0.2500` `=`~ | `0.2320` `=`~ | `0.2187` `=`~ | `0.4913` **`−`** | `0.5043` `=` |
| k10 | `0.2415` `=`~ | `0.2370` **`−`**~ | `0.2020` **`−`**~ | `0.2088` `=`~ | `0.4799` `=`~ | `0.4988` `=`~ |
| k8 | `0.2176` **`−`**~ | `0.2391` `=`~ | `0.2720` `=`~ | `0.2039` `=`~ | `0.4908` `=`~ | `0.4957` `=`~ |

### Qwen3-8B-Base (36 blocks)

| rung | arc_ch | arc_easy | obqa | csqa | piqa | winog (ctrl) |
|---|---|---|---|---|---|---|
| base | `0.9249` `+`~ | `0.9735` `+`~ | `0.8440` `+`~ | `0.8632` `+`~ | `0.8727` `+`~ | `0.6709` `+`~ |
| k14 | `0.2270` `=`~ | `0.2504` `=`~ | `0.2760` `=` | `0.1957` `=`~ | `0.4984` `=`~ | `0.4949` `=`~ |
| k12 | `0.2270` `=`~ | `0.2504` `=`~ | `0.2760` `=` | `0.1925` `=`~ | `0.4956` `=`~ | `0.4957` `=`~ |
| k10 | `0.2270` `=`~ | `0.2504` `=`~ | `0.2760` `=` | `0.1982` `=`~ | `0.4946` `=`~ | `0.4957` `=`~ |
| k8 | `0.2474` `=`~ | `0.2551` `=`~ | `0.2600` `=`~ | `0.1925` `=`~ | `0.5054` `=`~ | `0.5051` `=` |

Tally over the **60** damaged (non-base, non-control) cells:

| statistic | value |
|---|---|
| cells **above** their own floor | **0 / 60** |
| cells AT the floor (n.s.) | 53 / 60 |
| cells **BELOW** the floor (p<0.05) | 7 / 60 |
| negative point estimate | 51 / 60 |
| **"above chance" but NOT above floor** (the wrong-null flip) | **25 / 60** |
| **underpowered for MMLU's own −1.389 pp** | **52 / 60** |

Sign test 51/60, exact binomial p = 1.5e−08 — **descriptive only**: the 60 tests
share items, nest arms within families and reuse one null per task, so they are not
independent. Do not quote this as an inference.

## 4. The power problem, and the pooled test that partly fixes it

The per-benchmark nulls are underpowered **by construction** — these benchmarks are
6–28× smaller than MMLU:

| task | n | CI95 half-width on damaged letter cells (min / median / max, pp) |
|---|---|---|
| MMLU | 14042 | 0.85 / **1.18** / 1.19 |
| arc_easy | 2376 | 2.44 / 2.84 / 2.88 |
| piqa | 1838 | 0.84 / 4.07 / 4.62 |
| commonsense_qa | 1221 | 1.06 / 3.52 / 3.64 |
| arc_challenge | 1172 | 2.43 / 3.92 / 4.05 |
| openbookqa | 500 | 0.00 / 2.65 / **6.40** |

And the effects are **not** smaller — arc_challenge's median damaged effect is
**−3.840 pp vs MMLU's −3.603 pp**. The `=` verdicts are a measurement-precision
artefact, not evidence of "no collapse".

**Pooled test.** The five evidence tasks are **disjoint item sets** and the letter
null is a per-item input-blind 0/1 vector on each, so concatenating the five paired
difference vectors is the *same estimator on a larger item set* — n = **7107**,
comparable with MMLU's 14042. No new assumption. Winogrande excluded.

| family | rung | pooled acc | pooled floor | Δ pp | CI95 pp | boot p | McNemar p | verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | base | `0.495427` | `0.318700` | **+17.673** | [+16.00, +19.36] | 0.0001 | 1.2e−90 | above |
| | k14 | `0.301956` | | −1.674 | [−3.43, +0.04] | 0.0562 | 0.065 | AT |
| | k12 | `0.302097` | | −1.660 | [−3.42, +0.07] | 0.0625 | 0.069 | AT |
| | k10 | `0.294358` | | **−2.434** | [−4.07, −0.80] | **0.0030** | 0.0038 | **BELOW** |
| | k8 | `0.307443` | | −1.126 | [−2.57, +0.32] | 0.1236 | 0.132 | AT |
| Llama-3-8B | base | `0.806669` | `0.318700` | **+48.797** | [+47.36, +50.25] | 0.0001 | 0 | above |
| | k14 | `0.315745` | | −0.295 | [−1.45, +0.84] | 0.6049 | 0.626 | AT |
| | k12 | `0.307162` | | **−1.154** | [−2.28, +0.00] | **0.0493** | 0.048 | **BELOW** |
| | k10 | `0.293232` | | **−2.547** | [−4.26, −0.84] | **0.0047** | 0.0038 | **BELOW** |
| | k8 | `0.296890` | | **−2.181** | [−3.87, −0.49] | **0.0103** | 0.013 | **BELOW** |
| Qwen3-8B-Base | base | `0.911355` | `0.318700` | **+59.266** | [+57.97, +60.55] | 0.0001 | 0 | above |
| | k14 | `0.303081` | | −1.562 | [−3.32, +0.18] | 0.0795 | 0.087 | AT |
| | k12 | `0.301815` | | −1.688 | [−3.45, +0.06] | 0.0563 | 0.064 | AT |
| | k10 | `0.302519` | | −1.618 | [−3.29, +0.10] | 0.0658 | 0.069 | AT |
| | k8 | `0.308147` | | −1.055 | [−2.39, +0.31] | 0.1198 | 0.129 | AT |

**12 / 12 damaged arms negative; 4 / 12 significant; 0 / 12 above the floor.**

⚠️ **Never quote a pooled verdict as a per-benchmark verdict.** The pooled floor
`0.318700` is a weighted mix of five floors ranging `0.2088`–`0.5049`, and the
pooled accuracy mixes five accuracies. It answers "aggregated over this 5-benchmark
MC suite", nothing narrower. Also note pooling does **not** reach MMLU's precision
(half-widths 1.14–1.75 pp vs 1.18): three of the twelve still fail the
"could have detected −1.389 pp" test.

## 5. Head-to-head: did OLMo-2's gate-2 conclusion replicate?

### 5a. MMLU cross-family, recomputed with the R-7 mid-p estimators

`STATUS.json:gate1_third_model_family_DAMAGED` recorded point deltas only. Same
per-item records, same estimators as everything above:

| family | rung | letter acc | Δ vs always-D `0.268908` | boot p | McNemar p | modal | tie | verdict |
|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | base | `0.409984` | +14.108 | 0.0001 | 7.4e−125 | 0.4145 | 0.1579 | above |
| | k14 | `0.230736` | **−3.817** | **0.0001** | 1.7e−10 | 0.9639 | 0.0040 | **BELOW** |
| | k12 | `0.229454` | **−3.945** | **0.0001** | 3.7e−11 | 0.9997 | 0.0000 | **BELOW** |
| | k10 | `0.249323` | **−1.958** | **0.0008** | 0.0013 | 0.5716 | 0.0055 | **BELOW** |
| | k8 | `0.241490` | **−2.742** | **0.0001** | 5.7e−06 | 0.4486 | 0.0038 | **BELOW** |
| Llama-3-8B | base | `0.621991` | +35.308 | 0.0001 | 0 | 0.2921 | 0.0443 | above |
| | k14 | `0.254807` | **−1.410** | **0.0008** | 0.0010 | 0.5221 | 0.0031 | **BELOW** |
| | k12 | `0.252742` | **−1.617** | **0.0001** | 2.5e−04 | 0.4732 | 0.0031 | **BELOW** |
| | k8 | `0.232873` | **−3.603** | **0.0001** | 1.3e−10 | 0.8848 | 0.0014 | **BELOW** |
| Qwen3-8B-Base | base | `0.746404` | +47.750 | 0.0001 | 0 | 0.3057 | 0.0490 | above |
| | k12 | `0.229953` | **−3.895** | **0.0001** | 6.7e−11 | 0.9905 | 0.0003 | **BELOW** |
| | k8 | `0.228600` | **−4.031** | **0.0001** | 1.4e−11 | 0.7625 | 0.0135 | **BELOW** |

**9 / 9 damaged non-OLMo MMLU cells are significantly BELOW floor.** The
`STATUS.json` point deltas survive the proper estimators.

### 5b. Same arms, second benchmark: 9/9 → 7/60

| | MMLU (n=14042) | five non-MMLU benchmarks |
|---|---|---|
| damaged cells above their floor | 0 / 9 | **0 / 60** |
| damaged cells significantly BELOW | **9 / 9** | **7 / 60** |
| median CI95 half-width | 1.18 pp | 2.65–4.07 pp |
| median damaged effect (arc_ch) | −3.603 pp | **−3.840 pp** |

So the **direction and the floor verdict replicate perfectly (0/60 above floor)**,
the **significance collapses**, and the collapse is **fully explained by n** — the
effect is if anything larger. This is the same shape as #248's OLMo-2 finding, now
with three more families: paperC's substantive claim survives cross-family, its
*statistical* claim needs MMLU-scale n or the pooled construction of §4.

### 5c. Arm-by-arm against OLMo-2's healed arms (⚠️ regime confound)

| task | rung | OLMo-2 (healed) | Llama-2 (trunc) | Llama-3 (trunc) | Qwen3 (trunc) |
|---|---|---|---|---|---|
| arc_ch | k14 | `0.3353` `+` | `0.2261` `−` | `0.2628` `=` | `0.2270` `=` |
| | k12 | `0.2611` `=` | `0.2261` `−` | `0.2585` `=` | `0.2270` `=` |
| | k10 | `0.2816` `=` | `0.2218` `−` | `0.2415` `=` | `0.2270` `=` |
| | k8 | `0.2560` `=` | `0.2304` `=` | `0.2176` `−` | `0.2474` `=` |
| arc_easy | k14 | `0.4533` `+` | `0.2517` `=` | `0.2673` `=` | `0.2504` `=` |
| | k10 | `0.2395` `−` | `0.2433` `=` | `0.2370` `−` | `0.2504` `=` |
| obqa | k14 | `0.3700` `+` | `0.2720` `=` | `0.2640` `=` | `0.2760` `=` |
| csqa | k14 | `0.2735` `+` | `0.1966` `=` | `0.2138` `=` | `0.1957` `=` |
| piqa | k12 | `0.5005` `=` | `0.4946` `=` | `0.4913` `−` | `0.4956` `=` |

**Replicates:** the floor verdict at every deep-damage rung (nothing clears its
floor anywhere).
**Does NOT replicate:** OLMo-2's `k14` is `+` (clearly above floor) on
arc_ch / arc_easy / obqa / csqa, while **all three non-OLMo `k14`s are already at
or below their floors**. #248's "k14 is the last arm that still clears its floor"
is an **OLMo-2 (healed) statement**, not a family-general one.
**⚠️ This is a regime confound, not a family effect.** OLMo-2's arms had 121k–200k
heal steps; these had none. The correct reading is "heal buys back above-floor
letter competence at k14", which this experiment cannot separate from family.
Testing it needs healed non-OLMo arms — real training, not in scope here.

### 5d. The non-OLMo damaged ladder is a **cliff, not a gradient**

| family | task | base | k14 | k12 | k10 | k8 | damaged spread | monotone? |
|---|---|---|---|---|---|---|---|---|
| Llama-2 | arc_ch | 0.4343 | 0.2261 | 0.2261 | 0.2218 | 0.2304 | **0.85 pp** | no |
| Llama-2 | csqa | 0.3251 | 0.1966 | 0.1957 | 0.1974 | 0.2039 | **0.82 pp** | no |
| Llama-3 | arc_ch | 0.7782 | 0.2628 | 0.2585 | 0.2415 | 0.2176 | 4.52 pp | **yes** |
| Llama-3 | obqa | 0.7460 | 0.2640 | 0.2320 | 0.2020 | 0.2720 | 7.00 pp | no |
| Qwen3 | arc_easy | 0.9735 | 0.2504 | 0.2504 | 0.2504 | 0.2551 | **0.46 pp** | no |
| Qwen3 | arc_ch | 0.9249 | 0.2270 | 0.2270 | 0.2270 | 0.2474 | 2.05 pp | no |

Monotone-decreasing in **2 of 15** family×task ladders. Damaged spreads of
0.46–7.00 pp against base→k14 drops of **7.3–72.3 pp** (the 7.3 pp low end is
Llama-2 on piqa, whose intact accuracy is only 0.5664 to begin with). There is no
gradient to measure here: truncation to ≤14 of 32/36 blocks lands the letter
interface on the floor and further truncation moves it within noise. **Do not fit a
depth curve to these five rungs.** (This is consistent with `gate1_depth_curve`
having needed a much finer sweep to find the transition.)

### 5e. Many of these arms are **literally** constant emitters

Unlike OLMo-2 (which reaches the floor via 30.64% bf16 exact ties + argmax index
bias), 28 damaged non-OLMo cells have modal prediction share ≥ 0.99 with **near-zero
tie rate**, and in **16 of them the accuracy equals the marginal of the emitted
letter to machine precision**:

| family | rung | task | modal | share | acc | marginal of that letter | best-constant |
|---|---|---|---|---|---|---|---|
| Llama-2 | k12 | arc_easy | `A` | **1.0000** | `0.250842` | `0.250842` **exact** | `0.266414` |
| Llama-2 | k12 | obqa | `A` | **1.0000** | `0.276000` | `0.276000` **exact** | `0.276000` |
| Llama-2 | k12 | csqa | `A` | **1.0000** | `0.195741` | `0.195741` **exact** | `0.208845` |
| Qwen3 | k14 | arc_ch | `A` | **1.0000** | `0.226962` | `0.226962` **exact** | `0.265358` |
| Qwen3 | k10 | arc_ch | `A` | **1.0000** | `0.226962` | `0.226962` **exact** | `0.265358` |
| Qwen3 | k10 | obqa | `A` | **1.0000** | `0.276000` | `0.276000` **exact** | `0.276000` |
| Qwen3 | k8 | csqa | `E` | **1.0000** | `0.192465` | `0.192465` **exact** | `0.208845` |

Note `Llama-2 k12 / obqa` and `Qwen3 k10 / obqa`: these emit always-`A`, which on
OBQA *is* the best constant, so they land **exactly on** `0.276000` with
Δ = `+0.000 pp`, CI95 = `[0,0]`, p = 1.000. A model that has become a constant
predictor and happens to pick the *optimal* constant is the cleanest possible
illustration of paperC's point — and it is invisible against chance (`0.25`).

**But keep the narrowing.** `modal share` and `floor verdict` remain **DECOUPLED**:
Llama-2 k12 / arc_challenge is 99.91% modal yet only p = 0.0499, while Llama-3
k12 / piqa is 96.46% modal *and* p = 0.0015. High modal share does not imply a
significant below-floor verdict, nor conversely.

## 6. Two self-falsifications, against paperC's own text

**(i) `confirmed_general`'s "content_norm within ±3 pp of letter on every damaged
arm" fails again, in the opposite direction from #248.** #248 found the gap can be
*large* (arc_easy keep8 +38.76 pp). Here the damaged gaps are moderate (60 cells,
range **−2.20 to +9.00 pp**, median **+2.56** — so still outside ±3 pp on about
half of them) — but on the **healthy bases** the interfaces disagree enormously and
**with opposite sign per family**:

| family | arc_ch | arc_easy | obqa | csqa | piqa |
|---|---|---|---|---|---|
| Llama-2-7B | −0.09 | **−11.03** | +0.00 | **−14.33** | **−21.82** |
| Llama-3-8B | **+24.40** | +14.39 | **+28.40** | +10.89 | −2.45 |
| Qwen3-8B-Base | **+37.54** | +18.31 | **+38.00** | +25.23 | +6.80 |

(letter − content_norm, pp.) On Llama-2 **content is up to 21.8 pp better**; on
Qwen3 **letter is up to 38.0 pp better**. Neither "letter is the fair interface"
nor "content is the fair interface" is a family-general statement — reinforcing
paperC's existing retraction of "letter MC is *generally* unreliable", and adding
its mirror image.

**(ii) The longest-option content null is tokenizer-dependent (§2b)** — a third
under-specification, not the two `README.md` lists.

**Also newly measured:** damaged `content_norm` sits **BELOW its own longest-option
`split` floor on 12/12 OBQA cells** across all three families (e.g. Llama-3 k12
`0.244` vs floor `0.367`). Under-floor content on OBQA is the most consistent
signal in this run — consistent with #248's OBQA content floor (`0.3635` char /
`0.3680` token) being the largest floor−chance gap in the suite.

## 7. Negative control (winogrande) — passes, and is NOT evidence

All 15 winogrande cells are at their `0.504341` floor except intact Qwen3
(`0.6709`, genuinely above). No damaged arm shows a false positive.

#248's exact-equality signature (`letter == cnorm` bit-for-bit on OLMo-2 keep14,
because both options share the continuation → identical norm_lens) appears here
too but is **rare, not systematic**: exact `letter == cnorm` in **0 of 15**
cross-family cells, and in only 1 of 6 OLMo-2 cells (keep14). Several cells hit
exact equality against the *null* instead (modal share 1.0000, acc = marginal).
So the winogrande degeneracy is real but **prompt/tokenizer-contingent** — report
it from the tie diagnostics, never assume it.

⚠️ Note Llama-3's winogrande tied-longest fraction is **86.5%** vs Llama-2's 48.4%
— the same tokenizer effect as §2b, at its most extreme.

## 8. Shard integrity

* **90 non-MMLU cells** (15 arms × 6 tasks): every one 8/8 shards,
  `n_scored == EXPECTED_N`, `n_nan = 0`. 15/15 arms have 48/48 per-task shard
  jsonl + 8/8 shard meta json + 6/6 `summary_<task>.json`.
* **12 MMLU cells**: 8/8 shards, `n = 14042`, `n_nan = 0` each.
* Failure-syntax grep (`Traceback (most recent call last)` / `CUDA out of memory` /
  `AssertionError` / `INTEGRITY FAILURE` / `CARDINALITY FAILURE`) over all 128
  shard/merge logs: **zero hits**.
* `merge_task()` raises rather than merging a partial or wrong-cardinality set;
  the analysis loader re-asserts shard count, cardinality, id-uniqueness and
  `n_nan == 0` independently before any statistic is computed.
* ⚠️ The driver log contains **spurious `MERGE FAIL ...: 0/6 tasks merged` lines**
  for every arm. That was a cosmetic bug in the driver's own check
  (`grep -c "^\[merge\]"` cannot match timestamp-prefixed `_log()` output), fixed
  after the run. All 15 arms genuinely merged — 6/6 `summary_<task>.json` each,
  verified by direct `ls`. **The run is valid; only the driver's self-report was
  wrong.** The `wc -l`-based check is now authoritative.

## 9. Provenance

| what | where |
|---|---|
| harness (unchanged from #248) | `scripts/eval_olmo2_mc_letter_content.py` |
| damage constructor | `scripts/eval_olmo2_probe2_ppl.py::load_truncated_any_family` |
| driver | `scripts/_run_mc_letter_content_crossfamily_8gpu.sh` |
| nulls + stats (CPU, re-runnable) | `paperC/code/gate2_crossfamily_nulls.py` |
| results json/csv (1122 rows) | `paperC/evidence/second_mc_benchmark_crossfamily/` — **both disks** |
| per-item records (15 arms × 6 tasks × 8 shards, 130 MB) | `mc_lc_crossfamily_results/` — **both disks** |
| MMLU cross-family per-item records (190 MB) | `olmo2_mmlu_content_results/gate1_*` — **wzc1 ONLY** |
| Qwen3-8B-Base weights | wzc1 `../models/Qwen3-8B-Base`; copied to zwfy6 same name |
| driver log | zwfy6 `logs/gate2_xf_DRIVER.log` |
| analysis log | wzc1 `logs/gate2_xf_nulls_wzc1.log` |

Protocol: `chat_template=False`, `add_bos=0`, fp32 master weights + bf16-autocast
forward, `batch_size=48` (**same value as #248**, deliberately — bf16 batch
composition perturbs low-order bits of summed log-probs, so changing it would be a
protocol delta; measured peak 57–64 GiB of 97.8, and bs=32 vs bs=48 are equal in
wall time, so the eval is compute-bound and there is nothing to gain).
Statistics: paired bootstrap `n_boot=10000`, `boot_seed=7`, two-sided p from the
R-7-fixed mid-p `two_sided_boot_p` imported from `a01_gate3_fp32_vs_bf16.py` (with
local verbatim copies asserted identical, including on the large-zero-atom case
that triggered the original R-7 bug), exact McNemar against the deterministic
constant predictor.

## 10. What may and may not be claimed from this run

**May:**
* "Under structural damage the letter MC interface fails to clear its own
  best-constant floor in **four** model families (OLMo-2, Llama-2-7B, Llama-3-8B,
  Qwen3-8B-Base) on **six** MC benchmarks (MMLU + five non-MMLU): 0 of 60
  non-MMLU damaged cells and 0 of 9 non-OLMo MMLU cells are above their floor."
* "25 of 60 of those cells read 'above chance' under the naive null."
* "Several damaged arms are literally constant emitters whose accuracy equals the
  marginal of the emitted letter to machine precision, including two that land
  exactly on the *optimal* constant."
* "The longest-option content null is under-specified in **three** ways: tie
  convention, character-vs-token unit, and — within the token unit — tokenizer."

**May NOT:**
* ✗ that the below-floor verdict is *significant* per-benchmark off MMLU — 7/60,
  with 52/60 underpowered. Cite §4's power table with any null result.
* ✗ that "k14 is the last arm above its floor" generalises — false in all three
  non-OLMo families (§5c). It is an OLMo-2 **healed**-regime statement.
* ✗ any depth curve / monotone fragility from these five rungs (§5d, 2/15 monotone).
* ✗ that the OLMo-vs-non-OLMo difference is a **family** effect — it is confounded
  with **heal vs no-heal** and this run cannot separate them.
* ✗ a pooled verdict as a per-benchmark verdict (§4).
* ✗ acc-vs-acc_norm length sensitivity as ours — Oostermeijer, ICML 2026
  (arXiv:2607.12767). Any OBQA sign flip here is a replication under damage.
* ✗ anything in `STATUS.json:must_not_resurrect` — in particular `0.25` is not the
  null for any benchmark in this run, and MMLU's is `0.268908`.
* ✗ "the mechanism is exact ties" — quadruple-falsified already, and §5e adds a
  fifth family-specific route (direct modal collapse at ~0% tie rate).
