# paperC #251 — THE POWER WALL, pushed down with MMLU-scale n

Task #251 (OLMo-2 leg, §§1-5). Ran 2026-08-12 on `.73` (8×H20), 6 arm×task cells,
34.5 min wall-clock scoring + CPU analysis.
**Task #252** (cross-family leg, **§6**) extended it to 15 non-OLMo cells and
fixed two integrity defects of the first cross-family launch — see §6g. All 21
cells are now in one table, `mmlu_pro_power_nulls_v2.json`.

**Verdict: `POWER_WALL_CLEARED_AND_THE_MMLU_EFFECT_DOES_NOT_REPLICATE`**
**(on OLMo-2's healed arms — §6c shows it DOES replicate on two un-healed
cross-family arms, so the "MMLU-specific" reading of §2b is retracted).**

The acceptance criterion was one number, and it passed decisively: the achieved
CI95 half-width on the letter-vs-floor test is **0.582–0.906 pp on all 6** OLMo-2
cells and **0.083–0.968 pp on all 21** cells, against MMLU's own effect size of
**1.389 pp**. **21/21 cells are powered** — compared with 1/6 of #248's task set
and 8/60 of #250's cells. For the first time off MMLU, a null result here means
*"the effect is not there"* rather than *"we could not have seen it."*

And the answer that new resolution buys is **negative on the healed arms**. MMLU's
headline (the most damaged arm sits *significantly below* its own best-constant
floor, −1.389 pp, p=0.019) **does not reproduce on MMLU-Pro for OLMo-2**. `keep8`
is **−0.116 pp with CI95 `[−0.698, +0.465]`** — an interval that **excludes
MMLU's −1.389 pp entirely**. This is not a non-observation. It is a **positive
exclusion** of an effect of MMLU's magnitude on this benchmark, for this arm.

> This is the outcome the power table was built to make possible: #248 and #250
> could not distinguish "no effect" from "invisible". Now we can, and the answer
> is "no effect of that size, here". That is a **narrowing of paperC's headline**,
> not a confirmation, and it is the fourth self-falsification this direction has
> produced against itself.
>
> ⚠️ **Then §6 partially un-narrowed it**: two *un-healed* cross-family arms on
> the *same* benchmark ARE significantly below floor (`llama2_7b/k8` p=0.0168,
> `qwen3_8b_base/k8` p=0.0362), so the limiting factor is **heal vs no-heal**, not
> MMLU. And §6 added a *new* counterexample in the other direction
> (`qwen3_8b_base/k14` significantly **above** floor). Read §§1-5 and §6 together;
> neither alone gives the scope.

---

## 0. Why MMLU-Pro, and why no smaller benchmark could have done this

#248 and #250 both returned PARTIAL for the **same** reason: n.

| task | n | keep8 CI95 half-width | n needed for hw < 1.389 pp | multiple |
|---|---|---|---|---|
| MMLU | 14042 | **1.154 pp** | (reference) | — |
| arc_easy | 2376 | 1.305 pp | ~2097 | 0.9× ✅ |
| piqa | 1838 | 2.775 pp | ~7336 | 4.0× |
| commonsense_qa | 1221 | 3.399 pp | ~7312 | 6.0× |
| arc_challenge | 1172 | 3.882 pp | ~9154 | 7.8× |
| openbookqa | 500 | 6.400 pp | ~10615 | **21.2×** (test set is 500) |
| **mmlu_pro (this run)** | **12032** | **0.582 pp** | ~2112 (from its own hw) | **5.7× more n than needed** |

**Crucially this was never an effect-size problem.** #250 recorded that
arc_challenge's median damaged effect is **−3.840 pp, LARGER than MMLU's
−3.603 pp**, yet n.s. because its CI is wider than the effect. Adding more small
benchmarks can never fix that; only n can. MMLU-Pro is the first non-MMLU MC
benchmark at MMLU's order of magnitude (12032 vs 14042).

Measured empirically: the median half-width here implies **n ≈ 3186** would have
sufficed for the 1.389 pp threshold. MMLU-Pro's 12032 overshoots it comfortably,
which is why every cell — including the intact base arm, whose higher accuracy
widens its interval — clears the bar.

### Data provenance

MMLU-Pro was **on neither disk** (searched wzc1 `../data/hf_datasets/`,
`data/hf_datasets_cache/`, `~/.cache/huggingface/{datasets,hub}`, and zwfy6
equivalents from `.73`; the only hit was
`dllm_draft/vendor/Dream-Coder/base/lm_eval/tasks/mmlu_pro/`, which is **task YAML
only, no data**). Downloaded via the `hy-proxy` HTTP proxy from
`TIGER-Lab/MMLU-Pro`: **4.1 MB** test parquet + 43 KB validation, `scp -O`'d to
zwfy6, md5 verified identical on both disks
(`7e40550a7f75263861c7b9789eec9e3c`).

---

## 1. The 10-way letter null, computed on 0 GPU before a card was touched

This was the gate: if MMLU-Pro's letter floor were barely above chance, the
benchmark would be weak for paperC's rhetoric and not worth the cards.

| quantity | value |
|---|---|
| n | **12032** |
| n_opt | **NOT constant**: {10: 9981, 9: 801, 8: 320, 7: 158, 6: 93, 5: 52, 4: 606, 3: 21}, mean 9.474 |
| gold marginal | A `.116606` B `.112367` C `.109209` D `.111037` E `.095495` F `.093750` G `.098238` H `.092670` I `.092088` J `.078541` |
| **best-constant floor** | **always-A = `0.116606`** (1403/12032) |
| worst constant | always-J = `0.078541` |
| marginal spread | **3.807 pp** (A−J) |
| chance, naive 1/10 | `0.100000` → floor is **+1.661 pp**, **1.1661×** |
| chance, mean(1/n_opt) | `0.110877` → floor is **+0.573 pp**, **1.0517×** |

### 1a. The honest reading of "+1.661 pp": strong in RELATIVE terms, mid-pack in absolute

#248 recorded that on its five benchmarks the letter floor is only **+0.43 to
+2.60 pp** above chance, making paperC's "chance badly misstates the null"
rhetoric **weak there**. MMLU-Pro's **absolute** gap (+1.661 pp vs naive chance)
sits in the middle of that same range — it is **not** dramatic in pp.

But pp is the wrong unit when comparing a 10-way benchmark to a 4-way one. The
**relative** misstatement is:

| benchmark | floor | chance | floor − chance | **floor / chance** |
|---|---|---|---|---|
| **mmlu_pro (naive 1/10)** | `0.116606` | `0.100000` | +1.661 pp | **1.1661×** |
| MMLU | `0.268908` | `0.25` | +1.891 pp | 1.0756× |
| openbookqa | `0.276000` | `0.25` | +2.600 pp | 1.1040× |
| arc_easy | `0.266414` | `0.250161` | +1.625 pp | 1.0650× |
| arc_challenge | `0.265358` | `0.250156` | +1.520 pp | 1.0608× |
| commonsense_qa | `0.208845` | `0.20` | +0.885 pp | 1.0442× |
| piqa | `0.504897` | `0.50` | +0.490 pp | 1.0098× |

**On the relative scale MMLU-Pro is the STRONGEST case in the whole paper** —
a 16.6% overstatement of the model's headroom, vs MMLU's 7.6%. Report it
relatively, and say so; quoting only the pp gap understates it, and quoting only
the ratio would overstate it against 2-way tasks.

⚠️ **The `mean(1/n_opt)` reading nearly kills this.** Because 2051 of 12032 items
have fewer than 10 options, item-averaged chance is `0.110877`, not `0.10`, and
against *that* the floor is only **+0.573 pp / 1.0517×** — weaker than MMLU's.
Both readings are in the JSON. **Which one is "chance" is itself a degree of
freedom**, and a reader told "10-way" will assume `0.10`. The verdicts below are
identical under both (3/3 damaged arms above chance and not above floor either
way), so nothing here depends on the choice — but it must be **printed**, and
this is a *fourth* under-specification, now on the **letter** side, of a kind
paperC had only documented for the content null.

---

## 2. Letter interface vs the best-constant floor — the whole point of the task

`+` = above floor (p<0.05), `=` = AT the floor (n.s.), `−` = BELOW floor (p<0.05).

| arm | letter acc | Δ vs floor | CI95 | **hw** | boot p | McNemar p | verdict | powered? |
|---|---|---|---|---|---|---|---|---|
| `7B_base` | `0.271858` | **+15.525 pp** | [+14.611, +16.423] | 0.906 | 1e-4 | 1.6e-246 | `+` | ✅ |
| `shortgpt16` | `0.153341` | **+3.674 pp** | [+2.967, +4.372] | 0.702 | 1e-4 | 1.8e-24 | `+` | ✅ |
| `keep14` | `0.119847` | +0.324 pp | [−0.316, +0.956] | 0.636 | 0.3234 | 0.334 | `=` | ✅ |
| `keep12` | `0.113115` | −0.349 pp | [−1.105, +0.424] | 0.765 | 0.3805 | 0.379 | `=` | ✅ |
| `keep10` | `0.112450` | −0.416 pp | [−1.139, +0.316] | 0.727 | 0.2662 | 0.268 | `=` | ✅ |
| `keep8` | `0.115442` | **−0.116 pp** | **[−0.698, +0.465]** | **0.582** | 0.7118 | 0.714 | `=` | ✅ |

### 2a. What REPLICATES

1. **The wrong-null verdict flip — 3/3 damaged arms.** All of keep8/keep10/keep12
   sit **above** their naive chance line (`0.1154`, `0.1125`, `0.1131` vs `0.1000`,
   and also vs `0.110877`) yet **0/3 clear their own best-constant floor**
   `0.116606`. Under the wrong null all three read "above chance, residual
   competence"; under the right one all three are indistinguishable from a
   constant predictor. **This is paperC's core substantive claim and it holds on
   a 10-way benchmark with the strongest resolution in the paper, under BOTH
   chance definitions.**
2. **The healthy → damaged ordering.** base `0.2719` > shortgpt16 `0.1533` >
   keep14 `0.1198` > keep12 `0.1131` ≈ keep10 `0.1125` ≈ keep8 `0.1154`, matching
   MMLU and all five #248 tasks.
3. **The floor-arrival point.** `keep14` is the last arm above/at the boundary and
   keep12-and-below are at the floor — the same arrival point #248 found on 4/5
   tasks. (Here keep14 is already `AT`, not above, so MMLU-Pro is *slightly more*
   sensitive, like PIQA was.)
4. **All 3 damaged point estimates are negative** (−0.116, −0.349, −0.416 pp).

### 2b. What does NOT replicate — and this time it is a REAL null, not a blind spot

MMLU's headline is that `keep8` is **significantly below** its floor
(−1.389 pp, p=0.019). On MMLU-Pro:

* `keep8` Δ = **−0.116 pp**, p = 0.712, **CI95 [−0.698, +0.465]**.
* **That CI excludes −1.389 pp.** An effect of MMLU's magnitude is *ruled out*
  here at the 95% level, not merely undetected.
* No arm reaches p<0.05 below floor. The most negative is keep10 at −0.416 pp
  (p=0.266), also with a CI ([−1.139, +0.316]) that excludes −1.389.

So the correct statement is **not** "MMLU-Pro is another underpowered null". It is:

> **On MMLU-Pro the damaged letter interface lands ON its best-constant floor —
> statistically indistinguishable from a constant predictor, and measurably NOT
> below it by anything like the margin MMLU shows.**

Which is, note, still fully consistent with the *claim paperC actually makes after
its 2026-08-10 narrowing*: "damage drives letter **to or below** its
best-constant floor". `AT the floor` satisfies that. What fails is only the
**strong** below-floor form, and it now fails **with power**, which is worth more
than the five underpowered nulls it replaces.

### 2c. Head-to-head: the same arm, the same estimator, seven benchmarks

`keep8` letter vs its own floor, sorted by resolution:

| task | n | acc | Δ pp | hw pp | p | powered? | verdict |
|---|---|---|---|---|---|---|---|
| openbookqa | 500 | `0.2580` | −1.800 | 6.400 | 0.569 | ✗ | `=` |
| arc_challenge | 1172 | `0.2560` | −0.939 | 3.882 | 0.644 | ✗ | `=` |
| commonsense_qa | 1221 | `0.1982` | −1.065 | 3.399 | 0.530 | ✗ | `=` |
| piqa | 1838 | `0.5299` | +2.503 | 2.775 | 0.075 | ✗ | `=` |
| arc_easy | 2376 | `0.2584` | −0.800 | 1.305 | 0.232 | ✅ | `=` |
| winogrande (ctrl) | 1267 | `0.5099` | +0.552 | 1.184 | 0.357 | ✅ | `=` |
| **MMLU** | **14042** | `0.2550` | **−1.389** | **1.154** | **0.019** | ✅ | **`−`** |
| **mmlu_pro** | **12032** | `0.1154` | **−0.116** | **0.582** | 0.712 | ✅ | `=` |

**MMLU is now the only benchmark in the paper, of the three with adequate power,
that yields a significant below-floor verdict.** arc_easy (hw 1.305, powered) and
mmlu_pro (hw 0.582, best in class) both say `AT the floor`. That materially
weakens the generality of the below-floor headline and **strengthens** the
`AT-or-below` formulation.

> ⚠️ **CORRECTED 2026-08-12 by §6.** This subsection originally continued: "It
> also means the below-floor result may be an **MMLU-specific** property … treat
> 'significantly below floor' as **MMLU-specific until shown otherwise**." The
> cross-family leg **falsifies that**: on MMLU-Pro, `llama2_7b/k8` (−0.914 pp,
> p = 0.0168) and `qwen3_8b_base/k8` (−0.881 pp, p = 0.0362) are both
> significantly below floor. So MMLU-Pro *does* produce below-floor verdicts —
> just not on OLMo-2's **healed** arms. The live hypothesis is **heal vs
> no-heal**, which is #250's known confound, **not** benchmark identity. Do not
> cite the "MMLU-specific" phrasing; it was inferred from the OLMo-2 leg alone.

⚠️ The keep14 row is instructive in the opposite direction: on the four
underpowered small tasks keep14 reads `above the floor` (obqa +9.400 pp p=5e-4,
arc_ch +6.997 p=1e-4, csqa +6.470 p=1e-4, arc_easy +18.687 p=1e-4) but on
mmlu_pro it is `AT` (+0.324 pp, p=0.323). MMLU-Pro is simply a much harder
benchmark: keep14's absolute headroom over a 0.1166 floor is small. **Do not read
that as a contradiction of #248's ordering** — the ordering is preserved; only the
absolute headroom shrinks.

### 2d. Modal share and floor verdict stay DECOUPLED

Consistent with the 2026-08-10 narrowing, and here quite starkly: `keep8` is
**57.50% modal (A)** and `keep12` **57.45% modal (B)** — far from constant — yet
both sit at the floor. All six arms still emit **7-10 distinct letters**, so **no
arm is a literal constant emitter** on MMLU-Pro (unlike the 16 such cells #250
found on the small benchmarks). Letter exact-tie rates are 0.1197-0.2162 on the
damaged arms vs 0.0051 on base — the same OLMo-2 bf16-tie pattern, which gate-3
already **falsified as *the* mechanism**; do not resurrect it.

Note `keep8`'s modal letter is **A**, which is *also* the best-constant letter.
That is why its Δ is nearly exactly 0: an arm that over-emits A on a benchmark
whose gold marginal peaks at A gets the floor's accuracy almost by construction.
This is the "invisible against chance" pathology #250 flagged (its two cells
landing exactly on the optimal constant), reappearing in a milder form — and it
is a reason a floor test must be reported with the **modal letter identity**, not
just the modal share.

---

## 3. Content interface vs the longest-option null — 10-way puts it under real pressure

Longest-option floor, **continuation-token** unit, **OLMo-2 tokenizer**:

| convention | floor |
|---|---|
| `split` | `0.193150` |
| `first` | `0.195894` |
| `last` | `0.190824` |
| **`credit`** | **`0.532164`** |
| `wrong` | `0.125914` |

Diagnostics: **56.24%** of items have a tied-longest set, **mean winner-set size
4.178**, gold ∈ winner set on **53.22%**. The winner-set histogram is bimodal —
5265 items with a unique longest option and **3026 items where ALL ~10 options tie**.

**This is the most extreme demonstration in the paper of how badly
under-specified the longest-option null is.** The `credit` convention
("count a tie as correct") scores **0.532164** — 4.6× the `wrong` convention's
`0.125914`, a **40.6 pp** span across conventions on ONE dataset with ONE
tokenizer — and it **beats the intact base model's content_norm accuracy
(`0.207613`) by 32.5 pp**. A pure length heuristic with oracle tie-breaking
"outperforms" a healthy 7B model by a factor of 2.6 on this benchmark. The
10-way structure is exactly why: with ~9.5 candidates, ties are common and
`credit` harvests them all.

Consequence: `content_norm` is **significantly BELOW** its `split` floor on
**5 of 6 arms** (only base clears it, +1.446 pp, p=0.001), and **BELOW the
`credit` floor on 6/6** by 32-42 pp. Verdicts:

| arm | content_norm | vs `split` | vs `credit` |
|---|---|---|---|
| base | `0.207613` | +1.446 pp, p=0.0010 `+` | −32.455 pp `−` |
| shortgpt16 | `0.144033` | −4.912 pp, p=1e-4 `−` | −38.813 pp `−` |
| keep14 | `0.132979` | −6.017 pp, p=1e-4 `−` | −39.919 pp `−` |
| keep12 | `0.123504` | −6.965 pp, p=1e-4 `−` | −40.866 pp `−` |
| keep10 | `0.109458` | −8.369 pp, p=1e-4 `−` | −42.271 pp `−` |
| keep8 | `0.110040` | −8.311 pp, p=1e-4 `−` | −42.212 pp `−` |

⚠️ **A content floor must be quoted with its tokenizer**, per #250's third
self-falsification. These are OLMo-2's numbers; the cross-family run (in flight,
§6) will quantify the tokenizer spread at 10-way, where it should be larger than
the 1.50 pp / 10.6 pp #250 measured at 4-way. **Do not generalise these floors
across families.**

### 3a. `confirmed_general[2]`'s "±3 pp" survives HERE (unlike on arc_easy)

content_norm − letter, per arm:

| arm | Δ (content_norm − letter) | CI95 | McNemar p |
|---|---|---|---|
| base | **−6.425 pp** | [−7.414, −5.436] | 1.2e-35 |
| shortgpt16 | −0.931 pp | [−1.828, −0.083] | 0.038 |
| keep14 | +1.313 pp | [+0.482, +2.136] | 1.9e-03 |
| keep12 | +1.039 pp | [+0.208, +1.837] | 0.013 |
| keep10 | −0.299 pp | [−1.097, +0.490] | 0.472 |
| keep8 | −0.540 pp | [−1.338, +0.266] | 0.194 |

All four damaged arms are **within ±1.4 pp** — comfortably inside the ±3 pp the
MMLU-scoped bullet asserts, and nothing like arc_easy's **+38.76 pp** blow-up.
So the re-scoping #248 forced ("MMLU only") is **too pessimistic**: the bullet
holds on MMLU *and* MMLU-Pro and fails on the five small benchmarks. The
distinguishing feature is plausibly **benchmark difficulty** — on arc_easy the
damaged model retains real content-side competence (`0.6460`) that the letter
readout cannot express, whereas on MMLU-Pro there is little competence left to
express (content_norm `0.1100` vs floor-level letter `0.1154`). **State it as
"holds on MMLU and MMLU-Pro; fails on easier benchmarks where residual
competence survives", not as a family/dataset accident.**

Also note **base is the one arm that violates ±3 pp here, and in the direction
that favours letter** (letter beats content_norm by 6.4 pp) — consistent with
"content is the fair interface" being **not** a general claim.

### 3b. Residual-fraction inflation from using chance

Base arm, letter interface (acc `0.271858`, floor `0.116606`):

| null used | residual fraction | inflation |
|---|---|---|
| correct (always-A `0.116606`) | **57.11%** | — |
| naive chance `0.100000` | 63.22% | **1.107×** |
| mean(1/n_opt) `0.110877` | 59.22% | 1.037× |

Modest next to OpenBookQA's **2.11×**, and honestly so: at a floor of 0.117 with
accuracy 0.272 there is a lot of genuine headroom, so misstating the null by
1.7 pp changes the residual by only ~11%. **Chance inflates the claim here too,
but mildly.** Do not quote MMLU-Pro as a dramatic inflation case.

---

## 4. Pooling: MMLU-Pro nearly HALVES the pooled CI, and flips one verdict

#250 prototyped a pooled test over its five **disjoint** benchmarks (n=7107).
MMLU-Pro's items are disjoint from all five, so it extends the same construction
to **n=19139** with no new assumption — same estimator, same per-item
input-blind 0/1 null, concatenated paired differences.

| arm | 5-only n=7107 | | | +MMLU-Pro n=19139 | | |
|---|---|---|---|---|---|---|
| | Δ pp | hw | p | Δ pp | **hw** | p |
| base | +45.701 | 1.534 | 1e-4 `+` | +26.731 | **0.823** | 1e-4 `+` |
| shortgpt16 | +30.941 | 1.569 | 1e-4 `+` | +13.799 | **0.758** | 1e-4 `+` |
| keep14 | +9.849 | 1.625 | 1e-4 `+` | +3.861 | **0.734** | 1e-4 `+` |
| keep12 | −0.338 | 1.161 | 0.5624 `=` | −0.345 | **0.643** | 0.2968 `=` |
| **keep10** | −1.393 | 1.393 | 0.0541 `=` | **−0.779** | **0.682** | **0.0262 `−`** |
| keep8 | −0.084 | 1.287 | 0.9032 `=` | −0.104 | **0.596** | 0.7381 `=` |

Two things to report:

1. **Half-widths drop from 1.16–1.63 pp to 0.60–0.82 pp** — every pooled cell is
   now powered, where 3 of 6 of the 5-only cells were not.
2. **keep10 crosses into `BELOW the floor`** (p 0.0541 → 0.0262). That is the
   *only* significant below-floor pooled verdict, and it is driven by added
   precision, not by a larger effect (|Δ| actually *shrinks* 1.393 → 0.779 pp
   because MMLU-Pro's own Δ is small). **Be careful with this one**: a p-value
   moving across 0.05 on a shrinking point estimate is a power artefact, and
   keep10 was already the borderline cell in #250's pooled table too.

⚠️ **A pooled verdict must NEVER be quoted per-benchmark.** The pooled floor
`0.1917` mixes six floors spanning `0.1166` (mmlu_pro) to `0.5049` (piqa). What
is pooled is the **per-item deviation from each benchmark's own floor**, not raw
accuracy. Note the pooled floor *moved* from `0.3187` to `0.1917` purely because
MMLU-Pro contributes 12032 low-floor items — a vivid demonstration of why the
pooled number is not a benchmark property. Winogrande is excluded (control).

---

## 5. Shard integrity, and a REAL failure the guards caught

* **6/6 cells**: 8/8 per-example shards + 8/8 shard-meta json, **independently
  recounted** from the shard files (not trusting `summary_*.json`):
  **12032 records, 12032 unique item_ids, `n_nan = 0`, `n_trunc = 0`** on every
  arm. `max_len=1536` was chosen from a measured max letter-prompt length of
  **1226 tokens**, so zero truncation is expected and confirmed — important,
  because left-truncating the labelled option body would change the letter
  *interface itself*. ⚠️ **That 1226 is the OLMo-2 tokenizer's number and does
  NOT transfer**: Llama-2 encodes the same items up to **1678** tokens and Qwen3
  up to **1660**, so the cross-family leg needed `max_len=2048`. The first
  cross-family launch used 1536 and truncated 10 of 15 cells — see §6g.
* Failure grep with **failure syntax**
  (`Traceback \(most recent call last\)|CUDA out of memory|SHARD INTEGRITY
  FAILURE|CARDINALITY FAILURE|AssertionError|loss=nan`) over all 56 logs of the
  final run: **zero hits**.
* **The first launch FAILED and was correctly refused.** At `batch_size=48` the
  original harness OOM'd on 5 of 8 GPUs (`Tried to allocate 18.81-22.00 GiB`),
  and the driver reported `SHARD FAIL 7B_base: 3/8 shard json -- NOT merging`.
  **No partial merge happened**, which is exactly what the guard exists for. Root
  cause and fix in §5a. The partial results were deleted and the whole run
  redone from scratch.

### 5a. Root cause of the OOM, and why the fix changes no numbers

`score_examples` did `torch.log_softmax(out.logits.float(), dim=-1)` over the
**whole** `[B, L, V]` batch. With `V = 100352` that is `4·B·L·V` bytes of fp32 —
**~19-24 GiB** at `B=48` with MMLU-Pro's long labelled bodies. The 4-way tasks
survived it only because their prompts are ~5× shorter.

Only the teacher-forced continuation positions are ever read, so the fixed path
**gathers those positions first** and casts/normalises only them, in chunks
(`--lp_chunk`, default 512). Peak fp32 buffer is now `4·lp_chunk·V` ≈ **205 MiB**
instead of `4·B·L·V`. Measured effect: **peak memory 83.1 → ~49-53 GiB per GPU**,
and the run completed with zero OOM.

**Numerically this is a no-op, and that is a TEST, not a comment.** `log_softmax`
reduces over the vocab dimension independently per position, so gathering first
is the same arithmetic on the same fp32 inputs. `--selftest` now keeps the old
whole-tensor path as `_score_examples_wholetensor_REFERENCE` and asserts the new
path is **bit-identical** to it on every candidate score, at two chunk sizes
(4096 and 3, to cover the boundary logic): **44/44 scores exactly equal, max
|Δ| = 0.0**. This matters because #248/#250 numbers were produced by the old
path; had the fix moved bits, the MMLU-Pro cells would not be comparable with them.

> ⚠️ Methodological note worth keeping: the first version of that assert compared
> the **records** rather than the raw scores and reported a spurious ~5e-7
> disagreement. The records store `_safe_lp`-rounded values (6 dp), so a
> records-level comparison measures **JSON rounding**, not arithmetic. It would
> equally have **masked** a real 1e-7 change. Compare unrounded values when
> asserting bit-identity.

Two smaller driver bugs found and fixed: `bc` is **not installed** on the H20
nodes, so the `n_trunc` roll-up printed a cosmetic `?` (now `awk`, and a nonzero
`n_trunc` is now a hard failure); and the harness's `Counter` import was
function-local.

> ⚠️ **The `bc` bug and the `n_trunc` guard's weakness compounded.** Because the
> roll-up printed `?`, the OLMo-2 driver never actually *displayed* a truncation
> count — the zeros above were recovered afterwards by recounting the shard json.
> And because a nonzero count was only a **WARNING** at the time, the
> cross-family launch printed real nonzero counts and still wrote summaries
> (§6g). A cosmetic bug in a guard and a too-weak guard are the same class of
> problem: **the check must fail the run, not narrate it.** `n_trunc` is now a
> hard per-shard assert inside the scoring script, alongside `n_scored`/`n_nan`.

---

## 6. Cross-family extension — LANDED 2026-08-12 (task #252)

The same harness ran on #250's 15 non-OLMo arms (Llama-2-7B / Llama-3-8B /
Qwen3-8B-Base × {intact, k14, k12, k10, k8}; damage is **eval-time front-N
truncation, no fresh block, no heal**, so no training was needed).

> ⚠️ **The first launch of this run shipped two real integrity defects and its
> numbers must not be quoted.** They were root-caused and fixed in #252; the run
> was redone in full into `mmlu_pro_lc_crossfamily_results_fix/`. See §6e. The
> statistics below are all from the **redone** run.

⚠️ Uses `../models/Qwen3-8B-**Base**`. zwfy6's `models/Qwen--Qwen3-8b` and the
`Qwen3-8b-local` symlink are Qwen3-8B-**Instruct** (`eos 151645` = `<|im_end|>`,
ctx 40960) and are **not** valid base arms under `chat_template=False`. The
criterion is **eos id + ctx length**, not the presence of a `chat_template`
(both have one).

### 6a. All 21 cells, letter vs the best-constant floor

Letter null `always-A 0.116606` is **bit-identical across all 21 cells** (asserted,
not assumed: the script aborts if it drifts, because a drift would mean the item
sets differ). `+` above floor, `=` AT floor, `−` BELOW floor.

| family | arm | letter acc | Δ vs floor | CI95 | hw | boot p | modal | n_letters | tie | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| Llama-2-7B | base | `0.131981` | **+1.538 pp** | [+0.765, +2.311] | 0.773 | 1e-4 | D 0.494 | 10 | 0.165 | `+` |
| | k14 | `0.116772` | +0.017 | [−0.066, +0.100] | **0.083** | 0.707 | A 0.991 | 4 | 0.001 | `=` |
| | k12 | `0.115858` | −0.075 | [−0.158, +0.008] | **0.083** | 0.069 | A 0.988 | 4 | 0.000 | `=` |
| | k10 | `0.113198` | −0.341 | [−0.756, +0.075] | 0.416 | 0.113 | A 0.798 | 5 | 0.003 | `=` |
| | **k8** | `0.107463` | **−0.914 pp** | **[−1.646, −0.175]** | 0.736 | **0.0168** | C 0.291 | 6 | 0.001 | **`−`** |
| Llama-3-8B | base | `0.329205` | **+21.260 pp** | [+20.312, +22.207] | 0.947 | 1e-4 | A 0.297 | 10 | 0.111 | `+` |
| | k14 | `0.114694` | −0.191 | [−1.031, +0.673] | 0.852 | 0.663 | D 0.627 | 8 | 0.003 | `=` |
| | k12 | `0.111868` | −0.474 | [−1.322, +0.382] | 0.852 | 0.285 | G 0.339 | 8 | 0.003 | `=` |
| | k10 | `0.111951` | −0.465 | [−1.288, +0.366] | 0.827 | 0.289 | D 0.758 | 7 | 0.001 | `=` |
| | k8 | `0.113531` | −0.308 | [−0.723, +0.091] | 0.407 | 0.130 | A 0.775 | 4 | 0.002 | `=` |
| Qwen3-8B-Base | base | `0.461104` | **+34.450 pp** | [+33.485, +35.422] | 0.968 | 1e-4 | A 0.263 | 10 | 0.134 | `+` |
| | k14 | `0.118933` | **+0.233 pp** | [+0.042, +0.424] | 0.191 | **0.0192** | A 0.946 | 6 | 0.002 | **`+`** |
| | k12 | `0.115027` | −0.158 | [−0.582, +0.274] | 0.428 | 0.465 | A 0.730 | 4 | 0.007 | `=` |
| | k10 | `0.118185` | +0.158 | [−0.075, +0.391] | 0.233 | 0.187 | A 0.919 | 4 | 0.004 | `=` |
| | **k8** | `0.107796` | **−0.881 pp** | **[−1.695, −0.058]** | 0.819 | **0.0362** | E 0.945 | 5 | 0.001 | **`−`** |

**Power: 21/21 cells powered** (hw `0.083`–`0.968` pp, median `0.727`, all
< 1.389). The wall is down for the cross-family leg too, and the two Llama-2
`k14`/`k12` cells reach hw **0.083 pp** — the finest resolution anywhere in
paperC, ~14× better than MMLU's 1.154.

### 6b. What REPLICATES

1. **The wrong-null flip, 12/12 damaged non-OLMo cells.** Under `mean(1/n_opt)`
   = `0.110877`, **10 of 12** damaged cells read "above chance"; under naive
   `0.10`, **12 of 12** do. Yet only **1 of 12** clears its own floor. Combined
   with the OLMo-2 leg's 3/3, that is **15 damaged cells at MMLU-scale power in
   four families where the null choice alone flips the verdict.** This is
   paperC's core claim and it is now its best-powered result.
2. **`AT-or-below` holds 14/15**, and **9 of 12** damaged cross-family point
   estimates are negative.
3. **Floor arrival is immediate under un-healed damage**, confirming #250: every
   family's k14 is already at (or below) floor, with the single exception noted
   in §6c.

### 6c. What does NOT replicate, and one genuine reversal

* **Two cells ARE significantly below floor** — `llama2_7b/k8` (−0.914 pp,
  p = 0.0168) and `qwen3_8b_base/k8` (−0.881 pp, p = 0.0362). §2b's conclusion
  from the OLMo-2 leg alone ("treat below-floor as **MMLU-specific** until shown
  otherwise") is therefore **too pessimistic and must be corrected**: MMLU-Pro
  does yield significant below-floor verdicts, just not on OLMo-2's *healed*
  arms. The distinguishing factor is plausibly **heal vs no-heal**, not benchmark
  — and it is exactly the confound #250 already flagged as unresolvable without
  healed non-OLMo arms. **State it as heal-confounded, not as a family effect.**
* **`qwen3_8b_base/k14` is significantly ABOVE its floor** (+0.233 pp,
  p = 0.0192) — the only damaged cell in the paper that clears its floor at
  MMLU-scale power. With hw 0.191 pp this is a real, tiny effect, not noise.
  It is a **counterexample to any universal "damage ⇒ at-or-below floor"
  phrasing**: the honest form is "14/15 damaged cells at or below floor".
* **The ladder is still a cliff, not a gradient** (#250's finding, reconfirmed):
  k14→k8 is non-monotone in all three families (Llama-3's most-negative rung is
  k12; Qwen3's k10 is *above* its k12). **Do not fit a depth curve.**

### 6d. Modal/floor decoupling — a THIRD independent confirmation, at both extremes

This is the cleanest instance in the paper:

| cell | modal share | distinct letters | verdict |
|---|---|---|---|
| `llama2_7b/k14` | **99.13%** (A) | 4 | `=` AT floor |
| `llama2_7b/k12` | **98.83%** (A) | 4 | `=` AT floor |
| `qwen3_8b_base/k8` | **94.48%** (E) | 5 | **`−` BELOW floor** |
| `llama2_7b/k8` | **29.12%** (C) | 6 | **`−` BELOW floor** |
| `llama3_8b/k12` | 33.87% (G) | 8 | `=` AT floor |

Two cells at ~99% modal sit **AT** the floor while a 29%-modal cell sits
**BELOW** it. So modal share neither implies nor is implied by the floor verdict,
**in both directions** — the 2026-08-10 narrowing ("damage drives letter to or
below its floor", NOT "damage makes letter constant") is confirmed a third time,
independently of OLMo-2 MMLU and of #250.

Note also `qwen3_8b_base/k8`'s modal letter is **E**, not the best-constant `A`:
a 94% single-letter emitter that picked the *wrong* constant lands **below** the
floor. That is the mechanism by which a below-floor verdict arises here, and it is
**not** the bf16 exact-tie mechanism (tie rate 0.0012) that gate-3 falsified —
consistent with #250's "Llama/Qwen degrade by direct modal collapse".

### 6e. The 10-way content null is TOKENIZER-dependent, but far less than at 4-way

Longest-option floor, continuation-token unit, per tokenizer, on the identical
12032 items:

| tokenizer | `len(tok)` | `split` | `first` | `last` | `credit` | `wrong` | tied-longest frac | mean winner-set |
|---|---|---|---|---|---|---|---|---|
| OLMo-2 | 100278 | `0.193150` | `0.195894` | `0.190824` | `0.532164` | `0.125914` | 56.24% | 4.178 |
| Llama-2 | 32000 | `0.191402` | `0.196310` | `0.187749` | `0.439578` | `0.133311` | 49.72% | 3.317 |
| Llama-3 | 128256 | `0.192813` | `0.195229` | `0.190991` | `0.531998` | `0.125748` | 56.23% | 4.180 |
| Qwen3 | 151669 | `0.196083` | `0.199634` | `0.194648` | `0.455203` | `0.134309` | 51.94% | 3.407 |

> `len(tok)` is the **tokenizer's** vocabulary, which is what determines how the
> option strings segment. It is not always the model config's `vocab_size` (OLMo-2:
> 100278 vs 100352; Qwen3: 151669 vs 151936 — the gap is padding/reserved ids).

* `split` spread across tokenizers: **0.47 pp** (`0.191402`–`0.196083`) — *smaller*
  than #250's 1.50 pp at 4-way, contrary to the §6-prediction that it would be
  larger.
* `credit` spread: **9.26 pp** (`0.439578`–`0.532164`) — comparable to #250's
  10.6 pp. **The `credit` convention remains the dominant under-specification.**
* The vocabulary-size ordering **fails**: the 32k Llama-2 and the 152k Qwen3 tie
  *less* (49.7 / 51.9%) than the 100k OLMo-2 and 128k Llama-3 (both 56.2%).
  #250's "larger BPE vocabularies tie more option lengths" is therefore **not a
  monotone function of vocab size** — retract the mechanism, keep the fact of
  tokenizer dependence. The near-identity of OLMo-2 and Llama-3 (56.24 vs 56.23%,
  winner-set 4.178 vs 4.180) suggests what matters is tokenizer *lineage /
  training corpus*, not size.
* ⚠️ **A content floor must still be quoted with its tokenizer** — the claim
  survives, its explanation does not.

### 6f. content_norm vs letter: the ±3 pp bullet holds here too

All 12 damaged cross-family cells have |content_norm − letter| ≤ **3.97 pp**
(range −3.97 to −0.77 pp), versus arc_easy's +38.76 pp blow-up. Note the **sign**:
unlike OLMo-2's mixed signs, all 12 non-OLMo damaged cells have content_norm
*below* letter. The intact arms are far larger and also negative
(Qwen3 −17.28 pp, Llama-3 −11.31 pp; Llama-2 is the lone positive at +2.09 pp),
which again contradicts "content is the fair interface" as a general statement.

### 6g. The two integrity defects of the first launch, and the fix

**Defect 1 — truncation (10 of 15 cells).** The driver ran at `MAXLEN=1536`, a cap
measured with the **OLMo-2** tokenizer (max letter prompt 1226 tok). Re-encoding
all 12032 items with each tokenizer on CPU
(`paperC/code/mmlu_pro_trunc_audit.py`) reproduces the shipped counts exactly:

| family | `len(tok)` | max encoded tok | `n_trunc` @1536 | affected item_ids |
|---|---|---|---|---|
| OLMo-2 | 100278 | 1226 | **0** | — |
| Llama-3 | 128256 | 1226 | **0** | — |
| Qwen3 | 151669 | 1660 | **20** | 11603, 11790 |
| Llama-2 | 32000 | 1678 | **40** | 10500, 11603, 11790 |

`n_trunc` counts **candidate encodings** (10 per item per interface at 10-way),
not items — which is why it is a multiple of 10, constant across rungs, and
independent of damage. Only **three distinct items** overflow at all. The harness
*left*-truncates, so for those candidates part of the labelled `A. …`/`B. …` body
that the letter interface is defined to read was gone: the thing being scored
changed. Worse, the overflow set is tokenizer-specific, so the cross-family table
was **not item-matched**.

**Fix (a), not (b).** Raise the cap to `MAXLEN=2048` (clears the global max 1678
for all four tokenizers with 370 tok of headroom) and re-run, rather than
excluding the affected items. Excluding would have cost the full-n item match
with the archived MMLU / #248 / #250 cells for the sake of 3 items. Re-run
`n_trunc = 0` on **15/15** cells, independently recounted from the shard json.

**Measured impact — this defect was real but small.** Per-cell before/after
(`paperC/evidence/mmlu_scale_power/trunc_fix_before_after.json`):

* **0 of 14 comparable cells changed verdict.**
* Largest letter-accuracy change: **+0.0083 pp** (`qwen3_8b_base/k12`,
  `0.114943` → `0.115027`, i.e. one item). **13 of 14 cells changed by 0.0000 pp.**
* Total argmax flips: **9**, every one of them on a truncation-affected item;
  **0 flips on unaffected items** across all 14 cells × 12032 items.
* So the a-priori worst case (40/12032 = 0.332%, same order as the 0.1–0.9 pp
  effects) did **not** materialise: the truncated candidates were mostly still
  ranked the same way. **The defect had to be fixed to make the claim sound —
  and fixing it did not change a single conclusion.** Both statements matter.

**Defect 2 — `llama2_7b_base` produced nothing.** It appeared in the failure list
*without* a `:trunc` tag: it **OOM'd on 5 of 8 shards** and the driver correctly
reported `SHARD FAIL 3/8 -- NOT merging`. This is a **second, distinct** OOM from
§5a's whole-tensor `log_softmax`: it is the **KV cache**. Llama-2 has
`num_key_value_heads = 32` (**no GQA**) × 32 layers = **72.0 GiB** of fp32 KV at
B=48/L=1536, versus 18.0 (Llama-3) and 20.2 (Qwen3), both `num_kv_heads = 8`.
That is exactly why **only the intact Llama-2 arm** died while every GQA arm and
every *truncated* Llama-2 rung (fewer layers ⇒ smaller cache) survived.

Fixed by passing **`use_cache=False`**: a single teacher-forced forward writes the
cache and immediately discards it, so it was pure waste. Peak memory on `.73`
went **94 GiB (OOM) → 41–50 GiB**, and the arm now completes with n=12032,
`nan=0`, `n_trunc=0` — giving the intact-Llama-2 cell that was missing entirely:
letter `0.131981`, **+1.538 pp above floor**, p=1e-4. (Note how *low* that is: an
intact Llama-2-7B is only 1.5 pp above a constant predictor on MMLU-Pro, vs
Llama-3's +21.3 and Qwen3's +34.5.)

**The fix changes no numbers, and that is tested, not asserted.** `--selftest`
keeps `_score_examples_wholetensor_REFERENCE` calling the model *without*
`use_cache=False` and asserts the live path is bit-identical to it (44/44
candidate scores, two chunk sizes). On GPU, the `llama3_8b_base` cell — which had
`n_trunc = 0` in **both** runs, so `use_cache=False` is the *only* difference —
reproduces the old run to every printed digit: `letter=0.329205
craw=0.204704 cnorm=0.216090 modal=0.2974 tie=0.1107 McNemar_p=4.452e-100`.

**Hardening.** `n_trunc` was only a driver-level **WARNING**, which is precisely
why 10 truncated cells shipped with summaries written. It is now a **hard
per-shard assert** in `eval_olmo2_mc_letter_content.py` (opt out only via the new
`--allow_truncation`), so an arm that cannot be scored at full prompt length
produces **no summary at all**. `n_scored`/`n_nan` were already hard asserts;
`n_trunc` now joins them.

---

## 7. What must change elsewhere in paperC

1. **`README.md` Open defects item 2** — the "power limit" residual is **closed**
   for the OLMo-2 leg. Replace "52/60 underpowered" framing with: a powered
   benchmark now exists (n=12032, hw 0.582-0.906 pp), and on it the below-floor
   headline **does not reproduce** while the wrong-null flip **does** (3/3).
2. **Narrow the below-floor claim.** "Significantly below its floor" is now
   supported on **MMLU + arc_easy's keep10** — and, per §6, on **MMLU-Pro's
   `llama2_7b/k8` and `qwen3_8b_base/k8`**. What does *not* hold is that it
   reproduces on OLMo-2's **healed** arms at MMLU-scale power: there, all four
   say `AT the floor`. The safe general form remains the one the 2026-08-10
   narrowing adopted — **"to or below"** — now with the added qualifier that
   **14/15**, not 15/15, of the cross-family damaged cells satisfy it
   (`qwen3_8b_base/k14` is significantly *above*). Add MMLU-Pro's OLMo-2 `keep8`
   CI `[−0.698, +0.465]` as a **positive exclusion** of a −1.389 pp effect *on a
   healed arm*.
3. **Un-narrow `confirmed_general[2]` partially** — the ±3 pp interface-swap
   bullet holds on MMLU-Pro (max |Δ| 1.31 pp on damaged arms), so "MMLU only" is
   too tight. Re-scope to "holds where the damaged arm has little residual
   content competence (MMLU, MMLU-Pro); fails where it has a lot (arc_easy
   +38.76 pp)".
4. **Add a fourth under-specification, on the LETTER side**: when `n_opt` varies,
   "chance" is ambiguous (naive `1/max(n_opt)` = `0.10` vs `mean(1/n_opt)` =
   `0.110877`), moving the floor−chance gap from +1.661 pp to +0.573 pp. Print
   which one is meant. paperC had previously asserted the letter floor to be a
   **pure dataset property** — that is still true of the *floor*, but not of the
   *gap to chance*.
5. **Report the floor/chance RATIO alongside the pp gap** whenever benchmarks with
   different option counts are compared. MMLU-Pro is the paper's strongest
   relative case (1.1661×) and a mid-pack absolute one (+1.661 pp).
6. **The `credit` convention reductio gets its best number**: `0.532164` on
   MMLU-Pro, which **beats the intact base model's content_norm `0.207613` by
   32.5 pp**. Use this, with winogrande's `0.9337`, as the pair of examples that
   makes "print your tie convention" non-negotiable.
7. Add the harness memory fix + bit-identity test to the reproducibility notes:
   any re-run of #248/#250 on the new code is numerically identical to the old.
8. **`n_trunc` is now a hard assert, and `max_len` is a per-TOKENIZER quantity**
   (§6g). Any benchmark whose prompts approach the cap must have the cap probed
   per tokenizer before launch (`paperC/code/mmlu_pro_trunc_audit.py`, CPU-only);
   a cap validated on one tokenizer is not validated on another. This is the
   defect that shipped 10 of #251's 15 cross-family cells.
9. **Retract the vocab-size mechanism for content-null tokenizer dependence**
   (§6e). The dependence is real; "larger BPE vocabularies tie more option
   lengths" is **not** monotone in vocab size (32k Llama-2 and 152k Qwen3 both tie
   *less* than 100k OLMo-2 and 128k Llama-3). Keep the fact, drop the explanation.

**Not claimed / not resurrected:** nothing from
`STATUS.json:must_not_resurrect`; no MMLU null of `0.25`; no BoolQ null of
`0.50`; no step-function or phase-transition language; no exact-tie mechanism
claim; no acc/acc_norm length-sensitivity claim (Oostermeijer, ICML 2026,
arXiv:2607.12767); no depth curve fitted to the damaged rungs; no claim that
clearing a floor is sufficient.

---

## 8. Bottom line

* ✅ **The power wall is down.** hw **0.582-0.906 pp** on the OLMo-2 leg and
  **0.083-0.968 pp** across all **21** cells (§6), 21/21 powered, vs MMLU's 1.154
  and openbookqa's 6.400. Median hw implies n≈3186 sufficed; MMLU-Pro's 12032
  clears it **3.8×**. **No smaller benchmark could have done this**
  (obqa would need ~10615 items and has 500).
* ✅ **The wrong-null flip replicates, 3/3 on OLMo-2 and 12/12 cross-family**
  (10/12 read "above chance" under `mean(1/n_opt)`, 12/12 under naive `0.10`,
  while 1/12 clears its floor), in **four families** on a 10-way benchmark under
  **both** chance definitions — paperC's core claim, at the paper's best
  resolution.
* ❌ **MMLU's significant below-floor headline does NOT replicate on OLMo-2's
  healed arms.** keep8 is −0.116 pp, CI `[−0.698, +0.465]`, which **excludes
  −1.389 pp**. This is a *powered* null: an informative negative, not a blind
  spot.
* ⚠️ **But below-floor is NOT MMLU-specific** (§6c): on the same benchmark,
  `llama2_7b/k8` (−0.914 pp, p=0.0168) and `qwen3_8b_base/k8` (−0.881 pp,
  p=0.0362) *are* significantly below floor. The live explanation is **heal vs
  no-heal**, #250's known confound — not benchmark identity. An earlier draft of
  §2b said "MMLU-specific until shown otherwise"; §6 shows otherwise.
* ✅ **`AT-or-below` still satisfies the post-2026-08-10 claim**, at **14/15**
  cross-family damaged cells — with one honest counterexample,
  `qwen3_8b_base/k14`, significantly **above** its floor (+0.233 pp, p=0.0192,
  hw 0.191).
* ✅ **Modal/floor decoupling confirmed a THIRD time and at BOTH extremes** (§6d):
  99.13% and 98.83% modal cells sit **AT** the floor while a **29.12%** modal cell
  sits **BELOW** it.
* ⚠️ **A fourth under-specification found, on the letter side**: with variable
  `n_opt`, "chance" itself is ambiguous (+1.661 vs +0.573 pp).
* ⚠️ **The `credit` content null (`0.532164`) beats the intact base model by
  32.5 pp** — the paper's sharpest reductio for "print your tie convention".
* ✅ **Pooling to n=19139 halves the pooled CI** (hw 0.60-0.82 pp) and flips
  keep10 to significantly-below — but on a *shrinking* point estimate, so report
  it as a power artefact, and never per-benchmark.
* ✅ **The integrity guards earned their keep — twice.** Two genuinely distinct
  OOMs were caught and the partial merges **refused** (§5a whole-tensor
  `log_softmax`; §6g the KV cache on the non-GQA Llama-2), and both fixes are
  asserted **bit-identical** to the path that produced #248/#250.
* ⚠️ **One guard was too weak and it cost a full re-run**: `n_trunc` was a
  warning, not an assert, so 10 of 15 cross-family cells shipped truncated
  (§6g). Fixed at the source. The measured impact turned out to be **0/14
  verdicts and ≤0.0083 pp**, but that was not knowable in advance — the a-priori
  bound (0.332% of items vs 0.1-0.9 pp effects) was the same order as the signal.

The load-bearing part is that this task **replaced five uninterpretable nulls
with one interpretable one, and the interpretable one narrows our own headline.**
The cross-family leg then **partially un-narrowed it** (below-floor is not
MMLU-specific) while adding a fresh counterexample (`qwen3_8b_base/k14` above
floor) — self-falsification in both directions on the same run.

---

## 9. Provenance

| what | where |
|---|---|
| per-item records, 6 arms × 8 shards + merged (**~1.1 GB**) | **zwfy6 ONLY** `<REPO_ROOT>/mmlu_pro_letter_content_results/` (too large for wzc1 round-trip; not git-tracked, same as the MMLU records) |
| nulls + statistics, OLMo-2 leg only (66 rows) | `paperC/evidence/mmlu_scale_power/mmlu_pro_power_nulls.json` + `.csv` — **BOTH DISKS**, md5 `564c2f74…` / `891e7eb9…` |
| **nulls + statistics, ALL 21 CELLS (231 rows) — the canonical table for §6** | `paperC/evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.json` + `.csv` — **BOTH DISKS**, md5 `8a96c6c2…` / `204f910f…` |
| harness (extended for `mmlu_pro` + the memory fix + `use_cache=False` + the hard `n_trunc` assert) | `scripts/eval_olmo2_mc_letter_content.py` |
| driver | `scripts/_run_mmlu_pro_letter_content_8gpu.sh` (`MAXLEN` default now **2048**) |
| analysis (CPU, 0 GPU) | `paperC/code/mmlu_pro_power_nulls.py` |
| **truncation audit (CPU, 0 GPU, all 4 tokenizers)** | `paperC/code/mmlu_pro_trunc_audit.py` → `paperC/evidence/mmlu_scale_power/trunc_audit_1536.json`, md5 `6279a823…` |
| **before/after of the truncation + OOM fixes (per-cell, per-item flips)** | `paperC/code/mmlu_pro_trunc_fix_compare.py` → `paperC/evidence/mmlu_scale_power/trunc_fix_before_after.json`, md5 `2b84371c…` |
| logs (56 files: 48 shard + 6 merge + prepare + DRIVER) | **zwfy6** `logs/mmlu_pro_lc_*.log` |
| dataset | `TIGER-Lab/MMLU-Pro` test parquet, **BOTH DISKS** `data/hf_datasets/TIGER-Lab___mmlu_pro/data/test-00000-of-00001.parquet`, sha256 `0e24a191…`, md5 `7e40550a…` (4.1 MB) |
| ckpts | **zwfy6** `outputs/olmo2_probe2_7B_{keep8fresh2/step121000,keep10fresh2/step83500,keep12fresh2/step124000,keep14fresh2/step200000,shortgpt16/step200000}.pt` (all `ls`-verified before launch) |
| #248 comparison | `paperC/evidence/second_mc_benchmark/gate2_letter_content_nulls.json` |
| #250 comparison | `paperC/evidence/second_mc_benchmark_crossfamily/gate2_crossfamily_nulls.json` |
| **cross-family MMLU-Pro per-item records — USE THIS ONE** | **zwfy6 ONLY** `mmlu_pro_lc_crossfamily_results_fix/` (15 arms × 8 shards, `MAXLEN=2048`, `n_trunc=0`), `logs/mmlu_pro_lc_DRIVER_xf_fix.log` |
| ⛔ cross-family MMLU-Pro, FIRST launch — **defective, do not quote** | **zwfy6** `mmlu_pro_lc_crossfamily_results/` (`MAXLEN=1536`: 10 cells with `n_trunc>0`, `llama2_7b_base` only 3/8 shards), `logs/mmlu_pro_lc_DRIVER_xf.log`. Retained **only** as the BEFORE side of `trunc_fix_before_after.json`. |
| combined root used for the 21-cell table (symlinks, no data copied) | **zwfy6** `mmlu_pro_lc_COMBINED_fix/` → the 6 OLMo-2 dirs + the 15 `_fix` dirs |

Protocol: `chat_template=False`, `add_bos=0`, fp32 master weights + bf16-autocast
forward, `batch_size=48`, `lp_chunk=512`, `use_cache=False`, paired bootstrap
`n_boot=10000` `boot_seed=7`, R-7 mid-p `two_sided_boot_p`, exact McNemar.
`max_len` = **1536** for the OLMo-2 leg (max encoded 1226, `n_trunc=0` verified)
and **2048** for the cross-family leg (global max encoded **1678** over all four
tokenizers, `n_trunc=0` verified). ⚠️ `max_len` is a **per-tokenizer** quantity —
see §6g; the two values do not affect comparability because **both** yield
`n_trunc = 0`, i.e. no prompt is truncated under either.
`batch_size=48` is deliberately #248/#250's value (bf16 batch composition
perturbs low-order bits, so changing it is a protocol difference); measured
bs=96 gives **no** speedup (0.409 vs 0.398 s/item), so the eval is compute-bound
and raising it would buy nothing while costing comparability.
