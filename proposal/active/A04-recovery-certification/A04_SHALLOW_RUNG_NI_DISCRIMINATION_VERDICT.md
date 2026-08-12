# A04 — Can `NI(Δ)` ever be observed to ACCEPT? The shallow-rung 7B test

**Date**: 2026-08-12. **GPU spent: ZERO.** CPU-only re-analysis of per-example shards already on
disk, plus read-only `ssh` and `scp -O` to move 48 wzc1-only shard files onto zwfy6 so the whole
analysis runs under one loader on one disk. **No model was loaded. `.73` sat at 0% throughout and
was used only as a CPU host.**

**Reproduce:**
```bash
# on .73, PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
R=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
/opt/conda/envs/torch-base/bin/python \
  proposal/active/A04-recovery-certification/code/a04_shallow_rung_ni_7b.py \
  --raw_root $R \
  --shortgpt_cb $R/a04_shallow_stage/shortgpt16_step200k \
  --shortgpt_nq $R/a04_shallow_stage/shortgpt16_step200k_nqopen \
  --d5_cb $R/a04_shallow_stage/D5_intact_wzc1_cb \
  --d5_mm $R/a04_shallow_stage/D5_intact_wzc1_mm \
  --out_json proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_ni_7b.json
/opt/conda/envs/torch-base/bin/python \
  proposal/active/A04-recovery-certification/code/a04_shallow_rung_remeasurement_sensitivity.py \
  --raw_root $R --stage_root $R/a04_shallow_stage \
  --out_json proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_remeasurement_sensitivity.json
```

---

## 0. The answer, in four lines

1. **`NI(Δ)` has now been observed to ACCEPT.** Not on a damaged rung — on the **zero-damage
   control** `full32_dolmino_step25k`, MMLU-content, margin **+1.0495 pp**. This is the first
   accept anywhere in A04. It proves the rule is not a constant function.
2. **But no *damaged* rung accepts.** `keep14+fresh2` and `shortgpt16` — the shallowest damage in
   the repository, 200k heal steps each — **reject 3/3 decision axes**, at **16.5–72.4 bootstrap
   SE** past the flip point. The 7B ladder is still constant-REJECT.
3. **The guard retires nothing at 7B under the pre-registered convention.** All four axes are
   `CERTIFIABLE` under `split`. The measurement is not the blocker; the recovery level is.
4. **The decisive new fact is a rule *disagreement on the zero-damage control*.** `RATIO(ρ=0.85)`
   **accepts** `full32` (mean ratio 0.8515) while `NI` **rejects** it on 2 of 3 decision axes.
   That is A04's target disagreement — and it appears on the one arm where "recovered" is a
   defensible description. **`RATIO`'s accept, however, is fragile to 0.09 pp of harness jitter**
   (§6.2), which is itself a finding about `RATIO`, not about `NI`.

---

## 1. The intact anchor — the choice, and why

`Δ_x = 0.10 · residual(intact, x)`, so **the anchor *is* the margin**. Get it wrong and every
number below is meaningless.

### Chosen: vanilla `models/OLMo-2-1124-7B`

| axis | anchor directory | disk | meta |
|---|---|---|---|
| TriviaQA EM, PopQA EM | `olmo2_closedbook_results/base_full` | zwfy6 | `mode=base, num_hidden_layers=32, base_model=../models/OLMo-2-1124-7B, add_bos=false` |
| NQ-open EM | `olmo2_closedbook_results/base_full_nqopen` | zwfy6 | same |
| MMLU-content | `olmo2_mmlu_content_results/7B_base` | zwfy6 | `mode=base, num_hidden_layers=32, base_model=../models/OLMo-2-1124-7B, add_bos=false, content_desc=full` |

All 8/8 shards, exact item counts (17,944 / 14,267 / 3,610 / 14,042), `n_nan=0`, `add_bos=false`,
no chat template.

**Why this and not `full32_step25000`.** The guard's **G0** pinned the 1B anchor to `A03_1B_base`,
whose meta is `mode=base, num_hidden_layers=16, base_model=../models/OLMo-2-0425-1B` — i.e. the
**vanilla base model**. The 7B analogue of "vanilla base" is `base_full` / `7B_base`. By contrast:

```
full32_step25000 meta: mode=pruned, keep_front_layers=32, n_fresh_layers=0,
                       ckpt=outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt
```

`full32` is an **undamaged 32-layer model that has been continued-pretrained on the heal corpus for
25k steps**. It is *not* the intact target; it is what the intact model *becomes* under the heal
recipe. **They are measurably different, and in the direction that matters**: `full32` scores
**below** vanilla base on all four axes (TriviaQA 57.15% vs 63.55%, PopQA 18.42% vs 24.75%,
MMLU-content 46.62% vs 47.06%, NQ-open 15.82% vs 20.50%). Substituting it would have shrunk every
Δ **and** moved the comparison target down by up to 6.4 pp — manufacturing accepts. Guard **G2**
forbids changing the anchor, so `full32` is carried as an **arm** (a zero-damage control) and
never as the anchor. It cannot alter a verdict.

**This is not a cosmetic distinction.** It is the difference between "did the damaged model come
back to the model we broke" and "did it come back to whatever the heal corpus drifts towards".
A04's claim is the former.

### D5 — is the anchor unique? Measured, and it does not bite

A second admissible measurement of the same vanilla 7B exists on wzc1 (`7B_full32_base_wzc1`,
scored 2026-08-08 — an unfortunate directory name for a `mode=base` run; its meta is
`mode=base, num_hidden_layers=32`, i.e. vanilla, *not* the `full32_dolmino` checkpoint).

| axis | item flips | residual drift | `Δ` drift | drift / `Δ` | D5 fires? (≥0.10·Δ) |
|---|---:|---:|---:|---:|:--:|
| TriviaQA | 56 / 17,944 | −0.0446 pp | −0.0045 pp | 0.00070 | no |
| PopQA | 27 / 14,267 | +0.0491 pp | +0.0049 pp | 0.00218 | no |
| MMLU-content | 30 / 14,042 | −0.0285 pp | −0.0028 pp | 0.00153 | no |

Nulls are **bit-identical** between the two dumps (asserted in code, `<1e-12`), as they must be —
the null is a dataset/tokenizer property. D5 does not fire; the anchor is effectively unique to
within 0.2% of `Δ`.

---

## 2. Guard D1–D6, evaluated BEFORE `NI` (guard G1)

Pre-registered `split` convention. `p*_crit` uses the **frozen formula** of G3,
`p*_crit = n·(Δ_x/(100·1.959964))²`; its inputs `Δ_x` and `n` are per-scale, so the *numeric* value
is recomputed at 7B and both the 7B and the 1B pre-computed values are reported. **No threshold was
adjusted.**

| axis | residual(intact) | CI95 | `Δ` | n | `p*_crit` (7B) | `p*_crit` (1B prereg) | `p_disc` max | `Δ`/hw worst | class |
|---|---:|---|---:|---:|---:|---:|---:|---:|:--|
| TriviaQA EM | **+63.2914 pp** | [62.5836, 64.0047] | 6.3291 | 17,944 | 18.7116 | 7.6359 | 0.3837 | 6.98 | **CERTIFIABLE** |
| PopQA EM | **+22.4574 pp** | [21.7635, 23.1583] | 2.2457 | 14,267 | 1.8731 | 0.6476 | 0.1896 | 3.14 | **CERTIFIABLE** |
| MMLU-content | **+18.6138 pp** | [17.6702, 19.5812] | 1.8614 | 14,042 | 1.2665 | 0.3832 | 0.1885 | 2.59 | **CERTIFIABLE** |
| NQ-open EM | +19.9446 pp | [18.6427, 21.3296] | 1.9945 | 3,610 | 0.3738 | 0.0883 | 0.1795 | 1.44 | **CERTIFIABLE** *(still demoted by design §5.2)* |

All four two-sided paired-bootstrap CIs (10,000 resamples, A03's `paired_bootstrap`) exclude 0 at
`boot_p = 1e-4`, which is what makes **D3** inapplicable.

* **D1** no: every residual strongly positive. **D2** no: minimum 18.61 pp, vs the 1.0 pp floor.
  **D3** no: no CI straddles 0. **D4** no: no arm is below its null; the most frequent single
  output covers at most **29.60%** of items (MMLU-content `keep14`; QA axes ≤3.38%) against the
  >99% trigger, and the MMLU longest-option tie rate is **34.22%** against the ≥99% trigger.
  **D6** no: `p_disc` max 0.1885–0.3837 is well under every 7B `p*_crit`.
* **Decision family: 9 cells full → 9 after the guard. Nothing retired.** No reduced family size to
  declare under `split`.
* **Notable and favourable**: at 7B the residuals are **1.57–2.06× larger** than at 1B (TriviaQA
  63.29 vs 40.43 pp; MMLU-content 18.61 vs 10.24 pp), so every `Δ` is correspondingly wider and
  every axis has *more* headroom than at 1B. **MMLU-content's 1B `NEEDS_RECHECK_AFTER_DATA` status
  resolves cleanly at 7B**: `p_disc` 0.1885 vs `p*_crit` 1.2665 — a 6.7× margin, versus 1B's 79%-of-
  critical squeeze. **NQ-open, retired at 1B by D6, is `CERTIFIABLE` at 7B** (`p_disc` 0.1795 vs
  `p*_crit` 0.3738). It remains demoted because design §5.2 demoted it by *design*, not by the
  guard, and the guard cannot promote a design-demoted axis.

### Convention sensitivity (guard G5)

| convention | MMLU residual(intact) | `Δ` | `Δ`/hw | class |
|---|---:|---:|---:|:--|
| **split** ★ | +18.6138 | 1.8614 | 2.59 | CERTIFIABLE |
| first | +18.9503 | 1.8950 | 2.64 | CERTIFIABLE |
| last | +18.8435 | 1.8843 | 2.62 | CERTIFIABLE |
| **credit** | **+1.6878** | **0.1688** | **0.24** | **NOT_CERTIFIABLE — D6** |
| wrong | +27.4462 | 2.7446 | 3.82 | CERTIFIABLE |

**`credit` at 7B behaves exactly as the prereg predicted it would** (§2.2's "where D2 nearly
bites": residual +1.6878 pp, `Δ` 0.1688 pp, 6.2× finer than the half-width). It fires **D6**, not
D1 — at 7B the `credit` residual is *positive*, unlike 1B where it was −6.69 pp and fired D1+D4.
Under `credit` the decision family shrinks **9 → 6** (all three MMLU cells retired). **No verdict
below depends on this**: `credit` retires cells, and a retired cell is never reported as a reject.

---

## 3. The `NI` table — all four axes, both shallow rungs, plus the zero-damage control

Pre-registered `split`. `residual = reported − null`; the null cancels exactly in
`residual(arm) − residual(intact)`, so the *difference* is convention-invariant and only `Δ` moves
with the convention. `margin = lo95 + Δ`; **`NI` accepts iff `margin > 0`**.

Nulls (best-constant / longest-option `split`, computed on the anchor's own item set):
TriviaQA 0.2564%, PopQA 2.2920%, MMLU-content 28.4450%, NQ-open 0.5540%. Intact reported:
TriviaQA 63.5477%, PopQA 24.7494%, MMLU-content 47.0588%, NQ-open 20.4986%.

| arm | axis | reported | recovered | deficit | lo95 one-sided | `Δ` | **margin** | **`NI`** | SE to flip |
|---|---|---:|---:|---:|---:|---:|---:|:--:|---:|
| **keep14+fresh2** @200k | TriviaQA | 29.403% | 46.1% | 34.1451 | −34.7916 | 6.3291 | **−28.4624** | **REJECT** | 72.4 |
| | PopQA | 7.976% | 25.3% | 16.7730 | −17.3267 | 2.2457 | **−15.0810** | **REJECT** | 44.8 |
| | MMLU-content | 38.321% | 53.1% | 8.7381 | −9.3363 | 1.8614 | **−7.4749** | **REJECT** | 20.6 |
| | *NQ-open (demoted)* | 5.983% | 27.2% | 14.5152 | −15.5679 | 1.9945 | −13.5734 | REJECT | 21.2 |
| **shortgpt16** @200k | TriviaQA | 33.008% | 51.7% | 30.5395 | −31.1803 | 6.3291 | **−24.8512** | **REJECT** | 63.8 |
| | PopQA | 7.269% | 22.2% | 17.4809 | −18.0276 | 2.2457 | **−15.7819** | **REJECT** | 47.5 |
| | MMLU-content | 40.115% | 62.7% | 6.9435 | −7.5061 | 1.8614 | **−5.6447** | **REJECT** | 16.5 |
| | *NQ-open (demoted)* | 6.676% | 30.7% | 13.8227 | −14.9030 | 1.9945 | −12.9086 | REJECT | 19.7 |
| **full32_dolmino** @25k *(zero damage)* | TriviaQA | 57.150% | 89.9% | 6.3977 | −6.9327 | 6.3291 | **−0.6035** | **REJECT** | **1.86** |
| | PopQA | 18.420% | 71.8% | 6.3293 | −6.7849 | 2.2457 | **−4.5391** | **REJECT** | 16.4 |
| | MMLU-content | 46.624% | **97.7%** | 0.4344 | −0.8119 | 1.8614 | **+1.0495** | **★ ACCEPT** | 4.57 |
| | *NQ-open (demoted)* | 15.817% | 76.5% | 4.6814 | −5.6524 | 1.9945 | −3.6579 | REJECT | 6.20 |

Verdict per arm, `split`, threshold `≥2 of 3` decision axes (no rescaling needed — nothing
retired):

| arm | decision axes surviving | accepting | verdict |
|---|---:|---|:--|
| keep14+fresh2 @200k | 3 | **0** | ALL REJECT |
| shortgpt16 @200k | 3 | **0** | ALL REJECT |
| full32_dolmino @25k | 3 | **1** (`mmlu_content`) | 1 accept, below the 2/3 bar |

Identical under `first`, `last`, `wrong`. Under `credit`, MMLU is retired so `full32` shows 0/2.

---

## 4. The plain answer to the question

> **Is there a rung where `NI` ACCEPTS on ≥2 of 3 decision axes?**

**No. But the premise behind the blocker — "`NI` has never been observed to accept, so it has not
been shown to discriminate" — is now FALSE.**

Three separate statements, all of which need to be said:

### 4.1 `NI` is not a constant function. It accepts. (Blocker partially cleared)

`full32_dolmino_step25k` / MMLU-content: **margin +1.0495 pp, ACCEPT**. First accept in A04's
history. The rule can return both values on real data from this harness, so "a rule that only ever
rejects" no longer describes `NI`. Recovery there is **97.7%** of the intact residual — the accept
is not a fluke of a wide `Δ`; the deficit is genuinely small (0.4344 pp against `Δ` = 1.8614 pp).

**What this does *not* license.** It is an accept on an arm with **zero structural damage**. It
demonstrates *sensitivity* (the rule accepts when the model really is nearly intact), not that the
rule can accept a *damaged-then-healed* model. A04's claim is about damaged models. So this clears
"the rule is a constant" but **does not** clear "the rule can discriminate among damaged rungs".

### 4.2 No damaged rung in our reach is certifiable, and the 7B ladder is worse than 1B keep12

| rung | scale | depth kept | heal steps | best decision-axis recovery | `NI` |
|---|---|---:|---:|---:|:--|
| keep7+fresh2 | 1B | 9/16 = 56.3% | 220k | 36.6% (MMLU) | 4/4 REJECT |
| keep12+fresh2 | 1B | 14/16 = 87.5% | 5k | 31.8% (MMLU) | 4/4 REJECT |
| **keep14+fresh2** | **7B** | **16/32 = 50.0%** | **200k** | **53.1% (MMLU)** | **3/3 REJECT** |
| **shortgpt16** | **7B** | **16/32 = 50.0%** | **200k** | **62.7% (MMLU)** | **3/3 REJECT** |
| full32_dolmino | 7B | 32/32 = 100% | 25k | 97.7% (MMLU) | 1/3 ACCEPT |

The 7B rungs recover **substantially more** than any 1B rung (53–63% of the intact residual on
MMLU-content vs 31.8–36.6%) — and still reject on every decision axis, by **16.5–72.4 bootstrap
SE**. There is no realisable perturbation that flips them.

**A candid correction to the dispatch's framing**: these 7B rungs keep **50%** of depth, which is
*less* depth-fraction than 1B `keep12`'s 87.5%. They are "shallower damage" in absolute layer count
(16 layers survive, vs 14 at 1B) and are far better healed (200k steps at 7B), and they do recover
more — but they are not a depth-fraction interpolation between keep12 and intact. **The honest
statement is: across 50%–87.5% depth kept, at two scales, with heal budgets from 5k to 220k steps,
every damaged rung is constant-REJECT.** The gap between the best damaged rung (62.7% recovered)
and the accept threshold is not marginal.

### 4.3 The scientifically load-bearing result: the two rules disagree on the zero-damage control

This is the disagreement A04 exists to find, and it is **not** on a constant-REJECT arm:

| arm | `RATIO(ρ=0.85)` | `NI` (3 decision axes) | agree? |
|---|:--|:--|:--|
| keep14+fresh2 @200k | REJECT (0.4728) | REJECT 3/3 | agree |
| shortgpt16 @200k | REJECT (0.4978) | REJECT 3/3 | agree |
| **full32_dolmino @25k** | **ACCEPT (0.8515)** | **REJECT on TriviaQA + PopQA**, accept on MMLU | **DISAGREE** |

`RATIO` calls `full32` recovered. `NI` says its TriviaQA residual is **6.40 pp** short of intact
(89.9% recovered) and its PopQA residual **6.33 pp** short (71.8% recovered) — deficits **1.01×
and 2.82× `Δ`** respectively. **A retention-style headline of 85% conceals a 6.3 pp absolute
capability deficit on two of three axes.** That is a real, quantified, publishable disagreement,
and it is on the *only* arm where "recovered" is a defensible word — precisely escaping the "the
disagreement is automatic because the arm is simply bad" defect that killed the keep7 and keep12
rungs.

**Caveat that must travel with it**: `full32` is a *zero-damage* control, so this is a disagreement
about **continued-pretraining drift**, not about **recovery from structural injury**. It is
evidence that `RATIO` is too permissive, not yet evidence about the healing of damaged models.

---

## 5. Cross-scale caveat (mandatory)

**`sd_run` is a 1B quantity and is NOT imported as a 7B variance model.**

* A04's `sd_run` comes from `evidence/pilot_one_stage_b_s3_verdict.json`: **1B, `keep12+fresh2`,
  S=3 seeds (101/102/103), 5,000 steps, df=2** — TriviaQA 0.3023 pp, PopQA 0.3328 pp,
  MMLU-content 0.0783 pp, NQ-open 0.2091 pp.
* **Every 7B rung here has exactly ONE seed.** No 7B `sd_run` is computable. Worse, the historical
  7B ladder's seeds are *unrecorded* (`STATUS.json:warning.seed_unrecorded`: `--seed` postdates
  those runs; trainer `afdfa66` called no seeding function at all), so even a retrospective
  reconstruction is impossible.

**What is licensed**: reporting deficits, `Δ`, `lo95` bounds and margins at 7B; and the statement
that no realisable perturbation of the *item sample* flips these verdicts (that is the bootstrap
SE, measured at 7B — 16.5–72.4 SE to flip).

**What is NOT licensed**: any claim that the 7B deficits are large "relative to seed variance". The
deficit/`sd_run` ratios (`sensitivity.cross_scale_sd_run_1B_imported`: 5.5–112.9) are computed with
a **1B** `sd_run` on a **7B** deficit and are labelled in the JSON as a cross-scale extrapolation.
They are directionally reassuring — the 1B `sd_run` would have to grow **3.35–16.88 pp**
(11–215× its 1B value) to flip any shallow-rung reject — but they are **not** a 7B variance
statement, and this document does not make one.

**Also not licensed**: treating the 7B ladder as a controlled scaling law. `STATUS.json:warning`
still holds — the historical keepN ladder spans two corpora (2.0462× row ratio) and unequal step
counts. Here that is largely defused because the two shallow rungs share the same step count
(200k) and the *same* damage fraction (16/32), so they are comparable **to each other**; but
neither is comparable to the 1B rungs as a matched experiment.

---

## 6. Sensitivity — every verdict, not just point estimates

### 6.1 Re-measurement: same checkpoint, different scoring run

Every checkpoint here was scored **more than once** (different day, different disk, in one case a
different harness commit). `NI` was recomputed under **every** (anchor measurement × arm
measurement) combination — 1 to 6 variants per cell, 46 recomputations.

| quantity | value |
|---|---|
| max accuracy spread, same ckpt | **0.1353 pp** (`keep14`/MMLU-content, 65 item flips) |
| max item flips, same ckpt | 132 (`shortgpt16`/TriviaQA, 17,944 items) |
| max `NI` margin spread across variants | **0.1609 pp** (`keep14`/MMLU-content) |
| **cells whose `NI` verdict flips** | **0 of 12** |
| **`full32`/MMLU-content accept margin range** | **[+1.0567, +1.0680]** — stable ACCEPT |

**Every `NI` verdict, including the load-bearing accept, is stable under re-measurement.** Note
this jitter is a *third* distinct quantity: bootstrap SE is item-set sampling within one scoring
run; `sd_run` is seed variance (1B only); this is scoring-run variability on a fixed checkpoint.
They are not interchangeable and are reported separately.

### 6.2 The one fragile verdict is `RATIO`'s, not `NI`'s

`RATIO(ρ=0.85)` accepts `full32` at mean ratio **0.851495** — a margin of **+0.001495** over ρ.
Under all 8 anchor-measurement combinations the ratio ranges **[0.851127, 0.851803]**, so it does
not flip on anchor choice alone. But the accuracy drop needed to flip it is:

| axis | drop in `full32` accuracy that flips `RATIO` to REJECT |
|---|---:|
| NQ-open | **0.0924 pp** |
| PopQA | **0.1116 pp** |
| MMLU-content | 0.2121 pp |
| TriviaQA | 0.2865 pp |

**Observed harness jitter on this very ladder reaches 0.1353 pp** (§6.1). So a re-scoring of
`full32` on NQ-open or PopQA, within measured harness noise, **would flip `RATIO` from ACCEPT to
REJECT** and destroy the §4.3 disagreement. `full32` itself has only **one** scoring per axis, so
this cannot currently be checked — it is the single cheapest thing that would firm up the one
genuinely interesting result here.

**Stated plainly, as the dispatch requires**: §4.3's disagreement rests on a `RATIO` accept whose
margin (0.0015) is smaller than this harness's demonstrated measurement jitter. The `NI` side of
the disagreement is rock-solid (TriviaQA margin −0.6035 pp at 1.86 SE, PopQA −4.5391 pp at 16.4
SE); **it is `RATIO`'s accept that is one re-scoring away from vanishing.** Any writeup must carry
this, exactly as the K2 verdict must carry its σ-CI fragility.

### 6.3 The near-miss worth recording

`full32`/TriviaQA is the closest reject anywhere: margin **−0.6035 pp**, only **1.86 bootstrap SE**
from accepting, deficit 6.3977 pp against `Δ` = 6.3291 pp (`Δ`/deficit = 0.989). **A 1.1% smaller
deficit and `NI` would have accepted 2 of 3 axes and cleared the promotion bar outright.** This is
the sharpest evidence that `NI`'s margin at 7B is *calibrated at roughly the right scale* rather
than being trivially strict — the rule is operating near its decision boundary on a nearly-intact
model, which is exactly where a non-inferiority test should be informative.

---

## 7. What this means for A04, stated without dressing up

**Which of the dispatch's three outcomes occurred: a mixture of the first and second, and the
mixture is the finding.**

* **`NI` accepts somewhere** (outcome 1) — on the zero-damage control, stably, at 97.7% recovery.
  So the rule discriminates in the minimal sense of returning both values, and its margin is
  calibrated near the right scale (§6.3). **The "a rule that only ever rejects" blocker is
  answered.**
* **`NI` rejects on every damaged rung** (outcome 2) — at 50% depth kept and 200k heal steps at 7B,
  the best-healed damaged model in the repository, by 16.5–72.4 SE. **No damaged rung within reach
  is certifiable.**
* **Outcome 3 (NOT_CERTIFIABLE dominates) did not occur** under the pre-registered convention: 9/9
  decision cells survive, and 7B is *materially better instrumented* than 1B (residuals 1.57–2.06×
  larger; MMLU-content's 1B recheck flag resolves with 6.7× margin; even design-demoted NQ-open
  would pass the guard at 7B).

**Consequence for Pilot Two.** The honest reading is **not** "fund it" and **not** "the gate cannot
discriminate". It is:

> The disagreement A04 wants exists and has now been exhibited (`RATIO` accepts what `NI` rejects,
> §4.3) — but it was found on a **zero-damage control**, for **0 GPU-h**, and the `RATIO` side of
> it is fragile to 0.09 pp of harness jitter (§6.2). Meanwhile the *damaged* rungs A04 proposed to
> train are, at every depth and budget measured across two scales, constant-REJECT — which is the
> exact defect Pilot One's Stage B already identified and which 1,077–4,309 GPU-h of new training
> at a *new* depth has no measured reason to escape.

**Therefore the cheap next steps strictly dominate the expensive one**, and none of them is a new
training run:

1. **Re-score `full32_dolmino_step25k` on NQ-open and PopQA** (eval only, hours not days) to
   settle whether §4.3's disagreement survives (§6.2). This is the highest-value GPU in A04 right
   now and it is an *eval*, not a training tranche.
2. **Score the intermediate 7B checkpoints that already exist** on the `keep14fresh2` and
   `full32_dolmino` trajectories. `full32` sits at 97.7% MMLU recovery with one axis 1.86 SE from
   accepting; **the accept boundary is somewhere on that trajectory**, and finding it is what
   "the gate discriminates" would actually mean. Zero training.
3. **Only if (1)–(2) locate a genuine accept/reject boundary** does a new damaged-arm tranche have
   a defensible prior of landing near it. Absent that, funding Pilot Two buys another
   constant-REJECT rung.

**K1/K2/K3 disposition.** None of the three kill clauses fires on this evidence, and this document
does not claim any of them does — the 7B ladder is not the gate's pre-registered 1B arm set, so it
**cannot** fire a clause that is defined over those arms. What it does establish is a fact the
clauses were not written to see: **`NI`'s discriminating power is demonstrable, but only above the
damage level any of our arms has ever recovered to.**

---

## 8. Provenance

| item | path |
|---|---|
| main analysis code | `proposal/active/A04-recovery-certification/code/a04_shallow_rung_ni_7b.py` |
| re-measurement sensitivity code | `proposal/active/A04-recovery-certification/code/a04_shallow_rung_remeasurement_sensitivity.py` |
| main results | `proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_ni_7b.json` |
| sensitivity results | `proposal/active/A04-recovery-certification/evidence/a04_shallow_rung_remeasurement_sensitivity.json` |
| anchor (CB) | `zwfy6:olmo2_closedbook_results/base_full`, `..._nqopen` |
| anchor (MMLU) | `zwfy6:olmo2_mmlu_content_results/7B_base` |
| keep14+fresh2 | `zwfy6:olmo2_closedbook_results/keep14_step200k{,_nqopen}`, `zwfy6:olmo2_mmlu_content_results/7B_keep14_step200000` |
| shortgpt16 | `wzc1:olmo2_closedbook_results/shortgpt16_step200k{,_nqopen}` → staged to `zwfy6:a04_shallow_stage/` (md5-verified, 24/24 files), `zwfy6:olmo2_mmlu_content_results/7B_shortgpt16_step200000` |
| full32_dolmino | `zwfy6:olmo2_closedbook_results/full32_step25000{,_nqopen}`, `zwfy6:olmo2_mmlu_content_results/7B_full32_step25000` |
| D5 second anchor measurement | `wzc1:{olmo2_closedbook_results,olmo2_mmlu_content_results}/7B_full32_base_wzc1` → staged (md5-verified, 24/24) |

**Two-disk note.** `shortgpt16_step200k` exists on **both** disks but is **8/8-sharded only on
wzc1**; the zwfy6 copy holds merged `per_example_*.jsonl` with **no shards at all**, which the
canonical loader correctly refuses. Running this analysis only on zwfy6 without checking wzc1 would
have wrongly concluded shortgpt16's closed-book axes were unavailable. The wzc1 shards were
verified to merge to exactly 17,944 / 14,267 / 3,610 items and to agree bit-for-bit with the merged
file, then `scp -O`'d and md5-verified.

**Code reuse.** `ni_rule`, `ratio_rule`, `load_shards`, `build_nulls`, `mmlu_content_norm_vec`,
`qa_metric_vec`, `EXPECTED_N`, `AXES`, `DEMOTED_AXES`, `PREREG` are **imported** from
`pilot_zero_rule_disagreement`; `paired_bootstrap`, `longest_option_vector`, `best_constant_qa`,
`TIE_CONVS`, `N_BOOT`, `SEED` from A03's `analyze_1b_knowledge_floor` via
`proposal_paths.a03_code_dir()`. **Nothing was reimplemented.** Bootstrap seed offsets
(`97·(200+i) + 13·axis_index`) are disjoint from Pilot Zero's (`ai ∈ {0,1}`) and the step-100k
pass's (`ai ∈ 100..102`), so no archived cell is perturbed.

**A03 scorer differs across disks** (`md5 8b454b5…` on zwfy6 vs `4571d76…` on wzc1). Diffed: the
only difference is inside `cell()`'s reporting (`residual_fraction_of_reported` /
`_of_headroom` split, added 2026-08-10) and its console format. **No null-producing or
metric-producing function differs**, so the nulls and metrics here are identical under either copy.

**Cross-check against an independent computation.** A01's `gate3_content_null_conventions.json`
independently reports 7B MMLU `split` null **0.28445022076627263** and `7B_base` bf16
`content_norm_acc` **0.47058823529411764` → residual **18.6138 pp**, matching this analysis to all
printed digits. `7B_keep14_step200000` 0.3832075202962541 and `7B_shortgpt16_step200000`
0.40115368181170774 likewise match. The 7B MMLU axis is therefore reproduced by two independently
written pipelines.

**Verified, not assumed**: `add_bos=false` and no chat template on every dir used (base-LM
protocol, per the standing project rule); `n_nan=0` everywhere; item_id sequences identical across
all arms on every axis (asserted — a mismatch would silently pair different items).

**Not touched**: `.104` (paperC Qwen3 heal), `.82` (A02 j=0), `LOCAL`/`.21` (SparseForge). No
`paperB` Qwen number is used anywhere here, so the OPEN Instruct-mislabelled-as-base defect in
`status/ISSUES.jsonl` does not reach these results; all rows are OLMo-2.
