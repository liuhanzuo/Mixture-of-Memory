---
scope: A03 seed 45 — the third and final pre-registered sampler seed. Verdict + the σ_run recompute at pooled df=5.
date: 2026-08-12 00:10 GMT+8
status: NOT-CONFIRM → aggregate **ARTIFACT** (0/3 CONFIRM). Verdict unchanged from DATAORDER_VERDICT.md. A-2 stays retracted.
prereg: DATAORDER_PREREG.md (`a25d780`, 2026-08-10 19:20:02) — seed 45 declared in §5 before any seed's result was visible
predeclaration: SEED45_PREDECLARATION.md — CONFIRM rule, mechanical aggregation and the "both branches retract A-2" argument locked before launch
handoff: SEED45_HANDOFF.md — the operational recipe this doc executed
evidence: evidence/a03_cpt_trajectory_paired_full_with_seed45.json (md5 `7b5cc4c7040561d9cdb8bd9d4916ad83`)
          evidence/a03_sigma_run_n3.json (md5 `5fb6cd4c3d693831e50d0817bda93ab8`)
          evidence/a03_seed45_integrity.json (md5 `df1535f0bab24f4ebdeade806935b9fb`)
          per-item shards: zwfy6 olmo2_closedbook_results/A03_1B_dataorder_seed45_step220000{,_nq}/,
                           olmo2_mmlu_content_results/A03_1B_dataorder_seed45_step220000/
---

# 0. One-line result

Seed 45's primary axis came back **θ = −0.3622 pp, CI95 [−0.5517, −0.1838], SIG
negative** — the *third* consecutive non-confirmation and the *second* significantly
negative one. Aggregate: **0/3 CONFIRM → ARTIFACT**, status quo. **A-2 remains
retracted.** Pooled σ_run moves from 0.3620 pp (df=4) to **0.3666 pp (df=5, χ² 95 %
CI [0.229, 0.899])**; the two decisions that consume it (A03's ARCHIVE, A04's Pilot Two
threshold) are **unchanged in direction and strengthened in one respect** — see §4.

# 1. Provenance: the run, and why its `rc=1` is not a failure

`_run_a03_dataorder_repl.sh` on `.82` launched at 12:04:08 and its own trainer-stop
watcher fired at **23:29:10** once `step220000.pt` appeared; `training exited rc=1` at
23:29:30. That non-zero code is the expected consequence of `kill -TERM` on a live
`torchrun`, not a crash. Two independent checks confirm the checkpoint:

| check | result |
|---|---|
| byte size vs siblings | `step205000.pt` = `step210000.pt` = `step215000.pt` = `step220000.pt` = **12,181,311,650 B**, delta **+0 B** |
| `torch.load(..., weights_only=False)` probe | the ext driver's own independent guard (`a03_dataorder_ext_driver.sh:48`) returned `ok`, so the eval proceeded |

This matters because the driver ships the **v1** bare-`[ -f ]` stop watcher
(`kill -TERM`; `sleep 20`; `kill -9`) — the exact race that truncated Arm 4's
`step220000.pt` to 49 %. **It did not fire this time.** Had it fired, the
pre-registered remedy was a full 20k re-run from `step200000`, never a resume from
`step215000` (that reproduces Arm 4's dataloader-offset defect, original-vs-redo loss
r = −0.0667).

Eval: `SEEDS=45 a03_dataorder_ext_driver.sh`, 8-way sharded on `.82` GPUs 0-7,
23:51:01 → 23:54:46 (~4 min for all four axes). The driver and the recompute script
were **byte-identical on both disks** before use (`2246a2e0d7eb781cbdd6eeb1c06b3874`,
`6ff48dd260293431c6d28615301d7ae2`), and the relocated loaders
`proposal/shared/code/canonical_eval_loaders.py` matched at
`2ccce419839b17f0d8f29233b4b569ff`. On zwfy6 the A03 tree is still at
`proposal/active/...` — that disk's `proposal/` is a hand-copied tree, not a git
checkout, so the `git mv` to `archive/` did not propagate. Same bytes, different path.

## 1.1 Shard integrity — asserted per cell, not assumed

The canonical loaders hard-fail on partial shard sets, but the counts are recorded
explicitly here because a silently-merged 5-of-8 set has corrupted results in this
repository before. Every cell, **arm and baseline**:

| axis | shards | n_scored | expected | dup | nan | per-shard split |
|---|---|---:|---:|---:|---:|---|
| popqa (arm) | 8/8 | 14,267 | 14,267 | 0 | 0 | 1784×3, 1783×5 |
| popqa (baseline) | 8/8 | 14,267 | 14,267 | 0 | 0 | 1784×3, 1783×5 |
| triviaqa (arm) | 8/8 | **17,944** | 17,944 | 0 | 0 | 2243×8 |
| triviaqa (baseline) | 8/8 | 17,944 | 17,944 | 0 | 0 | 2243×8 |
| nq_open (arm) | 8/8 | 3,610 | 3,610 | 0 | 0 | 452×2, 451×6 |
| nq_open (baseline) | 8/8 | 3,610 | 3,610 | 0 | 0 | 452×2, 451×6 |
| mmlu content_norm (arm) | 8/8 | 14,042 | 14,042 | 0 | 0 | 1756×2, 1755×6 |
| mmlu content_norm (baseline) | 8/8 | 14,042 | 14,042 | 0 | 0 | 1756×2, 1755×6 |

`summary.json` independently reports `n_valid = 14042`, `n_nan = 0` for MMLU. No log
matched the failure syntax `Traceback \(most recent call last\)|CUDA out of memory|loss=nan`.
(A substring grep for `nan` alone matches the harness's own **passing** line
`✓ No NaN/Inf in model parameters` and must not be used as a failure test.)

# 2. The verdict — the rule applied verbatim, not re-argued

`CONFIRM_45` ⟺ **CI excludes 0** and **θ > 0** and **θ ∈ [+0.20, +0.80] pp**
(prereg §3.3, quoted in `SEED45_PREDECLARATION.md`). Primary axis is triviaqa `em`,
17,944 items, baseline `A03_1B_keep7_step200k`, n_boot=5000, seed=42, CI95 percentile —
unretuned.

| sampler seed | θ (pp) | CI95 (pp) | CI excl. 0 | θ > 0 | in band | **§3.3 class** |
|---|---:|---|:---:|:---:|:---:|---|
| 0 (original Arm 3) | **+0.4793** | [+0.2675, +0.6910] | ✅ | ✅ | ✅ | (the claim being tested) |
| 43 | **+0.1115** | [−0.0947, +0.3177] | ❌ | ✅ | ❌ | NOT-CONFIRM |
| 44 | **−0.3455** | [−0.5517, −0.1393] | ✅ | ❌ | ❌ | NOT-CONFIRM |
| **45** | **−0.3622** | **[−0.5517, −0.1838]** | ✅ | ❌ | ❌ | **NOT-CONFIRM** |

Seed 45 fails on **two** of the three conjuncts (θ < 0, and therefore out of band).
Supplementary bootstrap p (not part of the rule): **p < 2×10⁻⁴** (floored at 1/n_boot).

**Aggregate: 0/3 CONFIRM → ARTIFACT.** Prereg §3.4 disposition, verbatim: *"Effect is
a data-order artifact. `ARM6_FINAL_VERDICT.md`'s positive reading is RETRACTED, this
time grounded. A03 retains only its Gate-1 pilot."*

**Both reachable branches retracted A-2, and this was written down before the number
existed.** REPLICATES was unreachable at n=3 because seeds 43 and 44 were already
NOT-CONFIRM and on disk; the only other reachable branch, MIXED, is *pre-declared a
FAILURE* by prereg §3.5 ("the positive reading is retracted as a general claim, and
the headline may not be the confirming seed"). So this is not optional stopping and
the outcome was not contingent on which way seed 45 fell. **Seed 45 is the last run
under this prereg. No seed 46.**

## 2.1 The four axes — descriptive only (§3.6)

Secondaries are barred from rescuing *or* strengthening the primary. Recorded for
completeness:

| axis | n | θ (pp) | CI95 (pp) | boot p | sig |
|---|---:|---:|---|---|:---:|
| **triviaqa em (PRIMARY)** | 17,944 | **−0.3622** | [−0.5517, −0.1838] | <2e−4 | **SIG** |
| popqa em | 14,267 | −0.2523 | [−0.4065, −0.1051] | <2e−4 | SIG |
| nq_open em | 3,610 | −0.0831 | [−0.4155, +0.2493] | 0.668 | TIE |
| mmlu content_norm | 14,042 | −0.3205 | [−0.5911, −0.0570] | 0.017 | SIG |
| mmlu letter | 14,042 | −0.2136 | [−0.5697, +0.1353] | — | TIE |

All five point estimates are negative. Note mmlu content_norm is SIG-negative here
where seeds 43/44 were TIE-negative — that is a *secondary* axis and it does not enter
the verdict, but it removes any reading in which the CPT top-up is neutral-to-positive
on the knowledge axes.

## 2.2 The empirical question the verdict could not answer at n=2 — now answered

`SEED45_PREDECLARATION.md` §"What seed 45 IS for" item 2 posed a discriminating
question: is the effect ~0 with seed 44 an unlucky draw, or is 20k-step CPT at this LR
**mildly harmful** on TriviaQA with the original +0.4793 as the outlier? Three of four
draws are now ≤ 0, two of them significantly:

| draws | mean (pp) | s (pp) | df | t₀.₉₇₅ | CI95 (pp) | as % of the 31.10 pp deficit |
|---|---:|---:|---:|---:|---|---|
| {0, 43, 44} (was quoted) | **+0.0817** | 0.4132 | 2 | 4.303 | [−0.945, +1.108] | +0.26 %, CI [−3.04 %, +3.56 %] |
| **{0, 43, 44, 45}** | **−0.0293** | 0.4039 | 3 | 3.182 | **[−0.672, +0.613]** | **−0.09 %, CI [−2.16 %, +1.97 %]** |

The pooled CPT increment **crosses zero to a slightly negative point estimate**, and
its CI tightens by ~38 %. The honest reading is **(a): the effect is indistinguishable
from zero**, now on four draws, and the "mildly harmful" reading (b) is *not*
established either — the CI still contains 0 comfortably. What is now excluded is a
recovery of more than ~2 % of the TriviaQA deficit at this budget. This aggregate is
**post-hoc and descriptive** (prereg §3.6 makes triviaqa em the sole primary endpoint);
it is reported for the *design* question only, which is the one-directional use §4 of
the prereg permits.

# 3. ★ σ_run at n=3 draws per family — pooled df=4 → df=5

Estimator per `SEED45_HANDOFF.md`: per-axis **arm mean** (absolute accuracy of each
seed's own checkpoint), **not** the paired delta — the delta shares the baseline term
across seeds, so its spread is not a single arm's run-to-run spread. `s` = sample sd
(ddof=1); χ² interval `[s√(df/χ²₀.₉₇₅), s√(df/χ²₀.₀₂₅)]`; pooled
`√((df₁s₁²+df₂s₂²)/(df₁+df₂))`. All recomputed from per-item shards by
`code/recompute_sigma_run_n3.py`, not transcribed.

## 3.1 keep7+fresh2, 20k CPT — sampler seeds {0, 43, 44, **45**}: S=3 → **S=4, df=3**

| axis | arm means (%) | s (pp) | df | χ² 95 % CI for σ (pp) | width |
|---|---|---:|---:|---|---:|
| triviaqa em | 10.0646 / 9.6968 / 9.2399 / **9.2231** | **0.4039** | **3** | **[0.229, 1.506]** | 6.6× |
| popqa em | 3.9392 / 4.1004 / 3.7149 / **3.6868** | 0.1959 | 3 | [0.111, 0.730] | 6.6× |
| nq_open em | 2.9363 / 2.9086 / 2.9086 / **2.7701** | 0.0750 | 3 | [0.042, 0.280] | 6.6× |
| mmlu content_norm | 32.2390 / 32.2319 / 32.1963 / **32.1179** | 0.0555 | 3 | [0.031, 0.207] | 6.6× |

(Previously: triviaqa s = 0.4132, df = 2, χ² [0.215, 2.597], width 12.1×.)

★ **Three of these four are the repo's first real σ estimates on those axes for this
arm.** `ARM_SET_DECISION.md` §0/§4.2 could only show keep7 popqa/mmlu/nq_open as
**df = 1 pairwise ranges** (0.2726 / 0.0252 / 0.0000) and flagged them "must not be
quoted as σ". They are now df = 3 σ estimates with intervals. Note nq_open's df=1
range was **0.0000** — a coincidence of two draws scoring identically (105/3610 both,
documented in `DATAORDER_VERDICT.md`) — and the true s on that axis is 0.0750, not 0.

## 3.2 keep12+fresh2, 5k — seeds {101, 102, 103}: unchanged, S=3, df=2

| axis | s (pp) | df | χ² 95 % CI for σ (pp) |
|---|---:|---:|---|
| triviaqa em | 0.3023 | 2 | [0.157, 1.900] |
| popqa em | 0.3328 | 2 | [0.173, 2.092] |
| nq_open em | 0.2091 | 2 | [0.109, 1.314] |
| mmlu content_norm | 0.0783 | 2 | [0.041, 0.492] |

Re-derived here from shards with the same estimator; matches
`A04/evidence/stageB_S3_verdict.json` to 4 decimals.

## 3.3 Pooled — **df = 4 → df = 5** (the headline of this section)

| axis | σ old (df=4) | χ² CI old | **σ new (df=5)** | **χ² 95 % CI new** | width |
|---|---:|---|---:|---|---:|
| **triviaqa em** | **0.3620** | **[0.217, 1.040]** | **0.3666** | **[0.229, 0.899]** | 12.1×→**3.9×** |
| popqa em | — | — | 0.2595 | [0.162, 0.636] | 3.9× |
| nq_open em | — | — | 0.1445 | [0.090, 0.354] | 3.9× |
| mmlu content_norm | — | — | 0.0656 | [0.041, 0.161] | 3.9× |

**The point estimate barely moved (+1.3 %, 0.3620 → 0.3666 pp) while the χ² interval
tightened 22 % at the upper end (1.040 → 0.899 pp).** That is the whole value of seed
45: it did not change *where* σ is, it narrowed *how badly we know it*. The
multiplicative width of a χ² σ interval by d.o.f.: **71.5× (df 1), 12.1× (df 2), 6.6×
(df 3), 4.8× (df 4), 3.9× (df 5)**.

## 3.4 t-based MDE, triviaqa em

Two-sample, α = 0.05 two-sided, power 0.80, `(t₀.₉₇₅,₂ₛ₋₂ + t₀.₈₀,₂ₛ₋₂)·σ·√(2/S)`.
**t, not z** — d.o.f. is small and t is the conservative choice.

| S | MDE @ σ̂=0.3620 (df 4, OLD) | MDE @ σ=1.040 (OLD upper) | **MDE @ σ̂=0.3666 (df 5, NEW)** | **MDE @ σ=0.899 (NEW upper)** |
|---:|---:|---:|---:|---:|
| 3 | 1.10 pp | 3.16 pp | **1.11 pp** | **2.73 pp** |
| 4 | 0.86 pp | 2.47 pp | 0.87 pp | 2.13 pp |
| 5 | 0.73 pp | 2.10 pp | 0.74 pp | 1.82 pp |
| 8 | 0.55 pp | 1.57 pp | **0.55 pp** | **1.35 pp** |

At the point estimate the MDE is **unchanged to 2 significant figures** (1.10 → 1.11 pp
at S=3; 0.55 pp at S=8). At the honest pessimistic end it improves ~14 %
(3.16 → 2.73 pp at S=3; 1.57 → 1.35 pp at S=8).

# 4. Does this change the two decisions that consume σ?

## 4.1 (a) A03's ARCHIVE verdict — **NO, and it is now better supported**

`ARM_SET_DECISION.md` archived A03 on the ground that *every remaining
training-recipe arm targets an effect ≤ 1 pp, and no affordable S can detect it.*
Check both halves at df=5:

* **The bar barely moved.** S=3 MDE 1.10 → 1.11 pp; even S=8 (1,451 GPU-h for two
  arms at 20k steps) is 0.55 pp at σ̂ and 1.35 pp at the χ² upper bound. The claim
  "unresolvable at any S we can afford" is intact — in fact §2.1's own phrasing
  ("S=8 still only reaches 1.57 pp at the honest end of σ") should now read **1.35 pp**,
  which is *tighter* but still above the sub-1 pp targets.
* **The effect being chased got smaller, not larger.** The measured CPT increment moved
  from +0.0818 pp (CI [−0.945, +1.108]) to **−0.0293 pp (CI [−0.672, +0.613])** at
  n=4 — a *negative* point estimate whose CI still contains zero. The gap between
  "what there is to detect" and "what we can detect" **widened**.

So the ARCHIVE decision does not flip; it is reinforced from both sides. **This is
reported as it came out, not as it was wanted:** the σ tightening was the one outcome
that *could* have argued for un-archiving (a much smaller σ would have lowered the MDE
below the effects on offer). It did not — σ̂ went marginally **up**, and the effect went
**down**.

## 4.2 (b) A04's Pilot Two MDE threshold — **restated at df=5; direction unchanged**

`STAGE_B_DECISION.md`'s addendum item 4 and `A04/STATUS.json:next_gate[4]` state the
threshold A04 must clear pre-data as **"1.10 pp at S=3 and σ̂ = 0.362 pp; 3.16 pp at
the χ² 95 % upper bound"**. Restated: **1.11 pp at S=3 and σ̂ = 0.3666 pp (df=5);
2.73 pp at the χ² 95 % upper bound**. The requirement itself — *state what recovery
magnitude the certification adjudicates, and show it exceeds the MDE the chosen S
implies* — is unchanged. The pessimistic-end bar drops from 3.16 to 2.73 pp, which
makes A04's job **slightly easier**, not harder. This weakens nothing in
`ARM_SET_DECISION.md` §4.2's net conclusion (that A03's decision *weakens* the case for
the next tranche), because that conclusion rests on the **effect size** A04's rule
adjudicates being ≈0 — and §2.2 above just moved that estimate from +0.08 pp to
−0.03 pp. **The case for Pilot Two is weakened slightly further, not strengthened.**

## 4.3 popqa's K2 trigger at the χ² upper bound — **it still fires as pre-registered**

`ARM_SET_DECISION.md` §4.2(b) recorded: K2 does not fire at the S=3 point estimates
(margins 7.9× triviaqa / 2.4× popqa / 7.8× mmlu_content), **but at the χ² 95 % upper
bound of each df=2 σ, popqa would fire** (3.527 vs Δ=1.321), as would demoted nq_open
(2.215 vs 0.970). Does df=5 close that?

**As pre-registered: no. Under a pooled substitution: yes.** The distinction is
load-bearing and both are reported:

| axis | Δ (pp) | **PRE-REG form** (keep12 s, df=2, t₀.₀₅=2.920) | | **pooled substitute** (df=5, t₀.₀₅=2.015) | |
|---|---:|---:|---|---:|---|
| | | bound₃ @ point | @ χ² upper | bound₃ @ point | @ χ² upper |
| triviaqa | 4.043 | 0.510 | 3.203 → no | 0.426 | 1.046 → no |
| **popqa** | **1.321** | **0.561** | **3.526 → FIRES** | **0.302** | **0.740 → no** |
| mmlu_content | 1.024 | 0.132 | 0.830 → no | 0.076 | 0.187 → no |
| nq_open (demoted) | 0.970 | 0.353 | 2.216 → FIRES | 0.168 | 0.412 → no |
| | | | **1 decision axis fires** | | **0 fire** |

* **`PILOT_ONE_PREREG.md`'s K2 estimator is the keep12 family's own `sd_run` at
  df = 2** with `bound₃ = t₀.₀₅,df₂ · s/√3 = 2.920·s/√3`. **Seed 45 is a keep7 draw and
  adds nothing to the keep12 family**, so the pre-registered arithmetic is *numerically
  untouched*: popqa still fires at its χ² upper bound (3.526 vs 1.321), and the honest
  line from §4.2(b) stands verbatim — *"K2 does not fire at the point estimate, and one
  decision axis would fire at the pessimistic end of a df = 2 σ interval."* K2 itself
  still does not fire (its rule needs ≥2 of 3 decision axes; 1 is not 2).
* **If** one substituted the pooled df=5 σ (a *change of estimator*, not something
  seed 45 licenses on its own), every axis clears Δ at both ends and nothing fires.
  That is worth knowing but **must not be presented as "seed 45 closed the popqa
  trigger"** — it would be swapping the pre-registered per-family estimator for a
  cross-arm pooled one after seeing which answer each gives.
* Therefore the recommendation from `ARM_SET_DECISION.md` §4.2 — **more keep12 seeds
  before Pilot Two** — stands. The way to close popqa's pessimistic-end trigger is
  df on the *keep12* family, which only keep12 seeds buy.

## 4.4 Spread is still not monotone in damage

With keep7 now at df=3 the comparison is cleaner than the df=1-vs-df=2 version in
`ARM_SET_DECISION.md` §4.2, and the conclusion is **unchanged** — 3 of 4 axes go the
"wrong" way:

| axis | keep7 20k (S=4, df=3) | keep12 5k (S=3, df=2) | direction |
|---|---:|---:|---|
| triviaqa em | 0.4039 | 0.3023 | keep12 smaller |
| popqa em | 0.1959 | 0.3328 | **keep12 LARGER (1.7×)** |
| nq_open em | 0.0750 | 0.2091 | **keep12 LARGER (2.8×)** |
| mmlu content_norm | 0.0555 | 0.0783 | **keep12 LARGER (1.4×)** |

Any seed budget premised on "less damage ⇒ less variance" remains mis-budgeted. The
earlier df=1 pairwise entries (0.2726 / 0.0252 / 0.0000) are now **superseded by real
σ estimates** and must not be quoted at all.

# 5. Reporting discipline (prereg §4, still binding)

* **Never quote σ_run without its d.o.f. and χ² interval.** The canonical strings are
  now: keep7 20k triviaqa **0.4039 pp (df=3, χ² [0.229, 1.506])**; pooled
  **0.3666 pp (df=5, χ² [0.229, 0.899])**.
* Do not quote 0.3620 pp / df=4 / χ² [0.217, 1.040] as current — superseded here.
  The old values are retained in place with `SUPERSEDED` markers, not deleted:
  the retraction history is itself part of the record.
* **A tighter σ does not revive A03** (§4.1). Archiving was decided on effect size vs
  spread; seed 45 moved both in the archiving direction.
* n=3 draws per family is still a **weak** estimate (3.9× multiplicative width pooled).
  That is the honest answer and it goes in the record as such. There is no seed 46.
