---
scope: A04 Pilot One STAGE B — the S=3 `sd_run` at keep12+fresh2, the K2 verdict, and the falsifiability check. HARVEST + VERDICT.
date: 2026-08-12 GMT+8
status: TERMINAL for Stage B. The 135 GPU-h were spent 2026-08-11; this is the harvest.
prereg: PILOT_ONE_PREREG.md (commit 2ac0b5a, PRE-DATA) §2.2 rule, §3 Stage B design
gpu_h_additional_for_this_verdict: 0
decides: K2 (point estimate + pessimistic reading) AND prereg §5 known-unverified item 3
---

# Stage B verdict: K2 does **not** fire — and that is not good news for A04

> **One-line summary.** K2 does not fire (0 of 3 decision axes exceed Δ; margins
> 2.35–7.93×), so A04 does **not** die of seed variance. But the analysis K2 cannot
> perform shows `keep12+fresh2 @ 5,000 steps` is a **constant-REJECT** rung — NI
> rejects on 4/4 axes by **27–90 × the measured `sd_run`** — which is the exact
> defect that disqualified `keep7` and precisely what prereg §3 chose `keep12` to
> escape. **Stage B passed its kill gate and still failed to deliver its purpose.**

Every number below is machine-read from the two committed JSONs. No number in this
document was hand-transcribed.

* `evidence/pilot_one_stage_b_s3_verdict.json` (md5 `7dc77ca81551b708d9bfaa582739cfad`) — K2
* `evidence/pilot_one_stage_b_falsifiability.json` (md5 `1fd27cbed050a696cdf8a3b573ab5aa7`) — falsifiability
* drivers: `code/pilot_one_stage_b_s3.py`, `code/pilot_one_stage_b_falsifiability.py`

---

## 0. What was on disk, and what had been missed

The three pre-registered seeds `{101,102,103}` trained to step 5000 on zwfy6 on
2026-08-11 and **all three were scored on all four axes the same day**. `STATUS.json`
still said `stage_b.status: RUNNING`, and `evidence/` on **wzc1** contained no Stage B
result at all. A Stage-B S=3 verdict JSON did exist **on zwfy6 only**
(`stageB_S3_verdict.json`, md5 `7145d569f46ec0fa10dd56368071adf2`, written 14:53) and had
never been harvested across the disk boundary, never committed, and had no driver in
`code/`. Three defects in it, all fixed here:

| # | Defect in the un-harvested zwfy6 JSON | Fix |
|---|---|---|
| 1 | Mislabelled `"gate": "A04 Pilot One Stage A"` with verdict string `STAGE_A_DOES_NOT_FIRE`. A future agent grepping for the Stage-B verdict would have found a file claiming to be Stage A. | Correctly labelled `A04_pilot_one_stage_B_S3_sd_run` / `K2_DOES_NOT_FIRE_AT_STAGE_B`. |
| 2 | **No integrity counts whatsoever** — prereg §4 requires asserting `n_shards==8`, indices exactly `{0..7}`, and the exact item counts. Nothing in it was auditable. | Full `integrity` block, all 12 cells. |
| 3 | ★ **The exact-item-count assertion had never actually run on the closed-book axes.** `canonical_eval_loaders.load_cb` asserts only that 8 shard *files* exist; unlike `load_mmlu` it has **no** count check and **no** duplicate-`item_id` check — it merges into a dict, so an overlapping shard range is silently absorbed. So for `triviaqa`/`popqa`/`nq_open` the prereg's headline integrity requirement was unenforced. | Enforced **outside** the loader (the loader is shared with A03's archived numbers and is deliberately not edited). Also checks the shard **index set**, not just a glob count — `{0,…,5,6,6}` passes a count check. |

**Integrity result: all 12 cells clean.** 8/8 shards, indices exactly `{0..7}`,
`n_scored == expected` (triviaqa 17,944 / popqa 14,267 / mmlu 14,042 / nq_open 3,610),
`nan == 0`. Nothing was re-run; nothing needed to be.

The three checkpoints are the right artefacts: `arch_meta.json` confirms
`keep_front_layers 12`, `n_fresh_layers 2`, `num_hidden_layers 14`,
`transplant_max_abs_diff 0.0`, `n_copied 135 == expected_copied`, and each log ends
`DONE [healing_front12+fresh2] at step 5000`.

---

## 1. K2 — the pre-registered rule, applied

**Estimator (prereg §3).** Stage A used `sd_run = |m_a − m_b|/√2` at n=2. Stage B uses
the proper sample sd over 3 seed means (ddof=1, df=2), `t_{0.05,2} = 2.920`,
`bound₃ = 2.920·sd_run/√3`. These are the **same estimator**: at n=2 the ddof=1 sample
sd *is* `|a−b|/√2`. Δ and the ≥2-of-3 rule are **unchanged** from Stage A, as §3 directs.

| axis | seed 101 | seed 102 | seed 103 | range (pp) | `sd_run` (pp) | `bound₃` (pp) | Δ (pp) | Δ/bound | exceeds Δ? |
|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|
| `triviaqa` | 9.2566 | 9.6634 | 9.0727 | 0.5907 | **0.3023** | **0.5096** | 4.0431 | 7.93x | no |
| `popqa` | 5.5793 | 4.9415 | 5.0957 | 0.6378 | **0.3328** | **0.5610** | 1.3205 | 2.35x | no |
| `mmlu_content` | 31.6194 | 31.6978 | 31.7761 | 0.1567 | **0.0783** | **0.1321** | 1.0239 | 7.75x | no |
| `nq_open` *(demoted)* | 2.7147 | 2.2992 | 2.4654 | 0.4155 | **0.2091** | **0.3526** | 0.9695 | 2.75x | no |

### → **K2 DOES NOT FIRE.** 0 of 3 decision axes exceed Δ.

**A04 does not die of seed variance.** The design's self-declared "most likely killer"
missed. Independently cross-checked: these `sd_run` values agree to <1e-12 with the
`a04_k2` block of A03's `evidence/a03_sigma_run_n3.json`
(md5 `5fb6cd4c3d693831e50d0817bda93ab8`), computed by a different agent from the same shards.

### 1.1 Can Stage B emit `K2_CLEARED`? Asked because Stage A explicitly could not.

Stage A could never clear K2 for three stated reasons (§2.3). Stage B repairs two:

| Stage A objection | Stage B |
|---|---|
| wrong arm — `keep7` (56.2% depth), not `keep12` (87.5%) | **REPAIRED** — Stage B *is* `keep12+fresh2` |
| wrong budget — 20,000 steps, not 5,000 | **REPAIRED** — Stage B *is* 5,000 steps |
| n=2 is a range, not a variance with a usable CI | **only PARTIALLY** — S=3 gives a real sd but df=2, whose χ² 95% interval for σ spans a **12.1× multiplicative** width |

**The prereg never defines a Stage-B `K2_CLEARED` state**, so this harvest does not
invent one. The terminal value is the rule's own output: `K2_DOES_NOT_FIRE_AT_STAGE_B`.

**The rule is not ambiguous for S=3** — §3 explicitly states the S=3 estimator (`proper
sd, df=2, t=2.920`) and says to apply the §2.2 rule "unchanged", and §2.4 fixes the
1-of-3 case. So there is one reading, and it is the one applied. What §3 does *not*
define is a *clearance*, and that gap is reported rather than filled.

### 1.2 The pessimistic reading — one decision axis would fire, and seed 45 does not close it

At the χ² 97.5th-percentile end of each df=2 σ interval:

| axis | `sd_run` | σ χ² 95% CI (pp) | `bound₃` at σ upper | Δ | fires? |
|---|---:|---|---:|---:|:--:|
| `triviaqa` | 0.3023 | [0.1574, 1.8998] | 3.2028 | 4.0431 | no |
| `popqa` | 0.3328 | [0.1733, 2.0915] | 3.5260 | 1.3205 | **FIRES** |
| `mmlu_content` | 0.0783 | [0.0408, 0.4923] | 0.8300 | 1.0239 | no |
| `nq_open` *(demoted)* | 0.2091 | [0.1089, 1.3144] | 2.2158 | 0.9695 | **FIRES** |

**1 of 3 decision axes.** The rule needs ≥2, so **this does not reach a fire either** —
but the honest line is *"K2 does not fire at the point estimate, and `popqa` would fire
at the pessimistic end of a df=2 σ interval."*

A03's `SEED45_VERDICT.md` established this and it remains true: **seed 45 is a `keep7`
draw and adds nothing to `keep12`.** K2's pre-registered estimator is the `keep12`
family's own `sd_run`, so substituting the pooled df=5 σ (which would make nothing fire)
is a **change of estimator selected after seeing which answer each gives** and is not
licensed. Closing `popqa`'s trigger requires **more `keep12` seeds** — and prereg §3
forbids adding them to rescue a bound.

---

## 2. The finding K2 is blind to: `keep12` is a **constant-REJECT** rung

This is the part that matters, and no clause in the gate was watching for it.

**Why K2 cannot see it.** K2 compares run-to-run noise against Δ. A *saturated* deficit
is highly reproducible across seeds — low variance is exactly what a saturated axis looks
like. So a constant-REJECT rung sails through K2. Passing K2 says the gate is *readable*;
it says nothing about whether there is anything to read.

**What Stage B was for.** prereg §3, verbatim: `keep7+fresh2` is a confirmed
constant-REJECT rung, so "a rule tested only there can never be *observed to accept* and
the disagreement is automatic and uninformative. **keep12 at 87.5% depth is the candidate
most likely to let NI sometimes accept, which is what makes the disagreement test
falsifiable at all.**" And §5 item 3 flags as unverified: "Whether 5,000 steps at keep12
produces enough recovery for NI to ever accept."

Against the G0-pinned intact anchor (`A03_1B_base`; `deficit = residual(intact) −
residual(arm) = reported(intact) − reported(arm)`, so the null cancels exactly and the
deficit is convention-independent):

| axis | null | intact | intact residual | keep12 mean | recovery | **deficit** | Δ | **NI** | (deficit−Δ)/`sd_run` |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|
| `triviaqa` | 0.2564 | 40.6877 | 40.4313 | 9.3309 | 22.44% | **31.3568** | 4.0431 | **REJECT** | **+90.4** |
| `popqa` | 2.2920 | 15.4973 | 13.2053 | 5.2055 | 22.06% | **10.2918** | 1.3205 | **REJECT** | **+27.0** |
| `mmlu_content` | 28.4450 | 38.6839 | 10.2389 | 31.6978 | 31.77% | **6.9862** | 1.0239 | **REJECT** | **+76.1** |
| `nq_open` *(demoted)* | 0.5263 | 10.2493 | 9.7230 | 2.4931 | 20.23% | **7.7562** | 0.9695 | **REJECT** | **+32.5** |

### → `CONSTANT_REJECT_AT_KEEP12`. NI rejects on **4/4** axes.

The **smallest** margin on any decision axis is **+27.0 `sd_run`**. For NI to accept on
`popqa`, a seed draw would have to move the mean by 27 standard deviations. **No
realisable seed draw flips NI at this rung.** NI is a constant here, so the
NI-vs-PLATEAU disagreement is automatic and uninformative — the precise defect
`keep12` was selected to escape.

### 2.1 `keep12` @ 5k ≈ `keep7` @ 220k. Depth was not the bottleneck.

| axis | `keep12` 5k recovery | `keep7` 220k recovery | NI @ keep12 | NI @ keep7 |
|---|---:|---:|:--:|:--:|
| `triviaqa` | 22.44% | 23.00% | REJECT | REJECT |
| `popqa` | 22.06% | 11.88% | REJECT | REJECT |
| `mmlu_content` | 31.77% | 36.64% | REJECT | REJECT |
| `nq_open` | 20.23% | 24.22% | REJECT | REJECT |

`keep12` reaches a comparable-or-better recovery fraction in **1/44th the steps** — and
is still constant-REJECT. Recovery is stuck near ~20–32% of the intact residual whether
you remove 2 layers or 9, and whether you heal for 5k steps or 220k. That is the
substantive scientific result of the 135 GPU-h.

### 2.2 The prereg anticipated only the *opposite* failure, so its repair does not apply

§5 item 3: "If `keep12` turns out to be a **constant-ACCEPT** rung, the gate must bracket
downward to `keep10`." The observed failure is **constant-REJECT**. `keep10` (75% depth)
is *more* damaged than `keep12` (87.5%) and would reject harder — **bracketing down cannot
repair a constant-REJECT rung.** Escaping it needs either *less* damage (`keep14`) or far
more heal tokens; A03 already showed 10× the token budget (52.43 B tokens at `keep7`) does
not close the gap. Neither is authorised by any current prereg, and both are new GPU
tranches, not re-analyses.

---

## 3. Engaging `STAGE_B_DECISION.md`'s noise-floor worry, as required

`STAGE_B_DECISION.md` recommended HOLD, arguing the keep12 effect would be "plausibly
< 0.5pp, i.e. inside the noise floor". Its own 11:05 correction banner already retracted
the "noise floor we just measured" phrasing as circular (no floor had been measured).
With a real measured `sd_run` in hand, the verdict is: **the worry conflated two
different quantities, which land on opposite sides of the noise floor.**

1. **The level NI adjudicates — FAR OUTSIDE the noise floor.** The keep12 deficits are
   31.3568 / 10.2918 / 6.9862 pp = **104 / 31 / 89 × `sd_run`**. NI's accept/reject
   decision at keep12 is not noise-limited at all; it is **saturated**. The
   noise-floor worry simply does not apply to it.
2. **The marginal CPT effect — INSIDE the noise floor.** A03's measured increment
   (−0.0293 pp, CI95 [−0.672, +0.613], df=3) is indistinguishable from zero. The worry
   was right about *this* quantity.

**And being far outside the floor is worse for A04, not better.** A04 needed the deficit
to be *comparable* to Δ so NI could sometimes accept and sometimes reject. Instead the
deficit exceeds Δ by 27–90 σ. Overshooting in the REJECT direction is not a rescue — it
is the degeneracy. **A04 is not killed by noise; it is stalled by saturation, and no
number of extra seeds can fix saturation.**

`STAGE_B_DECISION.md`'s empirical prediction that keep12 spreads would be *smaller* than
keep7's was also wrong, as its banner conceded: keep12 is larger on 3 of 4 axes against
keep7's df=3 σ. Spread is not monotone in damage.

---

## 4. What Stage B did **not** establish — stated plainly

1. **The disagreement itself was not measured at `keep12`.** PLATEAU(T) needs in-domain
   val PPL on the checkpoint grid. **No val PPL exists for any Stage B checkpoint** — the
   training logs contain zero val/eval lines and `olmo2_ppl_results/` has no `*stageB*`
   directory. Only NI's half is on disk, so the NI-vs-PLATEAU **disagreement** at keep12
   is unevaluated and **K1 remains unadjudicated at this rung**, independently of §2.
2. **Grid coverage is 2 of 6.** The prereg grid is `[2500, 5000, 10000, 20000, 40000,
   80000]`; Stage B produced 2500 and 5000. K1's ≥24-cell denominator is untouched.
3. **`nq_open` Δ has a 2.77e-3 pp provenance drift.** Recomputing Δ from the pinned
   anchor reproduces the pre-registered constant **exactly on all three decision axes**
   (drift 0.00e0 / 2.22e-16 / 0.00e0). `nq_open` differs because the prereg used a rounded
   null (0.0055 → 0.5500 pp) where the harness's `majority_em` is 0.5263 pp. `nq_open` is
   **demoted and carries no decision weight**, so no verdict changes; recorded for honesty.
4. **Uniform LR, not differential.** All four optimizer groups logged at `2.00e-05`
   (`fresh_decay` 339.7M / `fresh_nodecay` 0.0M / `inh_decay` 1010.8M / `inh_nodecay`
   0.1M). **Differential LR must not be claimed.** Unlike the distill trainer's
   `_classify_param` defect, the fresh groups *do* exist here (339.7M is classified), so
   this is a config choice (`--lr 2e-5 == --lr_inherited 2e-5`), not the silent no-op bug.
5. **Manipulation check: data order genuinely varies — but the raw statistic is
   misleading here.** Raw tail (step ≥ 1000) loss correlations are r = +0.944 to +0.948,
   which would look phase-locked against A03's reference (+0.99966 locked / −0.0101
   independent). That raw number is **not interpretable for these runs**: A03's were
   *resumed* and flat, whereas these are fresh 5k-step runs with a strong shared loss
   *decay* that dominates the correlation. Detrended per-seed by its own rolling median
   (w=9): **r = +0.0089 / +0.0496 / +0.0118** — matching A03's independent-pair reference.
   No two loss series are identical. The `DistributedSampler(seed=)` fix (`ce5c298`) is
   active and the seeds are independent draws.

---

## 5. Terminal state and what follows

`STATUS.json: pilot_one.stage_b.status` → **`COMPLETE_K2_DOES_NOT_FIRE_BUT_RUNG_IS_CONSTANT_REJECT`**

**The gate design worked, in the way gate designs are supposed to.** 135 GPU-h bought a
decisive answer about a 1,077–4,309 GPU-h tranche. K2 — the clause the design nominated
as most likely killer — did not fire, and the honest thing is to record that it missed.
But the same 135 GPU-h surfaced a defect no clause was watching for: **the rung the gate
would run on cannot produce an informative disagreement, because NI is a constant on it
by 27–90 σ.**

The load-bearing consequence for the next tranche: **Pilot Two as designed would spend
1,077–4,309 GPU-h adjudicating a disagreement that is automatic at every rung measured so
far.** `keep7` @ 220k: constant-REJECT. `keep12` @ 5k: constant-REJECT. Recovery pinned at
~20–32% of the intact residual across a 44× span of heal budget and a 31.3pp span of
damage. Before any further GPU is committed, A04 must show — **pre-data** — that some
rung exists where NI can be *observed to accept*. Nothing measured to date is such a rung,
and the prereg's own repair (bracket down to `keep10`) provably goes the wrong way.

This verdict does **not** kill A04 by itself, and does not authorise Pilot Two either way:
Pilot Two requires explicit user approval per `STATUS.json:next_gate`. Pilot Zero's
level-at-one-checkpoint finding is untouched by all of the above.
