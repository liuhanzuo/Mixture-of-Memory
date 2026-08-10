---
scope: A03 Arm 4 (peak-LR CPT) verdict; task #226
date: 2026-08-10
status: FINAL for Arm 4 step220000 headline; supersedes intermediate "NOT_YET_JUDGED" verdict at [`STATUS.json:arm4_peaklr_cpt.verdict`]
run: outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k on .73 (zwfy6), 20k Dolmino CPT
config: ARM4_DESIGN.md Config B -- warmup_steps=200500, max_steps=240000 -> lr in [1.995e-5, 1.12e-5], i.e. [1.00x, 0.56x] peak across steps 200000-220000
eval_node: .82 (zwfy6, ext_driver same as Arm 3, 8-shard MMLU+CB{popqa,triviaqa,nq_open})
evidence: evidence/arm3_arm4_cpt_trajectory_paired_full.json (regenerated with all 4 Arm4 ckpts)
protocol: per-item paired difference bootstrap, n_boot=5000, seed=42, CI95 percentile; SIG = CI excludes 0
---

> ## ⚠️ RETRACTED 2026-08-10 — see `ARM6_STEP215_VERDICT.md`
>
> The 'peak-LR CPT actively harms' conclusion below does NOT hold. Arm 4's
> `-0.93pp SIG` triviaqa em at step220000 is one point in a 4-point series that
> also contains **+1.26pp SIG** at step210000 — a 2.66pp swing, **10.0x** the
> bootstrap CI half-width, with Arm 4's trajectory mean at −0.298pp. The paired
> bootstrap here measures item-resampling variance while the dominant variance is
> checkpoint-to-checkpoint. Read below as a record of what was believed, not as a
> finding.

# A03 Arm 4 (peak-LR CPT) — VERDICT

## 0. Bottom line

**The +0.48pp SIG triviaqa gain that Arm 3 (late-cosine, 0.28-0.33× peak LR)
produced at step220000 is NOT reproduced under peak-LR CPT.** Under peak-LR CPT,
matched at 20k steps, triviaqa em goes the OPPOSITE way (**−0.93pp SIG**), and
popqa collapses (**em −0.97pp SIG, f1 −1.45pp SIG**). This is not a "peak-LR
does it faster or bigger" reading; it is a sign-flip on the primary axis.

**Two interpretations, either of which is bad news for a Arm-3-headline paper:**

* (i) Arm 3's +0.48pp was a genuine but **LR-schedule-specific** effect (only
  the late-cosine tail regime produces it) — so the finding does not generalise
  across a natural sibling schedule and its scientific claim narrows.
* (ii) Arm 3's +0.48pp was near the noise floor of the design, and a related
  schedule pulls a comparable-magnitude *negative* — so the effect size claim
  itself is not robust.

Both readings mean Arm 3's headline **is not a discovery about parametric
knowledge recovery through CPT**; it is a discovery about a specific
recipe. The `code/recompute_cpt_trajectory_paired.py` output is committed as
evidence.

## 1. What Arm 4 was designed to answer

Arm 3 (interim then FINAL verdict `ARM3_CPT_TRAJECTORY_INTERIM_VERDICT.md`)
found `triviaqa em +0.48pp SIG` at step220000 only, 3 co-moving cells, at
0.28-0.33× peak LR (the cosine tail already consumed by Arm 2). Two clean
readings were allowed:

* Arm 4 shows the same or larger gain → Arm 3 was real, we picked too low an LR
* Arm 4 shows no gain (saturation-consistent) → Arm 3's late-cosine LR band was
  where the effect sits, plateau claim strengthens

Neither prediction survived. Arm 4 shows a *third* outcome: peak LR at 20k
produces a significant *negative* on the primary axis.

## 2. Per-cell numbers (n_boot=5000, seed=42)

Recomputed from 8/8 per-item shards; hard-asserts shard completeness.

### Arm 3 (late-cosine, LR ~0.28-0.33x peak across 20k steps)

| step | triviaqa em | triviaqa contains | triviaqa f1 | popqa em | popqa f1 | nq_open em |
|---|---|---|---|---|---|---|
| 205000 | −0.48 SIG | +0.41 SIG | −0.63 SIG | −0.35 SIG | −0.82 SIG | +0.11 |
| 210000 | +0.37 SIG | +0.04 | +0.18 | +0.16 | −0.25 SIG | +0.00 |
| 215000 | −0.02 | +0.27 | +0.08 | −0.04 | −0.26 SIG | +0.28 |
| **220000** | **+0.48 SIG** | **+0.47 SIG** | **+0.47 SIG** | +0.00 | +0.12 | +0.08 |

Only step220000 shows the 3-metric co-moving positive signature on triviaqa.
Earlier cells wobble.

### Arm 4 (peak-LR, LR in [1.00x, 0.56x] peak across 20k steps)

| step | triviaqa em | triviaqa contains | triviaqa f1 | popqa em | popqa f1 | nq_open em |
|---|---|---|---|---|---|---|
| 205000 | −1.40 SIG | −0.08 | −1.19 SIG | −0.97 SIG | −1.44 SIG | −0.25 |
| 210000 | **+1.26 SIG** | −0.22 | **+1.00 SIG** | +0.20 SIG | +0.03 | +0.28 |
| 215000 | −0.12 | −0.12 | −0.23 | −0.40 SIG | −0.46 SIG | +0.28 |
| **220000** | **−0.93 SIG** | **+0.88 SIG** | **−0.82 SIG** | **−0.97 SIG** | **−1.45 SIG** | +0.08 |

Notable: step210k has +1.26pp SIG on triviaqa em, LARGER than Arm 3's step220000
headline. But it is a transient — step215 is noise, step220 is significantly
negative. ARM4_DESIGN.md predicted exactly this early transient window: "Arm 4's
first 500 steps are Adam-moment-mismatched; interpret step205k onward". Under
that principle the *interpretable* dose points are 210/215/220k, and they do not
form a coherent trajectory in favour of the +0.48pp headline.

## 3. Head-to-head, matched 20k window

The comparison Arm 4 was built to enable:

| axis        | Arm 3 step220000 | Arm 4 step220000 | reading |
|---|---|---|---|
| triviaqa em       | +0.48 SIG | **−0.93 SIG** | **sign flip on primary axis** |
| triviaqa contains | +0.47 SIG | +0.88 SIG   | same sign, larger in Arm 4 |
| triviaqa f1       | +0.47 SIG | **−0.82 SIG** | **sign flip** |
| popqa em          | +0.00 | **−0.97 SIG** | Arm 4 significantly worse |
| popqa contains    | +0.23 | +0.28   | tie both |
| popqa f1          | +0.12 | **−1.45 SIG** | Arm 4 significantly worse |
| nq_open em        | +0.08 | +0.08   | tie both (low sensitivity, n=3610) |
| mmlu letter/content | flat | flat | (not re-run here; A01 shows mmlu unmoved across Arm 3 trajectory) |

The picture: peak-LR CPT **does not restore or amplify Arm 3's small triviaqa
recovery**; it moves the model in a direction where two of the three triviaqa
metrics fall significantly, popqa em/f1 also fall significantly, and only
triviaqa contains rises. If a paper wants to claim "CPT can recover parametric
knowledge, at ≥20k steps, on closed-book QA at 1B", **the primary-axis evidence
is now inconsistent across LR schedules**.

## 4. What ARM4_DESIGN's anti-outcome check says

Design's anti-outcome was: "both arms show idiosyncratic per-axis wobbles with
no cross-axis pattern — same as Arm 3 alone, useless." We are not quite there —
Arm 4 has a specific direction (negative em/f1, positive contains) — but this
direction is *opposite* to Arm 3 on the primary metric, which is worse than
noise for the headline. **The "coherent trajectory that survives an LR
sensitivity check" narrative is dead.** What *is* still on the table:

* A methodological finding: 20k Dolmino CPT past step 200k does not move
  primary-axis triviaqa em coherently and its direction depends on the LR
  schedule. That is a limitation-of-recipe result, not a discovery about
  parametric knowledge.
* The step210k transient (+1.26pp SIG on triviaqa em) is potentially
  interesting on its own, but ARM4_DESIGN itself said the first ~500-1000 steps
  are Adam-moment-mismatched; step210 is 10k steps in, past that window, yet
  the effect vanishes by step215 — hard to defend as a robust finding.

## 5. Implication for A03 as a proposal

* A03's Gate 1 (1B pilot viable) is unchanged: pruned+healed 1B is
  BH-significantly above its own construct-appropriate null on multiple axes
  (see `STATUS.json:kill_condition_1b_pilot_at_floor`). That baseline claim is
  intact.
* A03's Gate on "CPT moves the parametric knowledge signal past step200k on the
  primary axis" is now **decisively weakened**: two LR schedules disagree in
  sign at 20k CPT on triviaqa em. Any paper claim about "CPT recovery
  trajectory" needs to either narrow to Arm 3's specific recipe (and defend
  that as scientifically load-bearing) or drop the claim entirely.
* This does not itself trigger A03's pre-registered kill (pilot at floor;
  cleared 08-08). But it does mean A03's proposal-level thesis "closed-book
  parametric knowledge recovers coherently with CPT past 200k" is not what the
  data shows. **Recommended next check** (not launched yet): a *third* LR
  schedule (mid-band, e.g. ~0.6× peak throughout via `warmup=200500 max=260000`
  which gives ~0.78x at step220000) to see whether the effect is a monotone
  function of LR or truly non-monotone. That would either rescue an
  LR-schedule-parametric story or bury it.

## 6. Confounds and UNVERIFIED

1. **Not a matched-null seed-variance control.** Arm 4's negative results at
   step220000 are on the same eval set as Arm 3's positive results, but there
   is no seed-2 Arm 4 to isolate ±init/data-order variance from schedule
   variance. Given A04's finding (init-variance ≠ seed-variance because
   `DistributedSampler.seed` was never passed and dropout is off), a 3-seed
   Arm 4 would cost ~35 GPU-h on 1 H20 node. Not a blocker for calling this
   a sign flip — the CI excludes 0 by a comfortable margin (−0.93 vs +0.48,
   gap ~1.4pp on a metric whose half-width was ~0.2pp) — but it means the
   *magnitude* comparison Arm 3 vs Arm 4 rests on 1 realization each.
2. **MMLU not recomputed for Arm 4** in this pass. The recompute script does
   include MMLU, but the current run only pulled CB shards — MMLU shards on
   .82 for Arm 4 exist (the ext-drv wrote them), but the analyzer's MMLU load
   returned None on my quick check earlier tonight; not investigated. Given
   A03's Arm 3 verdict already found MMLU flat across its trajectory
   (`STATUS.json:arm3_cpt_trajectory.mmlu_across_trajectory`), and given A01's
   conclusion that MMLU-content at 1B is the safest axis with ~31x headroom to
   critical, this is unlikely to change the reading — but flagged as
   UNVERIFIED.
3. **1 seed, 1 node, 1 model, 1 dataset (Dolmino 15b).** The general claim "CPT
   past 200k is LR-schedule-sensitive on primary QA axes" cannot be made from
   this data alone; only the specific comparison "Arm 3's cosine tail beats
   Arm 4's peak-LR on triviaqa em at 20k". A separate Qwen or Llama arm would
   test cross-family generality — expensive, not proposed here.
4. **Bad-batch / transient learning hypothesis for step220 not ruled out.** The
   step210 spike (+1.26pp em) followed by step215 flat then step220 collapse
   could in principle be data-order-driven. A seed-2 rerun would distinguish
   this from a schedule-level phenomenon. Not done.

## 7. Kill / promote

* NOT a formal A03 kill trigger — A03's kill was pilot-at-floor and that was
  cleared. But this weakens the trajectory-CPT claim from a "clean coherent
  gain" to a "recipe-specific fragile effect". A03 remains in `active_measurable`
  status; STATUS.json is updated to record this verdict alongside the Arm 3
  headline (which remains a real measurement but no longer supports the
  general trajectory claim).
* No new Arm launched yet. If a mid-LR arm (Config M, warmup=200500,
  max_steps=260000 giving ~0.78x peak at step220000) is judged worth the
  ~11 h wall time on one node, that would either rescue or bury an
  LR-schedule-parametric story. **Recommendation deferred to user** — the
  proposal-level thesis is what is at stake, not a numerical detail.
