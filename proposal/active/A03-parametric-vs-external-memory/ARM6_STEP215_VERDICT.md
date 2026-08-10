---
scope: A03 CPT-trajectory dose-response — verdict after Arm 6 step215000 (task #235)
date: 2026-08-10
status: DECIDED — the trajectory-CPT claim is RETRACTED. Not "narrowed", retracted.
evidence: evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json (md5 36fed7ad8cce952c2c406c4abad80da7)
protocol: per-item paired difference bootstrap, n_boot=5000, seed=42, CI95 percentile; SIG = CI excludes 0
supersedes: ARM6_LOWERBAND_INTERIM.md (which correctly refused to call step210 a result)
---

# A03 CPT trajectory: the dose-response design cannot support any conclusion

## What the pre-registered outcomes said

`ARM6_LOWERBAND_INTERIM.md` committed in advance to three readings:

* Arm 6 step220 positive SIG → low-LR is a real regime, keep a narrow claim.
* Arm 6 step220 null/negative → **the trajectory-CPT claim dies.**
* Arm 6 step220 ≫ Arm 3 → non-monotone in LR, would need a 4th arm + seeds.

step215000 has now landed and it settles the question **before** step220 arrives,
because it exposes a defect in the measurement, not just an unfavourable value.

## step215000: all three arms are null on the headline axis

| axis | Arm 3 | **Arm 6** | Arm 4 |
|---|---|---|---|
| triviaqa em | −0.02 ns | **−0.06 ns** | −0.12 ns |
| triviaqa contains | +0.27 ns | **+0.19 ns** | −0.12 ns |
| triviaqa f1 | +0.08 ns | **−0.03 ns** | −0.23 ns |
| popqa em | −0.04 ns | **−0.18 SIG** | −0.40 SIG |
| popqa f1 | −0.26 SIG | **−0.35 SIG** | −0.46 SIG |
| nq_open em | +0.28 ns | **+0.25 ns** | +0.28 ns |

## The full trajectory on triviaqa EM — A03's headline axis

| arm | step205 | step210 | step215 | step220 | swing | mean |
|---|---|---|---|---|---|---|
| Arm 3 | −0.48 SIG | +0.37 SIG | −0.02 ns | **+0.48 SIG** | 0.96pp | **+0.088** |
| Arm 6 | −0.73 SIG | +0.77 SIG | −0.06 ns | (pending) | 1.50pp | −0.004 |
| Arm 4 | −1.40 SIG | +1.26 SIG | −0.12 ns | −0.93 SIG | 2.66pp | −0.298 |

**Every arm oscillates in sign. Arm 3 changes sign 3 times in 4 dose points.**
A dose-response design assumes the response is a function of the dose; here the
measured quantity is not even monotone in its own trajectory, in any arm.

## Why this kills the claim rather than narrowing it

### 1. The within-arm swing is several times the bootstrap uncertainty

Across all 27 (arm × task × metric) combinations:

* **median swing / mean CI half-width = 2.4×**
* **52 % of combinations exceed 2×**
* worst: Arm 4 triviaqa.em at **10.0×** (swing 2.66pp vs half-width 0.27pp)
* the headline axis, Arm 3 triviaqa.em: **4.7×** (swing 0.96pp vs half-width 0.21pp)

The bootstrap CI answers "what if I resampled eval items". The step-to-step swing
is 2–10× larger, so the dominant variance component is **not item sampling** — it
is whatever moves between adjacent checkpoints 5000 steps apart. The CIs are
therefore a measure of the wrong thing, and their tightness has been giving false
confidence.

### 2. Seven of twelve arm-axis combinations are internally sign-contradictory

Among arm-axis combinations with ≥2 SIG cells, **7/12 contain two SIG cells of
opposite sign**:

| arm | axis | contradicting SIG cells |
|---|---|---|
| Arm 3 | triviaqa.em | 205=−0.48, 210=+0.37, 220=+0.48 |
| Arm 3 | triviaqa.f1 | 205=−0.63, 220=+0.47 |
| Arm 6 | triviaqa.em | 205=−0.73, 210=+0.77 |
| Arm 6 | triviaqa.f1 | 205=−0.85, 210=+0.60 |
| Arm 4 | popqa.em | 205=−0.97, 210=+0.20, 215=−0.40, 220=−0.97 |
| Arm 4 | triviaqa.em | 205=−1.40, 210=+1.26, 220=−0.93 |
| Arm 4 | triviaqa.f1 | 205=−1.19, 210=+1.00, 220=−0.82 |

A significance procedure that stamps SIG on **both** +0.77 and −0.73 for the same
arm and axis is not identifying an effect. It is resolving noise precisely.

### 3. The headline was a trajectory endpoint, not an effect

Arm 3's `+0.48pp SIG` at step220000 is one of four dose points whose **mean is
+0.088pp**. Choosing step220000 as "the result" is choosing the argmax of an
oscillating series. Had the trajectory been stopped at step215000 the same arm
would have read −0.02 ns; at step205000, −0.48 SIG (*harm*). The reported sign of
A03's headline is determined by where the trajectory happened to be truncated.

## Verdict

**RETRACTED: "20k-step Dolmino CPT recovers parametric knowledge in pruned+healed
1B, in a low-LR band."** The evidence cannot distinguish this from zero, and the
apparatus that produced its p-values is measuring item-resampling variance while
the real variance is 2–10× larger.

**This supersedes and retracts:**
* `ARM4_PEAKLR_VERDICT.md`'s "peak-LR CPT actively harms" — Arm 4's −0.93 SIG at
  step220 is one point in a series that also contains +1.26 SIG. Same defect,
  opposite sign. It was never a finding.
* The step205000 "damage is monotone in LR" reading in
  `ARM6_LOWERBAND_INTERIM.md`. The ordering Arm3 < Arm6 < Arm4 does hold at
  step205000 on 4 axes, and it is the one pattern here with a mechanism
  (larger LR → larger early overshoot). But it is a **single dose point** in a
  series that reverses at the next one, so it is at most suggestive.

**What A03 keeps**: only its Gate-1 pilot result — pruned+healed 1B (keep7+fresh2
@200k) sits BH-significantly above its own construct-appropriate null on 4/5
knowledge interfaces. That result is a *level* measured at one checkpoint against
a computed floor, not a *difference* between adjacent checkpoints, so it does not
inherit this defect.

## Should step220000 still be evaluated?

Yes — it is ~1.5 h of already-committed compute and it completes the series
symmetrically. But it is now **confirmatory of the defect, not decisive of the
claim**: no value it takes can rescue a dose-response whose response oscillates.
If it lands positive SIG, that is the 4th sign change in Arm 6, not a
replication of Arm 3.

## What would be needed to ask this question properly

The blocker is not more dose points — it is that **run-to-run and
checkpoint-to-checkpoint variance were never measured**. Minimum viable design:

1. **≥3 seeds per LR band**, with the `DistributedSampler seed=` fix (`ce5c298`)
   actually in effect, so data order differs and the variance estimate is real.
   Note this repo's prior "seed variance" arms were init-variance only.
2. **A null arm: CPT on the same data at the same LR for the same steps, twice.**
   The spread between two identical runs is the floor every arm difference must
   clear. Nothing here has that.
3. **Average over a checkpoint window**, not a single step — with a swing of
   0.96pp between adjacent checkpoints, single-step readings are not estimates of
   the arm.
4. Pre-register the dose point *before* seeing the trajectory, or report the
   whole trajectory mean with a trajectory-level CI.

Cost estimate: 3 seeds × 3 bands × 20k steps ≈ 9 runs. At Arm 6's measured
2.05 s/step that is ~11.4 h/run on 8×H20, so ~4.3 node-days. That is affordable,
but it should only be spent if the question is worth it — and given that the
best estimate of the effect is +0.088pp on a floor-0.0026 benchmark, the honest
prior is that there is nothing here to find.

## Provenance

* Evidence: `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`,
  md5 `36fed7ad8cce952c2c406c4abad80da7`. Regenerated by
  `code/recompute_cpt_trajectory_paired.py`, which hard-fails on any incomplete
  shard set (step215000 verified 8/8 for popqa, triviaqa, nq_open, mmlu before
  recompute).
* All numbers in this file were computed directly from that JSON, not
  transcribed from prose.
* Arm 6 training continues on .73 (`step 215180/373000` at 13:35 GMT+8); the
  .82 watcher (pid 2477305) will fire step220000 automatically.
