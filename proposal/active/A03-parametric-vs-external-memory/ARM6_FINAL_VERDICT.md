---
scope: A03 CPT-trajectory — FINAL verdict, all 4 dose points × 3 arms complete (task #235)
date: 2026-08-10 16:30 GMT+8
status: FINAL. Supersedes ARM6_STEP215_VERDICT.md, which called it one dose point early.
evidence: evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json (md5 28584639f120aaff07bd1a52120f983e)
protocol: per-item paired difference bootstrap, n_boot=5000, seed=42, CI95 percentile; SIG = CI excludes 0
---

# A03 final: Arm 6 DID replicate Arm 3 — and that makes the design's real defect worse, not better

## ⚠️ First: I corrected myself, and the correction matters

At 13:37 I wrote `ARM6_STEP215_VERDICT.md` declaring the trajectory-CPT claim
**retracted**, on the grounds that step215000 was null in all three arms and that
this "fired the pre-registered `Arm 6 nulls → Arm 3 was a fluke` branch."

**That was wrong on the pre-registration.** `ARM6_LOWERBAND_INTERIM.md` explicitly
committed to **step220000** as the decision point ("Only step215000 and step220000
decide this", and every branch is phrased `arm6_step220_*`). step215000 being null
in all arms was *predicted* by that same document — Arm 3 and Arm 4 were already
known to be null at step215 (−0.02 ns, −0.12 ns). I treated an expected
intermediate null as the verdict.

step220000 landed at 16:20 and evaluated by 16:27:36, 32/32 shards verified:

| arm | step205 | step210 | step215 | **step220** |
|---|---|---|---|---|
| Arm 3 (LR 0.325→0.249x) | −0.48 SIG | +0.37 SIG | −0.02 ns | **+0.48 SIG** |
| **Arm 6 (LR 0.499→0.425x)** | −0.73 SIG | +0.77 SIG | −0.06 ns | **+0.50 SIG** |
| Arm 4 (LR 0.998→0.559x) | −1.40 SIG | +1.26 SIG | −0.12 ns | **−0.93 SIG** |

**Arm 6 replicated Arm 3 to within 0.02pp (+0.50 vs +0.48), and all three
triviaqa metrics are SIG positive in both arms.** The pre-registered
`arm6_step220_positive_SIG` branch fired. My step215 retraction is **withdrawn**.

## But the replication is not independent — the arms are phase-locked

Before reading +0.48/+0.50 as confirmation, note what the full trajectories do:

**Pearson r between arm trajectories (4 dose points):**

| pair | r |
|---|---|
| Arm 3 vs Arm 6 (all 4) | **+0.9642** |
| Arm 3 vs Arm 4 (first 3, before Arm 4's data path broke) | **+0.9974** |
| Arm 6 vs Arm 4 (first 3) | **+0.9992** |
| Arm 3 vs Arm 4 (all 4, incl. broken step220) | +0.5304 |

The arms do not merely agree at step220 — they trace **the same curve**, dipping
and rising together at every dose point, at r ≈ 0.96–0.999. That is not three
experiments agreeing. Combined with the tcodex finding that
`.73:train_olmo2_arch_probe2.py:863` still lacks `seed=` so **all three arms
consumed the identical minibatch sequence** (training-loss correlation Arm3–Arm6
= 0.99982), the reading is:

> **The oscillation is a deterministic function of data order, not noise, and all
> three arms share that data order. So "Arm 6 replicates Arm 3" means "the same
> data prefix produces the same effect at the same step" — which it must.**

Arm 4 is the exception at step220 (−0.93), and Arm 4 is precisely the one arm whose
data path is known-broken: its truncated step220 ckpt was redone from step215
without restoring the dataloader offset, so its last 5k steps replayed the epoch
opening (original-vs-redo loss r = −0.0667). Its first 3 points, taken *before*
the break, track the others at r = 0.999.

**So the most parsimonious account of the whole 3-arm dataset is: data order
determines the trajectory; the one arm that diverges is the one whose data order
changed.** LR is not doing the work the design attributes to it.

## What this does and does not establish

**Does establish** (upgraded from my step215 call):
* At step220000, on this fixed data order, both low-LR arms show
  `triviaqa em +0.48/+0.50 SIG` with `contains` and `f1` also SIG positive. This
  is a real, reproducible-under-identical-data effect. It is not a fluke of one run.
* Arm 3's headline is no longer "the argmax of an oscillating series" in the sense I
  claimed — an adjacent LR band lands on the same value at the same step.

**Does NOT establish:**
* **That LR is the causal variable.** Arm 3 (0.249x) and Arm 6 (0.425x) differ by
  1.7× in LR and land within 0.02pp of each other. Arm 4 differs and is confounded
  by data path. A design that varies LR across arms while holding data order fixed
  cannot separate "LR effect" from "data-order effect at this step."
* **That the effect generalizes past this data order.** Every arm saw the same
  batches in the same sequence. Zero data-order replication exists.
* **Anything about the trajectory shape being meaningful.** The step210 spike and
  step215 null are shared across all arms at r≈0.99. They are properties of the
  data stream at those steps, not of the arms.
* **Arm 4's "peak-LR harms".** Still retracted — its step220 is data-path-broken.
  Retraction banner on `ARM4_PEAKLR_VERDICT.md` stands.

## Corrected status of my earlier claims

| earlier claim | status |
|---|---|
| "trajectory-CPT claim RETRACTED" (step215 verdict) | **WITHDRAWN — called too early.** step220 is the pre-registered point and it replicated. |
| "swing/CI = 2.4× median, 10.0× worst" | **stands as arithmetic**, but my *interpretation* ("bootstrap measures the wrong variance / tight CIs are false confidence") was wrong in an important way — the swing is not random, it is reproducible at r≈0.99 across arms. The CI is not mis-measuring noise; the oscillation is real signal driven by data order. |
| "7/12 arm-axis combos are sign-contradictory" | stands, and is now *explained*: the sign flips are shared across arms because they are data-order-driven, not independent errors. |
| "Arm3's +0.48 is the argmax of a series with mean +0.088" | true arithmetically, but Arm 6 independently landing +0.50 at the same step weakens "argmax-picking" as the explanation. |
| Arm 4 "peak-LR harms" retraction | **stands** (data-path defect, independent of all the above). |

## The one experiment that would settle it

Not more LR bands, and not more init seeds. **Vary the data order.**

The `seed=` fix (`ce5c298`) exists on wzc1 but was never scp'd to zwfy6, which is
why every arm here shares sampler seed 0. The decisive run is:

1. `scp -O` the fixed `train_olmo2_arch_probe2.py` to zwfy6 (verify md5).
2. Re-run **Arm 3's exact config** (warmup=150, max_steps=300000) at
   **3 different `--seed` values**, 20k steps each, from the same step200000 ckpt.
3. Evaluate step220000 only (the pre-registered point), 4 axes, same harness.

If `triviaqa em ≈ +0.48 SIG` at all 3 data orders → the CPT recovery effect is
real and A03 has a publishable narrow claim. If it scatters across
[−0.9, +1.3] → the effect is a data-order artifact and the retraction returns,
this time correctly grounded.

Cost: 3 runs × 20k steps × 2.05 s/step ≈ **11.4 h each**, runnable in parallel
across .73/.82/.104 → **one overnight cycle**. This is cheap and it is the only
measurement that discriminates the two live hypotheses. Queue it when the H20s free.

## Provenance

* Evidence: `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`,
  md5 `28584639f120aaff07bd1a52120f983e` (supersedes `36fed7ad…` which predated
  step220000). Regenerated by `code/recompute_cpt_trajectory_paired.py`, which
  hard-fails on incomplete shard sets; step220000 verified 8/8 on popqa, triviaqa,
  nq_open, mmlu before recompute.
* Watcher `.82` pid 2477303 completed all 4 Arm 6 dose points and exited cleanly at
  16:28:36 ("all 4 Arm6 MMLU summaries exist").
* Every number here computed from that JSON, not transcribed from prose.
* tcodex audit: `evidence/TCODEX_AUDIT_20260810.md` (md5 8ba6b5da…) — source of the
  Arm 4 dataloader-offset and sampler-seed findings.
