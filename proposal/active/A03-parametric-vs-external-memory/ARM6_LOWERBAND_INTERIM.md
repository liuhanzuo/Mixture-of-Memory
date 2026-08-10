---
scope: A03 Arm 6 (mid-low-LR CPT) — running interim record, task #235
date: 2026-08-10
status: IN_PROGRESS — 2 of 4 dose points evaluated. NOT a verdict. Do not cite step205/210 as a result.
run: outputs/olmo2_probe2_1B_keep7f2_dolmino_arm6_lowerband20k on .73 (zwfy6), launched 04:56:33 GMT+8
config: warmup_steps=150, max_steps=373000 → LR band [0.499x, 0.425x] peak across steps 200000–220000
eval_node: .82 (zwfy6), watcher pid 2477305, same 4-axis ext-drv as Arm 3 / Arm 4
evidence: evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json (md5 f92fe250cbd38061d8cbc460be73d70f, same on both disks)
protocol: per-item paired difference bootstrap, n_boot=5000, seed=42, CI95 percentile; SIG = CI excludes 0
---

> ## ✅ SUPERSEDED 2026-08-10 — see `ARM6_STEP215_VERDICT.md`
>
> This interim record was correct to refuse to call step210000 a result. step215000
> landed at 13:37 GMT+8: **all three arms null** on triviaqa em (Arm3 −0.02, Arm6
> −0.06, Arm4 −0.12, all ns), firing the pre-registered 'Arm 6 nulls → Arm 3 was a
> fluke' branch. The trajectory-CPT claim is **retracted**. The step205000
> 'damage monotone in LR' reading survives only as weakly suggestive (one dose
> point, reverses at the next).

# A03 Arm 6 (mid-low-LR CPT) — interim record

## Why this arm exists

Arm 3 (LR band [0.325x, 0.249x] peak) produced `triviaqa em +0.48pp SIG` at
step220000. Arm 4 (LR band [0.998x, 0.559x]) **sign-flipped** it to
`-0.93pp SIG` at the matched 20k window (see `ARM4_PEAKLR_VERDICT.md`). Two
readings survived that:

* (i) **low LR is a real regime** — Arm 3's gain is genuine within a low-LR
  band, and peak-LR CPT actively harms. Then an arm just *above* Arm 3's band
  should also show the gain.
* (ii) **Arm 3 is a fluke** — a single realization that happened to land
  +0.48 SIG. Then an adjacent band should NOT reproduce it.

Arm 6 sits at [0.499x, 0.425x] — between Arm 3 and Arm 4, closer to Arm 3 —
and discriminates (i) from (ii).

## LR ladder (all three arms, verified from `get_lr` and confirmed in logs)

| arm | warmup | max_steps | LR @200000 | LR @220000 | band |
|---|---:|---:|---:|---:|---|
| Arm 3 | 150 | 300000 | 6.50e-06 | 4.98e-06 | [0.325x, 0.249x] |
| **Arm 6** | **150** | **373000** | **9.98e-06** | **8.50e-06** | **[0.499x, 0.425x]** |
| Arm 4 | 200500 | 240000 | 1.995e-05 | 1.118e-05 | [0.998x, 0.559x] |

Arm 3 and Arm 6 share `warmup=150`, so **neither** carries Arm 4's
Adam-moment mismatch (Arm 4 restored peak LR onto moments adapted to a
min_lr trajectory). Arm 3 vs Arm 6 is therefore the cleaner pair.

## Dose points measured so far

### step205000 (5k CPT) — evaluated 07:55:28

| axis | Arm 3 | **Arm 6** | Arm 4 |
|---|---|---|---|
| triviaqa em | −0.48 SIG | **−0.73 SIG** | −1.40 SIG |
| triviaqa contains | +0.41 SIG | **+0.48 SIG** | −0.08 ns |
| triviaqa f1 | −0.63 SIG | **−0.85 SIG** | −1.19 SIG |
| popqa em | −0.35 SIG | **−0.59 SIG** | −0.97 SIG |
| popqa contains | +0.06 ns | **+0.06 ns** | +0.06 ns |
| popqa f1 | −0.82 SIG | **−1.08 SIG** | −1.44 SIG |
| nq_open em | +0.11 ns | **−0.17 ns** | −0.25 ns |

**Damage is monotone in LR magnitude on all four SIG axes** (triviaqa em/f1,
popqa em/f1), with Arm 6 sitting exactly between Arm 3 and Arm 4. Since Arm 3
and Arm 6 have no moment mismatch, this is a genuine **early-CPT overshoot**
that scales with LR — not a warmup artifact.

### step210000 (10k CPT) — evaluated 10:46:26

| axis | Arm 3 | **Arm 6** | Arm 4 |
|---|---|---|---|
| triviaqa em | +0.37 SIG | **+0.77 SIG** | +1.26 SIG |
| triviaqa contains | +0.04 ns | **+0.04 ns** | −0.22 ns |
| triviaqa f1 | +0.18 ns | **+0.60 SIG** | +1.00 SIG |
| popqa em | +0.16 ns | **+0.08 ns** | +0.20 SIG |
| popqa contains | −0.04 ns | **−0.12 ns** | +0.03 ns |
| popqa f1 | −0.25 SIG | **−0.22 SIG** | +0.03 ns |
| nq_open em | +0.00 ns | **+0.08 ns** | +0.28 ns |

Arm 6's `+0.77 SIG` at 10k CPT **already exceeds Arm 3's step220000 headline
of +0.48 SIG** by +0.29pp, with 10k steps still to run.

## ⚠️ Why step210000 is NOT evidence for Arm 6

**All three arms show a positive step210000 excursion, and in two of the three
it does not survive:**

| arm | step205 | step210 | step215 | step220 |
|---|---|---|---|---|
| Arm 3 | −0.48 SIG | **+0.37 SIG** | −0.02 ns | +0.48 SIG |
| Arm 4 | −1.40 SIG | **+1.26 SIG** | −0.12 ns | **−0.93 SIG** ← collapse |
| **Arm 6** | −0.73 SIG | **+0.77 SIG** | ? | ? |

Arm 4 is the cautionary case: it had the *largest* step210 excursion
(+1.26 SIG, ~2.6× Arm 3's final headline) and then ended significantly
**negative**. So the step210k positive is a **shared non-monotone feature of
this CPT trajectory**, not an arm-specific signal. Ranking arms by their
step210 value would have picked Arm 4 as the winner — exactly backwards.

**Only step215000 and step220000 decide this.** step215 ETA ~12:19 GMT+8,
step220 ETA ~13:00 GMT+8.

## What each outcome will mean

* **Arm 6 step220 positive SIG** → low-LR is a real regime (Arm 3 + Arm 6
  agree, Arm 4 dissents). A03 keeps a **narrow** claim: "CPT recovers
  triviaqa EM only in a low-LR band; peak-LR CPT actively harms." Weaker than
  the original trajectory claim but defensible, and the LR-band boundary
  becomes the contribution.
* **Arm 6 step220 null or negative** → Arm 3's +0.48 SIG failed to replicate
  at an adjacent band. **The trajectory-CPT claim dies.** A03 retains only its
  Gate-1 pilot result (pruned+healed 1B is BH-significantly above its own
  construct-appropriate null on 4/5 knowledge interfaces).
* **Arm 6 step220 substantially larger than Arm 3's +0.48** → implies a
  non-monotone-in-LR effect peaking inside [0.425x, 0.499x]. Interesting, but
  locating an optimum from three single-seed arms is not defensible; would
  need a 4th arm plus seed replication.

## Confounds and UNVERIFIED

1. **n=1 seed per arm.** Every Arm 3 / Arm 4 / Arm 6 cell is one realization.
   Given the DistributedSampler `seed=` defect fixed in `ce5c298` (seed arms
   in this repo were init-variance, not data-order variance), there is
   currently **no** measured seed-variance floor for this trajectory. The
   Arm 3-vs-Arm 4 sign flip is large relative to the bootstrap CIs (gap ~1.4pp
   vs half-widths ~0.2pp) so it is unlikely to be seed noise, but the
   *magnitude* comparisons across arms rest on one run each.
2. **MMLU not tabulated here.** The ext-drv writes MMLU shards and the
   recompute script supports MMLU, but the current pass reports CB axes only.
   Arm 3's verdict already found MMLU flat across its whole trajectory
   (letter Δ ∈ [−0.23, +0.26]pp all TIE, content Δ ∈ [−0.24, −0.11]pp all
   TIE), so this is unlikely to change the reading — flagged as UNVERIFIED.
3. **The step205 monotonicity is 2 points of LR wide.** "Damage monotone in
   LR" is asserted from three bands; it is consistent, not established as a
   functional form.
4. **Single model / single dataset** (OLMo-2-1B keep7+fresh2, Dolmino 15b).
   No cross-family check. Any claim must stay scoped to this setting.

## Provenance

* Evidence: `evidence/arm3_arm4_arm6_cpt_trajectory_paired_full.json`,
  md5 `f92fe250cbd38061d8cbc460be73d70f`, verified identical on wzc1 and
  zwfy6. Regenerated from 8/8 per-item shards by
  `code/recompute_cpt_trajectory_paired.py`, which hard-fails on any
  incomplete shard set.
* Driver: `scripts/_run_a03_arm6_lowerband.sh` (commit `8e5b67d`).
* Eval: `code/a03_arm6_ext_driver.sh` + `code/a03_arm6_trajectory_watcher.sh`
  (v3 guard: size within 64 KiB of a known-good sibling, size stable across
  two probes 60 s apart, mtime age ≥ 120 s, plus an internal `torch.load`
  probe in the ext-drv).
* Commits: `8e5b67d` launch, `2fd5dc0` status, `f44a5fb` step205000,
  `028455d` step210000.
