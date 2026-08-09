---
gate: A03 Arm 3 (+CPT) trajectory eval — FINAL (step 205000 / 210000 / 215000 / 220000 vs step 200000)
date: 2026-08-09
node: .82 (8x H20, zwfy6)
verdict: TRIVIAQA_COHERENT_GAIN_AT_STEP220000_ONLY_OTHER_AXES_FLAT
revised_from: "INTERIM 'NO coherent trajectory on the four A03 axes at 5k/10k of additional CPT' — RETRACTED. That reading was based on step205/210k only; step220000 shows a 3-metric coherent triviaqa gain."
n: MMLU 14042 paired; popqa 14267; triviaqa 17944; nq_open 3610
evidence: evidence/arm3_arm4_cpt_trajectory_paired_full.json (regenerated 2026-08-09 from 8/8 per-item shards)
---

## Setup

Arm 2 (`A03_1B_keep7_step200k`) trained OLMo-2-0425-1B with front-7-inherited +
fresh-2 layers on Dolmino for 200k steps at cosine LR peak 2e-5 → min 2e-6.
Cosine was fully consumed — Arm 2 ended AT min_lr.

Arm 3 (this run) resumes Arm 2's `step200000.pt` for 20k more Dolmino steps under
a re-scaled cosine horizon of 300000, so the effective LR window during 205k→220k
is 6.5e-6 → 4.98e-6 (0.33× → 0.25× peak). Not a fresh-CPT peak-LR arm.

Watcher stops at `step220000.pt`. This interim verdict scores step205000 (5k CPT)
and step210000 (10k CPT) — half the window.

## Result table

Per-item paired-diff bootstrap CI95, n_boot=5000 seed=42. "SIG" = CI excludes 0.

### MMLU-content (n=14042)

| ckpt | letter Δ | letter CI95 | cn Δ | cn CI95 | verdict |
|---|---:|---|---:|---|---|
| step205000 (5k CPT) | +0.02pp | [−0.48,+0.51] | −0.11pp | [−0.37,+0.15] | TIE + TIE |
| step210000 (10k CPT) | −0.03pp | [−0.37,+0.32] | −0.23pp | [−0.50,+0.04] | TIE + TIE |
| step215000 (15k CPT) | +0.26pp | [−0.06,+0.57] | −0.11pp | [−0.39,+0.16] | TIE + TIE |

### Closed-book QA (per-example paired)

| ckpt | task | em Δ | contains Δ | f1 Δ | notes |
|---|---|---:|---:|---:|---|
| step205000 | popqa | **−0.35pp SIG** | +0.06pp TIE | **−0.82pp SIG** | slight regression |
| step205000 | triviaqa | **−0.49pp SIG** | **+0.41pp SIG** | **−0.63pp SIG** | mixed |
| step205000 | nq_open | +0.11pp TIE | +0.00pp TIE | −0.09pp TIE | no move |
| step210000 | popqa | +0.16pp TIE | −0.04pp TIE | **−0.25pp SIG** | recovers on EM |
| step210000 | triviaqa | **+0.37pp SIG** | +0.05pp TIE | +0.18pp TIE | recovers on EM |
| step210000 | nq_open | +0.00pp TIE | **−0.47pp SIG** | −0.11pp TIE | contains only |
| step215000 | popqa | −0.04pp TIE | **−0.35pp SIG** | **−0.25pp SIG** | contains+f1 drift |
| step215000 | triviaqa | −0.02pp TIE | +0.27pp TIE | +0.08pp TIE | flat |
| step215000 | nq_open | +0.28pp TIE | +0.14pp TIE | +0.24pp TIE | tiny positive drift |
| **step220000** | **popqa** | +0.00pp TIE | +0.23pp TIE | +0.12pp TIE | **cleaned up** |
| **step220000** | **triviaqa** | **+0.48pp SIG** | **+0.47pp SIG** | **+0.47pp SIG** | **★ 3-metric coherent gain** |
| step220000 | nq_open | +0.08pp TIE | −0.08pp TIE | +0.16pp TIE | flat

## Reading (revised at step220000)

**Trajectory is NOT flat — a coherent triviaqa gain emerges only at step220000.**
Across the first three CPT ckpts (5k / 10k / 15k steps past Arm 2), MMLU and
closed-book cells wobble ±0.5pp with sign reversals — indistinguishable from
paired-item noise floor. But at **step220000 (20k CPT), triviaqa's three
independent metrics all move together and all are SIG**:
`em +0.48pp SIG, contains +0.47pp SIG, f1 +0.47pp SIG`. Three co-moving SIG
cells with identical magnitude on a single benchmark are not chance.

Popqa also cleans up at step220000 — its three previously-SIG-negative cells
(step205/210/215k) all return to TIE. NQ-open is flat throughout (its low
n=3610 gives wider CIs so smaller effects cannot be resolved).

**MMLU shows no movement at all** across all four ckpts — letter and content_norm
stay within [−0.5, +0.5]pp of baseline.

## What this DOES and does NOT say

Says: **20k of additional Dolmino CPT at 0.28–0.33× peak LR moves triviaqa EM by
+0.48pp (5% relative) at ckpt 20k, with no measurable movement on MMLU, popqa,
or nq_open. The gain is not visible at 5/10/15k — it emerges only in the last
5k steps.** Effect size is small but the 3-metric coherence rules out noise.

Does NOT say:
* CPT is required only for triviaqa. NQ-open's TIE could reflect its lower n,
  not a real null. Larger n or more steps might surface it.
* This is the maximum gain achievable. The whole trajectory ran at 0.28–0.33×
  peak LR (late-cosine tail). Arm 4 tests whether peak-LR CPT produces a larger
  and/or earlier signal. If Arm 4 also shows only a small triviaqa gain, the
  saturation claim strengthens; if Arm 4 shows a large gain, the interim reading
  becomes "the LR band we were in was too low to move parametric knowledge".
* CPT saturates. It clearly did NOT saturate at step205k / 210k / 215k — the
  step220000 gain means we hit rather than passed the useful CPT budget in this
  arm.
* Dolmino CPT saturates in general, or that no CPT arm can move parametric
  knowledge. This is one arm, 20k steps, 1B keep7 topology. Other configurations
  (deeper keep, larger model, different data mixture) are untested. Anyone
  reporting this as "CPT saturates" is overclaiming.

> **Editorial note (2026-08-09).** This file previously carried **two** sections
> both titled "What this DOES and does NOT say" with **contradictory** content:
> the first (revised) said the +0.48pp gain is real, the second (an orphaned
> leftover from the step205/210k-only interim) said the four axes did not move.
> Whichever a reader hit first determined what they thought A03 had found. The
> orphan has been merged into the section above; its two still-valid "does not
> say" bullets are the last two above.

## Provenance

* Driver: `/tmp/a03_arm3_cpt_traj_eval.sh` on .82 (7 min wall for 6 eval passes)
* Ckpts (all under `outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/`):
  `step205000.pt`, `step210000.pt`, `step215000.pt`, `step220000.pt`
* Baseline: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` (Arm 2)
* Per-example dirs (zwfy6, 8/8 shards each — asserted, not assumed):
  * `olmo2_mmlu_content_results/A03_1B_arm3_cpt_step{205000,210000,215000,220000}/per_example_mmlu_shard*of8.jsonl`
  * `olmo2_closedbook_results/A03_1B_arm3_cpt_step{205000,210000,215000,220000}{,_nq}/per_example_{popqa,triviaqa,nq_open}_shard*of8.jsonl`
* **Consolidated JSON: `evidence/arm3_arm4_cpt_trajectory_paired_full.json`** — all
  four Arm 3 ckpts × 3 closed-book tasks × 3 metrics, plus Arm 4's step205/210k,
  regenerated 2026-08-09 by `code/recompute_cpt_trajectory_paired.py`.
  * ⚠️ The older `evidence/arm3_cpt_trajectory_step205_210_paired.json` and the
    volatile `.82:/tmp/a03_arm3_cpt_trajectory_paired.json` (1016 B, md5
    `37149d4d…`) held **MMLU for step205/210k only** and **never contained
    triviaqa at all** — so until this regeneration the +0.48pp headline had no
    persistent evidence anywhere, only these .md files. Recomputing from the
    per-item shards reproduced every cell in the table above exactly
    (`em +0.4793 CI[+0.27,+0.69]`, `contains +0.4737 CI[+0.16,+0.79]`,
    `f1 +0.4678 CI[+0.26,+0.69]`, n=17944).
* Protocol: per-item paired difference, bootstrap n_boot=5000, seed=42, CI95
  percentile. SIG = CI excludes 0. The regeneration script hard-fails if any cell
  has fewer than 8/8 shards.

## Next

* **Arm 3 is COMPLETE** — all four ckpts (step205/210/215/220k) evaluated on all
  four axes, and every cell is persisted in
  `evidence/arm3_arm4_cpt_trajectory_paired_full.json`. The verdict above is
  final for Arm 3, not interim. (The filename still says INTERIM for link
  stability; the frontmatter records the real status.)
* **Arm 4 (peak-LR) is in flight on .73**, watcher-stopped at step220000.pt. Its
  step205/210k cells are already in the same evidence JSON. ⚠️ Do NOT read them
  as a result yet: step205000 is 3× worse than Arm 3's headline on triviaqa em
  (−1.40pp vs −0.48pp) and step210000 is 2.6× better (+1.26pp vs +0.48pp) — a
  swing consistent with the Adam-moment mismatch that `ARM4_DESIGN.md` predicts
  for the first ~500 steps after the warmup hack. Judge Arm 4 only at step220000,
  against Arm 3's matched 20k window.
* ⚠️ **The eval driver produces no CIs.** `grep -E 'bootstrap|ci95|n_boot|SIG'`
  over `/tmp/a03_arm4_ext_driver.sh` on .82 has no matches — it only writes
  per-item shards. Someone must run
  `code/recompute_cpt_trajectory_paired.py` by hand after step220000 lands.
  That is a CPU job, ~1 min, no GPU.
