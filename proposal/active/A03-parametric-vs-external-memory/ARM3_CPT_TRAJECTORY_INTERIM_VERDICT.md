---
gate: A03 Arm 3 (+CPT) trajectory eval — interim (step 205000 + step 210000 vs step 200000)
date: 2026-08-09
node: .82 (8x H20, zwfy6)
verdict: INTERIM — NO coherent trajectory on the four A03 axes at 5k/10k of additional CPT
n: MMLU 14042 paired; popqa 14267; triviaqa 17944; nq_open 3610
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

## Reading

**No coherent trajectory across 3 measured points (5k / 10k / 15k CPT).** MMLU
letter and cn stay within ±0.5pp on every ckpt; closed-book cells wobble
±1pp with sign reversals between steps (popqa em: −0.35 → +0.16 → −0.04; triviaqa
em: −0.49 → +0.37 → −0.02). Only popqa `contains` and popqa `f1` are SIG
negative at BOTH step210000 and step215000, and only by 0.25–0.35pp. The pattern
is "flat with paired-item structure noise", not a trajectory.

**MMLU shows no movement at all** — letter and content_norm both within [−0.5,
+0.5]pp of baseline on all three ckpts.

## What this DOES and does NOT say

Says: **at 0.28–0.33× peak LR, 5–10k of additional Dolmino steps past Arm 2's
apex does not measurably move A03's four certified axes.**

Does NOT say:
* Dolmino CPT saturates in general. The LR window here is a late-cosine tail,
  not a fresh phase at peak LR. A separate peak-LR CPT arm (with new warmup)
  would test that claim. Anyone reporting this as "CPT saturates" is
  overclaiming.
* No CPT arm can move parametric knowledge. This is one arm, 20k steps, 1B
  keep7 topology. Other configurations (deeper keep, larger model, different
  data mixture) are not tested.

## Provenance

* Driver: `/tmp/a03_arm3_cpt_traj_eval.sh` on .82 (7 min wall for 6 eval passes)
* Ckpts:
  * `outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/step205000.pt`
  * `outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k/step210000.pt`
* Baseline: `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` (Arm 2)
* Per-example dirs:
  * `olmo2_mmlu_content_results/A03_1B_arm3_cpt_step{205000,210000}/per_example_mmlu_shard*of8.jsonl`
  * `olmo2_closedbook_results/A03_1B_arm3_cpt_step{205000,210000}{,_nq}/per_example_{popqa,triviaqa,nq_open}_shard*of8.jsonl`
* Consolidated JSON: `evidence/arm3_cpt_trajectory_step205_210_paired.json`

## Next

* step215000 and step220000 evals will be run when the .73 CPT trainer emits
  them (ETA ~5h from watcher stop). Same driver, incremental.
* If step220000 also shows the same "no coherent trajectory" pattern, the
  interim verdict above becomes the final Arm 3 verdict.
