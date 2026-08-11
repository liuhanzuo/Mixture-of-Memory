---
scope: A03 seed 45 — the third pre-registered sampler seed, launched AFTER the ARTIFACT verdict. Scope and limits declared BEFORE the run.
date: 2026-08-11 11:20 GMT+8
status: PRE-DECLARATION (written before launch; no seed-45 number existed when this was committed)
prereg: DATAORDER_PREREG.md (commit `a25d780`, 2026-08-10 19:20:02 GMT+8) — seed 45 is inside the pre-registered set {43,44,45}, declared there in §5 before any seed's result was visible
verdict_already_closed: DATAORDER_VERDICT.md — branch ARTIFACT, 0/2 landed seeds CONFIRM
---

# Why this is not optional stopping

The obvious objection: *"you saw a negative result at n=2, and now you're adding a run."* That is exactly the pattern §3.5 item 4 of the prereg forbids. It does not apply here, and the reason is structural, not rhetorical:

1. **Seed 45 was pre-registered.** §5 of `DATAORDER_PREREG.md` names seeds 43/44/**45** as "the whole pre-registered set", and explicitly says seed 45 "counts only because it is declared *now*, not after the fact". It was queued for `.104` at prereg time and blocked only by a keep12 7B run holding all 8 GPUs.

2. **The forbidden move is adding a run to resolve a MIXED split.** §3.5 item 4: *"No tie-breaking seed may be added to resolve a MIXED result."* The landed outcome is not MIXED — it is ARTIFACT (0/2 CONFIRM). There is no split to break.

3. **Seed 45 cannot change the verdict in the favourable direction, and this is provable from the branch definitions.** REPLICATES requires *every* landed seed to be CONFIRM. Seeds 43 and 44 are both NOT-CONFIRM and are already on disk. So REPLICATES is unreachable at n=3 regardless of what seed 45 returns. The reachable outcomes are:
   * seed 45 NOT-CONFIRM → still **ARTIFACT** (0/3 CONFIRM). Verdict unchanged.
   * seed 45 CONFIRM → **MIXED** (1 CONFIRM, 2 NOT-CONFIRM), and §3.5 **pre-declares MIXED a FAILURE of the claim**, with the disposition *"the positive reading is retracted as a general claim, and the headline may not be the confirming seed"*.

   **Both reachable branches retract A-2.** There is no result seed 45 can produce that rescues the claim. That is what makes running it safe: I am not buying a lottery ticket on my own hypothesis.

# What seed 45 IS for

Three things, none of them verdict-changing:

1. **A third draw of run-to-run spread on the same arm.** §4 of the prereg forbids estimating σ_run at n=2 (1 d.o.f.; χ² 95 % interval for σ from the triviaqa value 0.3231 pp is roughly **[0.14, 10.3] pp** — useless). At n=3 the d.o.f. goes to 2. That is still a weak estimate, but it is the difference between "unusable" and "a bound with a stated width". This matters for **A04**, whose Stage-A `sd_run` input currently rests on n=2.

2. **The one empirical question the verdict raised and could not answer.** Seed 44 came back **−0.3455 SIG negative** on the primary axis. Two readings are consistent with n=2: (a) the effect is ~0 and one draw happened to land significantly below, or (b) 20k-step CPT at this LR is mildly *harmful* on TriviaQA and the original +0.4793 was the outlier. A third draw discriminates these — not for the retracted claim's sake, but because the sign of the CPT effect is an input to whether the 6-arm plan in A03's `next_gate` is worth any GPU at all.

3. **Testing whether the 16.5 %-subset framing predicts anything.** Each seed consumes a different 16.53 % of the 15,491,607-row epoch (20,000 steps × eff-bs 128 = 2,560,000 sequences; a full epoch is 121,028 steps). If the spread is driven by *which* data is seen rather than by optimisation noise, a third disjoint subset should land in the same wide range rather than converging.

# Pre-declared analysis rules (locked before the run)

* **Primary axis stays triviaqa `em`.** Same 17,944-item set, same paired bootstrap against the same base `A03_1B_keep7_step200k`, n_boot=5000, seed=42, CI95 percentile. Same `code/recompute_cpt_trajectory_paired.py`. No re-tuning.
* **CONFIRM_45 ⟺** CI excludes 0 **and** θ > 0 **and** θ ∈ [+0.20, +0.80] pp — the §3.3 rule verbatim, unchanged.
* **Aggregate verdict after seed 45 is mechanical**: 0/3 CONFIRM → ARTIFACT (status quo); 1/3 CONFIRM → MIXED → §3.5 disposition (also a retraction). Written out above so neither branch can be re-argued after the number lands.
* **σ_run at n=3 must be reported with its d.o.f. and its χ² interval**, never as a bare point estimate. The n=2 mistake was quoting 0.3231 pp as "the apparatus noise floor"; the fix is not a bigger n, it is always reporting the interval.
* **The 4 secondary axes** (popqa / nq_open / mmlu_content) are run for completeness and remain barred by §3.6 from rescuing *or* strengthening the primary.
* **Seed 45 is the last run under this prereg.** No seed 46. If n=3 leaves σ_run wide, that is the honest answer and it goes in the record as such.

# Operational

* Node: `.82` (8×H20, zwfy6 disk), verified idle at 11:18 (0 MiB × 8).
* Driver: `scripts/_run_a03_dataorder_repl.sh` (commit `44840f1`) — the same script that ran seeds 43 and 44, parameterised by `$SEED`. Not modified.
* Assets verified present on zwfy6 before launch: `data/dolmino_now15b.npy` = 126,907,244,672 B (the 15.5M-row zwfy6 file, NOT wzc1's 62 GB same-named file); base `outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt` = 12,181,310,078 B; trainer has the `ce5c298` `seed=args.seed` fix (2 matches).
* `outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed45/` did not exist before launch — no overwrite.
* Eval: the same watcher + ext-driver pair used for 43/44, with `SEEDS=45`. Result namespace `A03_1B_dataorder_seed45_step220000` (+ `_nq`), matching the ARMS table in `recompute_cpt_trajectory_paired.py`.
* Expected wall time: ~11 h at the 2.02 s/step median observed on seeds 43/44 (20,000 steps ÷ 8×H20).
