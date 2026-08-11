---
scope: A03 data-order replication of the trajectory-CPT claim — VERDICT
date: 2026-08-11 04:53 GMT+8 (corrected 2026-08-11 11:05 GMT+8 after self-audit — see "Corrections" at the bottom)
status: ARTIFACT (the pre-registered branch name, DATAORDER_PREREG.md §3.4). Zero landed seeds are CONFIRM, so the Arm 3 step220000 headline (+0.4793 SIG on triviaqa em) is a sampler-seed artifact, not a real effect.
prereg: DATAORDER_PREREG.md, added by commit `a25d780` (2026-08-10 19:20:02 GMT+8) — band [+0.20, +0.80]pp on triviaqa em; branches enumerated in §3.4; MIXED pre-declared a FAILURE in §3.5
evidence: evidence/pilot_one_stage_a_verdict.json (md5 4ced4582cce6772a797a7f41e94e2a7a); per-example shards at olmo2_closedbook_results/A03_1B_dataorder_seed{43,44}_step220000/
seeds_pinned_pre_data: 43 (on .82), 44 (on .73). Seed 45 was queued for .104 but never run — per §4 the ARTIFACT branch is fully decidable at n=2.
supersedes: `A-2 (provisional)` in claims/A03_SURVIVING_CLAIMS.md — that claim's step220000 SIG is now known sampler-seed-dependent and must be retracted.
---

# The result

Both seeds, paired against the same intact-pruned+healed base (A03_1B_keep7_step200k), on the exact same 17,944-item TriviaQA-EM set, n_boot=5000, seed=42, CI95 percentile — the pre-registered protocol, unchanged:

| seed | Δ (arm − base), pp | CI95 (pp)              | verdict | in band [+0.20, +0.80]? | §3.3 class |
|------|--------------------|------------------------|---------|-------------------------|------------|
| 43   | **+0.1115**        | [−0.0947, +0.3177]     | TIE     | **NO** (below floor)    | NOT-CONFIRM |
| 44   | **−0.3455**        | [−0.5517, −0.1393]     | **SIG** | **NO** (below zero)     | NOT-CONFIRM |

Under the ORIGINAL sampler seed (Arm 3, trained pre-`ce5c298`, i.e. sampler seed 0), the same measurement returned +0.4793 SIG (`ARM6_FINAL_VERDICT.md`). Under two different sampler seeds it returns +0.11 (TIE) and −0.35 (SIG negative). **The headline effect flips sign when the sampler seed changes.**

# Which pre-registered branch fires, and why

`DATAORDER_PREREG.md` §3.4 enumerates exactly three aggregate outcomes, keyed on the per-seed CONFIRM/NOT-CONFIRM classification of §3.3:

| outcome | condition | fires here? |
|---|---|---|
| REPLICATES | every landed seed is CONFIRM | no — neither seed is CONFIRM |
| **ARTIFACT** | **zero** landed seeds are CONFIRM | **YES** |
| MIXED | ≥1 CONFIRM and ≥1 NOT-CONFIRM | no — there are no CONFIRM seeds |

**The verdict is ARTIFACT.** §3.4's disposition for that branch, verbatim: *"Effect is a data-order artifact. `ARM6_FINAL_VERDICT.md`'s positive reading is RETRACTED, this time grounded. A03 retains only its Gate-1 pilot."*

Seed 44's −0.35 SIG is a stronger empirical picture than ARTIFACT strictly requires (a sign flip rather than a null), but ARTIFACT is the branch, and the branch is what the disposition attaches to. There is no separate stronger branch to escalate into.

# What this means, precisely — and does NOT mean

**Does mean:**
1. **A03's trajectory-CPT claim is retracted.** The "20k-step Dolmino CPT recovers parametric knowledge at step220000" reading is sampler-seed noise, not a real effect. `claims/A03_SURVIVING_CLAIMS.md` §A-2 must be retracted.
2. **`ARM6_FINAL_VERDICT.md`'s interpretation is overturned.** That doc argued the Arm 3–Arm 6 replication at r≈0.99 across arms *validated* the +0.48 signal by showing it was "reproducible, not random". The correct reading was the opposite: it was reproducible **because the shared minibatch sequence made the runs deterministic**, and once the sampler seed varies the "signal" evaporates.

**Does NOT mean:**
1. **A04 is dead by this.** A04's Pilot Zero finding — that PLATEAU accepts a recovery run where NI(Δ=10%·residual) rejects on 3/3 axes — is measured at a SINGLE fixed checkpoint (`keep7+fresh2 step200000`) against intact, and does **not** use the Arm 3 trajectory at all. Those two proposals share only the training apparatus, not the claim.
2. **All CPT-recovery is dead.** This falsifies one specific claim (a 4.5pp-scale effect at one checkpoint under one arm). It does not say anything about larger effects, deeper arms, or longer schedules. It says nothing about whether the run-to-run spread at keep12/keep10 is different.
3. **The Arm 3–Arm 6–Arm 4 phase-locking analysis was wrong.** The phase-locking claim itself remains true and important: with sampler seed 0, all three arms consumed byte-identical minibatch sequences, and their headline agreement was engineered, not scientific. This ARTIFACT verdict is *consistent with* the phase-locking finding — it just draws the correct conclusion from it.
4. **The apparatus noise floor is now known.** ⚠️ It is NOT. See "Corrections" #3 — this doc previously claimed a ~0.3pp floor, which §4 of the prereg explicitly forbids estimating at n=2. Retracted.

# What the manipulation actually varied (corrected characterisation)

Calling this a "data order" manipulation is imprecise, and the imprecision matters. Both seeds ran **20,000 steps at effective batch 128 = 2,560,000 sequences**, against a `dolmino_now15b.npy` of **15,491,607 rows**. A full epoch is 121,028 steps. So each run consumed **16.53 % of one epoch**.

Changing the `DistributedSampler` seed therefore changes **which 16.5 % subset of the corpus is seen**, not merely the order in which a fixed set is seen. The two seeds trained on largely *different data*, with only incidental overlap.

This does not weaken the ARTIFACT verdict — if anything it makes the original Arm 3 result more clearly non-generalisable, since that result is now known to be specific to one particular 16.5 % slice. But every downstream statement must say **"sampler-seed / data-subset variation"**, not "data-order variation only". The prereg's own title (`DATAORDER_PREREG.md`) carries the same imprecision; the filename stays for provenance, the interpretation does not.

# The A04 side effect

The same data (via `code/pilot_one_stage_a_sd_run.py`, commit `49e665d`) simultaneously satisfied A04's Pilot One Stage A prereg (commit `2ac0b5a`). Applying that pre-registered rule verbatim:

    sd_run(triviaqa) = |9.6968 − 9.2399| / √2 = 0.3231 pp
    bound_3(triviaqa) = 2.920 · 0.3231 / √3 = 0.5448 pp
    Δ_triviaqa = 4.043 pp
    → bound_3 << Δ on all 3 decision axes → STAGE_A_DOES_NOT_FIRE

But Stage A **cannot clear K2** by prereg (it is one-directional), and see `A04-recovery-certification/STAGE_B_DECISION.md` for the follow-up.

# What was locked in advance vs decided after seeing data

Everything the verdict rests on was committed BEFORE the numbers existed:

| commit  | date (GMT+8)       | what              |
|---------|--------------------|-------------------|
| `a25d780` | 2026-08-10 19:20:02 | **DATAORDER_PREREG.md** (band, protocol, §3.3 classification, §3.4 branches, §3.5 MIXED-is-FAILURE, §4 n=2 limits) |
| `2ac0b5a` | 2026-08-10 23:50 | PILOT_ONE_PREREG.md (Stage A/K2 rule) |
| `49e665d` | 2026-08-10 (later) | Stage-A driver (imports canonical loaders, positive stale-copy guard) |

The seed 43/44 checkpoints landed 2026-08-11 04:19–04:21; the eval watchers auto-fired 04:23; both evals completed 04:27–04:29; MAIN ran the driver at 04:51. Nothing about the rule changed after the numbers were visible.

Timing audit (2026-08-11): the earliest dataorder checkpoint on disk is 19:47:23, **27 minutes after** the prereg commit at 19:20:02. Seed 45 genuinely never ran, so there is no optional-stopping concern.

# The 4-axis table — descriptive only, NOT part of the verdict

**UPDATE 2026-08-11 05:23, scope corrected 11:05.** The 4-axis paired table below was filled in after the verdict (CPU-only, canonical loaders on zwfy6). Per prereg §3.6 the primary endpoint is **triviaqa em alone**; popqa / nq_open / mmlu are **secondary, "reported for completeness"**. §3.6 forbids using a secondary axis to *rescue* a failed primary; the symmetric restriction applies to using one to *strengthen* an already-decided verdict. So this table is **descriptive context, not evidence the verdict rests on**:

| axis          | seed 43 Δpp | CI95              | sig | seed 44 Δpp | CI95              | sig | shape                            |
|---------------|:-----------:|-------------------|:---:|:-----------:|-------------------|:---:|----------------------------------|
| triviaqa em (**PRIMARY**) | **+0.1115** | [−0.0947, +0.3177] | TIE | **−0.3455** | [−0.5517, −0.1393] | **SIG** | sign-flip; seed 44 significantly negative |
| popqa em (secondary)     | **+0.1612** | [+0.0210, +0.3084] | **SIG** | **−0.2243** | [−0.3855, −0.0699] | **SIG** | two SIG cells of opposite sign |
| nq_open em (secondary)   | +0.0554     | [−0.2493, +0.3601] | TIE | +0.0554     | [−0.2493, +0.3601] | TIE | both TIE (n=3610, note identical numbers — resolved below) |
| mmlu_content (secondary) | −0.2065     | [−0.4702, +0.0571] | TIE | −0.2421     | [−0.5056, +0.0214] | TIE | both TIE, both mildly negative |

popqa's two opposite-signed SIG cells are the most visually striking pattern in the table, and they are consistent with the primary verdict. But popqa is secondary and the verdict does not depend on it — the earlier version of this doc called popqa "the more damning axis" and promoted it into the argument, which the prereg does not permit. Corrected.

MMLU-content behaves normally (both TIE, both mildly negative), consistent with the trajectory result that "MMLU is flat" — that observation survives.

**One nq_open anomaly, flagged not dismissed:** both seeds report *bitwise-identical* delta_pp = 0.05540166… and CI95 = [−0.2493, +0.3601], despite completely different weights.

**Resolved 2026-08-11 05:26.** Opening the per-example shards:
- byte-level predictions match on only **1583/3610** items (43.9 %) — the two models genuinely diverge on ~2000 items
- em-flag matches on **3570/3610** items (98.9 %) — the disagreements are within the "both wrong" region where em is 0 for both
- both models score exactly **105/3610 correct**, and the 40 items where em disagrees happen to cancel

So the identical means and CIs are a real coincidence on a small n and a heavily-zero score distribution, not a caching artifact or bug.

# What to do now

1. Retract A-2 from `claims/A03_SURVIVING_CLAIMS.md` and add this file to the ledger. **(done 2026-08-11 11:05 — see that file's A-2 entry)**
2. Update `STATUS.json.dataorder_replication.status` to `ARTIFACT` and `.arm6_midlowLR_cpt.claim` to reference the ARTIFACT branch. **(done 2026-08-11 11:05)**
3. Decide on A04 Stage B — see `A04-recovery-certification/STAGE_B_DECISION.md`.

---

# Corrections (self-audit 2026-08-11 11:05 GMT+8)

An audit of this document against `DATAORDER_PREREG.md` found five defects in the 04:53 version. All are fixed above; recorded here so the error is not silently rewritten out of history.

1. **Wrong prereg commit hash.** The header cited `44840f1`, and the body cited "§2.2". `44840f1` (2026-08-10 17:00:39) adds **only** `scripts/_run_a03_dataorder_repl.sh` — it does not contain the prereg, and its commit message pre-registers a *different* band ([+0.3,+0.7]). The prereg was added by **`a25d780` (19:20:02)**, and the branch enumeration is in **§3.4**, not §2.2. The pre-registration timing is nonetheless legitimate — see the timing audit above.

2. **"FALSIFIED" is not a pre-registered branch name.** `grep -c FALSIFIED DATAORDER_PREREG.md` = **0**. §3.4's table is exhaustive with exactly REPLICATES / ARTIFACT / MIXED. The old body also mis-stated the branch conditions as "both in band → REPLICATED / both outside → FALSIFIED / one in one out → MIXED", inventing a branch and mis-describing REPLICATES (whose actual condition is *every landed seed is CONFIRM*, which is stricter than "in band" — CONFIRM also requires the CI to exclude 0). Fixed: the verdict is **ARTIFACT**, and its disposition is quoted verbatim from §3.4.

3. **Noise-floor claim violated the prereg's own §4.** The old point 3 under "Does mean" asserted *"the measurement apparatus is at least ~0.3pp noisy on TriviaQA-EM"* and told future work to clear that floor. §4 "n = 2 CANNOT" states explicitly that n=2 *"cannot distinguish σ_run ≈ 0 from σ_run ≈ 0.3 pp"* and cannot support any "±Y pp" claim. The 0.3231 figure is a 1-d.o.f. point estimate whose χ²-based 95 % interval for σ is roughly **[0.14, 10.3] pp** — useless as a floor. **Retracted.** The number survives only in its legitimate role: the A04 Stage-A `sd_run` input, where the prereg's rule is explicitly one-directional and a small value clears nothing.

4. **popqa was improperly promoted.** The old text headlined *"PopQA is the more damning axis, not TriviaQA"* and called its pattern "the strongest falsification form". §3.6 designates popqa secondary and bars using secondaries to rescue a failed primary; using one to strengthen a verdict is the same move with the sign reversed. Additionally the phrase "the pre-registered 'sign-contradictory SIG cells' pattern" attributed a concept to `ARM6_STEP215_VERDICT.md`, a doc the claims file says not to cite. Fixed: the 4-axis table is now explicitly labelled descriptive-only.

5. **The manipulation is not order-only.** At 20,000 steps × eff-batch 128 = 2,560,000 sequences against 15,491,607 rows, each run sees **16.53 %** of one epoch (a full epoch is 121,028 steps). Changing the sampler seed changes *which subset* is consumed, not just its order. New section added above; every downstream statement should say "sampler-seed / data-subset variation".

Additionally: follow-through was incomplete — `claims/A03_SURVIVING_CLAIMS.md` still described A-2 as provisional despite this verdict ordering its retraction. Closed at 11:05.
