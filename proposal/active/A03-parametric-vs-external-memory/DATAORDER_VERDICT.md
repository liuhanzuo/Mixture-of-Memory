---
scope: A03 data-order replication of the trajectory-CPT claim — VERDICT
date: 2026-08-11 04:53 GMT+8
status: FALSIFIED. The Arm 3 step220000 headline (+0.4793 SIG on triviaqa em) does NOT survive a change of data order.
prereg: DATAORDER_PREREG.md (band [+0.20, +0.80]pp on triviaqa em; MIXED pre-declared a FAILURE)
evidence: evidence/pilot_one_stage_a_verdict.json (md5 4ced4582cce6772a797a7f41e94e2a7a); per-example shards at olmo2_closedbook_results/A03_1B_dataorder_seed{43,44}_step220000/
seeds_pinned_pre_data: 43 (on .82), 44 (on .73). Seed 45 was queued for .104 but never run — the outcome is decisive at n=2.
supersedes: `A-2 (provisional)` in claims/A03_SURVIVING_CLAIMS.md — that claim's step220000 SIG is now known data-order-dependent and must be retracted.
---

# The result

Both seeds, paired against the same intact-pruned+healed base (A03_1B_keep7_step200k), on the exact same 17,944-item TriviaQA-EM set, n_boot=5000, seed=42, CI95 percentile — the pre-registered protocol, unchanged:

| seed | Δ (arm − base), pp | CI95 (pp)              | verdict | in band [+0.20, +0.80]? |
|------|--------------------|------------------------|---------|-------------------------|
| 43   | **+0.1115**        | [−0.0947, +0.3177]     | TIE     | **NO** (below floor)    |
| 44   | **−0.3455**        | [−0.5517, −0.1393]     | **SIG** | **NO** (below zero)     |

Under the ORIGINAL data order (Arm 3, the training that used pre-`ce5c298` sampler seed 0), the same measurement returned +0.4793 SIG (`ARM6_FINAL_VERDICT.md`). Under two independent data orders it returns +0.11 (TIE) and −0.35 (SIG negative). **The headline effect flips sign across data orders.**

# Why this falsifies rather than merely narrows

`DATAORDER_PREREG.md` §2.2 (committed pre-data as `44840f1`) named three possible outcomes:

* both seeds in band → REPLICATED
* both outside → FALSIFIED
* one in, one out → MIXED, and MIXED was **pre-declared** a failure ("no tie-break seed permitted after seeing a split")

Both seeds are **outside** the band, so we are not even in the MIXED case; this is the FALSIFIED branch by the strictest reading. Seed 44's −0.35 SIG is a stronger form of falsification than the prereg required: not a null result, but a **sign flip**.

# What this means, precisely — and does NOT mean

**Does mean:**
1. **A03's trajectory-CPT claim is retracted.** The "20k-step Dolmino CPT recovers parametric knowledge at step220000" reading is data-order noise, not a real effect. `claims/A03_SURVIVING_CLAIMS.md` §A-2 must be retracted.
2. **`ARM6_FINAL_VERDICT.md`'s interpretation is overturned.** That doc argued the Arm 3–Arm 6 replication at r≈0.99 across arms *validated* the +0.48 signal by showing it was "reproducible, not random". The correct reading was the opposite: it was reproducible **because the shared minibatch sequence made the runs deterministic**, and once data order varies the "signal" evaporates.
3. **The measurement apparatus itself is at least ~0.3pp noisy on TriviaQA-EM.** Two data-order-only draws (sampler-fix on, all else pinned) span 0.457pp on the mean and diverge by ~0.8pp on the confidence intervals. Any future 1B/dolmino/keep7 claim under this apparatus must clear that floor.

**Does NOT mean:**
1. **A04 is dead by this.** A04's Pilot Zero finding — that PLATEAU accepts a recovery run where NI(Δ=10%·residual) rejects on 3/3 axes — is measured at a SINGLE fixed checkpoint (`keep7+fresh2 step200000`) against intact, and does **not** use the Arm 3 trajectory at all. Those two proposals share only the training apparatus, not the claim.
2. **All CPT-recovery is dead.** This falsifies one specific claim (a 4.5pp-scale effect at one checkpoint under one arm). It does not say anything about larger effects, deeper arms, or longer schedules. It says nothing about whether the noise floor at keep12/keep10 is different.
3. **The Arm 3–Arm 6–Arm 4 phase-locking analysis was wrong.** The phase-locking claim itself remains true and important: with sampler seed 0, all three arms consumed byte-identical minibatch sequences, and their headline agreement was engineered, not scientific. This falsification is *consistent with* the phase-locking finding — it just draws the correct conclusion from it.

# The A04 side effect

The same data (via `code/pilot_one_stage_a_sd_run.py`, commit `49e665d`) simultaneously satisfied A04's Pilot One Stage A prereg (commit `2ac0b5a`). Applying that pre-registered rule verbatim:

    sd_run(triviaqa) = |9.6968 − 9.2399| / √2 = 0.3231 pp
    bound_3(triviaqa) = 2.920 · 0.3231 / √3 = 0.5448 pp
    Δ_triviaqa = 4.043 pp
    → bound_3 << Δ on all 3 decision axes → STAGE_A_DOES_NOT_FIRE

But Stage A **cannot clear K2** by prereg (it is one-directional), and see §6 below for why Stage B is now less attractive than it was 12 hours ago.

# What was locked in advance vs decided after seeing data

Everything the verdict rests on was committed BEFORE the numbers existed:

| commit  | date               | what              |
|---------|--------------------|-------------------|
| `44840f1` | 2026-08-10 19:20 | DATAORDER_PREREG.md (band, protocol, MIXED-is-FAILURE rule) |
| `2ac0b5a` | 2026-08-10 23:50 | PILOT_ONE_PREREG.md (Stage A/K2 rule) |
| `49e665d` | 2026-08-10 (later) | Stage-A driver (imports canonical loaders, positive stale-copy guard) |

The seed 43/44 checkpoints landed 2026-08-11 04:19–04:21; the eval watchers auto-fired 04:23; both evals completed 04:27–04:29; MAIN ran the driver at 04:51. Nothing about the rule changed after the numbers were visible.

# The one thing I did NOT verify (flag, do not paper over)

I re-derived seed 43/44 vs base on TriviaQA-EM directly from the per-example shards (17,944 items each, complete against A03's shard-completeness assertions). But I did NOT re-derive them on the four A03 axes (popqa/mmlu_content/nq_open) at this decision point, because DATAORDER_PREREG.md's primary endpoint was TriviaQA-EM alone. The Stage-A driver reports the three A04 decision axes' means as arm-only (seed_a vs seed_b), not paired vs base, so a full 4-axis A03 replication table remains a to-do — but the pre-registered rule is a triviaqa-EM-only rule, so this table is descriptive at best, not verdict-changing.

# What to do now

1. Retract A-2 from `claims/A03_SURVIVING_CLAIMS.md` and add this file to the ledger.
2. Update `STATUS.json.arm6_midlowLR_cpt.claim` from "provisional" to "retracted (data-order falsification)".
3. Decide on A04 Stage B — see the follow-up note in `A04-recovery-certification/STAGE_B_DECISION.md`.
