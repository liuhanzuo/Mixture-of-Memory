---
scope: A04 Pilot One Stage B — should we spend the 135 GPU-h? DECISION.
date: 2026-08-11 05:00 GMT+8
status: SUPERSEDED 2026-08-11 11:05 — Path A was launched under standing autonomy (seeds 101/102 done, 103 running on .73). Original recommendation was HOLD; the reasoning contained a retracted noise-floor claim. See the CORRECTION banner below before citing anything here.
---

---

> ## ⚠️ CORRECTION 2026-08-11 11:05 GMT+8 — read before using this doc
>
> This doc was written at 05:00, before the A03 verdict was audited. Two things it
> relies on were wrong, and Path B was built on one of them:
>
> 1. **The A03 branch is ARTIFACT, not "FALSIFIED".** `DATAORDER_PREREG.md` §3.4
>    enumerates exactly REPLICATES / ARTIFACT / MIXED; "FALSIFIED" appears nowhere in
>    that prereg. The scientific content is unchanged (0/2 seeds CONFIRM → the positive
>    reading is retracted), but every "the A03 falsification" phrase below should read
>    "the A03 ARTIFACT verdict".
> 2. **★ There is NO measured noise floor.** Lines below say "the noise floor we just
>    measured" and "< 0.5pp, i.e. inside the noise floor". `DATAORDER_PREREG.md` §4
>    ("n = 2 CANNOT") explicitly forbids estimating σ_run at n=2: it "cannot distinguish
>    σ_run ≈ 0 from σ_run ≈ 0.3 pp". The 0.3231 pp figure is a 1-d.o.f. point estimate
>    whose χ² 95 % interval for σ is roughly **[0.14, 10.3] pp**. It is legitimate ONLY
>    as the `sd_run` input to Stage A, where `PILOT_ONE_PREREG.md` declares the inference
>    **one-directional** (a small sd_run clears nothing).
>
>    **Consequence for Path B:** Path B's stated purpose was "nail down the noise floor at
>    n=4". That is a coherent goal — but the argument *for* it below (that the keep12
>    effect would fall "inside the noise floor we just measured") is circular, since no
>    floor was measured. Path B is neither strengthened nor weakened by the A03 result;
>    it is simply a variance-estimation experiment that was never pre-registered.
>
> **What actually happened:** Path A was launched under standing autonomy. Seeds 101/102
> completed; seed 103 launched on `.73` at 10:22 GMT+8. Stage A returned
> `DOES_NOT_FIRE` at both keep7 and keep12 — and per prereg that is **not** a K2
> clearance in either case. Notably keep12's spread is **not** smaller than keep7's
> (popqa 0.451 vs 0.273, mmlu_content 0.055 vs 0.025), contradicting the "smaller effects
> at keep12" reasoning in §21-24 below. Spread is not monotone in damage.

---

---

> ## ⚠️ ADDENDUM 2026-08-11 21:10 GMT+8 — the A03 side has now been decided; three corrections to this doc
>
> A03 executed its own no-GPU arm-set gate (`../../archive/A03-parametric-vs-external-memory/ARM_SET_DECISION.md`)
> and returned **ARCHIVE**. That decision touches this doc in four ways. **This addendum
> does not decide A04** — Stage B is already spent and Pilot Two is A04's call requiring
> explicit user approval per `STATUS.json:next_gate[4]`.
>
> 1. **The question this doc asks is moot: the 135 GPU-h was spent.** Verified read-only
>    on zwfy6: all three pre-registered seeds `{101,102,103}` trained
>    (`step2500/step5000/final.pt` each), all three evaluated on all four axes
>    (17,944-row merged TriviaQA per-example files present for each), and
>    `evidence/stageB_S3_verdict.json` exists (md5 `7145d569f46ec0fa10dd56368071adf2`,
>    written 14:53) with `STAGE_A_DOES_NOT_FIRE`, `n_decision_axes_exceeding: 0`.
>    The live question is the **next** tranche (Pilot Two, 1,077-4,309 GPU-h).
>
> 2. **★ Path B is already bought — do NOT spend 135 GPU-h on it.** Path B proposed "two
>    extra keep7 seeds at 20,000 steps to nail down the noise floor at n=4 (df=3)". A
>    keep7 20k family at **n = 3** exists *now* at zero marginal cost: the original A03
>    Arm 3 **is** the sampler-seed-0 draw of the identical config —
>    `DATAORDER_VERDICT.md` line 20 labels it so, `_run_a03_dataorder_repl.sh` is
>    config-identical to `_run_a03_arm3_cpt.sh` apart from `--seed`, these are *resumed*
>    runs and the trainer has **no dropout**, so `--seed`'s only material channel is
>    `DistributedSampler(seed=args.seed)`. That family: **s = 0.4132 pp, df = 2, χ² 95 %
>    CI [0.215, 2.597]**. Seed 45 is **already running** and takes it to df = 3.
>    **Path B's deliverable costs 0 additional GPU-h, not 135.**
>
> 3. **Spread is not monotone in damage — now confirmed at S = 3, not S = 2.** The
>    banner above already flagged this; here it is with d.o.f. attached. keep12 5k
>    (seeds 101/102/103, df = 2 each): triviaqa **0.3023**, popqa **0.3328**,
>    mmlu_content **0.0783**, nq_open **0.2091** pp. keep7 20k (sampler seeds 0/43/44,
>    df = 2): triviaqa **0.4132** pp. keep12 is smaller on triviaqa only and **larger**
>    on the other three (mmlu_content 3.1×). Any seed budget premised on "less damage ⇒
>    less variance" is mis-budgeted. Do **not** quote the keep7 popqa/mmlu/nq_open
>    pairwise values (0.2726 / 0.0252 / 0.0000) as σ — they are df = 1 ranges.
>
> 4. **The dependency A04 must now write down, and the caveat that comes with it.**
>    A04 certifies *recovery*; its value scales with the size of the recovery effect its
>    rule adjudicates. A03 has measured the CPT recovery increment **on the shared
>    apparatus** at **+0.0818 pp, CI95 [−0.945, +1.108] pp** = **0.26 %** of the 31.10 pp
>    deficit (a CI containing zero), and shown that 10× that token budget (the 200k-step
>    heal, 52.43 B tokens) does not close the gap either. NI and PLATEAU can only
>    disagree *informatively* about a trajectory that moves. **This weakens the case for
>    Pilot Two.** Two things fairly push the other way and must be recorded alongside:
>    (a) **Pilot Zero is untouched** — it is a level at one checkpoint against intact,
>    with no second-checkpoint and no second-training-run term in its estimator, exactly
>    as `DATAORDER_VERDICT.md` "Does NOT mean" #1 states; A03's ARTIFACT verdict must not
>    be cited against it. (b) **K2 is better supported than at Stage B launch**: at the
>    S = 3 point estimates `bound₃` clears Δ on all three decision axes with margins
>    7.9× (triviaqa 0.510 vs 4.043), 2.4× (popqa 0.561 vs 1.321), 7.8× (mmlu_content
>    0.132 vs 1.024). **But** evaluated at the χ² 95 % *upper* bound of each σ (df = 2),
>    triviaqa (3.203 vs 4.043) and mmlu_content (0.829 vs 1.024) still do not fire while
>    **popqa would** (3.527 vs 1.321) and demoted nq_open would (2.215 vs 0.970). So the
>    honest line is: *"K2 does not fire at the point estimate, and one decision axis
>    would fire at the pessimistic end of a df = 2 σ interval"* — which argues for **more
>    seeds before Pilot Two**, not for Pilot Two.
>
> **What A04 should be required to state before committing the next tranche** (this is a
> recommendation to A04, not a ruling on it): in its own prereg, **pre-data**, *what
> recovery magnitude the certification is meant to adjudicate*, plus a showing that this
> magnitude exceeds the MDE its chosen S implies — **1.10 pp at S = 3 and σ̂ = 0.362 pp;
> 3.16 pp at the χ² 95 % upper bound**. That is precisely the discipline A03 lacked, and
> it is why A03 is archived.
>
> Full arithmetic and provenance: `../../archive/A03-parametric-vs-external-memory/ARM_SET_DECISION.md` §2, §4
> and `../../archive/A03-parametric-vs-external-memory/STATUS.json:consequence_for_A04_135gpuh`.

---

# Question

`PILOT_ONE_PREREG.md` §3 says Stage B (keep12+fresh2 × 3 seeds × 5,000 steps, ~135 GPU-h, ~5.6h wall on 3 nodes) proceeds when Stage A does not fire. Stage A did not fire. Should we launch Stage B?

# The case FOR launching Stage B (unchanged by the A03 ARTIFACT verdict)

* A04's Pilot Zero finding — PLATEAU accepts, NI rejects on 3/3 decision axes at `keep7+fresh2 step200000` — is a **level** measured at one checkpoint against intact, not a differences-between-steps trajectory. **The A03 ARTIFACT verdict does not touch that finding.** They share the training apparatus, not the claim.
* Pilot Zero explicitly said `keep7` is a constant-REJECT rung and useless for testing the *discriminative* power of NI-vs-PLATEAU. Only `keep12` (or `keep10` if `keep12` is constant-ACCEPT) can make the disagreement test **falsifiable**. Stage B is the cheapest way to get there.
* Stage-A's measured sd_run at keep7 (0.32pp TriviaQA, 0.27pp PopQA, 0.025pp MMLU-content) is comfortably below every Δ (4.04 / 1.32 / 1.02 pp), so at S=3 the bound is ~0.5pp — well inside the noise budget the gate needs. K2 will likely not fire at keep12 either, so the gate can actually run.
* K2 is the design's own "most likely killer". 135 GPU-h is 3–5% of the full-gate cost and closes it either way.

# The case AGAINST launching Stage B (new, from the A03 result)

* **The A03 ARTIFACT verdict is a warning about effect sizes at this scale.** Under the seed-fixed sampler, the same apparatus produced two independent draws that spanned **0.46pp** on the mean and **~0.8pp** on the CI, and the Arm 3 "signal" (+0.48pp) disappeared — even flipped sign — under randomization. That is not a K2 issue (K2 was about `sd_run` swamping Δ, which it doesn't) but it IS a signal that **effects small enough to be interesting on this arm are not stable across data orders**.
* Pilot One's expected effect size at `keep12` is *smaller* in absolute pp than at `keep7`, not larger. The reasoning:
  * keep7 has ~40pp intact residual, recovers ~23% (~9pp EM); a CPT-recovery marginal effect there was measured (falsified) at ~0.48pp.
  * keep12 has ~14pp intact residual (per A01's 7B keep12 ladder scaled down); recovery ceiling is smaller; the *marginal* CPT effect over baseline recovery scales with the room-to-recover.
  * So the effect we'd measure at keep12 is plausibly < 0.5pp. ⚠️ The original text said "i.e. inside the noise floor we just measured" — RETRACTED, no floor was measured (banner, item 2). The prediction that keep12 effects are smaller was ALSO wrong empirically: keep12's spread came back LARGER than keep7's on popqa (0.451 vs 0.273) and mmlu_content (0.055 vs 0.025).
  * The disagreement test at keep12 is falsifiable in principle (NI can accept), but the RECOVERY effect being tested has to itself be real to matter — and after A03, that is exactly what is in question.
* The 135 GPU-h buys **one arm at one depth**. If it comes back "keep12 recovers 60%, NI accepts, PLATEAU accepts" — great, A04 lives, but that is one uncontrolled data point. If it comes back "keep12 recovers 25%, both reject, disagreement persists" — hmm, but is the disagreement itself an artifact of the same apparatus that just gave us a falsified headline?
* **Stronger option, same GPU budget**: spend the 135 GPU-h on *two* seeds of `keep7` at 20,000 steps (matching the A03 arms) to build a proper 4-seed variance table at that arm, plus 1 seed of `keep12`. That gives a real sd_run at keep7 (n=4 → df=3 → t=2.353, tighter bound) AND a pilot data point at keep12. It also directly settles whether the A03 ARTIFACT verdict was a fluke of n=2.

# What I am NOT recommending

* **Killing A04 outright.** The Pilot Zero disagreement finding stands and is decoupled from A03. K1 remains INDETERMINATE (needs the ≥24-cell family the design specifies), and running the full ≥24-cell gate is a separate decision at 2,900 GPU-h.
* **Silently changing Stage B's scope.** If we do launch, launch what the prereg says, not a "clever" variant chosen after seeing data.

# Recommendation

**HOLD Stage B, pending user call.** Two paths:

**Path A — launch as prereg'd (keep12 × 3 seeds × 5,000 steps, ~135 GPU-h).** Defensible; the prereg licenses it. Risks: burns 135 GPU-h on a measurement whose signal may be small relative to run-to-run spread (magnitude unknown — see banner item 2), and does not directly address the A03 ARTIFACT verdict's implications.

**Path B — replace Stage B with a re-scoped diagnostic (~same 135 GPU-h).** Two extra `keep7` seeds at 20,000 steps to nail down the noise floor at n=4 (df=3 tightens the bound 20%), plus one `keep12` seed at 5,000 steps as an anchor point. Rules against Path B: it is not the pre-registered plan, so anything it produces cannot enter A04 with the same evidentiary weight. Rules for it: it directly answers the question the A03 result just raised, and it's cheaper science per GPU-h at this juncture.

**My call, if forced to decide autonomously**: **Path A**, because the prereg exists exactly to prevent me from reasoning myself into Path B after seeing data. But I am flagging this for user visibility rather than launching immediately, because the A03 result is genuinely new information that arrived AFTER the prereg was written, and the standing autonomy rule ("以后不用问我") is scoped to "unless巨额投机算力" (large speculative compute) — 135 GPU-h at a moment of new adverse evidence arguably qualifies.

# Operational readiness (if the user says Path A)

* Prereg: `PILOT_ONE_PREREG.md` (commit `2ac0b5a`)
* Arm: `--keep_front_layers 12 --n_fresh_layers 2` — trainer `transplant_front()` prunes from HF base directly (no pre-existing keep12 1B ckpt needed; §3 of prereg verified this).
* Seeds pinned: `{101, 102, 103}` — cannot be chosen after any run's result is seen.
* Nodes: **all 3 on zwfy6** — dolmino_now15b.npy is 62 GB on wzc1 vs 127 GB on zwfy6 (same name, different file; mixing disks silently mixes corpora). `.73`, `.82` are free now; `.104` frees when keep12 resume-to-200k finishes (~11h at 8.2s/step).
* Wall time per run: 2.81h (2.02s/step median × 5,000 steps ÷ 8×H20). Three seeds on three nodes = one wave ≈ 5.6h wall.
* No launch script written yet — deliberately. Will write one AFTER user decides, so a mid-second-guess can't accidentally trigger it.
