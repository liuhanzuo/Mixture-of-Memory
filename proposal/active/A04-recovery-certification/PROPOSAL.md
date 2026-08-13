# A04 — Recovery Certification after Structural Injury

**Rewritten 2026-08-13** to the narrowed `safe_residual_claim`, discharging
`STATUS.json:blocked_by.still_blocking_before_any_gate_gpu[0]` and
`next_gate[2]`. **GPU spent producing this rewrite: 0.**

The pre-2026-08-13 text is **not deleted**. It is preserved verbatim in
§9 under a `SUPERSEDED` banner, per `proposal/README.md`'s standing rule that
撤回史 must not be silently rewritten. Where §9 conflicts with §1–§8, **§1–§8
governs.**

> **Promotion-rule note.** `proposal/README.md` was relaxed on 2026-08-11
> (「只要有发现就可以继续做」). The four-condition promotion checklist is **not**
> an entry exam and is not applied here. This rewrite is a **scope correction**,
> not a promotion gate: it narrows what A04 may claim to what is measured.

---

## 1. Claim (the only one A04 may make)

> **A pre-registered, multi-seed, matched-corpus / matched-token STOPPING RULE
> for post-injury recovery training, whose accept decision is an EQUIVALENCE
> (non-inferiority) test against the intact target on capability axes each
> calibrated to its own best-constant floor, plus a demonstration that the
> stopping rules currently in use (likelihood/PPL plateau; aggregate retention
> ratio) accept models this rule rejects.**

Verbatim from `STATUS.json:safe_residual_claim`. Everything below either
implements it or bounds it.

### 1.1 What A04 is NOT

- Not a depth scaling law (the historical keepN ladder is confounded three ways
  — see `STATUS.json:warning_evidence_reverified_20260809`).
- Not a claim that "our rule is better" — only that the rules **disagree**, and
  by how much.
- Not a claim about 7B. Every 7B number in this proposal is **diagnostic or
  contradicting evidence about the rule's behaviour**, never a certification
  result: each 7B rung has exactly one seed and the historical seeds are
  unrecorded, so no 7B `sd_run` exists or is reconstructible.

---

## 2. Scope — `must_not_claim`, pasted from `STATUS.json`

Binding on any A04 writeup, slide, abstract or commit message. Pasted verbatim
from `STATUS.json:must_not_claim` (2026-08-09 novelty audit,
`RELATED_WORK.md`):

1. First to study layer-pruning recovery (C1/C2/C3 + the 2026-08-08 audit's own
   wording).
2. First to observe recovery is metric-dependent / classification recovers while
   generative reasoning does not — **arXiv:2602.01997 owns this.**
3. First to run a controlled token-matched pruned-init vs scratch-init
   comparison — **arXiv:2606.14150 owns this**, at 8B across six pruning methods
   and two budget regimes, broader than A04's 1B MVP.
4. First to note perplexity is an unreliable capability proxy —
   **arXiv:2601.22950 PROVES it**; 2606.03002 (quantization) and 2607.00368
   (TTT) show it empirically.
5. First to propose calibrating claims to the evidence that supports them —
   **arXiv:2607.00368 owns this framing.**
6. First to argue performance parity != equivalence for a compressed model —
   **arXiv:2508.13533 owns this.**
7. First to compare arms at matched perplexity — **arXiv:2606.03002 already does
   it.**
8. 'Certification' left undefined — it must mean equivalence-to-target on
   null-calibrated capability axes with a pre-registered threshold, else it
   collides directly with arXiv:2508.13533.
9. A depth SCALING LAW from the existing Paper B keepN ladder — confounded three
   ways in our own data (see `warning_evidence_reverified_20260809`).
10. That Paper B used differential LR — verified false.
11. Novelty for 'a layer-pruned model can be repaired by a linear map at the
    cut' — **arXiv:2605.15491 owns this** in closed form, training-free.
12. Permutation-null / BH novelty — A01's prior art rules apply here too; the
    null-calibration protocol is cited FROM A01, not re-claimed.

### 2.1 Additional `must_not_claim` entries added 2026-08-13

These come from the three verdicts of 2026-08-13 and from
`../../shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md`
(commit `75cf173`). They are **new** — not in `STATUS.json:must_not_claim` —
and are equally binding:

13. ❌ **`full32@step25000` may not be called "A04's accept".** See §4.3. It is
    the **only** checkpoint on its own 5-point trajectory that **fails** the
    2-of-3 bar.
14. ❌ **"Heal longer approaches certification."** Falsified at 7B on **both** a
    damaged arm (keep14+fresh2) and an undamaged one (full32). See §4.1, §4.3.
15. ❌ **"The neighbour range is the same order as Δ" / "49.9 % of Δ" / "60.2 %
    of Δ".** Those two figures are **cross-axis** (a triviaqa range divided by
    another axis's Δ). Same-axis values are **17.7 %** and **30.9 %**. See §5.
16. ❌ **First to observe non-monotone downstream benchmark development during
    training** — arXiv:2406.10229 owns it (MMLU Kendall τ = 0.09).
17. ❌ **First to treat checkpoint-to-checkpoint variability as a first-class
    eval quantity** — **arXiv:2508.13144, NeurIPS 2025 Spotlight**, owns it,
    defines it, validates it against seed and data-order noise, and prescribes
    averaging. See §6.
18. ❌ **First to note that comparing at a hand-picked checkpoint can flip a
    conclusion** — arXiv:2509.02046 (ICLR 2026 Poster) owns it.
19. ❌ **First to argue a monotone loss with a non-monotone task metric means
    stopping criteria must be task-level** — arXiv:2601.03858 (2026-01) owns it.
20. ❌ **"Damaged arms are noisier than intact arms"** as a *result*. It is a
    hypothesis from a cross-protocol, n=3-vs-n=30, different-harness comparison.
21. ❌ **"The popqa mid-heal dip is a property of healing."** The pre-registered
    cross-arm replication **failed**. It is an existence proof about
    **checkpoint selection**, nothing more. See §4.2.
22. ❌ **"A04's Δ construction is standard."** `Δ = 0.10 × residual(intact)` is a
    **data-dependent** equivalence margin; arXiv:2603.16213 is about exactly
    that problem and must be engaged. See §6.
23. ❌ **Treating checkpoints of one run as replicates**, or reading any
    trajectory spread as seed variance. No 7B `sd_run` exists.
24. ❌ **Quoting any margin to better than 0.01 pp across nodes** (numpy
    `Generator.multinomial` drifts up to 0.005294 pp between 2.4.6 and 2.5.1;
    see `memory/numpy-version-split-breaks-cross-node-bootstrap`).
25. ❌ **Putting any of this in A01 / `paperC`.** Different construct: A04's
    trajectory findings move the *reported* value; A01's degrees of freedom move
    the *null*.

---

## 3. Robust starting point (unchanged, and now stronger)

> Along a single recovery trajectory, PPL that keeps improving does not
> establish that the model is approaching the intact target.

2026-08-13 strengthens this from "does not establish" to "can be **anti**-correlated
with the target axis": on the zero-damage `full32` control, continued
pretraining moved **3 of 4** capability axes *away* from the intact anchor while
PPL was still decaying (§4.3).

---

## 4. The four pieces of negative / narrowing evidence of 2026-08-13

Each entry states **exactly which part of A04's claim it compresses**. All
numbers are from the canonical evidence JSONs, **not** from prose in the verdict
`.md`s (three hand-arithmetic errors were made on 2026-08-13 alone; canonical
JSON always wins).

### 4.1 keep14+fresh2 trajectory — REJECT everywhere, and popqa gets *resolvedly worse*

Source: `evidence/a04_keep14_trajectory_ni.json`, doc
`A04_KEEP14_TRAJECTORY_NI_VERDICT.md`. 2.0756 GPU-h, eval-only.

- 3 checkpoints (128 000 / 153 500 / 200 000), **0 of 3 decision axes accept at
  any of them**, under **all five** tie conventions. `RATIO(0.85)` also rejects
  (mean ratio 0.4644–0.4728) — **no rule disagreement anywhere on this arm.**
- Margins are **7.47–29.61 pp short**, i.e. 20.6–77.3 bootstrap SE from
  flipping. No perturbation of the item sample can change any verdict.
- **popqa got resolvedly WORSE over 25 500 steps of healing:**
  `acc_delta = −0.6729 pp`, CI95 `[−0.9252, −0.4206]`, `p = 0.0001`,
  **218 right→wrong vs 122 wrong→right** of 14 267. Not degeneracy (0.000 %
  empty predictions, top-constant share *falling* 2.481→1.213 %, distinct
  predictions *rising* 8436→9205).
- Naive last-interval extrapolation to margin 0: **7.7× / 10.6× / 244.0×** the
  entire 200k heal budget; under the all-points OLS slope popqa's slope is
  **negative** and never reaches zero.

**What it compresses.** It removes "heal longer" from A04's toolkit on damaged
arms, and it removes the option of reading a *single* checkpoint's margin as a
trajectory position. It also demotes `must_not_claim[14]`'s converse: A04 may
not price any future tranche on "more heal steps ⇒ closer to certifiable".

### 4.2 Neighbour variability — the loophole is real but ONE cell of six, and the pre-registered replication FAILED

Source: `evidence/a04_neighbour_variability.json`, doc
`A04_NEIGHBOUR_VARIABILITY_VERDICT.md`. 8.6556 GPU-h, eval-only.

A range of *k* noisy numbers is biased upward by noise even at zero true spread:
for iid `N(0, σ)`, **`E[range of 3] = 3/√π · σ = 1.6926 σ`** (exact for the
normal). Every range is therefore gated on `range > 1.6926 × mean(bootstrap SE)`.

| cluster | axis | range (pp) | noise gate (pp) | ratio | clears? |
|---|---|---|---|---|---|
| c2 (clean, 500-step) | **triviaqa** | **1.1202** | 0.6577 | **1.703** | **YES** |
| c2 | popqa | 0.2523 | 0.5818 | 0.434 | no |
| c2 | mmlu_content | 0.2208 | 0.6522 | 0.339 | no |
| c1 (resume seam) | triviaqa | 0.2675 | 0.6442 | 0.415 | no |
| c1 | popqa | 0.1192 | 0.5914 | 0.201 | no |
| c1 | mmlu_content | 0.2136 | 0.6449 | 0.331 | no |

**1 of 6** decision-axis range cells clears the gate. The one that does is a
single **resolved** event: triviaqa 130500→131000 cost **−1.0867 pp**, CI95
`[−1.3319, −0.8359]`, `p = 0.0001`, 355 right→wrong vs 160 wrong→right.

**The pre-registered cross-arm replication failed.** shortgpt16's popqa over the
identical 128 000→153 500 interval moves **+0.0841 pp**, CI95
`[−0.1542, +0.3224]`, **`p = 0.5084`** — opposite sign, CI straddles zero,
`REPLICATES = false`.

**What it compresses.** (a) The neighbour precondition survives — one
counterexample suffices to make a single-checkpoint accept uninterpretable — but
it must be stated **per-axis**, never as blanket distrust. (b) §4.1's popqa dip
is downgraded from a property of *healing* to *"there exists at least one arm on
which a later checkpoint scores worse."*

### 4.3 full32 five-point scan — there is a REJECT boundary in (20000, 25000], and the endpoint is the trajectory's ONLY failure

Source: `evidence/a04_full32_trajectory_ni.json` (**canonical; supersedes every
hand-computed full32 number, including MAIN's own interim note
`A04_FULL32_READING_B_IS_FIRING.md`**), doc
`A04_FULL32_TRAJECTORY_NI_VERDICT.md`. 6.53 GPU-h, eval-only.

| step | triviaqa | popqa | mmlu_content | axes accepting | NI (≥2/3) |
|---|---|---|---|---|---|
| 5 000 | **+3.6932 A** | −2.0509 R | **+0.5439 A** | 2/3 | **ACCEPT** |
| 10 000 | **+3.0467 A** | −2.1911 R | **+0.3730 A** | 2/3 | **ACCEPT** |
| 15 000 | **+2.3331 A** | −3.0742 R | **+0.9142 A** | 2/3 | **ACCEPT** |
| 20 000 | **+2.4504 A** | −3.5088 R | **+0.6222 A** | 2/3 | **ACCEPT** |
| 25 000 *(archived endpoint)* | **−0.6035 R** | −4.5391 R | **+1.0495 A** | 1/3 | **REJECT** |

Identical ordering and verdict flip under all five tie conventions. The archived
endpoint reproduces to **3.6e-07 pp**, so the four new points are on the
endpoint's scale.

**Three consequences, each binding:**

1. **`full32@step25000` is no longer A04's accept.** It was selected because it
   was the last save. Its own trajectory accepts at **4 of 5** points and fails
   only at the endpoint. The 97.7 % recovery framing belongs to checkpoints
   nobody had scored.
2. **"RATIO is too permissive" weakens sharply.** RATIO decays *monotonically*
   (0.8981 → 0.8515) and accepts at **all five** points. NI/RATIO disagreement
   occurs at **exactly 1 of 5** checkpoints — the endpoint, where RATIO's margin
   over ρ is **+0.001495**. The endpoint is where RATIO's decay happens to cross
   NI's, not where the rules differ in principle.
3. **"Heal longer" is falsified on an *undamaged* arm too.** popqa is monotone
   non-increasing across all five points; triviaqa EM falls 61.40 → 57.15;
   nq_open falls. Only mmlu_content drifts up, non-monotonically. Together with
   §4.1 this falsifies the premise on both a damaged and an undamaged arm.

**Correcting a related claim in the novelty note.** `MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md`
§C2 evaluated full32's mmlu_content range by hand as 0.2280 pp
(`1.2775 − 1.0495`) against a `k=2` gate of 0.4201 pp → ratio 0.543, FAIL. The
canonical 2-point value is **0.1353 pp** against a gate of **0.2540 pp** →
**ratio 0.533, still FAIL**. Its conclusion is right; its inputs were 1.69× off.
Recomputed from canonical JSON, the **full 5-point** full32 ranges *do* clear the
`k=5` gate (`c = 2.32593`) on all three decision axes — triviaqa 4.2967 pp
(6.02×), popqa 2.4883 pp (4.17×), mmlu_content 0.6765 pp (1.28×) — but that is a
**trajectory over 25 000 steps**, not a neighbour range, and may not be reported
as one.

### 4.4 ⚠️ THREATS TO VALIDITY — the axis that flips the verdict is ~half verbosity

**This is mandatory in any writeup that quotes §4.3's verdict flip.**

Of the **1313** items that went EM right→wrong over 20000→25000, **622
(47.37 %) still CONTAIN the gold answer**, and mean prediction length on exactly
those items explodes **10.92 → 50.37 characters**. Across the whole trajectory
triviaqa EM falls 61.40 → 57.15 (−4.25 pp) while `contains` moves only
69.87 → **68.38** (**−1.49** pp); the `contains − EM` gap **widens 8.47 → 11.23 pp**
against 3.87 pp for the vanilla anchor.

> ⚠️ **Two-decimal correction vs the verdict `.md`.**
> `A04_FULL32_TRAJECTORY_NI_VERDICT.md` §4 prints `contains` at the endpoint as
> **68.39** and the contains move as **−1.48 pp**. Canonical
> (`em_and_contains_per_checkpoint.triviaqa`) is **68.384975** → **68.38**, and
> **−1.487963** → **−1.49**. Rounded up in the md, rounded correctly here. Nothing
> else in §4.4 moves, and no verdict depends on it — but it is the fourth
> hand-arithmetic slip found in these documents on 2026-08-13, which is why §4
> opens by pinning canonical JSON above prose.

So the axis whose REJECT flips the endpoint's verdict is measuring, to nearly
**half its movement, a base LM that stopped emitting short answers** — expected
behaviour for continued pretraining on raw text with no SFT — and **not**
demonstrated forgetting. The other 52.6 % are genuine content substitutions
(`Richard Noble`→`Andy Green`, `David Hockney`→`David Bowie`), so verbosity is
not the whole story either.

**Consequences for the rule itself, not just for this cell:**

- A generative-EM decision axis in a base-LM regime **partly measures output
  format**. A04's certification rule inherits that. Any accept or reject on
  triviaqa/popqa/nq_open must ship this caveat.
- `contains` may **not** be substituted for EM (the decision metric is and stays
  EM); §4.4 characterises what an EM move *consists of*, it does not re-score any
  cell.
- The same coarseness shows up as churn: only **30–47 %** of generative
  prediction strings are identical between adjacent checkpoints while EM moves
  ≲ 0.7 pp. The rule reads a stable-looking scalar off a model whose outputs are
  being rewritten wholesale.
- **Design implication for any future tranche:** either add a
  format-insensitive axis whose null is equally well defined, or pre-register a
  verbosity diagnostic as a *reporting requirement* alongside every generative
  cell. Not yet decided; not yet funded.

---

## 5. Correction — the neighbour range as a fraction of Δ is 17.7 % / 30.9 %, not 49.9 % / 60.2 %

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §6.1 (and, following it,
`A04_GATE_DESIGN.md` §2.0.2's empirical-basis bullet) states the loophole is
worth *"1.1202 pp — 49.9 % of popqa's entire Δ (2.2457 pp), and 60.2 % of
mmlu_content's (1.8614 pp)"*. **Both ratios are cross-axis**: they divide a
**triviaqa** range by a **different axis's** Δ. `ni_rule` only ever compares a
margin to *its own axis's* `delta_pp`.

Same-axis, from `evidence/a04_neighbour_variability.json` and
`evidence/a04_keep14_trajectory_ni.json`:

| cell | axis | range / move (pp) | that axis's own Δ (pp) | % of own Δ |
|---|---|---|---|---|
| keep8 c2 (clean, 500-step) | triviaqa | 1.1202 | 6.3291 | **17.70 %** |
| keep8 c2 | popqa | 0.2523 | 2.2457 | 11.24 % *(sub-noise)* |
| keep8 c2 | mmlu_content | 0.2208 | 1.8614 | 11.86 % *(sub-noise)* |
| keep14 (25 500-step) | popqa | 0.6939 (margin) | 2.2457 | **30.90 %** |

Recompute: `1.1202/6.3291 = 0.1770`; `0.6939/2.2457 = 0.3090`.

**A04 uses 17.7 % and 30.9 %.** The 49.9 % / 60.2 % figures are **retired** and
must not be reused. A correction note has been appended to
`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` (original sentence left intact, per the
撤回史 rule). 17.7–30.9 % of the quantity under test is still material — this is
a correction of magnitude, **not** a retraction of the finding.

---

## 6. Related Work — 2026-08-13 addendum

`RELATED_WORK.md` (2026-08-09) remains the authority for the pruning-recovery
collisions C1–C3 / N1–N6 and for all 11 attribution obligations in its §4.2.
This section adds what that audit did not cover: the **checkpoint-noise** and
**equivalence-margin** literatures. Full audit, with venue provenance:
`../../shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md`
(commit `75cf173`).

### 6.1 Heineman et al., *Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation*, arXiv:2508.13144 — **NeurIPS 2025 Spotlight**

**Venue, verified:** OpenReview note `sAFottNlra`,
`venue = "NeurIPS 2025 spotlight"`, `venueid = NeurIPS.cc/2025/Conference`,
invitations include `Submission26329/-/Camera_Ready_Revision`.
⚠️ **DBLP has it as CoRR-only** (`journals/corr/abs-2508-13144`) and S2 lags —
**either alone would misread this as a preprint.** Per
`memory/venue-verify-must-use-openreview-2026`, OpenReview `venueid` +
`Camera_Ready_Revision` is the authority for the NeurIPS/ICLR/ICML family.

**What it reaches (and A04 must credit, not re-claim):**

- **Defines** benchmark "noise" as sensitivity to random variability between
  training steps, operationalised as `Rel.Std.(m)` = std/mean of the metric
  **over the final n intermediate checkpoints**.
- **Validates** that statistic against init-seed, data-order and whole-curve
  noise (R² = 0.82 / 0.86 / 0.95; 20 purpose-trained 1B runs, 3.2 K
  checkpoints).
- Publishes **per-task noise for the OLMo-2 family at 1.5B / 7B / 13B / 32B**,
  including **MMLU and TriviaQA** — the same family and two of the same axes
  A04 uses (their 7B-4T: TriviaQA 0.003, MMLU 0.023).
- **Prescribes an intervention**: average the final *k* checkpoints instead of
  reporting the last one (+2.4 % decision accuracy over 30 tasks), with guidance
  on choosing *k* (n = 5 for ±1σ, n = 20 for ±0.2σ).
- 465 models × 30 benchmarks, 900 K results.

**Therefore `must_not_claim[17]`.** A04 may not present checkpoint-to-checkpoint
variability as its own discovery, may not present "report the last checkpoint is
unsafe" as new, and may not present a rel-std-style statistic as new.

**What is left for A04 — the asymmetry of the equivalence decision:**

> P1's decision problem is **superiority / ranking** ("is corpus A better than
> B?") and scaling-law extrapolation. There, checkpoint noise is **symmetric**:
> it costs statistical **power**, and averaging the final *k* checkpoints
> repairs it.
>
> A04's decision problem is **non-inferiority**. There the same noise is
> **one-sided and free**: an analyst needs only **one** neighbour whose margin
> clears **−Δ** to report an accept, and nothing in the accept's own statistics
> reveals that the neighbours reject. Averaging does not repair this, because the
> defect is not variance in an estimate — it is a **selection** degree of freedom
> in a **decision**.
>
> **P1 structurally cannot discuss this: it has no accept.** Every quantity in
> P1 is a score, a ranking or a prediction error; there is no threshold whose
> crossing constitutes a claim. The asymmetry only exists once a rule is allowed
> to say "certified".

A04's contribution is therefore to (a) **name** that asymmetry, (b) **measure
the size of the option** on a damaged arm (1.1202 pp = **17.7 %** of that axis's
own Δ, **1.70×** the item-noise range floor — §4.2, §5), and (c) turn it into a
**per-axis reporting precondition** (`A04_GATE_DESIGN.md` §2.0.2). §4.3 supplies
a second, sharper instance: a **real accept** at `full32@step20000` that its
upper neighbour revokes (`ACCEPT_IS_CHECKPOINT_SELECTION_DEPENDENT`) — the
precondition firing on a live cell rather than on a hypothetical.

**Falsifiable follow-up that this addendum does NOT claim as done:** are damaged
arms noisier than intact arms *on one harness*? Converting A04's clusters into
P1's unit gives keep8-c2 triviaqa rel.std **0.0395** vs P1's OLMo-2-7B-4T
**0.003** (and keep8-c2 mmlu_content **0.0030** vs their **0.023** — the
opposite direction). That is **n=3 vs n=30, different harness, different metric
conventions** → a hypothesis worth a controlled test, **never a tabulated
result** (`must_not_claim[20]`).

### 6.2 Second-nearest works in this literature (cite, do not claim)

| cite | verified venue | owns |
|---|---|---|
| arXiv:2406.10229 Madaan et al. | **NeurIPS 2024 RegML workshop / preprint.** OpenReview `M9dCa4vYgp` = `venueid=NeurIPS.cc/2024/Workshop/RegML` (no Camera_Ready); `E2RyjrBMVZ` = `ICLR.cc/2025/Conference/**Rejected_Submission**`. **Never cite as ICLR.** | Benchmark **monotonicity** as Kendall τ over 21 intermediate checkpoints; **standard MMLU τ = 0.09**, cloze 0.95. → `must_not_claim[16]` |
| arXiv:2509.02046 Wen et al. | **ICLR 2026 Poster** (OpenReview `2J51qUZ0iG`, `venueid=ICLR.cc/2026/Conference`, Camera_Ready present) | "rankings between two optimizers can flip during training due to LR decay" → `must_not_claim[18]` |
| arXiv:2601.03858 | **PREPRINT** (CoRR 2026; the OpenReview note is a DBLP mirror, `venueid=dblp.org/journals/CORR/2026`) | "loss decreases monotonically while factual learning is unstable and non-monotonic … motivate evaluation of stopping criteria based on task-level learning dynamics" → `must_not_claim[19]`. **Closest paper to A04's motivation sentence.** |
| arXiv:2406.08446 **OLMES** | **Findings of NAACL 2025**, DOI `10.18653/v1/2025.findings-naacl.282` (ACL Anthology + DBLP `conf/naacl/GuTKHDH25`; ACL family → Anthology, **not** OpenReview, per `memory/venue-verify-acl-family-needs-anthology`) | The eval standard A04's protocol answers to. **Contains no checkpoint-neighbourhood requirement** — verified. So §2.0.2's precondition is not already in a benchmark standard; P1 prescribes *averaging*, nobody prescribes *reporting the range as a robustness check on a decision*. |
| arXiv:2304.15004 Schaeffer et al. | **NeurIPS 2023 Oral** | Discontinuous metrics manufacture apparent transitions. Ancestry for §4.4's "EM is a coarse read". |
| arXiv:2511.09864 | **ICLR 2026 Withdrawn_Submission** — not published | "RL finetuning exhibits high variance across model checkpoints … relying on the final checkpoint [is suboptimal]". Corroboration only; note withdrawn. |

### 6.3 arXiv:2603.16213 — data-dependent equivalence margins

*Equivalence testing with data-dependent and post-hoc equivalence margins.*
**PREPRINT** (arXiv only; cited for its statistical point, not its status).

A04's margin is `Δ_x = 0.10 · residual(intact, x)` — i.e. **estimated from the
same data that the test is run against**, which is precisely the object of this
paper. Consequences A04 must accept:

- **`must_not_claim[22]`:** A04's Δ construction is **not** standard, and
  "pre-registered" does not answer the objection. What is pre-registered is the
  *fraction* (0.10) and the *anchor*; `residual(intact)` is a **random
  quantity**, so Δ inherits its sampling error.
- The existing `A04_MARGIN_GUARD_PREREG.md` D1–D6 guard already handles the
  **degenerate** cases (residual negative / at zero / CI straddling zero / Δ
  below the item-level CI half-width). It does **not** handle the
  **non-degenerate** inferential question of a data-dependent margin. That gap is
  now on the record and must be engaged before Δ is defended in any writeup.
- Independently re-verified 2026-08-13 with fresh query strings: **zero**
  pruning/recovery papers use TOST / non-inferiority / equivalence margins. The
  **equivalence-direction novelty of §1 survives.** Only the *construction* of Δ
  now has a statistics paper to answer to.

---

## 7. Arms and design (narrowed)

Implements `A04_GATE_DESIGN.md` §3, which is the binding version. Reproduced
here so the proposal and the gate cannot drift apart.

| # | Arm | Trainer flags | Role |
|---|---|---|---|
| A1 | prefix + fresh tail | `--keep_front_layers j --n_fresh_layers 2` | the canonical construction; A04's claim is about this |
| A2 | contiguous keep-only | `--keep_front_layers j --n_fresh_layers 0` | isolates the fresh tail's contribution |
| A3 | random trunk, inherited interface | `--keep_front_layers j --n_fresh_layers 2 --random_trunk` | same depth/shape as A1; only trunk provenance differs |
| A4 | from scratch | `--from_scratch` at depth `j+2` | "did inheritance matter at all" floor |

`j` is selected by Pilot Zero, **not** fixed here. One damage depth, so depth is
not a variable.

### 7.1 Removed from the design (was in the pre-2026-08-13 text)

- ⛔ **ShortGPT / non-contiguous drop.** It changes **which** layers are removed
  — a *second* damage variable — in a gate whose entire purpose is
  one-variable-at-a-time. Dropping it also saves 25 % of the spend.
  (The existing `shortgpt16` 7B runs stay usable as **diagnostic** arms; §4.2's
  Leg B is exactly that use. `arm_architectures_are_different` in the evidence
  JSON records that shortgpt16 is `keep_layer_indices [0..12, 16, 17, 31]`,
  **non-contiguous** — so it is never a rung of the keepN ladder.)
- ⛔ **"3 token budgets = 3 runs".** The three budgets are **checkpoints of one
  run**. Reading the old "4 structures × 3 budgets × 2 seeds" as 24 independent
  trainings costs 3 771 GPU-h; as 6 checkpoints of 8 runs it costs 2 873 GPU-h.
  There is no scientific reason to retrain for a shorter budget when a
  checkpoint at that budget exists on the same trajectory.

### 7.2 Seeds — semantics, and the口径 break of 2026-08-09

**The `DistributedSampler` seed defect is FIXED** (commit `ce5c298`,
2026-08-09 23:21:09 +0800). Live line, `scripts/train_olmo2_arch_probe2.py:869`:

```python
sampler = DistributedSampler(ds, shuffle=True, seed=args.seed)
```

Pre-fix the call omitted `seed=`, so the sampler's private generator used
`self.seed = 0` regardless of `--seed`; the trainer has no dropout and
`NpyChunkDataset` is a deterministic mmap slice, so `--seed` moved **only** the
fresh-tail init and the data order was byte-identical across seeds. Arm A2
(`n_fresh_layers 0`) therefore had `sd_run` **identically 0** by construction —
a missing measurement, not a variance estimate.

> ### ⚠️ BINDING口径 RULE — pre-fix and post-fix runs may NOT be pooled
>
> Runs launched **after** `ce5c298` vary the data **subset**, not just the
> order: measured rank-0 Jaccard on the 16.53 %-of-epoch slice a 20k-step run
> consumes is **0.0102** post-fix (near-disjoint) vs **1.0000** pre-fix
> (identical).
> **A pre-fix seed arm and a post-fix seed arm are therefore not draws from the
> same distribution, and must never enter the same `σ_run` estimate.**
>
> - **Pre-fix arms are labelled `init-variance only`**, everywhere, including
>   `outputs/olmo2_probe2_7B_keep14fresh2_seed1234` — the only run on either
>   disk with `seed != 42`, historically filed as "seed-variance evidence",
>   which it is **not**.
> - Every 7B rung in this repository predates the fix **and** has unrecorded
>   seeds (`--seed` postdates the trainer revision that produced them). No 7B
>   `sd_run` is computable or reconstructible. (`must_not_claim[23]`.)
> - The six runs A04 actually consumes as `σ_run` input — A03 keep7 seeds
>   43/44/45 and Stage-B keep12 seeds 101/102/103 — are **all post-fix**, each
>   with a positive per-run preflight assertion of the fixed line in
>   `logs/*_progress.log`. That estimate is口径-clean **on its own**; it just may
>   not be extended backwards.
> - `A04_GATE_DESIGN.md` §3.3 still quotes the **pre-fix** line
>   (`DistributedSampler(ds, shuffle=True)` at "line 863") as live. That text is
>   **stale**; 863 is now the line number inside
>   `scripts/train_olmo2_arch_probe2.py.PRE_CE5C298_BAK` on zwfy6. The fix added
>   6 comment lines above the call.

Full defect analysis: `SEED_SEMANTICS_DEFECT.md`. Verification by execution:
`STATUS.json:sampler_fix_and_pilot_one_disposition_20260812`.

---

## 8. Status of the claim, and what it would take to move it

| component of §1's claim | state 2026-08-13 |
|---|---|
| equivalence (NI) test against a null-calibrated intact target | **built, frozen, and shown to discriminate** — it changes state along a trajectory and the change is resolved on the item sample (§4.3) |
| pre-registered, multi-seed, matched-corpus/matched-token | **design exists** (`A04_GATE_DESIGN.md` §3); **not run**. Needs user approval (1 077–4 309 GPU-h) |
| "current rules accept models this rule rejects" | **exists at exactly 1 of 5 checkpoints of 1 zero-damage arm**, with RATIO's margin over ρ = +0.0015. Much weaker than when it was first reported (§4.3) |
| an accept on a **damaged** arm | **none, anywhere.** 0 accepts across keep8 / keep14 / shortgpt16 × 15 checkpoints × 5 tie conventions |
| the accept that A04 did quote | **retracted as "the accept"** — `full32@step25000` is its own trajectory's only failure (§4.3) |

**What is honestly still open**, and it is not nothing:

1. The **asymmetry argument** of §6.1 is, as far as the 2026-08-13 audit could
   determine, unowned: nobody has priced checkpoint noise as a one-sided option
   on an equivalence *accept*. It is currently a **section**, not a paper —
   its own evidence is one arm wide.
2. Whether any damage depth admits an accept **at all** is unknown. Both
   measured rungs (keep7, keep12 at 1B; keep8/14, shortgpt16 at 7B) are
   constant-REJECT, which is why Stage B "passed its kill gate and still failed
   its purpose". **This is a rung-selection problem, not a variance problem** —
   more seeds on either family cannot fix it.
3. §4.4's verbosity confound means a generative-EM decision axis in a base-LM
   regime is partly a format measurement. Fixing that is a **design** change,
   not a bigger *n*.

**No K1/K2/K3 clause has fired.** They are defined over the pre-registered **1B**
arm set; no 7B trajectory can fire them. The gate remains unauthorised.

---

## 9. SUPERSEDED — the pre-2026-08-13 text, preserved verbatim

> ⚠️ **SUPERSEDED 2026-08-13. Retained for provenance only.**
> `STATUS.json:blocked_by.still_blocking_before_any_gate_gpu[0]` recorded why:
> *"its current '提案' list is largely arXiv:2606.14150's design and its
> motivation is largely 2607.00368/2601.22950's argument."* Additionally the
> "ShortGPT/non-contiguous" arm and the "3 budgets" reading below are both
> **removed** (§7.1), and the "成功条件 / Kill 条件" below are superseded by
> `A04_GATE_DESIGN.md` §2's verbatim K1/K2/K3. **Do not cite anything in this
> section as A04's current position.**

```markdown
# A04 — Recovery Certification after Structural Injury

## 状态

**ACTIVE DESIGN。现有 Paper B 提供案例和 harness，但不是干净 scaling law。**

## 稳健起点

现有最可信结论是：

> 单条 keep14 recovery 路径上，PPL 持续改善不能单独证明已经接近 intact target。

不应直接使用现有 depth ladder 推导普遍规律，因为它混入：

- 两个大小相差 2.046× 的训练语料；
- 不同 checkpoint steps；
- historical LR grouping bug；
- 未记录的原始 seed；
- partial-shard 与 runtime drift。

## 提案

建立预注册的 recovery certification protocol：

1. 同一物理训练数据，记录 SHA256、rows、tokenizer hash；
2. 同 token presentations/FLOPs，而非只按 optimizer steps；
3. 同 optimizer、LR、batch、runtime；
4. 至少 2 seeds，主结果 3 seeds；
5. checkpoint grid 在看结果前冻结；
6. 每个 checkpoint 联合报告：
   - in-domain/OOD PPL
   - MMLU letter/content 及各自 null
   - closed-book QA
   - core likelihood/MC
   - run-level uncertainty

## 1B MVP

四个结构：

1. prefix + fresh tail
2. contiguous keep-only
3. ShortGPT/non-contiguous policy
4. random trunk + inherited lexical/readout interface

在 3 个 token budgets 上评测，2 seeds。

## 关键研究问题

- likelihood recovery 何时能认证 target recovery？
- construction 的影响是否在 matched-PPL 后仍存在？
- final block、fresh tail、继承层数量分别贡献什么？
- 哪些 stopping rule 会错误提前宣布"恢复完成"？

## 成功条件

- 结构策略差异跨 seed `≥2pp`；
- matched-PPL 下 capability 差异仍显著；或
- 明确证否 PPL certification，并给出可复现的停止规则。

## Kill 条件

- 1B 全部目标指标处于 floor；
- 多 seed 后 construction 差异被训练方差完全吞没；
- 结果只能复述"不同指标不同"，无法提出 certification rule。
```

---

## 10. Provenance — every number in §1–§8 is recomputable

| claim | source | how |
|---|---|---|
| all keep14 margins, monotonicity, popqa dip | `evidence/a04_keep14_trajectory_ni.json` | `discrimination_curve.split.<axis>.per_step`, `non_monotonicity_detail_split` |
| all neighbour ranges + noise gates | `evidence/a04_neighbour_variability.json` | `leg_A_neighbour_variability.<cluster>.per_axis.<axis>.margin_range` (`range_pp`, `expected_range_if_pure_noise_pp`, `range_exceeds_item_noise`) |
| Leg B replication failure | same | `leg_B_cross_arm_replication.replication_verdict` |
| all full32 margins, accept boundary, RATIO/NI | `evidence/a04_full32_trajectory_ni.json` | `discrimination_curve.split`, `accept_boundary`, `ratio_vs_ni_along_trajectory`, `neighbour_precondition_2_0_2` |
| §4.4 verbosity numbers | same | `em_vs_contains_verbosity_diagnostic.triviaqa.per_interval["20000->25000"]`, `em_and_contains_per_checkpoint` |
| §5 same-axis ratios 17.70 % / 30.90 % | both JSONs | `range_pp ÷ per_convention.split.delta_pp[<same axis>]`; `1.1202/6.3291`, `0.6939/2.2457` |
| §4.3 full32 5-point gate ratios (6.02× / 4.17× / 1.28×) | `evidence/a04_full32_trajectory_ni.json` | `max−min` of `margin_pp` over the 5 steps ÷ (`E[range of 5] = 2.325929` × mean `bootstrap_se_pp`) |
| §4.3 canonical 2-pt mmlu range 0.1353 pp, gate 0.2540 pp | same | `abs(0.914222 − 1.049530)` ÷ (`1.128379` × mean of the two SEs) |
| §7.2 sampler fix + Jaccard 0.0102 / 1.0000 | `STATUS.json:sampler_fix_and_pilot_one_disposition_20260812`, `code/a04_sampler_seed_probe.py`, git `ce5c298` | executed probe, both disks, md5 `284b286f90b526e4e8ad93a68e2a3b16` |
| all venue strings in §6 | `../../shared/literature/MARGIN_TRAJECTORY_INSTABILITY_NOVELTY_20260813.md` §Q4 | OpenReview `/notes/search` `venueid` + `Camera_Ready_Revision`; ACL Anthology + DBLP for the ACL family |
