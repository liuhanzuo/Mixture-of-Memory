# Novelty check — "margin along a heal trajectory wanders non-monotonically at a scale comparable to Δ"

**Date:** 2026-08-13 · **GPU: 0** (literature + CPU recompute only; no node touched).
**Dispatched by:** MAIN, after converging claim **P** from three A04 evidence streams.
**Author note:** the dispatch invited rejection of its premises. Two of them do not survive.

---

## ⛳ One-line verdict

```
NEEDS_MORE_EVIDENCE: the "wander ≈ Δ" half of P rests on ONE cell that clears A04's own
noise gate (keep8 c2 triviaqa, 1.1202 pp = 17.7% of triviaqa's OWN Δ), the cross-arm
replication of the other cell FAILED, and the third stream (full32) FAILS the gate at
0.54× noise. Separately, the general phenomenon ("downstream benchmarks are non-monotone
in training step, and checkpoint-to-checkpoint variability is a first-class quantity that
must be reported") is ALREADY PUBLISHED at NeurIPS 2025 Spotlight and measured at larger
scale on the SAME model family (OLMo-2) with the SAME axes (MMLU/TriviaQA).
If the framing is repaired to "checkpoint-selection tolerance for an EQUIVALENCE decision
on a DAMAGED arm", the residual is → SECTION_OF_A04 (not A01, not a paper).
```

**Secondary verdict, on the question actually asked:** if MAIN insists on picking one of the
four labels for the *current* framing, the answer is **`SECTION_OF_A04`**, and it is **not**
`SECTION_OF_A01` — see §Q3. It is **not** `PREEMPTED_BY` in the repo's strict sense
("essentially identical scope"), but it is much closer to preemption than any collision in
`A04/RELATED_WORK.md`, because the closest work shares the *model family, the benchmark
axes, and the exact statistic*.

---

## Q0 (unasked but load-bearing): does claim P hold on its own evidence?

I recomputed every number from the JSONs before reading a single paper. **Three corrections.**

### C1. ⛔ The "same order as Δ" comparison in the source verdicts is **CROSS-AXIS**

`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §6.1 says the loophole is worth
*"1.1202 pp — 49.9 % of popqa's entire Δ (2.2457 pp), and 60.2 % of mmlu_content's
(1.8614 pp)"*. Both ratios divide a **triviaqa** range by a **different axis's** Δ.
A margin measured on triviaqa is only ever compared to triviaqa's Δ by the NI rule
(`ni_rule` uses the same axis's `delta_pp`; verified per-cell in the JSON). The
decision-relevant, same-axis ratios are:

| arm / cluster | axis | range or move (pp) | that axis's **own** Δ (pp) | % of own Δ | clears `range_exceeds_item_noise`? |
|---|---|---|---|---|---|
| keep8 c2 (clean, 500-step) | **triviaqa** | **1.1202** | 6.3291 | **17.7 %** | **YES (1.703×)** |
| keep8 c2 | popqa | 0.2523 | 2.2457 | 11.2 % | no (0.434×) |
| keep8 c2 | mmlu_content | 0.2208 | 1.8614 | 11.9 % | no (0.339×) |
| keep8 c1 (resume seam) | all 3 | 0.119–0.214 | — | 5–10 % | no (all) |
| keep14 (25 500-step) | **popqa** | **0.6939** (margin), −0.6729 (acc) | 2.2457 | **30.9 %** | resolved p=0.0001 |
| full32 (5 000-step) | mmlu_content | 0.2280 | 1.8614 | 12.3 % | **no (0.543×)** — see C2 |

So the honest statement is **17.7 %–30.9 % of Δ on the two cells that survive**, not
"the same order as Δ". Both are still material (a 31 % shift in the quantity being tested
is not a rounding error) — but "同阶 / same order" overstates it, and the 49.9 %/60.2 %
figures must not be reused. Recompute:
`1.1202/6.3291 = 0.1770`; `0.6939/2.2457 = 0.3090`.

### C2. ⛔ The third evidence stream **fails A04's own noise gate** and is hand arithmetic

The full32 numbers in the dispatch come from `A04_FULL32_READING_B_IS_FIRING.md`
(step15000 margin **+1.2775** pp vs step25000 **+1.0495** pp). Two problems:

1. **That file forbids its own use.** Its last line: *"Do not let my hand arithmetic into the
   record. I derived the residual by subtracting a recorded null instead of importing
   `build_nulls`, which is exactly the shortcut that made my keep14 margins ~0.5 pp off.
   **Canonical output only.**"* There is no `a04_full32_trajectory_ni.json` on disk
   (`evidence/` listing 2026-08-13 08:15; only the driver + analysis script exist in `code/`).
   **The dispatch is quoting a number its own author retracted in advance.**
2. **Even taken at face value it fails the gate.** A04's guard
   (`A04_NEIGHBOUR_VARIABILITY_VERDICT.md` §2.3, and `A04_GATE_DESIGN.md` §2.0.2) requires
   `range > E[range of k iid N(0,σ)]`, with `k=2 → 2/√π = 1.12838`. Using the mean
   `mmlu_content` bootstrap SE across all six measured 7B cells (0.3723 pp):
   `E[range of 2] = 1.12838 × 0.3723 = 0.4201 pp`, and the observed range is **0.2280 pp**
   → **ratio 0.543, `range_exceeds_item_noise = False`.**

So stream (3) is **not independent corroboration**; under A04's own rule it is
**not evidence at all**. The dispatch says "全部 ACCEPT 但非单调" — the non-monotonicity is
sub-noise. Note this cuts *both* ways: it also means the full32 accept is **not** shown to
be a fluctuation. The correct reading of full32 is "we cannot tell", not "reading (b) fires".

### C3. ⚠️ The base rate is 29 %, not "generally"

Across every 7B decision-axis interval measured in the two canonical JSONs
(keep14 traj ×6, keep8 c1 ×6, keep8 c2 ×6, shortgpt16 ×6 = **24 intervals**), only
**7 (29 %)** are resolved at 95 %, and of the **6** decision-axis *range* cells only
**1 (17 %)** clears the noise gate. And the one cross-arm replication that was
pre-registered **failed**: shortgpt16's popqa over the identical 128 000→153 500 interval
moves **+0.0841 pp, CI [−0.1542, +0.3224], p = 0.5084** (opposite sign, 152 w→r vs 140 r→w)
against keep14's **−0.6729 pp, CI [−0.9252, −0.4206], p = 0.0001** (122 w→r vs 218 r→w).

### What DOES survive (and it is not nothing)

- **Existence, at 500-step spacing, in one clean single-process segment:** a single 500-step
  interval cost **−1.0867 pp** of TriviaQA EM, CI95 [−1.3319, −0.8359], p = 0.0001,
  **355 right→wrong vs 160 wrong→right of 17 944**, with 0.000 % empty predictions,
  `top_constant_frac` falling 0.440→0.368 %, distinct predictions rising 9519→9686.
  Not degeneracy, not item noise, not (per `full32_rescore_v2` and my own reading of the
  harness pinning) runtime jitter.
- **A checkpoint-selection tolerance with a number**: best-of-3 buys **+1.1202 pp** on
  triviaqa and ≤0.35 pp elsewhere, on one arm.
- **P's *normative* clause is unharmed by C1–C3.** "An accept reported at one checkpoint
  without its neighbours' margins is not interpretable" needs only **one** counterexample,
  and there are two independent ones. That clause is already adopted as
  `A04_GATE_DESIGN.md` §2.0.2.

**Net:** P's normative half stands. P's quantitative half ("游走幅度与 Δ 同阶") must be
restated as **17.7 %–30.9 % of the same axis's Δ, on 1 of 6 range cells and 7 of 24
intervals, one arm each, replication failed once**. That is a **tolerance**, not a
phenomenon.

---

## Q1. Has this been published?

**The general form: YES, and by a group working on the same models.** The narrow form
(equivalence/non-inferiority decision on a *damaged* arm, with construct-appropriate nulls):
no.

Searched (arXiv API via `hy-proxy`, `User-Agent` set — the default UA returns an empty body,
same trap `A04/RELATED_WORK.md` §1 records): checkpoint selection × LLM; non-monotonic ×
downstream × pretraining; monotonicity × benchmark; signal-to-noise × checkpoints;
checkpoint-to-checkpoint / step-to-step noise; seed variance × evaluation; non-inferiority ×
LM; equivalence test × NLP; TOST × pruning/compression/quantization; equivalence margin;
winner's curse; early-stopping optimistic bias; researcher degrees of freedom / multiverse ×
eval; layer pruning × healing; continued pretraining × recovery × pruned; pruning ×
recovery × token budget; checkpoint averaging × noise; emergence-is-a-mirage;
`ti:"Show Your Work"`; reproducible LM evaluation.

### The one that matters: Heineman et al., **NeurIPS 2025 Spotlight** (arXiv:2508.13144)

*Signal and Noise: A Framework for Reducing Uncertainty in Language Model Evaluation.*
David Heineman, Valentin Hofmann, Ian Magnusson, Yuling Gu, Noah A. Smith,
Hannaneh Hajishirzi, Kyle Lo, Jesse Dodge (Ai2).

It **defines** "noise" as *a benchmark's sensitivity to random variability between training
steps*, operationalised (§3.1 of their PDF, my extraction) as
`Rel.Std.(m) = sqrt( Σ(m_i − m̄)² / (n−1) ) / m̄` **over the final n intermediate
checkpoints**, and justifies that choice by showing it correlates with init-seed noise,
data-order noise, and whole-curve noise at **R² = 0.82 / 0.86 / 0.95** (their Figure 7;
20 purpose-trained 1B runs, 3.2 K intermediate checkpoints). It then shows noise predicts
scaling-law error, gives per-task noise for **the OLMo-2 family at 1.5B/7B/13B/32B**
(their Table 4, final 30 checkpoints @1000 steps), and prescribes an intervention:
**average the final k checkpoints instead of reporting the last one** (+2.4 % decision
accuracy on a 30-task average). Appendix A.3.2 even answers "how many neighbours do you
need": `n = 5` for a ±1σ bound, `n = 20` for ±0.2σ (34/39 benchmarks).

This is the same construct A04 rediscovered, on the same model family, including the same
two axes. Their 7B-4T noise: **TriviaQA 0.003, MMLU 0.023**. My conversion of A04's
clusters into their unit (rel.std of the checkpoint accuracies):

| A04 cell | axis | rel.std | their OLMo-2 7B-4T noise |
|---|---|---|---|
| keep8 c2 (clean, 500-step) | triviaqa | **0.0395** | 0.003 |
| keep8 c2 | mmlu_content | 0.0030 | 0.023 |
| keep14 (25.5k/46.5k) | triviaqa | 0.0203 | 0.003 |
| keep14 | popqa | 0.0432 | (not in their suite) |

**This is A04's most defensible remaining foothold and it is empirical, not conceptual:**
A04's damaged arms are **6.8–13× noisier on TriviaQA than published intact-OLMo-2 noise**
at the same scale, while being *quieter* on MMLU-content. That is a **cross-cell
comparison at n=3 vs n=30, different protocol (their OLMES/OLMo-2 setup vs A04's
`cb_bs=32 / add_bos=False / max_new_tokens=32 / max_ctx_len=512` base protocol), and
different metric conventions** — so it is a *hypothesis worth a controlled test*, **not a
result**. It must not be tabulated as though the two numbers were measured on one harness.

### The second one: Madaan et al., arXiv:2406.10229 (**preprint** — see Q4)

*Quantifying Variance in Evaluation Benchmarks.* Madaan, Singh, Schaeffer, Poulton, Koyejo,
Stenetorp, Narang, Hupkes (Meta + Stanford + UCL).
Defines **monotonicity** as the Kendall-τ between a benchmark's score sequence over 21
intermediate checkpoints and a monotone array, computes it over 10 identically-configured
Llama-2-7B runs (210 B tokens, 280 models total), and reports it per benchmark alongside
seed variance and bootstrapped 95 % CIs. **Standard MMLU's discrete monotonicity is 0.09**;
MMLU-Cloze's is **0.95** (their Table 1). It states the design conclusion A04 keeps
re-deriving — *"benchmarks with low monotonicity do not stably represent model
improvement"* — and prescribes the same class of fix (continuous / cloze formulations).

**"MMLU accuracy is nearly non-monotone in training step (τ=0.09) and a formulation change
fixes it" is published.** A04's mmlu_content plateau is a re-observation of this.

---

## Q2. Overlap, per paper: what it did / what it did not reach / what is left

| # | Cite | **Verified venue** (source URL) | What it does | What it does **not** reach | Residual for us |
|---|---|---|---|---|---|
| **P1** | Heineman, Hofmann, Magnusson, Gu, Smith, Hajishirzi, Lo, Dodge. arXiv:2508.13144 | **NeurIPS 2025 Spotlight.** OpenReview note `sAFottNlra`, `venue="NeurIPS 2025 spotlight"`, `venueid=NeurIPS.cc/2025/Conference`, invitations include `Submission26329/-/Camera_Ready_Revision`. `https://api2.openreview.net/notes/search?term=Signal+and+Noise...`. DBLP has CoRR-only (`journals/corr/abs-2508-13144`) → **S2/DBLP alone would have misread this as a preprint.** | Defines noise = rel.std over final *n* checkpoints; validates it against init-seed/data-order/whole-curve noise (R²=0.82/0.86/0.95); per-task noise for OLMo-2 1.5B–32B; noise predicts scaling-law error; SNR predicts decision accuracy; **intervention = average final k checkpoints**; guidance on choosing *n*. 465 models × 30 benchmarks, 900 K results. | (a) **Intact models only** — no structural injury, no pruning, no heal trajectory. (b) Their decision problem is **superiority/ranking** ("is corpus A better than B?") and **scaling-law extrapolation**; never an **equivalence-to-target** decision. (c) No construct-appropriate/best-constant null — their baselines are chance and other models. (d) Their prescription is *reduce* noise by averaging; they never ask what a *single-checkpoint accept* is worth, because they have no accept. | **The equivalence-decision consequence.** In a superiority test, checkpoint noise costs you power. In a **non-inferiority** test it is worse than that: it is a **one-sided free option** — you keep sampling neighbours until one clears −Δ. Nobody has priced that. Plus: is damaged-arm noise larger than intact-arm noise on the same harness (the 6.8–13× hypothesis above)? |
| **P2** | Madaan, Singh, Schaeffer, Poulton, Koyejo, Stenetorp, Narang, Hupkes. arXiv:2406.10229 | **PREPRINT.** DBLP `journals/corr/abs-2406-10229` (CoRR 2024, `https://dblp.org/rec/journals/corr/abs-2406-10229.bib`). OpenReview shows **three** notes: `M9dCa4vYgp` = `venueid=NeurIPS.cc/2024/Workshop/RegML` (`venue="RegML 2024"`, **no** Camera_Ready invitation), and `E2RyjrBMVZ` = **`venueid=ICLR.cc/2025/Conference/Rejected_Submission`**. PDF footer reads "Preprint. Under review." **Cite as: NeurIPS 2024 RegML workshop / arXiv preprint. Do NOT cite as ICLR.** | Defines benchmark **monotonicity** (Kendall τ over 21 intermediate checkpoints); 10 seed-matched 7B runs; seed variance vs bootstrap CI per benchmark; MMLU τ=0.09 → cloze τ=0.95; item analysis / IRT fail to reduce variance. | (a) Intact pretraining from scratch — no injury, no recovery. (b) Monotonicity is a **benchmark-quality score**, not a per-decision tolerance: it never converts to "how much can one checkpoint choice buy me". (c) No equivalence test, no null calibration, no damaged models. | Same as P1 plus: **the per-axis, per-arm tolerance in pp** (a τ tells you a benchmark is unstable; it does not tell you the accept threshold must move by 1.12 pp). |
| **P3** | Wen, ... arXiv:2509.02046, *Fantastic Pretraining Optimizers and Where to Find Them* | **ICLR 2026 Poster.** OpenReview `2J51qUZ0iG`, `venueid=ICLR.cc/2026/Conference`, `Submission9895/-/Camera_Ready_Revision`. | Among two named methodological shortcomings: *"comparing intermediate checkpoints before reaching the target training budgets can be **misleading**, as **rankings between two optimizers can flip during training** due to learning-rate decay."* 10 optimizers × 4 scales. | Ranking flips between **two independent runs**, driven by an identified mechanism (LR decay), in a **superiority** comparison. Not a margin against a fixed target, not equivalence, no injury, no per-axis tolerance. | **"Checkpoint choice can flip a conclusion" is P3's, not ours.** Our residual is the *equivalence* version and the *quantified* version. Must cite. |
| **P4** | Zhang, ..., Kim, Ross. arXiv:2602.01997 (already A04 `RELATED_WORK` C2) | **UNDER REVIEW** (ACL ARR May 2026; DBLP CoRR-only). Re-verified 2026-08-13: unchanged. | Layer pruning + SFT recovery: classification recovers, generative reasoning does not; ~100 B post-pruning tokens still leaves arithmetic deficits. **Owns "recovery is metric-dependent".** | One run per config; no seeds; no decision rule; no trajectory statistic; no checkpoint-selection question. | Our trajectory/tolerance layer sits on top of P4's metric-dependence, which we must credit. |
| **P5** | Nguyen et al. arXiv:2511.09864, *Uncertainty-Guided Checkpoint Selection for RFT of LLMs* | **NOT PUBLISHED.** OpenReview `WrkArtWf3u`, `venueid=ICLR.cc/2026/Conference/**Withdrawn_Submission**`; DBLP CoRR 2025. | Explicitly: *"RL finetuning ... exhibits **high variance across model checkpoints**. In practice, selecting the best checkpoint is challenging ... relying on the final checkpoint [is suboptimal]."* Proposes uncertainty-guided selection. | RLHF/RFT, not pruning recovery. It wants to **select a better** checkpoint (an optimisation goal); we want to **discount** a selected one (an inference-validity goal). Opposite direction. | It is the closest *statement* of "checkpoint variance is a real degree of freedom" in a post-training regime; cite as corroboration, note it is withdrawn. |
| **P6** | Anon. arXiv:2601.03858, *What Does Loss Optimization Actually Teach, If Anything?* | **PREPRINT.** DBLP `journals/corr/abs-2601-03858` (CoRR 2026); OpenReview note `Occy4ivWMh` is a **DBLP mirror record** (`venueid=dblp.org/journals/CORR/2026`), not a submission. | CPT: *"loss decreases monotonically while factual learning is **unstable and non-monotonic**"*; facts rarely consolidate; OOD degrades from early epochs; **"motivate evaluation of stopping criteria based on task-level learning dynamics."** 3 instruction-tuned LLMs. | Intact models + CPT on new facts (knowledge *acquisition*), not injury *recovery*. No decision rule, no equivalence, no null. Epoch-level, not 500-step. | **This is the closest paper to A04's motivation sentence and it is 7 months old.** "Loss goes down monotonically while the task metric does not, therefore stopping criteria must be task-level" is now P6's, not ours. |
| **P7** | Schaeffer, Miranda, Koyejo. arXiv:2304.15004, *Are Emergent Abilities a Mirage?* | **NeurIPS 2023 Oral.** OpenReview `ITw9edRDlD`, `venue="NeurIPS 2023 oral"`, `venueid=NeurIPS.cc/2023/Conference`, Camera_Ready present. | Discontinuous metrics manufacture apparent sharp transitions; linear/continuous metrics give smooth curves. | About *scale*, not training step; about *apparent sharpness*, not margin instability; no equivalence. | Only the general "the metric, not the model, produced the shape" lesson. Cite as ancestry for "EM is a coarse read". |
| **P8** | Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi. **OLMES** | **Findings of NAACL 2025.** ACL Anthology `10.18653/v1/2025.findings-naacl.282` (`https://aclanthology.org/2025.findings-naacl.282/`, `citation_conference_title = "Findings of the Association for Computational Linguistics: NAACL 2025"`); DBLP `conf/naacl/GuTKHDH25`. | The eval standard A04's protocol is measured against. | **Contains no checkpoint-neighbourhood requirement** — I checked: it standardises prompt/format/interface, not trajectory reporting. | Answers the dispatch's Q1 last bullet: **"report a range over neighbouring checkpoints" is NOT in an existing benchmark standard.** P1 prescribes *averaging* the final k; nobody prescribes *reporting the range as a robustness check on a decision*. That gap is real, and small. |
| **P9** | Biderman et al. arXiv:2405.14782 (*Lessons from the Trenches*) · Miller arXiv:2411.00640 (*Adding Error Bars to Evals*) | **BOTH PREPRINT.** DBLP CoRR-only for both (`journals/corr/abs-2405-14782`, `journals/corr/abs-2411-00640`); DBLP XML search returns "Informal and Other Publications | CoRR 2024" for each; OpenReview has only DBLP mirror records. ⚠️ *Lessons* is widely cited as COLM 2024 — **I could not verify that** through Anthology, DBLP or OpenReview and therefore do **not** assert it. | The two standard "how to report an eval" documents. | Neither mentions trajectory/checkpoint-neighbourhood reporting; *Error Bars* is item-level (super-population) statistics only. | Confirms the reporting-standards gap from a second angle. |

### The A04 `RELATED_WORK.md` claim I independently re-verified — and it holds

§4.1 item 1 says TOST/equivalence testing is essentially absent from pruning-recovery.
I re-ran it independently on 2026-08-13 with different query strings
(`"two one-sided tests" AND (pruning OR compression OR distillation)`,
`"non-inferiority" AND (pruning OR compression OR quantization)`, `"equivalence margin"`):
**zero pruning/recovery hits.** Confirmed. Two adjacent finds worth knowing:
arXiv:2603.16213 (*Equivalence testing with data-dependent and post-hoc equivalence
margins*) — **directly relevant to A04's `Δ = 0.10 × residual(intact)`, which is
data-dependent**; and arXiv:2608.03169, a prespecified equivalence study on LM agents.
Neither is pruning. **A04's equivalence-direction novelty survives; its Δ construction now
has a statistics paper to answer to.**

---

## Q3. A01 section, A04 section, or its own paper? → **A04 section.** Not A01.

### Why not A01 (this is the clearer half)

A01 (now `paperC/`) is **input-blind null calibration**: is a *reported number*
distinguishable from a *baseline that ignores the input*? Its three degrees of freedom are
tie policy, length unit, tokenizer — all properties of the **null**, computed from the
**dataset**, with **no model and no training** involved. A01's letter floor is asserted
*invariant across all 15 arms* precisely because it is a pure dataset property.

Checkpoint-selection instability is a property of the **arm's trajectory**. It moves the
*reported* value, never the null. Nothing in A01's `PROPOSAL.md`, `claims/`, `STATUS.json:
claim_scope_after_gates`, or `paperC/README.md` mentions training steps, trajectories or
checkpoints — I checked all four. Putting a trajectory finding in A01 would import a
second, unrelated construct into a paper that is already in **MAJOR REVISION** with six
live retractions/narrowings (`paperC/README.md` "Scope discipline"). A01's own hard-won
lesson is the argument against: *"before calling a degree of freedom 'N equally defensible
choices', ask whether our baseline can actually execute each one"* — checkpoint choice is
not a choice the *null* makes.

### Why not its own paper (yet)

1. **The general phenomenon is P1's** (NeurIPS 2025 Spotlight), measured on the same model
   family, the same axes, at 30 checkpoints vs our 3, with a validated statistic and a
   prescribed fix. A paper whose contribution is "downstream margins wander
   non-monotonically" would be re-deriving P1 at 1/10th the scale.
2. **The motivation sentence is P6's** ("loss monotone, task metric not, so stopping
   criteria must be task-level", 2026-01) and the ranking-flip observation is **P3's**
   (ICLR 2026 Poster).
3. **The evidence is 1 range cell of 6 and 7 intervals of 24, one arm each, with the one
   pre-registered cross-arm replication failing.** By the repo's own standard —
   `memory/one-sample-is-not-a-trend-or-a-state` — that is not a phenomenon.
4. **Cross-node reproducibility is not yet clean.** `Generator.multinomial` differs in
   19/10 000 rows between numpy 2.5.1 (.73) and 2.4.6 (.82), max margin drift
   **0.005294 pp**, which is **10.6× looser than the 5e-4 pp hard-fail** in
   `a04_keep14_trajectory_ni.py`. Small vs the findings (211×), but a paper-level claim
   about a 1.12 pp effect should not ship on a cluster where the analysis layer is
   node-dependent (`memory/numpy-version-split-breaks-cross-node-bootstrap`).

### Why it IS a good A04 section — and where its real teeth are

A04 is *already* the paper about **when you are allowed to stop a recovery run**, and
§2.0.2 already adopted the precondition. The section writes itself in one page, and it has
a genuinely novel argument that P1 cannot make because P1 has no accept:

> **In a superiority comparison, checkpoint noise costs power and averaging fixes it
> (P1). In a non-inferiority comparison it is a one-sided option: the analyst only needs
> *one* neighbour to clear −Δ, and the same noise that P1 treats as symmetric becomes an
> asymmetric inflation of the accept rate. We measure the size of that option
> (1.1202 pp = 17.7 % of that axis's Δ, 1.70× the item-noise range floor) and turn it into
> a per-axis precondition.**

That is *"we corroborate P1's construct and extend it to the equivalence-decision regime
on damaged arms"* — legitimate and strong per `RELATED_WORK.md` §4's 2026-08-12 reframe and
`memory/prior-work-differentiate-dont-abandon`. **It is a section, not a paper**, because
its own evidence is one arm wide.

**⚠️ This is a RECOMMENDATION. Final framing is the user's call.** If the user wants a
paper here, the cheapest path to earning it is in "what would upgrade this" below — the
`ELIGIBLE_FOR_PROMOTION` route is not closed, it is unfunded.

### What would upgrade `NEEDS_MORE_EVIDENCE` → a paper (cheapest first)

1. **(≈0 GPU-h, do first)** Produce the canonical `a04_full32_trajectory_ni.json` and
   report its mmlu_content range **against the noise gate**, not by hand. Either it clears
   0.4201 pp or it does not; right now the record contains a hand-computed number its own
   author forbade.
2. **(0 GPU-h)** Recompute every A04 cluster in P1's unit (rel.std over the checkpoints) and
   put it beside P1's Table 4 — with the protocol mismatch stated. If damaged arms really
   are 6.8–13× noisier on TriviaQA, **that is the paper**, and it is P1-shaped (a property
   of an *injured* model's eval, which P1 never measured).
3. **(cheap, eval-only)** One more arm × one more clean 3-checkpoint cluster, to get the
   range cells from 1/6 to ≥2/12. §6's "declined until there is an accept to protect" is
   right for the *gate*, but a **paper** needs the second arm.
4. **(0 GPU-h, blocking)** Pin numpy cluster-wide, or loosen the 5e-4 pp assertion to
   ≥6e-3 pp with a disclosed reason.

---

## Q4. Venue verification, per repo rule (two families, two authorities)

Method: OpenReview `api2.openreview.net/notes/search` (`venueid` + look for a
`Camera_Ready_Revision` invitation) for ICLR/NeurIPS/ICML/TMLR; ACL Anthology + DBLP for
ACL-family. `User-Agent` set on every call. **Semantic Scholar not used as an authority.**

| Cite | Verified venue string | Authority + URL | CR invitation? |
|---|---|---|---|
| arXiv:2508.13144 | `NeurIPS 2025 spotlight`, `venueid=NeurIPS.cc/2025/Conference` | OpenReview note `sAFottNlra` — `https://api2.openreview.net/notes/search?term=Signal%20and%20Noise:%20A%20Framework%20for%20Reducing%20Uncertainty%20in%20Language%20Model%20Evaluation&content=all` | **YES** (`Submission26329/-/Camera_Ready_Revision`) |
| arXiv:2406.10229 | `RegML 2024`, `venueid=NeurIPS.cc/2024/Workshop/RegML` **and** `venueid=ICLR.cc/2025/Conference/Rejected_Submission` | OpenReview notes `M9dCa4vYgp`, `E2RyjrBMVZ`; DBLP `https://dblp.org/rec/journals/corr/abs-2406-10229.bib` = CoRR 2024 | **NO** → cite as **workshop / preprint**, never ICLR |
| arXiv:2509.02046 | `ICLR 2026 Poster`, `venueid=ICLR.cc/2026/Conference` | OpenReview note `2J51qUZ0iG` | **YES** (`Submission9895/-/Camera_Ready_Revision`) |
| arXiv:2304.15004 | `NeurIPS 2023 oral`, `venueid=NeurIPS.cc/2023/Conference` | OpenReview note `ITw9edRDlD` | **YES** |
| arXiv:2504.11393 (DataDecide) | `ICML 2025 poster`, `venueid=ICML.cc/2025/Conference` | OpenReview note `p9YlQPF8fE` | **YES** |
| arXiv:2509.23024 | `NeurIPS 2025 poster`, `venueid=NeurIPS.cc/2025/Conference` | OpenReview note `FDruZlKWUb` (also HiLD@ICML 2025 workshop note `9nKmDLXg9v`) | **YES** |
| arXiv:2406.08446 (OLMES) | `Findings of the Association for Computational Linguistics: NAACL 2025`, DOI `10.18653/v1/2025.findings-naacl.282` | **ACL Anthology** `https://aclanthology.org/2025.findings-naacl.282/` (`citation_conference_title` meta) + DBLP `conf/naacl/GuTKHDH25` | n/a (ACL family) |
| arXiv:2511.09864 | `ICLR 2026 Conference **Withdrawn** Submission`, `venueid=ICLR.cc/2026/Conference/Withdrawn_Submission` | OpenReview note `WrkArtWf3u`; DBLP CoRR 2025 | **NO** — withdrawn |
| arXiv:2601.03858 | **PREPRINT** (CoRR 2026) | DBLP `journals/corr/abs-2601-03858`; OpenReview `Occy4ivWMh` is a **DBLP mirror** (`venueid=dblp.org/journals/CORR/2026`), not a submission | **NO** |
| arXiv:2405.14782 | **PREPRINT** (CoRR 2024) — ⚠️ commonly cited as COLM 2024; **not verified** | DBLP XML search → "Informal and Other Publications \| CoRR 2024"; OpenReview `CfKbLzyDKU` = DBLP mirror | **NO** |
| arXiv:2411.00640 | **PREPRINT** (CoRR 2024) | DBLP `journals/corr/abs-2411-00640` | **NO** |
| arXiv:2602.01997 | **UNDER REVIEW** — ACL ARR May 2026 | unchanged from `A04/RELATED_WORK.md` C2 (re-checked) | n/a |
| arXiv:2603.16213 | **PREPRINT** (arXiv only; not chased further — cited for its statistical point, not its status) | arXiv API | — |

### Operational notes for the next agent

- **`api2.openreview.net/notes?forum=...` and `?content.title=...` now return HTTP 403
  `ChallengeRequiredError`.** Only `/notes/search?term=...&content=all` still works
  unauthenticated (2026-08-13). Plan queries around `/notes/search`.
- **DBLP `/search/publ/api?format=json` returns unparseable output; `format=xml` works.**
  Per-record `.bib` URLs also work. (A04's note said the endpoint 500s; today it is a
  format problem, not an outage.)
- **A DBLP-mirror OpenReview note looks like a venue hit.** `venueid=dblp.org/journals/
  CORR/2026` with `invitations=['DBLP.org/-/Record', ...]` is **a preprint**, not a
  publication. Reading only the `venue` field ("CoRR 2026") is fine; reading only
  "there is an OpenReview note" is a trap.
- **arXiv API needs a non-default `User-Agent`** (empty body otherwise). Still true.

---

## 「不得主张」 — must NOT claim

1. ❌ **"the wander is the same order as Δ" / "49.9 % of Δ" / "60.2 % of Δ".**
   Cross-axis. Same-axis: **17.7 %** (keep8 c2 triviaqa) and **30.9 %** (keep14 popqa).
2. ❌ **Any use of the full32 step15000/step25000 margins as evidence.** MAIN's own hand
   arithmetic, explicitly forbidden by the file that contains it, no canonical JSON on disk,
   and it **fails A04's own noise gate at 0.543×**. Re-derive with `build_nulls` first.
3. ❌ **"reading (b) is firing" / "the full32 accept is a fluctuation."** Sub-noise means
   *undetermined*, not *refuted*. Symmetric caution: also do not claim it is convergence.
4. ❌ **First to observe non-monotone downstream benchmark development during training.**
   **P2 (arXiv:2406.10229) owns it**, with an explicit monotonicity statistic and MMLU τ=0.09.
5. ❌ **First to treat checkpoint-to-checkpoint variability as a first-class eval quantity.**
   **P1 (NeurIPS 2025 Spotlight) owns it**, defines it, validates it against seed and
   data-order noise, and prescribes averaging.
6. ❌ **First to note that comparing at a hand-picked checkpoint can flip a conclusion.**
   **P3 (ICLR 2026 Poster) owns it** (rankings flip during training under LR decay).
7. ❌ **First to argue that a monotone loss with a non-monotone task metric means stopping
   criteria must be task-level.** **P6 (arXiv:2601.03858, 2026-01) owns it.**
8. ❌ **"Damaged arms are noisier than intact arms"** — as a *result*. It is a hypothesis
   from a **cross-protocol, n=3-vs-n=30, different-harness** comparison. Needs a controlled
   same-harness test before it may be tabulated.
9. ❌ **Treating checkpoints within a cluster as replicates**, or reading any of this as seed
   variance. No 7B `sd_run` exists or is reconstructible. (Carried from both source verdicts.)
10. ❌ **"the popqa mid-heal dip is a property of healing."** Leg B refuted it
    (shortgpt16: +0.0841 pp, p = 0.5084, opposite sign). It is an **existence proof about
    checkpoint selection**, nothing more.
11. ❌ **Reporting the 7 sub-noise ranges (0.119–0.332 pp) as measured gaps.**
12. ❌ **Quoting any margin to better than 0.01 pp across nodes** (numpy multinomial drift,
    max 0.005294 pp).
13. ❌ **Citing arXiv:2405.14782 as COLM 2024** without an Anthology/DBLP hit. I could not
    find one. Same for arXiv:2406.10229 as ICLR 2025 (it is a
    **`Rejected_Submission`** + RegML workshop note).
14. ❌ **Putting any of this in A01/`paperC`.** Different construct: this moves the
    *reported* value, A01's degrees of freedom move the *null*.
15. ❌ **"A04's Δ construction is standard."** `Δ = 0.10 × residual(intact)` is a
    **data-dependent** equivalence margin; arXiv:2603.16213 is about exactly that problem
    and must be engaged before the margin is defended as pre-registered.

---

## Provenance — every number here is recomputable

| Claim | Source | How |
|---|---|---|
| 17.7 % / 30.9 % same-axis ratios | `evidence/a04_neighbour_variability.json`, `evidence/a04_keep14_trajectory_ni.json` | `margin_range.range_pp ÷ per_step[*].delta_pp` (same axis); `1.1202/6.3291`, `0.6939/2.2457` |
| noise floor 0.4201 pp for k=2 | same two JSONs | `1.12838 × mean(mmlu_content bootstrap_se_pp over the 6 measured 7B cells = 0.3723)` |
| full32 range 0.2280 pp | `A04_FULL32_READING_B_IS_FIRING.md` (**hand arithmetic; not canonical**) | `1.2775 − 1.0495` |
| 7/24 intervals resolved; 1/6 range cells clear | same two JSONs | iterate `*_paired_tests[*].distinguishable_from_zero_at_95` and `margin_range.range_exceeds_item_noise` over decision axes only |
| rel.std conversions (0.0395 etc.) | same two JSONs | `stdev(acc)/mean(acc)` over each cluster's `per_step[*].acc` |
| P1 OLMo-2 7B-4T noise (TriviaQA 0.003, MMLU 0.023) | arXiv:2508.13144v1 Table 4 | `pdftotext -layout -f 23 -l 24` |
| P2 MMLU τ=0.09 → cloze 0.95 | arXiv:2406.10229v1 §3.3 + Table 1 | `pdftotext` |
| all venue strings | §Q4 table | URLs listed there |
