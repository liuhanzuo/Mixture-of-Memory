---
gate: A01 kill clause 3 — novelty / prior-art boundary
date: 2026-08-09
compute: CPU + web only, ZERO GPU
verdict: KILL CLAUSE 3 DOES NOT FIRE
---

> **⚠️ 2026-08-10 SCOPE NOTICE (verdict UNCHANGED).** An external audit
> (`../../archive/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md` §2.1)
> returned **Major revision** on A01. It attacks claim STRENGTH, not originality, so
> **kill clause 3 still does not fire**. But two things this file leans on have moved:
> 1. The phrase "**a letter readout is a step function of depth**" (§2.4's gap
>    argument) has been RETRACTED as a family-general claim — read it as "a large
>    single-layer letter jump in three of four families". See
>    `TCODEX_AUDIT_RESPONSE.md` §1.
> 2. `STATUS.json:novelty_check.strongest_remaining_novel_claim` used to name the
>    tie-convention finding. That was **demoted** (the three executable conventions
>    move the null 0.3365 pp and flip 0/6 arms; the 25.76 pp / 5-of-6 figures come
>    from two non-executable bounds). The strongest remaining A01-owned claim is now
>    the **tokenizer-dependence** of the longest-option null (0.9003 pp span, 2.68×
>    the executable convention span, 1 robust verdict flip in 63 arms). See
>    `TCODEX_AUDIT_RESPONSE.md` §3 and §6.
>
> The citation obligations and per-candidate venue verifications below are unaffected.

# A01 — novelty check against the third kill clause

## 0. The clause being tested, verbatim

`PROPOSAL.md` §Kill 条件, clause 3, verbatim:

> `- 论文只能退化为已有 similarity-null 方法的案例集合。`

("the paper can only degenerate into a case collection of existing similarity-null
methods.")

And the matching success condition, verbatim:

> `- 与已有 similarity-null prior art 的边界经正式 venue/全文核实。`

**Read the clause literally, because that is what it says.** Clause 3 does *not* ask
"is any part of A01 novel?" It asks a narrower question: *is A01 nothing but a set of
worked examples of already-published **similarity-null** methodology?* "similarity-null"
is the representation leg (C3: permutation nulls for CKA / representational similarity),
which is the one leg where A01 already concedes in its own `## 新颖性边界` section that
it **cannot** claim priority ("不能主张：首创 permutation null calibration；首创 BH；
表征相似性文献没有 null").

So the clause fires **iff** everything A01 contributes reduces to applying
similarity-null machinery to new data. It does **not** fire merely because some paper
overlaps some A01 leg. Under this repo's standing directive
(`prior-work-differentiate-dont-abandon`), the bar is "essentially identical scope", and
work released within 2–3 months is **concurrent**, not preempting.

## 1. Verdict

**KILL CLAUSE 3 DOES NOT FIRE.** Two independent reasons:

1. **The clause is about the similarity-null leg, and A01's load-bearing findings are
   not on that leg.** The findings that survived the gates — letter-interface collapse to
   at/below the best-constant floor under damage across four families
   (`GATE1_DAMAGED_VERDICT.md`), the letter-as-step-function-of-depth vs
   content-as-smooth contrast (`GATE1_DEPTHCURVE_VERDICT.md`), the fp32 refutation of the
   tie mechanism (`GATE3_VERDICT.md`), and the convention-sensitivity of the
   longest-option null itself (`evidence/gate3_content_null_conventions.json`) — are all
   *accuracy*-construct results on MC benchmarks. None of them is an instance of a
   similarity-null method. Even if every published similarity-null paper existed exactly
   as it does, these results would still be unpublished.
2. **No verified prior work does A01's scope.** The closest verified work is Balepur et
   al. (ACL 2024 main), which argues for *stronger baselines* in MCQA and measures
   choices-only accuracy against a majority baseline — but on **intact** models, as a
   *dataset*-artifact claim, with no damage axis, no depth curve, no best-constant floor
   as a per-arm validity gate, and no convention analysis of the null. See §2.1.

The one genuinely uncomfortable finding is **not** a preemption but a **scooping of one
sub-claim**: `arXiv:2607.12767` (ICML 2026) independently establishes that
length-normalised accuracy over-corrects, which overlaps A01's "acc vs acc_norm is not a
free axis" side observation. A01 must cite it and must not claim that observation as
novel. That is a citation obligation, not a kill. See §2.2.

## 2. Candidates, in decreasing closeness

### 2.1 Balepur, Ravichander, Rudinger — *Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?*

* **Citation.** Nishant Balepur, Abhilasha Ravichander, Rachel Rudinger. "Artifacts or
  Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?"
  *Proceedings of the 62nd Annual Meeting of the Association for Computational
  Linguistics (Volume 1: Long Papers)*, pp. 10308–10330, Bangkok, Thailand, Aug 2024.
  DOI `10.18653/v1/2024.acl-long.555`. arXiv:2402.12483v2.
* **Verified venue: ACL 2024 MAIN conference, Volume 1 Long Papers.** Verified via **ACL
  Anthology** (`https://aclanthology.org/2024.acl-long.555.bib`, fetched 2026-08-09),
  which returns `booktitle = "Proceedings of the 62nd Annual Meeting of the Association
  for Computational Linguistics (Volume 1: Long Papers)"`, `pages = "10308--10330"`,
  `doi = "10.18653/v1/2024.acl-long.555"`. This is the **ACL family**, so per this repo's
  two-family rule the authority is Anthology, not OpenReview. Note the anthology ID
  `acl-long`, **not** `findings-acl` — this is main-conference, not Findings.
  (The paper's own arXiv comment field says only "ACL 2024"; the Anthology record is what
  establishes main-vs-Findings.)
* **Precise overlap.** (a) Uses an *input-blind* prompt (choices-only, question removed).
  (b) Compares model accuracy to a **majority baseline** rather than to chance, and
  reports "bests a majority baseline in 11/12 cases". (c) Its explicit recommendation is
  "we advocate for the use of stronger baselines in MCQA benchmarks" — which is A01's
  protocol recommendation in one sentence. This is the single closest statement of A01's
  headline that exists in a verified venue.
* **Precise gap.**
  1. **Different object of study.** Balepur et al. ask whether the *dataset/benchmark* is
     cheatable by an LLM that never sees the question. A01 asks whether the *readout
     interface* remains a valid instrument for a *given model under structural damage*.
     Balepur's baseline is a property of the data; A01's floor is used as a per-arm
     *validity gate* that decides whether a specific arm's number may enter a comparison
     at all.
  2. **No damage axis at all.** Every model in Balepur et al. is intact. A01's confirmed
     result is specifically that damaged models fall **to/below** their own
     best-constant floor (6/6 non-OLMo truncated arms, `GATE1_DAMAGED_VERDICT.md`), and
     A01's own intact-base gate (`GATE1_VERDICT.md`) found 0/3 intact families showing the
     pathology. Balepur et al. measure the regime A01 shows is the *uninformative* one for
     this question.
  3. **No depth/phase-transition structure.** A01's Qwen3 curve is pinned within
     0.0855 pp across twelve measured depths k=4…24 and then moves +48.02 pp at k24→k25
     while content_norm moves +1.35 pp over the same forward passes (verified this session
     from `olmo2_mmlu_content_results/gate1_dmg_qwen3_8b_depth_k{24,25}/summary.json` on
     zwfy6). Nothing analogous exists in Balepur et al.
  4. **No treatment of the null's own convention degrees of freedom.** See §3.
  5. **No self-retraction leg.**
* **Preempts? NO.** Overlap is on the *recommendation* ("use stronger baselines"), not on
  the scope, the object, or the findings. A01 must cite it as the origin of the
  stronger-baseline argument in MCQA and must **not** claim priority for "compare MC
  accuracy to a majority/best-constant baseline rather than chance". A01's contribution
  relative to it is the damage axis, the interface-as-instrument framing, and the
  convention analysis. This is exactly the "differentiate / position as follow-up" case.

### 2.2 Oostermeijer — *Accuracy and Normalized Accuracy under Length Bias: Analysis, Guidelines, and a Bayesian Alternative*

* **Citation.** Koen Oostermeijer. "Accuracy and Normalized Accuracy under Length Bias:
  Analysis, Guidelines, and a Bayesian Alternative." arXiv:2607.12767v1, 2026-07-14.
* **Verified venue: ICML 2026 (main conference).** Verified via **OpenReview** (the
  correct authority for the ICML family): forum `SbSFZ9N6DN`,
  `venueid = "ICML.cc/2026/Conference"`, `venue = "ICML 2026 regular"`, and invitations
  include `ICML.cc/2026/Conference/Submission34463/-/Camera_Ready_Revision`. Both the
  `venueid` and the `Camera_Ready_Revision` requirements of this repo's rule are met.
  Semantic Scholar was rate-limited (HTTP 429) and DBLP returned HTTP 500 during this
  session — which is precisely why the OpenReview route is mandated.
* **Timing.** OpenReview `cdate` = 2026-01-24, `pdate` = 2026-04-30 (both computed from
  the API's epoch-ms fields this session). Both **predate** the concurrency window
  (which for a 2026-08-09 check starts 2026-05-09). So this is **not** concurrent work;
  it is established prior art that A01 must cite.
* **Precise overlap.** It shows that length-normalised accuracy (`acc_norm`) does not
  merely fix the length bias of raw sum-log-prob accuracy but **over-corrects**, and that
  which of the two is appropriate depends on the completion-length distribution. A01's
  gate-2 leg reports arm-pair sign flips between `acc` and `acc_norm` on OpenBookQA
  (3/15 pairs, per `STATUS.json:gate2_second_mc_benchmark`) and treats "the interface is
  not a free axis" as a finding. **That sub-claim is preempted for the length-bias
  mechanism.**
* **Precise gap.** Its object is a *scoring-rule* bias present in healthy evaluation, and
  its remedy is a better scoring rule (Bayesian accuracy). It does not use an input-blind
  null as a validity gate, does not study damaged/pruned models, and does not report a
  letter-vs-content interface contrast. Conversely, A01's `longest_option` null is a
  *length-derived input-blind baseline*: the sharpest quantitative statement in §3 below
  (the same MMLU items admit nulls from 0.1961 to 0.4537 purely by tie convention) is a
  statement about the null, not about the scoring rule, and is absent from this paper.
* **Preempts? NO — but it removes one A01 sub-claim.** A01 must (i) cite it, (ii) stop
  presenting acc-vs-acc_norm length sensitivity as its own discovery, and (iii) reframe
  the OBQA sign-flip observation as a *replication in a new setting* (damaged models) of
  an established scoring-rule pathology. The rest of A01 is untouched.

### 2.3 Cho, So, Lee — *Choices Speak Louder than Questions*

* **Citation.** Gyeongje Cho, Yeonkyoung So, Jaejin Lee. "Choices Speak Louder than
  Questions." arXiv:2502.18798v4 (v1 2025-02-26, v4 2026-01-12).
* **Verified venue: ICLR 2026 Poster.** Verified via **OpenReview**:
  `venueid = "ICLR.cc/2026/Conference"`, `venue = "ICLR 2026 Poster"`, invitations include
  `ICLR.cc/2026/Conference/Submission17078/-/Camera_Ready_Revision`.
* **Precise overlap.** Directly attacks the *scoring interface* of MCQA: shows that
  log-likelihood scoring and its length-normalised variant are "vulnerable to superficial
  characteristics of the answer choices", experiments across cloze / symbol / hybrid input
  formats (which is A01's content-vs-letter axis under different names), and proposes a
  replacement metric (NPSQ) that isolates the question's contribution. The "isolate the
  question's contribution" move is conceptually an input-blind subtraction.
* **Precise gap.** Its remedy is a *new metric* proposed as generally better; A01's
  position is metric-agnostic — report **any** metric against **that metric's own**
  input-blind null and refuse cross-arm comparison when an arm cannot clear it. Its models
  are intact; no damage axis, no depth curve, no per-arm floor gate, no convention
  analysis, and no phase-transition claim. It also does not report a case where the
  interface choice makes a *degenerate* arm look competent — which is A01's C5.
* **Preempts? NO.** Same axis (MC scoring interface validity), different object (better
  metric vs. validity gate), disjoint regime (intact vs damaged). It is the strongest
  "someone else is worried about the same thing" citation and should be cited as
  concurrent-adjacent independent motivation. **A01 must, however, drop any framing that
  presents "letter-vs-content interface choice can invalidate MCQA conclusions" as
  unprecedented** — this paper and §2.1 jointly establish that concern.

### 2.4 Bean, Kearns, Romanou, et al. — *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*

* **Citation.** Andrew M. Bean, Ryan Othniel Kearns, Angelika Romanou, Franziska Sofia
  Hafner, Harry Mayne, Jan Batzner, Negar Foroutan, Chris Schmitz, Karolina Korgul, Hunar
  Batra, Oishi Deb, Emma Beharry, et al. "Measuring what Matters: Construct Validity in
  Large Language Model Benchmarks." arXiv:2511.04703v1, 2025-11-03.
* **Verified venue: NeurIPS 2025 Datasets and Benchmarks Track (poster).** Verified via
  **OpenReview**: forum `mdA5lVvNcU`,
  `venueid = "NeurIPS.cc/2025/Datasets_and_Benchmarks_Track"`,
  `venue = "NeurIPS 2025 Datasets and Benchmarks Track poster"`, invitations include
  `.../Submission1976/-/Camera_Ready_Revision`.
* **Precise overlap.** It is *the* construct-validity paper for LLM benchmarks: a
  29-reviewer systematic review of 445 benchmarks, finding validity-undermining patterns
  in measured phenomena, tasks and **scoring metrics**, and issuing eight
  recommendations. It owns the "construct validity" vocabulary A01 uses.
* **Precise gap.** It is a **survey with normative recommendations**; it does not run the
  measurement. A01 is the opposite shape: a small number of constructs, measured, with
  per-item paired bootstraps, and with the protocol turned against the authors' own prior
  headline. A survey establishing that scoring metrics threaten validity does not contain
  the finding that a letter readout is a *step function of depth* while a content readout
  over the same forward passes is smooth.
* **Preempts? NO.** A01 must cite it as the framing authority and must not claim to have
  introduced construct validity to LLM evaluation. A01's positioning should be explicitly
  "an instance of what this survey asks for, executed, plus two findings the survey's
  recommendations do not anticipate".

### 2.5 Zheng, Pang, Du, Liu, Jiang, Lin — *Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates*

* **Citation.** Xiaosen Zheng, Tianyu Pang, Chao Du, Qian Liu, Jing Jiang, Min Lin.
  "Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates."
  arXiv:2410.07137v2.
* **Verified venue: ICLR 2025 Oral.** Verified via **OpenReview**:
  `venueid = "ICLR.cc/2025/Conference"`, `venue = "ICLR 2025 Oral"`. (A second OpenReview
  record shows a NeurIPS 2024 SafeGenAi workshop version,
  `venueid = "NeurIPS.cc/2024/Workshop/SafeGenAi"` — the conference record is the one to
  cite.)
* **Precise overlap.** Owns the phrase "**null model**" in LLM evaluation and demonstrates
  the exact structural point A01 makes: a **constant, input-independent output** scores
  highly on a benchmark that everyone reads as measuring capability (86.5% LC win rate on
  AlpacaEval 2.0). This is the closest thing to A01's thesis stated in the abstract of a
  verified top-venue paper.
* **Precise gap.** Its setting is **LLM-judge win-rate benchmarks** and its frame is
  **adversarial gaming** — an attacker crafts a constant response to cheat a leaderboard.
  A01's setting is likelihood-scored MC accuracy and its frame is **diagnostic**: the
  constant predictor is not an attacker, it is the *reference point* against which an
  honest arm's number must be judged, and the failure is not gaming but a model whose
  readout has degenerated into that constant. Also: their null is crafted (a chosen cheat
  string); A01's null is *derived* from the benchmark's own label/length statistics, which
  is what makes it a per-construct calibration rather than an attack.
* **Preempts? NO.** Different benchmark family, different threat model, but it is a
  mandatory citation and A01 must not present "a constant predictor can beat a benchmark"
  as new. A01's version is "a constant predictor is the correct *reference*, and real
  damaged models land at or below it".

### 2.6 Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi — *OLMES: A Standard for Language Model Evaluations*

* **Citation.** Yuling Gu, Oyvind Tafjord, Bailey Kuehl, Dany Haddad, Jesse Dodge,
  Hannaneh Hajishirzi. "OLMES: A Standard for Language Model Evaluations." *Findings of
  the Association for Computational Linguistics: NAACL 2025*, pp. 5020–5048, Albuquerque,
  New Mexico, Apr 2025. DOI `10.18653/v1/2025.findings-naacl.282`. arXiv:2406.08446v2.
* **Verified venue: Findings of NAACL 2025** (i.e. **Findings**, not main). Verified via
  **ACL Anthology** bibtex (`https://aclanthology.org/2025.findings-naacl.282.bib`,
  fetched 2026-08-09): `booktitle = "Findings of the Association for Computational
  Linguistics: NAACL 2025"`, ISBN `979-8-89176-195-7`. ACL family → Anthology is the
  authority; the Findings-vs-main distinction is exactly the trap this repo's
  `venue-verify-acl-family-needs-anthology` memory warns about, so it is stated explicitly.
* **Precise overlap.** It is the standard that *names* A01's two interfaces: it explicitly
  supports "meaningful comparisons between smaller base models that require the unnatural
  'cloze' formulation of multiple-choice questions against larger models that can utilize
  the original formulation", and reviews "probability normalizations" as a
  result-changing free parameter. So the letter-vs-cloze/content interface split and the
  fact that it changes measured performance are **established, standardised, and not
  A01's**.
* **Precise gap.** OLMES *prescribes* which interface to use as a function of model size,
  so that numbers are comparable. A01 shows that prescription is not sufficient: OLMES's
  size-based rule would put a damaged 7B model on the letter interface, where A01 measures
  it at/below its own best-constant floor (keep8 letter 0.2550 vs floor 0.2689, fp32
  −1.538 pp, boot p = 0.0062 per `evidence/gate3_dtype_runs/7B_keep8_step121000_dtype_summary.json`
  as reported in `STATUS.json`). **A01's honest positioning w.r.t. OLMES is a follow-up
  that identifies a defect: an interface-selection rule keyed on model *size* fails on
  models whose readout competence has been damaged, and the fix is to key it on the
  arm's own floor test.** That is a constructive, citable delta.
* **Preempts? NO.**

### 2.7 The similarity-null leg (the one the clause is actually about)

Named explicitly because clause 3 is about this leg and it must be shown to *not* carry
the paper.

* **Ding, Denain, Steinhardt.** "Grounding Representation Similarity with Statistical
  Testing." arXiv:2108.01661v2. Venue per the paper's own arXiv comment field: "Accepted
  at NeurIPS 2021". **UNVERIFIED at the OpenReview/`venueid` standard** — not re-checked
  against OpenReview in this session. Treat the venue as author-asserted until verified.
  Establishes sensitivity/specificity testing for CKA-style measures; this is the prior art
  A01 already concedes.
* **Hewitt, Liang.** "Designing and Interpreting Probes with Control Tasks."
  arXiv:1909.03368v1, comment "EMNLP 2019". **UNVERIFIED at the Anthology standard** —
  not fetched from ACL Anthology in this session. Establishes control tasks / selectivity,
  the canonical "your probe needs a null" result, directly relevant to A01's C4 probe leg.
* **Feng, Wallace, Boyd-Graber.** "Misleading Failures of Partial-input Baselines."
  arXiv:1905.05778v3, comment "ACL 2019". **UNVERIFIED at the Anthology standard.**
  Important as a *counter*-citation: it shows a partial-input (input-blind) baseline
  *failing* does not certify a dataset is artifact-free. A01 must therefore not claim that
  clearing the floor certifies validity — clearing the floor is necessary, not sufficient.
  A01's own claims are already one-directional in the safe direction (it uses
  floor-failure to *disqualify* arms, never floor-success to certify), but the writeup
  must say so explicitly and cite this paper.
* **Searches for A01's specific `layer-order null` for layer-correspondence returned
  nothing** (arXiv queries on `"layer permutation" AND "similarity"` and
  `"permutation null" AND "representational similarity"` returned zero relevant hits).
  So A01's claim 2 in its own novelty-boundary list survives, at the strength of "not
  found", which is weaker than "does not exist".

**Consequence for clause 3:** the similarity-null prior art (Ding et al., Hewitt & Liang)
is real and A01 already concedes it. But A01's surviving findings live on the MC-accuracy
construct, not on that leg. A01 therefore does **not** "degenerate into a case collection
of existing similarity-null methods". Clause 3 does not fire.

## 3. The strongest novelty claim A01 now has, and it came from free evidence

Not found in any candidate above: **the input-blind null that A01 recommends has its own
undeclared convention degree of freedom, large enough to reverse verdicts.**

On the same 14,042 MMLU items, the "longest-option" null takes five different values
depending only on how one breaks ties in option token length (34.22% of items have ≥2
maximal-length options; verified this session, `winner_set_size_hist`
`{1: 9237, 2: 2174, 3: 754, 4: 1877}`):

| convention | null | verdicts across the six OLMo-2-7B dtype arms (bf16 content_norm) |
|---|---:|---|
| `split` (canonical) | 0.2845 | 6/6 above |
| `first` | 0.2811 | 6/6 above |
| `last` | 0.2822 | 6/6 above |
| **`credit`** (optimistic) | **0.4537** | **1/6 above, 5/6 BELOW** |
| **`wrong`** (pessimistic) | **0.1961** | 6/6 above |

Source: `evidence/gate3_content_null_conventions.json` / `.csv`, regenerated by
`code/a01_gate3_content_conventions.py` from the six per-item dtype record sets; all
accuracies and all five null values reproduce the archived
`evidence/gate3_dtype_runs/*_dtype_summary.json` to <1e-12.

So a reader who picks the optimistic reading of the *same English sentence* ("the null is
the longest option") converts five of six arms from "above the null" to "significantly
below the null". **A01's own recommended instrument has the exact defect A01 accuses the
chance line of having.** No candidate paper above states this, for this null or any other.
It is A01's protocol applied reflexively, and it is the natural companion to the C5
self-retraction.

## 4. What A01 must change in response to this check (citation obligations)

1. Cite Balepur et al. (ACL 2024 main) as the origin of "use stronger than chance
   baselines in MCQA"; do not claim it.
2. Cite Zheng et al. (ICLR 2025 Oral) as the origin of "null model" terminology and of
   "a constant predictor can top a benchmark"; do not claim it.
3. Cite Oostermeijer (ICML 2026); **drop** the claim that acc-vs-acc_norm length
   sensitivity is an A01 finding; reframe the OBQA sign flips as replication under damage.
4. Cite OLMES (Findings of NAACL 2025) for the letter/cloze interface split; position
   A01's floor test as a **fix to a defect** in OLMES's size-keyed interface-selection rule.
5. Cite Cho et al. (ICLR 2026) and Bean et al. (NeurIPS 2025 D&B) as the framing/parallel
   literature; do not claim "MC interface validity is unexamined".
6. Cite Feng et al. (Misleading Failures of Partial-input Baselines) and state explicitly
   that clearing a floor is **necessary, not sufficient**.
7. Keep the layer-order-null claim, but downgrade the language from "first" to "we are not
   aware of a prior layer-order null for layer correspondence".

## 5. Method, and its limits

* **arXiv API**: 30 queries across the axes named in the task (majority/best-constant/
  input-blind nulls, construct validity, letter-vs-content and cloze interfaces,
  degenerate/constant predictors, above-chance-as-wrong-reference, partial-input and
  hypothesis-only baselines, length normalisation, chance correction, CKA/permutation
  nulls, control tasks, layer permutation, bf16 ties/precision-in-eval).
* **OpenReview API** (`api2.openreview.net/notes/search`): venue verification for the
  ICML/ICLR/NeurIPS family, checking both `venueid` and the presence of a
  `Camera_Ready_Revision` invitation.
* **ACL Anthology** bibtex + volume indices: venue verification for the ACL family
  (Balepur main-vs-Findings; OLMES Findings).
* **Semantic Scholar: UNAVAILABLE this session** (HTTP 429 rate limit).
  **DBLP: UNAVAILABLE this session** (HTTP 500 on `/search/publ/api`). Both are normally
  cross-checks; their absence is why every venue claim above names the *specific*
  authority actually used, and why three older papers are explicitly marked UNVERIFIED.
* **Not done:** full-text (PDF) reading of any candidate. All overlap/gap judgements above
  are from title + abstract + venue metadata. For Balepur et al. and Oostermeijer — the two
  that carry real citation obligations — a full-text pass before submission is advisable,
  and no A01 claim should be finalised on abstract-level reading alone.
* **Concurrency window:** for a 2026-08-09 check, ≤3 months means ≥2026-05-09. The ICML
  2026 length-bias paper's arXiv v1 is 2026-07-14 (inside the window) but its OpenReview
  `cdate` is 2026-01-24 and `pdate` 2026-04-30 (both outside). Judged **not concurrent**,
  i.e. treated as the stricter case: real prior art A01 must cite. No candidate was
  dismissed on concurrency grounds.

## 6. Verdict restated

**Does the third kill clause fire? NO.**

A01 does not reduce to a case collection of existing similarity-null methods. Its
surviving contributions are on a different construct (MC accuracy under structural
damage), and no verified prior work covers that scope. What the check *does* produce is a
list of six mandatory citations and one sub-claim (§2.2) that must be withdrawn — plus a
new, stronger, and apparently unclaimed contribution (§3) that arrived from re-analysing
evidence already on disk.
