# B10 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU. Adjudication + venue-verification pass only — this file runs nothing
and authorises nothing.**

This closes the blocker `proposal/ready_queue.py` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`).

Two facts about B10's scheduling state shape this document and must be stated first:

1. **B10 is absent from `RELATED_WORK_GAP_AUDIT_20260808.md`.** It was created 2026-08-11, three
   days after the audit. So there is no assigned collision-family list; §3's families are chosen
   here and the choice is defended in §2.3.
2. **`ready_queue.py` reports `! no next_gate field at all` and classifies B10
   `next_gate not operationalised -> writing it is 0 GPU and blocking`.** That is a *true* report of
   `STATUS.json` — it has `kill_gate` but no `next_gate`-family key. It is *not* true that B10 has no
   gate: `PROPOSAL.md` §5 / `STATUS.json.kill_gate` pre-register a four-gate ladder whose Gate 1 is
   explicitly **0 GPU**. Writing a `next_gate` key is a separate 0-GPU task and **this file is not
   it** — it deliberately does not touch scheduling fields.

---

## 1. What B10 claims RIGHT NOW — and it is **benchmark-scoped**, not general

⚠️ **This section is the most important in the file.** B10's history is a sequence of over-broad
claims being cut down, and a Related Work written against the *old* framing would be worse than none.

**Dead, and must never be revived** (`PROPOSAL.md` §1, `STATUS.json.corrections_to_slate` C1–C8):

* ❌ "A matched-lineage AR model beats masked diffusion on diffusion's own turf." Paired McNemar
  **p = 0.635**; margin 5 tasks of 1033; paired bootstrap CI **[−0.0116, +0.0213]** straddles zero;
  and on the gold-feasible subset the **sign flips** (diffusion .9337 vs AR .9324, p = 1.000).
* ❌ "DreamOn ships no matched-lineage AR native-FIM control." **False** — it is Table 1 of the
  camera-ready (Qwen2.5-Coder-7B **92.6** vs DreamCoder+DreamOn **92.1**), and this repo's own
  `TASK_SURFACE_LIT_GAPS.md` had already recorded the 92.6.
* ❌ "DreamOn's advertised length-elasticity kwargs are inert, so the capability was never
  exercised." The kwargs *are* no-ops, but expansion is **token-driven, not flag-gated**, and was
  live on **84.3 %** of tasks; the README never advertises them. Publishing this as a model defect
  would be a **false accusation**.
* ❌ "AR dominates on both cost units." False — under `attended_context_sum`, Dream-FIM is
  **0.88× = cheaper than AR**.
* ❌ Task **#186** ("Reframe dLLM direction: benchmark-scoped claim, not general") is marked
  `completed` and its text records that the *reframed* claim it produced is **dead, not merely
  rescoped** — A05's canvas sweep showed the +26.9 pp margin it rested on was an `initial_masks`
  artefact. **So "benchmark-scoped" must not be read as "the old claim, restricted".** It means: the
  only survivor is a **protocol-sensitivity** statement about *one benchmark surface*.

**What B10 may claim now — the operative, benchmark-scoped statement:**

> **On exactly one task surface — official HumanEval-SingleLineInfilling, n = 1033
> (`loubnabnl/humaneval_infilling`, sha256 `6fffc71e…`), Python-only, greedy T = 0 — three choices
> that are *not the model* each individually decide the reported AR-vs-diffusion comparison, and they
> point in different directions:**
> **(i) the grading axis and its unmeasured feasibility ceiling** (`--which plus` gold refill scores
> only **0.8025**, so ~19.75 % of items are unpassable by construction; restricting to the 829
> feasible items moves `qwen_fim` .7638 → **.9324** and `dreamon_oracle` .7590 → **.9337** — i.e.
> **the ranking inverts** and both land near the published 92.6 / 92.1);
> **(ii) the cost unit** (`tokens_fed` makes AR 8.5–24.4× cheaper; `attended_context_sum` makes
> Dream-FIM **0.88×** = cheaper than AR — and the two units are *identical by construction* for every
> diffusion arm while differing ~10× for AR because of KV caching, so **the unit choice is the winner
> choice**);
> **(iii) the oracle-length handout** (`dreamon_oracle` beats `dreamon_fim` by **+5.7 pp**,
> p = 4.1e-14, so *which diffusion configuration is the comparator* decides whether AR "wins").
> **Separately and most robustly: suffix visibility is worth +0.2314 (AR) and +0.2991 (diffusion)**,
> both p < 1e-56, difference-of-gains +0.0678 with paired CI [+0.0407, +0.0949] — so bidirectional
> context is an **affordance of the FIM task framing, available to AR too**, not a property of the
> model class.

Scope discipline, binding on every sentence below:

* **One benchmark, one split, one language, one decoding setting.** Not RandomSpan (1640), not
  MultiLine (5815), not the k-span surface (n = 415/408 — a *different experiment*, whose
  "diffusion's home turf" claim was already **WITHDRAWN**).
* **Not a general AR-vs-diffusion conclusion.** That thesis is a peer-reviewed **ICLR 2026 poster**
  (§3.2) and B10 must not restate it as a finding.
* **No absolute pass@1 from this surface is a capability measurement** without a decontaminated
  companion — `KSPAN_INFILLING_RESULTS.md` §4.5 measured **26–28 pp** loss from identifier renaming +
  docstring replacement on a set whose gold refill still scores 1.000 (`PROPOSAL.md` Gate 4).

---

## 2. Standing rules and method

### 2.1 Preemption bar

**`memory/prior-work-differentiate-dont-abandon.md`** (user, 2026-08-07): the bar is **完全相同 /
抄袭**, not overlap; ≤ 2–3 months is **concurrent**; the required response to a close work is
differentiation or a follow-up fixing a defect. **A direction dies from its own kill gate, never from
a literature count.**

This rule is load-bearing for B10 in a way it is for no other proposal here, because
`PROPOSAL.md` §4.5 records that **B10's ancestor was killed badly**: dropped on a scan reporting
"DreamOn covers this capability", i.e. **grading a title, not an artefact**. Re-examination found the
*reasoning* wrong and the *answer* right anyway. Both halves belong in any write-up, and this file
does not repeat the mistake: §3 grades artefacts (Table numbers, measured axes), not titles.

### 2.2 Venue verification, by family

| Family | Authority used, first-hand, 2026-08-15 |
|---|---|
| ICLR / NeurIPS / ICML | OpenReview `venueid` + `Camera_Ready_Revision` invitation |
| ACL / EMNLP / NAACL incl. Findings | ACL Anthology page + DBLP `conf/*` record |
| otherwise | DBLP; `arXiv-only` = *no peer-reviewed venue verifiable from this node* |

⚠️ **This family split mattered concretely, twice, in this file.** SAFIM (§3.4) is
`journals/corr/abs-2403-04814` in DBLP's CoRR record *and* `conf/icml/GongWEC24` — it is an
**ICML 2024 Oral**. Weasel-style lag also hit **CaRE** (§3.5), which is CoRR-only in DBLP while
carrying two ICML 2026 workshop records under a *different title* on OpenReview. Grading either by
its CoRR record alone would have mis-stated the field's maturity.

### 2.3 Which collision families, and why these

No audit row exists for B10, so the families are derived from **what the operative claim in §1
actually asserts**, one family per assertion:

| Claim component | Family that could own it | §  |
|---|---|---|
| the model comparison on this surface | diffusion LM code infilling (Dream / DreamOn / LLaDA lineage) | 3.1 |
| "AR is competitive at any-order generation" | AR-vs-NAR / any-order generation | 3.2 |
| the FIM task framing itself, suffix visibility | text infilling / fill-in-the-middle, span corruption objectives | 3.3 |
| the grading axis + feasibility ceiling | code-infilling benchmarks and their post-processing/grading | 3.4 |
| the cost unit | dLLM compute accounting / compute-matched evaluation protocols | 3.5 |
| the oracle-length handout | adaptive / variable infilling length in DLMs | 3.6 |

---

## 3. Named closest collisions (venues verified this session)

### 3.1 DreamOn — **maximal overlap; same benchmark, same split, and it already has the AR control**

* **Cite**: Wu, Zirui; Zheng, Lin; Xie, Zhihui; Ye, Jiacheng; Gao, Jiahui; Gong, Shansan; Feng,
  Yansong; Li, Zhenguo; Bi, Wei; Zhou, Guorui; Kong, Lingpeng. *DreamOn: Diffusion Language Models
  For Code Infilling Beyond Fixed-size Canvas*, `arXiv:2602.01326`.
* **Venue, re-verified independently this session**: **ICLR 2026 Poster** — OpenReview
  `venueid = ICLR.cc/2026/Conference`, `venue = "ICLR 2026 Poster"`, forum **`EQTPmqukiU`**,
  `Camera_Ready_Revision` invitation **present**. (`SOURCES.md` §7 recorded the same; this is a
  fresh, independent confirmation, not a carry-forward.)
* **What it does**: expansion/deletion tokens let a masked diffusion LM escape the fixed canvas for
  code infilling. **Table 1** reports Qwen2.5-Coder-7B **92.6** single-line / 58.7 multi-line vs
  DreamCoder-7B + DreamOn **92.1 / 63.8**; **Table 2** has an explicit **oracle-length** column;
  §4.1 sweeps initial mask length; §4.2 baselines include Deepseek-Coder-6.7B, Seed-Coder-8B,
  Qwen2.5-Coder-7B, LLaDA-8B, Dream-7B, DiffuCoder-7B.
* **Overlap: maximal.** Same benchmark, same split, and it contains the matched AR control B10's
  ancestor claimed was missing. The authors' own numbers put **AR ahead on single-line**; they claim
  to surpass AR only on **multi-line**, and describe single-line as "on par".
* **Residual gap, stated as what is measurably absent from the paper**: (a) **no compute / token /
  NFE accounting against the AR baseline** — so nothing in it can be contradicted by, or can
  preempt, a two-unit cost analysis; (b) **no feasibility-ceiling / gold-refill control** on the axis
  it grades; (c) its oracle column is a *reference point*, not a **controlled contrast against a
  non-oracle AR comparator**; (d) Python-only, single-line + multi-line, no RandomSpan.
* **Verdict: not preemption of the surviving claim, total preemption of the dead one.** It is
  **peer-reviewed and 6 months old**, so the concurrency clause does **not** apply. Any B10 write-up
  is a **follow-up audit of a published evaluation protocol** and must say so in the abstract.

### 3.2 A3 — the general thesis is already peer-reviewed

* **Cite**: *Autoregressive Models Rival Diffusion Models at ANY-ORDER Generation*,
  `arXiv:2601.13228`.
* **Venue, re-verified independently this session**: **ICLR 2026 Poster** — OpenReview
  `venueid = ICLR.cc/2026/Conference`, forum **`vtDUomlazQ`**, `Camera_Ready_Revision` present.
* **Overlap: thesis-level and decisive.** The conclusion B10's ancestor wanted — AR matches/beats
  diffusion at any-order generation including infilling — is a peer-reviewed poster.
* **Residual gap**: it proposes a **new AR training scheme**; it does not audit **evaluation-protocol
  sensitivity on a fixed public benchmark**. Those are different objects: A3 changes the model, B10
  changes nothing and varies the *measurement*.
* **Verdict: forecloses the ranking claim permanently; leaves the protocol claim open.** This is the
  clearest illustration of why B10 must be a measurement paper or nothing.

### 3.3 The FIM framing itself — Bavarian et al., and the span-corruption lineage

* **Bavarian et al., *Efficient Training of Language Models to Fill in the Middle*,
  `arXiv:2207.14255`** — **arXiv-only** (DBLP total = 1, `journals/corr/abs-2207-14255`, CoRR 2022;
  no conference record found from this node). Defines FIM **and this benchmark**, and establishes
  that AR models infill via sentinels. **B10's `qwen_fim` arm is an application of this, not a
  discovery.** Consequence: B10 may not present "AR can use the suffix" as a finding — only the
  **quantified symmetry** of the gains (+0.2314 vs +0.2991) is B10's.
  ⚠️ Its arXiv-only status is worth flagging: the paper that *defines* the surface has no verifiable
  peer-reviewed venue, which is itself relevant to a protocol-audit narrative.
* **Span-corruption / structure-aware objectives**: *Structure-Aware Fill-in-the-Middle Pretraining
  for Code*, `arXiv:2506.00204` — **arXiv-only** (OpenReview returns only a DBLP-mirrored `CoRR 2025`
  note). *Memorization Dynamics of Fill-in-the-Middle Pretraining*, `arXiv:2605.22981` —
  **arXiv-only**; directly relevant to B10's **Gate 4 memorisation threat** and should be cited there
  rather than as a competitor.
* **Verdict**: background, not collision. But it fixes the *ceiling* of B10's mechanism claim: the
  suffix-gain result is a **measurement inside an established framing**.

### 3.4 Code-infilling benchmarks and their grading — **the family that most nearly owns claim (i)**

| Cite | Venue, verified | What it does | Why it does not own B10's ceiling claim |
|---|---|---|---|
| Gong, Wang, Elhoushi, Cheung. *Evaluation of LLMs on Syntax-Aware Code Fill-in-the-Middle Tasks* (**SAFIM**), `arXiv:2403.04814` | **ICML 2024 Oral** — OpenReview `venueid = ICML.cc/2024/Conference`, forum `jKYyFbH8ap`, `Camera_Ready_Revision` present; DBLP `conf/icml/GongWEC24`. **Note: DBLP also lists it as `journals/corr/abs-2403-04814`** | 17,720 examples, multi-language, post-April-2022 to limit contamination; explicitly ships **"various prompt designs and novel syntax-aware post-processing techniques, facilitating accurate and fair comparisons"** | **This is the strongest single threat to claim (i)** and it must be cited head-on. But its instrument is **syntax-aware post-processing of model output** to make comparisons fair; it does **not** measure the *benchmark's own gold-refill feasibility* on the axis it grades. Different quantity: SAFIM fixes the *output side*, B10 measures the *reference side*. |
| Ho, Huang, Boudin, Aizawa. *From Output to Evaluation: Does Raw Instruction-Tuned Code LLMs Output Suffice for Fill-in-the-Middle Code Generation?*, `arXiv:2505.18789` | **arXiv-only** (DBLP CoRR 2025; OpenReview shows `ACL ARR 2025 July Submission`, i.e. **under review, not accepted**) | argues post-processing is *crucial* for automatic FIM evaluation because of extraneous code; finds SFT reduces the need, **but that post-processing remains necessary when the middle is a random span**; evaluates on HumanEval-Infilling and SAFIM | Closest in *spirit* to "the evaluation pipeline, not the model, is doing work". Still output-side truncation policy, on **Qwen2.5-Coder** — the same AR family as B10's arm. No gold-refill ceiling, no cost units, no oracle-length contrast. **Must be cited as a preprint under ARR review.** |
| Liu, Xia, Wang, Zhang. **EvalPlus** (*Is Your Code Generated by ChatGPT Really Correct?*), `arXiv:2305.01210` | **NeurIPS 2023 Poster** — OpenReview `venueid = NeurIPS.cc/2023/Conference`, forum `1qvx610Cu7`, `Camera_Ready_Revision` present | 80× test augmentation of HumanEval → HumanEval+; the *premise* that base tests are inadequate | **B10's grader IS EvalPlus** (`evalplus.eval.untrusted_check`), and the plus axis is EvalPlus's contribution. B10's ceiling finding is therefore **a finding about applying EvalPlus's stricter tests to an infilling split whose gold middles were never validated against them** — a *use* defect, not a defect in EvalPlus. This framing is mandatory and protective. |
| Han, *A Systematic Evaluation of Trajectory Data Curation…* — **not relevant to B10**; listed only to prevent confusion with the identically-named B09 entry | — | — | — |

**Net on claim (i)**: the *idea* that FIM evaluation is post-processing-sensitive is **published and
peer-reviewed** (SAFIM, ICML 2024 Oral) and independently argued in a 2025 preprint. What is **not**
in the located set is **measuring the benchmark's own gold-refill pass rate on the graded axis and
showing it re-orders arms.** B10's 0.8025 (traced to 23 buggy parent tasks) is that measurement.
**This is a narrower claim than "protocol sensitivity" and B10 must state it at that width.**

### 3.5 Compute accounting and compute-matched protocols — **claim (ii) is contested territory**

| Cite | Venue, verified | Relation |
|---|---|---|
| *CaRE: Compute-aware Remasking Evaluation Protocol for Masked Diffusion Language Models*, `arXiv:2607.24763` | **arXiv-only for this title** (DBLP `journals/corr/abs-2607-24763`, CoRR 2026). OpenReview has **two ICML 2026 workshop records under the title *"Re-evaluating Confidence Remasking in Masked Diffusion Language Models"*** — `venueid = ICML.cc/2026/Workshop/SPIGM` (forum `HIMiqnTqLD`) and `ICML.cc/2026/Workshop/AdaptFM` (forum `Bew2D82sWR`), both with `Camera_Ready_Revision`. ⚠️ **Whether these are the same work under a changed title was NOT established** — treat the arXiv entry as a preprint and the workshop records as *possibly* the same paper | **The single most dangerous neighbour for claim (ii).** Its thesis is B10's, one level up: seven remasking papers evaluate under incompatible settings, and **"compute-matched comparisons reverse several published strategy rankings"**, i.e. *cost-unit choice flips rankings* — already stated, at scale (7 strategies × LLaDA-8B-Base + Dream-7B-Base × 4 stochasticity levels × 3 step budgets), with a 12-model leaderboard. **Differentiation must be exact**: CaRE standardises **NFE within the diffusion family** to compare **remasking strategies**; B10's contrast is **across the AR/diffusion boundary**, where the two units are *identical by construction* on one side and differ ~10× on the other **because of KV caching**. That asymmetry is a property of the paradigm boundary and cannot arise inside CaRE's all-diffusion design. |
| *Diffusion Language Models Are Natively Length-Aware*, `arXiv:2603.06123` | **arXiv-only** | reports **FLOPs** for zero-shot context cropping across GSM8K/HumanEval/IfEval/LongFormQA. Confirms the cost-accounting niche is being actively occupied, **within** the diffusion family |
| *Autoregressive vs. Masked Diffusion Language Models: A Controlled Comparison*, `arXiv:2603.22075` | **arXiv-only** (DBLP call failed; see §6.3 — recorded as **unverified**, not as absent) | identical data/compute/hardware, paradigm as sole variable; ~50K tok/s both, MDLM +4.7 % wall-clock; diversity-fluency trade-off. **Directly occupies "controlled AR-vs-MDLM comparison"** — but at **50M tokens on TinyStories**, no code, no infilling, no per-unit cost inversion. Cite as the small-scale controlled precedent |
| *Speculative Refinement: A Hybrid AR-Diffusion Decoding Strategy and Its Behavior Across Benchmarks*, `arXiv:2606.27474` | **arXiv-only** (DBLP CoRR 2026) | ⚠️ **Reports three findings that are each adjacent to a B10 claim**: code benchmarks conflate structural discovery with logical correctness (a syntactic scaffold lifts accuracy from ~0 to >20 % **without changing the model**); **"benchmark saturation ceilings invisible to single-model evaluation"**; and **log-likelihood vs generative evaluation produce different rankings for the same model pair**. The first is the closest published analogue of A05's Scaffold result; the second is ceiling-adjacent (but *saturation* ceiling, not *gold-infeasibility* ceiling); the third is a ranking-flip claim on a different axis. **Must be cited; it narrows B10's ceiling claim to specifically the gold-refill construct.** |
| *Unsolvability Ceiling in Multi-LLM Routing: An Empirical Study of Evaluation Artifacts*, `arXiv:2605.07395` | **arXiv-only** (DBLP `journals/corr/abs-2605-07395`, CoRR 2026) | 206k query-model pairs across 6 benchmarks incl. HumanEval/MBPP; attributes a substantial part of reported "unsolvability" to **evaluation artifacts**: judge bias, **truncation under fixed generation budgets**, **output-format mismatches**. Nearest published statement of "a reported ceiling is partly an artefact" — but the artefacts are lumped into three categories for a **routing-headroom** purpose, with no per-operation ablation and no gold-refill measurement. Also shared with B11 (§3 there); the two proposals must not both claim it |

**Net on claim (ii)**: "the cost unit changes the winner" is **already stated within the diffusion
family** by CaRE. What survives for B10 is much narrower and mechanically specific: **at the AR /
diffusion boundary, `tokens_fed` and `attended_context_sum` are identical by construction for every
diffusion arm and differ ~10× for AR, so no unit is neutral and the choice is not a reporting
convention but a result.** State it that way or not at all.

### 3.6 Adaptive / variable infilling length — claim (iii)'s neighbourhood

| Cite | Venue, verified | Relation |
|---|---|---|
| *Diffusion LMs Can Approximate Optimal Infilling Lengths Implicitly* (CAL), `arXiv:2602.00476` | **arXiv-only** | training-free calibrated length search; **+47.7 % pass@1 over fixed-length** in code infilling. Competes with **DreamOn's** premise, not with B10 — but it means "the length handout matters a lot" is published, so B10's +5.7 pp is a *quantification within a known phenomenon*, not a discovery |
| *From Interface to Inference: Eliciting Any-Order Inference from Any-Order Models*, `arXiv:2607.26504` | **arXiv-only** | insertion-based masked diffusion (on FlexMDM) + latent-space masked diffusion; frames the gap as **positional uncertainty**: fixed-canvas models "may know *what* should appear without knowing *where*". 7B FlexMDM for Python |
| *Planning-Aware Code Infilling via Horizon-Length Prediction* (HLP), `arXiv:2410.03103` | **arXiv-only, and specifically NOT accepted**: OpenReview shows `venueid = ICLR.cc/2025/Conference/Rejected_Submission` (forum `tDANkt6X3D`) plus a **NeurIPS 2024 Sys2-Reasoning workshop** poster (forum `L0agONQi9X`); DBLP CoRR 2024 | trains the model to predict remaining middle tokens, "**without relying on dataset-specific post-processing**". Relevant twice: it is prior art for "the length is the problem", **and** its explicit framing of post-processing dependence as a defect to be removed is an argument B10 should engage. ⚠️ Do **not** cite as an ICLR paper |
| *Improving Variable-Length Generation via Length Regularization*, `arXiv:2602.07546`; *Any-Order Flexible Length Masked Diffusion* (FlexMDM), `arXiv:2509.01025` | **arXiv-only** (carried from `SOURCES.md`; **not re-verified this session**) | variable-length masked diffusion lineage |

**Net on claim (iii)**: the *importance* of infilling length is thoroughly established (CAL, HLP,
DreamOn's own Table 2). B10's residual is only the **controlled contrast with a non-oracle AR
comparator held fixed**, which none of the above runs. That is a small, real gap.

### 3.7 Concurrent (≤ 3 months; cannot preempt)

* *Distractor-Aware Truncation…*, `arXiv:2608.03297` (2026-08-04, **11 days before this pass**;
  DBLP total = **0**, arXiv-only). Not a B10 collision — it truncates **input context** on BABILong /
  GraphWalks — but it is B11's neighbour and is named here so the two proposals do not double-claim
  the word "truncation".
* *LLaDA MoE v2*, `arXiv:2608.03457` (2026-08-04). Scale/architecture, not protocol. Listed to record
  that the diffusion-LM field is moving weekly, which is an argument for B10's cost estimate being
  perishable, not for its urgency.

---

## 4. MUST-NOT-CLAIM (binding; extends `PROPOSAL.md` §4.4 and `STATUS.json.must_not_claim`)

All eight prohibitions in `PROPOSAL.md` §4.4 stand. This pass adds:

9. ❌ **"FIM/code-infilling evaluation is post-processing sensitive"** as a novel observation. →
   **SAFIM, ICML 2024 Oral** ships syntax-aware post-processing precisely to fix it, and
   `arXiv:2505.18789` argues it directly. B10's claim must be the narrower gold-refill measurement.
10. ❌ **"The cost unit changes the winner"** as a novel observation. → **CaRE** states that
    compute-matched comparison **reverses published rankings**. B10's residual is the *AR-vs-diffusion
    boundary asymmetry*, mechanically attributed to KV caching.
11. ❌ **"Benchmark ceilings are invisible in single-model evaluation"** as B10's. →
    `arXiv:2606.27474` says it; and it also owns **"a syntactic scaffold lifts accuracy without
    changing the model"**, which is adjacent to A05's Scaffold leg.
12. ❌ **"A reported ceiling is partly an evaluation artefact"** as novel. →
    `arXiv:2605.07395` (206k pairs, incl. HumanEval/MBPP).
13. ❌ **Citing SAFIM or EvalPlus as preprints.** ICML 2024 Oral and NeurIPS 2023 respectively.
14. ❌ **Citing HLP (`2410.03103`) as an ICLR paper.** OpenReview records it as
    `ICLR.cc/2025/Conference/Rejected_Submission`; only a NeurIPS 2024 workshop poster exists.
15. ❌ Presenting the **suffix-gain symmetry** as showing the gains are "comparable" — correction C7:
    the difference is +0.0678 with CI [+0.0407, +0.0949], **excluding zero**, and the two arms'
    handouts were **not matched** (`dream_prefix` had an oracle length, `qwen_prefix` did not).
    Until Gate 3 runs, the honest phrasing is *"both large; diffusion's reliably ~6.8 pp larger;
    measured under an unmatched handout."*
16. ❌ Any **generality** claim across benchmarks, languages, splits, or decoding settings. §1.

---

## 5. Safe residual claim — one falsifiable sentence

> **On HumanEval-SingleLineInfilling (n = 1033, Python, greedy), the reported ordering of a
> matched-architecture AR arm and the strongest masked-diffusion arm is not determined by the models:
> re-scoring the *same stored generations* on the benchmark's official `base` axis rather than the
> `plus` axis, and/or restricting to the 829 items whose own gold middle passes, changes the sign of
> the difference while leaving both arms within ~1 pp of each other — so at this n the surface has no
> resolvable AR-vs-diffusion answer, and the reported answer is a function of the grading axis, the
> cost unit and the oracle handout.**

**Why this is falsifiable and cheap**: it is exactly **Gate 1**, and Gate 1 costs **0 GPU** —
solutions are already on disk; `score_infilling.py --which base` decides it. It is falsified if the
base axis yields a **significant, ceiling-robust, directionally stable** AR advantage
(α = 0.05, |Δ| ≥ 0.02).

**And the pre-registered expectation is that this claim's own second half kills the direction.**
`PROPOSAL.md` §5 states plainly: the likely outcome is KILL, after which B10 "must be **rewritten as
a protocol note or archived** — it must NOT be re-framed to hunt a different ranking." §3 of this
file *strengthens* that: with SAFIM (ICML 2024 Oral) owning post-processing sensitivity, CaRE owning
compute-matched ranking reversal, and `2606.27474` owning invisible ceilings, **the protocol-note
niche is itself substantially occupied.** The honest read is that B10's realistic ceiling is a
**short, precise, benchmark-scoped measurement note or an appendix** — and that is worth 0 GPU-h to
find out, which is the whole point of Gate 1 being free.

---

## 6. Verdict

```
verdict: hold_in_backlog
novelty_gate: CLEARED for GATE 1 ONLY (0 GPU). Not cleared for Gates 2-4.
gpu: NONE authorised by this file (unchanged from PROPOSAL.md and STATUS.json).
promotion: NOT eligible. Requires Gate 1 to PROCEED (contrary to its own stated
           expectation), Gate 2 to remove the Instruct-vs-base lineage confound, and a
           novelty re-check establishing the survivor is not DreamOn's Table 1 re-plotted.
next_gate field: still ABSENT from STATUS.json. Writing it is a separate 0-GPU task
           and was deliberately not done here.
```

* **No candidate is 完全相同 / 抄袭 of the *surviving* claim.** DreamOn preempts the dead ranking
  claim totally; A3 preempts the dead general thesis; SAFIM/CaRE/2606.27474 each own one *level up*
  from one of B10's three protocol observations. None measures a benchmark's gold-refill feasibility
  on the axis it grades and shows it re-orders arms. `already_dead_should_archive` is therefore **not**
  warranted **on literature grounds** — but note §5: it may well be warranted on **Gate 1** grounds,
  and that is the right way for it to happen.
* **The narrowing here is the third successive one** (SLATE → `PROPOSAL.md` §1 → this file). That is
  a healthy trajectory, not a failure, and each step was forced by measurement or by a verified
  venue.

---

## 7. Honest gaps in this adjudication

1. **CaRE's identity is unresolved.** `arXiv:2607.24763` is CoRR-only in DBLP, while OpenReview has
   two ICML 2026 **workshop** records for a differently-titled paper by apparently the same design.
   **I did not establish they are the same work** (that needs an author/abstract diff, and the arXiv
   API was timing out by then). Since CaRE is the sharpest threat to claim (ii), this is the most
   consequential unresolved item in the file. Resolve before writing.
2. **`arXiv:2603.22075` (AR vs MDLM controlled comparison) venue is UNVERIFIED.** The DBLP call
   returned `PARSE FAIL` (empty body) and was not retried to success. Recorded as unverified, **not**
   as arXiv-only.
3. **DBLP and the arXiv API were both intermittent** — `curl: (28)` timeouts and
   `curl: (56) Failure when receiving data from the peer` throughout; some records needed 3 retries;
   `2509.16720` and `2504.11972` needed a background retry loop. Every row above comes from a call
   that **returned**; nothing is inferred from a failure. **Semantic Scholar was not queried at all**
   (repo rule: never a venue authority).
4. **`arXiv:2602.07546` and `arXiv:2509.01025` were not re-verified this session** — carried from
   `SOURCES.md` §7. Labelled accordingly in §3.6.
5. **No full text was read this session.** DreamOn's Table 1 / Table 2 / §4.1 / §4.2 contents are
   carried from `SOURCES.md` §7 and `NUMBER_AUDIT.md`, which *did* read the 11-page PDF via
   `pdftotext -layout`. Everything else is abstract + venue metadata. For the three most dangerous
   neighbours (**CaRE**, **SAFIM**, **`2606.27474`**) a **full-text differential read is mandatory
   before any write-up**, because §3's differentiations turn on design details an abstract can hide
   (does CaRE ever cross the AR boundary? does SAFIM anywhere validate its own gold middles?).
6. **Zero cross-disk verification.** All of B10's primary evidence — the six scored arms, the harness,
   the model weights — is **zwfy6-only**, and `/apdcephfs_zwfy6` is not mounted on LOCAL. Every path
   in `SOURCES.md` is **recorded-only** from here and must be `ls`-confirmed on `.73`/`.82`/`.104`
   before dispatch (`memory/two-disk-rule-applies-to-main-too.md`). The one wzc1-only file — the
   gold-ceiling JSON that carries the 0.8025 — is on the *other* disk from the arms it must be joined
   to, and `SOURCES.md` §5 records the verified `scp -O` transfer.
7. **No `.bib` entries emitted** (`memory/tcodex-exec-no-dash-c-flag.md`): with CaRE's identity open
   and `2603.22075` unverified, generating entries now would seed the bibliography with two rows that
   need to change.
8. **The `next_gate` scheduler defect is reported, not fixed.** `ready_queue.py` will keep printing
   `! no next_gate field at all` after this file lands, and it is **right to**. Fixing it means
   appending a `next_gate` key to `STATUS.json` (append-only, per `LIFECYCLE_SCHEMA.md`) naming
   Gate 1 — a distinct 0-GPU task, out of this file's scope.
