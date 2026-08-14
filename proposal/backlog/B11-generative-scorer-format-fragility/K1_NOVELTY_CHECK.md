# B11 — K1 novelty gate

**Verdict: `NEEDS_NARROWING`**
**Date: 2026-08-14 · GPU spent: 0 (literature + CPU-only re-verification of the scorer)**
**Raw search artefacts: `k1_raw/` in this directory (arXiv Atom XML, ACL Anthology HTML, DBLP JSON, OpenReview JSON, HELMET camera-ready HTML)**

K1's kill condition, verbatim from `STATUS.json`:

> `kill_if`: "the literature already covers 'preprocessing truncation changes model ranking'"

Read literally at that level of generality, **the kill condition fires**: multiple published,
venue-verified papers establish that the answer-extraction / answer-selection stage changes model
rankings. So B11 cannot proceed on the claim as currently written in `STATUS.json`.

Read at the level of the *specific construct B11 actually measures* — a **generative** long-context
scorer's own source code, ablated one operation at a time over **byte-fixed generations**, breaking a
ladder over **arms of one model** that differ in output-format habit — no located work does this.
Hence `NEEDS_NARROWING` rather than `FULLY_PREEMPTED`.

The narrowed claim is in §4. It is also **weaker** than the current claim, for reasons that have
nothing to do with the literature (§5): the surviving evidence is 2/6 cells, and the ablated
operation turns out not to be a one-directional bug.

---

## 1. What the prior art actually establishes

Venue verification followed the two-family rule. **OpenReview `venueid` + `Camera_Ready_Revision`**
for ICLR/NeurIPS/ICML; **aclanthology + DBLP** for the ACL family including Findings and EMNLP.
S2 was not used as a venue authority (it 429'd, and it is known to lag on recent conference papers).

| # | Work | Venue (how verified) | What it establishes | Why it does not fully preempt B11 |
|---|---|---|---|---|
| P1 | Alzahrani et al., *When Benchmarks are Targets: Revealing the Sensitivity of LLM Leaderboards*, arXiv:2402.01781 | **ACL 2024 Long**, `2024.acl-long.744`, pp. 13787–13805, DOI `10.18653/v1/2024.acl-long.744` (**aclanthology metadata block + DBLP `ACL 2024`**) | Perturbing MCQ benchmarks — **choice order** and **the method of answer selection** — moves leaderboard rank by up to 8 positions. Recommends a hybrid answer-selection scoring method. Ships a fork of `lm-evaluation-harness`. | The perturbed object is **the benchmark and the answer-selection method** (likelihood vs. generation vs. hybrid) across **different models** on **MCQ**. Not a text-preprocessing operation inside a generative scorer, and generations are not held fixed across conditions. |
| P2 | Molfese et al., *Right Answer, Wrong Score: Uncovering the Inconsistencies of LLM Evaluation in MCQA*, arXiv:2503.14996v2 | **Findings of ACL 2025**, `2025.findings-acl.950`, pp. 18477–18494, DOI `10.18653/v1/2025.findings-acl.950` (**aclanthology metadata block + DBLP `ACL 2025`**) | Existing MCQA answer-extraction methods misalign with human judgment; traditional extraction **underestimates** capability; LLM-based extractors have systematic errors; a real trade-off exists between prompt-side format constraints and free-form reasoning. | Explicitly and only **MCQA**. Full-text check of the arXiv v2 HTML: **0 occurrences** of `generative`, `long-context`, `ablation`, `BABILong`, `first sentence`, `source code`, `line of code`; `ranking` appears once, and only citing P1. It diagnoses extractor–human misalignment, not a scorer-code ablation that breaks a ranking. |
| P3 | Yu et al., *xFinder: LLMs as Automated Evaluators for Reliable Evaluation*, arXiv:2405.11874 | **ICLR 2025 Poster**, `venueid = ICLR.cc/2025/Conference`, forum `7UqQJUKaLM`, invitations include `Submission5699/-/Camera_Ready_Revision` (**OpenReview**) | RegEx answer extraction in mainstream harnesses is only 74.38% accurate; replacing it with a trained extractor raises judgment accuracy to 97.61%. Names "prompt format overfitting" as a cheating channel. | Frames RegEx extraction as an **accuracy** problem to be **replaced by a better module**. Does not ablate one operation of an existing scorer, and does not show a *ranking* being destroyed by it. Closest in spirit to B11's "the metric's code is the defect" stance, and the strongest citation obligation. |
| P4 | Sanz-Guerrero et al., *Mind the Gap: A Closer Look at Tokenization for MCQA with LLMs*, arXiv:2509.15020 | **EMNLP 2025 Main**, `2025.emnlp-main.988`, pp. 19573–19583 (**aclanthology metadata block + DBLP `EMNLP 2025`**) | How the space after `"Answer:"` is tokenized shifts accuracy by up to 11% and **reshuffles model rankings**. | The knob is **tokenization of the prompt suffix** feeding next-token-probability extraction. Prompt/tokenizer side, MCQ, probability-based. Not scorer text preprocessing over free generations. |
| P5 | Su, Zhang, Ullrich, Bottou, Ibrahim, *A Single Character can Make or Break Your LLM Evals*, arXiv:2510.05152 | **CoRR 2025 only** (DBLP: `CoRR 2025`, `Informal and Other Publications`). No conference venue located. | Choice of in-context-example **delimiter** swings MMLU by ±23% and "one can manipulate model rankings to put any model in the lead by only modifying the single character separating examples." | Strongest published statement of "a one-character choice controls ranking", but the character is in the **prompt**, not the scorer. `SOURCES.md` already flags exactly this prompt-side/scorer-side distinction as the whole novelty question — this paper confirms the prompt side is taken. Preprint, not a published venue. |
| P6 | Kim & Kim (et al.), *Finding Answers in Thought Matters: Revisiting Evaluation on LLMs with Reasoning*, arXiv:2510.14773 | **CoRR 2025** (DBLP). No conference venue located. | Reasoning-model scores and final-answer distributions are "highly sensitive to the answer extraction algorithm employed"; proposes extraction-rule-agnostic Answer Regeneration. | Sensitivity of **scores**, on math/open-ended QA; the fix is another inference pass. No ranking-destruction demonstration, no long-context, no code-line localisation. |
| P7 | *Time to Revisit Exact Match*, arXiv:2509.16720 | **arXiv only** (DBLP total = 0 for this title). | Replacing EM with sMAPE/MASE on temporal QA "**reshuffles model rankings** compared to EM". | Changes the **metric family** (string EM → numeric error) on temporal QA. A metric-choice argument, not a preprocessing-operation ablation. |
| P8 | Rangapur et al. (Sapienza/other), *Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge*, arXiv:2504.11972v3 | **arXiv only** (DBLP total = 0). | EM correlates with human judgment at only 0.22, F1 at 0.40, LLM-judge up to 0.85, on 4 extractive-QA datasets. | The canonical "judge beats string match" result. Establishes EM is a *bad* metric; does not show a specific preprocessing line reordering arms. |
| P9 | Yen et al., *HELMET: How to Evaluate Long-context Models Effectively and Thoroughly*, arXiv:2410.02694v3 | **ICLR 2025 Poster**, `venueid = ICLR.cc/2025/Conference`, forum `293V3bJbmE`, invitations include `Submission12024/-/Camera_Ready_Revision` (**OpenReview**). Note: DBLP still returns only `CoRR 2024` for this title — the two-family rule mattered here. | The single most dangerous neighbour: names "**Unreliable metrics**" as one of four defects of long-context benchmarks, and explicitly says zero-shot prompting "leads to **inconsistent output formats**", e.g. "the model may output a long answer in RAG when a short answer is required". | Read the camera-ready §2.2/§2.3 in full (`k1_raw/helmet.html`). Its metric fix is **replacing** n-gram/ROUGE with a **GPU-4o reference-based judge**, validated against humans (κ=0.91/0.76). Its format fix is **prompt-side**: add two-shot demonstrations. It uses substring EM (SubEM) for RAG **without auditing SubEM's own preprocessing**. It never ablates a scorer operation, never holds generations fixed, and never attributes a ranking change to an arm-wise output-format habit. It argues *"pick a better metric"*; B11 argues *"the metric you already use has an identifiable line that inverts your ladder"*. Disjoint, and B11 must cite it as the strongest adjacent framing. |
| P10 | Arjmandi, *Distractor-Aware Truncation: Disentangling Context-Length Effects from Signal Loss in Long-Context LLM Benchmarks*, arXiv:2608.03297 | **arXiv, 2026-08-04 (10 days before this gate)** | On **BABILong** + GraphWalks: naive middle-drop **truncation** collapses scores; distractor-aware truncation preserves them; "the naive protocol is not a measurement of context-window effects." | Same benchmark, same word "truncation", **opposite side of the pipeline**: it truncates the **input context**; B11's operation truncates the **model output inside the scorer**. Also **concurrent** (10 days), which per project rule does not constitute preemption. Must be cited to prevent a reviewer conflating the two truncations. |
| P11 | Garg & Sagtani, *Unsolvability Ceiling in Multi-LLM Routing*, arXiv:2605.07395 | **arXiv only** | Attributes reported "unsolvability" partly to evaluation artifacts including **"output format mismatches"** and truncation under generation budgets, across 206k query-model pairs. | Routing-headroom estimation. Format mismatch appears as one of three lumped artifact categories; no per-operation scorer ablation, no ranking-destruction claim. |

### Searches that returned nothing (evidence of the gap, not of absent effort)

All via `export.arxiv.org/api/query`, relevance- and recency-sorted, artefacts in `k1_raw/`:

`abs:"generative" AND abs:"scorer" AND abs:"output format"` → **0**;
`abs:"string matching" AND abs:"long context" AND abs:"evaluation"` → **0**;
`abs:"scorer" AND abs:"benchmark" AND abs:"artifact" AND abs:"conclusion"` → **0**;
`abs:"lenient" AND abs:"strict" AND abs:"match" AND abs:"benchmark"` → **0**;
`abs:"LongBench" AND abs:"metric" AND abs:"reliab"` → **0**;
`abs:"RULER" AND abs:"recall" AND abs:"string match"` → **0**;
`abs:"first sentence" AND abs:"evaluation" AND abs:"generation"` → **0** relevant.

Upstream bug-report channel checked: `booydar/babilong` has **18 issues, none** about `metrics.py`,
`preprocess_output`, or first-period truncation (the one reproduction-mismatch issue, #16, is about
Gemma chat templates and attention backends, and has **zero comments**). GitHub issue search for
`babilong metrics truncation` → **0 hits**. So the defect is **not** a known upstream issue, and K3's
"downgrade to an upstream bug report" exit is still open rather than already taken by someone else.

---

## 2. The line the prior art does not cross

Every located work varies one of:

- the **prompt** (delimiter P5, tokenization P4, demonstrations P9, format constraints P2),
- the **extraction module**, swapping it wholesale (RegEx→model P3, extra inference pass P6),
- the **metric family** (EM→sMAPE P7, ROUGE→judge P9, EM→judge P8),
- the **benchmark** (choice order P1, context truncation P10).

None varies **one operation inside the existing scorer while holding the generations byte-fixed**.
That is the only design under which the causal statement "*this operation*, and not the prompt, not
the model, not the metric family, moves the ordering" is identified. B11's
`analyze_a02_truncation_ablation.py` is that design: same 100 items/cell, same stored generations,
one operation removed, uniqueness requirement retained so multiple-choice lists still score 0 and no
chance inflation is introduced.

Second uncrossed line: the ordering B11 destroys is over **arms of a single model** that differ by an
architectural depth knob, **not over model identities**. All the ranking literature above is about
leaderboards, i.e. between-model. A between-arm ladder is what a methods paper actually reports, and
it is *more* fragile here because the arms' output-format habits differ systematically (list-format
rate A4 62–75% vs A5 25–42% on the four inverting cells, vs 2–6% on the two that do not invert).

---

## 3. New evidence found during K1 (0 GPU, CPU re-verification of the scorer)

While reading `third_party/babilong-pkg/babilong/metrics.py` to check what the prior art would have
had to find, I verified two things by direct execution. Both are new relative to `STATUS.json`.

### 3.1 A second, independent code-level defect: `metrics.py:31` is dead code

```python
def preprocess_output(output):
    output = output.lower()          # line 25  <-- lowercases FIRST
    output = output.split('.')[0]    # line 27  <-- the claim's subject
    output = output.split('<context>')[0]
    output = output.split('<example>')[0]
    output = output.split('Question')[0]   # line 31  <-- UNREACHABLE
    return output
```

`split('Question')` can never fire, because line 25 already lowercased the string to `question`.
Executed check: input `"...kitchen Question: Where is the football? Answer: garden"` →
`preprocess_output` returns `"...kitchen question: where is the football? answer: garden"`, i.e.
`'question' in output` is `True` and the guard did not trigger. Same class of bug applies to
`<CONTEXT>`/`<EXAMPLE>` in any casing other than lowercase.

This strengthens the surviving half of the claim — "the failure localises to specific auditable lines"
— because it is a second, sharper instance: a guard that the benchmark authors clearly intended as
the defence against continuation leakage **does not run at all**. It is independently checkable in
five lines and requires no GPU. Note it was already recorded in this repo's own
`scripts/score_with_stopfix.py` docstring (2026-06-30) as a Llama-3 diagnostic; this K1 pass confirms
it holds in the current canonical package and connects it to B11.

### 3.2 The ablated operation is a **trade-off**, not a one-directional bug

This is a correction to the framing, and it is unfavourable to B11's current wording.
Executed on the canonical scorer:

| output | canonical (with truncation) | notrunc (truncation removed) |
|---|---|---|
| `"Choices: A. In the kitchen B. In the garden. The answer is kitchen."` (list habit) | `False` | — truncation is what kills it |
| `"kitchen. Question: Where is the football? Answer: garden"` (continuation leak) | **`True`** | **`False`** — truncation is what **saves** it |

So first-period truncation *destroys* scores for list-format outputs and *rescues* scores for
continuation-leak outputs. The evidence JSON agrees at the cell level: removing truncation raises
qa1/qa2 accuracies but **lowers** qa5 (A0 61.0→59.0, A3 57.0→52.0 at qa5×32k). The correct
description is therefore *"a scorer operation whose sign depends on the arm's output-format habit"*,
not *"a bug"*. This is a better claim scientifically — it explains why the dissociation is perfectly
cell-aligned — but it forbids the word "fix" and forbids presenting `notrunc` as a corrected metric.

---

## 4. The narrowed claim

Replacing the `claim` field's scope (the original is retained verbatim in `STATUS.json`; this is the
scope B11 may proceed under):

> **Narrowed claim.** In a generative long-context benchmark whose scorer matches free-form output
> against a closed task-label vocabulary (`babilong.metrics.compare_answers`), a **single text-
> preprocessing operation in the scorer's source** — truncation of the model output at the first
> period, `metrics.py:27` — interacts with an **arm-dependent output-format habit** strongly enough
> that the scorer **fails to recover the ordering of a within-model architectural manipulation whose
> true effect is +58 to +84 pp on a retrieval-closed reference task**. The interaction is
> demonstrated by re-scoring **byte-fixed generations** with that one operation removed and nothing
> else changed: on 2 of 6 cells the inverted point estimate is removed, and on qa1×32k the 5-point
> ladder becomes ρ = −1.000 (exact permutation p = 0.0167). The operation is **not** a one-directional
> bug — it *lowers* scores for arms that emit choice-lists (list-format rate 62–75%) and *raises* them
> for arms that emit continuations — and the four inverting cells are **exactly** the four
> high-list-format cells ([[4,0],[0,2]], Fisher exact p = 0.0667, the minimum attainable at 6 cells,
> hence descriptive). The scorer contains a **second, independently verifiable defect**: the guard
> intended to stop continuation leakage, `metrics.py:31 split('Question')`, is **unreachable**
> because `metrics.py:25` lowercases first.

Deltas from the current `claim` string, each forced by something above:

1. **"destroy the RANKING" → "fail to recover the ordering of a within-model manipulation."**
   Between-model ranking destruction is P1/P4/P5's territory. The between-arm ladder is B11's.
2. **"+70 to +84 pp" → "+58 to +84 pp."** `established_measurements.true_effect_on_retrieval_closed_ruler_pp`
   is `[58.0, 54.0, 84.0, 84.0]`, whose minimum is **54**, not 70. The `claim` and `PROPOSAL.md` both
   say "+70", which no value in the evidence supports as a lower bound. Using the true range removes
   an overstatement a reviewer would find in one look. *(Flagged, not silently edited — the original
   field is append-only.)*
3. **Added: "byte-fixed generations, one operation removed."** This is the actual novelty axis and
   the only thing that separates B11 from four published papers. It must be in the claim sentence.
4. **Added: "not a one-directional bug."** Forced by §3.2.
5. **Added: the dead-code guard.** Second instance of "localises to auditable lines", and it is free.
6. **Removed nothing about significance** — the claim already says ranking failure, which is right.

### What B11 must now cite, and what it must never say

Mandatory citations: **P1** (rankings move under answer-selection changes — the general result B11
must not re-announce), **P3** (harness extraction is itself defective — the closest stance),
**P9 HELMET** (long-context "unreliable metrics" + "inconsistent output formats" — the closest
framing; distinguish judge-replacement from operation-ablation), **P4/P5** (the prompt-side version
of "one small choice controls ranking" — distinguish prompt side from scorer side **explicitly, in
the abstract**), **P10** (same benchmark, input-side truncation — distinguish or a reviewer will
conflate them).

Additions to `forbidden_claims`, on top of the five already there:

- **"Novel: preprocessing changes model ranking."** Not novel. P1 ACL 2024 and P4 EMNLP 2025 own it.
  Only the *generative scorer + one-operation ablation on fixed generations + within-model ladder*
  combination is open.
- **"We fix the metric" / presenting `notrunc` as corrected.** §3.2: it is a trade-off; `notrunc`
  lowers qa5.
- **"+70 pp" as the lower bound of the true effect.** The evidence's minimum is 54 pp.
- **"BABILong's scorer is uniquely broken."** Not tested — that is K3, still open.

---

## 5. Why this is `NEEDS_NARROWING` and not `PASS`

Two independent reasons, only one of which is about the literature.

**Literature.** The claim as written in `STATUS.json` asserts a general "scorer preprocessing destroys
ranking". That general statement is covered by P1 (ACL 2024) and P4 (EMNLP 2025) and is loudly stated
by P5. Proceeding without narrowing would walk into K1's kill condition.

**Evidence.** Independent of novelty, the surviving support is thin, and K1 should not launder that.
The inversion is not significant (best exact McNemar p = 0.0703; Holm 0.4219). The one-operation
ablation repairs **2 of 6** cells. The mechanism is **not identified** — retrieval and floor are
collinear at Spearman(recall, A0_acc) = +0.714 over 6 cells. The dissociation's Fisher p = 0.0667 is
the floor of what 6 cells can produce, so it is descriptive. And there is exactly **one model family**.
A paper-scale claim needs K2 (cross-family) to land; if K2 fails, K3's upstream-bug-report exit is
still genuinely available, since §1 shows nobody has filed it.

## 6. Gate status

- **K1: released**, for the **narrowed** claim in §4 only, under the citation and forbidden-claim
  obligations above. Not released for the `STATUS.json` `claim` string as written.
- `gpu_policy` "NO GPU until K1 passes" is therefore satisfied, and **K2 (cross-family replication)
  may now spend GPU**. K2 remains blocking for promotion.
- K2's design gains one requirement from §3.2: the second family must be measured on
  **list-format rate per arm** as a *pre-registered* covariate, because the mechanism is now
  explicitly an interaction with format habit. A second family with no arm-wise format asymmetry and
  no inversion triggers K2's existing kill clause; a second family with asymmetry **but** no
  inversion is *also* informative and should not be silently discarded.
- K3 (cross-benchmark) is unchanged and still `blocking: false`. §1 shows the upstream-report exit is
  unclaimed: `booydar/babilong` has no issue on `metrics.py`.
