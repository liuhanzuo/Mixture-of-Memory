# B11 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU. This file is the formalisation of an adjudication that has already been
run and released — it is not a second novelty check.**

This closes the blocker `proposal/ready_queue.py` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`).

## 0. Status: K1 is DONE and RELEASED. This file's job is to record it, and to re-verify it.

Read from `STATUS.json` (`k1_novelty`, `lifecycle`, `next_gate`) and the two dated adjudications on
disk, not reconstructed:

| | |
|---|---|
| **K1 (novelty, 0 GPU, blocking)** | **RUN 2026-08-14. Verdict `NEEDS_NARROWING`. RELEASED for the narrowed claim only.** Documents: `K1_NOVELTY_CHECK.md` (20.9 KB) + `K1_NARROWING.md` (12.1 KB); raw artefacts in `k1_raw/` |
| `novelty_checked` | `true` |
| `gpu_policy` "NO GPU until K1 passes" | **satisfied — no longer blocking.** K2 (~1.0 GPU-h) is technically permitted |
| `lifecycle` | `ready_cpu` — **declared, and deliberately so**: the recommended next step is 0 GPU by choice, not by prohibition |
| `next_gate` | **K3-EXIT FIRST (0 GPU)**: re-check that `booydar/babilong` still has no issue on `metrics.py` / `preprocess_output` / first-period truncation, then **file it** |

So B11 is the one proposal of the three whose novelty gate is **closed**. Accordingly this document
does three things and no more: **(1)** state the narrowed operative claim as the only claim (§1);
**(2)** present the venue-verified collision table in the standard form, with every venue
**re-verified first-hand today** (§3) — which incidentally closes K1's own admitted gap #1 and turns
up **three factual errors in K1's own table** (§3.3, §4); **(3)** report whether my searches support
or contradict K1's verdict (§5 — they **support** it, with three corrections and one narrowing of the
"searches returned nothing" evidence).

**On the mandated question "does your search contradict K1?"** — No. The verdict `NEEDS_NARROWING`
(not `PASS`, not `FULLY_PREEMPTED`) is the verdict my searches also reach, and the *reason* is the
same: the general form is owned by published ACL-family work, the specific form is not located.
Details, including the three venue corrections, are in §5.

---

## 1. What B11 claims RIGHT NOW — the narrowed claim, and only it

⚠️ The original `STATUS.json.claim` string is **withdrawn as operative** (append-only, so it is still
in the file; `K1_NARROWING.md` §1 records the withdrawal). Anyone writing from the `claim` key is
writing the version K1 killed.

**Operative claim** (`STATUS.json.k1_novelty.narrowed_claim`, condensed here without weakening):

> In a **generative** long-context benchmark whose scorer matches free-form output against a **closed
> task-label vocabulary** (`babilong.metrics.compare_answers`), a **single text-preprocessing
> operation in the scorer's own source** — truncation of the model output at the first period,
> `metrics.py:27` — interacts with an **arm-dependent output-format habit** strongly enough that the
> scorer **fails to recover the ordering** of a within-model architectural manipulation whose true
> effect is **+58 to +84 pp** on a retrieval-closed reference task. Demonstrated by re-scoring
> **byte-fixed generations** with that one operation removed and nothing else changed: on **2 of 6**
> cells the inverted point estimate is removed, and on qa1×32k the 5-point ladder becomes
> **ρ = −1.000** (exact permutation p = 0.0167). The operation is **not a one-directional bug** — it
> *lowers* scores for arms emitting choice-lists (list-format rate 62–75 %) and *raises* them for arms
> emitting continuations — and the four inverting cells are **exactly** the four high-list-format
> cells (`[[4,0],[0,2]]`, Fisher exact p = 0.0667, the minimum attainable at 6 cells, hence
> descriptive). The scorer contains a **second, independently verifiable defect**: the guard intended
> to stop continuation leakage, `metrics.py:31 split('Question')`, is **unreachable** because
> `metrics.py:25` lowercases first.

Five deltas from the withdrawn version, each forced:

1. "destroy the **RANKING**" → "**fail to recover the ORDERING of a within-model manipulation**."
   Between-model leaderboard ranking is §3.1/§3.4/§3.5's territory.
2. "+70 to +84 pp" → "**+58 to +84 pp**." `established_measurements.true_effect_…_pp` is
   `[58.0, 54.0, 84.0, 84.0]`; its minimum is 54, and the two cells carrying the ladder give 58.
   **"+70 as a lower bound" is now a forbidden claim** — our own numbers contradicted our own claim.
3. **Added: "byte-fixed generations, one operation removed."** This is the entire novelty axis.
4. **Added: "not a one-directional bug."** Forced by the executed two-row demonstration (§2.2).
5. **Added: the dead-code guard.**

### 1.1 The claim's own weaknesses, itemised (not hidden in a limitations section)

`K1_NARROWING.md` §3 sizes this honestly and this file does not soften it:

| # | Weakness | Number |
|---|---|---|
| 1 | headline inversion **not significant** | best exact McNemar **p = 0.0703** (b=1, c=7); Holm within the 6-cell family **0.4219** |
| 2 | one-operation ablation repairs **2 of 6** cells | qa2 unrepairable (floor, A4 = 1 %) |
| 3 | dissociation p is **the floor of what 6 cells can produce** | Fisher exact **0.0667** — descriptive |
| 4 | mechanism **NOT identified** | retrieval vs floor collinear, Spearman(recall, A0_acc) = **+0.714** over 6 cells |
| 5 | **one model family** | Qwen3 only; K2 unrun |
| 6 | strongest sub-result is a **single-cell** 5-point ladder | qa1×32k, ρ = −1.000, permutation **p = 0.0167** = the *minimum attainable* over 120 orderings |

The genuinely solid parts are the **two code facts** — deterministic, CPU-checkable in five lines, no
statistics. Everything quantitative rests on 6 cells of one family with a non-significant headline.
`K1_NARROWING.md` §3.1 concludes: **a recommendation to not pursue B11 as a paper.** This Related Work
does not overturn that; §6 endorses it.

---

## 2. What I re-verified by execution today (0 GPU)

Because §1's two load-bearing facts are code facts, they are checkable rather than citable. Both
re-executed on this node:

### 2.1 `metrics.py:31` is dead code — CONFIRMED, and the vendored copy is byte-identical to upstream

```
md5  0a5ecc52ade4e337d35b8f9c97c38310  third_party/babilong-pkg/babilong/metrics.py
md5  0a5ecc52ade4e337d35b8f9c97c38310  (raw.githubusercontent.com/booydar/babilong/main/babilong/metrics.py)
```

**This is new relative to both K1 documents** and it matters for the upstream report: the defect is in
**current upstream `main`**, not only in our vendored copy, so the issue is filable without a
"you're on an old version" rebuttal. Line numbers 25 / 27 / 29 / 30 / 31 match exactly.

Executed:

```
in : 'the football is in the kitchen Question: Where is Mary? Answer: garden'
out: 'the football is in the kitchen question: where is mary? answer: garden'
     -> 'question' survives; the guard did NOT fire
in : 'kitchen <CONTEXT> blah'  -> 'kitchen '   (line 29 DID fire, lowercase tag)
in : 'kitchen <context> blah'  -> 'kitchen '
mechanical: "'Question' in s.lower()" is False for every s   -> verified
```

So the defect is **specific to line 31** — lines 29/30 use lowercase tags and work. It is precisely
the one guard the authors wrote against continuation leakage. (I used `K1_NARROWING.md` §2.1's
corrected demonstration string, not K1's original, which had a leading `...` that made line 27
truncate to `''` — that correction is real and I reproduced its reasoning.)

### 2.2 The truncation is a sign-dependent trade-off — CONFIRMED

Executed against the canonical `compare_answers` with `TASK_LABELS['qa1']`:

| model output | canonical verdict | what truncation did |
|---|---|---|
| `"Choices: A. In the kitchen B. In the garden. The answer is kitchen."` | **False** | **killed** a correct list-format answer |
| `"kitchen. Question: Where is the football? Answer: garden"` | **True** | **saved** a correct answer from continuation leakage |

Both rows reproduce. So "we fix the metric" is forbidden and `notrunc` may never be called a
corrected metric — it *lowers* qa5 (A0 61.0→59.0, A3 57.0→52.0 at qa5×32k). This is scientifically
better than "a bug": it explains why the dissociation is perfectly cell-aligned.

### 2.3 K3-EXIT's precondition — RE-CHECKED TODAY, still unclaimed

`K1_NARROWING.md` §5 flagged that the "18 issues, none about `metrics.py`" check was carried from K1
and **should be re-run immediately before filing**. Re-run today via `api.github.com`:

* `repos/booydar/babilong/issues?state=all&per_page=100` → **18 records, max number 18: 10 issues +
  8 pull requests.** ⚠️ So the number in both K1 documents, "**18 issues**", is a *slight
  overstatement*: it is **10 issues and 8 PRs**. Immaterial to the conclusion, corrected for accuracy.
* Full issue list: #1, #3, #4, #5, #7, #8, #9, #10, #11, #16. **None** concerns `metrics.py`,
  `preprocess_output`, or first-period truncation. #16 (*Performance mismatch in gemma series model*,
  open, the only one search matches on "metrics") is about chat templates and attention backends.
* Search API: `repo:booydar/babilong preprocess_output type:issue` → **0**;
  `repo:booydar/babilong truncat type:issue` → **0**.

**K3-EXIT's exit is still open.** This is the operative precondition for `next_gate` and it is
satisfied as of 2026-08-15.

---

## 3. Named closest collisions (every venue re-verified first-hand 2026-08-15)

Venue rules per `memory/venue-verify-acl-family-needs-anthology.md` (ACL family → Anthology + DBLP)
and `memory/venue-verify-must-use-openreview-2026.md` (OpenReview `venueid` + `Camera_Ready_Revision`
for ICLR/NeurIPS/ICML). **Semantic Scholar not used as a venue authority.**

⚠️ **The ACL-family rule was load-bearing exactly as the memory predicts.** B11's collision set is
**dominated by ACL-family papers** — 4 of the 11 rows below are Anthology-published and would read as
CoRR preprints if graded by DBLP's `journals/corr/` record. One of them (§3.3) K1 **did** mis-grade.

### 3.1 Alzahrani et al. — *When Benchmarks are Targets* — **owns the general form**

* **Venue, re-verified today**: **ACL 2024 Long** — `aclanthology.org/2024.acl-long.744/` HTTP 200,
  `Anthology ID: 2024.acl-long.744`, `Venue: ACL`; DBLP `ACL 2024`. pp. 13787–13805,
  DOI `10.18653/v1/2024.acl-long.744`. `arXiv:2402.01781`.
* **Establishes**: perturbing MCQ benchmarks — **choice order** and **the method of answer
  selection** — moves leaderboard rank by up to **8 positions**; recommends a hybrid answer-selection
  method; ships a fork of `lm-evaluation-harness`.
* **Why not preemption**: the perturbed object is **the benchmark and the answer-selection method**
  (likelihood vs generation vs hybrid), across **different models**, on **MCQ**. Generations are not
  held fixed across conditions and no operation inside a generative scorer's text preprocessing is
  varied.
* **Obligation: MUST CITE as the owner of "answer-selection changes leaderboard rank."** B11 may not
  re-announce it. This single row is why the verdict is `NEEDS_NARROWING`.

### 3.2 Molfese et al. — *Right Answer, Wrong Score*

* **Venue, re-verified today**: **Findings of ACL 2025** — `2025.findings-acl.950` HTTP 200,
  `Venue: Findings`; DBLP `ACL 2025`. pp. 18477–18494, DOI `10.18653/v1/2025.findings-acl.950`.
  `arXiv:2503.14996`.
* **Establishes**: MCQA answer-extraction methods misalign with human judgment; traditional extraction
  **underestimates** capability; LLM extractors have systematic errors; a real trade-off between
  prompt-side format constraints and free-form reasoning.
* **Why not preemption**: **MCQA only.** K1 full-text-checked the arXiv v2 HTML: **0** occurrences of
  `generative`, `long-context`, `ablation`, `BABILong`, `first sentence`, `source code`, `line of
  code`; `ranking` appears once, citing Alzahrani.

### 3.3 ⚠️ Abbood, Meng, Collier — *Time to Revisit Exact Match* — **K1 GRADED THIS WRONG**

* **K1 (P7) says**: "**arXiv only** (DBLP total = 0 for this title)."
* **Measured today**: **Findings of EMNLP 2025.** DBLP search returns
  **`conf/emnlp/AbboodMC25`** — `booktitle = Findings of the Association for Computational
  Linguistics: EMNLP 2025, Suzhou, China`, pp. **11903–11926**,
  DOI `10.18653/v1/2025.findings-emnlp.637`. Anthology `2025.findings-emnlp.637` returns **HTTP 200**,
  title *Time to Revisit Exact Match*, `Venue: Findings`. `arXiv:2509.16720`.
* **This is precisely the failure mode `memory/venue-verify-acl-family-needs-anthology.md` describes**:
  a **Findings** paper reported as a preprint. K1 recorded `DBLP total = 0`, which is not what the API
  returns today (total = 1, and it is the `conf/emnlp` record). Either the query string differed or
  DBLP's index moved; the correction stands either way.
* **Establishes**: replacing EM with sMAPE/MASE on temporal QA "**reshuffles model rankings**
  compared to EM".
* **Why not preemption, unchanged**: it changes the **metric family** (string EM → numeric error), not
  one preprocessing operation inside a fixed scorer. **But the citation obligation is now stronger**,
  because it is peer-reviewed: B11 must cite it as **published** evidence that metric choice reorders
  models, and must therefore be even more explicit that its own axis is *one operation*, not *the
  metric*.

### 3.4 Sanz-Guerrero et al. — *Mind the Gap: Tokenization for MCQA*

* **Venue, re-verified today**: **EMNLP 2025 Main** — `2025.emnlp-main.988` HTTP 200,
  `Venue: EMNLP`; DBLP `EMNLP 2025`. pp. 19573–19583. `arXiv:2509.15020`.
* **Establishes**: how the space after `"Answer:"` is tokenized shifts accuracy by up to **11 %** and
  **reshuffles model rankings**.
* **Why not preemption**: the knob is **tokenization of the prompt suffix** feeding next-token
  probability extraction. **Prompt side, MCQ, probability-based.** B11 is scorer side, generative,
  string-matched. **The prompt-side / scorer-side distinction must appear in B11's abstract**, not only
  in its related work — it is the whole novelty question, and `SOURCES.md` already said so.

### 3.5 Su, Zhang, Ullrich, Bottou, Ibrahim — *A Single Character can Make or Break Your LLM Evals*

* **Venue, re-verified today**: **arXiv-only** — DBLP total = 1, `journals/corr/abs-2510-05152`,
  `CoRR 2025`, *Informal and Other Publications*. No conference record from this node.
  Authors confirmed via arXiv API: Jingtong Su, Jianyu Zhang, Karen Ullrich, Léon Bottou,
  Mark Ibrahim. `arXiv:2510.05152`, 2025-10-02.
* **Establishes**: the in-context-example **delimiter** swings MMLU by ±23 %, and "one can manipulate
  model rankings to put any model in the lead by only modifying the single character separating
  examples."
* **Why not preemption**: the strongest published statement of "a one-character choice controls
  ranking", but the character is in the **prompt**. This paper *confirms the prompt side is taken*,
  which is what makes the scorer side the only open axis.

### 3.6 Yu et al. — **xFinder** — the closest *stance*

* **Venue, re-verified today** (this closes K1's admitted gap #1, which carried this second-hand):
  **ICLR 2025 Poster** — OpenReview `venueid = ICLR.cc/2025/Conference`,
  `venue = "ICLR 2025 Poster"`, forum **`7UqQJUKaLM`**, `Camera_Ready_Revision` **present**.
  ⚠️ Note the **title changed between versions**: the ICLR record is *"xFinder: Large Language Models
  as Automated Evaluators for Reliable Evaluation"*, while DBLP's CoRR records read *"xFinder: Robust
  and Pinpoint Answer Extraction for Large Language Models"*. **Cite the ICLR title.**
  `arXiv:2405.11874`.
* **Establishes**: RegEx answer extraction in mainstream harnesses is only **74.38 %** accurate;
  a trained extractor raises judgment accuracy to **97.61 %**; names "prompt format overfitting" as a
  cheating channel.
* **Why not preemption**: it frames RegEx extraction as an **accuracy** problem to be **replaced
  wholesale by a better module**. It does not ablate *one operation of an existing scorer* and does
  not show a *ranking* destroyed by it. **Closest in spirit to B11's "the metric's code is the
  defect" stance, and the strongest citation obligation after §3.1.**

### 3.7 Yen et al. — **HELMET** — the closest *framing*, and the most dangerous neighbour

* **Venue, re-verified today** (also closes K1 gap #1): **ICLR 2025 Poster** — OpenReview
  `venueid = ICLR.cc/2025/Conference`, forum **`293V3bJbmE`**, `Camera_Ready_Revision` **present**.
  ⚠️ **DBLP returns only `CoRR 2024`** for this title — the family rule was decisive here, exactly as
  `memory/venue-verify-must-use-openreview-2026.md` warns. `arXiv:2410.02694`.
* **Establishes**: names "**Unreliable metrics**" as one of four defects of long-context benchmarks,
  and states that zero-shot prompting "leads to **inconsistent output formats**" (e.g. a long answer
  in RAG where a short one is required).
* **Why not preemption** (K1 read the camera-ready §2.2/§2.3 in full; `k1_raw/helmet.html`): its
  metric fix is **replacing** n-gram/ROUGE with a **reference-based model judge** validated against
  humans (κ = 0.91 / 0.76); its format fix is **prompt-side** (two-shot demonstrations). It uses
  substring EM for RAG **without auditing SubEM's own preprocessing**. It never ablates a scorer
  operation, never holds generations fixed, and never attributes a ranking change to an **arm-wise**
  output-format habit. It argues *"pick a better metric"*; B11 argues *"the metric you already use has
  an identifiable line that inverts your ladder."*
* **Obligation: MUST CITE as the closest adjacent framing**, and distinguish judge-replacement from
  operation-ablation explicitly.

### 3.8 Arjmandi — *Distractor-Aware Truncation* — same benchmark, **opposite side of the pipeline**, concurrent

* **Venue**: **arXiv-only.** DBLP total = **0** (checked today). `arXiv:2608.03297`, **2026-08-04 —
  11 days before this pass.**
* **Establishes**: on **BABILong** + GraphWalks, naive middle-drop **truncation** collapses scores
  (paired Wilcoxon, Holm-corrected p < 0.05 in all eight cells) while **distractor-aware** truncation
  preserves or improves them; replicates on GPT-5.5, ruling out a single-provider artefact;
  concludes "the naive protocol is not a measurement of context-window effects."
* **Why not preemption**: it truncates the **input context**; B11's operation truncates the **model
  output inside the scorer**. Same benchmark, same word, different pipeline stage. And it is
  **concurrent** (11 days), which per the standing rule cannot constitute preemption.
* **Obligation: MUST CITE, defensively.** A reviewer who sees "BABILong" + "truncation" will conflate
  these two papers unless B11 separates them in its first mention.

### 3.9 Jo, Lee, Lee, Lee, Park, Yoo — *Finding Answers in Thought Matters*

* **Venue, re-verified today**: **arXiv-only** — DBLP total = 1, `journals/corr/abs-2510-14773`,
  `CoRR 2025`. arXiv comment reads "ARR Submitted" → no accepted venue. `arXiv:2510.14773`,
  2025-10-16.
* ⚠️ **Author attribution corrected.** `K1_NOVELTY_CHECK.md` §1 lists P6 as "**Kim & Kim (et al.)**".
  There is **no author named Kim.** Re-confirmed from the arXiv API today: **Hwiyeol Jo, Joosung Lee,
  Jaehone Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo**. `K1_NARROWING.md` §4 already caught this;
  independently re-confirmed here. **No `.bib` entry may be generated from K1's row.**
* **Establishes**: reasoning-model scores and final-answer distributions are "highly sensitive to the
  answer extraction algorithm employed"; proposes extraction-rule-agnostic Answer Regeneration.
* **Why not preemption**: sensitivity of **scores**, on math/open-ended QA; the fix is another
  inference pass. No ranking-destruction demonstration, no long-context, no code-line localisation.

### 3.10 ⚠️ Ho, Huang, Boudin, Aizawa — extractive QA, judge vs string match — **K1's title is wrong**

* **K1 (P8) says**: "Rangapur et al. (Sapienza/other), *Reassessing Extractive QA Datasets at Scale:
  LLM-as-a-Judge*, arXiv:2504.11972v3."
* **Measured today**: the arXiv record for `2504.11972` is titled ***Reassessing Extractive QA Datasets
  at Scale: LLM-as-a-Judge and In-Depth Analyses*** (2025-04-16), authors **Xanh Ho, Jiahao Huang,
  Florian Boudin, Akiko Aizawa** — **no author named Rangapur.** OpenReview's DBLP mirror carries the
  work under yet another title, ***LLM-as-a-Judge: Reassessing the Performance of LLMs in Extractive
  QA***, and DBLP has that title as `journals/corr/abs-2504-11972`, `CoRR 2025`. **Venue: arXiv-only**
  (DBLP total = 0 for K1's title string; 1 for the alternate title, and it is CoRR).
* **Establishes**: on 4 extractive-QA datasets, EM correlates with human judgment at only **0.22**,
  F1 at **0.40**, LLM-judge up to **0.85**.
* **Why not preemption, unchanged**: the canonical "judge beats string match" result. Establishes EM
  is a *bad* metric; does not show a specific preprocessing line reordering arms.
* ⚠️ **Two title variants exist for one arXiv ID.** Pin the ID, not the title, and do not generate a
  `.bib` from K1's row.

### 3.11 Garg & Sagtani — *Unsolvability Ceiling in Multi-LLM Routing*

* **Venue, re-verified today**: **arXiv-only** — DBLP total = 1, `journals/corr/abs-2605-07395`,
  `CoRR 2026`. `arXiv:2605.07395`.
* **Establishes**: across 206k query-model pairs on 6 benchmarks, a substantial part of reported
  "unsolvability" comes from **evaluation artifacts**: judge bias favouring verbosity, **truncation
  under fixed generation budgets**, and **output-format mismatches**; these also distort router
  training signals.
* **Why not preemption**: routing-headroom estimation; format mismatch is one of three *lumped*
  artefact categories; no per-operation scorer ablation, no ranking-destruction claim.
* ⚠️ **Shared with B10** (§3.5 of `proposal/backlog/B10-dllm-infilling-ar-dominance/RELATED_WORK.md`).
  The two proposals must not both claim it as *their* nearest neighbour.

### 3.12 BABILong itself — the benchmark under audit

* **Cite**: Kuratov, Bulatov, Anokhin, Rodkin, Sorokin, Sorokin, Burtsev. *BABILong: Testing the
  Limits of LLMs with Long Context Reasoning-in-a-Haystack*, `arXiv:2406.10149`.
* **Venue, verified today**: **NeurIPS 2024** — DBLP `conf/nips/KuratovBARSS024` (+ CoRR 2024).
  ⚠️ OpenReview search returned **no titled note** from this node (likely the D&B track's forum naming);
  DBLP's `conf/nips` record is the authority used. **Track (main vs Datasets & Benchmarks) not
  established** — see §7.
* **Role**: this is the benchmark whose scorer B11 audits, and it is **peer-reviewed at NeurIPS**.
  That cuts both ways and both must be said: it makes the defect **consequential** (a NeurIPS
  benchmark's scorer, used by a public leaderboard), and it obliges B11 to be **scrupulously
  non-accusatory** — the truncation is a **design trade-off** (§2.2), the dead-code guard is a
  **five-line bug**, and neither is scientific misconduct.

---

## 4. MUST-NOT-CLAIM (binding; extends `STATUS.json.forbidden_claims`)

All five entries in `forbidden_claims` stand, plus the four `K1_NOVELTY_CHECK.md` §4 additions. This
pass adds:

1. ❌ **"Novel: preprocessing changes model ranking."** → §3.1 (**ACL 2024**), §3.4 (**EMNLP 2025**),
   §3.3 (**Findings of EMNLP 2025**), loudly §3.5. Only the *generative scorer + one-operation
   ablation on byte-fixed generations + within-model between-arm ladder* combination is open.
2. ❌ **"We fix the metric"** / presenting `notrunc` as a corrected metric. §2.2 — it is a
   sign-dependent trade-off and it *lowers* qa5.
3. ❌ **"+70 pp" as the lower bound** of the true effect. The evidence minimum is **54**.
4. ❌ **"BABILong's scorer is uniquely broken."** Not tested — that is **K3**, still open,
   `blocking: false`.
5. ❌ **Citing *Time to Revisit Exact Match* as a preprint.** It is **Findings of EMNLP 2025**
   (§3.3). K1's own table has this wrong; do not inherit the error.
6. ❌ **Generating a `.bib` entry from `K1_NOVELTY_CHECK.md` §1 rows P6 or P8.** Both have wrong
   author attributions, and P8's title does not match its arXiv ID (§3.9, §3.10).
7. ❌ **Citing xFinder or HELMET as preprints, or by their CoRR titles.** Both are **ICLR 2025
   Posters**; HELMET is CoRR-only in DBLP and xFinder's CoRR title differs from its ICLR title.
8. ❌ **Merging with B04.** B04 is *per-item `acc_norm` margin compression under damage*
   (likelihood ranking, no generation, no retrieval, no string matching), currently
   `NARROWED_TO_OLMO_2_ONLY`. Same phrase "eval fragility", **different construct, mechanism and
   failure surface.** Merging would let an OLMo-2-only margin claim borrow force from an unrelated
   mechanism.
9. ❌ **Claiming B11 and *Distractor-Aware Truncation* study the same thing** (§3.8), or letting a
   reader think so by mentioning "BABILong truncation" without disambiguating.
10. ❌ **"18 issues"** for `booydar/babilong`. It is **10 issues + 8 PRs** (§2.3).

---

## 5. Does my search support K1's released verdict?

**Yes — same verdict, same reason, with three factual corrections and one narrowing of the negative
evidence. No contradiction.**

**Supported, independently:**

1. **The general form is owned by published ACL-family work.** I re-verified §3.1 (ACL 2024), §3.2
   (Findings ACL 2025), §3.4 (EMNLP 2025 Main) first-hand at the Anthology, and **found one more
   published owner K1 had graded as a preprint** (§3.3, Findings EMNLP 2025). This makes K1's
   `kill_if` ("the literature already covers 'preprocessing truncation changes model ranking'") fire
   **more clearly** at the general level than K1 itself recorded. `NEEDS_NARROWING` was right.
2. **No located work varies one operation inside an existing generative scorer with generations held
   byte-fixed.** My searches agree. New empty results beyond K1's list:
   `abs:"re-scoring" AND abs:"same generations"` → **0**;
   `abs:"held fixed" AND abs:"scorer" AND abs:"generation"` → **0**;
   `abs:"fixed generations" AND abs:"metric" AND abs:"ablation"` → **1**, unrelated (biomedical RAG);
   `abs:"exact match" AND abs:"underestimat" AND abs:"language model"` → **0**;
   `abs:"scorer" AND abs:"fragil"` → **4**, none relevant;
   `abs:"metric" AND abs:"source code" AND abs:"ablation" AND abs:"evaluation"` → **13**, none about
   LM evaluation scorers;
   `abs:"benchmark" AND abs:"errors" AND abs:"annotation" AND abs:"leaderboard" AND abs:"re-evaluat"`
   → **0**.
3. **The between-arm (rather than between-model) axis is genuinely unoccupied.** Every ranking paper
   in §3 is about leaderboards. I found no exception.
4. **K3-EXIT's precondition holds today** (§2.3), which was the one thing `K1_NARROWING.md` §5 said
   must be re-checked before acting. It was, and it does.

**Corrections to K1 (all recorded, none changing the verdict):**

| # | K1 said | Measured today |
|---|---|---|
| C1 | P7 *Time to Revisit Exact Match* = "arXiv only, DBLP total = 0" | **Findings of EMNLP 2025**, `2025.findings-emnlp.637`, DBLP `conf/emnlp/AbboodMC25`, pp. 11903–11926 (§3.3) |
| C2 | P6 authors "Kim & Kim (et al.)" | **Jo, Lee, Lee, Lee, Park, Yoo** — no Kim (§3.9). Already caught by `K1_NARROWING.md` §4; independently re-confirmed |
| C3 | P8 = "Rangapur et al., *Reassessing Extractive QA Datasets at Scale: LLM-as-a-Judge*" | authors **Ho, Huang, Boudin, Aizawa**; arXiv title has "**and In-Depth Analyses**"; a second title variant exists for the same ID (§3.10) |
| C4 | "`booydar/babilong` has **18 issues**" | **10 issues + 8 PRs** = 18 records (§2.3). Conclusion unaffected |

**One narrowing of the negative evidence.** K1 §1 presents seven zero-result arXiv queries as
"evidence of the gap". They are — but they are **abstract-field (`abs:`) queries**, so they cannot see
a paper that does this without those words in its abstract. My additional zero-results have the same
limitation. **"Searched and not found" is the correct strength; "does not exist" is not available**
from this instrument, and B11's write-up must use the *"To the best of our knowledge, among the
audited literature…"* construction.

**One thing I could not check and K1 did not either**: none of §3's papers was read in full text
today. K1 did read HELMET's camera-ready §2.2/§2.3 and Molfese's v2 HTML (artefacts in `k1_raw/`).
For §3.3 — newly promoted to *published* — a full-text read is now required before citation, because
its citation obligation just got stronger.

---

## 6. Safe residual claim — one falsifiable sentence

> **Re-scoring a fixed set of BABILong generations with `metrics.py:27`'s first-period truncation
> removed and nothing else changed does not alter the sign of any cell's arm ordering** — i.e. the
> scorer operation is causally irrelevant to which arm looks better.

That is the null, and it is **already rejected on 2 of 6 cells** by
`analyze_a02_truncation_ablation.py` over the on-disk byte-fixed generations. It is 0 GPU, fully
CPU-reproducible, and the artefact exists (`a02_truncation_ablation.json`). Its companion:

> **`metrics.py:31`'s `split('Question')` guard fires for at least one input.**

Rejected mechanically: `'Question' in s.lower()` is `False` for every `s` (§2.1). No statistics, no
GPU, no second family — and true of **current upstream `main`**, not just our vendored copy.

**Everything beyond those two is not currently supported at paper scale**, per §1.1.

---

## 7. Verdict

```
verdict: hold_in_backlog -- and the recommendation is NOT to pursue B11 as a paper
K1 novelty gate: DONE 2026-08-14, NEEDS_NARROWING, RELEASED for the narrowed claim only.
                 Re-verified 2026-08-15: verdict SUPPORTED, 4 corrections (section 5).
gpu: K1 no longer blocks, so K2 (~1.0 GPU-h) is PERMITTED but NOT RECOMMENDED --
     at 6 cells the best attainable Fisher p is already 0.0667, so a second family
     cannot make the dissociation significant; it can only tell us whether the SIGN
     reproduces. Expected information is low and known to be low in advance.
next_gate: K3-EXIT (0 GPU) -- file the upstream issue. Precondition re-checked today
     and still unclaimed (section 2.3). This dominates K2 on information-per-cost.
promotion: NOT eligible, and per K1_NARROWING section 3.1 probably never will be.
     B11's solid content is two code facts; that is an issue report plus an A02 appendix.
```

* **No candidate is 完全相同 / 抄袭 of the narrowed claim.** Every located work varies the **prompt**
  (§3.4, §3.5, §3.7), swaps the **extraction module wholesale** (§3.6, §3.9), changes the **metric
  family** (§3.3, §3.10, §3.7), or perturbs the **benchmark** (§3.1, §3.8). None varies **one
  operation inside the existing scorer with generations byte-fixed**, and none breaks a ladder over
  **arms of one model**. So `already_dead_should_archive` is **not** warranted on literature grounds.
* **But B11 is nonetheless recommended for non-pursuit**, on **evidence** grounds — one family, a
  non-significant headline, 2/6 repair, an unidentified mechanism, and a best-attainable p that is
  already at the floor. `K1_NARROWING.md` §3.1 reached that recommendation and this file endorses it.
  **The distinction matters**: B11 is not being killed by the literature (which the standing rule
  forbids); it is being *declined* on its own measured thinness, which is exactly what a kill gate is
  for.
* **The right disposition is therefore the one already in `next_gate`**: spend **0 GPU**, file the
  upstream issue, keep the two code facts as an A02 appendix, and leave B11 in backlog.

---

## 8. Honest gaps in this adjudication

1. **DBLP and the arXiv API were both intermittent all session** — `curl: (28)` 30–45 s timeouts and
   `curl: (56) Failure when receiving data from the peer`; some records needed 3 retries;
   `2509.16720` and `2504.11972` had to be fetched in a background retry loop. Every row in §3 comes
   from a call that **returned**. **Semantic Scholar was not queried at all** (repo rule).
2. **`api2.openreview.net` worked from this node today** (HTTP 200 on `/notes/search`), which is why
   K1's gap #1 could be closed. ⚠️ This contradicts `A01/RELATED_WORK.md` §5.1, which recorded api2
   returning **HTTP 403 `ChallengeRequiredError`** on every path — so **api2 availability is
   intermittent across sessions and must be re-tested, not assumed, each time.**
3. **BABILong's NeurIPS 2024 *track* is unestablished.** DBLP `conf/nips/KuratovBARSS024` confirms
   NeurIPS 2024; whether it is the main track or Datasets & Benchmarks was not determined, because
   OpenReview search returned no titled note for it. **Do not write "NeurIPS 2024 D&B" without
   checking.**
4. **§3.5, §3.9, §3.10, §3.11, §3.8 are `arXiv-only` = *venue unverifiable from this node***, not
   *no venue exists*. Recent-conference lag in DBLP is documented
   (`memory/venue-verify-must-use-openreview-2026.md`), and §3.3 is a live example of exactly that
   lag biting. Re-verify per family before submission.
5. **No full text was read this session.** §3's characterisations are abstract + venue metadata, plus
   K1's own full-text reads (HELMET camera-ready, Molfese v2) carried from `k1_raw/`. **§3.3 now needs
   a full-text read** because its status changed to published.
6. **K1's statistics were not recomputed** — p = 0.0703 / 0.4219 / 0.0167 / ρ = +0.714 are carried
   from `STATUS.json` and `K1_NOVELTY_CHECK.md`. `K1_NARROWING.md` §5 makes the same disclosure.
   They are internally consistent across the two documents, which is **not** independent replication.
   I verified only the two **code** facts, by execution (§2).
7. **Zero cross-disk verification.** The A02 evidence JSONs are recorded as md5-identical on wzc1 and
   zwfy6; `/apdcephfs_zwfy6` is not mounted on LOCAL, so I confirmed neither side.
   Per `memory/two-disk-rule-applies-to-main-too.md`, that is recorded, not glossed.
8. **No `.bib` entries emitted.** Given C1–C3 in §5 — one wrong venue, two wrong author lists, one
   ID-vs-title mismatch — emitting a bibliography from the existing tables would propagate three
   errors. Entries must be generated from **§3's IDs**, per family, after the full-text reads in
   gap #5.
