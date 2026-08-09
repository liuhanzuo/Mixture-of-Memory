# A04 — Related Work / Novelty Boundary

**Date**: 2026-08-09. **Author**: CPU/design pass, zero GPU spent.
**Purpose**: satisfy the mandatory Related Work threshold in `proposal/README.md` (§"Related Work
强制门槛") and close the `RELATED_WORK_GAP_AUDIT_20260808.md` finding that A04 is **不足**
(insufficient: internal motivation only, no Related Work).

## 0. Established starting point (do not relitigate)

`proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md` states, and this document accepts
as settled:

> A04 不能主张首次研究 pruning recovery.
> (A04 cannot claim to be the first to study pruning recovery.)

Everything below assumes that. The question this document answers is narrower and harder: **after
conceding pruning recovery to prior work, is there a defensible A04 left, and what exactly is it?**

## 1. Venue verification method (per repo standing rule)

Two families, two authorities. Using the wrong one produces false calls, and this repo has been
burned in both directions (see `memory/venue-verify-must-use-openreview-2026.md` and
`memory/venue-verify-acl-family-needs-anthology.md`).

| Family | Authority used here |
|---|---|
| ICLR / NeurIPS / ICML / TMLR | OpenReview `venueid` (+ `Camera_Ready_Revision` invitation as the accept signal) |
| ACL / EMNLP / NAACL incl. Findings | ACL Anthology + DBLP |

Sources actually queried on 2026-08-09 through `hy-proxy.woa.com:3128`:

* `https://export.arxiv.org/api/query?id_list=<id>` — **note**: requires a non-default
  `User-Agent`; with curl's default UA the endpoint returns an empty body and any XML parse of it
  fails silently-looking. Two of my first three calls failed this way before I set
  `-A "Mozilla/5.0 (research-audit)"`.
* `https://api2.openreview.net/notes/search?term=<title>&content=all` — returned per-note
  `venue`, `venueid`, `domain`, `invitations`.
* `https://dblp.org/rec/journals/corr/abs-<id>.bib` — **DBLP's `/search/publ/api` endpoint was
  returning HTTP 500 for every query all session**; the per-record `.bib` URLs work (some need one
  retry past a 503). Any future agent hitting DBLP 500 should use the direct record URL, not
  conclude "not in DBLP".
* Semantic Scholar `graph/v1/paper/arXiv:<id>` — usable but rate-limited to 429 after ~3 calls;
  used only as a cross-check, **never as the venue authority**.
* ACL Anthology `https://aclanthology.org/search/?q=...` — the search page is client-side rendered,
  so the server HTML contains **no hits**; it cannot be scraped this way. See §4 for the one paper
  where this matters and how it was resolved instead.

## 2. The three collisions named by the audit

### C1. Ghosted Layers: Unconstrained Activation Alignment for Recovering Layer-Pruned LLMs

* **Citation**: Vincent-Daniel Yun, Junhyuk Jo, Sai Praneeth Karimireddy, Sunwoo Lee.
  arXiv:2605.15491v2 (v1 2026-05-15, v2 2026-06-07). cs.LG/cs.AI/cs.PF.
* **Verified venue**: **ICML 2026 Workshop AdaptFM, Poster.**
  Verified via OpenReview: note `U37zxXMEE0`, `venue = "AdaptFM Poster"`,
  `venueid = ICML.cc/2026/Workshop/AdaptFM`, `domain = ICML.cc/2026/Workshop/AdaptFM`,
  invitations = `[.../Submission, .../Submission_Change_Before_Reviewing, .../Submission_Release]`.
  No `Camera_Ready_Revision` invitation (workshop track, so none expected).
  DBLP has it only as `journals/corr/abs-2605-15491` (CoRR 2026). S2 reports venue `arXiv.org`
  — **this is the exact S2 lag the repo rule warns about**; S2 alone would have called it a
  preprint. **It is a workshop paper, not a main-conference paper.**
* **What it does**: training-free recovery of a layer-pruned LLM. Derives a closed-form optimal
  *linear* operator from a small calibration set to reconstruct the boundary-activation discrepancy
  the removed blocks would have produced. Argues its solution is the unconstrained optimum of the
  alignment objective where prior methods are restricted to constrained operator subspaces.
  Evaluated across multiple backbones and pruning strategies vs prior training-free baselines.
* **Overlap with A04**: same object of study (a layer-pruned LLM that must be brought back),
  same reported quantities (accuracy + perplexity), same intuition that the injury is a
  representation-interface mismatch at the cut.
* **Where it does not reach A04**: it is a **method for producing recovery**, measured by
  "did the number go up vs a baseline". It is **training-free by construction** — there is no
  training trajectory, therefore no token budget, no seed, no checkpoint grid, and **no possible
  question of when to stop**. It offers no criterion for declaring a recovered model equivalent to
  the intact target; it reports improvement, not sufficiency.
* **Relationship**: **not preemption, and it is one of the methods a certification protocol would
  adjudicate.** A04 is downstream of it, not competing with it.

### C2. On the Limits of Layer Pruning for Generative Reasoning in Large Language Models

* **Citation**: Safal Shrestha, Anubhav Shrestha, Aadim Nepal, Minwu Kim, Keith W. Ross.
  arXiv:2602.01997v4 (v1 2026-02-02, v4 2026-08-06 — i.e. **updated three days before this
  document**). cs.LG/cs.AI.
* **Verified venue**: **UNDER REVIEW, not published.** ACL family, so per repo rule the authority
  is Anthology/DBLP, not OpenReview:
  * DBLP: `journals/corr/abs-2602-01997`, `journal = CoRR`, `year = 2026`. **CoRR only — no
    Anthology-backed conference record.**
  * OpenReview additionally shows note `LUBdOuX62N`,
    `venueid = aclweb.org/ACL/ARR/2026/May/Submission`, whose own bibtex reads
    `booktitle = {Submitted to ACL Rolling Review - May 2026}, note = {under review}`.
    ARR submission ≠ acceptance; it is evidence of *pending* status only.
  * S2 reports `venue: ""` (empty), `publicationVenue: null`.
  * Anthology search could not be used (client-side rendering, §1); but a paper still in ARR May
    2026 review, whose own OpenReview record says "under review", cannot yet be in the Anthology.
  * **Must be cited as arXiv preprint / ACL ARR under review.** Do not cite as an ACL paper.
* **What it does**: shows layer pruning retains classification performance but that **generative
  reasoning (GSM8K, HumanEval+) recovers substantially worse**; identifies loss of specific
  algorithmic capabilities (arithmetic, balanced-parenthesis generation); tests recovery by SFT on
  self-generated responses under a deliberately realistic constraint (single 80GB GPU, no
  pretraining-scale data/compute), reaching up to 90% of baseline on classification but limited
  recovery on generative reasoning even in a task-aligned setting (full FT on self-generated GSM8K).
  As complementary evidence, analyses a depth-pruned model trained with **nearly 100B
  post-pruning tokens** and finds deficits persist on simple arithmetic.
* **Overlap with A04 — this is the sharpest collision of the three.** It already establishes the
  *scientific content* of A04's most quotable intuition: **recovery is metric-dependent, and one
  family of metrics can look recovered while another is not.** A04 must not present
  "different metrics disagree about recovery" as a discovery.
* **Where it does not reach A04**: (a) its recovery axis is **SFT on self-generated data** under a
  constrained post-training budget, not matched-corpus/matched-token continued pretraining;
  (b) its evidence is **one recovery run per configuration** — the abstract makes no claim about
  seeds, and there is no variance model, so it cannot say whether an observed gap exceeds
  run-to-run spread; (c) it reports *that* recovery is limited, and does not offer a **decision
  rule** — no threshold, no equivalence test, no stopping criterion, no statement of what evidence
  would license declaring a model recovered; (d) the 100B-token analysis is an *analysis of an
  existing model*, not a controlled ladder.
* **Relationship**: **the correct positioning is follow-up-that-fixes-a-defect, not
  differentiation-by-topic.** A04's contribution over C2 must be procedural (multi-seed, matched
  budget, pre-registered decision rule), and A04's introduction should credit C2 for the
  metric-dependence finding rather than restate it.

### C3. Layer as Puzzle Pieces: Compressing Large Language Models through Layer Concatenation (CoMe)

* **Citation**: Fei Wang, Li Shen, Liang Ding, Chao Xue, Ye Liu, Changxing Ding.
  arXiv:2510.15304v1 (2025-10-17). cs.CV/cs.LG.
* **Verified venue**: **NeurIPS 2025, Poster.** Verified via OpenReview: note `enhFXzKii4`,
  `venue = "NeurIPS 2025 poster"`, `venueid = NeurIPS.cc/2025/Conference`, and crucially the
  invitation list contains
  `NeurIPS.cc/2025/Conference/Submission22642/-/Camera_Ready_Revision` — the accept signal the
  repo rule asks for. Bibtex: `booktitle = {The Thirty-ninth Annual Conference on Neural
  Information Processing Systems}, year = 2025`.
  (S2 also says "Advances in Neural Information Processing Systems 38" — **note the volume
  disagreement**: S2 says 38, OpenReview's own bibtex says Thirty-ninth. Cite from OpenReview.)
* **What it does**: progressive layer pruning with concatenation-based merging of the most critical
  channels across adjacent layers (channel sensitivity from activation intensity + weight norms),
  plus a **hierarchical distillation** post-training stage that exploits the original↔pruned layer
  correspondences established during pruning. 30% of LLaMA-2-7b parameters removed retains 83% of
  original average accuracy across seven benchmarks.
* **Overlap with A04**: shares the "pruning creates a recovery problem, recovery needs a
  post-training stage" frame, and reports the same style of headline (fraction of original accuracy
  retained). The "83% of original average accuracy" number is exactly the kind of aggregate that
  A04's protocol would demand be decomposed and null-calibrated.
* **Where it does not reach A04**: it is a **compression-method paper** whose recovery stage is
  instrumental. The reported retention is a **ratio against an unconditioned baseline**, averaged
  over seven benchmarks — no per-benchmark best-constant floor, no seeds, no equivalence test, no
  stopping rule. Its own headline is in fact a target for A04's critique.
* **Relationship**: **not preemption.** Different contribution type (method vs measurement
  protocol). A04 must cite it as the state of the art in "recovery via distillation" and must not
  claim to invent post-pruning recovery training.

## 3. Collisions the audit did **not** name, found in this pass

These matter more than C1/C3 because they attack A04's *actual* proposed novelty (the
certification/stopping rule), not its topic.

### N1. Small LLMs: Pruning vs. Training from Scratch — **the closest methodological collision**

* **Citation**: Yufeng Xu, Taiming Lu, Kunjun Li, Jiachen Zhu, Mingjie Sun, Zhuang Liu.
  arXiv:2606.14150v3 (v1 2026-06-12, v3 2026-06-29). Code:
  `github.com/zlab-princeton/pruning-vs-scratch`.
* **Verified venue**: **preprint.** DBLP `journals/corr/abs-2606-14150`, `journal = CoRR`,
  `year = 2026`, timestamp 2026-07-07. OpenReview search returns **no match** under either the
  exact title or a paraphrase, so there is no OpenReview-visible submission (note: an OpenReview
  absence is weaker evidence than a positive `venueid`; recorded as such). S2 was 429 rate-limited
  on this ID and could not be used — **this is the one venue call resting on DBLP + OpenReview
  absence rather than two positive sources.**
* **What it does**: prunes Llama-3.1-8B at ratios 0.5–0.8 with six methods spanning depth, width,
  and sparse granularity, **under two controlled token-matched settings**: (1) same training token
  budget, pruned-init vs random-init; (2) from-scratch given the full token budget consumed by the
  whole pipeline. Finds pruned init consistently beats random init at matched budget, with the
  advantage narrowing as budget grows and pruning ratio rises and nearly vanishing at the highest
  ratio; and that at the full-pipeline budget, fine granularities keep an advantage while coarse
  structured pruning can be matched or surpassed.
* **Why this is the dangerous one**: A04's `## 提案` items 1–3 (same physical data, matched token
  presentations/FLOPs rather than optimizer steps, same optimizer/LR/batch) and the `## 1B MVP`
  arm "random trunk + inherited lexical/readout interface" are, at the level of *experimental
  design*, **largely this paper's design**. Specifically:
  * "matched token budget, pruned-init vs random-init" is N1's setting (1);
  * "compare against from-scratch given the whole pipeline's tokens" is N1's setting (2), and A04
    does not currently even have that arm;
  * N1 covers depth pruning explicitly, i.e. A04's damage type.
  **A04 must therefore drop any claim of novelty for "controlled token-matched comparison of
  pruned vs from-scratch initialisation."** That is N1's result.
* **What N1 still does not do**: the abstract's outcome variables are comparative
  ("pruned better than scratch", "advantage narrows"). It gives a **recommendation** ("with a
  limited token budget, prune; with unlimited budget, scratch can be competitive"), which is a
  *decision rule about which pipeline to choose*, **not** a rule for deciding whether a given
  recovered model has reached its target. There is no visible per-metric best-constant floor, no
  multi-seed variance model, and no equivalence/non-inferiority statistic. Its comparisons are
  superiority comparisons; certification is an equivalence question, which is a different and
  strictly harder statistical statement.
* **Relationship**: **narrow-and-follow-up, not abandon.** Within the standing directive
  (overlap ≠ preemption; the bar is "essentially identical scope"), N1's scope is *"is pruning a
  good way to get a small model?"* and A04's residual scope is *"given a recovery run, what
  evidence licenses stopping it?"*. Those are not identical. But A04 loses its design novelty and
  must cite N1 as the source of the matched-budget methodology it is adopting.

### N2. Beyond Perplexity: A Behavioral Evaluation Framework for Deployment-Memory Claims in LLM Test-Time Training

* **Citation**: Xiangchen Song, Zhenhao Chen, Lingjing Kong, Shaoan Xie, Xinshuai Dong,
  Guangyi Chen, Kun Zhang. arXiv:2607.00368v1 (2026-07-01).
* **Verified venue**: **preprint.** DBLP `journals/corr/abs-2607-00368`, `journal = CoRR`,
  `year = 2026`. S2: `venue: ""`, `publicationVenue: null`. OpenReview: no match.
* **What it does**: argues that TTT/memory work is judged by **local proxy metrics** (perplexity,
  future-token loss, reward) that are weak evidence for the capability claims they are used to
  motivate. Introduces a **claim-calibrated evidence ladder** separating stream/domain adaptation,
  bridge internalization, and deployment-time behavioural learning, plus an evaluation protocol
  with **matched explicit-memory baselines and mutually exclusive failure categories**. Validates
  it by auditing recent work and by a controlled diagnostic where one-step LoRA lowers support and
  answer loss on three Qwen3 scales while free-form recall stays at **zero**.
* **Overlap with A04 — this is the closest collision to A04's *framing***, closer than any of
  C1/C2/C3. "A protocol that calibrates a claim to the evidence that licenses it, because
  likelihood-style proxies improve while the target capability does not move" is a one-line summary
  of both N2 and of A04's proposed contribution. N2's LoRA diagnostic (loss down, recall flat) is
  structurally the same shape as A04's robust starting point ("PPL 持续改善不能单独证明已经接近
  intact target").
* **Where it does not reach A04**: different injury and different regime. N2's subject is
  **test-time training / deployment memory** on an intact model; A04's is **structural injury**
  (layers physically removed) and its recovery by continued pretraining. N2's ladder separates
  *claim types*; it does not supply a quantitative stopping threshold for a training run.
* **Relationship**: **A04's framing is no longer novel in the abstract and must be positioned as
  the structural-injury instance of an argument N2 makes for TTT.** A04 may not present
  "proxy metrics overstate capability, so calibrate claims to evidence" as its own idea.

### N3. Perplexity Cannot Always Tell Right from Wrong

* **Citation**: Petar Veličković, Federico Barbero, Christos Perivolaropoulos, Simon Osindero,
  Razvan Pascanu. arXiv:2601.22950v1 (2026-01-30), 11 pages, 4 figures.
* **Verified venue**: **ICML 2026 Workshop CTB — SUBMISSION, not an accepted poster.** OpenReview
  gives `venue = "ICML 2026 Workshop CTB Submission"`,
  `venueid = ICML.cc/2026/Workshop/CTB/Submission`. Note the `/Submission` suffix on the
  `venueid` — contrast C1's `ICML.cc/2026/Workshop/AdaptFM` with `venue = "AdaptFM Poster"`.
  **The suffix is the difference between submitted and accepted; do not cite this as a workshop
  paper.** DBLP: `journals/corr/abs-2601-22950`, CoRR 2026.
* **What it does**: a **theoretical** result. Using Transformer continuity, proves that if a
  compact decoder-only Transformer predicts any sequence accurately and confidently, there must
  exist another sequence with very low perplexity that the same model predicts incorrectly; and
  via **iso-perplexity plots** shows perplexity will not always select the more accurate model
  (any confidence increase must be matched by a commensurate accuracy rise for perplexity to
  prefer the new model).
* **Overlap with A04**: A04's central negative premise — "PPL improvement is not sufficient
  evidence of target recovery" — has a **theoretical proof** here, and the term
  "iso-perplexity" is already in use. A04 must cite this and must not claim the
  PPL-insufficiency insight.
* **Where it does not reach A04**: it is model-selection theory on intact models. It gives no
  protocol, no damage regime, no empirical recovery ladder.
* **Relationship**: strengthens A04's motivation while removing A04's ability to claim it.
  Useful: it means A04's matched-PPL arm is testing a *theoretically predicted* effect rather than
  fishing.

### N4. Perplexity Can Miss SAE Feature Damage Under Quantization

* **Citation**: Evan Duan. arXiv:2606.03002v2 (v1 2026-06-02, v2 2026-06-04).
* **Verified venue**: **Under review for TMLR** (not accepted). OpenReview `venue = "Under review
  for TMLR"`, `venueid = TMLR/Under_Review`, `domain = TMLR`. The paper's own arXiv comment says
  "Submitted to TMLR", consistent. DBLP: `journals/corr/abs-2606-03002`, CoRR 2026.
* **What it does**: uses a frozen SAE as a fixed measurement basis to show perplexity can *improve*
  while 18.7% of active features degrade (Gemma-2-2B INT7); INT6 improves perplexity with only
  51.3% feature survival. Notably also reports that RTN quantization and **matched-perplexity
  magnitude pruning** damage strongly overlapping feature sets (Jaccard 0.79–0.86, damage-score
  Spearman 0.98).
* **Overlap with A04**: same logical move (behavioural parity ≠ recovery) in a different injury
  (quantization), and it **already uses a matched-perplexity pruning comparison**. So
  "compare arms at matched PPL" is not a novel manoeuvre.
* **Where it does not reach A04**: its residual axis is *interpretability-feature fidelity*, not
  task capability, and it has no training-trajectory or stopping question.
* **Relationship**: cite as independent corroboration in a second injury modality; do not claim
  matched-PPL comparison as new.

### N5. Compressed Models are NOT Trust-equivalent to Their Large Counterparts

* **Citation**: Rohit Raj Rai, Chirag Kothari, Siddhesh Shelke, Amit Awekar. arXiv:2508.13533v1
  (2025-08-19). cs.CL/cs.LG.
* **Verified venue**: **preprint.** DBLP `journals/corr/abs-2508-13533`, CoRR 2025. OpenReview: no
  match. (S2 429-limited; not used.)
* **What it does**: explicitly separates **performance parity** from **equivalence**. Proposes a
  two-dimensional "trust-equivalence" evaluation — interpretability alignment (LIME/SHAP) and
  calibration similarity (ECE, MCE, Brier, reliability diagrams) — and finds low alignment and
  significant calibration mismatch between BERT-base and its compressed variants **even when
  accuracies are nearly identical**. Concludes drop-in replacement requires assessment beyond
  performance parity.
* **Overlap with A04**: it is the **closest paper to A04's word "certification"**: same claim
  structure ("matched top-line ⇏ equivalent"), same prescription (a multi-axis equivalence check
  before declaring a compressed model a substitute).
* **Where it does not reach A04**: BERT-base classification, compression not structural depth
  injury, **no recovery training at all**, and its axes are interpretability/calibration rather
  than capability-vs-null. It is a static two-model comparison, not a trajectory with a stopping
  decision.
* **Relationship**: this is the paper a reviewer will bring up if A04 says "certification". A04
  must cite it and state the difference in one sentence: N5 certifies *a finished compressed
  model against its parent on trust axes*; A04 would certify *a recovery trajectory's stopping
  point against a target on capability axes with construct-appropriate nulls*.

### N6. Baseline anchors that must be cited (not collisions, but non-negotiable citations)

Venues verified in this pass:

* **The Unreasonable Ineffectiveness of the Deeper Layers** (arXiv:2403.17887) — **ICLR 2025
  Poster**, OpenReview `venueid = ICLR.cc/2025/Conference` with
  `Submission13737/-/Camera_Ready_Revision`. (Also has an earlier NeurIPS 2024 SciForDL workshop
  record.) The canonical "prune deep layers + heal" reference.
* **Compact Language Models via Pruning and Knowledge Distillation** (Minitron, arXiv:2407.14679) —
  **NeurIPS 2024 Poster**, OpenReview `venueid = NeurIPS.cc/2024/Conference` with
  `Submission15087/-/Camera_Ready_Revision`. The "retrain with <3% of original tokens" reference,
  i.e. the standard against which any token budget claim is read.
* **Shortened LLaMA** (arXiv:2402.02834) — the arXiv title is *"Shortened LLaMA: Depth Pruning for
  Large Language Models with Comparison of Retraining Methods"*, but OpenReview/DBLP records carry
  the title *"Shortened LLaMA: A Simple Depth Pruning for Large Language Models"*, at
  **ICLR 2024 Workshop ME-FoMo, Poster** (`venueid = ICLR.cc/2024/Workshop/ME-FoMo`) plus CoRR
  2024. **Title differs between arXiv and the venue record** — cite carefully; this is the exact
  arXiv-vs-camera-ready divergence the repo memory warns about.
* **LLM Pruning and Distillation in Practice: The Minitron Approach** (arXiv:2408.11796) — venue
  NOT verified in this pass; treat as UNVERIFIED until checked.

## 4. Net verdict on the Related Work gate

**The gate is CLEARED for the topic, but it forces A04 to be renamed and narrowed. It is not
cleared for A04 as currently written in `PROPOSAL.md`.**

Nothing found does A04's residual thing at A04's scope, so **no paper preempts A04** under the
standing directive ("overlap is not preemption; the bar is essentially identical scope"; work
within 2–3 months is concurrent). The closest single paper is **N1 (arXiv:2606.14150)**, and what
it kills is A04's *experimental-design* novelty, not A04's question. **N1 is also concurrent
(2026-06-12, under two months before this pass), so by the standing rule it could not preempt A04
even if the scope matched.**

But the collisions jointly strip four things A04's current `PROPOSAL.md` implicitly claims:

| A04 currently implies | Killed by | Consequence |
|---|---|---|
| studying pruning recovery is the contribution | C1, C2, C3 | already conceded by the audit |
| controlled matched-token pruned-vs-scratch comparison is the contribution | **N1** | must cite N1 and adopt, not claim |
| "PPL improvement ⇏ recovery" is the finding | **N3** (theory), N4, N2 | must cite; motivation only |
| "proxy metrics overstate capability, calibrate claims to evidence" is the framing | **N2** | must cite; A04 is the structural-injury instance |
| "certification of a compressed model" is a new idea | **N5** | must cite and distinguish on all four axes |

### 4.1 What A04 may claim after this audit

A single sentence, and it is much narrower than the current `PROPOSAL.md`:

> A **pre-registered, multi-seed, matched-corpus/matched-token stopping rule** for post-injury
> recovery training, whose accept decision is an **equivalence (non-inferiority) test against the
> intact target on capability axes each calibrated to its own best-constant floor**, together with
> a demonstration that the stopping rules currently in use (likelihood/PPL plateau; aggregate
> retention ratio) accept models that this rule rejects.

The load-bearing, checked-for-absence components are:
1. **equivalence/non-inferiority direction, not superiority.** Every collision above runs
   superiority comparisons ("better than baseline", "retains 83%"). N5 argues for equivalence but
   supplies no test statistic. Searches for `abs:"non-inferiority" AND abs:"language model"`,
   `abs:"equivalence test" AND abs:"neural network"`, and `abs:"TOST" OR abs:"two one-sided tests"`
   returned **zero** pruning/recovery papers — the only ML-adjacent TOST hit was
   arXiv:2604.14419 (Equifinality in Mixture of Experts), which applies TOST to MoE *routing
   topology* PPL equivalence at 76–84M params, an unrelated object. So **TOST-style equivalence
   testing is essentially absent from the pruning-recovery literature**, and importing it is a real
   (if modest) methodological contribution.
2. **construct-appropriate nulls per capability axis**, imported from A01 (which is this project's
   own prior work and must be cited as such, not re-claimed).
3. **run-level variance as a first-class quantity** (seeds), which none of C1/C2/C3/N1 reports.
4. **decision rule stated before data**, which none of them has.

### 4.2 Must-NOT-claim list (binding on any A04 writeup)

1. ❌ First to study layer-pruning recovery. (C1, C2, C3, and the audit itself.)
2. ❌ First to observe that recovery is metric-dependent / that classification recovers while
   generative reasoning does not. **C2 owns this.**
3. ❌ First to run a controlled token-matched comparison of pruned-init vs from-scratch-init.
   **N1 owns this**, across six pruning methods and two budget regimes at 8B — a broader sweep
   than A04's 1B MVP.
4. ❌ First to note that perplexity is an unreliable proxy for capability. **N3 proves it;**
   N4 shows it for quantization; N2 shows it for TTT.
5. ❌ First to propose calibrating claims to the evidence that supports them. **N2 owns this
   framing.**
6. ❌ First to argue performance parity ≠ equivalence for a compressed model. **N5 owns this.**
7. ❌ First to use matched-perplexity comparison between arms. **N4 already does it.**
8. ❌ "Certification" without defining it as *equivalence-to-target on null-calibrated capability
   axes with a pre-registered threshold*. Undefined, the word collides directly with N5.
9. ❌ Any claim of a depth *scaling law* from the existing Paper B keepN ladder. Independent of the
   literature — it is confounded three ways in our own data (see `STATUS.json:warning` and §5).
10. ❌ Any claim that Paper B used differential LR. Verified false in
    `status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md` (all `fresh_*` param groups empty; the
    `module.` prefix strip landed only in commit `7a330ce`).
11. ❌ Novelty for "layer-pruned models can be repaired by a linear map at the cut" — **C1 owns
    this**, in closed form and training-free.

### 4.3 Kill recommendation for the direction

**Do not archive.** But A04 as written is not fundable: its `## 提案` list is now mostly N1's
design and its motivation is now mostly N2/N3's argument. The residual contribution is real but
**narrow and methodological**, so it is worth at most a short paper or a methods section, and only
if the gate in `A04_GATE_DESIGN.md` fires positively. If the gate's own kill condition fires,
archive with a POSTMORTEM rather than reframing — the reframing space has just been shown to be
occupied.

## 5. Internal-consistency note (not literature, but load-bearing)

The existing OLMo-2 keepN ladder cannot serve as A04's evidence base, for reasons verified in this
repo's own files, independent of any prior work:

* **Two different corpora.** `data/dolmino_now15b.npy` is **62,020,903,040 bytes on wzc1**
  (= 7,570,911 rows × 2048 × int32) but **126,907,244,672 bytes on zwfy6** (= 15,491,607 rows),
  ratio **2.0462×** — same filename, different file, wzc1 a byte-prefix of zwfy6. I re-derived both
  row counts from `ls -l` on both disks; they match the `dataset rows=` lines quoted in
  `status/PAPERB_TWO_CORPORA_DEFECT.md`. keep14/ShortGPT-16/freeze_front trained on the 7.57M-row
  array; keep8/keep10/keep12 on the 15.49M-row array.
* **Unequal steps**: keep14 200k, keep12 124k, keep10 83.5k, keep8 121k (per
  `status/PAPERB_TWO_CORPORA_DEFECT.md`), so depth × steps × corpus are entangled in the same table.
* **Seed unrecorded and unrecoverable** for the original runs: `--seed` did not exist until commit
  `c57c4cb`; the trainer version the originals ran against
  (`afdfa66`) never called any seeding function (`status/PAPERB_P12_SEED2.md` §1.1).
* **`--seed` does not control data order even now.** Verified live:
  `scripts/train_olmo2_arch_probe2.py:863` is `DistributedSampler(ds, shuffle=True)` with **no
  `seed=` argument**, so its private generator uses `self.seed = 0` regardless of `--seed`. `--seed`
  moves **only the fresh-tail init** (the inherited layers are transplanted bit-exactly;
  `attention_dropout: 0.0`, and `grep -c dropout` on the trainer = 0). **Any A04 "seed" arm is
  therefore fresh-block-initialisation variance, NOT training-seed variance**, unless the trainer
  is patched to pass a seed into `DistributedSampler`. This must be either fixed or disclosed.
