# B02 — RELATED WORK / NOVELTY BOUNDARY (post-mortem edition)

**Written 2026-08-15. 0 GPU, 0 SSH. Adjudication + venue verification only.**

This closes the blocker `proposal/ready_queue.py:542-554` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and discharges the Related Work
item `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:92` assigns B02
(rating **不足** / insufficient; required families: *early exit/adaptive depth; RAG/search
routing; dynamic top-k/multi-hop; SLA-constrained routing; adaptive cache budget*;
safe boundary: *"必须证明联合 controller 不只是独立 router 的线性组合"* — must prove the
joint controller is not merely a linear combination of independent routers).

---

## ⚠️ 0. READ THIS FIRST — B02 IS DEAD, AND IT WAS **NOT** THE LITERATURE THAT KILLED IT

`STATUS.json:19-22`:

```json
"lifecycle": "dead",
"lifecycle_set": "2026-08-14",
"lifecycle_reason": "Own pre-registered kill gate FIRED on both lengths, both clauses.
                     Not killed by prior art; killed by our own negative measurement.
                     See KILL_GATE_VERDICT.md."
```

**This file therefore has a different job from every other `RELATED_WORK.md` in
`proposal/`.** It is not arguing that a direction is viable. It exists to do three things,
in this order:

1. **Keep the cause of death correct.** B02 died from `B02-KILL-1`, its own pre-registered
   gate, on a 6.44 GPU-h confirmatory sweep. The novelty gate came back
   **`CLEARED_BUT_MOOT`** (`NOVELTY_B02.md`, `STATUS.json:132-179`): *no located work
   preempts the construct.* Per `memory/prior-work-differentiate-dont-abandon.md`
   (user, 2026-08-07 + 2026-08-12 强化), a direction may **only** be killed by
   experimental falsification, never by a literature count. B02 is the repository's
   **positive example** of that rule working as intended, and this file must say so
   without hedging.
2. **Preserve provenance so nobody re-opens it as "unexplored".** The searches below
   returned little that overlaps — which is exactly the condition under which a future
   agent, seeing an empty collision table, concludes "nobody has done this, let's fund
   it". §4 states the precise reason that inference is wrong: **the gap is real and B02
   still has no positive result to put in it.**
3. **Discharge the audit's one substantive demand.** The audit asked for proof that a
   *joint* controller is more than a linear combination of independent routers. §3 records
   that B02 **measured the precondition for that proof and it failed**: on the fractional
   scale the per-item oracle sits *below* its own independence floor
   (`Δ_excess = −0.039` at 16k, `−0.032` at 32k, both `p = 0.0008`, both 95% intervals
   excluding 0 **negatively**). Not "no interaction" — *negative* interaction. There is
   nothing left for a joint controller to exploit on this task, so the audit's demand is
   answered in the negative and cannot be re-litigated by adding a second knob.

### 0.1 The claim that is abandoned, verbatim

`STATUS.json:187` (`abandoned.original_claim`):

> *"The optimal configuration varies per query, so a learned controller that jointly
> selects read method (raw replay / CoMem / reusable KV), split depth `j`, evidence budget
> `k`, retrieval rounds, and a low-confidence fallback will beat any single fixed
> configuration — targeting quality within 1 pp of the per-example oracle on a held-out
> task family together with a ≥20 % reduction in mean Read latency."*

**Dead.** The controller, the +20 % latency target, the ≤1 pp-to-oracle target, and any
claim that query-adaptive depth/budget selection has exploitable headroom are all
withdrawn.

### 0.2 The two things that survive, and their exact size

| # | Survivor | Size, honestly stated |
|---|---|---|
| S1 | **A positive result about *fixed* depth.** The depth curve's shape and peak transfer across context length: Spearman over the 8 `j` values, 16k vs 32k mean recall = **+0.976**, and `j = 27` is the best arm at **both** lengths. | A useful negative-space result for paperA's split-depth story. **Not a paper.** Scope: ONE task (RULER `variable_tracking`), ONE model (Qwen3-32B), ONE selector (`iter_bm25`, `topk=12`), `n = 200`/arm. |
| S2 | **A methodological note.** For a **binary** per-item oracle, `max_j M[i,j] = 1[rowsum_i ≥ 1]`, so the oracle is a function of the **row margins alone**; any row-margin-preserving randomisation (curveball / swap / within-row permutation) leaves it **exactly** invariant (measured `sd = 2.22e-16`, `p = 1.0`). A both-margins null therefore cannot separate complementarity from difficulty. | `STATUS.json:183` calls this **"a paragraph, not a paper"** and this file does not upgrade it. A footnote or methods appendix wherever per-item oracle headroom is reported. §3.4 adjudicates its novelty separately. |

Mandatory pre-committed wording on every negative below (`KILL_GATE_VERDICT.md:54-57`):
`n = 200` resolves `|Δ_excess| ≥ 0.033` at 80 % power, so an interval **containing** 0 must
be written **"no effect larger than 3.3 pp"**, never "no effect". Intervals that exclude 0
on the negative side — what actually happened — are positive findings and may be stated as
such.

---

## 1. Verification discipline, and what was reachable from this node

Venue verification is **family-split**; using the wrong authority produces false calls in
both directions:

* ICLR / NeurIPS / ICML / TMLR / COLM → **OpenReview `venueid`** (+ `Camera_Ready_Revision`
  where obtainable) — `memory/venue-verify-must-use-openreview-2026.md`
* ACL / EMNLP / NAACL / EACL **including Findings** → **ACL Anthology + DBLP**
  — `memory/venue-verify-acl-family-needs-anthology.md`
* Non-CS journals → **Crossref DOI**
* **`arXiv-only` below means "I could not verify a peer-reviewed venue from this node",
  NOT "no venue exists".**

### 1.1 Endpoint status, measured 2026-08-15 (verbatim)

| Endpoint | Status this session | Consequence |
|---|---|---|
| **DBLP** `search/publ/api?…&format=json` | ✅ works, but **intermittently returns HTTP 500 / non-JSON** under back-to-back queries (hit on 4 queries; all were re-run successfully after a 4 s delay) | primary cross-check; every `total = 0` below was re-run at least once |
| **DBLP** `rec/<key>.bib` | ✅ works | authority for `booktitle` / `series` / Findings-vs-main |
| **ACL Anthology** `https://aclanthology.org/<id>.bib` | ✅ works | authority for every ACL-family call |
| **OpenReview API v2** `api2.openreview.net/notes/search` | ✅ **works this session** — and it returns the `invitations` list, so `Camera_Ready_Revision` **was** checkable | ⚠️ note the contrast with the A01 pass, which recorded api2 as 403 across the board |
| **OpenReview API v2** `api2.openreview.net/notes?id=…` / `?forum=…` | ❌ **HTTP 403 `ChallengeRequiredError`** (`reqId 2026-08-14-9997156`) | forum-level enumeration unavailable; `notes/search` was the only route used |
| **OpenReview API v1** `api.openreview.net/notes/search` | ✅ works, but its index is **stale for 2026 venues** (returned unrelated hits for two 2026 titles that v2 resolved correctly) | not used as an authority where v2 answered |
| **arXiv API** `export.arxiv.org` (https) | ✅ works (`http://` 301-redirects) | metadata + abstracts + `comment` / `journal_ref` |
| **Semantic Scholar** graph API | ❌ **HTTP 429** | **not used as an authority anywhere** (repo rule: cross-check only) |
| **Crossref** | ✅ works | not needed for B02 (no non-CS venues in scope) |

**Consequence worth recording:** the standing note that api2 is challenge-blocked is
**path-specific, not host-specific**. `notes/search` worked; `notes?id=` did not. A future
pass should try `notes/search` before declaring OpenReview unavailable — this session
recovered two venue upgrades (§2.1, §3.1) that a blanket "api2 is down" would have missed.

---

## 2. Named closest collisions, by the audit's five families

Every row: paper → arXiv id + date → **verified venue + which authority verified it** →
what it does → the precise difference from B02. Ordered within each family by closeness.

### 2.1 Family A — early exit / adaptive depth

| # | Work | Venue (authority) | What it does | Precise difference from B02 |
|---|---|---|---|---|
| A1 | **AdaPonderLM: Gated Pondering Language Models with Token-Wise Adaptive Depth**, arXiv:2603.01914v2, 2026-03-02 | **arXiv-only.** DBLP `journals/corr/abs-2603-01914`, `CoRR 2026`, type `Informal and Other Publications`. OpenReview v2 `notes/search` → **0 matching notes** (the one hit returned was an unrelated WISE 2021 paper). | Self-supervised recurrent LM that learns **token-wise** early exit during pretraining; iteration-specific MLP gates + monotonic halting mask + KV reuse for halted tokens. Pythia 70M–2.8B. ~10 % inference compute cut at comparable PPL. | **This is the audit's named direct collision and the difference is threefold.** (i) **Granularity**: token-wise halting vs B02's **per-query** configuration choice. (ii) **Direction of the cut**: AdaPonderLM *stops computing* at the exit; B02's `j` marks a **cache boundary** and every upper layer still runs. (iii) **What is being decided**: AdaPonderLM *trains* a gate; B02's Stage 0 was to first ask whether a per-query oracle over `j` has any headroom **at all** — and it does not. AdaPonderLM never measures per-example oracle-vs-best-fixed headroom, so it neither preempts B02's construct nor rescues it. |
| A2 | **Confident Adaptive Language Modeling (CALM)**, arXiv:2207.07061 | **NeurIPS 2022.** OpenReview v2 `venue = "NeurIPS 2022 Accept"`, `venueid = NeurIPS.cc/2022/Conference`, forum `uLYc4L3C81A`; DBLP `conf/nips/SchusterFG0B0TM22`. | The canonical LLM early-exit framework: per-token confidence thresholds with calibrated guarantees on output divergence. | Peer-reviewed anchor for "adaptive depth is an established field" — **B02 may never claim adaptive depth as novel.** But CALM's decision variable is *per-token continue/stop under a global confidence calibration*; it has no per-query action space, no evidence budget, no retrieval rounds, and no oracle-headroom diagnostic. |
| A3 | **DeeBERT: Dynamic Early Exiting for Accelerating BERT Inference**, arXiv:2004.12993 | **ACL 2020 main.** Anthology `2024`→ verified as `2020.acl-main.204`, `booktitle = "Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics"`, pp. 2246–2251, DOI `10.18653/v1/2020.acl-main.204`; DBLP `conf/acl/XinTLYL20`. **Anthology ID is `acl-main`, not `findings-acl` → main conference.** | Off-ramps at intermediate layers of BERT; per-instance early exit on entropy. | Historical anchor. Encoder classification, not generation; no cache boundary; no budget knob. |

**Family A verdict: overlap on "cut the stack at depth `j`", no preemption, and no rescue.**
Nothing in this family measures whether the *best* depth varies per query in an
*exploitable* way. B02 measured it and the answer was negative.

### 2.2 Family B — RAG / search routing

| # | Work | Venue (authority) | What it does | Precise difference from B02 |
|---|---|---|---|---|
| B1 | **When Should LLMs Search? Counterfactual Supervision for Search Routing**, arXiv:2607.05752v1, 2026-07-07 | **ICML 2026 FAGEN Workshop (poster).** OpenReview v2 `venue = "FAGEN@ICML 2026 Poster"`, `venueid = ICML.cc/2026/Workshop/FAGEN`; arXiv comment: *"Accepted at the FAGEN Workshop at ICML 2026"*. DBLP still `CoRR 2026`. ⚠️ **Workshop, not ICML main — do not cite as ICML 2026.** | Instance-level search routing. Builds an oracle over `{NO SEARCH, SEARCH, UNSOLVED}` by comparing no-search vs forced-search outcomes **on the same question**, then trains SFT/preference policies; routing macro-F1 0.708 → 0.824 (Gemma E2B), 0.705 → 0.837 (Qwen3.5-4B). | **The closest work in this family and the audit's named collision.** Same *shape* as B02's Stage 0: same-item paired outcomes across configurations, an oracle, then a learned router. **Two load-bearing differences.** (i) **Action space**: theirs is an *external* binary (call the retriever or don't); B02's is **within-model** — split depth `j`, evidence budget `k`, retrieval rounds, one weight set. (ii) **The diagnostic B02 adds and they do not run**: they report the oracle and the router's F1 against it; they never test the oracle against an **independence floor**, so a positive oracle-minus-best-fixed gap is reported without asking how much of it is max-over-noise. That is exactly the correction B02's Clause 1 exists to apply — and applying it is what killed B02. **So B02's methodology is a follow-up correction to B1, not a duplicate of it.** |
| B2 | **RAGRouter-Bench: A Dataset and Benchmark for Adaptive RAG Routing**, arXiv:2602.00296v2, 2026-01-30 | **arXiv-only.** DBLP `journals/corr/abs-2602-00296`, `CoRR 2026`, `Informal and Other Publications` (a companion baseline study is also `CoRR 2026`). | First benchmark for adaptive RAG *paradigm* routing, grounded in query–corpus compatibility; three query types × corpus indicators × unified quality+cost protocol. | Audit's named collision. Routes **between RAG paradigms** (a between-system choice); B02 routes **inside one model's forward pass**. RAGRouter-Bench *presupposes* that paradigm choice is query-dependent and builds a benchmark on it; B02 *tested* that presupposition for within-model depth and it failed. |
| B3 | **Adaptive-RAG: Learning to Adapt Retrieval-Augmented LLMs through Question Complexity**, arXiv:2403.14403 | **NAACL 2024 main.** Anthology `2024.naacl-long.389.bib` → `booktitle = "Proceedings of the 2024 Conference of the North American Chapter … (Volume 1: Long Papers)"`, pp. 7036–7050, DOI `10.18653/v1/2024.naacl-long.389`; DBLP `conf/naacl/JeongBCHP24`. **`naacl-long`, not `findings-naacl` → main conference.** | Classifies query complexity, then picks no-retrieval / single-step / multi-step RAG per query. | **The peer-reviewed anchor that makes "per-query pipeline configuration" a settled idea.** B02 may not claim it. Difference: Adaptive-RAG's knobs are *retrieval-side* and it optimises end quality/cost; it never estimates a per-example oracle against a null, and its action space contains no model-internal variable. |
| B4 | **Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**, arXiv:2310.11511 | **ICLR 2024 Oral.** OpenReview v2 `venue = "ICLR 2024 oral"`, `venueid = ICLR.cc/2024/Conference`, forum `hSyW5go0v8`; DBLP `conf/iclr/AsaiWWSH24`. | On-demand retrieval + self-critique tokens; the model decides when to retrieve. | Anchor for "the model itself can gate retrieval". Learned, generative, no oracle diagnostic, no depth axis. |
| B5 | **When Should Active RAG Retrieve? A Budget-Aware Evaluation of Utility, Calibration, and Cost**, arXiv:2607.24010v1, 2026-07-27 | **KDD 2026 Workshop on Evaluation and Trustworthiness of Agentic AI** per arXiv comment. ⚠️ **Comment-self-reported; DBLP returned HTTP 500 on this query across two attempts, so this venue is NOT independently verified from this node.** Treat as `workshop, self-reported`. | Recasts active retrieval as *utility estimation*: retrieval is valuable only through its **marginal correctness change over a no-retrieval answer**. Exact top-k utility frontiers, deployable threshold frontiers, conservative budget frontiers. Names the flaw that two systems claiming the same nominal budget realise different held-out usage. | **Methodologically the closest of the whole family**, and it is **concurrent** (2026-07-27, 18 days before this pass). Same discipline as B02 Clause 2: the honest comparator must be *deployable and budget-matched*, not an oracle. Difference: retrieval-side trigger, no model-internal knob, and it does **not** test the oracle against a permutation null. **Concurrent → no preemption** per the standing rule. |

**Family B verdict: the "route per query" idea is peer-reviewed and old (B3/B4). B02 never
had novelty there and must not claim it.** B02's novelty was the **within-model** action
space (`NOVELTY_B02.md` §1) — and that survives, uselessly, because the measurement failed.

### 2.3 Family C — dynamic top-k / multi-hop evidence budget

| # | Work | Venue (authority) | What it does | Precise difference from B02 |
|---|---|---|---|---|
| C1 | **Evidence-Unit Fairness and the Limits of Query-Adaptive Sparse-Dense Fusion in Financial Document Retrieval**, arXiv:2608.00183v1, 2026-07-31 | **arXiv-only.** DBLP title search → `total = 0` (re-run once). | On FinDER: an **oracle over the fusion-weight grid shows 21.8 % headroom**, yet *none* of three lightweight adaptive routers (score-confidence heuristic, random forest over query features, ridge over query embeddings) achieves a statistically reliable gain over the fixed blend under company-grouped CV with cluster-robust inference. Also names a measurement confound (retrieval unit larger than the encoder window). | **Structurally the single closest NEGATIVE result found this pass, and it is a near-exact echo of B02's Clause 2.** Positive oracle headroom + zero realizable gain from cheap ex-ante features + a grouped/robust estimator. **Concurrent (8 days).** Difference: retrieval-fusion weight (not model depth), and — the load-bearing one — it reports the **21.8 % headroom as headroom**, without correcting the oracle for max-over-noise. B02's Clause 1 says that quantity may not be reported uncorrected, and when corrected on B02's own matrices it went **negative**. So C1 is the paper B02's method would *revise*, not the paper that scoops it. |
| C2 | **DynamicKV: Task-Aware Adaptive KV Cache Compression for Long Context LLMs**, arXiv:2412.14838v4, 2024-12-19 | **Findings of EMNLP 2025.** Anthology `2025.findings-emnlp.426.bib` → `booktitle = "Findings of the Association for Computational Linguistics: EMNLP 2025"`, pp. 8042–8057, DOI `10.18653/v1/2025.findings-emnlp.426`; DBLP `conf/emnlp/ZhouWZGLSZD25`. ⚠️ **Findings, not EMNLP main.** | Task-aware, layer-adaptive KV budget allocation. | Peer-reviewed anchor that **"budget should adapt"** is established. Adapts per **task/layer**, not per **query instance**, and never estimates a per-example oracle. |
| C3 | **Three Sides of Retrieval: Factorial Evidence for Document-Side, Query-Side, and Answer-Side Complementarity in RAG**, arXiv:2607.24781v1, 2026-06-19 | **arXiv-only.** DBLP `journals/corr/abs-2607-24781`, `CoRR 2026`. | 1,280 conditions × 8 documents; **factorial** design testing whether document-, query-, and answer-side enhancements are **complementary**; reports main effects and an interaction contrast (d = +0.32, p = 0.036), plus a 480-condition sensitivity sweep finding no significant parameter effects. | **This is the closest thing found to the audit's exact demand** ("prove the joint controller is not a linear combination of independent routers") — it is a **factorial complementarity test across pipeline knobs**. Differences: (i) its knobs are all **retrieval-side**, none model-internal; (ii) it tests complementarity in **mean effects across conditions** (a factorial ANOVA-style interaction), whereas B02's Clause 1 tests **per-item complementarity against an independence null** — a strictly stronger and different question; (iii) `n = 8` documents. **Concurrent-adjacent (2026-06-19, ~2 months).** No preemption; it is a **template B02's successor should cite** if the joint-controller question is ever re-opened on a task with non-constant features. |

### 2.4 Family D — SLA / budget-constrained routing

| # | Work | Venue (authority) | What it does | Precise difference from B02 |
|---|---|---|---|---|
| D1 | **FrugalGPT: How to Use LLMs While Reducing Cost and Improving Performance**, arXiv:2305.05176 | **TMLR 2024.** DBLP `journals/tmlr/ChenZ024`. OpenReview v2 `notes/search` → three records: `ICLR.cc/2024/Conference/Rejected_Submission`, `dblp.org/journals/CORR/2023`, and **`venue = "Accepted by TMLR"`, `venueid = TMLR`** (forum `cSimKw5p6R`). ⚠️ **It was rejected at ICLR 2024 and later accepted at TMLR — cite TMLR, never ICLR.** | LLM cascade under a cost budget; per-query escalation. | Peer-reviewed anchor for **cost-constrained per-query routing**. Between-model cascade; the budget is monetary, not latency-SLA; no oracle-vs-null diagnostic. |
| D2 | **SCOPE: Cost-Efficient Model Selection for Compound AI Systems under Quality Constraints**, arXiv:2606.00774v2, 2026-05-30 | **KDD 2026** per arXiv comment (*"Technical report for the paper accepted at KDD 2026"*). ⚠️ **Comment-self-reported.** DBLP `journals/corr/abs-2606-00774`, `CoRR 2026` only. **Treat as jref/comment-self-reported, not independently verified.** | Formalises **constrained LLM selection per module** in a compound system: pick an LLM per module to minimise average cost subject to a user quality threshold; avoids dataset-level evaluation via per-instance estimates. | **The closest work to B02's *framing*** — a **joint** assignment over multiple pipeline slots under an explicit quality constraint, which is precisely "joint controller under an SLA". Differences: (i) the decision variable is **which model** per module (between-model), not a within-model depth/budget; (ii) it is an **optimisation method** presupposing that per-module choice matters, with no complementarity null; (iii) B02's abandoned target was a *latency* reduction (≥20 %) at bounded quality loss, the mirror image of SCOPE's cost-min-under-quality. Overlap on formulation, disjoint on action space, and **no preemption** — but SCOPE is the citation that makes "joint constrained configuration selection" **not** B02's idea. |
| D3 | **RouteLLM: Learning to Route LLMs with Preference Data**, arXiv:2406.18665 | **arXiv-only.** DBLP `journals/corr/abs-2406-18665`, `CoRR 2024`. | Learned strong/weak model router from preference data with cost-quality thresholds. | Between-model; widely used baseline. No within-model knob, no oracle null. |
| D4 | **Resample or Reroute? Budget-Aware Test-Time Model Selection for LLMs**, arXiv:2607.08665v2, 2026-07-09 | **arXiv-only** (already in `NOVELTY_B02.md` as N3). | Allocates one per-query budget between **resampling the committed model** and **rerouting** to another, under an imperfect verifier; grounds the policy in a "recoverability asymmetry". | Between-model; explicitly **presupposes** headroom exists ("motivated by the reported gap between deployed routers and a per-instance oracle") rather than testing it against a null. Its own abstract concedes the oracle guarantee "holds only under an idealized oracle equipped with correctness labels and an unconstrained budget" — which is B02's Clause 2 stated as a caveat rather than as a gate. |

### 2.5 Family E — adaptive cache budget

The audit named this family; the searches found it to be **crowded but structurally
distant**. `abs:"adaptive" AND abs:"KV cache budget"` returned EvolKV (2509.08315),
Task-KV (2501.15113), RippleKV (2608.08684), DynamicKV (§C2, the one with a verified
peer-reviewed venue), ReFreeKV (2502.16886), STaR-KV (2606.01790), Crystal-KV (2601.16986),
KeepKV (2504.09936).

**Every one of them allocates a *compression* budget across layers/heads/tokens to preserve
quality at fixed memory.** None of them treats the budget as a **per-query decision
variable chosen by a controller against a quality/latency objective**, and none estimates a
per-example oracle over budgets. The family is therefore a **citation obligation**
("adaptive KV budgeting is a mature area — B02 may not claim it") and **not a collision
with B02's construct**. Only C2 is carried into the table above, as the peer-reviewed
representative.

### 2.6 The four neighbours already adjudicated in `NOVELTY_B02.md`, with one venue upgrade

`NOVELTY_B02.md` §1 (2026-08-14) already tabled N1–N6. Re-checked this session:

| # | Work | `NOVELTY_B02.md` said | **This session** |
|---|---|---|---|
| N1 | *Opportunity Is Not Realizability: Selection-Valid Diagnostics for Multi-LLM Routing*, arXiv:2608.08265, 2026-08-08 | "arXiv only. DBLP title search → `total = 0`. OpenReview `notes/search` → 0 notes." | ⚠️ **VENUE UPGRADE — the earlier call is now incomplete.** DBLP still `total = 0` (re-confirmed). But OpenReview **v2** `notes/search` on the exact title (note the missing space in the original: `"Realizability:Selection-Valid"`) returns forum `Mc3oFoxbxu`, `venue = "ACL ARR 2026 August Submission"`, `venueid = aclweb.org/ACL/ARR/2026/August/Submission`, invitations `[…/-/Submission, …/-/Edit, …/-/Preprint_Post_Submission]`, `pdate = None`. **So it is under ACL ARR review, not merely a loose preprint — but it is NOT accepted anywhere** (`pdate` null, no `Camera_Ready_Revision`, no Anthology record). Correct citation posture: **arXiv preprint, under ACL ARR 2026 August review.** Still **concurrent** (6 days before the gate) → **no preemption**, unchanged. |
| N2 | *How Much of the Routing Gap Is Real?…*, arXiv:2607.03436v2, 2026-07-03 | DBLP `CoRR 2026`, `Informal and Other Publications` | ✅ **CONFIRMED** — DBLP `journals/corr/abs-2607-03436`, `CoRR 2026`. |
| N4 | *Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases*, arXiv:2605.09618, 2026-05-10 | arXiv-only | ✅ **CONFIRMED** — DBLP `journals/corr/abs-2605-09618`, `CoRR 2026`. |
| — | *Understanding Is Done Early: A Depth Division of Labor…*, arXiv:2607.28263 | **OUR OWN paperA preprint. Not prior art.** | ✅ **CONFIRMED as a trap** — DBLP `journals/corr/abs-2607-28263`, `CoRR 2026`. It surfaces on B02-shaped keyword queries and **must never be counted against B02's novelty**; corroborated at `paperA/review_history/v12_strict_2_GPT56.md:213`. Flagged again because an automated sweep **will** hit it. |

**N1's relation to B02 is the most important one in this file and is unchanged:** N1 names
*exactly* the two flaws B02's kill gate was built to avoid — (a) a best-fixed comparator
selected on the same examples invalidates paired inference, (b) a full-information oracle
sees outcomes no deployable router observes — and certifies oracle gaps of 9.7–30.7 points
of which the best deployable router recovers only **7.5–14.4 %**. B02's result is a
**sharper** negative: their realizable share is *small*; B02's is **zero, because the input
features have no variance** (§3.2).

---

## 3. The audit's demand, answered — and answered in the negative

> Audit, line 92: *"必须证明联合 controller 不只是独立 router 的线性组合。"*
> Audit, lines 130-131: *"B02 必须先做 per-example joint oracle headroom；没有超越最佳独立
> depth/router policy 的 headroom，就不训练联合 controller。"*

**Both were executed. Both failed.** This section is the evidence, so the demand cannot be
re-opened by simply adding a knob.

### 3.1 Clause 1 — the per-item oracle is *below* its own independence floor

`evidence/b02_confirmatory_vt16k_n200.json`, `evidence/b02_confirmatory_vt32k_n200.json`;
Null A = independent column permutation, `B = 10000`, seeded PCG64, single node.

| length | scale | `oracle_obs` | `best_fixed` | raw headroom | Null-A mean | **`Δ_excess`** | CI95 | `p` |
|---|---|---:|---:|---:|---:|---:|---|---:|
| **16k** | fractional | 0.5910 | 0.3420 | +0.2490 | 0.6300 | **−0.0390** | [−0.0590, −0.0190] | **0.0008** |
| 16k | binary (secondary) | 0.3600 | 0.2750 | +0.0850 | 0.3949 | −0.0349 | [−0.0600, −0.0100] | 0.0152 |
| **32k** | fractional | 0.5430 | 0.3310 | +0.2120 | 0.5745 | **−0.0315** | [−0.0480, −0.0140] | **0.0008** |
| 32k | binary (secondary) | 0.3000 | 0.2350 | +0.0650 | 0.3263 | −0.0263 | [−0.0450, −0.0050] | 0.0224 |

All four intervals exclude 0 **on the negative side**. The raw headroom looks large
(+21 to +25 pp) and is **entirely below its own independence floor**. Reading: the
configurations are **positively coupled** — hard items are hard at every depth — which is
**strictly worse for a router than no signal at all**. 32k is an independent replication
(different length, same sign, same magnitude, same `p`).

**This is a direct negative answer to the audit's "linear combination" question**, and it is
stronger than the question asked: a joint controller cannot beat independent routers when
the per-item oracle cannot even beat *independence*.

### 3.2 Clause 2 — the headroom is not reachable, and the features are constants

`evidence/b02_realizability_leg.json` (0 GPU). 300 random 50/50 splits; per-bucket best `j`
learned on train, applied to held-out test; comparator fixed `j` **also** train-selected.

| length | router | held-out gain | CI95 | beats fixed? |
|---|---|---:|---|---|
| 16k | length, 2 buckets | −0.0143 | [−0.0720, +0.0000] | **no** |
| 16k | length, 4 buckets | −0.0250 | [−0.1121, +0.0020] | **no** |
| 32k | length, 2 buckets | −0.0075 | [−0.0840, +0.0000] | **no** |
| 32k | length, 4 buckets | −0.0140 | [−0.0860, +0.0080] | **no** |

**And the structural finding is more decisive than the intervals.** Of the seven input
features `PROPOSAL.md:30-38` lists:

* **`document length`** — RULER pads every item to the target length. Measured span across
  200 items: **10 tokens out of 32 713–32 723**, `sd = 1.96`. A feature with no variance.
* **`estimated evidence count`** — **constant at 9** for every `variable_tracking` item. A
  router keyed on it is *identically* the best fixed config.
* **`BM25/dense score gap`, `retrieval entropy`** — not emitted per item by the harness;
  untestable without new instrumentation.

So Clause 2 **could not have been passed on `variable_tracking` by any amount of GPU**, and
B02's Stage 0 as written was never decidable on that task alone. That is a design defect
this file records honestly rather than as bad luck.

### 3.3 The one positive that came out of it

Spearman over the 8 `j` values between 16k and 32k mean recall = **+0.976**; `j = 27` is
best at **both** lengths. Combined with the negative `Δ_excess`:

> On this task, **one well-chosen fixed split depth is the right answer**, and it transfers
> across context length. Per-query depth adaptation has no complementarity to exploit.

Cost that bought it: **6.438 GPU-h** measured (16 cells, `elapsed_seconds` summed on `.73`,
8×H20 sm_90, `oom_count = 0` in all 16), + 0.32 GPU-h pilot = **6.76 GPU-h total**.
Wall clock 48.3 min vs 48 min projected — the projection was accurate to ~1 %.
**Remaining authorised GPU for B02 as a router direction: 0.**

### 3.4 Novelty of the methodological survivor (S2), adjudicated separately

The transferable product is: **for a binary per-item oracle, item difficulty and oracle
value are the same quantity**, so any row-margin-preserving null is *exactly* invariant and
a both-margins (curveball) null returns `p = 1.0` forever — which will be misread as "the
kill clause fired".

Novelty status: the curveball / both-margins null is **standard in ecology**
(`abs:"curveball" AND abs:"randomization"` → 5 hits, **all** ecology / bipartite-graph
null-model literature: 1609.05137, 1804.08487, 2112.04017, 1803.02624, 2607.29242), and
"the oracle max is upward biased" is **folklore that N2 and N4 gesture at**. The specific
statement that the both-margins null is degenerate **by construction** for a binary
max-statistic returned **0 hits** on four separate query shapes.

**`STATUS.json:183` sizes this as "a paragraph, not a paper" and this file does not upgrade
it.** It is worth a footnote or methods appendix wherever per-item oracle headroom is
reported, and it is worth recording so nobody in this repo re-pre-registers Null B.
**It is not grounds for promotion. B02 is not promoted.**

Binding carry-forward (`STATUS.json:184`): **any future per-item oracle gate in this repo
MUST use a column-margin (Null A) primary on a NON-BINARY outcome scale, and MUST include a
Clause-2 out-of-sample realizability leg with a TRAIN-SELECTED comparator.**

---

## 4. Safe residual gap — and why it does not resurrect B02

**The gap, stated as one falsifiable sentence:**

> **The oracle-routing literature is uniformly *between-model* (pick model A or B, or
> pipeline paradigm A or B, per query); no located work measures per-example oracle headroom
> over a *within-model* configuration axis — split depth `j`, evidence budget `k`, retrieval
> rounds on one weight set — against an independence null.**

**How it would be falsified:** exhibit a paper that (a) varies a model-internal inference
configuration per query on a single weight set, (b) computes a per-example oracle over those
configurations on a *fixed* item set, and (c) tests that oracle against a null that destroys
item×config coupling. Four query shapes returned **0 hits**
(`NOVELTY_B02.md` §1, artefacts in `novelty_raw/`):
`abs:"per-query" AND abs:"layer" AND abs:"depth" AND abs:"oracle" AND abs:"long context"`;
`abs:"exit layer" AND abs:"per-example" AND abs:"oracle"`;
`abs:"oracle" AND abs:"degenerate" AND abs:"null" AND abs:"binary"`;
`abs:"permutation null" AND abs:"maximum" AND abs:"bias"`;
`abs:"oracle" AND abs:"router" AND abs:"upper bound"`. Added this session, all also
returning nothing on-target: `abs:"item difficulty" AND abs:"oracle" AND abs:"selection"
AND abs:"LLM"` → **0**; `abs:"joint" AND abs:"controller" AND abs:"inference" AND
abs:"configuration"` → 8 hits, **all** unrelated (robotics, wireless, normalizing flows);
`abs:"interaction" AND abs:"factorial" AND abs:"inference" AND abs:"LLM"` → 4 hits, only
C3 on-topic.

### ⚠️ 4.1 THE GAP IS REAL AND B02 IS STILL DEAD

**This is the sentence this file exists to make unmissable.**

An empty collision table is **not** an argument to fund a direction. B02's novelty gate
returned **`CLEARED_BUT_MOOT`**: the construct is unoccupied *and* the direction is closed,
because **the two facts are independent**. Specifically:

* **Novelty intact ≠ headroom exists.** The reason nobody has published a within-model
  depth oracle might simply be that there is nothing there. B02 spent 6.76 GPU-h finding
  out, and on `variable_tracking` × Qwen3-32B there is nothing there — `Δ_excess` is
  **negative** at both lengths.
* **Per `memory/prior-work-differentiate-dont-abandon.md`, literature can never kill a
  direction — but novelty can never *save* one either.** The gate's own pre-registered
  decision rule says close. That rule is symmetric or it is worthless.
* **Two of B02's own seven input features are literally constants on its own task.** Any
  resurrection must first exhibit a task where they are not (`STATUS.json:189`:
  *"re-opening it requires a task where the input features are not constants"*).

**Correctly scoped negative (pre-committed wording):** *"no exploitable complementarity
larger than 3.3 pp on this task"* — ONE task (RULER `variable_tracking`), ONE model
(Qwen3-32B), ONE selector (`iter_bm25`, `topk=12`), `n = 200`/arm, greedy `k = 1`,
`chat_template = False`. **Not** "no complementarity anywhere."

### 4.2 What a successor would have to do (recorded, not authorised)

If anyone re-opens within-model configuration routing, the following are now **preconditions**,
each traceable to a measured failure above:

1. A task with **genuinely variable evidence count and document length** (B02's were
   constants — §3.2).
2. Harness instrumentation that **emits retrieval-side features per item** (BM25/dense score
   gap, retrieval entropy were not emitted — §3.2).
3. Clause 1 on a **non-binary** outcome scale with a **column-margin** null (§3.4).
4. Clause 2 with a **train-selected** comparator (the selection-validity flaw named by
   N1/arXiv:2608.08265).
5. A **factorial or per-item interaction test** if the claim is that a *joint* controller
   beats independent ones — C3 (arXiv:2607.24781) is the closest available template, and
   `Δ_excess` against an independence null is the stronger version.

---

## 5. MUST-NOT-CLAIM list (binding on any B02 writeup, including paperA footnotes)

1. ❌ **First to do adaptive depth / early exit in LLMs.** **CALM, NeurIPS 2022**
   (OpenReview `venueid = NeurIPS.cc/2022/Conference`) and **DeeBERT, ACL 2020 main**
   (Anthology `2020.acl-main.204`) own it. Token-wise adaptive depth in a modern LM is
   **AdaPonderLM, arXiv:2603.01914** (arXiv-only).
2. ❌ **First to route per query in a RAG pipeline.** **Adaptive-RAG, NAACL 2024 main**
   (Anthology `2024.naacl-long.389`) and **Self-RAG, ICLR 2024 Oral** (OpenReview
   `venueid = ICLR.cc/2024/Conference`) own it.
3. ❌ **First to adapt an evidence / KV budget.** **DynamicKV, Findings of EMNLP 2025**
   (Anthology `2025.findings-emnlp.426`) owns it, plus the eight-paper adaptive-KV family
   in §2.5. **Do not cite DynamicKV as EMNLP main.**
4. ❌ **First to formulate joint constrained configuration selection under a quality/cost
   constraint.** **FrugalGPT, TMLR 2024** (DBLP `journals/tmlr/ChenZ024`) and **SCOPE,
   arXiv:2606.00774 (KDD 2026, comment-self-reported)** own it. **Do not cite FrugalGPT as
   ICLR — it was rejected there.**
5. ❌ **That an oracle-minus-best-fixed gap is "headroom".** **N1/arXiv:2608.08265**
   (under ACL ARR 2026 August review) names both flaws; **N2/arXiv:2607.03436** names the
   stochastic-decoding component; **C1/arXiv:2608.00183** and **N4/arXiv:2605.09618** both
   exhibit large positive oracle gaps with ~zero realizable gain. B02 may claim only that
   it **corrected** this quantity, and that the correction was **negative**.
6. ❌ **That the curveball / both-margins null is a novel construction.** It is standard
   ecology / bipartite-graph null-model theory (1609.05137, 1804.08487, 2112.04017,
   1803.02624, 2607.29242). B02's residual is only the **degeneracy proof for a binary
   max-statistic**, and only at "0 hits on four query shapes" strength.
7. ❌ **That per-query depth adaptation is promising.** B02's own gate says the opposite on
   the one task it measured. Any paperA text touching split depth must present the
   **fixed-depth** result (S1) and cite the negative `Δ_excess`.
8. ❌ **`arXiv:2607.28263` as prior art.** It is **our own paperA preprint**.
9. ❌ **Any claim resting on the T21 sweep** (`resume_j` × {16k,32k}, `n = 50`/cell).
   `STATUS.json:6-13`: its sample sets are **disjoint across configs** (`j3 ∩ j6 = 0/50`,
   … `j3 ∩ j48 = 0/50`), so no per-example oracle is computable from it. Only the `n = 200`
   fixed-sample confirmatory sweep is admissible.

---

## 6. Honest gaps in this adjudication

1. ⚠️ **Semantic Scholar returned HTTP 429 for the entire session.** Not used as an
   authority anywhere (repo rule: cross-check only). A paper indexed *only* on S2's private
   index could have been missed — low probability for this topic, which is arXiv-native.
2. ⚠️ **`api2.openreview.net/notes?id=` and `?forum=` are HTTP 403 `ChallengeRequiredError`**
   (`reqId 2026-08-14-9997156`). `notes/search` **did** work and **does** return the
   `invitations` list, so `Camera_Ready_Revision` was checkable for the papers that have one.
   **No forum-level enumeration was performed**, so a paper whose `notes/search` title match
   fails (e.g. because of a typo in its own title — which is exactly what happened to N1,
   `"Realizability:Selection-Valid"` with no space) could be missed. **This bit me once this
   session and it will bite again.**
3. ⚠️ **DBLP intermittently returned HTTP 500 / non-JSON.** Four queries failed
   (`When Should Active RAG Retrieve`, `Ghosted Layers…` first attempt, `Shortened LLaMA…`,
   `Mixture-of-Depths…`). Three were recovered on retry with a 4 s delay; **`When Should
   Active RAG Retrieve` (B5) never resolved**, so its KDD-2026-workshop venue is
   **comment-self-reported only**. **Mixture-of-Depths (Raposo et al. 2024) was never
   resolved on either DBLP or OpenReview and is therefore NOT cited above** — it is a
   well-known adaptive-depth paper and its absence from §2.1 is a gap in this table, not
   evidence it does not exist.
4. ⚠️ **Two venues rest on the papers' own arXiv `comment` field**: B5 (KDD 2026 workshop)
   and D2/SCOPE (KDD 2026). Neither has a DBLP conference record or an OpenReview
   `venueid`. **Do not enter either into a `.bib` as a KDD paper until independently
   verified.**
5. ⚠️ **B1's venue is a WORKSHOP, not ICML main.** OpenReview `venueid =
   ICML.cc/2026/Workshop/FAGEN`, `venue = "FAGEN@ICML 2026 Poster"`. Miscited as "ICML 2026"
   it would look like a much stronger preemption threat than it is.
6. **No full-text PDF was read for any §2 paper.** All overlap judgements are from
   **abstract + venue metadata** (arXiv API `summary`, `comment`, `journal_ref`). The one
   exception is B02's own artefacts, which were read on disk.
7. **`arXiv-only` in every row means "no peer-reviewed venue verifiable from this node",
   not "no venue exists".** Eight of the rows above are `arXiv-only`: A1, B2, C1, D3, D4,
   N1 (upgraded to *under ARR review*), N2, N4.
8. **No `.bib` entries were emitted.** Per `memory/venue-verify-acl-family-needs-anthology.md`
   and `memory/tcodex-exec-no-dash-c-flag.md`, entries must not enter a bibliography until
   venue-verified by family. Safe to add now: A2 (NeurIPS 2022), A3 (ACL 2020 main),
   B3 (NAACL 2024 main), B4 (ICLR 2024 Oral), C2 (Findings EMNLP 2025), D1 (TMLR 2024).
   **Not safe**: B5, D2 (comment-only) and every `arXiv-only` row.
9. **Zero cross-disk verification.** Every path cited here is on **wzc1**. B02's 48 cell
   JSONs were copied from `.73`/zwfy6 to wzc1 precisely so they stay recomputable
   (`STATUS.json:129`), but the zwfy6 originals were **not** `ls`-confirmed this session.
   Per `memory/two-disk-rule-applies-to-main-too.md` that is "unverified from here", not
   "gone" — and **nothing above claims any such absence**.
10. **`PROPOSAL.md` was not edited.** It still states Stage 0 in the undecidable form
    (*"若 oracle 相对最佳 fixed config 的收益不足，方向关闭"*) that `FIXED_SAMPLE_PROTOCOL.md`
    §5 and `KILL_GATE_VERDICT.md` §1 both refute, and it still lists the seven input
    features two of which are constants. It is left as the **dated record of what was
    proposed**; `STATUS.json` + `KILL_GATE_VERDICT.md` are the authority on what happened.

---

## 7. Verdict

```
related_work_status:  audited
lifecycle:            dead  (UNCHANGED -- this file does not resurrect anything)
cause_of_death:       B02's OWN pre-registered kill gate B02-KILL-1, fired on both
                      clauses at both lengths (n=200, 6.44 GPU-h measured on .73).
                      NOT prior art. The novelty gate returned CLEARED_BUT_MOOT.
novelty_status:       CLEARED_BUT_MOOT -- no located work preempts the within-model
                      configuration-oracle construct; the direction closes on its own
                      negative measurement
strongest_collision:  When Should LLMs Search? Counterfactual Supervision for Search
                      Routing, arXiv:2607.05752 (ICML 2026 FAGEN *Workshop*, OpenReview
                      venueid=ICML.cc/2026/Workshop/FAGEN) -- same same-item paired-
                      oracle-then-router shape, but its action space is EXTERNAL
                      (call the retriever or not) vs B02's WITHIN-MODEL (split depth j),
                      and it never tests its oracle against an independence floor.
                      => differentiation + follow-up correction, NOT preemption.
runner_up_collision:  Evidence-Unit Fairness and the Limits of Query-Adaptive Sparse-
                      Dense Fusion, arXiv:2608.00183 (arXiv-only, CONCURRENT, 8 days) --
                      +21.8% oracle headroom, zero realizable router gain. Same negative
                      shape as B02 Clause 2, but reports the headroom UNCORRECTED.
audit_demand:         ANSWERED IN THE NEGATIVE. Per-example joint oracle headroom was
                      measured and sits BELOW its independence floor at both lengths
                      (Delta_excess -0.039 / -0.032, p=0.0008). A joint controller cannot
                      beat independent routers where the oracle cannot beat independence.
promotion:            NO. Not promotable, not resurrectable without a task on which
                      B02's own input features are not constants.
```

No candidate is 完全相同/抄袭. Every collision differs on at least one load-bearing axis:
**within-model vs between-model action space**, **per-item oracle vs mean-effect
factorial**, **null-corrected vs uncorrected headroom**, **cache boundary vs computation
skip**, or **method vs diagnostic**. Per
`memory/prior-work-differentiate-dont-abandon.md` the correct output of this pass is a
**citation-obligation list plus a differentiation map** — which §2 and §5 are — and
**explicitly not** a claim that the literature closed B02.

**B02 was killed by its own measurement. That is the finding, and it is the right way for a
direction to die.**
