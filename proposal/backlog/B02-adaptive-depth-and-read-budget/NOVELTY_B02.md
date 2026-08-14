# B02 — novelty adjudication (K-NOV)

**Verdict: `CLEARED_BUT_MOOT`** — 0 GPU spent. Date 2026-08-14.
Raw search artefacts: `novelty_raw/` (arXiv Atom XML, DBLP JSON).

Two-part verdict, because the two parts point opposite ways and collapsing them would be dishonest:

1. **On novelty: no located work preempts B02's construct.** The oracle-routing literature is
   uniformly **between-model** (pick model A or model B per query). B02's action space is
   **within-model**: split depth `j`, evidence budget `k`, retrieval rounds — one model, one weight
   set. Nothing found measures per-query oracle headroom over *split depth*.
2. **On consequence: it does not matter, because B02's own kill gate fired** (`KILL_GATE_VERDICT.md`).
   The direction is closed by our own negative measurement, not by prior art. Novelty is recorded as
   adjudicated so that the closure is attributable to evidence rather than to an unrun gate.

Recording this as "cleared" rather than "unchecked" matters for a second reason: **the one thing B02
produced that is worth keeping is methodological** (the Null-B degeneracy proof), and its novelty is a
separate question from the router direction's, answered in §3.

---

## 1. Venue verification, by family

**Family rule applied as required:** OpenReview `venueid` + `Camera_Ready_Revision` for
ICLR/NeurIPS/ICML; **aclanthology + DBLP** for the ACL family including Findings. S2 was **not** used
as a venue authority (it returned HTTP 429 on this pass, and it is known to lag on recent conference
papers).

**Outcome for B02: every located neighbour is a preprint, so neither family's positive path was
needed — but the negative checks were run against both, which is what makes "preprint" a finding
rather than an assumption.**

| # | Work | Venue, and how verified | Bears on B02 how |
|---|---|---|---|
| N1 | *Opportunity Is Not Realizability: Selection-Valid Diagnostics for Multi-LLM Routing*, arXiv:2608.08265, submitted **2026-08-08** | **arXiv only.** DBLP title search → `total = 0`. OpenReview `notes/search` → 0 notes (checked both `term` and `source=forum`). Not an ACL-family paper, so aclanthology is not applicable. | **Closest work by construct.** Names exactly the two flaws B02's kill gate is built to avoid: (a) "testing against a best fixed model selected on the same examples invalidates paired inference", (b) "a full-information oracle sees outcomes no deployable router observes". Reports certified oracle gaps of 9.7–30.7 pts of which the best deployable router recovers only **7.5–14.4 %**. **Concurrent: 6 days before this gate** → per project rule, does not constitute preemption. |
| N2 | *How Much of the Routing Gap Is Real? Decomposing the Router-to-Oracle Gap…*, arXiv:2607.03436v2, submitted **2026-07-03** | **DBLP: `CoRR` 2026, type `Informal and Other Publications`** (i.e. preprint; no conference venue). | Same disease, **different medicine**. Its inflation source is **stochastic decoding** — the oracle is a single Bernoulli draw — and its fix is fresh `k ≥ 20` resampling. B02 decodes **greedily** at `k = 1`, so its inflation source is **max-over-columns**, for which the correct correction is a column-margin permutation null (our Null A), not multi-sample re-estimation. |
| N3 | *Resample or Reroute? Budget-Aware Test-Time Model Selection*, arXiv:2607.08665v2, **2026-07-09** | **arXiv only** (per its own comment/DBLP posture). | Allocates a per-query budget between resampling and rerouting. Between-model, and presupposes headroom exists; does not test whether the oracle clears a null. |
| N4 | *Statistical Scouting Finds Debate-Safe but Not Debate-Useful Cases*, arXiv:2605.09618, **2026-05-10** | **arXiv only** ("Technical report / preprint" in its own comment). | **Structurally the closest negative result.** Per-example oracle over *protocols* (greedy / vote / debate) gains +14.0 and +13.7 pp over best fixed, and it too finds the headroom **hard to recover from cheap ex-ante signals** (only a vote-entropy threshold directionally beats fixed; CIs include zero; learned LR/GBT do not). Same shape as B02 Clause 2. **But: it does not correct its oracle for max-over-noise**, so its "+14 pp headroom" is exactly the uncorrected quantity B02's Clause 1 refuses to report. |
| N5 | *Unsolvability Ceiling in Multi-LLM Routing*, arXiv:2605.07395 | **arXiv only.** | Routing-headroom estimation; attributes part of apparent unsolvability to eval artifacts. Between-model. |
| N6 | *LLMRouterBench*, arXiv:2601.07206 | **arXiv only.** | The benchmark whose 20-pt oracle gap N2 re-estimates. Between-model, `T=0.2` single draws. |

### Searches returning nothing (evidence of the gap, artefacts in `novelty_raw/`)

- `abs:"per-query" AND abs:"layer" AND abs:"depth" AND abs:"oracle" AND abs:"long context"` → **0**
- `abs:"exit layer" AND abs:"per-example" AND abs:"oracle"` → **0**
- `abs:"oracle" AND abs:"degenerate" AND abs:"null" AND abs:"binary"` → **0**
- `abs:"permutation null" AND abs:"maximum" AND abs:"bias"` → **0**
- `abs:"oracle" AND abs:"router" AND abs:"upper bound"` → **0**
- `abs:"split" AND abs:"depth" AND abs:"KV cache" AND abs:"layer"` → 1 hit, unrelated
- `abs:"curveball" AND abs:"randomization"` → 5 hits, **all ecology/bipartite-graph literature**
  (1609.05137, 1804.08487, 2112.04017, 1803.02624, 2607.29242). Curveball is imported from network
  null-model theory; **nobody has applied it to a per-item × per-config LLM outcome matrix**, which is
  why its degeneracy there was not already documented (§3).

### ⚠️ Self-citation trap avoided

`abs:"adaptive" AND abs:"retrieval budget" AND abs:"per-query"` surfaced **arXiv:2607.28263**,
*Understanding Is Done Early: A Depth Division of Labor…*, 2026-07-30 — which uses "CoMem", split
depth `j`, and Qwen3-8B. **This is our own paperA preprint, not prior art.** Confirmed via
`paperA/review_history/v12_strict_2_GPT56.md:213` ("appears to be the public preprint of this same
work… It is not independent prior art and should not be counted against novelty") and by matching
distinctive numbers. Counting it would have manufactured a false preemption of B02 by its own parent
project. Flagged here because a future automated sweep on these keywords **will** hit it again.

---

## 2. Why "no preemption" does not rescue the direction

The between-model / within-model distinction is real, and it is the axis on which B02 was novel. But it
cuts both ways: **B02's negative result is also not preempted**, and that negative result is what we
have. Specifically, N1 and N4 both find realizable routers recover *little* of a *positive* oracle
excess. B02 finds something stronger and, as far as these searches go, unreported:

> On within-model split depth, the per-item oracle sits **below** its own independence floor
> (`Δ_excess = −0.039` at 16k, `−0.032` at 32k, both `p ≤ 0.0008`, both intervals excluding 0
> negatively). The configurations are **positively coupled**: hard items are hard at every depth.
> There is no complementarity to route over — not "little", but negative.

Per project rule, "prior work exists" is not a reason to abandon; the reason B02 closes is
`KILL_GATE_VERDICT.md`. Conversely, novelty being intact is **not** a reason to keep spending: the
gate's own pre-registered decision rule says close.

---

## 3. The methodological finding, and its separate novelty status

The transferable product of B02 is not about `resume_j`. It is:

> **For a binary per-item oracle, item difficulty and oracle value are the same quantity.**
> `max_j M[i,j] = 1[rowsum_i ≥ 1]`, so the oracle is a function of the row margins alone; any
> row-margin-preserving randomisation (curveball / swap / within-row permutation) leaves it **exactly**
> invariant. Measured `sd = 2.22e-16`, `p = 1.0` — for every draw, on every dataset, forever.
> A both-margins null therefore **cannot** separate complementarity from difficulty, and pre-registering
> one as primary guarantees a `p = 1.0` that will be misread as "the kill clause fired".

Novelty status of *that*: the curveball/both-margins null is standard in ecology (N-family hits above),
and "oracle max is upward biased" is folklore that N2/N4 gesture at. **The specific statement that the
both-margins null is degenerate *by construction* for a binary max-statistic** returned 0 hits on
four separate query shapes. It is a small, sharp, checkable result.

**Honest sizing: this is a paragraph, not a paper.** It is worth ~a footnote or a methods appendix in
any write-up that reports per-item oracle headroom, and it is worth recording here so nobody in this
project re-pre-registers Null B. It is **not** grounds for promotion, and B02 is **not** promoted.

Practical carry-forward: **any future per-item oracle gate in this repo must use a column-margin
(Null A) primary on a non-binary outcome scale**, and must include Clause 2 (out-of-sample
realizability with a train-selected comparator). Both are now encoded in `KILL_GATE_VERDICT.md` §2 and
implemented in `analyze_b02_oracle.py`.
