# B11 — K1 narrowing: the claim B11 may proceed under, and the honest sizing of what is left

**Date 2026-08-14 · 0 GPU · follows `K1_NOVELTY_CHECK.md` (verdict `NEEDS_NARROWING`).**

`K1_NOVELTY_CHECK.md` §4 wrote a narrowed claim but left it in prose; `STATUS.json`'s `claim` key still
asserted the general version, and `gpu_policy` still read "NO GPU until K1 passes". This file closes
that: it records the narrowed claim **as the operative one**, supplies the mandatory `abandoned` block
(§6.2 of the schema), and — the part K1 did not do — **states plainly how thin the survivor is** and
what would have to be true for it to be worth GPU.

---

## 1. What K1 actually did to the claim

| | text | status |
|---|---|---|
| **original** (`STATUS.json.claim`, retained verbatim, append-only) | "A generative long-context benchmark scorer's text preprocessing can encode output-format conventions strongly enough to **destroy the RANKING** of a true effect of **+70 to +84 pp**, and the failure localises to specific auditable lines of the metric." | **withdrawn as the operative claim** |
| **narrowed** | see `STATUS.json.k1_novelty.narrowed_claim`, reproduced in `K1_NOVELTY_CHECK.md` §4 | **operative** |

Three independent forces did the narrowing, and only one of them is the literature:

1. **Literature.** "Scorer/extraction preprocessing changes model ranking" is **owned**: Alzahrani et al.
   **ACL 2024** (`2024.acl-long.744`) moves leaderboard rank by up to 8 positions by changing the
   answer-selection method; Sanz-Guerrero et al. **EMNLP 2025** (`2025.emnlp-main.988`) reshuffles
   rankings via tokenization of the `"Answer:"` suffix; arXiv:2510.05152 states one can put any model in
   the lead by changing a single delimiter character. Asserting the general form walks into K1's own
   `kill_if`. **→ "destroy the RANKING" becomes "fail to recover the ORDERING of a within-model
   architectural manipulation."**
2. **Our own numbers contradicted our own claim.** `established_measurements.true_effect_on_retrieval_closed_ruler_pp`
   is `[58.0, 54.0, 84.0, 84.0]`, minimum **54**, yet the claim and `PROPOSAL.md` both said "+70". **→
   "+70 to +84" becomes "+58 to +84"** (the two cells that carry the ladder), and "+70 as a lower bound"
   is now a forbidden claim.
3. **The mechanism is a trade-off, not a bug.** Re-verified below. **→ the word "fix" is forbidden**, and
   `notrunc` may never be presented as a corrected metric.

---

## 2. CPU re-verification of the two code-level facts (executed today, independent of K1)

Both re-run against the current canonical package `third_party/babilong-pkg/babilong/metrics.py`.

### 2.1 `metrics.py:31` is dead code — **CONFIRMED, and K1's own demonstration was flawed**

```python
def preprocess_output(output):
    output = output.lower()                # line 25  <-- lowercases FIRST
    output = output.split('.')[0]          # line 27  <-- the claim's subject
    output = output.split('<context>')[0]  # line 29
    output = output.split('<example>')[0]  # line 30
    output = output.split('Question')[0]   # line 31  <-- UNREACHABLE
    return output
```

`split('Question')` can never fire: line 25 already lowercased the token to `question`.

⚠️ **K1's demonstration string does not demonstrate this.** It used
`"...kitchen Question: Where is the football? Answer: garden"`, whose leading `...` makes **line 27**
truncate at the first period and return `''`. Under that input the guard "does not fire" for the wrong
reason, and the printed output in `K1_NOVELTY_CHECK.md` §3.1 (`"...kitchen question: where is..."`) is
**not** what the function returns — it returns the empty string. The claim is still true; the evidence
offered for it was not. Corrected demonstration, no period before the token:

```
in : 'the football is in the kitchen Question: Where is Mary? Answer: garden'
out: 'the football is in the kitchen question: where is mary? answer: garden'
     -> 'question' survives in the output; the guard did NOT fire.
```

Mechanical proof, independent of any test string: `'Question' in s.lower()` is `False` for every `s`,
because `.lower()` cannot produce a capital `Q`. Contrast lines 29/30, which use the **lowercase** tags
`'<context>'` / `'<example>'` and therefore **do** fire (`preprocess_output("kitchen <CONTEXT> blah")`
→ `'kitchen '`). So the defect is specific to line 31, and it is the one guard the authors wrote against
continuation leakage.

### 2.2 The truncation is a sign-dependent trade-off — CONFIRMED

| model output | canonical (truncation on) | interpretation |
|---|---|---|
| `"Choices: A. In the kitchen B. In the garden. The answer is kitchen."` | `False` | truncation **kills** a correct list-format answer |
| `"kitchen. Question: Where is the football? Answer: garden"` | `True` | truncation **saves** a correct answer from continuation leakage |

Cell-level agreement: removing truncation raises qa1/qa2 but **lowers** qa5 (qa5×32k A0 61.0→59.0,
A3 57.0→52.0). So the operation's sign depends on the arm's output-format habit. This is scientifically
*better* than "a bug" — it explains why the dissociation is perfectly cell-aligned — but it forbids
"we fix the metric".

---

## 3. Honest sizing: how thin is the survivor?

The user's framing was "if what's left is thin, say so." It is thin. Itemised, worst-first:

| # | Weakness | Number |
|---|---|---|
| 1 | The headline inversion is **not statistically significant** | best exact McNemar **p = 0.0703** (b=1, c=7, 8 discordant items); Holm-adjusted within the 6-cell family **p = 0.4219** |
| 2 | The one-operation ablation repairs **2 of 6** cells | qa2 is unrepairable (floor, A4 = 1%) |
| 3 | The dissociation's p is **the floor of what 6 cells can produce** | Fisher exact **p = 0.0667**, i.e. descriptive, not a powered test |
| 4 | The mechanism is **NOT IDENTIFIED** | retrieval vs floor collinear: Spearman(recall, A0_acc) = **+0.714** over 6 cells |
| 5 | **One model family** | Qwen3 only; K2 unrun |
| 6 | The strongest surviving sub-result is **a 5-point ladder on a single cell** | qa1×32k, ρ = −1.000, exact permutation **p = 0.0167** — which is the *minimum attainable* p over 120 orderings, so it cannot be more significant than that no matter how real it is |

**What this adds up to.** The genuinely solid parts are the two **code facts** (§2.1, §2.2) — deterministic,
CPU-checkable in five lines, no statistics required. Everything *quantitative* about ranking damage rests
on 6 cells of one model family with a non-significant headline. That is an **appendix or a workshop-scale
measurement note**, not a paper.

### 3.1 Verdict on "is it worth doing"

**Not worth GPU on the current evidence — but there is one cheap path that is, and it is not K2.**

- **K2 (cross-family replication) is the expensive way to a likely-thin answer.** Measured cost basis
  (see `STATUS.json.gpu_cost_estimate_k2`): the A02 BABILong generations that produced B11's evidence
  cost **0.0739 GPU-h per 100-sample cell at 16k** and **0.0848 at 32k** (216 shard JSONs each, 8
  shards/cell, `.73` H20). A second family over the same 6 cells × the 2 arms that carry the contrast
  is ≈ **1.0 GPU-h**; over all 5 ladder arms ≈ **2.4 GPU-h**. So K2 is *cheap*. The problem is not cost,
  it is **expected information**: with 6 cells the best attainable Fisher p is already 0.0667, so a
  second family cannot make the dissociation significant — it can only tell us whether the *sign*
  reproduces. That is worth 1 GPU-h **only if** we are willing to publish "the sign reproduced in 2
  families, still not significant", which is close to weakness #1 restated.
- **The cheap path that dominates it: file the upstream bug report (K3's exit), 0 GPU.** §1 of
  `K1_NOVELTY_CHECK.md` establishes the exit is **unclaimed**: `booydar/babilong` has 18 issues, none
  about `metrics.py` / `preprocess_output` / first-period truncation, and a GitHub issue search returns
  0. The dead-code guard (§2.1) needs **no statistics, no GPU and no second family** — it is either
  unreachable or it isn't, and it is. This converts B11's most solid finding into its natural artefact.

**Recommendation, recorded as B11's `next_gate`:** do the 0-GPU upstream report first. Spend the ~1
GPU-h on K2 **only** if a maintainer response indicates the behaviour is intentional (which would make
the trade-off framing publishable as a measurement note) — otherwise B11's scientific content is
"a scorer line is dead code", which belongs in an issue tracker and an A02 appendix, not in a paper.

**This is a recommendation to not pursue B11 as a paper.** Recording that is a legitimate output.

---

## 4. Venue verification for this pass

Re-verified today, by family, and **not** via S2 (which 429'd):

| paper | family | authority checked today | result |
|---|---|---|---|
| Alzahrani et al., arXiv:2402.01781 | **ACL** | `aclanthology.org/2024.acl-long.744/` → HTTP 200, title *"When Benchmarks are Targets: Revealing the Sensitivity of Large Language Model Leaderboards"*, venue field **ACL** | **ACL 2024 Long** confirmed |
| Molfese et al., arXiv:2503.14996 | **ACL** | `aclanthology.org/2025.findings-acl.950/` → 200, venue field **Findings** | **Findings of ACL 2025** confirmed |
| Sanz-Guerrero et al., arXiv:2509.15020 | **ACL** | `aclanthology.org/2025.emnlp-main.988/` → 200, venue field **EMNLP** | **EMNLP 2025 Main** confirmed |
| Yu et al. xFinder, arXiv:2405.11874 | **OpenReview** | K1 recorded `venueid = ICLR.cc/2025/Conference`, forum `7UqQJUKaLM`, `Submission5699/-/Camera_Ready_Revision` | ICLR 2025 Poster (**carried from K1, not re-fetched today** — see §5) |
| Yen et al. HELMET, arXiv:2410.02694 | **OpenReview** | K1 recorded `venueid = ICLR.cc/2025/Conference`, forum `293V3bJbmE`, `Submission12024/-/Camera_Ready_Revision`. **DBLP returns only `CoRR 2024`** — the two-family rule was load-bearing | ICLR 2025 Poster (**carried from K1**) |

### ⚠️ Correction to K1's citation of P6

`K1_NOVELTY_CHECK.md` §1 lists P6 as **"Kim & Kim (et al.)"**. That attribution is **wrong**. DBLP and
the arXiv Atom record both give the authors of arXiv:2510.14773 (*Finding Answers in Thought Matters:
Revisiting Evaluation on Large Language Models with Reasoning*) as **Hwiyeol Jo, Joosung Lee, Jaehone
Lee, Sang-Woo Lee, Joonsuk Park, Kang Min Yoo** — there is no author named Kim. Venue **is** correctly
recorded (**DBLP: `CoRR` 2025, `Informal and Other Publications`**; arXiv comment says "ARR Submitted",
so no accepted venue). Corrected here rather than edited in place, since K1's file is the dated record.
**A `.bib` entry must not be generated from that row.**

Its substantive relation to B11 is unchanged and correctly stated by K1: it shows reasoning-model
*scores* are sensitive to the extraction algorithm and proposes an extra inference pass; it does not
demonstrate ranking destruction, is not long-context, and does not localise to a code line.

---

## 5. Honest gaps in this pass

- **The two OpenReview venue IDs (xFinder, HELMET) were not re-fetched today.** They are carried from
  K1's `k1_raw/` artefacts of 2026-08-14. They were verified once, by the correct family, with forum IDs
  and `Camera_Ready_Revision` invitations recorded — but "verified earlier today by the previous pass"
  is not the same as "verified by me", and I am labelling it rather than implying a fresh check.
- **K1's `p = 0.0703` / `0.4219` / `0.0167` / `+0.714` statistics were not recomputed.** I re-verified the
  two *code* facts by execution and corrected one flawed demonstration; I did **not** re-derive the
  McNemar, Holm, Fisher or Spearman values from the A02 per-item vectors. They are consistent between
  `STATUS.json` and `K1_NOVELTY_CHECK.md`, but that is internal consistency, not independent replication.
- **The upstream-issue check ("18 issues, none about metrics.py") was not re-run.** Carried from K1.
  Since the recommendation in §3.1 depends on that exit still being unclaimed, it should be re-checked
  immediately before filing.
