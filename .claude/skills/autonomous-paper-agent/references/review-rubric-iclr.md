# ICLR Review Rubric (paperC and any ICLR-template submission)

**Adopted 2026-08-16 on the owner's instruction ("把你的评分规则改为ICLR的评分规则").**
Applies to `paperC/` (verified: `paperC/main.tex` loads `iclr2026_conference`, and
`paperC/iclr2026_conference.sty` is present). `paperA/` and `paperB/` use ACL-family
templates and keep the generic rubric in `review-rubric.md`.

---

## PROVENANCE — read this before quoting any threshold as "ICLR's"

I could **not** verify the ICLR 2026 form online from this cluster. Recorded honestly:

| attempt | result |
|---|---|
| `iclr.cc/Conferences/2026/ReviewerGuide` via proxy | returned no content |
| `api2.openreview.net` invitation (carries the rating enum) | **HTTP 403** |
| `openreview.net/forum` | **HTTP 307** redirect, blocked |
| positive control (`raw.githubusercontent.com`) | **HTTP 200** — so the network works; OpenReview specifically is blocked |
| public ICLR-scraping mirrors on GitHub | 200 but neither contains the enum |

So the scale below is **the ICLR rating scale as I know it from the field, not a document I
fetched today**. Two consequences that must not be blurred:

1. **The six-point ladder `{1,3,5,6,8,10}` and `confidence 1–5` are the long-standing ICLR
   form** and are what this rubric uses. If a future agent gets network access to
   `api2.openreview.net`, re-verify and correct this file.
2. **ICLR does NOT use the NeurIPS-style Soundness / Presentation / Contribution 1–4
   triple.** I confirmed that triple belongs to a *different* venue's form by fetching
   `SakanaAI/AI-Scientist/ai_scientist/perform_review.py` (HTTP 200), which lists
   `Soundness 1-4 (poor/fair/good/excellent)` + `Presentation 1-4` + `Contribution 1-4`
   alongside `Overall 1-10` — that is the NeurIPS form. **Do not import it here.** This is
   the one thing I can state from a document I actually read: what ICLR's form is *not*.

**Rule: any claim in our records of the form "ICLR requires X" must cite either this
file's scale or a fetched document. Not memory.**

---

## Overall rating — the six allowed values

Reviewers pick **one** of these. **Intermediate values (2, 4, 7, 9) are not on the ICLR
form** and must not be emitted; 5.5-style half-scores are likewise invalid.

| value | meaning |
|---|---|
| **10** | strong accept, award quality |
| **8** | accept, good paper |
| **6** | marginally above the acceptance threshold |
| **5** | marginally below the acceptance threshold |
| **3** | reject, not good enough |
| **1** | strong reject |

**The decision boundary sits between 5 and 6.** That is the single most important property
of this scale and the reason for adopting it: it forces every reviewer to commit to a side
rather than parking on a "borderline" number. Our previous 1–10 rubric let a reviewer emit
5.5 and decline to decide — round_04 has exactly one such score.

## Confidence — 1 to 5

| value | meaning |
|---|---|
| **5** | absolutely certain; very familiar with the related work and checked the maths/details carefully |
| **4** | confident but not absolutely certain; unlikely, though conceivable, that something was missed |
| **3** | fairly confident; plausible that something was missed, or parts of the paper were not carefully checked |
| **2** | willing to defend the assessment, but likely that some central parts were not understood |
| **1** | educated guess; the work is not in the reviewer's area, or was hard to follow |

**Confidence weights adjudication, never the arithmetic.** Do not compute a
confidence-weighted mean — the ICLR form does not, and inventing one would let a
high-confidence outlier silently dominate. Use confidence only to decide *which
disagreements get an adjudicator*.

## Written sections each reviewer must produce

- **Summary** — what the paper claims, in the reviewer's own words.
- **Strengths** and **Weaknesses** — specific, with page/section/table anchors.
- **Questions** — what the authors could answer to change the score.
- Plus, kept from our own protocol because they have repeatedly caught real defects:
  - **Ceiling** — the highest rating this paper could reach if every stated weakness were
    fixed *without new experiments*. A low ceiling with a high rating is a contradiction to
    surface, not average away.
  - **What would change my score** — must be concrete and checkable.

## Dimension sub-scores

**Not part of the ICLR form.** Keep our internal 1–5 dimensions (novelty, significance,
technical soundness, experimental rigor, clarity, reproducibility, citation integrity,
limitations) as *diagnostics only*. They may inform the revision ledger. They must never be
combined into the overall rating, and they must never be reported as if ICLR asked for them.

---

## Gates under the ICLR scale

Because the values are discrete and the boundary is 5|6, the old thresholds
(`median>=7.0`, `LQ>=6.0`) are **not expressible** — 7 is not a legal value. Replacements:

| gate | threshold | rationale |
|---|---|---|
| **median rating** | **>= 6** | the median reviewer must be above the acceptance threshold |
| **lower quartile** | **>= 5** | at most a quarter of the panel may be at "reject" (3) or below |
| **no fatal** | **no rating of 1** | a strong reject is a blocking objection, not a low score |
| **min ceiling** | **>= 6** | if any reviewer thinks the paper *cannot* reach 6, that is a structural objection |
| **recommendation spread** | flag if `max - min >= 5` | e.g. one 3 and one 8 means the panel disagrees about what the paper *is*; adjudicate before revising |

**Integrity gates are unchanged and still override every score gate**: claims traceable to
registered evidence, numbers recomputable, citations verified by family (OpenReview `venueid`
for OpenReview venues; **aclanthology + DBLP for the ACL family**), compiles, no fabricated
evidence.

## Mapping the historical rounds — do NOT rescale

round_00 through round_04 were scored on the generic 1–10 rubric. **Do not convert those
numbers onto this scale.** A 5 on the old rubric meant "weak reject / negative borderline";
a 5 here means "marginally below the acceptance threshold" — close, but the old scale also
had legal values that this one does not, so any mapping would be invented precision.

`state/score_trajectory.json` already carries the rule that scores from different prompt
generations are not longitudinally comparable. This is another such boundary:

> **round_05 onward = ICLR scale. round_00–04 = generic scale. No round-over-round delta
> may cross that line.**

And round_04 carries its own separate caveat: all 12 of its reviewers read an artifact that
shipped 2 of 24 evidence files, so its median 5.0 is a **lower bound** regardless of scale.
