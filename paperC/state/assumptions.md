# paperC — assumptions register

Every entry is something the paper or this process **relies on but has not proved**. Each
carries what would falsify it and who can settle it. Created 2026-08-15 on resuming the
`autonomous-paper-agent` skill at round 3.

Rule for this file: an assumption that is later measured moves to `state/decision_log.md`
or an `evidence/*.json` file with its measurement. It does **not** get silently deleted.

---

## A-01 — ICLR 2026 main-text page limit is 9 pages ⚠️ UNVERIFIED, ACTIVELY BEING CHECKED

**Relied on by:** the venue-compliance gate and the size of the relocation fix (D-004).

**Status:** the number 9 is my **prior knowledge, not a measurement**. What I actually
verified:
- `iclr2026_conference.sty` contains **no** page-limit text (0 hits for `page limit`,
  `ethics`, `nine pages`, `9 pages`).
- `https://iclr.cc/Conferences/2026/CallForPapers` returns **HTTP 200 but a 3,835-character
  stub**: 0 hits for `limit`, `appendix`, `reference`, `supplement`. It only links onward.
- `WebFetch` refuses the `iclr.cc` domain outright in this environment.

**Falsified by:** any citable CFP/author-guide sentence giving a different number, or
showing that the Ethics and Reproducibility statements are excluded (which would shrink the
measured 2-page overage).

**Being settled by:** a delegated lookup writing `gate/venue_page_limit.json` with a
**mandatory verbatim quote**; it is instructed to return `status: "UNVERIFIED"` rather than
fill in an inferred number. Until that file exists with a quote, **the overage size is not
established** and no content should be relocated on the strength of it.

---

## A-02 — The 11 previously-verified citation venue strings are still correct

**Relied on by:** the citation-integrity gate.

**Status:** `gate/venue_verification_openreview.json` (6 keys) and
`gate/venue_verification_acl.json` (5 keys) were produced in an **earlier pass** and were
**not re-verified this round**. Venue metadata is mutable — a paper can move from
"submitted" to camera-ready, and DBLP/S2 are known to lag behind OpenReview for 2026
conference papers.

**Falsified by:** re-querying and finding a changed `venueid` or a withdrawn acceptance.

**Note on method, which must not be collapsed:** the two families need *different*
authorities. OpenReview family (ICLR/NeurIPS/ICML) = `venueid` +
`Camera_Ready_Revision`. ACL family **including Findings** = ACL Anthology + DBLP, **not**
OpenReview. Generalising the OpenReview rule to ACL has previously caused Findings papers
to be misreported as preprints.

---

## A-03 — The 6 classical statistics citations support the sentences that cite them

**Relied on by:** the paper's attribution of the best-constant / formula-scoring idea to
prior psychometrics.

**Status:** **UNVERIFIED on both axes** (metadata and local support). These 6 —
`bennett1954communications`, `brennan1981kappa`, `brenner1996weightedkappa`,
`cohen1960kappa`, `devries2008pooledkappa`, `frary1988formula` — are pre-digital-era works
that live in neither OpenReview nor the ACL Anthology, so **neither** existing family file
covers them. Their absence from those files is correct, not an oversight.

**Why this is the highest-risk assumption in the file:** paperC's contribution *is* a
measurement rule resting on the psychometric notion of a best-constant baseline. These
citations are precisely where a reviewer will probe whether a prior idea has been correctly
attributed — or whether the paper is claiming as novel something Frary (1988) already said.
An unverified *local-support* claim here is materially more dangerous than an unverified
conference venue string.

**Falsified by:** reading the actual sources and finding they do not support the local
sentence, or that one of them anticipates a claim the paper presents as its own.

---

## A-04 — The 14 "derived as difference" numbers are genuine derivations

**Relied on by:** the numbers gate verdict (`gate/numbers_check.json`, PASS at
`--max-unmatched 0`: 610 scanned, 468 direct, 128 correctly rounded, 14 derived).

**Status:** `check_numbers.py`'s difference-derivation is documented by its own author as a
**heuristic**, with a recorded negative test in which `12.3456` was matched by coincidence
as the difference of two unrelated evidence numbers. That is why the script counts
`derived_as_difference` **separately** from `direct_match` rather than merging them.

**Falsified by:** inspecting each of the 14 and finding one whose "derivation" is a
numerical coincidence rather than the arithmetic the sentence claims.

**Consequence if false:** at most 14 of 610 numbers (2.3%) lose their binding. It would not
overturn the gate, but each affected sentence would need a direct evidence anchor.

---

## A-05 — Relocating unreferenced tables to the appendix costs no scientific content

**Relied on by:** D-004's claim that the page fix is "strictly an improvement".

**Status:** believed on structural grounds — 8 of 11 tables are never `\ref`'d, so no prose
sentence currently depends on the reader seeing them at that position. Not yet tested by
rebuilding.

**Falsified by:** a reviewer identifying a relocated table as load-bearing for a main-text
claim, or a rebuild showing the main text still over the limit after relocation (which
would mean the prose, not the floats, is the binding constraint).

---

## A-06 — Internal panel scores do not predict venue acceptance

**Status:** treated as **true by construction** and stated in every deliverable. The 1–10
overall scale and ARR 1–5 scale are optimisation signals for finding defects, nothing more.

**Also assumed:** scores are **not comparable across prompt generations**. Round 3 uses
round_01's design (one defect per reviewer, numbers inlined); round_00 used a broad
"find problems" form and lost 3 of 6 reviewers to it. Comparing a round-3 median against a
round-0 median would be comparing two different instruments — and round_00 has **no median
at all**, because it aborted.

---

## A-07 — ~~The frozen round_03 snapshot is genuinely blind~~ **FALSIFIED for rounds 00–02, 2026-08-16**

> **⛔ RETRACTED AS A CLASS CLAIM.** The narrow statement about round_03 is true; the general
> statement it made about `freeze_round.py`'s behaviour is **false**, and rounds 00/01/02 shipped
> a real blindness breach. Full adjudication:
> `paperC/evidence/BLINDNESS_BREACH_ROUNDS_00_02_MAIN.json`.
>
> **What happened.** `evidence/EVIDENCE_PACK.md` (md5 `4d0013b52eed`, 77766 B) is present in the
> round_00, round_01 and round_02 submission trees, **byte-identical to the author-side
> `tcodex_out/EVIDENCE_PACK.md`**. It contains writer-steering imperatives, e.g.
> `DO NOT claim differential learning rates on either side` (line 649),
> `attacking it would be an easy referee kill` (line 690),
> `the corresponding limitation L1 in the writer prompt` (line 758).
> **Scores from rounds 00–02 are compromised** and must not enter any median, ceiling, trajectory
> or round-over-round improvement claim. `state/score_trajectory.json` is annotated accordingly.
>
> **Why this check missed it — three distinct failures, all worth remembering:**
>
> 1. **Wrong vocabulary family.** The eight grep terms are all *review-process* words. The lines
>    that actually leaked contain none of them — line 649 has zero hits for `reviewer` and zero
>    for `verdict`. A leak that **steers the author** is invisible to a search for leaks that
>    **reveal the panel**. Worse, the terms *did* hit (16× `verdict`, 2× `reviewer`) and every hit
>    was dismissed as the paper's own vocabulary, which for those hits was correct — so the check
>    produced a green result *by correctly explaining away the wrong hits*.
> 2. **Verified on the one instance that was vacuously safe.** round_03 packaged **zero** evidence
>    files, so the vector did not exist there. The grep was true for round_03 and then generalised
>    into a claim about the script's behaviour, which was already false three rounds over.
> 3. **The exclusion rule existed and was not enforced on the path that mattered.** `tcodex_out/`
>    *was* on the list; v1 applied the list only to the manuscript closure, and the `--evidence`
>    loop bypassed it.
>
> **The "residual risk" paragraph below was backwards.** It correctly identified the exact file
> that mattered, then reasoned from an unverified premise — that a reviewer "cannot read it (it is
> excluded)". A reviewer *could* read it, because it was shipped.

**Original text, preserved for provenance:**

**Status:** **verified, not assumed.** I grepped the frozen tree for `reviewer`, `round_0`,
`verdict`, `strict_[0-9]`, `meta.review`, `target.median`, `overall_score`, `panel median`.
Result: zero leaks. Every `verdict` hit is the **paper's own vocabulary** — its result
tables literally carry a "Verdict" column. `freeze_round.py` additionally excludes
`review_rounds/`, `review_history/`, `tcodex_out/`, `SCORE_HISTORY`, `review_prompts/`,
`WRITER_NOTES` by rule.

**Residual risk:** the manuscript cites `paperC/tcodex_out/EVIDENCE_PACK.md` by path in a
caption. A reviewer cannot read it (it is excluded), so this is a dangling internal
reference in the submitted artifact — a presentation defect worth fixing, not a blindness
breach.
