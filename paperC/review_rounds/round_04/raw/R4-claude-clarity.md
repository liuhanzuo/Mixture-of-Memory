# Review — R4-claude-clarity (round 4)

- **Reviewer**: `R4-claude-clarity`
- **Primary lens**: clarity and presentation
- **Snapshot sha256**: `7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a`
- **Overall**: 5 (weak reject / negative borderline)
- **Confidence**: 4

---

## 1. What the paper says it does

The submission proposes **null calibration** as a reporting protocol for multiple-choice LM
evaluation. The operational claim is small and clean: before you compare two model arms on an MC
construct, test the reported score against the construct's *best constant, input-blind predictor*
(the empirical gold-label marginal maximum for letter constructs; a longest-option heuristic for
content-likelihood constructs). The paper then argues four things:

1. That floor is not nominal chance (BoolQ 0.6217 vs 0.50; MMLU-Pro 0.116606 vs 0.100000), and the
   floor estimator is *itself* biased upward because it is a max over `k` noisy marginals — so the
   floor needs its own calibration. After that calibration only 3 of 8 letter constructs survive.
2. Content-side floors are under-specified by four independent degrees of freedom (tie convention,
   length unit, tokenizer, and the definition of "chance" under variable `n_opt`).
3. Choosing the wrong reference changes published verdicts on structurally damaged arms — headline
   14/15 at-or-below-floor on MMLU-Pro; under a *symmetric* evidentiary standard 3/12 vs 1/12.
4. A second, arm-conditional permutation null (read-out v2, the numerator of Cohen's kappa
   stratified by `n_opt`) is needed for a different question, and it re-sorts 27 previously scored
   cells in both directions.

I judged the whole paper on all rubric dimensions but read hardest for whether a competent reader
could reconstruct this argument from the text alone.

## 2. What I verified, and what I could not

Verified by direct computation from the shipped tables:

- Table 1 / Table 10 gap columns: every printed `Gap (pp)` equals `100*(Floor - Chance)` to the
  printed precision. Two rows round differently between the hand-authored Table 1 and the generated
  Table 10 (CommonsenseQA `+0.885` vs `+0.884`; BoolQ `+12.17` vs `+12.171`) — the exact values are
  0.8845 and 12.1713, so both roundings are defensible but the pair is inconsistent.
- Table 6's `+0.324` for `olmo2/keep14` reproduces from Table 7's accuracy 0.119847 against the
  stated 0.116606 floor. Good.
- The `36 items = 0.299 pp` ordering-slack figure reproduces: `(1439-1403)/12032 = 0.2992%`.
- The main text carries **zero** numbered tables and **zero** figures across all nine main-text
  pages, and makes **11 pointers into the appendix** across 9 distinct appendix subsections.
- The rendered PDF is 24 pages (main text ends p9 at REFERENCES; appendix runs p11–p24).

Verified against external sources (proxy positive control passed: Crossref returned a 200 with
1913 bytes for `10.1086/266520`):

- `arcon2026metalinguistic` (arXiv 2602.02182): abstract explicitly says "Although all models
  perform above chance, they fail to outperform the majority-class baseline." The opening anecdote
  is accurate.
- `bennett1954communications`, `brennan1981kappa`, `brenner1996weightedkappa`, `frary1988formula`,
  `devries2008pooledkappa`: all five resolve via Crossref with exactly the titles, authors,
  volumes, issues, pages and years in `refs.bib`. This bibliography is unusually clean.
- `gu2025olmes` is genuinely Findings of NAACL 2025 (aclanthology `2025.findings-naacl.282`), and
  the arXiv v2 full text confirms both the max-of-MCF/CF rule ("The 'max' average corresponds to
  the OLMES score, taking the best of MCF and CF for each task") and that its reference framing is
  a *random baseline*, not a label-marginal floor. §2's characterisation is fair.
- **`bean2025measuring` does not support the sentence attached to it.** The paper's intro says the
  review "offers 27 actionable checklist items." I fetched the arXiv HTML (2511.04703v1) and counted
  the checklist: **28 `\square` items** across 8 recommendation groups. The abstract advertises
  "eight key recommendations"; the numbers 21 and 30 that do appear in the paper are the codebook
  question count and the double-annotation codebook category count, not checklist items. The
  *substantive* half of the claim — that no checklist item asks for a null / chance level /
  constant predictor — I confirmed by reading all 28 items; none does. So the load-bearing gap
  claim survives, but the count "27" is wrong.

Could not verify: any per-item evidence. The snapshot ships only `build_record.json` and
`claim_evidence_map.tsv`; every `E-*` identifier resolves to a path (`tcodex_out/EVIDENCE_PACK.md`,
`evidence/heal_readout_v2_permutation_null.json`, …) that is **not in the snapshot**. I therefore
cannot check the bootstrap half-widths, the permutation decisions, the 200,000-draw balanced-null
`p`-values, the kappa-identity verification "on all 27 cells", or the claimed "610 numerals with
none unresolved". Those all go in review limitations, not into either column.

## 3. Strengths

1. **The scoping discipline is real and unusual.** The paper does not just add a caveat; it moves
   its own headline. Table 1's caption states outright that only 3 of 8 constructs clear the
   winner's-curse calibration, that CommonsenseQA and PIQA are in fact *below* `E[f̂]`, and that
   the remaining five rows support only the weaker claim. The abstract carries this narrowing
   rather than hiding it. §5.1's "Both sides of the flip, held to one standard" paragraph
   voluntarily reduces the headline flip from 10/12-vs-1/12 to 3/12-vs-1/12 and then reports that
   under BH or Bonferroni *neither* side retains a cell. Very few submissions volunteer the number
   that shrinks their own effect by 3x.
2. **The v1/v2 separation is genuinely clarifying.** The observation that an arm-independent
   best-constant floor cannot be a competence statistic (always-A / always-E / always-J score
   0.000 / -2.111 / -3.807 pp while carrying identical information) is the sharpest single idea in
   the paper, and it motivates v2 non-circularly.
3. **Novelty boundaries are drawn where they belong.** §3.2 states that Δ_perm *is* the numerator
   of Cohen's kappa, that the constant-collapse zero is the textbook property, and that
   option-count-aware chance correction is decades old — with five correctly-cited primary sources.
   The claim is reduced to (i) varying-`k` stratification and (ii) use as a pre-comparison gate.
   That is the right size of claim and it is stated precisely.
4. **The v1/v2 ordering is not oversold.** Appendix A.3 refuses the obvious identity claim, gives
   the exact counterexample (the `n_opt`-conditional emitter A,A,B,E,B,E,A,A attaining
   `f_const + 0.299` pp), explains why "27/27 hold" overstates the evidence (13 cells could not have
   violated it under any prediction vector), and reduces the binding evidence to one cell's
   0.085 pp attainable violation. The bolded instruction "Passing v2 must never be described as
   clearing the paper's floor" is exactly the sentence a careful reader needs.
5. **Table 11's retraction ledger and Table 9's integrity table** are honest and legible artifacts.
   Row 16's explicit "do not resurrect 4.8x, 0.2822, 58/91, …" is a form of provenance hygiene I
   have not seen in a submission before.

## 4. Issues

### C-01 (major, clarity/presentation) — the main text does not stand alone

The nine main-text pages contain **no table and no figure**. Every quantitative artifact the
argument turns on lives after the bibliography: Table 1 (construct floors, p13), Table 4 (power,
p15), Table 5 (per-cell MMLU-Pro, p16), Table 6 (v2 re-sort, p17), Table 7 (27 cells, p18).
Main text makes 11 forward pointers into 9 distinct appendix subsections (A.1–A.8, A.13). Section
"4 Under-Specifications" (§3.3) is a single 15-line paragraph carrying ~14 numbers whose full
development is A.1; the entire ordering argument is A.3; the entire multiplicity treatment is
A.4/A.5; the off-MMLU replication is A.6; the audit is A.7.

The comments in `main.tex` say the summary "states every headline number, so a reader who never
opens the appendix still learns what the four under-specifications are." That is true as a
statement about *naming* the four, and I want to give the authors credit for not deleting anything
in the move. But naming is not the reviewable bar. §3.3 asserts "on ARC-Challenge `credit` alone
moves five of six OLMo-2 arms from significantly above the floor to significantly below it" with no
in-line table; the reader must reach p11 to learn which four conventions form the contrast set and
p13 to see the numbers. §5.1's flip counts cannot be audited without p16.

This is a presentation failure with substantive consequences: a reviewer or reader who reads the
main text linearly *cannot check a single headline number*, and the paper's own thesis is that
numbers must be printed next to their reference. The irony is not lost.

*Fix*: promote exactly two artifacts into the main text — (a) Table 1 or a 4-row condensed version
of it, since the winner's-curse calibration is the paper's most decision-relevant result; and (b)
Table 5, since §5.1 is unreadable without per-cell deltas. Budget: fold §3.3 into two sentences
plus an appendix pointer (it is currently the least load-bearing dense passage), and cut the
§5.1/A.5 duplication in C-02. *Verification*: recompile; assert ≥2 numbered tables appear at or
before the REFERENCES page, and that every count in §5.1 is derivable from a main-text table.

### C-02 (minor, clarity) — §5.1 and Appendix A.5 duplicate 424 characters verbatim

A normalised diff of `05_analysis.tex` against `09a_relocated.tex` finds one shared run of 424
characters: the BH/Bonferroni sentence and the binomial-0.0196 sentence appear near-identically in
both. On a submission fighting for main-text space this is 424 characters that could have carried a
table row. It also creates the impression that A.5 adds something it does not.

*Fix*: keep the BH/Bonferroni result in the main text (it is decision-relevant) and reduce A.5 to
the one thing only it says — the McNemar non-transfer and the 0.60 expected false-positive yield.
*Verification*: re-run the diff; assert no shared run >120 chars between the two files.

### C-03 (major, clarity/reproducibility) — the build record does not describe the shipped PDF

`evidence/build_record.json` reports `pdf_pages: 22`, `pdf_bytes: 355196`,
`pdf_sha256: 56a376e128c3…`, and `build_gate_pass: true` with `n_overfull_hbox: 0`. The shipped
`manuscript/main.pdf` is **24 pages, 366583 bytes, sha256 `1fbaaf9983…`** (and the MANIFEST agrees
with the file, not with the build record). So the only artifact certifying "0 overfull hboxes, 0
undefined references, 0 undefined citations" was produced from a *different* document than the one
under review. The record also self-declares `pdf_visually_inspected: false`.

I re-derived what I could myself: the PDF has no undefined-reference markers (`??` absent), all 12
tables have captions and labels, all 17 bib keys resolve in the rendered reference list, and I found
no text block extending past the ICLR right margin. So I am not alleging a broken build — I am
saying the shipped provenance does not cover the shipped artifact, which for a paper whose central
methodological pitch is mechanical claim-to-evidence binding is a self-inflicted wound.

*Fix*: regenerate `build_record.json` from the final compile and assert
`pdf_sha256 == sha256(main.pdf)` in the emitter. *Verification*:
`sha256sum manuscript/main.pdf` equals `build_record.json:pdf_sha256`, and `pdf_pages` equals the
rendered page count.

### C-04 (major, citation integrity) — "27 actionable checklist items" is wrong, and it is the sentence that establishes the paper's gap

Intro sentence 3: "A systematic construct-validity review of 445 language-model benchmarks offers
27 actionable checklist items, but none asks authors to report a null, a chance level, or a
constant predictor." I counted the checklist in Bean et al. (arXiv 2511.04703v1, §"checklist here,
grouped by recommendation"): **28** items, in 8 groups. The paper's own abstract says "eight key
recommendations." The count 27 does not appear in the source; 21 (codebook questions) and 30
(double-annotation categories) do.

The second half of the sentence — the part that actually does work — is correct: I read all 28
items and none requests a null, chance level, or constant predictor. So this is a fixable numeric
error in a load-bearing citation, not a fabricated claim. But it is precisely the kind of
"reference exists, local sentence not verified" case the paper's own reproducibility statement
claims to have mechanically eliminated ("610 numerals with none unresolved"), which suggests the
checker validates numerals against the authors' *own* evidence set and not against cited sources.

*Fix*: change to "28 actionable checklist items across eight recommendations" (or cite the
recommendation count instead, which is more robust to versioning), and state which arXiv version
was counted. *Verification*: count `\square` occurrences in the cited version's checklist section;
assert the printed count matches.

### C-05 (minor, clarity/technical) — Table 1's caption reports a gap range that no row attains

Table 1's caption: "the `+0.43`–`+2.60` pp gaps there are the regime our own evidence record marks
'do not oversell'". "There" refers to the five inside-noise letter constructs. Their gaps, computed
from the table's own Floor and Chance columns, are PIQA 0.4897, CommonsenseQA 0.8845, ARC-Challenge
1.5202, ARC-Easy 1.6253, OpenBookQA 2.6000. The minimum is **+0.49**, not +0.43. The upper end is
correct. Either the lower bound is stale (it looks like a figure from a different construct set) or
the caption is describing a set that is not the five rows the sentence points at.

*Fix*: print `+0.49`–`+2.60`, or name the set the range actually covers. *Verification*: recompute
`100*(Floor-Chance)` over the five rows the sentence scopes and assert the printed endpoints equal
`min` and `max`.

### C-06 (minor, clarity) — the 15th designated damaged cell is never given its floor delta

Appendix A.4 makes an explicit promise: "any damaged arm excluded from that denominator is named at
the point of exclusion together with its own floor delta." The 14/15 headline needs 15 designated
damaged cells: 12 non-OLMo (Table 5) plus "the three designated damaged OLMo-2 cells" — keep8,
keep10, keep12 per Table 5. But `shortgpt16@200000` is listed in §4 as one of the OLMo-2
prune-then-heal arms, appears in Table 7 with accuracy 0.153341 (i.e. **+3.67 pp above the 0.116606
floor**, by far the largest damaged-arm floor delta in the paper), and is *not* in Table 5, *not* in
the 14/15, and *never* named at a point of exclusion with its floor delta. `keep14` is in the same
position (Table 6 gives its +0.324, so at least its delta is printed somewhere).

I am not alleging outcome-dependent selection — "designated damaged" plausibly means the k8–k14
ladder rungs and shortgpt16 is a differently-constructed baseline. The problem is that a reader
cannot tell, because the paper never says which arms are in the designated set, only how the set is
*defined*. And the one arm that would most obviously complicate a "damage drives letters to the
floor" reading is the silent one.

*Fix*: enumerate the 15 designated cells explicitly (a list or a Table 5 column), and state
shortgpt16's status with its floor delta at the point of exclusion, per A.4's own promise.
*Verification*: assert the enumerated set has cardinality 15 and that every OLMo-2 arm named in §4
is either in the set or excluded-with-delta.

### C-07 (minor, clarity) — the asymmetric-standard table is unnumbered and uncaptioned

The single most consequential retraction in the paper — 10/12 vs 1/12 collapsing to 3/12 vs 1/12
under one standard — is presented on p7 as a bare `tabular` in a `center` environment with no
number, no caption, no label, and no provenance identifier. Every other table in the paper carries
an `E-*` evidence tag. This one cannot be cited, cannot be cross-referenced, and is the only
tabular material in the main text.

*Fix*: promote to a numbered, captioned table with its evidence identifier (the claim map lists
`evidence/s2_03_symmetric_inference.json`, which does not appear in the manuscript at all).
*Verification*: `\label` resolves; caption names an `E-*` identifier present in Table 12.

### C-08 (minor, clarity) — undefined internal shorthand leaks into the text

Four items are used without definition:
- §5.3: "the **A01** own-modal null over-credits **P1** by 1.37 points and flips its sign." Neither
  `A01` nor `P1` is defined anywhere in the manuscript (each occurs exactly once). `A01` reads as an
  internal project identifier; `P1` as an internal arm label.
- §A.9: "**G1** and **G2** fire on no cell." G1/G2 occur nowhere else. A reader cannot tell what
  gates these are or whether "fire on no cell" is reassuring or vacuous.
- §5.2 introduces "**residual fraction**" and reports it moving by 2.11x / 0.90x / 0.98x without
  ever defining it. It is plainly not `recovery_fraction` (defined in Eq. 6), and the paper never
  says what it is a fraction *of*.
- `w_s` appears four times in the ordering argument (§3.2 and A.3) and is never defined; the reader
  must infer stratum weight `n_s/n` from context. `n_s` *is* defined; `w_s` is not.

*Fix*: define or delete each. `residual fraction` needs one sentence with a formula; `A01`/`P1`
should be replaced by the arm names used elsewhere; G1/G2 should be named or dropped; add
"where `w_s = n_s/n`".
*Verification*: grep for each token; assert first occurrence is preceded by a definition.

### C-09 (minor, clarity) — "cell" is the paper's primary unit of analysis and is never defined

Every headline denominator (14/15, 10/15, 3/12, 27, 0/60, 21, 52/60) is a count of "cells." The word
appears 40+ times across 14 files. It is never defined. From Tables 5 and 7 a reader can infer
(arm x benchmark x interface), but the ambiguity is load-bearing: "21 evaluated cells" on MMLU-Pro
vs "27 cells" in v2 vs "60 damaged cells" off-MMLU vs "15 designated" — these denominators cannot
be reconciled without knowing what varies. The abstract opens with "all 21 evaluated cells are
powered" before the reader has any referent at all.

*Fix*: one sentence in §4: "A *cell* is one (arm, benchmark, interface) triple scored under the
fixed protocol." Then state, per denominator, which factor is being counted.
*Verification*: assert every fraction in the abstract and §5 has a stated denominator set.

### C-10 (minor, clarity/consistency) — two roundings of the same two numbers

CommonsenseQA's gap prints as `+0.885` in Table 1 and `+0.884` in Table 10; BoolQ's as `+12.17` and
`+12.171`. Table 10's caption emphasises that it is machine-generated and that "Gap is
100(Floor-Chance), recomputed here rather than read", while Table 1 is hand-authored. Exact values
are 0.8845 (a genuine round-half ambiguity) and 12.1713. Two tables restating the same nine rows at
two precisions in the same appendix invites the reader to look for a discrepancy that is not there.

*Fix*: make the hand-authored Table 1 emit from the same source, or state the precision convention
once and apply it to both. *Verification*: assert every shared row agrees digit-for-digit.

### C-11 (minor, clarity/reproducibility) — captions defer to a statement that does not carry paths

Nine table captions say "see the reproducibility statement for the artifact path." The
Reproducibility Statement contains **no paths at all** — it references only Table 11. The paths are
in Table 12 (`app:provenance`, p24), which the Reproducibility Statement never mentions. So the
one-hop indirection the design intends is a dead end; the reader must find Table 12 by scanning.

*Fix*: point captions at `Table~\ref{tab:artifact-map}` directly, or add one sentence to the
Reproducibility Statement forwarding to it. *Verification*: follow each caption's pointer and reach
a path in ≤1 hop.

### C-12 (minor, clarity) — 417-word single-paragraph abstract

The abstract is one 417-word block carrying 14 distinct numeric results (three `p`-value regimes,
five fractions, two floors, a percentage, a ratio). ICLR abstracts are typically 150–250 words.
Nothing in it is *wrong* — I checked the 14/15, 3/12-vs-1/12, and 3-of-8 claims against the body and
they match, and I appreciate that the narrowing is carried here rather than buried. But at this
density the abstract cannot perform its function: a reader cannot extract the contribution on one
pass. The genuinely novel operational proposal (make the input-blind null a pre-comparison gate)
competes for attention with the winner's-curse calibration, the four DOFs, the flip counts, the
symmetric-standard retraction, the kappa attribution, the 27-cell re-analysis, and the fp32 control.

*Fix*: cut to ~220 words keeping the rule, the not-chance result with its 3-of-8 scope, the 14/15
flip with the 3/12-vs-1/12 correction, and one sentence on v2. Move the rest to the intro bullets
(which, to the authors' credit, share 0% verbatim text with the abstract — the compression is real,
not copy-paste).
*Verification*: word count ≤250; every abstract number still appears in the body.

## 5. Decision-relevant answers

**Strongest verified contribution.** The winner's-curse calibration of the best-constant floor, and
the authors' willingness to let it narrow their own claim from 8 constructs to 3. That
`f̂ = max_L m̂_L` is upward biased on a finite item set is obvious once said and, as far as this
reviewer can tell from the cited literature, is not said in the MC-evaluation baseline work the
paper positions against. Table 1's columns `E[f̂]`, `q_95`, `p` operationalise it, and the caption
does the honest thing (CommonsenseQA and PIQA floors are *below* `E[f̂]`). This is verified in the
sense that the arithmetic reproduces and the reasoning is sound; the 200,000-draw simulation itself
I could not re-run.

**Most severe unresolved issue.** C-01: the main text cannot be audited on its own. This is not a
cosmetic complaint at an area-chair level — combined with C-03 (build record describes a different
PDF) and C-11 (captions defer to a statement with no paths), the paper's verification chain looks
tighter in prose than it is in the shipped artifact.

**Weakest evidence-to-importance ratio.** §5.3's "V2 re-sorts rather than merely shrinks."
Table 6 shows five moved labels; of these, two are dissolutions of below-floor claims (shrinkage),
one is a withdrawal, and the two "re-sorts upward" are `qwen3/k14` (+0.233 → +0.267, both trivial)
and `olmo2/keep14` (at-floor → TRACE_SIGNAL at +0.608 pp with `p=0.0172`, uncorrected, among 27
cells where the paper's own A.4 says per-cell α is not simultaneously valid). "Re-sorting in both
directions is evidence that the criterion is doing more than shrinking effects" is a strong reading
of two borderline moves. The paper is elsewhere scrupulous about exactly this failure mode.

**Score ceiling without new experiments: 6.5.** All twelve issues above are writing, layout,
counting, and provenance-plumbing fixes. None requires a GPU. If the main text carried two tables,
the checklist count were corrected, the build record matched the PDF, the 15 designated cells were
enumerated, and the undefined shorthand were defined, this would be a clean, unusually honest
measurement paper — a solid accept in a venue that values evaluation methodology. It does not reach
higher than that under current evidence because the empirical core, after the authors' own honest
narrowing, is: three constructs with floors provably above chance, one benchmark with adequate
power, four families of which three are truncate-only (a stated regime confound), and a headline
flip of 3/12 vs 1/12 that vanishes under any multiplicity correction. That is a real and useful
contribution, correctly sized — but it is a modest one, and the paper's own Table 11 records that
six earlier claims did not survive.

**Single change that would most improve confidence.** Promote Table 1 and Table 5 into the main
text. Everything else on my list is a one-line fix.

## 6. What would move my score

**Up (to 6–6.5)**: two numbered tables in the main text with §5.1's counts derivable from them; the
Bean checklist count corrected to 28; a regenerated build record whose `pdf_sha256` matches the
shipped PDF; the 15 designated cells enumerated with shortgpt16's status and delta stated; `cell`,
`residual fraction`, `w_s`, `A01`, `P1`, `G1`/`G2` defined or removed; the p7 flip table numbered
and captioned with its evidence ID.

**Down (to 4)**: if the excluded 15th cell turns out to have been chosen with knowledge of its
floor delta; if the `E-*` evidence files, when shipped, do not reproduce Tables 1/5/7; if the
"610 numerals, none unresolved" checker is found to pass on the two rounding mismatches (C-10) and
the `+0.43` range endpoint (C-05), which would mean the mechanical binding the Reproducibility
Statement advertises does not actually cover printed table values.

## 7. Review limitations

- No `E-*` evidence file is in the snapshot. Every per-item number (bootstrap half-widths,
  permutation `p`-values, the 200k-draw balanced null, the kappa identity "verified on all 27
  cells", `recovery_fraction`, the 610-numeral checker) is **unverified** — neither confirmed nor
  refuted. My arithmetic checks were limited to internal consistency among printed table columns.
- I did not run the authors' checkers (prohibited: several write into `paperC/evidence/`).
- `cho2026choices` and `oostermeijer2026length` are 2026 venue claims (ICLR 2026 Poster, ICML 2026).
  I did not attempt OpenReview `venueid` verification; the paper's own Limitations already discloses
  that the Cho et al. camera-ready could not be diffed. Treat both venue strings as unchecked by me.
- Bean et al. checklist count is from arXiv v1 HTML (2511.04703v1). If the NeurIPS 2025 D&B
  camera-ready has 27 items, C-04 dissolves — but the paper does not say which version it counted,
  which is itself the fix.
- I inspected the PDF via PyMuPDF text/geometry extraction and page rasters at 110–300 dpi, not on a
  human display. Fine typographic defects (hyphenation, sub-point kerning) are outside what I
  checked; I did confirm no text block crosses the right margin and no `??` markers exist.
