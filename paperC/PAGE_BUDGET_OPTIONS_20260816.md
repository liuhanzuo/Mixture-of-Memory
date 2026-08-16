# paperC — ICLR 9-page budget: measured per-section cost and ranked options (2026-08-16)

**Headline: 9 main-text pages IS reachable without deleting a single number, table cell, or
claim — but only by relocating one whole section body plus 6-8 prose paragraphs to the appendix.
Measured best option is `C9` at 9 pages / extent 8.857, with 109.8 pt (17.0% of a page) of slack
left over. Every number below came out of a PDF that was actually built; nothing is predicted.**

This is a **diagnostic sweep**. **No change was applied to the real `paperC/`.** 36 variants were
built in an isolated scratch tree. The only files this task wrote into the repo are this document
and `paperC/code/measure_page_budget.py` + `paperC/state/page_budget_sweep_20260816.json`.

Builds on `paperC/PAGE_LIMIT_ACTION_20260816.md`, whose `baseline` numbers this sweep reproduces
**exactly** (12 pages, extent 11.301), and whose table-relocation rows it reproduces exactly as
well (`movetab_all4` → 11 pages / 10.057). So the harness is verified against the prior artifact
before any new claim is made.

---

## 1. Build command and measurement script

### 1.1 Build

TeX Live 2026 is installed **inside the repo** and is **not on `$PATH`**. `command -v pdflatex`
returns empty; that does not mean latex is absent.

```bash
export PATH="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.texlive/2026/bin/x86_64-linux:$PATH"
cd <variant_dir>
latexmk -pdf -bibtex -norc -gg -interaction=nonstopmode main.tex
```

`-gg` forces a full rebuild every time, so no stale `.aux` can give a wrong cross-reference or a
wrong length. **All 22 candidate/table variants built rc=0 with 0 LaTeX errors, 0 undefined
references, 0 undefined citations, 0 overfull hbox, 0 overfull vbox.** Per-variant build records
are in `paperC/state/page_budget_sweep_20260816.json` under each entry's `build` key.

### 1.2 Measurement (PyMuPDF)

`PyMuPDF 1.28.2` imports from `/opt/conda/envs/torch-base/bin/python3`. The full harness is
`paperC/code/measure_page_budget.py`; the measuring core is:

```python
import pymupdf
BODY_TOP_PT, BODY_H_PT = 85.6, 646.7      # measured on a full body page of this document

doc = pymupdf.open(pdf)
ref_page = ref_y = None
for i, page in enumerate(doc):
    for blk in page.get_text("dict")["blocks"]:
        for line in blk.get("lines", []):
            if "".join(s["text"] for s in line["spans"]).strip().upper() == "REFERENCES":
                ref_page, ref_y = i + 1, line["bbox"][1]; break
        if ref_page: break
    if ref_page:
        # TEXT-COLUMN blocks on this page that start ABOVE the References heading
        preceding_body = sum(1 for b in page.get_text("dict")["blocks"]
                             if b.get("lines") and 40 < b["bbox"][1] < 745
                             and b["bbox"][0] >= 100.0            # excl. line-number gutter
                             and b["bbox"][1] < ref_y - 1.0)
        break

extent              = ((ref_page - 1) * BODY_H_PT + (ref_y - BODY_TOP_PT)) / BODY_H_PT
main_pages_refpage  = ref_page                                  # conservative
main_pages_occupied = ref_page if preceding_body else ref_page - 1
```

Repeat the whole sweep with:

```bash
/opt/conda/envs/torch-base/bin/python3 paperC/code/measure_page_budget.py \
    --scratch /tmp/pcbudget --out paperC/state/page_budget_sweep_20260816.json
```

### 1.3 Three measures, and why the third one was necessary

| measure | definition | role |
|---|---|---|
| `main_pages_refpage` | the page the `References` heading sits on | conservative; what the prior artifact reported |
| `main_pages_occupied` | `ref_page - 1` when **no text-column block on `ref_page` precedes the heading**, else `ref_page` | the defensible count: if References is the first thing on a page, that page carries no main text |
| **slack (pt)** | `732.3 - (bottom of the last main-text block on the last occupied page)` | how close a variant is to spilling back over |

**The `occupied` measure was not in the original harness and it changes conclusions.** Six variants
(`C10`-`C15`, `C17`) put References at `ref_y = 82.8` on page 10 — the *first* line of that page.
Under `refpage` they read "10 pages"; they are in fact 9 pages of main text with page 10 holding
only the bibliography.

**A trap found and fixed while doing this:** the ICLR style prints a **line-number gutter** at
`x0 ≈ 73.1` (the real text column starts at `x0 ≈ 108.0`). A y-window-only body filter counts that
gutter as main text, which makes *every* page look occupied and silently destroys the `occupied`
measure. Hence the `x0 >= 100.0` term. Verified on both `baseline` (page 12) and `C11` (page 10).

**And a reason not to trust `extent` alone near the boundary:** `extent` *saturates*. `C10`-`C15`
and `C17` all report exactly `8.996` because References is pinned to the page top in each — the
metric cannot see that they differ. Their measured slack is **0.0 pt**: page 9 is filled to the
last available point. `extent` says they tie with room to spare; slack says they are on a razor
edge. That is why the recommendation below is not one of them.

---

## 2. Per-section cost table

One section's `\input` commented out per build, everything else untouched. Sorted by cost.

| section | bytes | ref page | **occupied pages** | extent | **Δ extent (pages saved)** | Δ int pages | build |
|---|---|---|---|---|---|---|---|
| `03_method` | 9526 | 10 | **10** | 9.058 | **−2.243** | −2 | clean |
| `05_analysis` | 9175 | 10 | **10** | 9.231 | **−2.070** | −2 | clean |
| `03b_nulls` | 3580 | 10 | **10** | 9.644 | **−1.657** | −2 | **4 undefined refs** |
| `04_experiments` | 4415 | 11 | **11** | 10.057 | **−1.244** | −1 | **1 undefined ref** |
| `01_introduction` | 5498 | 11 | **11** | 10.126 | **−1.175** | −1 | clean |
| `02_related` | 4159 | 11 | **11** | 10.403 | **−0.898** | −1 | clean |
| `00_abstract` | 2858 | 11 | **11** | 10.644 | **−0.657** | −1 | clean |
| `06_discussion` | 2796 | 11 | **11** | 10.652 | **−0.649** | −1 | clean |
| `07_limitations` | 2473 | 11 | **11** | 10.726 | **−0.575** | −1 | clean |
| *(baseline, nothing removed)* | — | 12 | **12** | 11.301 | — | — | clean |

**Do not sum these.** Page packing is non-linear. Every combination in §3 was built and measured
*as that combination*. (The prior artifact's warning about three single-table probes each reading
−0.244 being float-packing artefacts is confirmed here: `movetab_tab_conventions`,
`movetab_tab_two_nulls` and `movetab_tab_power` are all exactly 11.057, and moving all four
tables gives −1.244, not −0.485 − 3×0.244 = −1.217.)

### 2.1 Sections that cannot be removed cleanly (useful information, per the brief)

- **`03b_nulls` — 4 undefined references.** Removing it breaks `\ref{sec:nulls}` (cited from
  `03_method.tex:16`) and `\ref{tab:nulls}` (cited from `01_introduction.tex:19` and twice from
  `09_appendix.tex:46,57`), because `tab_nulls` and `tab_conventions` are `\input` *from inside*
  this section (`03b_nulls.tex:26-27`). Deletion needs repair work; **relocation does not** — see
  `C4`/`C9`, which move the same material and build with 0 undefined refs.
- **`04_experiments` — 1 undefined reference.** Breaks `\ref{sec:setup}`.
- The other seven sections build clean when dropped. `05_analysis` builds clean even though it
  owns `\ref{tab:integrity}`, because `tab_integrity` lives in the appendix.
- Note the whole paper uses `\ref`, never `\Cref` — a grep for `Cref` across `sections/` and
  `main.tex` returns zero hits. So the dangling-reference risk is `\ref`-shaped, and LaTeX reports
  it as a warning (`rc=0`) rather than an error: **`rc=0` is not sufficient evidence that a removal
  was clean.** `n_undef_ref` must be read too. The harness records it per variant.

### 2.2 Where the mass actually is

`03_method` (−2.243) and `05_analysis` (−2.070) hold ~4.3 pages of extent between them but are the
paper's method and its results — neither can leave the main text. The interesting entry is
**`03b_nulls`: −1.657 pages of extent from only 3580 bytes of source.** It is the cheapest section
by bytes-per-page-saved *by a wide margin* (0.474 pages of extent per KB, versus 0.241 for
`03_method` and 0.231 for `05_analysis`), because it carries two of the four remaining main-text
tables. That asymmetry is what makes the recommended option possible.

---

## 3. Measured candidate combinations, ranked

All candidates **relocate** rather than delete: material moves into the appendix (ICLR-legal —
appendices sit after the bibliography and are unlimited) and the main text keeps a real,
compiled in-text summary with an `Appendix~\ref{}` pointer. **The stubs used are the actual strings
that were compiled** (verbatim in `paperC/code/measure_page_budget.py`); measuring with a
placeholder stub would have understated every page count, since a stub costs lines too.

Ranked by "least scientific content lost per page saved", among those that reach 9.

| rank | candidate | **occupied pages** | ref page | extent | **slack on last page** | reaches 9? |
|---|---|---|---|---|---|---|
| **1** | **`C9`** — `03b_nulls` body → appx (+ 6 prose blocks), Discussion untouched | **9** | 9 | 8.857 | **109.8 pt (17.0%)** | **yes** |
| 2 | `C8` — `C9` + the two Discussion relocations | **9** | 9 | 8.761 | 171.7 pt (26.6%) | **yes** |
| 3 | `C17` — `C9` keeping prior-art **and** analysis-multiplicity in main text | **9** | 10 | 8.996 | **0.0 pt (0.0%)** | yes, but zero margin |
| 3= | `C11`–`C15` (leave-one-out of `C9`) | **9** | 10 | 8.996 | **0.0 pt** | yes, zero margin |
| 3= | `C10` — `C8` but `tab:power` stays in main text | **9** | 10 | 8.996 | **0.0 pt** | yes, zero margin |
| 7 | `C19` — `C9` keeping off-MMLU + experiments-multiplicity | 10 | 10 | 9.057 | 627.1 pt | **no** |
| 8 | `C16` — `C9` minus the v1/v2-ordering relocation | 10 | 10 | 9.091 | 605.2 pt | **no** |
| 9 | `C18` — `C9` keeping audit + experiments-multiplicity | 10 | 10 | 9.126 | 583.3 pt | **no** |
| 10 | `C20` — keep `03b_nulls` whole; relocate 10 prose blocks + all 4 tables | 10 | 10 | 9.180 | 548.2 pt | **no** |
| 11 | `C7` — `C3` + 2 more prose blocks | 10 | 10 | 9.286 | — | no |
| 12 | `C5` — `C4` + 2 prose blocks | 10 | 10 | 9.231 | — | no |
| 13 | `C3` — `C2` + Discussion dedup | 10 | 10 | 9.455 | — | no |
| 14 | `C4` — `03b_nulls` body → appx, nothing else | 10 | 10 | 9.492 | — | no |
| 15 | `C2` — 4 prose blocks + all 4 tables | 10 | 10 | 9.581 | — | no |
| 16 | `C21` — `C20` but `tab:nulls`/`tab:conventions` stay | 10 | 11 | 9.996 | 0.0 pt | **no** |
| 17 | `C6` — `C3` but `tab:nulls` stays | 11 | 11 | 10.126 | — | no |
| 18 | `C1` — 4 prose blocks only, all tables stay | 11 | 11 | 10.857 | — | no |

### 3.1 Exactly what each candidate relocates, and what a reader loses

Byte counts are of the original source block; each is replaced by a compiled stub.

**`C9` (rank 1) — 9 pages, extent 8.857, 109.8 pt slack.** Relocates:

| block | source | bytes | what the reader loses |
|---|---|---|---|
| `03b_nulls` **whole body** (`Four Under-Specifications`, §4 → appendix, with `tab:nulls` + `tab:conventions` travelling with it) | `03b_nulls.tex` all | 3580 | Nothing quantitative: the replacement stub keeps **every** headline number in the main text — the 0.125914/0.532164 40.6-point tie-convention span, the 32.5-point excess over the intact model, ARC-Challenge 5-of-6 sign flip, OpenBookQA 0.416 vs 0.644, 18.60%/50.85% tie rates, the 10.6/9.26-point tokenizer moves, and 0.100000/0.110877/0.116606 with both $+1.661$/$1.1661\times$ and $+0.573$/$1.0517\times$ framings. What moves out of line-of-sight is the *per-subsection development* of each of the four degrees of freedom and the character-unit recomputation provenance. A reader who wants the four-way structure follows one `\ref`. |
| prior-art paragraph (Bennett/Brennan/Frary/Brenner/de Vries) | `03_method.tex:50-66` | 1522 | The claim-by-claim novelty boundary. **This is the paper's defence against a "this is just Bennett's $S$" reviewer**, so the stub retains all five citations and the "what we are not aware of" sentence verbatim; only the per-citation elaboration moves. |
| v1/v2 ordering paragraph | `03_method.tex:80` | 2128 | The per-stratum argmax listing (B,E,B,E for `n_opt`=5,6,7,8), the 623/12032 and 1439/12032 counts, the "13 of 27 could not have violated it" argument, and the tightest cell's 0.085 pp attainable violation. The stub keeps the 36-item / 0.299 pp headline, the fact that the bound is **not** an identity, and the "passing v2 ≠ clearing the floor" warning. **This is the single largest prose block in the paper (2128 bytes on one line) and the most technical.** |
| analysis multiplicity qualification | `05_analysis.tex:24` | 1035 | Nothing: stub keeps the BH/Bonferroni 0/12 result, the "undefined rather than 3/12 vs 1/12" reading, and binomial $p=0.0196$. Only the McNemar-does-not-transfer explanation is compressed. |
| `Auditing the audit` subsection | `05_analysis.tex:54-59` | 1086 | Nothing numeric: stub keeps $p=1.042$, 0/24 verdicts, 0/30 $p$-values, 10/15 truncated cells, 0/14 verdicts, 0.0083 points, 5/8 shards, and the `Table~\ref{tab:integrity}` pointer. Loses the narrative ordering of the three defects. |
| experiments multiplicity + designated-damaged | `04_experiments.tex:24-26` | 1259 | Stub keeps the no-correction disclosure, the "not simultaneously valid" framing and 0/60. Loses the full *designated damaged* definition that fixes the 14/15 and 10/15 denominators — **the most consequential loss in `C9`**, because a reviewer checking a denominator has to follow a pointer to do it. |
| off-MMLU replication + constant emitters | `05_analysis.tex:28-30` | 910 | Nothing numeric: stub keeps 16 cells, 0.276000, $\Delta=0.000$, CI95 $[0,0]$, 2.6 points, 10/15, 0/15, 0/60, 25/60, 7/60, 52/60, and the $-3.840$ vs $-3.603$ / 3.92 vs 1.18 power argument. |

**Total: 11 520 bytes relocated, 0 bytes deleted.** Tables `tab_two_nulls` and `tab_power` also move
to the appendix (`tab_nulls`/`tab_conventions` travel inside `03b_nulls`), so **the main text ends
with zero tables** — the honest cost of `C9`, and the main argument for `C10` if a table must stay.

**`C8` (rank 2) — 9 pages, extent 8.761, 171.7 pt slack.** `C9` plus two Discussion paragraphs:

- `06_discussion.tex:8-9` (463 B), *Relationship to prediction debiasing*. **This is genuinely
  redundant**: `02_related.tex:7` already states PriDe, that it fixes the predictor while null
  calibration fixes the reference line, and that reporting only against chance can leave the
  gold-marginal floor unreported. The Discussion paragraph restates that with no new number.
- `06_discussion.tex:14-18` (845 B), *Current claim versus retracted claims*. Partly redundant with
  `07_limitations.tex:12-13`, which already says `H_heal`'s antecedent fails at $-0.139$/$p=0.0964$
  and `H_family` is unavailable. The stub keeps the surviving-claim sentence, 14/15, and both
  appendix pointers.

So on the "is `05_analysis` redundant with `06_discussion`?" question in the brief: **no, not with
`05_analysis` — `06_discussion` overlaps `02_related` and `07_limitations` instead.** `05_analysis`
reports per-cell numbers, `06_discussion` states the reporting rule and scope. The two Discussion
paragraphs above are the only real duplication I found, and they are worth only −0.096 extent
(8.857 → 8.761). Merging the sections is therefore not a lever; **it is a tidiness argument, not a
page-budget one.**

**`C17`, `C11`–`C15`, `C10` (rank 3=) — 9 pages but 0.0 pt slack.** These keep 1-2 more paragraphs
in the main text than `C9` and still land on 9 occupied pages. `C17` is the most attractive on
content grounds: it keeps **both** the prior-art paragraph and the analysis-multiplicity
qualification in the main text. But all seven fill page 9 to the last available point
(`slack = 0.0 pt`, References pinned to the top of page 10). **Any subsequent edit — one added
sentence, one reviewer-requested clarification, a `.bbl` that grows by one line — pushes References
back onto page 9 and returns the paper to 10 pages.** They are correct today and fragile tomorrow.

**Everything at rank 7+ fails.** Worth recording explicitly, because two of them look like they
should work:
- **`C20` (9.180, 10 pages) is the strongest *failing* option and the most informative.** It keeps
  `03b_nulls` whole in the main text and instead relocates **ten** prose blocks (all of `C8`'s
  eight, plus `Read-out versus retained knowledge` §5.2 and `Full precision falsifies the
  numerical-tie mechanism` §5.4) *and* all four tables. That is strictly more scientific content
  displaced than `C9`, and it still does not reach 9. **Conclusion: relocating `03b_nulls`'s body
  is not one option among several — it is load-bearing.** Prose condensation alone, even at ten
  blocks, cannot do it.
- **`C21` (9.996, 10 occupied pages, 0.0 pt slack)** confirms the same thing from the other side:
  `C20` minus the `tab:nulls`/`tab:conventions` moves costs a further 0.816 of extent and lands
  flush against the bottom of page 10.
- `C4` (`03b_nulls` relocation alone) gives 9.492 / 10 pages — necessary but not sufficient.
- `C1` (4 prose blocks, all tables kept) gives 10.857 / 11 pages — prose-only is far short.

---

## 4. Recommendation

**Adopt `C9`.** Measured: **9 occupied main-text pages, References heading on page 9 at
`y = 640.1`, extent 8.857, 109.8 pt (17.0% of a page) of slack, `rc=0`, 0 LaTeX errors, 0 undefined
references, 0 undefined citations, 0 overfull hbox/vbox.**
PDF: `/tmp/pcbudget/C9_C8_minus_discussion/main.pdf` (24 pages total; appendix absorbs everything).

Why `C9` over the alternatives:

- It is the **only** option that reaches 9 with real margin while leaving the Discussion untouched.
  `C8` has more slack (171.7 pt) but buys it by relocating two Discussion paragraphs; since those
  two are the *most* redundant material in the paper, `C8` is the right fallback if more slack is
  wanted later, and it is a strict superset of `C9` (so `C9` → `C8` is a one-step tightening, not a
  redesign).
- The zero-slack family (`C17`, `C11`–`C15`, `C10`) keeps more prose in the main text but leaves no
  room for any future edit. For a paper still under revision, 0.0 pt of slack is not a margin.
- Under the **conservative** `refpage` measure `C9` also reads 9 (References is on page 9, 82% down),
  so it clears the limit under *both* readings. The zero-slack family reads 9 only under `occupied`
  — i.e. only if a program-committee counter agrees that a page whose sole content is the
  bibliography is not a main-text page. `C9` does not depend on winning that argument. This matters
  because it is exactly the "AMBIGUITY 3" already flagged in `gate/venue_page_limit.json`.

### What it costs scientifically — honestly

1. **The main text ends with zero tables.** All four remaining main-text tables (`tab:nulls`,
   `tab:conventions`, `tab:two-nulls`, `tab:power`) move to the appendix. A reader who skims only
   the main text sees no tabular data at all. This is the largest single cost, and unavoidable:
   `movetab_all4` alone (11 pages) proves tables must go, and `C20` proves prose alone cannot
   substitute. If exactly one table must stay, `C10` keeps `tab:power` next to the power argument
   at 9 pages — at the price of 0.0 pt slack.
2. **§4 `Four Under-Specifications of the Reference` becomes an appendix subsection with a single
   dense in-text paragraph.** This is a *paper about how under-specified nulls are*, and its most
   striking single result (one dataset + one tokenizer → floors from 0.125914 to 0.532164) is one of
   its four headline bullets. The stub keeps every number, but the four-way structure — which is
   itself part of the argument that these are *four independent* degrees of freedom — is only
   visible via a pointer. **This is the cost I would flag to a co-author before proceeding**; it is
   a presentational demotion of a contribution, not just of a detail.
3. **The `designated damaged` definition leaves the main text.** Denominators like 14/15 and 10/15
   appear in the Introduction, the Analysis, and the Discussion; their governing definition would
   sit in the appendix. Mitigation: the stub explicitly names the appendix as the home of the
   definition, so a reviewer auditing a denominator is directed, not stranded.
4. **The v1/v2 ordering argument is compressed.** The stub preserves the load-bearing conclusions
   (not an identity; 36 items / 0.299 pp; passing v2 ≠ clearing the floor) but a reader must follow
   the pointer for the per-stratum argmaxes and the "13 of 27 could not have violated it" argument.
   This paragraph is the paper's own self-correction and deserves the caution.
5. **Nothing is deleted.** Verified mechanically, not asserted: a numeric-token census over all
   `sections/*.tex` (comments stripped) shows **0 numeric tokens lost** in `C8`, `C9` and `C17`
   relative to the on-disk paper (446 distinct → 447; every count non-decreasing, the gains being
   stub duplication of numbers now stated in both the main text and the appendix). No number was
   retyped, rounded, or dropped.
6. **No style-file fighting.** No font-size change, no margin change, no `\vspace`, no
   `\tabcolsep`/`\arraystretch` edit, no change to `iclr2026_conference.sty` in any variant. The
   only mechanisms used are moving `\input` positions, demoting one `\section` to `\subsection`, and
   substituting compiled summary prose.

### Integrity checkers on the recommended variants

Run inside isolated copies with their own `evidence/`:

```
code/validate_tex_static.py        -> rc=0, all files OK      (C8, C9, C17)
code/check_prose_vs_evidence.py    -> rc=1, n_mismatch=1      (C8, C9, C17)
```

The single mismatch is **pre-existing and unrelated to this task**: `README.md:23`
(`MMLU content | longest-option split = 0.2845` vs stored `0.268908`). Confirmed by running the
same checker on an **unmodified** copy of `paperC` → identical `n_checked=81 n_ok=80 n_mismatch=1`,
and `README.md` is byte-identical to `HEAD`. Note the committed
`evidence/prose_vs_evidence_check.json` records `n_mismatch=0`, so **this regression predates my
work and is worth a separate look** — it is flagged here, not fixed. No `sections/*.tex` mismatch
is introduced by any relocation.

---

## 5. What I could NOT measure, and why

- **Whether an ICLR program-committee counter accepts `main_pages_occupied`.** For `C9` this does
  not matter (it reads 9 under both measures) — which is precisely why `C9` is recommended over the
  zero-slack family, for whom it matters decisively. Same ambiguity as "AMBIGUITY 3" in
  `gate/venue_page_limit.json`.
- **Rendered visual correctness.** `n_overfull_hbox = n_overfull_vbox = 0` for every variant and
  headings/labels were located by text extraction, but page *images* were not compared. PyMuPDF
  `page.get_pixmap()` can do this on this host; it was not run.
- **Whether the stub prose is scientifically adequate.** I verified the stubs *compile*, *preserve
  every numeric token*, and produce 0 undefined refs. Whether a reviewer finds the compressed
  prior-art boundary or the compressed ordering argument sufficient is an editorial judgement, not
  a measurement. The stubs are drafts sized for the measurement, not final copy.
- **Sub-paragraph condensation.** Every variant relocates whole paragraphs. Rewriting a paragraph
  to be genuinely shorter (rather than moving it) was not measured, because its page cost depends
  on wording that does not exist yet. Given `C20`'s failure at ten relocated blocks, condensation
  is unlikely to substitute for the `03b_nulls` move — but it could plausibly convert a zero-slack
  variant like `C17` into one with margin, and that is the one unexplored direction I would test
  next.
- **Interaction with any *other* pending edit to paperC.** All measurements are against the on-disk
  state at commit-time (`baseline` = 12 pages / extent 11.301, reproducing
  `PAGE_LIMIT_ACTION_20260816.md` exactly). Any content added after this sweep invalidates the
  slack figures, though not the ranking.
- **GPU: zero used.** pdflatex + PyMuPDF on the login host only. No node was contacted.

### One incident to record

While running the repo checkers I first symlinked the *real* `paperC/evidence/` into two scratch
variants; `validate_tex_static.py` and `check_prose_vs_evidence.py` write their output JSON there,
so `evidence/tex_static_validation.json` and `evidence/prose_vs_evidence_check.json` were
overwritten in the live tree. **Both were restored with `git checkout --` and
`git status paperC/evidence/` is clean.** The two files that remain modified in `paperC/`
(`main.pdf`, `gate/build_record.json`, mtime 06:48) pre-date this session. Lesson: those checkers
are *writers*, not read-only probes — give a scratch build its own `evidence/` copy, never a
symlink.
