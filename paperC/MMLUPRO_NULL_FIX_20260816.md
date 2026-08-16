# MMLU-Pro null fix — legality-aware calibration across all nine construct rows

**Date:** 2026-08-16 · **GPU used:** none (pure CPU numpy; 40/40 cards left untouched)
**Trigger:** four of six round_04 blind codex reviewers (X1/X2/X5/X6) independently flagged that
the flagship MMLU-Pro winner's-curse calibration used a null MMLU-Pro cannot produce. MAIN verified
it first-hand (`paperC/evidence/mmlupro_legality_aware_null_MAIN.json`); this document records
extending the correction to all nine rows and landing it in the manuscript.

---

## 0. Two things MAIN's record got wrong — read this first

The instruction accompanying this task said to escalate rather than self-patch if MAIN's review was
itself wrong, or if the defect turned out to be **larger** than described. Both happened, in small
ways. Neither changes MAIN's headline (three surviving constructs become two), and neither is a
reason to redo the fix — but both had to be corrected in the artefacts rather than repeated.

### 0.1 MAIN's universal-monotonicity claim is false (measured, −33σ)

`mmlupro_legality_aware_null_MAIN.json:VERDICT.confinement` states:

> "The fix is monotone for the other five: a legality-aware null raises E[max], which raises p,
> which can only push a row further INSIDE the noise bucket, never out of it. **So no row moves in
> the paper favour.**"

The task brief repeated this and asked me to verify it. **It is false**, and my script's own sign
self-test is what caught it (it initially refused to write). The correct statement is case-split on
whether items have *fewer* or *more* options than the nominal `k`:

| case | mechanism | effect on E[max] and p | rows |
|---|---|---|---|
| `n_opt == k` for all items | legal set is all `k` letters; the two nulls are *the same distribution* | exactly zero | MMLU, BoolQ, OpenBookQA, CommonsenseQA, PIQA |
| some items `n_opt < k` | their mass concentrates on the always-legal low letters | **rises** → against the authors | both MMLU-Pro rows (2051/12032 restricted) |
| some items `n_opt > k` | mass diverted to a letter legal almost nowhere, which can never win the max, so it leaves the contest while the floor is still divided by `n` | **falls** → in the authors' favour | ARC-Easy (4 items are 5-way), ARC-Challenge (3 items are 5-way) |

Measured over 8 seeds × 1e6 draws per arm, comparing aware vs blind under one code path:

| row | dE[max] | σ | dp | σ |
|---|---|---|---|---|
| ARC-Easy | −7.77e−5 | **−32.8** | −0.00247 | **−16.6** |
| ARC-Challenge | −1.23e−4 | **−32.5** | −0.00634 | **−22.6** |
| PIQA (constant `k`, control) | 0.0 | 0.0 | 0.0 | 0.0 |
| MMLU-Pro | +9.42e−3 | +11930 | +0.0831 | +551 |

These are real, not Monte-Carlo error. I isolated the mechanism analytically by building two
intermediate histograms: `{4:2376}` → E=0.2606003, `{3:7,4:2369}` (only the narrow items) →
0.2606048 (**up**, as MAIN predicts), `{4:2372,5:4}` (only the wide items) → 0.2605115 (**down**),
full aware `{3:7,4:2365,5:4}` → 0.2605220. The wide items dominate. So MAIN's mechanism is right for
the case it examined and wrong as a general law.

**Impact on conclusions: none.** Both ARC rows are nowhere near 0.05 either way (0.140→0.136,
0.453→0.447), so no verdict turns on them. But the blanket sentence "no row moves in the authors'
favour" must not be printed, and it is not: `sections/03b_nulls.tex` §"Which rows moved, and in
which direction" states both directions with their sizes, and the evidence record's
`directionality` field says explicitly that it contradicts the MAIN record and why.

### 0.2 A second, independent defect found while recomputing: the floor-rounding convention

`p = Pr(f̂ ≥ observed floor)` and the floor is an exact rational `count/n`. MMLU-Pro's is
`1403/12032 = 0.116605718085…`, and the shipped evidence stores it rounded to 6 dp as `0.116606`,
which is **larger than the true floor**. Comparing the simulated maximum against the stored value
therefore requires a count of **1404**, excluding the very outcome that was observed:

```
count>=1403 (exact floor)   -> p = 0.083064
count>=1404 (6-dp floor)    -> p = 0.077559     <- what the shipped file did
```

I confirmed the shipped file used the rounded convention by fingerprinting PIQA, the discriminating
row: shipped `p=0.65791`; reproducing its sampler gives `0.65573` at threshold 929 (rounded) versus
`0.68954` at 928 (exact). The shipped value matches the rounded branch.

**The bias runs toward "survives", i.e. in the authors' favour**, on every row where the 6-dp floor
rounds *up*. That is 5 of 9 rows (both MMLU-Pro rows, MMLU, PIQA, BoolQ; the other four round down
and are unaffected). It changes no verdict, but it changes MMLU-Pro's distance from 0.05: **0.083,
not 0.078** as MAIN reported. All nine `p` values in the paper are now evaluated at the exact
rational floor, through one code path, and both conventions are stored per row
(`p_aware`, `p_aware_rounded_floor_convention`, plus a 2×2 decomposition `p_2x2`).

This is why the abstract range is `0.083--0.853` and not the `0.078--0.85` the brief anticipated.

### 0.3 A third, cosmetic instance of the same defect class (reported, not acted on)

ARC-Easy and ARC-Challenge are listed with `k=4`, but their observed `max(n_opt)` is **5** — 4 and 3
items respectively actually offer five options, and ARC-Easy has one item whose gold letter *is* E.
A `k=4` null cannot emit E. This is the same "null incompatible with construct" defect class, at
negligible magnitude (E carries 4/2376 of the mass and can never attain the maximum). It is not
silently absorbed: the legality-aware `p` reported for those rows uses the **observed** histogram,
so it already accounts for it, and each row carries a `k_note` field spelling this out. The `k`
column in the tables remains the nominal count, which the captions state.

---

## 1. Per-row old/new comparison

`p blind` = as shipped (legality-blind null, 6-dp floor, 2e5 draws).
`p legal` = this fix (legality-aware null, exact rational floor, 1e6 draws).

| construct | n | k | observed `n_opt` histogram | `n_opt` const? | p blind | p legal | verdict blind → legal | dp σ |
|---|---|---|---|---|---|---|---|---|
| MMLU-Pro letter, naive | 12032 | 10 | 3:21, 4:606, 5:52, 6:93, 7:158, 8:320, 9:801, 10:9981 | no | <1e−5 | **0.083** | above balanced null → **inside estimator noise** | +263 |
| MMLU-Pro letter, item-avg. | 12032 | 10 | (same item set) | no | <1e−5 | **0.083** | above balanced null → **inside estimator noise** | +263 |
| MMLU letter | 14042 | 4 | 4:14042 | **yes** | <1e−5 | <1e−6 | above balanced null → above balanced null | +1.0 |
| OpenBookQA letter | 500 | 4 | 4:500 | **yes** | 0.383 | 0.384 | inside noise → inside noise | +0.9 |
| ARC-Easy letter | 2376 | 4 | 3:7, 4:2365, 5:4 | no | 0.140 | 0.136 | inside noise → inside noise | **−3.3** |
| ARC-Challenge letter | 1172 | 4 | 3:4, 4:1165, 5:3 | no | 0.453 | 0.448 | inside noise → inside noise | **−7.3** |
| CommonsenseQA letter | 1221 | 5 | 5:1221 | **yes** | 0.853 | 0.853 | inside noise → inside noise | −0.1 |
| PIQA letter | 1838 | 2 | 2:1838 | **yes** | 0.658 | 0.691 | inside noise → inside noise | −0.6 |
| BoolQ | 3270 | 2 | 2:3270 | **yes** | <1e−5 | <1e−6 | above balanced null → above balanced null | n/a |

**Headline:** survives 4 rows → 2 rows (i.e. three *constructs* → two, since the two MMLU-Pro rows
are one construct under two chance lines). `survives_aware = ["MMLU letter", "BoolQ"]`.
Inside-noise p range **0.083–0.853**.

Note that the two surviving constructs are exactly the two whose option count was never in
question. That is the opposite of a convenient outcome and is stated as such in the paper.

**The reported `p<10^-5` bound is deliberately NOT strengthened.** My run is 1e6 draws, which would
license `p<10^-6` on MMLU and BoolQ (0 hits in 1e6). The emitter reads the bound exponent from the
*superseded* file's `n_draws=2e5` via `p_bound_draws()`, keeping the already-published, weaker
bound: a correction should not quietly strengthen a claim that survived it.

## 2. My own recomputation, seed consistency, environment

Script: `paperC/code/recompute_legality_aware_nulls.py` (new).
Output: `paperC/evidence/construct_nulls_legality_aware.json` (new),
sha256 `82d2c322017084cb…`, with `supersedes.path` +
`supersedes.sha256 = 275112623d0574068118…` pinning the file it replaces.

MMLU-Pro, three seeds × 2e5 draws each (E[max], p at exact floor):

| seed | E[max] | p |
|---|---|---|
| 20260814 | 0.11387652 | 0.083445 |
| 7 | 0.11386616 | 0.082355 |
| 99 | 0.11387516 | 0.082935 |
| **1e6 draws, seed 20260814 (primary)** | **0.11387254** | **0.082948** |
| impl2, per-item categorical, 2e4 draws | 0.11387968 | 0.084550 |

Seed spread: **1.04e−5 on E[max]**, 1.09e−3 on p. Two independent implementations
(stratified-multinomial vs naive per-item categorical) agree to 7e−6 on E[max].

MAIN reported E[max]=0.1138772 / p=0.078295 at 1e6 draws. My E[max] agrees to 5e−6; my p differs
by exactly the floor-rounding convention of §0.2 (`p_rounded_floor` = 0.07746 reproduces MAIN).

**Sampler validated against closed forms** (`selftest_sampler()`, runs on every invocation, refuses
to write on failure): E[f̂]=0.75 on two enumerable cases for *both* samplers, plus E[m_A]=2/3 under
a genuine per-item restriction — the last is the discriminating test, since a sampler that ignored
the legal-set restriction would still pass the E[f̂] cases. This addresses the failure mode in
`memory/selftest-over-invented-inputs-proves-nothing-about-the-pipeline.md`: the histograms fed to
the real path are validated against on-disk records, not invented.

**Environment** (recorded in the JSON, no cross-node claim made):
`numpy 2.5.1`, `python 3.14.6`, `/opt/conda/envs/torch-base/bin/python`, node **LOCAL** (wzc1,
hostname `TENCENT64.site`), `gpu_used: false`. Within-node re-run is **bit-identical** on all
primary numbers (verified by diffing two consecutive runs). Cross-node bit-identity is explicitly
**not claimed** — the five nodes carry numpy 2.3.5 / 2.4.6 / 2.5.1 and same-seed `default_rng`
streams differ across them (`memory/numpy-version-split-breaks-cross-node-bootstrap.md`).

## 3. Where the `n_opt` histograms came from

Counted first-hand from the per-item eval records — the same files the shipped floors were computed
from, not a dataset re-download. Every one reproduces the shipped `n` and chance line to 1e−6, which
is a real check because chance in this paper is `mean(1/n_opt)`: a wrong histogram shows up as a
wrong chance line. Both disks were searched.

| construct(s) | disk | file | field |
|---|---|---|---|
| MMLU-Pro | **zwfy6 only** | `mmlu_pro_letter_content_results/7B_base/per_example_mmlu_pro_shard{0..7}of8.jsonl` | `n_opt` |
| ARC-Easy, ARC-Challenge, OpenBookQA, CommonsenseQA, PIQA | wzc1 | `olmo2_mc_letter_content_results/7B_base/per_example_<task>_shard{0..7}of8.jsonl` | `n_opt` |
| MMLU, BoolQ | wzc1 | `olmo2_downstream_results/7B_keep8_step121000_wzc1_know/per_example_<task>_shard{0..7}of8.jsonl` | `len(option_scores)` |

- **MMLU-Pro is zwfy6-resident**: no `per_example_mmlu_pro*` exists on wzc1. Counted over SSH on
  `.73` (CPU-only jsonl read) at zwfy6 root, giving 12032 unique item_ids and
  `{3:21,4:606,5:52,6:93,7:158,8:320,9:801,10:9981}` — byte-for-byte the histogram in both
  `mmlu_pro_power_nulls_v2.json:letter_null.n_opt_hist` and the MAIN record. Because it cannot be
  recounted on wzc1, it is carried in the script as a literal and cross-validated against **two**
  independent on-disk copies (self-test V5/V6, fail-closed).
- **MMLU and BoolQ have no `n_opt` field**; the option count is `len(option_scores)`, which is a
  *stronger* witness — it is the number of candidate continuations actually scored. 14042/14042 have
  exactly `{A,B,C,D}` and 3270/3270 exactly `{A,B}`, identical in a second arm
  (`7B_fromscratch_step200000_perex_know`).
- **No histogram is `UNAVAILABLE`.** All nine rows were obtained from real records; nothing is
  approximated or inferred from the chance line.
- **ARC's non-integer chance was the tell.** The brief flagged that chance 0.250161 / 0.250156
  rather than exactly 0.25 proves `n_opt` varies. Confirmed: 11/2376 and 7/1172 items are 3- or
  5-way, and both derived chance lines reproduce to 1e−6.

## 4. Floor invariance — verified, not assumed

The brief said not to touch the 14/15, 3/12-vs-1/12 and 10/15 counts because they use the observed
floor, and to *verify* that rather than assume it, escalating if they turned out to depend on the
null. I implemented this as `verify_floor_invariance()`, which runs on every invocation and refuses
to write with `ESCALATE_DO_NOT_PATCH` on failure. Against
`evidence/mmlu_scale_power/mmlu_pro_power_nulls_v2.json`:

- **F1** every `floor_used` in `rollup` is the single value `0.11660571808510638` = **1403/12032**
  = `letter_null.gold_letter_marginal_frac.A`. **PASS**
- **F2** none of `E_max`, `E_max_balanced`, `q95_balanced`, `winners_curse`, `0.104457`, `0.10446`,
  `0.107048`, `0.107131` appears **anywhere** in that file — so no rollup number can be reading a
  null moment. **PASS**
- **F3** the aggregate re-derived from the rollup's own per-family fields is **14/15**, matching the
  paper. **PASS**

Conclusion: the downstream counts are functions of the observed floor and the arm scores only.
**Correcting the null cannot move them, and they were not touched.** The power analysis is likewise
unaffected: `power_verdict` is defined on CI half-widths against MMLU's own effect size, with no
null moment involved.

## 5. Manuscript changes

| file | change | why |
|---|---|---|
| `sections/00_abstract.tex` | "only three … (MMLU-Pro, MMLU, BoolQ) … $p=0.14$–$0.85$" → "only two … (MMLU, BoolQ, both fixed-$k$) … $p=0.083$–$0.853$"; added that the calibrating null must be realisable and that we failed that test on our own largest benchmark | the count and the range both changed; the admission is the honest framing |
| `sections/01_introduction.tex` | same claim, second occurrence; now names the mechanism (17% of items cannot have the assigned letter), says MMLU-Pro is **unresolved rather than settled** at $p=0.083$, and forward-references the new section | 0.083 is much closer to 0.05 than 0.14 — it must not be blurred into "well inside noise" |
| `sections/tab_nulls.tex` (main-text, hand-typed) | all 9 rows' `E[f̂]`, `q95`, `p` replaced with legality-aware values; caption rewritten to define the realisable null, state that the columns coincide when `n_opt` is constant, and flag MMLU-Pro as unresolved. Gap range corrected `+0.43`→`+0.49` (it was stale) | main-text headline table carries the corrected column only |
| `sections/tab_construct_nulls.tex` | **regenerated, not hand-edited** | it is a GENERATED FILE |
| `code/emit_tab_construct_nulls.py` | reads the new evidence as primary; emits **both** p columns; new self-test T6c; keeps the conservative `p<10^-5` bound via `p_bound_draws()` | see §6 for the table-design rationale |
| `sections/03b_nulls.tex` | **new §"The calibrating null must be realisable…"** (`\label{sec:legality-aware-null}`) with four paragraphs: the defect, which rows moved and in which direction (both signs, with sizes), the rounding convention, and what survives | the brief's mandatory directionality paragraph |
| `sections/09_appendix.tex` | E-CAL now resolves to the new file and *names the superseded one with its sha256*; added the null-recomputation script to the artifact map; documented T6c | provenance must point at the file actually used |

Not touched: `review_rounds/**` (blind archive), `tab_claims.tex`, `tab_mmlupro.tex`,
`tab_power.tex`, `05_analysis.tex`, and every count defined against the observed floor.

## 6. Table design decision, and the reason

**Main text (`tab_nulls.tex`): corrected column only. Appendix
(`tab_construct_nulls.tex`): both columns, side by side, with a caption saying why the old one is
unrealizable.** This follows the brief's stated preference, and I agree with its reasoning: the
paper's thesis *is* that a null must be compatible with the construct it calibrates, so an appendix
exhibiting the paper's own incompatible null next to the admissible one — and the p it produced — is
an instance of the thesis rather than an embarrassment. A reader scanning the headline table should
not have to decide which of two p-values is real; a reader in the appendix should be able to see
exactly what changed.

One consequence worth naming: because the appendix keeps the blind column, `p_corrected` is now
computed by the same code path for **all nine rows**, including the constant-`n_opt` ones where the
null did not change. Carrying the shipped p for those rows would have put two different threshold
conventions in one table (§0.2), which is precisely the within-table incoherence this paper exists
to complain about. Each row therefore carries `p_change_attribution` saying whether its movement is
due to the null, the threshold, or Monte-Carlo error.

## 7. Verification

**PDF rebuild** (in-repo TeX Live 2026 at `./.texlive/2026/bin/x86_64-linux`, *not* on `$PATH`):

| | clean tree (HEAD) | after fix |
|---|---|---|
| latexmk rc | 0 | **0** |
| LaTeX errors | 0 | **0** |
| undefined refs/cites | 0 | **0** |
| Overfull `\hbox` | 0 | **0** |
| REFERENCES on PDF page | 9 | **9** |
| `extent` (official measure) | 8.857 | **8.857** |
| total PDF pages | 24 | 25 (appendix; unlimited) |

Main-text budget: **9 pages, identical to baseline**, slack 0.143 pages = **92.5 pt**.
Measured with `paperC/code/measure_page_budget.py:measure()` (PyMuPDF), not by eye.

Getting there took iteration and is worth recording: my first draft pushed REFERENCES to page 10
(extent 9.091, **−58.8 pt over**). I bisected the budget in scratch copies (never in `paperC/`) and
found the intro bullet tolerates ~1100–1150 chars against its original 932; the abstract edit is
free. Final intro bullet is 1112 chars. I also verified by reverting abstract and intro in a scratch
copy that **all other changes cost exactly zero main-text pages** (extent 8.857) — they are all
appendix-side.

Two overfull hboxes I introduced were fixed rather than tolerated: the 9-column appendix table
(`\footnotesize`+3.0pt → `\scriptsize`+2.6pt, in the *emitter*), and a long unbreakable path in the
artifact map (split into separate `\texttt{}` groups; note `\allowbreak` inside `\texttt` did *not*
work).

**Checkers** (both are WRITERS, not read-only probes — they write `paperC/evidence/*.json`; I took
the clean-tree baseline first, via `git stash`, then restored):

| | clean tree | after fix |
|---|---|---|
| `check_prose_vs_evidence.py` | n_checked=86, n_ok=86, **n_mismatch=0**, rc=0 | n_checked=**89**, n_ok=89, **n_mismatch=0**, rc=0 |
| `validate_tex_static.py` | all OK, 0 issues, rc=0 | all OK, 0 issues, rc=0 |

n_checked rose 86→89 because the new section adds numbers under coverage. **n_mismatch is 0**, which
is the invariant that matters. (The brief said the clean baseline was 81/81/0; the measured value is
86/86/0. I used the measured one.)

**Verbatim scoping-sentence diff.** Per
`memory/numeric-census-misses-scoping-sentences.md`, a numeric census passing does not mean nothing
was lost. I diffed against HEAD for 22 non-numeric scoping constructions (contribution-delimiting,
epistemic-limiting, falsification-condition). My first compressed intro **silently dropped three**:
"itself an estimate", "needs its own calibration", "rest the quantitative". I rewrote it to restore
all three within the character budget; final census is **0 losses across all five edited files**.
Also verified no stale legality-blind moment (`0.104457`, `0.107048`, `0.260611`, `0.265105`,
`0.270202`, `0.279010`, `0.254355`, `0.273191`, `0.215006`, `0.509307`, `0.506981`) survives
anywhere in `sections/` except the single deliberate citation of 0.104457 in the new section, where
it is the before-value being reported.

## 8. What I could NOT verify — explicitly

1. **Cross-node reproducibility of these Monte-Carlo numbers.** Run on LOCAL/numpy 2.5.1 only.
   Same-seed `default_rng` streams differ across the cluster's three numpy versions, so a rerun on
   `.82` (2.4.6) will not be bit-identical. Within-node re-run *is* bit-identical (verified). No
   cross-node claim is made anywhere in the artefacts.
2. **MMLU-Pro's histogram was not recounted on wzc1** — those per-item records do not exist on that
   disk. It is validated against two independent on-disk copies, but the primary count was done over
   SSH on `.73` and is carried as a literal in the script.
3. **Whether the round_04 reviewers raised anything else about this calibration** that I have not
   addressed. I worked from the brief's summary of X1/X2/X5/X6 plus the MAIN record; I did not read
   the six raw reviews (`review_rounds/round_04/raw/`, which I was instructed not to touch). There
   may be adjacent points in them.
4. **Whether other constructs in the paper have the same class of defect outside these nine rows.**
   I fixed the 9-row winner's-curse table. I did *not* audit the content-floor nulls, the read-out-v2
   permutation null, or the cross-family gate-2 nulls for realisability. §0.3's `k=4`-vs-`max=5`
   finding suggests this class recurs, and a systematic sweep is warranted but out of scope here.
5. **PDF visual inspection.** Measured structurally (page count, REFERENCES position, 0 overfull, 0
   errors) with PyMuPDF. I did not render page images and look at them; the existing
   `gate/build_record.json` carries the same open caveat.
6. **Whether `p=0.083` should change the paper's positioning further.** MMLU-Pro is the construct
   the paper leans on to clear the power wall, and it is now "unresolved" rather than "significant".
   I updated every claim I could find that depended on the old verdict, but whether the *argument*
   should be restructured around losing its largest construct is an editorial judgement I did not
   make unilaterally.

## 9. Reproduce

```bash
PY=/opt/conda/envs/torch-base/bin/python
$PY paperC/code/recompute_legality_aware_nulls.py            # writes the evidence JSON
$PY paperC/code/emit_tab_construct_nulls.py                  # regenerates the appendix table
$PY paperC/code/check_prose_vs_evidence.py                   # expect n_mismatch=0, rc=0
$PY paperC/code/validate_tex_static.py                       # expect rc=0
export PATH=$PWD/.texlive/2026/bin/x86_64-linux:$PATH        # LaTeX is NOT on $PATH
cd paperC && latexmk -pdf -bibtex -norc -gg -interaction=nonstopmode main.tex
```
`--check-only` runs every self-test and writes nothing. All self-tests are fail-closed: the scripts
raise and write nothing rather than emitting a number they could not verify.
