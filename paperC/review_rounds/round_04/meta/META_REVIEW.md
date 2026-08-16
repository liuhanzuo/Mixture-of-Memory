# Meta-Review — paperC, round 04 (ICLR 2026)

> **Verdict: 4.5 / 10 — weak reject.** Ceiling under current evidence **6.0**, limited by the
> **evidence**, not the writing, the analysis or the artifact. Twelve reviewers, median 5.0, lower
> quartile 4.0, 10/12 below borderline, no accept. The decisive finding — reached independently by
> 9 of 12 and re-verified here without simulation — is that the paper's flagship calibration is
> misspecified in the direction that flatters it, and is detectable from two columns of the paper's
> own table. Every blocking fix is writing, re-analysis on data already collected, packaging or
> citation work. **No new GPU time is required for acceptance.**

**Meta-reviewer scope.** I read the twelve reviews in `round_04/raw/`, the frozen submission in
`round_04/submission/` (manuscript + its two evidence files + `MANIFEST.json`), the rubric, and the
output schema. I did not read any `*_MAIN.json`, `PANEL_AGGREGATE_FINAL.json`, any `*_FIX_*` /
`*NOTES*` / `PACKAGING_DEFECT*` deliverable, `paperC/state/`, any earlier round, the live
`paperC/sections/*.tex`, or git history. Where I needed to know whether a named evidence file exists
on disk I used `stat` only — filenames, sizes and mtimes, never contents.

- **Snapshot sha256** (attested identically by all twelve reviewers): `7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a`
- **Reviewers**: 12 — six codex-hosted (`X1`–`X6`), six claude-hosted (`R1`, `R4`, `R5`, `R-soundness`, `R-stats`, `R-adversary`)
- **0 GPU used.** All arithmetic below was recomputed by me in `/opt/conda/envs/torch-base/bin/python`.

---

## 1. Aggregate, computed from the reviewer headers

| reviewer | overall | conf | recommendation | ceiling | predicted after fixes |
|---|---|---|---|---|---|
| R-claude-soundness | 6 | 4 | weak_accept | 7 | 7 |
| R-claude-stats | 5.5 | 4 | borderline | 6.5 | 6.5 |
| R1-claude-novelty | 5 | 4 | weak_reject | 6.5 | 6.5 |
| R4-claude-clarity | 5 | 4 | weak_reject | 6.5 | 6–6.5 |
| R5-claude-repro | 5 | 4 | weak_reject | 6.5 | 6.0 |
| R-claude-adversary | 5 | 4 | weak_reject | 5.5 | 6.5 |
| X1-codex-novelty | 5 | 4 | weak_reject | 7 | 7 |
| X2-codex-soundness | 4 | 4 | reject | 6 | 6 |
| X3-codex-stats | 4 | 4 | reject | 6 | 6 |
| X4-codex-clarity | 4 | 4 | reject | 6 | 5.5 |
| X5-codex-repro | 4 | 4 | reject | 6 | 6 |
| X6-codex-adversary | 4 | 4 | reject | 6.5 | 6 |

**Overall scores** (n=12): `[4,4,4,4,4,5,5,5,5,5,5.5,6]`
median **5.00** · lower quartile **4.00** · min **4** · max **6** · range **2** ·
mean 4.708 · sample sd **0.690** · IQR 1.00 · MAD about median 0.75.
Confidence is **4 for all twelve** — no reviewer hedged on confidence.

**Recommendations**: reject 5 · weak_reject 5 · borderline 1 · weak_accept 1.
**10 of 12 (83%) sit below borderline.** No reviewer recommended accept or better.

**Dimension medians** (the 10 reviewers who emitted numeric dimension scores; R4 and R5 are prose-only):

| dimension | median | mean | range |
|---|---|---|---|
| novelty | 3.0 | 3.10 | 3–4 |
| significance | 4.0 | 3.70 | 3–4 |
| **technical_soundness** | **2.0** | 2.40 | 2–3 |
| **experimental_rigor** | **2.5** | 2.60 | 2–4 |
| clarity | 4.0 | 3.60 | 3–4 |
| **reproducibility** | **2.0** | 2.30 | 1–4 |
| citation_integrity | 3.0 | 3.20 | 2–5 |
| limitations & responsible claims | 4.0 | 3.60 | 3–4 |

**Issue counts**: 1 fatal, 54 major, 21 minor across the nine machine-readable reviews.
**Ceilings without new experiments**: median 6.5, mean 6.33, min 5.5, max 7. Mean claimed uplift
from fixes is +1.63 — i.e. *every* reviewer believes the paper's problems are repairable without
new GPU time. That unanimity is itself the most important aggregate fact here.

### 1.1 A systematic scoring pattern, not a content pattern

The two reviewer hosts did not overlap in score distribution:

- codex `X*` (n=6): `[4,4,4,4,4,5]`, mean **4.17**, median 4.00
- claude `R*` (n=6): `[5,5,5,5,5.5,6]`, mean **5.25**, median 5.00

A gap of **+1.08 in means with a single point of overlap (5)**. Two things follow, and they point in
opposite directions, so I state both.

First, the two panels **found substantially the same defects**. The calibration defect, the binomial
independence contradiction, the missing evidence, the stale build record, the `recovery_fraction`
gap and the denominator gap appear on both sides. The disagreement is almost entirely in *severity
calibration*, not in *discovery*. So the 1.08-point gap should not be read as one panel being better
informed.

Second, where the panels do differ substantively, the difference is **asymmetric in quality**, and
in a direction that penalises the codex panel:

- The claude panel produced the two **self-retractions** (§4) and the two explicit
  "I am not alleging X" brackets. The codex panel produced none at the finding level.
- Four codex reviewers (X2 `STAT-POWER-NOMENCLATURE`, X3 `ER-03`, X4 `X4-04`, and X6 by reference)
  assert that the abstract's "all 21 evaluated cells are powered" is unestablished or
  unreconstructable. **It is reconstructable and it is true at the conventional bar.** I computed it:
  the worst printed half-width (0.968 pp) gives Φ(1.389/(0.968/1.96) − 1.96) = **0.803**, i.e. 80.3%
  power at the reference effect. R-claude-stats found exactly this, **withdrew its own finding**, and
  narrowed the issue to `tab_power`'s five small-benchmark labels — where it is correct
  (ARC-Easy 55.0%, Winogrande 63.3%, MMLU reference 65.5%; I reproduced all three).

So the codex panel is over-severe on one specific, verifiable point, and the claude panel caught it.
Conversely the claude panel's median of 5 is inflated by R-soundness (6) and R1 (5), neither of which
weighted the technical_soundness consequences as heavily as the deductive argument warrants. **My
meta-score therefore sits below the claude median and above the codex median, not at the pooled
median.**

---

## 2. Consensus strengths (verified, not merely asserted)

1. **The winner's-curse calibration of the floor estimator is a genuine methodological
   contribution, and the authors let it demolish five of their own eight constructs.** Named as the
   strongest verified contribution by R1, R4, R-soundness and X1 independently. That
   `f̂ = max_L m̂_L` is upward biased on a finite item set is obvious once said and, per the panel's
   reading of the cited baseline literature, is not said there.
2. **The §5.1 symmetric-standard self-correction.** The authors noticed their own flip was measured
   with a CI on the floor side and a bare point comparison on the chance side, applied one standard
   to both, and reported that 10/12-vs-1/12 becomes 3/12-vs-1/12 — "about a third of the size".
   R-claude-stats reconstructed both floor-side counts exactly from `tab_mmlupro`. R4 called it "the
   single most consequential retraction in the paper".
3. **Internal arithmetic bookkeeping is unusually good.** R5 verified all 27 rows of `tab_v2_full`
   satisfy Δ_perm = 100·(acc − âcc) to printing precision (0/27 mismatches); every floor in Table 1
   is an exact integer label count over n; gaps and ratios reproduce in 9/9 rows. I independently
   confirmed the shortgpt16 delta (+3.674 pp), the 0.0196 binomial (0.019568), the 40.6-point span,
   and the 15-row expansion of `tab_mmlupro`.
4. **The v2 constant-collapse identity is true and stronger than claimed.** R-claude-stats wrote an
   independent implementation from the paper's equations alone and got Δ_perm = 0 to ≤1.4e-15 for
   all ten pure constant emitters, including letters illegal on most items.
5. **The disclosure apparatus is real, not decorative.** A 16-row retraction ledger with six
   retracted and two *prohibited* numeric claims; a bootstrap-p bug disclosed with measured impact
   (0/24 verdicts); a silent-truncation bug disclosed with the honest note that its benignity was
   unknowable in advance; the v1≥v2 ordering explicitly refused as a theorem with its violating
   emitter constructed. Every reviewer credited this; X6, the harshest codex reviewer, listed it as
   strength 4.
6. **The fp32 control is a falsification of the authors' own earlier mechanism, reported as a
   negative result** (4,303 ties removed, 18.03% of decisions changed, accuracy unmoved,
   McNemar p=0.5702).

---

## 3. Adjudicated issues

I state, for each, what I decided and **what evidence decided it**. "Verified by me" means I
recomputed it in this session from the frozen manuscript.

### 3.1 UPHELD, major — the MMLU-Pro balanced null cannot generate MMLU-Pro data

Found independently by **nine of twelve** reviewers (`ADV-1` fatal, `S-01`, `S1`, `STAT-CONSTRUCT-NULL`,
`ER-01`, `X4-02`, `R2`, `TS-1`, `R5-01`). This is the panel's central finding and it is correct.

**What decided it: a deductive argument that needs no simulation, no `n_opt` vector, and no
unshipped evidence.** `tab_nulls.tex` row 2 prints, on the same line, `Chance = 0.110877` (defined
in the caption and in §A.1.4 as `mean(1/n_opt)`) and `E[f̂] = 0.104460`. Under *any* null in which
each item's gold label is uniform over that item's legal letters, letter A is legal on every item
(n_opt ≥ 3), so `E[m̂_A] = mean(1/n_opt) = 0.110877` exactly. Since `f̂ = max_L m̂_L ≥ m̂_A`, we need
`E[f̂] ≥ 0.110877`. The row prints **`E[f̂] − Chance = −0.006417`**: the expectation of a maximum
printed below the expectation of one of its own arguments. I verified this by inspection of the
frozen table. The row is internally impossible, and 8 of the 9 rows pass the same test — the single
violation is the MMLU-Pro item-average row, the paper's flagship.

**Magnitude, distribution-free, no histogram needed.** `Var(m̂_A) = (1/n)[E p − E p²]`; bounding
`E p²` above by `p ≤ 1/3` and below by Jensen gives `sd(m̂_A) ∈ [0.002479, 0.002862]`. The observed
excess is `0.116606 − 0.110877 = 0.005729`, so `z ∈ [2.00, 2.31]` and the one-sided p **for letter A
alone** is in `[0.0104, 0.0227]`. Since `P(max_L m̂_L ≥ floor) ≥ P(m̂_A ≥ floor)`, the corrected p is
**≥ ~0.010 — at least three orders of magnitude above the printed `<10⁻⁵`**. I computed this myself;
it is histogram-independent and it alone is fatal to the printed p-value.

**Severity: major, not fatal.** I downgrade `ADV-1` from fatal by one notch. The defect is confined
to two table rows, one abstract parenthetical, one introduction bullet, and the Reproducibility
Statement's "load-bearing property #1". It does **not** touch the floor value itself (1403/12032 is
a descriptive label count), the flip counts (ordinal against a fixed floor, calibration-free), v2,
the fp32 control, or the paper's primary rule. R-soundness, R-stats and R5 all scoped it this way
and I agree with their scoping over `ADV-1`'s.

**Blast radius, which I checked and which bounds the finding.** R5 extended the defect to ARC-Easy
and ARC-Challenge on the ground that their printed chance lines (0.250161, 0.250156) are not exactly
1/4, which proves `n_opt` varies there while the generated table prints `k=4`. **R5 is right in kind
and I verified it** — 0.250161 is consistent with ~4–5 three-option items out of 2376. But the
magnitude is `|chance − 1/4| ≈ 1.6e-4`, versus `9e-3` on MMLU-Pro: a correction two orders of
magnitude smaller. So `ADV-1`'s worry that the defect might "affect the majority of the null table"
is **not** borne out. It is one construct at material scale.

**Consequence.** If the corrected p exceeds 0.05, the abstract's "only three of the eight letter
constructs (MMLU-Pro, MMLU, BoolQ)" becomes **two (MMLU, BoolQ)** — and MMLU and BoolQ have fixed k,
so they are provably unaffected (multiple reviewers confirmed their p-values are untouched). The
uncomfortable structural fact, which R-adversary, R-soundness and R-stats all identified: **MMLU-Pro
is the only construct carrying the 14/15 headline, the 21 powered cells, the 3/12-vs-1/12 symmetric
comparison, and all 27 v2 cells.** After correction, the constructs whose floors survive calibration
would not be the constructs on which the flip is measured. That must be said in the paper.

### 3.2 UPHELD, major — the 0.0196 binomial contradicts the paper's own dependence argument

Six reviewers (`ADV-3`, `S3`, `S1`(X1), `STAT-DEPENDENT-COUNT`, `ER-02`, `R3`, `TS-3`, `X4-03`).

**What decided it: the paper contradicting itself eleven lines apart in one file.**
`09a_relocated.tex:26` — "the cells share items, nest arms, and share a null, so the tests are
neither independent nor exchangeable, and an off-the-shelf Bonferroni or Benjamini–Hochberg
adjustment would not have a defensible family definition here." `09a_relocated.tex:37` — "observing
3 or more rejections out of 12 has binomial probability 0.0196 under the global null." Dependence is
invoked to *decline* correction and then ignored to *manufacture* a global p-value from the same
twelve decisions. I verified the arithmetic is right: `P(X≥3 | Bin(12, 1/20)) = 0.019568`.

**The two sharpest constructive forms, both verified by me.** R-adversary: the claim is one cell from
vanishing — `P(X≥2) = 0.118360`, a **6× jump across α**. R-stats: an exchangeable beta-binomial at
the same marginal gives 0.0501 at ρ=0.05 and 0.0678 at ρ=0.10, so any intra-class correlation above
about 0.05 pushes the "surviving" evidence above 0.05 — and shared items across 12 cells on the same
12,032 items make ρ≈0 indefensible.

### 3.3 UPHELD, major — `recovery_fraction` has no interval and its interval straddles its own bar

Five reviewers (`ADV-5`, `S-04`, `S4`, `ER-06`, `TS-5`, `R5-07`).

**What decided it: I propagated the paper's own printed half-width.** From the printed 0.049 and
"9.1% of the intact-family anchor", the implied Qwen3 intact anchor is 0.5385 and the bar is 0.05385
— missed by 0.00485. Propagating `tab_v2_full`'s own half-width on Δ_perm (0.188 on 0.267) through
Δ_max = 5.437 gives a materiality ratio of **[2.7%, 15.5%], which contains 10%**. So "a real but
immaterial exception" is a decision the data cannot make at the paper's own confidence level, and it
is the sentence that disposes of the paper's only above-floor counterexample.

**Compounding, verified.** The implied anchor 0.5385 **appears nowhere in the paper**. The only
printed intact `recovery_fraction` is Llama-2's 0.0545, and `0.049/0.0545 = 89.9%` — so a reader who
tries to verify the step using the one anchor supplied gets a number an order of magnitude off *and
a verdict that inverts*. R-soundness noted the asymmetry precisely: the paper prints the anchor when
it **blocks** a claim and omits it when it **dismisses a counterexample**. R5-07 adds a real
definitional gap: for a pure constant emitter both Δ_perm and the Δ_max headroom are zero, so the
gate statistic is **0/0** on exactly the class of arm the paper cares most about.

### 3.4 UPHELD, major — a designated damaged arm is missing from the headline denominator

Four reviewers (`ADV-2`, `S-03`, `DESIGN-DESIGNATED-DENOMINATOR`, `X4-05`, `R5-08`, `C-06`).

**What decided it: the paper's own promise, quoted verbatim, against its own tables.**
`09a_relocated.tex:24` — "any damaged arm excluded from that denominator is named at the point of
exclusion together with its own floor delta, so that no ratio is computed over an undisclosed
subset." `04_experiments.tex:8` lists OLMo-2 damaged arms as keep8, keep10, keep12, **keep14 and
shortgpt16**. I verified `tab_mmlupro`'s 12 printed rows expand to exactly 15 = 3 OLMo-2
(keep8/10/12) + 12 non-OLMo. keep14 and shortgpt16 are both absent and neither is named at the point
of exclusion.

I verified the stakes: `100·(0.153341 − 0.116606) = +3.674 pp`, which is **15.8×** the acknowledged
exception (qwen3/k14 at +0.233 pp), and `tab_v2_full` gives shortgpt16 `item-level signal` at
p=0.0001 with Δ_perm=+4.054 pp. It clears both nulls decisively. Restoring it makes 14/15 into 14/16
and turns "one materially negligible exception" into two, one of which is large and passes the
arm-conditional gate the paper itself proposes as the stronger test.

**Epistemic credit where due.** Both R-adversary and R1 explicitly bracketed this: "I cannot rule
out that shortgpt16 was excluded for a defensible reason (e.g. it is a different pruning method
rather than a depth rung)". That is the correct posture and it changes the *fix*: disclosure at the
point of exclusion may suffice, and restatement may not be required. But the disclosure rule is
stated and then not honoured for the one arm where honouring it would cost something, and that is a
defect regardless of which fix is chosen.

### 3.5 UPHELD, major but severity-reduced — v1 as a comparability gate

Four reviewers (`TS-VALIDITY-GATE`, `TS-01`, `X4-01`, `TS-2`). The algebra is trivially right and I
confirm it: a common additive constant cannot change a difference, so
`Δ_floor(A) − Δ_floor(B) = acc(A) − acc(B)` exactly, and v1 cannot alter an arm difference or
ordering. `tab_two_nulls.tex` nonetheless prints the v1 question as "**Is this interface valid for
comparing arms?**" That header overclaims and must be narrowed.

**But I reduce the severity, and I preserve the disagreement rather than averaging it.** Three of
the four reviewers write as if the paper never distinguished the absolute from the relative reading.
It did: the abstract says "This floor is a necessary, not sufficient, validity condition", and the
shipped `claim_evidence_map.tsv` row **C-15** reads "*Clearing the floor certifies validity* —
**Retracted** — Clearing is necessary, not sufficient; failure disqualifies, success does not
certify." So the overclaim is real but it is *localised to wording* — a table header and some
framing sentences — in a paper that has already retracted the strong form in its own ledger. This is
a `claim_narrowing` fix of a few sentences, not a soundness collapse. Scoring it as a major
technical_soundness failure (X2, X3, X4, X6 all did, and it is a large part of why their
technical_soundness medians are 2) over-weights it.

### 3.6 UPHELD, major — "0/60" is close to a criterion that cannot fail

**A single-reviewer finding (R-claude-stats `S2`) that I judge to be the strongest statistical
finding in the panel outside the calibration defect.** The paper says twice that "the conclusions
rest on the aggregate's near-unanimity (0/60 damaged cells clear their floor)" while stating in the
same breath that 52/60 are underpowered for the 1.389 pp reference effect. Using `tab_power`'s own
five half-widths (1.305, 2.775, 3.399, 3.882, 6.400), R-stats computed expected detections if
*every* arm truly cleared its floor: +0.5 pp → 3.5/60; +1.0 pp → 7.6/60; **+1.389 pp (the paper's
own reference effect) → 12.0/60**. Observing 0 is therefore consistent with a substantial positive
true effect. It is a failure-to-reject presented as near-unanimity.

I did not have the 60 half-widths, but the five printed ones are all ≥ the reference effect except
ARC-Easy marginally, so the "52/60 underpowered" figure and the expected-detection argument are
mutually consistent, and the finding follows in form from the paper's own numbers. It is not
vacuous — R-stats notes `P(0/60 | δ=+1.0 pp) = 2e-4` — so large effects *are* excluded. But
"near-unanimity" overstates by a wide margin what 0/60 licenses, and this is the paper's declared
substitute for family-wise correction. R-adversary reached a compatible conclusion by a different
route (`P(0 of 60 | zero effect) = 0.219`, 1.5 clears expected). Two reviewers, two methods, same
direction.

### 3.7 UPHELD, major — the 27-cell v2 family gets no correction, and the bidirectionality claim does not survive one

R-claude-stats `S5`. **I re-ran BH at q=0.05 over exactly the 27 printed p-values and reproduce the
reviewer's result exactly**: BH rejects 6 (the five intact/shortgpt16 anchors plus qwen3/k14 at
p=0.0066, threshold 0.01111), and **olmo2/keep14 (p=0.0172, threshold 0.01296) is RETAINED**.
Bonferroni (α/27 = 0.001852) rejects only the 5 anchors, so both "trace signal" cells fail.

This matters because olmo2/keep14 is the **only** cell moving upward, and §5.3's claim that
"re-sorting in both directions is evidence that the criterion is doing more than shrinking effects"
therefore rests on one uncorrected borderline cell. The paper's general defence of not correcting
(no defensible family definition) is weaker here than for the aggregates, because `tab_v2_full`
**is** the family: one table, one null, one statistic, one item set. R4 reached the same substantive
conclusion independently by inspection ("the two 're-sorts upward' are qwen3/k14 (+0.233 → +0.267,
both trivial) and olmo2/keep14 … 'Re-sorting in both directions' is a strong reading of two
borderline moves").

### 3.8 UPHELD, minor — the paper's numeral checker does not cover the numerals it is credited with

The Reproducibility Statement claims "610 numerals with none unresolved". **I verified four
in-snapshot numerals a strict checker should have caught**, all recomputable from the frozen source
alone:

| claim | printed | correct | verdict |
|---|---|---|---|
| `credit`/`wrong` ratio (§A.1.1) | 4.6× | `0.532164/0.125914 = 4.2264` | **wrong**, rounds to 4.2 |
| Table 1 caption gap range | +0.43–+2.60 pp | min over the five scoped rows is **+0.490** (PIQA) | lower endpoint in **no** row |
| CommonsenseQA gap | +0.885 (`tab_nulls`) vs +0.884 (generated) | exact 0.8845, a halfway tie | two shipped tables disagree |
| PIQA balanced-null p | 0.658 | inclusive `P(max ≥ 928) = 0.691725`; strict `P(max ≥ 929) = 0.657647` | matches the **strict** tail, not the **defined** `≥` event |

The PIQA case is the interesting one and it is R-stats `S7` / X2 / X3 / X5 converging. `1838 × 0.504897 = 928.0007`,
so the 6-dp stored decimal sits **above** the realised atom 928/1838 and silently drops it from the
tail. I computed both exact tails with `fractions.Fraction`; the printed 0.658 is the strict tail.
No verdict changes, but the defect class — a rounded threshold evaluated against a discrete atom —
is precisely what the paper's own thesis is about. A ratio, a caption extremum, a halfway rounding
and a test threshold are four distinct grammar gaps in the checker, and none of them requires the
unshipped evidence to demonstrate.

### 3.9 UPHELD, minor — the 60 does not decompose

R-soundness `S-03(b)`, a **single-reviewer** finding. `05_analysis.tex:29` and
`09a_relocated.tex:44`: "only 7/60 are significantly below because 52/60 are underpowered".
**7 + 52 = 59.** One cell is neither and is never named. In a paper that commits explicitly to
naming every excluded cell, this is a small violation of its own stated discipline. I verified the
arithmetic and the verbatim text.

### 3.10 UPHELD, minor — enumeration gaps that block independent re-analysis

- **The 21 cells are never enumerated.** `04_experiments.tex:20` asserts 21 letter-floor cells with
  half-widths 0.083–0.968. I checked both tables: `tab_mmlupro`'s half-widths span 0.083–0.852 and
  `tab_v2_full`'s max is 0.844. **0.968 appears in neither table.** The 21-cell set matches neither
  the 15 nor the 27.
- **`tab_mmlupro` collapses four Llama-3 arms into one range row**, so only 10 of 12 non-OLMo
  p-values are readable and a reader cannot independently redo the BH/Bonferroni check on the floor
  side. R-soundness could run it only on the 10 recoverable values (it reproduced 0/12).
- **The chance-side per-cell deltas appear nowhere.** R-stats flags that the paper's single
  most-quoted result (10/12 point-above and 3/12 CI-above chance) is therefore **not reconstructible
  even in principle** from the submission. It verified the floor side exactly and could not touch
  the chance side. This is the one place where the missing-evidence problem is also a
  *manuscript-completeness* problem.

---

## 4. Self-retractions: I treat these as favourable evidence, and I agree

Two reviewers retracted findings mid-review. I endorse reading both as reliability evidence in the
favourable direction, and in one case the retraction directly corrects four other reviewers.

**R-claude-stats, `S8`.** It initially judged that the abstract's "all 21 evaluated cells are powered
at the scale of the reference effect" was a 50%-power overstatement, then computed that the worst
half-width in the prose range (0.968 pp) reaches **80.3%** power at d = −1.389 pp, **withdrew that
part of the finding**, narrowed the issue to `tab_power`'s small-benchmark labels, and wrote in its
own `review_limitations`: "MY OWN FIRST READING WAS WRONG ONCE AND I CORRECTED IT … Reviewers of
this review should weight S8 accordingly." I recomputed 0.803 and confirm it. The retraction was
against the reviewer's own interest — it demoted a major finding to minor — and it turns out to be
the correction that **four codex reviewers needed and did not make**. This materially raises my
confidence in that reviewer's `S1` (the deductive form of the calibration defect) and `S2` (the
0/60 power argument), which is why I weight both heavily above.

**R-claude-adversary, `ADV-2`**, and **R1** on the same issue. Both wrote explicit non-allegations:
"I want to be precise about what I am NOT alleging: I cannot rule out that shortgpt16 was excluded
for a defensible reason"; "I could not rule out an undocumented reason for its exclusion." Neither
converted an unexplained absence into an accusation of outcome-dependent selection. R4 did the same
on the cohort question ("I am not alleging outcome-dependent selection") and on the build
("I am not alleging a broken build").

**The counter-pattern is worth naming.** Every reviewer wrote a substantive `review_limitations`
block — that is a well-behaved panel overall. But the *finding-level* hedging, and both
self-retractions, came exclusively from the claude side; and the codex side's uniform 4s rest in
part on a power finding that is wrong. A reader of this panel should not average the two hosts.

---

## 5. Unsupported, overstated, or reconstruction-grade reviewer claims

I reject or downgrade the following. Where a reviewer had already flagged the weakness themselves, I
say so — that is the reviewer being reliable, not the finding being sound.

1. **`ADV-1` severity "fatal" → major.** The reviewer's own `review_limitations` says the exact
   legal-support p "is not something I can pin down without the real n_opt vector, and it is
   possible though in my judgement unlikely that the true distribution lands below 0.05." A fatal
   rating resting on a reconstruction whose author says the magnitude is unpinnable is one notch too
   strong. The *deductive* core (E[f̂] < Chance) independently justifies major without any
   reconstruction, so the finding survives at full strength on better grounds than the reviewer's
   own headline number.
2. **All reported corrected p-values for MMLU-Pro (0.061–0.085, 0.064–0.087, 0.068–0.120) are
   reconstructions over inferred `n_opt` histograms and must not be cited as measurements.** All
   three reviewers said so explicitly and prominently. **Direction and existence: established
   deductively. Magnitude: not established.** Only the authors can compute the real value.
3. **X2 `STAT-POWER-NOMENCLATURE`, X3 `ER-03`, X4 `X4-04` — the strong form is refuted.** The
   claim that "all 21 cells are powered" is unestablished/unreconstructable is wrong: it is
   reconstructable from the printed half-width range and it clears 80.3%. The surviving finding —
   that `tab_power`'s yes/no column implements a 50%-power rule and that "yes, borderline" is 55.0%
   — is correct, was reached by R-stats, and is **minor**. The nomenclature point (call it achieved
   precision, not power) stands as a writing fix.
4. **R5's extension of the legality defect to ARC-Easy/ARC-Challenge is correct in kind but must not
   be propagated at MMLU-Pro scale.** I verified the correction there is ~1.6e-4 versus ~9e-3 on
   MMLU-Pro. Reporting it alongside the MMLU-Pro finding without the magnitude would misstate the
   blast radius.
5. **X5 `R5` (missing model/tokenizer revisions, dataset versions, prompts, software, hardware) is
   over-severe at "major".** The Reproducibility Statement does specify `chat_template=False`,
   `add_bos=0`, `desc_style=none`, fp32 master weights with bf16-autocast, batch 48, max length
   2048, and per-shard zero-truncation assertions. The residual request — immutable revisions,
   checksums, lockfiles, hardware — is legitimate supplementary-material work, not a soundness
   defect.
6. **X1 `N2` (Barlow/Lai/Azen 1991 stratified kappa) is not a preemption and the reviewer says so:**
   "I could not confirm that these methods implement the exact item-dependent legal-label blocked
   permutation, so this is not a confirmed preemption." Correctly self-limited. It remains open
   citation work (§7), not a novelty kill.
7. **The Bean et al. checklist count: three reviewers disagree, and I resolve it in R5's favour.**
   R1 verified 27 items and called the claim "precisely correct"; R4 counted 28 and called the paper
   wrong; X1 `C1` also says 28. **R5's reconciliation is the only one that explains both readings**:
   Appendix A has 28 box items (3+3+3+4+3+4+4+4 across eight groups) while the main-text §5 rendering
   shows 27 box glyphs. So the count is section- and version-dependent, and the defect is
   **unspecified provenance**, not a wrong number. All four reviewers independently confirmed the
   load-bearing half of the sentence (no checklist item requests a null, chance level, or constant
   predictor). Downgrade C-04 from "wrong" to "state which version and section you counted".
8. **No reviewer verified a single primary evidence record, and every one of them said so.** All
   twelve `review_limitations` blocks disclose it. Every bootstrap half-width, permutation p-value,
   McNemar result, fp32 tie count, off-MMLU 0/60-25/60-52/60-7/60 count, and the "610 numerals"
   claim is **unverified — neither confirmed nor refuted**. Findings resting on those numbers being
   *wrong* do not exist in this panel; findings resting on them being *uncheckable* do.
9. **Citation integrity scores of 2 (X1, X5) are not supported by what those reviewers checked.**
   Both disclose they had no external access or only partial access. Meanwhile R5 verified five
   venues on OpenReview (`cho2026choices` = ICLR 2026 Poster, `oostermeijer2026length` = ICML 2026,
   `bean2025measuring` = NeurIPS 2025 D&B, `zheng2025cheating` = ICLR 2025 Oral, `arcon2026metalinguistic`
   = arXiv 2602.02182) and R1 verified four local attributions against primary sources
   (Bean, Feng, OLMES, Balepur) and found all four accurate. The median of 3 is right; the 2s are
   scored on absence of access, not on defects found. **R1's genuine citation finding survives and
   is separate**: the flagship BoolQ 0.6217 figure is published in the BoolQ paper and in SuperGLUE's
   most-frequent-class row, and the core floor test is `caret`'s No Information Rate — none cited.
   R1 also notes zero dataset citations for MMLU, MMLU-Pro, ARC, OpenBookQA, CommonsenseQA, PIQA,
   BoolQ. That is real citation work to do.

---

## 6. The two-evidence-file question, reasoned out

**Eleven reviewers scored reproducibility at 1–3 and named the empty evidence directory as the
cause. I decide that this is predominantly a finding about the snapshot, not the paper — but the
paper is not thereby exonerated, and the split matters enormously for the revision plan.**

**What I established (existence and mtime only, contents unread):**

| named record | on disk? | mtime | pre-dates 08-16 11:47 freeze? |
|---|---|---|---|
| `floor_winners_curse_calibration.json` (E-CAL) | yes, 3296 B | 2026-08-14 22:26 | yes |
| `heal_readout_v2_permutation_null.json` (E-D) | yes, 91555 B | 2026-08-13 03:49 | yes |
| `construct_nulls_length_unit.json` | yes, 8253 B | 2026-08-14 15:49 | yes |
| `s2_03_symmetric_inference.json` | yes, 16981 B | 2026-08-15 03:34 | yes |
| `s2_02_stratified_ordering.json` | yes, 9331 B | 2026-08-15 05:13 | yes |
| `evidence/mmlu_scale_power/`, `evidence/second_mc_benchmark/` | both exist | — | — |

**So the records exist, and existed at freeze time.** X2's careful phrasing — "unverified rather
than refuted" — is exactly right. X5's would-lower-score condition, "the missing artifacts do not
exist rather than merely being unshipped", is **refuted**.

**And the omission was not a blinding decision.** `MANIFEST.json` declares
`excluded_by_blindness_rule: [review_rounds, review_history, tcodex_out, SCORE_HISTORY,
review_prompts, WRITER_NOTES]`. **`evidence/` is not on that list.** Worse, the manifest declares
`missing_dependencies: []` — it actively asserted nothing was missing while omitting eight-plus
named records. That is a packaging bug with a lying self-check, not a deliberate withholding.

**Therefore:** the reproducibility *dimension score* (median 2.0, min 1) measures the snapshot. A
correct repackage should move it to roughly 4 at essentially zero cost. **Authors must not read that
as a 2-point score win**, because three reproducibility defects survive repackaging and are
properties of the paper:

- **(a) Table 12 resolves evidence IDs to `tcodex_out/EVIDENCE_PACK.md`** — an internal repository
  path. Even a perfectly packaged snapshot ships a manuscript pointing at a path no external reader
  can resolve. That is a manuscript defect. (X4-07, X6 `TS-4`, X5 `R1` all identified it correctly.)
- **(b) `build_record.json` is stale and *was* shipped.** I verified: the frozen PDF is
  `sha256 1fbaaf99…`, 366583 bytes; the record certifies `56a376e1…`, 355196 bytes, 22 pages. The
  record's mtime (06:48) predates `main.pdf` (10:13) and `main.tex` (09:44), so this is an innocent
  late rebuild — but `build_gate_pass: true` does not attest to the submitted document. Additionally
  the record's own note claims no PDF rasteriser was available on the build host, while reviewers
  used PyMuPDF in this repository's own environment; the `pdf_visually_inspected` gate is closeable
  without new infrastructure.
- **(c) The "610 numerals, none unresolved" claim is about the paper's checker**, and §3.8 shows
  four in-snapshot numerals it does not cover. That is a paper-level defect, fully demonstrable from
  the frozen source, and entirely independent of packaging.

**The consequence that most changes the revision plan:** repackaging does not repair the calibration
defect — it makes it *confirmable from E-CAL*. The authors should assume that shipping E-CAL will let
reviewers verify §3.1 directly rather than by reconstruction. Fix the calibration first; ship the
evidence second.

---

## 7. Genuinely open — I cannot adjudicate these

These are not softenings. They are questions the frozen snapshot cannot answer, and each names what
would answer it.

1. **The exact corrected MMLU-Pro p-value.** Requires the real per-item `n_opt` vector, which only
   the authors hold. *Direction is settled deductively; magnitude is not.* Three reconstructions
   land in 0.06–0.12 and a distribution-free bound gives ≥ ~0.010, but the true value could differ.
   **What would resolve it:** re-run E-CAL with gold drawn uniformly over each item's legal letters,
   same seed and draw count, and publish the per-item `n_opt` histogram alongside.
2. **Whether shortgpt16's exclusion has a construction-based reason.** Two reviewers explicitly
   declined to rule one out. The exclusion may be legitimate (a different pruning method rather than
   a depth rung) — but if so it is undisclosed. **What would resolve it:** the authors stating the
   reason at the point of exclusion, and that reason being demonstrably independent of the measured
   score. I note the risk asymmetry: a reason that *cannot* be stated without reference to
   shortgpt16's score would convert a disclosure defect into a selection defect.
3. **Whether E-CAL was computed with the `n_opt` vector already in hand.** This decides oversight
   versus choice, and therefore decides severity. R-adversary lists it as a would-lower-score
   condition and I agree: the v2 stratification consumes that vector, so it plausibly existed.
   **What would resolve it:** the E-CAL emitter's inputs.
4. **Whether stratified-kappa prior art preempts the varying-k stratification.** X1 found
   Barlow/Lai/Azen (1991) but could not obtain the paywalled article. **What would resolve it:** the
   two estimators written algebraically side by side plus the smallest example where they differ.
5. **Whether the chance-side counts (10/12 point-above, 3/12 CI-above) reproduce.** *No reviewer
   could check the chance side at all* — the per-cell chance deltas are printed nowhere. The floor
   side was verified exactly. The paper's most-quoted result is half-verified. **What would resolve
   it:** three added columns in `tab_mmlupro`.
6. **The canonical Bean et al. checklist count.** Section- and version-dependent (28 in Appendix A,
   27 main-text glyphs). **What would resolve it:** the paper stating which version and section it
   counted.
7. **Whether the manuscript respects the ICLR 2026 9-page main-text limit.** R-stats did not open
   the PDF; X4 inspected all 24 pages and reported appendix float whitespace but did not report a
   limit violation; the build record says 22 pages and is stale. Unresolved by the panel.
8. **Every primary numeric record.** Bootstrap half-widths, permutation p-values, McNemar results,
   fp32 tie counts, all off-MMLU 60-cell counts, the "610 numerals" report, the "bit-identical floor
   across 21 cells" assertion, the 2.11×/0.90×/0.98× residual-fraction ratios. **Unverified, and
   not to be treated as refuted.**

---

## 8. Score ceiling under current evidence, and what limits it

**Ceiling: 6.0.** This is below the panel's median ceiling of 6.5, and I differ from the panel
deliberately.

Every reviewer converged on the same structural claim — that all blocking defects are writing,
re-analysis on existing data, packaging or citation work, with **no new GPU time required**. I agree
and I verified it: not one of the ten upheld issues above needs a new measurement. That is why the
ceiling is well above the current median of 5.

But the panel's 6.5 under-weights a consequence that only R5 stated plainly: **the correct fix will
most likely shrink the headline.** The corrected calibration probably moves MMLU-Pro into the "inside
estimator noise" partition, leaving MMLU and BoolQ as the only constructs with calibrated
floors-above-chance — and **neither carries any of the headline counts**. The paper becomes more
trustworthy and simultaneously smaller. R5 predicted post-revision 6.0 on exactly this reasoning and
I adopt it.

**What limits the ceiling is the evidence, not the writing, not the analysis, and not the artifact.**
This distinction has real consequences for how the authors should spend effort:

- **Writing** is not the limit. The clarity median is 4.0, the highest of the three weak dimensions,
  and R4 — the dedicated clarity reviewer — states that all twelve of its issues are "writing,
  layout, counting, and provenance-plumbing fixes. None requires a GPU."
- **The artifact** is not the limit. Repackaging is ~1 hour of code and lifts reproducibility from 2
  to ~4. Ceiling-neutral beyond that.
- **The analysis** is not the limit. The legality-respecting recalibration, the item-level joint
  null, BH q-values over the 27 cells, `recovery_fraction` bootstrap CIs, integer-count tails, the
  expected-detection calculation for 0/60 — all are re-analysis on data already collected. They
  raise trustworthiness to about 6.
- **The evidence is the limit.** Above ~6.5 requires a *second variable-k construct whose floor
  survives a legality-respecting null*, restoring cross-construct generality to the quantitative
  claim independently of MMLU-Pro. That is new measurement. Both R-adversary and R-soundness
  identified it as the single highest-leverage addition, and both correctly classified it as beyond
  the current evidence.

**Meta-score: 4.5 / 10 — weak_reject.** I place it below the pooled median of 5 and above the codex
median of 4. Below 5 because three of the four reviewers at 5 or above did not run the deductive
`E[f̂] < Chance` test and therefore under-weighted a defect that is verifiable from two columns of
the paper's own table, and because technical_soundness (median 2.0) is *not* a snapshot artifact —
the calibration defect, the binomial contradiction, the straddling materiality gate and the
comparability overclaim are all demonstrable from the manuscript alone. Above 4 because the codex
panel's uniform 4s partly rest on a power finding that is refuted, and because they score a
packaging defect as a paper defect on the paper's weakest dimension. The paper is a careful,
self-falsifying measurement note whose headline calibration is misspecified in the direction that
flatters it — and whose own appendix contains the material to detect that. It is not close to
acceptable as submitted, and it is close to acceptable after re-analysis that costs no GPU.

---

## 9. Prioritised revision plan

**Classes are kept strictly separate.** A defect repaired by re-running an analysis on data already
collected is categorically different from one needing new GPU time, and the ordering below reflects
that. `A1`–`A5` are analysis on existing data. `W1`–`W6` are writing/claim-narrowing. `C1`–`C2` are
code/packaging. `Z1`–`Z2` are citation. `E1` is the only item requiring new measurement, and it is
explicitly *not* required for acceptance.

### Tier 0 — blocking, analysis on data already in hand (no GPU)

| id | issue | action | cost | verification |
|---|---|---|---|---|
| **A1** | §3.1 | Re-run E-CAL for both MMLU-Pro rows with gold drawn uniformly over each item's **own legal** letters, same seed/draws/estimator. Publish the per-item `n_opt` histogram. Add an emitter assertion `E[f̂] ≥ Chance` for every row — it fails on the shipped row and must pass after. | low | assertion fires on old null, passes on new; abstract partition follows the corrected null |
| **A2** | §3.2 | Replace `Bin(12,0.05)` with an item-level joint null: resample items once, recompute all 12 cells per resample, report the rejection-count distribution. Or demote to descriptive. | low | reported p comes from the joint distribution; the sentence containing 0.0196 also carries `P(X≥2)=0.1184` |
| **A3** | §3.3 | Bootstrap `recovery_fraction` and its ratio to the intact anchor inside the existing 10,000 paired resamples, recomputing Δ_perm **and** Δ_max per resample. Print the intact-Qwen3 anchor at point of use. | low | "immaterial" appears only where the interval lies wholly below the bar; my propagation gives [2.7%, 15.5%] |
| **A4** | §3.7 | Add a BH q-value column to `tab_v2_full` and restate §5.3: olmo2/keep14 does **not** survive BH over the 27-cell family. | low | exactly 6 reject at BH q=0.05, 5 at Bonferroni; keep14 in neither |
| **A5** | §3.6 | Accompany 0/60 and 0/15 with the design's expected detection count under a stated alternative (≈12/60 at +1.389 pp), or replace with a pooled effect and interval. | low | reader can see 0/60 excludes δ≥+1.0 pp and does not exclude δ=+0.3 pp |

### Tier 1 — blocking, writing and claim-narrowing (no GPU)

| id | issue | action |
|---|---|---|
| **W1** | §3.1 | Propagate A1 honestly. If p>0.05, the abstract and both null tables carry **two** constructs (MMLU, BoolQ), and the paper states explicitly that the constructs whose floors survive calibration are **not** the constructs on which the flip is measured. Mark MMLU-Pro's `k` column as varying. |
| **W2** | §3.4 | Name shortgpt16 (and keep14) **in the same sentence as the 14/15**, with its +3.674 pp floor delta and its v2 verdict — or include it and report 14/16 with the exception language rewritten to describe two exceptions. If excluding, the reason must be construction-based and stated. |
| **W3** | §3.5 | Narrow v1 from a comparability gate to an absolute above-constant reference. Rewrite the `tab_two_nulls` v1 question. State in one sentence that a common floor leaves pairwise differences invariant. |
| **W4** | §3.8, §3.9 | Correct 4.6× → 4.2×; caption range +0.43 → +0.49; one rounding rule across both null tables; name the 60th cell so 7+52 reconciles. |
| **W5** | §3.10 | Enumerate the 21 cells with per-cell half-widths reproducing 0.083–0.968; expand the collapsed Llama-3 row into four; add per-cell chance-side delta, half-width and p to `tab_mmlupro`. Publish one machine-generated designated-damaged inventory reconciling 15, 60 and their union. |
| **W6** | §5 (X4-08, R4 C-01) | Promote Table 1 and Table 5 into the main text so §5.1's counts are derivable there; define `cell`, `arm`, `recovery_fraction`, `w_s`, `G1`/`G2`, `Δ_max` before use; number and caption the p7 flip table. Relabel `tab_power`'s column as achieved precision, or print power per row (ARC-Easy 0.550, Winogrande 0.633, MMLU 0.655). |

### Tier 2 — code / packaging (no GPU, ~1–2 hours)

| id | issue | action |
|---|---|---|
| **C1** | §6 | Re-emit the snapshot with every record cited by E-A…E-K and E-CAL **inside** it. Make `MANIFEST.json`'s `missing_dependencies` a real check that exits non-zero — it currently reports `[]` while omitting eight-plus named records. Make Table 12 resolve only to snapshot-relative paths; the `tcodex_out/EVIDENCE_PACK.md` target must go. |
| **C2** | §6(b) | Regenerate `build_record.json` against the final PDF (`sha256`, `pdf_bytes`, `pdf_pages`) and make the snapshot builder refuse to ship on mismatch. Close `pdf_visually_inspected` with PyMuPDF, which is present in this repository's environment. |
| **C3** | §3.8 | Extend the numeral checker to a fourth and fifth admissible class — **derived ratios** and **caption extrema** — and compare Monte-Carlo tails in **integer counts**, not rounded rates. Demonstrate it now catches 4.6×, the +0.43 endpoint, and the PIQA strict/inclusive threshold. Only then re-assert "610 numerals, none unresolved". |

### Tier 3 — citation (no GPU)

| id | action |
|---|---|
| **Z1** | Cite BoolQ's own most-frequent-class figure and SuperGLUE's most-frequent-class row for the 0.6217 flagship number, and `caret`'s No Information Rate for the floor test (R1). Add dataset citations for MMLU, MMLU-Pro, ARC-Easy/Challenge, OpenBookQA, CommonsenseQA, PIQA, BoolQ — currently zero. |
| **Z2** | Add the closest MCQA neighbours and state the residual novelty formula-level, not verbally: Cho/NPSQ (X1 `N1`), the ACL 2025 synthesis (`N3`), Molfese et al. Findings-ACL 2025 (`C2`), Pries et al. Dutch Draw (`N4`), Barlow/Lai/Azen 1991 stratified kappa (`N2`, open). Fix the Bean count by stating **which version and section** was counted. Per repository practice: OpenReview `venueid` for ICLR/ICML/NeurIPS, ACL Anthology + DBLP for ACL-family including Findings. |

### Tier 4 — NOT required for acceptance, requires new measurement

| id | action |
|---|---|
| **E1** | Calibrate a **second variable-k construct** under a legality-respecting null and show its floor survives. This is the only route above a ~6.5 ceiling, because it restores cross-construct generality to the quantitative claim independently of MMLU-Pro. It should be scoped as future work, not attempted under revision pressure. |

### Must resolve before acceptance

1. **A1 + W1** — the calibration must respect legal support, and its consequence must be propagated
   to the abstract, the introduction bullet, both null tables and the Reproducibility Statement,
   *even if the honest outcome is a smaller headline*.
2. **A2** — no independence-based global p-value while arguing dependence. Replace or demote.
3. **W2** — the paper's own A.4 disclosure rule must be honoured for shortgpt16 and keep14.
4. **A3** — no categorical "immaterial" verdict from a point estimate whose interval straddles its
   bar.
5. **W3** — remove the inference from a common-floor failure to arm incomparability.
6. **C1 + C2** — the artifact must contain the records it cites and the build record must attest to
   the shipped PDF.
7. **C3 + W4** — the mechanical-binding claim must be either made true or withdrawn.

### Score-gaming suggestions I explicitly reject

- **Do not delete the retraction ledger, the integrity table, the multiplicity disclosure or the
  §5.1 self-correction to reduce the surface area reviewers can criticise.** These are the paper's
  strongest assets and four reviewers named them as such. The panel's low scores come from a
  misspecified null, not from candour.
- **Do not repackage the evidence and treat the reproducibility score as fixed.** Shipping E-CAL
  makes §3.1 *directly confirmable*. Fix the calibration first.
- **Do not keep MMLU-Pro in the surviving-constructs set by arguing the nominal-k null is "a
  different question".** It is printed under a caption that says the null is at "that construct's
  own (n,k)" and is used to license an abstract claim about that construct. If retained, it must be
  labelled a nominal-k null that credits illegal letters, and the abstract claim must be withdrawn
  regardless.
- **Do not respond to the power criticism by deleting the "all 21 cells are powered" sentence.** It
  is true at the 80% bar (80.3% for the worst cell) and one reviewer retracted a finding to confirm
  it. Fix `tab_power`'s labels instead — that is where the defect actually is.
- **Do not narrow the paper by dropping the five inside-noise constructs.** They support the weaker
  and still-useful claim that chance is the wrong *reference*, which the paper already states
  correctly.
