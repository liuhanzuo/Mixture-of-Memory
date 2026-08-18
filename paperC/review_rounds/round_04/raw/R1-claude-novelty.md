# R1-claude-novelty — ICLR 2026 blind review, round 4

- **reviewer_id**: `R1-claude-novelty`
- **role**: novelty_positioning (scored on all rubric dimensions)
- **snapshot_sha256**: `7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a`
- **overall**: 5 / 10 — weak reject (negative borderline)
- **confidence**: 4
- **score ceiling without new experiments**: 6.5

---

## 1. What the paper claims

A measurement-protocol paper. Before comparing two model arms on a multiple-choice
construct, test the arm's score against the construct's *best constant, input-blind*
predictor (the "floor"), not against nominal chance. Four contributions are advanced:

1. **v1, the arm-independent floor** — `f_const = max_L (1/n) Σ 1[y_i = L]`, with an
   asymmetric decision rule (failing disqualifies; passing certifies nothing).
2. **Calibration of the floor estimator itself** — because `f̂ = max_L m̂_L` is a maximum
   over `k` noisy marginals it is upward biased even under exactly uniform labels; a
   balanced-null calibration kills 5 of the paper's own 8 letter constructs.
3. **Four under-specifications of the content floor** — tie convention, length unit,
   tokenizer, and the meaning of "chance" under varying `n_opt`.
4. **v2, an `n_opt`-stratified permutation null** — explicitly disclaimed as the numerator
   of Cohen's κ; only the within-stratum stratification and the gate use are claimed.

Empirically: 15 designated damaged cells across four families on MMLU-Pro (plus five
smaller benchmarks), an fp32-vs-bf16 falsification of a tie mechanism, and a 27-cell v2
re-judgement.

## 2. Strongest verified contribution

**The winner's-curse calibration of the floor estimator, and the fact that the authors let
it demolish five of their own eight constructs.** I re-derived the arithmetic and it holds:
the floor values are exact label counts (`1403/12032 = 0.116606`; `3776/14042 = 0.268908`;
`2033/3270 = 0.621713`), the gaps are `100(Floor − Chance)` to the printed digits, and the
partition into "above balanced null" (MMLU-Pro, MMLU, BoolQ, `p<1e-5`) versus "inside
estimator noise" (five constructs, `p = 0.140–0.853`) is stated consistently in the
abstract, the intro bullet, Table 2 and Table 10. On CommonsenseQA and PIQA the observed
floor is genuinely *below* `E[f̂]` and the caption says so. Papers that quantify the bias
of their own headline estimator and then retract the quantitative form of their own claim
are rare; this is the part of the submission I would want preserved.

I also verified the two structural arguments the paper is careful about:

- **The 36-item stratification gap.** `Σ_s w_s max_L m_{s,L} = 1439/12032` versus
  `f_const = 1403/12032`; the difference is exactly 36 items = 0.2992 pp, as stated in
  §3.2 and App A.3. The paper correctly refuses to call the v1 ≥ v2 ordering a theorem,
  states the regularity condition, and exhibits the `n_opt`-conditional emitter that
  attains `f_const + 0.299` pp. This is unusually disciplined.
- **The binomial back-stop.** `P(X ≥ 3 | n=12, p=0.05) = 0.019568`, matching the quoted
  `0.0196`.

## 3. Positioning problems (my primary lens)

### 3.1 The flagship illustrative number is a published number, uncited

The intro's first "main finding" bullet reads: *"The null is not 'chance.' BoolQ's best
constant is 0.6217 rather than 0.50."* This is the largest floor-vs-chance gap in the paper
(+12.171 pp, 1.2434×) and it carries the rhetorical weight of the opening claim.

`2033/3270 = 0.6217125…`. I downloaded Clark et al. (2019), *BoolQ: Exploring the
Surprising Difficulty of Natural Yes/No Questions* (arXiv 1905.10044; verified at ACL
Anthology `N19-1300`). §5.1 of that paper states verbatim: *"the majority-class baseline
accuracy (62.17% on the dev set)"*. The BoolQ paper further uses that baseline exactly the
way this submission recommends — as a reference that shallow models and question-only /
passage-only partial-input models fail to beat (*"the pre-trained BERT_L model reached
64.48% dev set accuracy using just the question … Given that the majority baseline is
62.17%, this suggests there is little signal in the question by itself"*).

The same number is also in the SuperGLUE baseline table (Wang et al., 2019, arXiv
1905.00537): the **"Most Frequent Class"** row reports BoolQ 62.3 (test) / 62.2 (dev),
alongside a full row of most-frequent-class baselines for every task.

Neither BoolQ nor SuperGLUE is in `refs.bib` (17 entries, no dataset papers at all). So the
paper's headline demonstration that "the null is not chance" is, for its most vivid
construct, a re-derivation of a number published in that dataset's own paper and shipped in
a standard benchmark's baseline table. That is a *reproduction presented as a finding*.

### 3.2 The core method is standard classification practice, uncited

"Compare accuracy to the largest-class rate and test the difference" is the **No Information
Rate** in `caret::confusionMatrix`, documented as: *"a one-sided test to see if the accuracy
is better than the 'no information rate,' which is taken to be the largest class percentage
in the data"* (verified against the package documentation; Kuhn 2008, *JSS* 28(5), verified
via Crossref `10.18637/jss.v028.i05`). That is read-out v1 plus a significance test,
shipped in a mainstream tool since 2008 and applied by default to every confusion matrix.

The paper's Related Work cites Balepur et al. (majority-class for MCQA), Zheng et al. 2025
(null models vs judges), Zheng et al. 2024 (PriDe), OLMES, Cho et al., Feng et al., Hewitt
& Liang, Ding et al., Bean et al. — and five kappa-family papers for v2. It never touches
the classification-metrics literature where the arm-independent floor test already lives.
The consequence is that the paper's framing ("the missing operational item") reads as
*newly invented* when what is actually true is narrower and still publishable: **the item is
missing from LLM benchmark reporting practice, not from measurement methodology.** The
paper should say the latter.

### 3.3 The headline phenomenon is a replication in a new regime — which the paper
partly admits

The wrong-null flip is exactly the pattern Arčon et al. report. I fetched arXiv 2602.02182
and the abstract states: *"Although all models perform above chance, they fail to outperform
the majority-class baseline."* The paper cites this openly and says the preprint *"does not
turn it into a pre-comparison protocol"* — honest, and I credit it. But it does cap the
empirical novelty: the flip itself is published, and this submission's addition is the
damaged-arm regime plus protocolization. After §3.1–§3.2, the residual genuinely-new content
is items (2) and (3) from §1 above, plus the `n_opt` stratification, which the paper already
scopes correctly.

### 3.4 What the paper does well on positioning

I checked the differentiations and they are real rather than decorative:

- **Bean et al.** — I extracted all 27 checklist items across the 8 recommendations of
  arXiv 2511.04703 and confirmed that **none** asks for a null, chance level, or constant
  predictor. §5.2's prose does mention subtask baselines, but that is not an input-blind
  reference and is not a checklist item. The claim "27 actionable checklist items, but none
  asks authors to report a null" is precisely correct. (Venue also verified: NeurIPS 2025
  Datasets & Benchmarks poster, `venueid NeurIPS.cc/2025/Datasets_and_Benchmarks_Track`.)
- **Feng et al.** — the paper says they *"construct a dataset where a partial-input baseline
  is at chance while artifacts remain exploitable"*. Confirmed in P19-1554 §3.1: the
  label-in-premise construction makes *"the best accuracy from any hypothesis-only model …
  chance"* while a full-input model is perfect. The "necessary, not sufficient" framing is
  correctly attributed.
- **OLMES** — the paper says OLMES takes the better of MCF/CF per task-model and frames its
  reference around a random baseline. Confirmed in the camera-ready: *"we standardize to
  evaluate each model using both the MCF and CF formulations, and the best performing one is
  used … (where MCF scores hovering around random baseline)"*. No majority/label-marginal
  floor appears anywhere in OLMES. The paper's "This is not a size-keyed rule" correction is
  accurate.
- **Balepur et al.** — footnote 4 of 2402.12483 defines the majority-class baseline as
  always predicting the most frequent choice; the invalid-output imputation of 0.25 is in
  §2. Both cited details check out. The differentiation (dataset cheatability under
  black-box generation vs a per-arm gate for likelihood-scored constructs) is thin but real.
- **v2 prior art** — all five kappa-family citations verified via Crossref (Bennett/Alpert/
  Goldstein 1954 POQ 18(3):303; Brennan & Prediger 1981 EPM 41(3):687–699; Frary 1988 EMIP
  7(2):33–38; Brenner & Kliebsch 1996 *Epidemiology* 7(2):199–202; De Vries et al. 2008
  *Field Methods* 20(3):272–282; Cohen 1960 EPM 20(1):37–46). The disclaimer *"we claim none
  of it"* followed by the specific `k`-varies-per-item boundary is the right way to write
  this paragraph.

**All venues verified with the correct authority.** ACL family via Anthology: Balepur
`2024.acl-long.555` pp. 10308–10330 (ACL 2024 main); OLMES `2025.findings-naacl.282`
pp. 5020–5048 (Findings of NAACL 2025 — matches the bib exactly, no CoRR fallback);
Hewitt `D19-1275` pp. 2733–2743 (EMNLP); Feng `P19-1554` pp. 5533–5538 (ACL). OpenReview
family via `venueid`: Cho et al. `ICLR.cc/2026/Conference`, venue "ICLR 2026 Poster", with a
`Camera_Ready_Revision` invitation; Oostermeijer `ICML.cc/2026/Conference` ("ICML 2026
regular"); Zheng 2025 `ICLR.cc/2025/Conference` ("ICLR 2025 Oral"); Zheng 2024
`ICLR.cc/2024/Conference` ("ICLR 2024 spotlight"). Ding et al. via DBLP: NeurIPS 2021
pp. 1556–1568 (Conference and Workshop Papers, **not** CoRR). Arčon et al. is correctly
labelled `@article … arXiv preprint`. **Zero fabricated or mis-venued references.** I note
Cho et al. is same-cycle concurrent work; the paper's "independently study" phrasing is
appropriate.

## 4. The most severe unresolved issue: an above-floor damaged arm dropped from the
denominator without being named

§4 defines the designation and makes a specific promise, restated in App A.4:

> *"OLMo-2 arms are prune-then-heal checkpoints, including `keep8`, `keep10`, `keep12`,
> `keep14`, and `shortgpt16`."* (§4)
>
> *"Where a denominator such as 14/15 or 10/15 is quoted, it is over exactly this set, and
> any damaged arm excluded from that denominator is named at the point of exclusion together
> with its own floor delta, so that no ratio is computed over an undisclosed subset."*
> (App A.4)

Reconciling the cell counts: the 21 MMLU-Pro letter-floor cells = 15 designated damaged
(Table 5: 4 Llama-2 + 4 Llama-3 + 4 Qwen3 + 3 OLMo-2) + 4 intact + `keep14` + `shortgpt16`.
So `shortgpt16` **is** evaluated on MMLU-Pro, **is** listed in §4 as a structurally pruned
OLMo-2 arm, and **is** excluded from the 15.

Table 9 (`tab_v2_full`) gives it: accuracy **0.153341**, `Δ_perm = +4.054` pp,
half-width 0.575, `p = 0.0001`, verdict **"item-level signal"**. Against the bit-identical
always-A floor 0.116606 its v1 floor delta is **+3.674 pp** — **15.8× the +0.233 pp
"exception"** the paper reports, and far outside any plausible half-width. On the paper's own
criterion this is a designated damaged arm that clears its floor *and* carries confirmed
item-level signal.

`shortgpt16` appears exactly twice in the entire manuscript: once in §4's arm list, once as a
row in Table 9. It is never named at the point of exclusion, its v1 floor delta is never
printed, and it is never discussed. (`keep14` is also excluded from the 15, but at least
surfaces in Table 6 with `+0.324` pp, `p = 0.3234`, "at floor" — a materially different
case.)

This damages three headline statements:

- *"the honest aggregate is 14/15 at or below the floor"* (abstract) — the honest aggregate
  over §4's stated designation set includes at least one more above-floor arm.
- *"0/60 damaged cells clear their floor"* (§4, §5.1, App A.4) — the load-bearing
  near-unanimity on which the paper explicitly rests its conclusions in place of a
  multiplicity correction.
- *"a real but immaterial exception"* (§5.1) — with `shortgpt16` in the set, one exception is
  **material** by the paper's own 10 % recovery-fraction bar, and the narrative "damage
  drives the letter score to or below its floor" (ledger row 2) needs an explicit
  ShortGPT-shaped carve-out.

I want to be precise about severity. This does **not** invalidate the reporting rule, which
is the paper's actual contribution and survives intact. It invalidates the specific
*honest-count* claim and the anti-cherry-picking argument — and it does so at exactly the
point where the paper asserts the counts "cannot be inflated by selecting on outcome."
That is why I grade it major and treat it as decision-blocking in the current form: the
promise in App A.4 is a strong one and this arm violates it.

## 5. Second major issue: the symmetric-standard table is not symmetric

§5.1's fix for an asymmetry is a two-row table with a shared column header
"CI₉₅ excludes 0", and the surrounding text is explicit about the criterion:

> *"the criterion is the two-sided 95 % interval on **both** sides, matching the verdict
> rule already used for the floor."*

The floor row reports **1/12**. But Table 5's own two-sided bootstrap mid-`p` values give
three sub-0.05 cells: `qwen3/k14` `p = 0.0192` (above), `llama2/k8` `p = 0.0168` (below),
`qwen3/k8` `p = 0.0362` (below). Under a criterion that counts both sides, the floor
reference has **3/12** rejections — identical to the chance row's 3/12, i.e. **no flip at
all** under the stated symmetric standard. The printed 1/12 is reachable only by counting
above-side rejections for the floor while counting both sides for chance, which is the very
asymmetry the subsection was added to remove.

(Caveat: percentile CIs and two-sided mid-`p` need not agree exactly at the boundary, so
`p = 0.0362` could conceivably pair with a CI that grazes zero. That is exactly why the
comparison needs to be re-emitted from the CI arrays rather than inferred.)

Note also that App A.5 already concedes 0/12 both sides under BH or Bonferroni, so the
decision-relevant residual of the paper's central empirical contrast is: 3/12 vs 3/12
uncorrected, 0/12 vs 0/12 corrected. The abstract's *"the null choice alone reverses the
reading"* is stronger than that.

## 6. Minor issues

- **"Ten target constructs" is unreconcilable.** The abstract says "ten target constructs
  and one negative control" and "eight of the … letter constructs". §4 lists 8 letter
  datasets + Winogrande control, and content-side longest-option on MMLU, MMLU-Pro **and the
  five non-MMLU tasks** — i.e. 7 content constructs, hence 15 (construct, interface) pairs,
  not 10. §4's parenthetical about OpenBookQA character/token not being double-counted does
  not close the gap. Table 2 has 9 rows for 8 constructs (MMLU-Pro twice) and no Winogrande.
  No verdict depends on this, but it is the abstract's first scope number.
- **`build_record.json` describes a different PDF than the one shipped.** The record asserts
  `pdf_pages 22`, `pdf_bytes 355196`, `pdf_sha256 56a376e1…`. The shipped
  `manuscript/main.pdf` is 24 pages, 366583 bytes, `1fbaaf99…` (matches MANIFEST). I
  recompiled the shipped source with the repository's TeX Live 2026 latexmk into a scratch
  directory: rc=0, **24 pages, 366583 bytes**, 0 overfull hbox, 0 undefined citations, 0
  undefined references. So the *substance* of the build gate holds for the shipped source —
  but the shipped provenance record certifies a stale artifact, and it also carries
  `pdf_visually_inspected: false`.
- **Zero dataset citations.** MMLU, MMLU-Pro, ARC-Easy/Challenge, OpenBookQA, CommonsenseQA,
  PIQA, Winogrande, BoolQ all appear as measured objects with no reference. For a paper whose
  entire subject is *how benchmark scores should be reported*, this is a conspicuous omission.
- **§3.3 is a summary whose development sits after the bibliography.** The `main.tex`
  comments state the page-budget motivation. The summary is self-contained and names all
  four degrees of freedom with every headline number, so I do not treat this as
  information-hiding — but a reader meeting Table 3's 0.933702 Winogrande `credit` floor
  before A.1.1 will be confused.
- **fp32 falsification is single-cell.** OLMo-2 `keep8` on MMLU only. §5.4 and ledger row 3
  are correctly scoped ("mechanisms are family-specific"), so this is a scope note rather
  than an overclaim.

## 7. Weakest evidence-to-importance ratio

The **"wrong-null verdict flips"** claim (§5.1, abstract). It gets the most prominent
framing and the largest share of the abstract, and after the paper's own two qualifications
plus the §5 issue above it reduces to 3/12 vs 3/12 uncorrected and 0/12 vs 0/12 corrected,
in a regime that mixes prune-then-heal (OLMo-2) with evaluation-time truncation (non-OLMo)
so that family and regime are confounded by construction — which the paper labels honestly
but which nonetheless means the 12-cell majority of the evidence comes from arms that were
never trained in the damaged configuration at all.

## 8. Single change that would most improve confidence

Re-emit the designated-damaged denominators from the shipped evidence with an explicit
inclusion/exclusion table listing every arm in `tab_v2_full`, its v1 floor delta, and — for
each exclusion — the stated reason. `shortgpt16` must appear with `+3.674` pp. Then rewrite
"14/15", "0/60", and "one immaterial exception" to whatever the audited set actually
supports.

## 9. Scores

| Dimension | Score |
|---|---|
| Novelty | 3 |
| Significance | 3 |
| Technical soundness | 3 |
| Experimental rigor | 3 |
| Clarity | 3 |
| Reproducibility | 4 |
| Citation integrity | 3 |
| Limitations & responsible claims | 4 |

**Overall 5 — weak reject.** The paper is careful, self-falsifying and unusually well
provenanced; those are real virtues and I do not want them read as faint praise. But two
material issues block a positive decision as submitted: a designated damaged arm that clears
its floor by 3.674 pp is dropped from the headline denominators without the disclosure the
paper's own appendix promises, and the table introduced to remove an evidentiary asymmetry
appears to retain it. On positioning, the flagship number (BoolQ 0.6217) is published in the
BoolQ paper and in SuperGLUE's most-frequent-class row, and the core floor test is `caret`'s
No Information Rate — none of which is cited.

Every required fix is writing, re-aggregation from shipped arrays, and citation work. No new
experiments are needed, which is why my ceiling (6.5) and my predicted post-revision score
(6.5) coincide. A version that names `shortgpt16`, recomputes the symmetric comparison
consistently, and repositions against NIR / SuperGLUE / BoolQ would be a solid, honest
measurement note whose contribution is the floor-estimator calibration and the four
under-specifications — not the flip.

## 10. Review limitations

- I did not have the per-item prediction records, the bootstrap CI arrays, or the
  `floor_winners_curse_calibration.json` / `heal_readout_v2_permutation_null.json` files.
  Only `build_record.json` and `claim_evidence_map.tsv` shipped, so every table value was
  checked for internal and arithmetic consistency, not against raw predictions. The `p<1e-5`
  balanced-null values and all half-widths are **unverified**.
- I inferred `shortgpt16`'s v1 floor delta as `0.153341 − 0.116606` using the paper's own
  bit-identical always-A floor. I could not verify its v1 bootstrap `p` or half-width, so
  "clears its floor significantly" rests on 3.674 pp being far outside its v2 half-width of
  0.575. I could not rule out an undocumented reason for its exclusion.
- The 3/12-vs-1/12 discrepancy is inferred from Table 5's mid-`p` values, not from the CI
  arrays themselves.
- I did not search for prior work on winner's-curse correction of majority-baseline
  estimators beyond arXiv and Crossref queries; absence there is weak evidence.
- I did not attempt to diff the Cho et al. camera-ready against arXiv v4 (the paper's own
  Limitations already discloses that gap).
- No visual page-by-page inspection of the rendered PDF beyond text extraction and
  heading/section mapping.
