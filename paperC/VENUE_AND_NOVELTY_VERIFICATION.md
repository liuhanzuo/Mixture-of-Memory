---
task: close paperC's two remaining NON-COMPUTE open defects
date: 2026-08-12
compute: web + CPU only, **ZERO GPU**, no SSH to any compute node
scope: (1) verify the three UNVERIFIED venues at the family-correct authority;
       (2) full-text PDF pass on every novelty candidate
verdict_defect_1: ALL THREE VENUES VERIFIED (one citation error found and corrected)
verdict_defect_2: FULL-TEXT PASS DONE ON 9/9 CANDIDATES — no candidate preempts;
       two prior-work DEFECTS found that paperC can correct; one paperC
       mis-description of OLMES found and must be fixed
---

# paperC — venue verification + full-text novelty pass

> ⚠️ **This file was written by a subagent that was instructed NOT to edit
> `paperC/README.md`** (a sibling agent held that file). §0 below is the exact
> replacement text for the two closed defect bullets — **MAIN must splice it in.**
> §0c is a *third*, unrequested correction: paperC currently mis-describes OLMES's
> interface-selection rule, and the full-text pass found the actual rule. That one
> is load-bearing for a positioning claim, so it is offered as replacement text too.

---

## 0. Exact replacement text for `paperC/README.md` (MAIN: splice these)

### 0a. REPLACE these two bullets at the end of `## Citation obligations inherited from the novelty check`

Current text to remove (README lines ~157–162):

```
- ⚠️ Three venues still **UNVERIFIED** at the Anthology/OpenReview level
  (Ding et al. NeurIPS 2021, Hewitt & Liang EMNLP 2019, Feng et al. ACL 2019) —
  verify per `memory/venue-verify-acl-family-needs-anthology` before submission.
  S2 and DBLP were both DOWN (HTTP 429 / 500) during the novelty check.
- No full-text PDF pass has been done on any candidate. All overlap judgements are
  from title + abstract + venue metadata.
```

Replacement:

```
- ~~Three venues still **UNVERIFIED** at the Anthology/OpenReview level.~~
  **CLOSED 2026-08-12** — all three verified at the family-correct authority; see
  [`VENUE_AND_NOVELTY_VERIFICATION.md`](VENUE_AND_NOVELTY_VERIFICATION.md) §1.
  **Ding et al.** = NeurIPS 2021 Poster, *Advances in NeurIPS* 34, pp. 1556–1568,
  OpenReview `venueid = NeurIPS.cc/2021/Conference` + DBLP `conf/nips/DingDS21` +
  the official proceedings page. ⚠️ **The camera-ready title differs from arXiv's:**
  cite "Grounding Representation Similarity **Through** Statistical Testing"
  (arXiv:2108.01661 says "**with**"). **Hewitt & Liang** = EMNLP-IJCNLP 2019 **main**,
  Anthology `D19-1275`, DOI `10.18653/v1/D19-1275`, pp. 2733–2743, DBLP
  `conf/emnlp/HewittL19` (`booktitle = EMNLP/IJCNLP (1)` = main volume, not Findings —
  Findings did not exist in 2019). **Feng et al.** = ACL 2019 **main**, Anthology
  `P19-1554`, DOI `10.18653/v1/P19-1554`, pp. 5533–5538, DBLP `conf/acl/FengWB19`
  (`booktitle = ACL (1)` = main). All three are peer-reviewed main-track; none is a
  workshop or preprint.
- ~~No full-text PDF pass has been done on any candidate.~~ **CLOSED 2026-08-12** —
  all nine candidates read in full (camera-ready where obtainable); see
  [`VENUE_AND_NOVELTY_VERIFICATION.md`](VENUE_AND_NOVELTY_VERIFICATION.md) §2.
  **No candidate preempts.** 0 of 9 computes a best-constant/input-blind null
  per-construct as a *precondition on arm comparison*; **none** reports any of
  paperC's floors (`0.2689` / `0.2845` / `0.6217` / `0.3635` / `0.116606` appear in
  zero candidate PDFs); only OLMES touches BoolQ and only Oostermeijer touches OBQA.
  Two of the mandatory citations contain **defects paperC can correct**: Balepur et al.
  impute `0.25` ("random guessing") for invalid outputs (ACL p. 10310) inside the very
  experiment that argues chance is the wrong reference — under their own MMLU letter
  marginal the correct imputation is `0.2689`; and OLMES's interface diagnostic is
  stated against "random", never against a label-marginal floor (see the corrected
  OLMES bullet above). One residual gap: **Cho et al.'s ICLR camera-ready PDF could
  not be fetched** (OpenReview `/pdf` is behind a bot challenge from this network), so
  the full-text read is of **arXiv v4 (2026-01-12)**, two weeks before the
  camera-ready `pdate` 2026-01-26. arXiv-vs-camera-ready was diffed successfully for
  Balepur (no substantive change).
```

### 0b. OPTIONAL one-line addition to the same section (provenance)

```
- Venue/full-text provenance: `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` (the
  authority actually queried is named per paper; the PDFs were read, not the abstracts).
```

### 0c. ⚠️ CORRECTION — REPLACE the OLMES bullet (README line ~149–151)

Current text to remove:

```
- **OLMES, Findings of NAACL 2025** (arXiv:2406.08446) — origin of the letter/cloze
  interface split. Position the floor test as a FIX to a defect in OLMES's
  SIZE-keyed interface-selection rule.
```

Replacement (the "SIZE-keyed rule" does not exist in OLMES — see §2.6 below):

```
- **OLMES, Findings of NAACL 2025** (arXiv:2406.08446, `2025.findings-naacl.282`) —
  origin of the letter(MCF)/cloze(CF) interface split. ⚠️ **Do NOT describe OLMES's
  rule as SIZE-keyed** — corrected 2026-08-12 after a full-text read. OLMES's actual
  rule is **max-over-interfaces, per task per model**: "we standardize to evaluate each
  model using both the MCF and CF formulations, and the best performing one is used"
  (p. 5026; Table 7's `max` column, p. 5038). Model size is only their *narrative* for
  why the max lands where it does, not the selection key. The correctly-stated defect,
  which is a *stronger* claim: OLMES's only reference line for "is MCF meaningful for
  this model" is **"random"** — the string "chance" occurs 0 times in the paper and a
  label-marginal/majority null occurs nowhere — and its Part-1 discussion asserts
  without measuring that a model preferring one label "would not be much better than
  random" because "the benchmarks in OLMES are generally balanced" (p. 5035). paperC
  measures exactly that quantity and it is **not** the chance line: always-B on
  ARC-Challenge is `0.265358` vs `0.250156`, always-A on OpenBookQA is `0.276000` vs
  `0.25`, always-A on MMLU-Pro is `0.116606` vs `0.100000` (**1.1661×**). Position the
  floor test as (i) supplying the null OLMES's own robustness argument presupposes, and
  (ii) noting that max-over-interfaces is an *uncalibrated* selection — it can report a
  number from an interface that does not clear its own floor, because no floor is ever
  computed. Also: OLMES dismisses the tokenizer objection to per-token normalisation
  ("does not seem like a relevant argument", p. 5037) — that dismissal is **valid in its
  own scope** (ranking choices at fixed model+tokenizer) and does **not** cover paperC's
  finding, which is that a *content floor compared across models* is tokenizer-dependent.
  Say so explicitly rather than presenting OLMES as simply wrong.
```

---

## 1. DEFECT 1 — the three UNVERIFIED venues

Method: family-routed per `memory/venue-verify-acl-family-needs-anthology` +
`memory/venue-verify-must-use-openreview-2026`. ACL family → ACL Anthology + DBLP.
OpenReview family → OpenReview `venueid`. All fetches through
`http://hy-proxy.woa.com:3128` on 2026-08-12. **Anthology, DBLP and OpenReview were all
UP this session** (the 429/500 outage that caused the original gap has cleared); S2 was
not needed because the family-correct authority answered in every case.

### 1.1 Ding, Denain, Steinhardt — NeurIPS 2021 → **VERIFIED, and the title was wrong**

| authority | identifier | result |
|---|---|---|
| **OpenReview** (family authority) | forum `_kwj6V53ZqB`, `invitation = NeurIPS.cc/2021/Conference/-/Blind_Submission` | `venue = "NeurIPS 2021 Poster"`, **`venueid = NeurIPS.cc/2021/Conference`** |
| DBLP (cross-check) | `conf/nips/DingDS21` | `booktitle = NeurIPS`, `pages = 1556-1568`, `crossref = conf/nips/2021` |
| Official proceedings (cross-check) | `proceedings.neurips.cc/paper/2021/hash/0c0bf917c7942b5a08df71f9da626f97` | HTTP 200, `<title>` and official bibtex `NEURIPS2021_0c0bf917` |

**Verdict: NeurIPS 2021 main conference, Poster. Peer-reviewed. Not a workshop.**

⚠️ **Citation error in the current novelty check.** `NOVELTY_CHECK.md` §2.7 cites the
**arXiv** title, "Grounding Representation Similarity **with** Statistical Testing".
The camera-ready title is "Grounding Representation Similarity **Through** Statistical
Testing" — confirmed by three independent records: the OpenReview NeurIPS note, DBLP
`conf/nips/DingDS21`, and the official NeurIPS bibtex. The arXiv PDF's own body still
prints "with" (I checked page 1 of `arXiv:2108.01661v2`), and arXiv's
`citation_title` meta tag also says "with" — so this is exactly the
arXiv-vs-camera-ready title drift the ACL-family memory warns about, occurring here in
the **NeurIPS** family. **Cite "Through".**

> ⚠️ Note the OpenReview API returns *three* records for this paper: the real NeurIPS
> note (`_kwj6V53ZqB`), a DBLP-mirrored NeurIPS record (`FifdmcZVOV`,
> `venueid = dblp.org/conf/NIPS/2021`), and a DBLP-mirrored **CoRR** record
> (`Sov1eh9hYog`, `venueid = dblp.org/journals/CORR/2021`, title "with"). Only the
> first has a `NeurIPS.cc/...` venueid. Reading the third one alone would produce a
> false "preprint" verdict — the same failure mode as `venue-verify-must-use-openreview-2026`,
> in mirror form. Also: the API2 endpoint (`api2.openreview.net`) returns **nothing
> relevant** for this 2021 paper; API1 (`api.openreview.net`) is required for pre-2023
> venues, and `notes?id=` is blocked by a bot challenge while `notes/search?query=` works.

**Full citation as it should appear:**
> Frances Ding, Jean-Stanislas Denain, Jacob Steinhardt. "Grounding Representation
> Similarity Through Statistical Testing." *Advances in Neural Information Processing
> Systems 34 (NeurIPS 2021)*, pp. 1556–1568, Curran Associates, 2021.
> arXiv:2108.01661 (arXiv title differs: "…with Statistical Testing").

**Is the README's description accurate? YES.** README/`NOVELTY_CHECK.md` §2.7 says it
"establishes sensitivity/specificity testing for CKA-style measures". Verified from the
PDF, p. 1: "measures should have *sensitivity* to changes that affect functional
behavior, and *specificity* against changes that do not." Accurate. One refinement worth
making: **the word "null" does not appear in the paper** (0 hits for `null|permutation`),
so it is prior art for *statistical-testing framing of similarity measures*, not for a
permutation null as such — which strengthens, not weakens, A01's surviving
"we are not aware of a prior layer-order null" language.

### 1.2 Hewitt & Liang — EMNLP 2019 → **VERIFIED (main track)**

| authority | identifier | result |
|---|---|---|
| **ACL Anthology** (family authority) | `https://aclanthology.org/D19-1275.bib` (HTTP 200) | `booktitle = "Proceedings of the 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP)"`, `pages = 2733--2743`, `doi = 10.18653/v1/D19-1275`, Hong Kong, Nov 2019 |
| DBLP (cross-check) | `conf/emnlp/HewittL19` | `booktitle = EMNLP/IJCNLP (1)` → **volume 1 = main**, `pages = 2733-2743`, `crossref = conf/emnlp/2019-1` |

**Verdict: EMNLP-IJCNLP 2019 main conference (long paper). Not Findings** — Findings
did not exist until 2020, and DBLP's `(1)` marks the main volume. Anthology ID prefix is
`D19-` (the pre-2020 EMNLP main-track prefix), not `D19-…findings`.

**Full citation:**
> John Hewitt, Percy Liang. "Designing and Interpreting Probes with Control Tasks."
> *Proceedings of the 2019 Conference on Empirical Methods in Natural Language
> Processing and the 9th International Joint Conference on Natural Language Processing
> (EMNLP-IJCNLP)*, pp. 2733–2743, Hong Kong, China, Nov 2019.
> DOI `10.18653/v1/D19-1275`. arXiv:1909.03368.

PDF-title vs metadata-title: **identical** ("Designing and Interpreting Probes with
Control Tasks", PDF p. 1) — no drift here.

**Is the README's description accurate? YES, with one sharpening.** README calls it
"control tasks / selectivity, the canonical 'your probe needs a null' result". Verified
p. 2733–2734: control tasks "associate word types with random outputs"; selectivity =
"the difference between linguistic task accuracy and control task accuracy" (Fig. 2).
Sharpening worth making in the writeup: Hewitt & Liang's control is a **randomised-label
task** (a null over the *supervision*), not an input-blind constant over the *input* —
and crucially their §4 uses selectivity to **overturn a layer comparison** (ELMo1 97.2
acc / 26.0 sel vs ELMo2 96.6 acc / 31.4 sel, Table 2, p. 2740: "does ELMo1 have a better
grasp of part-of-speech than ELMo2? … the alternative hypothesis"). That is the closest
*structural* precedent in the literature for paperC's move — a null that reverses a
comparison between two components of one model — and paperC should cite it for exactly
that, not merely as "probes need a null". It is on a different construct (probe accuracy,
not MC accuracy) and has no best-constant/input-blind baseline, so it does not preempt.

### 1.3 Feng, Wallace, Boyd-Graber — ACL 2019 → **VERIFIED (main track)**

| authority | identifier | result |
|---|---|---|
| **ACL Anthology** (family authority) | `https://aclanthology.org/P19-1554.bib` (HTTP 200) | `booktitle = "Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics"`, `pages = 5533--5538`, `doi = 10.18653/v1/P19-1554`, Florence, Jul 2019 |
| DBLP (cross-check) | `conf/acl/FengWB19` | `booktitle = ACL (1)` → **main**, `pages = 5533-5538`, `doi = 10.18653/V1/P19-1554` |

**Verdict: ACL 2019 main conference (short paper, 6 pages). Not Findings.**

**Full citation:**
> Shi Feng, Eric Wallace, Jordan Boyd-Graber. "Misleading Failures of Partial-input
> Baselines." *Proceedings of the 57th Annual Meeting of the Association for
> Computational Linguistics*, pp. 5533–5538, Florence, Italy, Jul 2019.
> DOI `10.18653/v1/P19-1554`. arXiv:1905.05778.

PDF-title vs metadata-title: **identical**. No drift.

**Is the README's description accurate? YES — and the full text makes it stronger than
the README states.** README says "a partial-input baseline *failing* does not certify a
dataset is artifact-free … clearing the floor is necessary, not sufficient." Verified
from the abstract, p. 5533: "When a partial-input baseline gets high accuracy, a dataset
is cheatable. However, **the converse is not necessarily true**: the failure of a
partial-input baseline does not mean a dataset is free of artifacts." Their construction
is stronger than the README implies: §3.1 builds an SNLI variant on which **the best
achievable hypothesis-only accuracy is exactly chance by construction** (three copies
with conflicting labels ⇒ maximal Bayes error) while a full-input model is perfect. So
the counter-example is not merely empirical, it is a proof by construction that
"null-clearing ⇒ valid" is unsound. **paperC's one-directional discipline (use floor
*failure* to disqualify, never floor *success* to certify) is exactly the right response
and should cite p. 5533 and §3.1.**

---

## 2. DEFECT 2 — full-text pass on every novelty candidate

Candidate list taken from `paperC/README.md` §"Citation obligations" +
`proposal/active/A01-null-calibration-methodology/NOVELTY_CHECK.md` §2.1–§2.7.
(There is **no `RELATED_WORK.md` under A01** — the only `RELATED_WORK.md` in
`proposal/active/` belongs to `A04-recovery-certification`, which is another agent's
territory and was not touched. A01's candidate list lives in `NOVELTY_CHECK.md` §2 and
`STATUS.json:novelty_check_2026_08_09`.) That yields **nine** candidates; all nine were
read in full.

PDFs fetched to `/tmp/pC_pdf/` and text-extracted with `pdftotext -layout` (also
`-raw` flow mode for the two-column ACL papers, because column interleaving corrupts
`-layout` grep context). Camera-ready obtained for 5 of 9.

| candidate | version read | pages | camera-ready? |
|---|---|---|---|
| Balepur et al., ACL 2024 main | `2024.acl-long.555.pdf` **+** arXiv v2, **diffed** | 23 | ✅ CR |
| OLMES (Gu et al.), Findings NAACL 2025 | `2025.findings-naacl.282.pdf` | 29 | ✅ CR |
| Feng et al., ACL 2019 main | `P19-1554.pdf` | 6 | ✅ CR |
| Hewitt & Liang, EMNLP 2019 main | `D19-1275.pdf` | 11 | ✅ CR |
| Ding et al., NeurIPS 2021 | official proceedings PDF | 13 | ✅ CR |
| Zheng et al., ICLR 2025 Oral | arXiv:2410.07137v2 | 34 | ❌ arXiv |
| Cho et al., ICLR 2026 Poster | arXiv:2502.18798**v4** | 11 | ❌ arXiv (see gap below) |
| Bean et al., NeurIPS 2025 D&B | arXiv:2511.04703v1 | 8+app | ❌ arXiv |
| Oostermeijer, ICML 2026 | arXiv:2607.12767v1 | ~10 | ❌ arXiv |

⚠️ **Honest gap.** OpenReview's `/pdf?id=…` and `/attachment?id=…` both return a 12,692-byte
**bot-challenge HTML page** from this network, so the ICLR/ICML/NeurIPS camera-readies
(Cho, Zheng, Bean, Oostermeijer) could not be diffed against arXiv. For **Cho** this
matters most: arXiv v4 is dated **2026-01-12** and the OpenReview camera-ready `pdate`
is **2026-01-26** (14 days later), and v4's header still reads "Under review as a
conference paper at ICLR 2026". So Cho's overlap verdict below is based on a
pre-camera-ready version. Marked accordingly. For Zheng/Bean/Oostermeijer the
arXiv-vs-CR risk is lower (their claims are stable across versions and none is
load-bearing against paperC), but it is **not zero and is not claimed as verified**.

### The three questions asked of every candidate

**Q1** Does it use an input-blind / best-constant null at all?
**Q2** Does it apply that null **per-construct, before arm comparison** (as a
precondition on whether an arm's number may enter a comparison)?
**Q3** Does it cover the MC **letter-vs-content** interface split?

| # | candidate | Q1 input-blind null | Q2 per-construct, pre-comparison gate | Q3 letter-vs-content split | **verdict** |
|---|---|---|---|---|---|
| 2.1 | Balepur, ACL 2024 | **YES** (majority-class = best-constant letter) | **NO** | **NO** (letter only) | **PARTIAL** |
| 2.2 | Oostermeijer, ICML 2026 | **NO** | NO | NO (content only) | PARTIAL (one sub-claim) |
| 2.3 | Cho, ICLR 2026 | partial (per-item choice-only *decomposition*) | **NO** | **YES** (cloze/symbols/hybrid) | **PARTIAL** |
| 2.4 | Bean, NeurIPS 2025 D&B | **NO** (recommends "baselines", unquantified) | NO | NO | NO OVERLAP (framing only) |
| 2.5 | Zheng, ICLR 2025 Oral | **YES** (crafted constant output) | NO | NO | NO OVERLAP on construct |
| 2.6 | OLMES, Findings NAACL 2025 | **NO** ("random" only) | NO | **YES** (MCF/CF) | **PARTIAL** |
| 2.7a | Ding, NeurIPS 2021 | NO | NO | NO | NO OVERLAP |
| 2.7b | Hewitt & Liang, EMNLP 2019 | partial (randomised-label control) | **YES**, on probes | NO | NO OVERLAP on construct |
| 2.7c | Feng, ACL 2019 | **YES** (partial-input) | NO — argues the *opposite* direction | NO | NO OVERLAP (counter-citation) |

**0 of 9 FULLY PREEMPTS.** Detail and page-level evidence follows.

---

### 2.1 Balepur, Ravichander, Rudinger — ACL 2024 main — **PARTIAL**

**Q1: YES, and it is literally a best-constant letter null.** Footnote 4, p. 10310:
> "A majority class baseline always predicts the most frequent answer choice found in
> the dataset."

That is definitionally paperC's letter floor (`always-D` on MMLU). So paperC **cannot**
claim "compare MC letter accuracy to the best-constant label predictor rather than to
chance" as new. This is the single sharpest overlap in the candidate set and the README's
existing obligation ("origin of 'use stronger than chance baselines in MCQA'; do not
claim it") is **correct and must be kept**.

**Q2: NO.** The majority baseline is used only as a **significance target for the
choices-only (input-blind *prompt*) accuracy** — "An asterisk (*) denotes that the
choices-only prompt significantly outperforms the majority class baseline (two-sample
t-test, p < 5e-5)" (Fig. 2 caption, p. 10311). It is never applied to the **full-input**
accuracy, and it never gates a comparison between models. The object is the **dataset**
("indicating that LLMs may be using artifacts in MCQA benchmarks", p. 10311), not a
per-arm validity certificate.

**Q3: NO.** Single interface throughout: generate the letter. §2.1, p. 10309: "the LLM
must give the letter of the correct option a ∈ {(A),(B),(C),(D)}". No cloze/content
scoring, no log-likelihood scoring at all — it is **generative/black-box**
("This black-box setup allows us to study LLM behavior without accessing LLM
internals", p. 10310), whereas every paperC number is likelihood-scored. Different
measurement apparatus.

**Other verified gaps.** Datasets = ARC, MMLU, HellaSwag only (grep: BoolQ 0 hits,
OpenBookQA 0, MMLU-Pro 0). Models = LLaMA-2 70B, Falcon 40B, Phi-2, Mixtral, **all
intact** (grep `damag|prun|truncat` → 0 substantive hits). No depth axis, no tie
convention, no tokenizer analysis, no per-item paired bootstrap (they use a
**two-sample t-test**, not a paired test; paperC's estimators are paired + mid-p).

**★ DEFECT IN THIS PRIOR WORK THAT paperC CAN CORRECT (new, from the full text).**
p. 10310:
> "when the LLM gives an invalid output, we treat it as random guessing and **assign a
> score of 0.25**."

This imputes the **chance** value into the measured accuracy of a paper whose own thesis
is that the majority baseline (not chance) is the right reference. On MMLU the letter
marginal is `0.2689`, so every invalid output is credited `0.25` against a construct
whose input-blind floor is `0.2689` — a systematic, self-inconsistent downward bias of
`−1.89 pp × P(invalid)`. Their Appendix A.5 shows the opposite extreme (invalid = wrong,
i.e. `0.0`) and reports "which do not alter our claims", so the two bounds they present
are `0.0` and `0.25` and **the construct-appropriate value `0.2689` is not among them**.
paperC has the exact quantity needed to fix this and should say so explicitly: it is a
concrete, citable, non-hostile follow-up that demonstrates the floor is *operationally*
load-bearing, not just rhetorical.

**Verdict: PARTIAL — overlap is (i) the majority/best-constant baseline for the letter
interface and (ii) the "use stronger baselines" recommendation. Does NOT preempt:**
different object (dataset cheatability vs per-arm instrument validity), different
apparatus (generation vs likelihood), intact-only, letter-only, no per-construct gate,
and it contains a chance-imputation defect paperC corrects.

**arXiv-vs-camera-ready diff: DONE, clean.** Word-level diff of arXiv v2 vs
`2024.acl-long.555` shows **only** the arXiv stamp, ACL page furniture (10308–10330),
and two-column reflow artifacts. **No substantive text change** — in particular the
`0.25` imputation, footnote 4, and the 11/12 claim are identical in both. Contrast with
the `venue-verify-acl-family-needs-anthology` case (2410.15225) where the camera-ready
added a concession; here there is none.

---

### 2.2 Oostermeijer — ICML 2026 — **PARTIAL (already-accepted obligation, unchanged)**

**Q1: NO.** Zero hits for `input.blind|question.free|choices.only|constant|majority`;
zero hits for `chance`. Its baselines are **other scoring rules** (standard,
length-normalised, PMI, ANPMI, Bayesian), not input-blind predictors.

**Q2: NO.** It measures **length bias** via Kendall's τ between candidate lengths and
scores *within each example* (§3.1) and proposes Bayesian accuracy as a drop-in
replacement. No validity gate, no per-arm disqualification.

**Q3: NO.** Content-side only. It discusses single-letter benchmarks only as a
degenerate case where its own concern vanishes — p. 5:
> "Standard (unnormalized) accuracy is mainly appropriate when (i) the answer text
> already appears in the prompt and the first one or two completion tokens effectively
> fix the option, or (ii) all candidates for a question have very similar lengths, with
> **single-letter prediction benchmarks as an extreme case**."

So the letter interface is explicitly *out of scope* for its analysis.

**Confirms the existing obligation, and no more.** The acc-vs-acc_norm over-correction
IS established here (abstract: length normalisation "frequently over-corrects,
introducing a bias toward longer answers instead"; Table 2 shows τ(S_byte, D) uniformly
positive, "On datasets such as ARC, ARC German, OpenbookQA, SciQ and WinoGrande,
length-normalization turns a modest negative or near-zero τ into a substantial positive
one"). paperC's existing retraction (drop the acc/acc_norm sub-claim; reframe OBQA sign
flips as replication under damage) is **exactly right and sufficient**.

**Newly relevant to paperC's tokenizer finding (a citation paperC is currently missing).**
p. 2, on token-based normalisation:
> "A drawback of token-based normalization is that the metric **becomes dependent on the
> tokenizer**, which complicates comparisons between models with different tokenization
> schemes. This can be mitigated by normalizing by the number of bytes (or characters)."

and p. 4: "we primarily use byte length, because of its common usage and desirable
**tokenizer-independence** property."

**This is prior art for the *existence* of tokenizer-dependence in length-based MC
scoring, and paperC's `NOVELTY_CHECK.md` §3 currently names the tokenizer-dependence of
the longest-option null as "the strongest remaining A01-owned claim".** That claim must
be **narrowed, not dropped**: Oostermeijer says the *scoring rule* is tokenizer-dependent
and prescribes bytes as the fix; paperC's finding is about the **null/floor itself** —
that a longest-option *floor* computed on identical items moves `0.9003 pp` (`split`)
and up to `10.6 pp` (`credit`) across tokenizers, i.e. the *reference point*, not the
model score, is tokenizer-valued. Honest form: *"length-based MC scoring is known to be
tokenizer-dependent (Oostermeijer, ICML 2026); we show the induced input-blind floor
inherits that dependence, so a content floor is a property of (dataset, convention, unit,
tokenizer) and must be quoted with its tokenizer."* paperC's character-vs-token finding
(OBQA `0.3635` char vs `0.3680` token) is also directly anticipated in *spirit* by his
bytes-vs-tokens recommendation — cite it there too. **Still not preempted** (he never
computes a null), but the "apparently unclaimed" framing in `NOVELTY_CHECK.md` §3 is now
too strong and should be softened to "not previously computed for an input-blind null".

**Verdict: PARTIAL — preempts the acc/acc_norm sub-claim (already retracted) and now
also the *generic* tokenizer-dependence-of-length-scoring observation. Does NOT preempt
the floor-level result.**

---

### 2.3 Cho, So, Lee — ICLR 2026 Poster — **PARTIAL** (⚠️ arXiv v4, not camera-ready)

**Q1: PARTIAL, and structurally different from a floor.** It computes a per-item
**decomposition**, not a baseline: §3.1, p. 3,
`Score(Q,C,x) = Score_choice(Q,C,x) + Score_question(Q,C,x)`, where the choice-driven
term "is determined by calculating the score with the question replaced by an empty
string". That is an input-blind *model* evaluation (question ablated), like Balepur's
choices-only prompt — **not** a best-constant predictor, and it is used per-item to
attribute a decision, never as a scalar floor a number must clear.

**Q2: NO.** `Choice sensitivity = (1/N) Σ 1[Δchoice > Δquestion]` (p. 3) is a *rate*
reported alongside accuracy; it never disqualifies an arm. Its remedy is a **new metric**
(NPSQ, §4, pp. 6–7), i.e. it replaces the instrument rather than gating it — the
opposite of paperC's metric-agnostic position. Zero hits for `floor`, `constant`,
`majority`, `best-constant`.

**Q3: YES — this is the closest thing in the literature to paperC's interface axis.**
p. 4: "We use three input formats for MCQA tasks … cloze, symbols, and hybrid", where
**symbols** = predict the label token 'A'/'B'/'C'/'D' (= paperC's letter) and **cloze** =
score the full answer text (= paperC's content). It reports the format contrast across
Qwen-2.5 (0.5B–72B), Llama-3.1, Mistral, on HellaSwag / ARC-Challenge / MMLU, and finds
"choice sensitivity ranges from approximately 0.2 to 0.4 for the symbols and hybrid
formats and from around 0.5 to 0.6 for the cloze format" (p. 4). **paperC must therefore
not claim the letter-vs-content interface contrast, nor "MC interface validity is
unexamined"** — the README's existing obligation already says this and it is confirmed
by the full text.

**Q gaps (verified by grep + read).** Zero hits for `damag|prun|truncat` — **all models
intact**. No depth ladder. No best-constant floor. No tie convention. No per-item paired
bootstrap or significance test against a null. And observation 3 (p. 4) — "Normalization
by token length fails to mitigate choice sensitivity … in some cases, particularly with
the ARC-Challenge benchmark and cloze format" — is *adjacent* to Oostermeijer but framed
as sensitivity, not length bias.

**Verdict: PARTIAL — preempts "the letter/content interface choice is worth examining"
and "an input-blind (question-ablated) reading of MCQA is informative". Does NOT preempt:**
no best-constant null, no pre-comparison gate, no damage regime, and its prescription
(adopt NPSQ) is orthogonal to paperC's (report against your own floor, whatever metric
you use). **⚠️ Caveat: read at arXiv v4 (2026-01-12); the camera-ready (`pdate`
2026-01-26) could not be fetched. Re-diff before submission.**

---

### 2.4 Bean, Kearns, Romanou, et al. — NeurIPS 2025 D&B — **NO OVERLAP (framing only)**

**Q1: NO.** Zero hits for `null`, `constant`, `majority`, `best-constant`. The single
relevant passage is a *qualitative* recommendation, p. 6:
> "Several strategies can mitigate these confounding effects. **Baselines can be
> established for performance on the relevant subtasks alone.** If a benchmark requires
> world knowledge but does not intend to measure it, models should first be tested on
> this world knowledge directly and scores adjusted…"

That is about **subtask-ability baselines** (test the auxiliary skill separately), not an
input-blind constant. **Q2: NO** — it is a 445-benchmark systematic review with eight
recommendations and an operational checklist; it does not run the measurement. **Q3: NO**
— MC format is mentioned once in passing (p. 8: "multiple-choice formats are easy to
score").

**★ A precise, checkable positioning claim paperC can now make.** I enumerated **all 27
main-body checklist items** across the eight recommendations (§5.1–§5.8; the appendix
repeats them): **zero of them mention a baseline, null, chance level, or constant
predictor.** The closest are §5.6's "Report the benchmark's sample size and justify its
statistical power" and "Report uncertainty estimates for all primary scores". So the
canonical construct-validity checklist for LLM benchmarks **has no null-calibration
item** — paperC's contribution is precisely the missing item, stated operationally and
measured. That is a much better positioning than "an instance of what this survey asks
for": it is **an item the survey does not ask for**. (Bean is also, usefully, the
authority for the *vocabulary* paperC uses — do not claim "construct validity".)

**Verdict: NO OVERLAP. Mandatory framing citation; also the evidence that paperC's item
is absent from the field's checklist.**

---

### 2.5 Zheng, Pang, Du, Liu, Jiang, Lin — ICLR 2025 Oral — **NO OVERLAP on construct**

**Q1: YES**, and it owns the term. p. 1: "even a 'null model' that always outputs a
constant response (irrelevant to input instructions) can cheat automatic benchmarks",
achieving "an 86.5% LC win rate on AlpacaEval 2.0" (p. 1; Table 2, arXiv p. 9:
Structured+RS 86.5 LC / 76.9 raw). paperC must **not** claim "a constant predictor
can top a benchmark" — the README obligation is correct.

**Q2: NO. Q3: NO.** Verified by grep: `multiple.choice|MCQA|MMLU|ARC|BoolQ` → **0
substantive hits** in a 34-page paper; `validity` → **0**; `log.?likelihood` → **0**;
`best.constant|most frequent` → **0** (the one `majority` hit is Table 3's caption, about
*human* annotator majority preference). The entire paper is LLM-judge **win-rate**
benchmarks (AlpacaEval 2.0, Arena-Hard-Auto, MT-Bench).

**The decisive structural difference, confirmed by the full text.** Their null is
**crafted and adversarial** — a "structured cheating response" plus a random-search
adversarial prefix (§ "Crafting adversarial prefix by random search (RS)") optimised
against the judge, i.e. an *attack*. paperC's null is **derived** from the benchmark's
own label marginal / option-length statistics, i.e. a *reference*. Their finding is
"the judge is gameable"; paperC's is "the honest arm has landed on the reference".
Their threat model requires an attacker; paperC's requires none.

**Verdict: NO OVERLAP on the construct. Mandatory terminological citation for
"null model"; do not claim it.**

---

### 2.6 OLMES (Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi) — Findings NAACL 2025 — **PARTIAL, and paperC currently mis-describes it**

**Q3 first: YES.** OLMES is the standard that *names* paperC's two interfaces —
**MCF** (predict the label A/B/C/D) vs **CF** (cloze/completion, score the answer text),
defined §2.1, and the abstract (p. 5020) says it "supports meaningful comparisons between
smaller base models that require the unnatural 'cloze' formulation … against larger
models that can utilize the original formulation". The interface split and the fact that
it changes measured performance are **established and not paperC's**. Confirmed.

**★ CORRECTION — the "SIZE-keyed interface-selection rule" does not exist.** Both
`paperC/README.md` and `NOVELTY_CHECK.md` §2.6 assert OLMES "*prescribes* which interface
to use **as a function of model size**" and position paperC as fixing "an
interface-selection rule keyed on model *size*". The full text says something different.
p. 5026:
> "In OLMES, we standardize to evaluate each model using **both the MCF and CF
> formulations, and the best performing one is used.** This allows for meaningful
> comparison of task evaluation numbers over a range of models, from the smaller, weaker
> base models which can only deal with the CF (where MCF scores hovering around random
> baseline), to the stronger models which can report more accurate performance using the
> MCF (where CF provides less clear signal)."

and Table 7's caption, p. 5038:
> "The 'max' average corresponds to the **OLMES score, taking the best of MCF and CF for
> each task**."

So the rule is **`max(MCF, CF)` per task per model**. Size is the *narrative* for why the
max lands where it does (Fig. 1, p. 5026, shows OLMo-7B-0424's MMLU MCF overtaking CF at
~400B tokens; §"CF often works better for weaker models while MCF is at random", p. 5033)
— it is **not the selection key**. I searched the whole paper for a size-conditional rule:
the only size-related content is the model list "covering a range of sizes from 1B to 70B"
(p. 5022) and Table 1's per-reference formulations. **There is no rule keyed on parameter
count.** Attacking a rule the paper does not state would be an easy referee kill; §0c
above supplies corrected replacement text.

**Q1: NO — and this is the *real*, stronger defect.** OLMES's only reference line for
"is this interface meaningful for this model" is **"random"**. Verified: the string
**"chance" occurs 0 times** in the 29-page paper; `majority|most frequent|best.constant|constant predictor`
→ **0 hits**; Fig. 2 (p. 5027) plots MCF and CF against a third series labelled
`random`; p. 5026 "MCF scores hovering around random baseline" and "the weakest 8 models
have near-random performance on the MCF version". And the robustness discussion asserts the
label-marginal question **without measuring it** — Appendix "[Part 1] Order of presenting
the options A/B/C/D", p. 5035:
> "For MCF it is indeed a confounder that some (especially weaker) models might highly
> prefer a given label (like B). **The benchmarks in OLMES are generally balanced such
> that such a model would not be much better than random.** Further, if this happens, CF
> would generally get a better score in such cases and OLMES would use that score…"

paperC computes exactly the asserted quantity, on **the same task list** (OLMES Table 2,
p. 5023: ARC_C 1172, ARC_E 1000, BoolQ 1000, CSQA 1221, HSwag 1000, MMLU **14042**,
OBQA 500, PIQA 1000, SIQA 1000, WinoG 1267 — paperC's six non-MMLU tasks and its
n=14042 MMLU are the same instances), and finds the label marginal is **not** the chance
line: ARC-C always-B `0.265358` vs `0.250156`; ARC-E always-C `0.266414`; OBQA always-A
`0.276000`; CSQA always-B `0.208845`; PIQA always-B `0.504897`; MMLU always-D `0.2689`;
MMLU-Pro always-A `0.116606` vs `0.100000` = **1.1661×**. So OLMES's own
balance-argument has a measurable error term it never quantifies, and its escape hatch
("CF would get a better score, and OLMES would use that") is a **testable** prediction
that paperC's data can check directly — paperC already has the counter-case shape on
record (arc_easy `keep8` letter at floor `0.2584` while content scores `0.6460`, +38.76 pp:
there the max rule *does* rescue; whereas on MMLU content_norm sits within ±3 pp of
letter, where it *cannot*). **That contrast is a genuine, empirical, citable
follow-up to OLMES and is a better contribution than the size-keyed story.**

**Q2: NO.** No floor is ever computed, so no arm is ever disqualified. The max rule is
an *uncalibrated* selection: it reports whichever of two numbers is larger without asking
whether either clears its own input-blind reference.

**One more precision the writeup needs (do not overclaim here).** OLMES **dismisses** the
tokenizer objection to per-token normalisation, p. 5037:
> "…normalizing per token is problematic since it depends on the tokenizer. Since the
> purpose of the normalization is simply to rank the answer choices within themselves
> (**keeping model and tokenizer fixed**), this **does not seem like a relevant
> argument**…"

Their dismissal is **valid in its own scope** and paperC must not present it as an error:
for ranking choices at fixed model+tokenizer, tokenizer-dependence indeed cancels.
paperC's finding lives in the scope OLMES's parenthetical excludes — comparing a
*floor* across models with different tokenizers. State the scope split explicitly.
(Also note OLMES prescribes per-task CF normalisations — pmi for ARC_C/CSQA/OBQA, char
for ARC_E/HSwag/MMLU/PIQA/SIQA, none for BoolQ/WinoG, Table 3 p. 5025 — which is a
fourth degree of freedom adjacent to paperC's convention/unit/tokenizer trio and worth a
sentence.)

**Verdict: PARTIAL — preempts the letter/cloze interface split (already conceded). Does
NOT preempt:** no null of any kind, no floor, no gate, intact models only. **But paperC's
current description of the OLMES rule is factually wrong and must be replaced (§0c).**

---

### 2.7a Ding, Denain, Steinhardt — NeurIPS 2021 — **NO OVERLAP**

Read in full. Object = **representation dissimilarity measures** (CKA, CCA, PWCCA,
Orthogonal Procrustes); method = sensitivity/specificity benchmarks with 30,480 examples
across random seed, layer depth, low-rank approximation (p. 1). Ground truth = probing
accuracy and OOD accuracy, scored by **rank correlation** with measured functional
differences. Zero hits for `null` or `permutation`; `chance` appears only as the reading
of a binary task ("drops significantly from 80% to 63% (chance is 50%)", p. 5) and as a
challenge-set statement ("none of the measures do statistically better than chance",
p. 1).

**No MC accuracy, no input-blind null, no interface, no gate.** This is the
similarity-null-adjacent prior art A01 already concedes on its C3 leg, and clause 3's
verdict (`does not fire`, because paperC's surviving findings are on the MC-accuracy
construct) is **unaffected by the full-text read**. Useful secondary point: since the
paper contains no permutation null, paperC's "we are not aware of a prior layer-order
null for layer correspondence" survives at the same strength as before.

**Verdict: NO OVERLAP.**

### 2.7b Hewitt & Liang — EMNLP 2019 — **NO OVERLAP on construct** (best structural precedent)

Read in full; see §1.2. Control task = random per-word-type labels; selectivity =
linguistic-task acc − control-task acc. **Q2 is the one YES in the whole candidate set** —
they use the null as a *precondition on interpretation*, and §4.2/Table 2 (p. 2740) uses
it to overturn a published layer comparison. **But the construct is probe accuracy, not
MC accuracy; the null is over supervision, not over input; and there is no
best-constant/input-blind predictor** (`best.constant|majority` → 0 hits;
`MCQA|multiple.choice` → 0 hits).

Also directly quotable for paperC's discipline, p. 2740:
> "Without considering selectivity, it might be thought that ELMo2 encodes nothing about
> part-of-speech, since it doesn't beat the Proj0 random representation baseline. Taking
> selectivity into account, we see that probes on ELMo2 are unable to rely on word
> identity features…"

i.e. *the wrong reference produced the wrong verdict about a component* — paperC's thesis
transposed to probing. Cite it as the methodological ancestor.

**Verdict: NO OVERLAP. Cite as the strongest structural precedent (a null that reverses a
within-model component comparison), on a different construct.**

### 2.7c Feng, Wallace, Boyd-Graber — ACL 2019 — **NO OVERLAP (counter-citation, load-bearing)**

Read in full (6 pages); see §1.3. Its direction is the **opposite** of a preemption: it
proves by construction (§3.1–3.2, pp. 5534–5535) that a partial-input baseline *failing*
certifies nothing, and its §5 "Hypothesis Testing" paragraph, p. 5536, gives the general
reason:
> "Validating datasets with partial-input baselines is a form of hypothesis-testing …
> While it is tempting to hypothesize other ways a model can cheat, it is **infeasible to
> enumerate over all of them**."

**This is the paper that bounds paperC's claim, and paperC's existing one-directional
discipline already respects the bound** (floor-failure disqualifies; floor-success never
certifies). The README obligation ("state explicitly that clearing a floor is necessary,
not sufficient") is confirmed as **necessary, and it should cite p. 5533 + §3.1 for the
constructive proof, not just the abstract**.

**Verdict: NO OVERLAP.**

---

## 3. Overall: what paperC must cite, and what it may still claim

**What paperC must cite and must not claim as its own** (all confirmed at full-text
level, none newly added or removed relative to the README's existing list, except as
noted): the **best-constant/majority baseline for the MC letter interface and the
"use stronger than chance baselines" recommendation** are Balepur et al., ACL 2024 main,
p. 10310 fn. 4 + p. 10311 — this is the tightest overlap and the README is right to
forbid claiming it; the **term "null model" and "a constant predictor can top a
benchmark"** are Zheng et al., ICLR 2025 Oral, p. 1; the **letter(MCF)/cloze(CF)
interface split** is OLMES, Findings NAACL 2025, §2.1 + p. 5026, and is *independently*
established for MCQA-validity purposes by Cho et al., ICLR 2026, p. 4; the
**acc-vs-acc_norm over-correction** is Oostermeijer, ICML 2026 (abstract + Table 2) and
must stay retracted; **construct-validity vocabulary** is Bean et al., NeurIPS 2025 D&B;
and **"clearing a null is necessary, not sufficient"** is Feng et al., ACL 2019, p. 5533
+ §3.1. Two additions the full-text pass forces: (i) Oostermeijer p. 2/p. 4 must now be
cited for the *generic* tokenizer-dependence of length-based MC scoring, so
`NOVELTY_CHECK.md` §3's "apparently unclaimed" framing of paperC's tokenizer finding must
soften to "not previously computed for an input-blind null"; and (ii) Hewitt & Liang
§4.2/Table 2 (p. 2740) should be cited as the structural ancestor — a null that reverses a
within-model component comparison — which is a *stronger* and more honest genealogy than
citing them merely for "probes need a null".

**What paperC may still claim as its own.** Nothing in the nine candidates computes a
best-constant/input-blind null **per construct** and uses it as a **precondition on
whether an arm's number may enter a comparison at all** — Q2 is NO in 8 of 9, and the one
YES (Hewitt & Liang) is on probe accuracy with a randomised-label control, not on MC
accuracy with an input-blind constant. Nothing in them touches a **damaged/pruned/
truncated** model: `damag|prun|truncat` returns zero substantive hits in Balepur, Cho,
Oostermeijer, and Zheng, and the whole regime in which paperC's headline lives (letter
degenerating **to or below** its own floor, 14/15 on cross-family MMLU-Pro) is therefore
untouched. None of paperC's floors appears in any candidate PDF (`0.2689`, `0.2845`,
`0.6217`, `0.3635`, `0.116606` → zero hits across all nine), only OLMES mentions BoolQ
and only Oostermeijer mentions OpenBookQA, and **no candidate mentions MMLU-Pro at all**,
so the power-wall leg is entirely paperC's. The four under-specifications of the null —
tie convention, character-vs-token unit, tokenizer, and the `mean(1/n_opt)`-vs-naive
ambiguity of "chance" itself when `n_opt` varies — are absent from all nine, and the
fourth is the sharpest because it is a defect in the *chance* line that everyone else
uses. Finally, paperC has two **corrective** contributions the full-text pass newly
licenses: Balepur's `0.25` invalid-output imputation is inconsistent with its own thesis
and paperC supplies the right value (`0.2689`), and OLMES's Part-1 balance argument
(p. 5035) makes a quantitative assertion it never measures while its `max(MCF,CF)` rule
makes a testable prediction paperC's arc_easy-vs-MMLU contrast can check. Per
`memory/prior-work-differentiate-dont-abandon`, that is the right shape of output here:
**nine verified citations, two narrowings, zero kills, and two places where paperC fixes
the prior work.**

---

## 4. Method and its limits

Network via `hy-proxy.woa.com:3128`. Authorities queried: **ACL Anthology** `.bib`
endpoints (`D19-1275`, `P19-1554`, and `2024.acl-long.555` / `2025.findings-naacl.282`
for cross-checks); **DBLP** `search/publ/api` + `rec/<key>.xml`
(`conf/emnlp/HewittL19`, `conf/acl/FengWB19`, `conf/nips/DingDS21`); **OpenReview API1**
`notes/search` (required for pre-2023 venues — API2 returns nothing for the 2021 paper)
and **API2** `notes/search` (Cho `ICLR.cc/2026/Conference` + `Submission17078/-/Camera_Ready_Revision`;
Oostermeijer `ICML.cc/2026/Conference` + `Submission34463/-/Camera_Ready_Revision`, both
re-confirmed this session); **official NeurIPS proceedings** page + bibtex. Nine PDFs
extracted with `pdftotext` (both `-layout` and raw flow; the two-column ACL papers need
raw mode or grep context interleaves columns).

**Limits, stated plainly.**
1. **OpenReview PDFs are unreachable from this network** (12,692-byte bot-challenge HTML
   on both `/pdf?id=` and `/attachment?id=`). So Cho / Zheng / Bean / Oostermeijer were
   read at their **arXiv** versions. Only **Balepur** got a real arXiv-vs-camera-ready
   diff (clean). **Cho is the one where this could matter** (arXiv v4 2026-01-12 vs
   camera-ready `pdate` 2026-01-26, and v4 still says "Under review"). Not claimed as
   camera-ready-verified.
2. `api.openreview.net/notes?id=<id>` is also challenge-blocked; only
   `notes/search?query=` works, so OpenReview metadata came from search hits rather than
   direct note fetches. Cross-checked against DBLP and official proceedings wherever a
   second record existed.
3. Semantic Scholar was **not** queried — the family-correct authority answered in all
   three venue cases, and per the two venue memories S2 is a cross-check, not an
   authority. No venue claim here rests on S2.
4. Q1/Q2/Q3 judgements are from reading the papers, but the *negative* claims
   ("0 hits for X") are grep-based over `pdftotext` output; a term rendered as an image or
   split by hyphenation could be missed. The load-bearing negatives (no best-constant
   null in OLMES/Cho/Oostermeijer/Bean; no damage axis anywhere; none of paperC's floor
   values anywhere) were each checked with several spellings.
5. **ZERO GPU used, no SSH to any compute node.** CPU + web only.
