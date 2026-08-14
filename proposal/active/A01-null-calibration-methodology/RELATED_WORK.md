# A01 — RELATED WORK / NOVELTY BOUNDARY (post-promotion)

**Written 2026-08-15. 0 GPU, 0 SSH. Adjudication + venue verification only.**

This closes the blocker `proposal/ready_queue.py:504-515` trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and discharges the two items
`proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:26` assigns A01
(rating **部分充分** / partially sufficient; prescribed fix: *"在 `PROPOSAL.md` 加
closest-collision 表；`SOURCES.md` 增加外部一手文献"* — add a closest-collision table and
external primary sources).

> ⚠️ **A01 IS NOT IN THE PROPOSAL PHASE.** `STATUS.json:promoted_to = "paperC"`,
> `status = "PROMOTED_to_paperC_2026_08_11_major_revision_carried_forward"`. So this file is
> **not** an exploratory scan for a candidate direction — it is the **novelty boundary of a
> manuscript currently in blind-review rounds** (`paperC/review_rounds/round_{00,01,02}/`).
> The standard is correspondingly higher: every "we may claim X" below has to survive a
> reviewer holding the actual `.tex` in hand. Where the manuscript has **already conceded**
> a point, this file concedes the same point in the same words. It does not re-open it, and
> it does not quietly widen it back.

---

## 0. What the paper claims RIGHT NOW (read from the .tex, not from PROPOSAL.md)

Read on 2026-08-15 from `paperC/sections/{00_abstract,01_introduction,02_related,03_method,03b_nulls,05_analysis}.tex`
and `paperC/sections/tab_claims.tex`. PROPOSAL.md's `## 可以主张` list is **older** than the
manuscript and in two places **wider**; where they disagree, the `.tex` governs.

| # | Live claim in the manuscript | Where |
|---|---|---|
| 1 | Before comparing arms, a construct's score must be tested against its **best constant, input-blind** predictor. This is a **necessary, not sufficient** validity condition (one-directional: failure disqualifies, success does not certify). | `01_introduction.tex` last ¶; `03_method.tex` §"Read-out v1"; `tab_claims.tex` row 15 |
| 2 | The floor **is not chance** (BoolQ 0.6217 vs 0.50; MMLU-Pro 0.116606 = 1.1661× naive ten-way). | `00_abstract.tex`, `03b_nulls.tex` §4 |
| 3 | **The floor itself needs calibration.** $\hat f=\max_L \hat m_L$ is a max over $k$ noisy marginals, hence upward biased even under exactly uniform labels; only **MMLU-Pro, MMLU, BoolQ** have floors a balanced null could not produce ($p<10^{-5}$), and the other five letter constructs sit inside the estimator's own noise ($p=0.14$–$0.85$). | `tab_nulls.tex`; `00_abstract.tex` |
| 4 | Content floors are **under-specified** by tie convention, character-vs-token length unit, and tokenizer; and "chance" itself is ambiguous when `n_opt` varies. | `03b_nulls.tex` §§1–4 |
| 5 | Wrong-null verdict flip at MMLU-scale power: 14/15 designated damaged cells at/below floor; held to **one** evidentiary standard it is **3/12 above chance vs 1/12 above floor**, ~⅓ the size the asymmetric point comparison suggested. | `05_analysis.tex` §"Both sides of the flip" |
| 6 | Read-out **v2** (arm-conditional permutation null, stratified by `n_opt`) **re-sorts** the 27-cell read-out in both directions and exposes a weak intact anchor (intact Llama-2-7B `recovery_fraction`=0.0545 < 0.10 materiality bar → relative claims blocked for that family). | `05_analysis.tex` §"V2 re-sorts" |
| 7 | Full-fp32 removes 100% of bf16 exact ties and moves 18.03% of letter argmaxes **without** recovering accuracy → the numerical-tie mechanism is falsified. | `05_analysis.tex` §"Full precision" |

### 0.1 What the manuscript ALREADY concedes as not new — and this file does not re-open

`03_method.tex:36-48`, verbatim heading and sentence:

> **"The statistic is not new; the stratification and its use are."** … *"We therefore claim
> neither this statistic nor the collapse identity below as new: that chance-corrected
> agreement vanishes for a constant rater is the textbook property, since $p_o=p_e$ there.
> Our contribution is (i) stratifying the permutation within variable option count, so
> letters that are illegal for an item are never credited, and (ii) using the quantity as an
> **arm-conditional pre-comparison gate** with an explicit materiality bar rather than as an
> agreement score."*

So: $\Delta_{\mathrm{perm}}$ **is** the numerator of Cohen's $\kappa$ within `n_opt` strata
($\kappa=(p_o-p_e)/(1-p_e)$, hence $\Delta_{\mathrm{perm}}=\kappa(1-p_e)$ identically,
verified on all 27 cells to printing precision), and the constant-collapse zero **is** the
$\kappa=0$ property. §2.7 below adjudicates whether the residual **(i)** and **(ii)** hold.

`02_related.tex:15` already states this in the Related Work section itself, and
`review_rounds/round_01/ROUND_01_FROZEN.json` records commit `7d39546` as *"retracted the v2
novelty claim (it is a stratified Cohen kappa numerator)"*. **This retraction is settled. Any
future A01/paperC text that re-claims the statistic is a regression, not a strengthening.**

---

## 1. Standing rules this adjudication obeys

1. **`memory/prior-work-differentiate-dont-abandon.md`** (user, 2026-08-07: 「别因1-2篇类似
   工作就放弃方向」). Preemption requires **完全相同 / 抄袭** (essentially identical scope /
   plagiarism), **not overlap**. Work within 2–3 months is **concurrent** and cannot preempt.
   A direction dies from its **own kill gate**, never from a literature count. A01's kill
   clause 3 is `NOVELTY_CHECK.md`'s object and **did not fire**; nothing below changes that.
2. **Venue verification is family-split**, and using the wrong authority produces false calls
   in both directions:
   * ICLR / NeurIPS / ICML / TMLR → **OpenReview `venueid`** (+ `Camera_Ready_Revision`)
     — `memory/venue-verify-must-use-openreview-2026.md`
   * ACL / EMNLP / NAACL / EACL **including Findings** → **ACL Anthology + DBLP**
     — `memory/venue-verify-acl-family-needs-anthology.md`
   * Non-CS venues (psychometrics, epidemiology) → **Crossref DOI**, which is the authority
     for those journals; neither Anthology nor OpenReview indexes them at all.
   * `arXiv-only` below means **"I could not verify a peer-reviewed venue from this node"**,
     NOT "no venue exists".
3. **A measurement paper collides on two sides**, so both authorities were walked: the MCQA /
   evaluation-protocol side is ACL-family, the null/statistic side is OpenReview-family, and
   the **chance-correction side is neither** — it is 1954–1988 psychometrics (§2.7).

### 1.1 Endpoints actually reachable from this node on 2026-08-15 (verbatim status)

| Endpoint | Status | Consequence |
|---|---|---|
| **ACL Anthology** `https://aclanthology.org/<id>.bib` | ✅ works | authority for every ACL-family call below |
| **DBLP** `search/publ/api?...&format=json` | ✅ works (unlike the A04 pass, where it 500'd all session) | primary cross-check |
| **OpenReview API v1** `api.openreview.net/notes/search` | ✅ works | the **only** OpenReview route available |
| **OpenReview API v2** `api2.openreview.net/*` | ❌ **HTTP 403 `ChallengeRequiredError`** on every path (`notes/search`, `notes?forum=`, `notes?content.venueid=`) | ⚠️ **the repo's mandated `venueid`+`Camera_Ready_Revision` route is DOWN.** v1 returns `venue`/`venueid` but **not** the invitation list, so *no* `Camera_Ready_Revision` check was possible this session. Where a 2026 venue matters I say so explicitly. |
| **arXiv API** `export.arxiv.org/api/query` | ❌ HTTP 429 `Rate exceeded` | fell back to `arxiv.org/abs/<id>` HTML meta (`citation_title`, `citation_date`, comments, `jref`) — works |
| **Semantic Scholar** graph API | ❌ HTTP 429 | not used as an authority anywhere (per repo rule it is only ever a cross-check) |
| **Crossref** `api.crossref.org` | ✅ works | authority for the psychometrics leg |

This matters for one specific call: **`api2` being challenge-blocked means I re-verified
Oostermeijer/Cho/Bean/Zheng only at the level `NOVELTY_CHECK.md` and
`paperC/VENUE_AND_NOVELTY_VERIFICATION.md` already established (both of which DID reach api2
on 2026-08-09 / 2026-08-12).** I am not re-asserting those `Camera_Ready_Revision` invitations
from this session; I am citing the earlier passes that saw them, and marking them as
second-hand *for this session*. That is an honest gap, recorded in §5.

---

## 2. Named closest collisions

Ordered by how close they come to **the live claims in §0**, not by topic similarity. Every
row: paper → year → **verified venue + which authority verified it** → what it does → the
precise difference.

### 2.1 Balepur, Ravichander, Rudinger — *Artifacts or Abduction: How Do LLMs Answer Multiple-Choice Questions Without the Question?* — **THE CLOSEST**

* **Venue: ACL 2024 MAIN, Volume 1 Long Papers**, pp. 10308–10330, DOI `10.18653/v1/2024.acl-long.555`.
  **Verified this session** via ACL Anthology `.bib` (`2024.acl-long.555.bib` →
  `booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational
  Linguistics (Volume 1: Long Papers)"`). Anthology ID is `acl-long`, **not** `findings-acl` →
  main conference. ACL family → Anthology is the correct authority.
* **What it does**: input-blind (choices-only, question removed) prompting; compares model
  accuracy to a **majority baseline** rather than chance ("bests a majority baseline in 11/12
  cases"); explicit recommendation *"we advocate for the use of stronger baselines in MCQA
  benchmarks"*.
* **Overlap**: this is A01's headline recommendation, in a verified main-conference venue,
  two years earlier. **A01 may not claim "compare MC accuracy to a majority/best-constant
  baseline rather than chance."**
* **Difference (four axes, all load-bearing)**:
  1. **Object**: theirs is *dataset cheatability* under a black-box **generative** setup;
     A01's is a **per-arm validity gate** on a **likelihood-scored** construct that decides
     whether an arm's number may enter a comparison at all.
  2. **Regime**: every model in Balepur et al. is **intact**. A01's confirmed finding is
     about **structurally damaged** arms landing **at or below** their own floor (14/15
     cross-family MMLU-Pro cells). `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §3 records that
     `damag|prun|truncat` returns **zero substantive hits** in their PDF.
  3. **Numeric inconsistency A01 corrects**: their invalid-output analysis imputes 0.0 or 0.25
     on MMLU, while the construct's best-constant letter value is **0.2689** — i.e. their own
     imputation floor is below the baseline their thesis argues for. `02_related.tex:4` already
     states this. That is a **follow-up that fixes a defect**, the shape the standing directive
     asks for.
  4. **No treatment of the null's own degrees of freedom** (§0 claim 4) and **no calibration of
     the floor estimator** (§0 claim 3).
* **Preempts? NO.** Overlap is on the *recommendation*, not the object, regime, or findings.
  Mandatory citation; priority conceded.

### 2.2 Zheng, Pang, Du, Liu, Jiang, Lin — *Cheating Automatic LLM Benchmarks: Null Models Achieve High Win Rates*

* **Venue: ICLR 2025 Oral** (arXiv:2410.07137). Verified via **OpenReview
  `venueid = ICLR.cc/2025/Conference`, `venue = "ICLR 2025 Oral"`** in
  `NOVELTY_CHECK.md` §2.5 (api2 pass, 2026-08-09). ⚠️ **Not re-verified this session** —
  api2 is challenge-blocked (§1.1).
* **What it does**: owns the term **"null model"** in LLM evaluation and shows a **constant,
  input-independent** output reaching 86.5% LC win rate on AlpacaEval 2.0.
* **Overlap**: "a constant predictor can top a benchmark" — the structural core of A01's
  argument, stated in the abstract of a verified top-venue paper.
* **Difference**: their null is **crafted by an attacker** to game an **LLM judge**
  (win-rate benchmark, adversarial threat model). A01's null is **derived from the
  benchmark's own label/length statistics**, needs no attacker, and is used **diagnostically**
  as the reference an honest arm must clear. `02_related.tex:4` states exactly this.
* **Preempts? NO.** Mandatory citation; A01 may not present "a constant predictor can beat a
  benchmark" as new.

### 2.3 Oostermeijer — *Accuracy and Normalized Accuracy under Length Bias: Analysis, Guidelines, and a Bayesian Alternative*

* **Venue: ICML 2026** (arXiv:2607.12767). Verified via **OpenReview forum `SbSFZ9N6DN`,
  `venueid = ICML.cc/2026/Conference`, + `Submission34463/-/Camera_Ready_Revision`** in
  `NOVELTY_CHECK.md` §2.2, re-confirmed in `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §2.2.
  ⚠️ **Not re-verified this session** (api2 blocked). **Not concurrent**: OpenReview
  `cdate` 2026-01-24 / `pdate` 2026-04-30 both predate the concurrency window — treated as
  the stricter case, i.e. real prior art.
* **What it does**: `acc_norm` does not merely fix length bias but **over-corrects**; which of
  `acc`/`acc_norm` is right depends on the completion-length distribution; also establishes the
  **generic tokenizer-dependence** of length-based MC scoring and recommends bytes over tokens.
* **Overlap — this one took a sub-claim off A01.** A01's `acc`-vs-`acc_norm` length-sensitivity
  observation is **preempted for the length-bias mechanism** (`tab_claims.tex` row 5:
  status **Preempted**), and the "apparently unclaimed" framing of A01's tokenizer finding was
  **softened to "not previously computed for an input-blind null"**
  (`NOVELTY_CHECK.md` §6 note, 2026-08-12).
* **Difference**: its object is a **scoring-rule** bias in **healthy** evaluation and its remedy
  is a **better scoring rule**. It computes **no null**. A01's statement is about the
  **induced input-blind floor** inheriting those choices — the 40.6-pp `wrong`→`credit` span on
  MMLU-Pro and the 0.9003-pp tokenizer span **are properties of the null**, which this paper
  never constructs. `02_related.tex:12` scopes it correctly.
* **Preempts? NO — but it removes one A01 sub-claim, which stays removed.**

### 2.4 Gu, Tafjord, Kuehl, Haddad, Dodge, Hajishirzi — *OLMES: A Standard for Language Model Evaluations*

* **Venue: Findings of NAACL 2025** (**Findings**, not main), pp. 5020–5048,
  DOI `10.18653/v1/2025.findings-naacl.282`. **Verified this session** via Anthology `.bib`
  (`booktitle = "Findings of the Association for Computational Linguistics: NAACL 2025"`).
  The Findings-vs-main distinction is exactly the trap
  `memory/venue-verify-acl-family-needs-anthology.md` warns about → stated explicitly.
  **Do not cite as NAACL main.**
* **What it does**: standardises **both** the multiple-choice (MCF) and cloze (CF)
  formulations, then takes the **better score per task and model**; reviews probability
  normalisation as a result-changing free parameter.
* **Overlap**: the letter-vs-content interface split and the fact that it changes measured
  performance are **established and standardised, and not A01's**.
* **Difference / the defect A01 fixes**: OLMES's `max(MCF, CF)` rule is **not size-keyed**
  (`02_related.tex:9` corrects an earlier A01 misdescription of it), but its reference
  discussion is framed around a **random baseline** rather than a measured label-marginal
  floor — so the max rule can select an interface **without testing whether that interface
  clears an input-blind reference**. A01's ARC-Easy result is a case where CF rescues an
  at-floor letter read-out (OLMo-2 `keep8`: letter 0.2584 at its 0.266414 floor vs
  `content_norm` 0.6460, paired gap +38.76 pp, McNemar $p=9.8\times10^{-148}$), and MMLU is a
  case where the interface swap does **not** rescue it. **That is a constructive, citable
  delta: key the interface decision on the arm's own floor test, not on a max over scores.**
* **Preempts? NO.**

### 2.5 Cho, So, Lee — *Choices Speak Louder than Questions*

* **Venue: ICLR 2026 Poster** (arXiv:2502.18798v4). Verified via **OpenReview
  `venueid = ICLR.cc/2026/Conference` + `Submission17078/-/Camera_Ready_Revision`** in
  `NOVELTY_CHECK.md` §2.3. ⚠️ Not re-verified this session (api2 blocked). ⚠️
  `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §2.3 flags that the text read was **arXiv v4
  (2026-01-12), which still says "Under review"**, vs camera-ready `pdate` 2026-01-26 — so
  **the camera-ready was never diffed**. Recorded as a gap, not smoothed.
* **What it does**: log-likelihood scoring and its length-normalised variant are "vulnerable to
  superficial characteristics of the answer choices"; experiments across cloze / symbol /
  hybrid formats; proposes a new question-sensitive score (NPSQ) isolating the question's
  contribution — conceptually an input-blind subtraction.
* **Overlap**: jointly with §2.1 and §2.4 this establishes that **MC interface validity is an
  active concern**. A01 must not present that concern as unprecedented.
* **Difference**: its remedy is **a new metric proposed as generally better**; A01 is
  **metric-agnostic** — report *any* metric against *that metric's own* input-blind null and
  refuse cross-arm comparison when an arm cannot clear it. Intact models only; no damage axis,
  no floor-estimator calibration, no convention analysis. `02_related.tex:9`: *"We do not claim
  the interface contrast; our method instead gates whichever construct is reported."*
* **Preempts? NO.**

### 2.6 Bean, Kearns, Romanou, et al. — *Measuring what Matters: Construct Validity in Large Language Model Benchmarks*

* **Venue: NeurIPS 2025 Datasets & Benchmarks Track (poster)** (arXiv:2511.04703). Verified via
  **OpenReview forum `mdA5lVvNcU`, `venueid = NeurIPS.cc/2025/Datasets_and_Benchmarks_Track`
  + `Submission1976/-/Camera_Ready_Revision`** in `NOVELTY_CHECK.md` §2.4. ⚠️ Not re-verified
  this session (api2 blocked).
* **What it does**: 29-reviewer systematic review of **445** benchmarks; finds
  validity-undermining patterns in phenomena, tasks, and **scoring metrics**; issues eight
  recommendations. It **owns the construct-validity vocabulary A01 uses.**
* **Overlap**: framing only. A01 must not claim to have introduced construct validity to LLM
  evaluation.
* **Difference**: it is a **survey with normative recommendations** and **does not run the
  measurement**. `01_introduction.tex` makes the sharpest available version of this point:
  the review offers **27 actionable checklist items, none of which asks authors to report a
  null, a chance level, or a constant predictor.** A01 is the **missing operational item,
  executed**. `02_related.tex:17`: *"We position null calibration as the missing operational
  item, not as the invention of construct validity."*
* **Preempts? NO.**

### 2.7 ⚠️ THE COLLISION FAMILY THE GAP AUDIT DID NOT NAME — chance-corrected agreement, and it is 1954–1988 psychometrics

This is the section that matters most, because it attacks the **two residual contributions**
the manuscript still claims after §0.1's concession: **(i)** stratifying the permutation within
variable option count, **(ii)** using the quantity as an arm-conditional pre-comparison gate
with a materiality bar. The gap audit listed "MC/QA option priors" but **not** the
chance-correction literature, and that literature is exactly where "correct for the number of
options" lives. Venues verified via **Crossref DOI** (the correct authority — none of these are
in Anthology or OpenReview).

| Paper | Year | Venue (Crossref-verified) | What it does | Distance from A01's (i)/(ii) |
|---|---|---|---|---|
| **Cohen**, *A Coefficient of Agreement for Nominal Scales* | 1960 | *Educational and Psychological Measurement* **20**(1):37–46, DOI `10.1177/001316446002000104` | $\kappa=(p_o-p_e)/(1-p_e)$ with $p_e$ from the **observed marginals of both raters**. | **Already cited** (`cohen1960kappa`) and already conceded: $\Delta_{\mathrm{perm}}$ is its numerator, collapse-zero is its $p_o=p_e$ property. |
| **Bennett, Alpert, Goldstein**, *Communications Through Limited Response Questioning* (Bennett's **S**) | 1954 | *Public Opinion Quarterly* **18**:303, DOI `10.1086/266520` | Chance corrector using $p_e=1/k$, i.e. **$k$ = the number of response categories** — the coefficient whose whole point is that the correction depends on **how many options there are**. | ⚠️ **This is the closest thing to (i) in existence.** It is option-count-aware *by construction*. But it uses **one global $k$**, assumes categories are **equiprobable**, and does **not** handle $k$ **varying item-to-item** — which is precisely MMLU-Pro (`n_opt` 3–10, 8 strata). It also has no per-item legality notion. |
| **Brennan & Prediger**, *Coefficient Kappa: Some Uses, Misuses, and Alternatives* | 1981 | *Educational and Psychological Measurement* **41**(3):687–699, DOI `10.1177/001316448104100307` | The canonical critique of $\kappa$'s marginal-dependence, recommending the free-marginal ($1/k$) alternative in exactly the situations where $\kappa$ misleads. | Same limitation: **one $k$ for the whole table.** Establishes that **choosing $p_e$ is a live methodological decision** — which A01 must credit rather than present as its own insight. |
| **Lord**, *Formula Scoring and Number-Right Scoring* | 1975 | *Journal of Educational Measurement* **12**(1):7–11, DOI `10.1111/j.1745-3984.1975.tb01003.x` | Formula scoring, i.e. correction for guessing on multiple-choice tests. | Corrects the **examinee's score** for guessing; A01 corrects the **reference line** for an arm's prediction vector. Different object; must not be claimed as A01's insight either. |
| **Frary**, *Formula Scoring of Multiple-Choice Tests (Correction for Guessing)* | 1988 | *Educational Measurement: Issues and Practice* **7**(2):33–38, DOI `10.1111/j.1745-3992.1988.tb00434.x` | The standard practitioner treatment; the correction is explicitly a function of the **number of options per item**. | Same: option-count-aware guessing correction is **decades old**. A01 may not claim "correct for the number of options" as new **in any form**. |
| **De Vries, Elliott, Kanouse, Teleki**, *Using Pooled Kappa to Summarize Interrater Agreement across Many Items* | 2008 | *Field Methods* **20**(3):272–282, DOI `10.1177/1525822X08317166` | **Pooled** $\kappa$ across many items, shown more efficient than averaging item-level $\kappa$s. | **This is the closest thing to (i) in the modern statistics literature**: aggregating a $\kappa$-type quantity across heterogeneous item groups. But it pools **across items to summarise raters**; A01 stratifies **by option count to keep an illegal letter from ever being credited** — an item-**legality** constraint, not an efficiency device. |
| **Brenner & Kliebsch**, *Dependence of Weighted Kappa Coefficients on the Number of Categories* | 1996 | *Epidemiology* **7**(2):199–202, DOI `10.1097/00001648-199603000-00016` | Shows weighted $\kappa$ **varies systematically with the number of categories**. | Directly establishes that **$k$-dependence of a $\kappa$-family statistic is a known problem**. A01 must cite this class rather than discover it. |

**Adjudication of (i) — stratifying the permutation within variable option count.**

The *idea* "the chance term must depend on how many options there are" is **Bennett 1954 /
Lord 1975 / Frary 1988**, and the *problem* "a $\kappa$-family statistic drifts with $k$" is
**Brenner & Kliebsch 1996**. Those are citation obligations and they are not currently in
`paperC/refs.bib` (which holds 12 entries; only `cohen1960kappa` is from this literature —
verified this session).

What I could **not** find, after targeted searching (§4), is any prior work that computes a
chance-corrected agreement statistic **where $k$ varies item-to-item within the same
evaluation set, using per-stratum uniform permutation so that letters illegal for an item are
never credited**. Bennett/Brennan–Prediger use a single global $k$. Pooled $\kappa$ aggregates
across items but for a different purpose. So **(i) survives, but at the strength of "not
found", which is weaker than "does not exist"** — the same standard `NOVELTY_CHECK.md` §2.7
already applies to A01's layer-order null. **The honest phrasing is "we are not aware of a
prior chance-corrected agreement estimator stratified by per-item option count", not "first".**

> ⚠️ **And (i) is small.** The manuscript's own §"Constant-collapse identity" and the
> ordering paragraph in `03_method.tex:62` show that stratification's *measurable*
> consequence on MMLU-Pro is **36 items = 0.299 pp** (the gap between
> $\sum_s w_s\max_L m_{s,L}=1439/12032$ and $f_{\mathrm{const}}=1403/12032$). That is the
> whole numerical footprint of (i) on the headline benchmark. A reviewer who reads
> `03_method.tex` will see that A01 itself computed it. **Do not oversell (i) as a
> methodological advance; sell it as the reason a specific loophole is closed by
> construction.**

**Adjudication of (ii) — the pre-comparison gate + materiality bar. This is the stronger half.**

Nothing in the seven psychometrics rows, and nothing in §§2.1–2.6, uses a chance-corrected
quantity as a **precondition on whether a number may enter a comparison at all**.
`paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §3 records this as a full-text-verified negative:
*"Nothing in the nine candidates computes a best-constant/input-blind null per construct and
uses it as a precondition on whether an arm's number may enter a comparison at all — Q2 is NO
in 8 of 9, and the one YES (Hewitt & Liang) is on probe accuracy with a randomised-label
control, not on MC accuracy with an input-blind constant."* The agreement literature's use of
$\kappa$ is **descriptive** — report reliability, interpret against Landis–Koch-style verbal
bands. A01 uses it **prescriptively and asymmetrically**: below the bar, the arm is
disqualified from comparison; above it, nothing is certified. Combined with an explicit
numeric materiality constant (`recovery_fraction` ≥ 0.10 × same-family intact anchor, and
**blocked entirely** if the intact anchor is itself below 0.10 — which is what actually happens
to Llama-2, `recovery_fraction`=0.0545), that is a decision procedure, not a score.

**Verdict on §0.1's residual: (ii) stands and is the defensible core; (i) stands only at
"not found" strength and is numerically small (0.299 pp on MMLU-Pro).** Evidence for both is
in §2.7 above and §4's query list. This is the answer to "do (i) and (ii) still hold up".

### 2.8 The similarity-null / probe-control leg (where A01 already concedes priority)

Named because A01's **kill clause 3** is specifically about this leg, and it must be shown
**not** to carry the paper.

* **Hewitt & Liang**, *Designing and Interpreting Probes with Control Tasks* — **EMNLP-IJCNLP
  2019 MAIN**, Anthology `D19-1275`, pp. 2733–2743, DOI `10.18653/v1/D19-1275`.
  **Verified this session** via Anthology `.bib`. (Findings did not exist in 2019, so "main" is
  unambiguous.) **Closest structural precedent**: a null that **reverses a within-model
  component comparison** — theirs randomises *supervision for a probe*, A01's removes *input
  dependence from an MC construct*. `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §3 argues this
  is the honest genealogy to cite, stronger than "probes need a null".
* **Feng, Wallace, Boyd-Graber**, *Misleading Failures of Partial-input Baselines* — **ACL 2019
  MAIN (short)**, Anthology `P19-1554`, pp. 5533–5538, DOI `10.18653/v1/P19-1554`.
  **Verified this session** via Anthology `.bib`. **Load-bearing counter-citation**: a
  partial-input baseline *failing* does not certify a dataset is artifact-free. This is why
  claim 1 must stay one-directional. `01_introduction.tex` and `03_method.tex:18` both carry it.
  (⚠️ `PROPOSAL.md:361` still cites this as "arXiv:1905.05778, venue 待核实" — **now verified,
  and the arXiv ID there is wrong for the Anthology record**; use `P19-1554`.)
* **Ding, Denain, Steinhardt**, *Grounding Representation Similarity **Through** Statistical
  Testing* — **NeurIPS 2021 Poster**, DBLP `conf/nips/DingDS21`. Verified in
  `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §1.1 (OpenReview forum `_kwj6V53ZqB` +
  `venueid = NeurIPS.cc/2021/Conference` + DBLP + official proceedings). ⚠️ **Cite the
  camera-ready title, which says "Through"** — arXiv:2108.01661 says "with". Sensitivity/
  specificity testing for CKA-style measures; **prior art A01 already concedes**.
* **Layer-order null**: `NOVELTY_CHECK.md` §2.7 reports zero relevant hits for
  `"layer permutation" AND "similarity"` / `"permutation null" AND "representational
  similarity"`. Keep the **downgraded** phrasing ("we are not aware of a prior layer-order
  null"), never "first".

**Consequence for kill clause 3**: the similarity-null prior art is real and conceded, but
A01's surviving findings (§0 claims 1–7) live on the **MC-accuracy construct**, so the paper
does not "degenerate into a case collection of existing similarity-null methods". **Clause 3
does not fire.** Unchanged from `NOVELTY_CHECK.md`.

### 2.9 Concurrent / adjacent work found this session (none preempts)

| Paper | arXiv | Venue | Why it is adjacent, and why it is not preemption |
|---|---|---|---|
| Arčon, Klemen, Robnik-Šikonja, Dobrovoljc, *Evaluating Metalinguistic Knowledge in LLMs across the World's Languages* | 2602.02182 (2026-02-02) | **arXiv-only** (DBLP: no record found this session; no comment/jref) | **The paper A01's own introduction opens with** (`arcon2026metalinguistic`). Abstract, fetched verbatim this session: *"Although all models perform above chance, they fail to outperform the majority-class baseline."* **It reports the exact flip A01 measures** — but **descriptively, on one new WALS-derived benchmark, on intact models**, because it happened to print both reference lines. It does **not** turn it into a pre-comparison protocol, has no damage axis, no floor-estimator calibration, no convention analysis, no arm-conditional null. `01_introduction.tex` and `02_related.tex:4` both credit it correctly. **Not preemption** — and citing it is *stronger* than not, since it is independent confirmation that the flip is real in the wild. |
| Anon., *The Illusion of Generalization in Tabular Language Models* | 2602.04031 (2026-02-03) | **ICML 2026** per its own arXiv `jref` (*"In Proc. 43th International Conference on Machine Learning (ICML 2026)"*) — ⚠️ **`venueid` NOT verifiable this session (api2 blocked); DBLP has no record. Treat as jref-self-reported.** | Finds *"near-zero median lift over **majority-class baselines**"* for Tabula-8B on 165 datasets, and attributes claimed generalization to **evaluation artifacts**. Same logical move as A01 (majority baseline dissolves a capability claim) in a **different modality (tabular prediction)**, plus contamination analysis A01 does not do. **Not preemption**: no MC letter/content interface, no damage axis, no null-calibration protocol, no floor-estimator calibration. **This is a corroborating cell, and by `memory/prior-work-differentiate-dont-abandon` the right response is to cite it as independent evidence in a second modality** — the same cross-modality-convergence framing A04's §4 reframe established. |
| Anon., *How Many Tools Should an LLM Agent See? A Chance-Corrected Answer* | 2605.24660 (2026-05-23) | **arXiv-only** (13 pages, no comment/jref; DBLP not checked positive) | Applies **Bits-over-Random**, an explicitly **chance-corrected** metric, to tool-shortlist depth — and the correction **depends on the number of candidates shown**, structurally analogous to A01's `n_opt` dependence. **Not preemption**: retrieval/agent setting, log-ratio metric not a permutation null, no MC construct, no gate semantics. **But it is evidence that "chance correction must track the option count" is being independently rediscovered in 2026**, which is another reason not to overstate (i). |
| Anon., *The 99% Success Paradox: When Near-Perfect Retrieval Equals Random Selection* | 2605.18857 (2026-05-14) | **ICLR 2026 Blog Track** per arXiv comment + `jref` (`ICLR Blog Track 2026`, `iclr.cc/virtual/2026/poster/10012083`). Blog track ≠ main conference — **do not cite as ICLR 2026 main.** | Same shape: >99% reported success with $BoR\approx 0$, i.e. a high score that is **random-level once chance-corrected**. **Not preemption** (IR/RAG, hypergeometric baseline, no MC, no damage), but a clean same-year external example of the phenomenon A01 formalises for MCQA. |
| Anon., *Language Models Agree With Each Other, Not With Readers* | 2607.29274 (2026-07-31) | **arXiv-only** (DBLP `journals/corr/abs-2607-29274`, CoRR 2026) | Uses a **within-stratum resampling null** ("resampled within its own depth-and-length bands") and **demonstrates the null's calibration rather than asserting it** ("every pair involving a random baseline lands within 0.006 of zero"). That is *methodologically* the same discipline as A01's stratified permutation + its ten-letter zero self-test. **Not preemption**: different task (sentence-highlight agreement vs MC accuracy), no interface/floor claim, no damage axis. **But it is the single closest methodological sibling found this session and should be cited if A01 keeps claiming (i) as a discipline rather than a statistic.** |

**None of the five is 完全相同/抄袭.** Three are 2026-02 to 2026-07, i.e. **inside the
concurrency window** for a 2026-08-15 pass (≥ 2026-05-15 for the strictest reading; the two
February papers are outside it and are therefore treated as real prior art to cite).

---

## 3. MUST-NOT-CLAIM list (binding on paperC and on any A01 writeup)

This list is the **union** of `NOVELTY_CHECK.md` §4, `tab_claims.tex`, and §2.7's new findings.
Items **1–7** are pre-existing and unchanged. Items **8–11** are added by this pass.

1. ❌ First to recommend a **stronger-than-chance / majority baseline for MCQA**.
   **Balepur et al., ACL 2024 main** owns it.
2. ❌ First to observe that a **constant, input-independent predictor** can score high on a
   benchmark, or the term **"null model"**. **Zheng et al., ICLR 2025 Oral** owns both.
3. ❌ **`acc` vs `acc_norm` length sensitivity**, or the **generic tokenizer-dependence of
   length-based MC scoring**. **Oostermeijer, ICML 2026** owns both.
   (`tab_claims.tex` row 5 = *Preempted*. A01's residual is **floor inheritance**, i.e. that
   the *input-blind null* inherits those choices — never the scoring-rule observation.)
4. ❌ First to establish the **letter/cloze (MCF/CF) interface split** or that it changes
   results. **OLMES, Findings of NAACL 2025** owns it. And do not describe OLMES's rule as
   **size-keyed** — `02_related.tex:9` already corrected that; it is `max(MCF, CF)` per task
   and model.
5. ❌ First to worry that the **MC scoring interface can invalidate conclusions**.
   **Cho et al., ICLR 2026** + §2.1 jointly own it.
6. ❌ First to bring **construct validity** to LLM benchmarks. **Bean et al., NeurIPS 2025
   D&B** owns it. (A01's licensed sharper form: their **27 checklist items include none that
   asks for a null / chance level / constant predictor.**)
7. ❌ That **clearing a floor certifies validity**. **Feng et al., ACL 2019 main** forbids it;
   the claim must stay one-directional. (`tab_claims.tex` row 15.)
8. ❌ **NEW — the $\Delta_{\mathrm{perm}}$ statistic itself, or the constant-collapse zero.**
   **Cohen 1960** owns the statistic (A01's is its numerator, $\Delta_{\mathrm{perm}}=\kappa(1-p_e)$
   identically); the collapse-zero is the textbook $p_o=p_e$ property. **The manuscript already
   concedes this at `03_method.tex:36-48` and `02_related.tex:15`; commit `7d39546` retracted it.
   Re-claiming it is a regression.**
9. ❌ **NEW — "the chance term should depend on the number of options"** in any form.
   **Bennett/Alpert/Goldstein 1954** (Bennett's $S$, $p_e=1/k$), **Brennan & Prediger 1981**
   (free-marginal alternative), **Lord 1975** and **Frary 1988** (formula scoring / correction
   for guessing, per-item option count) own it, 38–72 years earlier. A01's residual is
   **per-item-varying $k$ with per-stratum uniform permutation and per-item legality**, and
   even that must be phrased as "not aware of", not "first".
10. ❌ **NEW — that a $\kappa$-family statistic's $k$-dependence is a novel observation.**
    **Brenner & Kliebsch 1996** owns it; **De Vries et al. 2008** owns pooling a $\kappa$-type
    quantity across heterogeneous item groups.
11. ❌ **NEW — priority for "a high benchmark score can be at random level once
    chance-corrected"** as a 2026 observation. Independently present in
    **arXiv:2605.18857 (ICLR 2026 Blog Track)** and **arXiv:2605.24660**, and *descriptively*
    in **arXiv:2602.02182**, which A01's own introduction already cites. A01's contribution is
    the **protocol and the gate**, not the phenomenon.

Also still binding from `tab_claims.tex` (internal, not literature): rows 1, 3, 4, 7, 8, 11,
12, 14 are **Retracted**; row 9 and row 16 are **Prohibited**. In particular do not resurrect
4.8×, 0.2822, 58/91, MMLU 0.25, BoolQ 0.50, or 45.74 pp.

---

## 4. Safe residual claim — one falsifiable sentence

> **On likelihood-scored multiple-choice constructs, the input-blind reference is (a) not
> chance, (b) under-specified by tie convention / length unit / tokenizer, and (c) itself an
> upward-biased estimator; and when it is calibrated and applied as an arm-conditional
> pre-comparison gate with an explicit materiality bar, it re-sorts existing published
> verdicts in BOTH directions — withdrawing above-floor capability labels and dissolving
> below-floor competence labels — on structurally damaged models, a regime absent from every
> verified prior work above.**

**How to falsify it, per clause** (each already has an instrument in the manuscript):
* (a) fails if some construct's best constant equals chance to within its own floor-estimator
  noise **for the constructs the paper rests the quantitative claim on** — it already
  concedes this happens for 5 of 8 letter constructs ($p=0.14$–$0.85$, `tab_nulls.tex`) and
  restricts the claim to MMLU-Pro/MMLU/BoolQ.
* (b) fails if the floor is invariant to convention/unit/tokenizer — refuted by the 40.6-pp
  `wrong`→`credit` span and the 0.9003-pp tokenizer span on fixed items.
* (c) fails if $\hat f$ is unbiased under a balanced null — refuted by
  $\mathbb{E}[\hat f]>\text{chance}$ in all 8 rows of `tab_nulls.tex`.
* **The re-sorting clause is the load-bearing one and is the easiest to kill**: it fails if v2
  only ever *shrinks* effects. Refuted by `05_analysis.tex` §"V2 re-sorts": `qwen3/k14` moves
  **down** (above-floor label → `TRACE_SIGNAL`) while `olmo2/keep14` moves **up** (at-floor →
  `TRACE_SIGNAL`). Movement in both directions is what distinguishes a criterion from a
  conservatism penalty.

**Searches run this session that returned ZERO relevant results** (this is what "not found"
rests on; arXiv full-text/abstract search via `arxiv.org/search`, plus Crossref
`query.bibliographic`):
`"chance-corrected" AND "multiple-choice"`; `"Cohen's kappa" AND "multiple-choice" AND "large
language model"` (2 hits, both unrelated: 2602.06446 ontology relations, 2510.13734 clinical
benchmark); `abs:"varying number of options" AND abs:"multiple-choice"`;
`"unequal number of options"`; `abs:"stratified permutation" AND abs:"evaluation"`;
`all:"permutation null" AND all:"benchmark" AND all:"language model"`;
`"best-constant baseline"`; `"input-blind"` (7 hits, all unrelated — structural dynamics,
immunology, fighting-game audio); `"majority baseline" "MMLU"`; `abs:"MMLU-Pro" AND
abs:"chance"`; `"MMLU-Pro" "chance level"`; `"option count" "multiple-choice" "chance"`;
`"prediction marginal" "permutation" "accuracy"`; `"winner's curse" "baseline" "benchmark"
"language model"`; Crossref `stratified kappa varying number of categories agreement`,
`kappa coefficient different number of categories per item`, `extension of kappa to unequal
number of response categories across subjects`, `Mantel-Haenszel kappa stratified`.

---

## 5. Honest gaps in this adjudication

1. ⚠️ **The repo's mandated OpenReview route is DOWN from this node.** `api2.openreview.net`
   returns HTTP 403 `ChallengeRequiredError` on **every** path. Only API **v1** works, and v1
   exposes `venue`/`venueid` but **not** the invitation list — so **no `Camera_Ready_Revision`
   check was performed this session.** The four OpenReview-family venues
   (Oostermeijer ICML 2026, Cho ICLR 2026, Zheng ICLR 2025, Bean NeurIPS 2025 D&B) are
   carried from `NOVELTY_CHECK.md` (2026-08-09) and `paperC/VENUE_AND_NOVELTY_VERIFICATION.md`
   (2026-08-12), **both of which did reach api2**. They are **second-hand for this session**
   and are marked as such at each row. Re-verify before submission.
2. ⚠️ **arXiv API 429 and Semantic Scholar 429** all session. arXiv metadata came from
   `arxiv.org/abs/<id>` HTML `citation_*` meta + the comments/jref table cells. **No venue
   claim rests on S2** (per repo rule it is only a cross-check).
3. ⚠️ **`2602.04031` (Illusion of Generalization) venue is jref-self-reported only.** Its
   arXiv `jref` says ICML 2026, DBLP has no record, and api2 was blocked. Do **not** enter it
   into `refs.bib` as ICML 2026 until `venueid` is confirmed.
4. ⚠️ **`2602.02182` — the paper A01's introduction opens with — is `arXiv-only` from this
   node.** DBLP returned no record; it has no comment and no jref. `refs.bib` currently cites
   it as `@article{...arXiv preprint arXiv:2602.02182}`, which is **correct and honest**. If it
   is later published, update; do not upgrade it speculatively.
5. **No full-text PDF was read this session.** All overlap/gap judgements for §2.9's five new
   papers are from **abstract + venue metadata**. The §§2.1–2.6 judgements rest on
   `paperC/VENUE_AND_NOVELTY_VERIFICATION.md` §2, which **did** do a nine-PDF `pdftotext`
   pass. The **psychometrics rows in §2.7 were verified at Crossref-metadata level only** —
   title, journal, volume, pages, DOI, year. **I did not read Bennett 1954, Brennan &
   Prediger 1981, Lord 1975, Frary 1988, Brenner & Kliebsch 1996, or De Vries et al. 2008 in
   full.** Their *characterisation* above is from standard knowledge of these coefficients plus
   the Crossref titles/abstracts, and the load-bearing negative — "none of them handles $k$
   varying item-to-item" — is an inference from what those coefficients *are*, not from a
   full-text grep. **Before paperC's camera-ready, at least Bennett 1954 and
   Brennan & Prediger 1981 need a full-text pass**, because they are the two that could
   plausibly contain a stratified variant in a section an abstract would not mention. This is
   the single largest remaining risk to contribution (i).
6. **Zero cross-disk verification.** Every path cited here is on **wzc1**; the A01 evidence on
   **zwfy6** (`results/a01_gate3/dtype_runs/*`, three families' depth curves — see
   `SOURCES.md` §"On zwfy6") was **not** `ls`-confirmed from this session.
   Per `memory/two-disk-rule-applies-to-main-too.md`, absence is not established until both
   disks are checked — and nothing here claims any such absence.
7. **No `.bib` entries emitted.** Per `memory/tcodex-exec-no-dash-c-flag.md` and
   `memory/venue-verify-acl-family-needs-anthology.md`, entries must not enter the
   bibliography until venue-verified by family. The **seven psychometrics DOIs in §2.7 are
   Crossref-verified and safe to add**; the five §2.9 papers are **not** (three are
   `arXiv-only` and one is jref-only).
8. **`PROPOSAL.md` was not edited by this pass.** The gap audit asks for a closest-collision
   table "in `PROPOSAL.md`"; it is here instead, because (a) `ready_queue.py` reads
   `RELATED_WORK.md`, and (b) A01 is promoted, so the manuscript — not the proposal — is the
   live artifact. Two stale spots in `PROPOSAL.md` should be fixed by whoever next touches it:
   line 361 cites Feng et al. as "arXiv:1905.05778, venue 待核实" (**now verified as
   `P19-1554`, ACL 2019 main**, and that arXiv ID does not match the Anthology record), and
   `## 可以主张` item 2 is wider than what `03_method.tex` now claims.

---

## 6. Verdict

```
related_work_status: audited
novelty_status: kill clause 3 DOES NOT FIRE (unchanged); no paper preempts A01
strongest_collision: Balepur et al., ACL 2024 main (Anthology-verified) -- owns the
                     stronger-than-chance recommendation; does NOT reach A01's damaged
                     regime, per-arm gate semantics, floor-estimator calibration, or
                     null-convention analysis
new_this_pass:       the chance-correction psychometrics family (Bennett 1954 / Brennan &
                     Prediger 1981 / Lord 1975 / Frary 1988 / Brenner & Kliebsch 1996 /
                     De Vries 2008) was NOT in the gap audit and adds 4 must-not-claim items
residual_holds:      (ii) pre-comparison gate + materiality bar -- YES, strongest
                     (i)  per-item-varying-k stratification -- YES but only at
                          "not found" strength, and its measured footprint on MMLU-Pro is
                          36 items = 0.299 pp
```

No candidate is 完全相同/抄袭. Every collision differs on at least one load-bearing axis:
**damaged vs intact regime**, **gate vs description**, **derived vs crafted null**,
**protocol vs metric**, or **per-item-varying $k$ vs one global $k$**. Per
`memory/prior-work-differentiate-dont-abandon.md` the correct output of this pass is a
**citation-obligation list plus two places where A01 fixes prior work** (Balepur's 0.25
imputation vs the construct's 0.2689 floor; OLMES's untested interface selection), **not** a
scope reduction — and specifically **not** a claim that A01 is small because §0.1 gave the
statistic away. The paper's contribution is the **calibrated protocol and the verdict flips it
produces**, which no verified prior work performs.
