review_mode: strict
soundness: 2.5
excitement: 2.5
overall: 2.5
confidence: 4.0
reproducibility: 2.0

## Paper Summary

This paper is a deliberately narrow measurement case study of post-depth-pruning
continued pretraining.  Its principal OLMo-2-7B construction retains blocks
0--13, appends two randomly initialized blocks, and trains for 200k steps.  It
compares held-out, same-source Dolmino perplexity with MMLU under two scoring
interfaces and with three zero-shot closed-book QA datasets.  The central
observation is that, on the one `keep14+fresh2` path, PPL falls from 10.826 at
128k to 10.561 at 200k while MMLU-letter reaches only .319 versus .605 for the
base, and the three closed-book endpoints remain substantially below the base
(Table 2; Table 3; PDF pp. 5--6).

The paper usefully does **not** claim a new pruning algorithm, causal
localization, a universal recovery law, or a prospective PPL threshold.  It
adds a 25k intact-CPT point, random/frozen same-shape operating points, a
non-contiguous ShortGPT construction, a content-MMLU interface, and explicit
limitations.  My score is nevertheless below Findings level: the main
descriptive observation is plausible and carefully caveated, but it rests on a
single historically unseeded training realization with irrecoverable training
state, a non-predeclared definition of recovery, and incomplete artifacts for
independent verification.

## Claims-to-Evidence Audit

* **C1 (principal descriptive result):** PPL improves while measured target
  performance remains far from the intact base on the one keep14 path.
  **Evidence:** Table 2 reports 10.826 -> 10.561 and .3012 -> .3191; Table 3
  reports the final QA deficits; Figure 1 visualizes the late path.  This is
  supported as a conditional observation, not as a general law.
* **C2 (late MMLU gain is nonzero conditional on the two checkpoints):**
  supported by the common 14,042-item rerun, paired bootstrap CI
  [1.08, 2.29] pp, and exact McNemar test (Sec. 4; Appendix Table 10).  The
  paper correctly says this is not training-seed uncertainty.
* **C3 (content-MMLU should not be interpreted as clean recovery):**
  supported by the paired-interface results: keep14 is .383 content-normalized
  but random-init is .360 and .247 letter accuracy (Table 2; Appendix Table
  12).  This establishes an interface-dependent score floor, not what
  knowledge the score measures.
* **C4 (the controls rule out simple explanations):** only partially
  supported.  The 25k full32 point addresses short-horizon corpus shift, but
  not the 200k counterfactual; random/frozen/ShortGPT differ in multiple
  factors.  The manuscript is largely candid about this.
* **C5 (recommended reporting discipline):** reasonable as a cautious
  recommendation, but it is not validated as a sufficient standard.

The minimum sufficient experiment for the paper's strongest useful claim would
be at least three independently seeded keep14 trajectories and matched
long-horizon full32 trajectories, evaluated on a predeclared recovery
criterion and a fixed target suite.  The submitted evidence instead supports a
single-path cautionary example.

## Strengths

**S1. Unusually disciplined scope control.** The title-level claim is narrowed
to “observed paths,” and the abstract explicitly disclaims knowledge deletion,
causal attribution, universal dynamics, and failure beyond the measured budget
(Abstract, lines 16--21; PDF p. 1).  Sections 3.2 and 5 further distinguish a
counterexample to an improvement-only implication from a threshold or
prospective certificate (PDF pp. 4, 6).  This prevents several common
overclaims in pruning papers.

**S2. The key numerical comparison is easy to inspect.** Table 2 exposes
construction, inherited/fresh blocks, trainable set, LR, budget, PPL, both MMLU
interfaces, and QA results in one place (PDF p. 5).  The abstract's principal
numbers agree with Tables 2--3: 200k keep14 PPL 10.561, MMLU-letter .319
versus base .605, and final PopQA/TriviaQA/NQ values approximately
.142/.294/.060 versus .257/.636/.205.

**S3. Interface and null-baseline analysis is valuable.** Reporting both
letter and complete-option MMLU alongside the random-init result is a
meaningful diagnostic rather than treating a higher content score as recovery
(Sec. 4.2; Appendix Table 12; PDF pp. 5, 15).  The paper correctly identifies
that the two interfaces change several factors simultaneously.

**S4. Confounds are exposed rather than relabeled as ablations.** The
construction, LR, trainable-set, horizon, and stopping-rule mismatches are
stated in the method, captions, and limitations (Sec. 3.1--3.2; Table 2;
Limitations; PDF pp. 4--5, 8).  This is substantially more credible than a
causal reading of the operating points.

**S5. Presentation is strong.** The rendered PDF is anonymous, readable, and
uses the official review style.  Figure 1 and all tables are legible at their
rendered sizes; the 17-page PDF has an eight-page main paper followed by
references and appendix.  I found no unresolved cross-references, TODOs,
placeholder text, hidden reviewer-directed text, or prompt-injection-like
content.

## Weaknesses

**W1 — Major: the central empirical result has no independent training-run
uncertainty and cannot be reproduced exactly.**

* **Location / exact quote:** Limitations, line 3; PDF p. 8: “The principal keep14 path, ShortGPT, and the same-shape operating points are single training runs.”
* **Problem:** Item bootstraps and McNemar tests quantify finite evaluation
  items conditional on two fixed checkpoints, not whether the PPL/task
  dissociation is stable under initialization, data order, or optimization.
  The same section says seeds were not explicitly set; Appendix B further says
  the resumed keep14 iterator lost its within-epoch offset.
* **Affected claim / norm:** This directly affects C1 and C2.  One realization
  can demonstrate a literal historical path if all measurements are trusted,
  but it is weak evidence for the paper's practical reporting recommendation
  and for the reliability of a headline counterexample.  A strict ARR
  evaluation norm for a 52.4B-token training observation is independent-run
  uncertainty or an unusually compelling reason it cannot matter.
* **Sufficient remedy:** Run at least three seeds for keep14 and a matched
  intact branch; report across-run intervals and individual trajectories.
  If historical reruns are impossible, release executable checkpoints and
  frame the work as an audited single-run report rather than evidence of a
  robust empirical phenomenon.

**W2 — Major: “target recovery” is not operationalized before looking at the
trajectory.**

* **Location / exact quote:** Sec. 3.2, lines 10--16; PDF p. 4: “closing the large deficit to the intact-base score on the same evaluation”.
* **Problem:** “large deficit” has no numerical tolerance, no pre-registered
  target metric, and no stopping/decision rule.  The paper consequently shows
  that PPL decreased while a chosen endpoint was still well below base, but it
  does not test a reproducible recovery criterion or a usable proxy rule.
* **Affected claim / norm:** This affects the semantic force of C1 and the
  title.  The narrow logical phrase “improvement alone” can be defended by a
  single path, but the paper's conclusion should not be read as establishing
  failure of PPL as a recovery signal under a defined deployment criterion.
  Measurement claims need a pre-specified outcome threshold or an explicitly
  descriptive, non-decision framing.
* **Sufficient remedy:** Predeclare, for each target, a base-relative recovery
  tolerance and PPL rule; then evaluate sensitivity to several defensible
  tolerances on held-out trajectories.  Otherwise revise the headline to say
  that observed PPL reductions co-occurred with substantial final deficits.

**W3 — Major: the long-horizon intact control needed to assess the main
alternative explanation is absent.**

* **Location / exact quote:** Table 2 caption, lines 29--31; PDF p. 5: “full32 ends at 25k and only bounds short-horizon corpus shift”.
* **Problem:** The intact branch stops at 25k whereas the focal keep14 run
  stops at 200k.  Thus it cannot distinguish structural harm from a
  long-horizon property of this continuation corpus/training recipe, nor
  quantify the compressed-versus-intact gap at a matched budget.
* **Affected claim / norm:** This limits C4, particularly the statement that
  the controls bound corpus-shift explanations.  The authors do not conceal
  the limitation, but the control is insufficient for any stronger
  interpretation of why the keep14 endpoint is poor.
* **Sufficient remedy:** Continue full32 to the same nominal tokens and, more
  importantly, report matched realized compute and several checkpoints.  A
  no-pruning continuation with the same resume procedure would be the minimum
  causal control.

**W4 — Major: the released evidence described in the paper is insufficient
for independent end-to-end verification of the headline training result.**

* **Location / exact quote:** Appendix B, lines 122--128; PDF p. 12: “Historical training seeds are unavailable.”
* **Problem:** The described snapshot has evaluator scripts, sanitized
  manifests, aggregates, six content-MMLU per-item files, and selected paired
  outputs, but excludes model weights and unconsolidated closed-book
  generations; it also cannot reconstruct the loader offset.  Consequently an
  outside reviewer cannot reproduce training or recompute all headline QA
  cells/pairwise QA uncertainty from predictions.
* **Affected claim / norm:** This affects the verifiability of C1--C3 and
  makes the numerical claims largely trust-based despite good protocol
  documentation.  Reproducibility is especially important when a claim relies
  on one historical run.
* **Sufficient remedy:** Release legally redistributable checkpoint access
  instructions or hashes plus a controlled evaluation endpoint; release
  prediction IDs/correctness vectors for all closed-book arms, exact launch
  configs, and a deterministic rerun recipe.  If weights cannot be shared,
  provide a reproducible public evaluation service or an independently audited
  archive.

**W5 — Minor: the content-interface result is informative but cannot identify
the source of the apparent improvement.**

* **Location / exact quote:** Sec. 3.1, lines 51--53; PDF p. 4: “simultaneously change prompt, candidate string, tokenization, and normalization”.
* **Problem:** The content-versus-letter comparison changes four evaluation
  dimensions at once.  The random content floor therefore demonstrates that
  this *recipe* differs, but cannot establish whether prompting, length
  normalization, answer text, or candidate-tokenization causes the effect.
* **Affected claim / norm:** This constrains C3 and the discussion of
  interface artifacts.  The manuscript states this limitation correctly, so
  this is not a contradiction; it is a limit on the diagnostic's explanatory
  value.
* **Sufficient remedy:** Add a factorial protocol experiment holding three
  factors fixed at a time, plus random/base nulls for every protocol.

## Questions That Could Change the Score

1. Can the authors release, or arrange independent access to, the five
   headline checkpoints and the closed-book prediction/correctness files?  This
   would materially raise reproducibility and confidence in the tables.
2. Is a matched 200k full32 continuation available or feasible?  If it shows a
   much smaller target deficit under identical resume/data conditions, it would
   substantially strengthen the structural interpretation.
3. Across three fresh seeds, how often does keep14 show the same joint
   pattern—falling in-domain PPL with a large, pre-specified MMLU and
   closed-book deficit—and what is the variance in the 128k-to-200k change?
4. What base-relative threshold would the authors regard as “recovery,” and
   do the conclusions remain under a small grid of thresholds chosen before
   inspecting held-out trajectories?

## Non-scoring Suggestions / Typos

* Avoid calling the content score a “floor” without consistently qualifying it
  as the random-init floor under this particular prompt/candidate/normalization
  recipe.
* Put the one-run, no-seed, and unavailable-long-full32 facts in the first
  paragraph of the abstract, rather than relying on later caveats.
* Report realized FLOPs, wall time, throughput, latency, and memory in future
  work.  Nominal token presentations are useful but do not establish an
  efficiency trade-off.
* Consider moving the strongest paired-statistics qualification beside the
  main-table MMLU cells, since readers may otherwise overread the CIs as
  run-level uncertainty.

## Score Rationale

**Soundness: 2.5 / 5.0.** The arithmetic, protocol disclosure, fixed-checkpoint
paired MMLU analysis, and restrained language are good.  However, the evidence
does not support a robust empirical conclusion beyond the single historical
path: no seed uncertainty, no matched long-horizon intact control, no
predefined recovery criterion, and incomplete independent verification are
central limitations.

**Excitement: 2.5 / 5.0.** The message that optimization likelihood should not
be substituted for target evaluation is useful, and the interface/null analysis
is a worthwhile reminder.  But the paper proposes no method and explicitly
positions its novelty as a particular control combination; closely related
recovery-trajectory and beyond-perplexity work already exists.

**Overall: 2.5 / 5.0.** Strict reject.  This is a careful and readable
measurement report whose scoped descriptive finding may be useful as a
workshop-style case study or, with stronger artifacts and fresh replicated
controls, a Findings paper.  In its present form it is below the reliability
threshold I would use for an ARR main/Finding acceptance.

**Confidence: 4.0 / 5.0.** High confidence in the evidence audit and in the
limitations because the manuscript itself is unusually explicit.  Lower than
5.0 because I cannot independently execute the unavailable historical training
or inspect non-released predictions/checkpoints.

**Reproducibility: 2.0 / 5.0.** Evaluation details are unusually extensive and
some MMLU per-item artifacts/checksums are described, but no historical seed,
lost loader position, unavailable compute records, absent weights, and missing
closed-book prediction files prevent faithful end-to-end reproduction.

## Limitations, Ethics, and Desk-Reject Checks

The paper contains a distinct Limitations section (PDF p. 8) and an Ethical
Considerations section.  It acknowledges the principal scientific limitations:
single runs, unmatched horizons/compute, interface confounding, no
contamination audit or out-of-domain PPL, incomplete compute records, and
imperfect reconstruction.  The ethics discussion is proportionate for a
measurement paper and appropriately notes inherited model/data risks and
energy uncertainty.

I found **no desk-reject issue** in the frozen rendered submission: it is
anonymous; uses review mode; has a visible Limitations section; fits the normal
eight-page main-paper convention before references/appendix; has no unresolved
references or obvious placeholders; and its figure/table text is readable.
The `\scriptsize` tables are dense but legible in the rendered PDF and do not
appear to be a page-limit evasion.  The only reproducibility concern is
scientific rather than formatting-based.

## Complete Citation Audit

I audited the 33 entries actually cited in `main.bbl` (the `.bib` also contains
17 uncited background entries, which are not rendered references).  I verified
that all 33 cited keys resolve to a rendered bibliography entry and inspected
their rendered title/author/year/venue strings.  A full external, primary-record
resolution of every entry was not completed within this review's time budget;
therefore entries not separately checked in the targeted novelty search are
marked **Unverifiable**, rather than being overstated as Verified or Not found.

| Key | Status | Citation-use assessment |
|---|---|---|
| `benchmarktargets` | Unverifiable | Rendered metadata present; use for benchmark sensitivity is plausible. |
| `linearpatch` | Unverifiable | Rendered metadata present; use for repair context is plausible. |
| `prunecomp` | Unverifiable | Rendered metadata present; use for compensation context is plausible. |
| `deng2025drpruning` | Unverifiable | Rendered metadata present; use for integrated pruning context is plausible. |
| `gromov2024unreasonable` | Unverifiable | Load-bearing loss/task-dissociation antecedent; rendered metadata present. |
| `answerorder` | Unverifiable | Rendered metadata present; use for answer-order sensitivity is plausible. |
| `paser` | Unverifiable | Rendered metadata present; use for recovery-data selection is plausible. |
| `hendrycks2021mmlu` | Unverifiable | Rendered benchmark metadata present. |
| `jaiswal2024truth` | Unverifiable | Rendered metadata present; beyond-PPL motivation is plausible. |
| `joshi2017triviaqa` | Unverifiable | Rendered dataset metadata present. |
| `shortenedllama` | Unverifiable | Load-bearing CPT/retraining antecedent; rendered metadata present. |
| `calibration2026` | Unverifiable | Rendered metadata present; likely contemporaneous calibration context. |
| `kwiatkowski2019natural` | Unverifiable | Rendered dataset metadata present. |
| `lu2024reassessing` | Unverifiable | Rendered metadata present; use for pruning-selection dependence is plausible. |
| `mallen2023popqa` | Unverifiable | Rendered dataset metadata present. |
| `fragileknowledge` | Unverifiable | Rendered metadata present; adjacent width-pruning evidence only. |
| `men2024shortgpt` | Unverifiable | Load-bearing method antecedent; rendered metadata present. |
| `muralidharan2024compact` | Unverifiable | Rendered metadata present; integrated pruning/distillation context. |
| `costcompression` | Unverifiable | Load-bearing compression/knowledge context; rendered metadata present. |
| `olmo2` | Unverifiable | Rendered base-model metadata present. |
| `decisioncollapse` | Unverifiable | Rendered metadata present; contemporaneous adjacent analysis. |
| `siddiqui2024deeper` | Unverifiable | Rendered metadata present; pruning-selection context. |
| `song2024sleb` | Unverifiable | Rendered metadata present; layer-selection context. |
| `minitron` | Unverifiable | Load-bearing structured-pruning antecedent; rendered metadata present. |
| `slimqwen` | Unverifiable | Rendered metadata present; correctly treated as concurrent by the paper. |
| `myanswerisc` | Unverifiable | Rendered metadata present; interface-sensitivity use is plausible. |
| `iterabre` | Unverifiable | Rendered metadata present; recovery-trajectory use is plausible. |
| `xia2024sheared` | Unverifiable | Rendered metadata present; integrated pruning context. |
| `beyondperplexity` | Unverifiable | Rendered metadata present; safety-evaluation context. |
| `qwen3` | Unverifiable | Rendered metadata present; cross-family model citation. |
| `yang2024laco` | Unverifiable | Rendered metadata present; layer-merging context. |
| `shortopd` | Unverifiable | Rendered metadata present; correctly treated as concurrent by the paper. |
| `blockpruner` | Unverifiable | Rendered metadata present; block-level pruning context. |

**Citation-match spot checks (8):**

1. Gromov et al. is used for loss/task dissociation after deep-layer removal,
   which matches the cited work's positioning.
2. Shortened LLaMA is used for CPT curves and retraining/initialization
   comparisons, matching its stated scope.
3. Minitron is used as a structured pruning/distillation/retraining antecedent,
   not as evidence for this paper's exact construction.
4. IteRABRe is used for iterative removal/recovery trajectories and weak MMLU
   recovery; this is an appropriate comparison target.
5. The Cost of Compression and Jaiswal et al. support the general motivation
   that aggregate LM metrics need not track knowledge-sensitive measures; they
   do not establish the focal OLMo result, and the paper does not say they do.
6. `myanswerisc`, `benchmarktargets`, and `answerorder` are appropriately used
   for the narrower point that multiple-choice outcomes depend on interface and
   evaluation details.
7. PopQA, TriviaQA, Natural Questions, and MMLU are cited as benchmark sources,
   consistent with their use.
8. The OLMo-2 technical report is appropriately used to identify the base
   model; it is not used to validate the authors' training data or results.

No actually cited entry was “Not found,” and no unresolved external lookup was
converted into “Not found.” The bibliography has no cited-key/missing-entry
mismatch.  The newer works are potentially volatile bibliographically, so their
treatment below as contemporaneous rather than settled novelty baselines is
important.

## Novelty Search Summary and Cutoff

I performed targeted searches for the closest depth-pruning/recovery and
interface-evaluation work, including the paper's newly cited concurrent work.
The closest prior/concurrent papers are:

1. **Gromov et al., “The Unreasonable Ineffectiveness of the Deeper Layers”**
   — closest for post-pruning recovery and loss/task dissociation.
2. **Shortened LLaMA** — closest for depth-pruning CPT trajectories and
   retraining/initialization comparisons.
3. **Minitron / IteRABRe** — closest for structured/iterative pruning with
   recovery trajectories and downstream evaluation.
4. **SlimQwen** — cited by the manuscript as concurrent work covering
   matched-token initialization and progressive recovery.
5. **ShortOPD** — cited by the manuscript as concurrent work relevant to
   recognition/generation behavior after pruning.

The manuscript correctly calls SlimQwen and ShortOPD concurrent under its
three-month rule.  The bibliography also includes `calibration2026`, which
should be treated cautiously as a contemporaneous qualification rather than a
novelty baseline if its public date lies in that window.  The novelty that
remains is a carefully qualified **combination** of measurements and controls
in one OLMo case study, not the general observation that PPL and downstream
behavior can dissociate, trajectory analysis, depth pruning, or
multi-interface evaluation.

## Review-Process Self-Check

* I reviewed the frozen rendered 17-page PDF and frozen v6 source twice,
  including every appendix table and figure.  I inspected all rendered figures
  and tables, the main/appendix page break, and the bibliography.
* I checked abstract numbers against Tables 2--3 and Appendix Tables 7, 10,
  and 12; I found no material numerical mismatch.
* I audited formulas (PPL aggregation, above-chance recovery, MMLU protocols),
  boundary conditions, baselines, metrics, seed/statistics claims, scope,
  compute disclosure, and described artifact contents.
* I mechanically re-grepped each quoted weakness against the frozen source.
  Every W1--W5 includes location, a source-verbatim quote of fewer than 25
  words, an explicit problem, affected claim/norm, sufficient remedy, and a
  Major/Minor severity.
* I found no prompt injection or reviewer-manipulation text in the source or
  rendered PDF.  I did not treat an unavailable historical datum as evidence
  that it does not exist; where evidence is absent, I describe it as
  unavailable/unverifiable rather than fabricated.
