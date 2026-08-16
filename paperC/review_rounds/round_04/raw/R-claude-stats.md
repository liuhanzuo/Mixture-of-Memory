```json
{
  "reviewer_id": "R-claude-stats",
  "round": 4,
  "snapshot_sha256": "7fcb9ccc55c5c1d6ad1de868215b1d9253d0695a2f996340226aa6a6fa10db5a",
  "role": "experimental_rigor",
  "overall_score": 5.5,
  "confidence": 4,
  "recommendation": "borderline",
  "dimension_scores": {
    "novelty": 4,
    "significance": 4,
    "technical_soundness": 3,
    "experimental_rigor": 3,
    "clarity": 4,
    "reproducibility": 2,
    "citation_integrity": 4,
    "limitations_responsible_claims": 4
  },
  "paper_summary": "A measurement-methodology paper arguing that a multiple-choice score should be tested against the construct's best constant, input-blind predictor (a 'floor') before it is compared with another arm, and that 'above chance' is the wrong gate. It contributes (i) an arm-independent best-constant letter floor plus a longest-option content floor, and a demonstration that the content floor is under-specified until tie convention, length unit and tokenizer are fixed; (ii) a calibration of the floor estimator itself against a balanced null, since max over k noisy marginals is upward biased; (iii) an arm-conditional companion null (permutation of the arm's prediction vector within n_opt strata) whose statistic is the numerator of Cohen's kappa, correctly disclaimed as not new, used as a pre-comparison gate; and (iv) an empirical study over 4 base-model families, structurally damaged arms, MMLU-Pro (n=12032) plus five smaller benchmarks, showing that damaged arms read above chance while 14/15 sit at or below their floor. It also self-corrects its own headline (holding both references to one bootstrap standard shrinks the flip from 10/12-vs-1/12 to 3/12-vs-1/12), reports that BH/Bonferroni leaves 0/12 on both sides, ships a ledger of six retracted and two prohibited numeric claims, and shows that full-fp32 evaluation removes all bf16 ties and changes 18.03% of argmaxes without recovering accuracy.",
  "strongest_verified_contribution": "The arm-conditional stratified permutation null (read-out v2) and its constant-collapse identity, which I verified by writing an independent implementation from the paper's equations alone: on a synthetic 12032-item set with n_opt in {3..10} and eight strata, Delta_perm = 0 to <=1.4e-15 for all ten pure constant emitters, including letters illegal on most items. The paper's claim is therefore true and in fact stronger than stated, and it genuinely repairs a defect the paper honestly documents in its own v1 floor (which scores three equally empty emitters at 0.000, -2.111 and -3.807 pp). Second, and independent of any evidence file: the v1/v2 ordering disclosure in Appendix app:ordering is exactly self-consistent arithmetically (1403/12032 = 0.116606; 1439-1403 = 36 items; 36/12032 = 0.2992 pp), and the paper volunteers that its own ordering is not a theorem, names the violating n_opt-conditional emitter, bounds the tightest cell's attainable violation at 0.085 pp, and discounts its own 27-cell evidence count as pseudo-replication. Third, the section 5.1 self-correction, in which the authors notice their headline was asymmetric (CI on the floor side, bare point comparison on the chance side), apply one standard to both, and report that their effect shrinks to about a third of its advertised size, with the 3/12 and 1/12 counts reconstructing exactly from tab_mmlupro.",
  "strengths": [
    "Section 5.1's symmetric-standard self-correction is exemplary. The authors discovered that their own flip was measured with a CI on one side and a bare point comparison on the other, applied the paired bootstrap to both, and reported that 10/12-vs-1/12 becomes 3/12-vs-1/12 -- 'about a third of the size the asymmetric point-estimate comparison suggests'. I recomputed both floor-side counts from tab_mmlupro and they match exactly (point-above = 3/12: Llama-2 k14 +0.017, Qwen3 k14 +0.233, Qwen3 k10 +0.158; CI-excludes-0 = 1/12: Qwen3 k14 only).",
    "The paper reports that BH at q=0.05 and Bonferroni leave 0/12 cells on BOTH references, i.e. that its own corrected comparison is 'undefined rather than 3/12 versus 1/12'. Volunteering that a headline dissolves under correction is rare and correct.",
    "The 'what survives correction' figure is arithmetically right and is a criterion that could have failed: exact binomial P(X>=3 | n=12, p=1/20) = 0.019568 (paper: 0.0196), while P(X>=2) = 0.1184 -- so 2/12 would not have cleared the bar, and the floor side's own 1/12 gives P(X>=1) = 0.4596 and does not clear it.",
    "The winner's-curse calibration of the floor estimator is a genuine methodological contribution and it reproduces. Re-running the recipe stated in the tab_construct_nulls caption (balanced multinomial at each construct's (n,k), seed 20260814, 200,000 draws, numpy 2.5.1) matched the paper's E[f_hat] to ~1e-5 and q95 exactly on all nine rows, and matched p to ~2e-3. Recognising that a max over k noisy marginals is upward biased, and then narrowing the paper's own claim from eight constructs to three because five fall inside that noise, is the behaviour of authors who ran the check they did not need to run.",
    "Internal arithmetic of the two per-cell tables is clean under independent recomputation. All 27/27 rows of tab_v2_full satisfy Delta_perm = 100*(acc - acc_hat) to <=5e-4 pp; all 11/11 numeric rows of tab_mmlupro satisfy Delta = 100*(acc - 0.116606); and the |Delta| > half-width <=> p < 0.05 <=> verdict-label implication holds in 11/11 rows. Reconstructing a normal-approximation p from each v2 row's own Delta and half-width agrees with the printed bootstrap p within MC noise on all 27.",
    "The power table tab_power is internally consistent (6/6 yes/no labels agree with half-width <= 1.389) and its half-widths are not mechanical 1/sqrt(n) rescalings: implied per-item sd ranges 21.5-73.0 pp, and Winogrande (hw 1.184 at n=1267) vs CommonsenseQA (hw 3.399 at n=1221) differ 2.9x at nearly equal n, which is what a genuinely paired statistic should do. Making a power disclosure mandatory before any null result is interpreted is better practice than most papers in this area.",
    "Prior-art attribution is unusually disciplined. Section 3.2 and Appendix app:priorart state what is NOT claimed (Bennett's S, Brennan-Prediger, Frary formula scoring, Brenner-Kliebsch, De Vries pooled kappa), and the paper identifies its own statistic as the numerator of Cohen's kappa -- Delta_perm = kappa*(1-p_e) -- rather than letting a reader think it is new. All 17 bib keys are cited and all cited keys resolve; the shipped build record independently reports 0 undefined references, 0 undefined citations, 0 LaTeX errors.",
    "The claim ledger (six retracted, two prohibited numeric claims, with the retracted numbers named so a reader who meets them elsewhere knows they were withdrawn) and the 'auditing the audit' section (a doubled-tail bootstrap that produced an illegal p=1.042; a tokenizer-mismatched sequence cap that silently left-truncated 10/15 cells; an OOM that the merge guard correctly refused) are honest disclosures that most submissions would suppress. The fp32/McNemar numbers are also mutually coherent: 2532/14042 = 18.033%, -0.0015*14042 = -21.1 items implies |b-c| ~ 21, and the printed McNemar p=0.5702 implies ~1.3-1.4k correctness-discordant pairs, a proper subset of the 2532 argmax changes -- exactly as it should be."
  ],
  "issues": [
    {
      "id": "S1",
      "severity": "major",
      "location": "sections/tab_nulls.tex row 'MMLU-Pro, item-avg.' (E[f_hat]=0.104460, Chance=0.110877); sections/tab_construct_nulls.tex rows 1-2 (p<1e-5); abstract 'only three of the eight letter constructs (MMLU-Pro, MMLU, BoolQ) have floors a balanced null could not produce ($p<10^{-5}$)'; 01_introduction.tex bullet 1; 10_reproducibility.tex 'any attempt to reproduce a floor-above-chance claim must reproduce that calibration'. Evidence id E-CAL / floor_winners_curse_calibration.json.",
      "dimension": "experimental_rigor",
      "description": "The balanced null used to calibrate the MMLU-Pro floor is outside the support of any legal MMLU-Pro label vector, and the resulting p<1e-5 is not attainable under any support-respecting balanced null. TWO INDEPENDENT DEMONSTRATIONS. (a) Self-inconsistency visible from the paper's own printed columns, no simulation needed: for any balanced label process f_hat = max_L m_hat_L >= m_hat_A, hence E[f_hat] >= E[m_hat_A]; the paper defines Chance = mean(1/n_opt) = 0.110877, which IS E[m_hat_A] under the legal balanced process since A is legal on every item (n_opt >= 3); yet the same row prints E[f_hat] = 0.104460 under 'an exactly balanced null at that construct's own (n,k)'. 0.104460 < 0.110877: the expectation of a maximum is printed smaller than the expectation of one of its arguments. Applying this test to all nine rows, 8/9 pass and the ONE violation is the MMLU-Pro item-average row. (b) Quantified: the null is multinomial(12032, [1/10]x10), i.e. it places mass 1/10 on letter J for all 12032 items including those where J is not a legal option -- contradicting the paper's own n_opt in {3..10} and its own stated principle (section 3.2, app:priorart) that 'a letter illegal for an item can never be credited to it'. I reproduced the paper's recipe exactly (E[f]=0.104456 vs printed 0.104457; q95=0.107048 identical; p=0) confirming my implementation, then ran the legal null (gold uniform over each item's OWN legal options) over three very different n_opt histograms consistent with BOTH published constraints (mean(1/n_opt)=0.110877 and n_5+n_6+n_7+n_8=623 from app:ordering), seed 20260814, 100,000 draws, numpy 2.5.1: p = 0.0679, 0.0925, 0.1197. The observed floor 0.116606 sits BELOW the q95 of every legal null (0.1169-0.1176). Distribution-free cross-check with no MC: sd(m_A) is in [0.002479, 0.002862] by bounding mean(p^2) via Jensen and via p <= 1/3, so z = (0.116606-0.110877)/sd is in [2.00, 2.31], one-sided p in [0.010, 0.023] for letter A ALONE, and the statistic is a maximum over >=3 exchangeable candidates which can only raise p. So even the most favourable analytic treatment cannot approach 1e-5; the five-order-of-magnitude gap is created by the illegal support, not by Monte Carlo noise. BLAST RADIUS: two of the three constructs the abstract says survive calibration are the two MMLU-Pro rows; MMLU (k=4) and BoolQ (k=2) have legal nulls, and I re-verified both reproduce and stay at p~0. So the honest count is 2 of 8, not 3 of 8, and MMLU-Pro is not one of them -- while MMLU-Pro is the paper's primary benchmark (n=12032, all 21 powered cells, the entire section 5.1 flip, the 14/15 headline) and the only construct where variable n_opt applies. The defect lands exactly on the construct the paper leans on hardest and is absent from those where it would not have mattered. WHAT IS NOT DAMAGED: the floor value 1403/12032 = 0.116606 is a descriptive property of the item set and is unaffected; 'chance is the wrong reference' is unaffected; every arm-vs-floor comparison is unaffected. What breaks is only the magnitude claim for MMLU-Pro, which after correction joins the five constructs the paper already describes as lying inside the estimator's own noise. The paper's existing scoping language already covers the corrected case, so the fix is arithmetic plus a sentence, not new measurement.",
      "proposed_fix": "Re-run the winner's-curse calibration for the two MMLU-Pro rows with the gold label drawn uniformly over each item's own legal option set rather than over all ten letters (the per-item n_opt vector already exists -- the v2 stratification consumes it). Report the corrected p (expect 0.05-0.15) and move MMLU-Pro into the 'inside estimator noise' partition. Restate the abstract, the introduction bullet, both null tables and the reproducibility statement as 'two of the eight letter constructs (MMLU, BoolQ)'. Add an assertion to emit_tab_construct_nulls.py that E[f_hat] >= Chance for every row, since that single check would have caught this by inspection; and state in the caption that the balanced null respects each item's legal label set.",
      "verification_test": "(1) Assert E[f_hat] >= Chance for all nine rows of the emitted table; the current MMLU-Pro item-average row must FAIL this assertion before the fix and pass after. (2) Assert that the null's per-item support equals the item's legal option set: for every simulated draw, count nonzero mass on letters with index >= n_opt_i and require 0. (3) Assert E[f_hat] under the corrected null equals mean(1/n_opt) + a positive selection term, and that its numeric value exceeds 0.110877. (4) Confirm the corrected p is in [0.02, 0.20] and that the row's verdict string changes from 'above balanced null' to 'inside estimator noise'. (5) Confirm MMLU and BoolQ p-values are byte-unchanged by the fix, since their n_opt is constant."
    },
    {
      "id": "S2",
      "severity": "major",
      "location": "sections/04_experiments.tex 'Designated damaged cells, and multiplicity' ('the conclusions rest on the aggregate's near-unanimity (0/60 damaged cells clear their floor), not on one cell'); sections/09a_relocated.tex app:designated ('The paper's conclusions rest on the direction and near-unanimity of the aggregate (for instance 0/60 damaged cells clearing their floor)'); sections/05_analysis.tex paragraph 'The most vivid small-benchmark cases...' ('0/60 damaged cells clear their floor while 25/60 read above chance, and only 7/60 are significantly below because 52/60 are underpowered').",
      "dimension": "experimental_rigor",
      "description": "The aggregate on which the paper explicitly says its conclusions rest is largely forced by low power, so it is close to a criterion that cannot fail. The paper states in the same breath that 52/60 of those cells are underpowered for a 1.389 pp effect. Using tab_power's own five small-benchmark half-widths (1.305, 2.775, 3.399, 3.882, 6.400), I computed how many of 60 cells would show 'CI excludes 0 above floor' if EVERY damaged arm truly were above its floor by delta: +0.5 pp -> 3.5/60 expected; +1.0 pp -> 7.6/60; +1.389 pp (the paper's own reference effect) -> 12.0/60; +3.0 pp -> 29.3/60. So even in a world where all 60 arms genuinely clear their floor by the reference effect, this design expects only about 12 detections; observing 0 is therefore consistent with a substantial positive true effect. This is a failure-to-reject presented as near-unanimity. It is not vacuous -- P(0/60 | delta=+1.0 pp) = 2e-4 and P(0/60 | delta=+0.5 pp) = 0.026, so large positive effects are excluded -- but 'the conclusions rest on near-unanimity' overstates by a wide margin what 0/60 licenses. This is the paper's own 'Power is part of construct validity' principle applied to its own headline aggregate, and the aggregate does not survive it. The paper demonstrates elsewhere that it knows how to do this correctly (it labels OLMo-2 keep8 on MMLU a 'powered non-replication'), so the machinery exists.",
      "proposed_fix": "Replace the 0/60 count with a pooled effect estimate and its interval, or restate it as the negative result it is: 'across the 60 damaged cells the pooled above-floor effect is bounded above by roughly +0.5 pp at 95%', accompanied by the expected-detection calculation showing what 0/60 could and could not have distinguished. Wherever 0/60 or 0/15 is invoked as evidential support, add the design's expected detection count under a stated alternative so the reader can see the count's resolving power. Symmetrically, 25/60 'read above chance' should be reported with the same standard, since section 5.1 established that a bare point comparison is not a test.",
      "verification_test": "Publish, for each of the 60 cells, the half-width and the implied power at delta = +1.389 pp, plus the expected number of detections summed over cells (should be about 12). Then show that the reported count 0/60 lies inside the null predictive distribution for any |delta| <= 0.5 pp, and report the pooled random-effects effect with its CI. A reader must be able to see that 0/60 excludes delta >= +1.0 pp and does not exclude delta = +0.3 pp."
    },
    {
      "id": "S3",
      "severity": "major",
      "location": "sections/05_analysis.tex, paragraph 'Two qualifications belong with this two-reference comparison' ('observing 3 or more rejections out of 12 has binomial probability $0.0196$ under the global null'); duplicated verbatim in sections/09a_relocated.tex app:analysis-mult. Contradicts sections/09a_relocated.tex app:designated ('the cells share items, nest arms, and share a null, so the tests are neither independent nor exchangeable').",
      "dimension": "technical_soundness",
      "description": "The 0.0196 figure -- the paper's only surviving positive inferential claim on the chance side after it concedes 0/12 under BH and Bonferroni -- is computed under an iid Bernoulli(0.05) model over exactly the 12 tests that the paper elsewhere states are 'neither independent nor exchangeable'. Dependence is invoked to DECLINE family-wise correction and then ignored to MANUFACTURE a global p-value from the same 12 decisions. The arithmetic is correct (I confirmed exact binomial P(X>=3 | 12, 1/20) = 0.019568) but the model is the one the paper rejects two paragraphs earlier. Under positive dependence the count's tail is heavier. Quantifying with an exchangeable beta-binomial at the same marginal p=0.05 (exact, via log-Beta): rho=0 gives 0.0196, rho=0.05 gives 0.0501, rho=0.10 gives 0.0678, rho=0.20 gives 0.0829, rho=0.30 gives 0.0860. Any intra-class correlation above about 0.05 pushes the 'surviving' evidence above 0.05. Given that the 12 cells are four model families' ladder rungs scored on the SAME 12032 items against the SAME null, rho ~ 0 is not defensible; the shared item axis alone induces positive dependence.",
      "proposed_fix": "Either (a) obtain a joint null by permuting the shared item axis -- resample items once and recompute all 12 cells inside each resample, giving the null distribution of the rejection count under the actual dependence, which the existing 10,000-resample paired-bootstrap machinery can do at no new measurement cost -- and report that p; or (b) demote the sentence to a descriptive observation ('3 of 12 cells reject at an uncorrected alpha=0.05') and drop the probability, since the paper has already honestly stated that the corrected comparison is undefined. Do not present an independence-based global p while arguing that independence fails.",
      "verification_test": "Report the intra-class correlation of the 12 per-cell rejection indicators implied by the shared-item bootstrap, and the resulting P(X>=3). Show that the item-permutation joint null gives a p-value in the same paragraph as 0.0196, and that the two differ. Sensitivity table over rho in {0, 0.05, 0.1, 0.2} would suffice to show the claim's fragility."
    },
    {
      "id": "S4",
      "severity": "major",
      "location": "sections/05_analysis.tex, 'V2 confirms a small alignment effect, $+0.267$ points at $p=0.0066$, but assigns \\texttt{recovery\\_fraction}=0.049, only 9.1\\% of the intact-family anchor and below the 10\\% materiality bar. It is a real but immaterial exception'; the materiality rule in sections/03_method.tex ('require at least 0.10 times the same-family intact anchor'); half-width 0.188 in sections/tab_v2_full.tex row 'Qwen3 k14'.",
      "dimension": "experimental_rigor",
      "description": "The gate that decides the paper's single most-discussed cell -- Qwen3 k14, the lone above-floor exception, the withdrawn capability label, the '14/15 rather than universal' qualification -- is a point estimate with no uncertainty attached anywhere in the paper, and its interval straddles its own threshold. I verified the point value is internally coherent: from the printed 94.6% modal-A share, 11382 A-predictions and 650 others give best achievable alignment (1403+650)/12032 = 0.1706, so Delta_max = 5.437 pp and recovery = 0.267/5.437 = 0.0491, matching the printed 0.049. But propagating the table's own half-width (0.188 pp on Delta_perm = 0.267) gives: CI lower Delta=0.079 -> recovery 0.0145 -> 2.7% of anchor (immaterial); point 0.267 -> 9.1% (immaterial); CI upper 0.455 -> recovery 0.0837 -> 15.5% of anchor (MATERIAL). The 95% interval on the materiality ratio is [2.7%, 15.5%] and CONTAINS 10%. So 'a real but immaterial exception, not evidence that k14 retains MMLU-Pro competence' is a decision the data cannot make at the paper's own confidence level. recovery_fraction is the one statistic in the paper that never receives a CI, and it is the one that adjudicates the headline exception. Compounding this, the 0.10 constant is asserted without derivation and fires exactly twice -- Qwen3 k14 at 0.91x the bar and Llama-2 intact at 0.545x ('anchor blocked') -- so in 2/2 applications the choice of constant rather than the data is load-bearing, in a re-analysis the paper itself labels 'post-hoc by construction'.",
      "proposed_fix": "Bootstrap recovery_fraction directly: Delta_perm and Delta_max are both functions of the same resampled items, so recompute both inside each of the existing 10,000 paired resamples and report the CI on the ratio to the anchor. If it straddles 0.10, report the gate as indecisive for that cell rather than reporting 'immaterial' as a finding -- the honest statement is 'the exception is real (p=0.0066) and the materiality gate cannot resolve it'. Separately, report the sensitivity of both gate firings to the constant over 0.05-0.20, or give a basis for 0.10.",
      "verification_test": "Emit a bootstrap CI for recovery_fraction and for its ratio to the same-family intact anchor on all 27 cells; confirm that Qwen3 k14's ratio CI contains 0.10 and that Llama-2 intact's contains or excludes it. Then confirm whether any verdict string in tab_v2_resort changes when the constant is set to 0.05 or 0.20; if verdicts change, the sensitivity table must be printed."
    },
    {
      "id": "S5",
      "severity": "major",
      "location": "sections/05_analysis.tex section 'V2 re-sorts the 27-cell read-out' ('V2 is not uniformly conservative... It withdraws the published above-floor capability label for \\texttt{qwen3/k14}... while \\texttt{olmo2/keep14} moves from at-floor to \\texttt{TRACE\\_SIGNAL}. Re-sorting in both directions is evidence that the criterion is doing more than shrinking effects'); p-values in sections/tab_v2_full.tex.",
      "dimension": "experimental_rigor",
      "description": "The 27-cell v2 re-analysis receives no multiplicity correction, and neither 'trace signal' cell survives one over the paper's own tabulated family. The paper applies BH and Bonferroni to the 12-cell chance-vs-floor comparison (creditably reporting 0/12 survive) but not to the 27-cell re-analysis where 7 cells read p<0.05. Running BH at q=0.05 over exactly the 27 p-values printed in tab_v2_full: ranks 1-5 (p=0.0001; Llama-2, Llama-3, OLMo-2 and Qwen3 intact, plus OLMo-2 shortgpt16) reject; rank 6 Qwen3 k14 (p=0.0066, threshold 0.01111) rejects; rank 7 OLMo-2 keep14 (p=0.0172, threshold 0.01296) is RETAINED. Bonferroni (alpha/27 = 0.00185) rejects only the 5 anchors, so both trace-signal cells fail. This matters because OLMo-2 keep14 is the ONLY cell that moves in the upward direction, and it is the one BH drops. The bidirectionality claim -- offered as evidence that v2 'is doing more than shrinking effects' -- therefore rests on a single uncorrected borderline cell inside a 27-test family the paper itself constructed and tabulated, and the 5 rejections that do survive are all intact or shortgpt16 anchors where the answer was never in doubt. The paper's general defence of not correcting (no defensible family definition) is weaker here than for the near-unanimous aggregates, because tab_v2_full IS the family: it is one table, one null, one statistic, one item set.",
      "proposed_fix": "Add a BH-adjusted q-value column to tab_v2_full, and restate the section as: 'one cell moves upward (OLMo-2 keep14, p=0.0172 uncorrected, does not survive BH over the 27-cell family)'. Either drop the bidirectionality claim or support it with a cell that survives correction. If the authors prefer to keep uncorrected per-cell decisions, say plainly that the bidirectionality evidence is a single uncorrected cell.",
      "verification_test": "Print BH q-values for all 27 cells and confirm that exactly 6 reject at q=0.05 and 5 at Bonferroni, and that OLMo-2 keep14 is not among them. Then check whether any sentence in section 5.3 or the abstract ('re-sorts', 'not uniformly conservative') still has support; the abstract's 'withdraws one above-floor capability label, dissolves both below-floor competence labels' is unaffected and should be kept."
    },
    {
      "id": "S6",
      "severity": "major",
      "location": "The snapshot as a whole: review_rounds/round_04/submission/evidence/ contains only build_record.json and claim_evidence_map.tsv. Every quantitative evidence id used in a caption -- E-A, E-B, E-D, E-E, E-F, E-H, E-I, E-K, E-CAL -- resolves via sections/09_appendix.tex Table tab:artifact-map to a path outside the snapshot (evidence/floor_winners_curse_calibration.json, evidence/heal_readout_v2_permutation_null.json, evidence/construct_nulls_length_unit.json, evidence/second_mc_benchmark/, evidence/mmlu_scale_power/, tcodex_out/EVIDENCE_PACK.md). Also affects sections/10_reproducibility.tex ('at the time of writing it reports 610 numerals with none unresolved').",
      "dimension": "reproducibility",
      "description": "The paper's central reproducibility promise is that 'every quantitative claim in this paper is bound to a machine-readable record, and the binding is checked mechanically rather than by hand', but the shipped artifact contains no such record. The two files present are a LaTeX build log and a claim-status ledger; between them they contain zero per-item records, zero bootstrap draws and zero p-values. Consequently I could verify internal arithmetic and the behaviour of the stated nulls -- which is how I found S1 -- but could not check a single number against its source. Two specific and load-bearing gaps: (a) the per-cell CHANCE-side deltas for the 12 MMLU-Pro cells are printed nowhere in the manuscript, so the paper's most-quoted result (10/12 point-above and 3/12 CI-above chance) cannot be reconstructed even in principle from the submission, whereas I could reconstruct both floor-side counts exactly from tab_mmlupro; (b) the '610 numerals with none unresolved' claim cannot be evaluated, and I found at least one numeral that a strict checker should have caught -- the tab_nulls caption says the inside-noise gaps span '+0.43--+2.60 pp' while the smallest such gap in either table is PIQA's +0.490. Separately, build_record.json states 'pdf_visually_inspected: false' with the note that the integrity item 'remains OPEN', and the recorded pdf_sha256/pdf_bytes (56a376e1.../355196) do not match the manifest's entry for the shipped main.pdf (1fbaaf99.../366583), so the shipped PDF is not the artifact the build record describes.",
      "proposed_fix": "Ship the evidence files the captions cite, at minimum floor_winners_curse_calibration.json (E-CAL, which S1 turns on), heal_readout_v2_permutation_null.json (E-D), s2_03_symmetric_inference.json and the per-cell mmlu_scale_power records. Add the per-cell chance-side delta, half-width and p to tab_mmlupro as three columns, so the 10/12-vs-3/12 comparison is reconstructible from the paper alone. Regenerate build_record.json against the submitted PDF so the hashes agree, and either complete or explicitly flag the visual inspection.",
      "verification_test": "Confirm that for every evidence id appearing in a caption there is a corresponding file in the artifact, and that a fresh checkout can recompute at least the nine construct-null rows, the 27 v2 rows and the 12 chance-side deltas from shipped records without network access. Confirm build_record.json's pdf_sha256 matches sha256 of the submitted main.pdf. Re-run the numeral checker on a clean tree and publish its report alongside the paper, including the tolerance rule, so '610 numerals, none unresolved' is auditable and the +0.43 discrepancy is either resolved or explained."
    },
    {
      "id": "S7",
      "severity": "minor",
      "location": "sections/tab_nulls.tex and sections/tab_construct_nulls.tex, all rows: the p column is defined as $\\Pr(\\hat f\\ge\\text{observed floor})$ with the floor stored to 6 decimal places.",
      "dimension": "technical_soundness",
      "description": "A stored-precision effect on a discrete tail. The estimator f_hat = max_L count_L / n has atoms spaced 1/n, and the tail is evaluated as '>=' against a 6-dp rounded decimal. Exact rational recomputation of n*floor shows the rounded threshold does not consistently satisfy '>=' against the realised outcome: in 4 of 9 rows the stored decimal sits strictly ABOVE the realised atom and silently drops it from the tail (MMLU-Pro 1403.0034 -> stored 0.116606 > 1403/12032; MMLU 3776.0061; PIQA 928.0007; BoolQ 2033.0015), while in 3 rows it sits below (ARC-Easy 632.9997, ARC-Challenge 310.9996, CommonsenseQA 254.9997) and in 1 it is exact (OpenBookQA 138.0000). The direction is anti-conservative for MMLU-Pro/MMLU/PIQA/BoolQ and conservative for the others. I verified the effect is measurable: for PIQA the correct integer-count tail is 0.690 while the paper prints 0.658, a 3.2 pp difference caused entirely by the excluded atom. No verdict changes in any row, so this is reporting hygiene rather than a wrong conclusion, but it is exactly the class of defect the paper's own methodological thesis is about.",
      "proposed_fix": "Compare in integer counts, not rounded rates: evaluate p = Pr(max_L count_L >= ceil(n * floor - 0.5)) using the stored integer label count, and store the count alongside the rate in the evidence file. Recompute the nine p-values; expect PIQA to move from 0.658 to about 0.690 and the others to move negligibly.",
      "verification_test": "Assert for every row that n * floor rounds to an integer within 1e-6 and that the p-value is computed from that integer. Confirm the recomputed p differs from the published value only for PIQA (0.658 -> about 0.690) and that all nine verdict strings are unchanged."
    },
    {
      "id": "S8",
      "severity": "minor",
      "location": "sections/tab_power.tex, column 'Detect $-1.389$ pp?' (ARC-Easy 'yes, borderline' at half-width 1.305; Winogrande 'yes' at 1.184; MMLU 'reference' at 1.154).",
      "dimension": "experimental_rigor",
      "description": "The word 'detect' is used for a 50%-power rule, so two rows labelled affirmatively are near coin-flips. The rule the table implements is half-width <= |d|, which means 'a point estimate of exactly -1.389 pp would have its CI exclude 0' -- i.e. 50% power at that alternative. Computing actual power as Phi(d/se - 1.96) from each printed half-width: ARC-Easy 'yes, borderline' is 55.0%, Winogrande 'yes' is 63.3%, and the MMLU reference row itself is 65.5%. The 'no' rows are 6-16% and are correctly and conservatively labelled. IMPORTANT SELF-CORRECTION: I initially suspected this also undermined the abstract's 'all 21 evaluated cells are powered at the scale of the reference effect' and section 4's matching sentence. It does not. The prose gives the MMLU-Pro half-width range as 0.083-0.968, and the WORST of those has 80.3% power at d = -1.389 pp, so the 21-cell claim holds even at the conventional 80% bar. The issue is confined to tab_power's small-benchmark labels, and it cuts against the paper's own null results rather than for them: if anything the '52/60 underpowered' count is understated.",
      "proposed_fix": "Print the achieved power per row rather than a yes/no, or state the level in the column header ('half-width below the reference effect, i.e. 50% power at d = -1.389 pp'). Optionally add an 80%-power column (threshold half-width 0.972 pp), which no row of tab_power meets while all 21 MMLU-Pro cells do -- a contrast that strengthens the paper's argument for MMLU-scale evidence.",
      "verification_test": "Add a power column computed as Phi(|d|/se - 1.96) with se = half-width/1.96 and confirm ARC-Easy 0.550, Winogrande 0.633, MMLU 0.655, PIQA 0.164, CommonsenseQA 0.123, ARC-Challenge 0.104, OpenBookQA 0.062. Separately confirm the MMLU-Pro worst-case cell (half-width 0.968) has power >= 0.80 so the 21-cell claim can be stated at the 80% bar."
    },
    {
      "id": "S9",
      "severity": "minor",
      "location": "Cross-table numeric hygiene. (a) CommonsenseQA gap printed as +0.884 in sections/tab_construct_nulls.tex and +0.885 in sections/tab_nulls.tex (true value 0.8845). (b) sections/tab_nulls.tex rows 1-2 print different E[f_hat] (0.104457 vs 0.104460) and q95 (0.107048 vs 0.107131) although the caption says both use the same balanced null at (n,k)=(12032,10). (c) sections/tab_nulls.tex caption says the inside-noise gaps are '+0.43--+2.60 pp' but the smallest is PIQA's +0.490. (d) ARC-Challenge's letter floor (tab_nulls, L*=B, 0.265358) equals its content token-OLMo-2 'first' value (tab_conventions, 0.265358) to all six printed digits. (e) sections/00_abstract.tex and 01_introduction.tex claim 'ten target constructs' but section 4 lists 8 letter plus 7 content constructs.",
      "dimension": "clarity",
      "description": "A cluster of small inconsistencies that matter more than usual because the paper's central promise is mechanically checked numeral binding. (a) is a truncate-vs-round split on the same quantity in two tables of the same paper. (b) implies rows 1 and 2 are two independent Monte Carlo runs of the identical null, with 8.3e-5 of MC jitter presented as if it were two distinct quantities; my single run gives one value for both. (c) is a caption range with no supporting row. (d) may be coincidence -- both are plausibly 311/1172 at small n -- but it is the signature of a cross-table copy-paste, and the ARC-Challenge content row is load-bearing for the under-specification argument, so it should be confirmed rather than left ambiguous. (e) means the headline inventory cannot be reconstructed from the setup paragraph; the letter side is internally consistent and does check out (3 significant + 5 inside noise = 8 distinct letter constructs, i.e. the 9 table rows minus the duplicated MMLU-Pro row).",
      "proposed_fix": "Emit both null tables from the same record with one rounding rule; use a single Monte Carlo run for the shared (n,k) so rows 1-2 agree exactly; correct the caption range to +0.49--+2.60; confirm and annotate the ARC-Challenge coincidence (or fix the transcription); and print the ten target constructs as an explicit enumerated list.",
      "verification_test": "Assert that any quantity appearing in more than one table is byte-identical across tables; assert that rows sharing (n,k) share their E[f_hat] and q95; assert that every min/max stated in a caption equals the min/max of the rows it describes; and check the ten-construct list against the section 4 enumeration programmatically."
    },
    {
      "id": "S10",
      "severity": "minor",
      "location": "sections/03b_nulls_summary.tex ('a 40.6-point span whose upper end exceeds the intact base model's own \\texttt{content\\_norm} score by 32.5 points'; 'ARC-Challenge 18.60\\% tied-longest under characters against 50.85\\% under tokens'; 'moves \\texttt{credit} by up to 10.6 points on four-way tasks and 9.26 on MMLU-Pro').",
      "dimension": "reproducibility",
      "description": "Load-bearing numbers in the four-under-specifications summary have no anchor anywhere in the manuscript. The 32.5-point claim implies an intact content_norm of about 0.2072 that appears in no table; the ARC-Challenge 18.60%/50.85% tied-longest fractions have no table column; the tokenizer-shift figures 10.6 and 9.26 have no supporting table. Three of my five spot-checks in this passage did tie out against tab_conventions (MMLU-Pro credit 0.532164 and wrong 0.125914, span 0.406250 -> '40.6-point span'; OpenBookQA 0.416/0.644), so the passage is not fabricated -- but since section 4 was compressed into this summary for the page budget, the numbers that lost their table are exactly the ones a reviewer cannot check.",
      "proposed_fix": "Add the intact content_norm value and the tied-longest fractions as columns in tab_conventions (which already has room in the appendix), or state them with an explicit evidence id and value in the caption. A tokenizer-shift row would anchor the 10.6 and 9.26 figures.",
      "verification_test": "Every numeral in 03b_nulls_summary.tex should resolve to a printed table cell or a shipped evidence field; run the numeral checker restricted to that file and require zero unresolved, then confirm a reader can locate each of 32.5, 18.60, 50.85, 10.6 and 9.26 in a table."
    }
  ],
  "score_ceiling_under_current_evidence": 6.5,
  "predicted_score_after_required_changes": 6.5,
  "evidence_that_would_raise_score": [
    "Re-run the winner's-curse calibration for MMLU-Pro with the gold label drawn uniformly over each item's own legal option set, and restate the abstract, introduction and both null tables accordingly. This requires no new measurement -- the per-item n_opt vector already exists because the v2 stratification consumes it -- and it converts the paper's most serious defect into a demonstration of the paper's own thesis, which would be a genuinely strong outcome.",
    "Ship the cited evidence files, above all floor_winners_curse_calibration.json (E-CAL), heal_readout_v2_permutation_null.json (E-D) and the per-cell mmlu_scale_power records, so that the nine construct-null rows, the 27 v2 rows and the 12 chance-side deltas can be recomputed from the artifact. This alone would move reproducibility from 2 to 4.",
    "Add per-cell chance-side delta, half-width and p columns to tab_mmlupro so the 10/12-versus-3/12 comparison is reconstructible from the paper. I could reconstruct the floor side exactly and not the chance side, on the paper's single most-quoted result.",
    "Report a bootstrap CI on recovery_fraction and on its ratio to the intact anchor for all 27 cells, and restate the Qwen3 k14 verdict as indecisive if the interval straddles 0.10 (my propagation gives [2.7%, 15.5%]). Adding a sensitivity table for the 0.10 constant over 0.05-0.20 would close S4 entirely.",
    "Replace the 0/60 and 0/15 counts with a pooled effect and interval, or accompany each with the design's expected detection count under a stated alternative (about 12/60 at d = +1.389 pp). Stating the aggregate as 'excludes a mean above-floor effect larger than about +0.5 pp' is both defensible and stronger-sounding than a count that cannot fail.",
    "Obtain a joint null for the rejection-count claim by permuting the shared item axis inside the existing bootstrap, replacing the independence-based 0.0196; or demote that sentence to descriptive. Add BH q-values to tab_v2_full and restate the bidirectionality claim in light of OLMo-2 keep14 failing BH.",
    "Regenerate build_record.json against the submitted PDF (the recorded pdf_sha256 and pdf_bytes do not match the manifest entry for the shipped main.pdf) and either complete the visual inspection or keep it flagged as open."
  ],
  "evidence_that_would_lower_score": [
    "If the shipped E-CAL record shows the MMLU-Pro balanced null was generated over all ten letters uniformly for all 12032 items -- which is what E[f_hat]=0.104457 and q95=0.107048 imply and what my reproduction confirms -- then the p<1e-5 claim was not merely mis-described but mis-computed at source, and the same generator was used for the emitter that the appendix advertises as refusing to write on any inconsistency. That would make S1 fatal for the calibration contribution rather than major.",
    "If the per-cell chance-side deltas, once shipped, do not reproduce 10/12 point-above and 3/12 CI-above, the section 5.1 self-correction that is currently the paper's strongest moment would itself need re-adjudication.",
    "If the ARC-Challenge letter floor and content token-'first' value coinciding at 0.265358 turns out to be a cross-table transcription rather than an arithmetic coincidence, the four-under-specifications argument loses one of its two named non-MMLU anchors and the numeral-checker claim becomes untenable.",
    "If recovery_fraction's CI is reported and its ratio to the anchor excludes 10% from ABOVE (i.e. Qwen3 k14 is material), the 14/15 headline and the 'real but immaterial exception' framing would both need revision, and the abstract's 'withdraws one above-floor capability label' would reverse.",
    "If any load-bearing 2026 citation (oostermeijer2026length ICML 2026, cho2026choices ICLR 2026 Poster, arcon2026metalinguistic arXiv 2602.02182) does not support the sentence attached to it, citation integrity would fall: the first two carry the paper's 'we do not claim generic length sensitivity / the interface contrast' disclaimers, and the third is the paper's opening motivation. I had no network access and could not check these.",
    "If the numeral checker's '610 numerals with none unresolved' claim turns out to rely on tolerances loose enough to admit the +0.43-versus-+0.490 caption range and the 0.884-versus-0.885 split, the paper's central reproducibility promise would be weaker than advertised."
  ],
  "review_limitations": [
    "NO SHIPPED QUANTITATIVE EVIDENCE. The snapshot contains only build_record.json and claim_evidence_map.tsv. Every number in this review was checked either for internal consistency across the manuscript's own tables, or by re-implementing the stated procedure from the equations and captions. NOT ONE number was checked against its raw source. Any of my 'verified' items could still disagree with the underlying per-item records.",
    "I could not reconstruct the chance-side column of the two-reference comparison (10/12 point-above, 3/12 CI-above), because the per-cell chance deltas appear nowhere in the manuscript. I verified only the floor side (3/12 and 1/12), which matched exactly. The headline 3/12-versus-1/12 is therefore half-verified.",
    "MY LEGAL-NULL P-VALUES FOR MMLU-PRO (0.068-0.120) DEPEND ON AN n_opt HISTOGRAM I HAD TO RECONSTRUCT. The paper does not print the per-stratum item counts. I pinned the histogram with two published constraints (mean(1/n_opt)=0.110877 and n_5+n_6+n_7+n_8=623) and swept three very different completions; all three gave p in [0.068, 0.120]. The exact corrected p depends on the true histogram and could fall outside that range, though the distribution-free bound (one-sided p >= 0.010 for letter A alone, and the statistic is a max over >=3 candidates) is histogram-independent and already rules out 1e-5. The self-inconsistency argument (E[f_hat] = 0.104460 < Chance = 0.110877 in one row) needs no histogram at all and is the form I rely on.",
    "Environment: /opt/conda/envs/torch-base/bin/python, numpy 2.5.1. scipy is NOT installed on this node, so all tail probabilities were computed either exactly with fractions.Fraction and math.lgamma or by Monte Carlo with numpy Generator(PCG64). This cluster runs three numpy versions and same-seed draws are not bit-identical across nodes; my agreement with the paper's calibration (E[f_hat] to ~1e-5, q95 exact, p to ~2e-3) is agreement, not exact cross-machine reproduction, and I do not claim bit-identity.",
    "I did NOT verify a single citation's venue, existence, or claim-level support. No network access was used. Citation integrity was scored on internal evidence only: key/citation cross-check (17/17 both ways), the build record's 0 undefined citations, and the unusually explicit not-claimed attribution in section 3.2 and app:priorart. The three 2026 references (oostermeijer2026length, cho2026choices, arcon2026metalinguistic) are load-bearing and unconfirmed.",
    "I did NOT run the authors' checkers (check_prose_vs_evidence.py, validate_tex_static.py, emit_tab_construct_nulls.py) because they write into paperC/evidence/ and I was read-only. So the '610 numerals with none unresolved' claim, the emitter's advertised guards, and the '27 cells verified against the kappa identity to printing precision' claim are all UNVERIFIED. I did establish that no existing guard can test E[f_hat] >= Chance, since that row ships violating it.",
    "The '13 of the 27 clear the floor by more than 0.299 pp' claim in app:ordering is unverifiable from the manuscript: tab_v2_full prints Delta_perm and accuracy but not the per-cell v1 floor margin, and 14 of the 27 cells do not appear in tab_mmlupro either. Likewise the 'tightest cell attainable violation 0.085 pp' and '94.8% of its weight lies in A-argmax strata' figures, and the 'sixteen damaged cross-family cells have accuracy equal to the marginal of the emitted letter to machine precision' claim.",
    "I did NOT check the OpenBookQA character-unit reconstruction, the 12/12 character/token self-test, the healing-corpus token accounting (5.72 epochs of 5.541B SlimPajama vs 1.0 epoch of 31.7B Dolmino), the layer-fraction arithmetic beyond noting 8/36 = 22.2% and 8/32 = 25.0% are correct, or the section 5.2 ARC-Easy content-versus-letter gap (+38.76 pp, McNemar p=9.8e-148) and the residual-fraction multipliers (2.11x, 0.90x, 0.98x).",
    "I did not open main.pdf, so I checked no rendering, figure, page-count or layout property, and I cannot confirm the manuscript fits the ICLR 2026 9-page main-text limit. I note only that build_record.json reports 22 pages, 0 overfull boxes, and pdf_visually_inspected: false, and that its recorded pdf_sha256/pdf_bytes disagree with the manifest entry for the shipped PDF.",
    "MY OWN FIRST READING WAS WRONG ONCE AND I CORRECTED IT. I initially judged that the 'all 21 cells are powered at the scale of the reference effect' claim was a 50%-power overstatement. Computing power from the prose half-width range 0.083-0.968 shows the worst cell reaches 80.3%, so the claim survives the conventional bar and I withdrew that part of the finding, narrowing it to tab_power's small-benchmark labels (S8, minor). Reviewers of this review should weight S8 accordingly.",
    "Scope: I scored all eight rubric dimensions but my lens is experiments and statistics. Novelty and significance were judged from the paper's own positioning and its related-work section without an independent literature search, so those two dimension scores carry materially less weight than technical_soundness, experimental_rigor and reproducibility."
  ]
}
```

## PROSE REVIEW

### Summary and overall assessment

This paper argues that a multiple-choice score should be tested against the construct's best constant, input-blind predictor before it is compared with another arm, and that the conventional "above chance" comparison is not sufficient. The argument is correct, useful, and under-appreciated, and the paper prosecutes it with more statistical self-awareness than most submissions in evaluation methodology. It calibrates its own floor estimator for the winner's curse; it discovers that its own headline was measured asymmetrically and repairs it at the cost of two thirds of its effect size; it reports that BH and Bonferroni leave nothing standing on either side of that comparison; it ships a ledger of six retracted and two prohibited numeric claims; and it volunteers that its own v1/v2 null ordering is not a theorem, names the construction that violates it, and bounds the violation. I verified each of those and they hold up.

I nonetheless land at borderline, for one reason that dominates everything else and four that compound it.

### The central problem: the MMLU-Pro balanced null cannot generate MMLU-Pro data

The paper's most quantitatively confident claim — that MMLU-Pro, MMLU and BoolQ have floors "a balanced null could not produce ($p<10^{-5}$)" — is produced for MMLU-Pro by a null outside the support of any legal MMLU-Pro label vector. MMLU-Pro has variable option count; the paper says so, says `n_opt` runs from 3 to 10 across eight strata, and builds its entire v2 contribution on the principle that a letter illegal for an item must never be credited to it. The winner's-curse null in Tables 1 and 8 draws `multinomial(12032, [1/10]×10)`: it credits letter `J` on items where `J` does not exist.

This is visible from the paper's own printed columns, without simulation. For any balanced label process, `f_hat = max_L m_hat_L ≥ m_hat_A`, so `E[f_hat] ≥ E[m_hat_A]`. The row "MMLU-Pro, item-avg." prints `Chance = 0.110877` — which the paper itself defines as `mean(1/n_opt)`, and that is exactly `E[m_hat_A]` under the legal balanced process, since A is legal on every item — and prints `E[f_hat] = 0.104460` for the balanced null on the same item set. The expectation of a maximum is printed smaller than the expectation of one of its arguments. I applied that test to all nine rows: eight pass, and the single violation is the MMLU-Pro item-average row. The naive row conceals the defect only because `1/10` happens to fall below `0.104457`.

Quantitatively: I reproduced the paper's own recipe exactly (`E[f]=0.104456` against printed `0.104457`, `q95` identical, `p=0`), which confirms my implementation and locates the problem in the recipe. Then I ran the legal null — gold drawn uniformly over each item's own legal option set — across three very different `n_opt` histograms consistent with both constraints the paper publishes (`mean(1/n_opt) = 0.110877`, and 623 items in strata 5–8 from Appendix D). The result is `p = 0.068`, `0.093`, `0.120`. The observed floor sits below the 95th percentile of every legal null. A histogram-free bound agrees: `sd(m_A) ∈ [0.00248, 0.00286]`, so `z ∈ [2.00, 2.31]` and one-sided `p ∈ [0.010, 0.023]` for letter A alone, and the reported statistic is a maximum over at least three exchangeable candidates, which can only raise it.

I want to be precise about the blast radius, because it is narrower than "the paper is wrong." The floor value `1403/12032 = 0.116606` is a descriptive property of the item set and is untouched. Every arm-versus-floor comparison is untouched. The flip result, the v2 statistic and the collapse identity are untouched. MMLU (`k=4`) and BoolQ (`k=2`) have legal nulls, and I re-verified that both reproduce and stay at `p ≈ 0`. What breaks is the magnitude claim for MMLU-Pro, which after correction joins the five constructs the paper already describes as lying inside the estimator's own noise. The honest count becomes two of eight, not three, and MMLU-Pro is not among the survivors.

Two things make this more serious than a arithmetic slip. First, MMLU-Pro is the paper's primary benchmark and the only construct where variable `n_opt` applies — the defect lands exactly where the paper leans hardest and is absent everywhere it would not have mattered. Second, the appendix advertises an emitter that "refuses to write on any inconsistency," listing four specific guards; none of them tests `E[f_hat] ≥ Chance`, and the table ships violating it.

The redeeming feature is that the fix costs nothing. The per-item `n_opt` vector already exists, because the v2 stratification consumes it. Re-drawing the balanced null within each item's legal support, reporting the corrected `p`, and moving MMLU-Pro into the "inside estimator noise" partition would turn the paper's worst defect into an instance of its own thesis — that the reference must be specified before it can be trusted. The paper's existing scoping language already covers the corrected case.

### Three statistical claims that do not clear their own bars

**The `0/60` aggregate is close to a criterion that cannot fail.** Section 4 and Appendix D both say the conclusions "rest on the aggregate's near-unanimity (0/60 damaged cells clear their floor), not on one cell." The same paper reports that 52 of those 60 cells are underpowered for a 1.389 pp effect. Using the paper's own half-widths, I computed how many of 60 cells would show a CI excluding zero above floor if *every* arm truly cleared its floor: 3.5 at `+0.5` pp, 7.6 at `+1.0`, and 12.0 at `+1.389` — the paper's own reference effect. A design that expects twelve detections in a world where all sixty arms clear their floor cannot support "near-unanimity" from observing zero. It does exclude large positive effects (`P(0/60 | δ=+1.0 pp) = 2e-4`), which is a real but far weaker statement. This is the paper's own "power is part of construct validity" principle applied to its own headline aggregate, and the aggregate does not survive it. The paper knows how to do this correctly — it labels one MMLU result a "powered non-replication" — so the fix is available.

**The one surviving inferential claim on the chance side assumes independence the paper denies.** Appendix D declines family-wise correction on the explicit ground that the cells "share items, nest arms, and share a null, so the tests are neither independent nor exchangeable." Section 5.1 then computes `P(X≥3 | 12, 0.05) = 0.0196` as iid Bernoulli over exactly those twelve tests and offers it as what survives correction. The arithmetic is right — I confirmed `0.019568` exactly — and creditably it is a bar that could have failed, since `P(X≥2) = 0.118`. But the model is the one the paper rejects. Under an exchangeable beta-binomial at the same marginal, `ρ = 0.05` gives `0.050` and `ρ = 0.10` gives `0.068`. Twelve cells scored on the same 12032 items against the same null do not have `ρ ≈ 0`.

**The materiality gate is a point estimate whose interval straddles its own threshold.** The decision on Qwen3 `k14` — the lone above-floor exception, the withdrawn capability label, the reason the headline is 14/15 rather than universal — turns on `recovery_fraction = 0.049`, "only 9.1% of the intact-family anchor and below the 10% materiality bar." That is a 0.9-point margin on a ratio that receives no uncertainty anywhere in the paper. I verified the point value is internally coherent (the printed 94.6% modal share gives `Δ_max = 5.437` pp and `0.267/5.437 = 0.0491`), then propagated the table's own half-width of 0.188: the ratio's 95% interval is `[2.7%, 15.5%]` and contains 10%. "A real but immaterial exception" is a call the data cannot make at the paper's own confidence level. `recovery_fraction` is the only statistic in the paper that never gets a CI, and it is the one adjudicating the headline exception. The 0.10 constant is also asserted without derivation and fires exactly twice, both times within a factor of two of the bar, in a re-analysis the paper itself calls "post-hoc by construction."

Related: the 27-cell v2 re-analysis receives no correction, and neither "trace signal" cell survives BH over the paper's own tabulated family. Running BH at `q=0.05` over the 27 printed p-values, six reject — five intact/shortgpt16 anchors plus Qwen3 `k14` at `0.0066` — and OLMo-2 `keep14` at `0.0172` is retained. That is the *only* cell moving upward, so the claim that "re-sorting in both directions is evidence that the criterion is doing more than shrinking effects" rests on one uncorrected borderline cell. The general defence against correcting is weaker here than for the aggregates: Table 7 *is* the family — one table, one null, one statistic, one item set.

### What is genuinely strong

The stratified permutation null and its collapse identity are real. I implemented the estimator from the paper's equations alone on a synthetic 12032-item set with eight strata and fed it all ten pure constant emitters: `Δ_perm = 0` to `≤1.4e-15` for every letter, including letters illegal on most items. The claim is true and stronger than stated. It also genuinely repairs a defect the paper honestly documents in its own v1 floor, which assigns 0.000, −2.111 and −3.807 pp to three equally empty emitters. The paper is right to disclaim the statistic (it is the κ numerator, and it says so) and right to claim the stratification and the gating use.

The ordering appendix is the best-argued passage in the paper and is exactly self-consistent: `1403/12032 = 0.116606`, `1439 − 1403 = 36` items, `36/12032 = 0.2992` pp. The authors volunteer that their ordering is not a theorem under stratification, name the violating emitter, bound the tightest cell's attainable violation, and then discount their own 27-cell evidence count as pseudo-replication. I looked for something wrong with it and found nothing.

The internal bookkeeping is clean. All 27 rows of Table 7 satisfy `Δ_perm = 100·(acc − acc_hat)`; all 11 numeric rows of Table 6 satisfy `Δ = 100·(acc − 0.116606)`; the CI/p/verdict implication holds in 11 of 11; and normal-approximation p-values reconstructed from each row's own half-width agree with the printed bootstrap p within Monte Carlo noise on all 27. The power table's half-widths are not mechanical `1/sqrt(n)` rescalings — Winogrande and CommonsenseQA differ 2.9× at nearly equal `n`, which is what a genuinely paired statistic does. The fp32 block is coherent under forensic reconstruction: `2532/14042 = 18.033%`, and the printed McNemar `p=0.5702` implies roughly 1.3–1.4k correctness-discordant pairs, a proper subset of the argmax changes, exactly as it should be.

And Section 5.1 deserves saying plainly: the authors found their own headline was asymmetric, fixed it, and published a number a third the size. I reconstructed both floor-side counts from Table 6 and they match exactly. Papers do not usually do this.

### Reproducibility

The snapshot ships two evidence files: a LaTeX build log and a claim ledger. Every quantitative evidence id used in a caption — E-A, E-B, E-D, E-E, E-F, E-H, E-I, E-K, E-CAL — resolves to a path outside the artifact. Against a reproducibility statement that opens "every quantitative claim in this paper is bound to a machine-readable record," this is the gap that most constrains the review: I could check internal arithmetic and the behaviour of the stated nulls, which is how I found the central defect, but not one number against its source. Two consequences are concrete. The per-cell chance-side deltas are printed nowhere, so the paper's most-quoted result cannot be reconstructed from the submission even in principle, while I reconstructed the floor side exactly. And the "610 numerals with none unresolved" claim cannot be evaluated — though I did find one numeral a strict checker should have caught, the caption range "+0.43--+2.60 pp" whose smallest actual member is PIQA's `+0.490`. I also note that `build_record.json` reports `pdf_visually_inspected: false` and that its recorded `pdf_sha256` and `pdf_bytes` disagree with the manifest entry for the shipped `main.pdf`, so the build record does not describe the submitted PDF.

### Recommendation

Borderline, 5.5, ceiling 6.5 without new experiments. The methodological contribution is real, the writing is careful, and the self-correction record is better than the field's norm. But a paper whose thesis is "specify and calibrate your reference before you interpret a score" ships a headline calibration whose reference cannot generate the data, and it is detectable from two columns of the paper's own table. Alongside an aggregate that cannot fail, an independence-based global p in a paper that argues dependence, and a materiality gate whose interval straddles its bar, that is enough to hold the paper below acceptance in its current form.

Every one of these is fixable without a GPU. Correcting the null's support, reporting the pooled effect instead of `0/60`, bootstrapping `recovery_fraction`, adding BH q-values to Table 7, and shipping the cited evidence files would raise this to a clear accept, and — for the null in particular — would make the paper a better illustration of its own argument than it currently is. I would want to see the corrected MMLU-Pro `p` before recommending acceptance, and I would not want the authors to weaken the paper's genuinely honest passages while fixing it.

---

## APPENDIX: VERIFICATION LOG (written incrementally during review)

Environment: `/opt/conda/envs/torch-base/bin/python`, **numpy 2.5.1**, scipy NOT installed on this
node, so all tail probabilities were computed either exactly in `fractions.Fraction` / `math.lgamma`
or by Monte Carlo with numpy's `Generator(PCG64)`. Cross-node MC agreement is claimed only to ~1e-4,
per this cluster's known numpy split; no bit-identity is claimed.

### V1 — snapshot inventory (VERIFIED, neutral)
34 files, MANIFEST.json self-consistent with the prompt's snapshot sha256 `7fcb9cc...`. Only TWO
machine-readable evidence files are shipped: `evidence/build_record.json` and
`evidence/claim_evidence_map.tsv`. Every quantitative evidence artifact the manuscript cites by id
(E-A, E-CAL, E-D, E-F, `s2_03_symmetric_inference.json`, `floor_winners_curse_calibration.json`,
`construct_nulls_length_unit.json`, `s2_02_stratified_ordering.json`) is NOT in the snapshot. The two
shipped files are a LaTeX build record and a claim ledger; neither contains a single per-item record,
bootstrap draw, or p-value.

### V2 — stored-precision / discrete-atom check on `tab_nulls` + `tab_construct_nulls` (VERIFIED)
The floor estimator is discrete: `f_hat = max_L count_L / n`, atoms spaced `1/n`. Exact rational
recomputation of `n·floor`:

| construct | n | n·floor | count | exact count/n | stored vs exact | observed atom in `>=`? |
|---|---|---|---|---|---|---|
| MMLU-Pro | 12032 | 1403.0034 | 1403 | 0.1166057181 | stored **>** exact | **NO** |
| MMLU | 14042 | 3776.0061 | 3776 | 0.2689075630 | stored **>** exact | **NO** |
| OpenBookQA | 500 | 138.0000 | 138 | 0.2760000000 | equal | yes |
| ARC-Easy | 2376 | 632.9997 | 633 | 0.2664141414 | stored < exact | yes |
| ARC-Challenge | 1172 | 310.9996 | 311 | 0.2653583618 | stored < exact | yes |
| CommonsenseQA | 1221 | 254.9997 | 255 | 0.2088452088 | stored < exact | yes |
| PIQA | 1838 | 928.0007 | 928 | 0.5048966268 | stored **>** exact | **NO** |
| BoolQ | 3270 | 2033.0015 | 2033 | 0.6217125382 | stored **>** exact | **NO** |

In 4 of 9 rows the rounded decimal sits strictly above the realised atom and drops it from the tail.
Anti-conservative for MMLU-Pro/MMLU/PIQA/BoolQ, conservative for the rest. No verdict changes →
minor. Also verified: the `Gap (pp)` column is arithmetically correct in all 9 rows, except that
CommonsenseQA's 0.8845 is printed `+0.884` in one table and `+0.885` in the other.

### V3 — the winner's-curse null REPRODUCES (VERIFIED, in the paper's favour)
Re-ran the recipe stated in the `tab_construct_nulls` caption (balanced multinomial at each
construct's own `(n,k)`, seed 20260814, 200,000 draws):

| construct | E[f] paper | E[f] mine | q95 paper | q95 mine | p paper | p mine (dec) | p mine (int) |
|---|---|---|---|---|---|---|---|
| MMLU-Pro naive | 0.104457 | 0.104457 | 0.107048 | 0.107048 | <1e-5 | 0.00000 | 0.00000 |
| MMLU-Pro item-avg | 0.104460 | 0.104457 | 0.107131 | 0.107048 | <1e-5 | 0.00000 | 0.00000 |
| MMLU | 0.254355 | 0.254349 | 0.258225 | 0.258225 | <1e-5 | 0.00000 | 0.00000 |
| OpenBookQA | 0.273191 | 0.273183 | 0.294000 | 0.294000 | 0.383 | 0.38370 | 0.38370 |
| ARC-Easy | 0.260611 | 0.260604 | 0.270202 | 0.270202 | 0.140 | 0.13823 | 0.13823 |
| ARC-Challenge | 0.265105 | 0.265113 | 0.279010 | 0.279010 | 0.453 | 0.45364 | 0.45364 |
| CommonsenseQA | 0.215006 | 0.214986 | 0.226863 | 0.226863 | 0.853 | 0.85291 | 0.85291 |
| PIQA | 0.509307 | 0.509291 | 0.522851 | 0.522851 | 0.658 | 0.65573 | **0.68954** |
| BoolQ | 0.506981 | 0.506964 | 0.517125 | 0.517125 | <1e-5 | 0.00000 | 0.00000 |

Genuine, independently reproduced calibration. The two MMLU-Pro rows differ in E[f] and q95 despite
the caption saying both use the same `(n,k)` null — my single run gives one value for both, so those
appear to be two independent MC runs. PIQA is where the V2 rounding defect bites: correct
integer-count tail 0.690 vs printed 0.658.

### V5 — **MAJOR, QUANTIFIED.** Support-respecting balanced null gives MMLU-Pro p ≈ 0.07–0.12
Gold drawn uniformly over **each item's own legal options**. Constraints used: `mean(1/n_opt) =
0.110877` (§4 summary) and `n_5+n_6+n_7+n_8 = 623` (App. `app:ordering`). Seed 20260814, 100,000
draws:

| null | E[f_hat] | q95 | p = Pr(f_hat ≥ 1403/12032) |
|---|---|---|---|
| **paper's**: multinomial(12032, [1/10]×10) | 0.104456 | 0.107048 | **0.000000** |
| legal, excess via `n_opt=3` | 0.113441 | 0.116938 | **0.0679** |
| legal, excess via `n_opt=4` | 0.114012 | 0.117271 | **0.0925** |
| legal, excess via `n_opt=9` | 0.114502 | 0.117603 | **0.1197** |

The observed floor 0.116606 sits *below* the q95 of every legal null. Distribution-free cross-check:
`sd(m_A) ∈ [0.002479, 0.002862]` (bounding `mean(p²)` by Jensen and by `p ≤ 1/3`), so `z ∈ [2.00,
2.31]`, one-sided `p ∈ [0.010, 0.023]` for letter A alone; the statistic is a max over ≥3 exchangeable
candidates, which only raises p.

### V6 — the 0.0196 figure is arithmetically CORRECT and CAN fail
Exact `Fraction` binomial, n=12, p=1/20: `P(X≥3) = 0.019568…` → paper's `0.0196` right to 4 dp.
`P(X≥2) = 0.1184`, so 2/12 would have failed; the floor side's 1/12 gives `P(X≥1) = 0.4596`.

### V7 — MAJOR: that test assumes independence the paper elsewhere denies
Exchangeable beta-binomial at the same marginal p=0.05 (exact, via log-Beta):

| ρ | P(X≥3) |
|---|---|
| 0 (paper) | **0.0196** |
| 0.05 | 0.0501 |
| 0.10 | 0.0678 |
| 0.20 | 0.0829 |
| 0.30 | 0.0860 |

### V8 — power table internally consistent (VERIFIED, strength)
`half-width ≤ 1.389` reproduces 6/6 yes/no labels. Implied per-item sd 21.5–73.0 pp; Winogrande
(1.184 @ n=1267) vs CSQA (3.399 @ n=1221) differ 2.9× at near-equal n — legitimate for a paired
statistic.

### V9/V12 — power semantics, and MY OWN CORRECTION
The rule `hw ≤ |d|` is 50% power. 80% needs `hw ≤ d/1.4296 = 0.972` pp. **But** computing actual
power `Φ(d/se − 1.96)` from the MMLU-Pro prose range hw ∈ [0.083, 0.968]: the worst cell is **80.3%**,
so the abstract's 21-cell claim **survives at 80%** — I withdrew that part of V9. What remains:
`tab_power`'s ARC-Easy "yes, borderline" is **55.0%**, Winogrande "yes" is **63.3%**, MMLU reference
**65.5%**; the "no" rows are 6–16% and correctly labelled. Downgraded to minor.

### V10/V11 — internal arithmetic of both per-cell tables is CLEAN (VERIFIED, strength)
`tab_v2_full`: 27/27 rows satisfy `Δ_perm = 100·(acc − acc_hat)` to ≤5e-4 pp; row count exactly 27;
normal-approx p reconstructed from each row's own `|Δ|` and half-width agrees with the printed
bootstrap p within MC noise on all 27. `tab_mmlupro`: 11/11 rows satisfy `Δ = 100·(acc − 0.116606)`,
and `|Δ| > hw` ⇔ `p < 0.05` ⇔ verdict label holds 11/11.

### V13 — MAJOR: BH over the paper's own 27-cell family
| rank | p | BH thr | cell | outcome |
|---|---|---|---|---|
| 1–5 | 0.0001 | 0.00185–0.00926 | 4 intact + OLMo-2 shortgpt16 | REJECT |
| 6 | 0.0066 | 0.01111 | **Qwen3 k14** | REJECT |
| 7 | 0.0172 | 0.01296 | **OLMo-2 keep14** | **retained** |

Bonferroni (α/27 = 0.00185) rejects only the 5 anchors. OLMo-2 keep14 is the only upward-moving cell
and is the one BH drops.

### V14 — fp32/McNemar block internally coherent (VERIFIED)
2532/14042 = 18.033% ✓. −0.0015 × 14042 = −21.1 items → |b−c| ≈ 21. If all 2532 argmax changes were
correctness-discordant, exact McNemar gives p≈0.676; the printed 0.5702 implies ~1.3–1.4k discordant
pairs, a proper subset. Mutually consistent; not independently checkable.

### V15 — symmetric flip counts RECONSTRUCT (VERIFIED, strength)
12 non-OLMo cells: point-above-floor = **3/12** (Llama-2 k14 +0.017, Qwen3 k14 +0.233, Qwen3 k10
+0.158); `Δ − hw > 0` = **1/12** (Qwen3 k14). Both match. Llama-3 is a collapsed range with both
endpoints negative. The chance-side column (10/12, 3/12) is NOT reconstructible — per-cell chance
deltas are printed nowhere.

### V16/V17 — `recovery_fraction` coherent, but its CI straddles its bar
From modal-A 0.946: 11382 A-preds, 650 others → best alignment (1403+650)/12032 = 0.1706 →
`Δ_max = 5.437` pp → `recovery = 0.267/5.437 = 0.0491` → printed 0.049 ✓.
Propagating the table's own hw = 0.188:

| | Δ_perm | recovery | % of anchor | verdict |
|---|---|---|---|---|
| CI lower | 0.079 | 0.0145 | **2.7%** | immaterial |
| point | 0.267 | 0.0491 | **9.1%** | immaterial |
| CI upper | 0.455 | 0.0837 | **15.5%** | **MATERIAL** |

Interval `[2.7%, 15.5%]` **contains 10%**.

### V18 — the 0.10 constant is asserted and decides both cells it touches by < 2×
Qwen3 k14 at 0.91× the bar (blocked); Llama-2 intact at 0.545× ("anchor blocked").

### V19 — **MAJOR: `0/60` is largely forced by low power**
Using `tab_power`'s own five half-widths, expected "clears floor" count if every arm truly cleared by δ:

| true δ | expected count |
|---|---|
| +0.50 pp | 3.5/60 |
| +1.00 pp | 7.6/60 |
| +1.389 pp | **12.0/60** |
| +3.00 pp | 29.3/60 |

`P(0/60 | δ=+1.0 pp) = 2e-4`; `P(0/60 | δ=+0.5 pp) = 0.026`.

### V20 — MINOR: "ten target constructs" not derivable
Letter side checks out (8 distinct = 9 rows − duplicated MMLU-Pro row; 3 + 5 = 8 ✓). But §4 gives 7
content constructs, and 8 + 7 = 15 ≠ 10 even after the stated OBQA de-duplication.

### V21 — content-convention numbers: 3 of 5 spot-checks tie out
Verified: MMLU-Pro `credit` 0.532164 / `wrong` 0.125914 ✓ (span 0.406250 → "40.6-point span" ✓);
OBQA `credit` 0.416/0.644 ✓. Unverifiable: the "32.5 points" claim implies an intact `content_norm`
≈ 0.2072 printed nowhere; ARC-C "18.60% vs 50.85% tied-longest" has no table column. Oddity:
ARC-C letter floor 0.265358 (`tab_nulls`, L*=B) **equals** ARC-C content token-`first` 0.265358
(`tab_conventions`) to all six digits.

### V22 — citation/bib hygiene (VERIFIED, strength)
17/17 bib keys cited; 17/17 cited keys resolve; build record reports 0 undefined refs/citations, 0
LaTeX errors, 0 overfull boxes, 22 pages. No venue or claim-level support was verified (no network).

### V23 — the constant-collapse identity is REAL (VERIFIED by independent implementation)
Implemented the estimator from the §3.2 equations alone on a synthetic 12032-item set, `n_opt ∈
{3..10}`, eight strata. `Δ_perm = 0` to ≤1.4e-15 for **all ten** pure constant emitters, including
letters illegal on most items — stronger than the paper's stated claim.

### V24 — the ordering disclosure is exactly self-consistent (VERIFIED, notable strength)
`1403/12032 = 0.116606` ✓; `1439 − 1403 = 36` ✓; `36/12032 = 0.2992` pp → "0.299 pp" ✓. Could not
check "13 of 27" — the v1 floor margin is not printed per cell for 14 of the 27.

### V25 — one further unverifiable-but-suspicious item
`tab_nulls`'s caption says the inside-noise gaps are "+0.43--+2.60 pp"; the smallest such gap is
PIQA's **+0.490**. No row in either table has a gap of +0.43.

### V26 — **DECISIVE, MC-FREE FORM OF V5**
For any balanced process `E[f_hat] ≥ E[m_hat_A]`. Row "MMLU-Pro, item-avg." prints
`Chance = 0.110877` (= `mean(1/n_opt)` = `E[m_hat_A]`, A legal on every item) and
`E[f_hat] = 0.104460`. **0.104460 < 0.110877.** Applying the test to all nine rows: **8/9 pass; the
one violation is the MMLU-Pro item-average row** — the row using the paper's own honest chance line,
on the construct the paper rests on. The naive row hides it because 0.100 < 0.104457.
