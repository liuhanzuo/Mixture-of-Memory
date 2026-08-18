---
review_type: "artifact-verification-strict"
version: "v7"
review_object: "v7_20260804_025333.pdf + v7_source_20260804_025333/anonymous_artifact"
review_scope: "Independent read of only the supplied v7 PDF, v7 source tree, anonymous_artifact, and STRICT requirements. No prior reviews/history/TODO/status/current materials consulted."
overall_score: 3.0
confidence: 4.0
soundness: 3.0
presentation: 4.0
contribution: 2.5
artifact_score: 3.5
recommendation: "weak reject"
verification:
  artifact_file_count: "PASS: 38 regular files, exactly as claimed."
  artifact_sha256: "PASS: all 37 entries in anonymous_artifact/SHA256SUMS.txt verify."
  score_recomputation: "PASS (released records): all six 14,042-row content-MMLU JSONLs reproduce their stored letter/raw/norm accuracies exactly; keep14 128k→200k recomputes +1.680672 pp and bootstrap 95% CI [1.075345, 2.285999] pp."
  paired_headline: "PASS: released paired JSON and recomputation agree on the keep14 trajectory headline; fixed-checkpoint item uncertainty only."
  closed_book: "PARTIAL: aggregate summaries exactly support the reported headline values and sample sizes, but no aligned predictions/generations are released, so the aggregates and any paired uncertainty cannot be independently recomputed."
  source_hash_manifest: "UNVERIFIABLE: REPRO_SHA256.txt references two arrays and five checkpoints absent from the source package; sha256sum -c therefore fails to read all seven referenced objects. This is consistent with the stated compact/no-weights release but does not verify those hashes."
  scripts: "PARTIAL: all six scripts py_compile; eval_mmlu_content.py --selftest passes. Full evaluator execution is blocked here by absent checkpoints, validation array, locally cached model/tokenizer required by local_files_only=True, and datasets/runtime GPU resources."
  anonymity: "PASS with caveat: no email, private filesystem path, credential, private URL, or benchmark question/option text observed in the 38-file release. Scripts/configs deliberately contain generic prompt templates and public dataset/model identifiers."
  historical_training_replay: "NOT REPRODUCIBLE: weights/checkpoints, training array, historical seed, and keep14's within-epoch loader offset after the 34.5k resume are unavailable; local evaluator commit ancestry is not recoverable."
---

# Strict artifact-verification review — Paper B v7

## Summary and recommendation

This is a careful, unusually candid single-run OLMo-2 prune/regrow measurement case study. Its central supported statement is narrow: on the observed keep14 path, lower same-source held-out PPL did not imply recovery to intact-base performance on answer-letter MMLU or the three reported closed-book QA measures. The manuscript repeatedly distinguishes this from causal localization, a universal recovery law, or a prospective PPL certificate. That restraint, clear construction accounting, and readable Figure 1 are strengths.

The v7 frozen review object materially improves auditability: `anonymous_artifact/` is present, internally hash-consistent, anonymous on inspection, and sufficient to recompute the released six-arm MMLU summaries and the keep14 128k→200k paired headline. However, it remains a compact *evaluation snapshot*, not a reproducible experiment package: neither training nor end-to-end headline re-evaluation can be rerun from the supplied objects, PPL and closed-book headline aggregates cannot be regenerated, and the historical training realization is explicitly unrecoverable. The paper is sound as a bounded observational report, but novelty is modest and the evidence remains one confounded realization with no 200k intact control. **Weak reject** under a strict standard; the artifact itself is a meaningful partial pass rather than a reason to reject for anonymity or fabricated provenance.

## Artifact verification

### 1. Inventory, integrity, and consistency with the paper

- **38/38 files present.** The artifact consists of six scripts, three configs, one local-commit snapshot, ten closed-book aggregate summaries, six MMLU summaries plus six per-item MMLU JSONLs, two paired per-item trajectory JSONLs, two paired-analysis JSONs, README, and SHA256 manifest. This matches the Appendix’s described compact release.
- **Hash manifest passes.** `sha256sum -c anonymous_artifact/SHA256SUMS.txt` returned OK for every one of its 37 listed payload files. The base MMLU JSONL hash also matches the manuscript’s stated prefix/suffix: `52f60373c952dd86...c1e07766e0364701`.
- **The paper/source hash manifest does not pass on this package.** `REPRO_SHA256.txt` lists `data/dolmino_now15b.npy`, `data/dolmino_now_val.npy`, and five checkpoint paths; none is shipped. Verification therefore fails with seven “No such file or directory” entries. The manuscript says the source package is intentionally compact and excludes weights, so this is not a contradiction, but the claim “currently retained arrays and headline checkpoints have SHA-256 checksums” is **Unverifiable** from the submitted artifact.
- **Manuscript–artifact numerical alignment is good for covered records.** The MMLU protocol config specifies the paper’s base model, MMLU revision, 14,042 items, letter/content definitions, and 10,000 bootstrap resamples. The released closed-book summaries contain the same task counts and headline metrics used in the paper: e.g., base/keep14 PopQA containment `.25710/.14152`, TriviaQA EM `.63548/.29403`, and NQ-open EM `.20499/.05983`.

### 2. Per-item MMLU and paired headline

This is the strongest part of the release.

- Each of the six headline-arm MMLU files has **14,042 rows**, with stable item ID, subject, gold/predicted index or letter, option-level scores, continuation-token counts for content-normalized scoring, and correctness; no question or option strings occur in these records.
- Direct recomputation from the JSONL reproduces every stored summary exactly:
  - base: letter `.60539809`, content-norm `.47058824`;
  - full32-25k: `.58766557`, `.46624412`;
  - keep14-200k: `.31840194`, `.38320752`;
  - frozen: `.26235579`, `.36041874`;
  - random: `.24697337`, `.35977781`;
  - ShortGPT: `.47422020`, `.40115368`.
  These support the main table after rounding and the paper’s interface-sensitivity discussion.
- The keep14 paired data also support the late-path headline. Recomputing paired correctness gives 128k `.30116792`, 200k `.31797465`, difference **+1.680672 pp**, and paired-bootstrap 95% CI **[+1.075345, +2.285999] pp**. These match the stored JSON and manuscript. The stored exact McNemar statistic is also available. This verifies an *item-conditional* comparison of two fixed checkpoints, not training-run uncertainty.
- The generic `paired_analysis.py` is not directly wired to this release’s layout or the 128k/200k comparison (its defaults expect five task-specific `keep14/random_init` files). The paper’s trajectory headline is nevertheless independently recomputable by a small reader script from the supplied JSONLs. This is a usability defect, not a numerical discrepancy.

### 3. Closed-book, PPL, and executable code

- **Closed-book QA: aggregate-only.** The ten summaries are coherent with the main/appendix tables and list counts, hit counts, and EM/containment/F1. But generations and aligned per-item predictions are omitted, as the paper says. Hence I can verify table transcription and internal arithmetic only at the summary level; I cannot recreate the outputs, validate normalization/alias handling against actual examples, or calculate paired intervals.
- **PPL: not reproducible from this snapshot.** `eval_ppl.py` implements token-weighted shard merging and strict checkpoint loading, but it requires the missing validation `.npy`, checkpoint, and local model/config. The artifact has no PPL shard outputs or PPL summaries. Thus the reported PPL values and the central joint PPL–target observation are not independently rerunnable here.
- **Scripts: plausible and partially executed.** All six Python scripts compiled successfully. `eval_mmlu_content.py --selftest` passed its tiny CPU model, option-score normalization, independent log-probability, McNemar, bootstrap, and schema checks. The code pins MMLU’s public revision and implements the stated no-BOS/content protocol. Full run capability remains **Partial**, because all model loads use `local_files_only=True`, checkpoints are absent, PPL validation data are absent, and production evaluation requires a compatible CUDA environment plus datasets. The closed-book scripts name public datasets but do not pin revisions in the release config; this makes future exact reruns less secure than the MMLU path.

### 4. Anonymity, privacy, and benchmark-text audit

The v7 package passes the requested release-safety check.

- I found **no author names, emails, private filesystem paths, credentials, private URLs, or benchmark questions/options** in the 38 artifact files.
- The visible `Question: {q}\nAnswer:` strings are protocol templates in evaluator code/config, not leaked benchmark records. The MMLU records have item IDs, subjects, gold labels, scores, and token counts but not item text.
- The local provenance manifest truthfully gives abbreviated IDs only and explicitly says that public ancestry is not claimed. That is acceptable disclosure, not reproducible provenance.
- The source package’s figures/PDF are anonymous; no author metadata or identifying path was observed. Figure PDF metadata names rendering software and timestamps only.

### 5. What remains irreproducible

The manuscript appropriately discloses, but cannot remedy, the following:

1. exact historical training replay for keep14 and other trained arms: missing checkpoints/weights, training windows, seed, and full launch state;
2. exact keep14 data order after the 34.5k resume: the within-epoch loader offset was lost and the distributed shuffle restarted;
3. a 200k full32 control: no such checkpoint is available;
4. PPL recomputation: missing validation array/checkpoints and no released shard data;
5. full closed-book result regeneration or uncertainty: missing checkpoint/generations/per-item predictions;
6. recovery compute/GPU-hours and wall-time;
7. historical evaluator commit graph: source contents are snapshotted, but the local-only commit ancestry cannot be audited.

## Claims, design, and technical audit

### Claims and desk fit

The principal claim is accurately scoped and supported by the reported data *conditional on the unreproduced checkpoints*: keep14’s PPL decreases from 10.826 to 10.561 while late MMLU rises only from .301 to .319 and remains well below base .605; the endpoint also remains below base on the three closed-book metrics. The paper does not overstate this as “PPL is useless,” knowledge deletion, localization, causation, or a universal law. This is good scientific hygiene.

The desk concern is contribution scale. The work offers no new pruning/recovery method, has a single principal realization, and provides an observational diagnostic package rather than a new general result. Its contribution is potentially publishable as a careful measurement note, but likely too incremental for a high-selectivity main conference unless the artifact can support a stronger, reproducible empirical claim.

### Design and statistics

The paper correctly labels full32, random, frozen, and ShortGPT as operating points rather than clean ablations. Still, that means they only rule out limited complete stories, not the mechanisms implied by readers’ intuitive interpretation. ShortGPT changes selected/inherited layers, contiguity, final-layer retention, and fresh-tail use; random also changes LR and lexical modules; frozen changes the trainable set. The 25k-only full32 branch does not control 200k corpus exposure. All trained constructions are one run, and checkpoint retention/stopping was target-informed for shallow arms. Item bootstraps and McNemar tests are correctly described as conditional on realized checkpoints, but cannot address seed/data-order/block-selection variance.

### Numerical audit

Covered headline numbers are internally consistent. The artifact verifies the MMLU and paired values above, while the closed-book aggregate values match the tables. The discrepancies between `.3191` trajectory MMLU and `.3184` artifact content-MMLU letter accuracy are explained by the paper as distinct reruns; that distinction is explicit and appropriate. PPL values, PPL trajectory, and closed-book aggregate derivation remain **Unverifiable** without the missing primary outputs/checkpoints.

### Citations and novelty

Citation coverage is strong in the manuscript: cited keys resolve to the included bibliography, and related work candidly acknowledges prior recovery curves, loss–task gaps, initialization comparisons, and beyond-PPL evaluation. I did not perform prolonged external literature search per instruction, so priority/completeness beyond the included bibliography is **Unverifiable**. The paper’s own positioning makes the realistic novelty claim modest: a particular OLMo measurement/control combination and reporting discipline, not a new recovery mechanism or pruning criterion.

### Figures and presentation

The 17-page PDF is professionally typeset and legible. Figure 1 is particularly effective: it visually separates the literal keep14 trajectory from endpoint/null operating points and visibly states the full32 25k limitation and random content-score floor. Tables label interface and construction differences well. Minor presentation concern: the paper is dense for a narrow single-case contribution, and some appendix-heavy reproducibility prose cannot substitute for the missing release objects.

## Mechanical weakness ledger (four required elements each)

1. **Problem:** The central evidence cannot be independently regenerated end-to-end from the artifact.  
   **Evidence:** `REPRO_SHA256.txt` cannot be checked because it references two missing arrays and five missing checkpoints; PPL has neither data/checkpoints nor shard outputs; closed-book has only summaries.  
   **Impact:** The strongest conclusion is auditable only as a report of fixed stored numbers, not as a rerunnable measurement result.  
   **Required fix:** Release, under an appropriate access mechanism, the five evaluation checkpoints and validation shard outputs/arrays or a reproducible data-fetch recipe; release closed-book per-item predictions/generations (or sufficient privacy-safe hashes and shard outputs) plus a one-command verification script.

2. **Problem:** The training result remains a one-run, confounded observation with no long-horizon intact comparator.  
   **Evidence:** keep14/ShortGPT/random/frozen are single runs; full32 ends at 25k; ShortGPT/random/frozen alter multiple factors; historical seed and loader offset are unavailable.  
   **Impact:** The paper supports proxy insufficiency on literal paths, but not a robust estimate of recovery dynamics, construction effects, or causal mechanisms.  
   **Required fix:** Run preregistered multi-seed keep14 and full32 trajectories to a matched horizon, record seed/data cursor/hardware, and add factor-isolating constructions (selection, final layer, inherited count, fresh tail, LR/trainable set).

3. **Problem:** The claimed value beyond existing “beyond perplexity” and pruning-recovery literature is limited.  
   **Evidence:** The paper itself says it contributes neither a new pruning criterion nor recovery-path analysis, and Table 1 identifies substantial antecedents.  
   **Impact:** A careful case study may not meet main-track novelty expectations despite solid caveats.  
   **Required fix:** Either sharpen a demonstrably new, pre-specified measurement result with replication or reposition as an artifact/reporting/negative-results contribution with a correspondingly modest venue claim.

4. **Problem:** The released paired-analysis utility is not turnkey for the released files, and closed-book provenance is less pinned than MMLU.  
   **Evidence:** `paired_analysis.py` defaults to unavailable five-task directories rather than the supplied 128k/200k files; MMLU revision is pinned, while closed-book source revisions are not recorded in the protocol manifest.  
   **Impact:** A reviewer can recompute the MMLU headline manually, but a future user cannot execute a documented single command for all reported evaluation summaries.  
   **Required fix:** Add a top-level `verify_release.py`/Make target that verifies hashes and recomputes every available MMLU/paired/closed-book summary, plus pinned dataset revisions and dependency versions for every evaluator.

## Positive points

- The manuscript’s claim boundaries are unusually explicit and consistently honored.
- v7 fixes the key frozen-object issue: `anonymous_artifact` is included, contains exactly the advertised compact material, and passes its own complete hash manifest.
- Per-item MMLU records allow a real independent check of six headline arms and the late paired trajectory without distributing benchmark text.
- The authors distinguish item-level uncertainty from training-run uncertainty instead of using bootstrap intervals to overclaim replication.
- The artifact is anonymous and avoids private paths, identities, credentials, model weights, and benchmark text on inspection.

## Bottom line

Treat the artifact as **a credible, anonymous, hash-verified partial evaluation release**. It validates MMLU score accounting and one paired trajectory headline, but does not reproduce the underlying trained models, PPL evidence, or closed-book outputs. The manuscript’s restrained conclusion is largely sound as a descriptive case study; the strict recommendation remains **weak reject** because the result’s generality and novelty are limited and the end-to-end empirical claim is not yet reproducible.
