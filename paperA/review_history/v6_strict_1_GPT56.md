review_mode: strict
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 3.0

# Strict ARR Review

## Paper Summary

This paper studies a narrow repeated-query serving design point for long-context
decoder LMs. CoMem writes each document chunk once to a single intermediate
residual \(h_j\) per token, retrieves a bounded number of chunks, and executes
only layers \([j:L)\) online. Its central causal comparison is deliberately
internal: on Qwen3-8B, matched \(j=0\) raw-token replay and \(j=12\) residual
replay use the same examples, selected chunk IDs/order, mask, sink, and LoRA.
The latter reduces isolated selected-pack Read latency from 931.9 to 664.4 ms
(\(1.403\times\)) while reducing a paired 15-cell RULER macro from 99.19 to
96.07 (gap 3.12, 95% CI [2.36, 3.93]).

The paper also accounts for the 8 KiB/token residual store and one-time Write,
reports repeated-query break-even points, compares raw replay and CoMem under a
nine-cell equal-online-latency diagnostic, and investigates why independently
written chunk residuals lose quality. A \(2\times2\) context/position control
and an Overlap-Write intervention indicate that omitted lower-layer left context
is a substantial cause on one paired synthetic multikey cohort. The authors
appropriately frame the work as an internal operating-point measurement rather
than superiority over RAG, PIC, or modular-KV systems.

## Claims and Minimum-Sufficient-Evidence Map

I reconstructed the following substantive claims before evaluating the supplied
evidence.

| ID | Claim | Minimum sufficient experiment | Actual evidence | Assessment |
|---|---|---|---|---|
| C1 | Skipping lower layers provides an incremental Read-speed benefit at a measurable quality cost under identical evidence. | Same model, adapter, examples, pack, mask, and timing boundary; paired quality and repeated latency runs. | Main Table 2; Appendix Tables 31–32; 1,500 paired RULER-B examples; three independent latency processes. | **Supported for the measured 16k-source, ~6.5k-pack Read boundary.** |
| C2 | A small LoRA makes the independently written residual interface usable without changing the backbone. | Same split and inference protocol with adapter on/off. | Appendix Tables 9 and 11, plus Table 12 as a transfer diagnostic. | **Supported on tested tasks; not a general guarantee.** |
| C3 | The 8 KiB/token store can amortize under repeated queries. | End-to-end write-once comparison including fetch, model Read, and decode across reuse counts and storage tiers. | Main Table 3 and Appendix efficiency text; selected 32k and 128k cells. | **Supported at the reported harness points, with limited uncertainty reporting.** |
| C4 | At equal online latency, CoMem has no aggregate win; the result depends on the replay selector. | Disjoint latency calibration, explicit estimand, paired scores, and dependence-aware cell/task analysis. | Main Table 4; Appendix Table 8; stratified, hierarchical, and leave-one-cell-out analyses. | **Supported for the defined nine-cell mixture. The v6 reanalysis fixes the central dependence error.** |
| C5 | Missing lower-layer document context is a major source of the tested residual-interface loss. | Factorial intervention holding retrieval/adapter/upper Read fixed, ideally paired. | Main Table 5 and Appendix Table 32 on 8k/16k multikey. | **Supported only as a bounded diagnosis; the interaction prevents single-factor additive attribution.** |
| C6 | Small left overlap repairs much of that local loss without increasing persistent bytes or per-query Read. | Same examples and Read path, varying only Write overlap; report quality and Write cost. | Main Table 6; \(w=32\) gives +6.0 points, CI [3.0, 9.5], with theoretical Write-FLOP ratios. | **Supported on the displayed synthetic cohort, not on the full suite or repaired serving frontier.** |
| C7 | Model-side Read remains bounded as the persistent store grows, while selection/storage remain unbounded. | Fixed \(k\) over increasing stores, measuring read length, retrieval behavior, and lookup cost. | Appendix Table 25, plus store-I/O Table 33. | **Supported as an implementation property; quality remains evidence-budget/task dependent.** |
| C8 | The implementation ports beyond the principal dense 8B model. | Exact split-forward test and at least one quality run on another architecture. | Appendix Hy3 exact-logit split test and Tables 34–35. | **Supported as portability evidence, not as a matched replication or scaling law.** |

## Strengths

### S1. The central depth-reuse claim is unusually well isolated.

The strongest aspect is the matched \(j=0\rightarrow12\) endpoint. The authors
hold evidence, order, examples, mask, sink, and adapter fixed, then change only
where replay begins. Main §5.1 / PDF p.5, lines 316–329 and Table 2 / PDF p.6
make the causal boundary explicit. Appendix Table 31 / PDF p.21 further gives
paired quality, process-level latency replication, and exclusion boundaries.
This is much more informative than comparing unrelated long-context systems.

### S2. The paper is careful not to oversell the quality–efficiency trade-off.

The manuscript repeatedly states that the \(1.403\times\) result is isolated
model Read, not quality-preserving or end-to-end acceleration. It distinguishes
depth reuse from bounded selection and separates selected-pack Read,
store-ready prefill, Write-inclusive pipeline, and external store/index cost
(§1 / PDF pp.1–2; §5 setup / PDF p.5, lines 301–314; Appendix A.4 / PDF
pp.19–21). Negative findings are retained rather than hidden.

### S3. The dependence-aware reanalysis materially resolves the key statistical issue.

For the nine-cell equal-latency mixture, v6 defines the estimand as the
equal-weight mean of cell-level paired differences, reports a fixed-cell
stratified bootstrap, a hierarchical cell-then-example bootstrap, and
leave-one-cell-out sensitivity (Appendix Table 8 / PDF p.12; Appendix B.4 / PDF
p.24, lines 1208–1225). This is the right direction because the 900 observations
are not exchangeable across heterogeneous task cells.

The corrected conclusions are appropriately narrower:

* BM25 replay has a robust advantage: hierarchical CI
  \([-18.67,-5.11]\), fixed-cell CI \([-14.33,-8.78]\), and all nine
  leave-one-cell-out estimates remain negative.
* The frozen-BGE comparison is unresolved: hierarchical CI
  \([-10.67,8.33]\), fixed-cell CI \([-4.56,2.56]\), and the leave-one-cell-out
  range crosses zero.
* The paper no longer treats the pooled-IID interval as the task-mixture
  inference and explicitly says CoMem wins neither aggregate.

Thus, **yes: the dependence-aware reanalysis solves the principal inferential
problem for the stated nine-cell estimand.** Remaining concerns are about the
estimand/cohort itself and training-run uncertainty, not the previously
incorrect IID resampling unit.

### S4. The mechanism analysis is controlled and appropriately bounded.

The continuous-prefix control exactly recovers full replay, establishing that
upper-layer continuation itself can be faithful (Table 2 and Appendix Table
32). The \(2\times2\) context/position intervention then shows a non-additive
\(+4.5\) interaction rather than forcing a simplistic attribution (Table 5 /
PDF p.6). Overlap-Write is presented as a local engineering hypothesis and its
persistent-storage/Read invariants are stated precisely (Table 6 / PDF p.7).

### S5. Scope and reproducibility disclosures are substantially above average.

The appendices identify the backbone revision, adapter SHA-256, exact split,
mask, positions, retrieval parameters, prompts/decoding, benchmark supports,
sample counts, optimizer, training schedule, hardware, and timing boundaries
(Tables 26–28 / PDF pp.19–20). The paper discloses contamination concerns,
judge mutability, incomplete total compute accounting, single-run limitations,
and unavailable measurements rather than silently extrapolating them.

### S6. The rendered paper is readable despite a large appendix.

I inspected both figures and all 38 rendered table objects (34 `table`
environments plus four `captionof{table}` objects). The main tables are
legible; cohort labels A/B and timing boundaries are generally explicit; no
unresolved references were visible. Figure 1 gives a useful end-to-end diagram,
and Figure 2 is explicitly framed as motivation rather than validation.

## Weaknesses, Ordered by Severity

### W1. No matched nearest-system baseline, so excitement and external value remain limited. **Major**

* **Location:** Related Work, §2 / PDF p.3, lines 148–154; Limitations / PDF
  p.7, lines 441–449.
* **Exact quote (9 words):** “We lack a matched same-backbone implementation of these systems”
* **Problem:** The paper's nearest deployment alternatives are PIC,
  chunk-KV-repair, activation restoration, and learned modular-KV systems, but
  none is implemented under the same backbone, selected pack, storage tier,
  training budget, and latency boundary. Taxonomy and unmatched external rows
  cannot establish whether a single residual/token is a useful Pareto point
  relative to the closest reusable-context interfaces.
* **Affected claim/norm and why it matters:** This does not invalidate C1, but
  it limits novelty/excitement and prevents a main-conference-level systems
  conclusion. The internal endpoint may be carefully measured yet still be
  dominated by a nearby cache design.
* **Sufficient remedy:** Implement at least one nearest feasible comparator
  (e.g., independent per-chunk KV/PIC with its prescribed repair, or an
  HCache/KV-Direct-style reusable residual path) on Qwen3-8B using the same
  examples, selector/pack, quality metrics, H20 timing boundary, persistent
  bytes, Write cost, fetch tier, and decode settings; report a quality–latency–
  storage frontier rather than a single point.

### W2. Headline quality and adapter behavior are not supported by clean run-level uncertainty. **Major**

* **Location:** Limitations / PDF p.7, lines 431–440; Appendix Table 29 / PDF
  p.21.
* **Exact quote (7 words):** “The flagship is one batch-8 training run.”
* **Problem:** The exact 15-cell RULER-B, LongEval, and LoCoMo headline results
  come from one adapter run. The two extra adapters change effective batch
  size from 8 to 3 and are evaluated on reduced-support cells, so they conflate
  initialization with optimization-noise/batch effects and do not estimate
  uncertainty for the headline claims.
* **Affected claim/norm and why it matters:** C1's paired evaluation uncertainty
  is sound conditional on this trained adapter, but C2 and the claimed
  deployable operating point may be sensitive to training randomness. This is
  especially important because quality gaps are only 3.12 RULER points and
  3.32 LoCoMo points versus matched raw replay.
* **Sufficient remedy:** Train at least three independent adapters with
  identical effective batch, data order budget, objective, and hyperparameters;
  rerun the exact RULER-B headline and at least the key natural-task aggregates,
  reporting run-level means/intervals and the distribution of quality–latency
  trade-offs.

### W3. Natural-task evidence remains vulnerable to unmeasured training-corpus overlap. **Major**

* **Location:** Limitations / PDF p.8, lines 500–512; Appendix A.3 / PDF p.18,
  lines 902–923.
* **Exact quote (10 words):** “Equivalent overlap audits were not completed for all natural benchmarks”
* **Problem:** The adapter is trained on PG-19, and the paper found substantial
  overlap with one long-book benchmark support. It then removed that comparison,
  but did not complete equivalent audits for several natural benchmarks,
  including NarrativeQA. Labeling them “scope checks” is honest but does not
  establish that gains or retained quality are not partly due to corpus overlap.
* **Affected claim/norm and why it matters:** This weakens any natural-task
  generalization interpretation of C2 and the external scope checks. Synthetic
  RULER evidence remains useful, but the paper's broader empirical relevance
  relies on natural tasks.
* **Sufficient remedy:** Apply the same documented n-gram/near-duplicate audit
  to every natural benchmark, release aggregate overlap diagnostics, and report
  clean-subset or leave-overlap-out results where overlap is non-negligible.

### W4. The equal-latency dependence analysis is fixed, but its deployment estimand is still narrow and partly non-identifiable. **Minor**

* **Location:** Appendix Table 8 / PDF p.12; Appendix B.4 / PDF p.24, lines
  1208–1232.
* **Exact quote (9 words):** “The first 100 LoCoMo examples are all conversation 0”
* **Problem:** The hierarchical bootstrap validly treats the nine selected task
  cells as the heterogeneity unit, but the nine-cell set itself is a small,
  author-defined mixture with equal cell weights, and its LoCoMo cell has only
  one conversation cluster. Resampling cell labels quantifies sensitivity to
  the observed cells; it does not make them a random representative sample of
  deployment tasks, and within-LoCoMo conversation dependence cannot be
  estimated.
* **Affected claim/norm and why it matters:** C4 is sound only for the explicitly
  defined mixture. A selector-independent or broad deployment conclusion would
  still be unsupported; fortunately, the paper mostly avoids that
  overgeneralization.
* **Sufficient remedy:** Predefine a broader task/length mixture, sample LoCoMo
  across multiple conversations, and report both fixed-cell inference for that
  benchmark set and task-family/cluster sensitivity. Keep the current
  hierarchical analysis as a robustness analysis rather than a population
  guarantee.

### W5. The proposed overlap repair is not yet evaluated on the claims that matter for deployment. **Minor**

* **Location:** Method §4 / PDF p.4, lines 250–257; Main §5.3 / PDF pp.5–7.
* **Exact quote (15 words):** “This is a tested repair on one diagnostic, not the default for the full suite.”
* **Problem:** Overlap-Write repairs one paired synthetic multikey cohort, but
  the paper does not test it on RULER-B as a whole, natural tasks, repeated-query
  break-even, edit invalidation, or an end-to-end quality–latency–storage
  frontier. The reported theoretical Write-FLOP ratios also understate measured
  wall time.
* **Affected claim/norm and why it matters:** C6 is correctly local, but the
  mechanism result cannot yet support the natural next inference that the
  principal CoMem operating point is practically repairable.
* **Sufficient remedy:** Evaluate \(w=32\) on the exact headline benchmark suite
  and serving harness, report measured Write latency and amortization, and
  include edit/update invalidation cost. A smaller minimum is full RULER-B plus
  one natural benchmark and the 32k break-even grid.

### W6. Several systems and compute measurements remain insufficiently auditable from the frozen package. **Minor**

* **Location:** Table 3 caption / PDF p.6; Appendix A.4 / PDF pp.19–21; compute
  accounting / PDF p.19.
* **Exact quote (7 words):** “raw component times and dispersion are archived.”
* **Problem:** The frozen source/PDF supplies derived crossover values and some
  medians, but not the raw timing archive, scripts, score exports, adapter, or
  environment lockfiles referenced by the text. Total experimental GPU-hours
  and training peak memory are also unrecorded. Therefore I could check the
  displayed arithmetic and timing boundaries, but not independently reproduce
  the crossover grid, bootstrap outputs, or full compute accounting from the
  materials I was permitted to inspect.
* **Affected claim/norm and why it matters:** C3 and reproducibility are
  partially supported rather than independently verifiable. The issue is not
  that the paper necessarily lacks an external artifact, but that the frozen
  review package does not contain it.
* **Sufficient remedy:** Attach the anonymized artifact promised in the paper,
  including raw timing records, exact scripts/configs, score-only exports,
  manifests, pinned environment, and adapter weights; add run dispersion for
  every headline crossover cell and a complete compute ledger.

## Questions That Could Change the Score

1. Can the authors provide a matched Qwen3-8B comparison to one nearest reusable
   PIC/modular-KV or activation-restoration baseline under the exact Table 2/3
   boundaries? A non-dominated point could raise excitement and Overall.
2. Across clean same-effective-batch training seeds, how variable are the exact
   RULER-B gap, LoCoMo score, and LongEval score? If the central trade-off is
   stable, W2 would shrink substantially.
3. Does \(w=32\) improve full RULER-B and at least one natural task, and what is
   its measured Write-inclusive break-even? A positive answer would turn the
   bounded diagnosis into a more consequential systems result.
4. Are the promised raw timing records, score exports, bootstrap script/results,
   adapter, and hashes available in the anonymous artifact? If so, please point
   reviewers to the exact files and commands.
5. Was the nine-cell equal-latency mixture specified before inspecting the
   per-cell deltas? Please clarify the selection rationale and intended target
   population for cell-level resampling.

## Non-Scoring Suggestions and Typos

1. Table 18 reports matched raw replay RULER-B as 99.20, whereas Tables 2/31 and
   prose use 99.19. This appears to be rounding, but one convention should be
   used throughout.
2. In Table 18's caption, MemoryLLM is described as a “released 7B model,” while
   other captions call the released Llama-3-8B-chat checkpoint. Harmonize the
   description.
3. Define “Raw-text reader quality” in Table 14 more explicitly: it is unclear
   from the table alone whether it is unconditional accuracy or conditional on
   retrieval hit.
4. The distinction among Cohort A, Cohort B, the nine-cell robustness subset,
   and the equal-latency nine-cell mixture is careful but cognitively expensive.
   A one-page cohort inventory would improve readability.
5. The abstract contains many numerals and intervals. They are all traceable,
   but prioritizing the central endpoint, storage cost, and equal-latency
   conclusion would improve accessibility.

## Detailed Score Justification

### Soundness: 3.5 / 5

The core matched endpoint is technically credible, arithmetic is consistent,
the quality comparison is paired, and the timing claim is narrowly defined.
The dependence-aware equal-latency reanalysis is a real correction: it uses the
cell as the heterogeneity unit and reaches appropriately qualified conclusions.
I do not assign 4.0 because the central adapter is a single training run, natural
task overlap remains incomplete, and several artifact-dependent analyses cannot
be independently rerun from the frozen package.

### Excitement: 3.0 / 5

The depth-axis framing, transparent negative result, and context/position
diagnosis are useful. However, caching intermediate representations and reusable
document state are established ideas, and the paper itself acknowledges that
its contribution is an internal two-point measurement. Without a matched
nearest-system baseline or broad validation of the repair, the likely impact is
more diagnostic than state-of-the-art.

### Overall: 3.0 / 5

This is a solid **Findings-level** paper: careful, honest, and informative, with
a well-supported central trade-off and a materially improved statistical
analysis. I am uncertain between 3.0 and 3.5, and under the requested
calibration choose the lower bin. The decisive reasons are W1–W3: no matched
nearest reusable-cache baseline, no clean multi-run uncertainty for headline
quality, and incomplete contamination audits. These limit the work below ACL
main-conference level despite good internal validity.

### Confidence: 4.0 / 5

I read the complete 24-page PDF twice, including all appendices, inspected every
figure/table, reconstructed headline arithmetic, checked source/PDF consistency,
and audited all `main.bbl` entries. Confidence is not 5.0 because artifact
contents and a subset of web-based bibliography/novelty checks remained
unverifiable.

### Reproducibility: 3.0 / 5

The manuscript's procedural specification is strong and includes hashes,
versions, supports, seeds, masks, and timing boundaries. However, the permitted
frozen package contains manuscript source rather than the promised runnable
artifact; training peak memory and total experimental compute are missing; and
headline results lack clean repeated training runs.

## Limitations, Ethics, and Desk-Reject Risks

### Formal/desk audit

* **Main-paper length:** exactly eight numbered pages before references. Pass.
* **Total PDF:** 24 pages including references and appendices.
* **Limitations:** exact unnumbered `Limitations` section begins on PDF p.7,
  within the main eight pages. Pass.
* **Ethical considerations:** present on PDF p.8 and substantively discusses
  sensitive memory stores, inversion/membership risk, authorization, deletion,
  misuse, and energy. Pass.
* **Anonymity:** title page says “Anonymous ACL submission”; no author,
  affiliation, repository owner, or identifying acknowledgment was found in
  rendered text. Pass.
* **Style/rendering:** review mode, line numbers, A4 page geometry, embedded
  fonts, and readable two-column rendering. No unresolved cross-references were
  found. Pass, subject to venue-specific style-version validation.
* **TODO/placeholders:** no TODO/TBD/FIXME/XXX or placeholder text found. Pass.
* **Prompt injection/reviewer manipulation:** no instruction to reviewers,
  acceptance request, score manipulation, white-text command, zero-scale text,
  JavaScript, launch action, or suspicious hidden text was found in the frozen
  source/PDF/embedded figures. Pass.
* **Abstract numbers:** checked at least five headline values against tables:
  931.9→664.4 ms and \(1.403\times\) (Tables 2/31), 99.19→96.07 and CI
  [2.36,3.93] (Tables 2/31), 8 KiB/token (Eq. 1/Table 2), 8.9–10.9 and
  25.8–27.6 break-even (Table 3), \(-11.56\) and \(-1.00\) equal-latency gaps
  with hierarchical intervals (Tables 4/8), and 92.5→100.0 / 98.5 overlap
  diagnostics (Tables 5/6). Pass.

I see **no clear desk-reject condition** in the frozen submission.

## Formula, Boundary-Case, Metric, and Systems Audit

1. **Storage equation:** Eq. (1) is correct under standard GQA dimensions:
   residual bytes/token \(=d\), full per-layer KV across \(L\) layers
   \(=2Ln_{\mathrm{kv}}d_{\mathrm{head}}\), giving
   \(n_q/(2Ln_{\mathrm{kv}})=32/(2\cdot36\cdot8)=1/18\). In bf16 this is
   8,192 versus 147,456 bytes/token.
2. **Read-length bound:** \(\mathrm{sink}+kc+\mathrm{query}\) is independent of
   stored-document length for fixed \(k,c\), but selector, index, and store are
   correctly stated to be unbounded. Boundary failure when evidence exceeds
   top-\(k\) is explicitly demonstrated.
3. **Distillation objective:** the symmetric weighted KL on a shared
   teacher-top-64 support is defined consistently. Discarding outside-support
   mass makes it an approximate objective; the paper discloses that retained
   teacher mass was not logged.
4. **Matched endpoint:** using the same LoRA in both \(j=0\) and \(j=12\) arms
   avoids conflating adapter presence with depth reuse. The continuous-prefix
   oracle correctly acts as a compatibility ceiling, not a deployable cache.
5. **RULER arithmetic:** the 15 Cohort-B cells sum to 1441.0 and average
   96.0667; \(99.19-96.0667=3.1233\). The latency ratio is
   \(931.9/664.4=1.4026\).
6. **Equal-latency arithmetic:** the nine reported BM25 deltas average
   \(-11.5556\); the BGE deltas average \(-1.0000\). The dependence-aware
   estimand and resampling units are now stated consistently.
7. **Context/position interaction:** the difference-in-differences is
   \((100-88)-(100-92.5)=+4.5\), so the paper correctly rejects additive
   attribution.
8. **Metrics:** official RULER, BABILong, and LongBench scoring is named.
   LoCoMo's primary semantic judge is mutable and partly local for category 5;
   the paper reports date, parser, denominators, a conversation-cluster
   comparison, and an independent-judge subset, but exact future reproduction
   remains impossible.
9. **Seeds/statistics:** evaluation seeds and bootstrap seeds are disclosed.
   Paired/example and cluster units are mostly appropriate. The major remaining
   seed issue is training-run rather than evaluation-sample uncertainty.
10. **Compute:** final training cost is reported (~2.9 H20 GPU-hours), but total
    research compute and training peak memory are unavailable. Latency uses
    medians on single-query harnesses rather than concurrent p95/tail behavior.

## Complete Citation Audit (`main.bbl`)

All 43 bibliography entries are actually cited; no cited key is missing and no
`main.bbl` entry is unused. Status uses the required categories. “Verified”
means metadata was confirmed from an authoritative identifier/page during this
audit. Per the user's stop instruction, incomplete network checks are
**Unverifiable**, not “Not found.”

| Key | Work (short title) | Status | Audit note |
|---|---|---|---|
| `cachecraft` | Cache-Craft | Unverifiable | DOI present; full metadata not completed before stop. |
| `longbench` | LongBench | Verified | ACL Anthology/DOI metadata consistent. |
| `pyramidkv` | PyramidKV | Unverifiable | arXiv identifier present; venue metadata not fully checked. |
| `kvpacket` | KV Packet | Verified | arXiv 2604.13226, title/authors/date consistent; before cutoff. |
| `cartridgesbase` | Cartridges | Metadata error | arXiv/title/authors verified, but `volume=2026, pages=42642--42687` is not credible ICLR proceedings metadata and should be corrected. |
| `hcache` | HCache | Unverifiable | DOI present; not fully checked. |
| `llama3` | Llama 3 report | Unverifiable | arXiv identifier present; not fully checked. |
| `cartridges` | Cartridges at Scale | Verified | arXiv 2606.04557 metadata consistent; post-cutoff contemporaneous work. |
| `distillation` | Distilling the Knowledge | Unverifiable | Standard work; no identifier in `main.bbl` and not rechecked. |
| `ruler` | RULER | Unverifiable | arXiv identifier present; not fully checked. |
| `lora` | LoRA | Unverifiable | arXiv identifier present; not fully checked. |
| `epic` | EPIC | Unverifiable | arXiv identifier present; not fully checked. |
| `ragcache` | RAGCache | Unverifiable | DOI present; not fully checked. |
| `babilong` | BABILong | Unverifiable | DOI present; not fully checked. |
| `rag` | Retrieval-Augmented Generation | Unverifiable | Standard work; no identifier in `main.bbl` and not rechecked. |
| `longchat` | LongChat benchmark/blog | Unverifiable | URL present; not fully checked. |
| `snapkv` | SnapKV | Unverifiable | DOI present; not fully checked. |
| `ilre` | ILRe | Verified | arXiv 2508.17892 metadata consistent. |
| `readonce` | ReadOnce Transformers | Unverifiable | ACL DOI present; not fully checked. |
| `minicache` | MiniCache (KV depth compression) | Unverifiable | DOI present; not fully checked. |
| `turborag` | TurboRAG | Unverifiable | ACL DOI present; not fully checked. |
| `locomo` | LoCoMo | Unverifiable | ACL DOI present; not fully checked. |
| `xccache` | XC-Cache | Unverifiable | ACL DOI present; not fully checked. |
| `kvdirect` | Residual Stream Is All You Need / KV-Direct | Verified | arXiv 2603.19664 title/authors/date consistent; before cutoff. |
| `pg19` | Compressive Transformers / PG-19 | Unverifiable | arXiv identifier present; not fully checked. |
| `bm25` | BM25 and Beyond | Unverifiable | DOI present; not fully checked. |
| `embeddingrecycling` | Embedding Recycling | Unverifiable | ACL DOI present; not fully checked. |
| `gemfilter` | GemFilter | Verified | arXiv 2409.17422 title/authors consistent; cited ACL-2026 venue metadata not independently completed. |
| `reform` | REFORM | Unverifiable | arXiv identifier present; not fully checked. |
| `lloco` | LLoCO | Unverifiable | ACL DOI present; not fully checked. |
| `hunyuan` | Hy3 model page | Unverifiable | Hugging Face URL present; model-page metadata not fully checked. |
| `fusionrag` | Fusion RAG Cache | Verified | arXiv 2601.12904 metadata consistent; before cutoff. |
| `mepic` | MEPIC | Verified | arXiv 2512.16822 metadata consistent. |
| `longmem` | LongMem | Unverifiable | Standard venue entry; no identifier in `main.bbl` and not rechecked. |
| `memoryllm` | MemoryLLM | Unverifiable | arXiv identifier present; venue details not fully checked. |
| `infllm` | InfLLM | Unverifiable | DOI present; not fully checked. |
| `streamingllm` | StreamingLLM | Unverifiable | arXiv identifier present; not fully checked. |
| `sempic` | SemPIC | Verified | arXiv 2607.28069 metadata consistent; post-cutoff contemporaneous work. |
| `xu2024retrievallong` | Retrieval Meets Long Context LLMs | Unverifiable | Venue entry has no identifier in `main.bbl`; not rechecked. |
| `qwen3` | Qwen3 Technical Report | Unverifiable | arXiv identifier present; not fully checked. |
| `ape` | APE | Unverifiable | arXiv identifier present; not fully checked. |
| `cacheblend` | CacheBlend | Unverifiable | DOI present; not fully checked. |
| `h2o` | H2O | Unverifiable | Venue entry lacks an identifier in `main.bbl`; not rechecked. |

### Load-bearing citation–claim matches

1. **ReadOnce / Embedding Recycling → reusable intermediate text
   representations:** claim match is appropriate.
2. **HCache → activation checkpoint/state restoration:** appropriate, though its
   workload is state restoration rather than persistent document retrieval.
3. **KV-Direct → reconstructing layer-wise KV from residuals:** appropriate and
   especially close at the stored-object level.
4. **CacheBlend / TurboRAG / Cache-Craft → reusable chunk KV with fusion/repair:**
   appropriate as nearest serving-workload families.
5. **EPIC / MEPIC / APE → independent/parallel context encoding with position or
   boundary handling:** appropriate.
6. **Cartridges / KV Packet / SemPIC → learned reusable modular KV objects:**
   appropriate; however, Cartridges at Scale and SemPIC are post-cutoff
   contemporaneous works and should not reduce originality priority.
7. **SnapKV / PyramidKV → retained-KV compression after full prompt processing:**
   appropriate as secondary budget references, not nearest cross-query
   persistence baselines; the paper correctly states this distinction.
8. **RULER / BABILong / LongBench / LoCoMo / LongChat:** benchmark-attribution
   matches are appropriate, with LongEval tied to the LongChat source/blog.

## Novelty Search Summary

The requested novelty cutoff is **2026-05-04**. I conducted five query families
before the instruction to stop expansion:

1. persistent intermediate residual memory for LMs;
2. residual-stream caching/reuse in transformer inference;
3. reusable intermediate representations for text transformers;
4. activation replay / suffix-layer execution;
5. transformer depth reuse for long-context inference.

### Closest works

1. **ReadOnce Transformers (2021):** caches reusable intermediate text
   representations and adapts later computation. This is the clearest conceptual
   predecessor to “store an intermediate representation and read it later.”
2. **Embedding Recycling (2023):** reuses intermediate language-model
   representations across tasks/uses; another strong representation-reuse
   predecessor.
3. **HCache (2025):** checkpoints activations and resumes computation for state
   restoration; close in suffix execution, less close in repeated-query
   document retrieval.
4. **KV-Direct (2026-03-20, before cutoff):** uses residual streams to reconstruct
   layer-wise KV, making it the closest pre-cutoff stored-object/interface
   neighbor found.
5. **KV Packet (2026-04-14, before cutoff):** independently compiled,
   context-independent reusable KV with boundary adaptation; close in workload
   and deployment objective, though it stores a learned KV object rather than
   one residual per token.

Also relevant are CacheBlend, TurboRAG, EPIC/MEPIC/APE, Cartridges, and
RAGCache. **Cartridges at Scale (2026-06-03) and SemPIC (2026-07-30) are after
the 2026-05-04 cutoff and should be treated as contemporaneous/post-cutoff
context, not priority-destroying prior art.**

### Novelty judgment

I did not find evidence that the broad idea of reusable intermediate
representations or reusable document state is new. The credible novelty is the
specific combination of:

* one persistent depth-\(j\) residual per selected token;
* direct suffix execution on a bounded retrieved pack;
* an explicit split-depth deployment axis;
* a same-evidence/same-adapter \(j=0\) control with storage/Write/Read
  accounting; and
* a bounded context/position diagnosis.

That is a **narrow empirical/interface novelty**, not a new caching paradigm.
One additional search result, **LLMCache: Layer-Wise Caching Strategies for
Accelerated Reuse in Transformer Inference** (arXiv 2512.16843), appeared
potentially relevant but was not fully audited for validity or claim overlap
before the stop instruction; I therefore mark its bearing on novelty
**Unverifiable** rather than drawing a conclusion from it.

## Unverifiable

The following items could not be established from the frozen PDF/source alone
or were not completed before the explicit stop instruction:

1. Full authoritative metadata verification for every bibliography entry listed
   as Unverifiable above.
2. Completeness of the novelty search beyond the five query families and the
   cited literature; in particular, the status and substantive overlap of
   LLMCache (2512.16843).
3. Existence/completeness of the promised anonymous artifact, raw timing
   records, score-only exports, bootstrap script/results, adapter weights, and
   pinned environment.
4. Independent recomputation of bootstrap intervals, McNemar counts, judge
   results, and crossover points from raw predictions/timings.
5. Whether the included `acl.sty` is the exact currently required ARR style
   release rather than a locally copied compatible version.
6. Exact future reproducibility of the undated mutable `gpt-4o` judge endpoint.

## Review-Process Self-Check

* Read pass 1: complete main paper, references, both appendices, all tables,
  figures, limitations, ethics, and statistical appendix.
* Read pass 2: rechecked claims against minimum sufficient experiments,
  formulas, arithmetic, boundary cases, metrics, seeds, timing boundaries,
  compute, reproducibility, novelty, and bibliography.
* Inspected the rendered 24-page PDF as a contact sheet and page text; inspected
  both embedded figure PDFs separately.
* Checked all 43 citation keys against `main.bbl`: all are cited, none missing.
* Checked references/labels mechanically: 58 unique labels, no missing or
  duplicate references detected.
* Searched source/PDF/figures for prompt injection, reviewer manipulation,
  hidden/white/tiny text, unresolved references, TODOs, and anonymity leaks.
* Recomputed the principal means, differences, ratio, storage equation, and
  context/position interaction.
* Mechanically located every weakness quote in the frozen source; every quote is
  at most 25 words.
* Every “paper lacks X” criticism above is tied either to an explicit paper
  admission or to absence within the permitted frozen package; no weakness
  relies on another review, history, TODO, status, current, or calibration file.

