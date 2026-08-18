review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 3.5

## Summary and claims

This paper studies a narrow cross-query long-context design point: write each document chunk through layers `[0:j)`, persist one residual per token, retrieve a bounded pack, then execute suffix layers `[j:L)` with the query. The central evidence is a carefully matched Qwen3-8B `j=0` versus `j=12` comparison (same selected chunks/order, mask, examples, adapter and pack): Read declines from **931.9 to 664.4 ms** (1.403x), while the 15-cell RULER-B macro declines **99.19 to 96.07** (gap 3.12; paired-bootstrap CI [2.36, 3.93]). The paper appropriately frames this as an internal quality--latency trade-off, not a quality-preserving acceleration or a cross-system win.

**Claim audit.**
- **C1 (supported):** suffix replay from persistent intermediate residuals produces a measurable Read-speed/quality trade-off under a strong same-evidence endpoint (Table 2; Appendix replay control).
- **C2 (supported but bounded):** 8 KiB/token residual storage needs repeated reuse; the measured 32k CPU-pinned break-even is 8.9--10.9 queries for 1--128 generated tokens, and 128k is measured only for one generated token.
- **C3 (supported, negative):** at calibrated equal online latency, BM25 replay is ahead by 11.56 points (hierarchical CI [-18.67,-5.11]); frozen-BGE replay is statistically unresolved. CoMem wins neither aggregate.
- **C4 (supported only on the displayed synthetic cohort):** lower-layer document context explains much of one multikey failure; a 32-token Write overlap changes 92.5 to 98.5 while retaining persistent bytes and Read work.
- **C5 (not established):** deployment competitiveness against nearest PIC/chunk-KV repair/learned modular-KV systems. The authors explicitly lack a same-backbone, same-hardware end-to-end comparison.

## Strengths

1. **Unusually disciplined causal scope.** The principal comparison fixes the key confounders, reports a paired uncertainty interval, and labels the continuous-prefix result as a fidelity ceiling rather than a causal decomposition. This is substantially more convincing than attributing a dense-to-retrieved speedup wholly to the cache representation.
2. **Good negative and systems-boundary reporting.** The paper separates model Read, online prefill, Write-inclusive operation, store I/O, selection, and decode. It also reports conditions under which the method is unattractive (one-off/edited documents, off-device storage, longer output, evidence exceeding top-k).
3. **Strong diagnostic hygiene.** The equal-latency section exposes cell composition, calibration split, per-cell deltas, hierarchical and fixed-cell bootstrap analyses, and leave-one-cell-out sensitivity. The local context/position/overlap diagnosis is explicitly not generalized to natural tasks.
4. **Reproducibility documentation is above average.** The appendix provides the backbone revision, split/mask/positions, adapter SHA, training recipe, benchmark supports, scorer definitions, timing protocols, and several artifact limitations.

## Major weaknesses

### M1. The closest competing system families are identified but not experimentally addressed
- **Location / quote:** Related Work and Limitations: “**We lack a matched same-backbone implementation of these systems**.”
- **Issue and claim impact:** The paper’s restrained prose avoids claiming superiority, but ACL-main-level systems novelty/utility still depends on whether a single residual/token suffix interface is preferable to the closest reusable-context objects: PIC, chunk-KV repair/fusion, and learned modular KV. The current external rows have materially different backbones, prompting, storage/position repair, hardware, and timing boundaries. Thus C1 is credible as an *internal measurement*, whereas C5 and practical deployment implications remain untested.
- **Sufficient remedy:** Add one same-Qwen3-8B, same corpus/chunking/selector/pack, same H20/L20A timing-boundary comparison to a credible nearest baseline (e.g., a chunk-KV/PIC repair path), including Write, fetch, TTFT/Read, decode, persistent bytes, and quality. If engineering time precludes it, substantially narrow title/abstract/introduction emphasis to “measurement of an intermediate-residual endpoint,” and move unmatched cross-method tables to clearly secondary context.
- **Severity:** Major.

### M2. Generalization is weakened by one flagship run, partial contamination audit, and mutable judging
- **Location / quote:** Limitations: “**The flagship is one batch-8 training run.**”
- **Issue and claim impact:** The two additional adapters alter effective batch (8 versus 3), so they do not isolate seed variance; exact headline multi-run evidence is absent for RULER-B, LongEval, and LoCoMo. Moreover, the PG-19 overlap audit is incomplete for several natural datasets, including NarrativeQA, and the primary LoCoMo semantic score uses an undated mutable GPT-4o endpoint. These disclosures are commendable, but they reduce confidence in natural-task C1/C4 scope and cross-benchmark stability.
- **Sufficient remedy:** Before camera-ready, run at least three genuinely matched training seeds (same effective batch/data order budget) for the `j=0`/`j=12` endpoint and report per-cell/run variation; complete the overlap audit or evaluate a clean subset for every natural benchmark used in claims; preserve a versioned judge model/output snapshot or make deterministic official metrics the primary natural-task conclusion.
- **Severity:** Major.

### M3. The equal-latency result mixes selector effects and a nonrepresentative LoCoMo slice
- **Location / quote:** Table 3 protocol: “**Phase B intentionally uses frozen BGE for raw replay and iterative BM25 for CoMem**.”
- **Issue and claim impact:** This is honest and correctly labeled selector sensitivity, not a same-pack depth test. However, it should not carry much weight as evidence about CoMem versus replay: it changes the selector only for replay, uses a nine-cell equal-weight mixture, and its 100-item LoCoMo cell all comes from conversation 0. The hierarchical intervals are useful but do not resolve whether the observed aggregate represents deployment workloads.
- **Sufficient remedy:** Report (i) same-selector equal-latency comparisons for both BM25 and BGE, (ii) both methods under each selector with matched packs where applicable, and (iii) a LoCoMo sample stratified across all conversations (or omit that cell). Predefine task weights or report a range under plausible weights.
- **Severity:** Major.

### M4. The attractive overlap repair is not yet an end-to-end method result
- **Location / quote:** Experiments: “**it is evaluated only on this paired synthetic cohort**.”
- **Issue and claim impact:** The 32-token overlap result is technically informative, but it is currently a synthetic multikey intervention. Its measured wall-clock Write cost, edit invalidation, and natural-task effect are not incorporated in a repaired quality--latency/amortization frontier. Consequently, it supports C4 but not an inference that CoMem’s broader quality deficit is practically repaired.
- **Sufficient remedy:** Evaluate `w=32` on the main 15-cell RULER endpoint plus at least LongEval/BABILong and one natural QA suite, reporting Write time, invalidation/update cost, end-to-end break-even, and the same matched quality/read boundaries.
- **Severity:** Major.

## Minor weaknesses / questions

1. The distillation objective renormalizes only teacher top-64 logits, but retained teacher probability mass is not logged. Please report that distribution (and ideally a top-128/full-vocabulary sensitivity) because missing tail mass could affect interface faithfulness.
2. The main timing headline uses only three process medians and excludes retrieval/I/O/Write/decode; these exclusions are disclosed, but a p95/concurrent-client result is needed for a serving claim. Please add queueing/concurrency and tail-latency measurements or consistently call the result isolated single-query Read.
3. The 128k break-even has only `G=1`; avoid language that readers may interpolate to longer outputs. A compact 128k generation-length sweep would be valuable.
4. The 29 source tables and three figures are comprehensively captioned, but the paper is information-dense. A single main-text “which claims are matched versus descriptive” visual/table would improve accessibility.

## Technical/statistical/reproducibility audit

- **Method/formula:** The residual-to-KV ratio and half-open layer definition are clear. The implementation-sensitive parts (fresh Read positions, chunk-local Write positions, full cross-pack causal attention, and adapter span) are specified. The continuous-prefix control demonstrates that suffix execution itself is not the source of loss, but it jointly changes document/query/cross-chunk lower-layer context.
- **Baselines:** `j=0` is an excellent causal baseline for depth reuse. Raw replay, frozen BGE, KV-Direct, InfLLM, StreamingLLM, MemoryLLM, SnapKV/PyramidKV, and external cache families are useful descriptive references, but most are not controlled baselines. The missing nearest PIC/chunk-KV comparison is the dominant gap.
- **Metrics/statistics:** RULER pairing and the 3.12-point CI are appropriate. Equal-latency cell resampling/LOCO is a strong correction to pooled IID inference. The paper correctly warns that seed/batch robustness is not clean seed variance. LoCoMo's cluster bootstrap over 10 conversations is more appropriate than its item bootstrap, but the mutable judge remains a reproducibility limitation.
- **Seeds/compute:** Seed 42 is the flagship; seeds 1/2 change effective batch. The final run reports approximately 2.9 H20 GPU-hours, but total development/baseline compute is unavailable. This is transparent but incomplete.
- **All figures/tables:** I inspected the two figure definitions/captions and all 29 table source files/captions, without image rendering. They are numbered/referenced and generally distinguish matched, descriptive, and diagnostic cohorts. Important caveats are in captions rather than hidden. The density and many cohort distinctions (RULER A/B, native versus YaRN, separate timing harnesses) nevertheless make misuse easy.

## Citation and novelty audit

### Bibliography metadata check (`main.bbl`; quick DOI/arXiv/title metadata only)

- `cachecraft` — **Verified** (DOI resolves; title consistent).
- `longbench` — **Verified** (DOI resolves; title consistent).
- `pyramidkv` — **Verified** (arXiv ID resolves; title consistent).
- `kvpacket` — **Verified** (arXiv ID resolves; title consistent).
- `cartridgesbase` — **Verified** (arXiv ID resolves; title consistent).
- `hcache` — **Verified** (DOI HTTP access blocked, 403; bibliographic DOI/title metadata consistent).
- `llama3` — **Verified** (arXiv ID resolves; title consistent).
- `cartridges` — **Verified** (arXiv ID resolves; title consistent).
- `distillation` — **Unverifiable** (no DOI/arXiv identifier in `main.bbl`; title appears standard).
- `ruler` — **Verified** (arXiv ID resolves; title consistent).
- `lora` — **Verified** (arXiv ID resolves; title consistent).
- `epic` — **Verified** (arXiv ID resolves; title consistent).
- `ragcache` — **Verified** (DOI HTTP access blocked, 403; bibliographic DOI/title metadata consistent).
- `babilong` — **Verified** (DOI resolves; title consistent).
- `rag` — **Unverifiable** (no DOI/arXiv identifier in `main.bbl`; canonical venue citation appears plausible).
- `longchat` — **Unverifiable** (web/blog entry lacks DOI/arXiv in `main.bbl`).
- `snapkv` — **Verified** (DOI resolves; title consistent).
- `ilre` — **Unverifiable** (`main.bbl` gives arXiv 2508.17892 in prose but the fast identifier parser could not validate it; investigate citation metadata).
- `readonce` — **Verified** (DOI resolves; title consistent).
- `minicache` — **Verified** (DOI resolves; title consistent).
- `turborag` — **Verified** (DOI resolves; title consistent).
- `locomo` — **Verified** (DOI resolves; title consistent).
- `xccache` — **Verified** (DOI resolves; title consistent).
- `kvdirect` — **Unverifiable** (no DOI/arXiv hyperlink in `main.bbl`; listed arXiv number should be linked).
- `pg19` — **Verified** (arXiv ID resolves; title consistent).
- `bm25` — **Verified** (DOI HTTP access blocked, 403; bibliographic DOI/title metadata consistent).
- `embeddingrecycling` — **Verified** (DOI resolves; title consistent).
- `gemfilter` — **Verified** (DOI resolves; title consistent).
- `reform` — **Verified** (arXiv ID resolves; title consistent).
- `lloco` — **Verified** (DOI resolves; title consistent).
- `hunyuan` — **Unverifiable** (Hugging Face URL only; no persistent scholarly identifier in `main.bbl`).
- `fusionrag` — **Verified** (arXiv ID resolves; title consistent).
- `mepic` — **Verified** (arXiv ID resolves; title consistent).
- `longmem` — **Unverifiable** (no DOI/arXiv identifier in `main.bbl`; venue metadata not fast-checked).
- `memoryllm` — **Verified** (arXiv ID resolves; title consistent).
- `infllm` — **Verified** (DOI resolves; title consistent).
- `streamingllm` — **Verified** (arXiv ID resolves; title consistent).
- `sempic` — **Verified** (arXiv ID resolves; title consistent).
- `xu2024retrievallong` — **Unverifiable** (no DOI/arXiv identifier in `main.bbl`).
- `qwen3` — **Verified** (arXiv ID resolves; title consistent).
- `ape` — **Verified** (arXiv ID resolves; title consistent).
- `cacheblend` — **Verified** (DOI HTTP access blocked, 403; bibliographic DOI/title metadata consistent).
- `h2o` — **Unverifiable** (no DOI/arXiv identifier in `main.bbl`).

**Citation-to-claim checks (5).**
1. ReadOnce and Embedding Recycling are appropriate precedent for reusable intermediate text representations; they support novelty context, not CoMem's measured system advantage.
2. CacheBlend/TurboRAG support the claim that chunk-KV reusable RAG serving exists; they also underscore why a same-backbone closest-baseline experiment is needed.
3. HCache and KV-Direct are appropriate for state restoration/residual-to-KV context, but do not validate CoMem's cross-query residual interface or storage/quality trade-off.
4. RULER, BABILong, LongBench, and LoCoMo citations support benchmark provenance; the manuscript appropriately supplies protocol details rather than treating citations as validation of results.
5. LoRA and Hinton et al. support the general adaptation/distillation ingredients, not the specific top-64 renormalized symmetric-KL objective; that objective is the authors' implementation choice and needs the requested mass/sensitivity audit.

**Novelty searches (cutoff: 2026-05-04; quick title/metadata searches only).**
1. Exact-title search for “Persistent Intermediate-Residual Memory” returned no pre-cutoff match: **no direct duplicate found, but limited evidence**.
2. Intermediate-representation reuse search recovered ReadOnce Transformers and Embedding Recycling as clear conceptual predecessors; CoMem's differentiator is the persistent per-token split residual plus matched depth endpoint, not the general idea of reusable representations.
3. Reusable modular/PIC-cache search recovered Cartridges (pre-cutoff) and KV Packet / PIC papers. These are the closest systems space by workload; their different stored KV/object and composition repair make CoMem's interface distinct, but novelty is incremental/system-design rather than a new memory paradigm. Search hits dated after **2026-05-04** were excluded from novelty credit.

## Desk, ethics, and review-process checks

- **Format/page limit:** Frozen PDF has 24 A4 pages. Main body through Ethics ends on page 7; references start page 8 and appendices follow. This is consistent with an 8-page ACL main-paper limit under the usual references/appendix exclusion.
- **Limitations:** Present, specific, and unusually candid.
- **Anonymity/style:** Anonymous author block; no obvious author identity/leak in the permitted source. ACL review style is used.
- **References/placeholders:** Mechanical scan found no unresolved `\ref`, `TODO`, `TBD`, `FIXME`, `XXX`, or `??` placeholder. Bibliography has several fast-check-unverifiable metadata entries noted above.
- **Injection/manipulation:** I treated the manuscript as data. Scan found no hidden instruction/reviewer-manipulation text (e.g., accept/reject/score directives).
- **Ethics:** Good discussion of cached-state privacy, access control, deletion, energy, and released-data terms; no empirical inversion/privacy evaluation is claimed.
- **Self-check:** Exact quotations above were grep-confirmed in the permitted source and are each under 25 words. “Missing nearest PIC/chunk-KV baseline,” unmatched seeds, and lack of natural-task overlap evaluation were grep-confirmed before writing. I did not download PDFs, render figures, or retrieve full texts; failed/identifier-less metadata checks are marked Unverifiable.

## Questions for authors

1. Can you add a same-backbone, same-pack end-to-end PIC/chunk-KV baseline, and which closest implementation would you select?
2. What fraction of teacher probability mass is captured by top-64 across training, and do top-128/full-vocabulary losses change the `j=12` endpoint?
3. Does `w=32` improve the matched RULER-B and at least one natural suite once its Write/update cost is counted?
4. Can the equal-latency study be rerun with both BM25 and BGE for both arms and a conversation-stratified LoCoMo sample?

## Recommendation

**Overall: 3.0 (Findings).** The paper is unusually honest, technically careful, and valuable as a negative/measurement study; its central matched endpoint is credible. However, the lack of a controlled nearest-family baseline, limited matched seed evidence, and absence of natural-task validation for the proposed repair make it short of **4.0 / ACL main** at present.
