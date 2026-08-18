review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 4.0

# Summary

This paper studies a specific cross-query reuse point for long-context inference. CoMem writes each document chunk once through the first \(j\) transformer layers, stores one intermediate residual per token, retrieves a bounded number of chunks for each query, and resumes only layers \([j{:}L)\). Its main experiment is deliberately narrower than a system leaderboard: on Qwen3-8B, a same-pack, same-example, same-LoRA comparison between replay from \(j=0\) and replay from \(j=12\) reduces isolated model Read latency from 931.9 ms to 664.4 ms (\(1.403\times\)) while reducing a 15-cell RULER macro from 99.19 to 96.07 (paired difference 3.12, 95% CI \([2.36,3.93]\)). The paper also reports storage and write amortization, an equal-latency diagnostic against raw-text replay, a self-distillation LoRA, and a controlled diagnosis showing that adding lower-layer left context during Write repairs much of a synthetic multikey failure.

The paper is unusually explicit about negative results and measurement boundaries. It does not claim quality-preserving acceleration, does not claim to beat raw-text replay at equal latency, and repeatedly separates the incremental depth effect from the much larger effect of bounding the selected token set. This makes the central internal measurement credible and useful.

My main reservation is significance relative to the nearest reusable-context systems. The paper itself acknowledges that it has no same-backbone, same-hardware PIC, chunk-KV-repair, or learned modular-cache baseline. Consequently, the evidence establishes an informative internal design point, but not whether this point is competitive with the closest alternatives. In addition, the natural-task rows described as a matched \(j=0\) versus \(j=12\) comparison are not adapter-matched in the corresponding appendix ablation, and the promising Overlap-Write repair remains confined to one synthetic cohort. I therefore view the work as a solid Findings-level measurement/diagnosis paper rather than an ACL main-conference result in its present form.

# Claim and evidence map

| ID | Main claim | Evidence | Assessment |
|---|---|---|---|
| C1 | One residual per token at depth \(j\) permits direct suffix execution over a bounded retrieved pack. | Section 4, Figure 1, Eq. 1, Appendix Table 26, exact partition self-tests on Hy3. | Technically clear. The implementation semantics, mask, positions, and stored object are specified well. |
| C2 | The incremental \(j=0\rightarrow12\) depth reuse gives \(1.403\times\) faster isolated Read at a 3.12-point RULER cost. | Main Table 2; Appendix Tables 31–32; \(n=1{,}500\) paired examples; paired bootstrap and McNemar test; three latency processes. | This is the strongest and best-controlled result. Arithmetic checks: \(931.9/664.4=1.4026\) and \(99.19-96.07=3.12\). |
| C3 | A small upper-layer LoRA is necessary to make independently written residuals usable. | Tables 9, 11, and 12; same-\(j\) adapter on/off controls on RULER, LoCoMo/BABILong, and an HCache-style path. | Strong evidence that interface adaptation matters, although the precise top-64 distillation approximation is not fully characterized. |
| C4 | The 8 KiB/token residual store is smaller than full per-layer KV but requires repeated reuse to amortize Write. | Eq. 1; Tables 3, 27, and 33; 32k and limited 128k crossover measurements. | Correct under the stated bf16/full-KV accounting. The paper appropriately avoids equating this ratio with a matched competitive systems result. |
| C5 | At equal online latency, CoMem does not win the nine-cell aggregate; the conclusion depends on the replay selector. | Tables 4 and 8; Appendix B.4; stratified, hierarchical, pooled-IID, and leave-one-cell-out analyses. | Carefully reported negative result. BM25 replay is clearly ahead; the asymmetric frozen-BGE diagnostic is unresolved. |
| C6 | Missing lower-layer document context is a major failure source on a paired multikey diagnostic, and short left overlap repairs most of it. | Tables 5, 6, and 32; paired 8k/16k RULER multikey cohort. | Convincing within the tested synthetic cohort, but not yet a general or end-to-end deployment result. |
| C7 | Model-side Read remains bounded as the persistent store grows, while selector/index/store costs do not. | Tables 7, 25, 33, and 35; negative evidence-budget and aggregation tests. | Well scoped. The distinction between bounded model work and unbounded external-system work is explicit. |
| C8 | The split-forward mechanism is portable beyond one dense 8B model. | Appendix A.7, Tables 34–35, exact Hy3 partition checks, exploratory Qwen/MoE ports. | Supports implementation portability, not replicated quality or efficiency superiority. The paper states this limitation correctly. |
| C9 | Natural-task results provide scope checks for the representation. | Tables 18–24 on LongEval, LongBench, BABILong, and LoCoMo. | Useful breadth, but causal interpretation is weaker because of adapter mismatch in the displayed \(j=0\)/\(j=12\) natural-task rows, one principal training run, judge mutability, and incomplete contamination audits. |
| C10 | The contribution is a narrow depth-reuse measurement rather than the invention of reusable document states. | Related Work, Table 1, and Limitations. | This is an appropriately conservative novelty claim. Whether the remaining distinction is sufficiently novel relative to all concurrent work is externally Unverifiable in this audit. |

# Strengths

1. **The central causal comparison is unusually clean.**  
   The paper holds the selected examples, chunk IDs and order, mask, sink, and mounted LoRA fixed, changing only the replay start. Appendix Table 31 additionally reports process-level latency dispersion, and Table 32 uses a continuous-prefix \(h_{12}\) control that exactly reproduces full replay on the paired RULER cohort. This is substantially more informative than comparing a complete retrieval system against dense full-context inference and attributing the entire difference to the new memory representation.

2. **The paper is candid about quality loss and negative results.**  
   The sentence “This is not quality-preserving acceleration” is supported by the paired result rather than hidden in limitations. Likewise, the equal-latency section explicitly states that CoMem wins neither aggregate. This claim discipline materially increases trust in the measurements.

3. **Latency boundaries and storage assumptions are separated well.**  
   The manuscript distinguishes isolated selected-pack Read, store-ready online prefill, Write-inclusive execution, persistent fetch, selector/index construction, and decode. It also explains why the \(64.9\times\) dense-to-bounded number is not the incremental depth effect. Tables 3, 7, 8, 31, and 33 make these boundaries auditable.

4. **The statistical treatment is stronger than is typical for a systems-style NLP paper.**  
   The RULER comparison is paired; the paper reports both a paired bootstrap interval and exact McNemar counts. For the nine-cell equal-latency mixture, the authors correct the overly optimistic pooled-IID view with fixed-cell, hierarchical cell-resampling, and leave-one-cell-out analyses. For LoCoMo, they provide both item-level and conversation-cluster uncertainty and explicitly avoid treating 1,540 questions as independent.

5. **The diagnosis goes beyond a performance table.**  
   The continuous-prefix ceiling, context-by-position \(2\times2\) control, and Overlap-Write intervention form a coherent mechanistic story: suffix execution can be faithful when supplied compatible states, while independently written chunks omit useful lower-layer context. The interaction term is reported rather than forcing an additive explanation.

6. **Reproducibility reporting is detailed.**  
   The frozen source gives the backbone revision, layer split, rank and target modules, optimizer, data construction, number of steps/tokens, masks, positions, retrieval parameters, generation limits, scorers, seeds, hardware, hashes, and timing repetitions. The paper also reports what was not retained, including teacher top-64 mass, one \(j=12\) Write value, training peak memory, and total project compute.

7. **The limitations and ethics sections are substantive.**  
   The paper discusses model-version-specific stores, edit invalidation, storage volume, lack of tail-latency evaluation, mutable LLM judging, contamination, tensor inversion/membership risks, access control, deletion, and upstream licenses. These are directly relevant to the proposed persistent memory object.

# Major weaknesses

## W1. No matched comparison to the nearest reusable-context systems

**Location and exact quote.** Related Work states: “No artifact in this study supplies a same-backbone, same-hardware PIC, chunk-KV-repair, or learned modular-cache result under our pack, storage tier, and timing boundary.” The conclusion likewise says that the “next decisive step is a same-backbone, end-to-end comparison with PIC and learned modular-KV systems.”

**Problem.** CoMem is positioned against PIC, modular KV, chunk-KV repair, activation replay, and learned reusable document objects, but all empirical comparisons to these families are taxonomic or unmatched. The central \(j=0\) control answers whether prepaying 12 layers helps relative to recomputing the same retrieved text. It does not answer whether storing a single residual is a better quality/latency/storage point than storing or repairing selected KV states, reconstructing KV from residuals, or compiling modular KV objects.

**Impact.** This limits both novelty and practical significance. The measured endpoint is sound, but a reader cannot determine whether CoMem is an attractive reusable-context design or simply an inferior point relative to a nearby method that preserves more quality at comparable bytes or online work. This is the main reason I do not assign an ACL-main-level overall score.

**Minimal remedy.** Implement one closest feasible baseline on the same Qwen3-8B backbone and hardware—preferably a position-independent/chunk-KV method or KV-Direct-style residual reconstruction—and compare at matched persistent bytes and matched online TTFT. A compact Pareto table with quality, Write cost, fetched bytes, TTFT, decode, and break-even queries would be sufficient; reproducing every cited system is unnecessary.

**Severity: Major.**

## W2. The natural-task \(j=0\) versus \(j=12\) comparison is labeled “matched,” but the displayed adapter settings differ

**Location and exact quote.** Main Section 5.1 says: “The matched \(j=0\) and \(j=12\) rows score 97.2 versus 69.0 on LongEval, 41.59 versus 38.27 on LoCoMo, and 12.31 versus 12.15 on six LongBench QA datasets.” Table 18’s caption says: “Only the \(j=0\) and \(j=12\) rows form the paper’s matched depth comparison.” However, Appendix Table 9 explicitly lists “Matched raw-text replay” at \(j=0\) with “LoRA: no” and “CoMem distilled” at \(j=12\) with “LoRA: yes.”

**Problem.** The natural-task rows do not isolate replay depth if the \(j=0\) arm has no LoRA while the \(j=12\) arm uses the distilled upper-layer LoRA. This differs from the central RULER endpoint, for which the paper carefully mounts the same LoRA in both arms. The natural-task numbers may still be useful operating-point scope checks, but “matched depth comparison” is too strong for the protocol shown.

**Impact.** The discrepancy weakens interpretation of the LongEval, LongBench, BABILong, and LoCoMo gaps. It is unclear how much comes from the residual interface versus the adapter being present only in one arm. It also creates avoidable inconsistency in an otherwise careful paper.

**Minimal remedy.** Either (a) rerun the \(j=0\) natural-task arm with the identical mounted LoRA and all other settings fixed, or (b) relabel Tables 9 and 18 and the Section 5.1 sentence as an adapter-unmatched operating-point comparison. Option (b) is a textual fix, but option (a) would materially strengthen the paper.

**Severity: Major for the natural-task causal interpretation; it does not invalidate the paired RULER headline.**

## W3. Evidence for training robustness and clean natural-task generalization remains limited

**Location and exact quotes.** The Limitations section states: “The flagship is one batch-8 training run.” It also says the added runs use effective batch 3 and therefore are “not a clean estimate of training-seed variance.” Appendix A.5 adds that no multi-run aggregate was retained for “the exact 15-cell RULER-B, LoCoMo, or LongEval headlines.” Finally, the paper says: “Equivalent overlap audits were not completed for all natural benchmarks, including NarrativeQA.”

**Problem.** The exact headline adapter is represented by one principal run. The two auxiliary runs jointly change seed and effective batch and are evaluated on reduced supports. Natural-task evaluation also has two additional complications: the LoCoMo semantic judge is an undated mutable endpoint, and training-data overlap was only thoroughly audited for the removed InfiniteBench long-book comparison.

**Impact.** The main paired RULER difference is statistically precise conditional on the trained adapter, but uncertainty over adapter training and clean natural-task transfer is not quantified. This matters because the method depends strongly on self-distillation: same-\(j\) adapter ablations show very large gains.

**Minimal remedy.** Train two additional effective-batch-8 adapters with different seeds and evaluate at least the exact RULER-B cohort plus one natural benchmark. Complete a lightweight n-gram overlap audit for the selected natural benchmark. For LoCoMo, retain the current independent-judge audit but report the exact provider/snapshot if one becomes available.

**Severity: Major for generality, moderate for the internal RULER measurement.**

## W4. The proposed Overlap-Write repair is not yet shown to improve a natural or end-to-end frontier

**Location and exact quote.** The introduction states: “Because this repair is not evaluated on the full natural-task suite, we treat it as a localized diagnosis and engineering hypothesis.” The Limitations section adds that its wall-clock Write overhead “is not integrated into a repaired end-to-end frontier.”

**Problem.** Overlap-Write is the most actionable method improvement in the paper, raising the paired multikey macro from 92.5 to 98.5 at \(w=32\). However, it is tested only on one synthetic task family, and the reported FLOP ratio understates measured wall-clock overhead. The reader therefore cannot tell whether the repair improves the actual quality–latency–amortization trade-off on natural queries.

**Impact.** The diagnosis is convincing, but the paper stops immediately before demonstrating that the diagnosis yields a broadly useful system configuration. Without that result, the contribution remains primarily a measurement paper.

**Minimal remedy.** Evaluate \(w=0\) versus \(w=32\) on one natural benchmark where the baseline gap is material (LongEval or LoCoMo would suffice), and recompute Write-inclusive crossover for that repaired configuration. This is a focused experiment rather than a full benchmark sweep.

**Severity: Major for claims of a deployable repair; not a flaw in the bounded diagnostic as currently worded.**

# Minor weaknesses

## M1. The frozen-BGE equal-latency arm is intentionally asymmetric

**Location and quote.** Table 8 states: “Phase B intentionally uses frozen BGE for raw replay and iterative BM25 for CoMem, so it tests selector sensitivity rather than a same-pack depth effect.”

**Issue and impact.** This is valid as a sensitivity diagnostic, but it cannot identify whether BGE would help or hurt CoMem itself. The aggregate \(54.22\) versus \(53.22\) should therefore not be read as a representation comparison, and the current prose mostly—but not always immediately—requires the reader to track that asymmetry.

**Remedy.** Add BGE retrieval to both replay and CoMem, or place the asymmetric arm under a visibly labeled “selector-swap sensitivity” heading.

**Severity: Minor, because the paper explicitly disclaims a causal interpretation.**

## M2. The top-64 distillation objective lacks a basic approximation diagnostic

**Location and quote.** Section 4 says: “We did not retain the teacher mass captured by \(S_t\).”

**Issue and impact.** Both teacher and student distributions are renormalized over teacher top-64 logits, discarding all outside-support mass. Without retained teacher mass or a \(K\)-sweep, it is hard to know whether failures arise partly from the residual interface or from an aggressive approximate objective.

**Remedy.** Report mean/quantiles of teacher probability mass in the top 64 and, if feasible, a small \(K\in\{32,64,128\}\) ablation.

**Severity: Minor.**

## M3. Deployment measurements omit concurrency and tail latency

**Location and quote.** Limitations states: “The serving experiments report medians on single-query harnesses rather than concurrent p95 throughput or tail latency.”

**Issue and impact.** The storage-tier microbenchmark reports peak QPS, but the full selector/fetch/model path is not tested under concurrent load. Persistent-state traffic and scheduling could change the apparent crossover.

**Remedy.** Report p50/p95 TTFT and throughput at two or three concurrency levels for raw replay and CoMem.

**Severity: Minor for the current single-query claim; important for production deployment.**

## M4. Some appendix presentation is difficult to read

**Evidence.** Tables 8, 26–30, and several comparison tables use very small text, while PDF page 15 is mostly blank because of float placement. Figure 2 is also small relative to the amount of protocol qualification needed to interpret it.

**Impact.** The information is present, but auditability is reduced by dense typography and fragmented float placement.

**Remedy.** Reflow the appendix, enlarge the most important protocol tables, and use the blank page area more efficiently.

**Severity: Minor/editorial.**

## M5. Some baseline tables can still invite unintended ranking despite strong disclaimers

**Evidence.** Tables 18–24 place unmatched systems in common columns, sometimes bolding the best values, while captions explain differences in backbone, context extension, adapter training, and prompting.

**Impact.** Readers may overinterpret bold values as a fair leaderboard even though the paper says not to do so.

**Remedy.** Remove best-value bolding from unmatched rows or split matched internal controls from descriptive external references more sharply.

**Severity: Minor.**

# Questions for the authors

1. For Tables 9 and 18, can the authors confirm whether the natural-task \(j=0\) predictions were generated with the flagship LoRA mounted? The table currently says “LoRA: no,” which conflicts with the “matched depth comparison” wording.
2. Why was the same-selector frozen-BGE experiment not run for CoMem as well as raw replay? Was this prevented by missing residual-store indexing, or was the asymmetric selector swap intentional from the start?
3. How much teacher probability mass is typically retained by the top-64 support during distillation? Is the mass stable across query positions and training steps?
4. Does \(w=32\) Overlap-Write improve LongEval or LoCoMo, and what is the resulting measured Write-inclusive break-even?
5. Among PIC, chunk-KV repair, KV-Direct reconstruction, and learned modular KV, which system do the authors consider the single closest byte/latency competitor, and what implementation obstacle prevented a matched comparison?
6. Are the exact paired RULER examples and score vectors for \(j=0\), \(j=12\), and the continuous-prefix control included in the anonymous artifact, or only hashes/aggregate exports?

# Minimal experiments most likely to change my score

1. **P0: Same-backbone nearest-system baseline.** One matched PIC/chunk-KV/KV-reconstruction baseline with quality, stored/fetched bytes, Write, TTFT, decode, and crossover.
2. **P0: Resolve the natural-task adapter mismatch.** Rerun \(j=0\) with the identical LoRA on at least LongEval, LoCoMo, and the six LongBench QA datasets, or correct all “matched” wording.
3. **P1: Natural-task Overlap-Write validation.** \(w=0\) versus \(w=32\) on one natural benchmark plus repaired end-to-end crossover.
4. **P1: Clean seed replication.** Two additional effective-batch-8 seeds on exact RULER-B and one natural benchmark.
5. **P2: Distillation support diagnostic.** Teacher top-64 retained mass and a small support-size ablation.

# Novelty analysis

The locally available manuscript positions the closest families as:

- **ReadOnce Transformers / Embedding Recycling:** prior reuse of intermediate text representations with downstream adaptation.
- **HCache / KV-Direct:** activation checkpointing or reconstruction of layer-wise KV from residual state.
- **CacheBlend, TurboRAG, Cache-Craft, EPIC, MEPIC, and APE:** reusable chunk/prefix KV with composition, position, or boundary repair.
- **KV Packet, Cartridges, Cartridges at Scale, and SemPIC:** learned reusable per-document KV objects.
- **GemFilter, ILRe, and REFORM:** intermediate-layer selection/retrieval or gather-and-recompute approaches.

Relative to this set, the paper’s narrow distinction is: persist exactly one depth-\(j\) residual per token, retrieve a bounded chunk set, directly run the suffix layers, and measure a same-pack \(j=0\) endpoint with explicit storage/Write/Read accounting. That combination appears meaningfully distinct at the level claimed by the authors, but it is an incremental systems/design-point contribution rather than a new general memory paradigm.

The NORMAL protocol requests 3–5 external novelty searches and a three-month check. Per the instruction to stop further retrieval, these searches were not executed. The following external checks are therefore **Unverifiable** in this review:

1. `"persistent intermediate residual" transformer cross-query cache`
2. `"single residual per token" reusable document cache suffix layers`
3. `ReadOnce HCache KV-Direct intermediate activation reuse long-context`
4. `position independent caching residual stream modular KV 2026`
5. `CoMem long context residual memory`

The local bibliography contains very recent/concurrent 2026 items, including KV Packet, Cartridges at Scale, and SemPIC. Whether any work posted within the three months preceding August 4, 2026 more closely anticipates CoMem is **Unverifiable** without network search. I therefore do not penalize the paper for a specific uncited contemporaneous work, but my novelty confidence is lower than my technical-review confidence.

# Citation audit

## Structural audit of `main.bbl`

- `main.bbl` contains 43 entries.
- All 43 `main.bbl` entries are cited by the frozen manuscript.
- No manuscript citation key is missing from `qcmem.bib` or `main.bbl`.
- No undefined `\ref` target or duplicated label was found.
- `qcmem.bib` contains 26 unused entries that are correctly absent from `main.bbl`; this is source hygiene rather than a rendered-paper problem.

The 43 rendered bibliography keys checked were: `cachecraft`, `longbench`, `pyramidkv`, `kvpacket`, `cartridgesbase`, `hcache`, `llama3`, `cartridges`, `distillation`, `ruler`, `lora`, `epic`, `ragcache`, `babilong`, `rag`, `longchat`, `snapkv`, `ilre`, `readonce`, `minicache`, `turborag`, `locomo`, `xccache`, `kvdirect`, `pg19`, `bm25`, `embeddingrecycling`, `gemfilter`, `reform`, `lloco`, `hunyuan`, `fusionrag`, `mepic`, `longmem`, `memoryllm`, `infllm`, `streamingllm`, `sempic`, `xu2024retrievallong`, `qwen3`, `ape`, `cacheblend`, and `h2o`.

External bibliographic identity, publication status, page ranges, DOI correctness, and paper contents are **Unverifiable** here because network verification was not performed.

## Citation–claim spot checks

| Manuscript claim | Citation(s) | Local match assessment | External status |
|---|---|---|---|
| Raw-text retrieval bounds the selected online tokens but recomputes model layers. | RAG; Retrieval Meets Long Context LLMs | The titles and the manuscript’s use are directionally appropriate. | Unverifiable |
| CacheBlend/TurboRAG/RAGCache/Cache-Craft reuse retrieved chunk KV and address composition. | Those four works | Appropriate to the cited titles and taxonomy; not independently checked. | Unverifiable |
| EPIC/MEPIC provide position-independent caching; APE uses parallel context encoding. | EPIC, MEPIC, APE | Appropriate to titles and local bibliography metadata. | Unverifiable |
| ReadOnce and Embedding Recycling cache intermediate representations. | ReadOnce, Embedding Recycling | Strong title-level match. | Unverifiable |
| HCache restores activation state and KV-Direct reconstructs KV from residuals. | HCache, KV-Direct | Strong title-level match for HCache; KV-Direct interpretation is plausible from its title and paper description. | Unverifiable |
| GemFilter selects using early layers without persistent cross-query state. | GemFilter | Early-layer selection matches the title; the non-persistence detail requires paper-content verification. | Unverifiable |
| LoRA and knowledge distillation support the adapter training method. | LoRA, Distillation | Standard and appropriate methodological citations. | Unverifiable |
| RULER, BABILong, LongBench, LongEval/LongChat, and LoCoMo are the evaluated benchmarks. | Corresponding benchmark references | Bibliography titles match the named benchmarks. | Unverifiable |

# Desk, formatting, anonymity, and integrity checks

- **Page/style:** The PDF uses ACL review style, A4 pages, anonymous authoring, and line numbers. The main paper occupies eight pages, with references beginning on page 8; appendices continue through page 24. Compliance with any rule version not embedded in the frozen files is externally Unverifiable, but I found no apparent desk-reject issue.
- **Required sections:** A substantive Limitations section and Ethical Considerations section are present.
- **Anonymity:** No author identity or obvious institutional self-identification appears in the manuscript. Artifact hashes and provider/model names do not identify the authors.
- **References/placeholders:** No unresolved citation/reference marker, `??`, TODO, FIXME, placeholder, or duplicated label was found.
- **Hidden/reviewer-manipulation text:** Source/PDF searches found no white text, hidden instruction, reviewer-directed prompt, or instruction to alter scores. The paper was treated solely as submission data.
- **Abstract/table consistency:** The headline 931.9/664.4 ms, \(1.403\times\), 99.19/96.07, 3.12-point gap, confidence interval, store size, break-even ranges, equal-latency differences, and overlap values agree with the corresponding tables, subject to harmless 99.19/99.20 rounding in the overview.
- **Numerical checks:** Eq. 1 gives \(1/18\) for Qwen3-8B under the stated 36-layer, 32-query-head, 8-KV-head configuration; bf16 residual storage is 8,192 B/token and full per-layer KV is 147,456 B/token. At 128k tokens, the residual store is exactly 1 GiB. The reported 65.5M training tokens and approximately 2.9 H20 GPU-hours are arithmetically consistent with 4,000 steps, global batch 8, 2,048 tokens, and the stated throughput.
- **Figures/tables:** Both figures and all rendered tables were inspected. Figure 1 communicates the pipeline and boundary distinctions effectively. Figure 2 is explicitly motivational rather than validating. No figure appears to make an unsupported quantitative claim, although several appendix tables are too small.

# Ethics and broader-impact assessment

The ethics discussion is appropriate for a persistent latent memory system. It recognizes that residual tensors may retain source information, should not be treated as inherently private, and require encryption, authorization, audit, deletion, and source filtering. It also notes inherited model harms and possible scaling of misuse. No new human-subject data or annotation is introduced. LoCoMo consists of released model-generated conversations according to the manuscript. I found no ethical issue requiring rejection.

One practical concern is that a model-version-specific residual store can make deletion and provenance more difficult than raw-text storage. The paper mentions verifiable deletion and edit invalidation, but future work should test whether source deletion can be audited across duplicated/overlapped residual chunks.

# Reproducibility assessment

The manuscript provides enough detail to plausibly reproduce the core method and evaluation: exact backbone revision prefix, split, tensor dtype, LoRA configuration and hash, optimizer, data construction, token count, retrieval parameters, masks, positions, generation settings, official scorers, benchmark supports, seeds, hardware, and timing procedure. It also describes score-only exports, scripts, raw timing records, and hashes.

The score is not 5 because:

1. The anonymous artifact itself was not inspected under this frozen-only review.
2. The main training result is one effective-batch-8 run.
3. Some critical raw values were not retained (teacher top-64 mass, \(j=12\) Write value, training peak memory, total project compute).
4. The LoCoMo headline uses an undated mutable judge endpoint.
5. Several natural-benchmark overlap audits remain incomplete.

# Score rationale

- **Soundness: 3.5/5.0.** The main same-pack RULER measurement is carefully controlled and the statistical/system boundaries are strong. Soundness is reduced by the adapter mismatch in the natural-task “matched” rows and limited clean replication/generalization evidence.
- **Excitement: 3.0/5.0.** Treating depth as a reusable axis and measuring it honestly is useful, but the method is technically simple and close to several intermediate-state/modular-cache lines of work. The absence of a matched nearest-system baseline limits the likely impact.
- **Overall: 3.0/5.0.** I recommend Findings-level acceptance. The paper offers a credible negative/measurement contribution and a useful bounded diagnosis, but it does not yet establish an ACL-main-level advance over the closest reusable-context systems.
- **Confidence: 4.5/5.0.** I read the full frozen PDF and source, including two appendix passes, inspected every rendered figure/table, checked cross-references and headline arithmetic, and audited all rendered bibliography entries structurally. Confidence is below 5 only because external novelty and citation verification were not performed.
- **Reproducibility: 4.0/5.0.** Excellent reporting, with the caveats listed above.

# Review-process self-check

- Read only the frozen `v6_20260804_014520` PDF/source and the NORMAL template.
- Did not read any other review, review score/history content, TODO, status, current, or calibration file.
- Completed a first argument/evidence pass and a second appendix/table/consistency pass.
- Inspected the rendered PDF pages, including all figures, tables, references, and appendices.
- Mechanically checked exact quotations used in weaknesses against the frozen source.
- Mechanically checked citation-key coverage, labels/references, placeholders, headline arithmetic, storage arithmetic, and training-compute arithmetic.
- Found no reviewer-manipulation or hidden-instruction text.
- External citation verification, publication-status verification, novelty searches, and the three-month prior-art check are explicitly marked **Unverifiable** and did not affect the review as if completed.
