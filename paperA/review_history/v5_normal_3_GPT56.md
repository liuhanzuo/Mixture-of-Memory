---
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 3.5
---

# Summary

This paper studies a narrow but useful operating point for repeated queries over a stable corpus. CoMem writes each document chunk once through the lower \(j\) transformer layers, stores one intermediate residual vector per token, retrieves a bounded set of chunks, and runs only layers \([j{:}L)\) at query time. The principal experiment is deliberately internal rather than a cross-system leaderboard: on Qwen3-8B, matched raw-token replay at \(j=0\) and residual replay at \(j=12\) use the same examples, selected chunk IDs/order, causal pack, sink, mask, and LoRA. The latter reduces isolated selected-pack model Read from 931.9 ms to 664.4 ms (\(1.403\times\)) while reducing a paired 15-cell RULER macro from 99.19 to 96.07.

The paper also reports (i) a self-distillation LoRA needed to make independently written residuals readable, (ii) storage and repeated-query break-even measurements, (iii) selector-sensitive equal-latency diagnostics, and (iv) a controlled diagnosis showing that lower-layer document context, partially approximated by overlap writing, repairs much of one synthetic multikey failure. The presentation is unusually explicit about what is and is not causal: bounded selection is separated from depth reuse; the continuous-prefix control is called a non-deployable fidelity ceiling; the multi-depth curve is not represented as compute matched; and the authors do not claim superiority over raw-text RAG, PIC, or learned modular caches.

My overall assessment is **Findings-level / borderline for ACL main**. The matched \(j=0\) versus \(j=12\) experiment is careful and informative, and the negative/deployment-boundary results are valuable. However, the work still lacks the decisive same-backbone comparison against the nearest reusable-context systems, its main statistical headline is effectively based on one adapter run, and the equal-latency/natural-task evidence is not yet sufficient for a broader systems conclusion.

# Claim–evidence map

| Claim | Evidence in the frozen paper | Assessment |
|---|---|---|
| **C1. A persistent depth-\(j\) residual can support bounded query-time suffix execution.** | Method §4; Figure 1; Appendix Tables 24, 34. Qwen read length stays about 6.2–6.5k while the synthetic store reaches 4M tokens; Hy3 read length stays about 4.3–4.6k through 256k. | Supported as a model-side bounded-read property. The paper correctly notes that selector/index/store cost is not bounded. |
| **C2. Reusing the first 12 layers produces a measured speed/quality trade-off under matched evidence.** | Main Table 2; Appendix Table 30; 1,500 paired RULER-B examples; Read 931.9→664.4 ms, quality 99.19→96.07, paired CI \([2.36,3.93]\), McNemar \(p=8.79\times10^{-24}\). | Strongest claim and well supported. It is isolated model Read, not end-to-end latency or quality-preserving acceleration. |
| **C3. Self-distillation makes the independently written residual interface usable.** | §4 objective; Appendix Tables 9, 11, 12, 33. Same-\(j=12\) RULER and natural-task controls show large gains; Hy3 shows lower LM tax and higher top-1 agreement. | Supported for the tested adapters/tasks. Training robustness is limited because the flagship is one effective-batch-8 run and the two other runs change effective batch. |
| **C4. The residual store pays off only after repeated reuse.** | Main Table 3; explicit \(Q^\star\) equation; three-process medians. At 32k, \(Q^\star\) is roughly 5.5–10.9 for \(G\le128\), depending on placement; at 128k only \(G=1\) is retained, 25.8–27.6. | Supported for the stated single-query harness and hardware/storage boundaries. Not a production throughput/tail-latency result. |
| **C5. Equal-latency quality conclusions depend on the selector and CoMem wins neither tested aggregate.** | Main Table 4 and Appendix Table 8: nine equal-weight cells, BM25 replay 64.78 vs CoMem 53.22; frozen-BGE replay 54.22 vs 53.22. | Point estimates support the qualitative conclusion. Inferential strength is weaker than the displayed IID-example CIs suggest because the aggregate mixes tasks and one LoCoMo conversation. |
| **C6. Missing lower-layer document context is a major cause of one multikey gap, and overlap writing is a local repair.** | Main Tables 5–6; Appendix Table 31. Document-context states reach 100.0 vs 92.5 for chunk-local; \(w=32\) reaches 98.5. | Supported on the paired 8k/16k synthetic multikey cohort only. The paper appropriately does not generalize this to natural tasks. |
| **C7. CoMem is portable beyond one dense 8B model.** | Appendix Hy3 exact partition test, 16-document readout study, and \(n=50\) RULER cells; prose mentions other Qwen scales. | Evidence supports implementation portability, not a replicated quality/efficiency frontier. |

# Strengths

1. **The central comparison is unusually well controlled.**  
   Location: §5.1, Main Table 2, Appendix Table 30.  
   Anchor: “*Raw-text replay (\(j=0\)) and CoMem receive the same selected chunks in the same order and use the same sink, mask, examples, and LoRA.*”  
   The same-pack, same-adapter endpoint removes several confounds that commonly obscure systems papers. The paper also reports paired uncertainty and an exact McNemar test.

2. **The paper is candid about quality loss and scope.**  
   It explicitly says “*CoMem is therefore not quality preserving*,” separates the \(1.403\times\) depth effect from the \(64.9\times\) dense-to-bounded operating point, and states that CoMem wins neither equal-latency aggregate. This materially improves the reliability of the contribution.

3. **The systems boundary is documented rather than hidden.**  
   Storage is given in bytes/token; Write, fetch, Read, decode, selection, and index construction are separated; GPU/CPU/NVMe/network tiers are measured; and the 128k break-even grid is not extrapolated beyond the retained \(G=1\) measurement.

4. **The diagnosis is more informative than a generic ablation.**  
   The continuous-prefix ceiling and the context-scope × position control show that upper-layer continuation itself can be exact and that chunk-local lower-layer representations cause an important tested failure. The overlap intervention then converts the diagnosis into a plausible local engineering repair.

5. **The appendix is substantial and generally reproducibility-oriented.**  
   It includes architecture revision, masks, positions, optimizer, LoRA modules and hash, benchmark support, generation/scoring rules, timing protocol, seeds, sample counts, statistical units, data licenses, contamination caveats, and compute omissions.

6. **The manuscript is well structured and visually readable.**  
   Figure 1 clearly separates Write/Select/Read and the \(j=0\) control; Figure 2 is explicitly framed as motivation rather than validation. Across the 36 labeled tables, captions usually state the cohort, sample size, timing boundary, and key non-comparability caveat.

# Major weaknesses

## W1. The nearest competitive question remains unanswered

- **Location / short quote:** Related Work §2 and Limitations, p. 7: “*The paper does not provide a same-backbone, same-hardware implementation of the closest PIC, chunk-KV repair, or learned modular-cache systems.*”
- **Problem:** The paper's object is distinct—one residual per token at one split—but its intended workload overlaps directly with CacheBlend/TurboRAG/EPIC/MEPIC/KV Packet/Cartridges-like reusable document state. A matched \(j=0\) endpoint establishes the incremental effect of skipping lower layers, but it does not establish whether the proposed storage/quality/latency point is useful relative to the strongest alternative reusable objects.
- **Affected claim / review norm:** C1/C4 and the practical significance/novelty expected of an ACL systems contribution.
- **Importance:** **High.** This is the main reason I do not score the current paper at ACL-main level. The paper itself identifies this as “the next decisive step.”
- **Remedy:** Add at least one same-backbone, same-hardware nearest baseline under the same retrieved chunks, storage tier, and timing boundary. A minimal sufficient experiment would compare CoMem against (a) an independently encoded per-layer KV/PIC path with a documented repair strategy and (b) a residual-to-KV/recompute path such as KV-Direct, reporting quality, persistent bytes, Write, fetch, TTFT/Read, and break-even. If a full published system cannot be reproduced, a carefully defined upper/lower-bound implementation is still more informative than taxonomy.
- **Severity:** **Major.**

## W2. Main adapter-dependent conclusions lack clean run-to-run uncertainty

- **Location / short quote:** Limitations, p. 7: “*The flagship is one batch-8 training run.*” Appendix Table 28 labels the available evidence “*Seed-plus-effective-batch robustness, not clean seed variance.*”
- **Problem:** The main \(j=12\) quality, natural-task results, and overlap diagnosis all depend on a learned 58.2M-parameter adapter. The two additional adapters change effective batch from 8 to 3 and use reduced-support evaluation; no multi-run aggregate is retained for the exact 15-cell RULER-B, LongEval, or LoCoMo headlines.
- **Affected claim / review norm:** C2/C3/C6; reliability and statistical evidence for a learned-method paper.
- **Importance:** **High.** The paired test precisely estimates evaluation-example uncertainty conditional on one adapter, but does not measure training-run uncertainty. A different adapter could change the 3.12-point quality cost or the apparent success of the repair.
- **Remedy:** Train at least two additional adapters with the same effective batch, schedule, data order policy, and evaluation support. Report run-level mean/SD or confidence intervals for the exact RULER-B endpoint and at least one natural benchmark. For the overlap result, either repeat across adapters or state explicitly that its CI is conditional on the single trained adapter.
- **Severity:** **Major.**

## W3. The equal-latency confidence intervals do not match the heterogeneous aggregate's dependence structure

- **Location / short quote:** Appendix Table 8: “*The percentile bootstrap pools and resamples all 900 paired example differences IID ...; it does not resample task cells or LoCoMo conversations.*” The cohort includes “*first 100 LoCoMo items, which are all from conversation 0*.”
- **Problem:** The nine-cell aggregate gives each task/length cell equal weight, yet the CI pools all examples IID. This fails to reflect uncertainty over the task mixture/cell choice and ignores within-conversation dependence for LoCoMo. The BGE interval crossing zero is robustly a non-win, but the exact width and the strength of the BM25 margin are overstated for a deployment-level conclusion.
- **Affected claim / review norm:** C5; correct statistical unit and uncertainty for aggregate benchmark claims.
- **Importance:** **Medium–High.** This does not reverse the displayed point estimates, but it weakens the claim that the reported intervals quantify the decision boundary.
- **Remedy:** Use a hierarchical paired bootstrap: resample cells (or report leave-one-cell-out sensitivity), then examples within each selected cell, and conversations rather than questions for LoCoMo. Also report all nine cell-level differences and repeat LoCoMo on more than one conversation or remove it from the inferential aggregate.
- **Severity:** **Major.**

## W4. Natural-task and end-to-end evidence is too weak to validate the main deployment hypothesis

- **Location / short quote:** §5.1: natural results are “*scope checks rather than contamination-free generalization*”; §5.3: overlap is “*evaluated only on this paired synthetic cohort*”; Limitations: serving experiments are “*medians on single-query harnesses rather than concurrent p95 throughput or tail latency.*”
- **Problem:** The most convincing quality/latency claim is synthetic RULER; natural-task evidence is either very low-scoring, potentially contaminated, judged by a mutable model endpoint, or not paired with the proposed contextual-write repair. Meanwhile, the reuse story is evaluated as cumulative single-query latency rather than concurrent serving, and the repaired overlap writer is not integrated into the break-even frontier.
- **Affected claim / review norm:** C4/C6 and external validity of an NLP serving paper.
- **Importance:** **Medium–High.** The paper responsibly narrows its claims, so this is not a soundness failure; it limits impact and excitement.
- **Remedy:** A minimal sufficient extension is: choose one natural repeated-query workload with stable deterministic scoring; run matched \(j=0\), default CoMem, and \(w=32\) CoMem; report retrieval-hit-conditioned quality, persistent bytes, Write/fetch/TTFT/decode, and break-even under at least modest concurrency with p50/p95. This would test whether the diagnosed repair changes an actual deployment frontier.
- **Severity:** **Major.**

# Minor weaknesses

## m1. The distillation objective discards unmeasured teacher mass

- **Location / quote:** §4: “*We did not retain the teacher mass captured by \(S_t\).*”
- **Problem:** Both distributions are renormalized over teacher top-64 logits, so the objective cannot penalize probability moved outside that support, and the approximation error cannot be audited.
- **Affected claim / norm:** C3; complete specification and interpretation of the learning objective.
- **Importance:** Medium.
- **Remedy:** Report teacher top-64 captured mass and an ablation over support size or full-vocabulary/chunked KL on a manageable subset.
- **Severity:** Minor.

## m2. The multi-depth curve is useful but not a clean depth frontier

- **Location / quote:** §3: “*the retained \(j=12\) Write value is missing*”; Appendix Table 10: adapter spans and parameter counts differ.
- **Problem:** The curve mixes depth, trainable parameter count, and independently trained adapters, while one Write datum is absent.
- **Affected claim / norm:** Interpretation of C2 and completeness of the compute trade-off.
- **Importance:** Medium.
- **Remedy:** Retain matched Write measurements for every split and add either parameter-matched adapters or a fixed adapter budget across depths.
- **Severity:** Minor.

## m3. Citation metadata has several correctness/completeness issues

- **Location / quote:** `main.bbl`, e.g. TurboRAG is printed as pp. 6588–6601.
- **Problem:** Crossref returned pp. **6599–6612** for DOI `10.18653/v1/2025.emnlp-main.334`. ReadOnce and Embedding Recycling omit page ranges although authoritative metadata provides them. Cache-Craft omits volume/pages. Cartridges is dated 2026 in the bibliography although its arXiv record is 2025; if this denotes an ICLR 2026 proceeding, the unusual volume/page metadata should be checked carefully. RAGCache's online/issue year is also ambiguous across metadata.
- **Affected claim / norm:** Bibliographic accuracy.
- **Importance:** Low–Medium.
- **Remedy:** Regenerate and manually audit the bibliography from DOI/Anthology/proceedings records, preserving online-first versus issue year consistently.
- **Severity:** Minor.

## m4. The LongEval citation is indirect

- **Location / quote:** §5 cites “LongEval~\(\citep{longchat}\),” whose bibliography entry is the LMSYS LongChat blog.
- **Problem:** The paper does not give a dedicated versioned benchmark/repository citation for the exact line-retrieval data and scorer used.
- **Affected claim / norm:** Benchmark provenance and reproducibility.
- **Importance:** Low–Medium.
- **Remedy:** Cite the exact LongEval artifact/commit or archived release in addition to the blog.
- **Severity:** Minor.

## m5. Some compute and memory records are incomplete

- **Location / quote:** Appendix Table 26: “*training peak memory was not recorded*”; compute accounting: “*Total GPU-hours across preliminary probes, failed runs, baseline generation, and all ablations were not consistently logged.*”
- **Problem:** The final run is documented, but total research compute and training memory are not.
- **Affected claim / norm:** Reproducibility and responsible compute reporting.
- **Importance:** Low.
- **Remedy:** Include peak training memory and a best-effort ledger of all major experiment families in the artifact.
- **Severity:** Minor.

## m6. The main paper is dense and depends heavily on a very long appendix

- **Location / quote:** Appendix opening calls the body an “*eight-page main paper*,” followed by 13 pages of appendices.
- **Problem:** The most important protocol qualifications, cell definitions, and statistics are sometimes several pages away from the headline. This is compliant-looking, but increases the risk that readers overinterpret the central numbers.
- **Affected claim / norm:** Clarity and self-contained presentation.
- **Importance:** Low.
- **Remedy:** Move the exact statistical unit and a compact timing-boundary row into Main Tables 2–4; shorten secondary baseline grids.
- **Severity:** Minor.

# Questions for the authors

1. For the same-adapter \(j=0\) baseline, are all 168 LoRA modules active during full replay exactly as in the \(j=12\) arm? Appendix §A.5 says yes; please confirm that no lower-layer adapter module exists and that the same checkpoint is loaded identically in both arms.
2. How sensitive is the 3.12-point RULER gap to adapter training run? Can you provide the exact RULER-B endpoint for the two batch-3 adapters, even if only as a non-headline robustness result?
3. For the equal-latency aggregate, what are the nine individual CoMem-minus-replay differences for BM25 and BGE? Does the conclusion survive leave-one-cell-out aggregation and removal of the single-conversation LoCoMo cell?
4. Why was \(j=12\) chosen as the flagship rather than \(j=6\) or \(j=9\), which appear to lose less quality? Was the choice preregistered, or selected after observing the deployment curve?
5. Can the authors provide one same-pack comparison to a per-layer KV/PIC baseline with persistent bytes and TTFT measured on the same H20? This seems more decision-relevant than several unmatched external quality rows.
6. Does overlap writing improve any natural task or the full 15-cell RULER-B cohort, and what is its measured end-to-end Write time and revised \(Q^\star\)?
7. What fraction of teacher probability mass is typically captured by top-64 logits? If this is unavailable for the current run, can it be measured post hoc on a held-out subset?

# Suggestions

1. Make the paper's contribution label explicit in the title/abstract: this is primarily a **measurement and diagnosis** paper about a residual-state operating point, not a generally superior memory system.
2. Replace the heterogeneous IID bootstrap in the equal-latency section with hierarchical and leave-one-cell-out analyses.
3. Add one matched reusable-KV/PIC baseline before expanding the benchmark suite further.
4. Evaluate \(w=32\) on one deterministic natural workload and include its Write overhead in the repeated-query crossover.
5. Report clean same-batch multi-run results for the exact headline cohort.
6. Add top-64 mass coverage and support-size sensitivity for distillation.
7. Correct and complete bibliography metadata, especially TurboRAG pages and the venue/year representation of online-first or future-volume papers.
8. Archive an exact LongEval version/commit and a dated identifier for any model-based judge when the provider permits it.

# Novelty analysis

I used a **2026-05-04 cutoff** for novelty credit. Search was stopped at the user's request; any item that could not be confirmed within the available audit is marked **Unverifiable**.

## Searches performed

1. Persistent/intermediate residual memory for long-context transformers.
2. Residual-stream caching and reusable document state.
3. Reusable intermediate text representations.
4. Transformer depth reuse / layer-wise caching.
5. Position-independent and modular document KV caching.

## Closest prior work before the cutoff

| Work | Verified date / relation | Difference from CoMem |
|---|---|---|
| **ReadOnce Transformers** | 2020 arXiv / ACL 2021. Builds reusable, task-independent compressed text representations and adapts downstream computation. | Strong conceptual precedent for reusable intermediate representations. CoMem's narrower distinction is a decoder-only, per-token residual at a tunable native depth, bounded retrieval, and a matched \(j=0\) suffix-replay measurement. |
| **Embedding Recycling** | Findings EACL 2023. Reuses cached intermediate representations. | Similar precedent at the representation-reuse level; CoMem emphasizes repeated-query long-context serving and explicit Write/Read/storage boundaries. |
| **HCache** | EuroSys 2025. Checkpoints activations for state restoration. | Closest activation-replay systems precedent, but aimed at restoration/eviction rather than a persistent document store selected across queries. |
| **EPIC / APE / MEPIC / FusionRAG / CacheBlend / TurboRAG / Cache-Craft** | All available before 2026-05-04. Reuse or compose document/chunk KV while addressing positions, boundaries, or cross-chunk context. | These are closer in workload and may be stronger systems alternatives. CoMem stores one residual at one depth rather than per-layer KV and pays suffix recomputation, trading storage for online compute. |
| **KV-Direct** | arXiv 2026-03-20. Establishes residual-stream sufficiency and reconstructs/recomputes KV from residual checkpoints. | Very close representational precedent. CoMem differs by writing independently chunked document residuals for cross-query retrieval and directly continuing the suffix with a learned adapter, rather than primarily treating residuals as a lossless substitute for KV state. |
| **KV Packet** | arXiv 2026-04-14. Context-independent reusable KV packets with learned adapters and no document recomputation. | Close learned modular-object competitor. CoMem uses much smaller one-depth residual objects but must execute upper layers; a matched comparison is missing. |
| **Cartridges** | arXiv 2025-06-06; proceeding metadata claims ICLR 2026. Distills reusable per-corpus KV representations. | Similar amortized repeated-query premise and learned reusable state, but uses trained KV objects rather than one residual/token and synthetic self-study. |

## Three-month rule

- **Included for novelty:** works first public by **2026-05-04**, including KV-Direct (2026-03-20) and KV Packet (2026-04-14).
- **Not used to reduce novelty:** Cartridges at Scale (2026-06-03), SemPIC (2026-07-28), and other post-cutoff works found in search.
- The paper nevertheless cites some post-cutoff work for positioning; that is useful context but should not be treated as prior art that independently defeats novelty under this rule.

## Novelty judgment

The broad idea of caching reusable intermediate document representations is not new, and reusable document KV systems already establish the workload. The paper's defensible novelty is the **specific measured design point**: one persistent residual/token at a chosen split, direct suffix execution on a bounded selected pack, a same-adapter \(j=0\) endpoint, explicit storage/amortization accounting, and a context/position diagnosis. I view this as **incremental-to-moderate methodological novelty**, with value coming more from the controlled measurement and negative boundary findings than from a fundamentally new memory paradigm.

# Citation audit

## `main.bbl` entry-by-entry audit

All 43 `main.bbl` keys are cited at least once; there are no cited keys missing from `main.bbl`, no duplicate `\bibitem` keys, and no unresolved source cross-references.

Legend: **Verified** = title/identifier matched an authoritative DOI/arXiv/official page during this audit; **Partial** = identity verified but one or more venue/year/page fields need correction or were not fully confirmed; **Unverifiable** = the remaining metadata could not be independently checked before retrieval stopped.

| Key | Status | Audit note |
|---|---|---|
| cachecraft | Partial | DOI/title verified; volume/pages omitted from rendered entry. |
| longbench | Verified | ACL Anthology DOI/title/year/pages match. |
| pyramidkv | Partial | arXiv identity verified; venue/year should be checked against final COLM record. |
| kvpacket | Verified | arXiv 2604.13226 title/date match. |
| cartridgesbase | Partial | arXiv title verified; rendered 2026 ICLR volume/pages metadata should be checked carefully against the final proceedings record. |
| hcache | Verified | DOI/title/year/pages match. |
| llama3 | Verified | arXiv 2407.21783 matches. |
| cartridges | Verified | arXiv 2606.04557 matches; post-cutoff for novelty. |
| distillation | Verified | arXiv 1503.02531 matches. |
| ruler | Verified | arXiv/title match; venue representation plausible. |
| lora | Partial | Title/arXiv match; arXiv first year is 2021 while rendered entry uses ICLR 2022, which is acceptable if venue year is intended. |
| epic | Verified | arXiv/title and ICML 2025 identity match. |
| ragcache | Partial | DOI/title/journal/pages match; online-first 2025 versus issue/print 2026 requires consistent year policy. |
| babilong | Verified | DOI/title/year/pages match. |
| rag | Partial | Canonical paper identity is clear; rendered entry lacks DOI/pages and automated title-only search was ambiguous. |
| longchat | Verified | Official LMSYS blog title/page resolves. |
| snapkv | Verified | DOI/title/year/pages match. |
| ilre | Verified | arXiv 2508.17892 matches. |
| readonce | Partial | DOI/title/year verified; pages 7129–7141 omitted. |
| minicache | Verified | DOI/title/year/pages match. |
| turborag | **Partial—correction needed** | DOI/title/year verified; authoritative pages are 6599–6612, not 6588–6601. |
| locomo | Verified | ACL Anthology DOI/title/year/pages match. |
| xccache | Partial | DOI/title/year verified; pages 15284–15302 omitted. |
| kvdirect | Verified | arXiv 2603.19664 matches. |
| pg19 | Verified | Compressive Transformers/arXiv 1911.05507 is the source of PG-19. |
| bm25 | Verified | DOI/title/journal/pages match. |
| embeddingrecycling | Partial | DOI/title/year verified; pages 1933–1953 omitted. |
| gemfilter | Verified | ACL 2026 DOI/title/pages match. |
| reform | Verified | arXiv 2506.01215 matches. |
| lloco | Verified | ACL DOI/title/year/pages match. |
| hunyuan | Verified | Official Hugging Face model page resolves; this is a model-page citation rather than archival paper metadata. |
| fusionrag | Verified | arXiv 2601.12904 matches. |
| mepic | Verified | arXiv 2512.16822 matches. |
| longmem | Partial | Title/NeurIPS 2023 identity verified; volume/pages omitted. |
| memoryllm | Verified | arXiv/title and ICML record match. |
| infllm | Verified | DOI/title/year/pages match. |
| streamingllm | Verified | arXiv/title and ICLR identity match. |
| sempic | Verified | arXiv 2607.28069 matches; post-cutoff for novelty. |
| xu2024retrievallong | Partial | Well-known ICLR 2024 paper and title/authors are plausible, but the exact archival identifier was not independently confirmed in the stopped audit. |
| qwen3 | Verified | arXiv 2505.09388 matches. |
| ape | Verified | arXiv 2502.05431 matches. |
| cacheblend | Verified | DOI/title/year/pages match. |
| h2o | Partial | NeurIPS 2023 title identity and pages were found, but rendered entry is sparse. |

## Citation–claim match checks

| Paper text claim | Cited work(s) | Match |
|---|---|---|
| Raw-text retrieval bounds the selected token set but recomputes the model. | RAG; Retrieval Meets Long Context LLMs | **Reasonable.** These establish retrieval-then-generation over selected text; the exact “all layers” statement is architectural inference. |
| Modular/PIC systems reuse precomputed document KV and repair context/position dependencies. | CacheBlend, TurboRAG, EPIC, Cache-Craft, KV Packet | **Strong match.** Verified abstracts/metadata describe reusable KV plus fusion, recomputation, position, or adapter repair. |
| ReadOnce and Embedding Recycling cache intermediate text representations and adapt later computation. | ReadOnce; Embedding Recycling | **Strong match.** This is the closest conceptual precedent and is accurately described. |
| KV-Direct reconstructs layer-wise KV from residuals. | KV-Direct | **Strong match.** The abstract explicitly argues residual sufficiency and reconstructs/recomputes KV from residual checkpoints. |
| ILRe and REFORM select/gather tokens before recomputation. | ILRe; REFORM | **Reasonable to strong.** Their abstracts describe intermediate-layer retrieval/gathering and later recomputation, though their persistence/workload differs from CoMem. |
| Token/KV compression changes retained token/state budgets. | StreamingLLM, H2O, SnapKV, PyramidKV, MiniCache | **Strong family-level match.** |
| MemoryLLM, LongMem, and XC-Cache use latent memory pools or auxiliary readers. | Those three works | **Reasonable.** They are heterogeneous, but correctly presented as external memory references rather than matched controls. |
| The benchmark suite and scorers correspond to RULER, BABILong, LongBench, LoCoMo, and LongEval/LongChat. | Benchmark papers/blog | **Mostly match.** LongEval needs an exact versioned artifact citation. |
| SnapKV/PyramidKV are retained-KV baselines requiring full prefill before eviction. | SnapKV; PyramidKV | **Strong match.** |

# Method, formula, and boundary audit

1. **Interface definition:** Half-open layer ranges and \(h_j\) as block-\(j\) input are clear. The \(j=0\) endpoint is conceptually valid and uses token IDs rather than pretending to store a zero-depth residual.
2. **Storage formula:**  
   \[
   |h_j|/|\mathrm{KV}|=d/(2Ln_{\mathrm{kv}}d_{\mathrm{head}})
   =n_q/(2Ln_{\mathrm{kv}})
   \]
   is correct for a full per-layer K/V cache at a common dtype and \(d=n_qd_{\mathrm{head}}\). For Qwen3-8B, \(4096\times2=8192\) B/token and \(2\times36\times8\times128\times2=147456\) B/token, giving \(1/18\). \(128{,}000\times8192\) B is about 0.98 GiB, consistent with “about 1 GiB.”
3. **Read cap:** \(1+12\times512+512=6657\) tokens is arithmetically correct. Actual shorter tails/queries are disclosed.
4. **Distillation loss:** The symmetric weighted KL on the common teacher-selected support is mathematically specified. The omitted outside-support mass is a real approximation limitation, acknowledged but not quantified.
5. **Generation caching:** The claim that lower layers cache the query/generation prefix and upper layers cache the selected pack is plausible, but implementation-level validation is only indirectly described. **Unverifiable** from the frozen manuscript alone.
6. **Break-even equation:** The displayed \(Q^\star\) form is appropriate when selection cancels and the denominator is positive. The paper clearly reports unavailable/non-finite cells and does not extrapolate the 128k generation trend.
7. **Equal-latency calibration:** Latency calibration is disjoint from quality examples and uses a predeclared ±5% band. However, it uses only three reserved synthetic documents and reports no directly measured end-to-end p95.
8. **Exact split correctness:** Qwen continuous-prefix replay exactly matches \(j=0\) on all 1,500 examples; Hy3 reports \(\max|\Delta \mathrm{logit}|=0\) at four split points. These are good implementation checks, although raw artifacts were not available in the frozen source and are therefore **Unverifiable** beyond the reported values.

# Baselines, benchmarks, metrics, seeds, compute, and reproducibility

## Baselines

- **Good:** matched \(j=0\), same-\(j\) adapter off/on, continuous-prefix ceiling, context/position controls, selector variants, chunk and attention ablations.
- **Descriptive only:** KV-Direct, InfLLM, StreamingLLM, MemoryLLM, LLoCO, SnapKV, PyramidKV often differ in backbone, prompts, length extension, or timing boundary. The paper usually labels these differences correctly.
- **Missing:** a same-backbone reusable per-layer KV/PIC or learned modular-cache baseline under the same selected chunks and hardware.

## Benchmarks and metrics

- RULER and synthetic multikey evidence are well specified and paired.
- BABILong, LongBench, and LongEval use official/deterministic scorers, but some generation budgets differ between controls and primary rows.
- LoCoMo's primary metric uses a mutable undated `gpt-4o` endpoint; the date, parsed decisions, cluster bootstrap, and a 200-item DeepSeek-V3 audit mitigate but do not eliminate reproducibility concerns.
- The natural-task scores are generally low and the paper acknowledges incomplete PG-19 overlap audits.

## Seeds and statistics

- Evaluation seeds, shard counts, and sample sizes are usually given.
- The central paired RULER interval and McNemar test are appropriate conditional on the trained adapter.
- Training-run uncertainty is not cleanly measured.
- Equal-latency IID-example bootstrap is not aligned with cell/conversation structure.
- Probe Student-\(t\) intervals are correctly identified as task–split, not seed, uncertainty.

## Compute and efficiency

- Final adapter compute is reported as approximately 2.9 H20 GPU-hours; hardware, steps, token count, and throughput are given.
- Total project compute and training peak memory are missing.
- Main latency uses medians across independent processes, with warmups/repetitions stated. Production concurrency, p95/p99, and multi-tenant scheduling remain open.

## Reproducibility judgment

The paper contains enough detail to reproduce the algorithm and many evaluation protocols, assuming the promised anonymous artifact actually contains the stated adapter, configs, scripts, prediction hashes/shards, judge decisions, and timing records. Artifact contents and executable correctness are **Unverifiable** because only the frozen PDF/source were permitted. The mutable LoCoMo judge and missing exact total-compute/run-variance records prevent a higher score.

# Figure and table audit

## Figures

- **Figure 1:** Clear and faithful to the method. It visually distinguishes the \(j=0\) matched control, bounded selection, residual continuation, and overlap Write. No hidden/reviewer-directed text was found in extracted text, source commands, strings, or visual inspection.
- **Figure 2:** Properly labeled as motivation. The plotted “adapter pushes deeper” and knee/readout-gap summaries are protocol dependent; the caption and appendix state this. Axes are legible, though panel (a)'s qualitative star/arrow presentation should not be read as inferential evidence.

## Tables

All 36 labeled tables were inspected. Arithmetic spot checks passed for the central speedup, RULER macro, LongBench macros, BABILong means, LongEval means, LoCoMo weighted overall, seed means/SDs, storage ratio, and read cap.

Key table-specific observations:

- **Tables 2/30/31:** strongest evidence; boundaries and pairing are clear.
- **Tables 3/25/32:** useful serving/storage audit, but single-query medians and limited 128k generation support constrain conclusions.
- **Tables 4/8:** transparent protocol, but statistical resampling unit is the main concern.
- **Tables 5/6:** clean local diagnosis; not yet externally validated.
- **Tables 9–12:** adapter necessity is convincing; multi-depth training confounds remain.
- **Tables 14–17:** useful mechanism/selector diagnostics, mostly point estimates.
- **Tables 18–23:** many external rows are not strictly comparable; captions generally disclose this.
- **Table 29:** retained-KV quality and systems-cost panels have different context-extension/hardware boundaries; the caption prevents a direct ratio, but readers should not treat it as a matched systems comparison.
- **Tables 26–28:** good configuration detail; missing peak training memory and clean same-batch seeds are explicit.
- **Tables 33–34:** promising architecture-portability check, but small support and no matched systems baseline.
- **Tables 35–36:** cohort and LoCoMo denominator accounting are clear.

# Desk, compliance, anonymity, and ethics audit

| Check | Result |
|---|---|
| Frozen version | Reviewed only `v5_20260804_003238.pdf`, `v5_source_20260804_003238`, and the NORMAL template. |
| PDF/pages | 23 A4 pages. Main text reaches Conclusion/Limitations/Ethics by pp. 7–8; references begin on p. 8; appendix begins p. 11. |
| Long-paper page limit | The paper appears to use 8 main-content pages before references, consistent with the usual ACL long-paper layout. Exact compliance with the specific target ARR cycle's currently operative policy is **Unverifiable** from the frozen files alone. |
| Official style | Source uses `\usepackage[review]{acl}` at 11pt; visual inspection is ACL-like. The supplied `acl.sty` itself was not independently compared byte-for-byte with the official cycle release: **Unverifiable**. |
| Limitations | Present as an unnumbered section and substantive. |
| Ethics | Present and discusses privacy, inversion/membership risks, access control, deletion, misuse, and energy. |
| Anonymity | Author is “Anonymous ACL Submission”; no acknowledgments, personal paths, author URLs, affiliations, or obvious identifying text found in visible source/PDF. The cited official Hy3 page is a third-party model reference, not an identity leak. |
| Hidden injection/manipulation | No reviewer instructions, score requests, invisible colored text, `phantom`/overlay tricks, embedded files, JavaScript, launch actions, or suspicious strings found. Figure PDFs were separately inspected. |
| Placeholders | No TODO/TBD/FIXME/Lorem/placeholder text found. Em dashes denote genuinely unavailable measurements and are explained. |
| References | No missing/duplicate `main.bbl` keys; all 43 entries cited. Several metadata corrections/completions are needed as listed above. |
| Cross-references | 57 unique labels; no duplicate labels or references to missing labels were found mechanically. No `??` appeared in extracted PDF text. |
| Abstract/table consistency | The five principal numerical summaries—matched Read/quality, break-even, BM25 equal-latency margin, BGE tie, and context/overlap diagnosis—match the corresponding main/appendix tables. |
| PDF/source build | The frozen PDF is readable and internally resolved. A fresh compile was **Unverifiable** because no TeX engine was installed in the execution environment. |
| Ethical acceptability | No new human subjects; released datasets/models are used. Residual-state privacy and deletion concerns are appropriately raised. No ethics-based rejection concern. |

# Scores

## Soundness: 3.5 / 5

The central same-pack endpoint is carefully controlled and the paper is unusually disciplined about causal boundaries. Formulae and headline arithmetic check out. The main deductions are supported as conditional measurements. Soundness is held below 4 by single-run adapter uncertainty, the equal-latency bootstrap design, and the absence of a matched nearest systems baseline.

## Excitement: 3.0 / 5

The depth-reuse axis and diagnosis are interesting, but reusable intermediate representations and reusable document caches have substantial precedent. The strongest contribution is a transparent measurement/negative-boundary study rather than a clearly superior new system. A matched PIC/modular-KV result or natural-task repaired frontier would raise excitement.

## Overall: 3.0 / 5

**Findings-level.** I would support acceptance to Findings in its current form because the controlled measurement, negative equal-latency result, and failure diagnosis are useful and responsibly scoped. For ACL main, I would want at least one matched nearest reusable-context baseline and cleaner run-level/statistical evidence.

## Confidence: 4.0 / 5

I read the complete 23-page paper twice, including appendices, inspected all figures/tables and frozen source, mechanically checked references/cross-references/abstract numbers and selected arithmetic, and performed bounded citation/novelty verification. Confidence is not 5 because raw experiment artifacts were outside the permitted inputs, some network metadata remained Unverifiable, and a fresh TeX build was unavailable.

## Reproducibility: 3.5 / 5

The algorithm, configs, masks, prompts, scorers, timing boundaries, hashes, sample counts, and many seeds are documented well. The score is limited by the mutable LoCoMo judge, promised-but-uninspected artifact, missing clean same-batch multi-run results, missing total project compute/peak training memory, and inability to freshly compile in the audit environment.

# Review-process self-check

- Two-pass read completed, including both appendices.
- Claim–evidence map completed.
- Desk/page/style/Limitations/anonymity/hidden-injection/placeholders/abstract-number checks completed.
- `main.bbl`: 43/43 entries enumerated; all cited; identity checked where network permitted; unresolved metadata marked Partial/Unverifiable.
- Eight citation–claim families checked.
- Five novelty queries performed; closest-paper comparison and 2026-05-04 cutoff applied.
- Method/formula/boundary/baseline/benchmark/metric/seed/statistics/scope/compute/reproducibility audit completed.
- Both figures and all 36 labeled tables inspected.
- Exact quoted weakness anchors and “missing X” assertions were rechecked against frozen source before saving.
- No evidence from any other review/history/TODO/status/current/calibration file was used.
