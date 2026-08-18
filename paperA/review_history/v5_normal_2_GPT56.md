---
review_mode: normal
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.0
reproducibility: 3.5
---

# Paper A v5 — Normal Reviewer #2

## Summary and recommendation

This paper studies a repeated-query, stable-corpus serving regime in which each document is written once into one residual vector per token at split depth $j$, a bounded set of chunks is selected per query, and inference resumes only the upper layers. The paper's deliberately narrow primary claim is an internal, same-evidence depth-reuse measurement on Qwen3-8B rather than superiority over RAG/PIC/modular-KV systems. Its main matched endpoint compares $j=0$ raw-token replay with $j=12$ residual replay under the same selected chunks, order, sink, mask, examples, and LoRA: RULER-B changes from 99.19 to 96.07 (paired gap 3.12, 95% CI [2.36, 3.93]) while isolated selected-pack Read changes from 931.9 to 664.4 ms (1.403x). It also reports measured reuse break-even, selector-sensitive equal-latency results, and a focused context/position/overlap diagnosis.

I find the paper unusually careful about timing boundaries, negative results, contamination caveats, and claim scale. The central endpoint is credible and informative. However, the evidence still does not establish whether this residual-depth operating point is competitive with the nearest reusable-context alternatives under the paper's own workload; moreover, one headline equal-latency inferential interval uses an IID example bootstrap despite an explicitly heterogeneous, clustered mixture. These limitations keep my recommendation at **3.0 (Findings)** rather than **4.0 (main conference)**. This is not a rejection of the bounded measurement claim: the paper largely supports that claim, but the current contribution/evidence package is better calibrated to Findings.

## Claims and evidence map

| ID | Claim as reviewed | Main evidence | Assessment |
|---|---|---|---|
| C1 | A depth-$j$ residual/token can serve as a persistent document object for bounded selected-pack suffix execution. | Sec. 4, Eq. 1; Fig. 1; Appendix Tables 25–27; exact partition/control evidence in Tables 2, 12, and Hy3 discussion. | Supported as an implemented interface, not as a universal architecture result. |
| C2 | Under identical evidence and adapter, $j=12$ trades 3.12 RULER points for 1.403x faster isolated model Read than $j=0$. | Sec. 5.1 / Table 2; Appendix Table 30; $n=1,500$, paired bootstrap CI and McNemar test; three-process latency medians. | Strongest and well-supported claim. The paper correctly excludes retrieval, persistent I/O, reusable Write, and decode from this depth-only Read result. |
| C3 | A rank-32 self-distillation adapter makes the residual interface usable without updating the backbone. | Sec. 4; Appendix Tables 9–12 and 26; same-$j$ adapter on/off controls (e.g., RULER single/multikey and LoCoMo/BABILong). | Supported for the tested checkpoints/tasks. “Usable” remains task-dependent, which the paper acknowledges. |
| C4 | The 8 KiB/token store is worthwhile only under reuse, with measured break-even around 8–11 queries at 32k for generations through 128 tokens and 25.8–27.6 at 128k for one generated token. | Sec. 5.2 / Table 3; Appendix efficiency and store-I/O sections. | Supported for the stated hardware, placement, and timing boundaries; not a general deployment constant. |
| C5 | At equal online latency, aggregate quality is selector-dependent and CoMem wins neither tested aggregate. | Sec. 5.2 / Table 4; Appendix Table 8: BM25 replay 64.78 vs 53.22, BGE replay 54.22 vs 53.22. | Point estimates support the qualitative conclusion; uncertainty treatment for the nine-cell mixture needs a cluster/cell-aware robustness analysis (Major W2). |
| C6 | Missing lower-layer document context is a major Write-side failure on the paired multikey diagnostic; overlap recovers most of that local gap without changing persistent bytes or per-query Read. | Sec. 5.3 / Tables 5–6; Appendix Tables 12 and 31. | Supported only for the displayed synthetic cohort, exactly as scoped. It does not establish a natural-task or end-to-end repaired frontier. |
| C7 | Model-side Read is bounded by sink + $kc$ + query, while persistent store/index/selector costs are not bounded. | Sec. 4, lines 239–245; Appendix Table 24 and store-I/O Table 32. | Supported. The paper explicitly shows BM25 lookup growing approximately linearly with store size. |
| C8 | The contribution is an internal depth-reuse measurement/diagnosis, not a competitive win over raw-text retrieval or modular caches. | Abstract, Introduction, Related Work, Conclusion, Limitations. | Properly and repeatedly scoped. The absence of a nearest matched baseline limits impact rather than falsifying this claim. |

## Numerical spot checks

I mechanically recomputed/checked the following paper numbers from displayed cells or formulas:

1. RULER-B CoMem macro: $1441/15=96.0667$, reported as **96.07** (Appendix Table 35).
2. Read speedup: $931.9/664.4=1.40262$, reported as **1.403x** (Tables 2 and 30).
3. Matched RULER gap: $99.19-96.07=3.12$ points.
4. Storage ratio: $144\,\mathrm{KiB}/8\,\mathrm{KiB}=18x$; 128k tokens at 8,192 bytes/token is exactly **1 GiB**.
5. LongEval CoMem five-length macro: $(69+75+64+67+70)/5=69.0$; adding the 4k score 92 gives **72.83** over six lengths.
6. LongBench CoMem macro: $(4.12+11.01+11.62+12.83+25.41+7.91)/6=12.15$; the displayed full-context/KV-Direct row gives **12.17**.
7. BABILong CoMem means from the seven displayed lengths are **55.57**, **27.00**, and **68.71**, matching 55.6/27.0/68.7.
8. Context-position interaction: $(100-88)-(100-92.5)=4.5$ points, matching the stated non-additive interaction.
9. Equal-latency deltas use CoMem minus replay: $53.22-64.78=-11.56$ and $53.22-54.22=-1.00$.
10. Overlap gain at $w=32$: $98.5-92.5=6.0$ points.

One internal presentation inconsistency remains: Sec. 5.1 states a matched LongBench $j=0$ score of **12.31**, while Appendix Table 21 and its caption show **12.17** for the displayed full-context/KV-Direct row. See Minor W3.

## Strengths

1. **The causal comparison is unusually well isolated.** Table 2 fixes selected evidence, ordering, sink, mask, examples, and adapter, and changes the replay start only. The continuous-prefix control exactly matching $j=0$ is a useful fidelity ceiling. This supports the narrow depth-reuse trade-off rather than conflating it with retrieval or context truncation.

2. **The paper separates systems boundaries instead of marketing one large speedup.** It distinguishes selected-pack model Read, store-ready online prefill, Write-inclusive pipeline, external storage/I/O, selection, and decode. The abstract foregrounds 1.403x rather than the secondary 64.9x dense-to-bounded number, and the text explains why the latter is not a depth-only effect.

3. **Negative evidence and workload boundaries are treated as results.** The paper reports that CoMem loses the BM25 equal-latency aggregate, ties rather than wins under the frozen-BGE variant, needs reuse to amortize an 8 KiB/token store, and fails when evidence exceeds top-$k$ or requires global aggregation. This makes the deployment conclusion more trustworthy.

4. **Claim scale is disciplined.** The overlap result is explicitly limited to one paired multikey cohort; cross-scale/MoE ports are not presented as matched replications; the multi-depth curve is not called compute-matched; and natural-task results are not called contamination-free generalization.

5. **Reproducibility reporting is strong for a systems/NLP paper.** The appendix provides model revision, adapter SHA-256, architecture, masks/positions, BM25 parameters, generation/scoring rules, sample counts, software versions, training objective and schedule, timing warmups/repetitions, storage placement, and known missing logs. The frozen source also compiles to a 23-page PDF with no unresolved references/citations.

6. **The paper inspects benchmark/judge integrity rather than ignoring it.** It removes an InfiniteBench comparison after a PG-19 overlap audit, discloses incomplete audits elsewhere, distinguishes LoCoMo item and conversation-cluster bootstraps, dates the mutable judge, and provides an independent-judge audit.

## Major weaknesses

### W1. No matched nearest reusable-context baseline, so practical significance of the residual object remains unresolved

- **Issue.** The paper's closest deployment competitors are PIC/chunk-KV repair/learned modular-KV systems, yet no same-backbone, same-pack, same-hardware, same-storage-tier comparison is implemented. Consequently the study measures CoMem internally but cannot determine whether storing one residual/token and executing 24 suffix layers is a useful frontier relative to storing/repairing modular KV.
- **Why it matters.** This is central to excitement and deployment relevance: the observed endpoint loses 3.12 RULER points for only 1.403x isolated Read speedup, while prior reusable-context systems target the same repeated-document workload and often add context/boundary repair. Taxonomy alone cannot show whether CoMem's much smaller object compensates for its suffix compute and quality loss.
- **Evidence anchor.** Related Work, pp. 2–3 / lines 150–156 and 187–197; Limitations, p. 7 / lines 434–445. Exact quote (14 words): **“The paper does not provide a same-backbone, same-hardware implementation of the closest PIC”**.
- **Required remedy/test.** A minimal decisive experiment is one same-backbone nearest-baseline implementation—preferably EPIC/CacheBlend-like repair or KV Packet—using the identical retrieved packs, Qwen3-8B, H20, storage tier, quality cohort, TTFT boundary, and persistent-byte accounting. Report quality, Write, fetch, TTFT/p95, store bytes/token, and break-even. If implementation is infeasible, the current claim should remain explicitly measurement-only and the venue score should reflect that narrower contribution.
- **Severity: Major.** This limits the central contribution's comparative meaning, though it does not invalidate C2.
- **Mechanical verification.** Source search finds explicit statements that no such artifact/result exists; no table supplies a same-backbone modular-cache row. I do **not** request unrelated new benchmarks or a broad leaderboard.

### W2. The equal-latency confidence intervals use the wrong resampling level for the heterogeneous nine-cell mixture

- **Issue.** The paper presents 95% CIs for a macro built from nine equally weighted task-length cells, including 100 LoCoMo items all from conversation 0, but bootstraps 900 examples IID after pooling them. It does not resample task cells or conversations.
- **Why it matters.** The inferential population implied by an equal-cell macro is not 900 exchangeable examples. Pooling can understate uncertainty and gives each cell's within-cell observations an independence assumption that is especially questionable for conversational items. This matters most for the BGE “statistical tie” claim, whose CI is already close enough that a cluster-aware interval is decision-relevant.
- **Evidence anchor.** Appendix Table 8, p. 12, “Quality cohort” and “Statistical unit.” Exact quote (18 words): **“it does not resample task cells or LoCoMo conversations.”**
- **Required remedy/test.** Recompute both selector comparisons with a hierarchical paired bootstrap: resample the nine cells (or treat them as a fixed complete set and report cell-wise differences), then paired examples within each cell; for LoCoMo, resample conversations rather than treating the first 100 items from one conversation as independent. Also report the nine cell-level deltas. Keep the current point estimates.
- **Severity: Major.** The point estimates remain valid, but the uncertainty-backed wording “statistically tied” is not fully supported by the chosen resampling unit.
- **Mechanical verification.** Table 8 explicitly states 9 cells, $n=100$/cell, first 100 LoCoMo items all from conversation 0, equal cell weights, pooled IID resampling of 900 paired differences, and no cell/conversation resampling.

## Minor weaknesses

### W3. One natural-task matched-baseline number is internally inconsistent

- **Issue.** Main-text Sec. 5.1 reports 12.31 versus 12.15 on LongBench, but Appendix Table 21 reports 12.17 versus 12.15 for the displayed full-context/KV-Direct comparison.
- **Why it matters.** The difference is small and not headline-changing, but it makes the audit trail for the natural-task scope check ambiguous.
- **Evidence anchor.** p. 5 / lines 332–337 versus Appendix Table 21, p. 16. Exact quote (11 words): **“and 12.31 versus 12.15 on six LongBench QA datasets”**.
- **Required remedy.** Identify whether 12.31 is a distinct same-LoRA $j=0$ run. If so, add that row and protocol to Table 21; otherwise correct Sec. 5.1 to 12.17.
- **Severity: Minor.** Local consistency issue only.
- **Mechanical verification.** Recomputing the Table 21 row yields $(3.70+11.82+12.68+12.03+25.30+7.49)/6=12.17$; source search finds 12.31 only in Sec. 5.1.

### W4. The three-run robustness table confounds seed with effective batch and misses the exact headline cohorts

- **Issue.** The flagship is one effective-batch-8 run; seeds 1 and 2 use effective batch 3, and robustness is reported on reduced-support RULER/BABILong cells rather than the exact RULER-B, LongEval, or LoCoMo headline comparisons.
- **Why it matters.** Training stochasticity is a plausible source of variation for the learned interface, especially because C3 depends on the adapter. The current table is useful but cannot attribute variation to seed or establish uncertainty for the exact headline.
- **Evidence anchor.** Limitations, p. 7 / lines 424–432; Appendix Table 28, p. 20. Exact quote (13 words): **“not a clean estimate of training-seed variance.”**
- **Required remedy.** Preferably train two additional batch-8 seeds and evaluate the exact 15-cell RULER-B endpoint. If compute prevents this, keep the limitation and avoid treating Table 28 as seed variance; the current paper already mostly does so.
- **Severity: Minor.** The paired inference comparison is still measured precisely for the retained adapter; this affects generality over training runs, not arithmetic correctness.
- **Mechanical verification.** Table 28 labels adapters “batch 8 / 3 / 3” and states that the exact headline cohorts were not retained.

### W5. The self-distillation objective discards unmeasured teacher probability mass outside top-64

- **Issue.** Both teacher and student distributions are renormalized only over the teacher's top-64 logits, and the captured teacher mass was not retained.
- **Why it matters.** Without support-mass statistics, readers cannot judge how faithful this approximate symmetric-KL objective is across query positions or whether important tail behavior is ignored. This is relevant to reproducing C3 and interpreting interface failures.
- **Evidence anchor.** Sec. 4, p. 4 / lines 264–285; Appendix Table 26, p. 20. Exact quote (10 words): **“We did not retain the teacher mass captured by $S_t$.”**
- **Required remedy.** Log and report mean/quantiles of teacher top-64 mass on training and held-out text, plus a small top-32/top-64/top-128 or full-logit check on a held-out subset. This is a focused objective-validation test, not a request for new task families.
- **Severity: Minor.** Existing same-$j$ adapter controls show that the objective works empirically, but its approximation is under-characterized.
- **Mechanical verification.** The source formula and appendix both explicitly specify shared top-64 renormalization, discarded outside logits, and missing retained mass.

## Questions for the authors

1. Is the LongBench 12.31 $j=0$ number a distinct same-LoRA replay run from the 12.17 KV-Direct row? If yes, please expose its six cells and exact protocol.
2. For Table 8, how do the nine cell-wise CoMem-minus-replay differences look under BM25 and BGE? Does the BGE conclusion remain a tie under cell- or conversation-aware resampling?
3. Among EPIC/CacheBlend/KV Packet, which system do the authors consider the single nearest operational baseline, and what obstacle prevented one matched Qwen3-8B implementation?
4. What fraction of teacher probability mass is typically captured by top-64 during distillation? If unavailable for the frozen run, can it be measured on a held-out sample with the same teacher?
5. The Read-only speedup is stable across three processes, but end-to-end deployment is more relevant. Are component p95 records sufficient to reconstruct or rerun directly measured end-to-end TTFT p95 for the equal-latency arms?

## Focused suggestions

- Preserve the current narrow wording; it is one of the paper's strengths.
- Make Table 2 visually distinguish “depth-only Read” from all end-to-end timings, perhaps with a compact included/excluded component row.
- Add a nine-row cell-delta table for the equal-latency audit and a hierarchical bootstrap.
- Resolve 12.31 vs 12.17 and distinguish “same-LoRA $j=0$” from generic full-context/KV-Direct consistently.
- If only one extra experiment is possible, prioritize the same-backbone nearest modular-cache baseline over additional unrelated models/tasks.

## Citation and related-work audit

### `main.bbl` completeness and verification

- `main.bbl` contains **43 entries**; all 43 are cited in the frozen source, and no cited key is absent from `main.bbl`.
- Title/venue/year or identifier checks succeeded through DOI/Crossref, arXiv, ACL Anthology, or OpenAlex for most entries, including Cache-Craft, LongBench, PyramidKV, KV Packet, Cartridges, HCache, Distillation, RULER, LoRA, EPIC, RAGCache, BABILong, RAG, SnapKV, ReadOnce, MiniCache, TurboRAG, LoCoMo, XC-Cache, KV-Direct, PG-19, BM25, Embedding Recycling, GemFilter, LLoCO, MEPIC, LongMem, MemoryLLM, InfLLM, StreamingLLM, Retrieval Meets Long Context, Qwen3, APE, CacheBlend, and H2O.
- Direct external verification was **Unverifiable under rate limits or absent stable metadata** for some records (notably the Hy3 model-card citation and some newer arXiv-only records during batch queries). Their bibliographic strings are internally well formed, but I do not claim external confirmation.
- Date/cutoff note: Cartridges at Scale (arXiv 2606.04557, June 3, 2026) and SemPIC (arXiv 2607.28069, July 30, 2026) are after the requested novelty cutoff of **May 4, 2026** and therefore were not used to reduce pre-cutoff novelty. They are acceptable concurrent/later context. The bibliography's “RAGCache 2025” is defensible as online-first/arXiv dating, while Crossref lists the print issue in 2026; this is not a substantive citation error.

### Citation–claim spot checks (8)

| Citation(s) | Paper claim checked | Match assessment |
|---|---|---|
| Lewis et al. (2020); Xu et al. (2024) | Raw-text retrieval bounds selected input but recomputes the model over retrieved text. | **Match.** Appropriate high-level characterization of RAG/retrieval-plus-long-context inference. |
| CacheBlend; TurboRAG; Cache-Craft | Reusable chunk-KV systems precompute/reuse chunk state and address context-dependent composition/repair. | **Match.** Abstracts describe precomputed KV reuse, fusion/recomputation, and chunk-cache management. |
| EPIC; APE; MEPIC | PIC/parallel encoding enables modular/position-independent context reuse with boundary/position/alignment handling. | **Match.** The wording is broad but faithful; MEPIC is a later 2025 preprint within cutoff. |
| KV Packet | Learned context-independent packets use trainable adapters/distillation and avoid document recomputation. | **Match.** Its abstract explicitly describes immutable KV packets with lightweight soft-token adapters and self-supervised distillation. |
| ReadOnce; Embedding Recycling | Intermediate text representations can be cached/reused with later-layer adaptation. | **Match.** These are genuine intermediate-representation precedents and materially narrow novelty. |
| HCache; KV-Direct | Activations/residuals can checkpoint state or reconstruct/recompute KV/suffix computation. | **Match with nuance.** HCache is restoration-oriented; KV-Direct is the closest single-residual/token precedent but targets KV redundancy/bounded-memory inference rather than selected cross-query document serving. |
| SnapKV; PyramidKV; MiniCache; H2O | These methods compress/select retained token/KV state rather than instantiate CoMem's persistent selected document object. | **Match.** Correctly treated as retained-KV references, not the nearest repeated-document baseline. |
| RULER, BABILong, LongBench, LongEval, LoCoMo | Benchmark identities and scorer roles. | **Match.** The appendix names sample support and official/local scoring boundaries; LongEval is cited through the LongChat project/blog, which is bibliographically weaker but recognizable. |

### Missing/weak citation issue

The most important omission from the related-work discussion is **RSCE: Training-Free Residual Stream Encoding for Persistent Context Amortization** (KnowFM 2026, DOI 10.18653/v1/2026.knowfm-1.11). It predates the May 4, 2026 cutoff in bibliographic databases and encodes a document into a mean-pooled intermediate residual vector for persistent, amortized query-time use. It is not the same method—RSCE stores one vector per document and injects it additively, while CoMem stores one vector per token, retrieves chunks, and executes the suffix—but it is close enough in “persistent residual representation for repeated context” that it should be cited and contrasted. This omission lowers novelty confidence modestly, not fatally.

## Novelty search (cutoff: 2026-05-04)

I stopped after five focused searches, as requested. Search services were partially rate-limited; unavailable metadata is marked Unverifiable rather than extrapolated.

1. **“intermediate residual transformer cache/memory”** — no clear same-design result surfaced; mostly irrelevant results. This query alone is weak.
2. **“residual stream” + “KV cache”** — surfaced KV-Direct (March 20, 2026), which establishes that a single residual/token can reconstruct layerwise KV exactly and is the closest representation-level precedent.
3. **“position-independent caching” + LLM** — surfaced EPIC, MEPIC, and encoder-based native PIC. These are the closest serving-workload systems but store/reuse per-layer KV and use repair/alignment rather than one depth-$j$ residual and suffix execution.
4. **“reusable representations” + transformer** — surfaced ReadOnce Transformers and Embedding Recycling, establishing that reusable intermediate text representations with later-layer adaptation predate CoMem.
5. **“residual stream context encoding / persistent context amortization”** — surfaced RSCE (KnowFM 2026), a missing close paper using mean-pooled intermediate residuals as persistent document context.

Closest-paper comparison:

| Work | Shared core | Key difference from CoMem | Novelty implication |
|---|---|---|---|
| KV-Direct (2026-03-20) | One residual vector/token as sufficient transformer state; suffix/KV recomputation. | Targets within-sequence/cache redundancy and bounded-memory inference, not independently written/retrieved document chunks across repeated queries; no CoMem-style $j=0$ selected-pack serving endpoint. | Strongly narrows representation novelty; CoMem's novelty is the cross-query bounded selected-pack operating-point study. |
| RSCE (KnowFM 2026) | Persistent intermediate residual representation amortized across queries. | One mean-pooled vector/document, additive injection, training-free; not per-token states, chunk retrieval, or direct suffix execution. | Missing citation; conceptually close but not anticipatory of the complete interface. |
| ReadOnce Transformers (2021) | Reusable intermediate text representations with adapted later computation. | Earlier encoder/reader-style reuse, not modern decoder-only bounded RAG serving with explicit storage/Write/Read accounting. | Means “cache intermediate representations” is not novel; measurement/system instantiation may be. |
| EPIC / CacheBlend / TurboRAG | Repeated-document chunk reuse in RAG serving. | Per-layer KV plus repair/alignment/recomputation; different storage and online-compute frontier. | Establishes the workload and nearest baseline family; comparison gap is consequential. |
| KV Packet (2026-04-14) | Learned reusable document object, self-supervised distillation, low online recomputation. | Stores learned modular KV packets/soft-token adapters rather than one residual/token and suffix layers. | Very close deployment competitor within cutoff; supports novelty only at the object/interface level, not the broad reusable-cache idea. |

**Novelty judgment.** The broad ideas of persistent context, reusable intermediate representations, residual checkpoints, and modular document caches all predate CoMem. The defensible novelty is narrower: a **single tunable depth-$j$ residual per selected token, direct decoder suffix execution, and an unusually controlled/transparent measurement of the $j=0\rightarrow12$ depth endpoint plus workload boundaries**. That is a legitimate incremental contribution, but not a new memory paradigm. I rate novelty/excitement as moderate.

**Three-month rule.** The cutoff is May 4, 2026. KV Packet (April 14, 2026), KV-Direct (March 20, 2026), and RSCE (bibliographic year 2026) are inside the relevant pre-cutoff window and should inform novelty. Cartridges at Scale and SemPIC are after cutoff and are treated as concurrent/later only. Exact first-publication day for RSCE was **Unverifiable** from the ACL page beyond the 2026 proceedings record; because the record predates/aligns with the cutoff in indexed metadata, I conservatively treat it as relevant related work, not as grounds for a strong priority accusation.

## Method, formula, baseline, fairness, metrics, statistics, compute, and reproducibility audit

- **Method/interface:** The half-open layer convention and $h_j$ definition are clear. Query lower-layer processing and upper-layer full causal cross-pack attention are specified. Overlap-Write correctly changes Write work/edit invalidation but not persistent bytes or Read length.
- **Equation 1:** The residual/full-KV byte ratio $d/(2Ln_{kv}d_{head})=n_q/(2Ln_{kv})$ is correct for full layerwise K and V at a common dtype and $d=n_qd_{head}$. It intentionally compares one residual/token with all-layer KV/token, not with repaired/quantized/selected KV systems.
- **Equation 2:** The symmetric top-64 KL is mathematically defined, but the discarded support mass is unmeasured (W5).
- **Minimal experiment:** The smallest claim-deciding experiment for C2 is present: same pack, same adapter, $j=0$ vs $j=12$, paired quality, isolated Read timing, plus a continuous-prefix fidelity control. The smallest experiment missing for practical relevance is one matched nearest modular-cache baseline (W1).
- **Baseline fairness:** Internal $j=0$ fairness is strong. External rows are appropriately labeled descriptive because backbones, prompts/training, context extension, and timing differ. Equal-latency BGE intentionally changes the replay selector only and is correctly framed as selector sensitivity, not a same-pack depth comparison. However, W2 affects its inferential label.
- **Metrics:** RULER/BABILong/LongEval/LongBench scoring is named and generally appropriate. LoCoMo correctly prioritizes a semantic judge over formatting-sensitive lexical metrics, dates the endpoint, saves item decisions, and supplies a cluster bootstrap plus independent-judge audit. The mutable judge prevents exact future reproduction, as acknowledged.
- **Seeds/statistics:** Paired RULER inference is strong. LoCoMo dependence is handled thoughtfully. The equal-latency bootstrap level is the main statistical weakness. Training-run variability remains confounded by batch size and reduced support.
- **Claim scale:** The abstract and conclusion are consistent with the measured scope: no quality-preserving acceleration, no universal raw-replay margin, no modular-cache superiority, no universal overlap repair.
- **Compute:** Final training compute (8 H20, about 22 minutes/2.9 H20 GPU-hours) is reported; total exploratory/evaluation compute and training peak memory are not retained. This omission is disclosed rather than hidden.
- **Reproducibility:** High for configuration and evaluation protocol, but actual artifact executability was **Unverifiable** because only the frozen PDF/source and template were permitted for this review. The source promises an anonymous archive, yet I did not inspect it. Hy3's model-card-only citation and proprietary/huge hardware requirements also make that appendix result difficult to reproduce independently.

## Figure and table audit

I visually inspected the complete 23-page PDF and all source table files.

- **Figure 1:** Clear architecture and separation of Write/Select/Read; the caption correctly distinguishes the matched 1.403x depth effect from larger dense-to-bounded numbers. Legible at normal zoom, though dense.
- **Figure 2:** Properly labeled motivation rather than validation. Probe/native-readout definitions are deferred but supplied in Appendix A.1. It should not be interpreted causally, and the paper says so.
- **Tables 1–6 (main):** Table 1 is taxonomic, not a ranking; Table 2 is the core matched endpoint; Table 3 exposes missing/non-finite cells; Table 4 discloses selector asymmetry and under-filling; Tables 5–6 correctly limit diagnosis to one cohort. All main-table claims are substantially captioned.
- **Tables 7–36 (appendix):** I checked every displayed table/source for support, timing boundary, sample size, and caveat. Particularly useful are protocol-complete Table 8, multi-depth Table 10, HCache adaptation Table 12, store scaling Table 24, reproducibility Tables 25–28, replay control Table 30, store tiers Table 32, and LoCoMo category Table 36. No table visibly overflows or becomes unreadable in the frozen PDF.
- **Cohort consistency:** The paper distinguishes RULER Cohort A vs B and does not directly subtract them. LongEval five- vs six-length macros are distinguished. The one unresolved numeric inconsistency is LongBench 12.31 vs 12.17 (W3).
- **Missing cells:** Dashes are generally explained. The missing $j=12$ Write value and missing 128k longer-generation break-even cells are explicitly not inferred.

## Desk, format, anonymity, style, injection, and TODO audit

- **Frozen object:** Reviewed only `v5_20260804_003238.pdf`, its matching `v5_source_20260804_003238`, and the NORMAL template. PDF SHA-256 observed: `c98ed58c75ebb80c892216b18be91e92892efb1e53e0f9b789e3051ac069aad5`.
- **Pages:** 23 total. The numbered main paper runs through page 8, followed by references and appendix. This is consistent with an 8-page main-paper layout; exact current ARR policy compliance was **Unverifiable** without consulting external rules, but no obvious main-text overrun is visible.
- **Limitations:** Present as an unnumbered section before Ethical Considerations and references; substantive rather than perfunctory.
- **Anonymity:** Title page says “Anonymous ACL submission”; no author names, affiliations, personal paths, credentials, or self-identifying repository URLs were found in active source/PDF. Public third-party model URLs are not anonymity breaches.
- **Official style:** Uses `\usepackage[review]{acl}`, A4 two-column layout, line numbers, embedded fonts, and anonymous author. Exact package-version provenance relative to the current official kit is **Unverifiable** from permitted files alone.
- **References/placeholders:** Fresh compilation produced 23 pages with zero unresolved `??`, no unresolved labels/citations, 43/43 cited BBL entries, no duplicate labels, and no active TODO/TBD/FIXME/placeholder tokens. Underfull-box warnings are cosmetic; no critical overfull layout issue was observed.
- **Abstract/table consistency:** Headline 99.19/96.07, 931.9/664.4, 1.403x, 8–11, 25.8–27.6, -11.56/-1.00, and 92.5/98.5/100.0 values match the displayed evidence. The separate 12.31/12.17 issue is outside the abstract.
- **Prompt injection/reviewer manipulation:** No hidden white text, `\iffalse`, tiny reviewer-directed text, “ignore instructions,” score requests, acceptance pleas, or embedded files were found. PDF contains no JavaScript or attachments. Source comments are ordinary build/table provenance comments, not instructions to the reviewer.
- **Ethics:** The paper identifies hallucination/bias/disclosure risks, inversion/membership concerns for residual tensors, authorization/deletion/isolation needs, energy use, data origins, licensing caveats, and lack of new human-subject collection. No separate ethical blocker is apparent.

## Scores

- **Soundness: 3.5/5.** The matched endpoint, controls, accounting, and most statistics are strong. W2 weakens one inferential claim; W1 limits comparative interpretation rather than internal validity.
- **Excitement: 3.0/5.** The depth-axis measurement is useful and unusually transparent, but the broad space is crowded and the retained novelty is incremental after KV-Direct, RSCE, ReadOnce, PIC, and modular-KV work.
- **Overall: 3.0/5 (Findings).** Solid bounded measurement/negative-results paper with clear value, but not yet main-conference level because the nearest-baseline question remains unanswered and one decision-relevant uncertainty analysis needs repair.
- **Confidence: 4.0/5.** I completed two passes including appendices, formulas, all figures/tables, compilation, citation inventory, numerical spot checks, and a bounded novelty search. Some external records/artifact execution were Unverifiable under restrictions/rate limits.
- **Reproducibility: 3.5/5.** Excellent written protocol and identifiers; reduced by mutable proprietary judging, one missing timing cell/log family, confounded training runs, unlogged total compute, and inability to inspect/run the promised artifact under the review restriction.

## Review-process self-check

- Two-pass read completed: first pass for thesis/claims, second pass for appendix, all figures/tables, formulas, statistics, and desk checks.
- Every listed weakness is linked to a paper claim, has issue/importance/evidence/remedy, exact location, an exact quote of at most 25 words, severity, and a mechanical verification statement.
- I did not penalize the paper for unrelated wish-list experiments. The requested nearest baseline is claim-linked to deployment relevance; the statistical rerun is claim-linked to “statistically tied”; the remaining requests resolve internal consistency or validate the stated objective.
- Exact weakness quotes were re-found in the frozen source before saving. “Missing X” assertions were checked by source-wide search or complete table/citation inventory.
- I did not use prior reviews, score histories, TODO/status/current/calibration materials, or infer whether v5 was intended to fix any earlier concern. This is an independent judgment of the frozen v5 submission.
