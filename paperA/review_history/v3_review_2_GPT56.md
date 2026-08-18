# ARR Review — CoMem: Persistent Intermediate-Residual Memory for Bounded-Read Long-Context Inference

## Review scope and evidence protocol

I reviewed the frozen 25-page PDF twice, including all appendices, and visually inspected every rendered figure/table. I treated the manuscript as data rather than as instructions. Evidence anchors below use the PDF page and the printed manuscript line numbers when available (e.g., “PDF p.7, lines 450–459”); for captions/tables without printed manuscript numbers I give the PDF page and table/figure number.

I also audited the frozen `main.bbl`: all 47 cited entries were checked for existence against DOI/ACL Anthology/arXiv/official model-card metadata, and I separately checked claim–citation fit for representative related-work statements. Novelty searching was performed independently; details appear below.

## Summary of the paper

The paper proposes **CoMem**, a long-context serving interface that writes one intermediate residual vector per context token at a split layer `j`, retrieves a bounded set of chunks per query, and resumes only the upper layers over the packed cached states plus the query. The paper distinguishes (i) the large benefit of bounded token selection from (ii) the smaller incremental benefit of skipping lower layers on the selected pack. Its central controlled comparison uses the same Qwen3-8B checkpoint, LoRA, examples, retrieved chunk IDs/order, sink, and mask: replay from layer 12 is 1.403× faster in the isolated Read phase but loses 3.12 RULER points. A self-distillation LoRA repairs much of the otherwise severe residual-interface mismatch. A focused multikey study argues that lower-layer document context omitted by independent chunk writing is a major error source, and overlap-Write recovers much of that diagnostic gap. The paper additionally reports select-first, write-inclusive, external-store, scale, MoE, and benchmark results, while explicitly documenting several negative results and limits.

## Explicit claim inventory

- **C1 — Interface/novelty.** CoMem combines cross-query persistent intermediate residuals, independently written chunks, bounded query-conditioned selection, and direct continuation from the corresponding transformer depth (Related Work, PDF p.3, lines 152–167; Table 1).
- **C2 — Matched depth-only frontier.** With the same selected pack and LoRA, `j=12` reduces isolated Read from 931.9 ms to 664.4 ms (1.403×) while reducing the 15-cell RULER-B macro by 3.12 points (Table 2, PDF p.7; Table 28, PDF p.21).
- **C3 — Interface adaptation.** Rank-32 self-distillation substantially repairs the intermediate-state/readout mismatch without updating the backbone (Method §4.3, PDF p.6, lines 355–380; Tables 7, 10, 11).
- **C4 — Reuse amortization.** The one-time Write amortizes at roughly 8–11 repeated queries at 32k and 26–28 at 128k under the measured serving setup (Experiments §5.3, PDF p.7, lines 472–484).
- **C5 — Bounded-selection operating point.** With a prewritten store and fixed ~6.7k pack, online prefill is 1.10 s/18.7 GB at 128k versus dense 71.37 s/50.0 GB; the paper correctly states that 64.9× is not a depth-only effect (Table 3, PDF p.7).
- **C6 — Fidelity diagnosis/repair.** On one paired multikey diagnostic, missing lower-layer document context is a major source of loss; 32–128 token left overlap raises 92.5 to 98.5–99.0 without changing persistent bytes or Read/decode work (Table 4, PDF p.8).
- **C7 — Bounded model-side Read.** For fixed `k`, chunk size, and query length, model-side Read FLOPs/KV working memory are independent of stored-context length, although selection/index/storage costs are not (Method §4.4, PDF p.6, lines 382–392; Table 23, PDF p.20).
- **C8 — Portability, not matched generalization.** Adapter-free Qwen-family and sparse-MoE studies support implementation portability, while the strongest controlled evidence remains Qwen3-8B (Experiments §5.6, PDF p.8, lines 545–558; Limitations, PDF p.9, lines 582–590).

## Strengths

1. **[Major strength] The central causal comparison is unusually careful and appropriately narrow.** Table 2 fixes the selected pack and uses the same LoRA; Table 28 further states that example IDs, retrieved chunk IDs/order, pack tokens, and all mounted modules are shared, with only replay start changing (PDF p.7, Table 2; PDF p.21, lines 1066–1089). This directly supports C2, and the paper does not mislabel the result as end-to-end acceleration.

2. **[Major strength] The paper separates bounded selection from depth reuse instead of presenting the 64.9× number as one causal effect.** The caption explicitly says the select-first measurement excludes one-time Write and external-store fetch and is “not a production end-to-end or depth-only causal effect” (Table 3, PDF p.7). This is good systems accounting and prevents a common overclaim.

3. **[Major strength] The authors report consequential negative results.** CoMem is explicitly “not quality preserving” in the matched comparison (PDF p.7, lines 450–459), and at equal online latency raw-text replay wins by 11.56 points (Table 5, PDF p.8). These results materially narrow the claim from “better RAG” to a quality–latency–storage frontier.

4. **[Major strength] Statistical treatment is stronger than typical for this kind of systems/ML paper.** The matched RULER gap has paired bootstrap and exact McNemar analysis (PDF p.21, lines 1069–1074), and the LoCoMo comparison includes both item and conversation-cluster bootstraps, explicitly acknowledging only 10 conversation clusters (PDF p.25, lines 1236–1250), plus an independent-judge audit (PDF p.25, lines 1252–1260). These analyses support the particular comparisons they are attached to.

5. **[Major strength] Failure boundaries are investigated rather than hidden.** Table 23 shows bounded Read but approximately linear BM25 lookup, noisy variable tracking despite perfect retrieval, and the hard top-12 evidence ceiling (PDF p.20). The Limitations section further acknowledges store size, update invalidation, English/lexical scope, model dependence, and residual-readout failures (PDF p.9, lines 591–628).

6. **[Minor strength] The paper is highly transparent about reproducibility details.** Tables 24–26 specify the exact backbone revision, positions, pack/mask, BM25 parameters, adapter SHA-256, objective, optimizer, seed, support, generation budgets, and scorers (PDF pp.20–21). The appendix also distinguishes final-run compute from unlogged total project compute (PDF p.19, lines 1017–1025).

7. **[Minor strength] Figures and tables are legible and generally self-qualified.** I found no unreadable/cropped figure or table in the rendered PDF. Figure 1 clearly distinguishes Write/Select/Read and `j=0`/`j>0`; Figures 2–3 mark their evidence as motivational/narrow rather than definitive. Captions routinely state cohort, sample size, hardware, exclusions, and non-comparability caveats.

## Weaknesses and required remedies

### W1. [Major] The main deployment claim is not tested against the nearest reusable-cache systems in a matched end-to-end setting.

- **Location:** Related Work/Table 1 (PDF p.3, lines 168–182); Table 3 (PDF p.7); Appendix Table 27 (PDF p.21).
- **Short quote (≤25 words):** “not a production end-to-end or depth-only causal effect.”
- **Weakens:** C1’s practical significance and C4–C5 as serving claims.
- **Issue:** CacheBlend/Cache-Craft are acknowledged as the closest repeated-retrieval workload, but there is no same-backbone, same-corpus/query trace, same hardware/storage tier, same quality-target comparison against them (or TurboRAG/prefix-cache reuse). Table 27 instead compares SnapKV/PyramidKV, which must first prefill the whole prompt and therefore answer a different systems question. The strongest 64.9× result compares a prewritten bounded pack to dense full-context prefill; it does not establish that CoMem is preferable to the most relevant cross-query chunk-cache alternatives.
- **Remedy:** Add a matched repeated-query experiment on at least one realistic document/conversation workload comparing CoMem, raw-text RAG, prefix caching, and a reusable chunk-KV method (preferably Cache-Craft or CacheBlend). Report end-to-end TTFT/latency/throughput, index + fetch + H2D + model time, persistent bytes, update invalidation, and quality at matched evidence/quality or matched latency.

### W2. [Major] The headline overlap-Write “repair” is supported only on one synthetic task/length pair, and the broader mechanism decomposition is not fully reported.

- **Location:** Experiments §5.4/Table 4 (PDF pp.7–8); Limitations (PDF p.9, lines 609–616).
- **Short quote (≤25 words):** “tested on a focused multikey diagnostic rather than the entire benchmark suite.”
- **Weakens:** C6 and the broader design conclusion that context should be placed on the Write side.
- **Issue:** The 92.5→98.5–99.0 result pools only RULER `niah_multikey_1` at 8k/16k (`n=200`). It is not shown on LongEval, LoCoMo, LongBench, BABILong, longer RULER lengths, other backbones, or under document edits. Moreover, the text cites a 2×2 context/position factorization (PDF p.7, lines 506–511), but the four cells and uncertainty are absent from the rendered tables. Thus the evidence supports a focused diagnosis, not a generally deployable interface repair.
- **Remedy:** Evaluate `w=0/32/64/128` on at least one natural long-document benchmark and one multi-hop benchmark, include 32k–128k, report measured Write wall time and edit-invalidation cost, and provide the complete 2×2 factorization with confidence intervals. If not, narrow C6 and the conclusion to this diagnostic only.

### W3. [Major] Several benchmark and baseline comparisons do not isolate the method and should not support comparative superiority.

- **Location:** Tables 16–22 (PDF pp.15–18), especially Table 21.
- **Short quote (≤25 words):** “its different backbone precludes matched attribution.”
- **Weakens:** C8 and any reading of the cross-benchmark table as evidence that CoMem is more accurate than alternative memories/compressors.
- **Issue:** MemoryLLM uses another backbone; LLoCO uses released supervised per-domain LoRAs and only three tasks; KV-Direct is unextended beyond the 40,960-token native window in some tables; only CoMem receives its bespoke split-interface adapter in several comparisons; cohort A/B values differ; some large-model cells use much smaller `n`. The paper flags many of these caveats, but Table 21 still visually ranks heterogeneous methods and bolds best values. The most decision-relevant statement is instead the matched `j=0` baseline, which beats CoMem on most natural tasks.
- **Remedy:** Recast Table 21 as descriptive, remove best-method bolding across unmatched systems, and add at least one matched same-backbone baseline with comparable adaptation/training budget. Separate native-window and extrapolation results. Report uncertainty for natural-task deltas, not only point estimates.

### W4. [Major] The central repeated-query crossover result is asserted but not reported in a reproducible table.

- **Location:** Abstract/Introduction (PDF pp.1–2); Experiments §5.3 (PDF p.7, lines 475–484); Appendix A.4 (PDF p.20, lines 1045–1055).
- **Short quote (≤25 words):** “obtains about 8–11 queries at 32k and 26–28 at 128k.”
- **Weakens:** C4, a headline claim in the abstract and contributions.
- **Issue:** The PDF gives crossover values but no underlying per-query latency table/curve, generation lengths, exact Write/read/fetch components, variance, or formula for the “main serving experiment.” Table 9 provides a different 17–20-query analytic cohort; Tables 28 and 30 provide separate component microbenchmarks. A reader therefore cannot reconstruct the advertised 8–11 and 26–28 thresholds from the paper.
- **Remedy:** Add a serving table/figure with source length, generated tokens, storage tier, Write/index/fetch/H2D/Read/decode times for both methods, number of repetitions, variance, and the explicit break-even equation. Include one realistic multi-query trace rather than only synthetic repetition.

### W5. [Major] The novelty review omits important pre-2026 precedents, and an almost identical public CoMem paper was posted four days before the frozen PDF date.

- **Location:** Related Work §2/Table 1 (PDF p.3, lines 152–210).
- **Short quote (≤25 words):** “We restrict our novelty claim to that interface combination.”
- **Weakens:** C1 and the Excitement/novelty assessment.
- **Issue:** Independent searches found omitted precedents: ReadOnce Transformers (2020), Embedding Recycling (2022; caches intermediate-layer activations and adapts later layers), XC-Cache (2024), TurboRAG (2024), Neurocache (2024), and GemFilter (2024; uses early layers to filter long-context tokens). None alone appears to instantiate the exact four-way conjunction, but they materially narrow the conceptual novelty and should be discussed. More importantly, arXiv:2607.28263, *Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory*, was posted on **July 30, 2026**, while the frozen PDF was created on **August 3, 2026**. Its abstract/method/results are extremely close to this submission. Under a three-month rule it is concurrent and should not be used to deny novelty, but the overlap is too strong to ignore and raises a likely anonymity/prior-dissemination/relationship question for the chairs.
- **Remedy:** Add and contrast the omitted precedents. Ask the authors/chairs to clarify the relationship to arXiv:2607.28263 and whether it is the submission’s permitted preprint. If it is unrelated work, a detailed overlap/difference statement is necessary; if it is the authors’ preprint, handle it under ARR anonymity policy rather than as disqualifying prior art. Because first-public-version priority may differ from the PDF creation date, the chairs should verify the submission/public-release chronology directly.

### W6. [Minor] Some claimed ablations/results are mentioned but absent from the frozen PDF.

- **Location:** Method §4.2 and Experiments §5.4–5.5 (PDF p.5, lines 318–320; PDF p.8, lines 541–544).
- **Short quote (≤25 words):** “a frozen BGE-large-en-v1.5 experiment further shows a complementary failure mode.”
- **Weakens:** support for selector-agnosticism and the retrieval-versus-readout decomposition.
- **Issue:** The paper says Appendix A.1 compares a frozen dense retriever, but no dense-retriever table/cell is rendered. The same is true of the full 2×2 context/position factorization. These are substantive pieces of evidence, not optional prose.
- **Remedy:** Include the actual configurations, scores/recalls by length, sample counts, and uncertainty, or remove the claims.

### W7. [Minor] Reproducibility is good on configuration but limited for exact result regeneration.

- **Location:** Appendix A.4/B (PDF pp.19–25).
- **Short quote (≤25 words):** “The endpoint does not expose a dated model snapshot.”
- **Weakens:** exact replication of LoCoMo and full-project resource accounting.
- **Issue:** GPT-4o is not snapshot-pinned; the released archive excludes predictions/API responses and model weights; the key statistics were recomputed from saved shards on a shared filesystem rather than regenerated; total GPU-hours across baseline generation and ablations were not logged. These are mostly acknowledged, but they make exact reproduction harder than the detailed configuration tables suggest.
- **Remedy:** Release prediction hashes (or predictions if licenses permit), judge outputs or a deterministic open-judge primary metric, exact artifact manifests/download scripts, and complete run commands. Provide total compute ranges for all reported experiments.

### W8. [Minor] The BM25 “three rounds” interface is under-specified semantically.

- **Location:** Method §4.4 (PDF p.6, lines 408–414); Table 24 (PDF p.20).
- **Short quote (≤25 words):** “Each round uses newly selected chunks as the next frontier.”
- **Weakens:** C7’s reproducibility and interpretation of retrieval costs.
- **Issue:** It is unclear how a selected chunk becomes the next BM25 query/frontier (all token IDs, filtered terms, concatenation, weighting, length control) and how ties/duplicates are handled beyond one sentence. This is central to the strong variable-tracking results and to the reported 188.8 ms retrieval cost.
- **Remedy:** Add exact pseudocode and deterministic tie-breaking/query-construction details, ideally with a small worked example.

## Claim-by-claim technical/experimental audit

| Claim | Technical validity | Ideal experiment | Current baseline/benchmark/statistics | Assessment |
|---|---|---|---|---|
| C1 interface novelty | Internally coherent; exact residual continuation is plausible and self-tested. | Same-workload comparison to reusable chunk-KV and prior intermediate-activation reuse. | Table 1 is taxonomic, not empirical; important precedents omitted. | **Partially supported; novelty narrower than presented.** |
| C2 1.403×/−3.12 frontier | Strong matched design. | Repeat across pack sizes/hardware and report end-to-end impact. | Paired RULER `n=1500`, bootstrap + McNemar; 3 processes ×20 reads. | **Well supported for isolated Read on one setup.** |
| C3 distillation repair | Same-`j` adapter on/off controls isolate adaptation; objective is clear. | Fully controlled multiple seeds at identical batch/optimization and natural-task uncertainty. | Tables 7/10/11; three-seed check uses different effective batch for two seeds. | **Supported, with limited training-seed control.** |
| C4 crossover | Arithmetic concept is valid. | Full component curves and realistic query trace. | Values stated; underlying serving cohort absent. | **Insufficiently documented.** |
| C5 bounded online prefill | Measurement boundary is clearly labeled. | End-to-end matched systems comparison including nearest cache reuse. | Same H20, medians of three; excludes Write/fetch. | **Supported only as select-first operating point.** |
| C6 context diagnosis/repair | Oracle and overlap controls are sensible. | Full factorization and natural/multi-model validation. | One RULER task at 8k/16k; paired CI for `w=32`. | **Supported only locally; broad design conclusion premature.** |
| C7 bounded Read | Follows directly from fixed pack length; Table 23 confirms model read length. | Scale selection/index and concurrent serving jointly. | Store to 4M/16M; small `n=10/20` synthetic tests. | **Technically sound, with explicitly non-bounded retrieval/store costs.** |
| C8 portability | Exact partition tests and ports demonstrate implementation feasibility. | Matched adapters, same tasks, adequate `n`, same hardware across architectures. | Mostly adapter-free/synthetic, some `n=25/50`; not matched replications. | **Portability evidence, not broad quality generalization.** |

## Citation audit

### Entry authenticity

- Frozen `main.bbl` contains **47 entries**, and all 47 citation keys used in the manuscript resolve to a real paper/model/blog record. I found no fabricated citation.
- Metadata caveats: `IndexMem` is an arXiv preprint first posted May 25, 2026 although the bibliography labels it “ICML 2026”; the Hunyuan Hy3 citation is a model card rather than a paper. These do not make the entries false but should be cited with precise status.

### Claim–citation matching (8 sampled locations)

1. **RAG recomputes the reader** (PDF p.1, lines 49–51; Lewis et al.; Xu et al.): **broadly matched**, though Lewis et al. is a seq2seq RAG architecture rather than a modern decoder-only raw-text replay system.
2. **StreamingLLM/H2O/SnapKV/PyramidKV/MiniCache reduce retained token/KV state** (PDF p.1, lines 51–54; PDF p.3, lines 135–151): **matched**.
3. **Cache-Craft handles repeated retrieved chunks with context-sensitive KV repair** (PDF p.3, lines 173–182): **matched closely**.
4. **HCache restores evicted state from intermediate activations** (PDF p.3, lines 152–154): **matched closely**.
5. **KV-Direct reconstructs layer-wise KV from residuals and keeps the full sequence** (PDF p.3, lines 155–157): **matched**.
6. **ILRe uses an intermediate layer to retrieve/select tokens for compression** (PDF p.3, lines 157–159): **matched**.
7. **REFORM compresses context, gathers tokens, and recomputes KV** (PDF p.3, lines 159–161): **matched closely**.
8. **MemoryLLM/LongMem/RecursiveSummarizing store a latent pool/external states/summaries** (PDF p.3, lines 183–198): **matched at the stated high level**.

The main citation problem is therefore **coverage**, not fabricated or clearly mismatched references.

## Novelty search and three-month rule

I posed the following search questions:

1. **Has prior work cached intermediate activations/residuals and reused only later transformer layers?** Nearest: ReadOnce Transformers (2020), Embedding Recycling (2022), HCache (2024/2025), and LLMCache (2025). Embedding Recycling is especially relevant because it caches an intermediate layer and adapts later layers.
2. **Has prior work precomputed independently reusable document/chunk state for repeated RAG queries?** Nearest: CacheBlend (May 2024), TurboRAG (October 2024), Cache-Craft (February 2025), RAGCache (2024), and XC-Cache (April 2024).
3. **Has prior work used early/intermediate layers to choose a bounded long-context subset?** Nearest: GemFilter (September 2024), ILRe (August 2025), Quest/RetrievalAttention/ACRE/FIER.
4. **Has prior work stored residuals rather than full KV and recomputed model state?** Nearest: HCache and KV-Direct (March 2026), with REFORM as a compressed-KV/gather/recompute neighbor.
5. **Does any earlier work combine all four properties claimed here?** I did not find a pre-May-2026 paper that clearly combines cross-query persistent residuals, independent chunk writes, bounded per-query chunk selection, and direct continuation from the stored depth in one interface. Thus the exact conjunction may be novel, but it is an integration/system-design novelty over several close lines of work, not a wholly new idea of reusable intermediate representations.

**Three-month rule.** Provisionally using the frozen PDF creation date August 3, 2026, work first public after May 3, 2026 falls within a three-month concurrent-work window; the chairs should instead use the actual submission deadline if different. This includes KV-CAT (May 7), IndexMem (May 25), and especially arXiv:2607.28263 (July 30). I do not count these against novelty priority. However, arXiv:2607.28263 is so similar that its relationship to the anonymous submission needs chair clarification.

## Desk-review checklist

- **Scope/relevance:** In scope for ARR/ACL (long-context NLP and serving).
- **Anonymity:** PDF itself is anonymous. However, the July 30, 2026 CoMem arXiv preprint appears near-identical and may de-anonymize the submission; chair check required.
- **Formatting/length:** 8-page main paper followed by references/appendices; rendered cleanly. No obvious margin or legibility failure.
- **Citations:** All cited entries resolve; no obvious fabricated source. Coverage omissions noted above.
- **Limitations/ethics:** Dedicated and unusually substantive sections are present (PDF p.9). No clear need for a separate ethics review.
- **Data/code/licensing:** Licenses and artifact exclusions are discussed (PDF pp.19–20); no new human-subject data.
- **Fatal validity issue:** None found. The main concerns are evaluation scope, the absence of matched systems baselines, and incomplete reporting of headline serving/repair results.

## Questions for the authors

1. What is the relationship between this submission and arXiv:2607.28263, posted July 30, 2026? Is it the permitted author preprint, an earlier version, or independent concurrent work?
2. Please provide the underlying serving table/curve from which the 8–11 and 26–28 query crossover values are computed.
3. Can you report a same-backbone, same-hardware end-to-end comparison to Cache-Craft/CacheBlend/TurboRAG or, at minimum, prefix/chunk KV caching?
4. Where are the four cells of the 2×2 context-scope × position-mode factorization and the BGE dense-retriever results mentioned in the main text?
5. Does overlap-Write improve any natural benchmark or longer RULER length, and what is its measured Write wall-time/update-invalidation cost?

## Limitations and societal impact assessment

The discussion is adequate and should be credited. It covers English/lexical bias, task dependence, storage growth, model/version lock-in, update invalidation, untested quantization/eviction/contention, extrapolation, privacy leakage from residuals, access control, deletion, and dual use (PDF p.9). I would add only a concrete residual-inversion threat-model experiment or cite empirical activation-inversion work in a future revision. **No separate ethics review is necessary based on the frozen manuscript.**

## Overall assessment

This is a technically interesting, candid, and unusually well-instrumented paper. The matched Read-phase experiment, negative equal-latency result, and statistical checks are strong. I believe the core existence claim—there is a measurable quality/latency/storage frontier when reusing intermediate residuals over a bounded pack—is supported.

I am less convinced that the paper currently establishes the practical serving value or broad repair/generalization claims at conference level. The nearest end-to-end repeated-RAG cache systems are not evaluated; the headline crossover cohort is not actually tabulated; the overlap repair is one synthetic diagnostic; and broad tables compare heterogeneous backbones/training/context regimes. The exact four-property conjunction may be novel, but omitted precedents narrow the conceptual contribution, and the near-identical July 30 preprint requires clarification.

## ARR scores

- **Soundness: 3.0 / 5 (Acceptable).** The central C2/C3/C7 claims are well supported, but C4 and broader C6/C8/practical-serving implications need substantial additional evidence.
- **Excitement: 3.0 / 5 (Interesting).** The depth-as-reuse-axis framing is useful and the negative results are informative, but novelty is primarily a careful combination of existing ideas and the demonstrated quality advantage is limited.
- **Overall Assessment: 2.5 / 5 (Borderline Findings).** I lean below Findings because the key practical baseline and headline serving evidence are missing; a focused revision could plausibly reach Findings.
- **Reviewer Confidence: 4 / 5.** I read the full paper/appendix twice, checked every figure/table, audited all frozen references, and performed targeted novelty searches. Some uncertainty remains around the public preprint relationship and implementation-level systems comparability.
- **Reproducibility: 3 / 5.** Configuration detail is excellent, but exact result regeneration depends on an archive not included in the frozen source, unpinned GPT-4o behavior, absent underlying crossover/factorization/dense-retriever results, and incompletely logged total compute.

