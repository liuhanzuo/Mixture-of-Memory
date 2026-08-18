review_mode: strict
soundness: 3.0
excitement: 3.5
overall: 3.0
confidence: 4.5
reproducibility: 2.5

# Paper Summary

CoMem studies repeated queries over stable documents by precomputing one residual vector per token after decoder layer $j$, retrieving a bounded set of chunks, and executing only layers $[j{:}L)$ online. Its cleanest result is a matched Qwen3-8B comparison between $j=0$ raw-token replay and $j=12$ residual replay on exactly the same selected pack: selected-pack Read latency falls from 931.9 to 664.4 ms (1.403x), while the 15-cell RULER-B macro falls from 99.19 to 96.07. The paper also reports storage/amortization measurements, a continuous-prefix oracle, a context-position factorization and overlap-Write repair, an equal-latency selector audit, a minimal CacheBlend-style control, natural-task evaluations, and exploratory cross-scale/MoE ports.

I find the controlled depth-axis formulation useful and the paper unusually candid about scope and negative results. However, the practical system claim is supported by fragmented operating points rather than one matched end-to-end deployment frontier; the flagship adapter has no clean replicated headline evaluation; and the CacheBlend-style comparison is not faithful enough to sustain the paper's repeated quantitative contrast to CacheBlend. I therefore place the work at Findings level rather than ACL main level.

# Claims and Evidence Map

- **C1 — Tunable depth-reuse interface:** one persistent $h_j$ per token can be selected and resumed through the native suffix. **Evidence:** Sec. 4, PDF pp. 5--6, lines 314--389; Eq. 1; exact split tests in Appendix A.9.1, PDF p. 26, lines 1268--1275. **Assessment:** supported as an interface/implementation claim.
- **C2 — Matched depth trade-off:** $j=12$ gives 1.403x faster selected-pack Read at a 3.12-point RULER cost. **Minimum sufficient experiment:** same model, adapter, examples, evidence IDs/order, mask, positions, generation protocol, repeated latency processes, and paired uncertainty. **Actual evidence:** Table 2, PDF p. 7; Appendix Table 34, PDF p. 25; $n=1{,}500$, paired CI and three process medians. **Assessment:** well supported within its explicitly narrow Read boundary.
- **C3 — Upper suffix has adequate capacity; loss is at the reusable Write interface:** continuous-prefix $h_{12}$ recovers $j=0$. **Minimum sufficient experiment:** jointly computed lower-layer states with an otherwise identical upper suffix. **Actual evidence:** Table 2 and Appendix Table 35, PDF pp. 7 and 25. **Assessment:** supported for the tested pack/task cohort; not a general causal localization across tasks.
- **C4 — Missing lower-layer document context dominates the tested multikey gap; overlap repairs it:** **Minimum sufficient experiment:** factorial context scope x position with fixed retrieval/reader, followed by overlap dose response and uncertainty. **Actual evidence:** Tables 6--7, PDF p. 8; $n=200$, one CI for $w=32$. **Assessment:** supported for the stated 8k/16k RULER multikey cohort, not for natural tasks.
- **C5 — Favorable storage and repeated-query amortization:** 8 KiB/token and measured break-even counts. **Minimum sufficient experiment:** matched Write/index/fetch/Read/decode boundaries across storage tiers, repetitions, source lengths, and generations. **Actual evidence:** Eq. 1; Table 4, PDF p. 8; released 18 records for this table. **Assessment:** arithmetic and the table-specific measurements are supported, but general deployment conclusions remain workload- and hardware-specific.
- **C6 — Large practical speedups (64.9x store-ready; 2.74x Write-inclusive):** **Minimum sufficient experiment:** a single matched system/hardware/model/adapter cohort reporting full lifecycle latency, quality, throughput/tails, and break-even. **Actual evidence:** separate H20 and L20A cohorts with different adapters and exclusions (Sec. 5.5, PDF pp. 7--8; Tables 8 and 31). **Assessment:** the individual ratios are documented, but they do not establish one end-to-end deployment frontier.
- **C7 — CoMem outperforms a same-Qwen3 CacheBlend-style full-depth KV alternative while storing 18x fewer bytes:** **Minimum sufficient experiment:** faithful published CacheBlend token-selection/recompute algorithm or authors' implementation, identical evidence/protocol, quality plus online latency/TTFT and storage. **Actual evidence:** Table 3, PDF p. 7, and a custom baseline snapshot. **Assessment:** storage arithmetic is sound; the method comparison is weakened by algorithmic and systems differences discussed in W1.
- **C8 — Quality advantage is selector-dependent at equal latency:** **Minimum sufficient experiment:** disjoint calibration, matched hardware/timing, multiple selectors, task-cell-aware uncertainty. **Actual evidence:** Table 5 and Appendix Table 9, PDF pp. 8 and 15. **Assessment:** supported as a diagnostic; the paper correctly does not claim selector-independent superiority.
- **C9 — Broad transfer/generality:** **Evidence:** LongEval, LongBench, BABILong, LoCoMo, cross-scale and Hy3 tables. **Assessment:** interface portability is demonstrated, but quality preservation is uneven and the strongest cross-scale/MoE evidence is exploratory, as the paper itself acknowledges.

# Strengths

## S1. The central experiment isolates the claimed variable unusually well.

Table 2 holds selected chunks, order, sink, mask, examples, and LoRA fixed and changes only replay start. The paper further separates Read from decode and reports paired quality uncertainty and process-level latency stability (Sec. 5.1, PDF p. 6, lines 406--421; Appendix A.6/Table 34, PDF pp. 24--25). This is the paper's strongest and most credible contribution.

## S2. The paper reports boundaries and negative results instead of obscuring them.

Examples include the large LongEval loss (97.2 to 69.0), the modest total Read+decode speedup (~1.07--1.09x), BM25 replay's robust 11.56-point equal-latency lead, the non-finite crossover cell, retrieval coverage failures, and contamination caveats (Sec. 5, PDF pp. 6--8; Limitations, PDF p. 9; Appendix A.4, PDF pp. 20--21). This materially improves interpretability.

## S3. The context-position diagnosis is well targeted to a claim-relevant failure mode.

The $2\times2$ intervention in Table 6 and overlap sweep in Table 7 directly test whether independently written chunks fail because of lower-layer context or position remapping (PDF p. 8). The conclusion is appropriately narrowed to the displayed multikey cohort.

## S4. Systems accounting is more careful than usual.

The paper distinguishes selected-pack Read, store-ready prefill, Write-inclusive pipeline, external store I/O, and repeated-query break-even rather than silently combining incompatible denominators (Sec. 5.5, PDF pp. 7--8; Appendix A.6, PDF pp. 22--25). Eq. 1's residual/full-KV ratio is correct for Qwen3-8B GQA.

## S5. The appendices are comprehensive and visually legible.

All 28 pages, 40 tables, and Figures 1--2 were inspected. The PDF has no clipped figures/tables or unresolved references. Appendix Tables 27--29 specify masks, positions, model revision, adapter hash, optimizer, evaluation supports, generation limits, and scoring (PDF pp. 22--23).

## S6. The novelty claim is mostly narrow and defensible.

The paper does not claim invention of activation reuse; it claims the conjunction of one chosen document residual split, direct suffix continuation, and an identical-evidence $j=0$ endpoint as a measured serving axis (Related Work, PDF pp. 3--4, lines 171--283). The closest older work I found does not present this full measurement framework.

# Weaknesses

## W1. The quantitative CacheBlend contrast is not based on a faithful enough CacheBlend implementation. **Major**

- **Location:** Sec. 5.2, PDF p. 6, lines 432--457; Table 3, PDF p. 7; Appendix A.6, PDF p. 24, lines 1174--1190; supplied artifact `artifacts/cacheblend_143/CACHEBLEND_IMPL_NOTES.md`, lines 29--40.
- **Exact quote (15 words):** “Table 3 compares CoMem with a minimal faithful CacheBlend-style arm on the same Qwen3-8B backbone.”
- **Problem:** Published CacheBlend recomputes selected tokens on each layer and uses gradual per-layer filtering based on attention/KV deviation. The supplied implementation instead performs a full layer-0 bootstrap, chooses a set, and forwards that subset through later layers; it also forces sink/query tokens into recomputation and omits the native loader/fusor scheduler. Consequently, $r$ is not the same operational recomputation policy as CacheBlend, and the very low 67.80--74.70 scores cannot safely be interpreted as a CacheBlend quality result. The paper acknowledges omission of the scheduler, but repeatedly calls the arm “minimal faithful” and uses its scores in the abstract, conclusion, and novelty argument.
- **Affected claim/norm:** C7 and fair-baseline validity. This matters because Table 3 is the sole same-backbone empirical comparison to the nearest full-depth chunk-KV family and is one of the abstract's headline results.
- **Sufficient remedy:** Reimplement the published gradual per-layer HKVD selection/recompute algorithm (or use authors' code), validate intermediate selected-token traces and $r=1$, and report quality **and** online latency/TTFT at matched storage/evidence. Otherwise rename the arm as a custom fixed-subset selective-KV baseline and remove all numerical claims about CacheBlend itself.

## W2. The practical acceleration evidence is fragmented; no single experiment establishes an end-to-end, quality-matched deployment frontier. **Major**

- **Location:** Abstract, PDF p. 1, lines 17--22; Sec. 5.5, PDF pp. 7--8, lines 519--560; Limitations, PDF p. 9, lines 623--627.
- **Exact quote (20 words):** “a store-ready 128k online-prefill point is 64.9x faster than dense prefill, while a separate same-adapter, Write-inclusive pipeline is 2.74x faster.”
- **Problem:** The 64.9x point compares stock dense Qwen3-8B on H20 against adapted CoMem, excludes Write/store fetch/generation, and is not quality-matched. The 2.74x L20A cohort includes Write but excludes index construction/external I/O and does not report matched task quality. The break-even grid is a third harness and one example/task. Throughput, concurrency, and p95/p99 are absent. Although boundaries are disclosed, the headline juxtaposition invites a practical-system conclusion that no single matched cohort supports.
- **Affected claim/norm:** C6 and systems-evaluation validity. A repeated-query serving paper needs at least one coherent full-lifecycle frontier tying latency to quality under the same model, adapter, hardware, selector, storage placement, and workload.
- **Sufficient remedy:** Add one end-to-end experiment comparing $j=0$ and CoMem on the same hardware/model+adapter with index build, Write, fetch, Read, generation, measured task quality, repeated-query distributions, throughput/tails, and $Q$-dependent amortized latency. Keep 64.9x only as a component microbenchmark.

## W3. The flagship quality claims are effectively single-run; the available robustness runs confound seed with batch and do not cover the headline cohorts. **Major**

- **Location:** Limitations, PDF p. 9, lines 583--590; Appendix Table 30, PDF p. 23; Appendix A.7, PDF p. 25, lines 1221--1230.
- **Exact quote (21 words):** “The flagship adapter is one batch-8 run; two matched-data runs use effective batch 3, so they conflate seed and optimization noise.”
- **Problem:** The exact 15-cell RULER-B gap, LongEval, LoCoMo, overlap repair, and equal-latency headline do not have clean same-hyperparameter training-seed replication. The reduced-support three-run table changes effective batch from 8 to 3 and therefore cannot estimate seed variance. Paired example CIs quantify evaluation sampling conditional on one trained adapter, not training uncertainty.
- **Affected claim/norm:** C2--C4, C8 and empirical reliability. The method depends critically on a 58.2M-parameter learned interface; one favorable training run can materially affect all quality conclusions.
- **Sufficient remedy:** Train at least three adapters with identical effective batch, schedule, examples seen, and all hyperparameters; evaluate the exact headline RULER-B, LongEval, LoCoMo, and overlap/equal-latency cohorts; report run-level mean/SD or hierarchical intervals.

## W4. The supplied artifact snapshot is insufficient to reproduce most headline claims and includes non-anonymous internal planning/provenance material. **Major**

- **Location:** Ethical Considerations, PDF pp. 9--10, lines 678--685; Appendix A.5, PDF p. 21, lines 1065--1110; supplied artifact directories.
- **Exact quote (22 words):** “it provides the adapter, evaluation code, permissible prediction artifacts or hashes, timing records, licenses, and provenance needed to audit the reported results.”
- **Problem:** The frozen source package supplied for review contains only the manuscript plus two narrow artifact snapshots (Table 3's custom baseline and Table 4's serving grid). It does not contain the advertised CoMem implementation, adapter, environment/requirements, flagship predictions, scoring scripts, LoCoMo decisions, or source for most tables. The CacheBlend aggregator cannot run standalone because required evaluator modules are absent; its full 1,733-file raw tree is omitted. Internal files expose `Paper A`, task numbers, TODO/status references, project-relative model/output paths, and a git commit, which are unsuitable anonymous supplementary material.
- **Affected claim/norm:** reproducibility, artifact auditability, and anonymity. Hashes establish integrity of present files, not reproducibility of absent experiments.
- **Sufficient remedy:** Release a clean anonymous archive containing runnable code, pinned environment, adapter or stable anonymous download, configs, allowable score/prediction exports, exact aggregation scripts, and CPU smoke tests for every headline table; remove planning notes, task-board references, internal paths, and identifying repository metadata.

## W5. The paper's “tunable depth axis” is not a controlled multi-depth frontier because each nonzero split has a separately trained, differently sized adapter. **Minor**

- **Location:** Sec. 3, PDF pp. 4--5, lines 292--305; Appendix Table 13, PDF p. 16.
- **Exact quote (14 words):** “Each split uses a separately distilled suffix adapter, yielding a practical multi-depth deployment curve.”
- **Problem:** Adapter spans and parameter counts vary with $j$, and the $j=12$ Write value is missing. Thus the curve entangles split depth, adapter capacity, and training outcome; only $j=0\rightarrow12$ is a clean matched measurement, as the paper admits.
- **Affected claim/norm:** C1's stronger tunability interpretation. The interface is tunable in design, but the empirical curve is not a causal depth sweep.
- **Sufficient remedy:** Train a controlled adapter family with matched parameter/compute budgets (or one multi-depth adapter) and report Write/Read/quality/storage for every $j$ under identical seeds and data.

## W6. The novelty audit omits a close intermediate-state restoration line and needs a more explicit temporal cutoff. **Minor**

- **Location:** Related Work/Table 1, PDF pp. 3--4; abstract novelty sentence, PDF p. 1, lines 6--12.
- **Exact quote (16 words):** “Among document-reuse systems we are aware of, CoMem jointly makes split depth a tunable serving axis”
- **Problem:** Searches for residual/hidden-state cache restoration surfaced **RSCE: Accelerating Large Language Model Decoding with Cache-Compressed Context** (Findings ACL 2025), which is close enough to discuss explicitly alongside HCache/KV-Direct. Conversely, SemPIC first appeared July 30, 2026, only five days before this frozen August 4, 2026 manuscript, and should be labeled contemporaneous rather than used to narrow novelty retrospectively.
- **Affected claim/norm:** novelty positioning and complete related work.
- **Sufficient remedy:** Add RSCE to Table 1/Related Work, state the literature-search cutoff, and mark work first public within three months as contemporaneous. The likely surviving novelty is the matched depth-axis measurement, not intermediate residual storage itself.

# Questions That Could Change the Score

1. Can the authors provide a faithful CacheBlend gradual-filtering implementation and show whether Table 3's quality ordering survives at matched $r$, latency, evidence, and storage? A reversal or major narrowing would materially change my assessment.
2. Can the authors report a single same-hardware, same-adapter, quality-matched end-to-end repeated-query comparison, including Write/index/fetch/Read/decode and throughput/tails? Strong results here could raise Overall.
3. Do three same-effective-batch adapter seeds preserve the exact 3.12-point RULER-B gap, LongEval loss, LoCoMo result, and overlap repair? Large run-to-run variation would lower Soundness.
4. Will the submission artifact actually contain the implementation, adapter, exact configs, and permissible score/prediction exports promised in Appendix A.5, rather than only the two snapshots reviewed here?
5. What was the paper's first public/submission date? This determines which 2026 works should be treated as prior versus contemporaneous under the three-month rule.

# Non-scoring Suggestions / Typos

- Table 2's continuous-prefix oracle lists an 8,192 B/token “stored object” although it is explicitly recomputed per query and is “not a reusable cache” in Appendix Table 35; use “hypothetical object size” or leave storage blank.
- State prominently in Table 3 that CacheBlend $r$ and CoMem's suffix recomputation are different compute quantities; add latency/TTFT columns or avoid a winner framing.
- Report the teacher top-64 captured probability mass; without it, the distillation objective's truncation severity is unknown (Sec. 4, PDF p. 5, lines 360--366).
- The abstract contains substantially more than 3--5 numbers and is difficult to parse; retain the matched 1.403x/3.12-point result and one deployment boundary, moving the rest to the body.
- Clarify whether “128k” consistently means 128,000 or 131,072 tokens; the storage arithmetic uses the latter convention.
- Add training peak memory and total experimental compute if logs can be reconstructed; currently only the final adapter run is accounted.
- The LoCoMo judge should use a dated immutable snapshot if available; saved responses mitigate but do not eliminate judge drift.

# Scores

## Soundness: 3.0 / 5.0

The central matched depth experiment is sound and carefully scoped, and several diagnostics are strong. The score is limited by the non-faithful nearest-baseline implementation, lack of clean training replication, and fragmented systems evidence.

## Excitement: 3.5 / 5.0

Treating split depth as an explicit reusable-context serving coordinate is useful, and the paper contributes a good measurement framework plus informative failure analysis. The core mechanism is conceptually simple and adjacent to several activation/hidden-state/KV reuse lines, so novelty is meaningful but not transformative.

## Overall: 3.0 / 5.0

**Findings-level.** I am genuinely between 3.0 and 3.5: the paper is substantial, transparent, and contains one unusually clean result, but ACL-main calibration requires a credible nearest-family comparison, replicated learned-interface quality, and one coherent end-to-end frontier. Per the requested calibration, I choose the lower bin because these are claim-linked empirical deficiencies rather than presentation preferences.

## Confidence: 4.5 / 5.0

I inspected the complete frozen PDF twice, all appendices, every rendered figure/table, all supplied source/artifacts, formulas, claims, and bibliography metadata; I also checked the published CacheBlend algorithm and performed targeted novelty searches. Residual uncertainty concerns unavailable full artifacts and the paper's first-public date.

## Reproducibility: 2.5 / 5.0

The paper text specifies the flagship protocol in impressive detail and the two supplied snapshots have valid manifests; Table 4 can be mechanically regenerated. However, the provided frozen package lacks runnable CoMem code, adapter, environment, predictions, and aggregation material for most headline claims, and the CacheBlend snapshot is not standalone.

# Limitations / Ethics and Desk-Reject Risks

- **Limitations section:** present under the exact heading “Limitations” on PDF p. 9 and unusually comprehensive.
- **Ethics:** present and substantive; it covers hallucination/bias, sensitive cached text, possible inversion/membership risks, authorization/encryption/deletion, energy, and data redistribution.
- **Main-text length:** numbered main content ends on PDF p. 8; Limitations/Ethical Considerations begin on p. 9. This appears compliant with the eight-page long-paper body convention used by the manuscript.
- **Style/anonymity:** PDF is in ACL review style with line numbers and “Anonymous ACL submission”; no author metadata, hidden/white text, prompt injection, unresolved references, TODOs, or reviewer manipulation were found in the manuscript. All fonts are embedded.
- **Artifact anonymity risk:** if the supplied artifact directory is uploaded as supplementary material, internal planning files and repository/task references are a desk/anonymity risk (W4). A sanitized archive is needed.
- **Citation timing risk:** several references are 2026 preprints, including one only five days old at freeze time. This is not a desk reject, but the camera-ready/submission must apply the venue's contemporaneous-work policy consistently.
- **No other obvious desk-reject issue** was found in the frozen PDF.

# Citation Audit

## Complete bibliography-entry verification

Status meanings: **Verified** = identity/metadata matched a DOI, arXiv record, ACL/venue record, or official page; **Metadata error** = work exists but the rendered entry is materially incomplete/inaccurate; **Unverifiable** = network/index lookup was inconclusive, not “not found.” No entry is marked Not found merely because a lookup failed.

| # | Key | Status | Note |
|---:|---|---|---|
| 1 | cachecraft | Verified | DOI 10.1145/3725273. |
| 2 | longbench | Verified | ACL 2024 DOI and pages match. |
| 3 | llmcache | Verified | arXiv:2512.16843, first posted 2025-12-18. |
| 4 | pyramidkv | Verified | arXiv:2406.02069; CoLM 2025 venue is plausible/consistent. |
| 5 | kvpacket | Verified | arXiv:2604.13226, first posted 2026-04-14. |
| 6 | cartridgesbase | Verified | arXiv:2506.06266; ICLR 2026 metadata matched. |
| 7 | hcache | Verified | EuroSys 2025 DOI 10.1145/3689031.3696072. |
| 8 | promptcache | Verified | MLSys 2024, vol. 6, pp. 325--338; arXiv:2311.04934 also matches. |
| 9 | llama3 | Verified | arXiv:2407.21783. |
| 10 | cartridges | Verified | arXiv:2606.04557, first posted 2026-06-03. |
| 11 | distillation | Verified | arXiv:1503.02531. |
| 12 | ruler | Verified | arXiv:2404.06654 / COLM 2024. |
| 13 | lora | Verified | arXiv:2106.09685 / ICLR 2022. |
| 14 | epic | Verified | arXiv:2410.15332 / ICML 2025. |
| 15 | ragcache | Verified | DOI 10.1145/3768628. |
| 16 | babilong | Verified | NeurIPS 2024 DOI and pages match. |
| 17 | rag | Verified | NeurIPS 2020 identity and authors match. |
| 18 | longchat | Verified | Official LMSYS blog entry dated 2023-06-29. |
| 19 | snapkv | Verified | NeurIPS 2024 DOI and pages match. |
| 20 | ilre | Verified | arXiv:2508.17892, first posted 2025-08-25. |
| 21 | readonce | Verified | ACL 2021 DOI 10.18653/v1/2021.acl-long.554. |
| 22 | minicache | Verified | NeurIPS 2024 DOI and pages match. |
| 23 | turborag | Verified | EMNLP 2025 DOI and pages match. |
| 24 | blockattention | Verified | arXiv:2409.15355 / ICLR 2025. |
| 25 | locomo | Verified | ACL 2024 DOI and pages match. |
| 26 | xccache | Verified | Findings EMNLP 2024 DOI. |
| 27 | kvdirect | Metadata error | Work is arXiv:2603.19664 (2026-03-20), but `qcmem.bib` omits `eprint`/`archivePrefix`, so `main.bbl` provides no resolvable identifier. |
| 28 | pg19 | Verified | arXiv:1911.05507. |
| 29 | bm25 | Verified | DOI 10.1561/1500000019. |
| 30 | embeddingrecycling | Verified | Findings EACL 2023 DOI. |
| 31 | gemfilter | Verified | Findings ACL 2026 DOI 10.18653/v1/2026.findings-acl.677. |
| 32 | reform | Verified | arXiv:2506.01215. |
| 33 | lloco | Verified | EMNLP 2024 DOI and pages match. |
| 34 | hunyuan | Verified | Official Tencent Hy3 model page exists; metadata is an official technical/model-page citation rather than archival publication. |
| 35 | fusionrag | Verified | arXiv:2601.12904. |
| 36 | mepic | Verified | arXiv:2512.16822. |
| 37 | longmem | Verified | NeurIPS 2023 DOI 10.52202/075280-3259. |
| 38 | memoryllm | Verified | arXiv:2402.04624 / ICML 2024. |
| 39 | infllm | Verified | NeurIPS 2024 DOI and pages match. |
| 40 | streamingllm | Verified | arXiv:2309.17453 / ICLR 2024. |
| 41 | sempic | Verified | arXiv:2607.28069, first posted 2026-07-30; contemporaneous. |
| 42 | xu2024retrievallong | Verified | arXiv:2310.03025 / ICLR 2024. |
| 43 | qwen3 | Verified | arXiv:2505.09388. |
| 44 | ape | Verified | arXiv:2502.05431. |
| 45 | cacheblend | Verified | EuroSys 2025 DOI 10.1145/3689031.3696098. |
| 46 | h2o | Verified | NeurIPS 2023 DOI 10.52202/075280-1506. |

**Totals:** 45 Verified, 1 Metadata error, 0 Not found, 0 Unverifiable after fallback checks.

## Load-bearing citation--claim matches

1. **ReadOnce / Embedding Recycling:** the cited works do cache reusable intermediate representations and train downstream components; the paper's distinction from an autoregressive decoder-serving depth axis is fair. **Match.**
2. **LLMCache:** layer-wise banks plus semantic matching at arbitrary layers are supported; the absence of CoMem's fixed document split/$j=0$ measurement is a reasonable distinction. **Match.**
3. **HCache / KV-Direct:** hidden/residual checkpoints are used to restore or reconstruct per-layer KV. **Match**, though RSCE should also be discussed.
4. **Prompt Cache / Block-Attention / TurboRAG:** modular or independently prepared KV reuse with positional/linking treatment is accurately summarized. **Match.**
5. **CacheBlend:** selective per-layer KV recomputation for high-deviation tokens is accurately described in Related Work. **Citation claim matches, but the paper's custom implementation does not fully match the cited algorithm (W1).**
6. **EPIC / MEPIC / APE:** position-independent/parallel context caching and linking/realignment are accurately characterized. **Match.**
7. **KV Packet / Cartridges / SemPIC:** learned or compiled modular document KV representations are accurately grouped at a high level. **Match**, with SemPIC contemporaneous.
8. **ILRe / REFORM / GemFilter:** using intermediate layers to select/gather tokens without cross-query persistence is a fair distinction. **Match.**

# Novelty Search Summary

Five searches were run using combinations of: “persistent intermediate hidden states transformer reuse queries split layer decoder suffix,” “cache hidden state chosen layer repeated queries document,” “reusable residual stream long-context inference,” “precompute lower transformer layers and reuse upper layers,” and “layer-wise activation caching transformer inference.” I also followed title/author/arXiv/DOI records for the closest hits.

Closest works:

1. **ReadOnce Transformers (ACL 2021):** persistent task-independent document representations for repeated use. Closest conceptual predecessor, but encoder-style/compressed representation and no measured decoder split-depth serving axis.
2. **Embedding Recycling (Findings EACL 2023):** caches an intermediate encoder layer and trains layers/adapters above it. Very close mechanism family, but not document retrieval plus native autoregressive suffix continuation or matched $j=0$ serving measurement.
3. **HCache (EuroSys 2025):** stores hidden states across layers to reconstruct KV for state restoration. Close persistent-state systems work, but its object/goal is standard layer-wise KV restoration rather than one selected residual split with suffix execution.
4. **RSCE: Accelerating Large Language Model Decoding with Cache-Compressed Context (Findings ACL 2025):** a close residual/hidden-state cache-compression/restoration line omitted by the paper. It should be explicitly contrasted; based on available metadata, it still does not appear to provide CoMem's selected-document split-depth control and identical-evidence endpoint.
5. **LLMCache (first posted 2025-12-18):** semantically matched activation banks at arbitrary layers. Close in layer-aware reuse, but not the same persistent document object/interface.

Additional very recent close works include **KV-Direct** (2026-03-20), **KV Packet** (2026-04-14), **Cartridges at Scale** (2026-06-03), and **SemPIC** (2026-07-30). Under a three-month rule relative to the frozen date of August 4, 2026, Cartridges at Scale and SemPIC are contemporaneous and should be discussed but not used to penalize novelty. The surviving novelty appears to be the controlled *measurement formulation*—one chosen residual split plus matched $j=0$—rather than residual caching itself.

# Review-Process Self-Check

- [x] Reviewed only the frozen v12 PDF, its frozen source/artifacts, and the specified template; did not inspect other review history, live drafts, TODO/status files, or Paper B.
- [x] Read the full paper twice, including both appendices.
- [x] Inspected all 28 rendered pages, Figures 1--2, and Tables 1--40.
- [x] Checked abstract numbers against Tables 2--7, 8, 31, and 34--35.
- [x] Checked formulas, storage arithmetic, boundary cases, timing exclusions, seeds/statistics, and claim scope.
- [x] Searched source/PDF for prompt injection, hidden/white/tiny reviewer manipulation, unresolved references, placeholders, and anonymity leaks.
- [x] Verified all 46 cited `main.bbl` entries and 8 load-bearing citation--claim matches.
- [x] Ran five novelty searches and applied the three-month contemporaneous-work rule.
- [x] Verified supplied SHA-256 manifests; reran the Table 4 checker; inspected the custom baseline self-test and aggregation limitations.
- [x] Mechanically checked every weakness quote against the frozen source; every quote is <=25 words.
- [x] Mechanically checked every “paper lacks X” assertion against the frozen source/artifacts and retained only claim-linked deficiencies.
- [x] Did not edit the manuscript or any artifact.
