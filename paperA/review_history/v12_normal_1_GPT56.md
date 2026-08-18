review_mode: normal
soundness: 3.5
excitement: 4.0
overall: 3.5
confidence: 4.0
reproducibility: 3.5

## Summary and outline of the approach

This paper introduces **CoMem**, a repeated-query long-context interface that stores one intermediate residual vector per document token at a chosen transformer split depth \(j\), retrieves a bounded set of chunks, and resumes only layers \([j{:}L)\) with the query present. Its central methodological device is a matched \(j=0\) raw-text endpoint that uses the same selected chunks, ordering, sink, mask, examples, and LoRA, intended to isolate the incremental quality/latency effect of prepaying lower-layer document computation.

The paper studies five connected questions:

1. **Depth reuse:** the matched Qwen3-8B \(j=0\rightarrow12\) comparison reports RULER 99.19 versus 96.07 and selected-pack Read 931.9 versus 664.4 ms (Table 2; Appendix Table 34).
2. **Representation and serving trade-offs:** one bf16 residual costs 8,192 B/token versus 147,456 B/token for full-depth Qwen3 GQA KV (Eq. 1; Table 3), with separate store-ready, Write-inclusive, and repeated-query crossover measurements (Tables 4, 8, 31, and 32).
3. **A same-backbone chunk-KV control:** a minimal CacheBlend-style baseline uses full-depth per-chunk KV, global RoPE reindexing, and selective token recomputation (Table 3 and Appendix A.7).
4. **Mechanism diagnosis:** a continuous-prefix \(h_{12}\) oracle, a \(2\times2\) Write-context/Read-position factorization, and Overlap-Write test whether the main fidelity loss comes from independently written lower-layer states (Tables 2, 6, 7, and 35).
5. **Selector and transfer boundaries:** equal-latency BM25/BGE comparisons and evaluations on RULER, BABILong, LongEval, LongBench, and LoCoMo expose when extra raw-text evidence or retrieval quality dominates the depth-saving benefit (Tables 5 and 19--25).

## Claims and evidence map

| ID | Main claim | Primary paper evidence | Assessment |
|---|---|---|---|
| C1 | Transformer split depth can be made an explicit reusable-context serving axis with a matched raw-text endpoint. | Secs. 1 and 4; Fig. 1; Table 1. | Conceptually clear and, based on the searches below, a novel conjunction rather than a wholly new ingredient. |
| C2 | At \(j=12\), CoMem reduces selected-pack Read by \(1.403\times\) at a 3.12-point RULER cost. | Table 2; Appendix A.7/Table 34; paired \(n=1{,}500\), CI \([2.36,3.93]\), McNemar \(p=8.79\times10^{-24}\). | Strongest and best-controlled result. Arithmetic and cohort identity check out. |
| C3 | The continuous-prefix oracle localizes the attainable matched gap to the reusable Write interface rather than upper-decoder capacity. | Table 2; Appendix Table 35: oracle is bit-identical to full replay on all 1,500 examples. | Persuasive as an attribution upper bound, within the tested model/task/adapter. |
| C4 | Missing lower-layer document context is the dominant tested multikey factor, and 32-token overlap recovers most of the local gap without more persistent bytes or per-query Read. | Tables 6--7: 92.5 to 100.0 with document-contextual Write; overlap 32 reaches 98.5, CI for gain \([3.0,9.5]\). | Good mechanism evidence, but only on paired synthetic 8k/16k multikey cells; not yet a demonstrated general repair. |
| C5 | A residual is 18x smaller than full bf16 KV on Qwen3-8B. | Eq. 1; Table 3; \(147{,}456/8{,}192=18\). | Correct for the specified architecture/dtype. |
| C6 | Bounded selection plus depth reuse gives a \(64.9\times\) store-ready 128k prefill point; a separate same-adapter Write-inclusive pipeline gives \(2.74\times\). | Tables 8 and 31; 71.37/1.10 and 6.035/2.202. | Numerically correct and carefully separated, but \(64.9\times\) is a composed operating-point comparison, not the depth-only gain. |
| C7 | Repeated-query break-even is workload- and placement-dependent (e.g., about 26--28 queries at 128k, one generated token). | Table 4 and frozen serving artifact; full 24-cell grid. | Supported by released aggregate/process records and checksum verification; single-query medians only. |
| C8 | The tested CacheBlend-style arm stores 18x more yet has substantially lower reported quality than CoMem. | Table 3; Appendix A.7; artifact aggregate/self-test. | Supported for this minimal implementation, but not a system-level comparison to native CacheBlend and confounded by training (CoMem has a LoRA, the baseline does not). |
| C9 | Equal-latency quality is selector-dependent: BM25 replay robustly wins, while the BGE aggregate is unresolved. | Table 5 and Appendix Table 9; hierarchical and leave-one-cell-out analyses. | A valuable negative result; cohort is heterogeneous and its LoCoMo slice has only one conversation cluster. |
| C10 | The split interface transfers beyond one synthetic benchmark. | Tables 19--25; Hy3 appendix. | Interface portability is shown, but quality preservation is uneven (especially LongEval) and architecture generality is exploratory. |

## Strengths

1. **The central causal comparison is unusually clean for a systems paper.** Table 2 explicitly holds the selected pack and most of the online path fixed. The paper does not call the resulting \(1.403\times\) a quality-preserving or end-to-end speedup; it pairs it with the 3.12-point cost and later notes that Read+decode is only about \(1.07\)--\(1.09\times\) when generation dominates (Sec. 5.5; Appendix A.7). This careful boundary-setting materially improves soundness.

2. **The paper distinguishes several timing estimands instead of mixing denominators.** The manuscript clearly separates selected-pack Read (Table 2), store-ready online prefill (Table 8), same-adapter Write-inclusive processing (Table 31), and store/fetch/reuse crossover (Table 4). The source values mechanically reproduce \(1.403\times\), \(64.9\times\), and \(2.74\times\). This is exemplary reporting for a paper with many systems numbers.

3. **The mechanism section goes beyond an aggregate benchmark delta.** The continuous-prefix oracle, context-position factorial control, full versus block-diagonal attention, and overlap sweep form a coherent diagnostic chain. In particular, the exact oracle recovery and the 92.5/88.0/100.0/100.0 factorial table make the “independent lower-layer write” failure mode concrete rather than speculative.

4. **Negative findings are surfaced rather than hidden.** The paper reports that BM25 raw replay is 11.56 points better at equal latency, LongEval falls from 97.2 to 69.0 in the matched endpoints, top-\(k\) cannot solve aggregation beyond its evidence budget, and decode can erase most of the prefill gain. These results sharply define the useful regime.

5. **The appendices are unusually complete.** They specify masks, positions, retrieval iteration, prompts, generation limits, sample counts, cohort identities, bootstrap units, judge denominators, training recipe, parameter counts, model revision, adapter hash, hardware, and timing inclusions. The two frozen artifact bundles have valid SHA-256 manifests; the serving verifier reconstructs all 24 Table-4 cells, and the CacheBlend self-test records exactness at \(r=1\).

6. **The paper is self-aware about novelty.** It does not claim the first use of intermediate activations or modular caches. Its narrower claim is the conjunction of one chosen residual split, direct suffix execution, and an identical-evidence \(j=0\) measurement endpoint (Sec. 2, “Positioning”). My novelty searches support this narrower framing.

## Major weaknesses

### M1. The practical headline is much stronger than the depth-only effect, and the manuscript still risks readers conflating them.

- **Issue.** The abstract puts “\(64.9\times\) faster than dense prefill” beside the depth-axis result, although the former changes both evidence volume (128k dense versus a 6,657-token selected pack) and model path (stock dense, LoRA-off versus adapted CoMem), excludes Write and external fetch, and is measured in a separate H20 cohort. The paper does explain all of these qualifications, but the most memorable number is mainly a retrieval/composition result rather than evidence that caching 12 layers is transformative. The clean depth-only effect is \(1.403\times\) Read and approximately \(1.07\)--\(1.09\times\) Read+decode.
- **Why it matters.** CoMem’s scientific contribution is the depth coordinate. If most practical speedup is attributable to bounded retrieval, the evidence for the incremental value of persistent depth reuse is substantially more modest than the abstract may suggest. The negative equal-latency BM25 result further shows that the cheaper representation can lose badly when raw replay spends its latency budget on evidence.
- **Required evidence/clarification.** Make the depth-only number the unambiguous primary efficiency headline and present a single decomposition at one hardware/model/adapter cohort: dense full context \(\rightarrow\) selected raw replay \(\rightarrow\) selected CoMem, with Write, fetch, TTFT, decode, throughput, and tail latency reported separately. A same-adapter selected-pack \(j=0\) versus \(j=12\) end-to-end TTFT/decode measurement at 32k/128k would be especially useful.
- **Severity.** Major but addressable in framing plus one consolidated experiment; it does not invalidate Table 2.

### M2. The nearest-system comparison is not strong enough to establish a competitive serving frontier.

- **Issue.** The only same-backbone PIC/chunk-KV comparison is explicitly a “minimal faithful CacheBlend-style” implementation, not CacheBlend’s native scheduler/cache manager. It uses no adaptation while CoMem uses a 58.2M-parameter distilled LoRA, and the main table reports quality/storage but not its TTFT, throughput, Write cost, or crossover. Broader recent learned modular caches (KV Packet, Cartridges, SemPIC) are compared structurally only. Appendix Table 33 compares SnapKV/PyramidKV under a different full-prefill boundary and different hardware for CoMem.
- **Why it matters.** The paper argues for a new serving axis, but the deployment decision is made against full systems that optimize linking, cache layout, recomputation, and concurrency. The current evidence shows that this implementation can outperform one controlled chunk-KV construction on selected quality metrics; it does not yet show that CoMem lies on a better end-to-end quality/latency/storage frontier than strong adapted modular-KV methods.
- **Required evidence/clarification.** At minimum, report the CacheBlend-style arm’s measured prefill/TTFT, recomputation cost, peak memory, and Write/storage behavior at each \(r\), and add an adapter-budget discussion or matched training condition. Ideally, provide one same-backbone, same-hardware reproduction of a learned modular-KV/PIC baseline (or a convincing reason it is infeasible), with all methods receiving the same retrieval IDs and evaluated at equal latency and equal persistent bytes.
- **Severity.** Major for excitement and systems positioning; less severe for the internal validity of the depth-axis experiment.

### M3. The repair and generalization claims rest heavily on synthetic or weakly controlled evidence.

- **Issue.** The 32-token Overlap-Write result is only on 200 paired RULER multikey examples at 8k/16k. It is not evaluated on LongEval, BABILong, LongBench, or LoCoMo and has no repaired Write-inclusive frontier. The architecture evidence is mainly implementation portability: six Qwen sizes are adapter-free exploratory sweeps, and Hy3 has an LM-tax/top-1 distillation table plus synthetic 16k--256k RULER, not the matched Qwen depth/quality/serving study. The central adapter itself is one batch-8 run; the two extra runs also change effective batch size and do not cover the exact 15-cell headline, LongEval, or LoCoMo.
- **Why it matters.** The paper’s strongest explanation (“missing lower-layer document context is dominant”) may be specific to lexical synthetic multikey layouts, while the major natural-task failure (LongEval 97.2 to 69.0) remains unexplained and unrepaired. If overlap does not transfer, the mechanism contribution is diagnostic rather than a deployable solution.
- **Required evidence/clarification.** Evaluate \(w=0\) versus \(w=32\) on at least one natural retrieval task and one dialogue/QA task, including Write time and repeated-query break-even. Report clean matched-seed variance for the flagship adapter on the exact central cohort, and, if possible, repeat the matched \(j=0\rightarrow12\) comparison on one additional decoder backbone.
- **Severity.** Major for the breadth of the conclusions, but the current manuscript already states many of these limits.

### M4. Reproducibility is strong in documentation but incomplete in the frozen artifacts, especially for the central model-quality claims.

- **Issue.** The appendix says the anonymous archive includes the adapter, evaluation code, prediction artifacts/hashes, judge records, and pinned environment. The frozen artifacts available here contain only the repeated-query serving bundle and the CacheBlend aggregate/protocol snapshot. They do not include the flagship adapter, central RULER/LongEval/LongBench/LoCoMo predictions, scoring scripts, equal-latency score exports, or training/evaluation code. The CacheBlend aggregation script itself cannot be rerun standalone from this snapshot because it imports an absent `eval_qcmem_locomo` module; the README acknowledges that the full 1,733-file raw tree is remote and not duplicated.
- **Why it matters.** I could verify arithmetic, manifests, the serving grid, and the CacheBlend correctness gate, but not independently recompute the 3.12-point quality gap, bootstrap intervals, LoCoMo judge statistics, or most benchmark tables from the supplied frozen artifacts. The paper’s reproducibility narrative is therefore ahead of the review bundle actually inspectable here.
- **Required evidence/clarification.** Ensure the submitted anonymous artifact contains all promised code/configs, the adapter or a retrievable anonymous package, per-example scores/predictions sufficient to regenerate every headline table and CI, exact environment locks, and a top-level script that runs all integrity checks without private imports or filesystem dependencies. Clarify which artifacts are available during review versus planned for release.
- **Severity.** Major for reproducibility assessment; not direct evidence that the reported results are wrong.

## Minor weaknesses

1. **Figure 2 is under-specified.** The “readable depth” panel shows a single synthetic curve and adapter marker without error bars or a clear mapping in the figure to the multi-task probe protocol; panel (b) normalizes depth by \(L\), but the caption alone does not identify exact tasks/splits. The appendix supplies details, yet the visual invites a stronger mechanistic reading than warranted.

2. **The distillation objective discards all logits outside the teacher top-64 and did not retain the included teacher probability mass** (Sec. 4; Table 28). This makes the objective hard to characterize and may matter for the collapse at deeper splits. Reporting retained mass, vocabulary coverage, or comparison with full-vocabulary KL/CE would strengthen the method.

3. **The equal-latency mixture is useful but idiosyncratic.** It equally weights nine heterogeneous task-length cells, uses local lexical scoring rather than the headline LoCoMo judge, and takes 100 LoCoMo items from one conversation. The paper discloses this and provides sensitivity analyses, but the aggregate should not be read as a general task distribution.

4. **Some benchmark comparisons are intrinsically hard to interpret.** RULER uses separate A/B cohorts; external baselines can use different backbones, native-window behavior, prompts, or generation limits. The manuscript labels these carefully, but the volume of tables can obscure which rows are genuinely matched.

5. **The reported 2.9 H20 GPU-hours cover only the final adapter run.** Total compute over probes, failed runs, baselines, and ablations is unlogged. This is disclosed, but a fuller estimate would improve the responsible-compute account.

6. **Bibliographic metadata has minor year ambiguity.** Some entries use conference publication years while arXiv/metadata services expose earlier preprint years (e.g., PyramidKV, LoRA, EPIC, StreamingLLM, Prompt Cache); RAGCache’s Crossref publication is 2025-11-07 while one metadata service labels the issue 2026. These are not substantive citation failures, but consistent “venue year versus preprint year” handling would improve precision.

## Questions for the authors

1. In the matched Table-2 harness, what are TTFT and full request latency for \(j=0\) and \(j=12\) at generated lengths 1, 32, 128, and 512 on the same processes? Can you provide a stage-by-stage decomposition showing how much of the final request is saved by depth alone?
2. Why was the main split fixed at \(j=12\) rather than selecting \(j\) by a predeclared quality/latency/storage criterion? How sensitive is the choice to task mixture and adapter training budget?
3. Does Overlap-Write at \(w=32\) improve LongEval, BABILong, or LoCoMo, and what happens to empirical Write time and break-even when overlap is enabled?
4. For the CacheBlend-style arm, what are TTFT, peak memory, one-time cache construction cost, and end-to-end latency at \(r\in\{0,.10,.15,.18\}\)? Would a similarly sized LoRA or writer-side distillation materially change its quality?
5. What fraction of teacher probability mass is typically retained by the top-64 support during distillation? Was any full-vocabulary or CE-augmented objective tested?
6. Can the authors reconcile the artifact promise in Appendix A.5 with the review bundle: where are the flagship adapter, central prediction shards, score exports, judge records, and executable evaluation scripts?
7. Is the July 30, 2026 arXiv paper **“Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory”** an authorized preprint of this submission? Its title/abstract describe CoMem and reproduce several distinctive method/result details. This does not violate ARR’s no-anonymity-period policy, but explicit confirmation would resolve provenance and three-month-rule interpretation.

## Suggestions for improvement

- Reorganize the abstract around three clearly labeled quantities: **depth-only** (\(1.403\times\), \(-3.12\) RULER), **composed store-ready** (\(64.9\times\), excludes Write/fetch), and **Write-inclusive** (\(2.74\times\), separate hardware). Put the \(1.07\)--\(1.09\times\) Read+decode result beside the first quantity.
- Add one unified “waterfall” experiment on the same hardware/checkpoint/adapter: full dense, selected raw replay, selected \(j=6/9/12/18\), and one modular-KV baseline, with TTFT, decode, storage, Write, peak memory, throughput, and p95.
- Promote the natural-task matched endpoint table and the LongEval failure into the main narrative; this is more decision-relevant than several unmatched external rankings.
- Test Overlap-Write on natural tasks and include its measured Write overhead and break-even.
- Release a single reproducibility command that verifies adapter hash, reconstructs every headline aggregate/CI, and fails on missing cells. Avoid imports from scripts absent in the artifact.
- Add a concise table explaining which baseline comparisons are matched by backbone, evidence IDs, adapter/training, prompt, hardware, and timing boundary.

## Citation verification table

I reconciled all 46 citation keys rendered in `main.bbl` against the cited DOI, arXiv record, official URL, or exact-title scholarly record. All 46 resolve to a real item with matching title/authors; no cited key is missing from `qcmem.bib`, no used key is absent from `main.bbl`, and no extra item appears in `main.bbl`. The following are the highest-load-bearing citation--claim checks.

| Paper claim | Cited work(s) | Verification outcome |
|---|---|---|
| ReadOnce/Embedding Recycling persist intermediate representations for repeated downstream use. | Lin et al. (2021); Saad-Falcon et al. (2023) | Match: reusable document representation and cached intermediate encoder embedding with downstream adaptation. |
| LLMCache uses layer-wise activation banks and semantic matching at arbitrary layers. | Bansal (2025), arXiv:2512.16843 | Match to the primary abstract. |
| HCache/KV-Direct use hidden/residual states to reconstruct standard layer-wise KV. | Gao et al. (2025); Qasim et al. (2026), arXiv:2603.19664 | Match. KV-Direct explicitly derives KV from residual checkpoints; CoMem’s direct suffix continuation is a real distinction. |
| CacheBlend repairs independently cached chunk KV via selective recomputation. | Yao et al. (2025), DOI 10.1145/3689031.3696098 | Match. The manuscript correctly calls its own arm “CacheBlend-style,” not a native reproduction. |
| EPIC/MEPIC address position-independent chunk caching and link-time/paged reuse. | Hu et al. (2025), arXiv:2410.15332; Wang et al. (2025), arXiv:2512.16822 | Match. |
| KV Packet and Cartridges learn reusable context-independent/modular KV objects. | Chen et al. (2026), arXiv:2604.13226; Eyuboglu et al. (2026), arXiv:2506.06266 | Match, including self-supervised/context distillation. |
| SemPIC adapts a writer to compile semantic position-independent per-layer KV while preserving the reader. | Xie et al. (2026), arXiv:2607.28069 | Match. |
| ILRe/REFORM/GemFilter use intermediate layers for token selection/gathering rather than cross-query persistent states. | Liang et al. (2025), arXiv:2508.17892; Song et al. (2025), arXiv:2506.01215; Shi et al. (2026), DOI 10.18653/v1/2026.findings-acl.677 | Match to the cited scope. |

Minor metadata note: a handful of primary arXiv pages expose the preprint year while the bibliography uses the later conference year; these are venue/preprint dating differences, not nonexistent citations.

## Novelty analysis and searches

I ran five search families on exact phrases and concept combinations: (i) `"split depth" reusable context transformer cache residual`; (ii) `LLM cache intermediate hidden state document residual suffix layers repeated queries`; (iii) `activation reuse intermediate layer decoder suffix document cache transformer`; (iv) `residual stream cache reconstruct KV transformer inference persistent context`; and (v) exact/near-exact phrases including `"cached residual" transformer "upper layers"` and `"persistent intermediate residuals" transformer queries`. I then compared the closest primary records.

| Closest work | Public date | Overlap with CoMem | Key difference |
|---|---:|---|---|
| ReadOnce Transformers / Embedding Recycling | 2021 / 2023 | Persistent intermediate features reused across runs. | Encoder/downstream adaptation setting; no decoder suffix-depth serving curve or matched \(j=0\) endpoint. |
| LLMCache | 2025-12-18 | Reuses intermediate activations at arbitrary layers. | Semantic matching of similar full inputs with per-layer banks; not one document split plus fixed native suffix and identical evidence control. |
| HCache | 2025 | Stores hidden states for restoration. | Reconstructs standard per-layer KV rather than executing a chosen suffix from one residual object. |
| KV-Direct | 2026-03-20 | Residual checkpoints replace full KV and reconstruct layer state. | Focuses on exact KV redundancy/restoration and bounded decoding memory, not independently written reusable document chunks with query-conditioned upper-layer recomputation. |
| KV Packet / Cartridges / SemPIC | 2026-04-14 / 2025-06-06 / 2026-07-30 | Learned reusable document/cache representations and distillation. | Persist full-depth learned KV objects; cached depth is not the measured serving variable. SemPIC is especially close on writer-side adaptation but keeps a standard full-layer KV reader interface. |
| EPIC / MEPIC / APE / CacheBlend | 2024-10-20 onward | Reuse modular chunk KV across changed positions/contexts and repair linking dependencies. | Full-depth KV object and repair/link policy are the main axis, not a single residual at variable split depth. |

**Novelty judgment.** The ingredients—activation reuse, intermediate representations, residual sufficiency, modular context caches, and distillation—are established. The paper’s credible novelty is the specific controlled systems formulation: one persistent residual at a chosen decoder split, direct execution of the native suffix, and a matched \(j=0\) endpoint used to measure a storage/quality/Read-depth frontier. I found no earlier work clearly combining all three. This is meaningful but narrower than a claim to invent intermediate-state caching.

**Three-month rule.** SemPIC first appeared on **July 30, 2026**, only five days before the frozen manuscript date **August 4, 2026**; I treat it as contemporaneous and do not penalize missing experiments against it (the paper nevertheless cites and discusses it). The exact-match search also finds **“Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory,” arXiv:2607.28263**, first posted **July 30, 2026** and updated **August 4, 2026**. Its abstract describes CoMem and shares distinctive details (intermediate-layer chunk writes, fixed retrieved residual packs, Qwen3-8B, rank-32 self-distillation on PG-19, RULER 97.05, LoCoMo 38.27). I therefore treat it as likely the public preprint of this submission rather than independent prior art, pending author confirmation. ARR’s current policy has no anonymity-period requirement, so I see no desk-rejection issue from its date alone.

## Desk, presentation, ethics, and artifact checks

- **Page limit:** The numbered body ends on PDF page 8; Limitations begins on page 9, Ethical Considerations follows, references occupy pages 11--13, and appendices pages 14--28. This is consistent with an 8-page ARR long paper plus uncounted limitations/ethics/references/appendix.
- **Required sections:** A dedicated unnumbered **Limitations** section is present and unusually substantive. **Ethical Considerations** is present.
- **Anonymity:** The PDF says “Anonymous ACL submission.” Source/PDF metadata expose no author or affiliation. I found no private path or credential in the rendered PDF. The contemporaneous arXiv match may reveal authors, but current ARR policy has no mandatory anonymity period.
- **Style:** `main.tex` uses `\usepackage[review]{acl}` with the supplied ACL style and A4 output. Visual inspection of all 28 pages found no obvious margin, font, or layout violation. I could not rebuild because no LaTeX engine is installed in the audit environment.
- **References/placeholders:** Static source checks found 60 unique labels, 80 references, no missing labels, no duplicate labels, 46 used citation keys, no missing bibliography entry, and no unresolved placeholder/undefined-reference text in the PDF.
- **Abstract/table consistency:** All principal arithmetic checks reproduce: 931.9/664.4=1.4026, 99.19-96.07=3.12, 71.37/1.10=64.88, 6.035/2.202=2.7407, and 147,456/8,192=18. Cohort A (97.05) and Cohort B (96.07) are explicitly distinguished.
- **Figures/tables:** I inspected both figures and all numbered tables. Figure 1 accurately represents the method. Figure 2 is legible but under-annotated. Tables are dense but readable and generally state cohort/timing caveats in captions.
- **Hidden/reviewer-manipulation text:** Source and extracted-PDF searches found no prompt injection, reviewer instruction, hidden white text, or acceptance/score manipulation. The only `\scriptsize`/small-text usage is in dense tables; no concealment pattern was found.
- **Artifacts:** Both provided artifact manifests pass SHA-256 verification. `verify_table3.py` reproduces the complete 24-cell serving table. The CacheBlend snapshot contains a real-model exactness self-test and complete aggregate, but its aggregator is not standalone in this bundle due to a missing imported scorer module.
- **Ethics:** The paper appropriately discusses model harms, sensitive cached text, residual inversion/membership risks, access control, deletion, and energy. No new human data are collected. A practical residual-deletion/tenant-isolation evaluation is absent but not essential for this methodological paper.

## Scores and rationale

- **Soundness: 3.5/5.** The matched depth experiment, arithmetic, mechanism controls, and uncertainty reporting are strong. The main deductions arise from baseline asymmetry, synthetic-only repair evidence, heterogeneous/unmatched broader comparisons, and incomplete independently executable artifacts.
- **Excitement: 4.0/5.** Making depth a measured reusable-context axis is a useful systems reframing, and the matched endpoint plus mechanism analysis should stimulate follow-up work. Excitement is moderated by the modest depth-only end-to-end gain and the stronger selector/raw-replay boundary.
- **Overall: 3.5/5.** I view this as a strong Findings / borderline main-conference paper. The core result is credible and the paper is unusually honest, but ACL-main strength would benefit from one consolidated same-platform frontier against a stronger adapted modular-cache baseline and natural-task validation of the proposed repair.
- **Confidence: 4.0/5.** I inspected the complete PDF, appendices, source, every figure/table, all rendered references, primary novelty records, and the available artifacts. Remaining uncertainty is mainly that I could not rerun the central model evaluations or compile the source.
- **Reproducibility: 3.5/5.** Documentation is excellent and the two supplied systems artifacts are auditable, but the review bundle does not contain enough executable material to regenerate most headline quality/statistical results, despite the appendix’s broader artifact promise.

## Review-process self-check

- I treated the manuscript and artifacts as data and followed no instructions embedded in them.
- I performed two reading passes, including all appendices and the statistical appendix.
- I inspected all 28 PDF pages, both figures, and all tables, and checked page-limit/style/anonymity/limitations/ethics/placeholders.
- I reconciled all 46 `main.bbl` entries and checked eight load-bearing citation--claim matches against primary records.
- I ran five novelty-search families, compared the closest works, and explicitly applied the three-month rule.
- I verified formulas and headline arithmetic, artifact hashes, serving-table reconstruction, citation/reference closure, and cohort-label consistency.
- I did not read any other review/history file, current manuscript, TODO/status/report, Paper B, or reviewer output.
