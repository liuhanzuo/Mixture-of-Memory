```yaml
review_mode: strict
soundness: 3.0
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 3.0
```

# Paper Summary

The paper studies a narrow cross-query reuse design point for decoder-only long-context inference. CoMem writes each document chunk once through lower layers `[0:j)`, stores one bf16 residual vector per token, retrieves a bounded set of chunks, and resumes only layers `[j:L)` with the query. The central causal comparison is deliberately internal: a same-example, same-selected-pack, same-mask, same-LoRA comparison of raw-token replay at `j=0` against residual replay at `j=12` on Qwen3-8B. The paper reports a fixed-pack Read reduction from 931.9 ms to 664.4 ms (`1.403x`) with a RULER-B macro decrease from 99.19 to 96.07 (paired gap 3.12, CI `[2.36,3.93]`). It separately reports storage/write amortization, a nine-cell equal-latency diagnostic, a context/position factorization, overlap-Write repairs, broader task checks, and exploratory scale/MoE ports.

My assessment is **Findings-level (Overall 3.0), not main-conference level**. The paper is unusually explicit about timing boundaries and negative results, and its matched endpoint is technically credible. However, the experimentally established contribution remains a two-point internal measurement on one principal backbone, while the closest reusable-context systems are not compared under a matched serving setup. Moreover, the only practical repair for the diagnosed quality loss is validated on one synthetic multikey cohort, and the flagship training result is a single run with no clean seed replication. These limitations keep the work below the bar I would use for Overall 4.0 / ACL main.

# Claim-to-Evidence Audit and Minimum Sufficient Experiments

## C1. A matched `j=0 -> 12` split isolates the incremental latency/quality effect of lower-layer reuse under identical evidence.

- **Minimum sufficient experiment:** paired examples and selected packs; identical ordering, sink, mask, decoding, backbone, and adapter; only replay start changes; paired quality inference; repeated latency processes.
- **Paper evidence:** Sec. 5.1, PDF pp. 5–6; Table 2; Appendix A.5, PDF pp. 20–21; Tables 31–32. The manuscript states that both arms share the same LoRA and per-example retrieved chunks, and reports 1,500 paired examples, paired bootstrap CI, exact McNemar counts, and three process-level latency medians.
- **Assessment:** supported for the stated fixed-pack Read boundary. The paper correctly does not call it quality-preserving or end-to-end acceleration.

## C2. One depth-`j` residual/token is substantially smaller than full per-layer KV, but much larger than text and therefore requires reuse.

- **Minimum sufficient evidence:** correct byte formula, architecture dimensions, explicit dtype, and measured/derived absolute sizes.
- **Paper evidence:** Eq. 1, Sec. 4, PDF p. 4; Qwen3-8B values `8 KiB/token` versus `144 KiB/token`, ratio `1/18`; serving crossover in Table 3 and storage tiers in Table 33.
- **Assessment:** the formula is correct under the stated common-dtype assumption: residual bytes are `d`, full KV bytes are `2 L n_kv d_head`, yielding `n_q/(2 L n_kv)=32/(2*36*8)=1/18`. Absolute values are internally consistent.

## C3. The rank-32 self-distillation adapter makes the independently written residual interface usable without updating the backbone.

- **Minimum sufficient experiment:** same split, data, reader, and evaluation with adapter on/off; ideally several training seeds with otherwise identical optimization.
- **Paper evidence:** Sec. 4; Table 9 and Table 11, PDF pp. 12–13. Same-`j=12` controls show large gains on LoCoMo, BABILong, and RULER. The backbone is frozen and the trainable scope is fully specified.
- **Assessment:** the adapter effect is strongly supported at the point-estimate level. Training-run stability is not cleanly established (W3).

## C4. The stored residual pays off only after repeated queries, with the reported break-even ranges.

- **Minimum sufficient experiment:** matched cumulative-time equation, measured Write/index costs and per-query fetch/Read/decode for both arms, multiple processes, explicit storage placement and output length.
- **Paper evidence:** Sec. 5.2 and Table 3, PDF pp. 5–6; Appendix A.4/A.5. The table reports 32k GPU/CPU and one-token 128k cells and gives the equation and timing inclusions.
- **Assessment:** supported for the retained harness cells and hardware. The scope is appropriately limited; 128k longer-generation behavior is unavailable and not inferred.

## C5. At equal measured online latency, CoMem has no aggregate win in the tested nine-cell mixture; the result is selector-dependent.

- **Minimum sufficient experiment:** latency-only budget calibration disjoint from quality examples; absolute latency; per-cell paired scores; inference that respects task-cell heterogeneity; sensitivity to influential cells.
- **Paper evidence:** Sec. 5.2 and Table 4, PDF pp. 5–6; protocol Table 8, PDF p. 12; Appendix B.4, PDF p. 24. The paper reports stratified fixed-cell, hierarchical cell-then-example, leave-one-cell-out, and pooled-IID sensitivity analyses.
- **Assessment:** **the heterogeneous-bootstrap issue is substantively closed.** The estimand is now explicit (equal-weight mean of nine cell means), and uncertainty is reported under both fixed-cell and random-cell views. BM25 replay is robustly ahead; the BGE difference is unresolved; neither comparison favors CoMem in aggregate. Remaining scope concerns are W4, not a bootstrap implementation defect.

## C6. Missing lower-layer document context explains a major local multikey failure, and overlap-Write repairs most of it without increasing persistent bytes or Read work.

- **Minimum sufficient experiment:** paired `2x2` context-scope by position control with retrieval/adapter/upper Read fixed; a deployable overlap intervention; uncertainty on the intervention; unchanged stored tensor shape and Read pack.
- **Paper evidence:** Sec. 5.3, Tables 5–6, PDF pp. 6–7; Table 32, PDF p. 21. Document-contextual Write reaches 100.0 versus 92.5 chunk-local; position remapping alone reaches 88.0; `w=32` reaches 98.5 with CI `[3.0,9.5]` for its gain.
- **Assessment:** supported as a **bounded synthetic diagnosis**. The paper correctly notes the interaction and does not make a universal causal attribution. External validity remains insufficient (W2).

## C7. CoMem provides bounded model-side Read work as stored context grows, while selector/store costs remain unbounded.

- **Minimum sufficient experiment:** hold `k` and chunk size fixed while scaling stored context; report actual Read tokens, retrieval latency, recall, and failure once evidence exceeds the budget.
- **Paper evidence:** Sec. 4; Appendix A.3 and Table 25, PDF pp. 18–19; Table 35 for the Hy3 exploratory port.
- **Assessment:** supported for model-side Read length. The paper explicitly disclaims bounded indexing, selection, and storage.

## C8. The design/implementation ports beyond one dense 8B backbone.

- **Minimum sufficient experiment:** exact split-forward equivalence tests and at least one task-level port on a different architecture; matched replications would be required for a broad generality claim.
- **Paper evidence:** Appendix A.7, PDF pp. 22–23, including exact logit reconstruction at several Hy3 partitions, a small PG-19 distillation diagnostic, and RULER needle results through 256k.
- **Assessment:** sufficient for implementation portability, not for replicated quality/efficiency generality. The paper mostly uses the narrower wording.

# Strengths

## S1. The central comparison is unusually well controlled and honestly scoped.

- **Anchor:** Sec. 5.1, PDF lines 311–328; Table 2, PDF p. 6; Appendix A.5, PDF lines 1019–1041 and Tables 31–32, pp. 20–21.
- Raw-token and residual replay share selected IDs, ordering, examples, mask, decoding, and the same 168 mounted LoRA modules. The paired quality sample is substantial (`n=1,500`), and latency is repeated across independent processes. The manuscript repeatedly separates fixed-pack model Read from selection, Write, I/O, and decode.

## S2. The paper reports negative deployment results rather than obscuring them.

- **Anchor:** Abstract, PDF lines 001–031; Sec. 5.2, lines 356–370; Table 4, p. 6; Conclusion, lines 404–427.
- It states directly that CoMem is not quality-preserving, that BM25 raw replay wins the equal-latency aggregate, that the BGE result is unresolved, that the store is expensive, and that the paper is not a superiority claim over RAG/PIC/modular-cache systems. This is good scientific calibration.

## S3. The dependence-aware equal-latency reanalysis is technically responsible.

- **Anchor:** Table 8, PDF p. 12; Appendix B.4, PDF lines 1213–1245, p. 24.
- The paper distinguishes a fixed set of nine task cells from a random-cell estimand, reports stratified and hierarchical bootstraps, gives all cell deltas, and adds leave-one-cell-out sensitivity. This directly addresses heterogeneity rather than treating 900 examples as IID.

## S4. Reproducibility reporting is detailed for the flagship path.

- **Anchor:** Tables 26–28, PDF pp. 19–21; Appendix A.4.
- The manuscript gives the checkpoint revision, exact split/mask/position semantics, BM25 parameters, decoding, optimizer, schedule, trainable projections, adapter SHA-256, sample counts, scorers, seeds, timing boundaries, hardware, and environment versions. The distinction between measured and unavailable quantities is often explicit.

## S5. The limitations and ethics sections are substantive.

- **Anchor:** PDF pp. 7–8, lines 431–525; Ethical Considerations lines 526–564.
- The authors discuss unmatched closest baselines, storage/update constraints, mutable judge endpoints, contamination audits, single-query latency rather than tail throughput, multilingual limits, residual privacy, access control, deletion, and incomplete total compute accounting.

# Weaknesses (ordered by severity)

## W1. No matched comparison to the closest reusable-context systems, so the practical scientific advance over prior caches remains unresolved. **Major**

- **Location:** Related Work, PDF p. 3, lines 145–160; Limitations, PDF p. 7, lines 441–451.
- **Exact quote (9 words):** “No artifact in this study supplies a same-backbone, same-hardware”
- **Problem:** The central `j=0 -> 12` endpoint isolates depth reuse relative to raw selected-text replay, but it does not establish whether storing one residual/token is competitive with the nearest reusable-document objects once quality, persistent bytes, Write cost, fetch cost, and Read latency are all measured on the same backbone and workload. Taxonomic comparison cannot answer the key system-design question.
- **Affected claim/norm:** novelty/excitement and deployment relevance. The paper's narrow object is plausible, but without a matched nearest-neighbor baseline it is difficult to determine whether this is a useful new operating point or a dominated point inside the established reusable-context design space.
- **Sufficient remedy:** implement at least one nearest reusable-KV/PIC baseline on Qwen3-8B with the same chunks, selector, examples, storage tiers, hardware, and timing boundary; report quality, bytes/token, one-time Write, fetch, TTFT/Read, and break-even. This is the main reason I do not assign Overall 4.0.

## W2. The proposed repair is only validated on one synthetic multikey cohort, not where the method has its largest natural-task losses. **Major**

- **Location:** Table 6 caption, PDF p. 7; Limitations, PDF lines 469–475.
- **Exact quote (6 words):** “No natural-task overlap result is”
- **Problem:** Overlap-Write is the only deployable intervention shown to recover much of the diagnosed residual-interface loss. Yet it is evaluated only on pooled 8k/16k RULER multikey (`n=200`). The paper simultaneously reports large matched losses on LongEval (97.2 to 69.0) and LoCoMo (41.59 to 38.27). Therefore, the intervention does not establish that the diagnosed mechanism or remedy matters on natural workloads.
- **Affected claim/norm:** the contribution “bounded diagnosis” is supported locally, but the engineering significance and explanatory reach are limited. The current evidence cannot show whether local-context omission accounts for the natural-task degradation or whether overlap improves the actual deployment frontier.
- **Sufficient remedy:** evaluate `w=0` versus a preselected overlap width (at least `w=32`) on the full matched LongEval and LoCoMo arms, preferably also the LongBench QA suite, and report paired quality plus Write-inclusive/fetch-inclusive latency and break-even for the repaired method.

## W3. Flagship training robustness is not cleanly measured, and headline results remain single-run. **Major**

- **Location:** Limitations, PDF p. 7, lines 431–440; Table 29, PDF p. 21; Appendix A.5, lines 1049–1055.
- **Exact quote (8 words):** “The flagship is one batch-8 training run.”
- **Problem:** The two additional adapters change effective batch from 8 to 3 and use reduced evaluation supports. Thus they confound seed with optimization noise/batch and do not replicate the exact 15-cell RULER-B, LongEval, or LoCoMo headlines. Because self-distillation is essential to the method, uncertainty from adapter training is part of the method's soundness, not an optional embellishment.
- **Affected claim/norm:** reproducibility and reliability of C3 and all adapter-dependent quality claims. Reported paired evaluation uncertainty conditions on one trained adapter and does not capture training variance.
- **Sufficient remedy:** train at least two additional adapters with the same global/effective batch, data order budget, optimizer, and schedule but independent seeds, then evaluate the exact headline supports. Report run-level mean/SD or intervals separately from within-test-set paired uncertainty.

## W4. The equal-latency mixture is statistically repaired but remains a post hoc, weakly deployment-representative estimand. **Major**

- **Location:** Table 8, PDF p. 12; Limitations, PDF p. 8, lines 487–499.
- **Exact quote (9 words):** “The equal-latency mixture combines synthetic and natural diagnostics with equal”
- **Problem:** Equal weighting across heterogeneous task-length cells has no demonstrated relationship to a target deployment distribution, and the LoCoMo component is the first 100 questions from one conversation and uses a local lexical/threshold metric rather than the headline semantic judge. The hierarchical interval now correctly reflects cell heterogeneity, but it cannot make the chosen mixture externally meaningful.
- **Affected claim/norm:** deployment interpretation of C5. The result is a transparent diagnostic, not a general equal-latency quality comparison.
- **Sufficient remedy:** predefine a natural-task cohort and weighting tied to a stated workload; sample LoCoMo across conversations; use the primary semantic metric or a validated deterministic surrogate; then repeat the disjoint latency calibration and cell-aware inference. The current paper should keep this result strictly diagnostic, as it mostly does.

## W5. The distillation objective discards unlogged probability mass, preventing audit of objective fidelity. **Minor**

- **Location:** Sec. 4, PDF p. 4, lines 263–278; Table 27, PDF p. 20.
- **Exact quote (7 words):** “We did not retain the teacher mass”
- **Problem:** Teacher and student are renormalized only on the teacher top-64 support. Without the retained teacher mass, readers cannot determine whether this approximates full-vocabulary distillation uniformly across training tokens or whether a variable amount of probability is discarded.
- **Affected claim/norm:** technical transparency and reproducibility of the adapter objective. This does not invalidate the empirical adapter gain, but it blocks analysis of why it works and how faithfully another implementation can reproduce the loss.
- **Sufficient remedy:** report the distribution (mean, quantiles, minimum) of teacher top-64 retained mass on training and held-out windows; ideally add a full-vocabulary or larger-support ablation on a controlled subset.

# Explicit Audit of Local Consistency and Heterogeneous Bootstrap

## Local inconsistencies

The version is **materially improved but not completely free of local inconsistencies**.

- **Closed:** RULER Cohorts A and B are now explicitly separated, labelled, and not subtracted across cohorts; 96.07 is mechanically reconstructed as `1441/15` in Table 36. Timing boundaries are repeatedly separated. The matched raw-text endpoint is described as same-LoRA in the central experiment. Abstract numbers map to tables. Formula and storage arithmetic are consistent.
- **Remaining editorial inconsistencies:**
  1. Table 18 reports the matched raw-text RULER value as **99.20**, while Tables 2, 31, 32, the abstract, and `1441/15`-based paired accounting imply/report **99.19**. This is small rounding, but a frozen manuscript should use one convention.
  2. Table 18 says MemoryLLM uses a released **7B** model, while Tables 19–22 and their captions call it **Llama-3-8B-chat**. This should be corrected because backbone mismatch is central to how the external reference is interpreted.
  3. Table 8 labels a row “Original v5 analysis,” which is revision-history/meta wording in the submitted paper and should be replaced by a neutral label such as “pooled-IID sensitivity.”

These are Minor editorial issues rather than reasons to reject the central matched claim, but they mean the “local inconsistency” cleanup is not fully closed.

## Heterogeneous bootstrap

This issue is **closed at the statistical-method level**. The paper now gives: (i) the estimand; (ii) nine cell definitions and equal weights; (iii) per-cell deltas; (iv) fixed-cell stratified paired bootstrap; (v) hierarchical cell-then-example bootstrap; (vi) leave-one-cell-out ranges; and (vii) the original pooled-IID result only as sensitivity. The conclusions follow these analyses: BM25 replay remains ahead; BGE is unresolved; CoMem wins neither aggregate. W4 concerns the scientific meaning of the chosen mixture, not an unmodeled dependence error.

# Baselines, Metrics, Seeds, Statistics, Compute, and Reproducibility

- **Baselines:** The matched `j=0` endpoint is excellent for isolating depth reuse. Same-`j` adapter controls, continuous-prefix oracle, selector diagnostics, KV-compression references, and natural-task references are useful. The decisive missing baseline is a same-backbone nearest reusable-context cache (W1).
- **Benchmark validity:** RULER/BABILong are appropriate controlled diagnostics. LongEval, LongBench, and LoCoMo broaden scope, but the paper correctly reports incomplete overlap auditing for several natural datasets and removes the contaminated InfiniteBench comparison.
- **Metrics:** Official benchmark scorers are named. LoCoMo's mutable GPT-4o judge is dated and supplemented by conversation-cluster bootstrap and a DeepSeek audit, but exact future reproducibility is impossible. The equal-latency LoCoMo surrogate is weaker than the headline metric (W4).
- **Seeds/statistics:** Evaluation seeds and supports are reported. Paired bootstrap and McNemar are appropriate for the central RULER comparison. Conversation clustering is appropriately used for full LoCoMo. Training variance remains inadequately isolated (W3).
- **Compute:** Final training is reported as about 2.9 H20 GPU-hours, with hardware and throughput. Training peak memory and total project GPU-hours were not retained; this is transparently acknowledged. Several timing tables use H20 and one uses L20A, with cross-hardware ratios generally avoided.
- **Reproducibility:** Strong configuration detail and hashes support reimplementation, but the review could not inspect the claimed anonymous artifact because only the frozen PDF/source were in scope. Reproducibility is therefore 3.0 rather than higher.

# Figure and Table Audit

I inspected every rendered PDF page and all figures/tables in the frozen source.

- **Figure 1:** readable and accurately distinguishes Write/Select/Read, matched `j=0`, and overlap-Write.
- **Figure 2:** readable but highly compressed; it is explicitly motivational, with protocol details in the appendix.
- **Tables 1–36:** no unresolved references or visibly clipped tables were found. Several appendix tables use small type but remained legible at zoom. Main tables correctly flag unmatched comparisons and timing exclusions.
- **Notable table issues:** Table 18's 99.20/99.19 rounding mismatch and 7B/8B MemoryLLM mismatch; Table 8's “v5” meta-label; these are listed above.

# Abstract Five-Number Check

Five load-bearing numerical groups were checked against the frozen tables/source:

1. **931.9 -> 664.4 ms; 1.403x:** Tables 2 and 31; `931.9/664.4 = 1.4026`, rounding to 1.403.
2. **99.19 -> 96.07; gap 3.12; CI `[2.36,3.93]`:** Tables 2, 31, 32 and Appendix A.5; consistent (apart from Table 18's 99.20 rounding).
3. **8 KiB/token; 1/18 full KV; about 1 GiB at 128k:** Eq. 1 and Sec. 4; arithmetic is consistent.
4. **32k CPU break-even 8.9–10.9 for `G<=128`; full grid 5.5–10.9; 128k `G=1` 25.8–27.6:** Table 3; consistent.
5. **Equal-latency deltas and intervals:** Table 4 and Table 8; BM25 `53.22-64.78=-11.56`, BGE `53.22-54.22=-1.00`; hierarchical intervals match the appendix.

The overlap values 92.5/100.0/98.5 were also checked against Tables 5–6 and the abstract.

# Citation Audit

## Coverage

Mechanical source parsing found **43 unique citation keys and 43 entries in `main.bbl`**. Every cited key appears in `main.bbl`, and every `main.bbl` entry is actually cited. There are no orphan or missing citation keys.

Per the user's stop instruction, external network verification was not completed; therefore **all 43 entries are marked Unverifiable rather than Not found**. No entry is labelled Verified or Metadata error based solely on self-supplied `.bib/.bbl` data.

- `cachecraft` — Unverifiable
- `longbench` — Unverifiable
- `pyramidkv` — Unverifiable
- `kvpacket` — Unverifiable
- `cartridgesbase` — Unverifiable
- `hcache` — Unverifiable
- `llama3` — Unverifiable
- `cartridges` — Unverifiable
- `distillation` — Unverifiable
- `ruler` — Unverifiable
- `lora` — Unverifiable
- `epic` — Unverifiable
- `ragcache` — Unverifiable
- `babilong` — Unverifiable
- `rag` — Unverifiable
- `longchat` — Unverifiable
- `snapkv` — Unverifiable
- `ilre` — Unverifiable
- `readonce` — Unverifiable
- `minicache` — Unverifiable
- `turborag` — Unverifiable
- `locomo` — Unverifiable
- `xccache` — Unverifiable
- `kvdirect` — Unverifiable
- `pg19` — Unverifiable
- `bm25` — Unverifiable
- `embeddingrecycling` — Unverifiable
- `gemfilter` — Unverifiable
- `reform` — Unverifiable
- `lloco` — Unverifiable
- `hunyuan` — Unverifiable
- `fusionrag` — Unverifiable
- `mepic` — Unverifiable
- `longmem` — Unverifiable
- `memoryllm` — Unverifiable
- `infllm` — Unverifiable
- `streamingllm` — Unverifiable
- `sempic` — Unverifiable
- `xu2024retrievallong` — Unverifiable
- `qwen3` — Unverifiable
- `ape` — Unverifiable
- `cacheblend` — Unverifiable
- `h2o` — Unverifiable

## Load-bearing citation–claim matches (semantic audit from titles/descriptions; external metadata Unverifiable)

1. **RAG recomputes all layers after retrieving text** — `rag`, `xu2024retrievallong`: citation direction is plausible and appropriately broad.
2. **CacheBlend/TurboRAG/RAGCache/Cache-Craft reuse or precompute retrieved chunk KV** — corresponding citations: semantically matched to the stated serving family.
3. **EPIC/MEPIC/APE are position-independent/parallel encoding systems with repair or realignment** — titles and paper descriptions align with the taxonomy.
4. **ReadOnce/Embedding Recycling precede intermediate representation reuse** — citation-claim match is direct.
5. **HCache/KV-Direct concern activation/residual restoration or KV reconstruction** — direct match to the stated precedent.
6. **StreamingLLM/H2O/SnapKV/PyramidKV/MiniCache are token/KV retention or compression methods** — appropriate family-level support.
7. **Distillation and LoRA motivate the training method** — standard and directly matched.
8. **RULER, BABILong, LongBench, LongEval/LongChat, and LoCoMo citations identify the benchmark sources** — appropriate.

No citation-claim mismatch was established from the frozen material, but external verification remains Unverifiable.

# Novelty Search Summary and Cutoff

The novelty search was stopped before a complete external search, so all search conclusions are **Unverifiable**. The closest works identified from the paper's own cited landscape are:

1. **ReadOnce Transformers** — reusable intermediate text representations; older, load-bearing precedent.
2. **Embedding Recycling** — intermediate representation reuse; older precedent.
3. **HCache** — activation checkpoint/restoration with suffix replay; close in interface behavior.
4. **KV-Direct** — residual-to-KV reconstruction; close in stored-state representation.
5. **CacheBlend / EPIC / Cartridges** — closest practical reusable-context cache families for the serving workload.

The frozen bibliography also includes 2026 works such as KV Packet, Cartridges at Scale, and SemPIC. Their exact first-public dates and whether they fall within the ARR three-month contemporaneous-work window were not fully verified; hence novelty-cutoff classification is Unverifiable. Importantly, the manuscript itself avoids claiming invention of reusable document caches or intermediate-state reuse and frames the contribution as a matched internal measurement of one residual-per-token depth-reuse point. That narrower positioning is defensible, but W1 still limits excitement.

# Limitations, Ethics, and Desk-Reject Risks

## Desk audit

- **Page limit:** long-paper main content ends with the Conclusion on PDF p. 7; Limitations and Ethics occupy pp. 7–8; references start on p. 8; appendices follow references. The eight-page content limit is respected.
- **Required section:** exact unnumbered title `Limitations` appears after Conclusion and before References.
- **Style:** A4, two-column ACL review style, 11pt source setting, review line numbers, embedded fonts; appendix is two-column.
- **Anonymity:** title block is anonymous; no acknowledgments or author-identifying URL/path was found in the rendered PDF/source.
- **References/placeholders:** no unresolved refs/cites, TODO/TBD/FIXME, or placeholder markers found.
- **Reviewer manipulation/hidden text:** no prompt injection, white text, invisible text, or machine-reader manipulation found. Small fonts are used for tables in the normal visible way.
- **Figures/tables:** readable at zoom; no clipping found.

**Desk-reject risk: low** on the frozen PDF/source. One mild cleanliness issue is the visible “Original v5 analysis” label, but it is not an anonymity breach by itself.

## Ethics

The ethics discussion is adequate and relevant: residual inversion/membership risk, sensitive retrieval, authorization, isolation, encryption, deletion, misuse, monitoring, data provenance, licenses, and energy are addressed. No new human-subject data or annotators are used. The major ethical/reproducibility caveat is the mutable external judge endpoint, which the paper discloses.

# Questions That Could Change the Score

1. Can the authors provide a same-Qwen3-8B, same-hardware result for at least one nearest PIC/modular-KV baseline with matched storage/fetch/TTFT boundaries? A credible non-dominated operating point could raise excitement and Overall.
2. Does `w=32` overlap improve the full matched LongEval and LoCoMo results, and what is the repaired Write-inclusive break-even? A natural-task improvement could materially strengthen the diagnosis.
3. Do same-batch independent adapter seeds reproduce the exact RULER-B, LongEval, and LoCoMo headlines? Large run-to-run variance would lower soundness; stable replication would raise it.
4. What fraction of teacher probability mass lies in the top-64 support during training and held-out evaluation?
5. Which MemoryLLM checkpoint size was actually used—7B or Llama-3-8B-chat—and will all 99.19/99.20 RULER reporting be normalized?

# Non-Scoring Suggestions and Typos

1. Replace “Original v5 analysis” in Table 8 with “pooled-IID sensitivity.”
2. Standardize the matched raw-text RULER macro to 99.19 (or state a consistent rounding policy).
3. Standardize MemoryLLM as 7B versus 8B-chat throughout.
4. “Student-t intervals” would be clearer as “Student's t intervals.”
5. Table 25 says NIAH score is 90 at 2M despite recall 1.00; this is plausible reader error, but a short note would prevent readers from interpreting recall/score mismatch as a typo.
6. Report exact wall-clock overlap-Write timings alongside theoretical FLOP ratios in Table 6 if available.

# Scores

## Soundness: 3.0 / 5.0

The central fixed-pack endpoint and the revised equal-latency statistics are credible, and claims are generally scoped to evidence. Soundness is limited by single-run adapter training, lack of natural-task validation for the repair, and the absence of a matched nearest reusable-cache baseline.

## Excitement: 3.0 / 5.0

The depth-reuse axis and transparent systems accounting are interesting, but intermediate-state reuse and reusable document caches have strong precedents. The paper establishes an internal operating point rather than a clear state-of-the-art or non-dominated practical design.

## Overall: 3.0 / 5.0 — Findings

This is a careful, useful empirical measurement paper with negative results and a well-controlled central endpoint. It meets a Findings bar. It does not meet my main-conference 4.0 bar because the practical comparison to the nearest cache systems is absent, the repair is synthetic-only, and training robustness is not cleanly replicated.

## Confidence: 4.5 / 5.0

I read the complete 24-page PDF twice, inspected all source sections, appendices, figures, tables, equations, abstract numbers, citation keys, and desk criteria. Confidence is not 5.0 because external citation metadata and novelty searches were stopped and therefore remain Unverifiable.

## Reproducibility: 3.0 / 5.0

The paper gives strong configuration, hashing, benchmark, environment, and timing detail. Exact judge reproducibility is impossible, total project compute is incomplete, the artifact was outside the permitted frozen inputs, and the exact flagship claims lack clean same-batch multi-seed replication.

# Review-Process Self-Check

- Reviewed only the frozen v6 PDF/source and STRICT template; no prior review, score history, TODO, status, current, or calibration file was read.
- Completed two manuscript passes including all appendices and visually inspected all 24 rendered pages.
- Built claims C1–C8 and compared minimum sufficient experiments to actual evidence.
- Checked formulas, boundary cases, baselines, benchmarks, metrics, seeds/statistics, compute, reproducibility, desk criteria, all figures/tables, and five abstract number groups.
- Mechanically confirmed that all 43 cited keys appear in `main.bbl`, all 43 BBL entries are cited, and no citation key is missing.
- Mechanically grepped every weakness quote against the frozen source. No weakness asserts an absent item that is actually present in the appendix.
- External network verification was not completed after the stop instruction; affected citation and novelty items are marked Unverifiable, never Not found.
- Score calibration applied exactly as requested: **Overall 4.0 = main-conference; Overall 3.0 = Findings**. This review assigns 3.0.
