```yaml
review_mode: strict
soundness: 3.5
excitement: 3.5
overall: 3.0
confidence: 4.5
reproducibility: 3.5
```

# Paper Summary

This paper studies a narrow but useful systems question: for a reusable long-context document object, how much transformer depth should be prepaid? CoMem independently writes each 512-token chunk through layers `[0:j)`, stores one residual vector per token, retrieves a bounded top-k chunk set, and resumes layers `[j:L)` jointly with the query. Its strongest result is an internally matched `j=0` versus `j=12` frontier on Qwen3-8B: the selected pack, order, mask, examples, and LoRA are held fixed; model Read decreases from 931.9 to 664.4 ms (1.403×), while the paired 15-cell RULER macro decreases from 99.19 to 96.07 (PDF p.4 lines 272–286; Table 2, PDF p.5 lines 1–14). The paper also measures write amortization (Table 3), reports an equal-online-latency negative result against raw-text replay (Table 4), and performs a focused context/position/overlap diagnosis (Tables 5–6). The appendix is unusually extensive: complete benchmark tables, storage/I/O measurements, uncertainty, seed robustness, prompt sensitivity, and an 80-layer MoE port are included.

My assessment is **Findings-level**. The internally matched causal frontier is careful, honest, and reasonably convincing. However, two decision-relevant external claims remain under-specified or unmatched: the equal-latency diagnostic is not reproducible from the paper, and there is no same-backbone end-to-end comparison against the closest reusable-cache systems. A further scientific weakness is that the natural-task headline quality is judged by an undated mutable API model. These issues prevent a main-conference score, but they do not invalidate the central internal result.

# Claims

- **C1 — Method/interface.** A document can be represented by one persistent depth-`j` residual per token, followed by bounded retrieval and upper-layer continuation (`[j:L)`). **Evidence:** Section 4, PDF p.3 lines 193–203 and p.4 lines 204–218; Figure 1, PDF p.2 lines 1–49.
- **C2 — Storage and bounded model-side Read.** For Qwen3-8B, a bf16 residual is 8 KiB/token versus 144 KiB/token for full per-layer KV (1/18), and Read length is `sink + kc + query`, independent of stored-context length. **Evidence:** Eq. (1), PDF p.4 lines 206–218; store-scaling Table 23, PDF p.17 lines 1–13.
- **C3 — Matched depth-reuse frontier.** Under identical evidence, order, mask, examples, and LoRA, `j=12` gives 1.403× faster isolated model Read but loses 3.12 RULER points relative to `j=0`. **Minimum sufficient experiment:** paired examples; identical retrieved packs and adapter; same hardware/timing boundary; paired uncertainty. **Actual evidence:** Table 2 and Table 28 satisfy this well (PDF p.5 lines 1–14; p.19 lines 26–40), including CI and process-level timing details (Appendix A.4, PDF p.18 lines 864–891).
- **C4 — Self-distillation makes the split interface usable.** A rank-32 upper-layer LoRA substantially improves same-`j` quality without updating the backbone. **Minimum sufficient experiment:** same split, selector, cohort, and backbone with adapter on/off. **Actual evidence:** Table 10 on RULER and Table 8 on LoCoMo/BABILong (PDF p.11 lines 17–46; p.10 lines 1–27).
- **C5 — Write amortization requires repeated queries.** The one-time residual Write breaks even after roughly 8–11 queries at 32k and 26–28 at 128k under stated placements/generation lengths. **Minimum sufficient experiment:** include Write, fetch, Read, and decode for both arms across query counts. **Actual evidence:** Table 3, PDF p.5 lines 17–35 and text lines 297–305; hardware dependence is explicitly acknowledged.
- **C6 — Equal-online-latency raw-text replay is better in the tested cohort.** Top-10 raw replay scores 64.78 versus 53.22 for top-12 CoMem. **Minimum sufficient experiment:** define the cohort/tasks/metric/sample counts, publish calibration procedure and per-arm latency distributions, freeze the choice on a disjoint split, and provide paired predictions. **Actual evidence:** only Table 4 and a short paragraph (PDF p.5 lines 17–29, 306–313); this is insufficiently specified for independent checking.
- **C7 — Lower-layer document context is a major cause of one synthetic multikey failure.** The 2×2 context/position control moves 92.5 to 100.0 with document-contextual writing; position remapping alone lowers performance to 88.0. **Minimum sufficient experiment:** paired examples with retrieval, adapter, and upper Read fixed. **Actual evidence:** Table 5 and Appendix Table 29 (PDF p.5 lines 31–45; p.19 lines 50–68).
- **C8 — Overlap-Write is a local deployable repair.** A 32-token left overlap raises the focused multikey macro from 92.5 to 98.5 without increasing persistent bytes or per-query Read work. **Evidence:** Table 6, PDF p.6 lines 1–18. The paper appropriately limits this to a synthetic diagnostic.
- **C9 — Portability, not replicated quality generality.** Split-forward semantics port to other Qwen sizes and an 80-layer sparse MoE; Hy3 shows exact partition on a self-test and bounded Read through 256k. **Evidence:** Appendix A.7, PDF p.19 lines 937–955 and p.20 lines 1–54. The paper explicitly does not claim a scaling law or matched replication.

# Strengths

## S1. The central causal comparison is unusually well controlled.
The strongest contribution is not the large dense-context speedup but the same-pack `j=0`/`j=12` experiment. The paper holds retrieved chunk IDs/order, mask, examples, and adapter fixed, provides a paired bootstrap CI, and separates Read from Write/I/O/decode (Section 5.1, PDF p.4 lines 272–286; Table 2, PDF p.5 lines 7–14; Table 28, PDF p.19 lines 26–40). This directly supports C3 and avoids conflating retrieval sparsification with depth reuse.

## S2. Negative results and system boundaries are reported transparently.
The paper states that CoMem is not quality-preserving, that raw text wins at equal online latency in its diagnostic, that the 64.9× number combines selection and depth reuse, and that residual storage is far larger than text (Abstract, PDF p.1 lines 20–34; Introduction, p.1 lines 57–79; Table 7, p.10 lines 9–18). This restraint materially increases trust.

## S3. The mechanism diagnosis is well factorized and bounded.
The context-scope × Read-position intervention and continuous-prefix oracle distinguish interface compatibility from a narrower Write-side failure (Tables 5–6, PDF pp.5–6; Table 29, p.19 lines 50–68). Importantly, the paper does not overgeneralize the overlap repair to natural tasks (p.6 lines 8–18; Limitations p.6 lines 399–410).

## S4. Reproducibility reporting is substantially above average.
The appendix gives backbone revision, shapes, RoPE settings, mask/pack semantics, exact optimizer and LoRA targets, seeds, sample counts, scorers, hardware, hashes, timing boundaries, store tiers, and judge aggregation (Tables 24–26, PDF pp.17–18; Appendix B, pp.20–21). Training-seed robustness and conversation-cluster bootstrap analyses are also provided (PDF p.18 lines 892–907; p.21 lines 1022–1041).

## S5. The paper distinguishes several otherwise-confusable operating points.
Selected-pack model Read, store-ready prefill, Write-inclusive pipeline, and external store/index costs are explicitly separated (Section 5 Setup, PDF p.4 lines 257–271). This is essential for interpreting a systems paper with a persistent object and prevents the headline 1.403× result from being confused with end-to-end latency.

# Weaknesses

## W1. The decision-relevant equal-latency experiment is not independently reproducible from the paper. **Major.**
- **Affected claim/norm:** C6; ARR soundness and reproducibility.
- **Location:** Section 5.2/Table 4, PDF p.5 lines 17–29 and 306–313; source `sections/tab_equal_latency.tex:17–21`.
- **Verified quote (19 words):** “The retrieval budget is chosen on a disjoint latency-only calibration split, then frozen and evaluated on a mixed diagnostic cohort.”
- **Problem:** The paper never defines the mixed cohort's constituent tasks, metric, number of examples, source lengths, generation budget, exact latency anchor/distribution, or calibration sample size. A full-source grep finds the 64.78/53.22 result only in the abstract/introduction, Section 5.2, and Table 4; no appendix table supplies the missing cells or protocol. The caption gives a paired-bootstrap CI, but readers cannot reconstruct what was paired.
- **Impact:** This is labeled the “decision-relevant negative result” and is used to delimit practical usefulness. Without the cohort and latency protocol, neither the score nor the claim that the arms are matched within ±5% is auditable.
- **Sufficient remedy:** Add a complete table listing every task/source length/sample count and aggregation rule; report calibration split size and frozen choice rule; give per-arm median/p10/p90 latency under the same hardware and timing boundary; include per-example predictions/IDs or hashes in the anonymous artifact.

## W2. The closest systems question is left unanswered by unmatched baselines. **Major.**
- **Affected claim/norm:** practical relevance of C1–C5; ARR excitement/completeness for a systems contribution.
- **Location:** Related Work, PDF p.2 lines 130–137; Limitations, p.6 lines 373–380; Conclusion, p.6 lines 357–362.
- **Verified quote (11 words):** “We lack a matched same-backbone implementation of these systems.”
- **Problem:** The internal depth frontier is valid, but CoMem's practical object competes with PIC/chunk-KV/learned modular caches. Table 1 is explicitly taxonomic, while the empirical baselines either answer different questions (full-prompt compression), use different backbones, or are descriptive external references. Thus the paper cannot establish whether a single residual/token is a useful Pareto point once boundary repair, store I/O, Write, and quality are compared end to end.
- **Impact:** This limits novelty/excitement and prevents a main-conference-level systems conclusion. It does **not** invalidate C3; it limits the practical interpretation of C1/C2/C5.
- **Sufficient remedy:** Implement at least one closest method (e.g., EPIC/CacheBlend/TurboRAG or a learned modular-KV method) on the same Qwen3-8B, tasks, retrieval packs, hardware, storage tier, and timing boundary; report quality, persistent bytes, Write cost, TTFT/decode, and crossover. If implementation is impossible, narrow the paper throughout to a controlled measurement study and remove broad deployment framing.

## W3. The primary natural-task semantic metric is not exactly reproducible because the judge snapshot is mutable. **Major.**
- **Affected claim/norm:** C4 and external quality comparisons on LoCoMo; ARR reproducibility/statistical validity.
- **Location:** Appendix B.2, PDF p.21 lines 1006–1021; source `sections/08_statistics_appendix.tex:77–85`.
- **Verified quote (9 words):** “The endpoint does not expose a dated model snapshot.”
- **Problem:** The paper uses model name `gpt-4o`, seed 1, and no explicit temperature/top-p, but no dated snapshot is available. API behavior can change. The independent DeepSeek-V3 audit is helpful but only covers a stratified 200-item subset and itself is not snapshot-pinned.
- **Impact:** The 38.27/34.59 full-set points and +4.81 answerable-item difference may not be exactly reproducible later. Because LoCoMo is the main natural-task semantic comparison, this matters more than a cosmetic artifact issue.
- **Sufficient remedy:** Use a pinned dated judge if available; otherwise release all raw judge responses and rerun the full 1,540 answerable-item comparison with a fixed open-weight judge, reporting agreement and paired/cluster intervals. Deterministic lexical metrics should remain secondary, as the paper correctly notes their formatting sensitivity.

## W4. Evaluation uncertainty is stronger than training robustness; the added “seeds” change effective batch size. **Minor.**
- **Affected claim/norm:** C4 robustness; seed/statistics audit.
- **Location:** Appendix A.5, PDF p.18 lines 892–907; source `sections/08_appendix.tex:304–312`.
- **Verified quote (12 words):** “The two added seeds use effective batch 3 rather than 8.”
- **Problem:** The paper reports three adapters, but the two added runs do not preserve effective batch size, so seed and optimization-noise effects are confounded. The main natural-task benchmark is still based on one flagship adapter, and most CIs are evaluation resampling rather than training-run uncertainty.
- **Impact:** This does not undermine the large same-`j` adapter gains, but it weakens claims about stability of the trained interface and depth curve.
- **Sufficient remedy:** Train at least three fully matched runs (batch, optimizer, schedule, data order except seed) and report mean/SD or hierarchical intervals on the headline RULER and LoCoMo comparisons.

## W5. Several actually cited references use stale preprint metadata, and one author list is wrong. **Minor.**
- **Affected claim/norm:** ARR citation accuracy.
- **Location:** `main.bbl`, rendered References PDF pp.7–9.
- **Verified quote (14 words, Related Work source lines 34–36):** “Token/KV compression instead changes the retained token set or state budget after full or partial processing.”
- **Problem:** The bibliography audit found 11 metadata issues: LongBench, PyramidKV, RAGCache, SnapKV, MiniCache, TurboRAG, LoCoMo, LLoCO, InfLLM, and CacheBlend omit known proceedings metadata; PG-19's author list includes Chloe Hillier, absent from arXiv:1911.05507 metadata. These are real works, not fabricated citations.
- **Impact:** Low scientific impact, but the accumulation is below publication-ready bibliography quality and obscures which versions were actually used.
- **Sufficient remedy:** Update all entries to proceedings/DOI metadata where available and correct PG-19 authors; retain arXiv identifiers as secondary fields if desired.

# Questions That Could Change the Score

1. **Equal-latency cohort:** What exact tasks, lengths, sample counts, and metric compose the 64.78/53.22 cohort? Please provide the calibration and held-out splits plus per-arm latency quantiles. A complete, pre-specified protocol could substantially reduce W1.
2. **Closest baseline:** Can the authors provide one same-Qwen3-8B, same-hardware end-to-end comparison to a closest modular/PIC cache, including Write, store tier, retrieval, TTFT, decode, quality, and persistent bytes? A convincing result could raise excitement/overall.
3. **Judge stability:** Are raw GPT-4o decisions and API timestamps retained? Can the full answerable set be rescored with a fixed open-weight judge rather than only a 200-item audit?
4. **Training robustness:** Why did the added adapters use effective batch 3? Are fully matched multi-seed runs available for the headline cells?
5. **Submission material:** The frozen source states that an anonymous archive accompanies the release, but that archive was not among the allowed snapshot files. Is it actually attached to the ARR submission? This affects reproducibility, not soundness of the PDF evidence.

# Suggestions / Typos (Non-scoring)

- Define “model Read” once in the main text with an explicit inclusion/exclusion list; the paper eventually does this well, but readers must synthesize several captions.
- Table 12 says “single-pass selector sweep,” while the flagship uses iterative BM25. The caption does warn readers not to conflate them; making the table title “non-flagship single-pass sweep” would be clearer (PDF p.11 lines 17–42).
- Table 13's row label “Raw-text reader quality” is ambiguous: clarify whether it is unconditional score or hit-conditional accuracy (PDF p.12 lines 1–9).
- References on PDF pp.7–9 leave page 9 mostly blank; this is not a violation, but bibliography compaction could improve presentation.
- The appendix is information-rich but visually sparse on pp.14–15; combining tables or using a landscape appendix page could reduce page count without shrinking text.
- The source comment in `main.tex` mentions saved result shards and a date; it is not rendered and is not a reviewer-manipulation issue, but production source should avoid internal process comments.

# Score Rationales

## Soundness: 3.5 / 5.0
The central internally matched frontier (C3), storage equation (C2), adapter on/off ablations (C4), and focused mechanism controls (C7–C8) are sound and unusually transparent. Formula (1) is correct under the stated common-dtype, full-layer-KV assumptions: for Qwen3-8B, `nq/(2L nkv)=32/(2·36·8)=1/18`; 4096 bf16 elements give 8192 bytes/token. Boundary cases are stated: `j=0` is raw-token replay; `j=L` is only an exact partition self-test, not a useful deployable reader; top-k imposes an evidence ceiling. I lower the score because C6 is under-specified, closest-system practical validity is unmatched, and LoCoMo's primary judge is mutable.

## Excitement: 3.5 / 5.0
Treating depth as a cross-query reuse axis and measuring a clean depth/quality/storage frontier is conceptually interesting. The single-residual object is meaningfully different from per-layer KV objects, and the negative results make the paper useful. Excitement is capped by strong overlap with older intermediate-representation reuse and recent modular-cache work, and by the absence of a matched end-to-end nearest-baseline result.

## Overall: 3.0 / 5.0
This meets my bar for **Findings**: reliable central evidence, a clear bounded contribution, and real limitations. It falls below ACL main-conference level because the practical systems comparison is incomplete and the equal-latency result cannot be audited from the paper. I considered 3.5, but under strict calibration I choose the lower bin: W1–W3 are claim-linked, not merely requests for more experiments.

## Confidence: 4.5 / 5.0
I read the rendered 21-page PDF twice, including all appendices; inspected all rendered figures/tables (2 figures and 34 numbered tables); cross-checked formulas, numbers, labels, and quotes against the frozen source; audited all 43 actual `main.bbl` entries; and performed targeted novelty searches. Confidence is not 5.0 because the anonymous artifact was outside the allowed snapshot and one model-card citation remained Unverifiable after research was stopped.

## Reproducibility: 3.5 / 5.0
Configuration reporting is excellent: revisions, hashes, hardware, optimizer, masks, positions, scorers, sample counts, timing boundaries, and store tiers are documented (Tables 24–26). Reproducibility is reduced by the missing equal-latency cohort definition, mutable GPT-4o judge snapshot, unmatched training seeds, incomplete total compute logging, and the fact that the claimed archive was not part of the permitted review snapshot.

# Limitations / Ethics / Desk-Reject Risks

## Desk audit

- **Paper type/page boundary:** Long paper. Main content ends with Conclusion on PDF p.6 line 362. Limitations begins at p.6 line 363 and continues to p.7 line 420; Ethical Considerations follows at p.7 line 421; References begin at p.7 line 463; Appendix begins at p.10 line 649. Therefore the countable main content is **6 pages**, within the ARR long-paper limit of 8. No short-paper ambiguity.
- **Exact Limitations:** Present as exact heading `Limitations` before references (`\section*{Limitations}`), satisfying the desk rule. It discusses caveats and open scope rather than introducing new results (PDF pp.6–7 lines 363–420).
- **Ethics:** Present as `Ethical Considerations` (PDF p.7 lines 421–462). It covers hallucination/bias, sensitive-memory disclosure, inversion/membership inference, authorization/deletion, energy, data provenance, and redistribution constraints. No unresolved ethics violation identified.
- **Anonymity:** Author is “Anonymous ACL submission”; no rendered repository URL, author identity, affiliation, or self-identifying acknowledgment found. The cited Hy3/model URLs are third-party resources. The source's internal comments are not rendered.
- **ACL format:** Uses `\usepackage[review]{acl}`, 11pt article, A4 PDF, line numbers, two columns, and readable tables. No style-file modification was detected beyond bundling `acl.sty`; no apparent margin/font violation in rendered inspection.
- **Hidden/manipulative text:** Mechanical grep found no white/invisible text commands, prompt injection, reviewer-directed scoring language, or manipulation. `\scriptsize` is used for tables in ordinary visible form. PDF has no JavaScript/forms/attachments.
- **Placeholders/references:** No rendered TODO/FIXME/TBD/`??`, no missing `\ref` labels, and all 43 citation keys appear in `main.bbl`. One source occurrence of “undefined” is ordinary prose (“Oracle is undefined”), not a dangling reference.
- **Desk-risk conclusion:** **No evident desk-reject condition** in the frozen PDF/source. Remaining bibliography metadata errors are editorial, not desk-level.

## Abstract-number audit (at least five)

| Abstract claim | Cross-check | Result |
|---|---|---|
| 931.9 → 664.4 ms, 1.403× | Table 2 (PDF p.5 lines 1–14); Table 28 (p.19 lines 26–40) | Verified; 931.9/664.4 ≈ 1.4026. |
| RULER 99.19 → 96.07 | Table 2; Table 28; Appendix B cohort macro (PDF p.20 lines 1–9) | Verified; difference 3.12. |
| Break-even 8–11 at 32k | Table 3: 8.9, 9.2, 10.9 for CPU G≤128 and 8.4, 7.7, 5.5 for GPU | Verified as an approximate range; 512-token CPU case is separately 94.0. |
| Break-even 26–28 at 128k | Table 3: 25.8 GPU and 27.6 CPU for G=1 | Verified; only one-token-generation cells are available. |
| Equal latency 64.78 vs 53.22 | Table 4 (PDF p.5 lines 17–29) | Numerically verified; protocol under-specified (W1). |
| Context 92.5 → 100.0; position-only 88.0 | Table 5 (PDF p.5 lines 31–45) | Verified. |
| 32-token overlap 98.5 | Table 6 (PDF p.6 lines 1–18) | Verified. |
| 8 KiB/token | Eq. (1), PDF p.4 lines 206–218; Table 2 | Verified from d=4096, bf16: 4096×2=8192 bytes. |

# Complete Citation Audit (`main.bbl` is authoritative)

**Procedure/status rule.** I audited every one of the 43 entries actually present in `main.bbl`, using DOI resolution, arXiv API metadata, OpenAlex indexing, and official/primary records available before research was stopped. “Metadata error” means the work is real but the rendered entry is stale or incorrect. Network/API failure was never converted to “Not found.” Totals: **31 Verified, 11 Metadata error, 0 Not found, 1 Unverifiable**.

| Key | Status | Verification / issue |
|---|---|---|
| `cachecraft` | Verified | DOI 10.1145/3725273 resolves to the stated 2025 PACM article; title/authors match. |
| `longbench` | Metadata error | Work is real (arXiv:2308.14508), but main.bbl lists only the 2023 preprint; ACL 2024 publication DOI 10.18653/v1/2024.acl-long.172 exists. |
| `pyramidkv` | Metadata error | Work/title/arXiv:2406.02069 verified; a 2024 conference version DOI 10.52202/079017-4443 exists, but main.bbl gives only the preprint. |
| `kvpacket` | Verified | arXiv:2604.13226 title/authors/date (2026-04-14) match. |
| `cartridgesbase` | Verified | arXiv:2506.06266 title/authors/date match. |
| `hcache` | Verified | DOI 10.1145/3689031.3696072 resolves to the stated EuroSys 2025 paper. |
| `llama3` | Verified | arXiv:2407.21783 title and year match. |
| `cartridges` | Verified | arXiv:2606.04557 title/authors/date (2026-06-03) match; correctly treated as concurrent. |
| `distillation` | Verified | arXiv:1503.02531 title/authors/year match. |
| `ruler` | Verified | arXiv:2404.06654 title/year match. |
| `lora` | Verified | Title/authors and ICLR 2022 publication match; preprint is arXiv:2106.09685. |
| `epic` | Verified | arXiv:2410.15332 title/authors match; main.bbl's ICML 2025 venue was not independently resolved in OpenAlex, but source metadata supplied by the entry is consistent with the cited work. |
| `ragcache` | Metadata error | arXiv:2404.12457 is real, but a 2025 ACM TOCS article DOI 10.1145/3768628 exists; main.bbl reports only the 2024 preprint. |
| `babilong` | Verified | Title and NeurIPS Datasets & Benchmarks 2024 publication verified (DOI 10.52202/079017-3381). |
| `rag` | Verified | Canonical NeurIPS 2020 RAG paper; title/authors/venue match. |
| `longchat` | Verified | The cited 2023 LMSYS blog title/authors match the supplied URL; this is correctly cited as a blog, not a proceedings paper. |
| `snapkv` | Metadata error | Work/title verified; a 2024 conference paper DOI 10.52202/079017-0722 exists, while main.bbl gives only arXiv:2404.14469. |
| `ilre` | Verified | arXiv:2508.17892 title/authors/date match; post-cutoff preprint, used only as related work. |
| `readonce` | Verified | DOI 10.18653/v1/2021.acl-long.554 resolves; title/authors/ACL 2021 match. |
| `minicache` | Metadata error | Work/title verified; a 2024 conference paper DOI 10.52202/079017-4443 exists, while main.bbl gives only arXiv:2405.14366. |
| `turborag` | Metadata error | arXiv:2410.07590 is real, but EMNLP 2025 publication DOI 10.18653/v1/2025.emnlp-main.334 exists; main.bbl reports only the 2024 preprint. |
| `locomo` | Metadata error | arXiv:2402.17753 is real, but ACL 2024 publication DOI 10.18653/v1/2024.acl-long.747 exists; main.bbl reports only the preprint. |
| `xccache` | Verified | DOI 10.18653/v1/2024.findings-emnlp.896 resolves; title/authors/venue match. |
| `kvdirect` | Verified | arXiv:2603.19664 title/authors/date (2026-03-20) match. |
| `pg19` | Metadata error | Cited paper exists, but main.bbl author list includes Chloe Hillier whereas arXiv:1911.05507 metadata lists Rae, Potapenko, Jayakumar, and Lillicrap. |
| `bm25` | Verified | DOI 10.1561/1500000019 resolves; title/authors/journal/volume/pages match. |
| `embeddingrecycling` | Verified | DOI 10.18653/v1/2023.findings-eacl.145 resolves; title/authors/venue match. |
| `gemfilter` | Verified | arXiv:2409.17422 title/authors/year match. A later ACL Findings 2026 version is after the cutoff and not a pre-cutoff omission. |
| `reform` | Verified | arXiv:2506.01215 title/authors/year match. |
| `lloco` | Metadata error | arXiv:2404.07979 is real, but EMNLP 2024 publication DOI 10.18653/v1/2024.emnlp-main.975 exists; main.bbl reports only the preprint. |
| `hunyuan` | Unverifiable | The cited official Hugging Face Hy3 model page is plausible and was the supplied primary source, but exact 295B/21B metadata was not fully independently checked before research was stopped. |
| `fusionrag` | Verified | arXiv:2601.12904 title/authors/date match. |
| `mepic` | Verified | arXiv:2512.16822 title/authors/date match. |
| `longmem` | Verified | Title/authors and NeurIPS 2023 work match (preprint arXiv:2306.07174 also indexed). |
| `memoryllm` | Verified | arXiv:2402.04624 title/year match. |
| `infllm` | Metadata error | Work is real and appears in NeurIPS 2024 (DOI 10.52202/079017-3801), but main.bbl reports only arXiv:2402.04617. |
| `streamingllm` | Verified | ICLR 2024 paper title is correct; preprint arXiv:2309.17453 is indexed as 2023, consistent with a 2024 conference citation. |
| `sempic` | Verified | arXiv:2607.28069 title/authors/date (2026-07-30) match; correctly concurrent/post-cutoff. |
| `xu2024retrievallong` | Verified | Title/authors and ICLR 2024 venue match; preprint arXiv:2310.03025 is indexed. |
| `qwen3` | Verified | arXiv:2505.09388 title/year/date match. |
| `ape` | Verified | arXiv:2502.05431 title/authors/year match. |
| `cacheblend` | Metadata error | arXiv:2405.16444 is real, but EuroSys 2025 publication DOI 10.1145/3689031.3696098 exists; main.bbl reports only the 2024 preprint. |
| `h2o` | Verified | NeurIPS 2023 paper title/year match; DOI 10.52202/075280-1506 and preprint arXiv:2306.14048 are indexed. |

## Load-bearing citation–claim matches

| Paper claim/location | Citations checked | Match assessment |
|---|---|---|
| Raw-text retrieval bounds an online evidence set but recomputes model layers (Introduction, PDF p.1 lines 36–40). | Lewis et al. 2020 RAG; Xu et al. 2024 Retrieval Meets Long Context LLMs | **Good.** Both are retrieval-plus-generation/long-context retrieval references; neither claims persistent intermediate-state reuse. |
| CacheBlend/TurboRAG/RAGCache/Cache-Craft reuse or precompute retrieved chunk KV and address composition (Related Work, PDF p.2 lines 110–115). | Yao 2024/EuroSys 2025; Lu 2024/EMNLP 2025; Jin 2024/TOCS 2025; Agarwal 2025 | **Good at family level.** Their abstracts/metadata concern cached knowledge/chunk KV for RAG. Bibliography versions should be updated. |
| EPIC/MEPIC are position-independent caching systems; APE uses parallel encoding/attention alignment (PDF p.2 lines 116–121). | Hu et al. 2025 EPIC; Wang et al. 2025 MEPIC; Yang et al. 2025 APE | **Good.** Titles/abstracts directly support PIC and parallel context encoding. |
| ReadOnce and Embedding Recycling cache intermediate text representations and adapt later layers (PDF p.3 lines 138–142). | Lin et al. 2021; Saad-Falcon et al. 2023 | **Good and load-bearing for novelty.** Both are genuine intermediate-representation reuse precedents; the paper appropriately narrows its novelty rather than claiming the general idea. |
| HCache restores activation checkpoints; KV-Direct reconstructs KV from residuals (PDF p.3 lines 142–146). | Gao et al. 2025; Qasim et al. 2026 | **Good.** HCache is state restoration; KV-Direct's abstract explicitly proves deterministic KV reconstruction from residual streams. |
| Token/KV compression changes retained token/state budgets (PDF p.3 lines 149–153). | StreamingLLM, H2O, SnapKV, PyramidKV, MiniCache | **Good.** All are token/KV retention/compression methods, distinct from cross-query residual persistence. |
| Self-distillation/LoRA methodology (PDF p.4 lines 228–241). | Hinton et al. 2015; Hu et al. 2022 | **Good.** Standard and accurately characterized. |
| Benchmarks and scorers (Setup, PDF p.4 lines 257–268). | RULER, BABILong, LongEval/LongChat, LongBench, LoCoMo | **Mostly good.** The cited works are the correct benchmark sources; LongBench/LoCoMo bibliography entries should point to proceedings versions. LongEval is cited via the LongChat blog rather than a dedicated benchmark paper, which is acceptable but less archival. |

# Novelty Search Summary

**Freeze date applied: 2026-08-03.** I ran targeted searches around (i) persistent intermediate/residual states for long context, (ii) modular/PIC KV caches, (iii) reusable intermediate representations, and (iv) residual-stream/KV reconstruction. I distinguish formal pre-cutoff publications from preprints/concurrent work.

| Closest work | Date/status by freeze rule | Overlap with CoMem | Difference / review consequence |
|---|---|---|---|
| **ReadOnce Transformers** (Lin et al., ACL 2021) / **Embedding Recycling** (Saad-Falcon et al., Findings EACL 2023) | Formally published before 2026-05-03; cited | Reuse intermediate text representations and adapt downstream layers. | CoMem's narrower novelty is persistent per-token residuals at tunable depth, bounded retrieval, and a measured depth/quality/storage frontier. No omission weakness. |
| **LLoCO** (Tan et al., EMNLP 2024) | Formally published; cited | Learns offline compressed document representations for efficient long-context QA. | Uses compressed learned contexts and task/domain LoRAs rather than direct depth-`j` residual continuation. |
| **EPIC / CacheBlend / Cache-Craft / TurboRAG** | Formal versions by 2025; cited | Reusable chunk/document KV, positional/boundary repair, repeated-query serving. | These are the closest practical systems comparators. CoMem stores one residual/token and recomputes an upper suffix; lack of matched comparison is W2, not a citation omission. |
| **ACRE: Query-Guided Activation Refilling** (Qian et al., ACL 2025) | Formally published before cutoff; **not cited** | Uses a bi-layer activation/KV hierarchy and query-guided refilling for long-context information seeking. | Related to intermediate-layer activation use, but not persistent cross-query document residuals or a split-depth frontier. **Suggested citation**, not scored as a major novelty failure because the technical object and workload differ. |
| **FIER** (Wang et al., Findings EMNLP 2025) and **A2ATS** (He et al., Findings ACL 2025) | Formally published before cutoff; present in `.bib` but not actually cited/`main.bbl` | Query-aware retrieval/compression of KV entries under long context. | More relevant to token/KV selection than to cross-query residual persistence. Suggest citing in the compression paragraph; omission is minor and does not negate novelty. |
| **KV-Direct** (Qasim et al., arXiv 2026-03-20) | Preprint only before cutoff; cited | Stores residual streams and reconstructs/recomputes KV. | Very close stored object, but focuses bounded-memory inference/state reconstruction rather than reusable selected document chunks and depth-prepayment. Correctly cited; cannot be counted as a missed formal-work weakness. |
| **C²KV** (arXiv 2026-07-20, accepted KDD 2026) and **Understanding Is Done Early / CoMem** (arXiv 2026-07-30) | After 2026-05-03 or concurrent/preprint | Composable compressed KV reuse; the latter appears to be a concurrent near-identical title/method. | Under the three-month rule these are **concurrent suggestions only**, not weaknesses. I make no authorship/overlap inference from the frozen anonymous submission. |

**Novelty conclusion.** The broad idea “cache intermediate representations” is not new. The credible novelty is the **specific single-residual, tunable-depth persistent object plus a carefully matched depth-reuse frontier and bounded Write-context diagnosis**. That is enough for moderate novelty, not a breakthrough. No omitted formally published pre-cutoff paper found in the searches invalidates the core claim; ACRE/FIER/A2ATS should be added for completeness.

# Review-Process Self-Check

- [x] Reviewed only the frozen PDF, frozen source directory, strict template, and public primary metadata/search results; did not read any other review, score history, TODO, status, current draft, or calibration memo.
- [x] Read the complete 21-page PDF twice, including all appendices; inspected actual rendering of every page and all 2 figures/34 numbered tables.
- [x] Built claims C1–C9 and linked every scoring weakness to a claim or ARR norm.
- [x] Every W1–W5 includes location, ≤25-word exact quote, problem, impact, sufficient remedy, and Major/Minor label.
- [x] Mechanically grep-verified all quoted weakness text against the frozen source.
- [x] Mechanically checked “missing equal-latency protocol” against all `.tex` files; only repeated headline/summary/Table 4 occurrences exist, with no full cohort definition.
- [x] Checked page limit, exact Limitations, ethics, anonymity, ACL style, hidden/reviewer-manipulation text, TODO/`??`, dangling refs, and abstract numbers.
- [x] Checked Eq. (1), Eq. (2), `j=0`/deep-split boundaries, retrieval budget boundaries, baseline fairness, metrics, seeds/statistics, claim scope, compute, and reproducibility.
- [x] Audited all 43 entries actually cited in `main.bbl`; no network failure was labeled Not found.
- [x] Checked eight load-bearing citation–claim matches and ran four novelty-search themes with the 2026-05-03 weakness cutoff.
- [x] Applied strict calibration after writing the substantive review. I chose Overall 3.0 rather than 3.5 because W1–W3 remain real, claim-linked deficiencies; I did not lower the score for experiments the paper does not claim to support.
