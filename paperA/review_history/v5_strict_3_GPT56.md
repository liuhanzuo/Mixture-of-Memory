```yaml
review_mode: strict
soundness: 3.0
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 3.5
```

# Paper Summary

The paper studies a repeated-query serving regime in which a stable document corpus is written once, a bounded set of chunks is selected per query, and inference resumes from one stored intermediate residual per selected token at split depth $j$. Its central result is deliberately narrow: on Qwen3-8B, a same-pack, same-order, same-mask, same-example, same-LoRA comparison between full replay ($j=0$) and cached residual replay ($j=12$) reduces isolated selected-pack Read latency from 931.9 ms to 664.4 ms ($1.403\times$) while reducing a paired 15-cell RULER macro from 99.19 to 96.07 (gap 3.12, 95% CI [2.36, 3.93]). The paper also reports an 8 KiB/token store, measured repeated-query crossover, a selector-dependent equal-latency diagnostic, and a focused context/position/overlap diagnosis of the chunk-local Write failure.

I read the frozen 23-page PDF twice, including all appendices, and inspected both figures and all 36 numbered tables. I treat the work as an internal design-point measurement rather than a claim that reusable residuals are a new general caching idea or that CoMem beats PIC/modular-KV systems.

## Claim–evidence map and minimum sufficient experiments

- **C1 — Matched depth reuse has a measurable quality/latency trade-off.** Minimum evidence: identical model, adapter, examples, selected tokens/order, mask, and timing boundary, varying only replay start. **Actual evidence:** Table 2 / Table 30, Appendix A.4, PDF lines 318–331 and 1003–1030; paired $n=1{,}500$ quality examples and three-process latency medians. **Supported within the isolated Read boundary.**
- **C2 — The method has bounded model-side Read work as the persistent corpus grows.** Minimum evidence: fixed $k,c$, measured read length over increasing stores, plus disclosure of selector/store scaling. **Actual:** Eq. (1), Table 24, PDF lines 239–250 and 878–889. Read stays about 6.2–6.5k tokens through 4M stored tokens, while BM25 grows roughly linearly. **Supported; this is not a bounded end-to-end-system claim.**
- **C3 — The residual store pays off only after reuse.** Minimum evidence: same query workload and selection, measured one-time Write/index cost, per-query fetch/Read/decode, storage placement, generation length, and dispersion. **Actual:** Table 3 and Appendix A.4, PDF lines 343–355 and 977–996. **Partly supported:** point crossovers are reported, but timing uncertainty is insufficient for a strong threshold claim.
- **C4 — Equal-latency quality conclusions depend on the selector.** Minimum evidence: disjoint latency calibration; comparable TTFT boundary; same task distribution, scoring, and paired examples; variation of selector while holding other deployment choices explicit. **Actual:** Tables 4 and 8. **Supported only as a diagnostic of the specified nine-cell mixture, not as a population-level deployment result.**
- **C5 — Missing lower-layer document context is a major tested source of the chunk-local gap, and overlap repairs much of it locally.** Minimum evidence: crossed context/position interventions with fixed retrieval/adapter/upper Read, plus paired overlap tests. **Actual:** Tables 5, 6, and 31. **Supported on the 8k/16k synthetic multikey cohort only.**
- **C6 — Self-distillation makes the split interface substantially more usable.** Minimum evidence: same split and evaluation with adapter on/off. **Actual:** Tables 9, 11, and 12. **Supported for the tested tasks; training-run variance remains weakly characterized.**
- **C7 — The implementation ports beyond the flagship architecture.** Minimum evidence: exact split-forward reconstruction and at least one nontrivial quality test on another architecture. **Actual:** Hy3 exact partition test and Tables 33–34. **Supported as implementation portability, not as replicated system superiority or a scaling law.**

# Strengths

**S1. Strongest claim is unusually well isolated.** Section 5.1 states that the arms use “the same sink, mask, examples, and LoRA” (PDF lines 318–326), and Appendix A.4 further specifies shared example IDs, retrieved chunk IDs/order/tokens, decode parameters, checkpoint, and 168 mounted modules. The paper correctly labels the $1.403\times$ result as selected-pack model Read rather than end-to-end acceleration.

**S2. The paper is candid about negative results and scope.** It explicitly says CoMem is not quality preserving, loses to iterative-BM25 replay at equal latency, does not win the frozen-BGE aggregate, lacks matched PIC/modular-cache comparisons, and should be read as an internal measurement. This substantially reduces the risk of misleading systems claims.

**S3. Timing boundaries and storage costs are separated well.** Tables 2–4, 7, 24, 29–32 distinguish model Read, store-ready online prefill, Write-inclusive cost, persistent I/O, decode, and index/selector work. The 8,192 B/token cost, about 1 GiB at 128k, and failure of GPU residence at 16M tokens are clearly exposed rather than hidden behind a speedup number.

**S4. The diagnostic sequence is technically useful.** The exact continuous-prefix $h_{12}$ control (Table 31), context-by-position factorization (Table 5), and overlap intervention (Table 6) distinguish “upper layers cannot consume $h_{12}$” from “independent writes omit necessary lower-layer context.” The authors appropriately avoid turning this into a universal mechanism claim.

**S5. Reproducibility reporting is above average.** Tables 25–27 provide backbone revision, dimensions, masks, positions, selector constants, decoding, optimizer, trainable modules, token budget, hardware, seeds, scorers, sample counts, and a final adapter SHA-256. Tables 28 and 35–36 add integrity/statistical details. The code/artifact itself was not supplied in the frozen review materials, so executability remains **Unverifiable**, but the written protocol is detailed.

**S6. Figures and tables are generally honest and readable.** Figure 1 clearly separates Write/Select/Read and $j=0$/$j>0$ paths; Figure 2 is explicitly labeled motivation rather than validation. Every inspected table identifies important mismatches (different backbones, YaRN, full-prefill versus persistent reuse, cohort A/B, point estimates, or missing matched Write values).

# Weaknesses (ordered by severity)

## W1 — No matched comparison to the nearest reusable-context systems

- **Location:** Related Work, PDF lines 147–156; Limitations, lines 434–445; Conclusion, lines 416–419.
- **Exact quote (20 words):** “We lack a matched same-backbone implementation of these systems, so Table 1 supports taxonomy rather than superiority.”
- **Problem:** The paper isolates an internal $j=0\rightarrow12$ trade-off, but does not establish whether one residual/token is a useful Pareto point against the nearest alternatives: PIC/chunk-KV repair, learned modular KV, or residual-to-KV reconstruction on the same model, evidence pack, hardware, storage tier, and latency boundary.
- **Affected claim/norm:** This limits **novelty, practical significance, and excitement**, not the validity of C1. Caching intermediate representations predates this paper, and contemporaneous systems already target reusable document state. Without one matched nearest-neighbor baseline, the reader cannot tell whether the measured point advances the state of practice or merely characterizes another inferior interface.
- **Sufficient remedy:** Implement at least one closest representative (preferably EPIC/CacheBlend-style reusable KV and KV-Direct-style residual reconstruction) on Qwen3-8B, using identical selected chunks, LoRA/training budget where applicable, H20, storage tier, TTFT boundary, quality cohort, and persistent bytes. A convincing Pareto comparison could materially raise the score.
- **Severity:** **Major**.

## W2 — The equal-latency aggregate has an invalidly narrow inferential unit for its heterogeneous mixture

- **Location:** Appendix Table 8, “Quality cohort” and “Statistical unit,” PDF page 12 / lines 796–837.
- **Exact quote (14 words):** “it does not resample task cells or LoCoMo conversations.”
- **Problem:** The reported CIs pool 900 paired example differences IID across nine equally weighted task–length cells. The mixture includes synthetic and natural cells, and all 100 LoCoMo examples come from one conversation. IID resampling of examples does not reflect uncertainty over task cells or the conversation cluster, despite the paper drawing the decision-relevant “selector dependence” conclusion from this aggregate.
- **Affected claim/norm:** This weakens **C4 and statistical validity**. The point estimates remain descriptive, but the quoted 95% CIs—especially the frozen-BGE “statistically tied” conclusion—are not robust uncertainty estimates for the intended heterogeneous deployment mixture.
- **Sufficient remedy:** Report all nine cell-level deltas; add a hierarchical/stratified paired bootstrap that resamples within cells and treats cells as the aggregation unit; for LoCoMo, sample multiple conversations and cluster by conversation. Show sensitivity to cell weights. The wording should remain diagnostic unless signs are stable.
- **Severity:** **Major**.

## W3 — Training-run uncertainty does not cover the flagship headline

- **Location:** Limitations, PDF lines 423–433; Appendix Table 28 and A.5, lines 1031–1041.
- **Exact quote (8 words):** “The flagship is one batch-8 training run.”
- **Problem:** Two additional adapters change effective batch from 8 to 3 and are evaluated on reduced RULER/BABILong supports; no multi-run aggregate is retained for the exact RULER-B, LongEval, or LoCoMo headlines. Thus the 3.12-point central quality cost and natural-task results combine evaluation uncertainty with an essentially single-run learned interface.
- **Affected claim/norm:** This weakens **C1/C6 robustness** and the norm for learned-method empirical reliability. It does not invalidate the paired examples for the fixed released adapter, but it prevents estimating whether adapter training materially moves the advertised operating point.
- **Sufficient remedy:** Train at least three adapters with identical effective batch, data order/budget, and hyperparameters; evaluate all on RULER-B and the main natural-task supports; report run-level means/SDs or hierarchical intervals for quality and the resulting Pareto point.
- **Severity:** **Major**.

## W4 — Break-even thresholds lack uncertainty and workload sampling

- **Location:** Table 3 and Section 5.2, PDF lines 343–355; Appendix A.4, lines 977–996.
- **Exact quote (11 words):** “Values derive from three process medians and include store fetch, model Read,”
- **Problem:** The exact “8–11 queries” and “25.8–27.6 queries” thresholds are nonlinear ratios of measured component times, yet the paper reports only point estimates from three process medians. It does not provide a CI/sensitivity interval for $Q^\star$, document/query variation, or uncertainty when the denominator $T_{j=0}-T_{CoMem}$ is small; the 32k GPU $G=128$ point (5.5) also sits outside the abstract’s coarse 8–11 summary.
- **Affected claim/norm:** This weakens **C3 and systems statistical validity**. Deployment thresholds are among the paper’s most actionable claims, so their stability matters more than a single median.
- **Sufficient remedy:** Bootstrap or repeat complete Write/fetch/Read/decode measurements across processes and multiple documents/query packs; propagate uncertainty through $Q^\star$; report intervals and a sensitivity plot over generation length, storage tier, batch/concurrency, and pack length. Align abstract wording with the complete retained grid.
- **Severity:** **Major**.

## W5 — Contamination audit is incomplete for natural-task scope checks

- **Location:** Appendix A.3, PDF lines 890–908; Limitations, lines 487–499.
- **Exact quote (12 words):** “Equivalent overlap audits were not completed for all natural benchmarks, including NarrativeQA;”
- **Problem:** PG-19 is used to distill the adapter, and the completed audit found substantial overlap with InfiniteBench long-book support, forcing removal of that comparison. Equivalent audits were not completed for several natural datasets; therefore LongBench/LongEval/LoCoMo scope checks cannot be interpreted as clean transfer evidence.
- **Affected claim/norm:** This limits **external validity for C6 and natural-task generalization**. The paper mostly scopes this correctly, so the issue is not fatal, but natural-task numbers still contribute to perceived utility.
- **Sufficient remedy:** Run a benchmark-specific n-gram/document provenance audit for every natural dataset; remove or separately report contaminated items; evaluate a clean held-out corpus/task suite. Until then, keep these results explicitly descriptive.
- **Severity:** **Minor** because the paper already disclaims contamination-free generalization.

## W6 — Abstract exceeds the official 200-word guidance

- **Location:** Abstract, PDF lines 001–035.
- **Exact quote (12 words):** “Its primary evidence is a matched two-point depth comparison.”
- **Problem:** Mechanical TeX-stripped counting gives approximately 221 words, above the official style guidance of no more than 200 words.
- **Affected claim/norm:** **Style/compliance**, not scientific soundness. This is easily fixable and is not by itself a substantive score driver.
- **Sufficient remedy:** Remove roughly 20–25 words, preferably compressing secondary diagnostics while retaining the central trade-off and scope disclaimer.
- **Severity:** **Minor**.

# Questions That Could Change the Score

1. Can the authors provide one same-backbone/same-pack/same-hardware Pareto comparison against a nearest reusable-KV or residual-reconstruction baseline? A competitive result could move this from Findings-level to main-conference level.
2. For equal latency, what are the nine individual cell deltas under both selectors, and do the conclusions survive hierarchical cell-level resampling and conversation clustering?
3. Across clean same-batch adapter seeds, how variable are the exact RULER-B 3.12-point gap, LongEval, and LoCoMo results?
4. What are uncertainty intervals for every retained $Q^\star$ cell, and why does the abstract say 8–11 queries at 32k when Table 3 includes a 5.5-query GPU-resident $G=128$ cell?
5. Were the raw-replay and CoMem arms in the equal-latency experiment identical in adapter state and all generation settings? Table 8 should state this as explicitly as Table 30 does for the central comparison.

# Non-scoring Suggestions / Typos

- In Section 5.1, the prose cites “97.2 versus 69.0” for matched LongEval and “12.31 versus 12.15” for matched LongBench, but the referenced Tables 19 and 21 do not display those $j=0$ values. Add the matched rows or point to the exact table/artifact containing them.
- Table 14 labels the second row “Raw-text reader quality,” but values appear normalized to [0,1] while many other quality tables use percentages; state units explicitly.
- Clarify whether `Transformers 5.5.4` is a released/pinned version in the artifact; environment executability is **Unverifiable** from the frozen paper alone.
- Table 29’s “Native A macro” for SnapKV/PyramidKV includes left-truncated 64k/128k cases; a within-native-window macro would be more interpretable in the table body, not only the caption.
- Consider moving the exact equal-latency adapter configuration into Table 8.

# Citation Audit

## Procedure and result

I audited all 43 entries actually emitted by `main.bbl`, using DOI/Crossref where available, arXiv metadata for arXiv records, and direct official pages for the LMSYS blog and Hy3 model card. Network-limited searches were to be marked **Unverifiable**; none of the final 43 entry identities remained Unverifiable. Result: **40 Verified, 3 Metadata error, 0 Not found, 0 Unverifiable**. “Metadata error” below does not imply the cited work is nonexistent.

## Entry-by-entry status

1. Cache-Craft — **Verified**.
2. LongBench — **Verified**.
3. PyramidKV — **Metadata error**: arXiv author metadata includes Yucheng Li, omitted in `main.bbl`; conference-year versus preprint-year difference is otherwise plausible.
4. KV Packet — **Verified**.
5. Cartridges (base/self-study) — **Metadata error**: arXiv metadata includes Will Tennien, omitted in `main.bbl`; ICLR 2026 venue/year otherwise plausible.
6. HCache — **Verified**.
7. Llama 3 — **Verified**.
8. Cartridges at Scale — **Verified**.
9. Distillation — **Verified**.
10. RULER — **Verified**.
11. LoRA — **Verified**.
12. EPIC — **Verified**.
13. RAGCache — **Metadata error/clarification needed**: Crossref records online publication on 2025-11-07 and print volume 44 on 2026-02-28; the 2025 BBL year is defensible online-first but should be made consistent with the chosen venue record.
14. BABILong — **Verified**.
15. Retrieval-Augmented Generation — **Verified**.
16. LongChat/LongEval blog — **Verified**.
17. SnapKV — **Verified**.
18. ILRe — **Verified**.
19. ReadOnce Transformers — **Verified**.
20. MiniCache — **Verified**.
21. TurboRAG — **Verified**.
22. LoCoMo — **Verified**.
23. XC-Cache — **Verified**.
24. KV-Direct — **Verified**.
25. PG-19 / Compressive Transformers — **Verified**.
26. BM25 — **Verified**.
27. Embedding Recycling — **Verified**.
28. GemFilter — **Verified**.
29. REFORM — **Verified**.
30. LLoCO — **Verified**.
31. Tencent Hy3 — **Verified**.
32. Fusion RAG Cache — **Verified**.
33. MEPIC — **Verified**.
34. LongMem — **Verified**.
35. MemoryLLM — **Verified**.
36. InfLLM — **Verified**.
37. StreamingLLM — **Verified**.
38. SemPIC — **Verified**.
39. Retrieval Meets Long Context LLMs — **Verified**.
40. Qwen3 technical report — **Verified**.
41. APE — **Verified**.
42. CacheBlend — **Verified**.
43. H2O — **Verified**.

## Load-bearing citation–claim checks (8)

1. **Lewis et al. 2020 / Xu et al. 2024 → raw-text retrieval bounds online tokens but recomputes model layers.** **Supported at the family level.**
2. **CacheBlend/TurboRAG/Cache-Craft/RAGCache → reusable/precomputed chunk KV for RAG.** **Supported.** The papers differ in fusion/repair details, as the submission notes.
3. **EPIC/MEPIC/APE → position-independent or parallel reusable context encoding with repair/realignment.** **Supported.**
4. **ReadOnce/Embedding Recycling → reusable cached intermediate representations with adapted later layers.** **Strongly supported and directly novelty-relevant.**
5. **HCache/KV-Direct → activation or residual checkpoint/reconstruction precedents.** **Supported**, though KV-Direct is contemporaneous (2026-03-20) and should not be treated as long-established prior art.
6. **KV Packet/Cartridges → learned reusable context objects with adapters/distillation.** **Supported.** KV Packet (2026-04-14) is within the three-month contemporaneous window.
7. **RULER/BABILong/LongBench/LoCoMo → benchmark identities and official evaluation families.** **Supported.** The paper gives its own deviations and scoring details.
8. **Hinton et al./LoRA → distillation and low-rank adaptation basis.** **Supported**, but these citations justify ingredients, not the specific symmetric top-64-support objective.

# Novelty Search Summary

- **Freeze date:** 2026-08-04.
- **Prior-art cutoff:** 2026-05-04.
- **Three-month contemporaneous window:** 2026-05-05 through 2026-08-04; such work was not used to penalize novelty.
- I performed four targeted searches before the user stopped further retrieval: reusable intermediate representations; residual-stream/KV-cache equivalence; position-independent caching; and intermediate-layer retrieval/cache reuse. Semantic Scholar rate-limited later queries; those search branches are **Unverifiable** rather than treated as negative evidence.

## Closest works found

1. **ReadOnce Transformers** (arXiv 2020-10-24; ACL 2021): builds reusable, task-independent compressed text representations consumed by later computation. This is the clearest conceptual precedent for “read a document once, reuse an intermediate object.”
2. **Embedding Recycling** (arXiv 2022-07-11; EACL Findings 2023): explicitly caches activations from an intermediate layer and trains adapters on later layers for reuse. It substantially narrows the algorithmic novelty of persistent intermediate activation reuse.
3. **EPIC** (arXiv 2024-10-20; ICML 2025): modular position-independent reuse of document KV with boundary/attention repair; closest serving-workload comparator.
4. **KV-Direct** (arXiv 2026-03-20): stores residual vectors and reconstructs layer-wise KV/recomputes as needed. It is highly relevant to the stored-object choice and predates the 2026-05-04 cutoff.
5. **KV Packet** (arXiv 2026-04-14): context-independent reusable KV packets with distillation adapters and no document recomputation. This is very close in workload and learned-interface motivation, but falls inside the three-month contemporaneous-work rule and therefore should not reduce the score.

**Novelty judgment:** The broad ideas of caching intermediate representations and reusable document state are not new. The paper’s defensible novelty is the specific one-residual-per-token, tunable split-depth object; direct suffix execution on a bounded selected pack; and unusually controlled $j=0$ endpoint with explicit storage/Write/Read accounting and failure diagnosis. That is a meaningful empirical/system characterization, but not yet a main-conference-level advance without matched nearest-system evidence.

# Technical, Formula, Benchmark, and Reproducibility Audit

## Formula and boundary checks

- **Eq. (1):** $|h_j|/|KV|=d/(2Ln_{kv}d_{head})=n_q/(2Ln_{kv})$ is correct for storing one residual vector per token versus full per-layer K and V at a common dtype, assuming $d=n_qd_{head}$. For Qwen3-8B ($L=36,n_q=32,n_{kv}=8$), the ratio is $32/(2\cdot36\cdot8)=1/18$; 4096 bf16 elements are 8,192 B and full KV is 147,456 B = 144 KiB/token. At 128k, the residual store is exactly 1 GiB. Boundary caveat: the comparison excludes allocator/layout/metadata overhead and assumes the alternative retains full KV for all layers.
- **Read length:** `sink + kc + query`; the nominal 6,657 uses sink 1, $k=12$, $c=512$, and maximum query 512. Actual shorter tails/queries explain 6.2–6.5k means. The selector, index, and store are correctly stated as unbounded.
- **Eq. (2):** the weighted bidirectional KL on teacher top-64 support is mathematically defined. Important boundary: it is not full-vocabulary distillation because both distributions are renormalized on teacher-selected support and retained teacher mass was not logged; the paper discloses this.
- **$j$ boundaries:** $j=0$ is raw-ID full replay; the split-forward self-test also checks $j=L$ on Hy3. For deployment, the paper appropriately avoids claiming the separately trained $j=6/9/12/18$ curve is a compute-matched causal frontier.
- **Overlap-Write:** persistent bytes and online Read are unchanged because only the target chunk residuals are retained, while Write work/edit invalidation increase. This claim is technically consistent.
- **Break-even equation:** algebraically appropriate for cumulative Write-once versus replay, provided component times share a boundary and the denominator is positive. Statistical uncertainty is the main deficiency (W4).

## Baselines, benchmarks, metrics, and statistics

- The central baseline is excellent for the causal depth claim. External baseline tables are descriptive rather than matched, which the paper repeatedly acknowledges.
- RULER cohort A/B are carefully distinguished; direct differences use cohort B. Official scorers and sample counts are documented.
- LongEval uses different generation limits for flagship and frozen controls, though both exceed the expected numeric answer length; this is probably benign but should remain explicit.
- LoCoMo’s full-set judge combines GPT-4o semantic scoring for categories 1–4 with local abstention scoring for category 5. The paper reports denominators, date, parsing, failures, a conversation-cluster bootstrap, and a 200-item second-judge audit. Exact judge reproducibility is still **Unverifiable** because the endpoint has no dated snapshot.
- The central paired RULER CI and McNemar test are appropriate for fixed-adapter paired examples. The equal-latency aggregate’s IID bootstrap is not appropriate for task/conversation generalization (W2).
- Timing commonly uses three process medians; this demonstrates repeatability for the fixed Read microbenchmark (ratios 1.402–1.404) but is insufficient for broad performance distributions, concurrent p95, or crossover uncertainty.

## Compute and reproducibility

- Final training: one node, 8 H20s, 4,000 steps, about 65.5M tokens, approximately 22 minutes / 2.9 H20 GPU-hours. Total exploratory compute and peak training memory were not logged.
- The paper reports exact adapter hash, backbone revision prefix, environment versions, scorer choices, and release contents. Because only frozen PDF/source were permitted, actual archive completeness, dependency solvability, hashes, and rerun success are **Unverifiable**.

# Figure and Table Inspection

- **Figure 1:** accurate high-level pipeline and timing-scope warning; white `j` is a visible label inside a dark marker, not hidden reviewer text.
- **Figure 2:** readable but small; properly framed as motivational probe evidence.
- **Tables 1–6:** central taxonomy, matched trade-off, crossover, equal latency, context-position, and overlap diagnosis; all inspected. Tables 3–4 need the statistical qualifications in W2/W4.
- **Tables 7–17:** online prefill, full equal-latency protocol, depth/adapter/selector/context/chunk ablations; all inspected. Table 8 is especially valuable but exposes the clustered-statistics issue.
- **Tables 18–24:** full benchmark and store-scaling grids; all inspected. External rows use different backbones/protocols and are correctly marked descriptive.
- **Tables 25–32:** configuration, training, evaluation, seed/batch, KV compression, replay latency/oracle/store I/O; all inspected. These provide most reproducibility evidence.
- **Tables 33–36:** Hy3 distillation/RULER and statistical verification; all inspected. Hy3 uses $n=16$ PG-19 windows for the LM diagnostic and $n=50$ RULER cells, so it supports portability only.
- No clipped tables, unresolved references, or illegible main claims were found. Several appendix tables use dense scriptsize formatting, but remain readable when zoomed.

# Limitations, Ethics, and Desk-Reject Risks

- **Page limit:** The main content ends on page 8; references occupy pages 8–10; appendices begin page 11. This conforms to the long-paper 8-page content limit. A short-paper interpretation would fail, but the manuscript is clearly formatted as a long paper.
- **Limitations:** Exact unnumbered `Limitations` section is present on page 7 and is unusually substantive.
- **Anonymity:** “Anonymous ACL submission,” no acknowledgments, no author affiliations/emails/self-identifying URLs, and no identifying PDF metadata were found. Figure metadata contains tool/creation information but no identity.
- **Official style:** Frozen `acl.sty` and `acl_natbib.bst` are byte-identical to the official ACL style repository commit inspected during review; A4, two columns, line numbers, page numbers, and embedded fonts are present.
- **Style issue:** abstract is approximately 221 words versus the official 200-word guidance (W6).
- **Unresolved references/TODO/placeholders:** Mechanical searches found no `??`, unresolved citation/reference markers, TODO/TBD/FIXME/XXX, or draft placeholders.
- **Hidden manipulation/prompt injection:** Source and rendered-PDF text/color/size scans found no reviewer instructions, score manipulation, white/tiny hidden prose, or prompt injection. The only white text detected is the figure’s single-character `j` label rendered on a dark marker.
- **Ethics:** The paper discusses inherited generation risks, residual inversion/membership risk, authorization/isolation/deletion, sensitive-source handling, energy, licenses, and data scope. No new human-subject data or annotators are used. I see no ethics-based rejection issue.
- **Desk-reject assessment:** No clear desk-reject condition. The abstract length is a correctable style nonconformance, not a scientific desk-reject recommendation from this reviewer.

# Scores

## Soundness: 3.0 / 5.0

The central matched depth result is sound and carefully scoped. The score is held at 3.0 by the heterogeneous equal-latency inference, single-run learned-interface uncertainty, and imprecise crossover thresholds. These issues affect important secondary and deployment claims but do not invalidate the primary fixed-adapter trade-off.

## Excitement: 3.0 / 5.0

The measurement and diagnosis are useful, transparent, and likely valuable to researchers designing reusable context objects. However, ReadOnce, Embedding Recycling, PIC systems, and residual/KV work substantially narrow the conceptual novelty, and the paper does not show a matched Pareto gain over the nearest systems.

## Overall: 3.0 / 5.0

**Findings level.** The work is substantially better than a weak exploratory paper: it has a real controlled result, negative findings, careful boundaries, and strong documentation. It is not yet ACL-main level because the practical contribution is unresolved without a matched nearest-system baseline, and several deployment/statistical claims need stronger experimental support. Per the required calibration, I choose the lower bin rather than 3.5 because W1–W4 are claim-linked and verifiable.

## Confidence: 4.5 / 5.0

I inspected the complete frozen PDF/source twice, all appendices, every figure/table, formula boundaries, all emitted bibliography entries, and the relevant style/compliance properties. Some artifact-runtime and rate-limited search details remain Unverifiable.

## Reproducibility: 3.5 / 5.0

The paper provides unusually detailed configurations, hashes, scorers, sample counts, and timing boundaries. Reproducibility is reduced by a mutable judge endpoint, absent full artifact execution in the allowed materials, incomplete total compute logging, and lack of clean same-batch multi-run evidence for headline results.

# Review-Process Self-Check

- [x] Read only the frozen v5 PDF/source (`v5_20260804_003238`) and strict template; did not consult other reviews/history reports/TODO/status/current/calibration.
- [x] Completed two full passes including Appendices A and B.
- [x] Built claims C1–C7 and mapped minimum sufficient experiments to actual evidence.
- [x] Checked long/short page limits, exact Limitations, anonymity, official style, abstract length, unresolved references, TODO/placeholders, hidden text/manipulation, and at least five abstract numbers.
- [x] Verified abstract numbers against Tables 2–6/30–31: 931.9, 664.4, 1.403×, 99.19, 96.07, 3.12 and CI, 8–11/25.8–27.6, 11.56, −1.00 and CI, 92.5/100.0/98.5.
- [x] Audited every `main.bbl` item and checked eight load-bearing citation–claim matches.
- [x] Performed four cutoff-constrained novelty searches and applied the 2026-05-04 cutoff / contemporaneous-work rule.
- [x] Audited formulas, boundary cases, baselines, benchmark validity, metrics, seeds/statistics, compute, and reproducibility.
- [x] Inspected both figures and all 36 numbered tables in the rendered PDF.
- [x] Mechanically searched all weakness quotes and “lacks X” assertions in frozen source/PDF; every retained weakness has a location, ≤25-word quote, explicit problem, affected claim/norm, remedy, and severity.
- [x] Did not deduct for any issue merely because it may have existed in an earlier version.
