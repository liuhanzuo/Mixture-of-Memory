---
review_mode: normal
soundness: 3.5
excitement: 3.5
overall: 3.5
confidence: 4.0
reproducibility: 3.5
---

# Paper A v6 — Normal Reviewer #2

## Summary

The paper studies a repeated-query, stable-corpus serving regime and proposes **CoMem**, which writes one intermediate residual per document token at split depth $j$, retrieves a bounded set of chunks, and resumes only layers $[j{:}L)$. Its strongest experiment is intentionally narrow: on Qwen3-8B, a same-pack/same-adapter $j=0\rightarrow12$ comparison reduces isolated model Read from 931.9 to 664.4 ms (1.403×) while reducing a paired 15-cell RULER macro from 99.19 to 96.07 (gap 3.12, 95% CI [2.36, 3.93]). The paper also measures write amortization, gives a negative/selective equal-latency result, and diagnoses missing lower-layer document context using continuous-prefix, context×position, and overlap-Write controls.

My assessment is **borderline ACL-main / solid Findings**. The central internal measurement is carefully scoped, unusually transparent about negative results, and mostly sound. The main reason I stop at 3.5 rather than 4 is external validity: the paper does not implement the nearest reusable-context systems under the same backbone/hardware/timing boundary, and its proposed local repair is validated only on a narrow synthetic cohort rather than on the natural-task suite or a repaired end-to-end frontier.

## Claim–evidence map

| ID | Claim | Primary evidence | Audit judgment |
|---|---|---|---|
| C1 | Depth reuse yields a measurable Read/quality trade-off under identical evidence. | Abstract lines 4–7; §5.1 lines 316–329; Table 2; Appendix Tables 31–32. | **Supported.** Arithmetic checks: 931.9/664.4 = 1.4026 and 99.19−96.07 = 3.12. Same chunks/order/sink/mask/examples/LoRA are stated. |
| C2 | The residual store can amortize for repeated queries, but not for one-off use. | Abstract lines 14–17; §5.2 lines 345–356; Table 3. | **Supported within retained cells.** 32k has a generation grid; 128k retains only G=1, which the paper explicitly limits. |
| C3 | Saved depth does not automatically convert to better equal-latency quality. | §5.2 lines 357–376; Tables 4 and 8; Appendix B.4. | **Supported as a negative, mixture-specific diagnostic.** BM25 Δ=−11.56 is stable; BGE Δ=−1.00 is unresolved. |
| C4 | Missing lower-layer document context is a major tested source of multikey loss, and overlap can repair much of that local gap. | §5.3 lines 377–392; Tables 5–6; Appendix Table 32. | **Supported only for the displayed paired synthetic cohort.** The paper correctly avoids a universal attribution. |
| C5 | Model-side Read is bounded in stored-context length for fixed $k,c$, while index/store costs are not. | §4 lines 237–249; Eq. 1; Appendix Table 25. | **Supported and correctly scoped.** |
| C6 | The interface is portable across scales/MoE backbones. | §5 lines 395–403; Appendix A.7 and Tables 35–36. | **Suggestive, not a matched replication.** The paper labels it secondary portability evidence. |
| C7 | CoMem is superior to raw-text RAG or modular caches. | Abstract lines 28–31; Related Work lines 185–195; Limitations lines 441–452. | **Not claimed.** The manuscript repeatedly restricts itself to an internal operating-point measurement. |

## Strengths

1. **Cleanest evidence is genuinely matched.** Table 2 states that raw replay and CoMem share the selected evidence, order, sink, mask, examples, and LoRA. The continuous-prefix oracle then exactly recovers the full-replay result, making the upper-layer continuation claim credible while also showing that deployable independent writes are the fidelity bottleneck.

2. **Systems boundaries are unusually explicit.** The manuscript distinguishes selected-pack Read, store-ready prefill, write-inclusive pipeline, and external I/O/index cost. It does not pass off the 64.9× dense-to-bounded number as the 1.403× depth effect, and it reports the large 8 KiB/token storage burden and query-count crossover.

3. **The equal-latency result is a useful negative result rather than a hidden failure.** The new hierarchical and leave-one-cell-out analyses improve the audit materially. My independent reading agrees with the paper's restrained conclusion: for BM25, all nine cell effects are negative and LOCO remains [−13.13, −9.50], so replay's advantage is robust for this defined mixture; for BGE, the hierarchical interval [−10.67, 8.33] and LOCO [−3.50, 1.75] cross zero, so the aggregate is unresolved, not evidence of a CoMem win. This is better phrased as **selector/task-mixture dependence** than as a general method ranking.

4. **The paper contains relevant mechanism controls.** The same-$j$ LoRA on/off controls, multi-depth deployment curve, cross-chunk attention ablation, context×position table, continuous-prefix ceiling, and overlap widths cover several plausible failure modes.

5. **Scope and limitations are candid.** The manuscript explicitly acknowledges the missing nearest-system comparison, non-clean seed study, mutable GPT-4o judge, incomplete contamination audit, single-query median timing, native-window stress status, storage/edit invalidation costs, and narrow repair evaluation.

6. **Presentation is strong.** Figure 1 clearly separates Write/Select/Read and distinguishes the matched $j=0$ endpoint from residual continuation. Tables generally name cohorts, sample counts, timing exclusions, and whether comparisons are descriptive or causal.

## Major weaknesses

### W1. The central result is still an internal endpoint, not a competitive reusable-context evaluation

- **Location:** Related Work, lines 185–195; Limitations, lines 441–452; Conclusion, lines 424–429.
- **Short quote:** “**We lack a matched same-backbone implementation of these systems**” and “**pending matched comparison with the nearest reusable-context systems**.”
- **Severity:** **Major.** This is the main obstacle to an ACL-main score. The novelty/value question is not only whether $j=12$ beats the paper's own $j=0$ endpoint, but whether one residual/token offers a favorable quality–latency–storage frontier versus PIC/chunk-KV repair/learned modular-KV systems under the target repeated-query workload. Table 1 is taxonomic; descriptive external rows use different backbones, prompts, training, and timing boundaries.
- **Mechanical verification / requested evidence:** Implement at least one closest feasible system (e.g., PIC/EPIC-style independent KV, CacheBlend/Cache-Craft-style repaired chunk KV, or a learned modular-KV method) on the same Qwen3-8B, same selected chunk IDs, same H20, same store tier, and identical TTFT/Write/fetch/decode boundaries. Report quality, bytes/token, write time, read time, and crossover. If implementation is impossible, provide a parameterized accounting table that converts each published object into this paper's exact token count/dtype/store tier, but label it non-empirical.

### W2. The proposed fidelity repair is not validated where the paper's deployment claims matter

- **Location:** §5.3 lines 387–393; Table 6; Limitations lines 475–483.
- **Short quote:** “**evaluated only on this paired synthetic cohort**” and “**its wall-clock Write overhead is not integrated into a repaired end-to-end frontier**.”
- **Severity:** **Major.** The paper identifies overlap-Write as the actionable engineering repair, but the evidence is only RULER `niah_multikey_1` at 8k/16k (n=200 pooled). The natural-task results still use the unrepaired interface, and the paper itself shows a large LongEval gap (97.2 vs 69.0). Thus the diagnosis is informative, but it does not yet establish that the proposed repair changes practical utility.
- **Mechanical verification / requested evidence:** Rerun $w\in\{0,32,128\}$ on at least LongEval, BABILong qa1/qa2/qa5, and LoCoMo, using identical retrieval and seeds. Add measured Write wall time, invalidation radius, total bytes written, read/decode time, quality CIs, and repeated-query crossover. A minimal decisive experiment is one natural retrieval task plus one multi-hop task with the repaired end-to-end frontier.

### W3. Optimization uncertainty is not cleanly separated from seed effects, and the flagship conclusion relies on one canonical adapter

- **Location:** Limitations lines 431–440; Appendix A.5 / Table 29.
- **Short quote:** “**not a clean estimate of training-seed variance**.”
- **Severity:** **Moderate-to-major.** The flagship is one effective-batch-8 run; seeds 1 and 2 use effective batch 3 and reduced evaluation support. This is useful robustness evidence, but it conflates initialization, minibatch order, and optimizer-noise scale, and it does not cover the main 15-cell RULER-B, LongEval, or LoCoMo aggregate. Since the adapter is necessary for usability, uncertainty in adapter training directly affects the claimed operating point.
- **Mechanical verification / requested evidence:** Train at least three matched effective-batch-8 runs with the same token order/data budget and evaluate all on the full headline supports. Report run-level mean/SD or a hierarchical interval for RULER-B, LongEval, LoCoMo, Read latency, and (for repaired variants) Write time. At minimum, rerun one additional batch-8 seed on RULER-B and LongEval.

## Minor weaknesses

### W4. The equal-latency mixture remains partly arbitrary and has one non-identifiable LoCoMo cluster

- **Location:** Table 8; Limitations lines 487–499.
- **Short quote:** “**combines synthetic and natural diagnostics with equal cell weights**” and “**conversation-level clustering is impossible for this slice**.”
- **Severity:** **Minor to moderate.** The hierarchical/LOCO analysis is the correct improvement, but it quantifies sensitivity over the nine selected cells, not uncertainty over a well-defined deployment population. Equal cell weighting mixes heterogeneous metrics/tasks, and the first-100 LoCoMo slice is all conversation 0.
- **Mechanical verification / requested evidence:** Add a predeclared family-stratified summary (synthetic retrieval, line retrieval, conversational memory) and use a LoCoMo slice spanning conversations or all 1,986 items with conversation-level resampling. Report whether the BM25/BGE conclusions survive these changes.

### W5. Several natural-task generalization claims remain contamination-uncertain

- **Location:** §5.1 lines 337–342; Appendix A.3 lines 905–924; Limitations lines 506–512.
- **Short quote:** “**We did not complete equivalent overlap audits for every natural benchmark, including NarrativeQA**.”
- **Severity:** **Minor to moderate.** The paper responsibly downgrades these to scope checks, so this does not invalidate C1–C4. It does weaken claims about cross-task transfer of the PG-19-trained adapter and makes the natural-task table less informative than its size suggests.
- **Mechanical verification / requested evidence:** Apply the same normalized n-gram audit to all natural datasets, publish per-dataset flagged fractions, and rescore a clean subset where licenses permit. Otherwise keep all affected rows explicitly labeled “contamination-unknown scope check.”

### W6. The distillation target omits an important diagnostic

- **Location:** §4 lines 262–285; Appendix Table 27.
- **Short quote:** “**We did not retain the teacher mass captured by $S_t$**.”
- **Severity:** **Minor.** Renormalizing only on teacher top-64 can hide how much probability mass is discarded and complicates interpretation/reproduction of why the adapter works.
- **Mechanical verification / requested evidence:** Log top-64 teacher mass statistics (mean, quantiles, task-independent validation split) and compare top-64 KL against full-vocabulary KL or top-128 on a small matched training/evaluation budget.

### W7. Reproducibility is paper-documented but not fully self-contained in the frozen submission source

- **Location:** Appendix A.4 lines 925–978; Appendix B lines 1144–1248.
- **Short quote:** “**configuration files and evaluation scripts accompany the code release**.”
- **Severity:** **Minor.** The PDF gives unusually rich settings, hashes, seeds, masks, and scorer definitions, but exact rerunning still depends on an external anonymous artifact and substantial H20 resources; the mutable GPT-4o endpoint prevents exact judge reproduction.
- **Mechanical verification / requested evidence:** Ensure the ARR artifact includes executable configs/scripts, environment lock, score-only exports and manifests promised in Table 8, adapter hash, and a CPU smoke test. Include saved parsed judge decisions and evaluation date (the PDF states July 22, 2026) as the canonical reproducible record.

## Questions for the authors

1. Which nearest reusable-context baseline can you realistically implement on Qwen3-8B before final submission, and what prevents a same-pack comparison now?
2. Does overlap-Write improve LongEval or LoCoMo, or is the multikey repair specific to lexical synthetic evidence?
3. For the $j=0$ matched endpoint, why apply the same upper-layer LoRA rather than compare both “same adapter” and each method's best deployment adapter? Please keep the former as the causal control, but add the latter as a practical frontier if available.
4. How sensitive is $Q^\star$ to concurrent requests, batch size, p95 latency, and cache contention? The present medians are useful but single-query.
5. Can the equal-latency LoCoMo cell be reconstructed across multiple conversations without rerunning all nine cells?
6. What is the empirical distribution of teacher probability mass retained by top-64 during distillation?

## Suggestions

- Make the paper's two contributions visually explicit in one summary table: **(i) causal internal depth effect** and **(ii) deployable but unmatched operating points**. This would prevent readers from mixing Table 2, the 64.9× dense comparison, and the write-inclusive cohorts.
- Prioritize one same-backbone nearest-baseline implementation and one natural-task overlap experiment over additional scale/MoE appendix breadth.
- Report a repaired Pareto/crossover plot with axes quality, per-query TTFT, one-time Write, and persistent bytes.
- Keep the hierarchical and LOCO equal-latency analysis; demote pooled-IID intervals further because they are not the task-mixture estimand.
- Add a compact “claim boundary” box stating: fixed top-$k$ bounds model Read only; selector/index/store remain corpus-scale; the current method does not win either equal-latency aggregate.

## Citation and bibliography audit

### Completeness and consistency

- The compiled frozen paper contains **43 bibliography entries**, all cited; no cited key is unresolved, and no compiled entry is uncited.
- I mechanically checked **58 unique labels** and **61 references** in compiled source inputs: no duplicate labels and no missing targets.
- Build log: no undefined citations/references, no LaTeX errors, and no overfull boxes. PDF is 24 A4 pages, with main text ending on page 8, references on pages 8–10, and appendix starting on page 11.
- Full external bibliographic verification was not completed because network checking was stopped; those fields are marked **Unverifiable** below rather than inferred.

### Citation–claim match sample

| Citation(s) | Manuscript claim | Match judgment |
|---|---|---|
| RAG; Retrieval Meets Long Context | Raw-text retrieval bounds online evidence but reruns the reader. | **Match.** This is the intended retrieval-vs-model-compute distinction. |
| CacheBlend, TurboRAG, RAGCache, Cache-Craft | Retrieved chunk KV can be reused/repaired for repeated RAG serving. | **Match at family level.** Exact method-specific repair details: **Unverifiable externally**. |
| EPIC, MEPIC, APE | Position-independent/parallel context encoding composes independently encoded context with repair/realignment. | **Plausible match; external details Unverifiable.** |
| KV Packet, Cartridges, SemPIC | Learned modular per-document KV objects are nearest learned reusable-context systems. | **Plausible and appropriately treated as nearest objects; external details/cutoff Unverifiable.** |
| HCache, KV-Direct | Intermediate activation/residual replay precedents. | **Match at the stated interface level; external details Unverifiable.** |
| ReadOnce Transformers, Embedding Recycling | Intermediate text representations can be cached and later layers adapted. | **Plausible match; external details Unverifiable.** |
| RULER, BABILong, LongBench, LoCoMo, LongEval/LongChat | Benchmark identity and scoring context. | **Internally consistent with the paper's evaluation tables; official-version details Unverifiable externally.** |
| Hinton et al.; LoRA | Bidirectional-KL distillation plus low-rank adaptation motivation. | **Match as generic methodological precedent.** |

### Bibliographic issues found

- No internal citation-key or reference-resolution defect was found in the compiled v6.
- The bibliography contains future/concurrent 2026 work central to novelty positioning. Venue, date, DOI/arXiv identity, and first-public dates are **Unverifiable** without network checks.
- The Hy3 entry is a model-page citation rather than an archival paper; adequate for identifying the tested checkpoint, but external metadata is **Unverifiable**.

## Novelty analysis and cutoff

The defensible novelty is narrow: **one persistent depth-$j$ residual per token, bounded query-time selection, and direct suffix execution, studied through a matched $j=0$ endpoint with explicit Write/Read/storage accounting**. The broad ideas of reusable document state, intermediate representation reuse, activation replay, PIC, and modular KV caching are prior art acknowledged by the paper.

Closest internally identified families are:

1. **ReadOnce / Embedding Recycling:** reuse intermediate text representations and adapt later computation; closest conceptual representation precedent.
2. **HCache / KV-Direct:** checkpoint or reconstruct from intermediate/residual states; closest replay-object precedent.
3. **CacheBlend / TurboRAG / Cache-Craft:** repeated-query retrieved-chunk cache reuse with context repair; closest serving workload.
4. **EPIC / MEPIC / APE:** independent or parallel context encoding with position/boundary repair; closest compositional-cache family.
5. **KV Packet / Cartridges / SemPIC / Cartridges at Scale:** learned modular reusable KV objects; closest learned objects and strongest threat to a broad novelty claim.

The paper appropriately does **not** claim invention of reusable caches. I find the residual-object/depth-frontier measurement sufficiently distinct for a contribution, but novelty is **moderate**, not high, until compared empirically with at least one nearest family.

**Three-month rule / novelty cutoff:** The frozen PDF was created on **August 4, 2026**. The paper cites concurrent items through July 2026 (including SemPIC arXiv:2607.28069 in its bibliography), but network searches and first-public-date verification were not completed. Therefore the exact three-month cutoff status of 2026 nearest works is **Unverifiable**. My score does not penalize authors for unverifiable post-cutoff work; it relies only on the paper's own conservative positioning.

## Desk, formatting, anonymity, ethics, and figure/table audit

- **Anonymity:** Pass. Title page says “Anonymous ACL submission”; no author affiliation or obvious identity leakage was found in compiled text/source. Artifact paths/hashes are generic.
- **Style/page limit:** Uses `\usepackage[review]{acl}`, A4 ACL layout, line/page numbers. Main paper occupies pages 1–8; references begin on page 8; appendix begins page 11. Pass under the manuscript's apparent eight-page main-text format; current venue policy itself is **Unverifiable** without network.
- **Limitations/Ethics:** Present and substantive. Ethics discusses residual inversion/membership risk, authorization/deletion, sensitive retrieval, and energy. No new human-subject collection is claimed.
- **Unresolved references/placeholders:** None found. No reviewer-directed or hidden/manipulative text found; `scriptsize`/resize usage is confined to dense tables rather than concealed prose.
- **Abstract-number consistency:** Central values agree with Tables 2–4/6/8/31–32 and arithmetic checks. The abstract correctly says CoMem wins neither equal-latency aggregate.
- **Figures:** Figure 1 is informative and readable at normal zoom. Figure 2 is small but legible and explicitly labeled motivational, not causal. No apparent data/axis contradiction found.
- **Tables:** 36+ tables/controls are exhaustive but cognitively heavy. Cohort A/B superscripts and timing exclusions are mostly well labeled. Tables 4 and 8 correctly distinguish hierarchical, fixed-cell, LOCO, and pooled-IID analyses. Several appendix tables require high zoom due to `scriptsize`; this is a readability, not integrity, issue.

## Reproducibility assessment

**Score: 3.5/5.** Strong paper-level documentation: checkpoint revision/hash pointer, adapter SHA-256, layer/head dimensions, dtype/kernel, exact pack/mask/position behavior, BM25 parameters, distillation formula, optimizer/schedule, sample counts, generation budgets, official scorers, seeds, hardware, timing repetitions, uncertainty units, and score-integrity statements are all provided. Weaknesses are the reliance on an external anonymous artifact, unavailable total experimental compute, non-clean multi-seed setup, mutable GPT-4o judge, incomplete contamination audit, missing $j=12$ Write datum in one depth curve, and lack of a matched nearest-system implementation.

## Overall recommendation

**Overall 3.5/5 (borderline ACL main; comfortably Findings).** Soundness is above average because the central matched endpoint, negative equal-latency result, and bounded mechanism diagnosis are transparently and statistically reported. Excitement is moderate: depth as a reuse axis and the explicit systems accounting are useful, but the contribution remains an internal measurement rather than a demonstrated win against nearest reusable-context systems. A same-backbone nearest baseline plus natural-task overlap-Write validation would move me toward 4.0.

## Review-process self-check

- Reviewed only the frozen v6 PDF (`SHA-256 f329909825b68ebaf54f951c94e3980a7797310ab189422c90bab6ed65233d20`), its actual compiled source inputs identified by `main.fls`, and the NORMAL template.
- Did not use other reviews, review history contents, TODO/status/current/calibration files, or project reports.
- Completed two passes over all 24 PDF pages, including the full appendix and bibliography; inspected all figures and the complete table set.
- Mechanically checked central arithmetic, abstract/table consistency, citation/label resolution, build warnings, anonymity/manipulation patterns, and exact weakness quotes.
- Network novelty and bibliographic verification was not completed; all such items are explicitly marked **Unverifiable**.
