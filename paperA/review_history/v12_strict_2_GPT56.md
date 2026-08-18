review_mode: strict
soundness: 3.0
excitement: 3.5
overall: 3.0
confidence: 4.5
reproducibility: 2.0

# Paper Summary

This paper proposes **CoMem**, a repeated-query long-context interface that makes transformer split depth an explicit systems variable. An offline **Write** runs each document chunk through layers `[0:j)` and stores one intermediate residual per token; **Select** retrieves a bounded chunk set; **Read** packs those residuals with the query and executes only `[j:L)`. The main scientific control is an identical-evidence `j=0` raw-text replay endpoint. On Qwen3-8B, the paper reports a `1.403×` selected-pack Read speedup (931.9 to 664.4 ms) at a 3.12-point RULER-B loss (99.19 to 96.07). A continuous-prefix oracle attributes this loss to independently written lower-layer states, and a 32-token Write overlap recovers 6.0 of a 7.5-point synthetic multikey gap. The paper also reports storage, amortization, equal-latency selector sensitivity, a same-Qwen3 CacheBlend-style implementation, cross-benchmark transfer, and exploratory cross-scale/MoE results.

## Claims and minimum-sufficient evidence map

- **C1 — CoMem exposes a measurable depth-reuse axis.** Minimum sufficient evidence: identical examples, retrieved pack/order, mask, sink, adapter, and hardware, changing only replay start. **Actual evidence:** Table 2 / Appendix Table 34, 1,500 paired RULER-B examples and three latency processes; this is sufficient for the narrow selected-pack claim.
- **C2 — The deployed `j=12` loss comes from the Write interface rather than insufficient upper-decoder capacity.** Minimum: recompute compatible `h12` jointly and show output equivalence. **Actual:** continuous-prefix oracle is bit-identical to full replay on all 1,500 examples (Table 35); sufficient as an attribution upper bound.
- **C3 — Missing lower-layer document context is the dominant tested multikey factor.** Minimum: factorial intervention on Write context and positions with retrieval/reader fixed. **Actual:** 2×2 context–position table plus overlap sweep on 200 paired synthetic examples (Tables 6–7); sufficient only for that displayed cohort.
- **C4 — CoMem yields useful deployment trade-offs after amortization.** Minimum: end-to-end, same-platform, Write/fetch/Read/decode accounting across reuse counts and generation lengths. **Actual:** a 24-cell break-even grid with raw timing records (Table 4), plus separate prefill/pipeline cohorts. Evidence is informative but limited to single-request medians and the measured hardware/software configurations.
- **C5 — The same-backbone CacheBlend-style result shows a favorable storage/quality trade-off.** Minimum: comparable optimization/training budget and a faithful native baseline or a claim explicitly limited to the implemented arm. **Actual:** a validated minimal training-free implementation versus distilled CoMem. It supports only the implementation-level comparison, not a general ranking against CacheBlend/PIC.
- **C6 — The approach transfers beyond one synthetic benchmark/model.** Minimum: matched natural-task and second-backbone/architecture replications. **Actual:** several natural benchmarks on Qwen3-8B plus exploratory scale/MoE ports, but no matched replicated frontier; transfer evidence is mixed and mostly point-estimate based.

# Strengths

**S1. The central depth-only control is unusually clean.** Section 5.1, Table 2, PDF pp. 6–7 (lines 411–425): the paper holds the evidence pack, order, sink, mask, examples, and LoRA fixed while changing only `j=0` versus `j=12`. This directly supports the narrow result of `1.403×` faster selected-pack Read for a 3.12-point RULER cost, and the paper correctly avoids calling it quality-preserving or end-to-end acceleration.

**S2. The paper diagnoses rather than conceals the quality loss.** Section 5.4, Tables 6–7, PDF p. 8 (lines 534–555), and Appendix Table 35, PDF p. 25: the continuous-prefix oracle, context–position factorization, and overlap sweep form a coherent causal chain. The oracle establishes attainable recovery, the factorial control separates tested context and position effects, and overlap provides a deployable local intervention.

**S3. Systems boundaries are explicitly separated.** Section 5.5, PDF pp. 8–9 (lines 495–560), distinguishes selected-pack Read, store-ready prefill, Write-inclusive pipeline, and persistent-store I/O. This prevents the `1.403×`, `64.9×`, and `2.74×` numbers from being silently treated as interchangeable. Mechanical recalculation reproduced `931.9/664.4=1.4026`, `71.37/1.10=64.88`, and `6.035/2.202=2.7407`.

**S4. Negative and boundary results are reported.** Section 5.3, Table 5, PDF p. 8 (lines 480–509): BM25 raw replay beats CoMem by 11.56 points at equal latency, while the frozen-BGE difference spans zero under hierarchical cell resampling. This materially sharpens the contribution: the method offers a systems axis, not a selector-independent quality win.

**S5. The paper is candid about scope and uncertainty.** The exact unnumbered Limitations section, PDF p. 9 (lines 582–656), acknowledges the single flagship run, mixed effective batch, non-native CacheBlend implementation, synthetic-only overlap validation, mutable judge, contamination uncertainty, lack of throughput/tail measurements, and separate timing cohorts. This is better calibrated than many systems papers.

**S6. The included artifact subsets that are present are internally consistent.** I verified both SHA-256 manifests. `verify_table3.py` reproduces every break-even entry and reports a complete 24-cell grid. The CacheBlend aggregate reports no missing required cells, and its real-model self-test is documented. Labels/references are complete (60 labels, 80 references, no missing or duplicate labels), and all 46 cited keys appear in `main.bbl` with no uncited bibliography entries.

# Weaknesses

## W1. Reproducibility of the central method and headline quality claims is not supported by the frozen review object

- **Location:** Appendix A.5 “Reproducibility details,” PDF pp. 21–23, especially source `sections/08_appendix.tex` lines 145–276; frozen artifact inventory.
- **Exact quote (14 words):** “Subject to upstream terms, the anonymous archive includes the adapter, source, documentation, pinned requirements”
- **Problem:** The supplied frozen object contains manuscript source plus two narrow artifact subsets (`cacheblend_143` and `p1_8_serving`). It does **not** contain the CoMem implementation, flagship adapter, pinned environment/requirements, benchmark runners, main prediction shards/hashes, judge prompt/decisions, equal-latency exports, overlap experiment records, or scripts needed to regenerate Tables 2, 5–7, 13–40. The included CacheBlend aggregator itself imports absent project modules, so even that aggregate cannot be regenerated from raw predictions here. This is a direct mismatch between the paper’s reproducibility description and the reviewable frozen artifacts.
- **Affected claim/norm and why it matters:** This affects C1–C6 and ARR reproducibility norms. Most empirical claims cannot be independently rerun or even recomputed from item-level outputs; only the serving crossover and limited CacheBlend summaries are auditable.
- **Sufficient remedy:** Attach the promised anonymous archive: executable CoMem code, exact adapter weights/hash, lockfile or container, all non-default configs, permitted item-level outputs or complete score-only exports, judge template/parsed decisions, statistical scripts, and raw timing records for every headline table, with one command per table.
- **Severity:** **Major**.

## W2. The training evidence is insufficient to establish that the proposed interface is robust rather than dependent on one favorable adapter run

- **Location:** Limitations, PDF p. 9 (lines 583–590); Appendix A.7/Table 31, PDF p. 23; source `sections/07_limitations.tex` lines 3–7 and `sections/08_appendix.tex` lines 224–249.
- **Exact quote (14 words):** “The flagship adapter is one batch-8 run; two matched-data runs use effective batch 3”
- **Problem:** The principal RULER-B, LoCoMo, LongEval, latency-quality, and mechanism headlines all use one flagship adapter. The two additional runs change effective batch and are evaluated on reduced, different cells; they therefore do not estimate seed variance for the reported 15-cell RULER-B, LoCoMo, LongEval, or natural-task claims. The paper correctly labels this, but the central method includes learned self-distillation, so robustness of that learned interface is part of soundness, not merely presentation.
- **Affected claim/norm and why it matters:** C1’s quality cost, C3’s repair magnitude, and C6’s transfer may shift with adapter training. A single run can support an existence result, but not stable quantitative conclusions at ACL-main strength.
- **Sufficient remedy:** Train at least three matched runs with identical effective batch/data order budget for the flagship `j=12` configuration; report per-run Table 2 RULER-B, LoCoMo, LongEval, and the 8k/16k multikey overlap result, with run-level intervals or ranges.
- **Severity:** **Major**.

## W3. The strongest baseline comparison is confounded by training and does not justify broad claims about full-depth chunk-KV/PIC alternatives

- **Location:** Section 5.2/Table 3, PDF pp. 6–7 (lines 427–457); Limitations, PDF p. 9 (lines 590–601); artifact `cacheblend_143`.
- **Exact quote (11 words):** “The included CacheBlend-style experiment is a minimal faithful full-depth chunk-KV control”
- **Problem:** CoMem uses a 58.2M-parameter self-distilled LoRA, while the CacheBlend-style arm is training-free and omits the native scheduler/cache manager. Its recomputation sweep stops at 18% even though the self-test only establishes equivalence at `r=1`. Therefore, 97.05 versus 74.70 and 18× smaller storage jointly compare representation, recomputation budget, and learned adaptation. The result is valid for this arm, but it does not isolate “one residual and upper recomputation” from training or establish superiority over well-tuned PIC/modular-KV systems.
- **Affected claim/norm and why it matters:** C5 and the abstract/conclusion emphasis can be read as a broad algorithmic comparison. The nearest practical alternatives are not compared under matched training/quality targets or native implementations.
- **Sufficient remedy:** Add either (i) a trained CacheBlend/PIC writer with a comparable LoRA/distillation budget, or (ii) quality-targeted recomputation curves extending toward full recomputation with latency/storage, ideally using a native implementation. Otherwise limit the headline to “our minimal training-free arm.”
- **Severity:** **Major**.

## W4. The proposed overlap repair is validated only on a narrow synthetic diagnostic, so the paper does not establish that it repairs the deployed quality frontier

- **Location:** Section 5.4/Tables 6–7, PDF p. 8; Limitations, PDF p. 9 (lines 628–638).
- **Exact quote (9 words):** “Overlap-Write is validated only on paired synthetic multikey instances”
- **Problem:** The 32-token overlap is a prominent scientific finding in the abstract, but it is tested only on two RULER multikey lengths (`n=200`). There is no LongEval, BABILong, LongBench, or LoCoMo evaluation, and no repeated-query break-even or Write-inclusive repaired frontier. Since Table 2 shows the largest natural loss on LongEval (97.2 to 69.0), it is unknown whether overlap repairs the failures that matter outside the constructed diagnostic.
- **Affected claim/norm and why it matters:** C3 is sound as a local mechanism finding, but the broader implication that overlap repairs “the remaining loss” or provides a practical remedy is unsupported.
- **Sufficient remedy:** Evaluate `w=0/32/128` on at least the matched LongEval and one multi-fact natural benchmark, and report quality plus Write time, invalidation expansion, and break-even under the same serving harness.
- **Severity:** **Major**.

## W5. The main deployment evidence omits concurrent throughput and tail latency

- **Location:** Table 4 and Section 5.3, PDF pp. 7–8; Limitations, PDF p. 9 (lines 615–623).
- **Exact quote (11 words):** “serving reports single-query medians rather than concurrent throughput or p95/p99 tails”
- **Problem:** The deployment framing concerns repeated-query serving, but the principal model-serving measurements are single-request process medians. The separate storage microbenchmark reports peak QPS for storage/fetch only, not model inference under contention. Large residual stores, CPU-pinned fetch, shared HBM, and decode scheduling can change both throughput and latency tails.
- **Affected claim/norm and why it matters:** C4’s break-even values are workload-specific latency calculations, not production-serving evidence. The omission limits the strength of deployment conclusions.
- **Sufficient remedy:** Report throughput and p50/p95/p99 TTFT/end-to-end latency at several concurrent request counts for `j=0` and CoMem, including store fetch and decode, plus the resulting empirical crossover.
- **Severity:** **Minor** for the scientific depth-axis claim, but important for systems impact.

## W6. The distillation objective discards unmeasured probability mass

- **Location:** Section 4, PDF p. 5 (lines 350–371); Appendix Table 29, PDF p. 22.
- **Exact quote (9 words):** “We did not retain the teacher mass captured by”
- **Problem:** Teacher and student are renormalized on the teacher’s top-64 support, all outside-support logits are discarded, and the captured mass was not logged. Consequently, the objective can overstate agreement when the teacher distribution is diffuse; its approximation quality and sensitivity to support size are unknown.
- **Affected claim/norm and why it matters:** This affects interpretation and reproducibility of the learned interface in C1/C6, though downstream evaluations partially validate the final adapter.
- **Sufficient remedy:** Report the distribution of teacher top-64 mass and ablate support sizes (e.g., 32/64/256 or full-vocabulary KL on a subset), including downstream RULER/LoCoMo changes.
- **Severity:** **Minor**.

# Questions That Could Change the Score

1. Can the authors provide the complete anonymous artifact promised in Appendix A.5 and demonstrate one-command reproduction of Tables 2, 5, 7, 34–35, and 40? A complete, runnable release would materially raise reproducibility and confidence.
2. Across three matched batch-8 `j=12` adapter runs, what are the run-level RULER-B, LongEval, LoCoMo, and overlap gains? If the headline gaps are stable, W2 would substantially weaken.
3. What happens to the CacheBlend-style arm when recomputation is swept beyond 18%, and when it receives a comparable distillation/LoRA budget? Please provide quality–latency–storage curves rather than one strongest tested point.
4. Does 32-token Overlap-Write recover any of the 28.2-point matched LongEval loss, and what is its measured Write-inclusive break-even?
5. Is the public arXiv preprint “Understanding Is Done Early…” by the same work/authors? It was posted July 30, 2026 and appears to contain the same method and exact result numbers. This is within three months and should not count against novelty, but contemporaneous-work attribution/anonymity handling may require an ARR chair check.

# Non-scoring Suggestions / Typos

- Abstract line “raises 92.5 to 98.5” should explicitly say “on the pooled 8k/16k RULER multikey diagnostic” to avoid implying general repair.
- Table 8’s `Write` column labels `j=0` as “per query,” although the object is raw token IDs and no residual Write is needed; “lower-layer compute per query” would be clearer.
- Table 3 calls the evidence path “same” while provenance records `iter_rounds=0`; Appendix clarifies that this means automatic rounds. Repeat that clarification in the table caption.
- The paper alternates “full-context KV-Direct” and a retrieved/full-context baseline terminology. Define each external baseline’s actual context path once in Section 5.
- Figure 2 is legible but underspecified in the main text: the model families, probe task, number of examples, and knee definition are deferred to Appendix A.2. Adding `n` and the normalized-depth definition to the caption would help.
- The `64.9×` number is mathematically correct but compares stock dense full-source prefill against adapted bounded retrieval plus depth reuse. Keeping “composed operating point” adjacent to every occurrence is advisable.

# Scores

## Soundness: 3.0 / 5.0

The narrow matched depth-axis result and Write-interface attribution are technically credible and well controlled. However, the learned interface is supported by one flagship run; the main baseline is training-confounded; and the proposed repair is synthetic-only. These prevent a higher score.

## Excitement: 3.5 / 5.0

Treating split depth as an explicit repeated-query systems axis is interesting and potentially useful. The clean `j=0` endpoint, negative equal-latency result, and context diagnosis make the work more than a speed benchmark. Excitement is moderated by close prior work on intermediate-layer retrieval/residual reuse and by incomplete practical validation.

## Overall: 3.0 / 5.0

**Findings-level / borderline main.** The paper contains a real, carefully measured contribution with unusually good scope discipline. I would not place it at ACL-main level yet because the central learned result lacks matched replication, the closest baseline comparison is not training-matched, the repair is not demonstrated on natural tasks, and the frozen object does not provide the promised reproducibility package. I choose the lower bin under the requested calibration.

## Confidence: 4.5 / 5.0

I read the complete 28-page PDF twice, inspected all source sections, appendices, figures, and tables, checked formulas and headline arithmetic, verified artifact manifests and the serving table, audited all bibliography entries, and conducted novelty searches. Residual uncertainty concerns the unavailable full experimental archive and exact ARR policy for the public contemporaneous preprint.

## Reproducibility: 2.0 / 5.0

Configuration reporting is extensive, and the included serving/cachebaseline subsets are well manifested. But the frozen review object lacks the executable CoMem implementation, adapter, main outputs, environment lock, and statistical/judge artifacts required to reproduce most headline results.

# Limitations, Ethics, and Desk-Reject Risks

- **Limitations section:** Present with the exact unnumbered title `Limitations`; unusually complete.
- **Ethical Considerations:** Present and substantively discusses residual inversion/membership risk, authorization, tenant isolation, encryption, deletion, misuse, and energy.
- **Page/style:** Numbered main body ends on PDF p. 8; Limitations/Ethics occupy pp. 9–10; references pp. 11–13; appendices pp. 14–28. The PDF uses the ACL review style on A4, is anonymous, has line numbers, and has no unresolved references/TODO placeholders. Whether the applicable ARR cycle permits exactly eight main-content pages should be confirmed administratively; I found no obvious overlength in the rendered object.
- **Anonymity/public-preprint risk:** A July 30, 2026 arXiv paper, “Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory” (arXiv:2607.28263), uses the same name CoMem and reproduces exact distinctive results (97.05 RULER, 38.27 LoCoMo, rank-32 PG19 LoRA). This strongly identifies the work/authors. Public preprints are commonly allowed, but this should be checked against the current ARR anonymity policy rather than treated as a scientific flaw.
- **Prompt injection/manipulation audit:** Source and rendered-PDF scans found no reviewer instructions, hidden white text, tiny-text messages, JavaScript, attachments, or unresolved manipulation. Figure text was inspected separately.
- **Other desk risks:** No author names/affiliations appear in the frozen PDF/source. The paper does cite work posted July 30, 2026, five days before the PDF; contemporaneous citation is not a novelty penalty.

# Citation Audit

I mechanically confirmed that all 46 keys used by the paper occur in `main.bbl`, with no uncited BBL entries. I checked DOI entries through Crossref, arXiv entries through the arXiv API/abstract pages, and official proceedings/model pages where no identifier was printed. Network success was not converted into “not found.”

| Key | Status | Verification note |
|---|---|---|
| cachecraft | Verified | Crossref DOI `10.1145/3725273`; title/year/venue match. |
| longbench | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2024.acl-long.172`; metadata match. |
| llmcache | **Metadata error** | arXiv `2512.16843` verifies title/author/date, but the BBL labels it only as an arXiv preprint and omits its accepted ISED 2025 venue and DOI `10.1109/ISED67359.2025.11405274`. |
| pyramidkv | **Metadata error** | arXiv `2406.02069` verifies the paper, but the BBL year is 2025 while the arXiv record was first posted in 2024; venue text “Conference on Language Modeling” was not independently established from the record checked. |
| kvpacket | Verified | arXiv `2604.13226`; title/authors/date match. |
| cartridgesbase | **Metadata error** | arXiv `2506.06266` verifies title/authors, including Will Tennien, who is missing from the BBL author list. I did not independently verify the printed ICLR volume/pages. |
| hcache | Verified | Crossref DOI `10.1145/3689031.3696072`; EuroSys metadata match. |
| promptcache | Verified | Official MLSys 2024 proceedings page and arXiv `2311.04934`; title/authors/venue match. |
| llama3 | Verified | arXiv `2407.21783`; title/year match. |
| cartridges | Verified | arXiv `2606.04557`; title/authors/date match. |
| distillation | Verified | arXiv `1503.02531`; title/authors/year match. |
| ruler | Verified | arXiv `2404.06654`; title/authors and COLM 2024 record match. |
| lora | Verified | arXiv `2106.09685`; title/authors and ICLR record match. |
| epic | Verified | arXiv `2410.15332`; title/authors and ICML 2025 metadata match. |
| ragcache | Verified | Crossref DOI `10.1145/3768628`; title/journal metadata match. |
| babilong | Verified | Crossref DOI `10.52202/079017-3381`; NeurIPS 2024 metadata match. |
| rag | Verified | Official NeurIPS 2020 page and arXiv `2005.11401`; title/authors/venue match. |
| longchat | Verified | Official LMSYS blog page; title/date/authors match the non-archival citation. |
| snapkv | Verified | Crossref DOI `10.52202/079017-0722`; NeurIPS 2024 metadata match. |
| ilre | **Metadata error** | arXiv `2508.17892` verifies title/authors/date, but the BBL gives no URL/identifier, reducing traceability. |
| readonce | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2021.acl-long.554`; metadata match. |
| minicache | Verified | Crossref DOI `10.52202/079017-4443`; NeurIPS metadata match. |
| turborag | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2025.emnlp-main.334`; metadata match. |
| blockattention | Verified | arXiv `2409.15355` and its “ICLR 2025” record; title/authors match. |
| locomo | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2024.acl-long.747`; metadata match. |
| xccache | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2024.findings-emnlp.896`; metadata match. |
| kvdirect | **Metadata error** | arXiv `2603.19664` verifies title/authors/date, but the BBL omits the arXiv identifier/link. |
| pg19 | Verified | arXiv `1911.05507`; title/authors/year match. |
| bm25 | Verified | Crossref DOI `10.1561/1500000019`; metadata match. |
| embeddingrecycling | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2023.findings-eacl.145`; metadata match. |
| gemfilter | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2026.findings-acl.677`; metadata match. |
| reform | Verified | arXiv `2506.01215`; title/authors/date match. |
| lloco | Verified | ACL Anthology/Crossref DOI `10.18653/v1/2024.emnlp-main.975`; metadata match. |
| hunyuan | Verified | Official Hugging Face model page exists; title/model scale match the citation. |
| fusionrag | Verified | arXiv `2601.12904`; title/authors/date match. |
| mepic | Verified | arXiv `2512.16822`; title/authors/date match. |
| longmem | Verified | Official NeurIPS 2023 page and Crossref DOI `10.52202/075280-3259`; title/authors/venue match. |
| memoryllm | Verified | arXiv `2402.04624` and PMLR ICML 2024 record; metadata match. |
| infllm | Verified | Crossref DOI `10.52202/079017-3801`; NeurIPS metadata match. |
| streamingllm | Verified | arXiv `2309.17453`; title/authors and ICLR 2024 record match. |
| sempic | Verified | arXiv `2607.28069`, posted July 30, 2026; title/authors/date match. |
| xu2024retrievallong | Verified | arXiv `2310.03025` and ICLR 2024 record; title/authors match. |
| qwen3 | Verified | arXiv `2505.09388`; title/year match. |
| ape | Verified | arXiv `2502.05431`; title/authors/date match. |
| cacheblend | Verified | Crossref DOI `10.1145/3689031.3696098`; EuroSys metadata match. |
| h2o | **Metadata error** | Official NeurIPS/Crossref DOI `10.52202/075280-1506` verifies the work; the BBL truncates the full author list to “et al.” and omits pages/identifier. |

## Load-bearing citation–claim matches

1. **ReadOnce / Embedding Recycling → reusable intermediate representations:** supported; both cache/reuse intermediate text representations, though not the same decoder serving interface.
2. **LLMCache → arbitrary-layer semantic activation reuse:** supported by its abstract; it is close conceptual prior art and should temper broad novelty language.
3. **HCache / KV-Direct → residual/hidden-state restoration:** supported; KV-Direct is especially close at the state level but reconstructs standard KV rather than selecting a document residual split and executing a query-conditioned suffix.
4. **CacheBlend / EPIC / APE / TurboRAG → modular per-layer KV reuse and context/position repair:** supported by the cited abstracts/proceedings.
5. **ILRe / REFORM / GemFilter → intermediate-layer token selection/compression:** supported. ILRe is particularly close because it encodes chunks only to a selected decoder layer and performs retrieval there, but it recalls tokens rather than persisting selected residuals for suffix continuation.
6. **Cartridges / KV Packet / SemPIC → learned reusable modular KV objects:** supported. These are close in amortized learned document-state reuse, but store/compile per-layer KV rather than one tunable split residual.
7. **RULER, BABILong, LongBench, LoCoMo benchmark attributions:** supported by the official records.
8. **Knowledge distillation and LoRA citations for the training method:** broadly appropriate, although the paper’s top-64 symmetric-KL objective is its own approximation and needs its own ablation (W6).

# Novelty Search Summary

I ran five targeted searches: (i) “persistent intermediate residual transformer document queries reuse depth,” (ii) “split depth transformer cache reuse document residual,” (iii) “intermediate hidden states cross-query reuse transformer document,” (iv) “resume upper layers cached hidden states repeated queries,” and (v) “residual stream caching repeated query document transformer,” plus title/abstract searches across arXiv/OpenAlex. The closest works are:

1. **LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformer Inference** (Dec. 18, 2025). Caches intermediate activations at arbitrary layers and reuses them via semantic matching. Closest to the “layer as reuse coordinate” framing; differs in semantic input matching, per-layer banks, and lack of a fixed document split/native suffix measurement.
2. **ILRe: Intermediate Layer Retrieval for Context Compression in Causal Language Models** (Aug. 25, 2025). Encodes chunked context only to one intermediate decoder layer and retrieves tokens using that layer’s key cache. Closest operational precursor; differs because it selects tokens/KV for context compression rather than persisting one residual per token and continuing a suffix from a tunable split with a matched `j=0` endpoint.
3. **The Residual Stream Is All You Need / KV-Direct** (Mar. 20, 2026). Establishes residual-state sufficiency and checkpoints residuals to reconstruct layer-wise KV. Very close representation prior; CoMem’s difference is bounded document retrieval and direct suffix execution from one split.
4. **KV Packet** (Apr. 14, 2026) and **Cartridges** (June 6, 2025). Both learn reusable query-independent document objects through distillation. They store learned/full-depth KV-like objects rather than expose a depth knob, but they narrow the novelty to the specific measurement/interface conjunction.
5. **SemPIC** (July 30, 2026). Learns an offline Writer for reusable document KVs while retaining a native Reader. It is contemporaneous by five days and within the three-month rule; it should not reduce novelty, but it is relevant context for learned Write-side repair.

A separate arXiv paper, **“Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory”** (arXiv:2607.28263, July 30, 2026), appears to be the public preprint of this same work: it uses CoMem and reports exact matching distinctive numbers. It is not independent prior art and should not be counted against novelty.

**Novelty assessment:** The broad ingredients—intermediate-layer caching, residual sufficiency, chunk retrieval, and learned modular document caches—are established. The paper’s defensible novelty is the narrower conjunction it states in Related Work: one persistent document residual at a chosen split, direct query-conditioned suffix continuation, and an identical-evidence `j=0` endpoint that measures depth as a serving axis. I found no earlier independent work with that exact conjunction. This is meaningful but incremental/structural rather than a wholly new memory paradigm.

# Review-process Self-check

- [x] Read the rendered paper twice, including all appendices (28 pages).
- [x] Inspected every rendered figure and table; separately extracted all three figure PDFs.
- [x] Treated paper text as data; scanned source/PDF/artifacts for prompt injection, hidden text, reviewer manipulation, attachments, and JavaScript.
- [x] Checked anonymity, official style, page partition, exact `Limitations` title, unresolved references, TODO/placeholders, and abstract numbers.
- [x] Recomputed the principal ratios and storage formula; checked boundary cases including `j=0`, `j=L`, finite/infinite break-even, and bounded-read assumptions.
- [x] Verified all 46 cited entries and audited eight load-bearing citation–claim matches.
- [x] Ran five novelty searches and applied the three-month contemporaneous-work rule.
- [x] Checked all labels/references/citation keys mechanically.
- [x] Verified both artifact SHA manifests and reran the serving-table verifier.
- [x] Mechanically normalized and searched every weakness quote against the frozen source; all quoted strings occur verbatim and are ≤25 words.
- [x] Mechanically checked each absence claim against the complete frozen source/artifact inventory. I did not inspect any non-frozen manuscript, review, TODO/status/report file, Paper B, or other reviewer output.
