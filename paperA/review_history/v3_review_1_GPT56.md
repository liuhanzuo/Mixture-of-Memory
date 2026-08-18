# Independent ARR Review — Paper A, frozen version v3, review 1

## Review scope and evidence protocol

- **Frozen source:** `review_history/v3_source`
- **Frozen PDF:** `review_history/v3_latest_20260803_204224.pdf`
- **PDF fingerprint:** SHA-256 `28f3ab7cd5813fd9bd64f55ac3728ef146fcabf535ccf45a3fe099cfeae68e2a`
- **Review date:** 2026-08-03
- **Evidence restriction followed:** I treated the manuscript as data rather than instructions and read only the frozen source and frozen PDF inside `PAPER_DIR`. I did **not** read prior review/report/history files.
- **Reading protocol:** two passes over all 25 PDF pages, including appendices and references. Pass 1 reconstructed the argument and claim graph. Pass 2 audited claim support, mathematics, controls, statistics, reproducibility, citations, every rendered figure, and Tables 1–37.
- **Anchor convention:** `PDF pX:Lm–n` means the printed PDF page and the manuscript line numbers shown in the margin. For captions or table rows without margin numbers, I also give frozen source file and line anchors.

---

## 1. Paper summary

The paper proposes **CoMem**, a long-context serving interface that writes each context chunk once through the lower \(j\) transformer layers, stores one intermediate residual vector per token, retrieves a bounded top-\(k\) chunk set per query, and resumes only the upper layers over the selected residuals plus the query. A matched \(j=0\) path replays the same selected raw-text pack through all layers. The main empirical operating point uses Qwen3-8B, \(j=12/36\), 512-token chunks, iterative BM25 top-12 selection, and a rank-32 self-distillation LoRA over layers 12–35.

The central controlled result is a **quality–Read-latency trade-off**, not quality preservation: on the same selected pack, \(j=12\) lowers a 15-cell RULER-B macro from 99.19 to 96.07 while reducing the isolated Read phase from 931.9 ms to 664.4 ms (\(1.403\times\)). The paper also distinguishes that depth-only gain from the much larger \(64.9\times\) online-prefill number obtained when bounded selection is compared with dense 128k prefill. A focused multikey diagnostic attributes a substantial part of the deployable loss to independently written chunks lacking lower-layer left context; a 32-token overlap raises 92.5 to 98.5 without changing persistent bytes or per-query Read work, but the repair is not run across the full benchmark suite.

My overall reading is that this is a careful and unusually candid systems/empirical paper with a useful interface abstraction and strong internal controls. However, the current submission leaves two important publication-level gaps: (i) its closest systems comparison is not experimentally closed, especially against **TurboRAG/CacheBlend/Cache-Craft-style reusable chunk-state systems**, and TurboRAG is not cited at all; and (ii) the key overlap-Write repair is supported only on a narrow two-cell synthetic diagnostic, so the paper does not yet establish that its best repaired interface improves the broader quality–latency frontier. These are substantial but plausibly next-cycle-fixable issues.

---

## 2. Claim inventory

### C1 — Interface/novelty claim
**Claim.** CoMem is a cross-query memory interface combining persistent independently written intermediate residuals, bounded query-conditioned selection, and direct continuation from a chosen transformer depth.

- **Primary anchors:** Introduction, PDF p2:L108–126; Related Work, PDF p3:L173–210; Table 1, PDF p3; `sections/02_related.tex:23–43`; `sections/tab_priorart.tex:8–24`.
- **Scope stated by authors:** novelty is restricted to the conjunction of interface properties, not to inventing residual checkpoints, retrieval, or chunk caching individually.

### C2 — Matched depth-reuse frontier
**Claim.** On an identical selected pack and with the same LoRA, resuming at layer 12 gives a \(1.403\times\) isolated Read-phase speedup at a 3.12-point RULER-B cost.

- **Primary anchors:** Table 2, PDF p7; Experiments, PDF p6:L444–459 and p7:L450–469; Appendix Table 28, PDF p21:L1066–1089; `sections/tab_core_tradeoff.tex:11–30`.

### C3 — Bounded online working set and large dense-prefill operating-point gain
**Claim.** With fixed \(k,c\), model-side Read length is independent of stored-context length; at 128k, select-first online prefill is 1.10 s/18.7 GB versus dense 71.37 s/50.0 GB, i.e. \(64.9\times\), explicitly combining selection and depth reuse.

- **Primary anchors:** Method, PDF p5:L290–310 and p6:L383–392; Table 3, PDF p7; Experiments, PDF p7:L485–496; `sections/04_methodology.tex:39–45,132–149`; `sections/tab_online_prefill.tex:11–23`.

### C4 — Self-distillation repairs the residual/upper-layer interface
**Claim.** A rank-32 top-64-logit-support KL LoRA recovers most of the otherwise severe interface loss without updating the backbone and transfers across evaluated tasks.

- **Primary anchors:** Method, PDF p6:L353–380; Appendix Tables 7, 8, 10, 11, PDF pp12–13; `sections/04_methodology.tex:103–126`; `sections/tab_depth_tradeoff.tex:55–66`; `sections/tab_adapter.tex:19–23`.

### C5 — Repeated-query amortization and storage trade-off
**Claim.** The 8,192-byte/token residual store and one-time Write become advantageous only under reuse, with measured crossover around 8–11 queries at 32k and 25.8/27.6 queries at 128k for GPU/CPU-resident stores.

- **Primary anchors:** Method, PDF p5:L290–306; Experiments, PDF p7:L470–484; Appendix A.4, PDF pp19–20:L1026–1055; `sections/05_experiments.tex:50–58`; `sections/08_appendix.tex:223–248`.

### C6 — Lower-layer Write context is a major tested fidelity factor
**Claim.** On a paired 8k/16k RULER multikey diagnostic, missing lower-layer document context during independent writing is a major source of loss; a 32-token overlap raises 92.5 to 98.5 and a document-context control reaches 100.

- **Primary anchors:** Table 4, PDF p8; Experiments, PDF p7:L497–511 and p8:L512–518; Appendix Table 29, PDF p22; `sections/05_experiments.tex:71–85`; `sections/tab_write_context.tex:11–25`.

### C7 — Retrieval and residual readout are distinct bottlenecks
**Claim.** BM25 is adequate on lexical needle tasks, while other tasks exhibit retrieval misses, residual readout failures conditional on hits, or both.

- **Primary anchors:** Design Evidence, PDF p4:L239–265; Experiments, PDF p8:L519–528; Appendix Tables 12, 13, 17, 18, 23; `sections/03_motivation.tex:38–47`.

### C8 — Portability/generalization claim
**Claim.** The interface ports to multiple Qwen scales and sparse MoEs, including Qwen3-30B-A3B and Hy3, though strongest controlled claims remain Qwen3-8B.

- **Primary anchors:** Experiments, PDF p8:L545–558; Appendix A.7, PDF pp22–23:L1124–1173; Tables 31–35.

---

## 3. Desk-review checklist

| Item | Finding |
|---|---|
| In scope for ACL/ARR | **Yes.** Long-context NLP inference, retrieval, memory, and evaluation are directly relevant. |
| Main-paper length | **Appears compliant:** 8 content pages, followed by limitations/ethics/references and appendices. |
| Anonymity | **No obvious deanonymization in the rendered PDF.** The source contains internal artifact/log path comments, but these are not rendered and do not name authors. |
| Readability/format | **Pass.** Two-column layout is legible; no visible clipping or unresolved references. |
| Missing citations in compilation | **None:** all 47 cited keys occur in frozen `main.bbl`, and every `main.bbl` item is cited. |
| Ethics/limitations sections | **Present and substantive**, PDF p9:L581–666. |
| Human subjects/new annotation | **None claimed.** LoCoMo uses released synthetic conversations; no new annotators. |
| Artifact claim | Paper says an anonymous code archive accompanies the release, with hashes/configuration recorded; archive itself was outside the permitted evidence set and was not audited. |
| Obvious fabricated reference | **None found.** See Section 6 below. |
| Broken cross-reference | **None rendered.** Frozen source has duplicate unused labels for standalone/compact table variants, but the PDF resolves correctly. |
| Fatal mathematical inconsistency | **None found.** Storage ratio, pack size, RULER aggregates, and headline speed ratios recompute correctly. |
| Ethics escalation | **Not needed.** Privacy/inversion risks and access-control/deletion duties are acknowledged. |

**Desk outcome:** send to full review; no desk-reject issue detected.

---

## 4. Strengths

### S1. The central depth-only comparison is unusually well matched.

- **Anchor:** Experiments §5.2, PDF p6:L444–459 and Table 2 on p7; `sections/tab_core_tradeoff.tex:19–30`.
- The \(j=0\) and \(j=12\) arms share selected chunks, order, sink, mask, examples, and LoRA. The paper also states exactly what the 931.9/664.4 ms Read timing excludes.
- The continuous-prefix \(h_{12}\) oracle exactly matches \(j=0\), which is a strong implementation sanity check and properly framed as a ceiling rather than single-factor attribution.

### S2. The paper carefully separates bounded selection, depth reuse, Write, retrieval, I/O, and decode.

- **Anchor:** Experiments §5.1, PDF p6:L437–443; Table 3 p7; Appendix A.4, PDF pp19–21:L1026–1089.
- This accounting prevents the \(64.9\times\) dense-vs-bounded result from being misrepresented as a depth-only speedup. The abstract, main text, captions, and limitations repeat this distinction consistently.
- Store-tier microbenchmarks and measured crossover points make the systems trade-off more concrete than a FLOP-only analysis.

### S3. Negative results are prominent rather than buried.

- **Anchor:** §5.5 and Table 5, PDF p8:L529–544; `sections/tab_equal_latency.tex:16–21`.
- At equal online latency, raw-text replay beats CoMem by 11.56 points. The authors use this to narrow the claim rather than to alter the comparison.
- Other useful negative boundaries include one-off-query disadvantage, bounded-evidence ceilings, lexical-retrieval limits, super-window caveats, and the statement that CoMem is not quality preserving.

### S4. The mechanism diagnosis uses several meaningful controls.

- **Anchor:** §5.4, PDF p7:L497–511 and p8:L512–518; Tables 4 and 29, PDF pp8 and 22.
- Same-pack replay, a continuous-prefix oracle, context/position factorization, and deployable overlap widths address different confounds. The paper explicitly avoids an additive causal decomposition because factors interact.
- The resulting engineering hypothesis—put limited neighboring context on the Write side—is concrete and testable.

### S5. Statistical reporting is stronger than typical for this type of systems paper.

- **Anchor:** Appendix B, PDF pp24–25:L1174–1276.
- The paper gives exact sample counts and cohort definitions, paired bootstrap intervals, McNemar counts/tests for paired binary outcomes, conversation-cluster bootstrap for LoCoMo, and an independent-judge audit.
- It explicitly avoids treating a bootstrap tail fraction as a calibrated \(p\)-value and avoids pooling distinct RULER cohorts.

### S6. Reproducibility details are extensive.

- **Anchor:** Appendix Tables 24–26, PDF pp20–21; `sections/08_appendix.tex:113–218`.
- Exact model revision prefix, architecture, RoPE configuration, retrieval formula and iteration semantics, LoRA placement, objective, optimizer, seed, generation settings, sample counts, scorers, and artifact hashes are supplied.
- The storage equation is correct for Qwen3-8B: \(4096/(2\cdot36\cdot8\cdot128)=1/18\), i.e. 8 KiB residual versus 144 KiB full KV per token in bf16.

### S7. Limitations and societal-impact discussion are candid and specific.

- **Anchor:** Limitations/Ethics, PDF p9:L581–666; `sections/07_limitations.tex:4–37`; `sections/07_ethics.tex:3–25`.
- The paper acknowledges model/tokenizer coupling, rewrite-on-upgrade, linear store growth, edit invalidation under overlap, untested quantization/eviction/contention, English/lexical limitations, and residual inversion/membership risks.

---

## 5. Weaknesses and required remedies

### W1. **Major — Closest systems baselines and novelty positioning are incomplete.**

- **Location:** Related Work/Table 1, PDF p3:L173–210 and Table 1; frozen source `sections/02_related.tex:23–43`, `sections/tab_priorart.tex:8–24`.
- **Short quote (10 words):** “Cache-Craft is particularly close in workload”
- **What is weakened:** C1 (novelty/interface claim), C2/C3/C5 as a competitive systems contribution, and the ARR expectation that the nearest prior work be represented and experimentally distinguished.
- **Evidence:** The paper cites CacheBlend and Cache-Craft, but does not cite **TurboRAG** (arXiv:2410.07590, 2024-10-10), which independently precomputes per-document KV offline, retrieves reusable states, adapts masks/positions, fine-tunes the reader, and reports LongBench multi-document QA with 7.0–9.4× TTFT gains. More importantly, no CacheBlend/TurboRAG/Cache-Craft quality–latency–storage operating point is implemented on a common backbone, retrieval pack, task, hardware, and timing boundary. The only “HCache-style” empirical row is an adapter diagnostic, not a systems comparison.
- **Why this matters:** CoMem’s distinctive stored object and explicit depth knob remain plausible contributions, but without a matched near-neighbor experiment one cannot tell whether storing \(h_j\) is preferable to storing/repairing KV for the paper’s target repeated-query RAG workload. The current baseline suite is strongest against raw replay and token/KV compression, not against the closest reusable chunk-state systems.
- **Remedy:** Add TurboRAG to Related Work and Table 1. Implement at least one same-backbone, same selected chunks/order, same task, same storage tier, same hardware comparison against a representative reusable-KV system (TurboRAG, CacheBlend, or Cache-Craft). Report persistent bytes/token, offline preprocessing, online TTFT/Read, decode, external I/O, and quality. If faithful reimplementation is infeasible, sharply narrow the systems-superiority implication and provide a detailed non-comparability analysis.

### W2. **Major — The proposed overlap-Write repair is not validated on the broader benchmark suite.**

- **Location:** §5.4/Table 4, PDF p7:L497–511 and p8:L512–518; Limitations, PDF p9:L612–620; `sections/tab_write_context.tex:19–25`.
- **Short quote (9 words):** “tested on a focused multikey diagnostic”
- **What is weakened:** C6 as an actionable general repair and the conclusion that contextual writing is a promising interface direction for long-memory systems.
- **Evidence:** The 92.5→98.5/99.0 result pools only `niah_multikey_1` at 8k and 16k (\(n=200\)). The flagship results on LongEval, BABILong, LongBench, LoCoMo, longer RULER lengths, and equal-latency quality all use independent writing. Thus the paper identifies a mechanism on two synthetic cells but never shows whether the repair improves the actual headline frontier, transfers beyond lexical multikey retrieval, or changes crossover latency once measured Write overhead is included.
- **Why this matters:** This is the paper’s main technical diagnosis beyond the interface itself. On broader tasks, overlap could help, be neutral, or introduce contamination/distractor effects. The current evidence cannot distinguish these outcomes.
- **Remedy:** Evaluate \(w=32\) at minimum on the full RULER-B cohort plus LongEval, BABILong, and LoCoMo (or a justified representative subset), with paired confidence intervals. Measure actual Write latency and recompute reuse crossover. Include an equal-latency repaired-CoMem comparison. If broad evaluation is not possible, demote the repair to a diagnostic observation and soften the conclusion.

### W3. **Major — The reported speedups do not yet establish production end-to-end serving advantage.**

- **Location:** Table 2/3 and Appendix A.4, PDF p7 and pp19–22:L1026–1100; `sections/tab_core_tradeoff.tex:22–25`; `sections/tab_online_prefill.tex:17–23`.
- **Short quote (10 words):** “retrieval, persistent I/O, reusable Write, and decode are excluded”
- **What is weakened:** C2/C3/C5 as deployment claims.
- **Evidence:** The \(1.403\times\) headline is an isolated Read phase, while the appendix says total-decode medians are roughly 2.76–2.86 s and “total latency remains decode-dominated.” The \(64.9\times\) number excludes document Write and external fetch and compares a 6,657-token bounded pack with dense 128k. The Write-inclusive L20A result still excludes BM25 index construction and external-store I/O. Store-tier I/O is measured separately rather than integrated, and no concurrent multi-query end-to-end workload reports p50/p95 TTFT, throughput, or tail latency.
- **Why this matters:** The paper is honest about every boundary, which is a strength, but the practical importance of the depth-reuse increment remains uncertain once selection, fetch, scheduling, and generation dominate.
- **Remedy:** Add an integrated repeated-query serving experiment with store construction, index construction amortization, retrieval, fetch/H2D, query Write, Read, and decode. Report p50/p95 TTFT and end-to-end latency, throughput under concurrency, several output lengths, GPU- and off-GPU stores, and direct raw-text/near-neighbor baselines. Preserve the component breakdown.

### W4. **Minor — The equal-latency challenge is called “pre-registered” without a verifiable preregistration record.**

- **Location:** §5.5, PDF p8:L529–534; `sections/05_experiments.tex:98–102`.
- **Short quote (6 words):** “We pre-register a direct challenge”
- **What is weakened:** Experimental transparency/confirmatory-status interpretation for Table 5.
- **Evidence:** No timestamped protocol, registry identifier, archived analysis plan, or prospective split definition is given in the frozen paper.
- **Remedy:** Cite the timestamped preregistration or replace “pre-register” with “pre-specify”/“pose.” Also report calibration/evaluation cohort composition, per-task counts, exact online latencies for both arms, and the bootstrap resampling unit.

### W5. **Minor — LoCoMo’s primary judge cannot be exactly reproduced from the paper alone.**

- **Location:** Appendix B.2, PDF p24:L1216–1231; `sections/08_statistics_appendix.tex:77–88`.
- **Short quote (8 words):** “does not expose a dated model snapshot”
- **What is weakened:** C4 and cross-method LoCoMo comparisons; reproducibility.
- **Evidence:** The endpoint is named only `gpt-4o`, with no dated snapshot and no fixed temperature/top-\(p\) set by the client. The independent DeepSeek audit supports ordering, but absolute scores can drift.
- **Remedy:** Re-run and release judgments from a dated/frozen judge or an open-weight judge, archive raw judge outputs, and make the deterministic lexical metrics a co-primary reproducible reference rather than only diagnostics.

### W6. **Minor — Training robustness is not a fully controlled seed study.**

- **Location:** Appendix A.5, PDF pp21–22:L1090–1105; `sections/08_appendix.tex:279–287`.
- **Short quote (8 words):** “effective batch 3 rather than 8”
- **What is weakened:** C4’s stability across training randomness.
- **Evidence:** The two additional adapters change effective batch size, and robustness is summarized by median/max cell-wise standard deviation rather than headline aggregate intervals across identically configured seeds.
- **Remedy:** Train at least three identically configured global-batch-8 seeds; report mean±SD/CI for RULER-B, BABILong, LongEval, and LoCoMo, plus paired comparisons to \(j=0\).

### W7. **Minor — Several broad benchmark results are vulnerable to training-corpus overlap.**

- **Location:** Appendix A.3, PDF p19:L958–976; `sections/08_appendix.tex:100–111`.
- **Short quote (9 words):** “did not complete equivalent overlap audits for every natural benchmark”
- **What is weakened:** Broad generalization aspects of C4/C8, especially NarrativeQA/LongBench.
- **Evidence:** The authors responsibly remove one contaminated InfiniteBench comparison, but PG-19 overlap with other natural benchmarks is not fully audited and no clean-subset rescore is available.
- **Remedy:** Complete overlap checks for all natural-text benchmarks materially used in claims, release the audit method, and report clean-subset or synthetic-only sensitivity analyses where overlap is detected.

### W8. **Minor — Some appendix portability claims use small samples or incomplete measurements.**

- **Location:** Tables 8, 31, 34, 35, PDF pp13, 22–23; `sections/tab_distilled_depth_curve.tex:21–25`, `sections/tab_scale.tex:18–23`, `sections/tab_hy3_distill.tex:17–23`.
- **Short quote (10 words):** “The matched fixed-pack Write value was not retained”
- **What is weakened:** C8 and the shape of the depth frontier outside the flagship.
- **Evidence:** The 32B sweep uses \(n=25\), Hy3 distillation uses 16 PG-19 documents, Hy3 RULER uses \(n=50\), and the \(j=12\) fixed-pack Write value is absent from the per-depth table. These are useful exploratory ports, not strong generality evidence.
- **Remedy:** Label these rows consistently as exploratory, retain all timing components, increase samples, and add matched \(j=0\) or architecture-native baselines where feasible.

---

## 6. Citation audit

### 6.1 Frozen `main.bbl` authenticity check

- Frozen `main.bbl` contains **47 entries**; the source cites **47 unique keys**. There are no missing or unused compiled entries.
- I checked every entry’s title/identifier against at least one of: arXiv metadata, DOI/Crossref, ACL Anthology, official venue metadata, or the official model page.
- **Result:** no fabricated paper was detected. The recent 2026 items (e.g., KV-Direct, KV-CAT, REAL, HeteroCache, IndexMem) resolve to real records. The Hy3 entry resolves to the official Tencent model page and its published configuration is consistent with the paper’s 80-layer/192-expert/native-262k description.
- Minor bibliographic-quality caveats: several entries are abbreviated with “et al.”; Hy3 is a model-card citation rather than an archival paper; and a few venue records lack DOI/arXiv identifiers in `main.bbl`. These are not authenticity failures.

### 6.2 Citation-to-assertion matching (8 checks)

| # | Manuscript assertion | Cited work(s) | Match judgment |
|---|---|---|---|
| 1 | Text RAG selects raw text and recomputes the reader. Introduction, PDF p1:L048–051. | Lewis et al. 2020; Xu et al. 2024 | **Good.** Both are retrieval-plus-reader systems rather than reusable hidden-state interfaces. |
| 2 | StreamingLLM uses recent-window/sink retention; H2O/SnapKV evict/select token positions. Related Work, PDF p3:L133–139. | StreamingLLM, H2O, SnapKV | **Good.** Accurate family-level summary. |
| 3 | HCache restores evicted LLM state from intermediate activations. Related Work, PDF p3:L146–149. | HCache | **Good.** This is HCache’s central mechanism. |
| 4 | KV-Direct reconstructs layer-wise KV from residuals while retaining the full sequence. Related Work, PDF p3:L149–151. | KV-Direct | **Good.** Its main theorem and system checkpoint one residual/token and recompute KV. |
| 5 | ILRe encodes to an intermediate layer and uses that representation for token retrieval/context compression. Related Work, PDF p3:L151–153. | ILRe | **Good.** Accurate. |
| 6 | REFORM compresses incrementally, gathers salient tokens, and recomputes KV. Related Work, PDF p3:L153–156. | REFORM | **Good.** Accurate two-phase summary. |
| 7 | CacheBlend/Cache-Craft reuse or repair cached chunk states for RAG serving. Related Work, PDF p3:L173–181. | CacheBlend, Cache-Craft | **Good but incomplete.** The descriptions fit, but TurboRAG is a conspicuous omitted member of the same closest family. |
| 8 | The training objective follows knowledge distillation and LoRA. Method, PDF p6:L353–367. | Hinton et al. 2015; Hu et al. 2022 | **Good as background attribution.** The bidirectional top-64 KL mixture is the paper’s own objective, not claimed to be directly from either citation. |

### 6.3 Material missing references

1. **TurboRAG: Accelerating Retrieval-Augmented Generation with Precomputed KV Caches for Chunked Text** (arXiv:2410.07590; submitted 2024-10-10). This is the most important missing reference because it precomputes per-document KV offline, retrieves it online, modifies attention/positions, fine-tunes the reader, and evaluates LongBench multi-document QA.
2. The frozen `.bib` already contains but the paper does not cite **The Remarkable Robustness of LLMs: Stages of Inference?** (arXiv:2406.19384), a relevant depth-intervention paper for the Design Evidence discussion.
3. The frozen `.bib` also contains but does not cite **CompressKV** (arXiv:2606.24467; 2026-06-23), which is relevant to recent semantic-retrieval-guided KV compression, though less central than TurboRAG.

---

## 7. Novelty search and three-month rule

### Search questions

1. **Has prior work already precomputed reusable states for independently retrieved RAG chunks across queries?**
2. **Has prior work stored intermediate activations/residuals and resumed transformer execution from that depth?**
3. **Has prior work combined bounded retrieval with intermediate-layer/residual representations?**
4. **Has prior work diagnosed missing cross-chunk context and repaired independently cached chunks?**
5. **Were highly similar papers posted in the three months before the frozen date (2026-05-03 through 2026-08-03)?**

### Nearest-work map

| Work | First public date | Nearest overlap | Main distinction from CoMem |
|---|---:|---|---|
| CacheBlend | 2024-05-26 | Reusable per-chunk KV, cross-chunk repair, RAG workload | Stores full-depth KV and selectively recomputes KV; no explicit residual depth knob. |
| HCache | 2024-10-07 | Intermediate-activation checkpoint and resumed execution | Restores evicted request/context state; not independently written bounded RAG memory. |
| TurboRAG | 2024-10-10 | Offline per-document KV, online retrieval/injection, mask/position adaptation, reader fine-tuning, LongBench | Full-depth KV rather than one depth-\(j\) residual; nevertheless a very close workload/system baseline. |
| Cache-Craft | 2025-02-05 | Cross-query recurring chunks, contextual-dependency repair, storage/eviction | Full-depth chunk KV plus selective repair; no residual upper-layer continuation. |
| REFORM | 2025-06-01 | Incremental compression, bounded salient-token gathering, KV recomputation | Request-bound streaming representation, not persistent independent residual chunks. |
| ILRe | 2025-08-25 | Intermediate-layer chunked encoding and bounded token retrieval | Uses intermediate state to select tokens, then does not directly continue from stored selected residuals as CoMem does. |
| \(A^3\) / FusionRAG | 2025-11-13 / 2026-01-19 | Reusable KV fusion and missing-context repair | Full-depth KV recomputation/fusion, not a tunable residual split. |
| KV-Direct | 2026-03-20 | One residual vector/token and exact KV reconstruction | Full sequence/request-bound; no bounded cross-query chunk selection. |

### Novelty conclusion

I did **not** find a pre-May-2026 paper that clearly contains the exact four-way conjunction claimed by CoMem: cross-query persistence, independently written chunks, bounded query-conditioned selection, and direct continuation from the same stored residual depth. The novelty claim is therefore **plausible but narrow**. It is best understood as a new interface combination and operating-point study, not a wholly new principle of reusable hidden states.

The closest conceptual/system predecessors are **TurboRAG, CacheBlend, Cache-Craft, HCache, and KV-Direct**. Because TurboRAG is omitted and no matched reusable-KV baseline is run, the current paper does not fully establish the practical novelty/advantage of choosing residuals rather than repaired KV.

### Three-month window: 2026-05-03 to 2026-08-03

The following works fall inside the window and are relevant:

- **Decoupled Attention Fusion** — 2026-05-03: reusable RAG KV and cross-document context fusion.
- **KV-CAT** — 2026-05-07: trains representations for KV compressibility; already cited.
- **IndexMem** — 2026-05-25: latent compensation for bounded KV eviction; already cited.
- **SIFT** — 2026-06-08: repeated-query RAG, offline document analysis, bounded selective online computation.
- **CompressKV** — 2026-06-23: semantic-retrieval-guided KV compression; present in frozen `.bib` but not cited.
- **KV-cache systems survey** — 2026-06-30: useful for updated systems positioning.
- **InferScale** — 2026-07-29: persistent personalized facts, independently encoded reusable KV, bounded retrieval, context-window encoding, and LoCoMo; this is especially close in workload and its contextual encoding strongly parallels overlap-Write, but it appeared only five days before the freeze.

Under a reasonable three-month rule, these very recent works should not be used to deny novelty retroactively. They should, however, be acknowledged in a revision where feasible. **InferScale is the closest late-arriving parallel work**: it stores KV rather than residuals and has no depth knob, but independently discovers that limited preceding context during offline encoding repairs independently cached memory facts.

---

## 8. Claim-by-claim technical and experimental assessment

### C1 — Interface/novelty

- **Technical validity:** The interface is coherent, and Algorithm 1 plus the generation pathway specify how lower- and upper-band caches advance.
- **Current evidence:** Table 1 and related work distinguish stored object, lifetime, write independence, bounded selection, and continuation.
- **Ideal experiment:** Same-backbone CoMem versus TurboRAG/CacheBlend/Cache-Craft under identical retrieved evidence and storage tier.
- **Baseline gap:** Major; see W1.
- **Reproducibility:** Method semantics are sufficiently specified.
- **Verdict:** **Plausibly novel interface combination; competitive significance not yet established.**

### C2 — Matched depth-reuse frontier

- **Technical validity:** Strong. Arithmetic checks: \(931.9/664.4=1.4026\); RULER-B cells average 96.0667; the 99.19–96.07 gap is 3.12.
- **Experiment quality:** Same pack/examples/LoRA and paired statistics are excellent controls.
- **Ideal experiment:** Repeat on at least one second mainstream backbone with identical matched protocol and report full TTFT/end-to-end.
- **Statistics:** Paired bootstrap and exact McNemar are appropriate. Three independent timing processes are informative but still limited for tail latency.
- **Verdict:** **Well supported as an isolated Read-quality trade-off.**

### C3 — Bounded Read and \(64.9\times\) dense-prefill result

- **Technical validity:** For fixed \(k,c\), model-side pack length is bounded; the paper correctly states selector/index costs still scale.
- **Experiment quality:** The number is correct (\(71.37/1.10=64.88\)) and honestly labeled as combined selection+depth.
- **Ideal experiment:** Add a raw-text top-12 selected-pack online-prefill row in Table 3 and integrated end-to-end serving.
- **Baseline/benchmark issue:** Dense 128k is a different evidence budget, so this is an operating point, not a causal method comparison.
- **Verdict:** **Supported with appropriate caveats; practical interpretation remains limited by W3.**

### C4 — Self-distillation

- **Technical validity:** Objective and adapter placement are explicit. Same-\(j\) on/off controls show large improvements.
- **Experiment quality:** Strong task breadth, but the training-seed study changes batch size and the adapter is still evaluated primarily on one backbone.
- **Ideal experiment:** Identical multi-seed training and a baseline against CE, forward-KL only, reverse-KL only, and support size/temperature choices.
- **Statistics:** Evaluation uncertainty is strong; training uncertainty is weaker.
- **Verdict:** **Strong evidence that adaptation is necessary and useful; exact recipe optimality and seed stability are not established.**

### C5 — Storage/amortization

- **Technical validity:** Storage calculation is correct and the paper distinguishes raw IDs, residuals, and full KV.
- **Experiment quality:** Multiple cohorts and storage tiers are useful, but timing boundaries differ; integrated serving is absent.
- **Ideal experiment:** Workload trace with query arrival distribution, document edits, cache evictions, multiple output lengths, and p95 crossover.
- **Verdict:** **Reasonably supported for the measured harnesses, not yet a general production threshold.**

### C6 — Write-context diagnosis and overlap repair

- **Technical validity:** The 2×2 context/position controls and continuous-prefix oracle make the attribution plausible within the diagnostic.
- **Experiment quality:** Pairing and CI are good, but scope is only two synthetic cells; theoretical FLOPs do not substitute for measured Write latency.
- **Ideal experiment:** Full benchmark transfer and repaired equal-latency/end-to-end frontier.
- **Verdict:** **Strong local diagnostic, insufficient broad support for the design recommendation.**

### C7 — Retrieval versus readout

- **Technical validity:** Conditional-on-hit decompositions are the right analysis.
- **Experiment quality:** The paper covers lexical, recency, reader-attention, oracle, and one dense retriever; BM25 is intentionally favorable to entity/key overlap.
- **Ideal experiment:** Strong modern dense/multivector or learned retriever on all main tasks with recall-quality-latency decomposition.
- **Verdict:** **Supported for the tested lexical operating point; selector-agnostic claims should remain architectural, not empirical.**

### C8 — Portability

- **Technical validity:** Exact partition tests on Hy3 are useful.
- **Experiment quality:** Many rows are exploratory, with changed splits/configurations and small \(n\).
- **Ideal experiment:** Matched \(j=0/j>0\), same-pack, same-LoRA controls on a second model family and a sparse MoE, with full timing.
- **Verdict:** **Evidence of implementability/portability, not yet matched generality.**

---

## 9. Figure and table audit

### Figures

1. **Figure 1 (PDF p2):** Accurate high-level Write/Select/Read schematic. It clearly marks \(j=0\) replay and overlap-Write. No visible clipping.
2. **Figure 2 (PDF p4):** Readable but compact. The caption appropriately limits the probe to accessibility/readout, not “understanding.” Numerical definitions are deferred to Appendix Table 6.
3. **Figure 3 (PDF p4):** Supports only a narrow lexical single-needle selection claim; the caption correctly says selection is not always the bottleneck.

The Figure 2 accessibility evidence is correlational and not required for the system to function; the paper largely says this correctly. A more complete framing would cite prior layer-deletion/intervention work and avoid treating the probe knee as evidence for the selected long-memory split.

### Main-paper tables

1. **Table 1 (p3):** Useful interface taxonomy, but incomplete due to omitted TurboRAG; checkmarks compress nuanced conditions.
2. **Table 2 (p7):** Strongest table; matched and statistically supported.
3. **Table 3 (p7):** Correct operating-point accounting; would benefit from selected raw-text replay to isolate selection from depth in the same table.
4. **Table 4 (p8):** Strong paired local ablation; theoretical FLOPs and narrow task scope need broader measured validation.
5. **Table 5 (p8):** Valuable negative result, but table/caption omit cohort composition, exact latencies, and resampling unit.

### Appendix tables

- **Tables 6–15:** Mechanism/depth/selector controls are generally useful. Table 8 explicitly admits unequal adapter spans/parameters and missing \(j=12\) Write timing. Table 12 reports peak over \(k\), which is analysis rather than a fixed-budget selector comparison. Table 15 is point-estimate-only.
- **Tables 16–22:** Broad benchmark reporting is extensive. Cross-method rows sometimes use different backbones or prompting/training setups, but captions usually disclose this. Table 21 appropriately excludes \(j=0\) from ranking.
- **Table 23:** Directly tests bounded model-side Read over store growth; generation \(n=10\) is too small for strong quality conclusions.
- **Tables 24–26:** Excellent reproducibility tables.
- **Table 27:** Useful KV compression baselines, but not the closest reusable chunk-cache family and hardware differs from CoMem timing.
- **Tables 28–30:** Strong matched Read control and useful storage-tier microbenchmarks; still not integrated end-to-end.
- **Tables 31–35:** Helpful portability evidence but exploratory due to small samples and non-matched configurations.
- **Tables 36–37:** Cohort and LoCoMo denominator clarification is exemplary.

No arithmetic contradiction was found among the headline values. The source contains duplicate labels in unused standalone table files (`tab:slm`, `tab:crosschunk`, `tab:chunk`) because compact versions are included, but the frozen PDF has no unresolved references.

---

## 10. Questions for the authors

1. Can you provide a matched CoMem-vs-TurboRAG/CacheBlend/Cache-Craft experiment, or explain concretely why one cannot be implemented under the same backbone and evidence pack?
2. What are the full task composition, sample count, exact latencies, and bootstrap resampling unit for Table 5’s mixed diagnostic cohort?
3. Was “pre-register” backed by a timestamped protocol? If so, please provide it.
4. Does overlap-Write improve LongEval, BABILong, LongBench, and LoCoMo, and how does measured Write latency alter the reuse crossover?
5. Why does the paper not cite TurboRAG despite its offline per-document state reuse, mask/position adaptation, fine-tuning, and LongBench evaluation?
6. Can the LoCoMo judge outputs be archived and re-scored with a dated or open-weight judge?

---

## 11. Limitations and societal impact

**Assessment:** Adequately discussed.

The paper does a good job covering linear persistent-store growth, model-version coupling, update invalidation, English/lexical limitations, residual readout failure, positional extrapolation, and missing production scheduling. It also identifies tensor inversion/membership-inference risk and recommends encryption, access control, authorization-scoped retrieval, auditing, redaction before Write, and verifiable deletion.

One addition would improve the ethics discussion: because residuals are version-coupled and overlap-Write creates dependency chains, verifiable deletion and document correction require tracking derived-state lineage, not merely deleting one chunk ID. The paper gestures at edit invalidation but could connect this explicitly to privacy/deletion obligations.

**Needs ethics review:** No.

---

## 12. ARR-style ratings

### Soundness: **3.0 / 5 — Acceptable**

The central matched \(j=0\) versus \(j=12\) claim is sound and unusually well controlled, and the paper is transparent about negative results and timing boundaries. I choose 3 rather than 3.5/4 because two important claims are not yet sufficiently closed: the broad relevance of overlap-Write and competitiveness against the nearest reusable chunk-state systems.

### Excitement: **3.5 / 5 — Between Interesting and Exciting**

Treating transformer depth as an explicit cross-query reuse axis is a useful systems abstraction, and the paper contains several informative negative/diagnostic results. Excitement is moderated by the modest \(1.403\times\) isolated Read gain, decode-dominated total latency, large residual store, and incomplete closest-baseline comparison.

### Overall assessment: **2.0 / 5 — Resubmit next cycle**

I would encourage resubmission after adding a matched reusable-KV/chunk-cache baseline and broad overlap-Write validation with integrated serving measurements. These revisions appear feasible within a cycle and could move the paper to Findings or conference range. In its current form, I would not recommend acceptance because the closest prior-art comparison and the central repair’s generality are both under-supported.

### Reviewer confidence: **4.0 / 5 — Quite sure**

I read the full paper twice including appendices, checked all figures/tables, recomputed central arithmetic, verified the bibliography, and searched the nearest literature. It remains conceivable that an implementation detail in the unavailable artifact or a very recent parallel paper would change part of the assessment.

### Limitations and societal impact: **Adequate**

The limitations and ethical-risk discussion is specific and appropriately scoped. I suggest adding explicit derived-state lineage requirements for edits and verifiable deletion under overlap-Write.

### Ethical concerns: **No material unaddressed concern identified**

The work inherits privacy, disclosure, and misuse risks from long-lived memory systems; these are acknowledged with reasonable mitigations.

### Needs ethics review: **No**

### Reproducibility: **4.0 / 5**

The paper provides extensive settings, hashes, scorer definitions, seeds, and timing boundaries, and claims an anonymous code archive. Some variation or difficulty remains because the archive was not available in the permitted evidence set, the primary LoCoMo judge lacks a dated snapshot, several results rely on saved shards, and some timing/artifact values were not retained.

### Datasets

**Not applicable / no new dataset contribution.** Existing datasets and generated evaluation instances are used; the paper does not present a new reusable dataset as its contribution.

### Software: **4 / 5 — Useful, conditional on the claimed archive being complete**

The method would be a useful reference implementation and systems baseline. This score is based on the paper’s artifact description; I did not inspect the archive under the imposed evidence restriction.

### Best-paper justification

Not applicable.

### Knowledge of author identity

**1 / 5 — I do not have an educated guess.**

---

## 13. Concise author-facing recommendation

The paper has a strong controlled core and unusually good transparency. The fastest path to a materially stronger revision is:

1. add and experimentally match the closest reusable chunk-state baseline family, especially TurboRAG/CacheBlend/Cache-Craft;
2. run overlap-Write beyond the two-cell diagnostic and include measured Write/end-to-end latency;
3. integrate the serving pipeline and report tail latency/throughput;
4. correct “pre-register” unless a timestamped record exists;
5. freeze or archive the LoCoMo judging procedure and complete contamination/seed checks.

With these changes, the work could become a compelling empirical map of where persistent residual memory is preferable to raw replay and reusable KV.
