review_mode: normal
soundness: 4.0
excitement: 3.0
overall: 3.5
confidence: 4.5
reproducibility: 4.0

# Review

## 1. Summary and overall assessment

This paper studies a narrow but practically relevant design point for repeated-query
long-context inference. CoMem writes each document chunk once through the first
\(j\) transformer layers, stores one residual vector per token, retrieves a bounded
set of chunks for each query, and resumes only layers \([j{:}L)\). Its strongest
experiment is an unusually clean internal endpoint comparison: the \(j=0\) and
\(j=12\) paths use the same Qwen3-8B backbone, selected chunk IDs and order, sink,
mask, examples, and upper-layer LoRA, changing only the replay start.

The principal quantitative findings are:

1. Fixed-pack model Read decreases from **931.9 ms to 664.4 ms**
   (**1.403x**) at \(j=12\).
2. The same endpoint loses **3.12 RULER points**, from **99.19 to 96.07**,
   with paired-bootstrap 95% CI **[2.36, 3.93]**.
3. The bf16 residual store is **8,192 bytes/token**, stated and algebraically
   verified as **1/18** of full layer-wise KV for Qwen3-8B, or about **1 GiB at
   128k tokens**.
4. Measured break-even is approximately **8.4--10.9 repeated queries at 32k**
   for generations up to 128 tokens, while the retained 128k, one-token points
   are **25.8--27.6 queries**; the 32k CPU, 512-token case rises to **94 queries**.
5. At a calibrated online-latency budget, iterative-BM25 raw replay scores
   **64.78 versus 53.22** for CoMem, whereas frozen-BGE replay scores
   **54.22 versus 53.22**, a non-significant \(-1.00\)-point CoMem-minus-replay
   difference with CI **[-4.67, 2.67]**.
6. On the focused paired multikey diagnostic, contextual lower-layer writing
   raises **92.5 to 100.0**; a deployable 32-token overlap raises **92.5 to
   98.5**, with a \(+6.0\)-point CI **[3.0, 9.5]**.
7. The reported **64.9x** 128k dense-to-bounded online-prefill result is
   correctly labeled as combining selection and depth reuse rather than as the
   causal depth effect.

I view the paper as a careful measurement/negative-results paper rather than a
state-of-the-art systems paper. Its central same-evidence endpoint is sound,
well scoped, and reported with commendable candor. The paper repeatedly avoids
claiming quality-preserving acceleration or superiority over raw-text retrieval
and modular KV caches. The main limitation is that the closest reusable-context
systems are only taxonomically discussed, so the work establishes an internal
depth-reuse operating point but not its competitiveness. A second limitation is
that the learned interface's headline quality is conditional on one clean
batch-8 training run. I therefore place the paper between Findings and ACL main:
strong enough for a positive Findings-level assessment, but not yet a clear
main-conference systems result.

## 2. Claims and evidence map

| ID | Paper claim | Main evidence | Assessment |
|---|---|---|---|
| C1 | CoMem makes transformer depth a cross-query reuse axis by storing one depth-\(j\) residual per token and executing only the suffix layers after retrieval. | Figure 1; Section 4, pp. 4--5; inference configuration in Table 25. | Clearly defined. The half-open layer convention, write/read positions, causal mask, selected-pack ordering, and \(j=0\) endpoint are specified. |
| C2 | Residual storage is much smaller than full layer-wise KV while model-side Read length is bounded by the selected pack. | Equation 1; Section 4; Tables 24, 25, and 32. | Algebra is correct for the stated GQA dimensions: \(32/(2\cdot36\cdot8)=1/18\), yielding 8 KiB/token versus 144 KiB/token. The paper appropriately notes that store, index, and selector cost are not bounded. |
| C3 | Reusing the first 12 layers yields a 1.403x isolated Read speedup at a statistically resolved 3.12-point RULER cost. | Table 2, p. 6; Tables 30 and 35, pp. 21 and 23; Appendix A.4. | This is the strongest and best-controlled claim. Same examples, packs, order, LoRA, and replay harness are documented; the result is not mislabeled as end-to-end acceleration. |
| C4 | Self-distillation is needed to make the split residual interface usable. | Equation 2; Tables 9--12, pp. 12--13; Table 26. | Strong same-\(j\) adapter-on/off evidence on RULER, LoCoMo, BABILong, and an HCache-style path. The top-64 truncated objective is disclosed, but its retained teacher mass was not logged. |
| C5 | The persistent store pays off only with repeated reuse, with the reported workload-specific break-even values. | Table 3, p. 6; Appendix A.4. | Credibly bounded and carefully labeled. Results remain hardware-, generation-length-, storage-, and timing-boundary-specific. Only \(G=1\) is retained at 128k. |
| C6 | Equal-latency quality conclusions are selector-dependent and CoMem wins neither reported aggregate. | Table 4, p. 6; protocol-complete Table 8, p. 12. | The reported numbers support the literal conclusion. However, the BGE phase is not a fully crossed selector experiment, and the heterogeneous macro's uncertainty treatment is weaker than the central paired endpoint. |
| C7 | Much of one multikey failure can be localized to missing lower-layer document context; overlap writing repairs most of that local gap. | Tables 5--6, pp. 6--7; Table 31, p. 21. | Supported on the stated paired synthetic cohort. The paper correctly avoids generalizing this to natural tasks. The continuous-prefix oracle is a fidelity ceiling rather than a deployable method. |
| C8 | Natural-task behavior is mixed and does not support a universal quality claim. | Tables 18--23, pp. 15--17; Section 5.1. | Supported. LongEval shows a large selected-pack depth cost, LongBench is nearly tied with KV-Direct, BABILong is task dependent, and LoCoMo favors CoMem under the chosen judge. The paper labels these as scope checks. |
| C9 | Bounded model Read can coexist with increasing corpus size, but selection/index and external-store costs still grow. | Table 24, p. 19; Table 32, p. 21. | Supported by fixed-read synthetic scaling and storage-tier microbenchmarks. Sample sizes for some scaling cells are small, and these are stress diagnostics rather than workload traces. |
| C10 | The implementation concept ports beyond one dense 8B model, including an 80-layer sparse MoE. | Tables 33--34, p. 22; Appendix A.7. | The exact split-forward self-test and Hy3 diagnostics support implementation portability. They do not constitute matched replication of the flagship quality/latency frontier. |
| C11 | The paper does not establish superiority over PIC, chunk-KV repair, learned modular caches, or generic RAG. | Introduction, Related Work, Conclusion, and Limitations. | Accurately and repeatedly stated. This restraint improves soundness but also defines the main limitation in novelty and practical impact. |

## 3. Strengths

### S1. The central causal comparison is unusually well controlled

The paper's most persuasive contribution is not the largest speedup but the
matched \(j=0\rightarrow12\) endpoint. Table 2 and Appendix Table 30 hold fixed
the selected evidence, order, examples, mask, sink, adapter, and hardware
harness. This directly separates depth reuse from bounded retrieval. The authors
also report process-level latency dispersion and paired outcome statistics rather
than only aggregate means.

### S2. The paper distinguishes timing boundaries and negative outcomes

Selected-pack Read, store-ready online prefill, Write-inclusive operation,
external I/O, decode, and index/selection costs are not silently conflated. The
paper explicitly states that 64.9x is not the depth-only effect, that 1.403x is
not end-to-end acceleration, that the 3.12-point loss is real, and that CoMem
wins neither equal-latency aggregate. This is exemplary systems reporting.

### S3. The failure analysis is more informative than a leaderboard table

The continuous-prefix control, context-by-position \(2\times2\) design,
cross-chunk-attention ablation, overlap-write intervention, selector diagnostics,
and retrieval-hit/readout decomposition jointly identify where the interface
fails. In particular, the paper does not over-interpret the exact split
reconstruction control as a deployable cache or a single-factor explanation.

### S4. The appendices expose substantial protocol detail

The paper provides task supports, sample counts, seeds, generation limits,
official scorers, adapter SHA-256, backbone revision, optimizer schedule, exact
mask semantics, software versions, hardware, timing repetitions, bootstrap
units, LoCoMo denominators, and judge parsing behavior. Tables 25--27 are
especially useful. The RULER cohort-A/cohort-B distinction is explicit rather
than hidden.

### S5. Scope and limitations are stated with unusual precision

The authors acknowledge the one-backbone emphasis, lexical selector, English
scope, incomplete contamination audits, mutable GPT-4o judge, one clean flagship
training run, absence of matched modular-cache baselines, large residual store,
model-version coupling, edit invalidation, missing production tail latency, and
the synthetic nature of the overlap repair. These disclosures make the claims
substantially easier to trust.

### S6. Ethics and storage risks are meaningfully discussed

The ethics section treats residual tensors as sensitive representations rather
than assuming that non-readable storage is safe. It discusses inversion,
membership inference, access control, tenant isolation, deletion, encryption,
source filtering, and inherited generation risks.

## 4. Weaknesses

### W1. No matched comparison to the nearest reusable-context systems

- **Severity:** Major
- **Location:** Related Work, p. 3, lines 150--156; Limitations, p. 7,
  lines 434--441.
- **Exact quote (17 words):** “We lack a matched same-backbone implementation
  of these systems, so Table 1 supports taxonomy rather than superiority.”
- **Problem:** EPIC/APE-style position-independent caching, CacheBlend/TurboRAG
  chunk-KV repair, and learned modular objects such as KV Packet or Cartridges
  target nearly the same repeated-document serving workload. The paper compares
  CoMem causally only against its own \(j=0\) endpoint.
- **Impact:** The 1.403x/3.12-point internal trade-off is credible, but a reader
  cannot determine whether storing one residual at a split depth is preferable
  to storing/reusing KV with repair, learned adapters, or another modular object
  under the same storage and latency boundary. This substantially limits both
  novelty and deployment relevance.
- **Sufficient remedy:** Implement at least one strong PIC/chunk-KV method and
  one learned modular-cache method on the same Qwen3-8B checkpoint, selected
  chunk IDs, hardware, storage tier, and quality cohort. Report persistent
  bytes, one-time write/compile cost, TTFT, decode, throughput, quality, and
  repeated-query crossover. If such implementation is infeasible, recast the
  submission explicitly as an analysis/measurement paper and further reduce
  system-comparative language.

### W2. Training-run uncertainty is not measured for the exact headline adapter

- **Severity:** Major
- **Location:** Limitations, p. 7, lines 424--433; Appendix A.5 and Table 28,
  pp. 20--21.
- **Exact quote (7 words):** “The flagship is one batch-8 training run.”
- **Problem:** The paired bootstrap and McNemar analyses quantify evaluation
  uncertainty conditional on one trained adapter. The two additional adapters
  use effective batch 3, reduced supports, and do not cover the exact 15-cell
  RULER-B, LoCoMo, or LongEval headlines.
- **Impact:** The reported 96.07 operating point and the exact 3.12-point
  quality cost may partly reflect training stochasticity or the chosen run.
  This matters because the paper's main practical conclusion is an operating
  frontier induced by a learned interface.
- **Sufficient remedy:** Train at least three independent \(j=12\) adapters
  with identical effective batch, examples seen, optimizer, and selection
  protocol. Evaluate every seed on the exact RULER-B cohort plus at least one
  natural benchmark, and report run-level means/intervals and any checkpoint
  selection rule. Latency need not be repeated for every seed if architecture
  and kernels are identical, but quality should be.

### W3. The equal-latency selector result is not a fully crossed or
dependence-aware experiment

- **Severity:** Major
- **Location:** Table 8, p. 12, especially the Quality cohort, Frozen-BGE
  result, and Statistical unit rows.
- **Exact quote (9 words):** “it does not resample task cells or LoCoMo
  conversations.”
- **Problem:** The BGE comparison changes the replay selector while leaving
  CoMem on iterative BM25, so it demonstrates sensitivity of two deployment
  configurations rather than isolating selector effects. The macro gives equal
  weight to nine heterogeneous cells, uses the first 100 LoCoMo items from a
  single conversation, and bootstraps 900 examples IID without resampling task
  cells or conversations.
- **Impact:** The paper's literal statement that the reported aggregate changes
  with selector is true, but the confidence interval and broader deployment
  interpretation can be dominated by cell composition, one conversation, or
  unmodeled task-level dependence. This result is prominent in the abstract.
- **Sufficient remedy:** Run a fully crossed \(2\times2\) comparison:
  BM25/BGE selection for both \(j=0\) replay and CoMem, calibrated under the
  same latency rule. Sample LoCoMo across conversations, report every cell, and
  add a hierarchical bootstrap over task cells with conversation-cluster
  resampling for LoCoMo. A macro sensitivity analysis over reasonable cell
  weights would further clarify robustness.

### W4. The truncated distillation target is insufficiently characterized

- **Severity:** Minor
- **Location:** Section 4, p. 4, lines 270--279; Table 26, p. 20.
- **Exact quote (10 words):** “We did not retain the teacher mass captured by
  \(S_t\).”
- **Problem:** Teacher and student distributions are renormalized on the
  teacher's top-64 support, discarding the remaining vocabulary mass. Without
  retained-mass statistics, it is unclear how close this objective is to
  full-distribution distillation across token types and contexts.
- **Impact:** This does not invalidate the empirical adapter ablation, but it
  weakens mechanistic interpretation and makes the exact training recipe harder
  to diagnose or transfer.
- **Sufficient remedy:** Report the distribution of teacher probability mass
  captured by top-64 and compare at least two support sizes or a full-vocabulary
  KL on a representative subset. An ablation of the asymmetric KL weights would
  also be useful.

### W5. The proposed overlap repair is not validated on natural tasks or an
integrated serving frontier

- **Severity:** Minor
- **Location:** Table 6, p. 7; Section 5.3; Limitations, pp. 7--8.
- **Exact quote (6 words):** “No natural-task overlap result is claimed.”
- **Problem:** The 32-token overlap is the paper's only deployable repair for
  the diagnosed write-context failure, but it is tested only on one paired
  synthetic multikey cohort. Its measured wall-clock Write overhead is not
  integrated into the repeated-query crossover.
- **Impact:** The result is a valuable diagnosis, but it does not yet justify
  using overlap writing as a default deployment choice.
- **Sufficient remedy:** Evaluate \(w=0\) versus \(w=32\) on at least
  LongEval, LoCoMo, BABILong, and one LongBench subset, measure actual write
  time and edit invalidation, and recompute the end-to-end crossover.

### W6. Production-facing latency evidence lacks end-to-end tails

- **Severity:** Minor
- **Location:** Table 8, p. 12; Limitations, p. 8, lines 507--509.
- **Exact quote (9 words):** “no directly measured end-to-end TTFT p95 is
  reported.”
- **Problem:** The principal timings are medians from single-query harnesses.
  Store microbenchmarks include limited concurrency, but the full selector,
  fetch, model, and decode path is not evaluated under concurrent load.
- **Impact:** Median component measurements support the internal causal claim,
  but not production throughput, queueing behavior, or tail-latency claims.
- **Sufficient remedy:** Add a concurrent end-to-end serving experiment with
  p50/p95 TTFT, output-token latency, QPS, and memory across at least two
  concurrency levels and two storage tiers.

### W7. Total experimental compute is not fully accounted for

- **Severity:** Minor
- **Location:** Appendix A.4, p. 18, lines 951--959.
- **Exact quote (15 words):** “Total GPU-hours across preliminary probes,
  failed runs, baseline generation, and all ablations were not consistently
  logged.”
- **Problem:** The final adapter's approximately 2.9 H20 GPU-hours are reported,
  but the total cost of the broad experimental program is unknown.
- **Impact:** This limits cost reproducibility and environmental accounting,
  though it does not affect the reported central measurements.
- **Sufficient remedy:** Release a run manifest with hardware, wall time,
  status, and purpose for every retained baseline and ablation, including failed
  runs where available.

## 5. Questions for the authors

1. Can the authors provide a fully crossed BM25/BGE selector experiment for
   both replay and CoMem, rather than changing only replay in the BGE phase?
2. How much does the exact RULER-B \(j=12\) score vary across clean,
   same-effective-batch training seeds?
3. Why was the equal-latency LoCoMo cell restricted to the first 100 stored
   items, all from conversation 0, rather than a conversation-stratified sample?
4. What fraction of teacher probability mass does the top-64 support retain,
   and does increasing the support materially change adapter quality?
5. Can the measured wall-clock cost of \(w=32\) overlap writing be integrated
   into the same break-even harness used for Table 3?
6. Which nearest modular-cache implementation do the authors consider most
   feasible for a same-backbone comparison, and what engineering obstacle
   currently prevents it?
7. Is the claimed anonymous artifact available with a stable submission link?
   Artifact execution and completeness were **Unverifiable** under the
   permitted frozen review materials.

## 6. Method, experiment, statistics, scope, and reproducibility audit

| Audit item | Finding |
|---|---|
| Method definition | Write/Select/Read semantics, layer indexing, positions, ordering, mask, and generation cache behavior are adequately specified. |
| Storage formula | Equation 1 is dimensionally and numerically correct for Qwen3-8B: one bf16 \(d=4096\) residual is 8,192 bytes/token; full 36-layer GQA KV is 147,456 bytes/token. |
| Distillation formula | Equation 2 is mathematically defined and the absence of CE is explicit. The top-64 renormalization is an approximation whose captured teacher mass is unreported. |
| Minimal causal experiment | Table 2/Table 30 is sufficient to support the narrow depth-reuse claim because evidence, examples, ordering, LoRA, and pack are matched. |
| Internal controls | Strong: \(j=0\), continuous-prefix exact partition, adapter on/off, multiple split depths, block-diagonal attention, context/position factorial, overlap, selector, chunk size, and store scaling. |
| External baselines | Broad descriptive coverage (KV-Direct, InfLLM, StreamingLLM, MemoryLLM, SnapKV, PyramidKV, LLoCO), but nearest reusable-document caches are not matched on backbone/hardware/training/timing. |
| Benchmark coverage | RULER, BABILong, LongEval, six LongBench QA datasets, and all 1,986 LoCoMo questions provide breadth. Most conclusions still center on synthetic retrieval tasks and one English 8B backbone. |
| Metrics | Official benchmark scorers are named. LoCoMo's semantic judge is primary; deterministic F1/substring diagnostics are also reported. Prompt sensitivity is explicitly audited. |
| Sample sizes | Usually \(n=100\) per synthetic cell; RULER central endpoint has 1,500 paired examples. Some store-scaling and Hy3 diagnostics use \(n=10\), \(20\), or \(50\), appropriately labeled as stress/diagnostic evidence. |
| Seeds | Benchmark-generation seeds are reported. The flagship adapter has one clean batch-8 run; two robustness runs confound seed with effective batch and reduced evaluation support. |
| Statistical tests | Central RULER endpoint has paired bootstrap and exact McNemar; LoCoMo includes item and conversation-cluster bootstrap; overlap has paired CI. Equal-latency pooling should be hierarchical rather than IID across all examples. |
| Latency statistics | Three independent processes with warmups/repetitions are used for key Read/TTFT measurements. Most reported system numbers are medians; full end-to-end p95 under concurrency is missing. |
| Scope | The workload contract is explicit: stable corpus, open-weight model, repeated reuse, and nearby persistent states. One-off, frequently edited, low-reuse, multilingual, multimodal, code, and dynamic-update settings are out of scope. |
| Compute | Final training run: 8 H20 GPUs, about 22 minutes, approximately 2.9 H20 GPU-hours. Total experimental GPU-hours and training peak memory are incomplete. |
| Data contamination | A PG-19/InfiniteBench overlap issue was detected and the affected comparison removed. Equivalent audits were not completed for all natural benchmarks, so natural results are correctly labeled scope checks. |
| Reproducibility | Configuration detail is strong: hashes, revisions, optimizer, data construction, metrics, environment, seeds, and timing protocol are reported. The claimed archive itself and runnable reproduction are **Unverifiable** from the allowed frozen PDF/source. |

## 7. Figure and table audit

### Figures

- **Figure 1:** Correctly depicts independent chunk writes, bounded selection,
  matched \(j=0\) replay, suffix execution, and overlap writing. It is
  information-dense and small at normal zoom but internally consistent with the
  method text.
- **Figure 2:** Appropriately presented as motivation rather than validation.
  The caption warns that both probe and native-readout knees are
  protocol-dependent. Exact plotted uncertainty is deferred to Appendix A.1;
  this is acceptable, though the plot is too small for detailed quantitative
  reading.

### Tables

| Table | Audit result |
|---|---|
| 1 | Taxonomy of persistent objects and post-selection work is useful and explicitly non-ranking. |
| 2 | Central matched endpoint; numbers agree with abstract, prose, and Table 30. |
| 3 | Break-even values agree with abstract/prose; timing boundary and missing 128k generation grid are disclosed. |
| 4 | Equal-latency numbers agree with abstract; selector and sign convention are clear, but design/statistical caveats motivate W3. |
| 5 | Context-by-position \(2\times2\) values and interaction arithmetic are consistent. |
| 6 | Overlap results, CI, storage invariance, and theoretical FLOP ratios are clearly labeled. |
| 7 | 64.9x online-prefill number is explicitly separated from depth-only and end-to-end effects. |
| 8 | Exceptionally detailed protocol table; exposes the single-conversation LoCoMo cell and IID bootstrap limitation. |
| 9 | Same-pack frozen-depth and adapter comparison supports interface degradation and adapter utility. |
| 10 | Multi-depth deployment curve is not falsely presented as compute-matched; missing \(j=12\) Write value is disclosed. |
| 11 | Clean same-\(j=12\) RULER adapter on/off ablation. |
| 12 | HCache-style transfer diagnostic is appropriately not called a head-to-head system comparison. |
| 13 | Selector sweep is useful but reports the peak over \(k\), so it should not be read as a fixed-budget deployment comparison; caption states this. |
| 14 | Dense-retriever diagnostic separates recall from raw-reader quality; not a trained-retriever comparison. |
| 15 | Equal nominal token budget versus StreamingLLM; examples are not paired and this is disclosed. |
| 16 | Cross-chunk attention ablation supports the importance of joint recomputation for multi-fact tasks. |
| 17 | Chunk-size latency/accuracy trade-off is clearly labeled as point estimates rather than multi-seed evidence. |
| 18 | Full RULER Cohort-A grid is complete; Cohort A/B distinction is explicit. |
| 19 | LongEval grid is complete; unextended 64k/128k KV-Direct cells are appropriately caveated. |
| 20 | BABILong task-by-length grid is complete and shows task-dependent compression cost. |
| 21 | LongBench grid is complete; LLoCO's three-task macro is explicitly non-comparable to the six-task macro. |
| 22 | YaRN scaling table distinguishes native and extrapolated settings and reveals a variable-tracking tax. |
| 23 | LoCoMo comparison reports semantic and lexical metrics, denominators, paired item CI, and conversation-cluster CI. |
| 24 | Store scaling demonstrates bounded model Read but growing BM25 lookup; small generation sample sizes are stated. |
| 25 | Inference configuration is sufficiently detailed for implementation. |
| 26 | Training configuration includes modules, hash, objective, data, optimizer, steps, and hardware; retained top-64 mass and peak memory are missing. |
| 27 | Benchmark supports, generation limits, scorers, aggregation, and seeds are specified. |
| 28 | Robustness table honestly labels the seed/effective-batch confound and reduced support. |
| 29 | SnapKV/PyramidKV retained-budget comparison is useful but not a matched reusable-context systems comparison; hardware mismatch is disclosed. |
| 30 | Repeats and substantiates the central quality/latency endpoint with dispersion. |
| 31 | Continuous-prefix oracle cleanly verifies exact partition and bounds the chunk-local write loss. |
| 32 | External-store tiers quantify fetch/H2D/QPS and OOM boundaries; intentionally excludes selection/model inference. |
| 33 | Hy3 self-distillation result supports transfer of the readout-repair mechanism on 16 PG-19 documents. |
| 34 | Hy3 bounded-read result supports implementation portability through 256k but uses \(n=50\) synthetic needle tasks. |
| 35 | Exact RULER-B cells reproduce the 96.0667 macro used in the central comparison. |
| 36 | LoCoMo category denominators sum to 1,986 and clarify the mixed API/local judging protocol. |

No figure/table value contradiction was found in the inspected frozen PDF and
source. Several tables use very small type, and p. 16 is unusually sparse due to
float placement, but all content remained readable when zoomed.

## 8. Abstract-number consistency audit

The following abstract quantities were mechanically traced to the body/tables:

| Abstract quantity | Matching location | Result |
|---|---|---|
| 931.9 to 664.4 ms; 1.403x | Tables 2 and 30 | Match |
| 99.19 to 96.07; gap 3.12; CI [2.36, 3.93] | Tables 2, 30, 31, and 35 | Match |
| Rank-32 adapter | Section 4; Table 26 | Match |
| 8 KiB/token | Equation 1; Tables 2, 25, and Appendix A.4 | Match |
| 8--11 queries at 32k through \(G=128\) | Table 3 | Match after rounding the 8.4--10.9 cells |
| 25.8--27.6 queries at 128k, \(G=1\) | Table 3 | Match |
| BM25 gap 11.56 | Tables 4 and 8 | Match |
| BGE difference -1.00; CI [-4.67, 2.67] | Tables 4 and 8 | Match |
| 92.5 to 100.0 diagnostic | Tables 5 and 31 | Match |
| 32-token overlap reaches 98.5 | Table 6 | Match |

## 9. Citation audit

### 9.1 `main.bbl` completeness

- `main.bbl` contains **43 entries**.
- All **43 entries are cited** in the frozen source.
- There are no cited keys missing from `main.bbl` and no uncited `main.bbl`
  entries.
- Every entry contains an author or group author, year, title, and publication
  or preprint field.

### 9.2 Entry-by-entry verification

“Verified” means that title/identity and a primary publisher, proceedings,
OpenReview, Hugging Face, or arXiv record were reachable during the finite
audit. “Unverifiable” means access/rate limiting prevented independent online
confirmation; it does not by itself imply an incorrect citation.

| Key | Entry | Status |
|---|---|---|
| `cachecraft` | Cache-Craft | **Unverifiable online**: ACM/DOI endpoint returned 403; DOI and BBL metadata are present. |
| `longbench` | LongBench | Verified via ACL Anthology. |
| `pyramidkv` | PyramidKV | Title/arXiv identity verified; exact venue/year formatting not independently confirmed. |
| `kvpacket` | KV Packet | Verified via arXiv 2604.13226. |
| `cartridgesbase` | Cartridges | Title/arXiv identity verified; exact ICLR volume/page metadata not independently confirmed. |
| `hcache` | HCache | **Unverifiable online**: ACM/DOI endpoint returned 403; DOI and BBL metadata are present. |
| `llama3` | The Llama 3 Herd of Models | Verified via arXiv 2407.21783. |
| `cartridges` | Cartridges at Scale | Verified via arXiv 2606.04557; post-novelty-cutoff and excluded from novelty judgment. |
| `distillation` | Distilling the Knowledge in a Neural Network | Verified via arXiv 1503.02531. |
| `ruler` | RULER | Verified via arXiv/CoLM record. |
| `lora` | LoRA | Verified via arXiv/ICLR record. |
| `epic` | EPIC | Verified via arXiv 2410.15332. |
| `ragcache` | RAGCache | **Unverifiable online**: ACM/DOI endpoint returned 403; DOI and BBL metadata are present. |
| `babilong` | BABILong | Verified via proceedings record. |
| `rag` | Retrieval-Augmented Generation | Verified via NeurIPS proceedings. |
| `longchat` | LongChat context-length report | Verified via LMSYS page. |
| `snapkv` | SnapKV | Verified via proceedings record. |
| `ilre` | ILRe | Verified via arXiv 2508.17892. |
| `readonce` | ReadOnce Transformers | Verified via ACL Anthology. |
| `minicache` | MiniCache | Verified via proceedings record. |
| `turborag` | TurboRAG | Verified via ACL Anthology. |
| `locomo` | LoCoMo | Verified via ACL Anthology. |
| `xccache` | XC-Cache | Verified via ACL Anthology. |
| `kvdirect` | The Residual Stream Is All You Need / KV-Direct | Verified via arXiv 2603.19664. |
| `pg19` | Compressive Transformers / PG-19 | Verified via arXiv 1911.05507. |
| `bm25` | Probabilistic Relevance Framework: BM25 | **Unverifiable online**: DOI target returned 403; DOI and BBL metadata are present. |
| `embeddingrecycling` | Embedding Recycling | Verified via ACL Anthology. |
| `gemfilter` | GemFilter | Verified via ACL Anthology. |
| `reform` | REFORM | Verified via arXiv 2506.01215. |
| `lloco` | LLoCO | Verified via ACL Anthology. |
| `hunyuan` | Hy3 | Verified via the cited Hugging Face model page. |
| `fusionrag` | Fusion RAG Cache | Verified via arXiv 2601.12904. |
| `mepic` | MEPIC | Verified via arXiv 2512.16822. |
| `longmem` | LongMem | Verified via NeurIPS proceedings. |
| `memoryllm` | MemoryLLM | Verified via PMLR. |
| `infllm` | InfLLM | Verified via proceedings record. |
| `streamingllm` | StreamingLLM | Verified via arXiv/ICLR record. |
| `sempic` | SemPIC | Verified via arXiv 2607.28069; post-novelty-cutoff and excluded from novelty judgment. |
| `xu2024retrievallong` | Retrieval Meets Long Context LLMs | Verified via OpenReview. |
| `qwen3` | Qwen3 Technical Report | Verified via arXiv 2505.09388. |
| `ape` | APE | Verified via arXiv 2502.05431. |
| `cacheblend` | CacheBlend | **Unverifiable online**: ACM endpoint returned 403; DOI and BBL metadata are present. |
| `h2o` | H2O | Verified via NeurIPS proceedings. |

### 9.3 Citation-to-claim matching

| Paper statement | Cited work(s) | Match assessment |
|---|---|---|
| Raw-text RAG retrieves evidence and then processes concatenated text through the model. | Lewis et al. (`rag`) | Supported at the architectural level. The “all layers are recomputed” part is a direct consequence of ordinary decoder inference rather than a quoted contribution of RAG. |
| ReadOnce caches reusable intermediate text representations and adapts later processing. | Lin et al. (`readonce`) | Supported; this is an important conceptual precursor and appropriately cited. |
| Embedding Recycling caches intermediate activations for reuse by later computation. | Saad-Falcon et al. (`embeddingrecycling`) | Supported directly by the paper's method description. |
| EPIC enables position-independent reuse of cached KV while repairing boundary/attention effects. | Hu et al. (`epic`) | Supported directly. |
| APE independently encodes/caches contexts and realigns attention for composition. | Yang et al. (`ape`) | Supported directly. |
| KV Packet uses context-independent document KV with learned adapters/self-distillation. | Chen et al. (`kvpacket`) | Supported directly; it is among the closest concurrent learned-object comparisons. |
| KV-Direct uses residual checkpoints to reconstruct layer-wise KV. | Qasim et al. (`kvdirect`) | Supported directly; it is the closest residual-object precedent found in the concurrent window. |
| ILRe encodes only to an intermediate layer, retrieves/selects tokens, and then performs later computation. | Liang et al. (`ilre`) | Supported at the pipeline level. It is not the same cross-query persistent document object, so the paper's distinction is reasonable. |

No citation-claim mismatch severe enough to affect correctness was found.

## 10. Novelty analysis with cutoff and three-month rule

The requested novelty cutoff is **2026-05-04**. I applied a three-month
concurrency window beginning **2026-02-04**: work first appearing between
2026-02-04 and 2026-05-04 is treated as concurrent context rather than as a
strong basis for penalizing novelty. Work after 2026-05-04 was excluded from
the novelty judgment even if cited in the August 2026 manuscript.

### Search 1: reusable intermediate representations

- Query family: `"reusable representations" transformer`,
  `"intermediate representations" reuse`.
- Closest pre-window works: **ReadOnce Transformers** and
  **Embedding Recycling**.
- Comparison: both establish that intermediate text representations/activations
  can be cached and later computation adapted around them. CoMem is therefore
  not novel merely for caching an intermediate state. Its narrower difference is
  a query-selected bounded pack, one residual per token at a tunable split,
  direct suffix execution, and explicit repeated-query serving accounting.

### Search 2: position-independent/modular document caching

- Query family: `"position-independent caching" LLM`,
  modular document KV caching.
- Closest works: **EPIC**, **APE**, **MEPIC**, CacheBlend/TurboRAG, and the
  Cartridges line.
- Comparison: these works already address reusable document/context objects
  under changing prefixes and composition. They usually store layer-wise KV and
  repair positions/boundaries or learn modular KV objects. CoMem's object and
  depth/storage trade-off differ, but the repeated-document workload and
  composition problem are established prior art.

### Search 3: intermediate-layer retrieval/compression

- Query family: `"intermediate layer retrieval" context compression`,
  early-layer long-context retrieval.
- Closest works: **ILRe**, **GemFilter**, and **REFORM**.
- Comparison: these methods use early/intermediate computation to select or
  gather a smaller token set before further processing. CoMem adds
  cross-query persistence of the selected token states and a controlled
  \(j=0\) endpoint, but “use an intermediate layer to reduce later
  long-context work” is not itself new.

### Search 4: residual-stream cache

- Query family: `"residual stream" "KV cache"`,
  `"single residual per token" transformer cache`.
- Closest work: **KV-Direct / The Residual Stream Is All You Need**, first
  appearing 2026-03-20.
- Three-month treatment: concurrent under the requested rule.
- Comparison: KV-Direct establishes that one residual can reconstruct layer-wise
  KV and uses residual checkpoints for bounded-memory inference. CoMem instead
  persists a document residual at one split depth, retrieves selected chunks
  across queries, and directly executes the suffix. This is a meaningful
  distinction, but the stored residual object is concurrent rather than wholly
  unique.

### Search 5: learned context-independent reusable cache

- Query family: `"context-independent KV" cache`, learned reusable document
  cache with distillation.
- Closest work: **KV Packet**, first appearing 2026-04-14.
- Three-month treatment: concurrent under the requested rule.
- Comparison: KV Packet is especially close in workload and in using
  self-distillation/adapters to make independently cached document state usable.
  It stores context-independent KV packets and aims for little/no document
  recomputation; CoMem stores only a single split residual and trades persistent
  bytes for online suffix execution.

### Novelty conclusion

The component ideas—reusable intermediate representations, modular document
caches, early-layer selection, residual checkpoints, and adapter-based repair—
all have clear precedents. The paper's defensible novelty is narrower:

1. treating split depth as an explicit repeated-query systems axis;
2. storing exactly one residual per token at that split;
3. performing a matched same-evidence \(j=0\rightarrow12\) measurement;
4. jointly accounting for quality loss, Write amortization, storage placement,
   selection, and bounded-read failure modes; and
5. providing a controlled write-context diagnosis.

This is a useful empirical contribution, but it is incremental in mechanism and
not yet competitively established against the nearest cache systems. That
supports an **Excitement score of 3.0**, separate from the stronger soundness
assessment.

## 11. Desk, anonymity, style, ethics, and integrity checks

| Check | Result |
|---|---|
| Frozen version | Reviewed from scratch using only `v5_20260804_003238.pdf`, its corresponding frozen source, and `NORMAL_REVIEW_TEMPLATE.md`. |
| Page limit | PDF has 23 pages. Main paper content, including Limitations and Ethics, occupies pp. 1--8; references continue on pp. 8--10; appendix begins on p. 11. This is consistent with an eight-page ACL main-paper limit excluding references/appendices. |
| Limitations | Present and substantial on pp. 7--8. |
| Anonymity | Author is “Anonymous ACL submission”; no author affiliation, email, repository owner, or obvious identity leak was found in visible paper/source content. |
| Official style | Uses `\usepackage[review]{acl}` at 11 pt with line numbers. No custom margin or page-size override was found. |
| Unresolved references/citations | Mechanical label/ref and citation-key checks found no unresolved references, duplicate labels, missing bibliography keys, or orphan `main.bbl` entries. |
| Placeholders | Mechanical search found no `TODO`, `FIXME`, `XXX`, `TBD`, or `??`. |
| Hidden/reviewer manipulation | No white text, hidden reviewer instruction, score request, prompt injection, or instruction-like manipulation was found. Paper text was treated only as submission data. |
| Abstract consistency | All inspected headline numbers map consistently to the main or appendix tables; see Section 8. |
| Readability | Generally professional. Figure 1 and several resized tables are dense/small, and p. 16 is sparse because of float placement, but this is a presentation issue rather than a desk-reject condition. |
| Ethics | Risks of hallucination, sensitive retrieval, residual inversion/membership inference, access control, deletion, multi-tenant isolation, and energy are discussed. No new human-subject collection is reported. |
| Artifact | The manuscript claims an anonymous archive with adapter, scripts, hashes/shards, judge records, and timings. Availability and execution were **Unverifiable** within the permitted frozen materials. |

## 12. Score rationale

- **Soundness: 4.0/5.0.** The narrow central endpoint is carefully controlled,
  formulas and numerical accounting are coherent, and negative findings are
  reported honestly. Soundness is held below 4.5 by the single clean training
  run and the weaker equal-latency aggregate design.
- **Excitement: 3.0/5.0.** The measurement and diagnosis are useful, but the
  underlying ingredients have substantial prior art and no matched nearest
  modular-cache baseline establishes competitiveness.
- **Overall: 3.5/5.0.** Strong Findings / borderline ACL main. The paper is
  substantially better than a superficial speedup paper, but the missing
  matched nearest-system comparison prevents a confident main-conference score
  of 4.0.
- **Confidence: 4.5/5.0.** Based on two full readings including appendices,
  claims mapping, formula and arithmetic checks, all-figure/table inspection,
  bibliography coverage, targeted claim matching, and finite novelty searches.
  Some exact bibliography metadata and the artifact remained Unverifiable
  because of access limits and the frozen-material constraint.
- **Reproducibility: 4.0/5.0.** The written protocol is unusually complete,
  with hashes, revisions, configs, seeds, metrics, and hardware. The score is
  below 4.5 because the artifact was not executable in this review, the exact
  headline lacks clean multi-seed replication, and total compute accounting is
  incomplete.

## 13. Review-process self-check

- Completed two passes over the full 23-page PDF, including both appendices.
- Built and checked an explicit C1--C11 claims/evidence map.
- Inspected both figures and Tables 1--36.
- Checked the eight-page boundary, Limitations, anonymity, ACL review style,
  unresolved labels/citations, placeholders, abstract-number consistency,
  hidden text/reviewer manipulation, and ethics.
- Audited all 43 `main.bbl` entries and verified 43/43 citation-key coverage;
  access-limited records are labeled **Unverifiable** rather than guessed.
- Performed eight citation-to-claim checks.
- Performed five bounded novelty search families using the 2026-05-04 cutoff
  and 2026-02-04 three-month boundary.
- Audited method formulas, the minimal causal experiment, baselines, benchmarks,
  metrics, seeds, statistics, scope, compute, and reproducibility.
- Mechanically confirmed that all weakness quotations appear verbatim in the
  frozen source/PDF and are at most 25 words.
- Mechanically checked “missing X” assertions against the paper: the paper
  explicitly confirms the absence of matched nearest-system baselines, clean
  headline multi-seed variance, natural-task overlap results, end-to-end TTFT
  p95, and complete total-compute logging.

