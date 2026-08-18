review_mode: normal
soundness: 4.0
excitement: 3.5
overall: 3.5
confidence: 4.5
reproducibility: 3.5

# Summary

This paper proposes CoMem, a reusable-context interface for decoder-only
transformers. A document is written once through layers \([0,j)\), one residual
vector \(h_j\) is stored per token, a bounded set of chunks is selected for each
query, and inference resumes through layers \([j,L)\). The main methodological
contribution is not merely caching a residual state, but treating split depth as
an explicit serving variable and providing a matched \(j=0\) raw-text replay
endpoint that holds the selected evidence pack and most of the read pathway
fixed.

The central Qwen3-8B comparison reports a selected-pack Read reduction from
931.9 ms to 664.4 ms (1.403x) at a 3.12-point RULER-B cost. The paper then
separates this depth-only measurement from (i) a store-ready 128k
bounded-selection comparison (64.9x prefill), (ii) a separate Write-inclusive
same-adapter pipeline (2.74x), and (iii) measured repeated-query break-even
points. A continuous-prefix oracle and context/position ablations attribute
most of the tested degradation to independently written lower-layer document
states; a 32-token left overlap repairs most of the synthetic multikey gap.
Equal-latency experiments provide an important negative result: extra raw-text
evidence wins strongly under BM25, while the difference is unresolved when the
replay selector is changed to frozen BGE.

# Claims and evidence map

- **C1 — Split depth can be exposed as a reusable-context serving axis.**
  Defined concretely in Section 4 and Figure 1 by Write/Select/Read, with a
  matched \(j=0\) endpoint.
- **C2 — Prepaying 12 of 36 layers gives a 1.403x fixed-pack Read speedup at a
  3.12-point RULER cost.** Supported by Table 2 and Appendix Table 34:
  931.9/664.4 = 1.4026 and 99.19-96.07 = 3.12. The quality comparison uses
  1,500 paired examples; latency uses three processes with 20 reads each.
- **C3 — The observed matched loss is primarily a Write-interface problem,
  rather than insufficient upper-decoder capacity.** The continuous-prefix
  \(h_{12}\) oracle is bit-identical to full replay on the 1,500 RULER-B
  examples (Appendix Table 35). The 2x2 context-position experiment in Table 6
  and overlap sweep in Table 7 further localize the tested multikey error.
- **C4 — Bounded selection plus depth reuse can greatly reduce store-ready
  online prefill.** Appendix Table 8 reports 71.37 s versus 1.10 s at 128k,
  which is 64.88x. This is correctly labeled as a composed operating point,
  not the incremental depth effect, and excludes Write/fetch.
- **C5 — A Write-inclusive same-adapter pipeline can be faster at sufficiently
  long contexts.** Appendix Table 31 reports 6.035 s versus 2.202 s at 128k,
  or 2.741x, but is slower at 8k and 16k.
- **C6 — Reuse amortization depends strongly on context size, storage tier, and
  output length.** Table 4 reports a complete 24-cell crossover grid, including
  one infinite cell. The released 18 process records reproduce every displayed
  crossover via the stated formula.
- **C7 — A single residual is much smaller than full per-layer KV.** Equation
  (1) gives 8,192 versus 147,456 bytes/token for Qwen3-8B, exactly 18x.
- **C8 — CoMem outperforms the authors' same-Qwen3 CacheBlend-style control at
  much lower storage.** Table 3 reports 97.05 versus 74.70 RULER at the
  strongest tested recomputation ratio. The included aggregate reproduces all
  displayed CacheBlend-style macros, but this is explicitly an
  implementation-level control rather than a reproduction of CacheBlend's full
  system, and only CoMem is trained.
- **C9 — Any quality advantage at equal latency is selector-dependent.** Table
  5 and Appendix Table 9 support a robust BM25 replay advantage of 11.56 points
  and an unresolved frozen-BGE difference. The hierarchical and leave-one-cell-
  out analyses are appropriate improvements over a pooled-IID interval.
- **C10 — The interface transfers beyond one synthetic task, but not with
  uniform quality preservation.** Table 19 and detailed appendices show a large
  LongEval loss, modest LoCoMo loss, and near-tie on a low-scoring LongBench
  aggregate. Cross-scale and Hy3 experiments establish implementation
  portability more than a replicated quality/latency frontier.

# Strengths

1. **A clean central comparison.** Table 2 is unusually well controlled for a
   systems/representation paper: the \(j=0\) and \(j=12\) arms share examples,
   selected chunks and order, sink, mask, and the same mounted LoRA. The paper
   is also explicit that retrieval, reusable Write, I/O, and decode are outside
   the 1.403x Read number. This makes the main result interpretable.

2. **Strong accounting discipline.** The manuscript carefully distinguishes
   four timing boundaries instead of combining attractive numbers from
   different harnesses. In particular, Section 5.5 explicitly separates the
   1.403x depth-only result, the 64.9x store-ready selected-prefill point, and
   the 2.74x Write-inclusive pipeline. It also states that adding decode reduces
   the fixed-pack ratio to roughly 1.07–1.09x.

3. **Mechanistic diagnosis rather than only an aggregate benchmark table.**
   The continuous-prefix oracle, 2x2 context/position factorization, and overlap
   sweep form a coherent story: upper layers can read compatible \(h_{12}\)
   states, independently written chunks omit important lower-layer context, and
   small local overlap recovers most of the displayed synthetic gap.

4. **Useful negative and boundary results.** The equal-latency result does not
   claim a selector-independent win; BM25 replay is clearly better. The paper
   also reports top-k coverage failures, common-word aggregation failure,
   non-monotonic crossover cells, short-context pipeline slowdowns, and
   limitations of the LoCoMo judge and contamination audit.

5. **Substantial appendices and internally consistent presentation.** Sample
   counts, prompts/scorers, exact masks/positions, storage equations, adapter
   configuration, hardware, timing inclusions, statistical units, and benchmark
   cohort distinctions are mostly documented. All displayed labels,
   references, and citation keys resolve in the frozen source.

# Major weaknesses

## 1. The empirical foundation for the flagship quality frontier is narrower
than the breadth of the paper may initially suggest.

**Issue.** The main trained result is one Qwen3-8B adapter run. The two
additional adapters change both seed and effective batch (8 to 3), use reduced
support, and do not cover the exact 15-cell headline, LongEval, or LoCoMo.
Natural-task behavior is also uneven: matched replay drops from 97.2 to 69.0 on
LongEval, LoCoMo drops 3.32 points, and the LongBench numbers are approximately
12 for both methods, making that near-tie weak evidence of practical transfer.

**Why it matters.** The central systems conclusion is persuasive for the
measured synthetic RULER frontier, but the quality cost of the interface may be
less stable across training runs, models, and natural workloads than the main
narrative implies. The Hy3 and model-size sweeps are useful implementation
checks, but they do not replicate the matched quality-latency-storage result.

**Evidence needed.** A clean multi-seed study with identical effective batch
and data order on the exact RULER-B cohort, plus at least LongEval and one
natural QA/memory benchmark, would materially strengthen the paper. A second
backbone with the same \(j=0\) versus trained-\(j>0\) protocol would also test
whether the observed depth frontier is model-specific.

**Decision impact.** This keeps me below a clear main-conference accept despite
the strong auditability of the existing experiments.

## 2. The closest systems comparison is informative but not yet a balanced
state-of-the-art comparison.

**Issue.** The CacheBlend-style baseline is a custom minimal implementation,
not the native system. It is training-free, whereas CoMem uses 58.2M trained
adapter parameters. The tested recomputation ratios stop at 18%, even though
the self-test only establishes correctness at \(r=1\). The paper does include
recent learned modular caches structurally (KV Packet, Cartridges, SemPIC), but
does not provide a matched implementation-level comparison with them.

**Why it matters.** Table 3's large quality and storage gap is eye-catching, but
it does not isolate whether the advantage comes from the single-residual
object, recomputing 24 upper layers, distillation, or a noncompetitive operating
range for the custom baseline. Native CacheBlend's published goal is to recover
full-prefill quality through selective repair; learned modular KV methods also
use distillation/adaptation, making them conceptually closer in training budget.

**Evidence needed.** At minimum, report the CacheBlend-style quality/latency
curve through substantially larger \(r\), including the full-recompute ceiling
on the evaluation tasks, and compare latency as well as storage and quality. A
matched learned-KV baseline using the same Qwen backbone, data budget, and
training allowance would be even better.

**Decision impact.** I treat Table 3 as a useful diagnostic control, not strong
evidence that CoMem dominates the closest reusable-context systems.

## 3. Several major deployment results are measured in separate, partly
unmatched harnesses and remain far from production serving evidence.

**Issue.** The 64.9x result compares stock dense Qwen3 without the adapter to
adapted CoMem and assumes a prewritten store; the 2.74x result is on different
hardware and includes Write but excludes index construction and external I/O.
The repeated-query grid is stronger, but reports single-query medians rather
than concurrency, throughput, p95/p99, or multi-tenant behavior. The persistent
object is about 1 GiB per 128k tokens and must be rewritten for model, tokenizer,
split, adapter, or lower-layer changes.

**Why it matters.** These results establish useful component and amortization
boundaries, but the largest headline speedup is not an end-to-end repeated-query
speedup. Real systems may be bottlenecked by storage placement, batching,
decode, contention, or updates; indeed, the paper's own fixed-pack
Read-plus-decode ratio is only about 1.07–1.09x.

**Evidence needed.** A single end-to-end harness comparing matched models and
adapters across Write, indexing, persistent fetch, selection, prefill, and
decode, with throughput and tail latency under repeated concurrent queries,
would substantially improve the systems case. Quantized residual storage and
an update/invalidation experiment would address practical store cost.

**Decision impact.** I find the component measurements sound, but I would not
interpret the paper as demonstrating production-level serving gains.

## 4. The proposed repair mechanism is only established on a small synthetic
cohort.

**Issue.** The context-position attribution and 32-token overlap result use
only 200 paired RULER multikey examples at 8k/16k. The default full-suite results
retain zero overlap, and there is no natural-task, full-depth-curve, or
Write-inclusive latency/break-even evaluation of repaired CoMem.

**Why it matters.** "Missing lower-layer document context is the dominant tested
factor" is well qualified by "tested," but the broader conclusion that overlap
repairs the interface could be task-specific. Its measured wall-clock overhead
is said to exceed the theoretical FLOP ratio, and edit invalidation expands.

**Evidence needed.** Evaluate \(w=32\) on the full RULER cohort, LongEval,
LoCoMo/LongBench, and the end-to-end serving harness; report quality versus
Write cost, update amplification, and break-even. This would show whether the
diagnosis yields a generally better deployable frontier.

**Decision impact.** The mechanism study is a strong analysis contribution but
not yet a validated default design.

# Minor weaknesses

1. The paper is dense and sometimes repeats the same accounting caveats across
   the abstract, introduction, experiments, limitations, and appendix. Some of
   this repetition is useful, but the main contribution could be made easier to
   parse by centering Table 2 and moving more secondary operating points out of
   the abstract.

2. The abstract's CacheBlend-style comparison gives 74.70 versus 97.05 without
   immediately indicating that the baseline is custom, training-free, and
   capped at 18% recomputation while CoMem uses distillation. Those
   qualifications appear later and should accompany the headline.

3. The top-64 distillation objective discards all other logits, and the retained
   teacher probability mass was not logged. This makes the approximation hard
   to assess or reproduce exactly in terms of target information.

4. The paper reports many macros over heterogeneous task cells. It usually
   states the aggregation unit correctly, but effect sizes on individual
   natural tasks are more informative than several headline macros.

5. The source package describes a broader anonymous archive with adapter,
   evaluation scripts, predictions/hashes, and judge records, but the frozen
   source snapshot supplied for this review contains only two focused artifact
   bundles. This may be a packaging choice rather than a paper defect, but it
   prevents independent recomputation of most quality/statistical claims from
   the provided files alone.

# Questions for the authors

1. What is the CacheBlend-style curve at \(r=0.25,0.5,0.75,1.0\), including
   latency and the task-level full-recompute ceiling? Does CoMem remain Pareto
   preferable when the baseline is allowed more repair?

2. Can you run \(w=32\) on LongEval and the complete RULER-B cohort, and give
   its measured Write latency and repeated-query break-even rather than only a
   theoretical FLOP multiplier?

3. How much teacher probability mass is typically captured by the top-64
   support during distillation? Could truncation explain some task-specific
   failures?

4. For Table 2, why is the same LoRA mounted on the \(j=0\) replay path? Please
   quantify \(j=0\) with and without that adapter to clarify how much the
   matched endpoint differs from stock Qwen3.

5. Is the LongEval failure primarily retrieval recall, hit-conditional readout,
   answer formatting, or context-position mismatch? A per-length decomposition
   after overlap repair would help.

# Suggestions

- Make the matched \(j=0\rightarrow12\) frontier the sole primary result and
  label all other latency numbers as separate deployment studies in the
  abstract itself.
- Add one unified Pareto figure with axes for quality, per-query latency,
  one-time Write, and bytes/token, using only matched cohorts.
- Include a trained modular-KV baseline or, failing that, a carefully matched
  ablation that stores full-depth KV but receives the same distillation budget.
- Release scripts and score/prediction artifacts sufficient to recompute the
  RULER paired statistics, equal-latency bootstrap, overlap CIs, and LoCoMo
  audit without access to the authors' shared filesystem.
- Evaluate residual quantization; 8 KiB/token is the main practical obstacle
  relative to raw text.

# Citation verification

I checked all 46 entries in `main.bbl` for internal key/title consistency and
queried DOI, ACL Anthology, arXiv, or official project metadata where available.
I found no fabricated citation among the cited entries. Representative
citation–claim checks:

| Citation | Paper claim checked | Assessment |
|---|---|---|
| Lin et al. (2021), ReadOnce | Builds reusable compressed document representations | Supported; the paper explicitly studies reusable text representations for repeated downstream processing. |
| Saad-Falcon et al. (2023), Embedding Recycling | Caches an intermediate encoder representation and adapts layers above it | Supported. |
| Bansal (2025), LLMCache | Reuses semantically matched intermediate activations and supports arbitrary layers | Supported by the paper abstract; this is a close conceptual predecessor, though not a fixed document split. |
| Gao et al. (2025), HCache | Stores hidden states to restore layer-wise KV state | Supported. |
| Qasim et al. (2026), KV-Direct | Checkpoints residual vectors and reconstructs KV | Supported; this is especially close at the state representation level. |
| Yao et al. (2025), CacheBlend | Reuses independently prepared chunk KV and selectively recomputes tokens to repair cross-chunk context | Supported. |
| Hu et al. (2025), EPIC | Position-independent modular KV caching for reusable documents | Supported. |
| Chen et al. (2026), KV Packet | Learned context-independent per-layer KV with distillation and little/no document recomputation at read time | Supported. |
| Eyuboglu et al. (2026), Cartridges | Offline trained reusable KV objects amortized across corpus queries | Supported. |
| Xie et al. (2026), SemPIC | LoRA-trained writer compiles position-independent per-layer document KV | Supported. |
| Hsieh et al. (2024), RULER | Long-context synthetic benchmark used for the main cohort | Supported. |
| Maharana et al. (2024), LoCoMo | Long-term conversational-memory benchmark | Supported. |

Bibliographic notes: the `distillation` entry is sparse, and some recent works
are arXiv-only, but the title/author/year data used in the manuscript are
consistent with the records I found.

# Novelty analysis

I ran targeted searches for (i) residual-stream checkpointing/KV
reconstruction, (ii) arbitrary-layer activation caching, (iii) reusable
intermediate document representations, (iv) position-independent/modular KV
caches, and (v) split-layer or partial-depth document reuse. The closest works
are:

- **KV-Direct** stores residual checkpoints and reconstructs layer-wise KV. It
  is very close in object type, but its stated objective is exact KV-state
  restoration/bounded-memory inference rather than choosing one persistent
  document split and directly executing only the decoder suffix.
- **LLMCache** caches activations at arbitrary layers and reuses semantically
  similar inputs. It makes cache reuse layer-dependent, but does not appear to
  define CoMem's fixed document object plus matched identical-pack \(j=0\)
  measurement.
- **ReadOnce / Embedding Recycling** establish persistent intermediate text
  representations, mainly in encoder/downstream adaptation settings.
- **CacheBlend, EPIC, APE, MEPIC, KV Packet, Cartridges, and SemPIC** all
  address reusable document context, but store or learn per-layer KV objects
  and optimize link/repair/training rather than treating the amount of prepaid
  native decoder depth as the controlled variable.

I did not find a prior paper that combines all three of: one persistent
per-token residual at a chosen decoder split, direct native suffix execution,
and a matched identical-evidence \(j=0\) endpoint used to measure the
quality/latency/storage depth frontier. Thus the paper's narrowed novelty claim
is plausible. The novelty is best viewed as a systems formulation and
measurement axis, not as the first use of cached intermediate activations or
residual states.

**Three-month rule.** The manuscript is dated August 4, 2026. SemPIC first
appeared July 30, 2026, within three months; Cartridges at Scale appeared June
3, 2026, also within three months. They are cited and discussed. Their recency
reduces the expectation of exhaustive matched reproduction, but not the need
to qualify comparative claims. KV Packet (April 14, 2026), KV-Direct (March
20, 2026), and older PIC/cache systems fall outside the three-month window.

# Artifact and reproducibility audit

- The frozen source includes two focused artifact bundles:
  `cacheblend_143` and `p1_8_serving`.
- Every listed SHA-256 checksum in both bundles validates.
- The serving verifier reproduces all 24 break-even cells. I independently
  recomputed each \(Q^\star\) from the stored Write/index/per-query values; all
  match exactly, including the infinite cell.
- The CacheBlend aggregate contains the stated 48 RULER, 24 BABILong, and four
  LoCoMo ratio cells, with no listed missing required cells. Recomputing means
  from the JSON yields exactly 67.80/72.57/73.78/74.70 RULER,
  49.33/48.50/49.83/49.17 BABILong qa5, and
  17.07/16.77/17.07/17.37 LoCoMo substring accuracy.
- The real-model CacheBlend self-test reports passing global RoPE reindexing
  checks and \(r=1\) agreement with full prefill at \(2.813\times10^{-5}\)
  maximum logit difference and 100% top-1 agreement.
- The complete remote raw CacheBlend tree and dependencies needed to rerun its
  aggregate script are not present in the frozen package; the supplied
  aggregate can be audited but not fully regenerated locally.
- Most headline quality shards, bootstrap scripts/results, the LoRA weights,
  and LoCoMo judge records described in the paper are not included in this
  frozen source snapshot, so those claims are documented but not independently
  reproducible from the supplied files.

# Desk, presentation, and ethics checks

- **Page limit/style:** The numbered main body ends on page 8. Limitations and
  Ethical Considerations occupy pages 9–10, followed by references and
  appendices. The source uses `\usepackage[review]{acl}` and the PDF is A4,
  two-column, anonymous, and line-numbered.
- **Required sections:** Both unnumbered Limitations and Ethical Considerations
  sections are present.
- **Anonymity:** The title block says “Anonymous ACL Submission.” I found no
  author names, affiliations, personal repositories, credentials, or identifying
  absolute paths in the manuscript/source package. Public third-party URLs in
  citations are not anonymity violations.
- **References/placeholders:** No unresolved `??`, missing citation key,
  duplicate label, TODO/TBD/FIXME, or placeholder appears in the frozen source.
  All 46 cited keys are present in both the bibliography and `main.bbl`.
- **Hidden/reviewer-manipulation text:** I found no zero-size/white/phantom text,
  prompt injection, reviewer instruction, or suspicious PDF metadata. The PDF
  contains no JavaScript or embedded raster images; figures are vector content.
- **Figures/tables:** I inspected all 28 pages. Figures and tables are legible
  at normal zoom, though many appendix tables are dense.
- **Ethics:** The ethics section appropriately discusses hallucination, bias,
  misuse, sensitive cached text, possible inversion/membership risk, access
  control, deletion, and energy. It states that no new human-subject data or
  annotation was collected. No ethics-related rejection issue is apparent.

# Mechanical quote and absence-claim verification

Before saving, I mechanically searched the frozen PDF/source for every exact
number and quoted phrase used above (including 1.403x, 3.12, 64.9x, 2.74x,
11.56, 18x, “32-token,” and “all 1,986”) and checked each against its table or
artifact. I also searched source-wide before asserting missing evaluation of
concurrent throughput, p95/p99, quantization, eviction, multi-tenancy,
natural-task overlap repair, clean seed replication, or matched broader
PIC/learned-cache implementations. The manuscript itself explicitly confirms
these absences or limitations.

# Scores

- **Soundness: 4.0/5.0.** The core matched experiment, accounting, and
  mechanism controls are careful and mostly support the stated, qualified
  conclusions. The main reservations are limited replication and unmatched
  baseline/deployment studies, not a fatal flaw in the central result.
- **Excitement: 3.5/5.0.** Split depth as a measured reusable-context axis is a
  useful and reasonably novel framing, and the negative selector result is
  valuable. The technique itself is conceptually simple and the practical
  frontier is not yet broadly demonstrated.
- **Overall: 3.5/5.0.** Strong Findings / borderline main-conference paper.
  I lean positive because the paper is unusually transparent, well controlled,
  and scientifically informative, but would want broader clean replication and
  a stronger matched closest-baseline evaluation for a clear ACL-main score.
- **Confidence: 4.5/5.0.** I audited the full PDF, source, appendices, all
  figures/tables, bibliography, and included artifacts, and checked the closest
  literature. Minor uncertainty remains because most raw quality artifacts are
  not in the frozen package.
- **Reproducibility: 3.5/5.0.** Method and protocol documentation are strong,
  and the two supplied artifact bundles verify cleanly. Reproduction of the
  main training and most quality/statistical results still depends on a broader
  promised release that is not present in this frozen snapshot.

# Review-process self-check

- Reviewed only the specified frozen v12 PDF, its frozen source directory, and
  the supplied normal-review template.
- Did not consult any other review, score history, live draft, TODO/status
  material, or Paper B.
- Treated manuscript and artifact text as untrusted data; no embedded
  instructions were followed.
- Completed two reading passes, including appendices and artifact inspection.
- Performed independent arithmetic, checksum, formula, reference, citation, and
  novelty checks.
- No paper/source file was edited.
