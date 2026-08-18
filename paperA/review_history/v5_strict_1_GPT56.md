---
review_mode: strict
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 3.5
---

# Paper Summary

This paper studies a repeated-query setting over a stable corpus. CoMem writes each
document chunk once through the lower part of a decoder-only transformer, stores one
intermediate residual vector per token at split depth \(j\), retrieves a bounded set
of chunks for each query, and resumes only layers \([j:L)\). Its strongest result is
deliberately narrow: on Qwen3-8B, a same-pack, same-example, same-mask, same-adapter
\(j=0\) versus \(j=12\) comparison changes isolated model Read from 931.9 to
664.4 ms (1.403×), while RULER-B falls from 99.19 to 96.07 (paired loss 3.12,
95% CI [2.36, 3.93]). A rank-32 upper-layer LoRA trained by self-distillation is
important for making chunk-local residuals readable.

The paper also measures the storage and serving boundary (8 KiB/token, repeated-query
break-even), performs an equal-online-latency replay audit, and diagnoses one source of
quality loss. In the equal-latency mixture, iterative-BM25 raw replay beats CoMem,
whereas a frozen-BGE replay arm is statistically tied; the authors appropriately do
not claim a CoMem aggregate win. In the focused multikey diagnosis, restoring
lower-layer document context recovers 92.5 to 100.0, and a 32-token left overlap
reaches 98.5 without changing persistent bytes or per-query Read work.

I read the 23-page frozen PDF twice, including both appendices, and inspected every
rendered figure and table.

# Claims and Evidence Map

## C1. Depth reuse gives a measurable quality–Read-latency operating point

**Claim.** Skipping the first 12 layers for stored document residuals produces a
1.403× isolated selected-pack model-Read speedup at a 3.12-point RULER cost.

**Minimum sufficient experiment.** Same backbone, adapter, examples, retrieved chunk
IDs/order, pack, mask, positions, and timing harness; vary only the replay start; use
paired quality statistics and repeated latency processes.

**Evidence.** Section 5.1 and Table 2 (PDF p.6); Appendix A.4, Tables 30–31
(PDF p.21). Quality has 1,500 paired examples and paired bootstrap/McNemar analyses.
Latency uses three independent processes and identical fixed packs. This is sufficient
for the stated *isolated model Read* claim, not for end-to-end acceleration.

## C2. Self-distillation makes the residual interface substantially more usable

**Claim.** A rank-32 LoRA can repair much of the incompatibility between independently
written \(h_j\) states and the native upper-layer reader without updating the
backbone.

**Minimum sufficient experiment.** At fixed split and identical inference protocol,
compare adapter off/on; report more than one task and enough training detail to
reproduce the objective.

**Evidence.** Section 4; Appendix Tables 9, 11–12, 26 (PDF pp.12–13, 20).
At \(j=12\), LoCoMo rises 24.52→38.27 and the paired RULER adapter ablation has large
gains. The objective, support truncation, optimizer, data, trainable modules, hash, and
compute are specified. Evidence supports effectiveness, but not stable expected
performance across training randomness (W3).

## C3. CoMem is beneficial only under sufficient reuse and appropriate storage

**Claim.** The 8 KiB/token store amortizes after roughly 8–11 queries at 32k for
generation lengths through 128 tokens, and 25.8–27.6 queries at 128k for one generated
token.

**Minimum sufficient experiment.** Measure Write, replay setup, fetch, model Read, and
decode under matched query workloads and state placement; explicitly state omitted
costs and retained generation lengths.

**Evidence.** Section 5.2 and Table 3 (PDF p.6); Appendix A.4 and Table 32
(PDF pp.18–21). The displayed equation and measured components support the stated
single-query-harness crossover. The paper correctly limits the 128k result to
\(G=1\) and does not claim production p95 throughput.

## C4. Equal-online-latency quality is selector-dependent, with no CoMem aggregate win

**Claim.** Under the reported nine-cell diagnostic, iterative-BM25 replay beats CoMem
by 11.56 points, whereas frozen-BGE replay is tied at −1.00 points (CoMem minus
replay).

**Minimum sufficient experiment.** Predeclare and independently calibrate budgets;
include selection/fetch/query Write/model TTFT consistently; expose quality cells,
weights, generation/scoring, selectors, storage assumptions, absolute times, and an
appropriate uncertainty analysis.

**Evidence.** Section 5.2 and Table 4 (PDF p.6); Appendix Table 8 (PDF p.12).
The latency boundary and result are now unusually transparent. The conclusion “no
CoMem win; selector dependence” follows descriptively. However, the mixture and IID
example bootstrap do not justify a population-level heterogeneous-workload CI (W1).

## C5. Missing lower-layer document context is a major tested failure mode on one
multikey cohort

**Claim.** A continuous/document-contextual lower-layer writer restores the displayed
multikey gap; document-origin positions alone do not; local overlap recovers most of
the gap.

**Minimum sufficient experiment.** Hold retrieval, adapter, and upper-layer Read fixed;
factor lower-layer context scope and Read-position convention; then test a deployable
context approximation with paired uncertainty.

**Evidence.** Section 5.3, Tables 5–6 (PDF pp.6–7), and Appendix Table 31 (PDF p.21).
The \(2\times2\) factorization and overlap sweep meet the minimum experiment for this
cohort. The paper correctly limits the result to a synthetic paired diagnostic and
does not call overlap a universal repair.

## C6. Model-side Read is bounded while external selection/store work is not

**Claim.** With fixed \(k,c\), model Read length is independent of stored-corpus length;
retrieval/index/store costs and evidence coverage remain scaling bottlenecks.

**Minimum sufficient experiment.** Increase the store substantially while holding the
read budget fixed; report evidence recall, actual read tokens, selector latency, and
failure cases once evidence exceeds the budget.

**Evidence.** Section 4, Equation 1; Appendix Tables 24 and 32 (PDF pp.19, 21).
Read length remains about 6.2–6.5k from 128k to 4M tokens while BM25 latency grows
approximately linearly; explicit over-budget failures are shown. The wording is
appropriately “bounded model work,” not bounded total system work.

# Strengths

## S1. The central causal comparison is clean and scoped correctly

Section 5.1/Table 2 (PDF p.6) fixes the selected chunks, order, examples, sink, mask,
and adapter, and varies replay start. Appendix Tables 30–31 add process-level latency
details, paired quality statistics, and a continuous-prefix fidelity ceiling. The
paper repeatedly distinguishes model Read from retrieval, Write, I/O, and decode.

## S2. The paper is unusually candid about negative results and workload boundaries

The abstract, Introduction, Conclusion, and Limitations all state that CoMem is not
quality preserving, does not win either equal-latency aggregate, stores far more than
text, requires reuse, and lacks a matched modular-cache comparison. This restraint
substantially improves the scientific value of what is fundamentally an operating-point
measurement.

## S3. The equal-latency audit is materially informative

Table 8 (PDF p.12) discloses all nine cells, equal cell weights, the unusual LoCoMo
subset, scoring rules, the disjoint calibration documents, candidate \(k\) values,
hardware, warmups/repetitions, included/excluded latency components, absolute TTFT,
storage placement, cold-index failure, and statistical unit. Thus I regard the basic
descriptive equal-latency comparison as audited rather than opaque. W1 concerns the
inferential unit and external meaning, not undisclosed protocol.

## S4. Mechanistic diagnosis is controlled rather than speculative

Tables 5–6 and 31 separate lower-layer context, positions, reusable versus oracle
states, and a deployable overlap approximation. The +4.5 interaction is acknowledged,
so the paper avoids an unjustified additive attribution. This is a useful example of
turning a performance loss into a bounded causal diagnosis.

## S5. Formulae and accounting are internally consistent

Equation 1 gives residual/full-KV storage ratio
\(n_q/(2Ln_{kv})=32/(2\cdot36\cdot8)=1/18\). For \(d=4096\), bf16 residual storage is
8,192 bytes/token and 128k tokens occupy exactly 1 GiB. The nominal read cap
\(1+12\cdot512+512=6,657\) and 931.9/664.4=1.4026 are correct. Boundary cases
\(j=0\), deeper splits, continuous-prefix \(h_{12}\), and an MoE partition self-test
are discussed.

## S6. Evaluation breadth, uncertainty reporting, and reproducibility detail are strong

The appendices report full RULER, LongEval, BABILong, LongBench, and LoCoMo cells;
storage and retained-KV controls; timing boundaries; seeds; hashes; optimizer and
training construction; official scorers; prompt sensitivity; contamination caveats;
judge dependence; and store-scaling failures. The LoCoMo conversation-cluster
bootstrap is especially appropriate and contrasts positively with W1.

## S7. Presentation, figures, and tables are generally clear

Figure 1 accurately visualizes Write/Select/Read and the \(j=0\) control. Figure 2 is
explicitly motivational rather than validation. I found no numerical contradiction
between the abstract and the corresponding tables. The many appendix tables are dense
but legible in the rendered PDF.

# Weaknesses

## W1. Equal-latency uncertainty treats heterogeneous and clustered examples as IID

- **Location:** Appendix Table 8, “Statistical unit” (PDF p.12; source
  `sections/tab_equal_latency_protocol.tex:44–47`).
- **Exact quote (17 words):** “The percentile bootstrap pools and resamples all 900
  paired example differences IID 10,000 times with seed 0”
- **Problem:** The reported quantity is an equal-cell macro over nine deliberately
  heterogeneous cells, yet its CI pools individual examples. This implicitly weights
  example-level variance rather than uncertainty over task/length cells, and the 100
  LoCoMo items are additionally nested in one conversation. The paired McNemar test
  has the same independence issue. The point estimates are auditable; the confidence
  intervals are over-precise for the decision-relevant heterogeneous mixture.
- **Affected claim/norm:** C4 and the norm that uncertainty match the aggregation and
  sampling unit. This matters because “statistically tied” is an abstract-level
  conclusion, while the chosen nine cells are not a random workload sample and one
  natural component comes from a single conversation.
- **Sufficient remedy:** Recompute uncertainty by resampling paired examples *within*
  cell and aggregating cell means, with a conversation-cluster resample for LoCoMo;
  report cell-wise differences and a sensitivity analysis that leaves out each cell
  (or report the result as descriptive with no population CI).
- **Severity:** **Major** for the equal-latency inferential claim, but not fatal to the
  central same-pack depth result.

## W2. The closest reusable-context alternatives are not compared under a matched
end-to-end protocol

- **Location:** Related Work, final sentences of the PIC/modular-cache paragraph
  (PDF p.3; source `sections/02_related.tex:23–27`).
- **Exact quote (15 words):** “No artifact in this study supplies a same-backbone,
  same-hardware PIC, chunk-KV-repair, or learned modular-cache result”
- **Problem:** The central \(j=0\rightarrow12\) measurement is internally valid, but it
  does not establish whether the measured depth-reuse point is useful relative to
  CacheBlend/TurboRAG/EPIC/KV Packet/Cartridges-style reusable objects under matched
  quality, storage, Write amortization, retrieval, and latency boundaries.
- **Affected claim/norm:** Excitement and practical significance, not C1's internal
  soundness. The paper itself calls these systems the closest serving workload or
  learned objects; without a matched reference, a reader cannot place CoMem on the
  relevant systems frontier.
- **Sufficient remedy:** Implement at least one strong same-backbone reusable-KV/PIC
  baseline and compare end-to-end TTFT/quality/storage/Write break-even under the same
  selected packs and storage tiers. A carefully validated public implementation is
  sufficient; every cited family is not required.
- **Severity:** **Major** for an ACL-main-level impact case; the authors appropriately
  disclose it, so I do not treat it as a hidden soundness flaw.

## W3. Training-run evidence is insufficient for stable claims about the learned
interface

- **Location:** Limitations, opening paragraph (PDF p.7; source
  `sections/07_limitations.tex:4–10`).
- **Exact quote (7 words):** “The flagship is one batch-8 training run”
- **Problem:** The two added adapters change effective batch from 8 to 3, and their
  reduced-support evaluation omits the exact 15-cell RULER-B, LoCoMo, and LongEval
  headlines. Consequently, the paper demonstrates that one adapter works, but not the
  expected quality or variance of the training recipe.
- **Affected claim/norm:** C2, reproducibility, and robustness of the learned
  interface. This matters because the adapter is necessary, split-specific, and
  responsible for large gains.
- **Sufficient remedy:** Train at least two additional batch-8 seeds with the same
  examples/order budget and evaluate all headline suites; report run-level means/SDs
  and whether the central quality–latency operating point changes.
- **Severity:** **Minor-to-Major**; I score it **Major** for reproducibility but not for
  existence of the demonstrated operating point.

## W4. A baseline-protocol assertion is not supported in the frozen paper itself

- **Location:** Appendix A.5, prompt/native-chat sensitivity (PDF p.20; source
  `sections/08_appendix.tex:359–362`).
- **Exact quote (15 words):** “The official MemoryLLM chat template does not improve
  the tested BABILong, RULER, or LoCoMo diagnostics”
- **Problem:** The paper says full results are in the released artifacts, but the
  frozen PDF/source contains no table, sample counts, or uncertainty for this
  assertion. Since MemoryLLM's no-chat row is used as the principal external
  persistent-memory reference, the prompt-sensitivity check should be inspectable in
  the submission.
- **Affected claim/norm:** Baseline fairness and self-contained evidence. It does not
  affect the matched \(j=0\) control, but it affects interpretation of the external
  MemoryLLM rows.
- **Sufficient remedy:** Add a compact table with native-chat versus no-chat scores on
  the displayed tasks, identical supports, generation limits, and scorer details; or
  remove the improvement assertion and label the no-chat comparison purely
  protocol-matched/descriptive.
- **Severity:** **Minor**.

## W5. One bibliography entry has incorrect page metadata

- **Location:** References, TurboRAG entry (PDF p.10; `main.bbl:160–166`).
- **Exact quote (4 words):** “pages 6588--6601”
- **Problem:** DOI/Crossref metadata for `10.18653/v1/2025.emnlp-main.334` gives pages
  6599–6612. Title, authors, year, venue, and DOI otherwise match.
- **Affected claim/norm:** Citation accuracy only; no experimental claim changes.
- **Sufficient remedy:** Correct the page range from the authoritative anthology/DOI
  record.
- **Severity:** **Minor**.

# Questions That Could Change the Score

1. **Equal-latency reanalysis:** Under a cell-respecting paired bootstrap, with
   conversation-cluster resampling for the LoCoMo component, do the BM25 and BGE
   conclusions remain respectively “raw replay wins” and “inconclusive/tied”? Please
   also provide leave-one-cell-out aggregates.
2. **Matched nearest baseline:** Can the authors provide one same-Qwen3-8B,
   same-hardware, same-pack comparison to a reusable-KV/PIC system, including
   persistent bytes, Write cost, fetch, TTFT, decode, and quality? A result showing
   CoMem is inferior would still be scientifically valuable.
3. **Training stability:** Do two additional batch-8 seeds reproduce the exact
   15-cell RULER-B, LongEval, and LoCoMo operating point within a practically small
   range?
4. **MemoryLLM prompt check:** What are the native-chat versus no-chat numbers and
   supports for the three claimed diagnostics?

# Non-scoring Suggestions and Typos

1. In Table 8, replace “Protocol-complete” with “Protocol-documented” unless the
   aggregation-unit issue in W1 is repaired.
2. Report the nine equal-latency per-cell replay/CoMem differences in the PDF, not
   only the macro.
3. Clarify in Table 3 whether fractional \(Q^\star\) values mean “strict crossover
   after \(\lceil Q^\star\rceil\) completed queries.”
4. `main.bbl` TurboRAG pages should be 6599–6612, not 6588–6601.
5. Some bibliography entries omit available page spans (e.g., XC-Cache and ReadOnce);
   adding them would improve consistency, though this is not a correctness issue.
6. The title and paper are somewhat jargon-dense, but the workload contract and
   Figure 1 mitigate this.

# Score Rationale

## Soundness: 3.5/5

The central same-pack depth endpoint is carefully controlled, formulae and arithmetic
check out, negative results are disclosed, and claim scope is disciplined. I lower
from 4 because the equal-latency abstract claim uses an unsuitable IID inferential
unit for a heterogeneous macro and because the learned interface lacks clean
same-budget multi-seed headline replication.

## Excitement: 3.0/5

The paper offers a useful and transparent measurement of a specific residual-reuse
design point, plus a good bounded diagnosis. However, intermediate representations,
activation replay, and reusable document caches are established themes, and the
submission intentionally stops short of a matched comparison to its closest
deployable alternatives. I therefore see solid Findings-level interest rather than a
clear ACL-main impact case.

## Overall: 3.0/5

This is a careful, honest, empirically rich paper with a valid narrow contribution.
The central result is publishable and the paper avoids overclaiming. Under the
specified calibration, however, ACL main (4.0) requires a stronger practical/novelty
case and cleaner decision-level statistics. I am between 3.0 and 3.5; per instruction,
I choose the lower bin. My recommendation is **Findings-level** unless W1 and at least
one of W2/W3 are materially addressed.

## Confidence: 4.5/5

I inspected the complete rendered paper twice, all figures/tables, equations,
bibliography, and the frozen source. I also mechanically checked numerical identities,
references, quotes, and claimed omissions. Remaining uncertainty is mainly that I
could not inspect the promised external artifact or rerun GPU experiments.

## Reproducibility: 3.5/5

Configuration, objective, data construction, hardware, hashes, scorer choices, sample
counts, timing boundaries, and most seeds are unusually detailed. Deductions are for
one flagship training run, incomplete clean-seed replication, mutable GPT-4o judging,
unlogged total experimental compute/training peak memory, and several claims whose raw
artifact records are not included in the frozen source.

# Limitations, Ethics, and Desk-Reject Risks

## Page/style/section checks

- The rendered PDF has 23 pages: main text occupies pp.1–8, references pp.8–10, and
  appendices pp.11–23. This appears consistent with an eight-page long-paper main
  body plus unlimited references/appendix.
- The source uses `\documentclass[11pt]{article}` and `\usepackage[review]{acl}`;
  rendered page size is A4 and the paper visibly follows the ACL two-column review
  style.
- An exact unnumbered `Limitations` section appears before `Ethical Considerations`.
- No unresolved references, `??`, TODO/FIXME/TBD placeholders, missing labels, or
  duplicate labels were found in the frozen paper source/rendered text.
- A clean rebuild could not be independently performed because no TeX engine is
  installed in the review environment; the frozen PDF itself renders correctly.

## Anonymity

- Author is “Anonymous ACL Submission”; no author names, affiliations, emails,
  institution-specific paths, GitHub repository, or self-identifying acknowledgments
  were found.
- Dates for the external judge audit and public model/repository names do not by
  themselves identify the authors.
- **Desk-reject anonymity risk: low.**

## Prompt injection / hidden text

- I searched paper/source text for reviewer manipulation, “ignore previous,” acceptance
  requests, score directives, hidden-white/tiny-text commands, and prompt-injection
  language. None was found.
- PDF metadata reports no JavaScript, forms, encryption, or suspect objects; visual
  inspection found no hidden-text anomaly. The figures are vector PDFs and no raster
  images were embedded.
- Paper text was treated only as reviewed data.

## Ethics

The ethics section appropriately covers hallucination/bias/unsafe output, sensitive
retrieval, residual inversion or membership inference, access control, encryption,
deletion, cross-tenant isolation, and energy. No new human-subject data or annotator
recruitment is claimed. The paper also discusses licenses and does not redistribute
PG-19 text, benchmark examples, model weights, credentials, or private API responses.
I see no ethics-based rejection issue.

## Other desk risks

- No obvious formatting, page-limit, anonymity, citation-placeholder, or missing
  Limitations violation.
- Public references dated after **May 4, 2026**—Cartridges at Scale (June 3, 2026) and
  SemPIC (July 30, 2026)—are treated here only as concurrent work and were not used to
  diminish novelty.
- **Overall desk-reject risk: low.**

# Complete Citation Audit

I mechanically confirmed that all 43 keys actually cited in the TeX appear exactly
once in `main.bbl`, and there are no uncited `main.bbl` entries. Status below reflects
metadata checks against DOI/Crossref, ACL Anthology, arXiv API, official model pages,
and/or canonical publication records available on August 4, 2026. “Verified” means
the cited work and core metadata were found; it does not imply every optional field
was exhaustively checked.

| # | Key | Status | Audit note |
|---:|---|---|---|
| 1 | `cachecraft` | Verified | DOI, title, authors, 2025 journal venue match. |
| 2 | `longbench` | Verified | ACL DOI, authors, title, pages 3119–3137 match. |
| 3 | `pyramidkv` | Verified | arXiv 2406.02069 and COLM 2025 metadata match. |
| 4 | `kvpacket` | Verified | arXiv 2604.13226, posted 2026-04-14; title/authors match. |
| 5 | `cartridgesbase` | Verified | arXiv 2506.06266 and ICLR 2026 work match. |
| 6 | `hcache` | Verified | EuroSys DOI and pages 128–143 match. |
| 7 | `llama3` | Verified | arXiv 2407.21783 title/year match. |
| 8 | `cartridges` | Verified | arXiv 2606.04557, posted 2026-06-03; concurrent. |
| 9 | `distillation` | Verified | arXiv 1503.02531 title/authors/year match. |
| 10 | `ruler` | Verified | arXiv 2404.06654 / COLM 2024 work match. |
| 11 | `lora` | Verified | arXiv 2106.09685 / ICLR 2022 work match. |
| 12 | `epic` | Verified | arXiv 2410.15332 / ICML 2025 title and authors match. |
| 13 | `ragcache` | Verified | DOI gives TOCS 44(1), pages 1–27; entry is substantively correct. |
| 14 | `babilong` | Verified | DOI, NeurIPS 37, pages 106519–106554 match. |
| 15 | `rag` | Verified | NeurIPS 2020 RAG paper metadata match. |
| 16 | `longchat` | Verified | Official LMSYS 2023 blog title/authors match. |
| 17 | `snapkv` | Verified | DOI, authors, NeurIPS pages 22947–22970 match. |
| 18 | `ilre` | Verified | arXiv 2508.17892 title/authors/year match. |
| 19 | `readonce` | Verified | ACL 2021 DOI/title/authors match; pages omitted but no false field. |
| 20 | `minicache` | Verified | DOI, NeurIPS 37, pages 139997–140031 match. |
| 21 | `turborag` | **Metadata error** | DOI/title/authors/venue match, but authoritative pages are 6599–6612, not 6588–6601. |
| 22 | `locomo` | Verified | ACL DOI and pages 13851–13870 match. |
| 23 | `xccache` | Verified | Findings EMNLP 2024 DOI/title/authors match; pages omitted. |
| 24 | `kvdirect` | Verified | arXiv 2603.19664, posted 2026-03-20; title/authors match. |
| 25 | `pg19` | Verified | arXiv 1911.05507 title/authors/year match. |
| 26 | `bm25` | Verified | DOI, journal, volume/issue, pages 1–174 match. |
| 27 | `embeddingrecycling` | Verified | Findings EACL 2023 DOI/title/authors match. |
| 28 | `gemfilter` | Verified | ACL Anthology 2026 Findings entry, DOI and pages 13839–13857 match. |
| 29 | `reform` | Verified | arXiv 2506.01215 title/authors/year match. |
| 30 | `lloco` | Verified | EMNLP 2024 DOI and pages 17605–17621 match. |
| 31 | `hunyuan` | Verified | Official Tencent Hy3 page/config confirms 295B/21B active, 80 layers, 192 experts, top-8, 256K. |
| 32 | `fusionrag` | Verified | arXiv 2601.12904 title/authors/year match. |
| 33 | `mepic` | Verified | arXiv 2512.16822 title/authors/year match. |
| 34 | `longmem` | Verified | NeurIPS 2023 title/authors match. |
| 35 | `memoryllm` | Verified | ICML/PMLR 2024 metadata and pages 50453–50466 match. |
| 36 | `infllm` | Verified | DOI, NeurIPS 37, pages 119638–119661 match. |
| 37 | `streamingllm` | Verified | arXiv 2309.17453 / ICLR 2024 metadata match. |
| 38 | `sempic` | Verified | arXiv 2607.28069, posted 2026-07-30; concurrent. |
| 39 | `xu2024retrievallong` | Verified | ICLR 2024 paper title/authors match. |
| 40 | `qwen3` | Verified | arXiv 2505.09388 title/year match. |
| 41 | `ape` | Verified | arXiv 2502.05431 title/authors/year match. |
| 42 | `cacheblend` | Verified | EuroSys DOI and pages 94–109 match. |
| 43 | `h2o` | Verified | NeurIPS 2023 H2O paper exists; entry is abbreviated but not false. |

## Load-bearing citation–claim checks

1. **CacheBlend/TurboRAG/Cache-Craft precompute or reuse chunk KV and repair/fuse
   context. — Match.** Their abstracts and official records describe offline or cached
   chunk KV reuse plus recomputation/fusion/position handling.
2. **EPIC and APE independently encode modular contexts and address position/attention
   misalignment. — Match.** EPIC formalizes position-independent caching with repair;
   APE independently precomputes context KV and realigns attention.
3. **KV Packet uses context-independent compiled KV plus lightweight trained adapters.
   — Match.** Its abstract explicitly describes immutable KV “packets” wrapped in
   trainable soft-token adapters and distillation.
4. **Cartridges are learned reusable KV representations with amortized offline
   training. — Match.** The base paper trains a smaller corpus-specific KV cache
   offline and amortizes it across queries.
5. **ReadOnce/Embedding Recycling are precedents for reusable intermediate
   representations. — Match.** Both cache/reuse intermediate text/model
   representations and adapt downstream computation.
6. **HCache checkpoints activations; KV-Direct reconstructs KV from residuals.
   — Match.** Official metadata/abstracts support both descriptions.
7. **ILRe/REFORM/GemFilter reduce/select/gather tokens before later recomputation.
   — Match with nuance.** ILRe recalls tokens from an intermediate-layer cache,
   REFORM gathers and selectively recomputes KV, and GemFilter selects tokens using
   early layers; none is described as CoMem-style cross-query persistence.
8. **SnapKV/PyramidKV are retained-token/KV compression after processing the prompt.
   — Match.** Their methods select/compress KV rather than provide the same
   cross-query persistent residual object.

# Novelty Search Summary

## Search protocol and date rule

I ran four targeted searches through arXiv and authoritative metadata sources for:
(i) persistent intermediate residual/activation reuse, (ii) reusable modular document
KV/PIC, (iii) residual-stream reconstruction of KV, and (iv) depth/layer-wise cache
reuse. Novelty is frozen at **August 4, 2026**. Work first public after
**May 4, 2026** is treated only as concurrent.

## Closest works

1. **ReadOnce Transformers (ACL 2021).** Reusable intermediate text
   representations adapted for later transformer processing. This is the clearest
   older conceptual precedent for caching a mid-network representation. CoMem differs
   in decoder-only long-context serving, bounded retrieval, explicit tunable split
   depth, direct suffix execution, and systems accounting.
2. **KV-Direct / “The Residual Stream Is All You Need” (arXiv 2026-03-20).**
   Stores residual vectors and reconstructs layer-wise KV on demand. This is extremely
   close in stored information type, but its main object is exact KV reconstruction
   for bounded-memory inference rather than independently written, cross-query
   document residuals resumed only above one split.
3. **EPIC (arXiv 2024; ICML 2025) / APE (arXiv 2025).** Independently precompute
   reusable per-layer KV for modular contexts and repair positional/attention
   incompatibilities. They are closer deployment alternatives than generic RAG, but
   store a layer-wise KV object rather than one depth-\(j\) residual/token.
4. **KV Packet (arXiv 2026-04-14).** Context-independent reusable KV with
   distillation-trained adapters and no document recomputation. It is close in its
   learned offline Writer/standard Reader boundary and is within the novelty cutoff.
5. **Cartridges (arXiv 2025; ICLR 2026).** Offline learned reusable corpus-specific KV
   representations amortized across queries. It is close in workload and
   self-distillation motivation, but the persistent object and training objective are
   substantially different.

**Concurrent, not novelty-reducing:** Cartridges at Scale (2026-06-03) and SemPIC
(2026-07-30). SemPIC is particularly close in using a learned offline Writer and an
unchanged Reader for reusable document KV, but it appeared after the three-month
cutoff.

## Novelty judgment

I did not find an earlier paper that exactly combines: one persistent residual per
document token at a chosen depth, chunk-local independent Write, bounded retrieval,
direct execution of only \([j:L)\), and a matched \(j=0\) endpoint explicitly
measuring the incremental depth-reuse trade-off. Thus the *specific empirical design
point and audit* are novel. The broader ideas—reusable intermediate representations,
residual sufficiency, offline learned document objects, and modular context caches—are
not. Novelty is therefore **incremental but real**, consistent with an excitement
score of 3.0 rather than 4.0.

# Review-Process Self-Check

- [x] Used only the frozen PDF, frozen source directory, strict template, and public
  bibliographic/novelty sources; did not read other reviews, history files, TODOs,
  status files, alternate drafts, or reviewer-calibration material.
- [x] Completed two passes including all appendices.
- [x] Built claims C1–C6 and specified the minimum sufficient experiment before
  judging the paper's evidence.
- [x] Inspected both figures and all 36 numbered tables, including split minipage
  tables.
- [x] Recomputed at least five abstract numbers: storage ratio/bytes, read cap,
  speedup, RULER gap, equal-latency differences, overlap gains, and LoCoMo gap.
- [x] Checked equations, boundary cases, selectors, baselines, metrics, sample counts,
  seeds/statistics, claim scope, compute, and reproducibility.
- [x] Audited every actually cited `main.bbl` entry and checked eight load-bearing
  citation–claim matches.
- [x] Ran four novelty searches and applied the May 4, 2026 contemporaneous-work rule.
- [x] Checked page/style, exact Limitations, anonymity, unresolved references,
  placeholders, prompt injection, ethics, and desk risks.
- [x] Mechanically grepped every weakness quote against the frozen source; every quote
  is at most 25 words.
- [x] Mechanically verified each “missing/lacks/no table/no matched baseline/no clean
  seeds” assertion across all frozen source files and appendices.
- [x] Reassessed the equal-latency audit from this version alone. I give full credit
  for the protocol visible here and do not deduct for any presumed issue in another
  version; W1 is based solely on the displayed current statistical unit.
- [x] Network checks were bounded; no endless retries were used. No citation is marked
  “Not found” merely because of a transient network failure.
