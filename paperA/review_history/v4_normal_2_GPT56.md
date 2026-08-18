---
review_mode: normal
soundness: 4.0
excitement: 3.5
overall: 3.5
confidence: 4.0
reproducibility: 4.0
---

## Summary

This paper studies a useful and unusually well-scoped systems question: for a
document representation reused across queries, how much transformer depth
should be prepaid? CoMem independently writes each document chunk through
layers \([0,j)\), stores one residual vector \(h_j\) per token, retrieves a
bounded top-\(k\) set, and resumes layers \([j,L)\) over the packed residuals
and query. The matched \(j=0\) endpoint retrieves the same chunks but replays
their token IDs through all layers.

The central Qwen3-8B result is an internally controlled quality/latency/storage
frontier: with the same selected chunks, order, examples, mask, adapter, and
roughly 6.5k-token read pack, \(j=12\) changes isolated model Read latency from
931.9 ms to 664.4 ms (1.403x) while lowering a 15-cell, 1,500-example RULER-B
macro from 99.19 to 96.07. The paired quality loss is 3.12 points with 95% CI
[2.36, 3.93]. The representation costs 8,192 bytes/token, versus about 4--8
bytes/token for raw IDs and 144 KiB/token for full bf16 KV under the paper's
Qwen3 configuration. Measured write amortization is approximately 8--11
queries at 32k and 26--28 at 128k for the retained serving settings.

The paper also reports an important negative result: under its equal-online-
latency calibration, raw replay with top-10 scores 64.78 versus 53.22 for
CoMem with top-12. Finally, a paired synthetic multikey factorization suggests
that missing lower-layer document context is an important local failure mode:
document-contextual writing changes 92.5 to 100.0, while a deployable
32-token left overlap reaches 98.5 without changing persistent bytes or
per-query Read length. The paper appropriately limits this diagnosis to the
tested cohort.

My assessment is positive but below a clear main-conference accept. The
matched depth control, transparent negative result, storage/amortization
accounting, and extensive appendix make the paper technically credible.
However, the practical comparative claim remains unresolved because there is
no same-backbone, same-hardware end-to-end comparison with the nearest PIC,
chunk-KV-repair, or learned modular-cache systems. In addition, the promising
overlap repair is not incorporated into the natural-task or end-to-end
frontier. I therefore view the present contribution as strong Findings-level
work, with a plausible path to ACL main if those two gaps are closed.

## Claims and evidence map

- **C1 — CoMem defines a distinct depth-reuse interface.** One depth-\(j\)
  residual is stored per token, selected residuals are packed with the query,
  and only layers \([j,L)\) run online.  
  **Evidence:** Figure 1; Section 4; exact split-forward self-tests in
  Appendix A.7.1, including zero reported logit difference at several Hy3
  split points.  
  **Assessment:** Supported as an implementation/interface claim.

- **C2 — Reusing the lower 12 layers yields a measured incremental Read
  speedup, but not quality preservation.**  
  **Evidence:** Table 2 and Appendix Table 28: 931.9 to 664.4 ms, 1.403x;
  RULER-B 99.19 to 96.07; paired bootstrap CI [2.36, 3.93] and McNemar
  \(p=8.79\times10^{-24}\). The arms share examples, retrieved packs, order,
  and LoRA.  
  **Assessment:** Strongly supported for the stated isolated Read boundary,
  hardware, backbone, pack, and adapter.

- **C3 — The single-residual object reduces persistent state relative to
  full layer-wise KV, at a substantial cost relative to text.**  
  **Evidence:** Equation (1), \(n_q/(2Ln_{kv})=1/18\) for Qwen3-8B; 8 KiB
  versus 144 KiB/token in bf16; about 1 GiB at 128k.  
  **Assessment:** Algebraically correct under the stated common-dtype,
  standard-attention assumptions. It is not a universal ratio for MLA,
  quantized, or otherwise compressed KV systems.

- **C4 — The one-time Write is beneficial only under repeated reuse.**  
  **Evidence:** Table 3 reports measured \(Q^\star\) values across 32k/128k,
  CPU/GPU placement, and selected generation lengths; Table 30 separately
  measures storage tiers.  
  **Assessment:** Supported for the retained cells and timing boundaries.
  Generalization to production load, batching, and tail latency is explicitly
  not established.

- **C5 — CoMem is inferior to raw replay at matched online latency in the
  tested diagnostic setting.**  
  **Evidence:** Table 4: 64.78 versus 53.22, paired-bootstrap CI
  [-14.44, -8.67], with top-\(k\) chosen on a disjoint latency-only
  calibration split.  
  **Assessment:** Supported and decision-relevant, although the displayed
  paper does not fully enumerate the mixed cohort or raw latency values.

- **C6 — Missing lower-layer document context explains a major part of one
  tested multikey gap, and overlap is a local repair.**  
  **Evidence:** Tables 5, 6, and 29: chunk-local/local-position 92.5,
  document-contextual 100.0, position remapping alone 88.0, and overlap
  \(w=32/64/128\) giving 98.5/98.5/99.0.  
  **Assessment:** The factorization supports a bounded Write-side diagnosis,
  not a complete causal decomposition. The paper states this limitation
  correctly.

- **C7 — A small self-distillation adapter materially improves residual
  readout.**  
  **Evidence:** same-\(j=12\) RULER cells in Table 10 and broader LoCoMo/
  BABILong controls in Table 8; the separately trained depth curve in Table 9;
  an analogous Hy3 effect in Table 31.  
  **Assessment:** Supported. The exact depth curve is a deployment curve
  rather than a compute-matched causal depth ablation, as adapter spans and
  parameter counts differ.

- **C8 — Model-side Read work is bounded with respect to stored-context
  length, but retrieval/index/store costs are not.**  
  **Evidence:** the Read-length formula, Table 23 up to a 4M-token store,
  Table 30 at 16M tokens, and Hy3 Table 32 through 256k. BM25 time grows
  roughly linearly while the model pack remains about 4.3--6.5k tokens.  
  **Assessment:** Supported with the important qualification that top-\(k\)
  imposes an evidence ceiling and does not make the complete system
  constant-cost.

## Strengths

1. **The central experiment is genuinely matched and answers a narrow causal
   systems question.** Section 5.1 and Table 2 hold the evidence pack, order,
   mask, examples, and LoRA fixed, changing only replay start. This cleanly
   separates the incremental effect of depth reuse from the much larger
   benefit of bounded selection.

2. **The paper reports unfavorable evidence rather than hiding it.** The
   equal-latency result and the explicit statement, “This is not
   quality-preserving acceleration,” substantially increase trust. The paper
   does not convert the 64.9x select-first operating point into an unsupported
   depth-only or end-to-end claim.

3. **Systems accounting is unusually transparent.** The paper distinguishes
   isolated model Read, store-ready online prefill, Write-inclusive pipeline,
   retrieval/indexing, and external-store I/O. It reports persistent bytes,
   placement, amortization, unavailable cells, and a storage-tier
   microbenchmark instead of collapsing these into one favorable latency
   number.

4. **The statistical treatment is mostly appropriate to the data structure.**
   The main paired RULER gap has paired bootstrap and McNemar analyses.
   LoCoMo includes both item and conversation-cluster bootstrap intervals and
   correctly avoids treating 1,540 nested questions as independent. Sample
   counts, cohort definitions, seeds, and aggregation rules are generally
   explicit.

5. **The mechanism section is careful about scope.** The continuous-prefix
   control is called a fidelity ceiling, the 2x2 context/position result
   acknowledges interaction, and overlap is described as a local engineering
   hypothesis rather than a universal solution.

6. **Reproducibility documentation is strong for a paper-only review.** The
   appendix specifies the backbone revision, split, positions, masks, BM25
   parameters, generation, optimizer, data, hardware, adapter parameter count
   and SHA-256, benchmark supports, scorers, and timing boundaries. The paper
   also reports contamination concerns and removes a book-quality comparison
   rather than overclaiming it.

7. **The paper is readable and well organized despite a large appendix.**
   Figure 1 communicates the interface effectively, and the main eight pages
   preserve a coherent story: matched frontier, amortization/negative result,
   then bounded diagnosis.

## Weaknesses

### W1. No matched comparison to the nearest reusable-cache systems — **Major**

- **Location / exact quote:** Conclusion, PDF p.6, lines 351--353:
  “same-backbone, end-to-end comparison with PIC and learned modular-KV
  systems”
- **Problem:** The paper correctly identifies the decisive missing experiment,
  but it is also the main obstacle to judging practical significance. The
  internal \(j=0\) control establishes the cost of depth reuse relative to raw
  replay, not whether a single residual is a better reusable object than
  per-layer KV plus repair, learned modular KV, or another activation
  checkpoint under a shared deployment budget.
- **Affected claim / criterion:** This limits excitement and practical
  validation of C1--C4. It prevents a main-conference-level conclusion about
  where CoMem sits on the actual reusable-context Pareto frontier, even though
  the paper appropriately avoids a superiority claim.
- **Sufficient remedy:** Implement at least one strongest feasible
  same-backbone neighbor (e.g., a PIC/chunk-KV-repair method) and one learned
  modular-cache neighbor, or obtain official implementations, then compare on
  identical Qwen3 weights, retrieval IDs, natural tasks, hardware, storage
  tier, dtype, quality target, and end-to-end TTFT/throughput. Report persistent
  bytes and one-time build/training cost. If full reproduction is genuinely
  infeasible, a controlled common-interface approximation with a detailed
  deviation table would still materially strengthen the paper.
- **Mechanical confirmation:** The source contains only taxonomic Table 1 and
  explicitly says no matched same-backbone implementation is provided.

### W2. The proposed repair is not evaluated on the claims-bearing natural
task frontier — **Major**

- **Location / exact quote:** Limitations, PDF p.6, lines 403--405:
  “Overlap-Write is tested on a focused multikey diagnostic”
- **Problem:** Independent chunk writing is the deployed method's identified
  quality bottleneck, and overlap recovers 6--6.5 points on that diagnostic.
  Yet no RULER-wide, LongEval, LoCoMo, LongBench, or BABILong result measures
  whether overlap improves the actual operating point, nor is its measured
  Write overhead included in the amortization/equal-latency frontier.
- **Affected claim / criterion:** C6 is sound as a local diagnosis, but the
  paper leaves unresolved whether its own repair changes the decision-relevant
  conclusion in C2, C4, or C5. This is a minimal support experiment because
  the repair directly targets the paper's measured failure mode.
- **Sufficient remedy:** Evaluate at least \(w=32\) on the matched RULER-B
  frontier and two natural suites with different evidence structure, including
  LoCoMo or LongBench. Re-measure Write time, break-even, and the equal-latency
  comparison using the repaired writer. Report paired deltas and retain a
  negative conclusion if the repair does not transfer.
- **Mechanical confirmation:** All overlap numbers occur in Tables 5--6 on
  pooled 8k/16k RULER multikey; the source and Limitations explicitly state
  that no natural-task overlap result or repaired end-to-end frontier exists.

### W3. The equal-latency negative result is under-specified in the paper —
**Minor**

- **Location / exact quote:** Table 4 caption, PDF p.5:
  “a mixed diagnostic cohort”
- **Problem:** This is framed as the paper's “decision-relevant negative
  result,” but the PDF does not list the cohort's constituent tasks, number of
  examples, score aggregation, absolute latencies, calibration candidates, or
  hardware/timing boundary adjacent to the table. “within ±5%” is not enough
  to independently interpret the comparison from the paper.
- **Affected claim / criterion:** C5's numerical result may be valid, but its
  reproducibility and external interpretability are weaker than those of the
  central RULER experiment.
- **Sufficient remedy:** Add a compact appendix table giving cohort
  composition and \(n\), per-task scores, aggregate definition, calibration
  grid, actual median/p10/p90 latency for both arms, hardware, generation
  length, included/excluded components, and the calibration/evaluation split
  sizes.
- **Mechanical confirmation:** Searches for 64.78, 53.22, “mixed diagnostic,”
  and Table 4 found no such full specification elsewhere in the provided
  source.

### W4. The primary LoCoMo headline depends on an undated external judge —
**Minor**

- **Location / exact quote:** Appendix B.2, PDF p.21, lines 1007--1010:
  “The endpoint does not expose a dated model snapshot.”
- **Problem:** The protocol is documented and an independent 200-item judge
  audit preserves ordering, but exact regeneration of the 38.27/34.59 headline
  is not guaranteed when the provider silently changes the model behind
  `gpt-4o`. Absolute calibration also changes substantially under DeepSeek-V3.
- **Affected claim / criterion:** Reproducibility of C7's LoCoMo comparison,
  not the matched RULER frontier.
- **Sufficient remedy:** Release all item-level prompts, raw judge responses,
  parsed decisions, and hashes as promised; add a deterministic open-weight
  judge over all 1,540 answerable items or a sufficiently powered human audit.
  Treat the frozen/open result as primary or co-primary and the live API result
  as supporting evidence.
- **Mechanical confirmation:** The appendix explicitly records an undated
  endpoint and reports only a 200-item independent-judge subset.

### W5. Reproducibility is strong but not fully closed on compute and training
variance — **Minor**

- **Location / exact quote:** Limitations, PDF p.6, lines 369--371:
  “The principal adapter is trained once”
- **Problem:** Two extra adapters are evaluated, which is useful, but they use
  effective batch 3 rather than 8. Total GPU-hours across exploratory work and
  peak training memory are not recorded. Thus the paper does not provide a
  controlled three-seed estimate for the exact flagship recipe or complete
  project compute accounting.
- **Affected claim / criterion:** Reproducibility and uncertainty around the
  selected \(j=12\) adapter, rather than correctness of the paired evaluation
  conditional on that adapter.
- **Sufficient remedy:** Train two additional exact-recipe seeds with global
  batch 8 and report the central RULER/LoCoMo frontier, variance, and selection
  rule. Record peak memory and separate final-run compute from total
  experimental compute prospectively. This need not become an exhaustive
  hyperparameter study.
- **Mechanical confirmation:** Appendix A.6 states the added seeds use a
  different effective batch; Appendix A.4 states that total exploratory
  GPU-hours and training peak memory were not consistently logged.

## Questions for the authors

1. What exactly constitutes the “mixed diagnostic cohort” in Table 4, how many
   examples are in it, and what are the absolute latency distributions for the
   top-10 and top-12 arms?

2. If \(w=32\) overlap were included, what are the measured Write time and
   \(Q^\star\) at 32k/128k? Does the equal-latency result remain negative after
   paying this extra Write cost?

3. Can the authors provide one common-backbone comparison against a
   PIC/chunk-KV method using exactly the same selected chunks and quality
   cohort? If not, which implementation obstacle prevents it?

4. Does the same-adapter matched raw-replay control use LoRA activations in all
   36 layers exactly as CoMem does in layers 12--35, with the lower layers
   unchanged? Please make the adapter placement in the \(j=0\) arm explicit in
   the main table.

5. For the three training seeds, what are the seed-wise 15-cell RULER-B macros
   and LoCoMo scores? Was the flagship chosen before or after seeing benchmark
   results?

6. Are item-level Table 4 predictions, all LoCoMo judge outputs, and the three
   process-level timing logs included in the anonymous artifact, or only hashes?

## Suggestions

- Promote the matched \(j=0\) versus \(j=12\) frontier and the equal-latency
  negative result even more clearly; move heterogeneous external leaderboard
  tables to a secondary appendix subsection.
- Add a single “timing boundary matrix” listing every table and whether it
  includes selection, index build, store fetch, Write, prefill, and decode.
- In Equation (1), explicitly say “uncompressed standard per-layer KV” and
  note that the ratio changes for MLA, quantization, cache compression, or
  mixed precision.
- Report per-task Table 4 results and a repaired \(w=32\) operating point.
- Include the exact public artifact access procedure in the submission form;
  no artifact link is visible in the PDF, although the paper repeatedly says
  files are released.
- Correct bibliographic metadata where an archival publication is readily
  available (e.g., LongBench has an ACL 2024 version rather than only the 2023
  arXiv citation). This is presentation-level and does not affect the review
  score.

## Score rationale

### Soundness: 4.0 / 5.0

The main matched result is well controlled, statistically supported, and
careful about timing boundaries. Formula (1) checks out for the stated Qwen3
architecture: \(32/(2\cdot36\cdot8)=1/18\), giving 8 KiB versus 144 KiB/token
in bf16. The paper also distinguishes causal depth evidence from combined
bounded-selection operating points. Soundness is not 4.5--5 because the
equal-latency cohort is insufficiently specified in the paper, exact
training-seed replication is incomplete, and the local repair is not tested on
the main natural-task frontier.

### Excitement: 3.5 / 5.0

Making split depth an explicit, measured cross-query reuse axis is interesting,
and the honest negative result plus bounded mechanism diagnosis are valuable.
However, reusable intermediate representations, activation checkpoints,
position-independent caches, and learned modular KV objects are established
lines of work. The novelty is the particular single-residual object and the
matched depth/storage/quality frontier, not the general idea of reusable
document state. Excitement is capped by the absence of a controlled comparison
to the closest systems.

### Overall: 3.5 / 5.0

This is a solid **Findings-level** paper: technically careful, informative even
where results are negative, and substantially more transparent than many
systems papers. I lean above borderline because the matched experiment answers
a real question and the authors constrain their claims. I stop below 4.0
(ACL main level) because the paper has not yet established the practical value
of its representation against the nearest reusable-cache alternatives, and it
does not evaluate its proposed repair on the claims-bearing end-to-end
frontier.

### Confidence: 4.0 / 5.0

I read the complete 21-page PDF twice, including appendices, inspected the
provided TeX, all figures/tables, all 43 `main.bbl` entries, and relevant
closest-work abstracts. Confidence is not 4.5--5 because I did not execute the
unprovided experimental artifact or reproduce GPU measurements, and a few
publisher endpoints were inaccessible.

### Reproducibility: 4.0 / 5.0

The paper gives unusually complete model, adapter, optimizer, data, metric,
sample-count, seed, hardware, store, and timing details, plus an adapter hash.
The main deductions could likely be reproduced with the promised artifact.
The score is below 4.5 because the live LoCoMo judge lacks a dated snapshot,
Table 4 is under-documented, the exact flagship training recipe has only one
seed, peak memory/total experimental compute are missing, and the artifact was
not part of the permitted review snapshot.

## Desk, format, anonymity, and ethics audit

- **Main-text page limit:** Pass under the standard ACL long-paper convention.
  The substantive main paper ends on PDF p.6; references occupy pp.7--9 and
  appendices pp.10--21. The source itself describes an eight-page main-paper
  budget and does not use it to hide main claims in references.
- **Limitations:** Pass. A dedicated unnumbered Limitations section begins on
  p.6 and discusses backbone/selector/language scope, missing matched systems,
  amortization assumptions, storage growth, update invalidation, untested
  overlap transfer, seeds, extrapolation, and serving-tail limitations.
- **Anonymity:** Pass based on the permitted snapshot. The PDF says
  “Anonymous ACL submission”; no author names, affiliations, emails, repository
  links, or identifying PDF metadata were found. Source path names were not
  treated as paper evidence.
- **ACL format:** Pass. The source uses `\usepackage[review]{acl}`, 11pt article
  mode, line numbers, two columns, A4 output, and embedded fonts. No obvious
  margin, font-size, or float abuse was observed. Some appendix tables use
  `\scriptsize`, but remain legible and do not appear to evade the main-page
  limit.
- **Unresolved references/placeholders:** Pass. Mechanical search found no
  `TODO`, `FIXME`, `TBD`, `XXX`, `??`, missing labels, or duplicate labels.
  The compiled PDF shows no unresolved citation/reference markers.
- **Injection/hidden/reviewer-manipulation text:** Pass. Search of authored TeX
  found no white text, phantom text, negative-spacing concealment,
  zero-size text, conditional hidden blocks, or instructions to reviewers.
  The PDF has no JavaScript, forms, annotations, or embedded files. Included
  figure PDFs expose only visible figure text.
- **Abstract/table number consistency:** Pass for the headline numbers:
  931.9/664.4, 1.403x, 99.19/96.07, 8--11, 26--28, 64.78/53.22,
  92.5/100.0/88.0/98.5, 8,192 B/token, and 1/18 all match the relevant tables
  or derivations.
- **Ethics:** Adequate. The Ethical Considerations section covers inherited
  generation risks, sensitive-memory retrieval, inversion/membership concerns,
  access control, deletion, encryption/auditing, energy, licenses, and the lack
  of new human-subject collection. No ethics-review escalation is warranted.
  The main deployment concern is that residual tensors should be protected as
  sensitive source-derived data; the paper states this explicitly.
- **Data/licensing:** The appendix identifies major artifact licenses and says
  restricted text/model weights/API responses are not redistributed. I did not
  independently verify every upstream license, so those details remain
  partially unverifiable from the paper snapshot alone.

## Citation audit

### `main.bbl` entry-by-entry verification

“Verified” means that title/identifier and basic bibliographic existence were
confirmed through an arXiv API record, ACL Anthology, DOI metadata/redirect, or
the cited official model page. “Partial” means the work exists and the title is
right, but the displayed entry uses a preprint rather than a later archival
record or venue metadata could not be fully checked. “Unverifiable” is reserved
for an endpoint that did not provide enough information during this audit.

| # | Key / work | Status | Audit note |
|---:|---|---|---|
| 1 | `cachecraft` | Verified | DOI 10.1145/3725273 resolves; title matches. ACM full page returned 403, but DOI/OpenAlex metadata agree. |
| 2 | `longbench` | Partial | arXiv:2308.14508 exists and title matches; an ACL 2024 archival version also exists, so the entry is not the best available citation. |
| 3 | `pyramidkv` | Verified | arXiv:2406.02069 exists; title/date match. |
| 4 | `kvpacket` | Verified | arXiv:2604.13226, submitted 2026-04-14; title matches. |
| 5 | `cartridgesbase` | Verified | arXiv:2506.06266; title/authorship match. |
| 6 | `hcache` | Verified | DOI 10.1145/3689031.3696072 and arXiv:2410.05004 match the cited EuroSys work. |
| 7 | `llama3` | Verified | arXiv:2407.21783 exists; title matches. |
| 8 | `cartridges` | Verified, concurrent | arXiv:2606.04557, submitted 2026-06-03. |
| 9 | `distillation` | Verified | arXiv:1503.02531 exists; title/authors match. |
| 10 | `ruler` | Verified | arXiv:2404.06654 exists; title matches. |
| 11 | `lora` | Verified | ICLR paper / arXiv:2106.09685 exists; title matches. |
| 12 | `epic` | Verified | arXiv:2410.15332; cited as ICML 2025, consistent with available metadata. |
| 13 | `ragcache` | Verified | arXiv:2404.12457 exists; title matches. |
| 14 | `babilong` | Verified | NeurIPS 2024 record and arXiv:2406.10149 exist; title matches. |
| 15 | `rag` | Verified | NeurIPS 2020 paper / arXiv:2005.11401; title/authors match. |
| 16 | `longchat` | Verified | Official LMSYS blog page resolves; title matches. |
| 17 | `snapkv` | Verified | arXiv:2404.14469 and NeurIPS record exist; title matches. |
| 18 | `ilre` | Verified | arXiv:2508.17892 exists; title matches. |
| 19 | `readonce` | Verified | ACL Anthology 2021.acl-long.554; title/authors match. |
| 20 | `minicache` | Verified | arXiv:2405.14366 exists; title matches. |
| 21 | `turborag` | Verified | arXiv:2410.07590 exists; title/authors match. |
| 22 | `locomo` | Partial | arXiv:2402.17753 exists and title matches; ACL 2024 archival version is available. |
| 23 | `xccache` | Verified | ACL Anthology 2024.findings-emnlp.896; title/authors match. |
| 24 | `kvdirect` | Verified | arXiv:2603.19664, submitted 2026-03-20; title matches. |
| 25 | `pg19` | Verified | arXiv:1911.05507 is the Compressive Transformers paper introducing PG-19. |
| 26 | `bm25` | Verified | DOI 10.1561/1500000019 metadata match; publisher endpoint itself returned 403. |
| 27 | `embeddingrecycling` | Verified | ACL Anthology 2023.findings-eacl.145; title/authors match. |
| 28 | `gemfilter` | Verified | arXiv:2409.17422 exists; title matches. |
| 29 | `reform` | Verified | arXiv:2506.01215 exists; title matches. |
| 30 | `lloco` | Partial | arXiv:2404.07979 exists; EMNLP 2024 archival version is available. |
| 31 | `hunyuan` | Verified | Official Hugging Face model page resolves and describes Hy3. |
| 32 | `fusionrag` | Verified | arXiv:2601.12904 exists; title matches. |
| 33 | `mepic` | Verified | arXiv:2512.16822 exists; title matches. |
| 34 | `longmem` | Verified | NeurIPS 2023 / arXiv:2306.07174; title matches. |
| 35 | `memoryllm` | Verified | arXiv:2402.04624 exists; title matches. |
| 36 | `infllm` | Verified | arXiv:2402.04617 and NeurIPS record exist; title matches. |
| 37 | `streamingllm` | Verified | ICLR 2024 / arXiv:2309.17453; title matches. |
| 38 | `sempic` | Verified, concurrent | arXiv:2607.28069, submitted 2026-07-30. |
| 39 | `xu2024retrievallong` | Verified | ICLR 2024 paper / arXiv:2310.03025; title/authors match. |
| 40 | `qwen3` | Verified | arXiv:2505.09388 exists; title matches. |
| 41 | `ape` | Verified | arXiv:2502.05431 exists; title matches. |
| 42 | `cacheblend` | Verified | arXiv:2405.16444 exists; title matches. |
| 43 | `h2o` | Verified | NeurIPS 2023 / arXiv:2306.14048; title matches. |

All 43 `main.bbl` entries are cited in the paper; no cited key is missing from
the bibliography and no `main.bbl` entry is unused.

### Citation-to-claim checks

| Paper claim | Cited evidence checked | Verdict |
|---|---|---|
| CacheBlend/TurboRAG/Cache-Craft precompute or reuse chunk KV and address context-dependent composition. | Abstracts for CacheBlend (selective recomputation), TurboRAG (offline per-chunk KV plus positional/mask handling), and Cache-Craft metadata. | Supported. |
| EPIC/MEPIC are PIC systems that make independently cached chunks reusable through position/boundary repair or selective recomputation. | EPIC and MEPIC abstracts. | Supported, though MEPIC emphasizes page/block sharing and block-level recomputation more than the prose suggests. |
| APE uses parallel context encoding and attention realignment. | APE abstract: independently cached contexts plus shared prefix, attention temperature, and scaling. | Supported. |
| KV Packet uses independently compiled KV with lightweight boundary/soft-token adapters and self-supervised distillation. | KV Packet abstract. | Supported. |
| Cartridges and Cartridges at Scale learn reusable per-corpus/per-document KV objects; CAS retrieves modular objects at scale. | Both abstracts. | Supported. `cartridges` is concurrent under the three-month rule. |
| ReadOnce Transformers and Embedding Recycling cache reusable intermediate representations and adapt later computation. | ACL/Anthology abstracts. | Supported. |
| HCache restores evicted state from intermediate activations; KV-Direct reconstructs exact layer-wise KV from residuals. | HCache and KV-Direct abstracts. | Supported. |
| ILRe/REFORM/GemFilter select or gather tokens before later recomputation rather than persist CoMem-style states across queries. | Their abstracts. | Broadly supported. ILRe's offline intermediate-layer processing is especially close and deserves explicit comparison, but its selected object and online continuation differ. |

## Novelty search and closest-work analysis

### Search protocol

I froze novelty at **2026-08-03** and ran five concept searches over arXiv/API
metadata and titles/abstracts:

1. persistent/intermediate/residual/cache/LLM;
2. activation replay + long-context LLM;
3. residual retrieval + long-context language model;
4. position-independent caching + LLM;
5. modular document KV cache.

I then inspected the cited nearest works and additional search hits. Following
the requested three-month rule, work first posted **after 2026-05-03** is
treated as concurrent rather than novelty-defeating. This includes Cartridges
at Scale (2026-06-03), HYPIC (2026-07-01), C2KV (2026-07-20), SemPIC
(2026-07-30), and the arXiv item titled *Understanding Is Done Early...*
(2026-07-30). The latter appears to describe an earlier public version of this
same project and is not treated as independent prior art.

### Closest works

| Work | Similarity to CoMem | Material distinction |
|---|---|---|
| **HCache** (2025) | Stores intermediate activations and resumes computation to restore LLM state, establishing activation-checkpoint reuse as a systems primitive. | Targets restoration of evicted conversational/RAG state, not query-conditioned selection of reusable document chunks or a tunable quality/depth frontier. |
| **KV-Direct** (2026-03-20) | Stores one residual per token instead of all layer-wise KV and reconstructs KV exactly, making it the closest stored-object precedent. | Primarily a bounded-memory state-reconstruction method. CoMem stores the residual at a chosen intermediate depth and directly runs only the suffix on an independently written, retrieved pack; this intentionally permits a quality/depth trade-off. |
| **ReadOnce Transformers / Embedding Recycling** (2021/2023) | Reuse intermediate document representations and adapt later layers, closely preceding the high-level “read once, reuse later” idea. | Encoder/representation-recycling settings and task-specific models rather than decoder-only, query-conditioned bounded retrieval with an explicit split-depth/storage/latency frontier. |
| **EPIC / CacheBlend / TurboRAG** (2024--2025) | Reuse independently computed document/chunk state across non-prefix queries and repair context or positions. They are the nearest serving workload. | Persist per-layer KV and repair/fuse it; CoMem persists a single residual/token and pays suffix computation online. |
| **KV Packet** (2026-04-14) | Learned, immutable, context-independent document objects with distillation and adapters; closest learned pre-cutoff competitor. | Persists per-layer KV packets and aims for recomputation-free reuse; CoMem persists one tunable-depth residual and explicitly explores online suffix recomputation. |
| **Cartridges** (2025) | Offline distilled reusable KV representations amortized over repeated corpus queries. | Corpus-level learned KV objects, synthetic self-study, and much stronger offline object training; not a direct split-depth control. |
| **ILRe / REFORM / GemFilter** | Use intermediate layers to identify/select a bounded subset for later computation. | Primarily within-request token selection/compression; the selected intermediate state is not the same persistent cross-query document object. |
| **Cartridges at Scale / SemPIC / C2KV** (concurrent) | Modular learned per-document KV, writer/reader distillation, composition, and persistent-store scaling overlap strongly with CoMem's application. | All postdate 2026-05-03 and are therefore concurrent. They nevertheless make the missing common-backbone comparison more important for final positioning. |

### Novelty conclusion

The broad concepts—reusable document representations, cached intermediate
activations, PIC/modular KV, and early-layer token selection—are not new.
The credible novelty is narrower:

1. treating **split depth itself** as an explicit cross-query systems axis;
2. using **one persistent depth-\(j\) residual per selected token** followed by
   direct suffix execution;
3. measuring a **same-evidence, same-adapter depth/quality/latency/storage
   frontier**, including amortization and an equal-latency negative result;
4. providing a bounded context/position/overlap diagnosis for independent
   residual writing.

I found no pre-2026-05-03 work that duplicates this complete combination.
The novelty is therefore sufficient for publication, but incremental enough
that controlled comparisons to KV-Direct, HCache, PIC, and KV Packet are
important to establish impact.

## Review-process self-check

- [x] Used only the designated PDF, designated source snapshot, review
  template, and public literature endpoints; did not inspect other reviews,
  score histories, TODO/status/current files.
- [x] Completed two passes over all 21 PDF pages, including both appendices.
- [x] Inspected every displayed figure and table (Figures 1--2; Tables 1--34).
- [x] Mapped the principal claims to evidence and checked claim scale against
  timing boundaries and benchmark support.
- [x] Checked method equations, boundary cases \(j=0\) and deeper splits,
  persistent-byte arithmetic, generation caching, selector bounds, and
  overlap semantics.
- [x] Audited baselines, benchmark cohorts, metrics, sample sizes, seeds,
  statistics, compute, contamination discussion, and reproducibility details.
- [x] Mechanically searched for anonymity leaks, hidden/manipulative text,
  placeholders, unresolved references, and duplicate/missing labels.
- [x] Verified all 43 `main.bbl` entries to the extent public endpoints
  allowed, and checked eight citation-to-claim groups.
- [x] Ran five novelty searches, compared eight closest-work groups, and
  applied the 2026-05-03 concurrent-work cutoff.
- [x] Each weakness includes location, an exact quote of at most 25 words,
  problem, affected claim/criterion, sufficient remedy, severity, and
  mechanical confirmation.
- [x] Did not request experiments unrelated to the paper's own claims.
- [x] Independently set Soundness and Excitement before choosing Overall;
  calibrated 4.0 as ACL main and 3.0 as Findings.
- [x] Network-access limitations are marked transparently: ACM/publisher
  403 responses were cross-checked through DOI/arXiv/OpenAlex metadata rather
  than silently assumed; no unresolved bibliographic item was found.
