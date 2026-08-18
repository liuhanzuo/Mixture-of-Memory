review_mode: normal
soundness: 3.5
excitement: 3.5
overall: 3.5
confidence: 4.0
reproducibility: 3.5

## Summary and claim/evidence map

This paper studies repeated-query inference over a stable document collection. CoMem
writes each chunk through a chosen split layer \(j\), stores one residual vector per
token, retrieves a bounded set of chunks, and resumes only layers \([j{:}L)\) with
the query. A matched \(j=0\) path replays the same retrieved pack from token IDs.
The flagship Qwen3-8B system uses \(j=12\) and a rank-32 suffix LoRA trained by
self-distillation.

My reconstruction of the main claims is:

- **C1 — Depth is a useful reusable-context serving axis.** Evidence: the matched
  \(j=0\) versus \(j=12\) RULER-B comparison in Table 2 / Table 35, plus the
  separately trained multi-depth curve in Table 12.
- **C2 — Prepaying 12 layers trades 3.12 RULER points for a 1.403x selected-pack
  Read speedup.** Evidence: 1,500 paired examples, paired-bootstrap CI
  \([2.36,3.93]\), McNemar counts, and three process-level latency measurements.
- **C3 — The composed method can substantially reduce online prefill and can
  amortize its Write.** Evidence: the 64.9x store-ready H20 point, the separate
  2.74x same-adapter Write-inclusive L20A point, the break-even grid, and the
  store-I/O microbenchmark.
- **C4 — A single residual is much smaller than full layer-wise KV.** Evidence:
  Equation 1 and the Qwen3-8B calculation of 8,192 versus 147,456 bytes/token.
- **C5 — Much of the measured quality loss is caused by independently written
  lower-layer states lacking document context.** Evidence: the continuous-prefix
  oracle, the \(2\times2\) context/position table, and the overlap-Write sweep on
  8k/16k RULER multikey.
- **C6 — Full-depth independently prepared chunk KV is not sufficient in the
  authors' matched implementation.** Evidence: the Qwen3 CacheBlend-style
  \(r=0\ldots0.18\) comparison and an \(r=1\) correctness self-test.
- **C7 — Any deployment advantage is selector dependent rather than universal.**
  Evidence: at calibrated latency, BM25 raw replay beats CoMem by 11.56 points,
  while the frozen-BGE comparison is unresolved under hierarchical resampling.
- **C8 — The implementation is portable beyond the flagship model.** Evidence:
  exploratory Qwen-size sweeps and a Hy3 MoE split-forward/distillation study;
  these support implementation portability, not a replicated quality frontier.

## Strengths

1. **The central matched control is unusually careful.** The statement that the
   \(j=0\) and \(j=12\) arms use the same selected chunk IDs, order, sink, mask,
   examples, and LoRA is exactly the right way to isolate the incremental depth
   effect. The paper also resists conflating this 1.403x result with the much
   larger composed 64.9x bounded-selection result. This separation of timing
   boundaries materially improves the scientific value.

2. **The paper reports negative and boundary results rather than hiding them.**
   In particular, it states that Read+decode is only about 1.07--1.09x, that
   LongEval drops from 97.2 to 69.0, that raw replay robustly wins the BM25
   equal-latency comparison, that residual storage is much larger than text, and
   that one-off/low-reuse documents should favor replay. These disclosures make
   the claims substantially more credible.

3. **The Write-interface diagnosis is technically insightful.** The
   continuous-prefix oracle recovering 99.19, followed by the context/position
   factorization and 32-token overlap result, gives a useful mechanistic account
   of why independently written residuals fail. The paper appropriately limits
   this attribution to the tested multikey cohort.

4. **The systems accounting is broad and mostly explicit.** Persistent bytes,
   one-time Write, online Read, decode, external storage tiers, index behavior,
   peak memory, and empirical break-even are all addressed. Table 4's measured
   crossover grid and Table 36's off-GPU fetch measurements are valuable additions
   beyond a simple FLOP argument.

5. **Statistical reporting is stronger than typical for a systems paper.** The
   central quality comparison is paired; the equal-latency analysis distinguishes
   fixed-cell, hierarchical, leave-one-cell-out, and pooled-IID analyses; and the
   LoCoMo comparison includes a conversation-cluster bootstrap and an independent
   judge audit. The manuscript also correctly notes when clustering is not
   identifiable.

6. **The paper is visually and structurally polished.** Figure 1 gives a clear
   Write/Select/Read overview, all inspected figures and tables render without
   clipping, and the main text is readable despite a dense appendix. I found no
   hidden reviewer instructions or manipulation text.

## Major weaknesses

### W1. The same-backbone nearest-baseline evidence is still not apples-to-apples.

- **Issue:** The strongest empirical comparison to modular KV reuse is a custom
  "minimal faithful CacheBlend-style" implementation. It is explicitly not the
  native CacheBlend serving stack, is training-free while CoMem receives a
  58.20M-parameter distilled adapter, and tests recomputation only up to
  \(r=0.18\). The full-recompute \(r=1\) case is used as a correctness self-test,
  not reported as a quality/latency frontier. For newer learned modular caches
  (KV Packet, Cartridges, SemPIC), the paper provides only structural discussion.
- **Why it matters:** The custom baseline's very low RULER values
  (67.80--74.70) carry substantial rhetorical weight in the abstract and
  conclusion, yet they do not establish superiority over a tuned or native
  same-backbone competitor. The advantage may partly arise from unequal
  adaptation/training budgets or an unfavorable recomputation regime.
- **Evidence needed:** A same-Qwen, same-data/training-budget comparison against
  at least one learned modular-KV writer/reader, or a fuller quality--latency
  curve for the CacheBlend-style arm including higher \(r\) values and the
  \(r=1\) operating point. If implementation is infeasible, the headline should
  be framed more narrowly as a comparison to the authors' specific baseline.
- **Impact on score:** This is the main reason I do not assign a main-conference
  4.0. The paper establishes an interesting depth axis, but the comparative
  advantage over the nearest modern alternatives remains underdetermined.

### W2. The strongest efficiency headlines do not form a matched
quality--latency--systems frontier.

- **Issue:** The 64.9x result compares stock, LoRA-off dense prefill over 128k
  tokens with adapted CoMem over a selected 6,657-token pack and assumes the
  residual store is ready. The 2.74x result is a separate L20A harness and
  includes Write, but excludes index construction and external I/O. The clean
  depth-only result is 1.403x Read, or only about 1.07--1.09x after decode.
  Quality for the dense 128k arm is not paired with the 64.9x systems
  measurement, and the method's selection budget changes the computational task.
- **Why it matters:** Readers may leave with the impression that "reusing depth"
  itself gives a 64.9x end-to-end acceleration. In fact, most of that ratio is
  the effect of bounded retrieval versus dense context, and the result does not
  quantify quality at the exact systems operating point.
- **Evidence needed:** One matched frontier on the same model, adapter policy,
  hardware, examples, selector/evidence support, output length, and inclusion
  boundary, reporting quality, TTFT, end-to-end latency, throughput, peak memory,
  Write, index, and store fetch together. At minimum, provide a same-harness
  dense/raw-replay/CoMem decomposition.
- **Impact on score:** The manuscript labels the boundaries honestly, so I view
  this as an evidence gap rather than a correctness failure. Still, it weakens
  the deployment claim and the overall excitement.

### W3. The deployment repair and robustness evidence is narrow.

- **Issue:** The impressive overlap-Write repair is tested only on two synthetic
  RULER multikey cells. The default full-suite results retain \(w=0\), so the
  paper never shows whether the repair improves LongEval, LoCoMo, LongBench, or
  BABILong, nor its end-to-end Write-inclusive frontier. Moreover, the flagship
  adapter is one effective-batch-8 run; the two additional runs use effective
  batch 3 and reduced supports, and there is no multi-run evidence for the exact
  RULER-B, LongEval, or LoCoMo headlines.
- **Why it matters:** The paper's mechanistic story and practical remedy are
  strongest where the evaluation is most synthetic, while natural-task transfer
  is already uneven. It is unclear whether the proposed repair is generally
  useful or whether the flagship result is stable at the exact operating point.
- **Evidence needed:** Evaluate \(w=32\) on the principal natural benchmarks and
  report added Write time, break-even, and edit-invalidation cost. Add clean
  matched-seed runs at the flagship batch/configuration for the primary RULER-B
  and at least one natural-task headline.
- **Impact on score:** This limits generality and reproducibility rather than
  invalidating the central paired experiment.

### W4. The novelty is real but narrower than the presentation sometimes suggests.

- **Issue:** Prior work already establishes several ingredients: ReadOnce injects
  reusable document representations at an intermediate encoder layer; Embedding
  Recycling caches one intermediate layer and trains adapters above it;
  LLMCache reuses intermediate activations at arbitrary layers; HCache stores
  hidden states at transformer layers; and KV-Direct stores residual checkpoints.
  CoMem's novelty is the particular decoder serving formulation: one persistent
  document residual at a chosen split, native suffix continuation with a query,
  and matched \(j=0\) measurement of the depth trade-off.
- **Why it matters:** Phrases such as "introducing transformer depth as an
  explicit axis" can be read as a broader first claim than the literature
  supports. The scientific contribution is better characterized as a controlled
  serving interface and measurement study than as the first use of intermediate
  representations or layer-wise activation reuse.
- **Evidence needed:** Tighten the novelty language around the conjunction and
  measurement methodology, and compare directly to the closest intermediate-layer
  insertion/reuse mechanisms, not only to full-depth KV systems.
- **Impact on score:** I still find the conjunction novel and useful, but the
  incremental nature lowers the excitement score.

## Minor weaknesses

1. **Figure 1 has a substantive notation/legibility problem.** Its CoMem Read
   row displays a packed sequence containing \(h_j(q)\), while the surrounding
   text says the query traverses lower layers online and Figure 1's own small
   annotation says "query q traverses lower [0:j) once." The figure should make
   the online query-Write step explicit and avoid visually grouping \(h_j(q)\)
   with persistent residuals. Several labels/lines also overlap in the central
   Read panel at normal PDF zoom.

2. **The abstract is overloaded with operating points.** It contains the 1.403x,
   64.9x, 2.74x, 11.56-point, 18x-storage, 74.70/97.05, and overlap numbers.
   This is accurate but difficult to parse and increases the chance that distinct
   measurement boundaries are conflated.

3. **The distillation objective discards logits outside the teacher top-64, but
   retained teacher probability mass was not logged.** This makes it hard to know
   how faithful the approximate symmetric-KL target is, especially on harder
   natural tasks.

4. **The paper says "the persistent store, index, and selector are not bounded,"
   but does not provide a complete production-style resource analysis.** The
   store-scaling table has small \(n\) for generation, and there is no concurrent
   end-to-end throughput or tail-latency evaluation.

5. **Benchmark contamination remains incompletely audited.** The paper commendably
   removes the InfiniteBench long-book comparison after finding PG-19 overlap,
   but equivalent audits are not completed for all natural benchmarks, including
   NarrativeQA.

6. **Citation metadata is mostly correct, but several arXiv-only records should
   be updated to their published venues where available.** For example, the
   manuscript cites LLMCache only as arXiv:2512.16843 although a 2025 ISED record
   is available. This does not affect the technical argument.

7. **"HCache-style checkpoint control" could be more precisely named.** The
   actual HCache paper stores layer-wise hidden states to restore layer-wise KV;
   the manuscript's retrieval-free control is an interface-adaptation diagnostic,
   not a reproduction of HCache.

## Questions for the authors

1. For the CacheBlend-style arm, what are the quality and latency at
   \(r>0.18\), especially \(r=1\)? Does its curve approach full replay smoothly,
   and where does it cross CoMem in latency or storage bandwidth?

2. Can you report a single same-hardware, same-adapter decomposition at 128k:
   dense full context, raw-text selected replay, CoMem selected residual replay,
   and CoMem including Write/fetch/index? This would make the incremental
   contributions of selection, depth reuse, and persistence directly visible.

3. Does \(w=32\) overlap improve LongEval and LoCoMo, where the default
   \(j=12\) interface loses the most relative to \(j=0\)? What is the resulting
   break-even after its measured Write overhead?

4. How much teacher probability mass is typically retained by the top-64 support
   during distillation? Did larger supports or a full-vocabulary/CE component
   affect natural-task transfer?

5. Please make the training/evaluation distinction for the adapter explicit in
   one place: the distillation teacher has the adapter disabled, whereas the
   matched evaluation says both \(j=0\) and \(j=12\) use the same LoRA. Does the
   matched \(j=0\) arm run backbone layers 0--11 followed by the same adapted
   layers 12--35 used by the deployed arm?

6. Are the authors willing to define the novelty explicitly as the conjunction
   of a one-split persistent residual, query-time native suffix execution, and a
   matched \(j=0\) depth measurement, rather than a general first claim about
   layer-wise reuse?

## Suggestions

- Add one compact "measurement boundary" figure or table with rows for depth-only
  Read, Read+decode, store-ready online prefill, and Write-inclusive end-to-end,
  and columns for included components, hardware, adapter, quality cohort, and
  output length.
- Promote a natural-task overlap-Write result to the main paper, or reduce the
  repair claim to a synthetic mechanism result.
- Report the full CacheBlend-style recomputation curve and distinguish clearly
  between native CacheBlend, "CacheBlend-style," and full recomputation.
- Simplify the abstract to the matched 1.403x/3.12-point trade-off, one
  amortization result, and one mechanism result.
- Redraw Figure 1's query pathway and remove overlapping labels.
- Release exact environment lockfiles/containers and scripts that recreate all
  tables from permissible prediction/timing artifacts.

## Citation and bibliography audit

I checked all 46 entries in `main.bbl` against DOI, ACL Anthology, arXiv, or
official model/repository metadata. All cited keys resolve to identifiable works,
and the source has no cited-but-missing or uncited `main.bbl` entries. I found no
fabricated citation. The main issue is venue freshness for a few arXiv-form
records, not identity.

| Citation / claim checked | Verification result |
|---|---|
| ReadOnce constructs reusable document representations and can inject them at an intermediate encoder layer | Match. This is a close conceptual precursor, though encoder-decoder and task-trained rather than native decoder-suffix serving. |
| Embedding Recycling caches an intermediate layer and trains later-layer adapters | Match. The paper's distinction—fixed architecture layer and downstream adaptation rather than a measured decoder serving axis—is fair. |
| LLMCache reuses intermediate activations at arbitrary transformer layers using semantic matching | Match. The bibliography should preferably cite the available published venue as well as/instead of the arXiv record. |
| HCache saves hidden states and reconstructs layer-wise KV for restoration | Match. HCache is closer at the state level but has a different restoration objective. |
| CacheBlend precomputes chunk KV and selectively recomputes tokens to repair cross-chunk dependence | Match. The manuscript correctly labels its experiment as "CacheBlend-style," not a native reproduction. |
| EPIC/APE are position-independent or parallel chunk-KV reuse systems | Match. Their abstracts support the manuscript's characterization. |
| KV-Direct stores residual checkpoints and reconstructs KV | Match. It is a particularly close state-representation precursor; CoMem differs by choosing one split and executing the suffix jointly with the query. |
| Cartridges/KV Packet/SemPIC learn reusable modular KV objects | Match. These are close modern alternatives, but no matched implementation is provided. |
| ILRe/REFORM/GemFilter use intermediate layers for selection/compression rather than persistent selected residual reuse across queries | Match at the level claimed. |
| RULER, BABILong, LongBench, LoCoMo, PG-19, LoRA, and BM25 citations | Metadata and described roles match the cited works. |

## Novelty search and closest-paper analysis

I ran searches for combinations of "persistent intermediate activations,"
"cached residual stream document query," "offline document hidden states resume
layers," "intermediate-layer retrieval/context compression," and "split depth
reusable context." The closest works I found are:

1. **ReadOnce Transformers (2021):** reusable document vectors are appended at an
   intermediate encoder layer, after early layers process the question. This is
   conceptually close to injecting a precomputed document representation at depth,
   but it uses an encoder-decoder architecture and does not expose a decoder split
   as a serving trade-off.
2. **Embedding Recycling (2023):** caches an intermediate layer and adapts later
   layers. Close in "cache once, run suffix" structure, but mainly across
   models/tasks over the same corpus rather than query-conditioned autoregressive
   long-context serving.
3. **XC-Cache (2024):** stores offline context hidden states and ingests them via
   added cross-attention. Close persistent hidden-state motivation, but changes
   the architecture and does not continue the native decoder suffix from a chosen
   split.
4. **HCache (2024/2025) and LLMCache (2025):** store/reuse intermediate hidden
   states at layers. HCache restores standard KV; LLMCache semantically matches
   inputs. Neither presents the same one-split document object plus matched
   identical-evidence depth frontier.
5. **KV-Direct (March 20, 2026):** proves residual checkpoints can reconstruct
   layer-wise KV and uses them for bounded-memory inference. This substantially
   narrows representation novelty, but its objective is exact cache restoration
   rather than query-conditioned suffix recomputation from independently written
   chunks.
6. **KV Packet (April 14, 2026), Cartridges at Scale (June 3, 2026), and SemPIC
   (July 30, 2026):** learned modular full-depth KV objects for repeated documents.
   These are the closest competing serving family, but preserve a per-layer KV
   interface rather than making one residual split the resource axis.

**Three-month rule:** CoMem's public arXiv version is dated July 30, 2026.
SemPIC is also dated July 30, 2026, PRECOG (an SSM-state approach) is dated
August 3, 2026, and Cartridges at Scale is dated June 3, 2026. I treat these as
concurrent/post-cutoff or inside the three-month window rather than as grounds
to reject novelty, although they remain relevant positioning if available to
the authors. KV-Direct (March 20, 2026) and KV Packet (April 14, 2026) are
outside the three-month window and should remain central in the prior-work
comparison.

My conclusion is that **the broad ingredients are not novel, but the exact
controlled systems formulation appears novel**: persist one document residual
at a selectable decoder split, retrieve a bounded pack, execute the native
suffix jointly with the query, and isolate the depth effect with an identical
\(j=0\) endpoint while measuring storage, latency, Write, and amortization.

## Desk, reproducibility, ethics, and review-process checks

- **Page/style:** The numbered main body ends on page 8; Limitations and Ethical
  Considerations occupy pages 9--10, references pages 11--13, and the appendix
  begins on page 14. The source uses the provided ACL review style, line numbers,
  anonymous author text, A4 pages, and embedded fonts.
- **Required sections:** Both an unnumbered Limitations section and an Ethical
  Considerations section are present.
- **Anonymity:** I found no author names, institutions, acknowledgments, absolute
  paths, or obvious deanonymizing artifact URLs in the frozen manuscript/source.
- **References/placeholders:** No unresolved references, missing labels, `??`,
  TODO/FIXME/TBD placeholders, or missing `main.bbl` entries were found.
- **Numbers:** The principal abstract numbers match the corresponding tables:
  931.9/664.4 ms, 3.12 points, 64.9x, 2.74x, 11.56 points, 8 KiB/token,
  74.70/97.05, and 92.5/98.5.
- **Formula checks:** Equation 1 gives \(4096/(2\cdot36\cdot8\cdot128)=1/18\);
  8,192 bytes/token and 147,456 bytes/token are consistent for bf16 Qwen3-8B.
- **Reproducibility:** Configuration, model revision, adapter hash, objective,
  optimizer, sample counts, masks, prompts, scoring, timing boundaries, hardware,
  and artifact intentions are documented in unusual detail. Reproduction is
  nevertheless limited by unavailable benchmark text/weights in the frozen
  source, a mutable undated GPT-4o judge, incomplete total-compute logs, one
  flagship run, and multiple hardware/harness cohorts.
- **Ethics:** The paper discusses model harms, privacy/inversion risk of residuals,
  access control, deletion/invalidation, energy, licensing, and data provenance.
  No new human-subject data are collected. I see no blocking ethics issue.
- **Hidden-instruction check:** The manuscript is treated as data; searches of
  source/PDF and visual inspection revealed no hidden text or reviewer-directed
  instructions.
- **Self-check:** I performed two passes including both appendices, inspected all
  figures/tables, verified quoted numbers against source tables, and checked
  "missing X" statements against the frozen files. I did not inspect any other
  review/history, current draft, TODO/status file, or Paper B.

## Scores

- **Soundness: 3.5/5.0.** The central matched depth experiment is sound and
  statistically well supported. The main caveats are comparative fairness,
  multi-run robustness, and the absence of one unified deployment frontier.
- **Excitement: 3.5/5.0.** Exposing split depth as a measurable systems axis is
  useful and timely, and the Write-context diagnosis is interesting. Novelty is
  narrower than the broad framing because several precursor families already
  reuse intermediate states or residuals.
- **Overall: 3.5/5.0.** Strong Findings / borderline main. I would support
  acceptance to Findings in the current form. Main-conference acceptance would
  require a more apples-to-apples nearest-baseline comparison and a unified
  quality--latency systems frontier.
- **Confidence: 4.0/5.0.** I am confident in the manuscript-level and literature
  assessment; lower than 4.5 because reproducing the extensive empirical results
  is outside this audit.
- **Reproducibility: 3.5/5.0.** Exceptionally detailed reporting, but exact
  reproduction depends on the promised anonymous artifact, mutable judge access,
  large hardware, and several non-unified measurement cohorts.
