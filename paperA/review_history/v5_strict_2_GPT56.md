review_mode: strict
soundness: 3.5
excitement: 2.5
overall: 3.0
confidence: 4.5
reproducibility: 3.5

# Paper Summary

This paper studies repeated queries over a stable corpus and asks how much
transformer depth can be prepaid into a reusable document object. CoMem writes
one intermediate residual vector per token at split depth \(j\), retrieves a
bounded set of chunks, and resumes only layers \([j:L)\). The central experiment
is deliberately internal rather than cross-system: a same-pack, same-order,
same-mask, same-example, same-LoRA comparison between raw replay from \(j=0\)
and cached continuation from \(j=12\).

The most defensible result is a real quality--latency trade-off rather than
quality-preserving acceleration: selected-pack model Read falls from 931.9 ms to
664.4 ms (\(1.403\times\)), while a paired 15-cell RULER macro falls from 99.19
to 96.07 (3.12 points, paired 95% CI \([2.36,3.93]\)). The paper additionally
measures storage and amortization, runs a selector-dependent equal-latency
diagnostic, and attributes one synthetic multikey failure partly to missing
lower-layer document context. A 32-token left-overlap Write repairs much of that
local failure without increasing persistent bytes or per-query Read length.

The submission is unusually explicit that it does **not** establish superiority
over raw-text retrieval, position-independent caching, chunk-KV repair, or
learned modular-KV systems. My assessment is correspondingly narrow: the
central matched endpoint is credible and useful, but the work is not yet a
main-conference-level systems comparison. The closest matched baselines are
missing, the decision-level equal-latency aggregate is statistically fragile,
and the exact headline adapter has only one training run. I therefore place the
paper at **Findings level (Overall 3.0)** rather than ACL main level.

# Claim-by-Claim Evidence Audit

| Claim | Minimum sufficient evidence | Evidence actually supplied | Assessment |
|---|---|---|---|
| **C1. Reusing the first 12 layers gives a \(1.403\times\) isolated Read speedup at a 3.12-point RULER cost.** | Paired examples and packs; identical model, adapter, retrieval, masks, order, and decode settings; repeated same-hardware timing. | §5.1, Table 2, PDF p.6; Appendix A.4, Tables 30--31, PDF p.21; Appendix B.1, Table 35, PDF p.23. The paper reports 1,500 paired examples, exact McNemar counts, a paired bootstrap, three timing processes, and tight process-level ratios. | **Supported within the explicitly isolated Read boundary.** This is the strongest claim. |
| **C2. Self-distillation is necessary to make the split residual interface usable.** | Same split, evidence, model, and evaluator, changing only adapter availability. | Appendix A.1, Tables 9 and 11, PDF pp.12--13; §4, Eq. 2, PDF p.4. Same-\(j=12\) on/off results are large on RULER, LoCoMo, and BABILong. | **Supported for the tested adapter/model/tasks.** It is not a general theorem about all split interfaces. |
| **C3. The 8 KiB/token residual store requires repeated reuse and has the reported serving crossover.** | Correct byte accounting plus measured Write, fetch, Read, and decode components over relevant generation lengths and storage tiers. | §4, Eq. 1, PDF p.4; §5.2, Table 3, PDF p.6; Appendix A.4, Tables 25--32, PDF pp.19--21. | **Partly supported.** Byte accounting is correct. The table supports measured crossovers, but the abstract's “8--11” summary omits two displayed GPU-resident values, 7.7 and 5.5. |
| **C4. Equal-latency quality is selector-dependent, and CoMem wins neither aggregate.** | A predeclared latency calibration, representative quality cohort, identical metric definition, and dependence-aware paired inference. | §5.2, Table 4, PDF p.6; Appendix Table 8, PDF p.12. BM25 replay beats CoMem by 11.56 points; frozen-BGE replay is tied under the reported IID example bootstrap. | **Directionally supported but inferentially weak.** The cohort, calibration sample, and bootstrap hierarchy are inadequate for a deployment-level aggregate. |
| **C5. Missing lower-layer document context is a major cause of one multikey failure, and overlap is a local repair.** | Paired factorial intervention separating context from position, followed by a deployable approximation. | §5.3, Tables 5--6, PDF pp.6--7; Appendix Table 31, PDF p.21. | **Supported on the displayed synthetic 8k/16k multikey cohort.** The paper correctly avoids generalizing it to natural tasks. |
| **C6. Model-side Read work is bounded with store length, while selection/index cost is not.** | Fixed top-\(k\) model input over increasing store sizes, with separate retrieval/index measurements and evidence-budget failures. | §4, PDF p.4; Appendix A.3, Table 24, PDF p.19; Hy3 Table 34, PDF p.22. | **Supported as an interface property and synthetic stress result.** It is not constant-time end-to-end serving. |
| **C7. The split-forward implementation ports to a large sparse MoE.** | Exact partition test plus at least one quality/read-length evaluation on the port. | Appendix A.7, Tables 33--34, PDF pp.22--23. Exact self-test reports zero logit difference at several boundaries; \(n=16\) PG-19 windows and \(n=50\) RULER cells are then evaluated. | **Supported as implementation portability, not as a replicated quality frontier.** |
| **C8. CoMem is an internal depth-reuse measurement, not a superiority claim over reusable-context systems.** | Clear claim scope and no misleading cross-system causal inference. | §2 and Table 1, PDF p.3; Conclusion and Limitations, PDF pp.7--8. | **Appropriately scoped.** This honesty improves soundness but also limits novelty and excitement. |

# Abstract and Headline-Number Audit

I checked more than the requested five abstract numbers against the rendered
tables and formulas:

1. **931.9 \(\rightarrow\) 664.4 ms and \(1.403\times\): verified.**  
   \(931.9/664.4=1.4026\), consistent with Tables 2 and 30.
2. **99.19 \(\rightarrow\) 96.07, gap 3.12, CI \([2.36,3.93]\): verified.**  
   The 15 displayed RULER-B cells sum to 1441.0 and average 96.0667; the reported
   rounded gap is 3.12.
3. **8 KiB/token and \(1/18\) of full bf16 KV: verified.**  
   For \(d=4096\), one bf16 residual is 8,192 bytes. Full 36-layer GQA KV is
   \(2\cdot36\cdot8\cdot128\cdot2=147,456\) bytes/token, giving \(1/18\).
4. **“8--11 queries at 32k for generations up to 128 tokens”: not literally
   consistent with Table 3.**  
   Table 3 contains 8.9/9.2/10.9 for CPU-pinned storage but
   8.4/7.7/5.5 for GPU-resident storage. Across all displayed tiers and
   \(G\le128\), the range is therefore **5.5--10.9**, not 8--11.
5. **25.8--27.6 queries at 128k for one generated token: verified.**
6. **BM25 equal-latency gap 11.56 and BGE gap \(-1.00\), CI
   \([-4.67,2.67]\): verified against Tables 4 and 8.**
7. **92.5 \(\rightarrow\) 100.0 context control and 98.5 with 32-token overlap:
   verified against Tables 5--6.**

# Strengths

## S1. The central causal comparison is unusually well controlled

**Anchor:** §5.1, PDF p.5 lines 318--335; Table 2, PDF p.6; Appendix A.4,
Tables 30--31, PDF p.21.

The \(j=0\) and \(j=12\) arms share selected chunk IDs, order, examples, masks,
LoRA weights, and decode settings. The continuous-prefix \(h_{12}\) control
exactly reproduces full replay, providing a useful fidelity ceiling and showing
that suffix execution itself is not the source of the loss. The paper also
reports paired outcome counts, bootstrap intervals, exact McNemar significance,
and repeated-process latency. This is substantially stronger than comparing two
loosely matched long-context systems.

## S2. The paper separates selection gains, depth reuse, Write, Read, I/O, and decode

**Anchor:** Workload contract, PDF p.1 lines 49--54; §4, PDF p.4
lines 239--250; §5.2, PDF p.5 lines 343--369; Appendix A.4, PDF pp.18--21.

The manuscript repeatedly distinguishes the isolated \(1.403\times\) depth
effect from the much larger dense-to-bounded prefill number. It also distinguishes
store-ready online prefill, Write-inclusive timing, store placement, selector
cost, and decode. This prevents the common error of presenting a bounded
retrieval speedup as though it were caused solely by activation reuse.

## S3. Negative and selector-dependent results are reported rather than hidden

**Anchor:** §1, PDF p.2 lines 84--96; §5.2, Table 4, PDF p.6; Appendix Table 8,
PDF p.12.

The paper reports that iterative-BM25 replay substantially beats CoMem at
matched latency and that replacing only the replay selector with frozen BGE
changes the aggregate to a statistical tie. It explicitly states that CoMem
wins neither aggregate. This makes the study more scientifically useful even
though it lowers the method's competitive appeal.

## S4. The failure analysis is mechanistically focused and appropriately bounded

**Anchor:** §5.3, PDF p.5 lines 371--388; Tables 5--6, PDF pp.6--7; Appendix
Table 31, PDF p.21.

The context-by-position \(2\times2\) intervention is much more informative than
a generic ablation. It shows that position remapping alone is not a repair,
that document-contextual lower-layer states close the displayed gap, and that
the factors interact. The overlap intervention then converts this diagnosis
into a deployable local modification. Importantly, the authors explicitly limit
the claim to the synthetic paired cohort.

## S5. Reproducibility and scope disclosures are extensive

**Anchor:** Appendix Tables 25--28, PDF pp.19--21; Appendix B, PDF pp.22--23;
Limitations, PDF pp.7--8; Ethical Considerations, PDF p.8.

The source gives a checkpoint revision, adapter hash, exact layer span, optimizer,
data construction, masks, generation budgets, official scorers, sample counts,
software versions, hardware, and the final adapter's compute. It also discloses
the mutable GPT-4o judge, conversation clustering, incomplete contamination
audit, storage growth, model-version coupling, and the absence of production
p95/concurrency measurements. These disclosures are exemplary even where the
underlying evidence remains incomplete.

# Weaknesses

## W1. No matched comparison with the nearest reusable-context systems

- **Location:** Related Work §2, PDF p.3 lines 145--157; Limitations, PDF p.7
  lines 434--443; `sections/07_limitations.tex:12--19`.
- **Exact quote (19 words):** “The paper does not provide a same-backbone, same-hardware implementation of the closest PIC, chunk-KV repair, or learned modular-cache systems.”
- **Problem:** The paper's central endpoint is internally well controlled, but
  it does not establish where this residual object sits relative to EPIC-like
  PIC, CacheBlend/Cache-Craft-like repair, KV Packet, or learned modular KV under
  the same model, corpus, quality target, storage tier, and latency boundary.
- **Affected claim/norm and why it matters:** This limits **C8** and the
  contribution's decision relevance. For a systems paper, a new operating point
  should normally be compared with the nearest implementable alternatives, not
  only with raw replay and methods that answer materially different questions.
  Without this experiment, the reader cannot tell whether the measured
  depth/storage trade-off is competitive or dominated.
- **Sufficient remedy:** Implement at least one nearest PIC/chunk-KV baseline and
  one learned modular-cache or residual-reconstruction baseline on Qwen3-8B,
  using identical chunks, tasks, hardware, storage placement, and full TTFT
  accounting. Report quality, persistent bytes, one-time construction cost,
  online p50/p95 latency, and reuse crossover.
- **Severity:** **Major.**

## W2. The equal-latency aggregate is not supported by a dependence-aware or representative protocol

- **Location:** Appendix Table 8, PDF p.12, rows “Quality cohort,” “Calibration
  split,” and “Statistical unit”; `sections/tab_equal_latency_protocol.tex:9--47`.
- **Exact quote (24 words):** “bootstrap pools and resamples all 900 paired example differences IID 10,000 times with seed 0; it does not resample task cells or LoCoMo conversations.”
- **Problem:** The nine cells mix synthetic and natural tasks with equal weights;
  the LoCoMo cell is the first 100 stored items and therefore only conversation
  0; latency budgets are selected from three reserved RULER documents; and the
  reported interval treats all 900 examples as IID despite task and
  conversation clustering. The paper is transparent about each choice, but
  transparency does not make the aggregate statistically representative.
- **Affected claim/norm and why it matters:** This weakens **C4**, which is the
  only decision-level comparison asking whether saved compute can buy more
  evidence. The sign of an arbitrary equal-cell aggregate can change with task
  composition, selector, or clustering, so the reported “tie” should not carry
  deployment-level weight.
- **Sufficient remedy:** Predefine a representative task mixture; sample LoCoMo
  across all conversations; calibrate latency on a larger document set; report
  every cell and sign; and use a hierarchical paired bootstrap that resamples
  task cells and LoCoMo conversations before examples. A sensitivity analysis
  over plausible task weights is also needed.
- **Severity:** **Major.**

## W3. The exact headline adapter has no clean training-seed replication

- **Location:** Limitations, PDF p.7 lines 425--433;
  `sections/07_limitations.tex:4--10`; Appendix A.5, PDF p.20
  lines 1034--1041.
- **Exact quote (7 words):** “The flagship is one batch-8 training run.”
- **Problem:** The two added adapters change effective batch from 8 to 3, and
  the paper explicitly states that the exact 15-cell RULER-B, LoCoMo, and
  LongEval headlines have no multi-run aggregate. Thus initialization and
  optimization noise are confounded with effective batch, while the reported
  central quality loss is conditional on one trained interface.
- **Affected claim/norm and why it matters:** This affects **C1** and **C2**.
  The latency measurement is stable, but the quality coordinate of the claimed
  operating point may move with training. For a learned adapter supporting the
  main result, run-level robustness on the exact headline evaluation is a
  standard reliability requirement.
- **Sufficient remedy:** Train at least three \(j=12\) adapters with the same
  effective batch, data order budget, optimizer, and schedule but distinct seeds;
  evaluate all on the exact RULER-B cells, LongEval, and LoCoMo; and report
  run-level means, SDs/CIs, and paired \(j=0\) gaps.
- **Severity:** **Major.**

## W4. The abstract's 32k crossover range contradicts the displayed table

- **Location:** Abstract, PDF p.1 lines 16--20; §5.2, PDF p.5
  lines 345--354; Table 3, PDF p.6;
  `sections/00_abstract.tex:11--13`.
- **Exact quote (13 words):** “measured break-even is 8--11 queries at 32k for generations up to 128 tokens”
- **Problem:** Table 3 reports GPU-resident values of 8.4, 7.7, and 5.5 for
  \(G=1,32,128\), and CPU-pinned values of 8.9, 9.2, and 10.9. The all-tier range
  is 5.5--10.9, while 8--11 describes only a rough CPU-pinned range.
- **Affected claim/norm and why it matters:** This is a direct headline-number
  inconsistency in **C3**. It does not overturn the amortization conclusion, but
  abstract numbers must accurately summarize the displayed evidence.
- **Sufficient remedy:** Change the abstract/introduction/conclusion to
  “5.5--10.9 queries across the displayed GPU/CPU placements,” or explicitly say
  “8.9--10.9 for CPU-pinned storage.”
- **Severity:** **Minor.**

## W5. Two claimed matched natural-task baselines are absent from their cited tables

- **Location:** §5.1, PDF p.5 lines 328--337;
  `sections/05_experiments.tex:28--35`; cited Tables 19 and 21, PDF pp.15--16.
- **Exact quote (9 words):** “Matched \(j=0\) scores 97.2 versus 69.0 on LongEval”
- **Problem:** Table 19 does not display the claimed matched \(j=0\) LongEval
  score of 97.2, and Table 21 does not display the claimed matched \(j=0\)
  LongBench score of 12.31. Those tables show other baselines, including
  KV-Direct, but not the exact matched raw-replay rows cited in the prose.
- **Affected claim/norm and why it matters:** The natural-task scope extension of
  **C1** is therefore not auditable from the frozen paper. Readers cannot verify
  sample support, generation settings, pairing, or uncertainty for these two
  numbers.
- **Sufficient remedy:** Add explicit matched-\(j=0\) rows to the LongEval and
  LongBench tables, with sample counts, identical-pack confirmation, generation
  limits, and paired uncertainty where available.
- **Severity:** **Minor.**

## W6. The strongest speed number is not an end-to-end serving result

- **Location:** Table 2 caption, PDF p.6; Appendix A.4, PDF pp.19--21;
  `sections/tab_core_tradeoff.tex:21--30`.
- **Exact quote (9 words):** “retrieval, persistent I/O, reusable Write, and decode are excluded.”
- **Problem:** The \(1.403\times\) result isolates model Read on a fixed
  approximately 6.5k-token pack from 16k contexts. The appendix reports that
  total-decode medians are similar, the serving crossover uses only three
  process medians, raw component times are merely described as archived, and
  concurrent end-to-end p95/tail behavior is not measured.
- **Affected claim/norm and why it matters:** The isolated causal result in
  **C1** remains valid, but the systems impact and parts of **C3** remain
  uncertain. Production relevance depends on selector, storage, query Write,
  decode, batching, and tail latency rather than the isolated transformer
  suffix alone.
- **Sufficient remedy:** Provide a frozen-source component table and a
  same-hardware end-to-end benchmark including index lookup, state fetch,
  query Write, model Read, decode, and amortized document Write, with p50/p95,
  concurrency, multiple source lengths, and generation lengths.
- **Severity:** **Minor.**

# Questions That Could Change the Score

1. **Nearest matched baseline:** On Qwen3-8B and the same H20 harness, how does
   CoMem compare with one PIC/chunk-KV repair method and one learned modular-KV
   method at matched quality, storage placement, and end-to-end TTFT? A
   non-dominated result could raise Excitement and Overall.
2. **Equal-latency reanalysis:** Do the BM25 and BGE conclusions survive a
   hierarchical bootstrap over task cells and LoCoMo conversations, a stratified
   LoCoMo sample, and a larger latency-calibration set? A robust result could
   remove W2.
3. **Training robustness:** What are the exact RULER-B, LongEval, and LoCoMo
   results for at least three same-effective-batch \(j=12\) adapters? If the
   3.12-point gap and natural-task results are stable across runs, Soundness
   would increase.
4. **Missing matched rows:** Can the authors provide the exact per-example
   matched-\(j=0\) LongEval and LongBench rows underlying 97.2 and 12.31,
   including generation settings and hashes? This is needed to audit the
   natural-task extension.

# Method, Formula, Metric, and Boundary Audit

## Formulas and boundary cases

- **Equation 1 is algebraically correct** under its stated common-dtype,
  full-layer-KV accounting. For Qwen3-8B it yields \(1/18\), 8,192 bytes/token
  versus 147,456 bytes/token.
- The equation compares a single residual with **all-layer** KV, not with text,
  token IDs, quantized KV, or a partially retained cache. The paper states this
  boundary clearly.
- \(j=0\) is implemented as token-ID replay rather than storing a residual;
  \(j>0\) stores \(h_j\). The evaluated causal endpoint is only \(0\to12\).
  \(j=L\), quantized residuals, and model-upgrade compatibility are not evaluated.
- **Equation 2 is well-defined** as a symmetric weighted KL on the teacher's
  top-64 support. However, logits outside that support are discarded and the
  retained teacher mass was not logged, so the objective's approximation error
  cannot be quantified. This is disclosed and does not invalidate the empirical
  adapter result.
- Overlap-Write correctly leaves persistent bytes and Read length unchanged
  because only the target chunk's residuals are stored; it increases Write work
  and edit invalidation, which the paper states.

## Baselines and benchmark validity

- The matched \(j=0\) endpoint is the correct minimum experiment for the causal
  depth-reuse claim.
- KV-Direct, InfLLM, StreamingLLM, MemoryLLM, LLoCO, SnapKV, and PyramidKV are
  mostly descriptive because backbones, context extension, training, or timing
  boundaries differ. The captions generally state these limitations.
- The missing minimum experiment for competitive systems relevance is a matched
  nearest reusable-context baseline, as described in W1.
- RULER and BABILong provide controlled diagnostic coverage; LongEval,
  LongBench, and LoCoMo add natural-task scope. The paper correctly treats the
  latter as scope checks rather than a homogeneous leaderboard.
- Beyond-native-window results are labeled positional-extrapolation stress tests.

## Metrics and statistics

- RULER, BABILong, LongBench, and LongEval use named official or deterministic
  scorers with sample counts and generation limits.
- LoCoMo's primary judge is a mutable, undated `gpt-4o` endpoint. The evaluation
  date, parsing rules, failures, saved decisions, conversation-cluster bootstrap,
  and a 200-item DeepSeek-V3 audit are reported. Exact future reproduction is
  still impossible, as the paper acknowledges.
- The paired RULER analysis is strong: 1,500 examples, exact outcome counts,
  bootstrap CI, and McNemar test.
- The equal-latency IID bootstrap is not dependence-aware; see W2.
- The exact headline adapter lacks clean same-batch seed replication; see W3.

## Compute and reproducibility

- The final adapter is reported as approximately 2.9 H20 GPU-hours, with
  checkpoint revision, SHA-256, trainable parameter count, optimizer, schedule,
  software versions, and data construction.
- Total compute over probes, failed runs, baseline generation, and ablations was
  not logged.
- The allowed frozen materials contain source and detailed tables but not the
  claimed released raw predictions/timing artifacts, so those artifact-level
  claims could not be independently rerun in this review.

# Figure and Table Inspection

I visually inspected both rendered figures and every numbered table
(Tables 1--36) in the 23-page PDF.

- **Figure 1:** The Write/Select/Read decomposition, matched \(j=0\) path, and
  overlap variant are internally consistent with §4. It is dense but readable
  at normal PDF zoom; no clipping or hidden text was observed.
- **Figure 2:** The probe/readout distinction is visible and the caption
  appropriately says the plot motivates rather than validates depth reuse.
- **Tables 1--36:** No missing caption, obvious overflow, unresolved reference,
  or rendering corruption was found. Scriptsize/resized appendix tables remain
  readable when zoomed.
- Arithmetic spot checks passed for the RULER-B macro, 3.12-point gap,
  \(1.403\times\) speedup, LoCoMo category-weighted 38.27 score, LongBench
  macros 12.15/12.17, and the 8 KiB/token storage ratio.
- The substantive table/text discrepancies are the 32k crossover range in W4
  and the absent matched natural-task rows in W5.

# Citation-Claim Match Audit

Eight load-bearing citation uses were checked:

1. **Lewis et al. (RAG)** supports the raw-text retrieval framing.
2. **CacheBlend, TurboRAG, Cache-Craft, and RAGCache** support the claim that
   chunk-level KV reuse/cache systems precompute or reuse retrieved context
   state and address composition in different ways.
3. **EPIC, MEPIC, and APE** support the PIC/parallel-encoding positioning.
4. **ReadOnce Transformers and Embedding Recycling** support the precedent for
   reusable intermediate representations.
5. **HCache and KV-Direct** support activation/residual restoration or
   reconstruction as the closest intermediate-state precedent.
6. **StreamingLLM, H2O, SnapKV, PyramidKV, and MiniCache** support the separate
   token/KV compression family.
7. **RULER, BABILong, LongBench, LongEval/LongChat, and LoCoMo** match the named
   benchmark uses and metrics.
8. **Hinton et al. and LoRA** match the distillation and low-rank adaptation
   ingredients, without being used to claim that the paper's exact objective is
   standard.

I found no load-bearing citation that plainly contradicted its associated
claim. The main issue is missing matched empirical comparison, not false
literature attribution.

# Complete `main.bbl` Audit

All 43 entries in `main.bbl` are actually cited. Status reflects the metadata
that could be independently checked before the network/time cutoff. A network
or index limitation is marked **Unverifiable**, never **Not found**.

| Key | Status | Audit note |
|---|---|---|
| `cachecraft` | Verified | Title, authors, year, ACM venue/DOI consistent. |
| `longbench` | Verified | ACL Anthology title, authors, year, venue/DOI consistent. |
| `pyramidkv` | **Unverifiable** | Title, authors, and arXiv 2406.02069 verified; exact proceedings year in the BBL could not be conclusively resolved before cutoff. |
| `kvpacket` | Verified | Title, authors, arXiv 2604.13226, and 2026 date consistent. |
| `cartridgesbase` | **Unverifiable** | Title, authors, and arXiv 2506.06266 verified; exact ICLR 2026 volume/page metadata was not independently confirmed before cutoff. |
| `hcache` | Verified | Title, authors, EuroSys venue, year, and DOI consistent. |
| `llama3` | Verified | Title, year, and arXiv 2407.21783 consistent. |
| `cartridges` | Verified | Title, authors, and arXiv 2606.04557 consistent; post-novelty-cutoff contemporaneous work. |
| `distillation` | Verified | Title, authors, 2015 date, and arXiv 1503.02531 consistent. |
| `ruler` | Verified | Title, authors, arXiv 2404.06654, and venue metadata consistent. |
| `lora` | Verified | Title, authors, ICLR publication year, and arXiv 2106.09685 consistent. |
| `epic` | Verified | Title, authors, arXiv 2410.15332, and ICML publication metadata consistent. |
| `ragcache` | Verified | Title, authors, year, ACM venue/DOI consistent. |
| `babilong` | Verified | Title, authors, NeurIPS metadata, year, and DOI consistent. |
| `rag` | Verified | Title, authors, NeurIPS 2020 venue/year consistent. |
| `longchat` | Verified | Official LMSYS blog title, authors, and date consistent. |
| `snapkv` | Verified | Title, authors, NeurIPS metadata, and DOI consistent. |
| `ilre` | Verified | Title, authors, year, and arXiv 2508.17892 consistent. |
| `readonce` | Verified | Title, authors, ACL 2021 venue, and DOI consistent. |
| `minicache` | Verified | Title, authors, NeurIPS metadata, and DOI consistent. |
| `turborag` | Verified | Title, authors, EMNLP 2025 venue/pages/DOI consistent. |
| `locomo` | Verified | Title, authors, ACL 2024 venue/pages/DOI consistent. |
| `xccache` | Verified | Title, authors, Findings EMNLP 2024 venue/DOI consistent. |
| `kvdirect` | Verified | Title, authors, arXiv 2603.19664, and 2026 date consistent. |
| `pg19` | Verified | Compressive Transformers/PG-19 title, authors, year, and arXiv consistent. |
| `bm25` | Verified | Title, authors, journal metadata, year, and DOI consistent. |
| `embeddingrecycling` | Verified | Title, authors, Findings EACL 2023 venue/DOI consistent. |
| `gemfilter` | Verified | Title, authors, Findings ACL 2026 metadata/DOI consistent. |
| `reform` | Verified | Title, authors, year, and arXiv 2506.01215 consistent. |
| `lloco` | Verified | Title, authors, EMNLP 2024 venue/pages/DOI consistent. |
| `hunyuan` | Verified | Official Tencent/Hugging Face model-card title and model identity consistent. |
| `fusionrag` | Verified | Title, authors, year, and arXiv 2601.12904 consistent. |
| `mepic` | Verified | Title, authors, year, and arXiv 2512.16822 consistent. |
| `longmem` | Verified | Title, authors, NeurIPS 2023 venue/year consistent. |
| `memoryllm` | Verified | Title, authors, ICML/PMLR 2024 metadata and arXiv consistent. |
| `infllm` | Verified | Title, authors, NeurIPS 2024 metadata and DOI consistent. |
| `streamingllm` | Verified | Title, authors, ICLR publication metadata and arXiv consistent. |
| `sempic` | Verified | Title, authors, and arXiv 2607.28069 consistent; post-novelty-cutoff contemporaneous work. |
| `xu2024retrievallong` | Verified | Title, authors, ICLR 2024 publication metadata consistent. |
| `qwen3` | Verified | Qwen3 technical-report title, year, and arXiv 2505.09388 consistent. |
| `ape` | Verified | Title, authors, year, and arXiv 2502.05431 consistent. |
| `cacheblend` | Verified | Title, authors, EuroSys venue/pages/DOI consistent. |
| `h2o` | Verified | Title, year, and NeurIPS venue consistent. |

**Totals:** 41 Verified; 0 confirmed Metadata error; 0 Not found;
2 Unverifiable exact venue/year details.

# Novelty Search Summary

## Search protocol

I ran five focused searches, stopping rather than repeatedly retrying
rate-limited endpoints:

1. `"persistent intermediate residual" transformer cache long context`
2. `"residual stream" reusable document cache LLM`
3. `"position independent" KV cache retrieval augmented generation reusable chunks`
4. `"reusable representations" text transformers`
5. `"modular KV cache" document retrieval LLM`

The novelty cutoff was fixed at **2026-05-04**, three months before the frozen
paper date. Work first appearing after that date was treated as contemporaneous
and was not used to reduce novelty.

## Five closest pre-cutoff papers

1. **ReadOnce Transformers (2021):** Reusable task-independent intermediate text
   representations consumed by a modified downstream model. This is the closest
   conceptual precedent for “write a document representation once, reuse across
   queries,” but it is not the same decoder-only, one-residual-per-token,
   bounded-retrieval suffix interface.
2. **HCache (2024 preprint / EuroSys 2025):** Checkpoints activations to restore
   evicted inference state and resume computation. It is a strong precedent for
   activation-level restoration, but not a persistent selected document object
   for repeated RAG queries.
3. **KV-Direct (2026-03-20):** Shows that per-token residual state can reconstruct
   layer-wise KV and uses residual checkpoints for bounded-memory inference.
   This is very close in stored object, but its target is KV redundancy and
   reconstruction rather than direct suffix execution over a retrieved pack
   with a matched depth axis.
4. **KV Packet (2026-04-14):** Context-independent reusable KV packets with
   lightweight adapters trained by self-supervised distillation. It is very
   close in workload and adaptation strategy, but stores learned per-layer KV
   objects rather than one residual at a chosen split.
5. **Cartridges (2025):** Offline learned reusable KV representations for a
   corpus, amortized across queries. It is close in repeated-query workload and
   learned reusable object, but differs substantially in training, object, and
   online execution.

EPIC and CacheBlend are also close serving-system references for modular
position-independent/chunk-KV reuse, but they repair or recompute per-layer KV
rather than exposing transformer depth as the measured axis.

## Post-cutoff contemporaneous work

- **Cartridges at Scale**, first appearing 2026-06-03.
- **CacheTune / Adaptive KV Cache Reuse**, first appearing 2026-05-20.
- **SemPIC**, first appearing 2026-07-30.

These are after 2026-05-04 and therefore do not count against the paper's
pre-cutoff novelty. The paper appropriately calls the cited June/July works
concurrent.

## Novelty conclusion

I did not find a pre-cutoff paper with the exact combination of:

1. one persistent residual per token at a tunable split,
2. bounded retrieval of those residuals,
3. direct execution of only \([j:L)\), and
4. a same-pack \(j=0\) endpoint explicitly measuring prepaid depth.

The exact operating point and controlled measurement are therefore novel.
However, the broader ingredients—reusable intermediate representations,
activation checkpoints, residual sufficiency, PIC/chunk caching, and distilled
modular objects—are established. I view the novelty as a careful synthesis and
measurement paper, not a new general memory paradigm. This supports Findings
rather than main-conference excitement.

# Limitations, Ethics, and Desk-Reject Risk

## Required sections and format

- **Page limit:** The rendered PDF has 23 pages. The main paper, including
  Conclusion, Limitations, and Ethical Considerations, ends on PDF p.8;
  references begin on p.8 and appendices begin on p.11. This appears consistent
  with an eight-page ACL long-paper body. Exact 2026 call-specific counting
  language was not consulted because the review was restricted to the supplied
  materials; any rule beyond the rendered/style evidence is **Unverifiable**.
- **Limitations:** An exact unnumbered `Limitations` section is present
  (`\section*{Limitations}`), spanning PDF pp.7--8.
- **Ethics:** A substantive `Ethical Considerations` section is present on PDF
  p.8 and discusses sensitive-memory disclosure, inversion/membership risk,
  authorization, deletion, energy, licenses, and data handling.
- **Anonymity:** Author is “Anonymous ACL Submission.” I found no author names,
  affiliations, identifying acknowledgments, deanonymizing repository URL, or
  self-referential ownership claim in the frozen source/PDF.
- **Official style:** `\usepackage[review]{acl}` is used; the rendered paper uses
  the expected two-column review format.
- **References/TODOs:** Static source analysis found no undefined `\ref`, no
  undefined citation key, no unused `main.bbl` item, and no TODO/TBD/FIXME/
  placeholder marker. The rendered PDF contains no unresolved `??`.

## Prompt-injection and hidden-text audit

I treated all paper content as data. Source and rendered-PDF scans found:

- no instruction to the reviewer, score manipulation, prompt injection, or
  “ignore previous instructions” text;
- no white-on-white, invisible, opacity, off-page, or tiny hidden reviewer text;
- no JavaScript, forms, attachments, or suspicious PDF metadata;
- ordinary `\scriptsize`/`\resizebox` usage only for visible tables.

I therefore see **no prompt-injection or hidden-text desk-reject risk**.

## Substantive ethics/limitations assessment

The limitations section is candid and covers nearly all central boundaries:
single primary backbone, lexical retrieval, English-only evaluation, single
headline training run, absent matched modular-cache baselines, storage growth,
model-version coupling, task dependence, beyond-window extrapolation, mutable
judge, contamination, and lack of production concurrency/tail measurements.
The ethics section appropriately treats residual tensors as sensitive rather
than assuming they are safe because they are not human-readable.

I see **no clear desk-reject condition** in the frozen materials. The abstract
crossover inconsistency should be corrected, but it is not a desk-reject issue.

# Non-Scoring Suggestions and Typos

1. Record and report the teacher probability mass retained by the top-64 support;
   this would quantify the distillation objective's truncation error.
2. Move the surprising `rounds=0` meaning (“automatic
   \(\lceil k/h\rceil\) rounds”) from the appendix into the main method.
3. Add a one-page appendix index mapping every secondary claim to its table;
   36 tables are comprehensive but difficult to navigate.
4. Report confidence intervals for the serving crossover \(Q^\star\), not only
   point estimates from process medians.
5. In Table 14, define “Raw-text reader quality” explicitly as unconditional
   accuracy and also display hit-conditional accuracy, since the prose discusses
   that distinction.
6. In Table 29, make the final `ms/tok` column label explicitly say “decode
   ms/token,” matching the caption.
7. Figure 1 is informative but text-dense; reducing repeated labels would improve
   readability at print scale.
8. Recheck the exact proceedings metadata for PyramidKV and Cartridges before
   camera-ready submission.

# Scores

## Soundness: 3.5 / 5

The central matched endpoint is carefully controlled and statistically
well-supported. Formulae and headline quality/latency arithmetic mostly check
out. I do not assign 4.0 because the equal-latency inference is not
dependence-aware, the learned headline point has no clean seed replication, and
two matched natural-task numbers are not displayed in their cited tables.

## Excitement: 2.5 / 5

The precise “depth as a reuse axis” measurement is useful, and the negative
results are valuable. However, reusable intermediate states, residual
checkpoints, PIC, and learned modular caches are established directions. CoMem
loses the BM25 equal-latency aggregate, ties the BGE aggregate, and has no
matched nearest-system comparison. The contribution is therefore more
diagnostic than transformative.

## Overall: 3.0 / 5

This is a credible **Findings-level** paper: technically careful, unusually
transparent, and useful as an internal measurement and failure analysis. I am
not at 4.0/ACL-main because the paper does not yet demonstrate a competitive
non-dominated operating point against the nearest reusable-context systems, and
its decision-level and run-level robustness are incomplete. Under the required
calibration, uncertainty between Findings and main is resolved downward.

## Confidence: 4.5 / 5

I read the full PDF twice, including both appendices; inspected all figures and
tables; recomputed key arithmetic; checked all citation keys and bibliography
entries; and performed focused novelty searches. Residual uncertainty concerns
two exact venue/proceedings metadata records and unavailable artifact-level raw
files, not the central paper reading.

## Reproducibility: 3.5 / 5

Configuration detail is strong: model revision, adapter hash, optimizer,
software, hardware, masks, scorers, seeds, and sample counts are supplied.
Reproducibility is reduced by the mutable GPT-4o endpoint, missing raw component
records in the frozen materials, unlogged total project compute, absent clean
same-batch headline replications, and the fact that the claimed anonymous
artifact itself was not among the permitted review materials.

# Review-Process Self-Check

- Used only the frozen v5 PDF, frozen v5 source directory, and strict review
  template as paper/project materials; no other review, history, TODO, status,
  current, or calibration file was read.
- Completed two full-paper passes including all appendices.
- Built claims C1--C8 and compared each with a minimum sufficient experiment.
- Inspected both figures and Tables 1--36 in the rendered PDF.
- Checked desk-format items, exact Limitations, anonymity, ACL review style,
  prompt injection, hidden text, TODOs, unresolved references, and citation keys.
- Audited all 43 `main.bbl` entries and eight load-bearing citation--claim
  matches.
- Ran five bounded novelty searches with cutoff 2026-05-04 and did not count
  later contemporaneous work against novelty.
- Recomputed the storage ratio, RULER macro/gap, latency ratio, LoCoMo weighted
  score, and LongBench macros.
- Mechanically verified every weakness quote against the frozen source; every
  quote is at most 25 words.
- Checked each “missing/lacks/absent” weakness against both main text and
  appendices. Claims that could not be completed because of network or artifact
  availability are labeled **Unverifiable**, not **Not found**.
