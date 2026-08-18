```yaml
review_mode: strict
soundness: 3.0
excitement: 3.0
overall: 2.5
confidence: 4.0
reproducibility: 3.5
```

# Paper Summary

This paper studies a precise systems/model-interface question: how much lower-transformer computation can be prepaid once for a document and reused across later queries? CoMem independently writes each document chunk through layers `[0:j)`, stores one residual vector per token at depth \(j\), retrieves a bounded number of chunks, and resumes layers `[j:L)` with the query. The strongest experiment is deliberately internal and matched: on Qwen3-8B, the same selected pack, order, mask, examples, adapter, and hardware path are replayed either from tokens (\(j=0\)) or cached \(h_{12}\). The latter changes isolated model Read from 931.9 to 664.4 ms (1.403×) while reducing a 15-cell RULER-B macro from 99.19 to 96.07.

The paper additionally (i) trains a rank-32 upper-layer LoRA by self-distillation, (ii) measures storage and repeated-query break-even, (iii) reports an equal-online-latency result in which raw replay is better, and (iv) diagnoses one synthetic multikey failure through context/position controls and an overlap-Write repair. The authors repeatedly and appropriately disclaim superiority over RAG, PIC, or modular-KV systems.

My assessment is that the core matched frontier is real, carefully bounded, and potentially useful, but the paper is not yet a sufficiently complete empirical characterization of the advertised quality–latency–storage frontier for ACL main. In particular, the central frontier has only two directly matched endpoints, the decision-relevant equal-latency experiment is under-specified, and there is no same-backbone end-to-end closest-system comparison. I therefore place it below Findings with a 2.5 rather than 3.0; the lower bin is chosen because the missing evidence affects the paper's central deployment interpretation, not because the measured two-point result appears false.

# Claims and Claim–Evidence Map

| ID | Claim | Minimum sufficient evidence | Evidence actually supplied | Assessment |
|---|---|---|---|---|
| C1 | Depth is a usable cross-query reuse axis. | Same model/evidence/adapter with only replay start changed; quality and latency measured. | Sec. 5.1, Table 2; Appendix Table 28: \(j=0\) vs \(j=12\), paired RULER and fixed-pack Read. | Supported for one backbone, split, pack length, task macro, and isolated Read boundary. |
| C2 | \(j=12\) gives 1.403× faster model Read at a 3.12-point RULER cost. | Replicated timing and paired quality uncertainty. | 3 independent timing processes × 20 reads; paired bootstrap CI and McNemar test. | Well supported within the stated boundary. |
| C3 | Self-distillation makes the residual interface usable. | Same \(j\), same retrieval/evaluation, adapter on/off; ideally multiple seeds. | Tables 8 and 10; Appendix A.5 adds three training runs, though added runs use batch 3 vs 8. | Supported as a large intervention effect; seed evidence is imperfectly controlled. |
| C4 | The one-time Write pays off only after reuse, around 8–11 queries at 32k and 26–28 at 128k. | End-to-end cumulative latency under stated store tiers, generation lengths, and uncertainty. | Table 3 plus archived-component claim; fetch, Read, decode included, selector cancels. | Plausible for measured harness; dispersion/raw components are not in the paper and 128k coverage is only \(G=1\). |
| C5 | At matched online latency, raw replay is better than CoMem on a held-out mixed cohort. | Fully defined cohort, metric/aggregation, calibration protocol, per-task outcomes, exact latency distributions, paired IDs. | Table 4 gives only top-\(k\), aggregate 64.78/53.22, ±5% statement, and bootstrap CI. | Direction may be credible, but the experiment is not auditable from the paper. |
| C6 | Missing lower-layer document context is a major tested failure source; positions alone do not repair it. | Factorial context × position intervention with paired data. | Table 5: 92.5/88.0 vs 100/100 on pooled 8k/16k multikey. | Supported on that one synthetic cohort; interaction prevents a unique causal decomposition, as acknowledged. |
| C7 | A 32-token overlap recovers most of that local gap without increasing persistent bytes or per-query Read work. | Paired overlap-width experiment plus Write cost. | Table 6: 92.5→98.5, CI [3.0, 9.5], same persistent/Read footprint; theoretical FLOPs. | Supported locally; no measured repaired frontier or natural-task validation. |
| C8 | Model-side Read is bounded as stored context grows. | Fixed-\(k\) length derivation and scaling experiment separating selector/store costs. | Eq. 1/interface; Table 23 to 4M, Table 32 to 256k; selector and store explicitly unbounded. | Supported as a model-work claim, not as end-to-end constant latency. |
| C9 | The implementation is portable beyond Qwen3-8B. | Exact partition checks and quality experiments on materially different architecture(s). | Hy3 exact split self-test and \(n=50\) RULER results; other scales only described. | Supports implementation portability, not a replicated frontier or broad quality generalization. |
| C10 | CoMem establishes a measured quality–latency–storage frontier. | Multiple matched \(j\) values with quality, Read, Write, store/I/O, and preferably end-to-end latency under one protocol. | Central \(j=0/12\) comparison; separate per-depth deployment curve has different adapters and missing \(j=12\) Write; storage/I/O are separate cohorts. | Partially supported; “frontier” is stronger than the unified matched evidence. |

# Strengths

## S1. The paper makes an unusually clean, falsifiable central comparison

**Anchor:** Sec. 5.1, PDF p. 4 lines 273–286 and Table 2 on p. 5.

The matched \(j=0\)/\(j=12\) control fixes selected chunks, order, examples, mask, sink, adapter, and pack. This is much more informative than comparing heterogeneous long-context systems. The paper also correctly calls the result a trade-off rather than quality-preserving acceleration.

## S2. The authors expose negative results and system boundaries rather than hiding them

**Anchor:** Abstract lines 20–23; Sec. 5.2, PDF p. 5 lines 306–313; Limitations, PDF pp. 6–7.

The equal-latency loss, large 8 KiB/token store, weak-reuse regime, selector scaling, update invalidation, positional extrapolation, and lack of closest-system parity are all acknowledged. This restraint materially improves trust.

## S3. Timing boundaries are separated more carefully than in most efficiency papers

**Anchor:** Sec. 5 setup, PDF p. 4 lines 267–271; Appendix A.4, PDF pp. 17–19.

The manuscript distinguishes fixed-pack Read, store-ready online prefill, Write-inclusive pipeline, and external I/O/index cost. It does not present the 64.9× dense-context number as the causal depth effect. Three independent processes and process-level ratio consistency support the 1.403× isolated-Read result.

## S4. The mechanism diagnosis uses interventions rather than correlational probing alone

**Anchor:** Sec. 5.3, PDF p. 5 lines 314–331; Tables 5–6.

The continuous-prefix ceiling, context × position factorial, and overlap widths are sensible controls. The paper explicitly notes the non-additive interaction and limits the conclusion to one paired multikey cohort.

## S5. Reproducibility reporting is extensive

**Anchor:** Appendix Tables 24–26, PDF pp. 17–18; Appendix B, pp. 20–21.

The paper reports checkpoint revision, split, dimensions, masks, positions, BM25 details, decoding, optimizer, trainable projections, weight hash, training steps/tokens/hardware, benchmark support, prompts/scorers, shard integrity, uncertainty, and compute omissions. This is strong documentation even though the artifact itself was not part of the frozen material I was allowed to inspect.

## S6. The paper performs useful statistical checks

**Anchor:** Appendix A.4–A.6 and B.2–B.4, PDF pp. 18–21.

Examples include paired bootstrap and exact McNemar for RULER, conversation-cluster bootstrap for LoCoMo, an independent-judge subset, and explicit caveats on independence and seed mismatch. The paper generally distinguishes point estimates, evaluation uncertainty, and training-seed uncertainty.

# Weaknesses

## W1. The advertised “frontier” is not measured as one unified, matched multi-depth frontier

- **Location:** Abstract; Sec. 3; Table 2; Appendix Table 9 (PDF pp. 1, 3, 5, 11).
- **Exact quote (12 words):** “These findings establish a measured quality--latency--storage frontier and a narrow mechanism diagnosis”
- **Problem:** The only fully matched causal depth comparison is \(j=0\) versus \(j=12\). Table 9 adds \(j=6/9/18\), but each point has a separately trained adapter with a different span/parameter count; Write is missing for \(j=12\), and storage bytes/token are invariant across nonzero \(j\). I therefore see a valid two-point trade-off plus a deployment curve, not a jointly measured quality–latency–storage frontier over depth. The text partly acknowledges this but continues to make “frontier” the main contribution.
- **Affected claim/norm:** C10 and the headline contribution. A frontier claim should show enough consistently measured operating points to identify trade-offs, rather than connect unmatched cohorts and missing dimensions.
- **Why it matters:** This limits the main design conclusion—how much depth should be prepaid—to one chosen split. The paper shows that \(j=12\) is possible, but not that it is optimal or Pareto-efficient once Write, Read, quality, and reuse are jointly considered.
- **Sufficient remedy:** Under one frozen selector, examples, timing harness, store tier, generation lengths, and backbone, train/evaluate at least \(j\in\{0,6,9,12,18\}\) with either (a) a controlled adapter budget/span or (b) an explicitly deployment-oriented protocol, and report paired quality, Write, Read, end-to-end crossover, persistent bytes, and uncertainty for every point. Alternatively, narrow every headline from “frontier” to “matched \(j=0\) versus \(j=12\) operating-point comparison.”
- **Severity:** **Major**

## W2. The decision-relevant equal-latency experiment is not reproducibly specified

- **Location:** Sec. 5.2 and Table 4 (PDF p. 5, lines 306–313).
- **Exact quote (9 words):** “then frozen and evaluated on a mixed diagnostic cohort”
- **Problem:** The paper never defines the component tasks, sample counts, per-task weights, metric normalization, calibration sample size, exact online timing boundary, hardware/repetitions, actual latency values/distributions, or whether examples are paired within each component. The aggregate and CI cannot be reconstructed or interpreted: a 64.78 “mixed quality” has no defined unit.
- **Affected claim/norm:** C5, which the paper calls its “decision-relevant negative result.” ARR empirical claims need enough protocol detail to audit what is being averaged and whether “matched latency” is actually matched.
- **Why it matters:** This is the strongest deployment comparison in the paper and is repeated in the abstract, introduction, results, and conclusion. Without a defined cohort and latency measurement, its external meaning is unknowable even if its direction is honest.
- **Sufficient remedy:** Add a table listing each constituent task, held-out IDs/support, metric, normalization, weight, both per-task scores, calibration split and rule, exact median/p10/p90 online latencies for both arms, hardware/software, warmups/repetitions, inclusion/exclusion boundaries, and a paired resampling procedure. Release the frozen calibration and evaluation manifests.
- **Severity:** **Major**

## W3. There is no same-backbone end-to-end comparison against the closest reusable-context systems

- **Location:** Related Work, PDF p. 2 lines 134–137; Limitations, PDF p. 6 lines 373–380.
- **Exact quote (9 words):** “We lack a matched same-backbone implementation of these systems”
- **Problem:** The paper compares CoMem internally to raw replay and externally mostly to full-context, eviction/compression, or different-backbone systems. It does not establish where the proposed residual object sits relative to CacheBlend/TurboRAG/EPIC/APE/KV Packet/Cartridges under the same model, quality target, storage tier, corpus reuse, and timing boundary.
- **Affected claim/norm:** C10 and practical significance/novelty. The paper does not claim superiority, so this is not a contradiction; however, a systems paper centered on a new reusable object needs at least one closest-interface baseline to establish whether the object is competitive enough to matter.
- **Why it matters:** The measured 1.403× isolated-Read gain may disappear or be dominated by systems that reuse layer-wise KV with selective repair, and the residual store's 8 KiB/token advantage over full KV must be traded against reconstruction/repair and quality under identical conditions.
- **Sufficient remedy:** Implement at least one strongest feasible PIC/chunk-KV repair baseline and one learned modular-object baseline (or obtain authors' implementation), on Qwen3-8B and the same chunks/tasks. Report persistent bytes, Write, selector/fetch, TTFT/Read/decode, quality, crossover versus reuse count, and HBM/CPU/NVMe placement. If implementation is infeasible, materially narrow the contribution to an internal characterization paper and avoid broad system-design implications.
- **Severity:** **Major**

## W4. Some main-text cross-task matched-baseline numbers are not presented in any table or protocol-complete result

- **Location:** Sec. 5.1, PDF p. 4 lines 287–293.
- **Exact quote (10 words):** “97.2 versus 69.0 on LongEval, 41.59 versus 38.27 on LoCoMo”
- **Problem:** The source contains no row/table for the claimed matched \(j=0\) LongEval 97.2 or LongBench 12.31. Existing Tables 18 and 20 show CoMem and external/reference rows, but not the matched \(j=0\) arm described in this sentence. The LoCoMo 41.59 value does appear in Table 8.
- **Affected claim/norm:** The scope claim that the quality ordering “largely holds outside RULER.” Reported empirical numbers should be traceable to a displayed table with support and protocol.
- **Why it matters:** These values are used to generalize the core RULER trade-off to natural tasks, yet two of three matched comparisons cannot be audited from the frozen paper.
- **Sufficient remedy:** Add the matched \(j=0\) rows for every claimed benchmark, with task-level values, sample counts, prompt/generation settings, paired uncertainty, and exact relationship to Tables 18–20; otherwise delete the undisplayed numbers and weaken the cross-task statement.
- **Severity:** **Major**

## W5. The self-distillation objective is under-defined at the token/support level

- **Location:** Sec. 4, Eq. 2 (PDF p. 4 lines 228–241) and Appendix Table 25.
- **Exact quote (6 words):** “on the teacher's top-64 logit support”
- **Problem:** The paper states that teacher and student distributions are renormalized on the teacher's top-64 support and uses symmetric KL, but does not specify how student mass outside that support is treated, whether the support is recomputed at every query token, how padding/chunk boundaries are masked in the four-chunk training windows, or whether the teacher sees one continuous 2,048-token context while the student sees independently written chunks plus query. “The loss is applied only to the query segment” still leaves the construction ambiguous.
- **Affected claim/norm:** C3 and reproducibility of the principal learned component. Small differences in truncated-vocabulary KL and teacher/student context construction can change gradients materially.
- **Why it matters:** The adapter is necessary for the claimed operating point; an underspecified objective prevents faithful reproduction and makes it harder to reason about what the adapter learns.
- **Sufficient remedy:** Give pseudocode for teacher/student forward construction, token segmentation, causal masks/positions, top-64 selection per token, renormalization, treatment of outside-support logits, reduction over tokens/batch, and numerical stabilization. State explicitly which context interactions are available to each arm during training.
- **Severity:** **Minor**

## W6. Training-seed evidence is not a controlled estimate of the flagship training variance

- **Location:** Appendix A.5 (PDF p. 18, lines 895–907).
- **Exact quote (11 words):** “The two added seeds use effective batch 3 rather than 8”
- **Problem:** Seed is confounded with effective batch size and hence optimizer noise/update dynamics. The text correctly labels this a robustness check, but the main Limitations still says the principal adapter is trained once, and several headline natural-task results remain single-adapter point estimates.
- **Affected claim/norm:** C3 and uncertainty/reliability. Multi-seed statements should hold the training protocol fixed.
- **Why it matters:** Depth points differ by only 0.74–2.22 RULER points between \(j=6,9,12\), comparable to the reported cell-wise seed variation; model-selection stability is therefore not fully established.
- **Sufficient remedy:** Train at least three \(j=12\) adapters with identical global batch, schedule, data order policy, and evaluation suite; report mean/SD or paired hierarchical intervals on the headline RULER, LongEval, LoCoMo, and BABILong aggregates. Ideally repeat the nearest competing split \(j=9\).
- **Severity:** **Minor**

## W7. The break-even table is too sparse to support the broad 128k deployment summary

- **Location:** Table 3 and Sec. 5.2 (PDF p. 5).
- **Exact quote (10 words):** “At 128k and one generated token, break-even is 25.8/27.6 queries”
- **Problem:** At 128k, only \(G=1\) is retained; all longer-generation cells are absent. The abstract summarizes “26–28 at 128k, depending on generation length and store placement,” although the displayed 128k values vary only by store placement, not generation length. Raw component times and dispersion are said to be archived rather than displayed.
- **Affected claim/norm:** C4 and precision of headline efficiency claims. The abstract should not imply measured variation along an unmeasured dimension.
- **Why it matters:** Generation dominates the 32k \(G=512\) CPU case (94 queries), so extrapolating the 128k one-token result to realistic answer lengths may be highly misleading.
- **Sufficient remedy:** Measure 128k for the same \(G\in\{1,32,128,512\}\) grid on both placements, report component medians/dispersion and non-finite cases, or change the abstract to “25.8–27.6 queries at 128k for one generated token.”
- **Severity:** **Minor**

# Questions That Could Change the Score

1. **Equal-latency cohort:** What exact tasks, supports, metrics, weights, and latency values produce 64.78/53.22? If this is a large, preregistered, paired, protocol-complete natural/synthetic mixture and the paper can expose it in one table, W2 could be substantially reduced.
2. **Missing matched rows:** Where are the \(j=0\) LongEval 97.2 and LongBench 12.31 results? If these were inadvertently omitted but exist with paired task-level outputs and consistent prompting, W4 may be a presentation defect rather than an evidence gap.
3. **Closest-system parity:** Do the released artifacts include a same-Qwen implementation of any PIC/modular-cache method that is simply not reported? A credible matched end-to-end result could move the paper toward Findings.
4. **Frontier completion:** Are per-depth \(j=6/9/12/18\) Write and serving-crossover measurements available under the same harness? Completing this would directly strengthen the central design conclusion.
5. **Training construction:** Does the teacher process a continuous four-chunk sequence while the student receives four independently written residual chunks? Please provide exact pseudocode for the top-64 symmetric-KL computation.

# Non-scoring Suggestions and Typos

1. Figure 1 is informative but visibly crowded: several labels overlap or run into box boundaries (e.g., the overlap-Write note and the lower/upper-layer annotation). A simpler vector redraw would improve readability.
2. The source comment in `main.tex` says “verified from saved result shards (2026-07-23).” Comments are not rendered, but submission source should avoid unnecessary process-specific timestamps.
3. Table 12 is called a “Single-pass selector sweep,” while the flagship selector is iterative and `rounds=0` means three automatic rounds. Rename it more explicitly to prevent confusion.
4. Use one unit spelling consistently: KiB versus KB.
5. In Table 3, define \(I_{j=0}\) in prose, not only in the equation.
6. “full prefix” with “\(O(N)\) forward” in Table 6 could clarify whether each chunk's entire preceding document is rerun and whether this is quadratic total Write work over a document.
7. The bibliography has a large blank remainder on PDF p. 9. This is not a page-limit violation, but the references could be typeset more compactly.

# Score Rationale

## Soundness: 3.0 / 5

The central \(j=0\)/\(j=12\) result is technically plausible, well controlled, statistically supported, and honestly scoped. The formula for residual/full-KV storage is correct under the stated GQA architecture and common dtype; arithmetic checks agree. The main deductions about the continuous-prefix ceiling and local context repair follow from the experiments. Soundness is held at 3.0 because several prominent supporting claims are not protocol-complete (especially Table 4), some cross-task numbers are undisplayed, and the unified “frontier” interpretation exceeds the directly matched evidence.

## Excitement: 3.0 / 5

Treating depth as an explicit reusable-computation axis and measuring a residual-state operating point is interesting. However, the conceptual ingredients—reusable intermediate representations, activation/residual checkpoints, modular caches, and learned interface repair—have close precedents. The novelty is mainly the particular persistent one-residual/token object and its matched measurement. Lack of a closest-system matched comparison keeps the likely practical impact uncertain.

## Overall: 2.5 / 5

This is a serious, unusually transparent paper with a publishable core observation, but I am not confident it reaches Findings in the frozen form. Three issues touch the central contribution rather than peripheral polish: the “frontier” is effectively a two-point matched result, the headline equal-latency experiment is not auditable, and the closest reusable-cache alternatives are not compared end-to-end on the same backbone. I was uncertain between 2.5 and 3.0 and choose the lower bin as instructed because these omissions prevent a reliable deployment conclusion. A revision that fully specifies Table 4, displays all matched natural-task rows, and adds either a complete multi-depth frontier or one matched closest-system baseline would plausibly merit 3.0.

## Confidence: 4.0 / 5

I read the rendered 21-page PDF twice, including all appendices; inspected both figures and all 34 numbered tables; checked source equations, labels, references, placeholders, and rendered layout; recalculated central arithmetic; and audited all 43 bibliography entries to the extent allowed by available metadata and network limits. Confidence is not 5 because I could not execute the unreleased artifacts or independently reproduce GPU/API results, and several metadata endpoints were rate-limited.

## Reproducibility: 3.5 / 5

The paper's configuration reporting is substantially above average and includes a checkpoint revision and adapter SHA-256. Reproducibility is reduced by the underspecified equal-latency cohort, incomplete self-distillation pseudocode, undisplayed matched natural-task rows, unavailable raw timing dispersion in the paper, and lack of a recorded total compute budget/training peak memory. The claimed anonymous archive was not among the frozen materials I was permitted to read, so artifact executability remains unverified.

# Technical Audit

## Formulas and boundary cases

- **Eq. 1:** \(|h_j|/|\mathrm{KV}| = d/(2Ln_{\mathrm{kv}}d_{\mathrm{head}})=n_q/(2Ln_{\mathrm{kv}})\) is correct for one residual per token versus K and V for every layer at equal dtype, using \(d=n_qd_{\mathrm{head}}\). For Qwen3-8B, \(32/(2\cdot36\cdot8)=1/18\); 4096 bf16 values = 8192 B and full KV = 147,456 B = 144 KiB/token.
- **Store arithmetic:** 128k × 8192 B is approximately 0.98 GiB, consistent with “about 1 GiB.”
- **Eq. 2:** Symmetric weighted KL is mathematically valid after renormalization, but its truncated-support implementation needs more definition (W5).
- **Layer boundaries:** Half-open ranges are consistently used. \(j=0\) is a valid token-replay endpoint. \(j=L\) is not evaluated and would leave only final normalization/head behavior; the method text does not overclaim this boundary.
- **Read bound:** \(\mathrm{sink}+kc+\mathrm{query}=1+12\cdot512+512=6657\) is correct as a nominal cap. The paper appropriately states that store, index, and selector are not bounded.
- **Overlap FLOPs:** The theoretical ratios match \((c+w)/c\): 1.0625, 1.125, 1.25 before model-specific overhead; reported 1.057/1.115/1.229 are plausible but the exact FLOP accounting is not derived.
- **Break-even equation:** Algebraically sensible if selection is identical and cancels, but raw components/uncertainty should be displayed.

## Numerical checks (abstract/headline)

1. **Read speedup:** \(931.9/664.4=1.4026\), rounds to 1.403×.
2. **RULER-B CoMem macro:** displayed 15 cells sum to 1441.0; \(1441/15=96.0667\), rounds to 96.07.
3. **Quality gap:** \(99.19-96.07=3.12\).
4. **Equal-latency gap:** \(64.78-53.22=11.56\).
5. **Context repair:** \(100.0-92.5=7.5\); position-only change \(88.0-92.5=-4.5\); interaction is +4.5.
6. **Overlap repair:** \(98.5-92.5=6.0\).
7. **Training tokens:** \(4000\times8\times2048=65.536\)M, consistent with 65.5M.
8. **Final training compute:** \(8\times22/60=2.93\) H20 GPU-hours, consistent with 2.9.
9. **LoCoMo denominator:** 282+321+96+841=1540; +446=1986.

No arithmetic contradiction was found in these checks. The abstract's “26–28 at 128k, depending on generation length and store placement” is nevertheless broader than Table 3, which only retains \(G=1\) at 128k (W7).

## Baselines and benchmark validity

- The internal \(j=0\) baseline is excellent for isolating depth reuse.
- The continuous-prefix control is a fidelity ceiling, not deployable; the paper labels it correctly.
- External KV-Direct/InfLLM/StreamingLLM/MemoryLLM/LLoCO rows have heterogeneous backbone, adaptation, context extension, and prompts. The paper mostly treats them as descriptive, which is fair.
- SnapKV/PyramidKV answer a different full-prefill/eviction question and are not closest cross-query baselines; the paper says so.
- RULER and BABILong synthetic tasks are appropriate for controlled evidence/readout diagnosis but cannot alone establish natural-task utility.
- LongBench scores are very low across same-backbone methods, limiting practical interpretation.
- LoCoMo's primary score depends on an undated `gpt-4o` endpoint. The cluster bootstrap and independent-judge subset help, but exact future reproduction is impossible without a dated snapshot or released decisions.
- The PG-19/InfiniteBench overlap audit is commendable. NarrativeQA and other natural benchmarks were not equivalently audited, as disclosed.

## Statistics and seeds

- Paired RULER bootstrap and exact McNemar are appropriate and the effect is clear.
- Reporting 83 full-only versus 1 \(j=12\)-only correct makes the direction transparent.
- LoCoMo's conversation-cluster bootstrap is preferable to treating 1,540 questions as independent; only 10 clusters still yield unstable tail inference, which the paper acknowledges.
- Timing medians over three process medians are better than one run but remain thin for tail-latency/system claims.
- Chunk-size, dense-retriever, store-scaling, and Hy3 values are mostly point estimates with small \(n\); the paper generally scopes them as diagnostics.
- Three “seed” adapters are not a controlled seed study because global batch differs (W6).

## Compute and scope

- Final adapter compute is reported, but total project GPU-hours and training peak memory are missing by admission.
- The paper does not claim a production end-to-end speedup, and its main speedup excludes retrieval, Write, I/O, and decode. This boundary is explicit and acceptable.
- The practical scope is repeated queries over stable English text with enough reuse to amortize a large, model-version-specific store.

# All Figures and Tables Audit

## Figures

- **Figure 1 (p. 2):** Semantically consistent with the method and useful for distinguishing Write/Select/Read and \(j=0\). No hidden content detected. Several labels visibly collide/overflow; readability issue only.
- **Figure 2 (p. 3):** Caption correctly presents probes as motivation, not validation. Panel (a)'s star is schematic relative to sparse plotted points, and panel (b)'s normalized “knee” quantity is protocol-dependent; the appendix defines it. No central claim relies solely on this figure.

## Main-paper tables

- **Table 1:** Taxonomy only; no empirical ranking. Categories are broadly reasonable. It omits LLMCache (2025), a relevant layer-wise intermediate-activation reuse work found in novelty search.
- **Table 2:** Strong central matched result; timing excludes major pipeline components but states this clearly.
- **Table 3:** Useful crossover grid; sparse at 128k and raw components/dispersion are off-page (W7).
- **Table 4:** Numerically clear but protocol-incomplete (W2).
- **Table 5:** Valid paired factorial diagnostic; interaction is correctly noted.
- **Table 6:** Useful local repair; only theoretical Write FLOPs and synthetic support.

## Appendix tables

- **Table 7:** Correctly labels 64.9× as select-first online prefill, not depth-only/end-to-end.
- **Table 8:** Good same-\(j\) adapter control and frozen-depth sweep; no uncertainty shown for natural-task aggregates.
- **Table 9:** Valuable deployment curve, but adapters differ and \(j=12\) Write is missing; not a controlled depth frontier.
- **Table 10:** Strong same-\(j\) RULER adapter on/off effect.
- **Table 11:** HCache-style adaptation transfer is suggestive; exact HCache construction is insufficiently detailed for a head-to-head conclusion, which the paper avoids.
- **Table 12:** Peak-over-\(k\) selector results are exploratory and can be optimistic; caption discloses this.
- **Table 13:** Dense retriever diagnostic reports recall and reader quality, but “raw-text reader quality” metric/support differs by column and is not fully explained in the table.
- **Table 14:** Nominal-budget comparison supports retrieval versus recency, but examples are unpaired as stated.
- **Table 15:** Cross-chunk attention ablation is informative; small RULER \(n=50\) and no uncertainty.
- **Table 16:** Chunk-size point estimates are noisy/non-monotonic; caption appropriately avoids a stability claim.
- **Table 17:** Full RULER grid; cohort A/B distinction is carefully handled.
- **Table 18:** LongEval table is clear, but it does not contain the matched \(j=0=97.2\) main-text value.
- **Table 19:** Full BABILong grid; meaningful task heterogeneity.
- **Table 20:** LongBench table is clear, but it does not contain the matched \(j=0=12.31\) main-text value; LLoCO incomparability is disclosed.
- **Table 21:** YaRN comparison is useful; it also shows severe variable-tracking instability. Not a clean model comparison because methods interact differently with YaRN.
- **Table 22:** LoCoMo table and uncertainty references are detailed; undated judge snapshot remains a reproducibility limitation.
- **Table 23:** Directly demonstrates fixed model Read tokens and growing BM25 latency. Generation \(n=10\) is too small for stable score trends; caption says so.
- **Tables 24–26:** Excellent configuration/training/evaluation documentation. Missing total compute, peak train memory, and complete distillation pseudocode remain.
- **Table 27:** Correctly separates full-prefill KV compression from cross-query persistence; native macro truncation makes headline macro comparison weak, and caption explains it.
- **Table 28:** Replicates Table 2 timing with p10/p90 and process details; strong evidence for isolated Read.
- **Table 29:** Continuous-prefix oracle is a valid upper-bound attribution, not a reusable system.
- **Table 30:** Useful storage-tier microbenchmark; 16M-token stress point and fixed 50.3 MB fetch are clear. “Peak QPS” uses best concurrency rather than a latency–throughput curve.
- **Table 31:** Hy3 self-distillation uses only 16 PG-19 documents; useful implementation check, not broad evidence.
- **Table 32:** Hy3 \(n=50\) needle results support bounded operation to 256k; no raw replay or closest baseline, so not a replicated frontier.
- **Table 33:** Correctly exposes exact RULER-B cells and macro arithmetic.
- **Table 34:** Category denominators sum correctly; category 5 scores are extremely low for all methods and are locally rather than API judged, but this is clearly defined.

# Desk-Reject, Formatting, Anonymity, and Ethics Audit

- **Page limit:** Main text, including Limitations and Ethical Considerations, ends on PDF p. 7; references occupy pp. 7–9; appendix starts p. 10. This is consistent with an eight-page main-paper body allowance. I see no page-limit desk-reject risk.
- **Limitations:** An exact unnumbered `Limitations` section is present and substantive (PDF pp. 6–7).
- **Ethics:** A substantive `Ethical Considerations` section is present (p. 7). It discusses sensitive-source disclosure, residual inversion/membership inference, authorization/isolation/deletion, energy, and data provenance. No human-subject collection is claimed.
- **Anonymity:** Rendered author is “Anonymous ACL submission”; no author affiliation or self-identifying acknowledgment appears. PDF metadata has no author/custom metadata. The source includes generic timestamps, hardware, hashes, and references to a shared result filesystem, but no identity that I could infer. Low anonymity risk.
- **Official style:** Uses `\usepackage[review]{acl}`, line numbers, A4 ACL layout, and standard fonts. No style-manipulation issue found.
- **Unresolved references:** No rendered `??`; source label/ref check found 55 unique labels, no duplicates, and no missing referenced labels.
- **TODO/placeholders:** Mechanical scans found no TODO/TBD/FIXME/XXX/`??` placeholders.
- **Prompt injection/reviewer manipulation:** I searched source, bibliography, figure PDF strings/metadata, and rendered text for reviewer instructions, score manipulation, hidden/white/tiny text, and “ignore previous” style attacks. None found. `\scriptsize` is used for dense tables but not hidden text.
- **Ethical risk level:** Ordinary-to-moderate deployment risk for persistent potentially sensitive representations; mitigations are discussed. No ethics-based rejection recommendation.
- **Desk recommendation:** **No desk reject.**

# Citation Audit

I audited all 43 entries actually present in `main.bbl`. “Verified” means title/authorship/year/identifier or venue matched an authoritative DOI registry, ACL/Crossref record, arXiv API record, or official model page during this review. “Metadata error” means the work exists but the bibliography materially misstates publication metadata. “Unverifiable” means network rate limits or lack of a stable independent record prevented complete verification; it does **not** mean “not found.”

## Complete `main.bbl` metadata audit

| Key | Status | Audit note |
|---|---|---|
| `cachecraft` | Verified | DOI 10.1145/3725273 matched title, authors, and 2025 PACM record. |
| `longbench` | Metadata error | Work verified; bibliography gives only 2023 arXiv, while it has ACL 2024 long-paper publication (10.18653/v1/2024.acl-long.172). |
| `pyramidkv` | Verified | arXiv:2406.02069 title/date matched. |
| `kvpacket` | Verified | arXiv:2604.13226, first posted 2026-04-14; title/authors matched. |
| `cartridgesbase` | Verified | arXiv:2506.06266, 2025; title/authors matched. |
| `hcache` | Verified | DOI 10.1145/3689031.3696072 matched EuroSys 2025 metadata. |
| `llama3` | Verified | arXiv:2407.21783 exists with matching title/year; author truncation is acceptable. |
| `cartridges` | Verified, concurrent | arXiv:2606.04557 first posted 2026-06-03, inside the three-month window. |
| `distillation` | Verified | arXiv:1503.02531 matched title/authors/year. |
| `ruler` | Verified | arXiv:2404.06654 matched. |
| `lora` | Metadata error | Work verified; bibliography year/venue say ICLR 2022 but omit arXiv:2106.09685. This is minor, not claim-affecting. |
| `epic` | Verified | arXiv:2410.15332 and ICML 2025 metadata matched. |
| `ragcache` | Verified | arXiv:2404.12457 matched. |
| `babilong` | Verified | Title and NeurIPS Datasets/Benchmarks 2024 record matched. |
| `rag` | Verified | NeurIPS 2020 title/authors matched. |
| `longchat` | Verified | Official LMSYS 2023 blog title/authors matched; this is a blog, not an archival benchmark paper. |
| `snapkv` | Verified | arXiv:2404.14469/title matched. |
| `ilre` | Verified | arXiv:2508.17892 first posted 2025-08-25; title/authors matched. |
| `readonce` | Verified | ACL DOI 10.18653/v1/2021.acl-long.554 matched. |
| `minicache` | Verified | arXiv:2405.14366 matched. |
| `turborag` | Metadata error | Work verified; bibliography labels the 2024 arXiv only, but an EMNLP 2025 publication exists (10.18653/v1/2025.emnlp-main.334). |
| `locomo` | Metadata error | Work verified; bibliography gives only 2024 arXiv, while ACL 2024 publication exists (10.18653/v1/2024.acl-long.747). |
| `xccache` | Verified | ACL DOI 10.18653/v1/2024.findings-emnlp.896 matched. |
| `kvdirect` | Verified | arXiv:2603.19664, first posted 2026-03-20; title/authors matched. |
| `pg19` | Verified | arXiv:1911.05507/title/authors matched; citation is to the Compressive Transformer paper that introduced PG-19. |
| `bm25` | Verified | DOI 10.1561/1500000019 matched title/authors/pages. |
| `embeddingrecycling` | Verified | ACL DOI 10.18653/v1/2023.findings-eacl.145 matched. |
| `gemfilter` | Metadata error | arXiv:2409.17422 verified; a Findings of ACL 2026 publication exists, so arXiv-only metadata is stale at the 2026 freeze. |
| `reform` | Verified | arXiv:2506.01215 matched. |
| `lloco` | Metadata error | Work verified; bibliography gives only arXiv, while EMNLP 2024 publication exists (10.18653/v1/2024.emnlp-main.975). |
| `hunyuan` | Verified | Official `tencent/Hy3` model page exists; model card/API confirms Hy3, 192 experts, top-8, Apache-2.0. The bibliography is a model-page citation, not a paper. |
| `fusionrag` | Metadata error | arXiv:2601.12904 verified; arXiv metadata exposes DOI 10.1145/3786655, omitted here. |
| `mepic` | Verified | arXiv:2512.16822 matched. |
| `longmem` | Verified | NeurIPS 2023 work/title/authors matched. |
| `memoryllm` | Verified | arXiv:2402.04624 matched. |
| `infllm` | Verified | arXiv:2402.04617/title matched; conference metadata exists but arXiv citation is identifiable. |
| `streamingllm` | Metadata error | Work verified as arXiv:2309.17453 / ICLR 2024; bibliography lacks identifier and gives publication year only. |
| `sempic` | Verified, concurrent | arXiv:2607.28069 first posted 2026-07-30, inside the three-month window. |
| `xu2024retrievallong` | Verified | ICLR 2024 title/authors matched. |
| `qwen3` | Verified | arXiv:2505.09388 matched. |
| `ape` | Verified | arXiv:2502.05431 matched. |
| `cacheblend` | Metadata error | Work verified; bibliography gives 2024 arXiv only, while EuroSys 2025 publication exists (10.1145/3689031.3696098). |
| `h2o` | Verified | NeurIPS 2023 title matched. |

**Counts:** 33 Verified (including two explicitly concurrent), 10 Metadata error, 0 Not found, 0 Unverifiable in the final table. Some records were initially rate-limited, but enough independent metadata was available before the user-directed stop to classify them without treating network failure as absence.

The metadata errors are mostly stale arXiv-only citations rather than fabricated works. They should be corrected for archival accuracy but do not by themselves alter the paper's technical conclusions.

## Citation–claim match audit (load-bearing samples)

1. **CacheBlend/TurboRAG/Cache-Craft as reusable chunk-KV systems:** **Matched.** Their titles/abstracts explicitly concern precomputed/reused KV for RAG; the paper's broad taxonomy is supported.
2. **EPIC/MEPIC/APE as position-independent or parallel context encoding with repair/realignment:** **Matched with nuance.** EPIC and MEPIC directly fit PIC; APE is parallel encoding/attention adaptation, so grouping is reasonable but not identical.
3. **KV Packet as independently compiled KV plus trainable adapters and no document recomputation:** **Matched.** Its abstract explicitly describes immutable packets and lightweight trainable soft-token adapters.
4. **Cartridges as distilled reusable KV representations:** **Matched.** Cartridges train small reusable KV caches offline via self-study/context distillation.
5. **HCache and KV-Direct as activation/residual precedents:** **Matched.** HCache checkpoints activations for restoration; KV-Direct reconstructs KV from residual streams.
6. **ILRe/REFORM as selecting/gathering tokens before recomputation:** **Matched.** Their abstracts describe intermediate-layer token recall or gather/recompute pipelines; neither is identical to persistent CoMem, as the paper says.
7. **MemoryLLM/LongMem/XC-Cache as external-memory or auxiliary-reader references:** **Mostly matched.** LongMem and MemoryLLM are long-term-memory methods; XC-Cache uses cross-attention to cached context. Calling all three “latent pools or auxiliary readers” is a high-level compression but adequate for related-work positioning.
8. **PG-19 as the adapter corpus:** **Matched but indirect.** The cited Compressive Transformer paper introduced PG-19; a dataset-specific citation/URL would be clearer for licensing and versioning.

# Novelty Search and Closest Works

Freeze date is **2026-08-03**. Under the requested three-month rule, works first public after **2026-05-03** are treated only as concurrent, not prior art. I ran five targeted searches over reusable intermediate activations/residual streams, layer-wise caching, modular/PIC caches, offline hidden-state memory, and depth reuse. Search was stopped on user instruction; results below are the closest identified works, not a claim of exhaustive coverage.

## Search queries and findings

1. **“reusable representations text transformers intermediate layer cache”**  
   Found ReadOnce Transformers and, importantly, **LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformer Inference** (published 2025-12-18).
2. **“cache residual stream per token resume upper layers reusable document” / “residual stream cache retrieval”**  
   Found **KV-Direct / The Residual Stream Is All You Need** (2026-03-20).
3. **“position independent caching LLM reusable chunks KV residual”**  
   Found EPIC, MEPIC, KV Packet, SemPIC, and related PIC systems.
4. **“document cache self distillation reusable KV long context”**  
   Found Cartridges, KV Packet, and modular learned caches.
5. **“hidden state reusable language model memory”**  
   Found concurrent TransMem (2026-07-31) and Memory Grafting (2026-05-20); these use hidden states as memory but have materially different interfaces/objectives.

## Closest works

1. **LLMCache: Layer-Wise Caching Strategies for Accelerated Reuse in Transformer Inference** (2025-12-18; prior art).  
   This is the most important omission from Related Work. It explicitly caches intermediate activations at arbitrary transformer layers for reuse based on semantic similarity. Its models/workloads differ (BERT/GPT-2 and approximate input matching), and it does not appear to study a bounded retrieved residual pack plus exact \(j=0\) depth frontier. Nevertheless, it narrows the novelty of “depth as a reuse axis” and should be discussed.
2. **ReadOnce Transformers** (2021; prior art).  
   Reusable intermediate document representations with later adapted computation, evaluated across tasks. CoMem differs by using decoder residuals per token, retrieval-bounded packs, direct upper-layer continuation, and systems timing.
3. **KV-Direct / The Residual Stream Is All You Need** (2026-03-20; prior art).  
   Establishes that one residual per token can reconstruct per-layer KV and uses residual checkpoints for bounded-memory inference. CoMem differs in persistent cross-query document storage, selection, and resuming only upper layers rather than reconstructing/replaying the full cache path.
4. **KV Packet** (2026-04-14; prior art by 19 days before the cutoff).  
   Context-independent reusable document KV with self-supervised distillation and adapters, designed to avoid recomputation. This is a very close learned modular-object baseline; CoMem's distinction is a single residual/token at one split rather than per-layer KV packets.
5. **Cartridges / Cartridges at Scale** (base 2025-06-06 prior art; at-scale version 2026-06-03 concurrent).  
   Offline learned reusable KV objects amortized across queries. CoMem is less compressed and more direct, but its practical comparison to this family is unresolved.

**Concurrent only:** SemPIC (2026-07-30), TransMem (2026-07-31), Memory Grafting (2026-05-20), and Cartridges at Scale (2026-06-03) fall after 2026-05-03. They may be discussed as concurrent context but should not be used to reject novelty.

## Novelty assessment

The paper is **incrementally but meaningfully novel**, not a first demonstration that intermediate representations or document caches can be reused. Its strongest distinct contribution is the specific persistent one-residual/token, selectable decoder interface and the tightly matched \(j=0\)/\(j=12\) measurement with storage/amortization accounting. The omission of LLMCache and lack of empirical comparison to KV Packet/PIC/Cartridges weaken the novelty positioning, but I did not find an earlier work that exactly combines the same object, bounded retrieval, direct suffix execution, and matched depth-reuse experiment.

# Review-Process Self-Check

- [x] Read the full rendered PDF twice, including appendices.
- [x] Read only the allowed frozen PDF/source/template; did not inspect other reviews, histories, TODOs, status files, or current drafts.
- [x] Built explicit claims C1–C10 and mapped minimum evidence to actual evidence.
- [x] Inspected both figures and all 34 numbered tables in the rendered PDF.
- [x] Checked Eq. 1, Eq. 2 semantics, layer boundaries, store/read arithmetic, and break-even assumptions.
- [x] Recomputed at least five abstract/headline numbers (nine checks reported).
- [x] Audited benchmark supports, metrics, uncertainty, seeds, compute, scope, and reproducibility.
- [x] Audited all 43 actually cited `main.bbl` entries; did not turn rate limiting into “Not found.”
- [x] Checked eight load-bearing citation–claim matches.
- [x] Ran five novelty-search formulations and applied the 2026-05-03 concurrent-work boundary.
- [x] Checked page limit, exact Limitations, ethics, anonymity, ACL review style, unresolved refs, TODO/`??`, and prompt injection/hidden text.
- [x] Mechanically verified all weakness quotes against the frozen source.
- [x] Mechanically checked every “missing/lacks/no same-backbone/not defined” criticism against the complete frozen source and appendices.
- [x] Kept each scoring weakness tied to a claim/norm, significance, sufficient remedy, and Major/Minor label.
- [x] Applied the stated calibration: 4 = ACL main, 3 = Findings; chose the lower score where uncertain and explained why.
