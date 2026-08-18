```yaml
review_mode: strict
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 3.5
```

# Paper Summary

This paper studies a narrowly scoped reusable-context interface, CoMem, for repeated queries over a stable corpus. A one-time WRITE independently processes each 512-token chunk through layers `[0:j)` and persists one depth-`j` residual per token; SELECT retrieves a bounded top-`k` chunk set; READ packs those residuals with the query and executes only `[j:L)`. The main causal comparison fixes retrieved evidence, order, mask, examples, adapter, and hardware while changing replay start from `j=0` to `j=12`. On Qwen3-8B this reduces isolated selected-pack Read latency from 931.9 ms to 664.4 ms (1.403x) while reducing a paired 15-cell RULER-B macro from 99.19 to 96.07 (gap 3.12, 95% CI [2.36, 3.93]). The paper also accounts for the 8 KiB/token store and write amortization, reports selector-dependent equal-latency diagnostics, and localizes one synthetic multikey failure to missing lower-layer document context; a 32-token left-overlap WRITE improves that diagnostic from 92.5 to 98.5. The manuscript carefully frames the work as an internal measurement of a depth-reuse operating point, not a claim of superiority over raw-text RAG or modular KV caches.

# Claim–Evidence Audit

For each claim I first state the minimum sufficient experiment, then the paper's evidence.

- **C1 — Depth reuse yields a real latency/quality trade-off under fixed evidence.** Minimum: paired examples and identical selected tokens/order/mask/adapter, changing only replay start; matched timing boundary and uncertainty. Evidence: Section 5.1, Tables 2/31/32, PDF pp. 5–6 and 21. The same-LoRA `j=0`/`j=12` comparison supplies 1,500 paired RULER examples, paired bootstrap and exact McNemar statistics, plus three-process Read medians. **Supported for isolated selected-pack Read, not end-to-end service.**
- **C2 — One residual/token is much smaller than full layer-wise KV but much larger than text.** Minimum: correct architecture-level byte formula plus concrete dimensions. Evidence: Eq. (1), Section 4, Table 26, PDF pp. 4 and 19. For Qwen3-8B, `d=4096`, `L=36`, `n_kv=8`, `d_head=128`, bf16 give 8,192 B versus 147,456 B/token (1/18). **Supported.**
- **C3 — The store pays off only with reuse.** Minimum: measured Write and per-query costs for both arms, consistent timing boundary/storage tier, and explicit break-even equation. Evidence: Section 5.2, Table 3, appendix efficiency text, Table 33, PDF pp. 5–6 and 20–22. **Supported for the displayed single-query harnesses and retained cells; hardware/workload specific.**
- **C4 — Equal-latency deployment conclusions are selector dependent, and CoMem wins neither tested aggregate.** Minimum: latency-only budget calibration disjoint from quality, per-cell paired scores, dependence-aware uncertainty over heterogeneous cells. Evidence: Table 4, protocol-complete Table 8, Appendix B.4, PDF pp. 6, 12, 24. The new script/JSON reruns byte-identically and reproduces BM25 `-11.5556` with hierarchical CI `[-18.67,-5.11]`, and BGE `-1.00` with `[-10.67,8.33]`. **Supported for this nine-cell diagnostic mixture.**
- **C5 — Missing lower-layer document context is a major tested cause of one multikey gap, and overlap is a local repair.** Minimum: paired factorial intervention over write context and read positions, followed by a deployable intervention with unchanged persistent bytes/read work and uncertainty. Evidence: Tables 5/6/32, Sections 5.3/A.4, PDF pp. 5–7 and 21. **Supported only for the pooled 8k/16k synthetic multikey cohort, as the paper acknowledges.**
- **C6 — Self-distillation makes the residual interface substantially more usable without updating the backbone.** Minimum: same-split on/off adapter comparison and exact objective/training description. Evidence: Section 4, Tables 9/11/12/27, PDF pp. 4, 12–13, 20. **Supported at `j=12`; broader seed stability is limited.**
- **C7 — Model-side Read is bounded by selected-pack length rather than stored-context length.** Minimum: fixed `k,c` across increasing store/context size, reporting read tokens and lookup cost. Evidence: Figure 1, Tables 7/25/35, PDF pp. 2, 11, 19, 23. **Supported for model input length; the paper correctly excludes selector/index/store scaling from the bounded claim.**
- **C8 — The implementation ports beyond one dense 8B model.** Minimum: at least one independent architecture-level partition check and bounded-read evaluation. Evidence: Hy3 exact split self-test and Tables 34/35, PDF pp. 22–23. **Supports implementation portability, not a replicated quality/serving frontier.**

# Strengths

**S1. Strongly controlled central comparison.** Section 5.1 and Tables 2/31/32 (PDF pp. 5–6, 21) isolate replay start while holding evidence, order, sink, mask, examples, and the same 168 mounted LoRA modules fixed. This is substantially more credible than comparing an adapted residual reader against an unadapted text baseline.

**S2. Exceptionally transparent claim boundaries.** The paper repeatedly distinguishes depth-only Read speedup, bounded-selection speedup, Write-inclusive cost, external I/O, and decode (Introduction; Sections 5.1–5.2; Tables 2/3/7/31/33). It explicitly says the 1.403x result is not quality preserving and the 64.9x dense-to-bounded number is not a depth-only or end-to-end causal effect.

**S3. Negative and unresolved results are retained rather than hidden.** Table 4 and Appendix B.4 report the strong BM25 replay win, the unresolved BGE aggregate, fixed-cell/hierarchical/pooled intervals, and leave-one-cell-out sensitivity. Table 25 also exposes bounded-retrieval failure modes. This materially improves trustworthiness.

**S4. Good mechanism diagnosis.** The continuous-prefix fidelity ceiling, the context × position factorial (Table 5), block-diagonal control (Table 16), and overlap intervention (Table 6) separate several plausible causes instead of attributing all loss to “compressed representations.”

**S5. Reproducibility detail is above average.** Tables 26–28 give model revision, dimensions, positions, masks, exact BM25 settings, training objective, optimizer, data budget, decoding, sample supports, and scoring. The new equal-latency analysis script validates row counts, pairing, budgets, cell sizes, stored differences, and LoCoMo clustering; its output JSON was reproducible byte-for-byte from the frozen score exports.

**S6. Scope and ethics are unusually candid.** The exact `Limitations` section (PDF pp. 7–8) covers nearest-baseline absence, one-run training, corpus/model coupling, mutable judge, contamination uncertainty, non-concurrent timing, and overlap repair scope. Ethical Considerations addresses sensitive-state storage, access control, deletion, inversion/membership risk, and licensing.

# Weaknesses

## W1 — No matched nearest modular-cache baseline, so the paper does not establish competitive utility (Major)

- **Location:** Related Work, PDF p. 3, lines 148–154; Limitations, PDF p. 7, lines 440–451.
- **Exact quote:** “We lack a matched same-backbone implementation of”
- **Problem:** The closest deployment alternatives—PIC, chunk-KV repair, and learned modular KV—are only taxonomized. The empirical core compares CoMem to raw-text replay, while a deployer choosing a persistent object needs to know whether one residual/token is preferable to the nearest reusable-context object under the same backbone, hardware, storage tier, quality target, and timing boundary.
- **Affected claim/norm and impact:** This limits **excitement and practical significance**, not the internal `j=0→12` causal claim. Without a nearest-baseline head-to-head, the paper supports a well-measured design point but not an ACL-main-level systems conclusion about which reusable interface is useful.
- **Sufficient remedy:** Implement at least one nearest feasible baseline (e.g., CacheBlend/EPIC-style reusable KV or KV Packet/Cartridges-style modular object) on Qwen3-8B, and compare quality, persistent bytes, Write, fetch, TTFT, decode, and break-even under a matched selected pack and storage tier. If implementation is infeasible, further narrow the contribution to a measurement/diagnostic study and avoid deployment-motivated novelty claims.
- **Severity check:** **Major** because the missing comparison is the minimum experiment needed for external competitive relevance; it does not invalidate the reported internal trade-off.

## W2 — The only fully matched causal frontier has two depth points, and one headline depth lacks matched Write cost (Major)

- **Location:** Why Reuse Model Depth?, PDF p. 3, lines 204–215; Table 10, PDF p. 13.
- **Exact quote:** “the retained j=12 Write value is missing”
- **Problem:** The paper motivates choosing how much depth to prepay, but the rigorously matched result is only `j=0` versus `j=12`. The `j=6/9/12/18` curve uses separately trained adapters with different spans/parameter counts, and the `j=12` matched fixed-pack Write measurement was not retained. Therefore it cannot identify an optimal split or a quality–Read–Write–storage Pareto frontier.
- **Affected claim/norm and impact:** This constrains **C1/C3 and the title-level “depth as reuse axis” thesis**. The paper demonstrates one operating point convincingly, but not the broader design rule implied by the motivating question “how much transformer depth should ... prepay?”
- **Sufficient remedy:** Train/evaluate multiple depths under a compute-matched adapter protocol (or explicitly normalize trainable capacity), retain matched Write/Read/decode/fetch measurements for every depth, and report paired quality with a common cohort. At minimum, recover the `j=12` Write value and present `j=0/6/9/12/18` end-to-end break-even frontiers.
- **Severity check:** **Major** because multi-depth evidence is necessary for the broad depth-selection claim, though the paper carefully labels the current result a two-point endpoint.

## W3 — Main quality evidence relies on one flagship training run; available extra runs confound seed and effective batch (Major)

- **Location:** Limitations, PDF p. 7, lines 430–439; Table 29, PDF p. 21.
- **Exact quote:** “not a clean estimate of training-seed variance.”
- **Problem:** The flagship RULER-B, LongEval, and LoCoMo numbers come from one effective-batch-8 adapter. Seeds 1/2 use effective batch 3 and reduced/different evaluation supports, so their variation cannot isolate random seed, optimizer noise, or batch effects for the actual headline metrics.
- **Affected claim/norm and impact:** This weakens confidence in **C1/C6** and in small natural-task differences (e.g., LoCoMo 38.27 vs. 34.59). It is less concerning for the very large same-split adapter gains, but material for the exact quality cost and cross-method point estimates.
- **Sufficient remedy:** Run at least three adapters with identical effective batch, data order budget, objective, and evaluation cohorts; report run-level mean/SD or hierarchical uncertainty for RULER-B, LongEval, BABILong, and LoCoMo. Separate training-run variability from item/cell bootstrap uncertainty.
- **Severity check:** **Major** under strict calibration because the central learned interface is single-run and current “robustness” changes two factors simultaneously.

## W4 — Equal-latency inference treats nine hand-selected task–length cells as exchangeable and gives one conversation the same weight as a full synthetic cell (Minor)

- **Location:** Table 8, PDF p. 12; Appendix B.4, PDF p. 24.
- **Exact quote:** “The estimand is the equal-weight mean of the nine cell mean paired differences”
- **Problem:** The hierarchical bootstrap correctly reflects variation over the nine observed cell labels, but those labels are not a random sample from a defined deployment-task population. The LoCoMo cell is 100 questions from conversation 0, yet receives one-ninth weight, equal to each benchmark/length cell. Thus the hierarchical interval is a sensitivity analysis over this constructed mixture, not general uncertainty for deployment quality.
- **Affected claim/norm and impact:** This limits **C4's scope**. The directional BM25 conclusion is robust even to LOCO, but the BGE “unresolved” conclusion depends on the chosen mixture and weighting rather than a pre-specified population estimand.
- **Sufficient remedy:** Predefine and justify deployment weights, sample multiple LoCoMo conversations, report benchmark-level and family-level aggregates, and use cluster-aware resampling at the natural unit. Retain the current equal-cell result as one explicit sensitivity.
- **Severity check:** **Minor** because the paper discloses the estimand, cell deltas, and non-identifiable conversation bootstrap; no stronger selector-independent claim is made.

## W5 — Natural-task generalization and the local overlap repair remain weakly validated (Minor)

- **Location:** Section 5.3, PDF p. 5, lines 517–522; Limitations, PDF p. 7, lines 474–485.
- **Exact quote:** “evaluated only on this paired synthetic cohort.”
- **Problem:** The most effective repair (overlap WRITE) is shown only on pooled 8k/16k RULER multikey. Natural benchmarks also have incomplete PG-19 overlap audits, and LongEval shows a large `j=0→12` drop (97.2 to 69.0). Therefore the diagnosed failure mode and repair cannot yet explain or improve the natural-task losses.
- **Affected claim/norm and impact:** This limits **C5 and practical scope**: overlap is a credible engineering hypothesis, not evidence that contextual writing repairs real QA or conversation memory.
- **Sufficient remedy:** Evaluate `w=0/32/128` on at least LongEval, BABILong, LoCoMo, and the six LongBench QA datasets with matched examples, paired uncertainty, and Write-inclusive break-even; complete contamination audits or clearly isolate clean subsets.
- **Severity check:** **Minor** because the paper repeatedly labels the diagnosis local and makes no universal-repair claim.

## W6 — Reproducibility is documentation-rich but the frozen submission does not itself demonstrate a fully runnable end-to-end artifact (Minor)

- **Location:** Reproducibility details, PDF p. 18, lines 925–929; Ethical Considerations, PDF p. 8, lines 557–563.
- **Exact quote:** “The archived configuration files and evaluation scripts accompany the code release.”
- **Problem:** The manuscript names many archived artifacts, but the frozen PDF/source reviewed here does not expose a verifiable anonymous artifact locator or an end-to-end reproduction command. Some critical outputs depend on saved private-filesystem shards and a mutable `gpt-4o` endpoint; total experiment GPU-hours and training peak memory are not retained.
- **Affected claim/norm and impact:** This affects **reproducibility**, especially independent regeneration of headline predictions/judgments, while not undermining the internally reproducible equal-latency score-only reanalysis.
- **Sufficient remedy:** Provide an anonymous archival URL in the submission metadata/supplement, a manifest mapping every table cell to immutable inputs and commands, fixed permissible prediction shards, and a deterministic non-proprietary primary/secondary judge path. Add smoke tests and expected hashes for the main tables.
- **Severity check:** **Minor** because Tables 26–28 are unusually complete and the new statistical script/JSON are internally consistent; the remaining issue is independent artifact execution.

# Questions That Could Change the Score

1. Can the authors provide one same-backbone/same-hardware nearest modular-cache baseline with the same pack, storage tier, and timing boundary? A convincing result could move the paper toward 3.5–4.0.
2. Can the authors report clean same-effective-batch replications for the exact RULER-B/LongEval/LoCoMo headline cohorts? Large instability would lower soundness; tight stability would address W3.
3. Is a matched `j=12` Write measurement recoverable, and do multi-depth end-to-end Pareto/break-even curves preserve the apparent `j=12` operating-point choice?
4. How were the nine equal-latency cells and equal weights selected before seeing quality? Please distinguish a predeclared deployment mixture from an exploratory diagnostic mixture.
5. Does `w=32` improve natural-task quality enough to offset its measured Write overhead under repeated-query break-even, or is the repair specific to synthetic multikey retrieval?

# Technical, Formula, Metric, and Statistical Audit

- **Eq. (1):** Algebraically correct under common dtype and standard per-layer K/V storage: residual/full-KV bytes = `d/(2L n_kv d_head)=n_q/(2L n_kv)`. The Qwen3-8B 1/18 and 8 KiB vs. 144 KiB/token calculations check out.
- **Boundary cases:** `j=0` is clearly defined as token-ID replay through all layers. `j=L` appears only in the Hy3 exact-partition self-test; no deployable `j=L` quality claim is made. `k` is a maximum and selectors may under-fill. Tail chunks/queries explain actual reads below the 6,657 nominal cap.
- **Self-distillation objective:** Eq. (2) is a symmetric weighted KL over distributions renormalized on the teacher top-64 support. The paper correctly discloses that outside-support logits and retained teacher mass were not logged; this makes it an approximate objective, not full-vocabulary distillation.
- **Break-even:** `Q*=(W_CoMem-I_j=0)/(T_j=0-T_CoMem)` is dimensionally correct when shared selection cancels and denominator is positive. The text appropriately marks unavailable/non-finite cells and does not extrapolate 128k beyond one generated token.
- **Metrics:** RULER, BABILong, LongEval, and LongBench scorers/supports are specified. LoCoMo combines GPT-4o for categories 1–4 with local abstention scoring for category 5; denominators are explicit. The mutable judge limits exact reproducibility but a conversation-cluster bootstrap and 200-item independent-judge audit are useful.
- **Statistics:** The main RULER gap uses paired bootstrap and exact McNemar. Equal-latency uses per-example pairing, fixed-cell, hierarchical, pooled sensitivity, and LOCO. The script's definitions match the prose and JSON. Training-run uncertainty remains the main statistical gap.
- **Scope:** Claims are mostly appropriately bounded to selected-pack Read, repeated-query stable corpora, one principal 8B backbone, English, and synthetic/natural scope checks. No unsupported quality-preserving, constant-time retrieval, or universal-repair claim remains.

# New Statistical Script / JSON Consistency Check

I audited `analysis/equal_latency/reanalyze_equal_latency.py`, both 900-row score-only JSONL exports, and `equal_latency_dependence_results.json`.

- The script requires exactly 900 rows/selector, nine cells, 100 pairs/cell, CoMem `k=12`, replay `k=10`, unique `(cell, example_id)` pairs, and exact agreement between stored difference and `100*(CoMem-replay)`.
- It defines: pooled-IID over 900 pairs; fixed-cell stratified resampling of 100 pairs within each of nine cells; hierarchical resampling of nine cell labels followed by independent within-cell resampling; and leave-one-cell-out equal-cell means.
- Rerunning the script with seed 20260804 and 100,000 replicates produced a byte-identical JSON (SHA-256 `2b8789...8917`).
- Paper/JSON agreement: BM25 point `-11.56`, fixed-cell `[-14.33,-8.78]`, hierarchical `[-18.67,-5.11]`, pooled `[-14.44,-8.67]`, LOCO `[-13.13,-9.50]`; BGE point `-1.00`, fixed-cell `[-4.56,2.56]`, hierarchical `[-10.67,8.33]`, pooled `[-4.67,2.67]`, LOCO `[-3.50,1.75]`. Per-cell deltas and seven-negative/two-positive BGE LOCO signs also match.
- The LoCoMo non-clustering statement is consistent with the exports: all 100 retained items are `conv0`.

**Conclusion:** no numerical or definitional inconsistency found in the new statistical script/JSON relative to Tables 4/8 and Appendix B.4.

# Citation Audit

All 43 entries in `main.bbl` are actually cited. Status below follows the requested labels; any item not conclusively checked online is **Unverifiable**, not “Not found.”

- **Verified (39):** Cache-Craft; LongBench; PyramidKV; KV Packet; Cartridges; HCache; Llama 3; Cartridges at Scale; Distilling the Knowledge; RULER; LoRA; EPIC; RAGCache; BABILong; Retrieval-Augmented Generation; SnapKV; ILRe; ReadOnce Transformers; MiniCache; TurboRAG; LoCoMo; XC-Cache; KV-Direct / *The Residual Stream Is All You Need*; PG-19/Compressive Transformers; BM25; Embedding Recycling; GemFilter; REFORM; LLoCO; Fusion RAG Cache; MEPIC; LongMem; MemoryLLM; InfLLM; StreamingLLM; SemPIC; Qwen3; APE; CacheBlend.
- **Metadata error (0):** none established from the completed checks.
- **Not found (0):** none; network/search incompleteness was not converted to Not found.
- **Unverifiable (4):** LongEval/LongChat LMSYS blog citation; Tencent Hy3 model card; *Retrieval Meets Long Context Large Language Models*; H2O venue metadata. These appear plausible in the frozen bibliography, but I did not complete conclusive external metadata verification before the audit was stopped.

**Load-bearing citation–claim matches (8):**

1. CacheBlend/TurboRAG/Cache-Craft support the claim that chunk-KV systems reuse retrieved-document KV and repair/fuse context; match is appropriate.
2. EPIC/MEPIC/APE support position-independent/parallel encoding with boundary or positional repair; match is appropriate.
3. KV Packet/Cartridges/SemPIC support learned modular reusable KV objects; match is appropriate, with Cartridges at Scale/SemPIC treated as contemporaneous work.
4. ReadOnce/Embedding Recycling support reusable intermediate text representations adapted to later layers; match is appropriate and is the closest conceptual precedent.
5. HCache/KV-Direct support checkpoint/residual restoration or KV reconstruction; match is appropriate, though neither is a matched repeated-query baseline here.
6. StreamingLLM/H2O/SnapKV/PyramidKV/MiniCache support token/KV retention or compression rather than persistent single-residual replay; match is appropriate.
7. RULER/BABILong/LongBench/LoCoMo citations match the benchmark protocols used; LongEval's blog metadata is Unverifiable but the described register-content task is consistent with the cited artifact name.
8. Hinton et al. and LoRA correctly support the distillation/low-rank adaptation ingredients, but they do not by themselves validate the paper's top-64 symmetric-KL design; that design is properly presented as the authors' implementation choice.

# Novelty Search Summary (cutoff 2026-05-04)

I stopped further network expansion as requested. Based on the completed bibliography/metadata checks and frozen related-work search space, the closest papers are:

1. **ReadOnce Transformers (2021):** closest conceptual precedent for caching intermediate text representations and adapting later transformer layers.
2. **Embedding Recycling (2023):** reusable intermediate representations across downstream processing; similar representational-reuse motivation.
3. **HCache (2025):** activation checkpoint/replay precedent, but focused on serving-state restoration rather than persistent selected document residuals.
4. **KV Packet (arXiv:2604.13226; before cutoff):** closest learned modular reusable-context object; stores/compiles reusable KV rather than one tunable-depth residual/token.
5. **Cartridges (2025/ICLR 2026):** learned lightweight reusable long-context representations, a close alternative object/training approach.

Three targeted novelty axes were covered: (i) reusable intermediate activations/residual replay, (ii) position-independent or chunk-KV caches, and (iii) learned modular context objects. The defensible novelty is **not persistent context reuse itself**. It is the combination of one residual/token at a tunable split, direct suffix execution on a bounded selected pack, and an unusually controlled `j=0` matched endpoint with Write/Read/storage accounting. **Contemporaneous after-cutoff works** Cartridges at Scale (June 2026) and SemPIC (July 2026) should not be used to diminish novelty under the three-month rule; the paper appropriately labels them concurrent.

# Figures and Tables Audit

I visually inspected every rendered page, Figure 1, Figure 2, and Tables 1–37. Figures are legible at normal zoom; Figure 1 accurately distinguishes default/overlap WRITE, SELECT, and matched replay paths. Figure 2 is explicitly motivational and does not overclaim validation. Table captions generally state cohorts, timing exclusions, and unmatched-baseline caveats. No obvious plot/table value contradiction was found. The appendix is table-dense and several tables use small text, but content remains readable. Tables 18–24 risk casual leaderboard interpretation, yet captions repeatedly mark unmatched/descriptive rows and cohort superscripts.

# Limitations, Ethics, and Desk-Reject Risks

- **Page/style:** Main paper occupies eight pages (PDF pp. 1–8), followed by references and appendices; A4 ACL review style and line numbers are present. No unresolved references/citations or TODO/placeholders were found. The source compiles to 24 pages with only underfull-box warnings.
- **Required section:** Exact unnumbered `Limitations` heading is present before `Ethical Considerations`.
- **Anonymity:** Author is “Anonymous ACL submission”; no author names/affiliations or self-identifying prose found. The review-style footer exposes an anonymous OpenReview-style link, not author identity.
- **Prompt injection/manipulation:** No hidden/white/tiny reviewer-directed text or prompt-injection language found in source/PDF. Rendered and extracted text are consistent.
- **Abstract numbers:** 931.9→664.4 ms/1.403x, 99.19→96.07/gap 3.12 CI, 8 KiB/token, break-even ranges, equal-latency points/CIs, and 92.5→100/98.5 all match the cited tables/appendix.
- **Ethics:** Adequately discusses privacy, state inversion/membership risk, access control, deletion, authorization, misuse, energy, licenses, and absence of new human-subject data.
- **Desk-reject risk:** **Low based on the frozen PDF/source.** Potential administrative checks outside the frozen material (supplement registration, venue-specific submission-form requirements, and artifact URL validity) are Unverifiable.

# Non-Scoring Suggestions / Typos

1. Define “Read” once in a boxed timing-boundary glossary; despite careful captions, four boundaries require effort to track.
2. Use `KiB` consistently in tables as well as prose (8,192 B = 8 KiB); avoid visually mixing `k` tokens with decimal storage units.
3. Table 18's `99.20^B` versus the central `99.19` is explainable as rounded aggregation but should be harmonized or footnoted explicitly.
4. In Table 35, “recall” is used for an output-scored RULER result; “accuracy/string-match score” would avoid confusion with retrieval recall.
5. The title could signal the measurement-study scope more directly, e.g., “Measuring Persistent Intermediate-Residual Memory...”

# Scores

- **Soundness: 3.5/5.** The central matched endpoint, formulas, timing boundaries, paired statistics, and new dependence-aware reanalysis are sound and unusually transparent. I do not assign 4.0 because the learned interface is one-run, the broad depth frontier is not controlled, and natural repair evidence is narrow.
- **Excitement: 3.0/5.** The depth-reuse axis and diagnostic rigor are interesting, but reusable intermediate/context objects have substantial precedent, and no matched nearest modular-cache baseline establishes practical advantage.
- **Overall: 3.0/5 (Findings level).** This is a reliable, candid, technically useful measurement/diagnosis paper with real negative results. Under strict 4/3 calibration, the absence of a nearest-baseline head-to-head and clean run-level replication keeps it below ACL main-conference level.
- **Confidence: 4.5/5.** I read the full 24-page PDF twice including both appendices, inspected all figures/tables, audited formulas/numbers/citations, and reran the new statistics script. Remaining uncertainty is chiefly external novelty/citation coverage after the requested stop.
- **Reproducibility: 3.5/5.** Configuration and statistical detail are strong, and the new JSON is byte-reproducible. Full independent regeneration remains limited by artifact accessibility, large external dependencies, saved shards, mutable GPT-4o judging, and incompletely logged total compute.

# Review-Process Self-Check

- Reviewed only the frozen `v6_20260804_014520` PDF, its build-source inputs/artifacts needed to audit it, the new statistics script/JSON/score exports, `main.bbl`, and the STRICT template; no other review/history/TODO/status/current/calibration file was used.
- Completed two full appendix passes: first for methods/results/reproducibility and second for claims, statistics, denominators, and cross-table consistency.
- Mechanically searched the frozen source/PDF for every exact weakness quote and checked each “missing/lacks” assertion against appendices.
- Checked page/style/anonymity/Limitations/unresolved markers/prompt injection, abstract numbers, formulas, boundary cases, nearest baselines, metrics, seeds/statistics, scope, compute, reproducibility, and every rendered figure/table.
- Scores are independent for v6 and use strict 4.0-main / 3.0-Findings calibration.
