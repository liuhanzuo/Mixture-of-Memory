```yaml
review_mode: strict
soundness: 3.5
excitement: 3.5
overall: 3.0
confidence: 4.0
reproducibility: 2.5
```

## Paper Summary

This paper introduces **CoMem**, a repeated-query long-context serving interface that stores one intermediate residual vector per document token at a chosen transformer split depth $j$, retrieves a bounded set of chunks, and runs only the decoder suffix $[j{:}L)$ online. Its central controlled result compares the same retrieved pack at $j=0$ and $j=12$ on Qwen3-8B: selected-pack Read falls from 931.9 to 664.4 ms (1.403x), while a 15-cell RULER-B macro falls from 99.19 to 96.07 (3.12 points, paired CI [2.36, 3.93]). The paper separately evaluates storage/write amortization, store-ready and Write-inclusive timing boundaries, an approximate CacheBlend-style full-depth chunk-KV baseline, a continuous-prefix attribution oracle, and overlap/context-position interventions. The manuscript is unusually explicit that its large 64.9x number combines bounded selection with depth reuse, that decode reduces the matched end-to-end gain to roughly 1.07--1.09x, and that BM25 raw replay wins the equal-latency comparison.

### Claim/evidence map and minimum sufficient tests

- **C1: split depth is a useful measurable serving axis.** Minimum sufficient test: identical backbone, examples, retrieved tokens/order, adapter, masks, positions, and hardware, varying only replay start. Actual evidence: Table 2 / Table 34, PDF lines 431--447 and 1191--1219, gives the matched $j=0\rightarrow12$ quality/Read comparison. This is substantially sufficient for the narrow claim.
- **C2: the reusable residual object has favorable storage and amortization relative to full-depth KV.** Minimum sufficient test: correct byte accounting plus repeated-query timing including Write, fetch, Read, and decode across placements and generation lengths. Actual evidence: Eq. 1 (PDF lines 321--337), Table 4 (PDF lines 544--556), Table 35 (PDF lines 1221--1233), and the supplied serving artifact. This supports workload-specific crossover, not a production-wide efficiency claim.
- **C3: missing lower-layer document context is the dominant tested source of the matched interface gap, and overlap repairs it.** Minimum sufficient test: paired factorization holding retrieval, adapter, and upper Read fixed, followed by a deployable intervention on the same cohort. Actual evidence: Tables 6--7, PDF lines 491--515 and 559--579. This is sufficient only for the displayed 8k/16k synthetic multikey cohort.
- **C4: one residual can outperform independently prepared full-depth chunk KV on the same evidence path.** Minimum sufficient test: same model, evidence pack, scorers, comparable adaptation/training budget, and faithful baseline algorithm. Actual evidence: Table 3 / Appendix A.6 plus a baseline snapshot. Same model/evidence are controlled, but the comparison is not adaptation-matched and is explicitly not the complete CacheBlend stack, so it supports an implementation-level observation rather than a general method ranking.
- **C5: CoMem provides a useful deployment frontier.** Minimum sufficient test: quality and end-to-end latency under realistic selectors/storage, multiple workloads/models, concurrency/tails, and matched baselines. Actual evidence is broad but fragmented across separately bounded cohorts; the paper candidly limits the conclusion. This claim is promising but not yet main-conference decisive.

## Strengths

**S1. The central causal comparison is unusually clean.** Section 5.1 and Table 2 (PDF lines 405--447) hold the evidence pack, order, sink, mask, examples, and LoRA fixed while varying replay start. This directly isolates the incremental effect of prepaying 12 layers, and the paper reports both the 3.12-point quality loss and the actual 267.5 ms Read saving rather than presenting only a compound retrieval speedup.

**S2. Timing boundaries are separated rather than conflated.** Section 5.5 (PDF lines 520--564), Table 8 (PDF lines 934--942), and Appendix A.6 (PDF lines 1115--1219) distinguish selected-pack Read, store-ready prefill, Write-inclusive pipeline, persistent I/O, and Read-plus-decode. The explicit statement that the matched total ratio is only about 1.07--1.09x materially improves the trustworthiness of the systems claims.

**S3. The paper reports important negative and boundary results.** Table 5 (PDF lines 469--489) shows that equal-latency BM25 raw replay is 11.56 points better, and the BGE result is unresolved under hierarchical resampling. Table 4 also contains an infinite crossover cell and very large large-output crossover values. These results prevent an overly broad “reuse always wins” interpretation.

**S4. The mechanism analysis is claim-linked and experimentally disciplined.** The continuous-prefix oracle (Table 35, PDF lines 1221--1233) establishes that the upper 24 layers can recover full replay from compatible states. The context-position factorization and overlap sweep (Tables 6--7, PDF lines 491--515 and 559--579) then identify and repair a specific interface failure while preserving persistent bytes and online Read.

**S5. Scope, statistics, and cohort bookkeeping are careful.** The manuscript labels distinct RULER cohorts, avoids subtracting across them, gives paired and dependence-aware intervals, identifies the one-conversation LoCoMo limitation, and states where seed and effective batch are confounded (Appendix B, PDF lines 1287--1421; Limitations, PDF lines 582--656).

**S6. Presentation and compliance are strong.** The rendered manuscript is legible, anonymous, in ACL review style, has an exact unnumbered `Limitations` section, and contains no visible unresolved references, placeholders, or reviewer-directed/prompt-injection text. The eight-page numbered body ends before Limitations; references and appendices follow. All figures and tables inspected render without clipping or obvious corruption.

## Weaknesses

### W1 — The supplied artifact does not substantiate the paper-wide reproducibility claims

- **Location:** Appendix A.5, PDF lines 1070--1099; frozen source `sections/08_appendix.tex`, lines 148--149 and 252--276.
- **Exact quote (13 words):** “archive includes the adapter, source, documentation, pinned requirements, prediction hashes or permissible shards”
- **Problem:** The reviewed source package contains manuscript sources plus two narrow artifact snapshots (serving crossover and CacheBlend-style aggregation). It does not contain the claimed adapter, pinned environment/requirements, flagship evaluation scripts/configs, judge prompt/parsed outputs, score-only equal-latency exports, benchmark prediction hashes/shards, or most raw timing records. The CacheBlend aggregation script also cannot run standalone from the package because it imports absent project modules. Thus the headline 99.19/96.07 quality comparison, confidence interval, overlap diagnosis, equal-latency statistics, LoCoMo judge results, training, and Hy3 claims cannot be independently regenerated or even recomputed from the supplied artifact.
- **Affected claim/norm and why it matters:** Reproducibility/artifact sufficiency. The paper relies on many saved-result and exact-hash assertions; without the promised files, reviewers can verify only the 24-cell serving table and static CacheBlend aggregate, not the main empirical claims.
- **Sufficient remedy:** Supply an anonymous self-contained artifact matching the manuscript inventory: adapter/config/hash manifest; pinned lockfile or container; executable inference/evaluation and statistical scripts; permissible per-example scores/prediction hashes; judge template and parsed outputs; raw timing records; and baseline dependencies. Provide one command that recomputes every table from released inputs without private project imports.
- **Severity:** **Major**.

### W2 — The strongest cross-method comparison is not adaptation-matched and uses a partial baseline implementation

- **Location:** Section 5.2 and Table 3, PDF lines 432--468; Limitations, PDF lines 588--605.
- **Exact quote (13 words):** “the baseline is training-free, whereas CoMem uses its distilled adapter.”
- **Problem:** CoMem receives 58.2M trained LoRA parameters and the CacheBlend-style arm receives none; the latter also omits the native scheduler/cache manager and is evaluated only through 18% selective recomputation. Because the paper’s own adapter ablations show very large gains, the observed 97.05 vs. 74.70 RULER gap cannot be attributed to the residual object or upper-layer recomputation alone. Storage is cleanly comparable, but quality is not an adaptation-matched method comparison.
- **Affected claim/norm and why it matters:** C4 and the abstract/conclusion prominence of the “18x more yet 74.70 versus 97.05” result. Readers may infer a general superiority over CacheBlend-like full-depth KV, while the experiment identifies only a particular training-free implementation below its full-recompute endpoint.
- **Sufficient remedy:** Add (i) a quality curve through $r=1$ with matched timing, (ii) an adaptation-matched full-depth chunk-KV reader or a CoMem-without-LoRA comparison at the same split/evidence, and (iii) preferably the released native CacheBlend implementation on a supported backbone. Otherwise demote the quality contrast to a diagnostic and keep only the exact storage comparison as a central claim.
- **Severity:** **Major**.

### W3 — The clean depth result is one model, one retained split, and one primary synthetic benchmark macro

- **Location:** Limitations, PDF lines 582--605; source `sections/07_limitations.tex`, lines 3--17.
- **Exact quote (12 words):** “The controlled depth evidence is the $j{=}0\!\rightarrow\!12$ endpoint;”
- **Problem:** The only clean causal depth estimate is Qwen3-8B at $j=12$ on RULER-B. Other depths use separately trained adapters with changing parameter spans, natural-task transfer is highly uneven (e.g., LongEval 97.2 to 69.0), and Hy3 provides portability/stress evidence rather than a matched $j=0$ quality-latency frontier. Consequently the paper does not yet establish that “depth” behaves as a robust tunable axis across architectures/tasks, only that one selected split offers one useful trade-off.
- **Affected claim/norm and why it matters:** C1/C5 and novelty significance. A new systems axis is more convincing when a controlled curve, not a single endpoint, generalizes across at least a second model or natural workload.
- **Sufficient remedy:** On Qwen3-8B, train and evaluate matched-budget adapters at several splits and compare each with identical-pack $j=0$, with latency, storage, and natural-task quality. Replicate at least two matched endpoints on a second architecture. If unavailable, narrow the headline from a general tunable axis to a Qwen3-8B proof of concept.
- **Severity:** **Major**.

### W4 — The proposed repair is not evaluated on the claimed deployment frontier

- **Location:** Limitations, PDF lines 628--633; source `sections/07_limitations.tex`, lines 34--38.
- **Exact quote (15 words):** “Overlap-Write is validated only on paired synthetic multikey instances, without a natural-task or Write-inclusive repaired frontier.”
- **Problem:** The central diagnosis/repair is compelling, but it is demonstrated only on 200 synthetic multikey examples. There is no evidence that overlap improves LongEval, BABILong, LongBench, or LoCoMo, nor a measured break-even/latency curve incorporating overlap’s extra Write and invalidation cost. Thus the paper cannot yet show that the repaired interface improves the real deployment trade-off rather than a narrow mechanism probe.
- **Affected claim/norm and why it matters:** C3/C5 and the abstract’s “dominant tested multikey error”/repair narrative. The narrow wording is accurate, but the intervention’s practical value remains uncertain.
- **Sufficient remedy:** Evaluate $w\in\{0,32,128\}$ on at least LongEval and one compositional natural benchmark, report paired uncertainty, measured Write overhead, edit invalidation, and revised $Q^\star$ while holding the online pack fixed.
- **Severity:** **Major**.

### W5 — The flagship adapter’s run-to-run uncertainty remains weakly characterized

- **Location:** Limitations, PDF lines 582--589; Appendix A.7, PDF lines 1235--1243.
- **Exact quote (17 words):** “The flagship adapter is one batch-8 run; two matched-data runs use effective batch 3,”
- **Problem:** The headline model is a single training run. The two additional runs change both seed and effective batch and are evaluated on reduced supports, not on the exact RULER-B, LongEval, LoCoMo, overlap, or timing headlines. Cell bootstrap intervals quantify evaluation-example uncertainty but not training variance.
- **Affected claim/norm and why it matters:** Statistical reliability of the 3.12-point gap and cross-benchmark quality claims. With adaptation producing very large gains, optimization variance is a material source of uncertainty.
- **Sufficient remedy:** Run at least three same-batch, same-data, independently seeded adapters and report run-level mean/SD or intervals for the exact headline RULER-B macro plus at least LongEval and LoCoMo. Separate training variance from test-example resampling.
- **Severity:** **Minor** (the central matched comparison remains valid for the released checkpoint, but general reliability is uncertain).

## Questions That Could Change the Score

1. Can the authors provide the complete anonymous artifact described in Appendix A.5 and demonstrate a one-command recomputation of Table 2, Tables 5--7, and Appendix B statistics? A genuinely self-contained artifact would raise reproducibility substantially.
2. What are the CacheBlend-style quality and latency at $r=1$, and what happens under an adaptation-matched comparison? If full recomputation or a matched reader closes most of the quality gap, the current cross-method claim should be revised; if not, that would materially strengthen the paper.
3. Do controlled identical-pack $j=0$ comparisons at $j=6,9,18$ yield a coherent quality-latency curve when training budget and evaluation support are matched?
4. Does 32-token Overlap-Write improve LongEval/BABILong/LoCoMo, and what is the measured revised break-even after its extra Write cost?
5. What is the training-run variance of the exact 15-cell RULER-B and LongEval headlines under same-batch independent seeds?

## Non-scoring Suggestions / Typos

- Table 19 rounds the matched raw-text RULER-B value to 99.20, while Tables 2/34 use 99.19. Standardize precision to avoid apparent cohort ambiguity.
- The paper alternates among “CacheBlend-style,” “minimal faithful,” and “same-backbone full-depth chunk-KV.” Use one consistently qualified name wherever 74.70 vs. 97.05 is repeated.
- Consider moving the Read-plus-decode 1.07--1.09x number into the abstract; it is essential context for the 1.403x Read result.
- Add an explicit artifact-to-claim matrix listing which released file regenerates each headline table.
- Clarify whether “Write once per document chunk” includes BOS/sink and query-write accounting in every timing cohort; the appendix is precise, but the high-level terminology remains easy to misread.

## Detailed Scores

### Soundness: 3.5/5

The narrow matched depth claim is well controlled, statistics are mostly careful, formulas and byte accounting check out, and negative findings are disclosed. The score is below 4 because the cross-method comparison is not adaptation-matched, the repair evidence is narrow, and training-run uncertainty is incomplete.

### Excitement: 3.5/5

Treating reusable depth as an explicit systems coordinate is a useful and potentially influential framing, and the attribution/repair analysis is stronger than a simple benchmark paper. Excitement is tempered because the closest work already spans reusable intermediate representations, layer-wise activation caches, hidden-state/KV restoration, and modular KV caches; the novelty is the particular conjunction and measurement protocol rather than a wholly new primitive.

### Overall: 3.0/5

This is a solid **Findings-level** paper with a credible core result and unusually transparent systems accounting. I do not currently place it at ACL main-conference level because the general “tunable depth axis” case rests on one clean endpoint, the marquee CacheBlend-style comparison is confounded by adaptation and implementation scope, and the supplied artifact does not support the paper-wide reproducibility claims. A complete artifact plus adaptation-matched and multi-split evidence could move the score upward.

### Confidence: 4.0/5

I inspected the entire 28-page PDF twice, all appendices, every rendered figure/table, the frozen TeX, `main.bbl`, and the two included artifact snapshots. I mechanically checked the quoted weaknesses against the source and verified the provided SHA-256 manifests and serving-table script. Remaining uncertainty is mainly about external-system implementation details and contemporaneous 2026 work.

### Reproducibility: 2.5/5

The manuscript gives excellent protocol detail, seeds, hashes, sample counts, timing boundaries, and equations. The included serving snapshot is internally consistent: all manifests pass and the verifier reproduces the complete 24-cell crossover table. The CacheBlend snapshot preserves hashes, aggregates, and a correctness log. However, the reviewed package omits most artifacts it explicitly claims to include, and one supplied aggregation script depends on absent modules. Most headline quality/statistical results therefore cannot be independently recomputed from the frozen artifact.

## Limitations, Ethics, and Desk-Reject Risks

- **Limitations:** An exact `Limitations` section is present and unusually comprehensive. It covers single-model emphasis, baseline mismatch, stable-corpus assumptions, storage/update costs, selector dependence, timing boundaries, overlap scope, mutable judging, contamination, and unsupported modalities/languages.
- **Ethics:** The Ethical Considerations section discusses hallucination/bias/misuse, privacy/inversion/membership risks of residual stores, access control, deletion, energy, licenses, and the absence of new human-subject collection. This is adequate. The principal additional artifact concern is that unverifiable release claims should be corrected before publication.
- **Desk-reject/style risks:** I found no anonymity breach, author identity, hidden reviewer instruction, white/tiny manipulation, unresolved citation/reference, or TODO/placeholder in the manuscript. The paper uses official ACL review style, A4, anonymous authorship, and an eight-page numbered main body before the unnumbered Limitations/Ethics sections. The rendered PDF has 28 pages including references and appendices. I see no obvious formatting desk-reject trigger from the reviewed files.

## Citation Audit

### Completeness and metadata status

`main.bbl` contains **46 entries**, and all 46 are actually cited. I checked every entry against its DOI/arXiv/ACL Anthology/proceedings/model record where an identifier was available; failed DOI access was treated as **Unverifiable**, not “Not found.”

- **Verified (42):** `cachecraft`, `longbench`, `llmcache`, `pyramidkv`, `kvpacket`, `cartridgesbase`, `hcache`, `promptcache`, `llama3`, `cartridges`, `distillation`, `ruler`, `lora`, `epic`, `ragcache`, `babilong`, `rag`, `longchat`, `snapkv`, `ilre`, `readonce`, `minicache`, `turborag`, `locomo`, `xccache`, `kvdirect`, `pg19`, `bm25`, `embeddingrecycling`, `gemfilter`, `reform`, `lloco`, `hunyuan`, `fusionrag`, `mepic`, `longmem`, `memoryllm`, `infllm`, `streamingllm`, `qwen3`, `ape`, `cacheblend`.
- **Metadata error (1):** `h2o` is incomplete (“Zhenyu Zhang et al.” and “In NeurIPS”) relative to the full proceedings metadata (NeurIPS 36, pages 34661--34710, DOI 10.52202/075280-1506).
- **Not found (0).**
- **Unverifiable (3):** `sempic` (very recent arXiv record; existence verified but full metadata/claims were not independently inspectable); `xu2024retrievallong` (conference paper exists, but the bbl lacks a direct identifier and full metadata was not independently checked); `blockattention` (the title/venue are plausible, but the bbl lacks a direct identifier and I did not recover an authoritative metadata record).

### Load-bearing citation--claim matches

1. **ReadOnce / Embedding Recycling:** The cited works do cache reusable intermediate text/encoder representations; the paper fairly distinguishes their compression/downstream-adaptation focus from a decoder split-depth serving axis. **Match.**
2. **LLMCache:** The cited work is layer-wise semantic activation caching, so the description of per-layer banks/arbitrary-layer reuse is directionally accurate. It is extremely recent relative to this submission, so novelty claims should remain narrow. **Match with contemporaneous-work caution.**
3. **HCache / KV-Direct:** Both concern hidden/residual state as an alternative route to reconstructing/restoring KV state. The paper’s distinction—restoring standard layer-wise state versus directly executing one native suffix from one split—is meaningful, although KV-Direct is especially close. **Match.**
4. **CacheBlend / TurboRAG / EPIC / APE:** These works precompute or modularize chunk/context KV and address assembly dependencies/positions. The family-level characterization is broadly correct. **Match.**
5. **Cartridges / KV Packet / SemPIC:** These learn or compile reusable context-independent/modular KV objects. The paper accurately presents them as full-depth learned objects rather than one raw residual split. **Match.**
6. **GemFilter / ILRe / REFORM:** These use intermediate layers for selection/compression/gathering but do not primarily persist selected residuals across repeated queries. **Match.**
7. **RULER/BABILong/LongBench/LoCoMo:** The cited papers correspond to the benchmarks and official task families used. **Match.**
8. **Qwen3/LoRA/distillation:** The model architecture and adaptation/distillation citations are appropriate. **Match.**

## Novelty Search Summary

I performed five targeted searches: (i) persistent intermediate/residual caches for repeated transformer queries; (ii) layer-wise activation reuse; (iii) hidden-state/residual reconstruction of KV; (iv) position-independent/full-depth modular KV caches; and (v) early-layer token selection with later-layer computation. The closest works are:

1. **LLMCache: Layer-wise Caching Strategies for Accelerated Reuse in Transformer Inference** (2025) — arbitrary-layer activation banks with semantic matching. Closest on making reuse layer-dependent, but it does not study one persistent document split with identical-evidence $j=0$ measurement.
2. **The Residual Stream Is All You Need / KV-Direct** (2026) — stores residual checkpoints and reconstructs layer-wise KV. Closest state representation; differs in restoring KV versus directly continuing one decoder suffix.
3. **HCache** (EuroSys 2025) — stores hidden states across layers for fast state restoration. Close on hidden-state persistence, but optimizes restoration of standard cache state rather than a quality--depth axis.
4. **ReadOnce Transformers** (ACL 2021) and **Embedding Recycling** (EACL Findings 2023) — reusable intermediate document/encoder representations. Conceptually prior to persistent intermediate text states, but not the same autoregressive decoder-serving protocol.
5. **CacheBlend / EPIC / APE / KV Packet / Cartridges / SemPIC** (2025--2026) — the closest modular reusable-context systems. They primarily store/reconstruct/learn layer-wise KV and optimize linking/position independence rather than selecting a single residual depth.

The defensible novelty is therefore **narrow but real**: the conjunction of one persistent per-token residual at a chosen split, direct native-suffix execution, an identical-pack $j=0$ endpoint, and joint measurement of quality, Read latency, storage, Write, and amortization. I did not find a prior paper that cleanly reports that exact controlled axis. However, several closest papers are within roughly three months of the August 4, 2026 manuscript date (KV-Direct, Cartridges at Scale, SemPIC), so priority over those contemporaneous works should not be claimed; they should be treated as concurrent context.

## Review-process Self-check

- Read the frozen PDF twice, including both appendices, and inspected all 28 rendered pages.
- Inspected every figure and table for legibility and consistency.
- Searched the frozen source/PDF for prompt injection, hidden/white/tiny reviewer manipulation, TODOs, placeholders, unresolved references, and anonymity leaks; none found.
- Checked abstract numbers against Tables 2--7 and Appendix timing tables; the reported arithmetic is consistent (931.9/664.4 = 1.403; 99.19-96.07 = 3.12; 71.37/1.10 = 64.9; 6.035/2.202 = 2.74; 147,456/8,192 = 18).
- Checked Eq. 1 and Qwen3 byte arithmetic, including boundary behavior at $j=0$ and the non-reusable continuous-prefix oracle.
- Verified that every `main.bbl` entry is cited and completed the metadata status audit above.
- Ran the included SHA-256 checks and serving verifier; all included manifests pass and Table 4’s 24 cells reproduce.
- Mechanically grepped every weakness quote against the frozen source. Each “missing artifact/evidence” assertion was checked against the complete allowed source tree, including appendices and the two supplied artifact directories.
- I did not read any other review, history version, live status, TODO, or Paper B file, and I made no paper edits.
