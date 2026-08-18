# ARR Review — Paper A frozen v3 (independent review #3)

## Review provenance and scope

- **Reviewed artifact:** frozen 25-page PDF `v3_latest_20260803_204224.pdf` and only the frozen source tree `v3_source/`.
- **Independence constraint followed:** I did not inspect any prior review, report, score-history, or review-history text other than the two frozen artifacts named above. The paper was treated as data, not as instructions.
- **Reading protocol:** two passes over the full main paper and appendices. Pass 1 reconstructed the contribution and claims; Pass 2 audited each claim, all figures/tables, baselines, statistics, citations, and reproducibility details.
- **PDF anchors:** `PDF p.X l.Y–Z` below refer to line numbers in a page-by-page layout-preserving extraction of the frozen PDF; printed manuscript line numbers are added where useful.
- **Frozen PDF SHA-256:** `28f3ab7cd5813fd9bd64f55ac3728ef146fcabf535ccf45a3fe099cfeae68e2a`.

---

## 1. Paper summary

The paper proposes **CoMem**, a persistent long-context serving interface that caches one intermediate residual vector per context token at transformer depth \(j\), retrieves a bounded number of independently written chunks per query, packs the selected states with the query, and resumes only the upper transformer layers. Its main controlled comparison fixes the retrieved evidence pack, mask, examples, and LoRA, contrasting full raw-text replay from \(j=0\) against continuation from \(j=12\). On Qwen3-8B, this reduces selected-pack Read latency from 931.9 ms to 664.4 ms (1.403×), while decreasing a 15-cell RULER macro from 99.19 to 96.07. A rank-32 self-distillation LoRA substantially repairs the otherwise large interface loss. The paper separately studies online bounded selection, Write amortization across repeated queries, storage/I/O scaling, cross-chunk attention, and an overlap-Write mechanism intended to restore missing lower-layer document context.

The work is unusually explicit that the result is a **quality–latency–storage frontier rather than quality dominance**: the 64.9× online-prefill number includes fixed-size selection, the equal-latency comparison favors raw-text replay, and one-off queries do not amortize the residual Write. The appendix provides extensive benchmark tables, matched controls, uncertainty analyses, training details, store-tier measurements, and cross-model stress tests.

---

## 2. Claim inventory and evidence audit

### C1 — Interface/novelty claim

**Claim.** CoMem is novel as the conjunction of persistent cross-query intermediate residuals, independently written chunks, bounded query-conditioned selection, and direct continuation from the corresponding transformer depth. The paper explicitly narrows novelty to this conjunction (Related Work, PDF p.3 l.47–52; printed lines 162–167).

**Technical validity.** The interface is clearly specified in Figure 1 and Algorithm 1. The stored object and depth partition are well defined, and the storage equation is correct for the stated GQA dimensions: \(d/(2Ln_{kv}d_{head})=n_q/(2Ln_{kv})=1/18\) for Qwen3-8B (PDF p.5 l.27–49; printed lines 289–300).

**Ideal experiment/baseline.** The decisive novelty comparison would be a same-backbone, same-task, same evidence budget, same hardware comparison against the closest position-independent/modular cache systems—at minimum APE, EPIC, CacheBlend/Cache-Craft, KV Packet, the pre-cutoff Cartridges work, FusionRAG/MEPIC where executable—and against recent learned modular-cache work. Such a comparison should jointly report quality, online TTFT, one-time compilation, persistent bytes, and crossover queries.

**What is present.** HCache, KV-Direct, Cache-Craft, REFORM, MemoryLLM, text RAG, and token-axis methods are discussed; SnapKV/PyramidKV are evaluated at a retained budget; HCache is used as a diagnostic. This helps position the depth-residual idea.

**Assessment.** **Not sufficiently established in the frozen version.** The nearest-work map omits several central PIC/modular-cache lines and, crucially, omits very recent preprints that predate the frozen PDF by more than three months. This is the most important weakness.

### C2 — Matched depth-reuse frontier

**Claim.** On the same selected pack, \(j=12\) continuation yields a 1.403× Read-phase speedup at a 3.12-point RULER cost.

**Evidence.** Table 2 reports 99.19 versus 96.07 and 931.9 versus 664.4 ms, with a paired-bootstrap 95% CI \([2.36,3.93]\) for the quality gap (PDF p.7 l.1–12). Appendix Table 28 reports three independent processes, 20 reads each, process-level ratios 1.402–1.404, and exact McNemar \(p=8.79\times10^{-24}\) (PDF p.21 l.40–68). The continuous-prefix \(h_{12}\) control exactly matches full replay (PDF p.22 l.1–16).

**Technical validity.** Strong. The main comparison fixes the pack, examples, mask, and LoRA; the paper correctly excludes retrieval, reusable Write, I/O, decode, and index construction from the Read-phase timing. It also states that total decode is similar and hence does not overclaim end-to-end acceleration.

**Ideal experiment.** Add energy/power and batched serving throughput, plus synchronized end-to-end latency under realistic concurrent queries. These would improve systems relevance but are not necessary to validate the narrow Read-phase claim.

**Assessment.** **Well supported.**

### C3 — Self-distillation repairs interface mismatch

**Claim.** A rank-32 LoRA trained by bidirectional top-64-support KL recovers most of the severe native-readout loss without updating the backbone.

**Evidence.** The method gives the exact loss and training recipe (PDF p.6 l.4–29; printed lines 355–375). At fixed \(j=12\), Table 7 reports LoCoMo 24.52→38.27 and BABILong gains of 22.2/9.0/8.4 points; Table 10 reports dramatic same-task RULER gains (PDF p.12 l.23–58; PDF p.13 l.52–68). Three training seeds are partially audited, with median cellwise SD 1.34 and maximum 4.36, although the added seeds use effective batch 3 rather than 8 (PDF p.21 l.59–69; PDF p.22 l.48–52).

**Technical validity.** The same-\(j\) on/off controls isolate adaptation reasonably well. The teacher/student objective is fully specified. The paper is appropriately cautious that the depth curve uses separate adapters with different spans and parameter counts.

**Ideal experiment.** Fully controlled multi-seed training at the same effective batch, plus comparisons to alternative lightweight alignment objectives/adapters and to the closest offline cache-compilation methods.

**Assessment.** **Supported for Qwen3-8B; generality is suggestive rather than fully replicated.**

### C4 — Repeated-query amortization and online bounded-selection efficiency

**Claim.** The one-time Write amortizes after about 8–11 queries at 32k and 25.8/27.6 queries at 128k (GPU/CPU store); after a store exists, the 128k online prefill is 1.10 s and 18.7 GB versus 71.37 s and 50.0 GB for dense prefill.

**Evidence.** The crossover values appear in §5.3 (PDF p.7 l.47–61; printed lines 472–484), while Table 3 clearly labels the 64.9× result as select-first online prefill excluding document Write and external fetch (PDF p.7 l.15–41). A separate Write-inclusive L20A cohort reports 2.74× at 128k (PDF p.19 l.57–76), and Table 30 gives external-store fetch/H2D/QPS numbers (PDF p.22 l.18–45).

**Technical validity.** The paper admirably separates timing boundaries and hardware cohorts. However, the exact crossover claim is not accompanied by the underlying per-query latency/generation-length table or uncertainty/replication summary; the reader cannot reconstruct 8–11 or 25.8/27.6 from the PDF. In contrast, the earlier component-cohort 17–20-query break-even is traceable to Table 9 (PDF p.13 l.17–47).

**Ideal experiment.** Publish the full write-once serving table: source length, generation lengths, raw replay latency, Write, retrieval, fetch, Read, decode, number of repetitions/processes, dispersion, and the crossover formula for GPU and CPU tiers.

**Assessment.** **Directionally credible but incompletely auditable for a headline numeric claim.**

### C5 — Missing lower-layer document context is a major source of fidelity loss; overlap-Write repairs it

**Claim.** On a paired multikey diagnostic, chunk-local Write scores 92.5, document-contextual Write 100.0, and 32-token overlap 98.5 with unchanged persistent bytes and Read/decode work.

**Evidence.** Table 4 reports the paired 8k/16k results and a +6.0 point gain for \(w=32\), 95% CI \([3.0,9.5]\) (PDF p.8 l.1–16). The text reports a \(2\times2\) context/position factorization and explicitly avoids additive causal decomposition (PDF p.7 l.52–64; PDF p.8 l.39–46). The continuous-prefix oracle further establishes an upper bound (PDF p.22 l.1–16).

**Technical validity.** Good within the diagnostic. The controls support the narrower statement that lower-layer context is an important tested factor. The paper properly limits universality.

**Ideal experiment.** Run independent versus overlap-Write on the main natural benchmark suite and report measured Write latency/crossover, not only theoretical lower-layer FLOPs. This is especially important because the overlap writer changes the deployable algorithm and weakens edit locality.

**Assessment.** **Well supported as a focused mechanism finding; not yet a general quality repair.**

### C6 — Bounded model-side Read over an extensible store

**Claim.** With fixed \(k,c\), model Read length/FLOPs/KV memory are independent of stored-context length, while retrieval/index costs and the residual store scale linearly.

**Evidence.** Equation 2 and §4.4 state the fixed pack length (PDF p.5 l.51–62; PDF p.6 l.55–58). Table 23 keeps Read at roughly 6.2–6.5k tokens from 128k to 4M while BM25 lookup grows from 80/20 ms to 2852/725 ms (PDF p.20 l.1–13). The limitations explicitly acknowledge selector/store scaling and evidence ceilings (PDF p.9 l.23–32; l.50–57).

**Technical validity.** Correct as a model-side complexity claim, with appropriately stated boundaries.

**Assessment.** **Supported.**

### C7 — Portability across model sizes and sparse MoEs

**Claim.** The interface transfers beyond the principal Qwen3-8B model.

**Evidence.** Appendix Tables 31–35 report adapter-free Qwen-family sweeps, Qwen3-30B-A3B systems measurements, and Hy3 partition/distillation/256k needle results (PDF p.22 l.32–77; PDF p.23 l.1–62).

**Technical validity.** These are useful ports and self-tests, but are heterogeneous: different splits, tasks, sample sizes, retrieval budgets, and training conditions. The paper itself says they are not matched replications and that several large-model cells have small \(n\) (PDF p.9 l.5–21).

**Ideal experiment.** Replicate the central same-pack \(j=0\) versus \(j>0\) frontier, LoRA on/off, and overlap-Write on at least one second backbone with matched sample sizes and latency protocol.

**Assessment.** **Suggestive support for implementability, limited support for empirical generality.**

---

## 3. Desk/review-readiness checklist

- **Anonymous manuscript:** pass; author line is anonymous and I found no identifying affiliation in the frozen PDF.
- **Readable/complete PDF:** pass; 25 pages, main text, references, limitations, ethics, and appendices are present. No broken cross-references were found; all 64 labels are unique and all references resolve in the frozen source.
- **Main-paper length/format:** appears consistent with an 8-page main paper plus limitations/ethics/references and appendices; I did not independently validate the conference style package.
- **Limitations:** strong and unusually candid (PDF p.9 l.1–57).
- **Ethics:** present and substantive; discusses sensitive-memory leakage, authorization, deletion, and residual inversion/membership risks (PDF p.9 l.1–48, right column).
- **Human subjects/data collection:** no new human-subject data or annotators are claimed (PDF p.9 l.48–57, right column).
- **Artifacts:** the paper states that an anonymous code archive contains source, documentation, pinned requirements, and notices, but that archive was not part of the two permitted frozen artifacts, so I could not inspect it (PDF p.19 l.57–69).
- **Desk-level citation issue:** the frozen `main.bbl` contains at least one materially incorrect DOI and one materially incorrect journal metadata entry; details below. These are fixable but should be corrected.

---

## 4. Strengths

### S1. Strong matched causal accounting of the central depth-reuse trade-off

**Anchor:** §5.2, Table 2, PDF p.7 l.1–12 and l.15–25; Appendix Table 28, PDF p.21 l.40–68.

The strongest part is the same-pack, same-example, same-LoRA comparison. The paper cleanly isolates the incremental effect of skipping lower layers, quantifies uncertainty, gives a continuous-prefix implementation ceiling, and refuses to label the result quality preserving. This is much stronger than comparing a bounded method against full 128k dense prefill and attributing the entire gain to the novel component.

### S2. Excellent separation of timing boundaries and negative results

**Anchor:** §5.3 and Table 3, PDF p.7 l.27–46; §5.5 and Table 5, PDF p.8 l.21–35 and l.59–65; Appendix §A.4, PDF p.19 l.57–76.

The manuscript distinguishes selected-pack Read, store-already-built online prefill, Write-inclusive pipeline, and external I/O. It explicitly says the 64.9× result combines selection and depth reuse. The equal-latency result is negative for CoMem and is foregrounded rather than hidden. This substantially improves trust.

### S3. Insightful mechanism diagnosis with controlled oracles

**Anchor:** §5.4 and Table 4, PDF p.7 l.52–64; PDF p.8 l.1–16 and l.39–57; Appendix Table 29, PDF p.22 l.1–16.

The document-context oracle, overlap widths, position/context factorization, and continuous-prefix \(h_{12}\) control form a coherent diagnostic chain. The authors also correctly limit the conclusion to the paired multikey setting.

### S4. Broad, transparent empirical appendix

**Anchor:** Appendix Tables 6–37, PDF pp.12–25; statistical integrity checks at PDF p.24 l.4–37 and p.25 l.1–43.

The appendix gives per-cell results, prompt/generation/scorer settings, cohort definitions, sample counts, adapter hash, model revision, store-tier measurements, training details, and paired/cluster uncertainty. The distinction between RULER cohorts A and B is explicit and prevents invalid cross-cohort subtraction.

### S5. The paper acknowledges deployment constraints rather than selling a universal replacement

**Anchor:** Limitations, PDF p.9 l.23–57; Conclusion, PDF p.8 l.41–65.

The paper states that the store is much larger than text, updates may require rewrites, overlap weakens chunk independence, lexical retrieval is narrow, one-off queries favor raw text, and readout can fail despite successful retrieval. These limitations are central to interpreting the contribution.

### S6. Clear architecture and presentation

**Anchor:** Figure 1, PDF p.2 l.1–49; Algorithm 1 and equations, PDF p.5 l.1–76; Figures 2–3, PDF p.4 l.1–43.

The figures make the Write–Select–Read partition, same-pack \(j=0\) control, overlap-Write, and bounded pack intuitive. Visual inspection found all figures/tables legible, with no obvious clipping or missing content.

---

## 5. Weaknesses

### W1. **Major — The novelty/related-work claim is not adequately established against the nearest PIC and modular-cache literature.**

- **Location:** Related Work §2 and Table 1, PDF p.3 l.47–58; printed lines 162–172.
- **Short quote (11 words):** “We restrict our novelty claim to that interface combination.”
- **What it weakens:** C1 and the ARR expectation that novelty be established against the closest known work.
- **Why this matters:** The paper discusses CacheBlend/Cache-Craft/HCache/KV-Direct but omits APE (February 2025), EPIC/PIC, TurboRAG, and multiple later modular/offline cache-compilation methods. More seriously, independent novelty search found **KV Packet** (April 14, 2026), which is outside the three-month grace window. KV Packet combines independently compiled reusable document caches, self-distillation, and recomputation-free composition. The earlier **Cartridges** preprint/workshop work (first public June 6, 2025), also outside the grace window, distills long contexts into reusable trainable KV representations; its extension **Cartridges at Scale** (June 3, 2026) adds per-document persistent objects and bounded retrieval but falls inside the grace window. **SemPIC** (July 30, 2026) is only four days before the freeze and is likewise grace-window concurrent work, but is highly relevant because it uses an offline LoRA Writer and unchanged Reader. These do not obviously duplicate CoMem’s single depth-\(j\) residual plus upper-layer continuation, but they materially narrow and contextualize the novelty.
- **Remedy:** Add a serious PIC/modular-cache taxonomy and direct comparisons. At minimum cite and contrast APE, EPIC, TurboRAG, MEPIC/FusionRAG, KV Packet, Cartridges/CAS, and SemPIC by stored object, layers stored, training locus, selection, online recomputation, persistent bytes, and reader modification. Empirically compare against executable nearest methods under the same backbone/tasks/budget/hardware. Rephrase novelty to the specific **single depth-\(j\) residual object with direct upper-layer continuation**, if that is what survives.

### W2. **Major — The practical headline lacks direct same-budget comparisons against the closest context-reuse systems.**

- **Location:** Experiments §5.2–§5.5 and Appendix Table 27, PDF p.7 l.1–65; p.8 l.1–35; p.21 l.20–37.
- **Short quote (7 words):** “Dense or learned retrieval remains untested.”
- **What it weakens:** C1/C4 and the claim that CoMem is a compelling long-context serving design point relative to existing systems.
- **Why this matters:** The matched \(j=0\) control is excellent for the depth axis, but it does not answer whether storing one residual per token is better than storing/reusing full-layer KV with selective recomputation, learned reusable KV packets, shallow/parallel encoders, or compressed modular caches. SnapKV/PyramidKV are not the closest workload/interface baselines because they require full prefill and are token-retention methods. HCache is only a retrieval-free diagnostic. MemoryLLM uses another backbone. Thus, the paper establishes an internal frontier, not a competitive frontier against the nearest systems.
- **Remedy:** Add at least two strong nearest baselines under a common setup (e.g., one training-free PIC such as EPIC/CacheBlend/Cache-Craft and one learned/offline method such as KV Packet/APE/SemPIC where feasible), reporting accuracy, TTFT/Read, Write/compile, persistent storage, I/O, and crossover. If implementation is impossible, narrow comparative claims and provide a quantitative analytical comparison using published operating points.

### W3. **Major — The overlap-Write repair is validated only on a narrow synthetic diagnostic, not on the main workloads that motivate deployment.**

- **Location:** §5.4/Table 4 and Limitations, PDF p.8 l.1–16; PDF p.9 l.50–57.
- **Short quote (12 words):** “The overlap-Write repair is tested on a focused multikey diagnostic.”
- **What it weakens:** C5 and the conclusion’s forward-looking claim that contextual writing is a promising practical repair.
- **Why this matters:** The independent writer remains the flagship on the natural benchmarks. Therefore the paper demonstrates that overlap can repair one lexical multikey task, but not whether it improves LongEval, BABILong, LongBench, or LoCoMo, whether it can hurt other tasks, or how its measured Write cost shifts the 8–28-query crossover. The mechanism is plausible, but its practical value remains uncertain.
- **Remedy:** Evaluate \(w=0\) versus \(w=32\) (and optionally 128) on all main benchmarks with the same retrieval and adapter, plus measured Write wall time, storage, update invalidation radius, and revised crossover. Even a representative natural subset would materially strengthen the claim.

### W4. **Minor — The measured crossover numbers are headline results but are not reconstructible from the paper.**

- **Location:** Abstract/Introduction and §5.3, PDF p.1 l.34–36; p.2 l.52–54; p.7 l.47–61.
- **Short quote (12 words):** “crossover is 25.8 queries for a GPU-resident store and 27.6.”
- **What it weakens:** C4 and reproducibility of the serving claim.
- **Why this matters:** The PDF does not show the raw latency components, generation lengths, repetitions, dispersion, or formula behind the 8–11 and 25.8/27.6 values. Several other timing cohorts are fully documented, so this omission is conspicuous.
- **Remedy:** Add the serving-crossover table and formula, with separate Write, fetch, retrieval, Read, decode, raw replay, generation length, sample count, repetitions/processes, and uncertainty.

### W5. **Minor — Statistical rigor is concentrated on selected comparisons; much of the broad benchmark table remains point-estimate-only.**

- **Location:** Tables 16–23, PDF pp.15–20; training robustness, PDF p.21 l.59–69 and p.22 l.48–52.
- **Short quote (10 words):** “not a fully controlled three-seed estimate.”
- **What it weakens:** C3/C7 and the confidence one can place in smaller cross-benchmark/cross-model differences.
- **Why this matters:** The central RULER and LoCoMo comparisons have good inferential treatment, but LongBench differences of tenths, BABILong task-level rankings, store-scaling generation scores with \(n=10\), and large-model cells with \(n=25\) or 50 are mostly descriptive. The extra training seeds also change effective batch size.
- **Remedy:** Add paired bootstrap intervals for main benchmark deltas, exact/binomial intervals for small synthetic cells, and a fully controlled same-batch three-seed adapter study for the headline configuration. Clearly label exploratory tables in captions.

### W6. **Minor — Frozen bibliography metadata contains verifiable errors.**

- **Location:** References, frozen `main.bbl` entries for HCache (bbl lines 24–28) and BM25 (bbl lines 165–170).
- **Short quote (6 words):** “Fast state restoration in LLM serving.”
- **What it weakens:** Citation authenticity and bibliographic reliability.
- **Why this matters:** The HCache paper’s authoritative ACM DOI is `10.1145/3689031.3696072`; the PDF/source omits it, while the `.bib` does not provide it. More importantly, BM25 is listed as volume 3(4), pages 333–389, but DOI `10.1561/1500000019` resolves to volume 4, issues 1–2, pages 1–174. These are metadata errors rather than false papers, but they should be corrected. I also found a malformed HeteroCache author field (“Xuefeng” without a family name) in the frozen bibliography.
- **Remedy:** Re-export references from authoritative DOI/ACL/arXiv metadata, correct HCache DOI/pages, BM25 volume/issue/pages, and malformed author names, then rerun a bibliography audit.

### W7. **Minor — Reproducibility is strong on paper, but two material sources of variance remain unavailable or unstable.**

- **Location:** Appendix Tables 24–26 and LoCoMo judge protocol, PDF p.20 l.15–57; p.24 l.39–54.
- **Short quotes (7 and 8 words):** “training peak memory was not recorded”; “The endpoint does not expose a dated model snapshot.”
- **What it weakens:** Reproducibility score for training cost and exact LoCoMo replication.
- **Why this matters:** Exact model revision, adapter hash, seeds, prompts, and scorers are excellent. However, an undated `gpt-4o` endpoint may drift, and training peak memory/total experimental GPU-hours are not recorded. The independent DeepSeek-V3 audit mitigates judge-order concerns but not exact reproducibility.
- **Remedy:** Pin a dated judge snapshot or release all judge decisions plus a deterministic open judge; record training peak memory and complete compute logs in future runs.

---

## 6. All-figure/table audit

### Main figures

- **Figure 1 (PDF p.2 l.1–49):** clear and internally consistent. It distinguishes independent Write, overlap-Write, same-pack \(j=0\), and \(j>0\). The “same model output” label should be read as the same output interface/logit target, not guaranteed identical values; the text correctly documents quality loss.
- **Figure 2 (PDF p.4 l.1–42):** visually legible; appropriately described as motivational/correlational, not a causal “understanding layer” result. Appendix Table 6 supplies definitions and caveats.
- **Figure 3 (PDF p.4 l.9–43):** legible and narrowly scoped to lexical needle retrieval. The caption/text correctly warns that retrieval is not always the bottleneck.

### Main tables

- **Table 1:** useful interface taxonomy, but incomplete relative to the nearest PIC/modular-cache literature; this is the core novelty weakness.
- **Table 2:** strongest table; matched and statistically supported.
- **Table 3:** correctly caveated as store-already-built online prefill, not end-to-end or depth-only.
- **Table 4:** strong paired diagnostic, but no natural-task validation or measured overlap-Write latency.
- **Table 5:** valuable negative result; calibration/evaluation split details should be more explicit in the paper/artifact.

### Appendix tables

I inspected Tables 6–37. They are generally readable and captions usually state cohort, support, and non-comparability caveats. Specific concerns:

1. **Table 8:** the \(j=12\) matched Write value was not retained (PDF p.13 l.8–14), limiting a complete depth frontier.
2. **Table 9:** explicitly says it is not one matched Pareto frontier; quality cohorts and LoRA differ, so it should not be used for causal ranking (PDF p.13 l.17–47).
3. **Tables 13–15:** useful but mostly point estimates; Table 13 samples are not paired and Table 15 is not a stability ranking (PDF p.14 l.1–23).
4. **Tables 16–22:** broad coverage, but many method rows use different backbones/protocols or are not budget-matched. The captions acknowledge this. LongBench absolute scores are low and differences are mostly descriptive.
5. **Table 23:** directly supports fixed model Read but uses \(n=20\) recall and \(n=10\) generation; reasoning scores are exploratory (PDF p.20 l.1–13).
6. **Tables 24–26:** excellent reproducibility detail; artifact itself was outside permitted review inputs.
7. **Table 27:** useful retained-budget comparison, but SnapKV/PyramidKV are not nearest cross-query cache-reuse systems and hardware differs for systems numbers.
8. **Tables 28–30:** strong latency/quality control and useful external-store microbenchmark.
9. **Tables 31–35:** portability demonstrations are heterogeneous and partly small-sample; they do not replicate the central matched frontier.
10. **Tables 36–37:** cohort accounting and LoCoMo denominators are carefully documented.

I found no arithmetic contradiction in the headline numbers: \(931.9/664.4\approx1.403\), \(71.37/1.10\approx64.9\), and the RULER-B sum/macro reported in Table 36 is consistent.

---

## 7. Citation authenticity and citation–claim matching

### 7.1 Frozen `main.bbl` integrity

- All **47 cited keys** in the frozen source appear in `main.bbl`; there are no missing or uncited `main.bbl` entries.
- I checked authoritative metadata through DOI/Crossref, ACL records, arXiv metadata/abstracts, or official paper pages where available.
- The cited works themselves are generally real. The main authenticity issues are bibliographic metadata errors noted in W6.

### 7.2 Eight citation–claim spot checks

1. **Text RAG recomputes the reader** — Lewis et al. (2020) and Xu et al. (2024): **match.** These are standard retrieval-plus-reader references, and the paper limits the claim to conventional text retrieval.
2. **Cache-Craft reuses/repairs context-sensitive chunk KVs** — Agarwal et al. (2025): **match.** The authoritative abstract explicitly describes precomputed chunk-caches and partial recomputation to restore quality.
3. **HCache restores state from intermediate activations** — Gao et al. (2025): **match.** The official paper states that it restores KV from smaller intermediate activations; however, the frozen citation omits the authoritative DOI/pages.
4. **KV-Direct reconstructs layer-wise KV from residuals** — Qasim et al. (2026): **match.** The arXiv abstract explicitly claims deterministic reconstruction from residual streams.
5. **ILRe uses intermediate-layer representations for retrieval/context compression** — Liang et al. (2025): **match.** The arXiv abstract describes selecting an intermediate layer, streaming chunks to that layer, and recalling tokens by its key cache.
6. **REFORM compresses, gathers salient tokens, and recomputes KV** — Song et al. (2025): **match.** This is directly stated in the abstract.
7. **MemoryLLM maintains a fixed-size latent memory pool** — Wang et al. (2024): **match.** The abstract explicitly describes a fixed-size memory pool in transformer latent space.
8. **SnapKV/PyramidKV are token/KV compression methods** — Li et al. (2024), Cai et al. (2024): **match at family level.** Their use as retained-budget baselines is reasonable, though they are not close cross-query persistence baselines.

### 7.3 Citation corrections

- **HCache:** add DOI `10.1145/3689031.3696072`, pages 128–143.
- **BM25:** DOI `10.1561/1500000019` corresponds to *Foundations and Trends in Information Retrieval* **4(1–2):1–174**, not 3(4):333–389.
- Audit malformed author records, especially HeteroCache.

---

## 8. Novelty search and three-month rule

**Search date / frozen date:** August 3, 2026. I used a conservative **May 3, 2026 cutoff**: work first public after that date is treated as too recent to count against novelty, though it remains useful context.

### Q1. Has prior work independently cached retrieved chunks across changing positions/queries and composed them at inference?

**Nearest work found:** CacheBlend (May 2024), EPIC (October 2024), APE (February 2025), Cache-Craft (February 2025), TurboRAG (October 2024), MEPIC (December 2025), FusionRAG (January 2026).

**Finding:** Yes. This broad problem and the independent-chunk/PIC abstraction substantially predate CoMem. CoMem’s differentiator must therefore be the *stored object and continuation depth*, not independent cross-query chunk caching by itself.

### Q2. Has prior work stored intermediate activations/residuals rather than all-layer KV and recomputed later layers?

**Nearest work found:** HCache (October 2024 preprint; EuroSys 2025) stores intermediate activations to restore state; KV-Direct (March 20, 2026) stores residual vectors and reconstructs KV; ILRe (August 2025) uses intermediate-layer representations for retrieval; REFORM (June 2025) gathers tokens and recomputes KV.

**Finding:** Yes, the intermediate-state/recompute axis also predates CoMem. CoMem’s narrower contribution is a persistent retrieved corpus of one depth-\(j\) residual per token plus direct upper-layer continuation on a bounded pack.

### Q3. Has prior work combined independently persistent learned cache objects with bounded retrieval and frozen-reader inference?

**Nearest work found:** Cartridges (June 6, 2025 preprint/workshop), KV Packet (April 14, 2026), Cartridges at Scale (June 3, 2026), SemPIC (July 30, 2026).

**Three-month decision:**

- **Cartridges (June 6, 2025 preprint/workshop):** outside grace window; counts. It distills long contexts into lightweight reusable trainable KV representations while freezing the model. The frozen paper does not cite it.
- **KV Packet:** outside grace window; counts. It provides immutable independently compiled document KV packets, self-distillation, and recomputation-free composition.
- **Cartridges at Scale:** inside the May 3–August 3 three-month window (first public June 3, 2026), so I do **not** count its new per-document retrieval system as novelty-invalidating under the requested rule. It is nonetheless a very close contextual comparator: persistent per-document learned KV objects, persistent storage, and retrieval selecting \(k\) cartridges.
- **SemPIC:** inside the grace window (July 30, 2026), so I do **not** count it against novelty. It is highly relevant because it uses an offline LoRA Writer, behavioral distillation, independently compiled per-layer KVs, and an unchanged Reader.

**Finding:** Cartridges and KV Packet materially narrow the learned-persistent-cache, training/distillation, and independent-composition aspects before the cutoff. CAS/SemPIC should be discussed as concurrent work.

### Q4. Has prior work used lower-context/overlap or offline contextualization to repair independently written chunk states?

**Nearest work found:** APE uses a shared prefix; EPIC prepends/discards dummy tokens and recomputes initial tokens; Cache-Craft/CacheBlend selectively repair context-dependent cache regions; FusionRAG’s offline preprocessing embeds information from related chunks; SemPIC trains the Writer representation.

**Finding:** The general principle that independent cache construction needs contextual/interface repair is established. CoMem’s specific finding—that a small left document overlap at a single intermediate residual depth recovers a paired multikey gap without increasing stored bytes or Read work—appears more specific and plausibly novel.

### Q5. Is the exact conjunction claimed by CoMem already demonstrated?

I did **not** find, before the cutoff, a clear prior system that stores exactly one residual per token at a tunable intermediate depth, applies bounded query-conditioned selection over a persistent corpus, and directly resumes only upper layers over the selected pack. Thus, a narrow novelty claim may survive. However, because several omitted works cover three of the four surrounding components, the paper needs a much more careful comparison before the conjunction claim is convincing.

---

## 9. Reproducibility assessment

**Positive evidence:** exact backbone revision prefix, dimensions, split, retrieval hyperparameters, masks, prompts/scorers, sample counts, seeds, adapter modules/parameter count/hash, objective, optimizer, schedule, dtype, hardware, and environment are reported (PDF p.20 l.15–57; p.21 l.1–18). RULER shard integrity and rescoring checks are documented (PDF p.24 l.4–21).

**Remaining barriers:** the code archive was not part of the allowed frozen artifacts; the serving crossover table is absent; the LoCoMo endpoint is undated; training peak memory and total project compute are missing; some large-model results rely on small samples; and one central training-seed audit changes batch size.

**Assessment:** A competent group could likely reproduce the main method and approximate results, but exact numerical replication—especially LoCoMo and crossover measurements—may vary.

---

## 10. Ethics, limitations, and societal impact

The limitations and ethics sections are stronger than average. The authors discuss English-only evaluation, lexical retrieval bias, model-tied persistent states, update invalidation, storage growth, one-off-query disadvantage, lack of quantization/eviction/multi-tenant contention, and the narrowness of overlap-Write validation. The ethics section notes hallucination/bias inheritance, sensitive retrieval, inversion/membership risks, cross-tenant access, authorization, isolation, encryption, auditing, and deletion. I do not see a need for specialized ethics review based on the frozen paper. A useful addition would be an empirical deletion/inversion study, but it is not required for this methodological contribution.

---

## 11. Questions for the authors

1. How do you distinguish CoMem quantitatively and conceptually from APE/EPIC/PIC, KV Packet, and concurrent CAS/SemPIC beyond “one residual rather than all-layer KV”? Which method wins at equal quality, equal online latency, and equal persistent bytes?
2. Can you provide the full write-once serving table used to derive the 8–11 and 25.8/27.6 crossover values?
3. Does 32-token overlap-Write improve LongEval, BABILong, LongBench, and LoCoMo, and how much does its measured Write overhead move the crossover?
4. Why is the same LoRA mounted on the \(j=0\) teacher/replay arm in the central depth comparison, even though the adapter is trained only for layers 12–35? Please quantify the LoRA-on versus LoRA-off \(j=0\) effect on quality and latency in the same harness.
5. Can you release paired predictions and bootstrap scripts for all main benchmark deltas, not only RULER/LoCoMo?
6. What was the intended public cutoff for related work? If the paper froze on August 3, 2026, why are Cartridges, KV Packet, and the broader PIC literature absent?

---

## 12. Comments, suggestions, and minor corrections

- Clarify in Figure 1 that “same model output” means the same output interface, not identical predictions.
- Report measured overlap-Write wall-clock cost; theoretical FLOPs understate chunk-level overhead (already acknowledged in Table 4).
- The appendix says “Dense or learned retrieval remains untested” while the main text mentions a frozen BGE experiment. Reconcile the wording: apparently one dense retriever is tested, but learned/retrained retrieval is not.
- Correct the HCache and BM25 bibliography records and audit all 2026 metadata.
- Consider replacing “pre-register” in §5.5 unless an externally timestamped preregistration exists; “we pose a direct challenge” would be safer.
- Explain the 6.5k versus 6.7k/6,657 pack terminology consistently.

---

## 13. ARR ratings

### Soundness: **3.5 / 5**

The central matched depth-reuse claim, adapter effect, and focused Write-context diagnosis are technically sound and unusually well controlled. I lower from 4 because the practical crossover is not fully auditable, the overlap repair is not validated on main workloads, and the competitive frontier against the nearest cache-reuse systems is missing.

### Excitement: **3.5 / 5**

Treating transformer depth as an explicit cross-query reuse knob and measuring its quality/storage/latency frontier is interesting and useful. The paper’s honesty and mechanism analysis are compelling. Excitement is reduced because substantial surrounding ideas—independent reusable caches, PIC, residual/intermediate checkpoints, offline distillation—already exist, and the exact incremental advantage is not yet demonstrated head-to-head.

### Overall assessment: **3.0 / 5 — Findings**

I believe the paper is sound enough for Findings and contains a useful systems result with strong internal controls. I do not currently recommend conference acceptance because the novelty positioning and nearest-baseline evaluation are incomplete, and the deployable overlap repair remains narrow. A revision addressing W1–W3 could plausibly move this to conference level.

### Reviewer confidence: **4 / 5**

I read the full paper and appendices twice, checked all figures/tables, audited the frozen bibliography, and conducted targeted nearest-work searches. I am quite sure of the technical assessment, though exact novelty judgments in this fast-moving 2026 systems literature remain somewhat uncertain.

### Reproducibility: **4 / 5**

The paper provides most details needed to reproduce the main method, and reports exact hashes/revisions and evaluation protocols. Exact LoCoMo replication, crossover replication, and some systems details may vary or require the unavailable archive.

### Limitations and societal impact: **Adequately discussed**

### Ethical concerns: **None beyond those already acknowledged**

### Needs ethics review: **No**

### Software value (conditional on the stated archive being complete): **4 / 5 — Useful**

### Dataset value: **1 / 5 — No new dataset contribution**

---

## 14. Final recommendation rationale

This is a careful, candid, and technically valuable paper whose **internal evidence is stronger than its external positioning**. The same-pack frontier is convincing, the negative equal-latency result is important, and the overlap-Write diagnosis is insightful. The decisive obstacle to a higher rating is that the frozen version does not establish why this particular residual-depth interface is preferable to the closest PIC/modular-cache alternatives, nor does it discuss several very near works available before the freeze. I therefore recommend **Findings (3.0)** rather than conference, with the principal revision priority being a corrected, comprehensive novelty/baseline comparison and broader validation of overlap-Write.
