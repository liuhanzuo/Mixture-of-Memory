```yaml
review_mode: normal
soundness: 3.5
excitement: 3.5
overall: 3.5
confidence: 4.0
reproducibility: 3.5
```

## Summary

This paper studies **transformer depth as a cross-query reuse axis** for repeated-query long-context inference. CoMem writes each document chunk once up to split layer \(j\), persists one residual vector per token, retrieves a bounded set of chunks, and resumes layers \([j{:}L)\). Its central experiment is unusually well controlled: the \(j=0\) and \(j=12\) arms use the same examples, selected chunks, order, mask, sink, and LoRA. On Qwen3-8B, \(j=12\) reduces isolated selected-pack Read latency from 931.9 to 664.4 ms (1.403×), while RULER-B falls from 99.19 to 96.07. The paper explicitly presents this as a quality–latency–storage frontier, not as quality-preserving acceleration or superiority over RAG/PIC/modular-KV systems. It also reports repeated-query break-even estimates, an equal-latency negative result favoring raw-text replay, and a focused context/position diagnosis with an overlap-Write repair.

The paper is technically careful, transparent about negative results and scope, and rich in controls. My main reservation is external and decision-level validation: the central causal comparison is strong, but the practical claim still lacks a same-backbone end-to-end comparison against the closest modular-cache systems, and the highlighted equal-latency negative result is under-specified in the paper. I view this as a strong Findings paper / borderline ACL-main paper.

## Claims and evidence map

- **C1 — A tunable depth-reuse frontier exists.** Evidence: Table 2 / Appendix Table 28, paired RULER-B quality and fixed-pack latency for \(j=0\) versus \(j=12\), with identical evidence and adapter.
- **C2 — The \(j=12\) interface requires adaptation.** Evidence: Appendix Tables 8–10 and Table 11; same-\(j\) adapter on/off gaps and a separately distilled depth curve.
- **C3 — Reuse, not one-off use, is required for system benefit.** Evidence: Table 3 and storage accounting; measured crossover varies with context length, generation length, and placement.
- **C4 — Saved lower-layer computation does not beat raw replay at matched online latency in the tested setting.** Evidence: Table 4, top-10 raw replay versus top-12 CoMem, 64.78 versus 53.22.
- **C5 — One important failure source is incompatible lower-layer Write context, not position remapping alone.** Evidence: Tables 5–6 and Appendix Table 29; document-context states restore the paired multikey cohort, while position-only remapping hurts.
- **C6 — Model-side Read is bounded by selected-pack length, although selector/store costs are not.** Evidence: Eq. (1), the fixed \(1+kc+|q|\) construction, store-scaling Table 23, and store-tier Table 30.
- **C7 — The implementation ports beyond one dense backbone.** Evidence: appendix Qwen scale checks and Hy3 partition/readout experiments; appropriately framed as portability rather than a replicated frontier.

## Strengths

1. **Excellent isolation of the paper's central causal quantity.** The sentence “*Thus \(j\) varies prepaid depth while holding evidence fixed*” (Introduction, lines 27–33) is backed by a genuinely matched \(j=0\)/\(j=12\) design. This avoids conflating retrieval, token budget, and depth reuse.
2. **Unusually honest systems boundaries.** The paper repeatedly separates selected-pack Read, store-ready prefill, Write-inclusive pipeline, selection, persistent I/O, and decode. It also reports the unfavorable equal-latency result rather than hiding it.
3. **Good claim calibration.** The text explicitly states “*This is not quality-preserving acceleration*” (Introduction, lines 35–43) and avoids claiming superiority over heterogeneous prior systems. This restraint makes the positive results more credible.
4. **Strong mechanism controls.** The continuous-prefix oracle, context × position factorization, and overlap-Write sweep are well chosen. The paper correctly notes that the oracle jointly restores several interactions and is a fidelity ceiling, not a single-factor explanation.
5. **Substantial reproducibility detail.** The appendix gives the exact split, model revision, adapter scope/hash, objective, optimizer, training tokens, prompts, generation budgets, scorer choices, sample counts, seeds, timing boundaries, and artifact scope.
6. **Statistical care.** The paired RULER bootstrap/McNemar analysis and LoCoMo conversation-cluster bootstrap are preferable to treating all examples as independent. Training-seed robustness is reported with its optimization mismatch acknowledged.
7. **Readable presentation despite density.** Figure 1 clearly communicates Write–Select–Read and the \(j=0\) endpoint; all 34 tables/figures were inspected and were legible, with no clipping or unresolved references observed.

## Weaknesses

### Major W1 — The practical systems comparison remains incomplete

- **Problem:** There is no same-backbone, same-hardware, end-to-end comparison with a closest PIC/chunk-KV/learned modular-cache implementation.
- **Why it matters:** The matched \(j=0\) control establishes the incremental depth effect, but it cannot establish whether a single-residual object is a useful practical design point relative to systems that repair/reuse per-layer KV. That comparison is central to ACL-main-level systems impact, especially because raw replay wins the paper's own equal-latency test.
- **Requested claim-linked evidence:** Implement one representative nearest baseline (e.g., CacheBlend/EPIC-like repair or KV Packet/Cartridges-like modular object) on the same Qwen3-8B, selected evidence, hardware, storage tier, and timing boundary; report quality, TTFT/read, Write, bytes/token, and crossover. If infeasible, narrow the practical contribution further to an internal measurement study.
- **Location and exact quote (≤25 words):** Conclusion, lines 13–15: “*The next decisive step is a same-backbone, end-to-end comparison with PIC and learned modular-KV systems*.”
- **Severity:** **Major**.

### Major W2 — The headline equal-latency negative result is not reproducibly specified in the paper

- **Problem:** The “mixed diagnostic cohort,” its constituent tasks/weights/sample counts, latency calibration split, exact latency values/dispersion, and bootstrap unit are not given in the main text or appendix.
- **Why it matters:** This is called “decision-relevant” and appears in the abstract, but readers cannot determine what population 64.78/53.22 summarizes, whether top-10 and top-12 are truly matched beyond a ±5% label, or whether the CI respects task/example dependence.
- **Requested claim-linked evidence:** Add a table defining the cohort and aggregation, calibration/evaluation sizes and IDs or seed, measured latency for both arms (not only “within ±5%”), timing boundary/hardware, paired unit, bootstrap procedure/seed, and per-task scores. This does not require a new benchmark—only disclosure of the evidence supporting the claim.
- **Location and exact quote (≤25 words):** Section 5.2, lines 46–50: “*on the held-out cohort, raw replay leads by 11.56 points*.”
- **Severity:** **Major**.

### Minor W3 — Training-seed evidence is summarized too coarsely to judge headline stability

- **Problem:** The two added adapters are reported only through median/max cell-wise standard deviation, without seed-level headline aggregates, and they use effective batch 3 rather than 8.
- **Why it matters:** The paper says a rank-32 adapter “makes this operating point usable”; readers need to know whether the 96.07 frontier and natural-task scores are typical, not merely within a broad cell range.
- **Requested claim-linked evidence:** Report each seed's RULER-B macro and principal natural-task aggregates, mean ± SD/range, and clearly separate batch-size robustness from a controlled seed study. No additional seeds are necessary for this request.
- **Location and exact quote (≤25 words):** Appendix A.5, lines 304–312: “*The two added seeds use effective batch 3 rather than 8*.”
- **Severity:** **Minor**.

### Minor W4 — The bibliography is materially stale/incomplete in publication metadata

- **Problem:** Several entries are cited as arXiv preprints despite established proceedings/journal versions (e.g., LongBench and LoCoMo at ACL 2024; SnapKV, MiniCache, BABILong, and InfLLM at NeurIPS 2024; CacheBlend at EuroSys 2025; RAGCache in ACM TOCS 2025).
- **Why it matters:** This weakens the requested complete citation audit and makes venue/status comparisons harder, although it does not invalidate the technical claims.
- **Requested fix:** Update titles/authors/venues/DOIs from authoritative records and add arXiv identifiers to entries that currently omit them (notably ILRe and KV-Direct).
- **Location and exact quote (≤25 words):** References, pages 7–9: “*LongBench: A bilingual, multitask benchmark for long context understanding*.”
- **Severity:** **Minor**.

## Questions for the authors

1. What exactly constitutes the Table 4 mixed cohort: tasks, lengths, sample counts, weighting, and paired bootstrap unit?
2. What were the actual online latencies (with dispersion) for raw top-10 and CoMem top-12 on calibration and held-out splits?
3. Can you provide seed-level RULER-B and LoCoMo/LongEval aggregates for all three adapters?
4. For Eq. (2), is renormalizing only over the teacher top-64 support intended to discard the teacher's tail mass entirely for both KL directions? What fraction of teacher probability mass is retained on average?
5. Does the same-LoRA \(j=0\) control use the adapter in exactly the same upper layers during full replay, and is its 99.19 quality materially different from an adapter-disabled \(j=0\) run on the same RULER-B examples?

## Suggestions

- Promote a compact “timing boundary” diagram/table to the main paper; the distinctions are correct but distributed across several captions.
- Add a small seed-level table rather than only prose in Appendix A.5.
- Define “store-ready online prefill” at first use and consistently reserve “Read” for the isolated model phase.
- Provide retained teacher top-64 mass statistics and a full-vocabulary or larger-support distillation ablation if already available; otherwise state this as an objective approximation.
- Update bibliographic metadata and ensure all arXiv-only 2026 references have identifiers and first-posting dates in the artifact bibliography.

## Typos and presentation notes

- Table 32 says the context grows “16× (16k→256k),” while its mean context row is 16k→261k (about 16.3×); “approximately 16×” would be more precise.
- “8 KB/token” and “8 KiB/token” are used in different places. Since 8,192 bytes is exactly 8 KiB, use KiB consistently.
- The paper alternates among “Read,” “model Read,” “online prefill,” and “replay”; a glossary would reduce cognitive load.
- Table 4's column “Matched quality” is vague; name the metric and cohort directly.
- Page 3's Figure 2 is small but readable; larger axis/legend text would help print readability.

## Score rationale

- **Soundness 3.5/5:** The main internal comparison is well controlled, formulas and sampled numbers check out, statistical treatment is mostly careful, and limitations are explicit. The under-specified equal-latency cohort and absence of a nearest-system comparison prevent a 4+.
- **Excitement 3.5/5:** Treating depth as an explicit reusable-computation axis and measuring its frontier is interesting and useful. The core object has meaningful precedent in reusable intermediate representations/residual checkpointing, so novelty is more in the controlled frontier and systems accounting than in the broad idea of reusable state.
- **Overall 3.5/5:** Strong Findings / borderline ACL main. The paper is above a routine 3 because it is unusually rigorous and candid, but ACL-main-level practical impact would be much clearer with W1 resolved and Table 4 fully specified.
- **Confidence 4.0/5:** I read the full 21-page frozen PDF twice (including appendices), inspected source and every figure/table, checked formulas/numbers and citations, and ran targeted novelty searches. Some online bibliographic items were inaccessible and are marked Unverifiable below.
- **Reproducibility 3.5/5:** Configuration and artifact descriptions are strong, but the frozen source does not include the artifact/code itself, the equal-latency experiment is not sufficiently specified, and some environment versions are future/current snapshots that I could not independently execute.

## Desk, compliance, style, anonymity, ethics

- **Page/style:** Main content occupies pages 1–7, with Limitations and Ethics before the references; references end on page 9 and appendices start on page 10. This appears compliant with an 8-page ACL main-text limit and uses review-mode `acl.sty`, A4, 11pt. No overfull/clipped content was visible.
- **Limitations:** Present and substantive; covers backbone/language scope, baseline mismatch, linear store growth, update invalidation, retrieval ceilings, overlap scope, extrapolation, compute, and tail latency.
- **Anonymity:** “Anonymous ACL submission”; no author names, affiliations, repository URLs, or obvious self-identifying acknowledgments found. Model/provider names and artifact hashes are technical, not identity leaks.
- **Unresolved material:** Mechanical checks found no undefined refs/citations, duplicate labels, `??`, TODO/TBD/FIXME, or placeholder text in the rendered paper.
- **Hidden manipulation:** Source grep and visual/PDF inspection found no reviewer-directed instructions, hidden white/tiny text, or prompt-injection content. Source comments are ordinary build/result notes.
- **Ethics:** The paper discusses residual inversion/membership risks, access control, deletion, encryption, sensitive retrieval, inherited model harms, and energy. No new human-subject collection is claimed. I see no ethics-review trigger beyond normal data/license verification.
- **Licenses/artifacts:** The appendix gives a thoughtful license/data-scope discussion. Exact license statements were not all independently checked; see citation audit.

## Complete citation audit

I audited all **43 `main.bbl` entries** for bibliographic plausibility/status and checked representative claim-to-citation matches. Status vocabulary: **Verified** = authoritative DOI/venue or arXiv metadata checked; **Plausible** = internally consistent and/or well-known but not independently resolved in the time-limited network audit; **Unverifiable** = endpoint/record could not be independently confirmed. The audit uses the frozen date **2026-08-03**.

| # | Citation | Audit result |
|---:|---|---|
| 1 | Cache-Craft (Agarwal et al., 2025) | **Verified** DOI 10.1145/3725273; claim as chunk-KV cache system matches. |
| 2 | LongBench (Bai et al.) | **Verified content, metadata stale:** ACL 2024 long paper exists; bbl lists 2023 arXiv only. |
| 3 | PyramidKV (Cai et al., 2024) | **Plausible/Unverifiable venue:** arXiv 2406.02069 is plausible; no authoritative proceedings version confirmed. |
| 4 | KV Packet (Chen et al., 2026) | **Verified** arXiv 2604.13226, first posted 2026-04-13; before the three-month cutoff. |
| 5 | Cartridges (Eyuboglu et al., 2025) | **Verified** arXiv 2506.06266; claim as trained reusable KV representation matches abstract. |
| 6 | HCache (Gao et al., 2025) | **Verified** EuroSys DOI 10.1145/3689031.3696072. |
| 7 | Llama 3 (Grattafiori et al., 2024) | **Verified** arXiv 2407.21783. |
| 8 | Cartridges at Scale (Hardalov et al., 2026) | **Verified**, but first posted 2026-06-03, so post-cutoff and not a missing-citation weakness under the stated rule. |
| 9 | Distillation (Hinton et al., 2015) | **Verified/Plausible** classic arXiv 1503.02531. |
| 10 | RULER (Hsieh et al., 2024) | **Verified/Plausible** arXiv 2404.06654; benchmark claim matches. Venue metadata not confirmed. |
| 11 | LoRA (Hu et al., 2022) | **Verified** ICLR paper; method citation matches. |
| 12 | EPIC (Hu et al., 2025) | **Verified** arXiv 2410.15332 and ICML 2025 status plausible; PIC characterization matches abstract. |
| 13 | RAGCache (Jin et al., 2024) | **Verified content, metadata stale:** ACM TOCS 2025 DOI 10.1145/3768628 exists; bbl lists arXiv only. |
| 14 | BABILong (Kuratov et al., 2024) | **Verified content, metadata incomplete:** NeurIPS 2024 proceedings DOI record exists. |
| 15 | RAG (Lewis et al., 2020) | **Verified** NeurIPS 2020; raw-text retrieval characterization is appropriate. |
| 16 | LongChat/LongEval blog (Li et al., 2023) | **Plausible** LMSYS blog reference; direct benchmark provenance is acceptable, though a versioned repository citation would be stronger. |
| 17 | SnapKV (Li et al., 2024) | **Verified content, metadata stale:** NeurIPS 2024 proceedings record exists. |
| 18 | ILRe (Liang et al., 2025) | **Verified** arXiv 2508.17892; bbl should include the identifier. Characterization as intermediate-layer token retrieval is accurate. |
| 19 | ReadOnce (Lin et al., 2021) | **Verified** ACL DOI 10.18653/v1/2021.acl-long.554. |
| 20 | MiniCache (Liu et al., 2024) | **Verified content, metadata stale:** NeurIPS 2024 proceedings record exists. |
| 21 | TurboRAG (Lu et al., 2024) | **Verified** arXiv 2410.07590; precomputed chunk KV characterization matches. |
| 22 | LoCoMo (Maharana et al., 2024) | **Verified content, metadata stale:** ACL 2024 long paper DOI 10.18653/v1/2024.acl-long.747. |
| 23 | XC-Cache (Monteiro et al., 2024) | **Verified** EMNLP Findings DOI 10.18653/v1/2024.findings-emnlp.896. |
| 24 | KV-Direct (Qasim et al., 2026) | **Verified** arXiv 2603.19664; bbl omits identifier. Residual-checkpoint/reconstruction characterization matches abstract. |
| 25 | PG-19 / Compressive Transformers (Rae et al., 2019) | **Verified/Plausible** arXiv 1911.05507; dataset provenance is standard. |
| 26 | BM25 (Robertson & Zaragoza, 2009) | **Verified** DOI 10.1561/1500000019. |
| 27 | Embedding Recycling (Saad-Falcon et al., 2023) | **Verified** EACL Findings DOI 10.18653/v1/2023.findings-eacl.145. |
| 28 | GemFilter (Shi et al., 2024) | **Verified** arXiv 2409.17422; early-layer selection characterization matches. |
| 29 | REFORM (Song et al., 2025) | **Verified** arXiv 2506.01215; compress/gather/recompute characterization matches. |
| 30 | LLoCO (Tan et al., 2024) | **Plausible** arXiv 2404.07979; native supervised setup description is plausible but checkpoint details were not independently inspected. |
| 31 | Hy3 (Tencent Hunyuan Team, 2026) | **Unverifiable in full:** cited Hugging Face model page/architecture details were not independently archived during the audit. |
| 32 | Fusion RAG Cache (Wang et al., 2026) | **Verified** arXiv 2601.12904; selection/cache-fusion serving characterization matches. |
| 33 | MEPIC (Wang et al., 2025) | **Verified** arXiv 2512.16822; PIC characterization matches. |
| 34 | LongMem (Wang et al., 2023) | **Plausible** NeurIPS 2023 external-memory work; characterization is broad and appropriate. |
| 35 | MemoryLLM (Wang et al., 2024) | **Plausible** arXiv 2402.04624; self-updatable latent-memory characterization is appropriate. |
| 36 | InfLLM (Xiao et al., 2024a) | **Verified content, metadata stale:** NeurIPS 2024 proceedings record exists. |
| 37 | StreamingLLM (Xiao et al., 2024b) | **Verified/Plausible** ICLR 2024; attention-sink/bounded-window characterization matches. |
| 38 | SemPIC (Xie et al., 2026) | **Verified**, first posted 2026-07-30; post-cutoff and not a missing-citation weakness. |
| 39 | Retrieval Meets Long Context (Xu et al., 2024) | **Verified/Plausible** ICLR 2024; supports raw-text retrieval/recompute framing. |
| 40 | Qwen3 report (Yang et al., 2025a) | **Verified** arXiv 2505.09388; backbone citation matches. |
| 41 | APE (Yang et al., 2025b) | **Verified** arXiv 2502.05431; parallel encoding/attention realignment characterization matches. |
| 42 | CacheBlend (Yao et al., 2024) | **Verified content, metadata stale:** EuroSys 2025 DOI 10.1145/3689031.3696098 exists; bbl lists arXiv only. |
| 43 | H2O (Zhang et al., 2023) | **Verified/Plausible** NeurIPS 2023; KV eviction/compression characterization matches. |

### Citation–claim match checks (8)

| Paper claim | Cited work(s) | Match |
|---|---|---|
| Raw-text retrieval bounds online text but recomputes model layers | Lewis et al.; Xu et al. | **Good.** This is a fair abstraction of retrieval-augmented inference. |
| Modular/PIC caches reuse document KV while repairing context/position dependencies | CacheBlend, TurboRAG, EPIC, Cache-Craft, KV Packet | **Good, but heterogeneous.** The paper correctly avoids ranking them. |
| ReadOnce/Embedding Recycling cache intermediate representations and adapt later layers | ReadOnce; Embedding Recycling | **Good.** Closely relevant precedent. |
| HCache restores state; KV-Direct reconstructs KV from residuals | HCache; KV-Direct | **Good.** Distinctions from CoMem are accurately stated. |
| ILRe/REFORM select or gather before recomputation | ILRe; REFORM | **Good.** ILRe is especially close on intermediate-layer offline processing, though it retrieves tokens rather than persisting a selected residual pack for direct suffix execution. |
| SnapKV/PyramidKV change retained KV/token budget after prefill | SnapKV; PyramidKV | **Good.** The paper explicitly separates their full-prefill boundary. |
| External memory systems are not matched depth controls | MemoryLLM; LongMem; XC-Cache | **Good.** Appropriate use as external references. |
| Distillation + LoRA motivate the adapter recipe | Hinton et al.; Hu et al. | **Partial but acceptable.** They motivate generic techniques, not the paper's symmetric top-64 KL objective, which is the authors' implementation choice. |

## Novelty search and closest-work comparison

I ran five targeted searches over arXiv/DOI metadata, frozen at **2026-08-03**. Per the requested three-month rule, work first posted after **2026-05-03** (and preprint-only work in that window) was used for context but not counted as a missing-citation weakness.

1. **Reusable intermediate/residual representations:** ReadOnce (2021), Embedding Recycling (2023), HCache (2025), ILRe (2025), and KV-Direct (2026) are the closest precedents. CoMem's distinction is the explicit tunable split-depth frontier with one persistent residual/token, bounded selected-pack direct suffix execution, and matched \(j=0\) evidence—not the broad idea that intermediate state can be reused.
2. **Position-independent/modular document caches:** CacheBlend, TurboRAG, EPIC, MEPIC, KV Packet, and Cartridges establish the broader reusable-document-cache space. CoMem is materially different in stored object and post-selection work, but the lack of a matched head-to-head is the main novelty/impact uncertainty.
3. **Residual-versus-KV storage:** KV-Direct (first posted 2026-03-20) is very close in observing that one residual/token can replace full KV storage. It reconstructs/replays inference state for bounded-memory decoding; CoMem's novelty is cross-query document persistence, selection, split-depth tuning, and its measured frontier.
4. **Intermediate-layer retrieval/compression:** ILRe (2025-08-25) performs offline partial-depth chunk processing and uses an intermediate-layer cache to retrieve tokens, but it does not directly resume a bounded selected residual pack through the suffix as CoMem does. This supports incremental rather than foundational novelty.
5. **Post-cutoff context only:** Cartridges at Scale (2026-06-03), HYPIC (2026-07-01), and SemPIC (2026-07-30) are relevant but fall after the 2026-05-03 cutoff. They are not treated as omissions. Irminsul was first posted 2026-05-07, also post-cutoff. No pre-cutoff paper found in these searches exactly combined CoMem's object, direct suffix execution, matched depth axis, and systems accounting.

**Novelty conclusion:** The paper is **incrementally but meaningfully novel**. The strongest novelty is experimental framing and causal/systems measurement, not the general concept of caching intermediate representations. I found no decisive unacknowledged pre-cutoff duplicate.

## Review-process self-check

- [x] Read the frozen 21-page PDF twice, including all appendices.
- [x] Used only the frozen PDF, frozen source, and requested normal template locally; did not inspect other reviews/history/TODO/status/current drafts.
- [x] Built a C1–C7 claims/evidence map.
- [x] Checked page/style, Limitations, anonymity, unresolved refs/placeholders, abstract/table consistency, hidden manipulation, ethics, and all figures/tables.
- [x] Mechanically confirmed all weakness quotes occur verbatim and contain no more than 25 words.
- [x] Mechanically checked no undefined citation/ref keys and audited all 43 `main.bbl` entries.
- [x] Checked 8 citation–claim pairs and ran 5 novelty searches with the 2026-05-03 cutoff.
- [x] Recomputed more than five headline numbers: 1.403× speedup; RULER-A/B macros; LongEval 5/6-length means; BABILong means; LongBench macro; LoCoMo weighted score; 1/18 storage ratio; 1 GiB at 128k; 6,657-token cap; 64.9× dense-prefill ratio; equal-latency and LoCoMo gaps. They agree up to rounding.
- [x] Checked formulas and boundaries: Eq. (1) is correct for a standard GQA decoder at common dtype; Eq. (2) is well-defined after support renormalization, though retained mass is not reported; selector/index/store are correctly excluded from the bounded model-work claim.
- [x] Requested only experiments/disclosures needed for stated claims; no generic benchmark shopping list.
- [x] Separated Soundness from Excitement and derived scores from the written critique.
- [x] Network failures/insufficient authoritative records are explicitly labeled **Unverifiable**, not silently assumed.
