```yaml
review_mode: strict
soundness: 3.5
excitement: 3.0
overall: 3.0
confidence: 4.5
reproducibility: 3.5
```

# Summary

This paper studies a deliberately narrow systems question: for repeated queries over a stable document collection, how much transformer depth should be prepaid into a persistent reusable object? CoMem independently writes each document chunk through layers \([0:j)\), stores one depth-\(j\) residual per token, retrieves a fixed top-\(k\) set, and resumes layers \([j:L)\). Its strongest result is not a cross-system win but a same-evidence, same-adapter \(j=0\) versus \(j=12\) frontier on Qwen3-8B: selected-pack model Read drops from 931.9 to 664.4 ms (1.403×), while the paired 15-cell RULER-B macro drops from 99.19 to 96.07. The paper also measures Write amortization, reports a negative equal-online-latency comparison against raw replay, and uses context/position/overlap controls to localize one multikey failure to the Write-side representation.

The submission is unusually careful about timing boundaries, negative results, statistical pairing, and claim scope. I nevertheless see two material gaps. First, the main decision-relevant equal-latency result is not specified well enough in the frozen paper to audit or reproduce the quality mixture or actual latency equivalence. Second, the claimed broader quality scope depends heavily on one adapter trained on PG-19, while overlap with several natural evaluation sets remains unaudited and training-seed evidence is only a partial, batch-confounded robustness check. These issues keep me at Findings rather than ACL-main level.

# Claims and evidence map

For each claim, I first state the minimum sufficient evidence and then assess the supplied evidence.

**C1 — CoMem defines a reusable depth axis with a matched \(j=0\) endpoint.**  
Minimum sufficient evidence: exact Write/Select/Read semantics, boundary cases \(j=0,L\), masks/positions, and an implementation fidelity check.  
Evidence: §4, Fig. 1, Appendix Tables 24 and 29; the continuous-prefix \(h_{12}\) path is bit-identical to full replay on 1,500 examples. **Supported for the implemented decoder path.**

**C2 — Reusing the lower 12 layers yields a measured quality–Read-latency trade-off.**  
Minimum sufficient evidence: paired examples and packs, identical adapter and decode setup, isolated timing boundary with repeated processes, and uncertainty on quality.  
Evidence: §5.1, Table 2, Appendix Table 28: 99.19→96.07, paired-bootstrap 95% CI \([2.36,3.93]\), McNemar \(p=8.79\times10^{-24}\), 931.9→664.4 ms across three processes. **Strongly supported for isolated selected-pack Read on the reported H20 setup.**

**C3 — A rank-32 upper-layer LoRA makes the \(j=12\) residual interface usable without changing the backbone.**  
Minimum sufficient evidence: same-\(j\), same-data, same-evaluation adapter on/off comparison plus training details.  
Evidence: §4, Appendix Tables 8, 10, 11, 25. Large gains appear on RULER, LoCoMo, BABILong, and an HCache-style path; the backbone is frozen. **Supported, but robustness across fully matched training seeds is incomplete (W3).**

**C4 — The 8 KiB/token residual object is 18× smaller than full per-layer KV and bounded model-side Read is independent of stored-context length.**  
Minimum sufficient evidence: correct storage formula under the stated GQA architecture/dtype, fixed \(k,c,q\), and scaling measurements separating selector/store costs.  
Evidence: Eq. (1), §4, Tables 23, 24, 30. The algebra gives \(4096\times2=8192\) B/token and full KV \(2L n_{\rm kv}d_{\rm head}\times2=147{,}456\) B/token, hence \(1/18\). Table 23 keeps Read near 6.2–6.5k tokens while BM25 grows with store size. **Supported under common dtype and fixed top-\(k\); external selection/store are correctly excluded from “bounded.”**

**C5 — One-time Write amortizes only after repeated queries, roughly 8–11 at 32k and 26–28 at 128k in measured cells.**  
Minimum sufficient evidence: measured Write, raw-token index, fetch, Read and decode times under matched packs, plus explicit equation and placement/generation sensitivity.  
Evidence: §5.2, Table 3 and Appendix A.4. **Supported for the retained single-query serving harnesses; not production-concurrent throughput.**

**C6 — At matched online latency, raw replay is more accurate than CoMem.**  
Minimum sufficient evidence: exact held-out task composition and denominators, metric aggregation, calibration split, raw latency distributions/values under a single timing boundary, and paired quality samples.  
Evidence: §5.2 and Table 4 report 64.78 versus 53.22 and a paired CI, but the cohort and actual timing values are not defined. **Direction is plausible, but the headline evidence is not independently auditable (W1).**

**C7 — A focused 2×2 factorization localizes a tested multikey gap to the Write representation rather than position remapping alone.**  
Minimum sufficient evidence: same examples/retrieval/adapter/upper Read, crossed context-scope and position interventions, and interaction-aware interpretation.  
Evidence: §5.3, Table 5: 92.5/88.0 versus 100.0/100.0; the paper explicitly notes the +4.5 interaction and avoids additive attribution. **Supported on the stated 8k/16k synthetic cohort only.**

**C8 — A deployable 32-token left overlap recovers most of that local gap without increasing persistent bytes or per-query Read work.**  
Minimum sufficient evidence: paired overlap-width intervention, unchanged stored target residuals/Read pack, uncertainty, and Write-cost accounting.  
Evidence: §4, §5.3, Table 6: 92.5→98.5, CI \([3.0,9.5]\), same stored bytes and Read/decode, but higher Write FLOPs/wall time. **Supported as a local synthetic repair, not a general quality fix.**

**C9 — Results establish an internal frontier rather than superiority over RAG/PIC/modular-KV systems.**  
Minimum sufficient evidence: explicit non-superiority scope, taxonomy, and no misleading cross-system causal inference.  
Evidence: Abstract, §§1–2, Table 1, Limitations. **Appropriately scoped.** The missing same-backbone closest-system comparison is an acknowledged evidence boundary, not an unqualified superiority claim.

# Strengths

**S1. Excellent causal hygiene around the central result.**  
Anchor: §5.1, PDF p.4 lines 273–286 and Table 2, PDF p.5. The paper keeps examples, selected chunks, order, mask, sink, LoRA, and pack fixed, and changes replay start only. The continuous-prefix ceiling further distinguishes upper-layer continuation fidelity from chunk-independent Write incompatibility.

**S2. Transparent negative result and timing-boundary discipline.**  
Anchor: Abstract lines 20–23; §5.2; Appendix A.4, PDF pp.16–19. The authors repeatedly separate selected-pack Read, store-ready prefill, Write-inclusive pipeline, and external I/O. They explicitly state that 64.9× is not the incremental depth effect and that total decode is similar/decode-dominated.

**S3. The bounded mechanism diagnosis is appropriately factorial and scoped.**  
Anchor: Table 5, PDF p.5; Table 6, PDF p.6. The 2×2 context/position experiment reports the interaction rather than forcing a single-factor story, and the overlap intervention includes both statistical uncertainty and Write overhead.

**S4. Strong reporting of storage, amortization, and failure boundaries.**  
Anchor: Eq. (1), PDF p.4; Table 3, PDF p.5; Tables 23 and 30, PDF pp.17 and 19. Persistent bytes, HBM/CPU/NVMe/CEPH placement, selector scaling, evidence ceilings, and one-off-query disadvantages are all exposed.

**S5. Statistical reporting is substantially above the norm for this type of systems paper.**  
Anchor: Appendix B, PDF pp.20–21. The RULER gap is paired; LoCoMo includes both item and conversation-cluster bootstrap checks and an independent-judge subset; unsupported independence assumptions are explicitly avoided.

**S6. The writing is disciplined about non-claims.**  
Anchor: Abstract lines 30–34; Conclusion, PDF p.6; Limitations, PDF pp.6–7. The paper does not claim quality preservation, universal overlap repair, compression “for free,” a leaderboard win, or superiority over PIC/modular caches.

# Weaknesses

## W1 — The headline equal-latency negative result is not auditable from the frozen paper

- **Location:** §5.2 and Table 4, PDF p.5; Abstract lines 20–23.
- **Exact quote (13 words):** “The retrieval budget is chosen on a disjoint latency-only calibration split”
- **Problem:** The paper never defines the “mixed diagnostic cohort”: constituent tasks/cells, sample counts, per-task metric normalization, or aggregation weights. It also reports only “within ±5%,” not the two measured online-latency values, dispersion, hardware repetition protocol, or whether fetch/query-Write/decode are identically included. The paired CI is therefore not independently reconstructible from the manuscript.
- **Affected claim/norm and importance:** This weakens **C6**, which the paper calls its “decision-relevant negative result” and highlights in the Abstract. A heterogeneous scalar 64.78/53.22 is uninterpretable without its mixture and timing definition; the result could be dominated by one task family or a favorable latency calibration.
- **Sufficient remedy:** Add a compact appendix table giving every cohort component, \(n\), metric and normalization, aggregation formula, calibration/evaluation split rule, exact top-\(k\) latency medians with p10/p90 or bootstrap intervals, hardware/process repetitions, and full included/excluded timing boundary. Release the paired per-example quality and raw latency records referenced by the artifact statement.
- **Severity:** **Major.**

## W2 — Natural-task generalization is confounded by incompletely audited adapter-training overlap

- **Location:** Appendix A.3, PDF p.16 lines 752–771.
- **Exact quote (11 words):** “We did not complete equivalent overlap audits for every natural benchmark”
- **Problem:** The adapter is trained on PG-19, and the paper found substantial overlap with InfiniteBench long-book support, enough to remove that comparison. Equivalent audits are not supplied for NarrativeQA or the other natural benchmarks, yet LoCoMo/LongBench/BABILong/LongEval are used as scope checks for learned readout quality.
- **Affected claim/norm and importance:** This limits **C3** and any broad interpretation of **C9** beyond synthetic RULER. It does not invalidate the paired \(j=0\)/\(j=12\) causal comparison—both arms share the adapter—but it weakens claims that the learned interface transfers to unseen natural data.
- **Sufficient remedy:** Run the same or stronger contamination audit for every natural evaluation corpus; where overlap is detected, report clean-subset results or replace with verified-disjoint corpora. At minimum, separate synthetic causal evidence from potentially in-domain natural-task evidence in the headline scope.
- **Severity:** **Major.**

## W3 — Adapter robustness is not a fully controlled multi-seed estimate

- **Location:** Appendix A.5, PDF p.18 lines 893–906.
- **Exact quote (8 words):** “The two added seeds use effective batch 3”
- **Problem:** The flagship adapter is the only fully specified batch-8 run. The two added “seeds” change effective batch size, so seed variance is confounded with optimization noise; they are also evaluated on reduced RULER support (\(n=50\) per cell), not the full headline \(n=100\).
- **Affected claim/norm and importance:** This weakens the robustness component of **C3** and reproducibility of the exact \(j=12\) operating point. The observed median cell SD of 1.34 points is useful, but it is not a controlled estimate of training-seed uncertainty.
- **Sufficient remedy:** Train at least two additional adapters with exactly the flagship global batch, schedule, data order policy, and steps, varying seed only; evaluate all three on the full headline cells and report mean/SD or hierarchical paired intervals for quality and latency.
- **Severity:** **Minor.**

## W4 — The method’s position boundary is incompletely specified outside the Qwen implementation

- **Location:** §4 “Interface,” PDF p.3 lines 194–203; Appendix Table 24, PDF p.17.
- **Exact quote (5 words):** “orders them by document position”
- **Problem:** For RoPE, this is operationally clear because the stored residual precedes the upper-layer position rotation. The paper does not state the general condition under which cached \(h_j\) can be reassigned positions—e.g., learned absolute position embeddings or architectures that inject position before \(h_j\). Thus the generic “Let a decoder…” formulation is broader than the implemented interface guarantee.
- **Affected claim/norm and importance:** This narrows **C1** and the claimed portability of the formal method. It matters for readers trying to determine whether CoMem is architecture-agnostic or specifically compatible with position mechanisms injected inside the resumed blocks.
- **Sufficient remedy:** State an explicit compatibility condition for positional encoding and define the required transform, if any, for absolute/already-positioned residuals. Restrict the generic method claim to compatible decoder families unless experimentally validated.
- **Severity:** **Minor.**

## W5 — The main paper’s broad scope checks are not self-contained enough for an optional-appendix review

- **Location:** §5.1, PDF p.4 lines 287–292.
- **Exact quote (5 words):** “task-level BABILong results are mixed”
- **Problem:** The main text gives only aggregate natural-task numbers; almost all task definitions, denominators, prompts, generation lengths, judge details, and baseline caveats reside in twelve appendix pages. ACL review versions should be self-contained even though appendices are optional.
- **Affected claim/norm and importance:** This affects the presentation and verifiability of the non-RULER scope evidence supporting **C3/C9**, not the central paired RULER claim.
- **Sufficient remedy:** Put a compact main-text protocol table or footnote alongside the aggregate scope checks: datasets/support, \(n\), metric, generation budget, whether examples are paired, and which comparisons use different backbones/adapters.
- **Severity:** **Minor.**

# Questions that could change the score

**Q1 — Equal-latency cohort:** What exact tasks/cells and sample counts form the 64.78/53.22 “mixed diagnostic cohort,” and how are their metrics normalized and weighted? Please provide the two latency distributions and complete timing boundary. A satisfactory answer with auditable records could remove W1 and raise my score.

**Q2 — Natural-data overlap:** Were PG-19 overlap checks performed for LoCoMo, LongBench’s six QA datasets, LongEval, and BABILong? If yes, what are the clean-subset results? Evidence of disjointness or unchanged clean-subset conclusions could substantially reduce W2.

**Q3 — Matched adapter seeds:** Can the authors confirm whether any additional batch-8, otherwise identical \(j=12\) runs exist? Full-cell results from them would resolve W3.

**Q4 — Position compatibility:** Is \(h_j\) defined before every architecture-specific positional operation needed by layers \([j:L)\), or is the method intentionally limited to RoPE-like decoders?

# Suggestions / typos (non-scoring)

1. In Table 2, “Store/token \(\sim4\)–8 B” for token IDs depends on integer representation; label it as an implementation range rather than an intrinsic text-storage cost.
2. Correct archival metadata where available: several bibliography rows remain labeled only as arXiv despite later archival publication (for example TurboRAG), and the RULER/LongBench-style benchmark entries could cite their archival versions.
3. Define “Read,” “online prefill,” and “total decode” once in a boxed notation/table; the paper is careful, but readers must currently reconcile several cohorts.
4. Table 9 is information-dense and very small in the rendered PDF; consider splitting quality and timing or increasing type size.
5. Page 9 is mostly blank because the bibliography ends early. This is not a violation, but bibliography compaction could improve presentation.
6. Use one spelling consistently for “ReadOnce” (the bibliography renders “Readonce”).

# Score reasons

## Soundness: 3.5 / 5.0

The core \(j=0\) versus \(j=12\) trade-off is well controlled, statistically paired, and carefully bounded. The storage equation, timing separation, continuous-prefix ceiling, and 2×2 mechanism control are technically sound. I do not assign 4.0 because the headline equal-latency result is under-specified (W1), and natural-task transfer evidence has unresolved contamination and seed-control caveats (W2–W3).

## Excitement: 3.0 / 5.0

Treating depth as an explicit cross-query reuse axis and measuring the frontier is useful and timely, but the reusable-intermediate-state idea has clear precedents, and the paper does not yet establish a compelling end-to-end advantage over raw replay or closest modular-KV/PIC systems. Its most decision-relevant comparison is negative.

## Overall: 3.0 / 5.0

This is a solid Findings-level paper: honest, technically careful, and useful as a measured design point. It falls short of ACL-main (4.0) because two prominent empirical scopes are not yet adequately secured: the equal-latency cohort cannot be audited from the manuscript, and natural-task transfer is not contamination-clean. I considered 3.5, but under the requested lower-bin rule I choose 3.0 because W1 concerns an Abstract-level result and W2 affects the main evidence beyond synthetic diagnostics.

## Confidence: 4.5 / 5.0

I read the full 21-page PDF twice, including all appendices; inspected all figures/tables; checked formulas and boundary cases; mechanically compared references/labels/citations to the frozen source; and audited all 43 `main.bbl` entries. Remaining uncertainty is primarily artifact-level because only the frozen PDF/source were reviewable.

## Reproducibility: 3.5 / 5.0

The paper reports model revision, adapter hash, architecture, masks/positions, retrieval parameters, training objective and optimizer, seeds, benchmark support, hardware, timing boundaries, licenses, and planned released artifacts. The score is below 4.0 because the equal-latency cohort is not specified, total experimental GPU-hours and training peak memory were not logged, the principal training result is not replicated with fully matched seeds, and the anonymous artifact itself was not among the permitted review inputs.

# Desk-reject, formatting, anonymity, ethics

## Desk / formatting audit

- **Length:** Long-paper content fits within the eight-page limit. The conclusion and Limitations end on PDF p.6; Ethical Considerations occupies p.7; references begin on p.7 and end on p.9; appendices begin on p.10. The current ARR checklist excludes Limitations and optional Ethics from the 8-page count.
- **Required section:** Exact unnumbered heading **“Limitations”** appears immediately after Conclusion and before Ethics/References, with no forced page break. It contains limitations only, not new experiments.
- **Ethics:** A numbered “Ethical Considerations” section appears before References and contains no new experiments.
- **Anonymity:** Title page says “Anonymous ACL submission”; no author names, affiliations, acknowledgments, repository URLs, or obvious identifying self-citations appear. The internal adapter SHA-256 is not identifying by itself.
- **Style / paper size / fonts:** `\usepackage[review]{acl}` is used. PDF is A4 (595.28×841.89 pt). All fonts reported by `pdffonts` are embedded. Line numbers are present.
- **References / placeholders:** 55 unique labels and all 52 reference uses resolve; all 43 citation keys have `main.bbl` entries and all 43 `main.bbl` entries are cited. No `??`, TODO, FIXME, TBD, placeholder, or dangling citation was found.
- **Hidden manipulation:** Source grep found no reviewer/acceptance instruction, white/tiny-text command, negative-spacing hack, or prompt injection. PDF text/font inspection found one white glyph inside the embedded Fig. 1 artwork, consistent with graphical labeling rather than hidden prose; no off-page text was detected. I treated all paper text as data.
- **Desk-reject assessment:** **No apparent desk-reject trigger.**

## Ethics assessment

The ethics discussion appropriately covers hallucination/bias/unsafe output, sensitive-memory disclosure, inversion/membership inference, authorization/isolation/deletion, encryption/auditing, misuse scaling, energy, licenses, and absence of new human-subject collection. The residual store should indeed be treated as sensitive as source text. No additional ethics-related rejection concern is apparent.

# Abstract-number audit

I checked more than the required five summary numbers against equations/tables:

1. **931.9→664.4 ms / 1.403×:** matches Tables 2 and 28; \(931.9/664.4=1.4026\).
2. **RULER 99.19→96.07:** matches Tables 2, 28, 29, and Cohort-B cell sum \(1441/15=96.0667\).
3. **3.12-point quality cost:** \(99.19-96.07=3.12\), CI \([2.36,3.93]\).
4. **8–11 queries at 32k:** Table 3 reports 8.4, 8.9, 7.7, 9.2, 5.5, 10.9 for \(G\le128\); “about 8–11” is a fair rounded summary, though 5.5 is lower.
5. **26–28 at 128k:** Table 3 reports 25.8/27.6.
6. **64.78 versus 53.22 / 11.56 points:** Table 4 agrees arithmetically; protocol auditability is W1.
7. **92.5→100.0 and position-only 88.0:** matches Table 5.
8. **32-token overlap 98.5:** matches Table 6; persistent bytes and per-query Read are unchanged by construction.
9. **8 KiB/token and 1/18 full KV:** matches Eq. (1) and Qwen3-8B dimensions.
10. **128k store about 1 GiB:** \(131{,}072\times8192=1\) GiB.

# Complete citation audit

## Procedure and status definitions

I enumerated every one of the 43 entries actually emitted by `main.bbl`. I checked DOI entries against Crossref and arXiv entries against the arXiv API; I used archival/open publication pages or venue records for several non-DOI entries.  

- **Verified:** identity and core metadata match a resolvable primary record.
- **Metadata error:** identity is real, but the bibliography has a material venue/version/date/metadata issue.
- **Not found:** no matching work after a successful targeted lookup.
- **Unverifiable:** no stable identifier in `main.bbl` and the limited primary-record check was inconclusive. Per instruction, network/access failure is never converted to “Not found.”

## Entry-by-entry audit

1. `cachecraft` — **Verified** (DOI 10.1145/3725273; title/authors/2025/PACM DM match).
2. `longbench` — **Verified** (arXiv:2308.14508; title/year match).
3. `pyramidkv` — **Verified** (arXiv:2406.02069).
4. `kvpacket` — **Verified** (arXiv:2604.13226; first posted 2026-04-14, before freeze but concurrent/preprint).
5. `cartridgesbase` — **Verified** (arXiv:2506.06266; preprint).
6. `hcache` — **Verified** (DOI 10.1145/3689031.3696072; EuroSys 2025, pp.128–143).
7. `llama3` — **Verified** (arXiv:2407.21783).
8. `cartridges` — **Verified** (arXiv:2606.04557; after the 2026-05-03 cutoff, concurrent only).
9. `distillation` — **Verified** (arXiv:1503.02531).
10. `ruler` — **Verified** (arXiv:2404.06654).
11. `lora` — **Verified** (ICLR 2022 work identity; no material mismatch found).
12. `epic` — **Verified** (arXiv:2410.15332 and ICML 2025 identity).
13. `ragcache` — **Verified** (arXiv:2404.12457).
14. `babilong` — **Verified** (NeurIPS 2024 Datasets & Benchmarks identity; DOI 10.52202/079017-3381 located).
15. `rag` — **Verified** (NeurIPS 2020 proceedings identity).
16. `longchat` — **Verified** (LMSYS blog title/authors/date identity).
17. `snapkv` — **Verified** (arXiv:2404.14469).
18. `ilre` — **Verified** (arXiv:2508.17892; preprint).
19. `readonce` — **Verified** (DOI 10.18653/v1/2021.acl-long.554; ACL 2021).
20. `minicache` — **Verified** (arXiv:2405.14366).
21. `turborag` — **Metadata error**: arXiv:2410.07590 is real, but an archival EMNLP 2025 version exists (DOI 10.18653/v1/2025.emnlp-main.334); ACL guidance prefers the archival version.
22. `locomo` — **Verified** (arXiv:2402.17753).
23. `xccache` — **Verified** (DOI 10.18653/v1/2024.findings-emnlp.896).
24. `kvdirect` — **Verified** (arXiv:2603.19664; concurrent preprint).
25. `pg19` — **Verified** (arXiv:1911.05507; work/dataset identity matches).
26. `bm25` — **Verified** (DOI 10.1561/1500000019; journal, volume, pages match).
27. `embeddingrecycling` — **Verified** (DOI 10.18653/v1/2023.findings-eacl.145).
28. `gemfilter` — **Verified** (arXiv:2409.17422).
29. `reform` — **Verified** (arXiv:2506.01215; preprint).
30. `lloco` — **Metadata error**: arXiv:2404.07979 is real, but the archival EMNLP 2024 version exists (DOI 10.18653/v1/2024.emnlp-main.975).
31. `hunyuan` — **Verified** (official Tencent Hy3 model page; non-archival product/model citation).
32. `fusionrag` — **Verified** (arXiv:2601.12904; its primary record also exposes DOI 10.1145/3786655, but I did not establish an archival publication date from the limited check).
33. `mepic` — **Verified** (arXiv:2512.16822; preprint).
34. `longmem` — **Verified** (NeurIPS 2023 identity; DOI 10.52202/075280-3259 located).
35. `memoryllm` — **Verified** (arXiv:2402.04624).
36. `infllm` — **Verified** (arXiv:2402.04617).
37. `streamingllm` — **Verified** (ICLR 2024 identity).
38. `sempic` — **Verified** (arXiv:2607.28069; first posted after cutoff and near freeze, concurrent only).
39. `xu2024retrievallong` — **Verified** (ICLR 2024 work identity).
40. `qwen3` — **Verified** (arXiv:2505.09388).
41. `ape` — **Verified** (arXiv:2502.05431).
42. `cacheblend` — **Metadata error**: arXiv:2405.16444 is real, but the archival EuroSys 2025 version exists (DOI 10.1145/3689031.3696098).
43. `h2o` — **Verified** (NeurIPS 2023 identity; DOI 10.52202/075280-1506 located).

**Totals:** 40 Verified; 3 Metadata error; 0 Not found; 0 Unverifiable.

## Citation–claim match audit (8 load-bearing checks)

1. **RAG / retrieval recomputes model layers** (§1; `rag`, `xu2024retrievallong`) — **Match.** These works support retrieval-plus-model-processing as the raw-text endpoint; the paper correctly narrows its own \(j=0\) control.
2. **PIC/modular cache systems reuse precomputed chunk KV and repair composition** (§2; `cacheblend`, `turborag`, `cachecraft`, `epic`, `mepic`, `ape`) — **Match with nuance.** The taxonomy is broad but fair and explicitly non-ranking.
3. **ReadOnce / Embedding Recycling cache intermediate representations and adapt later layers** (§2; `readonce`, `embeddingrecycling`) — **Match.** These are genuine intermediate-representation reuse precedents.
4. **HCache checkpoints activations for restoration** (§2; `hcache`) — **Match.** HCache is correctly treated as restoration precedent, not a matched persistent-document baseline.
5. **KV-Direct reconstructs/reuses KV from residual state** (§2; `kvdirect`) — **Match, concurrent-preprint caveat.** The claim is consistent with the cited title/abstract-level contribution and is not used as priority evidence.
6. **ILRe/REFORM/GemFilter reduce tokens using intermediate/early-layer signals** (§2; `ilre`, `reform`, `gemfilter`) — **Match.** The paper correctly distinguishes these from persistent cross-query residual objects.
7. **LoRA and knowledge distillation motivate the adapter/training form** (§4; `lora`, `distillation`) — **Match.** These citations support the generic techniques, not the paper-specific symmetric top-64 KL recipe.
8. **Benchmark identities and official use** (§5; `ruler`, `babilong`, `longchat`, `longbench`, `locomo`) — **Match.** The cited works correspond to the evaluated benchmarks; protocol deviations and custom aggregation are separately disclosed.

# Novelty search and closest works

## Search summary

I ran four targeted searches, frozen at **2026-08-03**:

1. `"reusable intermediate representations transformer"` / reusable hidden states;
2. `"cached intermediate activations long context inference"` / activation checkpoint reuse;
3. `"context independent KV caching LLM"` / modular document caches;
4. `"prefix cache fusion RAG cache modular KV"` / RAG serving and composition repair.

I also checked exact-title/venue records for the main neighboring works. No targeted search surfaced a pre-2026-05-03 archival paper that clearly evaluates the same object—**one persistent residual per token at a tunable split depth, with a same-backbone \(j=0\) endpoint and an explicit quality/Read/storage frontier**. This is a scoped novelty finding, not proof of exhaustive uniqueness.

## Closest works

1. **ReadOnce Transformers (ACL 2021).** Caches reusable intermediate text representations and adapts later processing. Closest conceptual precedent for “write once, read later,” but not decoder-layer suffix replay, bounded chunk retrieval, or a tunable depth/storage frontier.
2. **Embedding Recycling (Findings EACL 2023).** Reuses intermediate embeddings across language-model computations. Strong intermediate-state precedent, but not a persistent long-context serving object with selection and amortization.
3. **HCache (EuroSys 2025).** Checkpoints activations to restore LLM serving state. Closest residual/activation systems precedent, but designed for state restoration rather than independently written document chunks and repeated query-conditioned retrieval.
4. **CacheBlend (EuroSys 2025) / EPIC (ICML 2025).** Closest serving workload: persistent per-layer KV objects with composition/position repair. They address reusable context caches directly, but store a different object and do not expose CoMem’s single-residual split-depth axis.
5. **Cartridges (arXiv 2025) / KV Packet (arXiv 2026).** Closest learned modular document objects. Both are highly relevant to novelty, but Cartridges is non-archival in the cited form and KV Packet is a concurrent preprint before freeze; neither is a valid “missing formally published pre-cutoff work” weakness under the requested rule.

## Cutoff judgment

- I found **no omitted formally published work on or before 2026-05-03** that clearly defeats the paper’s narrow novelty claim.
- Work first appearing after 2026-05-03 (e.g., Cartridges at Scale, SemPIC) is treated only as concurrent/suggested context.
- Preprints, including KV-Direct/KV Packet/Cartridges, are not counted as omission weaknesses.
- Therefore novelty is **incremental but defensible**: the component ideas are established, while the explicit tunable residual-depth frontier and its unusually controlled measurement appear distinct.

# Review-process self-check

- [x] Reviewed only the specified frozen PDF, specified source directory, and specified strict template; did not read other reviews, score history, TODO/status/report/current-paper files.
- [x] Read all 21 PDF pages twice, including all appendices.
- [x] Inspected every rendered figure and table (Figures 1–2; Tables 1–34).
- [x] Built claims C1–C9 and mapped each to minimum sufficient evidence and actual evidence.
- [x] Checked method equations and boundary cases, including \(j=0\), \(j=L\), storage arithmetic, fixed-\(k\) Read, Write amortization, positional assumptions, and decode caching.
- [x] Audited baselines, benchmark scope, metrics, sample counts, uncertainty, seeds, compute, software, artifacts, and storage tiers.
- [x] Checked at least five Abstract numbers; ten are listed above.
- [x] Completed desk audit: anonymity, A4, review style, page limit, exact Limitations, Ethics, embedded fonts, references, placeholders, and hidden manipulation.
- [x] Enumerated and audited every `main.bbl` entry; checked eight citation–claim matches.
- [x] Ran four novelty searches and compared five closest work families under the 2026-05-03 publication cutoff rule.
- [x] Every weakness contains a source/PDF anchor, an exact quote of at most 25 words, explicit problem, affected claim/norm and importance, sufficient remedy, and Major/Minor label.
- [x] Mechanically searched the frozen source for every weakness quote.
- [x] Mechanically searched for generic missing-item assertions: none are used; each absence claim above is tied to a source-verified omission and phrased specifically.
- [x] Paper A was assessed independently; no Paper B comparison or mutual-citation request is made.
