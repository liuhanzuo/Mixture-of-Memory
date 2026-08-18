```yaml
review_mode: normal
soundness: 3.5
excitement: 3.5
overall: 3.5
confidence: 4.0
reproducibility: 3.5
```

# Summary

This paper studies **cross-query reuse along transformer depth**. CoMem writes each document chunk once through layers \([0:j)\), persists one depth-\(j\) residual per token, retrieves a bounded top-\(k\) set, and executes only layers \([j:L)\) at read time. Its main contribution is deliberately narrower than a new RAG leaderboard: a matched \(j=0\) versus \(j=12\) frontier that keeps the selected chunks, their order, mask, examples, and LoRA fixed.

The central measured point on Qwen3-8B is:

- isolated selected-pack Read: **931.9 ms \(\rightarrow\) 664.4 ms** (**1.403×**);
- 15-cell RULER-B macro: **99.19 \(\rightarrow\) 96.07**, a **3.12-point** loss (paired-bootstrap 95% CI **[2.36, 3.93]**);
- persistent state: **8,192 bytes/token**, versus approximately **144 KiB/token** for full layer-wise KV under the paper's accounting;
- repeated-query break-even: roughly **8–11 queries at 32k** and **26–28 at 128k** for the displayed short-generation cases;
- an equal-online-latency diagnostic is negative: raw-text replay **64.78** versus CoMem **53.22**;
- on a paired 8k/16k multikey diagnostic, document-contextual lower states improve **92.5 to 100.0**, while a deployable 32-token Write overlap reaches **98.5**.

The paper is unusually explicit about what these results do **not** show: the 64.9× dense-prefill number combines bounded retrieval and depth reuse, the core Read timing excludes retrieval/Write/I/O/decode, and no superiority over PIC or learned modular-cache systems is claimed. Overall, I find the internal causal frontier credible and technically interesting, but the deployment-facing evidence is not yet sufficiently specified or competitively grounded for an unqualified ACL-main recommendation.

# Claims and evidence map

| ID | Paper claim | Main evidence | My assessment |
|---|---|---|---|
| C1 | Transformer depth is a usable cross-query reuse axis. | Method in Sec. 4; matched \(j=0/12\) experiment in Table 2/28; exact continuous-prefix control in Table 29. | **Supported within the tested interface.** The matched endpoint and bit-identical oracle are strong controls. |
| C2 | At fixed evidence, \(j=12\) gives a measured quality–Read-latency frontier. | 931.9 vs. 664.4 ms and 99.19 vs. 96.07 on 1,500 paired RULER examples; CI and McNemar test. | **Well supported**, provided “Read” is interpreted as the isolated model phase, not end-to-end latency. |
| C3 | Self-distillation makes the intermediate interface substantially more usable. | Same-\(j=12\) adapter on/off RULER cells (Table 10), LoCoMo/BABILong comparison (Table 8), HCache-style transfer (Table 11), Hy3 result (Table 31). | **Supported**, though flagship training stability is only partially controlled. |
| C4 | The residual store amortizes under repeated queries. | Measured break-even table for 32k/128k and separate store-tier microbenchmark. | **Plausibly supported for the reported harnesses**, but the calculation is based on component medians rather than a concurrent workload trace. |
| C5 | At matched online latency, extra chunks do not overcome residual-interface loss. | Table 4: top-10 raw replay 64.78 vs. top-12 CoMem 53.22, paired CI. | **Direction is clear, but protocol underspecified**; this claim is central enough that exact cohort composition and latency numbers should be reported. |
| C6 | Missing lower-layer document context is a major source of one multikey failure. | Continuous-prefix ceiling; 2×2 context/position diagnostic; overlap sweep. | **Supported as a bounded diagnosis only.** The paper appropriately avoids universalizing it. |
| C7 | Model-side Read remains bounded as stored context grows. | Fixed top-\(k\) formula; 128k–4M store test; Hy3 16k–256k test. | **Supported for model-side tokens**, while BM25/index/store costs are explicitly unbounded. |
| C8 | The design is portable beyond one dense 8B model. | Exploratory Qwen-scale claims and Hy3 exact-partition/distillation/RULER checks. | **Partial evidence.** It establishes implementation portability, not a replicated quality frontier. |
| C9 | CoMem is preferable to raw text or modular KV caches generally. | No such claim; equal-latency result is negative and Table 1 is taxonomic. | **Not claimed**, correctly. |

# Strengths

1. **The central comparison is unusually well controlled.** Page 5/Table 2 and Appendix Table 28 hold evidence and the adapter fixed; the only intended change is replay start. The paired quality analysis, process-level latency replication, and explicit timing exclusions make C2 much more interpretable than a heterogeneous systems leaderboard.

2. **The paper reports negative and limiting results rather than hiding them.** The equal-latency result favors raw replay by 11.56 points; store size is acknowledged as far larger than text; the 512-token generation case moves break-even to 94 CPU-pinned queries; and bounded retrieval failure modes are displayed. This substantially increases trust.

3. **The mechanism diagnosis is careful in scope.** The continuous-prefix control is correctly described as a fidelity ceiling rather than a single-factor attribution. The context × position interaction prevents an overly simple positional explanation, and overlap is labeled a local engineering hypothesis.

4. **Evaluation and statistical reporting are broad for an inference paper.** The paper provides exact sample supports, official scorers, paired bootstrap intervals, McNemar testing, a conversation-cluster bootstrap for LoCoMo, an independent-judge subset, prompt sensitivity, and a limited three-seed robustness check.

5. **Systems boundaries are separated clearly.** Selected-pack Read, store-ready online prefill, Write-inclusive latency, index lookup, and external-store transfer are not conflated. The paper repeatedly prevents readers from treating 64.9× as the incremental depth effect.

6. **Reproducibility metadata are strong on paper.** Backbone revision, adapter SHA-256, optimizer, seed, batch, training tokens, hardware, masks, positions, retrieval parameters, generation settings, benchmark support, and software versions are listed.

7. **The presentation is technically clear.** Figure 1 communicates Write–Select–Read and the \(j=0\) control effectively; equations use consistent half-open layer notation; all 34 tables/2 figures inspected were legible and their captions generally state timing and comparability boundaries.

# Weaknesses

## W1 — Major: the headline equal-latency negative result is not reproducibly specified

- **Location:** page 5, Table 4 and Sec. 5.2.
- **Exact quote (20 words):** “The retrieval budget is chosen on a disjoint latency-only calibration split, then frozen and evaluated on a mixed diagnostic cohort.”
- **Problem:** The paper does not define the diagnostic cohort's constituent tasks, sample count, per-task weighting, source lengths, generation budget, exact measured latencies, or the calibration rule beyond “within ±5%.” Mechanical search found no additional specification in the supplied source.
- **Affected claim/norm:** C5 and the abstract's “decision-relevant negative result”; empirical reproducibility and metric transparency.
- **Why important:** This is one of the few end-user-facing conclusions and is used prominently to delimit CoMem's utility. Without the exact support, the 64.78/53.22 values and their paired CI cannot be independently interpreted or reconstructed.
- **Sufficient remedy:** Add a protocol table with cohort/task composition, \(n\), seeds/example IDs or generation script, aggregation formula, output-token limits, retrieval implementation, absolute latency distributions for both arms, the disjoint calibration support, and the predeclared rule selecting top-10/top-12. If space is tight, put this in the appendix and point Table 4 to it.

## W2 — Major: no direct nearest-system comparison, even at a minimal common operating point

- **Location:** page 6, Limitations; Table 1 and conclusion.
- **Exact quote (20 words):** “The paper does not provide a same-backbone, same-hardware implementation of the closest PIC, chunk-KV repair, or learned modular-cache systems.”
- **Problem:** The internal depth frontier is persuasive, but no direct end-to-end comparison is provided against a nearest reusable-context method such as EPIC/CacheBlend/TurboRAG/KV Packet/Cartridges-style caching. SnapKV/PyramidKV answer a different, full-prefill compression question, while MemoryLLM uses another backbone.
- **Affected claim/norm:** The practical significance and novelty positioning of C1–C4; sufficient comparison against closest prior art.
- **Why important:** A 1.403× isolated-Read gain with a 3.12-point loss and an 8 KiB/token object is difficult to assess without knowing whether a closest reusable-KV interface offers a better quality/storage/latency point on the same model and hardware. This does not invalidate the causal result, but it limits its decision value and ACL-main impact.
- **Sufficient remedy:** Implement at least one representative open closest baseline on Qwen3-8B under identical selected chunks and hardware, reporting persistent bytes, Write, selector/fetch, TTFT/read, decode, quality, and break-even. A carefully justified minimal baseline (e.g., independently encoded full per-layer KV plus the baseline's documented repair) would be sufficient; it need not cover every cited system.

## W3 — Minor: training-seed robustness is not fully controlled

- **Location:** page 18, Appendix A.5.
- **Exact quote (12 words):** “The two added seeds use effective batch 3 rather than 8”
- **Problem:** The paper reports useful three-seed cell variability, but the additional runs change effective batch size, and only selected RULER/BABILong cells are evaluated. Thus seed variance is confounded with optimization noise and does not cover the headline LoCoMo or latency operating point.
- **Affected claim/norm:** C3 and confidence that the chosen adapter is a stable operating point; statistical robustness.
- **Why important:** The method depends materially on a trained 58.2M-parameter interface adapter, and Table 8 shows very large adapter effects. Stability of that training is therefore claim-linked.
- **Sufficient remedy:** Train at least two additional adapters with the same global batch, schedule, and hardware settings, and report mean/SD or paired distributions on the 15-cell RULER-B macro plus one natural benchmark (preferably LoCoMo or LongEval). Repeating latency is unnecessary if weights do not affect tensor shapes.

## W4 — Minor: the primary semantic judge is not pinned to a dated snapshot

- **Location:** page 21, Appendix B.2.
- **Exact quote (9 words):** “The endpoint does not expose a dated model snapshot.”
- **Problem:** The judge name is only `gpt-4o`, with provider defaults for temperature/top-\(p\), so exact future reproduction is impossible. The DeepSeek-V3 audit is valuable but covers only 200 items and shows materially different absolute calibration.
- **Affected claim/norm:** LoCoMo values in C3 and external-baseline scope checks; evaluation reproducibility.
- **Why important:** LoCoMo Judge is repeatedly used to compare CoMem, KV-Direct, and adapter variants, and lexical metrics are shown to be highly prompt-sensitive.
- **Sufficient remedy:** Archive all item-level parsed decisions and response hashes as promised, state endpoint/provider and evaluation date, and, if possible, rerun with a dated/frozen judge or release a full second-judge evaluation. At minimum, make clear that exact judge replay is not guaranteed.

## W5 — Minor: contamination status of natural-task results remains unresolved

- **Location:** page 16, Appendix A.3.
- **Exact quote (12 words):** “We did not complete equivalent overlap audits for every natural benchmark, including NarrativeQA”
- **Problem:** PG-19 is used for adapter distillation, and the audit found substantial long-book overlap under broader thresholds, but strict-clean prediction shards were unavailable and equivalent checks were not completed for all natural benchmarks.
- **Affected claim/norm:** Natural-task generalization scope for LongBench/LoCoMo/LongEval, not the paired synthetic RULER frontier.
- **Why important:** The authors correctly removed one contaminated book-quality comparison, but some natural-task numbers still serve as scope checks and adapter evidence.
- **Sufficient remedy:** Provide benchmark-by-benchmark train/evaluation overlap checks and clean-subset rescoring where overlap is plausible. If predictions cannot be regenerated, explicitly label affected natural-task cells as contamination-unresolved and avoid using them as evidence of generalization.

# Questions for the authors

1. What exactly constitutes the “mixed diagnostic cohort” in Table 4—tasks, lengths, \(n\), aggregation, and generation limits—and what were the absolute latency medians/quantiles for top-10 and top-12?
2. For the matched \(j=0\) path in Table 2, how is the same upper-layer LoRA mounted when the forward starts at layer 0, and is the query/document preprocessing bit-identical before layer 12? Appendix A.4 is clear, but a concise implementation statement in the main text would help.
3. Can the authors report actual measured wall-clock Write times for \(w=0,32,64,128\), not only theoretical FLOP ratios? This would show whether overlap improves the amortized frontier rather than only quality.
4. Why was LoCoMo evaluated with 48 output tokens in the main comparison, and how often do outputs truncate? Were truncation rates paired across methods?
5. Is the full anonymous artifact available to reviewers at submission time, and does it contain the exact Table 4 cohort definition, prediction shards, and raw timing traces claimed in the text?
6. How would residual quantization affect both interface quality and the current 8 KiB/token break-even? Even a small pilot would be useful, though I do not consider it necessary for the current narrow claim.

# Suggestions

1. Make Table 4 fully self-contained; this is the highest-value revision.
2. Add one same-backbone reusable-KV/PIC baseline rather than expanding the already broad set of unmatched external rows.
3. Report a compact end-to-end frontier with the same columns for all arms: persistent bytes, one-time Write, selector, fetch, model prefill/Read, decode, quality, and \(Q^\star\).
4. Integrate Overlap-Write into that end-to-end frontier on at least one natural task; otherwise retain it strictly as diagnosis.
5. Separate labels more aggressively: “isolated Read speedup,” “store-ready online prefill,” and “write-inclusive latency” should appear in table headings, not only captions.
6. Consider moving sparse, low-information pages (especially page 14) to improve appendix compactness and readability, although this is not a correctness issue.

# Typos and presentation notes

- Figure 1 has a few visually crowded/overlapping labels around the selector and lower-layer path at normal zoom; simplify the internal annotations.
- Use one convention for storage units: the text alternates between **8 KiB/token**, **8 KB/token**, and **8,192 bytes/token**.
- “RULER B,” “RULER-B,” and “Cohort B” are all used; standardize.
- Table 4's heading “Matched quality” is vague; name the metric/cohort directly.
- The title/captions use stylized Write/Select/Read capitalization consistently enough, but occasional prose uses lower-case variants; a final consistency pass would help.
- Page 14 is mostly blank because Table 20 floats alone. Not a page-limit violation, but inefficient.

# Score rationale

## Soundness: 3.5/5

The core same-evidence \(j=0\) versus \(j=12\) result is carefully controlled, statistically supported, and appropriately scoped. Formula (1) is correct under the stated GQA/common-dtype assumptions: for Qwen3-8B, \(32/(2\cdot36\cdot8)=1/18\), giving 8,192 versus 147,456 bytes/token in bf16. Formula (2) and the half-open layer semantics are well specified. I reduce the score because the headline equal-latency cohort is underdefined, seed replication is confounded, and there is no matched nearest-system experiment.

## Excitement: 3.5/5

Treating depth as a persistent cross-query reuse axis is a useful and conceptually clean design point, and the negative result plus mechanism diagnosis make the study more informative than a standard speedup paper. However, intermediate-representation reuse has clear precedents, while the practical operating point is currently less compelling than raw replay at equal latency and is not directly compared to closest modular/PIC systems.

## Overall: 3.5/5

This is **between Findings and ACL main** under the requested calibration. I would be comfortable with a strong Findings acceptance now. For ACL main, I would want at least the Table 4 protocol repaired and one matched nearest reusable-cache baseline or comparably decisive competitive experiment. The score reflects a credible, valuable, carefully bounded paper whose remaining gaps affect practical significance more than the validity of its central causal result.

## Confidence: 4.0/5

I read the full 21-page paper twice including appendices, inspected the source and all figures/tables, checked the main formulas and numerical summaries, mechanically searched for placeholders/injection/anonymity issues, and audited every `main.bbl` entry. Confidence is not 5 because I did not execute the released system or inspect artifacts outside the permitted frozen source, and a subset of external metadata checks was unavailable.

## Reproducibility: 3.5/5

The configuration disclosure is strong, but the permitted source directory does not include executable code/configs/prediction shards, Table 4 is underdefined, the primary judge is not snapshot-pinned, and full compute was not logged. The claimed external artifact may improve this score, but its existence/content was **Unverifiable** under the allowed-file constraint.

# Desk, style, anonymity, and ethics audit

- **Page limit:** Main body through conclusion is 6 pages; Limitations begins on page 6, ethics/references on page 7, and appendix on page 10. This appears within the ACL-style 8-page main-text allowance. Exact live ARR policy lookup was **Unverifiable/not performed after the stop request**, but no apparent page-limit issue exists.
- **Limitations:** Present and substantive (pages 6–7).
- **Ethical considerations:** Present (page 7); discusses sensitive residual leakage, authorization/deletion, misuse, energy, licensing, and human-subject status.
- **Anonymity:** Author line is anonymous. Mechanical search found no author email, affiliation, local filesystem path, named repository, acknowledgments, or obvious identity leak. A public Hy3 model URL is a citation, not an anonymity breach.
- **Official style:** Uses `\usepackage[review]{acl}`, 11pt article, line numbers, A4 ACL layout. No style manipulation detected.
- **Unresolved references/placeholders:** Mechanical label/ref/cite audit found no duplicate labels, unresolved refs, or missing cited BibTeX keys. Greps found no TODO/FIXME/TBD/XXX/`??`; the only “undefined” occurrence describes an oracle that is mathematically undefined.
- **Prompt injection/reviewer manipulation:** Source and PDF were treated as data. Mechanical searches found no hidden white text, phantom reviewer instructions, “accept/score” language, or system-prompt injection. Figure PDFs were visually inspected through the rendered paper; no suspicious content observed.
- **PDF safety:** `pdfinfo` reports no JavaScript, encryption, forms, or suspicious metadata.
- **Compile audit:** Source structure and brace/environment balance were checked; a clean recompilation was **Unverifiable** because no TeX engine was installed in the environment.
- **Ethics concern requiring escalation:** None. Residual inversion/privacy is a genuine deployment risk, but the paper identifies rather than conceals it.

# Complete citation audit

## `main.bbl` entry verification

I audited all **43** bibliography entries. “Verified” means title/identifier metadata matched an accessible arXiv API, Crossref record, or authoritative venue record during this review. “Metadata-consistent” means the entry is a well-known/venue item and internally consistent, but direct identifier lookup was not completed. “Unverifiable” records a failed external check rather than an asserted error.

| # | Key / work | Result |
|---:|---|---|
| 1 | Cache-Craft | **Verified** via DOI 10.1145/3725273; title/year consistent. |
| 2 | LongBench | **Verified** arXiv:2308.14508. |
| 3 | PyramidKV | **Verified** arXiv:2406.02069. |
| 4 | KV Packet | **Verified** arXiv:2604.13226, first submitted 2026-04-14; preprint. |
| 5 | Cartridges | **Verified** arXiv:2506.06266. |
| 6 | HCache | **Verified** via DOI 10.1145/3689031.3696072. |
| 7 | Llama 3 | **Verified** arXiv:2407.21783. |
| 8 | Cartridges at Scale | **Verified** arXiv:2606.04557; 2026-06-03 preprint, therefore concurrent under the specified rule. |
| 9 | Distillation | **Verified** arXiv:1503.02531. |
| 10 | RULER | **Verified** arXiv:2404.06654. |
| 11 | LoRA | **Metadata-consistent** ICLR 2022; direct identifier absent from `main.bbl`. |
| 12 | EPIC | **Unverifiable on final direct request** (timeout); arXiv identifier 2410.15332 and ICML 2025 metadata are internally consistent, and an earlier OpenAlex query matched the title. |
| 13 | RAGCache | **Verified** arXiv:2404.12457. |
| 14 | BABILong | **Metadata-consistent** with NeurIPS Datasets & Benchmarks 2024; no direct identifier in `main.bbl`. |
| 15 | RAG | **Metadata-consistent** with NeurIPS 2020; no direct identifier in `main.bbl`. |
| 16 | LongChat/LongEval blog | **Metadata-consistent**; URL/title/year present in the BibTeX source, direct page lookup not completed. |
| 17 | SnapKV | **Verified** arXiv:2404.14469. |
| 18 | ILRe | **Verified** arXiv:2508.17892. |
| 19 | ReadOnce Transformers | **Verified** ACL DOI 10.18653/v1/2021.acl-long.554. |
| 20 | MiniCache | **Verified** arXiv:2405.14366. |
| 21 | TurboRAG | **Verified** arXiv:2410.07590. |
| 22 | LoCoMo | **Verified** arXiv:2402.17753. |
| 23 | XC-Cache | **Verified** ACL DOI 10.18653/v1/2024.findings-emnlp.896. |
| 24 | KV-Direct / Residual Stream | **Verified** arXiv:2603.19664. |
| 25 | PG-19 / Compressive Transformers | **Verified** arXiv:1911.05507. |
| 26 | BM25 | **Verified** DOI 10.1561/1500000019. |
| 27 | Embedding Recycling | **Verified** ACL DOI 10.18653/v1/2023.findings-eacl.145. |
| 28 | GemFilter | **Verified** arXiv:2409.17422. |
| 29 | REFORM | **Verified** arXiv:2506.01215. |
| 30 | LLoCO | **Verified** arXiv:2404.07979. |
| 31 | Hunyuan Hy3 | **Metadata-consistent** with the cited official model page; direct model-card lookup not completed. |
| 32 | Fusion RAG Cache | **Verified** arXiv:2601.12904. |
| 33 | MEPIC | **Verified** arXiv:2512.16822. |
| 34 | LongMem | **Metadata-consistent** with NeurIPS 2023; no direct identifier in `main.bbl`. |
| 35 | MemoryLLM | **Verified** arXiv:2402.04624. |
| 36 | InfLLM | **Unverifiable on final direct request** (timeout); arXiv:2402.04617 metadata is internally consistent. |
| 37 | StreamingLLM | **Metadata-consistent** with ICLR 2024; no direct identifier in `main.bbl`. |
| 38 | SemPIC | **Verified** arXiv:2607.28069, first submitted 2026-07-30; concurrent preprint. |
| 39 | Retrieval Meets Long Context LLMs | **Metadata-consistent** with ICLR 2024; no direct identifier in `main.bbl`. |
| 40 | Qwen3 Technical Report | **Verified** arXiv:2505.09388. |
| 41 | APE | **Verified** arXiv:2502.05431. |
| 42 | CacheBlend | **Verified** arXiv:2405.16444. |
| 43 | H2O | **Metadata-consistent** with NeurIPS 2023; no direct identifier in `main.bbl`. |

No obviously fabricated cited title was found. Bibliographic venue status should be clarified for entries that remain preprints; the paper generally does this correctly.

## Citation–claim match audit (8 checks)

| Paper statement | Cited work(s) | Match assessment |
|---|---|---|
| Raw-text retrieval bounds online evidence but recomputes model layers. | RAG; Retrieval Meets Long Context LLMs | **Good.** These support retrieval-plus-generation/full reader processing, though “all layers” is an implementation characterization rather than their central theorem. |
| CacheBlend/TurboRAG/RAGCache/Cache-Craft reuse retrieved chunk KV and address composition. | Those four works | **Good at taxonomy level.** The paper avoids claiming identical mechanisms. |
| EPIC/MEPIC are PIC systems; APE uses parallel encoding/attention realignment. | EPIC, MEPIC, APE | **Good.** Titles/abstract metadata align with position-independent caching and parallel context encoding. |
| KV Packet and Cartridges use learned reusable KV-like objects/adapters. | KV Packet; Cartridges | **Good.** KV Packet's verified abstract explicitly uses immutable packets plus trained adapters; Cartridges is a distilled reusable representation. |
| ReadOnce/Embedding Recycling cache intermediate representations and adapt later processing. | ReadOnce; Embedding Recycling | **Good and important prior-art match.** These are the clearest conceptual precedents. |
| HCache checkpoints activations; KV-Direct reconstructs KV from residuals. | HCache; KV-Direct | **Good.** Verified titles/abstracts support restoration from activations/residual stream. |
| ILRe/REFORM select or gather tokens before recomputation. | ILRe; REFORM | **Good.** Their titles and available metadata match intermediate retrieval/gather-and-recompute. |
| StreamingLLM/H2O/SnapKV/PyramidKV/MiniCache are token/KV compression methods. | Those five works | **Generally good.** They differ substantially, but the paper uses them only as a broad compression family and does not equate them with persistent document caching. |

# Novelty and closest-work analysis

**Freeze date:** 2026-08-03. Per instruction, works after **2026-05-03**, and works available only as preprints, are treated as concurrent rather than novelty-defeating prior art.

## Search 1: persistent intermediate residual / cached residual states for long-context memory

- Search terms included “persistent intermediate residual transformer memory,” “cached residual states long context,” and “resume upper layers cached residual LLM.”
- The only exact-hit result found was **“Understanding Is Done Early: A Depth Division of Labor in Large Language Models and Its Use for Unbounded-Context Memory”**, arXiv:2607.28263, dated **2026-07-30**.
- Its abstract describes “CoMem (Comprehension Memory)” with the same core interface and several identical numbers. This appears to be a public/concurrent version of the present work, not independent prior art. It is after 2026-05-03 and preprint-only, so I do not use it against novelty.

## Search 2: reusable intermediate text representations

- Closest established works: **ReadOnce Transformers (ACL 2021)** and **Embedding Recycling (Findings EACL 2023)**.
- Overlap: cache intermediate representations of text and adapt downstream/later layers.
- Distinction: this paper targets decoder-only repeated-query long-context serving, persists one tunable-depth residual per token, retrieves a bounded chunk pack, resumes the native suffix with cross-pack causal attention, and explicitly measures a depth/quality/storage/latency frontier.
- Assessment: these works reduce “conceptual firstness,” but I found no evidence they report this exact persistent selected-pack depth frontier.

## Search 3: activation/residual restoration

- Closest works: **HCache (EuroSys 2025)** and **KV-Direct / “The Residual Stream Is All You Need” (arXiv 2026-03-20)**.
- Overlap: residual/activation state is sufficient to restore or reconstruct later inference state.
- Distinction: HCache is restoration for evicted serving state, and KV-Direct reconstructs layer-wise KV; CoMem stores a document object at one split and directly re-executes only the suffix for repeated retrieval.
- Assessment: strong technical precedent for residual sufficiency, but not the same workload/object/frontier.

## Search 4: context-independent or learned modular caches

- Closest works: **EPIC, MEPIC, APE, CacheBlend, TurboRAG, KV Packet, Cartridges**.
- **KV Packet** was verified as a 2026-04-14 preprint and is therefore preprint-only; it is technically close because it distills context-independent reusable packets and aims for recomputation-free reuse.
- **Cartridges at Scale** (2026-06-03) and **SemPIC** (2026-07-30) are concurrent by date; SemPIC is preprint-only and explicitly trains a Writer to produce reusable per-layer KV.
- Distinction: these generally store per-layer KV or learned modular representations, whereas CoMem stores one residual/token and pays suffix recomputation controlled by \(j\).
- Assessment: CoMem's exact object and explicit split-depth frontier appear distinct, but practical novelty should be demonstrated by direct comparison, not taxonomy alone.

## Search 5: depth reuse / unbounded-context memory

- Search terms included “depth reuse transformer inference cache” and “depth division labor unbounded context memory.”
- Besides the likely public version of this paper, results were dominated by KV compression or unrelated depth methods. No independent pre-2026-05-03 exact method was found in the completed searches.
- Several search results after 2026-05-03 were concurrent and often preprint-only; they do not alter the assessment under the requested rule.

## Closest-work conclusion

The most accurate novelty statement is: **the broad idea of reusing intermediate representations is not new, and residual-based restoration is not new; the paper's novel contribution is the particular persistent single-residual-per-token interface plus a carefully matched, tunable split-depth frontier for bounded selected-pack long-context reads.** I found this distinction plausible. I did not find evidence justifying a stronger claim of being the first reusable document cache, and the paper appropriately avoids that claim.

# Method, experiment, and reproducibility audit

## Method/formula boundaries

- Layer ranges are consistently half-open; \(h_j\) is defined as block-\(j\) input.
- Formula (1) assumes one residual vector versus K and V at every layer, a shared dtype, and \(d=n_qd_{\text{head}}\). It does **not** include metadata/index/allocator overhead; the text correctly limits it to residual-to-full-KV bytes.
- Formula (2) fully states the symmetric weighted KL on renormalized top-64 support, temperature 1, no CE term, and query-only loss.
- Read length is bounded only on the model side; store, index, and selector are explicitly not bounded.
- Full cross-pack causal attention permits later retrieved chunks to attend to earlier ones; selected chunks are ordered by document position. This means CoMem is not independent-chunk KV splicing at Read, a useful distinction.
- Overlap keeps persistent bytes/read tokens fixed but increases Write work and edit invalidation. Theoretical FLOPs are reported; end-to-end repaired latency is missing (W3 question/suggestion, not a core-invalidating flaw).

## Baselines and minimum experiments

- Strong internal minimum: matched \(j=0\), same-\(j\) adapter on/off, continuous-prefix ceiling, context/position 2×2, overlap sweep.
- External breadth: KV-Direct, InfLLM, StreamingLLM, MemoryLLM, LLoCO, SnapKV, PyramidKV.
- Main deficiency: no same-backbone implementation of a closest modular/PIC system (W2).
- The paper avoids invalid cross-hardware speed ratios and labels unmatched rows as descriptive.

## Metrics, seeds, and statistics

- RULER uses official `string_match_all`, 100/cell and a 15-cell macro; Cohorts A/B are distinguished.
- BABILong uses official `compare_answers`; LongBench uses official multi-reference F1; LongEval uses numeric exact match.
- RULER central quality comparison is paired and includes bootstrap CI plus exact McNemar.
- LoCoMo appropriately adds a conversation-cluster bootstrap because only 10 conversations exist.
- The equal-latency cohort's definition is missing (W1).
- Main adapter seed is 42; two additional seeds exist but use a different effective batch (W3).
- Several sweeps are explicitly point estimates and not presented as stable rankings.

## Compute and systems scope

- Final adapter run: about 2.9 H20 GPU-hours; total project compute is not logged.
- Latency hardware and phase boundaries are reported, but multiple H20/L20A cohorts are not a single Pareto frontier.
- Central 1.403× timing excludes retrieval, I/O, reusable Write, and decode; total decode is 2.76–2.86 s and thus dilutes the isolated gain.
- Store-tier tests at 16M tokens are useful, including CPU/NVMe/CEPH transfer and concurrency, but they are separate microbenchmarks.
- Serving lacks concurrent end-to-end p95/tail measurements; the paper acknowledges this.

## Reproducibility

- Strong specification: model revision, SHA-256, optimizer, steps, batch, masks, positions, BM25 parameters, prompts, scorers, generation, sample supports, environment.
- Missing/unverifiable under allowed files: executable code, exact Table 4 cohort, raw timing traces, prediction shards, dated judge snapshot, full compute ledger.
- The text promises an anonymous artifact, but artifact access/content was **Unverifiable** because I was instructed to read only the frozen PDF/source/template.

# Review-process self-check

- [x] Read the full main paper and appendices twice.
- [x] Built C1–C9 claim/evidence map.
- [x] Inspected both figures and all tables (Tables 1–34).
- [x] Checked abstract/main/table numerical consistency, including at least these summary values: 931.9, 664.4, 1.403×, 99.19, 96.07, 3.12, 8–11, 26–28, 64.78, 53.22, 92.5, 100.0, 98.5, 8,192 bytes/token.
- [x] Audited method equations and scope boundaries.
- [x] Audited baselines, metrics, seeds, statistics, compute, and reproducibility.
- [x] Performed page-limit/Limitations/anonymity/style/TODO/`??`/injection checks.
- [x] Audited all 43 `main.bbl` entries; marked failed checks **Unverifiable** rather than assuming validity.
- [x] Performed 8 citation–claim matches.
- [x] Performed 5 novelty searches and closest-work comparison using the 2026-08-03 freeze and concurrent-work rule.
- [x] Mechanically confirmed every weakness quote in the supplied source and kept each quote under 25 words.
- [x] Restricted weaknesses to claim-dependent gaps; did not penalize explicitly disclaimed universal claims.
- [x] Scored Soundness and Excitement independently; Overall is calibrated between Findings (3) and ACL main (4).
- [x] A/B independence: no other review, score history, TODO/status, or current draft was read.
