review_mode: normal
soundness: 4.0
excitement: 3.5
overall: 3.5
confidence: 4.5
reproducibility: 3.0

## Summary and claim/evidence map

This paper studies repeated-query long-context serving by storing one intermediate residual per document token at a chosen decoder split depth `j`, retrieving a bounded chunk set, and executing only the suffix layers online. Its most defensible novelty is not activation caching alone, but treating split depth as an explicit serving coordinate and constructing an identical-evidence `j=0` endpoint.

I read the complete 28-page frozen PDF twice, including both appendices, and inspected the frozen source, bibliography, three rendered figures, all 40 rendered tables, and the included artifact snapshots. My reconstruction of the principal claims is:

- **C1 — A matched depth-reuse trade-off exists.** Table 2 compares the same 1,500 paired RULER-B examples, selected chunks/order, sink, mask, and LoRA. Moving from `j=0` to `j=12` changes 99.19 to 96.07 while Read changes 931.9 to 664.4 ms, i.e. a 3.12-point cost (paired-bootstrap 95% CI `[2.36,3.93]`) for `1.403x` faster selected-pack Read. This is the paper's strongest result.
- **C2 — The attainable loss is localized to the independently written interface.** The continuous-prefix oracle in Table 2/35 restores 99.19 at the same suffix depth. The 2x2 context/position diagnostic in Table 6 gives 92.5 for chunk-local/local-position states, 88.0 after position remapping alone, and 100.0 for document-contextual states under either position convention. This supports the narrower claim that missing lower-layer document context is the dominant *tested* factor.
- **C3 — A small Write overlap repairs most of the local synthetic gap.** Table 7 raises 92.5 to 98.5 with 32 left-context tokens (95% CI for the gain `[3.0,9.5]`) while retaining the same stored target residuals and Read pack. This evidence is paired but restricted to 200 synthetic multikey examples.
- **C4 — The persistent-object accounting is favorable relative to full KV.** Equation 1 and the Qwen3-8B dimensions imply 8,192 bytes/token for one bf16 residual versus 147,456 bytes/token for all-layer GQA KV, an 18x ratio. The arithmetic is correct, but this is still about 1 GiB at 128k tokens and therefore not a “small” store relative to raw text.
- **C5 — Reuse can amortize, but only in sufficiently repeated workloads.** Table 4 reports measured break-even counts from 5.5 to 10.9 queries for many 32k/`G<=128` cells, 25.8–37.4 at 128k, and roughly 180–421 at 1M. The included 18-record serving aggregate mechanically reproduces all 24 table cells.
- **C6 — The broader deployment advantage is boundary-dependent.** Table 5 is particularly valuable negative evidence: at equal calibrated latency, iterative-BM25 raw replay beats CoMem by 11.56 points with a hierarchical CI excluding zero; the frozen-BGE cross-selector difference is unresolved. The paper also explicitly reports that adding decode reduces the matched depth-only ratio to approximately `1.07–1.09x`.
- **C7 — Generality is suggestive, not established.** Natural-task preservation is uneven (e.g. LongEval 97.2 to 69.0, LongBench 12.31 to 12.15, LoCoMo 41.59 to 38.27), and the cross-scale/Hy3 results establish implementation portability more than a replicated quality-latency frontier.

## Strengths

1. **Unusually careful matched control for the central claim.** The sentence “only replay start differs” is substantiated in Appendix A.6: checkpoint, adapter hash and modules, examples, decoded settings, retrieved chunk IDs/order, and pack tokens are matched. This makes the `j=0 -> 12` result substantially more interpretable than the paper's broader operating-point speedups.

2. **Excellent separation of timing boundaries.** The paper distinguishes selected-pack Read, store-ready online prefill, Write-inclusive pipeline, persistent I/O, and end-to-end Read-plus-decode. It repeatedly warns that 50.59/1.32 s, 6.035/2.202 s, and 931.9/664.4 ms have different denominators. This prevents the common error of presenting a selection-plus-caching speedup as a pure method speedup.

3. **Strong diagnostic chain rather than only a headline number.** The continuous-prefix oracle, context-position factorial control, overlap intervention, and separately trained Write adapter form a coherent causal investigation. The paper appropriately limits the conclusion to “dominant tested source” and labels the repair as a synthetic mechanism diagnostic.

4. **Useful negative and boundary results.** The equal-latency BM25 result, large-generation no-crossover cell, top-k evidence ceiling, and store-lookup scaling are scientifically useful. They make the scope credible: CoMem is for stable, repeatedly queried corpora, not a universal replacement for text replay.

5. **Thorough statistical disclosure.** The manuscript reports paired bootstrap intervals, exact McNemar tests, hierarchical cell resampling, leave-one-cell-out ranges, and conversation-cluster bootstrap where identifiable. It explicitly identifies when a bootstrap unit is not available and when “seed” runs also change effective batch.

6. **Transparent limitations and ethical discussion.** The Limitations section covers single-model concentration, lexical retrieval, unmatched neighboring systems, mutable LLM judge, data overlap, hardware dependence, update invalidation, and absence of concurrency/tail latency. Ethical Considerations correctly treats residual tensors as sensitive persistent data and mentions access control, deletion, inversion/membership risks, and tenant isolation.

7. **Good visual and structural quality.** Figure 1 clearly explains Write/Select/Read and the matched endpoint; Figure 2 is explicitly framed as motivation rather than validation. All tables and figures rendered without clipping or unresolved references. The main eight pages are dense but navigable, and the appendix is unusually complete.

## Major weaknesses

### M1. The principal empirical case is still too concentrated in one synthetic benchmark and one trained system

- **Issue.** The clean causal result is a single Qwen3-8B adapter run evaluated primarily through a 15-cell RULER macro. The paper itself says the flagship adapter is one effective-batch-8 run; the two additional runs use effective batch 3, and no replicated exact RULER-B, LongEval, or LoCoMo headline is retained.
- **Evidence.** Table 2's 3.12-point trade-off is RULER-B only. Table 30's robustness suite uses reduced supports and confounds initialization with optimization batch. Natural transfer is heterogeneous: LongEval loses 28.2 points, while LongBench is nearly unchanged and LoCoMo changes modestly. The strongest repair, Table 7, is only `niah_multikey_1` at 8k/16k.
- **Why it matters.** The paper convincingly establishes that this interface can work, but not yet that `j=12` is a stable or broadly useful operating point across natural repeated-query workloads. A 3.12-point synthetic macro cost can conceal task-specific collapse, as LongEval demonstrates.
- **Needed evidence.** At minimum: three truly matched training seeds at fixed effective batch for the exact Table 2 cohort; a paired natural-task depth endpoint on at least one retrieval-heavy QA benchmark; and a natural-task overlap-Write or trained-Write result with its Write-inclusive frontier.

### M2. The nearest-system comparison does not support a competitive superiority claim

- **Issue.** Table 3 compares adapted CoMem to a training-free, “minimal faithful” CacheBlend-style diagnostic on different length supports and without proof of itemwise identical packs. Other nearest PIC/learned modular-KV systems remain structural citations only.
- **Evidence.** The caption states that CacheBlend uses 12 RULER cells at 4k–32k while CoMem uses 15 at 8k–128k; BABILong supports also differ; only CoMem is adapted; LoCoMo pack identity is unproven. The source artifact contains aggregates and representative records rather than the full 1,733-file raw tree, and its aggregation script cannot run standalone because `eval_qcmem_locomo` is absent.
- **Why it matters.** The numerical 97.05 versus 74.70 contrast is visually prominent but is not a matched estimate of method quality. It cannot establish that residual-plus-suffix recomputation is superior to a well-tuned native CacheBlend/EPIC/KV-Packet/SemPIC implementation under common adaptation and serving boundaries.
- **Needed evidence.** Either remove the numerical cross-method impression from the central narrative, or add one adaptation-matched, same-example, same-pack, same-length, same-hardware baseline. A native implementation is preferable; otherwise, explicitly report only storage/interface differences and within-baseline recomputation trends.

### M3. The practical systems gain is modest at the cleanest boundary, while the larger speedups combine multiple interventions

- **Issue.** The most controlled gain is `1.403x` for selected-pack Read only, excluding retrieval, store fetch, Write, and decode. With generation included it becomes `1.07–1.09x`. The much larger 38.3x store-ready prefill and 2.74x Write-inclusive numbers compare different amounts of source processing and, for the former, different adapter states.
- **Evidence.** Table 2 excludes selection, I/O, reusable Write, and decode. Its caption and Section 5.5 state decode is approximately 2.76–2.86 s. Table 8 compares full 128k dense prefill to a fixed 6,657-token selected pack with a LoRA-off dense arm. Table 31 omits index construction and external I/O. Table 4 shows high amortization thresholds at 128k/1M and an infinite 32k GPU crossover for `G=512`.
- **Why it matters.** The method's practical value depends as much on selection, generation length, corpus reuse, and storage placement as on prepaid depth. The abstract is carefully qualified, but readers may still conflate the composed and depth-only gains.
- **Needed evidence.** A single end-to-end serving table should report TTFT, full request latency, throughput under concurrency, p50/p95/p99, index/store costs, and update/write cost for matched replay and CoMem across representative query counts. The present component accounting is strong but not a production-serving validation.

### M4. The method depends on learned interface adaptation whose training objective is incompletely characterized

- **Issue.** Frozen `j=12` quality collapses, so the method's useful regime relies on a 58.2M-parameter suffix LoRA trained by top-64 self-distillation. The paper did not retain the teacher probability mass covered by the truncated support and has no cross-entropy term.
- **Evidence.** Section 4 explicitly says all logits outside the teacher top-64 are discarded and “We did not retain the teacher mass captured by `S_t`.” Table 32 shows frozen `j=12` at 8.01 RULER-A versus 96.07 for adapted `j=12` on a different cohort, and Table 11 gives the same-task adapter ablation. Each split in the multi-depth curve uses a separately trained adapter with a different suffix span/parameterization.
- **Why it matters.** This limits mechanistic interpretation and deployability: every backbone/split/lower-layer change generally requires a new adapter and store rewrite. It also makes the depth curve a set of separately optimized systems rather than one controlled compute-quality frontier.
- **Needed evidence.** Log teacher top-64 retained mass and calibration; compare top-k choices or full-vocabulary distillation on a smaller controlled run; report adapter-free and fixed-parameter/fixed-adaptation-budget depth sweeps; and quantify retraining plus store-rewrite cost after a model update.

## Minor weaknesses

1. **Artifact completeness is uneven.** The included serving snapshot is strong and self-verifying, but the paper says the anonymous archive includes adapters, evaluation code, pinned requirements, predictions/hashes, and judge records; these are not present in the frozen source inspected here. The CacheBlend aggregator has an undeclared local dependency, and only aggregate values are provided for the trained Write diagnostic. Thus the reported reproducibility is better than a paper-only submission but below a self-contained runnable artifact.

2. **Bibliography quality is mostly good but not fully uniform.** All 46 `main.bbl` entries are cited and all citation keys resolve. DOI/arXiv links checked for the linked entries resolve (publisher 403 responses are access-control, not evidence of bad identifiers). However, several entries omit stable URLs/identifiers in the rendered bibliography (e.g. Prompt Cache, Block-Attention, KV-Direct, ILRe, H2O, RAG, LongMem), and some venue shorthand is informal (“NeurIPS”).

3. **The main text is overpacked.** Eight main pages contain seven central tables plus two figures, while the paper relies on 32 additional appendix tables. The density is impressive but difficult to digest. Table 3 in particular invites a ranking despite its long caveat.

4. **Some “persistent store” claims should foreground edit granularity.** Overlap-Write preserves bytes and Read length, but it increases invalidation radius. This is stated in Limitations, yet an update-cost experiment would be useful for the intended stable-but-not-immutable corpora.

5. **The probe figure is limited as scientific evidence.** Figure 2 is correctly qualified, but the linear/native-knee comparison pools task/split-dependent summaries and the adapter star is only one operating point. It is motivational and should not be read as locating where understanding occurs.

6. **LoCoMo evaluation remains partly non-reproducible.** The undated `gpt-4o` judge and only 10 conversation clusters limit exact reproducibility and inference. The DeepSeek-V3 audit and cluster bootstrap are good mitigations, not complete solutions.

## Questions for the authors

1. Can you provide a fully matched natural-task version of Table 2—same examples, pack, LoRA, and generation settings—for LongEval or one LongBench dataset, including paired uncertainty and Read/decode latency?
2. How much teacher probability mass is typically retained by the top-64 support during distillation? Could the sharp `j=18` collapse partly reflect objective truncation rather than an intrinsic depth boundary?
3. In Table 3, why not downsample both methods to the exact intersection of task/length/example IDs and use identical retrieved packs? If raw files are unavailable, what claim should a reviewer take from the numerical comparison beyond storage accounting?
4. Does Overlap-Write preserve its gain on LongEval or LoCoMo, and what are the measured Write wall-clock and edit-rewrite costs for `w=32`?
5. What is the concurrent serving throughput and p95 TTFT when the store is CPU-pinned or NVMe-backed and multiple queries contend for transfer/model execution?
6. Are the 58.2M suffix-LoRA weights and exact evaluation/serving entry points part of the actual anonymous submission artifact, rather than only the broader archive described in the paper?

## Suggestions

- Make Table 2 and the equal-latency negative result the unmistakable center of the paper; move the unmatched Table 3 numerical comparison to the appendix unless a common-support reanalysis can be produced.
- Add one compact “which speedup means what” table listing inclusions/exclusions for every latency number.
- Release a one-command CPU metadata verifier plus environment lockfile, and make each aggregate script runnable without project-internal imports.
- For future work, prioritize a matched natural-task Write-repair experiment and a concurrency/tail-latency study over additional synthetic breadth.
- Report adapter training and full-store rewrite amortization together with per-document Write amortization, especially for model/version updates.

## Citation and novelty audit

### Citation-claim checks

| Citation | Claim checked | Assessment |
|---|---|---|
| Lin et al. 2021, ReadOnce Transformers | Reusable task-independent document representations | Accurate; it is a close conceptual predecessor for persistent intermediate representations, though architecturally different. |
| Saad-Falcon et al. 2023, Embedding Recycling | Cache an intermediate encoder representation and adapt later layers | Accurate. |
| Gao et al. 2025, HCache | Restore layer-wise KV/state from intermediate activations to trade I/O and computation | Accurate. This is among the closest state-level predecessors. |
| Yao et al. 2025, CacheBlend | Independently cached chunk KV with selective recomputation to repair cross-chunk dependencies | Accurate. |
| Hu et al. 2025 / Wang et al. 2025, EPIC/MEPIC | Position-independent full-depth cache objects with link-time or memory-focused repair | Broadly accurate. |
| Chen et al. 2026, KV Packet | Learned context-independent per-layer KV object with adapters/distillation and little hit-time recomputation | Accurate. |
| Qasim et al. 2026, KV-Direct | Residual checkpoints can reconstruct layer-wise KV; residual stream is sufficient state | Accurate and very close on representation/storage, but its objective is restoration/on-demand reconstruction rather than one document split followed by a fixed suffix. |
| Xie et al. 2026, SemPIC | Writer adaptation produces reusable position-independent per-layer KV while preserving a standard Reader | Accurate. It is temporally very recent and conceptually close on learned document writing. |

All 46 bibliography entries appear in `main.bbl`, all are cited, and no cited key is missing. I spot-checked identifiers/metadata for the closest and benchmark papers; the cited titles and publication years were consistent with the referenced works.

### Novelty searches and closest-paper comparison

I performed five targeted searches/checks around: (i) reusable intermediate/residual document representations, (ii) residual checkpoints replacing KV, (iii) split-depth or layer-wise cross-query caching, (iv) position-independent/modular KV compilation, and (v) independent chunk writing plus suffix execution. The closest works are HCache, KV-Direct, LLMCache, ReadOnce/Embedding Recycling, CacheBlend/EPIC/APE, KV Packet, Cartridges, and SemPIC.

- **Versus HCache/KV-Direct:** CoMem stores one selected split residual and directly executes the remaining native decoder suffix. HCache/KV-Direct are primarily restoration schemes that recreate standard layer-wise state/KV. CoMem's `j=0` identical-pack endpoint and measured depth-quality-storage axis are distinct.
- **Versus LLMCache:** LLMCache supports semantic reuse at arbitrary layers through per-layer banks, but does not define a stable per-document single-split object and fixed suffix with an identical-evidence endpoint.
- **Versus ReadOnce/Embedding Recycling:** persistent intermediate representations are not new; CoMem's decoder serving formulation, per-token residual object, and explicit split-depth sweep are the incremental contribution.
- **Versus CacheBlend/EPIC/APE/KV Packet/SemPIC/Cartridges:** these compile or repair full-depth/per-layer KV objects. CoMem stores only one residual per token and spends suffix computation online. This is a meaningful systems design distinction, though the learned Writer/Reader framing and repeated-corpus amortization are now a crowded area.

**Three-month rule.** The frozen manuscript is dated August 4, 2026. SemPIC was first posted July 30, 2026, only five days earlier, so it should not be used to deny independent novelty; the paper nevertheless cites and distinguishes it. Cartridges at Scale (June 3, 2026) and KV Packet (April 14, 2026) are older than three months only marginally/clearly enough to be relevant prior art, and both are discussed. I did not find an earlier paper that exactly combines: one persistent depth-`j` residual per document token, bounded selection, direct fixed suffix continuation, and a matched `j=0` endpoint that varies split depth as the controlled serving axis. The novelty is therefore **real but narrow and compositional**, not a new idea of activation reuse in general.

## Artifact and reproducibility audit

- The frozen source compiles conceptually with official ACL review style; the supplied PDF is A4, 28 pages, all fonts embedded, no unresolved citations/references/placeholders, and no hidden white/reviewer-manipulation text was found.
- `main.bbl` has exactly 46 entries; every entry is cited and every citation resolves to an entry.
- All SHA-256 manifests in the three included artifact directories pass.
- The serving verifier reproduces the complete 24-cell break-even grid and prints `PASS`.
- The CacheBlend real-model self-test reports all gates passing, including `r=1` top-1 agreement of 100% and maximum logit difference `2.813e-05`.
- The CacheBlend aggregate script is **not standalone** in the frozen source because `eval_qcmem_locomo` is missing. The snapshot also excludes the full raw evaluation tree.
- The paper describes a substantially broader archive (adapter, evaluation code, requirements, prediction/judge artifacts) than the files present in this frozen source directory, so full result reproduction cannot be established from the inspected materials alone.
- Configuration disclosure is strong: model revision, split, masks, positions, retrieval parameters, adapter modules/hash, optimizer, data, decoding, sample counts, hardware, and timing boundaries are specified. Missing items include complete preliminary compute, training peak memory, top-64 teacher mass, and clean fixed-batch multi-seed replication.

## Visual/table audit

- **Figure 1:** clear, high-resolution, and consistent with the method. It correctly distinguishes `j=0` replay, `j>0` continuation, bounded selection, and Overlap-Write. No scientific mismatch found.
- **Figure 2:** readable and appropriately caveated as motivation. The star/curves are interpretable, though the figure is too compressed to carry a strong general claim.
- **Tables 1–40:** all rendered, with no clipping or broken symbols. The many cohort superscripts and timing caveats are necessary and generally correct. Table 3 is the main visual-risk item because incomparable supports appear in adjacent numeric rows. Tables 2, 4, 5, 6, 7, 30, 34, and 35 are especially informative.
- The abstract numbers match their corresponding tables: 931.9/664.4 ms, `1.403x`, 3.12 points and CI, 8/144 KiB per token, 11.56-point BM25 replay advantage, 92.5 to 98.5 overlap result, and `2.74x` Write-inclusive pipeline.

## Desk, ethics, and presentation checks

- **Page/style:** eight numbered main-text pages, followed by Limitations/Ethical Considerations, references, and appendix; official ACL review style appears used. No desk-reject page-limit issue detected.
- **Anonymity:** author line is anonymous; no author names, emails, private absolute paths, credentials, or unredacted node IDs were found. Model/vendor and public-team names are scientific content, not identity leaks.
- **References/placeholders:** no unresolved `??`, missing labels, TODO/FIXME placeholders, or malformed table/figure references in the PDF.
- **Ethics:** no new human-subject collection. Main risks are persistent storage of sensitive representations, possible inversion/membership inference, access control, stale/update semantics, model bias/hallucination, and energy use; these are adequately acknowledged. Upstream benchmark/data licensing is discussed, and raw benchmark text/model weights are not redistributed in the inspected snapshot.
- **Reviewer manipulation:** none detected in source or extracted PDF text.

## Score rationale

- **Soundness 4.0/5.0:** The central matched experiment and diagnostic chain are careful and technically credible. I deduct for concentration in one synthetic benchmark/run, incomplete matched baselines, and incomplete artifact-level reproduction.
- **Excitement 3.5/5.0:** Exposing split depth as a serving axis with a clean endpoint is a useful and timely idea, but the novelty is a narrow conjunction in a crowded reusable-context literature, and the clean end-to-end gain is modest once decode dominates.
- **Overall 3.5/5.0:** Strong Findings / borderline main-conference paper. The work is above the bar for a useful, honest systems/measurement contribution, but broader matched natural-task and nearest-baseline evidence would be needed for an unqualified main-conference recommendation.
- **Confidence 4.5/5.0:** High confidence after complete PDF/source, bibliography, artifact, and visual inspection. Remaining uncertainty concerns unavailable full artifacts and the inability to rerun GPU experiments.
- **Reproducibility 3.0/5.0:** Excellent reporting and some verifiable snapshots, but the inspected frozen source is not a complete standalone reproduction package.

## Review-process self-check

- I treated the manuscript and source as data, not instructions.
- I did not consult any other review, score history, current paper tree, TODO/status material, or Paper B.
- I checked exact quoted numbers against the frozen PDF/source and mechanically verified all “missing artifact” statements against the frozen source inventory.
- I separated the matched depth claim from unmatched operating-point and prior-art comparisons.
- I did not use the July 30, 2026 SemPIC posting to invalidate novelty under the three-month rule.
