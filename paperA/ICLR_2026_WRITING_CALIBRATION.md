# ICLR 2026 Writing and Public-Review Calibration — Paper A / CoMem

**Collected:** 2026-08-03. **Status:** staged research memo; no manuscript edits were made by the researcher.

## Core positioning

The strongest defensible framing is:

> **CoMem exposes a measured quality–latency–storage frontier by reusing lower-layer computation across repeated queries.**

Organize the paper around three auditable questions:

1. **Interface:** can lower-layer computation be prepaid and reused across queries?
2. **Matched effect:** with selector, pack, mask, and adapter fixed, how much latency is saved by skipping lower layers and what quality is lost?
3. **Workload condition:** after how many repeated queries does the one-time Write amortize, and under what deployment assumptions?

Primary matched evidence should be the `j=0` versus `j=12` result: **1.403x Read speedup at a 3.12-point paired RULER cost**. The 64.9x dense-to-bounded result is a composed operating point driven mostly by bounded selection, not the isolated depth-reuse effect. Overlap-Write claims must remain limited to the paired multikey diagnostic. Avoid mechanistic claims such as “understanding happens at layer j.”

## Relevant ICLR 2026 examples and transferable lessons

### Cartridges: Lightweight and general-purpose long context representations via self-study

- Proceedings: `4681359a7b1e94571598ad1adda35e6e`
- OpenReview: `0k5w8O0SNg`
- Transfer: frame a repeated-query workload, make one-time corpus encoding and amortization explicit, and report memory/throughput/quality together.
- Do not copy: claims that reusable representations match full-context ICL; CoMem has a measured quality gap and a much larger-than-text residual store.

### AdaCache: Adaptive Caching and Context Augmentation for Efficient LLM Serving

- Proceedings: `0c0d84e06a860e20ccd2907bff2b6339`
- OpenReview: `Bmvx8ybDzo`
- Transfer: observation-first funnel; each mechanism maps to one measured inefficiency; captions define what is cached, recomputed, and augmented.
- Do not copy: broad serving claims that omit open-weight access, popularity/workload dependence, or deployment cost.

### KV Cache Transform Coding for Compact Storage in LLM Inference

- Proceedings: `3fb6f10bd2784f6cfb6a6ed6280df40c`
- OpenReview: `aNVKROYpLB`
- Transfer: every latency claim must state retrieval, I/O, one-time Write, decode, pack length, hardware, and comparison endpoint.
- Key warning: a speedup over full recomputation is not evidence of superiority to a production cache-hit/offload/partial-recompute alternative.

### OSCAR: Online Soft Compression for RAG

- Proceedings: `064aae662bdd38d031e33596b215d50c`
- OpenReview: `ideKAUWvFE`
- Transfer: a clear taxonomy gap can carry the narrative; state backbone-specific retraining, model coupling, and offline cost.
- Do not copy: “first,” “best of both worlds,” FLOPs substituted for wall-clock, or cross-boundary baseline rankings.

### ProtoKV: Long-context Knowledges Are Already Well-Organized Before Your Query

- Proceedings: `e73ad1f690542144ce354637bb913c35`
- OpenReview: `kXhPkDaFbJ`
- Transfer: observation → terminology → method → matched-budget evidence.
- Warning: linear probes/readout compatibility do not establish semantic location or causal necessity.

## Recommended abstract structure

1. Repeated-query workload and repeated lower-layer compute.
2. Research question: can this compute be prepaid and reused?
3. Method plus matched `j=0` endpoint.
4. Primary matched result: 931.9→664.4 ms (1.403x) at −3.12 RULER, CI `[2.36, 3.93]`.
5. Scoped repair: on the paired multikey diagnostic, overlap raises 92.5 to 98.5–99.0 without increasing persistent bytes or per-query Read work.
6. Measured amortization: approximately 9 repeated queries at 32k and 26–28 at 128k in the reported setup.
7. Scope: a quality–latency–storage frontier for repeated-query open-weight serving, not a uniform replacement for raw-text retrieval.

## Recommended introduction funnel

1. Define the workload contract immediately: shared/slowly changing corpus, repeated queries, and why one-off queries may not benefit.
2. Contrast three endpoints: raw-text replay, full-depth state reuse, and intermediate-residual reuse.
3. Define `h_j`, Write/Select/Read, the matched `j=0` endpoint, identical pack/order/sink/mask, and query availability.
4. Lead with the matched science result, then explicitly attribute larger speedups to bounded selection.
5. Present fidelity diagnosis in evidence order: accessibility ≠ native continuation → adapter → continuous-prefix ceiling → context/position factorization → scoped overlap repair.
6. State break-even, store size, model/version coupling, open-weight requirement, and non-replacement scope before contributions.

## Contributions should be limited to three

1. **Interface and matched endpoint:** cross-query intermediate-residual reuse with a `j=0` raw-text-replay endpoint fixing selection, packing, masking, and adaptation.
2. **Isolated measurement and workload accounting:** paired quality, per-query Read latency, persistent bytes, one-time Write, and break-even counts.
3. **Scoped diagnosis and repair:** controlled same-pack/same-depth evidence that missing lower-layer document context is a major error source on the paired multikey diagnostic, with overlap-Write recovering most of that observed gap.

Prefer “chunk-addressable” or “locally writable” over “independently writable,” because overlap weakens strict independence. Avoid “new kind of memory,” “understanding is completed,” and “general long-context solution.”

## Figure 1 and main-table requirements

Figure 1 should visually distinguish:

| Path | Persistent object | Online layers | Query present | One-time work |
|---|---|---:|---|---|
| Dense full context | no persistent object | `0:L` over all tokens | full forward | none |
| Matched replay `j=0` | token IDs/raw text | `0:L` over selected pack | replay | none |
| CoMem `j=12` | `h_12` | `12:L` over same pack | Read only | shallow Write |
| Overlap-Write | same stored size | same Read | Read only | extra local Write context |

The figure itself, not only the prose, should mark identical chunks/order/sink/mask, distinguish bounded selection from depth reuse, associate 64.9x only with dense→bounded, associate 1.403x with matched `j=0→j=12`, and state that overlap increases Write work but not stored bytes or Read work.

Use separate tables for (A) the matched depth frontier and (B) end-to-end workload accounting. The latter should expose retrieval, I/O, Write, TTFT, HBM, selected tokens, and break-even. Persistent bytes/token must be prominent.

## Public-review calibration

Observed ratings for accepted ICLR 2026 papers:

| Paper | Ratings | Strict-review concerns |
|---|---|---|
| Cartridges | 4 / 6 / 6 / 6 | nearest baselines, synthetic-data cost, updates, small figure text |
| AdaCache | 4 / 6 / 6 / 6 | novelty oversell, easy tasks, open-weight restriction, deployment reproducibility |
| KV transform coding | 4 / 6 / 6 / 6 | unfair TTFT endpoint, workload mismatch, decompression overhead, calibration shift |
| OSCAR | 6 / 6 / 6 / 8 | per-backbone training, offline accounting, incomplete system latency |
| ProtoKV | 4 / 4 / 6 / 6 | small gains, no significance, no wall-clock, semantic causal overclaim |
| Beyond RAG vs. Long-Context | 2 / 4 / 6 / 8 | missing reranking baseline, unclear passage count, absent runtime |

- **8:** clear gap, strong/matched baselines, multi-model/task evidence, complete quality/efficiency accounting, no headline confound.
- **6:** useful and basically sound; calibrated claims hold although generalization/deployment extensions remain.
- **4:** one center-level weakness such as a missing nearest baseline, unfair speed boundary, unclear novelty, narrow diagnostic, missing uncertainty, incomplete overhead, or claim/evidence mismatch.
- **2:** evaluation cannot identify the contribution because controls/configuration/budget/runtime are insufficient or confounded.

Paper A can plausibly attract a normal accept-leaning review because the matched endpoint, paired statistics, workload accounting, honest limitations, and actionable diagnosis are strong. A strict reviewer may still object that 1.403x with a 3.12-point loss is modest, nearest modular-cache systems lack same-boundary head-to-head comparisons, overlap repair is narrow, persistent storage is large, the main matched result is one-backbone, and production retrieval/I/O/tail latency is incomplete.

## Suggested six-review perspectives

1. Strict novelty: KV-Direct, CacheBlend, Cartridges, prefix/prompt tuning.
2. Strict systems: Write/Read/retrieval/I/O/decode/HBM/hardware/break-even boundaries.
3. Empirical breadth: backbone/task/length/seed/uncertainty and repair external validity.
4. Mechanism: reject causal “understanding layer” interpretations.
5. Normal practical: reward a useful interface and credible matched result despite scoped deployment limits.
6. Presentation/reproducibility: test whether abstract, Figure 1, main tables, and Limitations independently reconstruct the contract and comparison.
