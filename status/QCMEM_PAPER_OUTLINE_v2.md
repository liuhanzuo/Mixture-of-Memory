> QCMem Paper A 大纲 v2（2026-07-13 workflow wzpqfc9jr 产出，SparseForge 结构）。Paper B/C 已判定分离。

# QCMem — "Depth-Partitioned Retrieval Readout" — NEW Outline (SparseForge structure)

Target file order (rename/reorder `\input` in `main.tex`):
`00_abstract → 01_introduction → 02_related → 03_motivation → 04_methodology → 05_experiments → 06_conclusion → 07_limitations`

Legend per block: **REUSE** = paste existing tex/prose ~verbatim · **REWRITE** = existing material, restructured/expanded · **NEW** = write from scratch.

---

## 1. Abstract — **REUSE** (`00_abstract.tex`, verbatim)

One dense paragraph, already SparseForge-shaped. It already contains: problem (long-context bottlenecked twice), gap (existing methods keep full-depth KV / all tokens → costly and collapse past the window), our method in one line (cache a single mid-layer hidden per chunk, retrieve top-k, recompute only upper layers), headline result vs SOTA (128k: full-ctx→0/OOM while QCMem RULER 100 / LongEval 0.98, 7.8× prefill speedup, constant ~6.7k-tok / ~18 GB), and the "understand-then-generate division of labor" thesis + "consistent across model families" (Qwen3-8B, Llama, Hunyuan-80L-MoE).
- Minor edit: append the "consistent across model families" clause explicitly (Qwen3-8B + Hunyuan Hy3 80-layer MoE) to match SparseForge's closing beat.

---

## 2. Introduction — **REUSE + light edit** (`01_introduction.tex`)

Keep the existing prose flow:
- "Long-context inference is bottlenecked twice" (O(L²) prefill + O(L) KV) and the **collapse-not-degrade** framing: Qwen3-8B native window 40,960 (131,072 is un-activated YaRN), 128k → RULER = 0.
- Existing depth-partition Write/Read itemize bullets.
- The **j-knob spectrum** (j=0 = exact full RAG, |Δlogit|<1e-4; j=L = closed-book).

### Contribution bullets (list 4–5 — **REWRITE** the current 3 into 5)
1. **A layer-partial reformulation of long-context memory.** We recast reuse from the *token axis* (recompute a subset of tokens at full depth, à la RAG/KV-reuse) to the *layer axis*: cache one depth-j residual hidden per 512-token chunk (~1/72 of full KV), retrieve, and recompute only layers[j:L]. Read cost and memory are **constant in context length** (~6.7k tokens / ~17–18 GB).
2. **The only method usable past the backbone's extrapolation limit.** At 128k, full-context and full-KV baselines collapse to 0 / OOM (packed length exceeds the native window → RoPE breaks); QCMem holds RULER niah 100 / LongEval 0.98, with a 7.83× prefill speedup and crossover at ~16k.
3. **A depth-partition that is provably exact and cross-family.** j=0 read equals the full forward to max|Δlogit|<1e-4 on dense 8B and **=0.0 exactly** on a 597 GB Hunyuan 80-layer MoE (through MoE routing). The cacheable sweet spot lands at ~0.375–0.40·L on three backbones (Qwen-8B, Llama, Hy3).
4. **Self-distillation that deepens the cacheable point with no synthetic data.** A LoRA (r=32, α=64) KL self-distill on plain PG19 (4000 steps) pushes usable cache depth j≤9 → j≥12 on 8B and cuts the Hy3 16k LM-tax +49.6% → +7.8% (top1 0.773→0.871).
5. **A retrieval readout that scales to multi-hop.** BM25 top-k feeds the pack; an iterative, forward-free `iter_bm25` follows reference chains and lifts variable-tracking from 20–28 → 92–97 (non-degrading with length), while cross-chunk full-attention recompute is shown load-bearing for multi-fact disambiguation.

---

## 3. Related Work — **REWRITE / EXPAND** (`02_related.tex`, from `05_related.tex` + Inventory C clusters)

Promote to position 2. Expand the current single "delta note" paragraph into 4 bold-lead topic clusters, ordered **broad → closest** so the gap lands last. (Inventory C ordering + verified arxiv ids.)

**Fixed-budget KV-cache compression.** Token-axis eviction and per-layer budgets that keep a fraction of the *full-depth* KV. Cites: StreamingLLM (2309.17453), H2O (2306.14048), SnapKV (2404.14469), PyramidKV (2406.02069), LCKV (2405.10637), CompressKV (2606.24467, token-axis retrieval-head eviction — flagged **orthogonal**), Activation Beacon (2401.03462). *Gap seed:* all retain full-depth KV; QCMem caches one mid-depth hidden. (StreamingLLM is our same-budget quantitative rival — forward-ref the head-to-head.)

**Retrieval-augmented long context.** Motivates our READ path. Cites: Landmark (2305.16300), FoT/LongLLaMA (2307.03170), CacheBlend (2405.16444), RAGCache (2404.12457), InfLLM (2402.04617). *Gap seed:* all store & attend full-layer KV; QCMem retrieves via **external, non-differentiable, one-shot BM25** and *recomputes* rather than attends.

**Fixed-size / architectural long-context memory.** Cites: MemoryLLM (2402.04624, also a babilong rival), RecursiveSummarizing (2308.15022), YOCO (2405.05254, 1M niah anchor, cited-not-reproduced). *Gap seed:* memory is baked into parameters or requires retraining the architecture; QCMem uses a stock backbone + self-distilled LoRA and an external, non-evolving hidden store.

**Caching intermediate activations and recomputing upper layers (closest).** Cites: HCache (2410.05004, post-hoc/no-retrieval/no-training), KV-Direct = "The Residual Stream Is All You Need" (2603.19664, full-depth recompute, exact but 128k=0), KV-CAT (2605.05971, "compressibility must be induced in training" — **must cite, cannot claim as ours**), and StagesOfInference (2406.19384) as the depth-semantics rationale for using depth as a partition. **Gap stated here (hand-off to contributions):** QCMem's unique intersection = layer-*partial* recompute (layers[j:] only) + BM25 retrieval + constant read length + self-distillation that pushes the cacheable depth deeper + j as a RAG↔closed-book knob. KV-CAT compresses on the token/slot axis via continued pretraining; QCMem partitions on the depth axis and works zero-training.

> Keep the "efficiency / ultra-long-context, not accuracy-SOTA" framing OUT of Related Work (put it in Limitations).

---

## 4. Motivation — **NEW SECTION** (promote `04_mechanism.tex` prose here; add Hy3 + selector evidence)

This is the load-bearing device. Formal insights, each = sentence + evidence + the exact design choice it forces. Tag each as *[motivation intuition]* vs *[confirmatory result → also appears in Experiments]* per the reorder judgment call.

**(I1) Semantic content saturates at mid-depth, so a mid-layer hidden is a sufficient statistic for "understanding" — it is what should be cached.** *[intuition + probing]*
- Evidence: semantic-task probes peak mid-stack and fall at the top while next-token ability forms only at the top (Qwen + Llama, draft §3.1); causal truncation — the first ~4–8 layers (depth 0.12–0.22) already reach 95% of full downstream semantics; RTE mid > top by +0.06–0.10; zero-train j-sweep RULER niah 16k: j≤9 = 100. **NEW 3rd backbone:** Hy3 80-layer MoE j-sweep shows a flat plateau j=8..32 (top1≈0.90, KL≈0.21) → cacheable ceiling ~0.4·L.
- → Design: **split the transformer at depth j; WRITE = cache the depth-j residual hidden** (h_j ∈ ℝ^{512×d}, ~1/(2L) of full KV) and recompute only layers[j:L].

**(I2) The top layers do the query-conditioned generation, so their outputs cannot be cached query-blind — cache BELOW the split, never a top layer.** *[intuition + causal proof]*
- Evidence: top-prepay divergence — chunk-local (query-blind) h_{L−b} diverges from the query-aware value (b=12→cos 0.865/relL2 11.3; b=8→0.902/5.8; b=6→0.916/4.2). Direction-B negative control: babilong qa5 (12,0)=61/50 collapses to (12,6)=29/20 and (6,12)=9/10. RTE top verbalizer 0.79 > mid probe 0.62 (top trades separability for generation). Hy3 "fidelity smile": resuming too *deep* (past j≈36) rises again because deep encoding has committed to a query-blind continuation.
- → Design: **READ packs [sink; selected h_j; query h_j] and recomputes the query-conditioned upper layers** with the query present — the cache point sits below the generation band. Pull the top-prepay divergence table INTO this section as the causal proof.

**(I3) The cacheable depth is a bounded knee: j is a RAG↔closed-book knob whose sweet spot is ~0.375–0.40·L.** *[intuition + j-sweep]*
- Evidence: zero-train j-sweep j≤9=100, j12 cliff→14, j18=0; babilong qa5 oracle j0/j6/j12/j18 = 69/50/39/16; self-distillation pushes the ceiling j9→j12+; Hy3 converged split-j≈32/80 = 0.40·L matches 8B j≈12/32 = 0.375·L. Endpoints: j=0 = full RAG, exact-equals full-ctx (|Δlogit|<1e-4); j=L = closed-book.
- → Design: **choose j≈0.375–0.40·L; expose j as a knob; self-distill to push usable j deeper.**

**(I4) Caching deeper costs monotonically more LM quality, so j is a *trade-off*, not a tax-minimum — and the cache point can be made more compressible by design.** *[intuition; SEVERABLE — see verdict]*
- Evidence: semantic-bottleneck layer sweep (1B, dim512): L1/3/6/9/12 = +4.2/5.8/6.0/9.5/9.6% LM-tax (monotone in depth); dim sweep @L6: d1024/512/256 = +4.5/5.9/8.5%; 3B tax < 1B (cheaper on bigger models).
- → Design: **j=12 as the compromise (LM-tax wants shallow; cacheable semantics want deep)**; optionally a from-scratch low-rank funnel at the cache layer (kept as a clearly-labeled *severable* subsection; see §"separate-paper verdict").
- Frame with the **prediction-not-reconstruction** IB view (keep I(X;Y_pred), drop I(X;X)) — explains why the cache is compressible and why token-reconstruction aux was the wrong objective.

**(I5) At long range, needles/facts drown without retrieval — so retrieval + full-attention recompute is required, and the long-range bottleneck is retrieval quality, not compression fidelity.** *[intuition + confirmatory]*
- Evidence: HCache (no retrieval) collapses niah_single 8k=34→16k=2, LongEval all-0, babilong qa1 all-crash, BUT short redundant-fact survives (qa5 2k=68/8k=52) → failure = retrieval-missing, not model incapacity. StreamingLLM (keeps recent, same ~6.7k budget) 128k=4 vs QCMem 100 (25×). **NEW:** 4-selector oracle = 100% on NIAH given the right chunk → retrieval is the bottleneck.
- → Design: **BM25 top-k selector feeds the READ pack**; motivates cross-chunk full attention (see I7).

**(I6) Multi-hop reference chains need *iterative* retrieval — a single pass drowns the chain.** *[intuition + NEW result]*
- Evidence: RULER VT single-pass @16k = bm25 27.6 / reader_attn 60.2 / oracle 9.2, all collapse <23 @32k; `iter_bm25` (follow the variable chain, forward-free, no training) lifts VT to 8k=95.2 / 16k=93.8 / 32k=96.8 (hop4, tk16), non-degrading with length. Scope: only reference chains — multikey iter 92 < single 97–99 (slight hurt), single-needle 100=100 (no effect).
- → Design: **`iter_bm25` selector variant** in Methodology, applied to variable-tracking / multi-hop; scoped so it is not applied to independent-needle tasks.

**(I7) Cross-chunk full-attention recompute over the pack is load-bearing for multi-fact disambiguation.** *[design choice; quantitative proof is an ablation in Experiments]*
- Evidence (cite forward to ablation): niah_multikey std 88/92 vs block-diagonal 44/40 (Δ+44/+52); babilong qa2 36/24 vs 16/12, qa5 68/65 vs 49/53; single-needle 100=100.
- → Design: **READ recomputes upper layers with FULL attention over the entire pack, not block-diagonal KV-reuse.**

*(Motivation carries I1–I7 as intuition; the confirmatory numbers for I5/I6/I7 also appear as tables in Experiments.)*

---

## 5. Methodology — **REUSE + ADD** (`04_methodology.tex`, from `02_method.tex`)

Now placed AFTER Motivation. Reuse the four labeled paragraphs verbatim, add the selector module.

- **Formal notation.** L layers, split depth j, chunk size c=512, h_j ∈ ℝ^{c×d}, storage ~1/(2L) of full KV, pack length = sink + k·c + query ≈ 6.7k for k=12/c=512. **REUSE.**
- **WRITE.** Encode each chunk through layers[0:j], store depth-j residual hidden. **REUSE.**
- **READ.** Retrieve top-k chunk hiddens; pack [sink; selected h_j; query h_j]; recompute layers[j:L] with full cross-pack attention; constant compute/memory. **REUSE**, add explicit note that full attention (not block-diagonal) is used (I7).
- **Selector (EXPAND to a first-class subsection).** (a) **BM25** one-shot top-k (default). (b) **`iter_bm25`** — forward-free chain-following for multi-hop/variable-tracking (I6): retrieve → read variable name → re-retrieve, no training. (c) Note reader_attn / recency / oracle exist as comparison selectors (→ selector ablation in Experiments). **NEW prose** (from QCMEM_SELECTOR_COMPARISON.md + commit 20f7ffe).
- **The j knob (RAG↔closed-book).** j=0 exact full-ctx; j=L closed-book; operating point j≈0.375–0.40·L. **REUSE.**
- **Self-distillation training.** j=12 student, LoRA r=32/α=64, 4000 steps, 8-GPU DDP, plain PG19 KL, no synthetic long-context data; pushes usable cache depth j≤9→j≥12. **REUSE.**
- **(Optional, severable) Compressibility-by-design pretrain.** One clearly-labeled paragraph pointing to the low-rank funnel results, explicitly stating QCMem needs *no* special pretraining (see verdict). **NEW / DEFER.**

---

## 6. Experiments — **REUSE + FOLD + NEW tables** (`05_experiments.tex`, from `03_experiments.tex` + `07_hunyuan_moe.tex`)

Bold-lead setup paragraphs (SparseForge style), then tables in this order.

**Models.** Qwen3-8B (36 layers, j=12, native window 40,960); Llama (probing); Hunyuan Hy3 80-layer MoE (597 GB, native 262k, j≈32). **REWRITE** (merge from §03 setup + §07).
**Baselines.** KV-Direct = full-context reference (resume_j=0, no retrieval, exact to |Δlogit|<1e-4, so KV-Direct crash = full-ctx crash); HCache (post-hoc, no retrieval, no training); MemoryLLM (fixed memory bank); StreamingLLM (same fixed budget, keeps recent). **REUSE** (baseline defs from Inventory B).
**Eval setting / metrics.** RULER string_match_all, BABILong official compare_answers, LongBench F1, LoCoMo; per-task best-k; **explicitly annotate n=50 head-to-head vs n=100 elsewhere**, and the three 128k-multikey values (98 best-tk8 / 96 fixed-tk12 / 84 length-adaptive-tk24) each carry their topk label. **REUSE + add calibration note (铁律2).**
**Hardware.** median-of-3 timing, chunk 512. **REUSE.**

### Results tables — ORDER + labels
1. **Five-benchmark overview** → `tab:overview` (**REUSE** `tab_overview.tex`). Headline.
2. **RULER NIAH head-to-head** vs KV-Direct/HCache/MemoryLLM + read-length block → `tab:h2h` (**REUSE** `tab_h2h.tex`).
3. **StreamingLLM equal-budget comparison** (90/42/16/4 vs 100/100/100/100) → `tab:slm` (**NEW tex** — currently prose-only in §03; tabulate).
4. **Super-context scaling to 256k** (niah_single/multikey/var-track, QCMem vs full-ctx, 128k=0, 256k=OOM) → `tab:scaling` (**REUSE** `tab_scaling.tex`).
5. **Efficiency** (prefill speedup 0.97×→7.83×, mem 20→89 GB vs flat 17–18 GB, crossover ~16k) → `tab:eff` (**REUSE** `tab_eff.tex`).
6. **Selector ablation** (bm25/recency/reader_attn/oracle, oracle=100% on NIAH → retrieval is the bottleneck) → `tab:selector` (**NEW tex** from QCMEM_SELECTOR_COMPARISON.md).
7. **iter_bm25 variable-tracking improvement** (VT 20–28 → 92–97, non-degrading; scope table) → `tab:itervt` (**NEW tex**, commit 20f7ffe). Headline sub-result.
8. **Cross-chunk attention ablation** (std 88/92 vs block-diag 44/40) → `tab:crosschunk` (**NEW tex** from draft §3.5).
9. **chunk_size ablation** (128/256/512/1024 → read_len/peak/prefill/decode/multikey; sweet spot 512) → `tab:chunk` (**NEW tex** from draft §2.5).
10. **Per-benchmark real long-doc/dialogue QA:**
    - LongEval → `tab:longeval` (**NEW tex**, draft §2.7).
    - BABILong three-way + SOTA-range (incl. MemoryLLM calibrated baseline) → `tab:babilong` (**NEW tex**, draft §2.4 + §3 SOTA table).
    - LongBench F1 → `tab:longbench` (**NEW tex**, draft §2.8).
    - LoCoMo overall + per-category → `tab:locomo` (**NEW tex**, draft §2.9).
11. **Generality: Hunyuan Hy3 80-layer MoE** (fold `07_hunyuan_moe.tex` as the final subsection): exact depth-partition through MoE routing (max|Δlogit|=0.0); fidelity smile / split-j≈0.40·L; self-distillation tax +49.6%→+7.8%; constant read ~4.3–4.6k to 256k, zero OOM → `tab:hy3distill` + `tab:hy3ruler` (**REUSE**). Fix the `07` cross-ref `\S\ref{sec:results}`.

Analysis paragraphs (**REUSE** from §03/§07): three RULER findings; scaling-past-extrapolation; efficiency crossover; per-benchmark reads; ablation reads.

---

## 7. Conclusion — **NEW** (write from scratch; ~1 paragraph, thesis-sentence heading)

Heading = the takeaway thesis: **"Long-context memory is a layer-partition problem, not a token-budget problem."**
Synthesize abstract + contributions: understanding saturates mid-stack and is cacheable in one hidden; generation lives in the top layers and must be recomputed query-conditioned; retrieval (iterative for chains) supplies the right chunks; the result is the only method that survives past the extrapolation window at constant cost, verified exact across three model families including an 80-layer MoE. Close by naming the knob (j as a RAG↔closed-book dial) and the path forward (resumed-band KV for faster decode).

---

## 8. Limitations — **REWRITE** (`07_limitations.tex`, from `06_limitations.tex`; update per new results)

Keep + update:
- Efficiency / ultra-long-context method, **not** precision-SOTA: within the extrapolation window (≤64k) full-ctx ≥ QCMem.
- Watershed shifts with backbone (~2× native window): Llama-3 (native 8k) full-ctx crashes 16k=0; Qwen holds to 64k.
- **UPDATE:** the old "var-track weak (64k=21)" limitation is now largely **retired by `iter_bm25`** (VT 92–97) — re-state as "single-pass retrieval weak on multi-hop; iterative retrieval closes most of the gap; decode-time cost of iteration still to be characterized."
- BABILong qa1/qa2 (single/double-fact localization) favor KV-Direct in-window.
- **Calibration honesty:** head-to-head n=50 (vs n=100 elsewhere), sparse topk grid, three 128k-multikey values by convention.
- Slow decode (recompute layers[j:] each step, ~2.4s; cs1024 5.5s) — resumed-band KV future work.
- The "compressibility" claim is feature-axis (only need to store ONE layer), not "shallow hiddens are more compressible" (that was falsified: shallow j6 int4 err 0.84).
- LoCoMo cat5 adversarial weak (needs refusal).

---

## (1) Content → Section mapping table

| Content asset | Source | Target section | Action |
|---|---|---|---|
| Abstract paragraph | `00_abstract.tex` | Abstract | REUSE |
| Bottlenecked-twice / j-spectrum / bullets | `01_introduction.tex` | Introduction | REUSE + add 2 bullets |
| Baseline delta note | `05_related.tex` | Related Work | REWRITE into 4 clusters |
| 16-work cluster paragraphs + KV-CAT/CompressKV/StagesOfInference | Inventory C | Related Work | NEW prose (verified ids) |
| Division-of-labor probing (§3.1) | `04_mechanism.tex` | Motivation I1 | REUSE (moved) |
| Top-prepay divergence + Direction-B | Inventory D / RUN_REGISTRY | Motivation I2 | REUSE + tabulate |
| Zero-train + oracle j-sweep | `04_mechanism` / draft §3.2 | Motivation I3 | REUSE |
| Semantic-bottleneck layer/dim sweep | draft §3.4 | Motivation I4 | REUSE (severable) |
| HCache-no-retrieval / SLM / selector-oracle | draft §2 / selector doc | Motivation I5 | REUSE + NEW selector |
| iter_bm25 VT | commit 20f7ffe | Motivation I6 + Exp `tab:itervt` | NEW |
| Cross-chunk ablation | draft §3.5 | Motivation I7 + Exp `tab:crosschunk` | REUSE + NEW tex |
| Write/Read/j-knob/Training | `02_method.tex` | Methodology | REUSE |
| Selector + iter_bm25 module | selector doc | Methodology | NEW prose |
| overview/h2h/scaling/eff | `tab_*.tex` | Experiments | REUSE |
| SLM / chunk / crosschunk / longeval / babilong / longbench / locomo / selector | draft §2–3 markdown | Experiments | NEW tex tables |
| Hunyuan Hy3 (prose + 2 tables) | `07_hunyuan_moe.tex` + `tab_hy3_*` | Experiments (final subsec) | REUSE (fold in) |
| — | — | Conclusion | NEW |
| Limitations + iter_bm25 update | `06_limitations.tex` | Limitations | REWRITE |

## (2) CUT or DEFER

- **DEFER to companion paper (do NOT include as a claim):** minimal-architecture / prune-heal (Qwen 14-layer 12.75 vs 11.11; 1B inheritance −1.7/depth −4.1; Hy-MT2-30B prune-heal frontier). Different research direction (depth pruning), budget-confounded 8B armB-vs-scratch14 7.5× gap must NOT be presented as an inheritance claim. At most a one-sentence pointer in Motivation I1.
- **DEFER to a labeled severable subsection (Methodology/Appendix):** the full from-scratch low-rank-funnel pretrain narrative + 1B/3B ΔNLL model-size curve (§3.3). Keep only the layer/dim LM-tax numbers in Motivation I4 as the "j is a trade-off" evidence.
- **MOVE to Appendix:** full topk-sweep grids, per-category LoCoMo breakdown detail, self-test |Δlogit| tables, RTE probe details.
- **CUT:** the "compressibility" framing as a *motivation* (falsified) — replace everywhere with "only one layer is stored" (layer-axis saving).

## (3) OPEN GAPS / numbers still needed before submission

1. **Real attribution for placeholder cites** (currently "Anonymous"): HCache 2410.05004, KV-Direct 2603.19664, KV-CAT 2605.05971, CompressKV 2606.24467 — confirm real authors or keep as arXiv-only.
2. **n=50 → n=100** upgrade on the RULER head-to-head (currently annotated caveat).
3. **StreamingLLM & Direction-B numbers** are prose-only → must be put into `tab:slm` / motivation table.
4. **MemoryLLM on LongEval/LongBench/LoCoMo** missing (LongEval driver preds=0 bug) — either backfill or state "same-class comparison limited to RULER/BABILong."
5. **HCache 64k-multikey / babilong low-length** backfill (some cells "—").
6. **iter_bm25 decode-time cost** not yet measured — needed to honestly scope the retired var-track limitation.
7. Confirm the single canonical 128k-multikey value to headline (recommend best-tk8=98 with the 96/84 variants footnoted).
8. KV-CAT (2605.05971) citation-graph / novelty-collision double-check before claiming any "train-for-compressibility" language.

## (4) Recommended figure list (currently ZERO figures)

- **Fig 1 (teaser):** Token-partial vs layer-partial schematic — WRITE (split at j, cache h_j through [0:j]) and READ (retrieve top-k h_j → pack [sink; sel; query] → recompute [j:L]); annotate "read length constant regardless of context."
- **Fig 2 (Motivation I1):** Depth-vs-ability curves — semantic-probe accuracy (peaks mid) and next-token ability (forms at top) across Qwen / Llama / Hy3-80L on one normalized-depth x-axis.
- **Fig 3 (Motivation I3):** The j-sweep knee / "fidelity smile" — accuracy (or KL/top1) vs split depth j/L for 8B and Hy3-80L, marking the ~0.375–0.40·L sweet spot and the j=0/j=L endpoints.
- **Fig 4 (headline):** Accuracy vs context length 8k–256k — QCMem flat ~100 vs full-ctx/KV-Direct collapse to 0 / OOM, with the constant read-length and constant-memory lines on a twin axis.
- **Fig 5 (optional, efficiency):** Prefill speedup (→7.83×) and peak memory (flat 18 GB vs 89 GB) vs context length, crossover marked at ~16k.

---

## Separate-paper verdict (explicit recommendation)

**Semantic-bottleneck pretrain (I4 / draft §3.3–3.4): keep only the motivation slice in THIS paper; defer the full contribution to a companion paper (or a clearly-labeled severable Methodology/Appendix subsection).**
Reasoning: QCMem's headline is that it works **zero-training on stock backbones** (self-distill LoRA is a light add-on, not a from-scratch requirement). The from-scratch low-rank-funnel pretrain (1B/3B layer/dim sweeps, ΔNLL model-size curve) is logically severable and, if intermixed, misleads reviewers into thinking QCMem *needs* special pretraining. So: include the layer/dim **LM-tax numbers** as evidence for I4 ("deeper cache = more tax → j is a trade-off"), but present the full "make the cache point compressible by design" story as an optional, explicitly-flagged subsection or push it to a follow-up.

**Minimal-architecture / prune-heal material: SEPARATE PAPER.**
Reasoning: it is a different research direction (depth pruning / model-efficiency, 2403.17887-adjacent), not long-context memory. The honest 1B causal decomposition shows the depth effect (−4.1) dominates inheritance (−1.7), so it does not cleanly prove "top layers are redundant," and the 8B armB-vs-scratch14 gap is budget-confounded. It shares only the depth-partition *premise* with QCMem. Include at most a one-sentence pointer ("the same understanding-saturates-mid-depth premise also enables architectural depth pruning; see companion work") in Motivation I1 — nothing more.