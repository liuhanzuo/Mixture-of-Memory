# A02 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU spent. Adjudication only — this file runs nothing and
launches nothing.**

Purpose: close the blocker `proposal/ready_queue.py:504-506` actually trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`) and answer the five
collision families named for A02 in
`proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:27` (rating: **不足**
— insufficient: internal motivation and experiment design only, no Related Work;
priority **2 of 7** in that audit's fill order).

---

## 0. What A02 claims *now* — and why that matters for this document

**This is the single most important section, because the audit's rating (2026-08-08)
predates three A02 gates, and the proposal's headline died in all three.** Writing
Related Work against `PROPOSAL.md`'s original thesis would defend a claim A02 no
longer makes.

`PROPOSAL.md` still says **"ACTIVE；近期最高 ROI 的模型/系统实验"** and its "核心成功条件"
asks for `≥2pp` over Read-only on LongEval/multikey and `≥1.5pp` on the LoCoMo judge.
**`STATUS.json` supersedes it.** Verbatim from `STATUS.json:3`:

```
"status": "CLOSED_NO_THESIS_DIAGNOSTIC_ASSETS_RETAINED"
```

What the three gates measured (all in-repo, all re-derivable from the named evidence
JSONs):

| gate | date | finding |
|---|---|---|
| `phase1_c1_vs_c2_gate` | 2026-08-09 | **Own kill clause FIRED.** No protocol-clean benchmark shows CoMem (C2) significantly better in aggregate. LoCoMo judge with an open-weight judge: `+1.76 pp CI [-0.35,+3.78]`, crosses 0. |
| `storage_readcompute_reframe_gate` | 2026-08-09 | **Storage form DEAD on arithmetic**: `h12` is **2048× larger** than raw text (8192 vs 4 bytes/token, exact and constant in L). Read-compute survives weakly at **1.03–1.37×** per-query vs a matched-pack text-RAG control, `N* = 8–226` queries to repay the Write. |
| `depth_vs_retrieval_quality_gate` | 2026-08-10 | **Phase-1's own quality evidence RE-ATTRIBUTED**: 54.9–78.6 % of the BABILong qa1/qa2 C1→C2 movement is the **retrieval** axis, not mid-layer read (recall@12 = 22.9–63.2 % there). |
| `read_tax_ruler_gate` | 2026-08-12 | On the retrieval-**closed** slice, **A0 (no adapter at all) is the best arm** at 99.75. Read tax by depth: `j≤9` ≈ free → `j=12` **−9 pp** → `j=18` **−79 pp**. Depth survives exact capacity matching (A2 vs A6, both **72,744,960** params: −8.75 pp). |

**Therefore the claim this document adjudicates is not "CoMem beats RAG".** It is the
one thing `STATUS.json:232` records as surviving, restricted to what A02 still owns
after the methodological finding was reassigned to B11:

> **A02's live claim.** In a mid-depth reusable-memory read (cache `h_j`, resume the
> frozen upper stack from layer *j*), the **accuracy cost of the depth knob** is a
> *cliff, not a slope* — free to `j≈9`, **−9 pp at `j=12`**, **−79 pp at `j=18`** —
> measured on a **retrieval-closed** slice with the retrieval pack held byte-identical
> across arms, and **the cliff is depth, not adapter capacity** (exact match at
> 72,744,960 trainable params). The **optimal `j=0` adapter is the identity** (measured:
> 0/400 correctness flips), so the deployed `j=0 → j=12` step already *is* the
> matched-quality depth contrast.

Everything below asks: **is that claim preempted?** Not: is mid-layer hidden-state
memory novel (it is not, and A02 must never claim it is).

> ⚠️ **Explicitly NOT adjudicated here.** The BABILong misorder / scorer-format
> finding was **reassigned to `B11-generative-scorer-format-fragility`**
> (`STATUS.json:365-370`: "A02 keeps provenance, not ownership"). Its novelty is
> checked in B11, not here. Do not let A02 borrow support from it.

## 0.1 Standing rules this adjudication obeys

**Rule 1 — overlap is not preemption.**
`memory/prior-work-differentiate-dont-abandon.md` (user 2026-08-07, verbatim:
「别因1-2个类似工作就放弃方向」): the bar for preemption is **完全相同 / 抄袭**
(identical / plagiarism), *not* overlap. Work within **2–3 months is concurrent** and
cannot preempt. When a similar paper is found, the correct output is a
**differentiation + follow-up-that-fixes-a-defect**, never a death notice.
The 2026-08-12 strengthening applies too: **a novelty audit may output a
citation-obligation list, and may NOT output a scope ceiling** ("must be narrowed",
"worth at most a short paper"). Note the asymmetry that makes this rule safe here:
A02 is **already dead by its own kill gate**, so nothing in this file can be
misread as literature-assisted killing. The audit's only job is attribution.

**Rule 2 — venue verification is per-family, and "unverified" ≠ "unpublished".**
Two authorities, per `memory/venue-verify-must-use-openreview-2026.md` and
`memory/venue-verify-acl-family-needs-anthology.md`:

| family | authority used in this pass |
|---|---|
| ICLR / NeurIPS / ICML / TMLR (+ their workshops) | **OpenReview `venueid`**, with `Camera_Ready_Revision` in `invitations` as the accept signal |
| ACL / EMNLP / NAACL / EACL **incl. Findings** | **ACL Anthology** (page fetched, title compared) **+ DBLP** record |
| everything else | DBLP record `type` field: `Conference and Workshop Papers` / `Journal Articles` = real record; `Informal and Other Publications` + `venue = CoRR` = **preprint only** |

`arXiv-only` below means **"I could not verify a peer-reviewed venue from this node"**,
not "it has no venue". Recorded as an honest gap in §5, never smoothed over.

---

## 1. Named closest collisions

### 1.1 Family: reusable prefix / KV / hidden-state memory

| paper | year | verified venue | what it does | precise difference from A02's live claim |
|---|---|---|---|---|
| **Prompt Cache: Modular Attention Reuse for Low-Latency Inference** | 2024 | **MLSys 2024** (DBLP `conf/mlsys/GimCLSK024`; CoRR 2023 `abs-2311-04934`) | precomputes and reuses attention states for reusable prompt modules across requests | reuses **full-depth** KV. There is no depth knob, therefore **no depth-tax curve is even definable**. A02's entire dependent variable does not exist in this paper. |
| **Block-Attention for Efficient RAG** (OpenReview title: *Block-Attention for Efficient Prefilling*) | 2024 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision` present; DBLP CoRR `abs-2409-15355`). ⚠️ **title differs between arXiv and the venue record** — cite carefully | blockwise independent prefill of retrieved passages so their KV is reusable; fine-tunes to recover the quality lost by dropping cross-block attention | this is the **chunk-local-encoding tax at full depth**, and it is *exactly* the tax A02 does not measure. Block-Attention repairs by training; A02 measures how the tax moves **with cache depth `j`** at a fixed pack. Their fix is A02's `w=32/w=128` overlap arms' family, which the gates left unpursued. |
| **CacheBlend: Fast LLM Serving for RAG with Cached Knowledge Fusion** | 2025 | **EuroSys 2025** (DBLP `conf/eurosys/YaoLLRCZD0J25`; CoRR `abs-2405-16444`) | reuses precomputed KV of multiple chunks and selectively recomputes a small subset of tokens to restore cross-chunk attention | same object (reusable chunk KV, repaired), but the repair axis is **which tokens to recompute at full depth**, not **at which depth to resume**. CacheBlend also has no accuracy-vs-depth curve. |
| **EPIC: Efficient Position-Independent Context Caching for Serving LLMs** | 2024/25 | **ICML 2025 Poster** (OpenReview `venueid = ICML.cc/2025/Conference`, `Camera_Ready_Revision`; DBLP CoRR `abs-2410-15332`) | position-independent chunk KV caching with static+selective recompute | same as CacheBlend on the axis question. Position-independence is orthogonal to depth. |
| **RAGCache: Efficient Knowledge Caching for RAG** | 2026 | **ACM Trans. Comput. Syst. 2026** (DBLP `journals/tocs/JinZJLLLJ26`; CoRR `abs-2404-12457`) | multilevel KV cache hierarchy for RAG, GPU/host tiering, reuse-aware replacement | a **systems/tiering** paper. It is the correct citation for "reuse-aware admission", and A02's `N*` crossover analysis is downstream of this literature, not novel against it. |
| **Adaptive KV Cache Reuse for Fast Long-Context LLM Serving** | 2026-05 | **arXiv-only** (`2605.24022`) | adaptively decides when reused prefix KV is safe | concurrent (< 3 months) and about **whether to reuse**, not **at what depth**. |
| **Cache-Craft: Managing Chunk-Caches for Efficient RAG** | 2025-02 | **arXiv-only** (`2502.15734`) from this node | chunk-cache management / selective recompute for RAG | same family as CacheBlend; venue unverified from here (§5). |
| **Cartridges: Lightweight and general-purpose long context representations via self-study** | 2025/26 | **ICLR 2026 Poster** (OpenReview `venueid = ICLR.cc/2026/Conference`, `Camera_Ready_Revision`; earlier ICML 2025 ES-FoMo-III Oral; DBLP CoRR `abs-2506-06266`) | trains a small reusable artifact (KV/params) per corpus offline, amortised across queries | **the closest thing to A02's Write-amortisation framing**, and it beats A02 on the axis A02 pitched (it makes the amortised artifact *good*, not just cheap). A02 must cite it and must not claim novelty for "amortise an offline Write across many queries". It still carries **no depth axis**. |
| **LLoCO: Learning Long Contexts Offline** | 2024 | **EMNLP 2024** (DBLP `conf/emnlp/TanLPWZK0P24`; CoRR `abs-2404-07979`) | offline context compression + LoRA finetune, retrieve-then-read on the compressed representation | same amortisation frame, full depth. |
| **MemoryLLM: Towards Self-Updatable LLMs** | 2024 | **ICML 2024** (DBLP `conf/icml/WangGCJLYYLLYSM24`; CoRR `abs-2402-04624`) | self-updating latent memory pool inside the model | memory as *parameters/latents updated by the model*, not a cache resumed at depth `j`. Cited as the memory-LM anchor. |
| **Landmark Attention** | 2023 | **ICML 2023 Workshop ES-FoMO, Poster** (OpenReview `venueid = ICML.cc/2023/Workshop/ES-FoMO`; DBLP CoRR `abs-2305-16300`). **Workshop, not main conference** | landmark tokens for random access to distant blocks | retrieval-into-attention, full depth. |
| **Memorizing Transformers** | 2022 | **ICLR 2022** (DBLP `conf/iclr/WuRHS22`) | kNN lookup into a non-differentiable memory of past (k,v) | full-depth external KV memory; the canonical anchor for "external memory of key/value states". |
| **Focused Transformer (LongLLaMA)** | 2023 | **NeurIPS 2023** (DBLP `conf/nips/TworkowskiSPWMM23`) | contrastive training to fix the distraction issue in kNN memory | trains the *retrieval* interface, not the depth interface. |
| **Compressive Transformer** | 2020 | **ICLR 2020** (DBLP `conf/iclr/RaePJHL20`) | compresses old activations into a coarser memory | the historical anchor for compressed activation memory. |
| **Leave No Context Behind (Infini-attention)** | 2024 | **arXiv-only** (DBLP `journals/corr/abs-2404-07143`, CoRR 2024; OpenReview shows only the CoRR mirror). ⚠️ widely cited but **I found no peer-reviewed record** | compressive linear-attention memory with a fixed-size state | fixed-size recurrent state, no cached-depth axis. |
| **KBLaM: Knowledge Base augmented Language Model** | 2025 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision`) | encodes KB triples into continuous key-value pairs injected via rectangular attention | injects at all layers; no depth-of-cache question. |

**Family verdict:** dense and mature. **A02 must never claim novelty for caching an
intermediate/attention state and reusing it across queries.** But no paper in this
family varies *the depth at which the cache is taken* and reports an accuracy curve
against it, which is A02's dependent variable.

### 1.2 Family: activation / KV compression and *selective decompression* — the audit's flagged direct collision

| paper | year | verified venue | what it does | precise difference from A02's live claim |
|---|---|---|---|---|
| **SeDeM: Selective Decompression of Hidden-State Memories for Long-Context Question Answering** | **2026-07-31** | **arXiv-only** (`2608.00311v1`, cs.CL; **no DBLP record yet** — `dblp.org/rec/journals/corr/abs-2608-00311.bib` returns **404**; OpenReview: no match). Authors: Haghifam, Cong, Sun | extracts hidden states from **a chosen intermediate Transformer layer**, a lightweight compressor stores them as memory blocks, a **query-conditioned selector** picks blocks, a **decompressor** expands only the selected blocks into states compatible with **an intermediate decoder layer**. 1B/3B same-backbone, four long-context QA benchmarks; beats compression baselines and (at 3B) full-context finetuning on 3 datasets; reduces TTFT and improves decode throughput vs ICAE | **Abstract fetched and read this session** (not paraphrased from the audit). **(a) Concurrent** — 2026-07-31 vs A02's gates 2026-08-09/10/12, *under two weeks apart*; the standing rule makes preemption impossible. **(b) SeDeM *fixes* its layer; A02's dependent variable IS the layer.** SeDeM says "a chosen intermediate layer" and optimises QA score at that choice; A02 sweeps `j ∈ {0,6,9,12,18}` with the recipe matched on 22 fields and reports the accuracy curve. **(c) Sign of the result is opposite and that is the point**: SeDeM reports *wins*; A02 measured that **A0, no adapter at all, is the best arm** (99.75 on the retrieval-closed slice) — every A02 number is a **tax**. **(d) Benchmark family**: SeDeM evaluates long-context **QA**; A02's own `depth_vs_retrieval` gate showed **54.9–78.6 %** of the movement on that family is *retrieval* (recall@12 22.9–63.2 %), so A02 restricts its read-out to retrieval-**closed** RULER cells (recall@12 **99–100 %**, VT directly measured **100.0 %**). That is a **defect-fixing follow-up** on how this family is measured, which is exactly what Rule 1 asks for. |
| **RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference** | **2026-08-05** | **arXiv-only** (`2608.04991v1`, **cs.DC**; **no DBLP record** — 404; 10 pages) | reference-aware codec over **boundary hidden states** of a split (head/tail local, middle cloud) deployment: retrieves exact-token historical spans for prefill uplinks, reuses reconstructed uplink state for downlinks, causal predictors for decode references, grouped affine alignment + calibrated residual quantization. 3 models, 9 model-link pairs, TTFT/TPOT ratios 1.24–2.72× / 1.01–2.79×, 12 non-PPL task-score changes −0.40 to +2.50 | **Abstract fetched and read this session.** Concurrent (2026-08-05). Same *tensor* (a boundary hidden state at a layer cut), completely different **objective and unit**: RAC minimises **bytes on a network link** under a device–cloud partition and reports TTFT/TPOT; A02 has **no network, no device, no privacy claim**, and its unit is **bytes on disk vs raw text** (where A02 lost: 2048×) plus **paired accuracy pp**. RAC's layer cut is a deployment given, not a variable. |
| **PromptDistill: Query-based Selective Token Retention in Intermediate Layers** | 2025 | **arXiv-only** (DBLP `journals/corr/abs-2503-23274`, CoRR 2025 = `Informal and Other Publications`; OpenReview shows **`aclweb.org/ACL/ARR/2025/October/Submission`** = **ACL ARR under review, NOT accepted**). ⚠️ do not cite as an ACL paper | selects/retains tokens' hidden states in early layers, query-based, to cut compute while keeping quality | **token-axis** selection inside intermediate layers. A02 keeps all tokens of the selected chunks and varies **depth**. Cited as proof that "operate on intermediate hidden states" is not a blank space. |
| **Q-Filters: Leveraging QK Geometry for Efficient KV Cache Compression** | 2025 | **ICLR 2025 Workshop SLLM, Camera-Ready** (OpenReview `venueid = ICLR.cc/2025/Workshop/SLLM`, `Camera_Ready_Revision`; DBLP CoRR `abs-2503-02812`). **Workshop, not main conference** | training-free KV compression using QK geometry to score keys | compresses the **KV of one forward pass**; no cross-query store, no depth axis. |
| **Dynamic Memory Compression (DMC)** | 2024 | **ICML 2024** (DBLP `conf/icml/NawrotLCTP24`) | retrofits LLMs to learn per-head online KV merging | learned compression *rate*, full depth. |
| **Long Context Compression with Activation Beacon** | 2025 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision`; DBLP `conf/iclr/Zhang0XSYD25`) | special beacon tokens condense activations into a compact form, trained | **the strongest "compress activations by training" baseline.** A02 never beat doing nothing, so A02 cannot position itself against Beacon on quality at all. |
| **In-context Autoencoder (ICAE)** | 2024 | **ICLR 2024** (DBLP `conf/iclr/00010WWCW24`; CoRR `abs-2307-06945`) | learned encoder compresses context into memory slots a frozen LM reads | SeDeM's named baseline. Slot-based, full-depth decoding. |
| **Learning to Compress Prompts with Gist Tokens** | 2023 | **NeurIPS 2023** (DBLP `conf/nips/Mu0G23`) | trains gist tokens that stand in for a prompt | the origin of "learned reusable compressed prefix". |
| **xRAG: Extreme Context Compression for RAG with One Token** | 2024 | **NeurIPS 2024** (DBLP `conf/nips/0002W00CWZ024`) | compresses a retrieved document to one token via modality fusion | extreme-ratio compression, full depth. |
| **LLMLingua** / **LLMLingua-2** | 2023 / 2024 | **EMNLP 2023** (Anthology `2023.emnlp-main.825`, verified) / **Findings of ACL 2024** (Anthology `2024.findings-acl.57`, DOI `10.18653/v1/2024.findings-acl.57`, DBLP `conf/acl/PanWJXLZLR0LZQ024`, pp. 963–981) | token-level prompt compression in **text space** | text-space, not activation-space. Cited as the text-space arm of the compression axis. |
| **TurboRAG: Accelerating RAG with Precomputed KV Caches for Chunked Text** | 2025 | **EMNLP 2025** (Anthology `2025.emnlp-main.334`, verified; DBLP `conf/emnlp/LuWRCT25`, pp. 6588–6601) | offline-precomputed chunk KV, reordered/repaired at serve time | the **published** version of A02's own cost story, at full depth. A02's `N*` crossover must be positioned relative to this, not as new. |

**Family verdict:** this is where the audit correctly located the pressure. Both flagged
2026-08 papers (SeDeM, RAC) are **concurrent by weeks** and therefore cannot preempt
under Rule 1; and both **fix** the layer that A02 **varies**. But A02 must concede
outright: intermediate-hidden-state storage, query-conditioned selection, and
decompression into decoder-compatible states are **SeDeM's, not A02's**.

### 1.3 Family: chunk-local vs contextual encoding

| paper | year | verified venue | what it does | precise difference from A02's live claim |
|---|---|---|---|---|
| **Block-Attention** (see §1.1) | 2024/25 | **ICLR 2025 Poster** | isolates blocks during prefill, then trains to recover the lost cross-block context | **owns the finding that chunk-local encoding costs accuracy and that training recovers it.** `PROPOSAL.md`'s framing ("独立 chunk Write 缺少 lower-layer document context", overlap `w=32/w=128`, lower-layer Write LoRA) is **the same diagnosis + the same two fixes** at a different depth. A02 must cite it and **must not present chunk-local-context loss as its discovery.** |
| **CacheBlend** | 2025 | **EuroSys 2025** | selectively recomputes a token subset to restore cross-chunk attention | the "partial recompute" repair; same ownership note. |
| **The Power of Noise: Redefining Retrieval for RAG Systems** | 2024 | **SIGIR 2024** (DBLP `conf/sigir/CuconasuTSFCMTS24`) | pack composition (incl. random distractors) changes RAG accuracy | **directly relevant to an A02 sub-finding**: the dvr gate found the retrieval step stays significantly negative even on the retrieval-HIT subset, i.e. narrowing 30–60 chunks to 12 removes useful distractor/aggregation context. That effect is **this paper's territory**; A02 may cite it as corroboration but not claim it. |

### 1.4 Family: split inference / readout repair

| paper | year | verified venue | what it does | precise difference from A02's live claim |
|---|---|---|---|---|
| **RAC** (see §1.2) | 2026-08 | **arXiv-only** | boundary-activation codec for split LLM inference | concurrent; bytes-on-wire objective, no depth sweep. |
| **LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding** | 2024 | **ACL 2024** (Anthology `2024.acl-long.681`, verified; DBLP `conf/acl/ElhoushiSLHWL0A24`, pp. 12622–12642) | trains for early exit at intermediate layers + self-speculative decoding | **direction is opposite, and that is the whole distinction.** Early exit **discards** the layers above `j` to save latency; A02 **runs every layer above `j`** and only moves where the *cache boundary* sits. In A02 the upper stack is the consumer; in early exit it is the thing removed. |
| **BUDDY: Budget-Driven Dynamic Depth Routing** | 2026-06 | **arXiv-only** (`2606.09514`) | per-request depth routing under a budget | concurrent, and it is **B02's** family (B02's own kill gate already fired). Depth-as-a-latency-knob, not depth-as-a-cache-boundary. |
| **Skip a Layer or Loop It? Learning Program-of-Layers in LLMs** | 2026-06 | **arXiv-only** (`2606.06574`) | learns per-input layer execution programs | concurrent; changes *which* layers run, not where a reusable cache is cut. |

### 1.5 Family: write/read joint distillation

| paper | year | verified venue | what it does | precise difference from A02's live claim |
|---|---|---|---|---|
| **Why Limit the Residual Stream to Layers and Not Tokens? Persistent Memory for Continuous Latent Reasoning** | 2026-06 | **ICML 2026 Workshop MusIML, Poster with `Camera_Ready_Revision`** (OpenReview note `UlcVF4RsBz`, `venueid = ICML.cc/2026/Workshop/MusIML`); **also** AdaptFM Poster (`1CjuJgwTqF`) and FoGen Workshop Poster (`ZR9ieOo0J1`) — three separate workshop records. DBLP `journals/corr/abs-2606-07720` (CoRR 2026). **Workshops only; no main-conference record.** Authors: Farhan, Chaudhary | learned **write / read / forget gates** over a persistent residual memory for latent reasoning | the audit flagged this as an A02 collision. It owns **learned write/read gating over persistent memory**, i.e. A02's *stage-2* joint-training design (odd steps Write, even steps Read). **A02 never ran stage 2** — `phase1_c1_vs_c2_gate` fired first and `STATUS.json:59-61` explicitly blocks building Configs 3/4/5. So this collision costs A02 a design it did not execute; nothing measured is affected. |
| **Cartridges** (see §1.1) | 2026 | **ICLR 2026 Poster** | self-study training of the reusable artifact | the strongest "train the Write" reference. A02's Write-LoRA is a weaker instance of this idea. |
| **Semformer: Transformer LMs with Semantic Planning** | 2024 | **EMNLP 2024** (Anthology `2024.emnlp-main.1039`, verified; DBLP `conf/emnlp/YinDS024`) | trains an LM to predict a latent semantic plan of the continuation | latent-target training, not read/write over a store. |

**A02's own distillation recipe is *not* novel and A02 never claimed it was**: the
trainer `scripts/train_qcmem_distill.py` was used unmodified, and the `j=0`
degeneracy (teacher ≡ student at `resume_j=0`) is **stated in its own docstring** —
recorded in `memory/read-the-trainer-docstring-before-designing-a-control.md`.

---

## 2. The strongest single collision, stated head-on

**SeDeM (`arXiv:2608.00311`, 2026-07-31, arXiv-only).** It is the closest paper in
the literature to A02's mechanism: intermediate-layer hidden states → compressed
memory blocks → query-conditioned selection → decompression into states compatible
with an intermediate decoder layer. If A02 had a live *method* thesis, SeDeM would be
its main rival.

**It does not preempt A02's live claim, for four independent reasons:**

1. **Concurrency.** 2026-07-31 vs A02's gates on 2026-08-09/10/12 — **under two
   weeks.** Rule 1 makes preemption impossible on the date alone. No further argument
   is needed for the *formal* verdict; the three below are what makes the
   differentiation *substantive*.
2. **Method vs price.** SeDeM proposes a method and reports it winning. A02's live
   claim is a **priced knob**: the accuracy cost of depth, with the honest result that
   **A0 (no adapter) is best**. These are not the same kind of statement, and A02's
   is the one nobody publishes because it is negative.
3. **Fixed layer vs swept layer.** SeDeM's abstract says "a chosen intermediate
   layer" — singular, fixed. A02's independent variable *is* `j`, swept over
   `{0,6,9,12,18}` with 22 recipe fields asserted matched and an **exact
   capacity-matched pair** (A2 `j=6,r32` vs A6 `j=12,r40`, both **72,744,960**
   trainable params, machine-asserted pre-launch) that separates depth from adapter
   capacity. **No paper found in any family above runs that control.**
4. **Retrieval-closed measurement.** SeDeM's four benchmarks are long-context **QA**.
   A02's own `depth_vs_retrieval` gate measured that on that family **54.9–78.6 %** of
   the effect is retrieval (recall@12 22.9–63.2 %), and A02's read-tax gate further
   measured that BABILong **fails to recover the ordering** of arms whose true RULER
   gap is 70–84 pp (4/6 point estimates inverted, though the inversion is **not
   statistically significant** — `STATUS.json:448`). A02 therefore reads out only on
   cells with recall@12 = **99–100 %** (VT **100.0 %** directly measured). That is a
   **follow-up correcting a measurement defect in the family SeDeM evaluates on** —
   the disposition Rule 1 prescribes instead of abandonment.

**Required of any A02 write-up:** cite SeDeM as concurrent work, state that A02
prices a depth knob rather than proposing a compression method, and **never** claim
priority on intermediate-layer hidden-state memory or on selective decompression.

---

## 3. Must-NOT-claim list (binding on any A02 text)

Each line names the owner. Per Rule 1's 2026-08-12 strengthening, this is a
**citation-obligation list, not a scope ceiling.**

1. ❌ **First to cache an intermediate/attention state and reuse it across queries.**
   Owned by Prompt Cache (MLSys 2024), Memorizing Transformers (ICLR 2022),
   Compressive Transformer (ICLR 2020).
2. ❌ **First to store hidden states from a chosen intermediate layer, select blocks
   by query, and decompress them into decoder-compatible states.** **SeDeM owns
   this** (`2608.00311`), in full and as a working method.
3. ❌ **First to compress boundary activations at a layer cut.** **RAC owns this**
   (`2608.04991`) for the split-inference setting.
4. ❌ **First to operate on / select within intermediate-layer hidden states for
   efficiency.** PromptDistill (`2503.23274`, ACL ARR under review) and Q-Filters
   (ICLR 2025 Workshop SLLM) precede it.
5. ❌ **First to observe that chunk-local encoding loses document context, or that
   overlap / a trained adapter repairs it.** **Block-Attention owns this**
   (ICLR 2025 Poster); CacheBlend (EuroSys 2025) owns selective recompute as the
   repair. This kills the novelty of `PROPOSAL.md`'s `w=32/w=128` and lower-layer
   Write-LoRA arms as *ideas* — they remain valid *measurements*.
6. ❌ **First to precompute chunk KV offline and amortise it across queries, or to
   compute a break-even query count.** TurboRAG (EMNLP 2025), Cartridges
   (ICLR 2026 Poster), LLoCO (EMNLP 2024), RAGCache (ACM TOCS 2026) own the
   amortisation frame. A02's `N*` is an instance, not an invention.
7. ❌ **First to use learned write/read/forget gating over a persistent memory.**
   *Persistent Memory for Continuous Latent Reasoning* (`2606.07720`, ICML 2026
   Workshop MusIML Poster) owns this. A02's unexecuted stage 2 is the same design.
8. ❌ **First to train an LM to be compressible / to make representations more
   compressible by training.** **KV-CAT owns this** (`2605.05971`, ICML 2026
   Workshop HiLD Poster) — see B01's `RELATED_WORK.md` §1.2. A02 must not borrow it
   either.
9. ❌ **Any claim that CoMem beats RAG on quality.** Killed by A02's *own* gate
   (`phase1_c1_vs_c2_gate`, 2026-08-09), not by literature. Independent of everything
   above.
10. ❌ **Any storage-method claim.** `h12` = **2048×** raw text bytes/token;
    CoMem-total/RAG-total 632–1129×. The only favourable ratio (18× smaller than
    full-depth KV) **must never be reported without the 2048× alongside it** — that
    asymmetric disclosure is already an open defect against `paperA`
    (`STATUS.json:247`).
11. ❌ **Reading `j12_frozen`'s −97 pp as "depth costs 97 pp".** It means
    *untrained depth-12 resume is non-functional*. A02's own defensible depth numbers
    are the `read_deployed` ones (−3 to −12 pp).
12. ❌ **Any BABILong-based depth claim, and any pooled BABILong / LongEval figure.**
    The banned values are −17.89 pp and +2.00 pp; they average over cells with
    opposite true signs.
13. ❌ **"The knob is free to `j≈9`" without the de-saturation caveat.** On the
    harder retrieval-closed cell (`niah_single_3 × 16k`), `j=6` already costs
    **−8.00 pp SIGNIFICANT** (`STATUS.json:439`). The shallow end was not free; it
    was **unmeasurable** on saturated cells.
14. ❌ **Ownership of the BABILong scorer-format/misorder finding.** Reassigned to
    **B11**; A02 keeps provenance only (`STATUS.json:365-370`).
15. ❌ **Any cross-model / cross-family / multi-seed claim.** Qwen3-8B only,
    **one seed (42)**, H20/bf16/sdpa, n=100/cell.
16. ❌ **Any claim of differential LR.** Verified false repo-wide
    (`status/PAPERB_DIFFERENTIAL_LR_NEVER_ACTIVE.md`; the `module.`-prefix strip
    landed only in `train_olmo2_arch_probe2.py:316`).

---

## 4. Safe gap — one sentence, experimentally checkable

> **On a retrieval-closed read-out (gold-support recall@12 ≥ 99 %, retrieval pack
> byte-identical across arms), the accuracy cost of resuming a frozen LLM's upper
> stack from a cached layer-`j` hidden state is a *cliff in `j`*, not a slope, and the
> cliff is attributable to depth rather than to adapter capacity at exactly matched
> trainable-parameter count.**

**Why this sentence and no larger one.** Each clause is load-bearing and each was
checked for absence in §1:

* *retrieval-closed* — none of the 30-odd papers above holds retrieval recall fixed
  at ~100 % while varying a read-side variable. A02 measured why that matters: on the
  retrieval-open family, 54.9–78.6 % of the effect is retrieval.
* *cliff, not slope* — requires a **swept `j`**. Every collision fixes `j`.
* *depth not capacity* — requires the **exactly capacity-matched pair** (72,744,960
  params both). No paper in any family runs this control, because none is asking a
  depth question.
* *frozen upper stack* — distinguishes A02 from early exit (LayerSkip), which removes
  the upper stack, and from split inference (RAC), which relocates it.

**What this sentence deliberately does NOT say:** nothing about quality wins, nothing
about storage, nothing about generality across models/seeds/tasks. **It is a priced
knob on a mechanism this repo already declared dead.** `STATUS.json:332` answers the
obvious follow-up question directly and in the negative:

```
"is_a_priced_knob_plus_a_benchmark_finding_a_paper": "NO."
```

**So the honest disposition is unchanged by this literature pass:** A02 stays
`CLOSED_NO_THESIS_DIAGNOSTIC_ASSETS_RETAINED`, `gpu_policy: NO further A02 GPU`.

> **Direction of the kill matters and must not be blurred.** A02 was killed **by its
> own pre-registered kill clause on 2026-08-09**, before any of this literature was
> consulted. Nothing in this file kills anything. Rule 1's final clause —
> 「判死一个方向只能靠实验证伪」— is satisfied in the correct order: experiment first,
> attribution second. Conversely, **the literature does not revive A02 either**:
> resurrection requires a **new mechanism** (`STATUS.json:333`), not a citation.

---

## 5. Honest gaps in this adjudication

1. **Semantic Scholar unusable: HTTP 429 on every call** (2 attempts, both rate-limited
   through `hy-proxy.woa.com:3128`). Per the standing rule S2 is only ever a
   cross-check and never the venue authority, so no verdict above rests on it — but
   the intended cross-check **did not happen** and every venue call is therefore
   single-authority (DBLP, OpenReview, or Anthology, as tabulated).
2. **arXiv API was rate-limited (`Rate exceeded`, HTTP 429) for the first ~15 minutes**
   of this session and required a retry-with-backoff client. Two search queries
   (`abs:"hidden state" AND abs:"memory" AND abs:"long context"` and
   `abs:"chunk" AND abs:"context" AND abs:"hidden states" AND abs:"reuse"`)
   returned **zero** entries, which for such generic terms is implausible and is more
   likely a throttled/empty response than a true absence. **Those two searches should
   be re-run before submission.**
3. **DBLP `/search/publ/api` intermittently returned HTTP 500 and 503** (5 queries
   affected; recovered after raising the inter-query delay to 4 s). One query,
   `Efficient Position-Independent Context Caching`, 500'd on first attempt and
   succeeded on retry — so a DBLP error must never be read as "not in DBLP"
   (this is the same failure mode A04's pass recorded on 2026-08-09).
4. **`arXiv-only` entries whose venue I could NOT verify from this node**, listed
   explicitly so none is silently treated as unpublished:
   SeDeM (`2608.00311`, **no DBLP record at all — 404**),
   RAC (`2608.04991`, **404**),
   Infini-attention (`2404.07143`, CoRR only despite heavy citation),
   TransMLA (`2502.07864`), Cache-Craft (`2502.15734`),
   Adaptive KV Cache Reuse (`2605.24022`), BUDDY (`2606.09514`),
   Skip-a-Layer-or-Loop-It (`2606.06574`).
   For the two 2026-08 papers a missing DBLP record is **expected** (DBLP's CoRR
   ingest lags ~4–8 weeks); it is evidence of recency, not of low quality.
5. **Two papers must not be miscited as accepted conference papers:**
   **PromptDistill** is `aclweb.org/ACL/ARR/2025/October/Submission` = **ACL ARR under
   review** (cite as arXiv preprint / ARR under review), and **Q-Filters** /
   **Landmark Attention** / **KV-CAT** / **Persistent Memory** are **workshop**
   records (`ICLR.cc/2025/Workshop/SLLM`, `ICML.cc/2023/Workshop/ES-FoMO`,
   `ICML.cc/2026/Workshop/HiLD`, `ICML.cc/2026/Workshop/MusIML`), not
   main-conference papers.
6. **Title divergence between arXiv and venue record** on **Block-Attention**
   (arXiv: *"Block-Attention for Efficient RAG"*; OpenReview: *"Block-Attention for
   Efficient Prefilling"*). Cite from the venue record and check the camera-ready
   before quoting anything from it — the exact hazard
   `memory/venue-verify-acl-family-needs-anthology.md` warns about.
7. **No `.bib` entries are emitted by this file, deliberately.** Per
   `memory/tcodex-exec-no-dash-c-flag.md`, nothing enters a bibliography until
   venue-verified by family; four entries above (§5.4) are not yet eligible.
8. **Abstracts read, full texts not read.** I fetched and read the arXiv abstract
   pages for SeDeM, RAC, KV-CAT and "Through the Bottleneck". Every other entry's
   "what it does" column is from its title + venue record + prior repo notes.
   **Any A02 sentence that turns on a specific number or control inside one of those
   papers must be checked against its PDF first.**
9. **Zero cross-disk verification.** All A02 raw results named in `STATUS.json`
   (`babilong_results/a02_dvr_*`, `ruler_results/a02_*`) live on **zwfy6**, which is
   not mounted here. Every such path in this document is **recorded-only** and must be
   `ls`-confirmed on `.73`/`.82`/`.104` before it is cited as evidence
   (`memory/two-disk-rule-applies-to-main-too.md`).

---

## 6. Verdict

```
related_work_status: audited
novelty_verdict: NOT PREEMPTED on the live claim; but the live claim is a priced knob,
                 and A02 remains CLOSED by its OWN kill gate (2026-08-09), not by prior art.
promotion: NO. Unchanged by this pass.
gpu_policy: unchanged -- NO further A02 GPU. Resurrection needs a NEW MECHANISM, not a citation.
```

* **No paper found is 完全相同 / 抄袭.** The two nearest (SeDeM, RAC) are **concurrent
  by days-to-weeks** and both **fix the layer A02 sweeps**. `already_dead_should_archive`
  on novelty grounds is **not** warranted — and note A02 is deliberately **not** archived
  for a separate reason (`STATUS.json:331`): its artefacts are load-bearing for B05, B11
  and any future depth/read claim, and archiving would signal "do not reuse".
* **The audit's 不足 rating is discharged**: all five named collision families
  (§1.1–§1.5) now have named papers with verified venues and per-paper differences,
  plus a 16-item must-not-claim list and a one-sentence safe gap.
* **What this pass changed:** it did **not** rescue A02, and it was not trying to. It
  produced the attribution list a write-up needs, and it moved three items out of
  A02's possible-claims space that were still implicitly in `PROPOSAL.md` — the
  chunk-local-context diagnosis (→ Block-Attention), the offline amortisation +
  break-even frame (→ TurboRAG / Cartridges), and the write/read gating design
  (→ `2606.07720`).
