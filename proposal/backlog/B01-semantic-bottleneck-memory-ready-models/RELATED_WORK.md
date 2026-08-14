# B01 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU spent. Adjudication only — this file runs nothing and
launches nothing.**

Purpose: close the blocker `proposal/ready_queue.py:504-506` actually trips on
(`RELATED_WORK.md absent (blocks PROMOTION; 0-GPU task)`), flip
`novelty_checked: false` (`STATUS.json:89`), and answer the five collision families
named for B01 in `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:91`
(rating: **不足** — insufficient; priority **4 of 7** in that audit's fill order).

---

## 0. What B01 claims, and the exact distinction the audit demands

**B01's claim** (from `PROPOSAL.md` "核心主张", condensed to one sentence):

> A vanilla LLM's mid-stack hidden state `h_j` is **not** naturally low-rank; inserting
> a **residual-free, feature-axis bottleneck** near the semantic handoff and training
> **from scratch** (or continuing pretraining) makes the representation *itself*
> low-dimensional and rank-truncation-robust, at a fixed and scale-shrinking LM tax —
> i.e. **the model is trained so that its mid-stack state is persistable**, rather than
> compressed after the fact.

The audit's prescribed distinction, verbatim from
`RELATED_WORK_GAP_AUDIT_20260808.md:114-117`:

```text
这些工作压缩已有 representation；
B01 声称通过 pretraining 让 representation 本身成为低维、可持久化 latent。
```

and its warning at line 119: **if the method still needs to restore a full-width hidden
to be readable, the difference weakens sharply.**

**That warning is currently a live, verified defect, not a hypothetical.**
`STATUS.json:78-83` (`blocking_dependency`), re-derived from code this session:

* `scripts/train_qwen_bottleneck_continued.py:112-133` — `inject_bottleneck` wraps
  layer `j` in `BottleneckLayer` (**down → GELU → up, NO residual**); I read the
  docstring and the body directly. The wrapper's **output is back at
  `hidden_size`**, and `QCMemModel` caches *that* layer output as `h_j`.
* **Consequence:** today the funnel constrains the **rank** of what is stored but not
  the **width of the bytes** stored. `bytes/token` is **identical** between the
  bottleneck and vanilla arms. Arms 2/3/4 of the next gate would differ **only in
  quality, not in storage**.

**So B01's honest current claim is the *precondition*, not the headline.** What is
measured (all re-derived from disk per `STATUS.json:27-77`, `_provenance_note`
2026-08-14):

| quantity | vanilla | bottleneck `d512` | source |
|---|---|---|---|
| PCA `dim99` of `h_j`, 1B | **1825** | **438** | `outputs/e2e_1b_deltanll/ctx2048_gpu0.json` (stable at ctx 3072/4096: 1829/439) |
| PCA `dim99` of `h_j`, 3B | **2790** | **467** | `outputs/e2e_3b_deltanll/ctx2048_gpu0.json` (2797/467, 2800/467) |
| ΔNLL @ rank 128, 1B | 0.0304 | **0.0022** | same JSONs, `by_rank['128'].dNLL` |
| ΔNLL @ rank 128, 3B | 0.0669 | **0.0135** | same |
| ΔNLL @ rank 256, bottleneck | — | **negative** (−0.0028 / −0.0015) → truncation is free | same |
| fixed LM tax (training PPL, 16k-step from-scratch dim sweep) | — | 1B: **+4.5 / +5.9 / +8.5 %** at `d1024/512/256`; 3B: **+4.7 / +5.8 %** at `d512/256` | `status/RUN_REGISTRY.md` dim-sweep; arithmetic re-checked in `STATUS.json:66` |

⚠️ **Unit trap that any B01 text must respect** (`STATUS.json:67`): the "4–8.5 %"
range is **training-PPL tax from the from-scratch dim sweep**. It is **not** the
`r_max` NLL gap in `outputs/e2e_*_deltanll/` (+0.16 / −0.19 / −0.13 % at 1B;
+0.73 / +0.82 / +0.84 % at 3B). **Never mix the two.**
Also: the tax **rises monotonically with bottleneck depth** (L1 +4.2 % → L12 +9.6 % at
fixed `d512`; `memory/bottleneck-layer-sweep-monotone.md`), so `j` is a
**cost-vs-cacheability trade-off point, not a tax minimum**.

## 0.1 Standing rules this adjudication obeys

**Rule 1 — overlap is not preemption.**
`memory/prior-work-differentiate-dont-abandon.md` (user 2026-08-07): the bar is
**完全相同 / 抄袭**, not overlap; work within **2–3 months is concurrent** and cannot
preempt; the correct output on a near-hit is **differentiation or a
follow-up-that-fixes-a-defect**. The 2026-08-12 strengthening applies: a novelty audit
may emit a **citation-obligation list** and **may not** emit a scope ceiling.
**B01 may be killed only by its own kill gate** (`PROPOSAL.md` "Kill 条件", copied
verbatim into `STATUS.json:18-25`), never by a literature count.
Status of that gate today: **not firing.** Condition 2 ("fixed LM tax 在强模型上扩大")
points the *right* way so far — the tax shrinks 1B → 3B — but has never been tested at
8B (`STATUS.json:25`).

**Rule 2 — venue verification is per-family, and "unverified" ≠ "unpublished".**

| family | authority used in this pass |
|---|---|
| ICLR / NeurIPS / ICML / TMLR (+ workshops) | **OpenReview `venueid`** + `Camera_Ready_Revision` in `invitations` as the accept signal |
| ACL / EMNLP / NAACL **incl. Findings** | **ACL Anthology** page (fetched, title compared) **+ DBLP** record |
| everything else | DBLP `type`: `Conference and Workshop Papers` / `Journal Articles` = real record; `Informal and Other Publications` + `venue = CoRR` = **preprint only** |

`arXiv-only` means **"I could not verify a peer-reviewed venue from this node"**, not
"it has no venue". Recorded honestly in §5.

---

## 1. Named closest collisions

### 1.1 Family: bottleneck Transformer / non-uniform width

| paper | year | verified venue | what it does | precise difference from B01 |
|---|---|---|---|---|
| **Variable-Width Transformers** | **2026-06-16** | **arXiv-only** (`2606.18246`; DBLP `journals/corr/abs-2606-18246`, CoRR 2026 = `Informal and Other Publications`; OpenReview: **no match**) | **This is the single most dangerous hit in this entire pass and it is not in the audit's list.** Trains an ×-shaped (narrow-in-the-middle) LM **from scratch** with **non-uniform width across depth**; reports it **outperforming parameter-matched uniform baselines on LM loss**, with **22 % fewer FLOPs** under loss-matched scaling and **15 % smaller KV cache memory / I-O**; and explicitly analyses that "this bottleneck structure results in **qualitatively different representations in residual streams**". Abstract fetched and read this session | **Concurrent** (2026-06-16, under 2 months before this pass) → cannot preempt under Rule 1. But it takes **three things B01 currently claims**: (a) that a mid-depth width bottleneck trained from scratch is viable; (b) that it changes residual-stream representations qualitatively; (c) that it **saves KV-cache bytes**. **The differences are real and load-bearing, and one of them is a defect B01 fixes:** VWT's bottleneck is **residual-preserving inside a standard block** and its objective is **resource-optimal scaling** (loss per FLOP); B01's `BottleneckLayer` is **residual-FREE by construction** (verified in code, §0) and its objective is a **rank property of one designated layer's output for cross-query persistence**. VWT reports **LM loss and FLOPs**; B01 reports **PCA `dim99`** and **ΔNLL under rank truncation** — neither appears in VWT's abstract. Most importantly, VWT's tax is **negative** (it *beats* the uniform baseline) while B01's is **positive (+4.5–8.5 %)**, which is strong evidence the two are **not the same construction**. ⚠️ **B01 must cite it and must drop any claim of novelty for "a mid-depth width bottleneck trained from scratch changes the residual stream".** |
| **Funnel-Transformer** | 2020 | **NeurIPS 2020** (DBLP `conf/nips/DaiLY020`; CoRR `abs-2006-03236`). OpenReview: no match (pre-dates their coverage of NeurIPS) | pools along the **sequence** axis to reduce length, with an optional up-sampling decoder | **orthogonal axis.** Funnel compresses **tokens**; B01 compresses the **feature dimension** at fixed sequence length. B01's launcher name (`qwenbott_funnel_L12_d512`) borrows the word "funnel" and this **must not be allowed to imply the method** — a reviewer will make that association immediately. |
| **Bottleneck Transformers for Visual Recognition (BoTNet)** | 2021 | **CVPR 2021** (DBLP `conf/cvpr/SrinivasLPSAV21`) | replaces spatial convolutions with self-attention in ResNet bottleneck blocks | different meaning of "bottleneck" (ResNet block), vision, no cache/persistence question. Cited only to disambiguate terminology. |
| **Low-Rank Bottleneck in Multi-head Attention Models** | 2020 | **ICML 2020** (DBLP `conf/icml/BhojanapalliYRR20`) | shows the head-dimension `d/h` induces a low-rank bottleneck limiting attention expressivity, and decouples head size from depth/width | **the theoretical anchor for "a width bottleneck constrains what attention can express"**, i.e. the *mechanism behind B01's LM tax*. B01 must cite it as the reason a tax exists at all, and must not present the tax's existence as a new observation. |
| **Perceiver** | 2021 | **ICML 2021** (DBLP `conf/icml/JaegleGBVZC21`) | iterative cross-attention into a small fixed-size latent array | fixed-size latent **bottleneck by architecture**, and it is trained end-to-end — the closest classical "the latent *is* the representation" design. But it is a **replacement architecture** for the whole encoder, not a rank constraint at one designated layer of a decoder-only LM, and it has no cross-query persistence axis. |
| **Set Transformer** | 2019 | **ICML 2019** (DBLP `conf/icml/LeeLKKCT19`) | inducing points reduce set-attention cost | inducing-point bottleneck; same orthogonality note as Perceiver. |
| **Sentence Bottleneck Autoencoders from Transformer LMs** | 2021 | **EMNLP 2021** (Anthology `2021.emnlp-main.137` verified; DOI `10.18653/v1/2021.emnlp-main.137`; DBLP `conf/emnlp/Montero0S21`, pp. 1822–1831) | adds a **single-vector bottleneck** to a pretrained Transformer LM and trains an autoencoding objective through it | closest classical "insert a bottleneck into an LM and train" paper. Its bottleneck is **one vector per sentence** for representation-learning/generation, not a **per-token mid-stack state for a persistent cache**. |
| **Semformer** | 2024 | **EMNLP 2024** (Anthology `2024.emnlp-main.1039` verified; DBLP `conf/emnlp/YinDS024`) | trains an LM to predict a **latent semantic plan** of the continuation | latent-target *training signal*, not a width constraint on the residual stream. |

### 1.2 Family: activation / KV codec — including the audit's flagged direct collisions

| paper | year | verified venue | what it does | precise difference from B01 |
|---|---|---|---|---|
| **Training Transformers for KV Cache Compressibility (KV-CAT)** | **2026-05-07 (v2 05-12)** | **ICML 2026 Workshop HiLD, Poster** — OpenReview note `FcCMc3s0TL`, `venue = "HiLD at ICML 2026 Poster"`, `venueid = ICML.cc/2026/Workshop/HiLD`; **invitations contain NO `Camera_Ready_Revision`** (workshop track). DBLP `journals/corr/abs-2605-05971` (CoRR 2026). **Workshop, NOT main conference.** Authors: Gelberg, Eitan, Bronstein, Gal, Maron. 32 pp | **The single closest paper to B01's *thesis*, and it is not in the audit's list.** Abstract fetched and read this session. It (a) **formalises KV compressibility as a property of the learned representations rather than of the context**; (b) **proves** almost any sequence-to-vector function admits both highly compressible **and** inherently non-compressible transformer implementations, "highlighting the need to guide transformers toward compressible representations **during training**"; (c) proposes **KV-CAT**, a **continued-pretraining** procedure with a train-time KV sparsification policy that masks KV slots to force fewer slots and induce post-hoc-compressible representations; (d) shows improved quality–budget tradeoffs on retrieval, long-context QA and compressed-prefix PPL | **Concurrent** (2026-05-07 — ~3 months, at the outer edge of Rule 1's window; treat as concurrent and say so). **This is the paper that owns B01's framing sentence.** "Post-hoc methods are limited by how compressible the model's representations are, therefore train for compressibility" is **KV-CAT's thesis, stated and proved**. B01 **must cite it and must not claim that framing.** Four surviving differences, each checkable: **(i) axis** — KV-CAT masks **KV slots** (the token/slot axis of the KV tensor); B01 constrains the **feature dimension of the residual-stream hidden state**. **(ii) object cached** — KV-CAT keeps the object a **KV cache**; B01's object is `h_j`, a **mid-stack hidden state** resumed by the upper stack. **(iii) mechanism** — KV-CAT's inductive pressure is a **stochastic masking policy**; B01's is a **hard architectural residual-free width constraint**, which is why B01 can report a *deterministic* rank number (`dim99` 1825 → 438) rather than a downstream-method tradeoff curve. **(iv) reported quantity** — KV-CAT reports downstream compression-method quality; B01 reports the **intrinsic rank of the representation** (`dim99`, ΔNLL-vs-rank) *independent of any compressor*. **The honest positioning is follow-up: KV-CAT proves the need and gives a soft/stochastic method; B01 tests whether a hard architectural constraint yields a representation whose low-dimensionality is intrinsic and measurable without choosing a compressor.** |
| **MatryoshkaKV: Adaptive KV Compression via Trainable Orthogonal Projection** | 2024/25 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision`; DBLP `conf/iclr/LinZXKH0ZD25`) | Abstract read this session. Explicitly targets **the feature-dimension axis** of the KV tensor (noting prior work covered the other three axes), tunes **orthogonal projection matrices** with a **distillation objective** and a **Matryoshka training strategy**, then adaptively searches per-layer/per-head compression rates; sustains >90 % performance at ~60 % KV compression on LLaMA-2-7B / Mistral-7B | **The most precise collision on B01's *axis*.** It owns **"compress the feature dimension of the cache by training a nested/truncatable projection"** — which is very close to how B01's `dim99`/ΔNLL-vs-rank framing reads. Two differences that hold: **(i)** MatryoshkaKV **starts from a pretrained LLM and learns a projection *on top of* frozen representations** ("can easily embrace pre-trained LLMs"); B01 changes the **pretraining/CPT objective so the representation itself is low-dimensional** — precisely the post-hoc-vs-formation distinction the audit demands. **(ii)** its object is the **KV cache**; B01's is `h_j`. ⚠️ **B01 must NOT claim novelty for "feature-dimension compression of a cache" or for "nested/truncatable ranks".** |
| **Palu: KV-Cache Compression with Low-Rank Projection** | 2024/25 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision`; DBLP CoRR `abs-2407-21118`. ⚠️ **title differs**: arXiv *"Palu: Compressing KV-Cache with Low-Rank Projection"* vs OpenReview *"Palu: KV-Cache Compression…"*) | low-rank projection of KV, reconstructing on the fly | post-hoc low-rank on a frozen model. Same distinction as MatryoshkaKV. |
| **RAC: Reference-Aware Activation Compression for Split LLM Inference** | **2026-08-05** | **arXiv-only** (`2608.04991v1`, cs.DC; **no DBLP record — 404**) | boundary-activation codec for split inference; grouped affine alignment + calibrated residual quantization; TTFT/TPOT 1.24–2.72× / 1.01–2.79× | Named by the audit. Concurrent by **10 days**. Post-hoc codec on a **frozen** model, objective = **bytes on a network link**; B01 changes the model so the state is intrinsically low-rank, objective = **bytes on disk per token**. |
| **SeDeM: Selective Decompression of Hidden-State Memories** | **2026-07-31** | **arXiv-only** (`2608.00311v1`, cs.CL; **no DBLP record — 404**) | stores intermediate-layer hidden states as compressed memory blocks, query-conditioned selection, decompression into decoder-compatible states; 1B/3B; beats compression baselines and (3B) full-context finetuning on 3/4 datasets | Named by the audit. Concurrent by **2 weeks**. **It compresses the hidden state of a fixed pretrained model with a trained compressor/decompressor; B01 changes the model so no compressor is needed.** But note the sharp practical point: **SeDeM's decompressor restores decoder-compatible full-width states**, which is exactly the pattern the audit says weakens B01's difference — B01's own store does the same today (§0), so **until the `d_bottle` persist path exists, B01 and SeDeM are separated by construction rather than by measurement.** |
| **PromptDistill** | 2025 | **arXiv-only** — DBLP CoRR `abs-2503-23274`; OpenReview shows `aclweb.org/ACL/ARR/2025/October/Submission` = **ACL ARR under review, NOT accepted**. Do not cite as an ACL paper | query-based selective retention of token hidden states in early layers | Named by the audit. **Token axis**, post-hoc, frozen model. |
| **Q-Filters** | 2025 | **ICLR 2025 Workshop SLLM, camera-ready** (OpenReview `venueid = ICLR.cc/2025/Workshop/SLLM`, `Camera_Ready_Revision`; DBLP CoRR `abs-2503-02812`). **Workshop, not main conference** | training-free KV compression from QK geometry | Named by the audit. Training-free, frozen model, single forward pass. |
| **PyramidKV** | 2024/25 | **COLM 2025** (OpenReview `venueid = colmweb.org/COLM/2025/Conference`, `Camera_Ready_Revision`). ⚠️ **also carries `ICLR.cc/2025/Conference/Rejected_Submission`** — cite COLM 2025, never ICLR. DBLP CoRR `abs-2406-02069` only | layer-wise non-uniform KV budgets ("pyramidal information funneling") | **non-uniform budget across depth, post-hoc.** Relevant because B01's own layer sweep found the tax monotone in depth; PyramidKV is the post-hoc counterpart of that observation and must be cited if B01 discusses depth allocation. |
| **Squeezed Attention** | 2025 | **ACL 2025** (Anthology `2025.acl-long.1568` verified; DBLP `conf/acl/HooperKMMZPMKG25`, pp. 32631–32652) | clusters fixed-context keys offline for cheap sparse lookup at serve time | offline preprocessing of a **frozen** model's keys. |
| **LeanKV** | 2024 | **arXiv-only** (DBLP `journals/corr/abs-2412-03131`, CoRR 2024) | unified KV compression (quantization + sparsity) | post-hoc; venue unverified from this node. |
| **CommVQ** | 2025-06 | **arXiv-only** from this node (`2506.18879`) | commutative vector quantization for KV cache | post-hoc codec; venue not checked (§5). |
| **Dynamic Memory Compression (DMC)** | 2024 | **ICML 2024** (DBLP `conf/icml/NawrotLCTP24`) | **retrofits** an LLM with continued training to learn online KV merging | **important nuance for B01's framing:** DMC already *trains* the model to accommodate compression, so "training is involved" is not itself B01's distinction. DMC's axis is again the **token/slot** axis (adaptive merge rate), not feature width. |
| **Long Context Compression with Activation Beacon** | 2025 | **ICLR 2025 Poster** (OpenReview `venueid = ICLR.cc/2025/Conference`, `Camera_Ready_Revision`; DBLP `conf/iclr/Zhang0XSYD25`) | trains beacon tokens that condense activations | trained compression **into extra tokens** of an otherwise unchanged model. |
| **In-context Autoencoder (ICAE)** | 2024 | **ICLR 2024** (DBLP `conf/iclr/00010WWCW24`) | learned encoder → memory slots read by a frozen LM | SeDeM's baseline; slot-based. |
| **Learning to Compress Prompts with Gist Tokens** | 2023 | **NeurIPS 2023** (DBLP `conf/nips/Mu0G23`) | trained gist tokens replace a prompt | origin of trained reusable compressed prefixes. |
| **LLMLingua / LLMLingua-2** | 2023 / 2024 | **EMNLP 2023** (Anthology `2023.emnlp-main.825`) / **Findings of ACL 2024** (Anthology `2024.findings-acl.57`, DOI `10.18653/v1/2024.findings-acl.57`, DBLP `conf/acl/PanWJXLZLR0LZQ024`, pp. 963–981) | text-space prompt compression | text space, not activation space. |

### 1.3 Family: architectural latent KV (the "the cache *is* low-dimensional by design" family)

**This family is B01's biggest attribution risk and the audit does not name it.**
These are not post-hoc compressors — the low-dimensional cache is **built into the
architecture and trained**, which is precisely B01's claimed novelty pattern.

| paper | year | verified venue | what it does | precise difference from B01 |
|---|---|---|---|---|
| **DeepSeek-V2 (Multi-head Latent Attention, MLA)** | 2024 | **arXiv-only** (DBLP `journals/corr/abs-2405-04434`, CoRR 2024; OpenReview: CoRR mirror only). ⚠️ enormously influential yet **I found no peer-reviewed record from this node** | MLA compresses KV into a **shared low-rank latent `cKV`** that is what gets cached, **trained from scratch this way**; ~81 % KV-cache reduction at production scale | **A production-scale LM whose cached object is low-dimensional *by architecture and by pretraining*.** This is the strongest possible challenge to a bare claim of "pretrain so the cached state is a low-dimensional latent" — MLA **did it, at scale, two years earlier**. **B01 must cite MLA and must not claim the general idea.** What survives, and B01 must state it in exactly these terms: MLA compresses the **per-head K/V** consumed by attention **within the same forward pass**; B01 constrains the **residual-stream hidden state `h_j` itself**, the object that a *depth-partitioned* memory hands to the frozen upper stack **across queries**. Different tensor, different consumer, different lifetime. |
| **Through the Bottleneck: How MLA Separates Content from Position** | **2026-07-25** | **arXiv-only** (`2607.23054`; DBLP `journals/corr/abs-2607-23054`, CoRR 2026; OpenReview: no match) | Abstract read this session. First mechanistic-interpretability study of MLA's `cKV` bottleneck on a 114M model: the bottleneck learns a **pure content representation** (98 % entity-identity retention, positional info discarded); a single "semantic hub" layer has the highest SVD effective rank; **the bottleneck is globally over-provisioned, using only 46 % of capacity** | **Concurrent** (3 weeks). It is the closest paper to B01's *measurement style* (effective rank of a bottlenecked representation, per layer). Differences: 114M scale, MLA's per-head `cKV` not the residual stream, and **no persistence/storage axis at all**. **Useful to B01 as corroboration, and it is also a warning:** "the bottleneck is over-provisioned" predicts B01's own rank-256 result (ΔNLL slightly **negative** → truncation free), so B01's `dim99` finding should be presented as **consistent with** this, not as a discovery. |
| **CARE: Covariance-Aware and Rank-Enhanced Decomposition for Enabling MLA** | 2026-03 | **ICLR 2026 Poster** (OpenReview `venueid = ICLR.cc/2026/Conference`, `Camera_Ready_Revision`; arXiv comment self-reports "Accepted at ICLR 2026"; DBLP CoRR `abs-2603-17946`) | converts pretrained GQA → MLA under a fixed KV width via activation-preserving factorization, **non-uniform per-layer rank allocation**, and KV-parity mapping; up to 215× one-shot PPL reduction vs uniform-rank SVD | **Peer-reviewed, 2026, and it owns "allocate a fixed low-rank cache budget non-uniformly across layers, aligning to activations rather than weights".** B01's layer sweep + `d`-sweep is the from-scratch analogue. **B01 must not claim per-layer rank allocation as novel.** Difference: conversion of a pretrained model's attention KV vs from-scratch formation of the residual state. |
| **TransMLA: Multi-Head Latent Attention Is All You Need** | 2025 | **arXiv-only** (DBLP `journals/corr/abs-2502-07864`, CoRR 2025; OpenReview: CoRR mirror only) | converts GQA models to MLA | conversion, not formation. |
| **You Only Cache Once (YOCO)** | 2024 | **NeurIPS 2024 Oral** (OpenReview `venueid = NeurIPS.cc/2024/Conference`, `Camera_Ready_Revision`; DBLP `conf/nips/Sun0ZHWMZ0W24`) | decoder-decoder architecture caching **one** global KV consumed by a cross-decoder | **an architecture designed so that what is cached is small, trained that way.** Its saving is on the **layer/duplication** axis, not the feature axis; B01 must cite it as prior art for "design the architecture around what gets cached". |
| **Cross-Layer Attention (CLA)** | 2024 | **NeurIPS 2024 Poster** (OpenReview `venueid = NeurIPS.cc/2024/Conference`, `Camera_Ready_Revision`; DBLP `conf/nips/BrandonMNPR24`) | shares KV across adjacent layers, trained | same note as YOCO: architectural cache reduction on a non-feature axis. |
| **Tensor Product Attention (T6)** | 2025 | **NeurIPS 2025** (DBLP `conf/nips/ZhangLYQYGY25`; CoRR `abs-2501-06425`) | factorised (tensor-product) QKV so the cache is intrinsically small | another "cache is small by design, trained" point. |
| **Native Sparse Attention (NSA)** | 2025 | **ACL 2025** (Anthology `2025.acl-long.1126` verified; DBLP `conf/acl/YuanGD0ZZXWW0WR25`, pp. 23078–23097) | **natively trainable** sparse attention, pretrained sparse rather than sparsified after | **the cleanest published statement of B01's meta-argument in a neighbouring axis**: train the property in rather than bolt it on. Sparsity axis, not feature-width axis. B01's "formation vs post-hoc" argument must be credited to this lineage. |
| **Kimi Linear** | 2025-10 | **arXiv-only** (DBLP `journals/corr/abs-2510-26692`, CoRR 2025) | hybrid linear-attention architecture with a small recurrent state | fixed-size state by design; venue unverified. |

### 1.4 Family: split-inference compression

| paper | year | verified venue | what it does | precise difference from B01 |
|---|---|---|---|---|
| **RAC** (see §1.2) | 2026-08 | **arXiv-only** | boundary-activation codec, device↔cloud | concurrent; bytes-on-wire, frozen model, no formation claim. |
| **LayerSkip** | 2024 | **ACL 2024** (Anthology `2024.acl-long.681` verified; DBLP `conf/acl/ElhoushiSLHWL0A24`, pp. 12622–12642) | trains for early exit at intermediate layers | **direction is opposite**: early exit **removes** the upper layers; B01 keeps and *feeds* them. |
| **Variable-Width Transformers** (see §1.1) | 2026-06 | **arXiv-only** | non-uniform width, reports KV memory/IO savings | the strongest overlap on "narrow middle saves cache bytes". |

B01 has **no device, no network, no privacy claim**, and its bytes axis is
**bytes/token written to a store** — which, per §0, **B01 cannot currently measure at
all**. That is disclosed, not hidden.

### 1.5 Family: recurrent / compressive memory, and semantic / latent codecs

| paper | year | verified venue | what it does | precise difference from B01 |
|---|---|---|---|---|
| **Compressive Transformer** | 2020 | **ICLR 2020** (DBLP `conf/iclr/RaePJHL20`) | compresses old activations into a coarser memory, trained | **trained** activation compression — again, "training is involved" is not B01's distinction. Sequence/time axis. |
| **Memorizing Transformers** | 2022 | **ICLR 2022** (DBLP `conf/iclr/WuRHS22`) | kNN retrieval into a non-differentiable (k,v) memory | full-width stored states. |
| **Leave No Context Behind (Infini-attention)** | 2024 | **arXiv-only** (DBLP `journals/corr/abs-2404-07143`, CoRR 2024) — heavily cited, **no peer-reviewed record found from this node** | fixed-size compressive memory in linear attention | fixed-size **recurrent** state; no persistable per-chunk latent, no rank measurement. |
| **MemoryLLM** | 2024 | **ICML 2024** (DBLP `conf/icml/WangGCJLYYLLYSM24`) | self-updating latent memory pool | memory as updated latents, not a rank-constrained residual stream. |
| **Matryoshka Representation Learning (MRL)** | 2022 | **NeurIPS 2022** (DBLP `conf/nips/KusupatiBRWSRHC22`). ⚠️ OpenReview title search returns **noise hits** (Federated-MRL NeurIPS 2024, ACL ARR submissions) — resolved via DBLP, which is why the per-family rule matters | trains embeddings whose **prefixes** are independently usable, so one representation serves many dimensionalities | **the origin of "train the representation to be truncatable"**, and MatryoshkaKV (§1.2) already carried it into KV caches. **B01 must not claim novelty for training-for-rank-truncatability.** Difference: MRL trains **task embeddings** with an explicit nested loss; B01 imposes a **hard architectural width constraint on a mid-stack LM state** and measures the resulting rank without any nested objective. |
| **VQ-VAE / discrete latent codecs** | 2017– | **not verified in this pass** (DBLP query returned no hit for my phrasing) | learn a discrete latent code as the representation | the general "learn the latent, then store the code" lineage. Recorded as an **unverified gap** (§5) rather than asserted. |
| **Cartridges** | 2025/26 | **ICLR 2026 Poster** (OpenReview `venueid = ICLR.cc/2026/Conference`, `Camera_Ready_Revision`; DBLP CoRR `abs-2506-06266`) | trains a small reusable per-corpus artifact offline, amortised over queries | the strongest "offline-compiled reusable representation" point; B01's four-arm gate (bottleneck × Read-LoRA × Write-LoRA) is in this design space. |

---

## 2. The two strongest collisions, stated head-on

**Neither is in the audit's list. Both were found in this pass, and both are more
dangerous to B01 than the four papers the audit named.**

### 2.1 KV-CAT (`arXiv:2605.05971`, ICML 2026 Workshop HiLD Poster) — owns B01's *framing*

KV-CAT's abstract states B01's motivating argument as its own thesis: post-hoc methods
operate on a fixed pretrained model, "so their effectiveness is fundamentally limited
by how well the model's internal representations can be compressed", therefore one must
"guide transformers toward compressible representations **during training**" — and it
**proves** a separation result to justify it, then delivers a **continued-pretraining**
procedure.

**Not preemption, for three reasons:**

1. **Concurrent** (2026-05-07; ~3 months, at the edge of the window — B01 should say
   "concurrent" and give the date rather than lean on the classification).
2. **Different axis and different mechanism.** KV-CAT masks **KV slots** with a
   stochastic train-time policy; B01 imposes a **hard, residual-free width constraint
   on the residual stream**. A slot-masking policy cannot produce the quantity B01
   reports (`dim99` of `h_j`), and a width constraint cannot produce KV-CAT's
   slot-budget curves.
3. **Different reported quantity, and this is B01's real edge.** KV-CAT's outcome is
   *"downstream compression methods get a better quality-budget tradeoff"* — i.e. it is
   **still measured through a compressor**. B01 reports an **intrinsic, compressor-free
   rank property** (`dim99` 1825 → 438 at 1B; 2790 → 467 at 3B) and **ΔNLL under
   plain rank truncation** (0.0304 → 0.0022 at rank 128), which is a claim about the
   representation itself. **Follow-up framing:** KV-CAT proves the need and gives a
   soft method; B01 asks whether a **hard architectural** constraint makes the
   low-dimensionality intrinsic — and whether the LM tax for that shrinks with scale
   (1B +5.9 % → 3B +4.7 % at `d512`, so far yes).

**Required:** cite KV-CAT wherever B01 states the post-hoc-limitation argument. **Do
not present that argument as B01's own.**

### 2.2 Variable-Width Transformers (`arXiv:2606.18246`, arXiv-only) — owns B01's *construction*

VWT trains an ×-shaped LM from scratch with a **narrow middle**, beats
parameter-matched uniform baselines on LM loss, saves **22 % FLOPs** and **15 % KV
cache memory/IO**, and reports that the bottleneck yields **qualitatively different
residual-stream representations**.

**Not preemption, and the differences are testable rather than rhetorical:**

1. **Concurrent** (2026-06-16, under two months).
2. **Residual-free vs residual-preserving.** B01's `BottleneckLayer` is
   `down → GELU → up` with **NO residual** (verified in
   `scripts/train_qwen_bottleneck_continued.py:112-133`). A narrowed-but-residual block
   does **not** force the layer *output* through a `d_bottle`-dimensional subspace;
   B01's does. **This is the mechanical reason the two get opposite-signed taxes**
   (VWT: better than baseline; B01: +4.5–8.5 % worse) and is the sharpest available
   differentiator.
3. **Objective and outcome variable.** VWT optimises **loss per FLOP** and reports
   LM loss / FLOPs / KV bytes. B01's outcome variables are **PCA `dim99` of the
   designated layer's output** and **ΔNLL under rank truncation** — a *cacheability*
   claim, not a *scaling* claim. Neither appears in VWT's abstract.
4. **What is cached and by whom.** VWT's KV saving is an inference-time byproduct of a
   narrower layer. B01's target is a **cross-query persistent store** of `h_j` read by
   the frozen upper stack.

**Required:** cite VWT, drop any claim of novelty for "mid-depth width bottleneck
trained into an LM changes the residual stream", and **state the residual-free
distinction explicitly and early** — a reviewer who knows VWT will otherwise assume
B01 is a worse-performing rediscovery. ⚠️ **A single ablation would settle it:
residual-free vs residual-preserving at the same `(j, d)`.** B01 does not currently
have that arm and should add it.

---

## 3. Must-NOT-claim list (binding on any B01 text)

Each line names the owner. This is a **citation-obligation list, not a scope ceiling**
(Rule 1, 2026-08-12 strengthening).

1. ❌ **First to pretrain a model whose cached state is a low-dimensional latent.**
   **MLA / DeepSeek-V2 owns this** (`2405.04434`), at production scale, in 2024.
   Also YOCO (NeurIPS 2024 Oral), CLA (NeurIPS 2024 Poster), TPA (NeurIPS 2025).
2. ❌ **First to argue that post-hoc compression is limited by the model's
   representations, therefore train for compressibility.** **KV-CAT owns this**
   (`2605.05971`, ICML 2026 Workshop HiLD Poster), with a proof. NSA (ACL 2025) owns
   the same meta-argument in the sparsity axis.
3. ❌ **First to train a mid-depth width bottleneck into an LM from scratch, or to
   report that it changes residual-stream representations.** **Variable-Width
   Transformers owns this** (`2606.18246`).
4. ❌ **First to compress the *feature-dimension* axis of a cache, or to train nested /
   truncatable ranks.** **MatryoshkaKV owns this** (ICLR 2025 Poster); the truncatable
   representation idea is **MRL's** (NeurIPS 2022); Palu (ICLR 2025 Poster) is the
   low-rank-projection counterpart.
5. ❌ **First to allocate a low-rank cache budget non-uniformly across layers.**
   **CARE owns this** (ICLR 2026 Poster); PyramidKV (COLM 2025) owns the post-hoc
   layer-wise budget.
6. ❌ **First to observe that a width bottleneck costs LM quality.** The mechanism is
   **Low-Rank Bottleneck in Multi-head Attention Models** (ICML 2020).
7. ❌ **First to insert a bottleneck into a Transformer LM and train through it.**
   **Sentence Bottleneck Autoencoders** (EMNLP 2021); Perceiver (ICML 2021) and
   Set Transformer (ICML 2019) for architectural latent bottlenecks.
8. ❌ **First to store intermediate-layer hidden states as a compressed memory and
   decompress on demand.** **SeDeM owns this** (`2608.00311`, concurrent).
9. ❌ **First to codec boundary activations at a layer cut.** **RAC owns this**
   (`2608.04991`, concurrent).
10. ❌ **Any storage / bytes-per-token saving claim, in any form.** **Blocked by
    B01's own verified code defect** (`STATUS.json:78-83`, re-verified this session):
    the store holds the **restored full-width** hidden, so `bytes/token` is
    **identical across arms today**. This is not a literature matter — a bytes claim
    would be *false about our own implementation*.
11. ❌ **"Fixed LM tax of 4–8.5 %" without stating it is TRAINING PPL from the
    16k-step from-scratch dim sweep.** It is **not** the `r_max` NLL gap
    (+0.16 / −0.19 / −0.13 % at 1B; +0.73 / +0.82 / +0.84 % at 3B)
    (`STATUS.json:67`).
12. ❌ **Any claim the tax shrinks with scale beyond 1B → 3B.** Two points only;
    **never tested at 8B** (`STATUS.json:25`). The two 8B endpoints
    (`outputs/qwenbott_funnel_L12_d512/final.pt`,
    `outputs/qwenbott_baseline_L12/final.pt`) exist on wzc1 and are **NEVER
    EVALUATED** — a repo-wide grep returns 0 hits outside the launch script
    (`STATUS.json:74`).
13. ❌ **Any depth claim that ignores the monotone depth tax.** L1 +4.2 % → L12 +9.6 %
    at fixed `d512` (`memory/bottleneck-layer-sweep-monotone.md`); `j` is a
    **trade-off point, not a tax minimum**.
14. ❌ **The word "funnel" without disambiguation.** Funnel-Transformer (NeurIPS 2020)
    pools the **sequence** axis; B01's launcher is literally named
    `qwenbott_funnel_L12_d512`, so the collision is invited by our own naming.
15. ❌ **Any downstream quality claim for the bottleneck arms.** No long-memory
    quality result exists yet — that *is* the next gate
    (`STATUS.json:next_gate_executable_20260814`).

---

## 4. Safe gap — one sentence, experimentally checkable

> **A residual-free feature-axis bottleneck trained into the residual stream at one
> designated depth makes that layer's output intrinsically low-dimensional — measurable
> as PCA `dim99` and as ΔNLL under plain rank truncation, with no compressor in the
> loop — at an LM tax that does not grow with model scale; and a memory that persists
> `d_bottle`-width latents (not restored full-width hiddens) then retains exact
> evidence on retrieval-closed long-context tasks at a bytes/token that no post-hoc
> codec on a stock model attains at equal quality.**

Clause-by-clause justification (each checked for absence in §1):

* **"residual-free"** — the one mechanical difference from VWT (§2.2), and the reason
  the layer *output* is confined to a `d_bottle` subspace. **Requires the
  residual-free-vs-residual-preserving ablation at matched `(j,d)`, which B01 does not
  yet have.**
* **"no compressor in the loop"** — the one measurement difference from KV-CAT (§2.1)
  and MatryoshkaKV. `dim99` and ΔNLL-vs-rank are properties of the representation,
  not of a chosen compressor.
* **"residual stream at one designated depth"** — distinguishes from MLA/CARE/TransMLA
  (per-head KV *inside* attention) and from Funnel-Transformer (sequence axis).
* **"tax does not grow with scale"** — this is **kill-gate condition 2 read as a
  prediction**. Currently supported by two points (1B +5.9 % → 3B +4.7 % at `d512`)
  and **must be tested at 8B**, where the two never-evaluated CPT endpoints already
  sit on disk.
* **"persists `d_bottle`-width latents (not restored full-width hiddens)"** — this is
  the **currently false half of the sentence**, and it is stated as the gap
  deliberately. Until the persist path exists, B01 has the **precondition** (low rank)
  and not the **headline** (persistable latent). The audit said exactly this at line
  119, and the code confirms it.

**Cheapest decisive next step, and it is not the four-arm gate.** Two 0-GPU/low-GPU
items dominate on information per hour:

1. **Fix the persist path** so the store writes `d_bottle` width. Without it, arms
   2/3/4 differ only in quality and the headline is untestable — spending 25–40 GPU-h
   before this is spending it on the wrong axis.
2. **Evaluate the two existing 8B endpoints** (`qwenbott_funnel_L12_d512` vs
   `qwenbott_baseline_L12`; `STATUS.json:70-76` records `eval_qcmem_locomo.py
   --bottleneck_ckpt` already loads a funnel-Qwen arm and the required
   `arch_meta.json` is present). This tests **kill-gate condition 2 at 8B** — the only
   kill condition currently answerable from assets already on disk.

---

## 5. Honest gaps in this adjudication

1. **Semantic Scholar unusable: HTTP 429 on every call** (2 attempts through
   `hy-proxy.woa.com:3128`). S2 is only ever a cross-check per the standing rule, so no
   verdict rests on it — but the intended cross-check **did not happen** and every
   venue call above is **single-authority**.
2. **arXiv API was rate-limited (`Rate exceeded`, HTTP 429) for the first ~15 minutes**
   and needed a retry-with-backoff client. Several searches returned **zero** entries
   for terms where that is implausible (e.g.
   `abs:"rank" AND abs:"truncation" AND abs:"hidden state" AND abs:"robust"`,
   `abs:"memory-ready"`), which is more likely throttling than true absence.
   **Re-run before submission.**
3. **DBLP `/search/publ/api` intermittently returned HTTP 500 / 503** (5 queries;
   recovered after raising the delay to 4 s). One query 500'd then succeeded on retry
   — a DBLP error is **never** evidence of "not in DBLP".
4. **`arXiv-only` entries whose venue I could NOT verify from this node**, listed so
   none is silently treated as unpublished:
   **DeepSeek-V2 / MLA** (`2405.04434` — CoRR only, which is startling for a paper of
   its influence and is the single venue call most worth re-checking),
   **Variable-Width Transformers** (`2606.18246`, CoRR 2026, OpenReview no match),
   **Through the Bottleneck** (`2607.23054`),
   **SeDeM** (`2608.00311` — **no DBLP record at all, 404**),
   **RAC** (`2608.04991` — **404**),
   **Infini-attention** (`2404.07143`),
   **TransMLA** (`2502.07864`), **LeanKV** (`2412.03131`),
   **Kimi Linear** (`2510.26692`), **CommVQ** (`2506.18879`).
   For the 2026-07/08 papers a missing DBLP record is **expected** (CoRR ingest lags
   ~4–8 weeks) and indicates recency, not low quality.
5. **Three papers must not be miscited as accepted main-conference papers:**
   **KV-CAT** = `ICML.cc/2026/Workshop/HiLD`, **no `Camera_Ready_Revision`** →
   *workshop poster*; **Q-Filters** = `ICLR.cc/2025/Workshop/SLLM` → *workshop*;
   **PromptDistill** = `aclweb.org/ACL/ARR/2025/October/Submission` → *ARR under
   review*. And **PyramidKV carries a `Rejected_Submission` record at ICLR 2025** —
   cite **COLM 2025** only.
6. **Title divergence between arXiv and venue record** on **Palu** (arXiv
   *"Compressing KV-Cache with Low-Rank Projection"* vs OpenReview *"KV-Cache
   Compression with Low-Rank Projection"*). Cite from the venue record; the same
   hazard `memory/venue-verify-acl-family-needs-anthology.md` warns about.
7. **Matryoshka (MRL) required a family switch to resolve.** OpenReview title search
   returned only noise (Federated-MRL NeurIPS 2024, two ACL ARR submissions); DBLP gave
   **NeurIPS 2022** `conf/nips/KusupatiBRWSRHC22`. Concrete instance of why the
   per-family rule exists.
8. **One family was NOT verified at all: discrete latent codecs (VQ-VAE and
   successors).** My DBLP phrasing returned no hit and I did not retry. §1.5 records
   it as unverified rather than asserting a venue. **A B01 submission must close this**
   — "learn a discrete code and store the code" is an obvious reviewer question.
9. **Abstracts read, full texts not read.** I fetched and read the arXiv abstract pages
   for **KV-CAT, Variable-Width Transformers, Through the Bottleneck, MatryoshkaKV,
   CARE, SeDeM, RAC**. Every other "what it does" column comes from title + venue
   record + repo notes. **Any B01 sentence turning on a specific number or control
   inside one of those papers must be checked against its PDF first** — especially the
   VWT residual-preserving claim in §2.2, which I inferred from "×-shaped ... maintain
   a constant width across all layers" phrasing and **have not confirmed from its
   architecture section**. If VWT turns out to be residual-free too, B01's sharpest
   differentiator weakens and §2.2 must be rewritten.
10. **No `.bib` entries emitted, deliberately** (`memory/tcodex-exec-no-dash-c-flag.md`):
    nothing enters a bibliography until venue-verified by family, and ten entries
    (§5.4) are not yet eligible.
11. **Cross-disk status not checked.** All B01 assets named here
    (`outputs/e2e_{1b,3b}_deltanll/`, `outputs/sembott_*`,
    `outputs/qwenbott_funnel_L12_d512/final.pt`, `outputs/qwenbott_baseline_L12/final.pt`)
    are recorded as **wzc1**. I did **not** verify presence on zwfy6, so any dispatch to
    `.73/.82/.104` must `ls`-confirm first
    (`memory/two-disk-rule-applies-to-main-too.md`).

---

## 6. Verdict

```
related_work_status: audited
novelty_verdict: NOT PREEMPTED, but B01's FRAMING and its CONSTRUCTION are each
                 independently owned by a concurrent paper (KV-CAT; Variable-Width
                 Transformers), and the general "pretrain a low-dimensional cached
                 latent" idea is owned outright by MLA/DeepSeek-V2 (2024).
                 The surviving claim is narrower than PROPOSAL.md and rests on two
                 mechanical distinctions: residual-FREE width constraint on the
                 RESIDUAL STREAM, and a COMPRESSOR-FREE rank measurement.
promotion: NOT YET -- the kill gate has not run, and its blocking dependency
           (d_bottle persist path) is a verified code defect, not a literature issue.
gpu_policy: novelty gate no longer blocks GPU. But the four-arm gate should NOT be the
            next GPU spend: fix the persist path (0-GPU code work) and evaluate the two
            existing, never-evaluated 8B endpoints first.
```

* **No paper found is 完全相同 / 抄袭.** The three nearest (KV-CAT, VWT, MatryoshkaKV)
  each differ on at least one load-bearing, *testable* axis — stochastic-slot-masking vs
  hard-architectural-width; residual-preserving vs residual-free; post-hoc projection on
  a frozen model vs formation during training. `already_dead_should_archive` is **not**
  warranted.
* **The audit's 不足 rating is discharged**: all five named families (§1.1, §1.2, §1.4,
  §1.5, plus §1.3 which the audit missed and which is the largest attribution risk) have
  named papers with verified venues and per-paper differences, plus a 15-item
  must-not-claim list and a one-sentence safe gap.
* **The audit's own required distinction is now *sharper* than "post-hoc vs
  formation"** — because formation is **not** unclaimed (MLA, NSA, DMC, YOCO, CLA, TPA,
  KV-CAT all train the property in). The distinction that survives is
  **which tensor** (residual stream vs per-head KV), **what constraint**
  (hard residual-free width vs stochastic masking / low-rank projection), and
  **what is measured** (intrinsic rank with no compressor vs a compressor's
  quality-budget curve).
* **What this pass did NOT do:** it did not make B01 promotable, and it did not touch
  the kill gate. It moved four items out of B01's claimable space (the framing → KV-CAT;
  the construction → VWT; feature-axis/nested ranks → MatryoshkaKV + MRL; per-layer rank
  allocation → CARE), and it surfaced one new required ablation (**residual-free vs
  residual-preserving at matched `(j,d)`**) without which VWT is hard to distinguish.
