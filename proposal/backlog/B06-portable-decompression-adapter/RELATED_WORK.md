# B06 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU, 0 ssh. Adjudication only — this file runs nothing and launches nothing.**

Closes the blocker `proposal/ready_queue.py:542-553` actually stats (`RELATED_WORK.md absent
(blocks PROMOTION; 0-GPU task)`) and answers the five collision families named at
`proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:96`.

**Audit rating: 不足 (insufficient).** Families demanded: *activation decompression adapter*;
*split-compute reconstruction*; *adapter transfer*; *intermediate self-distillation*;
*cross-codec portability*. The audit's own bar for the word **"portable"** is quoted verbatim
because §1 is scored against it:

> 「"portable" 至少要求多 task、多 compressor、多 model 或明确 layer/module transfer。」
> (audit line 96, backlog table, 安全边界 column)

---

## 0. Standing rules this adjudication obeys

1. **`memory/prior-work-differentiate-dont-abandon.md`** (user 2026-08-07). Preemption requires
   **完全相同 / 抄袭**, not overlap. Work within 2–3 months is **concurrent** and cannot preempt.
   **A direction dies from its own kill gate, never from a literature count.** Where a hit
   overlaps, the required action is differentiation or a defect-fixing follow-up.
2. **Venue verification is per family, not per API** (`memory/venue-verify-must-use-openreview-2026.md`,
   `memory/venue-verify-acl-family-needs-anthology.md`): OpenReview `venueid` for
   ICLR/NeurIPS/ICML; ACL Anthology + DBLP for ACL/EMNLP/NAACL/EACL **including Findings**;
   DBLP for systems venues. `arXiv-only` below means **"I could not verify a peer-reviewed venue
   from this node"**, never "no venue exists".
3. **No `.bib` entry is emitted here.** Per `memory/tcodex-exec-no-dash-c-flag.md`, nothing enters
   a bibliography until venue-verified by family.

### 0.1 Endpoint status from this node, 2026-08-15 (verbatim, so §5 is checkable)

| endpoint | status |
|---|---|
| `dblp.org/search/publ/api` | **200 OK**, intermittent HTML-error bodies → retried with backoff (4 tries) |
| `aclanthology.org` | **200 OK** — anthology IDs below were fetched and their `<title>` read |
| `api2.openreview.net/notes/search` | **200 OK** this session (the 403 `ChallengeRequiredError` recorded in `B07_SERVING_GATE_PREREG.md:265` and in A01's §5 did **not** reproduce today) |
| `api.semanticscholar.org` | **HTTP 429** on every call → **not used** |
| `export.arxiv.org/api/query` | 200, but rate-limits to 429 / read-timeout under bursts → backoff |

⚠️ **DBLP `venue` strings are not trustworthy for the ACL family.** Measured this session: DBLP
prints `venue=EACL` for Embedding Recycling whose DOI is `10.18653/v1/2023.findings-eacl.145`
(**Findings**), `venue=EMNLP` for XC-Cache whose DOI is `2024.findings-emnlp.896` (**Findings**),
and `venue=ACL` for LLMLingua-2 whose DOI is `2024.findings-acl.57` (**Findings**). Every ACL-family
row below therefore reports the **Anthology ID**, which is the authority, and marks Findings
explicitly. This is exactly the failure mode `memory/venue-verify-acl-family-needs-anthology.md`
was written about.

---

## 1. What B06 claims RIGHT NOW, and what it is NOT allowed to claim

Read from `STATUS.json.claim_scope_discipline` and `PROPOSAL.md`, not from the direction's name.

**The measured claim (single instrument, exactly paired, re-derived from raw judge caches):**

> On a **retrieval-free** HCache-style read path at Qwen3-8B, `j=12`, a rank-32 self-distilled
> LoRA moves LoCoMo Judge$_{1:4}$ from **16.69 → 39.81** (**+23.12 pp**, n=1540, McNemar
> b=414/c=58, exact two-sided p=2.6e-67, paired item bootstrap 95% CI [20.58, 25.58]).
> Because `no_retrieval=true` in **both** arms (`scripts/eval_qcmem_locomo.py:914` sets
> `no_retrieval = (args.baseline in ("kvdirect","hcache"))`), no selector is in the path, so the
> contrast isolates the adapter.

**The interpretation B06 wants:** the adapter is a *shared mid-recompute readout / decompression
repair*, not a CoMem-retrieval-pack specialisation.

**The word the audit is attacking is "portable".** Scored against the audit's own four-way bar:

| audit requirement | B06 status today | evidence |
|---|---|---|
| **multi-task** | ❌ **NOT MET.** One task (LoCoMo). BABILong/RULER/LongEval are listed as *next steps*, not results. | `PROPOSAL.md` 下一步 item 2 |
| **multi-compressor** | ❌ **NOT MET.** `condition_3_status: "UNTESTED -- no second compressor has been run"`. One compressor (HCache-style mid-layer residual). | `STATUS.json.kill_gate` |
| **multi-model** | ❌ **NOT MET.** One backbone (Qwen3-8B). A Llama-3 distill dir exists (`outputs/qcmem_distill_llama3_j12_r32_4k/final`, cited in B08's `blocking_asset_finding`) but **no B06 arm has been run on it**. | STATUS.json has no such measurement |
| **explicit layer/module transfer** | ⚠️ **PARTIAL, and it is the ONE that is actually met.** The adapter was distilled for the **CoMem retrieval-pack** read path and is evaluated on the **HCache pack-everything** read path at the same `j=12`. That is a *read-path* transfer with the layer held fixed, activated by `--force_lora_with_baseline` (`eval_qcmem_locomo.py:932-937`, which otherwise **drops** the LoRA for `baseline=hcache`). | `established_measurements.single_variable_verified` |

**So B06 currently satisfies 1 of the audit's 4 disjuncts, and the one it satisfies is the
weakest reading of it** (same model, same layer, same compressor family, different *pack policy*).
That is the honest answer the audit asked for. It is **not** a kill: the direction's own kill
conditions are about *whether the gain survives*, not about how many axes have been run yet.

⚠️ **An additional risk the audit did not raise, already on disk:** kill condition 1
(«只在 LoCoMo open-domain category 有益») is **live**. `status/PAPERA_RESULTS_CONSOLIDATED.md:401`
records cat4 moving 23.31 → 55.77 = **+32.46**, called 「最大驱动」 there; cat4 is the largest
bucket at **n=841** = 54.6% of the 1540 (`:177`).
(⚠️ Two pointer defects found while checking this, both recorded so the next agent does not think
the evidence moved: `STATUS.json.kill_gate.condition_1_status` cites line **403**, but the string is
at **401** in this session's checkout; and the repo names cat4 **`open_domain`** at `:401` but
**`single-hop`** at `:177` — resolve against `scripts/eval_qcmem_locomo.py`'s category map before
publishing a per-category table.) A per-category breakdown on the corrected instrument is **0 GPU** and
has not been computed. If the gain is cat4-only, condition 1 fires and the word "portable" dies
from our own gate — which is the correct way for it to die.

---

## 2. Named closest collisions, by the audit's five families

### 2.1 Family: activation decompression adapter

| work | year | venue (verified) | what it does | precise difference from B06 |
|---|---|---|---|---|
| **SeDeM: Selective Decompression of Hidden-State Memories for Long-Context QA** | 2026-07-31 | `arXiv:2608.00311`, **arXiv-only** | LLM extracts hidden states at a chosen intermediate layer → lightweight compressor stores memory blocks → query-conditioned selector picks blocks → **decompressor expands selected blocks into states compatible with an intermediate decoder layer**. 1B/3B, four long-context QA sets. | **THE CLOSEST in this family, and it is CONCURRENT** (2026-07-31 vs B06's measurement 2026-08; same month → per rule 0.1 cannot preempt). Differences that are load-bearing: (a) SeDeM's decompressor is trained **jointly with its own compressor** and is a *component of one pipeline*; B06's question is whether a **single adapter trained under one write/read policy still repairs a different one**, which requires ≥2 compressors — the axis SeDeM does not have and B06 has not run either. (b) SeDeM's selector is *in* the path; B06's decisive property is `no_retrieval=true` in both arms. (c) SeDeM optimises QA score; B06 is a **transfer diagnostic**. **Not preempting, but it does mean "we invented decompression adapters" is unclaimable.** |
| **RAC: Reference-Aware Activation Compression for Communication-Efficient Split LLM Inference** | 2026-08-05 | `arXiv:2608.04991`, **arXiv-only** | Codec on **boundary hidden states** of a split (local-head / cloud-middle / local-tail) LLM: retrieves exact-token historical spans for prefill uplinks, grouped affine alignment + calibrated residual quantisation, causal decode-reference predictors. 3 models, 9 model-link pairs. | Overlap = "a trained module reconstructs intermediate activations". **Objective is different and that is decisive:** RAC minimises **wire bytes across a network link** under privacy constraints; its quality target is *fidelity to the uncompressed tensor*. B06 does not transmit anything and does not target fidelity — it targets **downstream task accuracy of a frozen upper stack**. Also RAC's 3-model / 9-pair sweep is precisely the breadth B06 lacks: **RAC is the standard B06 will be held to on the "multi-model" disjunct.** |
| **HCache: Fast State Restoration in LLM Serving** | 2025 | **EuroSys 2025**, DBLP `conf/eurosys/GaoCS25`, DOI `10.1145/3689031.3696072` (arXiv `2410.05004`) | Stores **intermediate activations** instead of KV and restores state by recomputing the remainder. Training-free. | **This is B06's own control arm**, not a competitor: `eval_qcmem_locomo.py:56` cites `2410.05004` as the `hcache` baseline. HCache claims *bit-level state restoration is cheaper*; it never asks whether a **trained** module is needed. **B06's contribution can only be the delta over this training-free path**, which is exactly what the 16.69→39.81 contrast measures. Must be cited as the substrate, and B06 must never present the HCache read path as its own design. |
| **PromptDistill** | 2025-03 | `arXiv:2503.23274`; DBLP **CoRR 2025** → **arXiv-only** | Query-based selective token retention in intermediate layers. | Selects *which* intermediate states to keep; adds **no trained repair module**. Orthogonal: it is a write-side policy, B06 is a read-side repair. |

### 2.2 Family: split-compute reconstruction

Covered by **RAC** (2.1) plus the split-computing line already adjudicated for B05
(`proposal/backlog/B05-semantic-handoff-phase-diagram/RELATED_WORK.md` §1.5: Salted Inference —
DBLP **HotMobile 2024**; SplitTracr — DBLP **ICPE 2025**). **Not re-litigated here.** The
one-sentence difference: split computing's dependent variable is **bandwidth/privacy under a
device–server partition**; B06 has no device, no link, and no privacy claim, and its dependent
variable is a judge score.

### 2.3 Family: adapter transfer

| work | year | venue (verified) | what it does | precise difference from B06 |
|---|---|---|---|---|
| **Trans-LoRA: towards data-free Transferable Parameter Efficient Finetuning** | 2024 | **NeurIPS 2024**, DBLP `conf/nips/WangGCAOFK24` (arXiv `2405.17258`) | Transfers a LoRA from a **source base model to a different target base model**, data-free, via synthetic-data distillation + a discriminator. | **This is the strongest *named* prior art on the phrase "portable adapter", and it defines the bar for B06's unmet "multi-model" disjunct.** Difference: Trans-LoRA transfers **across backbones with the task fixed**. B06 (as measured) holds the **backbone fixed** and transfers **across cache/read policies**. Those are different axes — but note the consequence: **if B06 ever runs the Qwen→Llama-3 arm, Trans-LoRA becomes a required baseline, not merely a citation.** |
| **LoRA-X: Bridging Foundation Models with Training-Free Cross-Model Adaptation** | 2025 | **ICLR 2025**, OpenReview `venueid=ICLR.cc/2025/Conference` (Poster), id `6cQ6cBqzV3`; DBLP `conf/iclr/FarhadzadehDBP25` | Training-free cross-model adapter transfer by projecting the adapter into the target model's subspace. | Same axis as Trans-LoRA (**across models**), training-free. Same disposition: **a future baseline for a multi-model B06 arm; not a collision with what B06 measured.** |
| **Embedding Recycling for Language Models** | 2023 | **Findings of EACL 2023**, Anthology **`2023.findings-eacl.145`** (DBLP prints `venue=EACL` — wrong, DOI says findings) | Caches **intermediate-layer activations** across tasks and **adapts the later layers** to consume them. | ⚠️ **This is the oldest and most under-cited near-hit for B06's actual mechanism** — cached mid-layer activations + trained upper-layer adaptation. Difference: Embedding Recycling recycles across **tasks/epochs of encoder fine-tuning** for throughput, on encoders, and its adaptation is full-layer fine-tuning, not a rank-32 LoRA on a frozen decoder read path under a paired judge. **B06 must cite it and must not claim novelty for "adapt later layers to consume cached activations".** (It is already in `paperA/qcmem.bib` as `embeddingrecycling`.) |
| **XC-Cache: Cross-Attending to Cached Context** | 2024 | **Findings of EMNLP 2024**, Anthology **`2024.findings-emnlp.896`** (DBLP prints `venue=EMNLP` — wrong) | Trains cross-attention modules so a decoder consumes cached encoder context instead of in-context text. | Trained module consuming a cache — overlap. But XC-Cache **changes the architecture** (adds cross-attention) and its cache is an **encoder** output. B06 adds no module: it is a LoRA on the *existing* upper layers of a single decoder. |
| **ReadOnce Transformers** | 2021 | **ACL-IJCNLP 2021 (main, long)**, Anthology **`2021.acl-long.554`**, DOI `10.18653/v1/2021.acl-long.554` | Compute a **reusable document representation once**, reuse it across queries. | The origin of "encode once, reuse across queries". No adapter portability question, no depth axis. Citation, not collision. |

### 2.4 Family: intermediate self-distillation

| work | year | venue (verified) | what it does | precise difference from B06 |
|---|---|---|---|---|
| **Self-Distillation Bridges Distribution Gap in Language Model Fine-Tuning** | 2024 | **ACL 2024 (main, long)**, Anthology **`2024.acl-long.58`**, DOI `10.18653/v1/2024.acl-long.58`; DBLP `conf/acl/YangPFWCZL24` | Self-distillation (teacher = the model itself) to close a train/serve distribution gap during fine-tuning. | Establishes "self-distillation repairs a distribution gap" as **prior art at the level of the idea**. B06's gap is *architectural* (states arrive from a cache at depth j, not from the layers below), not a data-distribution gap. **"Self-distillation is our idea" is unclaimable; "self-distillation repairs a cache-boundary readout gap, and the repair is reusable across read policies" is the only live framing.** |
| **In-context Autoencoder (ICAE)** | 2024 | **ICLR 2024**, DBLP `conf/iclr/00010WWCW24` (arXiv `2307.06945`, v4 comment: "Final camera ready for ICLR'24"). ⚠️ OpenReview text search returned only unrelated `CONTEXT` conference hits for this title on both attempts → **venue rests on DBLP + the arXiv camera-ready comment, not on `venueid`** | Autoencode long context into memory slots, decoder conditions on them. | The canonical compress→decode-into-a-decoder pipeline. Trains **compressor and decoder together**; B06's whole question is *reuse of the decoder-side repair when the compressor changes.* |
| **Activation Beacon** | 2025 | **ICLR 2025**, OpenReview `venueid=ICLR.cc/2025/Conference` (Poster); DBLP `conf/iclr/Zhang0XSYD25` | Learned activation condensation for long-context compression, trained end-to-end. | Same disposition as ICAE. |
| **Cartridges: Lightweight and general-purpose long context representations via self-study** | 2026 | **ICLR 2026**, OpenReview `venueid=ICLR.cc/2026/Conference` (Poster); arXiv `2506.06266` | Trains a per-corpus KV cache ("Cartridge") offline with a **context-distillation** objective; amortises across queries. | Overlap = "train something offline so the cache is cheap to serve, using self-generated supervision". **Difference is which side is trained:** Cartridges trains **the cache** and keeps the model frozen; B06 trains **the model's readout** (a LoRA) and keeps the cache production rule fixed. This is a clean, statable dichotomy and B06 should state it in exactly these terms. |
| **LLoCO: Learning Long Contexts Offline** | 2024 | **EMNLP 2024 (main)**, Anthology **`2024.emnlp-main.975`**, DOI `10.18653/v1/2024.emnlp-main.975` | Compress context offline + **LoRA finetune the reader** to consume the compressed form. | ⚠️ **The closest thing to "LoRA that teaches a reader to consume a compressed context" that has a peer-reviewed venue.** Difference: LLoCO's LoRA is trained *per domain* on the compressed representation of **one** compressor (AutoCompressor) and is never tested for transfer to a different compression path. **B06's claim reduces to "LLoCO's LoRA, but portable across read policies" — which means the portability measurement IS the contribution, and without it B06 is a re-run of LLoCO's setup.** |

### 2.5 Family: cross-codec portability

**This is the family where I could not find a direct hit, and that is B06's opening — with a caveat.**

Searches run (arXiv API, all fields, exact phrases in quotes):
`all:"cache format" AND all:"adapter" AND all:"transfer"` → 0;
`abs:"same adapter" AND abs:"different cache"` → 0;
`all:"one adapter" AND all:"multiple" AND all:"compression rates"` → 0;
`all:"decompression" AND all:"adapter" AND all:"transfers" AND all:"tasks"` → 1 hit, unrelated
(GPU floating-point lossless compression, `2511.04140`);
`all:"universal decoder" AND all:"multiple codecs"` → 0;
`all:"codec" AND all:"portable" AND all:"representation"` → 2 hits, both unrelated (ERA5 climate
compression `2405.03376`; program revectorization `1902.02816`);
`all:"compression-aware" AND all:"adapter" AND all:"language model" AND all:"generalization"` → 1
hit, unrelated (multimodal retrieval, `2602.19091`);
`all:"KV cache" AND all:"repair" AND all:"LoRA"` → **429, NOT RUN** (see §5).

**Caveat that must not be smoothed over:** three of the intended queries in this family died on
arXiv 429 / read-timeout and were not retried to completion (§5 item 2). The absence of a
cross-codec-portability hit is therefore **weaker evidence than the other four families**, where
searches completed. And note the adjacent literature is *already doing per-method repair*:

| work | year | venue (verified) | why it is adjacent |
|---|---|---|---|
| **AgentKVShift: Efficient KV Cache Reuse for Agentic Memory Systems** | 2026-05 | `arXiv:2607.21604`, **arXiv-only** | Training-free, probe-guided **KV residual correction** per retrieved memory unit; explicitly observes that reuse methods "designed for RAG-style raw passages **degrade on structured agentic memories**". **That sentence is a cross-codec-portability *finding* stated as a motivation** — i.e. someone has already noticed that repair does not transfer across cache-content types. B06 must cite it and position its own claim as the *positive* direction (a trained repair that **does** transfer) rather than as the first observation of the phenomenon. |
| **CacheClip** | 2025-10 | `arXiv:2510.10129`, **arXiv-only** | Auxiliary-model-guided selective recomputation to restore inter-chunk attention. Repair, but training-free and method-specific. |
| **Cache-Craft** | 2025 | **PACMMOD / SIGMOD 2025**, DBLP `journals/pacmmod/AgarwalSMMGSKYS25`, DOI `10.1145/3725273` | Decides *which* chunk-cache to reuse and recomputes a small fraction to fix quality. Repair by **recomputation**, not by a trained module. **The right systems-side foil for "why not just recompute?"** |
| **CacheBlend** | 2025 | **EuroSys 2025**, DBLP `conf/eurosys/YaoLLRCZD0J25`, DOI `10.1145/3689031.3696098` | Selective recomputation of a token subset to fuse non-prefix caches. Same foil; already a `--baseline` in our own harness (`eval_qcmem_locomo.py:764`). |

---

## 3. MUST-NOT-CLAIM list (binding on any B06 writeup)

Each item names the work that forecloses it.

1. ❌ **"We introduce a decompression adapter for cached intermediate states."**
   Foreclosed by **SeDeM** (`2608.00311`, concurrent) and **ICAE** (ICLR 2024).
2. ❌ **"We are the first to adapt later layers to consume cached mid-layer activations."**
   Foreclosed by **Embedding Recycling** (Findings-EACL 2023, `2023.findings-eacl.145`).
3. ❌ **"We introduce a LoRA that teaches a reader to consume a compressed context."**
   Foreclosed by **LLoCO** (EMNLP 2024, `2024.emnlp-main.975`).
4. ❌ **"Self-distillation to repair a train/serve mismatch is our contribution."**
   Foreclosed by **Self-Distillation Bridges Distribution Gap** (ACL 2024, `2024.acl-long.58`).
5. ❌ **"Portable."** Unqualified, this is foreclosed **by our own audit bar and our own data**:
   1 task, 1 compressor, 1 model. Permitted phrasing is exactly and only:
   *"transfers across read policies at fixed backbone and fixed depth"*.
6. ❌ **"Adapter transfer across models."** Not measured at all, and **Trans-LoRA** (NeurIPS 2024)
   + **LoRA-X** (ICLR 2025) own that axis with baselines B06 has not run.
7. ❌ **Any number on the n=1986 blended scale** (13.29 / 31.17 / +17.88). Retracted
   2026-08-10, `paperA/ERRATA_LOCOMO_MIXED_INSTRUMENT_20260810.md`; the cat-5 refusal-regex term
   moves the *wrong way* for the treatment arm (7→6), so the blend does not merely shrink the
   effect, it adds a term of opposite sign.
8. ❌ **"The gain is not concentrated in one category."** Not computed on the corrected
   instrument; the old blended breakdown shows cat4 (55% of judged items) supplying +32.46.
9. ❌ **"Our repair beats recomputation-based repair."** No arm was run against **Cache-Craft**
   (SIGMOD 2025) or **CacheBlend** (EuroSys 2025) on a matched budget.
10. ❌ **"First to observe that cache repair does not transfer across cache content types."**
    **AgentKVShift** (`2607.21604`) states it as motivation.
11. ❌ **Cross-table comparison of these numbers to any `iter_bm25` retrieval arm.**
    Both B06 arms use `selector=bm25`; it is inert here (`no_retrieval=true`) but the comparison is
    off-protocol per `memory/qcmem-eval-selector-iterbm25.md`.

---

## 4. Safe residual claim — one falsifiable sentence

> **A rank-32 self-distillation LoRA trained on the CoMem retrieval-pack read path at depth
> j=12, applied unchanged to a retrieval-free HCache-style pack-everything read path on the same
> backbone and the same depth, recovers +23.12 pp LoCoMo Judge$_{1:4}$ (paired, p=2.6e-67) —
> i.e. the repair is a property of the depth-j readout interface and not of the retrieval pack
> it was trained with.**

**How to falsify it, in the order the checks get cheaper:**

1. **0 GPU, do this first.** Per-category breakdown on the corrected Judge$_{1:4}$ instrument.
   If the +23.12 is cat4-only, the "interface not pack" reading collapses into "open-domain
   dialogue only" and **kill condition 1 fires.**
2. **0 GPU.** Report all **three** same-instrument noLoRA replicates — 10.13 (canonical),
   15.45 (older local run), 16.69 (B06 control) — a **6.6 pp** spread on the *same* instrument.
   The effect survives that, but a write-up quoting one number and calling the rest "drift"
   is understating its own measurement noise.
3. **~0.9 GPU-h/arm (measured).** A second compressor. If the adapter does not transfer at all,
   **kill condition 3 fires** and the sentence above loses the word "interface".
4. **Unknown, needs a 1-cell timing.** A second task family (RULER / BABILong / LongEval).

**Note the asymmetry:** clauses 1 and 2 are free and can kill the claim; clauses 3 and 4 cost GPU
and can only broaden it. **The free ones must run first.** That ordering is the actual
recommendation of this file.

---

## 5. Honest gaps in this adjudication

1. **Semantic Scholar returned HTTP 429 on every attempt** and was not used. Where a row says
   `arXiv-only`, exactly one authority was consulted (DBLP CoRR and/or the arXiv comment field).
2. ⚠️ **Three cross-codec-portability queries died on arXiv 429 / read-timeout and were NOT
   retried to completion**: `all:"KV cache" AND all:"repair" AND all:"LoRA"`,
   `all:"CacheBlend" AND all:"adapter"`,
   `all:"cross-method" AND all:"cache reuse" AND all:"generalization"`.
   **§2.5's "no direct hit" is therefore the weakest finding in this file** and must be re-run
   before any submission. I am flagging it rather than presenting the family as cleanly open.
3. ⚠️ **ICAE's venue rests on DBLP + the arXiv v4 comment, not on OpenReview `venueid`.** Two
   OpenReview text searches for that title returned only unrelated hits from the `CONTEXT`
   conference series. Per rule 0.2 that is a **partial** verification and is labelled as such.
4. ⚠️ **DBLP's `venue` field mislabels Findings papers as main-conference** (three measured cases,
   §0.1). Anthology IDs are given for every ACL-family row; **anyone converting this file into
   `.bib` must copy the Anthology ID, not the DBLP `venue` string.**
5. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted on this node and this task
   forbade ssh. The canonical HCache `judge_cache.jsonl` (needed to confirm the 10.13 conversion)
   is zwfy6-resident and was **not** read here; per `memory/two-disk-rule-applies-to-main-too.md`
   the honest statement is **"not checked on zwfy6"**, never "absent".
6. **No arXiv-vs-camera-ready diff was performed** for any peer-reviewed row. Per
   `memory/venue-verify-acl-family-needs-anthology.md` that diff is required before citing a
   camera-ready claim as if the arXiv text said it.
7. **SeDeM and RAC were adjudicated from abstracts only** (fetched this session from the arXiv API).
   Neither full text was read. For two papers this close, the differentiation in §2.1 must be
   re-checked against their method sections before it is written into a paper.

---

## 6. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED for the NARROWED claim; the word
         "portable" is NOT yet earned, and the two cheapest tests that could kill it
         are 0 GPU and have not been run
related_work_status: audited
already_dead_should_archive: NO
```

- **No candidate is 完全相同 / 抄袭.** The two closest (**SeDeM** `2608.00311`, **RAC**
  `2608.04991`) are both **2026-07/08 = concurrent** and both differ on a load-bearing axis
  (joint-pipeline component vs cross-policy transfer; wire bytes vs task accuracy).
  The two closest *peer-reviewed* hits (**LLoCO** EMNLP 2024, **Embedding Recycling**
  Findings-EACL 2023) establish the **setup** as prior art, which forces B06's contribution to be
  the **portability measurement itself** — not the mechanism.
- **The direction is not killed by literature.** It is **narrowed by its own audit bar**: 1 of 4
  disjuncts met, and the one met is the weakest.
- **Promotion is not warranted yet.** Promotion requires the free per-category read-out (which can
  fire kill condition 1) plus at least a second compressor (kill condition 3). Both are named in
  `PROPOSAL.md`'s own success conditions; nothing here changes them.

### 6.1 ⚠️ SCHEDULER TRAP — do NOT copy the `related_work_status: audited` line above into `STATUS.json`

**`STATUS.json` was deliberately left UNMODIFIED for all of B06/B07/B08.** Measured this session by
importing `proposal/ready_queue.py` and calling `read_one()` on temp copies in `/tmp` (**no repo
file was touched**):

| proposal | today | if `related_work_status: "audited"` were appended to STATUS.json |
|---|---|---|
| **B06** | `ready_cpu` | ⚠️⚠️ **`ready_gpu`** |
| B07 | `ready_cpu` | `ready_cpu` (held by `blocking_dependency`) |
| B08 | `ready_cpu` | `ready_cpu` (held by `prior_gate`) |

Mechanism: `ready_queue.py:203-209` — `NOVELTY_VERDICT_KEYS` includes `related_work_status`, and
`VERDICT_CLEARED` includes the literal string `"audited"`. So the one-word append flips
`novelty_checked` to `True`, and **B06 has no live blocker left to hold it**, so it would be
promoted to the front of the GPU queue **by an agent-written field, with no adversarial review of
this file.**

That is exactly the failure mode in `memory/a-declared-lifecycle-is-not-an-adjudicated-one.md`
(2026-08-14: *"agent 自写 lifecycle: ready_gpu ≠ gate 通过审查"*). The `related_work_status: audited`
line in §6 is **this file's own self-assessment for a human reader**, not a scheduler token.

**Whoever promotes B06 must do so deliberately**, and should note that this file's §4 argues the
next B06 action is **0 GPU** (the per-category read-out, which can *kill* the claim) — so flipping
it to `ready_gpu` would send the next agent to spend cards on the generalisation legs **before** the
free test that could make them pointless.
