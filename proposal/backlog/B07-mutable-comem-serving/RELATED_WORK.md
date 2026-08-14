# B07 — RELATED WORK / NOVELTY ADJUDICATION

**Written 2026-08-15. 0 GPU, 0 ssh. Adjudication only — this file runs nothing and launches nothing.**

Closes the blocker `proposal/ready_queue.py:542-553` stats (`RELATED_WORK.md absent`) and the one
`B07_SERVING_GATE_PREREG.md:256-277` names as **"B07's true critical path"**.

**Audit rating: 不足 (insufficient)**, `RELATED_WORK_GAP_AUDIT_20260808.md:97`. Families demanded:
*prefix/KV caching; paged/disaggregated KV; versioning/invalidation; memory tiering; reuse-aware
admission; incremental recompute.* The audit's specific instruction (line 97 + 133-152):

> 「需要逐 feature systems collision table。」
> …「B07/B08 不能依靠"可版本化/可更新/分层 memory"这一功能清单主张新颖性。必须给出：
> 与现有系统逐 feature 差异；端到端质量/latency/bytes；stale/conflict failure；明确 workload
> (尤其跨 query reuse)。」

§2 is that table, one row per feature, in the audit's order. **Warning to the reader: this is the
most collided direction in the repository, and §3 is correspondingly long. That is the correct
outcome, not a failure of the search.**

---

## 0. Standing rules and endpoint status

1. **`memory/prior-work-differentiate-dont-abandon.md`** (user 2026-08-07): the bar is
   **完全相同 / 抄袭**, not overlap; 2-3 months = **concurrent**, cannot preempt; a direction dies
   **only from its own kill gate**. `B07_SERVING_GATE_PREREG.md:271-277` already anticipates this
   file and states it: *"Do not upgrade that triage line into a death certificate."*
   **⚠️ KV-cache serving is a hot area. Collision COUNT is not the verdict. The verdict is whether
   any single system does the same thing.**
2. **Venue verification by family.** B07's collisions land overwhelmingly in **systems venues**
   (SOSP / OSDI / ATC / EuroSys / FAST / NSDI / SIGCOMM / MLSys / SIGMOD), so per the task's own
   rule the authority is **DBLP**. ICLR/ICML/NeurIPS rows use OpenReview `venueid`.
   `arXiv-only` = **"I could not verify a peer-reviewed venue from this node"**.
3. **Endpoints, 2026-08-15, verbatim** (this *contradicts* the note frozen in
   `B07_SERVING_GATE_PREREG.md:264-268`, which recorded OpenReview 403 and Anthology timeout on
   2026-08-14 — both were **reachable today**, so the prereg's stated blocker on venue
   verification is **now discharged**):

| endpoint | status today |
|---|---|
| `dblp.org/search/publ/api` | **200** (intermittent HTML bodies → 4-try backoff) |
| `api2.openreview.net/notes/search` | **200** — no `ChallengeRequiredError` today |
| `aclanthology.org` | **200** |
| `api.semanticscholar.org` | **429 on every call → unused** |
| `export.arxiv.org/api/query` | 200, bursts → 429 / read-timeout |

---

## 1. What B07 claims RIGHT NOW

`PROPOSAL.md` lists **five features** (concurrency, versioning, incremental edit,
HBM/CPU/NVMe/network tiering, reuse-aware Write-vs-replay admission). Per the audit that list is
**not a novelty claim**, and B07's own prereg already agreed and collapsed it to one killable
sentence (`B07_SERVING_GATE_PREREG.md:13-32`):

> **B07 serving thesis.** The CoMem depth-residual store's per-query advantage over a matched
> `j=0` top-12 raw-text replay — measured single-stream at **246.4 ms** (128k store, cpu-pinned
> tier, G=1: comem 688.5 ms vs j0 934.9 ms, `paperA/artifacts/p1_8_serving/p1_8_serving_aggregate.json`
> `cells["128k|cpu"].per_G["1"]`) — **survives concurrency**, i.e. is still present, resolvable,
> and same-signed with 8 requests in flight.

**Two facts about B07's state that this file must not paper over** (both from its own
`STATUS.json`, and both bear on how the collision table should be read):

1. **`established_measurements._status`: "NONE FOR B07'S OWN CLAIMS."** No B07 experiment has ever
   run. There is no concurrency, versioning, edit, or tiering measurement.
2. **`blocking_dependency`: every mechanism in 「关键设计」 is a system that does not exist yet.**
   Version/content hashes, compatibility hash, fail-closed stale objects, the overlap dependency
   graph, mixed-version fallback — **none is implemented**. `lifecycle_why_20260814` confirms the
   concurrency axis is absent from `scripts/bench_p1_8_serving_curve.py` (`_serve_comem:226`,
   `_serve_j0:289`, no thread/pool/TTFT/p95/p99 code).

**Consequence for §2: for most features, the honest entry in the "our measured difference" column
is "we have no measurement, and the prior system does."** Written that way on purpose.

---

## 2. Per-feature systems collision table

Six sub-tables, one per family the audit named. **Column 4 is the one the audit actually asked
for and is where the differences have to be earned.**

### 2.1 Feature: prefix / KV caching (reuse of an exact shared prefix)

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **vLLM / PagedAttention** | 2023 | **SOSP 2023**, DBLP `conf/sosp/KwonLZ0ZY0ZS23`, DOI `10.1145/3600006.3613165` | Paged KV allocation, copy-on-write sharing, continuous batching. The substrate every later system assumes. | Reuse requires an **exact prefix match**; no notion of a persisted mid-depth object; no versioning. | **NONE — this is the baseline platform B07 must be measured *inside*, not against.** B07's own artefacts do **not** use continuous batching (`A02_STORAGE_READCOMPUTE_VERDICT.md:297-299`: "single-query, batch=1, no continuous batching or paged attention"), which is *exactly why* the prereg's K1 exists. |
| **SGLang / RadixAttention** | 2024 | **NeurIPS 2024**, OpenReview `venueid=NeurIPS.cc/2024/Conference` (poster); DBLP `conf/nips/ZhengYXS0YCKSGB24` | Radix-tree prefix cache with LRU eviction + a frontend language. | Token-level KV only; no mid-depth activation object; no explicit version/invalidation API. | Object is different (h_j residual vs KV). **But note: SGLang's radix tree already gives prefix-level sharing under concurrency — the thing B07 has never measured.** |
| **Prompt Cache** | 2024 | **MLSys 2024**, DBLP `conf/mlsys/GimCLSK024` | Modular attention reuse via a **schema** declaring reusable prompt segments at arbitrary positions. | KV-level; requires a schema; no update/invalidate semantics; no tiering. | Different object; B07's chunking is content-derived, not schema-declared. |
| **CachedAttention** | 2024 | **USENIX ATC 2024**, DBLP `conf/usenix/GaoHSKJDYYZ24` (arXiv `2403.19708`, "AttentionStore") | Hierarchical KV store for multi-turn conversation reuse, with prefetch/eviction across HBM/DRAM/SSD. | KV, append-only turns; no edit-in-place, no version graph. | ⚠️ **Closest on the *storage-hierarchy-for-reuse* axis and it is peer-reviewed at ATC.** B07 has **no** measurement here; CachedAttention has a full one. |
| **HCache** | 2025 | **EuroSys 2025**, DBLP `conf/eurosys/GaoCS25`, DOI `10.1145/3689031.3696072` | **Stores intermediate activations rather than KV** and recomputes the rest to restore state. | Training-free; no concurrency-vs-quality study; no versioned edits. | ⚠️⚠️ **THE CLOSEST SYSTEM TO B07's OBJECT, FULL STOP.** HCache already established that persisting mid-layer activations is a *serving* win. B07's object is not novel; only the **CoMem variant (depth-j residual + retrieval pack + trained Read repair) under concurrency** could be, and that is unmeasured. **This system alone forbids "we introduce activation-level caching for serving."** |

### 2.2 Feature: paged / disaggregated KV, and position-independent reuse

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **DistServe** | 2024 | **OSDI 2024**, DBLP `conf/osdi/ZhongLCHZL0024` | Disaggregates prefill from decode across GPUs for goodput under TTFT/TPOT SLOs. | No cache-content semantics; no versioning. | **B07 has no disaggregation and no SLO model.** DistServe owns the TTFT-percentile-under-load framing that B07's K1 borrows. |
| **Mooncake** | 2025 | **FAST 2025**, DBLP `conf/fast/QinLHCRZ0ZX25` (arXiv `2407.00079` = CoRR) | KVCache-centric disaggregated architecture, distributed KV pool, prediction-based early rejection. | KV; no mid-depth object; no edit/versioning. | Same. **Mooncake is the production-scale reference B07's "concurrent CoMem service" would be compared to; B07 has 0 measurements at that scale.** |
| **MemServe** | 2024 | `arXiv:2406.17565`; DBLP **CoRR 2024** → **arXiv-only** | Elastic memory pool + context caching for disaggregated serving. | Same as above. | Same. |
| **CacheBlend** | 2025 | **EuroSys 2025**, DBLP `conf/eurosys/YaoLLRCZD0J25`, DOI `10.1145/3689031.3696098` | Fuses **non-prefix** precomputed chunk KVs by selectively recomputing a small token subset. | KV; training-free; no store lifecycle. | Already a `--baseline` in our harness (`eval_qcmem_locomo.py:764`). **Differentiation is the object (h_j vs KV) and the trained repair — not the serving story.** |
| **EPIC** | 2025 | **ICML 2025**, OpenReview `venueid=ICML.cc/2025/Conference`; DBLP `conf/ICML/2025` | Formalises **Position-Independent Caching**; LegoLink fixes the per-document attention-sink pathology. | KV; no versioning; no tiering. | B07 does not claim position independence. |
| **MiniPIC** | 2026-06 | `arXiv:2606.13126`, **arXiv-only** | PIC in **<100 LOC** of vLLM: unrotated K in cache, RoPE applied per-request inside attention, three user-facing primitives (block-aligned padding, span separator, prompt depend). Realises Block-Attention / EPIC / Prompt Cache in one running instance. | Not an activation store; no version graph. | ⚠️ **Directly relevant as a cautionary datum: MiniPIC shows that "expose cache-reuse primitives to the user" is a <100-LOC change to vLLM.** B07's `blocking_dependency` treats its own primitive set as a large engineering project. **A reviewer will ask why.** |
| **Irminsul / HYPIC / LinearKV / SemPIC** | 2026 | `arXiv:2605.05696` (DBLP **CoRR 2026**), `2607.01299`, `2608.11231`, `2607.28069` — all **arXiv-only** | PIC for MLA-native agentic serving; hybrid-attention; hybrid LLMs; learned semantic PIC. | — | Listed for completeness: **the PIC sub-area is saturated with 2026 concurrent work.** B07 makes no PIC claim and must keep it that way. |

### 2.3 Feature: versioning / invalidation / mutable cache

**This is the feature where B07's PROPOSAL.md is most exposed, and where the two hits below are
the most serious in this entire file.**

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **Leyline: KV Cache Directives for Agentic Inference** | 2026-05-31 | `arXiv:2606.01065`; DBLP **CoRR 2026** → **arXiv-only** | ⚠️⚠️ **Names B07's exact gap in its abstract:** *"a policy may need to direct the serving system to actively remove or replace a span of cached content and continue without re-prefilling everything that came after. No existing primitive offers this."* Introduces a **declarative directive 4-tuple** separating *what to edit* from *how to preserve position correctness*; in-place splice **or** prefix-trimmed re-prefill; architecture-agnostic interface → per-architecture kernel with a closed-form RoPE-rotation correction. Measured: splice lifts replay cache-hit **+11.2 pp**, cuts latency up to **241 ms**; a ten-line truncation rule lifts agentic solve rate **+14.3 pp** on debug-gym. | Operates on **token-level KV**, not on a mid-depth activation object. No content/compatibility-hash compatibility model, no fail-closed stale-object policy, no cross-query-reuse quality read-out, no document-edit workload (its workload is **agentic trajectory editing**). | ⚠️ **This is the strongest collision in B07 and it is NOT preemption — it is CONCURRENT (2026-05-31) and its object is different.** But it forecloses a great deal: **B07 may no longer claim "no primitive exists for mutating a served cache", may not claim novelty for the edit-vs-reprefill decision, and may not claim the position-correction insight.** Our own measured difference is *the K4 finding*: `p0_17_e2_overlap` shows the stored `h12` of a chunk is **not a function of that chunk alone** (chunk-local 92.5 vs document-contextual 100.0, w32 recovers +6.0 pp, CI [3.0,9.5], McNemar b=12/c=0, p=4.883e-4). **Leyline's splice is exact by RoPE algebra; a depth-j residual has no such closed form, because the residual already absorbed cross-chunk attention.** That asymmetry is real, is ours, and is the only defensible edit-leg contribution. |
| **Models Take Notes at Prefill: KV Cache Can Be Editable and Composable** | 2026-06-14 | `arXiv:2606.17107`; DBLP **CoRR 2026** → **arXiv-only** | ⚠️⚠️ Establishes **causally, across four model families**, that at prefill the model has already written the field-conditioned conclusion into downstream tokens and the edited field's own K/V drives **<1%** of the decision — so naive overwrite leaves the model acting on the old value. Two capabilities: **append-only erratum editing** (recovers the decision at 1.00 on 8B with CoT, ~1% compute) and **position-portable composition** (RoPE-repositioned splice, logit cosine 0.90–0.999 across twelve models, O(L) TTFT). Unified agent stays decision-identical to recompute at up to **14.9×** lower latency; in an online vLLM benchmark keeps 98.5% prefix-hit and cuts p90 TTFT **53–398×**. | KV cache, not depth-j residual; no store versioning/compatibility hashes; no tiering; workload is single-document field edits and skill splicing, not a versioned multi-chunk document store. | ⚠️ **The single most dangerous paper for B07's "incremental edit" leg, and CONCURRENT (2026-06).** It independently discovers the *mechanism* behind our K4 result — that the stored state has already absorbed the conclusion, so editing the span alone is insufficient — and then **solves it** with an append-only erratum. **B07 must not claim to be first to observe that "edit the changed chunk only" is wrong-by-construction; our contribution shrinks to (a) demonstrating it at *depth-j residual* granularity with a paired n=200 measurement, and (b) the invalidation fan-out arithmetic (w/512 extra chunks per single-token edit = 0.0625 at w32, 0.25 at w128; Write cost +5.73%/+22.92% FLOPs).** |
| **Semantic Commit** | 2025 | **UIST 2025**, DBLP `conf/uist/VaithilingamKAL25`, DOI `10.1145/3746059.3747778` | Impact analysis + "semantic conflict resolution" when committing changed intent into an existing NL memory store; KG-RAG conflict detection; 12-participant within-subjects study. | HCI/interface work; no serving latency, no cache. | Different discipline. **Cite so that "versioned, conflict-aware memory updates" is not presented as an unexamined idea.** |
| **KVEraser** | 2026-06 | `arXiv:2606.17034`, **arXiv-only** (comment: Oral at an ICML 2026 *workshop* — **workshop ≠ conference**, recorded as such) | Learns to steer the KV cache for **localized context erasing**. | Erasure, not replacement; KV. | Adjacent: the "semantic forgetting" half of B07's edit leg already has a learned competitor. |

### 2.4 Feature: memory tiering (HBM / DRAM / CXL / NVMe / network)

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **CachedAttention** | 2024 | **USENIX ATC 2024**, DBLP `conf/usenix/GaoHSKJDYYZ24` | Hierarchical HBM/DRAM/SSD KV store with scheduler-aware prefetch and eviction. | KV, append-only. | See 2.1. |
| **InfiniGen** | 2024 | **OSDI 2024**, DBLP `conf/osdi/LeeLSS24` | Dynamic KV management with CPU-side speculation of the important KV entries. | KV; no store lifecycle. | — |
| **CacheGen** | 2024 | **ACM SIGCOMM 2024**, DBLP `conf/sigcomm/LiuLCRHZDY0AMHH24`, DOI `10.1145/3651890.3672274` | Encodes KV into compact bitstreams and **streams** them over a network with adaptive quality. | KV; no editing. | ⚠️ Owns the "**network** tier for cache objects" claim outright. |
| **Predictive Multi-Tier Memory Management for KV Cache** | 2026-04 | `arXiv:2604.26968`; DBLP **CoRR 2026** → **arXiv-only** | A **six-tier** hierarchy (HBM / DRAM / CXL / NVMe-GDS / RDMA / parallel FS) extending effective KV capacity 40 GB → **38 TB/node** at sub-ms TTFT for hot entries, plus architecture-variant-aware sizing. | KV. | ⚠️⚠️ **This is a superset of B07's tiering feature list, published as a system, four months earlier.** B07 has **zero** tiering measurements. |
| **ITME** | 2026-06 | `arXiv:2606.12556`, **arXiv-only** | Disaggregated CXL-hybrid tiered memory expansion for inference. | KV. | Same. |
| **LMCache** | 2025-10 | `arXiv:2510.09665`; DBLP **CoRR 2025** → **arXiv-only** | Production-oriented enterprise KV cache layer (the widely deployed one). | KV. | Same. |
| **Pancake** | 2026-02 | `arXiv:2602.21477`, **arXiv-only** | Multi-tier **agentic memory** system: multi-level index caching, cross-agent coordinated index management, GPU-CPU collaborative ANN; >4.29× e2e throughput; integrates with MemGPT/LangChain. | Indexes/embeddings, not model activations. | Adjacent but relevant: **the words "multi-tier agentic memory serving system" are taken.** |

> **B07's own K3 clause already kills this feature at 0 GPU, and it should be stated as our
> result rather than defended.** From `STATUS.json.kill_gate_executable_20260814.K3`: measured peak
> backend fetch QPS at 128k is GPU 6443 / CPU-pinned 956 / NVMe 256.5 / CEPH 47, versus model
> throughput 11.62 q/s (8 GPUs, G=1) and 1.36 q/s (G=128) — **the worst backend is still 4.04×
> the model at G=1 and 34.6× at G=128**, and measured fetch at 128k|cpu|G=1 is 10.7 ms of a
> 688.5 ms budget = **1.55%**. **There is no tiering headroom to compete for.** Against the six-tier
> system above, that is the right answer: *we do not need the tier, and we can show why.*

### 2.5 Feature: reuse-aware admission (decide Write vs raw replay by expected reuse)

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **RAGCache** | 2025/2026 | **ACM Trans. Comput. Syst.**, DBLP `journals/tocs/JinZJLLLJ26`, DOI `10.1145/3768628` (arXiv `2404.12457`) | Knowledge tree over retrieved-document KV with a **reuse/frequency-aware replacement policy** and PD overlap. | KV; retrieval-document granularity; no mid-depth object. | ⚠️ **Owns "admit/evict cached knowledge by expected reuse in a RAG serving system", peer-reviewed in a journal.** |
| **Cache-Craft** | 2025 | **PACMMOD (SIGMOD) 2025**, DBLP `journals/pacmmod/AgarwalSMMGSKYS25`, DOI `10.1145/3725273` | Identifies which chunk-caches are reusable, recomputes a small fraction to fix quality, and **stores/evicts chunk-caches to maximise reuse**. | KV; no version graph; no trained repair. | ⚠️⚠️ **This is B07's admission feature, already built and evaluated at SIGMOD, on chunk granularity — the same granularity B07 uses.** |
| **Preble** | 2025 | **ICLR 2025**, OpenReview `venueid=ICLR.cc/2025/Conference` (poster); DBLP `conf/iclr/SrivatsaHAL025` | Distributed prompt scheduling that co-optimises KV reuse and load across GPUs. | — | Owns the *scheduling* half. |
| **PRISM** | 2026-05 | `arXiv:2605.08581`; DBLP **CoRR 2026** → **arXiv-only** | Co-designs a query-aware scheduler with a demand-aware radix tree to **align request admission with KV retention**; −23.3%/−37.1% per-QPS P99 TTFT, +5.9/+12.2 pp exact-prefix hit. | KV; exact-prefix. | ⚠️ **"Align admission with cache retention" is the literal name of B07's feature, measured on P99 TTFT — the same metric as B07's K1.** |
| **Apt-Serve** | 2025 | **PACMMOD 2025**, DBLP `journals/pacmmod/GaoZSC25`, DOI `10.1145/3725394` | Hybrid cache = KV **plus a memory-efficient *hidden-state* cache** for reusable input hidden states, with adaptive batch composition to raise **request concurrency** and TTFT SLO attainment. | Hidden states are reused for *scheduling flexibility*, not as a persisted cross-query semantic object; no versioning; no trained read repair. | ⚠️⚠️⚠️ **THE SINGLE CLOSEST SYSTEM TO B07's NEXT GATE.** It is peer-reviewed (SIGMOD/PACMMOD 2025), it caches **hidden states**, and its dependent variable is **TTFT under concurrency** — i.e. it already occupies the exact (object × metric × axis) cell B07's K1 proposes to measure. **B07's remaining differences are narrow and must be stated narrowly: (i) our hidden state is a *single mid-depth layer* residual persisted across queries and documents, not a transient full-depth input-hidden cache; (ii) our read path recomputes only the upper 24 of 36 layers over a *retrieved pack*, so read compute is independent of stored-context length; (iii) we have a trained Read repair. None of (i)-(iii) has ever been measured under concurrency by us.** |

### 2.6 Feature: incremental recompute (partial re-execution after a change)

| system | year | venue (verified) | what it does | what it does NOT do | our measured difference |
|---|---|---|---|---|---|
| **CacheBlend** | 2025 | **EuroSys 2025** (as above) | Selectively recomputes a token subset to make non-prefix caches behave like a full prefill. | Token-level; no store versioning. | Our K4 asks the *quality* version of this question at chunk granularity on h_j. |
| **Cache-Craft** | 2025 | **PACMMOD/SIGMOD 2025** (as above) | "Recompute a small fraction to fix the cache." | — | Same. |
| **CacheClip** | 2025-10 | `arXiv:2510.10129`, **arXiv-only** | Auxiliary-LLM-guided token selection for selective recomputation + shared prefixes + sliding-window grouping. | Training-free; KV. | — |
| **AgentKVShift** | 2026-05 | `arXiv:2607.21604`, **arXiv-only** | Probe-guided **KV residual correction** per retrieved memory unit; shows the reuse residual = shared memory-level offset + small token-wise fluctuation, and **corrects even the tokens it does not recompute**. Explicitly reports that RAG-tuned reuse methods **degrade on structured agentic memories**. | KV; training-free; no version graph. | ⚠️ Concurrent. Its decomposition is the natural competitor to a trained repair: **a reviewer will ask why a LoRA is needed if a probe-estimated offset suffices.** B07 has no answer on file. |
| **QCFuse** | 2026-06 | `arXiv:2606.05875`, **arXiv-only** | Query-aware compressed-view selector for RAG cache fusion in SGLang; 1.7× prefill speedup at matched quality. | KV. | — |
| **EFIM** | 2025-05 | `arXiv:2505.21889`, **arXiv-only** (comment claims Euro-Par 2025 — **not independently verified in DBLP from this node**) | KV reuse for infilling, where an edit lands mid-sequence. | — | Recorded for the honest reason that its *workload* (mid-sequence edit) is B07's edit leg. |

---

## 3. MUST-NOT-CLAIM list (binding on any B07 writeup)

**This section is long because the area is crowded. Length here is compliance with the audit, not
capitulation.** Each item names the work that forecloses it.

**On the object and the caching story**
1. ❌ *"We introduce persisting intermediate activations instead of KV for serving."* —
   **HCache**, EuroSys 2025.
2. ❌ *"We introduce caching hidden states to raise serving concurrency."* — **Apt-Serve**,
   PACMMOD 2025 (hybrid KV + hidden cache, explicitly to improve request concurrency and TTFT SLO).
3. ❌ *"Reuse across queries of a precomputed context representation is new."* —
   **ReadOnce** (ACL 2021, `2021.acl-long.554`), **Prompt Cache** (MLSys 2024), **TurboRAG**
   (EMNLP 2025, `2025.emnlp-main.334`).
4. ❌ *"Non-prefix / position-independent reuse is our contribution."* — **CacheBlend**
   (EuroSys 2025), **EPIC** (ICML 2025), **MiniPIC** / **Irminsul** / **HYPIC** / **SemPIC** (2026).

**On mutability, versioning, and editing**
5. ❌ *"No primitive exists for mutating a served cache."* — **Leyline** (`2606.01065`, 2026-05)
   says exactly this and then builds one. **Repeating the sentence after Leyline is a factual error,
   not just a novelty problem.**
6. ❌ *"We are first to show that editing only the changed chunk is wrong."* —
   **Models Take Notes at Prefill** (`2606.17107`, 2026-06) establishes the mechanism causally
   across four model families (<1% of the decision comes from the edited field's own K/V).
7. ❌ *"Append-only erratum / mixed-version read with fallback is our design."* — same paper
   (append-only erratum, composes with production prefix caching at 98.5% hit-rate).
8. ❌ *"Edit-vs-reprefill as a policy decision is our design."* — **Leyline**'s directive 4-tuple
   is precisely `{in-place splice, prefix-trimmed re-prefill}` chosen by policy.
9. ❌ *"Conflict-aware versioned memory updating is unexplored."* — **Semantic Commit**
   (UIST 2025), **LatticeMind** (`2608.08236`), **MemConflict** (`2605.20926`).
10. ❌ *"Learned localized removal from a cache is unexplored."* — **KVEraser** (`2606.17034`).

**On tiering**
11. ❌ *"Multi-tier HBM/CPU/NVMe/network placement of cache objects is our contribution."* —
    **CachedAttention** (ATC 2024), **CacheGen** (SIGCOMM 2024, network tier),
    **Predictive Multi-Tier** (`2604.26968`, six tiers, 38 TB/node), **ITME** (`2606.12556`),
    **Pancake** (`2602.21477`, multi-tier agentic memory).
12. ❌ **And B07 must not claim it needs tiering at all**: its own K3 arithmetic shows the worst
    backend is ≥4.04× the model's throughput and fetch is 1.55% of the TTFT budget.
    **The honest B07 statement is a negative result: tiering has no headroom here.**

**On admission / scheduling**
13. ❌ *"Reuse-aware admission (Write vs raw replay) is our contribution."* —
    **RAGCache** (ACM TOCS), **Cache-Craft** (SIGMOD 2025), **PRISM** (`2605.08581`,
    admission aligned to retention, measured on P99 TTFT), **Preble** (ICLR 2025).

**On what our own artefacts do and do not support**
14. ❌ *Any concurrency claim whatsoever, today.* `STATUS.json.established_measurements._status`:
    **"NONE FOR B07'S OWN CLAIMS."** `bench_p1_8_serving_curve.py` has **no** concurrency axis and
    **no** TTFT percentiles; `bench_persistent_store_io.py` sweeps threads but with **no model
    loaded**. Citing either as serving evidence is the mixed-construct error the repo has already
    made once.
15. ❌ *"CoMem is a storage method."* **DEAD** — A02 clause (c) failed at 2048× vs a
    pre-registered 100× (8192 B/tok vs 4 B/tok). Do not re-litigate.
16. ❌ *A serving win at unmatched quality.* `paperA/TODOList.md:763` (P0.20): BM25 −11.56 pp
    CI [−14.44, −8.67]; dense TIE −1.0 pp p=0.637 — 「最好情形是打平，从不是赢」.
17. ❌ *`c1_all` (pack-everything) as the comparator.* Forbidden: A02 §5 hands 86–93% of the
    apparent win to plain retrieval and it OOMs at 1M.
18. ❌ *Any retention **ratio** Δ(C=8)/Δ(C=1) when the denominator's CI includes 0.* The harness
    already prints `Q*=inf` at `32k|gpu|G=512` and the `128k|cpu|G=512` denominator is
    **0.013 s**. Absolute ms only.
    (⚠️ pointer correction: `B07_SERVING_GATE_PREREG.md:123` and `STATUS.json…delta_guard` both cite
    this as `aggregate.out:8`; verified this session it is **line 7** — line 8 is the
    `wrote …aggregate.json` message. Same fact, wrong line number, corrected here because
    `STATUS.json` is append-only.)
    (`memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md`)
19. ❌ *"The mechanisms in 关键设计 exist."* They do not
    (`STATUS.json.blocking_dependency`). Do not write a systems paper in the present tense about
    unimplemented components.

---

## 4. Safe residual claim — one falsifiable sentence

Everything in §3 removes; this is what is left, and it is **narrow on purpose**:

> **For a mid-depth (j=12) residual store whose read recomputes only the upper 24 of 36 layers
> over a fixed retrieved pack, the single-stream 246.4 ms per-query advantage over a matched
> j=0 raw-text replay of the same pack either survives 8-way concurrency with a paired TTFT-p99
> Δ ≥ 123.2 ms (95% CI excluding 0, positive sign) or it does not — and it is the shared serial
> CPU select stage (117.7 ms per request at 128k, paid identically by both arms) that decides
> which.**

**Why this is falsifiable and can actually lose** (B07's own self-test, §1.7 of the prereg):
at C=8 with one selector worker, `iter_bm25` costs 8 × 117.7 = **941.6 ms** of serial CPU per
batch, **exceeding comem's entire 677.9 ms GPU read**. If C=8 is bottlenecked there, both arms'
p99 is dominated by an identical term, Δ collapses below 123.2 ms, **K1 fires, thesis dead**.
P2.2 independently records that CPU path as a GIL-bound single-copy loop, so the mechanism is live.
Conversely if C=8 is GPU-read-bound, queue wait scales with service time, the cheaper 24-layer arm
drains faster, and the gap **amplifies**. Both branches follow from data already on disk.

**And a second, cheaper falsifiable sentence that is fully ours** (the edit leg, after §2.3
removed the mechanism claim):

> **A depth-j residual admits no exact local edit: because h_j has already absorbed cross-chunk
> attention, rewriting the edited chunk alone silently ships the chunk-local arm (92.5 vs 100.0
> document-contextual, n=200 paired), and restoring w=32 tokens of original left context recovers
> +6.0 pp (95% CI [3.0, 9.5], McNemar b=12/c=0, p=4.883e-4) at a cost of 0.0625 extra rewritten
> chunks and +5.73% Write FLOPs per single-token edit — whereas KV-level caches admit a
> closed-form RoPE splice.**

That contrast (**closed-form splice for KV vs no closed form for h_j**) is the one thing in B07 that
neither Leyline nor *Models Take Notes at Prefill* can claim, because neither has a depth-j object.
**It is also small.** See §6.

---

## 5. Honest gaps in this adjudication

1. **Semantic Scholar 429 on every call → unused.** Every `arXiv-only` row rests on DBLP CoRR
   and/or the arXiv comment field alone.
2. **DBLP returned malformed/HTML bodies intermittently**; recovered by 4-try backoff. Two
   queries never resolved and were re-phrased instead of retried indefinitely
   (`Pensieve retaining conversation state multi-turn LLM serving`, and
   `cross-model KV cache reuse transfer` → 0 hits both times). **`Pensieve` is therefore
   UNVERIFIED, not absent** — it is a known multi-turn stateful-serving paper and its omission from
   §2.1 is a gap, not a finding.
3. **arXiv bursts hit 429 / read-timeout.** Three queries were abandoned mid-family
   (`all:"CacheBlend" AND all:"adapter"`, `all:"cross-method" AND all:"cache reuse" AND
   all:"generalization"`, `all:"fine-tuning" AND all:"cache reuse" AND all:"transfers to"`).
4. **`EFIM`'s Euro-Par 2025 acceptance is from the arXiv comment field only**, not DBLP-confirmed.
   **`KVEraser`'s claim is an ICML 2026 *workshop* oral** — recorded as workshop, which is not a
   conference venue.
5. **All 2026 systems hits (Leyline, Models-Take-Notes, PRISM, Predictive Multi-Tier, MiniPIC,
   AgentKVShift, QCFuse, Pancake, Irminsul) are DBLP `CoRR 2026` = preprints.** Several may be under
   review at OSDI/SOSP/NSDI/EuroSys right now; **OpenReview `venueid` does not cover systems
   venues**, so there is no route from this node to check that. This is the largest structural gap
   in this file.
6. **The three most dangerous papers (Leyline, Models Take Notes at Prefill, Apt-Serve) were
   adjudicated from abstracts only.** Their method sections were **not** read. Every "what it does
   NOT do" cell for those three is therefore provisional and must be re-checked before submission.
7. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted here and ssh was barred; the
   five harnesses in `needs_disk_20260814` remain wzc1-verified and **zwfy6-unverified, not absent**
   (`memory/two-disk-rule-applies-to-main-too.md`).
8. **No arXiv-vs-camera-ready diff** for any peer-reviewed row.
9. **This file discharges the prereg's stated venue-verification blocker**
   (`B07_SERVING_GATE_PREREG.md:264-268` recorded OpenReview 403 + Anthology timeout on 08-14).
   Both were reachable on 08-15. That prereg line is now stale; it is append-only, so it is
   corrected **here** rather than edited there.

---

## 6. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED only for the NARROWED two sentences in
         section 4; FOUR of the five PROPOSAL.md features are foreclosed as claims by
         peer-reviewed or concurrent systems and must be demoted to engineering
related_work_status: audited
already_dead_should_archive: NO
```

**Per-feature disposition, which is the actual deliverable:**

| PROPOSAL.md feature | disposition after this audit |
|---|---|
| **concurrency** | **KEEP as the gate.** Closest occupant is **Apt-Serve** (PACMMOD 2025), which already caches hidden states and measures TTFT under concurrency. Our cell survives only as: *mid-depth cross-query residual + upper-24-layer read over a retrieved pack*. Must cite Apt-Serve as the nearest system, not as a distant relative. |
| **versioning / edit** | **DEMOTE to a measurement, not a design.** Leyline + Models-Take-Notes own the primitive and the mechanism. Only the *"no closed-form splice for h_j"* asymmetry and the fan-out arithmetic remain ours. |
| **tiering** | **DROP.** Our own K3 arithmetic says no headroom; the six-tier system (`2604.26968`) says the space is taken. Report as a negative result. |
| **reuse-aware admission** | **DROP as a claim.** RAGCache (TOCS) + Cache-Craft (SIGMOD) + PRISM own it. Keep only the break-even reuse N* numbers as a *cost model*, which A02 already measured (8 → 25 → 186 across 32k → 128k → 1M). |
| **"production end-to-end system"** | **NOT a contribution.** vLLM/SGLang/Mooncake/LMCache are production systems; B07 has no measurement and no implementation. |

- **Nothing is 完全相同 / 抄袭.** The two closest mutable-cache papers are **2026-05 and 2026-06 =
  concurrent** and both operate on token-level KV, not a depth-j residual. The closest
  peer-reviewed hit (**Apt-Serve**) caches *transient input hidden states for scheduling*, not a
  persisted cross-query semantic object. **`already_dead_should_archive` is NOT warranted.**
- **But B07 is now clearly the weakest of the three directions audited today**, and the reason is
  not literature count — it is that **it has zero measurements of its own** while five prior
  systems have full ones on each of its features. The honest reading: B07 is worth **one 1.84 GPU-h
  concurrency run** to settle K1, and if K1 fires it collapses to the edit-leg asymmetry, which is
  at most a short-paper section inside Paper A, not a systems paper.
- **Promotion is not warranted.** Blockers unchanged and both 0-GPU: (a) the concurrency axis does
  not exist in code; (b) `KILL_KEYS` / cost-key precedence in `ready_queue.py` still surfaces stale
  sentinels (the prereg names the two-line fix and it is deliberately not made here).

### 6.1 `STATUS.json` deliberately NOT modified

The `related_work_status: audited` line in §6 is **this file's self-assessment for a human reader,
not a scheduler token.** Measured this session by importing `proposal/ready_queue.py` and calling
`read_one()` on a temp copy in `/tmp` (**no repo file touched**): appending
`"related_work_status": "audited"` to B07's `STATUS.json` flips `novelty_checked` to `True`
(`ready_queue.py:203-209`: `NOVELTY_VERDICT_KEYS` contains `related_work_status`, `VERDICT_CLEARED`
contains the literal `"audited"`) but B07 **stays `ready_cpu`**, held by its own live
`blocking_dependency`. So the append would be harmless here — and it was still not made, because
per `memory/a-declared-lifecycle-is-not-an-adjudicated-one.md` an agent writing its own clearance
field is not the same thing as the clearance being reviewed. **B06 is the one where that append
WOULD auto-promote to `ready_gpu`; see B06's §6.1.**
