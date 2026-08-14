# B08 — RELATED WORK / NOVELTY ADJUDICATION (three legs, adjudicated separately)

**Written 2026-08-15. 0 GPU, 0 ssh. Adjudication only — this file runs nothing and launches nothing.**

Closes the blocker `proposal/ready_queue.py:542-553` stats (`RELATED_WORK.md absent`) and
`STATUS.json.remaining_blockers_all_CPU[1]`.

**Audit rating: ★ 严重不足 (severely insufficient) — the worst rating of any proposal in the
repository**, `RELATED_WORK_GAP_AUDIT_20260808.md:98`. Families demanded: *query-focused context
compression; grounded notes/provenance; personal/conversational memory; temporal KG/event
sourcing; hierarchical memory*. The audit's instruction:

> 「三个子方向最好拆开，各写 Related Work。」 (line 98)
> …「B07/B08 不能依靠"可版本化/可更新/分层 memory"这一功能清单主张新颖性。必须给出：
> 与现有系统逐 feature 差异；端到端质量/latency/bytes；stale/conflict failure；
> 明确 workload (尤其跨 query reuse)。」 (lines 133-152)

**§3, §4, §5 are the three legs, adjudicated independently. §2 explains why their literature
pressure is structurally different — that is the part the audit actually asked for and the part
that determines which leg should be cut.**

---

## 1. Standing rules and endpoint status

1. **`memory/prior-work-differentiate-dont-abandon.md`** (user 2026-08-07): bar is
   **完全相同 / 抄袭**, not overlap; 2-3 months = **concurrent**, cannot preempt; a direction dies
   **only from its own kill gate**. B08's own `novelty_status_detail` already records this:
   *"EXPLICITLY NOT A FALSIFICATION … This is a WRITING constraint."*
   **Agent memory is the single most crowded area in this repo's literature. Collision count is
   not the verdict.**
2. **Venue verification by family**: OpenReview `venueid` for ICLR/NeurIPS/ICML; ACL Anthology ID
   + DBLP for ACL/EMNLP/NAACL/EACL **including Findings**; DBLP for systems/AI conferences
   (AAAI/ECAI/UIST); `arXiv-only` = **"I could not verify a peer-reviewed venue from this node"**.
3. **Endpoints, 2026-08-15**: DBLP **200** (intermittent HTML bodies → 4-try backoff);
   `api2.openreview.net` **200** (no `ChallengeRequiredError` today);
   `aclanthology.org` **200**; **Semantic Scholar 429 on every call → unused**;
   arXiv API 200 but 429/read-timeout under bursts.
4. ⚠️ **DBLP mislabels Findings papers.** Measured: DBLP says `venue=ACL` for LLMLingua-2 whose
   DOI is `10.18653/v1/2024.findings-acl.57`. Every ACL-family row below carries the **Anthology
   ID**, which is the authority.

---

## 2. Why the three legs have DIFFERENT literature pressure

This is the structural answer, and it drives §6.

| | leg 1 — notes + raw | leg 2 — typed ledger | leg 3 — pyramid |
|---|---|---|---|
| **Is the architecture prior art?** | **YES, and our own tree says so.** `longmemeval/compressor.py:8` credits "raw evidence + notes" as **the LongMemEval-V2 winning pattern**. | **YES, every primitive.** validity intervals, supersession, tombstones, provenance/confidence → all built (§4). | **YES.** near/far/profile tiering → MemGPT, HippoRAG, G-Memory, HMO, Pancake (§5). |
| **Is the *measurement* prior art?** | **NO / thin.** No verified work holds retrieval at a **measured** `any_hit = 1.000` and varies only context composition while scoring an **unsupported-claim rate** on a notes-only arm. | **YES.** MemConflict, MemSyco-Bench, and *Ground Truth First* already measure stale/conflict/validity-interval failure directly (§4). | **N/A — there is nothing to measure.** No cell, no arm, no metric on disk. |
| **Do WE have adjacent evidence?** | **Supporting**: the retrieval-closed stratum is measured (`any_hit` 1.000 on KU n=78 + SSA n=56, re-verified at the gate's own `budget=4000`). | **Neutral**: the cell is the same n=78; `src/eval/update_eval.py` is git-tracked and reusable. | **AGAINST, and it is ours**: A02 measured the fixed-Read advantage at **1.03–1.37×** with break-even reuse `N*` **8 → 25 → 186** across 32k → 128k → 1M — the economics get *worse* exactly where a pyramid is pitched. |
| **Code status** | harness runs; 2 arms missing (§3.4) | `src/eval/update_eval.py` git-tracked ✅ | ⚠️ **REVENANT**: `src/memory/l2/`, `src/memory/l3/`, `src/agents/memory_agent.py` are **byte-identical to `legacy/src_dead_subsystems/`** (commit `b63b5a1`) and are **untracked in git**. |
| **Pressure verdict** | **methodological gap survives; the mechanism claim does not** | **foreclosed on BOTH axes** | **occupied AND negatively pre-measured by us** |

**The distinction the task asked me to keep straight — "occupied" vs "we did too little":**
leg 1 is *"we did too little"* (the gap is real, the arms are unbuilt). Leg 2 is *"occupied"* (the
gap is closed by others, our arms would be redundant). **Leg 3 is both at once**, which is why it
is the one to cut (§6).

---

## 3. LEG 1 — Query-conditioned notes + raw evidence

### 3.1 What leg 1 claims now

Per `STATUS.json.next_gate`: on the retrieval-closed LongMemEval-S stratum (`knowledge-update`
n=78 + `single-session-assistant` n=56, stratum n=134, measured `any_hit_recall == 1.000` at BM25
top_k=10 **and re-verified at the gate's own `evidence_token_budget=4000`**), freeze retrieval and
vary **only** context composition across three arms — `A-raw`, `A-notes+raw`, `A-notes-only` —
reading out **ACC** and **U** (unsupported-claim rate). PASS on Δ_aug CI > 0 **or** Δ_U CI > +5.0 pp.

**The retrieval premise is a constraint, not a win**: with `any_hit` 96.8% at top_k=10 overall,
only ~3.2 pp of retrieval headroom is left, so `established_measurements.consequence_is_a_CONSTRAINT_not_a_win`
already forbids framing leg 1 as a retrieval contribution.

### 3.2 Family: query-focused context compression

| work | year | venue (verified) | what it does | precise difference |
|---|---|---|---|---|
| **RECOMP** | 2024 | **ICLR 2024**, OpenReview `venueid=ICLR.cc/2024/Conference` (poster) | Compresses retrieved documents into extractive **or abstractive summaries** before the reader, with selective augmentation (skip when retrieval is unhelpful). | ⚠️ **The canonical "summarise the retrieved evidence for the reader" paper, peer-reviewed.** Difference: RECOMP optimises **end-task accuracy at a token budget** and does not measure whether the abstractive summary **invents** facts; it also never pins retrieval at a *measured* recall of 1.000, so its deltas remain retrieval-confounded. **"We introduce query-conditioned summarisation of retrieved evidence" is unclaimable.** |
| **LLMLingua** | 2023 | **EMNLP 2023 (main)**, Anthology `2023.emnlp-main.825`, DOI `10.18653/v1/2023.emnlp-main.825` | Coarse-to-fine budget-controlled prompt compression with a small LM's perplexity signal. | Token-deletion compression, task-agnostic. No notes, no faithfulness read-out. |
| **LongLLMLingua** | 2024 | **ACL 2024 (main, long)**, Anthology `2024.acl-long.91` | Question-aware coarse-to-fine compression for long context. | ⚠️ **Owns "query-aware compression".** Still deletion-based, still no unsupported-claim metric. |
| **LLMLingua-2** | 2024 | **Findings of ACL 2024**, Anthology `2024.findings-acl.57` (⚠️ DBLP prints `venue=ACL` — wrong) | Data-distilled task-agnostic compression, and the title's own word is **"Faithful"**. | ⚠️ **The word "faithful compression" is taken.** Difference: its faithfulness is *token-level fidelity to the source* on compression benchmarks; B08's `U` is a **downstream unsupported-claim rate on the reader's answer** with the notes as sole context. Adjacent, not identical — but the phrasing must be careful. |
| **Provence** | 2025 | **ICLR 2025**, OpenReview `venueid=ICLR.cc/2025/Conference` (poster) | Efficient, robust **context pruning** for RAG (a reranker+pruner in one). | Pruning, not note-generation; no abstraction, hence no fabrication risk to measure. |
| **xRAG** | 2024 | **NeurIPS 2024**, OpenReview `venueid=NeurIPS.cc/2024/Conference` (poster) | Extreme compression of a retrieved document to **one token** via modality fusion. | The extreme end of "notes replace raw". No faithfulness metric; and its one-token form makes B08's `A-notes-only` arm look conservative. |
| **FILCO — Learning to Filter Context for RAG** | 2023 | `arXiv:2311.08377`; DBLP **CoRR 2023** → **arXiv-only** | Learns to filter retrieved context by a lexical/NLI/output-conditioned signal. | Filtering, not summarising. |
| **EDU-based Context Compressor** | 2025-12 | `arXiv:2512.14244`, **arXiv-only** | Structure-then-select compression over Elementary Discourse Units **anchored strictly to source indices to eliminate hallucination**. Releases StructBench. | ⚠️ **Explicitly engineered against the exact failure mode B08 proposes to measure**, and it *solves* it structurally (index anchoring) rather than measuring it. Difference: it is a compression **method** with a design-level guarantee; B08 asks the empirical question "does a free-form generative note fabricate, and does raw evidence alongside it protect the reader?" **B08 must not imply the fabrication risk is unaddressed in the literature.** |

### 3.3 Family: grounded notes / provenance

| work | year | venue (verified) | what it does | precise difference |
|---|---|---|---|---|
| **Chain-of-Note** | 2024 | **EMNLP 2024 (main)**, Anthology `2024.emnlp-main.813`, DOI `10.18653/v1/2024.emnlp-main.813` | Generates **sequential reading notes per retrieved document**, then answers — improving robustness to noisy/irrelevant retrieval and enabling "unknown" responses. | ⚠️⚠️ **THE closest architectural collision for leg 1 and it is peer-reviewed at EMNLP main.** It literally generates notes over retrieved evidence and reads both. Differences that remain: (a) CoN's notes are **per-document and sequential**, B08's are a single **query-conditioned synopsis prepended to a budget-limited raw set** (`run_baseline.py:152-162`); (b) CoN's contribution is **robustness to noisy retrieval** — i.e. it *needs* retrieval to be imperfect, whereas B08 pins `any_hit=1.000` to remove that very axis; (c) CoN has **no notes-only arm and no unsupported-claim metric**. **"We introduce notes over retrieved evidence" is dead. What survives is the notes-only faithfulness contrast on a retrieval-closed cell.** |
| **Self-Notes** | 2023 | **NeurIPS 2023**, DBLP `conf/nips/LanchantinTWSS23` | Model interleaves its own notes into the context to reason and memorise. | Origin of "self-generated notes as memory". Not retrieval-grounded, no faithfulness metric. Our gate's `A-notes+raw` generator **is** self-notes (prereg §4.1) — so this is a **required citation for our method, not a competitor**. |
| **ALCE — Enabling LLMs to Generate Text with Citations** | 2023 | **EMNLP 2023 (main)**, Anthology `2023.emnlp-main.398` | Benchmark + metrics for **citation precision/recall** in generated text: is each claim supported by the cited passage? | ⚠️ **This is the closest thing to B08's `U` metric that has a venue.** Difference: ALCE scores *citation attribution* of a final answer against a retrieved corpus. `U` scores whether a claim is absent from **that arm's own context**, which is a per-arm-conditional denominator (that conditioning is what makes the notes-only arm interpretable). **B08 must build `U` on ALCE/NLI-entailment machinery and say so, not present it as a new metric family.** |
| **FActScore** | 2023 | **EMNLP 2023 (main)**, Anthology `2023.emnlp-main.741` | Atomic-fact decomposition + per-fact support checking against a knowledge source. | Same disposition: the **method** for computing `U` already exists. **"No faithfulness scorer exists anywhere in the tree"** (`STATUS.json` blocker 3) is true of **our repo**, not of the literature. Those are different statements and the write-up must not blur them. |
| **SummaC** | 2022 | **TACL 2022**, DBLP `journals/tacl/LabanSBH22`, DOI `10.1162/tacl_a_00453` | NLI-based sentence-level inconsistency detection for summarisation. | The classical instrument for "did the summary invent something". Ditto. |
| **Retain or Consolidate? Budget-Dependent Operator Selection for Language Agent Memory** | 2026-07-20 | `arXiv:2607.17545`; DBLP **CoRR 2026** → **arXiv-only** | ⚠️⚠️⚠️ Formalises exactly leg 1's trade-off: **retention (raw records, exact detail, may not fit) vs consolidation (compress/combine, better coverage per token, risks losing query-critical detail)**; decomposes each operator's utility into a *coverage effect on omitted evidence* + a *signed replacement effect on raw evidence that already fits*; learns the choice (OAS) with **held-out harm calibration**. Evaluated **on public LongMemEval and LoCoMo**, reporting up to **+48% absolute accuracy** from consolidation under tight budgets and retention preferable under loose ones. | ⚠️ **The single most dangerous paper for leg 1, and it is CONCURRENT (2026-07 vs our prereg 2026-08-14) so it cannot preempt.** But it does foreclose a lot: the raw-vs-notes trade-off framing, the budget-dependence, and the LongMemEval workload are all taken, **with a learner on top**. What it does **not** do: (a) it never pins retrieval at a **measured** `any_hit = 1.000`, so its deltas mix retrieval with composition — the exact confound B08's stratum was chosen to remove; (b) its harm signal is a *calibrated utility estimate*, **not a measured unsupported-claim rate on a notes-only arm**; (c) it does not run notes-only-vs-raw as a paired faithfulness contrast. **Leg 1's residual claim must be stated against this paper explicitly, and it shrinks to (a)+(b).** |
| **LongMemEval-V2** | 2026-05 | `arXiv:2605.12493`, **arXiv-only** | Web-agent memory benchmark (451 questions, ≤500 trajectories, 115M tokens); "context gathering" formulation; **AgentRunbook-R** keeps knowledge pools of raw observations, events **and strategy notes**. | This is the source our own `compressor.py:8` credits for the notes+raw pattern. **The architecture is credited prior art inside our own repository. Nothing in leg 1 may claim it.** |

### 3.4 Leg 1 — what still blocks it, and it is not literature

From `STATUS.json.remaining_blockers_all_CPU`, all **0 GPU**: the `A-notes-only` arm **cannot be
expressed** (`run_baseline.py:162` hardcodes `reader_evidence = [notes_block] + list(evidence)`,
no withhold path); the `U` scorer **does not exist** in the tree; no `SelfNotesCompressor` exists
(`--compressor mom_notes` is unrunnable — zero of 45 `adapter_config.json` under `outputs/` carry
`num_slots`/`slot_dim`; all are PEFT-LoRA); the judge input adapter is missing.
**Two of those (the notes-only arm, the `U` scorer) ARE the novelty.** A leg whose contribution is
unimplemented is not blocked by prior art; it is blocked by us.

---

## 4. LEG 2 — Typed personal memory ledger

### 4.1 What leg 2 claims

`PROPOSAL.md`: immutable event memory; derived profile; **validity interval**; **supersedes /
tombstone**; source **provenance / confidence**; current vs historical vs **abstain**. Tasks:
overwrite, stale, contradiction, temporary, LongMemEval update/temporal.

**This is precisely the "功能清单" (feature list) the audit at line 146 says cannot support a
novelty claim.** Below, every item on the list is matched to a system that already has it.

### 4.2 Family: personal / conversational memory

| work | year | venue (verified) | which of leg 2's primitives it already has |
|---|---|---|---|
| **LongMemEval** | 2025 | **ICLR 2025**, OpenReview `venueid=ICLR.cc/2025/Conference` (poster), id `pZiyCaVuti`; DBLP `conf/iclr/WuWYZCY25` | Defines the **knowledge-update** and temporal-reasoning question types leg 2 would be scored on. It is our **benchmark**, so leg 2 cannot claim the task design. |
| **MemoryBank** | 2024 | **AAAI 2024**, DBLP `conf/aaai/ZhongGGYW24`, DOI `10.1609/AAAI.V38I17.29946` | Long-term memory with Ebbinghaus-style forgetting + evolving **user portrait** = leg 2's "derived profile". |
| **MemGPT** | 2023 | `arXiv:2310.08560`; DBLP **CoRR 2023** → **arXiv-only** (widely deployed as Letta) | Tiered main/external context with explicit self-directed **read/write/edit** of memory. |
| **Mem0** | 2025 | **ECAI 2025**, DBLP `conf/ecai/ChhikaraKASY25`, DOI `10.3233/FAIA251160` | Production memory layer with extract → **conflict-resolve → update/delete** operations, i.e. supersession and tombstoning, shipped. |
| **A-Mem** | 2025 | **NeurIPS 2025**, OpenReview `venueid=NeurIPS.cc/2025/Conference` (poster); DBLP `conf/nips/XuLMGTZ25` | Agentic memory notes with attributes + link generation + **memory evolution** on new evidence. |
| **MemSyco-Bench** | 2026-07 | `arXiv:2607.01071`, **arXiv-only** | ⚠️ Five tasks that measure whether an agent can **reject memory as evidence, respect its applicable scope, resolve memory-vs-evidence conflict, track memory updates**, and personalise from valid memory. **That is leg 2's task list, as a benchmark.** |
| **MemConflict** | 2026-05 | `arXiv:2605.20926`; DBLP **CoRR 2026** → **arXiv-only** | ⚠️ Formalises **dynamic / static / conditional** conflicts over **temporal validity, factual correctness, contextual applicability**; injects cross-session conflicts + similar distractors; white-box analysis of supporting-memory ranking; six memory systems evaluated. **This is leg 2's stale/conflict read-out, already built and already run on six systems.** |

### 4.3 Family: temporal KG / event sourcing / validity intervals

| work | year | venue (verified) | which primitive it has |
|---|---|---|---|
| **Zep / Graphiti** | 2025 | `arXiv:2501.13956`; DBLP **CoRR 2025** → **arXiv-only** (a DBLP query with "Graphiti" returned NO HITS; the base title resolves to CoRR only) | ⚠️⚠️ A **temporally-aware knowledge-graph engine** that synthesises conversational + business data **while maintaining historical relationships**, evaluated on DMR **and LongMemEval**. This is the bitemporal / validity-interval / supersession design of leg 2, on leg 2's benchmark. |
| **Ground Truth First** | 2026-07-24 | `arXiv:2607.21962`, **arXiv-only** | ⚠️⚠️⚠️ **The closest single work to leg 2 in existence.** A seeded life-script sampler emits facts with **per-fact validity intervals, volatility classes, and source channels** *before any text exists*; a fidelity verifier confirms every planted fact; questions instantiated mechanically so gold answers are script-valid by construction; **sent/received trust distinctions**, injection probes, **as-of-date question sets**; five memory architectures + a no-memory control, versioned judge, three replicates, two horizons — and it reports a **ranking inversion with history length** (a curated-map memory leading at three weeks loses evicted content by nine weeks). **Leg 2's entire type system is this paper's data-generation schema, and its evaluation is stronger than anything leg 2 has designed.** Concurrent (2026-07), so no preemption — but there is nothing left for leg 2 to introduce. |
| **Nous — belief-based agent memory** | 2026-06 | `arXiv:2606.22030`, **arXiv-only** | ⚠️ Per-entity-attribute categorical beliefs updated by closed-form Bayes; **reliability-conditioned** updating; **provenance-capped** trust bounded by source rather than textual confidence. **This is leg 2's "source provenance / confidence" field, with a mechanism and a poisoning threat model.** It also reports the finding that matters most to leg 2: **belief updating gives little benefit over naive last-write-wins on existing conversational benchmarks because they rarely contain contradictory or differently-reliable evidence.** That is an *adverse* result for leg 2's premise, from a third party. |
| **LatticeMind** | 2026-08-08 | `arXiv:2608.08236`, **arXiv-only** | Conflict-aware structured memory handling contradiction **at write time** with explicit **item status** + symbolic checks + LLM reconciliation; 0.97 vs 0.61 on label-blind ConflictBank, p<1e-6 paired McNemar. Leg 2's "current vs historical vs abstain" status field, built. |
| **Semantic Commit** | 2025 | **UIST 2025**, DBLP `conf/uist/VaithilingamKAL25`, DOI `10.1145/3746059.3747778` | Impact analysis + semantic conflict resolution when committing changed intent into an existing memory store. The HCI framing of the same problem. |
| **Temporal KG QA: A Survey** | 2024 | `arXiv:2406.14191`; DBLP **CoRR 2024** → **arXiv-only** | Shows temporal-validity QA is a **mature sub-field with its own survey**, predating all of the above. |

### 4.4 Leg 2 disposition

**Every primitive in leg 2's list is implemented in a named system, and the read-out is
implemented in at least three benchmarks (MemConflict, MemSyco-Bench, Ground Truth First).**
Leg 2 is also *pre-answered adversely* by Nous: on existing conversational benchmarks, principled
updating ≈ last-write-wins because the benchmarks lack reliability contrast. Leg 2's own kill
clause ("typed ledger 不降低 stale/conflict error") is therefore **likely to fire on the n=78
knowledge-update cell** for a reason that has nothing to do with our implementation.
**Not dead by literature — but it must not be gated before leg 1, and it must never be pitched as
a design contribution.** The prereg's decision to fold it into leg 1's cell is correct.

---

## 5. LEG 3 — Multi-tier pyramid memory

### 5.1 What leg 3 claims

`PROPOSAL.md`: **near** = exact residual; **far** = compressed latent/notes; **profile** =
structured long-term state. Two-layer MVP first; the proposal itself says the full pyramid is
high-risk and must not be invested in at scale.

### 5.2 Family: hierarchical memory

| work | year | venue (verified) | what it does | difference |
|---|---|---|---|---|
| **MemGPT** | 2023 | `arXiv:2310.08560`, **arXiv-only** | Two-tier main-context / external-context with paging. | The origin of tiered LLM memory. Leg 3's MVP **is** this. |
| **HippoRAG** | 2024 | **NeurIPS 2024**, OpenReview `venueid=NeurIPS.cc/2024/Conference` (poster); DBLP `conf/nips/GutierrezS0Y024` | Hippocampal-indexing long-term memory: a KG index + PPR over it, layered above passages. | Peer-reviewed occupant of "structured long-term layer above raw evidence". |
| **G-Memory** | 2025 | **NeurIPS 2025 spotlight**, OpenReview `venueid=NeurIPS.cc/2025/Conference`; DBLP `conf/nips/ZhangFWWYY25` | **Three-tier** hierarchical memory (insight / query / interaction graphs) for multi-agent systems. | ⚠️ **A NeurIPS spotlight with exactly leg 3's three-tier shape.** |
| **Hierarchical Memory Orchestration (HMO)** | 2026-04 | `arXiv:2604.01670`, **arXiv-only** | **Three-tiered** directory — compact primary cache (recent+pivotal) **coupled with an evolving user profile**, a high-priority secondary layer, and a global archive; **persona-driven redistribution across tiers**. | ⚠️⚠️ **This is leg 3's near / far / profile design, with promotion policy, four months earlier.** |
| **Pancake** | 2026-02 | `arXiv:2602.21477`, **arXiv-only** | Multi-tier agentic memory **serving** system: multi-level index caching, cross-agent index coordination, GPU-CPU ANN; >4.29× e2e throughput; integrates with MemGPT/LangChain. | Owns the *systems* half of leg 3. |
| **MM-Mem (From Verbatim to Gist)** | 2026-03 | `arXiv:2603.01455`; the arXiv comment claims **"Accepted by ACL 2026 Main"** — ⚠️ an Anthology search from this node returned **no result**, so per rule 1.2 this is recorded as **arXiv-only / venue claim UNVERIFIED**, not as ACL 2026 | **Pyramidal** memory (Sensory Buffer → Episodic Stream → Symbolic Schema) with progressive verbatim→gist distillation under a **Semantic Information Bottleneck** objective + entropy-driven top-down retrieval. | ⚠️⚠️⚠️ **The word "pyramidal memory", the verbatim→gist axis, AND a trained objective for the tier transition — all taken.** Multimodal/video rather than text, which is the only daylight. |
| **MemoRAG** | 2024 | `arXiv:2409.05591`; DBLP **CoRR 2024** → **arXiv-only** | Global memory module producing draft clues that steer retrieval over raw evidence. | Two-tier memory-over-corpus. |
| **PyramidKV** | 2024 | `arXiv:2406.02069`, **arXiv-only** | Layer-wise pyramidal KV budget allocation. | ⚠️ **Named for clarity, NOT a collision** — it is a per-layer KV budget schedule, an orthogonal mechanism. Listed so nobody mistakes name-overlap for idea-overlap. |

### 5.3 Leg 3 disposition — the negative evidence is OURS

1. **Its own kill clause is already answered against it by our own measurement.** The clause is
   "far-memory read cost swallows the fixed Read advantage". A02 measured that advantage at
   **1.03–1.37×** with break-even reuse `N*` **8 → 25 → 186** across 32k → 128k → 1M. A pyramid is
   pitched at large corpora; that is where `N*` is 186.
2. **Its architecture is occupied at NeurIPS-spotlight level** (G-Memory) **and duplicated twice in
   2026** (HMO, MM-Mem) including the word "pyramidal" and the verbatim→gist axis.
3. **Its code is a revenant.** `src/memory/l2/`, `src/memory/l3/`, `src/agents/memory_agent.py` are
   byte-identical to `legacy/src_dead_subsystems/` (commit `b63b5a1`) and **untracked in git**.
   The infrastructure leg 3 cites is a restored working-tree copy of code this project **already
   abandoned**. (`STATUS.json.code_revenant_warning`.)
4. **It has no cell, no arm, no metric, and no gate.** `portfolio_narrowing_20260814` records the
   deprioritisation as a *scheduling* statement; this file adds the *literature* reason.

---

## 6. THE JUDGEMENT ASKED FOR: which leg has the smallest safe gap

> **Leg 3 (multi-tier pyramid memory) has the smallest safe gap and is the one that should be cut.
> Cut, not deferred.**

**Why leg 3 and not leg 2**, since both are heavily occupied — the deciding criterion is whether a
*measurement of ours could still differ from the literature's*:

| | leg 2 | leg 3 |
|---|---|---|
| occupied by prior art? | yes (§4.2–4.3) | yes (§5.2) |
| does a cell exist where our number could differ? | **yes** — the n=78 `knowledge-update` cell, with a git-tracked metric set (`src/eval/update_eval.py`) and a measured retrieval-closed premise | **no** — no cell, no arm, no metric |
| do we have our own evidence about it? | neutral | **adverse, and quantitative** (1.03–1.37×; `N*` 8→25→186) |
| is the supporting code real? | yes, git-tracked | **no — byte-identical to `legacy/src_dead_subsystems/`, untracked** |
| marginal cost to learn something | ~0 (it rides leg 1's run) | a new subsystem, from dead code, against a NeurIPS spotlight and two 2026 duplicates |

**Leg 3 is the only leg where "it is occupied" and "we did too little" point the same way, and
where our own prior measurement predicts failure before any literature is consulted.** Leg 2 is
occupied but *cheap and decidable* — it costs nothing because it folds into leg 1's cell, so
cutting it buys nothing. Leg 1 is the only leg with a residual methodological gap.

**Distinguishing the two failure modes explicitly, as required:**
- **Leg 3 = OCCUPIED.** G-Memory (NeurIPS 2025 spotlight), HMO (`2604.01670`), MM-Mem
  (`2603.01455`) between them hold the three-tier shape, the near/far/profile assignment, the
  promotion policy, and the verbatim→gist objective. There is no version of leg 3 that is
  differentiated by anything we own.
- **Leg 2 = OCCUPIED, but the redundancy is free.** Ground Truth First (`2607.21962`) and
  MemConflict (`2605.20926`) have already built the type system and the read-out. Our arm would be
  a replication on 78 items. Keep it folded, never gate it separately, never call it a design.
- **Leg 1 = WE DID TOO LITTLE.** Chain-of-Note (EMNLP 2024), RECOMP (ICLR 2024) and
  Retain-or-Consolidate (`2607.17545`, concurrent) own the mechanism, but **not** the pairing of a
  measured retrieval-closed stratum with a notes-only unsupported-claim contrast. That gap is real
  and the reason it is unmeasured is that **two of our own arms do not exist yet** (§3.4).

---

## 7. MUST-NOT-CLAIM list (binding on any B08 writeup)

**Leg 1**
1. ❌ *"We introduce query-conditioned notes over retrieved evidence."* — **Chain-of-Note**
   (EMNLP 2024, `2024.emnlp-main.813`), **RECOMP** (ICLR 2024), **Self-Notes** (NeurIPS 2023).
2. ❌ *"notes + raw evidence is our architecture."* — credited **inside our own repo** at
   `longmemeval/compressor.py:8` to the **LongMemEval-V2** winning pattern (`2605.12493`).
3. ❌ *"We introduce query-aware / faithful prompt compression."* — **LongLLMLingua**
   (`2024.acl-long.91`), **LLMLingua-2** (`2024.findings-acl.57`, whose title word is "Faithful").
4. ❌ *"Fabrication risk of abstractive compression is unaddressed."* —
   **EDU-based Context Compressor** (`2512.14244`) anchors EDUs to source indices for exactly this.
5. ❌ *"We introduce a faithfulness / unsupported-claim metric."* — **ALCE**
   (`2023.emnlp-main.398`), **FActScore** (`2023.emnlp-main.741`), **SummaC** (TACL 2022).
   The true statement is *"no such scorer exists in THIS repo"*.
6. ❌ *"We are first to study the raw-vs-consolidated-memory trade-off on LongMemEval."* —
   **Retain or Consolidate?** (`2607.17545`, concurrent) does exactly this, with a learner.
7. ❌ *A retrieval or reranker/RRF improvement as a B08 result.* Forbidden by our own
   `established_measurements.consequence_is_a_CONSTRAINT_not_a_win` (it would help plain text-RAG
   identically).

**Leg 2**
8. ❌ *"Typed memory with validity intervals / supersedes / tombstones / provenance is our
   design."* — **Zep** (`2501.13956`), **Mem0** (ECAI 2025), **Ground Truth First**
   (`2607.21962`), **LatticeMind** (`2608.08236`), **Nous** (`2606.22030`).
9. ❌ *"Stale / contradiction / temporal-update failure is unmeasured."* — **MemConflict**
   (`2605.20926`), **MemSyco-Bench** (`2607.01071`), **Ground Truth First** (`2607.21962`).
10. ❌ *"Principled belief updating will beat last-write-wins on conversational memory."* —
    **Nous** measured the opposite on LoCoMo and explained why (benchmarks lack reliability
    contrast). Claiming it without a reliability-contrastive cell is claiming against evidence.
11. ❌ *"Derived profile from immutable events is our design."* — **MemoryBank** (AAAI 2024) user
    portrait; **HMO** (`2604.01670`) evolving persona.

**Leg 3**
12. ❌ *"Three-tier / pyramidal memory (near / far / profile) is our architecture."* —
    **G-Memory** (NeurIPS 2025 spotlight), **HMO** (`2604.01670`), **MM-Mem** (`2603.01455`),
    **MemGPT** (`2310.08560`).
13. ❌ *"Progressive verbatim→gist distillation across tiers is our idea."* — **MM-Mem**, with an
    information-bottleneck objective.
14. ❌ *"Multi-tier memory serving is unexplored."* — **Pancake** (`2602.21477`).
15. ❌ *Anything presenting `src/memory/l2/`, `src/memory/l3/`, `src/agents/memory_agent.py` as
    current infrastructure.* They are byte-identical to `legacy/src_dead_subsystems/`
    (commit `b63b5a1`) and untracked.

**All legs**
16. ❌ *"可版本化 / 可更新 / 分层 memory" as a novelty claim.* Explicitly forbidden by
    `RELATED_WORK_GAP_AUDIT_20260808.md:146`, and §4/§5 above show every feature has an owner.
17. ❌ Any pooled n=134 number when the two per-type estimates differ in sign
    (`readout_preregistration.per_cell_reporting_rule`; A02's −17.89 pp pooled BABILong headline
    averaged 9 cells with opposite true signs).
18. ❌ Any "fraction of headroom recovered" ratio unless `ACC(A-raw) ≤ 95.0%` and the guard was
    committed **before** scoring (`kill_gate.denominator_guard`;
    `memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md`).

---

## 8. Safe residual claim — one falsifiable sentence (leg 1 only)

Legs 2 and 3 have **no** safe residual claim of their own after §4 and §5; leg 2's is subsumed
into the sentence below via the `knowledge-update` cell, and leg 3's does not exist.

> **On a stratum where first-stage retrieval is measured closed (`any_hit_recall = 1.000` at BM25
> top_k=10 AND re-verified at `evidence_token_budget=4000`; `knowledge-update` n=78 +
> `single-session-assistant` n=56), self-generated query-conditioned notes are an ADJUNCT and
> never a SUBSTITUTE: `Δ_U = U(notes-only) − U(raw)` exceeds +5.0 pp with a 95% paired-bootstrap
> CI entirely above 0, while `Δ_aug = ACC(notes+raw) − ACC(raw)` does not require notes to beat
> raw for the claim to hold.**

**Why this is falsifiable and can lose** (the gate is two-sided, `kill_gate.falsifiability_worked_example`):
KILL iff **all three** fire — K1 `Δ_aug` CI contains 0 (i.e. < 10.82 pp MDE at n=134, disc=0.20),
K2 `Δ_U` CI upper bound < +5.0 pp, K3 `Δ_sub` CI not entirely above −2.0 pp. Then notes merely
**omit** facts without inventing them and adding them changes nothing resolvable — no paper.
The worked KILL numbers are on file (ACC 61.9 / 64.9 / 53.7; U 8.2 / 8.9).

⚠️ **And K2 carries a pre-registered evaluability precondition that must be honoured**: at n=134 the
paired half-width is `1.96·sqrt(disc_U/134)`, so K2's upper bound lands below +5.0 pp **only if
`disc_U ≤ 0.0872`**. If observed `disc_U > 0.0872`, the ALL-THREE kill branch is **unreachable** and
the stratum is **mandatorily** extended to n=500 (4.31 GPU-h). The escalation is a function of
`disc_U` alone, decided **before** any Δ's sign is inspected — that is what stops it being an
n-hack.

**What differentiates this sentence from the closest three works, stated so a reviewer can check
it in one line each:** Chain-of-Note needs retrieval to be *noisy* (we pin it closed);
Retain-or-Consolidate estimates a *calibrated utility* (we measure an *unsupported-claim rate*);
RECOMP optimises *accuracy at a budget* (we hold the budget fixed and vary only composition).

---

## 9. Honest gaps in this adjudication

1. **Semantic Scholar 429 on every call → unused.** Every `arXiv-only` row rests on DBLP CoRR
   and/or the arXiv comment field alone.
2. ⚠️ **MM-Mem's "Accepted by ACL 2026 Main" is an arXiv COMMENT ONLY.** An Anthology search from
   this node returned no result. Per `memory/venue-verify-acl-family-needs-anthology.md` that is
   **unverified**, and I have labelled it so rather than promoting it. **Before citing it, resolve
   the Anthology ID** — if it is genuinely ACL 2026 Main, leg 3's foreclosure gets *stronger*, not
   weaker.
3. ⚠️ **Zep**: a DBLP query including "Graphiti" returned **NO HITS**; only the base title resolves,
   to CoRR 2025. So Zep is `arXiv-only` here. Given how central it is to §4.3, **this is the
   weakest venue attribution in the file for a load-bearing paper.**
4. ⚠️ **DBLP mislabels Findings as main conference** (measured: LLMLingua-2). Anthology IDs are
   given for every ACL-family row; **anyone building `.bib` must copy the Anthology ID.**
5. ⚠️ **Nine of the most load-bearing papers are 2026 preprints** (Retain-or-Consolidate, Ground
   Truth First, MemConflict, MemSyco-Bench, Nous, LatticeMind, HMO, Pancake, MM-Mem, LME-V2). All
   are DBLP `CoRR 2026` or unindexed. Several are certainly under review; **there is no route from
   this node to check systems/ACL-2026 submission status.**
6. ⚠️ **Every 2026 paper here was adjudicated from its ABSTRACT ONLY.** No full text was read. The
   three that decide leg 1's and leg 2's fate — **Retain or Consolidate?** (`2607.17545`),
   **Ground Truth First** (`2607.21962`), **Chain-of-Note** (peer-reviewed, also abstract-level
   here) — must have their method and evaluation sections read before any submission. If
   Retain-or-Consolidate turns out to pin retrieval, **leg 1's residual claim collapses too.**
7. **Searches that returned ZERO relevant results** (this is what "the measurement gap is open"
   rests on): `all:"notes" AND all:"raw evidence" AND all:"reader"`;
   `all:"summary" AND all:"unsupported claims" AND all:"retrieval-augmented"`;
   `all:"compression" AND all:"omission" AND all:"fabrication"`;
   `all:"retrieval" AND all:"held constant" AND all:"context composition"`;
   `all:"summaries" AND all:"instead of" AND all:"raw passages" AND all:"RAG"`;
   `all:"notes" AND all:"long-term memory" AND all:"conversational" AND all:"faithful"`;
   `all:"memory writing" AND all:"summary" AND all:"information loss" AND all:"agent memory"`;
   `all:"tombstone" AND all:"memory" AND all:"LLM"`;
   `all:"supersede" AND all:"memory" AND all:"user profile" AND all:"LLM"`;
   `all:"bitemporal" AND all:"knowledge graph" AND all:"agent memory"`;
   `all:"query-focused summarization" AND all:"long document" AND all:"large language model"`.
   **Caveat: negative arXiv full-text searches are weak evidence** — the field's vocabulary for
   this contrast is unsettled ("consolidation", "gist", "synopsis", "runbook", "notes"), so the
   open gap in §3 could be closed by a paper using a word I did not query.
8. **Zero cross-disk verification.** `/apdcephfs_zwfy6` is not mounted here and ssh was barred, so
   every presence claim inherited from `STATUS.json` is **wzc1-scoped**; before booking a card,
   confirm `models/Meta-Llama-3-8B`, `data/longmemeval/longmemeval_s.json`, and the judge weights
   **on the target node** (`memory/two-disk-rule-applies-to-main-too.md`).
9. **No arXiv-vs-camera-ready diff** for any peer-reviewed row.

---

## 10. Verdict

```
verdict: hold_in_backlog -- novelty gate CLEARED for LEG 1 ONLY, and only for the
         narrowed sentence in section 8; LEG 2 stays FOLDED (never separately gated);
         LEG 3 recommended CUT on literature + our own adverse measurement + dead code
related_work_status: audited
already_dead_should_archive: NO
```

- **Nothing is 完全相同 / 抄袭.** The three most dangerous hits — **Retain or Consolidate?**
  (2026-07), **Ground Truth First** (2026-07), **MM-Mem** (2026-03) — are all **concurrent** and
  none runs B08's decisive contrast (retrieval-closed stratum × notes-only unsupported-claim rate).
  `already_dead_should_archive` is **NOT** warranted.
- **The 严重不足 rating is deserved and this file does not overturn it.** It narrows a
  three-leg portfolio to **one** leg with **one** residual methodological claim, and that claim's
  own implementation (the notes-only arm, the `U` scorer) **does not exist**.
- **Leg 3 should be cut** — see §6. That is a stronger statement than the on-disk
  `portfolio_narrowing_20260814`, which deprioritised leg 3 on *scheduling* grounds only; the
  literature grounds are now on record.
- **Promotion is not warranted.** Blockers unchanged from `remaining_blockers_all_CPU`, all 0 GPU,
  and two of them **are** the novelty.

### 10.1 `STATUS.json` deliberately NOT modified

The `related_work_status: audited` line in §10 is **this file's self-assessment for a human reader,
not a scheduler token.** Measured this session via `proposal/ready_queue.py::read_one()` on a temp
copy in `/tmp` (**no repo file touched**): appending `"related_work_status": "audited"` to B08's
`STATUS.json` flips `novelty_checked` to `True` (`ready_queue.py:203-209`) but B08 **stays
`ready_cpu`**, held by `prior_gate` (its four CPU items, two of which — the `A-notes-only` arm and
the `U` scorer — **are** the novelty per §3.4). The append was still not made, per
`memory/a-declared-lifecycle-is-not-an-adjudicated-one.md`: an agent writing its own clearance field
is not the clearance being reviewed. **B06 is the one where that append WOULD auto-promote to
`ready_gpu`; see B06's §6.1.**
