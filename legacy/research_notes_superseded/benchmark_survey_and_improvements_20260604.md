# Benchmark survey + base-comparison protocol + improvement backlog

date: 2026-06-04 | author: general-purpose-22 (researcher) | mode: research-only (no code, no training)

Context read first: `status/MEMORY_PROTOCOL_PLAN.md`, `status/BENCHMARK_RESULTS.md`,
`ops/research_notes/toy_vs_full_routing_collapse_20260604.md`.

Our method = fixed-size memory bank (128 slots, top_k=16, slot_query+temp40 routing,
per-doc chunk128/256 CPT on Dolmino, cross-chunk EMA writeback) bolted onto a frozen
Llama-3-8B with a light adapter. Current blocker: **retrieval collapses to the 1-2%
noise floor at ≥4k** on BABILong qa1-5; two diagnosed causes — (a) injection dilution
(slot KV gets ~0.2% attn mass) and (b) routing collapse (top1_sim≈uniform 0.01-0.03).
route_aux (routing-supervision aux loss) is the current bet to rescue ≥4k retrieval.

---

## BLOCK 1 — Better benchmarks

### What we use now
BABILong (qa1-5, synthetic bAbI facts in PG19 haystack, 0k-32k) + LongBench (6 QA tasks).
BABILong is good for *needle reasoning under distractors* but: (i) it is largely a
single-fact-retrieval-under-noise probe, low task diversity; (ii) every length cell is
the *same* questions re-padded, so it does not separate "lost-in-the-middle" from
"compression-budget" failures; (iii) saturates fast on strong models (Llama-3.1-8B
hits 80-100 on qa1) → low discrimination at the top.

### Candidate benchmarks (verified arXiv IDs + HF availability)

| Benchmark | arXiv | What it tests | Length | HF dataset | Fit for fixed-memory compression | Eval difficulty |
|---|---|---|---|---|---|---|
| **RULER** | 2404.06654 | 13 synthetic tasks: multi-needle (NIAH single/multi-key/multi-value/multi-query), variable tracking, frequent-word extraction, QA | 4k-128k, controllable | `simonjegou/ruler` + NVIDIA gen scripts | **High** — synthetic, *length-controlled*, multi-value/multi-query directly stress a fixed slot bank; isolates effective capacity | Medium (need their generation harness; flexible-match scoring) |
| **HELMET** | 2410.02694 | 7 categories incl. RAG, re-ranking, long-doc QA, ICL, summarization; **model-based + controlled-length** scoring | 8k-128k | `princeton-nlp/HELMET` | **High** — explicitly designed for *comparable* length controls + robust scoring; reduces saturation/variance issues of older suites | Medium-High (model-judge scoring, more compute) |
| **SCBench** | 2412.10319 | **KV-cache-centric**: string/semantic retrieval, multi-task, global vs local, **shared-context / multi-turn** reuse | up to 128k | `microsoft/SCBench` | **Very High (most aligned)** — the only suite built to evaluate *KV-cache reuse / compression methods specifically*; measures whether a compressed cache survives multi-turn reuse, exactly our claim | Medium |
| **∞Bench (InfiniteBench)** | 2402.13718 | 12 tasks (retrieve, code, math, novel QA/summarization) avg >100k tokens | 100k+ | `xinrongzhang2022/InfiniteBench` | Medium — good for *extreme* length stress past 32k, but realistic tasks are hard and may conflate LM quality with compression | Medium |
| **LongBench-v2** | 2412.15204 | 503 hard multiple-choice, deep understanding/reasoning, human-hard | 8k-2M words | `THUDM/LongBench-v2` | Medium — MCQ → clean scoring, but reasoning-heavy; tests downstream usefulness not addressing per se | Low (MCQ accuracy) |
| **LongMemEval** | 2410.10813 | 5 long-term *interactive memory* abilities (multi-session reasoning, temporal, knowledge update, abstention) over chat history | ~115k | `xiaowu0162/LongMemEval` (GitHub release) | Medium-High — closest to "memory module" framing; tests write-then-recall across sessions, which is our EMA-writeback claim, but it is chat/agent-shaped (needs instruct behavior) | Medium |
| **LOFT** | 2406.13121 | retrieval/RAG/SQL/multi-hop reframed as long-context, up to 1M | Google release (GitHub) | Hard — 1M tokens infeasible at our 8B+adapter scale right now | High |
| **ZeroSCROLLS** | 2305.14196 | 10 zero-shot long-doc NLP tasks (summ/QA/aggregation) | up to ~10k | `tau/zero_scrolls` | Low-Medium — mostly <32k, older, lower discrimination | Low |
| **NIAH multi-needle / multi-value** | (covered by RULER) | pure addressing capacity | controllable | via RULER | High as a *diagnostic*, not a paper headline | Low |
| KV-compression benchmark (longctx_bench) | 2407.01527 | unified harness comparing 10+ KV-compression methods across 7 task types | — | GitHub `henryzhongsc/longctx_bench` | **High as a methodology reference** — shows the standard task taxonomy + how to report compression-vs-quality tradeoff curves | (reference, not a dataset) |

### Ranked recommendation —接入哪 2-3 个

1. **SCBench (2412.10319) — TOP PRIORITY.** It is the *only* benchmark purpose-built to
   evaluate KV-cache compression / reuse, which is literally our thesis ("fixed memory
   buffer compresses long context"). It measures whether a compressed representation
   still answers correctly *and* survives multi-turn reuse — our EMA-writeback story.
   Gives us the most defensible "our compression is competitive" headline. **confidence: high.**

2. **RULER (2404.06654) — second.** Length-controlled synthetic addressing/tracking with
   **multi-value / multi-query NIAH** variants. These directly probe whether a 128-slot
   bank has enough effective capacity — our exact failure mode. Cheaper than HELMET,
   string-match scoring, and far more discriminative than BABILong at the top. Use it as
   the *capacity diagnostic* replacing/augmenting BABILong. **confidence: high.**

3. **HELMET (2410.02694) — third, for the paper.** Provides comparable length controls
   and robust (model-judge) scoring across realistic categories (RAG/long-QA/ICL), which
   reviewers now expect over BABILong-only tables. More compute, so adopt after SCBench+RULER.
   **confidence: medium.**

Keep BABILong qa1-5 as a legacy continuity anchor (we already have P8/baseline numbers),
but stop treating it as the primary discriminator. Skip LOFT for now (1M tokens infeasible
at current scale); revisit ∞Bench/LongMemEval once we clear the ≥4k cliff.

---

## BLOCK 2 — Correct methodology for comparing against the base model

### Standard protocol in memory-augmented LLM papers
Memory papers (LM2 2502.06049, Memorizing Transformers 2203.08913, Titans 2501.00663,
RMT, Infini-attention 2404.07143) all report the **same backbone with and without the
memory mechanism**, holding everything else fixed, and add **at least one truncation/
sliding-window base** at a matched compute/KV budget. The base is *not* "a different
model" — it is "our adapter removed". Three canonical base arms:

- **B0 — truncation base**: frozen Llama-3-8B, no memory, feed only the last *W* tokens
  that fit the same KV budget the memory occupies (e.g. last 128/256·k tokens). This is
  the honest "what you lose without memory" arm.
- **B1 — sliding-window base**: same frozen model, sliding window over the full input
  (no compression), answers from the window that contains the query. Tests whether memory
  beats naive locality.
- **B2 — full-attention-if-fits base**: frozen Llama-3-8B with full attention up to its
  native 8k context (Llama-3) / 128k (Llama-3.1). This is the *upper bound* the
  compression should approach without paying full KV cost. (Our `Plain Llama` numbers in
  BENCHMARK_RESULTS.md are this arm at 1B; redo at 8B.)

### What must be controlled for a fair claim
1. **Matched KV budget** — the headline. Memory = 128 slots; the truncation base must be
   given a KV cache of *the same token-equivalent size* (slots × slot-dim ≈ N base tokens).
   Reporting "we beat full truncation to 256 tokens using a 128-slot memory" is the
   compression win. SCBench/longctx_bench (2407.01527) plot quality-vs-compression-ratio
   curves — adopt that framing.
2. **Identical prompt construction & answer parsing** — same chat template (or same
   no-template), same chunking of the haystack, same generation config (greedy, max_new),
   same metric (babilong.metrics flexible match). Differences here are the #1 source of
   spurious gaps.
3. **Matched FLOPs (secondary)** — note inference FLOPs/latency; memory should win on
   *KV memory* even if FLOPs are comparable. Report both.
4. **Same eval data & n** — same qa tasks, same lengths, same n=100 cells.

### Known base anchors (already in BENCHMARK_RESULTS.md / literature)
- **Llama-3-8B-Instruct vanilla, BABILong (paper 2406.10149, Table 4):** qa1 0k=98 →
  4k=16 → 8k=7 → 32k=23; qa2 0k=47→32k=2; qa5 0k=85→32k=50. **Sharp cliff at 4k for
  Llama-3-8B (8k native ctx)** — this is the curve our memory must beat past 4k.
- **Llama-3.1-8B-Instruct (same table):** qa1 4k=95, 8k=83, 32k=87 — the 128k-native
  model barely degrades. If we want to claim "memory helps", compare against **Llama-3-8B
  (8k)** as the budget-limited base, not 3.1; otherwise the base already solves it.
- **Our P8 (8B, L1+L3, 500 steps):** overall 49.1 vs paper Llama-3-8B-It ≈42.6 (+6.5pp) —
  but that gap is mostly ≥8k where the 8k base falls off. Re-run a clean B0/B1/B2 at 8B
  with *our* harness for an apples-to-apples table.

### Concrete base-vs-ours experiment design (uses existing scripts)
- **Ours arm**: current mem_space adapter checkpoint, `eval_longbench_mem_space.py` +
  BABILong eval (qa1-5 × 0k-32k × n=100), babilong.metrics.
- **B0 truncation**: same `Plain Llama` path you already ran for 1B, but at Llama-3-8B,
  truncate input to last K tokens where K = slot-token-equivalent of the 128-slot bank
  (compute K from slot_dim·num_slots / hidden). Run on the *same* qa cells.
- **B1 sliding-window**: HF generate with a sliding window = native ctx, no memory.
- **B2 full-if-fits**: Llama-3-8B full attention, cells ≤8k only (mark >8k as OOC).
- **Report**: one table, 4 arms × (qa × length), plus a **quality-vs-KV-budget curve**
  (x = KV tokens stored, y = avg acc) showing our 128-slot point sits above the B0
  truncation curve at equal budget. That curve *is* the paper's central figure.

confidence: high that this is the expected protocol; the only judgment call is the exact
slot→token-equivalent budget mapping (state your assumption explicitly in the table).

---

## BLOCK 3 — How to fix our method (≥4k retrieval collapse + injection dilution)

Two sub-problems, with the most relevant recent work for each.

### 3a. Against injection dilution (memory gets ~0.2% softmax mass)

| # | Work | arXiv | Takeaway | What we borrow |
|---|---|---|---|---|
| 1 | **YOCO (You Only Cache Once)** | 2405.05254 | Decoder-decoder: a self-decoder builds a *global* KV cache once, a cross-decoder reads it via **dedicated cross-attention** — memory is a first-class, separately-attended store, not diluted into the local softmax | Give slots their **own cross-attention layer** (separate softmax) instead of prepending to the live-token KV. Directly cures the "0.2% mass" problem — memory no longer competes with 1024 live tokens. |
| 2 | **Memorizing Transformers** | 2203.08913 | A single layer attends to a **non-differentiable kNN external memory** via a *separate* attention head with a learned gate combining local vs memory attention | The **per-head learnable gate** between local and memory attention is exactly the `inject_gate` we have, but theirs is content-dependent and actually opens. Make our gate per-head + initialize larger. |
| 3 | **Infini-attention** | 2404.07143 | Compressive memory with a **delta rule** update + a learned scalar gate β mixing local attention and memory readout; memory readout is normalized so it is not drowned out | Adopt the **delta-rule writeback** (write only the *residual* not already retrievable) and the **normalized memory readout** so its magnitude is comparable to local attention output before gating. |
| 4 | **Titans** | 2501.00663 | Neural long-term memory updated by a **surprise** signal at test time; memory is injected as extra tokens/branch with a gating MLP; persistent + contextual memory separated | The **surprise-gated write** (write more when prediction error is high) is a principled writeback signal beyond EMA. Also their **persistent memory tokens** = register slots (see 3b #3). |
| 5 | **Gist tokens** | 2304.08467 | Compress a prompt into a few **gist tokens** via attention masking; the gist KV is then the only thing attended | Validates that a tiny fixed set of tokens *can* carry compressed context — but only when the LM is **forced** to read solely from them (masking). Suggests an aux/masking curriculum where live tokens are dropped so slots must carry the signal. |
| 6 | **ICAE (In-Context AutoEncoder)** | 2307.06945 | Learn memory slots with an **autoencoding reconstruction objective** (encode context → few slots → decode) | This is our rejected P1 recon, but ICAE shows it works *when the decoder reads only the slots* (information bottleneck enforced). Our P1 failed likely because the LM still saw live tokens → no bottleneck pressure. Reconsider recon **with live-token masking**. |

### 3b. Against routing collapse (top1_sim→uniform, dead slots)

| # | Work | arXiv | Takeaway | What we borrow |
|---|---|---|---|---|
| 1 | **Loss-Free Balancing for MoE** | 2408.15664 | Replaces load-balance *aux loss* (which fights the task gradient and can *cause* collapse toward uniform) with a **per-expert bias added to routing logits**, adjusted online to equalize usage — no gradient interference | **Direct fix for our diagnosed bug**: our `load_balance`+`entropy` aux *push routing toward uniform* and starve the selector (toy_vs_full report cause #2). Replace them with a usage-bias term that balances slots **without** a uniform-pushing gradient. |
| 2 | **Memory Layers at Scale** | 2412.09764 | Product-key memory with up to 128B params; uses **product-key lookup** + careful init + balancing for sparse addressing at scale; shows sparse memory addressing *can* be trained stably | Their **product-key trick** makes top-k over many slots cheap and trainable; their balancing/init recipe is a reference for keeping 128 slots all alive. |
| 3 | **Vision Transformers Need Registers** | 2309.16588 | Adding a few **register tokens** absorbs the "attention sink" garbage that otherwise corrupts routing | Add **always-on register slots** (we already have `--num_global_slots`, v7) so the addressable slots stop being hijacked as attention sinks → cleaner content routing. Cheap, confidence high. |
| 4 | **Landmark Attention** | 2305.16300 | Each block gets a **landmark token**; attention first selects landmarks then attends within — a *differentiable, trainable* coarse-to-fine retrieval that does not collapse | Supervised-by-construction coarse routing: a landmark/slot is *tied to a chunk*, giving a built-in write target → exactly the **route-supervision target** route_aux needs. |
| 5 | (method note) ST-Gumbel / orthogonality | — | Straight-through Gumbel top-k keeps routing differentiable but discrete; key-orthogonality reg prevents key smear (note: our `key_repulsion=1.0` is 20× the toy and may *over*-smear keys per the collapse report) | **Lower key_repulsion** (1.0→0.05) and consider ST-Gumbel on the top-k selection so the selected slots get a real gradient even with decoupled-read. |

### Synthesis with our diagnosed root cause
The toy_vs_full report's decisive finding: **LM loss alone never bootstraps content
addressing**, and the uniform-pushing aux losses actively cause collapse. The literature
agrees on the two-pronged fix:
1. **Replace uniform-pushing aux with loss-free balancing (2408.15664)** so balancing
   stops fighting the task — *and* add an explicit **route-supervision aux** (write-target
   CE, à la Landmark 2305.16300) so `scores` get a real gradient. (route_aux you are
   testing = the right direction; pair it with removing/neutralizing load_balance+entropy.)
2. **Give memory its own cross-attention (YOCO 2405.05254 / Memorizing Transformers
   2203.08913)** with a content-dependent, larger-init gate (Infini-attention 2404.07143)
   so the read path actually carries ≥several-% mass — otherwise even perfect routing
   yields ~0 LM gradient (collapse report cause #4).

### Prioritized improvement backlog (for plan P7+)

| Pri | Item | Borrowed from | Confidence | Est. change | Rationale |
|---|---|---|---|---|---|
| **P7** | **Route-supervision aux** (CE of scores → known write_idx) + **neutralize load_balance/entropy** (drop or switch to loss-free bias 2408.15664) | 2305.16300, 2408.15664 | **high** | small (layer.py aux + train loss collect; remove 2 aux terms) | Directly targets the *measured* collapse mechanism (severed selector gradient + uniform-pushing aux). Cheapest decisive lever. Already partly in flight (route_aux). |
| **P8** | **Dedicated memory cross-attention** read path (separate softmax) replacing KV-prepend dilution; per-head content-dependent gate, larger init | 2405.05254, 2203.08913, 2404.07143 | **high** | medium (new read module + gate; isolate behind `--use_memory_xattn`) | Cures the ~0.2% attn-mass dilution that makes LM gradient ≈0. Without this, P7 routing fix still has no signal to learn from. |
| **P9** | **Always-on register slots** (extend `--num_global_slots`) | 2309.16588 | **high** | small (config + selector) | Absorbs attention-sink garbage so addressable slots do real content routing; we already have the knob (v7). Quick win. |
| **P10** | **Lower key_repulsion 1.0→0.05** + ST-Gumbel top-k | collapse report cause #5, ST-Gumbel | **medium** | tiny (hyperparam) + small (ST gating) | key_repulsion=1.0 (20× toy) smears keys toward uniform; cheap to test. |
| **P11** | **Delta-rule + normalized writeback** (write residual not already retrievable; normalize readout magnitude) | 2404.07143 | **medium** | medium | Improves *what* gets stored and keeps readout magnitude comparable to local attention before gating. |
| **P12** | **Reconsider recon WITH live-token masking / bottleneck** (P1 failed w/o bottleneck) | 2307.06945, 2304.08467 | **low-medium** | medium | ICAE/gist show recon works only when the LM is forced to read solely from slots. Our P1 had no bottleneck → re-test with masking before fully discarding. |
| **P13** | **Surprise-gated write** (write strength ∝ prediction error) instead of fixed EMA | 2501.00663 | **low** | medium | Principled writeback signal; lower priority until P7/P8 unblock retrieval. |

Recommended order: **P7 + P9 together** (both small, both attack collapse) → if retrieval
moves off the noise floor but stays weak, add **P8** (read-path mass) → then P10/P11 tuning.
P12/P13 are research bets after the cliff is broken.

---

## Verified arXiv IDs (for citation)
RULER 2404.06654 · HELMET 2410.02694 · SCBench 2412.10319 · ∞Bench/InfiniteBench
2402.13718 · LongBench-v2 2412.15204 · LongMemEval 2410.10813 · LOFT 2406.13121 ·
ZeroSCROLLS 2305.14196 · KV-compression benchmark (longctx_bench) 2407.01527 · BABILong
2406.10149 · LM2 2502.06049 · YOCO 2405.05254 · Memorizing Transformers 2203.08913 ·
Infini-attention 2404.07143 · Titans 2501.00663 · Gist tokens 2304.08467 · ICAE
2307.06945 · Loss-Free Balancing (MoE) 2408.15664 · Memory Layers at Scale 2412.09764 ·
ViT Need Registers 2309.16588 · Landmark Attention 2305.16300.
