# Raw-KV Readout Proposal — replacing the trained selector + lossy slots

> **Scope.** Pure design doc for main-session + coder. No code changed, no GPU, no training.
> **Author intent.** Answer the user's question — *"如果不用 selector,该怎么做?直接用当前 chunk 和 mem space 做 cross-attention 么?"* — with a code-grounded, falsifiable migration path that does **not** repeat the negatives already exhausted in `docs/32K_WALL_FINDINGS.md`.
> **Anchors read.** `selector.py:39` (TopKSelector), `selector.py:1375` (MemoryCrossAttentionRead), `selector.py:1509` (`.read`), `memory_bank.py:33` (MemoryBank), `chunk_memory_bank.py:16` (ChunkMemoryBank), `layer.py:639/2408/2435` (read wiring), `config.py:444` (use_memory_xattn), `inattn_kv.py` (TRUE in-attention KV concat, in-graph + multi-layer capable).
> Written 2026-06-19.

---

## 0. TL;DR

- **Root-cause confirmation (code-level):** our failure is **NOT in the read mechanism** — `MemoryCrossAttentionRead.read` already does a dense softmax over **all N slots** (+ null sink); the trained `TopKSelector` only gates **writes**. The failure is in the **memory-unit content**: ~95% of read mass lands on never-written slots that still hold chunk-0 token snapshots, because the *write* side (lossy compression + sparse top-k write) never deposits the needle into a recoverable form. The "selector is broken (0% needle precision)" finding is a **write-gating** symptom, not a read-path symptom.
- **Recommended direction: Method A (per-chunk raw-KV + emergent gist-key soft attention), but it MUST be paired with partial reader unfreeze.** Method A removes the lossy bottleneck (the one bottleneck the negatives have NOT closed) and removes the trained selector (replaced by train-free emergent selection, exactly the Landmark mechanism). It is the only candidate that differs *structurally* from every exhausted negative.
- **Hard caveat from our own evidence:** the in-attention **oracle-only** probe proved a *frozen* Llama-3-8B cannot consume even perfectly-injected raw KV (21.0 ≈ OFF 22.0, `76efbd4`). Landmark breaks the wall *because it full-fine-tunes the reader on the grouped-softmax path*. **Therefore raw-KV readout without unfreezing the reader will reproduce the in-attn oracle negative.** Method A's novelty over the dead probe is precisely: train the reader (or L16–31) *on* the raw-KV-concat path so it learns to consume it — the one thing the probe never did.
- **Minimum landing entry point:** `inattn_kv.py` (already in-graph, already multi-layer via `--inattn_kv_layers`) + a new per-chunk raw-KV store + a tiny gist-scorer, trained with `--use_inattn_kv` and partial unfreeze (`launch_sft_unfreeze_inattn.sh` plumbing). **Largest risk: gist-scorer is train-free in Landmark but our single-window dolmino training never exercises a cross-chunk retrieval path → the scorer gets no gradient and stays random (same dead-retriever trap landmark-s5 found).** Mitigation = multi-window training (S3) is a *prerequisite*, not optional.

---

## 1. 现状确认 — 读机制 vs 记忆单元,根因到底在哪

### 1.1 读已经是 dense all-N(逐行核对代码)

`MemoryCrossAttentionRead.read` (`selector.py:1509-1672`):

```
Q = q_proj(hidden_states)            # [B,T,H,D]  live-token queries
K = k_proj(slot_keys)                # [B,N,Hkv,D]  ALL N slots
V = v_proj(slot_values)              # [B,N,Hkv,D]  ALL N slots
# (optional) append ONE null/sink column  → [B,H,N+1,D]
attn_logits = Q·Kᵀ·scale            # [B,H,T,N+1]   over ALL slots
attn_weights = softmax(attn_logits)  # dense over every slot
read_out = out_proj( (attn_weights·V) * per_head_gate )
```

- The softmax denominator spans **all N slots** (+ sink). There is **no top-k mask on the read path**. `TopKSelector` is never called here.
- Wiring (`layer.py:2435`): `memory_xattn.read(hidden, xattn_slots, xattn_slots, ...)` passes the **full** `[B,N,slot_dim]` bank. `dead_mask` / `mass` are **diagnostic / additive-bias only** (`_last_dead_slot_read_mass` is `no_grad` telemetry; mass bias is a detached constant). They do not gate which slots are visible.
- This matches SESSION_HANDOFF §1: *"P8 用 use_memory_xattn 时,读已是 all-N,top-k 只 gate 写入。"* **Confirmed in code.**

### 1.2 那 0% needle precision 是哪里的问题?

`TopKSelector` (`selector.py:39`) scores N slots against a **pooled** chunk-hidden summary (`Q_sel`/`K_sel` + learnable per-slot keys), STE hard top-k, MoE load-balance aux. Its output `idx:[B,top_k]` is consumed **only by the writeback path** (which slots get the delta-rule / dual-gate update). So:

- 0% needle precision = **the write selector never deposits the needle's chunk into a slot in a recoverable form**, OR routes it to a slot that later gets overwritten / averaged away.
- Diagnosis (`32K_WALL_FINDINGS.md §1` + SESSION_HANDOFF §1): only ~91/128 slots are ever written (`dead_slot_frac`), so the read's dense softmax spreads ~95% of its mass over **never-written slots** that still hold the lazy-init chunk-0 token snapshot (`memory_bank.py:11-17` lazy `hidden_pool` init + noise). Live slots carry only ~5%.

### 1.3 根因判定

> **The read mechanism is fine. The memory-unit *content* is the problem, in two compounding ways:**
> 1. **Lossy compression** — each written slot is an adapter-compressed summary of a 512-token chunk (250:1 at 32k). ACL2025 gist data: 16× compression → 9.6% reconstruction. At 250:1 a specific 6-digit needle is information-theoretically **not byte-recoverable**. (`32K_WALL_FINDINGS.md §3`.)
> 2. **Dead-slot pollution** — sparse top-k writes leave ~91/128 slots frozen at init snapshots, which dominate the dense read softmax.

Critically, **`32K_WALL_FINDINGS.md §家族C` already proved** that even if you hand the *frozen* reader perfect content (in-attn oracle-only, top1_sim=0.94), it scores 21.0 ≈ OFF 22.0. So fixing content alone (giving it raw KV) is **necessary but not sufficient** — see §4.

---

## 2. 三个候选重构方案

### Method A — gist-key + raw-KV soft attention (Landmark-式,emergent selection)

**架构 (forward 伪代码).** Replace the 128 lossy slots with a **per-chunk raw-KV store** + a **per-chunk gist key**; selection is an *attention weight*, not a separate head.

```
# WRITE (per chunk c, no compression):
store.K[c], store.V[c] = native k_proj/v_proj(chunk_tokens)   # raw KV, NOT compressed
store.gist[c] = pool_or_landmark_token(chunk_tokens)          # ONE gist vector per chunk
# (gist = a trainable <landmark>-style token's hidden, or attention-pooled chunk repr)

# READ (current chunk query q, at reader layer ℓ):
sim = q · store.gist[:]ᵀ                       # [T, n_chunks]  query vs every chunk's gist-key
w   = softmax(sim)                             # soft selection — NO STE, NO top-k head, NO load-balance
# (optionally soft-top-k: keep highest-w chunks, renormalise — still differentiable, still no separate head)
# Inject the selected chunks' RAW KV directly into layer ℓ's self-attention (ONE softmax):
attn over [native_KV ; Σ_c w[c]·store.KV[c]]   # landmark §4b in-attention concat
```

**相对现有代码的最小 diff.**
- **Reuse `inattn_kv.py` almost wholesale** — `build_retrieved_kv` already projects retrieved raw hidden → native k/v_proj + real-source RoPE; `make_inattn_attention_forward` already concats `[native_KV ; retrieved_KV]` in one softmax. It is **already in-graph** (`forward` runs under autograd) and **already multi-layer** (`--inattn_kv_layers "16,20,24"`). This is the single biggest reuse win.
- **New: a per-chunk raw-KV store** — a thin variant of `ChunkMemoryBank` (`chunk_memory_bank.py:16`) that, instead of one pooled hidden per chunk, keeps the chunk's *raw token hidden states* (or pre-projected K/V) + one gist vector. `ChunkMemoryBank.top_k` (cosine sim) is the embryo of the gist-scorer.
- **New: gist-scorer** = a single trainable `<landmark>`-token embedding (Landmark-faithful) **or** a 1-layer attention-pool producing `store.gist[c]`. **No `Q_sel`/`K_sel`/STE/load-balance** — those are all `TopKSelector` machinery to be *deleted* from this path.
- **Bypass `TopKSelector` and `MemoryBank` entirely** for this readout (keep them for backward-compat behind the existing flags).

**为什么可能破墙.** Removes BOTH §1.3 bottlenecks simultaneously: (1) raw KV = no lossy compression → needle is byte-recoverable; (2) soft attention over per-chunk gist-keys = no dead slots, no separate broken selector. This is *exactly* the Landmark mechanism that reproduced the wall-break to ~31k (`LANDMARK_VS_MEMSPACE_DIFF.md` S0).

**风险/已知坑.**
- **(critical) gist-scorer gets no gradient under single-window training.** landmark-s5's `llama_mem.py` scan (SESSION_HANDOFF §0 重磅发现): Landmark *trains* gist-keys via in-window grouped-softmax and the top-k block selection is *inference-only / train-free*. Our dolmino training is single-window (n_ctx small) → if retrieval never fires in the training forward, `store.gist`'s scorer is never in the loss graph. **Prerequisite: multi-window training (DIFF axis S3)** so the cross-chunk retrieval path actually fires and back-props. This is the same dead-retriever trap landmark-s5 flagged (FSDP grad-ckpt forces `use_cache=False` → cross-window KV not cached → retrieval dead). Must verify gradient flow (cheap fwd+bwd probe) **before** committing a full run.
- **Memory cost of raw KV** — see Method B; at 32k this is O(n) KV cache, not O(n²). Manageable with chunked store.
- **Frozen reader cannot consume it** — see §4. **Must unfreeze.**

---

### Method B — dense full cross-attention over all raw KV (no selection at all)

**架构.** Skip gist-scoring; current chunk's query attends over the raw KV of **all** prior chunks directly.

```
read: attn over [native_KV ; concat_c store.KV[c]]   # every historical token visible, one softmax
```

**最小 diff.** Same `inattn_kv.py` concat, but `retrieved_KV` = *all* stored chunks (no scorer). Even simpler than A — delete the scorer too.

**显存可行性 (must address).** This is the crux. Naïve full attention is **not** O(n²) in *memory* — it is O(n) KV cache + O(n²) *compute* per query block. At 32k tokens, Llama-3-8B: KV cache = 32k × 32 layers × 8 kv-heads × 128 dim × 2 (K+V) × 2 bytes ≈ **~34 GB** for full-layer injection — feasible on H20 (97.8 GB) / L20A (183 GB) for a *single layer* (~1 GB) but heavy if injected at all 32 layers. The O(n²) *attention compute* at 32k is what Landmark/streaming avoid by **top-k block retrieval**. Mitigations: (a) inject at a *few* layers only (`--inattn_kv_layers "16,20,24"`); (b) sliding-window + the raw-KV store as the "long-range" tail (StreamingLLM/landmark hybrid); (c) chunked/flash attention over the concatenated KV. **But full dense over all 32k tokens largely defeats the project's purpose (fixed-budget compression)** — it becomes "just use long context", which is the open-book upper bound we measure *against*, not a memory method.

**为什么可能破墙 / 风险.** It would trivially break the wall (it is open-book), but it is **not a fixed-budget memory architecture** — it contradicts the project's core constraint (固定大小 memory buffer). Useful only as an **upper-bound diagnostic / ablation** (does the reader consume raw KV at all when given everything?), not as the destination. **This is essentially the SWA(W6) open-book ceiling we already treat as "过高标尺" (`32K_WALL_FINDINGS.md` 方法论).**

---

### Method C — keep compressed slots, drop trained selector, read all-N

**架构.** Stop training `TopKSelector`; keep the 128 lossy slots; read stays dense all-N (which it already is).

**最小 diff.** Essentially none on the read path — set write-gating to dense or freeze the selector.

**论证.** **This is ~already the current state and is therefore ineffective.** §1.1 proved the read is *already* dense all-N regardless of the selector. The selector only gates *writes*. Dropping selector training doesn't fix the two real bottlenecks (lossy slots + dead-slot pollution). The `dense 全局写(top_k=N)` variant was **already tried and failed** (SESSION_HANDOFF §「dense 全局写」: generation collapse, qa5 2k=0%). So C is either a no-op (freeze selector, read unchanged) or reproduces the known dense-write collapse. **Reject.**

---

## 3. 横向比较 + 推荐

| 维度 | Method A (gist-key + raw-KV) | Method B (dense full raw-KV) | Method C (slots, no selector) |
|---|---|---|---|
| **破墙潜力** | **High** — removes both bottlenecks, == Landmark mechanism that demonstrably broke the wall | Trivially high but it's open-book, not a memory method | **None** — ≈ current state / known dense-write collapse |
| **改动成本** | Medium — reuse `inattn_kv.py` (in-graph, multi-layer) + new per-chunk raw store + tiny gist-scorer; **needs multi-window training (S3) + partial unfreeze** | Low (delete scorer) but needs memory engineering | Trivial |
| **显存** | O(budget) — store top-k chunks' raw KV; tunable | O(n) KV cache, O(n²) compute at 32k — heavy, defeats fixed-budget premise | O(N·slot_dim) — cheap (current) |
| **与已穷尽负结果的区别** | **Differs structurally**: (1) content = raw KV (not lossy slot, not prefix block); (2) injected in-attention *and trained on that path* (not eval-time wrapper); (3) selection emergent (not trained selector). The in-attn oracle negative used a **frozen** reader + eval-time injection — A trains the reader on the path. | Same in-attn concat as the dead oracle probe, just more KV. **Frozen → reproduces oracle negative.** Only differs if reader unfrozen. | **Identical** to current / known dense-write negative |

**Recommended: Method A, gated behind two hard prerequisites.**

**Reasoning.** A is the only candidate that differs *structurally* from every exhausted negative in `32K_WALL_FINDINGS.md`:
- vs **raw-KV-prefix (+1.0)** and **evidence-prefix (+2.5)**: those injected as a **position-0 prefix block** the frozen reader couldn't consume. A injects **in-attention with real source RoPE** AND **trains the reader on that path**.
- vs **TRUE in-attn oracle (−1.0, the decisive negative)**: that was a **frozen** reader + **eval-time** wrapper (never trained-in). A's whole novelty = the reader is **trained to consume** the concatenated raw KV (this is precisely how Landmark differs — it full-fine-tunes on grouped-softmax).
- vs **dense 全局写 collapse**: A doesn't broaden the *write* over lossy slots; it stores *raw* KV and selects via *emergent* attention.
- vs **training-window / capacity / mass sweeps**: A changes the **memory unit + readout path**, the axes those sweeps never touched.

B is reserved as an **upper-bound ablation** (one cheap run: does the reader consume raw KV when given everything? If even B with unfreeze fails, the project premise is in deeper trouble). C is rejected (no-op / known collapse).

---

## 4. 与 frozen-reader 决定性负结果的关系 — 必须解冻

This is the load-bearing constraint, and it is non-negotiable based on our own evidence.

**The decisive negative (`32K_WALL_FINDINGS.md §家族C`, commit `76efbd4`):** TRUE in-attention **oracle-only** injection — perfect needle, top1_sim=0.94, real RoPE, injected into L16 self-attn in one softmax — scored **21.0 ≈ OFF 22.0**. Conclusion the team already drew: *"冻结 reader 即便拿到完美注入的正确 KV 内容,也无法 CONSUME 来答题。"*

**Method A's read path is mechanically the same `inattn_kv.py` concat as that dead oracle probe.** So if Method A uses a **frozen** reader, it **will reproduce the oracle negative**, full stop. Storing raw KV instead of slots fixes the *content* bottleneck but the oracle probe already had perfect content — the bottleneck it exposed was **consumption**, and that is the frozen reader.

**Why Landmark is not subject to this:** Landmark **full-fine-tunes every parameter** (embed + attn + FFN + lm_head + the new `<landmark>` embedding) **on the grouped-softmax retrieval path** (`LANDMARK_VS_MEMSPACE_DIFF.md` axis 1). The reader *learns* to consume retrieved raw KV. That training-on-path is the difference between Landmark's 94–100%@31k and our oracle's 21.0.

**Therefore Method A MUST be paired with reader unfreeze** — at minimum the upper layers L16–31 (the `launch_sft_unfreeze_inattn.sh` v2 partial plan), ideally a fuller fine-tune if compute allows (Landmark did full + 15k steps × 0.98B tok). `32K_WALL_FINDINGS.md §4` already names this as **"唯一尚未关闭的路径 = 解冻 reader 部分层 finetune"**. Method A is the concrete architecture that makes that unfreeze meaningful (gives the reader a *raw-KV* path worth learning to consume, instead of a lossy-slot path that has no recoverable needle to consume).

**Known risk on unfreeze:** `LANDMARK_VS_MEMSPACE_DIFF.md` axis 1 hazard — a **1k-step full FT on narrow data damaged base NIAH (OFF 22→11/12)**. Too few steps / too narrow data hurts before it helps. So unfreeze must come with **enough data + steps** (Landmark's 0.98B-tok / 15k-step scale is the reference), not a quick 1k-step pass. This couples Method A to the **S2 data axis** (more/broader data) as well.

**Net dependency chain for Method A to have a chance:**
`raw-KV store + gist-scorer (replaces selector)` **AND** `multi-window training so the scorer gets gradient (S3)` **AND** `reader unfreeze L16–31+ on sufficient data/steps (S2 + axis-1)`. Removing any one reproduces a known negative (frozen→oracle negative; single-window→dead retriever; lossy slots→compression negative).

---

## 5. 最小落地路径 (供 coder)

1. **New `RawKVChunkStore`** (variant of `chunk_memory_bank.py`): per chunk store raw token hidden states (or pre-projected K/V) + one gist vector; cap to a budget of B chunks (FIFO or salience). Reuse `ChunkMemoryBank.top_k` cosine-sim as the gist scorer embryo.
2. **Gist key** = trainable `<landmark>`-style token embedding appended per chunk (Landmark-faithful, minimal params) OR attention-pool. **No `TopKSelector`.**
3. **Readout** = reuse `inattn_kv.py`: at read, compute `softmax(q·gistᵀ)` → soft-top-k chunks → feed their raw hidden as `retrieved_hidden` to `build_retrieved_kv` → `make_inattn_attention_forward` concats in-attention at `--inattn_kv_layers`. Already in-graph.
4. **Training:** `--use_inattn_kv` + multi-window context (S3) so the retrieval path fires and the gist-scorer + reader get gradient. **Verify gradient flow with a cheap fwd+bwd probe first** (`_inattn_grad_probe`, already in `train_mem_space_dolmino_cpt.py:2672`) — confirm `store.gist` scorer params have non-zero grad. If gradient does not flow, that is itself a finding (pivot).
5. **Unfreeze L16–31** via `launch_sft_unfreeze_inattn.sh` plumbing, with **S2-scale data + steps** (not a 1k narrow pass).
6. **Gate:** native passkey 0–32k + BABILong qa1 (the locked DIFF protocol). Killer = a cliff at a length the Landmark anchor did not have.

---

## 6. 一句话结论

**走 Method A(per-chunk raw-KV + emergent gist-key soft attention,复用 `inattn_kv.py`,删掉 `TopKSelector`),但必须同时解冻 reader L16–31 并用多窗口训练——否则会精确重演 in-attn oracle 决定性负结果(冻结 reader 拿到完美 KV 也消费不了)。这是把 `32K_WALL_FINDINGS.md` 点名的"唯一未关闭路径(解冻 reader)"落到一个 raw-KV、值得学习消费的具体架构上。**
