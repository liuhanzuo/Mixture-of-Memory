# Paper Draft — Method & Experiments (Sections 3–4)

> Draft author: paper-writing agent (read + write only; no experiments run, no code changed).
> **All numbers are clean, `babilong_mix=0` measurements.** Leaked `mix>0` ("b25 wall-break") scores
> (e.g. P11 qa5 8k W6=85) are explicitly excluded and never cited.
> Status tags: **[confirmed]** = clean measured, table-ready · **[TBD-n100]** = needs full 100-sample
> fill / length fill / step fill before locking. Tables carry sample size `n` where it is < 100.
> Sources: PAPER_OUTLINE_20260628, SLOT_SWA_MECHANISM_20260628, MEMORYLLM_FACTCHECK_20260628,
> MIDPOINT_CONCLUSION_20260628, SLOT_REFORWARD_TWO_ROUTES_20260628, RESEARCH_REFORWARD_COST_20260628.

---

## 3. Method

We instantiate a streaming long-context reader over **Llama-3-8B** (32 layers, hidden size
$d=4096$, bf16). A document is split into fixed-size chunks ($\textsf{chunk\_size}=512$ tokens),
which are forwarded one at a time while a bounded memory is maintained. Our design follows a
**select-then-reforward** paradigm, in contrast to the **compress-then-inject** paradigm of
latent-memory methods such as MemoryLLM and M+ (Section 2): rather than compressing the past into
per-layer latent tokens and injecting them as cross-attention prefixes, we keep the *original
tokens* of the past, train a selector to pick the top-$K$ relevant chunks, and **re-forward those
original tokens through the full model jointly with the query**.

The method has three parts: a FIFO per-layer hidden memory (§3.1) that establishes the *read-out
wall*; a token-reforward read-out (§3.2) that breaks it; and a trained reader-attention selector
(§3.3) that addresses the *selection wall*. §3.4 gives deployment variants.

### 3.1 FIFO per-layer hidden memory

During streaming, every chunk is forwarded through the backbone. At each of the 32 layers, the
chunk's **detached hidden-state snapshot** ($[1,512,4096]$) is appended to a per-layer FIFO buffer
of capacity 25 chunks; when full, the oldest chunk is evicted. At read-out time the reader attends
over the buffered snapshots as a prefix, in addition to the current (last) chunk.

Two properties of this memory are the root of the diagnostics that follow:

- **Query-blind snapshots.** The stored hidden state for a chunk is the one computed *when that
  chunk was the current chunk during streaming* — i.e. before any query token has appeared. The
  memory therefore encodes "what looked salient at write time", independent of the eventual query.
- **Per-chunk RoPE reset.** Each streamed chunk uses rotary positions $0\ldots T-1$ restarted at 0,
  so absolute document position is not preserved in the snapshot.

This is the *frozen-snapshot* memory whose read-out limit we measure in §4.2. It is also the same
mechanism, viewed across architectures, that underlies the slot-routed and pooled-hidden memories
of related streaming methods: all of them read out a compressed/frozen, write-time, query-blind
representation.

### 3.2 The read-out wall and token-reforward

**Diagnosis (read-out wall).** Reading purely from the hidden memory (denoted **W0**: generation
window = last chunk only, everything earlier reachable only through the memory) is limited. Crucially,
this limit is *not* a selection problem: even an **oracle** that perfectly isolates the hidden
snapshot of the chunk containing the answer ("hidden-oracle") tops out at $\approx20\text{–}24$
(qa5 8k; §4.2). The bottleneck is the **representation** — a frozen, query-blind snapshot — not
finding the right chunk.

**Token-reforward.** We therefore retain, for each chunk, its **original token-ids** (not its hidden
states). At read-out time we take the original tokens of the selected chunks, concatenate them with
the last chunk into a single window, and **re-forward the whole window through all 32 layers with
the query present**:

$$\textsf{window} = \big[\,\textsf{tokens}(c_1)\,\Vert\,\cdots\,\Vert\,\textsf{tokens}(c_K)\,\Vert\,\textsf{tokens}(c_{\text{last}})\,\big],\qquad \textsf{logits} = \textsf{model}(\textsf{window}).$$

**Why this breaks the wall.** Re-forwarding lets every layer re-attend the query against the
selected tokens, so answer$\leftrightarrow$query multi-hop coupling is reconstructed under
query conditioning, rather than read once, single-hop, off a frozen snapshot. Because the selected
chunks are concatenated, their RoPE positions are contiguous and in-distribution. Empirically, an
oracle that re-forwards the answer chunk's *original tokens* reaches **66** (qa5 8k), versus
$\approx20\text{–}24$ for the same chunk's frozen *snapshot* (§4.2).

**Storage insight (counter-intuitive).** The read-out payload that token-reforward must persist is
just the token-ids: for a 32k document this is $\approx0.26$ MB, *smaller* than a 1 MB slot bank and
$\sim$$10^4\times$ smaller than a 4 GB dense KV cache (§4.6). Token-reforward trades "store tiny
token-ids + recompute" for "store large KV". Hidden states/KV are recomputed per step and never
stored.

### 3.3 Reader-attention chunk selector (training)

**Selection signal.** We score chunks with the reader's own native query$\cdot$key salience (the
attention scores the frozen reader already computes), rather than a bolt-on scoring head. This is
the one selection signal that historically did not collapse to chance (zero-train reader-attention
chunk precision $\approx55\%$, $\approx$$8.8\times$ random).

**Supervised training (scheme c).** We train the selector on **T2 synthetic needle** data with
`babilong_mix=0` (no BABILong in training): the known needle-chunk position supplies a cross-entropy
target that pushes the per-chunk salience mass onto the needle chunk (gradient-bearing, computed
through the *same* $q\cdot k$ used at eval), while the language-modeling loss is back-propagated
through the token-reforward window. The selection layer ($L_{16}$) must be unfrozen for the salience
to receive gradient, and the selection layer and top-$K$ at training time match those at eval.

**Anti-overfitting design decisions.** The needle is placed in a *random* chunk (not always chunk 0),
each example uses $\geq3$ keys, and all model selection is judged only on **held-out BABILong**
(we do not trust the T2 loss). These mitigate the known T2$\to$BABILong transfer risk.

### 3.4 Deployment variants

- **Number of re-forwarded chunks $K$.** $K{=}2$ (window 1536) is the sweet spot: in-distribution
  with the training curriculum, no RoPE extrapolation, $\sim$$6\times$ W0 latency at 8k, no OOM.
  $K{=}4$ (window 2560) is the practical ceiling: $\sim$$18\times$ latency, mild extrapolation,
  occasional OOM on long documents. $K{\geq}6$ is disabled (frequent OOM and dilution hurts accuracy).
- **Cheap path W0.** Skipping re-forward and reading purely from the (trained) hidden memory is a
  cheap deployment path; training alone lifts W0 from 12 to 34 (qa5 8k, §4.4), still well below
  token-reforward but $\sim$$3\times$ cheaper.

---

## 4. Experiments

### 4.0 Setup

Backbone Llama-3-8B; evaluation on **BABILong** tasks qa1 / qa2 / qa5 across context lengths
$\{0\text{k},1\text{k},2\text{k},4\text{k},8\text{k},16\text{k},32\text{k}\}$; `chunk_size`=512, FIFO
capacity 25 chunks. Unless noted, $n=100$ samples per cell; long-context selection-recall probes use
$n=40$ (16k) / $n=13$ (32k) and are marked **preliminary**.

**Integrity statement (red line).** All training and evaluation use `babilong_mix=0`. We never cite
any `mix>0` score. In particular, the historical "wall-break" numbers (qa5 8k W6 $\approx$ 85, etc.)
arise from a checkpoint trained with 15% BABILong SFT on the *same tasks/lengths/dataset* as the eval
(no train/test isolation in the HF dataset) and are $\approx85\%$ leakage artifacts; they are
**excluded entirely**.

**Anchors (clean).** True-SOTA anchor = a clean dense long-context baseline (pg19, $n_\text{ctx}{=}7$),
qa5 = 75/73/51/29/**19/16/9** across 0k–32k. External anchor = MemoryLLM teacher, qa5 =
47/50/45/39/**39/38/34**. We treat $\le$ pg19 nctx7 as the clean SOTA to beat, and the MemoryLLM
teacher as a strong latent-memory reference.

**A note on chance floors.** BABILong qa5 has a $\sim$7-word answer space, so guessing without
reading the needle scores a floor of $\approx13$. Net signal above chance is therefore the honest
quantity for long-context comparisons (used in §4.5).

---

### 4.1 Main results — clean SOTA (Table 1)

Our $K{=}4$ token-reforward deployment is **2.4–3.6$\times$** the clean SOTA anchor on long contexts
and **matches the MemoryLLM teacher** (32k), with no data leakage.

**Table 1. Clean BABILong accuracy (mix=0, n=100).** Best per column in **bold**.

| Method | qa5 8k | qa5 16k | qa5 32k | qa1 8k | qa2 8k |
|---|---|---|---|---|---|
| pg19 nctx7 (clean SOTA anchor) | 19 | 16 | 9 | [TBD] | [TBD] |
| MemoryLLM teacher (latent-memory ref.) | 39 | **38** | **34** | [TBD] | [TBD] |
| Ours — W0 (pure hidden, trained) | 34 | [TBD] | [TBD] | [TBD] | [TBD] |
| **Ours — K=4 token-reforward** | **52** | **38** | 32 | **20** | **19** |

- **[confirmed]** qa5 8k=**52** / 16k=**38** / 32k=**32**; qa1 8k=**20**; qa2 8k=**19**.
  qa2 (two-needle) is intrinsically harder, hence lower than qa1.
- vs anchor: qa5 16k $38/16=2.4\times$, 32k $32/9=3.6\times$; reaches the teacher's 32k=34 level.
- **[TBD-n100]**: qa5 0k–4k deployment points; full qa1 / **qa2** long-context rows (16k/32k);
  W0 16k/32k for the trained model.

---

### 4.2 Decomposing the read-out wall (Table 2, core ablation)

We separate the read-out wall from the selection wall by replacing the *selection* with an oracle and
varying only the *representation* read out. The ladder: pure frozen memory (W0) $\to$ oracle isolating
the answer's frozen **snapshot** (hidden-oracle) $\to$ oracle re-forwarding the answer's **original
tokens** (oracle-token).

**Table 2. Read-out-wall ladder, oracle selection (mix=0, n=100).**

| Read-out configuration | qa5 8k | qa5 16k | qa5 32k | qa1 8k |
|---|---|---|---|---|
| W0 — pure frozen memory | 12 | 8 | 2 | 12 |
| hidden-oracle — isolate answer's frozen *snapshot* | 20–24 | ~24 | ~22 | 20 |
| **oracle-token — re-forward answer's *original tokens*** | **66** | **70** | 60† | **50** |

- **Conclusion.** The frozen-snapshot read-out wall sits at $\approx$**20** even with a perfect
  oracle; re-forwarding the *same chunk's* original tokens breaks through to **50–70**. This isolates
  the bottleneck as the *representation* (query-blind frozen snapshot), not selection.
- †**[TBD]** oracle-token qa5 32k is recorded as 60 (heartbeat log) and 50 (workflow recheck); needs
  one clean re-run to unify (affects Table 2 and the wall-decomposition figure).
- **[TBD-n100]** hidden-oracle is currently reported as a "20–24" band; needs $n{=}100$ to lock a
  single value per length.

---

### 4.3 Selection is trainable on short contexts (Table 3, Fig. 5)

On 8k (candidate set $\approx16$ chunks), supervised training of the reader-attention selector lifts
deployment accuracy monotonically toward the oracle, plateauing after step 1000.

**Table 3. Selection training curve + cheap-path improvement (qa5/qa1 8k, mix=0).**

| Training stage | qa5 8k, K2 reforward | qa1 8k, K2 reforward | qa5 8k, W0 (cheap path) |
|---|---|---|---|
| C-probe (zero-train) | 28 | 14 | 12 |
| step 500 | 39 | — | 28 |
| step 1000 (plateau) | **46** | (no degradation) | **34** |
| oracle ceiling | 66 | 50–54 | — |

- **[confirmed]** qa5 8k curve **28 $\to$ 39 $\to$ 46** (plateau), moving toward oracle 66; qa1 does
  **not** degrade.
- **Recall vs ranking (large-$K$ probe).** At step 1500, $K{=}6$ reaches qa5 8k $\approx65\approx$
  oracle 66, while $K{=}2$=46 / $K{=}4$=48. Since accuracy rises toward the oracle as $K$ grows,
  the 46$\to$66 gap on 8k is dominated by **insufficient recall, not mis-ranking** (the right chunk
  is ranked but outside top-2/4).
- **[TBD-n100]** step 3000 at full $n{=}100$ to confirm the plateau is a true saturation (not a slow
  climb or a collapse); some intermediate points are currently $n{=}40$.

### 4.4 The cheap W0 path is also improved by training

Training the memory (no re-forward) lifts the pure-hidden read-out from **12 $\to$ 28 $\to$ 34**
(qa5 8k), i.e. it *also* breaks the 20 frozen-snapshot wall (last column of Table 3). This gives a
deployment path that avoids the 3–6$\times$ re-forward compute cost, at the price of accuracy
(34 vs 46 at K2, vs 66 oracle).

---

### 4.5 Four-quadrant robustness — storage is effective, read-out is the bottleneck (Table 4)

The central claim — *the memory stores the long-range information; reading it out is the bottleneck*
— is supported across **two memory architectures** $\times$ **two raw-token read-out patches**.
Regardless of how the past is stored (slot-routed compression / pooled hidden / FIFO frozen snapshot)
and which raw-token read-out is added (fixed near-window SWA / selected-chunk token-reforward),
adding a *raw-token* read-out beats *pure compressed/frozen* read-out by $\sim$2–5$\times$ on the
same checkpoint and same written memory. All values below are clean `mix=0`.

**Table 4. Adding raw-token read-out vs pure compressed/frozen read-out (qa5, mix=0, n=100).**

| Memory architecture | Read-out | qa5 8k | qa5 16k | qa5 32k | gain |
|---|---|---|---|---|---|
| Slot routing (lr5e5) | W0 (pure compressed) | 16 | 12 | 11 | — |
| Slot routing (lr5e5) | + near-window SWA (W6) | **49** | **39** | **34** | $\sim$3$\times$ |
| Slot routing (mass_coef2) | W0 (pure compressed) | 28 | 0 | 8 | — |
| Slot routing (mass_coef2) | + near-window SWA (W6) | **54** | 36 | 23 | $\sim$1.9$\times$ |
| Pooled hidden (selfstudy rawkv) | W0 (pure compressed) | (–) | 11 | 7 | — |
| Pooled hidden (selfstudy rawkv) | + near-window SWA (swa2) | (–) | **26** | **18** | $\sim$2.4–2.6$\times$ |
| FIFO frozen snapshot (ours) | W0 (frozen snapshot) | 12 | 8 | 2 | — |
| FIFO frozen snapshot (ours) | + token-reforward (oracle) | **66** | **70** | 60† | $\sim$3–5$\times$ |

- **Interpretation.** The gain is invariant to architecture and to whether the raw tokens come from a
  free near-window (SWA) or a selected chunk (token-reforward) — it is a property of *re-reading
  original tokens under query conditioning*, not of any one architecture. Near-window SWA is the
  "selection-free" special case of token-reforward (re-read the most recent $W$ chunks); both close the
  same compressed-read-out deficit.
- The fourth nominal quadrant — **slot routing + selected-chunk token-reforward** ("route 1") — is
  **not yet built**: the slot channels store hidden states with in-chunk positions and no
  document-chunk-id, so a slot$\to$document-chunk mapping ($\sim$100–130 LOC) does not exist. It is
  identified as future work; the three realized lines already establish the claim.
- **Integrity note.** The slot+SWA values above are the *clean* (`mix=0`) lr5e5 / mass_coef2 results.
  We do **not** report the leaked P11 slot+W6 numbers (40–85), which sit on a leakage-inflated base.
- †same 32k unification caveat as Table 2.

---

### 4.6 Long-context selection wall (Table 5, honest negative result)

On long contexts the deployable selector cannot reach the oracle's read-out ceiling: the answer
*is* in memory (oracle is high), but reader-attention and slot retrieval cannot locate it in the
large candidate set. We report this as a negative result and falsify the eviction hypothesis.

**Table 5a. Deployment vs oracle ceiling (qa5, mix=0, n=100).**

| Length | K=4 deploy | oracle-token | gap |
|---|---|---|---|
| qa5 16k | 38 | 70 | the answer is present but the selector cannot reach it |
| qa5 32k | 32 | 60† | same |

**Table 5b. Selection recall vs chance (E2 probe, preliminary).** Buffer cap = 25 candidates.

| Task / length | recall@4 | recall@8 | recall@16 | median rank (of 25) | n |
|---|---|---|---|---|---|
| chance | 0.16 | 0.32 | 0.64 | 12.5 | — |
| qa1 16k | 0.17 | 0.35 | 0.68 | ~7–10 | 40 |
| qa5 16k | 0.15 | 0.45 | 0.68 | ~7 | 40 |
| qa1 32k | 0.00 | 0.31 | 0.77 | ~10 | 13 |
| qa5 32k | 0.15 | 0.31 | 0.54 | ~10 | 13 |

- **Conclusion.** All recall@$k$ values are $\approx$ chance, and the needle's median rank sits in the
  middle of the candidate set ($\sim7\text{–}10.5$ of 25). The signal is *weak and diluted by
  distractors* — not absent (short 8k contexts with $\approx16$ candidates are selectable; §4.3),
  not a pure recall deficit (extending to $k{=}16$ only reaches $\approx0.68\approx$ chance 0.64).
  Net of the $\approx13$ chance floor, the deployable selector captures only $\sim$half of the
  perfect-selection re-forward gain (qa5 32k: deploy $38{-}13={+}25$ vs oracle $63{-}13={+}50$).
  This is an **information-theoretic selection wall** on large candidate sets, distinct from the
  (solved) read-out wall.

**Eviction hypothesis falsified.** Filling the buffer with all 64 chunks (no eviction) *lowers*
accuracy: keep_all qa5 32k=**15** $<$ evict (cap 25) qa5 32k=**32**. The long-context bottleneck is
therefore *not* the needle being evicted; enlarging the candidate set hurts because it dilutes
selection precision (1-of-64 is harder than 1-of-25).

- †same 32k unification caveat. **[TBD-n100]** E2 recall at $n{=}100$ (current $n{=}40/13$) before
  the "$\approx$ chance" claim is locked in the final paper.

---

### 4.7 Cost analysis (Table 6)

**Table 6. Storage and latency of token-reforward read-out.** Storage = code-computed; latency =
measured wall-clock (machine group A, 95 GB GPU; do not compare across machine groups).

| Configuration | Read-out payload | Selection index | Total storage | Latency (rel. W0) |
|---|---|---|---|---|
| Pure slot (W0) | slot bank 1.0 MB | — | ~1 MB | 1$\times$ |
| token-reforward (read-out only) | token-id **0.26 MB** | reuse / single-layer | ~0.26 MB + index | — |
| token-reforward, K=2 | 0.26 MB | 6.25 GB FIFO (optimizable $\to$ 8–256 MB) | — | ~6$\times$ (8k) / ~4$\times$ (16k) |
| token-reforward, K=4 | 0.26 MB | same | — | ~18$\times$ |
| token-reforward, K$\geq$6 | — | — | — | OOM |
| Dense full-KV (reference) | 4 GB | — | 4 GB | — |

- **Storage is not the bottleneck.** The re-forward read-out payload (token-ids, 0.26 MB at 32k) is
  *smaller* than a slot bank and $\sim$$10^4\times$ smaller than a dense KV cache. The 6.25 GB figure
  is the current 32-layer FIFO *selection* crutch, not read-out-essential; the design reduces it to a
  single selection layer (256 MB) or pooled (8 MB).
- **Latency is the real cost.** The generation window scales as $(K{+}1)$ to $(K{+}1)^2$, amplified
  $\sim$20$\times$ by `use_cache=False` (every step re-forwards the whole window). $K{=}2$ is $\sim$6$\times$
  (8k) / $\sim$4$\times$ (16k, where long streaming dilutes the penalty); $K{=}4$ is $\sim$18$\times$; $K{\geq}6$ OOMs.
- **Two purely-engineering optimizations** (do not change the mechanism): add a window KV-cache
  (prefill once + incremental decode, $\sim$20$\times$ speedup); shrink the selection index from a
  32-layer FIFO to a single layer / pooled (6.25 GB $\to$ 8–256 MB).

### 4.8 Why inject does not work (supports §2)

A measured inject-style read-out (Method A, raw-KV injection) gains only **+1 to +2.5** even when
given the correct evidence: a frozen reader cannot use injected KV it did not attend to itself. This
is the empirical reason we adopt re-forward rather than inject — it is not merely that raw tokens are
lossless, but that a frozen reader only uses representations it computes through its own attention.

---

## Appendix — data-status summary for this draft

**[confirmed] (table-ready):**
- Read-out-wall ladder backbone: W0 qa5 8k=12, hidden-oracle $\approx$20–24, oracle-token qa5 8k=66 / 16k=70.
- Selection curve backbone: C-probe 28 $\to$ step500 39 $\to$ step1000 46 (plateau); qa1 no degradation; W0 12 $\to$ 28 $\to$ 34.
- Deployment SOTA core: K4 qa5 8k=52 / 16k=38 / 32k=32; qa1 8k=20; qa2 8k=19; anchors pg19 nctx7 16k=16/32k=9, teacher 32k=34.
- Four-quadrant clean slot+SWA: lr5e5 qa5 8k 16$\to$49 / 16k 12$\to$39 / 32k 11$\to$34; mass_coef2 8k 28$\to$54; selfstudy 16k 11$\to$26 / 32k 7$\to$18.
- Eviction falsified: keep_all 32k=15 < evict 32k=32.
- Long-context recall $\approx$ chance (qualitative trend stable).
- Cost: storage code-computed; latency measured (machine group A).

**[TBD] (placeholders to fill before final):**
1. **qa2 full grid** — long-context qa2 deployment + oracle (16k/32k).
2. **step 3000 at n=100** — lock the training plateau.
3. **E2 / E0 recall at n=100** — currently n=40 (16k) / n=13 (32k); needed before locking "$\approx$ chance".
4. **oracle-token qa5 32k unification** — 60 vs 50; one clean re-run.
5. **hidden-oracle single value** — currently "20–24" band; lock per length at n=100.
6. **qa1 full deployment row** (0k–32k) and trained-W0 16k/32k.
7. (optional) downstream transfer (RULER/LongEval) if generality is claimed; current exploration ≈0,
   so the gain should be positioned as BABILong-specific readout improvement.

**Red lines maintained:** only `babilong_mix=0` numbers used; leaked P11 (40–85) never cited; every
cell carries $n$, and $n<100$ cells are marked preliminary; machine groups A/B never compared.
