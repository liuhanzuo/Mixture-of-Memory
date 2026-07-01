# Paper Draft — Abstract, Introduction & Related Work (Sections 1–2)

> Draft author: paper-writing agent (read + write only; no experiments run, no code changed).
> **Companion to** `PAPER_DRAFT_METHOD_EXP_20260628.md` (Method §3 + Experiments §4). Terminology,
> notation (W0 / oracle / token-reforward / read-out wall / selection wall / select-then-reforward
> vs. compress-then-inject) and all numbers are kept consistent with that draft.
> **All numbers are clean `babilong_mix=0` measurements.** Leaked `mix>0` ("b25 wall-break") scores
> are never cited.
> External-reference status: **[cite?]** marks a claim that needs a precise citation we have not yet
> verified. MemoryLLM (arXiv:2402.04624) and M+ (arXiv:2502.00592) facts are web-verified in
> `MEMORYLLM_FACTCHECK_20260628.md` and are **not** flagged.
> Sources: PAPER_OUTLINE_20260628, MEMORYLLM_FACTCHECK_20260628, SLOT_SWA_MECHANISM_20260628,
> MIDPOINT_CONCLUSION_20260628, PAPER_DRAFT_METHOD_EXP_20260628.

---

## Abstract

Equipping large language models with long-range memory is usually framed as a problem of *capacity*
or *forgetting*. We argue instead that, for a fixed and well-populated memory, the bottleneck
decomposes into two independent walls that can be measured separately. The **read-out wall**: a
frozen, query-blind compressed memory cannot be read out well — even an oracle that perfectly isolates
the memory of the answer-bearing chunk tops out at $\approx20$–$24$ on BABILong qa5. The **selection
wall**: on a large candidate set, an unsupervised salience score cannot locate the relevant chunk.
We break the read-out wall with **token-reforward**: rather than compressing the past into per-layer
latent tokens and injecting them as cross-attention prefixes (the *compress-then-inject* paradigm of
MemoryLLM and M+), we keep the *original tokens*, select the top-$K$ relevant chunks, and re-forward
them through the full model jointly with the query (*select-then-reforward*). Re-forwarding the
answer chunk's original tokens reaches $66$, versus $\approx20$ for its frozen snapshot. Selection is
trainable on short contexts (zero-train $28\to46$, toward the oracle's $66$), giving a clean,
leakage-free deployment that is **2.4–3.6$\times$** a dense long-context anchor and matches a
latent-memory teacher at 32k. We additionally report, as an honest negative result, that on long
contexts selection becomes an information-theoretic wall, and we falsify the eviction hypothesis.
Our two-wall decomposition is a reusable diagnostic for long-range memory.

*(≈195 words.)*

---

## 1. Introduction

**Long-range memory and the limited working set.** Transformer language models operate over a bounded
context window, yet many tasks — multi-hop question answering, long-document reasoning, lifelong
dialogue — require information that lies far outside any affordable window [cite?: long-context
LLM survey / motivation]. Two broad families address this. *Long-context* methods enlarge the window
itself (positional extrapolation, sparse / linear attention, KV-cache compression) but still pay a
growing memory or compute cost in the document length [cite?: long-context / efficient-attention
references]. *Memory* methods instead maintain a **bounded working set**: they stream the document,
compress the past into a fixed-size store, and read from that store at generation time. Latent-memory
methods such as **MemoryLLM** and **M+** are the leading instance of this family: they compress past
context into per-layer latent tokens and **inject** those tokens as cross-attention prefixes when the
query arrives — a paradigm we call **compress-then-inject**.

**The blind spot of compressed memory.** Compress-then-inject couples two design choices that, we
find, are the source of its long-range limits. First, the stored representation is written
**query-blind**: each chunk is encoded as it streams past, before any query token exists, so the
memory holds "what looked salient at write time" rather than "what this query needs." Second, the
read-out is by **injection**: the stored latent vectors are presented as keys/values that the frozen
reader did not itself attend through. We show empirically (§2, §4.8) that injection is the binding
constraint, not merely lossy compression: feeding a frozen reader the *correct* evidence as injected
KV (a Method-A raw-KV read-out) lifts accuracy by only **+1 to +2.5** — a frozen model barely uses
representations it did not compute through its own attention.

**A diagnostic framework: two walls.** Our central contribution is conceptual before it is
methodological. We decompose the long-range-memory bottleneck into two independently measurable walls,
on a streaming reader built over Llama-3-8B [cite?: Llama-3] with a FIFO per-layer hidden memory and
evaluated on BABILong [cite?: BABILong] qa1/qa2/qa5 across 0k–32k.

- **The read-out wall.** Reading *purely* from the compressed/frozen hidden memory (denoted **W0**:
  the generation window is the last chunk only, everything earlier reachable only through the memory)
  is limited — and crucially, this is *not* a selection problem. An **oracle** that perfectly isolates
  the frozen snapshot of the chunk that contains the answer ("hidden-oracle") still tops out at
  $\approx20$–$24$ (qa5 8k). The wall is in the *representation*: a query-blind frozen snapshot, read
  once and single-hop.

- **The selection wall.** Even with a perfect read-out, the deployed system must *find* the relevant
  chunk in the streamed candidate set. On short contexts the candidate set is small ($\approx16$
  chunks at 8k) and this is solvable; on long contexts (a buffer of 25 candidates at 16k/32k) it is
  not, as we quantify below.

**Key findings.** (i) **Token-reforward breaks the read-out wall.** Instead of compressing the past,
we retain each chunk's *original token-ids*, select the top-$K$ relevant chunks, and **re-forward
those original tokens through all 32 layers jointly with the query**. Re-forwarding lets every layer
re-attend the query against the selected tokens, reconstructing answer$\leftrightarrow$query multi-hop
coupling under query conditioning. An oracle that re-forwards the answer chunk's original tokens
reaches **66** (qa5 8k), versus $\approx20$–$24$ for the same chunk's frozen snapshot — a $\sim3$$\times$
jump from *the same stored information*, read differently. (ii) **Selection is trainable on short
contexts.** Using the reader's own native query$\cdot$key salience as the selection signal (the one
signal that did not collapse to chance), supervised training lifts deployment accuracy from a
zero-train **28** to **46** at step 1000 (qa5 8k), moving monotonically toward the oracle's 66. (iii)
**Clean long-context SOTA.** Our $K{=}4$ token-reforward deployment reaches qa5 16k=**38** / 32k=**32**,
which is **2.4–3.6$\times$** a clean dense long-context anchor (pg19 $n_\text{ctx}{=}7$: 16k=16,
32k=9) and matches a MemoryLLM teacher reference (32k=34) — with **no data leakage** (every number is
`babilong_mix=0`; the historical "wall-break" scores are $\approx85\%$ leakage artifacts and are
excluded entirely). (iv) **An honest negative result.** On long contexts the deployable selector
*cannot* reach the oracle's read-out ceiling: the answer is present in memory (oracle is high) but
reader-attention and slot retrieval both score $\approx$ chance on the large candidate set
(recall@8 $\approx0.31$–$0.45$ vs. chance $0.32$; median needle rank $\sim7$–$10$ of 25). We further
**falsify the eviction hypothesis** — keeping all 64 chunks *lowers* qa5 32k accuracy (15) below the
evicting buffer (32) — showing the long-context bottleneck is selection precision on a larger
candidate set, not the needle being forgotten.

**An architecture-invariant claim.** The diagnosis is not specific to our FIFO memory. Across a
$2{\times}2$ grid of $\{$slot-routed compression, FIFO frozen snapshot$\}$ memories $\times$
$\{$fixed near-window read-out, selected-chunk token-reforward$\}$ raw-token read-outs, adding a
*raw-token* read-out beats *pure compressed/frozen* read-out by $\sim2$–$5$$\times$ on the same
checkpoint and the same written memory (§4.5). The information is stored; reading it out is the
bottleneck — and re-reading original tokens under query conditioning is what closes the gap,
regardless of architecture.

**Contributions.**
1. **A two-wall decomposition** of the long-range-memory bottleneck into an independently measurable
   *read-out wall* and *selection wall*, with oracle-based probes that attribute each lost point to
   the right wall (§3.2, §4.2).
2. **Token-reforward**, a select-then-reforward read-out that breaks the read-out wall (hidden-oracle
   $\approx20\to$ token-reforward $66$); a cheap pure-hidden W0 path that training also lifts past the
   wall ($12\to28\to34$); and the empirical finding that *injection*, not lossy compression, is why
   compress-then-inject under-reads (§4.8).
3. **Clean, leakage-free long-context results**: qa5 16k=38 / 32k=32, 2.4–3.6$\times$ a dense anchor
   and matching a latent-memory teacher, all under a strict `babilong_mix=0` integrity protocol.
4. **An honest characterization of the selection wall**: short-context selection is trainable
   ($28\to46$), long-context selection is an information-theoretic wall (recall $\approx$ chance), and
   the eviction hypothesis is falsified.

**Scope and honesty.** Our gains are demonstrated on BABILong with a Llama-3-8B backbone; we did not
observe transfer of the read-out improvement to other long-context formats (e.g. RULER / LongEval
[cite?]) in preliminary exploration, so we position the contribution as a BABILong-specific read-out
improvement plus an architecture-invariant *diagnosis*, rather than a universal long-context method.
The selector is trained on synthetic needle data and its long-context generalization is limited by
design. We state these limits up front because the negative result (the selection wall) is itself a
contribution.

---

## 2. Related Work

### 2.1 Latent / token-compression memory (closest prior work)

The methods most directly comparable to ours maintain a fixed-size, per-layer **latent memory** and
read from it by injection. We base the following on the verified original texts of MemoryLLM and M+
(`MEMORYLLM_FACTCHECK_20260628.md`).

**MemoryLLM** [arXiv:2402.04624] augments a backbone (Llama2-7B) with a fixed-size memory pool
composed of *memory tokens at every layer*. Each per-layer pool $\theta_l$ has shape $N\times d$ with
$N{=}7680$ tokens and $d{=}4096$ equal to the **full model hidden size** — i.e. each memory token is
a full-dimensional hidden-state vector. Crucially, MemoryLLM's "compression" is a reduction in the
*number* of tokens (a chunk's hidden states are summarized into $K{=}256$ new memory tokens), **not a
reduction of the feature dimension**; there is no down-projection of the stored vectors. Self-update
appends the new $K$ tokens and **randomly drops** $K$ existing tokens, yielding an exponential
(Ebbinghaus-style) forgetting schedule [cite?: MemoryBank / Ebbinghaus, as cited by MemoryLLM]. At
generation time, *all* memory tokens are presented as a cross-attention prefix that the query's hidden
states attend over — read-out is by **injection**, never by re-forwarding original text. Training uses
only a language-modeling cross-entropy loss, with **no explicit reconstruction loss**.

**M+** [arXiv:2502.00592] extends MemoryLLM (now on Llama-3.1-8B) with a CPU-resident **long-term
memory**: tokens that MemoryLLM would discard are instead retained with an "age" attribute, and a
**co-trained retriever** selects which long-term latent tokens to bring back. The retriever's
key/query projections map to a low dimension $d_\text{proj}{=}d/20$ — but this down-projection lives
**only in the retrieval-scoring space**; the stored memory tokens themselves remain full-$d$ latent
hidden states. M+ adds a contrastive retriever loss and a multi-LoRA (separate update / generation
adapters) on top of the LM loss.

**A common misattribution we explicitly avoid.** It is tempting to summarize these methods as "storing
hidden states at reduced dimension." That is inaccurate: *vanilla MemoryLLM does not down-project*
(its memory tokens are full $d{=}4096$); the only $d\to d/20$ projection is in **M+'s retriever
scoring space**, not in either method's stored representation. We state this precisely to keep the
comparison defensible.

**How we differ.** Both methods follow **compress-then-inject**: past context is compressed (lossily,
in token count) into per-layer latent vectors written *query-independently* at stream time, and read
by *injecting* those vectors as cross-attention prefixes. We follow **select-then-reforward**: we do
not compress the content at all — we keep the *original tokens*, treat "compression" as the selector's
top-$K$ choice, and re-forward the selected tokens through the *full model with the query present*,
yielding a **query-conditioned, lossless** representation. We share the family's motivation (a bounded
working set, an end-to-end-trained selection/compression module, extending effective context under a
fixed GPU budget); the difference is in the **memory representation** (raw token vs. latent) and the
**read-out mechanism** (re-forward vs. inject).

**Table A. Compress-then-inject vs. select-then-reforward** (MemoryLLM / M+ facts web-verified).

| Axis | MemoryLLM (2402.04624) | M+ (2502.00592) | Ours (Mixture-of-Memory) |
|---|---|---|---|
| What is stored | per-layer latent memory tokens (compressed hidden states) | + CPU long-term latent pool (with age) | **original tokens (raw text); no latent compression** |
| Form of compression | token-*count* compression (chunk $\to K$ latents); **no dim. reduction** | same; retriever key/query projected to $d/20$ (scoring only) | "compression" = selector's top-$K$ choice |
| Read-out | **inject**: latents as KV / prefix, query cross-attends | same (short-term + retrieved long-term latents) | **re-forward**: selected original tokens re-run through full model with query |
| Selection | none (all memory always attended); random-drop forgetting | **co-trained retriever** ($d/20$) picks latents | **trained selector** picks top-$K$ chunks |
| Query-awareness | write-time query-independent | retrieval query-aware; token content still query-independent | **fully query-conditioned** (query present at re-forward) |
| Fidelity | lossy (compressed) | lossy (compressed) | **lossless (original tokens)** |
| Training signal | LM cross-entropy (no reconstruction loss) | LM CE + retriever contrastive loss + multi-LoRA | selector CE on selection salience + LM loss through the re-forward window |

**Empirical reason we do not inject (preview of §4.8).** Our choice is not merely "raw tokens are
lossless." We measured an inject-style read-out (Method A, raw-KV injection) and found it gains only
**+1 to +2.5** even when handed the correct evidence: a frozen reader cannot exploit injected KV it
did not attend through itself. This is the concrete, measured reason we re-forward rather than inject,
and it is the mechanism behind the read-out wall.

### 2.2 Long-context windows and KV-cache compression

A parallel line enlarges or compresses the *attention window* itself rather than maintaining a
separate memory. KV-cache eviction / compression methods keep only a salient subset of past
keys/values — e.g. SnapKV, H2O, StreamingLLM, FastGen [cite?: SnapKV; H2O; StreamingLLM; FastGen] —
often using the model's *own attention scores* to decide what to retain. Our selector shares this
"reader-native salience" intuition (we score chunks with the reader's native $q\cdot k$ rather than a
bolt-on head), but our read-out is fundamentally different: these methods *keep compressed KV and
attend over it*, whereas we *discard KV, keep original token-ids, and re-forward*. Positional
extrapolation and efficient-attention methods that extend the window directly [cite?: RoPE (Su et
al.); positional interpolation; long-context attention references] are complementary — they pay a
cost growing in document length, while our working set stays bounded. Our clean dense long-context
anchor (a pg19-trained $n_\text{ctx}{=}7$ baseline [cite?: pg19 / Compressive Transformer dataset])
is the strongest member of this family that we beat by 2.4–3.6$\times$ on long BABILong.

### 2.3 Recurrent and segment-level memory

Recurrent-memory transformers carry a fixed set of memory states across segments — e.g. Recurrent
Memory Transformer (RMT) [cite?: Bulatov et al.], Transformer-XL [cite?], Compressive Transformer
[cite?]. These also write a *query-blind* segment summary and read it back as state, so they sit on
the same side of the read-out wall as latent memory: the stored representation is fixed before the
query is seen. Our diagnosis (the $2{\times}2$ raw-token-vs-compressed grid in §4.5) is meant to apply
to this whole class — any memory that reads out a frozen, write-time, query-blind representation faces
the read-out wall, and re-reading original tokens under query conditioning is what closes it.

### 2.4 Retrieval-augmented generation

Retrieval-augmented methods fetch external passages and prepend their *raw text* to the query —
e.g. RAG, REALM, RETRO, kNN-LM [cite?: RAG (Lewis et al.); REALM; RETRO; kNN-LM]. This shares our
"re-read original tokens" property, but in a different setting: retrieval operates over an *external
corpus* with a separately trained dense retriever, whereas we operate *within a single streamed
document*, selecting chunks the model itself just processed, with a selector that is the reader's own
attention. The read-out wall we identify explains *why* re-reading text (as in RAG) outperforms
injecting compressed states (as in latent memory): the frozen reader only uses representations it
recomputes through its own attention. Our selection wall is the in-document analogue of retrieval's
recall/ranking problem, and our negative result quantifies where it becomes information-theoretically
hard.

### 2.5 Positioning

We sit at the intersection of these lines: a **bounded streaming memory** (as in latent / recurrent
memory) whose read-out is **re-reading original tokens** (as in retrieval / long-context KV) under
**query conditioning**, with selection driven by the **reader's own attention** (as in KV
eviction). Our contribution is to (a) show this combination — *select-then-reforward* — breaks the
read-out wall that all compress-then-inject memories share, and (b) decompose the residual error into
a trainable short-context selection problem and an information-theoretic long-context selection wall.

---

## Appendix — external-reference status for this draft

**Web-verified (do NOT flag), from `MEMORYLLM_FACTCHECK_20260628.md`:**
- MemoryLLM (arXiv:2402.04624): per-layer $N{\times}d$ pool, $N{=}7680$, $d{=}4096$ full hidden size,
  Llama2-7B + 1B memory; token-count compression (chunk$\to K{=}256$), **no feature down-projection**;
  random-drop / exponential forgetting; read-out by **injection** (cross-attention prefix); **LM CE
  only, no reconstruction loss**.
- M+ (arXiv:2502.00592): CPU long-term memory with age; **co-trained retriever**, key/query projected
  to $d/20$ **for scoring only** (stored tokens stay full-$d$); multi-LoRA; Llama-3.1-8B.

**[cite?] — needed before submission (not yet verified):**
1. **BABILong** benchmark (used throughout) — precise citation.
2. **Llama-3 / Llama-3-8B** backbone — precise citation.
3. **pg19** dataset / Compressive Transformer (clean dense anchor's training data).
4. **KV-cache compression**: SnapKV, H2O, StreamingLLM, FastGen — confirm names + venues + the
   "reader-native attention salience" attribution per method.
5. **Positional / long-context**: RoPE (Su et al.), positional interpolation, and a long-context
   survey for the Introduction motivation paragraph.
6. **Recurrent / segment memory**: Recurrent Memory Transformer (Bulatov et al.), Transformer-XL,
   Compressive Transformer.
7. **Retrieval-augmented**: RAG (Lewis et al.), REALM, RETRO, kNN-LM.
8. **MemoryBank / Ebbinghaus** forgetting analogy (as cited by MemoryLLM) — only if we keep that aside.
9. **RULER / LongEval** (cited in the scope/limitations paragraph as formats where transfer was not
   observed).

**Red lines maintained:** only `babilong_mix=0` numbers used; leaked "wall-break" scores never cited;
the MemoryLLM "down-projection" misattribution explicitly avoided (it belongs to M+'s retriever
scoring space, not to either method's stored representation).
