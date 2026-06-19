# Landmark Attention vs. mem_space — authoritative migration diff map

**Status.** Phase-1 **S0 anchor reproduced and committed (62c2d68)**: the official
Landmark weight-diff recovered on **LLaMA-1-7B** reproduces the passkey wall-break —
base LLaMA dies between 2.2k→4k tok (98%→0%), landmark-mem holds **94–100% out to
~31k tok**. This is our trusted "known-working" anchor.

**Purpose.** Authoritative reference for **Phase-3 diff-based migration**: starting
from the working Landmark anchor, change **exactly one axis at a time** toward our
`mem_space`. The migration step where passkey accuracy develops a *cliff that the
anchor did not have* pinpoints which difference kills long-range recall.

**Consistency.** Aligns with `status/LANDMARK_REPRODUCE_PLAN.md` (Phases 1–3,
gating protocol) and supersedes the loose 7-axis draft in TRAINER_ACTIVITY 09:15.

**Sources.**
- Paper: Mohtashami & Jaggi, *Landmark Attention*, arXiv:2305.16300 (NeurIPS 2023),
  §4.1 (from-scratch PPL), §4.2 + Fig.3 + App.G (LLaMA passkey).
- Repo `epfml/landmark-attention`: `llama/train.py`, `llama/run_test.py`,
  `llama/llama_mem.py`, `llama/requirements.txt`, `weight_diff.py`.
- Ours: `src/memory/mem_space/{memory_bank.py,selector.py,inattn_kv.py,layer.py,
  l2_compressor.py,l3_summary.py}`, `scripts/train_mem_space_dolmino_cpt.py`,
  `scripts/launch_sft_unfreeze_inattn.sh`.

---

## Locked gating protocol

- **Primary gate = native `run_test.py` passkey**, evaluated at **0 / 1k / 2k / 4k /
  8k / 16k / 32k tok**. Native settings: `top_k=5`, `num_tests=50`/length,
  passkey `randint(1,50000)` hidden at a random position in garbage filler, fixed
  question `"What is the pass key? The pass key is"`, `max_new_tokens=10`, regex
  digit-extract for correctness.
- **Control gate = our BABILong qa1** (single-fact NIAH, same single-needle
  semantics as passkey), lengths 0k–32k, n=100, `babilong.metrics`. Ensures the
  migrated mechanism remains measurable in our own harness with continuous口径.
- **PPL excluded** — the Landmark fine-tune line (§4.2) reports no PPL; PG19/arXiv
  PPL (§4.1) is the off-scope from-scratch GPT-2 line. qa2/qa5 are advanced
  observation only (Landmark has no multi-fact baseline) and do NOT drive diff
  verdicts.
- **"Killer" definition.** A diff is the long-range killer iff swapping it in
  introduces a **passkey cliff at a length where the immediately-prior step had
  none**. No-cliff ⇒ that axis is not (alone) responsible.

---

## The 7 migration axes

| # | Axis | Landmark (working anchor) value | mem_space (broken) value | Step | Expected passkey impact **if this axis is the killer** | Eng. cost / risk |
|---|------|--------------------------------|--------------------------|------|--------------------------------------------------------|------------------|
| 1 | **Unfreeze scope** | Full fine-tune: every param (embed + attn + FFN + lm_head + new `<landmark>` embedding) trainable; `--fsdp full_shard auto_wrap` over all `LlamaDecoderLayer` (`train.py`). No LoRA, no freezing. | v1 `--unfreeze_backbone` full 8B + adapter, lr 2e-5, 1k steps (`launch_sft_unfreeze_inattn.sh`); v2 (HOLD) partial L16–31. | held in anchor (revisit last) | If partial-unfreeze were the killer: cliff appears only when we *restrict* trainable params; full-unfreeze anchor shows no cliff. | **Low** (flag/layer-filter). Risk: 1k-step full FT on narrow data *damaged* base NIAH (OFF 22→11/12) — too few steps hurts before it helps. |
| 2 | **Data volume + source** | RedPajama-1T-Sample, **~0.98B tok** (15k steps × eff-batch 128 × 512), **1 epoch**, **7 sources** (CC, C4, GitHub, Books, ArXiv, Wiki, StackExch). Full-sequence LM loss (`DataCollatorForLanguageModeling`). | dolmino per_doc **single source**, ~**32.8M fwd / 8.2M loss tok** (`last_chunk_loss_only`), 1k steps. **~30× fewer fwd / ~120× fewer loss tok.** | **S2** | If data is the killer: anchor mechanism + Llama-3 holds passkey, but swapping to dolmino/short/single-source produces a cliff at mid–long lengths (≥4k). | **Medium**: point loader at dolmino + drop `last_chunk_loss_only`; both already exist. |
| 3 | **ctx / block structure** | Train ctx **512**, block **50**, one `<landmark>` token inserted every 50 tok (`add_mem_tokens`, repo `mem_freq=63` default / 50 in paper). | chunk **512**, **n_ctx=3** (eff ctx 2048), 128 slots, curriculum 0:3 (`train_mem_space_dolmino_cpt.py`). | **S3** | If chunking is the killer: cliff emerges once landmark-every-50 is replaced by our coarse chunk512/n_ctx3 grouping (fewer, larger addressable units). | **Medium**: changes data prep + grouping; mechanically contained. |
| 4 | **Retrieval mechanism** | Train: all blocks softly visible; grouped-softmax *trains* block gating via landmark scores. Infer (`run_test.py`, `cache_top_k=5`): retrieve **actual raw K/V** of top-5 landmark-scored blocks — selection signal == trained attention. | `TopKSelector` (`selector.py`): scores N slots vs a **pooled** hidden summary (`Q_sel`/`K_sel` + learnable per-slot keys), STE hard top-k, MoE load-balance aux. **Measured 0% needle precision.** | **S4** | If retrieval is the killer: cliff appears exactly when grouped-softmax soft selection → pooled routing-q selector; needle-bearing block stops being retrieved ⇒ accuracy floors at long lengths. | **Medium**: retrieval is a self-contained module; swap scoring source. |
| 5 | **Injection / readout layering** | Grouped-softmax in **every attention layer** (intra-block softmax-denominator isolation + landmark gating: cross-block weight = token-score × landmark-score); retrieved KV carry real source RoPE; reader **trained on this exact path**. | `inattn_kv.py`: retrieved raw K/V projected via native `k_proj`/`v_proj`, concatenated onto native K/V in **one softmax** at a **single layer (16)** with real source RoPE (prefix variant failed earlier). Eval-time wrapper, not trained-in. | **S5** | If single-layer readout is the killer: cliff appears when all-layer grouped-softmax → single-layer concat; one shot to consume retrieved KV is insufficient at long range. | **High**: single→all-layer is a mechanism rewrite across the stack (`layer.py` + wrapper). |
| 6 | **Memory unit** | No memory bank. `<landmark>` token per block, representation computed **in-context**, **ephemeral** (recomputed each forward), "trainable" only via shared token embedding + LM loss. Memory = **raw past-block KV cache**, addressable by landmark scores. | `MemoryBank` (`memory_bank.py`): **128 fixed learned slots** `[B,N,slot_dim]`, written by adapter that **lossily compresses** chunk hidden → slots (dual-gate, L2 compressor, L3 summary). Memory = learned compression, **not raw KV**. | **S6** | If the slot unit is the killer: cliff appears when raw-KV blocks → 128 lossy slots; a specific needle is no longer byte-recoverable from compressed slots. | **High**: this is mem_space's architectural identity — the final, largest jump. |
| 7 | **Base model** | **LLaMA-1-7B** (official tuned weight-diff `epfml/landmark-attention-llama7b-wdiff`, recovered via `weight_diff.py`). tf4.28 venv. | `models/Meta-Llama-3-8B`, main `.venv` tf5.5.4. Different tokenizer/vocab, GQA, RoPE θ. | **S1** | If base is the killer (unlikely): cliff appears purely from swapping LLaMA-1→Llama-3 with mechanism otherwise identical. | **High (hidden) — see hazard below.** |

---

## ⚠️ S1 base-swap is a TWO-diff-at-once hazard

The naïve reading is "S1 = swap base, low cost". It is **not**:

- The reproduced anchor runs in a **transformers==4.28.1** venv, because
  `llama_mem.py` monkeypatches the **tf-4.28 `LlamaAttention` internal API**.
- **Llama-3-8B cannot load under tf4.28** — it needs **tf≥4.40** (Llama-3 tokenizer
  + GQA support). Our main `.venv` is **tf5.5.4**.
- Therefore S1 (base→Llama-3) **forces simultaneously**: (a) base model change AND
  (b) a **full reimplementation of the grouped-softmax landmark mechanism** in
  tf5.5.4 (the 4.28 monkeypatch will not run). That is **two diffs in one step** —
  it violates the one-diff-at-a-time invariant and would confound any cliff.

**Mitigation (mandated):**
1. **Do S1 LAST, or SPLIT it.** Preferred split:
   - **S1a (mechanism port, base held LLaMA-1):** re-implement grouped-softmax +
     landmark top-k raw-KV retrieval in tf5.5.4 against **LLaMA-1-7B**, and
     **re-verify it reproduces the S0 passkey anchor** (94–100% to ~31k). This
     isolates "did the port preserve the mechanism" with base unchanged.
   - **S1b (base swap, mechanism held = ported tf5.5.4):** only after S1a passes,
     swap LLaMA-1-7B → Llama-3-8B under the *same* ported mechanism. Now base is the
     sole variable.
2. Until the tf5.5.4 port (S1a) is verified, **S2–S6 should be explored on the
   tf4.28 LLaMA-1 anchor** wherever the diff does not itself require Llama-3 — so
   that data/ctx/retrieval/readout/slot diffs are measured against a *stable*
   mechanism+base, not entangled with the port.

This is the single biggest sequencing trap in the whole migration; flagged here so
no coder treats S1 as a quick swap.

---

## Independent vs. causally-dependent axes

- **Independent (parallelizable across Group-A / Group-B):**
  - **S2 (data)** and **S4 (retrieval)** are mechanistically orthogonal — data
    swap touches the loader, retrieval swap touches the selector. They can run
    concurrently on the two disk-groups against the same anchor, yielding two
    attribution points per round.
  - **S3 (ctx/block)** is largely independent of S2/S4 and can fill a third slot,
    though it interacts mildly with S4 (block granularity ↔ what gets retrieved).
- **Causally dependent (must serialize):**
  - **S5 (single- vs all-layer readout)** must be evaluated with the retrieval
    decision settled, because a broken retriever (S4) would mask any readout
    signal. → **S4 before S5.**
  - **S6 (raw-KV blocks → compressed slots)** depends on the readout layering
    conclusion (S5): swapping the memory unit only makes sense once we know whether
    the readout path can consume *anything*. → **S5 before S6.**
  - **S1b (Llama-3 base)** depends on S1a (tf5.5.4 mechanism port) — see hazard.
  - **S1 axis as a whole** is held inside the anchor and resolved last to avoid the
    two-diff hazard contaminating earlier attributions.

---

## Prioritized migration order (recommended)

**Most-suspect-first, subject to the dependency constraints above.**

| Order | Step | Change from → to | Gate | If cliff appears here ⇒ |
|-------|------|------------------|------|--------------------------|
| 0 | **S0** | reproduce anchor (LLaMA-1-7B, wdiff) | passkey to ~31k = 94–100% | infra/eval口径 broken — fix first (already GO, 62c2d68) |
| 1 | **S2** ∥ **S4** | data RedPajama→dolmino · retrieval grouped-softmax→TopKSelector | passkey + qa1 | S2: single-source/short data is killer · S4: selector 0% precision is killer |
| 2 | **S3** | ctx landmark-every-50 → chunk512/n_ctx3 | passkey + qa1 | our coarse chunking hurts addressing |
| 3 | **S5** | readout all-layer grouped-softmax → single-layer in-attn concat (requires S4 settled) | passkey + qa1 | single-layer/concat readout insufficient |
| 4 | **S6** | memory unit raw-KV blocks → 128 lossy slots+adapter (requires S5 settled) | passkey + qa1 | lossy slot compression loses the needle |
| 5 | **S1a→S1b** | port mechanism to tf5.5.4 (base=LLaMA-1, re-verify) → swap base to Llama-3-8B | passkey + qa1 | S1a-fail: port lost the mechanism · S1b-cliff: Llama-3 base incompatible |

**Rationale & reconciliation with damage-investigator.**
- My prior mechanism-first ranking: **S4 retrieval > S5 single-layer readout > S2
  data**. damage-investigator's empirical ranking: **data #1, single-layer-injection
  #2**.
- **Reconciliation:** the two rankings agree that **single-layer readout (S5)** is a
  top-2 suspect, and disagree only on data vs retrieval at #1. Rather than pick, the
  recommended order **runs S2 (data) and S4 (retrieval) in the FIRST parallel round
  on the two disk-groups** — they are independent, so we get *both* attribution
  points simultaneously and let the data decide which (if either) produces the
  cliff. This honours damage-investigator's "data is most likely" (S2 in round 1)
  AND my "retrieval is most likely" (S4 in round 1) without serial guesswork. S5
  then follows once S4 is settled (dependency), and S6 last among the mechanism
  diffs. S1 is deferred to the end because of the two-diff tf-version hazard.

**One-line order:** **S0 (done) → {S2 ∥ S4} → S3 → S5 → S6 → S1a→S1b.**
