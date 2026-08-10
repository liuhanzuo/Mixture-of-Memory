# SWA + Memory Space: Design Notes & Curriculum Scaling Experiment

**Date**: 2026-04-27
**Status**: Active design doc — curriculum section added 2026-04-27
**Related notes**:
- `20260426_memory_space_design_direction.md` — architecture v0, prior art, training recipe
- `20260427_memory_space_mechanism_explainer.md` — Branch-3 A.2 winner config (PPL=1.9051)
- `20260427_niah_v8_diagnosis.md` — root cause: LM-only training cannot teach associative retrieval

---

## Context Summary (one paragraph)

The current **Memory-Space v0** architecture (Branch-3 A.2 winner) uses a **shared 512-slot bank across 32 layers**, top-k=64 selection via a learned `TopKSelector`, KV-prepend joint self-attention, and EMA writeback with Flamingo gate warmup. It achieves **PPL=1.9051** on PG19 (chunk_size=4096, Llama-3-8B frozen backbone). However, the NIAH v8 eval scored **0/60** because the LM training objective (next-token CE on PG19) gives zero gradient signal for associative retrieval — the selector learns to fetch "thematically recent" slots, not "content-addressed needles stored N chunks ago". The curriculum below is the path from the current PPL-passing baseline to a system that can provably retrieve facts over long gaps.

---

## § Curriculum Scaling Design

### Overview

Two training stages with increasing SWA window and chunk size, evaluated against a NIAH "retrieval distance scaling curve". The goal is to answer: **how far back (in chunks) can L1 memory retrieve a needle, and does that distance scale with W?**

```
Stage 1: W=512,  chunk=4096  →  train  →  eval NIAH @ N∈{1,2,4,8,16,32}
Stage 2: W=1024, chunk=8192  →  train  →  eval NIAH @ N∈{1,2,4,8,16,32}

Plot: retrieval accuracy (%) vs N (chunks between needle and query)
      Line 1: W=512  / chunk=4096
      Line 2: W=1024 / chunk=8192
```

The ratio **W / chunk_size = 12.5%** is held constant across stages (see §3 below). This keeps the memory-pressure per chunk identical so the two lines are directly comparable.

---

### Stage 1 — W=512, chunk_size=4096

**Training config (extends Branch-3 A.2 ship config)**:

```bash
# backbone + memory params (unchanged from A.2 winner)
--num_slots 512 --top_k 64 --selector_dim 128
--slot_init random --slot_init_noise 0.02
--writeback_warmup_steps 500 --writeback_gate_max 0.3
--shared_memory_bank --unfreeze_hidden_to_slot
--load_balance_weight 0.01

# SWA window (NEW for Stage 1)
--swa_window 512
--chunk_size 4096

# data
--dataset pg19 --split train
--tokens_per_update 4096   # one chunk per forward pass

# Stage 1: frozen backbone + memory adapters only
--freeze_backbone True
--lr 3e-4 --warmup_steps 500 --max_steps 30000    # ~1.2B tokens @ batch=40

# NIAH supervision mixture (critical — without this, retrieval stays 0%)
--niah_mix_fraction 0.10   # 10% of training batches are synthetic NIAH sequences
--niah_max_N 16            # max chunks between needle and query during training
```

**NIAH synthetic data format** (one batch item looks like):

```
[chunk_0: pg19 text]
...
[chunk_k: contains needle "The secret code for agent <name> is <5-digit-code>."]
...
[chunk_{k+N}: question "The secret code for agent <name> is "]  ← model must predict code
```

The NIAH sequences are generated on-the-fly during training (no disk storage needed). The needle is inserted at a random depth within chunk_k; subsequent chunks are pg19 text. The question appears N chunks later. Loss is computed only on the answer tokens (CE on the code digits), not on the haystack.

**Stage 1 acceptance criteria**:
- PG19 PPL ≤ 1.95 (≤ +0.05 regression from baseline 1.9051, allowing for SWA effect)
- NIAH accuracy @ N=1: ≥ 90%
- NIAH accuracy @ N=4: ≥ 60%  ← main signal that associative retrieval works
- NIAH accuracy @ N=16: measure and plot (don't gate on it)
- Slot utilization entropy ≥ 0.8 × log(512) (no slot collapse)

---

### Stage 2 — W=1024, chunk_size=8192

**Config** (delta from Stage 1):

```bash
--swa_window 1024
--chunk_size 8192

# Stage 1→2 transfer decision: FINETUNE (see §2 below)
--init_from  <stage1_checkpoint>
--lr 3e-5           # 10× lower than Stage 1
--writeback_warmup_steps 200   # re-warm writeback gate for new chunk size
--max_steps 15000   # ~2.4B tokens — same wall-clock as Stage 1

--niah_max_N 16     # same N budget; see §3 note on absolute vs chunk gap
```

**Stage 2 acceptance criteria** (same structure as Stage 1):
- PG19 PPL ≤ 1.95
- NIAH accuracy @ N=1: ≥ 90%
- NIAH accuracy @ N=4: ≥ 70%  ← should be >= Stage 1 due to larger window
- NIAH accuracy @ N=16: measure and plot

---

### §1 — Stage 1 → Stage 2 Transfer: Scratch vs Finetune

**Option A: Train Stage 2 from scratch**

| Pros | Cons |
|---|---|
| No distribution mismatch from Stage 1 | 2× compute budget |
| SWA positional patterns learned correctly from day 1 for W=1024 | Selector re-learns memory addressing from scratch |
| Cleaner ablation (stages fully independent) | Wastes the fact that Stage 1 already taught the selector to do NIAH retrieval |

**Option B: Finetune Stage 2 from Stage 1 checkpoint** ← RECOMMENDED

| Pros | Cons |
|---|---|
| Selector already knows content-based addressing → faster convergence | Stage 1 trained with chunk=4096; Stage 2 uses chunk=8192 — the "chunk boundary" pattern in positional embeddings is different |
| Writeback gate β has been calibrated; only needs to adapt to new chunk length | Mild distribution mismatch in the first ~200 warmup steps |
| Empirically: LongLLaMA finetuning from a full-attn checkpoint → SWA converges in ~20% of scratch training time | |

**Why finetune is correct here**: The key component that needs transfer is the **TopKSelector's Q/K projections** — these encode "how to address a slot by content". This skill was learned under NIAH supervision and is geometrically independent of window size. The SWA window affects *which local tokens* each position sees, not *how slots are addressed*. The writeback gate warmup reset (200 steps) gives the model time to re-calibrate slot write frequency for the 2× longer chunks.

**Practical recipe for Stage 2 finetune**:
1. Load Stage 1 checkpoint (all adapter weights: selector, slot_to_hidden, hidden_to_slot, gates, bank).
2. Reset `writeback_warmup_steps` counter to 0 in all 32 layers.
3. Use LR = 3e-5 (10% of Stage 1 LR) with cosine decay.
4. Run `--niah_max_N 16` with the same 10% mixture fraction.
5. Monitor PG19 PPL: should dip slightly in first 500 steps then recover. If it exceeds 2.1, the writeback gate is adapting too aggressively — reduce `writeback_gate_max` to 0.2 temporarily.

---

### §2 — Why chunk_size Must Double with W

**The invariant**: keep `W / chunk_size = constant = 12.5%`.

```
Stage 1:  W=512,  chunk=4096  →  W/chunk = 12.5%
Stage 2:  W=1024, chunk=8192  →  W/chunk = 12.5%
```

**Why this matters** (counter-example: W=1024, chunk=4096):

If we doubled W but kept chunk=4096, then within each chunk the SWA window covers W/chunk = 1024/4096 = **25%** of the chunk (vs 12.5% in Stage 1). That means:
- Tokens at positions 0..3071 within the chunk (positions < chunk - W) are still outside the window of any query at the end of the chunk.
- BUT the window now covers the last 1024 positions locally, so "medium-range" information (positions −512 to −1024 within the chunk) that previously required memory is now in the local window.
- The retrieval difficulty per chunk decreases — it's an **easier task at Stage 2, not the same task at larger scale**.

The scaling curve would then confound two things: (a) larger W helping local context, and (b) actual long-range memory improvement.

**By doubling chunk_size**: each chunk now contains 8192 tokens. The local window still covers only the last 12.5% of the chunk (1024/8192). The same "bottleneck pressure" applies — the memory must compress and retrieve the first 7168 tokens of each chunk. The experiment is **isomorphic** across stages; only the absolute token scale changes.

**Corollary for eval**: at the same N (number of chunks), Stage 2 tests 2× the absolute token gap. To compare the same absolute gap, use N_Stage2 = N_Stage1 / 2. Both interpretations are useful; the plot should label each X-axis tick as both N (chunks) and the corresponding absolute gap in K-tokens.

---

### §3 — Eval Design for the Scaling Curve

#### Eval grid

| Axis | Values |
|---|---|
| Stage | W=512/chunk=4096 , W=1024/chunk=8192 |
| N (chunks between needle and query) | {1, 2, 4, 8, 16, 32} |
| Samples per cell | 5 |
| Haystack content | PG19 test split (zero overlap with training) |
| Needle format | `"The secret code for agent <6-char-random-name> is <5-digit-code>."` |
| Query format | `"The secret code for agent <name> is "` (model must predict code) |
| Match criterion | Exact 5-digit code substring match in first 20 generated tokens |

Total cells: 2 stages × 6 N-values × 5 samples = **60 eval runs per stage** = 120 total.

At 5 samples per cell, a 100%→0% transition is detectable with ±20% confidence. For publication quality, increase to 20 samples (480 runs total), but 5 is sufficient for the research signal.

#### Absolute token gaps

```
Stage 1 (chunk=4096):
  N=1  →    4,096 tokens    N=8  →   32,768 tokens
  N=2  →    8,192 tokens    N=16 →   65,536 tokens
  N=4  →   16,384 tokens    N=32 →  131,072 tokens

Stage 2 (chunk=8192):
  N=1  →    8,192 tokens    N=8  →   65,536 tokens
  N=2  →   16,384 tokens    N=16 →  131,072 tokens
  N=4  →   32,768 tokens    N=32 →  262,144 tokens
```

The maximum gap tested (N=32, Stage 2) = 262K tokens, well within a realistic LLM long-context target.

#### Expected "ceiling" — at what N does even perfect memory fail?

Three failure modes, ordered by N at which each bites:

**Mode 1 — EMA decay wipes the needle slot** (N-dependent, fast):
With writeback gate β_max=0.3, each chunk that *addresses the needle's slot* overwrites it partially. After n overwrites, residual signal ≈ (1−β)^n. With β=0.3, the signal halves in n ≈ 2 overwrites. However, this only applies if subsequent chunks select the needle's slot. With load-balance active and 512 slots, the probability that any one chunk addresses a specific slot is ≈ k/N_slots = 64/512 = 12.5% per chunk. Expected number of overwrites in N chunks: N × 0.125. EMA half-life in chunks: ~2/0.125 = **16 chunks**. So N=16 is the natural EMA ceiling for Stage 1 (absolute gap = 65K tokens).

**Mode 2 — Slot selector noise overwhelms signal** (N-independent, fundamental):
With k=64 retrieved out of 512 slots, the fraction of the joint-attention sequence that is the needle slot = 1/64 ≈ 1.6%. The question-pool query must put the needle slot in the top 64 to retrieve it at all. If the selector has not been trained with NIAH supervision, this is essentially random (1/512 ≈ 0.2%). With NIAH training, the selector should learn to boost the relevant slot to top-1, but realistic accuracy depends on training diversity. **This is why Stage 1 NIAH supervision (10% mixture) is non-negotiable.**

**Mode 3 — All 512 slots fully occupied, oldest overwritten** (N-dependent, slower):
At 10% NIAH mixture, most training batches are pg19 (9 out of 10). The bank is constantly receiving LM writes. With 512 slots and k=64 selected per chunk, each slot is expected to be overwritten every 512/64 = 8 chunks. A needle stored at chunk t is therefore at risk of being overwritten in 8 chunks even if NO subsequent batch explicitly targets its slot. At N=32 (32 chunks later), the needle has survived approximately 32/8 = 4 expected overwrite cycles. With β=0.3, survival probability ≈ (0.7)^4 ≈ 24%. This aligns with expected N=32 accuracy ≈ 20-30% for a working memory system (non-trivially above the 0% baseline, but clearly degraded).

**Summary**: Expected curve shape per stage:
```
N=1:     90-100%  (needle written in last chunk, still hot)
N=2:     80-90%
N=4:     60-80%
N=8:     40-60%
N=16:    20-40%   ← EMA half-life threshold
N=32:    10-25%   ← near chance for a 5-digit code (0.001% chance)
```
If Stage 2 (W=1024, chunk=8192) achieves the same percentages at the same N but with 2× the absolute token gap, that is a **positive scaling result**.

---

### §4 — Slot Capacity Analysis

#### Configuration

```
N_slots   = 512   (shared across 32 layers, shared_memory_bank=True)
k_top     = 64    (retrieved per forward pass)
slot_dim  = 4096  (= Llama d_model)
N_layers  = 32    (all wrapped, but ONE shared bank)
```

#### Theoretical maximum distinct facts storable

Raw information capacity of the bank:
```
512 slots × 4096 floats/slot × 16 bits/float (bf16) ≈ 4.2 MB = 33.6 Mbits
```

A needle fact (6-char name + 5-digit code) ≈ 11 ASCII chars ≈ 88 bits. Theoretical ceiling:
```
33.6 Mbits / 88 bits ≈ 380,000 facts
```

This upper bound is meaningless in practice because:
1. Slots are **dense distributed vectors**, not binary key-value stores. Superposition interference limits practical capacity to **O(√N_slots) if random, O(N_slots / log N_slots) with structured addressing** — i.e., ~50-300 reliably separable facts for N=512.
2. The top-k selector retrieves k=64 slots simultaneously. For a needle in slot i to be retrieved, the query must rank slot i in the top 64 out of 512. With a well-trained selector and a distinctive query, this is achievable, but SNR = 1/k = 1/64. Multiple competing facts in nearby slots degrade SNR.
3. EMA writes mix new content into old slots. Unless the load-balance loss perfectly routes each "topic" to a dedicated slot (unlikely), content drifts.

#### Practical capacity estimate

Assuming the selector achieves top-1 precision (the needle slot is always ranked #1 given the correct query), and k=64 slots are retrieved, retrieval of the needle content requires the model to "filter" 1 relevant slot from 63 noisy slots in the joint attention. With NIAH training that specifically rewards this filtering, **practical capacity ≈ 100-300 facts simultaneously** for a 512-slot bank. This is consistent with published FIFO memory models (MemoryLLM 7680 slots achieves ~30% passkey at 128K context — a much larger bank).

#### Is slot capacity the binding constraint on retrieval distance?

No — for the N-values we test (N ≤ 32, 5-digit code facts), **the binding constraint is EMA decay**, not slot capacity. At N=32 there is only **1 needle** in the entire 512-slot bank. Slot capacity would only become binding if we tested "retrieve N_facts distinct needles simultaneously" — which is a different experiment.

**For the scaling curve experiment, capacity is not the bottleneck.** The bottleneck is:
1. EMA decay over chunks (N ≥ 16 regime)
2. Selector query quality: can the 6-char name query correctly address the slot that encoded the needle?

#### Increasing k or N_slots to extend retrieval distance

| Change | Effect on ceiling | Cost |
|---|---|---|
| N_slots: 512 → 1024 | EMA half-life doubles (more slots → lower overwrite rate per chunk) | +2× bank parameter storage |
| k: 64 → 128 | More recall (less chance needle slot is below cutoff), but worse SNR (128 noisy slots vs 64) | +50% selector compute |
| β_max: 0.3 → 0.1 | Slower EMA decay → longer retention, but slower learning of new content | No param cost, but may hurt adaptation |
| chunk_size: double (Stage 2) | At same N, 2× absolute token gap — same effect on EMA | Needs larger W to maintain ratio |

**Recommendation for coder**: Expose `--num_slots` and `--top_k` as eval-time flags so we can quickly probe capacity tradeoffs without retraining.

---

### Scaling Curve Plot Specification

```
Figure: "Memory Retrieval Distance Scaling Curve"

X-axis: N (number of chunks between needle and query) ∈ {1, 2, 4, 8, 16, 32}
        secondary label: absolute token gap (K-tokens)

Y-axis: Retrieval accuracy (%) — exact match, 5 samples per cell

Lines:
  ● W=512,  chunk=4096  (Stage 1)
  ■ W=1024, chunk=8192  (Stage 2)

Shaded band: ±1σ across 5 samples (just error bars, not full band, given n=5)

Reference line: 0.001% chance level (random 5-digit code) — nearly invisible at 0

Interpretation guide (add as caption):
  - If both curves have the same shape but Stage 2 spans 2× the absolute gap at
    each N: "memory retrieval distance scales with chunk_size (∝ W)."
  - If Stage 2 curve is shifted right by 1 tick on the N-axis (equal accuracy
    at N_Stage2 = N_Stage1 / 2): "same absolute retrieval distance, but more
    efficient (fewer chunk-boundary crossings)."
  - If Stage 2 is uniformly higher at each N: "larger window improves memory
    quality per chunk, not just gap distance."
```

---

### Action Checklist

**For coder** (`/coder`):

- [ ] Add `--swa_window W` flag to `scripts/train_mem_space_pg19.py` (currently absent; training uses full local attention within each chunk)
- [ ] Implement NIAH synthetic data generator as a `torch.utils.data.IterableDataset` that yields mixed pg19 + NIAH batches; accept `--niah_mix_fraction` and `--niah_max_N`
- [ ] Verify that the NIAH loss is masked to answer tokens only (not haystack CE)
- [ ] Add `--init_from <ckpt>` finetune entry point with LR and warmup_steps override
- [ ] Extend `scripts/eval_niah_mem_space.py` to accept `--N_list 1,2,4,8,16,32` and `--samples_per_cell 5`; output a CSV with columns: `[stage, N, sample_id, expected_code, generated_text, exact_match]`
- [ ] Add `--chunk_size` flag (currently hardcoded as 4096 in the data loader); make it configurable
- [ ] Unit test: verify SWA mask is correctly constructed (slot tokens see everything; content tokens see slots + causal W-window)

**For trainer** (`/trainer`):

- [ ] Stage 1: launch on 8×B200 using A.2 ship config + `--swa_window 512 --niah_mix_fraction 0.10 --niah_max_N 16 --max_steps 30000`
- [ ] Stage 1 acceptance check at step 5000 (early stop gate): PPL ≤ 2.0, NIAH@N=1 ≥ 70%
- [ ] Stage 2: launch finetune from best Stage 1 checkpoint with `--swa_window 1024 --chunk_size 8192 --lr 3e-5 --writeback_warmup_steps 200 --max_steps 15000`
- [ ] Run full eval grid after each stage (120 eval runs, log to `ops/eval_results/curriculum_scaling_YYYYMMDD.csv`)
- [ ] Plot scaling curve (matplotlib, save to `ops/figures/retrieval_distance_scaling.pdf`)
- [ ] Monitor `gpu_runs.jsonl` + `ACTIVE_SWEEPS.jsonl` for run health; kill and re-launch at lower LR if Stage 2 PPL exceeds 2.1 at step 500

---

### Open Questions (for next researcher turn)

1. **NIAH training format**: should the question appear within the same chunk as the haystack, or as a separate N-th chunk? The current spec uses "N-th chunk = question only". This means the model must have memorized the answer through slots by the time it sees the question. Alternative: include a short "retrieval cue" preamble in the question chunk (e.g., replay last 128 tokens before the question) — reduces the short-T pathology documented in `niah_v8_diagnosis.md §1.2`.

2. **Should Stage 1 NIAH training use W=512 or full attention within chunks?** If SWA is enabled during NIAH training and the needle happens to be in a position outside the window, the model cannot even encode it locally. Recommendation: during NIAH training only, force `swa_window = chunk_size` (= full attention within each chunk) so the encoding is not impaired. Apply SWA only during pg19 batches.

3. **Transfer learning stability**: if Stage 2 PPL does not recover within 1K steps of finetuning, fall back to scratch training and report both numbers.
