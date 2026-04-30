# Schedule Regression Root-Cause Analysis
**Date**: 2026-04-27  
**Experiment under investigation**: Branch-3 A.2, 1000-step extension  
**Analyst**: Claude (researcher subagent)

---

## 1. The Numbers

| Run | Steps | σ | warmup | PPL | Δ |
|-----|-------|---|--------|-----|---|
| 200-step champion (original) | 200 | 0.02 | 500 | **1.9051** | baseline |
| 200-step champion (new) | 200 | 0.05 | 1000 | **1.8131** | −0.092 |
| **1000-step extension** | **1000** | **0.02** | **500** | **4.8860** | **+2.981 (+156%)** |

All three runs used identical flags: `--shared_memory_bank`, `--unfreeze_hidden_to_slot`, `--lr 1e-3`, `--top_k 64`, `--num_slots 512`, `--batch_size 1`, `--attn_impl sdpa`, `--dtype bfloat16`.

---

## 2. Code Audit — What Changes When Steps Go From 200 → 1000

### 2.1 LR Scheduler — Does Not Exist

**File**: `scripts/train_mem_space_pg19.py`

```python
optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))
# NO scheduler object created. No scheduler.step() call anywhere in the training loop.
```

The optimizer runs at **constant lr=1e-3 for all 1000 steps**. The 200-step champion naturally stops at step 200; the 1000-step run applies the same step size for 5× more gradient updates with no decay. AdamW's second-moment buffer `v` accumulates across all 1000 steps — later-step effective LR per-parameter can *increase* as `v` saturates for some parameters and decreases for others, creating heterogeneous drift.

### 2.2 Writeback Gate Schedule — Asymmetric Plateau

**File**: `src/memory/mem_space/layer.py`, method `_current_beta`

```python
def _current_beta(self) -> torch.Tensor:
    warmup = cfg.writeback_gate_warmup_steps     # 500 in both runs
    if warmup <= 0:
        warmup_frac = 1.0
    else:
        warmup_frac = min(float(self.step_counter) / float(warmup), 1.0)
    return torch.sigmoid(self.gate_param) * warmup_frac * cfg.writeback_gate_max
```

With `gate_param` initialized to 0.0 (`writeback_gate_init=0.0` in config.py):
- `sigmoid(0.0) = 0.5`
- `gate_max = 0.3` (`--writeback_gate_max 0.3`)

**Beta values across training:**

| Step | warmup_frac | β |
|------|-------------|---|
| 1 | 1/500 = 0.002 | 0.0003 |
| 100 | 100/500 = 0.20 | 0.030 |
| 200 | 200/500 = **0.40** | **0.060** |
| 500 | 500/500 = **1.00** | **0.150** |
| 501–1000 | 1.00 (clamped) | **0.150** (plateau) |

**Critical finding**: The 200-step champion terminates at step 200 with `β = 0.060`. The warmup is 40% complete — the gate is barely open. The 1000-step run:
1. Reaches full `β = 0.150` at step 500
2. Maintains `β = 0.150` for **500 additional steps** (steps 501–1000)

### 2.3 EMA Time Constant at Full Gate

**File**: `src/memory/mem_space/memory_bank.py`, method `write`

```python
updated = (1.0 - gate_t) * current + gate_t * new_contrib
self.slots = self.slots.scatter(1, idx_exp, updated)
```

At `β = 0.150`:
- EMA time constant τ = 1/β ≈ **6.7 training steps**
- After 500 plateau steps (steps 501–1000): each slot written `top_k=64` times; with 512 slots, average slot hit rate = 64/512 = 12.5% per step
- Expected full content replacement per slot: ≈ 6.7/(0.125) = **~54 steps** for a slot to forget 63% of its content

For a **shared bank across 32 layers**: all 32 layers read the same bank and write back independently within a single chunk. The effective write rate per slot per chunk is 32 × hit_rate = 32 × 12.5% = **400% per chunk** (every slot is touched ~4 times per chunk on average). This means at full gate (β=0.15), slot content turns over in approximately **2 chunks**, not 54 steps.

### 2.4 Chunk Reset — Inter-Chunk Break Preserved

**File**: `scripts/train_mem_space_pg19.py`, `_reset_banks`

```python
def _reset_banks(model):
    for w in getattr(model, "_mem_space_layers", []):
        w.memory_bank.reset()
```

Called before every chunk in the training loop. `memory_bank.reset()` sets `self.slots = None`, forcing re-init from `torch.randn * init_noise` at next forward. This **disconnects gradient flow between chunks** (correct behavior — prevents BPTT through the bank across chunk boundaries).

Effect: each chunk is a fresh episode. The question is whether the slot parameters *within one chunk* (slot init noise → EMA writes → final slot state → output gate) learn stable representations. With full gate at β=0.15 and 32-layer shared writes, slot content at the end of one 4096-token chunk is entirely determined by the last few layer-writes — early-in-chunk information is overwritten.

### 2.5 Data Distribution — Chunks 200–1199 vs 200–399

The 200-step champion trains on `skip_chunks=200`, `max_chunks=200` → pg19 chunks [200, 399].  
The 1000-step run trains on `skip_chunks=200`, `max_chunks=1000` → pg19 chunks [200, 1199].

Chunks 400–1199 are unseen in the champion. If the pg19 chunk distribution shifts (e.g., entering a different book's content with different vocabulary patterns), the memory selector and slot assignments could require re-adaptation. This is a lower-probability cause but worth noting.

### 2.6 hidden_to_slot — 1000 Gradient Steps at Constant lr

**File**: `src/memory/mem_space/config.py`

```python
hidden_to_slot_frozen: bool = True   # default; overridden by --unfreeze_hidden_to_slot
```

The `hidden_to_slot: Linear(4096→4096)` projection (16.8M parameters) receives gradients via the writeback path: `O_mem_slot = hidden_to_slot(O_mem_hidden)`. With:
- Constant lr=1e-3 (no decay)
- 1000 gradient steps
- Full gate (β=0.15) at steps 500–1000: writeback now carries strong gradient signal

The projection can drift toward degenerate modes — e.g., mapping all hidden states to similar vectors, causing slot collapse, or developing rank-deficient representations that corrupt the slot bank.

---

## 3. Root Cause Hypotheses

### H1 — No LR Decay → Over-training Past the Optimal Point

**Mechanism**: The 200-step champion implicitly benefits from "early stopping" at a point where AdamW momentum hasn't yet caused parameter drift. At step 200, the optimizer has seen only 200 gradient updates at lr=1e-3. By step 1000, AdamW's second-moment buffer `v` has absorbed 5× more gradient history; for parameters with low gradient signal (e.g., `slot_output_gate` near zero), `v` decreases → *effective* lr increases → these parameters become noisy. The result is degraded `slot_output_gate` → `alpha = tanh(gate)` overfit, disrupting the Flamingo bypass that was working at step 200.

**Evidence in favor**: No LR scheduler exists in `train_mem_space_pg19.py`. This is a structural absence, not a misconfiguration. Every iteration beyond 200 runs at the same lr=1e-3 with growing momentum history.

**Evidence against**: AdamW lr=1e-3 is not extreme for this scale. The model parameters outside the memory module are *frozen* (only memory module parameters are trained), so parameter count is small — fewer bad minima to fall into.

**Probability: Medium-High (35%)**

---

### H2 — Full Writeback Gate Plateau (β=0.15 for 500 Steps) → Slot Content Collapse

**Mechanism**: This is the most structurally unique difference between the 200-step and 1000-step runs.

- 200-step champion: gate completes warmup at step 500 — **but training ends at step 200**, so the gate *never reaches β=0.15*. The entire champion experiment operates at β ≤ 0.06 (partial warmup).
- 1000-step run: gate reaches β=0.15 at step 500, then holds there for 500 more steps.

At full gate with shared bank and 32 layers:
- Effective slot turnover: ~2 chunks per full slot content refresh
- The slot bank is being aggressively overwritten across 32 layers simultaneously
- EMA at β=0.15 provides negligible smoothing (τ≈7 steps) — slots behave more like FIFO than stable memory
- This aggressive overwrite may cause the slot bank to lose stable, reusable features across chunks (even though intra-chunk gradients exist, if slots are fully overwritten mid-chunk by lower layers, upper-layer reads become incoherent)

The warming phase (steps 1–500) may have trained the selector and output gate correctly for the "soft gate" regime. Sudden transition to "hard gate" at step 500 creates a **distributional shift** in what the memory module outputs — the output gate (`alpha`) was trained for low-writeback signals and now receives high-writeback (high-noise) slot content.

**Evidence in favor**:
- Mathematical asymmetry is directly verifiable from code
- Flamingo gate (`alpha = tanh(slot_output_gate)`) has no schedule of its own — it can grow to compensate, but may overshoot
- The new 200-step champion (σ=0.05, warmup=1000) avoids this: at step 200 with warmup=1000, β = 0.5×(200/1000)×0.3 = **0.03** — even lower gate. The higher σ and lower gate combination may be more stable precisely *because* the gate is more conservative.

**Evidence against**: The writeback gate_param can be learned. `torch.sigmoid(self.gate_param)` can decrease if the gradient signal rewards lower beta. If the training correctly learns to reduce gate_param, the plateau may self-regulate.

**Probability: High (50%)**

---

### H3 — hidden_to_slot Gradient Drift at Constant lr=1e-3 for 1000 Steps

**Mechanism**: `hidden_to_slot: Linear(4096→4096)` starts from a random init (or pretrained init from Llama's mlp?) and receives gradients through the EMA writeback path. At full gate (β=0.15, steps 500–1000), the writeback path becomes the dominant learning signal. With no LR decay, 1000 gradient steps at lr=1e-3 can push this 16.8M-parameter matrix to degenerate solutions:
- **Mode collapse**: map all tokens to similar slot vectors → all slots converge → slot diversity drops → load-balance loss explodes
- **Rank collapse**: hidden_to_slot becomes low-rank → slot representations lose expressiveness → memory reads become uninformative

This interacts with H2: high β amplifies the gradient through `hidden_to_slot`, while constant LR provides no regularization.

**Evidence in favor**: The interaction between high-gate writeback and trainable `hidden_to_slot` is unique to runs that survive past warmup_steps. This is supported by the new champion (σ=0.05, warmup=1000) showing better 200-step PPL — its lower β at 200 steps means weaker `hidden_to_slot` gradients throughout training.

**Evidence against**: `hidden_to_slot` only receives gradients through the slot write → read → output gate path. If the output gate `alpha` is small (early in training), the gradient magnitude through this path is attenuated by `tanh`. The gradient may be small enough that `hidden_to_slot` doesn't drift significantly in the first 1000 steps.

**Probability: Medium (15%)**

---

## 4. Verification Experiments

All three experiments are ≤200 steps, targeting specific hypotheses, and runnable in ≤30 minutes on a single B200 node (8×B200).

### Exp-V1: Freeze Writeback Gate at β=0.06 for the Full 1000 Steps (Tests H2)

**Hypothesis tested**: H2 — if the gate plateau is the cause, then running 1000 steps with the gate clamped to the level seen at step 200 should prevent regression.

**Method**: Set `writeback_gate_max = 0.3` and `writeback_warmup_steps = 5000`. At step 1000, warmup_frac = 1000/5000 = 0.20, so `β = 0.5 × 0.20 × 0.3 = 0.03`. This is even lower than the 200-step champion's β=0.06 at termination, but tests whether "never reaching full gate" preserves low PPL.

**Alternative**: Set `writeback_gate_max = 0.06` with `warmup_steps = 0` (no warmup, constant gate). This directly reproduces the β the champion operated under.

**Script modification** (one-line change to `_run_branch3_A2_schedule_match_1000.sh`):
```bash
--writeback_gate_max 0.06 \   # replaces 0.3
--writeback_warmup_steps 0 \  # no warmup, constant gate
```

**Expected outcome**:
- If H2 is correct: PPL stays ≤ 2.0 even at 1000 steps
- If H2 is wrong: PPL still degrades → H1 or H3 is dominant

**Cost**: 8×B200, 1000 steps × 4096 tokens × 200 chunks ≈ 30 min

---

### Exp-V2: Cosine LR Decay + Original Gate Schedule (Tests H1)

**Hypothesis tested**: H1 — if constant LR is the cause, adding cosine decay should prevent regression.

**Method**: Add a cosine LR scheduler from step 1 to 1000. Peak lr=1e-3, final lr=1e-5 (0.01× ratio). Keep all other params identical to the 1000-step regression run.

**Script modification** (add after optimizer creation):
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=args.max_train_steps, eta_min=args.lr * 0.01
)
# Add scheduler.step() at end of each optimizer step
```

**Expected outcome**:
- If H1 is correct: PPL stays ≤ 2.5 even at 1000 steps
- If H1 is wrong and gate plateau (H2) dominates: PPL still degrades past β=0.15

**Cost**: 8×B200, 1000 steps ≈ 30 min  
**Note**: This requires a code modification to `train_mem_space_pg19.py`.

---

### Exp-V3: 200-Step Run with warmup=200 (Gate Fully Open at Termination) (Tests H2 Inversely)

**Hypothesis tested**: H2 — if the gate reaching full strength is harmful, then a 200-step run that reaches full β at exactly step 200 should perform worse than the champion.

**Method**: Same config as the 200-step champion (σ=0.02), but set `warmup=200` instead of `warmup=500`. At step 200: β = 0.5×(200/200)×0.3 = **0.15** (full gate at termination).

**Script modification** (one line in ablation script):
```bash
--writeback_warmup_steps 200 \   # replaces 500
```

**Expected outcome**:
- If H2 is correct: PPL significantly worse than 1.9051 (e.g., PPL ≥ 3.0)
- If H2 is wrong: PPL similar to or better than 1.9051

**Cost**: 8×B200, 200 steps × 4096 tokens × 200 chunks ≈ 6 min  
**Recommended priority**: **Run this first** — cheapest, most direct test of H2.

---

## 5. No-Memory Baseline

**Goal**: Establish the no-memory PPL baseline using the new champion config (σ=0.05, warmup=1000, 200 steps) with zero code changes.

**Method**: The `--bypass_memory` flag is already implemented in `train_mem_space_pg19.py`:

```python
# From train_mem_space_pg19.py:
if args.bypass_memory:
    from src.memory.mem_space.layer import MemorySpaceLayer
    mem_layers = getattr(model, "_mem_space_layers", [])
    for w in mem_layers:
        w.forward = w.forward_no_memory.__get__(w, MemorySpaceLayer)
```

Where `forward_no_memory` in `layer.py` is:
```python
def forward_no_memory(self, hidden_states, ...):
    return self.wrapped_layer(hidden_states, ...)  # pure Llama, no memory
```

**Command** (copy champion script, add one flag):
```bash
torchrun --nproc_per_node=8 --master_port=29522 \
    "$PROJECT_DIR/scripts/train_mem_space_pg19.py" \
    --model "$MODEL" \
    --data "$DATA" \
    --max_chunks 200 \
    --skip_chunks 200 \
    --seq_len 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --selector_dim 128 \
    --writeback_gate_max 0.3 \
    --writeback_warmup_steps 1000 \
    --load_balance_weight 0.01 \
    --max_train_steps 200 \
    --lr 1e-3 \
    --attn_impl sdpa \
    --dtype bfloat16 \
    --slot_init random \
    --slot_init_noise 0.05 \
    --shared_memory_bank \
    --unfreeze_hidden_to_slot \
    --bypass_memory \                       # <-- THE ONLY DIFFERENCE
    --output_dir "$PROJECT_DIR/outputs/branch3_A2_no_memory_baseline"
```

**Zero code changes required**. The flag already exists.

**Expected outcome**: PPL ≈ the frozen Llama-3-8B perplexity on pg19 chunks [200, 399] in sequential mode (likely 8–15 PPL range). The gap between this baseline and 1.8131 is the "memory contribution" claim.

**Important caveat**: In `--bypass_memory` mode, training still runs (optimizing selector and other module params that receive no gradient in bypass mode). The eval PPL will reflect a *frozen* Llama-3-8B, which is a valid baseline.

---

## 6. Prioritized Experiment Plan

| Priority | Experiment | Estimated Cost | Hypothesis | Node |
|----------|-----------|---------------|-----------|------|
| **1** | Exp-V3: warmup=200 (full gate at 200 steps) | **6 min** | H2 (inverse) | b200-1 |
| **2** | No-Memory Baseline | 6 min | Baseline | b200-2 |
| **3** | Exp-V1: constant β=0.06 for 1000 steps | 30 min | H2 (direct) | b200-3 |
| **4** | Exp-V2: cosine LR for 1000 steps | 30 min | H1 | b200-4 |

Run Exp-V3 first: if PPL degrades with warmup=200, H2 is confirmed in <10 minutes without needing to run the full 1000-step experiments.

---

## 7. Summary

**Most likely cause**: **H2 — Writeback gate plateau asymmetry**.

The fundamental asymmetry: the 200-step champion operates at β ≤ 0.06 (warmup only 40% complete at termination), while the 1000-step run plateaus at β = 0.15 for 500 consecutive steps. The shared memory bank under 32-layer concurrent EMA writes at full gate has an effective slot turnover of ~2 chunks — making the bank unstable as a persistent memory.

**Supporting evidence from new champion**: The best performing 200-step run (σ=0.05, warmup=1000, PPL=1.8131) also terminates at step 200 with β = 0.5×(200/1000)×0.3 = **0.03** — an even lower gate than the original champion. The trend is consistent: lower β at training termination → better PPL.

**Second factor**: No LR scheduler (H1) is a compounding issue for runs that go beyond the natural "sweet spot" of ~200 steps. Both H1 and H2 would be solved by: (a) reducing gate_max or using longer warmup, and (b) adding cosine LR decay.

**Immediate action recommendation**:  
Run Exp-V3 (warmup=200, 6 min) on any free B200 node to confirm/deny H2 before committing to the 30-minute experiments.

---

```json
{
  "report": "20260427_schedule_regression_analysis",
  "date": "2026-04-27",
  "experiment": "branch3_A2_schedule_match_1000",
  "regression": {
    "ppl_champion_200step": 1.9051,
    "ppl_new_champion_200step": 1.8131,
    "ppl_1000step": 4.886,
    "delta_from_original_champion": 2.981
  },
  "most_likely_cause": "H2",
  "hypotheses": {
    "H1": {"name": "no_lr_scheduler_overtraining", "probability": 0.35},
    "H2": {"name": "writeback_gate_plateau_slot_collapse", "probability": 0.50},
    "H3": {"name": "hidden_to_slot_gradient_drift", "probability": 0.15}
  },
  "key_finding": "200-step champion terminates at beta=0.06 (40% warmup); 1000-step run plateaus at beta=0.15 for 500 steps. The champion NEVER experiences full writeback gate strength.",
  "new_champion_insight": "sigma=0.05 warmup=1000 also terminates at beta=0.03 (20% warmup) -- even lower gate -- further supporting H2.",
  "no_memory_baseline": {
    "flag": "--bypass_memory",
    "code_changes_required": 0,
    "description": "Already implemented in train_mem_space_pg19.py"
  },
  "recommended_experiments": [
    {
      "id": "V3",
      "description": "200-step run with warmup=200 (gate reaches full beta=0.15 at termination)",
      "cost_minutes": 6,
      "tests": "H2 (inverse prediction: should degrade vs warmup=500)",
      "priority": 1
    },
    {
      "id": "V1",
      "description": "1000-step run with constant beta=0.06 (gate_max=0.06, warmup=0)",
      "cost_minutes": 30,
      "tests": "H2 (direct prediction: should not regress)",
      "priority": 3
    },
    {
      "id": "V2",
      "description": "1000-step run with cosine LR decay (peak 1e-3 -> final 1e-5)",
      "cost_minutes": 30,
      "tests": "H1",
      "priority": 4
    }
  ]
}
```
