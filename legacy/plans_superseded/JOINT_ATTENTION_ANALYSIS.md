# CURRENT "JOINT ATTENTION" MEMORY IMPLEMENTATION — DETAILED ANALYSIS

## Executive Summary

The **active training** (Experiments E & F, as of 2026-05-08) uses **"slot-forward" mode**, which implements **joint attention** by concatenating memory slots to hidden states and forwarding them through ALL decoder layers with a single unified softmax attention.

**Key finding**: This is NOT cross-attention. It's MemLong-style **joint attention**: 
```python
# Line 630: train_cross_attn_memory.py
extended = torch.cat([slots, hidden_states], dim=1)  # [B, S+T, d_model]
# Then forward through vanilla LlamaDecoderLayer with extended sequence
```

---

## 1. ACTIVE MEMORY FOLDER & IMPLEMENTATION

**Location**: `/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/src/memory/mem_space/`

**Primary implementation file**: 
- `scripts/train_cross_attn_memory.py` — Training script with `slot_forward` mode
- Lines **606–679**: `_forward_slot_forward()` method (the active joint attention implementation)

**Key supporting classes** (in `src/memory/mem_space/selector.py`):
- `CrossAttentionMemoryV2` (lines 714–920) — Only used in **non-slot-forward mode** (disabled)
- `TopKSelector` (lines 39–326) — Only used in non-slot-forward mode

---

## 2. ATTENTION FORMULA & ARCHITECTURE

### Joint Attention (Slot-Forward Mode)

**Forward formula** (lines 630–643, train_cross_attn_memory.py):

```python
# Per transformer layer:
slots = self.slot_values[layer_idx]              # [B, S, d_model]
hidden_states = <input to layer>                 # [B, T, d_model]

# JOINT CONCATENATION (line 630):
extended = torch.cat([slots, hidden_states], dim=1)  # [B, S+T, d_model]

# SINGLE UNIFIED ATTENTION:
layer_out = layer(
    extended,
    attention_mask=ext_attn_mask,           # Prefix causal mask
    position_ids=None,
    use_cache=False,
    position_embeddings=ext_pos_emb,        # RoPE: slots at pos=0
)

# OUTPUT SPLIT:
new_slots = output[:, :S, :]                # [B, S, d_model]
hidden_states = output[:, S:, :]            # [B, T, d_model]
```

**NOT two separate attentions**: The entire [S+T] sequence goes through the **same MHA layer** with:
- Single shared Q/K/V projections (from base model, **not** frozen — trained via LoRA/fine-tune)
- Single softmax over [S+T] positions
- Gradients flow to ALL parameters (shared Q/K/V)

### Attention Mask Pattern (Lines 569–589)

Prefix-causal mask for [S+T] sequence:

```
           Slots (S)  |  Tokens (T)
         [0..S-1]     | [S..S+T-1]
Slots    [  0    ]    | [  0    ]    <- slots attend to everything
Tokens   [  0    ]    | [causal ]    <- tokens causal mask
```

Result: 
- Slots can attend to all other slots + all tokens (no masking)
- Tokens can attend to all slots + causal positions of tokens
- Prevents "future token leakage" but slots act as summary prefix

---

## 3. PROJECTION SHARING & WRITE LOCATIONS

### Q/K/V Projections
- **Shared with base model**: YES, use Llama-3-8B pretrained Q/K/V projections
- **Frozen**: NO — trained via **LoRA fine-tuning** (args.lora_rank defaults to 16)
- **Separate memory projections**: NO

### Which Layers Read Memory?
- **ALL 32 layers** of LlamaForCausalLM read from slots
- Each layer `i` has its own `slot_values[i]` (per-layer memory)

### Which Layers Write Memory?
- **ALL 32 layers** write to their own slots
- Write happens at **layer output** (line 646):
  ```python
  new_slots = output[:, :S, :]  # Extract slot portion from layer output
  self.slot_values[layer_idx] = new_slots.detach()  # Store for next chunk
  ```
- No special "middle layer" — unlike MemLong's layer-13 bottleneck, every layer maintains its own bank

**Key insight**: Each layer's slots contain that layer's semantic representation:
- Layer 1: near-embedding space
- Layer 16: mid-layer semantics  
- Layer 32: logit-space representation

---

## 4. VERSION & ACTIVE CONFIGURATION

**Current version**: **v4_chunk_last_hidden_memory.md** (most recent, but **NOT** being actively trained)

**Actually running**: Modified **v2_cross_attention.py** with `--slot_forward` flag enabled

Status file evidence (TRAINER_ACTIVE.md, lines 12–18):
```
Experiment E: C Replicate (b200-3) — RUNNING
  Step | ~190/5000, first eval imminent
  contrastive=0 (disabled for slot_forward)

Experiment F: Enhanced Retrieval (b200-4) — JUST LAUNCHED
  Config: Same as C + lambda_retrieve=10.0, niah_mix_fraction=0.50, niah_warmup=200
```

**Launch command** (scripts/launch_v4_phase2.sh):
```bash
torchrun --nproc_per_node=8 scripts/train_v4_chunk_memory.py
```

But Exp E/F actually use:
```bash
# inferred from TRAINER_ACTIVE.md config mentions
scripts/train_cross_attn_memory.py \
  --slot_forward \
  --lambda_retrieve 10.0 \
  --niah_mix_fraction 0.50
```

---

## 5. MEMORY READ/WRITE LAYER BREAKDOWN

| Layer ID | Reads From | Writes To | Semantic Level |
|----------|-----------|-----------|----------------|
| 0 (embed layer) | — | — | (embedding input) |
| 1–31 (transformer) | slot_values[i] | slot_values[i] | Layer-i hidden semantics |
| 32 (final) | slot_values[31] only | (no write) | Logit-space |

**No special middle layer**: Unlike MemLong (layer 13) or RMT (configurable), slot-forward uses **all-layer memory**.

**Write timing**:
- Happens **within each chunk** (per forward pass)
- Detached (`new_slots.detach()`, line 648) to prevent gradient flow across chunks
- Persists across chunks (stored in `self.slot_values[layer_idx]`)

**Per-layer slot dimension**: All slots are [B, S, d_model] where d_model=4096 for Llama3-8B

---

## 6. SPECIFIC FILE REFERENCES

### Core Joint Attention Implementation

| File | Lines | Content |
|------|-------|---------|
| `scripts/train_cross_attn_memory.py` | 606–679 | `_forward_slot_forward()` — joint attention forward pass |
| `scripts/train_cross_attn_memory.py` | 630 | **`torch.cat([slots, hidden_states], dim=1)` — the concat** |
| `scripts/train_cross_attn_memory.py` | 569–589 | `_build_extended_attn_mask()` — prefix causal mask |
| `scripts/train_cross_attn_memory.py` | 591–604 | `_extend_position_embeddings()` — RoPE handling (slots at pos=0) |
| `scripts/train_cross_attn_memory.py` | 426–442 | Model initialization (slot_forward bypasses CrossAttentionMemoryV2) |

### Version Documentation

| File | Content |
|------|---------|
| `versions/v4_chunk_last_hidden_memory.md` | Described architecture (sparse per-layer banks, not actively used) |
| `versions/v3_infini_attention.md` | Previous Infini-Attention attempt (replaced) |
| `versions/v2_cross_attention.md` | Earlier cross-attention (baseline, now replaced by slot_forward) |

---

## 7. CRITICAL ARCHITECTURAL DETAILS

### Memory Slot Initialization

**Slot_forward mode** supports three initialization methods (lines 459–481):

| Method | Code | Semantics |
|--------|------|-----------|
| `learnable` | Trained `nn.Parameter` vectors | Learnable memory bank (most flexible) |
| `mlp` | MLP projects hidden → slots | Dynamic compression (more adaptive) |
| `strided` | Strided sampling from hidden states | Deterministic but needs careful tuning |

Active experiments use: **strided** (default, line 479)

### Enable Write Gradient

**Important distinction**:
- Training mode: `enable_write_grad=True` (line 801)
- Inference mode: `enable_write_grad=False` (frozen slots)

When `enable_write_grad=True`: slot gradients participate in loss, so model learns to write useful information.
When `enable_write_grad=False`: slots don't receive gradients, only read path is supervised.

---

## SUMMARY TABLE

| Aspect | Value |
|--------|-------|
| **Memory pattern** | Joint attention (concat + single softmax) |
| **Folder** | src/memory/mem_space/ (selector.py utilities, but slot_forward is in train_cross_attn_memory.py) |
| **Write layer(s)** | ALL 32 layers (per-layer banks) |
| **Read layer(s)** | ALL 32 layers (same per-layer banks) |
| **Q/K/V sharing** | Yes, shared with base model (not frozen, trained via LoRA) |
| **Attention formula** | Single unified MHA over [slots; tokens] with prefix-causal mask |
| **Slots per layer** | Always [B, 8, 4096] (8 slots, d_model=4096) |
| **Initialization** | Strided sampling from hidden states (active default) |
| **Detach strategy** | `detach()` at chunk boundaries (line 648), write gradients enabled during training |
| **Version** | v4 described, but v2+slot_forward actually running |

