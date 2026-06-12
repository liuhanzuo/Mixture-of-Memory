# v3: RMT-Slot Hybrid — Top-k Slot Retrieval + RMT Sandwich

## Architecture Overview

Combines two previously separate approaches:
- **RMT v10 sandwich injection**: `[old_mem | content | placeholder_mem]` with continuous position IDs
- **Slot memory top-k retrieval**: N=64 persistent slots, retrieve top-8 per segment, EMA write-back only to selected slots

**Key innovation**: Slots become the memory bank, retrieved slots become "first-class" input tokens via RMT sandwich (not just mid-layer cross-attention like H/H2).

## Core Components

### RMTSlotModel
- Backbone: Llama-3-8B (full fine-tune)
- Memory bank: N=64 slots, slot_dim=4096 (Llama hidden size)
- Selector: TopKSelector with selector_dim=128, top_k=8
- Sandwich: `[retrieved_slots (8) | content (1024) | placeholder (8)]` = 1040 tokens
- Position IDs: continuous 0..1039 (no offset)
- Attention mask: lower-triangular with memory prefix bidirectional

### Training Protocol
- Segment length: 1024 tokens
- Max segments: 4 (4096 context)
- BPTT: full (-1) or truncated (positive depth)
- EMA gate: learnable sigmoid logit (init 0.3)
- LR: 5e-6, grad_accum=4, max_steps=2000

## Key Differences from Previous Approaches

| Approach | Memory Mechanism | Token Access | Position IDs |
|----------|------------------|--------------|--------------|
| **H/H2** (middle-layer) | Cross-attention at specific layers | Hidden states only | Original segment-local |
| **RMT v10** | Fixed K memory tokens | Sandwich injection | Continuous 0-based |
| **RMT-Slot** | Top-k from N slots | Sandwich injection | Continuous 0-based |

## Expected Advantages

1. **Better NIAH**: Slots as first-class tokens → backbone sees them uniformly across all layers
2. **Avoids RMT collapse**: Only update top-k slots per segment → prevents "train all K at once" dilution
3. **Leverages proven components**: Reuses working TopKSelector + MemoryBank from H/H2
4. **Continuous positions**: No position_type="Zero" trap (fixed in MemLong)

## Files Created

- `src/memory/rmt_slot/__init__.py`
- `src/memory/rmt_slot/rmt_slot_model.py`
- `scripts/train_rmt_slot.py`
- `scripts/launch_rmt_slot.sh`
- `versions/v3_rmt_slot.md`

## Target Node

b200-2 (28.89.17.144) — currently FREE

## Success Metrics

- Memory ratio < H/H2 at same step (target < 0.977)
- NIAH accuracy > 0% (breakthrough over H/H2's 0%)
- Training completes 2000 steps without NaN/crash

## Risk Mitigation

- Start with simple EMA write-back (no complex routing initially)
- Use proven hyperparameters from H/H2 (LR=5e-6, etc.)
- Early validation at step 100 to catch issues
