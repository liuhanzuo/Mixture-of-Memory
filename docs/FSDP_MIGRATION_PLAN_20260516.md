# FSDP Migration Plan: DDP → Fully Sharded Data Parallel

**Date**: 2026-05-16  
**Target**: Llama-3-8B + L1+L2+L3 memory architecture  
**Problem**: Phase 11 OOM on H20 (97 GB/card) due to PyTorch caching allocator fragmentation  
**Solution**: FSDP with full sharding (ZeRO-3) to shard params + grads + optimizer state across 8 ranks

---

## 1. Current DDP Setup — Exact Code Locations

### 1.1 DDP Initialization
**File**: `scripts/train_mem_space_babilong.py`
- **Line 54**: `from torch.nn.parallel import DistributedDataParallel as DDP`
- **Line 141**: `dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)` in `init_distributed()`
- **Line 709-710**: DDP wrapping
  ```python
  model = DDP(model, device_ids=[local_rank], output_device=local_rank,
              find_unused_parameters=True)
  ```

### 1.2 Optimizer Setup
**File**: `scripts/train_mem_space_babilong.py`
- **Lines 822-826**: Optimizer initialization
  ```python
  trainable = _mem_space_params(model.module if isinstance(model, DDP) else model)
  optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0,
                                betas=(0.9, 0.95))
  ```
  Note: Optimizer targets only trainable mem_space params (frozen backbone excluded)

### 1.3 Training Loop — Backward & Step
**File**: `scripts/train_mem_space_babilong.py`
- **Line 880**: `loss.backward()`
- **Line 888**: `optimizer.step()`
- **Lines 850, 886**: Gradient clipping applied

### 1.4 State Dict Gathering for Checkpointing
**File**: `scripts/train_mem_space_babilong.py`
- **Lines 915-938**: `_save_adapter()` function
  ```python
  root = model.module if isinstance(model, DDP) else model
  state = {
      k: v.detach().cpu()
      for k, v in root.state_dict().items()
      if any(frag in k for frag in fragments)
  }
  torch.save(state, ckpt_path)
  ```
  Currently uses direct `.state_dict()` on the unwrapped model; FSDP requires context manager for gather.

---

## 2. FSDP Wrapping Strategy

### 2.1 Module-Level Wrapping Decision
**Recommendation: Option (b) — Wrap only trainable mem_space modules**

| Approach | Memory Saved | Throughput Impact | Recommendation |
|----------|-------------|------------------|---|
| **(a)** Wrap frozen backbone + trainable modules | 2 GB per rank | 5-10% slower (AllGather) | ❌ Overkill, adds complexity |
| **(b)** Keep backbone replicated, wrap only trainable layers | 22 GB sharded (→ 2.75 GB/rank savings) | 1-2% slower (only trainable AllGather) | ✅ **Optimal** |

**Rationale**: 
- Frozen backbone (16 GB) is read-only, doesn't generate gradients
- Wrapping it adds AllGather overhead on every forward without gradient benefit
- Trainable params (2.8 GB params + 5.5 GB grads + 16.6 GB AdamW state) shrink to 2.75 GB/rank
- Target memory after FSDP: ~43 GB/rank (fits comfortably in 97 GB H20)

### 2.2 Wrapping Granularity
**Granularity**: Per-`MemorySpaceLayer` (not per-model)

Rationale:
- `MemorySpaceLayer` wraps each `LlamaDecoderLayer` — one unit per decoder layer (32 total)
- FSDP "flattens" within each wrapped unit; multiple layers → per-layer sharding
- Matches standard transformer + FSDP practice (wrap at decoder-layer level)
- Avoids wrapping frozen parts (rotary embeddings, embedding tables)

### 2.3 FSDP Configuration

```python
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    ShardingStrategy,
    MixedPrecision,
    BackwardPrefetch,
)
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import checkpoint_wrapper

# MixedPrecision policy: compute bf16, reduce/buffer fp32
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    buffer_dtype=torch.float32,
)

# Wrapping strategy per layer
for layer in model.model.layers:
    mem_space_wrapper = layer.mem_space_wrapper  # or equivalent
    if mem_space_wrapper is not None:
        mem_space_wrapper = FSDP(
            mem_space_wrapper,
            sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
            cpu_offload=False,  # No CPU offload; GPU memory is sufficient once sharded
        )
```

**Settings Rationale**:
- **`FULL_SHARD`**: ZeRO-3, shards params + grads + optimizer state (needed for 22 GB savings)
- **`MixedPrecision(param=bf16, reduce=fp32, buffer=fp32)`**: Matches upstream Llama training
- **`BackwardPrefetch.BACKWARD_PRE`**: Prefetch next rank's params while computing current gradient (overlaps communication)
- **`cpu_offload=False`**: We have enough GPU memory; CPU offload kills throughput (5-20% slower than FSDP without offload)

---

## 3. FSDP + Gradient Checkpointing Interaction

### 3.1 Current Implementation
**File**: `src/memory/mem_space/layer.py`
- **Lines 475-481**: `_maybe_ckpt_wrapped_layer()` wraps the forward in `torch.utils.checkpoint`
  ```python
  def _ckpt_fn(h: torch.Tensor) -> Any:
      return self.wrapped_layer(h, **kwargs)
  return _ckpt.checkpoint(_ckpt_fn, hidden_states, use_reentrant=False)
  ```
  Uses `use_reentrant=False` (recompute activation on backward, no saved activations)

### 3.2 FSDP Compatibility Issue
- **Problem**: `torch.utils.checkpoint` + `use_reentrant=False` + FSDP `reshard_after_forward=True` can cause issues in older PyTorch versions (< 2.1)
  - Checkpoint recomputes without resharded params → numerical issues
- **Solution**: Replace manual checkpoint with FSDP-native `checkpoint_wrapper`

### 3.3 Recommended Fix: Use FSDP's Checkpoint Wrapper

Replace the existing `torch.utils.checkpoint` call in `_maybe_ckpt_wrapped_layer()` with FSDP-native activation checkpointing:

```python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    CheckpointImpl,
)

# Wrap each MemorySpaceLayer at construction time
mem_space_wrapper = checkpoint_wrapper(
    mem_space_wrapper,
    checkpoint_impl=CheckpointImpl.REENTRANT,  # or NO_REENTRANT
)
```

**Why this works**:
- FSDP's checkpoint wrapper integrates with `reshard_after_forward` and parameter gathering
- Guaranteed compatible with all PyTorch 2.x versions
- Recomputation happens *after* parameters are gathered (consistent state)

### 3.4 Migration Decision
**New behavior with FSDP**:
- When `--use_fsdp` is set AND `--gradient_checkpointing` is set:
  - Remove manual `torch.utils.checkpoint` from `_maybe_ckpt_wrapped_layer()`
  - Apply `checkpoint_wrapper` at the FSDP wrapping layer (in the new `_wrap_model_fsdp()` helper)
  - This ensures both use FSDP's native mechanism, avoiding conflicts

---

## 4. Frozen Backbone with FSDP

### 4.1 Design Constraint
- Llama-3-8B backbone (32 decoder layers): **16 GB frozen** (all params `requires_grad=False`)
- Only mem_space modules (L1 selector + slot bank + L2 compressor + L3 pool): ~1.4 GB trainable

### 4.2 Implementation Strategy
**Do NOT wrap frozen modules under FSDP.** Instead:

1. **Wrap only trainable mem_space modules** (Option b from §2.1)
2. Leave frozen backbone replicated across all ranks
3. Optimizer only tracks trainable params

**Code structure**:
```python
def _wrap_model_fsdp(model, mem_layers, use_checkpoint=False):
    """Wrap trainable mem_space layers in FSDP, leave backbone replicated."""
    # mem_layers = list of MemorySpaceLayer instances attached to model
    for mem_layer in mem_layers:
        # Wrap in FSDP
        mem_layer = FSDP(
            mem_layer,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            mixed_precision=mp_policy,
            backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
            device_id=local_rank,
        )
        
        # Optionally wrap in activation checkpoint (FSDP-native)
        if use_checkpoint:
            mem_layer = checkpoint_wrapper(mem_layer, checkpoint_impl=...)
        
        # Replace in-place
        # (Details depend on how mem_layers are stored; see patch.py)
    return model
```

---

## 5. State Dict Gathering for `_save_adapter`

### 5.1 Current Approach (DDP)
Uses `.state_dict()` directly on `model.module`. Works because DDP replicates everything; rank 0 has full state.

### 5.2 FSDP State Dict Challenge
FSDP shards parameters across ranks. `.state_dict()` on rank 0 returns only shard 0 (incomplete).

### 5.3 FSDP State Dict Solution
Use PyTorch's `FSDP.state_dict_type()` context manager to gather full state to rank 0:

```python
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, StateDictType

def _save_adapter_fsdp(model, args, step, final=False):
    """FSDP-aware adapter checkpoint save."""
    root = model.module if isinstance(model, FSDP) else model
    
    # Gather full state_dict on rank 0 (other ranks get empty dict)
    with FSDP.state_dict_type(
        root,
        state_dict_type=StateDictType.FULL_STATE_DICT,  # Gather
        rank0_only=True,  # Only rank 0 returns full state
    ):
        state = {
            k: v.detach().cpu()
            for k, v in root.state_dict().items()
            if any(frag in k for frag in fragments)
        }
    
    # Only rank 0 saves (avoids 8 parallel writes)
    if dist.get_rank() == 0:
        ckpt_path = os.path.join(args.output_dir, ...)
        torch.save(state, ckpt_path)
        logger.info("Saved adapter ckpt: %s", ckpt_path)
```

**Why `rank0_only=True`**:
- `StateDictType.FULL_STATE_DICT` with `rank0_only=True` gathers to rank 0 only
- All 8 ranks execute the code, but only rank 0's `state_dict()` contains the full state
- Other ranks get empty dict → skip the save
- Avoids 8 simultaneous writes to the same file

---

## 6. Migration Implementation Checklist

### 6.1 Code Changes (~80–150 lines total)

- [ ] **CLI flag**: Add `--use_fsdp` (off by default, DDP path remains as fallback)

- [ ] **Helper function** `_wrap_model_fsdp()` (~40 lines):
  - Wraps each trainable mem_space layer in FSDP
  - Optionally applies FSDP-native checkpoint wrapper
  - Returns FSDP-wrapped model

- [ ] **Training loop branching** (~20 lines):
  - If `--use_fsdp`: call `_wrap_model_fsdp()` instead of DDP wrapping
  - Use FSDP-style backward (same as DDP, but AllGather happens internally)
  - Remove DDP-specific logic (e.g., `model.module` unwrapping for optimizer)

- [ ] **Checkpoint save refactor** `_save_adapter()` (~30 lines):
  - Add `if args.use_fsdp:` branch with `FSDP.state_dict_type()` context manager
  - Keep DDP branch unchanged for backward compatibility

- [ ] **Gradient checkpointing refactor** (~15 lines):
  - When `--use_fsdp` AND `--gradient_checkpointing`:
    - Disable manual `torch.utils.checkpoint` in `_maybe_ckpt_wrapped_layer()`
    - Apply FSDP-native `checkpoint_wrapper` in `_wrap_model_fsdp()`

### 6.2 No Changes Required
- `_mem_space_params()` function — works unchanged (iterates `model._mem_space_layers`)
- `_freeze_backbone()` function — works unchanged (sets `requires_grad`)
- Training loop backward/step — works unchanged (FSDP handles sharding transparently)
- Optimizer construction — works unchanged (still targets trainable params only)

---

## 7. Risk Analysis & Mitigations

### 7.1 FSDP + HuggingFace Transformers + SDPA
**Risk**: FSDP with HF transformers using SDPA (scaled dot-product attention) can cause re-compilation issues on some PyTorch versions.

**Mitigation**:
- Set `--attn_impl eager` or `flash_attention_2` when testing FSDP
- SDPA is the default; if OOM, switch to eager
- Document in the plan that SDPA may need tuning per environment

### 7.2 Unused Parameters
**Current behavior**: DDP uses `find_unused_parameters=True` because:
- Frozen backbone params are not in the optimizer
- Some aux losses (key_repulsion, peak_routing) are conditionally added
- DDP needs the flag to handle partial gradient flow

**FSDP behavior**:
- FSDP does NOT have `find_unused_parameters` flag
- Automatically detects which params received gradients
- **No action needed** — FSDP will shard/gather only params with gradients

### 7.3 Activation Checkpointing Compatibility
**Risk**: `torch.utils.checkpoint(use_reentrant=False)` + FSDP `reshard_after_forward=True` in PyTorch 2.0 can cause numerical issues.

**Mitigation**:
- Use FSDP's native `checkpoint_wrapper` instead (guaranteed compatible)
- If manually checkpointing is needed, set `use_reentrant=True` or avoid FSDP's `reshard_after_forward`

### 7.4 Throughput Degradation
**Expected**: 5-15% slower than DDP due to AllGather communication overhead.

**Why it's acceptable**:
- Current DDP hit OOM fragmentation → cannot run at all
- FSDP may run at 85-90% of DDP throughput but WITHOUT OOM
- Net effect: **enables training that was impossible before**

---

## 8. Smoke Test Plan

### 8.1 Single-GPU Smoke (world_size=1)
```bash
torchrun --nproc_per_node=1 \
  scripts/train_mem_space_babilong.py \
    --model_path models/Llama-3-8B \
    --output_dir outputs/fsdp_smoke_1gpu \
    --use_fsdp \
    --total_steps 5 \
    --babilong_tasks qa1 --babilong_lengths 1k \
    --chunk_size 512 --max_seq_len 2048 \
    --use_l1 --use_l2 --use_l3
```

**Expected**: FSDP with world_size=1 should reduce to near-DDP behavior (no sharding communication). Verify:
- Training completes without OOM
- Loss values in expected range
- Checkpoint saves successfully

### 8.2 2-GPU Smoke (world_size=2)
```bash
torchrun --nproc_per_node=2 \
  scripts/train_mem_space_babilong.py \
    --model_path models/Llama-3-8B \
    --output_dir outputs/fsdp_smoke_2gpu \
    --use_fsdp \
    --total_steps 5 \
    --babilong_tasks qa1 --babilong_lengths 1k \
    --chunk_size 1024 --max_seq_len 4096 \
    --gradient_checkpointing \
    --use_l1 --use_l2 --use_l3
```

**Expected**: 
- Backward communication works (AllGather on backward)
- Gradient accumulation / backward/step proceed
- Memory per GPU should be ~50 GB (with checkpointing)
- Loss similar to 1-GPU run (within numerical tolerance)

### 8.3 8-GPU Full Launch (world_size=8)
```bash
torchrun --nproc_per_node=8 \
  scripts/train_mem_space_babilong.py \
    --model_path models/Llama-3-8B \
    --output_dir outputs/fsdp_full_8gpu \
    --use_fsdp \
    --total_steps 100 \
    --babilong_tasks qa1,qa2 --babilong_lengths 1k,2k \
    --chunk_size 1024 --max_seq_len 4096 \
    --batch_size 1 \
    --gradient_checkpointing \
    --use_l1 --use_l2 --use_l3 \
    --lr 1e-4 \
    --pg19_mix_fraction 0.2
```

**Expected**:
- Runs to completion without OOM
- Per-rank memory: ~43 GB (within H20 budget)
- Throughput: ~85-90% of equivalent DDP run
- Final checkpoint can be loaded for eval (test with `scripts/run_babilong_mem_space.py`)

---

## 9. Post-FSDP Memory Budget (Numerical Estimate)

### 9.1 Per-Rank Breakdown (Option b: Only Trainable Under FSDP)

| Component | Size | Sharded? | Per-Rank |
|-----------|------|----------|----------|
| Backbone Llama-3-8B (frozen, bf16) | 16 GB | No | 16 GB |
| Trainable params (L1/L2/L3, bf16) | 2.8 GB | Yes (÷8) | 0.35 GB |
| AdamW state (fp32 master + 2 moments) | 16.6 GB | Yes (÷8) | 2.1 GB |
| Gradients (fp32) | 5.5 GB | Yes (÷8) | 0.7 GB |
| FSDP AllGather temp buffers | ~2 GB | Transient | 2 GB |
| Activations (with checkpointing) per chunk | ~10 GB | Transient | 10 GB |
| 2 chunks BPTT (pipelining) | – | – | 20 GB |
| **Subtotal (model + opt)** | – | – | **19.15 GB** |
| **Add activations peak (2 chunks)** | – | – | **~41 GB** |
| cuBLAS workspace (typical) | – | – | +2 GB |
| **Total peak** | – | – | **~43 GB** |

### 9.2 H20 Headroom
- H20 per-GPU VRAM: 97.8 GB
- Peak memory requirement: 43 GB
- **Headroom**: 54.8 GB (56% free)
- **Safety margin**: ✅ Comfortable; 97 GB fragmentation issues should disappear

### 9.3 Comparison with Phase 11 DDP
| Metric | DDP (Phase 11, OOM) | FSDP Option (b) | Delta |
|--------|-------------------|-----------------|-------|
| Per-rank peak | ~110 GB (fragmented) | ~43 GB | -67 GB |
| AdamW state | 16.6 GB replicated | 2.1 GB sharded | -14.5 GB |
| Trainable params | 2.8 GB replicated | 0.35 GB sharded | -2.45 GB |
| Grads | 5.5 GB replicated | 0.7 GB sharded | -4.8 GB |

---

## 10. Known Limitations & Future Work

### 10.1 Activation Checkpointing Trade-Off
- FSDP-native `checkpoint_wrapper` recomputes activations on backward
- **2x compute cost** for wrapped layers
- **50% activation memory savings** (Phase 11: 20 GB activations → 10 GB)
- **Net effect**: Latency trade-off is worth the OOM fix (throughput still 85-90% of unchecked)

### 10.2 Potential Bottleneck: AllGather Communication
- Each backward pass triggers AllGather of sharded trainable params
- Effective bandwidth on H20 NVLink: ~500 GB/s (theoretical)
- Time to gather ~2.8 GB sharded params + grads: ~10 ms per backward
- Estimate: <5% training overhead if overlapped with compute (BackwardPrefetch.BACKWARD_PRE handles this)

### 10.3 Checkpoint Loading for Eval
- Eval script `scripts/run_babilong_mem_space.py` expects flat checkpoint state_dict
- Current save format (fragments only, not full model) **compatible** with FSDP (only mem_space weights, no backbone)
- ✅ No changes needed to eval loader

---

## 11. Implementation Order

1. **Add CLI flag** `--use_fsdp` (off by default)
2. **Implement `_wrap_model_fsdp()` helper** (~40 lines)
3. **Branch main() training path**:
   - If `--use_fsdp`: wrap model with FSDP
   - Else: wrap model with DDP (existing path)
4. **Refactor `_save_adapter()`** to handle both DDP and FSDP state dict gathering
5. **Update `_maybe_ckpt_wrapped_layer()`** to skip manual checkpoint when FSDP is active (checkpoint_wrapper handles it)
6. **Test on 1-GPU, 2-GPU, 8-GPU** in order
7. **Document in CODEBUDDY.md** the new `--use_fsdp` flag and when to use it

---

## 12. References

- **PyTorch FSDP Docs**: https://pytorch.org/docs/stable/fsdp.html
- **ZeRO Paper**: DeepSpeed ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (arXiv:1910.02054)
- **FSDP + Checkpoint Wrapper**: https://pytorch.org/docs/stable/fsdp.html#torch.distributed.algorithms._checkpoint.checkpoint_wrapper
- **Current Implementation**:
  - DDP wrap: `scripts/train_mem_space_babilong.py:709-710`
  - Gradient checkpoint: `src/memory/mem_space/layer.py:455-481`
  - State dict save: `scripts/train_mem_space_babilong.py:915-938`

---

## Summary

**Go/No-Go**: ✅ **GO**

FSDP migration is technically sound and directly addresses Phase 11's OOM issue. The sharded AdamW state (16.6 GB → 2.1 GB per rank) and grads (5.5 GB → 0.7 GB) alone save 19+ GB per rank. With activation checkpointing, peak memory per rank drops from ~110 GB (fragmented) to ~43 GB (well within H20's 97 GB budget).

**Estimated code effort**: 80–150 lines across 2–3 functions.  
**Estimated testing time**: 1 day (1-GPU smoke → 8-GPU full run).  
**Expected outcome**: Phase 11 (L1+L2+L3 cold-start) trainable on H20 without OOM.
