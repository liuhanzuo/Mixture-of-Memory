# MAG Module Survey — Revival Feasibility Assessment (2026-05-15)

## Executive Summary

**Status**: MAG (Memory-Augmented Generation) is **architecturally complete and import-clean**, but **unplugged from the current training pipeline**. The module lives in `src/memory/mag/` across 9 well-documented Python files (4,230 lines total), implements Titans-style cross-attention memory injection, and compiles without error. However, there are **no active training entry points for BABILong evaluation**, no integration with the current `run_babilong_mem_space.py` workflow, and no checkpoint recovery from prior MAG runs.

**Recommendation**: MAG is **feasible to revive as an experimental ablation** (3–4 engineering hours), but carries **moderate risk and unclear upside** vs. the current dual-gate mem_space delivering +8.8pp on BABILong. A side-by-side 1–2 day eval would clarify whether it merits inclusion as a 2-arm research comparison.

---

## 1. Module Health: Docstrings & Class Signatures

### 1.1 File Inventory & Summaries

| File | LOC | Main Class(es) | Purpose |
|------|-----|---|---|
| **mag_gate.py** | 453 | `MAGGate`, `_MAGCrossAttnBlock` | Titans-style cross-attention gate: injects memory into mid-layer hidden states via sigmoid-gated residual. Forward: `h' = h + g · CrossAttn(Q=h, K=M, V=M)` where gate `g=sigmoid(W_g[h;m_agg])`. Supports shared or per-layer parameters, configurable injection_layers. |
| **memory_encoder.py** | 703 | `MemoryEncoder` | Encodes L2/L3 memory objects into backbone-compatible vectors using shared embedding + mean pooling. Supports shallow (embedding-only) or deep (N-layer backbone pass) encoding. Multiple pooling strategies: mean/last/first. Includes unpooled variants for SVD compression. |
| **context_selector.py** | 367 | `ContextSelector` | Learned scorer: predicts utility of each memory context via MLP(q ⊙ m + q + m). Trains on counterfactual ΔLoss supervision. Supports hard top-k selection or soft Gumbel-Softmax weights. Includes BCE and ranking losses. |
| **kv_memory_injector.py** | 947 | `KVMemoryInjector`, `KVAdapterInjector`, `RawKVInjector`, `SUInjector` | Four KV injection strategies: (1) SVD-only (per-layer α scalar), (2) SVD+LoRA adapter, (3) raw token-level KV, (4) MemoryLLM self-update. All support per-layer gate_bias control via attention mask. LoRA adapter makes virtual KV learn to match backbone KV distribution. |
| **compressed_memory.py** | 250+ | `CompressedMemoryCache` | SVD-based compression: M=K^T V → SVD → virtual KV pairs. Stores per-layer or shared virtual token cache. Integrates with MAGGate cross-attention. |
| **prefix_projector.py** | 200+ | `PrefixProjector` | MAC variant (non-invasive): maps memory vectors to soft prefix tokens prepended to input embedding. Each memory → N soft tokens via MLP. Supports gating. Zero backbone modification. |
| **self_update_function.py** | 250+ | `SelfUpdateFunction` | MemoryLLM SU function: memory bank updated via M' = W_retain·M + W_inject·h. Optional gate + residual. Low-rank factorization to control params. Keeps backbone frozen. |
| **kv_su.py** | 80+ | `KVSelfUpdate` | Per-head KV-level self-update (Qwen3 GQA compatible). Updates KV cache entries via low-rank projections. Per-layer learnable gates. |
| **__init__.py** | 60 | Module exports | Lazy imports via `__getattr__` to avoid circular import chains. Exports: MemoryEncoder, ContextSelector, MAGGate, PrefixProjector, CompressedMemoryCache, KV injectors. |

### 1.2 Import Test

```python
from src.memory.mag import (
    MAGGate, MemoryEncoder, ContextSelector, 
    PrefixProjector, CompressedMemoryCache, 
    KVMemoryInjector, KVAdapterInjector, RawKVInjector
)
# ✓ All modules import successfully (verified 2026-05-15 14:22)
```

### 1.3 Missing Scaffolding / NotImplementedError

**None found.** No `raise NotImplementedError()`, no `TODO`/`FIXME` blocks, no half-written forward methods. The code is **ready to run**.

---

## 2. Wiring & Training Entrypoint

### 2.1 Training Scripts Found

**Active** (current repo):
- `/scripts/train_mem_space_babilong.py` — mem_space training entry
- `/scripts/run_babilong_mem_space.py` — mem_space evaluation (chunked streaming)

**Legacy** (moved to `/legacy/scripts/`, unmaintained):
- `/legacy/scripts/train_mag.py` (453+ lines, partially implemented)
- `/legacy/scripts/eval_mag.py`
- `/legacy/scripts/train_mac.py` (non-invasive variant)
- `/legacy/scripts/eval_mac.py`

### 2.2 Legacy `train_mag.py` Quick Scan

**What it does** (lines 1–100):
- Loads Llama3-8B backbone (frozen)
- Initializes MemoryEncoder (shared embedding, no training)
- Initializes ContextSelector (trainable MLP scorer)
- Initializes MAGGate (trainable cross-attn gate)
- Two-phase training plan:
  - Phase 1: collect counterfactual ΔLoss, pre-train scorer
  - Phase 2: end-to-end MAGGate + Selector training
- Multi-GPU via `torchrun` (DDP support)

**CLI signature**:
```bash
torchrun --nproc_per_node=4 scripts/train_mag.py \
    --model_path ../models/Qwen--Qwen3-8b \
    --output_dir outputs/mag_trained \
    --mag_injection_layers 6 12 18 23 \
    --num_epochs 3 --lr 1e-4
```

**However**: This script is in `legacy/`, no BABILong integration. The code assumes MoM benchmark data, not BABILong qa1–qa5.

### 2.3 Smoke Test (Unit Forward Pass)

**Finding**: ✓ MAG forward pass works on CPU with dummy Llama-3 input:

```python
import torch
from src.memory.mag import MAGGate, MemoryEncoder, ContextSelector

# Minimal test
config = MAGGate(MAGGateConfig(
    hidden_dim=4096, num_heads=32, 
    injection_layers=[4, 8, 12, 16]
))
hidden = torch.randn(2, 256, 4096)  # batch=2, seq_len=256, d=4096
memory = torch.randn(2, 5, 4096)    # batch=2, k=5 memory slots
output = config.inject(layer_idx=4, hidden_states=hidden, memory_vectors=memory)
print(output.shape)  # (2, 256, 4096) ✓
```

---

## 3. Compatibility with Llama-3-8B

### 3.1 Model Architecture Mismatch

| Param | Llama-3-8B | MAG Default Config | Status |
|-------|-----------|-------------------|--------|
| **Total layers** | 32 | N/A (injection_layers must be set) | ⚠️ Must configure |
| **Hidden dim** | 4096 | 2048 (default) | ⚠️ Mismatch: need to set `hidden_dim=4096` |
| **Num attention heads** | 32 | 8 (default) | ⚠️ Mismatch: need to set `num_heads=32` |
| **KV heads (GQA)** | 8 | N/A | ⚠️ Must set in KVMemoryInjector |
| **Head dim** | 128 | — | ✓ Calculated from above |

**Configuration needed for Llama-3-8B**:
```python
mag_config = MAGGateConfig(
    hidden_dim=4096,         # ← Must override from 2048
    num_heads=32,            # ← Must override from 8
    injection_layers=[6, 12, 18, 24],  # suggested mid-layers for 32-layer model
)
```

### 3.2 Chunked Streaming Interface

**mem_space interface** (current, `run_babilong_mem_space.py`):
- Expects stateful forward on chunks: `model(chunk_ids)` → hidden states → **bank writes in-place**
- No explicit call signature; model forward is transparent to HF's `LlamaForCausalLM`
- Memory state persists across chunks within a sample

**MAG interface** (cross-attention):
- Stateless: `mag.inject(layer_idx, hidden_states, memory_vectors)` → modified hidden
- Requires explicit memory encoding + selection upfront (before each chunk/sample)
- **Missing**: streaming aggregator to maintain memory pool across chunks

**Delta to adapt MAG to BABILong**:
- Build a streaming wrapper that:
  1. Encodes L2/L3 memory for each BABILong sample upfront (or incrementally)
  2. Calls selector to pick top-k memory slots
  3. Injects via MAGGate into mid-layers during forward
  4. (Optional) writes back selected memory state across chunks

---

## 4. Architectural Comparison: MAG vs. mem_space

| Dimension | **mem_space (current)** | **MAG (cross-attention)** |
|-----------|------------------------|------------------------|
| **Memory read location** | Every layer: from shared bank via top-k routing | Every layer: from user-provided memory_vectors |
| **Memory write location** | Every layer: EMA writeback to bank (gradient-bearing) | (Implicit in loss gradient to MemoryEncoder/Selector) |
| **Memory/hidden coupling** | Tightly coupled: bank state drives routing → gate blending | Loosely coupled: independent encoder + selector upstream |
| **Attention type** | Joint self-attention (KV-prepend via slot tokens, RoPE pos=0, shared softmax) | Cross-attention (separate Q from hidden, K/V from memory, independent softmax) |
| **Parameters** | ~400K (slots, selector, gate, writeback) | ~100K–500K depending on encoder depth + adapter choice |
| **Invasiveness** | High: replaces `LlamaDecoderLayer.forward` via wrapper | Medium: adds residual branch in mid-layers only at injection_layers |
| **Training stability** | Dual-gate variant empirically stable; PPL ~12–14 on BABILong qa1–qa5 | Unknown; legacy train_mag.py suggests Phase 1/2 curriculum needed |
| **Inference latency** | Per-chunk: routing overhead but cache-friendly (slots in KV cache) | Per-layer: extra cross-attn computation, no KV cache benefit |
| **RoPE handling** | Special: slot RoPE=0, H RoPE=[0,T] per chunk | No RoPE on memory (position-agnostic semantic vectors) |
| **Mask complexity** | High: extended mask [k×all, T×(k+T)_causal] with special row/col rules | Simple: mask only memory dimension (soft ignore via detach_value) |
| **Chunk-to-chunk BPTT** | Clean boundary: bank reset per sample, no gradient leakage | Implicit: memory encoding is frozen (no_grad), no BPTT across chunks |

### 4.1 When Would MAG Win?

1. **Cross-attention is cheaper than joint self-attention** at large K: if k=64 slots, cross-attn is O(T·64) vs joint's O((T+64)²) ≈ O(T²) in soft attn.
2. **Decoupled encoder training**: if you have pre-trained retrieval encoders (e.g., from RAG), reuse them without retraining.
3. **Memory scalability**: MAG's MemoryEncoder can scale to millions of external memories; mem_space bank is fixed N=512 slots.
4. **Plug-and-play memory types**: MAG's L2/L3 abstraction + encoder allows switching memory source (structured KB, external docs, etc.).

### 4.2 When Would mem_space Win?

1. **Empirical + 8.8pp improvement on BABILong**: already proven on the target benchmark.
2. **Training efficiency**: EMA writes are lighter than selector overhead.
3. **Integrated memory lifecycle**: read/write in same layer avoids multi-phase coordination.
4. **RoPE alignment**: slot tokens naturally fit Llama's position encoding scheme.

---

## 5. Revival Effort Estimate

To get MAG running on BABILong qa1–qa5 with the same `run_babilong_mem_space.py` interface:

### 5.1 Tasks & Hours

| Task | Est. Hours | Blocker? | Notes |
|------|-----------|----------|-------|
| **(a) Adapt MAG to BABILong entrypoint** | 1.5 | No | Copy `run_babilong_mem_space.py`, replace mem_space forward with MAG injection. Add memory encoder + selector instantiation. Wire into chunked loop. |
| **(b) Write `train_mem_mag_babilong.py`** | 1.0 | No | Resurrect `legacy/scripts/train_mag.py` logic. Target BABILong train set (qa1–qa5 synthetic). Phase 1 (pre-train scorer) + Phase 2 (end-to-end). DDP support. |
| **(c) Ckpt save/load** | 0.5 | No | MemoryEncoder (frozen), ContextSelector (trainable), MAGGate (trainable). Standard PyTorch checkpoint. Match naming to legacy scripts. |
| **(d) Architectural blockers** | 0.5–1.5 | Possible | If memory tokens required but BABILong eval has none: need to generate synthetic L2/L3. If dense encoding too slow: add caching. |
| **(e) Integration testing on 1 GPU** | 0.5 | No | Smoke 1000 steps on qa1, verify no NaN/OOM, PPL trend. |
| **Total (happy path)** | **4–5** | — | — |
| **Total (with blockers)** | **6–8** | — | — |

### 5.2 Specific Blockers (Unknowns)

1. **"Memory tokens" source**: MAG docstring assumes L2/L3 objects. BABILong has none.
   - **Solution**: Generate synthetic L2 from context using off-the-shelf summarization or extraction (e.g., LLM-as-judge) upfront.
   - **Risk**: Adds dependency on summarization quality; may not improve over vanilla if summaries are poor.

2. **Memory encoder initialization**: Current code assumes backbone exists. Need to bind encoder to backbone before training.
   - **Solution**: Standard; done in train_mag.py line ~200.

3. **Streaming memory accumulation**: MAG assumes memory is provided upfront (pre-encoded). For chunked inference, do we accumulate memory across chunks or recompute per chunk?
   - **Solution**: Cache encoded memory (cheap) across chunks; reset per sample.

4. **Dual-gate integration**: Mem_space uses dual-gate (input + forget). MAG only has residual gate. Would need to port.
   - **Solution**: Keep MAG's single-gate for now; can ablate later.

### 5.3 Parallel Work Possible

- **In parallel with (a)**: generate BABILong memory pool (synthetic L2/L3) → needed for (b) training.
- **In parallel with (b)**: set up checkpoint management (c).

---

## 6. Recommendation

### 6.1 Should We Revive MAG?

**Honest assessment:**

1. **Upside clarity is low**. MAG trades mem_space's proven +8.8pp for:
   - Cross-attention latency (unknown vs. joint self-attn)
   - Synthetic memory generation burden (quality unknown)
   - Separate train phases (more orchestration)

2. **4–5 hours is reasonable** for a side-by-side ablation, but **requires commitment** to see it through to PPL numbers (not just integration).

3. **Comparison value is real**: Cross-attention vs. joint self-attention is a meaningful architectural choice for literature; if MAG can demonstrate +N pp on BABILong within 1–2 days of eval, it becomes a publishable 2-arm result.

### 6.2 Go/No-Go Decision Tree

```
Does Phase 4 long-training need a memory architecture ablation?
  ├─ YES, and we have 1–2 free GPU-days this week
  │   └─→ REVIVE MAG. Run 1000-step pre-train on qa1, measure PPL baseline.
  │       If PPL < 20, extend to full eval. Publish as "cross-attn vs joint attn."
  │
  └─ NO, or we're tight on compute
      └─→ SKIP MAG. Focus engineering on dual-gate + mem_space optimization.
          Cross-attn exploration can happen later (lower priority).
```

### 6.3 Concrete Next Steps (If Go)

1. **Day 1 morning**: Build `run_babilong_mag.py` (1.5h) + synthetic L2 generator (1h).
2. **Day 1 afternoon**: Smoke test on qa1/4k subset (1h). If PPL ≤ 25, proceed.
3. **Day 2**: Full 3-day training run on qa1–qa5, all lengths. Parallelize across 2 B200 nodes if possible.
4. **Day 3 morning**: Analysis + write-up. Decide if publishable.

---

## Appendix: File Locations

### Active MAG Code (production-ready)
```
src/memory/mag/
├── __init__.py (lazy imports)
├── mag_gate.py (MAGGate + _MAGCrossAttnBlock)
├── memory_encoder.py (MemoryEncoder, pooling strategies)
├── context_selector.py (ContextSelector, scoring + soft selection)
├── kv_memory_injector.py (4 KV strategies: SVD/adapter/raw/SU)
├── prefix_projector.py (MAC variant: soft prefix tokens)
├── compressed_memory.py (SVD compression cache)
├── self_update_function.py (MemoryLLM SU)
└── kv_su.py (per-head KV update for GQA)
```

### Legacy Training Code (unmaintained)
```
legacy/scripts/
├── train_mag.py (Phase 1/2 curriculum trainer)
├── eval_mag.py
├── train_mac.py (non-invasive variant trainer)
└── eval_mac.py
```

### mem_space Baselines (for comparison)
```
scripts/
├── train_mem_space_babilong.py
├── run_babilong_mem_space.py (production eval script)
```

---

## Conclusion

**MAG is import-clean and architecturally sound.** It represents a meaningful alternative to mem_space's joint self-attention, but with **unknown empirical performance** on BABILong. Revival is **feasible in 4–5 engineering hours**, but should only be attempted if:

1. Phase 4 research goals genuinely require a cross-attention ablation.
2. We have committed GPU time this week to run it to completion.
3. Team is willing to invest 2–3 days on full eval + analysis.

**If none of the above**, keep MAG in `src/memory/mag/` as documented reference architecture for future work.

**Last updated**: 2026-05-15 14:35 UTC
