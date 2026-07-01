# L2 Token-Compressed KV Memory — Implementation Plan (Phase 11)

**Date**: 2026-05-16  
**Status**: Research complete; ready for coder dispatch  
**Effort estimate**: 8–12 hours for working prototype  

---

## 1. Re-Validation Against Phase 8 Findings

### 1.1 Phase 8 Recipe Confirmation
Phase 8 (`babilong_sft_phase8_l1l3_lr2e5`) achieved **59.14 mean accuracy** via:
- **L1 (slot memory) + L3 (dense summary)** dual hierarchy on Llama-3-8B + BABILong
- Cold-start initialization (zero-init L1/L3 at step 0, no checkpoint warm-up)
- **lr=2e-5, total_steps=500** (500 steps ≈ 0.5 epochs over BABILong train split)
- Longer training (1000+ steps) over-fits to {1k, 2k, 4k} context ≈ ~3pp NIAH accuracy loss

### 1.2 L2 Initialization Strategy
**Recommendation: Cold-init L2 alongside L1+L3, not warm-init from P8.**

**Why:**
- Phase 8 proved that cold-start + short training generalizes better than warm-start
- Adding L2 (278M new params) mid-training risks overfitting to the {1k,2k,4k} BABILong distribution
- **Cleanest comparison**: L2 treated identically to L1/L3 — all three tiers initialize near-zero
- If L2 helps, it should help from step 0 (cold-init tells us if the *architecture* works)
- If L2 hurts, cold-init failure is cleaner to debug than warm-init interference

**Implementation**: Set `l2_compressor.kv_b.weight.std=0.001` (near-zero), matching the Flamingo gate pattern already used for `slot_output_gate`.

### 1.3 Parameter Budget & Trainable Count

**Phase 8 baseline**: ~1378M trainable params (frozen backbone + L1 memory bank + L3 cross-attn pool + selector/gate projections).

**L2 parameter addition** (per-layer, 32 layers total):

| Component | Per-layer | Total (32×) | Notes |
|---|---|---|---|
| `w_kv` (d × d_c) | 4096 × 512 = 2.10M | 67.2M | Content projection |
| `w_gate` (d × d_c) | 4096 × 512 = 2.10M | 67.2M | Gate scoring |
| `ape` (g × d_c) | 16 × 512 = 0.008M | 0.256M | Learned absolute-position bias |
| `norm` (d_c) | 0.512 | 0.016M | RMSNorm (negligible) |
| `kv_b` (d_c × 2n_h d_h) | 512 × 8192 = 4.19M | 134.1M | Up-projection to model space |
| `w_kR` (d × d_h_R) | 4096 × 64 = 0.262M | 8.4M | Decoupled-RoPE key |
| **Total** | **8.7M** | **277.2M** | ~20% of P8 baseline |

**New total trainable**: 1378M + 277M = **1655M** (±20% overhead).

**Stage-2 optimization** (future): drop `kv_b` and reuse the wrapped layer's existing K/V projections → per-layer cost drops to 4.5M, total ~144M (comparable to LoRA-r=64).

### 1.4 Training Schedule for L2
**Proposal: Identical to P8**: lr=2e-5, total_steps=500, no warmup.

**Rationale:**
- L2 is a **sidecar module** (does not modify backbone or L1/L3 directly)
- If cold-init works for L1/L3, it should work for L2
- 500 steps is the empirical sweet spot (longer = overfitting to BABILong distribution)
- Same learning rate; if L2 shows gradient instability, can sweep {1e-5, 2e-5, 5e-5} in Phase 12

---

## 2. Code Validation Against Actual Wiring

### 2.1 MemorySpaceLayer.forward Insertion Point ✓
**File**: `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/src/memory/mem_space/layer.py`

**Current wiring** (lines 600–632):
```python
# Extended sequence layout: [L3(k_l3) | L1(k) | H(T)]
k_l3 = 0
if l3_summaries is not None:
    k_l3 = l3_summaries.shape[1]
    if k_slots > 0:
        extended_hidden = torch.cat([l3_summaries, M_sel_hidden, hidden_states], dim=1)
    else:
        extended_hidden = torch.cat([l3_summaries, hidden_states], dim=1)
else:
    if k_slots > 0:
        extended_hidden = torch.cat([M_sel_hidden, hidden_states], dim=1)
    else:
        extended_hidden = hidden_states
```

**Insertion sketch** (new extended sequence will be `[L3 | L2 | L1 | H]`):
- **Before** `extended_hidden = torch.cat(...)`, add L2 path:
  - Read `self.l2.prev_latents` (from previous chunk)
  - Up-project via `self.l2.kv_b` (content) + reconstruct K/V
  - Prepend L2 tokens between L3 and L1
- **After** the wrapped layer's forward, compute new L2 latents:
  - Call `self.l2.compress(next_hidden.detach())` on post-layer hidden states
  - Store in `self.l2.prev_latents` for next chunk

**Status**: ✓ Code structure confirmed. Insertion at lines 615–617 is correct.

### 2.2 Config Fields Required ✓
**File**: `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/src/memory/mem_space/config.py`

**Fields to add** (after line 152 `disable_l1_inject`):
```python
use_l2: bool = False                # Enable L2 token compression
l2_compress_ratio: int = 16        # g: tokens → latents ratio
l2_d_c: int = 512                  # Latent dimension (matches V2 MLA)
l2_d_h_rope: int = 64              # Decoupled-RoPE dimension per latent
l2_chunk_size: int = 4096          # (Derived from outer chunk_size; kept for clarity)
l2_init_scale: float = 0.001       # kv_b weight initialization std (near-zero)
```

**Backward compat**: All new fields default to disabled/sensible values.

### 2.3 Patch.py Wiring ✓
**File**: `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/src/memory/mem_space/patch.py`

**Changes**:
1. **Line ~70–90**: After `MemorySpaceLayer(...)` instantiation, add:
   ```python
   if config.use_l2:
       l2_comp = L2Compressor(
           d_model=model.config.hidden_size,
           n_heads=model.config.num_attention_heads,
           d_head=model.config.hidden_size // model.config.num_attention_heads,
           compress_ratio=config.l2_compress_ratio,
           d_c=config.l2_d_c,
           d_h_rope=config.l2_d_h_rope,
           chunk_size=config.l2_chunk_size,
       )
       mem_layer.l2 = l2_comp
   ```
2. **New function** `_reset_l2()` (parallel to `_reset_banks()`):
   ```python
   def _reset_l2(model: nn.Module) -> None:
       for layer in model._mem_space_layers:
           if hasattr(layer, "l2") and layer.l2 is not None:
               layer.l2.reset()
   ```
3. **Export** `_reset_l2` in `__all__`.

### 2.4 Training Script ✓
**File**: `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/scripts/train_mem_space_babilong.py`

**Changes**:
1. **Line ~395** (after `--disable_l1_inject`), add CLI args:
   ```python
   p.add_argument("--use_l2", action="store_true", default=False)
   p.add_argument("--l2_compress_ratio", type=int, default=16)
   p.add_argument("--l2_d_c", type=int, default=512)
   p.add_argument("--l2_d_h_rope", type=int, default=64)
   p.add_argument("--l2_init_scale", type=float, default=0.001)
   ```
2. **Line ~502** (in MemorySpaceConfig merge), add:
   ```python
   use_l2=args.use_l2,
   l2_compress_ratio=args.l2_compress_ratio,
   l2_d_c=args.l2_d_c,
   l2_d_h_rope=args.l2_d_h_rope,
   ```
3. **Line ~934** (in checkpoint save dict), add the new fields to `adapter_config.json`.

### 2.5 Eval Script ✓
**File**: `/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/scripts/run_babilong_mem_space.py`

**Changes**:
- Import `_reset_l2` from `src.memory.mem_space.patch`
- Call `_reset_l2(model)` between BABILong tasks (alongside existing `_reset_banks(model)`)
- No other changes needed — L2 is transparent to the eval loop (chunk boundary logic unchanged)

---

## 3. Concrete TODO Checklist

| File | Task | Lines | Est. time |
|------|------|-------|-----------|
| `src/memory/mem_space/l2_compressor.py` | **NEW**: L2Compressor class (~200 lines, mostly from doc §5) | — | 1.5h |
| `src/memory/mem_space/config.py` | Add L2 config fields | After line 152 | 0.25h |
| `src/memory/mem_space/layer.py` | Modify forward: read L2, prepend, write L2 | Lines 600–620 (insert before `extended_hidden = ...`) | 2h |
| `src/memory/mem_space/layer.py` | Add mask helper for L2 tokens | After line 632 (new fn `_build_extended_attn_mask_l2`) | 0.5h |
| `src/memory/mem_space/patch.py` | Instantiate L2 per layer + reset hook | Lines 70–90 (in `apply_mem_space_to_model`) | 1h |
| `scripts/train_mem_space_babilong.py` | Add CLI args + merge into config | Lines ~395, ~502 | 0.5h |
| `scripts/train_mem_space_babilong.py` | Save L2 config to adapter_config.json | Line ~934 | 0.25h |
| `scripts/run_babilong_mem_space.py` | Call `_reset_l2()` at doc boundary | Around line 200–250 (find reset_banks call) | 0.5h |
| — | **Unit tests** + smoke test (single chunk, L2 bypass) | — | 1.5h |
| — | Two-chunk validation (latents flow, no NaN, gradient backprop) | — | 1h |

**Total**: 8.5 hours (matches estimate from L2 research doc).

---

## 4. Risk Analysis

### 4.1 Most Likely Failure Modes

| Failure mode | Symptom | Root cause | Mitigation |
|---|---|---|---|
| **Parameter scale crash** | NaN after 5–10 steps | Large `kv_b` weight norm overwhelming attention; gate softmax overflow | Start with `l2_init_scale=0.001` (near-zero); monitor `L2_tokens.norm()` at step 0–50 |
| **Gradient explosion** | L2 weight gradients → ∞; loss NaN by step 20 | BPTT through 4 layers of compression + up-proj + compressed attention creates exploding gradients; `detach()` logic wrong | Check `grad_clip=1.0` applied to L2 weights; use `new_latents.detach()` correctly (no backprop through latent compression across chunks) |
| **RoPE position confusion** | L2 tokens fuse with L1 in attention; retrieval degrades | L2 latents all at RoPE position 0 → indistinguishable from L1 slots in attention; model "forgets" which is which | v0 uses position 0 for all L2 (acceptable for ablation); Stage-2 upgrades to per-window-centroid positions (§4.5 option 3 in research doc) |
| **Mask mistake** | L2 tokens can attend to future H tokens (causal break) | Mask builder forgets to include L2 rows/cols in the extended-sequence causal structure | Verify mask shape = (1, 1, k_l3+k_l2+k_l1+T, k_l3+k_l2+k_l1+T); unit test on toy mask |
| **BPTT broken across chunks** | Gradient stops flowing to L1/L3 when L2 added | Detach logic in the `next_hidden` path incorrect; L2 reads `prev_latents` but that path shares state with L1 bank | Ensure `new_latents = self.l2.compress(next_hidden.detach())` uses post-layer H; L1 slot writes still use `O_mem_slot.detach()` (no change) |
| **Double-projection mess** | Attention output differs 10% from vanilla per-head | L2 tokens → `kv_b` → K/V → wrapped layer's K/V proj again. Wasteful + gradient routing unclear | v0 accepts this. Stage-2: inject L2 K/V directly into attention (no double-proj) by directly hooking wrapped layer's attention module |

### 4.2 L3 Backward-Through-Graph Issue — Does L2 Face It?

**L3 case** (2026-05-15): L3 summary tokens are computed from post-layer hidden states, but gradients need to flow *backward* through the layer to improve the summary pool weights. Prior fix: stash-detached-H + recompute-in-current-chunk.

**L2 case**: L2 latents are **stored at chunk boundary** and **read in the *next* chunk's attention**. Gradient flow is:
- Forward: next chunk's loss → next chunk's attention → L2 tokens → `kv_b` weight gradient ✓ (normal backprop)
- Cross-chunk: does NOT backprop through the `prev_latents` buffer (we call `.detach()` on `next_hidden` before compression)
- Inter-chunk coupling: minimal (only through the forward path; no BPTT through L2 compression itself)

**Verdict**: L2 does **not** face the same issue. The design is intentionally per-chunk-local:
- Chunk i writes compressed latents
- Chunk i+1 reads them but doesn't backprop through the compression
- Each chunk's L2 compression is trained only on the loss of *that chunk* (via the wrapped layer's gradients flowing to `w_kv`, `w_gate`, `kv_b`)

**Clean design** — matches long-context Transformer practices (e.g., Infini-Attention, Landmark Attention).

### 4.3 Double-Projection Issue — Cleaner Approach?

**Current v0 design**: L2 latents → `kv_b` (L2-specific up-proj) → "pseudo-tokens" → wrapped layer's K/V projections (standard).

**Concern**: Wasteful; the wrapped layer projects pseudo-tokens to K/V, then attention uses the result. Gradient routing: loss → wrapped-layer-K/V-grad → pseudo-tokens → kv_b-grad. Works, but indirect.

**Cleaner Stage-2 alternative**:
- Compute K, V **directly** from L2 latents (bypass the wrapped layer's projections)
- Inject them into attention's KV cache *before* the wrapped layer runs
- Requires hooking `LlamaAttention.forward` to prepend L2 K, V to its cache

**v0 verdict**: Stick with double-projection. Simpler to implement, easier to debug (standard forward pass). Stage-2 (post-eval) can optimize.

---

## 5. Phase 11 Launch Command Template

```bash
# Cold-start L2 + L1 + L3 with lr=2e-5, 500 steps
torchrun --nproc_per_node=8 --nnodes=1 --rdzv_backend=c10d \
    scripts/train_mem_space_babilong.py \
    --model_path models/meta-llama/Llama-3-8B-Instruct \
    --output_dir outputs/babilong_sft_phase11_l1l2l3_lr2e5 \
    --init_checkpoint outputs/babilong_sft_phase8_l1l3_lr2e5/mem_space_adapter.pt \
    --init_adapter_config outputs/babilong_sft_phase8_l1l3_lr2e5/adapter_config.json \
    --babilong_tasks qa1,qa2,qa3,qa4,qa5 \
    --babilong_lengths 1k,2k,4k,8k,16k \
    --total_steps 500 \
    --lr 2e-5 \
    --chunk_size 4096 \
    --batch_size 1 \
    --num_slots 512 \
    --top_k 64 \
    --use_l3_summary \
    --l3_n_summary 64 \
    --use_l2 \
    --l2_compress_ratio 16 \
    --l2_d_c 512 \
    --l2_d_h_rope 64 \
    --l2_init_scale 0.001
```

**Eval after Phase 11**:
```bash
python scripts/run_babilong_mem_space.py \
    --model_path models/meta-llama/Llama-3-8B-Instruct \
    --checkpoint outputs/babilong_sft_phase11_l1l2l3_lr2e5/mem_space_adapter.pt \
    --adapter_config outputs/babilong_sft_phase11_l1l2l3_lr2e5/adapter_config.json \
    --output_name phase11_l1l2l3_eval \
    --tasks qa1 qa2 qa3 qa4 qa5 \
    --lengths 0k 1k 2k 4k 8k 16k 32k
```

---

## 6. Phase 12 Follow-Up Plans

### 6.1 If Phase 11 Shows Improvement (+2–5pp NIAH accuracy)
1. **L2 learning rate sweep** (4-arm ablation):
   - `lr ∈ {1e-5, 2e-5, 5e-5, 1e-4}`, same 500 steps
   - Goal: find L2's optimal LR (may differ from L1+L3 regime)
2. **L2 compression ratio sweep**:
   - `g ∈ {8, 16, 32}` (8 = 512 latents/chunk, 32 = 128 latents/chunk)
   - Tradeoff: quality vs KV cache size
3. **Per-window-centroid RoPE positions** (Stage-2 upgrade):
   - Implement §4.5 option 3 from L2 research doc
   - Test if learnable per-position attention helps disambiguation
4. **L1 + L2 + L3 ablation matrix**:
   - `{L1 alone, L2 alone, L1+L2, L1+L3, L2+L3, L1+L2+L3}`
   - Map synergies: does L2 better complement L1 or L3?

### 6.2 If Phase 11 Shows No Improvement or Degradation
1. **Diagnostic**: check L2 latent statistics at step 0–50:
   - Latent norm distribution (should grow from ~0.001 to ~0.1–1.0 by step 50)
   - Gate softmax entropy (should not collapse to one token per window)
   - Gradient flow to `w_kv`, `w_gate`, `kv_b` (should be non-zero by step 10)
2. **If latents are near-zero**: cold-init too aggressive; try `l2_init_scale=0.01` or `0.1`
3. **If gate is collapsing**: add entropy regularization on the gate softmax
4. **If gradient is zero**: check detach logic; run toy-chunk test with 1 example

### 6.3 Direct Injection Stage-2 (if v0 works)
- Implement the "cleaner approach" from §4.3
- Inject L2 K, V directly into `LlamaAttention`'s KV cache (no double-projection)
- Expect ~5–10% speedup during eval (fewer linear layers per chunk)

---

## 7. Effort Breakdown & Timeline

| Phase | Effort | Elapsed | Dependencies |
|---|---|---|---|
| **Coder dispatch** (implement 8 tasks above) | 8–10h | Day 1 | None |
| **Smoke test** (single chunk, no L2) | 0.5h | Day 1 | Coder impl complete |
| **Two-chunk test** (L2 read/write, gradient flow) | 1h | Day 1 | Smoke pass |
| **BABILong eval Phase 11** (500 steps, 1 B200 node) | 2–3h (wall time) | Day 1–2 | Two-chunk test pass |
| **Analysis + Phase 12 planning** | 1–2h | Day 2 | Phase 11 complete |
| **Phase 12a launch** (if improving) | 4–6h (wall time) | Day 2–3 | Phase 11 analysis |

**Gate for Phase 12**: If Phase 11 NIAH accuracy ≥ 56 (no significant degradation) → proceed with ablations.

---

## 8. References

- **L2 Design Doc**: `docs/L2_DEEPSEEK_MLA_RESEARCH.md` (§4–5 concrete parameters, §5 code sketch)
- **Phase 8 Results**: `outputs/babilong_sft_phase8_l1l3_lr2e5/` (checkpoint + metrics)
- **Baseline MemorySpaceLayer**: `src/memory/mem_space/layer.py:560–770` (current joint-attn forward)
- **L3 Integration Reference**: `src/memory/mem_space/layer.py:600–632` (L3 prepending pattern)

---

## Summary

L2 token-compressed KV memory is **architecturally validated** (per DeepSeek-V4), **code-ready** (insertion points confirmed), and **strategically sound** (cold-init aligns with Phase 8 success). The 8–12 hour implementation timeline is firm; follow-up ablations (Phase 12) depend on Phase 11 eval results. Recommend **starting coder dispatch immediately** to prototype and test against BABILong by end of day.
