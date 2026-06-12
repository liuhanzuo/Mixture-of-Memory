# RMT++ v10 Code Review

**Date**: 2026-04-19
**Reviewer**: coder subagent
**Files reviewed**:
- `src/memory/rmt/rmt_v10.py`
- `scripts/train_rmt_v10.py`
- `scripts/run_train_v10.sh`
- Reference: `/tmp/rmt_lm.py`, `/tmp/rmt_base.py`, `/tmp/rmt_enc_mem.py`

---

## 1. Logic Correctness

### 1a. Sandwich attention mask — **PASS**
`build_sandwich_fast` correctly builds the mask:
- Old mem (rows 0:K): `causal[:, :K, K:] = True` makes them bidirectional (see all mem + content + placeholder) ✓
- Content (rows K:K+S): lower-triangular already gives causal, which includes seeing old_mem ✓
- Placeholder (rows K+S:K+S+K): causal, can see old_mem + all content + earlier placeholder ✓

### 1b. Memory extraction — **PASS**
`extract_new_memory` returns `hidden_states[:, -K:, :]` — from appended placeholder positions. Matches official RMT `MemoryCell.process_output()` which uses `[:, -self.num_mem_tokens:]` ✓

### 1c. Full BPTT — **PASS**
All segment logits/labels are collected, concatenated, and a single CE loss is computed over the full sequence. One `.backward()` call in training script. Memory state is NOT detached when `bptt_depth == -1` (default) ✓

### 1d. vary_n_segments — **PASS**
`torch.randint(1, max_segs + 1, (1,)).item()` randomly picks 1..max_segments. Only first N segments of input are used via `_segment_input` truncation ✓

---

## 2. Alignment with Official RMT

### 2a. Sandwich injection — **PASS**
Our `[old_mem | content | placeholder_mem]` matches official `MemoryCell.process_input()` which does `torch.cat([memory_state, inputs_embeds, memory_state], dim=1)` ✓

### 2b. Memory init scale — **PASS** (after fix)
`init_memory()` uses `embed_std` from `embedding.weight.data.std()`. Matches official `create_memory()` ✓

### 2c. BPTT gradient management — **PASS**
Our truncated BPTT logic (detach when `bptt_depth != -1` and segment is outside window) is semantically equivalent to official `manage_gradients()` ✓

---

## 3. Qwen3 Compatibility

### 3a. Attention mask format — **PASS**
`_make_4d_attn_mask` creates `[B, 1, T, T]` float mask with `-inf` for masked positions, passed as `{"full_attention": mask_4d}`. This is the correct format for Qwen3's custom attention implementation ✓

### 3b. position_ids — **WARNING**
We pass continuous 0-based IDs (`torch.arange(total_len)`), which means memory tokens get positions 0..K-1 and content is shifted by K. The official RMT doesn't modify positions either, so this is consistent. However, this means content in segment 2 starts at position 0 again (memory gets 0..K-1, content gets K..K+S-1). For RoPE-based models this is technically incorrect (each segment's content should have globally consistent positions), but the official RMT takes the same approach. **Low risk for now; revisit if quality is poor.**

### 3c. LoRA + DDP — **PASS**
Model: LoRA first → RMTv10Model wraps → DDP wraps. Order is correct. `find_unused_parameters=True` is appropriate since L1/L2 may be disabled ✓

---

## 4. Training Script

### 4a. collate_fn — **PASS** (double-shift bug is fixed)
Labels = input_ids (identity copy). Shifting happens in `RMTv10Model.forward()` via `shift_logits[..., :-1, :]` and `shift_labels[..., 1:]`. No double shift ✓

### 4b. Gradient accumulation — **PASS**
`no_sync` for non-last accumulation steps, `nullcontext` for last step. Correct ✓

### 4c. Optimizer parameter groups — **PASS**
Params with `l0`/`l1`/`l2`/`recon_head` in name get `rmt_lr`; others get backbone `lr` ✓

### 4d. heartbeat.json — **PASS**
Updated every `log_every` steps with loss, lr, progress. Final status set to "completed" ✓

---

## 5. L1/L2 Code Paths (disabled by default)

### 5a. Forward hook registration — **PASS** (after fix)
Hooks are now registered after `recon_head` and `init_memory` in `__init__` ✓

### 5b. Update logic — **PASS**
`should_update` checks `(seg_idx + 1) % update_freq == 0 and seg_idx > 0`. Correct — updates every N segments, not on first segment ✓

### 5c. get_injection broadcast — **PASS** (after fix)
`[1, num_tokens, D].expand(B, -1, -1)` → `[B, num_tokens, D]`. Hook now adds only to first `num_tokens` positions: `hidden[:, :num_toks, :] += injection` ✓

---

## 6. Runtime Issues

### 6a. CUDA OOM risk — **WARNING**
Full BPTT (`bptt_depth=-1`) retains computation graphs for ALL segments. With 6 segments × (2×16 + 1024) tokens, this is significant. For 8B model in bf16: ~6× the memory of single-segment forward. Consider `bptt_depth=3` if OOM occurs.

### 6b. DDP find_unused_parameters — **WARNING**
Slows down DDP by ~10-20% because it must scan for unused params every step. Low priority — acceptable for now.

### 6c. bfloat16 precision — **PASS**
Model in bf16, memory in bf16. No float32 accumulation issues — PyTorch handles bf16 gradients natively ✓

### 6d. Buffer vs parameter — **PASS**
L1/L2 use `register_buffer(..., persistent=False)` for dynamic state (not learnable). L0 uses `nn.Parameter` (learnable). Correct ✓

---

## Bugs Fixed During Review

### BUG-1 (CRITICAL): `recon_head` and `init_memory` inside `_register_layer_hook`
- **File**: `src/memory/rmt/rmt_v10.py`, lines ~417-424
- **Problem**: `self.recon_head = ReconstructionHead(...)` and `self.l0.init_memory(embed_std)` were inside `_register_layer_hook()`, which is only called when L1 or L2 is enabled. With default config (both OFF), these never executed → `AttributeError` on `self.recon_head` during forward, and memory weights left at random `torch.empty()` init.
- **Fix**: Moved `recon_head` creation and `init_memory` call to `__init__`, before the hook registration calls.

### BUG-2 (CRITICAL): L1/L2 hook dimension mismatch
- **File**: `src/memory/rmt/rmt_v10.py`, line ~411
- **Problem**: Hook did `hidden + injection` where `hidden` is `[B, T, D]` (T = sequence length) and `injection` is `[B, num_tokens, D]`. Would crash with size mismatch if L1/L2 enabled.
- **Fix**: Changed to `hidden[:, :num_toks, :] += injection` — adds only to first `num_tokens` positions.

---

## Known Issues (previously fixed, verified)

1. **collate_fn double-shift** — ✅ Verified: labels = input_ids (identity), shift happens once in forward
2. **L1/L2 injection not working** — ✅ Verified: hook approach is now correct with dimension fix

---

## Overall Assessment

**CAN TRAIN** ✅ — after the two critical bugs are fixed.

The two bugs would have caused an immediate crash on the first forward pass (BUG-1) or when L1/L2 is enabled (BUG-2). Both are now fixed.

Remaining warnings (position_ids, OOM risk, DDP overhead) are non-blocking and can be addressed later if needed.
