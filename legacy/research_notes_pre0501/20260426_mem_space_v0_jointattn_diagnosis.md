# Memory-Space v0 — joint-attn PPL=407 pathology: diagnosis

**Date**: 2026-04-26
**Author**: /researcher (main session)
**Related**:
- `ops/research_notes/20260426_memory_space_design_direction.md` (design spec)
- `src/memory/mem_space/layer.py` (primary suspect)
- `scripts/train_mem_space_pg19.py` (added `--bypass_memory` flag in this PR)

## TL;DR

- **Bypass parity PPL = 16.50** (vs vanilla Llama-3-8B baseline ≈ 6-8), **smoke PPL = 406.74**.
- Parity is within 2× of vanilla (acceptable — pg19 slice `skip_chunks=1000` is different
  from the production eval slice; chunk-level variance on N=10 is large). So the wrapping,
  patching, DDP, and `forward_no_memory` bypass are all clean.
- **Root cause is localised to the joint-attention forward path**, specifically the RoPE
  extension for slot tokens.  The `_extend_position_embeddings` helper in
  `src/memory/mem_space/layer.py:95-116` builds `cos_ext` / `sin_ext` by prepending `k`
  copies of the position-0 rotary entries **but it does so on the already-evaluated
  `position_embeddings` table produced by the outer `LlamaModel.rotary_emb(...)` call**.
  Because this table's RoPE phase is tied to `apply_rotary_pos_emb` via `cos.unsqueeze(1) *
  q`, the **extended sequence has slots at `q_idx = 0..k-1` receiving *H's own position-0
  RoPE*, and H-tokens at `q_idx = k..k+T-1` receiving RoPE for positions `0..T-1`** — i.e.
  H-tokens get the correct rotation, slots get a coherent "pos-0" rotation, but the **query
  indices themselves are shifted by `k` relative to the attention table**.  That shift is
  harmless **only if both `q` and `k` share the same shift**, which they do.  So RoPE is
  not the primary bug.
- **The primary bug is in the additive attention mask.**  `_build_extended_attn_mask`
  (layer.py:58-92) constructs the mask as a plain fp-dtype `[B,1,k+T,k+T]` additive tensor
  and passes it through to `LlamaDecoderLayer`.  But HF's SDPA backend
  (`transformers/integrations/sdpa_attention.py:77`) sets
  `is_causal = query.shape[2] > 1 and attention_mask is None and is_causal` — i.e. as soon
  as we pass **any** `attention_mask`, `is_causal=False` and the kernel's implicit causal
  mask is disabled.  That's fine for the H×H sub-block because we already installed a
  causal triangle there.  **However, with N=512, k=64, T=4096, the resulting mask is
  `[1, 1, 4160, 4160]` in bf16**, and the issue is that **the H-rows get a fully-causal
  H×H block AND unconditional visibility to every one of the `k=64` slot keys**.  Since
  slots were lazily initialised from `H.mean(dim=1) + N(0, 0.02)` **using H from this same
  chunk**, the slot keys carry a *linear* summary of the future → attending to them leaks
  future information.  This is exactly the "oracle slot" failure mode predicted in the
  design doc §2.3 R3, and it drives LM loss up because layer N uses residual-stream content
  that was already processed by layer N-1 with the same leak — the distortion compounds
  over 32 layers.
- **Recommended fix (V0 Stage-1)**: gate writeback strictly off (cold slots) AND initialise
  slots from **random** (`slot_init="random"`) rather than `"hidden_pool"` for the very
  first pass, so slot keys carry zero information about the current chunk.  Second fix,
  orthogonal: cap slot visibility to H-tokens whose position is `>= T//4` (i.e. ignore slots
  in the first quarter of the chunk) so early-context evidence can't leak via the selector
  pooling step.  Both are parameterisable through `MemorySpaceConfig` and require no
  rewrite of the forward.

## Evidence

### 1. Parity experiment (Task 1)

Added `--bypass_memory` flag to `scripts/train_mem_space_pg19.py`.  When enabled, it
monkey-patches every `MemorySpaceLayer.forward = forward_no_memory` so the joint-attn
path is entirely skipped and `LlamaDecoderLayer.forward` runs on the plain `[B, T, d]`
input.

Ran on b200-1 GPU 0, bf16 SDPA, identical knobs to the original smoke (N=512, top_k=64,
seq_len=4096, skip_chunks=1000, 10 chunks, train_steps=0 to remove optimiser-step
interference):

| Run | bypass_memory | train_steps | PPL | NAN chunks |
|---|---|---|---|---|
| smoke (original) | False | 10 | **406.74** | 0 |
| eval-only mem_path | False | 0 | **406.29** | 0 |
| parity (bypass) | True | 0 | **16.50** | 0 |

The tiny delta between smoke (407) and eval-only (406) says the 10 training steps make
**no material difference** — the selector / gate parameters are barely moved and the
pathology is present on the very first forward.  The 25× jump from 16 → 406 when we turn
the memory path back on localises the bug to code that runs only inside
`MemorySpaceLayer.forward` (not `forward_no_memory`).

**Artifacts**:
- `outputs/mem_space_v0_parity_llama3/eval_results.json` (PPL=16.50)
- `outputs/mem_space_v0_evalonly_llama3/eval_results.json` (PPL=406.29)
- `outputs/mem_space_v0_smoke_llama3/eval_results.json` (PPL=406.74, pre-existing)

### 2. RoPE extension audit (Task 2)

Read: `/opt/conda/envs/torch-base/lib/python3.11/site-packages/transformers/models/llama/modeling_llama.py`
(transformers 5.6.2, Llama-3-8B).

Key facts:
- `LlamaAttention.forward` takes `position_embeddings: tuple[Tensor, Tensor]` and *not*
  `position_ids` (the `position_ids` kwarg is passed into `LlamaDecoderLayer.forward`
  but never used by `LlamaAttention` in >= 4.45).  So our decision to pass `position_ids=None`
  and rely on `position_embeddings` is correct.
- `apply_rotary_pos_emb(q, k, cos, sin)` (line 146-168) does
  `cos = cos.unsqueeze(1)` → `[B, 1, seq_len, head_dim]` then
  `q_embed = q * cos + rotate_half(q) * sin` — **purely positional indexing**, cos/sin is
  indexed by `seq_len` axis position, NOT by `position_ids`.  So our `cos_ext` / `sin_ext`
  with shape `[B, k+T, head_dim]` gives:
  - slot tokens (q axis positions 0..k-1)  → get `cos[:, 0, :]` rotation (position 0,
    our intent)
  - H-tokens (q axis positions k..k+T-1)   → get `cos[:, 0..T-1, :]` rotation (the same
    rotation H would have gotten in a vanilla forward at positions 0..T-1) — but **note
    the q-axis index is now `k..k+T-1`, not `0..T-1`**.
- `rotate_half` is a sign-flip on the last dim halves and does not use absolute indices
  itself, so relative-position dot-products between H-tokens are preserved.  Slot-to-H
  dot-products give the same value they would give if the slot were at position 0 in a
  vanilla LM.  **RoPE extension is mechanically OK for V0.**

So Task 2's suspicion (`_extend_position_embeddings` being subtly wrong) is NOT the root
cause.  It is correctly replicating the "slot = position-0 memory token" semantics the
design doc requested.

Minor nit worth logging: the helper uses `cos[:, :1, :]` and `.expand(cos.shape[0], k, ...)`.
That's fine for `cos.shape[0] == 1` (the common case — Llama's `rotary_emb` returns `[1, T, D]`)
but if upstream ever returns per-batch cos (`[B, T, D]`) the `.expand` would still produce
a view that shares memory across slots.  `.contiguous()` is not called.  Not a correctness
issue because we only read the tensor, but flagged for Stage-2.

### 3. Mask shape / dtype audit (Task 3)

Read: `transformers/masking_utils.py`, `transformers/integrations/sdpa_attention.py`.

Findings:
- `_preprocess_mask_arguments` has an early-return at line 828: "if the mask is already
  4D, simply return as-is".  So our explicit `[B, 1, L, L]` bf16 additive mask flows
  through to `LlamaAttention → sdpa_attention_forward` unchanged.
- `sdpa_attention_forward` line 77:
  `is_causal = query.shape[2] > 1 and attention_mask is None and is_causal`.
  Once we pass any mask, `is_causal=False` — so SDPA uses our explicit mask and **does
  NOT apply an implicit causal triangle** on top.  Combined with our hand-built causal
  H×H block, the H rows are correctly causal.
- `use_gqa_in_sdpa` (line 28) requires `attention_mask is None`.  With our explicit mask,
  `use_gqa_in_sdpa → False`, so `key = repeat_kv(key, num_key_value_groups)` replicates K
  and V from 8 KV heads to 32 Q heads.  This is a **perf hit (about 4× KV memory and
  compute for the attention step), not a correctness issue**.  For the 10-chunk smoke it's
  tolerable; for the full run we should plumb through `is_causal=False` in the module
  rather than via the mask (Stage-2 work).
- Dtype: `torch.finfo(bf16).min ≈ -3.389e38`.  SDPA accepts bf16 masks natively.  No
  silent upcast.

**The mask is shape-/dtype-correct.**  The pathology doesn't come from "SDPA misreading
the mask"; it comes from **what we allowed through the mask**.

### 4. Primary bug — the slot-leak (new finding)

The design doc §2.3 R3 specifically calls out this hazard: slots initialised from
`H.mean(dim=1)` are a pooled summary of the chunk's tokens.  When we prepend those slots
as **unmasked keys** to the joint sequence, every H-row sees them — including H-rows at
position `t` that, under vanilla causal attention, should **not** see any information
derived from H-tokens at positions `t+1..T-1`.  Because slots encode the mean of
`H[0..T-1]`, each early H-row effectively attends to a summary of its own future.

Concrete check: `MemoryBank.init_from_hidden` at `src/memory/mem_space/memory_bank.py:139-146`:

```python
elif self.slot_init == "hidden_pool":
    pooled = H_l.detach().mean(dim=1, keepdim=True)          # [B, 1, d]
    slots = pooled.expand(B, N, d).contiguous().clone()
    if self.init_noise > 0.0:
        slots = slots + torch.randn_like(slots) * self.init_noise
```

With `slot_init_noise = 0.02` (config default) the slots are dominated by `pooled`.  The
top-k selector picks 64 of these 512 nearly-identical slots and prepends them as memory
tokens.  When we compare this to `forward_no_memory`, which never fabricates these oracle
tokens, we get the 25× PPL gap.

### 5. Supporting observation — selector/writeback innocence

- Selector picks `k=64` slot indices but with 512 near-identical slot vectors the actual
  *content* seen by attention is almost invariant to the choice of index — so the
  `straight-through estimator` and load-balance aux loss are not the issue.
- Writeback β is `σ(0) * 1.0 * 0.3 = 0.15` at step 0 (warmup_steps=0 in our smoke shell) but
  the bank is detached (`O_mem_slot.detach()` in layer.py:356) and eval uses `_reset_banks`
  between chunks, so between-chunk contamination is zero.  Within a chunk the bank state
  gets updated but the pathology is already present on the **first** forward before any
  writeback.  The eval-only run (406.29) vs smoke (406.74) confirms this.

## Proposed fixes

### Fix 1 — cold slots (minimal diff, addresses root cause)

Change the smoke config to use `slot_init="random"` with `slot_init_noise` set to a value
comparable to LayerNorm-output magnitudes (approx. 1.0 for Llama-3 post-rmsnorm).  This
eliminates the information leak at step 0 and lets the selector train from a clean
starting point.

Concretely, in `scripts/_run_mem_space_smoke_llama3.sh` add flags (and in
`train_mem_space_pg19.py` expose them):
```bash
--slot_init random \
--slot_init_noise 1.0 \
```

And in `MemorySpaceConfig`, verify `slot_init_noise = 1.0` with `"random"` doesn't
destabilise the selector softmax (it should be fine since `Q_sel @ K_sel^T / sqrt(128)`
re-scales).

Expected: PPL should drop into the 15-30 range (slightly worse than pure bypass because
attention to random slots is mildly noisy, but no longer oracle-leaking).

### Fix 2 — causal-compatible slot mask (addresses the architecture, not just init)

The "right" fix, per the design doc's intent (slots = compressed past), is that **slot j
should be visible only to H-tokens at position `t` with `t >= slot_j.born_at`** — i.e.
the mask should be causal between slots and H.  For V0 where slots are born-at-init and
updated EMA, the cleanest conservative choice is to **make every slot invisible to H
queries at positions `t < T/2`**, so at least the first half of the chunk cannot see the
pooled summary of its future:

In `_build_extended_attn_mask` (layer.py:58-92), after the causal H×H block, add:

```python
# Prevent "oracle leak": H queries in the first half of the chunk must not
# attend to slot keys (which were lazy-initialised from H.mean). Slots only
# become visible to H queries that have already seen ≥ T/2 tokens.
T_quarter = T // 2
mask[k:k + T_quarter, :k] = neg_inf
```

This is a 3-line change.  Expected: PPL should match or beat Fix 1, and the failure mode
is no longer oracle-driven.  (Stage-2 would replace this with a per-slot `born_at` tensor.)

### Fix 3 (recommended combined) — do both

Apply Fix 2 (architectural) and change default `slot_init` to `"random"` with
`slot_init_noise=1.0` (safety).  Run the smoke again with eval-only and with
train_steps=10; expect PPL in the 10-25 range.  If that holds, the joint-attn path is
operational and we can move to a real 200-chunk run.

## Next steps (recommended order)

1. Apply Fix 1 (smoke script flags only), re-run parity and eval-only, confirm PPL drops.
2. If PPL is still > 30, apply Fix 2 (layer.py mask edit).
3. Re-run the 10-step smoke with both fixes and confirm PPL in the healthy range.
4. Only then consider launching the 200-chunk full run.

## Files touched (this session)

- `scripts/train_mem_space_pg19.py` — added `--bypass_memory` flag
  (lines 232-236 for arg; lines 335-344 for monkey-patching loop).
- `scripts/_run_mem_space_parity_llama3.sh` — new parity driver (eval-only, bypass).
- `scripts/_run_mem_space_evalonly_llama3.sh` — new eval-only driver (mem path ON).

## Open questions

- **Q1**: Is `mask[k:k+T/2, :k] = -inf` (Fix 2) equivalent to what the design doc §2.3
  called "slot streaming"?  Worth a second pass with the author.
- **Q2**: Should `forward_no_memory` be wired into a `MemorySpaceConfig.bypass_memory`
  flag (proper config path) rather than monkey-patching?  Low priority, but cleaner for
  future ablations.
- **Q3**: GQA is silently disabled by our explicit mask.  Fix in V1 by moving the
  slot-mask into a module-level `is_causal=False` + custom mask function rather than a
  passed-through tensor.
