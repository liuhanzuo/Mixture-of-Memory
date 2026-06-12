# Fix B Design: Learnable `slot_keys` nn.Parameter for K_sel Routing Degeneracy

**Date:** 2026-04-28  
**Author:** researcher subagent  
**Purpose:** Concrete implementation plan for Fix B — adding separate learnable slot key parameters to break K_sel routing degeneracy. This note is the sole input to the `/coder` subagent; it must be self-contained.

---

## TL;DR

Fix B replaces the computed keys `K_sel(slots)` in `TopKSelector.forward` with a standalone `nn.Parameter slot_keys ∈ ℝ^{N × selector_dim}` that lives directly in `TopKSelector`. This decouples slot ADDRESSING (what key label identifies each slot) from slot CONTENT (what value the slot stores), eliminating the chain `K_sel → K_sel(slots) → key depends on slot values → key can only differentiate slots if values are already different`. With `slot_keys` as a directly-learnable parameter, gradient from the question-chunk LM loss flows straight to `slot_keys` without requiring slot values to be pre-differentiated. **Only `selector.py` requires code changes** (plus one docstring note in `layer.py`). Checkpoint loading already uses `strict=False` and needs no modification. The recommended implementation is Option A (replace, not hybrid): remove K_sel from the forward path, freeze it in-place for backward compat.

---

## 1. Code Reading Findings

### 1.1 MemoryBank (`src/memory/mem_space/memory_bank.py`)

```python
class MemoryBank(nn.Module):
    def __init__(self, num_slots, slot_dim, *, init_noise=0.02, slot_init="hidden_pool"):
        ...
        self.num_slots = num_slots     # N = 512 in current runs
        self.slot_dim  = slot_dim      # = d_model = 4096 (Llama-3-8B hidden_size)
        self.slots: Optional[torch.Tensor] = None   # NOT nn.Parameter
        self.frozen: bool = False
```

Key facts:
- **`slots` is NOT an `nn.Parameter`**. It is plain runtime state reset at every chunk boundary. It is NEVER saved to state_dict (intentionally).
- MemoryBank has **zero learnable parameters** currently.
- `slot_dim = d_model = 4096` for Llama-3-8B (because `config.slot_dim = None → use backbone hidden_size`).
- `init_noise` (default now `1.0` after Fix A) controls the gaussian noise added on `hidden_pool` init.
- `slots` is explicitly `.detach()`-ed on every `init_from_hidden` call and at chunk boundaries via `detach_()`.

### 1.2 TopKSelector (`src/memory/mem_space/selector.py`)

```python
class TopKSelector(nn.Module):
    def __init__(self, d_model, slot_dim, *, selector_dim=128, top_k=16, num_slots=128):
        ...
        self.Q_sel = nn.Linear(d_model, selector_dim, bias=False)   # query projection
        self.K_sel = nn.Linear(slot_dim, selector_dim, bias=False)  # key projection FROM slots
        nn.init.normal_(self.Q_sel.weight, std=0.02)
        nn.init.normal_(self.K_sel.weight, std=0.02)

    def forward(self, pool_of_H: Tensor, slots: Tensor) -> Tuple[...]:
        # pool_of_H: [B, d_model]
        # slots:     [B, N, slot_dim]   ← KEY DEPENDENCY
        q = F.normalize(self.Q_sel(pool_of_H), dim=-1)   # [B, S]
        k = F.normalize(self.K_sel(slots), dim=-1)        # [B, N, S]  ← Fix B replaces this
        temperature = 10.0
        logits = torch.einsum("bs,bns->bn", q, k) * temperature   # [B, N]
        scores = F.softmax(logits, dim=-1)
        _, idx = torch.topk(scores, k=self.top_k, dim=-1, ...)
        idx = idx.detach()
        ...
        ste_weights = scores + (one_hot_scores - scores).detach()
        return idx, scores, ste_weights
```

Critical observation (current code): `K_sel` projects from `slot_dim=4096` to `selector_dim=128`, then cosine-normalized (Fix C is already applied). The key depends on **slot values**, making it volatile and non-learnable in a meaningful sense.

### 1.3 MemorySpaceConfig (`src/memory/mem_space/config.py`)

Relevant defaults:
```python
selector_dim: int = 128
num_slots:    int = 128  # overridden to 512 in current runs via CLI
slot_dim:     Optional[int] = None  # → uses d_model = 4096
```

So in the current training run: `num_slots=512`, `selector_dim=128`, `slot_dim=4096`.

### 1.4 Call path: training loop → selector

**`scripts/train_mem_space_pg19.py`** → model forward → **`src/memory/mem_space/layer.py`** → `MemorySpaceLayer.forward`:

```python
# MemorySpaceLayer.forward() (layer.py lines 421-425):
slots = self.memory_bank.get()                         # [B, N, slot_dim]
pool = hidden_states.mean(dim=1)                       # [B, d_model]
idx, scores, ste_weights = self.selector(pool, slots)  # ← call to TopKSelector
```

`TopKSelector` does NOT hold a reference to `MemoryBank`. Slots are passed as a tensor argument. `self.selector` is a standalone `nn.Module` submodule of `MemorySpaceLayer`.

### 1.5 Trainable parameter collection (`train_mem_space_pg19.py` lines 137-169)

```python
def _mem_space_params(model):
    for wrapper in mem_layers:
        for p in wrapper.selector.parameters():   # ← automatically includes slot_keys if we add it to TopKSelector
            ...
        params.append(wrapper.gate_param)
        slot_gate = getattr(wrapper, "slot_output_gate", None)
        ...
        for p in wrapper.slot_to_hidden.parameters():
            ...
    return params
```

Key: `wrapper.selector.parameters()` is already being iterated. Any new `nn.Parameter` added to `TopKSelector` is automatically included in training without script changes.

### 1.6 Checkpoint save/load

**Save** (lines 735-748):
```python
_ADAPTER_KEY_FRAGS = (
    "selector", "gate_param", "slot_output_gate",
    "slot_to_hidden", "hidden_to_slot", "memory_bank",
)
_ckpt_state = {k: v.detach().cpu() for k, v in _root.state_dict().items()
               if any(frag in k for frag in _ADAPTER_KEY_FRAGS)}
```
`"selector"` is already in the filter → `selector.slot_keys` will automatically be saved.

**Load** (line 522):
```python
missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
```
`strict=False` is already used. New `slot_keys` will be in `missing` when loading an old checkpoint → initialised to random values (correct behavior).

---

## 2. Answers to Design Questions

### Q1: MemoryBank initialization and slot_dim

- `MemoryBank.__init__(num_slots=512, slot_dim=4096, *, init_noise=1.0, slot_init="hidden_pool")`
- `slots` is **NOT** an `nn.Parameter` — it is runtime state, reset every chunk, never in state_dict.
- `slot_dim = d_model = 4096` (hidden size of Llama-3-8B; config.slot_dim=None defaults to d_model).
- `selector_dim = 128` (from MemorySpaceConfig default, unchanged in current runs).
- MemoryBank currently has **zero trainable parameters**.

### Q2: How TopKSelector receives slot keys

TopKSelector does **NOT** hold a reference to MemoryBank. The full call path is:

```
train_mem_space_pg19.py:main()
  → (training loop) model(input_ids, labels)
  → LlamaForCausalLM.forward()
  → LlamaModel.forward()
  → MemorySpaceLayer.forward(hidden_states, ...)   [layer.py]
      slots = self.memory_bank.get()               # [B, 512, 4096]
      pool  = hidden_states.mean(dim=1)            # [B, 4096]
      idx, scores, ste_weights = self.selector(pool, slots)   # TopKSelector.__call__
  → TopKSelector.forward(pool_of_H=[B,4096], slots=[B,512,4096])
      q = F.normalize(Q_sel(pool_of_H), dim=-1)   # [B, 128]
      k = F.normalize(K_sel(slots), dim=-1)        # [B, 512, 128]  ← Fix B target
      logits = einsum(q, k) * 10.0                 # [B, 512]
      scores = softmax(logits, dim=-1)
      return idx, scores, ste_weights
```

`slots` is passed as a raw tensor (not gradient-bearing, because `memory_bank.get()` returns `self.slots` which is either `.detach()`-ed at init or detached via `detach_()`). Key insight: **slots do not carry gradients into the selector**.

### Q3: Option A (replace) vs Option B (hybrid)

**Recommendation: Option A (replace K_sel with slot_keys directly).**

**Option A** — `k = F.normalize(self.slot_keys.unsqueeze(0).expand(B, -1, -1), dim=-1)`:
- Slot keys become pure, standalone learnable parameters — no derivation from slot values
- Gradient path is clean: loss → logits → slot_keys directly
- K_sel layer becomes vestigial (unused in forward)
- No cross-contamination between key learning and value learning

**Option B** — `k = F.normalize(K_sel(slots) + slot_keys.unsqueeze(0).expand(B, -1, -1), dim=-1)`:
- Keeps K_sel in the forward path
- BUT: K_sel(slots) still has the original problem (slots are `.detach()`-ed, so K_sel gets no gradient from slot values in the forward pass)
- The gradient through K_sel from the question chunk goes to K_sel weights, not to slot values
- Net effect: K_sel still converges to near-zero contribution (same degenerate dynamic) while slot_keys works
- Hybrid adds complexity without benefit

**Why Option A is definitively better:**
1. No other code path uses K_sel (it's internal to TopKSelector forward)
2. Removing K_sel from the active forward removes ~67M params per layer × 32 = 2.1B params from the optimizer (saves Adam moment memory)
3. The diagnostic value of a "clean" fix is higher for ablation purposes
4. K_sel was a workaround; slot_keys is the architectural intent

**Action for K_sel**: Do NOT delete K_sel from `__init__` (for checkpoint backward compat — old ckpts have `selector.K_sel.weight`). Instead: **freeze K_sel** with `for p in self.K_sel.parameters(): p.requires_grad = False`. This keeps it in state_dict for compat but excludes it from optimizer (0 memory overhead for unused momentum).

### Q4: Gradient flow analysis for Fix B

With `slot_keys` as `nn.Parameter` in `TopKSelector`:

**During question chunk (gradient-bearing forward):**
```
LM loss (CE on answer tokens)
  ← logits of LlamaForCausalLM
  ← MemorySpaceLayer.forward outputs (next_hidden)
  ← slot_delta = ext_h[:, k:, :] - bypass_h, gated by tanh(alpha)
  ← ext_h from joint attention on [M_sel_hidden, hidden_states]
  ← M_sel_hidden = slot_to_hidden(M_sel_slot_w)
  ← M_sel_slot_w = M_sel_slot * w_gathered  (w_gathered from ste_weights)
  ← ste_weights: scores + (one_hot_scores - scores).detach()
     backward: d/d(scores) = 1 (straight-through)
  ← scores = softmax(logits * 10.0)
  ← logits = einsum(q, k) * 10.0
  ← k = F.normalize(slot_keys.unsqueeze(0).expand(B, -1, -1), dim=-1)
  ← slot_keys  ← GRADIENT ARRIVES HERE ✓
```

**Does haystack `torch.no_grad()` block the gradient?**

**No.** `slot_keys` is a standalone `nn.Parameter` — it does NOT depend on any computation inside the haystack forward. Its gradient comes ONLY from computation during the **question chunk** forward (the last chunk, which runs WITHOUT `torch.no_grad()`). The haystack chunks are:

```python
with torch.no_grad():
    for _c, _l_c in zip(chunks[:-1], label_chunks[:-1]):
        model(input_ids=_c_in, use_cache=False)
# slot_keys is USED in haystack forward too, but torch.no_grad() blocks ALL gradient accumulation
# However: slot_keys does NOT become a different tensor due to haystack — it IS the same Parameter
# The question chunk forward uses the SAME slot_keys, and its gradient is NOT blocked
```

Critical: `torch.no_grad()` in haystack chunks prevents gradient ACCUMULATION from those chunks, but:
1. `slot_keys` is not modified by haystack computation
2. `slot_keys` retains `requires_grad=True`
3. The question chunk runs without `no_grad` and fully differentiates through `slot_keys`
4. `optimizer.step()` uses only the question-chunk gradient

This is fundamentally different from `K_sel`: K_sel's gradient requires discriminative signal from the interaction between slot values (written in haystack) and query. `slot_keys` only needs to know "which label helps the question chunk LM loss" — a signal available entirely within the question chunk.

**Practical gradient magnitude:** The ste_weights straight-through estimator means `d(loss)/d(logits)` flows back through scores as if the soft scores were directly used (not the hard one-hot). This gives `slot_keys` a well-conditioned gradient signal proportional to how much each slot's selection helped the LM loss.

### Q5: Checkpoint backward compatibility

**Existing checkpoint keys** (from `_ADAPTER_KEY_FRAGS` filter, per layer):
```
model.model.layers.{i}.selector.Q_sel.weight
model.model.layers.{i}.selector.K_sel.weight
model.model.layers.{i}.gate_param
model.model.layers.{i}.slot_output_gate
model.model.layers.{i}.slot_to_hidden.weight
model.model.layers.{i}.hidden_to_slot.weight
```

**After Fix B, new keys:**
```
model.model.layers.{i}.selector.slot_keys    # NEW — will be in `missing`
model.model.layers.{i}.selector.K_sel.weight # STILL PRESENT (frozen, vestigial)
```

**Current checkpoint loading** (line 522 of `train_mem_space_pg19.py`):
```python
ckpt_state = torch.load(args.init_from, map_location=device)
missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
logger.info("init_from loaded: %d keys  missing=%d  unexpected=%d",
            len(ckpt_state), len(missing), len(unexpected))
```

Already uses `strict=False`. Loading an old checkpoint will show `missing=32` (one `slot_keys` per layer), which is expected and safe — PyTorch leaves missing parameters at their `__init__` values (the random init we set).

**No code change needed** in `train_mem_space_pg19.py` for checkpoint loading. The existing `strict=False` + log is sufficient.

**Optional enhancement** (not required but good practice): add an explicit log warning if `slot_keys` is missing from the checkpoint:
```python
slot_keys_missing = [k for k in missing if "slot_keys" in k]
if slot_keys_missing and is_main(rank):
    logger.warning("slot_keys not found in checkpoint (%d keys missing) — "
                   "will use random init for slot_keys. This is expected "
                   "when loading a pre-Fix-B checkpoint.", len(slot_keys_missing))
```

### Q6: Init scale for slot_keys

**Proposed init:** `torch.randn(N, selector_dim) * 0.1` → std=0.1 in ℝ^{128}.

**Analysis:**
- After `F.normalize(slot_keys, dim=-1)`, the SCALE of `slot_keys` is projected out — only DIRECTION matters for the normalized key.
- Expected L2 norm of a single key: `sqrt(selector_dim) × std = sqrt(128) × 0.1 ≈ 1.131`.
- The Jacobian of F.normalize(x) is `(I - x_hat x_hat^T) / ||x||`. With `||x|| ≈ 1.131`, gradient scaling ≈ `1/1.131 ≈ 0.88` — well-conditioned.
- For std=0.01: `||x|| ≈ 0.113` → gradient scaling ≈ 8.8 → too large, unstable.
- For std=1.0: `||x|| ≈ 11.3` → gradient scaling ≈ 0.088 → suppressed gradient at init.

**Direction diversity:** For N=512 keys in ℝ^{128}, random Gaussian vectors normalized to the unit sphere give maximum spread. Expected cosine similarity between any two random 128-dim unit vectors: mean 0, std ≈ 1/sqrt(128) ≈ 0.088. This means any two slot keys have ~11% typical pairwise similarity at init — excellent diversity from step 0.

**Recommended init:** `nn.init.normal_(self.slot_keys, std=0.1)` or equivalently `torch.randn(N, selector_dim) * 0.1`. The scale 0.1 is the right tradeoff.

**Do NOT use** `std = 0.02` (same as Q_sel/K_sel init): at std=0.02, `||slot_key|| ≈ 0.226`, Jacobian scaling ≈ 4.4 — gradient too amplified.

### Q7: Where should slot_keys live — TopKSelector or MemoryBank?

**Recommendation: `slot_keys` in `TopKSelector`.**

**Rationale:**
1. **State_dict registration**: `self.selector = TopKSelector(...)` in `MemorySpaceLayer.__init__` registers `selector` as a proper `nn.Module` submodule via standard `nn.Module.__setattr__`. Any `nn.Parameter` in TopKSelector automatically appears in `state_dict` and is tracked by autograd. 
   
   By contrast, in the `shared_memory_bank=True` case (the DEFAULT), the MemoryBank is registered via `object.__setattr__` (deliberately bypassing nn.Module registration) so it does NOT appear in `state_dict`. Adding `slot_keys` to MemoryBank would make them invisible to `state_dict`, checkpoint saving, and the optimizer — they would silently not be trained or saved.

2. **Per-layer semantics**: With `shared_memory_bank=True`, 32 layers share ONE MemoryBank (one set of slot values). But each layer's TopKSelector has DIFFERENT Q_sel weights (it addresses the shared bank differently). Having per-layer `slot_keys` in TopKSelector is consistent with this design: each layer uses the shared bank but addresses its slots with its own learned key labels.

3. **Checkpoint filter**: `_ADAPTER_KEY_FRAGS` already includes `"selector"` → `selector.slot_keys` is automatically saved. If `slot_keys` were in MemoryBank, the shared bank case would require a new filter fragment AND a new submodule registration.

4. **`_mem_space_params` collection**: already iterates `wrapper.selector.parameters()` — no script change needed.

5. **API**: `self.selector(pool, slots)` in `layer.py` unchanged. `slot_keys` is used internally by TopKSelector without callers needing to change.

**API design decision**: Keep `slots` as an argument to `TopKSelector.forward` even though it's no longer used for key computation. Reasons:
- No change needed in `layer.py`
- The shape validation of `slots` is still useful for catching bank-dimension bugs
- API stability: future hybrid implementations may re-add K_sel(slots)

---

## 3. Concrete Implementation Diff

### `src/memory/mem_space/selector.py` — FULL REPLACEMENT

Only `selector.py` needs changes. Here is the exact diff:

**`__init__` changes:**
```python
# BEFORE:
self.Q_sel = nn.Linear(d_model, selector_dim, bias=False)
self.K_sel = nn.Linear(slot_dim, selector_dim, bias=False)
nn.init.normal_(self.Q_sel.weight, std=0.02)
nn.init.normal_(self.K_sel.weight, std=0.02)

# AFTER:
self.Q_sel = nn.Linear(d_model, selector_dim, bias=False)
self.K_sel = nn.Linear(slot_dim, selector_dim, bias=False)  # kept but frozen (Fix-B: vestigial)
nn.init.normal_(self.Q_sel.weight, std=0.02)
nn.init.normal_(self.K_sel.weight, std=0.02)

# Fix B (2026-04-28): standalone learnable key parameters per slot.
# These are the "address labels" for each slot, updated by gradient from
# question-chunk LM loss. Crucially, they do NOT depend on slot values,
# breaking the K_sel(slots) chain that caused uniform-softmax degeneracy.
# std=0.1 chosen so ||slot_key||_2 ≈ sqrt(128)*0.1 ≈ 1.13 → F.normalize
# Jacobian scaling ≈ 0.88 (well-conditioned). Direction diversity at init:
# E[cos_sim(key_i, key_j)] ≈ 0 ± 0.088 for any two of the 512 keys.
self.slot_keys = nn.Parameter(torch.randn(num_slots, selector_dim) * 0.1)

# Freeze K_sel: it is no longer used in forward (Fix B replaced it with
# slot_keys). We keep the weight in state_dict for checkpoint backward
# compat but exclude it from optimizer momentum (saves ~2.1 GB GPU mem
# across 32 layers at bf16). Old checkpoints that have K_sel.weight will
# load fine (unexpected key, ignored by strict=False).
for p in self.K_sel.parameters():
    p.requires_grad = False
```

**`forward` changes:**
```python
# BEFORE (line 124):
k = F.normalize(self.K_sel(slots), dim=-1)      # [B, N, S], unit vectors

# AFTER:
# Fix B (2026-04-28): use standalone learnable key params instead of
# deriving keys from slot values. slot_keys: [N, S] → [B, N, S] via expand.
# The `slots` argument is kept for shape validation and API compatibility
# but is no longer used to compute keys.
k = F.normalize(
    self.slot_keys.unsqueeze(0).expand(B, -1, -1),
    dim=-1,
)                                                # [B, N, S], unit vectors
```

**Docstring update** (class-level docstring, add note):
```
    Fix B (2026-04-28): The ``slot_keys`` parameter replaces the
    ``K_sel(slots)`` key computation. Each slot has a direct learnable
    key vector (shape ``[N, selector_dim]``), decoupling slot addressing
    from slot content. ``K_sel`` is retained but frozen for checkpoint
    backward compatibility.
```

### Complete modified `forward` method

```python
def forward(
    self,
    pool_of_H: torch.Tensor,
    slots: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if pool_of_H.dim() != 2:
        raise ValueError(
            f"pool_of_H must be [B, d_model]; got {tuple(pool_of_H.shape)}"
        )
    if slots.dim() != 3:
        raise ValueError(
            f"slots must be [B, N, slot_dim]; got {tuple(slots.shape)}"
        )
    B, d_in = pool_of_H.shape
    if d_in != self.d_model:
        raise ValueError(f"pool_of_H last-dim {d_in} != d_model {self.d_model}")
    if slots.shape[0] != B:
        raise ValueError(
            f"batch mismatch: pool={B}, slots={slots.shape[0]}"
        )
    if slots.shape[1] != self.num_slots:
        raise ValueError(
            f"slots.shape[1] {slots.shape[1]} != num_slots {self.num_slots}"
        )

    q = F.normalize(self.Q_sel(pool_of_H), dim=-1)  # [B, S], unit vector

    # Fix B (2026-04-28): standalone learnable key params, not derived from slot values.
    # slot_keys: [N, S] → unsqueeze → [1, N, S] → expand → [B, N, S]
    # `slots` is kept as arg for validation and future hybrid use, but not for keys.
    k = F.normalize(
        self.slot_keys.unsqueeze(0).expand(B, -1, -1),
        dim=-1,
    )                                                # [B, N, S], unit vectors

    temperature = 10.0
    logits = torch.einsum("bs,bns->bn", q, k) * temperature  # cosine sim × T, [B, N]
    scores = F.softmax(logits, dim=-1)                # [B, N]

    # Hard top-k indices (no gradient through this op).
    _, idx = torch.topk(scores, k=self.top_k, dim=-1, largest=True, sorted=False)
    idx = idx.detach()                               # [B, top_k]

    # Build the one-hot mask: [B, N].
    one_hot = torch.zeros_like(scores).scatter_(
        dim=-1, index=idx, value=1.0
    )

    # Straight-through: forward = one_hot * scores, backward = scores.
    one_hot_scores = one_hot * scores
    ste_weights = scores + (one_hot_scores - scores).detach()  # [B, N]

    return idx, scores, ste_weights
```

### Files that need changes

| File | Change | Complexity |
|------|--------|------------|
| `src/memory/mem_space/selector.py` | Add `slot_keys` param; freeze K_sel; replace K_sel(slots) in forward | **LOW** — ~10 lines |
| `src/memory/mem_space/memory_bank.py` | **None** | — |
| `src/memory/mem_space/layer.py` | **None** (API unchanged) | — |
| `src/memory/mem_space/config.py` | **None** (optional: add `use_learnable_slot_keys` flag) | — |
| `scripts/train_mem_space_pg19.py` | **None** (strict=False already; selector params already collected) | — |
| `src/memory/mem_space/patch.py` | **None** | — |

---

## 4. Checkpoint Loading Fix

**No code change required.**

The current `train_mem_space_pg19.py` already uses:
```python
missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
logger.info("init_from loaded: %d keys  missing=%d  unexpected=%d",
            len(ckpt_state), len(missing), len(unexpected))
```

When loading a pre-Fix-B checkpoint:
- `selector.slot_keys` will appear in `missing` (32 entries, one per layer)
- PyTorch retains the `__init__` value for missing keys → `slot_keys` initialised to `randn * 0.1` ✓
- `selector.K_sel.weight` may appear in `unexpected` if old checkpoint has it → silently ignored ✓

**Optional enhancement** to add after the existing load block (not required):
```python
if args.init_from is not None:
    if is_main(rank):
        logger.info("Loading warm-start adapter from %s ...", args.init_from)
    ckpt_state = torch.load(args.init_from, map_location=device)
    missing, unexpected = model.load_state_dict(ckpt_state, strict=False)
    if is_main(rank):
        logger.info("init_from loaded: %d keys  missing=%d  unexpected=%d",
                    len(ckpt_state), len(missing), len(unexpected))
        # NEW: warn explicitly about slot_keys so it's visible in logs
        slot_keys_missing = [k for k in missing if "slot_keys" in k]
        if slot_keys_missing:
            logger.warning(
                "Fix-B: slot_keys missing from checkpoint (%d keys) — "
                "using random init. Expected when loading pre-Fix-B ckpt.",
                len(slot_keys_missing)
            )
```

---

## 5. Ablation Design: Milestones After Fix B

Run with the same config as the current swa+niah run (Fix A + Fix C already applied):
- `num_slots=512, top_k=64, selector_dim=128, swa_window=512, niah_mix_fraction=0.10`
- 8×GPU DDP on B200 node

### Milestone 1 — Symmetry Break (steps 0–200)

**What to check:** `top1_sim_mean` in QUERY_DIAG logs.

| Step | Expected with Fix B | Fix B failed if... |
|------|--------------------|--------------------|
| 0–50 | `top1_sim_mean ≈ 1/512 = 0.00195` (init is uniform; slot_keys freshly random) | — |
| 100 | `top1_sim_mean > 0.003` (first discrimination; slot_keys getting gradients) | Still 0.00195 |
| 200 | `top1_sim_mean > 0.005` (steady growth) | Growth plateaued at 0.002 |

**At step 50**, check the QUERY_DIAG log closely. Unlike Fix A+C without Fix B (which plateaued at 0.002060 and never moved), Fix B should show MONOTONIC INCREASE because slot_keys accumulate gradient directly.

### Milestone 2 — Meaningful Discrimination (steps 500–2000)

| Metric | Target | Failure threshold |
|--------|--------|------------------|
| `top1_sim_mean` | > 0.05 | < 0.01 at step 1000 |
| LM loss | < 0.90 (stable, not regressing) | > 1.2 (Fix B polluted LM) |
| `aux` (load balance) | Decreasing from ~20 toward < 10 | Stuck at 20+ (slots still uniform) |

The load-balance aux loss starts at ≈20 (N × top_k²/N = 64; 32 layers × 0.01 weight = 20.48) and should decrease as slot routing becomes non-uniform.

### Milestone 3 — NIAH Signal (steps 2000–5000)

After fixing the off-by-one metric bug (see `20260428_niah_acc_zero_diagnosis.md`) AND Fix B:
- `niah_acc > 0.05` by step 2000 (5% retrieval accuracy is non-trivial for N=512 slots)
- `niah_acc > 0.20` by step 5000

If `niah_acc` remains 0 after the metric fix + Fix B, the issue is either:
1. `slot_keys` are not actually getting meaningful gradient (check `slot_keys.grad.norm()` explicitly)
2. The slot CONTENT quality is insufficient (needle isn't written distinctively)
3. SWA window is too narrow to carry needle signal

### Diagnostic additions for Fix B

Add to QUERY_DIAG in `layer.py` (or as a separate FIXB_DIAG):
```python
if _should_log_diag:
    with torch.no_grad():
        # Distribution of slot key cosine similarities (should grow from near-0)
        _sk = F.normalize(self.selector.slot_keys, dim=-1)  # [N, S]
        _sim_mat = _sk @ _sk.T                               # [N, N]
        _off_diag = _sim_mat.masked_fill(torch.eye(N, dtype=torch.bool, device=_sim_mat.device), 0.0)
        _avg_cos_sim = _off_diag.abs().mean().item()
    print(f"[FIXB_DIAG step={self.step_counter}] avg_slot_key_cos_sim={_avg_cos_sim:.6f}", flush=True)
```

Expected: `avg_slot_key_cos_sim` starts at ~0.088 (random init) and should DECREASE (more orthogonal keys → better discrimination) or at least not increase (keys shouldn't collapse to identical).

---

## 6. Risk Assessment and Tradeoffs

| Aspect | Risk | Notes |
|--------|------|-------|
| Gradient flow correctness | **LOW** | Direct `nn.Parameter` in nn.Module; standard autograd applies |
| Checkpoint compat | **LOW** | strict=False already in place; missing slot_keys → random init is correct |
| Memory overhead | **MEDIUM** | 32 layers × 512 × 128 × 4 bytes = 8.4 MB params + ~16.8 MB Adam moments. Acceptable. |
| K_sel freeze | **LOW** | Frozen params have 0 gradient; no optimizer state allocated |
| LM parity risk | **LOW** | Fix B only changes which slots are selected, not how they're used |
| slot_keys collapse | **LOW-MEDIUM** | Add `avg_slot_key_cos_sim` diagnostic (see §5) to catch if keys collapse |

---

## 7. Summary of Code Changes (for `/coder`)

### File 1: `src/memory/mem_space/selector.py`

**Change 1: Add `slot_keys` and freeze `K_sel` in `__init__`** (after `nn.init.normal_` lines):
```python
# Fix B (2026-04-28): standalone learnable key params (replaces K_sel(slots)).
self.slot_keys = nn.Parameter(torch.randn(num_slots, selector_dim) * 0.1)

# Freeze K_sel — no longer used in forward. Kept in state_dict for backward
# compat with pre-Fix-B checkpoints. Requires_grad=False saves ~2.1 GB
# optimizer memory across 32 layers (no Adam moments allocated for 0-grad params).
for p in self.K_sel.parameters():
    p.requires_grad = False
```

**Change 2: Replace `K_sel(slots)` line in `forward`**:
```python
# REMOVE this line:
k = F.normalize(self.K_sel(slots), dim=-1)      # [B, N, S], unit vectors

# REPLACE with:
k = F.normalize(
    self.slot_keys.unsqueeze(0).expand(B, -1, -1),
    dim=-1,
)                                                # [B, N, S], unit vectors
# Note: `slots` arg kept for shape validation + API compat; not used for keys.
```

**Change 3: Update class docstring** to mention Fix B and slot_keys.

That is the entirety of Fix B. No other files need modification.

---

## 8. Related Notes

- `ops/research_notes/20260428_niah_key_degenerate_diagnosis.md` — root cause analysis, Fix B originally proposed here
- `ops/research_notes/20260428_niah_acc_zero_diagnosis.md` — separate off-by-one metric bug (must also be fixed to see non-zero niah_acc)
- `ops/research_notes/20260427_swa_memory_design.md` — SWA + NIAH training design
- `ops/research_notes/20260426_memory_space_design_direction.md` — original mem_space architecture spec
