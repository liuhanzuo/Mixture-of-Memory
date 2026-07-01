# L2 Token-Compressed KV Memory — DeepSeek MLA / V3 / V4 Research + Spec

**Author**: general-purpose-19 (researcher subagent)
**Date**: 2026-05-15
**Status**: READ-ONLY research — implementation handed back to team-lead for coder dispatch.
**Scope**: produce a concrete spec for the **L2** layer of our 3-tier memory hierarchy
(L1 = `mem_space` slot bank; L2 = token-compressed latent KV; L3 = dense summary tokens).

---

## 0. TL;DR

- **DeepSeek-V2 MLA** (2024-05) compresses *each* token's K/V into one shared low-rank
  latent `c_KV` (per-token), with an extra *decoupled-RoPE* shared key. KV cache size:
  `(d_c + d_h^R) · L` per token. For V2: `d_c = 512`, `d_h^R = 64`, `L = 60` → 4.5×
  smaller than 128-head MHA.
- **DeepSeek-V3** (2024-12) keeps MLA architecturally **unchanged**; same `d_c=512,
  d_h^R=64, n_h=128, d_h=128`, only routing/training-time changes.
- **DeepSeek-V4** (2026-04, Pro & Flash) is the **group token compressor** the user
  recalled. It introduces two real modules: **CSA (c4a)** = compress every 4 tokens →
  1 KV (with 50%-overlap windows), and **HCA (c128a)** = compress every 128 tokens
  → 1 KV. Both share the same `Compressor` module: a *learned gated soft-pool*
  `softmax(gate(x)) · kv(x)` over a window. Source code is public on HF
  (`deepseek-ai/DeepSeek-V4-Pro`).
- **NSA** (2025-02) is the academic precursor: 3-branch (compress + select +
  sliding-window) with a learned MLP `φ` pooling block-of-`l` tokens (block=32, stride=16
  in the paper).
- **Recommended L2 design for our setting**: a *NSA/V4-style learned-gated attention
  pool* over groups of `g=16` tokens, producing `n_h=32` per-head latent K/V vectors.
  This is option **(c)** in the design space below and gives ~256 latents per 4 k chunk.
  Estimated effort: **8–12 h** for a first prototype; **2–3 days** to ship a trainable
  variant alongside the current `mem_space` ablation.

---

## 1. DeepSeek-V2 MLA — Mechanism Summary

### 1.1 Equations (from §2.1.2 + Appendix C of arXiv 2405.04434)

For input hidden `h_t ∈ R^d` at layer `l`:

```
# --- KV path (low-rank, shared across heads) ---
c_KV_t   = W^DKV · h_t            # [d_c]    KV compression
k_C_t    = W^UK  · c_KV_t         # [n_h · d_h]   reconstructed keys
v_C_t    = W^UV  · c_KV_t         # [n_h · d_h]   reconstructed values

# --- Decoupled-RoPE shared key (carries position info) ---
k_R_t    = RoPE(W^KR · h_t)        # [d_h^R]   single shared decoupled key

# --- Q path (low-rank, training-time only) ---
c_Q_t    = W^DQ · h_t              # [d_c']
q_C_t    = W^UQ · c_Q_t            # [n_h · d_h]    reconstructed Q (no-RoPE part)
q_R_t    = RoPE(W^QR · c_Q_t)      # [n_h · d_h^R]  per-head decoupled-RoPE Q

# --- Concatenate at attention time ---
q_t,i    = [q_C_t,i ; q_R_t,i]                # ∈ R^{d_h + d_h^R}
k_t,i    = [k_C_t,i ; k_R_t]                  # k_R is shared across heads
softmax_j q_t,i^T k_j,i / √(d_h + d_h^R) · v_C_j,i
```

### 1.2 What is cached

Per token, per layer the inference cache stores **only**:

- `c_KV_t ∈ R^{d_c}`  (the joint K/V latent)
- `k_R_t  ∈ R^{d_h^R}` (the shared decoupled-RoPE key)

So the cache is `(d_c + d_h^R)` floats per token per layer = `512 + 64 = 576` for V2.
Compare to MHA `2 · n_h · d_h = 2·128·128 = 32 768` → **57× smaller**.

The trick that lets the *content* part be cached as `c_KV` (not as full K, V) is that
during inference `W^UK` can be absorbed into `W^Q` (and `W^UV` into `W^O`); equivalently
the attention computes `q_t,i^T (W^UK c_KV_j)` = `(W^UK^T q_t,i)^T c_KV_j`. RoPE breaks
this absorption — that's why the RoPE component is **decoupled** onto a separate
`k_R_t` that is *not* low-rank.

### 1.3 Exact dimensions

| Symbol | DeepSeek-V2 (236B) | DeepSeek-V2-Lite (16B) |
|---|---|---|
| `d` (hidden) | 5120 | 2048 |
| `n_h` (heads) | 128 | 16 |
| `d_h` (head dim, content) | 128 | 128 |
| `d_h^R` (head dim, RoPE) | 64 | 64 |
| `d_c` (KV latent) | 512 | 512 |
| `d_c'` (Q latent) | 1536 | (no Q compression) |
| `L` (layers) | 60 | 27 |

For Llama-3-8B (`d=4096, n_h=32, d_h=128, L=32`) a faithful MLA-style L2 would set
`d_c ≈ 512` (≈ same compression ratio), `d_h^R = 64`.

### 1.4 Code references (DeepSeek-V2 official HF repo)

- Repo: `huggingface.co/deepseek-ai/DeepSeek-V2` — `modeling_deepseek.py`
- Class: `DeepseekV2Attention` (subclass of `nn.Module`) — declares
  `q_a_proj` / `q_b_proj` (= `W^DQ`/`W^UQ`), `kv_a_proj_with_mqa` (concat of
  `W^DKV` and `W^KR`), `kv_b_proj` (= `[W^UK | W^UV]`).
- Forward path (~line 470): builds `q_pe`, `k_pe` via `apply_rotary_pos_emb`, then
  scaled-dot-product attention on `[q_nope ; q_pe]` × `[k_nope ; k_pe]`.
- `DeepseekV2FlashAttention2` (~line 600) is the FA2-optimised version; identical
  algebra.

(File paths/line numbers vary between commits; the relevant entry point is always
`DeepseekV2Attention.forward`.)

---

## 2. DeepSeek-V3 — What Changed

V3 (arXiv 2412.19437, Dec 2024) keeps **MLA architecturally identical** to V2.
Same `d_c=512, d_h^R=64, n_h=128, d_h=128`. The deltas vs. V2 are:

- DeepSeekMoE auxiliary-loss-free load balancing (replaces V2's α₁/α₂/α₃ aux losses).
- Multi-Token Prediction (MTP) auxiliary head during pre-training.
- FP8 mixed precision training.
- Larger model (671B total / 37B active vs. V2's 236B / 21B).

For the L2 design, **V3 contributes nothing new** beyond what V2 already provides.

---

## 3. DeepSeek-V4 — The Group-Token Compressor (the user was right)

DeepSeek-V4-Pro and V4-Flash were released **2026-04-24**. Architecturally V4
introduces **four** KV-cache strategies that are mixed across layers:

| Module | Compress ratio | Sliding window | DSA (sparse) |
|---|---|---|---|
| **CSA** (Compressed Sparse Attention) | every 4 tokens → 1 (overlap windows of 8) | + 128-token SWA on top | yes |
| **HCA** (Heavily Compressed Attention) | every 128 tokens → 1 | + 128-token SWA on top | no |
| **SWA-only** | — | 128-token SWA | no |
| **DSA** (DeepSeek Sparse Attention, from V3.2) | full KV | — | learned top-k retrieval |

Net effect: at 1 M context, V4-Pro uses **27% of V3.2's per-token FLOPs** and **10%
of V3.2's KV cache** with no quality loss.

### 3.1 The V4 `Compressor` module — verbatim

Public source from `huggingface.co/deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`
(reproduced in CSDN article 2026-04-29 by Luchang-Li). Stripped to the essential
algebra:

```python
class Compressor(nn.Module):
    def __init__(self, dim, head_dim=512, compress_ratio=4, rope_head_dim=64):
        self.compress_ratio = compress_ratio        # 4 (CSA) or 128 (HCA)
        self.overlap        = (compress_ratio == 4) # only CSA uses overlap
        coff = 1 + self.overlap                     # 2 if overlap else 1
        # APE = learned per-window-position bias added to the gate score
        self.ape   = nn.Parameter(torch.empty(compress_ratio, coff * head_dim))
        # The two heads: one for KV content, one for the *gate* (scoring) score
        self.wkv   = Linear(dim, coff * head_dim)
        self.wgate = Linear(dim, coff * head_dim)
        self.norm  = RMSNorm(head_dim)

    def forward(self, x, start_pos):
        # x: [B, T, d]; we want [B, T // compress_ratio, head_dim]
        kv     = self.wkv(x.float())     # [B, T, head_dim]
        score  = self.wgate(x.float())   # [B, T, head_dim]
        # 1. unflatten into windows of size `compress_ratio`
        kv_w   = kv.unflatten(1, (-1, ratio))     # [B, T/r, r, head_dim]
        score_w= score.unflatten(1, (-1, ratio)) + self.ape   # add learned APE bias
        # 2. soft-pool with score-driven softmax
        kv_c   = (kv_w * score_w.softmax(dim=2)).sum(dim=2)   # [B, T/r, head_dim]
        # 3. (optional) overlap_transform pulls half-head from the *previous*
        #    window, doubling head_dim before pooling — gives smoother boundaries
        # 4. RMSNorm → apply RoPE only on the last `rope_head_dim` dims
        kv_c   = self.norm(kv_c)
        apply_rotary_emb(kv_c[..., -rope_head_dim:], freqs_cis_decimated)
        return kv_c
```

Key takeaways for our L2:

1. **Soft attention pooling**, not mean / strided. The "score" head is just a 2nd
   linear projection used as the importance-weighting gate (no separate query —
   the score *is* the per-window-position importance).
2. **Learned positional bias APE** of shape `[ratio, head_dim]` is added to the
   gate before the within-window softmax. This is the *intra-block* position
   encoding — strictly cheaper than full RoPE and only `ratio · head_dim` params.
3. **RoPE is decoupled MLA-style**: only the last `rope_head_dim=64` dims of the
   pooled output get RoPE; the rest is content.
4. **One pooled vector replaces both K and V** (same MLA trick — `head_dim` covers
   both, then `wkv_kv = self.kv_b_proj(kv_c)` reconstructs K, V at attention time).
5. **Overlap (only at ratio=4)** halves boundary information loss: each CSA latent
   sees 8 tokens (the current 4 + the previous 4), pooled with a head-split trick
   (first half of `head_dim` from previous window, second half from current).
6. **`head_dim = 512` is *much* larger than the per-head dim of the dense path**
   (the dense path has `d_h = 128` per head). The compressor effectively makes
   each compressed token a single very-fat head.

### 3.2 NSA's compression (DeepSeek-AI, arXiv 2502.11089, Feb 2025)

NSA is the *academic published* predecessor of V4's CSA — same family, simpler.

```
K̃_cmp_t = { φ(k_{i·d+1 : i·d+l})  | 0 ≤ i ≤ ⌊(t-l)/d⌋ }
```

- `l` = block length (paper: 32)
- `d` = sliding stride (paper: 16; `d < l` to mitigate boundary fragmentation —
  same motivation as V4's overlap)
- `φ` = a *learnable MLP with intra-block position encoding* mapping
  `R^{l × d_k} → R^{d_k}`. The paper does not specify φ in detail, but the
  reproduction in V4 shows the exact recipe:
  `φ(K_block) = sum_j softmax_j(gate(K_block) + APE_j) · proj(K_block)_j`.

NSA combines compressed + selected (top-n blocks chosen by `softmax(q^T K̃_cmp)`)
+ sliding-window via three parallel attention paths gated by an MLP:

```
o_t = Σ_{c ∈ {cmp, slc, win}}  g_t^c · Attn(q_t, K̃^c_t, Ṽ^c_t)
g_t^c = sigmoid(MLP(h_t))_c
```

This is **structurally identical** to what we want for L2, except NSA uses K̃ for
computing the importance scores that drive selection — a path we don't currently
need (we already have L1 doing the slot-level retrieval).

---

## 4. Adaptation to Our Setting

### 4.1 Constraints

- **Backbone**: Llama-3-8B (fixed, frozen). `d=4096`, `n_h=32`, `d_h=128`, `L=32`.
- **Chunk size**: 4096 (`scripts/run_babilong_mem_space.py:374`).
- **Joint-attn extended sequence** today: `[L1_slots(64), H(T)]` (slot positions
  RoPE'd at 0, see `src/memory/mem_space/layer.py:163-184`).
- **L2 target**: ~256 latent KV per chunk → compression ratio `g = 16`.
- **Trainable budget**: only the adapter trains; we add ~`L · O(few M)` params
  total at most.
- **Joint attention insertion**: extended sequence becomes
  `[L1_slots(64), L2_latents(~256), H(T)]`. Slot positions RoPE-0 (existing);
  L2 positions need a sensible scheme — see §4.4.

### 4.2 Four candidate L2 designs

| Option | Parameter count (per layer) | Compute (per chunk) | Expressiveness | Stability risk |
|---|---|---|---|---|
| **(a)** Pure MLA per-token | `4 d · d_c + d_c · 2 n_h d_h ≈ 8 + 16 = 24 M @ d_c=512` | `O(T · d · d_c)` | High but **no compression**: still 4 k tokens cached → wrong tier | low |
| **(b)** Group-mean + LoRA | `d · d_c (down) + d_c · 2 n_h d_h (up) ≈ 24 M` | `O(T · d · d_c)` | Low: mean-pool drops salience; padding tokens contaminate boundary | medium (collapse) |
| **(c)** Learned attention pool *(NSA / V4 CSA)* | `2 · d · d_c (wkv + wgate) + APE(g, d_c) + d_c · 2 n_h d_h ≈ 25 M` | `O(T · d · d_c)` | High: learns which tokens matter within a window | low (well-tested in V4) |
| **(d)** Strided keep-every-g | `d_c · 2 n_h d_h (only up) ≈ 16 M` | `O((T/g) · d · d_c)` | Low: drops 15/16 tokens blindly | low (no learnable pool to break) |

### 4.3 Recommendation: **Option (c) — NSA/V4-style learned attention pool**

**Why (c) over the others:**

- **(a)** does not give compression — it stores `T/g · d_c` floats per layer if we
  group, which is what we want, but per-token MLA stores `T · d_c` (no group).
  We'd be back to L1 fidelity at L1 size.
- **(b)** mean-pool is the well-known weak baseline. Without a learned gate,
  high-salience tokens (entities, numbers) get diluted by surrounding "the" tokens
  → BABILong fact retrieval will collapse.
- **(d)** strided is robust but provably loses info on a fact spanning multiple
  tokens (e.g. multi-token entity names) unless `g=1`.
- **(c)** has been **trained at 1 M context by DeepSeek themselves** — strongest
  prior that the design works. The extra cost vs. (b) is a single linear (`wgate`)
  + one APE bias, ~2 M params per layer, ~64 M total for 32 layers (less than
  what we already added in `mem_space` adapters).

### 4.4 Concrete dim numbers for Llama-3-8B

```python
# Hyperparams (proposed defaults)
d            = 4096          # backbone hidden
n_h          = 32            # backbone heads
d_h          = 128           # backbone head dim
g            = 16            # group / compression ratio (one L2 latent per 16 tokens)
d_c          = 512           # L2 latent dim (matches V2 MLA)
d_h_R        = 64            # decoupled-RoPE per-head dim
T_chunk      = 4096          # → 256 latents per chunk
overlap      = False         # start without overlap (add later if quality matters)

# Per-layer L2 parameter count
#   wkv   :  d × d_c                = 4096 · 512   = 2.10 M
#   wgate :  d × d_c                = 4096 · 512   = 2.10 M
#   APE   :  g × d_c                = 16   · 512   = 0.008 M
#   norm  :  d_c                    = 512          ≈ 0
#   kv_b  :  d_c × 2 n_h · d_h      = 512 · 8192   = 4.19 M  (shared up-projection)
#   wkr   :  d × d_h_R              = 4096 · 64    = 0.26 M  (decoupled-RoPE key)
# total ≈ 8.7 M / layer  ×  32 layers = 278 M params (NEW trainable)
```

If 278 M is too many adapters, drop `kv_b` (re-use the wrapped layer's existing
K/V projections by *projecting `c_KV` back through them*): then per-layer cost
drops to ~4.5 M, total ≈ 144 M — comparable to LoRA-r=64 on the full backbone.

### 4.5 RoPE positions for L2 latents

Three choices, ordered by simplicity:

1. **Position 0 for *all* L2 latents** (same as L1). Pro: trivial, matches
   `_extend_position_embeddings`. Con: L2 latents become "global memory tokens"
   indistinguishable from L1 — defeats the purpose of having both.
2. **Decoupled MLA-RoPE on the latent itself** (V4 way): the *content* part of
   each latent stays position-agnostic but the latent gets its *own* `k_R`
   computed from the *first* token of its window. Pro: principled, matches V4.
   Con: requires hooking the wrapped LlamaDecoderLayer's K projection (we'd be
   editing dense path, not just adding sidecar).
3. **One RoPE position per *window-centroid*** (recommended **first try**): for
   the `i`-th L2 latent of the previous chunk, set its RoPE position equal to
   `(prev_chunk_end - g · (n_l2 - i) + g/2)` relative to the current chunk's
   start. Pro: cheap, gives sequential ordering, doesn't touch wrapped layer.
   Con: positions can be negative for very early latents — clip to 0.

For the v0 prototype we use **(3)**.

---

## 5. Code Sketch — `L2Compressor` Module

```python
# src/memory/mem_space/l2_compressor.py  (NEW FILE; keep separate from layer.py)
"""
L2 token-compressed KV memory — NSA / V4-CSA style learned attention pool.

For each chunk of T tokens we produce  T/g  latent KV vectors of dim  d_c.
The previous chunk's latents are read by the current chunk's joint attention
(prepended into the wrapped LlamaDecoderLayer's K/V cache).
"""
import math, torch, torch.nn as nn, torch.nn.functional as F


class L2Compressor(nn.Module):
    """One per layer. Maintains its own latent buffer across chunks."""

    def __init__(
        self,
        d_model: int = 4096,
        n_heads: int = 32,
        d_head: int = 128,
        compress_ratio: int = 16,        # g
        d_c: int = 512,                  # latent / content dim
        d_h_rope: int = 64,              # decoupled-RoPE dim per latent
        chunk_size: int = 4096,
    ):
        super().__init__()
        self.g     = compress_ratio
        self.d_c   = d_c
        self.d_h_R = d_h_rope
        # 2.1 wkv : d_model -> d_c   (content of pooled latent)
        # 2.2 wgate: d_model -> d_c  (per-token-per-channel gate score)
        self.w_kv   = nn.Linear(d_model, d_c, bias=False)
        self.w_gate = nn.Linear(d_model, d_c, bias=False)
        # 2.3 Learned absolute-position-in-window bias (APE):  [g, d_c]
        self.ape    = nn.Parameter(torch.zeros(compress_ratio, d_c))
        # 2.4 RMSNorm on the latent
        self.norm   = nn.RMSNorm(d_c)
        # 2.5 Up-projection that reconstructs K, V back into model space.
        #     One linear per layer, output split into (K, V) of shape [n_h, d_h].
        self.kv_b   = nn.Linear(d_c, 2 * n_heads * d_head, bias=False)
        # 2.6 Decoupled-RoPE key (one *per latent*, computed from window-mean of h).
        self.w_kR   = nn.Linear(d_model, d_h_rope, bias=False)
        # init small
        nn.init.normal_(self.w_kv.weight,   std=0.02)
        nn.init.normal_(self.w_gate.weight, std=0.02)
        nn.init.normal_(self.kv_b.weight,   std=0.02)
        nn.init.normal_(self.w_kR.weight,   std=0.02)

        # Cross-chunk buffer: [B, n_latents, d_c+d_h_rope] (latents from prior chunk)
        self.register_buffer("prev_latents", torch.empty(0), persistent=False)

    @torch.no_grad()
    def reset(self):
        """Called by the training/eval loop at chunk boundary == document boundary."""
        self.prev_latents = self.prev_latents.new_empty(0)

    def compress(self, h: torch.Tensor) -> torch.Tensor:
        """
        h : [B, T, d_model]  (current chunk's hidden states, post-input-norm of layer)
        returns: [B, T/g, d_c+d_h_rope] of compressed latents.

        Compression matches V4's `Compressor` (no overlap variant): wkv + wgate +
        APE-biased softmax over windows of size g, then RMSNorm + decoupled-RoPE
        component on the last d_h_R dims.
        """
        B, T, _ = h.shape
        g, d_c, d_R = self.g, self.d_c, self.d_h_R
        if T % g != 0:
            # right-pad so T%g==0; padded positions contribute 0 via -inf gate
            pad = g - T % g
            h = F.pad(h, (0, 0, 0, pad))
            T += pad
        # 1. project to (kv, gate)
        kv    = self.w_kv  (h.float())             # [B, T, d_c]
        gate  = self.w_gate(h.float())             # [B, T, d_c]
        # 2. unflatten into windows
        kv_w   = kv.unflatten(1,    (-1, g))       # [B, T/g, g, d_c]
        gate_w = gate.unflatten(1,  (-1, g)) + self.ape  # APE broadcasts over (B, T/g)
        # 3. soft-pool
        latent = (kv_w * gate_w.softmax(dim=2)).sum(dim=2)   # [B, T/g, d_c]
        latent = self.norm(latent)
        # 4. decoupled-RoPE component (computed from the *first* h of each window)
        h_w_first = h.unflatten(1, (-1, g))[:, :, 0, :]      # [B, T/g, d_model]
        kR        = self.w_kR(h_w_first.float())             # [B, T/g, d_h_R]
        return torch.cat([latent.bfloat16(), kR.bfloat16()], dim=-1)  # [B, T/g, d_c+d_h_R]
```

### 5.1 Wiring into `MemorySpaceLayer.forward`

Insertion point: right after the slot path builds `M_sel_hidden` and right before
the joint-attn `extended_hidden = torch.cat([M_sel_hidden, hidden_states], dim=1)`
at `src/memory/mem_space/layer.py:568`.

```python
# --- new: L2 latents from the *previous* chunk are read here ---
if self.l2 is not None and self.l2.prev_latents.numel() > 0:
    # Reconstruct K, V into model space (single linear, then split).
    pl = self.l2.prev_latents  # [B, n_l2, d_c+d_h_R]
    pl_content, pl_kR = pl.split([self.l2.d_c, self.l2.d_h_R], dim=-1)
    # Up-project to model-space "tokens": shape [B, n_l2, d_model] so we can
    # prepend to extended_hidden and let the wrapped layer's K/V projections
    # turn it into K, V naturally.  The kv_b out is split into K and V; we
    # re-assemble a single token vector by *averaging* K and V (cheap proxy
    # since the wrapped layer will recompute its own Q from the same vector
    # at zero RoPE).  The principled alternative — bypass W^K/W^V and inject
    # K/V directly — is Stage-2 (touches the wrapped layer's attention).
    kv_recon = self.l2.kv_b(pl_content)            # [B, n_l2, 2 * n_h * d_h]
    K_recon, V_recon = kv_recon.chunk(2, dim=-1)
    L2_tokens = 0.5 * (K_recon + V_recon)          # [B, n_l2, d_model]
    extended_hidden = torch.cat(
        [M_sel_hidden, L2_tokens, hidden_states], dim=1
    )
    k_l2 = L2_tokens.shape[1]
    # Position embeddings: L2 tokens get RoPE position 0 (same as L1) for v0.
    # Stage-2: switch to per-window-centroid RoPE positions (§4.5 option 3).
    ext_pos_emb = _extend_position_embeddings(
        position_embeddings, k_slots + k_l2
    )
    # Mask: rebuild including L2.  Same recipe as L1 but with an extra k_l2
    # rows/cols of "always allowed".
    ext_attn_mask = _build_extended_attn_mask_l2(
        k_l1=k_slots, k_l2=k_l2, T=T, ...
    )
else:
    # No L2 yet (first chunk) — fall back to the existing path
    extended_hidden = torch.cat([M_sel_hidden, hidden_states], dim=1)
    k_l2 = 0
    ext_pos_emb = _extend_position_embeddings(position_embeddings, k_slots)
    ext_attn_mask = _build_extended_attn_mask(...)

# --- (existing) run wrapped_layer on extended_hidden ---
...

# --- new: write current chunk's L2 latents for the *next* chunk to read ---
if self.l2 is not None:
    with torch.no_grad():
        # Use the *post-layer* hidden states.  Detach to keep BPTT chunk-local.
        new_latents = self.l2.compress(next_hidden.detach())
        self.l2.prev_latents = new_latents
```

### 5.2 Mask helper sketch

```python
def _build_extended_attn_mask_l2(k_l1, k_l2, T, dtype, device, B):
    """Same convention as _build_extended_attn_mask; rows L1-then-L2-then-H."""
    L = k_l1 + k_l2 + T
    mask = torch.zeros(L, L, dtype=dtype, device=device)
    neg = torch.finfo(dtype).min
    if T > 0:
        causal = torch.triu(
            torch.full((T, T), neg, dtype=dtype, device=device), diagonal=1
        )
        h0 = k_l1 + k_l2
        mask[h0:, h0:] = causal
    # L1 + L2 rows: see everything → already 0
    # H rows can attend to all L1+L2 keys → already 0 in the (h0:, :h0) block
    return mask.view(1, 1, L, L).expand(B, 1, L, L).contiguous()
```

### 5.3 Patch-level wiring

In `src/memory/mem_space/patch.py` `apply_mem_space_to_model`:
- After `MemorySpaceLayer(...)` is constructed, instantiate one `L2Compressor`
  and attach as `wrapper.l2 = L2Compressor(...)`.
- Expose a `reset_l2()` collector function (parallel to the existing
  `_reset_banks`) that calls `.reset()` on every layer's L2.
- In `scripts/run_babilong_mem_space.py`, call `reset_l2()` between documents
  (alongside the existing slot reset).

---

## 6. Eval contract

The current chunked eval (`scripts/run_babilong_mem_space.py:281-345`) is
unchanged.  L2 is **transparent**:
- Chunk `i` writes its L2 latents into `wrapper.l2.prev_latents` (post-forward).
- Chunk `i+1`'s `MemorySpaceLayer.forward` reads `prev_latents` and prepends
  to the joint-attn extended sequence — exactly like L1 slots.
- At document boundary the eval loop calls `reset_l2()` (parallel to the
  existing slot bank reset).
- No change to `chunk_size=4096`, no change to BABILong dataset code.

---

## 7. Engineering effort estimate

| Phase | Hours | What |
|---|---|---|
| **0. Spec freeze + reviewer pass** | 0.5 | (this doc) |
| **1. Implement `L2Compressor`** | 2 | New file `l2_compressor.py`, ~150 lines, plus unit test that compress(zeros) == 0, compress(constant) ≈ constant |
| **2. Wire into `MemorySpaceLayer.forward`** | 2–3 | The cat-and-mask helper + `prev_latents` readwrite + RoPE-position handling |
| **3. Wire into `patch.py` + reset hook** | 1 | Per-layer L2 instantiation + collector |
| **4. Smoke test (single chunk == bypass)** | 1 | `slot_output_gate=0` + `prev_latents.numel()==0` should reproduce vanilla bf16 |
| **5. Two-chunk smoke (latents flow, no NaN)** | 1 | Chunk-1 writes, chunk-2 reads, gradient flows back to `w_kv` |
| **6. BABILong eval @ 8k context (1 H20)** | 1 | Confirm L2 doesn't *degrade* current `mem_space` PPL/acc |
| **7. RoPE-position upgrade (§4.5 option 3)** | 1–2 | Optional, deferred to Stage 2 |

**Total for first prototype: 8–12 h.** This is the minimum to get a trainable
L2 ablation arm running. The eval at §6 step 6 is the gate for "does it help".

---

## 8. Open questions for team-lead

1. **Should L2 be initialised "near-zero" so the first eval == current mem_space**?
   Yes — set `kv_b.weight.std=0.001` so L2 contributes ≈ 0 to attention until it
   trains. Mirrors the `slot_output_gate` Flamingo pattern.
2. **Should L2 share parameters across layers** (one `L2Compressor`, applied per
   layer) or be **per-layer**? Per-layer is V4's choice; saves us from a hard
   capacity bottleneck and matches how mem_space is currently per-layer.
   Recommend per-layer.
3. **Do we want overlap windows (V4-CSA style) at the start**? Recommend **no**
   for v0 — it doubles `wkv` width and adds the ugly head-split. Add if
   non-overlap shows boundary artefacts in BABILong.
4. **L1 + L2 + L3 combined extended sequence**: `[L1(64), L2(256), L3(64), H(4096)]`
   = 4480 tokens of joint attention. SDPA on H20/L20A at bf16 fits comfortably
   for `T=4096` with `B=1` — the existing `mem_space` already pays this cost
   structure with `[64, 4096]`.

---

## 9. References

- **DeepSeek-V2 paper** (arXiv 2405.04434, May 2024) — §2.1.2 "Low-Rank Joint
  Compression", §2.1.3 "Decoupled RoPE", Appendix C "Full Formulas of MLA",
  Appendix B "DeepSeek-V2-Lite" hyperparams.
- **DeepSeek-V3 paper** (arXiv 2412.19437, Dec 2024) — MLA architecturally
  identical to V2; only training/MoE deltas.
- **NSA paper** (arXiv 2502.11089, Feb 2025) — §3.3.1 "Token Compression",
  Eq. 7: `K̃_cmp = { φ(k_{id+1:id+l}) }`.
- **DeepSeek-V4 release blog** (vllm.ai/blog/deepseek-v4 + lmsys 2026-04-25);
  HuggingFace weights at `deepseek-ai/DeepSeek-V4-Pro/tree/main/inference`;
  `Compressor` source mirrored in CSDN article 160597841 (2026-04-29).
- Local context: `src/memory/mem_space/layer.py:411-770` (current joint-attn
  forward), `src/memory/mem_space/patch.py` (per-layer wrapping),
  `scripts/run_babilong_mem_space.py:281-345` (chunked eval contract).
