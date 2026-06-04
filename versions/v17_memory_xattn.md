# v17 — P8: Dedicated Memory Cross-Attention READ (`use_memory_xattn`)

**Date:** 2026-06-05
**Flag:** `--use_memory_xattn` (config: `use_memory_xattn: bool = False`, `memory_xattn_gate_init: float = 0.4`)
**Plan:** `status/MEMORY_PROTOCOL_PLAN.md` [P8]
**References:** YOCO (2405.05254), Memorizing Transformers (2203.08913), Infini-attention (2404.07143)

## Motivation

Two hard data points (2026-06-05):
- Memory adapter on LongBench open-ended QA: avg **F1=2.94 vs base Llama-3-8B F1=13.95** — ~5x WORSE. The read-back destroys natural-language context.
- P7 (route_aux + loss-free balancing + register slots): routing is now stable and learnable (top1_sim ~0.3, balanced usage), but BABILong accuracy is only COMPARABLE to a plain route_aux arm. **Routing is no longer the bottleneck.**

Researcher P2 root cause: memory slots are injected by **KV-prepend** — the ~16 slot key/value vectors are prepended into the live-token KV and share the SAME softmax as the real tokens. With hundreds/thousands of live tokens and only a handful of slots, the slots get only ~0.2% of the attention mass → diluted to irrelevance → even with perfect routing there is no gradient/signal flowing through memory.

## Architecture

When `use_memory_xattn=True`:

```
# READ path (per MemorySpaceLayer.forward):
mask H -> L1 prepend block        # live tokens no longer see slot KV (mask_h_to_l1)
slot_delta = ext_h[H] - bypass_h  # now carries only L3/L2 (≈0 when absent)

xattn_slots = slot_to_hidden(slots) if slot_dim != d_model else slots   # [B, N, d]
read_out = MemoryCrossAttentionRead.read(
    Q = hidden_states,            # live-token queries
    K = V = xattn_slots,          # ALL slots
)                                 # OWN softmax over N slots; per-head gated; [B, T, d]

next_hidden = bypass_h + g * slot_delta + fast_mem_out + read_out
#                                                         ^^^^^^^^ added DIRECTLY
#   (read_out is already per-head content-gated inside the module — NOT scaled
#    by the shared scalar inject_gate g, unlike P2's decoupled_read)
```

`MemoryCrossAttentionRead.read` (src/memory/mem_space/selector.py):
```
Q = q_proj(hidden); K = k_proj(slots); V = v_proj(slots)     # GQA: n_kv_heads may differ
attn = softmax(Q·Kᵀ / sqrt(d_head))        # INDEPENDENT softmax over N slots
o    = attn · V                            # [B, H, T, d_head]
g_h  = sigmoid(gate_proj(hidden))          # [B, T, H] per-HEAD content gate
o    = o * g_h                             # per-head gated
read_out = out_proj(merge_heads(o))        # small-random out_proj (NOT zero)
```

## Initialization (the load-bearing choices)

| param | init | why |
|-------|------|-----|
| `q/k/v_proj.weight` | `N(0, 0.02)` | standard small init |
| `out_proj.weight` | `N(0, 0.02)` **(small-random, NOT zero)** | the read output must be ACTIVE at init so gradient flows through memory from step 0 |
| `gate_proj.weight` | `N(0, 0.02/√d)` | per-token gate variation from step 0 (avoid constant-gate deadlock) |
| `gate_proj.bias` | `logit(gate_init)` = `logit(0.4) ≈ -0.405` | effective per-head contribution ≈ `gate_init` (0.4, in the 0.3-0.5 band) at init |

## Relationship to prior work / other versions

- **vs P2 `use_decoupled_read` (v15):** both give slots a standalone cross-attention with its own softmax and mask off the H→L1 prepend (shared `mask_h_to_l1` plumbing). The DIFFERENCE: P2 uses `CrossAttentionMemoryV2` with **zero-init out_proj** blended via the shared scalar inject_gate (g≈0.12 with `inject_gate_bias_init=-2.0`) → the read path is ~dead at init, inheriting the "no gradient through memory" problem from the other side. P8 uses **small-random out_proj + a per-head content gate init≈0.4** → active from step 0. `use_memory_xattn` takes precedence over `use_decoupled_read` if both set (mask plumbing is OR'd; both reads would add, but the launch script never sets both).
- **vs KV-prepend (default):** removes the ~0.2% mass dilution by not putting slots in the live-token softmax at all.
- **YOCO / Memorizing Transformers / Infini-attention:** all use a separate, gated cross-attention to read from a memory/KV cache rather than concatenating into the local attention — P8 follows this pattern.

## Scope / invariants

- **READ path only.** Writeback (EMA / dual-gate / lowrank / diag), top-k routing, loss-free balancing, register slots, L3/L2 — all UNCHANGED.
- When `use_memory_xattn=False`: byte-for-byte the legacy prepend path. The `memory_xattn` submodule is `None` and `mask_h_to_l1` stays off. Verified: `tests/test_mem_space_smoke.py` (6 passed), and the new flag-off check in `tests/test_mem_space_p8_xattn_smoke.py`.
- Round-trips through `adapter_config.json` (`use_memory_xattn`, `memory_xattn_gate_init` are saved + reconstructed generically by both eval loaders), so a checkpoint trained with the flag evals with the flag.

## Known issues / risks

- **Selector gradient needs ≥2 patched layers.** With the H→L1 mask, the selector's in-layer STE-via-`slot_delta` path is severed; routing gradient then flows through the cross-layer write→read chain (shared bank). Single-patched-layer toy configs show selector grad = 0 (true for P2 as well). The real 32-layer + `shared_memory_bank` config threads it. Verified in `test_xattn_gradients_flow` (2 layers).
- The per-head gate output `read_out` is added directly to the residual (not norm-clipped). Init contribution ~0.4 of a small-random-projected attention output; if it proves too hot at scale, lower `--memory_xattn_gate_init`.
- Cold-start: honours `zero_alpha_on_cold_start` (read zeroed on the first, uninitialised-slot chunk).

## Smoke test

`tests/test_mem_space_p8_xattn_smoke.py` (CPU, tiny random Llama):
1. flag off → `memory_xattn is None`.
2. flag on → forward shape OK, no NaN/Inf, and **read active at init** (mean|xattn − no_mem| = 1.2e-2 > 0).
3. gradients flow into all xattn read params (q/k/v/out/gate) + selector (2-layer config).
