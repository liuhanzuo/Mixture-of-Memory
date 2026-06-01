# v12 — Summary Reconstruction Auxiliary Loss (P1)

> Date: 2026-06-01. Commit: see git log. Status: implemented, toy-smoke verified.
> Plan ref: `status/MEMORY_PROTOCOL_PLAN.md` [P1]. Toy motivation: commit `e5bb181`.

## Motivation (toy passcode diagnostic)

The 2-chunk passcode toy task (`scripts/toy_memory_bootstrap.py`) writes
`"The passcode is 7392."` into memory in chunk 1, then forces chunk 2
(`"The passcode is"`) to read the answer back from memory only (the two chunks
are forwarded separately, so the vanilla attention path in chunk 2 cannot see
the digits). The three-arm 500-step study gave a decisive result:

| arm | top1_sim | chunk1→2 overlap | retrieval_exact_acc |
|---|---|---|---|
| multi_query baseline | 0.018 | 0.125 | **0.0** |
| multi_query + force_gate | 0.016 | 0.105 | **0.0** |
| **slot_query t40 + force_gate** | **0.32** | **0.29** | **0.0** |

**Routing is NOT the gap.** slot_query+temp40 addressed the written slots well
(top1_sim 0.32, overlap 0.29) yet `retrieval_exact_acc` stayed **0.0**: even
when the right slot is selected, its written content cannot be decoded into the
answer. The missing piece is the **WRITE PROTOCOL** — the writer has no
near-distance objective teaching it to "store readable content"; it only gets
the very indirect LM-loss signal, which is too diffuse to learn from.

## Architecture

New module `src/memory/mem_space/recon_decoder.py` → `MemoryReconDecoder`, a
small 1-layer cross-attention decoder shared as a singleton across all 32
layers (peer to `l3_pool`; attached via `object.__setattr__` to avoid 32×
state_dict duplication, registered once on the model root by `patch.py`).

Forward pass (per chunk, layer_idx==0 only):

```
M_write ∈ [B, k_write, d_slot]      # post-write VALUES of slots updated this chunk
kv      = kv_proj(M_write)          # [B, k_write, d_model]
S       = queries (learnable, orthogonal-init) broadcast to B  # [B, num_summary, d_model]
S       = S + cross_attn(LN_q(S), LN_kv(kv), LN_kv(kv))        # pre-LN residual
S       = S + FFN(LN_ffn(S))                                    # pre-LN residual
S_hat   = LN_out(S)                 # [B, num_summary, d_model]

L_recon = MSE(S_hat, stopgrad(S_L3))          # S_L3 = this chunk's L3 summary tokens
aux["recon"] = L_recon * cfg.l_recon_weight
```

`num_summary == l3_n_summary` so `S_hat` aligns with the L3 summary target.
The decoder is deliberately tiny (1 cross-attn block) — it must NOT be powerful
enough to reconstruct `S_L3` from *generic* slots; the only way to push the loss
down is for the slots to actually carry the chunk's content.

## Key design decisions

### Why stopgrad on the target (`S_L3`)
`S_L3` is detached. Without stopgrad the L3 summary pool would be pulled toward
whatever the slots happen to encode, letting the (decoder, slots, L3) triple
collapse into a trivial mutually-agreed representation that drives MSE to 0
without storing anything useful. Detaching the target forces the gradient to
flow ONLY into the recon decoder and — through `M_write` — into the slot write
path (`hidden_to_slot`, the dual/single writeback gates). The reconstruction
target stays a fixed, independently-produced summary of the chunk.

### Why act on the written slot VALUE (`M_write`), not slot keys / O_mem_hidden
The diagnostic localized the failure to "what got stored is not decodable".
The quantity that ends up persisted and later read back is the slot *value*
after the writeback mix (EMA / dual-gate). Supervising exactly that tensor
gives the most direct credit assignment to the write path. We take the
gradient-bearing return of `memory_bank.write` for the REGULAR top-k slots
(not the always-on global slots) so the target reflects content-routed writes.

### Why `memory_bank.write` now returns the written values
`MemoryBank.write` re-binds `self.slots` under `torch.no_grad()` to apply the
global `slot_value_norm_cap`. Reading `self.slots` after the call would
therefore yield a **detached** tensor (no gradient to the writer). `write` now
returns the gradient-bearing `updated` content (pre norm-cap-rebind) for the
selected slots; the layer uses that as `M_write`. Returns `None` on no-op
(frozen bank / β≈0 / writeback skipped), in which case the recon loss is not
computed that step.

### Computed on layer_idx==0 only
Same guard as `l3_diversity` / `q_multi_diversity`: the recon decoder is a
shared singleton, so collecting the loss on every layer would count it 32× and
inflate its magnitude. Only the first layer's `M_write` feeds the decoder.

### L3 dependency
The recon target IS the L3 summary, so `use_l3_summary=True` must be on (the
toy script forces this). On the cold-start first chunk `l3_summaries is None`
→ the loss is skipped. `slot_query` routing does not itself use L3, but recon
still needs L3 as the reconstruction target, so L3 stays enabled during
training even under slot_query.

## Config / CLI

- `MemorySpaceConfig.l_recon_weight: float = 0.0` (default 0 ⇒ disabled,
  fully backward-compatible — the decoder is not even constructed).
- `--l_recon_weight` added to both `scripts/train_mem_space_dolmino_cpt.py`
  and `scripts/toy_memory_bootstrap.py`; wired into `MemorySpaceConfig`.
- `"recon"` added to the `_collect_aux_loss` key tuple.
- recon decoder params added to `_mem_space_params` (so they enter the
  optimizer + `_freeze_backbone` keeps them trainable).
- toy training log now prints `recon=<value>` per log step.

## Validation (2026-06-01, .venv, single H20)

1. **Unit sanity**: `MemoryReconDecoder` with `M_write [2,16,64]` → `S_hat
   [2,8,64]`, MSE loss scalar+finite, backward flows grad to all 19 decoder
   params and to `M_write`; verified `stopgrad` keeps the target's grad `None`.
2. **30-step toy smoke** (`slot_query`, temp 40, `l_recon_weight 0.1`): no
   crash, 0 non-finite, TOY_DIAG/QUERY_DIAG/WRITEBACK_DIAG print normally,
   top1_sim ≈ 0.49 (healthy), and the recon loss falls monotonically:
   `0.1999 → 0.0479 → 0.0216 → 0.0122 → 0.0068 → 0.0045`. (30 steps is too
   short to move `retrieval_exact_acc`; that is the Round-1 sweep's job.)

## Relationship to prior work
- Unlike the L3 `query_diversity_loss` (v9/v10), which regularizes the routing
  *query* space, this loss supervises the *stored content* — a write-side
  objective, not a routing-side one.
- Conceptually closest to auto-encoding / reconstruction auxiliary objectives
  in memory networks (e.g. compressive transformer's reconstruction/attention
  losses), but here the bottleneck is the discrete top-k slot write and the
  reconstruction target is the higher-level L3 summary rather than raw tokens.

## Known issues / open questions
- The decoder reconstructs *all* `num_summary` L3 tokens from only `k_write`
  (top-k) slot values. If a single chunk's information legitimately spans more
  than the written slots, the target may be partly unreconstructable — this is
  intentional pressure but could cap how low the loss can go.
- `M_write` for the always-on **global** slots (v7) is currently excluded from
  the recon target. If global slots become the primary store for a config,
  revisit whether to include them.
- The loss is collected on layer_idx==0 only; the deeper layers' write paths
  get the recon signal only indirectly via the shared bank's BPTT. Whether
  per-layer recon helps is an untested ablation.
- Weight sweep (`l_recon_weight ∈ {0.05, 0.1, 0.3, 1.0}`) and the gold-metric
  check (does `retrieval_exact_acc` rise above 0?) are Round-1 follow-ups, not
  covered by this smoke.
