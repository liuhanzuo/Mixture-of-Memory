# Memory Bank Bug Audit & Fix — 2026-04-25

Author: autonomous chain (coder role).
Scope: sparse memory write/update logic only. Forget logic is out of scope per user.

## Context

User question: "测试一下训练过程中memory slot的写入频率. memory有问题只可能是因为:
写入逻辑有问题/更新逻辑有问题/遗忘逻辑有问题(这个我们现在还涉及不到). 可以调研一下"

Two sparse-memory implementations live in the tree:

| path | class | consumers |
|------|-------|-----------|
| `src/memory/sparse/memory_bank.py` | `SparseMemoryBank` | `scripts/train_gated_sparse_memory.py`, `scripts/eval_phase1_gate.py`, `scripts/eval_nih_extended_sparse.py` |
| `src/memory/sparse_memory/memory_bank.py` | `MemoryBank` | `scripts/train_sparse_memory.py`, `scripts/eval_sparse_memory_ppl.py`, demos |

`src/memory/mag/self_update_function.py::update_kv_pool` is a third write path
(MAG pipeline). MAG is already marked **abandoned** in `CLAUDE.md` §"已完成的工作",
so bugs there are documented but not fixed.

## Bugs identified

### Bug #1 — `write_top_k=0` default overwrites entire buffer every chunk

**Where:**
- `src/memory/sparse/memory_bank.py:38` (constructor default)
- `src/memory/sparse_memory/memory_bank.py:41` (constructor default)
- `scripts/train_gated_sparse_memory.py:169` (CLI arg default)
- `scripts/train_sparse_memory.py:188` (CLI arg default)

**What happens:** with `T=4096` tokens per chunk and `num_slots=128`, writing all
T tokens wraps the circular buffer ~32× per chunk, so every chunk's last 128 tokens
become the entire memory. The first 3968 tokens contribute to EMA-blended intermediate
states that are promptly overwritten. The bank effectively becomes a 128-token
recency buffer — identical in spirit to a 128-wide sliding window.

**Fix:** default `write_top_k=8` (matches read `top_k`). Zero remains legal but
documented as "legacy, wraps buffer every chunk".

### Bug #3 — Frozen random write gate halves every write

**Where:** `src/memory/sparse/memory_bank.py:61-68`.

```python
self.write_gate = nn.ModuleList([nn.Linear(d, 1) for _ in ...])
for gate in self.write_gate:
    nn.init.kaiming_normal_(gate.weight, nonlinearity='linear')
    nn.init.constant_(gate.bias, gate_bias_init)  # was 0.0
    gate.weight.requires_grad_(False)
    gate.bias.requires_grad_(False)
```

**Why frozen:** writes go through `self.memory.data[layer_idx, slot_ids] = updated`
(line 113) — the `.data` access detaches from autograd. Gradients cannot flow back
from the LM loss into the write gate. The gate was frozen to avoid
DDP "unused parameter" deadlocks when `find_unused_parameters=False`.

**What's wrong:** with `gate_bias_init=0.0`, `σ(0)=0.5`, and the random
Kaiming-normal weight perturbs that 0.5 with input-dependent noise. Every EMA
update is multiplied by `~0.5 ± random(input)`, so half the write signal is lost
and the other half is random.

**Fix:** set `gate_bias_init=4.0`. σ(4)≈0.98, so the gate is ~pass-through at init
and the frozen Kaiming weight's ±noise only perturbs the 98% figure slightly. This
is a workaround for the underlying limitation — the gate is still frozen — but it
restores the original write semantic (α·values + (1-α)·current with minimal gate
attenuation).

**Proper long-term fix (deferred):** either

1. switch to non-`.data` write (e.g. `register_buffer` + scatter) and enable
   `find_unused_parameters=True`, or
2. add a learned importance scorer whose output multiplies the value BEFORE
   `.data` assignment AND is also read at inference time, so gradients reach
   it via the read path.

`src/memory/sparse_memory/memory_bank.py` already implements option (2) via
`compute_importance()` + `learned_importance()`. It's the canonical pattern.

### Bug #2 — MAG `update_kv_pool` broadcasts pooled delta to all N tokens

**Where:** `src/memory/mag/self_update_function.py:230-267`.

```python
key_repr   = keys.mean(dim=2).reshape(B, D)         # (B, D) — pool over tokens
...
new_key_repr = updated_keys_repr[i].reshape(B, H, d).unsqueeze(2)  # (B, H, 1, d)
key_delta   = new_key_repr - key_memory_pool[i].reshape(B, H, d).unsqueeze(2)
updated_keys = orig_keys + key_delta.expand(-1, -1, N, -1)  # broadcast to all N
```

The update module runs on pooled representations then adds the *same* delta to
every token. Functionally this is a uniform shift of the KV cache's mean — it
cannot modify per-token variation. Explains part of MAG's failure mode.

**Status:** MAG is abandoned. Documenting only; no code change.

## Instrumentation added

`SparseMemoryBank` now registers two long buffers and a helper:

```python
self.register_buffer("_write_tokens", torch.zeros(num_layers, dtype=torch.long))
self.register_buffer("_write_calls",  torch.zeros(num_layers, dtype=torch.long))

def write_stats(self) -> dict:
    ...  # returns per-layer token/call counts, totals, tokens/call
```

Counters are bumped inside `write()` after each EMA update. `reset()` clears them.
The existing `src/memory/sparse_memory/memory_bank.py` already exposed
`get_write_stats()` — its interface mirrored here for consistency.

## Verification

Local smoke:

```
write_top_k default: 4
gate bias after init (layer 0): 4.0
gate weight requires_grad: False
sigmoid(bias) ≈ 0.9820137619972229
Stats: {'tokens_per_layer': [12, 0], 'calls_per_layer': [3, 0],
        'tokens_total': 12, 'calls_total': 3, 'tokens_per_call': 4.0}
SMOKE OK
```

- `write_top_k=4` honored (3 calls × 4 = 12 tokens, not 3 × 10 = 30).
- Gate σ(bias) ≈ 0.98 — gate is effectively open.
- Instrumentation counters work and are per-layer.

## Files changed

```
src/memory/sparse/memory_bank.py              — Bug #1 & #3 default values + instrumentation
src/memory/sparse_memory/memory_bank.py       — Bug #1 default value
scripts/train_gated_sparse_memory.py          — Bug #1 CLI default
scripts/train_sparse_memory.py                — Bug #1 CLI default
```

## Open follow-ups

- Task #63: fix double-label-shift bug in 6 other training/eval scripts (pre-existing).
- Properly unfreeze the sparse-memory write gate via the learned-importance pattern
  once `find_unused_parameters` is safe to enable.
- Re-run Phase-1 `train_gated_sparse_memory.py` with the new defaults and compare
  train-loss trajectory vs. the pre-fix runs. Expectation: gate-open + top-8
  writing should let the memory retain useful signal instead of thrashing.
