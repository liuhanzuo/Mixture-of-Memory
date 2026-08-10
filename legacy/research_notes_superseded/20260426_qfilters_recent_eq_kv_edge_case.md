# Q-Filters edge case: `recent_window == kv_budget` silently disables the filter

## Setup
Llama-2-7B, pg19, 200×4096, bf16, sdpa, `kv_budget=256`, `filter_rank=1`,
`sub_window_len=1024`; thread-B sweep on b200-2 shows PPL 610.87 → 297.50 → 147.04
as `recent_window` grows 64→128→192, then collapses to **1685.88 at 256**.

## Code-path analysis
- `src/memory/qfilters/layer.py:70-74` — `QFiltersConfig.__post_init__` rejects
  `recent_window > kv_budget` but **allows** `recent_window == kv_budget`.
- `src/memory/qfilters/compression.py:110-111` — `r = min(recent_window, budget, T)`;
  `keep_old = budget - r`. With equality ⇒ `r = 256`, `keep_old = 0`.
- `src/memory/qfilters/compression.py:119-121` — branch
  `if keep_old <= 0: gather_idx = recent_idx`. The filter-scoring path
  (lines 122-131) is **skipped entirely** — no `score_keys` call, no top-k,
  no filter I/O.
- `scripts/eval_qfilters.py:356-361` — CLI passes `recent_window` straight
  through without clamp; no warning emitted.

## Root cause
When `recent_window == kv_budget`, `compress_kv` silently degenerates to pure
sliding-window truncation: each compression step keeps exactly the last 256
post-RoPE tokens and throws everything earlier away. On Llama-2-7B / pg19 this
is catastrophic not because the "0-slot softmax" is malformed (the branch is
clean, no NaN, no duplicate indices) but because **the attention-sink tokens
(positions 0-3) are lost**. At `r=192` the 64 filter-scored old-block slots
reliably capture those sinks (they have the largest projections onto the top
right-singular vector of Q); at `r=256` they vanish and the model loses its
variance anchor — the StreamingLLM-style PPL cliff. Empirically this matches
our prior post-fix SW-b512-r64 Llama-2 result (PPL 1468.97, run
`postfix_llama2_sw_b512_r64`), so 1685.88 at budget 256 is consistent with
"filter disabled → pure SW on Llama-2 pg19," not a new numerical pathology.

## Recommended fix
Two-line, fail-loud at config time **plus** a soft safety floor inside
`compress_kv`:
```python
# src/memory/qfilters/layer.py:70
if self.recent_window >= self.kv_budget:
    raise ValueError(f"recent_window ({self.recent_window}) must be < kv_budget "
                     f"({self.kv_budget}); equality nullifies the filter.")
# src/memory/qfilters/compression.py:110
r = min(recent_window, budget - 1, T)   # keep_old >= 1 so filter always fires
```
The config check surfaces the misuse in sweep configs; the clamp guarantees
that even a future caller bypassing the config never enters the degenerate
branch.

## Follow-up experiment
Single 5-point sweep on b200-2 (same config as thread B, ~5 min wallclock):
`recent_window ∈ {240, 248, 252, 254, 255}`. If PPL stays near 147 up to
`r=255` and only jumps at `r=256`, the jump is **abrupt at equality** — filter
disabling alone explains it, and the one-line config guard is sufficient.
If PPL rises monotonically (e.g. 240→200, 248→400, 255→1000), the regression
is **gradual capacity exhaustion** and we must keep `kv_budget ≥ 2·recent_window`
as a protocol rule, not just a guard.
