# S5 readout-axis patch (single-layer landmark grouped-softmax)

**Axis (Part Y only):** restrict the landmark `grouped_softmax` readout to a single
decoder layer; every other layer normalizes the *same* attention weights (same
retrieval + KV + landmark tokens) with a plain causal softmax. No change to KV
caching, retrieval, windowing, or the data pipeline → does NOT touch S6.

## Files
- `llama_mem.S5.py` — full modified `external/landmark-attention/llama/llama_mem.py`
  (built from the pristine anchor = nested-repo HEAD d963e50, md5 99631a8).
- `llama_landmark_config.S5.py` — full modified `llama_landmark_config.py`
  (adds `single_layer_mem=None` config field).
- `llama_mem.S5.diff`, `llama_landmark_config.S5.diff` — unified diffs vs the anchor.
- `s5_smoke_worker.py`, `run_s5_smoke_remote.sh` — CPU-only smoke harness.
- `apply_s5.sh` — apply the S5 readout gate onto a live `llama/` package by diff
  (so it can be layered on top of whatever base is current when S5 launches).

## What the diff does (≈30 lines)
1. Config: new `single_layer_mem` field, default `None` = all-layer anchor.
2. `LlamaAttention.__init__(config, layer_idx=None)` stores `layer_idx` +
   `config.single_layer_mem`.
3. `LlamaDecoderLayer.__init__(config, layer_idx=None)` and
   `LlamaModel.__init__` thread the enumerate index in.
4. At the normalization site (the old `if is_mem is None: raise ValueError` /
   `else: grouped_softmax`): gate
   `use_grouped = (single_layer_mem is None) or (layer_idx == single_layer_mem)`.
   - `use_grouped` → `landmark_grouped_softmax` (anchor path).
   - else → `nn.functional.softmax` over the same `attn_weights` (incl. retrieved
     prefix cols). The dead `softmax` line the original authors already wrote is
     now the real plain-softmax path. The `ValueError` stays as a guard above it.
5. Optional instrumentation: env `LM_S5_DEBUG_COUNTER=1` records which layers took
   the grouped vs plain branch (no numeric effect; default-off no-op).

## Smoke result (CPU-only, diskB, isolated `_s5_smoke/` dir)
- **Regression**: `single_layer_mem=None` vs unmodified anchor forward —
  **max-abs-diff = 0.000e+00** (byte-identical). ✅
- `single_layer_mem=None`: grouped_layers `[0,1,2,3]`, plain `[]` (4-layer toy). ✅
- `single_layer_mem=1`: grouped_layers `[1]`, plain `[0,2,3]`, **no ValueError**,
  output changes vs anchor (max-abs-diff 0.403). ✅

## ⚠ Collision with S4b
`external/landmark-attention/llama/llama_mem.py` is gitignored vendored code (a
nested repo) and is the SAME physical file landmark-s5's S4b
(`learned_block_gate`) edits. S4b is the live working-tree version. Do NOT
overwrite the shared file with `llama_mem.S5.py`. When S5 launches, either:
- (a) layer the S5 gate onto the then-current base via `apply_s5.sh` (the two axes
  are orthogonal: S4b adds a learned logit *before* grouped-softmax; S5 chooses
  *whether* grouped-softmax runs per-layer — they compose cleanly), or
- (b) run S5 on a fresh anchor checkout in a separate package dir.

## Launch (later, when a 16-GPU group frees AND team-lead confirms)
`external/landmark/train_landmark.sh` already threads `LM_SINGLE_LAYER` via env;
set the launcher to pass `single_layer_mem` into `from_pretrained` (config kwarg)
— see RUN_REGISTRY S5 [READY] row.
