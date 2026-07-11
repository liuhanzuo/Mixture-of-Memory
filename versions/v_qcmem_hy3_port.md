# QCMem → Hy3 (hy_v3 80-layer MoE) port — v1 (2026-07-11)

Depth-partitioned retrieval read-out (QCMem) adapted to the Tencent Hunyuan
**Hy3** (`hy_v3`) 80-layer sparse-MoE backbone, sharded across GPUs.

## Files
- `src/memory/qcmem/qcmem_hy3.py` — `QCMemHy3Model` (device-aware subclass of
  `QCMemModel`) + `load_hy3_qcmem()` loader.
- `scripts/qcmem_hy3_selftest.py` — tiny-CPU + real-model self-test.

## Env (isolated, does NOT touch the shared `.venv` used by the 24-card QCMem grid)
- `.venv_hy3/` — venv based on the project interpreter (py 3.11.13), with the
  project `.venv` site-packages wired as a *fallback* `.pth` so it inherits the
  L20A-compatible **torch 2.10.0+cu128**, then **transformers==5.13.1** installed
  locally to SHADOW the inherited 5.5.4 (which does NOT know `hy_v3`).
- `.venv_hy3` is NOT committed.

## Architecture (verified against HF `modeling_hy_v3.py`, tf 5.13.1)
- `HYV3ForCausalLM` exposes `model.model.{embed_tokens, layers(80), norm,
  rotary_emb}` + `model.lm_head` + `model.config` — exactly the read-only surface
  QCMem needs. `_no_split_modules=["HYV3DecoderLayer"]` → device_map keeps each
  layer whole on one GPU.
- `HYV3DecoderLayer.forward(hidden_states, attention_mask=, position_ids=,
  past_key_values=, use_cache=, position_embeddings=) -> Tensor` — a DROP-IN match
  for `QCMemModel._run_layers`' existing call; returns a bare residual-stream
  hidden (`h = residual + block(h)`).
- `HYV3RotaryEmbedding.forward(x, position_ids) -> (cos, sin)` — same interface as
  Llama; internally `inv_freq.to(x.device)`. `rope_theta=11158840`, `head_dim=128`,
  64 heads / 8 KV heads, `qk_norm=True` (per-head RMSNorm on q,k).
- `first_k_dense_replace=1` → layer 0 is dense `HYV3MLP`, layers 1..79 sparse
  `HYV3MoE` (192 experts, top-8, +1 shared). Router `HYV3TopKRouter` is a pure
  function of the token hidden (`gate(hidden, e_score_correction_bias)`),
  position-blind → chunk-local WRITE is reproducible.

## The port (WRITE / READ semantics unchanged from `QCMemModel`)
Only device-crossing is new, since Hy3 must be sharded across 8× L20A:
- `_run_layers` overridden: before each layer call, move
  `(hidden, attention_mask, position_ids, position_embeddings)` to that layer's
  parameter device (resolved once from `hf_device_map` / layer params). No-op on a
  single-device model → strict superset of the parent behaviour.
- `norm` / `lm_head` wrapped in `_DeviceMovingCall` so the parent's read tail lands
  the hidden on their GPU without reimplementing `read_core` / `resume_forward_ids`.
- WRITE = `layers[0:j]` chunk-local RoPE (positions `0:T`) → cache `h_j`.
  READ = pack `[sink; ctx h_j...; query h_j]`, fresh contiguous RoPE `0:H`, causal
  mask, resume `layers[j:L]` → norm → lm_head. `j=0` == RAG upper bound (self-test
  gate); `j=L` == closed-book.

## Self-test results (tiny random HYV3, CPU, fp32)
```
(A1) j=0 write/read packing   max|logit diff| = 0.000e+00  PASS
(A2) resume_forward_ids j={0,1,2,4} max|diff| = 0.000e+00  PASS  (incl. dense→MoE L0 boundary)
(B1) MoE-block INPUT hidden max|diff| over RoPE shift = 4.77e-07  PASS (position-invariant)
(B2) discrete expert selection identical across shift: 3/3 layers  PASS
```
Device-hop loop separately validated to match the un-hopped result bit-for-bit
(diff 0.0). Real 597GB device_map="auto" run is **待卡** (all 8 L20A busy with
armB training + QCMem RULER grid at port time).

## Known issues / TODO
- Real-model self-test not yet run on GPU (待卡). Command ready:
  `.venv_hy3/bin/python scripts/qcmem_hy3_selftest.py --model_path <Hy3> --device_map auto --dtype bfloat16 --attn_impl sdpa --tol 1e-2`
- `enable_moe_fp32_combine`: config.json has it `false` (shared-expert combine in
  base dtype). The write/read both re-execute the same MoE forward, so this is
  irrelevant to depth-partition exactness (both sides identical).
- split-j sweep for 80 layers not started (see report §5): suggest bracket
  j ∈ {16, 24, 32, 40} first (semantic-saturation vs LM-tax tradeoff scales with
  depth; the 8B-era sweet spot j≈12/32L ≈ 0.375·L → ~30/80 for Hy3).
