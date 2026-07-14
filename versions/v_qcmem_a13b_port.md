# QCMem → Hunyuan-A13B (public 32-layer HunYuanMoEV1) j-sweep port — v1 (2026-07-14)

Ports the QCMem depth-partition **j-sweep** (the tool that determines *split-j*)
from the 80-layer internal Hy3 (`HYV3`) backbone to the **public 32-layer
Hunyuan-A13B-Pretrain** (`HunYuanMoEV1`, `model_type=hunyuan`, 64 experts,
top-8, +1 shared, GQA 32/8 heads, head_dim 128, vocab 128167). Run on `lhz`
(8× H200 143 GB) with `.venv_hy3` (transformers 5.13.1 + torch 2.8-nv + tiktoken).

## Files
- `src/memory/qcmem/qcmem_hy3.py` — **added** `load_a13b_qcmem()`. The existing
  `load_hy3_qcmem()` (80-layer HYV3) and the `QCMemHy3Model` class are left
  **unchanged**. `QCMemHy3Model` is architecture-generic (device-aware QCMem that
  only reads `model.model.{embed_tokens,layers,norm,rotary_emb}` + `model.lm_head`
  + `model.config` and hops tensors across the shard boundary), so it is reused
  as-is for A13B — verified the HunYuanMoEV1 interface matches exactly.
- `scripts/qcmem_a13b_jsweep.py` — **new** sibling of `scripts/qcmem_hy3_jsweep.py`.
  Same metric defs / control flow; differs only in (a) model family (A13B via
  `load_a13b_qcmem`), (b) data source: a pre-tokenised Hunyuan `.npy`
  (`data/slimpajama_val_2048_hunyuan.npy`, shape (4442,2048) uint32) sliced into
  windows — the ids are already Hunyuan-vocab so no tokeniser round-trip, and it
  sidesteps the missing `data/pg19_train.jsonl`. `--j_list` defaults to a 32-layer
  bracket around the predicted 0.4·L.

## Why the class is shared and only the loader is new (verified tf 5.13.1)
`HunYuanMoEV1DecoderLayer.forward(hidden_states, attention_mask=, position_ids=,
past_key_values=, use_cache=False, position_embeddings=) -> Tensor` — the exact
kwargs `QCMemModel._run_layers` passes; returns a BARE residual-stream hidden.
Module tree, `HunYuanMoEV1RotaryEmbedding.forward(x, position_ids)`, and
`create_causal_mask(config, inputs_embeds, attention_mask, past_key_values,
position_ids, ...)` all match what the parent already calls.
`_no_split_modules=['HunYuanMoEV1DecoderLayer']` → device_map keeps each layer
whole for the per-layer device hop.

## The three A13B loading gotchas `load_a13b_qcmem()` handles (from probe2)
(a) `HunYuanMoEV1Config.from_pretrained` leaves `head_dim=None` (checkpoint only
    carries `attention_head_dim=128`) → native attn does `head_dim**-0.5` and
    crashes → set `cfg.head_dim=128`.
(b) on-disk `model_type="hunyuan"` → force `cfg.model_type="hunyuan_v1_moe"` so the
    native class is instantiated (no trust_remote_code).
(c) ★ `experts_implementation="eager"`: default fused `grouped_mm_experts_forward`
    (torch 2.8-nv) trips `GroupMMCommon.cuh:51 delta%16==0`; the eager per-expert
    `index_add_` loop is numerically equivalent and alignment-free. Only verified
    stable path.
Note this is NOT the failed `probe_minimal_arch_hunyuan.py` logit-lens/hard-truncation
probe (which breaks on Hunyuan massive activations). QCMem feeds `h_j` to
`layers[j:]` for a normal recompute → never off-manifold.

## Correctness self-test (tiny random HunYuanMoEV1, CPU, fp32)
```
(A1) j=0 write/read packing       max|logit diff| = 0.000e+00  PASS
(A2) resume_forward_ids j={0,1,2,L//2,L} max|diff| = 0.000e+00  PASS
```
On the REAL A13B (bf16, device_map=auto): j=0 gives ppl_gap=1.000×, KL=0.0000,
top1=1.000 vs the full-context forward → the depth partition is exact, MoE routing
re-executed correctly on resume.

## j-sweep results (real A13B, 8× H200, bf16, all-chunks-selected, 8 docs)

### Short context (chunk 256, ctx=6×256=1536 tok, `logs/a13b_jsweep_results.json`)
```
 j  frac    ppl    gap    KL    top1
 0  0.000  11.124  1.000  0.0000 1.000   ← == full forward (exact)
 4  0.125  11.662  1.048  0.0738 0.860
 8  0.250  12.922  1.162  0.1735 0.795
10  0.312  13.509  1.214  0.2149 0.781
12  0.375  13.917  1.251  0.2619 0.765
13  0.406  14.249  1.281  0.2815 0.761
14  0.438  14.504  1.304  0.3046 0.754
16  0.500  14.757  1.327  0.3267 0.749
20  0.625  15.249  1.371  0.3563 0.742
24  0.750  15.589  1.401  0.3841 0.741
```
Monotonic degradation with j — exactly like Hy3 at short context (where the
"fidelity smile" / mid-depth minimum did NOT appear; it only emerged at 8k/16k).
So short ctx alone does not locate split-j; a long-context run is required (below).

### Long context (chunk 512, ctx ∈ {6,16,32}×512 = 3k/8k/16k, 8 docs)
`logs/a13b_jsweep_longctx.json`. ppl_gap (= QCMem-readout ppl / full-context ppl):
```
 j  frac | ctx=3k gap  top1 | ctx=8k gap  top1 | ctx=16k gap  top1
 0  0.00 |  1.000  1.000    |  1.000  1.000    |  1.000  1.000    (== full forward, exact)
 4  0.12 |  1.060  0.895    |  1.066  0.847    |  1.131  0.834
 8  0.25 |  1.181  0.838    |  1.190  0.772    |  1.353  0.777
10  0.31 |  1.233  0.814    |  1.243  0.767    |  1.425  0.764
12  0.38 |  1.276  0.797    |  1.300  0.749    |  1.472  0.758
13  0.41 |  1.285  0.790    |  1.333  0.731    |  1.544  0.742
14  0.44 |  1.309  0.786    |  1.365  0.725    |  1.601  0.739  ← 16k peak tax
16  0.50 |  1.323  0.786    |  1.385  0.723    |  1.597  0.745  ← 16k dip begins
20  0.62 |  1.364  0.773    |  1.389  0.728    |  1.591  0.748  ← 16k minimum
24  0.75 |  1.390  0.773    |  1.406  0.728    |  1.599  0.747
```

### Verdict: split-j ≈ 13/32 (frac ≈ 0.4·L), matching the prediction.
1. **Short/mid ctx (1.5k, 3k, 8k):** ppl_gap and KL rise ~monotonically with j —
   the same shape Hy3 showed at short context, where its "fidelity smile" did NOT
   appear either. So the strict cheapest-faithful knee is early: the automated hint
   (gap≤1.15× & top1≥0.80) reports j=4 at every ctx, and the low-tax band is
   j≲8 (frac ≲0.25).
2. **Long ctx (16k): the "fidelity smile" emerges** — non-monotonic ppl_gap. It
   *peaks* at j=14 (1.601), then DIPS and flattens across j=16–24 (1.597 / 1.591 /
   1.599) while top1 rises 0.739 → 0.748. I.e. resuming from the MIDDLE of the
   32-layer stack is measurably *more re-composable* than resuming just below it —
   exactly the mid-depth "cacheable-semantic ceiling" QCMem predicts and Hy3 showed
   at 8k/16k. On A13B the smile bottom is broad (j≈16–20, frac 0.5–0.62) and the
   predicted **0.4·L = j≈13** sits right at its shallow-onset; the 8B-era / Hy3
   ≈0.375–0.40·L sweet spot therefore transfers to A13B at ≈ **j=13** (frac 0.4).
3. **Absolute LM tax is high (1.3–1.6× at long ctx)** because this is ZERO-shot (no
   self-distillation) AND all-chunks-selected (no bm25 top-k to shrink the pack) —
   identical caveat to Hy3. The j=0 point matches the full forward at gap 1.000× /
   KL 0.0 / top1 1.000 on the real model, so the tax is purely the depth-partition
   readout gap that split-j self-distillation is expected to close.

**Operating point: split-j ≈ 13/32 (0.4·L)** — the QCMem mid-depth minimum, robust
across the 8B → Hy3-80L → A13B-32L family at ~0.4·L. (Strictly-cheapest-faithful
if minimising zero-shot tax: j≈4–8; the 0.4·L point is the *re-composability*
optimum that distillation targets.)

## Reproduce (on lhz)
```bash
# tiny CPU self-test (correctness gate, j=0 read == full forward)
PYTHONPATH=/volume/haru/Mixture-of-Memory HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  /volume/haru/Mixture-of-Memory/.venv_hy3/bin/python -c "<tiny HunYuanMoEV1 A1/A2 check>"

# real j-sweep, short ctx
PYTHONPATH=/volume/haru/Mixture-of-Memory HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /volume/haru/Mixture-of-Memory/.venv_hy3/bin/python \
  /volume/haru/Mixture-of-Memory/scripts/qcmem_a13b_jsweep.py \
  --model_path /volume/haru/Mixture-of-Memory/models/Hunyuan-A13B-Pretrain \
  --data_path /volume/haru/Mixture-of-Memory/data/slimpajama_val_2048_hunyuan.npy \
  --j_list 0,4,8,10,12,13,14,16,20,24 --chunk_size 256 --num_ctx_chunks 6 \
  --query_len 256 --num_docs 8 --out /volume/haru/Mixture-of-Memory/logs/a13b_jsweep_results.json

# real j-sweep, long ctx (reveals the mid-depth minimum)
#  ... same but: --chunk_size 512 --num_ctx_list 6,16,32 --out .../a13b_jsweep_longctx.json
```
```
```
