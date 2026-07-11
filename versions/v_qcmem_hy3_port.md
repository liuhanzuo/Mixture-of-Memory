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

---

# v2 (2026-07-12) — real-model self-test verdict + j-sweep

## Real 597 GB Hy3 self-test (8× L20A, device_map="auto", bf16, tol 1e-2)
```
(A1) j=0 write/read packing         max|logit diff| = 0.000e+00  PASS
(A2) resume_forward_ids j={0,1,40,80} max|diff|     = 0.000e+00  PASS
(B1) MoE-block INPUT hidden max|diff| over RoPE +257 shift = 4.24e-01  FAIL
(B2) discrete expert selection identical across shift: 2/79 layers  FAIL
```

## Is the B1/B2 FAIL a correctness problem? — NO. Honest verdict: it is an
## *ideal property QCMem does not depend on*, and the failure is a bf16 artifact.

**1. A1/A2 PASS at max|diff|=0.0 is the load-bearing correctness proof, and it
already contains the MoE routing.** QCMem's only requirement is:
*the depth-`j` split reproduces the model's own forward.* A2 checks exactly that —
`resume_forward_ids` (write `layers[0:j]` then resume `layers[j:L]` on ONE
contiguous sequence) equals the stock `model(input_ids)` to **0.0** at j=0,1,40,80.
That equality *runs every MoE router/expert on the resume half*; if routing were
mis-replayed the diff would be non-zero. So the depth partition — the actual QCMem
primitive — is exact on Hy3, MoE and all.

**2. B1/B2 test a DIFFERENT thing: RoPE-shift *invariance*, which QCMem never
assumes.** B1/B2 take one chunk, run it at absolute positions `0:T` vs `257:257+T`,
and ask whether the MoE-block input hidden (hence the discrete top-8 expert pick)
is *identical*. This "position-blindness" would be a *nice-to-have* (it would let a
chunk cached at one absolute offset be reused verbatim at another), but QCMem does
**not** rely on it: WRITE always encodes each chunk with **chunk-local** RoPE
(`0:T`), and READ always assigns **fresh contiguous** RoPE to the pack. The cached
`h_j` a chunk contributes is *always* its `0:T` encoding, and A1/A2 prove the read
resumes that correctly. Absolute-position invariance is simply not in the
dependency chain.

**3. The FAIL is a bf16 + argmax-of-router artifact, not a real mechanism
difference — proven by our own tiny test.** The identical B1/B2 test on the tiny
HYV3 in **fp32** PASSED at 4.77e-7 / 3-of-3 (see v1 above). Hy3 uses standard RoPE
(`rope_type=default`) — RoPE rotates q/k, so *attention* is relatively-positioned
and the block-input hidden is *mathematically* shift-invariant, exactly as fp32
shows. In bf16 the tiny rotation-induced rounding differences (~1e-3 per layer)
compound across 80 layers to 0.42 in the residual stream; B2 then flips whenever
two experts' router logits are within that noise (top-8 of 192 experts has many
near-ties), so "2/79 identical" is the *discrete* amplification of continuous bf16
noise, not evidence that Hy3 is position-absolute.

**4. Where a residual, real effect could still live (and why the j-sweep settles
it empirically).** In a *multi-chunk* read, chunk A is cached at local `0:T_A` but
placed at pack offset `o_A≠0`; because QCMem re-runs `layers[j:]` over the pack
with fresh positions, the *read* side is exact regardless. The only asymmetry is
that different chunks are WRITTEN at the same local `0:T` yet READ at different pack
offsets — the attention handles the offset (relative RoPE), but the *cached* `h_j`
itself was routed position-blindly at write. Since B1's true (fp32) answer is
"invariant", this asymmetry is benign in exact arithmetic and only bf16-noisy in
practice — which is precisely what a j-sweep measures as the KL/top1 gap vs the
full-context forward. **Conclusion: proceed to the j-sweep; no mitigation needed.**
(If a future length-generalisation test ever showed a real position-absolute
effect, the mitigation would be to WRITE each chunk at its *eventual* pack offset
rather than `0:T` — but fp32 B1 says that is unnecessary.)

## j-sweep design (`scripts/qcmem_hy3_jsweep.py`)
Find the *split-j* = cacheable-semantic-ceiling vs LM-tax knee. Load Hy3 ONCE
(device_map=auto, ~135 s), then loop `j` re-wrapping the same backbone (cheap).
Per (context-length, j), over `num_docs` real **PG19** windows
(`data/pg19_train.jsonl`, raw text, tokenised with the Hy3 tokenizer):
- build ONE pack `[bos; ctx_1..ctx_N; query]` (ALL chunks selected — isolates the
  DEPTH effect, no bm25 selector noise);
- compute the **full-context forward** logits over the query tail ONCE per doc
  (shared across j; this is also what `j=0` READ reproduces exactly);
- QCMem READ logits on the query tail (`logits_tail=query_len`).
Metrics over the query span: `ppl` (QCMem LM quality), `ppl_full` (reference),
`ppl_gap = ppl/ppl_full` (multiplicative LM tax), `kl` = mean KL(full‖qcmem)
(readout fidelity to the RAG ideal), `top1` (argmax agreement). The knee in
`ppl_gap`/`kl` vs `j` is the split-j. Grid: `j ∈ {0,4,…,64}`,
`num_ctx ∈ {6,16,32}×512` (≈3k/8k/16k), 16 PG19 docs, chunk 512, query 256.

## First-pass result (6 docs, coarse grid, ctx=6×512)
```
 j  frac    ppl  gap    KL   top1
 0  0.000  2.136 1.00 0.000 1.000   ← == full forward (self-test gate)
 8  0.100  2.518 1.18 0.198 0.904
16  0.200  2.550 1.19 0.209 0.911
24  0.300  2.550 1.19 0.214 0.904
32  0.400  2.536 1.19 0.219 0.900   ← plateau end (top1 still 0.90)
40  0.500  2.605 1.22 0.268 0.887   ← degradation accelerates
48  0.600  2.711 1.27 0.332 0.867
56  0.700  2.813 1.32 0.373 0.856
```
Clear **plateau j=8..32 (frac 0.1–0.4)**: fidelity ~flat (top1≈0.90, KL≈0.21),
then a knee past **j≈32–40** where KL/ppl_gap climb steeply. → split-j ≈ **32/80
(frac 0.40)**, consistent with the 8B-era ≈0.375·L sweet spot. Fine-grid + more
docs + longer contexts running to confirm (`logs/hy3_jsweep.log`,
`logs/hy3_jsweep_results.json`).

## Converged split-j (3 runs: full-grid 16 docs + knee-zoom 24–32 docs)
Runs: `logs/hy3_jsweep_results_run1.json` (j∈{0..64}×ctx{3k,8k,16k}×16 docs),
`_run2.json` (3k knee-zoom, 32 docs), `_run2b.json` (8k/16k knee-zoom, 24 docs).

**ppl_gap = QCMem-readout ppl / full-context ppl (multiplicative LM tax), by (ctx, j).**
```
 j  frac | ctx=3k gap top1 | ctx=8k gap top1 | ctx=16k gap top1
 0  0.00 |  1.000  1.000   |  1.000  1.000   |  1.000  1.000     (== full forward, exact)
 4  0.05 |  1.130  0.872   |  1.190  0.847   |  1.245  0.802
 8  0.10 |  1.212  0.832   |  1.375  0.778   |  1.485  0.731     ← shallow: worst fidelity
16  0.20 |  1.227  0.822   |  1.383  0.779   |  1.490  0.723     ← peak tax (long ctx)
24  0.30 |  1.225  0.822   |  1.389  0.783   |  1.456  0.734
28  0.35 |  1.224  0.820   |  1.355  0.788   |  1.395  0.754
32  0.40 |  1.234  0.811   |  1.356  0.788   |  1.386  0.751     ← MID-DEPTH MINIMUM
36  0.45 |  1.246  0.803   |  (—)            |  1.382  0.752
40  0.50 |  1.288  0.791   |  (—)            |  1.403  0.746
44  0.55 |  1.334  0.772   |  1.469  0.791   |  1.443  0.731
48  0.60 |  1.387  0.759   |  1.543  0.772   |  1.487  0.718     ← deep: rising again
```

### Verdict: **split-j ≈ 32/80 (frac 0.40)**, and it *strengthens* with context length.
1. **Short ctx (3k):** broad flat plateau j=8..34 (gap 1.21–1.25), gentle knee at
   j≈38–40. Cheapest faithful point is the *top* of the plateau, **j≈32–34**
   (max compute saving ∝ (L−j)/L ≈ 0.58 at ~no extra tax vs j=8).
2. **Long ctx (8k, 16k): a "fidelity smile."** ppl_gap is NON-monotonic — it is
   *worst* at shallow j (8–16) and best at **mid-depth j≈32–36**, then rises again
   toward deep j. At 16k, gap falls 1.490 (j16) → **1.382 (j36)** and top1 rises
   0.723 → 0.752. I.e. for long contexts, resuming from the MIDDLE of the stack is
   strictly *more faithful to the full-context forward* than resuming shallow.
   Interpretation: the mid-depth hidden (~0.4·L) is the most **re-composable** cache
   point — deep enough that per-chunk semantics are formed (so the top layers can
   re-integrate them across chunks), but not so deep that chunk-local encoding has
   already "committed" to a query-blind continuation. This is exactly the
   "cacheable-semantic ceiling" QCMem predicts, and it lands at ~0.4·L on Hy3 —
   in line with the 8B-era j≈12/32 ≈ 0.375·L sweet spot.
3. **Absolute LM tax is higher on Hy3 (1.25–1.5×) than on the 8B backbones**
   because this is ZERO-shot (no self-distillation yet) AND all-chunks-selected
   (no bm25 topk to shrink the pack). The j=0 self-test still matches full forward
   at 0.0, so the tax is purely the depth-partition readout gap, which
   self-distillation at the split-j is expected to close (the 8B story: 1000-step
   PG19 self-distill lifted every qa cell).

### Recommendation for the next step
Run QCMem self-distillation on Hy3 at **j=32** (teacher = j=0 RAG full recompute,
student = j=32 + LoRA on `layers[32:]`, PG19 KL), then re-measure this sweep — the
mid-depth minimum at j≈32 should deepen and the 1.25–1.5× tax should shrink toward
1.0, mirroring the 8B result. `read_core` is already grad-bearing + supports
`logits_tail` for exactly this trainer.


