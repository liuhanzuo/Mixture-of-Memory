# CacheBlend baseline — implementation notes (Paper A #143 / A-P1.3)

> Companion to `paperA/CACHEBLEND_BASELINE_DESIGN.md` (the read-only scout plan).
> This file records the AS-BUILT implementation: what was added, the self-test
> result, and the launch recipe. No `.tex` / `TODOList` / `status/` file was
> touched. Nothing here has been run on GPU at scale — MAIN launches the real
> eval when a diskB node frees.

CacheBlend (Yao et al., **EuroSys'25**, **arXiv:2405.16444**) = full-depth
per-chunk KV reuse + global RoPE reindex + **selective boundary-token recompute**
(the load-bearing "HKVD" step). Single-variable control vs flagship CoMem: SAME
selector (iter_bm25, hop=4), SAME chunk=512 / topk=12 / sink=bos / pack order —
the **ONLY** difference is the cache object: full 36-layer KV (**144 KiB/tok**,
GQA-correct `2·36·8·128·2 B`) vs one depth-12 residual `h_j` (**8 KiB/tok**). 18×.

## 1. What was added (additive only — no existing branch touched)

### `src/memory/qcmem/qcmem_model.py` (new methods on `QCMemModel`, public accessors only)
- `cacheblend_kv_bytes_per_tok(dtype_bytes=2)` → `2·num_layers·n_kv·head_dim·dtype_bytes`
  = **147456** for Qwen3-8B bf16.
- `prefill_chunk_full(token_ids, rope_start=0)` — mirrors `write_prefill` but over
  the FULL band `slice(0, L)` with a `DynamicCache` (`use_cache=True`), chunk-local
  causal mask, RoPE positions `arange(T)+rope_start`. Returns `(kv_layers, T)` with
  `kv_layers[l] = (K,V)` per layer (post-RoPE K, as HF stores it).
- `concat_kv_reindex(chunk_kv_list, chunk_offsets)` — per layer: K reindexed
  local→global by `_rotate_k_by_offset`, V untouched, concatenated in pack order.
- `_rotate_half`, `_rope_delta_cos_sin`, `_rotate_k_by_offset` — the RoPE reindex
  primitives (see §3).
- `cacheblend_read(pack_ids, merged_kv, sink_len, query_len, recompute_ratio, stats)`
  — bootstrap layer 0 full over all H tokens → deviation-rank context tokens →
  `R = sink ∪ query ∪ top-⌈r·n_ctx⌉` → forward only `|R|` tokens through layers
  `1..L-1` against the blended cache (R positions overwritten fresh via a duck-typed
  `_CacheBlendSparseCache`). Returns `(logits_R, R_idx, mixed)`; fills `stats` with
  `cacheblend_kv_bytes_per_tok` / `recompute_ratio` / `n_recompute_ctx` / etc.
- `cacheblend_decode_cache(mixed)` + `cacheblend_decode_step(...)` — seed a standard
  `DynamicCache` from the blended full-H cache, then plain single-token decode over
  all L layers.
- `_CacheBlendSparseCache` — minimal duck-typed cache exposing only `.update()`
  (safe: Qwen3 attention/decoder only call `.update()`); overwrites the R-subset
  rows with freshly-computed K/V and returns the full (K,V) for attention.

### `scripts/eval_qcmem_babilong.py`
- `cacheblend_generate(...)` and `run_cacheblend_self_test(model, tokenizer, device)`
  (module-level; imported by the RULER/LoCoMo drivers).
- `main()`: added `cacheblend` to `--baseline` choices + `--recompute_ratio` arg;
  `no_retrieval = (baseline in {kvdirect,hcache})` so cacheblend KEEPS retrieval;
  a cacheblend resolution branch (drop LoRA, validate `r∈[0,1]`); self_test dispatch;
  generate dispatch; per-cell `cacheblend` block in the cell JSON
  (`recompute_ratio` / `kv_bytes_per_tok` / `avg_prefill_latency_ms` / `peak_mem`).

### `scripts/eval_ruler_qcmem.py` and `scripts/eval_qcmem_locomo.py`
- Both: import `cacheblend_generate` / `run_cacheblend_self_test` from the babilong
  driver; add `cacheblend` to `--baseline` choices + `--recompute_ratio`; fix
  `no_retrieval` to exclude cacheblend; add the cacheblend resolution branch;
  dispatch self_test + generate to the cacheblend path; emit the additive
  efficiency fields (RULER → cell summary + cfg JSON; LoCoMo → per-sample fields +
  a `cacheblend_efficiency{shard_tag}.json`).
- The existing `none` / `kvdirect` / `hcache` / CoMem branches are byte-unchanged.

### `scripts/_run_cacheblend_8gpu.sh` (new)
DRY-by-default (`RUN=1` to execute) 8-GPU flock task-pool, mirroring
`_run_p0_19_ruler_paired.sh`. STEP 0 = self-test gate (aborts on failure); STEP 1
builds a `(bench|task|length|r|shard)` queue; STEP 2 = 8 GPU workers pop jobs.

## 2. Self-test result (CPU, fp32, tiny random Qwen3 — strict 1e-3 gate)

Ran `run_cacheblend_self_test` on a tiny random Qwen3 (L=4, GQA n_kv=2, head_dim=16,
`rope_theta=1e6`), CPU, float32:

```
(A) reindex  max|dK|=1.907e-06  max|dV|=2.459e-07   PASS
(B) r=1.0 vs full prefill  max|logit diff|=0.000e+00  top1_agree=100.00%  R=all:True  PASS
(C) r=0.0 finite (no NaN): True   PASS
CACHEBLEND SELF-TEST: ALL PASS
```

- **(A)** proves the RoPE reindex is numerically exact (chunk-local prefill + delta
  rotate == direct prefill at the global offset).
- **(B)** proves KV concat + reindex + selective recompute reduces to the vanilla
  full prefill token-by-token at `r=1.0` (`max|logit diff|=0`, `R` = all tokens).
- **(C)** the pure-reuse floor (`r=0.0`) is finite.

The driver `--self_test` runs the SAME gate on the real Qwen3-8B backbone (forces
fp32); it is wired as STEP 0 of the launch script and aborts the run on failure.

## 3. RoPE reindex correctness (the classic TurboRAG/PIC failure point)

RoPE is a rotation, so `R(a)·R(b) = R(a+b)`. HF stores **post-RoPE** K at chunk-local
positions `0:T`. To move a chunk to global pack offset `Δ`, apply one extra uniform
rotation by `Δ` to every token of the chunk:

```
k_global = k_local · cos(Δ) + rotate_half(k_local) · sin(Δ)
```

- `cos/sin` come from `self.rotary_emb(x, position_ids=[[Δ]])`. HF's `rotary_emb`
  multiplies cos/sin by `attention_scaling`; `_rope_delta_cos_sin` **divides it out**
  so the composition is a pure rotation. (For Qwen3 default `attention_scaling=1.0`
  this is a no-op, but it is REQUIRED for YaRN/`rope_scaling≠None`.)
- **V is not rotated.** Offset 0 (the BOS sink at pack pos 0) is the identity.
- Confirmed exact by self-test (A): `max|dK| ~ 2e-6` (fp32 roundoff), `max|dV| ~ 2e-7`.

## 4. Launch recipe (MAIN, on a FREE diskB torch-base node)

```bash
PROJECT_ROOT=<PROJECT_ROOT> \
PYTHON_BIN=python \
RUN=1 setsid nohup bash scripts/_run_cacheblend_8gpu.sh \
  >logs/cacheblend/sched.out 2>&1 &
# DRY preview (default): omit RUN=1 — prints the 608 shard-jobs + commands, no forward.
```

- Config (fixed): Qwen3-8B `models/Qwen3-8b-local`, selector=iter_bm25, topk=12,
  iter_hop_topk=4, chunk=512, sink=bos, **chat_template=False** (BASE LM),
  enable_thinking=False, bf16, sdpa, seed=42, n=100/cell.
- Targets: RULER {niah_single_2, niah_multikey_1, variable_tracking} × {4k,8k,16k,32k}
  + LoCoMo full + BABILong qa5 × {0k…16k}; recompute-ratio sweep `{0.0,0.10,0.15,0.18}`
  (r=1.0 = self-test/full-prefill ceiling, not swept).
- Results → `bench_results/cacheblend/`. Merge shards with `score_nested_babilong.py`
  (RULER/BABILong) or `--score_only` (LoCoMo); read `cacheblend_kv_bytes_per_tok`
  (=147456) / `prefill_latency_ms` / `peak_mem` / `recompute_ratio` per cell.
- ⚠ Full-KV memory ≈ 144 KiB/tok (19.3 GB at 128k); RULER 32k is fine on one H20.

## 5. Storage accounting to report (from the design doc §2.5)

CacheBlend caches the SAME bytes as a full KV cache — it does **NOT** compress
storage; it wins only on prefill/TTFT. Per-token: **144 KiB** (full 36-layer KV,
GQA) vs CoMem **8 KiB** (depth-12 residual). At a 1 GiB budget: CacheBlend ≈ **7.1k
tokens** vs CoMem **128k** — ~18× fewer context tokens under equal storage. The fair
table must show the 144 KiB/tok tier next to any prefill-latency win and never file
CacheBlend as storage-saving.
