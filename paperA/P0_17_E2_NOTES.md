# P0.17 — E2 overlapping-chunk Write: build + result notes

**Status**: CODE COMPLETE + CPU-validated + **GPU RUN DONE on .104** (zero training).
**Pre-registered PRIMARY target MET**: multikey pooled deployable 92.5 → **99.0** (best
w=128) at **unchanged store/Read cost**, threshold ≥97.0. All widths reported below.

**Task**: conditional follow-up to P0.16, which established E0 (document-contextual
Write control) == A (full replay) == C (continuous oracle) = 100/100/100 while the
DEPLOYABLE chunk-local Arm B = 92.5 pooled (E0−B = +7.5pp, CI[4.0,11.5], McNemar
p=6.1e-5, b=15/c=0). ⇒ the deployable gap is ENTIRELY chunk-local Write lacking
document context; the Read/repositioning is near-lossless. P0.17 injects a small
left-context window into the *persistent* Write while keeping the deployable Read
UNCHANGED.

---

## 1. What E2 does (algorithm)

When encoding a 512-token context chunk to depth 12, PREPEND the `w` tokens
immediately preceding it in the ORIGINAL DOCUMENT (`w ∈ {32,64,128}`), run
layers[0:12) over the `(w+512)` span CHUNK-LOCALLY (isolated causal, RoPE
`0:(w+512)`), then DISCARD the first `w` prefix hidden states and store ONLY the
original 512-token chunk's `h12` (shape `[1,512,4096]`). Everything else — the sink
write, the query write, the store pack layout, the persistent bytes/token, the Read
(fresh contiguous pack positions from layer 12), and the O(1) two-coordinate decode —
is BIT-IDENTICAL to the deployable Arm B. The ONLY change vs B is a one-time,
per-chunk, longer lower-12 Write forward (extra cost = the `w` prefix tokens, which
never enter the store, the pack, the Read, or the decode).

At `w=0`, E2 degenerates to Arm B by construction — proven bit-identical end-to-end.

---

## 2. New files (all under the repo root; NO shared-module edits)

| File | Role |
|------|------|
| `scripts/eval_p017_e2_overlap_write.py` | 6-arm paired harness (standalone; imports P0.16/P1.7/P0.13 verbatim). Modes: `manifest`, `e2_sanity`, `quality`, `latency`, `aggregate`. |
| `scripts/_run_p017_e2_8gpu.sh` | DRY-BY-DEFAULT 8-GPU flock launcher (RUN=1 executes). Path-parameterized. |
| `paperA/P0_17_E2_NOTES.md` | this file. |

**Reuse-not-rewrite**: `import bench_p1_7_h12_oracle as p017` +
`import eval_p016_e0_write_control as p016`, pulling EVERY shared primitive
(`_build_pack`, `_run_arm` for arms A/B, `_run_e0`+`_e0_doc_spans`+`_e0_h12_residual`
for E0, `_load`, `_eos_ids`, stats `_macro_and_cells`/`_pairwise`/`_agree_means`/
`_mcnemar_exact`, provenance/strict-fix hashes, `QCMemModel`, `ruler`, `qcb`). So
**arms A / B / E0 are byte-for-byte the P0.16 / P1.7 / P0.13 headline paths.** The ONLY
new forward is E2's overlap Write (`_e2_write_chunk`, `_e2_left_ctx`, `_run_e2`,
`_write_span_tokens`), which uses only `QCMemModel`'s public low-level accessors
(`embed_tokens`, `_make_mask_and_rope`, `_run_layers`, `write_chunk`, `write_prefill`,
`read_prefill`, `decode_step`) — no backbone patching, no edits to any shared module.

---

## 3. The six arms (differ ONLY in how context-chunk h12 is produced before the SHARED read+decode)

All consume the **identical pack** built ONCE per example by `_build_pack`
(forward-free `iter_bm25` selection ⇒ `resume_j`-independent ⇒ same selected chunk
ids / order / packed token ids / pack sha across arms and vs P0.13/P1.7/P0.16).

- **A** — `resume_j=0` full 36-layer continuous replay + flagship LoRA (RAG upper
  anchor; == P0.16/P1.7/P0.13 A). `p017._run_arm`.
- **B (w0)** — `resume_j=12` DEPLOYABLE chunk-local h12 Write + SAME LoRA (== P0.16 B).
  This IS the E2 `w=0` baseline (numeric identity gate: `_e2_write_chunk(no prefix)`
  == `write_chunk`, max_abs 0). `p017._run_arm`.
- **E2_w32 / E2_w64 / E2_w128** — overlapping Write at left-context widths 32/64/128.
  `_run_e2`.
- **E0** — DOCUMENT-CONTEXTUAL Write control (== P0.16 E0; O(L), cross-query-reusable
  control, NOT a shipping config nor a strict upper bound). The recovery ceiling.
  `p016._run_e0`.

---

## 4. RESULTS (GPU, .104, real Qwen3-8B + flagship LoRA, n=100/cell)

Cohort `niah_multikey_1 × {8k,16k}`, n=200 paired, reusing P1.7's 200 paired examples
(pack sha cross-checked bit-identical, all 200 match).

### Macro (pooled over the 2 cells)
| arm | macro | vs w0 (B) | vs E0 |
|-----|-------|-----------|-------|
| A (full replay) | 100.00 | +7.50 | 0.00 |
| **B (w0 deployable)** | **92.50** | — | −7.50 |
| **E2_w32 (overlap)** | **98.50** | **+6.00** | −1.50 |
| **E2_w64 (overlap)** | **98.50** | **+6.00** | −1.50 |
| **E2_w128 (overlap)** | **99.00** | **+6.50** | −1.00 |
| E0 (doc-ctx control) | 100.00 | +7.50 | — |

### Per cell
| cell | A | B(w0) | w32 | w64 | w128 | E0 |
|------|---|-------|-----|-----|------|-----|
| niah_multikey_1/8k  | 100 | 94 | 99 | 99 | **100** | 100 |
| niah_multikey_1/16k | 100 | 91 | 98 | 98 | 98 | 100 |

### Paired stats (bootstrap 95% CI + exact McNemar), n_boot=10000
| pair | Δmacro | 95% CI | McNemar p | b / c |
|------|--------|--------|-----------|-------|
| A − B | +7.50 | [4.0, 11.5] | 6.1e-5 | 15 / 0 |
| E0 − B | +7.50 | [4.0, 11.5] | 6.1e-5 | 15 / 0 |
| **E2_w32 − B** | **+6.00** | **[3.0, 9.5]** | **4.9e-4** | **12 / 0** |
| **E2_w64 − B** | **+6.00** | **[3.0, 9.5]** | **4.9e-4** | **12 / 0** |
| **E2_w128 − B** | **+6.50** | **[3.5, 10.0]** | **2.4e-4** | **13 / 0** |
| E2_w32 − E0 | −1.50 | [−3.5, 0.0] | 0.25 | 0 / 3 |
| E2_w64 − E0 | −1.50 | [−3.5, 0.0] | 0.25 | 0 / 3 |
| E2_w128 − E0 | −1.00 | [−2.5, 0.0] | 0.50 | 0 / 2 |

**Reading**: every E2 width significantly beats the deployable w0 baseline (all McNemar
b≥12, c=0, CI excludes 0), recovering **~6.0–6.5pp of the +7.5pp E0-vs-B document-context
gap** — i.e. ≈80–87% of the recoverable gap — while leaving a small non-significant
residual to the O(L) E0 control (−1.0 to −1.5pp, McNemar p≥0.25, CI touches 0). Wider
w helps marginally (w128 best, closes the 8k cell to a perfect 100).

### Write cost (the ONLY thing that changes vs w0; store/Read/decode identical)
| arm | mean ctx Write tokens | extra prefix tokens | lower-12 Write FLOPs ratio vs w0 |
|-----|-----------------------|---------------------|-----------------------------------|
| w0 (B) | 6144.0 | 0 | 1.000× |
| w32 | 6496.0 | 352.0 | **1.057×** |
| w64 | 6848.0 | 704.0 | **1.115×** |
| w128 | 7552.0 | 1408.0 | **1.229×** |

⇒ **+6.0pp deployable accuracy for a one-time +5.7% lower-12 Write compute (w32)**;
persistent bytes/token, Read pack, Read compute and decode are unchanged from w0.

### Measured latency (`--mode latency`, .104 GPU0, single-proc, niah_multikey_1/16k, warmup 3 × n_repeat 20)
| arm | Read (ms) | Write (ms) | Write Δ vs w0 |
|-----|-----------|------------|---------------|
| A (full replay, j0) | 957.7 | — | — |
| B (w0 deployable, j12) | 681.9 | 262.1 | — |
| E2_w32 | 679.4 | 311.7 | +18.9% |
| E2_w64 | 677.4 | 324.9 | +24.0% |
| E2_w128 | 677.6 | 346.8 | +32.3% |

**Confirms the design end-to-end at the wall-clock level:** E2 Read is cost-identical to B
(679–682ms vs 681.9ms, within run-to-run noise) — E2 changes ONLY the one-time Write. The
deployable j12 Read (~680ms) is **~29% faster** than full-replay A's Read (957.7ms). The
measured Write overhead (w32 +18.9% … w128 +32.3%) exceeds the marginal-FLOPs ratio
(1.057×…1.229×) because wall-clock includes fixed per-chunk Write overhead beyond the extra
prefix tokens; either way it is a one-time cost that never touches persistent store, Read
pack, Read compute or decode. Per-proc raw:
`bench_results/p0_17_e2_overlap/latency/latency_proc0.json` (the top-level `latency.json`
n_procs=0 is a stale 14:01 pre-run artifact, superseded by this per-proc file).

### Integrity guards (all PASS)
- `packs_paired_1to1 = True` (every arm's read_len == pack_read_len, all 200).
- `p013_pack_sha_all_match = True` (200/200 pack shas == P1.7 manifest).
- `oom = 0`, `nonfinite = 0`.
- **Manifest gate** (real model): LoRA sha `dd09cd17…`, 168 modules, layers [12..35];
  torch 2.13.0 / tf 5.5.4 / peft 0.19.1.
- **e2_sanity gate** (real model, 8k example 0): E0 doc-ctx lower-12 vs stock
  `hidden_states[12]` **max_abs = 0.000e+00** (tol 5e-2); E2 w=0 write vs `write_chunk`
  **max_abs = 0.000e+00** (tol 1e-3) ⇒ the w0 baseline IS the deployable Arm B, exactly.

---

## 5. CPU static validation done (this node, `.venv/bin/python`, tiny random Llama)

- `py_compile` of the harness: **COMPILE_OK** (local + .104). `bash -n` launcher: **OK**.
  DRY launcher prints the full command tree + 8 queued jobs. `--widths` nargs parses
  `32 64 128` correctly through the launcher's word-split.
- Tiny `LlamaForCausalLM` (L=6, resume_j=3, d=64), 7-test E2 suite (`ALL E2 CPU TESTS
  PASSED`):
  - **TEST1** `_e2_write_residual`: `_e2_write_chunk(chunk, None)` == `write_chunk(chunk)`
    → max_abs **0.0** (the w=0 identity).
  - **TEST2** prefix discarded ⇒ stored h shape == `[1, chunk_len, d]`.
  - **TEST3** w=32 vs w=0 chunk-h12 max_abs diff > 0 (left context has real effect).
  - **TEST4** `_e2_left_ctx` provenance (widths, chunk-0 ⇒ None) + fail-closed raise on
    a corrupted pack tensor.
  - **TEST5** `_write_span_tokens` accounting (prefix/total, clamped at doc start).
  - **TEST6** `_run_e2` end-to-end returns the same 6-tuple as `_run_arm`, read_len ==
    expected, finite.
  - **TEST7** `_run_e2` with all-None left-ctx (w=0) == `_run_arm` (Arm B) BIT-IDENTICAL:
    same generations, same read_len, first-logits max_abs **0.0**.
- Full `run_aggregate` on synthetic 6-arm records: per-width pairwise (vs B and vs E0),
  prereg target, write-cost FLOPs ratios, pairing guards all wired correctly.

---

## 6. Exact GPU launch (what was run on .104)

`.104` project root = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`;
PYBIN = `/opt/conda/envs/torch-base/bin/python` (torch 2.13.0 + tf 5.5.4 + peft 0.19.1).
The launcher is DRY unless `RUN=1`.

```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory   # .104 root
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
COHORT=min RUN=1 \
setsid nohup bash scripts/_run_p017_e2_8gpu.sh >logs/p0_17_e2.out 2>&1 &
```
Runs: manifest gate → e2_sanity gate → 8-GPU quality pool (`niah_multikey_1 × {8k,16k}`,
n=100/cell, 6 arms A/B/E2_w32/E2_w64/E2_w128/E0, 4 shards/cell,
`--p013_manifest_dir bench_results/p1_7_h12_oracle`, `--verify` on shard 0) → aggregate.

Optional latency (per-arm write/read/decode timing incl. E2's longer Write):
```bash
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_p017_e2_overlap_write.py --mode latency \
  --model_path models/Qwen3-8b-local --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j_a 0 --resume_j_b 12 --resume_j_e0 12 --widths 32 64 128 \
  --topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --task niah_multikey_1 --length 16k --example_index 0 --warmup 3 --n_repeat 20 --proc_id 0 \
  --output_dir bench_results/p0_17_e2_overlap
```

Outputs on .104: `bench_results/p0_17_e2_overlap/{manifest.json, e2_sanity.json,
summary.json, stats.json, latency.json, quality/<task>_<length>_shard*.jsonl}`.

---

## 7. Interpretation (for MAIN → paper)

E2 overlapping-chunk Write is a **deployable** recovery of the document-context gap:
prepending a small left-context window (as little as w=32 → +5.7% one-time Write FLOPs)
to the chunk-local Write recovers ~80% of the +7.5pp E0-vs-deployable gap
(deployable 92.5 → 98.5–99.0 pooled), **at unchanged persistent store bytes/token, Read
pack and Read/decode compute** — clearing the pre-registered ≥97.0 target. The small
residual to E0 (−1.0 to −1.5pp, not significant) is the part of document context that a
finite left window cannot capture; E0 remains the O(L), non-deployable ceiling. This
promotes E2 (best w=128, or the near-tied cheaper w=32) as a candidate deployable Write
variant to fold into Cohort-B, and confirms the P0.16 attribution: the deployable QCMem
gap was chunk-local Write's missing document context, not the Read repositioning.
