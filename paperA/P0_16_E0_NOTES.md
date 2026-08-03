# P0.16 — E0 document-contextual Write control: build + validation notes

**Status**: CODE COMPLETE + CPU-validated. NO GPU eval run (per task boundary —
MAIN launches the GPU eval on a freed H20). Zero training.

**Task**: build a strictly-paired **4-arm** inference harness that places a new
`E0` document-contextual Write operating point between the two P0.16 endpoints
(`j=0` full replay and the deployable chunk-local `j=12` Write) and against the
P1.7 continuous-pack oracle, to attribute the P0.13 deployable gap between
(i) chunk-independent Write **lacking document context** vs (ii) the Write→Read
**RoPE repositioning**.

---

## 1. Located assets (absolute paths)

### LOCAL (this node — wzc1 `/apdcephfs_wzc1/share_304376610/...`)
- Backbone: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/models/Qwen3-8b-local`
- Flagship LoRA: `/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/outputs/qcmem_distill_qwen_j12_r32_4k/final`
  - `adapter_config.json`: `r=32`, `layers_to_transform=[12..35]`,
    `target_modules=[gate_proj,v_proj,q_proj,o_proj,k_proj,up_proj,down_proj]`
  - LoRA weight sha256 = `dd09cd17457c63578c0f38dab79b287ab5da6e3f14c119aedafec1c34400536f`
    (== `EXPECTED_LORA_SHA`; 168 modules = 24 layers × 7 target modules)
- **P1.7 200 paired examples are NOT on this node** (`bench_results/p1_7_h12_oracle/`
  absent locally; `bench_results/p0_13_quality_latency/` also absent).

### `.104` = `28.83.24.104:36000` (diskB, `/apdcephfs_zwfy6/share_304376610/...`), READ-ONLY probe
SSH: `sshpass -f configs/password_h20_24104.txt ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 -o PreferredAuthentications=password -p 36000 root@28.83.24.104`
- Project root on .104: `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`
- Same `models/Qwen3-8b-local` + `outputs/qcmem_distill_qwen_j12_r32_4k/final` present.
- **P1.7 (#121) per-example JSONL present** (the pack-sha cross-check source, and the
  provenance that these are the SAME 200 paired examples):
  - `bench_results/p1_7_h12_oracle/quality/niah_multikey_1_8k_shard{0..3}of4.jsonl` (100 unique example_id)
  - `bench_results/p1_7_h12_oracle/quality/niah_multikey_1_16k_shard{0..3}of4.jsonl` (100 unique example_id)
  - `bench_results/p1_7_h12_oracle/summary.json`
  - JSONL schema confirmed: fields `example_id`, `packed_ids_sha256`, `retrieved_chunk_ids`,
    `pack_read_len`, `h12_sanity.max_abs=0.0`, `lora_sha256=dd09cd17…` — exactly what
    the E0 harness's `--p013_manifest_dir` cross-check reads.

**⇒ Run the E0 GPU eval on .104** (or any diskB node that has synced
`bench_results/p1_7_h12_oracle/`), so `--p013_manifest_dir bench_results/p1_7_h12_oracle`
resolves and every E0 pack sha is fail-closed cross-checked bit-identical to the P1.7
(== P0.13) pack. If run on a node WITHOUT that dir, the cross-check is skipped (still
correct — the packs are rebuilt from the same seed/selector — but you lose the extra
sha guard); prefer .104.

---

## 2. New files (all under the repo root)

| File | Role |
|------|------|
| `scripts/eval_p016_e0_write_control.py` | 4-arm paired harness (standalone; imports P1.7 verbatim). Modes: `manifest`, `e0_h12_sanity`, `quality`, `latency`, `aggregate`. |
| `scripts/_run_p016_e0_8gpu.sh` | DRY-BY-DEFAULT 8-GPU flock launcher (RUN=1 executes). Path-parameterized. |
| `paperA/P0_16_E0_NOTES.md` | this file. |

**Reuse-not-rewrite**: the harness does `import bench_p1_7_h12_oracle as p017` (which
itself imports `bench_p0_13_quality_latency` verbatim) and pulls EVERY shared
primitive from it — pack builder (`_build_pack`), the P0.13 per-arm generate replica
(`_run_arm`, used for arms A and B), the P1.7 oracle generate (`_run_oracle`, arm C),
loader (`_load`), eos (`_eos_ids`), stats (`_macro_and_cells`, `_pairwise`,
`_agree_means`, `_mcnemar_exact`), provenance/strict-fix hashes, `QCMemModel`,
`ruler`, `qcb`. So **arms A / B / C are byte-for-byte the P0.13 / P1.7 headline
paths.** The ONLY new forward is E0's document-contextual Write branch
(`_e0_doc_lower12`, `_run_e0`, `_e0_doc_spans`, `_e0_h12_residual`), which uses only
`QCMemModel`'s public low-level accessors (`embed_tokens`, `_make_mask_and_rope`,
`_run_layers`, `read_prefill`, `decode_step`) — **no backbone patching, no edits to
any shared module.**

---

## 3. The four arms (differ ONLY in how h12 is produced before the SHARED read+decode)

All four consume the **identical pack** built ONCE per example by `_build_pack`
(forward-free `iter_bm25` selection ⇒ `resume_j`-independent ⇒ same selected chunk
ids / order / packed token ids / pack sha across arms and vs P0.13/P1.7).

- **A — `resume_j=0` full replay** (RAG upper bound; == P0.13/P1.7 A): pack run
  through layers[0:36] continuously with fresh contiguous positions. `p017._run_arm`.
- **B — `resume_j=12` chunk-local h12** (DEPLOYABLE; == P0.13/P1.7 B): each selected
  chunk / sink / query encoded to depth 12 chunk-locally (isolated, RoPE 0:T each),
  h12 repositioned into a fresh contiguous pack, layers[12:36] resume. `p017._run_arm`.
  Included by default (`--no_armB` to drop) so the E0-vs-deployable comparison is
  paired IN THE SAME run.
- **C — continuous-pack oracle** (NOT deployable; == P1.7 C): layers[0:12] run
  continuously/full-causal over the SELECTED PACK (contiguous pack positions 0:H, no
  repositioning), pack-level h12 captured once, layers[12:36] resume. `p017._run_oracle`.
- **E0 — document-contextual Write** (NEW; query-INDEPENDENT, O(L)):

### E0 algorithm (pseudocode)
```
# WRITE (query-independent, O(L) over the whole document; cacheable per document):
doc_ids = full original prompt token ids          # [N]  (original causal order)
emb     = embed_tokens(doc_ids)
pos     = arange(N)                                # DOCUMENT-ORIGIN RoPE positions 0:N
mask,pe = make_mask_and_rope(emb, pos)             # full causal
h12_doc = run_layers(emb, layers[0:12], mask, pos, pe, cache=bottom_cache)  # [1,N,d]
#   -> h12_doc[p] sees tokens[0:p+1]: full preceding DOCUMENT context, query-independent
#      (context chunks precede the query in the doc, so they never see the query).

# SLICE h12_doc at the BM25-selected chunk boundaries into the SAME pack layout as B:
sink_hj      = h12_doc[:, 0:1, :]                         # sink = doc token 0 (=bos for Qwen)
selected_hj  = [ h12_doc[:, i*512 : i*512+len_i, :]       # doc-order selected context chunks
                 for i in sel_idx ]
query_hj     = h12_doc[:, n_ctx_chunks*512 : N, :]        # the last (query) chunk

# READ (deployable interface; == Arm B's repositioning):
logits1, top_cache, H = read_prefill(sink_hj, selected_hj, query_hj)
#   -> packs [sink; selected; query], FRESH CONTIGUOUS pack positions 0:H,
#      resumes layers[12:36]. The store->read RoPE remap (doc coords -> pack coords)
#      is IDENTICAL to Arm B; ONLY the lower-12 attention scope differs (doc-wide vs chunk-local).

# DECODE (O(1)/step, two coordinates, like Arm B):
q_local_pos = N ; pack_pos = H                            # bottom band continues in DOC coords
loop: logits = decode_step(tok, bottom_cache, top_cache, q_local_pos, pack_pos)
      q_local_pos += 1 ; pack_pos += 1                    # top band continues in PACK coords
```

**E0 vs B** (same repositioning, different lower-12 scope) isolates the value of
DOCUMENT CONTEXT at fixed Write→Read repositioning.
**E0 vs C** (both context-aware; C has no repositioning, E0 does) isolates the cost of
the Write→Read REPOSITIONING at fixed context availability.

**Wording (mandated)**: E0 is a "cross-query-reusable document-contextual **control**",
NOT a strict upper bound. Its Write is **O(L)** over the whole document — the harness
reports E0 Write latency + peak memory in `latency` mode, and the notes below record
the long-document position-extension / document-update caveats.

### E0 caveats to report alongside the numbers
- **O(L) Write / re-write on document update**: the document-contextual h12 must be
  (re)computed by a lower-12 forward over the whole document; any document edit
  invalidates the cache from the edit point onward. (Context-chunk h12 IS reusable
  across *queries* on the same document — that is E0's "cross-query-reusable"
  property — but not across document edits.)
- **Long-document position extension**: E0's Write assigns document-origin RoPE
  positions 0:N, so at long N it exercises the backbone's native position range (no
  compression), unlike the deployable pack whose read positions are 0:H≪N.
- **Query representation is document-contextual too**: E0's query h12 (doc positions
  `[q0,q1)`) sees the full preceding document, unlike B's chunk-local query h12 and
  C's pack-contextual query h12. This is intended (E0 = "what if the writer had full
  document context"); it means E0 is NOT deployable as-is (the query slice depends on
  the document forward). E0 remains a control, never reported as a shipping config.

---

## 4. Acceptance hard constraints — how each is enforced (fail-closed)

1. **3 (4) arms strictly paired, 1:1 aligned** — one `_build_pack` per example feeds
   all arms; `run_quality` asserts `rlA == rlC == rlE0 == pack_read_len` (and `rlB`
   when armB on); `aggregate` recomputes `all_packs_paired_1to1` over every kept
   record; dedup by `(task,length,example_id)`.
2. **E0 h12 numerically == stock lower-12 full-document hidden** — `_e0_h12_residual`
   compares E0's OWN `_e0_doc_lower12` output (the exact forward/cache path `_run_e0`
   uses) against the stock model's `output_hidden_states[12]` on the same ids, on a
   document PREFIX (causal ⇒ prefix result == the full-document result restricted to
   those positions). Exposed as `--mode e0_h12_sanity` (launcher STEP 1 gate, aborts
   on fail) and `--verify` (runs the assert on shard 0's first example inside
   `quality`). `assert max_abs < --h12_tol` (default `5e-2` bf16). Because the LoRA is
   on layers 12..35, `hidden_states[12]` is the pure lower-12 forward
   (adapter-independent), so the check is well-defined.
3. **No cohort mixing** — one `(task,length[,shard])` per process; `aggregate` keys
   cells by `(task,length)`; the launcher's COHORT=min = `niah_multikey_1 × {8k,16k}`.
4. **Results → mechanism table regardless of sign** — per-sample JSONL
   (`quality/<task>_<length>[_shardX].jsonl`) + per-cell json + `aggregate` emits
   `summary.json` / `stats.json` / `latency.json` with per-cell + macro means, paired
   bootstrap 95% CI, exact McNemar, agreement, and an `attribution_hint`.
5. **Pack sha == P0.13/P1.7** — `--p013_manifest_dir` cross-checks each example's
   `packed_ids_sha256` against the P1.7 JSONL; mismatch raises (pairing broken).
6. **Slicing parity** — `_e0_doc_spans` asserts each selected chunk span sits strictly
   in the context region `[0, q_start)` and that the reconstructed E0 read_len ==
   `pack_read_len` (fail-closed on any doc-coordinate/slice mismatch).

### Per-sample fields recorded (mandated set)
`doc_len`, `doc_ids_sha256`, `retrieved_chunk_ids`, `n_ctx_chunks`, `chunk_size`,
`e0_doc_slices` (sink/chunk/query document spans), `pack_token_count`,
`pack_read_len`, `packed_ids_sha256`, `p013_pack_sha_match`, `lora_sha256`,
`rope_positions` (E0 Write doc positions `[0,N)`, E0 Read pack positions `[0,H)`, the
query's write-vs-read position ranges = the repositioning, oracle pack positions),
per-arm `prediction` / `score` / `correct` / `gen_len` / `read_len` / `latency_s`
(write+read+decode breakdown) / `peak_gb` / `finite`, per-sample diffs, and per-pair
first-token/decode agreement.

---

## 5. Exact 3(4)-arm GPU launch commands (run on .104 or a diskB node with the P1.7 dir)

`.104` project root = `/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory`;
PYBIN there = `/opt/conda/envs/torch-base/bin/python` (has transformers/peft; the
CLAUDE.md dllm-h20-node memory confirms .104 torch-base was patched). On a wzc1/L20A
node use `.venv/bin/python`. **The launcher is DRY unless `RUN=1`.**

### One-shot (manifest gate → E0 h12 gate → 8-GPU quality pool → aggregate)
```bash
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory   # .104 root
PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
COHORT=min RUN=1 \
setsid nohup bash scripts/_run_p016_e0_8gpu.sh >logs/p0_16_e0.out 2>&1 &
```
This runs the mandated cohort: `niah_multikey_1 × {8k,16k}`, n=100/cell, 4 arms
(A/B/C/E0), 4 shards/cell across the 8-GPU flock pool, `--p013_manifest_dir
bench_results/p1_7_h12_oracle` (pack-sha cross-check), `--verify` on shard 0.

### Preview first (default DRY — no forward), recommended:
```bash
COHORT=min bash scripts/_run_p016_e0_8gpu.sh      # prints every command, runs nothing
```

### Or the individual stages (what the launcher runs):
```bash
PY=/opt/conda/envs/torch-base/bin/python
COMMON="--model_path models/Qwen3-8b-local \
  --lora_adapter outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j_a 0 --resume_j_b 12 --resume_j_c 12 --resume_j_e0 12 \
  --topk 12 --iter_hop_topk 4 --chunk_size 512 --dtype bfloat16 --attn_impl sdpa \
  --seed 42 --output_dir bench_results/p0_16_e0_write_control"

# 0) strict-fix gate (LoRA/backbone hashes, layers [12..35])
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_p016_e0_write_control.py --mode manifest $COMMON

# 1) E0 document-contextual invariant gate (E0 lower-12 == stock lower-12 on a prefix)
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_p016_e0_write_control.py --mode e0_h12_sanity $COMMON \
  --task niah_multikey_1 --length 8k --example_index 0 --h12_tol 5e-2 --h12_check_prefix 1024

# 2) quality (one shard shown; the launcher fans 4 shards × 2 cells across 8 GPUs)
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_p016_e0_write_control.py --mode quality $COMMON \
  --task niah_multikey_1 --length 16k --limit 100 --num_shards 4 --shard_index 0 --verify \
  --h12_tol 5e-2 --h12_check_prefix 1024 --p013_manifest_dir bench_results/p1_7_h12_oracle

# 3) aggregate (CPU-only; run after all shards finish)
$PY scripts/eval_p016_e0_write_control.py --mode aggregate \
  --output_dir bench_results/p0_16_e0_write_control

# optional latency (per-arm write/read/decode timing incl. E0's O(L) Write)
CUDA_VISIBLE_DEVICES=0 $PY scripts/eval_p016_e0_write_control.py --mode latency $COMMON \
  --task niah_multikey_1 --length 16k --example_index 0 --warmup 3 --n_repeat 20 --proc_id 0
```

---

## 6. CPU static validation done (this node, `.venv/bin/python`, tiny random Llama)

- `py_compile` of the harness: **COMPILE_OK**. `bash -n` of the launcher: **OK**.
  Launcher DRY run prints the full command tree + 8 queued jobs correctly.
- Tiny `LlamaForCausalLM` (L=6, resume_j=3, d=64) end-to-end (`QCMEM_E0_CPU_OK`):
  - **TEST1** `_e0_h12_residual`: E0 document-contextual lower-3 forward vs stock
    `hidden_states[3]` on a synthetic document → **max_abs = 0.000e+00** (fp32). This
    is the acceptance invariant, exercised on E0's exact forward/cache path.
  - **TEST2** `_e0_doc_spans`: each selected-chunk document span reconstructs the pack
    tensors bit-identically; query span matches; E0 read_len == pack_read_len.
  - **TEST3** `_run_e0` runs end-to-end (write→slice→read_prefill→decode), returns the
    same 6-tuple as `_run_arm`/`_run_oracle`, `finite=True`, `read_len` == oracle's ==
    `pack_read_len`, first-token logits shape == vocab.
  - **TEST4** fail-closed: a pack with wrong chunk lengths makes `_e0_doc_spans` raise
    `AssertionError` (read_len parity guard fires) — pairing cannot silently break.
  - **TEST5** `_pair_agree` wiring returns the expected agreement keys.
- Import chain (`bench_p1_7_h12_oracle` → `bench_p0_13_quality_latency` → `ruler`,
  `qcb`, `QCMemModel`) resolves; all reused function handles present with matching
  signatures.

## 7. What still needs GPU (MAIN)

- Real Qwen3-8B + flagship LoRA run of all 4 arms on the mandated cohort
  (`niah_multikey_1 × {8k,16k}`, n=100/cell) — the CPU test only validates plumbing +
  the numeric invariant on a tiny model.
- The `--mode manifest` strict-fix gate + `--mode e0_h12_sanity` numeric gate on the
  REAL model (expect max_abs ≈ 0 as in P1.7's `h12_sanity.max_abs=0.0`; tol 5e-2 bf16).
- `--verify` inside quality re-runs the E0 invariant on the first real 8k/16k example.
- `aggregate` → mechanism table rows + the P0.16 attribution decision (P0.17 vs P0.18
  vs both), per the TODOList interpretation rule.

---

## 8. Numeric-check approach (why it is valid)

E0's store pack is **sliced verbatim** from a single continuous lower-12 forward over
the whole document (`_e0_doc_lower12`, `keep_cache=True`). A causal lower-12 forward at
position `p` depends only on `tokens[0:p+1]`, so restricting the full-document forward
to its first `P` positions equals a lower-12 forward over the `P`-token document
prefix. The gate (`_e0_h12_residual`) therefore runs E0's forward on a short prefix and
compares to the stock model's `output_hidden_states[12]` (an INDEPENDENT HF code path
— HF's own masking/RoPE) on the same prefix, asserting bf16 `max_abs < 5e-2`. Because
the flagship LoRA lives on layers 12..35, `hidden_states[12]` is the pure lower-12
forward (adapter-independent), so the reference is exactly the quantity E0 must match.
If the continuous forward matches stock, every sliced chunk/sink/query h12 matches by
construction — the pack is not stitched from per-chunk caches (that is the invariant
that would be violated if E0 were mis-implemented).

---

## 9. TODOList P0.16 spec — concerns / notes (NOT edited; flagged for MAIN)

- **Pack-sha cross-check source path**: the P0.16 spec says "reuse P1.7's 200 paired
  examples". The harness reuses them two ways: (a) it rebuilds packs from the SAME
  seed/task/length/chunk_size/selector (deterministic ⇒ identical samples), and (b) it
  fail-closed cross-checks each pack sha against the P1.7 per-example JSONL via
  `--p013_manifest_dir`. That JSONL lives at `bench_results/p1_7_h12_oracle/` on **.104
  (diskB zwfy6)**, not on this wzc1 node — so the GPU run should be on .104 (or a node
  that has synced it) for the sha guard to be active. Harmless if absent (guard just
  skips), but the guard is worth keeping.
- **"E0 state numerically verified vs stock lower-12 full-document hidden"**: taken
  literally this is an O(L) full-document forward per verified example. The harness
  verifies on a document PREFIX (`--h12_check_prefix`, default 1024) which — by
  causality — is mathematically equivalent to checking the first P positions of the
  full-document hidden, at negligible cost. If MAIN wants the literal full-document
  check, raise `--h12_check_prefix` to ≥ the max doc length (slower, same result).
- **`niah_multikey_1` 8k pack is near-full-document** (observed on .104: 8k example 0
  selects 12 of 14 context chunks, pack_read_len 6560 ≈ doc 7583). E0's advantage over
  the deployable arm may therefore be muted at 8k simply because little context is
  dropped; the 16k cell (and, if E0 separates from an endpoint, the Cohort-B extension
  the spec allows) is where E0-vs-B is most informative. Not a bug — just a
  sensitivity to note when reading the attribution.
- No spec bug found otherwise; the four-arm design, wording limit, and acceptance
  constraints are all implementable as written.
