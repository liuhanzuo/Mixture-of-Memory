# P0.20 Phase B — dense-retriever equal-latency frontier (design notes)

NEW notes file (2026-08-03). Author: automated P0.20 Phase B implementation pass.
MAIN owns the .tex / TODOList / RUN_REGISTRY; this file is a scratch design record
for the Phase B harness only.

## 1. What Phase B is

Phase B == Phase A (`scripts/eval_p0_20_equal_latency.py`) with **exactly one thing
changed**: the raw-text-RAG arm's retrieval selector is swapped from the lexical
flagship `iter_bm25` to a **frozen public DENSE retriever (BGE-large-en-v1.5,
CLS+L2+cosine)** — the very same retriever used by P1.9 (`scripts/eval_p1_9_dense_rag.py`).

Everything else is held byte-identical to Phase A / config#2:

* Reader (both arms): `models/Qwen3-8b-local`, bf16, sdpa, seed 42, chat_template=False,
  add_special_tokens per base LM (Qwen3 adds no BOS; pack sink=bos prepended explicitly),
  chunk_size=512.
* **CoMem arm — COMPLETELY UNCHANGED from Phase A**: resume_j=12, flagship LoRA
  `outputs/qcmem_distill_qwen_j12_r32_4k/final` (sha dd09cd17…, layers 12..35),
  selector `iter_bm25`, iter_hop_topk=4, context h12 modelled as a pre-stored fetch+H2D
  (GPU-resident AND CPU-pinned reported separately). Its anchor TTFT therefore equals
  Phase A's (≈698.3 ms gpu-resident, k=12).
* text-RAG arm (now "dense-RAG"): resume_j=0, LoRA DISABLED (`peft_model.disable_adapter()`)
  == vanilla Qwen3-8B full 36-layer recompute over the DENSE-selected pack.

The two arms therefore read **different packs** (CoMem = iter_bm25 chunks, dense-RAG =
BGE chunks) — this is intended, it is a full-deployment-vs-full-deployment comparison.
Pairing is at the EXAMPLE level (same document + query for both arms, same provider +
seed as Phase A). Invariant kept: because both arms select exactly `k` full 512-token
context chunks over the same document + query chunk, `read_len = 1 + k*512 + |q|` is
IDENTICAL across arms even though the chunk indices differ — asserted in sanity.

## 2. The two reconciled design points (surfaced per the "report blocker first" rule)

### 2a. "Reuse P1.9 rankings, don't recompute" vs the k-sweep

P1.9's persisted jsonl stores only the **top-12** `dense_sel_idx`, already collapsed to
document order (rank order lost). An equal-latency k-sweep (k = 2..24) is impossible to
reconstruct from that: k<12 loses order, k>12 was never stored.

Resolution (NOT a gate-loosen): reuse P1.9's **frozen `DenseRetriever` class verbatim**
(same weights, same sha256 fail-closed gate, same CLS+L2+cosine contract, deterministic).
Its `select_topk` returns the **full per-index score dict**, so any k's ranking is
recoverable. We recompute the ranking on Phase B's own examples and **prove it reproduces
P1.9** with a fail-closed gate: for families whose sample construction matches P1.9
(babilong / longeval / locomo), the recomputed top-12 (matched by `input_ids_sha256`)
must equal P1.9's stored `dense_sel_idx`. So "reuse P1.9's ranking" is honoured in
substance (identical frozen ranking function, verified identical output) — the recompute
is a *necessity*, not a re-selection with a different retriever.

Caveat: Phase A's RULER provider seeds with `hash((task,length))` while P1.9's `iter_ruler`
seeds with `zlib.crc32(...)` → RULER example i differs between the two. Phase B builds
examples with **Phase A's providers** (so CoMem is byte-identical to Phase A and both
arms see the same example) → the P1.9 cross-check gate is applied only to
babilong/longeval/locomo (where the seed convention matches) and skipped-with-warning
for RULER. RULER quality is still valid: it is paired within Phase B and identical to
Phase A's CoMem cohort.

### 2b. Dense retrieval latency accounting in the ±5% band

Dense retrieval has two honest cost models; we report BOTH and freeze on the
deployment one (apples-to-apples with CoMem's pre-stored h12):

* **deployment (PRIMARY)** — passages pre-encoded & indexed OFFLINE (exactly as CoMem
  pre-stores h12 offline). Online cost charged into TTFT = query-encode + vector search
  (cosine matmul over the pre-built index + top-k). This is the fair parallel to CoMem's
  fetch+H2D of pre-stored state.
* **cold-index (SENSITIVITY)** — no offline index; online cost = encode ALL context
  passages + query + search (== P1.9's `retrieval_latency_ms`, ≈1082 ms for qa1@16k).
  Reported honestly; at this cost the ±5% band around the 698 ms CoMem anchor is already
  blown by the encode alone → k_dense* ≈ 0 (documented, not hidden).

Storage analog reported alongside: dense index size (n_ctx·hidden·dtype_bytes) vs CoMem
h12 store size (k·4 MiB), mirroring P1.9's index-size accounting.

`k_dense*` = the largest dense-RAG k whose **deployment** TTFT lands in the pre-registered
CoMem(k=12) ±5% band. Cold-index freeze reported as a second number.

## 3. Files (all NEW; shared modules imported, never edited)

* `scripts/eval_p0_20_phaseB_dense.py` — driver. Imports Phase A driver
  (`eval_p0_20_equal_latency`) for all reused helpers (`_load_with_peft`,
  `_provider_*`, `_build_calib_pack`, `_make_store_fetchers`, `_freeze_k`, `_timeit`,
  stats) and P1.9's `DenseRetriever` + provenance constants. Adds: dense pack builder,
  dense selection timers (deploy + cold-index), dense reproduction gate, dense-vs-CoMem
  quality + aggregate.
* `scripts/_run_p0_20_phaseB_dense.sh` — DRY-by-default 8-GPU launcher (RUN=1 to execute).
* `paperA/P0_20_PHASEB_NOTES.md` — this file.

## 4. Fail-closed gates (all preserved; NONE loosened)

manifest: backbone key-tensor sha + LoRA sha (dd09cd17…) + layers[12..35] + 168 modules
(imported from P0.13) **AND** BGE weight sha256 == P1.9's EXPECTED_BGE_SHA256.
sanity: LoRA disable_adapter toggle; calib/quality split disjoint (calib_offset ≥ limit);
dense determinism (recompute-twice identical); P1.9 reproduction cross-check (where seed
matches); read_len arm-equality; finite logits.
quality: finite-logits per arm; per-example provenance (input_ids_sha256, dense_sel_sha,
comem_sel_sha, lora_sha).
aggregate: pairs only on shared example ids; verdict reported for ALL cells (never
dense-wins-only). k_dense* frozen on latency ONLY, never after seeing quality.
