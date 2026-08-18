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

## 5. RESULTS (GPU run on .104, 2026-08-03; n=9000 paired = 9 cells × k∈{2..24})

Run COMPLETE 21:07 GMT+8. Outputs on .104 diskB:
`bench_results/p0_20_phaseB_dense/{decision.json, summary.json}`,
aggregate log `logs/p0_20_phaseB/aggregate.out`.

### 5a. Headline (pre-registered anchor: CoMem@k=12 vs latency-matched dense-RAG)
* **k_dense\* = 10** (deployment model, both gpu-resident and cpu-pinned) — the largest
  dense-RAG k whose deployment TTFT lands in CoMem(k=12)'s ±5% band.
* **PRIMARY (deployment): CoMem = 53.22, dense-RAG@k\*=10 = 54.22, diff(CoMem−dense) = −1.0pp,
  95%CI [−4.667, 2.667], McNemar p=0.637 → STATISTICAL TIE** (gpu-resident == cpu-pinned).
* **Cold-index (sensitivity): k_dense\* = None** — encoding all passages online alone blows
  the ±5% band around the 698 ms CoMem anchor, so dense-RAG cannot match CoMem's TTFT at
  ANY k (documented, not hidden).
* Reference (NOT latency-matched, both read k=12): CoMem 53.22 vs dense 58.56, diff −5.33,
  CI [−8.889, −1.778], p=0.00387 — dense wins ONLY when latency is ignored.
* **VERDICT: MIXED** — dense-RAG ahead on point estimate (+1.0pp) but CI includes 0.

### 5b. Phase A → Phase B (the selector-dependence story)
CoMem arm is byte-identical across phases (both 53.22). Swapping ONLY the text-RAG selector:
* Phase A (lexical iter_bm25): text-RAG@k\*=10 = **64.78** → CoMem −11.56pp, CI[−14.444,−8.667],
  McNemar b=41 — **text-RAG WON significantly**.
* Phase B (dense BGE-large): dense-RAG@k\*=10 = **54.22** → CoMem −1.0pp, ns — **TIE**.
The general dense retriever retrieves *worse* than lexical BM25 on this needle-heavy cohort
(64.78 → 54.22, a 10.6pp drop), collapsing text-RAG's Phase-A edge to a tie. ⇒ the
equal-latency verdict is **selector-dependent**; CoMem's best case vs a deployable text-RAG
is a tie (dense), not a win. Do NOT conflate with P1.8 (latency-amortization).

### 5c. Per-cell structural finding (k-curve shape)
CoMem accuracy is **non-monotonic in k** (peaks at low k, degrades as k grows); dense-RAG is
**monotone-increasing** in k. Examples (comem / dense_rag by k=2..24):
* LongEval-16k CoMem: 96 95 97 93 85 70 58 47 43 34 ; dense: 15 28 34 44 54 57 62 66 79 90
* RULER-mk1-16k CoMem: 98 96 97 97 92 91 92 88 88 88 ; dense: 32 48 60 64 72 79 81 86 91 97
* LoCoMo ~tied and low both (14–18) throughout.
At TIGHT budgets (low k ⇒ low latency) CoMem dominates hugely (LongEval-16k k=2 +81pp;
RULER-16k k=2 +66pp); dense only overtakes at high k. The crossover sits ≈k=12–14 on these
cells, so the flagship CoMem@12 anchor lands right at/just past it ⇒ the aggregate tie.
Honest reading: CoMem's structural advantage is at tight latency budgets; k=12 is a
conservative (not cherry-picked) operating point — we do NOT re-select k post-hoc.

### 5d. Provenance / integrity
All gates PASSED (manifest backbone+LoRA sha dd09cd17… layers[12..35] 168 mods + BGE sha;
sanity LoRA-toggle + dense determinism + P1.9 repro + split-disjoint). One harness bug found
and fixed mid-run (commit `306ccbe`): the read_len 4-way equal-arm assert crashed on
babilong-qa1 where iter_bm25 under-fills below k (positive-overlap filter + early break) while
dense returns exactly min(k,n); relaxed to per-arm self-consistency + additive
`comem_read_len`/`dense_read_len` fields (equal-latency anchors on TTFT, not equal read_len).
Backward-compatible: 67 pre-fix rc=0 shards preserved; only the 8 crashed qa1 shard-1 cells
(4k:k4,k8; 16k:k4,k8,k12,k16,k20,k24) re-ran with the fixed code → all 8 rc=0, aggregate clean.
