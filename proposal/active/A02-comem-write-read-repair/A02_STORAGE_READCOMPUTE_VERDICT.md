# A02 — Storage / Read-Compute Reframe Gate: VERDICT

**Gate run**: 2026-08-09, node `.82` (8× H20, zwfy6 disk), bounded 4-GPU pool
**Driver**: `proposal/active/A02-comem-write-read-repair/code/bench_a02_storage_readcompute.py`
**Launcher**: `.../code/run_a02_storage_readcompute.sh`
**Verdict builder**: `.../code/build_a02_reframe_verdict.py`
**Raw cost data**: `bench_results/a02_storage_readcompute/` (18/18 units, all rc=0)
**Evidence JSON**: `evidence/a02_storage_readcompute_verdict.json`,
`evidence/a02_storage_readcompute_aggregate.json`

**Bottom line, stated first: the reframe SURVIVES ONLY IN ITS WEAK FORM, AND ITS
STRONG FORM IS DEAD.** CoMem does have a finite, trivially small crossover
against the arm phase-1 used as C1 (N\* ≈ 0.13–0.38 queries), but **86–93 % of
that win is retrieval, which plain text-RAG obtains for free without any
precomputed memory.** Against a *matched-pack* text-RAG baseline — the honest
control — CoMem's own contribution is a 1.03–1.37× per-query speedup that needs
N\* ≈ 8–225 queries to repay its Write, while costing **2048× more bytes per
token than raw text**. So "CoMem is a storage method" is **DEAD**; "CoMem is a
modest read-compute optimisation on top of retrieval" **survives**, and it is a
much smaller claim than the kill clause's wording implies.

---

## 1. What was measured

The kill clause prescribed the replacement framing verbatim: *"定位为高复用
workload 的 storage/read-compute 方案"*. `STATUS.json` recorded that reframe as
completely unmeasured. This gate measures it.

**Claim under test**: for a workload where one fixed corpus is queried many
times, CoMem's precomputed memory amortises, so its per-query cost beats
text-RAG even though its quality does not exceed it.

**Decision rule, fixed before the numbers were known** (in the driver docstring
and `evidence/a02_storage_readcompute_verdict.json:decision_rule`):
SURVIVES iff a finite N\* exists against the phase-1 C1 arm, AND N\* ≤ 1e5
queries/corpus, AND the storage premium is bounded enough to still call it a
"storage" method.

> **⚠️ Gate-code defect found and fixed 2026-08-10 — the third conjunct was never
> evaluated.** `code/build_a02_reframe_verdict.py` computed the storage ratios and
> then gated the verdict on `cost_survives` **alone**, so
> `evidence/a02_storage_readcompute_verdict.json` carried a bare
> **`"verdict": "SURVIVES"`** — asserting the *opposite* of this document's own
> "storage form is DEAD" bottom line. Anything reading the JSON instead of this
> prose (a downstream aggregator, a reviewer reproducing the gate) would have been
> told the storage reframe passed. Flagged as A02 retraction item #1 by
> `TCODEX_AUDIT_20260810.md`.
>
> The clause now evaluates against a pre-registered bound
> `PREREG_STORAGE_PREMIUM_MAX = 100.0`, justified in the module docstring on
> principle: RAG stores raw text at ~4 B/token, so 100× already admits ~400
> B/token (≈ a fp16 vector of dim 200 per token); admitting the measured 2048×
> would make the word "storage" vacuous. **The bar was set after the ratios were
> measured, and that is disclosed in the code rather than hidden.**
>
> Regenerated verdict: **`SURVIVES_AS_READ_COMPUTE_ONLY`**, plus a new
> `clause_evaluation` block recording each clause's boolean, the value tested and
> its threshold, so a silently-skipped clause cannot recur:
>
> | clause | passed | tested | threshold |
> |---|---|---|---|
> | (a) finite N\* | ✅ | 12 cells | > 0 cells |
> | (b) N\* reachable | ✅ | 12 cells | > 0 cells with N\* ≤ 1e5 |
> | (c) storage premium bounded | ❌ | **2048.0** | ≤ 100 |
>
> **Every measurement is unchanged.** `verdict_basis`, `cost` and
> `quality_phase1_per_cell` are byte-identical to the pre-fix payload; only
> `verdict` and `decision_rule` changed, and `clause_evaluation` was added. The
> pre-fix JSON said `SURVIVES` while carrying the very
> `storage_ratio_h12_over_raw_range: [2048.0, 2048.0]` that convicts it.

### The arm the published bench was missing

The critical design point. Phase-1's on-disk eval configs show **C1 =
`kvdirect` → `resume_j=0` + `no_retrieval=True` + LoRA dropped**. The
`no_retrieval` branch (`eval_qcmem_babilong.py:650`) sets
`sel_idx = range(len(context_chunks))` — it packs **every** chunk at **full
depth, every query**, so its read is O(L) per query.

The pre-existing `scripts/bench_p1_8_serving_curve.py` (released as Paper A
Table 3) compares CoMem against a `j0` arm that **retrieves top-12** and replays
only those. That is a *different, far cheaper* arm than phase-1's C1. So the
published crossover numbers never answered "does CoMem beat the arm that beat it
on quality?" This gate adds that arm (`c1_all`) and keeps the matched one.

### Three arms, arranged so each adjacent pair differs in ONE thing

| arm | depth | LoRA | pack | per-query read cost |
|---|---|---|---|---|
| `comem` | j=12 | yes | top-12 (iter_bm25) | O(1) in L |
| `j0_top12` | j=0 | no | **same** top-12 pack | O(1) in L |
| `c1_all` = **phase-1 C1** | j=0 | no | **all N chunks** | **O(L)** |

* `comem` vs `j0_top12`: same selector, same pack (asserted equal `read_len`
  6177 and shared `packed_ids_sha256`), same decode → isolates **read depth**.
* `j0_top12` vs `c1_all`: same depth, same absence of LoRA → isolates
  **retrieval vs pack-all**.

This is deliberately *not* the phase-1 comparison, which moved four axes at
once. See §6 for the one confound that remains.

### Cross-check against published numbers
`j0_top12` and `comem` reproduce the released P1.8 artifact to **0.06 % and
0.21 %** respectively (128k|gpu, G=1: read 938.4 vs published 937.87 ms;
677.9 vs 679.34 ms). The new harness is measuring the same thing the paper did.

---

## 2. Crossover N\*

Cost model `cumulative(N) = W + N·(select + fetch + read + decode)`;
`N* = (W_a − W_b)/(perq_b − perq_a)`. Unlike the P1.8 comem-vs-j0 pairing,
**selection does NOT cancel here**: `c1_all` runs no selector at all, so select
is measured (canonical `_select_context_chunk_indices`, imported) and charged
only to the arms that retrieve.

**vs phase-1 C1 (`c1_all`) — the decisive pairing:**

| store L | tier | N\* (G=1) | N\* (G=32) | N\* (G=128) |
|---|---|---|---|---|
| 32k | gpu | **0.35** | 0.35 | 0.35 |
| 32k | cpu | **0.37** | 0.37 | 0.38 |
| 128k | gpu | **0.13** | 0.13 | 0.13 |
| 128k | cpu | **0.13** | 0.13 | 0.13 |
| 1M | gpu/cpu | **N/A — `c1_all` OOMs** | N/A | N/A |

N\* < 1 means CoMem's one-time Write is repaid **before the first query
completes**. Formally 12/12 measured cells pass the pre-registered N\* ≤ 1e5
bar. **But see §5 — this number is nearly meaningless, because it is dominated by
retrieval rather than by memory.**

**vs matched-pack text-RAG (`j0_top12`) — the honest control:**

| store L | tier | N\* (G=1) | N\* (G=32) | N\* (G=128) |
|---|---|---|---|---|
| 32k | gpu | 8.10 | 8.61 | 10.31 |
| 32k | cpu | 8.75 | 9.68 | 11.85 |
| 128k | gpu | 25.15 | 28.35 | 33.54 |
| 128k | cpu | 27.28 | 29.68 | 42.58 |
| 1M | gpu | 185.52 | 209.69 | 219.76 |
| 1M | cpu | 199.85 | 220.03 | 225.45 |

All finite and all well within 1e5. **N\* grows ~linearly in L** (8 → 25 → 186
from 32k → 128k → 1M) because the Write is O(L) while the per-query saving is
constant — so the reuse count needed to break even gets *worse* for exactly the
big corpora the method is pitched at.

**`c1_all` OOMs at 1M** (needed 17.2 GB for one allocation on a 97.8 GB H20; its
peak reached 88.3 GB at 1M vs CoMem's 25.3 GB). This is recorded as
`c1_all_status: "OOM"` and scored as **arm-absent, NOT as an infinite N\* and NOT
as a CoMem win** — a distinction the aggregator and verdict builder now both
enforce explicitly. It is nonetheless a real qualitative result: **pack-all
full-depth is not merely slow at 1M, it does not run at all.**

---

## 3. Storage ratio (real bytes, measured — not estimated)

Measured both as in-memory tensor bytes and as **actual file size after
`torch.save` + `fsync`**:

| quantity | bytes/token | 1M-token corpus |
|---|---|---|
| CoMem h12 store (bf16, d=4096) | **8192** | 8.00 GiB (file: 8 589 936 221 B) |
| raw token IDs (int32) | **4** | 4.00 MiB (file: 4 195 881 B) |
| BM25 index (measured postings) | ~8.4 | 8.81 MB |
| full-depth KV cache (for reference) | 147 456 | 144 GiB |

* **h12 / raw text = 2048× (exact, constant across all L).**
* CoMem-total / RAG-total (store + index both sides) = **632–1129×**.
* On-disk file sizes match in-memory bytes to within a 1577-byte pickle header.

The one direction where CoMem's storage looks good: it is **18× smaller than
caching full-depth KV** (8192 vs 147 456 B/tok). That is a real advantage *over
KV-caching*, and it is the only sense in which "storage method" holds. Against
the baseline that actually matters — the raw text that RAG stores — CoMem needs
**three orders of magnitude more bytes**. A method whose persistent footprint is
2048× the baseline's cannot honestly be sold as a *storage* win.

---

## 4. Per-query latency decomposition (128k, gpu, G=1, ms)

| arm | select | fetch | read | decode | **total** | read_len |
|---|---|---|---|---|---|---|
| `comem` | 117.7 | 1.50 | **677.9** | 0.0 | **797.2** | 6177 |
| `j0_top12` | 117.7 | 0.02 | **938.4** | 0.0 | **1056.2** | 6177 |
| `c1_all` | 0.0 | 0.22 | **53 009.6** | 0.0 | **53 009.8** | 131 105 |

What the memory removes is visible and narrow: **read 938.4 → 677.9 ms, i.e. the
12 skipped bottom layers, a 1.38× saving on the read component of an identical
pack** (which becomes **1.33× on the per-query total**, since the 117.7 ms
selection is paid identically by both arms). Everything else is unchanged. Fetch
is negligible even from pinned host memory (1.5 ms gpu-resident, 5.6 ms
cpu-pinned for the 50 MB pack). Decode is identical across arms by construction
(0.03 % apart), so **larger G dilutes CoMem's advantage**: the CoMem-specific
per-query speedup falls 1.33× → 1.11× → 1.03× as G goes 1 → 32 → 128. At G=128
CoMem is within 3 % of matched-pack RAG.

**`select` is a serious and previously unreported cost.** iter_bm25 selection
over the full store is O(L) on CPU per query: 17.1 ms at 32k, 117.7 ms at 128k,
**1014.8 ms at 1M — 60 % of CoMem's entire per-query budget at 1M, and larger
than its 678 ms read.** CoMem does not remove this; it pays it identically to
RAG. Any "constant per-query cost in L" claim for CoMem is false once selection
is counted, and the published P1.8 crossover let it cancel because both its arms
retrieved.

---

## 5. Why the headline N\* ≈ 0.13 does not rescue the reframe

Decomposing the 66.5× total speedup at 128k, G=1 into its two independent
factors:

| factor | 32k | 128k | who gets it |
|---|---|---|---|
| retrieval (`c1_all` → `j0_top12`) | **7.0×** | **50.2×** | **plain text-RAG, for free** |
| mid-depth memory (`j0_top12` → `comem`) | 1.37× | 1.33× | CoMem only |
| share of log-speedup from retrieval | **86.0 %** | **93.3 %** | — |

The enormous apparent win against phase-1's C1 is almost entirely the
consequence of **C1 being a badly-chosen baseline that re-reads the whole corpus
at full depth on every query.** Fixing only that — keeping plain text, no
precomputed memory, no LoRA, no h12 store, no 2048× storage premium — captures
86–93 % of the benefit. CoMem's own marginal contribution is the residual
1.03–1.37×.

So the honest reading of the reframe:

* **"Storage method": DEAD.** 2048× the bytes of raw text. Only favourable
  against full-depth KV caching (18× better), which is not what RAG does.
* **"Read-compute method for high-reuse workloads": SURVIVES, weakly.** Real,
  reproducible, single-variable 1.33× per-query saving at G=1 (1.38× on the read
  component alone), decaying to 1.03× at G=128, repaid after 8–226 queries
  depending on L, with N\* worsening as L grows.
* **The framing that the data actually supports**: *retrieval* is the dominant
  serving-cost lever; CoMem is a second-order optimisation on top of it.

## 5b. Quality is NOT held fixed — so even the weak win is not free

"Same quality, better cost" requires the quality leg. It does not hold. Per-cell
(never pooled — see §7), C2/CoMem vs C1:

* **BABILong**: qa1×16k −55pp, qa1×32k −51pp, qa2×16k −42pp, qa2×32k −35pp (all
  four CIs entirely below 0); qa5×4k **+16pp**, qa5×16k **+14pp** (both CIs
  entirely above 0); 3 cells n.s.
* **RULER**: −5.17pp, CI [−6.77, −3.70] — C1 wins.
* **LongEval**: −30pp at 8k … **+68pp at 128k** (where C1 scores literally 0.0).
* **LongBench** −0.13pp, **LoCoMo F1** +0.20pp, **LoCoMo judge (open-weight)**
  +1.76pp — all three CIs cross zero (ties).

So a 1.03–1.37× read-compute win is bought at a large accuracy loss on
single-span retrieval at ≥16k, while helping at 128k where the pack-all baseline
collapses. **A cost win at degraded quality is not a win**, which is why this
gate does not upgrade A02's status beyond the weak read-compute claim.

---

## 6. Confounds and UNVERIFIED items

1. **CONFOUND (irreducible): depth and LoRA move together.** `comem` is
   j=12 + Read-LoRA; `j0_top12` is j=0 + no LoRA. A j=12 read without the
   Read-LoRA is not a functional arm, so the 1.33× cannot be split into
   "depth" vs "adapter". For *latency* this is near-harmless (LoRA adds
   negligible FLOPs vs 12 skipped layers), but it is not zero and is not
   separately measured. Phase-1's C1/C2 carried the identical coupling.
2. **CONFOUND (partially remaining): phase-1's four-way confound is reduced,
   not eliminated.** This gate separates {read depth} and {retrieval vs
   pack-all}. The **selector** axis is now held fixed at `iter_bm25` for both
   retrieving arms (phase-1 had C1=`bm25`+no_retrieval vs C2=`iter_bm25`), and
   `c1_all` runs no selector by construction. So phase-1's quality attribution
   to "mid-layer read" is still not de-confounded from retrieval recall — **this
   gate does not resolve that**, and the §5b quality losses may be retrieval
   failures rather than depth failures. Resolving it needs a
   `j0_top12`-quality run, which is a quality gate, not this cost gate.
3. **UNVERIFIED: single example, single task.** All timings come from
   `niah_multikey_1`, `example_index=0`, `read_sample_length=32k`, 3 independent
   processes × 5 repeats. Cost is dominated by tensor shapes rather than content
   so this is defensible for latency, but no cross-task/cross-example variance
   was measured. Matches the released P1.8 protocol.
4. **UNVERIFIED: synthetic store tail.** Beyond the real 32k sample's 62 chunks,
   the store is padded with random-token chunks to reach N (identical
   construction for the h12 store and the raw-token store, so the two describe
   the same corpus). Random ids affect only cost/size axes, never the top-12
   read pack. But **BM25 selection time over random tokens may not match real
   text** — the 1014.8 ms select at 1M is the number I am least confident in,
   and it is load-bearing for §4's conclusion.
5. **UNVERIFIED: `c1_all` at 1M is an OOM, not a timing.** Its per-query cost at
   1M is unmeasured; only "does not fit in 97.8 GB" is established. The 1M row
   of the decisive table is therefore absent, not infinite.
6. **UNVERIFIED: no disaggregated / networked storage tier.** Only
   gpu-resident and cpu-pinned were measured. A real high-reuse deployment would
   read the h12 store from NVMe or over the network, where an 8 GiB/1M-token
   store is far more punishing than 5.4 ms of pinned H2D.
   `scripts/bench_persistent_store_io.py` exists for this and was NOT run here.
7. **Scope**: single-query, batch=1, no continuous batching or paged attention.
   A served RAG system amortises prefill across concurrent requests, which would
   further erode a 1.33× single-stream per-query advantage.
8. **Machine-specific**: H20 (cc 9.0), torch 2.13.0, bf16, sdpa. The L20A nodes
   would give different absolute latencies; the ratios should be more stable.

---

## 7. Aggregation hygiene

Per the standing correction, **no pooled BABILong or LongEval figure is quoted
in this document.** The pooled BABILong −17.89pp averages 4 significantly
negative with 2 significantly positive cells; pooled LongEval +2.00pp averages
−30pp at 8k with +68pp at 128k. Both summarise nothing. The verdict JSON stores
them only under the key `pooled_diff_pt_DO_NOT_QUOTE` so that their exclusion is
auditable, and classifies every cell by CI sign instead.

## 8. Integrity gates (all passed)

| gate | result |
|---|---|
| 1. Read-LoRA sha == flagship `dd09cd17…` | PASS (fail-closed) |
| 2. `comem` / `j0_top12` share ONE pack | PASS (`read_len` 6177 both, all cells) |
| 3. `c1_all` packs exactly `range(N)` | PASS (matches `no_retrieval` verbatim) |
| 4. persistent bytes == N·chunk·d·2 exactly | PASS |
| 5. finite logits on every measured read/decode | PASS |
| 6. store-fetched h12 == fresh recompute | PASS, `max_abs == 0.0` (bit-identical) |
| 7. shard/proc completeness before merge | PASS, 18/18, aggregator aborts on partial |

Two bugs were found and fixed by these gates during bring-up, both in this
gate's own new code: an out-of-bounds `index_select` when
`store_length < read_sample_length` (now a fail-closed GATE 0 with an actionable
message instead of an opaque device-side assert), and a CPU/CUDA device mismatch
in the on-disk probe. A third reporting bug was caught in review — the
aggregator printed `N*=inf` for an arm that had OOM'd, conflating "measured but
never cheaper" with "could not run"; both the console output and the verdict
builder now keep those categories separate.

---

## 9. Verdict

**The reframe SURVIVES in its weak read-compute form and is DEAD in its storage
form.**

* ✅ finite N\* exists vs phase-1 C1 (0.13–0.38) and vs matched-pack RAG (8–226),
  all ≪ the pre-registered 1e5 bar → the *amortisation* precondition holds.
* ❌ storage is **2048× raw text** per token → "storage method" fails. (Only
  18× better than full-depth KV caching.)
* ⚠️ **86–93 % of the apparent win is retrieval**, which text-RAG gets for free.
  CoMem's own share is 1.03–1.37×, shrinking with G and with L.
* ❌ quality is not held fixed (−35 to −55pp on BABILong qa1/qa2 at ≥16k), so
  even the weak win is not a same-quality win.

**Recommended positioning**: do not claim CoMem is a storage method, and do not
claim a large serving win. The defensible claim is narrow: *given that you are
already doing retrieval, caching mid-depth residuals removes the bottom-12-layer
prefill for a 1.03–1.37× per-query saving, repaid after ~8–226 queries per corpus, at
a 2048× storage premium over raw text and with a measurable accuracy cost on
long-context single-span retrieval.* That is a systems micro-optimisation, not a
paper thesis.

**Consequence for A02**: this gate does **not** clear A02 for promotion. The
strong form of the only framing its own kill clause left open has failed. If A02
continues, the honest next question is the one this gate exposed rather than
answered: **selection, not read, is the dominant per-query cost at 1M
(1014.8 ms vs 678 ms)** — so the highest-value remaining work is sublinear
retrieval, which is not CoMem-specific and would benefit text-RAG equally.
