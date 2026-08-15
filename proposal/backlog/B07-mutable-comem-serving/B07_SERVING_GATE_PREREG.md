# B07 — Kill gate + next gate, PRE-REGISTERED (written 2026-08-14, 0 GPU, PRE-DATA)

> ## ⚠️ SUPERSEDED 2026-08-15 by `B07_GATE_PREREG.md` (REV-1). Read that first.
>
> **This text is retained unedited** per `proposal/README.md` («改写 gate，并在 proposal 里
> 记下为什么改（旧文本保留、标 SUPERSEDED）»). **Every number below was re-derived from the
> raw per-process artefacts on 2026-08-15 and is arithmetically CORRECT.** What is
> superseded is *which statistic the gate is scored on*:
>
> | clause | disposition in REV-1 |
> |---|---|
> | **K1** (TTFT **p99**, "45 **paired** draws") | **statistic replaced.** p99 at n=45 **is the maximum order statistic** (`ceil(0.99·44)=44`), so it is a single observation, not a percentile; and the draws are **not paired** — `bench_p1_8_serving_curve.py:517` and `:524` call `_measure` once per arm in two sequential loops. REV-1 gates the **Δ of TTFT medians** with an **unpaired** bootstrap. **The 123.2 ms threshold is unchanged.** |
> | **K2** (G=128 **total-latency** margin vs dispersion, 1.09×) | **replaced.** 71 % of that margin is *decode*, which is identical in both arms by construction; on **TTFT** the same cell gives **201.9×**, not 1.09×. Also the summed-component-p90 dispersion construct **overstates** the true per-request p90 by **1.11×**. REV-1's K2 is a resolution floor on TTFT whose outcome is **INCONCLUSIVE, never dead**. |
> | **K3** (tiering headroom) | **kept**; NVMe 256.5 vs 272.2 two-file provenance split fixed. |
> | **K4** (edit leg, **< 6.0 pp**) | **bar moved to 3.0 pp** = the measured CI95 lower bound. 6.0 pp *is the point estimate*, so a bar there fires on ~half of faithful re-cohorts by noise alone. |
> | **cost 1.84 GPU-h** | **restated as 0.5–1.9 GPU-h, ceiling 3.** The C=8 term here is `17×8×13.584` = **8 sequential requests**, which presumes the very serialisation the experiment is meant to measure. Also **C=8 is infeasible as 8 processes**: 8 × 17.29 GiB = 138.3 GiB > 95 GiB usable H20 → OOM; it must be **in-process**. |
> | `needs_arch: sm_90`, `c1_all` forbidden, "edited chunk only" struck | **kept verbatim.** |
>
> § 6's blocker list is still accurate, and its `_precedence_warning` about `KILL_KEYS`
> **was correct and has since been fixed** (`ready_queue.py:139-179`).

> **Status of this file.** This is the falsification contract B07 did not have. It is
> written BEFORE any B07 GPU is spent, per `proposal/README.md` («新方向先写 PROPOSAL.md
> 和 kill gate，再启动 GPU»). Every threshold below is a **number with units** taken from
> an artefact already on disk — no threshold is invented, and none is chosen after seeing
> B07 data, because no B07 data exists yet.
>
> **This file does NOT clear B07 for GPU.** See §6.

---

## 0. What B07 actually claims, stated so it can lose

PROPOSAL.md bundles five systems features (concurrency, versioning, incremental edit,
HBM/CPU/NVMe/network tiering, reuse-aware admission). A feature list cannot be killed.
The **one** claim that can:

> **B07 serving thesis.** The CoMem depth-residual store's per-query advantage over a
> matched `j=0` top-12 raw-text replay — measured single-stream at **246.4 ms**
> (128k store, cpu-pinned tier, G=1) — **survives concurrency**, i.e. it is still
> present, resolvable, and directionally the same when 8 requests are in flight.

The adverse prediction against this claim is already **pre-registered by a different
proposal**, before B07's gate existed:

> `proposal/backlog/A02-comem-write-read-repair/A02_STORAGE_READCOMPUTE_VERDICT.md:297-299`
> — «**Scope**: single-query, batch=1, no continuous batching or paged attention. A served
> RAG system amortises prefill across concurrent requests, which would **further erode a
> 1.33× single-stream per-query advantage**.»

A gate that cannot let that prediction win is not a gate. §1 is written so it can.

---

## 1. KILL GATE (primary, concurrency)

### 1.1 The measured baseline this is scored against

From `paperA/artifacts/p1_8_serving/p1_8_serving_aggregate.json`,
`cells["128k|cpu"].per_G["1"]` (measured 2026-08-03, `.104`, 8×H20, torch 2.13.0):

| arm | per_query median | provenance |
|---|---|---|
| `comem` (j=12 + Read-LoRA `dd09cd17…`) | **688.5 ms** | `.per_G["1"].comem.per_query_s = 0.6884593861177564` |
| `j0_top12` (j=0, 36-layer replay, same pack) | **934.9 ms** | `.per_G["1"].j0.per_query_s = 0.9349067667499185` |
| **Δ (single-stream advantage)** | **246.4 ms** | difference of the two above |

### 1.2 Clause K1 — RETENTION (the kill clause)

> **K1.** At store `128k`, tier `cpu`-pinned, generation `G=1`, concurrency **C=8**,
> arm `comem` vs arm `j0_top12`, both arms sharing ONE `packed_ids_sha256`:
> if the paired per-request **TTFT p99** advantage Δ(C=8) is **< 123.2 ms**, OR its
> 95 % paired bootstrap CI **includes 0**, OR its sign is **negative**,
> then **the B07 serving thesis is KILLED** and the concurrency leg is closed.

- **Threshold 123.2 ms = 50 % of the measured 246.4 ms single-stream Δ.** 50 % is chosen
  because A02 confound 7 predicts *erosion*, not reversal: a claim that survives must keep
  at least half of the only advantage it has. It is a number, not "better than baseline".
- **Comparator is a named arm**: `j0_top12` = j=0 top-12 iter_bm25 raw-text replay, the same
  arm measured at 934.9 ms above. **Not** `c1_all` (pack-everything), which A02 §5 convicted
  as a badly-chosen baseline that hands 86–93 % of the apparent win to plain retrieval.

### 1.3 Clause K2 — DEPLOYABILITY FLOOR (second kill clause)

PROPOSAL.md's own generation setting is **G=128**, where the single-stream margin is already
gone: Δ = **182.0 ms = 3.09 %** (`per_G["128"]`: comem 5.8895 s vs j0 6.0715 s), while the
**within-arm** p90/median dispersion of `comem` at that same cell is **2.83 %**
(fetch_p90+read_p90+decode_p90 = 6.0562 s vs median 5.8895 s). Effect/floor = **1.09×**.

> **K2.** If at G=128, C=8 the Δ **cannot be resolved above the within-arm p90/median
> dispersion measured in the same run** (i.e. Δ ≤ that run's own dispersion band), then
> B07's serving claim **does not exist at its own deployable generation length**, and no
> ratio may be reported for that cell — only the absolute ms and the statement
> "below this run's resolution".

This encodes `memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md`: at G=128
the effect is 1.09× its own noise, so a "3 % win" is **not** a measurement yet.

### 1.4 Clause K3 — TIERING HEADROOM (pre-emptive, arithmetic, 0 GPU)

Cross-run composition, same node/arch/runtime (`.104`, H20 cc 9.0, torch 2.13.0):
model-side throughput at 128k|cpu is 1/0.6885 = **1.45 q/s per GPU = 11.62 q/s on 8 GPUs**
(G=1); at G=128 it is **1.36 q/s on 8 GPUs**. Measured store-backend peak fetch QPS at 128k
(`status/P2_2_PERSISTENT_STORE_IO.md`): GPU-resident 6443, CPU-pinned 956, NVMe 256.5,
CEPH-network 47. **Every backend measured is ≥ 4.04× faster than the model it feeds** (worst
case CEPH at G=1); at G=128 the worst backend is 34.6× faster.

> **K3.** Unless the C=8 run shows a cell where measured fetch time is **> 10 %** of that
> cell's per-request TTFT, the **HBM/CPU/NVMe/network tiering leg of B07 is declared out of
> headroom and dropped** from the proposal. (Measured fetch at 128k|cpu G=1 is 10.7 ms of a
> 688.5 ms budget = **1.55 %**.)

This narrows B07 by arithmetic rather than by opinion, and costs no GPU beyond the K1 run.

### 1.5 Clause K4 — EDIT LEG (the strategy list is already partly refuted)

`paperA/anonymous_artifact/scores/p0_17_e2_overlap/{summary,stats}.json`, n=200 strictly
paired, `niah_multikey_1 × {8k,16k}`: chunk-local Write (w=0) = **92.5** vs
document-contextual E0 = **100.0**; prepending w tokens of ORIGINAL-DOCUMENT left context
recovers it — w32 = **98.5** (Δ vs w0 = **+6.0**, 95 % CI **[3.0, 9.5]**, McNemar b=12/c=0,
p = **4.883e-4**), w128 = **99.0** (+6.5, CI [3.5, 10.0], p = 2.441e-4).
`e2_sanity.json` proves w=0 ≡ the deployable arm bit-for-bit (max_abs 0.0, tol 1e-3).

Consequence: **the stored `h12` of a chunk is NOT a function of that chunk alone.** So
PROPOSAL.md's strategy "edited chunk only" is **wrong-by-construction** — it silently ships
the 92.5 arm. It is struck from the proposal now, at 0 GPU.

> **K4.** If rewriting the edited chunk **plus its w=32 left-neighbour** recovers
> **< 6.0 pp** of the chunk-local→document-contextual gap (the measured E2_w32 recovery,
> CI [3.0, 9.5]), then incremental edit **cannot** preserve quality, and **only full
> document rewrite is admissible** — which removes B07's "incremental edit" selling point.

Predicted invalidation fan-out is already quantified and must be reported, not re-derived:
**w/512 extra chunks per random single-token edit** = 0.0625 at w=32, 0.25 at w=128; Write-cost
side likewise measured (**+5.73 %** FLOPs at w32, **+22.92 %** at w128; store/Read/decode
byte-identical to w0, `summary.json:write_cost`).

### 1.6 Δ GUARD — mandatory, because the denominator can be ≤ 0 here

Any **ratio / retention-fraction** in this gate has an ill-defined denominator, and this is
not hypothetical in this harness:

- **Concurrency leg.** Retention = Δ(C=8)/Δ(C=1). At `128k|cpu|G=512` the measured
  denominator is already **0.013 s** (21.6202 − 21.6072), and `aggregate.out:8` **already
  prints `Q*=inf`** at `32k|gpu|G=512` — i.e. the break-even is *undefined* there, not
  favourable.
  > **Guard.** If the C=1 denominator Δ(C=1)'s 95 % paired bootstrap CI **includes 0**, the
  > retention fraction is **UNDEFINED**. Report the **absolute** TTFT Δ in ms and the CI;
  > **never** report a large ratio, and **never** report `inf` as a win. K1 then falls back
  > to its absolute clause (Δ(C=8) ≥ 123.2 ms with CI excluding 0).
- **Edit leg.** Recovery fraction divides by (E0 − chunk_local) = **7.5 pp**, which is only
  **15 discordant items out of 200** and can be **≤ 0** on a re-cohort.
  > **Guard.** If the re-cohort's (E0 − chunk_local) 95 % CI includes 0, the recovery
  > fraction is **UNDEFINED**; report absolute pp only. K4 then requires the absolute
  > +6.0 pp, not a fraction.

### 1.7 Falsifiability self-test (a gate that can only pass is not a gate)

**Concrete result that KILLS B07, and its mechanism.** `select` (iter_bm25) is **CPU** and
**O(L)**: 117.7 ms at 128k, paid *identically* by both arms (A02 §4). At C=8 with a single
selector worker that is **8 × 117.7 = 941.6 ms of serial CPU work per batch**, which
*exceeds* `comem`'s entire 677.9 ms GPU read. If the C=8 bottleneck is that shared serial CPU
stage, both arms' TTFT p99 is dominated by an identical term and Δ collapses toward a small
fraction of 246.4 ms → **Δ(C=8) < 123.2 ms → K1 fires → thesis dead.** P2.2 independently
records the CPU path as "GIL-bound single-copy loop", so this is a live mechanism, not a
strawman. **Concurrency must therefore be instrumented on CPU as well as GPU**, or the
queueing will be mis-attributed.

**Concrete result that PASSES.** If the C=8 bottleneck is instead the GPU read, queue wait
scales with per-request service time, so the cheaper 24-layer arm drains faster and the p99
gap *amplifies* (≈ queue-depth × per-request Δ) → Δ(C=8) ≥ 123.2 ms with CI excluding 0.

Both branches are physically plausible from measurements already on disk. That is the test.

---

## 2. NEXT GATE (the one decisive measurement)

**Single variable: concurrency C ∈ {1, 8}.** Everything else is pinned to the archived
flagship config so the C=1 arm is a *free correctness check* against
`p1_8_serving_aggregate.json` (must reproduce 688.5 / 934.9 ms).

| axis | value | why fixed |
|---|---|---|
| store L | `128k` | the cell with the published anchor |
| tier | `cpu`-pinned | the deployable tier (GPU-resident is HBM-capped at ~8M tok) |
| generation G | `1` **and** `128` | G=1 = powered primary (effect/floor 4.09×); G=128 = PROPOSAL.md's own setting, K2 |
| arms | `comem` (j=12 + LoRA) vs `j0_top12` | A02's matched comparator; **`c1_all` excluded** |
| concurrency C | **1, 8** | the single variable |
| procs × repeats | 3 × 15 | ≥3 independent procs is the existing protocol; 15 repeats → n=45 paired draws |
| task / example | `niah_multikey_1`, `example_index=0`, `read_sample_length=32k` | identical to the anchor |
| model / adapter | `models/Qwen3-8b-local` + `outputs/qcmem_distill_qwen_j12_r32_4k/final` | LoRA sha **must** equal `dd09cd17457c63578c0f38dab79b287ab5da6e3f14c119aedafec1c34400536f` |

**Reused assets (this is what keeps the cost at ~2 GPU-h instead of tens):**
`scripts/bench_p1_8_serving_curve.py` (40 205 B) — `_load` / `_build_pack` / `_eos_ids` /
`_summ` / `QCMemModel` / `EXPECTED_LORA_SHA` imported verbatim from
`scripts/eval_p016_e0_write_control.py`; `scripts/_run_p1_8_serving.sh` (flock GPU pool,
DRY-by-default). All 5 fail-closed gates stay armed: LoRA sha match, `GATE2 store==recompute
max_abs==0.0`, both arms share ONE pack sha, exact `persistent_bytes`, finite logits.

**Engineering delta (the honest part):** the harness has **no concurrency axis and no
percentile beyond p90**. Verified by grep — it contains `_serve_comem` (line 226) and
`_serve_j0` (line 289) and **no** thread/pool/TTFT/p95/p99 code. Adding C and TTFT
percentiles is the whole implementation task, and it is 0-GPU work.

**Decidable outcome — both branches:**
- **PASS** → Δ(C=8) p99 ≥ 123.2 ms, 95 % paired CI excludes 0, sign positive, AND G=128 Δ
  clears its own within-arm dispersion. B07 keeps the concurrency leg; the tiering leg is
  still dropped unless K3's >10 % fetch share appears.
- **FAIL** → any of K1's three sub-conditions, or K2. The concurrency leg closes and B07
  collapses to the edit leg alone — where the E2 evidence already on disk does most of the
  work at 0 GPU, and K4 governs.

---

## 3. Read-out point, pre-registered

**The read-out is `C=8, G=1, 128k|cpu`, TTFT p99, paired against `C=1` from the same
processes, n = 3 procs × 15 repeats = 45 paired draws — and NOT any other cell.**
Committed before data. The pass/fail sentence is evaluated **once**, on that cell.
G=128 is a **gating precondition (K2)**, not a second chance at the primary. The full
{Q, G} grid is *reported* but is **not** the decision point — this mirrors the paperC lesson
that `--max_steps 200000` was never the decision point (pre-registered read-out was step
121000).

---

## 4. Cost, from a real measured anchor

**Anchor** (not a guess): `paperA/artifacts/p1_8_serving/serve/serve_128k_cpu_niah_multikey_1_proc0.json`,
per-component medians, × iteration counts from the recorded protocol (`warmup=2`,
`n_repeat=5` → 7 iterations):

```
W (comem write-once)              6.824 s     one_time.comem_write_once_s.median
I (j0 index)                      0.021 s     one_time.j0_index_s.median
per-iter, both G{1,128}, both arms 13.584 s   sum of the 4 per_query_s medians
```

Gate cost per proc at `n_repeat=15` (17 iterations): setup 120 s + W + I
+ 17×13.584 (C=1 leg) + 17×8×13.584 (C=8 leg) = **2205 s = 36.75 min**.
**3 procs = 110.3 GPU-min = 1.84 GPU-h**; **×1.5 safety = 2.76 GPU-h**. Call it **2–3 GPU-h,
1 GPU at a time** (each (L,tier,proc) is a self-contained single-GPU unit).

**Independent cross-check of the anchor.** Summing the same medians over the *original* full
G-grid {1,32,128,512} gives 433.6 s = 7.23 min of measured compute per unit, ~9.2 min with
load; the archived wall-clock was `selfcheck.out` 17:52:33 → `aggregate.out` 18:17:26 ≈ 25 min
for 18 units on 8 GPUs ⇒ ~11 GPU-min/unit. **The two independent routes agree**, so the rate
is real. STATUS.json's `UNKNOWN -- 需先做 1-cell 计时` was **over-cautious**: the 1-cell timing
already existed on disk and simply was not used.

---

## 5. Node requirement — `sm_90` (H20), and it is load-bearing

The C=1 arm's entire value is that it must **reproduce** `128k|cpu|G=1` = 688.5 / 934.9 ms.
That anchor was measured on **`.104`, 8×H20, cc 9.0, torch 2.13.0** (`status/P2_2_PERSISTENT_STORE_IO.md:17-19`;
same cohort for P1.8). A02 §6 item 8 states it explicitly: «**Machine-specific**: H20 (cc 9.0),
torch 2.13.0, bf16, sdpa. The **L20A nodes would give different absolute latencies**; the
ratios should be more stable.»

- ⇒ **Must run on `sm_90`: `.73` / `.82` / `.104`.** Running on B200 (`sm_100`, LOCAL/`.212`)
  would mix hardware drift into a latency comparison whose reference point is an H20 number —
  the reproduction check would be uninterpretable and K1's 123.2 ms bar meaningless.
- ⚠️ **Cross-disk prerequisite.** All five reusable harnesses are verified present on **wzc1**
  and **UNVERIFIED (not absent) on zwfy6** — `/apdcephfs_zwfy6` is not mounted on LOCAL, and
  this task was forbidden from ssh. Since the three H20s live on **zwfy6**, dispatch requires
  `scp -O` first (`.82` sftp subsystem is broken) + md5 verify. Set
  `PROJECT_ROOT=/apdcephfs_zwfy6/...`, `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`.

---

## 6. This gate does NOT make B07 `ready_gpu`. What is still missing.

`proposal/ready_queue.py:262-283` fires three checks in order. §1–§4 above close **two** of
them (kill gate, next gate). The remaining blockers are real:

1. **`RELATED_WORK.md` absent.** `ready_queue.py:46-51` hard-codes B07 among B01/B05/B06/B07/B08
   whose missing `RELATED_WORK.md` forbids GPU *regardless of gate quality*.
   `proposal/shared/literature/RELATED_WORK_GAP_AUDIT_20260808.md:97` rates B07 «不足» and
   :146 demands a **per-feature systems collision table** (prefix/KV caching; paged/disaggregated
   KV; versioning/invalidation; memory tiering; reuse-aware admission; incremental recompute)
   × {closest prior system, what it does NOT do, our measured difference}.
   **This is 0-GPU work and is B07's true critical path.**
   ⚠️ Venue verification is currently **not executable from this node**: `api2.openreview.net`
   returned **HTTP 403** and `aclanthology.org` **timed out** through `hy-proxy.woa.com:3128`.
   Per `memory/venue-verify-acl-family-needs-anthology.md`, ACL-family venues need
   aclanthology+DBLP and OpenReview-family need `venueid` — so the table cannot be finished
   until one of those is reachable. **No `.bib` entry from an LLM report may be admitted unchecked.**
   > **Not a kill reason.** Per the standing correction (2026-08-07,
   > `memory/prior-work-differentiate-dont-abandon.md`) **overlap ≠ preemption**; the bar is
   > 完全相同/抄袭. The audit's demand is *constructive*: stop claiming a feature list, give a
   > collision table + end-to-end quality/latency/bytes + a named workload with cross-query reuse.
   > Likewise `status/scout_21/lane4_cheapest_killer.md:33,282` ("NO — systems measurement, no
   > scientific kill") describes the **old PROPOSAL.md text**, not a falsification: **no B07
   > experiment has ever run.** Do not upgrade that triage line into a death certificate.
2. **The concurrency axis does not exist in code.** §2's engineering delta must land first.
3. **STATUS.json key precedence — VERIFIED EMPIRICALLY, two keys affected.**
   `NEXT_GATE_KEYS[0] = "next_gate_executable_20260814"` gives the next-gate a dated-priority
   slot, but **`KILL_KEYS` (`ready_queue.py:81`) and the cost key list
   (`["gpu_cost_estimate","cost","cost_to_first_result"]`, ~line 189) do not.** Since
   STATUS.json is **append-only** (`proposal/LIFECYCLE_SCHEMA.md §0`: the only permitted byte
   change is `}` → `,`), the older honest sentinels cannot be overwritten. I ran
   `python proposal/ready_queue.py` after the append and it confirms the consequence exactly:

   ```
   B07-mutable-comem-serving
      why: DECLARED lifecycle=ready_cpu in STATUS.json (authoritative; no reason field)
      gate[next_gate_executable_20260814]: ONE decisive measurement, single variable = concurrency C in {1,8}…   <- READ
      cost: UNKNOWN -- 需先做 1-cell 计时                                                                        <- STALE
      ! kill gate undefined (NO_KILL_GATE_DEFINED -- PROPOSAL.md has no Kill section. It…)                       <- STALE
   ```

   So `kill_gate_executable_20260814` and `gpu_cost_estimate_20260814` **are on disk and are
   authoritative for humans, but the scheduler still shows the stale sentinels.** Fix is two
   entries: `KILL_KEYS` → prepend `"kill_gate_executable_20260814"`, cost list → prepend
   `"gpu_cost_estimate_20260814"`. **Deliberately not made here** (this task was scoped to
   0 GPU, no launch-script edits, no commit). Until that lands, B07 would remain `ready_cpu`
   *even if* Related Work were finished — which is a reporting artefact, not a real blocker,
   and must not be mistaken for one.

**Honest lifecycle: `ready_cpu`.** The actionable next task is the `RELATED_WORK.md`
collision table + the concurrency-axis implementation, both 0 GPU. B07 is not starved of
cards; it was starved of a decidable question, and now of paperwork plus one harness feature.

---

## 7. Struck from PROPOSAL.md by evidence already on disk (0 GPU narrowing)

| PROPOSAL.md item | disposition | evidence |
|---|---|---|
| "edited chunk only" update strategy | **STRUCK, wrong-by-construction** | `p0_17_e2_overlap/summary.json`: ships the 92.5 arm vs 100.0 |
| HBM/CPU/NVMe/network tiering | **out of headroom pending K3** | every measured backend ≥4.04× the model's 8-GPU throughput |
| "CoMem is a storage method" | **DEAD, do not re-litigate** | A02 clause (c) FAILED: 2048× vs pre-registered 100× (8192 B/tok vs 4 B/tok) |
| serving win at any quality | **must be matched-quality or not claimed** | `paperA/TODOList.md:763` P0.20: BM25 −11.56 pp CI[−14.44,−8.67]; dense TIE −1.0 pp p=0.637 — «最好情形是打平，从不是赢» |
| `c1_all` as comparator | **forbidden** | A02 §5: hands 86–93 % of the apparent win to plain retrieval; also OOMs at 1M (score arm-absent, never a CoMem win) |
