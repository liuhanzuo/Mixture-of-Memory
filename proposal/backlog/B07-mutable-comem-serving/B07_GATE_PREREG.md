# B07 — Kill gate, PRE-REGISTRATION **REV-1** (written 2026-08-15, 0 GPU, PRE-DATA)

> **Read this first.** This is **not** a from-scratch gate. `B07_SERVING_GATE_PREREG.md`
> (2026-08-14) already wrote a four-clause gate, and `STATUS.json` already carries
> `next_gate_executable_20260814` + `kill_gate_executable_20260814`. The task that
> commissioned this file was briefed on `next_gate = "NOT_SPECIFIED"`, which is the
> **v0 sentinel that append-only forbids overwriting** — the sentinel is still the
> first `next_gate*` string in the file, but it has been superseded on disk since
> 08-14 and `proposal/ready_queue.py:130-134` already resolves the dated key first
> (verified by running it, §8).
>
> **So the honest deliverable is not "write the missing gate". It is: audit the gate
> that exists, and fix the defects that make it not-yet-decidable.** I found four,
> two of them fatal to the gate as written. REV-1 keeps the parts that survived
> re-derivation from the raw artefacts and replaces the parts that did not.
>
> **`B07_SERVING_GATE_PREREG.md` is NOT deleted and NOT edited.** It is `SUPERSEDED`
> by this file per `proposal/README.md`'s rule *「改写 gate，并在 proposal 里记下为什么改
> （旧文本保留、标 SUPERSEDED）」*. Every number it states is arithmetically correct;
> what changed is which *statistic* the gate is allowed to be written on.

---

## 0. Audit of the 08-14 gate: what survived, what did not

I re-derived every threshold in `B07_SERVING_GATE_PREREG.md` from the raw
per-process artefacts (`paperA/artifacts/p1_8_serving/serve/serve_128k_cpu_niah_multikey_1_proc{0,1,2}.json`),
not from the aggregate, and re-ran its cited greps.

| 08-14 clause | verdict | why |
|---|---|---|
| K1 threshold **123.2 ms = 50 % of 246.4 ms** | **arithmetic CORRECT, statistic WRONG** | Δ = 0.9349067667499185 − 0.6884593861177564 = 246.447 ms exactly, and half is 123.223 ms. But it is scored on **TTFT p99 at n=45**, where p99 **is the maximum order statistic** — an unestimable single-observation target. **Fatal. Replaced in §3.** |
| K2 deployability floor at **G=128**, effect/floor = 1.09× | **CORRECT but MEASURES THE WRONG THING** | 3.09 % / 2.83 % = 1.0930×, reproduced exactly. But that margin is the **total-latency** Δ, which at G=128 is 71 % *decode*, and decode is where the two arms are **identical by construction**. The TTFT Δ at G=128 is **251.3 ms with effect/floor = 202×**. **The 08-14 gate would have killed B07 on a number that measures neither arm's difference. Fatal. Replaced in §4.** |
| K3 tiering headroom, worst backend ≥ 4.04× the model | **CORRECT, one number mis-cited** | `47/11.62 = 4.045`. But NVMe **256.5** is the `p2_2_full.json` figure while GPU 6443 / CPU 956 / CEPH 47 are read from **two different files**; `p2_2_file_isolated.json` gives NVMe **272.2**. Both are on disk; mixing them in one row is a provenance defect, not an arithmetic one. **KEPT with the provenance fixed (§5).** |
| K4 edit leg, **< 6.0 pp** recovery | **CORRECT, but the bar is the point estimate itself** | +6.0 pp, CI [3.0, 9.5], McNemar b=12/c=0, p=4.883e-4 all reproduce from `stats.json`. Setting the kill bar **at the point estimate** means a faithful re-cohort has ≈50 % chance of firing it by sampling noise alone. **KEPT with the bar moved to the CI lower bound (§6).** |
| "edited chunk only" is wrong-by-construction | **CORRECT and important** | `e2_sanity.json`: `e2_w0_residual.max_abs = 0.0` ⇒ w=0 **is** the deployable arm bit-for-bit, and it scores 92.5 vs E0's 100.0. **KEPT verbatim (§6).** |
| `c1_all` forbidden as comparator | **KEPT** | A02 §5. Unchanged. |
| Cost **1.84 GPU-h** | **arithmetic CORRECT, protocol INFEASIBLE** | Recomputed: Σ of the four `per_query_s` medians = 13.5843 s; 120 + 6.824 + 0.021 + 17×13.5843 + 17×8×13.5843 = 2205.24 s = 36.754 min; ×3 = 1.8377 GPU-h. **But it silently assumes C=8 is 8× the serial work, i.e. 8 sequential requests — that is not concurrency.** And the *process*-based reading OOMs (§7). **Cost re-derived in §7.** |
| `needs_arch: sm_90` | **KEPT, and it is load-bearing** | `status/P2_2_PERSISTENT_STORE_IO.md:17-19` = `.104`, 8×H20, torch 2.13.0; A02 verdict `:301-302` says the L20A nodes give different absolute latencies. Confirmed at those exact lines. |

**Net:** the 08-14 gate is decidable in *form* but two of its four clauses are
scored on statistics the protocol cannot estimate (K1) or that are dominated by a
term identical in both arms (K2). Fixing those is what REV-1 does.

---

## 1. The claim, stated so it can lose

Unchanged from 08-14, because it is correct:

> **B07 serving thesis.** The CoMem depth-residual store's per-query advantage over
> a matched `j=0` top-12 raw-text replay of the **same pack** survives concurrency —
> it is still present, resolvable, and directionally the same with 8 requests in flight.

The adverse prediction is pre-registered by a *different* proposal, before B07 had a
gate, which is what makes it a real risk and not a strawman
(`proposal/backlog/A02-comem-write-read-repair/A02_STORAGE_READCOMPUTE_VERDICT.md:297-299`,
read this session):

> «**Scope**: single-query, batch=1, no continuous batching or paged attention. A
> served RAG system amortises prefill across concurrent requests, which would
> **further erode a 1.33× single-stream per-query advantage**.»

---

## 2. Which statistic the gate may be written on — and why the 08-14 choice fails

This is the core of REV-1, so it is derived rather than asserted.

### 2.1 The advantage lives **entirely** in prefill, not in the request

Decomposing the archived `128k|cpu` cell by component (all four values per row read
from `p1_8_serving_aggregate.json`, differences computed this session):

| G | Δ fetch | Δ read (prefill) | Δ decode | **Δ total** | **Δ TTFT = Δ(fetch+read)** |
|---|---|---|---|---|---|
| 1 | −10.32 ms | **+256.76** | −0.00 | **+246.45** | **+246.45** |
| 32 | −5.89 | **+256.71** | −21.39 | +229.43 | **+250.82** |
| 128 | −5.61 | **+256.92** | −69.33 | +181.98 | **+251.31** |
| 512 | −8.65 | **+260.43** | −238.70 | **+13.08** | **+251.78** |

Two facts follow, and they decide the metric:

1. **Δ read is flat at ≈ 257 ms across a 512× change in G.** That is the whole
   effect, and it is a *prefill* quantity: `comem` resumes at layer 12 of 36, `j0`
   replays all 36 over the same 6177-token pack (`read_len` = 6177 for **both** arms
   in **all 6** proc×G records I checked — verified, so the pack really is matched).
2. **Δ total decays to +13.08 ms at G=512 purely by dilution.** Per-token decode is
   `comem` 39.88 / 40.67 / 40.86 ms/tok vs `j0` 39.21 / 40.13 / 40.39 ms/tok at
   G = 32/128/512 — i.e. **constant in G, and `comem` is 1.15–1.70 % *slower***
   (it decodes with a longer KV history from the same pack). Decode is the term the
   two arms **share by construction**, so every extra generated token adds noise
   with zero signal.

> **Ruling (pre-data).** The gate is scored on **TTFT**, defined as
> `fetch_s + read_s` — the elapsed time to the first token, which is exactly the
> component in which the arms differ. **`per_query_s` (total latency) is BANNED as
> the decision statistic** and may only be *reported*. The 08-14 K2 clause violated
> this: it took the G=128 total-latency margin (3.09 %) — which is 71 % decode — and
> compared it to a dispersion band, concluding effect/floor = 1.09× and threatening
> to kill the direction. **On TTFT, the same cell gives effect/floor = 202×.**
> Killing a claim on a metric that is three-quarters common-mode is the
> `memory/a-range-is-not-a-measurement-until-it-clears-its-floor.md` error inverted:
> not a ratio below its floor, but a *floor computed for the wrong numerator*.

### 2.2 p99 is not estimable at this n, and REV-1 does not pretend otherwise

The archived protocol is `warmup=2, n_repeat=5`, 3 procs ⇒ **15 draws per arm per G**
(verified: `config.n_repeat = 5`, and each `raw` list has 5 entries × 3 procs).
The 08-14 gate raised it to `n_repeat=15` ⇒ **45 draws** and then scored **p99**.

With n = 45, `ceil(0.99 × (n−1)) = 44 = n−1`: **the p99 estimate IS the maximum**.
Measured on the archived n=15: Δ(p50) = 247.60 ms, Δ(p90) = 246.15 ms, but
Δ(p99) = 299.88 ms = Δ(max) = 304.77 ms to within one draw — the "p99" is one
observation, and it is 21 % higher than the median for that reason alone.
A p99 with even 5 order statistics above it needs **n ≥ 500**.

> **Ruling (pre-data).** **PRIMARY statistic = Δ of TTFT MEDIANS**, with a
> bootstrap CI. **p99 is demoted to a reported diagnostic** and is *not* a kill
> criterion at any n below 500. A tail claim may be made only from the C=8 leg,
> where 8 concurrent requests × 15 repeats × 3 procs = **360 draws** puts p99 at the
> ~4th largest order statistic — reportable, still not gate-worthy.

### 2.3 The draws are **not paired**, and the 08-14 gate says they are

`scripts/bench_p1_8_serving_curve.py:517-526`: `_measure` is called **once for
`comem` (line 517) and then once for `j0` (line 524)** — two sequential loops over
`warmup + n_repeat`, with `torch.cuda.reset_peak_memory_stats()` between them.
Draw *i* of `comem` and draw *i* of `j0` are **not the same request, not the same
moment, and share no per-draw nuisance term**. The 08-14 gate demands a "paired
bootstrap CI" four times and calls the read-out "45 **paired** draws". That
pairing does not exist in this harness.

> **Ruling (pre-data).** The CI is an **unpaired two-sample bootstrap on the
> difference of medians**, B = 20000, seed 42. If the C=8 implementation
> (§7) interleaves the arms request-by-request within one loop, a paired estimator
> becomes legitimate and **must be declared in the run's own manifest before the
> run**, not chosen afterwards. Reporting whichever of the two gives the nicer
> interval is forbidden.

**Baseline CIs, computed now from the 15 archived draws so they cannot be tuned later**
(unpaired, B=20000, seed 42):

| cell | Δ TTFT medians | bootstrap CI95 | raw ranges | comem TTFT median / p90 | effect / floor |
|---|---|---|---|---|---|
| `128k\|cpu`, G=1 | **+247.60 ms** | **[218.72, 253.54]** | comem [681.0, 764.6] vs j0 [934.5, 1069.3] → **DISJOINT** | 687.40 / 732.70 ms (disp 45.30) | **5.47×** |
| `128k\|cpu`, G=128 | **+251.44 ms** | **[249.65, 255.36]** | comem [680.4, 702.5] vs j0 [933.6, 990.9] → **DISJOINT** | 683.61 / 684.86 ms (disp 1.25) | **201.9×** |

Both CIs exclude 0 by a wide margin and the arms' raw draw ranges do not even
overlap. **The single-stream anchor is a real measurement on TTFT** — which is
precisely why the concurrency question is worth one run, and why it must be asked
on this metric.

---

## 3. K1 — RETENTION (the primary kill clause) **[REV-1]**

> **K1.** At store `128k`, tier `cpu`-pinned, generation **G = 1**, arms `comem`
> (j=12 + Read-LoRA `dd09cd17…`) vs `j0_top12` (j=0, 36-layer replay), both arms
> sharing **one** `packed_ids_sha256`, run at concurrency **C = 8**:
>
> let **Δ₈ = median(TTFT of j0_top12) − median(TTFT of comem)**, TTFT = `fetch_s + read_s`,
> over **n = 3 procs × 15 repeats × 8 in-flight = 360 requests per arm**.
>
> **B07's serving thesis is KILLED, and the concurrency leg closes, if ANY of:**
> | # | condition | 
> |---|---|
> | K1a | **Δ₈ < 123.2 ms** |
> | K1b | the 95 % bootstrap CI of Δ₈ **includes 0** |
> | K1c | **sign(Δ₈) < 0** |
> | K1d | the **C = 1 control fails to reproduce** the archived anchor: median TTFT outside `comem` **687.40 ± 68.74 ms** or `j0` **935.01 ± 93.50 ms** (±10 %) |

- **Threshold 123.2 ms is unchanged from 08-14 and is pre-data**: it is
  **50 % of the measured 246.447 ms single-stream Δ** at that exact cell
  (`per_G["1"]`: 0.9349067667499185 − 0.6884593861177564). 50 % because A02
  confound 7 predicts **erosion, not reversal** — a surviving claim must keep at
  least half of the only advantage it has. Keeping the 08-14 number is deliberate:
  REV-1 changes the *statistic*, and re-tuning the *bar* at the same time would make
  the revision unfalsifiable.
- **Noise floor, and where it comes from.** 123.2 ms must clear the floor of the
  quantity being measured. Two independent floors, both from the **same harness,
  same node, same 15 archived draws** (not from an assumed model):
  1. **Within-arm dispersion** of `comem` TTFT at the primary cell: p90 − median =
     **45.30 ms** (G=1). 123.2 / 45.30 = **2.72×**.
  2. **Bootstrap half-width** of Δ itself: CI95 [218.72, 253.54] ⇒ half-width
     **17.41 ms**. 123.2 / 17.41 = **7.08×**.
  The bar clears the larger floor by 2.72×. Recorded pre-data so a post-hoc
  "the threshold was inside the noise" objection cannot be raised either way.
- **K1d is a free correctness check, not a hypothesis test.** Everything except C is
  pinned to the archived flagship config, so C=1 must reproduce 687.40 / 935.01 ms.
  ±10 % is chosen because the archived draws themselves span
  `comem` [681.0, 764.6] ms = −0.9 %/+11.2 % of the median; a tighter band would fire
  on known-good jitter. **If K1d fires, the run is void and K1a–c are NOT evaluated** —
  a broken apparatus must not be allowed to "kill" a direction.

### 3.1 Falsifiability self-test — a concrete result that KILLS, with a mechanism

`select` (`iter_bm25`) is **CPU** and **O(L)**: **117.7 ms at 128k**, paid
*identically* by both arms (`A02_STORAGE_READCOMPUTE_VERDICT.md:187-188,201-202` —
the table's `select` column is 117.7 for both `comem` and `j0_top12`). At C=8 behind
one selector worker that is **8 × 117.7 = 941.6 ms** of serial CPU per batch, which
**exceeds `comem`'s entire 677.9 ms GPU read**. If C=8 bottlenecks there, both arms'
TTFT is dominated by an identical term, Δ₈ collapses below 123.2 ms, **K1a fires,
thesis dead.** `status/P2_2_PERSISTENT_STORE_IO.md:134` independently records the
CPU path as a "GIL-bound single-copy loop", so this is live.

⚠️ **Therefore the selector stage MUST be instrumented separately from `read_s`**,
or a CPU-queueing kill will be mis-attributed to the GPU read. Note the archived
harness excludes `select` from `read_s` entirely (`run_serve` builds the pack
**once**, at line 419, outside `_measure`) — so the archived 246.4 ms Δ contains
**no** selector time. Under real concurrency the selector is per-request and shared.
**This is the single largest reason the gate can fire.**

**A concrete result that PASSES.** If C=8 is GPU-read-bound, queue wait scales with
service time; the cheaper 24-layer arm drains faster and the TTFT gap **amplifies**
toward queue-depth × per-request Δ. Both branches follow from data already on disk.

### 3.2 Δ guard — the denominator can be ≤ 0 here, and this is not hypothetical

Any **retention fraction** Δ₈/Δ₁ has an ill-defined denominator in this harness:
at `128k|cpu|G=512` the measured *total-latency* denominator is already **0.0131 s**
(21.620244916528463 − 21.607163473963737), and `aggregate.out:8` already prints
`Q*=inf` at `32k|gpu|G=512` — *undefined*, not favourable.

> **Guard.** If Δ₁'s CI95 includes 0, the retention fraction is **UNDEFINED**:
> report the **absolute** TTFT Δ in ms with its CI, never a large ratio, and never
> `inf` as a win. K1 then rests on its absolute clause (K1a) alone.
> On TTFT at G=1 the denominator is 247.60 ms with CI [218.72, 253.54], so the
> fraction *is* defined at the primary cell — but the guard stands for every other
> cell, and for the G=512 column it is already binding.

---

## 4. K2 — RESOLUTION FLOOR **[REV-1: replaces the G=128 deployability clause]**

The 08-14 K2 threatened to kill B07 because the G=128 **total-latency** margin
(3.09 %) barely exceeded a dispersion band (2.83 %). §2.1 shows that margin is
71 % decode, a term identical in both arms by construction. On TTFT the same cell
gives Δ = 251.44 ms against a 1.25 ms dispersion = **201.9×**. So the clause is
replaced by one that tests resolution *of the statistic actually gated*:

> **K2.** At the primary cell, if Δ₈ **cannot be resolved above the within-arm
> TTFT dispersion measured in the same run** — i.e. Δ₈ ≤ (`comem` TTFT p90 − median)
> of that run — then the C=8 result is **below its own resolution**: report absolute
> ms plus "below this run's resolution", report **no ratio**, and the concurrency leg
> is **INCONCLUSIVE, not passed**. An inconclusive C=8 leg does not kill B07 and does
> not license a second attempt at the primary; it requires a **larger n declared in
> writing before re-running**.

Two properties make this the honest version:
- the floor is **measured in the same run** as the effect, so it cannot be imported
  from a friendlier cell;
- **inconclusive ≠ pass and ≠ kill.** The 08-14 clause conflated "unresolvable" with
  "dead", which is how a direction gets killed by its own apparatus.

> ⚠️ **Dispersion must be computed from true per-request totals, not from summed
> component p90s.** Measured this session at G=128: the aggregate's
> `fetch_p90 + read_p90 + decode_p90` = 6.055967 s implies a 166.49 ms dispersion,
> while the **true** p90 of the 15 per-draw totals is 6.040556 s = **150.64 ms** — the
> summed-p90 construct **overstates dispersion by 1.11×** because it adds three
> independently-pooled 90th percentiles as if they co-occurred. The 08-14 K2 used the
> summed construct. **REV-1 requires the true per-request order statistic.**

---

## 5. K3 — TIERING HEADROOM (pre-emptive, arithmetic, 0 GPU) **[KEPT, provenance fixed]**

Model-side throughput at `128k|cpu` is 1/0.6885 = **1.45 q/s per GPU = 11.62 q/s on
8 GPUs** (G=1); at G=128, **1.36 q/s on 8 GPUs**. Measured store-backend peak fetch
QPS at 128k, **each now cited to the file it came from** (read this session from the
raw JSON, not the markdown table):

| backend | peak QPS @128k | source file | × the model @G=1 |
|---|---|---|---|
| GPU-resident | 6443.4 | `ruler_results/p2_2/p2_2_full.json` | 554× |
| CPU-pinned | 956.0 | `p2_2_full.json` | 82× |
| NVMe (O_DIRECT) | **256.5** (`p2_2_full`) / **272.2** (`p2_2_file_isolated`) | both on disk | 22.1× / 23.4× |
| CEPH network | 47.5 (`p2_2_full`) / 47.2 (`p2_2_file_isolated`) | both on disk | **4.09×** |

> **K3.** Unless the C=8 run shows a cell where **measured fetch time exceeds 10 % of
> that cell's per-request TTFT**, the HBM/CPU/NVMe/network tiering leg is **DROPPED**
> from B07 and reported as a negative result. (Measured fetch at `128k|cpu|G=1` is
> **10.675 ms of a 688.45 ms** TTFT = **1.55 %**.)

Provenance note, since the 08-14 text mixed two files in one row: the two P2.2 runs
differ by design — `p2_2_full.json` measures all four backends in one process,
`p2_2_file_isolated.json` re-measures only the file backends in isolation. **Both
NVMe numbers are valid; quoting one alongside three from the other file is not.**
The weakest backend is CEPH at 4.09× (using the 47.5 figure from the same file as
the GPU/CPU rows), which is what K3 turns on, and both NVMe readings are ≫ 10× so
the conclusion is insensitive to the choice.

---

## 6. K4 — EDIT LEG **[REV-1: bar moved off the point estimate]**

`paperA/anonymous_artifact/scores/p0_17_e2_overlap/{summary,stats,e2_sanity}.json`,
n = 200 strictly paired, `niah_multikey_1 × {8k,16k}` — every figure below read from
those files this session:

| arm | score | vs w0 | CI95 | McNemar |
|---|---|---|---|---|
| A (full replay) | 100.0 | +7.5 | [4.0, 11.5] | b=15 c=0, p=6.104e-5 |
| **B = w0 (chunk-local, DEPLOYABLE)** | **92.5** | — | — | — |
| E0 (document-contextual control) | 100.0 | +7.5 | [4.0, 11.5] | b=15 c=0, p=6.104e-5 |
| E2 w32 | 98.5 | **+6.0** | **[3.0, 9.5]** | b=12 c=0, p=4.883e-4 |
| E2 w64 | 98.5 | +6.0 | [3.0, 9.5] | b=12 c=0, p=4.883e-4 |
| E2 w128 | 99.0 | +6.5 | [3.5, 10.0] | b=13 c=0, p=2.441e-4 |

`e2_sanity.json`: `e2_w0_residual.max_abs = 0.0` (tol 1e-3) and
`e0_h12_residual.max_abs = 0.0` ⇒ **w=0 is the deployable arm bit-for-bit.**

**Struck at 0 GPU, and this survives REV-1 unchanged:** the stored `h_j` of a chunk
is **not a function of that chunk alone** (it has already absorbed cross-chunk
attention), so PROPOSAL.md's **"edited chunk only"** strategy silently ships the
**92.5** arm. It is struck from the proposal now.

> **K4 [REV-1].** If rewriting the edited chunk **plus its w=32 left-neighbour**
> recovers **< 3.0 pp** of the chunk-local → document-contextual gap on a faithful
> re-cohort, incremental edit cannot preserve quality and **only full document
> rewrite is admissible** — which removes B07's "incremental edit" selling point.

**Why 3.0 pp and not the 08-14 bar of 6.0 pp.** 6.0 pp *is the point estimate* of
the very effect being re-measured. A kill bar set at the point estimate fires on
roughly half of all faithful re-cohorts by sampling noise alone — it is a coin flip
dressed as a criterion. **3.0 pp is the measured CI95 lower bound** of the same
quantity ([3.0, 9.5]), so K4 now fires only when the re-cohort is *inconsistent with*
the established effect rather than merely unlucky within it. This is the same
pre-registration discipline A04 was forced into (`A04/STATUS.json:next_gate[4]`:
a threshold must be stated against the σ/MDE of its own apparatus, not against its
own point estimate).

**Secondary K4 note, pre-registered so it cannot be spun later.** E2_w32 does **not**
statistically reach E0: `E2_w32_vs_E0` is **−1.5 pp, CI [−3.5, 0.0], b=0/c=3,
p=0.25**. So w=32 recovers *most* of the gap and the residual is **not resolvable at
n=200** — that is the honest statement, and "w=32 fully restores document context" is
**forbidden**.

**Invalidation fan-out is already quantified; report it, do not re-derive it**
(`summary.json:write_cost`): per random single-token edit, **w/512 extra chunks** =
**0.0625** at w=32, 0.25 at w=128; extra Write FLOPs **+5.73 %** at w32
(`extra_write_flops_ratio_vs_w0 = 1.0573`) and **+22.92 %** at w128 (1.2292);
store/Read/decode **byte-identical** to w0.

**Edit-leg Δ guard.** The recovery *fraction* divides by (E0 − w0) = **7.5 pp**,
which is only **15 discordant items of 200** and can be ≤ 0 on a re-cohort. If the
re-cohort's (E0 − w0) CI includes 0, the fraction is **UNDEFINED** — report absolute
pp only, and K4 rests on the absolute 3.0 pp.

---

## 7. Cost and feasibility — and the reason C=8 is **not** 8× the serial work

### 7.1 The 08-14 cost model measures the wrong thing

Its C=8 term is `17 × 8 × 13.584 s`, i.e. **8 sequential requests**. That is
throughput of a serial loop, not concurrency: with 8 requests genuinely in flight
the wall-clock is bounded by the *bottleneck stage*, not by 8× the service time, and
**the whole scientific question is which stage that is** (§3.1). A cost model that
assumes serialisation has assumed the answer.

### 7.2 C=8 is infeasible as 8 processes on one card — verified arithmetic

- Qwen3-8B bf16 weights = `model.safetensors.index.json:metadata.total_size` =
  16,381,470,720 B = **15.256 GiB**.
- Measured serve peak at `128k|cpu` = **17.29 GiB** (`peak_gpu_serve_comem_gb`,
  median of 3 procs; per-proc 17.57/17.29/17.29) ⇒ per-request transient
  **≈ 2.03 GiB**.
- **8 separate processes:** 8 × 17.29 = **138.3 GiB > 95.00 GiB usable H20** → **OOM.**
- **8 concurrent requests in ONE process** (weights loaded once):
  15.256 + 8 × 2.03 = **31.5 GiB** → **fits with ~63 GiB to spare.**

⇒ **C=8 must be in-process concurrency** (threads / CUDA streams / a batched read),
which is exactly the code that does not exist. The archived 3 procs were **one per
GPU** via the flock pool (`scripts/_run_p1_8_serving.sh:151-177`, and its own header
at `:32-33`: *"1 GPU is SUFFICIENT … Each (L, tier, proc) is a self-contained
single-GPU unit"*), never 3 on one card.

### 7.3 Budget

Under the only feasible design, the C=8 leg's wall-clock is between **1×** (perfectly
parallel) and **8×** (fully serialised) the C=1 leg's per-iteration cost, and *which
one it is* is the measurement. So the cost is stated as a **range with a hard ceiling**:

```
per-iteration measured compute, TTFT-relevant components only,
both arms, G in {1,128}:  Sigma per_query_s medians = 13.584 s   [see below]
per proc: setup 120 s + W 6.824 + I 0.021
          + 17 iters x 13.584 (C=1 leg)
          + 17 iters x 13.584 x k (C=8 leg), k in [1, 8]
  k=1 -> 588 s = 9.8 min      k=8 -> 2205 s = 36.8 min
x 3 procs:  0.49 GPU-h  ...  1.84 GPU-h
```

> **BUDGET: 0.5–1.9 GPU-h point range; 3 GPU-h hard ceiling (1.5× the worst case).
> ONE GPU at a time.** Exceeding 3 GPU-h means the harness is mis-built — stop and
> re-scope, do not keep buying.

The k=8 endpoint reproduces the 08-14 figure exactly (1.8377 GPU-h), so REV-1's
ceiling is not looser; it just no longer *presumes* serialisation.

**Anchor cross-check (independent route).** Σ of the same medians over the original
full G-grid {1,32,128,512} = **60.96 s** of measured compute per unit; the archived
wall clock was `selfcheck.out` 17:52:33 → `aggregate.out` 18:17:26 ≈ **25 min for 18
units on 8 GPUs ⇒ ~11 GPU-min/unit**, against ~1.0 min of pure per-query compute
plus ~2 min model load and the 6.8 s write — consistent once setup dominates. Both
routes agree that a unit is ~10 GPU-min, so the rate is real.
`gpu_cost_estimate.value = "UNKNOWN -- 需先做 1-cell 计时"` was **over-cautious**:
the 1-cell timing was already on disk.

---

## 8. Node requirement — `sm_90` (H20), load-bearing

K1d's entire value is that C=1 **reproduces** `128k|cpu|G=1` = 687.40 / 935.01 ms
TTFT. That anchor was measured on **`.104`, 8×H20, cc 9.0, torch 2.13.0**
(`status/P2_2_PERSISTENT_STORE_IO.md:17-19`, same cohort for P1.8), and A02 states
the constraint explicitly (`A02_STORAGE_READCOMPUTE_VERDICT.md:300-302`):

> «**Machine-specific**: H20 (cc 9.0), torch 2.13.0, bf16, sdpa. The **L20A nodes
> would give different absolute latencies**; the ratios should be more stable.»

- ⇒ **MUST run on `sm_90`: `.73` / `.82` / `.104`.** On B200 (`sm_100`, LOCAL/`.212`)
  hardware drift would enter a latency comparison whose reference point is an H20
  number: K1d becomes uninterpretable and the 123.2 ms bar meaningless.
- ⚠️ **Cross-disk prerequisite.** The three H20s are **zwfy6**-resident; all five
  reusable harnesses are verified present on **wzc1** and **UNVERIFIED (not absent)**
  on zwfy6 — `/apdcephfs_zwfy6` is not mounted on LOCAL and this task was barred from
  ssh (`memory/two-disk-rule-applies-to-main-too.md`). Dispatch requires `scp -O`
  (`.82`'s sftp subsystem is broken) + md5 verify, then
  `PROJECT_ROOT=/apdcephfs_zwfy6/…`, `PYTHON_BIN=/opt/conda/envs/torch-base/bin/python`.

---

## 9. Read-out, pre-registered

**PRIMARY read-out = cell (`C=8`, `G=1`, store `128k`, tier `cpu`), statistic
Δ of TTFT medians (TTFT = `fetch_s + read_s`), n = 3 procs × 15 repeats × 8 in-flight
= 360 requests per arm, unpaired bootstrap CI95 (B=20000, seed 42).**
Committed **before** data. Evaluated **once**, on that cell.

- **G=128 is a REPORTED companion, not a second chance at the primary** and not a
  kill clause (that was the 08-14 K2 defect).
- **p99 / p90 are reported diagnostics**, never kill criteria at n < 500 (§2.2).
- The full {C, G} grid is reported but is **not** the decision point — same
  discipline as paperC, where `--max_steps 200000` was never the decision point and
  the pre-registered read-out was step 121000.
- **Selector time is reported as its own column**, separate from `read_s` (§3.1).

### 9.1 Branches — both spelled out

**if_killed** (K1a/b/c fires, or K4 fires on the edit leg):
1. Write `B07_K1_VERDICT.md` recording Δ₈, its CI, the per-stage breakdown, and
   **which stage was the bottleneck**;
2. append `lifecycle: dead` + `kill_gate_fired_*` to `STATUS.json` (append-only) and
   set the concurrency leg closed;
3. **B07 collapses to the edit leg**, whose surviving content is 0-GPU and is the
   asymmetry named in `RELATED_WORK.md:245-252`: *a depth-j residual admits no exact
   local edit, whereas KV-level caches admit a closed-form RoPE splice.* Per that
   file's own §6 that is **at most a short-paper section inside Paper A, not a
   systems paper** — so the correct action is a `PROTOCOL_NOTE.md` (the B10 pattern),
   **not** a new proposal and **not** more GPU;
4. **do not** re-litigate on another cell of the same grid — nested-ladder re-reads
   are the error retracted twice already (B10 Retractions 6/7).

**if_survives** (Δ₈ ≥ 123.2 ms, CI excludes 0, sign positive, K1d reproduced, K2
resolved):
1. Write `B07_K1_VERDICT.md` with the same contents plus the amplification factor
   Δ₈/Δ₁ **only if** the §3.2 guard permits;
2. **the tiering leg is still DROPPED unless K3's >10 % fetch share appeared** — a
   passing K1 does not resurrect it;
3. **the surviving claim is exactly the one sentence in `RELATED_WORK.md:236-243`**,
   and **Apt-Serve (PACMMOD 2025) must be cited as the nearest system**, not as a
   distant relative — it already caches hidden states and measures TTFT under
   concurrency;
4. next step is **not** more concurrency points; it is the **matched-quality**
   precondition (§10, `paperA/TODOList.md:763`: BM25 −11.56 pp CI [−14.44, −8.67];
   dense TIE −1.0 pp p=0.637 — *「最好情形是打平，从不是赢」*). A latency win at
   unmatched quality is not a result.

### 9.2 Deviations

Any deviation from §3/§4/§6/§9 discovered after the run must be recorded in
`B07_K1_VERDICT.md` under a heading **"DEVIATIONS FROM PRE-REGISTRATION"**, with
the pre-registered rule quoted verbatim beside what was actually done. Thresholds
may **not** be re-derived after seeing Δ₈.

---

## 10. Struck from PROPOSAL.md by evidence already on disk (0 GPU narrowing)

| PROPOSAL.md item | disposition | evidence |
|---|---|---|
| "edited chunk only" update strategy | **STRUCK, wrong-by-construction** | `p0_17_e2_overlap/summary.json` + `e2_sanity.json` max_abs 0.0: it ships the 92.5 arm vs 100.0 |
| HBM/CPU/NVMe/network tiering | **out of headroom pending K3** | worst measured backend is 4.09× the model's 8-GPU throughput; fetch is 1.55 % of TTFT |
| "CoMem is a storage method" | **DEAD, do not re-litigate** | A02 clause (c) FAILED: 2048× vs pre-registered 100× (8192 B/tok vs 4 B/tok) |
| serving win at any quality | **matched-quality or not claimed** | `paperA/TODOList.md:763` P0.20 |
| `c1_all` as comparator | **FORBIDDEN** | A02 §5: hands 86–93 % of the apparent win to plain retrieval; also OOMs at 1M (score arm-absent, never a CoMem win) |
| `per_query_s` (total latency) as the gated statistic | **BANNED (new in REV-1)** | 71 % of it at G=128 is decode, which is identical in both arms by construction; it decays to +13.08 ms at G=512 by dilution alone |
| "concurrency = 8× the serial work" cost model | **BANNED (new in REV-1)** | assumes the answer to the only question being asked; and 8 procs × 17.29 GiB OOMs a 95 GiB H20 |
| versioning / reuse-aware admission / "production system" as *claims* | **DEMOTED to engineering** | `RELATED_WORK.md:302-330`: Leyline + Models-Take-Notes own the primitive; RAGCache (TOCS) + Cache-Craft (SIGMOD) + PRISM own admission |

---

## 11. What still blocks GPU (honest list)

REV-1 closes the **statistic** defects. It does **not** make B07 `ready_gpu`:

1. **The concurrency axis does not exist in code, and REV-1 raises the bar on what
   must be built.** Verified this session: `scripts/bench_p1_8_serving_curve.py`
   matches **zero** of
   `concurren|ThreadPool|ProcessPool|threading|asyncio|multiprocessing|queue\.|p95|p99|ttft|TTFT|time_to_first`.
   The implementation must additionally (a) be **in-process** (§7.2 — the process
   design OOMs), (b) emit **per-request TTFT**, not pooled component medians,
   (c) time the **selector separately** (§3.1), and (d) record `n_gen` per draw
   (the harness computes it at `:280`/`:341` but **drops it** — verified absent from
   the archived per-proc JSON, so early-EOS cannot currently be ruled out post hoc).
   **All of this is 0-GPU work.**
2. **No mutable-store code path exists at all** — see `STATUS.json`'s new
   `blocking_dependency_20260815`. Greps returning **zero** matches across `src/` and
   `scripts/`: `content_hash|chunk_hash|store_version|version_id|generation_id|epoch_id`,
   and no `def update|edit|invalidate|delete|evict|replace|patch|rewrite|upsert` in
   `src/memory/qcmem/` (the single `update` at `qcmem_model.py:1091` is
   `_CacheBlendSparseCache.update`, a per-forward duck-typed KV cache for CacheBlend
   selective recompute — **not** a store mutation API). The store is built once by
   `_build_store` (`bench_p1_8_serving_curve.py:178-216`) and never written again; the
   only file-backed store writer opens with **`O_TRUNC`** (`bench_persistent_store_io.py:248`)
   = whole-store rewrite, no partial/offset write. **The edit leg has no apparatus,
   only the E2 evidence it can reason from.**
3. **Matched quality is unestablished** (§9.1 branch 4). This bounds any surviving
   claim regardless of K1's outcome.
4. **`ready_queue.py` cost-key precedence.** `KILL_KEYS` gained its dated slot on
   2026-08-15 (`ready_queue.py:139-179` — B07's own `_precedence_warning` predicted
   the bug and the fix landed), but the cost list at `:596`
   (`["gpu_cost_estimate","cost","cost_to_first_result"]`) still has **no** dated
   slot, so the report shows the stale `UNKNOWN -- 需先做 1-cell 计时`. One-line fix,
   **deliberately not made here** (out of scope; it edits the scheduler, not the
   proposal). Reporting artefact, **not** a real blocker.

**Honest lifecycle: `ready_cpu`.** Unchanged. The actionable next task is the
in-process concurrency harness (item 1), and it costs no cards.

---

## 12. Provenance — every file opened for this revision

| artefact | what was taken from it |
|---|---|
| `paperA/artifacts/p1_8_serving/p1_8_serving_aggregate.json` | `cells["128k\|cpu"]`: all four `per_G` rows, component medians and p90s, one-time W/I, peaks, `crossover` |
| `paperA/artifacts/p1_8_serving/serve/serve_128k_cpu_niah_multikey_1_proc{0,1,2}.json` | **raw per-iteration draw lists** (n=5 each) → all CIs, true p90s, p99-estimability, `read_len` pairing, `config.n_repeat`, LoRA sha |
| `paperA/anonymous_artifact/scores/p0_17_e2_overlap/{summary,stats,e2_sanity}.json` | K4 arms, CIs, McNemar, `write_cost`, `max_abs = 0.0` |
| `ruler_results/p2_2/{p2_2_full,p2_2_file_isolated,p2_2_gpu_fixed}.json` | per-backend QPS at 128k, per file (found the NVMe 256.5-vs-272.2 provenance split) |
| `status/P2_2_PERSISTENT_STORE_IO.md:17-19,73,134` | node/arch/runtime, backend table, "GIL-bound single-copy loop" |
| `proposal/backlog/A02-comem-write-read-repair/A02_STORAGE_READCOMPUTE_VERDICT.md:187-188,201-202,297-302` | select = 117.7 ms both arms, confound 7, machine-specificity |
| `scripts/bench_p1_8_serving_curve.py:178-216,226,289,385-560,605-700` | `_build_store`, `_serve_comem`/`_serve_j0`, sequential `_measure` calls (⇒ unpaired), `n_gen` dropped, aggregation construct |
| `scripts/bench_persistent_store_io.py:248` | `O_TRUNC` whole-store rewrite |
| `scripts/_run_p1_8_serving.sh:32-33,151-177` | one proc per GPU via flock pool |
| `src/memory/qcmem/qcmem_model.py:374-395,470-560,1075-1097` | `write_chunk` chunk-locality, `read_core` is `[1,…]` single-sequence, CacheBlend cache is not a store API |
| `models/Qwen3-8b-local/model.safetensors.index.json` | 15.256 GiB weights → the C=8 memory arithmetic |
| `proposal/backlog/B07-mutable-comem-serving/RELATED_WORK.md:232-330` | safe residual claim, per-feature disposition, Apt-Serve as nearest system |
