# B02 — FIXED SAMPLE PROTOCOL

> **v0.2, 2026-08-14.** §5 amended: the exploratory pilot **falsified this
> protocol's own pre-registered primary null (Null B)**, which is provably
> degenerate for a binary oracle. See §5b. The amendment was made *before* any
> confirmatory run, and the falsifying pilot is archived at
> `evidence/b02_pilot_EXPLORATORY_vt16k_n20.json`. v0.1 (commit `b983c8f`) is the
> pre-data record; this file supersedes only its §5.
>
> **Status: PRE-CONFIRMATORY.** Committed before any confirmatory number exists.
> Satisfies `STATUS.json → required_before_stage0[0]` ("fix n and a per-config RNG
> seed protocol so the same items are scored under every `resume_j`").
> Version 0.1 pre-registers the *mechanism*, the *estimator*, the *nulls* and the
> *decision rule*. `n` is left to a labelled exploratory pilot (§6); it is pinned in
> v1.0 before the confirmatory run, and pilot items are **not pooled** into the
> confirmatory estimate.

---

## 1. What went wrong in T21 (root cause, now identified)

`STATUS.json → premise_falsified` recorded the *symptom*: the eight T21 `resume_j`
cells share **0/50** samples. This protocol adds the *cause*, and it changes the
remedy from "invent a new pairing scheme" to "re-run on the already-fixed code".

`scripts/eval_ruler_qcmem.py` derives the per-cell RNG seed as

```python
base_seed = args.seed + (zlib.crc32(f"{task}\x00{length}".encode()) % 100000)
```

That `zlib.crc32` was introduced by **commit `d1e1389`, 2026-08-03 13:41:07 +0800**
— `fix(ruler): stable PYTHONHASHSEED-independent per-(task,length) seed so
shards/arms share sample set`. Before it, the expression used Python's builtin
`hash()`, which is **per-process salted** unless `PYTHONHASHSEED` is pinned.

T21 ran **2026-07-20 04:34–04:54**, i.e. **14 days before the fix**, and its cell
JSONs record `"pythonhashseed": null` (verified in
`.73:ruler_results/qcmem_32b_t21_vt_j3/t21/variable_tracking_16k.json`). Each of the
eight `resume_j` processes therefore drew a *different* `base_seed`, which is
precisely why the sample sets are mutually disjoint.

**Consequence for B02: no new pairing code is required.** The pairing defect was a
harness bug that is already fixed on both disks. B02's re-run only has to (a) run on
post-`d1e1389` code, (b) hold `--seed` and `--limit` fixed across arms, and (c) assert
byte-identity rather than assume it.

## 2. Verified pairing mechanism

Two properties were measured on `.73` (zwfy6) at HEAD `2d98c5a`, which contains the
fix, **before** spending any GPU:

**(a) Cross-process determinism.** Two independent Python processes (different PIDs,
`PYTHONHASHSEED` unset) built RULER `variable_tracking` 16k samples via the same code
path the eval driver uses:

```
pidA 3859098  pidB 3859603   base_seed 63881 63881
items identical: True
i=0  384ee35edd23a3e1 == 384ee35edd23a3e1   n_tok=16287
i=1  4ffc609ef32daa1e == 4ffc609ef32daa1e   n_tok=16286
i=2  e1e8e34f13a19e15 == e1e8e34f13a19e15   n_tok=16286
i=3  b565ad6e58cbc21a == b565ad6e58cbc21a   n_tok=16288
```

Probe: `scripts/_b02_determinism_probe.py` (CPU only, loads tokenizer not weights).

**(b) `n`-independence of the sample set.** `base_seed` is a function of
`(task, length, args.seed)` only, and item `i` is built from
`random.Random(base_seed * 1000 + i)`. **Neither depends on `--limit`.** The VT
in-context example is likewise `_make_vt_icl(random.Random(base_seed + 777), 4)`.
Therefore the item set for `--limit n` is a strict **prefix** of the set for any
`n' > n`: sample sets are *nested* in `n` and *identical* across `resume_j`.
A grep confirmed `scripts/eval_ruler_mem_space.py` uses **no** module-level
`random.seed` / `np.random` in the sample-construction path, so there is no hidden
global-RNG channel that process ordering could perturb.

This is what makes the sweep paired **by construction**, and it is why (b) also lets
us grow `n` later without invalidating already-scored items.

## 3. Frozen identity axis

| field | value | rationale |
|---|---|---|
| `task` | `variable_tracking` | the T21 axis B02 inherits |
| `--seed` | **42** | harness default; also T21's recorded `seed` |
| `PYTHONHASHSEED` | **`0`, exported explicitly** | belt-and-braces: makes the run reproducible even on pre-`d1e1389` code |
| item identity | `sample_index` **plus** `input_ids_sha256` | index alone is not identity; the hash is the fail-closed gate |
| `--limit` | identical across all arms of a confirmatory run | prefix-nesting only holds arm-to-arm if `n` matches |

`resume_j` is the **only** free variable. Everything else — `selector=iter_bm25`,
`topk=12`, `chunk_size=512`, `sink_tokens=bos`, `max_new_tokens`, dtype `bfloat16`,
`attn_impl=sdpa`, no LoRA adapter, `models/Qwen3-32B` (L=64) — is held fixed and
re-asserted from each cell's emitted JSON.

## 4. Integrity assertions (fail-closed, run before any statistic)

The harness already emits `{task}_{length}[_shard...].records.json` carrying
`sample_index`, `input_ids_sha256`, `recall`, `correct`, `n_tok` (added by
`d15252b`, the P0.19 pairing leg). The analyzer **must** assert, and abort on failure:

1. **Shard completeness** — if sharded, the observed shard-index **set** equals
   `{0..S-1}` exactly (a *set*, not a count: a silent 5-of-8 merge that happens to
   contain a duplicate would pass a count check).
2. **Item count** — `len(records) == limit` exactly, per cell.
3. **Zero duplicates** — `sample_index` values unique within a cell.
4. **Zero NaN** — every `recall` is a finite float in `[0, 1]`.
5. **Cross-arm byte identity** — for every `sample_index` present in ≥2 arms,
   `input_ids_sha256` is equal across those arms. **This is the assertion whose
   absence produced the T21 defect.** Any mismatch aborts; it is never repaired by
   dropping items.
6. **口径** — each cell JSON satisfies `chat_template is not True` for the
   `chat=False`口径 (and the recorded value is carried into the evidence file, never
   assumed), `enable_thinking is False`, `selector == "iter_bm25"`, `lora_adapter is None`.
7. **Identity axis** — `seed`, `topk`, `chunk_size`, `model_path`, `num_hidden_layers`
   identical across arms; only `resume_j` differs.

## 5. Estimator and — the load-bearing part — its null floor

`PROPOSAL.md` Stage 0 says: *"若 oracle 相对最佳 fixed config 的收益不足，方向关闭"*.
**Taken literally that rule is not decidable, and pre-registering it unguarded would
manufacture a false positive.** With `C` configs and binary per-item correctness, the
per-example oracle is upward-biased by construction: even with **zero** exploitable
per-item structure, independent configs with marginals `p_1..p_C` give

```
E[oracle_null] = 1 - Π_j (1 - p_j)
```

Using T21's own 16k marginals as an order-of-magnitude illustration
(`p = .936, .392, .140, .480, .264, .332, .036, .020`) this evaluates to
**≈ 0.992** versus a best-fixed-config of **0.936** — i.e. ~5.6 pp of "headroom"
that is *entirely* an artefact of taking a max over 8 noisy columns. Reporting that
5.6 pp as router headroom would be exactly the
"a range is not a measurement until it clears its floor" error.

So the pre-registered quantity is the **excess over a null**, not the raw oracle:

- `oracle_obs = mean_i max_j correct(i, j)`
- `best_fixed = max_j mean_i correct(i, j)`
- `raw_headroom = oracle_obs − best_fixed`  ← **reported but never used as the gate**
- **Null A (independence / column-margin-preserving).** Independently permute each
  config's correctness column across items. Preserves every `p_j`, destroys item×config
  coupling. Gives `oracle_null_A`.
- **Null B (both-margins-preserving, the primary null).** Condition on **both** the
  item margins (`some items are simply easy`) and the config margins, via
  curveball/swap randomisation of the binary item×config matrix. Null B is primary
  because Null A would credit a router for the trivially-predictable fact that easy
  items are easy under every `j`.
- **Gate statistic** `Δ_excess = oracle_obs − oracle_null_B`, with a two-sided
  permutation interval from `B = 10000` draws.

Decision rule, pre-registered:

| outcome | reading |
|---|---|
| `Δ_excess > 0`, interval excludes 0 | configs are **complementary** per item → real router headroom → Stage 0 proceeds |
| interval contains 0 | **no per-item interaction beyond chance** → the oracle is a max-over-noise artefact → B02's own kill clause fires |
| `Δ_excess < 0`, interval excludes 0 | configs are **positively** coupled (hard items hard everywhere) → strictly worse than no-signal; direction closes |

`regret = oracle_obs − best_fixed` is reported **only** alongside `Δ_excess` and the
null, never alone. A realizable router is bounded above by the oracle, so a null-floor
failure kills the direction regardless of how large `raw_headroom` looks.

**Bootstrap determinism caveat.** Cross-node bootstrap is *not* reproducible on this
cluster: the five nodes carry three different numpy versions
(LOCAL 2.3.5 / `.82` 2.4.6 / `.73`+others 2.5.1) and same-seed `multinomial` diverges.
All permutation/bootstrap draws for B02 are therefore pinned to **one node**, that node
and its `numpy.__version__` are recorded in the evidence JSON, and the resampler uses
an explicitly seeded `numpy.random.Generator(PCG64(seed))` rather than legacy global state.

## 5b. AMENDMENT — the pilot falsified Null B, this protocol's own primary null

The exploratory pilot (§6, VT 16k, n=20, all 8 `j`) was run and analysed. It produced
one result that matters more than any number about `resume_j`: **the primary null this
protocol pre-registered in §5 is mathematically degenerate, and the analyzer measured
its degeneracy rather than hiding it.**

Measured on the pilot:

```
null_B (both margins, declared PRIMARY):  sd = 2.22e-16   CI95 = [0.0, 0.0]   p = 1.0
```

An `sd` of 2e-16 is floating-point zero. The reason is a proof, not a sampling accident:

> For **binary** correctness, `max_j M[i,j] = 1[rowsum_i >= 1]`. The oracle
> `mean_i max_j M[i,j]` is therefore a function of the **row margins alone**.
> Curveball / swap randomisation preserves every row margin *exactly*. Hence the
> oracle is **exactly invariant** under Null B, for every draw, by construction.

Confirmed numerically: the pilot's row sums `[2,1,1,2,0,1,0,3,1,0,0,1,0,0,1,0,2,0,1,1]`
give `12/20 = 0.600`, exactly the observed oracle, and no row-preserving permutation can
move it. The same argument kills any within-row permutation null: `max_j` depends only on
the row's multiset of outcomes.

**This is a real methodological finding about the estimator, and it generalises beyond
B02:** for a binary per-item oracle, "item difficulty" and "oracle value" are *the same
quantity*, so a both-margins null cannot separate complementarity from difficulty at all.
Had this protocol been executed as written in v0.1, the primary gate would have reported
`p = 1.0` on every dataset forever and been misread as "the kill clause fires".

### Amended estimator (v0.2)

1. **Null A (column-margin-preserving) is promoted to PRIMARY** for the binary oracle.
   It is the only admissible null of the two, because permuting columns independently does
   *not* preserve row margins and therefore genuinely tests item×config coupling.
   Verified non-degenerate on the pilot: `null_A sd = 0.0572`.
2. **Null B is retired for the binary oracle** and recorded as a *proven-invariant*
   negative result, not a failed test. It is **not** reported as evidence about `resume_j`.
3. **The confirmatory statistic moves to fractional recall**, `mean_i max_j recall(i,j)`,
   with Null A. Two independent reasons, both from the pilot:
   - *Non-degeneracy:* the fractional oracle is not a pure function of row margins, so
     Null A retains variance (checked: sd ≈ 0.013 on a synthetic 20×8 control).
   - *Information:* the pilot's **binary** marginals were `[.20, .00, .00, .05, .50, .10,
     .00, .00]` — **4 of 8 configs are exactly zero**, i.e. half the ladder contributes no
     signal whatsoever to a binary analysis at this `n`. The **fractional** marginals
     `[.39, .12, .14, .07, .58, .30, .02, .02]` are informative for all 8. VT is scored by
     `string_match_all` (fraction of reference strings matched); binarising at 1.0 discards
     most of what was measured.
4. **Direction of the pilot's Null-A point estimate.** Excess was **negative**
   (`oracle 0.600` vs `null_A mean 0.657`, `Δ = −0.057`, CI95 `[−0.150, +0.050]`, p = 0.58).
   The interval contains 0, so at n=20 this is **not** a verdict — it is consistent with
   both no interaction and mild positive coupling. It is quoted only as the pilot's
   exploratory reading and **must not** be cited as a B02 result. What it does establish is
   that the raw `+0.100` headroom (`0.600 − 0.500`) sits **below** its own independence
   floor of `0.658`, so reporting that `+10 pp` as router headroom would have been the
   exact error §5 was written to prevent.

### Consequence for the confirmatory design

The kill/proceed rule of §5 is unchanged in form; only the null and the outcome scale
change (Null A, fractional recall). The `n` for the confirmatory run must be sized against
Null A's sd at the fractional scale, which the pilot now makes estimable.

## 6. `n`, and the exploratory pilot that sets it

`n` is **not** guessed. Measured cost from the T21 cells makes the sweep cheap enough
that `n` can be chosen for power rather than for budget — per-cell wall time at n=50
on one H20 was 320.6–343.5 s at 16k and 379.8–395.9 s at 32k (all 16 T21 cells,
`elapsed_seconds`, `oom_count=0`), i.e. ≈ **7.0 s/item** at 16k and ≈ **7.8 s/item**
at 32k, scaling linearly in `n`.

A **labelled exploratory pilot** (small `n`, one length) is run first to settle two
things a power calculation cannot be done without:

1. **口径 non-degeneracy.** T21 ran `chat_template=True`; the project口径铁律 for
   reported results is `chat_template=False`. Those are different measurements and
   T21's numbers do **not** transfer. If VT recall under `chat=False` sits at the
   floor for most `j`, the item×config matrix is degenerate and `Δ_excess` is
   undefined — that must be discovered with a ~10-minute probe, not after a full sweep.
2. **Marginals `p_j`** feeding the power curve for `Δ_excess`.

Pilot data are **exploratory and are not pooled** into the confirmatory estimate; the
confirmatory `n` and the length set are pinned in **v1.0 of this file, committed before
the confirmatory run**. Because sample sets are prefix-nested in `n` (§2b), enlarging
`n` later is a valid extension rather than a re-randomisation.

### Pilot outcome (executed 2026-08-14, `.73`, 8 GPUs, 0.32 GPU-h)

VT 16k, n=20, `resume_j ∈ {3,6,13,20,27,34,41,48}`, `chat_template=False` in all 8 cells,
`oom_count=0` in all 8, 142 s/cell wall (7.1 s/item, matching the 7.0 s/item projection).

| `j` | 3 | 6 | 13 | 20 | 27 | 34 | 41 | 48 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| mean recall (%) | 39.0 | 12.0 | 14.0 | 7.0 | **58.0** | 30.0 | 2.0 | 2.0 |
| binary `correct` /20 | 4 | 0 | 0 | 1 | **10** | 2 | 0 | 0 |

Findings, in order of importance:

1. **The primary null was falsified** — see §5b. This is the pilot's main product.
2. **口径 is non-degenerate under `chat=False`.** VT does not collapse to the floor; the
   matrix is informative on the fractional scale for all 8 arms. The §6 degeneracy risk is
   cleared, but only for fractional scoring — the binary matrix *is* half-degenerate
   (4 of 8 marginals exactly 0).
3. **Pairing is confirmed end-to-end on real GPU output.** All 8 arms cover the identical
   `sample_index` set and every cross-arm `input_ids_sha256` matches (item 0 = `2ad1b7914361…`
   in all eight). This is the assertion whose absence caused the T21 defect, now passing.
4. **`chat=True → chat=False` is not a relabelling.** T21's j27 was 26.4 % at 16k under
   `chat=True`; the pilot's j27 is 58.0 % under `chat=False`, and j27 — not the T21 peak
   j3 — is the pilot's best arm. Small `n` accounts for part of this, but it confirms
   T21's numbers cannot be reused for B02 under the current口径 and that the peak's
   *location* along `j` is itself口径-dependent.

## 6b. CONFIRMATORY DESIGN — pinned (v1.0), pre-data

Pinned from the pilot's measured Null-A sd, before the confirmatory run starts.

| item | value |
|---|---|
| **`n`** | **200** items per arm |
| lengths | **16k** and **32k** (the pilot only probed 16k) |
| `resume_j` | `{3, 6, 13, 20, 27, 34, 41, 48}` (all 8, as T21) |
| outcome scale | **fractional recall** (primary), binary reported alongside |
| primary null | **Null A**, column-margin-preserving, `B = 10000` draws |
| cost | **6.44 GPU-h**, ~48 min wall on 8 GPUs |

**Power justification.** The pilot measured `null_A sd = 0.0376` at n=20 on the
fractional scale. A permutation-null sd scales as `1/sqrt(n)`, so:

| `n` | projected Null-A sd | min detectable abs(Δ_excess) (α=.05, 80 % power) |
|---:|---:|---:|
| 50 | 0.0238 | 0.067 |
| 100 | 0.0168 | 0.047 |
| **200** | **0.0119** | **0.033** |
| 300 | 0.0097 | 0.027 |

`n = 200` resolves an excess of `>= 0.033`, i.e. ~3.3 pp of recall. That is the right
target because **a router that beats its own null by less than ~3 pp is not actionable** —
the engineering cost of query-adaptive `j` selection cannot be repaid by a sub-3 pp gain.

**What `n=200` explicitly does NOT buy, stated in advance.** The pilot's point estimate
was `abs(Δ) = 0.0126`. Resolving an effect *that* small at 80 % power would need
`n ≈ 1394` per arm ≈ 22 GPU-h at 16k alone. **We are choosing not to.** So if the
confirmatory run returns an interval containing 0, the honest reading is
**"no effect larger than 3.3 pp"**, *not* "no effect". The kill clause fires on
"no *actionable* headroom", and the write-up must use those words. Pre-committing to this
wording is what stops an underpowered null from being reported as a proof of absence.

## 7. Cost, measured not guessed

Derived from the T21 `elapsed_seconds` above (single H20, one cell per GPU, 8 GPUs
available on the target node):

| design | cells | GPU-h | wall-clock on 8 GPUs |
|---|---|---|---|
| 8 `j` × {16k,32k}, n=50 | 16 | **1.61** | ~13 min |
| 8 `j` × {16k,32k}, n=100 | 16 | **3.22** | ~25 min |
| 8 `j` × {16k,32k}, n=200 | 16 | **6.44** | ~48 min |
| 8 `j` × {16k,32k}, n=300 | 16 | **9.66** | ~72 min |

All four are far under the 24 GPU-h ceiling this task was given, so the sweep is **not**
narrowed for budget. `n` is chosen for power on `Δ_excess`, and the pilot cost is
counted on top.

## 8. Node and environment

`.73` (`28.85.35.73`, zwfy6, 8×H20 sm_90), python
`/opt/conda/envs/torch-base/bin/python` (torch 2.13.0, transformers 5.5.4,
pandas 3.0.3 — import chain verified before launch, since the node has been rebooted).
`.104` / LOCAL / `.212` are **off-limits** for this task and are not touched.
Model `models/Qwen3-32B` on zwfy6 (`num_hidden_layers=64`, `eos_token_id=151645`).

## 9. Products

- fixed-set sweep cells → `.73:ruler_results/qcmem_32b_b02_fixed_*/`
- paired oracle/regret evidence → `proposal/backlog/B02-adaptive-depth-and-read-budget/evidence/`
- `STATUS.json` updated **append-only** (existing keys untouched)
