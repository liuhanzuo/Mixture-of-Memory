# B02 — FIXED SAMPLE PROTOCOL (pre-data)

> **Status: PRE-DATA.** Committed before any confirmatory number exists.
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
