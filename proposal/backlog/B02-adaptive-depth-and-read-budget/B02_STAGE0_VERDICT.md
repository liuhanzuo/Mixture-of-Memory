# B02 — Stage 0 verdict: per-example oracle headroom for query-adaptive `resume_j`

> Confirmatory run, 2026-08-14. Protocol: `FIXED_SAMPLE_PROTOCOL.md` v0.2/v1.0
> (pinned before data). Node `.73` (zwfy6, 8×H20). Evidence:
> `evidence/b02_confirmatory_vt16k_n200.json`, `evidence/b02_confirmatory_vt32k_n200.json`.
> Total GPU cost: **6.8 GPU-h** (pilot 0.3 + confirmatory 6.5).

---

## 1. What was blocking B02, and what actually caused it

`STATUS.json` recorded that the eight T21 `resume_j` cells share **0/50** samples,
which makes a per-example oracle uncomputable. That symptom was correct. The
**cause** had not been identified, and it turns out to be a harness bug that was
already fixed:

- `eval_ruler_qcmem.py` derives its per-cell seed as
  `base_seed = args.seed + (zlib.crc32(f"{task}\x00{length}") % 100000)`.
- The `zlib.crc32` arrived in commit **`d1e1389`, 2026-08-03**
  ("stable PYTHONHASHSEED-independent per-(task,length) seed so shards/arms share
  sample set"). Before it, the expression used Python's **per-process salted
  `hash()`**.
- **T21 ran 2026-07-20 — 14 days earlier** — and its cell JSONs record
  `"pythonhashseed": null`. Each of the eight arms therefore drew a *different*
  `base_seed`.

So B02 never needed new pairing machinery. It needed to re-run on post-`d1e1389`
code and then **assert** byte-identity instead of assuming it.

Verified before spending GPU: two independent processes produce identical
`base_seed = 63881` and byte-identical `input_ids_sha256`; `base_seed` does not depend
on `--limit`, so sample sets are prefix-nested in `n` and identical across `resume_j`.

## 2. Integrity (fail-closed, asserted before any statistic)

| assertion | 16k | 32k |
|---|---|---|
| shard index **set** == `{0..S-1}` | pass | pass |
| item count exact (200) per cell | pass | pass |
| duplicate `sample_index` | 0 | 0 |
| NaN / out-of-range recall | 0 | 0 |
| cross-arm `input_ids_sha256` identical | **pass, all 200 items × 8 arms** | **pass** |
| `chat_template` | **`False`** | **`False`** |
| `enable_thinking` / `selector` / `lora_adapter` | `False` / `iter_bm25` / `None` | same |
| `oom_count`, `status` | 0, `completed` | 0, `completed` |

The cross-arm hash assertion is the one whose absence produced the T21 defect. It
now passes on real GPU output, so the sweep is genuinely paired.

## 3. The result: raw headroom is large, and it is an artefact

**16k**, fractional recall, n=200:

| quantity | value |
|---|---|
| per-example oracle | 0.591 |
| best fixed config (`j27`) | 0.342 |
| **raw headroom** | **+0.249** ( +24.9 pp ) |
| **independence floor** `E[oracle | no coupling]` | **0.630** |
| **`Δ_excess` = oracle − null** | **−0.039** |
| CI95 (10 000 permutations) | **[−0.059, −0.019]** |
| p (two-sided) | **0.0008** |

Binary scale agrees: oracle 0.360, best_fixed 0.275, raw +8.5 pp, floor 0.395,
`Δ_excess = −0.035`, CI95 `[−0.060, −0.010]`, p = 0.0152.

**Reading.** A naive Stage-0 analysis would have reported "+24.9 pp of oracle
headroom" and concluded there is large upside in a query-adaptive depth router.
That is wrong. Once the max-over-8-noisy-columns floor is computed, the observed
oracle lands **below** it, and the interval **excludes zero with p < 0.001**.

This is not an underpowered null. The sign is negative and significant, which
carries positive information: **`resume_j` configs are positively coupled.** Items
that VT gets wrong at one depth tend to be wrong at *every* depth. A per-query
depth selector therefore has strictly **less** to exploit than eight independent
configs of the same marginal accuracy would offer. The apparent +24.9 pp is
max-over-noise plus shared item difficulty, not routable signal.

## 4. Verdict

**B02's own kill clause fires.** `PROPOSAL.md` Stage 0: *"若 oracle 相对最佳 fixed
config 的收益不足，方向关闭"*. The gate is met in its strongest form — not merely
"insufficient gain" but a **negative, significant excess** over the null.

Scope of the claim, stated precisely:

- **Established:** on RULER `variable_tracking`, Qwen3-32B zero-shot, `chat=False`,
  `iter_bm25`/topk12, the per-example oracle over `resume_j ∈ {3..48}` does **not**
  exceed its independence floor at 16k or 32k; it falls significantly below it.
  A per-query **depth** router over this ladder has no headroom to capture.
- **Not established:** nothing about the *other* axes B02 proposed (evidence budget
  `k`, retrieval rounds, raw-replay vs CoMem vs reusable-KV, low-confidence
  fallback). Those were never measured here. This gate closes the **split-depth `j`**
  axis only, which is the axis the SOURCES sweep could speak to.
- **Not established:** generalisation beyond VT. VT is the most `j`-sensitive and
  noisiest RULER task. The T21b evidence that niah tasks are *flat* in `j` up to the
  cliff actually makes complementarity **less** likely there, not more — a flat
  quality curve gives a depth router nothing to choose between — but that is an
  argument, not a measurement.

**Recommendation: B02 stays in `backlog`, with the `j`-axis marked closed by
measurement.** It should not be promoted, and it should not be re-run at larger `n`:
the effect is not "too small to see", it is the **wrong sign**.

## 5. A methodological finding worth more than the verdict

The pilot falsified **this protocol's own pre-registered primary null**, one commit
after it was registered. v0.1 named a both-margins-preserving (curveball) null as
primary. That null is **provably degenerate** for a binary oracle:

> `max_j M[i,j] = 1[rowsum_i >= 1]` for binary `M`, so the oracle is a function of
> the **row margins alone**. Curveball preserves row margins exactly. The statistic
> is therefore **exactly invariant** under the null, for every draw.

Measured `null_B sd = 2.2e-16` (pilot) and `5.6e-17` (confirmatory) — floating-point
zero, twice. Had v0.1 been executed as written, the primary gate would have returned
`p = 1.0` on **every** dataset and been misread as "the kill clause fires" for reasons
that had nothing to do with `resume_j`.

Generalisation: **for a binary per-item oracle, "item difficulty" and "oracle value"
are the same quantity**, so no row-margin-preserving null can separate
complementarity from difficulty. The admissible null is column-permutation; the
analyzer now carries a degeneracy guard that refuses to emit a verdict from a
zero-variance null, and it was validated against a positive control (disjoint
winners → p = 0.0) and a negative control (independent columns → p = 0.74).

## 6. Cost

| stage | cells | GPU-h |
|---|---|---|
| exploratory pilot (16k, n=20) | 8 | 0.32 |
| confirmatory (16k + 32k, n=200) | 16 | 6.47 |
| **total** | 24 | **6.79** |

Well inside the 24 GPU-h ceiling, so the design was chosen for statistical power
rather than trimmed for budget. Nodes `.104`, LOCAL and `.212` were not touched.
