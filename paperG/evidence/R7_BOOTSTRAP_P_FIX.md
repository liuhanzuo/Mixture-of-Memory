# R-7 — the illegal bootstrap p-value: root cause, fix, and re-emission

**Date**: 2026-08-11 · **Task**: #247 · **GPU used**: 0 (pure CPU re-merge of on-disk shards)
**Code**: `proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py`
**Evidence re-emitted**: `proposal/active/A01-null-calibration-methodology/evidence/gate3_dtype_runs/*_dtype_summary.json` (6 files)
**Status**: **CLOSED**. Headline: the p-value is now legal, and **no verdict anywhere changed**.

---

## 1. The defect

`evidence/gate3_dtype_runs/7B_base_dtype_summary.json` carried

```json
"contrast": { ..., "letter_acc_diff_boot_p": 1.042 }
```

A p-value cannot exceed 1. Raised as **R-7** by the 2026-08-10 external audit
(`proposal/active/A03-parametric-vs-external-memory/evidence/TCODEX_AUDIT_20260810.md` §7),
accepted as a real defect in `TCODEX_AUDIT_RESPONSE.md` §7 but explicitly *not*
fixed in that pass ("needs the estimator patched, not a doc edit"), and carried
through the promotion to `paperG` as open defect #1.

## 2. Root cause — it is **not** a missing clamp

The audit's own guess was "almost certainly a doubled one-sided tail without a
`min(1, ·)` clamp". That is the *symptom*. The actual mechanism is a **double-count
of the atom at zero**.

The construction, which appeared in **two** places
(`analyse()` for the dtype contrast, and `paired_bootstrap()` for every
`*_vs_null_boot_p`), was

```python
p = 2 * min((bs <= 0).mean(), (bs >= 0).mean())
```

where `bs` holds the 10 000 bootstrap resample means. `<=` and `>=` **both** include
the resamples whose mean is exactly 0, so

```
(bs <= 0).mean() + (bs >= 0).mean() == 1 + P(bs == 0)
```

rather than `1`. Once `P(bs == 0)` is large enough that the *smaller* of the two
tails exceeds 0.5, doubling it exceeds 1. For a genuinely continuous `bs` the atom
is empty and the formula is fine — which is why this sat undetected.

**Why the base arm and only the base arm.** Here `d = L_fp32 − L_bf16` is the
difference of two 0/1 correctness vectors, so it is integer-valued in {−1, 0, +1}
and overwhelmingly 0. The base arm is precisely the arm whose letter accuracy is
byte-identical across dtypes (0.6053980914399658 in both), with only
b = 28 bf16-right/fp32-wrong and c = 28 bf16-wrong/fp32-right discordant items out
of 14 042 — i.e. `d ≡ 0` except for 56 items that cancel exactly. A bootstrap mean
of such a `d` lands *exactly* on 0 with substantial probability.

**Verified on the real per-example shards** (not simulated), for all six arms:

| arm | b | c | #(d == 0) | **P(bs == 0)** | p_old | p_new |
|---|---|---|---|---|---|---|
| `base` | 28 | 28 | 13986 | **5.440%** | **1.0420** | 0.9876 |
| `keep8_step121000` | 631 | 610 | 12801 | 1.070% | 0.5598 | 0.5491 |
| `keep10_step83500` | 282 | 253 | 13507 | 0.830% | 0.2260 | 0.2177 |
| `keep12_step124000` | 562 | 623 | 12857 | 0.200% | 0.0748 | 0.0728 |
| `keep14_step200000` | 470 | 619 | 12953 | 0.000% | 0.0000 | 0.0001 |
| `shortgpt16_step200000` | 226 | 279 | 13537 | 0.120% | 0.0174 | 0.0162 |

The base row reproduces the archived `1.0420` **bit-for-bit** (and the archived
CI95 `[−0.0010682238997293833, +0.0010682238997293833]`), which confirms the
mechanism rather than merely being consistent with it. Arithmetic check on the real
records: `(bs<=0).mean() = 0.5210`, `(bs>=0).mean() = 0.5334`, sum
`= 1.0544 = 1 + 0.0544 = 1 + P(bs==0)`, and `2 × 0.5210 = 1.0420` — exactly the
archived value. The atom size also matches the analytic expectation for the
discordant-pair count: a Skellam(28, 28) has `P(=0) ≈ 1/√(2π·56) = 5.33%`.

**So the answer to "is the stated root cause correct?" is: yes on the mechanism
(unclamped doubled tail with 0 double-counted), and the initial framing "just add
`min(1,·)`" is wrong about the remedy** — see §3.

## 3. The fix

New shared helper `two_sided_boot_p(bs, n_boot)`. It keeps the same estimand
(doubled smaller tail) but splits the zero atom evenly between the two tails —
the standard **mid-p / half-correction** for a discrete null:

```python
tie  = (bs == 0).mean()
p_lo = (bs < 0).mean() + 0.5 * tie
p_hi = (bs > 0).mean() + 0.5 * tie      # p_lo + p_hi == 1  exactly
p    = min(1.0, max(2 * min(p_lo, p_hi), 1.0 / n_boot))
```

**Why this construction and not the alternatives.**

* Because `p_lo + p_hi == 1` exactly, `min(p_lo, p_hi) ≤ 0.5`, so `p ≤ 1` is
  **structural** — a property of the estimator, not a truncation applied after the
  fact. That is the whole point: `min(p, 1.0)` would have produced a legal-looking
  number for the base arm while leaving the doubled-zero-mass bias intact in every
  *other* near-null value that merely failed to cross 1 (e.g. `keep8` 0.5598,
  `keep10` 0.2260 were all inflated by the same bug, just not visibly).
* The centred alternative `p = (Σ|bs − mean(bs)| ≥ |mean(bs)| + 1)/(n_boot + 1)`
  was rejected because it changes the estimand from a percentile-tail p to a
  symmetric-deviation p, which would not be comparable with the CI95 already
  reported next to it in the same JSON, and would silently redefine the
  `*_vs_null_boot_p` values that A01/paperG's floor verdicts are keyed to.
* **Behaviour at `d ≡ 0`** (complete absence of any difference): `bs ≡ 0`, so
  `p_lo = p_hi = 0.5` and **`p = 1.0` exactly** — the correct answer. The old code
  returned `2.0` in that limit, which is what 1.042 is an instance of.
* Range is now `[1/n_boot, 1]` in all cases (lower floor retained from the old
  `paired_bootstrap`). This also closes a **second, opposite latent defect**: the
  contrast path had *no* lower clamp, and `keep14` accordingly reported
  `letter_acc_diff_boot_p = 0.0000`, which is equally unattainable for a bootstrap
  p; it is now `0.0001 = 1/10000`.
* Applied to **both** call sites. `paired_bootstrap()` — which feeds every
  `{letter,content_norm}_vs_null_boot_p`, i.e. the significance of the
  "arm is below/at its own floor" claims — had the identical double-count and was
  clamped only from below (`max(p, 1/n_boot)`), never from above.

Unit tests (run before shipping): `d ≡ 0 → 1.0`; all-positive / all-negative
`→ 1e-4`; symmetric continuous `→ 0.991` (identical to old, no atom);
shifted-by-3σ `→ 0.0026`; 200 random shifts all land in `[0.048, 0.9996] ⊂ [1e-4, 1]`.

## 4. Re-emission of the six summaries

Per-example records live on the **zwfy6** disk
(`results/a01_gate3/dtype_runs/<arm>/per_example_dtype_shard{0..7}of8.jsonl`),
reachable only from `.73`/`.82`/`.104`. The patched script was pushed to `.73` with
`scp -O` (md5 `f63b1707e2215b1e83ef5ccb51c5cf6c` verified on both ends; the
pre-existing remote copy was confirmed identical to wzc1 git HEAD beforehand, so
the patch was the only delta), then each arm re-merged with

```bash
CUDA_VISIBLE_DEVICES="" /opt/conda/envs/torch-base/bin/python \
  proposal/active/A01-null-calibration-methodology/code/a01_gate3_fp32_vs_bf16.py \
  --merge --output_name <arm> --num_shards 8 --expect_n 14042
```

`--merge` returns before any CUDA call, so this is a 0-GPU path. The previous
summaries were preserved remotely as `dtype_summary.PRE_R7FIX.json`.

**Shard integrity — `read_shards()` hard assertions left in place, none bypassed:**

| arm | shards | n_scored | nan |
|---|---|---|---|
| `7B_base_dtype` | 8/8 | 14042 | 0 |
| `7B_keep8_step121000_dtype` | 8/8 | 14042 | 0 |
| `7B_keep10_step83500_dtype` | 8/8 | 14042 | 0 |
| `7B_keep12_step124000_dtype` | 8/8 | 14042 | 0 |
| `7B_keep14_step200000_dtype` | 8/8 | 14042 | 0 |
| `7B_shortgpt16_step200000_dtype` | 8/8 | 14042 | 0 |

All six complete; no partial merge was performed or needed.

## 5. Old → new, every p-value in all six summaries

`**` marks a changed value. Nothing else in these files changed at all.

| arm | P(bs==0) | contrast `letter_acc_diff_boot_p` | bf16 `letter_vs_null` | fp32 `letter_vs_null` | bf16 `content_norm_vs_null` | fp32 `content_norm_vs_null` |
|---|---|---|---|---|---|---|
| `base` | 5.44% | 1.0420 → **0.9876** ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 |
| `keep8_step121000` | 1.07% | 0.5598 → 0.5491 ** | 0.0192 → 0.0190 ** | 0.0062 → 0.0060 ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 |
| `keep10_step83500` | 0.83% | 0.2260 → 0.2177 ** | 0.4150 → 0.4094 ** | 0.7756 → 0.7671 ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 |
| `keep12_step124000` | 0.20% | 0.0748 → 0.0728 ** | 0.3736 → 0.3692 ** | 0.0272 → 0.0268 ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 |
| `keep14_step200000` | 0.00% | 0.0000 → 0.0001 ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 |
| `shortgpt16_step200000` | 0.12% | 0.0174 → 0.0162 ** | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 | 0.0001 → 0.0001 |

Exhaustive field-level diff over the six JSONs (flattened, key sets asserted
identical): **exactly three field paths moved** —
`contrast.letter_acc_diff_boot_p`, `by_dtype.bf16.letter_vs_null_boot_p`,
`by_dtype.fp32.letter_vs_null_boot_p`. Everything else — accuracies, tie counts,
tie-multiplicity histograms, gaps, prediction distributions, floors, CI95s,
McNemar p-values, residual fractions, meta — is **byte-identical**. The re-merge
therefore doubles as a regression test on the merge path.

The `content_norm_vs_null_boot_p` values are unchanged because their `d`
(accuracy minus a *fractional* longest-option null) has an essentially continuous
bootstrap distribution with no atom at 0, so the buggy and fixed formulas coincide.
All changed p-values move **down**, as expected: removing a double-counted mass
from the smaller tail can only shrink it (the `keep14` 0 → 1e-4 move is the lower
floor, not the atom correction).

## 6. Impact on conclusions — **none**

| | count | changed |
|---|---|---|
| verdicts (6 arms × 2 dtypes × {letter, content_norm}) | 24 | **0** |
| p-values crossing α = 0.05 in either direction | 30 | **0** |
| non-p fields across the six summaries | all | **0** |

Verdicts after the fix (identical to before):

| arm | bf16 letter | fp32 letter | bf16 content_norm | fp32 content_norm |
|---|---|---|---|---|
| `base` | above the floor | above the floor | above the floor | above the floor |
| `keep8_step121000` | **BELOW** the floor (sig.) | **BELOW** the floor (sig.) | above the floor | above the floor |
| `keep10_step83500` | AT the floor | AT the floor | above the floor | above the floor |
| `keep12_step124000` | AT the floor | above the floor | above the floor | above the floor |
| `keep14_step200000` | above the floor | above the floor | above the floor | above the floor |
| `shortgpt16_step200000` | above the floor | above the floor | above the floor | above the floor |

In particular the load-bearing claims are untouched:

* **`keep8` letter reads significantly BELOW its own best-constant floor** under
  both dtypes — p = 0.0192 → **0.0190** (bf16) and 0.0062 → **0.0060** (fp32).
  Both were and remain < 0.05 by a wide margin.
* **`keep10` letter is AT the floor** under both dtypes (0.415 → 0.409,
  0.776 → 0.767) and **`keep12` letter is AT the floor under bf16 but above it
  under fp32** (0.3736 → 0.3692 vs 0.0272 → 0.0268) — the dtype-sensitivity of
  that single verdict is a pre-existing, separately reported finding and is
  *not* created or removed by this fix.
* The base arm's own conclusion never depended on the malformed number: letter
  accuracy is byte-identical across dtypes, McNemar p = 1.000, CI95
  [−0.0011, +0.0011]. The fix simply makes the bootstrap p agree (0.9876) instead
  of contradicting arithmetic (1.042).

**Nothing in A01 or paperG needs to be re-argued as a result of this fix.** What
changes is that the numbers are now legal, the estimator is defined for the
discrete/degenerate case it actually faces, and the same bug is no longer sitting
under the `*_vs_null_boot_p` values that carry the floor verdicts.

## 7. Provenance

* patched estimator + `_log` banner: `.../code/a01_gate3_fp32_vs_bf16.py`
  (`two_sided_boot_p`, `paired_bootstrap`, `analyse`, `main`)
* re-emitted evidence: `.../evidence/gate3_dtype_runs/*_dtype_summary.json` (6)
* pre-fix copies retained on zwfy6 as
  `results/a01_gate3/dtype_runs/<arm>/dtype_summary.PRE_R7FIX.json`
* ledger entries: `.../STATUS.json:audit_response.fixed_2026_08_11`
  (original `open_defect_not_fixed_this_pass` text retained verbatim and marked
  SUPERSEDED, per the retraction-history rule in `proposal/README.md`);
  `paperG/README.md` open defect #1 struck through and closed.
