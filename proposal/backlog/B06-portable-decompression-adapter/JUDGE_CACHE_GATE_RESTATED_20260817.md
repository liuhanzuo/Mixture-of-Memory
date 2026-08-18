# B06 — The judge-cache gate, restated (its decision rule was off by ~6x)

**Date**: 2026-08-17. **Cost**: 0 GPU, 0 ssh.
**What this supersedes** (nothing deleted — all three still stand as history):
* `STATUS.json.drift_resolution_leg_20260815.highest_value_next_0_gpu_followup`
* `STATUS.json.drift_resolution_leg_20260815.not_verified[1]`
* `DRIFT_RESOLUTION_VERDICT.md` §5.1 item 1 / §5 item 2

All three say, in substance: *"If `locomo_results/hcache_8b_chatFALSE/judge_cache.jsonl` on zwfy6
holds **~1439** records instead of 1540, silent judge-API failures are the **PROVEN** mechanism
for the canonical 8.11."*

**That decision rule is arithmetically wrong, and in the wrong direction: observing ~1439 would
REFUTE the hypothesis, not prove it.** An agent that ssh'd in, saw ~1439 and wrote "mechanism
proven" would produce a confident wrong causal claim that *looked* like a clean win because the
number matched the prediction written in the file.

---

## 1. The mechanism, from the code (verified, not quoted from the docs)

`scripts/eval_qcmem_locomo.py`, in `llm_judge_preds`:

```python
item, v, raw = fut.result()
if v is None:
    n_fail += 1
    item["judge"] = 0.0          # count API failures as WRONG (rare)
else:
    item["judge"] = v
    rec = {...}
    cache_fh.write(json.dumps(rec, ensure_ascii=False) + "\n")   # <-- only on success
```

and the warning it prints:

```
[QCMem-LoCoMo][WARN] {n_fail}/{len(todo)} judge calls failed (counted as WRONG; not cached — re-run to retry them).
```

So **N** silent failures leave **1540 − N** cache records, and each failed item is scored 0.

## 2. Where the docs' arithmetic breaks

The canonical row is `156/1540 = 10.1299%` (`status/PAPERA_RESULTS_CONSOLIDATED.md:175`).
The B06 local control is `257/1540 = 16.6883%` — **I recomputed this from the raw cache**, not from
the published percentage: `locomo_results/hcache_j12_noLoRA_chatFALSE/judge_cache.jsonl`,
1540 records / 1540 unique ids, 257 with `judge == 1.0`.

`257 − 156 = 101`. **The docs read that 101 as a count of MISSING CACHE RECORDS. It is a count of
MISSING CORRECT ITEMS.** Those are different by a factor of `1/p`.

With base rate `p = 257/1540 = 0.166883`, N failures leave `R = 1540 − N` graded records and
`R·p` expected correct:

| N_fail | cache records R | E[correct] = R·p | E[Judge_1:4] |
|---:|---:|---:|---:|
| 0 | 1540 | 257.00 | 16.6883 |
| **101** | **1439** | **240.14** | **15.5938** |
| 300 | 1240 | 206.94 | 13.4373 |
| **605** | **935** | **156.04** | **10.1322** ← canonical |

So **N=101 lands at 15.59, not 10.13** — the docs' predicted observation is off from the canonical
value by **1.54×** in the metric, and reproducing 156 correct requires **N ≈ 605 failures,
i.e. ≈ 935 cache records, not ≈ 1439**.

Verified with `/opt/conda/envs/torch-base/bin/python`:
`156 / (257/1540) = 934.79` records → `N_fail = 605.21`.

## 3. The gate, restated so its read-out is decidable

**Command** (0 GPU, 1 ssh, ~1 second; whoever has ssh should run it):

```bash
sshpass -f configs/password_h20_853573.txt ssh -o StrictHostKeyChecking=no \
  -o PreferredAuthentications=password root@28.85.35.73 \
  'wc -l /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/locomo_results/hcache_8b_chatFALSE/judge_cache.jsonl; \
   grep -c "\"judge\": 1.0" /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/locomo_results/hcache_8b_chatFALSE/judge_cache.jsonl'
```

Note the path must be the **zwfy6** root (`/apdcephfs_wzc1` on .73 is a symlink to zwfy6, so the
wzc1 string "works" but is misleading — see `memory/cluster-two-disks-not-shared.md`).
**Report BOTH numbers**: total records AND correct records. The original one-liner asked only for
`wc -l`, which is the weaker of the two and is exactly why the rule inverted.

**Pre-registered read-out** (window derived from the two same-method local replicates,
`p ∈ [238/1540, 257/1540] = [15.45%, 16.69%]`):

| observation | verdict |
|---|---|
| **R ∈ [935, 1009]** and correct ≈ 156 | silent judge failures are a **SUFFICIENT** explanation of the canonical 8.11/10.13 |
| **R = 1540** (complete pass), correct = 156 | failures explain **NOTHING**; the canonical gap is a real harness/node/prediction difference and must be reported as such |
| **R ≈ 1439** (the docs' number) | mechanism explains **only ~1/6 of the gap**. Implied complete-pass rate would be `156/1439 = 10.84%`, still far below the local 15.45–16.69%. ⇒ hypothesis **REFUTED as the whole story** |
| **R < 935** | over-explains; something else is also wrong (e.g. a truncated run) |

If `correct` on zwfy6 is **not** 156, then the arithmetic recovery of `(156, 5)` in
`canonical_8_11_conversion_status` is itself wrong and that key must be superseded too.

---

## 4. ⚠️ A cheaper, ALREADY-ON-DISK test that partially answers this WITHOUT ssh

This is the part the earlier legs missed. `canonical_records_located_on_wzc1_after_all` notes that
`locomo_results_openjudge_qwen3_MIRROR/hcache_8b_chatFALSE/` exists on wzc1. What it does not say
is **what that mirror is**: per `paperA/TODOList.md:170`, it is the **canonical run's own
predictions rescored by an open-weight `qwen3-8b-judge`** (confirmed: `judge_meta.json` →
`judge_model: "qwen3-8b-judge"`, `base_url: http://127.0.0.1:8412/v1`). So **a second,
independent judge pass over the canonical predictions is already on this disk.**

Recovered integer counts (unique inversion of the published per-category percentages, verified by
reconstructing `overall_judge = 100·(531+5)/1986 = 26.988922`, exactly matching the file):

| | cat1 (282) | cat2 (321) | cat3 (96) | cat4 (841) | Judge_1:4 |
|---|---:|---:|---:|---:|---:|
| canonical, **gpt-4o** | 19 (6.74%) | 8 (2.49%) | 9 (9.38%) | 120 (14.27%) | **156/1540 = 10.13%** |
| canonical, **qwen3-8b** | 80 (28.37%) | 66 (20.56%) | 30 (31.25%) | 355 (42.21%) | **531/1540 = 34.48%** |

### What this rules IN and OUT

**RULES OUT "the canonical predictions are simply worse."** The judge-independent metrics are
essentially identical across the canonical run and both local same-method runs:

| run | overall_f1 | overall_em | overall_acc | cat4 acc |
|---|---:|---:|---:|---:|
| canonical (mirror) | 4.6700 | 0.2518 | 6.2941 | 10.2259 |
| local `hcache` | 4.7275 | 0.3021 | 6.3948 | 10.2259 |
| local `hcache_j12_noLoRA_chatFALSE` | 4.7741 | 0.3525 | 6.6465 | 10.5826 |

cat4 acc is **identical to 4 decimals** between canonical and local `hcache`. So the *generations*
are equivalent; the divergence is entirely in the **judge layer**. That is consistent with the
failure hypothesis — but it is equally consistent with a plain judge-instrument difference.

**DOES NOT rule in the failure mechanism, and here I must correct my own first analysis.** My
first cut computed the qwen3/gpt-4o **ratio** per arm (HCache 3.404 vs sibling median 1.579,
"z = 14.2σ") and that statistic is **invalid**: a ratio of two rates is mechanically inflated when
the denominator is small, and HCache has by far the smallest gpt-4o rate, so the ratio is
confounded with the very thing being tested. On the **additive** scale, which is not so
confounded:

| arm | qwen3 − gpt-4o (pp) |
|---|---:|
| kvdirect | +20.26 |
| streamingllm | +20.00 |
| infllm | +17.99 |
| qcmem (CoMem) | +17.79 |
| memoryllm | +11.95 |
| **hcache (canonical)** | **+24.35** |

HCache's additive gap is **z = +2.01σ** against the five siblings (mean +17.60, sd 3.35) —
elevated, **not** the 14σ my ratio version suggested, and **not** significant at any
multiplicity-corrected threshold with n=5 comparators. **A ~24pp open-judge leniency gap is within
the spread of an instrument that is systematically more lenient on every arm.** The mirror
therefore **narrows** the space (the preds are fine; it is a judge-layer effect) but **cannot
distinguish** "some gpt-4o calls silently failed" from "gpt-4o is simply harsher on this arm's
answer style". Only the record count on zwfy6 separates those.

---

## 5. Honest scope of this file

* I did **not** read the canonical `judge_cache.jsonl`. It is not on wzc1 — re-verified today:
  `ls locomo_results/hcache_8b_chatFALSE/` → `No such file or directory`; the only hcache judge
  caches on wzc1 are `locomo_results/{hcache, hcache_j12_noLoRA_chatFALSE, hcache_j12_LoRA_chatFALSE}`,
  each exactly 1540 lines. zwfy6 is not mounted on this node and ssh was out of budget.
* The canonical `(156, 5)` counts remain an **arithmetic inversion of published rounded
  percentages**, not a read of raw records. Everything in §2-§3 is conditional on that inversion.
* The `p ∈ [15.45%, 16.69%]` window comes from **two** local replicates. n=2 gives no real
  interval; the window is a plausibility band, not a confidence interval.
* §4's additive-gap test has **n=5** comparators and I ran **no** formal test — "z = 2.01" is a
  descriptive standardisation against 5 points, not a p-value. It should not be reported as
  significance.
* My own first version of §4 used an invalid ratio statistic and reached a 7x-too-strong
  conclusion (14σ). It is corrected above rather than quietly dropped, because the failure mode
  ("a ratio between two rates looks like a huge effect when the denominator is the smallest in the
  cohort") is the kind of thing that will recur.
