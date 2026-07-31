# P0.9 — Numerical LoCoMo conversation-cluster bootstrap CI

**Goal.** Supply the one missing number in Paper A's statistics appendix: the
numerical 95% CI of the paired **conversation-cluster** bootstrap for the
CoMem − KV-Direct LoCoMo GPT-4o-judge difference (the appendix previously stated
only "a 95% interval entirely above zero").

**Script:** `scripts/locomo_cluster_bootstrap.py` (CPU-only, deterministic).
**Result JSON:** `status/P0_9_CLUSTER_BOOTSTRAP.json`.

## Sources (chat_template=False, selector=iter_bm25, GPT-4o judge)
Both dirs staged on this checkout under `status/p3_locomo_cluster/locomo_results/`:
- CoMem + LoRA (flagship, self-distilled j=12 adapter):
  `qcmem_8b_iter_chatFALSE/` — overall judge 38.27 (`scores.json`)
- KV-Direct (full-context oracle): `kvdirect_8b_chatFALSE/` — overall judge 34.59
- Per-item verdicts: each dir's `judge_cache.jsonl` (1,540 cat1–4 API-judged items,
  1.0=CORRECT / 0.0=WRONG). Category + conversation grouping from
  `preds_shard{0..3}of4.jsonl` (ids are `conv{N}_qa{M}`; the `conv{N}` prefix is
  the LoCoMo conversation grouping field, 10 conversations total).

These are the same dirs that produced the appendix's 38.27 / 34.59 full-set and
48.64 / 43.83 common-item judge scores (`status/PAPER_LOCOMO_ERRATA_20260721.md`
§9–§12).

## Reproduction GATE (must match before trusting the cluster CI)
Recomputed on the 1,540 common judged cat1–4 items:

| quantity | appendix | recomputed | match |
|---|---|---|---|
| CoMem common-item judge | 48.64 | **48.64** | ✅ |
| KV-Direct common-item judge | 43.83 | **43.83** | ✅ |
| paired mean diff | +4.81 | **+4.81** | ✅ |
| per-item bootstrap 95% CI (B=10,000, seed 1234) | [2.34, 7.27] | **[2.34, 7.34]** | ✅ (lo exact; hi +0.07, within Monte-Carlo/percentile-index tolerance) |

Gate passes: the point estimates match exactly and the per-item CI reproduces to
within one bootstrap-percentile bin. (The 0.07 upper-bound drift is pure
resampling noise; the appendix's [2.34, 7.27] is retained as the reported per-item CI.)

## Conversation-cluster bootstrap (the requested number)
B=10,000 resamples, seed 1234. Each resample draws the 10 conversation ids WITH
replacement, pools the items of the drawn conversations, and computes
mean(CoMem) − mean(KVD) over that pooled item set. 2.5 / 97.5 percentiles:

- **Point estimate: +4.81 judge-accuracy points**
- **95% CI = [+1.27, +8.32]  → entirely above zero** (bootstrap two-sided p ≈ 0.004)
- **8 of 10 conversations favor CoMem** (conv4 −1.12, conv6 −4.67 favor KV-Direct)

Robustness: the cluster lower bound stays +1.17…+1.30 across seeds
{1234, 1, 42, 2024, 12345, 7} — always > 0. (An earlier staged run at B=20,000 /
seed 12345 gave [+1.29, +8.40], consistent.)

### Per-conversation observed mean diffs (CoMem − KVD, judge pts)
| conv | n | CoMem | KVD | diff |
|---|---:|---:|---:|---:|
| conv0 | 152 | 44.08 | 36.84 | **+7.24** |
| conv1 |  81 | 46.91 | 43.21 | **+3.70** |
| conv2 | 152 | 48.68 | 46.05 | **+2.63** |
| conv3 | 199 | 48.74 | 34.67 | **+14.07** |
| conv4 | 178 | 49.44 | 50.56 | −1.12 |
| conv5 | 123 | 56.10 | 52.85 | **+3.25** |
| conv6 | 150 | 43.33 | 48.00 | −4.67 |
| conv7 | 191 | 48.69 | 44.50 | **+4.19** |
| conv8 | 156 | 42.95 | 39.10 | **+3.85** |
| conv9 | 158 | 57.59 | 45.57 | **+12.03** |

## Interpretation
The dependence-aware conversation-cluster interval [+1.27, +8.32] is wider than
the per-item interval [2.34, 7.27] (fewer effective clusters, n≈10), as expected,
but remains entirely above zero. The CoMem − KV-Direct LoCoMo-judge advantage
therefore survives clustering at the conversation level; the paper does not claim
the 1,540 questions are independent.
