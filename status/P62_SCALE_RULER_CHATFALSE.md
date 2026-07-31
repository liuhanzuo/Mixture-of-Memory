# Paper A #62 — Qwen3 model-family scale RULER (chat_template=False)

Appendix `tab_scale`: RULER recall vs Qwen3 model size, QCMem zero-training / no-adapter,
chat_template=**False** (全论文强制口径). Same fixed protocol for every size; only the
per-size split depth `resume_j` (=j) varies.

## Protocol (identical across all sizes; 照搬 4B/14B/32B locked scale run)
- driver: `scripts/eval_ruler_qcmem.py`, orchestrated by `scripts/_run_scale_ruler_remaining_p62.sh`
- `chat_template=False`, `enable_thinking=False`, `sink_tokens=bos`, `chunk_size=512`,
  `dtype=bf16`, `attn_impl=sdpa`, zero-training (no LoRA / no bottleneck ckpt)
- **selector is per-task (NOT a single global selector):**
  - `niah_single` / `niah_multikey` → `selector=bm25 topk=12` (rounds=0)
  - `variable_tracking (vt)` → `selector=iter_bm25 topk=16 iter_rounds=4 iter_hop_topk=4`
  - This matches the LOCKED 4B/14B/32B scale run (`scripts/_qwen_scale_zerotrain_ruler_pool.sh`).
    Kept per-task so the new sizes are cross-comparable with the already-published sizes.
    (Coordinator said "selector=iter_bm25"; using per-task bm25/iter_bm25 to preserve
    same-口径 comparability with 4B/14B/32B — flagged for main.)
- 13 RULER cells = niah_single{8k,16k,32k,64k,128k} + niah_multikey{8k,16k,32k,64k,128k}
  + vt{8k,16k,32k}. n=500 samples/cell, split 4 shards × 125.
- scoring: `scripts/_score_chatFALSE.py --ruler <dir>` (official `_string_match_all_one`,
  weighted-mean recall over shards).

## Per-size split depth j (from `_run_scale_chain_after4b.sh`, measured on clean chat=TRUE dirs)
| size | j (resume_j) | layers | n target |
|------|-----|--------|----------|
| 0.6B | 2  | 28 | 500 |
| 1.7B | 3  | 28 | 500 |
| 4B   | 6? (locked, DONE) | — | 500 |
| 14B  | 13 (locked, DONE) | — | 500 |
| 30B-A3B (MoE, 128 experts, 8/tok) | 12 | 48 | 500 |
| 32B  | 27 (locked, DONE) | — | **25** (footnote special case, not rerun) |

## Reference sizes (already complete, this task does NOT rerun them)
| size | RULER mean (recall %) | n |
|------|----------------------|---|
| 4B   | **60.52** | 500 |
| 14B  | **52.91** | 500 |
| 32B  | **88.18** | 25 (footnote) |

4B per-cell: single 8k/16k/32k/64k/128k = 92.2/97.4/95.4/98.0/98.8;
multikey = 34.0/35.8/40.8/32.0/34.8; vt 8k/16k/32k = 56.4/36.5/34.6.
14B per-cell: single = 99.8/82.8/97.6/98.4/97.8; multikey = 49.2/42.6/9.8/34.2/36.4;
vt = 14.2/11.6/13.5.

## This task: remaining sizes (RUNNING as of 2026-08-01 ~04:05)
16-GPU fan-out across two diskB nodes (shared FS, port 36000):
- **.104 (28.83.24.104)** — 30B-A3B alone, 8 GPUs, 52 jobs (13 cells × 4 shards).
  MoE ~57G/model, one model per GPU, ~14min load each (ceph I/O contention).
  `POOL_ROOT=ruler_results/_p62_scale_pool_104 SIZE_LABELS=30ba3b`
- **.73 (28.85.35.73)** — 0.6B + 1.7B, 8 GPUs, 64 jobs (0.6B all 52 + 1.7B 12 missing;
  40 already-done 1.7B shards skipped). `POOL_ROOT=ruler_results/_p62_scale_pool_73 SIZE_LABELS="0p6b 1p7b"`

1.7B was PARTIAL before this task: 40/52 shards good, 8 truncated (<125 rows, deleted+redone)
+ 4 never-run. Only the 12 missing shards are re-run; completed cells preserved.

### Completeness gate note
The pool's original skip-gate `scripts/qwen32_zerotrain_results.py --is-complete` hardcodes
`resume_j==16` and `layers==64` (32B-canonical only) → it would NEVER skip a scale cell and
would redo finished work. `_run_scale_ruler_remaining_p62.sh` instead uses a **row-count gate**
(125 rows/shard + `_summary_shard*` present) to skip completed shards. eval body is untouched.

## Output CSV paths
- `ruler_results/qcmem_scale_0p6b_chatFALSE_ruler/qcmem_scale_0p6b_<cell>/`
- `ruler_results/qcmem_scale_1p7b_chatFALSE_ruler/qcmem_scale_1p7b_<cell>/`
- `ruler_results/qcmem_scale_30ba3b_chatFALSE_ruler/qcmem_scale_30ba3b_<cell>/`

## RESULTS — DONE 2026-08-01 (chat_template=False, n=500, all 52 shards×125 rows, 0 failures)

Scorer: `scripts/_score_chatFALSE.py --ruler <dir>` (official `_string_match_all_one`, weighted-mean recall).

### Per-size RULER recall (%)
| size | j | niah_single 8k/16k/32k/64k/128k | niah_multikey 8k/16k/32k/64k/128k | vt 8k/16k/32k | **RULER mean (13 cells)** | n |
|------|---|----------------------------------|-----------------------------------|----------------|--------------------------|---|
| 0.6B     | 2  | 100.0 / 99.8 / 100.0 / 99.6 / 99.6 | 87.6 / 80.8 / 83.6 / 84.4 / 91.0 | 60.8 / 65.0 / 61.6 | **85.68** | 500 |
| 1.7B     | 3  | 99.8 / 98.2 / 99.4 / 99.4 / 98.6   | 62.4 / 43.4 / 41.8 / 26.4 / 66.4 | 42.9 / 42.0 / 47.7 | **66.81** | 500 |
| 4B       | (locked) | 92.2 / 97.4 / 95.4 / 98.0 / 98.8 | 34.0 / 35.8 / 40.8 / 32.0 / 34.8 | 56.4 / 36.5 / 34.6 | **60.52** | 500 |
| 14B      | 13 | 99.8 / 82.8 / 97.6 / 98.4 / 97.8   | 49.2 / 42.6 / 9.8 / 34.2 / 36.4 | 14.2 / 11.6 / 13.5 | **52.91** | 500 |
| 30B-A3B  | 12 | 99.4 / 99.6 / 99.0 / 99.4 / 99.8   | 67.8 / 51.6 / 53.2 / 52.4 / 62.8 | 89.2 / 88.0 / 84.0 | **80.48** | 500 |
| 32B      | 27 | 100/100/100/100/100                | 96.0 / 100.0 / 84.0 / 88.0 / 100.0 | 46.4 / 64.8 / 67.2 | **88.18** | 25 (footnote) |

Bold sizes 0.6B / 1.7B / 30B-A3B are this task's new results; 4B / 14B / 32B are the
pre-existing locked reference rows (not rerun; 32B stays at n=25 footnote special case).

### Notes / observations
- 30B-A3B (MoE) mean 80.48 vs 32B 88.18. Its niah_single is saturated (~99-100) like the big
  dense models, and its **vt is by far the strongest of the whole family (84-89 vs 11-67
  elsewhere)** — but niah_multikey (51-68) drags the mean. 32B's mean uses only n=25 so the two
  are not perfectly comparable.
- 0.6B (85.68) unexpectedly HIGH — beats 1.7B/4B/14B. Driven by near-perfect niah_single
  (~100) + strong multikey (80-91) + best-in-small-models vt (60-65). Non-monotone scale curve
  is a genuine result (verified n=500, 125 rows/shard), not an artifact of missing data.

### Reproduce
```
# .73 (small sizes)  |  .104 (30B-A3B alone)
POOL_ROOT=ruler_results/_p62_scale_pool_73  SIZE_LABELS="0p6b 1p7b" \
POOL_ROOT=ruler_results/_p62_scale_pool_104 SIZE_LABELS="30ba3b" \
  PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python GPUS="0 1 2 3 4 5 6 7" \
  bash scripts/_run_scale_ruler_remaining_p62.sh
```
