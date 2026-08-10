# P0.13 — Same-Pack / Same-LoRA / Same-Examples Quality↔Latency Paired Benchmark

## Status

**DONE (2026-08-02).** FINAL required Paper A model run before submission. Two paired arms differ ONLY in `resume_j` on 1:1-paired packs. Result branch = **quality–latency TRADEOFF** (Arm B is significantly less accurate, not non-inferior). All headline numbers below were recomputed/verified by MAIN against the JSON artifacts, not merely copied from the run agent.

- Node: .82 (28.82.250.82:36000), 8×H20, diskB, torch-base env (torch 2.13.0 / cuda 13.2 / transformers 5.5.4 / peft 0.19.1 — same env as P0.12)
- Run agent: a0853dad. Local commit `29243d9` (`scripts/bench_p0_13_quality_latency.py` +870, `scripts/launch_p0_13_82.sh` +149), author LiuHanzuo, **NOT pushed**. Runtime manifest git_commit = `21c124e` (diskB checkout at run time).
- Artifacts (on .82 diskB): `bench_results/p0_13_quality_latency/` → `manifest.json`, `summary.json`, `stats.json`, `latency.json`, `quality/` (120 files: per-example JSONL + per-cell json), `latency/` (3 proc json).

## Two arms

- **Arm A** — `resume_j=0` + flagship rank-32 LoRA, full 36-layer replay.
- **Arm B** — `resume_j=12` + the SAME LoRA, replay of upper 24 layers from cached residual state `h12`.

Held fixed across both arms: Qwen3-8B checkpoint, LoRA artifact (SHA `dd09cd17…`, `lora_sha_match=true`, 168 modules, layers-to-transform [12..35]), example IDs / gold / decode params, per-example retrieved chunk IDs + order + pack token IDs (`packs_paired_1to1=True`; iter_bm25 is forward-free → packs are resume_j-independent), selector `iter_bm25 topk=12 iter_hop_topk=4 sink=bos chunk_size=512`, chat=False, enable_thinking=False, bf16, SDPA. `abort_reasons=[]`. backbone key-tensor sha `7a478390…` == P0.12.

## Quality result (15 cells, n=100/cell, n_paired=1500, OOM=0, non-finite=0)

- Arm A macro = **99.19**
- Arm B macro = **96.07**
- macro diff (A−B) = **+3.12 pp**, paired bootstrap 95% CI **[2.36, 3.93]** (10k resamples)
- McNemar exact two-sided **p = 8.79e-24** (A-only-correct b=83, B-only-correct c=1, both=1404, neither=12)
- Agreement: prediction-exact 2.8%, first-token 90.8%, first-token cosine 0.977, decode top-1 42.1%

| Task | 8k | 16k | 32k | 64k | 128k |
|---|---|---|---|---|---|
| niah_single_3 (A/B, diff) | 100.0/100.0 (+0.0) | 97.0/91.0 (+6.0) | 97.0/97.0 (+0.0) | 99.0/98.0 (+1.0) | 98.0/98.0 (+0.0) |
| niah_multikey_1 (A/B, diff) | 100.0/94.0 (+6.0) | 100.0/91.0 (+9.0) | 100.0/99.0 (+1.0) | 97.0/90.0 (+7.0) | 100.0/93.0 (+7.0) |
| variable_tracking (A/B, diff) | 99.8/96.2 (+3.6) | 100.0/98.0 (+2.0) | 100.0/98.2 (+1.8) | 100.0/98.6 (+1.4) | 100.0/99.0 (+1.0) |

The gap concentrates on **niah_multikey_1** (4-distractor-key retrieval, +1..+9 across all lengths) and to a lesser degree VT; niah_single_3 arms are essentially equal. The accuracy Arm B loses is retrieval-disambiguation under distractors — caching residual `h12` and skipping the lower-12-layer replay degrades it.

## Latency result (3 independent procs × 3 warmup + 20 timed reads; fixed 16k niah_single_3 pack)

| Arm | read median | p10 | p90 |
|---|---:|---:|---:|
| A (`resume_j=0`) | 931.9 ms | 931.6 | 942.0 |
| B (`resume_j=12`) | 664.4 ms | 663.8 | 667.1 |

- **read speedup A/B = 1.4027×** (per-proc 1.403 / 1.402 / 1.404 — direction-consistent, B always faster; matches P0.12's ~1.373×).
- Total-decode medians ~equal (~2.76–2.86 s both). Read is a small share of end-to-end at 16k, so the speedup is a **read-phase (prefill / QC-read) effect, not total wall-clock**.

## Decision-rule branch → QUALITY-LATENCY TRADEOFF

The macro diff is +3.12 pp with CI entirely above zero and far above the pre-registered non-inferiority bound of −1.0 pt. So Arm B is **NOT non-inferior**; it is significantly less accurate (McNemar p≈9e-24). Arm A (full replay) is the accuracy-optimal configuration; Arm B (`resume_j=12`) buys a consistent **1.40× read-phase speedup** at a **~3 pp** macro accuracy cost concentrated in distractor-heavy multi-key retrieval.

### Allowed paper claim

> On 15 paired RULER cells (same packs, same LoRA, same examples), starting replay at layer 12 rather than layer 0 gives a consistent 1.40× read-phase speedup at a 3.12-pp macro-accuracy cost (95% CI [2.36, 3.93]; McNemar p≈9e-24), concentrated in distractor-heavy multi-key retrieval. This is a quality–latency trade-off, not a quality-preserving reduction.

Forbidden: "quality-preserving", "pure-depth", or end-to-end wall-clock speedup phrasing.

## `.tex` integration — DONE 2026-08-02

Integrated the paired quality–latency result into `paperA`: the table and abstract/introduction/experiments/conclusion/limitations/reproducibility appendix now report a 1.403× read-phase speedup at a 3.12-pp RULER cost, explicitly not quality-preserving or end-to-end acceleration.
