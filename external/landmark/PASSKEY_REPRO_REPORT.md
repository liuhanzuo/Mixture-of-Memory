# Landmark Attention — Phase 1 Faithful Reproduction (Passkey)

**Verdict: GO.** Landmark's published 32k wall-break is reproduced.

## Setup
- Base: `huggyllama/llama-7b` (LLaMA-1-7B). Checksum confirms LLaMA-1 (param-sum 49798.7656).
- Tuned weights: `epfml/landmark-attention-llama7b-wdiff` recovered via `recover_weights.sh` (base + weight-diff).
- Task: passkey retrieval, 50 tests per length point (long points sharded 3-way across GPUs and pooled).
- `base` = vanilla LLaMA-1-7B (2048-token training window). `mem` = Landmark landmark-attention model.

## Accuracy vs. context length

| n_garbage | ~tokens | base | mem (Landmark) |
|-----------|---------|------|----------------|
| 0         | 71      | 100% (50/50) | 100% (50/50) |
| 4000      | 1139    | 100% (50/50) | 100% (50/50) |
| 8000      | ~2205   | 98% (49/50)  | 94% (47/50)  |
| 15000     | 4073    | **0% (0/50)** | 100% (50/50) |
| 30000     | 8072    | **0% (0/50)** | 96% (48/50)  |
| 60000     | 16072   | — (base already collapsed) | 96% (48/50) |
| 115000    | 30739   | —            | 96% (48/50)  |

## Findings
1. **Wall reproduced**: vanilla base cliffs 98%→0% just past its 2048-token window (between ~2.2k and ~4k tokens).
2. **Wall broken**: Landmark mem holds 94–100% out to ~30.7k tokens (the 32k regime) with no degradation trend — matches the published positive result.
3. Sharded long points pool cleanly (e.g. 115k: 17/17 + 16/17 + 15/16 = 48/50 = 96%).

## Notes
- `run_sweep_parallel.sh` reported `fail=1`: the redundant unsharded 60k/115k single-GPU jobs OOM'd on the ~30k-token forward. Those points are fully covered by the sharded `run_reshard_long.sh` runs. No real blocker.

## Data
- Pooled table: `results/passkey_full.csv`
- Per-point logs/CSVs: `results/mem_n*.{csv,log}`, `results/mem_n{60000,115000}_shard*of3.{csv,log}`, `results/base_all.csv`
