#!/usr/bin/env bash
# Chain: 4 MBPP+ scaffold baselines -> 4 MBPP+ refinement runs.
# Full 4x2 capacity x refinement grid replication of HE+ on MBPP+ (n=378).
set -u
ROOT=/apdcephfs_wzc1/share_304376610/pighzliu_code/dllm_draft
cd "$ROOT" || exit 1
LOG=logs/mbpp_full_grid_$(date +%Y%m%d_%H%M).log
mkdir -p logs
echo "[$(date '+%F %T')] starting MBPP+ full grid, log=$LOG"
{
  bash scripts/_run_capacity_ladder_mbpp_wzc1.sh
  bash scripts/_run_capacity_refine_mbpp_chain.sh
  echo "[$(date '+%F %T')] MBPP+ full grid COMPLETE"
} 2>&1 | tee -a "$LOG"
