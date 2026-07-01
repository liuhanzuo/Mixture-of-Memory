#!/usr/bin/env bash
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
DIR="$ROOT/outputs/HARDOBJ_lastchunk_ctx3_N192_diskA"
CFG="$DIR/adapter_config.json"
LOG=logs/driver_ctx3_N192_eval.log
echo "[$(date '+%F %T')] ctx3_N192 eval driver start" >>"$LOG"
for W in 0 6; do
  RUN_PREFIX="HARDOBJ_ctx3_N192_swa${W}" \
  CKPT_FILES="$DIR/mem_space_adapter_step000500.pt $DIR/mem_space_adapter.pt" \
  CK_NAMES="HARDOBJ_ctx3_N192_step500_swa${W} HARDOBJ_ctx3_N192_step1000_swa${W}" \
  ADAPTER_CONFIG="$CFG" MODEL=models/Meta-Llama-3-8B \
  EXTRA_ARGS="--swa_eval_chunks ${W}" PROJECT_ROOT="$ROOT" \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/_eval_taskpool_2group.sh >"logs/eval_ctx3_N192_swa${W}.sched.out" 2>&1
  echo "[$(date '+%F %T')] swa${W} eval finished" >>"$LOG"
done
echo "[$(date '+%F %T')] DRIVER_DONE ctx3_N192 dual eval" >>"$LOG"
