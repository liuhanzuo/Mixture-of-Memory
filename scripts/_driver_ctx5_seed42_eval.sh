#!/usr/bin/env bash
# Eval HARDOBJ ctx5 seed42 (curriculum 0:5, depth midpoint between ctx3 and ctx7)
# step500+step1000 on .196, full 8 GPU 2-group. chunked W0 then SWA-full W6. diskA, conda python.
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
DIR="$ROOT/outputs/HARDOBJ_lastchunk_ctx5_seed42_diskA"
CFG="$DIR/adapter_config.json"
LOG=logs/driver_ctx5_seed42_eval.log
echo "[$(date '+%F %T')] ctx5 seed42 eval driver start" >>"$LOG"
for W in 0 6; do
  RUN_PREFIX="HARDOBJ_ctx5_seed42_swa${W}" \
  CKPT_FILES="$DIR/mem_space_adapter_step000500.pt $DIR/mem_space_adapter.pt" \
  CK_NAMES="HARDOBJ_ctx5_seed42_step500_swa${W} HARDOBJ_ctx5_seed42_step1000_swa${W}" \
  ADAPTER_CONFIG="$CFG" \
  MODEL=models/Meta-Llama-3-8B \
  EXTRA_ARGS="--swa_eval_chunks ${W}" \
  PROJECT_ROOT="$ROOT" \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/_eval_taskpool_2group.sh \
    >"logs/eval_ctx5_seed42_swa${W}.sched.out" 2>&1
  echo "[$(date '+%F %T')] swa${W} eval finished" >>"$LOG"
done
echo "[$(date '+%F %T')] DRIVER_DONE ctx5 seed42 dual eval complete" >>"$LOG"
