#!/usr/bin/env bash
# Drive HARDOBJ ctx3 seed42 step1000 (final) eval on LOCAL, full 8 GPU 2-group.
# chunked (W0) then SWA-full (W6). diskA ckpt, evaluated locally.
# Judgment: long-range qa5 8k/16k/32k vs BASE_mix0 anchor 5/5/3.
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
CKPT="$ROOT/outputs/HARDOBJ_lastchunk_N128/mem_space_adapter.pt"
CFG="$ROOT/outputs/HARDOBJ_lastchunk_N128/adapter_config.json"
LOG=logs/driver_hardobj_ctx3_step1000_eval.log
echo "[$(date '+%F %T')] step1000 eval driver start" >>"$LOG"
for W in 0 6; do
  RUN_PREFIX="HARDOBJ_ctx3_seed42_step1000_swa${W}" \
  CKPT_FILES="$CKPT" \
  CK_NAMES="HARDOBJ_ctx3_seed42_step1000_swa${W}" \
  ADAPTER_CONFIG="$CFG" \
  MODEL=models/Meta-Llama-3-8B \
  EXTRA_ARGS="--swa_eval_chunks ${W}" \
  PROJECT_ROOT="$ROOT" \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/_eval_taskpool_2group.sh \
    >"logs/eval_HARDOBJ_ctx3_step1000_swa${W}.sched.out" 2>&1
  echo "[$(date '+%F %T')] swa${W} eval finished" >>"$LOG"
done
echo "[$(date '+%F %T')] DRIVER_DONE step1000 dual eval complete" >>"$LOG"
