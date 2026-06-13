#!/usr/bin/env bash
# Auto-waiter: fire HARDOBJ ctx3 seed42 step500 task-pool eval on .196.
# A task only needs ONE group of 4 GPUs, so we run a SINGLE group on GPU4-7
# (which are free now) WITHOUT waiting for the c256 straggler on GPU0-1.
# Only gate on the step500 ckpt being fully written.
# Eval = chunked (W0) + SWA-full (W6) dual, qa1/qa2/qa5 × 0k-32k.
# Judgment: long-range qa5 8k/16k/32k vs BASE_mix0 anchor 5/5/3.
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
CKPT="$ROOT/outputs/HARDOBJ_lastchunk_N128/mem_space_adapter_step000500.pt"
CFG="$ROOT/outputs/HARDOBJ_lastchunk_N128/adapter_config.json"
LOG=logs/waiter_hardobj_ctx3_step500.log
echo "[$(date '+%F %T')] waiter start (single-group GPU4-7); polling for $CKPT" >>"$LOG"

# wait for step500 ckpt + config, fully written (size stable, >1GB)
while true; do
  if [[ -f "$CKPT" && -f "$CFG" ]]; then
    sz=$(stat -c%s "$CKPT" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000000000 ]]; then
      sleep 20; sz2=$(stat -c%s "$CKPT" 2>/dev/null || echo 0)
      [[ "$sz" == "$sz2" ]] && break
    fi
  fi
  sleep 30
done
echo "[$(date '+%F %T')] ckpt ready ($(stat -c%s "$CKPT")B); launching single-group eval on GPU4-7" >>"$LOG"

# launch chunked (W0) + SWA-full (W6) sequentially, single group on GPU4-7
for W in 0 6; do
  RUN_PREFIX="HARDOBJ_ctx3_seed42_step500_swa${W}" \
  CKPT_FILES="$CKPT" \
  CK_NAMES="HARDOBJ_ctx3_seed42_step500_swa${W}" \
  ADAPTER_CONFIG="$CFG" \
  MODEL=models/Meta-Llama-3-8B \
  EXTRA_ARGS="--swa_eval_chunks ${W}" \
  NUM_GROUPS=1 GROUP0_GPUS="4 5 6 7" \
  PROJECT_ROOT="$ROOT" \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/_eval_taskpool_2group.sh \
    >"logs/eval_HARDOBJ_ctx3_step500_swa${W}.sched.out" 2>&1
  echo "[$(date '+%F %T')] swa${W} eval finished" >>"$LOG"
done
echo "[$(date '+%F %T')] WAITER_DONE all dual eval complete" >>"$LOG"
