#!/usr/bin/env bash
# Wait for T2_recall_chunk512 step1000 final ckpt, then eval step500+step1000
# on BABILong, CHUNK_SIZE=512 (must match training granularity), W0 + W6.
# Readout judge: does T2 recall pressure push BABILong W0 long-range past
# the harder-objective plateau ~11-15?
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
DIR="$ROOT/outputs/T2_recall_chunk512_N128"
FINAL="$DIR/mem_space_adapter.pt"
CFG="$DIR/adapter_config.json"
LOG=logs/waiter_T2_chunk512_eval.log
echo "[$(date '+%F %T')] waiter start; polling $FINAL" >>"$LOG"
while true; do
  if [[ -f "$FINAL" ]]; then
    sz=$(stat -c%s "$FINAL" 2>/dev/null || echo 0)
    if [[ "$sz" -gt 1000000000 ]]; then sleep 20; sz2=$(stat -c%s "$FINAL" 2>/dev/null || echo 0); [[ "$sz" == "$sz2" ]] && break; fi
  fi
  sleep 30
done
# wait for LOCAL GPUs to free (training process exits)
while pgrep -f "T2_recall_chunk512_N128" | grep -qv $$ 2>/dev/null; do
  if ! ps aux | grep "train_mem_space.*T2_recall_chunk512" | grep -qv grep; then break; fi
  sleep 30
done
sleep 20
echo "[$(date '+%F %T')] ckpt ready, GPUs freed; launching dual eval CHUNK_SIZE=512" >>"$LOG"
for W in 0 6; do
  RUN_PREFIX="T2_chunk512_swa${W}" \
  CKPT_FILES="$DIR/mem_space_adapter_step000500.pt $FINAL" \
  CK_NAMES="T2_chunk512_step500_swa${W} T2_chunk512_step1000_swa${W}" \
  ADAPTER_CONFIG="$CFG" MODEL=models/Meta-Llama-3-8B \
  CHUNK_SIZE=512 EXTRA_ARGS="--swa_eval_chunks ${W}" PROJECT_ROOT="$ROOT" \
  PYTHON_BIN=/opt/conda/envs/torch-base/bin/python \
  bash scripts/_eval_taskpool_2group.sh >"logs/eval_T2_chunk512_swa${W}.sched.out" 2>&1
  echo "[$(date '+%F %T')] swa${W} eval finished" >>"$LOG"
done
echo "[$(date '+%F %T')] DRIVER_DONE T2 chunk512 dual eval" >>"$LOG"
