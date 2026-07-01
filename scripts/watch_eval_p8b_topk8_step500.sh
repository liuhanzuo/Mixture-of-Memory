#!/usr/bin/env bash
# Watchdog: wait for the P8b chunk512 top_k8 step500 ckpt to land on disk-B,
# then auto-dispatch the offline BABILong eval (same protocol as the topk16
# baseline) on the least-used GPU, then score.
#
# Designed to run on .76 under nohup so it survives the launcher's session.
# Writes a heartbeat + status to logs/watch_eval_p8b_topk8_step500.status
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"

CKPT="outputs/mem_space_p8b_chunk512_topk8_diskB/mem_space_adapter_step000500.pt"
STATUS="logs/watch_eval_p8b_topk8_step500.status"
EVAL_SH="scripts/eval_p8b_chunk512_topk8_step500.sh"
SCORE_PY="scripts/score_p8b_chunk512_topk8_step500.py"
PYBIN="/opt/conda/envs/torch-base/bin/python"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$STATUS"; }

log "watchdog started; waiting for $CKPT"

# 1) wait for ckpt (poll every 60s, also require file size stable to avoid partial)
while true; do
  if [ -f "$CKPT" ]; then
    s1=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    sleep 15
    s2=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    if [ "$s1" = "$s2" ] && [ "$s1" -gt 1000000 ]; then
      log "ckpt landed and stable (size=$s1). proceeding to eval."
      break
    fi
  fi
  sleep 60
done

# also need adapter_config.json
if [ ! -f "outputs/mem_space_p8b_chunk512_topk8_diskB/adapter_config.json" ]; then
  log "WARNING: adapter_config.json missing; eval may fail."
fi

# 2) pick least-used GPU
pick_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits 2>/dev/null \
    | sort -t, -k2 -n | head -1 | cut -d, -f1 | tr -d ' '
}
G=$(pick_gpu)
log "selected GPU $G for eval (least memory used)."

# 3) run eval sequentially on that GPU
log "launching eval ($EVAL_SH) on GPU $G ..."
GPUS="$G" PYTHON_BIN="$PYBIN" bash "$EVAL_SH" >>"$STATUS" 2>&1
EVAL_RC=$?
log "eval finished rc=$EVAL_RC"

# 4) score
log "scoring ..."
$PYBIN "$SCORE_PY" >>logs/score_p8b_chunk512_topk8_step500.out 2>&1
log "score done. results in logs/score_p8b_chunk512_topk8_step500.out"
log "WATCHDOG_COMPLETE"
