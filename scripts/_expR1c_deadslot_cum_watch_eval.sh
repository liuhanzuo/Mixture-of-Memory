#!/usr/bin/env bash
# Watcher: wait for expR1c_deadslot_cum training procs to exit + both ckpts present,
# then run offline BABILong eval scheduler and score both ckpts.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
PYBIN="$PROJECT_ROOT/.venv/bin/python"

CKPT_DIR="outputs/expR1c_deadslot_cum"
CK500="$CKPT_DIR/mem_space_adapter_step000500.pt"
CK1000="$CKPT_DIR/mem_space_adapter.pt"
WATCHLOG="logs/expR1c_deadslot_cum_watch.log"
SCORES="logs/expR1c_deadslot_cum_scores.txt"
mkdir -p logs

log(){ echo "[$(date '+%F %T')] $*" >> "$WATCHLOG"; }

log "watcher started; waiting for training procs to exit + both ckpts present"

# 1. Wait until no compute-apps on GPU (training fully exited) AND both ckpts present.
while true; do
  NPROC=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c . || echo 0)
  HAVE500=0; HAVE1000=0
  [ -f "$CK500" ] && HAVE500=1
  [ -f "$CK1000" ] && HAVE1000=1
  log "poll: gpu_procs=$NPROC have_step500=$HAVE500 have_step1000=$HAVE1000"
  if [ "$NPROC" -eq 0 ] && [ "$HAVE1000" -eq 1 ] && [ "$HAVE500" -eq 1 ]; then
    log "training exited + both ckpts ready -> launching eval"
    break
  fi
  sleep 60
done

# Ensure adapter_config.json exists (training writes it at save time)
if [ ! -f "$CKPT_DIR/adapter_config.json" ]; then
  log "ERROR: adapter_config.json missing in $CKPT_DIR; aborting"
  exit 1
fi

# 2. Run eval scheduler (blocks until SCHED_DONE)
log "running eval scheduler"
bash scripts/_expR1c_deadslot_cum_eval_sched.sh >> "$WATCHLOG" 2>&1
log "eval scheduler returned"

# 3. Score both ckpts
log "scoring"
{
  echo "===== expR1c_deadslot_cum scores ($(date '+%F %T')) ====="
  for run in expR1c_deadslot_cum_step500 expR1c_deadslot_cum_step1000; do
    echo ""
    echo "########## $run ##########"
    $PYBIN scripts/score_nested_babilong.py "babilong_results/$run" 2>&1
  done
} > "$SCORES" 2>&1
log "scoring done -> $SCORES"
touch "logs/expR1c_deadslot_cum_watch_DONE"
log "watcher complete"
