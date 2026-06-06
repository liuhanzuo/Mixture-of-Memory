#!/usr/bin/env bash
# Lightweight watchdog for the H800 progressive-chunk chain.
# Runs ON node0 (detached via setsid/nohup), polls the active stage log + GPU util
# every POLL_SEC, and writes a compact status line to logs/monitor_progressive_h800.log.
# Detects two failure modes:
#   1) NCCL watchdog hang  -> "watchdog got stuck" / "ProcessGroupNCCL" in stage log
#   2) step stall          -> stage log mtime not advancing for STALL_SEC while procs alive
# It does NOT kill anything (avoid racing a recovering run); it just records a loud
# ALERT line so a human / heartbeat can act. Self-exits when the chain finishes
# (driver log prints "chain DONE") or when no train procs remain for 3 consecutive polls.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_jn2/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
POLL_SEC="${POLL_SEC:-120}"
STALL_SEC="${STALL_SEC:-900}"   # 15 min no log growth = stall alert
MON_LOG="logs/monitor_progressive_h800.log"
DRIVER_LOG="logs/progressive_launch_node0_driver.log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$MON_LOG"; }

log "monitor start (poll=${POLL_SEC}s stall=${STALL_SEC}s)"
empty_polls=0
while true; do
  # active stage = newest progressive_chunk_h800_stage*_node0.log
  STAGE_LOG=$(ls -t logs/progressive_chunk_h800_stage*_node0.log 2>/dev/null | head -1)
  NPROC=$(pgrep -f train_mem_space_dolmino_cpt | wc -l)
  GPU_UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d ' %' | paste -sd, -)

  if [[ -z "$STAGE_LOG" ]]; then
    log "no stage log yet; nproc=$NPROC util=[$GPU_UTIL]"
  else
    LAST_STEP=$(grep -oE 'step [0-9]+/[0-9]+' "$STAGE_LOG" 2>/dev/null | tail -1)
    LAST_LM=$(grep -oE 'lm=[0-9.]+' "$STAGE_LOG" 2>/dev/null | tail -1)
    AGE=$(( $(date +%s) - $(stat -c %Y "$STAGE_LOG" 2>/dev/null || echo 0) ))
    STAGE_NAME=$(basename "$STAGE_LOG" | sed 's/progressive_chunk_h800_//; s/_node0.log//')
    log "stage=$STAGE_NAME ${LAST_STEP:-step?} ${LAST_LM:-lm?} log_age=${AGE}s nproc=$NPROC util=[$GPU_UTIL]"

    # NCCL hang detection
    if tail -40 "$STAGE_LOG" 2>/dev/null | grep -qE 'watchdog got stuck|ProcessGroupNCCL.*preparing to dump|ncclSystemError|ibv_reg_mr'; then
      log "ALERT: NCCL hang/error signature in $STAGE_LOG tail -> run likely dead, needs kill+relaunch"
    fi
    # Stall detection (procs alive but log frozen)
    if [[ "$NPROC" -gt 0 && "$AGE" -gt "$STALL_SEC" ]]; then
      log "ALERT: stage log frozen ${AGE}s (> ${STALL_SEC}s) but $NPROC procs alive -> stall/hang"
    fi
  fi

  # Chain-complete check
  if grep -q 'chain DONE' "$DRIVER_LOG" 2>/dev/null; then
    log "chain DONE detected in driver log -> monitor exiting"
    break
  fi
  # No-procs check (with debounce)
  if [[ "$NPROC" -eq 0 ]]; then
    empty_polls=$((empty_polls+1))
    log "no train procs (empty_polls=$empty_polls/3)"
    [[ "$empty_polls" -ge 3 ]] && { log "no procs x3 -> monitor exiting (run ended or crashed)"; break; }
  else
    empty_polls=0
  fi
  sleep "$POLL_SEC"
done
log "monitor stop"
