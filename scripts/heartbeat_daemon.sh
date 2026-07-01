#!/usr/bin/env bash
# Independent heartbeat daemon loop.
# The container has NO cron/crond daemon, so system crontab entries never fire.
# This loop replaces it: a detached process that calls heartbeat_cron.sh every
# INTERVAL seconds. Each call is a fresh stateless `codebuddy -p` process
# (zero conversation history) — see heartbeat_cron.sh header.
#
# Start (detached, survives this shell/session):
#   nohup setsid bash scripts/heartbeat_daemon.sh > logs/heartbeat_cron/daemon.out 2>&1 &
#
# Stop:
#   touch logs/heartbeat_cron/STOP        # graceful, checked each loop
#   # or: pkill -f heartbeat_daemon.sh
#
# State lives in status/ files; losing in-loop memory is fine.

set -uo pipefail

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
INTERVAL="${HEARTBEAT_INTERVAL:-7200}"   # 2h default
LOG_DIR="${PROJECT_ROOT}/logs/heartbeat_cron"
STOP_FILE="${LOG_DIR}/STOP"
DAEMON_LOG="${LOG_DIR}/daemon.log"

mkdir -p "$LOG_DIR"

echo "$(date '+%Y-%m-%d %H:%M:%S') daemon START pid=$$ interval=${INTERVAL}s" >> "$DAEMON_LOG"

# Run one heartbeat immediately on start, then loop.
while true; do
  if [[ -f "$STOP_FILE" ]]; then
    echo "$(date '+%Y-%m-%d %H:%M:%S') daemon STOP (found STOP file) pid=$$" >> "$DAEMON_LOG"
    rm -f "$STOP_FILE"
    exit 0
  fi
  echo "$(date '+%Y-%m-%d %H:%M:%S') daemon -> invoking heartbeat_cron.sh" >> "$DAEMON_LOG"
  bash "${PROJECT_ROOT}/scripts/heartbeat_cron.sh"
  echo "$(date '+%Y-%m-%d %H:%M:%S') daemon <- heartbeat_cron.sh exit=$?; sleeping ${INTERVAL}s" >> "$DAEMON_LOG"
  sleep "$INTERVAL"
done
