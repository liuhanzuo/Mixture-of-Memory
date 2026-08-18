#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THREAD_ID="${CODEX_AUTOMATION_THREAD_ID:-019f87a9-eaa0-7e30-b3d4-2b74ace83e79}"
INTERVAL_SECONDS="${CODEX_AUTOMATION_INTERVAL_SECONDS:-1800}"
INITIAL_DELAY_SECONDS="${CODEX_PANE_INITIAL_DELAY_SECONDS:-1800}"
STATE_DIR="$ROOT/ops/state"
LOG_DIR="$ROOT/ops/logs"
NEXT_RUN_FILE="$STATE_DIR/codex_pane_next_run_at"
PENDING_FILE="$STATE_DIR/codex_pane_heartbeat_pending.json"
LOG_FILE="$LOG_DIR/codex_pane_heartbeat.log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

find_pane() {
  local tty pane
  tty="$(
    ps -eo tty,args \
      | grep -F "$THREAD_ID" \
      | grep -F 'codex-linux' \
      | awk 'NR==1 {print "/dev/" $1}'
  )"
  [[ -n "$tty" ]] || return 1
  pane="$(
    tmux list-panes -a -F '#{pane_id} #{pane_tty}' \
      | awk -v tty="$tty" '$2 == tty {print $1; exit}'
  )"
  [[ -n "$pane" ]] || return 1
  printf '%s' "$pane"
}

write_pending() {
  local timestamp="$1" pane="$2"
  python3 - "$PENDING_FILE" "$timestamp" "$pane" "$THREAD_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(
        {
            "sent_at": sys.argv[2],
            "pane": sys.argv[3],
            "thread_id": sys.argv[4],
            "status": "awaiting_ack",
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY
}

delay_seconds="$INITIAL_DELAY_SECONDS"
while true; do
  next_epoch="$(( $(date +%s) + delay_seconds ))"
  date --date="@$next_epoch" --iso-8601=seconds >"$NEXT_RUN_FILE"
  sleep "$delay_seconds"
  delay_seconds="$INTERVAL_SECONDS"

  if [[ -e "$PENDING_FILE" ]]; then
    printf '%s SKIP awaiting_ack\n' \
      "$(date --iso-8601=seconds)" >>"$LOG_FILE"
    continue
  fi
  if ! pane="$(find_pane)"; then
    printf '%s SKIP current_thread_pane_unavailable\n' \
      "$(date --iso-8601=seconds)" >>"$LOG_FILE"
    continue
  fi

  timestamp="$(date --iso-8601=seconds)"
  message="$(
    printf '%s' \
      '【30分钟自动巡检】请先执行 ops/ack_codex_pane_heartbeat.sh，' \
      '然后巡检 .104 当前注册任务和 GPU，按 ' \
      'ELASTIC_SCAFFOLD_EXPERIMENT_TODO.md 推进第一项可执行工作。' \
      '只处理已注册任务，严禁终止未知进程。完成后在本对话中汇报实际进展。'
  )"
  pane_before="$(tmux capture-pane -p -t "$pane" -S -40)"
  busy=false
  if grep -Eq 'Working|esc to interrupt|tab to queue message' \
    <<<"$pane_before"; then
    busy=true
  fi

  write_pending "$timestamp" "$pane"
  if ! tmux set-buffer -- "$message" \
    || ! tmux paste-buffer -t "$pane"; then
    unlink "$PENDING_FILE" 2>/dev/null || true
    printf '%s ERROR paste_failed pane=%s\n' \
      "$timestamp" "$pane" >>"$LOG_FILE"
    continue
  fi
  if [[ "$busy" == true ]]; then
    tmux send-keys -t "$pane" Tab
    sleep 0.2
  fi
  tmux send-keys -t "$pane" Enter
  printf '%s SENT pane=%s thread=%s busy=%s\n' \
    "$timestamp" "$pane" "$THREAD_ID" "$busy" >>"$LOG_FILE"
done
