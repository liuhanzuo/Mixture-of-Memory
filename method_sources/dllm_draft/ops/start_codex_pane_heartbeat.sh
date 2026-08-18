#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="scaffold-codex-pane-heartbeat"
INITIAL_DELAY_SECONDS="${CODEX_PANE_INITIAL_DELAY_SECONDS:-1800}"
INTERVAL_SECONDS="${CODEX_AUTOMATION_INTERVAL_SECONDS:-1800}"

mkdir -p "$ROOT/ops/logs" "$ROOT/ops/state"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Codex pane heartbeat already running: $SESSION"
  exit 0
fi
tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && CODEX_PANE_INITIAL_DELAY_SECONDS='$INITIAL_DELAY_SECONDS' CODEX_AUTOMATION_INTERVAL_SECONDS='$INTERVAL_SECONDS' exec '$ROOT/ops/codex_pane_heartbeat.sh'"
echo "Started single-pending Codex pane heartbeat: $SESSION"
