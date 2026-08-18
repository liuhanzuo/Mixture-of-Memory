#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="scaffold-heartbeat"

mkdir -p "$ROOT/ops/logs" "$ROOT/ops/state" "$ROOT/ops/control" \
  "$ROOT/ops/snapshots"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "heartbeat already running in tmux session: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec '$ROOT/ops/heartbeat_daemon.sh'"
echo "started heartbeat in tmux session: $SESSION"

