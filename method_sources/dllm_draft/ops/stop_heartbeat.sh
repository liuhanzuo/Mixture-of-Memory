#!/usr/bin/env bash
set -euo pipefail

SESSION="scaffold-heartbeat"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "stopped heartbeat session: $SESSION"
else
  echo "heartbeat session is not running"
fi

