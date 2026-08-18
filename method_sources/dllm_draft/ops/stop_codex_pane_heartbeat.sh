#!/usr/bin/env bash
set -euo pipefail

SESSION="scaffold-codex-pane-heartbeat"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  tmux kill-session -t "$SESSION"
  echo "Stopped Codex pane heartbeat: $SESSION"
else
  echo "Codex pane heartbeat is not running"
fi
