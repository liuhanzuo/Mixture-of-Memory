#!/usr/bin/env bash
set -euo pipefail

SESSION="scaffold-codex-automation"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  pane_pid="$(tmux display-message -p -t "$SESSION:0" '#{pane_pid}')"
  tmux kill-session -t "$SESSION"
  for child in $(pgrep -P "$pane_pid" 2>/dev/null || true); do
    kill -TERM "$child" 2>/dev/null || true
  done
  echo "Stopped Codex automation: $SESSION"
else
  echo "Codex automation is not running"
fi
