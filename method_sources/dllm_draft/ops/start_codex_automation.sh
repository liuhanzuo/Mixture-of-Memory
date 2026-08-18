#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="scaffold-codex-automation"

mkdir -p "$ROOT/ops/logs" "$ROOT/ops/state"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "Codex automation already running: $SESSION"
  exit 0
fi

tmux new-session -d -s "$SESSION" \
  "cd '$ROOT' && exec '$ROOT/ops/codex_automation_daemon.sh'"
echo "Started 30-minute Codex automation: $SESSION"
