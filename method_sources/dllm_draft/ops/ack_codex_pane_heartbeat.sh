#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PENDING_FILE="$ROOT/ops/state/codex_pane_heartbeat_pending.json"
LOG_FILE="$ROOT/ops/logs/codex_pane_heartbeat.log"

if [[ -e "$PENDING_FILE" ]]; then
  sent_at="$(
    python3 - "$PENDING_FILE" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1], encoding="utf-8")).get("sent_at", "unknown"))
PY
  )"
  unlink "$PENDING_FILE"
  printf '%s ACK sent_at=%s\n' \
    "$(date --iso-8601=seconds)" "$sent_at" >>"$LOG_FILE"
  echo "Acknowledged pending Codex pane heartbeat from $sent_at"
else
  echo "No pending Codex pane heartbeat"
fi
