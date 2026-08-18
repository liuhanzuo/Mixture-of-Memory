#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STATE_DIR="$ROOT/ops/state"
LOG_DIR="$ROOT/ops/logs"
PROMPT_FILE="$ROOT/ops/codex_automation_prompt.md"
LOCK_FILE="$STATE_DIR/codex_automation.lock"
HEARTBEAT_FILE="$STATE_DIR/codex_automation_last_run.json"
THREAD_ID="${CODEX_AUTOMATION_THREAD_ID:-019f87a9-eaa0-7e30-b3d4-2b74ace83e79}"
CODEX_HOME_DIR="$(
  printf '%s' "${CODEX_HOME:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory/.codex}"
)"
PROXY_PORT="${CODEX_AUTOMATION_PROXY_PORT:-}"
MODEL_CATALOG="${CODEX_AUTOMATION_MODEL_CATALOG:-}"
MODEL="${CODEX_AUTOMATION_MODEL:-gpt-5.6-sol}"
CODEX_BIN="${CODEX_AUTOMATION_BIN:-/root/.nvm/versions/node/v22.23.1/lib/node_modules/@tencent/tcodex/node_modules/@openai/codex-linux-x64/vendor/x86_64-unknown-linux-musl/bin/codex}"
TIMEOUT_SECONDS="${CODEX_AUTOMATION_TIMEOUT_SECONDS:-1500}"

mkdir -p "$STATE_DIR" "$LOG_DIR"
touch "$LOCK_FILE"

exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  printf '%s SKIP previous automation is still active\n' \
    "$(date --iso-8601=seconds)" >>"$LOG_DIR/codex_automation.log"
  exit 0
fi

if [[ -z "$PROXY_PORT" ]]; then
  PROXY_PORT="$(
    ps -eo args \
      | grep -F "$THREAD_ID" \
      | sed -n 's/.*base_url = "http:\/\/127\.0\.0\.1:\([0-9][0-9]*\)".*/\1/p' \
      | head -n 1
  )"
fi
if [[ -z "$PROXY_PORT" ]] \
  || ! lsof -iTCP:"$PROXY_PORT" -sTCP:LISTEN -nP >/dev/null 2>&1; then
  printf '%s SKIP active thread gateway is unavailable\n' \
    "$(date --iso-8601=seconds)" \
    >>"$LOG_DIR/codex_automation.log"
  exit 0
fi
MODEL_CATALOG="${MODEL_CATALOG:-/root/.tcodex/instances/$PROXY_PORT/models.json}"
if [[ ! -s "$MODEL_CATALOG" || ! -s "$PROMPT_FILE" ]]; then
  printf '%s SKIP model catalog or prompt is missing\n' \
    "$(date --iso-8601=seconds)" >>"$LOG_DIR/codex_automation.log"
  exit 0
fi

timestamp="$(date +%Y%m%dT%H%M%S%z)"
run_log="$LOG_DIR/codex_automation_${timestamp}.log"
last_message="$LOG_DIR/codex_automation_${timestamp}.last.md"
started="$(date --iso-8601=seconds)"
printf '%s START thread=%s log=%s\n' \
  "$started" "$THREAD_ID" "$run_log" >>"$LOG_DIR/codex_automation.log"

set +e
CODEX_HOME="$CODEX_HOME_DIR" timeout --signal=TERM --kill-after=60 \
  "$TIMEOUT_SECONDS" \
  "$CODEX_BIN" \
  -c "model_providers.tencent={ name = \"Tencent Internal Gateway\", base_url = \"http://127.0.0.1:$PROXY_PORT\", wire_api = \"responses\" }" \
  -c 'model_provider="tencent"' \
  -c "model_catalog_json=\"$MODEL_CATALOG\"" \
  -c "model=\"$MODEL\"" \
  -c 'otel.exporter="none"' \
  -c 'otel.trace_exporter="none"' \
  -c 'otel.metrics_exporter="none"' \
  -c 'feedback.enabled=false' \
  -c 'check_for_update_on_startup=false' \
  -c 'suppress_unstable_features_warning=true' \
  -c 'features.item_ids=true' \
  exec resume "$THREAD_ID" \
  --dangerously-bypass-approvals-and-sandbox \
  -o "$last_message" \
  - <"$PROMPT_FILE" >"$run_log" 2>&1
status=$?
set -e

ended="$(date --iso-8601=seconds)"
python3 - "$HEARTBEAT_FILE" "$started" "$ended" "$status" "$run_log" \
  "$last_message" "$THREAD_ID" <<'PY'
import json
import os
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = {
    "started_at": sys.argv[2],
    "ended_at": sys.argv[3],
    "exit_code": int(sys.argv[4]),
    "run_log": sys.argv[5],
    "last_message": sys.argv[6],
    "thread_id": sys.argv[7],
    "last_message_bytes": (
        os.path.getsize(sys.argv[6]) if os.path.exists(sys.argv[6]) else 0
    ),
}
temporary = path.with_suffix(".json.tmp")
temporary.write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
os.replace(temporary, path)
PY

printf '%s END thread=%s exit=%s log=%s\n' \
  "$ended" "$THREAD_ID" "$status" "$run_log" \
  >>"$LOG_DIR/codex_automation.log"
exit "$status"
