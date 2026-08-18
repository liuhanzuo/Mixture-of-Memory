#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERVAL_SECONDS="${CODEX_AUTOMATION_INTERVAL_SECONDS:-1800}"
INITIAL_DELAY_SECONDS="${CODEX_AUTOMATION_INITIAL_DELAY_SECONDS:-1800}"
NEXT_RUN_FILE="$ROOT/ops/state/codex_automation_next_run_at"

if (( INITIAL_DELAY_SECONDS > 0 )); then
  next_epoch="$(( $(date +%s) + INITIAL_DELAY_SECONDS ))"
  date --date="@$next_epoch" --iso-8601=seconds >"$NEXT_RUN_FILE"
  sleep "$INITIAL_DELAY_SECONDS"
fi

while true; do
  cycle_started="$(date +%s)"
  "$ROOT/ops/codex_automation.sh" || true
  elapsed="$(( $(date +%s) - cycle_started ))"
  sleep_seconds="$(( INTERVAL_SECONDS - elapsed ))"
  if (( sleep_seconds < 1 )); then
    sleep_seconds=1
  fi
  next_epoch="$(( $(date +%s) + sleep_seconds ))"
  date --date="@$next_epoch" --iso-8601=seconds >"$NEXT_RUN_FILE"
  sleep "$sleep_seconds"
done
