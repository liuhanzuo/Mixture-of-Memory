#!/usr/bin/env bash
# hb_emit_alert.sh — append a structured, de-duplicated alert to
# status/HEARTBEAT_ALERTS.jsonl so the main session's probe cron can wake up.
#
# WHY: the stateless /heartbeat process (scripts/heartbeat_cron.sh) is isolated
# and cannot talk to the main session. Routine health checks stay SILENT (only
# TRAINER_ACTIVITY.jsonl flows). Only three event classes leave an alert here:
#   train_done | train_anomaly | needs_code
#
# Dedup: an alert is keyed by `id`. If a line with the same `id` already exists
# in the file (regardless of ack value), this script is a no-op. So a run that
# finished step 5000 only ever produces ONE train_done alert, no matter how many
# heartbeats observe it. Build a STABLE id, e.g. "train_done:<exp>:<step>".
#
# ack semantics:
#   - WRITER (this script / heartbeat) always writes "ack": false.
#   - READER (main session probe) flips it to true after handling. Never written here.
#
# Usage:
#   scripts/hb_emit_alert.sh \
#     --event-class train_done|train_anomaly|needs_code \
#     --severity   info|warning|critical \
#     --id         "train_done:dolmino_bugfix:2000" \
#     --summary    "one-line human readable" \
#     --detail     "run/step/loss/log path + suggested action" \
#     [--node      "local|<ip>|<run>"]
#
# Exit codes: 0 = appended OR skipped-as-duplicate (both success); 2 = bad args.
set -uo pipefail

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
ALERTS_FILE="${PROJECT_ROOT}/status/HEARTBEAT_ALERTS.jsonl"
PYBIN="${PYTHON_BIN:-python3}"

EVENT_CLASS=""
SEVERITY=""
ALERT_ID=""
SUMMARY=""
DETAIL=""
NODE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --event-class) EVENT_CLASS="$2"; shift 2 ;;
    --severity)    SEVERITY="$2";    shift 2 ;;
    --id)          ALERT_ID="$2";    shift 2 ;;
    --summary)     SUMMARY="$2";     shift 2 ;;
    --detail)      DETAIL="$2";      shift 2 ;;
    --node)        NODE="$2";        shift 2 ;;
    *) echo "[hb_emit_alert] unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Validate required fields.
if [[ -z "$EVENT_CLASS" || -z "$SEVERITY" || -z "$ALERT_ID" || -z "$SUMMARY" ]]; then
  echo "[hb_emit_alert] missing required arg (need --event-class --severity --id --summary)" >&2
  exit 2
fi
case "$EVENT_CLASS" in
  train_done|train_anomaly|needs_code) ;;
  *) echo "[hb_emit_alert] bad --event-class: $EVENT_CLASS (train_done|train_anomaly|needs_code)" >&2; exit 2 ;;
esac
case "$SEVERITY" in
  info|warning|critical) ;;
  *) echo "[hb_emit_alert] bad --severity: $SEVERITY (info|warning|critical)" >&2; exit 2 ;;
esac

mkdir -p "$(dirname "$ALERTS_FILE")"
touch "$ALERTS_FILE"

# Dedup by exact id match on the JSON "id" field. fixed-string + literal key
# pattern avoids regex surprises in the id.
if grep -qF "\"id\": \"${ALERT_ID}\"" "$ALERTS_FILE" 2>/dev/null \
   || grep -qF "\"id\":\"${ALERT_ID}\"" "$ALERTS_FILE" 2>/dev/null; then
  echo "[hb_emit_alert] duplicate id '${ALERT_ID}' — skipping append" >&2
  exit 0
fi

# Build the JSON line with python for safe escaping of arbitrary detail strings.
EVENT_CLASS="$EVENT_CLASS" SEVERITY="$SEVERITY" ALERT_ID="$ALERT_ID" \
SUMMARY="$SUMMARY" DETAIL="$DETAIL" NODE="$NODE" ALERTS_FILE="$ALERTS_FILE" \
"$PYBIN" - <<'PY'
import json, os, datetime
alerts_file = os.environ["ALERTS_FILE"]
ts = datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z")
# normalize +0800 -> +08:00
if len(ts) >= 5 and (ts[-5] in "+-"):
    ts = ts[:-2] + ":" + ts[-2:]
rec = {
    "ts": ts,
    "id": os.environ["ALERT_ID"],
    "severity": os.environ["SEVERITY"],
    "event_class": os.environ["EVENT_CLASS"],
    "node": os.environ.get("NODE", "") or "",
    "summary": os.environ["SUMMARY"],
    "detail": os.environ.get("DETAIL", "") or "",
    "ack": False,
}
# FIX (2026-06-10): use compact separators so the key/value is written as
# "ack":false (no space). The default json.dumps emits "ack": false (WITH a
# space), which silently broke the probe's `grep '"ack":false'` — alerts went
# unnoticed for ~hours. Compact form matches BOTH the naive grep and the
# space-tolerant `grep -E '"ack":[[:space:]]*false'`.
with open(alerts_file, "a", encoding="utf-8") as f:
    f.write(json.dumps(rec, ensure_ascii=False, separators=(",", ":")) + "\n")
print("[hb_emit_alert] appended alert id=%s class=%s sev=%s" %
      (rec["id"], rec["event_class"], rec["severity"]))
PY
exit 0
