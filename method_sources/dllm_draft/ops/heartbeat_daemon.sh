#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
while true; do
  "$ROOT/ops/heartbeat.py" >>"$ROOT/ops/logs/heartbeat_daemon.log" 2>&1 || true
  INTERVAL="$(
    python3 - "$ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
config = json.load(open(root / "ops" / "config.json"))
if (root / "ops" / "state" / "active_run.json").exists():
    print(config.get("active_poll_seconds", 60))
else:
    ready = False
    queue = root / "ops" / "queue.tsv"
    if queue.exists():
        for line in queue.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#"):
                continue
            row = next(csv.reader([line], delimiter="\t"))
            if row and row[0] == "READY":
                ready = True
                break
    if ready:
        print(config.get("queued_poll_seconds", 300))
    else:
        print(config["heartbeat_interval_seconds"])
PY
  )"
  sleep "$INTERVAL"
done
