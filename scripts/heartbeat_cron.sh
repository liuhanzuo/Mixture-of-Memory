#!/usr/bin/env bash
# Stateless heartbeat runner — invoked by system crontab.
# Each run is a FRESH `codebuddy -p` process with NO conversation history,
# so input tokens are bounded (~system prompt + heartbeat.md + status tails).
# This replaces the in-session cron job that re-sent the whole growing
# conversation every 30 min (the real ~$10 cost driver).
#
# Install (every 2h):
#   crontab -e
#   0 */2 * * * /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/scripts/heartbeat_cron.sh
#
# State is preserved entirely in status/ files (TRAINER_ACTIVITY.jsonl,
# MEMORY_PROTOCOL_PLAN.md, PENDING_TASKS.md, TRAINER_ACTIVE.md) — the
# heartbeat reads them fresh each time, so losing in-process memory is fine.

set -uo pipefail

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
CODEBUDDY_BIN="/root/.nvm/versions/node/v20.20.2/bin/codebuddy"
MODEL="claude-opus-4.8-1m"
LOG_DIR="${PROJECT_ROOT}/logs/heartbeat_cron"

cd "$PROJECT_ROOT" || exit 1
mkdir -p "$LOG_DIR"

TS="$(date '+%Y%m%d_%H%M%S')"
OUT="${LOG_DIR}/hb_${TS}.log"

# Wandb key so any launched training can log; offline-safe.
export WANDB_API_KEY="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"

# Fresh process, non-interactive, no history. -y bypasses prompts so it can
# act (kill/restart/launch) autonomously per CODEBUDDY.md authorization.
# -k 60: if it ignores SIGTERM after 30min, SIGKILL 60s later (node child reap).
# stdin from /dev/null so print-mode doesn't wait on a tty.
timeout -k 60 1800 "$CODEBUDDY_BIN" \
  --print \
  --model "$MODEL" \
  --dangerously-skip-permissions \
  "/heartbeat" \
  < /dev/null > "$OUT" 2>&1

EC=$?
echo "[heartbeat_cron] ${TS} exit=${EC} log=${OUT}" >> "${LOG_DIR}/runner.log"

# Keep only last 200 per-run logs to avoid unbounded growth.
ls -1t "${LOG_DIR}"/hb_*.log 2>/dev/null | tail -n +201 | xargs -r rm -f
exit "$EC"
