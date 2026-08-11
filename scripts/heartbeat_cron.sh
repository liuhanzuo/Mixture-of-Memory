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

# ⚠️ STALE (2026-08-11 实测): 此脚本当前 **未在 crontab 中**, 且下面的 PROJECT_ROOT
# (/apdcephfs_zwfy6/share_303098609/...) 在任何节点上都已不存在 —— 盘 303098609 已退役。
# 若要复活: PROJECT_ROOT 改为 /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
# (wzc1) 或 /apdcephfs_zwfy6/share_304376610/... (zwfy6)。
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
CODEBUDDY_BIN="/root/.nvm/versions/node/v20.20.2/bin/codebuddy"
# 模型别名 `opus` → 由 ANTHROPIC_DEFAULT_OPUS_MODEL 解析 (当前 claude-opus-5[1m])。
# 不要写死版本号 —— 写死会在网关升级后静默降级。见 CODEBUDDY.md 模型配置章节。
MODEL="opus"
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
