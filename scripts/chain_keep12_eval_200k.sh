#!/usr/bin/env bash
# Chain watcher: fire the Paper B ladder eval the moment keep12 reaches step200000.
#
# WHY THIS EXISTS
# ---------------
# .73 is ~10,900 steps (about 24 h) from finishing keep12fresh2 at step200000. Nothing was
# watching for that, on either disk -- `pgrep -af 'watch|chain|eval_paperb'` returned empty
# on LOCAL and on .73. Without a chain the run completes, 8 H20 cards go idle, and the eval
# waits for whenever a heartbeat happens to notice. That is the "cards free, next baton
# dropped" failure this project has already paid for.
#
# WHAT IT WAITS FOR
# -----------------
# The ckpt FILE, not the log line. A log line saying step200000 can appear before the 43.9 GB
# write completes, and eval_paperb_ladder_200k.sh asserts the ckpt exists -- so triggering on
# the log would hand the driver a partial file. This waits for the path to exist AND for its
# size to stop changing across two consecutive polls, which is the cheap way to see a
# completed write without parsing torch's format.
#
# WHAT IT DOES NOT DO
# -------------------
# It does not kill anything, does not touch the running trainer, and does not choose a node:
# it runs ON .73 and uses .73's own disk, because the ckpt is zwfy6-resident and REQUIRE_SM=9.0
# in the driver matches H20. It runs the driver exactly once and exits.
set -uo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
ARM="${ARM:-keep12}"
EXPECT_STEP="${EXPECT_STEP:-200000}"
CKPT="${CKPT:-$PROJECT_ROOT/outputs/olmo2_probe2_7B_${ARM}fresh2/step${EXPECT_STEP}.pt}"
POLL="${POLL:-300}"
MAX_WAIT_H="${MAX_WAIT_H:-48}"
LOG="${LOG:-$PROJECT_ROOT/logs/chain_${ARM}_eval_200k.log}"

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

say "=== chain watcher start: ARM=$ARM waiting for $CKPT ==="
say "poll=${POLL}s  max_wait=${MAX_WAIT_H}h  driver=scripts/eval_paperb_ladder_200k.sh"

deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
prev_size=-1
stable=0

while :; do
  now=$(date +%s)
  if [ "$now" -ge "$deadline" ]; then
    say "FATAL: ${MAX_WAIT_H}h elapsed without a complete ckpt. NOT launching eval."
    say "       Check the trainer: it may have stalled, crashed, or changed save_every."
    exit 3
  fi

  if [ -f "$CKPT" ]; then
    size=$(stat -c %s "$CKPT" 2>/dev/null || echo 0)
    if [ "$size" -gt 0 ] && [ "$size" -eq "$prev_size" ]; then
      stable=$(( stable + 1 ))
      say "ckpt present, size stable at $size bytes (${stable} consecutive polls)"
      # two consecutive equal sizes == the write finished at least POLL seconds ago
      if [ "$stable" -ge 2 ]; then
        say "ckpt write confirmed complete. Launching the ladder eval."
        break
      fi
    else
      say "ckpt present, size $size (was $prev_size) -- still writing"
      stable=0
    fi
    prev_size="$size"
  else
    # report progress from the trainer's own log so a human reading this file sees why we wait
    tl=$(ls -t "$PROJECT_ROOT"/logs/olmo2_7B_${ARM}fresh2_resume200k_*.log 2>/dev/null | head -1)
    step=""
    if [ -n "$tl" ]; then
      step=$(tail -c 2000 "$tl" | tr '\r' '\n' | grep -aoE '\[step [0-9]+/[0-9]+\]' | tail -1)
    fi
    say "ckpt absent; trainer at ${step:-unknown}"
  fi
  sleep "$POLL"
done

say "=== launching: ARM=$ARM PROJECT_ROOT=$PROJECT_ROOT scripts/eval_paperb_ladder_200k.sh ==="
cd "$PROJECT_ROOT" || { say "FATAL: cannot cd $PROJECT_ROOT"; exit 4; }
ARM="$ARM" PROJECT_ROOT="$PROJECT_ROOT" EXPECT_STEP="$EXPECT_STEP" \
  bash scripts/eval_paperb_ladder_200k.sh >>"$LOG" 2>&1
rc=$?
say "=== eval driver exited rc=$rc ==="
exit "$rc"
