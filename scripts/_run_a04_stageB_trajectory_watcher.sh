#!/usr/bin/env bash
# A04 Pilot One Stage B -- eval watcher. Same v3 write-guard shape as the
# a03_dataorder watcher; only these config lines change:
#   STEP=5000
#   CKDIR=outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed<S>
#   TAG=A04_1B_stageB_keep12_seed<S>_step5000
#   SEED whitelist 101|102|103
#   ext_drv = /tmp/a04_stageB_ext_driver.sh (or wherever the caller puts it)
#
# v3 guard, all four required before handing off to the ext driver:
#   (a) mtime age >= 120 s
#   (b) size within 64 KiB of a settled sibling (step2500.pt)
#   (c) size unchanged across two probes 60 s apart
#   (d) torch.load succeeds (inside ext_drv)
#
# Sibling reference: this run has ONE earlier ckpt (step2500.pt at --save_every 2500),
# so ref_size() looks there only. If step2500.pt is missing (should never happen
# since the safestop in _run_a04_stageB.sh runs to step5000), we still enforce
# guards (a), (c), (d) -- one weaker but not disarmed. Do NOT change the sibling
# list without matching --save_every.
#
# Usage: SEED=101 EXT_DRV=/tmp/a04_stageB_ext_driver.sh bash scripts/_run_a04_stageB_trajectory_watcher.sh
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W || exit 3
SEED="${SEED:?SEED must be set (101|102|103)}"
case "$SEED" in 101|102|103) ;; *)
  echo "FATAL seed=$SEED outside pre-registered whitelist {101,102,103}"; exit 4 ;;
esac
STEP=5000
CKDIR=outputs/olmo2_probe2_1B_keep12f2_dolmino_stageB_seed${SEED}
TAG=A04_1B_stageB_keep12_seed${SEED}_step${STEP}
PROG=logs/a04_stageB_seed${SEED}_eval_progress.log
EXT_DRV="${EXT_DRV:-/tmp/a04_stageB_ext_driver.sh}"
STALE_S=120
MAX_LOOPS=600           # ~10 h at 60 s/loop
REFUSE_STREAK=0
REFUSE_ALARM_AT=10

note() { printf "[%s] a04-watcher(seed%s): %s\n" "$(date "+%m-%d %H:%M:%S")" "$SEED" "$*" | tee -a $PROG; }

ref_size() {
  local r=$CKDIR/step2500.pt
  if [ -f "$r" ]; then stat -c %s "$r"; return 0; fi
  return 1
}

note "v3 started; watching $CKDIR/step${STEP}.pt; tag=$TAG; ext_drv=$EXT_DRV"

fully_written() {
  local f="$1" s1 s2 age diff R
  [ -f "$f" ] || return 1
  age=$(( $(date +%s) - $(stat -c %Y "$f") ))
  if [ "$age" -lt $STALE_S ]; then note "$(basename "$f") only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  if R=$(ref_size); then
    diff=$(( s1 > R ? s1 - R : R - s1 ))
    if [ "$diff" -gt 65536 ]; then
      REFUSE_STREAK=$((REFUSE_STREAK+1))
      note "REFUSE: size $s1 vs sibling ref $R (diff ${diff}B > 64KiB) -- likely mid-save/corrupt [streak=$REFUSE_STREAK]"
      if [ $REFUSE_STREAK -ge $REFUSE_ALARM_AT ]; then
        echo "TRUNCATED at ${diff}B off ref $R after ${REFUSE_ALARM_AT} refusals" > $CKDIR/TRUNCATED_step${STEP}.ALARM
        note "ALARM: $CKDIR/TRUNCATED_step${STEP}.ALARM written -- refusing to score for seed$SEED"
        return 2
      fi
      return 1
    fi
  else
    note "no settled sibling (step2500.pt missing) -- relying on stability + torch.load"
  fi
  sleep 60
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  if [ "$s1" != "$s2" ]; then note "still growing ($s1 -> $s2); waiting"; return 1; fi
  REFUSE_STREAK=0
  return 0
}

LOOPS=0
while [ $LOOPS -lt $MAX_LOOPS ]; do
  LOOPS=$((LOOPS+1))
  # exit cleanly if all 3 result summaries are already on disk
  if [ -f "olmo2_mmlu_content_results/$TAG/summary.json" ] \
     && [ -f "olmo2_closedbook_results/$TAG/summary.json" ] \
     && [ -f "olmo2_closedbook_results/${TAG}_nq/summary.json" ]; then
    note "all 3 summaries exist for $TAG; exiting"
    exit 0
  fi
  fully_written "$CKDIR/step${STEP}.pt"
  rc=$?
  if [ "$rc" -eq 0 ]; then
    note "firing ext-drv for seed$SEED step${STEP}"
    SEEDS="$SEED" bash "$EXT_DRV"
  elif [ "$rc" -eq 2 ]; then
    exit 2   # TRUNCATED alarm -- do not score
  fi
  sleep 60
done
note "watcher loop budget exhausted"
exit 9
