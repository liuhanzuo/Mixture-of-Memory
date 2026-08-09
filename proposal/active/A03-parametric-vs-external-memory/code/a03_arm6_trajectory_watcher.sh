#!/usr/bin/env bash
# Watch for Arm 6 (mid-low-LR CPT) ckpts on the shared zwfy6 disk, fire the
# arm6 ext-drv per ckpt. Same v3 guard as Arm 4 watcher (size ± 64 KiB
# tolerance vs a known-good sibling ckpt AND stable across 60 s), so
# post-SIGKILL truncations and torch.save pickle byte drift are both handled.
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W
PROG=logs/a03_arm6_trajectory_progress.log
CKDIR=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm6_lowerband20k
STALE_S=120
EXT_DRV=/tmp/a03_arm6_ext_driver.sh

note() { printf "[%s] watcher: %s\n" "$(date "+%m-%d %H:%M:%S")" "$*" | tee -a $PROG; }

REF_SIZE=""
for ref in $CKDIR/step205000.pt $CKDIR/step210000.pt $CKDIR/step215000.pt; do
  [ -f "$ref" ] && REF_SIZE=$(stat -c %s "$ref") && break
done
note "v3 started for Arm 6; REF_SIZE=${REF_SIZE:-unknown} bytes; watching $CKDIR"

fully_written() {
  local f="$1" s1 s2 age diff
  [ -f "$f" ] || return 1
  age=$(($(date +%s) - $(stat -c %Y "$f")))
  if [ $age -lt $STALE_S ]; then note "$(basename $f) only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  if [ -n "$REF_SIZE" ]; then
    diff=$(( s1 > REF_SIZE ? s1 - REF_SIZE : REF_SIZE - s1 ))
    if [ "$diff" -gt 65536 ]; then
      note "REFUSE $(basename $f): size $s1 vs reference $REF_SIZE (diff ${diff} B > 64 KiB -- likely corrupt)"
      return 1
    fi
  fi
  sleep 60
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  [ "$s1" = "$s2" ] || { note "$(basename $f) still growing ($s1 -> $s2); waiting"; return 1; }
  return 0
}

LOOPS=0
MAX_LOOPS=900
while [ $LOOPS -lt $MAX_LOOPS ]; do
  LOOPS=$((LOOPS+1))
  done_count=0
  for STEP in 205000 210000 215000 220000; do
    [ -f olmo2_mmlu_content_results/A03_1B_arm6_lowerband_step${STEP}/summary.json ] && done_count=$((done_count+1))
  done
  if [ $done_count -eq 4 ]; then
    note "all 4 Arm6 MMLU summaries exist; exiting"
    break
  fi
  should_fire=0
  for STEP in 205000 210000 215000 220000; do
    CK=$CKDIR/step${STEP}.pt
    if [ ! -f "olmo2_mmlu_content_results/A03_1B_arm6_lowerband_step${STEP}/summary.json" ]; then
      if fully_written "$CK"; then should_fire=1; fi
    fi
  done
  if [ $should_fire -eq 1 ]; then
    note "firing ext-drv"
    bash $EXT_DRV
  fi
  sleep 60
done
note "watcher exit"
