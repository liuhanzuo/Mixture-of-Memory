#!/usr/bin/env bash
# Watch for Arm 4 (peak-LR CPT) ckpts on the shared zwfy6 disk, fire same
# idempotent 4-axis eval driver as Arm 3 (MMLU + CB pt + CB nq).
#
# 2026-08-10 FIX. The v1 guard was mtime-age >= 120s ONLY. Its header claimed a
# "torch.load dry-run" that was never in the script. Age alone cannot tell
# "finished writing" from "SIGKILLed mid-write, so mtime froze" -- exactly what
# happened at 00:47 when the trainer-side watcher truncated step220000.pt at
# 5,956,287,104 B / 12,181,311,650 B (49%). This watcher DID fire on that corrupt
# file at 00:49:54; it was stopped only by an unrelated GPU-held REFUSE. Luck,
# not design.
#
# v2 guard, all three must hold before firing:
#   (a) size == a known-good sibling ckpt (all ckpts of one run are equal-size)
#   (b) size unchanged across two probes 60s apart (not still streaming)
#   (c) mtime age >= 120s (kept from v1)
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W
PROG=logs/a03_arm4_trajectory_progress.log
CKDIR=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k
STALE_S=120

note() { printf "[%s] watcher: %s\n" "$(date "+%m-%d %H:%M:%S")" "$*" | tee -a $PROG; }

REF_SIZE=""
for ref in $CKDIR/step205000.pt $CKDIR/step210000.pt $CKDIR/step215000.pt; do
  [ -f "$ref" ] && REF_SIZE=$(stat -c %s "$ref") && break
done
note "v2 started; REF_SIZE=${REF_SIZE:-unknown} bytes; watching $CKDIR for step205000/210000/215000/220000"

fully_written() {  # 0 iff $1 is complete by (a)+(b)+(c)
  local f="$1" s1 s2 age
  [ -f "$f" ] || return 1
  age=$(($(date +%s) - $(stat -c %Y "$f")))
  if [ $age -lt $STALE_S ]; then note "$(basename $f) only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  if [ -n "$REF_SIZE" ] && [ "$s1" != "$REF_SIZE" ]; then
    note "REFUSE $(basename $f): size $s1 != reference $REF_SIZE (truncated/corrupt)"; return 1
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
    [ -f olmo2_mmlu_content_results/A03_1B_arm4_peaklr_step${STEP}/summary.json ] && done_count=$((done_count+1))
  done
  if [ $done_count -eq 4 ]; then
    note "all 4 Arm4 MMLU summaries exist; exiting"
    break
  fi
  should_fire=0
  for STEP in 205000 210000 215000 220000; do
    CK=$CKDIR/step${STEP}.pt
    if [ ! -f "olmo2_mmlu_content_results/A03_1B_arm4_peaklr_step${STEP}/summary.json" ]; then
      if fully_written "$CK"; then should_fire=1; fi
    fi
  done
  if [ $should_fire -eq 1 ]; then
    note "firing ext-drv"
    bash /tmp/a03_arm4_ext_driver.sh
  fi
  sleep 60
done
note "watcher exit"
