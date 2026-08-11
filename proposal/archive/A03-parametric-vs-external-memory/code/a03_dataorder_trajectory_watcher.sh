#!/usr/bin/env bash
# A03 data-order replication -- eval watcher (one instance per node, one seed).
#
# Structure copied from code/a03_arm6_trajectory_watcher.sh (known-good: it drove
# all 4 Arm 6 dose points on 2026-08-10 and exited cleanly). Same v3 write guard.
# Differences, both deliberate:
#   * watches step220000 ONLY (see DATAORDER_PREREG.md -- one pre-registered step)
#   * SEED-parameterised, because zwfy6 is shared by .73/.82/.104 and each node
#     must only ever touch its own run's result dirs.
#
# v3 write guard, all four must hold before the ckpt is handed to the eval driver:
#   (a) mtime age >= 120 s
#   (b) size within 64 KiB of a known-good sibling ckpt of the SAME run
#   (c) size unchanged across two probes 60 s apart
#   (d) torch.load succeeds (probe lives inside the ext driver)
# (b) is what caught Arm 4's step220000.pt truncated at 5,956,287,104 B of
# 12,181,311,650 B (49%). Do not weaken it.
#
# ⚠️ ONE REAL FIX vs the Arm 6 watcher, do not "simplify" it back:
# the Arm 6 / Arm 4 watchers resolve REF_SIZE **once at startup**. Arm 6's
# watcher started 04:58:33, its first sibling ckpt landed 07:47:51, so it logged
# "REF_SIZE=unknown bytes" and guard (b) was silently DISABLED for all four of
# its dose points (0 REFUSE lines in logs/a03_arm6_trajectory_progress.log). Arm 6
# is therefore only "known-good" on guards (a)+(c)+(d); its size guard never ran.
# By contrast Arm 4's watcher started with siblings present, captured
# REF_SIZE=12181311650, and logged 12 REFUSE lines -- that is the incident the
# tolerance exists for. Here ref_size() is re-resolved EVERY loop, so the guard
# arms itself as soon as a sibling appears, and the watcher refuses to score at
# all while no sibling exists (better to wait than to score unguarded).
# Safe against retention: rotation keeps every multiple of --milestone_every=5000
# (keep_milestones=0 = unlimited), so step205000/210000/215000 all survive to
# serve as references -- verified on Arm 6, whose four ckpts are all still on disk
# at identical size 12,181,311,650 B.
#
# ---------------------------------------------------------------------------
# ⚠️ KNOWN UPSTREAM DEFECT THIS WATCHER CANNOT FIX (2026-08-10, MAIN)
# ---------------------------------------------------------------------------
# scripts/_run_a03_dataorder_repl.sh -- the script currently RUNNING both seed 43
# (.82) and seed 44 (.73) -- carries the *v1* trainer-side stop watcher:
#     if [ -f "$OUT/step220000.pt" ]; then kill -TERM; sleep 20; kill -9 ...
# That is the exact bare `-f` race that truncated Arm 4's step220000.pt. It was
# fixed in _run_a03_arm4_peaklr.sh and _run_a03_arm6_lowerband.sh (v3 settled-size
# guard) but never back-ported here. Measured save window on this filesystem:
# arm6 step220000 logged "saved" 13 s after its step line; a kill inside that
# window truncates. Poll interval is 60 s => roughly a 1-in-5 chance per run.
# The runs cannot be patched now (bash reads a running script incrementally;
# editing it mid-flight is itself a corruption risk), so this watcher instead
# DETECTS the outcome and refuses to score a truncated ckpt, writing a
# TRUNCATED marker. Remedy is pre-registered in DATAORDER_PREREG.md: re-run the
# full 20k from step200000. NEVER resume from step215000 -- that is precisely
# Arm 4's dataloader-offset defect (original-vs-redo loss r = -0.0667).
set -u
W=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
cd $W || exit 3
SEED="${SEED:?SEED must be set (43|44|45)}"
STEP=220000
CKDIR=outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed${SEED}
TAG=A03_1B_dataorder_seed${SEED}_step${STEP}
PROG=logs/a03_dataorder_seed${SEED}_eval_progress.log
EXT_DRV="${EXT_DRV:-/tmp/a03_dataorder_ext_driver.sh}"
STALE_S=120
MAX_LOOPS=1500          # ~25 h at 60 s/loop; step220000 due ~04:15 GMT+8
REFUSE_STREAK=0
REFUSE_ALARM_AT=10      # ~10 consecutive size refusals => call it truncated

note() { printf "[%s] watcher(seed%s): %s\n" "$(date "+%m-%d %H:%M:%S")" "$SEED" "$*" | tee -a $PROG; }

ref_size() {   # recomputed every loop: siblings appear as the run progresses
  local r
  for r in $CKDIR/step205000.pt $CKDIR/step210000.pt $CKDIR/step215000.pt; do
    if [ -f "$r" ]; then stat -c %s "$r"; return 0; fi
  done
  return 1
}

note "v3 started; watching $CKDIR/step${STEP}.pt; tag=$TAG; ext_drv=$EXT_DRV"

fully_written() {
  local f="$1" s1 s2 age diff R
  [ -f "$f" ] || return 1
  age=$(($(date +%s) - $(stat -c %Y "$f")))
  if [ $age -lt $STALE_S ]; then note "step${STEP}.pt only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  if R=$(ref_size); then
    diff=$(( s1 > R ? s1 - R : R - s1 ))
    if [ "$diff" -gt 65536 ]; then
      REFUSE_STREAK=$((REFUSE_STREAK+1))
      note "REFUSE step${STEP}.pt: size $s1 vs sibling reference $R (diff ${diff} B > 64 KiB -- likely truncated) [streak $REFUSE_STREAK]"
      if [ $REFUSE_STREAK -ge $REFUSE_ALARM_AT ]; then
        printf '%s\n' \
          "TRUNCATED $CKDIR/step${STEP}.pt" \
          "observed_size=$s1 reference_sibling_size=$R diff_bytes=$diff" \
          "detected=$(date -Iseconds)" \
          "cause: scripts/_run_a03_dataorder_repl.sh ships the v1 bare-[-f] trainer-side" \
          "stop watcher (kill -TERM; sleep 20; kill -9) and fired inside the ~13 s" \
          "torch.save window. Same defect as Arm 4's step220000.pt (49% truncation)." \
          "REMEDY (pre-registered, DATAORDER_PREREG.md): re-run the full 20k from" \
          "step200000 with the v3 settled-size stop guard. DO NOT resume from" \
          "step215000 -- that reproduces Arm 4's dataloader-offset defect" \
          "(original-vs-redo loss r = -0.0667) and voids the matched-20k-exposure" \
          "premise of the comparison." \
          > $CKDIR/TRUNCATED_step${STEP}.ALARM
        note "ALARM written: $CKDIR/TRUNCATED_step${STEP}.ALARM -- no eval will be run for seed$SEED"
      fi
      return 1
    fi
  else
    note "no sibling ckpt yet; size guard cannot run -- waiting (will not score unguarded)"
    return 1
  fi
  sleep 60
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  [ "$s1" = "$s2" ] || { note "step${STEP}.pt still growing ($s1 -> $s2); waiting"; return 1; }
  REFUSE_STREAK=0
  return 0
}

LOOPS=0
while [ $LOOPS -lt $MAX_LOOPS ]; do
  LOOPS=$((LOOPS+1))
  if [ -f "olmo2_mmlu_content_results/$TAG/summary.json" ] \
     && [ -f "olmo2_closedbook_results/$TAG/summary.json" ] \
     && [ -f "olmo2_closedbook_results/${TAG}_nq/summary.json" ]; then
    note "all 3 summaries exist for $TAG (mmlu + cb(pt) + cb(nq)); exiting"
    break
  fi
  if fully_written "$CKDIR/step${STEP}.pt"; then
    note "firing ext-drv for seed$SEED step${STEP}"
    SEEDS="$SEED" bash "$EXT_DRV"
  fi
  sleep 60
done
note "watcher exit (loops=$LOOPS)"
