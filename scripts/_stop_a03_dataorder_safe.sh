#!/usr/bin/env bash
# A03 data-order runs: SAFE stopper. Replaces the racy stop-loop inside
# scripts/_run_a03_dataorder_repl.sh, which fires `kill -TERM` on a bare
# `[ -f step220000.pt ]` test and `kill -9` twenty seconds later.
#
# WHY THIS EXISTS
# ---------------
# A 12 GB ckpt takes ~13 s to write (measured from Arm 6's log: 16:20:04 START ->
# 16:20:17 present). The racy loop polls every 60 s, so a poll can land mid-save
# and the 20 s grace can expire before torch.save finishes -> truncated ckpt.
# That is exactly how Arm 4's step220000 was truncated at 49%. The risk is WORSE
# here than for Arm 4: two 12 GB saves now contend for the same zwfy6 disk, so
# the save window can stretch well past 20 s.
#
# The ckpt at step220000 is the ONLY artifact these runs exist to produce. A
# truncated one costs a full 11.4 h re-run, so a few extra minutes of training
# past step220000 is a trivial price for a guaranteed-intact file.
#
# GUARD (same shape as the arm4/arm6 v3 guard, with REF_SIZE re-resolved every
# loop rather than once at startup -- Arm 6's watcher resolved it once before any
# sibling existed and therefore ran with REF_SIZE=unknown, i.e. unarmed):
#   1. file exists
#   2. mtime age >= 120 s
#   3. size within 64 KiB of a known-good sibling ckpt from THIS run
#   4. size stable across two probes 60 s apart
#   5. torch.load structural probe succeeds
# Only then stop training.
#
# Usage: SEED=43 bash scripts/_stop_a03_dataorder_safe.sh
set -u

SEED="${SEED:?SEED must be set}"
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed${SEED}
CK=$OUT/step220000.pt
PROG=logs/a03_dataorder_seed${SEED}_safestop.log
STALE_S=120
TOL=65536

note() { printf '[%s] safestop(seed%s): %s\n' "$(date '+%m-%d %H:%M:%S')" "$SEED" "$*" | tee -a "$PROG"; }

note "started; watching $CK; will NOT kill until the file is provably complete"

ref_size() {
  local f s
  for f in $OUT/step205000.pt $OUT/step210000.pt $OUT/step215000.pt; do
    if [ -f "$f" ]; then
      s=$(stat -c %s "$f" 2>/dev/null) || continue
      # only trust a sibling that is itself settled
      if [ $(( $(date +%s) - $(stat -c %Y "$f") )) -ge $STALE_S ]; then
        echo "$s"; return 0
      fi
    fi
  done
  return 1
}

complete() {
  [ -f "$CK" ] || return 1
  local age s1 s2 ref diff
  age=$(( $(date +%s) - $(stat -c %Y "$CK") ))
  if [ "$age" -lt $STALE_S ]; then note "ckpt only ${age}s old; waiting"; return 1; fi
  s1=$(stat -c %s "$CK" 2>/dev/null) || return 1
  if ref=$(ref_size); then
    diff=$(( s1 > ref ? s1 - ref : ref - s1 ))
    if [ "$diff" -gt $TOL ]; then
      note "REFUSE: size $s1 vs sibling ref $ref (diff ${diff}B > 64KiB) -- likely mid-save or corrupt"
      return 1
    fi
  else
    note "no settled sibling yet for a size reference; relying on stability+torch.load"
  fi
  sleep 60
  s2=$(stat -c %s "$CK" 2>/dev/null) || return 1
  if [ "$s1" != "$s2" ]; then note "still growing ($s1 -> $s2); waiting"; return 1; fi
  if ! "$PY" - "$CK" <<'PYEOF' >>"$PROG" 2>&1
import sys, torch
p = sys.argv[1]
sd = torch.load(p, map_location="cpu", weights_only=False)
n = len(sd["model"]) if isinstance(sd, dict) and "model" in sd else len(sd)
print(f"  torch.load OK: {n} top-level entries")
PYEOF
  then
    note "REFUSE: torch.load probe FAILED -- file is not a loadable ckpt yet"
    return 1
  fi
  note "ckpt COMPLETE (size $s1, stable, torch.load OK)"
  return 0
}

LOOPS=0
while [ $LOOPS -lt 1200 ]; do
  LOOPS=$((LOOPS+1))
  # stop if training already exited on its own
  if ! pgrep -f "train_olmo2_arch_probe2.py.*dataorder_seed${SEED}" >/dev/null 2>&1; then
    note "no training process for seed$SEED; nothing to stop; exiting"
    exit 0
  fi
  if complete; then
    note "stopping training now that step220000.pt is verified intact"
    for p in $(pgrep -f "torch.distributed.run.*dataorder_seed${SEED}" 2>/dev/null); do
      note "kill -TERM torchrun $p"; kill -TERM "$p" 2>/dev/null
    done
    sleep 30
    for p in $(pgrep -f "train_olmo2_arch_probe2.py.*dataorder_seed${SEED}" 2>/dev/null); do
      note "kill -9 worker $p"; kill -9 "$p" 2>/dev/null
    done
    sleep 5
    note "stopped. ckpts: $(ls $OUT/step*.pt 2>/dev/null | tr '\n' ' ')"
    note "eval is handled separately by code/a03_dataorder_trajectory_watcher.sh (SEED-pinned)"
    exit 0
  fi
  sleep 60
done
note "loop budget exhausted without a complete ckpt -- NOT killing; investigate"
exit 9
