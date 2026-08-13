#!/usr/bin/env bash
# A04 SHALLOW RUNG LADDER -- chain watcher: wait for THIS node's training to
# finish, then run the 4-axis eval on the same node.
#
# WHY A CHAIN. The two arms finish ~4.4 h after launch and the eval is only ~5
# min. Waiting for a human/heartbeat to notice would leave 8 idle H20s for an
# unknown interval. This waits on the ACTUAL completion conditions and fires
# immediately.
#
# THE WAIT CONDITIONS ARE STRUCTURAL, NOT A SINGLE SAMPLE (memory:
# one-sample-is-not-a-trend-or-state). It requires ALL of:
#   1. step5000.pt exists, is >= 120 s old, and its size is stable across a 60 s
#      re-stat (so a partial write is never scored);
#   2. torch.load succeeds AND the ckpt's own keep/fresh/depth/step/seed match the
#      arm we think we are evaluating;
#   3. NO training worker matching this arm's --output_dir is alive (checked by
#      pgrep on the full output_dir token, never a bare `pkill -f` pattern that
#      could match an eval process's --output_name);
#   4. total GPU memory held < 8000 MiB, sampled 3 TIMES 20 s apart -- a single
#      sample can catch the instant between two allocations.
# It NEVER kills anything. If the training is stuck it just keeps waiting and the
# loop budget expires, leaving the situation for a human to read.
#
# Usage (ON the target node): KEEP=14 setsid nohup bash <this> &
set -u
KEEP="${KEEP:?KEEP must be set (13 or 14)}"
SEED="${SEED:-101}"
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || exit 3
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
STEP=5000
FRESH=2
OUT=outputs/olmo2_probe2_1B_keep${KEEP}f2_dolmino_shallow_seed${SEED}
CK=$OUT/step${STEP}.pt
PROG=logs/a04_shallow_keep${KEEP}_chain.log

note() { printf '[%s] chain(keep%s): %s\n' "$(date '+%m-%d %H:%M:%S')" "$KEEP" "$*" | tee -a "$PROG"; }

case "$KEEP" in 13|14) ;; *) note "FATAL KEEP=$KEEP not in {13,14}"; exit 5 ;; esac

gpu_total() { nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}'; }

note "watching $CK (training ETA ~4.4 h from launch)"
LOOPS=0
while [ $LOOPS -lt 400 ]; do            # 400 x 60 s = 6 h 40 m budget
  LOOPS=$((LOOPS+1))
  sleep 60

  [ -f "$CK" ] || continue

  AGE=$(( $(date +%s) - $(stat -c %Y "$CK") ))
  [ "$AGE" -ge 120 ] || { note "ckpt only ${AGE}s old; waiting"; continue; }

  S1=$(stat -c %s "$CK"); sleep 60; S2=$(stat -c %s "$CK")
  [ "$S1" = "$S2" ] || { note "ckpt still growing ($S1 -> $S2); waiting"; continue; }

  if pgrep -f "train_olmo2_arch_probe2.py.*keep${KEEP}f2_dolmino_shallow_seed${SEED}" >/dev/null 2>&1; then
    note "training worker still alive; waiting for the launcher to stop it"
    continue
  fi

  # GPU idle, sampled THREE times -- one sample is not a state.
  G1=$(gpu_total); sleep 20; G2=$(gpu_total); sleep 20; G3=$(gpu_total)
  if [ "$G1" -gt 8000 ] || [ "$G2" -gt 8000 ] || [ "$G3" -gt 8000 ]; then
    note "GPU still held (${G1}/${G2}/${G3} MiB across 3 samples); waiting"
    continue
  fi

  META=$("$PY" - "$CK" <<'PYEOF' 2>/dev/null || echo bad
import sys, torch
d = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
print(f"{d.get('keep_front_layers')} {d.get('n_fresh_layers')} "
      f"{d.get('num_hidden_layers')} {d.get('step')} {d.get('seed')}")
PYEOF
)
  if [ "$META" = "bad" ]; then note "torch.load failed; waiting"; continue; fi
  set -- $META
  if [ "$1" != "$KEEP" ] || [ "$2" != "$FRESH" ] || [ "$3" != "$((KEEP+FRESH))" ] \
     || [ "$4" != "$STEP" ] || [ "$5" != "$SEED" ]; then
    note "FATAL ckpt mismatch: keep=$1 fresh=$2 depth=$3 step=$4 seed=$5 (expected $KEEP/$FRESH/$((KEEP+FRESH))/$STEP/$SEED). NOT evaluating."
    exit 6
  fi
  note "ALL CONDITIONS MET (age ${AGE}s, size stable $S1, no worker, GPU ${G1}/${G2}/${G3} MiB, ckpt keep=$1 fresh=$2 depth=$3 step=$4 seed=$5)"
  note "launching 4-axis eval"
  ARMS="$KEEP" SEED="$SEED" bash proposal/active/A04-recovery-certification/code/a04_shallow_ladder_eval_driver.sh \
    >> logs/a04_shallow_keep${KEEP}_evaldriver.out 2>&1
  note "eval driver exited rc=$?"
  exit 0
done
note "loop budget exhausted; NOT killing anything; investigate"
exit 9
