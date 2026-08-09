#!/usr/bin/env bash
# A03 Arm 6 (mid-low-LR CPT band): resume Arm 2 step200000 for 20k more Dolmino
# steps in the LR band [0.50x, 0.43x] peak -- above Arm 3's tail band but well
# below Arm 4's peak-anchored band. Tests whether Arm 3's +0.48pp SIG triviaqa
# em is a general low-LR phenomenon or specific to the exact tail schedule.
#
# ---------------------------------------------------------------------------
# WHY ARM 6 EXISTS
# ---------------------------------------------------------------------------
# Arm 4 (peak-anchored, [1.00x, 0.56x] peak) at step220000 SIGN-FLIPS triviaqa
# em vs Arm 3 (late-cosine tail, [0.32x, 0.25x] peak). Two natural readings of
# Arm 3's +0.48 SIG:
#   (i)  low-LR CPT works, peak-anchored CPT destroys -> a "low LR regime"
#        finding.  Arm 3 is a real recipe.
#   (ii) Arm 3's exact schedule is a fluke that happens to produce +0.48 SIG.
#        Arm 6 (just above Arm 3's band) should also sign-flip / null out.
# Arm 6 discriminates (i) vs (ii).
#
# ---------------------------------------------------------------------------
# LR HORIZON -- read this before changing --max_steps / --warmup_steps
# ---------------------------------------------------------------------------
# get_lr uses ABSOLUTE step and ABSOLUTE max_steps
# (train_semantic_bottleneck_1b.py:76). Resume keeps the step counter
# (train_olmo2_arch_probe2.py:988). Effective LR at resume step=200000 is
# entirely a function of (warmup_steps, max_steps).
#
# Config: warmup_steps=150 (matches Arm 3), max_steps=373000 (extends Arm 3's
# horizon from 300000 to 373000, giving less-consumed cosine at 200k-220k).
#
# Verified by scripts/train_semantic_bottleneck_1b.py:76 get_lr:
#   step=200000 -> lr = 9.98e-06 = 0.499x peak
#   step=205000 -> lr = 9.61e-06 = 0.480x peak
#   step=210000 -> lr = 9.24e-06 = 0.462x peak
#   step=215000 -> lr = 8.87e-06 = 0.443x peak
#   step=220000 -> lr = 8.50e-06 = 0.425x peak
# Band [0.50x, 0.43x] peak across the 20k window.
# ---------------------------------------------------------------------------

set -u

W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
LOG=logs/a03_arm6_lowerband20k.log
PROG=logs/a03_arm6_progress.log

CKPT=outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm6_lowerband20k

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- preflight ---
for f in "$CKPT" "$DATA" "$BASE/config.json"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 7; }
done
note "preflight OK: ckpt=$(du -h $CKPT | cut -f1) data=$(du -h $DATA | cut -f1)"

# --- refuse to start if GPUs are held ---
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 8000 ]; then
  note "FATAL ${used}MiB GPU memory held by another process; not launching"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | tee -a "$PROG"
  exit 9
fi
note "GPUs clear (${used}MiB held); launching Arm 6"

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

note "cmd: max_steps=373000 warmup_steps=150 -> lr[step200000]=9.98e-6 (0.50x), lr[step220000]=8.50e-6 (0.43x)"
note "WATCHER will stop the run when step220000.pt exists (matched 20k window with Arm 3 and Arm 4)"

"$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --model_path "$BASE" \
    --resume_from "$CKPT" \
    --keep_front_layers 7 --n_fresh_layers 2 \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps 373000 \
    --lr 2e-5 --min_lr 2e-6 \
    --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
    --seq_len 2048 --batch_size 8 \
    --warmup_steps 150 \
    --save_every 5000 \
    --gradient_checkpointing 1 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
note "launched torchrun pid=$TRAIN_PID"

# --- watcher: stop at step220000.pt (matched 20k window w/ Arm 3 & Arm 4)
# v3 guard: size within 64 KiB of a known-good sibling ckpt, AND size stable
# across two probes 60s apart. SIGTERM then poll up to 10 min.
# See scripts/_run_a03_arm4_peaklr.sh for the corruption / drift incident this
# guard is designed for; commit 2d4f679 for the tolerance-vs-exact rationale.
REF_SIZE=""
for ref in "$OUT"/step205000.pt "$OUT"/step210000.pt "$OUT"/step215000.pt; do
  [ -f "$ref" ] && REF_SIZE=$(stat -c %s "$ref") && break
done
note "watcher: reference ckpt size = ${REF_SIZE:-unknown} bytes"

settled_size() {
  local f="$1" s1 s2 diff
  [ -f "$f" ] || return 1
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  sleep 60
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  [ "$s1" = "$s2" ] || { note "watcher: $f still growing ($s1 -> $s2); waiting"; return 1; }
  if [ -n "$REF_SIZE" ]; then
    diff=$(( s2 > REF_SIZE ? s2 - REF_SIZE : REF_SIZE - s2 ))
    if [ "$diff" -gt 65536 ]; then
      note "watcher: $f settled at $s2 but reference is $REF_SIZE (diff ${diff} B > 64 KiB tolerance) -- NOT stopping"
      return 1
    fi
  fi
  echo "$s2"
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if sz=$(settled_size "$OUT/step220000.pt"); then
    note "step220000.pt fully written ($sz bytes) -> stopping run (20k mid-low-LR CPT complete)"
    kill -TERM "$TRAIN_PID" 2>/dev/null
    for _ in $(seq 1 60); do
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      sleep 10
    done
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      note "watcher: pid $TRAIN_PID still alive after 10 min of SIGTERM -- escalating"
      for p in $(pgrep -f "train_olmo2_arch_probe2.py.*arm6_lowerband" 2>/dev/null); do
        kill -9 "$p" 2>/dev/null
      done
    fi
    break
  fi
  sleep 60
done
wait "$TRAIN_PID" 2>/dev/null
rc=$?
note "training exited rc=$rc"
note "ckpts on disk: $(ls -la $OUT/step*.pt 2>/dev/null | awk '{print $9"="$5"B"}' | tr '\n' ' ')"
note "next: eval step205000/210000/215000/220000 vs Arm 2 baseline; compare vs Arm 3 (+0.48 SIG) and Arm 4 (-0.93 SIG)"
exit 0
