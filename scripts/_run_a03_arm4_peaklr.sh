#!/usr/bin/env bash
# A03 Arm 4 (+CPT peak-LR): resume Arm 2 step200000 for 20k more Dolmino steps
# at PEAK LR, to disambiguate the step220000 finding from Arm 3.
#
# ---------------------------------------------------------------------------
# WHY THIS ARM EXISTS
# ---------------------------------------------------------------------------
# Arm 3 (20k CPT at 0.28-0.33x peak LR, late-cosine tail) produced a coherent
# +0.48pp SIG triviaqa gain (3 metrics moving together) at step220000, with
# earlier ckpts (step205/210/215k) showing only noise-floor wobbles. That is
# a real but weak signal. Two readings compatible with Arm 3's data:
#
#   (A) 20k steps was near the CPT budget and the gain would keep growing;
#       peak-LR CPT would produce a larger gain in the same 20k window.
#   (B) 20k was already saturating; peak-LR CPT would not do better.
#
# Arm 4 distinguishes (A) from (B).
#
# ---------------------------------------------------------------------------
# LR HORIZON -- read this before changing --max_steps / --warmup_steps
# ---------------------------------------------------------------------------
# get_lr uses ABSOLUTE step and ABSOLUTE max_steps (train_semantic_bottleneck_1b.py:76).
# Resume keeps the step counter (train_olmo2_arch_probe2.py:988). So effective LR
# at resume step=200000 is entirely a function of (warmup_steps, max_steps).
#
# ARM4_DESIGN.md Config B: warmup_steps=200500, max_steps=240000
#   -> at step 200000 (resume): lr = base * 200000/200500 = 1.995e-5 (~peak)
#   -> at step 200500 (warmup done): lr = peak = 2e-5
#   -> at step 220000: lr = min_lr + 0.5*(peak-min)*(1+cos(pi*(220000-200500)/(240000-200500)))
#                        = 2e-6 + 0.5*(1.8e-5)*(1+cos(pi*0.494))
#                        ~ 1.12e-5 = 0.56x peak
# So Arm 4 holds LR in the band [peak, 0.56x peak] across 20k steps -- vastly
# above Arm 3's [0.33x, 0.28x peak] band.
#
# NOTE: this is a warmup-hack. Adam moments carried over from Arm 2 were adapted
# to a min_lr trajectory ending at 2e-6, so the first ~500 steps under peak LR
# are moment-mismatched. Interpret step205k onward as "moment-stabilised".
# ---------------------------------------------------------------------------

set -u

W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
LOG=logs/a03_arm4_peaklr20k.log
PROG=logs/a03_arm4_progress.log

CKPT=outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_arm4_peaklr20k

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
note "GPUs clear (${used}MiB held); launching Arm 4"

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

note "cmd: max_steps=240000 warmup_steps=200500 -> lr[step200000]=1.995e-5 (~peak), lr[step220000]=1.12e-5 (0.56x peak)"
note "WATCHER will stop the run when step220000.pt exists (matched 20k window with Arm 3)"

"$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --model_path "$BASE" \
    --resume_from "$CKPT" \
    --keep_front_layers 7 --n_fresh_layers 2 \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps 240000 \
    --lr 2e-5 --min_lr 2e-6 \
    --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
    --seq_len 2048 --batch_size 8 \
    --warmup_steps 200500 \
    --save_every 5000 \
    --gradient_checkpointing 1 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
note "launched torchrun pid=$TRAIN_PID"

# --- watcher: stop at step220000.pt (matched 20k window w/ Arm 3) ---
#
# ⚠️ 2026-08-10: the v1 watcher used a bare `[ -f "$OUT/step220000.pt" ]` and it
# CORRUPTED the checkpoint. torch.save creates the file immediately and then
# streams ~12.18 GB into it, so `-f` is true from the first byte. Observed:
# step line 00:47:45 -> watcher fired 00:47:53 -> SIGTERM, sleep 20, kill -9 at
# 00:48:13. A 12.18 GB save cannot complete in 28 s on this filesystem, so the
# file was truncated at 5,956,287,104 B (49%) and torch.load fails with
# "PytorchStreamReader ... failed finding central directory".
#
# This is the SAME race that was already fixed in the .82 eval-side watcher; it
# simply also existed here on the trainer-stop side. Both need the guard.
#
# Guard: require (a) the size to match a known-good sibling checkpoint, and
# (b) the size to be unchanged across two probes 60 s apart, before signalling.
# Then SIGTERM and wait for the process to leave on its own -- never kill -9 a
# process that may still be flushing a checkpoint.
REF_SIZE=""
for ref in "$OUT"/step205000.pt "$OUT"/step210000.pt "$OUT"/step215000.pt; do
  [ -f "$ref" ] && REF_SIZE=$(stat -c %s "$ref") && break
done
note "watcher: reference ckpt size = ${REF_SIZE:-unknown} bytes"

settled_size() {  # echo size if $1 is fully written, else nothing
  local f="$1" s1 s2
  [ -f "$f" ] || return 1
  s1=$(stat -c %s "$f" 2>/dev/null) || return 1
  sleep 60
  s2=$(stat -c %s "$f" 2>/dev/null) || return 1
  [ "$s1" = "$s2" ] || { note "watcher: $f still growing ($s1 -> $s2); waiting"; return 1; }
  if [ -n "$REF_SIZE" ] && [ "$s2" != "$REF_SIZE" ]; then
    note "watcher: $f settled at $s2 but reference is $REF_SIZE -- NOT stopping"
    return 1
  fi
  echo "$s2"
}

while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if sz=$(settled_size "$OUT/step220000.pt"); then
    note "step220000.pt fully written ($sz bytes) -> stopping run (20k peak-LR CPT complete)"
    kill -TERM "$TRAIN_PID" 2>/dev/null
    # Wait for a clean exit rather than kill -9 on a timer. Cap at 10 min so a
    # genuinely hung process still gets reaped, but give a real save room.
    for _ in $(seq 1 60); do
      kill -0 "$TRAIN_PID" 2>/dev/null || break
      sleep 10
    done
    if kill -0 "$TRAIN_PID" 2>/dev/null; then
      note "watcher: pid $TRAIN_PID still alive after 10 min of SIGTERM -- escalating"
      for p in $(pgrep -f "train_olmo2_arch_probe2.py.*arm4_peaklr" 2>/dev/null); do
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
note "next: eval step205000/210000/215000/220000 vs Arm 2 baseline; compare to Arm 3 dose-response"
exit 0
