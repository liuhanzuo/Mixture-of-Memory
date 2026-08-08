#!/usr/bin/env bash
# A03 Arm 3 (+CPT): resume the 1B keep7+fresh2 heal past step200000 for 20k more
# Dolmino steps, to test whether continued pretraining beyond fluency-repair
# meaningfully improves closed-book factual recall on A03's four certified axes.
#
# Chains behind any running MMLU depth-curve eval on this node (cannot share 8 GPUs).
#
# ---------------------------------------------------------------------------
# LR HORIZON -- read this before changing --max_steps
# ---------------------------------------------------------------------------
# Arm 2's schedule was --lr 2e-5 --min_lr 2e-6 --warmup_steps 150 over
# max_steps=200000 (verified from its log: step40 lr=5.20e-6, step200000 lr=2.00e-6).
# The cosine was therefore FULLY CONSUMED -- Arm 2 ended AT min_lr.
#
# get_lr() re-scales the cosine to whatever --max_steps you pass, so the resume LR
# depends entirely on the new horizon:
#
#     max_steps   lr@200k    frac of 2e-5 peak
#       220000    2.37e-06   0.12x   <- near-frozen; a null result here would be
#                                       confounded by LR, not evidence about CPT
#       250000    3.72e-06   0.19x
#       300000    6.50e-06   0.33x   <- CHOSEN
#       400000    1.10e-05   0.55x
#
# We choose max_steps=300000 so the 20k CPT steps run in the band
# 6.50e-6 -> 4.98e-6 (0.33x -> 0.25x of Arm 2's peak). That is a real late-stage
# CPT learning rate.
#
# IMPORTANT: the trainer has NO --stop_at_step and NO --eval_interval flag (verified
# against its argparse). --max_steps IS both the cosine horizon AND the stop point.
# Since we need horizon=300000 for the LR but only want 20k steps, we launch with
# max_steps=300000 and a WATCHER kills the run once step 220000 is checkpointed.
# --save_every 5000 gives us 205000/210000/215000/220000.
#
# CAVEAT TO REPORT: this is a LATE-COSINE continuation, not a fresh CPT phase at
# peak LR. If Arm 3 shows no factual gain, the honest claim is "no gain from 20k
# additional Dolmino steps at 0.25-0.33x peak LR", NOT "Dolmino CPT saturates".
# A peak-LR CPT arm would need a new warmup and is a separate run.
# ---------------------------------------------------------------------------
set -u

W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
LOG=logs/a03_arm3_cpt20k.log
PROG=logs/a03_arm3_progress.log

CKPT=outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_cpt20k

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- preflight: every asset must exist, on THIS disk ---
for f in "$CKPT" "$DATA" "$BASE/config.json"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 7; }
done
note "preflight OK: ckpt=$(du -h $CKPT | cut -f1) data=$(du -h $DATA | cut -f1)"

# --- wait for any MMLU depth eval on this node to release the GPUs ---
waited=0
while pgrep -f 'eval_olmo2_mmlu_content\.py' >/dev/null 2>&1; do
  if [ "$waited" -eq 0 ]; then note "waiting for eval_olmo2_mmlu_content to finish before claiming 8 GPUs"; fi
  sleep 30; waited=$((waited+30))
  if [ "$waited" -gt 3600 ]; then note "FATAL waited 60min, eval still running -- aborting rather than contending"; exit 8; fi
done
[ "$waited" -gt 0 ] && note "GPUs released after ${waited}s"

# --- refuse to start if anything else holds GPU memory ---
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 8000 ]; then
  note "FATAL ${used}MiB GPU memory still held by another process -- not launching"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | tee -a "$PROG"
  exit 9
fi
note "GPUs clear (${used}MiB held). Launching Arm 3."

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

note "cmd: max_steps=300000 (cosine horizon) but WATCHER stops at step 220000; lr=2e-5 min_lr=2e-6 -> resume LR ~6.5e-6 = 0.33x peak"
"$PY" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_olmo2_arch_probe2.py \
    --model_path "$BASE" \
    --resume_from "$CKPT" \
    --keep_front_layers 7 --n_fresh_layers 2 \
    --data_path "$DATA" \
    --output_dir "$OUT" \
    --max_steps 300000 \
    --lr 2e-5 --min_lr 2e-6 \
    --lr_inherited 2e-5 --min_lr_inherited 2e-6 \
    --seq_len 2048 --batch_size 8 \
    --warmup_steps 150 \
    --save_every 5000 \
    --gradient_checkpointing 1 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
note "launched torchrun pid=$TRAIN_PID"

# --- watcher: stop once step220000.pt is on disk (max_steps is the only stop flag,
#     and we need horizon 300000 for the LR band, so we enforce the stop here) ---
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if [ -f "$OUT/step220000.pt" ]; then
    note "step220000.pt present -> stopping run (20k CPT steps complete)"
    # kill the torchrun launcher, then any surviving workers by their script name
    kill -TERM "$TRAIN_PID" 2>/dev/null
    sleep 20
    for p in $(pgrep -f "train_olmo2_arch_probe2.py.*cpt20k" 2>/dev/null); do
      kill -9 "$p" 2>/dev/null
    done
    break
  fi
  sleep 60
done
wait "$TRAIN_PID" 2>/dev/null
rc=$?
note "training exited rc=$rc"
note "ckpts on disk: $(ls $OUT/step*.pt 2>/dev/null | tr '\n' ' ')"
note "next: eval step205000/210000/220000 on the four A03 axes vs Arm 2 (step200000)"
exit 0
