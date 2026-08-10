#!/usr/bin/env bash
# A03 data-order replication: re-run Arm 3's EXACT config with the ce5c298
# `seed=args.seed` fix now on disk, varying only `--seed` across parallel nodes.
# Tests whether Arm 3's +0.48 SIG triviaqa em @step220000 (replicated by Arm 6
# at +0.50) survives a change in data order, given that all previous 3 arms
# shared sampler seed 0 (Arm3-Arm6 loss corr = 0.99982; trajectory Pearson
# r=+0.96-0.999).
#
# Config identical to _run_a03_arm3_cpt.sh except:
#   - OUT differs by seed
#   - --seed passed explicitly (with fixed trainer, this now propagates to
#     DistributedSampler(seed=) at scripts/train_olmo2_arch_probe2.py:869)
#   - PROG/LOG names carry seed
#
# Usage: SEED=43 bash scripts/_run_a03_dataorder_repl.sh
set -u

SEED="${SEED:?SEED must be set (integer, e.g. 43)}"
W="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$W" || { echo "FATAL: cannot cd $W"; exit 3; }
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

LOG=logs/a03_dataorder_seed${SEED}.log
PROG=logs/a03_dataorder_seed${SEED}_progress.log

CKPT=outputs/olmo2_probe2_1B_keep7fresh2_16card/step200000.pt
DATA=data/dolmino_now15b.npy
BASE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/models/OLMo-2-0425-1B
OUT=outputs/olmo2_probe2_1B_keep7f2_dolmino_dataorder_seed${SEED}

note() { printf '[%s] %s\n' "$(date '+%m-%d %H:%M:%S')" "$*" | tee -a "$PROG"; }

# --- preflight: trainer must be POST-ce5c298 (i.e. have seed= on line 869) ---
FIX_LINE=$(grep -n 'DistributedSampler(ds, shuffle=True, seed=args.seed)' scripts/train_olmo2_arch_probe2.py | head -1)
[ -z "$FIX_LINE" ] && { note "FATAL trainer missing ce5c298 fix (no 'seed=args.seed' on DistributedSampler line)"; exit 4; }
note "trainer post-ce5c298 OK: $FIX_LINE"

for f in "$CKPT" "$DATA" "$BASE/config.json"; do
  [ -e "$f" ] || { note "FATAL missing asset: $f"; exit 7; }
done
note "preflight OK: SEED=$SEED ckpt=$(du -h $CKPT | cut -f1) data=$(du -h $DATA | cut -f1)"

used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | awk '{s+=$1} END {print s+0}')
if [ "$used" -gt 8000 ]; then
  note "FATAL ${used}MiB GPU memory still held by another process -- not launching"
  nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | tee -a "$PROG"
  exit 9
fi
note "GPUs clear (${used}MiB held). Launching seed=$SEED replication."

export OMP_NUM_THREADS=4
export WANDB_MODE=offline

note "cmd: same as Arm 3 (max_steps=300000, warmup=150, lr 2e-5, min_lr 2e-6) EXCEPT --seed $SEED. Watcher stops at step220000."
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
    --seed "$SEED" \
    --gradient_checkpointing 1 \
  > "$LOG" 2>&1 &
TRAIN_PID=$!
note "launched torchrun pid=$TRAIN_PID seed=$SEED"

# --- watcher: stop once step220000.pt lands ---
while kill -0 "$TRAIN_PID" 2>/dev/null; do
  if [ -f "$OUT/step220000.pt" ]; then
    note "step220000.pt present -> stopping run (20k steps complete)"
    kill -TERM "$TRAIN_PID" 2>/dev/null
    sleep 20
    for p in $(pgrep -f "train_olmo2_arch_probe2.py.*dataorder_seed${SEED}" 2>/dev/null); do
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
note "next: eval step220000 on the four A03 axes vs Arm 3 step220000 (+0.48 SIG triviaqa em)"
exit 0
