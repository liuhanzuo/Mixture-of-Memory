#!/usr/bin/env bash
# CAST repro training launcher — bypasses launch_cast_llama.sh's layout assumption.
# Runs on .21 (wzc1, L20A 8x). Assumes CWD = repo root Mixture-of-Memory.
set -u

W="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
CAST=$W/baselines/cast_repro
PY=${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}
TR=${TORCHRUN_BIN:-/opt/conda/envs/torch-base/bin/torchrun}
DATA=$W/data/dolmino-mix-1124-llama2
PROJ=/apdcephfs_wzc1/share_304376610/pighzliu_code   # so root/models/Llama--Llama2-7b resolves
LOG=$W/logs/cast_repro_ddp_direct_$(date +%m%d_%H%M%S).log

[ -f "$DATA/metadata.json" ] || { echo "FATAL: $DATA missing metadata.json"; exit 3; }
[ -d "$PROJ/models/Llama--Llama2-7b" ] || { echo "FATAL: model not at $PROJ/models/Llama--Llama2-7b"; exit 4; }

cd "$CAST" || { echo "FATAL: cannot cd $CAST"; exit 5; }
echo "LOG=$LOG"
"$TR" --nproc_per_node 8 cast/train_cast_llama.py \
  --project-root "$PROJ" \
  --data "$DATA" --data-dtype uint16 \
  --out outputs/cast_repro_ddp \
  --max-steps 7500 \
  --lr 2e-5 --l1-decay 4e-7 --global-batch 256 --seq-len 4096 \
  --mask-period 10 --scale-groups 2 --eta 0.3333333333333333 \
  --kl-temperature 1.0 --min-lr 2e-6 --warmup 375 \
  --micro-batch 1 \
  --save-every 500 --diag-every 250 --log-every 10 \
  > "$LOG" 2>&1
