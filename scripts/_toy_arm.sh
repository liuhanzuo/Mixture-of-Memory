#!/bin/bash
# Launch one toy_memory_bootstrap arm on a specific local GPU. One arm per call.
# Usage: GPU=0 RUN=name WEIGHT=0.1 SEED=42 [STEPS=800] [MODE=slot_query] [TEMP=40] bash scripts/_toy_arm.sh
set -euo pipefail
cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export WANDB_API_KEY="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"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
GPU="${GPU:?}"; RUN="${RUN:?}"; WEIGHT="${WEIGHT:?}"; SEED="${SEED:-42}"
STEPS="${STEPS:-800}"; MODE="${MODE:-slot_query}"; TEMP="${TEMP:-40}"
PYBIN="${PYTHON_BIN:-.venv/bin/python}"
FORCE_GATE_ARGS=""
if [ -n "${FORCE_GATE_ALPHA:-}" ]; then
  FORCE_GATE_ARGS="--force_gate_alpha $FORCE_GATE_ALPHA --force_gate_steps ${FORCE_GATE_STEPS:-400}"
fi
CUDA_VISIBLE_DEVICES=$GPU setsid bash -c "$PYBIN -u scripts/toy_memory_bootstrap.py \
  --total_steps $STEPS --routing_pool_mode $MODE --selector_temperature $TEMP \
  --l_recon_weight $WEIGHT $FORCE_GATE_ARGS \
  --seed $SEED --wandb_run_name $RUN --output_dir outputs/$RUN" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN on GPU$GPU (weight=$WEIGHT seed=$SEED steps=$STEPS mode=$MODE temp=$TEMP) pid=$!"
