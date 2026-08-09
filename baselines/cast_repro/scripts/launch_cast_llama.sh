#!/usr/bin/env bash
# Launch the CAST reproduction on .21 (8x L20A, 183 GiB/card, wzc1 disk).
#
# Defaults are PARALLEL=zero2 (DDP + ZeroRedundancyOptimizer(AdamS); plain ddp
# OOMs an L20A) and LR_SCHEDULE=constant (paper-literal, Appendix B).
# NEVER switch this to FSDP -- it flattens the Parameter and silently breaks
# weight<->mask alignment. See cast/train_cast_llama.py's docstring and
# tools/fsdp_misalignment_demo.py.
#
# Usage:
#   bash scripts/launch_cast_llama.sh smoke          # ~50 steps, proves e2e
#   bash scripts/launch_cast_llama.sh full           # 7500 steps -- read README first
#   DATA=data/dolmino-mix-1124-llama2 bash scripts/launch_cast_llama.sh full
#   PARALLEL=ddp LR_SCHEDULE=cosine bash scripts/launch_cast_llama.sh smoke
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
TORCHRUN="$(dirname "$PY")/torchrun"
DATA="${DATA:-data/c4_llama}"          # FALLBACK by default; see README
DATA_DTYPE="${DATA_DTYPE:-uint16}"
NPROC="${NPROC:-8}"
# ddp = no sharding, ~131.8 GB/rank static -> OOMs an L20A once activations land.
# zero2 = DDP + ZeroRedundancyOptimizer(AdamS), Adam state sharded whole-tensor.
PARALLEL="${PARALLEL:-zero2}"
# constant = paper-literal (Appendix B "consistent learning rate", Table XI 2e-5).
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
CODE="$ROOT/baselines/cast_repro"

cd "$CODE"

case "$MODE" in
  smoke)
    STEPS=50; OUT="outputs/cast_repro_smoke"; EXTRA="--save-every 0 --diag-every 25 --log-every 5"
    ;;
  full)
    STEPS=7500; OUT="outputs/cast_repro_${PARALLEL}"; EXTRA="--save-every 500 --diag-every 250 --log-every 10"
    ;;
  *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/cast_repro_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== pre-flight ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# Under a constant LR the usable decay distance is sum_t lr*alpha_t, i.e. the
# budget tool must be told min-lr == lr and no warmup (11.19x headroom vs the
# cosine schedule's 4.32x).
if [ "$LR_SCHEDULE" = "constant" ]; then
  "$PY" tools/decay_budget.py --steps "$STEPS" --lr 2e-5 --min-lr 2e-5 --warmup 0
else
  "$PY" tools/decay_budget.py --steps "$STEPS" --lr 2e-5 --min-lr 2e-6 --warmup 375
fi
echo

echo "mode=$MODE steps=$STEPS parallel=$PARALLEL lr_schedule=$LR_SCHEDULE data=$DATA out=$OUT log=$LOG"
setsid nohup "$TORCHRUN" --nproc_per_node "$NPROC" cast/train_cast_llama.py \
  --project-root "$ROOT" \
  --data "$DATA" --data-dtype "$DATA_DTYPE" \
  --out "$OUT" \
  --max-steps "$STEPS" \
  --parallel "$PARALLEL" \
  --lr 2e-5 --l1-decay 4e-7 --global-batch 256 --seq-len 4096 \
  --mask-period 10 --scale-groups 2 --eta 0.3333333333333333 \
  --kl-temperature 1.0 --lr-schedule "$LR_SCHEDULE" --min-lr 2e-6 --warmup 375 \
  --micro-batch 1 \
  --gradient-checkpointing \
  $EXTRA \
  > "$LOG" 2>&1 &

echo "launched pid=$! ; tail -f $LOG"
