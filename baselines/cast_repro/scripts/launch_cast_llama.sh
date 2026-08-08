#!/usr/bin/env bash
# Launch the CAST reproduction on .21 (8x L20A, 183 GiB/card, wzc1 disk).
#
# The 8-GPU DDP smoke is what has actually been validated so far; the full
# 7500-step run is gated on a data decision (see ../README.md "Data").
#
# Usage:
#   bash scripts/launch_cast_llama.sh smoke          # ~50 steps, proves e2e
#   bash scripts/launch_cast_llama.sh full           # 7500 steps -- read README first
#   DATA=data/dolmino-mix-1124-llama2 bash scripts/launch_cast_llama.sh full
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
TORCHRUN="$(dirname "$PY")/torchrun"
DATA="${DATA:-data/c4_llama}"          # FALLBACK by default; see README
DATA_DTYPE="${DATA_DTYPE:-uint16}"
NPROC="${NPROC:-8}"
CODE="$ROOT/baselines/cast_repro"

cd "$CODE"

case "$MODE" in
  smoke)
    STEPS=50; OUT="outputs/cast_repro_smoke"; EXTRA="--save-every 0 --diag-every 25 --log-every 5"
    ;;
  full)
    STEPS=7500; OUT="outputs/cast_repro_ddp"; EXTRA="--save-every 500 --diag-every 250 --log-every 10"
    ;;
  *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;;
esac

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/cast_repro_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== pre-flight ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
"$PY" tools/decay_budget.py --steps "$STEPS"
echo

echo "mode=$MODE steps=$STEPS data=$DATA out=$OUT log=$LOG"
setsid nohup "$TORCHRUN" --nproc_per_node "$NPROC" cast/train_cast_llama.py \
  --project-root "$ROOT" \
  --data "$DATA" --data-dtype "$DATA_DTYPE" \
  --out "$OUT" \
  --max-steps "$STEPS" \
  --lr 2e-5 --l1-decay 4e-7 --global-batch 256 --seq-len 4096 \
  --mask-period 10 --scale-groups 2 --eta 0.3333333333333333 \
  --kl-temperature 1.0 --min-lr 2e-6 --warmup 375 \
  --micro-batch 1 \
  $EXTRA \
  > "$LOG" 2>&1 &

echo "launched pid=$! ; tail -f $LOG"
