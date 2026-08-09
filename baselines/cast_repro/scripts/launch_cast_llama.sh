#!/usr/bin/env bash
# Launch the CAST reproduction on .21 (8x L20A, 183 GiB/card, wzc1 disk).
#
# Defaults are PARALLEL=zero2 (DDP + ZeroRedundancyOptimizer(AdamS); plain ddp
# OOMs an L20A) and LR_SCHEDULE=constant (paper-literal, Appendix B).
# NEVER switch this to FSDP -- it flattens the Parameter and silently breaks
# weight<->mask alignment. See cast/train_cast_llama.py's docstring and
# tools/fsdp_misalignment_demo.py.
#
# DATA defaults to Dolmino-Mix-1124 (Sec. VI-A), NOT C4. C4 is the corpus the
# audit flagged as failure #2 -- it is what the *broken* run used and it is the
# OPT/GPT-2 corpus, not LLaMA's. DATA_DTYPE defaults to `auto`, which reads
# <data>/metadata.json: the dolmino tokenizer wrote **uint32** (vocab is 32000 so
# uint16 would have fit, but the writer did not narrow it). Passing uint16 by hand
# would reinterpret each 4-byte token as two 2-byte ones -- silently doubling the
# stream and injecting zeros, with no error anywhere.
#
# Usage:
#   bash scripts/launch_cast_llama.sh smoke          # ~50 steps, proves e2e
#   bash scripts/launch_cast_llama.sh full           # 7500 steps -- read README first
#   RESUME=auto bash scripts/launch_cast_llama.sh full   # continue after a crash
#   DATA=data/c4_llama DATA_DTYPE=uint16 bash scripts/launch_cast_llama.sh smoke  # fallback
set -euo pipefail

MODE="${1:-smoke}"
ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code}"
PY="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
TORCHRUN="$(dirname "$PY")/torchrun"
# PRIMARY (paper, Sec. VI-A). 77,721,665,859 tokens, uint32 per metadata.json.
# NOTE the path is under Mixture-of-Memory/, not directly under PROJECT_ROOT:
# $ROOT/data/ is a different (older) data tree and does NOT contain dolmino.
DATA="${DATA:-Mixture-of-Memory/data/dolmino-mix-1124-llama2}"
DATA_DTYPE="${DATA_DTYPE:-auto}"       # auto => read metadata.json, never guess
NPROC="${NPROC:-8}"
# ddp = no sharding, ~131.8 GB/rank static -> OOMs an L20A once activations land.
# zero2 = DDP + ZeroRedundancyOptimizer(AdamS), Adam state sharded whole-tensor.
PARALLEL="${PARALLEL:-zero2}"
# constant = paper-literal (Appendix B "consistent learning rate", Table XI 2e-5).
LR_SCHEDULE="${LR_SCHEDULE:-constant}"
RESUME="${RESUME:-}"
CODE="$ROOT/baselines/cast_repro"

cd "$CODE"

case "$MODE" in
  smoke)
    STEPS=50; OUT="outputs/cast_repro_smoke"; EXTRA="--save-every 0 --diag-every 25 --log-every 5"
    ;;
  full)
    # save-every 250 => ~14 min of lost work worst case at the measured ~14 s/step,
    # 30 checkpoints over the run. keep-last 2 caps disk at ~174 GB (each ckpt is
    # model fp32 27 GB + masks 6.5 GB + Adam moments 54 GB ~= 87 GB).
    STEPS=7500; OUT="outputs/cast_repro_${PARALLEL}"
    EXTRA="--save-every 250 --keep-last 2 --diag-every 250 --log-every 10"
    ;;
  *) echo "usage: $0 {smoke|full}" >&2; exit 2 ;;
esac

if [ -n "$RESUME" ]; then
  EXTRA="$EXTRA --resume $RESUME"
fi

mkdir -p "$ROOT/logs"
LOG="$ROOT/logs/cast_repro_${MODE}_$(date +%Y%m%d_%H%M%S).log"

echo "=== pre-flight ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
# Fail fast if the data is missing or its dtype cannot be established: feeding
# uint32 tokens as uint16 corrupts the stream *silently*, so this must never be
# left to a default.
if [ ! -f "$ROOT/$DATA/train.bin" ]; then
  echo "FATAL: $ROOT/$DATA/train.bin not found" >&2; exit 3
fi
if [ "$DATA_DTYPE" = "auto" ] && [ ! -f "$ROOT/$DATA/metadata.json" ]; then
  echo "FATAL: DATA_DTYPE=auto but $ROOT/$DATA/metadata.json is missing -- refusing to" >&2
  echo "       guess the token width. Pass DATA_DTYPE explicitly if you are sure." >&2
  exit 3
fi
"$PY" - "$ROOT/$DATA" <<'EOF'
import json, os, sys
d = sys.argv[1]
m = json.load(open(os.path.join(d, "metadata.json")))
n = os.path.getsize(os.path.join(d, "train.bin"))
w = {"uint16": 2, "uint32": 4}[m["dtype"]]
print(f"data: {d}\n  dataset={m.get('dataset')} tokenizer={os.path.basename(str(m.get('tokenizer')))}")
print(f"  dtype={m['dtype']} ({w} B/token)  train.bin={n/2**30:.1f} GiB -> {n//w:,} tokens")
assert n % w == 0, "train.bin size is not a multiple of the token width"
assert n // w == m["total_tokens"], f"size implies {n//w:,} tokens, metadata says {m['total_tokens']:,}"
print(f"  OK: byte size and metadata total_tokens agree ({m['total_tokens']:,})")
EOF
# Under a constant LR the usable decay distance is sum_t lr*alpha_t, i.e. the
# budget tool must be told min-lr == lr and no warmup (11.19x headroom vs the
# cosine schedule's 4.32x).
if [ "$LR_SCHEDULE" = "constant" ]; then
  "$PY" tools/decay_budget.py --steps "$STEPS" --lr 2e-5 --min-lr 2e-5 --warmup 0
else
  "$PY" tools/decay_budget.py --steps "$STEPS" --lr 2e-5 --min-lr 2e-6 --warmup 375
fi
echo

echo "mode=$MODE steps=$STEPS parallel=$PARALLEL lr_schedule=$LR_SCHEDULE data=$DATA dtype=$DATA_DTYPE out=$OUT log=$LOG"
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
