#!/usr/bin/env bash
# Paper C P-C1 launcher: freeze-graft / from-scratch / full-FT arms on ONE H20
# node, driving the UNMODIFIED scripts/train_olmo2_arch_probe2.py on SQuAD SFT
# packed chunks. A2 (LoRA) has its own launcher (run_paperC_pc1_lora.sh) because
# it needs the LoRA trainer fork.
#
# Arms (ARM env):
#   A4  freeze_front keep14+fresh2  -> --freeze_front (HERO; trains fresh2+norm+lm_head)
#   A3  from_scratch 16L            -> --from_scratch --keep_front_layers 14 --n_fresh_layers 2
#   A1  full-FT 32L                 -> keep32+fresh0, single-LR (lr==lr_inherited), all layers
#
# eff_bs = BS * GA * nGPU is pinned to 128 for every arm (comparability).
# Node: .104 8xH20. Python: conda torch-base (.venv is BROKEN on .104).
#
# Usage:
#   ARM=A4 GPUS=0,1,2,3,4,5,6,7 PORT=29551 bash scripts/run_paperC_pc1.sh
#   ARM=A3 GPUS=0,1             PORT=29552 bash scripts/run_paperC_pc1.sh
#   ARM=A1 GPUS=2,3,4,5 OPT=bnb8bit PORT=29553 bash scripts/run_paperC_pc1.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

ARM="${ARM:?set ARM=A4|A3|A1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29551}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
MAX_STEPS="${MAX_STEPS:-2000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
OPT="${OPT:-adamw}"          # adamw | bnb8bit  (bnb8bit -> --optimizer bnb_adamw8bit)
EFF_BS="${EFF_BS:-128}"

nGPU=$(awk -F, '{print NF}' <<< "$GPUS")

case "$ARM" in
  A4)  # HERO freeze-graft keep14+fresh2
    KEEP=14; FRESH=2; EXTRA="--freeze_front"; LR=1e-4; LR_INH=2e-5 ;;
  A3)  # from-scratch depth-matched 16L
    KEEP=14; FRESH=2; EXTRA="--from_scratch"; LR=3e-4; LR_INH=3e-4 ;;
  A1)  # full-FT original 32L (keep32+fresh0 -> full transplant, all layers), single-LR
    KEEP=32; FRESH=0; EXTRA=""; LR=1e-5; LR_INH=1e-5 ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

# pin per-GPU BS so BS*GA*nGPU == EFF_BS. Default GA to hit EFF_BS with BS as
# large as the arm can afford; caller can override BS/GA.
BS="${BS:-}"; GA="${GA:-}"
if [ -z "$BS" ] || [ -z "$GA" ]; then
  # default: BS=4 (safe for 7B fp32), GA = EFF_BS/(BS*nGPU)
  BS="${BS:-4}"
  GA=$(( EFF_BS / (BS * nGPU) ))
  [ "$GA" -lt 1 ] && GA=1
fi
REAL_EFF=$(( BS * GA * nGPU ))

OUT_DIR="$PROJECT_ROOT/outputs/paperC_pc1_squad_${ARM}"
LOG_FILE="$PROJECT_ROOT/logs/paperC_pc1_squad_${ARM}.log"
mkdir -p "$OUT_DIR" "$PROJECT_ROOT/logs"

OPT_FLAG=""
[ "$OPT" = "bnb8bit" ] && OPT_FLAG="--optimizer bnb_adamw8bit"

echo "[paperC_pc1] ARM=$ARM KEEP=$KEEP FRESH=$FRESH GPUS=$GPUS nGPU=$nGPU BS=$BS GA=$GA eff_bs=$REAL_EFF (target $EFF_BS) OPT=$OPT -> $OUT_DIR"
if [ "$REAL_EFF" -ne "$EFF_BS" ]; then
  echo "[paperC_pc1] WARNING eff_bs=$REAL_EFF != target $EFF_BS (adjust BS/GA)"
fi
: > "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

setsid nohup "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes 1 --nproc_per_node "$nGPU" --rdzv_backend c10d --rdzv_endpoint "localhost:$PORT" \
  scripts/train_olmo2_arch_probe2.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --keep_front_layers "$KEEP" \
    --n_fresh_layers "$FRESH" \
    --batch_size "$BS" \
    --grad_accumulation_steps "$GA" \
    --seq_len "$SEQ_LEN" \
    --lr "$LR" \
    --lr_inherited "$LR_INH" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps 150 \
    --save_every 500 \
    --log_every 10 \
    --seed "$SEED" \
    --gradient_checkpointing 1 \
    $EXTRA $OPT_FLAG \
  >>"$LOG_FILE" 2>&1 &

echo "[paperC_pc1] launched pid=$! (setsid) ; tail -f $LOG_FILE"
