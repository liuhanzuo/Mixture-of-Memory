#!/usr/bin/env bash
# Paper C P-C1 arm A2 (param-matched LoRA) launcher: full 32L OLMo-2 + LoRA on
# SQuAD SFT packed chunks, single H20 node. eff_bs pinned to 128.
#
# Usage:
#   GPUS=0,1 PORT=29554 R=160 bash scripts/run_paperC_pc1_lora.sh   # param-matched
#   GPUS=2,3 PORT=29555 R=64  bash scripts/run_paperC_pc1_lora.sh   # reference
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
PORT="${PORT:-29554}"
R="${R:-160}"
DATA_PATH="${DATA_PATH:-$PROJECT_ROOT/data/squad_sft_olmo2_2048_train.npy}"
MODEL_PATH="${MODEL_PATH:-/apdcephfs_wzc1/share_304376610/pighzliu_code/models/OLMo-2-1124-7B}"
MAX_STEPS="${MAX_STEPS:-1000}"
SEQ_LEN="${SEQ_LEN:-2048}"
SEED="${SEED:-42}"
EFF_BS="${EFF_BS:-128}"

nGPU=$(awk -F, '{print NF}' <<< "$GPUS")
BS="${BS:-4}"
GA="${GA:-}"
if [ -z "$GA" ]; then GA=$(( EFF_BS / (BS * nGPU) )); [ "$GA" -lt 1 ] && GA=1; fi
REAL_EFF=$(( BS * GA * nGPU ))

OUT_DIR="$PROJECT_ROOT/outputs/paperC_pc1_squad_A2_lora_r${R}"
LOG_FILE="$PROJECT_ROOT/logs/paperC_pc1_squad_A2_lora_r${R}.log"
mkdir -p "$OUT_DIR" "$PROJECT_ROOT/logs"

echo "[paperC_pc1_lora] R=$R GPUS=$GPUS nGPU=$nGPU BS=$BS GA=$GA eff_bs=$REAL_EFF (target $EFF_BS) -> $OUT_DIR"
[ "$REAL_EFF" -ne "$EFF_BS" ] && echo "[paperC_pc1_lora] WARNING eff_bs=$REAL_EFF != $EFF_BS"
: > "$LOG_FILE"

export CUDA_VISIBLE_DEVICES="$GPUS"
export WANDB_MODE=offline
export TOKENIZERS_PARALLELISM=false

if [ "${FOREGROUND:-0}" = "1" ]; then
  "$PYTHON_BIN" -m torch.distributed.run \
    --nnodes 1 --nproc_per_node "$nGPU" --rdzv_backend c10d --rdzv_endpoint "localhost:$PORT" \
    scripts/train_olmo2_lora_sft.py \
      --data_path "$DATA_PATH" --output_dir "$OUT_DIR" --model_path "$MODEL_PATH" \
      --lora_rank "$R" --batch_size "$BS" --grad_accumulation_steps "$GA" \
      --seq_len "$SEQ_LEN" --max_steps "$MAX_STEPS" --warmup_steps 150 \
      --save_every "${SAVE_EVERY:-500}" --log_every 10 --seed "$SEED" --gradient_checkpointing 1 \
      --keep_last_n "${KEEP_LAST_N:-3}" ${KEEP_STEPS:+--keep_steps "$KEEP_STEPS"} \
    >>"$LOG_FILE" 2>&1
  echo "[paperC_pc1_lora] FOREGROUND done R=$R (exit $?)"
  exit 0
fi

setsid nohup "$PYTHON_BIN" -m torch.distributed.run \
  --nnodes 1 --nproc_per_node "$nGPU" --rdzv_backend c10d --rdzv_endpoint "localhost:$PORT" \
  scripts/train_olmo2_lora_sft.py \
    --data_path "$DATA_PATH" \
    --output_dir "$OUT_DIR" \
    --model_path "$MODEL_PATH" \
    --lora_rank "$R" \
    --batch_size "$BS" \
    --grad_accumulation_steps "$GA" \
    --seq_len "$SEQ_LEN" \
    --max_steps "$MAX_STEPS" \
    --warmup_steps 150 \
    --save_every "${SAVE_EVERY:-500}" \
    --keep_last_n "${KEEP_LAST_N:-3}" \
    ${KEEP_STEPS:+--keep_steps "$KEEP_STEPS"} \
    --log_every 10 \
    --seed "$SEED" \
    --gradient_checkpointing 1 \
  >>"$LOG_FILE" 2>&1 &

echo "[paperC_pc1_lora] launched pid=$! (setsid) ; tail -f $LOG_FILE"
