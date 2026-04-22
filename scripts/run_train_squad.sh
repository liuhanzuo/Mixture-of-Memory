#!/usr/bin/env bash
# 对比实验：SQuAD 数据训练 — 验证英文 QA 场景下记忆注入效果
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# ====== 配置 ======
NUM_GPUS="${1:-4}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODEL_PATH="../models/Qwen--Qwen3-8b"
DATA_PATH="data/squad_train.jsonl"

INJECTION_LAYERS="9 18 27 35"
LR=3e-4
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_SEQ_LEN=512
NUM_EPOCHS=3

SVD_RANK=8
INIT_ALPHA=1.0
MAX_ALPHA=1.0
MAX_RAW_KV_TOKENS=128

CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_TEMP=1.0
SELECTOR_WARMUP=0

LOG_EVERY=100
SAVE_EVERY=2000
OUTPUT_DIR="outputs/squad_bce_${TIMESTAMP}"

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  🧠 SQuAD Recall Training — ${NUM_GPUS} GPUs"
echo "============================================================"
echo "  模型:           $MODEL_PATH"
echo "  数据:           $DATA_PATH (10K SQuAD recall 样本)"
echo "  输出:           $OUTPUT_DIR"
echo "  GPU:            $NUM_GPUS"
echo "  Loss:           BCE (temp=1.0)"
echo "============================================================"

LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"

python3 -c "
from heartbeat_monitor import HeartbeatMonitor
hb = HeartbeatMonitor('${OUTPUT_DIR}/heartbeat.json')
hb.update('init', progress=0.0, extra={'num_gpus': ${NUM_GPUS}, 'data': 'squad', 'loss': 'bce'})
"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

COMMON_ARGS="--model_path $MODEL_PATH --output_dir $OUTPUT_DIR --injection_mode su --mag_injection_layers $INJECTION_LAYERS --num_epochs $NUM_EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accumulation_steps $GRAD_ACCUM --max_seq_len $MAX_SEQ_LEN --seed 42 --deep_encode_layers 8 --svd_rank $SVD_RANK --kv_init_alpha $INIT_ALPHA --kv_max_alpha $MAX_ALPHA --max_raw_kv_tokens $MAX_RAW_KV_TOKENS --contrastive_weight $CONTRASTIVE_WEIGHT --selector_warmup_steps $SELECTOR_WARMUP --progressive_warmup_steps 0 --contrastive_temperature $CONTRASTIVE_TEMP --log_every $LOG_EVERY --save_every $SAVE_EVERY --data_source jsonl --data_path $DATA_PATH --train_stage joint"

if [ "$NUM_GPUS" -eq 1 ]; then
    python3 scripts/train_mag.py $COMMON_ARGS 2>&1 | tee "$LOG_FILE"
else
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29503 scripts/train_mag.py $COMMON_ARGS 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "\e[32m[  OK]\e[0m SQuAD 训练完成! 输出: $OUTPUT_DIR"
