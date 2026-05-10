#!/usr/bin/env bash
# 两阶段 SU 训练 — Stage 1: 固定 alpha=1.0 训 SU; Stage 2: 解冻 alpha 微调
set -euo pipefail

cd "$(dirname "$0")/.."
source .venv/bin/activate

# ====== 配置 ======
NUM_GPUS="${1:-8}"
STAGE="${2:-stage1}"    # stage1 或 stage2
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

MODEL_PATH="../models/Qwen--Qwen3-8b"
DATA_PATH="data/mag_train_generated_causal.jsonl"

# 注入层
INJECTION_LAYERS="9 18 27 35"

# 训练超参
LR=3e-4
BATCH_SIZE=2
GRAD_ACCUM=4
MAX_SEQ_LEN=512

# SU / Injector 超参
SVD_RANK=8
INIT_ALPHA=1.0       # ★ Stage 1 & 2 都从 1.0 开始
MAX_ALPHA=1.0        # ★ alpha 上限也是 1.0
MAX_RAW_KV_TOKENS=128

CONTRASTIVE_WEIGHT=1.0
CONTRASTIVE_MARGIN=1.0
CONTRASTIVE_TEMP=0.1
SELECTOR_WARMUP=0
LOG_EVERY=100
SAVE_EVERY=2000

if [ "$STAGE" = "stage1" ]; then
    NUM_EPOCHS=5
    OUTPUT_DIR="outputs/su_stage1_${TIMESTAMP}"
    TRAIN_STAGE_ARG="--train_stage stage1_sfu_only"
    RESUME_ARG=""
    ECHO_STAGE="Stage 1: 固定 alpha=1.0, 只训 SU + Selector"
elif [ "$STAGE" = "stage2" ]; then
    # Stage 2: 从 Stage 1 的最新输出加载
    STAGE1_DIR="${3:-$(ls -td outputs/su_stage1_* 2>/dev/null | head -1)}"
    if [ -z "$STAGE1_DIR" ] || [ ! -d "$STAGE1_DIR" ]; then
        echo "❌ 找不到 Stage 1 输出目录，请指定: $0 8 stage2 <stage1_dir>"
        exit 1
    fi
    NUM_EPOCHS=3
    OUTPUT_DIR="outputs/su_stage2_${TIMESTAMP}"
    TRAIN_STAGE_ARG="--train_stage stage2_finetune"
    RESUME_ARG="--resume_from $STAGE1_DIR"
    ECHO_STAGE="Stage 2: 解冻 alpha, 联合微调 (从 $STAGE1_DIR 加载)"
else
    echo "用法: $0 <num_gpus> [stage1|stage2] [stage1_dir]"
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

echo "============================================================"
echo "  🧠 SU Two-Stage Training — ${NUM_GPUS} GPUs"
echo "============================================================"
echo "  模式:           $ECHO_STAGE"
echo "  模型:           $MODEL_PATH"
echo "  数据:           $DATA_PATH"
echo "  输出:           $OUTPUT_DIR"
echo "  GPU:            $NUM_GPUS"
echo "  注入层:         $INJECTION_LAYERS"
echo "  LR:             $LR"
echo "  Epochs:         $NUM_EPOCHS"
echo "  Eff. Batch:     $BATCH_SIZE × $GRAD_ACCUM × $NUM_GPUS = $((BATCH_SIZE * GRAD_ACCUM * NUM_GPUS))"
echo "  Alpha:          init=$INIT_ALPHA, max=$MAX_ALPHA (固定)"
echo "============================================================"

LOG_FILE="${OUTPUT_DIR}/train_${TIMESTAMP}.log"
echo "[INFO] 日志: $LOG_FILE"

# 心跳初始化
python3 -c "
from heartbeat_monitor import HeartbeatMonitor
hb = HeartbeatMonitor('${OUTPUT_DIR}/heartbeat.json')
hb.update('init', progress=0.0, extra={'phase': '${STAGE}', 'num_gpus': ${NUM_GPUS}})
print(f'[INFO] Heartbeat: ${OUTPUT_DIR}/heartbeat.json')
"

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4

COMMON_ARGS="--model_path $MODEL_PATH --output_dir $OUTPUT_DIR --injection_mode su --mag_injection_layers $INJECTION_LAYERS --num_epochs $NUM_EPOCHS --lr $LR --batch_size $BATCH_SIZE --grad_accumulation_steps $GRAD_ACCUM --max_seq_len $MAX_SEQ_LEN --seed 42 --deep_encode_layers 8 --svd_rank $SVD_RANK --kv_init_alpha $INIT_ALPHA --kv_max_alpha $MAX_ALPHA --max_raw_kv_tokens $MAX_RAW_KV_TOKENS --contrastive_weight $CONTRASTIVE_WEIGHT --selector_warmup_steps $SELECTOR_WARMUP --progressive_warmup_steps 0 --contrastive_margin $CONTRASTIVE_MARGIN --contrastive_temperature $CONTRASTIVE_TEMP --log_every $LOG_EVERY --save_every $SAVE_EVERY --data_source jsonl --data_path $DATA_PATH $TRAIN_STAGE_ARG $RESUME_ARG"

if [ "$NUM_GPUS" -eq 1 ]; then
    echo "[INFO] 单卡模式..."
    python3 scripts/train_mag.py $COMMON_ARGS 2>&1 | tee "$LOG_FILE"
else
    echo "[INFO] 多卡模式: ${NUM_GPUS} GPUs..."
    torchrun --nproc_per_node=$NUM_GPUS --master_port=29501 scripts/train_mag.py $COMMON_ARGS 2>&1 | tee "$LOG_FILE"
fi

echo ""
echo -e "\e[32m[  OK]\e[0m ${STAGE} 训练完成! 输出: $OUTPUT_DIR"
