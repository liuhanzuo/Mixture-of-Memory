#!/usr/bin/env bash
# ============================================================
# run_train_mag.sh — MAG (Memory-Augmented Generation) 多节点训练启动脚本
#
# 用法:
#   # 单机 8 卡
#   bash scripts/run_train_mag.sh
#
#   # 多机多卡 (在每个节点上执行, 修改 NODE_RANK)
#   NNODES=4 NPROC=8 MASTER_ADDR=10.0.0.1 MASTER_PORT=29500 NODE_RANK=0 \
#       bash scripts/run_train_mag.sh
#
#   # 快速 debug (单卡, 合成数据)
#   MODE=debug bash scripts/run_train_mag.sh
#
#   # 使用自己生成的 long_dialogue 数据 (单卡测试)
#   MODE=test bash scripts/run_train_mag.sh
# ============================================================
set -euo pipefail

# ---- 颜色 ---- #
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

info()  { echo -e "${BLUE}[INFO]${NC} $*"; }
ok()    { echo -e "${GREEN}[  OK]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; exit 1; }

# ============================================================
# 配置区 (可通过环境变量覆盖)
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- 模型路径 ---- #
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/../models/Qwen--Qwen3-8b}"

# ---- 数据配置 ---- #
# 数据源: locomo | jsonl | msc | synthetic
DATA_SOURCE="${DATA_SOURCE:-locomo}"
# LoCoMo 数据集 (7050条, 正式训练用)
LOCOMO_DATA="${PROJECT_ROOT}/data/raw/locomo_train.jsonl"
# 自己生成的长对话数据 (62条, 小规模测试/追加训练)
LONG_DIALOGUE_DATA="${PROJECT_ROOT}/data/raw/long_dialogue_test.jsonl"
# 实际使用的数据路径 (根据 DATA_SOURCE 和 MODE 决定)
DATA_PATH="${DATA_PATH:-${LOCOMO_DATA}}"

# ---- 训练超参 ---- #
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_REAL_SAMPLES="${MAX_REAL_SAMPLES:-10000}"
SEED="${SEED:-42}"
SLIDING_WINDOW="${SLIDING_WINDOW:-0}"

# ---- MAG 架构 ---- #
# 注入层: 默认让代码自动决定, 或手动指定 (如 "6 12 18 23")
MAG_INJECTION_LAYERS="${MAG_INJECTION_LAYERS:-}"
MAG_NUM_HEADS="${MAG_NUM_HEADS:-8}"
DEEP_ENCODE_LAYERS="${DEEP_ENCODE_LAYERS:-8}"
# Per-Layer Multi-Chunk SVD: 记忆文本过 backbone 全部层, 每层独立 SVD 压缩
USE_PER_LAYER_MEMORY="${USE_PER_LAYER_MEMORY:-false}"

# ---- Anti-Teacher-Forcing 配置 ---- #
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
KL_BETA="${KL_BETA:-0.5}"
KL_TEMPERATURE="${KL_TEMPERATURE:-2.0}"
DETACH_VALUE="${DETACH_VALUE:-true}"        # true | false
SCHEDULED_SAMPLING="${SCHEDULED_SAMPLING:-false}"  # true | false
SS_START_EPOCH="${SS_START_EPOCH:-1}"
SS_MAX_RATIO="${SS_MAX_RATIO:-0.3}"

# ---- 分布式配置 ---- #
NNODES="${NNODES:-1}"
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"
SVD_RANK="${SVD_RANK:-8}"
SVD_CHUNK_SIZE="${SVD_CHUNK_SIZE:-0}"

# ---- 运行模式 ---- #
# MODE: full (正式训练) | test (小数据测试) | debug (单卡合成数据)
MODE="${MODE:-full}"

# ---- 输出目录 ---- #
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mag_train_${MODE}_${TIMESTAMP}}"

# ============================================================
# 根据 MODE 调整配置
# ============================================================

case "${MODE}" in
    debug)
        info "🐛 Debug 模式: 单卡, 合成数据, 1 epoch"
        NNODES=1
        NPROC=1
        DATA_SOURCE="synthetic"
        DATA_PATH=""
        NUM_EPOCHS=1
        MAX_REAL_SAMPLES=100
        GRAD_ACCUM=1
        SCHEDULED_SAMPLING="false"
        ;;
    test)
        info "🧪 Test 模式: 使用自己生成的 long_dialogue 数据"
        DATA_SOURCE="jsonl"
        DATA_PATH="${LONG_DIALOGUE_DATA}"
        MAX_REAL_SAMPLES=200
        NUM_EPOCHS=2
        ;;
    full)
        info "🚀 Full 模式: 正式多节点训练"
        ;;
    *)
        fail "未知的 MODE: ${MODE}, 可选: full | test | debug"
        ;;
esac

# ============================================================
# 环境检查
# ============================================================

# 检查模型路径
if [[ ! -d "${MODEL_PATH}" ]]; then
    warn "模型目录不存在: ${MODEL_PATH}"
    warn "请确认模型路径或设置 MODEL_PATH 环境变量"
    warn "  例如: MODEL_PATH=/path/to/Qwen--Qwen3-8b bash scripts/run_train_mag.sh"
    fail "模型路径无效"
fi

# 检查数据文件
if [[ "${DATA_SOURCE}" != "synthetic" && "${DATA_SOURCE}" != "msc" ]]; then
    if [[ ! -f "${DATA_PATH}" ]]; then
        warn "数据文件不存在: ${DATA_PATH}"
        fail "请确认数据路径或设置 DATA_PATH 环境变量"
    fi
fi

# 检查 GPU
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
    fail "未检测到 GPU"
fi
if [[ "${NPROC}" -gt "${GPU_COUNT}" ]]; then
    warn "NPROC=${NPROC} > 本机 GPU 数=${GPU_COUNT}, 自动调整为 ${GPU_COUNT}"
    NPROC="${GPU_COUNT}"
fi

# ============================================================
# NCCL 优化 (CephFS 多节点环境)
# ============================================================
# 参考用户的分布式训练经验: 共享文件系统(CephFS)上多 Rank 同时读写会 I/O 竞争
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
# 减少 NCCL 超时风险
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

# 避免 HuggingFace 多进程同时下载
export HF_HOME="${PROJECT_ROOT}/../hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ============================================================
# 打印配置
# ============================================================

echo ""
echo "============================================================"
echo "  🧠 MAG Training — ${MODE} mode"
echo "============================================================"
echo "  模型:           ${MODEL_PATH}"
echo "  数据源:         ${DATA_SOURCE}"
[[ -n "${DATA_PATH}" ]] && echo "  数据路径:       ${DATA_PATH}"
echo "  输出目录:       ${OUTPUT_DIR}"
echo ""
echo "  --- 分布式 ---"
echo "  节点数:         ${NNODES}"
echo "  每节点 GPU:     ${NPROC}"
echo "  总进程数:       $((NNODES * NPROC))"
echo "  Master:         ${MASTER_ADDR}:${MASTER_PORT}"
echo "  当前 Node Rank: ${NODE_RANK}"
echo ""
echo "  --- 训练超参 ---"
echo "  Epochs:         ${NUM_EPOCHS}"
echo "  LR:             ${LR}"
echo "  Batch Size:     ${BATCH_SIZE}"
echo "  Grad Accum:     ${GRAD_ACCUM}"
echo "  Max Seq Len:    ${MAX_SEQ_LEN}"
echo "  Seed:           ${SEED}"
[[ "${SLIDING_WINDOW}" -gt 0 ]] && echo "  SWA Window:     ${SLIDING_WINDOW}"
echo ""
echo "  --- SVD 压缩记忆 ---"
echo "  SVD Rank:        ${SVD_RANK}"
echo "  SVD Chunk Size:  ${SVD_CHUNK_SIZE}"
echo "  Per-Layer SVD:   ${USE_PER_LAYER_MEMORY}"
echo ""
echo "  --- Anti-Teacher-Forcing ---"
echo "  Label Smoothing:     ${LABEL_SMOOTHING}"
echo "  KL Beta:             ${KL_BETA}"
echo "  KL Temperature:      ${KL_TEMPERATURE}"
echo "  Detach Value (V):    ${DETACH_VALUE}"
echo "  Scheduled Sampling:  ${SCHEDULED_SAMPLING}"
[[ "${SCHEDULED_SAMPLING}" == "true" ]] && echo "    start_epoch=${SS_START_EPOCH}, max_ratio=${SS_MAX_RATIO}"
echo "============================================================"
echo ""

# ============================================================
# 构造命令
# ============================================================

# ---- torchrun 参数 ---- #
TORCHRUN_ARGS=""
if [[ "${NNODES}" -gt 1 ]]; then
    # 多机多卡
    TORCHRUN_ARGS="--nnodes=${NNODES} --nproc_per_node=${NPROC}"
    TORCHRUN_ARGS+=" --rdzv_backend=c10d"
    TORCHRUN_ARGS+=" --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
    TORCHRUN_ARGS+=" --node_rank=${NODE_RANK}"
elif [[ "${NPROC}" -gt 1 ]]; then
    # 单机多卡
    TORCHRUN_ARGS="--nproc_per_node=${NPROC} --master_port=${MASTER_PORT}"
else
    # 单卡: 不用 torchrun, 直接 python
    TORCHRUN_ARGS=""
fi

# ---- train_mag.py 参数 ---- #
TRAIN_ARGS=""
TRAIN_ARGS+=" --model_path ${MODEL_PATH}"
TRAIN_ARGS+=" --output_dir ${OUTPUT_DIR}"
TRAIN_ARGS+=" --data_source ${DATA_SOURCE}"
[[ -n "${DATA_PATH}" ]] && TRAIN_ARGS+=" --data_path ${DATA_PATH}"
TRAIN_ARGS+=" --num_epochs ${NUM_EPOCHS}"
TRAIN_ARGS+=" --lr ${LR}"
TRAIN_ARGS+=" --batch_size ${BATCH_SIZE}"
TRAIN_ARGS+=" --grad_accumulation_steps ${GRAD_ACCUM}"
TRAIN_ARGS+=" --max_seq_len ${MAX_SEQ_LEN}"
TRAIN_ARGS+=" --max_real_samples ${MAX_REAL_SAMPLES}"
TRAIN_ARGS+=" --seed ${SEED}"
TRAIN_ARGS+=" --mag_num_heads ${MAG_NUM_HEADS}"
TRAIN_ARGS+=" --deep_encode_layers ${DEEP_ENCODE_LAYERS}"
TRAIN_ARGS+=" --use_compressed_memory"
TRAIN_ARGS+=" --svd_rank ${SVD_RANK}"
TRAIN_ARGS+=" --svd_chunk_size ${SVD_CHUNK_SIZE}"

# Per-Layer Multi-Chunk SVD
if [[ "${USE_PER_LAYER_MEMORY}" == "true" ]]; then
    TRAIN_ARGS+=" --use_per_layer_memory"
fi

# MAG 注入层 (手动指定或自动)
if [[ -n "${MAG_INJECTION_LAYERS}" ]]; then
    TRAIN_ARGS+=" --mag_injection_layers ${MAG_INJECTION_LAYERS}"
fi

# SWA 滑窗
if [[ "${SLIDING_WINDOW}" -gt 0 ]]; then
    TRAIN_ARGS+=" --sliding_window ${SLIDING_WINDOW}"
fi

# Anti-Teacher-Forcing 参数
TRAIN_ARGS+=" --label_smoothing ${LABEL_SMOOTHING}"
TRAIN_ARGS+=" --kl_beta ${KL_BETA}"
TRAIN_ARGS+=" --kl_temperature ${KL_TEMPERATURE}"

if [[ "${DETACH_VALUE}" == "true" ]]; then
    TRAIN_ARGS+=" --detach_value"
else
    TRAIN_ARGS+=" --no_detach_value"
fi

if [[ "${SCHEDULED_SAMPLING}" == "true" ]]; then
    TRAIN_ARGS+=" --scheduled_sampling"
    TRAIN_ARGS+=" --ss_start_epoch ${SS_START_EPOCH}"
    TRAIN_ARGS+=" --ss_max_ratio ${SS_MAX_RATIO}"
fi

# ============================================================
# 启动训练
# ============================================================

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"

info "训练日志: ${LOG_FILE}"
echo ""

if [[ -n "${TORCHRUN_ARGS}" ]]; then
    # 分布式训练 (torchrun)
    info "使用 torchrun 启动分布式训练..."
    set -x
    torchrun ${TORCHRUN_ARGS} \
        "${PROJECT_ROOT}/scripts/train_mag.py" \
        ${TRAIN_ARGS} \
        2>&1 | tee "${LOG_FILE}"
    set +x
else
    # 单卡训练
    info "使用单卡模式启动训练..."
    set -x
    python3 "${PROJECT_ROOT}/scripts/train_mag.py" \
        ${TRAIN_ARGS} \
        2>&1 | tee "${LOG_FILE}"
    set +x
fi

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    ok "训练完成! 输出目录: ${OUTPUT_DIR}"
else
    fail "训练失败 (exit code: ${EXIT_CODE}), 请检查日志: ${LOG_FILE}"
fi
