#!/usr/bin/env bash
# ============================================================
# run_train_mac.sh — MAC (Memory-Augmented Context) 训练启动脚本
#
# MAC vs MAG:
#   MAG: 在 backbone 中间层做 CrossAttn + Gate 残差注入 → 侵入 backbone → 可能输出崩溃
#   MAC: 在输入端拼接 soft prefix tokens → 零侵入 → backbone 语言能力完全保持
#
# 用法:
#   # 单机 8 卡
#   bash scripts/run_train_mac.sh
#
#   # 快速 debug (单卡, 合成数据)
#   MODE=debug bash scripts/run_train_mac.sh
#
#   # 使用已训练的 MAG Selector 权重
#   PRETRAINED_SELECTOR=outputs/mag_anti_tf/context_selector.pt bash scripts/run_train_mac.sh
#
#   # 多机多卡
#   NNODES=4 NPROC=8 MASTER_ADDR=10.0.0.1 NODE_RANK=0 bash scripts/run_train_mac.sh
# ============================================================
set -euo pipefail

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
# 配置区
# ============================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- 清理旧字节码缓存 (防止循环导入等问题) ---- #
find "${PROJECT_ROOT}" -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true
find "${PROJECT_ROOT}" -name "*.pyc" -delete 2>/dev/null || true

# ---- 模型路径 ---- #
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/../models/Qwen--Qwen3-8b}"

# ---- 数据配置 ---- #
DATA_SOURCE="${DATA_SOURCE:-locomo}"
LOCOMO_DATA="${PROJECT_ROOT}/data/raw/locomo_train.jsonl"
LONG_DIALOGUE_DATA="${PROJECT_ROOT}/data/raw/long_dialogue_test.jsonl"
DATA_PATH="${DATA_PATH:-${LOCOMO_DATA}}"

# ---- 训练超参 ---- #
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-1e-4}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
MAX_REAL_SAMPLES="${MAX_REAL_SAMPLES:-10000}"
SEED="${SEED:-42}"
SLIDING_WINDOW="${SLIDING_WINDOW:-4096}"  # MAC 默认开启 SWA

# ---- MAC 架构 ---- #
TOKENS_PER_MEMORY="${TOKENS_PER_MEMORY:-4}"
PROJECTOR_LAYERS="${PROJECTOR_LAYERS:-2}"
PROJECTOR_EXPANSION="${PROJECTOR_EXPANSION:-1.5}"
PROJECTOR_INIT_SCALE="${PROJECTOR_INIT_SCALE:-0.01}"
USE_GATING="${USE_GATING:-true}"
DEEP_ENCODE_LAYERS="${DEEP_ENCODE_LAYERS:-8}"
SELECTOR_TOP_K="${SELECTOR_TOP_K:-5}"

# ---- 预训练权重 (可选, 从 MAG 复用) ---- #
PRETRAINED_SELECTOR="${PRETRAINED_SELECTOR:-}"

# ---- Anti-Teacher-Forcing (MAC 下可以更轻量) ---- #
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.05}"
KL_BETA="${KL_BETA:-0.1}"
KL_TEMPERATURE="${KL_TEMPERATURE:-2.0}"

# ---- 分布式配置 ---- #
NNODES="${NNODES:-1}"
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
NODE_RANK="${NODE_RANK:-0}"

# ---- 运行模式 ---- #
MODE="${MODE:-full}"

# ---- 输出目录 ---- #
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/mac_train_${MODE}_${TIMESTAMP}}"

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
        SLIDING_WINDOW=0
        ;;
    test)
        info "🧪 Test 模式: 使用 long_dialogue 数据"
        DATA_SOURCE="jsonl"
        DATA_PATH="${LONG_DIALOGUE_DATA}"
        MAX_REAL_SAMPLES=200
        NUM_EPOCHS=2
        ;;
    full)
        info "🚀 Full 模式: 正式训练"
        ;;
    *)
        fail "未知的 MODE: ${MODE}, 可选: full | test | debug"
        ;;
esac

# ============================================================
# 环境检查
# ============================================================

if [[ ! -d "${MODEL_PATH}" ]]; then
    fail "模型路径无效: ${MODEL_PATH}"
fi

if [[ "${DATA_SOURCE}" != "synthetic" && "${DATA_SOURCE}" != "msc" ]]; then
    if [[ ! -f "${DATA_PATH}" ]]; then
        fail "数据文件不存在: ${DATA_PATH}"
    fi
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
    fail "未检测到 GPU"
fi
if [[ "${NPROC}" -gt "${GPU_COUNT}" ]]; then
    warn "NPROC=${NPROC} > GPU数=${GPU_COUNT}, 自动调整"
    NPROC="${GPU_COUNT}"
fi

# ============================================================
# NCCL 优化
# ============================================================
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME=bond1
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NCCL_BLOCKING_WAIT="${NCCL_BLOCKING_WAIT:-1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"

export HF_HOME="${PROJECT_ROOT}/../hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ============================================================
# 打印配置
# ============================================================

echo ""
echo "============================================================"
echo "  🧠 MAC Training — ${MODE} mode"
echo "============================================================"
echo "  模型:              ${MODEL_PATH}"
echo "  数据源:            ${DATA_SOURCE}"
[[ -n "${DATA_PATH}" ]] && echo "  数据路径:          ${DATA_PATH}"
echo "  输出目录:          ${OUTPUT_DIR}"
echo ""
echo "  --- 分布式 ---"
echo "  节点数:            ${NNODES}"
echo "  每节点 GPU:        ${NPROC}"
echo "  总进程数:          $((NNODES * NPROC))"
echo ""
echo "  --- MAC 架构 ---"
echo "  tokens_per_memory: ${TOKENS_PER_MEMORY}"
echo "  projector_layers:  ${PROJECTOR_LAYERS}"
echo "  use_gating:        ${USE_GATING}"
echo "  selector_top_k:    ${SELECTOR_TOP_K}"
[[ -n "${PRETRAINED_SELECTOR}" ]] && echo "  预训练 Selector:   ${PRETRAINED_SELECTOR}"
echo ""
echo "  --- 训练超参 ---"
echo "  Epochs:            ${NUM_EPOCHS}"
echo "  LR:                ${LR}"
echo "  Batch Size:        ${BATCH_SIZE}"
echo "  Grad Accum:        ${GRAD_ACCUM}"
[[ "${SLIDING_WINDOW}" -gt 0 ]] && echo "  SWA Window:        ${SLIDING_WINDOW}"
echo "  Label Smoothing:   ${LABEL_SMOOTHING}"
echo "  KL Beta:           ${KL_BETA}"
echo "============================================================"
echo ""

# ============================================================
# 构造命令
# ============================================================

TORCHRUN_ARGS=""
if [[ "${NNODES}" -gt 1 ]]; then
    TORCHRUN_ARGS="--nnodes=${NNODES} --nproc_per_node=${NPROC}"
    TORCHRUN_ARGS+=" --rdzv_backend=c10d"
    TORCHRUN_ARGS+=" --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT}"
    TORCHRUN_ARGS+=" --node_rank=${NODE_RANK}"
elif [[ "${NPROC}" -gt 1 ]]; then
    TORCHRUN_ARGS="--nproc_per_node=${NPROC} --master_port=${MASTER_PORT}"
fi

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
TRAIN_ARGS+=" --tokens_per_memory ${TOKENS_PER_MEMORY}"
TRAIN_ARGS+=" --projector_layers ${PROJECTOR_LAYERS}"
TRAIN_ARGS+=" --projector_expansion ${PROJECTOR_EXPANSION}"
TRAIN_ARGS+=" --projector_init_scale ${PROJECTOR_INIT_SCALE}"
TRAIN_ARGS+=" --deep_encode_layers ${DEEP_ENCODE_LAYERS}"
TRAIN_ARGS+=" --selector_top_k ${SELECTOR_TOP_K}"
TRAIN_ARGS+=" --label_smoothing ${LABEL_SMOOTHING}"
TRAIN_ARGS+=" --kl_beta ${KL_BETA}"
TRAIN_ARGS+=" --kl_temperature ${KL_TEMPERATURE}"

if [[ "${USE_GATING}" == "true" ]]; then
    TRAIN_ARGS+=" --use_gating"
else
    TRAIN_ARGS+=" --no_gating"
fi

if [[ "${SLIDING_WINDOW}" -gt 0 ]]; then
    TRAIN_ARGS+=" --sliding_window ${SLIDING_WINDOW}"
fi

if [[ -n "${PRETRAINED_SELECTOR}" ]]; then
    TRAIN_ARGS+=" --pretrained_selector ${PRETRAINED_SELECTOR}"
fi

# ============================================================
# 启动训练
# ============================================================

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S)_node${NODE_RANK}.log"
info "训练日志: ${LOG_FILE}"
echo ""

if [[ -n "${TORCHRUN_ARGS}" ]]; then
    info "使用 torchrun 启动分布式训练..."
    set -x
    torchrun ${TORCHRUN_ARGS} \
        "${PROJECT_ROOT}/scripts/train_mac.py" \
        ${TRAIN_ARGS} \
        2>&1 | tee "${LOG_FILE}"
    set +x
else
    info "使用单卡模式启动训练..."
    set -x
    python3 "${PROJECT_ROOT}/scripts/train_mac.py" \
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
