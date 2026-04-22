#!/bin/bash
# ==============================================================================
# run_train_su.sh — MemoryLLM Self-Update Function 训练脚本
#
# 修复 raw_kv 的问题:
#   1. 只注入 4 个中间层 (而非全部 36 层)
#   2. backbone 完全冻结 (只训练 SU + selector + injector)
#   3. SU 函数可学习记忆更新, 避免 alpha 卡死
#   4. 启用对比学习辅助训练
#
# 用法:
#   bash scripts/run_train_su.sh                    # 默认 8 卡
#   bash scripts/run_train_su.sh                    # 自动检测 GPU 数
#   MODE=test bash scripts/run_train_su.sh          # 测试模式 (小数据)
# ==============================================================================
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# ---- 颜色 ---- #
info()  { echo -e "\033[0;34m[INFO]\033[0m $*"; }
ok()    { echo -e "\033[0;32m[  OK]\033[0m $*"; }
fail()  { echo -e "\033[0;31m[FAIL]\033[0m $*"; exit 1; }

# ============================================================
# 配置区 (可通过环境变量覆盖)
# ============================================================

# ---- 模型路径 ---- #
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/../models/Qwen--Qwen3-8b}"

# ---- 数据 ---- #
DATA_PATH="${DATA_PATH:-${PROJECT_ROOT}/data/mag_train_generated.jsonl}"

# ---- 注入模式 ---- #
INJECTION_MODE="su"  # ★ SU 模式

# ---- 训练超参 ---- #
NUM_EPOCHS="${NUM_EPOCHS:-3}"
LR="${LR:-5e-4}"            # SU 模式用更高学习率 (可训练参数少)
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-512}"
SEED="${SEED:-42}"

# ---- 注入层: 选 Qwen3-8b 的 4 个中间层 (总 36 层: 0-35) ---- #
# 上次全 36 层注入 alpha 全卡在 0.01, 这回只选 4 层
INJECTION_LAYERS="${INJECTION_LAYERS:-9 18 27 35}"

# ---- KV 配置 ---- #
KV_INIT_ALPHA="${KV_INIT_ALPHA:-0.1}"     # ★ 从 0.1 开始 (不是 0.01)
KV_MAX_ALPHA="${KV_MAX_ALPHA:-1.0}"       # ★ 放开上限
MAX_RAW_KV_TOKENS="${MAX_RAW_KV_TOKENS:-128}"
DEEP_ENCODE_LAYERS="${DEEP_ENCODE_LAYERS:-8}"
SVD_RANK="${SVD_RANK:-8}"

# ---- 对比学习 (重要: 辅助 selector 训练) ---- #
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-1.0}"
SELECTOR_WARMUP="${SELECTOR_WARMUP:-100}"

# ---- 分布式 ---- #
NPROC="${NPROC:-$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)}"
MASTER_PORT="${MASTER_PORT:-29501}"

# ---- 运行模式 ---- #
MODE="${MODE:-full}"

# ---- 输出目录 ---- #
TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/outputs/su_train_${MODE}_${TIMESTAMP}}"

# ============================================================
# 根据 MODE 调整
# ============================================================
case "${MODE}" in
    debug)
        info "Debug 模式: 单卡, 合成数据"
        NPROC=1
        DATA_PATH=""
        NUM_EPOCHS=1
        BATCH_SIZE=2
        GRAD_ACCUM=1
        ;;
    test)
        info "Test 模式: 小数据测试"
        MAX_REAL_SAMPLES=200
        NUM_EPOCHS=1
        ;;
    full)
        info "Full 模式: 正式训练"
        ;;
    *)
        fail "未知 MODE: ${MODE}"
        ;;
esac

# ============================================================
# 环境检查
# ============================================================
if [[ ! -d "${MODEL_PATH}" ]]; then
    fail "模型目录不存在: ${MODEL_PATH}"
fi

GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if [[ "${NPROC}" -gt "${GPU_COUNT}" ]]; then
    NPROC="${GPU_COUNT}"
fi

# NCCL
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export TORCH_NCCL_BLOCKING_WAIT="${TORCH_NCCL_BLOCKING_WAIT:-1}"
export HF_HOME="${PROJECT_ROOT}/../hf_cache"
export TRANSFORMERS_CACHE="${HF_HOME}"

mkdir -p "${OUTPUT_DIR}" "${HF_HOME}"

# ============================================================
# 打印配置
# ============================================================
echo ""
echo "============================================================"
echo "  🧠 SU Training — ${MODE} mode (${INJECTION_MODE})"
echo "============================================================"
echo "  模型:           ${MODEL_PATH}"
echo "  数据:           ${DATA_PATH:-synthetic}"
echo "  输出:           ${OUTPUT_DIR}"
echo "  GPU:            ${NPROC}"
echo "  注入层:         ${INJECTION_LAYERS}"
echo "  LR:             ${LR}"
echo "  Epochs:         ${NUM_EPOCHS}"
echo "  Batch:          ${BATCH_SIZE} × ${GRAD_ACCUM} × ${NPROC} = $((BATCH_SIZE * GRAD_ACCUM * NPROC))"
echo "  Alpha:          init=${KV_INIT_ALPHA}, max=${KV_MAX_ALPHA}"
echo "  Contrastive:    ${CONTRASTIVE_WEIGHT}"
echo "============================================================"
echo ""

# ============================================================
# 构造训练命令
# ============================================================
COMMON_ARGS=""
COMMON_ARGS+=" --model_path ${MODEL_PATH}"
COMMON_ARGS+=" --output_dir ${OUTPUT_DIR}"
COMMON_ARGS+=" --injection_mode ${INJECTION_MODE}"
COMMON_ARGS+=" --mag_injection_layers ${INJECTION_LAYERS}"
COMMON_ARGS+=" --num_epochs ${NUM_EPOCHS}"
COMMON_ARGS+=" --lr ${LR}"
COMMON_ARGS+=" --batch_size ${BATCH_SIZE}"
COMMON_ARGS+=" --grad_accumulation_steps ${GRAD_ACCUM}"
COMMON_ARGS+=" --max_seq_len ${MAX_SEQ_LEN}"
COMMON_ARGS+=" --seed ${SEED}"
COMMON_ARGS+=" --deep_encode_layers ${DEEP_ENCODE_LAYERS}"
COMMON_ARGS+=" --svd_rank ${SVD_RANK}"
COMMON_ARGS+=" --kv_init_alpha ${KV_INIT_ALPHA}"
COMMON_ARGS+=" --kv_max_alpha ${KV_MAX_ALPHA}"
COMMON_ARGS+=" --max_raw_kv_tokens ${MAX_RAW_KV_TOKENS}"
COMMON_ARGS+=" --contrastive_weight ${CONTRASTIVE_WEIGHT}"
COMMON_ARGS+=" --selector_warmup_steps ${SELECTOR_WARMUP}"
COMMON_ARGS+=" --contrastive_margin 1.0"
COMMON_ARGS+=" --contrastive_temperature 0.1"
COMMON_ARGS+=" --progressive_injection"
COMMON_ARGS+=" --progressive_warmup_steps 200"
COMMON_ARGS+=" --log_every 50"
COMMON_ARGS+=" --save_every 500"

if [[ -n "${DATA_PATH}" && -f "${DATA_PATH}" ]]; then
    COMMON_ARGS+=" --data_source jsonl"
    COMMON_ARGS+=" --data_path ${DATA_PATH}"
else
    COMMON_ARGS+=" --data_source synthetic"
fi

if [[ "${MODE}" == "test" ]]; then
    COMMON_ARGS+=" --max_real_samples ${MAX_REAL_SAMPLES:-200}"
fi

LOG_FILE="${OUTPUT_DIR}/train_$(date +%Y%m%d_%H%M%S).log"
info "日志: ${LOG_FILE}"

# ============================================================
# 启动训练
# ============================================================
if [[ "${NPROC}" -gt 1 ]]; then
    info "torchrun ${NPROC} GPU..."
    set -x
    torchrun --nproc_per_node="${NPROC}" --master_port="${MASTER_PORT}" \
        "${PROJECT_ROOT}/scripts/train_mag.py" \
        ${COMMON_ARGS} \
        2>&1 | tee "${LOG_FILE}"
    set +x
else
    info "单卡模式..."
    set -x
    python3 "${PROJECT_ROOT}/scripts/train_mag.py" \
        ${COMMON_ARGS} \
        2>&1 | tee "${LOG_FILE}"
    set +x
fi

EXIT_CODE=${PIPESTATUS[0]}

echo ""
if [[ "${EXIT_CODE}" -eq 0 ]]; then
    ok "训练完成! 输出: ${OUTPUT_DIR}"
else
    fail "训练失败 (exit: ${EXIT_CODE}), 日志: ${LOG_FILE}"
fi
