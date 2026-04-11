#!/usr/bin/env bash
# ============================================================
# run_eval_mag.sh — MAG 评估启动脚本
#
# 用法:
#   # 使用默认配置 (自动查找最新训练输出)
#   bash scripts/run_eval_mag.sh
#
#   # 指定权重目录
#   MAG_WEIGHTS_DIR=outputs/mag_train_full_20260408_125330 bash scripts/run_eval_mag.sh
#
#   # 快速评估 (少量样本)
#   NUM_EVAL_SAMPLES=20 bash scripts/run_eval_mag.sh
#
#   # 调整推理时记忆注入强度
#   GATE_SCALE=0.5 bash scripts/run_eval_mag.sh
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

# ---- MAG 权重目录 ---- #
# 默认自动查找最新的训练输出目录
if [[ -z "${MAG_WEIGHTS_DIR:-}" ]]; then
    LATEST_DIR="$(ls -td "${PROJECT_ROOT}"/outputs/mag_train_*/ 2>/dev/null | head -1 || true)"
    if [[ -n "${LATEST_DIR}" && -f "${LATEST_DIR}/mag_config.json" ]]; then
        MAG_WEIGHTS_DIR="${LATEST_DIR}"
    else
        fail "未找到训练输出目录, 请设置 MAG_WEIGHTS_DIR 环境变量"
    fi
fi

# ---- 评估数据 ---- #
EVAL_DATA="${EVAL_DATA:-${PROJECT_ROOT}/data/mag_eval_generated.jsonl}"

# ---- 评估参数 ---- #
NUM_EVAL_SAMPLES="${NUM_EVAL_SAMPLES:-200}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-1024}"
NUM_GENERATE_SAMPLES="${NUM_GENERATE_SAMPLES:-10}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"
DEEP_ENCODE_LAYERS="${DEEP_ENCODE_LAYERS:-8}"
SLIDING_WINDOW="${SLIDING_WINDOW:-4096}"

# ---- 推理注入控制 ---- #
GATE_SCALE="${GATE_SCALE:-0.5}"
LAYER_GATE_SCALES="${LAYER_GATE_SCALES:-}"
MAG_INJECT_STEPS="${MAG_INJECT_STEPS:-10}"

# ---- 评估维度开关 (默认全开) ---- #
EVAL_SELECTOR="${EVAL_SELECTOR:-true}"
EVAL_GATE="${EVAL_GATE:-true}"
EVAL_PPL="${EVAL_PPL:-true}"
EVAL_GENERATION="${EVAL_GENERATION:-true}"

# ============================================================
# 环境检查
# ============================================================

# 检查模型路径
if [[ ! -d "${MODEL_PATH}" ]]; then
    fail "模型目录不存在: ${MODEL_PATH}"
fi

# 检查权重目录
if [[ ! -f "${MAG_WEIGHTS_DIR}/mag_config.json" ]]; then
    fail "权重目录无效 (缺少 mag_config.json): ${MAG_WEIGHTS_DIR}"
fi

# 检查评估数据
if [[ ! -f "${EVAL_DATA}" ]]; then
    fail "评估数据不存在: ${EVAL_DATA}"
fi

# 检查 GPU
GPU_COUNT="$(nvidia-smi -L 2>/dev/null | wc -l || echo 0)"
if [[ "${GPU_COUNT}" -eq 0 ]]; then
    fail "未检测到 GPU"
fi

# ============================================================
# 打印配置
# ============================================================

echo ""
echo "============================================================"
echo "  🧪 MAG Evaluation"
echo "============================================================"
echo "  模型:           ${MODEL_PATH}"
echo "  MAG 权重:       ${MAG_WEIGHTS_DIR}"
echo "  评估数据:       ${EVAL_DATA}"
echo ""
echo "  --- 评估参数 ---"
echo "  样本数:         ${NUM_EVAL_SAMPLES}"
echo "  Max Seq Len:    ${MAX_SEQ_LEN}"
echo "  生成样本数:     ${NUM_GENERATE_SAMPLES}"
echo "  Max New Tokens:  ${MAX_NEW_TOKENS}"
[[ "${SLIDING_WINDOW}" -gt 0 ]] && echo "  SWA Window:     ${SLIDING_WINDOW}"
echo ""
echo "  --- 注入控制 ---"
echo "  Gate Scale:      ${GATE_SCALE}"
[[ -n "${LAYER_GATE_SCALES}" ]] && echo "  Layer Scales:    ${LAYER_GATE_SCALES}"
echo "  Inject Steps:    ${MAG_INJECT_STEPS}"
echo ""
echo "  --- 评估维度 ---"
echo "  Selector:        ${EVAL_SELECTOR}"
echo "  Gate:            ${EVAL_GATE}"
echo "  PPL:             ${EVAL_PPL}"
echo "  Generation:      ${EVAL_GENERATION}"
echo "============================================================"
echo ""

# ============================================================
# 构造命令
# ============================================================

EVAL_ARGS=""
EVAL_ARGS+=" --model_path ${MODEL_PATH}"
EVAL_ARGS+=" --mag_weights_dir ${MAG_WEIGHTS_DIR}"
EVAL_ARGS+=" --data_path ${EVAL_DATA}"
EVAL_ARGS+=" --num_eval_samples ${NUM_EVAL_SAMPLES}"
EVAL_ARGS+=" --max_seq_len ${MAX_SEQ_LEN}"
EVAL_ARGS+=" --num_generate_samples ${NUM_GENERATE_SAMPLES}"
EVAL_ARGS+=" --max_new_tokens ${MAX_NEW_TOKENS}"
EVAL_ARGS+=" --deep_encode_layers ${DEEP_ENCODE_LAYERS}"
EVAL_ARGS+=" --inference_gate_scale ${GATE_SCALE}"
EVAL_ARGS+=" --mag_inject_steps ${MAG_INJECT_STEPS}"

if [[ -n "${LAYER_GATE_SCALES}" ]]; then
    EVAL_ARGS+=" --layer_gate_scales ${LAYER_GATE_SCALES}"
fi

if [[ "${SLIDING_WINDOW}" -gt 0 ]]; then
    EVAL_ARGS+=" --sliding_window ${SLIDING_WINDOW}"
fi

# 评估维度开关
if [[ "${EVAL_SELECTOR}" == "true" ]]; then
    EVAL_ARGS+=" --eval_selector"
fi
if [[ "${EVAL_GATE}" == "true" ]]; then
    EVAL_ARGS+=" --eval_gate"
fi
if [[ "${EVAL_PPL}" == "true" ]]; then
    EVAL_ARGS+=" --eval_ppl"
fi
if [[ "${EVAL_GENERATION}" == "true" ]]; then
    EVAL_ARGS+=" --eval_generation"
fi

# ============================================================
# 启动评估
# ============================================================

# ---- Gate Scale 扫描模式 ---- #
# 设置 SWEEP_GATE_SCALE=true 启用扫描模式
# 会依次用不同的 gate_scale 值运行 PPL 评估，找到"开始爆炸"的临界点
SWEEP_GATE_SCALE="${SWEEP_GATE_SCALE:-false}"
SWEEP_SCALES="${SWEEP_SCALES:-0.0 0.001 0.005 0.01 0.02 0.05 0.1 0.2 0.3 0.5 1.0}"

if [[ "${SWEEP_GATE_SCALE}" == "true" ]]; then
    info "🔍 Gate Scale 扫描模式: 依次评估 gate_scale = ${SWEEP_SCALES}"
    echo ""
    echo "============================================================"
    echo "  Gate Scale Sweep Results"
    echo "============================================================"
    printf "  %-12s %-15s %-15s %-10s\n" "gate_scale" "PPL(有记忆)" "PPL(无记忆)" "比值"
    echo "  --------------------------------------------------------"

    for SCALE in ${SWEEP_SCALES}; do
        # 构造扫描命令: 只跑 PPL 评估, 少量样本快速扫描
        SWEEP_ARGS=""
        SWEEP_ARGS+=" --model_path ${MODEL_PATH}"
        SWEEP_ARGS+=" --mag_weights_dir ${MAG_WEIGHTS_DIR}"
        SWEEP_ARGS+=" --data_path ${EVAL_DATA}"
        SWEEP_ARGS+=" --num_eval_samples ${SWEEP_NUM_SAMPLES:-50}"
        SWEEP_ARGS+=" --max_seq_len ${MAX_SEQ_LEN}"
        SWEEP_ARGS+=" --deep_encode_layers ${DEEP_ENCODE_LAYERS}"
        SWEEP_ARGS+=" --inference_gate_scale ${SCALE}"
        SWEEP_ARGS+=" --mag_inject_steps ${MAG_INJECT_STEPS}"
        SWEEP_ARGS+=" --eval_ppl"
        # 只跑 PPL, 不跑其他评估
        # (不加 --eval_selector --eval_gate --eval_generation)

        if [[ "${SLIDING_WINDOW}" -gt 0 ]]; then
            SWEEP_ARGS+=" --sliding_window ${SLIDING_WINDOW}"
        fi

        # 运行并提取 PPL 结果
        RESULT=$(python3 "${PROJECT_ROOT}/scripts/eval_mag.py" ${SWEEP_ARGS} 2>&1 | \
                 grep -E "PPL \(有记忆\)|PPL \(无记忆\)" || echo "ERROR")

        if [[ "${RESULT}" == "ERROR" ]]; then
            printf "  %-12s %-15s %-15s %-10s\n" "${SCALE}" "ERROR" "-" "-"
        else
            PPL_WITH=$(echo "${RESULT}" | grep "PPL (有记忆)" | awk -F'[: ]+' '{for(i=1;i<=NF;i++) if($i+0==$i && $i>0) print $i}' | head -1)
            PPL_WITHOUT=$(echo "${RESULT}" | grep "PPL (无记忆)" | awk -F'[: ]+' '{for(i=1;i<=NF;i++) if($i+0==$i && $i>0) print $i}' | head -1)
            if [[ -n "${PPL_WITH}" && -n "${PPL_WITHOUT}" ]]; then
                RATIO=$(python3 -c "print(f'{float(${PPL_WITH})/float(${PPL_WITHOUT}):.2f}x')" 2>/dev/null || echo "?")
                printf "  %-12s %-15.2f %-15.2f %-10s\n" "${SCALE}" "${PPL_WITH}" "${PPL_WITHOUT}" "${RATIO}"
            else
                printf "  %-12s %-15s %-15s %-10s\n" "${SCALE}" "${PPL_WITH:-?}" "${PPL_WITHOUT:-?}" "?"
            fi
        fi
    done

    echo "  --------------------------------------------------------"
    echo ""
    ok "Gate Scale 扫描完成!"
    exit 0
fi

# ---- 标准评估模式 ---- #
info "启动 MAG 评估..."
echo ""

set -x
python3 "${PROJECT_ROOT}/scripts/eval_mag.py" ${EVAL_ARGS}
set +x

echo ""
ok "评估完成! 结果保存在: ${MAG_WEIGHTS_DIR}/"
