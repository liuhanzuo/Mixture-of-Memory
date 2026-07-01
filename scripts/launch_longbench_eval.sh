#!/bin/bash
# Launch LongBench evaluation with 8-GPU data parallelism.
#
# Each GPU handles 1/8 of the samples for each dataset. After all processes
# complete, the scoring aggregation runs to compute final F1/EM metrics.
#
# Usage:
#   bash scripts/launch_longbench_eval.sh
#   bash scripts/launch_longbench_eval.sh --output_dir longbench_results/custom_run
#   bash scripts/launch_longbench_eval.sh --checkpoint outputs/other_ckpt/mem_space_adapter.pt

set -e

# ==================== Configuration ====================

# Paths (relative to PROJECT_ROOT)
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_longbench_mem_space.py"

# Default arguments (can be overridden via CLI or env vars)
MODEL_PATH="${MODEL_PATH:-${PROJECT_ROOT}/models/Meta-Llama-3-8B-Instruct}"
CHECKPOINT="${CHECKPOINT:-${PROJECT_ROOT}/outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/mem_space_adapter.pt}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${PROJECT_ROOT}/outputs/babilong_sft_phase8b_scale_chunk1024_20260519_1416/adapter_config.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/longbench_results/mem_space_p8_chunk1024}"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"
NUM_GPUS="${NUM_GPUS:-8}"
DATASETS="${DATASETS:-hotpotqa narrativeqa qasper multifieldqa_en 2wikimqa musique}"

# Override from positional/named args
while [[ $# -gt 0 ]]; do
    case $1 in
        --model_path) MODEL_PATH="$2"; shift 2;;
        --checkpoint) CHECKPOINT="$2"; shift 2;;
        --adapter_config) ADAPTER_CONFIG="$2"; shift 2;;
        --output_dir) OUTPUT_DIR="$2"; shift 2;;
        --chunk_size) CHUNK_SIZE="$2"; shift 2;;
        --num_gpus) NUM_GPUS="$2"; shift 2;;
        --datasets) DATASETS="$2"; shift 2;;
        --python) PYTHON_BIN="$2"; shift 2;;
        *) echo "Unknown argument: $1"; exit 1;;
    esac
done

# ==================== Environment ====================

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false

# Proxy for HuggingFace downloads
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export HF_HUB_OFFLINE=0

# ==================== Pre-flight checks ====================

echo "=============================================="
echo "  LongBench Evaluation - mem_space"
echo "=============================================="
echo "  Python:       ${PYTHON_BIN}"
echo "  Model:        ${MODEL_PATH}"
echo "  Checkpoint:   ${CHECKPOINT}"
echo "  Adapter cfg:  ${ADAPTER_CONFIG}"
echo "  Output dir:   ${OUTPUT_DIR}"
echo "  Chunk size:   ${CHUNK_SIZE}"
echo "  Num GPUs:     ${NUM_GPUS}"
echo "  Datasets:     ${DATASETS}"
echo "=============================================="

# Verify files exist
if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint not found: ${CHECKPOINT}"
    exit 1
fi
if [ ! -f "${ADAPTER_CONFIG}" ]; then
    echo "ERROR: Adapter config not found: ${ADAPTER_CONFIG}"
    exit 1
fi
if [ ! -d "${MODEL_PATH}" ]; then
    echo "ERROR: Model path not found: ${MODEL_PATH}"
    exit 1
fi

# Check GPU availability
echo ""
echo "[Pre-flight] Checking GPU availability..."
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null || true
echo ""

# Create output directory
mkdir -p "${OUTPUT_DIR}"
LOG_DIR="${OUTPUT_DIR}/logs"
mkdir -p "${LOG_DIR}"

# ==================== Launch parallel GPU processes ====================

echo "[Launch] Starting ${NUM_GPUS} parallel evaluation processes..."
echo ""

PIDS=()
for GPU_ID in $(seq 0 $((NUM_GPUS - 1))); do
    LOG_FILE="${LOG_DIR}/gpu_${GPU_ID}.log"

    echo "  GPU ${GPU_ID}: launching... (log: ${LOG_FILE})"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} "${EVAL_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --checkpoint "${CHECKPOINT}" \
        --adapter_config "${ADAPTER_CONFIG}" \
        --output_dir "${OUTPUT_DIR}" \
        --chunk_size ${CHUNK_SIZE} \
        --gpu_id ${GPU_ID} \
        --num_gpus ${NUM_GPUS} \
        --datasets ${DATASETS} \
        ${CHAT_TEMPLATE_FLAG:---no_chat_template} \
        > "${LOG_FILE}" 2>&1 &

    PIDS+=($!)
done

echo ""
echo "[Launch] All ${NUM_GPUS} processes started. PIDs: ${PIDS[*]}"
echo "[Launch] Waiting for all processes to complete..."
echo ""

# ==================== Wait for completion ====================

FAILED=0
for i in "${!PIDS[@]}"; do
    PID=${PIDS[$i]}
    if wait ${PID}; then
        echo "  GPU ${i}: completed successfully (PID ${PID})"
    else
        EXIT_CODE=$?
        echo "  GPU ${i}: FAILED with exit code ${EXIT_CODE} (PID ${PID})"
        echo "    Log: ${LOG_DIR}/gpu_${i}.log"
        FAILED=$((FAILED + 1))
    fi
done

echo ""

if [ ${FAILED} -gt 0 ]; then
    echo "[WARNING] ${FAILED}/${NUM_GPUS} processes failed!"
    echo "[WARNING] Check logs in: ${LOG_DIR}/"
    echo ""
    echo "Failed process logs (last 20 lines each):"
    for i in $(seq 0 $((NUM_GPUS - 1))); do
        if [ -f "${LOG_DIR}/gpu_${i}.log" ]; then
            # Check if this GPU's process had an error
            if grep -q "Traceback\|Error\|FAILED" "${LOG_DIR}/gpu_${i}.log" 2>/dev/null; then
                echo ""
                echo "=== GPU ${i} (last 20 lines) ==="
                tail -20 "${LOG_DIR}/gpu_${i}.log"
            fi
        fi
    done
    echo ""
    echo "[WARNING] Continuing with scoring on available results..."
fi

# ==================== Scoring aggregation ====================

echo "[Scoring] Running final scoring aggregation..."
echo ""

${PYTHON_BIN} "${EVAL_SCRIPT}" \
    --score_only \
    --output_dir "${OUTPUT_DIR}" \
    --datasets ${DATASETS} \
    2>&1 | tee "${LOG_DIR}/scoring.log"

echo ""
echo "=============================================="
echo "  LongBench Evaluation Complete!"
echo "  Results: ${OUTPUT_DIR}/scores.json"
echo "  Logs:    ${LOG_DIR}/"
echo "=============================================="
