#!/bin/bash
# Parallel LongBench eval across all Dolmino CPT checkpoints (one GPU per ckpt).
# Designed to run on remote 8-GPU node (e.g. 28.59.80.196).
#
# Note: Dolmino CPT trains on Meta-Llama-3-8B *base* — must pass --no_chat_template
# (raw Llama-3-8B has no chat_template; previous step15k eval crashed because the
#  default launcher passes --use_chat_template).

set -u

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
EVAL_SCRIPT="${PROJECT_ROOT}/scripts/eval_longbench_mem_space.py"
MODEL_PATH="${PROJECT_ROOT}/models/Meta-Llama-3-8B"
CKPT_DIR="${PROJECT_ROOT}/outputs/dolmino_cpt_local_long"
ADAPTER_CFG="${CKPT_DIR}/adapter_config.json"
DATASETS="hotpotqa narrativeqa qasper multifieldqa_en 2wikimqa musique"
CHUNK_SIZE="${CHUNK_SIZE:-1024}"

# 8 checkpoints, 8 GPUs, 1:1 mapping
STEPS=(5000 10000 15000 20000 25000 30000 35000 40000)
RUN_STAMP=$(date +%Y%m%d_%H%M%S)
OUT_BASE="${PROJECT_ROOT}/longbench_results/dolmino_cpt_v2_${RUN_STAMP}"
mkdir -p "${OUT_BASE}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HUB_OFFLINE=0

echo "=============================================="
echo "  LongBench parallel eval (one ckpt per GPU)"
echo "  Run stamp: ${RUN_STAMP}"
echo "  Python:    ${PYTHON_BIN}"
echo "  Model:     ${MODEL_PATH}"
echo "  Ckpt dir:  ${CKPT_DIR}"
echo "  Output:    ${OUT_BASE}/"
echo "  Datasets:  ${DATASETS}"
echo "  Steps:     ${STEPS[*]}"
echo "=============================================="
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader || true
echo ""

# Pre-flight: verify each checkpoint exists
for STEP in "${STEPS[@]}"; do
    STEP_STR=$(printf "%06d" "${STEP}")
    CKPT="${CKPT_DIR}/mem_space_adapter_step${STEP_STR}.pt"
    if [ ! -f "${CKPT}" ]; then
        echo "ERROR: missing checkpoint: ${CKPT}"
        exit 1
    fi
done
[ -f "${ADAPTER_CFG}" ] || { echo "ERROR: adapter_config missing"; exit 1; }
[ -d "${MODEL_PATH}" ] || { echo "ERROR: model missing"; exit 1; }

PIDS=()
for IDX in "${!STEPS[@]}"; do
    GPU_ID="${IDX}"
    STEP="${STEPS[$IDX]}"
    STEP_STR=$(printf "%06d" "${STEP}")
    CKPT="${CKPT_DIR}/mem_space_adapter_step${STEP_STR}.pt"
    OUT="${OUT_BASE}/step${STEP}"
    mkdir -p "${OUT}/logs"
    LOG="${OUT}/logs/eval.log"

    echo "[Launch] GPU ${GPU_ID} -> step${STEP} -> ${OUT}"
    CUDA_VISIBLE_DEVICES=${GPU_ID} ${PYTHON_BIN} "${EVAL_SCRIPT}" \
        --model_path "${MODEL_PATH}" \
        --checkpoint "${CKPT}" \
        --adapter_config "${ADAPTER_CFG}" \
        --output_dir "${OUT}" \
        --chunk_size ${CHUNK_SIZE} \
        --gpu_id 0 \
        --num_gpus 1 \
        --datasets ${DATASETS} \
        --no_chat_template \
        > "${LOG}" 2>&1 &
    PIDS+=($!)
done

echo ""
echo "[Launch] All 8 processes started. PIDs: ${PIDS[*]}"
echo "[Launch] Waiting for all processes to complete..."
echo ""

FAILED=0
for IDX in "${!PIDS[@]}"; do
    PID="${PIDS[$IDX]}"
    STEP="${STEPS[$IDX]}"
    if wait "${PID}"; then
        echo "  step${STEP} (GPU ${IDX}, PID ${PID}): completed"
    else
        EXIT_CODE=$?
        echo "  step${STEP} (GPU ${IDX}, PID ${PID}): FAILED exit=${EXIT_CODE}"
        FAILED=$((FAILED + 1))
    fi
done

echo ""
if [ ${FAILED} -gt 0 ]; then
    echo "[WARNING] ${FAILED}/8 processes failed. See per-step logs."
fi

echo ""
echo "[Scoring] Aggregating per-step scores..."
for STEP in "${STEPS[@]}"; do
    OUT="${OUT_BASE}/step${STEP}"
    if [ -d "${OUT}" ]; then
        ${PYTHON_BIN} "${EVAL_SCRIPT}" \
            --score_only \
            --output_dir "${OUT}" \
            --datasets ${DATASETS} \
            2>&1 | tee "${OUT}/logs/scoring.log"
        echo "  step${STEP}: $(${PYTHON_BIN} -c "import json,sys; d=json.load(open('${OUT}/scores.json')); print(d.get('AVERAGE',{}))" 2>/dev/null)"
    fi
done

echo ""
echo "=============================================="
echo "  LongBench parallel eval DONE"
echo "  Base output dir: ${OUT_BASE}/"
echo "=============================================="
