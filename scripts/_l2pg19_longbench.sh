#!/bin/bash
# LongBench W0 eval for L2+pg19 ckpt (expL2ON_pg19_N128) on .249 diskB.
# Strictly matches dolmino-arm 口径 (commit adf2d0d):
#   base Meta-Llama-3-8B, no_chat_template, chunk512, W0 (memory-only),
#   tasks={multifieldqa_en,2wikimqa,musique}, max_samples=100, 4-shard x 25.
# Single ckpt -> all 8 GPUs (num_gpus=8) for max throughput, one-run-per-node.
set -u

PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
PYTHON_BIN="${PROJECT_ROOT}/.venv/bin/python"
EVAL="${PROJECT_ROOT}/scripts/eval_longbench_mem_space.py"
MODEL="${PROJECT_ROOT}/models/Meta-Llama-3-8B"
DATASETS="multifieldqa_en 2wikimqa musique"
CHUNK=512
MAXS=100

CKPT="${PROJECT_ROOT}/outputs/expL2ON_pg19_N128/mem_space_adapter.pt"
ACFG="${PROJECT_ROOT}/outputs/expL2ON_pg19_N128/adapter_config.json"
OUT="${PROJECT_ROOT}/longbench_results/L2pg19_chunk512_final"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

mkdir -p "${OUT}/logs"
PIDS=()
for SH in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=${SH} ${PYTHON_BIN} "${EVAL}" \
        --model_path "${MODEL}" \
        --checkpoint "${CKPT}" \
        --adapter_config "${ACFG}" \
        --output_dir "${OUT}" \
        --chunk_size ${CHUNK} \
        --gpu_id ${SH} \
        --num_gpus 8 \
        --datasets ${DATASETS} \
        --max_samples ${MAXS} \
        --no_chat_template \
        > "${OUT}/logs/gpu_${SH}.log" 2>&1 &
    PIDS+=($!)
done
echo "[${OUT}] 8 shards launched: ${PIDS[*]}"
for P in "${PIDS[@]}"; do wait ${P}; done

${PYTHON_BIN} "${EVAL}" --score_only --output_dir "${OUT}" --datasets ${DATASETS} \
    > "${OUT}/logs/scoring.log" 2>&1
echo "DONE_SCORING"
