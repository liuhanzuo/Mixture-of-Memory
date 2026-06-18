#!/bin/bash
# Parallel LongBench eval of the two self-study distill ckpts.
# ckpt1 (AB_dolmino)        -> GPU 0-3
# ckpt2 (AB_MASS0p5_dolmino)-> GPU 4-7
# Matches P11 baseline 口径: base Meta-Llama-3-8B, no_chat_template, chunk512,
# W0 (memory-only), tasks={multifieldqa_en,2wikimqa,musique}, max_samples=100.
set -u

PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
PYTHON_BIN="/opt/conda/envs/torch-base/bin/python"
EVAL="${PROJECT_ROOT}/scripts/eval_longbench_mem_space.py"
MODEL="${PROJECT_ROOT}/models/Meta-Llama-3-8B"
DATASETS="multifieldqa_en 2wikimqa musique"
CHUNK=512
MAXS=100

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1

run_ckpt () {
    local CKPT="$1" ACFG="$2" OUT="$3" G0="$4"
    mkdir -p "${OUT}/logs"
    local PIDS=()
    for SH in 0 1 2 3; do
        local GPU=$((G0 + SH))
        CUDA_VISIBLE_DEVICES=${GPU} ${PYTHON_BIN} "${EVAL}" \
            --model_path "${MODEL}" \
            --checkpoint "${CKPT}" \
            --adapter_config "${ACFG}" \
            --output_dir "${OUT}" \
            --chunk_size ${CHUNK} \
            --gpu_id ${SH} \
            --num_gpus 4 \
            --datasets ${DATASETS} \
            --max_samples ${MAXS} \
            --no_chat_template \
            > "${OUT}/logs/gpu_${SH}.log" 2>&1 &
        PIDS+=($!)
    done
    echo "[${OUT}] shards launched on GPU ${G0}-$((G0+3)): ${PIDS[*]}"
    for P in "${PIDS[@]}"; do wait ${P}; done
    ${PYTHON_BIN} "${EVAL}" --score_only --output_dir "${OUT}" --datasets ${DATASETS} \
        > "${OUT}/logs/scoring.log" 2>&1
    echo "[${OUT}] DONE_SCORING"
}

OUT1="${PROJECT_ROOT}/longbench_results/distill_chunk512_AB_dolmino"
OUT2="${PROJECT_ROOT}/longbench_results/distill_chunk512_AB_MASS0p5_dolmino"

run_ckpt "${PROJECT_ROOT}/outputs/distill_chunk512_AB_dolmino/mem_space_adapter.pt" \
         "${PROJECT_ROOT}/outputs/distill_chunk512_AB_dolmino/adapter_config.json" \
         "${OUT1}" 0 &
PID_A=$!
run_ckpt "${PROJECT_ROOT}/outputs/distill_chunk512_AB_MASS0p5_dolmino/mem_space_adapter.pt" \
         "${PROJECT_ROOT}/outputs/distill_chunk512_AB_MASS0p5_dolmino/adapter_config.json" \
         "${OUT2}" 4 &
PID_B=$!
wait ${PID_A}; wait ${PID_B}
echo "ALL_DONE_SCORING"
