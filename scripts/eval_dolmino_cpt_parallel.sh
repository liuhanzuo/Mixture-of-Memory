#!/bin/bash
# Parallel BABILong eval on remote H20 — one GPU per checkpoint
# Evaluates Dolmino CPT checkpoints: step 25000, 30000, 35000

cd /apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
export PYTHONUNBUFFERED=1
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export PYTHONPATH="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory:/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory/third_party/babilong-pkg"

PYTHON=/opt/conda/envs/torch-base/bin/python
MODEL=models/Meta-Llama-3-8B
CKPT_DIR=outputs/dolmino_cpt_local_long
ADAPTER_CONFIG=${CKPT_DIR}/adapter_config.json
RESULTS=babilong_results/dolmino_cpt_eval

mkdir -p $RESULTS logs

# Checkpoints to evaluate
CKPTS=(
    "mem_space_adapter_step025000.pt:25k"
    "mem_space_adapter_step030000.pt:30k"
    "mem_space_adapter_step035000.pt:35k"
)

# Tasks and lengths for eval
TASKS="qa1 qa2 qa3 qa4 qa5"
LENGTHS="0k 1k 2k 4k 8k 16k 32k"

GPU=0
for entry in "${CKPTS[@]}"; do
    CKPT_FILE="${entry%%:*}"
    CKPT_NAME="${entry##*:}"

    echo "[$(date)] GPU $GPU: Evaluating $CKPT_NAME ($CKPT_FILE)"

    CUDA_VISIBLE_DEVICES=$GPU $PYTHON scripts/run_babilong_mem_space.py \
        --model_path $MODEL \
        --checkpoint ${CKPT_DIR}/${CKPT_FILE} \
        --adapter_config $ADAPTER_CONFIG \
        --results_folder $RESULTS \
        --output_name dolmino_cpt_${CKPT_NAME} \
        --tasks $TASKS \
        --lengths $LENGTHS \
        --chunk_size 1024 \
        > logs/eval_dolmino_cpt_${CKPT_NAME}.log 2>&1 &

    GPU=$((GPU + 1))
done

echo "[$(date)] Launched ${#CKPTS[@]} eval jobs on GPUs 0-$((GPU-1))"
echo "Waiting for all jobs to finish..."
wait
echo "[$(date)] All eval jobs completed!"

# Score results
echo "Scoring..."
$PYTHON scripts/score_babilong_results.py --results_dir $RESULTS 2>&1 || true
echo "[$(date)] Done!"
