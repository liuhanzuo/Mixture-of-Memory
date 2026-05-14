#!/bin/bash
# Launch continued pretraining across 4 nodes via multi-node DDP.
# Usage: bash scripts/launch_continued_pretrain.sh [EXTRA_ARGS...]
#
# This script:
# 1. Launches workers on b200-2/3/4 via SSH
# 2. Launches master on local (b200-1)
#
# All nodes share CephFS at /apdcephfs_wzc1/share_303098609/pighzliu_code/

set -e

cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory

MASTER_IP="28.89.17.143"
MASTER_PORT=29500
NNODES=4
NPROC_PER_NODE=8
RDZV_ID="continued_pretrain_$(date +%Y%m%d_%H%M%S)"

# Default args
SHARD_DIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/data/dolmino-mix-1124-llama3"
MODEL="/apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b"
OUTPUT_DIR="outputs/continued_pretrain_dolmino"
WIKITEXT="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/data/wikitext_chunks_llama3_4096.npy"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs"

mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

# SSH config
SSH_OPTS="-o StrictHostKeyChecking=no -o ConnectTimeout=10"
SSH_PASS="sshpass -f /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password.txt"

# Common torchrun command (everything after --)
COMMON_CMD="scripts/train_continued_pretrain.py \
    --shard_dir ${SHARD_DIR} \
    --model ${MODEL} \
    --output_dir ${OUTPUT_DIR} \
    --wikitext_path ${WIKITEXT} \
    --num_shards 100 \
    --seq_len 4096 \
    --chunks_per_doc 32 \
    --num_slots 64 \
    --top_k 8 \
    --lora_rank 16 \
    --lr 3e-5 \
    --warmup_steps 200 \
    --max_steps 2000 \
    --gradient_accumulation_steps 4 \
    --eval_interval 200 \
    --save_interval 500 \
    --kl_weight 0.1 \
    $@"

TORCHRUN_BASE="torchrun --nnodes=${NNODES} --nproc_per_node=${NPROC_PER_NODE} \
    --rdzv_backend=c10d --rdzv_endpoint=${MASTER_IP}:${MASTER_PORT} \
    --rdzv_id=${RDZV_ID}"

echo "=============================================="
echo "Continued Pretraining - Multi-node DDP"
echo "=============================================="
echo "Master: ${MASTER_IP}:${MASTER_PORT}"
echo "Nodes: ${NNODES} x ${NPROC_PER_NODE} GPUs = $((NNODES * NPROC_PER_NODE)) GPUs"
echo "Data: 100 Dolmino shards (~10B tokens)"
echo "Output: ${OUTPUT_DIR}"
echo "Log: ${LOG_DIR}/continued_pretrain_${TIMESTAMP}.log"
echo "RDZV ID: ${RDZV_ID}"
echo "=============================================="

# Function to launch a worker node
launch_worker() {
    local IP=$1
    local NAME=$2
    echo "Launching worker on ${NAME} (${IP})..."
    ${SSH_PASS} ssh ${SSH_OPTS} root@${IP} \
        "cd /apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory && \
         nohup ${TORCHRUN_BASE} \
         --node_rank=$(get_node_rank ${IP}) \
         ${COMMON_CMD} \
         > ${LOG_DIR}/continued_pretrain_${NAME}_${TIMESTAMP}.log 2>&1 &" &
    echo "Worker ${NAME} launched (PID=$!)"
}

get_node_rank() {
    case $1 in
        28.89.17.143) echo 0 ;;
        28.89.17.144) echo 1 ;;
        28.89.17.85)  echo 2 ;;
        28.89.19.134) echo 3 ;;
        *) echo 99 ;;
    esac
}

# Launch workers first (in background)
launch_worker 28.89.17.144 "b200-2"
launch_worker 28.89.17.85 "b200-3"
launch_worker 28.89.19.134 "b200-4"

# Give workers a few seconds to connect
sleep 5

# Launch master (local, foreground with tee)
echo "Launching master on local (b200-1)..."
echo "Following master log. Workers log to ${LOG_DIR}/continued_pretrain_b200-*_${TIMESTAMP}.log"

${TORCHRUN_BASE} --node_rank=0 ${COMMON_CMD} \
    2>&1 | tee "${LOG_DIR}/continued_pretrain_b200-1_${TIMESTAMP}.log"

echo "Training complete."
