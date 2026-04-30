#!/bin/bash
# launch_swa_stage1_4node.sh — 4-node DDP launch for swa_stage1_v3
#
# Topology:
#   b200-1 (28.89.17.143) — MASTER (NODE_RANK=0)
#   b200-2 (28.89.17.144) — NODE_RANK=1
#   b200-3 (28.89.17.85)  — NODE_RANK=2
#   b200-4 (28.89.19.134) — NODE_RANK=3
#
# All nodes share /apdcephfs_wzc1 — no rsync needed.
# Run this script from b200-1 (master node).
#
# 2026-04-27: v1 killed step 2730 (GPU mem-bw 17-21%, single node)
#             v2 killed step ~6900: NIAH acc=0.000 (niah_loader batch_size=1 bug)
#             v3: fixed batch loading (pg19 full batch_size=2, NIAH batch_size=1 separate)
#                 added --skip_chunks 0 (pg19_chunks_llama3.npy has 5916 chunks < default 40000)

set -euo pipefail

MASTER_ADDR="28.89.17.143"
MASTER_PORT="29500"
NNODES=4
NPROC_PER_NODE=8
WORKDIR="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory"
PASSWORD_FILE="/apdcephfs_wzc1/share_303098609/pighzliu_code/Mixture-of-Memory/configs/password.txt"
CONDA_ENV="torch-base"
OUTPUT_DIR="outputs/swa_stage1_v3"
LOG_DIR="$WORKDIR/logs"

TRAIN_ARGS="--model /apdcephfs_wzc1/share_303098609/pighzliu_code/models/Llama--Llama3-8b \
  --data data/pg19_chunks_llama3.npy \
  --swa_window 512 \
  --seq_len 4096 \
  --niah_mix_fraction 0.10 \
  --niah_max_N 16 \
  --max_steps 30000 \
  --lr 3e-4 \
  --num_slots 512 \
  --top_k 64 \
  --shared_memory_bank \
  --unfreeze_hidden_to_slot \
  --writeback_warmup_steps 500 \
  --writeback_gate_max 0.3 \
  --load_balance_weight 0.01 \
  --batch_size 2 \
  --skip_chunks 0 \
  --output_dir $OUTPUT_DIR"

mkdir -p "$LOG_DIR"
TS=$(TZ='Asia/Shanghai' date +"%Y%m%d_%H%M")

echo "[$(TZ='Asia/Shanghai' date)] Launching swa_stage1_v3 on 4 nodes x 8 GPUs = 32 GPUs"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Helper: build torchrun command for a given node rank
make_torchrun_cmd() {
    local RANK=$1
    echo "source /opt/conda/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && \
  cd ${WORKDIR} && \
  torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank=${RANK} \
    scripts/train_mem_space_pg19.py ${TRAIN_ARGS}"
}

# Launch worker nodes (rank 1/2/3)
WORKER_IPS=("28.89.17.144" "28.89.17.85" "28.89.19.134")
WORKER_RANKS=(1 2 3)

for i in "${!WORKER_IPS[@]}"; do
    IP="${WORKER_IPS[$i]}"
    RANK="${WORKER_RANKS[$i]}"
    LOG="$LOG_DIR/swa_stage1_v3_node${RANK}_${TS}.log"
    CMD=$(make_torchrun_cmd $RANK)
    echo "[$(TZ='Asia/Shanghai' date)] Starting node rank=$RANK at $IP -> log: $LOG"
    sshpass -f "$PASSWORD_FILE" ssh -o StrictHostKeyChecking=no -o ConnectTimeout=15 \
        root@"$IP" \
        "nohup bash -c \"${CMD}\" > ${LOG} 2>&1 & echo \$!" &
done

# Small delay to let workers start connecting
sleep 5

# Launch master (rank 0) in foreground
LOG_MASTER="$LOG_DIR/swa_stage1_v3_node0_${TS}.log"
echo "[$(TZ='Asia/Shanghai' date)] Starting master (rank 0) -> log: $LOG_MASTER"
CMD_MASTER=$(make_torchrun_cmd 0)
bash -c "$CMD_MASTER" 2>&1 | tee "$LOG_MASTER"

echo "[$(TZ='Asia/Shanghai' date)] swa_stage1_v3 master exited."
