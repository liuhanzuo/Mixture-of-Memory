#!/usr/bin/env bash
# supervised-selection 双机16卡DDP (.7.53 master + .245 worker, 共享diskB FS)
# 用法: 在每个节点上跑, 传 NODE_RANK (0=master .7.53, 1=worker .245)
#   bash _launch_t2_supervised_16gpu.sh <NODE_RANK>
# grad_accum 4→2 (16卡 global batch 同, 等效~2x吞吐 vs 单机8卡)
set -euo pipefail
NODE_RANK="${1:?need NODE_RANK 0 or 1}"
MASTER="28.48.7.53"; MPORT=29850
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"; cd "$R"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"; export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,28.48.7.53,28.58.245.174"
export HF_HOME="$R/.hf_cache"; export HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# ★NCCL跨机: 指定网卡(bond, 避免选错挂死) + 超时
export NCCL_SOCKET_IFNAME=bond0,bond2,bond8,eth0
export NCCL_IB_DISABLE=0; export NCCL_DEBUG=WARN
PY="$R/.venv/bin/python"
RUN="mem_space_fifo_b25_c512_supervised_select_16gpu"
mkdir -p logs
echo "[launch16] node_rank=$NODE_RANK master=$MASTER:$MPORT run=$RUN"
setsid bash -c "$PY -m torch.distributed.run \
  --nnodes=2 --nproc_per_node=8 --node_rank=$NODE_RANK \
  --rdzv_backend=c10d --rdzv_endpoint=$MASTER:$MPORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 3000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 2 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 3 --t2_gap_tokens 3584 --t2_background_skip 0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/${RUN}_node${NODE_RANK}.log 2>&1 &
echo "launched node$NODE_RANK pid=$!"
