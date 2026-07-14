#!/usr/bin/env bash
# Hunyuan-A13B-Pretrain (public 32-layer MoE) minimal-arch continue-train on lhz (8x H200).
#   keep_front_layers=13 (split-j = 13/32 ~= 0.4L, QCMem j-sweep; see HARU.md core result)
#   n_fresh_layers=2  (drop layers[13:], append 2 FRESH NTP decoder layers, continue-train)
#   backbone NOT frozen: front-13 inherited layers at low LR (2e-5), fresh tail at high LR (1e-4).
# max_steps=20000, seq_len=1024.
#   NOTE (2026-07-14): seq_len reduced 2048 -> 1024 to fit on 8x H200 (143GB). The
#   37.9B keep13+fresh2 with FULL_SHARD (no CPU offload) has ~76GB/rank persistent
#   (fp32 master 19 + fp32 grad 19 + AdamW m/v 38) + bf16 layer unshards + activations.
#   At seq_len=2048 it OOM'd mid-training even with BACKWARD_POST (peak ~136/140GB).
#   Halving seq_len ~halves the activation transient -> fits. This DEVIATES from the
#   Qwen/Hy3 seq_len=2048 alignment; re-raise to 2048 if we free memory later
#   (e.g. optimizer-only offload, or 16-card multi-node to shrink the per-rank shard).
#
# Fix history (2026-07-12, carried over from the old-cluster keep24 launcher):
#   (1) NCCL init race (rank0's ~10min transplant vs 600s PG timeout) -> the train
#       script inits the PG with timeout=2h + eager device_id comm formation.
#   (2) heartbeat monitor 480s false-kill -> TORCH_NCCL_ENABLE_MONITORING=0.
#   (3) FSDP first-forward tied-embedding deadlock -> untie lm_head before wrap (in-script).
#   (4) MoE grouped_mm kernel crash (GroupMMCommon.cuh:51 delta%16==0) on torch2.8-nv ->
#       train script now sets cfg._experts_implementation="eager" (numerically equivalent
#       per-expert index_add_ loop, no alignment constraint).
#   (5) 65B-on-8xH200 first-step OOM -> FSDP CPU offload (fp32 master + AdamW states on
#       CPU pinned RAM) is the train script default (--fsdp_cpu_offload 1).
set -euo pipefail
cd /volume/haru/Mixture-of-Memory

export WANDB_MODE=offline OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

PYTHON_BIN="${PYTHON_BIN:-.venv_hy3/bin/python}"
mkdir -p logs outputs/hunyuan_a13b_keep13_fresh2

"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_hunyuan_a13b_probe2.py \
  --model_path models/Hunyuan-A13B-Pretrain \
  --data_path data/slimpajama_chunks_2048_hunyuan.npy \
  --output_dir outputs/hunyuan_a13b_keep13_fresh2 \
  --keep_front_layers 13 --n_fresh_layers 2 \
  --max_steps 20000 --seq_len 1024 \
  --batch_size 1 --grad_accumulation_steps 8 \
  --lr 1e-4 --lr_inherited 2e-5 --warmup_steps 100 \
  --save_every 1000 --log_every 10 --gradient_checkpointing 1 \
  --fsdp_cpu_offload 0 \
  > logs/hunyuan_a13b_keep13_fresh2.log 2>&1
echo "A13B_KEEP13F2_EXIT_$?" >> logs/hunyuan_a13b_keep13_fresh2.log
