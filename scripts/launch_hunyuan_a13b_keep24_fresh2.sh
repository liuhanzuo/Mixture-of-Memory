#!/usr/bin/env bash
# 混元 Hunyuan-A13B-Pretrain 剪层(keep24)+补层(fresh2) continue-train — 8卡FSDP
# 修复史(2026-07-12): (1)NCCL init竞争 commit 1d23b4b; (2)心跳监控器480s误杀→MONITORING=0;
#   (3)FSDP首forward tied-embedding死锁 commit 7f6f049(wrap前untie lm_head).
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
export WANDB_MODE=offline OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD PATH=/opt/conda/bin:$PATH
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
.venv_hy3/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_hunyuan_a13b_probe2.py \
  --data_path data/slimpajama_chunks_2048_hunyuan.npy \
  --output_dir outputs/hunyuan_a13b_keep24_fresh2 \
  --keep_front_layers 24 --n_fresh_layers 2 --max_steps 200 \
  --seq_len 2048 --batch_size 1 --grad_accumulation_steps 8 \
  --lr 1e-4 --lr_inherited 2e-5 --warmup_steps 20 \
  --log_every 1 --save_every 100 --gradient_checkpointing 1 \
  > logs/hunyuan_a13b_keep24_fresh2.log 2>&1
echo "A13B_KEEP24F2_EXIT_$?" >> logs/hunyuan_a13b_keep24_fresh2.log
