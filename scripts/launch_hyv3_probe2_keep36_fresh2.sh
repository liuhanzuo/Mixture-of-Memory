#!/usr/bin/env bash
# Hy-MT2-30B-A3B (hy_v3, 48层) 剪层(keep36)+补层(fresh2) continue-train — 8卡FSDP on-GPU
# coder commit dcb2691. 优势: 30B on-GPU全速(~0.9s/step 2卡smoke, vs A13B 65B cpu_offload 113s/step).
# tie_emb=False无untie坑. 数据用Hy-MT2 tokenizer预处理(vocab120832, 避A13B tokenizer越界).
cd /apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
export WANDB_MODE=offline OMP_NUM_THREADS=8 TOKENIZERS_PARALLELISM=false PYTHONPATH=$PWD PATH=/opt/conda/bin:$PATH
export HF_HOME=$PWD/.hf_cache HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
.venv_hy3/bin/python -m torch.distributed.run --standalone --nproc_per_node 8 \
  scripts/train_hyv3_probe2.py \
  --data_path data/slimpajama_chunks_2048_hymt2.npy \
  --output_dir outputs/hyv3_probe2_keep36_fresh2 \
  --keep_front_layers 36 --n_fresh_layers 2 \
  --max_steps 200 --batch_size 1 --grad_accumulation_steps 8 --log_every 1 \
  --seq_len 2048 --lr 1e-4 --lr_inherited 2e-5 --warmup_steps 20 \
  --save_every 100 --gradient_checkpointing 1 --fsdp_cpu_offload 0 \
  > logs/hyv3_probe2_keep36_fresh2.log 2>&1
echo "HYV3_KEEP36F2_EXIT_$?" >> logs/hyv3_probe2_keep36_fresh2.log
