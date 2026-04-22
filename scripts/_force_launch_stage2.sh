#!/bin/bash
# Aggressively kill any v10 processes and immediately launch Stage 2
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
source .venv/bin/activate
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Kill all competing training processes
pkill -9 -f "train_rmt_v10" 2>/dev/null || true
pkill -9 -f "run_train_v10" 2>/dev/null || true
sleep 2
pkill -9 -f "train_rmt_v10" 2>/dev/null || true
sleep 1

# Verify GPUs are free
echo "[$(date)] GPUs after cleanup:"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

# Launch Stage 2 immediately
echo "[$(date)] Launching Stage 2..."
exec torchrun --nproc_per_node=8 --master_port=29512 scripts/train_slot_memory.py \
  --data data/rmt_train_wiki_zh_10k.jsonl \
  --base_model ../models/Qwen--Qwen3-8b \
  --output_dir outputs/slot_memory_stage2_resume \
  --num_slots 16 \
  --slot_dim 256 \
  --slot_iterations 3 \
  --segment_length 1024 \
  --max_segments 6 \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --batch_size 1 \
  --grad_accumulation_steps 8 \
  --warmup_steps 30 \
  --log_every 10 \
  --ddp \
  --stage 2 \
  --resume_from outputs/slot_memory_8gpu_20260419_143943_20260419_144019/stage1/final \
  --seed 42
