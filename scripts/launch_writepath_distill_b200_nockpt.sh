#!/usr/bin/env bash
# #142 write-path distill — B200 (.252) variant: gradient_checkpointing OFF for speed.
# Restart-from-0 (trainer has no training-state resume). Distinct output_dir + port so it
# never collides with the H20 (.104) run or the #103 crossing-PPL eval also hosted on .252.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_MODE=offline
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RUN=qcmem_writepath_distill_qwen_j12_r32_b200
OUTPUT_DIR="outputs/$RUN"
mkdir -p logs "$OUTPUT_DIR"
echo "[launch] RUN=$RUN grad_ckpt=OFF batch=24 steps=4000 pybin=$PYBIN"
exec $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29974 \
  scripts/train_qcmem_writepath_distill.py \
  --model_path models/Qwen3-8b-local \
  --read_lora_path outputs/qcmem_distill_qwen_j12_r32_4k/final \
  --resume_j 12 --lora_rank 32 --lora_alpha 64 \
  --chunk_size 512 --n_ctx 3 --batch_size 24 \
  --teacher_topk 64 --distill_lambda 0.6 --ce_weight 0.0 \
  --total_steps 4000 --lr 8e-5 --warmup_steps 100 --grad_accum 1 \
  --max_steps_smoke 0 \
  --output_dir "$OUTPUT_DIR" --save_interval 500 --log_interval 10 \
  --keep_last_n "${KEEP_LAST_N:-3}" --keep_steps "${KEEP_STEPS:-1000,1500,2000}" \
  --dtype bfloat16 --attn_impl sdpa --seed 42 \
  --wandb_project mixture-of-memory --wandb_run_name "$RUN"
