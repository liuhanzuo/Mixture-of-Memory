#!/usr/bin/env bash
# MINIMAL single-node smoke: prove the Hunyuan-A13B keep13+fresh2 trainer emits steps
# on lhz (8x H200). seq_len 512, --max_rows 2000 subset, max_steps 50, log_every 1.
# NOT the real run — just fast step1/2/3 loss + s/step to rule out "trainer/FSDP broken".
set -euo pipefail
cd /volume/haru/Mixture-of-Memory
export WANDB_MODE=offline OMP_NUM_THREADS=16 TOKENIZERS_PARALLELISM=false PYTHONPATH="$PWD"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_NCCL_ENABLE_MONITORING=0 TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=7200 NCCL_TIMEOUT=7200
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
NPROC="${NPROC:-8}"
SMOKE_SEQ="${SMOKE_SEQ:-512}"
SMOKE_TAG="${SMOKE_TAG:-}"
PYTHON_BIN="${PYTHON_BIN:-.venv_hy3/bin/python}"
LOG="logs/hunyuan_a13b_keep13_fresh2_smoke${SMOKE_TAG}.log"
mkdir -p logs outputs/hunyuan_a13b_keep13_fresh2_smoke
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc_per_node "$NPROC" \
  scripts/train_hunyuan_a13b_probe2.py \
  --model_path models/Hunyuan-A13B-Pretrain \
  --data_path data/slimpajama_chunks_2048_hunyuan.npy \
  --output_dir outputs/hunyuan_a13b_keep13_fresh2_smoke \
  --keep_front_layers 13 --n_fresh_layers 2 \
  --max_steps 50 --seq_len "$SMOKE_SEQ" --max_rows 2000 \
  --batch_size 1 --grad_accumulation_steps 8 \
  --lr 1e-4 --lr_inherited 2e-5 --warmup_steps 10 \
  --save_every 100 --log_every 1 --gradient_checkpointing 1 \
  --fsdp_cpu_offload 0 --wandb 0 \
  > "$LOG" 2>&1
echo "A13B_SMOKE_EXIT_$?" >> "$LOG"
