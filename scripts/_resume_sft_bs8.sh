#!/usr/bin/env bash
# Resume recipe-fix SFT with a NEW batch config (throughput test, 2026-07-04).
# User plan: at step250/500, switch to bs8 + grad_accum1 (effective batch
# 8*1*8gpu=64, == main-server level8 SFT scale) to speed up — IF bs8 fits in
# 183GB (bs4 already uses 111GB, so bs8 may OOM; this launcher is the test).
#
# Usage: RESUME_CKPT=<path> START_STEP=<n> [BS=8] [ACCUM=1] bash scripts/_resume_sft_bs8.sh
# Loads weights from RESUME_CKPT (warm-start, fresh optimizer), continues LR
# schedule/curriculum from START_STEP. Same recipe (dense-LM main + recall0.15,
# level8, buffer64, mix=0) — ONLY batch config changes.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_MODE="offline"
export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYBIN="$PROJECT_ROOT/.venv/bin/python"
PORT="${PORT:-29932}"
RUN="${RUN:-mem_space_sft_L8_denselm_recall15_bs8}"
RESUME_CKPT="${RESUME_CKPT:?set RESUME_CKPT=<path to full_model_stepNNNNNN.pt>}"
IACFG="${IACFG:-outputs/mem_space_sft_L8_denselm_recall15/adapter_config.json}"
START_STEP="${START_STEP:?set START_STEP=<n>}"
BS="${BS:-8}"
ACCUM="${ACCUM:-1}"
mkdir -p logs outputs/$RUN
if [ ! -f "$RESUME_CKPT" ]; then echo "ABORT: ckpt not found: $RESUME_CKPT"; exit 4; fi
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi

echo "[resume] $RUN bs=$BS accum=$ACCUM eff_batch=$((BS*ACCUM*8)) from $RESUME_CKPT @step$START_STEP"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run \
  --nproc_per_node=8 --master_port=$PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --init_checkpoint $RESUME_CKPT --init_adapter_config $IACFG \
  --start_step $START_STEP --total_steps 1500 --lr 3e-5 --warmup_steps 50 \
  --chunk_size 512 --batch_size $BS --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 64 --fifo_detach --last_chunk_loss_only \
  --unfreeze_backbone --unfreeze_layers_from 16 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps $ACCUM --curriculum 0:3 \
  --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.15 \
  --t2_background_data data/pg19_chunks_llama3_noeos.npy \
  --t2_num_keys 1 --t2_gap_tokens 4096 --t2_gap_mix 2048,4096,8192 \
  --t2_difficulty_curriculum 0:8=1.0 \
  --t2_select_loss_weight 0 \
  --save_interval 250 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN \
  --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$! log=logs/$RUN.log"
