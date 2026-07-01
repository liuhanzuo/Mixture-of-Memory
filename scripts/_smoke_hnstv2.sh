#!/bin/bash
# HNST v2 SMOKE (2026-06-25): trainable tree-summary + unfrozen reader, ~40 steps
# on 1 GPU to confirm (a) loss decreases, (b) select_ce is non-zero & falling,
# (c) tree pool params receive gradient. NOT a real training run.
# Red lines: babilong_mix_fraction 0; all-synthetic; warm-start from CLEAN A-model
# step2000 (never a leaked ckpt).
set -euo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_hnstv2_smoke"
GPU="${1:-0}"
mkdir -p logs
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[smoke] $RUN on GPU $GPU (HNST v2 trainable tree + unfreeze reader, 40 steps)"
CUDA_VISIBLE_DEVICES=$GPU $PY scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 40 --lr 3e-5 --warmup_steps 5 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --use_tree_summary --tree_summary_layers 1 --tree_summary_heads 8 \
  --t2_tree_branch 8 --t2_tree_beam 2 \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 2 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 1.0 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 1 --t2_gap_tokens 3584 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:6=1.0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 1000 --eval_interval 0 --log_interval 2 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42 \
  2>&1 | tee logs/$RUN.log
echo "smoke done -> logs/$RUN.log"
