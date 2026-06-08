#!/usr/bin/env bash
# [F2] Long-document training — chunk1024 ARM (parallel control vs chunk512 on .196).
#
# Goal: same F2 long-doc subset (wiki min4k), same F1-best P11 recipe, but chunk_size
# 1024 instead of 512. This is a single-variable parallel arm to the .196 chunk512 F2
# run: it probes the chunk_size x #chunk interaction on long docs (at chunk1024 each
# long doc still produces many chunks, but fewer than chunk512 -> coarser memory write
# granularity). Per F1 v3 convention warmup/accum scale ~ 1/chunk relative to chunk512
# (warmup 600->300, accum 4->2 keeps effective grad-token/step constant), sigma 3.0.
# Everything else byte-for-byte identical to launch_f2_longdoc_chunk512_diskA.sh.
#
# DISK B USAGE (.249 = 28.59.33.249, 8x H20, share_304376610 = separate ceph from
# diskA -> long-doc data must be rsynced first to
# MemLong/data/processed/dolmino_longdoc_wiki_min4k). Uses project .venv (torch 2.10
# + transformers, H20 OK):
#   PROJECT_ROOT=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory \
#   bash scripts/launch_f2_longdoc_chunk1024_diskB.sh
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

NPROC_PER_NODE="${NPROC_PER_NODE:-8}"

export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1

GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-5000}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
PORT="${MASTER_PORT:-29851}"

DOLMINO_PATH="${DOLMINO_PATH:-MemLong/data/processed/dolmino_longdoc_wiki_min4k/train}"
OUT_DIR="${OUT_DIR:-outputs/f2_longdoc_chunk1024}"
LOG="logs/f2_longdoc_chunk1024.log"
mkdir -p logs "$OUT_DIR"

if [[ ! -d "$DOLMINO_PATH" ]]; then
  echo "[FATAL] F2 long-doc dataset missing: $DOLMINO_PATH" | tee -a "$LOG"
  exit 1
fi

echo "=== [F2] long-doc chunk1024 train start $(date) ===" | tee -a "$LOG"
echo "[F2] data=$DOLMINO_PATH out=$OUT_DIR port=$PORT total_steps=$TOTAL_STEPS" | tee -a "$LOG"

CUDA_VISIBLE_DEVICES="$GPUS" "$PYBIN" -m torch.distributed.run \
  --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" --master_port="$PORT" \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path "$DOLMINO_PATH" \
  --output_dir "$OUT_DIR" --total_steps "$TOTAL_STEPS" --lr 1e-4 --warmup_steps 300 \
  --chunk_size 1024 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --use_delta_rule_writeback --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --loss_spike_skip --loss_spike_sigma 3.0 \
  --save_interval "$SAVE_INTERVAL" --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name "f2_longdoc_chunk1024" --dtype bfloat16 --attn_impl sdpa --seed 42 \
  >>"$LOG" 2>&1

echo "=== [F2] long-doc chunk1024 train DONE $(date) ===" | tee -a "$LOG"
