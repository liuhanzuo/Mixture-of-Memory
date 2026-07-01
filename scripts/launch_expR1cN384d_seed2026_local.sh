#!/usr/bin/env bash
# EXP-R1c N384d (2026-06-12): 4th independent N384 seed for VARIANCE ATTRIBUTION.
# Identical config to N384/N384b/N384c (P11 chunk512 base + cumulative dead-slot
# judge + num_slots 384) EXCEPT --seed 2026 and run name. Goal: build a >=4-seed
# ensemble of the same N384 config to decide whether the N384 "breakthrough"
# (qa5 85/68/74/53/39/42/30) is a real capacity sweet-spot or single-run noise.
# Runs on LOCAL 8xH20 (disk-A, share_303098609), project .venv python.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="expR1cN384d_cum_slots384"
# H20 chunk512: physical bs4 x grad_accum1 x 8gpu = eff_batch 32 (same as P11).
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29860 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 1000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 4 --num_slots 384 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --use_delta_rule_writeback --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 1 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --dead_slot_reset_interval 8 --dead_slot_grace_chunks 1 --dead_slot_reset_mode strided_current \
  --dead_slot_criterion cumulative \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 2026" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
