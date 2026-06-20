#!/usr/bin/env bash
# mem_space multi-layer-readout FIX experiment — ARM A2 (layer-coverage axis).
# 2026-06-20, landmark-repro. Group-A diskA (本机 8×H20).
#
# Tests the LAYER-COVERAGE hypothesis from S5 (single-layer readout = 0% long-range
# cliff): does widening the raw-KV readout from 3 layers (16,20,24 = current Method A
# A1, still崩) to the full upper half (16-31, 16 layers) rescue cross-chunk W0?
#
# SINGLE AXIS vs A1 (Method A h1fix): ONLY --rawkv_readout_layers changes
# (16,20,24 -> 16,17,...,31). Everything else held = Method A h1fix recipe:
#   raw-KV lossless readout, unfreeze L16-31, last_chunk_loss_only (loss axis held
#   for A2; the loss axis is tested separately in arm B1), T2+pg19 0.5 mix,
#   gist_pool max, topk_chunks 2, lr 2e-5 wd 0.1, 2000 steps save 500.
# RETRAIN (not eval-only): layers 17,18,19,21,... were unfrozen in A1 but their
# attention never saw the concatenated raw-KV during A1 training (only 16/20/24
# did). Eval-only would feed them OOD inputs -> not a fair layer-coverage test.
# Reader consumes retrieved KV through each layer's OWN native k/v_proj (must be
# TRAINED on the concat path), so widening the set requires retraining.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-mem_space_A2_readout16to31_diskA}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29611}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
# Full upper-half readout: layers 16..31 (16 layers) — the A2 single-axis change.
RO_LAYERS="${RO_LAYERS:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers $RO_LAYERS \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks 2 --rawkv_readout_temp 1.0 \
  --rawkv_gist_pool max --rawkv_gist_lr_mult 1.0 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum 0:16 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy \
  --t2_gap_tokens 8192 --t2_num_keys 3 \
  --save_interval $SAVE_INTERVAL --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
