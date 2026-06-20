#!/usr/bin/env bash
# ============================================================================
# Self-study distillation + raw-KV readout (direction-1, 2026-06-20, methodA-eval)
# ============================================================================
# Merges the self-study distillation objective (teacher = frozen Llama-3-8B over
# the FULL flat context; student = mem_space raw-KV readout) with the raw-KV
# readout + grouped two-stage readout recipe. The distillation target (match the
# full-context teacher's top-64 logits + layer-{12,20,28} hidden) is naturally
# leak-resistant: copying an adjacent token does NOT reproduce the teacher's
# long-range output distribution, so the readout MUST carry long-range info.
#
# Teacher cache: distill_cache/pg19_512_nctx15  (chunk_size=512, n_ctx=15,
#   distill_layers=12,20,28, group_len=8192, topk=64) — ALREADY BUILT (394G).
#   The trainer asserts cache meta matches (chunk_size/n_ctx/distill_layers);
#   curriculum 0:15 + chunk512 + distill_layers 12,20,28 below MATCH this cache.
#
# Runs on whatever 8-GPU node is free (set CUDA_VISIBLE_DEVICES / PROJECT_ROOT /
# PYTHON_BIN). Default local 8 GPU. Single-axis vs the pure-T2 grouped run: this
# swaps the T2/last-chunk-LM objective for the self-study distill objective.
# ============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-selfstudy_rawkv_chunk512_nctx15}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29655}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"      # frequent ckpt: avoid losing steps to external OOM
CHUNK="${CHUNK:-512}"
CACHE_DIR="${CACHE_DIR:-distill_cache/pg19_512_nctx15}"
# nctx15 cache => 15 context chunks + 1 target. curriculum must match.
CURRICULUM="${CURRICULUM:-0:15}"
RO_LAYERS="${RO_LAYERS:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
# Distillation layers MUST match the cache meta (12,20,28).
DISTILL_LAYERS="${DISTILL_LAYERS:-12,20,28}"

mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers $RO_LAYERS \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks 0 --rawkv_readout_temp 1.0 \
  --rawkv_gist_pool max --rawkv_gist_lr_mult 1.0 \
  --rawkv_grouped_readout --rawkv_subblock_size 64 \
  --chunk_size $CHUNK --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum $CURRICULUM --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --distill_logits --distill_hidden --distill_lambda 0.6 --distill_layers $DISTILL_LAYERS \
  --distill_weight 1.0 --distill_hidden_beta 1.0 --distill_cache_dir $CACHE_DIR \
  --save_interval $SAVE_INTERVAL --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42 --grad_flow_diag" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
