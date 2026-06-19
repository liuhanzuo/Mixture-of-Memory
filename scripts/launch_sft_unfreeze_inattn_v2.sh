#!/usr/bin/env bash
# Landmark-faithful SFT v2 (2026-06-19). Bundles the 3 fixes that address v1's
# confounds (v1 = full-unfreeze + single-layer inject + short dolmino → DECISIVE
# DOUBLE-NEGATIVE: niah OFF 22→11 backbone DAMAGE, oracle 10≈OFF wall held):
#
#   1. PARTIAL unfreeze (--unfreeze_layers_from 16): only decoder layers 16-31 +
#      the memory adapter + final-norm + lm_head are trainable; layers 0-15 +
#      embed_tokens stay FROZEN. Protects the base LM's lower-level competence
#      (v1 full-unfreeze on short data halved niah) while letting the upper
#      layers (where injection lives) learn to consume injected KV. Also cuts the
#      FSDP optimizer state + first-step gather transient → fits <85GB/GPU.
#   2. MULTI-LAYER injection (--inattn_kv_layers 16,20,24): inject retrieved K/V
#      at THREE layers (all within the unfrozen 16-31 range so they can learn to
#      attend), not just L16. Landmark trains the readout mechanism into many
#      layers; v1's single injection layer under-parameterised what the reader
#      could learn. The store WRITE is owned by L16 (smallest idx); L16/L20/L24
#      each re-project the retrieved hidden through THEIR OWN k/v_proj + RoPE.
#   3. LONG-RANGE data (pg19_perbook_min8k, curriculum 0:7 → n_ctx=7 ~4k eff):
#      replaces dolmino's ~2k short docs (suspected cause of v1 backbone damage)
#      with real long books that actually exercise long-range dependency.
#
# Keep (from v1, verified): in-graph injection (K_raw.requires_grad=True), lr
# 2e-5 cosine + 3% warmup, wd 0.1, bf16, FSDP FULL_SHARD, gradient_checkpointing,
# PURE LM loss. total_steps 1500 (real attempt, look for signal; not full 15k).
# save_interval 500.
#
# Memory hyperparams copied verbatim from the chunk512 P11/P8 recipe (comparable
# to RUN_REGISTRY adapter-only baselines). Eval at step 500/1000/1500: oracle +
# OFF (niah_single_1 4k n=100) vs frozen-P11 baseline (OFF 22 / oracle 21).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-sft_unfreeze_inattn_v2}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-1500}"
MASTER_PORT="${MASTER_PORT:-29953}"
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps 45 \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_inattn_kv --inattn_kv_layer 16 --inattn_kv_layers 16,20,24 --inattn_kv_topk 64 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum 0:7 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 500 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
