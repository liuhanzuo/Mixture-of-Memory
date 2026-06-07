#!/usr/bin/env bash
# Per-doc CPT arm1 (local 8x H20): chunk_size=128, intra-document chunk groups.
# Data = MemLong/data/processed/dolmino_per_doc/train (one complete doc per row).
# n_ctx fixed at 3 via curriculum -> group_len = (3+1)*128 = 512 tokens, all from
# the SAME document (context=3 chunks, target=1 chunk, adjacent in-doc).
# Recipe mirrors the established P1 norecon control (launch_dolmino_norecon_local.sh),
# only swapping chunk_size 1024->128 and adding --per_doc_data + per_doc path + curriculum 0:3.
# Pairs with arm2 chunk256 on a remote node (run by another teammate).
# eval_interval=0: inline BABILong eval causes NCCL desync/SIGABRT (CODEBUDDY.md).
set -euo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="$PROJECT_ROOT/.venv/bin/python"
RUN="mem_space_perdoc_chunk128"
TS="$(date +%Y%m%d_%H%M)"
LOG="logs/${RUN}_${TS}.log"
mkdir -p logs outputs/$RUN
# H20 batch tuning (2026-06-07): physical bs4 x grad_accum1 x 8gpu = eff_batch 32, held constant
# vs prior bs1 x grad_accum4. NOTE: physical H20 ceiling for chunk128 is much higher (~bs24, 88GiB;
# bs4=64GiB), but eff_batch-constant rule caps physical bs at 4 since grad_accum was only 4 (bs x ga
# must = 4 with integer ga). To go higher physical bs you'd have to raise eff_batch. Identical
# optimization dynamics (bs2==2xbs1 verified rel<1e-4) -> pure throughput win.
setsid bash -c "$PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29780 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 128 --batch_size 4 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.01 --entropy_aux_weight 0.001 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 1 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >"$LOG" 2>&1 &
echo "launched $RUN pid=$! log=$LOG"
