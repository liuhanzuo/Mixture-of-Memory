#!/usr/bin/env bash
# Per-doc CPT arm2 (remote H20-2, 28.59.80.196): chunk_size=128 + route_aux ON.
# Replaces the prior chunk256_remote arm (killed 2026-06-04 per user). Now the
# two arms differ ONLY in --route_aux_weight (local chunk128 = 0 / OFF, this = 1.0
# / ON), turning the ablation into a clean routing-supervision on/off contrast.
# Goal: route_aux should lift top1_sim off the noise floor (E2/E5: top1_sim
# 0.015->0.10+, key_max_cos 0.47->0.58) where the OFF arm stays ~0.04-0.27.
# wandb ONLINE: remote 28.59.80.196 CAN reach wandb.ai via hy-proxy:3128 (verified
# 2026-06-04, HTTP 200/404), so this run shows up live on the web dashboard.
# eval_interval=0: inline BABILong eval causes NCCL desync/SIGABRT (CODEBUDDY.md).
set -euo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE=online
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-/opt/conda/envs/torch-base/bin/python}"
RUN="mem_space_perdoc_chunk128_routeaux_remote"
setsid bash -c "$PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29782 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 128 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.01 --entropy_aux_weight 0.001 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
