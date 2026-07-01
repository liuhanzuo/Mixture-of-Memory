#!/usr/bin/env bash
# P1 recon arm: identical to local dolmino_bugfix_slotq_t2h but WITH summary-reconstruction
# aux loss (--l_recon_weight 0.1). Local run = no-recon control; this = recon arm.
# Target: remote disk-A node 28.59.80.196 (shares share_303098609 FS, torch-base env).
set -euo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_MODE=offline
export WANDB_API_KEY="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"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="/opt/conda/envs/torch-base/bin/python"
RUN="dolmino_recon_diskA"
setsid bash -c "$PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29765 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --dolmino_path MemLong/data/processed/dolmino_0.5B_1024/train \
  --output_dir outputs/$RUN --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 1024 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.01 --entropy_aux_weight 0.001 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
  --curriculum 0:2,250:4,1000:8 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.1 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
