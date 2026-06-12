#!/usr/bin/env bash
# P8 NULLSINK rerun (dedicated memory cross-attention READ path + null/sink slot).
# Built from launch_mem_space_p8.sh. Two changes vs the original P8 arm:
#   1. Code: commit 1f46b4d added a null_key/null_value sink to
#      MemoryCrossAttentionRead so the read softmax has an "attend to nothing"
#      escape valve (fixes the always-sums-to-1 full-magnitude-V injection root
#      cause in 20260605_p8_xattn_regression.md).
#   2. Trainer fix (commit c69cd8d): memory_xattn params are now COLLECTED by
#      _mem_space_params + SAVED in the adapter fragments whitelist. Original P8
#      left the whole xattn read path (incl. zero-init null_value) FROZEN, so it
#      injected pure random-init noise and the sink could never learn. This is
#      the FIRST P8 run where memory_xattn is actually trainable + persisted.
# Run on 4 GPUs (CUDA_VISIBLE_DEVICES=0,1,2,3); GPUs 5,6 are running the P8 eval
# tail and MUST NOT be touched. Effective batch preserved at 32 via ga=8
# (4 GPUs x bs1 x ga8 = 32, matching original 8x1x4=32).
# eval_interval=0 (inline BABILong eval -> NCCL SIGABRT, CODEBUDDY.md). Offline
# BABILong on the checkpoint is the judge.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="mem_space_perdoc_chunk128_p8_nullsink"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3 $PYBIN -m torch.distributed.run --nproc_per_node=4 --master_port=29786 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 128 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 8 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
