#!/usr/bin/env bash
# L1-only ablation (2026-06-10): P11 chunk512 delta-rule + normalized-readout recipe,
#   with the L3 summary channel DISABLED. Runs on B200 (28.89.18.188, 8x L20A 183GB).
#
# GOAL: isolate the net contribution of the L3 summary pool. There is currently NO
#   data on whether the L3 summary channel helps at all. This is a clean single-variable
#   ablation: IDENTICAL to launch_mem_space_p11_chunk512_remote196.sh EXCEPT the L3
#   summary group is removed:
#     - DROPPED: --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8
#       (use_l3_summary is an action flag; not passing it => L3 off.)
#     - --l3_diversity_weight 0.0 / --l_recon_weight 0.0 kept (already 0, inert w/ L3 off).
#   Every other hyperparameter is byte-for-byte identical to P11 chunk512.
#
# Baseline to compare against: outputs/mem_space_p11_chunk512_deltarule_normreadout
#   (same recipe WITH L3). Compare step500 BABILong to read off L3's net effect.
#
# B200 adaptation (mirrors launch_d1_slotdim16384_b200.sh):
#   1. PROJECT_ROOT -> B200 wzc1 disk; PYBIN -> wzc1 .venv (torch 2.10+cu128, L20A sm_100).
#   2. unique master_port (29795; D1 uses 29793) + isolated output_dir/log.
#   3. HF fully offline (B200 wzc1 has no direct internet) + woa proxy fallback.
#   4. batch: L20A 183GB; L3 off => even more headroom than P11, so keep P11's
#      bs4 x grad_accum1 x 8gpu = eff_batch 32.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="mem_space_p11_chunk512_noL3_L1only"
mkdir -p logs
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29795 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 5000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 4 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --use_delta_rule_writeback --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 1 \
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
