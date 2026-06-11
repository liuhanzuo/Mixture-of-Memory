#!/usr/bin/env bash
# lowrankgate_slot4096 (2026-06-11): single-variable control on B200 (28.89.18.188, 8x L20A 183GB).
#
# GOAL: isolate the effect of writeback_mode=lowrank_gate(r=256) ALONE, holding
#   slot_dim at the default 4096. This closes the D1 dual-variable confound:
#     - D1 = slot_dim 16384 + lowrank_gate(r=256)  (two variables changed vs P11)
#     - P11 = slot_dim 4096 + dual_gate            (the base recipe)
#     - THIS run = slot_dim 4096 + lowrank_gate(r=256)
#   Comparing THIS vs P11 isolates lowrank_gate's own effect (gate parameterisation);
#   comparing THIS vs D1 isolates the slot_dim 4096->16384 effect.
#   RUN_REGISTRY section 4 long-flagged "lowrank_gate@slot4096 control missing".
#
# This run is IDENTICAL to launch_d1_slotdim16384_b200.sh EXCEPT:
#   1. --slot_dim 16384 REMOVED  (restores default 4096 = backbone d_model).
#   2. RUN name lowrankgate_slot4096, isolated output_dir/log, master_port 29825.
#   3. total_steps 1000, save_interval 500 (new user convention; D1 was 5000/500).
# lowrank_gate writeback (r=256), delta_rule + normalized readout, eff_batch 32
# (bs1 x grad_accum4 x 8gpu) all PRESERVED from the D1 B200 script.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
# B200 wzc1 has no direct internet; babilong cache (0k-32k) is already complete
# under .hf_cache, so force HF fully offline to avoid the metadata-fetch hang at
# "Pre-fetching BABILong cache..." (observed 2026-06-09: rank0 hung w/o proxy).
# Proxy still exported as a belt-and-suspenders fallback (woa proxy reaches HF).
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
RUN="lowrankgate_slot4096"
mkdir -p logs
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29825 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 1000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --writeback_mode lowrank_gate --lowrank_gate_rank 256 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --use_delta_rule_writeback --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
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
