#!/usr/bin/env bash
# D1 (2026-06-09): slot_dim=16384 controlled run on B200 (28.89.18.188, 8x L20A 183GB).
#
# GOAL: does scaling slot_dim 4096 -> 16384 improve BABILong, holding the P11
#   chunk512 delta-rule + normalized-readout recipe fixed?
#
# Baseline to compare against: outputs/mem_space_p11_chunk512_deltarule_normreadout
#   (num_slots=128, top_k=16, delta_rule writeback, normalize_readout, selector_temp=40,
#    chunk512, total_steps 5000, save_interval 500, eval_interval 0, eff_batch 32).
#
# This run is IDENTICAL to launch_mem_space_p11_chunk512_remote196.sh EXCEPT:
#   1. --slot_dim 16384                       (the variable under study; P11 = null = 4096)
#   2. writeback_mode: lowrank_gate (r=256)   instead of P11's dual_gate.
#      ---> UNAVOIDABLE CONFOUND: dual_gate at slot_dim=16384 needs two
#           Linear(16384, 32768) per layer x 32 layers ~= 34B params, which is
#           infeasible even on the 183GB B200. lowrank_gate (r=256) is the
#           cheapest content-conditioned dual-gate parameterisation that fits
#           (4*slot_dim*r/layer). The delta-rule writeback + normalized readout
#           are PRESERVED (lowrank_gate also routes through delta_rule). This is
#           the same gate the original (crashed) 16384 run used.
#   3. eff_batch held at 32 via bs1 x grad_accum4 x 8gpu (P11 = bs4 x ga1 x 8gpu;
#      bs scaling is exact per the P11 script note, so optimization dynamics match).
#      Smaller physical bs because slot_dim=16384 inflates activations + the
#      ~6.5B trainable mem_space params (52GB AdamW state, DDP-replicated).
#   4. PROJECT_ROOT -> B200 wzc1 disk; PYBIN -> wzc1 .venv (torch 2.10+cu128, L20A sm_100).
#   5. unique master_port (29793) + isolated output_dir/log.
#
# ROOT CAUSE of the original 16384 crash (logs/wbmode_lowrank.log:736): plain
#   CUDA OOM at window_loss.backward() on H20 (91.5GiB used / 95GiB cap) -- NOT a
#   shape/init bug. The 183GB B200 has the headroom H20 lacked.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="d1_slotdim16384"
mkdir -p logs
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29793 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 5000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --slot_dim 16384 --writeback_mode lowrank_gate --lowrank_gate_rank 256 \
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
