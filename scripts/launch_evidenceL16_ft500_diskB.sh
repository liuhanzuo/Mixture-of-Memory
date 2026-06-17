#!/usr/bin/env bash
# Stage-3 evidence fine-tune (2026-06-17): warm-start from frozen-backbone p11
# SOTA adapter and train the memory adapter with Slot-Routed Evidence ENABLED at
# evidence_layer=16 (oracle probe showed L16 34% > L0 28% — train the read
# pathway at the layer with most signal). Short go/no-go run: 500 steps.
# Recipe is IDENTICAL to launch_mem_space_p11_chunk1024_local.sh except:
#   + --init_checkpoint <p11 final adapter>  (warm start)
#   + --use_slot_evidence --evidence_buffer_size 64 --evidence_topr 64 --evidence_layer 16
#   + --total_steps 500 --save_interval 250 --warmup_steps 20
# Runs on diskB .76 (8x H20, .venv). PROJECT_ROOT/PYBIN default to diskB+.venv.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_MODE="offline"
# diskB has no direct internet. The babilong dataset is already in .hf_cache, so
# force HF fully offline (local-cache-only) — this avoids the rank0 startup hang
# where the babilong loader retries HF HEAD/dataset_infos 404s before step 1
# (proxy reaches HF but the repeated 404 retries stall the prefetch). Offline is
# simpler and the cache is present (see reference_h800_babilong_proxy).
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
# Evidence@L16 inflates the extended-attn seq at the evidence layer
# (T + k_ev = 1024 + (top_k16+global4)*topr); curriculum growth + bf16
# fragmentation caused intermittent "Cuda failure 2 out of memory" mid-run.
# expandable_segments reduces allocator fragmentation so the transient peak fits.
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
P11="outputs/mem_space_p11_chunk1024_deltarule_normreadout/mem_space_adapter.pt"
RUN="mem_space_p11_evidenceL16_ft500"
mkdir -p logs outputs
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29812 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --init_checkpoint $P11 \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 500 --lr 1e-4 --warmup_steps 20 \
  --chunk_size 1024 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --use_delta_rule_writeback --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
  --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --use_slot_evidence --evidence_buffer_size 64 --evidence_topr 64 --evidence_layer 16 \
  --save_interval 250 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
