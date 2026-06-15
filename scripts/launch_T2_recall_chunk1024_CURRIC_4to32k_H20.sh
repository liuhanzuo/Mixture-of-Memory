#!/usr/bin/env bash
# T2 curriculum recall — chunk_size=1024, PROGRESSIVE T2 needle distance 4K->8K->16K->32K.
# Rationale (user 2026-06-15): training directly on a very long context, the model
# struggles to learn readout; ramp the needle->query distance gradually instead.
# CRITICAL FIX (2026-06-15): the curriculum drives ONLY the T2 stream (--t2_curriculum,
# pg19 long background supports n_ctx up to 32+). dolmino stays LOCKED at n_ctx=3
# (--curriculum 0:3) because per-doc dolmino data is capped at 4096 tokens; pushing it
# to n_ctx>=4 -> (n_ctx+1)*1024>4096 -> zero eligible docs -> loader starves -> the
# observed DDP first-step hang. T2 n_ctx = 4->8->16->32 (x chunk 1024 = 4K..32K).
# Equal data per stage: 16000 samples/stage. bs8 -> 128 samples/step -> 125 steps/stage.
# NOTE: eval must use --chunk_size 1024 to match memory write granularity.
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
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="T2_recall_chunk1024_CURRIC_4to32k_H20bs4ga4_N128"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29950 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 500 --lr 1e-4 --warmup_steps 12 \
  --chunk_size 1024 --batch_size 4 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 4 \
  --last_chunk_loss_only --curriculum 0:3 --t2_curriculum 0:4,125:8,250:16,375:32 \
  --bptt_window 2 \
  --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 1.0 --babilong_mix_fraction 0.0 \
  --t2_recall_mix_fraction 0.5 --t2_num_keys 1 --t2_gap_tokens 4096 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 125 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
