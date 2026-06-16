#!/usr/bin/env bash
# Self-study distillation on PG19 real long books (2026-06-16) — chunk_size=512, n_ctx=7.
# Derived from launch_distill_chunk512_AB_MASS0p5_s1234_diskB.sh, but switched to
# the PG19 per-book dataset (single books 60k-100k+ tokens, 78% >=32k) so the
# student sees genuinely long context, and to n_ctx=7 (group_len=(7+1)*512=4096)
# to exploit the longer windows. This is the PURE pg19 distillation arm — NO mass
# bias (vs the dolmino distillation arm), so it isolates pg19-vs-dolmino. Distill:
#   A: bidirectional KL (lambda=0.6) on teacher top-64 logits, and
#   B: 1-cos hidden matching on decoder layers 12,20,28,
# both read from the OFFLINE teacher cache distill_cache/pg19_512_nctx7.
#
# === USAGE ===================================================================
# Teacher cache already built (B200 wzc1): distill_cache/pg19_512_nctx7
#   meta.json: n_ctx=7, chunk_size=512, distill_layers=[12,20,28], topk=64,
#   group_len=4096, dataset_fingerprint=f29ccf61ef017440, dataset_num_docs=2850.
# Cache keys MUST match (--chunk_size 512 / --curriculum 0:7 / --distill_layers
# 12,20,28 / pg19_perbook dataset) or the training script refuses to start.
# Run:  setsid bash scripts/launch_distill_pg19_chunk512_nctx7.sh
# ==============================================================================
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
export PYTORCH_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="distill_pg19_chunk512_nctx7"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=29975 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps 500 --lr 1e-4 --warmup_steps 50 \
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
  --l_recon_weight 0.0 --route_aux_weight 1.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --distill_logits --distill_hidden --distill_lambda 0.6 --distill_layers 12,20,28 \
  --distill_weight 1.0 --distill_hidden_beta 1.0 --distill_cache_dir distill_cache/pg19_512_nctx7 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 250 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
