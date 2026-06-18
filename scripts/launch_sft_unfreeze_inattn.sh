#!/usr/bin/env bash
# Landmark-faithful FULL fine-tune SFT with TRUE in-attention K/V readout
# (2026-06-18). New direction approved by the user: the frozen reader CANNOT
# consume injected precise KV through an attention path it was never trained on
# (oracle-perfect needle still ≈ OFF → OOD mismatch). Landmark works because it
# trains END-TO-END on the injection/selection mechanism, so the model
# recalibrates to the injected-KV distribution. This run does exactly that:
#
#   * --unfreeze_backbone : the ENTIRE Llama-3-8B backbone (attn+FFN+embed+
#     lm_head) is trainable alongside the memory adapter.
#   * --use_inattn_kv : retrieved raw KV are concatenated DIRECTLY onto layer-16
#     self-attention's native K/V (ONE softmax over [native_KV ; retrieved_KV],
#     real source RoPE positions). The injection K/V are produced by the LIVE
#     k_proj/v_proj, so the softmax→o_proj path is differentiable → with the
#     backbone unfrozen the gradient flows INTO the injection and the model
#     learns to attend it (the key fix vs. all prior frozen-reader probes).
#
# Recipe (Landmark full-FT, 8B):
#   lr 2e-5 (≪ the 1e-4 adapter-only lr — full 8B FT), cosine + 3% warmup
#   (=30/1000 steps), weight_decay 0.1, AdamW betas (0.9,0.95), bf16.
#   gradient_checkpointing + FSDP FULL_SHARD (shards the 7B decoder stack +
#   embed/norm/lm_head optimizer state across 8 ranks — 8B full FT + Adam state
#   ≈ 64 GB fp32 will NOT fit per-GPU under plain DDP on H20 97 GB).
#   total_steps 1000 (EXPLORATORY — not Landmark's 15k; we look for a signal
#   first), save_interval 250.
#   Data: dolmino per_doc, chunk512, curriculum 0:3 (n_ctx=3 → ~2k effective ctx),
#   last_chunk_loss_only (pressure the memory channel), PURE LM loss (no aux /
#   no distill — Landmark shows the selection is learned end-to-end by LM loss).
#
# Memory-architecture hyperparams copied verbatim from the chunk512 P11/P8 recipe
# so this run is comparable to the adapter-only baselines (RUN_REGISTRY).
#
# ⚠️ NOT auto-launched at multi-GPU by sft-coder: smoke + mem feasibility were
#    verified first; team-lead reviews then launches the full 8-GPU run.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
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
RUN="sft_unfreeze_inattn_full"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=29951 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 1000 --lr 2e-5 --warmup_steps 30 \
  --weight_decay 0.1 \
  --unfreeze_backbone --use_fsdp \
  --use_inattn_kv --inattn_kv_layer 16 --inattn_kv_topk 64 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 1.0 --key_repulsion_threshold 0.3 --slot_value_norm_cap 5.0 \
  --normalize_readout \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum 0:3 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l3_diversity_threshold 0.5 \
  --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --use_memory_xattn --memory_xattn_gate_init 0.4 \
  --save_interval 250 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
