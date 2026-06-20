#!/usr/bin/env bash
# ============================================================================
# Self-study distillation + raw-KV readout (方向1, 2026-06-20, methodA-eval)
# ============================================================================
# 用户方向1: teacher(frozen Llama-3-8B 看完整 context)的输出分布 vs student
# (只看 memory readout)逼一致 —— 天然抗 leak(匹配 teacher 长程输出分布,抄邻接
# token 满足不了),比纯 T2/keep_all 更接近真实目标。
#
# 这是 self-study distill(已有 v21 基础设施)+ raw-KV grouped readout(我们的
# 破墙机制)的合并:
#   - distill: --distill_logits(teacher top-64 KL)+ --distill_hidden(layer MSE)
#     teacher cache 离线已建(distill_cache/pg19_512_nctx15, chunk512/n_ctx15/
#     layers[12,20,28]/topk64),teacher = 纯 frozen backbone 看 flat 完整 context。
#   - student = raw-KV grouped readout(--use_rawkv_readout + --rawkv_grouped_readout
#     + keep_all),只经 memory 读出,被 distill loss 逼"复现 teacher 长程输出"。
#
# ★cache 一致性: meta.json 强制 chunk_size=512 / n_ctx=15 / distill_layers=12,20,28
#   完全匹配下面配置, 否则训练脚本 assert_distill_cache_consistent 会 refuse。
#   curriculum 锁 0:15(= cache n_ctx,不能动,否则 cache miss → distill 静默失效)。
#
# ★为什么可能绕开 leak(对比 A/纯T2):
#   - A keep_all + dolmino LM loss: target 经 readout 抄邻接 raw-KV → lm→0 假象。
#   - self-study: 目标不是 next-token LM,是匹配 teacher(看完整 context)的 top-64
#     分布 + hidden。抄邻接 token 不能复现 teacher 对长程依赖的输出 → 逼真用 memory。
#
# 节点: 等空闲 8 卡(本机/.196/B200 任一)。默认本机, 用 CUDA_VISIBLE_DEVICES 覆盖。
# ============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-mem_space_selfstudy_rawkv_chunk512}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29655}"
SAVE_INTERVAL="${SAVE_INTERVAL:-100}"   # save100 防外部 OOM 丢步
RO_LAYERS="${RO_LAYERS:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"
# ★cache 锁定: distill_cache/pg19_512_nctx15 (chunk512/n_ctx15/layers12,20,28).
#   curriculum 必须 = cache n_ctx = 15 (否则 cache miss → distill 静默失效).
DISTILL_CACHE="${DISTILL_CACHE:-distill_cache/pg19_512_nctx15}"
CURRICULUM="${CURRICULUM:-0:15}"
# dolmino_path 必须是 cache 构建时用的同一份 (fingerprint 校验). nctx15 cache 建
#   在 pg19 per-book; 用前确认 meta.json 的 dataset_fingerprint 对得上.
DOLMINO_PATH="${DOLMINO_PATH:-MemLong/data/processed/pg19_perbook_min8k/train}"

mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path $DOLMINO_PATH \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers $RO_LAYERS \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks 0 --rawkv_readout_temp 1.0 \
  --rawkv_gist_pool max --rawkv_gist_lr_mult 1.0 \
  --rawkv_grouped_readout --rawkv_subblock_size 64 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum $CURRICULUM --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --distill_logits --distill_hidden --distill_lambda 0.6 --distill_layers 12,20,28 \
  --distill_weight 1.0 --distill_hidden_beta 1.0 --distill_cache_dir $DISTILL_CACHE \
  --save_interval $SAVE_INTERVAL --eval_interval 0 --log_interval 5 --grad_flow_diag \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
