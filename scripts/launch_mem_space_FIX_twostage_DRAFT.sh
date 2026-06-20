#!/usr/bin/env bash
# ============================================================================
# DRAFT — Formal mem_space 长程修复训练 launcher (待实验2确认后秒起)
# 2026-06-20, landmark-repro. Group-A 本机 8×H20 (或 2-node 16卡).
#
# ★状态: DRAFT / 待命。等 methodA-eval 实验2(真实 BABILong 两阶段 readout
#   确认 3.3→~90 成立)+ 确认两阶段 readout 的 flag 名后,补上 §TBD 即可起。
#
# 破墙配方(已锁定,基于 Landmark cache_top_k + grouped-softmax 机制反推):
#   1. raw-KV 无损记忆单元(use_rawkv_readout)— 已有
#   2. 解冻 reader L16-31(unfreeze_backbone + unfreeze_layers_from 16)— 已有
#   3. ★细粒度 64-token block(chunk_size 512→64)— 改 chunk_size 即可
#   4. ★两阶段层级 selection + gather isolation(block级选 × block内token grouped-softmax)
#      — methodA-eval 实验2 实装的新 flag,§TBD 待填
#   5. 多层 readout(rawkv_readout_layers 16-31 或上半层)— 已有
#   6. 全序列/滑动 target loss(可选,B1 的 sliding_target_loss)— 已有但 Level1 证
#      消费非瓶颈,默认 OFF;若两阶段后仍欠再开
# ============================================================================
# ★命门(2026-06-20 methodA-eval+landmark-repro): --rawkv_readout_topk_chunks 0
#   (keep_all) 是训练时的硬要求——所有块进 group_lse 候选, needle 块一定在,
#   内层选择(group_lse)才有梯度可学. hard topk(=2)会先 gather 2 块再 group_lse,
#   若 needle 不在那 2 块(chunk512 ~27%)→ group_lse 候选无 needle → 选择无从学
#   = H2 翻版. 选择 key = k_proj(detached stored hidden), k_proj L16-31 unfrozen
#   在 group_lse 梯度路径 → answer 梯度学"抬 needle sub-block 的 group_lse".
# ============================================================================
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-mem_space_FIX_twostage_chunk512_diskA}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"     # 取 step1000/2000 判据(过训铁律,避 3000)
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29615}"
SAVE_INTERVAL="${SAVE_INTERVAL:-500}"
# ★第一个重训 = chunk512(单轴: 只 +grouped_readout vs Method A h1fix, 避 per_doc cap).
#   chunk64(粒度轴)留作后续: 变体A 已证 chunk512+grouped64 纯eval 边际有效(14 vs flat10),
#   重训放大 grouped 效应;确认后再上 chunk64 加粒度增益(needle 占比~40%)。
#   set CHUNK=64 切换到粒度轴(注意 per_doc cap, 见下).
CHUNK="${CHUNK:-512}"
# 多层 readout (上半层全覆盖). grouped readout(stage2)默认开, stage1_select 不开
#   (变体B 显式 reader-attn 选择纯eval 已证有害; stage1 靠 grouped 顶层 group_lse 涌现).
RO_LAYERS="${RO_LAYERS:-16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31}"

# n_ctx/curriculum: chunk512 → T2 n_ctx = t2_gap_tokens/chunk = 8192/512 = 16 (现成, 无 cap).
#   ⚠️ chunk64 时: dolmino per_doc 单文档 capped 4096 → dolmino 路径 (n_ctx+1)*64<=4096
#   即 n_ctx<=63; 但 T2 recall 路径用独立 background(不受 4096 cap), T2 n_ctx=8192/64=128
#   照跑 → chunk64 方案 dolmino 用小 n_ctx, 长距离靠 T2(n_ctx128)扛. chunk512 不涉及.
CURRICULUM="${CURRICULUM:-0:16}"      # chunk512: n_ctx=16. chunk64 时改 0:128 (T2 路径)
# ★keep_all readout (topk_chunks=0): grouped readout 必须 keep_all, 不能 topk2.
#   理由: needle 块必须在 group_lse 候选里选择才学得到; topk2 会把 needle(27%样本)踢出
#   候选 → 选择无梯度=H2翻版. grouped(64 sub-block 分组归一)已解 keep_all 的 cross-block
#   稀释(每块独立组内归一), 所以 grouped+keep_all 才是对的组合.
TOPK_CHUNKS="${TOPK_CHUNKS:-0}"


mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers $RO_LAYERS \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks $TOPK_CHUNKS --rawkv_readout_temp 1.0 \
  --rawkv_gist_pool max --rawkv_gist_lr_mult 1.0 \
  --rawkv_grouped_readout --rawkv_subblock_size 64 \
  --chunk_size $CHUNK --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
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
  --babilong_mix_fraction 0.0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy \
  --t2_gap_tokens 8192 --t2_num_keys 3 \
  --save_interval $SAVE_INTERVAL --eval_interval 0 --log_interval 5 --grad_flow_diag \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
