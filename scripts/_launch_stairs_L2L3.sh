#!/usr/bin/env bash
# reader-attn信号难度课程(用户阶梯思路, 2026-06-29)
# t2_difficulty_curriculum: 0步纯L2(mention)→400掺L3(format)→800纯L3。逼selector先掌握易信号再学难的。
# 从step2000续训, mix=0, 全合成零babilong污染。@.196(diskA, 代码直接可见)
set -euo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"; export HF_HOME="$R/.hf_cache"; export HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"; export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_select_stairs_L2L3_g8k"
INIT="$R/outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
ICFG="$R/outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
mkdir -p logs outputs/$RUN
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi
echo "[launch] $RUN — 阶梯L2→L3 difficulty curriculum, gap8192+nctx curriculum, step2000续训, mix=0"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29817 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --init_checkpoint $INIT --init_adapter_config $ICFG \
  --output_dir outputs/$RUN --total_steps 1500 --lr 5e-5 --warmup_steps 50 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 3 --t2_gap_tokens 8192 --t2_background_skip 0 \
  --t2_hard_distractors 3 \
  --t2_curriculum 0:4,300:8,600:16,1000:24 \
  --t2_difficulty_curriculum '0:2=1.0,400:2=0.5|3=0.5,800:3=1.0' \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 300 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 44" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
