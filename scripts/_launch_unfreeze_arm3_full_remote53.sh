#!/bin/bash
# UNFREEZE-SWEEP arm3 on REMOTE diskB (.53): FULL unfreeze all 32 layers + norm + lm_head.
# GPUs 4-7, DDP + ZeRO-1 (shards the ~60GB AdamW state across ranks; params/grads
# stay replicated so the t2_select salience helper works — FSDP resharded them
# to 0-size and crashed).
set -euo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOM_ZERO1=1
PY="$R/.venv/bin/python"
RUN="mem_space_unfreeze_arm3_full"
mkdir -p logs
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[launch] $RUN (FULL unfreeze all 32 layers, DDP+ZeRO-1 GPUs 4-7, from step2000, 400steps, mix=0)"
setsid bash -c "CUDA_VISIBLE_DEVICES=4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=4 --master_port=29815 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 400 --lr 3e-5 --warmup_steps 50 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --unfreeze_backbone \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 1 --t2_gap_tokens 3584 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:6=1.0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 200 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
