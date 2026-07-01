#!/usr/bin/env bash
# ============================================================================
# 训练时 reader-native top-k keep-set 隔离 (攻"稀释是元凶"结论).
#   把 FIFO-oracle 揭示的"选对 chunk 就能读"bake 进训练: 训练时 FIFO readout
#   只 attend reader-attn 选中的 top-k chunk (+recency), keep-all-store.
#   usage: bash _launch_t2_keepset.sh <TOPK> [MASTER_PORT]
# ALL: --babilong_mix_fraction 0 (零泄漏) + T2 合成 needle 格式对齐.
# ============================================================================
set -euo pipefail
TOPK="${1:-8}"
MPORT="${2:-29806}"
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export http_proxy="http://hy-proxy.woa.com:3128"; export https_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_cache"; export HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1; export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

RUN="mem_space_fifo_b25_chunk512_t2_keepset${TOPK}"
mkdir -p logs
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN already running"; exit 3; fi

echo "[launch] $RUN  train-keepset topk=$TOPK recency=2 keep_all_buffer  port=$MPORT (babilong_mix=0, t2=0.15)"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run --nproc_per_node=8 --master_port=$MPORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --total_steps 3000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --fifo_train_keep_set_mode flat_readerattn --fifo_train_keep_topk $TOPK \
  --fifo_train_keep_recency 2 --fifo_train_keep_all_buffer \
  --unfreeze_backbone --unfreeze_layers_from 24 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 \
  --curriculum 0:3 --bptt_window 1 \
  --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.15 \
  --t2_background_data data/pg19_chunks_llama3.npy \
  --t2_num_keys 1 --t2_gap_tokens 3584 --t2_background_skip 0 \
  --save_interval 500 --eval_interval 0 --eval_samples 30 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  log=logs/$RUN.log"
