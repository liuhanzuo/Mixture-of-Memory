#!/bin/bash
# HNST v2 REAL TRAIN (2026-06-25): trainable tree-summary + unfrozen reader.
# Attacks BOTH walls: (1) selection wall via a hierarchical per-level tree
# navigation CE (learned aggregation keeps needle-routing info that v1 max-pool
# destroyed), (2) readout wall via token-reforward LM loss on the selected
# chunks with the reader (layers {16,28-31}) unfrozen.
#
# branch=4 (NOT 8): with buffer=25/chunk=512 and a gap mix up to 7680 (n_ctx=15),
# branch=4 gives 2 tree LEVELS so the INTERNAL-node aggregation (the thing v1
# died on) is actually exercised & supervised. gap_mix {1536,3584,7680} spreads
# the fix across 2k/4k/8k length档 (DIRECTION_C_RESULT: single-gap only translates
# capability along length). Ceiling 7680 (not 12288): the 12288-gap ranks
# (n_ctx=24) made the DDP step ~40s (load-imbalanced barrier gated by the slowest
# rank) → ~11h/1000steps. 7680 halves the slowest-rank streaming cost while still
# training a genuine multi-level tree.
#
# Red lines: --babilong_mix_fraction 0; all-synthetic (never reads babilong test);
# warm-start from CLEAN A-model step2000 (NOT a leaked ckpt); pg19 ppl guardrail.
set -euo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_hnstv2_tree_b25"
NPROC="${1:-8}"
MPORT="${2:-29820}"
mkdir -p logs
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[launch] $RUN (HNST v2 trainable tree, branch=4, mixed-gap, from step2000, 1000 steps, mix=0)"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MPORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 400 --lr 3e-5 --warmup_steps 30 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --use_tree_summary --tree_summary_layers 1 --tree_summary_heads 8 \
  --t2_tree_branch 4 --t2_tree_beam 2 \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 1 \
  --t2_gap_tokens 3584 --t2_gap_mix 1536,3584,7680 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:6=1.0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 100 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  log=logs/$RUN.log"
