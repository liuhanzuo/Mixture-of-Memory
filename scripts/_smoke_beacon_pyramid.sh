#!/bin/bash
# Beacon Pyramid (idea #3) SMOKE — 30 steps, confirm loss drops + no crash.
# Multi-scale COMPRESSED FIFO prefix consumed DIRECTLY by the (unfrozen) reader
# + jointly-trained beacon pool (Activation-Beacon-style, NOT frozen injection).
#
# Recipe note (critical): --t2_select_loss_weight 0 so T2 needle samples flow
# through dolmino_train_step (stream ctx -> FIFO buffer -> grad target forward),
# and the target forward reads the beacon prefix via _forward_fifo_beacon. The
# LM answer-digit CE backprops into the beacon pool + unfrozen reader layers.
# (t2_select>0 would route to the token-reforward window and BYPASS beacons.)
#
# Red lines: --babilong_mix_fraction 0 (all-synthetic, never touches babilong
# test); warm-start from CLEAN A-model step2000; buffer=25/chunk=512.
set -euo pipefail
R="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_beacon_pyramid_smoke"
NPROC="${1:-2}"
MPORT="${2:-29841}"
mkdir -p logs
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[smoke] $RUN (beacon pyramid, nproc=$NPROC, 30 steps, mix=0)"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1 $PY -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MPORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 30 --lr 3e-5 --warmup_steps 5 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --use_beacon_pyramid --beacon_k 8 --beacon_fine_chunks 2 --beacon_mid_chunks 6 --beacon_branch 4 \
  --beacon_heads 8 --beacon_layers 1 --beacon_ffn_mult 2 \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 1.0 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 1 \
  --t2_gap_tokens 3584 --t2_gap_mix 1536,3584,7680 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:6=1.0 \
  --t2_select_loss_weight 0 \
  --save_interval 100 --eval_interval 0 --log_interval 1 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!  log=logs/$RUN.log"
