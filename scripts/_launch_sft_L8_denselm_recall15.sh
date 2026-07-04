#!/usr/bin/env bash
# Level8 SFT — LITERATURE-CORRECTED data recipe (2026-07-04, local B200/L20A).
#
# WHY (status/LITERATURE_TRAINING_DATA_20260702.md 三方共识 Beacon/AutoCompressor/ARMT):
#   The main-server level8 SFT used --t2_recall_mix_fraction 0.6 => recall is the
#   MAJORITY (60%) training signal. The literature (and our own t2_needle->0
#   overfitting signal) says: low-entropy synthetic NIAH recall as the MAIN signal
#   is the overfitting root cause. The fix is to make DENSE LM (full-target-chunk
#   NTP on generic PG19/dolmino text, streamed through memory) the MAIN signal and
#   demote recall to a <=20% side-dish.
#
# THIS RECIPE (single change of substance vs main-server level8):
#   --t2_recall_mix_fraction 0.6  ->  0.15   (recall demoted to side-dish;
#                                             dense-LM dolmino step is now ~85%)
#   Everything else kept from the validated level8 SFT recipe:
#     - level8 multi-template qa5 (--t2_difficulty_curriculum 0:8=1.0)
#     - mixed-length gap (--t2_gap_mix 2048,4096,8192) covering 2k-8k档
#     - pure hidden readout (--t2_select_loss_weight 0, no reforward)
#     - buffer64 (救16k: 16k~31chunk < 64, no eviction)
#     - warm-start from the b64 PG19 pretrain step2000 ckpt
#     - --babilong_mix_fraction 0 (RED LINE: never train on babilong test)
#
# JUDGING: after step500/1000, eval on REAL babilong qa5 全档 n100 (official
# compare_answers). Success = 8k/16k >= the level8-0.6 recipe (8k28/16k16) WITHOUT
# the short-档 regression (2k/4k should recover toward pretrain 67/53 since dense
# LM keeps fluency). This isolates "is recall-as-side-dish better than
# recall-as-main?" — the literature's central claim, never tested in this project.
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_MODE="offline"
export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYBIN="$PROJECT_ROOT/.venv/bin/python"
PORT="${PORT:-29931}"
RUN="mem_space_sft_L8_denselm_recall15"
# Warm-start = b64 PG19 pretrain step2000 (same as the validated main-server SFT).
# ckpt lives on the copied mirror (same physical wzc1 disk, no re-copy).
INIT="${INIT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/MoM_mainserver_20260704/outputs/mem_space_pg19base_fifo_b64/full_model_step002000.pt}"
IACFG="${IACFG:-/apdcephfs_wzc1/share_304376610/pighzliu_code/MoM_mainserver_20260704/outputs/mem_space_pg19base_fifo_b64/adapter_config.json}"
mkdir -p logs outputs/$RUN

if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN already running"; exit 3; fi
if [ ! -f "$INIT" ]; then echo "ABORT: warm-start ckpt not found: $INIT"; exit 4; fi

echo "[launch] $RUN — literature-corrected recipe (recall 0.6->0.15, dense-LM main)"
echo "         warm-start: $INIT"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run \
  --nproc_per_node=8 --master_port=$PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 1500 --lr 3e-5 --warmup_steps 50 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 64 --fifo_detach --last_chunk_loss_only \
  --unfreeze_backbone --unfreeze_layers_from 16 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 \
  --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.15 \
  --t2_background_data data/pg19_chunks_llama3_noeos.npy \
  --t2_num_keys 1 --t2_gap_tokens 4096 --t2_gap_mix 2048,4096,8192 \
  --t2_difficulty_curriculum 0:8=1.0 \
  --t2_select_loss_weight 0 \
  --save_interval 250 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN \
  --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$! log=logs/$RUN.log"
