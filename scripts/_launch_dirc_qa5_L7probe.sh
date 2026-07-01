#!/bin/bash
# Direction-C L7 test: continue-train the A-model on the NEW babilong-shaped
# synthetic (difficulty level 7) to test whether matching the NOISE STRUCTURE
# (scattered mid-prose bAbI + recency + entity recurrence + continuous PG19)
# transfers to REAL babilong qa5 readout BETTER than the old L6 clean synthetic
# (which DAMAGED readout: unfreeze top16 36<base45; DIRECTION_C平移).
#
# Deltas vs _launch_dirc_qa5sft_mvp.sh (the L6 run):
#   * --t2_difficulty_curriculum 0:7=1.0    (L7 babilong-shaped, was 0:6=1.0)
#   * --t2_background_data data/pg19_chunks_llama3_noeos.npy  (continuous prose,
#       was pg19_chunks_llama3.npy which is 6.8% <|end_of_text|> = fragmented)
#   * --t2_gap_mix 4096,8192,11776   (mixed length; 11776=23*512 keeps needle in
#       the b25 FIFO buffer; end-biased needle survives eviction). Was single 3584.
#   * 300 steps (small probe, save every 100). Everything else IDENTICAL.
# RED LINES (unchanged): --babilong_mix_fraction 0; all-synthetic; warm-start
# step2000; pg19 ppl guardrail.
set -euo pipefail
R="/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory"
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_dirc_qa5_L7probe"
mkdir -p logs
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[launch] $RUN (L7 babilong-shaped qa5 SFT, noeos bg, mixgap, from step2000, 300 steps, mix=0)"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29815 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 300 --lr 3e-5 --warmup_steps 30 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3_noeos.npy --t2_num_keys 1 \
  --t2_gap_tokens 8192 --t2_gap_mix 4096,8192,11776 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:7=1.0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 100 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
