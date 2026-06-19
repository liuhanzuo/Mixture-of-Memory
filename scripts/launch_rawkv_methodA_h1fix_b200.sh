#!/usr/bin/env bash
# Method A raw-KV readout — H1-FIX retrain on B200 (2026-06-20, methodA-eval).
#
# Diagnosis (commit 5b87057): the original Method A run trained with
# rawkv_readout_topk_chunks=8 >= n_ctx=7 -> GistReadout.retrieve keep_all=True
# -> top-k selection was a NO-OP for the whole run. With all chunks always
# retrieved + uniform weights, the col_bias log-weight is constant across
# columns and CANCELS in the native softmax -> the gist scorer got ~zero
# discriminative gradient. The reader brute-force solved T2 by attending to ALL
# raw-KV chunks -> loss->0 but scorer stayed RANDOM (needle precision 22.5%,
# proj norms at init). The "Method A doesn't break the wall" verdict was thus
# CONFOUNDED — retrieval was never actually tested.
#
# This run FORCES selection so the cross-chunk scorer must learn (tests H1):
#   1. --rawkv_readout_topk_chunks 2   (<< n_ctx=16 -> keep_all=False, selection
#      is real; brute-force-attend-all is impossible).
#   2. --t2_num_keys 3                 (2 distractor needles in OTHER chunks ->
#      the scorer MUST distinguish the queried key's chunk from distractors;
#      this is the core "selection pressure").
#   3. --t2_gap_tokens 8192 --curriculum 0:16  (n_ctx=16 chunks >> topk=2; the
#      query->needle distance is 16*512=8192, deep out-of-window).
#   4. --rawkv_gist_pool max           (anti-dilution: mean-pool of a 512-tok
#      chunk diluted the 25-tok needle -> cross-chunk gist cosine 0.90; max-pool
#      lets the salient needle token survive the pooling).
#
# Everything else held vs the original run (single-axis): raw-KV lossless content,
# unfreeze L16-31, readout 16/20/24, T2 + pg19 0.5 mix, babilong_mix 0, lr 2e-5
# wd 0.1, save 500, diagnostic TOTAL_STEPS 2000 (comparable to the original).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-rawkv_methodA_h1fix_b200}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29873}"
mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers 16,20,24 \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks 2 --rawkv_readout_temp 1.0 \
  --rawkv_gist_pool max \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum 0:16 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy \
  --t2_gap_tokens 8192 --t2_num_keys 3 \
  --save_interval 500 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
