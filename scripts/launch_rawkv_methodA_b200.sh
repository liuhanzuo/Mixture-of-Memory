#!/usr/bin/env bash
# Method A raw-KV readout — real training on B200 (2026-06-19, rawkv-protoA).
#
# Validates the FORWARD hypothesis of docs/RAWKV_READOUT_PROPOSAL.md §2 Method A:
# raw-KV (lossless) + emergent trainable gist-key soft attention + UNFROZEN reader
# → can the reader consume long-range needles that frozen-reader + lossy-slot
# mem_space could not (the dead end 30 experiments hit)?
#
# Recipe (anti-base-damage, grounded in our own evidence):
#   1. PARTIAL unfreeze L16-31 (--unfreeze_layers_from 16). MANDATORY: a frozen
#      reader reproduces the in-attn oracle negative (21≈OFF 22, 76efbd4). Partial
#      (not full) protects layers 0-15 + embed_tokens — v1 FULL unfreeze on short
#      data halved base NIAH (22→11).
#   2. Multi-layer readout injection 16,20,24 (all in the unfrozen range; verified
#      ≈0 extra GPU memory per layer on the realmodel probe). WRITE owned by L16.
#   3. DATA = pg19_perbook_min8k (real long books). This is the load-bearing
#      anti-damage choice: long NATURAL text simultaneously (a) exercises the
#      long-range dependency Method A must learn and (b) is base-distribution
#      text → self-anchors the LM against catastrophic forgetting (the "base
#      replay / 90-10" lesson, but built into the data, not a separate stream).
#      It is NOT the short dolmino that is the suspected cause of v1's base damage,
#      and NOT babilong QA (which would contaminate the held-out eval).
#   4. babilong_mix 0.0 (eval-validity) + conservative lr 2e-5 cosine, 3% warmup,
#      wd 0.1, grad_clip 1.0, proj_grad_clip 0.1. Early ckpts (save 500) so base
#      NIAH can be re-checked per ckpt and the run killed if base degrades.
#   5. raw-KV readout: --use_rawkv_readout, gist soft-top-k=8, gist_dim 128, temp
#      1.0. TopKSelector is OFF the read path. Slot xattn read NOT enabled
#      (--use_memory_xattn omitted) → ONLY the rawkv readout supplies long-range
#      context (clean test). Slot aux weights zeroed to keep the loss pure-LM-ish.
#
# DIAGNOSTIC run: TOTAL_STEPS 2000 (look for long-range signal; not full 15k).
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT"
export WANDB_API_KEY="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"
export WANDB_MODE="offline"
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
PYBIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"
RUN="${RUN:-rawkv_methodA_b200}"
NPROC="${NPROC:-8}"
GPUS="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
TOTAL_STEPS="${TOTAL_STEPS:-2000}"
WARMUP="${WARMUP:-60}"
MASTER_PORT="${MASTER_PORT:-29871}"
mkdir -p logs outputs/$RUN
setsid bash -c "CUDA_VISIBLE_DEVICES=$GPUS $PYBIN -m torch.distributed.run --nproc_per_node=$NPROC --master_port=$MASTER_PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --per_doc_data --dolmino_path MemLong/data/processed/pg19_perbook_min8k/train \
  --output_dir outputs/$RUN --total_steps $TOTAL_STEPS --lr 2e-5 --warmup_steps $WARMUP \
  --weight_decay 0.1 \
  --unfreeze_backbone --unfreeze_layers_from 16 --use_fsdp \
  --use_rawkv_readout --rawkv_readout_layer 16 --rawkv_readout_layers 16,20,24 \
  --rawkv_gist_dim 128 --rawkv_readout_topk_chunks 8 --rawkv_readout_temp 1.0 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_loss_free_balance --loss_free_update_rate 0.001 --num_global_slots 4 \
  --key_repulsion_weight 0.0 --slot_value_norm_cap 5.0 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --unfreeze_hidden_to_slot --use_dual_gate --forget_bias_init 2.0 --input_bias_init 0.0 \
  --dual_gate_tanh_new --use_l3_summary --l3_n_summary 64 --l3_n_layers 2 --l3_n_heads 8 \
  --shared_memory_bank --gradient_checkpointing --gradient_accumulation_steps 2 \
  --last_chunk_loss_only --curriculum 0:7 --bptt_window 2 --no_detach_slots_in_selector \
  --no_slot_delta_clip --inject_gate_bias_init -2.0 --routing_pool_mode slot_query \
  --multi_query_tau 1.0 --l3_diversity_weight 0.0 --l_recon_weight 0.0 --route_aux_weight 0.0 \
  --babilong_mix_fraction 0.0 --t2_recall_mix_fraction 0.0 \
  --save_interval 500 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory \
  --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
