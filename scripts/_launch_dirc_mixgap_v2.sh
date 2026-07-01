#!/bin/bash
# Direction-C v2: MIXED-GAP qa5 give-event SFT from A-model step2000.
# Goal (DIRECTION_C_RESULT §4): kill the single-gap "capability translation"
# (H1 over-fit to a fixed 3.5k needle->query distance). Instead of one 16k gap,
# each T2 sample draws a gap from a MIXTURE {1536,3584,7680,12288} so the readout
# fix covers 2k/4k/8k/12k length档 at once (ProLong 2410.02660 processing recipe:
# single-distribution CPT translates capability; mixing spreads the fix).
#
# WHY the 12288 ceiling (not 16k): the FIFO buffer keeps only fifo_buffer_chunks=25
# chunks = 25*512 = 12800 tokens. A 16k gap -> n_ctx=31 -> the needle's chunk is
# EVICTED before t2_select_train_step reads the buffer -> did_select=0 -> the step
# degenerates (no selection supervision, LM window falls back to last-k). So the
# largest gap the b25 buffer can actually train is n_ctx<=24 (~12.3k). 12288=24*512.
# Training 16k+ requires a bigger buffer (see the b50 sweep note in the report).
#
# Red lines (unchanged from MVP): --babilong_mix_fraction 0; all-synthetic (never
# reads babilong test); warm-start from step2000 (NOT a leaked ckpt); pg19 ppl held.
#
# PATHS: the idle nodes (.53/.245) mount ONLY diskB (share_304376610); diskA is
# NOT mounted there. Everything below is diskB-absolute. Run this ON a remote node.
set -euo pipefail
R="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"   # diskB
cd "$R"
export WANDB_MODE="offline"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="$R/.venv/bin/python"
RUN="mem_space_dirc_mixgap_v2"
mkdir -p logs
if pgrep -f "wandb_run_name $RUN" >/dev/null 2>&1; then echo "REFUSE: $RUN running"; exit 3; fi
INIT="outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
IACFG="outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
echo "[launch] $RUN (mixed-gap {1536,3584,7680,12288} qa5 SFT, from step2000, lr3e-5, 1600 steps, mix=0)"
# Step budget = 1600 (vs MVP 800): keep t2_recall_mix_fraction 0.5 IDENTICAL to
# the MVP (so the pg19-dolmino anchoring fraction — the ppl red line — is
# unchanged), and DOUBLE total_steps so the 4-way gap mixture gets ~2x the T2
# micros the MVP spent on its single gap, i.e. ~half MVP exposure PER gap. This
# trades per-gap saturation for coverage (the point of mixing) while holding the
# ppl guardrail constant. Save every 400 so all length档 can be eval'd at several
# steps (the "看单档=修复错觉" lesson: must check every档 at multiple ckpts).
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PY -m torch.distributed.run --nproc_per_node=8 --master_port=29814 \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B --per_doc_data --dolmino_path MemLong/data/processed/dolmino_per_doc/train \
  --output_dir outputs/$RUN \
  --init_checkpoint $INIT --init_adapter_config $IACFG \
  --total_steps 1600 --lr 3e-5 --warmup_steps 60 \
  --chunk_size 512 --batch_size 1 --num_slots 128 --top_k 16 --selector_dim 128 --selector_temperature 40 \
  --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 25 --fifo_detach \
  --unfreeze_backbone --unfreeze_layers_set 16,28,29,30,31 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0.5 --t2_background_data data/pg19_chunks_llama3.npy --t2_num_keys 1 \
  --t2_gap_tokens 3584 --t2_gap_mix 1536,3584,7680,12288 --t2_background_skip 0 \
  --t2_difficulty_curriculum 0:6=1.0 \
  --t2_select_loss_weight 1.0 --t2_select_layer 16 --t2_select_topk 4 \
  --save_interval 400 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 --wandb_project mixture-of-memory --wandb_run_name $RUN --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$!"
