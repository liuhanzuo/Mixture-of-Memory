#!/usr/bin/env bash
# SlimPajama pure-prediction self-supervision (2026-07-05, 本机 B200/L20A).
#
# WHY: the babilong-qa5 recall-sweep SFT overfits qa-token shortcuts (recall0.02
# got qa5 16k=40 but LongBench f1<11 / RULER 8k=8 / qa2=0 — NOT general memory).
# Literature (Beacon/MemoryLLM/M+/AutoCompressor/CEPE) learns real memory readout
# via GENERIC long-document dense-LM, never small-answer synthetic NIAH. User
# directive: "useful memory = can I still generate the FOLLOWING text from what I
# remember (prediction, not reconstruction)". That IS --last_chunk_loss_only:
# context chunks stream into memory (no_grad→detach), target chunk's next-token
# prediction can ONLY use memory.
#
# RECIPE (no babilong, no synthetic recall, no answer_mask):
#   --slimpajama_data data/slimpajama_chunks_4096.npy  (generic long docs, 1.57M rows)
#   --last_chunk_loss_only                             (prediction self-supervision)
#   --t2_recall_mix_fraction 0                         (NO synthetic recall)
#   --babilong_mix_fraction 0                          (RED LINE)
#   warm from CLEAN Llama-3 base (no babilong-SFT-contaminated ckpt) OR from the
#   clean b64 PG19 pretrain — here we start from clean base for a pure signal.
#   Eval = ZERO-SHOT LongBench/RULER (never SFT on any benchmark format).
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export WANDB_MODE="offline"
export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_ROOT/third_party/babilong-pkg:$PROJECT_ROOT:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PYBIN="$PROJECT_ROOT/.venv/bin/python"
PORT="${PORT:-29943}"
RUN="${RUN:-mem_space_slimpajama_prediction}"
# Warm from clean b64 PG19 pretrain (generic, NOT babilong-contaminated) so we
# build on existing long-context memory rather than from scratch. Set INIT="" to
# train from clean Llama-3 base instead.
INIT="${INIT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/MoM_mainserver_20260704/outputs/mem_space_pg19base_fifo_b64/full_model_step002000.pt}"
IACFG="${IACFG:-/apdcephfs_wzc1/share_304376610/pighzliu_code/MoM_mainserver_20260704/outputs/mem_space_pg19base_fifo_b64/adapter_config.json}"
mkdir -p logs outputs/$RUN
if pgrep -f "wandb_run_name $RUN" 2>/dev/null | while read _p; do grep -q "torch.distributed.run\|train_mem_space" "/proc/$_p/cmdline" 2>/dev/null && echo hit; done | grep -q hit; then echo "REFUSE: $RUN running"; exit 3; fi

INIT_ARGS=""
[ -n "$INIT" ] && [ -f "$INIT" ] && INIT_ARGS="--init_checkpoint $INIT --init_adapter_config $IACFG"
echo "[launch] $RUN — SlimPajama pure-prediction (recall=0, no babilong). init='${INIT:-clean-base}'"
setsid bash -c "CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 $PYBIN -m torch.distributed.run \
  --nproc_per_node=8 --master_port=$PORT \
  scripts/train_mem_space_dolmino_cpt.py \
  --model_path models/Meta-Llama-3-8B \
  --slimpajama_data data/slimpajama_chunks_4096.npy \
  $INIT_ARGS \
  --output_dir outputs/$RUN \
  --total_steps 2000 --lr 1e-4 --warmup_steps 100 \
  --chunk_size 512 --batch_size 4 --num_slots 128 --top_k 16 --selector_dim 128 \
  --selector_temperature 40 --load_balance_weight 0.0 --entropy_aux_weight 0.0 \
  --use_fifo_memory --fifo_buffer_chunks 64 --fifo_detach --last_chunk_loss_only \
  --unfreeze_backbone --unfreeze_layers_from 16 \
  --slot_init strided_token --slot_init_noise 0.0 --writeback_gate_max 1.0 \
  --gradient_checkpointing --gradient_accumulation_steps 4 --curriculum 0:3 \
  --bptt_window 1 --inject_gate_bias_init -2.0 \
  --babilong_mix_fraction 0 \
  --t2_recall_mix_fraction 0 \
  --save_interval 250 --eval_interval 0 --log_interval 5 \
  --grad_clip 1.0 --proj_grad_clip 0.1 \
  --wandb_project mixture-of-memory --wandb_run_name $RUN \
  --dtype bfloat16 --attn_impl sdpa --seed 42" \
  </dev/null >logs/$RUN.log 2>&1 &
echo "launched $RUN pid=$! log=logs/$RUN.log"
