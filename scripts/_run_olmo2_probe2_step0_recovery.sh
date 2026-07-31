#!/usr/bin/env bash
# Paper B P0 -- step-0 (pruned, UNHEALED) recovery-fraction baseline eval.
#
# Generates the step-0 checkpoint for each keep depth-ladder rung
# keep{8,10,12,14}+fresh2 using the EXACT trainer recipe (front keep_front layers
# + embed/norm/lm_head transplanted from vanilla OLMo-2-1124-7B, top layers dropped,
# n_fresh=2 FRESH Olmo2-init tail layers, ZERO training steps) via the trainer's
# own --save_step0_and_exit path -> byte-identical to what continue-training starts
# from. Then evals each step-0 ckpt with the SAME base protocol / same harness /
# same 8-shard [g::8] scheme as the healed-run ledgers:
#   * held-out PPL          (scripts/eval_olmo2_probe2_ppl.py, val dolmino_now_val.npy)
#   * core-6 downstream MC   (hellaswag/arc_c/arc_e/piqa/winogrande/openbookqa)
#   * knowledge-5 downstream (mmlu/lambada_openai/boolq/commonsense_qa/social_iqa)
# Base 口径: NO chat_template, NO BOS (--add_bos 0), fp32 weights + bf16-autocast
# forward, likelihood MC (acc + acc_norm). => recovery fraction vs the healed
# ledgers is directly comparable. Vanilla base (100% ref) is NOT re-run here (its
# summaries already exist: olmo2_ppl_results/7B_base_full, olmo2_downstream_results/
# 7B_base_full{,_know}).
set -u
cd /apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory
PY=/opt/conda/envs/torch-base/bin/python
BASE=../models/OLMo-2-1124-7B
VAL=data/dolmino_now_val.npy

# outbound proxy for HF datasets download; project data-dir cache (diskB, persists)
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory/data/hf_datasets_cache
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$HF_DATASETS_CACHE"

KEEPS="8 10 12 14"
CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
DONE=logs/olmo2_step0_recovery_DONE
rm -f "$DONE"

# ---------------------------------------------------------------------------
# Phase 1: generate step-0 checkpoints (exact trainer recipe, no training)
# ---------------------------------------------------------------------------
for N in $KEEPS; do
  OUT=outputs/olmo2_probe2_7B_keep${N}_step0_pruned
  if [ -f "$OUT/step0.pt" ]; then
    echo "[$(date '+%F %T')] step0 keep$N already exists ($OUT/step0.pt), skip gen"
    continue
  fi
  echo "[$(date '+%F %T')] GEN step0 keep$N -> $OUT"
  CUDA_VISIBLE_DEVICES=0 $PY scripts/train_olmo2_arch_probe2.py \
    --save_step0_and_exit \
    --keep_front_layers "$N" --n_fresh_layers 2 \
    --model_path "$BASE" \
    --data_path "$VAL" \
    --output_dir "$OUT" \
    > "logs/olmo2_step0_gen_keep${N}.log" 2>&1
  tail -4 "logs/olmo2_step0_gen_keep${N}.log"
done

# ---------------------------------------------------------------------------
# Phase 2: held-out PPL (8-shard per config)
# ---------------------------------------------------------------------------
for N in $KEEPS; do
  NAME=7B_keep${N}_step0
  CKPT=outputs/olmo2_probe2_7B_keep${N}_step0_pruned/step0.pt
  echo "=========================================================="
  echo "[$(date '+%F %T')] PPL $NAME ckpt=$CKPT"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
      --base_model "$BASE" --ckpt "$CKPT" \
      --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
      --output_name "$NAME" \
      > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
done

# ---------------------------------------------------------------------------
# Phase 3: core-6 downstream MC
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] prepare_data core"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
  > logs/olmo2_step0_core_prepare.log 2>&1
tail -6 logs/olmo2_step0_core_prepare.log
for N in $KEEPS; do
  NAME=7B_keep${N}_step0
  CKPT=outputs/olmo2_probe2_7B_keep${N}_step0_pruned/step0.pt
  echo "=========================================================="
  echo "[$(date '+%F %T')] CORE $NAME"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
      --num_shards 8 --shard_index $g --batch_size 16 \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
done

# ---------------------------------------------------------------------------
# Phase 4: knowledge-5 downstream MC
# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] prepare_data knowledge"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
  > logs/olmo2_step0_know_prepare.log 2>&1
tail -12 logs/olmo2_step0_know_prepare.log
for N in $KEEPS; do
  NAME=7B_keep${N}_step0_know
  CKPT=outputs/olmo2_probe2_7B_keep${N}_step0_pruned/step0.pt
  echo "=========================================================="
  echo "[$(date '+%F %T')] KNOW $NAME"
  for g in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
      --base_model "$BASE" --ckpt "$CKPT" --tasks "$KNOW_TASKS" \
      --num_shards 8 --shard_index $g --batch_size 8 \
      --output_name "$NAME" \
      > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
  done
  wait
  $PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
done

# ---------------------------------------------------------------------------
echo "[$(date '+%F %T')] ALL DONE" | tee "$DONE"
for N in $KEEPS; do
  for suf in "" "_know"; do
    echo "--- 7B_keep${N}_step0${suf} (downstream) ---" >> "$DONE"
    cat "olmo2_downstream_results/7B_keep${N}_step0${suf}/summary.json" >> "$DONE" 2>/dev/null
  done
  echo "--- 7B_keep${N}_step0 (ppl) ---" >> "$DONE"
  cat "olmo2_ppl_results/7B_keep${N}_step0/summary.json" >> "$DONE" 2>/dev/null
done
echo "[$(date '+%F %T')] wrote $DONE"
