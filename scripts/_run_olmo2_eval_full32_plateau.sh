#!/usr/bin/env bash
# PaperB P1.1: full-32L continued-pretraining CONTROL, plateau endpoint eval (base protocol), LOCAL B200 (wzc1).
# #100 stopped at plateau step25000 (ppl locked 8.1-8.4 for 17k+ steps, no downward trend) per user
# "到平台期就可以了" — recorded as the 200k-equivalent endpoint. Same harness as keep14/ShortGPT eval.
# Config: keep_front_layers=32 n_fresh_layers=0 (no pruning, no fresh tail — full inheritance, continued-pretrain).
# Three components, sequential, each 8-GPU sharded: (1) held-out NTP PPL (2) core6 (3) know5 incl MMLU.
set -u
WD=/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory
cd "$WD" || exit 1
PY=$WD/.venv/bin/python
BASE="../models/OLMo-2-1124-7B"
CKPT="outputs/olmo2_probe2_7B_full32_dolmino/step25000.pt"
VAL=data/dolmino_now_val.npy

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs olmo2_ppl_results olmo2_downstream_results "$HF_DATASETS_CACHE"

echo "[$(date '+%F %T')] ===== full32 plateau@step25000 FULL eval start (ckpt=$CKPT) ====="

# ---------- (1) held-out PPL ----------
NAME="7B_full32_step25000"
echo "[$(date '+%F %T')] (1) PPL $NAME"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_ppl.py \
    --base_model "$BASE" --ckpt "$CKPT" \
    --keep_front_layers 32 --n_fresh_layers 0 \
    --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
    --output_name "$NAME" \
    > "logs/olmo2_ppl_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_ppl.py --merge --output_name "$NAME" 2>&1
echo "[$(date '+%F %T')] PPL done:"; cat "olmo2_ppl_results/${NAME}/summary.json" 2>/dev/null

# ---------- (2) core 6-task downstream ----------
CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
echo "[$(date '+%F %T')] (2) core downstream $NAME"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
  > "logs/olmo2_downstream_${NAME}_prepare.log" 2>&1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$CORE_TASKS" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$NAME" \
    > "logs/olmo2_downstream_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" 2>&1
echo "[$(date '+%F %T')] core done:"; cat "olmo2_downstream_results/${NAME}/summary.json" 2>/dev/null | head -c 400; echo

# ---------- (3) knowledge 5-task (incl MMLU) ----------
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"
NAMEK="7B_full32_step25000_know"
echo "[$(date '+%F %T')] (3) knowledge $NAMEK"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
  > "logs/olmo2_downstream_${NAMEK}_prepare.log" 2>&1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$KNOW_TASKS" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$NAMEK" \
    > "logs/olmo2_downstream_${NAMEK}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAMEK" 2>&1
echo "[$(date '+%F %T')] knowledge done:"; cat "olmo2_downstream_results/${NAMEK}/summary.json" 2>/dev/null | head -c 400; echo

echo "[$(date '+%F %T')] ===== full32 plateau@step25000 FULL eval ALL DONE ====="
