#!/usr/bin/env bash
# FULL base-protocol eval for the Qwen3 prune-then-heal probe (Paper B P2.3,
# cross-family control for the OLMo-2 prune-then-heal PPL/MMLU dissociation).
# Mirrors scripts/_run_olmo2_eval_shortgpt.sh but for the Qwen3 family and a
# SINGLE checkpoint (parameterised by env vars), running 8-GPU sharded + sequential:
#   (1) held-out NTP PPL   (2) core 6-task downstream   (3) knowledge 5-task incl MMLU
#
# Two modes (switched by CKPT):
#   * CKPT=""            -> Control 0: full-depth 36-layer Qwen3-8B base (no prune).
#                          KEEP_FRONT / N_FRESH are ignored.
#   * CKPT=<path>.pt     -> pruned prune-then-heal ckpt; KEEP_FRONT / N_FRESH set the
#                          (keep_front + n_fresh)-layer shell (default 12 / 2 = 14L).
#
# Env vars (all overridable):
#   WD          working dir (default diskB canonical path)
#   PY          python (default diskB olmo2_venv)
#   BASE        Qwen3-8B base path (cfg source + base-mode model)
#   CKPT        ckpt .pt path ("" -> base mode)
#   KEEP_FRONT  kept front layers (pruned mode; default 12)
#   N_FRESH     fresh tail layers (pruned mode; default 2)
#   VAL         held-out val npy (Qwen-tokenized SlimPajama)
#   NAME        output_name for core+PPL results (know results = ${NAME}_know)
set -u
WD="${WD:-/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$WD" || exit 1
PY="${PY:-$WD/olmo2_venv/bin/python}"
BASE="${BASE:-models/Qwen3-8b-local}"
CKPT="${CKPT:-}"
KEEP_FRONT="${KEEP_FRONT:-12}"
N_FRESH="${N_FRESH:-2}"
VAL="${VAL:-data/slimpajama_val_2048_qwen3.npy}"
NAME="${NAME:-qwen3_eval}"
NAMEK="${NAME}_know"

export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs qwen3_probe2_ppl_results qwen3_probe2_downstream_results "$HF_DATASETS_CACHE"

CORE_TASKS="hellaswag,arc_challenge,arc_easy,piqa,winogrande,openbookqa"
KNOW_TASKS="mmlu,lambada_openai,boolq,commonsense_qa,social_iqa"

# Build the ckpt-specific CLI args (empty in base mode).
CKPT_ARGS=""
if [ -n "$CKPT" ]; then
  if [ ! -f "$CKPT" ]; then
    echo "[$(date '+%F %T')] ERROR: CKPT=$CKPT does not exist"; exit 1
  fi
  CKPT_ARGS="--ckpt $CKPT --keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH"
  echo "[$(date '+%F %T')] ===== Qwen3 FULL eval start [PRUNED name=$NAME ckpt=$CKPT keep=$KEEP_FRONT fresh=$N_FRESH] ====="
else
  echo "[$(date '+%F %T')] ===== Qwen3 FULL eval start [BASE name=$NAME (full 36L, no prune)] ====="
fi

# ---------- (1) held-out PPL ----------
echo "[$(date '+%F %T')] (1) PPL $NAME"
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qwen3_probe2_ppl.py \
    --base_model "$BASE" $CKPT_ARGS \
    --val_path "$VAL" --num_shards 8 --shard_index $g --batch_size 4 \
    --output_name "$NAME" \
    > "logs/qwen3_ppl_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_qwen3_probe2_ppl.py --merge --output_name "$NAME" 2>&1
echo "[$(date '+%F %T')] PPL done:"; cat "qwen3_probe2_ppl_results/${NAME}/summary.json" 2>/dev/null

# ---------- (2) core 6-task ----------
echo "[$(date '+%F %T')] (2) core downstream $NAME"
$PY scripts/eval_qwen3_probe2_downstream.py --prepare_data --tasks "$CORE_TASKS" \
  > "logs/qwen3_downstream_${NAME}_prepare.log" 2>&1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qwen3_probe2_downstream.py \
    --base_model "$BASE" $CKPT_ARGS --tasks "$CORE_TASKS" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$NAME" \
    > "logs/qwen3_downstream_${NAME}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_qwen3_probe2_downstream.py --merge --output_name "$NAME" 2>&1
echo "[$(date '+%F %T')] core done:"; cat "qwen3_probe2_downstream_results/${NAME}/summary.json" 2>/dev/null | head -c 500; echo

# ---------- (3) knowledge 5-task (incl MMLU) ----------
echo "[$(date '+%F %T')] (3) knowledge $NAMEK"
$PY scripts/eval_qwen3_probe2_downstream.py --prepare_data --tasks "$KNOW_TASKS" \
  > "logs/qwen3_downstream_${NAMEK}_prepare.log" 2>&1
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_qwen3_probe2_downstream.py \
    --base_model "$BASE" $CKPT_ARGS --tasks "$KNOW_TASKS" \
    --num_shards 8 --shard_index $g --batch_size 8 \
    --output_name "$NAMEK" \
    > "logs/qwen3_downstream_${NAMEK}_shard${g}.log" 2>&1 &
done
wait
$PY scripts/eval_qwen3_probe2_downstream.py --merge --output_name "$NAMEK" 2>&1
echo "[$(date '+%F %T')] knowledge done:"; cat "qwen3_probe2_downstream_results/${NAMEK}/summary.json" 2>/dev/null | head -c 500; echo

echo "[$(date '+%F %T')] ===== Qwen3 FULL eval ALL DONE (name=$NAME) ====="
