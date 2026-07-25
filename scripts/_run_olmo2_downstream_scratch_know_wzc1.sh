#!/usr/bin/env bash
# ============================================================================
# from_scratch (random-init 16L OLMo-2-7B, keep14/fresh2 shell) KNOWLEDGE MC
# — the DECISIVE Paper-B control-2 read.  Core 6-task (surface/reasoning) is
# learnable from Dolmino from scratch, so from_scratch ~ ties healed there;
# the discriminating axis is MMLU/knowledge: the healed keep14 recovered MMLU
# to .312 ONLY because it inherited pretrained front layers carrying world
# knowledge.  from_scratch never saw OLMo-2's pretraining corpus -> MMLU should
# sit near the .25 chance floor even after MORE heal steps (200k) -> clean
# "inherited knowledge, not heal-training" control message.
# WZC1 variant: PY=.venv (sm_100 B200/L20A).  prepare_data(proxy) -> 8-GPU MC -> merge.
# ============================================================================
set -uo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$PROJECT_ROOT" || { echo "CD_FAILED $PROJECT_ROOT"; exit 3; }
PY="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

NAME=7B_scratch16L_step200000_know
BASE=../models/OLMo-2-1124-7B
CKPT=outputs/olmo2_probe2_7B_keep14fresh2_fromscratch/final.pt
TASKS=mmlu,lambada_openai,boolq,commonsense_qa,social_iqa
BS=16
export HF_HOME="$PROJECT_ROOT/.hf_cache" HF_DATASETS_CACHE="$PROJECT_ROOT/.hf_cache/datasets"
mkdir -p logs olmo2_downstream_results

# (1) prepare_data — fetch/cache datasets via hy-proxy (CPU only, no ckpt read)
export http_proxy=http://hy-proxy.woa.com:3128 https_proxy=http://hy-proxy.woa.com:3128 \
       all_proxy=http://hy-proxy.woa.com:3128 no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
echo "[$(date '+%F %T')] prepare_data (cache $TASKS)"
$PY scripts/eval_olmo2_probe2_downstream.py --prepare_data --tasks "$TASKS" \
    >logs/olmo2_downstream_scratch_know_prep.log 2>&1
echo "[$(date '+%F %T')] prepare_data done rc=$?"

# (2) 8 shard MC procs — model local; keep proxy for any dataset metadata refetch
echo "[$(date '+%F %T')] fan out 8 shards ($NAME)"
PIDS=()
for g in 0 1 2 3 4 5 6 7; do
  CUDA_VISIBLE_DEVICES=$g $PY scripts/eval_olmo2_probe2_downstream.py \
    --base_model "$BASE" --ckpt "$CKPT" --tasks "$TASKS" \
    --num_shards 8 --shard_index $g --batch_size $BS \
    --output_name "$NAME" \
    >logs/olmo2_downstream_scratch_know_shard${g}.log 2>&1 &
  PIDS+=($!)
  sleep 5   # stagger cold 48GB-ckpt reads
done
for p in "${PIDS[@]}"; do wait "$p"; done

# (3) merge
echo "[$(date '+%F %T')] merge $NAME"
$PY scripts/eval_olmo2_probe2_downstream.py --merge --output_name "$NAME" --tasks "$TASKS" \
    >logs/olmo2_downstream_scratch_know_merge.log 2>&1
cat "olmo2_downstream_results/$NAME/summary.json" 2>/dev/null | head -80
touch logs/olmo2_downstream_scratch_know_DONE
echo "[$(date '+%F %T')] DONE -> olmo2_downstream_results/$NAME/summary.json"
