#!/usr/bin/env bash
# ============================================================================
# Paper B P2.5 — Qwen3 cross-family protocol-complete supplementary eval driver.
#
# PURE EVAL. Runs, for ONE arm (Qwen3-8B full base OR the f12k2 prune-then-heal
# ckpt), the two protocol-complete evaluators ported from OLMo-2 P0.6 / P0.3:
#   * content-MMLU  scripts/eval_qwen3_mmlu_content.py   (letter + content_raw +
#                   content_norm; cais/mmlu "all" test, 14,042 items)
#   * closed-book   scripts/eval_qwen3_closedbook_qa.py  (popqa/triviaqa/nq_open;
#                   em/contains/f1, majority floor)
# Each is fanned out over $NGPU GPU shards (examples[shard::N]) then merged
# (count/weight-summed, never mean-of-metric). Results:
#   qwen3_mmlu_content_results/<TAG>/summary.json + per_example_mmlu.jsonl
#   qwen3_closedbook_results/<TAG>/summary.json   + per_example_<task>.jsonl
#
# Base protocol (aligned with OLMo, both are BASE LMs, NO SFT): chat_template=
# False, add_special_tokens=False -> add_bos=0. Qwen3 tokenizer has bos_token=
# None / add_bos_token=False so no BOS is ever prepended (OLMo-equivalent no-BOS).
# Cross-family: only compare base-normalised recovery / direction vs OLMo.
#
# ── PRECONDITION (TODOList hard requirement) ────────────────────────────────
#   Run the 32-item base sanity FIRST (SANITY=1), confirm the harness runs and
#   the letter protocol produces plausible numbers, THEN full runs. The full
#   base letter-MMLU must reproduce the P2.3 letter-MMLU aggregate (base .7297,
#   f12k2 .2495) item-for-item because the letter prompt + encode_pair are
#   byte-identical to scripts/eval_qwen3_probe2_downstream.py.
#
# ── env knobs ────────────────────────────────────────────────────────────────
#   WD           project root (default wzc1 canonical share_304376610 path)
#   PY           python (default /opt/conda/envs/torch-base/bin/python for .104)
#   BASE         pretrained Qwen3-8B path (default models/Qwen3-8b-local; the
#                SAME base P2.3 used -> letter numbers reproduce)
#   CKPT         prune-then-heal .pt; EMPTY => full-base mode
#                (f12k2@200k = outputs/qwen3_minarch_armB_f12k2_200k/final.pt)
#   KEEP_FRONT   pruned-shell kept front layers (f12k2 -> 12; default 12)
#   N_FRESH      pruned-shell fresh layers       (f12k2 -> 2;  default 2)
#   TAG          output-name (e.g. qwen3_base_full, qwen3_f12k2_step200k)
#   MODE         mmlu | closedbook | both        (default both)
#   CONTENT_DESC content prompt: full (default) | none
#   TASKS        closed-book tasks (default popqa,triviaqa,nq_open)
#   BS_MMLU      content-MMLU per-shard batch (default 16)
#   BS_QA        closed-book per-shard batch  (default 32)
#   MAX_NEW      closed-book max_new_tokens    (default 32)
#   NGPU         #GPUs to shard over           (default 8)
#   N_BOOT       paired-bootstrap resamples    (default 10000)
#   LIMIT        >0 caps examples per shard (post-strided); sanity only
#   SANITY       1 => single-GPU 32-item content-MMLU base smoke then EXIT
# ============================================================================
set -u
WD="${WD:-/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory}"
cd "$WD" || exit 1
PY="${PY:-/opt/conda/envs/torch-base/bin/python}"
BASE="${BASE:-models/Qwen3-8b-local}"
CKPT="${CKPT:-}"                       # empty => full-base mode
KEEP_FRONT="${KEEP_FRONT:-12}"
N_FRESH="${N_FRESH:-2}"
TAG="${TAG:-qwen3_base_full}"
MODE="${MODE:-both}"
CONTENT_DESC="${CONTENT_DESC:-full}"
TASKS="${TASKS:-popqa,triviaqa,nq_open}"
BS_MMLU="${BS_MMLU:-16}"
BS_QA="${BS_QA:-32}"
MAX_NEW="${MAX_NEW:-32}"
NGPU="${NGPU:-8}"
N_BOOT="${N_BOOT:-10000}"
LIMIT="${LIMIT:-0}"
SANITY="${SANITY:-0}"

MMLU=scripts/eval_qwen3_mmlu_content.py
QA=scripts/eval_qwen3_closedbook_qa.py
MMLU_ROOT=qwen3_mmlu_content_results
QA_ROOT=qwen3_closedbook_results

# proxy for the ONE-TIME prepare_data (datasets are shared with OLMo P0.3/P0.6
# and likely already cached; proxy just guards a cold cache).
export http_proxy=http://hy-proxy.woa.com:3128
export https_proxy=http://hy-proxy.woa.com:3128
export all_proxy=http://hy-proxy.woa.com:3128
export no_proxy="localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_DATASETS_CACHE=$WD/data/hf_datasets_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs "$MMLU_ROOT" "$QA_ROOT" "$HF_DATASETS_CACHE"

# ckpt / shell CLI (empty in base mode)
CKARG=""
SHELL_ARGS=""
if [ -n "$CKPT" ]; then
  if [ ! -f "$CKPT" ]; then
    echo "[FATAL] CKPT not found: $CKPT" >&2; exit 2
  fi
  CKARG="--ckpt $CKPT"
  SHELL_ARGS="--keep_front_layers $KEEP_FRONT --n_fresh_layers $N_FRESH"
fi
LIMIT_ARG=""
[ "$LIMIT" -gt 0 ] 2>/dev/null && LIMIT_ARG="--limit $LIMIT"

echo "[$(date '+%F %T')] P2.5 arm=$TAG base=$BASE ckpt=[$CKPT] shell=[$SHELL_ARGS] mode=$MODE"

# ── 32-item base sanity (single GPU, content-MMLU) ────────────────────────────
if [ "$SANITY" = "1" ]; then
  SN="${TAG}_sanity32"
  echo "[$(date '+%F %T')] SANITY: 32-item content-MMLU smoke on GPU0 -> $MMLU_ROOT/$SN"
  $PY $MMLU --prepare_data --content_desc "$CONTENT_DESC" \
      > "logs/qwen3_p25_${SN}_prepare.log" 2>&1
  CUDA_VISIBLE_DEVICES=0 $PY $MMLU \
    --base_model "$BASE" $CKARG $SHELL_ARGS \
    --content_desc "$CONTENT_DESC" \
    --num_shards 1 --shard_index 0 --batch_size "$BS_MMLU" \
    --limit 32 --output_name "$SN" --results_root "$MMLU_ROOT" \
    2>&1 | tee "logs/qwen3_p25_${SN}_shard0.log"
  $PY $MMLU --merge --output_name "$SN" --results_root "$MMLU_ROOT" --n_boot 1000 2>&1
  echo "[$(date '+%F %T')] SANITY summary:"
  cat "$MMLU_ROOT/$SN/summary.json" 2>/dev/null | head -c 900; echo
  echo "[$(date '+%F %T')] SANITY DONE (confirm letter_acc plausible, THEN run full)"
  exit 0
fi

run_mmlu() {   # content-MMLU: prepare once -> NGPU shards -> merge
  echo "[$(date '+%F %T')] ===== content-MMLU $TAG ====="
  $PY $MMLU --prepare_data --content_desc "$CONTENT_DESC" \
      > "logs/qwen3_p25_mmlu_${TAG}_prepare.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $MMLU \
      --base_model "$BASE" $CKARG $SHELL_ARGS \
      --content_desc "$CONTENT_DESC" \
      --num_shards $NGPU --shard_index $g --batch_size "$BS_MMLU" $LIMIT_ARG \
      --output_name "$TAG" --results_root "$MMLU_ROOT" \
      > "logs/qwen3_p25_mmlu_${TAG}_shard${g}.log" 2>&1 &
  done
  wait
  $PY $MMLU --merge --output_name "$TAG" --results_root "$MMLU_ROOT" --n_boot "$N_BOOT" 2>&1
  echo "[$(date '+%F %T')] content-MMLU $TAG summary:"
  cat "$MMLU_ROOT/$TAG/summary.json" 2>/dev/null | head -c 900; echo
}

run_closedbook() {  # closed-book QA: prepare once -> NGPU shards -> merge
  echo "[$(date '+%F %T')] ===== closed-book $TAG (tasks=$TASKS) ====="
  $PY $QA --prepare_data --tasks "$TASKS" \
      > "logs/qwen3_p25_cb_${TAG}_prepare.log" 2>&1
  for g in $(seq 0 $((NGPU-1))); do
    CUDA_VISIBLE_DEVICES=$g $PY $QA \
      --base_model "$BASE" $CKARG $SHELL_ARGS \
      --tasks "$TASKS" --max_new_tokens "$MAX_NEW" \
      --num_shards $NGPU --shard_index $g --batch_size "$BS_QA" --add_bos 0 $LIMIT_ARG \
      --output_name "$TAG" --results_root "$QA_ROOT" \
      > "logs/qwen3_p25_cb_${TAG}_shard${g}.log" 2>&1 &
  done
  wait
  $PY $QA --merge --output_name "$TAG" --results_root "$QA_ROOT" 2>&1
  echo "[$(date '+%F %T')] closed-book $TAG summary:"
  cat "$QA_ROOT/$TAG/summary.json" 2>/dev/null | head -c 900; echo
}

case "$MODE" in
  mmlu)       run_mmlu ;;
  closedbook) run_closedbook ;;
  both)       run_mmlu; run_closedbook ;;
  *) echo "[FATAL] unknown MODE=$MODE (mmlu|closedbook|both)" >&2; exit 2 ;;
esac
echo "[$(date '+%F %T')] ===== P2.5 eval DONE (TAG=$TAG, MODE=$MODE) ====="
