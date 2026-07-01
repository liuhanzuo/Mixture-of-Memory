#!/usr/bin/env bash
# Eval a beacon-pyramid ckpt on BABILong qa5 (official compare_answers, n=100).
# The beacon path activates automatically from the ckpt adapter_config
# (use_beacon_pyramid=true) — eval streams chunks through _forward_fifo_beacon,
# so NO extra eval flag is needed. 8-GPU LPT-balanced over the length cells.
# ENV: CKPT_DIR (required), LENGTHS, LIMIT, TASKS.
# OFFLINE isolated node — NO proxy (babilong cache is local on wzc1).
set -uo pipefail
R="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$R"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$R:$R/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_CACHE="$R/.hf_cache/datasets"
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
PYBIN="$R/.venv/bin/python"
MODEL="models/Meta-Llama-3-8B"
CKPT_DIR="${CKPT_DIR:?set CKPT_DIR}"
ADAPTER_CONFIG="${ADAPTER_CONFIG:-${CKPT_DIR}/adapter_config.json}"
# pick the final full_model.pt if present, else the highest-step ckpt
if [ -z "${CKPT:-}" ]; then
  if [ -f "${CKPT_DIR}/full_model.pt" ]; then
    CKPT="${CKPT_DIR}/full_model.pt"
  else
    CKPT="$(ls -1 ${CKPT_DIR}/full_model_step*.pt 2>/dev/null | sort | tail -1)"
  fi
fi
RESULTS="${RESULTS:-babilong_results/$(basename "$CKPT_DIR")_qa5}"
OUTPREFIX="${OUTPREFIX:-$(basename "$CKPT_DIR")}"
TASKS="${TASKS:-qa5}"
CHUNK_SIZE="${CHUNK_SIZE:-512}"
LIMIT="${LIMIT:-100}"
DTYPE="${DTYPE:-bfloat16}"; ATTN_IMPL="${ATTN_IMPL:-sdpa}"; MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-20}"
LOGDIR="${LOGDIR:-logs/eval_$(basename "$CKPT_DIR")_qa5}"
mkdir -p "$RESULTS" "$LOGDIR"
read -r -a LENGTHS <<< "${LENGTHS:-2k 4k 8k 16k}"
read -r -a GPUS <<< "${GPUS:-0 1 2 3 4 5 6 7}"
NG=${#GPUS[@]}
echo "[$(date)] beacon eval ckpt=$CKPT cfg=$ADAPTER_CONFIG tasks=$TASKS lengths=${LENGTHS[*]} n=$LIMIT"
# Flat cell list (task x length), round-robin onto GPUs (few cells, simple).
declare -a CELLS
for task in $TASKS; do for L in "${LENGTHS[@]}"; do CELLS+=("$task:$L"); done; done
i=0
for cell in "${CELLS[@]}"; do
  G=${GPUS[$(( i % NG ))]}
  task="${cell%%:*}"; L="${cell##*:}"
  echo "[$(date)] GPU $G -> $task $L"
  CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ADAPTER_CONFIG" \
    --results_folder "$RESULTS" --output_name "${OUTPREFIX}_${task}_${L}" \
    --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
    --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
    --dtype "$DTYPE" --attn_impl "$ATTN_IMPL" \
    </dev/null >"$LOGDIR/${task}_${L}.log" 2>&1 &
  i=$((i+1))
done
wait
echo "[$(date)] all cells done -> $RESULTS"
