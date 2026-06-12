#!/usr/bin/env bash
# EXP-2 l3_n_summary=128 BABILong eval — FINE scheduler.
# 2 ckpt x 3 tasks x 7 lengths = 42 cells, each cell = one python proc on
# one GPU. Skips cells already complete (101 csv rows). LPT-balanced over 8
# GPUs so heavy 32k qa1/qa2/qa5 spread across GPUs (vs prior bundling that
# serialized all 3 tasks of a length on one GPU).
set -uo pipefail
PROJECT_ROOT="/apdcephfs_zwfy6/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export http_proxy="http://hy-proxy.woa.com:3128"
export https_proxy="http://hy-proxy.woa.com:3128"
export all_proxy="http://hy-proxy.woa.com:3128"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
export HF_HOME="$PROJECT_ROOT/.hf_home"
PYBIN="$PROJECT_ROOT/.venv/bin/python"

MODEL="models/Meta-Llama-3-8B"
CKPT_DIR="outputs/exp2_l3summary128"
ADAPTER_CONFIG="$CKPT_DIR/adapter_config.json"
RESULTS="babilong_results/exp2_l3sum128"
LOGDIR="logs/eval_exp2_l3sum128"
CHUNK_SIZE=512
LIMIT=100
MAX_NEW_TOKENS=20
SUFFIX="_instruction_yes_examples_yes_post_prompt_yes_chat_template_no_system_prompt_no.csv"
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
TASKS=(qa1 qa2 qa5)
declare -A CKPT_FILE
CKPT_FILE[step500]="$CKPT_DIR/mem_space_adapter_step000500.pt"
CKPT_FILE[step1000]="$CKPT_DIR/mem_space_adapter.pt"
CKPTS=(step500 step1000)

length_weight() {
  case "$1" in
    0k) echo 1;; 1k) echo 2;; 2k) echo 4;; 4k) echo 8;;
    8k) echo 16;; 16k) echo 32;; 32k) echo 64;; *) echo 8;;
  esac
}

# Build flat cells (ckpt,task,length), skip complete ones.
declare -a C_CK C_TASK C_LEN C_W
ci=0
for ck in "${CKPTS[@]}"; do
  for task in "${TASKS[@]}"; do
    for L in "${LENGTHS[@]}"; do
      out_dir="$RESULTS/exp2_l3sum128_${ck}_${task}_${L}"
      csv="$out_dir/${task}_${L}${SUFFIX}"
      if [ -f "$csv" ] && [ "$(wc -l < "$csv")" -eq 101 ]; then
        echo "[skip-complete] $ck $task $L"
        continue
      fi
      C_CK[$ci]="$ck"; C_TASK[$ci]="$task"; C_LEN[$ci]="$L"; C_W[$ci]="$(length_weight "$L")"
      ci=$((ci+1))
    done
  done
done
NC=$ci
echo "[$(date)] fine sched: $NC cells to (re)run"

# LPT sort desc by weight
declare -a ORDER
for ((i=0;i<NC;i++)); do ORDER[$i]=$i; done
for ((a=0;a<NC;a++)); do
  best=$a
  for ((b=a+1;b<NC;b++)); do
    if (( C_W[ORDER[b]] > C_W[ORDER[best]] )); then best=$b; fi
  done
  t=${ORDER[$a]}; ORDER[$a]=${ORDER[$best]}; ORDER[$best]=$t
done

NG=8
declare -a GPU_LOAD GPU_CELLS
for ((g=0;g<NG;g++)); do GPU_LOAD[$g]=0; GPU_CELLS[$g]=""; done
for ((o=0;o<NC;o++)); do
  c=${ORDER[$o]}
  ming=0
  for ((g=1;g<NG;g++)); do
    if (( GPU_LOAD[g] < GPU_LOAD[ming] )); then ming=$g; fi
  done
  GPU_LOAD[$ming]=$(( GPU_LOAD[ming] + C_W[c] ))
  GPU_CELLS[$ming]="${GPU_CELLS[$ming]} $c"
done

echo "[$(date)] $NC cells over $NG GPUs"
for ((g=0;g<NG;g++)); do echo "  GPU $g load=${GPU_LOAD[$g]} cells:${GPU_CELLS[$g]}"; done

run_gpu_cells() {
  local G=$1; shift
  local cells=("$@")
  for c in "${cells[@]}"; do
    local ck="${C_CK[$c]}"; local task="${C_TASK[$c]}"; local L="${C_LEN[$c]}"
    local ckfile="${CKPT_FILE[$ck]}"
    echo "[$(date)] GPU $G -> $ck $task $L (w=${C_W[$c]})"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$ckfile" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$RESULTS" --output_name "exp2_l3sum128_${ck}_${task}_${L}" \
      --tasks "$task" --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype bfloat16 --attn_impl sdpa \
      </dev/null >"$LOGDIR/${ck}_${task}_${L}.log" 2>&1
  done
}

for ((g=0;g<NG;g++)); do
  read -r -a cells_arr <<< "${GPU_CELLS[$g]}"
  if [ ${#cells_arr[@]} -gt 0 ]; then
    run_gpu_cells "$g" "${cells_arr[@]}" &
  fi
done
wait
echo "[$(date)] ALL_EVAL_DONE -> $RESULTS"
touch "$LOGDIR/SCHED_DONE"
