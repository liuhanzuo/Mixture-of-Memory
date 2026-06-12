#!/usr/bin/env bash
# EXP-2 l3_n_summary=128 offline BABILong eval scheduler (diskB .249).
# 2 ckpt (step500, step1000) x 7 lengths = 14 units, each unit runs
# tasks="qa1 qa2 qa5" for a single length. LPT-balanced over 8 GPUs.
# Each unit is its own python proc pinned to one GPU. setsid-detached.
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
TASKS="qa1 qa2 qa5"
CHUNK_SIZE=512
LIMIT=100
MAX_NEW_TOKENS=20
mkdir -p "$RESULTS" "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
# ckpt label -> file
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

# Build flat units: (ckpt,length) -> weight (3 tasks share the unit, but
# cost is dominated by length so weight = length_weight).
declare -a U_CK U_LEN U_W
ui=0
for ck in "${CKPTS[@]}"; do
  for L in "${LENGTHS[@]}"; do
    U_CK[$ui]="$ck"; U_LEN[$ui]="$L"; U_W[$ui]="$(length_weight "$L")"
    ui=$((ui+1))
  done
done
NU=$ui

# LPT: sort unit indices desc by weight
declare -a ORDER
for ((i=0;i<NU;i++)); do ORDER[$i]=$i; done
for ((a=0;a<NU;a++)); do
  best=$a
  for ((b=a+1;b<NU;b++)); do
    if (( U_W[ORDER[b]] > U_W[ORDER[best]] )); then best=$b; fi
  done
  t=${ORDER[$a]}; ORDER[$a]=${ORDER[$best]}; ORDER[$best]=$t
done

NG=8
declare -a GPU_LOAD GPU_UNITS
for ((g=0;g<NG;g++)); do GPU_LOAD[$g]=0; GPU_UNITS[$g]=""; done
for ((o=0;o<NU;o++)); do
  c=${ORDER[$o]}
  ming=0
  for ((g=1;g<NG;g++)); do
    if (( GPU_LOAD[g] < GPU_LOAD[ming] )); then ming=$g; fi
  done
  GPU_LOAD[$ming]=$(( GPU_LOAD[ming] + U_W[c] ))
  GPU_UNITS[$ming]="${GPU_UNITS[$ming]} $c"
done

echo "[$(date)] exp2_l3sum128 eval: $NU units over $NG GPUs"
for ((g=0;g<NG;g++)); do echo "  GPU $g load=${GPU_LOAD[$g]} units:${GPU_UNITS[$g]}"; done

run_gpu_units() {
  local G=$1; shift
  local units=("$@")
  for c in "${units[@]}"; do
    local ck="${U_CK[$c]}"; local L="${U_LEN[$c]}"
    local ckfile="${CKPT_FILE[$ck]}"
    echo "[$(date)] GPU $G -> ckpt $ck length $L"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$ckfile" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "$RESULTS" --output_name "exp2_l3sum128_${ck}_${L}" \
      --tasks $TASKS --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --batch_size 1 --max_new_tokens "$MAX_NEW_TOKENS" \
      --dtype bfloat16 --attn_impl sdpa \
      </dev/null >"$LOGDIR/${ck}_${L}.log" 2>&1
  done
}

for ((g=0;g<NG;g++)); do
  read -r -a uarr <<< "${GPU_UNITS[$g]}"
  if [ ${#uarr[@]} -gt 0 ]; then
    run_gpu_units "$g" "${uarr[@]}" &
  fi
done
wait
echo "[$(date)] ALL_EVAL_DONE -> $RESULTS"
touch "$LOGDIR/SCHED_DONE"
