#!/usr/bin/env bash
# ROUTE-A arm4 (st-gumbel-topk) offline BABILong eval scheduler.
# 2 ckpt (step500, step1000) x 7 lengths = 14 units, balanced over 8 GPUs by
# length cost (LPT). Each unit runs all 3 tasks (qa1 qa2 qa5) for one length.
# Output layout matches scripts/score_nested_babilong.py:
#   babilong_results/<run>/<run>_<L>/{task}_{L}_*.csv
set -uo pipefail
PROJECT_ROOT="/apdcephfs_wzc1/share_304376610/pighzliu_code/Mixture-of-Memory"
cd "$PROJECT_ROOT"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$PROJECT_ROOT:$PROJECT_ROOT/third_party/babilong-pkg:${PYTHONPATH:-}"
export HF_HOME="$PROJECT_ROOT/.hf_cache"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
# proxy fallback (in case a length needs to touch HF)
export http_proxy="${http_proxy:-http://hy-proxy.woa.com:3128}"
export https_proxy="${https_proxy:-http://hy-proxy.woa.com:3128}"
export all_proxy="${all_proxy:-http://hy-proxy.woa.com:3128}"
export no_proxy="mirrors.cloud.tencent.com,tlinux-mirror.tencent-cloud.com,localhost,127.0.0.1,.oa.com,.woa.com,.local"
PYBIN="$PROJECT_ROOT/.venv/bin/python"

MODEL="models/Meta-Llama-3-8B"
CKPT_DIR="outputs/routeA_arm4_stgumbel"
ADAPTER_CONFIG="$CKPT_DIR/adapter_config.json"
TASKS="qa1 qa2 qa5"
CHUNK_SIZE=512
LIMIT=100
MAX_NEW_TOKENS=20
LOGDIR="logs/routeA_arm4_eval"
mkdir -p "$LOGDIR"

LENGTHS=(0k 1k 2k 4k 8k 16k 32k)
GPUS=(0 1 2 3 4 5 6 7)
NG=${#GPUS[@]}

# ckpt list: name|ckpt_file
declare -a CK_NAME CK_FILE
CK_NAME[0]="routeA_arm4_step500";  CK_FILE[0]="$CKPT_DIR/mem_space_adapter_step000500.pt"
CK_NAME[1]="routeA_arm4_step1000"; CK_FILE[1]="$CKPT_DIR/mem_space_adapter.pt"

length_weight() {
  case "$1" in
    0k) echo 1;; 1k) echo 2;; 2k) echo 4;; 4k) echo 8;;
    8k) echo 16;; 16k) echo 32;; 32k) echo 64;; *) echo 8;;
  esac
}

# Build flat unit list: (ckpt_idx, length), weight.
declare -a U_CK U_LEN U_W
ui=0
for ck in 0 1; do
  for L in "${LENGTHS[@]}"; do
    U_CK[$ui]=$ck; U_LEN[$ui]="$L"; U_W[$ui]="$(length_weight "$L")"
    ui=$((ui+1))
  done
done
NU=$ui

# LPT: sort unit indices by weight desc.
declare -a ORDER
for ((i=0;i<NU;i++)); do ORDER[$i]=$i; done
for ((a=0;a<NU;a++)); do
  best=$a
  for ((b=a+1;b<NU;b++)); do
    if (( U_W[ORDER[b]] > U_W[ORDER[best]] )); then best=$b; fi
  done
  tmp=${ORDER[$a]}; ORDER[$a]=${ORDER[$best]}; ORDER[$best]=$tmp
done

declare -a GPU_LOAD GPU_UNITS
for ((g=0;g<NG;g++)); do GPU_LOAD[$g]=0; GPU_UNITS[$g]=""; done
for ((o=0;o<NU;o++)); do
  u=${ORDER[$o]}
  ming=0
  for ((g=1;g<NG;g++)); do
    if (( GPU_LOAD[g] < GPU_LOAD[ming] )); then ming=$g; fi
  done
  GPU_LOAD[$ming]=$(( GPU_LOAD[ming] + U_W[u] ))
  GPU_UNITS[$ming]="${GPU_UNITS[$ming]} $u"
done

echo "[$(date)] routeA_arm4 eval: $NU units over $NG GPUs"
for ((g=0;g<NG;g++)); do
  echo "  GPU ${GPUS[$g]} load=${GPU_LOAD[$g]} units:${GPU_UNITS[$g]}"
done

run_gpu_units() {
  local G=$1; shift
  local units=("$@")
  for u in "${units[@]}"; do
    local ck=${U_CK[$u]}
    local L="${U_LEN[$u]}"
    local run="${CK_NAME[$ck]}"
    local ckfile="${CK_FILE[$ck]}"
    echo "[$(date)] GPU $G -> $run length $L"
    CUDA_VISIBLE_DEVICES=$G $PYBIN scripts/run_babilong_mem_space.py \
      --model_path "$MODEL" --checkpoint "$ckfile" --adapter_config "$ADAPTER_CONFIG" \
      --results_folder "babilong_results/$run" --output_name "${run}_${L}" \
      --tasks $TASKS --lengths "$L" --limit "$LIMIT" --chunk_size "$CHUNK_SIZE" \
      --max_new_tokens "$MAX_NEW_TOKENS" --dtype bfloat16 --attn_impl sdpa \
      </dev/null >"$LOGDIR/${run}_${L}.log" 2>&1
  done
}

for ((g=0;g<NG;g++)); do
  read -r -a units_arr <<< "${GPU_UNITS[$g]}"
  if [ ${#units_arr[@]} -gt 0 ]; then
    run_gpu_units "${GPUS[$g]}" "${units_arr[@]}" &
  fi
done
wait
echo "[$(date)] ALL routeA_arm4 eval units done"
touch "$LOGDIR/SCHED_DONE"
