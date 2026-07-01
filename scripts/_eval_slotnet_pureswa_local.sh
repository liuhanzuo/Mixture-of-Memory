#!/usr/bin/env bash
# 纯SWA对照 + slot+SWA满100 (辨slot净贡献)
# slot+SWA: distill_pg19 nctx63 模型 + 滑窗W; 纯SWA: 同模型 --memory_disabled (bypass slot, vanilla Llama+滑窗)
# 净贡献 = slot+SWA - 纯SWA。已有 slot+SWA n=50: swa0=12/swa2=32/swa4=40/swa6=54
set -uo pipefail
ROOT=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$ROOT"
PY="$ROOT/.venv/bin/python"
export PYTHONUNBUFFERED=1
export PYTHONPATH="$ROOT/third_party/babilong-pkg:$ROOT:${PYTHONPATH:-}"
export HF_HOME="$ROOT/.hf_cache"
export HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$ROOT/outputs/distill_pg19_chunk512_nctx63/mem_space_adapter.pt"
ACFG="$ROOT/outputs/distill_pg19_chunk512_nctx63/adapter_config.json"
MODEL="$ROOT/models/Meta-Llama-3-8B"
mkdir -p logs babilong_results

run() {  # gpu mode W
  local gpu=$1 mode=$2 W=$3 extra="" name
  if [ "$mode" = "pureswa" ]; then extra="--memory_disabled"; name="pureSWA_nctx63/qa5_16k_swa${W}"; else name="slotSWA100_nctx63/qa5_16k_swa${W}"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "$name" \
    --tasks qa5 --lengths 16k --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    $extra --max_new_tokens 20 \
    >logs/slotnet_${mode}_swa${W}.log 2>&1
  echo "DONE $mode swa$W"
}

# 本机4空卡(0-3假设忙, 用空的): 自动挑空卡
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}'))
echo "空卡: ${GPUS[*]}"
i=0
# 优先纯SWA对照(直答slot净贡献), 4个W; 再slot+SWA满100补
JOBS=("pureswa 0" "pureswa 2" "pureswa 4" "pureswa 6" "slotswa 0" "slotswa 2" "slotswa 4" "slotswa 6")
for j in "${JOBS[@]}"; do
  read -r mode W <<< "$j"
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  run "$g" "$mode" "$W" &
  i=$((i+1))
  # 每张卡串行: 当排满空卡数, wait一批
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "SLOTNET_PURESWA_DONE"
