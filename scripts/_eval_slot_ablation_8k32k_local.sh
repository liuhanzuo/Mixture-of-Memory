#!/usr/bin/env bash
# slot ablation 稳健性: 8k+32k, pureSWA(bypass slot) vs slotSWA(slot on), swa{0,4,6}
# 16k已确认slot净负(n100): pure 14/55/64 vs slot 12/42/50。验证8k/32k是否同样净负。
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$R/outputs/distill_pg19_chunk512_nctx63/mem_space_adapter.pt"
ACFG="$R/outputs/distill_pg19_chunk512_nctx63/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
mkdir -p logs babilong_results

run() { # gpu mode W L
  local gpu=$1 mode=$2 W=$3 L=$4 extra="" name
  if [ "$mode" = "pureswa" ]; then extra="--memory_disabled"; name="pureSWA_nctx63/qa5_${L}_swa${W}"; else name="slotSWA100_nctx63/qa5_${L}_swa${W}"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "$name" \
    --tasks qa5 --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W \
    $extra --max_new_tokens 20 \
    >logs/slotabl_${mode}_swa${W}_${L}.log 2>&1
  echo "DONE $mode swa$W $L"
}

GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}'))
echo "空卡: ${GPUS[*]}"
# 8k先(快), 再32k; swa4/swa6差距最大优先, swa0基线
JOBS=("pureswa 6 8k" "slotswa 6 8k" "pureswa 4 8k" "slotswa 4 8k" "pureswa 0 8k" "slotswa 0 8k" \
      "pureswa 6 32k" "slotswa 6 32k" "pureswa 4 32k" "slotswa 4 32k" "pureswa 0 32k" "slotswa 0 32k")
i=0
for j in "${JOBS[@]}"; do
  read -r mode W L <<< "$j"
  g=${GPUS[$((i % ${#GPUS[@]}))]}
  run "$g" "$mode" "$W" "$L" &
  i=$((i+1))
  if [ $((i % ${#GPUS[@]})) -eq 0 ]; then wait; fi
done
wait
echo "SLOT_ABLATION_8k32k_DONE"
