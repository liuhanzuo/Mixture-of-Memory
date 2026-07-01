#!/usr/bin/env bash
# 补slot ablation W2中间点(8k/32k), 填driver wait间隙的空卡。不撞driver(它跑W0/4/6)
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$R/outputs/distill_pg19_chunk512_nctx63/mem_space_adapter.pt"
ACFG="$R/outputs/distill_pg19_chunk512_nctx63/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
run() { local gpu=$1 mode=$2 L=$3 extra="" name
  if [ "$mode" = "pureswa" ]; then extra="--memory_disabled"; name="pureSWA_nctx63/qa5_${L}_swa2"; else name="slotSWA100_nctx63/qa5_${L}_swa2"; fi
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "$name" \
    --tasks qa5 --lengths $L --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks 2 $extra --max_new_tokens 20 \
    >logs/slotablW2_${mode}_${L}.log 2>&1; echo "DONE $mode W2 $L"
}
# 只用driver没占的空卡(动态挑)
sleep 3
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk '$2<2000{print $1}'))
echo "空卡:${GPUS[*]}"
[ ${#GPUS[@]} -eq 0 ] && { echo "无空卡, 退出"; exit 0; }
JOBS=("pureswa 8k" "slotswa 8k" "pureswa 32k" "slotswa 32k")
i=0
for j in "${JOBS[@]}"; do read -r mode L <<< "$j"; g=${GPUS[$((i % ${#GPUS[@]}))]}; run "$g" "$mode" "$L" & i=$((i+1)); [ $((i % ${#GPUS[@]})) -eq 0 ]&&wait; done
wait; echo "SLOTABL_W2_DONE"
