#!/usr/bin/env bash
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R" HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$R/outputs/distill_pg19_chunk512_nctx63/mem_space_adapter.pt"
ACFG="$R/outputs/distill_pg19_chunk512_nctx63/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
run(){ local gpu=$1 mode=$2 W=$3 extra="" name
  [ "$mode" = pureswa ]&&{ extra="--memory_disabled"; name="pureSWA_nctx63/qa5_32k_swa$W"; }||name="slotSWA100_nctx63/qa5_32k_swa$W"
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "$name" --tasks qa5 --lengths 32k --limit 100 --chunk_size 512 \
    --dtype bfloat16 --attn_impl sdpa --swa_eval_chunks $W $extra --max_new_tokens 20 >logs/slot32kfill_${mode}_W${W}.log 2>&1; echo "DONE $mode W$W"
}
GPUS=($(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits|awk '$2<2000{print $1}'))
echo "空卡:${GPUS[*]}"
# 缺: slotswa W4(全缺), W0 pure+slot。优先slotswa W4(净贡献关键), 再W0
JOBS=("slotswa 4" "pureswa 0" "slotswa 0" "slotswa 6")
i=0; for j in "${JOBS[@]}"; do read -r m w <<<"$j"; g=${GPUS[$((i%${#GPUS[@]}))]}; run $g $m $w & i=$((i+1)); [ $((i%${#GPUS[@]}))-eq 0 ]&&wait; done; wait
echo SLOT32K_FILL_DONE
