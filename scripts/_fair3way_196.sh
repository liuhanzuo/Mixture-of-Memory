#!/usr/bin/env bash
# 控制变量三方对照: 同ckpt A step2000, max_new_tokens=20, topk=4, 同批样本(limit50)
# 只切选择方法 {bm25, readerattn, oracle}。qa5 16k。
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$R/outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
ACFG="$R/outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
mkdir -p logs babilong_results
run() { # gpu method extraflag
  local gpu=$1 m=$2 extra=$3
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "FAIR3/qa5_16k_${m}" \
    --tasks qa5 --lengths 16k --limit 50 --chunk_size 512 \
    --batch_size 1 --max_new_tokens 20 --dtype bfloat16 --attn_impl sdpa \
    --use_instruction --use_examples --use_post_prompt \
    $extra >logs/fair3_${m}.log 2>&1
  echo "DONE $m"
}
# 3方各一张卡, 并行
run 0 bm25       "--swa_bm25_token --swa_bm25_topk 4 --swa_bm25_select_layer 16" &
run 1 readerattn "--swa_readerattn_token --swa_readerattn_topk 4 --swa_readerattn_select_layer 16" &
run 2 oracle     "--swa_oracle_token" &
wait
echo "FAIR3WAY_DONE"
