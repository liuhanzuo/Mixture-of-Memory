#!/usr/bin/env bash
# 8卡全并行: bm25 4-shard(GPU0-3) + readerattn 4-shard(GPU4-7) 同时跑。同ckpt/mnt20/topk4。
set -uo pipefail
R=/apdcephfs_zwfy6/share_303098609/pighzliu_code/Mixture-of-Memory
L=$1
cd "$R"; PY="$R/.venv/bin/python"
export PYTHONUNBUFFERED=1 PYTHONPATH="$R/third_party/babilong-pkg:$R:${PYTHONPATH:-}"
export HF_HOME="$R/.hf_cache" HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
CKPT="$R/outputs/mem_space_fifo_b25_c512_supervised_select/full_model_step002000.pt"
ACFG="$R/outputs/mem_space_fifo_b25_c512_supervised_select/adapter_config.json"
MODEL="$R/models/Meta-Llama-3-8B"
shard() { local gpu=$1 m=$2 si=$3 extra=$4
  CUDA_VISIBLE_DEVICES=$gpu $PY scripts/run_babilong_mem_space.py \
    --model_path "$MODEL" --checkpoint "$CKPT" --adapter_config "$ACFG" \
    --results_folder babilong_results --output_name "FAIR3sh/qa5_${L}_${m}" \
    --tasks qa5 --lengths $L --limit 50 --chunk_size 512 --batch_size 1 \
    --max_new_tokens 20 --dtype bfloat16 --attn_impl sdpa \
    --use_instruction --use_examples --use_post_prompt \
    --num_shards 4 --shard_index $si $extra \
    >logs/fair3sh_${m}_${L}_s${si}.log 2>&1; echo "DONE $m $L s$si"
}
BM="--swa_bm25_token --swa_bm25_topk 4 --swa_bm25_select_layer 16"
RA="--swa_readerattn_token --swa_readerattn_topk 4 --swa_readerattn_select_layer 16"
for si in 0 1 2 3; do shard $si bm25 $si "$BM" & done
for si in 0 1 2 3; do shard $((si+4)) readerattn $si "$RA" & done
wait
echo "FAIR3_8GPU_${L}_DONE"
